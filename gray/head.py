#!/usr/bin/env python3
"""The head-path model on moments: text and what the camera saw -> where the head goes in the next 5 s.

    python -m gray.head --save-dir <runs>/head-v3 --amp

A sample is a moment of gray/moments.py. The model is given

    past      the head poses and the image feature (one vector per frame) of up to 30 frames, 2 s, up to and including now
    text      the narration sentence that begins at now
    patches   optionally, the 7 x 7 grid of image features of the frame at now (gray/patches.py)

    depth     optionally, the estimated depth of the frame at now (gray/depth.py): one value per cell of the image grid, and
              one extra token with the distance to the nearest thing at body height in 16 directions, read off the depth

With ``--ground`` the model also has to say, during training only, how far the way is free
around the camera at now (gray/ground.py: 16 directions, the walked distance and the distance to
the first blocked cell). One extra token reads the past frames, the text and the image grid, but
not the future path (which would give the answer away), and the future tokens can read it. The
room maps are thus a training signal and never an input.

and generates the head poses of the 75 frames after now. All poses are in the frame anchored at
now (gray/frames.py) and standardized. Nothing else is given: no body motion, no gaze, no map.

It is a flow model that predicts the clean path. A noise level between 0 (pure noise) and 1
(clean) is drawn per sample, the future poses are mixed with noise accordingly, and the
transformer predicts the clean poses; the loss is the squared error. Sampling starts from noise
and takes Euler steps (``sample``).

In training the past is cut to a random length (1 to 30 frames), now is moved by -5 to +15 frames
around the start of the sentence, and the text is left out for 10 % of the samples.

``--explore K`` trains "exploratively" (Gladstone, Ji, Du: Explorative Modeling, arXiv 2607.27372, their
Forward XM added to a flow model): for every sample K different noises are tried at the drawn noise level, without
gradients, and the model is trained only on the one whose prediction is closest to the recorded path. The model then
learns to let the noise decide which of several valid paths it commits to, instead of blending them.
``--one-step`` goes to the paper's end-to-end form: the noise level is always 0, so the model maps noise to a whole
path in one pass, and sampling takes one step (score such a run with --steps 1).

Saved in the run folder: config.json, log.jsonl (one line per evaluation), last.pt.
gray/score.py scores a trained run against the room maps and the text.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from gray import frames
from gray.features import CACHE, RATE, time_embedding
from gray.joint import level_embedding, to_device
from gray.moments import FUTURE, HISTORY, Moments, collate

NOW = HISTORY - 1  # the index of now in a sample's poses


class HeadPath(nn.Module):
    """A transformer over the past frames, the future frames, the text tokens and (optionally) the image grid."""

    def __init__(self, width: int = 512, layers: int = 8, heads: int = 8, text_dim: int = 768, image_dim: int = 384, patches: bool = False, cells: int = 49,
                 free: bool = False, depth: bool = False) -> None:
        super().__init__()
        self.width, self.heads = width, heads
        if depth:
            self.depth_in = nn.Linear(1, width)
            self.rays_in = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        if free:
            self.free_token = nn.Parameter(torch.randn(width) * 0.02)
            self.free_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32))
        self.pose_in = nn.Linear(9, width)
        self.image_in = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, width))
        self.time_in = nn.Linear(width, width)
        self.level_in = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.kind = nn.Parameter(torch.randn(4, width) * 0.02)  # past, future, text, image grid
        self.text_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width))
        self.no_text = nn.Parameter(torch.randn(width) * 0.02)
        if patches:
            self.patch_in = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, width))
            self.patch_where = nn.Parameter(torch.randn(cells, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 9))

    def forward(self, past: torch.Tensor, image: torch.Tensor, seen: torch.Tensor, x: torch.Tensor, level: torch.Tensor,
                text: torch.Tensor, text_valid: torch.Tensor, patches: torch.Tensor | None = None, with_free: bool = False,
                depth: torch.Tensor | None = None, rays: torch.Tensor | None = None):
        """past (B, 30, 9), image (B, 30, 384), seen (B, 30) which past frames are given, x (B, 75, 9) noised future,
        level (B,), text (B, S, 768), text_valid (B, S), patches (B, 49, 384) -> the clean future (B, 75, 9);
        with ``with_free`` also the free distances (B, 32) in units of 5 m."""
        size, device = past.shape[0], past.device
        seconds = (torch.arange(HISTORY + FUTURE, device=device) - NOW) / RATE
        clock = self.time_in(time_embedding(seconds[None], self.width))  # (1, 105, width)
        before = self.pose_in(past) + self.image_in(image) + clock[:, :HISTORY] + self.kind[0]
        after = self.pose_in(x) + clock[:, HISTORY:] + self.kind[1] + self.level_in(level_embedding(level))[:, None]
        has_text = text_valid.any(dim=1)
        words = torch.where(has_text[:, None, None], self.text_in(text) + self.kind[2], self.no_text)
        valid = torch.where(has_text[:, None], text_valid, torch.arange(text.shape[1], device=device)[None] == 0)  # one placeholder token without text
        sequence = [before, after.to(before.dtype), words.to(before.dtype)]
        padding = [~seen, torch.zeros(size, FUTURE, dtype=torch.bool, device=device), ~valid]
        if hasattr(self, "patch_in"):
            if patches is None:
                raise ValueError("this model was built with patches=True and needs the image grid of the frame at now")
            cells = self.patch_in(patches) + self.patch_where + self.kind[3]
            if hasattr(self, "depth_in"):
                cells = cells + self.depth_in(depth[..., None] / 5.0)  # the depth of each cell, in units of 5 m
            sequence.append(cells.to(before.dtype))
            padding.append(torch.zeros(sequence[-1].shape[:2], dtype=torch.bool, device=device))
        if hasattr(self, "rays_in"):
            sequence.append((self.rays_in(rays / 5.0) + self.kind[3])[:, None].to(before.dtype))
            padding.append(torch.zeros(size, 1, dtype=torch.bool, device=device))
        mask = None
        if hasattr(self, "free_token"):
            sequence.append(self.free_token.to(before.dtype).expand(size, 1, -1))
            padding.append(torch.zeros(size, 1, dtype=torch.bool, device=device))
            length = sum(part.shape[1] for part in sequence)
            mask = torch.zeros(length, length, dtype=torch.bool, device=device)
            mask[-1, HISTORY : HISTORY + FUTURE] = True  # the free-space token does not look at the future path
        hidden = self.blocks(torch.cat(sequence, dim=1), mask=mask, src_key_padding_mask=torch.cat(padding, dim=1))
        future = self.out(hidden[:, HISTORY : HISTORY + FUTURE])
        return (future, self.free_out(hidden[:, -1])) if with_free else future


def anchored(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The poses (B, 105, 9) of a batch in the frame anchored at now, in metres."""
    return frames.anchor(batch["pose"], torch.full((batch["pose"].shape[0],), NOW, device=batch["pose"].device))


def inputs(batch: dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor, history: torch.Tensor | int = HISTORY):
    """What the model is given, and the clean future: past, image, seen, text, text_valid, patches, future."""
    x = (anchored(batch) - mean) / std
    size, device = x.shape[0], x.device
    history = torch.full((size,), history, device=device) if isinstance(history, int) else history
    seen = torch.arange(HISTORY, device=device)[None] >= HISTORY - history[:, None]
    return x[:, :HISTORY], batch["image"], seen, batch["text"], batch["text_valid"], batch.get("patches"), x[:, HISTORY:]


def geometry(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The depth inputs of a batch, if it has them."""
    return {"depth": batch["depth"], "rays": batch["rays"]} if "depth" in batch else {}


def flow_loss(model: HeadPath, batch: dict, mean: torch.Tensor, std: torch.Tensor, text_drop: float = 0.0, random_history: bool = True,
              generator: torch.Generator | None = None, amp: bool = False, free_weight: float = 0.0, explore: int = 1, one_step: bool = False):
    """The loss of the path; with ``free_weight`` above 0 a pair: the total loss, and the error of the free distances in metres (nan if none are known)."""
    size, device = batch["pose"].shape[0], batch["pose"].device
    draw = lambda *shape: torch.rand(*shape, generator=generator, device=device)  # noqa: E731
    history = (draw(size) * HISTORY).long() + 1 if random_history else torch.full((size,), HISTORY, device=device)
    if "history_max" in batch:  # samples that begin less than 2 s into what is known of them (gray/ee4d.py)
        history = torch.minimum(history, batch["history_max"])
    past, image, seen, text, text_valid, patches, future = inputs(batch, mean, std, history)
    text_valid = text_valid & (draw(size) >= text_drop)[:, None]
    level = torch.zeros(size, device=device) if one_step else draw(size)
    noise = torch.randn(future.shape, generator=generator, device=device)
    if explore > 1:  # keep, per sample, the noise from which the model comes closest to the recorded path
        with torch.no_grad():
            best = torch.full((size,), float("inf"), device=device)
            chosen = noise
            for _try in range(explore):
                candidate = torch.randn(future.shape, generator=generator, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    guess = model(past, image, seen, (1 - level[:, None, None]) * candidate + level[:, None, None] * future, level, text, text_valid, patches, **geometry(batch))
                error = ((guess.float() - future) ** 2).mean((1, 2))
                better = error < best
                best, chosen = torch.where(better, error, best), torch.where(better[:, None, None], candidate, chosen)
            noise = chosen
    mixed = (1 - level[:, None, None]) * noise + level[:, None, None] * future
    if free_weight <= 0:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            clean = model(past, image, seen, mixed, level, text, text_valid, patches, **geometry(batch))
        return ((clean.float() - future) ** 2).mean()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        clean, free = model(past, image, seen, mixed, level, text, text_valid, patches, with_free=True, **geometry(batch))
    path = ((clean.float() - future) ** 2).mean()
    known = torch.isfinite(batch["free"])
    if not known.any():
        return path, torch.full((), float("nan"), device=device)
    target = batch["free"][known] / 5.0
    extra = nn.functional.smooth_l1_loss(free.float()[known], target, beta=0.1)
    return path + free_weight * extra, (free.float()[known] - target).abs().mean().detach() * 5.0


@torch.no_grad()
def sample(model: HeadPath, batch: dict, mean: torch.Tensor, std: torch.Tensor, steps: int = 20, generator: torch.Generator | None = None,
           amp: bool = False, text: tuple[torch.Tensor, torch.Tensor] | None = None, history: int = HISTORY, guidance: float = 1.0,
           temperature: float = 1.0, poses: bool = False) -> torch.Tensor:
    """One generated future per sample: positions (B, 75, 3) in the track's ground frame (x, height above the floor at now, z).

    ``poses``: return instead the whole poses (B, 75, 9) in the frame anchored at now, in metres, with the rotation made a proper one.

    ``text``: other text features and their mask to use in place of the batch's own.
    ``temperature``: the starting noise is multiplied by this; below 1 the draws stay nearer the model's average path.
    ``guidance``: above 1 the text counts for more: each step uses  without text + guidance * (with text - without text).
    """
    past, image, seen, words, valid, patches, future = inputs(batch, mean, std, history)
    if text is not None:
        words, valid = text
    x = torch.randn(future.shape, generator=generator, device=future.device) * temperature
    for step in range(steps):
        level = torch.full((x.shape[0],), step / steps, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            clean = model(past, image, seen, x, level, words, valid, patches, **geometry(batch)).float()
            if guidance != 1.0:
                plain = model(past, image, seen, x, level, words, torch.zeros_like(valid), patches, **geometry(batch)).float()
                clean = plain + guidance * (clean - plain)
        x = x + (clean - x) / (1 - step / steps) / steps  # the last step lands on the prediction
    if poses:
        raw = x * std + mean
        rotation = frames.matrix(raw[..., 3:])
        return torch.cat([raw[..., :3], rotation[..., 0, :], rotation[..., 1, :]], dim=-1)
    position = (x * std + mean)[..., :3]
    here = batch["pose"][:, NOW]
    yaw = frames.heading(here[:, 3:])
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    return torch.stack([c * position[..., 0] - s * position[..., 2] + here[:, None, 0], position[..., 1],
                        s * position[..., 0] + c * position[..., 2] + here[:, None, 2]], dim=-1)


@torch.no_grad()
def free_guess(model: HeadPath, batch: dict, mean: torch.Tensor, std: torch.Tensor, amp: bool = False) -> torch.Tensor:
    """What a model trained with --ground says about the free distances around the camera at now: (B, 32) metres."""
    past, image, seen, text, text_valid, patches, future = inputs(batch, mean, std)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        _clean, guess = model(past, image, seen, torch.zeros_like(future), torch.zeros(len(past), device=past.device), text, text_valid, patches,
                              with_free=True, **geometry(batch))
    return guess.float().clamp(0, 1) * 5.0


@torch.no_grad()
def evaluate(model: HeadPath, batches: list, mean: torch.Tensor, std: torch.Tensor, device: str, steps: int, amp: bool, one_step: bool = False) -> dict[str, float]:
    """On held-out moments: the training loss, and how far one generated path and standing still are from the recorded path (cm)."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(0)
    loss, drawn, stay, going_drawn, going_stay, count, going_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
    free_sums = {"walkable": 0.0, "walkable count": 0, "blocked": 0.0, "blocked count": 0}
    for batch in batches:
        batch = to_device(batch, device)
        size = batch["pose"].shape[0]
        loss += flow_loss(model, batch, mean, std, random_history=False, generator=generator, amp=amp).item() * size
        if hasattr(model, "free_token") and "free" in batch:
            past, image, seen, text, text_valid, patches, future = inputs(batch, mean, std)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                _clean, guess = model(past, image, seen, torch.randn(future.shape, generator=generator, device=device), torch.zeros(size, device=device),
                                      text, text_valid, patches, with_free=True, **geometry(batch))
            guess = guess.float().clamp(0, 1) * 5.0
            for name, columns in (("walkable", [15, 0, 1]), ("blocked", [31, 16, 17])):  # the directions inside the camera's view
                known = torch.isfinite(batch["free"][:, columns])
                free_sums[name] += (guess[:, columns] - batch["free"][:, columns]).abs()[known].sum().item()
                free_sums[name + " count"] += int(known.sum())
        truth = batch["pose"][:, HISTORY:, :3]
        flat = lambda delta: torch.hypot(delta[..., 0], delta[..., 2]).mean(-1) * 100  # noqa: E731
        error = flat(sample(model, batch, mean, std, 1 if one_step else steps, generator, amp) - truth)
        still = flat(batch["pose"][:, NOW : NOW + 1, :3] - truth)
        away = batch["pose"][:, NOW : NOW + 1, :3] - truth
        going = torch.hypot(away[..., 0], away[..., 2]).amax(-1) >= 1.0  # gets 1 m or further, the rule of gray/moments.py and gray/score.py
        drawn, stay, count = drawn + error.sum().item(), stay + still.sum().item(), count + size
        going_drawn, going_stay, going_count = going_drawn + error[going].sum().item(), going_stay + still[going].sum().item(), going_count + int(going.sum())
    model.train()
    result = {"loss": loss / count, "one_draw_cm": drawn / count, "stay_cm": stay / count,
              "going_one_draw_cm": going_drawn / max(going_count, 1), "going_stay_cm": going_stay / max(going_count, 1)}
    if free_sums["walkable count"]:
        result |= {"walkable_in_view_m": free_sums["walkable"] / free_sums["walkable count"], "blocked_in_view_m": free_sums["blocked"] / free_sums["blocked count"]}
    return result


def load(folder: Path, device: str) -> tuple[HeadPath, torch.Tensor, torch.Tensor, dict]:
    """A trained model from its run folder, with the mean and standard deviation of the poses."""
    saved = torch.load(folder / "last.pt", map_location=device)
    config = saved["config"]
    model = HeadPath(config["width"], config["layers"], config["heads"], patches=bool(config.get("patches")), free=bool(config.get("ground")),
                     depth=bool(config.get("depth")), image_dim=int(config.get("image_dim", 384))).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved["mean"].to(device), saved["std"].to(device), config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--moments", type=Path, default=Path("/ptmp/enazaren/gray/moments/moments.npz"))
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--images", type=Path, default=Path("/ptmp/enazaren/gray/pinhole"))
    parser.add_argument("--patches", type=Path, default=None, help="folder of gray/patches.py: the model also sees the image grid of the frame at now")
    parser.add_argument("--depth", type=Path, default=None, help="folder of gray/depth.py: the model also gets the estimated depth of the frame at now (needs --patches)")
    parser.add_argument("--ee4d", type=Path, default=None, help="folder of gray/ee4d.py: train and score on UniEgoMotion's data in place of Nymeria (no text, no image grid, no maps)")
    parser.add_argument("--image-dim", type=int, default=384, help="numbers per image feature (1024 for --ee4d)")
    parser.add_argument("--explore", type=int, default=1, help="train on the best of this many noises per sample (explorative training); 1 is ordinary training")
    parser.add_argument("--one-step", action="store_true", help="always start from pure noise: the model generates a path in one pass")
    parser.add_argument("--ground", type=Path, default=None, help="folder of gray/ground.py: the model also learns to say how far the way is free (training signal only)")
    parser.add_argument("--free-weight", type=float, default=1.0, help="weight of that extra loss")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--text-drop", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--val-moments", type=int, default=1024)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.amp and device == "cuda"

    if args.ee4d is not None:
        from gray.ee4d import EeMoments

        args.image_dim, args.patches, args.ground, args.depth = 1024, None, None, None
        training, held = EeMoments(args.ee4d, "train"), EeMoments(args.ee4d, "val", jitter=False)
    else:
        data = dict(listing=args.moments, tracks=args.tracks, images=args.images, cache=args.cache, patches=args.patches, ground=args.ground, depth=args.depth)
        training, held = Moments(**data, held=False, jitter=True), Moments(**data, held=True)
    print(f"{len(training):,d} training moments, {len(held):,d} held-out moments, device {device}", flush=True)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0, collate_fn=collate)
    picked = np.random.default_rng(args.seed).permutation(len(held))[: args.val_moments]
    val_batches = list(DataLoader(torch.utils.data.Subset(held, picked.tolist()), **options))

    last = args.save_dir / "last.pt" if args.save_dir else None
    saved = torch.load(last, map_location=device) if args.resume and last and last.is_file() else None
    if saved:
        mean, std = saved["mean"].to(device), saved["std"].to(device)
    else:
        some = torch.utils.data.Subset(training, np.random.default_rng(args.seed).permutation(len(training))[:2048].tolist())
        # with workers: opening a recording's files takes about half a second on this file system, and these are 2048 moments of some 800 recordings
        poses = torch.cat([anchored(batch) for batch in DataLoader(some, batch_size=256, collate_fn=collate, num_workers=args.workers)])
        mean, std = poses.mean((0, 1)).to(device), poses.std((0, 1)).clamp_min(1e-2).to(device)

    model = HeadPath(args.width, args.layers, args.heads, patches=args.patches is not None, free=args.ground is not None, depth=args.depth is not None,
                     image_dim=args.image_dim).to(device)
    free_weight = args.free_weight if args.ground is not None else 0.0
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    step = 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        step = saved["step"]
        print(f"resumed at step {step}", flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        (args.save_dir / "config.json").write_text(json.dumps(config, indent=1))

    loader = DataLoader(training, shuffle=True, drop_last=True, **options)
    running, free_running, started = [], [], time.time()
    model.train()
    while step < args.steps:
        for batch in loader:
            loss = flow_loss(model, to_device(batch, device), mean, std, args.text_drop, amp=amp, free_weight=free_weight, explore=args.explore, one_step=args.one_step)
            if free_weight > 0:
                loss, free_error = loss
                free_running.append(free_error.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            running.append(loss.item())
            if step % args.eval_every == 0 or step == args.steps:
                train_loss, running = float(np.mean(running)), []
                result = evaluate(model, val_batches, mean, std, device, args.sample_steps, amp, args.one_step)
                print(f"step {step} | loss {train_loss:.4f} | held-out loss {result['loss']:.4f} | one draw {result['one_draw_cm']:.1f} cm, standing still {result['stay_cm']:.1f} cm | "
                      f"moments that go: {result['going_one_draw_cm']:.1f} cm, standing still {result['going_stay_cm']:.1f} cm | {time.time() - started:.0f} s", flush=True)
                if "walkable_in_view_m" in result:
                    print(f"    free distance in view, held-out: walkable off by {result['walkable_in_view_m']:.2f} m, blocked by {result['blocked_in_view_m']:.2f} m "
                          f"(training, all directions: {np.nanmean(free_running):.2f} m)", flush=True)
                    free_running = []
                if args.save_dir:
                    with (args.save_dir / "log.jsonl").open("a") as handle:
                        handle.write(json.dumps({"step": step, "train_loss": train_loss, **result}) + "\n")
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                                "step": step, "config": config, "mean": mean.cpu(), "std": std.cpu()}, last)
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
