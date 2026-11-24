#!/usr/bin/env python3
"""The body model on moments: the head path of the next 5 s -> the body motion along it.

    python -m gray.motion --save-dir <runs>/motion-v1 --amp

Third part of the chain (text and image -> head path -> what the camera should show -> body).
A sample is a moment of gray/moments.py with the body track of gray/body.py. The model is given

    past      the head poses and the image feature of up to 30 frames up to and including now, and the body in those
              frames (a robot knows its own body). For a share of the samples (``--cold``) the past body is left
              out, as at the very start of a deployment.
    path      the head poses of the 75 frames after now. In training and in the scores below these are the recorded
              ones; in the chain they come from the head-path model (gray/head.py).
    text      the narration sentence that begins at now
    patches   optionally, the 7 x 7 image grid of the frame at now

and generates the body of the 75 frames after now: per frame the pose latent (32 numbers, MoLingo's
motion autoencoder) and the root path (pelvis x and z, sine and cosine of the root yaw), 36 numbers.
Everything is in the frame anchored at now (gray/frames.py) and standardized.

It is a flow model that predicts the clean value, built like gray/head.py: a noise level between 0
and 1 per sample, the future body mixed with noise accordingly, squared error to the clean body.

``--hml`` mixes in HumanML3D motions (gray/hml.py), which have text and a body but no images and no real camera: a learned
token stands in for the image, and (``--hml-path masked``, the default) for the head path too, so that these samples teach
what motion looks like without teaching that a head path fixes the body exactly, as it does when the "camera" is glued to
the head joint. Scores are on Nymeria's held-out homes as before.

Moments whose body is not fully known (no motion capture in part of the 7 s) are left out of
the loss and of the scores.

Every ``--eval-every`` steps, on held-out homes, with the past body (warm) and without (cold),
for one generated body per moment:

    pelvis      mean distance on the ground between the generated and the recorded pelvis (cm). For comparison:
                the pelvis placed straight under the camera
    yaw         mean difference of the root yaw (degrees)
    latent      root mean squared error of the standardized pose latent; 1.0 is what the average pose scores

Saved in the run folder: config.json, log.jsonl, last.pt, and best.pt (lowest held-out loss).
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
from gray.head import NOW, anchored
from gray.joint import level_embedding, to_device
from gray.moments import FUTURE, HISTORY, Moments, collate

BODY = 36  # the pose latent (32) and the root path (4)


class Body(nn.Module):
    """A transformer over the past frames, the future frames with their head poses, the text and (optionally) the image grid."""

    def __init__(self, width: int = 512, layers: int = 8, heads: int = 8, text_dim: int = 768, image_dim: int = 384, patches: bool = False, cells: int = 49) -> None:
        super().__init__()
        self.width = width
        self.pose_in = nn.Linear(9, width)
        self.body_in = nn.Linear(BODY, width)
        self.no_body = nn.Parameter(torch.randn(width) * 0.02)
        self.no_image = nn.Parameter(torch.randn(width) * 0.02)  # for samples without images (gray/hml.py)
        self.no_path = nn.Parameter(torch.randn(width) * 0.02)  # for samples whose head path is withheld
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
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, BODY))

    def forward(self, head: torch.Tensor, image: torch.Tensor, seen: torch.Tensor, past: torch.Tensor, warm: torch.Tensor, x: torch.Tensor,
                level: torch.Tensor, text: torch.Tensor, text_valid: torch.Tensor, patches: torch.Tensor | None = None,
                no_image: torch.Tensor | None = None, no_path: torch.Tensor | None = None) -> torch.Tensor:
        """head (B, 105, 9), image (B, 30, 384), seen (B, 30) which past frames are given, past (B, 30, 36) the body in them,
        warm (B,) whether that body is given, x (B, 75, 36) the noised future body, level (B,) -> the clean future body (B, 75, 36).
        no_image (B,): the sample has no images; no_path (B,): its head path is withheld. Both default to False."""
        size, device = head.shape[0], head.device
        seconds = (torch.arange(HISTORY + FUTURE, device=device) - NOW) / RATE
        clock = self.time_in(time_embedding(seconds[None], self.width))
        own = torch.where(warm[:, None, None], self.body_in(past), self.no_body)
        blind = torch.zeros(size, dtype=torch.bool, device=device) if no_image is None else no_image
        lost = torch.zeros(size, dtype=torch.bool, device=device) if no_path is None else no_path
        where = torch.where(lost[:, None, None], self.no_path, self.pose_in(head))
        picture = torch.where(blind[:, None, None], self.no_image, self.image_in(image))
        before = where[:, :HISTORY] + picture + own + clock[:, :HISTORY] + self.kind[0]
        after = where[:, HISTORY:] + self.body_in(x) + clock[:, HISTORY:] + self.kind[1] + self.level_in(level_embedding(level))[:, None]
        has_text = text_valid.any(dim=1)
        words = torch.where(has_text[:, None, None], self.text_in(text) + self.kind[2], self.no_text)
        valid = torch.where(has_text[:, None], text_valid, torch.arange(text.shape[1], device=device)[None] == 0)
        sequence = [before, after.to(before.dtype), words.to(before.dtype)]
        padding = [~seen, torch.zeros(size, FUTURE, dtype=torch.bool, device=device), ~valid]
        if hasattr(self, "patch_in"):
            sequence.append((self.patch_in(patches) + self.patch_where + self.kind[3]).to(before.dtype))
            padding.append(blind[:, None].expand(-1, sequence[-1].shape[1]))  # no image, no image grid
        hidden = self.blocks(torch.cat(sequence, dim=1), src_key_padding_mask=torch.cat(padding, dim=1))
        return self.out(hidden[:, HISTORY : HISTORY + FUTURE])


def body_anchored(batch: dict) -> torch.Tensor:
    """The body (B, 105, 36) of a batch with its root path in the frame anchored at now."""
    now = torch.full((batch["pose"].shape[0],), NOW, device=batch["pose"].device)
    return torch.cat([batch["body"][..., :32], frames.anchor_root(batch["body"][..., 32:], batch["pose"], now)], dim=-1)


def inputs(batch: dict, stats: dict, history: torch.Tensor | int = HISTORY):
    """head, image, seen, past body, future body (all standardized), text, text_valid, patches."""
    head = (anchored(batch) - stats["head_mean"]) / stats["head_std"]
    body = (body_anchored(batch) - stats["body_mean"]) / stats["body_std"]
    size, device = head.shape[0], head.device
    history = torch.full((size,), history, device=device) if isinstance(history, int) else history
    seen = torch.arange(HISTORY, device=device)[None] >= HISTORY - history[:, None]
    return head, batch["image"], seen, body[:, :HISTORY], body[:, HISTORY:], batch["text"], batch["text_valid"], batch.get("patches")


def flow_loss(model: Body, batch: dict, stats: dict, text_drop: float = 0.0, cold: float = 0.0, random_history: bool = True,
              generator: torch.Generator | None = None, amp: bool = False, hide_path: bool = True) -> torch.Tensor:
    """``hide_path``: samples without a real camera (HumanML3D) are given a token in place of their head path."""
    size, device = batch["pose"].shape[0], batch["pose"].device
    draw = lambda *shape: torch.rand(*shape, generator=generator, device=device)  # noqa: E731
    history = (draw(size) * HISTORY).long() + 1 if random_history else torch.full((size,), HISTORY, device=device)
    foreign = batch["foreign"] if "foreign" in batch else torch.zeros(size, dtype=torch.bool, device=device)
    if "history_max" in batch:
        history = torch.minimum(history, batch["history_max"])
    head, image, seen, past, future, text, text_valid, patches = inputs(batch, stats, history)
    text_valid = text_valid & (draw(size) >= text_drop)[:, None]
    warm = draw(size) >= cold
    level = draw(size)
    mixed = (1 - level[:, None, None]) * torch.randn(future.shape, generator=generator, device=device) + level[:, None, None] * future
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        clean = model(head, image, seen, past, warm, mixed, level, text, text_valid, patches, no_image=foreign, no_path=foreign & hide_path)
    error = ((clean.float() - future) ** 2).mean((1, 2))
    known = batch["body_known"].float()
    return (error * known).sum() / known.sum().clamp_min(1)


@torch.no_grad()
def sample(model: Body, batch: dict, stats: dict, steps: int = 20, generator: torch.Generator | None = None, amp: bool = False, warm: bool = True,
           temperature: float = 1.0, head: torch.Tensor | None = None) -> torch.Tensor:
    """One generated body per sample (B, 75, 36), in the frame anchored at now, not standardized.

    ``head``: other head poses (B, 105, 9), anchored at now and in metres, to use in place of the recorded ones (a generated path).
    """
    own, image, seen, past, future, text, text_valid, patches = inputs(batch, stats)
    if head is not None:
        own = (head - stats["head_mean"]) / stats["head_std"]
    known = torch.full((own.shape[0],), warm, dtype=torch.bool, device=own.device)
    x = torch.randn(future.shape, generator=generator, device=own.device) * temperature
    for step in range(steps):
        level = torch.full((x.shape[0],), step / steps, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            clean = model(own, image, seen, past, known, x, level, text, text_valid, patches).float()
        x = x + (clean - x) / (1 - step / steps) / steps
    return x * stats["body_std"] + stats["body_mean"]


@torch.no_grad()
def evaluate(model: Body, batches: list, stats: dict, device: str, steps: int, amp: bool) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(0)
    sums = {key: 0.0 for key in ("loss", "under_cm", "warm_pelvis_cm", "warm_yaw_deg", "warm_latent", "cold_pelvis_cm", "cold_yaw_deg", "cold_latent")}
    count = 0
    for batch in batches:
        batch = to_device(batch, device)
        known = batch["body_known"]
        if not known.any():
            continue
        size = int(known.sum())
        sums["loss"] += flow_loss(model, batch, stats, random_history=False, generator=generator, amp=amp).item() * size
        truth = body_anchored(batch)[:, HISTORY:]
        head = anchored(batch)[:, HISTORY:]
        sums["under_cm"] += (torch.hypot(head[..., 0] - truth[..., 32], head[..., 2] - truth[..., 33]).mean(-1)[known] * 100).sum().item()
        for name, warm in (("warm", True), ("cold", False)):
            drawn = sample(model, batch, stats, steps, generator, amp, warm)
            turn = torch.atan2(drawn[..., 34], drawn[..., 35]) - torch.atan2(truth[..., 34], truth[..., 35])
            sums[f"{name}_pelvis_cm"] += (torch.hypot(drawn[..., 32] - truth[..., 32], drawn[..., 33] - truth[..., 33]).mean(-1)[known] * 100).sum().item()
            sums[f"{name}_yaw_deg"] += torch.rad2deg(torch.atan2(torch.sin(turn), torch.cos(turn)).abs()).mean(-1)[known].sum().item()
            sums[f"{name}_latent"] += (((drawn[..., :32] - truth[..., :32]) / stats["body_std"][:32]) ** 2).mean((1, 2))[known].sum().item()
        count += size
    model.train()
    result = {key: value / max(count, 1) for key, value in sums.items()}
    result["warm_latent"], result["cold_latent"] = math.sqrt(result["warm_latent"]), math.sqrt(result["cold_latent"])
    return result | {"moments": count}


def load(folder: Path, device: str, best: bool = True) -> tuple[Body, dict, dict]:
    """A trained body model, its statistics and its configuration."""
    path = folder / "best.pt" if best and (folder / "best.pt").is_file() else folder / "last.pt"
    saved = torch.load(path, map_location=device)
    config = saved["config"]
    model = Body(config["width"], config["layers"], config["heads"], patches=config.get("patches") not in (None, "None")).to(device)
    model.load_state_dict(saved["model"], strict=False)  # runs from before the tokens for missing images and paths lack them
    return model.eval(), {key: value.to(device) for key, value in saved["stats"].items()}, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--moments", type=Path, default=Path("/ptmp/enazaren/gray/moments/moments.npz"))
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--images", type=Path, default=Path("/ptmp/enazaren/gray/pinhole"))
    parser.add_argument("--body", type=Path, default=Path("/ptmp/enazaren/gray/body"))
    parser.add_argument("--patches", type=Path, default=None, help="folder of gray/patches.py: the model also sees the image grid of the frame at now")
    parser.add_argument("--everything", action="store_true", help="train on all moments, not only the navigation subset")
    parser.add_argument("--wide", type=int, default=None, help="with --everything: move now by up to this many frames either way")
    parser.add_argument("--cold", type=float, default=0.3, help="share of training samples without the past body")
    parser.add_argument("--hml", type=Path, default=None, help="folder of gray/hml.py: also train on HumanML3D motions (no images; text and body)")
    parser.add_argument("--hml-share", type=float, default=0.3, help="share of the training samples drawn from HumanML3D")
    parser.add_argument("--hml-path", choices=("masked", "given"), default="masked",
                        help="masked: HumanML3D samples get a token in place of their head path, which is made from the head joint; given: they get that path")
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

    data = dict(listing=args.moments, tracks=args.tracks, images=args.images, cache=args.cache, patches=args.patches, body=args.body)
    training, held = Moments(**data, held=False, jitter=True, everything=args.everything, wide=args.wide), Moments(**data, held=True)
    print(f"{len(training):,d} training moments, {len(held):,d} held-out moments, device {device}", flush=True)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0, collate_fn=collate)
    picked = np.random.default_rng(args.seed).permutation(len(held))[: args.val_moments]
    val_batches = list(DataLoader(torch.utils.data.Subset(held, picked.tolist()), **options))

    last = args.save_dir / "last.pt" if args.save_dir else None
    saved = torch.load(last, map_location=device) if args.resume and last and last.is_file() else None
    if saved:
        stats = {key: value.to(device) for key, value in saved["stats"].items()}
    else:
        some = torch.utils.data.Subset(training, np.random.default_rng(args.seed).permutation(len(training))[:2048].tolist())
        heads, bodies = [], []
        for batch in DataLoader(some, batch_size=256, collate_fn=collate, num_workers=args.workers):
            heads.append(anchored(batch))
            bodies.append(body_anchored(batch)[batch["body_known"]])
        heads, bodies = torch.cat(heads), torch.cat(bodies)
        stats = {"head_mean": heads.mean((0, 1)), "head_std": heads.std((0, 1)).clamp_min(1e-2), "body_mean": bodies.mean((0, 1)), "body_std": bodies.std((0, 1)).clamp_min(1e-2)}
        stats = {key: value.to(device) for key, value in stats.items()}
        print(f"the body is known in {len(bodies) / len(heads):.0%} of the training moments", flush=True)

    model = Body(args.width, args.layers, args.heads, patches=args.patches is not None).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    step, best = 0, float("inf")
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        step, best = saved["step"], saved.get("best", float("inf"))
        print(f"resumed at step {step}", flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        (args.save_dir / "config.json").write_text(json.dumps(config, indent=1))

    if args.hml is not None:
        from gray.hml import HmlMoments

        other = HmlMoments(args.hml)
        weights = torch.cat([torch.full((len(training),), (1 - args.hml_share) / len(training)), torch.full((len(other),), args.hml_share / len(other))])
        print(f"also {len(other):,d} samples of {other.motions:,d} HumanML3D motions, drawn for {args.hml_share:.0%} of the training samples; their head path is {args.hml_path}", flush=True)
        sampler = torch.utils.data.WeightedRandomSampler(weights.double(), num_samples=len(training), replacement=True)
        loader = DataLoader(torch.utils.data.ConcatDataset([training, other]), sampler=sampler, drop_last=True, **options)
    else:
        loader = DataLoader(training, shuffle=True, drop_last=True, **options)
    running, started = [], time.time()
    model.train()
    while step < args.steps:
        for batch in loader:
            loss = flow_loss(model, to_device(batch, device), stats, args.text_drop, args.cold, amp=amp, hide_path=args.hml_path == "masked")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            running.append(loss.item())
            if step % args.eval_every == 0 or step == args.steps:
                train_loss, running = float(np.mean(running)), []
                r = evaluate(model, val_batches, stats, device, args.sample_steps, amp)
                print(f"step {step} | loss {train_loss:.4f} | held-out loss {r['loss']:.4f} | pelvis: warm {r['warm_pelvis_cm']:.1f} cm, cold {r['cold_pelvis_cm']:.1f} cm, "
                      f"under the camera {r['under_cm']:.1f} cm | yaw: warm {r['warm_yaw_deg']:.1f}, cold {r['cold_yaw_deg']:.1f} degrees | "
                      f"latent: warm {r['warm_latent']:.2f}, cold {r['cold_latent']:.2f} | {r['moments']} moments | {time.time() - started:.0f} s", flush=True)
                if args.save_dir:
                    with (args.save_dir / "log.jsonl").open("a") as handle:
                        handle.write(json.dumps({"step": step, "train_loss": train_loss, **r}) + "\n")
                    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "step": step, "config": config,
                             "stats": {key: value.cpu() for key, value in stats.items()}, "best": min(best, r["loss"])}
                    torch.save(state, last)
                    if r["loss"] < best:
                        best = r["loss"]
                        torch.save(state, args.save_dir / "best.pt")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
