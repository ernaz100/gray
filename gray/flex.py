#!/usr/bin/env python3
"""The chain with a noise level per frame: any frames of a window may be given, and the rest are generated.

    python -m gray.flex --part head --ee4d <folder> --save-dir <runs>/ee-head-v2 --amp
    python -m gray.flex --part body --ee4d <folder> --save-dir <runs>/ee-motion-v2 --amp

gray/head.py and gray/motion.py split a sample at "now": 2 s are given, 5 s are generated, with one noise level for all
of them. Here a sample is one window of ``--frames`` frames (120, 8 s, for UniEgoMotion's data; a Nymeria moment would be
105) without such a split. Every frame has its own noise level: 1 where the frame is given, and the level of the current
sampling step where it is generated. What is given is therefore chosen when sampling, not when training:

    part    generates              per frame, each of these may be given or not
    head    the head path          the head pose (a given one is clean), the image feature
    body    the body along it      the body (a given one is clean), the head pose, the image feature

UniEgoMotion's three tasks are three such choices (no body is given in any of them):

    generation       head: the first pose and image             body: the generated head path, the first image
    forecasting      head: poses and images of the first 2 s    body: the generated head path, the images of 2 s
    reconstruction   (the head path is given)                    body: the whole head path, all images

and ours with 2 s of history is a fourth: head and body given the poses, images and body of the first 2 s.

All poses are in the frame anchored at the window's first frame (gray/frames.py) and standardized. Both parts are flow
models that predict the clean value, as in gray/head.py: generated frames are mixed with noise at the drawn level, the
loss is the squared error on them. A window that reaches past the end of its recording is shorter; its missing frames
are left out of the model's view and of the loss.

In training what is given is drawn per sample (``draw``): nothing, the first frames (any number), or frames scattered
over the window. Every ``--eval-every`` steps, on held-out windows, for one generated sample each:

    head    distance on the ground to the recorded path (cm), from the first frame and from 2 s; next to it standing still
    body    pelvis distance (cm) and error of the pose latent along the recorded head path, with no body given
            (reconstruction) and with the body of the first 2 s

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
from gray.features import RATE, time_embedding
from gray.joint import level_embedding, to_device
from gray.moments import collate

SIZE = {"head": 9, "body": 36}  # numbers per frame: a head pose; the pose latent (32) and the root path (4)
SECONDS_GIVEN = 2  # "the first 2 s" of forecasting and of our history


class Flex(nn.Module):
    """A transformer over one token per frame, and the text tokens."""

    def __init__(self, part: str, width: int = 512, layers: int = 8, heads: int = 8, image_dim: int = 1024, text_dim: int = 768) -> None:
        super().__init__()
        self.part, self.width = part, width
        self.x_in = nn.Linear(SIZE[part], width)
        self.level_in = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.image_in = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, width))
        self.no_image = nn.Parameter(torch.randn(width) * 0.02)
        if part == "body":
            self.path_in = nn.Linear(SIZE["head"], width)
            self.no_path = nn.Parameter(torch.randn(width) * 0.02)
        self.time_in = nn.Linear(width, width)
        self.text_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width))
        self.no_text = nn.Parameter(torch.randn(width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, SIZE[part]))

    def forward(self, x: torch.Tensor, level: torch.Tensor, image: torch.Tensor, shown: torch.Tensor, exists: torch.Tensor,
                text: torch.Tensor, text_valid: torch.Tensor, path: torch.Tensor | None = None, led: torch.Tensor | None = None) -> torch.Tensor:
        """x (B, L, 9 or 36) the part's frames, each mixed with noise at its level (B, L); image (B, L, D) and shown (B, L),
        which of them are given; exists (B, L), which frames the window has; text (B, S, 768), text_valid (B, S).
        Body only: path (B, L, 9) the head poses and led (B, L), which of them are given. -> the clean frames (B, L, 9 or 36)."""
        length, device = x.shape[1], x.device
        clock = self.time_in(time_embedding(torch.arange(length, device=device)[None] / RATE, self.width))
        tokens = self.x_in(x) + self.level_in(level_embedding(level)) + torch.where(shown[..., None], self.image_in(image), self.no_image) + clock
        if self.part == "body":
            tokens = tokens + torch.where(led[..., None], self.path_in(path), self.no_path)
        has_text = text_valid.any(dim=1)
        words = torch.where(has_text[:, None, None], self.text_in(text), self.no_text)
        valid = torch.where(has_text[:, None], text_valid, torch.arange(text.shape[1], device=device)[None] == 0)  # one placeholder token without text
        hidden = self.blocks(torch.cat([tokens, words.to(tokens.dtype)], dim=1), src_key_padding_mask=torch.cat([~exists, ~valid], dim=1))
        return self.out(hidden[:, :length])


def values(batch: dict, stats: dict) -> dict[str, torch.Tensor]:
    """The head poses (B, L, 9) and the body (B, L, 36) of a batch in the frame anchored at the first frame, standardized."""
    first = torch.zeros(batch["pose"].shape[0], dtype=torch.long, device=batch["pose"].device)
    body = torch.cat([batch["body"][..., :32], frames.anchor_root(batch["body"][..., 32:], batch["pose"], first)], dim=-1)
    return {"head": (frames.anchor(batch["pose"], first) - stats["head_mean"]) / stats["head_std"], "body": (body - stats["body_mean"]) / stats["body_std"]}


def metres(x: torch.Tensor, stats: dict, part: str) -> torch.Tensor:
    return x * stats[f"{part}_std"] + stats[f"{part}_mean"]


def first(exists: torch.Tensor, count: int | torch.Tensor | None) -> torch.Tensor:
    """(B, L): the first ``count`` frames of each window (None: all of them), as far as they exist."""
    if count is None:
        return exists.clone()
    count = count[:, None] if isinstance(count, torch.Tensor) else count
    return exists & (torch.arange(exists.shape[1], device=exists.device)[None] < count)


def draw(part: str, exists: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """What a training sample is given, each (B, L): "clean" the part's own frames, "shown" the images, and for the body "led" the head poses.

    The part's own frames: the first ones, any number that leaves 1 s to generate (head: half of the samples, body: half),
    frames scattered over the window, each with a chance of up to 30 % (a fifth), or as little as can be (the rest: for the
    head its first pose, which says only how high and how tilted the camera is; for the body nothing).
    Images: for the head wherever its pose is given (a camera sees where it is). For the body the first frame's, those of
    the first frames, or all (a third each), within the head poses it is given.
    Head poses, for the body: all (70 %), the first frames (20 %) or only the first (10 %).
    """
    size, length, device = *exists.shape, exists.device
    uniform = lambda *shape: torch.rand(*shape, generator=generator, device=device)  # noqa: E731
    count = exists.sum(1)
    some = lambda least: least + (uniform(size) * (count - least + 1)).long()  # noqa: E731  between least and all of the frames
    kind = uniform(size)
    opening = first(exists, 1 + (uniform(size) * (count - RATE)).long().clamp_min(0))
    scattered = exists & (uniform(size, length) < 0.3 * uniform(size, 1))
    clean = torch.where((kind < 0.5)[:, None], opening, torch.where((kind < 0.7)[:, None], scattered, torch.zeros_like(exists)))
    if part == "head":
        clean = clean | first(exists, 1)
        return {"clean": clean, "shown": clean}
    kind = uniform(size)
    led = torch.where((kind < 0.7)[:, None], exists, torch.where((kind < 0.9)[:, None], first(exists, some(1)), first(exists, 1)))
    kind = uniform(size)
    shown = torch.where((kind < 1 / 3)[:, None], first(exists, 1), torch.where((kind < 2 / 3)[:, None], first(exists, some(1)), exists))
    return {"clean": clean, "shown": shown & (led | first(exists, 1)), "led": led}


def flow_loss(model: Flex, batch: dict, stats: dict, given: dict | None = None, text_drop: float = 0.0, generator: torch.Generator | None = None, amp: bool = False) -> torch.Tensor:
    """``given``: what the samples are given (as ``draw`` returns it); drawn if None."""
    own, exists = values(batch, stats), batch["exists"]
    target, device = own[model.part], exists.device
    given = draw(model.part, exists, generator) if given is None else given
    clean = given["clean"]
    text_valid = batch["text_valid"] & (torch.rand(len(exists), generator=generator, device=device) >= text_drop)[:, None]
    level = torch.where(clean, 1.0, torch.rand(len(exists), 1, generator=generator, device=device))
    mixed = (1 - level[..., None]) * torch.randn(target.shape, generator=generator, device=device) + level[..., None] * target
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        guess = model(mixed, level, batch["image"], given["shown"], exists, batch["text"], text_valid, own["head"], given.get("led"))
    made = (exists & ~clean).float()
    return (((guess.float() - target) ** 2).mean(-1) * made).sum() / made.sum().clamp_min(1)


@torch.no_grad()
def sample(model: Flex, batch: dict, stats: dict, given: dict, steps: int = 20, generator: torch.Generator | None = None, amp: bool = False,
           temperature: float = 1.0, path: torch.Tensor | None = None) -> torch.Tensor:
    """One generated window per sample (B, L, 9 or 36), in the frame anchored at the first frame, in metres; given frames are returned as given.

    ``given``: "clean", "shown" and for the body "led", each (B, L), as ``draw`` returns them (``first`` makes the usual ones).
    ``path``: for the body, other head poses (B, L, 9) in that frame and in metres to use in place of the recorded ones (a generated path).
    ``temperature``: the starting noise is multiplied by this.
    """
    own, exists, clean = values(batch, stats), batch["exists"], given["clean"]
    target = own[model.part]
    head = own["head"] if path is None else (path - stats["head_mean"]) / stats["head_std"]
    x = torch.where(clean[..., None], target, torch.randn(target.shape, generator=generator, device=target.device) * temperature)
    for step in range(steps):
        level = torch.where(clean, 1.0, step / steps)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            guess = model(x, level, batch["image"], given["shown"], exists, batch["text"], batch["text_valid"], head, given.get("led")).float()
        x = torch.where(clean[..., None], target, x + (guess - x) / (1 - step / steps) / steps)  # the last step lands on the prediction
    return metres(x, stats, model.part)


@torch.no_grad()
def evaluate(model: Flex, batches: list, stats: dict, device: str, steps: int, amp: bool) -> dict[str, float]:
    """On held-out windows: the training loss, and one generated sample per window against the recorded one (see the top)."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(0)
    sums: dict[str, float] = {}
    opening = int(SECONDS_GIVEN * RATE)

    def add(name: str, per_frame: torch.Tensor, frames_scored: torch.Tensor) -> None:
        sums[name] = sums.get(name, 0.0) + (per_frame * frames_scored).sum().item()
        sums[name + " frames"] = sums.get(name + " frames", 0.0) + frames_scored.sum().item()

    for batch in batches:
        batch = to_device(batch, device)
        exists, truth = batch["exists"], {part: metres(x, stats, part) for part, x in values(batch, stats).items()}
        add("loss", flow_loss(model, batch, stats, generator=generator, amp=amp).expand(len(exists)), torch.ones(len(exists), device=device))
        ground = lambda a, b: torch.hypot(a[..., 0] - b[..., 0], a[..., -1] - b[..., -1]) * 100  # noqa: E731  x and z of a head pose, or of the pelvis
        if model.part == "head":
            for name, count in (("first_frame", 1), ("two_seconds", opening)):
                clean = first(exists, count)
                drawn = sample(model, batch, stats, {"clean": clean, "shown": clean}, steps, generator, amp)
                scored = (exists & ~clean).float()
                add(f"{name}_cm", ground(drawn[..., [0, 2]], truth["head"][..., [0, 2]]), scored)
                add(f"{name}_still_cm", ground(truth["head"][:, count - 1 : count, [0, 2]], truth["head"][..., [0, 2]]), scored)
        else:
            pelvis, scale = truth["body"][..., 32:34], stats["body_std"][:32]
            add("under_camera_cm", ground(truth["head"][..., [0, 2]], pelvis), exists.float())
            for name, count in (("reconstruction", 0), ("two_seconds", opening)):
                clean = first(exists, count)
                drawn = sample(model, batch, stats, {"clean": clean, "shown": exists if count == 0 else clean, "led": exists}, steps, generator, amp)
                scored = (exists & ~clean).float()
                add(f"{name}_pelvis_cm", ground(drawn[..., 32:34], pelvis), scored)
                add(f"{name}_latent", (((drawn[..., :32] - truth["body"][..., :32]) / scale) ** 2).mean(-1), scored)
    model.train()
    result = {name: total / max(sums[name + " frames"], 1) for name, total in sums.items() if not name.endswith(" frames")}
    return {name: math.sqrt(value) if name.endswith("latent") else value for name, value in result.items()}


def report(part: str, r: dict[str, float]) -> str:
    if part == "head":
        return (f"from the first frame {r['first_frame_cm']:.1f} cm (standing still {r['first_frame_still_cm']:.1f}) | "
                f"from 2 s {r['two_seconds_cm']:.1f} cm (standing still {r['two_seconds_still_cm']:.1f})")
    return (f"pelvis: reconstruction {r['reconstruction_pelvis_cm']:.1f} cm, with 2 s of body {r['two_seconds_pelvis_cm']:.1f} cm, under the camera {r['under_camera_cm']:.1f} cm | "
            f"latent: {r['reconstruction_latent']:.2f}, {r['two_seconds_latent']:.2f}")


def load(folder: Path, device: str, best: bool = True) -> tuple[Flex, dict, dict]:
    """A trained model, its statistics and its configuration."""
    saved = torch.load(folder / "best.pt" if best and (folder / "best.pt").is_file() else folder / "last.pt", map_location=device)
    config = saved["config"]
    model = Flex(config["part"], config["width"], config["layers"], config["heads"], config["image_dim"]).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), {key: value.to(device) for key, value in saved["stats"].items()}, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", choices=tuple(SIZE), required=True)
    parser.add_argument("--ee4d", type=Path, required=True, help="folder of gray/ee4d.py: UniEgoMotion's data (no text)")
    parser.add_argument("--frames", type=int, default=120, help="frames per window, 15 per second")
    parser.add_argument("--image-dim", type=int, default=1024)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--text-drop", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--val-windows", type=int, default=1024)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from gray.ee4d import EeWindows

    torch.manual_seed(args.seed)
    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.amp and device == "cuda"

    training, held = EeWindows(args.ee4d, "train", args.frames), EeWindows(args.ee4d, "val", args.frames, jitter=False)
    print(f"{len(training):,d} training windows, {len(held):,d} held-out windows of up to {args.frames} frames, device {device}", flush=True)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0, collate_fn=collate)
    picked = np.random.default_rng(args.seed).permutation(len(held))[: args.val_windows]
    val_batches = list(DataLoader(torch.utils.data.Subset(held, picked.tolist()), **options))

    last = args.save_dir / "last.pt" if args.save_dir else None
    saved = torch.load(last, map_location=device) if args.resume and last and last.is_file() else None
    if saved:
        stats = {key: value.to(device) for key, value in saved["stats"].items()}
    else:  # mean and standard deviation per number, over the real frames of 2,048 training windows
        some = torch.utils.data.Subset(training, np.random.default_rng(args.seed).permutation(len(training))[:2048].tolist())
        plain = {"head_mean": torch.zeros(9), "head_std": torch.ones(9), "body_mean": torch.zeros(36), "body_std": torch.ones(36)}
        parts: dict[str, list] = {"head": [], "body": []}
        for batch in DataLoader(some, batch_size=256, collate_fn=collate, num_workers=args.workers):
            for part, x in values(batch, plain).items():
                parts[part].append(x[batch["exists"]])
        stats = {f"{part}_mean": torch.cat(rows).mean(0).to(device) for part, rows in parts.items()}
        stats |= {f"{part}_std": torch.cat(rows).std(0).clamp_min(1e-2).to(device) for part, rows in parts.items()}

    model = Flex(args.part, args.width, args.layers, args.heads, args.image_dim).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    step, best = 0, float("inf")
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        step, best = saved["step"], saved["best"]
        print(f"resumed at step {step}", flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        (args.save_dir / "config.json").write_text(json.dumps(config, indent=1))

    loader = DataLoader(training, shuffle=True, drop_last=True, **options)
    running, started = [], time.time()
    model.train()
    while step < args.steps:
        for batch in loader:
            loss = flow_loss(model, to_device(batch, device), stats, text_drop=args.text_drop, amp=amp)
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
                print(f"step {step} | loss {train_loss:.4f} | held-out loss {r['loss']:.4f} | {report(args.part, r)} | {time.time() - started:.0f} s", flush=True)
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
