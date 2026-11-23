#!/usr/bin/env python3
"""Experiment 1: can image features be predicted along a known head path?

    python -m gray.features --pose full --save-dir <dir>     # sees the head path
    python -m gray.features --pose none --save-dir <dir>     # sees only how much time has passed

The comparator needs a prediction of what the camera will show from a given pose. This trains
only that part, on STEER's cache: DINOv3 class tokens (384-d, 15 Hz) and head poses of 8 s
windows. A window is split at "now": the frames up to now are given (1 to ``--max-history`` of
them), the later ones are hidden, and the model predicts their features from the head poses.
All poses are in the frame anchored at now (``frames.anchor``).

Three predictors are compared on held-out locations (the same ones STEER's training holds out):

    copy    the feature at now, unchanged
    none    a model that knows the time since now but not the pose
    full    a model that knows the pose of every frame

Errors are mean squared errors on standardized features, so 1.0 is what predicting the average
frame of the training set scores. They are reported by time since now, by how far the view has
turned and by how far the head has moved. If ``full`` is not clearly below ``none`` and ``copy``
beyond the first second, the class token does not carry enough of the geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gray import frames

CACHE = Path(os.environ.get("GRAY_CACHE", "/ptmp/enazaren/nymeria/egoprimo_cache_all"))
TOKENS = 120
RATE = 15.0  # tokens per second
BINS = {
    "seconds": (0.5, 1.0, 2.0, 4.0),
    "turn_deg": (15.0, 45.0, 90.0),
    "distance_m": (0.25, 1.0, 2.0),
}
COLUMNS = ("mse", "mse_copy", "cosine", "cosine_copy")


class Windows(Dataset):
    """(image features (120, 384), head poses (120, 9)) of cached windows; arrays are memory-mapped."""

    def __init__(self, cache: Path, items: list[tuple[str, int]]) -> None:
        self.cache, self.items = cache, items
        self._open: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, window = self.items[index]
        if sequence not in self._open:
            self._open[sequence] = {n: np.load(self.cache / sequence / f"{n}.npy", mmap_mode="r") for n in ("dino", "starts", "head")}
        a = self._open[sequence]
        first = int(a["starts"][window]) // 2
        image = np.array(a["dino"][first : first + TOKENS], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(np.array(a["head"][window], dtype=np.float32))


def split(cache: Path, holdout: float, min_clock: float) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Training and held-out windows; every k-th location by name is held out, as in STEER's train.py."""
    sequences = json.loads((cache / "index.json").read_text())["sequences"]
    locations = sorted({m["location"] for m in sequences})
    every = max(2, round(1 / holdout))
    held = set(locations[every - 1 :: every])
    usable = [m for m in sequences if m["clock_gyro_correlation"] >= min_clock]
    train = [(m["sequence_id"], w) for m in usable if m["location"] not in held for w in range(m["windows"])]
    val = [(m["sequence_id"], w) for m in usable if m["location"] in held for w in range(m["windows"])]
    return train, val


def time_embedding(seconds: torch.Tensor, width: int) -> torch.Tensor:
    """(B, L) seconds from now -> (B, L, width) sines and cosines, periods from 16 s down to 1/8 s."""
    frequency = 2 * math.pi * 2 ** torch.linspace(-4, 3, width // 2, device=seconds.device)
    angle = seconds[..., None] * frequency
    return torch.cat([angle.sin(), angle.cos()], dim=-1)


class Predictor(nn.Module):
    """A transformer over the window's tokens: seen frames carry their feature, hidden ones a learned placeholder."""

    def __init__(self, dim: int, width: int, layers: int, heads: int, pose: bool) -> None:
        super().__init__()
        self.width, self.pose = width, pose
        self.feature_in = nn.Linear(dim, width)
        self.hidden = nn.Parameter(torch.zeros(width))
        self.time_in = nn.Linear(width, width)
        self.pose_in = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width)) if pose else None
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))

    def forward(self, features: torch.Tensor, seen: torch.Tensor, head: torch.Tensor, seconds: torch.Tensor) -> torch.Tensor:
        x = torch.where(seen[..., None], self.feature_in(features), self.hidden.to(features.dtype))
        x = x + self.time_in(time_embedding(seconds, self.width))
        if self.pose_in is not None:
            x = x + self.pose_in(head)
        return self.out(self.blocks(x))


def prepare(image: torch.Tensor, head: torch.Tensor, history: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """A batch split at now = history - 1: standardized features, seen mask, anchored poses, seconds from now."""
    now = history - 1
    position = torch.arange(image.shape[1], device=image.device)[None]
    seen = position <= now[:, None]
    seconds = (position - now[:, None]) / RATE
    return (image - mean) / std, seen, frames.anchor(head, now), seconds


@torch.no_grad()
def evaluate(model: Predictor, batches: list, mean: torch.Tensor, std: torch.Tensor, device: str, history: int) -> dict:
    """Errors on the hidden frames with a fixed history, as tables over time, turning and distance."""
    model.eval()
    sums = {name: torch.zeros(len(edges) + 1, len(COLUMNS) + 1, dtype=torch.float64) for name, edges in BINS.items()}
    for image, head in batches:
        image, head = image.to(device), head.to(device)
        size = image.shape[0]
        now = torch.full((size,), history - 1, device=device)
        target, seen, anchored, seconds = prepare(image, head, now + 1, mean, std)
        predicted = model(target, seen, anchored, seconds)
        copy = target[:, history - 1 : history].expand_as(target)
        rows = torch.stack([
            ((predicted - target) ** 2).mean(-1),
            ((copy - target) ** 2).mean(-1),
            torch.cosine_similarity(predicted * std + mean, image, dim=-1),
            torch.cosine_similarity(image[:, history - 1 : history].expand_as(image), image, dim=-1),
            torch.ones_like(seconds),
        ], dim=-1)[~seen].double().cpu()
        keys = {
            "seconds": seconds,
            "turn_deg": frames.heading(anchored[..., 3:]).abs() * 180 / math.pi,
            "distance_m": torch.hypot(anchored[..., 0], anchored[..., 2]),
        }
        for name, value in keys.items():
            index = torch.bucketize(value[~seen].cpu(), torch.tensor(BINS[name]))
            sums[name].index_add_(0, index, rows)
    model.train()
    result = {}
    for name, table in sums.items():
        edges = (0.0, *BINS[name], math.inf)
        result[name] = []
        for i, row in enumerate(table):
            count = max(float(row[-1]), 1.0)
            entry = {"from": edges[i], "to": edges[i + 1], "frames": int(row[-1])}
            entry.update({column: float(row[j]) / count for j, column in enumerate(COLUMNS)})
            result[name].append(entry)
    total = sums["seconds"].sum(0)
    result["all"] = {column: float(total[j] / total[-1]) for j, column in enumerate(COLUMNS)}
    return result


def load(folder: Path, device: str) -> tuple[Predictor, torch.Tensor, torch.Tensor, dict]:
    """A trained predictor from its run folder, with the feature mean and standard deviation it was trained with."""
    saved = torch.load(folder / "model.pt", map_location=device)
    config = saved["config"]
    model = Predictor(saved["mean"].shape[0], config["width"], config["layers"], config["heads"], config["pose"] == "full").to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved["mean"].to(device), saved["std"].to(device), config


def table_text(result: dict) -> str:
    lines = []
    for name in BINS:
        lines.append(f"  {name:<12}      frames     mse    copy  cosine    copy")
        for e in result[name]:
            span = f"{e['from']:g} to {e['to']:g}"
            lines.append(f"  {span:<12} {e['frames']:>11,d}  {e['mse']:6.3f}  {e['mse_copy']:6.3f}  {e['cosine']:6.3f}  {e['cosine_copy']:6.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--pose", choices=("full", "none"), default="full")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--max-history", type=int, default=30, help="most frames given, in tokens (15 per second)")
    parser.add_argument("--eval-history", type=int, nargs="+", default=[1, 15])
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--min-clock", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--val-windows", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if torch.version.hip:  # Viper's MI300A: rocBLAS, see STEER's paths.setup_torch
        torch.backends.cuda.preferred_blas_library("cublas")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # three memory maps per sequence and worker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    order = np.random.default_rng(args.seed)

    train_items, val_items = split(args.cache, args.holdout, args.min_clock)
    print(f"{len(train_items):,d} training windows, {len(val_items):,d} held-out windows, device {device}", flush=True)
    val_items = [val_items[i] for i in order.permutation(len(val_items))[: args.val_windows]]
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0)
    val_batches = list(DataLoader(Windows(args.cache, val_items), **loader_options))

    sample = Windows(args.cache, [train_items[i] for i in order.permutation(len(train_items))[:2048]])
    images, heads = zip(*DataLoader(sample, **loader_options))
    images, heads = torch.cat(images), torch.cat(heads)
    mean, std = images.mean((0, 1)).to(device), images.std((0, 1)).clamp_min(1e-4).to(device)
    drift = (frames.anchor(heads, torch.zeros(len(heads), dtype=torch.long)) - heads).abs().amax((1, 2))
    print(f"re-anchoring at token 0 moves the cached poses by {float(drift.median()):.1e} (median), {float(drift.max()):.1e} (max)", flush=True)

    model = Predictor(images.shape[-1], args.width, args.layers, args.heads, args.pose == "full").to(device)
    print(f"pose {args.pose}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    loader = DataLoader(Windows(args.cache, train_items), shuffle=True, drop_last=True, **loader_options)

    step, running, started, log = 0, 0.0, time.time(), []
    while step < args.steps:
        for image, head in loader:
            image, head = image.to(device, non_blocking=True), head.to(device, non_blocking=True)
            history = torch.randint(1, args.max_history + 1, (image.shape[0],), device=device)
            target, seen, anchored, seconds = prepare(image, head, history, mean, std)
            loss = ((model(target, seen, anchored, seconds) - target) ** 2).mean(-1)[~seen].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step, running = step + 1, running + loss.item()
            if step % args.eval_every == 0 or step == args.steps:
                results = {h: evaluate(model, val_batches, mean, std, device, h) for h in args.eval_history}
                train_loss, running = running / (args.eval_every if step % args.eval_every == 0 else step % args.eval_every), 0.0
                parts = [f"history {h}: mse {r['all']['mse']:.3f} (copy {r['all']['mse_copy']:.3f})" for h, r in results.items()]
                print(f"step {step} loss {train_loss:.3f} | " + " | ".join(parts) + f" | {time.time() - started:.0f} s", flush=True)
                log.append({"step": step, "train_loss": train_loss, "history": {str(h): r for h, r in results.items()}})
            if step == args.steps:
                break

    for h, r in results.items():
        print(f"\nheld-out, {h} frame(s) given, pose {args.pose}\n{table_text(r)}", flush=True)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (args.save_dir / "results.json").write_text(json.dumps({"config": config, "log": log}, indent=1))
        torch.save({"model": model.state_dict(), "config": config, "mean": mean.cpu(), "std": std.cpu()}, args.save_dir / "model.pt")
        print(f"wrote {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
