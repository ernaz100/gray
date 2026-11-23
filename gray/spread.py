#!/usr/bin/env python3
"""Does the joint model know when its feature prediction is uncertain?

    python -m gray.spread --run <runs>/joint-v1

The comparator so far thresholds the raw mismatch between the predicted and the observed image
feature. A tracking filter (Kalman) would divide that mismatch by how uncertain the prediction
was: a large mismatch after turning into an unseen room is expected, the same mismatch in a
hallway that was in view is not. This script tests whether such an uncertainty is already in the
joint model of gray/joint.py, without training anything.

For held-out windows, with 1 s of history and the real head path given, the image feature of
every later frame is drawn several times. Per frame:

    mismatch   squared error of the expected feature (one step from noise) against the observed one, as in Experiment 3
    spread     variance between the draws, averaged over the 384 values: the model's own uncertainty
    ratio      mismatch / spread

Reported: whether the spread is large where the mismatch is large (rank correlation, and the mean
mismatch per fifth of the spread), both against how far the head has turned since now and how
many seconds have passed, and whether the ratio is steadier than the raw mismatch (its 90th and
99th percentile relative to its median). Recordings contain no unexpected changes of the world,
so this cannot show that the ratio detects them; it shows whether the expected surprises are removed.
Writes <run>/spread.json.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gray import frames
from gray.features import split
from gray.joint import FEATURE, HEAD, SIZES, Windows, collate, expected, generate, load, raw_tokens, seconds_from, to_device

HISTORY = 15
TURNS = (0, 15, 45, 90, 181)  # degrees
SECONDS = (0, 1, 2, 4, 8)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=1024)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    model, mean, std, config = load(args.run, device)
    cache = Path(config["cache"])
    images = Path(config["images"]) if config.get("images") else None
    _train, items = split(cache, float(config["holdout"]), float(config["min_clock"]))
    items = [items[i] for i in np.random.default_rng(0).permutation(len(items))[: args.windows]]
    loader = DataLoader(Windows(cache, items, images), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    feature = slice(SIZES[0], SIZES[0] + SIZES[1])
    generator = torch.Generator(device=device)
    columns = {key: [] for key in ("mismatch", "spread", "normalized", "turn", "seconds")}
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            size = batch["head"].shape[0]
            now = torch.full((size,), HISTORY - 1, device=device)
            seconds = seconds_from(now)
            raw = raw_tokens(batch, now)
            x1 = (raw - mean) / std
            known = (seconds <= 0)[..., None].repeat(1, 1, 3)
            known[..., 2] = False  # no body motion
            known[..., HEAD] = True  # the real head path
            text = (batch["text"], batch["text_valid"])
            generator.manual_seed(0)
            middle = expected(model, x1, known, seconds, *text, generator=generator, amp=amp)[..., feature]
            draws = []
            for draw in range(args.draws):
                generator.manual_seed(1000 + draw)
                draws.append(generate(model, x1, known, seconds, *text, order=(FEATURE,), steps=args.steps, generator=generator, amp=amp)[..., feature])
            variance = torch.stack(draws).var(dim=0)  # (B, L, 384)
            error = (middle - x1[..., feature]) ** 2
            future = seconds > 0
            columns["mismatch"].append(error.mean(-1)[future].cpu())
            columns["spread"].append(variance.mean(-1)[future].cpu())
            columns["normalized"].append((error / variance.clamp_min(1e-4)).mean(-1)[future].cpu())  # per value, as a filter would
            columns["turn"].append((frames.heading(raw[..., 3:9]).abs() * 180 / math.pi)[future].cpu())
            columns["seconds"].append(seconds[future].cpu())
            print(f"  {sum(len(c) for c in columns['mismatch']):,d} frames", flush=True)
    a = {key: torch.cat(value).double().numpy() for key, value in columns.items()}
    a["ratio"] = a["mismatch"] / np.maximum(a["spread"], 1e-6)
    total = len(a["mismatch"])
    correlation = float(np.corrcoef(ranks(a["mismatch"]), ranks(a["spread"]))[0, 1])
    print(f"\n{total:,d} future frames of {len(items):,d} held-out windows, {args.draws} draws; mismatch {a['mismatch'].mean():.3f}, spread {a['spread'].mean():.3f} (1.0 is the average frame)")
    print(f"rank correlation of spread and mismatch: {correlation:.2f}")
    result = {"frames": total, "rank_correlation": correlation, "by_spread": [], "by_turn": [], "by_seconds": [], "steadiness": {}}

    edges = np.quantile(a["spread"], np.linspace(0, 1, 6))
    print("\nmean mismatch per fifth of the spread (a calibrated spread rises with it)\n  spread            mismatch   ratio")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (a["spread"] >= lo) & (a["spread"] <= hi)
        result["by_spread"].append({"spread": [float(lo), float(hi)], "mismatch": float(a["mismatch"][mask].mean()), "ratio": float(a["ratio"][mask].mean())})
        print(f"  {lo:.3f} to {hi:.3f} {a['mismatch'][mask].mean():10.3f} {a['ratio'][mask].mean():7.2f}")

    for title, key, bins, unit, store in (("how far the head has turned since now", "turn", TURNS, "deg", "by_turn"), ("time since now", "seconds", SECONDS, "s", "by_seconds")):
        print(f"\nby {title}\n  {'':14s} frames   mismatch   spread   ratio   per-value ratio")
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (a[key] > lo) & (a[key] <= hi) if key == "seconds" else (a[key] >= lo) & (a[key] < hi)
            if not mask.any():
                continue
            row = {"range": [lo, hi], "frames": int(mask.sum()), "mismatch": float(a["mismatch"][mask].mean()), "spread": float(a["spread"][mask].mean()),
                   "ratio": float(a["ratio"][mask].mean()), "normalized": float(a["normalized"][mask].mean())}
            result[store].append(row)
            print(f"  {lo:3g} to {hi:3g} {unit:4s} {row['frames']:8,d} {row['mismatch']:10.3f} {row['spread']:8.3f} {row['ratio']:7.2f} {row['normalized']:12.2f}")

    print("\nhow steady is each signal? its 90th and 99th percentile relative to its median (smaller is steadier)")
    for label, key in (("raw mismatch", "mismatch"), ("mismatch / spread", "ratio"), ("per-value ratio", "normalized")):
        median, high, top = np.quantile(a[key], [0.5, 0.9, 0.99])
        result["steadiness"][label] = {"p90_over_median": float(high / median), "p99_over_median": float(top / median)}
        print(f"  {label:20s} {high / median:6.2f} {top / median:6.2f}")
    worst = result["by_turn"][-1]["mismatch"] / result["by_turn"][0]["mismatch"], result["by_turn"][-1]["ratio"] / result["by_turn"][0]["ratio"]
    print(f"\nlargest turns against smallest: the raw mismatch is {worst[0]:.2f} times as large, the ratio {worst[1]:.2f} times")
    (args.run / "spread.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {args.run / 'spread.json'}", flush=True)


if __name__ == "__main__":
    main()
