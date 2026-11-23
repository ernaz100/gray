#!/usr/bin/env python3
"""Experiment 2, second look: are the generated head paths wrong, or valid but different?

    python -m gray.paths --run <runs>/joint-v1

One sampled head path against one recording cannot tell the two apart. So, on held-out windows
that have a narration, for a warm start (1 s of history) and a cold start (one frame, no body):

1. Several paths per window (``--draws``), all from the real text. Reported in cm by horizon:

       one draw      error of a single draw, averaged over the draws
       best draw     error of the draw that is closest to the recording over the whole window
       expected      error of the model's expected path (one step from pure noise, no sampling)
       stay          error of staying at the pose of now
       spread        distance between two draws for the same window

   A best draw well under "stay", with a spread of the size of the error, says the recording is
   among the futures the model offers. "Expected" under "stay" says it knows something about
   where the head goes on average.

2. The same draws (same noise) with no text and with the text of another window:

       no text, other text      error of a single draw
       moved by ...             distance to the draw made with the real text from the same noise

   If the text matters, the real text gives the smallest error and the paths move when it changes.

It also reports how well the distance to the path's end after 8 s is predicted (correlation of
the mean over the draws with the recording): does the model know whether the person walks or stays?
"""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gray.features import CACHE, RATE, TOKENS, split
from gray.joint import HEAD, MOTION, SECONDS, Windows, collate, expected, generate, load, raw_tokens, seconds_from, to_device, restrict

ROWS = ("one draw", "best draw", "expected", "stay", "spread", "no text", "other text", "moved by no text", "moved by other text")


@torch.no_grad()
def score(model, batches: list, mean: torch.Tensor, std: torch.Tensor, device: str, history: int, cold: bool, draws: int, steps: int, amp: bool) -> dict:
    generator = torch.Generator(device=device)
    sums = torch.zeros(len(ROWS) + 1, len(SECONDS) + 1, dtype=torch.float64, device=device)
    ends = []
    for batch in batches:
        batch = to_device(batch, device)
        batch = {key: value[batch["text_valid"].any(dim=1)] for key, value in batch.items()}
        size = batch["head"].shape[0]
        if size < 2:
            continue
        now = torch.full((size,), history - 1, device=device)
        seconds = seconds_from(now)
        raw = raw_tokens(batch, now)
        x1 = (raw - mean) / std
        known = (seconds <= 0)[..., None].repeat(1, 1, 3)
        if cold:
            known[..., MOTION] = False
        texts = {
            "real": (batch["text"], batch["text_valid"]),
            "none": (batch["text"], torch.zeros_like(batch["text_valid"])),
            "other": (batch["text"].roll(1, dims=0), batch["text_valid"].roll(1, dims=0)),
        }

        def positions(text: tuple, draw: int) -> torch.Tensor:  # (B, L, 3) in metres; the same noise for the same draw
            generator.manual_seed(1000 + draw)
            return (generate(model, x1, known, seconds, *text, order=(HEAD,), steps=steps, generator=generator, amp=amp) * std + mean)[..., :3]

        paths = {name: torch.stack([positions(text, d) for d in range(draws)]) for name, text in texts.items()}  # (K, B, L, 3)
        truth = raw[..., :3]
        future = seconds > 0
        error = {name: (p - truth).norm(dim=-1) * 100 for name, p in paths.items()}  # (K, B, L) cm
        best = (error["real"] * future).sum(-1).argmin(dim=0)  # (B,)
        generator.manual_seed(0)
        mean_path = (expected(model, x1, known, seconds, *texts["real"], generator=generator, amp=amp) * std + mean)[..., :3]
        values = [
            error["real"].mean(0),
            error["real"][best, torch.arange(size, device=device)],
            (mean_path - truth).norm(dim=-1) * 100,
            (raw[:, history - 1 : history, :3] - truth).norm(dim=-1) * 100,
            (paths["real"] - paths["real"].roll(1, dims=0)).norm(dim=-1).mean(0) * 100,
            error["none"].mean(0),
            error["other"].mean(0),
            (paths["none"] - paths["real"]).norm(dim=-1).mean(0) * 100,
            (paths["other"] - paths["real"]).norm(dim=-1).mean(0) * 100,
            torch.ones_like(seconds),
        ]
        index = torch.bucketize(seconds[future], torch.tensor(SECONDS, device=device))
        sums.index_add_(1, index, torch.stack([v[future] for v in values]).double())
        horizontal = lambda p: torch.hypot(p[..., -1, 0], p[..., -1, 2])  # distance from now to the last frame, on the ground
        ends.append(torch.stack([horizontal(truth), horizontal(paths["real"]).mean(0), horizontal(paths["real"]).std(0)], dim=-1).cpu())
    ends = torch.cat(ends)
    edges = (0.0, *SECONDS, TOKENS / RATE)
    table = torch.cat([sums[:-1] / sums[-1], (sums[:-1].sum(1) / sums[-1].sum())[:, None]], dim=1).cpu()
    return {
        "windows": len(ends), "horizons": [f"{a:g} to {b:g} s" for a, b in zip(edges[:-1], edges[1:])] + ["all"],
        "cm": {name: [float(v) for v in row] for name, row in zip(ROWS, table)},
        "end_distance": {
            "correlation": float(torch.corrcoef(ends[:, :2].T)[0, 1]),
            "recorded_m": float(ends[:, 0].mean()), "generated_m": float(ends[:, 1].mean()), "between_draws_m": float(ends[:, 2].mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std, config = load(args.run, device)
    cache = args.cache or Path(config["cache"])
    images = Path(config["images"]) if config.get("images") else None  # the image features the run was trained on
    _train, items = split(cache, float(config["holdout"]), float(config["min_clock"]))
    if config.get("subset"):
        items = restrict(items, Path(config["subset"]))
    order = np.random.default_rng(int(config["seed"])).permutation(len(items))[: args.windows]  # the first of the windows training evaluated on
    batches = list(DataLoader(Windows(cache, [items[i] for i in order], images), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate))
    results = {}
    for name, history, cold in (("warm", 15, False), ("cold", 1, True)):
        results[name] = r = score(model, batches, mean, std, device, history, cold, args.draws, args.steps, device == "cuda")
        e = r["end_distance"]
        print(f"\n{name} start, {r['windows']} windows with text, {args.draws} draws each; head position in cm", flush=True)
        print(f"  {'':<22}" + "".join(f"{h:>12}" for h in r["horizons"]))
        for row, numbers in r["cm"].items():
            print(f"  {row:<22}" + "".join(f"{v:12.1f}" for v in numbers), flush=True)
        print(f"  distance from now to the end of the window: recorded {e['recorded_m']:.2f} m, generated {e['generated_m']:.2f} m "
              f"(between draws {e['between_draws_m']:.2f} m), correlation {e['correlation']:.2f}", flush=True)
    (args.run / "paths.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.run / 'paths.json'}", flush=True)


if __name__ == "__main__":
    main()
