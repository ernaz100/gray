#!/usr/bin/env python3
"""How far do generated head paths go, against the recorded ones, at each sampling setting?

    python -m gray.travel --run <runs>/head-v5

Fewer sampling steps and less starting noise cut how often a path goes through something (gray/score.py). A path that
goes less far also goes through less. This counts, on held-out moments that go 1 m or further and for the same settings,
how far the camera travels in the 5 s and how far it ends from where it began, next to the recorded paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gray.head import load, sample
from gray.joint import to_device
from gray.moments import HISTORY, Moments, collate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--moments", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if getattr(torch.version, "hip", None):
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std, config = load(args.run, device)
    listing = dict(np.load(config["moments"]))
    data = Moments(Path(config["moments"]), Path(config["tracks"]), Path(config["images"]), Path(config["cache"]), held=True, patches=Path(config["patches"]), only=listing["going"])
    picked = np.random.default_rng(args.seed).permutation(len(data))[: args.moments]
    loader = DataLoader(torch.utils.data.Subset(data, picked.tolist()), batch_size=256, num_workers=args.workers, collate_fn=collate)
    settings = [(20, 1.0), (4, 1.0), (4, 0.7), (4, 0.5), (4, 0.3), (4, 0.0)]
    length = {s: [] for s in settings} | {"recorded": []}
    reach = {s: [] for s in settings} | {"recorded": []}
    generator = torch.Generator(device=device)

    def measure(path: torch.Tensor, start: torch.Tensor) -> tuple[list, list]:
        ground = torch.cat([start[:, None], path], dim=1)[..., [0, 2]]
        return (ground[:, 1:] - ground[:, :-1]).norm(dim=-1).sum(1).cpu().tolist(), (ground[:, -1] - ground[:, 0]).norm(dim=-1).cpu().tolist()

    for batch in loader:
        batch = to_device(batch, device)
        start = batch["pose"][:, HISTORY - 1, :3]
        a, b = measure(batch["pose"][:, HISTORY:, :3], start)
        length["recorded"] += a
        reach["recorded"] += b
        for steps, noise in settings:
            generator.manual_seed(args.seed)
            a, b = measure(sample(model, batch, mean, std, steps, generator, device == "cuda", temperature=noise), start)
            length[(steps, noise)] += a
            reach[(steps, noise)] += b
    result = {}
    print(f"{len(length['recorded']):,d} held-out moments that go 1 m or further\n  {'':28s} path length   ends this far from the start   ends 1 m or further away   path length against recorded")
    for key in ["recorded"] + settings:
        name = key if key == "recorded" else f"{key[0]} steps, noise {key[1]:g}"
        r = {"length_m": float(np.mean(length[key])), "reach_m": float(np.mean(reach[key])), "reach_over_1m": float(np.mean(np.array(reach[key]) >= 1.0)),
             "ratio": float(np.mean(length[key]) / np.mean(length["recorded"]))}
        result[name] = r
        print(f"  {name:28s} {r['length_m']:8.2f} m {r['reach_m']:22.2f} m {r['reach_over_1m']:24.0%} {r['ratio']:26.0%}")
    (args.run / "travel.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.run / 'travel.json'}", flush=True)


if __name__ == "__main__":
    main()
