#!/usr/bin/env python3
"""Experiment 1, second look: how good is the feature prediction in a location the model knows?

    python -m gray.evaluate --run <runs>/features-v1

The predictors of features.py are scored on four sets of windows:

    seen windows       a sample of the windows they were trained on
    known location     recordings they never saw, in locations they were trained on
    Loc_08             the one such recording in Loc_08 (the location being rebuilt in 3D)
    new location       recordings in the six held-out locations, as in features.py

The unseen recordings in known locations are the sequences training left out for a weak fit of
the body clock (STEER's ``--min-clock``). Their image features and head poses do not depend on
that fit, but they are not a random draw: Workout and Simon says are over-represented. So read
each row against its own copy column.
"""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gray.features import BINS, CACHE, Windows, evaluate, load


def sets(cache: Path, holdout: float, min_clock: float) -> dict[str, list[tuple[str, int]]]:
    sequences = json.loads((cache / "index.json").read_text())["sequences"]
    locations = sorted({m["location"] for m in sequences})
    held = set(locations[max(2, round(1 / holdout)) - 1 :: max(2, round(1 / holdout))])
    windows = lambda keep: [(m["sequence_id"], w) for m in sequences if keep(m) for w in range(m["windows"])]
    weak = lambda m: m["clock_gyro_correlation"] < min_clock
    return {
        "seen windows": windows(lambda m: not weak(m) and m["location"] not in held),
        "known location": windows(lambda m: weak(m) and m["location"] not in held),
        "Loc_08": windows(lambda m: weak(m) and m["location"] == "Loc_08"),
        "new location": windows(lambda m: not weak(m) and m["location"] in held),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="folder with full/ and none/ from features.sbatch")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--windows", type=int, default=8192, help="at most this many windows per set")
    parser.add_argument("--history", type=int, nargs="+", default=[1, 15])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {pose: load(args.run / pose, device) for pose in ("full", "none")}
    config = models["full"][3]

    order, results = np.random.default_rng(1), {}
    for name, items in sets(args.cache, float(config["holdout"]), float(config["min_clock"])).items():
        if not items:
            print(f"{name}: no windows", flush=True)
            continue
        picked = [items[i] for i in order.permutation(len(items))[: args.windows]]
        batches = list(DataLoader(Windows(args.cache, picked), batch_size=256, num_workers=args.workers))
        results[name] = {"windows": len(picked), "of": len(items), "sequences": len({s for s, _ in items})}
        for history in args.history:
            scored = {pose: evaluate(model, batches, mean, std, device, history) for pose, (model, mean, std, _c) in models.items()}
            results[name][str(history)] = scored
            print(f"\n{name}: {len(picked):,d} of {len(items):,d} windows from {results[name]['sequences']} sequences, {history} frame(s) given", flush=True)
            print("  seconds       with poses  time only        copy")
            rows = [(f"{a['from']:g} to {a['to']:g}", a, b) for a, b in zip(scored["full"]["seconds"], scored["none"]["seconds"])]
            rows.append(("all", scored["full"]["all"], scored["none"]["all"]))
            for label, full, none in rows:
                print(f"  {label:<12}  {full['mse']:10.3f}  {none['mse']:9.3f}  {full['mse_copy']:10.3f}", flush=True)
    (args.run / "evaluate.json").write_text(json.dumps({"bins": BINS, "sets": results}, indent=1))
    print(f"\nwrote {args.run / 'evaluate.json'}", flush=True)


if __name__ == "__main__":
    main()
