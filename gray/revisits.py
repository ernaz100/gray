#!/usr/bin/env python3
"""For every frame of every recording: another frame of the same recording that shows the same place again.

    python -m gray.revisits

A revisit of a frame is a frame of the same recording, at least 10 s away in time, taken from
nearly the same camera pose (the limits of gray/alarm.py times 0.35: 9 cm, 5 degrees of
heading, 3.5 degrees of tilt, 10 cm of height). Nothing in the room has changed between the two, but
hands, people and what is being handled have. gray/compare.py can train on such frames as
targets, so that its predicted uncertainty learns how much an unchanged place varies.

    <out>/<sequence id>.npy   (N,) i32: for each frame of the track one revisit, picked at random among those found; -1 if none
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from gray import frames
from gray.alarm import APART, HEIGHT, NEAR, TILT, TURN
from gray.tracks import to_world

OUT = Path("/ptmp/enazaren/gray/revisits")
TIGHT = 0.35


def one(job: tuple[str, str]) -> tuple[str, int, int]:
    path, out = job
    torch.set_num_threads(1)
    name = Path(path).stem
    track = np.load(path)
    pose = track["pose"]
    world = to_world(pose[:, :3].astype(np.float64), track["floor"])
    yaw = frames.heading(torch.from_numpy(pose[:, 3:])).numpy()
    heading = np.arctan2(-np.cos(yaw), -np.sin(yaw))
    valid = np.flatnonzero(track["valid"])
    tree = cKDTree(world[valid, :2])
    rng = np.random.default_rng(len(pose))
    found = np.full(len(pose), -1, dtype=np.int32)
    for i, near in zip(valid, tree.query_ball_point(world[valid, :2], NEAR * TIGHT)):
        other = valid[np.array(near, dtype=int)]
        other = other[np.abs(other - i) >= APART]
        if len(other) == 0:
            continue
        turn = np.abs((heading[other] - heading[i] + math.pi) % (2 * math.pi) - math.pi)
        ok = (turn < math.radians(TURN) * TIGHT) & (np.abs(pose[other, 8] - pose[i, 8]) < TILT * TIGHT) & (np.abs(world[other, 2] - world[i, 2]) < HEIGHT * TIGHT)
        if ok.any():
            found[i] = rng.choice(other[ok])
    np.save(Path(out) / f"{name}.npy", found)
    return name, len(pose), int((found >= 0).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(args.out)) for p in sorted(args.tracks.glob("*.npz"))]
    results = []
    # "spawn": a forked worker hung in its first torch call on Viper (job 13603238, 15 minutes without output)
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(one, jobs, chunksize=4):
            results.append(result)
            if len(results) % 100 == 0:
                print(f"{len(results)}/{len(jobs)}", flush=True)
    total, have = sum(r[1] for r in results), sum(r[2] for r in results)
    print(f"{len(results)} recordings, {total:,d} frames; {have:,d} have a revisit ({have / total:.0%})", flush=True)


if __name__ == "__main__":
    main()
