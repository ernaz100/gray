#!/usr/bin/env python3
"""In which description of an image does "the same place again" look most alike, and a changed place most different?

    python -m gray.spaces

The comparator compares a predicted with an observed description of the image. Whatever it
compares, two views of the same unchanged place differ a little (hands, people, what is being
handled), and that is its false alarms. This asks, with no predictor involved, which description
has the least of that variation relative to real change. For frames of the held-out homes it
takes three other frames (as gray/alarm.py does):

    revisit    the same recording, nearly the same camera pose, at least 10 s apart     nothing changed
    other day  another recording of the same home, nearly the same pose                 furniture, clutter, light may differ
    elsewhere  the same recording, a different place                                    a different scene

and measures the distance to each in several descriptions:

    class token            DINOv3 class token of the pinhole view (gray/pinhole.py), each value scaled by its spread
    grid                   the 7 x 7 grid of patch features (gray/patches.py), likewise, averaged over the cells
    grid, ahead            the same over the lower middle of the image only (rows 4 to 7, columns 3 to 5), where the way lies
    depth                  estimated depth (gray/depth.py), 28 x 28: mean absolute difference of the logarithm
    depth, scale removed   the same after dividing each depth map by its median: the model's scale drifts from frame to frame
    depth, ahead           the latter over the lower middle only

For each description: the median distance per kind, and how well the distance tells a revisit
from the other two (area under the ROC curve). The best description for a comparator is the
one in which a revisit is told from "other day" and "elsewhere" most reliably.
Writes <out>/spaces.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gray.alarm import APART, Places
from gray.features import CACHE
from gray.moments import held_locations
from gray.probe import area
from gray.tracks import to_world
import math

import torch

from gray import frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("/ptmp/enazaren/gray/runs/spaces"))
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--images", type=Path, default=Path("/ptmp/enazaren/gray/pinhole"))
    parser.add_argument("--grids", type=Path, default=Path("/ptmp/enazaren/gray/grids"))
    parser.add_argument("--depth", type=Path, default=Path("/ptmp/enazaren/gray/depth"))
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--frames", type=int, default=60, help="frames per recording")
    parser.add_argument("--tight", type=float, default=0.35)
    args = parser.parse_args()

    location = {m["sequence_id"]: m["location"] for m in json.loads((args.cache / "index.json").read_text())["sequences"]}
    held = held_locations(args.cache)
    by_location = defaultdict(list)
    for name in sorted(n for n in location if location[n] in held and (args.maps / f"{n}.npz").exists() and (args.tracks / f"{n}.npz").exists()):
        by_location[location[name]].append(name)
    rng = np.random.default_rng(0)
    opened: dict[tuple[str, str], np.ndarray] = {}

    def read(kind: str, name: str, frame: int) -> np.ndarray:
        if (kind, name) not in opened:
            path = {"token": args.images / name / "dino.npy", "grid": args.grids / name / "patches.npy", "depth": args.depth / name / "depth.npy"}[kind]
            opened[(kind, name)] = np.load(path, mmap_mode="r")
        return np.array(opened[(kind, name)][frame], dtype=np.float32)

    groups = []  # per anchor frame: {"true": (name, frame), "revisit": ..., ...}
    for where, names in by_location.items():
        places = Places(names, args.tracks)
        for number, name in enumerate(names):
            track = np.load(args.tracks / f"{name}.npz")
            usable = np.flatnonzero(track["valid"])
            for frame in rng.choice(usable, size=min(args.frames, len(usable)), replace=False):
                pose = track["pose"][frame]
                world = to_world(pose[None, :3].astype(np.float64), float(track["floor"][frame]))[0]
                yaw = float(frames.heading(torch.from_numpy(pose[None, 3:])))
                here = np.array([*world, math.atan2(-math.cos(yaw), -math.sin(yaw)), pose[8]])
                found = {"true": (name, int(frame))}
                away = int(frame) + int(rng.integers(APART, 20 * APART)) * (1 if rng.random() < 0.5 else -1)
                if 0 <= away < len(track["pose"]):
                    found["elsewhere"] = (name, away)
                for kind, same in (("revisit", True), ("other day", False)):
                    match = places.like(here, number, int(frame), same, rng, args.tight)
                    if match is not None:
                        found[kind] = (names[match[0]], match[1])
                groups.append(found)
        print(f"{where}: {len(names)} recordings", flush=True)

    spread = {"token": np.stack([read("token", *g["true"]) for g in groups[::7]]).std(0) + 1e-6,
              "grid": np.concatenate([read("grid", *g["true"]) for g in groups[::7]]).std(0) + 1e-6}
    ahead7 = np.zeros((7, 7), bool); ahead7[3:, 2:5] = True
    ahead28 = np.zeros((28, 28), bool); ahead28[12:, 8:20] = True

    def distances(a: tuple[str, int], b: tuple[str, int]) -> dict[str, float]:
        token = ((read("token", *a) - read("token", *b)) / spread["token"]) ** 2
        grid = (((read("grid", *a) - read("grid", *b)) / spread["grid"]) ** 2).mean(-1)  # (49,)
        da, db = np.log(np.clip(read("depth", *a), 0.05, None)), np.log(np.clip(read("depth", *b), 0.05, None))
        free = np.abs((da - np.median(da)) - (db - np.median(db)))
        return {"class token": float(token.mean()), "grid": float(grid.mean()), "grid, ahead": float(grid[ahead7.ravel()].mean()),
                "depth": float(np.abs(da - db).mean()), "depth, scale removed": float(free.mean()), "depth, ahead": float(free[ahead28].mean())}

    table = defaultdict(lambda: defaultdict(list))
    for g in groups:
        if "revisit" not in g:
            continue
        for kind in ("revisit", "other day", "elsewhere"):
            if kind in g:
                for space, value in distances(g["true"], g[kind]).items():
                    table[space][kind].append(value)
    counts = {kind: len(table["class token"][kind]) for kind in ("revisit", "other day", "elsewhere")}
    print(f"\n{len(groups):,d} frames; with a revisit {counts['revisit']:,d}, of which with another day {counts['other day']:,d}; nearly the same pose: "
          f"within {0.25 * args.tight:.2f} m and {15 * args.tight:.0f} degrees")
    print(f"  {'description':24s} median distance: revisit   other day   elsewhere   |   told from a revisit (AUC): other day   elsewhere")
    result = {}
    for space, kinds in table.items():
        revisit, other, away = (np.array(kinds[k]) for k in ("revisit", "other day", "elsewhere"))
        told = lambda x: area(np.concatenate([revisit, x]), np.concatenate([np.zeros(len(revisit), bool), np.ones(len(x), bool)]))  # noqa: E731
        result[space] = {"revisit": float(np.median(revisit)), "other day": float(np.median(other)), "elsewhere": float(np.median(away)),
                         "auc other day": told(other), "auc elsewhere": told(away)}
        r = result[space]
        print(f"  {space:24s} {r['revisit']:24.3f} {r['other day']:11.3f} {r['elsewhere']:11.3f}   |   {r['auc other day']:36.2f} {r['auc elsewhere']:11.2f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "spaces.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out / 'spaces.json'}", flush=True)


if __name__ == "__main__":
    main()
