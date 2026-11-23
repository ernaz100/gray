#!/usr/bin/env python3
"""Does the comparator notice when the camera shows something other than what belongs there?

    python -m gray.alarm --run <runs>/compare-v2

Recordings contain no unexpected changes, so the comparator (gray/compare.py) is handed one. For
held-out moments it predicts the image feature of the frame 1, 2 and 4 s after now, with its
uncertainty, along the recorded head path. That prediction is then compared with four frames:

    true       the frame the camera really recorded                            the world is as expected
    revisit    another frame of the same recording, at least 10 s away in      the same place seen again: nothing changed,
               time, taken from nearly the same camera pose                    so this must look like "true" (false-alarm check)
    other day  a frame of another recording of the same home, from nearly      the same place on another day: furniture moved,
               the same camera pose                                            clutter, light, people
    elsewhere  a frame of the same recording from a different place            a different scene altogether

"Nearly the same pose": within 0.25 m, 15 degrees of heading, 10 degrees of tilt and 0.3 m of height.
All recordings of a home share one world frame, so such frames can be looked up.

Three signals are compared: the raw mismatch; the surprise, which divides the error of each value by
the predicted variance; and the surprise with a floor, which divides by that variance plus a fixed
amount per value. The floor is how much a value differs between two views of the same place at
different times (hands, people, what is being handled: variation that is not a change of the
room), measured on revisits in one half of the moments and applied to the other half. A tracking
filter has the same term: the noise of the measurement itself.

For a comparator that predicts the image grid (compare.py --target grid) the same is done on the grid of the frame, and
there is a fourth signal: the surprise with floor over only the cells where the path goes next (gray.compare.corridor),
so that what is at the edge of the image does not count.

Two more signals test cheap ways to quieten false alarms. For a comparator of the class token: the surprise with floor
averaged over five consecutive frames (a third of a second), since a hand in front of the lens passes and an obstacle
stays. For a grid comparator: the mean of the three most surprised cells on the path, since a chair fills a few cells
and averaging over the whole corridor dilutes it.

For each kind and signal it reports the median, and how well the signal tells that kind from
"true" on the same moments (area under the ROC curve: 0.5 cannot tell, 1 always can), and the
share above the level that 5 % of the true frames exceed. Writes <run>/alarm.json.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Subset

from gray import frames
from gray.compare import KEPT, corridor, load, predict
from gray.joint import to_device
from gray.moments import HISTORY, Moments, collate
from gray.probe import area
from gray.tracks import to_world

AHEAD = (15, 30, 60)  # frames after now: 1, 2 and 4 s
NEAR, TURN, TILT, HEIGHT, APART = 0.25, 15.0, 0.17, 0.3, 150  # metres, degrees, up-part of the view axis, metres, frames


class Places:
    """Every third frame of all recordings of one location, to look up frames taken from nearly the same pose."""

    def __init__(self, names: list[str], tracks: Path) -> None:
        parts = []
        for number, name in enumerate(names):
            track = np.load(tracks / f"{name}.npz")
            index = np.flatnonzero(track["valid"])[::3]
            pose = track["pose"][index]
            world = to_world(pose[:, :3].astype(np.float64), track["floor"][index])
            yaw = frames.heading(torch.from_numpy(pose[:, 3:])).numpy()
            parts.append(np.column_stack([world, np.arctan2(-np.cos(yaw), -np.sin(yaw)), pose[:, 8], np.full(len(index), number), index]))
        self.names = names
        self.table = np.concatenate(parts)  # x, y, z, heading, tilt, recording, frame
        self.tree = cKDTree(self.table[:, :2])

    def like(self, row: np.ndarray, recording: int, frame: int, same: bool, rng: np.random.Generator, tight: float = 1.0) -> tuple[int, int] | None:
        """A frame (recording, frame) from nearly the pose of ``row`` (x, y, z, heading, tilt): in the same recording at another time, or in another one."""
        found = np.array(self.tree.query_ball_point(row[:2], NEAR * tight), dtype=int)
        if len(found) == 0:
            return None
        t = self.table[found]
        turn = np.abs((t[:, 3] - row[3] + math.pi) % (2 * math.pi) - math.pi)
        ok = (turn < math.radians(TURN) * tight) & (np.abs(t[:, 4] - row[4]) < TILT * tight) & (np.abs(t[:, 2] - row[2]) < HEIGHT * tight)
        ok &= ((t[:, 5] == recording) & (np.abs(t[:, 6] - frame) >= APART)) if same else (t[:, 5] != recording)
        if not ok.any():
            return None
        pick = t[ok][rng.integers(int(ok.sum()))]
        return int(pick[5]), int(pick[6])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--moments", type=int, default=4000)
    parser.add_argument("--tight", type=float, default=1.0, help="factor on what counts as nearly the same pose; 0.35 is about 9 cm and 5 degrees")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scale, config = load(args.run, device)
    cache, tracks, images = Path(config["cache"]), Path(config["tracks"]), Path(config["images"])
    grid = config.get("target") == "grid"
    grids = Path(config["patches"])
    data = Moments(Path(config["moments"]), tracks, images, cache, held=True, patches=grids, future=True, everything=True, future_grid=grid)
    picked = np.random.default_rng(0).permutation(len(data))[: args.moments]
    loader = DataLoader(Subset(data, picked.tolist()), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    location = {m["sequence_id"]: m["location"] for m in json.loads((cache / "index.json").read_text())["sequences"]}
    by_location = defaultdict(list)
    for name in sorted(set(data.table["sequence"][data.rows].tolist())):
        by_location[location[name]].append(name)
    places = {where: Places(names, tracks) for where, names in by_location.items()}
    print(f"{len(picked):,d} held-out moments in {len(places)} locations ({sum(len(n) for n in by_location.values())} recordings); nearly the same pose: within "
          f"{NEAR * args.tight:.2f} m, {TURN * args.tight:.0f} degrees of heading, {HEIGHT * args.tight:.2f} m of height", flush=True)

    opened: dict[str, np.ndarray] = {}
    loaded: dict[str, dict] = {}
    def feature(name: str, frame: int) -> np.ndarray:
        """The observed feature of one frame as the comparator sees it: the standardized class token, or the reduced grid (49, 64)."""
        if name not in opened:
            opened[name] = np.load(grids / name / "patches.npy", mmap_mode="r") if grid else np.load(images / name / "dino.npy", mmap_mode="r")
        raw = np.array(opened[name][frame], dtype=np.float32)
        if grid:  # the same reduction as gray.compare.reduced, here without a trip to the GPU per frame
            return ((raw - cell_mean) @ cell_basis) / cell_std
        return (raw - mean_image) / std_image

    rng = np.random.default_rng(1)
    mean_image, std_image = scale["image_mean"].cpu().numpy(), scale["image_std"].cpu().numpy()
    if grid:
        cell_mean, cell_basis, cell_std = (scale[key].cpu().numpy() for key in ("cell_mean", "cell_basis", "cell_std"))
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            mean, variance, target = (v.cpu().numpy() for v in predict(model, batch, scale, amp=device == "cuda"))
            for b in range(len(mean)):
                row = int(batch["row"][b])
                name, now = str(data.table["sequence"][row]), int(batch["now"][b])
                where = places[location[name]]
                number = where.names.index(name)
                if name not in loaded:
                    loaded[name] = np.load(tracks / f"{name}.npz")
                track = loaded[name]
                for ahead in AHEAD:
                    frame = now + ahead
                    pose = track["pose"][frame]
                    world = to_world(pose[None, :3].astype(np.float64), float(track["floor"][frame]))[0]
                    yaw = float(frames.heading(torch.from_numpy(pose[None, 3:])))
                    here = np.array([*world, math.atan2(-math.cos(yaw), -math.sin(yaw)), pose[8]])
                    place = KEPT.index(ahead - 1) if grid else ahead - 1  # where this frame is among the predicted ones
                    seen = {"true": target[b, place]}
                    run = {"true": (name, frame)}  # where each observed frame comes from, to fetch its neighbours in time
                    away = int(rng.integers(APART, 20 * APART)) * (1 if rng.random() < 0.5 else -1)
                    other_frame = frame + away if 0 <= frame + away < len(track["pose"]) else frame - away
                    if 0 <= other_frame < len(track["pose"]):
                        seen["elsewhere"] = feature(name, other_frame)
                        run["elsewhere"] = (name, other_frame)
                    for kind, same in (("revisit", True), ("other day", False)):
                        match = where.like(here, number, frame, same, rng, args.tight)
                        if match is not None:
                            seen[kind] = feature(where.names[match[0]], match[1])
                            run[kind] = (where.names[match[0]], match[1])
                    item = {"ahead": ahead, "half": len(rows) // len(AHEAD) % 2, "variance": variance[b, place].astype(np.float32)}
                    if grid:
                        item["corridor"] = corridor(batch["pose"][b].cpu(), HISTORY - 1 + ahead, reach=min(30, batch["pose"].shape[1] - HISTORY - ahead - 1)).numpy()
                    for kind, observed in seen.items():
                        item[kind] = ((observed - mean[b, place]) ** 2).astype(np.float32)
                        if not grid:  # the two frames before and after as well: each against the prediction for its own time
                            source, at = run[kind]
                            near = [(at + shift, place + shift) for shift in (-2, -1, 1, 2)]
                            if all(0 <= f < len(opened[source]) for f, _p in near) if source in opened else False:
                                item[kind + " run"] = np.stack([item[kind]] + [((feature(source, f) - mean[b, p]) ** 2) for f, p in near]).astype(np.float16)
                                item["variance run"] = np.stack([variance[b, place]] + [variance[b, p] for _f, p in near]).astype(np.float16)
                    if "revisit" in seen:
                        item["gap"] = ((seen["revisit"] - seen["true"]) ** 2).astype(np.float32)
                    rows.append(item)
            print(f"  {len(rows) // len(AHEAD):,d} moments", flush=True)

    # the floor: per value, half the mean squared difference between a revisit and the true frame (two views, each with its own variation)
    floor = np.mean([r["gap"] for r in rows if r["half"] == 0 and "gap" in r], axis=0) / 2
    print(f"\nfloor from {sum(1 for r in rows if r['half'] == 0 and 'gap' in r):,d} revisits of the first half of the moments: {floor.mean():.3f} per value on average "
          f"(the predicted variance is {np.mean([r['variance'].mean() for r in rows]):.3f} on average)")
    signals = {"mismatch": lambda r, kind: float(r[kind].mean()), "surprise": lambda r, kind: float((r[kind] / r["variance"]).mean()),
               "surprise with floor": lambda r, kind: float((r[kind] / (r["variance"] + floor)).mean())}
    if grid:
        signals["the same on the path only"] = lambda r, kind: float((r[kind] / (r["variance"] + floor))[r["corridor"]].mean())
        signals["3 strongest cells on the path"] = lambda r, kind: float(np.sort((r[kind] / (r["variance"] + floor)).mean(-1)[r["corridor"]])[-3:].mean())
    else:
        signals["the same over 5 frames"] = lambda r, kind: (float((r[kind + " run"].astype(np.float32) / (r["variance run"].astype(np.float32) + floor)).mean())
                                                              if kind + " run" in r else float("nan"))
    if grid:
        print(f"cells on the path: {np.mean([r['corridor'].mean() for r in rows]):.0%} of the grid on average")
    result = {"floor": float(floor.mean())}
    for ahead in AHEAD:
        part = [r for r in rows if r["ahead"] == ahead and r["half"] == 1]
        print(f"\n{ahead / 15:g} s after now ({len(part):,d} moments of the second half)")
        print(f"  {'observed frame':12s} {'moments':>8s}" + "".join(f"   |   {name}: median   told from true (AUC)   above the 5 % level" for name in signals))
        result[f"{ahead / 15:g} s"] = table = {}
        for kind in ("true", "revisit", "other day", "elsewhere"):
            have = [r for r in part if kind in r]
            if not have:
                continue
            line = {"moments": len(have)}
            text = f"  {kind:12s} {len(have):8,d}"
            for name, signal in signals.items():
                value, true = np.array([signal(r, kind) for r in have]), np.array([signal(r, "true") for r in have])
                usable = np.isfinite(value) & np.isfinite(true)  # the 5-frame signal needs neighbours in time
                value, true = value[usable], true[usable]
                everyone = np.array([signal(r, "true") for r in part])
                level = np.quantile(everyone[np.isfinite(everyone)], 0.95)
                both, label = np.concatenate([true, value]), np.concatenate([np.zeros(len(true), bool), np.ones(len(value), bool)])
                line[name] = {"median": float(np.median(value)), "auc": area(both, label) if kind != "true" else 0.5, "above": float((value > level).mean())}
                text += f"   |   {np.median(value):{len(name) + 8}.2f} {line[name]['auc']:22.2f} {line[name]['above']:21.0%}"
            table[kind] = line
            print(text, flush=True)
    target = args.run / ("alarm.json" if args.tight == 1.0 else f"alarm-tight-{args.tight:g}.json")
    target.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {target}", flush=True)


if __name__ == "__main__":
    main()
