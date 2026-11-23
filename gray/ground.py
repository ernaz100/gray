#!/usr/bin/env python3
"""A truth about free and blocked space that does not rest on the furniture boxes.

    python -m gray.ground --out <folder>

The furniture boxes of the room maps (gray/layout.py) are too rough to judge at close range
(gray/cloud.py). Two things are known far better, per location, from all its recordings together:

    free      where somebody walked. Every cell within 0.2 m of a recorded pelvis position at which the pelvis moved
              faster than 0.4 m/s and the head was more than 1.2 m above the floor, in any recording of the location,
              is free: people do not walk through things. (Not every place a head was, and not every place a head
              moved fast: heads are over sofas when sitting and sweep over beds and counters when leaning.)
    blocked   the walls of the room map (the part of the annotation that held up), and what Aria's SLAM
              measured in the same place in many recordings: a cell at body height that holds measured points
              in at least 30 % of the location's recordings (and at least 3) is something that stands
              there. Points on people do not repeat in the same cell across recordings, except along the
              paths everybody takes, and those cells are free by the first rule.

Everything else is unknown: free space nobody entered, or furniture the points missed.

The well-measured points of each recording are read once and kept in <out>/points/<id>.npy.
Per location this writes <out>/<location>.npz on one 5 cm grid for all its recordings:
origin, levels, free, blocked, wall, steady (the measured part of blocked), recordings.

``Ground`` reads it. For a camera position and view direction it gives, along 16 directions,

    walkable   the distance until the free cells end        (never too long: what is known to be free)
    blocked    the distance to the first blocked cell       (never too short, except for what the points missed)

The true free distance lies between the two. Where they agree within 0.3 m the truth is known
to that accuracy ("sure"); features are compared on those directions (gray/probe.py).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

from gray.cloud import MAX_DISTANCE, MAX_INVERSE
from gray.layout import BAND, CELL, HEAD, UNMAPPED
from gray.rectify import SEQUENCES

OUT = Path("/ptmp/enazaren/gray/ground")
BODY = 0.2  # metres around a walked position that are free
WALKING = 0.4  # metres per second
UPRIGHT = 1.2  # metres of head height above the floor
SHARE, AT_LEAST, POINTS = 0.3, 3, 3  # a cell is steady if at least this share of the recordings (and this many) have this many points in it
RAYS, FAR, STEP = 16, 5.0, 0.05


def measured(job: tuple[str, str, str]) -> np.ndarray:
    """The well-measured points (N, 3) of one recording; read from the recording once and kept in <out>/points/<id>.npy."""
    name, videos, out = job
    kept = Path(out) / "points" / f"{name}.npy"
    if kept.exists():
        return np.load(kept)
    with gzip.open(Path(videos) / name / "recording_head" / "mps" / "slam" / "semidense_points.csv.gz", "rt") as handle:
        points = np.loadtxt(handle, delimiter=",", skiprows=1, usecols=(2, 3, 4, 5, 6), dtype=np.float32)
    points = points[(points[:, 3] < MAX_INVERSE) & (points[:, 4] < MAX_DISTANCE), :3]
    kept.parent.mkdir(parents=True, exist_ok=True)
    np.save(kept.with_suffix(".tmp.npy"), points)
    kept.with_suffix(".tmp.npy").rename(kept)
    return points


def fetch(job: tuple[str, str, str]) -> int:
    return len(measured(job))


def location(job: tuple) -> dict:
    name, recordings, maps, tracks, videos, out, redo = job
    target = Path(out) / f"{name}.npz"
    if target.exists() and not redo:
        return {"location": name, "skipped": True}
    try:
        grids = {r: np.load(Path(maps) / f"{r}.npz") for r in recordings}
        levels = grids[recordings[0]]["levels"]
        low = np.min([g["origin"] for g in grids.values()], axis=0)
        high = np.max([g["origin"] + np.array(g["floor"].shape[:0:-1]) * CELL for g in grids.values()], axis=0)
        cols, rows = (int(v) for v in np.ceil((high - low) / CELL).astype(int) + 1)
        shape = (len(levels), rows, cols)
        visited, wall, seen = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.uint16)
        for r in recordings:
            g = grids[r]
            dx, dy = (int(v) for v in np.round((g["origin"] - low) / CELL))
            h, w = g["wall"].shape[1:]
            wall[:, dy : dy + h, dx : dx + w] |= g["wall"]
            # the pelvis and the head of the cached windows (their first 2 s each, so nothing is counted twice), in the world frame
            pelvis, above = np.load(Path(tracks) / r / "root.npy")[:, :35].astype(np.float64), np.load(Path(tracks) / r / "head.npy")[:, :35, :3].astype(np.float64)
            to_world_frame = lambda points: np.einsum("wni,wij->wnj", points + g["shift"][:, None], g["turn"])  # noqa: E731
            hips = to_world_frame(np.stack([pelvis[..., 0], np.zeros(pelvis.shape[:2]), pelvis[..., 1]], axis=-1))
            head = to_world_frame(above)
            step = np.linalg.norm(np.diff(hips[..., :2], axis=1), axis=-1) * 15  # (W, 34) metres per second
            pace = np.stack([np.convolve(row, np.ones(5) / 5, mode="same") for row in step])[:, :30]
            walking = (pace > WALKING) & (above[:, :30, 1] > UPRIGHT)
            head, hips = head[:, :30][walking], hips[:, :30][walking]
            gap = np.abs(head[:, 2:3] - HEAD - levels[None])
            storey = np.where(gap.min(1) <= UNMAPPED, gap.argmin(1), -1)
            cell = np.floor((hips[:, :2] - low) / CELL).astype(int)
            ok = (storey >= 0) & (cell[:, 0] >= 0) & (cell[:, 0] < cols) & (cell[:, 1] >= 0) & (cell[:, 1] < rows)
            visited[storey[ok], cell[ok, 1], cell[ok, 0]] = True
            points = measured((r, videos, out))
            cell = np.floor((points[:, :2] - low) / CELL).astype(int)
            inside = (cell[:, 0] >= 0) & (cell[:, 0] < cols) & (cell[:, 1] >= 0) & (cell[:, 1] < rows)
            for level, z in enumerate(levels):
                band = inside & (points[:, 2] > z + BAND[0]) & (points[:, 2] < z + BAND[1])
                count = np.zeros((rows, cols), dtype=np.uint16)
                np.add.at(count, (cell[band, 1], cell[band, 0]), 1)
                seen[level] += count >= POINTS
        free = np.stack([ndimage.distance_transform_edt(~v) * CELL <= BODY if v.any() else v for v in visited])
        steady = (seen >= max(AT_LEAST, math.ceil(SHARE * len(recordings)))) & ~free
        blocked = (wall | steady) & ~free
        np.savez_compressed(target.with_suffix(".tmp.npz"), origin=low, levels=levels, free=free, blocked=blocked, wall=wall & ~free, steady=steady,
                            recordings=np.array(recordings))
        target.with_suffix(".tmp.npz").rename(target)
        return {"location": name, "recordings": len(recordings), "free": int(free.sum()), "wall": int((wall & ~free).sum()), "steady": int(steady.sum()),
                "wall_walked": int((wall & free).sum()), "cells": int(np.prod(shape))}
    except Exception as error:  # noqa: BLE001
        return {"location": name, "failed": f"{type(error).__name__}: {error}"}


class Ground:
    """The free and blocked cells of one location."""

    def __init__(self, path: Path) -> None:
        self.data = dict(np.load(path))

    def flags(self, head: np.ndarray) -> dict[str, np.ndarray]:
        """For world positions (N, 3) of a head: is the cell under it free (somebody walked there), blocked, or neither (off the map counts as neither)."""
        d = self.data
        gap = np.abs(head[:, 2:3] - HEAD - d["levels"][None])
        storey = np.where(gap.min(1) <= UNMAPPED, gap.argmin(1), -1)
        cell = np.floor((head[:, :2] - d["origin"]) / CELL).astype(int)
        rows, cols = d["free"].shape[1:]
        inside = (storey >= 0) & (cell[:, 0] >= 0) & (cell[:, 0] < cols) & (cell[:, 1] >= 0) & (cell[:, 1] < rows)
        x, y, level = np.clip(cell[:, 0], 0, cols - 1), np.clip(cell[:, 1], 0, rows - 1), np.maximum(storey, 0)
        return {"free": inside & d["free"][level, y, x], "blocked": inside & d["blocked"][level, y, x]}

    def distances(self, position: np.ndarray, ahead: np.ndarray) -> dict[str, np.ndarray]:
        """For world positions (F, 3) of a head and the world angle (F,) it looks along: walkable and blocked (F, 16) in metres, nan off the map."""
        d = self.data
        gap = np.abs(position[:, 2:3] - HEAD - d["levels"][None])
        storey = np.where(gap.min(1) <= UNMAPPED, gap.argmin(1), -1)
        angle = ahead[:, None] + np.arange(RAYS)[None] * 2 * math.pi / RAYS
        reach = np.arange(STEP, FAR + STEP / 2, STEP)
        points = position[:, None, None, :2] + reach[None, None, :, None] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)[:, :, None, :]
        cell = np.floor((points - d["origin"]) / CELL).astype(int)
        rows, cols = d["free"].shape[1:]
        inside = (cell[..., 0] >= 0) & (cell[..., 0] < cols) & (cell[..., 1] >= 0) & (cell[..., 1] < rows)
        x, y = np.clip(cell[..., 0], 0, cols - 1), np.clip(cell[..., 1], 0, rows - 1)
        level = np.maximum(storey, 0)[:, None, None]
        out = {}
        for key, stop in (("walkable", ~(inside & d["free"][level, y, x])), ("blocked", inside & d["blocked"][level, y, x])):
            out[key] = np.where(stop.any(-1), reach[stop.argmax(-1)], FAR)
            out[key][storey < 0] = np.nan
        return out


def picture(path: Path, boxes: Path | None) -> None:
    """<location>.png next to the file: free green, walls black, repeatedly measured red, unknown white; the outlines of furniture boxes of one recording in blue."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(path)
    level = int(d["free"].reshape(len(d["levels"]), -1).sum(1).argmax())
    image = np.full((*d["free"].shape[1:], 3), 255, dtype=np.uint8)
    image[d["free"][level]] = (190, 230, 190)
    image[d["steady"][level]] = (215, 60, 50)
    image[d["wall"][level]] = (15, 15, 15)
    rows, cols = image.shape[:2]
    extent = (d["origin"][0], d["origin"][0] + cols * CELL, d["origin"][1], d["origin"][1] + rows * CELL)
    figure, axis = plt.subplots(figsize=(13, 13 * rows / cols + 0.6))
    axis.imshow(image, origin="lower", extent=extent, interpolation="nearest")
    if boxes is not None:
        g = np.load(boxes / f"{str(d['recordings'][0])}.npz")
        furniture = g["furniture"][min(level, len(g["furniture"]) - 1)] > 0
        h, w = furniture.shape
        axis.contour(np.arange(w) * CELL + g["origin"][0] + CELL / 2, np.arange(h) * CELL + g["origin"][1] + CELL / 2, furniture.astype(float), levels=[0.5], colors="tab:blue", linewidths=0.8)
    axis.set(title=f"{path.stem}, storey at {d['levels'][level]:+.1f} m, {len(d['recordings'])} recordings: free green, walls black, measured in many recordings red, "
                   f"unknown white; blue: furniture boxes of one recording", xlabel="m")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=80)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/nymeria/egoprimo_cache_all"), help="the feature cache: pelvis and head per window")
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--redo", action="store_true", help="build locations again that already exist")
    parser.add_argument("--show", nargs="*", default=[], help="only draw these locations (already built)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.show:
        for name in args.show:
            picture(args.out / f"{name}.npz", args.maps)
        return
    groups = defaultdict(list)
    for r in json.loads((args.maps / "summary.json").read_text())["recordings"]:
        groups[r["location"]].append(r["name"])
    names = sorted(groups, key=lambda n: -len(groups[n]))[args.part :: args.parts]  # the largest first
    jobs = [(n, sorted(groups[n]), str(args.maps), str(args.tracks), str(args.videos), str(args.out), args.redo) for n in names]
    todo = [(r, str(args.videos), str(args.out)) for n in names if args.redo or not (args.out / f"{n}.npz").exists() for r in groups[n]]
    with Pool(args.workers) as pool:  # reading the points is the slow part: one recording per process first, kept on disk
        total = sum(pool.imap_unordered(fetch, todo, chunksize=1))
    print(f"measured points of {len(todo)} recordings ready ({total / 1e6:.0f} M)", flush=True)
    with Pool(max(1, args.workers // 3)) as pool:
        for r in pool.imap_unordered(location, jobs, chunksize=1):
            if "failed" in r:
                print(f"failed {r['location']}: {r['failed']}", flush=True)
            elif "skipped" not in r:
                print(f"{r['location']}: {r['recordings']} recordings; free {r['free'] * CELL**2:.0f} m2, walls {r['wall']:,d} cells, measured in many recordings {r['steady']:,d} cells; "
                      f"wall cells that somebody was in (not counted as blocked) {r['wall_walked']:,d}", flush=True)
    print(f"{len(list(args.out.glob('Loc_*.npz')))} locations in {args.out}", flush=True)


if __name__ == "__main__":
    main()
