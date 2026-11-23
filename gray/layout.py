#!/usr/bin/env python3
"""A top-down map of the room for every recording, to check whether a head path is physically possible.

    python -m gray.layout --out <folder>

NymeriaPlus annotates walls, floors, doors, windows and furniture as 3D boxes (objects/boxy).
STEER's scenebench.scene.nymeria merges the recordings of one location: walls with the door
openings cut out, floors, and the furniture as it stood in each recording. This script draws that
onto a grid of 5 cm cells seen from above, one grid per storey:

    floor       there is an annotated floor under the cell
    wall        a wall passes through the cell between 0.3 and 1.5 m above the floor
    furniture   the number of a piece of furniture in that height band, 0 for none (its kind, name and
                lowest and highest point in the world are stored per number)

and stores, for every cached window, how to get from the window's own frame (the one head.npy and
root.npy are in) to the world frame of the map. ``Layout.flags`` then says for any path given in
the window's frame which of its points are over floor, inside a wall or inside furniture.

The annotation is not exact: wall boxes reach a little into doorways, and some openings have no
door box (looking at the video of such places showed doorways, closet doors and open passages,
never a wall). Where the pelvis of a person in any recording of the location passes through a
wall, a doorway of 0.4 m around that point is opened; under recorded head points only the cell
itself. Furniture is left as it is, since people sit and lean on it.

Some houses have a storey without annotated floor (an attic). Points whose head height fits no
annotated floor are flagged ``unmapped`` and not checked.

Stairs: a map is flat, so on a staircase a path runs "through" walls of the storey above or
below. Every point is therefore looked up on the storey its head height belongs to, and walls
are not checked within 0.5 m of an annotated stairs box (``stairs`` in the flags).

Output: <out>/<sequence id>.npz per recording, <out>/summary.json, pictures of a few maps, and
<out>/windows.npz with, for every cached window, the share of its recorded head path that is over
annotated floor (used to tell indoor from outdoor windows), in a wall, and near stairs.
The run ends with a test on the recorded paths themselves: they must hardly ever cross a wall,
while the same paths turned by 90 or 180 degrees about their start must do so much more often.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gray.features import CACHE
from gray.rectify import SEQUENCES, rotation

OUT = Path("/ptmp/enazaren/gray/layout")
STEER = Path("/u/enazaren/projects/STEER")
CELL = 0.05  # metres
BAND = (0.3, 1.5)  # metres above the floor: what a walking body runs into
STOREY = 0.8  # floors further apart in height than this are different storeys
VIDEO_OFFSET = 0.89  # seconds, as in the feature cache
DOORWAY = 0.4  # metres: where a recorded pelvis path passes through a wall, the wall is opened this far around it
UNMAPPED = 1.3  # metres: a head whose floor would be further than this from every annotated floor is on a storey without a map
STAIRS = 0.5  # metres around an annotated stairs box in which walls are not checked
HEAD = 1.3  # metres: a head is on the storey whose floor is nearest to this far below it
MAX_CELLS = 8000  # a map wider than this many cells (400 m) is not drawn


def nymeria():
    """STEER's reader of the box annotation, pointed at this cluster's copy of the recordings."""
    os.environ.setdefault("STEER_NYMERIA_SEQUENCES", str(SEQUENCES))
    if str(STEER) not in sys.path:
        sys.path.insert(0, str(STEER))
    from scenebench.scene import nymeria as module

    return module


def trajectory(sequence: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Device time (s), position (N, 3) and rotation (N, 3, 3) of the glasses in the world frame (+z up)."""
    times, values = [], []
    with (sequence / "recording_head" / "mps" / "slam" / "closed_loop_trajectory.csv").open() as handle:
        header = handle.readline().strip().split(",")
        columns = [header.index(n) for n in ("tx_world_device", "ty_world_device", "tz_world_device", "qw_world_device",
                                             "qx_world_device", "qy_world_device", "qz_world_device")]
        clock = header.index("tracking_timestamp_us")
        for line in handle:
            parts = line.split(",")
            times.append(float(parts[clock]) / 1e6)
            values.append([float(parts[c]) for c in columns])
    values = np.array(values)
    w, x, y, z = values[:, 3:].T
    turn = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)
    return np.array(times), values[:, :3], turn


def graph_of(sequence: Path) -> str:
    """The id of the world frame of a recording; recordings with the same id share their coordinates."""
    with (sequence / "recording_head" / "mps" / "slam" / "closed_loop_trajectory.csv").open() as handle:
        handle.readline()
        return handle.readline().split(",")[0]


def camera_on_glasses(sequence: Path) -> tuple[np.ndarray, np.ndarray]:
    """Rotation (3, 3) and position (3,) of the RGB camera on the glasses."""
    with (sequence / "recording_head" / "mps" / "slam" / "online_calibration.jsonl").open() as handle:
        record = json.loads(handle.readline())
    for camera in record["CameraCalibrations"]:
        if camera["Label"] == "camera-rgb":
            w, xyz = camera["T_Device_Camera"]["UnitQuaternion"]
            return rotation(w, *xyz), np.array(camera["T_Device_Camera"]["Translation"])
    raise KeyError("camera-rgb")


def frames_of(sequence: Path, folder: Path) -> dict[str, np.ndarray]:
    """For every cached window: ``turn`` (3, 3) and ``shift`` (3,) with  window point = turn @ world point - shift."""
    times, position, turn = trajectory(sequence)
    on_glasses, offset = camera_on_glasses(sequence)
    starts, head = np.load(folder / "starts.npy"), np.load(folder / "head.npy").astype(np.float64)
    turns, shifts, errors, floors = [], [], [], []
    for start, poses in zip(starts, head):
        at = start / 30.0 + times[0] - VIDEO_OFFSET + np.arange(poses.shape[0]) / 15.0
        rows = np.clip(np.searchsorted(times, at), 0, len(times) - 1)
        camera = position[rows] + turn[rows] @ offset
        first, second = poses[0, 3:6], poses[0, 6:9]
        cached = np.stack([first, second, np.cross(first, second)])  # camera to window frame
        to_window = cached @ (turn[rows[0]] @ on_glasses).T
        shift = to_window @ camera[0] - poses[0, :3]
        turns.append(to_window)
        shifts.append(shift)
        errors.append(np.abs(camera @ to_window.T - shift - poses[:, :3]).max())
        floors.append(camera[0, 2] - poses[0, 1])  # the cached height is above the floor
    return {"turn": np.array(turns), "shift": np.array(shifts), "frame_error": np.array(errors), "floor_z": np.array(floors)}


class Grid:
    """Cells of CELL metres seen from above; ``origin`` is the world position of the corner of cell (0, 0)."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.origin = np.floor(low / CELL) * CELL
        self.shape = tuple(int(v) for v in np.ceil((high - self.origin) / CELL).astype(int)[::-1] + 1)  # rows are y, columns are x

    def patch(self, corners: np.ndarray) -> tuple[slice, slice, np.ndarray]:
        """The cells around some world points (N, 2): row and column slices and the cell centres (rows, cols, 2)."""
        low = np.clip(np.floor((corners.min(0) - self.origin) / CELL).astype(int) - 1, 0, None)
        high = np.minimum(np.ceil((corners.max(0) - self.origin) / CELL).astype(int) + 2, self.shape[::-1])
        xs = self.origin[0] + (np.arange(low[0], max(high[0], low[0])) + 0.5) * CELL
        ys = self.origin[1] + (np.arange(low[1], max(high[1], low[1])) + 0.5) * CELL
        centres = np.stack(np.meshgrid(xs, ys), axis=-1)
        return slice(low[1], low[1] + len(ys)), slice(low[0], low[0] + len(xs)), centres


def draw(module, location, recording: str, levels: list[float], walked: list[np.ndarray] | None = None) -> dict[str, np.ndarray] | None:
    """The maps of one recording, one per storey; ``walked``: per storey the world points (N, 2) where people were."""
    things = [t.box for t in location.things(recording) if t.box.category not in module.NO_COLLIDER]
    points = [quad for _uid, _top, quad in location.floors] + [b.corners[:, :2] for b in things]
    for wall, rects in zip(location.walls, location.wall_rects):
        for a0, a1, _z0, _z1 in rects:
            points.append(np.array([wall.offset * wall.normal[:2] + a * wall.along[:2] for a in (a0, a1)]))
    if not points:
        return None
    points = np.concatenate(points)
    grid = Grid(points.min(0) - 0.5, points.max(0) + 0.5)
    if max(grid.shape) > MAX_CELLS:
        return None
    floor = np.zeros((len(levels), *grid.shape), dtype=bool)
    wall = np.zeros_like(floor)
    stairs = np.zeros(grid.shape, dtype=bool)
    for box in location.stairs:
        quad = box.footprint()
        centre = quad.mean(0)
        wide = centre + (quad - centre) * (1 + STAIRS / np.maximum(np.linalg.norm(quad - centre, axis=1, keepdims=True), 1e-6))
        rows, cols, centres = grid.patch(wide)
        stairs[rows, cols] |= module._in_convex(centres.reshape(-1, 2), wide).reshape(centres.shape[:2])
    furniture = np.zeros(floor.shape, dtype=np.uint16)
    ids = sorted({b.uid for b in things})
    number = {uid: i + 1 for i, uid in enumerate(ids)}
    for level, z in enumerate(levels):
        low, high = z + BAND[0], z + BAND[1]
        for _uid, top, quad in location.floors:
            if abs(top - z) < STOREY:
                rows, cols, centres = grid.patch(quad)
                floor[level, rows, cols] |= module._in_convex(centres.reshape(-1, 2), quad).reshape(centres.shape[:2])
        for plane, rects in zip(location.walls, location.wall_rects):
            half = max(plane.thickness, CELL) / 2
            for a0, a1, z0, z1 in rects:
                if z1 <= low or z0 >= high:
                    continue
                ends = np.array([plane.offset * plane.normal[:2] + a * plane.along[:2] + s * half * plane.normal[:2] for a in (a0, a1) for s in (-1, 1)])
                rows, cols, centres = grid.patch(ends)
                across, along = centres @ plane.normal[:2] - plane.offset, centres @ plane.along[:2]
                wall[level, rows, cols] |= (np.abs(across) <= half) & (along >= a0) & (along <= a1)
        heights = np.linspace(low, high, 4)
        solid = [(b, None) for _name, b in location.odd_structure if b.category == "Wall"] + [(b, number[b.uid]) for b in things]
        for box, mark in solid:
            z0, z1 = box.z_range
            if z1 <= low or z0 >= high:
                continue
            rows, cols, centres = grid.patch(box.corners[:, :2])
            flat = centres.reshape(-1, 2)
            inside = np.zeros(len(flat), dtype=bool)
            for h in heights:
                inside |= box.contains(np.concatenate([flat, np.full((len(flat), 1), h)], axis=1))
            inside = inside.reshape(centres.shape[:2])
            if mark is None:
                wall[level, rows, cols] |= inside
            else:
                furniture[level, rows, cols] = np.where(inside, mark, furniture[level, rows, cols])
    opened = 0
    reach = int(round(DOORWAY / CELL))
    disk = [(dy, dx) for dy in range(-reach, reach + 1) for dx in range(-reach, reach + 1) if dy * dy + dx * dx <= reach * reach]
    for level, (heads, hips) in enumerate(walked or []):
        before = int(wall[level].sum())
        solid = wall[level].copy()
        for points, offsets in ((hips, disk), (heads, [(0, 0)])):
            if len(points) == 0:
                continue
            cells = np.unique(np.floor((points - grid.origin) / CELL).astype(np.int64), axis=0)
            cells = cells[(cells[:, 1] >= 0) & (cells[:, 1] < grid.shape[0]) & (cells[:, 0] >= 0) & (cells[:, 0] < grid.shape[1])]
            cells = cells[solid[cells[:, 1], cells[:, 0]]]  # only where the path is inside a wall
            for dy, dx in offsets:
                y, x = cells[:, 1] + dy, cells[:, 0] + dx
                ok = (y >= 0) & (y < grid.shape[0]) & (x >= 0) & (x < grid.shape[1])
                wall[level, y[ok], x[ok]] = False
        opened += before - int(wall[level].sum())
    by_uid = {b.uid: b for b in things}
    return {"origin": grid.origin, "levels": np.array(levels), "floor": floor, "wall": wall, "furniture": furniture, "stairs": stairs,
            "furniture_uid": np.array(ids, dtype=np.int64), "furniture_category": np.array([by_uid[u].category for u in ids]),
            "furniture_name": np.array([by_uid[u].name for u in ids]), "opened_cells": np.array(opened),
            "furniture_low": np.array([by_uid[u].z_range[0] for u in ids], dtype=np.float32), "furniture_high": np.array([by_uid[u].z_range[1] for u in ids], dtype=np.float32)}


class Layout:
    """The stored maps of one recording."""

    def __init__(self, path: Path) -> None:
        self.data = dict(np.load(path))

    def world(self, window: int, points: np.ndarray) -> np.ndarray:
        """Points (N, 3) in the window's frame (x, height, z) -> world (N, 3), +z up."""
        return (points + self.data["shift"][window]) @ self.data["turn"][window]

    def storeys(self, window: int, head: np.ndarray) -> np.ndarray:
        """The storey (N,) of every point of a head path (N, 3) in the window's frame, from the height of the head; -1 where no annotated floor fits."""
        return self.storeys_world(self.world(window, head))

    def storeys_world(self, head: np.ndarray) -> np.ndarray:
        """The same for head positions (N, 3) given in the world frame."""
        gap = np.abs(head[:, 2:3] - HEAD - self.data["levels"][None])
        return np.where(gap.min(1) <= UNMAPPED, gap.argmin(1), -1)

    def flags(self, window: int, points: np.ndarray, storeys: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """For points (N, 3) in the window's frame: over annotated floor, inside a wall, the furniture number (0 for none), near stairs.

        ``storeys``: the storey of every point (from ``storeys`` of the head path); default the storey the window starts on.
        """
        level = np.full(len(points), int(self.data["level"][window])) if storeys is None else storeys
        return self.flags_world(self.world(window, points), level)

    def flags_world(self, points: np.ndarray, storeys: np.ndarray) -> dict[str, np.ndarray]:
        """The same for points (N, 2 or 3) in the world frame, with the storey of every point."""
        d = self.data
        cell = np.floor((points[:, :2] - d["origin"]) / CELL).astype(int)
        mapped = storeys >= 0
        level = np.maximum(storeys, 0)
        rows, cols = d["floor"].shape[1:]
        inside = mapped & (cell[:, 0] >= 0) & (cell[:, 0] < cols) & (cell[:, 1] >= 0) & (cell[:, 1] < rows)
        x, y = np.clip(cell[:, 0], 0, cols - 1), np.clip(cell[:, 1], 0, rows - 1)
        stairs = inside & d["stairs"][y, x]
        return {"floor": inside & d["floor"][level, y, x], "wall": inside & d["wall"][level, y, x] & ~stairs,
                "furniture": np.where(inside, d["furniture"][level, y, x], 0), "stairs": stairs, "unmapped": ~mapped}


def turned(points: np.ndarray, degrees: float) -> np.ndarray:
    """A path (N, 3) in the window's frame turned about its first point (about the vertical)."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    x, z = points[:, 0] - points[0, 0], points[:, 2] - points[0, 2]
    return np.stack([points[0, 0] + c * x - s * z, points[:, 1], points[0, 2] + s * x + c * z], axis=1)


def picture(name: str, layout: Layout, head: np.ndarray, target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = layout.data
    level = int(np.bincount(d["level"]).argmax())
    image = np.full((*d["floor"].shape[1:], 3), 255, dtype=np.uint8)
    image[d["floor"][level]] = (225, 225, 225)
    image[d["furniture"][level] > 0] = (205, 160, 100)
    image[d["wall"][level]] = (20, 20, 20)
    rows, cols = image.shape[:2]
    extent = (d["origin"][0], d["origin"][0] + cols * CELL, d["origin"][1], d["origin"][1] + rows * CELL)
    figure, axis = plt.subplots(figsize=(11, 11 * rows / max(cols, 1) + 0.6))
    axis.imshow(image, origin="lower", extent=extent)
    for window in np.flatnonzero(d["level"] == level)[::2]:
        path = layout.world(window, head[window, ::5, :3].astype(np.float64))
        axis.plot(path[:, 0], path[:, 1], color="tab:blue", linewidth=0.5, alpha=0.6)
    axis.set(title=f"{name}: floor grey, walls black, furniture brown, recorded head paths blue (storey at {d['levels'][level]:.2f} m)", xlabel="m", ylabel="m")
    figure.tight_layout()
    figure.savefig(target / f"{name}.png", dpi=80)
    plt.close(figure)


def group(job: tuple) -> list[dict]:
    """All recordings that share one world frame."""
    recordings, cached, cache, videos, out, wanted, shown, opening = job
    module = nymeria()
    location = module.Location(recordings[0][1], [name for name, _loc in recordings])
    tops = sorted(top for _uid, top, _quad in location.floors)
    levels: list[list[float]] = []
    for top in tops:
        if levels and top - levels[-1][-1] <= STOREY:
            levels[-1].append(top)
        else:
            levels.append([top])
    results = []
    heights = [float(np.median(l)) for l in levels]
    every, failed = {}, {}
    for name in cached:  # first where everybody walked, then the maps
        try:
            every[name] = frames_of(Path(videos) / name, Path(cache) / name)
        except Exception as error:  # noqa: BLE001
            failed[name] = f"{type(error).__name__}: {error}"
    if not heights and every:
        heights = [float(np.median(np.concatenate([f["floor_z"] for f in every.values()])))]
    walked = [([], []) for _ in heights]
    for name, frames in every.items():
        frames["level"] = np.abs(frames["floor_z"][:, None] - np.array(heights)[None]).argmin(1)
        head, root = np.load(Path(cache) / name / "head.npy"), np.load(Path(cache) / name / "root.npy")
        pelvis = np.stack([root[..., 0], np.zeros(root.shape[:2]), root[..., 1]], axis=-1)
        to_world = lambda points: np.einsum("wni,wij->wnj", points + frames["shift"][:, None], frames["turn"])  # noqa: E731
        above = to_world(head[..., :3].astype(np.float64))
        gap = np.abs(above[..., 2:3] - HEAD - np.array(heights)[None, None])  # per point, from the head height
        storey = np.where(gap.min(-1) <= UNMAPPED, gap.argmin(-1), -1)
        hips = to_world(pelvis.astype(np.float64))
        for level in range(len(heights)):
            walked[level][0].append(above[..., :2][storey == level])
            walked[level][1].append(hips[..., :2][storey == level])
    walked = [tuple(np.concatenate(w) if w else np.zeros((0, 2)) for w in pair) for pair in walked]
    for name in cached:
        try:
            if name in failed:
                results.append({"name": name, "skipped": failed[name]})
                continue
            frames = every[name]
            maps = draw(module, location, name, heights, walked if opening else None)
            if maps is None:
                results.append({"name": name, "skipped": "nothing to draw or the map is too large"})
                continue
            np.savez_compressed(Path(out) / f"{name}.tmp.npz", **maps, **frames)  # written whole, then put in place: other jobs may be reading the maps
            (Path(out) / f"{name}.tmp.npz").rename(Path(out) / f"{name}.npz")
            layout = Layout(Path(out) / f"{name}.npz")
            head = np.load(Path(cache) / name / "head.npy")
            root = np.load(Path(cache) / name / "root.npy")
            if name in shown:
                picture(name, layout, head, Path(out))
            category = np.concatenate([["none"], maps["furniture_category"]])
            rows = []
            listed = set(wanted.get(name, range(len(head))))
            for window in range(len(head)):
                path = head[window, ::3, :3].astype(np.float64)
                pelvis = np.stack([root[window, ::3, 0], np.zeros(len(path)), root[window, ::3, 1]], axis=1).astype(np.float64)
                row = {"reach": float(np.hypot(path[:, 0], path[:, 2]).max()), "name": name, "window": window, "listed": window in listed}
                storeys = layout.storeys(window, path)
                row["stairs"] = float(layout.flags(window, path, storeys)["stairs"].mean())
                row["unmapped"] = float((storeys < 0).mean())
                row["storeys"] = int(len(np.unique(storeys[storeys >= 0])))
                for label, points in (("head", path), ("pelvis", pelvis), ("head turned 90", turned(path, 90)), ("head turned 180", turned(path, 180))):
                    f = layout.flags(window, points, storeys)
                    row[label] = (float(f["floor"].mean()), float(f["wall"].mean()), float((f["furniture"] > 0).mean()))
                    if label in ("head", "pelvis"):
                        row[label + " in"] = dict(zip(*np.unique(category[f["furniture"][f["furniture"] > 0]], return_counts=True)))
                rows.append(row)
            results.append({"name": name, "location": recordings[0][1], "windows": len(head), "storeys": len(heights), "cells": list(maps["floor"].shape),
                            "furniture": len(maps["furniture_uid"]), "floor_cells": int(maps["floor"].sum()), "wall_cells": int(maps["wall"].sum()), "opened_cells": int(maps["opened_cells"]),
                            "frame_error_cm": float(frames["frame_error"].max() * 100), "upright": float(frames["turn"][:, 1, 2].mean()), "rows": rows})
        except Exception as error:  # noqa: BLE001
            results.append({"name": name, "skipped": f"{type(error).__name__}: {error}"})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--subset", type=Path, default=Path("/ptmp/enazaren/gray/runs/census/navigation.npz"), help="the windows the test is run on")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="only the first groups of recordings (a test)")
    parser.add_argument("--keep-walls", action="store_true", help="do not open the wall cells where people walked (to study the annotation)")
    parser.add_argument("--show", nargs="*", default=["20230621_s1_kevin_davis_act3_bw8qui", "20230703_s0_tommy_ayala_act0_c07rbi"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    annotated = sorted(p.parents[2].name for p in args.videos.glob("*/objects/boxy/scene_objects.csv"))
    cached = {p.parent.name for p in args.cache.glob("*/head.npy")}
    groups = defaultdict(list)
    for name in annotated:
        location = json.loads((args.videos / name / "metadata.json").read_text())["location"]
        groups[(location, graph_of(args.videos / name))].append((name, location))
    wanted = defaultdict(list)
    if args.subset.exists():
        listed = np.load(args.subset)
        for name, window in zip(listed["sequence"].tolist(), listed["window"].tolist()):
            wanted[name].append(window)
    keys = sorted(groups)[: args.limit]
    extra = [name for key in keys[:: max(1, len(keys) // 6)] for name, _l in groups[key][:1] if name in cached]
    shown = set(args.show) | set(extra)
    jobs = [(groups[key], [n for n, _l in groups[key] if n in cached], str(args.cache), str(args.videos), str(args.out),
             {n: wanted[n] for n, _l in groups[key] if n in wanted}, shown, not args.keep_walls) for key in keys]
    print(f"{len(annotated)} annotated recordings in {len(groups)} world frames ({len({k[0] for k in groups})} locations); "
          f"{sum(len(j[1]) for j in jobs)} of them are in the cache", flush=True)
    with Pool(args.workers) as pool:
        results = [r for part in pool.imap_unordered(group, sorted(jobs, key=lambda j: -len(j[1]))) for r in part]

    done = [r for r in results if "rows" in r]
    skipped = [r for r in results if "skipped" in r]
    every = [row for r in done for row in r["rows"]]
    np.savez_compressed(args.out / "windows.npz", sequence=np.array([row["name"] for row in every]), window=np.array([row["window"] for row in every]),
                        floor=np.array([row["head"][0] for row in every], dtype=np.float32), wall=np.array([row["head"][1] for row in every], dtype=np.float32),
                        stairs=np.array([row["stairs"] for row in every], dtype=np.float32), unmapped=np.array([row["unmapped"] for row in every], dtype=np.float32), storeys=np.array([row["storeys"] for row in every]))
    rows = [row for row in every if row["listed"]]
    print(f"\nmaps for {len(done)} recordings, {sum(r['windows'] for r in done):,d} windows; {len(skipped)} recordings skipped")
    for r in skipped[:12]:
        print(f"  skipped {r['name']}: {r['skipped']}")
    errors = np.array([r["frame_error_cm"] for r in done])
    print(f"window frame against the head path read from the recording: largest difference {np.median(errors):.1f} cm (median recording), "
          f"{errors.max():.1f} cm (worst); the window's up axis is the world's up axis to {np.mean([r['upright'] for r in done]):.4f}")
    print(f"storeys per recording: {dict(zip(*np.unique([r['storeys'] for r in done], return_counts=True)))}; "
          f"recordings without any annotated floor: {sum(r['floor_cells'] == 0 for r in done)}")

    opened, left = sum(r["opened_cells"] for r in done), sum(r["wall_cells"] for r in done)
    print(f"wall cells opened because somebody walked there: {opened:,d} of {opened + left:,d} ({opened / max(opened + left, 1):.1%})")
    print(f"\ntest on {len(rows):,d} windows of the navigation subset (points of each path, every 0.2 s); "
          f"{np.mean([row['stairs'] > 0 for row in rows]):.1%} of the windows come near annotated stairs, {np.mean([row['storeys'] > 1 for row in rows]):.1%} change storey, "
          f"{np.mean([row['unmapped'] > 0 for row in rows]):.1%} are partly on a storey without a map")
    reach = np.array([row["reach"] for row in rows])
    summary = {}
    for title, mask in (("all windows", np.ones(len(rows), dtype=bool)), ("windows that get 1 m or further", reach >= 1.0), ("windows that get 2 m or further", reach >= 2.0)):
        print(f"  {title} ({mask.sum():,d})\n    path                  over floor   in a wall   in furniture   windows touching a wall")
        summary[title] = {}
        for label in ("head", "pelvis", "head turned 90", "head turned 180"):
            values = np.array([row[label] for row in rows])[mask]
            summary[title][label] = {"floor": float(values[:, 0].mean()), "wall": float(values[:, 1].mean()), "furniture": float(values[:, 2].mean()),
                                     "windows_touching_wall": float((values[:, 1] > 0).mean())}
            print(f"    {label:20s} {values[:, 0].mean():10.1%} {values[:, 1].mean():11.2%} {values[:, 2].mean():14.1%} {(values[:, 1] > 0).mean():14.1%}")
    points = 40 * len(rows)
    for label in ("head", "pelvis"):
        counts = defaultdict(int)
        for row in rows:
            for category, count in row[label + " in"].items():
                counts[str(category)] += int(count)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:14]
        summary[label + " furniture"] = {k: v / points for k, v in top}
        print(f"\n  recorded {label} points inside furniture, by kind (share of all points): " + ", ".join(f"{k} {v / points:.1%}" for k, v in top))

    for r in done:
        del r["rows"]
    (args.out / "summary.json").write_text(json.dumps({"recordings": done, "skipped": skipped, "test": summary}, indent=1))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
