#!/usr/bin/env python3
"""How far can the room maps be trusted? A comparison with the points Aria's SLAM measured.

    python -m gray.cloud maps --out <folder>
    python -m gray.cloud depth --names depth-base depth-large      # how accurate is the estimated depth? (below)

The room maps (gray/layout.py) come from annotated boxes. A box around a table encloses the air
between its legs, a box around a chair the gap behind it. Every recording also comes with the
"semidense" point cloud of Aria's SLAM (recording_head/mps/slam/semidense_points.csv.gz): 3D
points that were really measured, in the same world frame as the maps. They sit on edges and
texture, so plain walls have few of them, but where there is a point there is something.

For each recording of the held-out locations the well-measured points (the usual limits on their
uncertainty) between 0.3 and 1.5 m above each floor are counted on the map's own 5 cm grid; a
cell with ``--min-points`` or more is "occupied by points". The cameras also see the wearer's own
arms and hands and SLAM measures points on them, so cells within ``--body`` metres of the recorded
head path are cleared. Stored as <out>/<id>.npz (count and occupied, per storey and cell). Then, only where the head came within 3 m:

    map against points   of the cells on the outline of furniture boxes, and of the wall cells: how many have a point within 10, 20, 30 cm
    points against map   of the occupied cells: how many are further than 20 cm from any wall or furniture of the map (things the map does not have)
    walking              of the recorded head positions while moving faster than 0.5 m/s: how many are inside a furniture box, and how many in a cell occupied by points.
                         People do not walk through things, so the lower number belongs to the better description of the room
    distances            for the frames of the test bed (gray/probe.py): the distance to the nearest obstacle in the camera's view,
                         once from the map and once from the points; how far the two are apart, by how near the obstacle is

**depth** measures a depth model against the same points, without the maps. For the frames of the
test bed (gray/probe.py) that have the model's depth, the well-measured points are projected into
the image. In each of the 28 x 28 cells of the stored depth that at least five points fall into,
the measured depth is the near end of those points (their 10th percentile: points further back
may be on a surface behind the one the camera sees). Points nearer than 0.5 m are left out (the
wearer's arms). Reported: the factor between estimated and measured depth, and the error that is
left after correcting it once for all frames, once per recording, and once per frame.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

from gray.features import CACHE
from gray.layout import BAND, CELL, Layout
from gray.moments import held_locations
from gray.pinhole import VIEW
from gray.probe import FAR, IN_VIEW, SET, distances
from gray.rectify import SEQUENCES
from gray.tracks import to_world

MAX_INVERSE, MAX_DISTANCE = 0.005, 0.05  # the limits Project Aria recommends for inv_dist_std and dist_std
NEAR_TRACK = 3.0  # metres


def one(job: tuple) -> dict | None:
    name, videos, maps, tracks, probeset, out, min_points, body = job
    try:
        layout = Layout(Path(maps) / f"{name}.npz")
        d = layout.data
        with gzip.open(Path(videos) / name / "recording_head" / "mps" / "slam" / "semidense_points.csv.gz", "rt") as handle:
            points = np.loadtxt(handle, delimiter=",", skiprows=1, usecols=(2, 3, 4, 5, 6), dtype=np.float32)
        total = len(points)
        points = points[(points[:, 3] < MAX_INVERSE) & (points[:, 4] < MAX_DISTANCE), :3]
        rows, cols = d["floor"].shape[1:]
        count = np.zeros((len(d["levels"]), rows, cols), dtype=np.uint16)
        cell = np.floor((points[:, :2] - d["origin"]) / CELL).astype(int)
        inside = (cell[:, 0] >= 0) & (cell[:, 0] < cols) & (cell[:, 1] >= 0) & (cell[:, 1] < rows)
        for level, z in enumerate(d["levels"]):
            band = inside & (points[:, 2] > z + BAND[0]) & (points[:, 2] < z + BAND[1])
            np.add.at(count[level], (cell[band, 1], cell[band, 0]), 1)
        occupied = count >= min_points

        track = np.load(Path(tracks) / f"{name}.npz")
        head = to_world(track["pose"][:, :3].astype(np.float64), track["floor"])[track["valid"]]
        storey = layout.storeys_world(head)
        speed = np.linalg.norm(np.diff(head[:, :2], axis=0, prepend=head[:1, :2]), axis=1) * 15
        at = np.floor((head[:, :2] - d["origin"]) / CELL).astype(int)
        on_map = (storey >= 0) & (at[:, 0] >= 0) & (at[:, 0] < cols) & (at[:, 1] >= 0) & (at[:, 1] < rows)
        # The cameras also see the wearer's own arms and hands, and SLAM measures points on them. They lie where the
        # person was, so everything within ``body`` metres of a recorded head position is cleared.
        raw = int(occupied.sum())
        for level in range(len(d["levels"])):
            visited = np.zeros((rows, cols), dtype=bool)
            here = on_map & (storey == level)
            visited[at[here, 1], at[here, 0]] = True
            if visited.any():
                occupied[level] &= ndimage.distance_transform_edt(~visited) * CELL > body
        np.savez_compressed(Path(out) / f"{name}.npz", count=count, occupied=occupied)
        result = {"name": name, "points": total, "kept": int(len(points)), "occupied_cells": int(occupied.sum()), "occupied_raw": raw}
        sums = {key: 0 for key in ("outline", "outline10", "outline20", "outline30", "wall", "wall10", "wall20", "wall30", "occupied", "unexplained")}
        for level in range(len(d["levels"])):
            visited = np.zeros((rows, cols), dtype=bool)
            here = on_map & (storey == level)
            visited[at[here, 1], at[here, 0]] = True
            if not visited.any():
                continue
            near = ndimage.distance_transform_edt(~visited) * CELL <= NEAR_TRACK
            to_point = ndimage.distance_transform_edt(~occupied[level]) * CELL if occupied[level].any() else np.full((rows, cols), np.inf)
            furniture = d["furniture"][level] > 0
            outline = furniture & ~ndimage.binary_erosion(furniture) & near
            wall = d["wall"][level] & near
            solid = d["wall"][level] | furniture
            to_map = ndimage.distance_transform_edt(~solid) * CELL if solid.any() else np.full((rows, cols), np.inf)
            sums["outline"] += int(outline.sum())
            sums["wall"] += int(wall.sum())
            for limit in (10, 20, 30):
                sums[f"outline{limit}"] += int((to_point[outline] <= limit / 100).sum())
                sums[f"wall{limit}"] += int((to_point[wall] <= limit / 100).sum())
            sums["occupied"] += int((occupied[level] & near).sum())
            sums["unexplained"] += int((occupied[level] & near & (to_map > 0.2)).sum())
        result |= sums
        moving = on_map & (speed > 0.5)
        result["moving"] = int(moving.sum())
        result["in_box"] = int((d["furniture"][storey[moving], at[moving, 1], at[moving, 0]] > 0).sum())
        result["in_points"] = int(occupied[storey[moving], at[moving, 1], at[moving, 0]].sum())
        wide = np.stack([ndimage.binary_dilation(o) for o in occupied])
        result["in_points_wide"] = int(wide[storey[moving], at[moving, 1], at[moving, 0]].sum())

        frames = Path(probeset) / f"{name}.npz"
        if frames.exists():
            index = np.load(frames)["frame"]
            truth = np.load(frames.with_name(f"{name}.truth.npz"))
            pose = track["pose"][index]
            position = to_world(pose[:, :3].astype(np.float64), track["floor"][index])
            import torch

            from gray import frames as anchoring

            yaw = anchoring.heading(torch.from_numpy(pose[:, 3:])).numpy()
            ahead = np.arctan2(-np.cos(yaw), -np.sin(yaw))
            pointed = Layout.__new__(Layout)  # the same grid, with the points as the only obstacles
            pointed.data = dict(d) | {"wall": wide, "furniture": np.zeros_like(d["furniture"])}
            result["from_points"] = distances(pointed, position, ahead, False)[:, IN_VIEW].astype(np.float32)
            result["from_map"] = truth["all"][:, IN_VIEW]
            result["from_walls"] = truth["walls"][:, IN_VIEW]
        return result
    except Exception as error:  # noqa: BLE001
        return {"name": name, "failed": f"{type(error).__name__}: {error}"}


def against(job: tuple) -> dict | None:
    """Estimated against measured depth for the test-bed frames of one recording: pairs (estimated, measured, frame number)."""
    name, videos, tracks, probeset, names = job
    try:
        frames = Path(probeset) / f"{name}.npz"
        index = np.load(frames)["frame"]
        estimated = {n: np.load(frames.with_name(f"{name}.{n}.npy")).astype(np.float32).reshape(len(index), 28, 28) for n in names}
        with gzip.open(Path(videos) / name / "recording_head" / "mps" / "slam" / "semidense_points.csv.gz", "rt") as handle:
            points = np.loadtxt(handle, delimiter=",", skiprows=1, usecols=(2, 3, 4, 5, 6), dtype=np.float32)
        points = points[(points[:, 3] < MAX_INVERSE) & (points[:, 4] < MAX_DISTANCE), :3].astype(np.float64)
        track = np.load(Path(tracks) / f"{name}.npz")
        focal = 14 / math.tan(math.radians(VIEW) / 2)  # in cells of the 28 x 28 depth
        pairs = {n: [] for n in names}
        for i, frame in enumerate(index):
            pose = track["pose"][frame].astype(np.float64)
            camera = to_world(pose[:3], float(track["floor"][frame]))
            first, second = pose[3:6], pose[6:9]
            to_ground = np.stack([first, second, np.cross(first, second)])  # sensor axes -> ground frame (x, up, z)
            near = points[np.abs(points - camera).max(1) < 8.0] - camera
            ground = np.stack([near[:, 0], near[:, 2], -near[:, 1]], axis=1)  # world (x, y, z up) -> ground (x, up, z)
            sensor = ground @ to_ground  # x is down in the image, y is left, z looks ahead
            depth = sensor[:, 2]
            ok = depth > 0.5
            column, row = 13.5 - focal * sensor[ok, 1] / depth[ok], 13.5 + focal * sensor[ok, 0] / depth[ok]
            depth = depth[ok]
            inside = (column > -0.5) & (column < 27.5) & (row > -0.5) & (row < 27.5)
            cell = np.round(row[inside]).astype(int) * 28 + np.round(column[inside]).astype(int)
            depth = depth[inside]
            order = np.lexsort((depth, cell))
            cell, depth = cell[order], depth[order]
            which, start, count = np.unique(cell, return_index=True, return_counts=True)
            enough = count >= 5
            measured = depth[start[enough] + (0.1 * count[enough]).astype(int)]
            # a depth model may mean the distance along the optical axis (as measured here) or along the line of sight of each pixel
            slant = np.sqrt(1 + ((which[enough] % 28 - 13.5) / focal) ** 2 + ((which[enough] // 28 - 13.5) / focal) ** 2)
            for n in names:
                pairs[n].append(np.stack([estimated[n][i].ravel()[which[enough]], measured, np.full(enough.sum(), i), slant], axis=1))
        return {"name": name, "pairs": {n: np.concatenate(v).astype(np.float32) for n, v in pairs.items()}}
    except Exception as error:  # noqa: BLE001
        return {"name": name, "failed": f"{type(error).__name__}: {error}"}


def depth_check(args, names: list[str]) -> None:
    jobs = [(n, str(args.videos), str(args.tracks), str(args.probeset), args.names) for n in names
            if all((args.probeset / f"{n}.{d}.npy").exists() for d in args.names)]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(against, jobs, chunksize=1))
    for r in results:
        if "failed" in r:
            print(f"failed {r['name']}: {r['failed']}")
    done = [r for r in results if "failed" not in r]
    summary = {}
    for name, along_sight in [(n, flag) for n in args.names for flag in (False, True)]:
        ratios, table = [], []
        for r in done:
            pair = r["pairs"][name].copy()
            if along_sight:
                pair[:, 1] *= pair[:, 3]
            for frame in np.unique(pair[:, 2]):
                one_frame = pair[pair[:, 2] == frame]
                if len(one_frame) >= 20:
                    table.append((one_frame[:, 0], one_frame[:, 1], np.median(one_frame[:, 1] / one_frame[:, 0]), r["name"]))
        estimated, measured = np.concatenate([t[0] for t in table]), np.concatenate([t[1] for t in table])
        overall = float(np.median(measured / estimated))
        by_recording = {n: float(np.median(np.concatenate([t[1] / t[0] for t in table if t[3] == n]))) for n in {t[3] for t in table}}
        per_frame = np.array([t[2] for t in table])
        fixes = {"one factor for all": np.concatenate([np.full(len(t[0]), overall) for t in table]),
                 "one factor per recording": np.concatenate([np.full(len(t[0]), by_recording[t[3]]) for t in table]),
                 "one factor per frame": np.concatenate([np.full(len(t[0]), t[2]) for t in table])}
        name = name + (", read as distance along the line of sight" if along_sight else ", read as distance along the optical axis")
        print(f"\n{name}: {len(table):,d} frames of {len(done)} recordings, {len(measured):,d} cells with a measured depth (median {np.median(measured):.2f} m)")
        print(f"  measured / estimated: {overall:.2f} over all; per recording {np.quantile(list(by_recording.values()), 0.1):.2f} to {np.quantile(list(by_recording.values()), 0.9):.2f} "
              f"(10 % to 90 %); per frame {np.quantile(per_frame, 0.1):.2f} to {np.quantile(per_frame, 0.9):.2f}")
        print(f"  {'error after correcting with':28s} median, all   0.5 to 1 m   1 to 2 m   2 to 4 m   4 m and more   within 10 %   within 25 %")
        summary[name] = {"factor": overall, "rows": {}}
        for label, factor in fixes.items():
            error = np.abs(estimated * factor - measured)
            bands = [np.median(error[(measured >= lo) & (measured < hi)]) for lo, hi in ((0.5, 1), (1, 2), (2, 4), (4, 99))]
            relative = error / measured
            summary[name]["rows"][label] = {"median_m": float(np.median(error)), "bands_m": [float(b) for b in bands],
                                            "within_10": float((relative < 0.10).mean()), "within_25": float((relative < 0.25).mean())}
            print(f"  {label:28s} {np.median(error):11.2f} " + "".join(f"{b:11.2f}" for b in bands) + f"{(relative < 0.10).mean():14.0%} {(relative < 0.25).mean():13.0%}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "depth.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out / 'depth.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("maps", "depth"))
    parser.add_argument("--names", nargs="+", default=["depth-large"], help="depth: the depth candidates of gray/probe.py")
    parser.add_argument("--out", type=Path, default=Path("/ptmp/enazaren/gray/cloud"))
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--probeset", type=Path, default=SET)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--body", type=float, default=0.3, help="metres around the recorded head path that are cleared of points (the wearer's own body)")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="every recording that has a map, not only the held-out locations")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    location = {m["sequence_id"]: m["location"] for m in json.loads((args.cache / "index.json").read_text())["sequences"]}
    held = held_locations(args.cache)
    names = sorted(p.stem for p in args.maps.glob("*.npz") if p.stem in location and (args.all or location[p.stem] in held))[: args.limit]
    if args.step == "depth":
        depth_check(args, names)
        return
    jobs = [(n, str(args.videos), str(args.maps), str(args.tracks), str(args.probeset), str(args.out), args.min_points, args.body) for n in names]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(one, jobs, chunksize=1))
    for r in results:
        if "failed" in r:
            print(f"failed {r['name']}: {r['failed']}")
    done = [r for r in results if "failed" not in r]
    total = lambda key: sum(r[key] for r in done)  # noqa: E731
    print(f"{len(done)} recordings; {total('points') / 1e6:.0f} M points, {total('kept') / total('points'):.0%} well measured; "
          f"{total('occupied_raw'):,d} cells with {args.min_points} or more points at body height, {total('occupied_cells'):,d} after clearing {args.body:g} m around the head path")
    print(f"\nmap against points (within {NEAR_TRACK:g} m of where the head was): share that has a measured point within 10, 20, 30 cm")
    print(f"  outline of furniture boxes ({total('outline'):,d} cells)   {total('outline10') / total('outline'):.0%}  {total('outline20') / total('outline'):.0%}  {total('outline30') / total('outline'):.0%}")
    print(f"  walls ({total('wall'):,d} cells)                        {total('wall10') / total('wall'):.0%}  {total('wall20') / total('wall'):.0%}  {total('wall30') / total('wall'):.0%}")
    print(f"points against map: {total('unexplained') / total('occupied'):.0%} of the {total('occupied'):,d} cells occupied by points are further than 20 cm from any wall or furniture of the map")
    print(f"\nwalking: of {total('moving'):,d} recorded head positions moving faster than 0.5 m/s, {total('in_box') / total('moving'):.1%} are inside a furniture box, "
          f"{total('in_points') / total('moving'):.1%} in a cell occupied by points ({total('in_points_wide') / total('moving'):.1%} counting the neighbouring cells too)")
    summary = {key: total(key) for key in ("points", "kept", "occupied_cells", "occupied_raw", "outline", "outline10", "outline20", "outline30", "wall", "wall10", "wall20", "wall30",
                                             "occupied", "unexplained", "moving", "in_box", "in_points", "in_points_wide")}
    with_frames = [r for r in done if "from_points" in r]
    if with_frames:
        points, maps, walls = (np.concatenate([r[key] for r in with_frames]).ravel() for key in ("from_points", "from_map", "from_walls"))
        ok = np.isfinite(points) & np.isfinite(maps)
        points, maps, walls = points[ok], maps[ok], walls[ok]
        print(f"\ndistances in the camera's view, {len(points):,d} directions of {sum(len(r['from_points']) for r in with_frames):,d} test-bed frames: "
              f"map (walls or furniture) mean {maps.mean():.2f} m, points mean {points.mean():.2f} m ({(points >= FAR).mean():.0%} see no point within 5 m)")
        print("  where the map says the obstacle is    directions   points say nearer by 0.3 m+   within 0.3 m   further by 0.3 m+   median |difference|")
        summary["distances"] = []
        for label, mask in (("under 1 m", maps < 1.0), ("1 to 2.5 m", (maps >= 1.0) & (maps < 2.5)), ("2.5 m and more", maps >= 2.5), ("all", np.ones(len(maps), bool))):
            delta = points[mask] - maps[mask]
            row = {"where": label, "directions": int(mask.sum()), "nearer": float((delta < -0.3).mean()), "same": float((np.abs(delta) <= 0.3).mean()),
                   "further": float((delta > 0.3).mean()), "median_abs": float(np.median(np.abs(delta)))}
            summary["distances"].append(row)
            print(f"  {label:36s} {row['directions']:11,d} {row['nearer']:28.0%} {row['same']:14.0%} {row['further']:19.0%} {row['median_abs']:21.2f}")
        furniture_first = maps < walls - 0.05  # the nearest thing in the map is furniture, not a wall
        delta = points[furniture_first] - maps[furniture_first]
        print(f"  where the map's nearest obstacle is furniture ({int(furniture_first.sum()):,d} directions): points nearer by 0.3 m+ {float((delta < -0.3).mean()):.0%}, "
              f"within 0.3 m {float((np.abs(delta) <= 0.3).mean()):.0%}, further by 0.3 m+ {float((delta > 0.3).mean()):.0%}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
