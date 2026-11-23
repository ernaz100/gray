#!/usr/bin/env python3
"""Where recorded paths cross a wall of the room maps, and what kind of wall it is.

    python -m gray.crossings --maps <folder of gray/layout.py> --limit 6

For every recording that has a map, every recorded pelvis point that lies in a wall cell is
looked up in the annotation: which wall it is, how high the wall reaches, in how many recordings
it was annotated, and how far the nearest annotated door is. Prints the counts per kind and
writes pictures with the crossings in red and the doors in green (cut into a wall) or magenta (not cut).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gray.features import CACHE
from gray.layout import BAND, CELL, STOREY, Layout, graph_of, nymeria
from gray.rectify import SEQUENCES

NEAR_DOOR = 0.6  # metres
STAIRS = 0.5  # metres from an annotated stairs box
LOW = 1.2  # a wall that ends lower than this above the floor is a half wall


def segment(plane, rect) -> np.ndarray:
    """The two ends (2, 2) of a rectangle of a plane, seen from above."""
    return np.array([plane.offset * plane.normal[:2] + a * plane.along[:2] for a in rect[:2]])


def distance(point: np.ndarray, ends: np.ndarray) -> float:
    a, b = ends
    t = np.clip((point - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1)
    return float(np.linalg.norm(point - (a + t * (b - a))))


def group(job: tuple) -> dict:
    recordings, cached, cache, maps, out, shown = job
    module = nymeria()
    location = module.Location(recordings[0][1], [name for name, _l in recordings])
    cut = {door for door, _wall in location.door_cuts}
    doors = [(door, segment(door, door.median_rect()), door.uid in cut) for door in location.doors]
    stairs = [box.footprint() for box in location.stairs]
    counts, events, pictures, seen = defaultdict(int), defaultdict(int), [], []
    points_total = 0
    for name in cached:
        path = Path(maps) / f"{name}.npz"
        if not path.exists():
            continue
        layout = Layout(path)
        d = layout.data
        root = np.load(Path(cache) / name / "root.npy")
        marks = []
        for window in range(len(root)):
            pelvis = np.stack([root[window, ::3, 0], np.zeros(40), root[window, ::3, 1]], axis=1).astype(np.float64)
            flags = layout.flags(window, pelvis, layout.storeys(window, np.load(Path(cache) / name / "head.npy", mmap_mode="r")[window, ::3, :3].astype(np.float64)))
            points_total += len(pelvis)
            if not flags["wall"].any():
                continue
            z = float(d["levels"][d["level"][window]])
            kinds = set()
            head = np.load(Path(cache) / name / "head.npy", mmap_mode="r")[window, :, 1]
            climbs = float(head.max() - head.min()) > 1.0  # the head height changes by more than a metre: stairs, or lying down
            for point in layout.world(window, pelvis)[flags["wall"]][:, :2]:
                kind = "odd-shaped wall box"
                on_stairs = any(module._in_convex(point[None], quad)[0] or min(distance(point, quad[[i, (i + 1) % 4]]) for i in range(4)) <= STAIRS for quad in stairs)
                for plane, rects in zip(location.walls, location.wall_rects):
                    across, along = point @ plane.normal[:2] - plane.offset, point @ plane.along[:2]
                    hit = [r for r in rects if abs(across) <= max(plane.thickness, CELL) / 2 + CELL and r[0] - CELL <= along <= r[1] + CELL
                           and r[3] > z + BAND[0] and r[2] < z + BAND[1]]
                    if hit and on_stairs:
                        kind = "on or beside annotated stairs"
                        break
                    if hit and climbs:
                        kind = "head height changes by over 1 m in the window, no stairs annotated there"
                        break
                    if hit:
                        near = min([(distance(point, ends), was_cut) for _door, ends, was_cut in doors] or [(np.inf, False)])
                        if near[0] <= NEAR_DOOR:
                            kind = "door nearby, cut into the wall" if near[1] else "door nearby, not cut into the wall"
                        elif max(r[3] for r in hit) < z + LOW:
                            kind = "half wall"
                        elif len(set(plane.recordings)) == 1:
                            kind = "wall annotated in one recording only"
                        else:
                            kind = "full wall, no door annotated nearby"
                        break
                counts[kind] += 1
                kinds.add(kind)
                marks.append(point)
            for kind in kinds:
                events[kind] += 1
            first = int(np.flatnonzero(flags["wall"])[0])
            seen.append((sorted(kinds)[0], name, window, first * 3, *layout.world(window, pelvis)[first, :2].tolist()))
        if name in shown and marks:
            pictures.append((name, np.array(marks)))
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            level = int(np.bincount(d["level"]).argmax())
            image = np.full((*d["floor"].shape[1:], 3), 255, dtype=np.uint8)
            image[d["floor"][level]] = (225, 225, 225)
            image[d["furniture"][level] > 0] = (225, 200, 165)
            image[d["wall"][level]] = (20, 20, 20)
            rows, cols = image.shape[:2]
            figure, axis = plt.subplots(figsize=(12, 12 * rows / max(cols, 1) + 0.6))
            axis.imshow(image, origin="lower", extent=(d["origin"][0], d["origin"][0] + cols * CELL, d["origin"][1], d["origin"][1] + rows * CELL))
            for window in np.flatnonzero(d["level"] == level)[::2]:
                p = layout.world(window, np.stack([root[window, ::5, 0], np.zeros(24), root[window, ::5, 1]], axis=1).astype(np.float64))
                axis.plot(p[:, 0], p[:, 1], color="tab:blue", linewidth=0.4, alpha=0.5)
            for _door, ends, was_cut in doors:
                axis.plot(ends[:, 0], ends[:, 1], color="limegreen" if was_cut else "magenta", linewidth=3)
            m = np.array(marks)
            axis.scatter(m[:, 0], m[:, 1], color="red", s=6, zorder=5)
            axis.set(title=f"{name}: recorded pelvis paths blue, wall crossings red, doors green (cut) or magenta (not cut)", xlim=(d["origin"][0], d["origin"][0] + cols * CELL),
                     ylim=(d["origin"][1], d["origin"][1] + rows * CELL))
            figure.tight_layout()
            figure.savefig(Path(out) / f"{name}-crossings.png", dpi=80)
            plt.close(figure)
    order = np.random.default_rng(0).permutation(len(seen))[:80]
    return {"seen": [seen[i] for i in order], "location": recordings[0][1], "recordings": len(cached), "points": points_total, "counts": dict(counts), "events": dict(events),
            "doors": len(doors), "doors_cut": sum(c for _d, _e, c in doors), "walls": len(location.walls),
            "walls_one_recording": sum(len(set(w.recordings)) == 1 for w in location.walls), "stairs": len(stairs), "storeys": len({round(t, 1) for _u, t, _q in location.floors})}


def look(results: list[dict], args) -> None:
    """Pictures of single crossings: the map around the crossing, and the video one second before, at, and one second after it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gray.rectify import video_frame

    wanted = (("full wall, no door annotated nearby", 10), ("door nearby, cut into the wall", 4), ("on or beside annotated stairs", 2))
    picked = []
    for kind, count in wanted:
        pools = [[e for e in r["seen"] if e[0] == kind] for r in results]
        used = set()
        while len([p for p in picked if p[0] == kind]) < count and any(pools):
            for pool in pools:  # one location after the other, a different recording each time where possible
                fresh = [e for e in pool if e[1] not in used] or pool
                if fresh and len([p for p in picked if p[0] == kind]) < count:
                    picked.append(fresh[0]); used.add(fresh[0][1]); pool.remove(fresh[0])
    for part in range(0, len(picked), 4):
        rows = picked[part : part + 4]
        figure, axes = plt.subplots(len(rows), 4, figsize=(20, 5 * len(rows)), squeeze=False)
        for axis_row, (kind, name, window, token, x, y) in zip(axes, rows):
            layout = Layout(args.maps / f"{name}.npz")
            d = layout.data
            folder = args.cache / name
            head, root, starts = np.load(folder / "head.npy", mmap_mode="r"), np.load(folder / "root.npy", mmap_mode="r"), np.load(folder / "starts.npy")
            level = int(layout.storeys(window, np.array(head[window, token : token + 1, :3], dtype=np.float64))[0])
            image = np.full((*d["floor"].shape[1:], 3), 255, dtype=np.uint8)
            image[d["floor"][level]] = (225, 225, 225)
            image[d["furniture"][level] > 0] = (225, 200, 165)
            image[d["stairs"]] = (190, 215, 240)
            image[d["wall"][level]] = (20, 20, 20)
            size = image.shape[:2]
            axis_row[0].imshow(image, origin="lower", extent=(d["origin"][0], d["origin"][0] + size[1] * CELL, d["origin"][1], d["origin"][1] + size[0] * CELL))
            pelvis = layout.world(window, np.stack([root[window, :, 0], np.zeros(root.shape[1]), root[window, :, 1]], axis=1).astype(np.float64))
            above = layout.world(window, np.array(head[window, :, :3], dtype=np.float64))
            axis_row[0].plot(pelvis[:, 0], pelvis[:, 1], color="tab:blue", linewidth=2, label="pelvis")
            axis_row[0].plot(above[:, 0], above[:, 1], color="tab:cyan", linewidth=1, label="head")
            axis_row[0].scatter(pelvis[0, 0], pelvis[0, 1], color="tab:blue", s=40, marker="o")
            axis_row[0].scatter([x], [y], color="red", s=60, zorder=5, label="crossing")
            axis_row[0].set(xlim=(x - 2.5, x + 2.5), ylim=(y - 2.5, y + 2.5), title=f"{kind}\n{name}, window {window}")
            axis_row[0].legend(fontsize=7, loc="upper right")
            sentences = json.loads((folder / "sentences.json").read_text())
            index = np.load(folder / "text_index.npy")[window]
            text = " ".join(sentences[i]["text"] for i in index if i >= 0)
            at = (int(starts[window]) + 2 * token) / 30.0
            for axis, shift in zip(axis_row[1:], (-1.0, 0.0, 1.0)):
                try:
                    axis.imshow(video_frame(args.videos / name / "preview_rgb.mp4", max(at + shift, 0))[::2, ::2])
                except Exception as error:  # noqa: BLE001
                    axis.text(0.1, 0.5, type(error).__name__)
                axis.set_title(f"video {shift:+.0f} s" + (f" (at {int(at // 60)}:{at % 60:04.1f})" if shift == 0 else ""), fontsize=9)
                axis.axis("off")
            axis_row[1].text(0, -0.02, text[:330], transform=axis_row[1].transAxes, fontsize=7, va="top", wrap=True)
        figure.tight_layout()
        figure.savefig(args.maps / f"look-{part // 4}.png", dpi=60)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    annotated = sorted(p.parents[2].name for p in args.videos.glob("*/objects/boxy/scene_objects.csv"))
    cached = {p.parent.name for p in args.cache.glob("*/head.npy")}
    groups = defaultdict(list)
    for name in annotated:
        location = json.loads((args.videos / name / "metadata.json").read_text())["location"]
        groups[(location, graph_of(args.videos / name))].append((name, location))
    keys = sorted(groups)[: args.limit]
    shown = {name for key in keys for name, _l in [g for g in groups[key] if g[0] in cached][:2]}
    jobs = [(groups[key], [n for n, _l in groups[key] if n in cached], str(args.cache), str(args.maps), str(args.maps), shown) for key in keys]
    with Pool(args.workers) as pool:
        results = pool.map(group, jobs)
    total, events = defaultdict(int), defaultdict(int)
    for r in results:
        print(f"{r['location']}: {r['recordings']} recordings, {r['walls']} walls ({r['walls_one_recording']} annotated in one recording only), "
              f"{r['doors']} doors of which {r['doors_cut']} are cut into a wall, {r['stairs']} stairs boxes, {r['storeys']} floor heights; pelvis points in a wall {sum(r['counts'].values()):,d} of {r['points']:,d}: "
              + ", ".join(f"{k} {v:,d}" for k, v in sorted(r["counts"].items(), key=lambda kv: -kv[1])))
        for k, v in r["counts"].items():
            total[k] += v
        for k, v in r["events"].items():
            events[k] += v
    if not results:
        return
    points = sum(r["points"] for r in results)
    print(f"\nall {len(results)} locations: {sum(total.values()):,d} of {points:,d} pelvis points are in a wall ({sum(total.values()) / max(points, 1):.2%})")
    for k, v in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f"  {k:75s} {v:7,d} points ({v / max(sum(total.values()), 1):4.0%}), {events[k]:6,d} windows")
    look(results, args)
    print(f"wrote pictures to {args.maps}", flush=True)


if __name__ == "__main__":
    main()
