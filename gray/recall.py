#!/usr/bin/env python3
"""How many moments need memory: the place the person goes to is not in view at now, but was seen earlier.

    python -m gray.recall --out <folder>

The models are given 2 s of history. Whether they would gain from remembering more can only be
measured on moments where memory is needed. This counts them, from the head paths alone (no model).

For every moment that goes 1 m or further, two targets are looked at:

    end      where the head is 5 s after now (every such moment has one)
    object   the cell of the object the sentence names that is nearest to that end, where the room map has the
             object, the person starts 1.5 m or further from it and ends within 1 m (the destination test of gray/score.py)

Both are taken 0.8 m above the floor. A target counts as in view of a frame when it lies inside the
79.6 degree pinhole view the models are given (gray/pinhole.py), 0.3 to ``--far`` metres from the camera,
the camera stands on the same storey, and, where the recording has a room map, no wall of the map lies
between them. Furniture is not checked: a target behind a sofa counts as seen.

For each moment: how long before now the target was last in view (0: in view at now).

    <out>/recall.npz   row (index into moments.npz), end_ago and object_ago in frames (-1: never before now; -2: no such target)
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from gray import frames
from gray.features import CACHE
from gray.layout import CELL, Layout
from gray.moments import FUTURE, HISTORY
from gray.navigate import AWAY, NEAR, targets
from gray.pinhole import VIEW
from gray.tracks import to_world

HEIGHT = 0.8  # metres above the floor
STOREY = 0.5  # metres: frames whose floor differs by more are on another storey
BINS = (("in view at now", 0, 0), ("not now, within the 2 s given", 1, HISTORY - 1), ("2 to 10 s before", HISTORY, 150), ("10 to 30 s before", 151, 450),
        ("30 to 60 s before", 451, 900), ("longer ago", 901, 10**9), ("never before now", -1, -1))


def last_seen(pose: np.ndarray, floor: np.ndarray, valid: np.ndarray, now: int, target: np.ndarray, far: float, layout: Layout | None) -> int:
    """Frames between now and the last frame up to now that has the ground point ``target`` (x, height, z) in view; -1 if none."""
    rotation = frames.matrix(torch.from_numpy(pose[: now + 1, 3:])).numpy()  # columns: the camera's axes in the ground frame; the third is the view direction
    offset = target[None] - pose[: now + 1, :3]
    along = (offset * rotation[:, :, 2]).sum(1)
    edge = np.tan(np.radians(VIEW / 2)) * along
    seen = (valid[: now + 1] & (along >= 0.3) & (np.linalg.norm(offset, axis=1) <= far) & (np.abs((offset * rotation[:, :, 0]).sum(1)) <= edge)
            & (np.abs((offset * rotation[:, :, 1]).sum(1)) <= edge) & (np.abs(floor[: now + 1] - floor[now]) <= STOREY))
    for f in np.flatnonzero(seen)[::-1]:  # the latest first; most are not behind a wall
        if layout is None:
            return now - int(f)
        steps = max(2, int(np.linalg.norm(offset[f]) / CELL))
        line = pose[f, :3][None] + np.linspace(0, 1, steps)[1:-1, None] * offset[f][None]
        world = to_world(line.astype(np.float64), float(floor[f]))
        camera = to_world(pose[f : f + 1, :3].astype(np.float64), float(floor[f]))
        if not layout.flags_world(world, np.repeat(layout.storeys_world(camera), len(world)))["wall"].any():
            return now - int(f)
    return -1


def one(job: tuple) -> list[tuple[int, int, int]]:
    name, rows, nows, sentences, tracks, maps, cache, far = job
    track = np.load(Path(tracks) / f"{name}.npz")
    pose, floor, valid = track["pose"], track["floor"], track["valid"]
    layout = Layout(Path(maps) / f"{name}.npz") if (Path(maps) / f"{name}.npz").exists() else None
    spoken = [s["text"] for s in json.loads((Path(cache) / name / "sentences.json").read_text())]
    out = []
    for row, now, sentence in zip(rows, nows, sentences):
        end = pose[now + FUTURE, :3].copy()
        end[1] = HEIGHT
        thing = -2
        if layout is not None:
            window = int(np.abs(layout.data["floor_z"] - floor[now]).argmin())
            cells = targets(layout, window, spoken[sentence])
            if cells is not None:
                start, stop = to_world(pose[now, :3].astype(np.float64), float(floor[now])), to_world(pose[now + FUTURE, :3].astype(np.float64), float(floor[now + FUTURE]))
                gaps = np.linalg.norm(cells - stop[:2], axis=1)
                if gaps.min() <= NEAR and np.linalg.norm(cells - start[:2], axis=1).min() >= AWAY:
                    x, y = cells[int(gaps.argmin())]
                    thing = last_seen(pose, floor, valid, now, np.array([x, HEIGHT, -y], dtype=np.float32), far, layout)
        out.append((int(row), last_seen(pose, floor, valid, now, end, far, layout), thing))
    return out


def table(title: str, ago: np.ndarray) -> dict[str, int]:
    ago = ago[ago > -2]
    counts = {label: int(((ago >= low) & (ago <= high)).sum()) for label, low, high in BINS}
    print(f"\n{title}: {len(ago):,d} moments")
    for label, count in counts.items():
        print(f"  {label:32s} {count:7,d}  {count / max(1, len(ago)):6.1%}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--moments", type=Path, default=Path("/ptmp/enazaren/gray/moments/moments.npz"))
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=Path("/ptmp/enazaren/gray/runs/recall"))
    parser.add_argument("--far", type=float, default=8.0, help="metres: further than this from the camera does not count as seen")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="only the first recordings (a test)")
    args = parser.parse_args()

    listing = dict(np.load(args.moments))
    going = np.flatnonzero(listing["going"])
    by_name: dict[str, list[int]] = {}
    for row in going:
        by_name.setdefault(str(listing["sequence"][row]), []).append(int(row))
    names = [n for n in sorted(by_name) if (args.tracks / f"{n}.npz").exists() and (args.cache / n / "sentences.json").exists()][: args.limit]
    jobs = [(n, by_name[n], listing["now"][by_name[n]].tolist(), listing["sentence"][by_name[n]].tolist(), str(args.tracks), str(args.maps), str(args.cache), args.far) for n in names]
    with Pool(args.workers) as pool:
        found = np.array([r for part in pool.imap(one, jobs, chunksize=2) for r in part], dtype=np.int64)
    row, end, thing = found[:, 0], found[:, 1], found[:, 2]
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "recall.npz", row=row, end_ago=end, object_ago=thing)

    held, subset, mapped = listing["held"][row], listing["subset"][row], listing["has_map"][row]
    print(f"{len(row):,d} moments that go 1 m or further, from {len(names)} recordings; in view: inside the {VIEW:g} degree view, 0.3 to {args.far:g} m, same storey, no wall between")
    result = {}
    for title, pick in (("held out, navigation subset", held & subset), ("held out, navigation subset, with a room map (walls checked)", held & subset & mapped),
                        ("held out, all moments", held), ("training, navigation subset", ~held & subset)):
        result[title] = {"where the head is after 5 s": table(f"{title} | where the head is after 5 s", end[pick]),
                         "the named object": table(f"{title} | the named object", thing[pick])}
    (args.out / "recall.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
