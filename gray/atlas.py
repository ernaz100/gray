#!/usr/bin/env python3
"""Pictures of every location: the room map and everything that was walked in it.

    python -m gray.atlas --maps <folder of gray/layout.py> --out <folder>

Per location it writes <location>.png with one panel per storey (floor grey, walls black,
furniture of one recording brown, stairs light blue, the recorded head paths of all recordings
in blue), and sheet-<n>.png with the busiest storey of every location, 15 locations per sheet.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gray.features import CACHE
from gray.layout import CELL, HEAD, UNMAPPED

TILE = 640  # pixels


def base(d: dict, level: int) -> np.ndarray:
    image = np.full((*d["floor"].shape[1:], 3), 255, dtype=np.uint8)
    image[d["floor"][level]] = (228, 228, 228)
    image[d["furniture"][level] > 0] = (222, 190, 150)
    image[d["stairs"] & d["floor"][level]] = (190, 215, 240)
    image[d["wall"][level]] = (15, 15, 15)
    return image


def location(job: tuple) -> dict:
    name, recordings, maps, cache, out = job
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    first = dict(np.load(Path(maps) / f"{recordings[0]}.npz"))
    levels = first["levels"]
    lines = [[] for _ in levels]
    metres, windows = 0.0, 0
    for recording in recordings:
        d = np.load(Path(maps) / f"{recording}.npz")
        head = np.load(Path(cache) / recording / "head.npy")[:, 0:31:3, :3].astype(np.float64)  # the first 2 s of every window: no overlap
        world = np.einsum("wni,wij->wnj", head + d["shift"][:, None], d["turn"])
        gap = np.abs(world[..., 2:3] - HEAD - levels[None, None])
        storey = np.where(gap.min(-1) <= UNMAPPED, gap.argmin(-1), -1)
        metres += float(np.linalg.norm(np.diff(world[..., :2], axis=1), axis=-1).sum())
        windows += len(head)
        for level in range(len(levels)):
            on = (storey == level).all(1)
            lines[level].extend(world[on][..., :2])
    count = np.array([len(l) for l in lines])
    rows, cols = first["floor"].shape[1:]
    extent = (first["origin"][0], first["origin"][0] + cols * CELL, first["origin"][1], first["origin"][1] + rows * CELL)

    def panel(axis, level: int, width: float) -> None:
        axis.imshow(base(first, level), origin="lower", extent=extent, interpolation="nearest")
        axis.add_collection(LineCollection(lines[level], colors=(0.1, 0.35, 0.9, 0.35), linewidths=width))
        axis.set(xlim=extent[:2], ylim=extent[2:])
        axis.set_aspect("equal")

    shown = [int(i) for i in np.argsort(-count) if count[i] > 0] or [0]
    figure, axes = plt.subplots(1, len(shown), figsize=(9 * len(shown), 9 * rows / max(cols, 1) + 0.8), squeeze=False)
    for axis, level in zip(axes[0], shown):
        panel(axis, level, 0.5)
        axis.set_title(f"storey at {levels[level]:+.1f} m: {count[level] * 2 / 60:.0f} min of walking and standing", fontsize=10)
        axis.set_xlabel("m")
    figure.suptitle(f"{name}: {len(recordings)} recordings, {metres / 1000:.1f} km of head path", fontsize=12)
    figure.tight_layout()
    figure.savefig(Path(out) / f"{name}.png", dpi=70)
    plt.close(figure)

    figure = plt.figure(figsize=(TILE / 100, TILE / 100), dpi=100)
    axis = figure.add_axes([0.01, 0.01, 0.98, 0.9])
    panel(axis, shown[0], 0.35)
    axis.axis("off")
    figure.text(0.5, 0.955, f"{name}: {len(recordings)} recordings, {metres / 1000:.1f} km" + (f", {len(shown)} storeys" if len(shown) > 1 else ""),
                ha="center", va="center", fontsize=11)
    figure.savefig(Path(out) / f"tile-{name}.png")
    plt.close(figure)
    return {"location": name, "recordings": len(recordings), "km": metres / 1000, "windows": windows, "storeys_walked": len(shown),
            "size_m": [cols * CELL, rows * CELL]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--out", type=Path, default=Path("/ptmp/enazaren/gray/atlas"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for r in json.loads((args.maps / "summary.json").read_text())["recordings"]:
        groups[r["location"]].append(r["name"])
    jobs = [(name, sorted(recordings), str(args.maps), str(args.cache), str(args.out)) for name, recordings in sorted(groups.items())]
    with Pool(args.workers) as pool:
        results = pool.map(location, jobs, chunksize=1)
    from PIL import Image

    per_sheet, columns = 15, 3
    for sheet in range(0, len(results), per_sheet):
        part = results[sheet : sheet + per_sheet]
        rows = -(-len(part) // columns)
        canvas = Image.new("RGB", (columns * TILE, rows * TILE), "white")
        for i, r in enumerate(part):
            canvas.paste(Image.open(args.out / f"tile-{r['location']}.png").convert("RGB"), ((i % columns) * TILE, (i // columns) * TILE))
        canvas.save(args.out / f"sheet-{sheet // per_sheet}.png")
    for r in results:
        (args.out / f"tile-{r['location']}.png").unlink()
        print(f"{r['location']}: {r['recordings']:3d} recordings, {r['km']:5.1f} km, map {r['size_m'][0]:.0f} x {r['size_m'][1]:.0f} m, {r['storeys_walked']} storeys walked")
    print(f"{len(results)} locations, {sum(r['recordings'] for r in results)} recordings, {sum(r['km'] for r in results):.0f} km of head path")
    (args.out / "atlas.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
