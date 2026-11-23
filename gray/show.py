#!/usr/bin/env python3
"""Pictures of what the head-path model does: the camera's view, the sentence, and the paths it draws on the room map.

    python -m gray.show --run <runs>/head-v5 --out <folder>

For a few held-out moments it writes pages with one moment per row:

    left    the pinhole view of the frame at now (what the model sees last)
    right   the home from above around the camera (gray/ground.py: green where people walked, black walls, red what
            was measured in many recordings, white unknown), the 2 s before now in grey, the recorded 5 s in black,
            and the paths the model draws in colour. The triangle is the camera at now, pointing where it looks.

Half of the moments are ones whose sentence names a destination. The paths are drawn as the
planner is used: 4 sampling steps, half the starting noise.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import torch

from gray import frames
from gray.ground import Ground
from gray.head import NOW, load, sample
from gray.joint import to_device
from gray.layout import CELL, HEAD
from gray.moments import HISTORY, Moments, collate
from gray.pinhole import VIEW
from gray.rectify import SEQUENCES, calibration, lookup, mounting, rectify, video_frame
from gray.tracks import to_world


def view(sequence: Path, frame: int, side: int = 448) -> np.ndarray:
    """The pinhole view of one frame, (side, side, 3)."""
    lens, on_glasses = calibration(sequence)
    clockwise = bool(mounting(sequence, on_glasses)[0] > 0)
    full = torch.from_numpy(video_frame(sequence / "preview_rgb.mp4", frame / 15.0).copy()).permute(2, 0, 1)[None].float()
    where = lookup(lens, full.shape[-1], clockwise, view=VIEW, wide=2 * side, high=2 * side).float()
    return torch.nn.functional.avg_pool2d(rectify(full, where), 2)[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ground", type=Path, default=Path("/ptmp/enazaren/gray/ground"))
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--moments", type=int, default=12)
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std, config = load(args.run, device)
    cache, tracks = Path(config["cache"]), Path(config["tracks"])
    listing = dict(np.load(config["moments"]))
    data = Moments(Path(config["moments"]), tracks, Path(config["images"]), cache, held=True, patches=Path(config["patches"]) if config.get("patches") else None,
                   only=listing["has_map"] & listing["going"], depth=Path(config["depth"]) if config.get("depth") else None)
    rng = np.random.default_rng(args.seed)
    named = np.flatnonzero(listing["destination"][data.rows] != "")
    plain = np.flatnonzero(listing["destination"][data.rows] == "")
    picked = np.concatenate([rng.choice(named, args.moments // 2, replace=False), rng.choice(plain, args.moments - args.moments // 2, replace=False)])
    batch = to_device(collate([data[int(i)] for i in picked]), device)
    generator = torch.Generator(device=device)
    draws = []
    for k in range(args.draws):
        generator.manual_seed(1000 + k)
        draws.append(sample(model, batch, mean, std, args.steps, generator, device == "cuda", temperature=args.temperature))
    draws = torch.stack(draws).double().cpu().numpy()  # (K, B, 75, 3) ground frame
    pose = batch["pose"].double().cpu().numpy()
    location = {m["sequence_id"]: m["location"] for m in json.loads((cache / "index.json").read_text())["sequences"]}
    colours = plt.get_cmap("tab10")
    args.out.mkdir(parents=True, exist_ok=True)

    per_page = 3
    for page in range(0, len(picked), per_page):
        rows = range(page, min(page + per_page, len(picked)))
        figure, axes = plt.subplots(len(rows), 2, figsize=(13, 6.3 * len(rows)), squeeze=False)
        for axis_row, b in zip(axes, rows):
            row = int(batch["row"][b])
            name, now = str(listing["sequence"][row]), int(batch["now"][b])
            sentence = json.loads((cache / name / "sentences.json").read_text())[int(listing["sentence"][row])]["text"]
            floor = np.load(tracks / f"{name}.npz")["floor"]
            ground = Ground(args.ground / f"{location[name]}.npz")
            d = ground.data
            past = to_world(pose[b, :HISTORY, :3], floor[now - HISTORY + 1 : now + 1])
            future = to_world(pose[b, HISTORY:, :3], floor[now + 1 : now + 1 + pose.shape[1] - HISTORY])
            here = past[-1]
            level = int(np.abs(here[2] - HEAD - d["levels"]).argmin())
            image = np.full((*d["free"].shape[1:], 3), 255, dtype=np.uint8)
            image[d["free"][level]] = (200, 235, 200)
            image[d["steady"][level]] = (225, 110, 100)
            image[d["wall"][level]] = (15, 15, 15)
            size = image.shape[:2]
            try:
                axis_row[0].imshow(view(args.videos / name, now))
            except Exception as error:  # noqa: BLE001
                axis_row[0].text(0.1, 0.5, f"no frame: {type(error).__name__}")
            axis_row[0].axis("off")
            axis_row[0].set_title("\n".join(textwrap.wrap(sentence, 70)[:4]), fontsize=9, loc="left")
            axis = axis_row[1]
            axis.imshow(image, origin="lower", extent=(d["origin"][0], d["origin"][0] + size[1] * CELL, d["origin"][1], d["origin"][1] + size[0] * CELL), interpolation="nearest")
            axis.plot(past[:, 0], past[:, 1], color="0.45", linewidth=2, label="the 2 s before now")
            for k in range(args.draws):
                path = to_world(draws[k, b], floor[now])
                axis.plot(np.append(here[0], path[:, 0]), np.append(here[1], path[:, 1]), color=colours(k), linewidth=1.6, alpha=0.9, label="drawn by the model" if k == 0 else None)
                axis.scatter(path[-1, 0], path[-1, 1], color=colours(k), s=18, zorder=4)
            axis.plot(np.append(here[0], future[:, 0]), np.append(here[1], future[:, 1]), color="black", linewidth=2.2, linestyle="--", label="recorded 5 s")
            yaw = float(frames.heading(batch["pose"][b : b + 1, NOW, 3:]))
            angle = math.atan2(-math.cos(yaw), -math.sin(yaw))
            axis.scatter(here[0], here[1], marker=(3, 0, math.degrees(angle) - 90), s=260, color="tab:blue", edgecolor="white", zorder=5, label="camera at now")
            half = max(3.0, float(np.abs(np.concatenate([future[:, :2], *[to_world(draws[k, b], floor[now])[:, :2] for k in range(args.draws)]]) - here[:2]).max()) + 0.7)
            axis.set(xlim=(here[0] - half, here[0] + half), ylim=(here[1] - half, here[1] + half), xlabel="m",
                     title=f"{name[:40]}, {now // 15 // 60}:{now // 15 % 60:02d}; named: {listing['destination'][row] or 'nothing'}")
            axis.title.set_fontsize(9)
            axis.set_aspect("equal")
            axis.legend(fontsize=7, loc="upper right")
        figure.tight_layout()
        figure.savefig(args.out / f"paths-{page // per_page}.png", dpi=75)
        plt.close(figure)
        print(f"wrote {args.out / f'paths-{page // per_page}.png'}", flush=True)


if __name__ == "__main__":
    main()
