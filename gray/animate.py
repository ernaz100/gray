#!/usr/bin/env python3
"""Videos of generated motion in a home: the body as a stick figure, from above on the room map and from the side.

    python -m gray.animate --replies <folder with reply-*.json> --ground <location>.npz --out <folder>

Runs on a workstation (matplotlib and ffmpeg). The replies are those of gray/scene_worker.py
(``--demo`` writes them as files). Per reply one MP4 at 30 frames per second:

    left    the home from above (gray/ground.py: green where people walked, red measured, black walls), the planned camera
            path as a dashed line, the pelvis trail, and the figure seen from above
    right   the figure in 3D, the view following the pelvis; the grey dot is the planned camera position of that frame

``--sheet`` also writes one picture with six stages of every clip side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CELL = 0.05


def background(ground: dict, height: float) -> tuple[np.ndarray, tuple]:
    level = int(np.abs(height - 1.3 - ground["levels"]).argmin())
    image = np.full((*ground["free"].shape[1:], 3), 255, dtype=np.uint8)
    image[ground["free"][level]] = (200, 235, 200)
    image[ground["steady"][level]] = (225, 110, 100)
    image[ground["wall"][level]] = (15, 15, 15)
    origin = ground["origin"]
    return image, (origin[0], origin[0] + image.shape[1] * CELL, origin[1], origin[1] + image.shape[0] * CELL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replies", type=Path, required=True)
    parser.add_argument("--ground", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sheet", action="store_true")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    ground = dict(np.load(args.ground, allow_pickle=True))
    args.out.mkdir(parents=True, exist_ok=True)
    clips = []
    for path in sorted(args.replies.glob("reply-*.json")):
        reply = json.loads(path.read_text())
        joints = np.array(reply["joints_mm"], dtype=np.float64).reshape(reply["frames"], 22, 3) / 1000  # x, up, z
        camera = np.array(reply["head_mm"], dtype=np.float64).reshape(-1, 3) / 1000
        spawn_height = float(reply["match"]["pos"][1])  # in the world; picks the storey
        image, extent = background(ground, spawn_height)
        clips.append((path.stem, reply, joints, camera, image, extent))

    def draw(axes, reply, joints, camera, image, extent, f):
        top, side = axes
        top.clear(); side.clear()
        centre = joints[:, 0, [0, 2]].mean(0)
        half = max(2.5, float(np.abs(joints[:, 0, [0, 2]] - centre).max()) + 1.2)
        top.imshow(image, origin="lower", extent=extent, interpolation="nearest")
        top.plot(camera[:, 0], -camera[:, 2], color="0.25", linestyle="--", linewidth=1.2, label="planned camera path")
        top.plot(joints[: f + 1, 0, 0], -joints[: f + 1, 0, 2], color="tab:blue", linewidth=2, label="pelvis so far")
        for j, p in enumerate(PARENTS):
            if p >= 0:
                top.plot(joints[f, [p, j], 0], -joints[f, [p, j], 2], color="black", linewidth=1.6)
                side.plot(joints[f, [p, j], 0], -joints[f, [p, j], 2], joints[f, [p, j], 1], color="black", linewidth=2)
        top.scatter(joints[f, 15, 0], -joints[f, 15, 2], color="tab:orange", s=30, zorder=5, label="head")
        top.set(xlim=(centre[0] - half, centre[0] + half), ylim=(-centre[1] - half, -centre[1] + half), xlabel="m")
        top.set_aspect("equal")
        top.legend(fontsize=7, loc="upper right")
        top.set_title(f"\"{reply['text']}\"   {f / reply['fps']:.1f} s", fontsize=10)
        at = camera[min(f // 2, len(camera) - 1)]
        side.scatter(at[0], -at[2], at[1], color="0.5", s=40)
        side.scatter(joints[f, 15, 0], -joints[f, 15, 2], joints[f, 15, 1], color="tab:orange", s=30)
        here = joints[f, 0]
        side.set(xlim=(here[0] - 1, here[0] + 1), ylim=(-here[2] - 1, -here[2] + 1), zlim=(0, 2))
        side.set_box_aspect((1, 1, 1))
        side.view_init(elev=12, azim=-60)
        side.set_title(f"pelvis travels {reply['travel_m']} m in 5 s; head joint {reply['head_to_camera_cm']} cm from the planned camera", fontsize=9)

    for name, reply, joints, camera, image, extent in clips:
        figure = plt.figure(figsize=(12, 6))
        axes = (figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2, projection="3d"))
        writer = FFMpegWriter(fps=reply["fps"], bitrate=1800)
        with writer.saving(figure, str(args.out / f"{name}.mp4"), dpi=80):
            for f in range(reply["frames"]):
                draw(axes, reply, joints, camera, image, extent, f)
                writer.grab_frame()
        plt.close(figure)
        print(f"wrote {args.out / f'{name}.mp4'}", flush=True)

    if args.sheet and clips:
        stages = 6
        figure, grid = plt.subplots(len(clips), stages, figsize=(3 * stages, 3.2 * len(clips)), squeeze=False)
        for row, (name, reply, joints, camera, image, extent) in zip(grid, clips):
            centre = joints[:, 0, [0, 2]].mean(0)
            half = max(2.0, float(np.abs(joints[:, 0, [0, 2]] - centre).max()) + 0.8)
            for axis, f in zip(row, np.linspace(0, reply["frames"] - 1, stages).astype(int)):
                axis.imshow(image, origin="lower", extent=extent, interpolation="nearest")
                axis.plot(camera[:, 0], -camera[:, 2], color="0.25", linestyle="--", linewidth=1)
                axis.plot(joints[: f + 1, 0, 0], -joints[: f + 1, 0, 2], color="tab:blue", linewidth=2)
                for j, p in enumerate(PARENTS):
                    if p >= 0:
                        axis.plot(joints[f, [p, j], 0], -joints[f, [p, j], 2], color="black", linewidth=1.4)
                axis.scatter(joints[f, 15, 0], -joints[f, 15, 2], color="tab:orange", s=18, zorder=5)
                axis.set(xlim=(centre[0] - half, centre[0] + half), ylim=(-centre[1] - half, -centre[1] + half), xticks=[], yticks=[])
                axis.set_aspect("equal")
                axis.set_title(f"{f / reply['fps']:.0f} s", fontsize=8)
            row[0].set_ylabel("\n".join(reply["text"][i : i + 22] for i in range(0, len(reply["text"]), 22)), fontsize=8)
        figure.tight_layout()
        figure.savefig(args.out / "sheet.png", dpi=80)
        print(f"wrote {args.out / 'sheet.png'}", flush=True)


if __name__ == "__main__":
    main()
