#!/usr/bin/env python3
"""What a robot camera would see from the recorded head position, next to what the Aria camera saw.

    python -m gray.views --out <folder> <sequence id> ...

Three pinhole views of 69 x 42 degrees (the colour camera of the Unitree G1) are cut out of the
fisheye frames with gray/rectify.py:

    along the Aria axis   wherever the person's camera pointed, tilted with the head
    level                 same heading, horizon in the middle of the image, no sideways tilt
    G1 mount              same heading, 47.6 degrees below level as in Unitree's robot description

The last two are held level with gravity, taken from the head path at the time of the frame. Parts
of a view that the fisheye did not record are grey.

Per recording this writes <id>-views.png and prints, over the whole recording, how far the Aria
axis pointed below level and how much of each view the fisheye covers.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from gray.rectify import G1_VIEW, HIGH, SEQUENCES, WIDE, calibration, project, rectify, rotation, to_sensor, to_video, video_frame

G1_MOUNT = 47.6  # degrees below level: d435_joint in unitree_ros g1_29dof.urdf, pitch 0.8308 rad
VIDEO_OFFSET = 0.89  # seconds the video starts before the head path, as in the feature cache
VIEWS = (("along the Aria axis", None, "yellow"), ("level", 0.0, "lime"), (f"G1 mount, {G1_MOUNT:.1f} degrees down", G1_MOUNT, "cyan"))


def downs(sequence: Path, on_glasses: np.ndarray, every: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Seconds into the video and the direction of "down" in the sensor frame, for every ``every``-th pose."""
    seconds, down, first = [], [], None
    with (sequence / "recording_head" / "mps" / "slam" / "closed_loop_trajectory.csv").open() as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if first is None:
                first = float(row["tracking_timestamp_us"])
            if i % every:
                continue
            head = rotation(*(float(row[f"q{n}_world_device"]) for n in "wxyz"))
            gravity = np.array([float(row[f"gravity_{n}_world"]) for n in "xyz"])
            down.append((head @ on_glasses).T @ (gravity / np.linalg.norm(gravity)))
            seconds.append((float(row["tracking_timestamp_us"]) - first) / 1e6 + VIDEO_OFFSET)
    return np.array(seconds), np.array(down)


def edge_angle(lens: np.ndarray, size: int) -> float:
    """Degrees from the optical axis at which the fisheye image reaches the border of the frame."""
    angles = torch.linspace(0.01, 1.4, 2000, dtype=torch.float64)
    rays = torch.stack([torch.sin(angles), torch.zeros_like(angles), torch.cos(angles)], dim=-1)
    radius = project(rays, lens)[:, 0] - lens[1]
    return math.degrees(float(angles[int((radius < size / 2).sum()) - 1]))


def rays(down: np.ndarray | None, pitch: float, wide: int, high: int, clockwise: bool) -> torch.Tensor:
    """Sensor-frame directions (high, wide, 3) of a pinhole view; held level with ``down`` unless it is None."""
    focal = wide / 2 / math.tan(math.radians(G1_VIEW) / 2)
    v, u = torch.meshgrid(torch.arange(high, dtype=torch.float64), torch.arange(wide, dtype=torch.float64), indexing="ij")
    x, y = (u - (wide - 1) / 2) / focal, (v - (high - 1) / 2) / focal
    if down is None:
        return to_sensor(torch.stack([x, y, torch.ones_like(x)], dim=-1), clockwise)
    d = torch.from_numpy(down / np.linalg.norm(down))
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    ahead = axis - (axis @ d) * d  # the optical axis laid flat: the heading
    ahead = ahead / ahead.norm()
    right = torch.linalg.cross(d, ahead)
    c, s = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return x[..., None] * right + y[..., None] * (c * d - s * ahead) + (c * ahead + s * d)


def covered(directions: torch.Tensor, edge: float) -> torch.Tensor:
    """Which directions the fisheye recorded."""
    return directions[..., 2] / directions.norm(dim=-1) > math.cos(math.radians(edge))


def figure(name: str, sequence: Path, target: Path, times: list[float], scale: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lens, on_glasses = calibration(sequence)
    seconds, down = downs(sequence, on_glasses)
    clockwise = bool(down[:, 0].mean() > 0)
    video = sequence / "preview_rgb.mp4"
    size = video_frame(video, 1.0).shape[0]
    edge = edge_angle(lens, size)
    below = np.degrees(np.arcsin(np.clip(down[:, 2], -1, 1)))
    print(f"{name}: {len(down):,d} poses over {seconds[-1] / 60:.0f} min; the fisheye reaches {edge:.0f} degrees from its axis "
          f"({2 * edge:.0f} degrees across); the Aria axis points below level by {np.median(below):.0f} degrees (median), "
          f"{np.quantile(below, 0.1):.0f} to {np.quantile(below, 0.9):.0f} (10 % to 90 %)", flush=True)
    for label, pitch in (("level", 0.0), ("20 degrees down", 20.0), ("30 degrees down", 30.0), ("G1 mount", G1_MOUNT)):
        share = np.array([covered(rays(d, pitch, 40, 22, clockwise), edge).double().mean().item() for d in down])
        print(f"  {label:16s} the fisheye covers {share.mean():4.0%} of the view on average, all of it in {np.mean(share == 1):4.0%} "
              f"of the poses, less than half in {np.mean(share < 0.5):4.0%}", flush=True)

    fig, axes = plt.subplots(len(times), 4, figsize=(22, 3.5 * len(times)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, WIDE / HIGH, WIDE / HIGH, WIDE / HIGH]})
    for row, at in zip(axes, times):
        frame = video_frame(video, at)
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1)[None].double()
        d = down[int(np.argmin(np.abs(seconds - at)))]
        row[0].imshow(frame)
        row[0].set_title(f"Aria fisheye at {int(at // 60)}:{at % 60:04.1f}, {2 * edge:.0f} degrees across", fontsize=9, loc="left")
        for axis, (label, pitch, colour) in zip(row[1:], VIEWS):
            directions = rays(None if pitch is None else d, pitch or 0.0, WIDE * scale, HIGH * scale, clockwise)
            where = to_video(project(directions, lens), size, clockwise)
            seen = covered(directions, edge)
            image = rectify(tensor, torch.where(seen[..., None], where, torch.full_like(where, -10.0 * size)))[0].permute(1, 2, 0)
            image[~seen] = 128
            axis.imshow(image.clamp(0, 255).byte().numpy())
            axis.set_title(f"{label} ({colour} outline); the fisheye covers {seen.double().mean():.0%}", fontsize=9, loc="left")
            border = torch.cat([where[0], where[:, -1], where[-1].flip(0), where[:, 0].flip(0)])
            inside = torch.cat([seen[0], seen[:, -1], seen[-1].flip(0), seen[:, 0].flip(0)])
            border = torch.where(inside[:, None], border, torch.full_like(border, float("nan"))).numpy()
            row[0].plot(border[:, 0], border[:, 1], color=colour, linewidth=1.3)
        row[0].set(xlim=(0, size - 1), ylim=(size - 1, 0))
        for axis in row:
            axis.axis("off")
    fig.suptitle(f"{name}: pinhole views of {G1_VIEW:.0f} x 42 degrees cut out of the fisheye frame", fontsize=11)
    fig.tight_layout()
    fig.savefig(target / f"{name}-views.png", dpi=75)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--times", type=float, nargs="+", default=[30, 150, 300, 450, 600, 750], help="seconds into the video")
    parser.add_argument("--scale", type=int, default=2, help="the figures show the views at this multiple of 400 x 224")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.sequences:
        figure(name, args.videos / name, args.out, args.times, args.scale)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
