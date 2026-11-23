#!/usr/bin/env python3
"""Turn Nymeria's fisheye video frames into the view of a normal (pinhole) camera, and show both.

    python -m gray.rectify --out <folder> <sequence id> ...

The Aria RGB camera is a fisheye of about 110 degrees. A robot camera such as the RealSense in the
head of the Unitree G1 is a pinhole camera of about 69 x 42 degrees. With the lens calibration that
is stored with every recording, each pixel of such a pinhole image is looked up in the fisheye image.

Two frames of reference for directions seen by the camera, both with z along the optical axis:

    sensor   as the calibration describes it; the sensor is mounted turned by a quarter turn
    upright  as the stored video shows it: x to the right, y down

Which way the sensor is turned is measured from gravity (``mounting``) rather than assumed.

Per recording this writes <id>-rectified.png: the fisheye frame with the outline of the cut-out,
the cut-out along the optical axis, and the cut-out looking further down by ``--pitch`` degrees.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

SEQUENCES = Path("/ptmp/enazaren/nymeria/sequences")
G1_VIEW = 69.0  # degrees left to right of the RealSense D435i colour camera
WIDE, HIGH = 400, 224  # 25 x 14 patches of 16 pixels; gives 42 degrees top to bottom


def rotation(w: float, x: float, y: float, z: float) -> np.ndarray:
    """The rotation matrix of a unit quaternion."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def calibration(sequence: Path) -> tuple[np.ndarray, np.ndarray]:
    """The RGB camera's 15 lens values and its rotation on the glasses, from the first calibration record."""
    with (sequence / "recording_head" / "mps" / "slam" / "online_calibration.jsonl").open() as handle:
        record = json.loads(handle.readline())
    for camera in record["CameraCalibrations"]:
        if camera["Label"] == "camera-rgb":
            if camera["Projection"]["Name"] != "FisheyeRadTanThinPrism":
                raise ValueError(camera["Projection"]["Name"])
            w, xyz = camera["T_Device_Camera"]["UnitQuaternion"]
            return np.array(camera["Projection"]["Params"]), rotation(w, *xyz)
    raise KeyError("camera-rgb")


def project(rays: torch.Tensor, lens: np.ndarray) -> torch.Tensor:
    """Sensor pixels (..., 2) hit by directions (..., 3) in the sensor frame (Aria's FisheyeRadTanThinPrism)."""
    focal, cx, cy = lens[:3]
    radial, (p0, p1), (s0, s1, s2, s3) = lens[3:9], lens[9:11], lens[11:15]
    a, b = rays[..., 0] / rays[..., 2], rays[..., 1] / rays[..., 2]
    r = torch.hypot(a, b)
    theta = torch.atan(r)
    bent, power = torch.ones_like(theta), theta**2
    for k in radial:
        bent = bent + k * power
        power = power * theta**2
    scale = torch.where(r > 1e-8, theta * bent / r.clamp_min(1e-8), torch.ones_like(r))
    x, y = a * scale, b * scale
    r2 = x * x + y * y
    tangent = 2 * (x * p0 + y * p1)
    u = x * (1 + tangent) + r2 * p0 + s0 * r2 + s1 * r2 * r2
    v = y * (1 + tangent) + r2 * p1 + s2 * r2 + s3 * r2 * r2
    return torch.stack([focal * u + cx, focal * v + cy], dim=-1)


def mounting(sequence: Path, on_glasses: np.ndarray, rows: int = 200_000, every: int = 200) -> np.ndarray:
    """The average direction of "down" in the sensor frame over the first rows of the head path."""
    down = []
    with (sequence / "recording_head" / "mps" / "slam" / "closed_loop_trajectory.csv").open() as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if i >= rows:
                break
            if i % every:
                continue
            head = rotation(*(float(row[f"q{n}_world_device"]) for n in "wxyz"))
            gravity = np.array([float(row[f"gravity_{n}_world"]) for n in "xyz"])
            down.append((head @ on_glasses).T @ (gravity / np.linalg.norm(gravity)))
    return np.mean(down, axis=0)


def to_sensor(upright: torch.Tensor, clockwise: bool) -> torch.Tensor:
    """Directions in the upright frame, expressed in the sensor frame."""
    x, y, z = upright.unbind(-1)
    return torch.stack([y, -x, z] if clockwise else [-y, x, z], dim=-1)


def to_video(pixels: torch.Tensor, size: int, clockwise: bool) -> torch.Tensor:
    """Sensor pixels as pixels of the upright video."""
    u, v = pixels.unbind(-1)
    return torch.stack([size - 1 - v, u] if clockwise else [v, size - 1 - u], dim=-1)


def lookup(lens: np.ndarray, size: int, clockwise: bool, pitch: float = 0.0, view: float = G1_VIEW,
           wide: int = WIDE, high: int = HIGH) -> torch.Tensor:
    """For every pixel of the pinhole image (high, wide), the pixel of the upright video to read, (high, wide, 2).

    The pinhole camera looks along the fisheye's optical axis, turned down by ``pitch`` degrees.
    """
    focal = wide / 2 / math.tan(math.radians(view) / 2)
    v, u = torch.meshgrid(torch.arange(high, dtype=torch.float64), torch.arange(wide, dtype=torch.float64), indexing="ij")
    x, y, z = (u - (wide - 1) / 2) / focal, (v - (high - 1) / 2) / focal, torch.ones_like(u)
    c, s = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    rays = torch.stack([x, y * c + z * s, -y * s + z * c], dim=-1)
    return to_video(project(to_sensor(rays, clockwise), lens), size, clockwise)


def rectify(frames: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
    """Pinhole images (B, 3, high, wide) read from upright fisheye frames (B, 3, size, size) at ``where``."""
    size = frames.shape[-1]
    grid = (where.to(frames) / (size - 1) * 2 - 1).expand(len(frames), -1, -1, -1)
    return torch.nn.functional.grid_sample(frames, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def video_frame(video: Path, seconds: float) -> np.ndarray:
    """One frame of the video at its own resolution, (size, size, 3) u8."""
    import imageio_ffmpeg

    width, height = next(imageio_ffmpeg.read_frames(str(video)))["size"]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(video),
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    data = subprocess.run(command, capture_output=True, check=True, timeout=300).stdout
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def figure(name: str, sequence: Path, target: Path, times: list[float], pitch: float, scale: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lens, on_glasses = calibration(sequence)
    down = mounting(sequence, on_glasses)
    clockwise = bool(down[0] > 0)
    below = math.degrees(math.asin(float(np.clip(down[2], -1, 1))))
    print(f"{name}: focal {lens[0]:.0f}, centre {lens[1]:.0f} / {lens[2]:.0f}; down in the sensor frame "
          f"({down[0]:+.2f}, {down[1]:+.2f}, {down[2]:+.2f}): the video is the sensor image turned "
          f"{'clockwise' if clockwise else 'anticlockwise'}; the optical axis points {below:.0f} degrees below level on average", flush=True)

    video = sequence / "preview_rgb.mp4"
    pitches = (0.0, pitch)
    fig, axes = plt.subplots(len(times), 3, figsize=(17, 3.6 * len(times)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, WIDE / HIGH, WIDE / HIGH]})
    for row, seconds in zip(axes, times):
        frame = video_frame(video, seconds)
        size = frame.shape[0]
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1)[None].double()
        row[0].imshow(frame)
        row[0].set_title(f"fisheye frame at {int(seconds // 60)}:{seconds % 60:04.1f}, {size} x {size}", fontsize=9, loc="left")
        for axis, angle, colour in zip(row[1:], pitches, ("yellow", "cyan")):
            where = lookup(lens, size, clockwise, angle, wide=WIDE * scale, high=HIGH * scale)
            axis.imshow(rectify(tensor, where)[0].permute(1, 2, 0).clamp(0, 255).byte().numpy())
            axis.set_title(f"pinhole {G1_VIEW:.0f} degrees wide, looking {angle:.0f} degrees further down ({colour} outline)", fontsize=9, loc="left")
            edge = torch.cat([where[0], where[:, -1], where[-1].flip(0), where[:, 0].flip(0)]).numpy()
            row[0].plot(edge[:, 0], edge[:, 1], color=colour, linewidth=1.2)
            covered = (where[..., 0].max() - where[..., 0].min()).item(), (where[..., 1].max() - where[..., 1].min()).item()
        row[0].set(xlim=(0, size - 1), ylim=(size - 1, 0))
        for axis in row:
            axis.axis("off")
    print(f"  the last cut-out covers {covered[0]:.0f} x {covered[1]:.0f} fisheye pixels for {WIDE} x {HIGH} pinhole pixels", flush=True)
    fig.suptitle(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(target / f"{name}-rectified.png", dpi=80)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--times", type=float, nargs="+", default=[30, 150, 300, 450, 600, 750], help="seconds into the video")
    parser.add_argument("--pitch", type=float, default=20.0, help="degrees the second cut-out looks further down")
    parser.add_argument("--scale", type=int, default=2, help="the figures show the cut-out at this multiple of 400 x 224")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.sequences:
        figure(name, args.videos / name, args.out, args.times, args.pitch, args.scale)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
