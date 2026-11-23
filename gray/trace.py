#!/usr/bin/env python3
"""The comparator's signal along whole recordings, and what its peaks look like.

    python -m gray.trace --run <runs>/features-v1 --out <folder> <sequence id> ...

For every frame of a recording, the pose-aware predictor of features.py gets the second before
"now" (15 frames) and the real head poses after it, and predicts the frame 0.5, 1 and 2 s after
now. The mismatch is its mean squared error against the frame the camera really recorded
(standardized features; 1.0 is the average frame). This is what a comparator would threshold.

Per recording it writes

    <id>.npz          mismatch, copy error, turning and distance for every frame, lead and window kind
    <id>-trace.png    the mismatch over time at a lead of 1 s
    <id>-events.png   pairs of video frames (now, 1 s later) for the largest mismatches, for the
                      largest ones while the head barely moved, and for a few typical frames

The model was trained on whole windows of 120 tokens whose history starts at the window's first
token. To give every frame the same history, windows are cut to 77 tokens here. As a check the
mismatch is also computed from whole windows (history 1 to 30 frames, depending on the frame),
and the two are compared on the frames where only the window length differs; if they disagree by
more than 10 %, the whole-window version is used for the figures.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import torch

from gray import frames
from gray.features import CACHE, RATE, load, prepare

HISTORY = 15
LENGTH = 77
LEADS = (8, 15, 30)  # tokens: 0.53, 1 and 2 s
STEP = 30  # tokens between cached windows
SHOWN = 15  # the lead of the plots and events
CALM_TURN, CALM_DISTANCE = 15.0, 0.25  # degrees and metres: "the head barely moved"
SEQUENCES = Path("/ptmp/enazaren/nymeria/sequences")


@torch.no_grad()
def trace(model, mean, std, folder: Path, device: str) -> dict[str, np.ndarray]:
    """Mismatch for every frame, twice: from cut windows (fixed history) and from whole windows (history 1 to 30)."""
    dino, starts, head = (np.load(folder / f"{n}.npy", mmap_mode="r") for n in ("dino", "starts", "head"))
    count = dino.shape[0]
    names = ("mismatch", "copy", "turn", "distance")
    out = {f"{name}_{mode}_{lead}": np.full(count, np.nan, dtype=np.float32) for name in names for mode in ("cut", "whole") for lead in LEADS}
    shift = torch.arange(STEP, device=device)
    every = torch.arange(STEP, device=device)
    modes = {  # rows of the window used by each of the 30 inputs, and where "now" is in each
        "cut": (shift[:, None] + torch.arange(LENGTH, device=device)[None], torch.full((STEP,), HISTORY - 1, device=device), shift),
        "whole": (torch.arange(120, device=device)[None].expand(STEP, 120), shift, torch.zeros_like(shift)),
    }
    same = []  # frames predicted from the same now and history by both: only the window length differs
    for w in range(len(starts)):
        first = int(starts[w]) // 2
        image = torch.from_numpy(np.array(dino[first : first + 120], dtype=np.float32)).to(device)
        poses = torch.from_numpy(np.array(head[w], dtype=np.float32)).to(device)
        for mode, (index, now, offset) in modes.items():
            target, seen, anchored, seconds = prepare(image[index], poses[index], now + 1, mean, std)
            predicted = model(target, seen, anchored, seconds)
            for lead in LEADS:
                at = now + lead
                rows = (first + offset + at).cpu().numpy()  # the frame each input predicts
                keep = rows < count
                values = {
                    "mismatch": ((predicted[every, at] - target[every, at]) ** 2).mean(-1),
                    "copy": ((target[every, now] - target[every, at]) ** 2).mean(-1),
                    "turn": frames.heading(anchored[every, at, 3:]).abs() * 180 / math.pi,
                    "distance": torch.hypot(anchored[every, at, 0], anchored[every, at, 2]),
                }
                for name, value in values.items():
                    out[f"{name}_{mode}_{lead}"][rows[keep]] = value.cpu().numpy()[keep]
        if first + HISTORY - 1 + SHOWN < count:
            same.append(first + HISTORY - 1 + SHOWN)
    same = np.array(same)
    cut, whole = float(np.nanmean(out[f"mismatch_cut_{SHOWN}"][same])), float(np.nanmean(out[f"mismatch_whole_{SHOWN}"][same]))
    out["check"] = np.array([cut, whole])
    out["mode"] = np.array("cut" if abs(cut / whole - 1) < 0.1 else "whole")
    return out


def peaks(values: np.ndarray, allowed: np.ndarray, count: int, apart: int = 45) -> list[int]:
    """The largest values among the allowed frames, at least ``apart`` frames from each other."""
    score = np.where(allowed & np.isfinite(values), values, -np.inf)
    picked: list[int] = []
    for i in np.argsort(-score):
        if len(picked) == count or not np.isfinite(score[i]):
            break
        if all(abs(int(i) - j) >= apart for j in picked):
            picked.append(int(i))
    return picked


def video_frame(video: Path, seconds: float, size: int = 256) -> np.ndarray:
    """One frame of the video as (size, size, 3); grey when the video cannot be read."""
    import imageio_ffmpeg
    from PIL import Image

    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", f"{max(seconds, 0):.3f}", "-i", str(video),
               "-frames:v", "1", "-vf", f"scale={size}:{size}", "-f", "image2pipe", "-vcodec", "png", "-"]
    try:
        data = subprocess.run(command, capture_output=True, check=True, timeout=120).stdout
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception as error:  # noqa: BLE001
        print(f"  no frame at {seconds:.1f} s of {video.parent.name}: {type(error).__name__}", flush=True)
        return np.full((size, size, 3), 128, dtype=np.uint8)


def narration(folder: Path, seconds: float) -> str:
    """The narrated sentence at this time of the video, if there is one."""
    meta = json.loads((folder / "meta.json").read_text())
    device_time = seconds + meta["trajectory_t0"] - meta["video_offset_s"]
    for sentence in json.loads((folder / "sentences.json").read_text()):
        if sentence["start"] <= device_time <= sentence["end"]:
            return sentence["text"]
    return ""


def figures(name: str, folder: Path, video: Path, out: dict[str, np.ndarray], target: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = str(out["mode"])
    mismatch, copy = out[f"mismatch_{mode}_{SHOWN}"], out[f"copy_{mode}_{SHOWN}"]
    turn, distance = out[f"turn_{mode}_{SHOWN}"], out[f"distance_{mode}_{SHOWN}"]
    valid = np.isfinite(mismatch)
    calm = valid & (turn < CALM_TURN) & (distance < CALM_DISTANCE)
    median = float(np.nanmedian(mismatch))
    groups = {
        "largest mismatch": peaks(mismatch, valid, 8),
        "largest mismatch, head barely moved": peaks(mismatch, calm, 8),
        "typical": peaks(-np.abs(mismatch - median), valid, 4, apart=450),
    }
    seconds = np.arange(len(mismatch)) / RATE

    figure, axis = plt.subplots(figsize=(16, 4))
    axis.plot(seconds / 60, copy, color="0.75", linewidth=0.6, label="copy of the frame 1 s earlier")
    axis.plot(seconds / 60, mismatch, color="tab:blue", linewidth=0.6, label="prediction made 1 s earlier")
    for (label, frames_), colour in zip(groups.items(), ("tab:red", "tab:orange", "tab:green")):
        axis.scatter(seconds[frames_] / 60, mismatch[frames_], color=colour, s=25, zorder=3, label=label)
    axis.set(xlabel="minutes", ylabel="mismatch (1.0 = average frame)", title=name, ylim=(0, None))
    axis.legend(loc="upper right", fontsize=8, ncol=5)
    figure.tight_layout()
    figure.savefig(target / f"{name}-trace.png", dpi=90)
    plt.close(figure)

    cells = [(label, i) for label, frames_ in groups.items() for i in frames_]
    columns = 4
    rows = math.ceil(len(cells) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.6 * rows), squeeze=False)
    events = []
    for axis in axes.flat:
        axis.axis("off")
    for axis, (label, i) in zip(axes.flat, cells):
        now_s, then_s = (i - SHOWN) / RATE, i / RATE
        pair = np.concatenate([video_frame(video, now_s), video_frame(video, then_s)], axis=1)
        text = narration(folder, then_s)
        axis.imshow(pair)
        axis.set_title(
            f"{label}\n{int(then_s // 60)}:{then_s % 60:04.1f}  mismatch {mismatch[i]:.2f}, copy {copy[i]:.2f}, "
            f"turn {turn[i]:.0f}°, {distance[i]:.2f} m\n" + "\n".join(textwrap.wrap(text, 62)[:2]),
            fontsize=8, loc="left",
        )
        events.append({"group": label, "seconds": then_s, "mismatch": float(mismatch[i]), "copy": float(copy[i]),
                       "turn_deg": float(turn[i]), "distance_m": float(distance[i]), "narration": text})
    figure.suptitle(f"{name}: left the frame at now, right the frame 1 s later (the one predicted)", fontsize=11)
    figure.tight_layout()
    figure.savefig(target / f"{name}-events.png", dpi=80)
    plt.close(figure)

    quantile = lambda q, mask: float(np.quantile(mismatch[mask], q)) if mask.any() else float("nan")
    return {
        "frames": int(valid.sum()), "calm_share": float(calm.sum() / max(valid.sum(), 1)),
        "all": {q: quantile(q, valid) for q in (0.5, 0.9, 0.99)}, "calm": {q: quantile(q, calm) for q in (0.5, 0.9, 0.99)},
        "max": float(np.nanmax(mismatch)), "events": events,
        "cut_against_whole": [float(v) for v in out["check"]], "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    args = parser.parse_args()

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std, _config = load(args.run / "full", device)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.sequences:
        folder = args.cache / name
        out = trace(model, mean, std, folder, device)
        np.savez_compressed(args.out / f"{name}.npz", **out)
        summary[name] = s = figures(name, folder, args.videos / name / "preview_rgb.mp4", out, args.out)
        print(
            f"{name}: {s['frames']:,d} frames; mismatch at 1 s: median {s['all'][0.5]:.2f}, 90 % {s['all'][0.9]:.2f}, "
            f"99 % {s['all'][0.99]:.2f}, max {s['max']:.2f}; head barely moved in {s['calm_share']:.0%} of frames: "
            f"median {s['calm'][0.5]:.2f}, 99 % {s['calm'][0.99]:.2f}; same frames from cut windows {s['cut_against_whole'][0]:.3f} "
            f"and whole windows {s['cut_against_whole'][1]:.3f}, using {s['mode']}", flush=True,
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
