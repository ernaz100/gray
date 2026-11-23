#!/usr/bin/env python3
"""Estimated depth of the pinhole view for every frame of every recording.

    python -m gray.depth --part 0 --parts 16
    python -m gray.depth --limit 4500 --out <folder> <sequence id>      # a test

In the test bed (gray/probe.py) estimated depth told free space from blocked space better than
the DINO features at equal data. This computes it for all frames: Depth Anything V2 (metric,
indoor, base size) on the same pinhole view as gray/pinhole.py, at 448 x 448 pixels, averaged
to 28 x 28 cells.

    <out>/<id>/depth.npy   (N, 28, 28) f16, metres as the model gives them, row for row the frames of the cache's dino.npy
    <out>/<id>/meta.json   counts and a check

The values are too large for this camera by a factor that varies from frame to frame (about
0.42 on average against Aria's measured points, gray/cloud.py); they are stored uncorrected.

Check per recording: for the frames that are also in the test bed, the depth is compared with
the one stored there (extracted from the video frame by frame, by other code).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from gray.features import CACHE
from gray.pinhole import VIEW, frames_of
from gray.probe import DEPTH, SET, WEIGHTS
from gray.rectify import SEQUENCES, calibration, lookup, mounting, rectify

OUT = Path("/ptmp/enazaren/gray/depth")
SIDE, CELLS = 448, 28


@torch.no_grad()
def encode(name: str, args, model, device: str) -> dict:
    began = time.time()
    folder = args.cache / name
    count = len(np.load(folder / "dino.npy", mmap_mode="r"))
    if args.limit is not None:
        count = min(count, args.limit)
    every = round(json.loads((folder / "meta.json").read_text())["video_fps"] / 15)
    sequence = args.videos / name
    lens, on_glasses = calibration(sequence)
    clockwise = bool(mounting(sequence, on_glasses)[0] > 0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    where, out = None, []
    for frames in frames_of(sequence / "preview_rgb.mp4", every, count, args.batch):
        frames = frames.to(device).float()
        if where is None:
            where = lookup(lens, frames.shape[-1], clockwise, view=VIEW, wide=2 * SIDE, high=2 * SIDE).to(device).float()
        view = torch.nn.functional.avg_pool2d(rectify(frames, where), 2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            depth = model(pixel_values=(view / 255 - mean) / std).predicted_depth
        out.append(torch.nn.functional.adaptive_avg_pool2d(depth.float()[:, None], CELLS)[:, 0].half().cpu())
    depth = torch.cat(out).numpy()
    if len(depth) != count:
        raise RuntimeError(f"{name}: wanted {count} frames, decoded {len(depth)}")
    result = {"sequence_id": name, "frames": count, "complete": args.limit is None, "model": DEPTH[args.model], "seconds": round(time.time() - began)}
    kept = args.probeset / f"{name}.{args.model}.npy"
    if kept.exists():
        index = np.load(args.probeset / f"{name}.npz")["frame"]
        index_ok = index < count
        if index_ok.any():
            other = np.load(kept).astype(np.float32).reshape(-1, CELLS, CELLS)[index_ok]
            mine = depth[index[index_ok]].astype(np.float32)
            result["against_test_bed"] = float(np.median(np.abs(mine - other) / other))
    target = args.out / name
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "depth.tmp.npy", depth)
    (target / "depth.tmp.npy").rename(target / "depth.npy")
    (target / "meta.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sequences", nargs="*", help="default: every recording of the cache")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--probeset", type=Path, default=SET)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model", default="depth-base", choices=sorted(DEPTH))
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForDepthEstimation

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH[args.model], cache_dir=WEIGHTS).to(device).eval()
    names = args.sequences or sorted(p.parent.name for p in args.cache.glob("*/dino.npy"))[args.part :: args.parts]
    print(f"{len(names)} recordings on {device}, {args.model}", flush=True)
    for i, name in enumerate(names):
        finished = args.out / name / "meta.json"
        if finished.exists() and not args.redo and json.loads(finished.read_text())["complete"]:
            continue
        r = encode(name, args, model, device)
        print(f"{i + 1}/{len(names)} {name}: {r['frames']:,d} frames in {r['seconds']} s; median difference to the test bed's depth "
              f"{r.get('against_test_bed', float('nan')):.1%}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
