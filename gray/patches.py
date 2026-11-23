#!/usr/bin/env python3
"""A coarse grid of image features of the pinhole view, for the frames that can be "now" in the navigation subset.

    python -m gray.patches --part 0 --parts 16
    python -m gray.patches --all --out <folder> --part 0 --parts 16     # every frame of every recording
    python -m gray.patches --limit 4500 <sequence id>       # a test: only the first 4500 video features

gray/pinhole.py keeps one vector per frame (the class token), which says little about where things
are in the image. This keeps the patch tokens of the same DINOv3 pass: 14 x 14 patches of the
224 x 224 pinhole view, averaged 2 x 2 down to 7 x 7 cells of 384 numbers.

That is 49 times more per frame (37.6 KB). By default only the frames that training on windows
can use as "now" are kept: the first ``--history`` (30) frames of every window of the navigation
subset (171 GB). With ``--all`` every frame of every recording is kept (about 670 GB), which any
setup can use, the moments of gray/moments.py included.

    <out>/<id>/patches.npy   (M, 49, 384) f16, cells row by row from the top left of the image
    <out>/<id>/rows.npy      (N,) i32: for each of the N frames of the cache's dino.npy its row in patches.npy, -1 if not kept
    <out>/<id>/meta.json     counts, and a check: the class token of the same pass against gray/pinhole.py's (cosine near 1)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gray.features import CACHE
from gray.pinhole import FINE, IMAGE, STEER, VIEW, frames_of
from gray.rectify import SEQUENCES, calibration, lookup, mounting, rectify

OUT = Path("/ptmp/enazaren/gray/patches")
GRID = 7


@torch.no_grad()
def encode(name: str, windows: np.ndarray, args, backbone, device: str) -> dict:
    began = time.time()
    folder = args.cache / name
    starts = np.load(folder / "starts.npy")
    count = len(np.load(folder / "dino.npy", mmap_mode="r"))
    every = round(json.loads((folder / "meta.json").read_text())["video_fps"] / 15)
    wanted = np.full(count, args.all, dtype=bool)
    for window in windows:
        first = int(starts[window]) // 2
        wanted[first : first + args.history] = True
    if args.limit is not None:
        wanted[args.limit :] = False
    rows = np.full(count, -1, dtype=np.int32)
    rows[wanted] = np.arange(int(wanted.sum()))
    target = args.out / name
    target.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(target / "patches.tmp.npy", mode="w+", dtype=np.float16, shape=(int(wanted.sum()), GRID * GRID, 384))
    reference = np.load(args.pinhole / name / "dino.npy", mmap_mode="r")
    sequence = args.videos / name
    lens, on_glasses = calibration(sequence)
    clockwise = bool(mounting(sequence, on_glasses)[0] > 0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    where, done, cosines = None, 0, []
    last = int(np.flatnonzero(wanted)[-1]) + 1 if wanted.any() else 0
    for frames in frames_of(sequence / "preview_rgb.mp4", every, last, args.batch):
        keep = wanted[done : done + len(frames)]
        if keep.any():
            picked = frames[torch.from_numpy(keep)].to(device).float()
            if where is None:
                where = lookup(lens, picked.shape[-1], clockwise, view=VIEW, wide=IMAGE * FINE, high=IMAGE * FINE).to(device).float()
            view = torch.nn.functional.avg_pool2d(rectify(picked, where), FINE)
            features = backbone.forward_features((view / 255 - mean) / std)
            tokens = features["x_norm_patchtokens"].float()  # (b, 196, 384), rows of 14 patches from the top
            side = int(round(tokens.shape[1] ** 0.5))
            grid = torch.nn.functional.avg_pool2d(tokens.reshape(-1, side, side, tokens.shape[-1]).permute(0, 3, 1, 2), side // GRID)
            index = rows[done : done + len(frames)][keep]
            out[index] = grid.flatten(2).permute(0, 2, 1).half().cpu().numpy()
            if len(cosines) < 8:
                stored = torch.from_numpy(np.array(reference[np.flatnonzero(keep) + done], dtype=np.float32)).to(device)
                cosines.append(torch.nn.functional.cosine_similarity(features["x_norm_clstoken"].float(), stored, dim=-1).cpu())
        done += len(frames)
    out.flush()
    del out
    (target / "patches.tmp.npy").rename(target / "patches.npy")
    np.save(target / "rows.npy", rows)
    result = {"sequence_id": name, "frames": count, "kept": int(wanted.sum()), "windows": int(len(windows)), "grid": GRID, "history": args.history,
              "complete": args.limit is None, "cosine_class_token_to_pinhole": float(torch.cat(cosines).mean()) if cosines else None,
              "seconds": round(time.time() - began)}
    (target / "meta.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sequences", nargs="*", help="default: every recording of the subset")
    parser.add_argument("--subset", type=Path, default=Path("/ptmp/enazaren/gray/runs/census/navigation.npz"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--pinhole", type=Path, default=Path("/ptmp/enazaren/gray/pinhole"))
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--all", action="store_true", help="keep every frame of every recording of the cache")
    parser.add_argument("--history", type=int, default=30, help="frames from the start of a window that can be now")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(STEER))
    from data.nymeria.ego_encoder import DINOV3_REPO, DINOV3_WEIGHTS, _load_dinov3

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = _load_dinov3(DINOV3_REPO, DINOV3_WEIGHTS).to(device).eval()
    listed = np.load(args.subset)
    by_sequence: dict[str, list[int]] = {}
    for name, window in zip(listed["sequence"].tolist(), listed["window"].tolist()):
        by_sequence.setdefault(name, []).append(window)
    everything = sorted(p.parent.name for p in args.cache.glob("*/dino.npy")) if args.all else sorted(by_sequence)
    names = args.sequences or everything[args.part :: args.parts]
    print(f"{len(names)} recordings on {device}", flush=True)
    for i, name in enumerate(names):
        finished = args.out / name / "meta.json"
        if finished.exists() and not args.redo and json.loads(finished.read_text())["complete"]:
            continue
        r = encode(name, np.array(by_sequence.get(name, []), dtype=np.int64), args, backbone, device)
        print(f"{i + 1}/{len(names)} {name}: kept {r['kept']:,d} of {r['frames']:,d} frames for {r['windows']} windows in {r['seconds']} s; "
              f"class token against the pinhole features {r['cosine_class_token_to_pinhole']}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
