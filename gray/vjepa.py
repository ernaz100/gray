#!/usr/bin/env python3
"""V-JEPA 2 features of the pinhole view: one video clip of the 2 s up to "now", for every frame that can be now.

    python -m gray.vjepa --part 0 --parts 16
    python -m gray.vjepa --limit 3 <sequence id>          # a test: only the first 3 moments of one recording

DINOv3 (gray/pinhole.py, gray/patches.py) encodes each frame on its own; the model has to work out
from 30 separate vectors how the scene moves. V-JEPA 2 encodes a clip, so every token already
carries the 2 s before now. This replaces both DINOv3 inputs by one pass over that clip:

    clip     16 frames: now - 30, now - 28, ..., now (2 s at 7.5 frames per second), the same
             pinhole view as gray/pinhole.py (79.6 degrees), here at 256 x 256
    tokens   V-JEPA 2 ViT-L (facebook/vjepa2-vitl-fpc64-256): 8 time slices (2 frames each) x 16 x 16 patches x 1024

    <out>/<id>/patches.npy   (M, 49, 1024) f16: the last time slice (the frames now - 2 and now) averaged to 7 x 7 cells,
                             row by row from the top left; read as the image grid at now (``--patches``)
    <out>/<id>/slices.npy    (M, 8, 1024) f16: each time slice averaged over the image, oldest first; read in place of
                             the 30 per-frame vectors (``--images``, see gray.moments.Moments)
    <out>/<id>/rows.npy      (N,) i32: for each of the N frames of the cache's dino.npy its row in the two files, -1 if not kept
    <out>/<id>/meta.json     counts and timing

The frames kept are those that can be now in training or evaluation: for every moment of the
navigation subset (training and held out), its now moved by -5 to +15 frames (gray.moments.EARLY, LATE).
The model then sees exactly the same 2 s as with DINOv3, only encoded together.
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
from gray.moments import EARLY, HISTORY, LATE
from gray.pinhole import VIEW, frames_of
from gray.rectify import SEQUENCES, calibration, lookup, mounting, rectify

OUT = Path("/ptmp/enazaren/gray/vjepa")
MODEL = "facebook/vjepa2-vitl-fpc64-256"
SIZE, FINE, GRID = 256, 4, 7
FRAMES, STRIDE = 16, 2  # the clip: 16 frames, every second one, ending at now: spans HISTORY = 30 frames before now
SLICES = FRAMES // 2


def clip_frames(now: int) -> np.ndarray:
    return np.maximum(now - STRIDE * (FRAMES - 1) + STRIDE * np.arange(FRAMES), 0)


@torch.no_grad()
def encode(name: str, nows: np.ndarray, args, model, device: str) -> dict:
    began = time.time()
    folder = args.cache / name
    count = len(np.load(folder / "dino.npy", mmap_mode="r"))
    every = round(json.loads((folder / "meta.json").read_text())["video_fps"] / 15)
    nows = np.unique(nows[(nows >= 0) & (nows < count)])
    if args.limit is not None:
        nows = nows[: args.limit]
    rows = np.full(count, -1, dtype=np.int32)
    rows[nows] = np.arange(len(nows))
    needed = np.zeros(count, dtype=bool)
    for now in nows:
        needed[clip_frames(now)] = True
    target = args.out / name
    target.mkdir(parents=True, exist_ok=True)
    grids = np.lib.format.open_memmap(target / "patches.tmp.npy", mode="w+", dtype=np.float16, shape=(len(nows), GRID * GRID, 1024))
    slices = np.lib.format.open_memmap(target / "slices.tmp.npy", mode="w+", dtype=np.float16, shape=(len(nows), SLICES, 1024))
    sequence = args.videos / name
    lens, on_glasses = calibration(sequence)
    clockwise = bool(mounting(sequence, on_glasses)[0] > 0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    where, store, done, pending, shape = None, {}, 0, 0, None

    def flush(upto: int) -> None:
        """Encode the clips of every pending now whose last frame has been decoded."""
        nonlocal pending, shape
        while pending < len(nows) and nows[pending] < upto:
            chunk = [n for n in nows[pending : pending + args.batch] if n < upto]
            video = torch.stack([torch.stack([store[int(f)] for f in clip_frames(n)]) for n in chunk]).to(device).float()  # (b, 16, 3, S, S)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                tokens = model(pixel_values_videos=(video / 255 - mean) / std, skip_predictor=True).last_hidden_state.float()
            side = SIZE // 16
            tokens = tokens.reshape(len(chunk), SLICES, side, side, -1)  # time, height, width
            if shape is None:
                shape = list(tokens.shape)
            last = tokens[:, -1].permute(0, 3, 1, 2)
            grid = torch.nn.functional.adaptive_avg_pool2d(last, GRID).flatten(2).permute(0, 2, 1)
            index = rows[np.array(chunk)]
            grids[index] = grid.half().cpu().numpy()
            slices[index] = tokens.mean((2, 3)).half().cpu().numpy()
            pending += len(chunk)
            oldest = int(clip_frames(nows[pending])[0]) if pending < len(nows) else count
            for f in [f for f in store if f < oldest]:
                del store[f]

    last_needed = int(nows[-1]) + 1 if len(nows) else 0
    for frames in frames_of(sequence / "preview_rgb.mp4", every, last_needed, 32):
        keep = needed[done : done + len(frames)]
        if keep.any():
            picked = frames[torch.from_numpy(keep)].to(device).float()
            if where is None:
                where = lookup(lens, picked.shape[-1], clockwise, view=VIEW, wide=SIZE * FINE, high=SIZE * FINE).to(device).float()
            views = torch.nn.functional.avg_pool2d(rectify(picked, where), FINE).round().clamp(0, 255).to(torch.uint8).cpu()
            for f, view in zip(np.flatnonzero(keep) + done, views):
                store[int(f)] = view
        done += len(frames)
        flush(done)
    if pending != len(nows):
        raise RuntimeError(f"{name}: decoded {done} frames, encoded {pending} of {len(nows)} clips")
    grids.flush(), slices.flush()
    del grids, slices
    (target / "patches.tmp.npy").rename(target / "patches.npy")
    (target / "slices.tmp.npy").rename(target / "slices.npy")
    np.save(target / "rows.npy", rows)
    result = {"sequence_id": name, "frames": count, "kept": int(len(nows)), "model": MODEL, "size": SIZE, "clip_frames": FRAMES, "stride": STRIDE,
              "tokens": shape, "view_degrees": VIEW, "complete": args.limit is None, "seconds": round(time.time() - began)}
    (target / "meta.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sequences", nargs="*", help="default: every recording with a moment of the navigation subset")
    parser.add_argument("--moments", type=Path, default=Path("/ptmp/enazaren/gray/moments/moments.npz"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--weights", default=MODEL, help="Hugging Face name or local folder of the V-JEPA 2 model")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="only the first frames that can be now of each recording (a test)")
    parser.add_argument("--batch", type=int, default=16, help="clips per pass")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModel

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.weights).to(device).eval()
    table = np.load(args.moments)
    by_sequence: dict[str, list[int]] = {}
    for name, now in zip(table["sequence"][table["subset"]].tolist(), table["now"][table["subset"]].tolist()):
        by_sequence.setdefault(name, []).extend(range(now - EARLY, now + LATE + 1))
    names = args.sequences or sorted(by_sequence)[args.part :: args.parts]
    print(f"{len(names)} recordings on {device}; clip of {FRAMES} frames every {STRIDE} (history {HISTORY})", flush=True)
    for i, name in enumerate(names):
        finished = args.out / name / "meta.json"
        if finished.exists() and not args.redo and json.loads(finished.read_text())["complete"]:
            continue
        r = encode(name, np.array(by_sequence.get(name, []), dtype=np.int64), args, model, device)
        print(f"{i + 1}/{len(names)} {name}: {r['kept']:,d} clips of {r['frames']:,d} frames in {r['seconds']} s, tokens {r['tokens']}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
