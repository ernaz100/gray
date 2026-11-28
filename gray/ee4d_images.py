#!/usr/bin/env python3
"""Our own image features for UniEgoMotion's data, from the Ego-Exo4D videos: DINOv3, one vector and a 7 x 7 grid per frame.

    python -m gray.ee4d_images prepare --ee4d <folder>                        # makes the (empty) files
    python -m gray.ee4d_images encode --ee4d <folder> --part 0 --parts 16     # fills them, take by take

UniEgoMotion releases DINOv2 class tokens at 5 per second and no videos (gray/ee4d.py). With the videos
(``lab22 download egoexo4d --parts downscaled_takes/448 --views ego``, in <videos>/takes/<take>/frame_aligned_videos/
downscaled/448/aria*_214-1.mp4: the glasses' colour camera, 448 x 448, 30 per second, upright) every frame of ours gets
its own. A segment is named <take>___<first frame>___<last frame>, and its frame j is video frame first + 2 j.

Written next to gray_<split>.npz, row for row the frames of that split:

    gray_<split>_dino3.npy   (N, 384) f16      the class token
    gray_<split>_grid.npy    (N, 49, 384) f16  the patch tokens, 14 x 14 averaged to 7 x 7, rows from the top left

The frame is the fisheye image as recorded, averaged 2 x 2 down to 224 x 224. It is not turned into a pinhole view as
for Nymeria (gray/pinhole.py): the calibration is in a part of the dataset that was not downloaded.
<ee4d>/images_done/<split>/<take>.json marks a finished take, so that a rerun goes on where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gray.ee4d import arrays
from gray.patches import GRID
from gray.pinhole import STEER, frames_of

VIDEOS = Path("/ptmp/enazaren/egoexo4d")
SPLITS = ("val", "train")
WIDTH = 384


def files(folder: Path, split: str, count: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    shapes = {"dino3": (count, WIDTH), "grid": (count, GRID * GRID, WIDTH)}
    if mode == "w+":
        return tuple(np.lib.format.open_memmap(folder / f"gray_{split}_{name}.npy", mode="w+", dtype=np.float16, shape=shape) for name, shape in shapes.items())
    return tuple(np.load(folder / f"gray_{split}_{name}.npy", mmap_mode=mode) for name in shapes)


def prepare(args) -> None:
    for split in SPLITS:
        count = len(arrays(args.ee4d, split)[1])
        if not (args.ee4d / f"gray_{split}_grid.npy").is_file():
            files(args.ee4d, split, count, "w+")
        print(f"{split}: {count:,d} frames")


@torch.no_grad()
def encode(args) -> None:
    sys.path.insert(0, str(STEER))
    from data.nymeria.ego_encoder import DINOV3_REPO, DINOV3_WEIGHTS, _load_dinov3

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = _load_dinov3(DINOV3_REPO, DINOV3_WEIGHTS).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for split in SPLITS:
        *_rest, names, start = arrays(args.ee4d, split)
        dino, grid = files(args.ee4d, split, int(start[-1]), "r+")
        by_take: dict[str, list[int]] = {}
        for i, name in enumerate(names):
            by_take.setdefault(str(name).split("___")[0], []).append(i)
        takes = args.takes or sorted(by_take)[args.part :: args.parts]
        (args.ee4d / "images_done" / split).mkdir(parents=True, exist_ok=True)
        print(f"{split}: {len(takes)} takes on {device}", flush=True)
        for t, take in enumerate(takes):
            finished = args.ee4d / "images_done" / split / f"{take}.json"
            if take not in by_take or (finished.exists() and not args.redo):
                continue
            began = time.time()
            # the rows of this split's arrays and the video frame each of them shows
            rows = np.concatenate([np.arange(start[i], start[i + 1]) for i in by_take[take]])
            shown = np.concatenate([int(str(names[i]).split("___")[1]) + 2 * np.arange(start[i + 1] - start[i]) for i in by_take[take]])
            if args.limit is not None:
                rows, shown = rows[shown < args.limit], shown[shown < args.limit]
            order = np.argsort(shown, kind="stable")
            rows, shown = rows[order], shown[order]
            videos = sorted((args.videos / "takes" / take / "frame_aligned_videos" / "downscaled" / "448").glob("aria*_214-1.mp4"))
            if not videos or not len(rows):
                print(f"{take}: {'no video' if not videos else 'no frames'}", flush=True)
                continue
            done, filled = 0, 0
            for frames in frames_of(videos[0], 1, int(shown[-1]) + 1, args.batch):
                low, high = np.searchsorted(shown, [done, done + len(frames)])
                if high > low:
                    wanted, which = np.unique(shown[low:high] - done, return_inverse=True)  # a frame can be in two segments
                    view = torch.nn.functional.avg_pool2d(frames[torch.from_numpy(wanted)].to(device).float(), frames.shape[-1] // 224)
                    features = backbone.forward_features((view / 255 - mean) / std)
                    tokens = features["x_norm_patchtokens"].float()  # (b, 196, 384), rows of 14 patches from the top
                    side = int(round(tokens.shape[1] ** 0.5))
                    cells = torch.nn.functional.avg_pool2d(tokens.reshape(-1, side, side, WIDTH).permute(0, 3, 1, 2), side // GRID).flatten(2).permute(0, 2, 1)
                    into = np.argsort(rows[low:high], kind="stable")  # write in the order of the file
                    dino[rows[low:high][into]] = features["x_norm_clstoken"].half().cpu().numpy()[which[into]]
                    grid[rows[low:high][into]] = cells.half().cpu().numpy()[which[into]]
                    filled += high - low
                done += len(frames)
            dino.flush()
            grid.flush()
            result = {"take": take, "rows": int(len(rows)), "filled": int(filled), "video_frames_read": done, "complete": args.limit is None and filled == len(rows),
                      "seconds": round(time.time() - began)}
            if result["complete"]:
                finished.write_text(json.dumps(result, indent=1))
            print(f"{split} {t + 1}/{len(takes)} {take}: {filled:,d} of {len(rows):,d} frames in {result['seconds']} s{'' if filled == len(rows) else '  (the video is shorter)'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("prepare", "encode"))
    parser.add_argument("takes", nargs="*", help="default: every take of the split")
    parser.add_argument("--ee4d", type=Path, required=True)
    parser.add_argument("--videos", type=Path, default=VIDEOS)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1, help="do takes part, part + parts, ... of the sorted list")
    parser.add_argument("--limit", type=int, default=None, help="only video frames before this one (a test)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "encode": encode}[args.step](args)


if __name__ == "__main__":
    main()
