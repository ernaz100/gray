#!/usr/bin/env python3
"""Image features of Nymeria as a normal (pinhole) camera would see it, one file per recording.

    python -m gray.pinhole --part 0 --parts 16          # every 16th recording, starting at the first
    python -m gray.pinhole --limit 600 <sequence id>    # the first 600 features of one recording

The feature cache holds the DINOv3 class token of the whole fisheye frame, squeezed to 224 x 224.
This writes the same thing for a pinhole view cut out of every frame with gray/rectify.py:

    <out>/<id>/dino.npy    (N, 384) f16, row for row the same video frames as the cache's dino.npy
    <out>/<id>/meta.json   the camera, and two checks (below)

The camera looks along the Aria camera's axis and is 79.6 degrees wide and high (focal length 600
for 1000 pixels), the same as the rectified frames the Loc_08 reconstruction was built from. The
view is cut out at 896 x 896 and averaged 4 x 4 down to 224 x 224, so fine detail does not alias.

Checks on the first frames of every recording: the fisheye frame is also encoded the old way and
compared with the cache (cosine near 1 means the frames and the encoder are the same ones), and
the pinhole feature is compared with the fisheye one (how different the new input is).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gray.features import CACHE
from gray.rectify import SEQUENCES, calibration, lookup, mounting, rectify

OUT = Path("/ptmp/enazaren/gray/pinhole")
STEER = Path("/u/enazaren/projects/STEER")  # for the image encoder the cache was made with
VIEW = 79.6  # degrees, wide and high
IMAGE = 224
FINE = 4  # the view is cut out at IMAGE * FINE and averaged down
CHECKED = 256  # frames per recording that are also encoded the old way


def frames_of(video: Path, every: int, count: int, batch: int):
    """Batches (B, 3, size, size) u8 of every ``every``-th video frame, ``count`` frames in all."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(video)], check=True, capture_output=True, text=True).stdout.strip()
    width, height = (int(v) for v in probe.split(","))
    command = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=not(mod(n\\,{every}))", "-fps_mode", "passthrough",
               "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    size = width * height * 3
    with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=size * 4) as process:
        while True:
            raw = process.stdout.read(size * batch)
            if not raw:
                break
            yield torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, height, width, 3).permute(0, 3, 1, 2)
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {video}")


@torch.no_grad()
def encode(name: str, cache: Path, videos: Path, out: Path, backbone, dino_cls, device: str, limit: int | None, batch: int) -> dict:
    began = time.time()
    old = np.load(cache / name / "dino.npy", mmap_mode="r")
    meta = json.loads((cache / name / "meta.json").read_text())
    every = round(meta["video_fps"] / 15)
    count = len(old) if limit is None else min(limit, len(old))
    sequence = videos / name
    lens, on_glasses = calibration(sequence)
    clockwise = bool(mounting(sequence, on_glasses)[0] > 0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    where, features, same, changed = None, [], [], []
    for frames in frames_of(sequence / "preview_rgb.mp4", every, count, batch):
        frames = frames.to(device).float()
        if where is None:
            where = lookup(lens, frames.shape[-1], clockwise, view=VIEW, wide=IMAGE * FINE, high=IMAGE * FINE).to(device).float()
        view = torch.nn.functional.avg_pool2d(rectify(frames, where), FINE)
        feature = dino_cls(backbone, (view / 255 - mean) / std).float()
        done = sum(len(f) for f in features)
        if done < CHECKED:
            small = torch.nn.functional.interpolate(frames, size=(IMAGE, IMAGE), mode="bicubic", antialias=True, align_corners=False)
            fisheye = dino_cls(backbone, (small.clamp(0, 255) / 255 - mean) / std).float()
            cached = torch.from_numpy(np.array(old[done : done + len(frames)], dtype=np.float32)).to(device)
            same.append(torch.nn.functional.cosine_similarity(fisheye, cached, dim=-1).cpu())
            changed.append(torch.nn.functional.cosine_similarity(feature, fisheye, dim=-1).cpu())
        features.append(feature.half().cpu())
    features = torch.cat(features).numpy()
    if len(features) != count:
        raise RuntimeError(f"{name}: wanted {count} frames, decoded {len(features)}")
    result = {
        "sequence_id": name, "features": count, "complete": limit is None or count == len(old), "view_degrees": VIEW, "image": IMAGE,
        "direction": "along the Aria camera axis", "video_turned_clockwise": clockwise, "lens": [float(v) for v in lens],
        "cosine_fisheye_to_cache": float(torch.cat(same).mean()), "cosine_pinhole_to_fisheye": float(torch.cat(changed).mean()),
        "seconds": round(time.time() - began),
    }
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "dino.tmp.npy", features)
    (folder / "dino.tmp.npy").rename(folder / "dino.npy")
    (folder / "meta.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sequences", nargs="*", help="default: every recording of the cache")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1, help="do recordings part, part + parts, ... of the sorted list")
    parser.add_argument("--limit", type=int, default=None, help="only the first features of each recording (a test)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(STEER))
    from data.nymeria.ego_encoder import DINOV3_REPO, DINOV3_WEIGHTS, _dino_cls, _load_dinov3

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = _load_dinov3(DINOV3_REPO, DINOV3_WEIGHTS).to(device).eval()
    names = args.sequences or sorted(p.parent.name for p in args.cache.glob("*/dino.npy"))[args.part :: args.parts]
    print(f"{len(names)} recordings on {device}", flush=True)
    for i, name in enumerate(names):
        finished = args.out / name / "meta.json"
        if finished.exists() and not args.redo and json.loads(finished.read_text())["complete"]:
            continue
        r = encode(name, args.cache, args.videos, args.out, backbone, _dino_cls, device, args.limit, args.batch)
        print(f"{i + 1}/{len(names)} {name}: {r['features']:,d} features in {r['seconds']} s; fisheye against the cache "
              f"{r['cosine_fisheye_to_cache']:.3f}, pinhole against fisheye {r['cosine_pinhole_to_fisheye']:.3f}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
