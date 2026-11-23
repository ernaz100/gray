#!/usr/bin/env python3
"""The head path of every recording as one continuous track, and where its narration sentences begin.

    python -m gray.tracks --out <folder>

The feature cache stores head poses per 8 s window. For training samples that may start at any
frame (gray/moments.py) this script writes the whole path instead, one pose per image feature
(15 per second), read from the recording's trajectory in the same way the cache did.

    <out>/<sequence id>.npz
        pose       (N, 9) f32   camera position (x, height above the floor, z) and rotation (first two rows of the
                                camera-to-ground matrix), in a ground frame with +y up; gray.frames.anchor moves it to any frame
        floor      (N,) f32     height of the floor under each frame in the recording's world frame
        valid      (N,) bool    the frame lies within the recorded trajectory
        sentences  (S, 2) i32   first and last frame of each narration sentence, in the order of the cache's sentences.json

The ground frame is the recording's world frame turned upright: ground (x, y, z) = world (x, z - floor, -y),
so a ground position is world (x, -z, y + floor). ``to_world`` does this.

The floor height comes from the cached windows (camera height in the world minus the cached
height above the floor at the window's first frame) and is held until the next window starts.

Check, printed per run: the track cut into the cached windows and anchored at their first frame
must reproduce the cache's head.npy.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from gray import frames
from gray.features import CACHE, RATE
from gray.layout import VIDEO_OFFSET, camera_on_glasses, trajectory
from gray.rectify import SEQUENCES

OUT = Path("/ptmp/enazaren/gray/tracks")
UPRIGHT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # world (+z up) to ground (+y up)


def to_world(position: np.ndarray, floor: np.ndarray | float) -> np.ndarray:
    """Ground positions (..., 3) as (x, height, z) -> the recording's world frame (..., 3), +z up."""
    return np.stack([position[..., 0], -position[..., 2], position[..., 1] + floor], axis=-1)


def one(job: tuple[str, str, str, str]) -> dict:
    name, cache, videos, out = job
    try:
        folder, sequence = Path(cache) / name, Path(videos) / name
        times, position, turn = trajectory(sequence)
        on_glasses, offset = camera_on_glasses(sequence)
        count = len(np.load(folder / "dino.npy", mmap_mode="r"))
        at = times[0] - VIDEO_OFFSET + np.arange(count) / RATE
        rows = np.clip(np.searchsorted(times, at), 0, len(times) - 1)
        camera = position[rows] + turn[rows] @ offset
        upright = UPRIGHT @ (turn[rows] @ on_glasses)
        starts, head = np.load(folder / "starts.npy"), np.load(folder / "head.npy")
        first = starts // 2
        floor_of_window = camera[first, 2] - head[:, 0, 1]
        floor = floor_of_window[np.clip(np.searchsorted(first, np.arange(count), side="right") - 1, 0, len(first) - 1)]
        pose = np.concatenate([camera[:, :1], (camera[:, 2] - floor)[:, None], -camera[:, 1:2], upright[:, 0], upright[:, 1]], axis=1).astype(np.float32)
        spoken = json.loads((folder / "sentences.json").read_text())
        to_frame = lambda t: int(round((t - times[0] + VIDEO_OFFSET) * RATE))  # noqa: E731
        sentences = np.array([[to_frame(s["start"]), to_frame(s["end"])] for s in spoken], dtype=np.int32).reshape(-1, 2)
        np.savez_compressed(Path(out) / f"{name}.npz", pose=pose, floor=floor.astype(np.float32), valid=(at >= times[0]) & (at <= times[-1]), sentences=sentences)
        # the check: cached windows from the track
        picked = np.arange(0, len(first), max(1, len(first) // 40))
        cut = torch.from_numpy(np.stack([pose[f : f + head.shape[1]] for f in first[picked]]))
        mine = frames.anchor(cut, torch.zeros(len(picked), dtype=torch.long)).numpy()
        theirs = head[picked].astype(np.float32)
        return {"name": name, "frames": count, "sentences": len(sentences), "position_cm": float(np.abs(mine[..., [0, 2]] - theirs[..., [0, 2]]).max() * 100),
                "rotation": float(np.abs(mine[..., 3:] - theirs[..., 3:]).max()), "height_cm": float(np.abs(mine[:, 0, 1] - theirs[:, 0, 1]).max() * 100)}
    except Exception as error:  # noqa: BLE001
        return {"name": name, "failed": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="only the first recordings (a test)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.parent.name for p in args.cache.glob("*/head.npy"))[: args.limit]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(one, [(n, str(args.cache), str(args.videos), str(args.out)) for n in names], chunksize=2))
    done = [r for r in results if "failed" not in r]
    for r in results:
        if "failed" in r:
            print(f"failed {r['name']}: {r['failed']}")
    print(f"tracks for {len(done)} of {len(names)} recordings: {sum(r['frames'] for r in done):,d} frames ({sum(r['frames'] for r in done) / RATE / 3600:.0f} h), "
          f"{sum(r['sentences'] for r in done):,d} sentences in {sum(r['sentences'] > 0 for r in done)} recordings")
    print(f"against the cached windows: position differs by at most {max(r['position_cm'] for r in done):.2f} cm, the height at the window start by "
          f"{max(r['height_cm'] for r in done):.2f} cm, the rotation values by {max(r['rotation'] for r in done):.4f}")
    (args.out / "summary.json").write_text(json.dumps(sorted(results, key=lambda r: r["name"]), indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
