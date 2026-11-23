#!/usr/bin/env python3
"""The body motion of every recording as one continuous track, next to the head track of gray/tracks.py.

    python -m gray.body --out <folder>

The feature cache holds the body per 8 s window, each window encoded for itself. Moments
(gray/moments.py) begin at any frame, so this encodes each recording once, from its motion capture,
in the same representation as the cache (code of the STEER repository):

    SMPL joints and rotations at 30 per second  ->  272 numbers per frame (to_actionplan.features_272)
    ->  MoLingo's motion autoencoder  ->  32 numbers per image frame (15 per second)

and keeps the root path outside the latent, as the cache does (egoprimo/root.py), but in the
ground frame of the tracks instead of a window's own frame.

    <out>/<sequence id>.npz
        latent   (N, 32) f32   the pose latent of each image frame
        pelvis   (N, 3) f32    pelvis position (x, height above the floor, z) in the ground frame of the track
        yaw      (N,) f32      root yaw: the body faces (-sin yaw, 0, cos yaw)
        valid    (N,) bool     motion capture exists here and for the 2 s before (the encoder looks back; it does not look ahead)

Two things differ from the cache. The floor under the body is the track's floor, smoothed
over 20 s, where the cache took the lowest joint of each window; so the heights in the latent do
not jump between windows. And nothing is anchored to a window: gray.frames.anchor and
``channels`` below move the root path to the frame at "now", as for the head.

Check printed per run, on some recordings: the latent against the cache's for the same frames
(they agree where the two floors do), and the distance from the head joint to the camera.

A recording that is done is skipped, so the job can be submitted again until all are done.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import torch

from gray.features import CACHE
from gray.pinhole import STEER
from gray.rectify import SEQUENCES

OUT = Path("/ptmp/enazaren/gray/body")
VIDEO_OFFSET = 0.89  # as egoprimo/preprocess.py: video time = device time - first trajectory timestamp + this
LOOK_BACK = 30  # image frames the encoder is given before a frame counts as valid
HEAD_JOINT = 15


def smooth(floor: np.ndarray) -> np.ndarray:
    """The track's floor height without its steps: median over 20 s, then mean over 5 s."""
    from scipy.ndimage import median_filter, uniform_filter1d

    return uniform_filter1d(median_filter(floor.astype(np.float64), size=301, mode="nearest"), size=75, mode="nearest")


def features(job: tuple) -> tuple[str, np.ndarray | None, dict]:
    """The 272 numbers per 30 fps frame of one recording, its root path, and where motion capture exists."""
    name, cache, tracks, sequences = job
    torch.set_num_threads(1)
    sys.path.insert(0, str(STEER))
    from data.nymeria import to_actionplan as ap

    try:
        meta = json.loads((Path(cache) / name / "meta.json").read_text())
        track = np.load(Path(tracks) / f"{name}.npz")
        count = len(track["floor"])
        with np.load(Path(sequences) / name / "body" / "xdata_smpl_neutral.npz") as raw:
            times = raw["timestamps"].astype(np.float64) / 1e6
            betas = torch.from_numpy(raw["betas"][0]).float()
            wanted = np.arange(2 * count) / 30.0 + meta["trajectory_t0"] - VIDEO_OFFSET - meta["clock_offset_s"]  # in the motion capture's clock
            index = np.clip(np.searchsorted(times, wanted, side="right") - 1, 0, len(times) - 1)
            inside = (wanted >= times[0]) & (wanted <= times[-1])
            rows, back = np.unique(index, return_inverse=True)
            aa = torch.cat([torch.from_numpy(raw["global_orient"][rows]).float()[:, None], torch.from_numpy(raw["body_pose"][rows]).float().reshape(-1, 23, 3)], dim=1)
            transl = torch.from_numpy(raw["transl"][rows]).float()
        rest = ap.rest_joints(np.load(ap.MODEL), betas)
        joints, rotations = [], []
        for i in range(0, len(rows), 4000):
            local = ap.rodrigues(aa[i : i + 4000].reshape(-1, 3)).reshape(-1, 24, 3, 3)
            position = ap.forward_kinematics(local, transl[i : i + 4000], rest)
            joints.append(torch.einsum("ij,tkj->tki", ap.TO_YUP, position[:, : ap.JOINTS]))
            turned = local[:, : ap.JOINTS].clone()
            turned[:, 0] = ap.TO_YUP @ local[:, 0]
            rotations.append(turned)
        joints, rotations = torch.cat(joints)[back], torch.cat(rotations)[back]  # (2 N, 22, 3), (2 N, 22, 3, 3)
        floor = torch.from_numpy(np.repeat(smooth(track["floor"]), 2)).float()
        # as to_actionplan.features_272, with the floor given per frame instead of the lowest joint of the clip
        grounded = joints.clone()
        grounded[:, :, 1] -= floor[:, None]
        root = grounded[:, 0]
        local_pos = grounded.clone()
        local_pos[:, :, 0] -= grounded[:, :1, 0]
        local_pos[:, :, 2] -= grounded[:, :1, 2]
        yaw = -torch.atan2(rotations[:, 0, 0, 2], rotations[:, 0, 2, 2])
        heading = ap.rot_yaw(yaw)
        local_pos = torch.einsum("tij,tkj->tki", heading, local_pos)
        feat = torch.zeros(len(joints), 272)
        feat[0, 2] = feat[0, 6] = 1.0
        feat[1:, :2] = torch.einsum("tij,tj->ti", heading[:-1], root[1:] - root[:-1])[:, [0, 2]]
        feat[1:, 2:8] = ap.matrix_to_rotation_6d(ap.rot_yaw(yaw[1:] - yaw[:-1]))
        feat[:, 8:74] = local_pos.reshape(len(joints), -1)
        feat[1:, 74:140] = (local_pos[1:] - local_pos[:-1]).reshape(len(joints) - 1, -1)
        rotations[:, 0] = heading @ rotations[:, 0]
        feat[:, 140:] = ap.matrix_to_rotation_6d(rotations).reshape(len(joints), -1)
        pose = track["pose"]
        head = (grounded[::2, HEAD_JOINT] - torch.from_numpy(np.stack([pose[:, 0], pose[:, 1] + track["floor"] - smooth(track["floor"]), pose[:, 2]], 1))).norm(dim=1).numpy()
        both = inside[::2] & inside[1::2] & track["valid"]
        return name, feat.numpy(), {"pelvis": root[::2].numpy().astype(np.float32), "yaw": yaw[::2].numpy().astype(np.float32), "inside": both,
                                    "head_to_camera": float(np.median(head[both])) if both.any() else float("nan")}
    except Exception as error:  # noqa: BLE001  one odd recording must not end the run
        return name, None, {"error": f"{type(error).__name__}: {error}"}


def channels(pelvis: torch.Tensor, yaw: torch.Tensor, origin: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """The root path (..., T, 4) as the cache has it (pelvis x and z, sin and cos of the root yaw), in the frame with this origin (..., 2) on the ground and heading (...,).

    ``heading`` is the yaw of the frame's +z axis, as gray.frames.heading gives for a camera.
    """
    cosine, sine = torch.cos(heading)[..., None], torch.sin(heading)[..., None]
    x, z = pelvis[..., 0] - origin[..., None, 0], pelvis[..., 2] - origin[..., None, 1]
    psi = yaw - heading[..., None]
    return torch.stack([cosine * x + sine * z, -sine * x + cosine * z, torch.sin(psi), torch.cos(psi)], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--sequences", type=Path, default=SEQUENCES)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(STEER))
    from egoprimo.codec import MotionCodec

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = MotionCodec().to(device)
    names = [p.stem for p in sorted(args.tracks.glob("*.npz")) if not (args.out / f"{p.stem}.npz").exists()][: args.limit]
    print(f"{len(names)} recordings to do on {device}", flush=True)
    done = 0
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for name, feat, extra in pool.imap_unordered(features, [(n, str(args.cache), str(args.tracks), str(args.sequences)) for n in names], chunksize=1):
            if feat is None:
                print(f"{name}: left out, {extra['error']}", flush=True)
                continue
            piece, margin = 3600, 240  # 30 fps frames; the encoder only looks back
            latent = []
            for start in range(0, len(feat), piece):
                first = max(0, start - margin)
                out = codec.encode(torch.from_numpy(feat[first : start + piece]).to(device)[None])[0]
                latent.append(out[(start - first) // 2 :].float().cpu())
            latent = torch.cat(latent).numpy()
            inside = extra["inside"]
            run = np.concatenate([[0], np.cumsum(~inside)])
            valid = inside & (np.arange(len(inside)) >= LOOK_BACK) & (run[np.arange(len(inside)) + 1] - run[np.maximum(np.arange(len(inside)) - LOOK_BACK, 0)] == 0)
            note = ""
            if done % 50 == 0:  # against the cache, away from the start of its windows
                starts, cached = np.load(args.cache / name / "starts.npy"), np.load(args.cache / name / "latents.npy", mmap_mode="r")
                picks = [w for w in range(0, len(starts), max(1, len(starts) // 20)) if valid[starts[w] // 2 + 20 : starts[w] // 2 + 120].all()]
                gaps = [float(np.sqrt(((np.array(cached[w][20:]) - latent[starts[w] // 2 + 20 : starts[w] // 2 + 120]) ** 2).mean())) for w in picks]
                if gaps:
                    note = f"; against the cache's latents (size {float(np.sqrt((latent[valid] ** 2).mean())):.2f}): difference median {np.median(gaps):.2f}, smallest {min(gaps):.2f}, largest {max(gaps):.2f}"
            temporary = args.out / f"{name}.partial.npz"
            np.savez(temporary, latent=latent, pelvis=extra["pelvis"], yaw=extra["yaw"], valid=valid)
            temporary.rename(args.out / f"{name}.npz")
            done += 1
            if note or done % 100 == 0:
                print(f"{done}/{len(names)} {name}: {len(latent):,d} frames, {valid.mean():.0%} valid, head joint {extra['head_to_camera'] * 100:.0f} cm from the camera{note}", flush=True)
    print(f"{done} recordings written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
