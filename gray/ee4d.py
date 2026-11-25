#!/usr/bin/env python3
"""UniEgoMotion's data (EE4D-Motion, from EgoExo4D) in the form of our samples, to train our chain on what it was trained on.

    python -m gray.ee4d convert --data <folder> --steer <folder> --split val     # then --split train

<data>/ee4d/uniegomotion holds what UniEgoMotion releases (ee_<split>.pt, egoview_dinov2_<split>.pt): per segment of a
take, at 10 frames per second, the SMPL-X body, the head trajectory of the glasses and the floor height; per take, DINOv2
features of the egocentric frames at 5 per second. There are no captions.

**convert** writes <data>/ee4d/gray_<split>.npz and gray_<split>_image.npy, all segments end to end at our 15 per second:

    latent       the pose latent, from the same motion autoencoder as gray/body.py. The body is resampled from 10 to 30
                 frames per second (positions linearly, rotations along the shortest arc) and brought into the
                 272-number representation first. SMPL-X's 22 body joints are those of the SMPL body the autoencoder knows.
    pelvis, yaw  the root path, as gray/body.py
    pose         the camera pose, (x, height above the floor, z) and the first two rows of its rotation, as gray/tracks.py.
                 UniEgoMotion stores the glasses' camera with its axes renamed (x left, y up, z ahead); they are named
                 back to ours (x down the image, y to its left, z ahead). The world is turned from z-up to y-up.
    image        (N, 1024) the DINOv2 class token of the frame nearest in time (one every 0.2 s)
    start, names where each segment begins, and its name (<take>___<first frame>___<last frame>)

Checked on the validation split (25 Nov 2026): the camera's heading and the body's facing agree (cosine 0.93, median
14.5 degrees apart); the camera's x axis points down (-0.83); decoding the latents again puts the head joint 12.2 cm from the
camera (11.5 cm in their own joints).

``EeMoments`` gives samples from it shaped like the moments of gray/moments.py: up to 30 frames given, 75 generated, no text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from gray import frames

OURS = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)  # our camera axes in theirs, as columns


def matrix(six: torch.Tensor) -> torch.Tensor:
    """(..., 6), the first two rows of a rotation as UniEgoMotion stores it -> (..., 3, 3)."""
    first = torch.nn.functional.normalize(six[..., :3], dim=-1)
    second = torch.nn.functional.normalize(six[..., 3:] - (first * six[..., 3:]).sum(-1, keepdim=True) * first, dim=-1)
    return torch.stack([first, second, torch.cross(first, second, dim=-1)], dim=-2)


def finer(positions: np.ndarray, turns: np.ndarray, factor: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Positions (T, ..., 3) and rotations (T, J, 3, 3) at 10 per second -> at 30 per second, an even number of frames."""
    from scipy.spatial.transform import Rotation, Slerp

    count = len(positions)
    old, new = np.arange(count), np.arange((count - 1) * factor + 1) / factor
    new = new[: len(new) // 2 * 2]
    low = np.minimum(np.floor(new).astype(int), count - 2)
    part = (new - low).reshape(-1, *([1] * (positions.ndim - 1)))
    fine = positions[low] * (1 - part) + positions[low + 1] * part
    out = np.stack([Slerp(old, Rotation.from_matrix(turns[:, j]))(new).as_matrix() for j in range(turns.shape[1])], axis=1)
    return fine, out


def features_272(ap, joints: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """As to_actionplan.features_272, for joints (T, 22, 3) already above the floor (see gray/body.py)."""
    root = joints[:, 0]
    local_pos = joints.clone()
    local_pos[:, :, 0] -= joints[:, :1, 0]
    local_pos[:, :, 2] -= joints[:, :1, 2]
    yaw = -torch.atan2(rotations[:, 0, 0, 2], rotations[:, 0, 2, 2])
    heading = ap.rot_yaw(yaw)
    local_pos = torch.einsum("tij,tkj->tki", heading, local_pos)
    feat = torch.zeros(len(joints), 272)
    feat[0, 2] = feat[0, 6] = 1.0
    feat[1:, :2] = torch.einsum("tij,tj->ti", heading[:-1], root[1:] - root[:-1])[:, [0, 2]]
    feat[1:, 2:8] = ap.matrix_to_rotation_6d(ap.rot_yaw(yaw[1:] - yaw[:-1]))
    feat[:, 8:74] = local_pos.reshape(len(joints), -1)
    feat[1:, 74:140] = (local_pos[1:] - local_pos[:-1]).reshape(len(joints) - 1, -1)
    turned = rotations.clone()
    turned[:, 0] = heading @ rotations[:, 0]
    feat[:, 140:] = ap.matrix_to_rotation_6d(turned).reshape(len(joints), -1)
    return feat


def convert(args) -> None:
    sys.path.insert(0, str(args.steer))
    from data.nymeria import to_actionplan as ap
    from egoprimo.codec import MotionCodec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = MotionCodec().to(device)
    folder = args.data / "ee4d" / "uniegomotion"
    segments = torch.load(folder / f"ee_{args.split}.pt", map_location="cpu", weights_only=False)
    views = torch.load(folder / f"egoview_dinov2_{args.split}.pt", map_location="cpu", weights_only=False)["feats"]
    up = ap.TO_YUP.double()
    out = {key: [] for key in ("latent", "pose", "pelvis", "yaw", "image")}
    names, start, total, agree = [], [0], 0, []
    for i, (name, seg) in enumerate(list(segments.items())[: args.limit]):
        take, first, _last = name.split("___")
        if take not in views or seg["num_frames"] < 30:
            continue
        floor = float(seg["floor_height"])
        params = seg["smpl_params"]
        local = torch.cat([matrix(params["global_orient"].double())[:, None], matrix(params["body_pose"].double())], dim=1)  # (T, 22, 3, 3)
        joints = seg["kp3d"][:, :22].double()
        camera = matrix(seg["aria_traj"][:, :6].double())[:, None]
        at = seg["aria_traj"][:, 6:9].double()
        positions, turns = finer(torch.cat([joints, at[:, None]], dim=1).numpy(), torch.cat([local, camera], dim=1).numpy())
        positions, turns = torch.from_numpy(positions), torch.from_numpy(turns)
        positions = positions @ up.T  # z-up to y-up
        positions[..., 1] -= floor
        rotations = turns[:, :22].clone()
        rotations[:, 0] = up @ rotations[:, 0]
        feat = features_272(ap, positions[:, :22].float(), rotations.float())
        with torch.no_grad():
            latent = codec.encode(feat[None].to(device))[0].cpu()
        seen = (up @ turns[:, 22] @ OURS)[::2]
        where = positions[::2, 22]
        tokens = len(latent)
        index = ((int(first) + 2 * np.arange(tokens)) // 6).clip(0, len(views[take]) - 1)
        out["latent"].append(latent.numpy().astype(np.float32))
        out["pose"].append(torch.cat([where, seen[:, 0], seen[:, 1]], dim=1).numpy().astype(np.float32))
        out["pelvis"].append(positions[::2, 0].numpy().astype(np.float32))
        out["yaw"].append((-torch.atan2(rotations[::2, 0, 0, 2], rotations[::2, 0, 2, 2])).numpy().astype(np.float32))
        out["image"].append(views[take][index, 0].numpy().astype(np.float16))
        move = where[-1, [0, 2]] - where[0, [0, 2]]
        if move.norm() > 2.0:  # does the camera look where the person goes? a rough check of the axes
            heading = frames.heading(torch.from_numpy(out["pose"][-1][:, 3:]))
            agree.append(float(((-torch.sin(heading)) * move[0] + torch.cos(heading) * move[1]).mean() / move.norm()))
        names.append(name)
        total += tokens
        start.append(total)
        if i % 1000 == 0:
            print(f"{i:,d}/{len(segments):,d}", flush=True)
    target = args.data / "ee4d"
    np.save(target / f"gray_{args.split}_image.npy", np.concatenate(out.pop("image")))
    np.savez(target / f"gray_{args.split}.npz", **{key: np.concatenate(value) for key, value in out.items()}, start=np.array(start), names=np.array(names))
    pose = np.concatenate(out["pose"])
    print(f"{len(names):,d} segments, {total:,d} frames at 15 per second ({total / 15 / 3600:.1f} h); {np.mean(np.diff(start) >= 76):.0%} are 5.1 s or longer")
    print(f"camera {np.median(pose[:, 1]):.2f} m above the floor (median); pelvis {np.median(np.concatenate(out['pelvis'])[:, 1]):.2f} m; "
          f"latent size {float(np.sqrt((np.concatenate(out['latent'])[::20] ** 2).mean())):.2f}")
    print(f"in {len(agree)} segments that travel 2 m or further, the camera's heading has cosine {np.mean(agree):.2f} with the direction of travel")


class EeMoments(torch.utils.data.Dataset):
    """Stretches of EE4D-Motion segments as samples: up to 30 frames before now and the 75 after it; "image" has 1024 numbers per frame; no text.

    ``starts``: (segment, frame) pairs to use as "now" in place of the regular grid (for scoring on given clips).
    """

    def __init__(self, folder: Path, split: str, every: int = 15, jitter: bool = True, starts: list[tuple[int, int]] | None = None) -> None:
        d = np.load(folder / f"gray_{split}.npz")
        self.latent, self.pose, self.pelvis, self.yaw = d["latent"], d["pose"], d["pelvis"], d["yaw"]
        self.image = np.load(folder / f"gray_{split}_image.npy", mmap_mode="r")
        self.names, self.start, self.jitter, self.every = d["names"], d["start"], jitter and starts is None, every
        if starts is not None:
            self.samples = [(i, int(self.start[i]), int(self.start[i + 1]), s) for i, s in starts]
        else:
            self.samples = [(i, int(a), int(b), int(s)) for i, (a, b) in enumerate(zip(self.start[:-1], self.start[1:])) if b - a >= 76 for s in range(0, b - a - 76 + 1, every)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        segment, a, b, s = self.samples[index]
        now = a + min(s + (int(torch.randint(0, self.every, (1,))) if self.jitter else 0), b - a - 76)
        rows = np.clip(np.arange(now - 29, now + 76), a, b - 1)
        own = np.concatenate([self.latent[rows], self.pelvis[rows][:, [0, 2]], np.sin(self.yaw[rows])[:, None], np.cos(self.yaw[rows])[:, None]], axis=1)
        return {"pose": torch.from_numpy(self.pose[rows].copy()), "image": torch.from_numpy(np.array(self.image[rows[:30]], dtype=np.float32)),
                "text": torch.zeros(0, 768), "row": torch.tensor(segment), "now": torch.tensor(now - a), "body": torch.from_numpy(own.astype(np.float32)),
                "body_known": torch.tensor(True), "foreign": torch.tensor(False), "history_max": torch.tensor(min(now - a + 1, 30))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("convert",))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steer", type=Path, default=Path("/u/enazaren/projects/STEER"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.data, args.steer = args.data.expanduser(), args.steer.expanduser()
    convert(args)


if __name__ == "__main__":
    main()
