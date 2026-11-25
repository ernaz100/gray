#!/usr/bin/env python3
"""Our chain against UniEgoMotion on UniEgoMotion's validation clips: generation from the first frame, scored on the same joints.

    python -m gray.versus_uem --ee4d <folder> --uem <folder> --head-run <run> --body-run <run>

<uem> holds what UniEgoMotion's own evaluation uses and writes (gray/ee4d.py for where it comes from):
ee_val_gt_for_evaluation.pkl (the recorded body and camera of 4,405 clips of 8 s at 10 frames per second, in a frame laid
on the first camera pose), preds_ee4d_gen.pkl (the released model's generation from the first image of each clip), and
smplx/SMPLX_NEUTRAL.npz. Both files are read with plain pickle.

For every clip, "now" is its first frame. Our models, trained on UniEgoMotion's training data (--ee4d), are given that one
frame (its image feature and camera pose; no earlier frames, no body) and generate 5 s: the head path (4 steps, half the
noise), then the body along it (4 steps, noise 0.3). The bodies are decoded to 22 joints and moved into the clip's frame
through the camera pose at the first frame, which both sides have. UniEgoMotion's predictions and the recorded body
are turned into the same 22 joints with the SMPL-X model.

Scored every 0.2 s over the first 5 s (and UniEgoMotion also over its 8 s, to check against its own evaluation):

    joints      mean distance of the 22 joints (m): where the body is in the room
    aligned     the same after the best rigid fit per frame: the pose alone
    head, pelvis  distance of those joints

Rows: UniEgoMotion as released; our chain; our body model along the recorded head path (what the body model alone
does); and the recorded body put through our encoder and decoder (what our representation costs: its skeleton has
average proportions, the recorded one the person's).

What differs between the two sides is said in DESIGN.md with the result.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from gray import frames, head, motion
from gray.chain import Decoder, feet
from gray.ee4d import OURS, EeMoments
from gray.joint import to_device
from gray.moments import HISTORY, collate

NOW = HISTORY - 1


def rigid(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Points a (..., J, 3) after the rigid motion that brings them nearest to b."""
    ca, cb = a.mean(-2, keepdim=True), b.mean(-2, keepdim=True)
    u, _s, vt = torch.linalg.svd((a - ca).transpose(-1, -2) @ (b - cb))
    d = torch.sign(torch.det(u @ vt))
    fix = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return (a - ca) @ (u @ fix @ vt) + cb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ee4d", type=Path, required=True)
    parser.add_argument("--uem", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--body-run", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=None)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import smplx

    if getattr(torch.version, "hip", None):
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    with open(args.uem / "ee_val_gt_for_evaluation.pkl", "rb") as handle:
        truth = pickle.load(handle)
    with open(args.uem / "preds_ee4d_gen.pkl", "rb") as handle:
        theirs = pickle.load(handle)
    layer = smplx.SMPLXLayer(str(args.uem / "smplx"), gender="NEUTRAL", use_pca=True, num_pca_comps=12).to(device)

    @torch.no_grad()
    def skeleton(params: dict) -> torch.Tensor:
        """SMPL-X parameters of one clip -> its 22 body joints (80, 22, 3) in the clip's frame."""
        given = {k: params[k].to(device).float() for k in ("global_orient", "body_pose", "betas", "transl")}
        return layer(**given).joints[:, :22]

    data = EeMoments(args.ee4d, "val", jitter=False)
    index = {str(n): i for i, n in enumerate(data.names)}
    keys, starts = [], []
    for key in list(truth)[: args.clips]:
        name, first = key.rsplit("_start_", 1)
        if name in index and key in theirs and int(first) % 2 == 0:
            i, s = index[name], int(first) * 3 // 2
            if data.start[i] + s + 76 <= data.start[i + 1] and len(truth[key]["aria_traj_T"]) > 50:
                keys.append(key)
                starts.append((i, s))
    clips = EeMoments(args.ee4d, "val", starts=starts)
    print(f"{len(keys):,d} of {len(truth):,d} clips are in our copy of the validation data", flush=True)

    planner, head_mean, head_std, _config = head.load(args.head_run, device)
    model, stats, _config = motion.load(args.body_run, device)
    decoder = Decoder(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    ours_at = np.arange(3, 76, 3) - 1  # of our 75 generated frames (15 per second), those at 0.2, 0.4, ..., 5.0 s
    their_at = np.arange(2, 51, 2)  # the same instants at 10 per second
    rows = {name: [] for name in ("UniEgoMotion", "UniEgoMotion, 8 s", "our chain", "our body, recorded head path", "recorded body through our encoder")}
    slides = {"recorded": [], "our chain": []}
    for first in range(0, len(keys), args.batch):
        batch = to_device(collate([clips[i] for i in range(first, min(first + args.batch, len(keys)))]), device)
        past = head.anchored(batch)
        path = head.sample(planner, batch, head_mean, head_std, 4, generator, amp, history=1, temperature=0.5, poses=True)
        chain = motion.sample(model, batch, stats, 4, generator, amp, False, 0.3, head=torch.cat([past[:, :HISTORY], path], dim=1), history=1)
        along = motion.sample(model, batch, stats, 4, generator, amp, False, 0.3, history=1)
        recorded = motion.body_anchored(batch)[:, HISTORY:]
        made = {"our chain": decoder.joints(chain), "our body, recorded head path": decoder.joints(along), "recorded body through our encoder": decoder.joints(recorded)}
        slides["our chain"].append(feet(made["our chain"])[0].cpu())
        slides["recorded"].append(feet(made["recorded body through our encoder"])[0].cpu())
        # our frame (y up, anchored at the camera at now) -> the clip's frame (z up, laid on its first camera pose), through that camera
        here = frames.matrix(past[:, NOW, 3:].double())  # camera axes in our frame, as columns
        for b in range(batch["pose"].shape[0]):
            key = keys[first + b]
            camera = truth[key]["aria_traj_T"].to(device).double()
            turn = camera[0, :3, :3] @ OURS.to(device) @ here[b].T
            shift = camera[0, :3, 3] - turn @ past[b, NOW, :3].double()
            real = skeleton(truth[key]["smpl_params"]).double()
            guess = skeleton(theirs[key]["smpl_params"]).double()

            def score(joints: torch.Tensor, against: torch.Tensor) -> list[float]:
                gap = (joints - against).norm(dim=-1)
                return [float(gap.mean()), float((rigid(joints, against) - against).norm(dim=-1).mean()), float(gap[:, 15].mean()), float(gap[:, 0].mean())]

            both = min(len(real), len(guess))  # some clips at the end of a segment are shorter than 8 s
            rows["UniEgoMotion"].append(score(guess[their_at], real[their_at]))
            rows["UniEgoMotion, 8 s"].append(score(guess[:both], real[:both]))
            for name, joints in made.items():
                rows[name].append(score(joints[b, ours_at].double() @ turn.T + shift, real[their_at]))
        print(f"  {min(first + args.batch, len(keys)):,d} clips", flush=True)

    result = {name: dict(zip(("joints_m", "aligned_m", "head_m", "pelvis_m"), np.mean(values, axis=0).tolist())) for name, values in rows.items()}
    print(f"\n{len(keys):,d} clips, the first 5 s, every 0.2 s, 22 body joints (m)\n  {'':38s} joints  aligned    head  pelvis")
    for name, r in result.items():
        print(f"  {name:38s} {r['joints_m']:6.3f} {r['aligned_m']:8.3f} {r['head_m']:7.3f} {r['pelvis_m']:7.3f}")
    result["foot_slide_cm_s"] = {name: float(torch.cat(values).mean()) for name, values in slides.items()}
    print(f"\ntoes slide while down (gray/chain.py): our chain {result['foot_slide_cm_s']['our chain']:.1f} cm/s, the recorded body through our encoder {result['foot_slide_cm_s']['recorded']:.1f} cm/s")
    (args.body_run / "versus_uem.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.body_run / 'versus_uem.json'}", flush=True)


if __name__ == "__main__":
    main()
