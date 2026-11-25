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

``--tasks fore recon`` does the same for UniEgoMotion's other two tasks. Forecasting: it is shown the clip's first 2 s
(camera path and images); ours is given the same 2 s (no body) and the 5 s after them are scored. Reconstruction: it is
given the whole camera path and all images; the row to compare with is our body model along the recorded head path, which
is given that path and the first frame's image only.

Rows: UniEgoMotion as released; our chain; our body model along the recorded head path (what the body model alone
does); and the recorded body put through our encoder and decoder (what our representation costs: its skeleton has
average proportions, the recorded one the person's).

What differs between the two sides is said in DESIGN.md with the result.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
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
    parser.add_argument("--steps", type=int, default=4, help="sampling steps of our two models")
    parser.add_argument("--head-noise", type=float, default=0.5, help="starting noise of the head path (1: a full sample)")
    parser.add_argument("--body-noise", type=float, default=0.3, help="starting noise of the body")
    parser.add_argument("--clips", type=int, default=None)
    parser.add_argument("--videos", type=Path, default=None, help="with the task gen: write the bodies of a few clips here, for gray/versus_scene.py floor")
    parser.add_argument("--video-clips", type=int, default=6, help="half of them among clips where the recorded pelvis goes 1 m or further")
    parser.add_argument("--tasks", nargs="+", default=["gen"], choices=("gen", "fore", "recon"), help="UniEgoMotion's tasks: generation, forecasting, reconstruction")
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
    layer = smplx.SMPLXLayer(str(args.uem / "smplx"), gender="NEUTRAL", use_pca=True, num_pca_comps=12).to(device)

    @torch.no_grad()
    def skeleton(params: dict) -> torch.Tensor:
        """SMPL-X parameters of one clip -> its 22 body joints (80, 22, 3) in the clip's frame."""
        given = {k: params[k].to(device).float() for k in ("global_orient", "body_pose", "betas", "transl")}
        return layer(**given).joints[:, :22]

    @torch.no_grad()
    def surface(params: dict) -> torch.Tensor:
        given = {k: params[k].to(device).float() for k in ("global_orient", "body_pose", "betas", "transl")}
        return layer(**given).vertices

    if args.videos is not None:
        from gray.pinhole import STEER
        from gray.scene_worker import Mesh, about_up

        sys.path.insert(0, str(STEER))
        from data.nymeria import to_actionplan as ap

        shape = Mesh(ap.MODEL, ap.rodrigues)
        wanted = {True: args.video_clips // 2, False: args.video_clips - args.video_clips // 2}
        notes = []

    data = EeMoments(args.ee4d, "val", jitter=False)
    index = {str(n): i for i, n in enumerate(data.names)}
    planner, head_mean, head_std, _config = head.load(args.head_run, device)
    model, stats, _config = motion.load(args.body_run, device)
    decoder = Decoder(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = {}
    for task in args.tasks:
        # generation and reconstruction: now is the clip's first frame. Forecasting: UniEgoMotion is shown the clip's first 2 s
        # (camera path and images), so now is the frame at 2 s and ours is given the same 2 s (no body); the 5 s after it are scored.
        lead = 20 if task == "fore" else 0
        with open(args.uem / f"preds_ee4d_{task}.pkl", "rb") as handle:
            theirs = pickle.load(handle)
        keys, starts = [], []
        for key in list(truth)[: args.clips]:
            name, first = key.rsplit("_start_", 1)
            if name in index and key in theirs and int(first) % 2 == 0:
                i, s = index[name], (int(first) + lead) * 3 // 2
                if data.start[i] + s + 76 <= data.start[i + 1] and len(truth[key]["aria_traj_T"]) > lead + 50 and len(theirs[key]["smpl_params"]["transl"]) > lead + 50:
                    keys.append(key)
                    starts.append((i, s))
        clips = EeMoments(args.ee4d, "val", starts=starts)
        history = HISTORY if task == "fore" else 1
        ours_at = np.arange(3, 76, 3) - 1  # of our 75 generated frames (15 per second), those 0.2, 0.4, ..., 5.0 s after now
        their_at = lead + np.arange(2, 51, 2)  # the same instants at 10 per second
        names = ["UniEgoMotion", "our chain", "our body, recorded head path", "recorded body through our encoder", "standing still (the body at now, held)"]
        rows = {name: [] for name in names}
        moved = {name: [] for name in ("recorded", "UniEgoMotion", "our chain")}
        slides = {"recorded": [], "our chain": []}
        for first in range(0, len(keys), args.batch):
            batch = to_device(collate([clips[i] for i in range(first, min(first + args.batch, len(keys)))]), device)
            past = head.anchored(batch)
            path = head.sample(planner, batch, head_mean, head_std, args.steps, generator, amp, history=history, temperature=args.head_noise, poses=True)
            chain = motion.sample(model, batch, stats, args.steps, generator, amp, False, args.body_noise, head=torch.cat([past[:, :HISTORY], path], dim=1), history=history)
            along = motion.sample(model, batch, stats, args.steps, generator, amp, False, args.body_noise, history=history)
            recorded = motion.body_anchored(batch)[:, HISTORY:]
            made = {"our chain": decoder.joints(chain), "our body, recorded head path": decoder.joints(along), "recorded body through our encoder": decoder.joints(recorded)}
            slides["our chain"].append(feet(made["our chain"])[0].cpu())
            slides["recorded"].append(feet(made["recorded body through our encoder"])[0].cpu())
            here = frames.matrix(past[:, NOW, 3:].double())  # camera axes in our frame, as columns
            whole_truth = motion.body_anchored(batch)
            for b in range(batch["pose"].shape[0]):
                key = keys[first + b]
                camera = truth[key]["aria_traj_T"].to(device).double()
                turn = camera[lead, :3, :3] @ OURS.to(device) @ here[b].T  # our frame -> the clip's frame, through the camera at now
                shift = camera[lead, :3, 3] - turn @ past[b, NOW, :3].double()
                real = skeleton(truth[key]["smpl_params"]).double()
                guess = skeleton(theirs[key]["smpl_params"]).double()

                def score(joints: torch.Tensor, against: torch.Tensor) -> list[float]:
                    gap = (joints - against).norm(dim=-1)
                    return [float(gap.mean()), float((rigid(joints, against) - against).norm(dim=-1).mean()), float(gap[:, 15].mean()), float(gap[:, 0].mean())]

                travel = lambda joints: float((joints[1:, 0, :2] - joints[:-1, 0, :2]).norm(dim=-1).sum())  # noqa: E731  on the ground (z is up here)
                stretch = np.append(lead, their_at)
                rows["UniEgoMotion"].append(score(guess[their_at], real[their_at]))
                rows["standing still (the body at now, held)"].append(score(real[lead : lead + 1].expand(len(their_at), -1, -1), real[their_at]))
                moved["recorded"].append(travel(real[stretch]))
                moved["UniEgoMotion"].append(travel(guess[stretch]))
                for name, joints in made.items():
                    placed = joints[b, ours_at].double() @ turn.T + shift
                    rows[name].append(score(placed, real[their_at]))
                    if name == "our chain":
                        moved[name].append(travel(placed))
                goes = moved["recorded"][-1] >= 1.0
                if args.videos is not None and task == "gen" and wanted[goes] > 0 and (first + b) % 37 == 0:  # spread over the clips
                    wanted[goes] -= 1
                    work = args.videos / f"clip-{len(notes)}"
                    work.mkdir(parents=True, exist_ok=True)
                    whole = torch.cat([whole_truth[b : b + 1, :HISTORY], chain[b : b + 1]], dim=1).float()
                    smpl = ap.streamer272_to_smpl(decoder.codec.decode_with_root(whole[..., :32], whole[..., 32:]).cpu())
                    begin = whole[0, 0, 32:].cpu()
                    skin = about_up(shape.skin(smpl["poses"][0].double().numpy(), smpl["joints"][0, :, 0].double().numpy()), float(torch.atan2(begin[2], begin[3])), *begin[:2].tolist())[2 * HISTORY :]
                    own = skin[:, shape.joints4]
                    mine = ((np.einsum("tvkab,vb->tvka", own[..., :3], shape.vertices) + own[..., 3]) * shape.weights4[None, :, :, None]).sum(2)
                    mine = (torch.from_numpy(mine).to(device) @ turn.T + shift).cpu().numpy()
                    at = torch.arange(1, 151, device=device) / 3  # 30 per second, in their frames of 10 per second
                    low, part = at.floor().long().clamp(max=49), (at - at.floor().long().clamp(max=49))[:, None, None]
                    for k, (label, verts) in enumerate((("recorded", surface(truth[key]["smpl_params"])), ("ours", None), ("UniEgoMotion", surface(theirs[key]["smpl_params"])))):
                        np.save(work / f"verts-{k}.npy", mine.astype(np.float32) if verts is None else (verts[low] * (1 - part) + verts[low + 1] * part).cpu().numpy().astype(np.float32))
                        np.save(work / f"faces-{k}.npy", shape.faces if verts is None else layer.faces.astype(np.int64))
                    notes.append({"clip": key, "floor": float(real[..., 2].min()), "bodies": {
                        "recorded": {"travel_m": moved["recorded"][-1]}, "ours": {"travel_m": moved["our chain"][-1], "joints_cm": rows["our chain"][-1][0] * 100},
                        "UniEgoMotion": {"travel_m": moved["UniEgoMotion"][-1], "joints_cm": rows["UniEgoMotion"][-1][0] * 100}}})
                    (args.videos / "clips.json").write_text(json.dumps(notes, indent=1))
            print(f"  {task}: {min(first + args.batch, len(keys)):,d} clips", flush=True)
        given = {"gen": "from the first frame", "fore": "from the first 2 s (camera path and images); the 5 s after them", "recon": "UniEgoMotion is given the whole camera path and all images"}[task]
        table = {name: dict(zip(("joints_m", "aligned_m", "head_m", "pelvis_m"), np.mean(values, axis=0).tolist())) for name, values in rows.items()}
        print(f"\n{task}: {given}. {len(keys):,d} clips, 5 s, every 0.2 s, 22 body joints (m)\n  {'':46s} joints  aligned    head  pelvis")
        for name, r in table.items():
            print(f"  {name:46s} {r['joints_m']:6.3f} {r['aligned_m']:8.3f} {r['head_m']:7.3f} {r['pelvis_m']:7.3f}")
        table["pelvis_travel_m"] = {name: {"mean": float(np.mean(values)), "over_1m": float(np.mean(np.array(values) >= 1.0))} for name, values in moved.items()}
        print("  how far the pelvis travels in the 5 s: " + "; ".join(f"{name} {r['mean']:.2f} m, 1 m or more in {r['over_1m']:.0%}" for name, r in table["pelvis_travel_m"].items()))
        table["foot_slide_cm_s"] = {name: float(torch.cat(values).mean()) for name, values in slides.items()}
        print(f"  toes slide while down: our chain {table['foot_slide_cm_s']['our chain']:.1f} cm/s, recorded {table['foot_slide_cm_s']['recorded']:.1f} cm/s", flush=True)
        table["clips"] = len(keys)
        result[task] = table
    target = args.body_run / f"versus_uem-{'-'.join(args.tasks)}-steps-{args.steps}-noise-{args.head_noise:g}-{args.body_noise:g}.json"
    target.write_text(json.dumps(result, indent=1))
    print(f"\nour sampling: {args.steps} steps, starting noise {args.head_noise:g} (head path) and {args.body_noise:g} (body)\nwrote {target}", flush=True)


if __name__ == "__main__":
    main()
