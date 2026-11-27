#!/usr/bin/env python3
"""Our chain on UniEgoMotion's validation clips, written as the files UniEgoMotion's own evaluation reads: the paper's protocol.

    python -m gray.as_uem write --ee4d <folder> --uem <folder> --head-run <run> --body-run <run> --out <folder>
    python eval/compute_3d_metrics.py --EXP_PATH <out> --EVAL_TASK recon      # in UniEgoMotion's repository; also gen, fore
    python -m gray.as_uem table <out> [<another folder> ...]                    # needs joblib, as their script

The paper (arXiv 2508.01126, tables 1 and 2) scores all 4,405 validation clips of 2.1 to 8 s at 10 frames per second:

    reconstruction   given the whole camera path and all images     scored over the whole clip
    forecasting      given the camera path and images of 2 s        scored from 2 to 4 s
    generation       given the first camera pose and image          scored from 0 to 2 s

Its "MPJPE" is the 22 body joints (``mpjpe_body`` in its script; hands are left out), "PA" the same after a fit with
scale per frame. Its script writes every measure per clip and per stretch of 2 s, which is where the windows come from.

**write** runs the models of gray/flex.py on every clip, the window starting at the clip's first frame, with what each
task gives (the table at the top of gray/flex.py; no body is given in any of them), and writes <out>/preds_ee4d_<task>.pkl
as UniEgoMotion's ``save_uem_preds.py`` does: per clip the SMPL-X parameters and the camera path, in the clip's frame.

    body      our decoder's joint rotations are SMPL-X's (gray/ee4d.py made the latents from them), so they are written
              as they are; the pelvis is put where ours is. Shape: the average one (betas 0). Hands: flat.
    height    with ``--ankles`` the average body is then raised or lowered as a whole, once per clip, until its
              ankles are as high as the decoder's own joint positions have them (median over the clip), plus ``LOW``.
              Their foot slide counts a foot as down by its height, and the average body hung from the right pelvis
              stands about 2 cm low. It brings the slide to the recorded level and costs 1 to 2 cm of MPJPE (DESIGN.md)
    frames    ours are 15 per second, decoded at 30; a clip's frame k is our decoded frame 3 k
    place     our frame is anchored at the window's first camera pose, which the clip also has: that gives the turn
              and the shift between the two
    camera    the head path the body was made along (recorded in reconstruction, generated otherwise)

``--codec`` writes the recorded body through our encoder and decoder as <out>/codec/preds_ee4d_recon.pkl: what the
representation and the average shape cost before any model.

**table** prints the paper's rows from the metric files in the given folders.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from gray import flex, frames
from gray.ee4d import OURS, EeWindows
from gray.joint import to_device
from gray.moments import collate

OPENING = {"gen": 1, "fore": int(flex.SECONDS_GIVEN * flex.RATE)}  # frames of ours a task gives; reconstruction gives all
LOW = 0.011  # m: the frozen autoencoder gives the pelvis back this much lower than it was given (300 training segments, 27 Nov)
SCORED = {"recon": "", "gen": "_seg_0", "fore": "_seg_20"}  # the stretch of a clip the paper scores, as their script names it


def write(args) -> None:
    import smplx

    from gray.chain import Decoder

    if getattr(torch.version, "hip", None):
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    with open(args.uem / "ee_val_gt_for_evaluation.pkl", "rb") as handle:
        truth = pickle.load(handle)
    index = {str(name): i for i, name in enumerate(EeWindows(args.ee4d, "val", starts=[]).names)}
    keys = list(truth)[: args.clips]
    missing = [key for key in keys if key.rsplit("_start_", 1)[0] not in index]
    if missing:
        sys.exit(f"{len(missing)} clips are from segments that are not in {args.ee4d} (python -m gray.ee4d convert --split val), such as {missing[0]}")
    clips = EeWindows(args.ee4d, "val", starts=[(index[key.rsplit("_start_", 1)[0]], int(key.rsplit("_start_", 1)[1]) * 3 // 2) for key in keys])
    planner, head_stats, _config = flex.load(args.head_run, device)
    model, body_stats, _config = flex.load(args.body_run, device)
    decoder = Decoder(device)
    layer = smplx.SMPLXLayer(str(args.uem / "smplx"), gender="NEUTRAL", use_pca=True, num_pca_comps=12)
    rest_pelvis = layer().joints[0, 0].double()  # of the average shape
    ankles = [7, 8]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def in_clip_frame(body: torch.Tensor, head: torch.Tensor, exists: torch.Tensor, some: list[str]) -> dict:
        """Bodies (B, L, 36) and the head paths they go with (B, L, 9), in our anchored frame -> UniEgoMotion's record per clip."""
        size = len(some)
        decoded = decoder.ap.streamer272_to_smpl(decoder.codec.decode_with_root(body[..., :32].float(), body[..., 32:].float()).cpu())
        turns = decoder.ap.rodrigues(decoded["poses"].double().reshape(size, -1, 22, 3))  # (B, 2 L, 22, 3, 3)
        # the decoder starts with the pelvis over the origin and yaw 0; the root channels say where it really starts (egoprimo.root.place)
        begin = body[:, 0, 32:].double().cpu()
        yaw = torch.atan2(begin[:, 2], begin[:, 3])
        c, s, zero, one = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
        about_up = torch.stack([torch.stack([c, zero, -s], -1), torch.stack([zero, one, zero], -1), torch.stack([s, zero, c], -1)], -2)
        orient = about_up[:, None] @ turns[:, :, 0]
        pelvis = decoded["joints"][:, :, 0].double() @ about_up.transpose(-1, -2)
        pelvis[..., 0] += begin[:, None, 0]
        pelvis[..., 2] += begin[:, None, 1]
        head, count = head.double().cpu(), exists.sum(1).cpu()
        out = {}
        for b, key in enumerate(some):
            camera = truth[key]["aria_traj_T"].double()
            n, last = len(camera), int(count[b]) - 1
            here = frames.matrix(head[b, :, 3:])  # the camera's axes in our frame
            turn = camera[0, :3, :3] @ OURS @ here[0].T  # our frame -> the clip's, through the first camera pose
            shift = camera[0, :3, 3] - turn @ head[b, 0, :3]
            at = (3 * torch.arange(n)).clamp(max=2 * last + 1)  # decoded frames, 30 per second
            flat = torch.eye(3).expand(n, 15, 3, 3)
            params = {"global_orient": turn @ orient[b, at], "body_pose": turns[b, at, 1:], "betas": torch.zeros(n, 10), "left_hand_pose": flat,
                      "right_hand_pose": flat, "transl": pelvis[b, at] @ turn.T + shift - rest_pelvis}
            if args.ankles:
                given = {name: params[name].float() for name in ("global_orient", "body_pose", "betas", "transl")}
                decoded_ankles = (decoded["joints"][b, at][:, ankles].double() @ about_up[b].T + torch.stack([begin[b, 0], zero[b], begin[b, 1]])) @ turn.T + shift
                with torch.no_grad():
                    params["transl"][:, 2] += (decoded_ankles[..., 2] - layer(**given).joints[:, ankles, 2]).median() + LOW  # the clip's frame has z up
            token = 1.5 * torch.arange(n)  # head poses are 15 per second: every other frame of the clip lies between two
            low, high = token.floor().long().clamp(max=last), token.ceil().long().clamp(max=last)
            between = (head[b, low] + head[b, high]) / 2
            path = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
            path[:, :3, :3] = turn @ frames.matrix(between[:, 3:]) @ OURS.T
            path[:, :3, 3] = between[:, :3] @ turn.T + shift
            out[key] = {"smpl_params": {name: value.float().clone() for name, value in params.items()}, "aria_traj_T": path.float()}
        return out

    records: dict[str, dict] = {task: {} for task in args.tasks} | ({"codec": {}} if args.codec else {})
    for start in range(0, len(keys), args.batch):
        some = keys[start : start + args.batch]
        batch = to_device(collate([clips[i] for i in range(start, start + len(some))]), device)
        exists = batch["exists"]
        recorded = flex.metres(flex.values(batch, head_stats)["head"], head_stats, "head")
        for task in args.tasks:
            if task == "recon":
                path, shown = recorded, exists
            else:
                shown = flex.first(exists, OPENING[task])
                path = flex.sample(planner, batch, head_stats, {"clean": shown, "shown": shown}, args.steps, generator, amp, args.head_noise)
            given = {"clean": torch.zeros_like(exists), "shown": shown, "led": exists}
            body = flex.sample(model, batch, body_stats, given, args.steps, generator, amp, args.body_noise, path=path)
            records[task] |= in_clip_frame(body, path, exists, some)
        if args.codec:
            records["codec"] |= in_clip_frame(flex.metres(flex.values(batch, body_stats)["body"], body_stats, "body"), recorded, exists, some)
        print(f"{start + len(some):,d} of {len(keys):,d} clips", flush=True)
    for task, record in records.items():
        target = args.out / "codec" / "preds_ee4d_recon.pkl" if task == "codec" else args.out / f"preds_ee4d_{task}.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            pickle.dump(record, handle)
        print(f"wrote {target}")
    print(f"sampling: {args.steps} steps, starting noise {args.head_noise:g} (head path) and {args.body_noise:g} (body)", flush=True)


def table(args) -> None:
    import joblib

    columns = (("MPJPE", "mpjpe_body", 100), ("PA", "mpjpe_body_pa", 100), ("head", "head_trans_error", 100), ("camera", "traj_trans_error", 100), ("slide", "sliding_pred", 1))
    print(f"{'':44s} clips " + " ".join(f"{name:>7s}" for name, _key, _scale in columns) + "   (cm; slide as their script gives it)")
    for folder in args.folders:
        for task, stretch in SCORED.items():
            file = folder / f"metrics_3d_ee4d_{task}.pkl"
            if file.is_file():
                m = joblib.load(file)
                print(f"{str(folder)[-34:]:34s} {task:5s}{stretch:7s} {len(m['mpjpe']):5d} " + " ".join(f"{np.nanmean(m[key + stretch]) * scale:7.2f}" for _name, key, scale in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    w = steps.add_parser("write")
    w.add_argument("--ee4d", type=Path, required=True)
    w.add_argument("--uem", type=Path, required=True, help="ee_val_gt_for_evaluation.pkl and smplx/, as gray/versus_uem.py")
    w.add_argument("--head-run", type=Path, required=True)
    w.add_argument("--body-run", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--tasks", nargs="+", default=list(SCORED), choices=tuple(SCORED))
    w.add_argument("--codec", action="store_true", help="also the recorded body through our encoder and decoder")
    w.add_argument("--ankles", action="store_true", help="set the body's height by the decoded ankles instead of the decoded pelvis")
    w.add_argument("--steps", type=int, default=20)
    w.add_argument("--head-noise", type=float, default=1.0, help="starting noise of the head path (1: a full sample)")
    w.add_argument("--body-noise", type=float, default=1.0)
    w.add_argument("--clips", type=int, default=None)
    w.add_argument("--batch", type=int, default=256)
    w.add_argument("--seed", type=int, default=0)
    t = steps.add_parser("table")
    t.add_argument("folders", nargs="+", type=Path)
    args = parser.parse_args()
    {"write": write, "table": table}[args.step](args)


if __name__ == "__main__":
    main()
