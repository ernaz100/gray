#!/usr/bin/env python3
"""Checks of the body model before the loop is built: how to sample it, and whether it holds along generated head paths.

    python -m gray.chain --body-run <runs>/motion-v1 --head-run <runs>/head-v5

On held-out moments whose body is known, three tables:

**1. Sampling.** The body along the recorded head path, for several numbers of sampling steps and
amounts of starting noise, with the past body (warm) and without (cold): pelvis distance to the
recorded one, root yaw, latent error (as gray/motion.py), and how far the head joint of the decoded
body is from the camera. The recorded body gives the reference for the last number.

**2. The chain.** The head path comes from the head-path model (4 steps, half the starting noise, as
the planner is used), the body from the body model along it. There is no recorded body for such a
path, so the test is whether the body belongs to the path: head joint to camera per second, and the
difference between where the body faces and where the camera looks, against the same for recorded moments.

**3. Rolled forward.** Along the recorded head path, the body model generates 5 s, the first second is
kept, now moves on by 1 s, and the model continues from its own output as the past body; five times. Per
second: pelvis distance to the recorded one and head joint to camera, against generating the 5 s at once.
(Rolling the head path forward as well would need the images the camera would have shown along a path
nobody walked; that needs a renderer.)

Decoding follows the STEER repository: MoLingo's decoder gives 272 numbers per 30 fps frame, whose joint
positions are relative to the pelvis in the root's heading; pelvis and heading come from the root path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from gray import frames, head, motion
from gray.joint import to_device
from gray.moments import FUTURE, HISTORY, Moments, collate
from gray.pinhole import STEER

NOW = HISTORY - 1
HEAD_JOINT = 15
SECOND = 15


class Decoder:
    """Pose latents and root path -> joint positions per image frame, in the frame of the root path."""

    def __init__(self, device: str) -> None:
        sys.path.insert(0, str(STEER))
        from data.nymeria import to_actionplan as ap
        from egoprimo import root
        from egoprimo.codec import MotionCodec

        self.codec, self.ap, self.root = MotionCodec().to(device), ap, root

    @torch.no_grad()
    def joints(self, body: torch.Tensor) -> torch.Tensor:
        """body (B, L, 36) -> joints (B, L, 22, 3): x, height above the floor, z."""
        local = self.codec.decode(body[..., :32].float())[..., 8:74]
        local = local.reshape(*local.shape[:-1], 22, 3)
        xz, yaw = self.root.root_path(body[..., 32:].float())
        c, s, zero, one = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
        heading = torch.stack([torch.stack([c, zero, s], -1), torch.stack([zero, one, zero], -1), torch.stack([-s, zero, c], -1)], -2)  # as to_actionplan.rot_yaw
        turned = torch.einsum("...ji,...kj->...ki", heading, local)
        turned[..., 0] += xz[..., None, 0]
        turned[..., 2] += xz[..., None, 1]
        return turned[:, ::2]


TOES = [10, 11]


def feet(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two measures of a body's feet that need no recorded motion, from joints (B, T, 22, 3) at 15 per second, each (B,):

    slide   how fast a toe moves over the ground while it is down: within 5 cm of its lowest height in the clip (cm/s;
            as foot_slide in STEER's egoprimo/evaluate.py)
    hover   how high the lower of the two toes is above the lowest point either reaches in the clip, on average (cm):
            large if the body floats or both feet leave the ground
    """
    toes = joints[:, :, TOES]  # (B, T, 2, 3)
    low = toes[..., 1].amin(dim=1, keepdim=True)
    down = (toes[..., 1] - low < 0.05)[:, 1:]
    speed = (toes[:, 1:, :, [0, 2]] - toes[:, :-1, :, [0, 2]]).norm(dim=-1) * 15 * 100
    slide = (speed * down).sum((1, 2)) / down.sum((1, 2)).clamp_min(1)
    hover = (toes[..., 1].amin(dim=2) - toes[..., 1].amin(dim=(1, 2))[:, None]).mean(1) * 100
    return slide, hover


def to_ground(channels: torch.Tensor, here: torch.Tensor) -> torch.Tensor:
    """A root path (B, T, 4) in the frame anchored at the pose ``here`` (B, 9) -> the track's ground frame (the inverse of frames.anchor_root)."""
    yaw = frames.heading(here[:, 3:])
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    return torch.stack([c * channels[..., 0] - s * channels[..., 1] + here[:, None, 0], s * channels[..., 0] + c * channels[..., 1] + here[:, None, 2],
                        channels[..., 2] * c + channels[..., 3] * s, channels[..., 3] * c - channels[..., 2] * s], dim=-1)


def facing_gap(body: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """Degrees between where the body faces and where the camera looks, (B, T)."""
    turn = torch.atan2(body[..., 34], body[..., 35]) - frames.heading(path[..., 3:])
    return torch.rad2deg(torch.atan2(torch.sin(turn), torch.cos(turn)).abs())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--body-run", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--moments", type=int, default=768)
    parser.add_argument("--rolls", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    model, stats, config = motion.load(args.body_run, device)
    planner, head_mean, head_std, _head_config = head.load(args.head_run, device)
    decoder = Decoder(device)
    data = Moments(Path(config["moments"]), Path(config["tracks"]), Path(config["images"]), Path(config["cache"]), held=True,
                   patches=Path(config["patches"]) if config.get("patches") not in (None, "None") else None, body=Path(config["body"]))
    picked = np.random.default_rng(args.seed).permutation(len(data))[: args.moments].tolist()
    lengths = {str(n): len(np.load(Path(config["tracks"]) / f"{n}.npz")["valid"]) for n in {str(data.table["sequence"][data.rows[i]]) for i in picked}}
    reach = lambda i: int(data.table["now"][data.rows[i]]) + SECOND * (args.rolls - 1) + FUTURE < lengths[str(data.table["sequence"][data.rows[i]])]  # noqa: E731
    steps_of = [[data.item(i, SECOND * k) for k in range(args.rolls)] for i in picked if reach(i)]
    steps_of = [items for items in steps_of if all(bool(item["body_known"]) for item in items)]
    print(f"{len(steps_of)} of {len(picked)} held-out moments have a body for the whole {SECOND * (args.rolls - 1) / 15 + 7:.0f} s", flush=True)
    rolled = [to_device(collate([items[k] for items in steps_of]), device) for k in range(args.rolls)]
    batch = rolled[0]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    path = head.anchored(batch)
    truth = motion.body_anchored(batch)
    past = truth[:, :HISTORY]

    def skeleton(body: torch.Tensor, before: torch.Tensor = past) -> torch.Tensor:
        """The decoded joints (B, 75, 22, 3); the decoder is given the past body for context."""
        return decoder.joints(torch.cat([before, body], dim=1))[:, HISTORY:]

    def to_camera(body: torch.Tensor, along: torch.Tensor, before: torch.Tensor = past) -> torch.Tensor:
        """Distance (B, 75) in cm from the decoded head joint to the camera."""
        return (skeleton(body, before)[:, :, HEAD_JOINT] - along[:, HISTORY:, :3]).norm(dim=-1) * 100

    per_second = lambda values: " ".join(f"{values[:, s : s + SECOND].mean().item():6.1f}" for s in range(0, FUTURE, SECOND))  # noqa: E731
    results = {}
    reference = to_camera(truth[:, HISTORY:], path)
    true_slide, true_hover = feet(skeleton(truth[:, HISTORY:]))
    results["recorded"] = {"head_to_camera_cm": reference.mean().item(), "foot_slide_cm_s": true_slide.mean().item(), "foot_hover_cm": true_hover.mean().item()}
    print(f"\nrecorded body: toes slide {true_slide.mean().item():.1f} cm/s while down, the lower toe is {true_hover.mean().item():.1f} cm above the clip's lowest point on average")
    print(f"recorded body: head joint {reference.mean().item():.1f} cm from the camera; the body faces {facing_gap(truth[:, HISTORY:], path[:, HISTORY:]).mean().item():.1f} degrees away from where the camera looks")

    print("\n1. Sampling, along the recorded head path\n   steps  noise  past body | pelvis cm   yaw deg   latent   head joint to camera cm   toes slide cm/s   lower toe above lowest cm")
    for steps, temperature in ((20, 1.0), (4, 1.0), (4, 0.5), (4, 0.3), (4, 0.0), (1, 0.0)):
        for warm in (True, False):
            drawn = motion.sample(model, batch, stats, steps, generator, amp, warm, temperature)
            turn = torch.atan2(drawn[..., 34], drawn[..., 35]) - torch.atan2(truth[:, HISTORY:, 34], truth[:, HISTORY:, 35])
            row = {"pelvis_cm": (torch.hypot(drawn[..., 32] - truth[:, HISTORY:, 32], drawn[..., 33] - truth[:, HISTORY:, 33]).mean() * 100).item(),
                   "yaw_deg": torch.rad2deg(torch.atan2(torch.sin(turn), torch.cos(turn)).abs()).mean().item(),
                   "latent": (((drawn[..., :32] - truth[:, HISTORY:, :32]) / stats["body_std"][:32]) ** 2).mean().sqrt().item(),
                   "head_to_camera_cm": to_camera(drawn, path).mean().item()}
            slide, hover = feet(skeleton(drawn))
            row |= {"foot_slide_cm_s": slide.mean().item(), "foot_hover_cm": hover.mean().item()}
            results[f"sampling steps {steps} noise {temperature} {'warm' if warm else 'cold'}"] = row
            print(f"   {steps:5d}  {temperature:5.1f}  {'yes' if warm else 'no ':9s} | {row['pelvis_cm']:9.1f} {row['yaw_deg']:9.1f} {row['latent']:8.2f} {row['head_to_camera_cm']:10.1f} {row['foot_slide_cm_s']:20.1f} {row['foot_hover_cm']:18.1f}", flush=True)

    print("\n2. The chain: head path from the head-path model (4 steps, half the noise), body along it (4 steps, noise 0.3)")
    planned = torch.cat([path[:, :HISTORY], head.sample(planner, batch, head_mean, head_std, 4, generator, amp, temperature=0.5, poses=True)], dim=1)
    print("                                   second:      1      2      3      4      5")
    for label, along in (("recorded head path", path), ("generated head path", planned)):
        drawn = motion.sample(model, batch, stats, 4, generator, amp, True, 0.3, head=along)
        gap, facing = to_camera(drawn, along), facing_gap(drawn, along[:, HISTORY:])
        offset = torch.hypot(drawn[..., 32] - along[:, HISTORY:, 0], drawn[..., 33] - along[:, HISTORY:, 2]) * 100
        slide, hover = feet(skeleton(drawn))
        results[f"chain {label}"] = {"head_to_camera_cm": gap.mean().item(), "facing_deg": facing.mean().item(), "pelvis_from_under_camera_cm": offset.mean().item(),
                                     "foot_slide_cm_s": slide.mean().item(), "foot_hover_cm": hover.mean().item()}
        print(f"   {label:20s} toes slide {slide.mean().item():.1f} cm/s while down; lower toe {hover.mean().item():.1f} cm above the lowest point")
        print(f"   {label:20s} head joint to camera cm: {per_second(gap)}\n   {'':20s} body facing against camera, degrees: {per_second(facing)}\n   {'':20s} pelvis from under the camera, cm: {per_second(offset)}", flush=True)
    print(f"   {'recorded body':20s} head joint to camera cm: {per_second(reference)}\n   {'':20s} body facing against camera, degrees: {per_second(facing_gap(truth[:, HISTORY:], path[:, HISTORY:]))}\n"
          f"   {'':20s} pelvis from under the camera, cm: {per_second(torch.hypot(truth[:, HISTORY:, 32] - path[:, HISTORY:, 0], truth[:, HISTORY:, 33] - path[:, HISTORY:, 2]) * 100)}")

    print(f"\n3. Rolled forward along the recorded head path, {args.rolls} times 1 s (4 steps, noise 0.3)")
    # the recorded body of the whole stretch in the ground frame of the track, and what the model generated of it
    recorded = torch.cat([rolled[0]["body"]] + [b["body"][:, -SECOND:] for b in rolled[1:]], dim=1)  # (B, 105 + 15 (rolls - 1), 36)
    made = recorded.clone()
    pelvis, gaps = [], []
    for k, step in enumerate(rolled):
        step = dict(step)
        first = SECOND * k
        step["body"] = made[:, first : first + HISTORY + FUTURE].clone()  # its past holds what was generated in the rolls before
        drawn = motion.sample(model, step, stats, 4, generator, amp, True, 0.3)
        ground = torch.cat([drawn[..., :32], to_ground(drawn[..., 32:], step["pose"][:, NOW])], dim=-1)
        made[:, first + HISTORY : first + HISTORY + SECOND] = ground[:, :SECOND]
        target = recorded[:, first + HISTORY : first + HISTORY + SECOND]
        pelvis.append(torch.hypot(ground[:, :SECOND, 32] - target[..., 32], ground[:, :SECOND, 33] - target[..., 33]) * 100)
        gaps.append(to_camera(drawn, head.anchored(step), motion.body_anchored(step)[:, :HISTORY])[:, :SECOND])
    once = motion.sample(model, batch, stats, 4, generator, amp, True, 0.3)
    once_pelvis = torch.hypot(once[..., 32] - truth[:, HISTORY:, 32], once[..., 33] - truth[:, HISTORY:, 33]) * 100
    print("                                   second:      1      2      3      4      5")
    print(f"   pelvis cm, rolled forward:               {per_second(torch.cat(pelvis, dim=1))}\n   pelvis cm, the 5 s at once:              {per_second(once_pelvis)}")
    print(f"   head joint to camera cm, rolled forward: {per_second(torch.cat(gaps, dim=1))}\n   head joint to camera cm, at once:        {per_second(to_camera(once, path))}", flush=True)
    results["rolled"] = {"pelvis_cm_per_second": [float(p.mean()) for p in pelvis], "head_to_camera_cm_per_second": [float(g.mean()) for g in gaps],
                         "at_once_pelvis_cm_per_second": [float(once_pelvis[:, s : s + SECOND].mean()) for s in range(0, FUTURE, SECOND)]}
    (args.body_run / "chain.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.body_run / 'chain.json'}", flush=True)


if __name__ == "__main__":
    main()
