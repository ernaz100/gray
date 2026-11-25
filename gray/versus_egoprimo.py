#!/usr/bin/env python3
"""Our chain against the EgoPriMo reimplementation on held-out Nymeria moments: generation from the first frame and the sentence.

    python -m gray.versus_egoprimo --ego-run <STEER run folder> --head-run <run> --body-run <run>

The EgoPriMo reimplementation (STEER repository, egoprimo/; trained on all of Nymeria with the same held-out homes) generates
8 s of body from the first image and text in one model. Ours plans a head path and then generates the body along it. Both
are run on the same held-out moments of the navigation subset that have a body and a map: "now" is where the sentence
begins, each model is given the frame at now and that one sentence, and the first 5 s are scored.

What each side is given differs where the models differ:
    EgoPriMo   the image feature of the frame at now as its cache has it (from the fisheye frame), the sentence. It was
               trained on 8 s windows cut anywhere, with all sentences that overlap them.
    ours       the pinhole image feature and image grid of the frame at now, the camera pose at now, the sentence; no earlier
               frames and no body ("cold"). A second row gives it the 2 s before now and its own body in them, as it is used.

Bodies are decoded with the same decoder to 22 joints in the frame anchored at now. The reference is the recorded body
through the same encoder and decoder. Per row, over the first 5 s:

    joints, aligned, head, pelvis   distance to the recorded body (m); aligned: after the best rigid fit per frame
    travels, 1 m or more            how far the pelvis goes, and in what share of the moments it goes 1 m or further
    through something               share of bodies whose head passes through a blocked cell of the home's map while moving
    slide                           toes sliding while down (gray/chain.py)

A row for standing still (the body at now, held) is there because distance to the recorded body rewards not moving.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from gray import frames, head, motion
from gray.chain import Decoder, feet
from gray.ground import Ground
from gray.joint import to_device
from gray.moments import HISTORY, Moments, collate
from gray.pinhole import STEER
from gray.tracks import to_world
from gray.versus_uem import rigid

NOW = HISTORY - 1
MOVING = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ego-run", type=Path, default=Path("/ptmp/enazaren/nymeria/ego_runs/egoprimo/all-medium-v1"))
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--body-run", type=Path, required=True)
    parser.add_argument("--ground", type=Path, default=Path("/ptmp/enazaren/gray/ground"))
    parser.add_argument("--moments", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ego-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if getattr(torch.version, "hip", None):
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    sys.path.insert(0, str(STEER))
    from egoprimo import train as ego
    from egoprimo.model import EgoMotionDiT, ModelConfig

    ego_config = json.loads((args.ego_run / "config.json").read_text())
    ego_model = EgoMotionDiT(ModelConfig(**ego_config["model"])).to(device)
    state = torch.load(args.ego_run / ("best.pt" if (args.ego_run / "best.pt").exists() else "last.pt"), map_location=device, weights_only=True)
    ego_model.load_state_dict(state["model"])
    ego_model.eval()
    print(f"EgoPriMo reimplementation: {args.ego_run.name}, step {state.get('step')}", flush=True)

    planner, head_mean, head_std, config = head.load(args.head_run, device)
    model, stats, body_config = motion.load(args.body_run, device)
    decoder = Decoder(device)
    cache, tracks = Path(config["cache"]), Path(config["tracks"])
    listing = dict(np.load(config["moments"]))
    data = Moments(Path(config["moments"]), tracks, Path(config["images"]), cache, held=True, patches=Path(config["patches"]), body=Path(body_config["body"]),
                   only=listing["has_map"])
    order = np.random.default_rng(args.seed).permutation(len(data))
    location = {m["sequence_id"]: m["location"] for m in json.loads((cache / "index.json").read_text())["sequences"]}
    files: dict[str, dict] = {}
    grounds: dict[str, Ground] = {}

    names = ("recorded", "standing still", "EgoPriMo", "ours, cold, usual sampling", "ours, cold, full sample", "ours, with the 2 s before, usual sampling")
    scores = {name: [] for name in names[1:]}
    moves = {name: [] for name in names}
    blocked = {name: [] for name in names}
    slides = {name: [] for name in names}
    generator = torch.Generator(device=device)
    items, done = [], 0
    for position, i in enumerate(order):
        item = data[int(i)]
        if bool(item["body_known"]):
            items.append(item)
        if len(items) < args.batch and done + len(items) < args.moments and position < len(order) - 1:
            continue
        if not items:
            break
        batch = to_device(collate(items), device)
        size = batch["pose"].shape[0]
        rows_ = batch["row"].tolist()
        # what EgoPriMo is given: its own image feature of the frame at now, the sentence
        image, sentence = torch.zeros(size, ego.TOKENS, 384), torch.zeros(size, 768)
        for b, row in enumerate(rows_):
            name = str(listing["sequence"][row])
            if name not in files:
                files[name] = {"dino": np.load(cache / name / "dino.npy", mmap_mode="r"), "said": np.load(cache / name / "text_sentence.npy", mmap_mode="r"),
                               "floor": np.load(tracks / f"{name}.npz")["floor"]}
            image[b, 0] = torch.from_numpy(np.array(files[name]["dino"][int(batch["now"][b])], dtype=np.float32))
            sentence[b] = torch.from_numpy(np.array(files[name]["said"][int(listing["sentence"][row])], dtype=np.float32))
        theirs = {"motion": torch.zeros(size, ego.TOKENS, 36), "image": image, "head": torch.zeros(size, ego.TOKENS, 9), "has_text": torch.ones(size, dtype=torch.bool),
                  "text": batch["text"][:, : ego.MAX_TEXT].cpu(), "sentence": sentence, "text_valid": batch["text_valid"][:, : ego.MAX_TEXT].cpu()}
        theirs = ego.to_device(theirs, device)
        with torch.no_grad():
            out = ego.sample(ego_model, theirs, torch.ones(size, dtype=torch.long, device=device), args.ego_steps, seed=args.seed + done, amp=amp)
        ego_body = torch.cat([out[..., :32] * ego_config["latent_sigma"] + ego_config["latent_mean"], out[..., 32:]], dim=-1)[:, 1:76]  # token 0 is the frame at now

        past = head.anchored(batch)
        truth = motion.body_anchored(batch)
        made = {"recorded": truth[:, HISTORY:], "standing still": truth[:, NOW : NOW + 1].expand(-1, 75, -1), "EgoPriMo": ego_body}
        for label, steps, head_noise, body_noise, history, warm in (("ours, cold, usual sampling", 4, 0.5, 0.3, 1, False), ("ours, cold, full sample", 20, 1.0, 1.0, 1, False),
                                                                     ("ours, with the 2 s before, usual sampling", 4, 0.5, 0.3, HISTORY, True)):
            generator.manual_seed(args.seed + done)
            path = head.sample(planner, batch, head_mean, head_std, steps, generator, amp, history=history, temperature=head_noise, poses=True)
            made[label] = motion.sample(model, batch, stats, steps, generator, amp, warm, body_noise, head=torch.cat([past[:, :HISTORY], path], dim=1), history=history)
        joints = {name: decoder.joints(body.float()) for name, body in made.items()}  # (B, 75, 22, 3), anchored at now
        real = joints["recorded"].double()
        here = batch["pose"][:, NOW].double().cpu().numpy()
        yaw = frames.heading(batch["pose"][:, NOW, 3:]).double().cpu().numpy()
        for name in names:
            own = joints[name].double()
            if name != "recorded":
                gap = (own - real).norm(dim=-1)
                fitted = (rigid(own, real) - real).norm(dim=-1)
                scores[name] += torch.stack([gap.mean((1, 2)), fitted.mean((1, 2)), gap[:, :, 15].mean(1), gap[:, :, 0].mean(1)], dim=1).cpu().tolist()
            pelvis = own[:, :, 0, [0, 2]]
            moves[name] += (pelvis[:, 1:] - pelvis[:, :-1]).norm(dim=-1).sum(1).cpu().tolist()
            slides[name] += feet(joints[name])[0].cpu().tolist()
            head_joint = own[:, :, 15].cpu().numpy()
            for b, row in enumerate(rows_):
                sequence = str(listing["sequence"][row])
                where = location[sequence]
                if where not in grounds:
                    grounds[where] = Ground(args.ground / f"{where}.npz")
                c, s = np.cos(yaw[b]), np.sin(yaw[b])
                p = head_joint[b, ::3]
                world = to_world(np.stack([c * p[:, 0] - s * p[:, 2] + here[b, 0], p[:, 1], s * p[:, 0] + c * p[:, 2] + here[b, 2]], axis=1), files[sequence]["floor"][int(batch["now"][b])])
                moving = np.linalg.norm(np.diff(world[:, :2], axis=0, prepend=world[:1, :2]), axis=1) / 0.2 > MOVING
                blocked[name].append(bool((grounds[where].flags(world)["blocked"] & moving).any()))
        done += size
        items = []
        print(f"  {done:,d} moments", flush=True)
        if done >= args.moments:
            break

    result = {}
    print(f"\n{done:,d} held-out moments, the first 5 s after now\n  {'':44s} joints  aligned    head  pelvis | travels  1 m or more  through something  slide cm/s")
    for name in names:
        r = {"travel_m": float(np.mean(moves[name])), "over_1m": float(np.mean(np.array(moves[name]) >= 1.0)), "through": float(np.mean(blocked[name])), "slide_cm_s": float(np.mean(slides[name]))}
        if name != "recorded":
            r |= dict(zip(("joints_m", "aligned_m", "head_m", "pelvis_m"), np.mean(scores[name], axis=0).tolist()))
        result[name] = r
        left = f"{r['joints_m']:6.3f} {r['aligned_m']:8.3f} {r['head_m']:7.3f} {r['pelvis_m']:7.3f}" if name != "recorded" else " " * 31
        print(f"  {name:44s} {left} | {r['travel_m']:5.2f} m {r['over_1m']:11.0%} {r['through']:17.1%} {r['slide_cm_s']:11.1f}")
    (args.body_run / "versus_egoprimo.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.body_run / 'versus_egoprimo.json'}", flush=True)


if __name__ == "__main__":
    main()
