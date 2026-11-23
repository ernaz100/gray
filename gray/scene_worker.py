#!/usr/bin/env python3
"""Prompt the chain in a recorded home: one JSON request per stdin line, one reply per stdout line.

    python -u -m gray.scene_worker                      # serves requests; gray/scene_server.py talks to it
    python -u -m gray.scene_worker --demo 6 --out <dir> # no server: a few spawns and prompts, replies written as files

It speaks the protocol of STEER's egoprimo/scene_worker.py, so the Loc_08 viewer page works with it
unchanged (gray/scene_server.py serves the page and forwards the requests):

Request   {"id": 1, "pos": [x, y, z], "fwd": [x, y, z], "text": "C walks to the fridge.", "seed": 0}
          pos and fwd: head position and view direction in the Y-up world of the scene GLB.
Reply     {"id": 1, "joints_mm": [...], "skin": [...], "frames": 150, "fps": 30, "head_mm": [...], "match": {...}, ...}
          joints_mm is frames x 22 x 3 (world x and z, y above the floor), skin frames x 22 x 12 for the body mesh,
          head_mm the planned camera path, 75 x 3.
Request   {"id": 2, "body": true}  ->  the rest-pose body mesh, as in STEER's worker.

What happens per request:

1. The recorded frame of this home whose camera was nearest to the spawn in position and view direction is looked up
   (``match`` says which and how far off). Its image, and the 2 s before it (head poses, images, body), are what the models
   are given: there is no renderer whose images the models can read, so a real view from nearly the same pose stands in.
2. gray/head.py generates the head path of the next 5 s from that and the text (4 steps, half the starting noise).
3. gray/motion.py generates the body along that path (4 steps, noise 0.3), continuing the recorded body of the 2 s before.
4. The body is decoded to joints and joint rotations and placed at the spawn.

So the motion starts from how the recorded person stood or moved at the matched frame, and the head height is theirs.
The comparator (gray/compare.py) is not part of this yet. Loc_08 is a home the models trained on.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gray import frames, head, motion
from gray.features import CACHE
from gray.joint import to_device
from gray.moments import FUTURE, HISTORY
from gray.pinhole import STEER

NOW = HISTORY - 1
EVERY = 3  # frames between two entries of the table of recorded views
LEVEL_M = 0.6  # a recorded camera this far above or below the spawn is on another storey
TURN_WEIGHT = 1.5  # metres that looking the opposite way costs, halved (as STEER's worker)
HEAD_JOINT = 15
DEMO = ["C walks forward.", "C walks to the kitchen.", "C turns around and walks to the sofa.", "C stands still and looks around.",
        "C walks to the table.", "C walks down the hallway to the bedroom."]


class Views:
    """Every third frame of the recordings of one home: camera position and view direction in the Y-up world."""

    def __init__(self, location: str, cache: Path, tracks: Path, images: Path, grids: Path, body: Path) -> None:
        index = json.loads((cache / "index.json").read_text())["sequences"]
        self.names = [m["sequence_id"] for m in index if m["location"] == location and (tracks / f"{m['sequence_id']}.npz").exists()
                      and (grids / m["sequence_id"] / "rows.npy").exists() and (body / f"{m['sequence_id']}.npz").exists()]
        self.data, sequence, frame, position, forward = {}, [], [], [], []
        for i, name in enumerate(self.names):
            track, own = np.load(tracks / f"{name}.npz"), np.load(body / f"{name}.npz")
            rows = np.load(grids / name / "rows.npy")
            self.data[name] = {"pose": track["pose"], "floor": track["floor"], "image": np.load(images / name / "dino.npy", mmap_mode="r"), "rows": rows,
                               "grid": np.load(grids / name / "patches.npy", mmap_mode="r"), "latent": own["latent"], "pelvis": own["pelvis"],
                               "yaw": own["yaw"], "body_valid": own["valid"]}
            valid = track["valid"]
            at = np.arange(HISTORY, len(valid) - 1, EVERY)
            at = at[valid[at] & valid[at - HISTORY + 1] & (rows[at] >= 0)]
            pose = track["pose"][at]
            yaw = frames.heading(torch.from_numpy(pose[:, 3:])).numpy()
            sequence.append(np.full(len(at), i)); frame.append(at)
            position.append(np.stack([pose[:, 0], pose[:, 1] + track["floor"][at], pose[:, 2]], axis=1))
            forward.append(np.stack([-np.sin(yaw), np.zeros_like(yaw), np.cos(yaw)], axis=1))
        self.sequence, self.frame = np.concatenate(sequence), np.concatenate(frame)
        self.position, self.forward = np.concatenate(position).astype(np.float32), np.concatenate(forward).astype(np.float32)

    def nearest(self, position: np.ndarray, forward: np.ndarray) -> tuple[int, dict]:
        offset = self.position - position
        distance = np.hypot(offset[:, 0], offset[:, 2])
        cosine = self.forward @ forward
        best = int((distance + TURN_WEIGHT * (1 - cosine) + np.where(np.abs(offset[:, 1]) > LEVEL_M, 1e3, 0.0)).argmin())
        return best, {"sequence": self.names[int(self.sequence[best])], "feature": int(self.frame[best]), "video_s": round(int(self.frame[best]) / 15, 2),
                      "distance_m": round(float(distance[best]), 2), "angle_deg": round(math.degrees(math.acos(float(np.clip(cosine[best], -1, 1)))), 1),
                      "other_storey": bool(abs(float(offset[best, 1])) > LEVEL_M), "pos": [round(float(v), 3) for v in self.position[best]],
                      "fwd": [round(float(v), 3) for v in self.forward[best]]}

    def batch(self, best: int, words: torch.Tensor) -> dict:
        """What the models are given for the matched frame: its past, with the future filled by the pose at now."""
        a, now = self.data[self.names[int(self.sequence[best])]], int(self.frame[best])
        span = slice(now - HISTORY + 1, now + 1)
        tensor = lambda values: torch.from_numpy(np.array(values, dtype=np.float32))  # noqa: E731
        pose = tensor(a["pose"][span])
        own = tensor(np.concatenate([a["latent"][span], a["pelvis"][span][:, [0, 2]], np.sin(a["yaw"][span])[:, None], np.cos(a["yaw"][span])[:, None]], axis=1))
        return {"pose": torch.cat([pose, pose[-1:].expand(FUTURE, -1)])[None], "image": tensor(a["image"][span])[None], "patches": tensor(a["grid"][int(a["rows"][now])])[None],
                "body": torch.cat([own, own[-1:].expand(FUTURE, -1)])[None], "body_known": torch.tensor([bool(a["body_valid"][span].all())]),
                "text": words[None], "text_valid": torch.ones(1, words.shape[0], dtype=torch.bool)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--location", default="Loc_08")
    parser.add_argument("--head-run", type=Path, default=Path("/ptmp/enazaren/gray/runs/head-v5"))
    parser.add_argument("--body-run", type=Path, default=Path("/ptmp/enazaren/gray/runs/motion-v1"))
    parser.add_argument("--demo", type=int, default=0, help="answer this many made-up requests instead of reading stdin")
    parser.add_argument("--out", type=Path, default=None, help="demo: folder for the replies")
    args = parser.parse_args()

    sys.path.insert(0, str(STEER))
    from data.nymeria import to_actionplan as ap
    from egoprimo import root
    from egoprimo.codec import MotionCodec
    from egoprimo.scene_worker import Body, Text, about_up, first_image, reply, say
    from gray.rectify import SEQUENCES

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    planner, head_mean, head_std, _config = head.load(args.head_run, device)
    model, stats, config = motion.load(args.body_run, device)
    codec, text, body = MotionCodec().to(device), Text(device), Body()
    views = Views(args.location, CACHE, Path(config["tracks"]), Path(config["images"]), Path(config["patches"]), Path(config["body"]))
    say(f"ready: {args.head_run.name} and {args.body_run.name}, {len(views.frame):,d} recorded views of {len(views.names)} recordings in {args.location}")
    reply({"ready": True, "run": f"{args.head_run.name} + {args.body_run.name}", "checkpoint": "", "step": None, "views": int(len(views.frame)), "sequences": len(views.names)})

    def answer(request: dict) -> dict:
        began = time.time()
        position = np.array(request["pos"], dtype=np.float32)
        forward = np.array(request["fwd"], dtype=np.float32)
        forward[1] = 0.0
        forward /= max(float(np.linalg.norm(forward)), 1e-6)
        best, match = views.nearest(position, forward)
        sentence = " ".join(str(request.get("text", "")).split())
        words = text(sentence)[0].cpu() if sentence else torch.zeros(1, 768)
        batch = to_device(views.batch(best, words), device)
        if not sentence:
            batch["text_valid"][:] = False
        generator = torch.Generator(device=device).manual_seed(int(request.get("seed", 0)))
        past = head.anchored(batch)[:, :HISTORY]
        path = head.sample(planner, batch, head_mean, head_std, 4, generator, amp, temperature=0.5, poses=True)  # (1, 75, 9), anchored at now
        warm = bool(batch["body_known"][0])
        drawn = motion.sample(model, batch, stats, 4, generator, amp, warm, 0.3, head=torch.cat([past, path], dim=1))
        # decode with the recorded 2 s in front, which the decoder needs as context, and cut them off again
        whole = torch.cat([motion.body_anchored(batch)[:, :HISTORY], drawn], dim=1) if warm else drawn
        lead = 2 * HISTORY if warm else 0
        with torch.no_grad():
            smpl = ap.streamer272_to_smpl(codec.decode_with_root(whole[..., :32].float(), whole[..., 32:].float()).cpu())
        channels = whole[..., 32:].float().cpu()
        joints = root.place(smpl["joints"], channels)[0].double().numpy()[lead:]  # (150, 22, 3) in the frame anchored at now
        yaw = float(root.yaw_of(torch.from_numpy(forward)))
        c, s = math.cos(yaw), math.sin(yaw)
        place = lambda p: np.stack([c * p[..., 0] - s * p[..., 2] + float(position[0]), p[..., 1], s * p[..., 0] + c * p[..., 2] + float(position[2])], axis=-1)  # noqa: E731
        start = channels[0, 0]
        skin, _posed = body.skin(smpl["poses"][0].double().numpy(), smpl["joints"][0, :, 0].double().numpy())
        skin = about_up(about_up(skin, float(torch.atan2(start[2], start[3])), *(start[:2] * root.POSITION_SCALE).tolist()), yaw, float(position[0]), float(position[2]))[lead:]
        world, camera = place(joints), place(path[0, :, :3].double().cpu().numpy())
        _xz, psi = root.root_path(channels[0])
        facing = (-math.sin(float(psi[-1])), math.cos(float(psi[-1])))
        try:
            shown = first_image(SEQUENCES / match["sequence"] / "preview_rgb.mp4", match["feature"], 15.0)
        except Exception as error:  # noqa: BLE001  the motion is still worth returning
            say(f"first image failed: {error!r}")
            shown = None
        pelvis = world[:, 0, [0, 2]]
        gap = np.linalg.norm(world[::2, HEAD_JOINT][: len(camera)] - camera, axis=1)
        return {"id": request.get("id"), "text": sentence, "frames": int(world.shape[0]), "fps": 30, "joints_mm": np.round(world * 1000).astype(int).reshape(-1).tolist(),
                "skin": np.round(skin, 4).reshape(-1).tolist(), "head_mm": np.round(camera * 1000).astype(int).reshape(-1).tolist(), "match": match,
                "image_jpeg": shown, "image_source": "recording", "recorded_cosine": None, "recorded_jpeg": shown, "past_body": warm,
                "head_to_camera_cm": round(float(gap.mean() * 100), 1),
                "end": {"head": [round(float(v), 3) for v in world[-1, HEAD_JOINT]], "fwd": [round(c * facing[0] - s * facing[1], 4), 0.0, round(s * facing[0] + c * facing[1], 4)]},
                "travel_m": round(float(np.linalg.norm(np.diff(pelvis, axis=0), axis=1).sum()), 2), "seconds": round(time.time() - began, 2)}

    if args.demo:
        args.out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for i, at in enumerate(rng.choice(len(views.frame), args.demo, replace=False)):
            request = {"id": i, "pos": views.position[at].tolist(), "fwd": views.forward[at].tolist(), "text": DEMO[i % len(DEMO)], "seed": i}
            result = answer(request)
            (args.out / f"reply-{i}.json").write_text(json.dumps({"request": request, **result}))
            say(f"{i}: \"{request['text']}\" from {result['match']['sequence'][:32]} at {result['match']['video_s']} s: pelvis travels {result['travel_m']} m, "
                f"head joint {result['head_to_camera_cm']} cm from the planned camera, past body {result['past_body']}, {result['seconds']} s")
        return
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = {}
        try:
            request = json.loads(line)
            reply({"id": request.get("id"), **body.static()} if request.get("body") else answer(request))
        except Exception as error:  # noqa: BLE001  keep serving after a bad request
            say(f"request failed: {error!r}")
            reply({"id": request.get("id") if isinstance(request, dict) else None, "error": repr(error)})


if __name__ == "__main__":
    main()
