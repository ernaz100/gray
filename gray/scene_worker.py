#!/usr/bin/env python3
"""Prompt the chain in a recorded home: one JSON request per stdin line, one reply per stdout line.

    python -u -m gray.scene_worker                      # serves requests; gray/scene_server.py talks to it
    python -u -m gray.scene_worker --demo 6 --out <dir> # no server: a few spawns and prompts, replies written as files
    python -u -m gray.scene_worker --data <folder> --steer <folder>   # on a workstation, see "Away from the cluster" below

It speaks the protocol of STEER's egoprimo/scene_worker.py, so the Loc_08 viewer page works with it
unchanged (gray/scene_server.py serves the page and forwards the requests):

Request   {"id": 1, "pos": [x, y, z], "fwd": [x, y, z], "text": "C walks to the fridge.", "seed": 0}
          pos and fwd: head position and view direction in the Y-up world of the scene GLB.
Reply     {"id": 1, "joints_mm": [...], "skin": [...], "frames": 150, "fps": 30, "head_mm": [...], "match": {...}, ...}
          joints_mm is frames x 22 x 3 (world x and z, y above the floor), skin frames x 22 x 12 for the body mesh,
          head_mm the planned camera path, 75 x 3, and head_axes its rotations (75 x 9, camera axes in the columns, in the world);
          image_jpeg the pinhole view the models were given at now, if the recording's video is there.
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

Away from the cluster: ``--data`` is a folder with runs/<head run>, runs/<body run>, index.json (the cache's list of
recordings), and tracks/, body/, pinhole/, grids/ for the recordings of the home (13 GB for Loc_08); with clip/ in it the
text encoder's weights are read from there. ``--steer`` is a copy of the parts of the STEER repository that decode the body
(data/nymeria/to_actionplan.py, egoprimo/root.py and codec.py, models/ActionPlan-Code/src and its SMPL-H model,
models/MoLingo/mogen with the autoencoder's checkpoint). With videos/<id>/preview_rgb.mp4 and lens.json in the data folder
the reply carries the picture the models were shown (gray/seen.py). The models are small; a request takes well under a second on one GPU.
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
from gray.joint import to_device
from gray.moments import FUTURE, HISTORY

NOW = HISTORY - 1
EVERY = 3  # frames between two entries of the table of recorded views
LEVEL_M = 0.6  # a recorded camera this far above or below the spawn is on another storey
TURN_WEIGHT = 1.5  # metres that looking the opposite way costs, halved (as STEER's worker)
HEAD_JOINT = 15
DEMO = ["C walks forward.", "C walks to the kitchen.", "C turns around and walks to the sofa.", "C stands still and looks around.",
        "C walks to the table.", "C walks down the hallway to the bedroom."]


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def reply(payload: dict) -> None:
    sys.stdout.write("@@" + json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def about_up(matrices: np.ndarray, angle: float, x: float, z: float) -> np.ndarray:
    """Turn (..., 3, 4) transforms about the vertical (x' = c x - s z, z' = s x + c z), then shift. As STEER's scene worker."""
    cosine, sine = math.cos(angle), math.sin(angle)
    out = np.array([[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]]) @ matrices
    out[..., 0, 3] += x
    out[..., 2, 3] += z
    return out


class Mesh:
    """The neutral SMPL-H mesh, posed by linear blend skinning with the 22 body joints. As ``Body`` in STEER's scene worker."""

    def __init__(self, model_path: Path, rodrigues) -> None:
        self.rodrigues = rodrigues
        with np.load(model_path, allow_pickle=True) as model:
            self.vertices = np.asarray(model["v_template"], dtype=np.float64)
            self.faces = np.asarray(model["f"], dtype=np.int64)
            weights = np.asarray(model["weights"], dtype=np.float64)
            regressor = model["J_regressor"]
            regressor = regressor.item().toarray() if regressor.shape == () else np.asarray(regressor)
            parents = np.asarray(model["kintree_table"][0], dtype=np.int64)
        self.rest = (regressor @ self.vertices)[:22]
        self.parents = [-1] + [int(p) for p in parents[1:22]]
        folded = weights[:, :22].copy()  # the fingers move with the wrists
        folded[:, 20] += weights[:, 22:37].sum(axis=1)
        folded[:, 21] += weights[:, 37:52].sum(axis=1)
        self.joints4 = np.argsort(-folded, axis=1)[:, :4]
        kept = np.take_along_axis(folded, self.joints4, axis=1)
        self.weights4 = kept / kept.sum(axis=1, keepdims=True)

    def static(self) -> dict:
        return {"vertices_mm": np.round(self.vertices * 1000).astype(int).reshape(-1).tolist(), "faces": self.faces.reshape(-1).tolist(),
                "joints4": self.joints4.reshape(-1).tolist(), "weights4": np.round(self.weights4, 3).reshape(-1).tolist()}

    def skin(self, poses: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
        """Axis-angle poses (T, 66) and pelvis positions (T, 3) -> skinning matrices (T, 22, 3, 4)."""
        count = poses.shape[0]
        local = self.rodrigues(torch.from_numpy(poses).double().reshape(-1, 3)).reshape(count, 22, 3, 3).numpy()
        turn, place = np.zeros((count, 22, 3, 3)), np.zeros((count, 22, 3))
        for joint, parent in enumerate(self.parents):
            if parent < 0:
                turn[:, joint], place[:, joint] = local[:, joint], pelvis
            else:
                turn[:, joint] = turn[:, parent] @ local[:, joint]
                place[:, joint] = place[:, parent] + turn[:, parent] @ (self.rest[joint] - self.rest[parent])
        shift = place - np.einsum("tjab,jb->tja", turn, self.rest)
        return np.concatenate([turn, shift[..., None]], axis=-1)


class Text:
    """CLIP-L token outputs of one sentence, as the cache has them."""

    def __init__(self, device: str, weights: Path | None) -> None:
        import clip

        self.clip, self.device = clip, device
        self.model, _preprocess = clip.load("ViT-L/14", device=device, download_root=str(weights) if weights else None)

    @torch.no_grad()
    def __call__(self, sentence: str) -> torch.Tensor:
        model = self.model
        tokens = self.clip.tokenize([sentence], truncate=True).to(self.device)
        x = model.token_embedding(tokens).type(model.dtype) + model.positional_embedding.type(model.dtype)
        x = model.ln_final(model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)).type(model.dtype)
        return x[0, : int(tokens.argmax(dim=-1)) + 1].float()


class Views:
    """Every third frame of the recordings of one home: camera position and view direction in the Y-up world."""

    def __init__(self, location: str, listing: Path, tracks: Path, images: Path, grids: Path, body: Path) -> None:
        index = json.loads(listing.read_text())["sequences"]
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
    parser.add_argument("--data", type=Path, default=None, help="a folder with runs/, index.json, tracks/, body/, pinhole/, grids/ (and clip/): use it in place of the cluster's paths")
    parser.add_argument("--steer", type=Path, default=Path("/u/enazaren/projects/STEER"), help="the STEER repository, or a copy of the parts that decode the body")
    parser.add_argument("--videos", type=Path, default=None, help="folder of the recordings (for the picture of what the model was shown); default: the cluster's, if there")
    parser.add_argument("--demo", type=int, default=0, help="answer this many made-up requests instead of reading stdin")
    parser.add_argument("--out", type=Path, default=None, help="demo: folder for the replies")
    args = parser.parse_args()

    sys.path.insert(0, str(args.steer))
    from data.nymeria import to_actionplan as ap
    from egoprimo import root
    from egoprimo.codec import MotionCodec

    if args.data is not None:
        args.head_run, args.body_run = args.data / "runs" / args.head_run.name, args.data / "runs" / args.body_run.name
    from gray.seen import Seen

    seen = None
    if args.videos is not None or (args.data is not None and (args.data / "videos").is_dir()):
        seen = Seen(args.videos or args.data / "videos", args.data / "lens.json" if args.data is not None and (args.data / "lens.json").exists() else None)
    elif args.data is None:
        from gray.rectify import SEQUENCES

        seen = Seen(SEQUENCES)
    if getattr(torch.version, "hip", None):
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    planner, head_mean, head_std, _config = head.load(args.head_run, device)
    model, stats, config = motion.load(args.body_run, device)
    weights = args.data / "clip" if args.data is not None and (args.data / "clip").is_dir() else (args.steer.parent.parent / "steer_assets" / "weights" / "clip")
    codec, text, body = MotionCodec().to(device), Text(device, weights if weights.is_dir() else None), Mesh(ap.MODEL, ap.rodrigues)
    if args.data is not None:
        views = Views(args.location, args.data / "index.json", args.data / "tracks", args.data / "pinhole", args.data / "grids", args.data / "body")
    else:
        from gray.features import CACHE

        views = Views(args.location, CACHE / "index.json", Path(config["tracks"]), Path(config["images"]), Path(config["patches"]), Path(config["body"]))
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
        words = text(sentence).cpu() if sentence else torch.zeros(1, 768)
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
        skin = body.skin(smpl["poses"][0].double().numpy(), smpl["joints"][0, :, 0].double().numpy())
        skin = about_up(about_up(skin, float(torch.atan2(start[2], start[3])), *(start[:2] * root.POSITION_SCALE).tolist()), yaw, float(position[0]), float(position[2]))[lead:]
        world, camera = place(joints), place(path[0, :, :3].double().cpu().numpy())
        _xz, psi = root.root_path(channels[0])
        facing = (-math.sin(float(psi[-1])), math.cos(float(psi[-1])))
        shown = None
        if seen is not None:
            try:
                shown = seen.jpeg(match["sequence"], match["feature"])
            except Exception as error:  # noqa: BLE001  the motion is still worth returning
                say(f"the picture of what the model was shown failed: {error!r}")
        axes = frames.matrix(path[0, :, 3:].float()).double().cpu().numpy()  # (75, 3, 3) camera to anchored frame; its rows are those of the pose
        axes = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]) @ axes
        floor = float(match["pos"][1]) - float(batch["pose"][0, NOW, 1])
        pelvis = world[:, 0, [0, 2]]
        gap = np.linalg.norm(world[::2, HEAD_JOINT][: len(camera)] - camera, axis=1)
        return {"id": request.get("id"), "text": sentence, "frames": int(world.shape[0]), "fps": 30, "joints_mm": np.round(world * 1000).astype(int).reshape(-1).tolist(),
                "skin": np.round(skin, 4).reshape(-1).tolist(), "head_mm": np.round(camera * 1000).astype(int).reshape(-1).tolist(), "head_axes": np.round(axes, 4).reshape(-1).tolist(),
                "floor_m": round(floor, 3), "match": match,
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
