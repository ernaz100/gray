#!/usr/bin/env python3
"""HumanML3D in the form of our samples: body latents, a root path, and a head path made from the head joint.

    python -m gray.hml offset  --data <folder> --steer <folder>     # how the camera sits on the head joint, from Nymeria
    python -m gray.hml encode  --data <folder> --steer <folder>     # all motions
    python -m gray.hml census  --data <folder>                      # how many motions candidate subsets keep
    python -m gray.hml text    --data <folder>                      # text features of the captions (text_tokens.npy, text_index.json)

HumanML3D (the 272-number representation at 30 frames per second, Hugging Face royrhea/272-dim-HumanML3D:
<data>/hml3d/motion_data.zip, texts.zip, split/) has bodies and captions but no camera. To train on it next to Nymeria
(gray/head.py, gray/motion.py) every motion needs what a moment has:

    latent   the pose latent per image frame (15 per second), from the same motion autoencoder as gray/body.py. The
             autoencoder was trained on this data, so nothing has to be adapted.
    pelvis, yaw   the root path, as gray/body.py
    pose     a camera pose per frame, (x, height above the floor, z) and the first two rows of its rotation, as gray/tracks.py.
             It is the head joint moved by a fixed transform: the average of how Nymeria's camera sits on the head joint
             (**offset**; the decoded Nymeria bodies of the recordings in <data> against their camera tracks).

There are no images: models are given a "no image" token for these samples.

**encode** writes <data>/hml3d/hml.npz (all motions end to end, with ``start`` saying where each begins) and texts.json
(per motion its captions that describe the whole clip). **census** prints, for candidate rules, how many motions and
how many 7 s samples remain.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from gray import frames
from gray.body import channels

HEAD_JOINT = 15
CHAIN = [0, 3, 6, 9, 12, 15]  # pelvis to head in the SMPL tree
TOKENS = 105  # a moment: 30 frames given, 75 generated


class HmlMoments(torch.utils.data.Dataset):
    """HumanML3D motions as samples shaped like the moments of gray/moments.py with a body, for training next to them.

    A sample is a stretch of one motion: up to 30 frames before "now" (fewer if the motion begins later; "history_max" says
    how many are real) and the 75 after it. "image" and "patches" are zeros and "foreign" is True: the models put a learned
    token in place of the image, and may do the same for the head path, which here is made from the head joint and is not
    a real camera's. The text is one of the motion's captions, chosen at random. Only motions of the training split that
    are 5.1 s or longer and whose pelvis gets ``reach`` metres or further from its start are used.
    """

    def __init__(self, folder: Path, reach: float = 1.0) -> None:
        d = np.load(folder / "hml.npz")
        self.latent, self.pose, self.pelvis, self.yaw, start = d["latent"], d["pose"], d["pelvis"], d["yaw"], d["start"]
        self.index = json.loads((folder / "text_index.json").read_text())
        self.tokens = np.load(folder / "text_tokens.npy", mmap_mode="r")
        train = set((folder / "split" / "train.txt").read_text().split())
        self.samples, self.motions = [], 0
        for name, a, b in zip(d["names"], start[:-1], start[1:]):
            name = str(name)
            far = np.hypot(*(self.pelvis[a:b, [0, 2]] - self.pelvis[a, [0, 2]]).T).max()
            if b - a < 76 or far < reach or not self.index.get(name) or name.lstrip("M") not in train:
                continue
            self.motions += 1
            self.samples += [(name, int(a), int(b), int(s)) for s in range(0, b - a - 76 + 1, 15)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        name, a, b, s = self.samples[index]
        now = a + min(s + int(torch.randint(0, 15, (1,))), b - a - 76)  # the frame at now; 75 frames follow it
        rows = np.clip(np.arange(now - 29, now + 76), a, b - 1)  # frames before the motion begins repeat its first one; they are not shown
        own = np.concatenate([self.latent[rows], self.pelvis[rows][:, [0, 2]], np.sin(self.yaw[rows])[:, None], np.cos(self.yaw[rows])[:, None]], axis=1)
        first, last = self.index[name][int(torch.randint(0, len(self.index[name]), (1,)))]
        return {"pose": torch.from_numpy(self.pose[rows].copy()), "image": torch.zeros(30, 384), "patches": torch.zeros(49, 384),
                "text": torch.from_numpy(np.array(self.tokens[first:last], dtype=np.float32)), "row": torch.tensor(-1), "now": torch.tensor(now - a),
                "body": torch.from_numpy(own.astype(np.float32)), "body_known": torch.tensor(True), "foreign": torch.tensor(True),
                "history_max": torch.tensor(min(now - a + 1, 30))}


def tools(steer: Path, device: str):
    sys.path.insert(0, str(steer))
    from data.nymeria import to_actionplan as ap
    from egoprimo import root
    from egoprimo.codec import MotionCodec

    return ap, root, MotionCodec().to(device)


def head_of(ap, smpl: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Position (T, 3) and rotation (T, 3, 3) of the head joint from one decoded motion, in the frame the decoder starts in."""
    poses = smpl["poses"][0].double()
    local = ap.rodrigues(poses.reshape(-1, 3)).reshape(len(poses), 22, 3, 3)
    turn = local[:, 0]
    for joint in CHAIN[1:]:
        turn = turn @ local[:, joint]
    return smpl["joints"][0, :, HEAD_JOINT].double(), turn


def about_up(angle: torch.Tensor) -> torch.Tensor:
    """Rotation about the vertical by ``angle`` (...,): x' = c x - s z, z' = s x + c z, as egoprimo/root.place."""
    c, s, zero, one = torch.cos(angle), torch.sin(angle), torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack([torch.stack([c, zero, -s], -1), torch.stack([zero, one, zero], -1), torch.stack([s, zero, c], -1)], -2)


def offset(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ap, root, codec = tools(args.steer, device)
    rng = np.random.default_rng(0)
    turns, shifts = [], []
    names = sorted(p.stem for p in (args.data / "body").glob("*.npz") if (args.data / "tracks" / f"{p.stem}.npz").exists())
    for name in names[: args.recordings]:
        own, track = np.load(args.data / "body" / f"{name}.npz"), np.load(args.data / "tracks" / f"{name}.npz")
        ok = own["valid"] & track["valid"]
        starts = [int(s) for s in rng.permutation(len(ok) - 60)[:400] if ok[s : s + 60].all()][:12]
        for s in starts:
            span = slice(s, s + 60)
            pose = torch.from_numpy(track["pose"][span]).float()[None]
            anchored = frames.anchor(pose, torch.zeros(1, dtype=torch.long))[0].double()  # the camera, in the frame of its first pose
            heading = frames.heading(pose[:, 0, 3:])
            path = channels(torch.from_numpy(own["pelvis"][span]).float()[None], torch.from_numpy(own["yaw"][span]).float()[None], pose[:, 0, [0, 2]], heading)
            with torch.no_grad():
                smpl = ap.streamer272_to_smpl(codec.decode_with_root(torch.from_numpy(own["latent"][span]).float()[None].to(device), path.to(device)).cpu())
            at, turn = head_of(ap, smpl)
            start = path[0, 0].double()
            first = about_up(torch.atan2(start[2], start[3]))
            at = (root.place(smpl["joints"], path)[0, :, HEAD_JOINT].double())[::2]
            turn = (first @ turn)[::2]
            camera = frames.matrix(anchored[:, 3:])  # camera axes in the columns
            turns.append(turn.transpose(1, 2) @ camera)
            shifts.append(torch.einsum("tji,tj->ti", turn, anchored[:, :3] - at))
    turns, shifts = torch.cat(turns), torch.cat(shifts)
    u, _s, vt = torch.linalg.svd(turns.mean(0))
    mean_turn = u @ vt
    apart = torch.rad2deg(torch.acos((((mean_turn.T @ turns).diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1)))
    mean_shift = shifts.mean(0)
    np.savez(args.data / "hml3d" / "offset.npz", turn=mean_turn.numpy(), shift=mean_shift.numpy())
    print(f"{len(turns):,d} frames of {min(len(names), args.recordings)} recordings")
    print(f"the camera sits {mean_shift.norm() * 100:.1f} cm from the head joint ({(mean_shift * 100).numpy().round(1)} cm in the joint's frame); single frames differ from that by "
          f"{(shifts - mean_shift).norm(dim=1).median() * 100:.1f} cm (median)")
    print(f"its rotation against the head joint differs from the mean by {apart.median():.1f} degrees (median), {apart.quantile(0.9):.1f} (90 %)")
    print("the camera's viewing direction in the head joint's frame:", mean_turn[:, 2].numpy().round(3), "(x left, y up, z ahead of the body)")


def encode(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ap, _root, codec = tools(args.steer, device)
    fixed = np.load(args.data / "hml3d" / "offset.npz")
    turn_off, shift_off = torch.from_numpy(fixed["turn"]), torch.from_numpy(fixed["shift"])
    motions = zipfile.ZipFile(args.data / "hml3d" / "motion_data.zip")
    members = sorted(m for m in motions.namelist() if m.endswith(".npy"))[: args.limit]
    out = {key: [] for key in ("latent", "pose", "pelvis", "yaw")}
    names, start, total = [], [0], 0
    for i, member in enumerate(members):
        feat = torch.from_numpy(np.load(io.BytesIO(motions.read(member)))).float()
        feat = feat[: len(feat) // 2 * 2]
        if len(feat) < 40:
            continue
        with torch.no_grad():
            latent = codec.encode(feat[None].to(device))[0].cpu()
        smpl = ap.streamer272_to_smpl(feat[None])
        at, turn = head_of(ap, smpl)
        camera_turn = turn @ turn_off
        camera_at = at + torch.einsum("tij,j->ti", turn, shift_off)
        pelvis = smpl["joints"][0, :, 0]
        local = ap.rodrigues(smpl["poses"][0, :, :3].double())
        out["latent"].append(latent.numpy().astype(np.float32))
        out["pose"].append(torch.cat([camera_at, camera_turn[:, 0], camera_turn[:, 1]], dim=1)[::2].numpy().astype(np.float32))
        out["pelvis"].append(pelvis[::2].numpy().astype(np.float32))
        out["yaw"].append((-torch.atan2(local[:, 0, 2], local[:, 2, 2]))[::2].numpy().astype(np.float32))
        names.append(Path(member).stem)
        total += len(latent)
        start.append(total)
        if i % 2000 == 0:
            print(f"{i:,d}/{len(members):,d}", flush=True)
    np.savez(args.data / "hml3d" / "hml.npz", **{key: np.concatenate(value) for key, value in out.items()}, start=np.array(start), names=np.array(names))
    texts = zipfile.ZipFile(args.data / "hml3d" / "texts.zip")
    captions = {}
    for name in names:
        try:
            lines = texts.read(f"texts/{name}.txt").decode(errors="replace").splitlines()
        except KeyError:
            continue
        whole = [line.split("#")[0].strip() for line in lines if line.count("#") >= 3 and line.split("#")[2] in ("0.0", "nan") and line.split("#")[3] in ("0.0", "nan")]
        captions[name] = whole
    (args.data / "hml3d" / "texts.json").write_text(json.dumps(captions))
    print(f"{len(names):,d} motions, {total:,d} frames at 15 per second ({total / 15 / 3600:.1f} h); {sum(bool(v) for v in captions.values()):,d} have a caption for the whole clip")


def census(args) -> None:
    from gray.census import WALK

    d = np.load(args.data / "hml3d" / "hml.npz")
    captions = json.loads((args.data / "hml3d" / "texts.json").read_text())
    start, pelvis, names = d["start"], d["pelvis"], d["names"]
    length = np.diff(start)
    reach = np.array([np.hypot(*(pelvis[a:b, [0, 2]] - pelvis[a, [0, 2]]).T).max() for a, b in zip(start[:-1], start[1:])])
    travel = np.array([np.linalg.norm(np.diff(pelvis[a:b, [0, 2]], axis=0), axis=1).sum() for a, b in zip(start[:-1], start[1:])])
    text = [" ".join(captions.get(str(n), [])).lower() for n in names]
    said = np.array([bool(WALK.search(t)) for t in text])
    some = np.array([bool(t) for t in text])
    mirrored = np.array([str(n).startswith("M") for n in names])
    split = {name: key for key in ("train", "val", "test") for name in (args.data / "hml3d" / "split" / f"{key}.txt").read_text().split()}
    part = np.array([split.get(str(n).lstrip("M"), "none") for n in names])
    words = {"dance": r"\bdanc", "jump or hop": r"\b(jump|hop|leap)", "sit": r"\b(sit|sits|sitting|seated)\b", "kick or punch or fight": r"\b(kick|punch|box|fight)", "run or jog": r"\b(run|runs|running|jog)",
             "stairs or climb": r"\b(stair|climb)", "crawl or lie": r"\b(crawl|lie|lies|lying|lay)", "throw or catch": r"\b(throw|catch)", "turn": r"\bturn"}
    print(f"{len(names):,d} motions ({mirrored.sum():,d} of them mirrored copies), {length.sum() / 15 / 3600:.1f} h; split of the originals: " + ", ".join(f"{k} {int(((part == k) & ~mirrored).sum()):,d}" for k in ("train", "val", "test")))
    print(f"length: median {np.median(length) / 15:.1f} s; 5.1 s or longer (one frame given, 5 s generated) {np.mean(length >= 76):.0%}; 7 s or longer (a full moment) {np.mean(length >= TOKENS):.0%}")
    print(f"the pelvis gets 1 m or further from its start in {np.mean(reach >= 1):.0%}; median path length {np.median(travel):.1f} m; a caption mentions walking or going in {np.mean(said):.0%}")
    rules = {"all with a caption": some, "5.1 s or longer": some & (length >= 76), "... and gets 1 m or further": some & (length >= 76) & (reach >= 1.0),
             "... and the caption mentions walking": some & (length >= 76) & (reach >= 1.0) & said}
    never = np.zeros(len(names), dtype=bool)
    for label in ("dance", "jump or hop", "kick or punch or fight", "crawl or lie", "throw or catch"):
        never |= np.array([bool(re.search(words[label], t)) for t in text])
    rules["... and no dancing, jumping, fighting, crawling, throwing"] = rules["... and gets 1 m or further"] & ~never
    print(f"\n{'rule':58s} motions   hours   samples of 7 s (a new one every 1 s)")
    for label, keep in rules.items():
        samples = np.maximum(length[keep] - 76, 0) // 15 + 1
        print(f"{label:58s} {int(keep.sum()):7,d} {length[keep].sum() / 15 / 3600:7.1f} {int(samples.sum()):9,d}")
    print("\nwhat the captions of the motions that get 1 m or further mention:")
    going = some & (reach >= 1.0)
    for label, pattern in words.items():
        hit = np.array([bool(re.search(pattern, t)) for t in text])
        print(f"  {label:24s} {np.mean(hit[going]):5.0%} of those that go, {np.mean(hit[some & ~going]):5.0%} of those that stay")
    np.savez(args.data / "hml3d" / "census.npz", names=names, length=length, reach=reach, travel=travel, walk=said, mirrored=mirrored, part=part,
             **{"rule " + k: v for k, v in rules.items()})


def text(args) -> None:
    """The CLIP token features of every whole-clip caption of the motions that are 5.1 s or longer, as the cache has them for Nymeria's sentences."""
    from gray.scene_worker import Text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = args.data / "clip"
    reader = Text(device, weights if weights.is_dir() else None)
    captions = json.loads((args.data / "hml3d" / "texts.json").read_text())
    d = np.load(args.data / "hml3d" / "hml.npz")
    long_enough = {str(n) for n, length in zip(d["names"], np.diff(d["start"])) if length >= 76}
    tokens, where, total = [], {}, 0
    for name in sorted(long_enough):
        where[name] = []
        for caption in captions.get(name, []):
            feature = reader(caption).half().cpu().numpy()
            tokens.append(feature)
            where[name].append([total, total + len(feature)])
            total += len(feature)
    np.save(args.data / "hml3d" / "text_tokens.npy", np.concatenate(tokens))
    (args.data / "hml3d" / "text_index.json").write_text(json.dumps(where))
    print(f"{len(tokens):,d} captions of {len(where):,d} motions, {total:,d} tokens ({total * 768 * 2 / 1e9:.1f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("offset", "encode", "census", "text"))
    parser.add_argument("--data", type=Path, required=True, help="the folder with hml3d/ (and, for offset, body/ and tracks/ of some Nymeria recordings)")
    parser.add_argument("--steer", type=Path, default=Path("/u/enazaren/projects/STEER"))
    parser.add_argument("--recordings", type=int, default=61, help="offset: how many Nymeria recordings")
    parser.add_argument("--limit", type=int, default=None, help="encode: only the first motions")
    args = parser.parse_args()
    args.data, args.steer = args.data.expanduser(), args.steer.expanduser()
    {"offset": offset, "encode": encode, "census": census, "text": text}[args.step](args)


if __name__ == "__main__":
    main()
