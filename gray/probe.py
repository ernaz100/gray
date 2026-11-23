#!/usr/bin/env python3
"""Which image features let a small model read the room? A fixed test bed for candidate features.

    python -m gray.probe set                          # once: the frames, their pinhole images and the truth
    python -m gray.probe features --name dino-s-14    # once per candidate: its features for those frames
    python -m gray.probe fit --names token dino-s-7 dino-s-14 depth-base

If a small model cannot tell from the image features how far the next obstacle is, a generator
that gets the same features cannot be expected to avoid obstacles. So candidates are compared on
this before any generator is trained on them.

**set** picks frames of recordings that have a room map (gray/layout.py), 30 per recording, and
stores per recording <set>/<id>.npz with

    image    (F, 448, 448, 3) u8   the pinhole view of gray/pinhole.py, at 448 pixels
    walls    (F, 16) f32           distance to the nearest wall along 16 horizontal directions (0 is where the
                                   camera looks, counted to the left in steps of 22.5 degrees), up to 5 m, on the storey of the head
    free     (F, 16) f32           (truth step) the free distance from gray/ground.py, which does not use the furniture boxes: how far
                                   people are known to have walked (walkable) and where walls or repeatedly measured points begin
                                   (blocked); where the two agree within 0.3 m their mean, elsewhere nan. The features are compared on this
    seen     (F, 16) f32           (truth step) as "all", but only obstacles that the camera can see: some point between the
                                   obstacle's lowest and highest point falls inside the image. A counter edge under the camera does not count
    all      (F, 16) f32           the same for walls or furniture, not counting the piece of furniture the camera itself is in
                                   or over (someone sitting on a sofa is not blocked by it)
    tilt     (F, 4) f32            how the camera is tilted (the vertical part of its three axes) and its height above the floor;
                                   a robot knows these, so a probe may be given them
    frame    (F,) i32              the frame in the recording

**truth** recomputes walls and all for the frames of an existing set into <set>/<id>.truth.npz, which
the other steps read (so the images need not be extracted again when the definition changes).

**features** runs one candidate over the images and writes <set>/<id>.<name>.npy, (F, cells, numbers):

    token        DINOv3 ViT-S/16 class token at 224 px                       (1, 384)    what gray/pinhole.py stores
    dino-s-7     its patches, 14 x 14 averaged to 7 x 7                      (49, 384)   what gray/patches.py stores
    dino-s-14    its patches, 14 x 14                                        (196, 384)
    dino-s-448   its patches at 448 px, 28 x 28 averaged to 14 x 14          (196, 384)
    depth-small, depth-base, depth-large
                 Depth Anything V2 (metric, indoor) at 448 px: metres per pixel, averaged to 28 x 28   (1, 784)
    zoe          ZoeDepth (metric, indoor), the same output                  (1, 784)

In fit, two names joined by "+" (depth-large+dino-s-7) give the probe both.

**fit** trains one small probe per candidate (one hidden layer; with and without the camera tilt
as extra input) on the frames of the training locations and scores it on the held-out locations:

    error in view    mean absolute error (m) of the distance, for the directions inside the camera's view
                     (ahead and 22.5 degrees to either side; the view is 79.6 degrees wide)
    near ahead       how well "an obstacle within 1 m straight ahead" is told apart (area under the ROC curve; 0.5 is chance, 1 is perfect)

The main target is "free" (see above), on the directions where it is known; the box-based ones are kept for reference.

"constant" (the median distance per direction) is the reference. Results go to <set>/fit.json.

**geometry** needs no probe: a depth candidate gives metres per pixel, the camera's tilt and
height are known, so every pixel becomes a point in the room. Points between 0.3 and 1.5 m above
the floor are obstacles; per direction the nearest ones give the distance. This tests the depth
model against the room maps directly, and the maps against what the camera saw.

**memory** asks whether earlier frames help: on frames that have a 7 x 7 image grid now, 1 s ago and
2 s ago (gray/patches.py), a probe that gets only the grid of now is compared with one that gets all
three, each with where its camera was. An obstacle under the camera now was in view a moment ago.

**look** draws a few held-out frames: the image, the depth, and seen from above the obstacle
points from the depth next to the map's distances, to check by eye that image and truth agree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gray import frames
from gray.features import CACHE, RATE
from gray.layout import CELL, Layout
from gray.moments import held_locations
from gray.pinhole import STEER, VIEW
from gray.rectify import SEQUENCES, calibration, lookup, mounting, rectify, video_frame
from gray.tracks import to_world

SET = Path("/ptmp/enazaren/gray/probeset")
WEIGHTS = Path("/u/enazaren/steer_assets/weights/hf")  # models from the Hugging Face hub, fetched on the login node
RAYS, FAR, STEP = 16, 5.0, 0.05
IN_VIEW = [15, 0, 1]
SIDE = 448
DEPTH = {"depth-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "depth-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
         "depth-large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "zoe": "Intel/zoedepth-nyu"}
HALF = ("zoe",)  # models that expect colours scaled to -1..1 instead of the ImageNet mean and deviation


def distances(layout: Layout, position: np.ndarray, ahead: np.ndarray, furniture: bool, up: np.ndarray | None = None) -> np.ndarray:
    """Distance (F, 16) to the nearest obstacle from world positions (F, 3) whose view direction has the world angle ``ahead`` (F,); nan off the map.

    ``up`` (F, 3): the room's up direction in the camera's sensor frame. With it only obstacles count that the camera can
    see: some point of the obstacle, between its lowest and highest point, falls inside the image.
    """
    d = layout.data
    storey = layout.storeys_world(position)
    turn = np.arange(RAYS) * 2 * math.pi / RAYS
    angle = ahead[:, None] + turn[None]  # (F, 16)
    reach = np.arange(STEP, FAR + STEP / 2, STEP)  # (R,)
    points = position[:, None, None, :2] + reach[None, None, :, None] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)[:, :, None, :]  # (F, 16, R, 2)
    cell = np.floor((points - d["origin"]) / CELL).astype(int)
    rows, cols = d["floor"].shape[1:]
    inside = (cell[..., 0] >= 0) & (cell[..., 0] < cols) & (cell[..., 1] >= 0) & (cell[..., 1] < rows)
    x, y = np.clip(cell[..., 0], 0, cols - 1), np.clip(cell[..., 1], 0, rows - 1)
    level = np.maximum(storey, 0)[:, None, None]
    wall = d["wall"][level, y, x]
    solid = wall
    piece = d["furniture"][level, y, x]
    if furniture:
        # not the piece the camera itself is in or over (a person sitting on a sofa or leaning over a table is not blocked by it)
        here = np.floor((position[:, :2] - d["origin"]) / CELL).astype(int)
        own = d["furniture"][np.maximum(storey, 0), np.clip(here[:, 1], 0, rows - 1), np.clip(here[:, 0], 0, cols - 1)]
        solid = solid | ((piece > 0) & (piece != own[:, None, None]))
    if up is not None:
        floor = d["levels"][np.maximum(storey, 0)][:, None, None]
        low = np.where(wall, floor, np.concatenate([[0.0], d["furniture_low"]])[piece])
        high = np.where(wall, floor + 2.5, np.concatenate([[0.0], d["furniture_high"]])[piece])
        up = up / np.linalg.norm(up, axis=1, keepdims=True)
        level_ahead = np.array([0.0, 0.0, 1.0]) - up[:, 2:3] * up  # the view direction laid flat, in the sensor frame
        level_ahead /= np.maximum(np.linalg.norm(level_ahead, axis=1, keepdims=True), 1e-6)
        left = np.cross(up, level_ahead)
        forward, sideways = reach[None, None, :] * np.cos(turn)[None, :, None], reach[None, None, :] * np.sin(turn)[None, :, None]  # (1, 16, R)
        limit = math.tan(math.radians(VIEW) / 2)
        seen = np.zeros(solid.shape, dtype=bool)
        for share in (0.0, 0.5, 1.0):
            rise = np.clip(low + share * (high - low), floor, floor + 2.5) - position[:, 2, None, None]
            point = (forward[..., None] * level_ahead[:, None, None] + sideways[..., None] * left[:, None, None] + rise[..., None] * up[:, None, None])  # (F, 16, R, 3)
            seen |= (point[..., 2] > 0.05) & (np.abs(point[..., 0]) <= limit * point[..., 2]) & (np.abs(point[..., 1]) <= limit * point[..., 2])
        solid = solid & seen
    hit = inside & solid
    out = np.where(hit.any(-1), reach[hit.argmax(-1)], FAR)
    out[storey < 0] = np.nan
    return out


def collect(job: tuple) -> str | None:
    """The frames, images and truth of one recording."""
    name, tracks, maps, videos, out, count, seed = job
    target = Path(out) / f"{name}.npz"
    if target.exists():
        return name
    if not (Path(maps) / f"{name}.npz").exists() or not (Path(tracks) / f"{name}.npz").exists():
        return None
    track = np.load(Path(tracks) / f"{name}.npz")
    layout = Layout(Path(maps) / f"{name}.npz")
    usable = np.flatnonzero(track["valid"])
    picked = np.sort(np.random.default_rng(seed).choice(usable, size=min(3 * count, len(usable)), replace=False))
    pose = track["pose"][picked]
    position = to_world(pose[:, :3].astype(np.float64), track["floor"][picked])
    picked = picked[layout.storeys_world(position) >= 0][:: 3][:count]  # on a storey that has a map
    if len(picked) == 0:
        return None
    pose = track["pose"][picked]
    position = to_world(pose[:, :3].astype(np.float64), track["floor"][picked])
    yaw = frames.heading(torch.from_numpy(pose[:, 3:])).numpy()  # the ground direction (-sin, 0, cos) is world (-sin, -cos)
    ahead = np.arctan2(-np.cos(yaw), -np.sin(yaw))
    sequence = Path(videos) / name
    lens, on_glasses = calibration(sequence)
    clockwise = bool(mounting(sequence, on_glasses)[0] > 0)
    where, images = None, []
    for frame in picked:
        full = torch.from_numpy(video_frame(sequence / "preview_rgb.mp4", frame / RATE).copy()).permute(2, 0, 1)[None].float()
        if where is None:
            where = lookup(lens, full.shape[-1], clockwise, view=VIEW, wide=2 * SIDE, high=2 * SIDE).float()
        images.append(torch.nn.functional.avg_pool2d(rectify(full, where), 2)[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy())
    np.savez(target, image=np.stack(images), walls=distances(layout, position, ahead, False).astype(np.float32),
             all=distances(layout, position, ahead, True).astype(np.float32),
             tilt=np.concatenate([pose[:, 6:9], pose[:, 1:2]], axis=1).astype(np.float32), frame=picked.astype(np.int32))
    return name


def again(job: tuple) -> bool:
    """The truth of one recording of the set, recomputed."""
    path, tracks, maps, ground, location = job
    path = Path(path)
    frame = np.load(path)["frame"]
    track, layout = np.load(Path(tracks) / f"{path.stem}.npz"), Layout(Path(maps) / f"{path.stem}.npz")
    pose = track["pose"][frame]
    position = to_world(pose[:, :3].astype(np.float64), track["floor"][frame])
    yaw = frames.heading(torch.from_numpy(pose[:, 3:])).numpy()
    ahead = np.arctan2(-np.cos(yaw), -np.sin(yaw))
    here = np.floor((position[:, :2] - layout.data["origin"]) / CELL).astype(int)
    rows, cols = layout.data["floor"].shape[1:]
    inside = layout.data["furniture"][np.maximum(layout.storeys_world(position), 0), np.clip(here[:, 1], 0, rows - 1), np.clip(here[:, 0], 0, cols - 1)] > 0
    extra = {}
    if (Path(ground) / f"{location}.npz").exists():  # the truth that does not rest on the furniture boxes (gray/ground.py)
        from gray.ground import Ground

        both = Ground(Path(ground) / f"{location}.npz").distances(position, ahead)
        sure = np.abs(both["walkable"] - both["blocked"]) <= 0.3
        extra = {"walkable": both["walkable"].astype(np.float32), "blocked": both["blocked"].astype(np.float32),
                 "free": np.where(sure, (both["walkable"] + both["blocked"]) / 2, np.nan).astype(np.float32)}
    np.savez(path.with_name(f"{path.stem}.truth.npz"), walls=distances(layout, position, ahead, False).astype(np.float32),
             all=distances(layout, position, ahead, True).astype(np.float32), inside=inside,
             seen=distances(layout, position, ahead, True, pose[:, 6:9].astype(np.float64)).astype(np.float32), **extra)
    return True


def truth_again(args) -> None:
    files = [p for p in sorted(args.set.glob("*.npz")) if not p.name.endswith(".truth.npz")]
    location = {m["sequence_id"]: m["location"] for m in json.loads((args.cache / "index.json").read_text())["sequences"]}
    with Pool(args.workers) as pool:
        done = sum(pool.imap_unordered(again, [(str(p), str(args.tracks), str(args.maps), str(args.ground), location[p.stem]) for p in files], chunksize=4))
    print(f"truth recomputed for {done} recordings", flush=True)


def read(paths: list[Path], key: str) -> torch.Tensor:
    """One array of the set for some recordings; walls and all come from the recomputed truth when it exists."""
    def source(p: Path) -> Path:
        other = p.with_name(f"{p.stem}.truth.npz")
        return other if key in ("walls", "all", "inside", "seen", "walkable", "blocked", "free") and other.exists() else p
    return torch.from_numpy(np.concatenate([np.load(source(p))[key] for p in paths]))


def build(args) -> None:
    args.set.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in args.maps.glob("*.npz") if p.stem != "windows")[args.part :: args.parts]  # the maps, not the set
    jobs = [(n, str(args.tracks), str(args.maps), str(args.videos), str(args.set), args.frames, i) for i, n in enumerate(names)]
    with Pool(args.workers) as pool:
        done = [n for n in pool.imap_unordered(collect, jobs, chunksize=1) if n]
    print(f"{len(done)} of {len(names)} recordings in {args.set}", flush=True)


@torch.no_grad()
def extract(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if args.name in DEPTH:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModelForDepthEstimation

        model = AutoModelForDepthEstimation.from_pretrained(DEPTH[args.name], cache_dir=WEIGHTS).to(device).eval()
    else:
        sys.path.insert(0, str(STEER))
        from data.nymeria.ego_encoder import DINOV3_REPO, DINOV3_WEIGHTS, _load_dinov3

        model = _load_dinov3(DINOV3_REPO, DINOV3_WEIGHTS).to(device).eval()
    files = sorted(p for p in args.set.glob("*.npz") if not p.name.endswith(".truth.npz"))
    for i, path in enumerate(files):
        target = path.with_name(f"{path.stem}.{args.name}.npy")
        if target.exists():
            continue
        image = torch.from_numpy(np.load(path)["image"]).to(device).permute(0, 3, 1, 2).float() / 255  # (F, 3, 448, 448)
        image = (image - 0.5) / 0.5 if args.name in HALF else (image - mean) / std
        small = torch.nn.functional.avg_pool2d(image, 2)
        if args.name in DEPTH:
            depth = torch.cat([model(pixel_values=image[i : i + 8]).predicted_depth.float() for i in range(0, len(image), 8)])
            out = torch.nn.functional.adaptive_avg_pool2d(depth.reshape(len(image), 1, *depth.shape[-2:]), 28).flatten(1)[:, None]
        elif args.name == "token":
            out = model.forward_features(small)["x_norm_clstoken"][:, None]
        else:
            tokens = model.forward_features(image if args.name == "dino-s-448" else small)["x_norm_patchtokens"].float()
            side = int(round(tokens.shape[1] ** 0.5))
            grid = tokens.reshape(-1, side, side, tokens.shape[-1]).permute(0, 3, 1, 2)
            pool = {"dino-s-7": 2, "dino-s-14": 1, "dino-s-448": 2}[args.name]
            out = torch.nn.functional.avg_pool2d(grid, pool).flatten(2).permute(0, 2, 1)
        np.save(target, out.half().cpu().numpy())
        if i % 100 == 0:
            print(f"  {i}/{len(files)} {tuple(out.shape[1:])}", flush=True)
    print(f"wrote {args.name} for {len(files)} recordings", flush=True)


class Probe(nn.Module):
    """One hidden layer on the features (each cell first reduced to 64 numbers), optionally with the camera tilt."""

    def __init__(self, cells: int, numbers: int, tilt: int, normalize: bool) -> None:
        """``tilt``: how many extra numbers per frame are appended (4 for the camera tilt and height, 0 for none)."""
        super().__init__()
        self.tilt, self.many = tilt, cells > 1
        norm = nn.LayerNorm(numbers) if normalize else nn.Identity()
        self.cell = nn.Sequential(norm, nn.Linear(numbers, 64)) if self.many else norm
        self.where = nn.Parameter(torch.zeros(cells, 64))
        width = (cells * 64 if self.many else numbers) + tilt
        self.head = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Linear(512, RAYS))

    def forward(self, x: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
        x = self.cell(x.float())
        x = (x + self.where).flatten(1) if self.many else x.flatten(1)
        return self.head(torch.cat([x, tilt[:, : self.tilt]], dim=1) if self.tilt else x)


def area(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve."""
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    n, m = positive.sum(), (~positive).sum()
    return float((rank[positive].sum() - n * (n + 1) / 2) / max(n * m, 1))


def train(x: dict, tilt: dict, y: dict, with_tilt: bool, normalize: bool, device: str, epochs: int) -> torch.Tensor:
    """Predicted distances (N, 16) for the held-out frames."""
    torch.manual_seed(0)
    model = Probe(x["train"].shape[1], x["train"].shape[2], tilt["train"].shape[1] if with_tilt else 0, normalize).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    target, lean = y["train"].to(device), tilt["train"].to(device)
    for _epoch in range(epochs):
        for index in torch.randperm(len(target)).split(256):
            known = torch.isfinite(target[index])  # "free" is only known for some directions
            loss = nn.functional.smooth_l1_loss(model(x["train"][index].to(device), lean[index])[known], target[index][known])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x["held"][i : i + 1024].to(device), tilt["held"][i : i + 1024].to(device)) for i in range(0, len(y["held"]), 1024)]).clamp(0, FAR).cpu()


def fit(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    location = {m["sequence_id"]: m["location"] for m in json.loads((args.cache / "index.json").read_text())["sequences"]}
    held_out = held_locations(args.cache)
    needed = [n for name in args.names for n in name.split("+")]
    files = [p for p in sorted(args.set.glob("*.npz")) if not p.name.endswith(".truth.npz") and all(p.with_name(f"{p.stem}.{n}.npy").exists() for n in needed)]
    part = {"train": [p for p in files if location[p.stem] not in held_out], "held": [p for p in files if location[p.stem] in held_out]}
    truth = {key: {split: read(paths, key) for split, paths in part.items()} for key in ("walls", "all", "free", "tilt")}
    sure = torch.isfinite(truth["free"]["held"][:, IN_VIEW])
    span = truth["free"]["held"][:, IN_VIEW][sure]
    print(f"free distance without the boxes: known for {float(sure.float().mean()):.0%} of the held-out in-view directions; there mean {span.mean():.2f} m, "
          f"under 1 m in {float((span < 1).float().mean()):.0%}, 1 to 2.5 m in {float(((span >= 1) & (span < 2.5)).float().mean()):.0%}")
    ahead_known = torch.isfinite(truth["free"]["held"][:, 0]).numpy()
    ahead_near = (truth["free"]["held"][:, 0] < 1.0).numpy()[ahead_known]
    def free_scores(predicted: torch.Tensor) -> dict:
        error = (predicted - truth["free"]["held"]).abs()[:, IN_VIEW][sure]
        return {"free": float(error.mean()), "free under 1 m": float(error[span < 1].mean()), "free 1 to 2.5 m": float(error[(span >= 1) & (span < 2.5)].mean()),
                "free 2.5 m and more": float(error[span >= 2.5].mean()), "free near": area(-predicted[:, 0].numpy()[ahead_known], ahead_near)}
    print(f"{len(truth['all']['train']):,d} training frames from {len(part['train'])} recordings, {len(truth['all']['held']):,d} held-out frames from {len(part['held'])}", flush=True)
    near = (truth["all"]["held"][:, 0] < 1.0).numpy()
    print(f"held-out frames: mean distance in view {truth['walls']['held'][:, IN_VIEW].mean():.2f} m to a wall, {truth['all']['held'][:, IN_VIEW].mean():.2f} m to a wall or furniture; "
          f"an obstacle within 1 m straight ahead in {near.mean():.0%}")
    rows = {}
    constant = {key: truth[key]["train"].nanmedian(0).values for key in ("walls", "all", "free")}
    rows["constant"] = {"walls": float((truth["walls"]["held"] - constant["walls"]).abs()[:, IN_VIEW].mean()),
                        "all": float((truth["all"]["held"] - constant["all"]).abs()[:, IN_VIEW].mean()), "near": 0.5} | free_scores(constant["free"][None].expand_as(truth["free"]["held"]).clone())
    rows["constant"]["free near"] = 0.5
    none = {split: torch.zeros(len(truth["all"][split]), 1, 1) for split in part}
    candidates = [("camera tilt only", none, truth["tilt"], True, False)]
    load = lambda name: {split: torch.from_numpy(np.concatenate([np.load(p.with_name(f"{p.stem}.{name}.npy")) for p in paths])) for split, paths in part.items()}  # noqa: E731
    for name in args.names:
        if "+" in name:  # a depth candidate and another one together: the depth joins the camera tilt as extra numbers
            depth_name, other = name.split("+")
            depth, x = load(depth_name), load(other)
            both = {split: torch.cat([truth["tilt"][split], depth[split].float().flatten(1) / FAR], dim=1) for split in part}
            candidates += [(name + " + camera tilt", x, both, True)]
            continue
        x = load(name)
        if name in DEPTH:
            x = {split: value.float() / FAR for split, value in x.items()}
        candidates += [(name, x, truth["tilt"], False, name not in DEPTH), (name + " + camera tilt", x, truth["tilt"], True, name not in DEPTH)]
    candidates = [c if len(c) == 5 else (c[0], c[1], c[2], True, True) for c in candidates]
    far = truth["all"]["held"][:, IN_VIEW]
    bands = {"under 1 m": far < 1.0, "1 to 2.5 m": (far >= 1.0) & (far < 2.5), "2.5 m and more": far >= 2.5}
    print("in-view directions by the true distance to a wall or furniture: " + ", ".join(f"{k} {float(v.float().mean()):.0%}" for k, v in bands.items()))
    banded = lambda predicted: {k: float((predicted - truth["all"]["held"]).abs()[:, IN_VIEW][v].mean()) for k, v in bands.items()}  # noqa: E731
    rows["constant"] |= banded(constant["all"][None].expand_as(truth["all"]["held"]))
    for label, x, extra, with_tilt, normalize in candidates:
        predicted = {key: train(x, extra, truth[key], with_tilt, normalize, device, args.epochs) for key in ("walls", "all", "free")}
        rows[label] = {"walls": float((predicted["walls"] - truth["walls"]["held"]).abs()[:, IN_VIEW].mean()),
                       "all": float((predicted["all"] - truth["all"]["held"]).abs()[:, IN_VIEW].mean()), "near": area(-predicted["all"][:, 0].numpy(), near)} | banded(predicted["all"]) | free_scores(predicted["free"])
    print(f"\n  {'features':34s} free distance (without the boxes): error in view   under 1 m   1 to 2.5 m   2.5 m and more   near ahead (AUC)  |  boxes: walls   walls or furniture   near ahead")
    for label, row in rows.items():
        print(f"  {label:34s} {row['free']:50.2f} {row['free under 1 m']:11.2f} {row['free 1 to 2.5 m']:12.2f} {row['free 2.5 m and more']:16.2f} {row['free near']:18.2f}  |  "
              f"{row['walls']:12.2f} {row['all']:20.2f} {row['near']:12.2f}", flush=True)
    previous = json.loads((args.set / "fit.json").read_text()) if (args.set / "fit.json").exists() else {}
    (args.set / "fit.json").write_text(json.dumps(previous | rows, indent=1))
    print(f"\nwrote {args.set / 'fit.json'}", flush=True)


def points(depth: torch.Tensor, tilt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Depth (N, 28, 28) in metres and camera tilt (N, 4) -> for every cell its height above the floor, distance on the ground
    and direction (radians to the left of where the camera looks), each (N, 784)."""
    side = depth.shape[-1]
    focal = side / 2 / math.tan(math.radians(VIEW) / 2)
    v, u = torch.meshgrid(torch.arange(side, dtype=torch.float32), torch.arange(side, dtype=torch.float32), indexing="ij")
    right, down = (u - (side - 1) / 2) / focal, (v - (side - 1) / 2) / focal
    # the sensor's own axes (gray/rectify.py, video turned clockwise): x is down in the image, y is left, z looks ahead
    ray = torch.stack([down, -right, torch.ones_like(right)], dim=-1).reshape(1, -1, 3)
    cloud = depth.reshape(len(depth), -1, 1).float() * ray  # (N, 784, 3) in the sensor frame
    up = tilt[:, None, :3].float()  # the room's up direction in the sensor frame
    ahead = torch.tensor([0.0, 0.0, 1.0]) - up[..., 2:3] * up
    ahead = ahead / ahead.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left = torch.linalg.cross(up, ahead, dim=-1)
    forward, sideways = (cloud * ahead).sum(-1), (cloud * left).sum(-1)
    return tilt[:, 3:4].float() + (cloud * up).sum(-1), torch.hypot(forward, sideways), torch.atan2(sideways, forward)


def from_depth(depth: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
    """Distance (N, 16) to the nearest obstacle per direction, read off the depth; 5 m where no obstacle point falls."""
    height, distance, angle = points(depth, tilt)
    solid = (height > 0.3) & (height < 1.5)
    ray = torch.round(angle / (2 * math.pi / RAYS)).long() % RAYS
    out = torch.full((len(depth), RAYS), FAR)
    for r in range(RAYS):
        out[:, r] = torch.where(solid & (ray == r), distance, torch.full_like(distance, FAR)).amin(dim=1)
    return out.clamp(max=FAR)


def recall(job: tuple) -> dict | None:
    """Frames of one recording that have an image grid now, 1 s ago and 2 s ago, with the truth now."""
    name, tracks, maps, grids, count, seed, ground = job
    if not all(p.exists() for p in (Path(maps) / f"{name}.npz", Path(grids) / name / "rows.npy", Path(tracks) / f"{name}.npz", Path(ground))):
        return None
    track, rows = np.load(Path(tracks) / f"{name}.npz"), np.load(Path(grids) / name / "rows.npy")
    t = np.arange(30, len(rows))
    usable = t[(rows[t] >= 0) & (rows[t - 15] >= 0) & (rows[t - 30] >= 0) & track["valid"][t]]
    if len(usable) == 0:
        return None
    picked = np.sort(np.random.default_rng(seed).choice(usable, size=min(count, len(usable)), replace=False))
    layout = Layout(Path(maps) / f"{name}.npz")
    pose = track["pose"][picked]
    position = to_world(pose[:, :3].astype(np.float64), track["floor"][picked])
    keep = layout.storeys_world(position) >= 0
    picked, pose, position = picked[keep], pose[keep], position[keep]
    if len(picked) == 0:
        return None
    yaw = frames.heading(torch.from_numpy(pose[:, 3:])).numpy()
    ahead = np.arctan2(-np.cos(yaw), -np.sin(yaw))
    three = torch.from_numpy(np.stack([track["pose"][picked - back] for back in (0, 15, 30)], axis=1))  # (F, 3, 9): now, 1 s ago, 2 s ago
    moved = frames.anchor(three, torch.zeros(len(picked), dtype=torch.long))  # in the frame of now
    where = torch.stack([moved[..., 0], moved[..., 2], torch.sin(frames.heading(moved[..., 3:])), torch.cos(frames.heading(moved[..., 3:]))], dim=-1)
    camera = torch.cat([where, three[..., 6:9], three[..., 1:2]], dim=-1).numpy().astype(np.float32)  # (F, 3, 8): where it was, and its tilt and height
    grid = np.load(Path(grids) / name / "patches.npy", mmap_mode="r")
    from gray.ground import Ground

    both = Ground(Path(ground)).distances(position, ahead)
    free = np.where(np.abs(both["walkable"] - both["blocked"]) <= 0.3, (both["walkable"] + both["blocked"]) / 2, np.nan).astype(np.float32)
    return {"name": name, "grid": np.stack([np.stack([grid[rows[f - back]] for back in (0, 15, 30)]) for f in picked]), "camera": camera, "free": free,
            "walls": distances(layout, position, ahead, False).astype(np.float32), "all": distances(layout, position, ahead, True).astype(np.float32),
            "seen": distances(layout, position, ahead, True, pose[:, 6:9].astype(np.float64)).astype(np.float32)}


class Recall(nn.Module):
    """The probe for one or several frames: every cell reduced to 64 numbers, every frame with where its camera was."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count
        self.cell = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 64))
        self.where = nn.Parameter(torch.zeros(49, 64))
        self.head = nn.Sequential(nn.Linear(count * (49 * 64 + 8), 512), nn.GELU(), nn.Linear(512, RAYS))

    def forward(self, grid: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
        grid, camera = grid[:, : self.count].float(), camera[:, : self.count]
        cells = (self.cell(grid) + self.where).flatten(2)
        return self.head(torch.cat([cells, camera], dim=-1).flatten(1))


def memory(args) -> None:
    """Does the probe read more of the room when it also gets the image grids of 1 and 2 s ago?"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    location = {m["sequence_id"]: m["location"] for m in json.loads((args.cache / "index.json").read_text())["sequences"]}
    held_out = held_locations(args.cache)
    jobs = [(n, str(args.tracks), str(args.maps), str(args.grids), args.frames, i, str(args.ground / f"{location[n]}.npz")) for i, n in enumerate(sorted(location))]
    with Pool(args.workers) as pool:
        results = [r for r in pool.imap_unordered(recall, jobs, chunksize=2) if r]
    sets = {}
    for split, part in (("train", [r for r in results if location[r["name"]] not in held_out]), ("held", [r for r in results if location[r["name"]] in held_out])):
        sets[split] = {key: torch.from_numpy(np.concatenate([r[key] for r in part])) for key in ("grid", "camera", "walls", "all", "seen", "free")}
        print(f"{split}: {len(sets[split]['all']):,d} frames from {len(part)} recordings", flush=True)
    held = sets["held"]
    under = held["all"][:, IN_VIEW] < 1.0
    print(f"held-out frames, directions in view: mean distance {held['all'][:, IN_VIEW].mean():.2f} m to a wall or furniture, {held['seen'][:, IN_VIEW].mean():.2f} m to one the camera can see; "
          f"under 1 m in {float(under.float().mean()):.0%}, of which the camera can see {float((held['seen'][:, IN_VIEW] < 1.0)[under].float().mean()):.0%}")
    rows = {}
    known = torch.isfinite(held["free"][:, IN_VIEW])
    span = held["free"][:, IN_VIEW][known]
    ahead_known = torch.isfinite(held["free"][:, 0]).numpy()
    print(f"free distance (gray/ground.py) known for {float(known.float().mean()):.0%} of the held-out in-view directions; under 1 m in {float((span < 1).float().mean()):.0%}")
    for label, count in (("constant", 0), ("grid of now", 1), ("grids of now, 1 s and 2 s ago", 3)):
        row = {}
        for key in ("walls", "free"):
            if count == 0:
                predicted = sets["train"][key].nanmedian(0).values[None].expand_as(held[key])
            else:
                torch.manual_seed(0)
                model = Recall(count).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
                target, camera = sets["train"][key].to(device), sets["train"]["camera"].to(device)
                for _epoch in range(args.epochs):
                    for index in torch.randperm(len(target)).split(512):
                        sure = torch.isfinite(target[index])
                        loss = nn.functional.smooth_l1_loss(model(sets["train"]["grid"][index].to(device), camera[index])[sure], target[index][sure])
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                model.eval()
                with torch.no_grad():
                    predicted = torch.cat([model(held["grid"][i : i + 1024].to(device), held["camera"][i : i + 1024].to(device)) for i in range(0, len(held[key]), 1024)]).clamp(0, FAR).cpu()
            if key == "walls":
                row["walls"] = float((predicted - held["walls"]).abs()[:, IN_VIEW].mean())
            else:
                error = (predicted - held["free"]).abs()[:, IN_VIEW][known]
                row |= {"free": float(error.mean()), "under": float(error[span < 1].mean()), "middle": float(error[(span >= 1) & (span < 2.5)].mean()),
                        "far": float(error[span >= 2.5].mean()),
                        "near": area(-predicted[:, 0].numpy()[ahead_known], (held["free"][:, 0] < 1.0).numpy()[ahead_known]) if count else 0.5}
        rows[label] = row
    print(f"\n  {'probe':32s} free distance: error in view   under 1 m   1 to 2.5 m   2.5 m and more   blocked ahead (AUC)  |  walls of the maps")
    for label, row in rows.items():
        print(f"  {label:32s} {row['free']:29.2f} {row['under']:11.2f} {row['middle']:12.2f} {row['far']:16.2f} {row['near']:21.2f}  |  {row['walls']:17.2f}", flush=True)
    args.set.mkdir(parents=True, exist_ok=True)
    (args.set / "memory.json").write_text(json.dumps(rows, indent=1))


def held_frames(args, names: list[str], held: bool = True) -> tuple[list[Path], dict]:
    location = {m["sequence_id"]: m["location"] for m in json.loads((args.cache / "index.json").read_text())["sequences"]}
    held_out = held_locations(args.cache)
    files = [p for p in sorted(args.set.glob("*.npz")) if not p.name.endswith(".truth.npz") and (location[p.stem] in held_out) == held
             and all(p.with_name(f"{p.stem}.{n}.npy").exists() for n in names)]
    return files, {key: read(files, key) for key in ("walls", "all", "tilt")}


def geometry(args) -> None:
    """Obstacle distances read off the depth, as it comes and with its scale corrected."""
    rows = {}
    grid = np.arange(0.40, 1.21, 0.05)
    print(f"  {'depth model, no probe':38s} error in view, walls   error in view, walls or furniture   near ahead (AUC)")
    for name in args.names:
        scales = {"as it comes": 1.0}
        # the model was trained on images of one focal length; depth in metres has to be scaled by the ratio to ours
        files, truth = held_frames(args, [name], held=False)
        picked = files[:: max(1, len(files) // 150)]
        depth = torch.from_numpy(np.concatenate([np.load(p.with_name(f"{p.stem}.{name}.npy")) for p in picked])).float().reshape(-1, 28, 28)
        target, tilt = read(picked, "all"), read(picked, "tilt")
        errors = [float((from_depth(depth * g, tilt) - target).abs()[:, IN_VIEW].mean()) for g in grid]
        scales[f"scaled by {grid[int(np.argmin(errors))]:.2f} (best on training locations)"] = float(grid[int(np.argmin(errors))])
        files, truth = held_frames(args, [name])
        depth = torch.from_numpy(np.concatenate([np.load(p.with_name(f"{p.stem}.{name}.npy")) for p in files])).float().reshape(-1, 28, 28)
        near = (truth["all"][:, 0] < 1.0).numpy()
        for label, scale in scales.items():
            predicted = from_depth(depth * scale, truth["tilt"])
            rows[f"{name}, geometry, {label}"] = row = {"walls": float((predicted - truth["walls"]).abs()[:, IN_VIEW].mean()),
                                                         "all": float((predicted - truth["all"]).abs()[:, IN_VIEW].mean()), "near": area(-predicted[:, 0].numpy(), near)}
            print(f"  {name + ', ' + label:38s} {row['walls']:20.2f} {row['all']:34.2f} {row['near']:18.2f}", flush=True)
    previous = json.loads((args.set / "fit.json").read_text()) if (args.set / "fit.json").exists() else {}
    (args.set / "fit.json").write_text(json.dumps(previous | rows, indent=1))


def look(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = args.names[0]
    files, _truth = held_frames(args, [name])
    rng = np.random.default_rng(0)
    chosen = [(files[i], int(rng.integers(0, 20))) for i in rng.choice(len(files), size=args.frames_shown, replace=False)]
    figure, axes = plt.subplots(len(chosen), 3, figsize=(15, 4.6 * len(chosen)), squeeze=False)
    angles = np.arange(RAYS) * 2 * math.pi / RAYS
    for row, (path, index) in zip(axes, chosen):
        data = np.load(path)
        index = min(index, len(data["frame"]) - 1)
        depth = torch.from_numpy(np.load(path.with_name(f"{path.stem}.{name}.npy"))[index]).float().reshape(1, 28, 28) * args.scale
        tilt = torch.from_numpy(data["tilt"][index : index + 1])
        height, distance, angle = (v[0].numpy() for v in points(depth, tilt))
        row[0].imshow(data["image"][index])
        row[0].set_title(f"{path.stem}, frame {int(data['frame'][index])}", fontsize=8)
        shown = row[1].imshow(depth[0], cmap="viridis", vmin=0, vmax=5)
        figure.colorbar(shown, ax=row[1], fraction=0.046, label="m")
        row[1].set_title(f"{name}: depth; camera {tilt[0, 3]:.2f} m above the floor", fontsize=8)
        solid = (height > 0.3) & (height < 1.5)
        top = row[2]
        top.scatter(-distance[~solid] * np.sin(angle[~solid]), distance[~solid] * np.cos(angle[~solid]), s=4, color="0.8", label="depth points: floor, ceiling")
        top.scatter(-distance[solid] * np.sin(angle[solid]), distance[solid] * np.cos(angle[solid]), s=6, color="tab:blue", label="depth points at body height")
        other = path.with_name(f"{path.stem}.truth.npz")
        maps = np.load(other) if other.exists() else data
        for key, colour in (("walls", "black"), ("all", "tab:red")):
            d = maps[key][index]
            top.plot(-np.append(d, d[0]) * np.sin(np.append(angles, 0)), np.append(d, d[0]) * np.cos(np.append(angles, 0)), color=colour, marker="o", markersize=3,
                     linewidth=1, label={"walls": "map: nearest wall", "all": "map: nearest wall or furniture"}[key])
        half = math.radians(VIEW / 2)
        for sign in (-1, 1):
            top.plot([0, -5 * math.sin(sign * half)], [0, 5 * math.cos(sign * half)], color="tab:green", linewidth=0.8, linestyle="--")
        top.set(xlim=(-5.2, 5.2), ylim=(-5.2, 5.2), title="from above: the camera at the centre, looking up the page (green: edge of view)")
        top.set_aspect("equal")
        top.legend(fontsize=6, loc="lower left")
        for axis in row[:2]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.set / f"look-{name}.png", dpi=70)
    print(f"wrote {args.set / f'look-{name}.png'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("set", "truth", "features", "fit", "geometry", "look", "memory"))
    parser.add_argument("--ground", type=Path, default=Path("/ptmp/enazaren/gray/ground"), help="truth: the free and blocked space of gray/ground.py")
    parser.add_argument("--grids", type=Path, default=Path("/ptmp/enazaren/gray/patches"), help="memory: the image grids of gray/patches.py")
    parser.add_argument("--scale", type=float, default=1.0, help="look: factor on the depth")
    parser.add_argument("--frames-shown", type=int, default=8, help="look: how many frames")
    parser.add_argument("--set", type=Path, default=SET)
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--frames", type=int, default=30, help="set: frames per recording")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1, help="set: do recordings part, part + parts, ...")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--name", help="features: the candidate")
    parser.add_argument("--names", nargs="+", default=[], help="fit: the candidates to compare")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    {"set": build, "truth": truth_again, "features": extract, "fit": fit, "geometry": geometry, "look": look, "memory": memory}[args.step](args)


if __name__ == "__main__":
    main()
