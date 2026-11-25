#!/usr/bin/env python3
"""HumanML3D motions placed into recorded frames of real homes, keeping those that do not go through anything.

    python -m gray.place solid                                   # per home, what stands where in 3D; writes <ground>/solid/<home>.npz
    python -m gray.place joints --hml <folder>                   # the joints of every HumanML3D frame; writes <hml>/joints.npy
    python -m gray.place pairs --hml <folder> [--draws 16]       # which motions fit where; writes <hml>/placed.npz

HumanML3D has many more kinds of motion and wording than Nymeria, but no images (gray/hml.py), so until now its samples
could teach nothing about scenes. Here a HumanML3D motion is given a real image: take a training moment of Nymeria,
put the motion's camera at now onto the recorded camera at now (same place on the ground, same direction of view), and
check the motion's next 5 s against the home's map (gray/ground.py). If it fits, the pair (this frame, this motion) is a
new sample: the image of a real room, a sentence, and a body and head path that are possible in that room.

A motion fits a frame when, over the 5 s after now (every 0.2 s):
    not blocked   neither the camera nor the pelvis lies in a blocked cell of the map
    walked        and both lie where somebody has walked in the recordings (stricter: the map knows no floor elsewhere)
    body          and no part of the body is inside a wall or inside something that stands there: all 22 joints and the middles
                  of the 21 bones between them, against the walls of the map and against **solid**
    height        and the motion's camera at now is within 10 cm of the recorded camera's height (the image was taken from there)

**solid** is the 3D counterpart of the blocked cells of gray/ground.py, from the same measured points: a 10 cm cube is
solid when it holds measured points in at least 30 % of the home's recordings (and at least 3). Things that stand there
are measured every time; people and what they carry are not. The 15 cm above each floor are left out (the floor itself),
and so is every cube above a cell people walked through. So a hand may pass over a table, and a knee may not pass through it.
The recorded bodies of a sample of the same frames are put through the same rule: what they score is the rule's error.

Printed: how many draws pass each rule, by how far the motion goes, next to what the recorded 5 s of the same frames do
under the same rules. Written: the pairs that pass all three, at most ``--keep`` per frame.

    placed.npz   row (the moment's row in the listing), sample and now (the motion: its index in gray/hml.py's samples, and the
                 frame at now in hml.npz), travel (m the camera goes in the 5 s)

``Placed`` gives these pairs as training samples (gray/head.py and gray/motion.py, ``--placed``): the image feature and image
grid of the recorded frame at now; the motion's own camera poses, body and caption; nothing before now (the 2 s before now in
the room were another person's), so the models learn these as starts from a single frame.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from gray import frames
from gray.ground import Ground
from gray.hml import HmlMoments
from gray.moments import FUTURE, HISTORY
from gray.tracks import to_world

NOW = HISTORY - 1
CUBE = 0.10  # metres
FLOOR = 0.15  # metres above a floor that are not solid
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]  # of the 22 joints


class Solid:
    """What stands where in one home, in 10 cm cubes."""

    def __init__(self, path: Path) -> None:
        d = np.load(path)
        self.origin, self.grid = d["origin"], d["solid"]

    def inside(self, points: np.ndarray) -> np.ndarray:
        """World positions (N, 3) -> is the cube there solid."""
        cell = np.floor((points - self.origin) / CUBE).astype(int)
        levels, rows, cols = self.grid.shape
        ok = (cell >= 0).all(1) & (cell[:, 0] < cols) & (cell[:, 1] < rows) & (cell[:, 2] < levels)
        return ok & self.grid[np.clip(cell[:, 2], 0, levels - 1), np.clip(cell[:, 1], 0, rows - 1), np.clip(cell[:, 0], 0, cols - 1)]


def one_home(job: tuple[str, str]) -> str:
    home, folder = job
    from gray.ground import AT_LEAST, SHARE
    from gray.layout import CELL

    folder = Path(folder)
    target = folder / "solid" / f"{home}.npz"
    if target.exists():
        return f"{home}: there already"
    d = np.load(folder / f"{home}.npz")
    recordings = [str(r) for r in d["recordings"]]
    low = np.array([*d["origin"], float(d["levels"].min()) - 0.5])
    rows, cols = (int(math.ceil(v * CELL / CUBE)) for v in d["free"].shape[1:])
    levels = int(math.ceil((float(d["levels"].max()) + 2.5 - low[2]) / CUBE))
    seen = np.zeros((levels, rows, cols), dtype=np.uint16)
    for r in recordings:
        cell = np.floor((np.load(folder / "points" / f"{r}.npy") - low) / CUBE).astype(int)
        cell = cell[(cell >= 0).all(1) & (cell[:, 0] < cols) & (cell[:, 1] < rows) & (cell[:, 2] < levels)]
        flat = np.bincount(np.ravel_multi_index((cell[:, 2], cell[:, 1], cell[:, 0]), seen.shape), minlength=seen.size)
        seen += (flat >= 2).reshape(seen.shape)
    solid = seen >= max(AT_LEAST, math.ceil(SHARE * len(recordings)))
    height = low[2] + (np.arange(levels) + 0.5) * CUBE
    for z in d["levels"]:
        solid[np.abs(height - z) < FLOOR] = False
    walked = d["free"].any(0)  # on any storey
    y, x = np.nonzero(walked)
    over = np.zeros((rows, cols), dtype=bool)
    over[np.minimum((y * CELL / CUBE).astype(int), rows - 1), np.minimum((x * CELL / CUBE).astype(int), cols - 1)] = True
    solid[:, over] = False
    target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target.with_suffix(".tmp.npz"), origin=low, solid=solid)
    target.with_suffix(".tmp.npz").rename(target)
    return f"{home}: {len(recordings)} recordings, {int(solid.sum()):,d} solid cubes of {solid.size:,d}"


def solid(args: argparse.Namespace) -> None:
    from multiprocessing import get_context

    homes = sorted(p.stem for p in args.ground.glob("*.npz"))
    with get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(one_home, [(h, str(args.ground)) for h in homes]):
            print("  " + line, flush=True)


def joints(args: argparse.Namespace) -> None:
    from gray.chain import Decoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if getattr(torch.version, "hip", None):
        torch.backends.cuda.preferred_blas_library("cublas")
    d = np.load(args.hml / "hml.npz")
    decoder = Decoder(device)
    out = np.zeros((len(d["latent"]), 22, 3), dtype=np.float16)
    used = sorted({(a, b) for _name, a, b, _s in HmlMoments(args.hml).samples})
    for a, b in used:
        own = np.concatenate([d["latent"][a:b], d["pelvis"][a:b][:, [0, 2]], np.sin(d["yaw"][a:b])[:, None], np.cos(d["yaw"][a:b])[:, None]], axis=1)
        out[a:b] = decoder.joints(torch.from_numpy(own.astype(np.float32))[None].to(device))[0].cpu().numpy()
    np.save(args.hml / "joints.npy", out)
    print(f"wrote {args.hml / 'joints.npy'}: the joints of {len(used):,d} motions", flush=True)


def limbs(points):
    """Joints (..., 22, 3) -> the joints and the middles of the bones (..., 43, 3)."""
    return np.concatenate([points, (points[..., 1:, :] + points[..., PARENTS[1:], :]) / 2], axis=-2)
EVERY = 3  # checked every 0.2 s
HEIGHT = 0.10


class Placed(torch.utils.data.Dataset):
    """The pairs of placed.npz as samples shaped like those of gray/moments.py with a body."""

    def __init__(self, hml: Path, listing: Path, images: Path, patches: Path, pairs: Path | None = None) -> None:
        self.motions = HmlMoments(hml)
        self.pairs = dict(np.load(pairs or hml / "placed.npz"))
        self.table = np.load(listing)
        self.sequence, self.frame = self.table["sequence"], self.table["now"]
        self.images, self.patches = images, patches
        self._open: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.pairs["row"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.pairs["row"][index])
        out = self.motions.item(int(self.pairs["sample"][index]), now=int(self.pairs["now"][index]))
        name, frame = str(self.sequence[row]), int(self.frame[row])
        if name not in self._open:
            self._open[name] = {"image": np.load(self.images / name / "dino.npy", mmap_mode="r"), "rows": np.load(self.patches / name / "rows.npy"),
                                "grid": np.load(self.patches / name / "patches.npy", mmap_mode="r")}
        a = self._open[name]
        image = torch.zeros(HISTORY, 384)
        image[-1] = torch.from_numpy(np.array(a["image"][frame], dtype=np.float32))
        out |= {"image": image, "patches": torch.from_numpy(np.array(a["grid"][int(a["rows"][frame])], dtype=np.float32)), "foreign": torch.tensor(False), "history_max": torch.tensor(1)}
        return out


def pairs(args: argparse.Namespace) -> None:
    table = dict(np.load(args.listing))
    index = json.loads((args.cache / "index.json").read_text())["sequences"]
    location = {m["sequence_id"]: m["location"] for m in index}
    rows = np.flatnonzero(table["subset"] & ~table["held"] & table["has_map"])
    motions = HmlMoments(args.hml)
    pose_all, pelvis_all, yaw_all = (torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in (motions.pose, motions.pelvis, motions.yaw))
    spans = np.array([(a, b, s) for _name, a, b, s in motions.samples])
    rng = np.random.default_rng(args.seed)
    steps = np.arange(EVERY, FUTURE + 1, EVERY)
    print(f"{len(rows):,d} training moments with a map; {len(motions):,d} samples of {motions.motions:,d} HumanML3D motions; {args.draws} draws per moment", flush=True)
    whole = (args.hml / "joints.npy").exists() and (args.ground / "solid").exists()
    if not whole:
        print("no joints.npy or no solid/: the body rule is left out (run the steps joints and solid first)", flush=True)
    else:
        joints_all = np.load(args.hml / "joints.npy", mmap_mode="r")
        from gray.chain import Decoder

        decoder = Decoder("cuda" if torch.cuda.is_available() else "cpu")

    by_home: dict[str, list[int]] = {}
    for row in rows:
        by_home.setdefault(location[str(table["sequence"][row])], []).append(int(row))
    tracks: dict[str, dict] = {}
    kept = {"row": [], "sample": [], "now": [], "travel": []}
    tally = {key: [] for key in ("travel", "not_blocked", "walked", "body", "height", "all")}
    recorded = {key: [] for key in ("travel", "not_blocked", "walked")}
    recorded_body = []
    for home, everything in sorted(by_home.items()):
        ground = Ground(args.ground / f"{home}.npz")
        things = Solid(args.ground / "solid" / f"{home}.npz") if whole else None
        fits = 0
        for part in range(0, len(everything), 500):
            own = everything[part : part + 500]
            here, floor = [], []
            for row in own:
                name = str(table["sequence"][row])
                if name not in tracks:
                    track = np.load(args.tracks / f"{name}.npz")
                    tracks[name] = {"pose": track["pose"], "floor": track["floor"]}
                frame = int(table["now"][row])
                here.append(tracks[name]["pose"][frame : frame + FUTURE + 1])
                floor.append(tracks[name]["floor"][frame])
            here, floor = torch.from_numpy(np.stack(here).astype(np.float32)), np.array(floor, dtype=np.float64)  # (M, 76, 9)
            yaw = frames.heading(here[:, 0, 3:]).double().numpy()
            c, s = np.cos(yaw)[:, None, None], np.sin(yaw)[:, None, None]

            def check(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                """Points (M, D, P, 3) as (x, height, z) in each moment's frame anchored at now -> not blocked, walked (M, D)."""
                x = c * points[..., 0] - s * points[..., 2] + here[:, 0, 0].double().numpy()[:, None, None]
                z = s * points[..., 0] + c * points[..., 2] + here[:, 0, 2].double().numpy()[:, None, None]
                world = to_world(np.stack([x, points[..., 1], z], axis=-1), floor[:, None, None])
                flags = ground.flags(world.reshape(-1, 3))
                shape = points.shape[:-1]
                return ~flags["blocked"].reshape(shape).any(-1), flags["free"].reshape(shape).all(-1)

            def clear(points: np.ndarray) -> np.ndarray:
                """Body points (M, D, P, 3) in each moment's frame anchored at now -> (M, D): none of them in a wall or in something solid."""
                x = c * points[..., 0] - s * points[..., 2] + here[:, 0, 0].double().numpy()[:, None, None]
                z = s * points[..., 0] + c * points[..., 2] + here[:, 0, 2].double().numpy()[:, None, None]
                world = to_world(np.stack([x, points[..., 1], z], axis=-1), floor[:, None, None]).reshape(-1, 3)
                d = ground.data
                cell = np.floor((world[:, :2] - d["origin"]) / 0.05).astype(int)
                rows_, cols_ = d["wall"].shape[1:]
                ok = (cell[:, 0] >= 0) & (cell[:, 0] < cols_) & (cell[:, 1] >= 0) & (cell[:, 1] < rows_)
                wall = ok & d["wall"].any(0)[np.clip(cell[:, 1], 0, rows_ - 1), np.clip(cell[:, 0], 0, cols_ - 1)]
                return ~(wall | things.inside(world)).reshape(points.shape[:-1]).any(-1)

            # what the recorded 5 s of these frames do under the same rules (camera only)
            own_path = frames.anchor(here, torch.zeros(len(own), dtype=torch.long))[:, steps, :3].double().numpy()
            ok, walked = check(own_path[:, None])
            recorded["not_blocked"] += ok[:, 0].tolist()
            recorded["walked"] += walked[:, 0].tolist()
            recorded["travel"] += np.linalg.norm(np.diff(own_path[..., [0, 2]], axis=1, prepend=0 * own_path[:, :1, [0, 2]]), axis=-1).sum(1).tolist()

            picked = rng.integers(0, len(spans), size=(len(own), args.draws))
            a, b, start = (spans[picked][..., k] for k in range(3))
            now = a + np.minimum(start + rng.integers(0, 15, size=picked.shape), b - a - FUTURE - 1)
            at = torch.from_numpy(now.reshape(-1)[:, None] + np.arange(FUTURE + 1)[None])  # (M * D, 76): now and the 75 frames after it
            zero = torch.zeros(at.shape[0], dtype=torch.long)
            head = frames.anchor(pose_all[at], zero)
            root = torch.cat([pelvis_all[at][..., [0, 2]], torch.sin(yaw_all[at])[..., None], torch.cos(yaw_all[at])[..., None]], dim=-1)
            pelvis = frames.anchor_root(root, pose_all[at], zero)
            camera = head[:, steps, :3].double().numpy()
            low = np.stack([pelvis[:, steps, 0].double().numpy(), camera[..., 1], pelvis[:, steps, 1].double().numpy()], axis=-1)  # the pelvis, at the camera's height for the storey
            points = np.concatenate([camera, low], axis=1).reshape(len(own), args.draws, -1, 3)
            ok, walked = check(points)
            level = np.abs(pose_all[at[:, 0], 1].numpy().reshape(len(own), args.draws) - here[:, 0, 1].numpy()[:, None]) <= HEIGHT
            travel = np.linalg.norm(np.diff(camera[..., [0, 2]], axis=1, prepend=0 * camera[:, :1, [0, 2]]), axis=-1).sum(1).reshape(len(own), args.draws)
            if whole:
                # the joints of the draws, from the motion's frame into the frame anchored at its camera at now
                body = np.asarray(joints_all[at[:, steps].numpy()], dtype=np.float64)  # (M * D, 25, 22, 3)
                start = pose_all[at[:, 0]]
                turn = frames.heading(start[:, 3:]).double().numpy()[:, None, None]
                x, z = body[..., 0] - start[:, 0].double().numpy()[:, None, None], body[..., 2] - start[:, 2].double().numpy()[:, None, None]
                body = limbs(np.stack([np.cos(turn) * x + np.sin(turn) * z, body[..., 1], -np.sin(turn) * x + np.cos(turn) * z], axis=-1))
                free_body = clear(body.reshape(len(own), args.draws, -1, 3))
                if len(recorded_body) < args.reference:  # the recorded bodies of these frames, under the same rule
                    some, bodies = [], []
                    for m, row in enumerate(own[:: max(1, len(own) // 40)]):
                        name, frame = str(table["sequence"][row]), int(table["now"][row])
                        if "body" not in tracks[name]:
                            tracks[name]["body"] = dict(np.load(args.body / f"{name}.npz")) if (args.body / f"{name}.npz").exists() else None
                        b = tracks[name]["body"]
                        if b is not None and b["valid"][frame : frame + FUTURE + 1].all() and len(b["latent"][frame : frame + FUTURE + 1]) == FUTURE + 1:
                            span = slice(frame, frame + FUTURE + 1)
                            bodies.append(np.concatenate([b["latent"][span], b["pelvis"][span][:, [0, 2]], np.sin(b["yaw"][span])[:, None], np.cos(b["yaw"][span])[:, None]], axis=1))
                            some.append(own.index(row))
                    if some:
                        made = decoder.joints(torch.from_numpy(np.stack(bodies).astype(np.float32)).to(decoder.codec.parameters().__next__().device))[:, steps].double().cpu().numpy()
                        origin, angle = here[some, 0], yaw[some][:, None, None]
                        x, z = made[..., 0] - origin[:, 0].double().numpy()[:, None, None], made[..., 2] - origin[:, 2].double().numpy()[:, None, None]
                        made = limbs(np.stack([np.cos(angle) * x + np.sin(angle) * z, made[..., 1], -np.sin(angle) * x + np.cos(angle) * z], axis=-1))
                        padded = np.zeros((len(own), 1, made.shape[1] * made.shape[2], 3))
                        padded[:, 0, :, 1] = 99.0  # far above everything
                        padded[some, 0] = made.reshape(len(some), -1, 3)
                        recorded_body += clear(padded)[some, 0].tolist()
            else:
                free_body = np.ones_like(ok)
            passes = ok & walked & level & free_body
            fits += int(passes.sum())
            for key, value in (("travel", travel), ("not_blocked", ok), ("walked", ok & walked), ("body", ok & walked & free_body), ("height", level), ("all", passes)):
                tally[key].append(value.reshape(-1))
            for m, row in enumerate(own):
                for d in np.flatnonzero(passes[m])[: args.keep]:
                    kept["row"].append(row)
                    kept["sample"].append(int(picked[m, d]))
                    kept["now"].append(int(now[m, d]))
                    kept["travel"].append(float(travel[m, d]))
        print(f"  {home}: {len(everything):,d} moments, {fits / max(1, len(everything) * args.draws):.1%} of the draws fit", flush=True)

    tally = {k: np.concatenate(v) for k, v in tally.items()}
    recorded = {k: np.array(v) for k, v in recorded.items()}
    edges = [0, 1, 2, 3, 5, 99]
    if recorded_body:
        print(f"\nthe recorded bodies of {len(recorded_body):,d} of these frames: {np.mean(recorded_body):.1%} pass the body rule (they all should: the rest is the rule's error)")
    print(f"\n{len(tally['all']):,d} draws\n  {'the motion goes':20s} draws  not blocked  and walked  and body  height within 10 cm  all | recorded 5 s of these frames: not blocked  walked")
    report = {}
    for lo, hi in [(0, 99)] + list(zip(edges[:-1], edges[1:])):
        m, r = (tally["travel"] >= lo) & (tally["travel"] < hi), (recorded["travel"] >= lo) & (recorded["travel"] < hi)
        name = "any distance" if hi - lo > 50 else (f"{lo} to {hi} m" if hi < 99 else f"{lo} m or more")
        row = {"draws": int(m.sum()), **{k: float(tally[k][m].mean()) for k in ("not_blocked", "walked", "body", "height", "all")},
               "recorded": int(r.sum()), "recorded_not_blocked": float(recorded["not_blocked"][r].mean()) if r.any() else None, "recorded_walked": float(recorded["walked"][r].mean()) if r.any() else None}
        report[name] = row
        tail = f"{row['recorded_not_blocked']:11.1%} {row['recorded_walked']:7.1%}  ({row['recorded']:,d})" if r.any() else ""
        print(f"  {name:20s} {row['draws']:7,d} {row['not_blocked']:10.1%} {row['walked']:11.1%} {row['body']:9.1%} {row['height']:16.1%} {row['all']:8.1%}      | {tail}")
    kept = {k: np.array(v) for k, v in kept.items()}
    frames_with = len(np.unique(kept["row"]))
    print(f"\n{len(kept['row']):,d} pairs kept (at most {args.keep} per moment): {frames_with:,d} of {len(rows):,d} moments have one, {len(np.unique(kept['sample'])):,d} of {len(motions):,d} HumanML3D samples are used; "
          f"they go {kept['travel'].mean():.2f} m on average (all draws {tally['travel'].mean():.2f} m)")
    np.savez(args.hml / "placed.npz", **kept)
    (args.hml / "placed.json").write_text(json.dumps({"draws": args.draws, "keep": args.keep, "pairs": len(kept["row"]), "moments_with_a_pair": frames_with, "body_rule": whole,
                                                      "recorded_bodies_pass": float(np.mean(recorded_body)) if recorded_body else None, "by_distance": report}, indent=1))
    print(f"wrote {args.hml / 'placed.npz'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    one = steps.add_parser("pairs")
    one.add_argument("--hml", type=Path, required=True)
    one.add_argument("--listing", type=Path, default=Path("/ptmp/enazaren/gray/moments/moments.npz"))
    one.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    one.add_argument("--ground", type=Path, default=Path("/ptmp/enazaren/gray/ground"))
    one.add_argument("--cache", type=Path, default=None)
    one.add_argument("--draws", type=int, default=16)
    one.add_argument("--keep", type=int, default=4)
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--body", type=Path, default=Path("/ptmp/enazaren/gray/body"))
    one.add_argument("--reference", type=int, default=3000, help="about this many recorded bodies are put through the body rule")
    two = steps.add_parser("solid")
    two.add_argument("--ground", type=Path, default=Path("/ptmp/enazaren/gray/ground"))
    two.add_argument("--workers", type=int, default=8)
    three = steps.add_parser("joints")
    three.add_argument("--hml", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "solid":
        return solid(args)
    if args.step == "joints":
        return joints(args)
    if args.cache is None:
        from gray.moments import CACHE

        args.cache = CACHE
    pairs(args)


if __name__ == "__main__":
    main()
