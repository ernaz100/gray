#!/usr/bin/env python3
"""Does reading a room from the image get better with more homes? A test with homes from another dataset.

    python -m gray.ropedia fetch       # on the login node: download the walk-throughs (about 100 GB)
    python -m gray.ropedia set         # views like ours and the free distance from each session's mesh
    python -m gray.probe features --set /ptmp/enazaren/gray/ropeset --name dino-s-7
    python -m gray.ropedia fit         # probes trained on Nymeria, on Ropedia, on both; scored on held-out Nymeria homes

The head-path model reads the free space around the camera to 0.28 m in homes it trained on and
to about 0.55 m in others (DESIGN.md). The likely reason is that Nymeria has only 39 training
homes. Ropedia Xperience 2 (arXiv 2611.17060, Hugging Face ropedia-ai/Ropedia-Xperience-2,
CC BY-NC 4.0) has walk-throughs of other homes with a head-worn camera rig: undistorted keyframes
of four cameras, their poses, and a mesh of what was seen, all in one metric frame with +z up.
Its first batch has 105 indoor walk-throughs in 95 scenes (gray/ropedia_indoor.txt). No text and no
body are needed for this question, only image, pose and geometry.

**set** writes one file per walk-through, in the layout of the probe's frame set (gray/probe.py),
so that its feature step runs on it unchanged:

    image     (F, 448, 448, 3) u8   a view of 79.6 degrees, as gray/pinhole.py cuts from Nymeria, rendered from the two
                                    front cameras (cam0, cam1; 120 x 97 degrees, looking about 28 degrees below level and 25
                                    degrees to either side of straight ahead). The view is turned up by 6 degrees against
                                    the camera, as far as its image allows, to come nearer to how Nymeria's camera looks.
    free      (F, 16) f32           the free distance along 16 directions, as in the probe's set; nan where it is not known
    walkable, blocked  (F, 16) f32  the two distances it is made of
    tilt      (F, 4) f32            the camera's tilt in the convention of the tracks, and its height above the floor
    frame, camera  (F,) i32         the keyframe (position in the sorted timestamps) and the camera

The truth follows gray/ground.py as far as a single walk-through allows. On a 5 cm grid, from the
mesh vertices: **blocked** is a cell with two or more vertices between 0.3 and 1.5 m above the
floor; **free** is a cell with floor under it and not blocked, or within 0.2 m of where the camera
went (the wearer walked there). Walkable is the distance until free ends, blocked the distance
to the first blocked cell, and the free distance is their mean where they agree within 0.3 m.
The floor height is the lowest height below the camera that holds much of the mesh.

Only 41 of the 105 walk-throughs can be used: the other 64 are an older export whose mesh.ply
holds camera markers and sparse points, not a surface. The scenes' LiDAR scans would cover them,
but the registration of the walk-throughs to the scans is not published yet.

Differences to Nymeria that remain: the camera sits higher (about 1.75 m against 1.2 to 1.6 m)
and looks further down; the truth comes from one walk-through, not from many recordings; the
device and the wearer's body show at the lower edge of the image.

**fit** trains the probe of gray/probe.py on the 7 x 7 image grid of Nymeria's training homes, of
Ropedia, and of both, and scores each on the held-out Nymeria homes and on held-out Ropedia
scenes (every 8th scene). Three inputs: the grid alone, with the camera's tilt, with tilt and height.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = "ropedia-ai/Ropedia-Xperience-2"
DATA = Path("/ptmp/enazaren/ropedia")
SET = Path("/ptmp/enazaren/gray/ropeset")
LIST = Path(__file__).with_name("ropedia_indoor.txt")
VIEW = 79.6  # degrees, as gray/pinhole.py
SIDE, RAYS, FAR, STEP, CELL = 448, 16, 5.0, 0.05, 0.05  # as gray/probe.py and gray/layout.py
BAND = (0.3, 1.5)  # as gray/layout.py
IN_VIEW = [15, 0, 1]
UP = 6.0  # degrees the view is turned up against the camera
FRONT = (0, 1)
VERSION = 2  # of the rules for the truth; 2: the floor is the lowest well-filled level, not the fullest
# The keyframes are undistorted. All 41 walk-throughs whose file states the pinhole calibration have this focal length, a width of
# 1280 and the centre in the middle of the image, whatever the sensor's own calibration; it is assumed for the other 64.
FOCAL = 369.504172281


def walkthroughs() -> list[str]:
    return [line.split("\t")[0] for line in LIST.read_text().splitlines() if line and not line.startswith("#")]


def fetch(args) -> None:
    from huggingface_hub import snapshot_download

    wanted = [f"{path}{name}" for path in walkthroughs() for name in ("rgb.tar", "poses/*", "calibration.json", "mesh.ply", "metadata.json")]
    snapshot_download(REPO, repo_type="dataset", allow_patterns=wanted, local_dir=args.data, max_workers=args.workers)
    print(f"fetched {len(walkthroughs())} walk-throughs into {args.data}", flush=True)


def vertices(path: Path) -> np.ndarray:
    """The vertex positions (V, 3) of a binary PLY mesh."""
    with open(path, "rb") as file:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += file.readline()
        lines = head.decode().splitlines()
        if "format ascii" in lines[1]:
            # 64 of the 105 walk-throughs (the ones with the older calibration file) have camera markers and sparse points here, not a surface
            raise ValueError("mesh.ply is not a surface mesh (older export: camera markers and sparse points)")
        count = int(next(line for line in lines if line.startswith("element vertex")).split()[-1])
        after = lines[next(i for i, line in enumerate(lines) if line.startswith("element vertex")) + 1 :]
        kinds = {"double": "<f8", "float": "<f4", "float32": "<f4", "float64": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4"}
        fields = []
        for line in after:
            if not line.startswith("property"):
                break
            fields.append((line.split()[2], kinds[line.split()[1]]))
        data = np.frombuffer(file.read(count * np.dtype(fields).itemsize), dtype=np.dtype(fields))
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def rotation(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) of quaternions (..., 4) in x y z w order."""
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def space(mesh: np.ndarray, path: np.ndarray) -> dict:
    """Free and blocked cells of one walk-through from its mesh vertices (V, 3) and the camera positions (P, 3)."""
    below = mesh[:, 2][mesh[:, 2] < np.median(path[:, 2]) - 0.8]
    if len(below) < 1000:
        raise ValueError(f"no floor in the mesh under the camera: mesh heights {mesh[:, 2].min():.2f} to {mesh[:, 2].max():.2f}, camera at {np.median(path[:, 2]):.2f}, {len(mesh):,d} vertices")
    edges = np.arange(below.min(), below.max() + 0.02, 0.02)
    counts = np.convolve(np.histogram(below, bins=edges)[0], np.ones(3), mode="same")
    floor = float(edges[np.flatnonzero(counts >= 0.3 * counts.max())[0]] + 0.01)  # the lowest level that holds much of the mesh: a bed or a table can hold more than the floor
    origin = mesh[:, :2].min(0) - 0.5
    shape = tuple(np.ceil((mesh[:, :2].max(0) + 0.5 - origin) / CELL).astype(int))
    cell = np.floor((mesh[:, :2] - origin) / CELL).astype(int)
    height = mesh[:, 2] - floor

    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(cell[mask, 0] * shape[1] + cell[mask, 1], minlength=shape[0] * shape[1]).reshape(shape)

    blocked = count((height > BAND[0]) & (height < BAND[1])) >= 2
    ground = count(np.abs(height) < 0.15) > 0
    walked = np.zeros(shape, dtype=bool)
    reach = int(round(0.2 / CELL))
    offsets = [(i, j) for i in range(-reach, reach + 1) for j in range(-reach, reach + 1) if i * i + j * j <= reach * reach]
    here = np.floor((path[:, :2] - origin) / CELL).astype(int)
    for i, j in offsets:
        walked[np.clip(here[:, 0] + i, 0, shape[0] - 1), np.clip(here[:, 1] + j, 0, shape[1] - 1)] = True
    return {"origin": origin, "floor": floor, "free": (ground & ~blocked) | walked, "blocked": blocked & ~walked}


def distances(room: dict, position: np.ndarray, ahead: np.ndarray) -> dict:
    """Walkable and blocked distance (F, 16) from positions (F, 3) looking along the world angle ``ahead`` (F,); direction k is 22.5 k degrees to the left."""
    angle = ahead[:, None] + np.arange(RAYS)[None] * 2 * math.pi / RAYS
    reach = np.arange(STEP, FAR + STEP / 2, STEP)
    points = position[:, None, None, :2] + reach[None, None, :, None] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)[:, :, None, :]
    cell = np.floor((points - room["origin"]) / CELL).astype(int)
    rows, cols = room["free"].shape
    inside = (cell[..., 0] >= 0) & (cell[..., 0] < rows) & (cell[..., 1] >= 0) & (cell[..., 1] < cols)
    x, y = np.clip(cell[..., 0], 0, rows - 1), np.clip(cell[..., 1], 0, cols - 1)
    out = {}
    for key, stop in (("walkable", ~(inside & room["free"][x, y])), ("blocked", inside & room["blocked"][x, y])):
        out[key] = np.where(stop.any(-1), reach[stop.argmax(-1)], FAR).astype(np.float32)
    return out


def turned() -> np.ndarray:
    """The view's axes in the camera's frame (x right, y down, z ahead): turned up by UP degrees."""
    a = math.radians(UP)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def lookup(k: np.ndarray, wide: int, high: int) -> tuple[torch.Tensor, float]:
    """Where each pixel of the view (2 SIDE square) lies in the camera image, for grid_sample, and the share of the view the image covers."""
    side = 2 * SIDE
    focal = (side / 2) / math.tan(math.radians(VIEW) / 2)
    u = (torch.arange(side, dtype=torch.float64) + 0.5 - side / 2) / focal
    ray = torch.stack([u[None].expand(side, side), u[:, None].expand(side, side), torch.ones(side, side, dtype=torch.float64)], -1) @ torch.from_numpy(turned()).T
    x = k[0, 0] * ray[..., 0] / ray[..., 2] + k[0, 2]
    y = k[1, 1] * ray[..., 1] / ray[..., 2] + k[1, 2]
    covered = float(((x >= 0) & (x <= wide) & (y >= 0) & (y <= high)).double().mean())
    return torch.stack([x / wide * 2 - 1, y / high * 2 - 1], -1)[None].float(), covered


def next_member(archive: tarfile.TarFile) -> tarfile.TarInfo | None:
    """The next file of an archive; None at its end, also where an archive was cut short."""
    try:
        return archive.next()
    except (tarfile.ReadError, EOFError, OSError):
        return None


def collect(job: tuple) -> tuple | None:
    """The views and the truth of one walk-through; a walk-through that cannot be used gives its name and the reason."""
    try:
        return collect_one(job)
    except Exception as error:  # noqa: BLE001  one odd walk-through must not end the run
        return Path(job[0]).name, f"{type(error).__name__}: {error}"


def collect_one(job: tuple) -> tuple | None:
    from PIL import Image

    path, data, out, views = job
    torch.set_num_threads(1)
    folder = Path(data) / path
    target = Path(out) / f"{Path(path).name}.npz"
    try:  # files written under an earlier rule for the truth, or cut short, are made again
        if target.exists() and int(np.load(target)["version"]) == VERSION:
            return None
    except Exception:  # noqa: BLE001
        pass
    calibration = json.loads((folder / "calibration.json").read_text()).get("cameras")  # 64 of the 105 files give only the sensor's calibration
    pose = {}
    for camera in FRONT:
        for line in (folder / "poses" / f"cam{camera}.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                part = line.split()
                pose.setdefault(part[0].split("/")[1].split(".")[0], {})[camera] = np.array(part[1:8], dtype=np.float64)
    stamps = sorted(s for s, v in pose.items() if len(v) == len(FRONT))
    if len(stamps) < 20:
        raise ValueError(f"only {len(stamps)} keyframes with both front cameras")
    picked = sorted({stamps[i] for i in np.linspace(0, len(stamps) - 1, min(views // len(FRONT), len(stamps))).round().astype(int)})
    order = {s: i for i, s in enumerate(stamps)}
    position = np.array([pose[s][c][:3] for s in picked for c in FRONT])
    axes = rotation(np.array([pose[s][c][3:] for s in picked for c in FRONT])) @ turned()  # view to world
    room = space(vertices(folder / "mesh.ply"), np.array([pose[s][FRONT[0]][:3] for s in stamps]))
    ahead = np.arctan2(axes[:, 1, 2], axes[:, 0, 2])
    both = distances(room, position, ahead)
    sure = np.abs(both["walkable"] - both["blocked"]) <= 0.3
    where, images = {}, {}
    wanted = {f"{s}.cam{c}.jpg" for s in picked for c in FRONT}
    with tarfile.open(folder / "rgb.tar", "r|") as archive:
        for member in iter(lambda: next_member(archive), None):
            if member.name not in wanted:
                continue
            camera = int(member.name.split(".cam")[1][0])
            try:
                full = torch.from_numpy(np.asarray(Image.open(io.BytesIO(archive.extractfile(member).read())).convert("RGB")).copy()).permute(2, 0, 1)[None].float()
            except (tarfile.ReadError, EOFError, OSError):
                break
            if camera not in where:
                k = np.array(calibration[f"cam{camera}"]["K"], dtype=np.float64) if calibration else np.array(
                    [[FOCAL, 0, full.shape[-1] / 2], [0, FOCAL, full.shape[-2] / 2], [0, 0, 1]])
                where[camera] = lookup(k, full.shape[-1], full.shape[-2])
            view = torch.nn.functional.grid_sample(full, where[camera][0], mode="bilinear", padding_mode="zeros", align_corners=False)
            images[member.name] = torch.nn.functional.avg_pool2d(view, 2)[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()
            if len(images) == len(wanted):
                break
    names = [f"{s}.cam{c}.jpg" for s in picked for c in FRONT]
    have = np.array([n in images for n in names])
    if have.sum() < 20:
        raise ValueError(f"only {int(have.sum())} of the wanted images in the archive")
    # the tracks' tilt is the height (up) part of Nymeria's camera axes: its x points down the image, its y to the image's left
    tilt = np.stack([axes[:, 2, 1], -axes[:, 2, 0], axes[:, 2, 2], position[:, 2] - room["floor"]], axis=1)
    np.savez(target.with_suffix(".partial.npz"), image=np.stack([images[n] for n, h in zip(names, have) if h]), free=np.where(sure, (both["walkable"] + both["blocked"]) / 2, np.nan).astype(np.float32)[have],
             walkable=both["walkable"][have], blocked=both["blocked"][have], tilt=tilt.astype(np.float32)[have],
             frame=np.array([order[s] for s in picked for _ in FRONT], dtype=np.int32)[have], camera=np.array([c for _ in picked for c in FRONT], dtype=np.int32)[have],
             scene=path.split("/")[1], version=VERSION, floor=room["floor"], covered=min(v[1] for v in where.values()))
    target.with_suffix(".partial.npz").rename(target)
    return Path(path).name, int(have.sum()), float(sure[have][:, IN_VIEW].mean()), float(np.median(tilt[:, 3])), min(v[1] for v in where.values())


def build(args) -> None:
    args.set.mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(args.data), str(args.set), args.views) for p in walkthroughs() if (args.data / p / "rgb.tar").exists()][: args.limit]
    print(f"{len(jobs)} of {len(walkthroughs())} walk-throughs are on disk", flush=True)
    total = 0
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(collect, jobs, chunksize=1)):
            if result and len(result) == 2:
                print(f"{i + 1}/{len(jobs)} {result[0]}: left out, {result[1]}", flush=True)
            elif result:
                total += result[1]
                print(f"{i + 1}/{len(jobs)} {result[0]}: {result[1]} views, free distance known for {result[2]:.0%} of the in-view directions, "
                      f"camera {result[3]:.2f} m above the floor, image covers {result[4]:.1%} of the view", flush=True)
    print(f"{total:,d} new views in {args.set}", flush=True)


def look(args) -> None:
    """A few views with their distances, to check by eye: <set>/look.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(p for p in args.set.glob("*.npz") if ".partial" not in p.name)
    rng = np.random.default_rng(0)
    figure, axes = plt.subplots(2, 6, figsize=(21, 7.4))
    for column, index in enumerate(rng.choice(len(files), 6, replace=len(files) < 6)):
        d = np.load(files[index])
        i = int(rng.integers(len(d["frame"])))
        axes[0, column].imshow(d["image"][i])
        axes[0, column].axis("off")
        axes[0, column].set_title(f"{files[index].stem[:8]}, camera {int(d['camera'][i])}, {float(d['tilt'][i, 3]):.2f} m up", fontsize=8)
        axis = axes[1, column]
        angle = math.pi / 2 + np.arange(RAYS) * 2 * math.pi / RAYS  # ahead is up, direction k to the left
        for key, colour in (("blocked", "tab:red"), ("walkable", "tab:green")):
            r = d[key][i]
            axis.plot(np.append(r * np.cos(angle), r[0] * np.cos(angle[0])), np.append(r * np.sin(angle), r[0] * np.sin(angle[0])), color=colour, label=key)
        known = np.isfinite(d["free"][i])
        axis.scatter((d["free"][i] * np.cos(angle))[known], (d["free"][i] * np.sin(angle))[known], color="black", s=14, zorder=3, label="known")
        half = math.radians(VIEW / 2)
        for side in (-half, half):
            axis.plot([0, FAR * math.cos(math.pi / 2 + side)], [0, FAR * math.sin(math.pi / 2 + side)], color="0.7", linewidth=0.8)
        axis.set(xlim=(-FAR, FAR), ylim=(-FAR, FAR))
        axis.set_aspect("equal")
        if column == 0:
            axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.set / "look.png", dpi=70)
    print(f"wrote {args.set / 'look.png'}", flush=True)


def fit(args) -> None:
    from gray.features import CACHE
    from gray.moments import held_locations
    from gray.probe import SET as NYMERIA, Probe, area, read

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    name = args.name
    location = {m["sequence_id"]: m["location"] for m in json.loads((CACHE / "index.json").read_text())["sequences"]}
    held_out = held_locations(CACHE)
    ours = [p for p in sorted(NYMERIA.glob("*.npz")) if not p.name.endswith(".truth.npz") and p.with_name(f"{p.stem}.{name}.npy").exists()]
    theirs = [p for p in sorted(args.set.glob("*.npz")) if p.with_name(f"{p.stem}.{name}.npy").exists()]
    scenes = sorted({str(np.load(p)["scene"]) for p in theirs})
    kept_back = set(scenes[::8])
    files = {"Nymeria": [p for p in ours if location[p.stem] not in held_out], "Nymeria held": [p for p in ours if location[p.stem] in held_out],
             "Ropedia": [p for p in theirs if str(np.load(p)["scene"]) not in kept_back], "Ropedia held": [p for p in theirs if str(np.load(p)["scene"]) in kept_back]}
    x = {k: torch.from_numpy(np.concatenate([np.load(p.with_name(f"{p.stem}.{name}.npy")) for p in paths])) for k, paths in files.items()}
    y = {k: read(paths, "free") for k, paths in files.items()}
    tilt = {k: read(paths, "tilt") for k, paths in files.items()}
    homes = {"Nymeria": len({location[p.stem] for p in files["Nymeria"]}), "Ropedia": len(scenes) - len(kept_back)}
    for k in files:
        known = torch.isfinite(y[k][:, IN_VIEW])
        span = y[k][:, IN_VIEW][known]
        print(f"{k:13s} {len(y[k]):6,d} views from {len(files[k]):3d} recordings; free distance known for {float(known.float().mean()):.0%} of the in-view directions, "
              f"mean {span.mean():.2f} m, under 1 m in {float((span < 1).float().mean()):.0%}; camera {tilt[k][:, 3].median():.2f} m up, looks {math.degrees(math.asin(float(tilt[k][:, 2].median()))):.0f} degrees above level", flush=True)

    def scores(predicted: torch.Tensor, k: str) -> str:
        known = torch.isfinite(y[k][:, IN_VIEW])
        span = y[k][:, IN_VIEW][known]
        error = (predicted - y[k]).abs()[:, IN_VIEW][known]
        ahead = torch.isfinite(y[k][:, 0]).numpy()
        return f"{error.mean():5.2f} {error[span < 1].mean():5.2f} {error[(span >= 1) & (span < 2.5)].mean():5.2f} {error[span >= 2.5].mean():5.2f}  {area(-predicted[:, 0].numpy()[ahead], (y[k][:, 0] < 1.0).numpy()[ahead]):4.2f}"

    rng = torch.Generator().manual_seed(0)
    smaller = min(len(y["Nymeria"]), len(y["Ropedia"]))
    part = {k: torch.randperm(len(y[k]), generator=rng)[:smaller] for k in ("Nymeria", "Ropedia")}
    half = {k: v[: smaller // 2] for k, v in part.items()}
    training = {"Nymeria": [("Nymeria", None)], "Ropedia": [("Ropedia", None)], "both": [("Nymeria", None), ("Ropedia", None)],
                f"Nymeria, {smaller:,d} views": [("Nymeria", part["Nymeria"])], f"Ropedia, {smaller:,d} views": [("Ropedia", part["Ropedia"])],
                f"both, {smaller // 2:,d} views of each": [("Nymeria", half["Nymeria"]), ("Ropedia", half["Ropedia"])]}
    print(f"\ntraining homes: Nymeria {homes['Nymeria']}, Ropedia {homes['Ropedia']} scenes. Columns per test set: error in view (m), under 1 m, 1 to 2.5 m, 2.5 m and more, near ahead (AUC)")
    print(f"  {'input':22s} {'trained on':34s} | {'held-out Nymeria homes':30s} | held-out Ropedia scenes", flush=True)
    rows = {}
    for label, sources in [("constant", None)] + [(f"{t}|{k}", v) for t in ("grid", "grid, tilt", "grid, tilt, height") for k, v in training.items()]:
        if sources is None:
            for k in ("Nymeria", "Ropedia"):
                guess = y[k].nanmedian(0).values
                rows[f"constant|{k}"] = {test: scores(guess[None].expand_as(y[test]).clone(), test) for test in ("Nymeria held", "Ropedia held")}
                print(f"  {'the median distance':22s} {k:34s} | {rows[f'constant|{k}']['Nymeria held']:30s} | {rows[f'constant|{k}']['Ropedia held']}", flush=True)
            continue
        numbers = {"grid": 0, "grid, tilt": 3, "grid, tilt, height": 4}[label.split("|")[0]]
        pick = lambda value: torch.cat([value[k] if index is None else value[k][index] for k, index in sources])  # noqa: E731
        features, target, lean = pick(x), pick(y).to(device), pick(tilt).to(device)
        torch.manual_seed(0)
        model = Probe(features.shape[1], features.shape[2], numbers, True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        steps = args.epochs * math.ceil(len(y["Nymeria"]) / 256)  # the same number of steps whatever the amount of data
        done = 0
        while done < steps:
            for index in torch.randperm(len(target)).split(256):
                known = torch.isfinite(target[index])
                loss = nn.functional.smooth_l1_loss(model(features[index].to(device), lean[index])[known], target[index][known])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                done += 1
                if done >= steps:
                    break
        model.eval()
        with torch.no_grad():
            predicted = {test: torch.cat([model(x[test][i : i + 1024].to(device), tilt[test][i : i + 1024].to(device)) for i in range(0, len(y[test]), 1024)]).clamp(0, FAR).cpu()
                         for test in ("Nymeria held", "Ropedia held")}
        rows[label] = {test: scores(predicted[test], test) for test in predicted}
        print(f"  {label.split('|')[0]:22s} {label.split('|')[1]:34s} | {rows[label]['Nymeria held']:30s} | {rows[label]['Ropedia held']}", flush=True)
    (args.set / "fit.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.set / 'fit.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("fetch", "set", "look", "fit"))
    parser.add_argument("--data", type=Path, default=DATA, help="where the downloaded walk-throughs are")
    parser.add_argument("--set", type=Path, default=SET)
    parser.add_argument("--views", type=int, default=400, help="set: views per walk-through, half from each front camera")
    parser.add_argument("--limit", type=int, default=None, help="set: only the first walk-throughs")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--name", default="dino-s-7", help="fit: the features of gray/probe.py")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    {"fetch": fetch, "set": build, "look": look, "fit": fit}[args.step](args)


if __name__ == "__main__":
    main()
