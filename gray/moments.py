#!/usr/bin/env python3
"""Training samples that start where a narration sentence starts ("moments").

    python -m gray.moments --out <folder>          # build the list of moments

Nymeria's narration describes a recording in consecutive pieces of 5 s, one sentence each. A
moment is one such piece: "now" is the frame where the sentence begins, the text is that one
sentence, and the head path to predict is the 5 s it describes. The model is given the 2 s
before now. This differs from the 8 s windows of gray/joint.py, whose text is every sentence
overlapping the window, wherever "now" happens to fall in it.

    past     HISTORY = 30 frames up to and including now (2 s)
    future   FUTURE = 75 frames after now (5 s)

The list (moments.npz) has one row per sentence that has enough frames around it:

    sequence, sentence   the recording and the sentence's number in the cache's sentences.json
    now                  the frame where the sentence begins (index into the track and the image features)
    reach                largest horizontal distance (m) of the head from where it is at now, within the future
    going                reach >= 1 m
    walk                 the sentence mentions walking (gray.census.WALK)
    destination          the places it says the person goes to (gray.census.destinations), joined by "; "
    has_map, floor       a room map exists (gray/layout.py); share of the future path over annotated floor
    wall                 share of the future path inside a wall of the map (a check: should be 0)
    held                 the recording's location is held out (the same locations as in gray.features.split)
    subset               the navigation subset: indoor, every going moment and 12 % of the others

``Moments`` is the dataset on top of the list, the tracks (gray/tracks.py), the image features
(gray/pinhole.py) and the cache's text features.
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gray.census import WALK, destinations
from gray.features import CACHE
from gray.layout import Layout
from gray.tracks import to_world

HISTORY, FUTURE = 30, 75
DEPTH_SCALE = 0.42  # measured / estimated depth, the mean over the held-out recordings (gray/cloud.py)
EARLY, LATE = 5, 15  # in training now is moved by up to 5 frames before and 15 after the start of the sentence
OUT = Path("/ptmp/enazaren/gray/moments")


def held_locations(cache: Path, holdout: float = 0.15) -> set[str]:
    """The locations held out for evaluation: every k-th by name, as in gray.features.split."""
    locations = sorted({m["location"] for m in json.loads((cache / "index.json").read_text())["sequences"]})
    every = max(2, round(1 / holdout))
    return set(locations[every - 1 :: every])


def one(job: tuple[str, str, str, str]) -> dict | None:
    name, cache, tracks, maps = job
    path = Path(tracks) / f"{name}.npz"
    if not path.exists():
        return None
    track = np.load(path)
    pose, floor, valid = track["pose"], track["floor"], track["valid"]
    spoken = [s["text"] for s in json.loads((Path(cache) / name / "sentences.json").read_text())]
    layout = Layout(Path(maps) / f"{name}.npz") if (Path(maps) / f"{name}.npz").exists() else None
    rows = []
    for sentence, (begin, _end) in enumerate(track["sentences"]):
        now = int(begin)
        if now - HISTORY + 1 - EARLY < 0 or now + FUTURE + LATE >= len(pose) or not valid[now - HISTORY + 1 - EARLY : now + FUTURE + LATE + 1].all():
            continue
        future = pose[now + 1 : now + FUTURE + 1, :3]
        row = {"sentence": sentence, "now": now, "reach": float(np.hypot(future[:, 0] - pose[now, 0], future[:, 2] - pose[now, 2]).max()),
               "walk": bool(WALK.search(spoken[sentence])), "destination": "; ".join(sorted({p for p, _room in destinations(spoken[sentence])})),
               "floor": np.nan, "wall": np.nan}
        if layout is not None:
            world = to_world(future[::3].astype(np.float64), floor[now + 1 : now + FUTURE + 1 : 3, None][:, 0])
            flags = layout.flags_world(world, layout.storeys_world(world))
            row["floor"], row["wall"] = float(flags["floor"].mean()), float(flags["wall"].mean())
        rows.append(row)
    meta = json.loads((Path(cache) / name / "meta.json").read_text())
    return {"name": name, "location": meta["location"], "script": meta["script"], "rows": rows}


def build(args) -> None:
    names = sorted(p.parent.name for p in args.cache.glob("*/head.npy"))[: args.limit]
    with Pool(args.workers) as pool:
        results = [r for r in pool.imap(one, [(n, str(args.cache), str(args.tracks), str(args.maps)) for n in names], chunksize=4) if r]
    held = held_locations(args.cache)
    with args.catalog.open() as handle:
        setting = {row["script"]: row["setting"] for row in csv.DictReader(handle)}
    column = lambda key, kind: np.array([row[key] for r in results for row in r["rows"]], dtype=kind)  # noqa: E731
    per = lambda value: np.array([value(r) for r in results for _row in r["rows"]])  # noqa: E731
    reach, floor, walk = column("reach", np.float32), column("floor", np.float32), column("walk", bool)
    where = per(lambda r: setting.get(r["script"], "unknown"))
    going, has_map = reach >= args.min_reach, np.isfinite(floor)
    indoor = (where != "outdoor") & np.where(has_map, floor >= args.min_floor, where == "indoor")
    subset = indoor & (going | (np.random.default_rng(0).random(len(reach)) < args.keep_staying))
    is_held = per(lambda r: r["location"] in held)
    destination = column("destination", str)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "moments.npz", sequence=per(lambda r: r["name"]), sentence=column("sentence", np.int32), now=column("now", np.int32),
                        reach=reach, going=going, walk=walk, destination=destination, has_map=has_map, floor=floor, wall=column("wall", np.float32),
                        held=is_held, subset=subset)

    total = len(reach)
    print(f"{total:,d} moments from {sum(bool(r['rows']) for r in results)} recordings with narration ({len(results)} recordings have a track)")
    print(f"  get {args.min_reach:g} m or further within {FUTURE / 15:g} s: {going.mean():.0%}; the sentence mentions walking: {walk.mean():.0%}; "
          f"it names a destination: {(destination != '').mean():.0%}; a room map exists: {has_map.mean():.0%}")
    print(f"  text against movement: of the moments whose sentence mentions walking {going[walk].mean():.0%} go, of the others {going[~walk].mean():.0%}")
    print(f"  recorded future paths inside a wall of the map: {np.nanmean(column('wall', np.float32)):.3%} of the points (should be 0)")
    print(f"  dropped as not indoor: {(where == 'outdoor').sum():,d} from outdoor activities, {((where != 'outdoor') & has_map & (floor < args.min_floor)).sum():,d} with a map "
          f"but less than {args.min_floor:.0%} of the path over annotated floor, {((where == 'mixed') & ~has_map).sum():,d} from mixed activities without a map")
    print(f"navigation subset: {subset.sum():,d} moments; {(subset & going).sum():,d} go and {(subset & ~going).sum():,d} stay; "
          f"{(subset & ~is_held).sum():,d} for training, {(subset & is_held).sum():,d} held out ({(subset & is_held & going).sum():,d} of them go, "
          f"{(subset & is_held & has_map).sum():,d} have a map, {(subset & is_held & (destination != '')).sum():,d} name a destination)")
    print(f"wrote {args.out / 'moments.npz'}", flush=True)


class Moments(Dataset):
    """Moments of the navigation subset, for training (``held`` False) or evaluation (True).

    An item holds the poses of the 30 frames up to now and the 75 after it in the track's ground
    frame (105, 9), the image feature of the 30 past frames (30, 384), the text features of the
    sentence, and its row in the list. ``jitter`` moves now by -5 to +15 frames at random.
    ``patches``: a folder of gray/patches.py; the item then has the image grid of the frame at now.
    ``everything``: all moments of the split, not only the navigation subset. ``wide``: with jitter, now is moved by up to this
    many frames either way instead of -5 to +15 (37 covers a whole sentence, so any frame of a narrated recording can be now).
    ``future``: the item also has "target" (75, 384), the image feature of the 75 frames after now (for gray/compare.py).
    ``body``: a folder of gray/body.py; the item then has "body" (105, 36): the pose latent (32) and the root path (pelvis x and z,
    sine and cosine of the root yaw) of the same frames in the track's ground frame, and "body_known": whether all of it exists.
    ``revisits``: a folder of gray/revisits.py. Each frame of "target" and "grid_target" is then, with probability
    ``revisit_share`` and where one exists, taken from a revisit of that frame: the same place seen again later in the recording.
    ``future_grid``: the item also has "grid_target" (15, 49, 384), the image grid of every fifth frame after now (needs ``patches``).
    ``depth``: a folder of gray/depth.py; the item then has, for the frame at now, "depth" (49,): the estimated depth in metres
    of each cell of the 7 x 7 image grid, and "rays" (16,): how far the nearest thing at body height is in 16 directions as read
    off that depth with the camera's tilt and height (5 m where the depth shows nothing; directions outside the view are 5 m).
    Both are scaled by 0.42, the mean factor between this depth and Aria's measured points (gray/cloud.py).
    ``ground``: a folder of gray/ground.py; the item then has "free" (32,): how far the way is free from the camera at now
    along 16 directions (0 is where it looks, counted to the left), first as far as people are known to have walked, then up
    to the first blocked cell, in metres up to 5; nan where the location has no such map or the head is on a storey without one.
    """

    def __init__(self, listing: Path, tracks: Path, images: Path, cache: Path = CACHE, held: bool = False, jitter: bool = False,
                 patches: Path | None = None, only: np.ndarray | None = None, ground: Path | None = None, depth: Path | None = None, future: bool = False, everything: bool = False, wide: int | None = None,
                 future_grid: bool = False, revisits: Path | None = None, revisit_share: float = 0.5, body: Path | None = None) -> None:
        self.table = dict(np.load(listing))
        keep = (self.table["subset"] | everything) & (self.table["held"] == held)
        self.wide = wide
        self.rows = np.flatnonzero(keep if only is None else keep & only)
        self.tracks, self.images, self.cache, self.jitter, self.patches, self.ground = tracks, images, cache, jitter, patches, ground
        self.depth, self.future, self.future_grid = depth, future, future_grid
        self.revisits, self.revisit_share, self.body = revisits, revisit_share, body
        self._open: dict[str, dict] = {}
        self._grounds: dict[str, object] = {}
        if ground is not None:
            self.location = {m["sequence_id"]: m["location"] for m in json.loads((cache / "index.json").read_text())["sequences"]}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.item(index)

    def item(self, index: int, shift: int = 0) -> dict[str, torch.Tensor]:
        """The moment; with ``shift`` the same with now moved by that many frames (the caller sees to it that the track reaches that far)."""
        row = int(self.rows[index])
        name, sentence, now = str(self.table["sequence"][row]), int(self.table["sentence"][row]), int(self.table["now"][row])
        if name not in self._open:
            folder = self.cache / name
            track = np.load(self.tracks / f"{name}.npz")
            self._open[name] = {"pose": track["pose"], "floor": track["floor"], "valid": track["valid"], "image": np.load(self.images / name / "dino.npy", mmap_mode="r"),
                                "text": np.load(folder / "text_tokens.npy", mmap_mode="r"), "length": np.load(folder / "text_lengths.npy")}
            if self.revisits is not None:
                self._open[name]["again"] = np.load(self.revisits / f"{name}.npy")
            if self.body is not None:
                self._open[name]["body"] = dict(np.load(self.body / f"{name}.npz"))
            if self.depth is not None:
                self._open[name]["depth"] = np.load(self.depth / name / "depth.npy", mmap_mode="r")
            if self.patches is not None:
                self._open[name]["rows"] = np.load(self.patches / name / "rows.npy")
                self._open[name]["grid"] = np.load(self.patches / name / "patches.npy", mmap_mode="r")
        a = self._open[name]
        if self.jitter and self.wide is None:
            now += int(torch.randint(-EARLY, LATE + 1, (1,)))
        elif self.jitter:
            moved = now + int(torch.randint(-self.wide, self.wide + 1, (1,)))
            if moved - HISTORY + 1 >= 0 and moved + FUTURE < len(a["pose"]) and a["valid"][moved - HISTORY + 1 : moved + FUTURE + 1].all():
                now = moved
        now += shift
        tensor = lambda values: torch.from_numpy(np.array(values, dtype=np.float32))  # noqa: E731
        out = {"pose": tensor(a["pose"][now - HISTORY + 1 : now + FUTURE + 1]), "image": tensor(a["image"][now - HISTORY + 1 : now + 1]),
               "text": tensor(a["text"][sentence, : a["length"][sentence]]), "row": torch.tensor(row), "now": torch.tensor(now)}
        if self.patches is not None:
            at = int(a["rows"][now])
            if at < 0:
                raise KeyError(f"{name}, frame {now}: no image grid in {self.patches}")
            out["patches"] = tensor(a["grid"][at])
            if self.future_grid:
                out["grid_target"] = tensor(np.stack([a["grid"][int(a["rows"][f])] for f in self.maybe_again(a, np.arange(now + 5, now + FUTURE + 1, 5))]))
        if self.future:
            out["target"] = tensor(np.stack([a["image"][int(f)] for f in self.maybe_again(a, np.arange(now + 1, now + FUTURE + 1))]) if self.revisits is not None
                                   else a["image"][now + 1 : now + FUTURE + 1])
        out["foreign"], out["history_max"] = torch.tensor(False), torch.tensor(HISTORY)  # see gray/hml.py: samples without images or a real camera
        if self.body is not None:
            b, span = a["body"], slice(now - HISTORY + 1, now + FUTURE + 1)
            out["body"] = tensor(np.concatenate([b["latent"][span], b["pelvis"][span][:, [0, 2]], np.sin(b["yaw"][span])[:, None], np.cos(b["yaw"][span])[:, None]], axis=1))
            out["body_known"] = torch.tensor(bool(b["valid"][span].all()) and bool(a["valid"][span].all()) and len(b["latent"][span]) == HISTORY + FUTURE)
        if self.depth is not None:
            from gray.probe import from_depth

            metres = tensor(a["depth"][now])[None] * DEPTH_SCALE  # (1, 28, 28)
            tilt = torch.from_numpy(np.concatenate([a["pose"][now, 6:9], a["pose"][now, 1:2]]).astype(np.float32))[None]
            out["depth"] = torch.nn.functional.avg_pool2d(metres[None], 4).flatten()
            out["rays"] = from_depth(metres, tilt)[0]
        if self.ground is not None:
            out["free"] = tensor(self.free(name, a["pose"][now], float(a["floor"][now])))
        return out

    def maybe_again(self, a: dict, wanted: np.ndarray) -> np.ndarray:
        """The frames to read for the wanted ones: some replaced by a revisit of themselves."""
        if self.revisits is None:
            return wanted
        again = a["again"][wanted]
        swap = (again >= 0) & (torch.rand(len(wanted)).numpy() < self.revisit_share)
        return np.where(swap, again, wanted)

    def free(self, name: str, pose: np.ndarray, floor: float) -> np.ndarray:
        """Walkable and blocked distances (32,) from the camera pose of one frame; nan without a map."""
        from gray import frames
        from gray.ground import Ground

        where = self.location.get(name)
        if where not in self._grounds:
            path = self.ground / f"{where}.npz"
            self._grounds[where] = Ground(path) if path.exists() else None
        ground = self._grounds[where]
        if ground is None or name not in set(ground.data["recordings"].tolist()):
            return np.full(32, np.nan, dtype=np.float32)
        yaw = float(frames.heading(torch.from_numpy(pose[None, 3:])))
        both = ground.distances(to_world(pose[None, :3].astype(np.float64), floor), np.array([np.arctan2(-np.cos(yaw), -np.sin(yaw))]))
        return np.concatenate([both["walkable"][0], both["blocked"][0]]).astype(np.float32)


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = {key: torch.stack([item[key] for item in items]) for key in items[0] if key != "text"}
    longest = max(1, max(item["text"].shape[0] for item in items))
    batch["text"] = torch.zeros(len(items), longest, items[0]["text"].shape[-1])
    batch["text_valid"] = torch.zeros(len(items), longest, dtype=torch.bool)
    for i, item in enumerate(items):
        count = item["text"].shape[0]
        batch["text"][i, :count] = item["text"]
        batch["text_valid"][i, :count] = True
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--catalog", type=Path, default=Path("/u/enazaren/projects/STEER/data/nymeria_catalog.csv"), help="says per activity whether it is indoor, outdoor or mixed")
    parser.add_argument("--min-reach", type=float, default=1.0)
    parser.add_argument("--keep-staying", type=float, default=0.12)
    parser.add_argument("--min-floor", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
