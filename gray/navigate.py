#!/usr/bin/env python3
"""Score generated head paths on the navigation subset: are they possible, and do they follow the text?

    python -m gray.navigate --run <runs>/head-v1

On held-out windows of the navigation subset that have a room map, the model gets one second of
history (head poses and image features, no body motion) and the text, and draws several head
paths for the 7 s after it. Each path is scored without comparing it to the recorded one:

    walls        the path touches a wall of the room map (gray/layout.py)
    furniture    it enters furniture other than beds, couches, chairs and tables
    off the map  share of its points that are not over annotated floor
    goes         it gets 1 m or further from where it starts; compared with whether the text mentions walking
    destination  where the text names an object that the map has and the person really walked up
                 to it, the path ends within 1 m of that object

Each of these is also computed for paths drawn with the text of another window, for the recorded
path, and for the recorded path turned by 90 degrees. The distance to the recorded path is kept as well.
Writes <run>/navigate.json.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gray import frames
from gray.census import WALK, destinations
from gray.features import split
from gray.joint import HEAD, MOTION, Windows, collate, generate, load, raw_tokens, restrict, seconds_from, to_device
from gray.layout import CELL, Layout, turned

SOFT = ("Bed", "Couch", "Chair", "Table")  # people sit on these or lean over them
GOES = 1.0  # metres
NEAR = 1.0  # metres from the named object at the end of the path
AWAY = 1.5  # metres from it at the start, for the destination test
HISTORY = 15


def to_window(head: torch.Tensor, now: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Positions (..., B, L, 3) in the frame anchored at ``now`` -> the window's own frame (the one head.npy is in)."""
    here = head[torch.arange(head.shape[0], device=head.device), now]
    yaw = frames.heading(here[:, 3:])
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    x, z = positions[..., 0], positions[..., 2]
    return torch.stack([c * x - s * z + here[:, None, 0], positions[..., 1], s * x + c * z + here[:, None, 2]], dim=-1)


def targets(layout: Layout, window: int, text: str) -> np.ndarray | None:
    """World positions (N, 2) of the cells of the objects the text says the person goes to, on the storey the window starts on."""
    d = layout.data
    words = [phrase.split()[-1] for phrase, room in destinations(text) if not room]
    if not words:
        return None
    names = [set(re.findall("[a-z]+", str(n).lower())) | {str(c).lower()} for n, c in zip(d["furniture_name"], d["furniture_category"])]
    numbers = [i + 1 for i, known in enumerate(names) if any(w in known or w.rstrip("s") in known for w in words)]
    if not numbers:
        return None
    rows, cols = np.nonzero(np.isin(d["furniture"][int(d["level"][window])], numbers))
    if len(rows) == 0:
        return None
    return d["origin"] + (np.stack([cols, rows], axis=1) + 0.5) * CELL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=Path("/ptmp/enazaren/gray/runs/census/navigation.npz"))
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--windows", type=int, default=3000)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    model, mean, std, config = load(args.run, device)
    cache = Path(config["cache"])
    images = Path(config["images"]) if config.get("images") else None
    _train, items = split(cache, float(config["holdout"]), float(config["min_clock"]))
    items = [item for item in restrict(items, args.subset) if (args.maps / f"{item[0]}.npz").exists()]
    items = [items[i] for i in np.random.default_rng(0).permutation(len(items))[: args.windows]]
    listed = np.load(args.subset)
    going = {pair: bool(g) for pair, g in zip(zip(listed["sequence"].tolist(), listed["window"].tolist()), listed["going"])}
    print(f"{len(items):,d} held-out windows of the navigation subset with a room map, {args.draws} draws each, {HISTORY} frames of history, "
          f"image features from {images or cache}", flush=True)

    layouts: dict[str, Layout] = {}
    texts: dict[str, tuple] = {}

    def text_of(sequence: str, window: int) -> str:
        if sequence not in texts:
            texts[sequence] = ([s["text"] for s in json.loads((cache / sequence / "sentences.json").read_text())], np.load(cache / sequence / "text_index.npy"))
        sentences, index = texts[sequence]
        return " ".join(sentences[i] for i in index[window] if i >= 0)

    def check(layout: Layout, window: int, points: np.ndarray) -> tuple[float, float, float]:
        """Touches a wall, enters hard furniture, share of points off the annotated floor."""
        storeys = layout.storeys(window, points)
        f = layout.flags(window, points, storeys)
        kind = np.concatenate([["none"], layout.data["furniture_category"]])[f["furniture"]]
        return float(f["wall"].any()), float(((f["furniture"] > 0) & ~np.isin(kind, SOFT)).any()), float((~f["floor"] & ~f["unmapped"]).mean())

    generator = torch.Generator(device=device)
    rows = []
    grids = Path(config["patches"]) if config.get("patches") else None
    loader = DataLoader(Windows(cache, items, images, grids, HISTORY - 1), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    done, frame_error = 0, 0.0
    for batch in loader:
        batch = to_device(batch, device)
        size = batch["head"].shape[0]
        model.see(batch.get("patches"))
        now = torch.full((size,), HISTORY - 1, device=device)
        seconds = seconds_from(now)
        raw = raw_tokens(batch, now)
        x1 = (raw - mean) / std
        known = (seconds <= 0)[..., None].repeat(1, 1, 3)
        known[..., MOTION] = False  # a robot has no human body motion to start from
        given = {"real": (batch["text"], batch["text_valid"]), "other": (batch["text"].roll(1, dims=0), batch["text_valid"].roll(1, dims=0))}
        with torch.no_grad():
            drawn = {}
            for name, text in given.items():
                paths = []
                for draw in range(args.draws):
                    generator.manual_seed(1000 + draw)
                    out = generate(model, x1, known, seconds, *text, order=(HEAD,), steps=args.steps, generator=generator, amp=amp)
                    paths.append((out * std + mean)[..., :3])
                drawn[name] = to_window(batch["head"], now, torch.stack(paths)).double().cpu().numpy()  # (K, B, L, 3)
        frame_error = max(frame_error, float((to_window(batch["head"], now, raw[..., :3]) - batch["head"][..., :3]).abs().max()))
        truth = batch["head"][..., :3].double().cpu().numpy()
        future = slice(HISTORY, None, 3)
        for b in range(size):
            sequence, window = items[done + b]
            other = items[done + (b - 1) % size]
            if sequence not in layouts:
                layouts[sequence] = Layout(args.maps / f"{sequence}.npz")
            layout = layouts[sequence]
            text, text_other = text_of(sequence, window), text_of(*other)
            start = truth[b, HISTORY - 1]
            reach = lambda path: float(np.hypot(path[HISTORY:, 0] - start[0], path[HISTORY:, 2] - start[2]).max())  # noqa: E731
            row = {"going": going[(sequence, window)], "has_text": bool(text), "walk": bool(WALK.search(text)), "has_other": bool(text_other),
                   "walk_other": bool(WALK.search(text_other)), "recorded_reach": reach(truth[b]),
                   "recorded": check(layout, window, truth[b, future]), "turned": check(layout, window, turned(truth[b, future], 90))}
            for name in given:
                row[name] = np.mean([check(layout, window, path[b, future]) for path in drawn[name]], axis=0).tolist()
                row[name + "_reach"] = [reach(path[b]) for path in drawn[name]]
                error = np.linalg.norm(drawn[name][:, b, HISTORY:] - truth[b, HISTORY:], axis=-1).mean(-1) * 100
                row[name + "_cm"] = [float(error.mean()), float(error.min())]
            row["stay_cm"] = float(np.linalg.norm(truth[b, HISTORY:] - start, axis=-1).mean() * 100)
            cells = targets(layout, window, text) if text else None
            if cells is not None:
                gap = lambda point: float(np.linalg.norm(cells - layout.world(window, point[None])[0, :2], axis=1).min())  # noqa: E731
                row["target"] = {"start": gap(start), "recorded": gap(truth[b, -1]), "real": [gap(path[b, -1]) for path in drawn["real"]],
                                 "other": [gap(path[b, -1]) for path in drawn["other"]]}
            rows.append(row)
        done += size
        print(f"  {done:,d} windows", flush=True)

    print(f"\nframe check: recorded paths taken to the window frame and back differ by at most {frame_error * 100:.2f} cm")
    result = {"windows": len(rows), "draws": args.draws}
    groups = (("all windows", lambda r: True), ("windows that go 1 m or further", lambda r: r["going"]), ("windows that stay", lambda r: not r["going"]))
    for title, pick in groups:
        part = [r for r in rows if pick(r)]
        if not part:
            continue
        table = {}
        print(f"\n{title} ({len(part):,d})\n  path                      touches a wall   enters hard furniture   points off the map   gets 1 m or further")
        for label, key in (("recorded", "recorded"), ("recorded, turned 90 deg", "turned"), ("generated", "real"), ("generated, other text", "other")):
            values = np.array([r[key] for r in part])
            if key in ("real", "other"):
                goes = float(np.mean([np.mean(np.array(r[key + "_reach"]) >= GOES) for r in part]))
            else:
                goes = float(np.mean([r["recorded_reach"] >= GOES for r in part])) if key == "recorded" else float("nan")
            table[label] = {"wall": float(values[:, 0].mean()), "hard_furniture": float(values[:, 1].mean()), "off_map": float(values[:, 2].mean()), "goes": goes}
            print(f"  {label:25s} {values[:, 0].mean():14.1%} {values[:, 1].mean():23.1%} {values[:, 2].mean():20.1%} {goes:21.1%}")
        cm = {"one draw": float(np.mean([r["real_cm"][0] for r in part])), "best draw": float(np.mean([r["real_cm"][1] for r in part])),
              "other text": float(np.mean([r["other_cm"][0] for r in part])), "stay": float(np.mean([r["stay_cm"] for r in part]))}
        print("  distance to the recorded path, cm: " + ", ".join(f"{k} {v:.1f}" for k, v in cm.items()))
        result[title] = {"paths": table, "cm": cm}

    def goes_share(part: list, key: str) -> float:
        return float(np.mean([np.mean(np.array(r[key + "_reach"]) >= GOES) for r in part])) if part else float("nan")

    said = [r for r in rows if r["has_text"] and r["walk"]]
    unsaid = [r for r in rows if r["has_text"] and not r["walk"]]
    swap_said = [r for r in rows if r["has_other"] and r["walk_other"]]
    swap_unsaid = [r for r in rows if r["has_other"] and not r["walk_other"]]
    follow = {
        "recorded": [float(np.mean([r["recorded_reach"] >= GOES for r in said])), float(np.mean([r["recorded_reach"] >= GOES for r in unsaid]))],
        "generated": [goes_share(said, "real"), goes_share(unsaid, "real")],
        "generated, other text (by what that text says)": [goes_share(swap_said, "other"), goes_share(swap_unsaid, "other")],
    }
    print(f"\ndoes it go when the text says so? share of paths that get 1 m or further in the {(120 - HISTORY) / 15:.0f} s after now")
    print(f"  {'':48s} text mentions walking ({len(said):,d})   text does not ({len(unsaid):,d})")
    for label, (yes, no) in follow.items():
        print(f"  {label:48s} {yes:20.1%} {no:25.1%}")
    result["goes_by_text"] = follow

    aimed = [r["target"] for r in rows if "target" in r]
    reached = [t for t in aimed if t["recorded"] <= NEAR and t["start"] >= AWAY]
    print(f"\ndestination: {len(aimed):,d} windows name an object that the map has; in {len(reached):,d} the person starts {AWAY:g} m or further from it and ends within {NEAR:g} m")
    if reached:
        destination = {
            "windows": len(reached), "start_m": float(np.median([t["start"] for t in reached])),
            "real_within": float(np.mean([np.mean(np.array(t["real"]) <= NEAR) for t in reached])), "other_within": float(np.mean([np.mean(np.array(t["other"]) <= NEAR) for t in reached])),
            "real_best_within": float(np.mean([min(t["real"]) <= NEAR for t in reached])),
            "real_m": float(np.median([np.median(t["real"]) for t in reached])), "other_m": float(np.median([np.median(t["other"]) for t in reached])),
        }
        print(f"  paths ending within {NEAR:g} m of the object: generated {destination['real_within']:.1%}, at least one of the {args.draws} draws {destination['real_best_within']:.1%}, "
              f"generated with other text {destination['other_within']:.1%}")
        print(f"  median distance to the object at the end: start {destination['start_m']:.2f} m, generated {destination['real_m']:.2f} m, with other text {destination['other_m']:.2f} m")
        result["destination"] = destination
    (args.run / "navigate.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.run / 'navigate.json'}", flush=True)


if __name__ == "__main__":
    main()
