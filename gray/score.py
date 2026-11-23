#!/usr/bin/env python3
"""Score a head-path model of gray/head.py on held-out moments: are its paths possible, and do they follow the text?

    python -m gray.score --run <runs>/head-v3

For held-out moments of the navigation subset that have a room map, the model draws several
paths for the 5 s after now, from 2 s of history and the sentence that begins at now. No path is
compared with the recorded one for these measures:

    walls        the path touches a wall of the room map (gray/layout.py)
    blocked      it moves through something: a point in a blocked cell of gray/ground.py (a wall, or a cell where Aria measured
                 points in many recordings) while the head moves faster than 0.5 m/s. People lean over counters, so being
                 over a blocked cell while slow is not a collision. This replaces the furniture boxes, which are too rough
    walked       share of its points while moving that lie where people are known to have walked (the free cells of gray/ground.py)
    off the map  share of its points that are not over annotated floor
    goes         it gets 1 m or further from where it starts; compared with whether the sentence mentions walking
    destination  where the sentence names an object that the map has, and the person really ends
                 within 1 m of it after starting 1.5 m or further away: the path ends within 1 m of it

Because a robot would plan again every second or so, the blocked measure is also given for the first second and the
first two seconds of the path alone. And because several paths can be drawn, two ways of choosing among the 8 draws are
scored: the best that could be done (a draw that goes through nothing, if there is one: needs the truth, so it is a
ceiling and not a method), and, for models trained with --ground, the draw that best stays inside the free distances
the model itself predicts.

Each is also computed for paths drawn with the sentence of another moment, for the recorded path,
and for the recorded path turned by 90 degrees about its start. The distance to the recorded
path (one draw, the best draw, standing still) is reported as well. Writes <run>/score.json.

The same measures for the models of gray/joint.py, on 8 s windows, are in gray/navigate.py.
"""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gray import frames
from gray.census import WALK
from gray.head import NOW, free_guess, load, sample
from gray.joint import to_device
from gray.ground import Ground
from gray.layout import Layout
from gray.moments import HISTORY, Moments, collate
from gray.navigate import AWAY, GOES, NEAR, targets
from gray.tracks import to_world


MOVING = 0.5  # metres per second


def turned(points: np.ndarray, start: np.ndarray, degrees: float = 90.0) -> np.ndarray:
    """World points (N, 3) turned about the vertical through ``start``."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    x, y = points[:, 0] - start[0], points[:, 1] - start[1]
    return np.stack([start[0] + c * x - s * y, start[1] + s * x + c * y, points[:, 2]], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--ground", type=Path, default=Path("/ptmp/enazaren/gray/ground"))
    parser.add_argument("--moments", type=int, default=3000)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None, help="sampling steps; default 20, or 1 for a run trained with --one-step")
    parser.add_argument("--temperature", type=float, default=1.0, help="factor on the starting noise of sampling; other than 1 the result goes to a file of its own")
    parser.add_argument("--guidance", type=float, default=1.0, help="weight of the text in sampling (gray.head.sample); other than 1 the result goes to score-guidance-<value>.json")
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
    if args.steps is None:
        args.steps = 1 if config.get("one_step") else 20
    cache = Path(config["cache"])
    listing = dict(np.load(config["moments"]))
    data = Moments(Path(config["moments"]), Path(config["tracks"]), Path(config["images"]), cache, held=True,
                   patches=Path(config["patches"]) if config.get("patches") else None, only=listing["has_map"],
                   depth=Path(config["depth"]) if config.get("depth") and config["depth"] != "None" else None)
    picked = np.random.default_rng(0).permutation(len(data))[: args.moments]
    loader = DataLoader(Subset(data, picked.tolist()), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    print(f"{len(picked):,d} held-out moments of the navigation subset with a room map, {args.draws} draws each, {HISTORY} frames of history, "
          f"text guidance {args.guidance:g}, {args.steps} sampling step(s), noise times {args.temperature:g}", flush=True)

    layouts: dict[str, Layout] = {}
    grounds: dict[str, Ground] = {}
    location = {m["sequence_id"]: m["location"] for m in json.loads((cache / "index.json").read_text())["sequences"]}
    floors: dict[str, np.ndarray] = {}
    spoken: dict[str, list[str]] = {}

    def sentence_of(row: int) -> str:
        name = str(listing["sequence"][row])
        if name not in spoken:
            spoken[name] = [s["text"] for s in json.loads((cache / name / "sentences.json").read_text())]
        return spoken[name][int(listing["sentence"][row])]

    def check(layout: Layout, ground: Ground, world: np.ndarray) -> tuple[float, float, float, float, float, float]:
        """Touches a wall, moves through a blocked cell, share of the moving points where people walked (nan if never moving),
        share of points off the annotated floor, and moves through a blocked cell within the first 1 s and the first 2 s.
        ``world``: points 0.2 s apart."""
        f = layout.flags_world(world, layout.storeys_world(world))
        g = ground.flags(world)
        moving = np.linalg.norm(np.diff(world[:, :2], axis=0, prepend=world[:1, :2]), axis=1) / 0.2 > MOVING
        walked = float(g["free"][moving].mean()) if moving.any() else float("nan")
        hit = g["blocked"] & moving
        return (float(f["wall"].any()), float(hit.any()), walked, float((~f["floor"] & ~f["unmapped"]).mean()), float(hit[:5].any()), float(hit[:10].any()))

    generator = torch.Generator(device=device)
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        size = batch["pose"].shape[0]
        given = {"real": None, "other": (batch["text"].roll(1, dims=0), batch["text_valid"].roll(1, dims=0))}
        drawn = {}
        for name, text in given.items():
            paths = []
            for draw in range(args.draws):
                generator.manual_seed(1000 + draw)
                paths.append(sample(model, batch, mean, std, args.steps, generator, amp, text, guidance=args.guidance, temperature=args.temperature))
            drawn[name] = torch.stack(paths).double().cpu().numpy()  # (K, B, 75, 3) ground frame
        pose = batch["pose"].double().cpu().numpy()
        guess = free_guess(model, batch, mean, std, amp).cpu().numpy() if hasattr(model, "free_token") else None  # (B, 32)
        yaw = frames.heading(batch["pose"][:, NOW, 3:]).double().cpu().numpy()
        for b in range(size):
            row, other = int(batch["row"][b]), int(batch["row"][(b - 1) % size])
            name, now = str(listing["sequence"][row]), int(batch["now"][b])
            if name not in layouts:
                layouts[name] = Layout(args.maps / f"{name}.npz")
                floors[name] = np.load(Path(config["tracks"]) / f"{name}.npz")["floor"]
            if location[name] not in grounds:
                grounds[location[name]] = Ground(args.ground / f"{location[name]}.npz")
            layout, floor, ground = layouts[name], floors[name], grounds[location[name]]
            start = to_world(pose[b, NOW, :3], floor[now])
            truth = to_world(pose[b, HISTORY:, :3], floor[now + 1 : now + 1 + pose.shape[1] - HISTORY])
            world = {key: to_world(value[:, b], floor[now]) for key, value in drawn.items()}  # (K, 75, 3); generated heights are above the floor at now
            reach = lambda path: float(np.hypot(path[:, 0] - start[0], path[:, 1] - start[1]).max())  # noqa: E731
            text, text_other = sentence_of(row), sentence_of(other)
            item = {"going": bool(listing["going"][row]), "walk": bool(WALK.search(text)), "walk_other": bool(WALK.search(text_other)), "recorded_reach": reach(truth),
                    "recorded": check(layout, ground, truth[::3]), "turned": check(layout, ground, turned(truth[::3], start)),
                    "stay_cm": float(np.linalg.norm(truth[:, :2] - start[:2], axis=1).mean() * 100)}
            each = np.array([check(layout, ground, path[::3]) for path in world["real"]])  # (K, 6)
            item["best_draw"] = float(each[:, 1].min())  # 0 if some draw goes through nothing
            if guess is not None:
                # how far each draw leaves the free distances the model predicts (the walked ones), summed over its points
                step = drawn["real"][:, b, ::3, :] - pose[b, NOW, :3]  # (K, P, 3) ground frame
                forward = -np.sin(yaw[b]) * step[..., 0] + np.cos(yaw[b]) * step[..., 2]
                left = np.cos(yaw[b]) * step[..., 0] + np.sin(yaw[b]) * step[..., 2]
                ray = np.round(np.arctan2(left, forward) / (2 * np.pi / 16)).astype(int) % 16
                outside = np.clip(np.hypot(forward, left) - guess[b, :16][ray], 0, None).sum(1)  # (K,)
                item["chosen_draw"] = float(each[int(outside.argmin()), 1])
                item["chosen_goes"] = float(reach(world["real"][int(outside.argmin())]) >= GOES)
            for key, paths in world.items():
                item[key] = np.nanmean([check(layout, ground, path[::3]) for path in paths], axis=0).tolist()
                item[key + "_reach"] = [reach(path) for path in paths]
                error = np.linalg.norm(paths[:, :, :2] - truth[None, :, :2], axis=-1).mean(-1) * 100
                item[key + "_cm"] = [float(error.mean()), float(error.min())]
            window = int(np.abs(layout.data["floor_z"] - floor[now]).argmin())  # a cached window on the same storey, for the map's lookup of named objects
            cells = targets(layout, window, text)
            if cells is not None:
                gap = lambda point: float(np.linalg.norm(cells - point[:2], axis=1).min())  # noqa: E731
                item["target"] = {"start": gap(start), "recorded": gap(truth[-1]), "real": [gap(path[-1]) for path in world["real"]],
                                  "other": [gap(path[-1]) for path in world["other"]]}
            rows.append(item)
        print(f"  {len(rows):,d} moments", flush=True)

    result = {"moments": len(rows), "draws": args.draws}
    for title, pick in (("all moments", lambda r: True), ("moments that go 1 m or further", lambda r: r["going"]), ("moments that stay", lambda r: not r["going"])):
        part = [r for r in rows if pick(r)]
        if not part:
            continue
        table = {}
        print(f"\n{title} ({len(part):,d})\n  path                      touches a wall   moves through something blocked   moving where people walked   points off the map   gets 1 m or further")
        for label, key in (("recorded", "recorded"), ("recorded, turned 90 deg", "turned"), ("generated", "real"), ("generated, other text", "other")):
            values = np.array([r[key] for r in part])
            goes = (float(np.mean([np.mean(np.array(r[key + "_reach"]) >= GOES) for r in part])) if key in ("real", "other")
                    else float(np.mean([r["recorded_reach"] >= GOES for r in part])) if key == "recorded" else float("nan"))
            table[label] = {"wall": float(values[:, 0].mean()), "blocked": float(values[:, 1].mean()), "walked": float(np.nanmean(values[:, 2])),
                            "off_map": float(values[:, 3].mean()), "goes": goes}
            print(f"  {label:25s} {values[:, 0].mean():14.1%} {values[:, 1].mean():33.1%} {np.nanmean(values[:, 2]):28.1%} {values[:, 3].mean():20.1%} {goes:21.1%}")
        first = {"1 s": 4, "2 s": 5}
        soon = {label: {name: float(np.mean([r[key][column] for r in part])) for name, key in (("recorded", "recorded"), ("turned", "turned"), ("generated", "real"))}
                for label, column in first.items()}
        print("  moves through something blocked within the first " + "; ".join(
            f"{label}: recorded {v['recorded']:.1%}, turned {v['turned']:.1%}, generated {v['generated']:.1%}" for label, v in soon.items()))
        choice = {"any draw": float(np.mean([r["real"][1] for r in part])), "best of 8 (needs the truth)": float(np.mean([r["best_draw"] for r in part]))}
        if "chosen_draw" in part[0]:
            choice["the draw that fits the model's own free distances"] = float(np.mean([r["chosen_draw"] for r in part]))
            choice["... and it gets 1 m or further"] = float(np.mean([r["chosen_goes"] for r in part]))
        print("  moves through something blocked, by how the draw is chosen: " + ", ".join(f"{k} {v:.1%}" for k, v in choice.items()))
        table["first seconds"], table["choice"] = soon, choice
        cm = {"one draw": float(np.mean([r["real_cm"][0] for r in part])), "best draw": float(np.mean([r["real_cm"][1] for r in part])),
              "other text": float(np.mean([r["other_cm"][0] for r in part])), "stay": float(np.mean([r["stay_cm"] for r in part]))}
        print("  distance to the recorded path, cm: " + ", ".join(f"{k} {v:.1f}" for k, v in cm.items()))
        result[title] = {"paths": table, "cm": cm}

    share = lambda part, key: float(np.mean([np.mean(np.array(r[key + "_reach"]) >= GOES) for r in part])) if part else float("nan")  # noqa: E731
    said, unsaid = [r for r in rows if r["walk"]], [r for r in rows if not r["walk"]]
    swap_said, swap_unsaid = [r for r in rows if r["walk_other"]], [r for r in rows if not r["walk_other"]]
    follow = {"recorded": [float(np.mean([r["recorded_reach"] >= GOES for r in said])), float(np.mean([r["recorded_reach"] >= GOES for r in unsaid]))],
              "generated": [share(said, "real"), share(unsaid, "real")],
              "generated, other text (by what that text says)": [share(swap_said, "other"), share(swap_unsaid, "other")]}
    print(f"\ndoes it go when the text says so? share of paths that get 1 m or further in the 5 s after now")
    print(f"  {'':48s} sentence mentions walking ({len(said):,d})   it does not ({len(unsaid):,d})")
    for label, (yes, no) in follow.items():
        print(f"  {label:48s} {yes:24.1%} {no:21.1%}")
    result["goes_by_text"] = follow

    aimed = [r["target"] for r in rows if "target" in r]
    reached = [t for t in aimed if t["recorded"] <= NEAR and t["start"] >= AWAY]
    print(f"\ndestination: {len(aimed):,d} moments name an object that the map has; in {len(reached):,d} the person starts {AWAY:g} m or further from it and ends within {NEAR:g} m")
    if reached:
        destination = {"moments": len(reached), "start_m": float(np.median([t["start"] for t in reached])),
                       "real_within": float(np.mean([np.mean(np.array(t["real"]) <= NEAR) for t in reached])),
                       "other_within": float(np.mean([np.mean(np.array(t["other"]) <= NEAR) for t in reached])),
                       "real_best_within": float(np.mean([min(t["real"]) <= NEAR for t in reached])),
                       "real_m": float(np.median([np.median(t["real"]) for t in reached])), "other_m": float(np.median([np.median(t["other"]) for t in reached]))}
        print(f"  paths ending within {NEAR:g} m of the object: generated {destination['real_within']:.1%}, at least one of the {args.draws} draws {destination['real_best_within']:.1%}, "
              f"generated with other text {destination['other_within']:.1%}")
        print(f"  median distance to the object at the end: start {destination['start_m']:.2f} m, generated {destination['real_m']:.2f} m, with other text {destination['other_m']:.2f} m")
        result["destination"] = destination
    usual = 1 if config.get("one_step") else 20
    plain = args.guidance == 1.0 and args.steps == usual and args.temperature == 1.0
    target = args.run / ("score.json" if plain else f"score-guidance-{args.guidance:g}-steps-{args.steps}-temperature-{args.temperature:g}.json")
    target.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {target}", flush=True)


if __name__ == "__main__":
    main()
