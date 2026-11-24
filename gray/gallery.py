#!/usr/bin/env python3
"""Head paths one can judge by eye: per held-out moment one picture with the view, the plans, other prompts and other noise.

    python -m gray.gallery --run <runs>/head-v5 --out <folder>

For some held-out moments that go 1 m or further and have a map, one picture each:

    view            the camera view at now with the sentence, and below it what the camera really showed 1, 2.5 and 5 s later
    plans           the home from above (gray/ground.py) with six generated paths for the recorded sentence, sampled as the
                    planner is used (4 steps, half the starting noise). A path is green if it goes through nothing and red if
                    it does (a point in a blocked cell while moving, as gray/score.py). The recorded path is dashed. If the
                    sentence names an object the map has, the object is outlined in blue.
    other prompts   the same view with five other sentences, one path each without noise (the model's single best guess)
    noise           six paths from full starting noise, and the one path without noise

<out>/card-<n>.png and <out>/gallery.json (per moment: the sentence, how many of the six plans are clear, the distance of each
plan's end to the named object).
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import torch

from gray import frames
from gray.ground import Ground
from gray.head import NOW, load, sample
from gray.joint import to_device
from gray.layout import CELL, HEAD, Layout
from gray.moments import HISTORY, Moments, collate
from gray.navigate import targets
from gray.seen import Seen
from gray.tracks import to_world

PROMPTS = ["C walks straight ahead.", "C turns left and walks.", "C turns right and walks.", "C turns around and walks back.", "C stands still."]
MOVING = 0.5  # metres per second, as gray/score.py


def through(ground: Ground, path: np.ndarray) -> bool:
    """Whether a path (75, 3) in the world goes through something: a point in a blocked cell while moving (points 0.2 s apart)."""
    points = path[::3]
    moving = np.linalg.norm(np.diff(points[:, :2], axis=0, prepend=points[:1, :2]), axis=1) / 0.2 > MOVING
    return bool((ground.flags(points)["blocked"] & moving).any())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ground", type=Path, default=Path("/ptmp/enazaren/gray/ground"))
    parser.add_argument("--maps", type=Path, default=Path("/ptmp/enazaren/gray/layout"))
    parser.add_argument("--videos", type=Path, default=Path("/ptmp/enazaren/nymeria/sequences"))
    parser.add_argument("--clip", type=Path, default=Path("/u/enazaren/steer_assets/weights/clip"))
    parser.add_argument("--data", type=Path, default=None, help="a folder with cache/, tracks/, pinhole/, grids/, moments/, ground/, layout/, videos/, lens.json and clip/: "
                        "use it in place of the cluster's paths (a workstation copy of some held-out recordings); only recordings that are there are used")
    parser.add_argument("--moments", type=int, default=12)
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gray.scene_worker import Text

    if getattr(torch.version, "hip", None):
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    model, mean, std, config = load(args.run, device)
    cache, tracks, images, grids, moments = Path(config["cache"]), Path(config["tracks"]), Path(config["images"]), Path(config["patches"]), Path(config["moments"])
    lens = None
    if args.data is not None:
        root = args.data.expanduser()
        cache, tracks, images, grids, moments = root / "cache", root / "tracks", root / "pinhole", root / "grids", root / "moments" / "moments.npz"
        args.ground, args.maps, args.videos, args.clip, lens = root / "ground", root / "layout", root / "videos", root / "clip", root / "lens.json"
    listing = dict(np.load(moments))
    usable = listing["has_map"] & listing["going"]
    if args.data is not None:
        there = {n for n in set(listing["sequence"].tolist()) if (grids / n / "patches.npy").exists() and (images / n / "dino.npy").exists()
                 and (tracks / f"{n}.npz").exists() and (cache / n / "text_tokens.npy").exists() and (args.videos / n / "preview_rgb.mp4").exists()}
        usable &= np.isin(listing["sequence"], sorted(there))
    data = Moments(moments, tracks, images, cache, held=True, patches=grids, only=usable)
    rng = np.random.default_rng(args.seed)
    named = np.flatnonzero(listing["destination"][data.rows] != "")
    plain = np.flatnonzero(listing["destination"][data.rows] == "")
    picked = np.concatenate([rng.choice(named, args.moments // 2, replace=False), rng.choice(plain, args.moments - args.moments // 2, replace=False)])
    batch = to_device(collate([data[int(i)] for i in picked]), device)
    size = len(picked)
    generator = torch.Generator(device=device)

    def draw(count: int, temperature: float, text=None) -> np.ndarray:
        out = []
        for k in range(count):
            generator.manual_seed(1000 + k)
            out.append(sample(model, batch, mean, std, 4, generator, amp, text, temperature=temperature))
        return torch.stack(out).double().cpu().numpy()  # (K, B, 75, 3) ground frame

    plans, noisy, calm = draw(args.draws, 0.5), draw(args.draws, 1.0), draw(1, 0.0)
    reader = Text(device, args.clip if args.clip.is_dir() else None)
    others = []
    for prompt in PROMPTS:
        words = reader(prompt)
        others.append(draw(1, 0.0, (words[None].expand(size, -1, -1).contiguous(), torch.ones(size, words.shape[0], dtype=torch.bool, device=device)))[0])
    pose = batch["pose"].double().cpu().numpy()
    location = {m["sequence_id"]: m["location"] for m in json.loads((cache / "index.json").read_text())["sequences"]}
    seen = Seen(args.videos, lens)
    args.out.mkdir(parents=True, exist_ok=True)
    palette = plt.get_cmap("tab10")
    report = []

    for b in range(size):
        row = int(batch["row"][b])
        name, now = str(listing["sequence"][row]), int(batch["now"][b])
        sentence = json.loads((cache / name / "sentences.json").read_text())[int(listing["sentence"][row])]["text"]
        floor = np.load(tracks / f"{name}.npz")["floor"]
        ground = Ground(args.ground / f"{location[name]}.npz")
        d = ground.data
        past = to_world(pose[b, :HISTORY, :3], floor[now - HISTORY + 1 : now + 1])
        future = to_world(pose[b, HISTORY:, :3], floor[now + 1 : now + 1 + pose.shape[1] - HISTORY])
        here = past[-1]
        world = lambda paths: [to_world(p, floor[now]) for p in paths]  # noqa: E731
        mine, loud, quiet, asked = world(plans[:, b]), world(noisy[:, b]), world(calm[:, b]), world([o[b] for o in others])
        level = int(np.abs(here[2] - HEAD - d["levels"]).argmin())
        image = np.full((*d["free"].shape[1:], 3), 255, dtype=np.uint8)
        image[d["free"][level]] = (206, 236, 206)
        image[d["steady"][level]] = (232, 150, 140)
        image[d["wall"][level]] = (20, 20, 20)
        extent = (d["origin"][0], d["origin"][0] + image.shape[1] * CELL, d["origin"][1], d["origin"][1] + image.shape[0] * CELL)
        everything = np.concatenate([future[:, :2], *[p[:, :2] for p in mine + loud + asked]])
        half = max(2.5, float(np.abs(everything - here[:2]).max()) + 0.6)
        cells = None
        try:
            layout = Layout(args.maps / f"{name}.npz")
            cells = targets(layout, int(np.abs(layout.data["floor_z"] - floor[now]).argmin()), sentence)
        except Exception:  # noqa: BLE001  no annotated map for this recording
            pass
        yaw = float(frames.heading(batch["pose"][b : b + 1, NOW, 3:]))
        angle = math.degrees(math.atan2(-math.cos(yaw), -math.sin(yaw))) - 90
        hits = [through(ground, p) for p in mine]
        ends = [float(np.linalg.norm(cells - p[-1, :2], axis=1).min()) for p in mine] if cells is not None else None

        def room(axis, title):
            axis.imshow(image, origin="lower", extent=extent, interpolation="nearest")
            if cells is not None:
                axis.scatter(cells[:, 0], cells[:, 1], s=3, color="tab:blue", alpha=0.25, linewidths=0)
            axis.plot(past[:, 0], past[:, 1], color="0.5", linewidth=2)
            axis.scatter(here[0], here[1], marker=(3, 0, angle), s=230, color="black", edgecolor="white", zorder=6)
            axis.set(xlim=(here[0] - half, here[0] + half), ylim=(here[1] - half, here[1] + half), xticks=[], yticks=[])
            axis.set_aspect("equal")
            axis.set_title(title, fontsize=11, loc="left")

        def line(axis, path, **style):
            axis.plot(np.append(here[0], path[:, 0]), np.append(here[1], path[:, 1]), **style)
            axis.scatter(path[-1, 0], path[-1, 1], color=style.get("color"), s=22, zorder=5)

        figure = plt.figure(figsize=(23, 7.6))
        grid = figure.add_gridspec(4, 17, hspace=0.35, wspace=0.6)
        view = figure.add_subplot(grid[:3, 0:4])
        try:
            view.imshow(seen.view(name, now))
        except Exception as error:  # noqa: BLE001
            view.text(0.1, 0.5, f"no frame: {type(error).__name__}")
        view.axis("off")
        view.set_title("camera view now", fontsize=11, loc="left")
        later = figure.add_subplot(grid[3, 0:4])
        try:
            later.imshow(np.concatenate([seen.view(name, now + ahead, 224) for ahead in (15, 37, 75)], axis=1))
        except Exception:  # noqa: BLE001
            pass
        later.axis("off")
        later.set_title("what the camera really showed 1, 2.5 and 5 s later", fontsize=9, loc="left")
        main_axis = figure.add_subplot(grid[:, 4:9])
        clear = sum(not h for h in hits)
        room(main_axis, f"six plans for this sentence: {clear} of {len(hits)} go through nothing")
        for path, hit in zip(mine, hits):
            line(main_axis, path, color="#d03b3b" if hit else "#0a8f0a", linewidth=2.2, alpha=0.9)
        line(main_axis, future, color="black", linewidth=2.2, linestyle="--")
        main_axis.plot([], [], color="#0a8f0a", linewidth=2.2, label="plan, clear")
        main_axis.plot([], [], color="#d03b3b", linewidth=2.2, label="plan, goes through something")
        main_axis.plot([], [], color="black", linewidth=2.2, linestyle="--", label="what the person did")
        if cells is not None:
            main_axis.scatter([], [], s=30, color="tab:blue", alpha=0.5, label=f"named: {listing['destination'][row]}")
        main_axis.legend(fontsize=9, loc="upper right", framealpha=0.9)
        prompts_axis = figure.add_subplot(grid[:, 9:14])
        room(prompts_axis, "other prompts, same view")
        for k, (path, prompt) in enumerate(zip(asked, PROMPTS)):
            line(prompts_axis, path, color=palette(k), linewidth=2.2, label=prompt.replace("C ", "").rstrip("."))
        prompts_axis.legend(fontsize=9, loc="upper right", framealpha=0.9)
        loud_axis = figure.add_subplot(grid[:2, 14:17])
        room(loud_axis, f"full noise: {sum(not through(ground, p) for p in loud)} of {len(loud)} clear")
        for path in loud:
            line(loud_axis, path, color="#d03b3b" if through(ground, path) else "#0a8f0a", linewidth=1.6, alpha=0.9)
        quiet_axis = figure.add_subplot(grid[2:, 14:17])
        room(quiet_axis, "no noise: one best guess")
        line(quiet_axis, quiet[0], color="#d03b3b" if through(ground, quiet[0]) else "#0a8f0a", linewidth=2.2)
        figure.suptitle("\n".join(textwrap.wrap(f"\"{sentence}\"", 150)[:2]), fontsize=15, x=0.125, ha="left", y=0.99)
        figure.savefig(args.out / f"card-{b:02d}.png", dpi=70, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        report.append({"card": f"card-{b:02d}.png", "sequence": name, "location": location[name], "sentence": sentence, "named": str(listing["destination"][row]),
                       "clear": clear, "draws": len(hits), "recorded_through": through(ground, future), "end_to_named_m": ends,
                       "travel_m": [float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()) for p in mine],
                       "recorded_travel_m": float(np.linalg.norm(np.diff(future[:, :2], axis=0), axis=1).sum()), "full_noise_clear": sum(not through(ground, p) for p in loud),
                       "no_noise_clear": not through(ground, quiet[0])})
        print(f"{b}: {location[name]} {clear}/{len(hits)} clear; \"{sentence[:70]}\"", flush=True)
    (args.out / "gallery.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
