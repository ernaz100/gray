#!/usr/bin/env python3
"""How far are V-JEPA features of a rendered home from those of the real recording?

    1. python -m gray.rendered frames --static ~/cursor-workspace/loc08 --data <data> --out <folder> [--clips 60]
    2. python -m gray.rendered compare --out <folder> --weights <V-JEPA 2 folder or name>

Runs on the workstation (Blender, the home's scene file, its recordings' videos and tracks). Step 1 draws clips at
random from the recordings of one home: 16 frames, now - 30 to now every second frame, as gray/vjepa.py. For each frame it
writes the real pinhole view (gray/seen.py, 256 x 256) and the home's mesh scene rendered from the same recorded camera
pose (gray/rendered_blender.py), and the real frames of the clip 1 s later. Step 2 encodes them with V-JEPA 2 exactly as
gray/vjepa.py does and compares each real clip with

    rendered          the scene rendered along the same camera path
    1 s later         the real clip that ends 15 frames later (how much features change within a second)
    blurred           the real clip blurred (Gaussian, sigma 2 px at 256): detail missing, everything else the same
    elsewhere         the real clip of another draw (another place in the home, or another recording of it)
    rendered elsewhere  the render of another draw: does a render match its own place better than another?

by the cosine of three things: the mean of all tokens, the 8 time slices (each averaged over the image), and the cells
of the 7 x 7 grid of the last slice (what gray/head.py is given). Writes compare.json and sheet.png (real over rendered).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

FRAMES, STRIDE, SIZE, VIEW = 16, 2, 256, 79.6


def span(seen, video: Path, name: str, first: int, count: int) -> np.ndarray:
    """The pinhole views (count, SIZE, SIZE, 3) u8 of ``count`` frames at 15 per second from frame ``first``, in one pass over the video."""
    import imageio_ffmpeg
    import torch

    from gray.rectify import lookup, rectify

    width, height = next(imageio_ffmpeg.read_frames(str(video)))["size"]
    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", f"{first / 15:.3f}", "-i", str(video), "-vf", "fps=15", "-frames:v", str(count),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True, timeout=600).stdout
    full = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3).copy()).permute(0, 3, 1, 2).float()
    if len(full) != count:
        raise RuntimeError(f"{name}: wanted {count} frames from {first}, decoded {len(full)}")
    lens, clockwise = np.array(seen.lens[name]["lens"]), bool(seen.lens[name]["clockwise"])
    where = lookup(lens, full.shape[-1], clockwise, view=VIEW, wide=2 * SIZE, high=2 * SIZE).float()
    return torch.nn.functional.avg_pool2d(rectify(full, where), 2).permute(0, 2, 3, 1).round().clamp(0, 255).byte().numpy()


def frames_step(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image

    from gray import frames
    from gray.seen import Seen

    data = args.data
    names = sorted(m["sequence_id"] for m in json.loads((data / "index.json").read_text())["sequences"] if m["location"] == args.location
                   and (data / "tracks" / f"{m['sequence_id']}.npz").exists() and (data / "videos" / m["sequence_id"] / "preview_rgb.mp4").exists())
    rng = np.random.default_rng(args.seed)
    seen = Seen(data / "videos", data / "lens.json")
    args.out.mkdir(parents=True, exist_ok=True)
    clips, at, axes, files = [], [], [], []
    while len(clips) < args.clips:
        name = names[int(rng.integers(len(names)))]
        track = np.load(data / "tracks" / f"{name}.npz")
        pose, floor, valid = track["pose"], track["floor"], track["valid"]
        now = int(rng.integers(40, len(pose) - 20))
        if not valid[now - 32 : now + 16].all():
            continue
        number = len(clips)
        clips.append({"name": name, "now": now})
        first = now - STRIDE * (FRAMES - 1)
        views = span(seen, data / "videos" / name / "preview_rgb.mp4", name, first, now + 15 - first + 1)
        for i in range(FRAMES):
            for kind, shift in (("real", 0), ("later", 15)):
                Image.fromarray(views[STRIDE * i + shift]).save(args.out / f"{kind}-{number:03d}-{i:02d}.png")
            f = now - STRIDE * (FRAMES - 1 - i)
            camera = frames.matrix(torch.from_numpy(pose[f : f + 1, 3:]).float())[0].double().numpy()  # columns: the camera's x (down the image), y (to its left), z (ahead), Y-up world
            turn = np.stack([-camera[:, 1], -camera[:, 0], -camera[:, 2]], axis=-1)  # Blender's camera: x right, y up, looks along -z (as gray/film.py)
            turn = np.stack([turn[0], -turn[2], turn[1]], axis=0)  # rows to Blender's frame
            where = np.array([pose[f, 0], pose[f, 1] + floor[f], pose[f, 2]], dtype=np.float64)
            at.append([where[0], -where[2], where[1]])
            axes.append(turn.reshape(-1).tolist())
            files.append(f"rendered-{number:03d}-{i:02d}.png")
        print(f"clip {number}: {name} frame {now}", flush=True)
    (args.out / "clips.json").write_text(json.dumps(clips, indent=1))
    (args.out / "scene.json").write_text(json.dumps({"scene": str(args.static / args.scene), "at": at, "axes": axes, "names": files, "view": VIEW, "size": SIZE, "gamma": args.gamma}))
    done = subprocess.run([args.blender, "-b", "-P", str(Path(__file__).with_name("rendered_blender.py")), "--", str(args.out)], capture_output=True, text=True)
    print("\n".join(line for line in done.stdout.splitlines() if line.startswith(("material", "rendered"))) or done.stderr[-2000:], flush=True)


def compare_step(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image, ImageFilter
    from transformers import AutoModel

    clips = json.loads((args.out / "clips.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.weights).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    def clip(kind: str, number: int, blur: float = 0.0) -> torch.Tensor:
        images = [Image.open(args.out / f"{kind}-{number:03d}-{i:02d}.png").convert("RGB") for i in range(FRAMES)]
        if blur:
            images = [image.filter(ImageFilter.GaussianBlur(blur)) for image in images]
        return torch.from_numpy(np.stack([np.asarray(image) for image in images])).permute(0, 3, 1, 2)

    @torch.no_grad()
    def encode(kind: str, blur: float = 0.0) -> torch.Tensor:
        out = []
        for start in range(0, len(clips), args.batch):
            video = torch.stack([clip(kind, n, blur) for n in range(start, min(start + args.batch, len(clips)))]).to(device).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                tokens = model(pixel_values_videos=(video / 255 - mean) / std, skip_predictor=True).last_hidden_state.float()
            out.append(tokens.reshape(len(video), FRAMES // 2, SIZE // 16, SIZE // 16, -1).cpu())
        return torch.cat(out)

    def parts(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = torch.nn.functional.adaptive_avg_pool2d(tokens[:, -1].permute(0, 3, 1, 2), 7).flatten(2).permute(0, 2, 1)
        return {"all tokens averaged": tokens.mean((1, 2, 3)), "time slices": tokens.mean((2, 3)), "grid cells at now": grid}

    cosine = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())  # noqa: E731
    real, rendered, later, blurred = parts(encode("real")), parts(encode("rendered")), parts(encode("later")), parts(encode("real", 2.0))
    other = torch.roll(torch.arange(len(clips)), 1)
    result = {"clips": len(clips), "weights": str(args.weights)}
    for key in real:
        centre = real[key].reshape(-1, real[key].shape[-1]).mean(0)  # what every clip of this home shares; removed for the second number
        rows = {"rendered": (real[key], rendered[key]), "1 s later": (real[key], later[key]), "blurred": (real[key], blurred[key]),
                "elsewhere": (real[key], real[key][other]), "rendered elsewhere": (real[key], rendered[key][other])}
        result[key] = {label: {"cosine": cosine(a, b), "cosine, mean removed": cosine(a - centre, b - centre)} for label, (a, b) in rows.items()}
        print(f"\n{key}\n  {'':20s} cosine   with the mean of the real clips removed")
        for label, v in result[key].items():
            print(f"  {label:20s} {v['cosine']:6.3f}   {v['cosine, mean removed']:6.3f}")
    a, b = real["all tokens averaged"], rendered["all tokens averaged"]
    centre = a.mean(0)
    table = torch.nn.functional.normalize(a - centre, dim=-1) @ torch.nn.functional.normalize(b - b.mean(0), dim=-1).T
    result["retrieval"] = {"real clip finds its own render among all renders (each side centred on its own mean)": float((table.argmax(1) == torch.arange(len(clips))).float().mean()),
                           "chance": 1 / len(clips)}
    print(f"\na real clip finds its own render among the {len(clips)} renders (each side centred on its own mean): {result['retrieval'][next(iter(result['retrieval']))]:.1%}")
    (args.out / "compare.json").write_text(json.dumps(result, indent=1))
    sheet = Image.new("RGB", (8 * SIZE, 2 * SIZE))
    for column in range(8):
        for row, kind in enumerate(("real", "rendered")):
            sheet.paste(Image.open(args.out / f"{kind}-{column:03d}-{FRAMES - 1:02d}.png").convert("RGB"), (column * SIZE, row * SIZE))
    sheet.save(args.out / "sheet.png")
    print(f"wrote {args.out / 'compare.json'} and sheet.png", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    one = steps.add_parser("frames")
    one.add_argument("--static", type=Path, required=True, help="folder with the home's scene file")
    one.add_argument("--scene", default="scene_painted.glb")
    one.add_argument("--data", type=Path, required=True)
    one.add_argument("--location", default="Loc_08")
    one.add_argument("--clips", type=int, default=60)
    one.add_argument("--gamma", type=float, default=2.2, help="the painted colours are stored as sRGB; 1 reads them as linear")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--blender", default="blender")
    two = steps.add_parser("compare")
    two.add_argument("--weights", default="facebook/vjepa2-vitl-fpc64-256")
    two.add_argument("--batch", type=int, default=4)
    for step in (one, two):
        step.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames_step(args) if args.step == "frames" else compare_step(args)


if __name__ == "__main__":
    main()
