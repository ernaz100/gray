#!/usr/bin/env python3
"""A video of one prompted motion: what the models were given, the body as a mesh in the home's scene, and the view from the planned camera.

    python -m gray.film --static ~/cursor-workspace/loc08 --data ~/agent-workspace/gray/local/data --out <folder> \\
        --text "C walks to the kitchen." [--pos x y z --fwd x y z | --spot 3] [--seed 0]

Runs on a workstation, next to a running gray/scene_server.py (it asks that server for the motion), with Blender and
ffmpeg. One MP4 per call, 5 s at 30 frames per second, in three columns:

    given      the prompt, and the pinhole views the models were shown: the frame at now, large, and the frames 1 s and 2 s
               before (gray/seen.py; the same pictures the image features were computed from)
    generated  the home's mesh scene from above and behind, walls see-through, with the planned head path as an orange line
               (and its trace on the floor) and the generated body as an SMPL mesh walking along it
    planned view   the same scene rendered from the planned camera poses: what the camera would see along the plan

``--spot N`` takes the N-th of a fixed list of recorded places in the home; ``--pos`` and ``--fwd`` give any place (Y-up world).
The body mesh is the neutral SMPL-H shape posed by blend skinning, as in the viewer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
import urllib.request
from pathlib import Path

import numpy as np

VIEW = 79.6


def ask(server: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(server + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))


def to_blender(points: np.ndarray) -> np.ndarray:
    """The Y-up world of the scene file (x, up, z) -> Blender's frame after import (x, -z, up)."""
    return np.stack([points[..., 0], -points[..., 2], points[..., 1]], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--static", type=Path, required=True, help="the folder with the scene (scene.glb)")
    parser.add_argument("--data", type=Path, required=True, help="the worker's data folder (videos/ and lens.json are used here)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--pos", type=float, nargs=3)
    parser.add_argument("--fwd", type=float, nargs=3)
    parser.add_argument("--spot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--server", default="http://127.0.0.1:8771")
    parser.add_argument("--scene", default="scene.glb")
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    from gray.seen import Seen

    static, data = args.static.expanduser(), args.data.expanduser()
    if args.pos is None:
        tracks = [t for t in sorted((data / "tracks").glob("*.npz")) if (data / "videos" / t.stem / "preview_rgb.mp4").exists()]
        track = np.load(tracks[(7 * args.spot) % len(tracks)])
        valid = np.flatnonzero(track["valid"])
        pose = track["pose"][valid[(len(valid) * (1 + args.spot % 9)) // 11]]
        row3 = np.cross(pose[3:6], pose[6:9])  # third row of the camera-to-ground rotation; its third column is where the camera looks
        args.pos, args.fwd = [float(pose[0]), float(pose[1] + track["floor"][0]), float(pose[2])], [float(pose[5]), 0.0, float(row3[2])]
    reply = ask(args.server, "/api/generate", {"pos": args.pos, "fwd": args.fwd, "text": args.text, "seed": args.seed})
    if "error" in reply:
        raise SystemExit(reply["error"])
    rest = ask(args.server, "/api/body")
    name = args.name or "-".join(args.text.lower().replace(".", "").split()[:6])
    work = args.out.expanduser() / name
    work.mkdir(parents=True, exist_ok=True)

    count, floor = reply["frames"], reply["floor_m"]
    skin = np.array(reply["skin"]).reshape(count, 22, 3, 4)
    vertices = np.array(rest["vertices_mm"], dtype=np.float64).reshape(-1, 3) / 1000
    joints4, weights4 = np.array(rest["joints4"]).reshape(-1, 4), np.array(rest["weights4"]).reshape(-1, 4)
    own = skin[:, joints4]  # (T, V, 4, 3, 4): the transforms of each vertex's four joints
    moved = np.einsum("tvkab,vb->tvka", own[..., :3], vertices) + own[..., 3]  # (T, V, 4, 3)
    posed = (moved * weights4[None, :, :, None]).sum(2)
    posed[..., 1] += floor
    camera = np.array(reply["head_mm"], dtype=np.float64).reshape(-1, 3) / 1000
    camera[:, 1] += floor
    axes = np.array(reply["head_axes"]).reshape(-1, 3, 3)  # columns: the camera's x (down the image), y (to its left), z (ahead), in the Y-up world
    blender_axes = np.stack([-axes[:, :, 1], -axes[:, :, 0], -axes[:, :, 2]], axis=-1)  # Blender's camera: x right, y up, looks along -z
    blender_axes = np.stack([blender_axes[:, 0], -blender_axes[:, 2], blender_axes[:, 1]], axis=1)  # rows to Blender's frame
    path = to_blender(camera)
    pelvis = to_blender(posed[:, :, :].mean(1))
    centre = np.array([*((pelvis[:, :2].min(0) + pelvis[:, :2].max(0)) / 2), floor + 0.9])
    travel = pelvis[-1, :2] - pelvis[0, :2]
    side = np.array([-travel[1], travel[0]]) / max(float(np.linalg.norm(travel)), 1e-6) if np.linalg.norm(travel) > 0.3 else np.array([0.0, -1.0])
    reach = max(3.2, float(np.abs(pelvis[:, :2] - centre[:2]).max()) * 1.6 + 2.2)
    eye = [float(centre[0] + side[0] * reach * 0.75), float(centre[1] + side[1] * reach * 0.75), float(floor + reach * 1.25)]
    np.save(work / "verts.npy", to_blender(posed).astype(np.float32))
    np.save(work / "faces.npy", np.array(rest["faces"]).reshape(-1, 3))
    (work / "scene.json").write_text(json.dumps({"scene": str(static / args.scene), "path": path.tolist(), "axes": blender_axes.reshape(-1, 9).tolist(), "floor": floor,
                                                  "centre": centre.tolist(), "eye": eye, "third_size": [760, 760], "first_size": [400, 400], "view": VIEW}))
    (work / "reply.json").write_text(json.dumps({key: value for key, value in reply.items() if key not in ("skin", "image_jpeg", "recorded_jpeg")}))
    subprocess.run([args.blender, "-b", "-P", str(Path(__file__).with_name("film_blender.py")), "--", str(work)], check=True, stdout=subprocess.DEVNULL)

    seen = Seen(data / "videos", data / "lens.json")
    match = reply["match"]
    views = {back: Image.fromarray(seen.view(match["sequence"], match["feature"] - back, 400)) for back in (0, 15, 29)}

    found = subprocess.run(["fc-match", "-f", "%{file}", "sans"], capture_output=True, text=True).stdout.strip()

    def font(size: int, bold: bool = False):
        """The system's sans face; a variable font is set to its bold weight."""
        if not found or not Path(found).exists():
            return ImageFont.load_default(size)
        made = ImageFont.truetype(found, size)
        try:
            made.set_variation_by_axes([700 if bold else 400])
        except Exception:  # noqa: BLE001  not a variable font
            pass
        return made

    wide, high, ink, soft, paper = 1680, 900, (20, 20, 20), (95, 94, 90), (249, 249, 247)
    (work / "frames").mkdir(exist_ok=True)
    for f in range(count):
        page = Image.new("RGB", (wide, high), paper)
        draw = ImageDraw.Draw(page)
        draw.text((30, 22), "Given to the models", font=font(22, True), fill=ink)
        draw.text((470, 22), "Generated: head path and body", font=font(22, True), fill=ink)
        draw.text((1250, 22), "From the planned camera", font=font(22, True), fill=ink)
        y = 66
        draw.text((30, y), "Prompt", font=font(14), fill=soft)
        for line in textwrap.wrap(f"\"{reply['text']}\"", 34):
            y += 26
            draw.text((30, y), line, font=font(20, True), fill=ink)
        y += 44
        draw.text((30, y), "Camera view now", font=font(14), fill=soft)
        page.paste(views[0], (30, y + 22))
        y += 22 + 400 + 14
        draw.text((30, y), "1 s before", font=font(14), fill=soft)
        draw.text((235, y), "2 s before", font=font(14), fill=soft)
        page.paste(views[15].resize((195, 195)), (30, y + 22))
        page.paste(views[29].resize((195, 195)), (235, y + 22))
        draw.text((30, high - 40), "plus its own head poses and body of those 2 s", font=font(13), fill=soft)
        page.paste(Image.open(work / "third" / f"{f:04d}.png").convert("RGB"), (470, 66))
        draw.text((470, 836), "orange: planned camera path (and its trace on the floor); blue: generated body. Walls are drawn see-through.", font=font(13), fill=soft)
        page.paste(Image.open(work / "first" / f"{min(f // 2, len(camera) - 1):04d}.png").convert("RGB"), (1250, 66))
        draw.text((1250, 476), "the scene model rendered from the planned", font=font(13), fill=soft)
        draw.text((1250, 494), "camera poses (body hidden)", font=font(13), fill=soft)
        seconds = f / reply["fps"]
        draw.text((1250, 560), f"{seconds:0.1f} s", font=font(40, True), fill=ink)
        draw.rectangle((1250, 620, 1650, 628), fill=(225, 224, 217))
        draw.rectangle((1250, 620, 1250 + int(400 * (f + 1) / count), 628), fill=(42, 120, 214))
        for i, line in enumerate((f"pelvis travels {reply['travel_m']} m in 5 s", f"head joint {reply['head_to_camera_cm']} cm from the planned camera",
                                  f"starts from a recorded frame {match['distance_m']} m from the spot", "head path: head-v5   body: motion-v1")):
            draw.text((1250, 650 + 24 * i), line, font=font(15), fill=soft)
        page.save(work / "frames" / f"{f:04d}.png")
    target = args.out.expanduser() / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(reply["fps"]), "-i", str(work / "frames" / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(target)], check=True)
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
