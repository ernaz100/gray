#!/usr/bin/env python3
"""Videos to compare body models by eye: the recorded body and each model's body for the same moment, side by side.

    python -m gray.versus --data <folder> --steer <folder> --runs motion-v1 motion-v3 --out <folder>

Runs on a workstation with a copy of some held-out recordings (see gray/gallery.py for the folder), Blender and ffmpeg.
One process, a few minutes; no training. For some held-out moments that go 1 m or further and whose body is known, each
body model generates the body along the recorded head path (4 steps, noise 0.3, with the past body), from the same noise.
The bodies are decoded, posed as SMPL meshes and rendered next to one another on a floor of 0.5 m squares, the recorded one
in grey on the left: a foot that slides shows against the squares. Left of the render: the camera view at now and the sentence.

One MP4 per moment, 5 s at 30 frames per second, and <out>/versus.json with the foot measures of gray/chain.py per body.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

from gray import motion
from gray.chain import feet
from gray.head import anchored
from gray.joint import to_device
from gray.moments import HISTORY, Moments, collate
from gray.scene_worker import Mesh, about_up
from gray.seen import Seen

COLOURS = [(0.45, 0.45, 0.45), (0.05, 0.19, 0.67), (0.83, 0.13, 0.03), (0.0, 0.22, 0.0)]  # recorded, then the runs (linear RGB)
SHOWN = ["#b4b4b4", "#2a78d6", "#eb6834", "#008300"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steer", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="names of body runs under <data>/runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--moments", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    root_dir = args.data.expanduser()

    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(args.steer.expanduser()))
    from data.nymeria import to_actionplan as ap
    from egoprimo import root
    from egoprimo.codec import MotionCodec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec, body = MotionCodec().to(device), Mesh(ap.MODEL, ap.rodrigues)
    models = [motion.load(root_dir / "runs" / name, device) for name in args.runs]
    cache, tracks, images, grids = root_dir / "cache", root_dir / "tracks", root_dir / "pinhole", root_dir / "grids"
    listing = dict(np.load(root_dir / "moments" / "moments.npz"))
    there = {n for n in set(listing["sequence"].tolist()) if (grids / n / "patches.npy").exists() and (root_dir / "body" / f"{n}.npz").exists()
             and (root_dir / "videos" / n / "preview_rgb.mp4").exists() and (cache / n / "text_tokens.npy").exists()}
    data = Moments(root_dir / "moments" / "moments.npz", tracks, images, cache, held=True, patches=grids, body=root_dir / "body",
                   only=listing["going"] & np.isin(listing["sequence"], sorted(there)))
    order = np.random.default_rng(args.seed).permutation(len(data))
    items = []
    for i in order:
        item = data[int(i)]
        if bool(item["body_known"]):
            items.append(item)
        if len(items) == args.moments:
            break
    batch = to_device(collate(items), device)
    truth = motion.body_anchored(batch)
    bodies = [truth[:, HISTORY:]]
    for model, stats, _config in models:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        bodies.append(motion.sample(model, batch, stats, 4, generator, device == "cuda", True, 0.3))
    path = anchored(batch)
    seen = Seen(root_dir / "videos", root_dir / "lens.json")
    found = subprocess.run(["fc-match", "-f", "%{file}", "sans"], capture_output=True, text=True).stdout.strip()

    def font(size: int, bold: bool = False):
        made = ImageFont.truetype(found, size) if found and Path(found).exists() else ImageFont.load_default(size)
        try:
            made.set_variation_by_axes([700 if bold else 400])
        except Exception:  # noqa: BLE001  not a variable font
            pass
        return made

    args.out.expanduser().mkdir(parents=True, exist_ok=True)
    report = []
    labels = ["recorded"] + list(args.runs)
    for b in range(len(items)):
        row = int(batch["row"][b])
        name, now = str(listing["sequence"][row]), int(batch["now"][b])
        sentence = json.loads((cache / name / "sentences.json").read_text())[int(listing["sentence"][row])]["text"]
        work = args.out.expanduser() / f"moment-{b}"
        work.mkdir(exist_ok=True)
        travel = (path[b, -1, [0, 2]] - path[b, HISTORY - 1, [0, 2]]).double().cpu().numpy()
        ahead = travel / max(float(np.linalg.norm(travel)), 1e-6) if np.linalg.norm(travel) > 0.3 else np.array([0.0, 1.0])
        side = np.array([ahead[1], -ahead[0]])
        measures, all_verts = {}, []
        for k, own in enumerate(bodies):
            whole = torch.cat([truth[b : b + 1, :HISTORY], own[b : b + 1]], dim=1).float()
            with torch.no_grad():
                smpl = ap.streamer272_to_smpl(codec.decode_with_root(whole[..., :32], whole[..., 32:]).cpu())
            channels = whole[..., 32:].cpu()
            start = channels[0, 0]
            skin = about_up(body.skin(smpl["poses"][0].double().numpy(), smpl["joints"][0, :, 0].double().numpy()), float(torch.atan2(start[2], start[3])), *start[:2].tolist())[2 * HISTORY :]
            own_joints = skin[:, body.joints4]
            verts = ((np.einsum("tvkab,vb->tvka", own_joints[..., :3], body.vertices) + own_joints[..., 3]) * body.weights4[None, :, :, None]).sum(2)
            shift = ((len(bodies) - 1) / 2 - k) * 1.5 * side  # seen from behind, the recorded body is on the left
            verts[..., 0] += shift[0]
            verts[..., 2] += shift[1]
            all_verts.append(np.stack([verts[..., 0], -verts[..., 2], verts[..., 1]], axis=-1).astype(np.float32))  # to Blender's frame
            joints = root.place(smpl["joints"], channels)[:, 2 * HISTORY :: 2]
            slide, hover = feet(joints)
            measures[labels[k]] = {"foot_slide_cm_s": float(slide), "foot_hover_cm": float(hover)}
            np.save(work / f"verts-{k}.npy", all_verts[-1])
        np.save(work / "faces.npy", body.faces)
        everything = np.concatenate([v.reshape(-1, 3) for v in all_verts])
        centre = [float((everything[:, 0].min() + everything[:, 0].max()) / 2), float((everything[:, 1].min() + everything[:, 1].max()) / 2), 0.9]
        spread = float(max(np.ptp(everything[:, 0]), np.ptp(everything[:, 1])))
        back = np.array([-ahead[0], ahead[1]])  # behind the walkers, in Blender's ground plane (x, -z)
        eye = [centre[0] + back[0] * (spread * 0.55 + 2.6), centre[1] + back[1] * (spread * 0.55 + 2.6), 0.9 + spread * 0.4 + 2.0]
        (work / "scene.json").write_text(json.dumps({"colours": COLOURS[: len(bodies)], "centre": centre, "eye": eye, "size": [1100, 760], "lens": 32}))
        subprocess.run([args.blender, "-b", "-P", str(Path(__file__).with_name("versus_blender.py")), "--", str(work)], check=True, stdout=subprocess.DEVNULL)
        view = Image.fromarray(seen.view(name, now, 400))
        (work / "page").mkdir(exist_ok=True)
        for f in range(len(all_verts[0])):
            page = Image.new("RGB", (1580, 860), (249, 249, 247))
            draw = ImageDraw.Draw(page)
            draw.text((30, 24), "Camera view at now", font=font(18, True), fill=(20, 20, 20))
            page.paste(view, (30, 60))
            y = 480
            for line in textwrap.wrap(f"\"{sentence}\"", 38)[:7]:
                draw.text((30, y), line, font=font(17), fill=(20, 20, 20))
                y += 24
            page.paste(Image.open(work / "frames" / f"{f:04d}.png").convert("RGB"), (460, 60))
            x = 460
            draw.text((x, 24), "The same 5 s, along the recorded head path:", font=font(18, True), fill=(20, 20, 20))
            x += 420
            for k, label in enumerate(labels):
                draw.rectangle((x, 28, x + 16, 44), fill=SHOWN[k])
                text = f"{label} (slides {measures[label]['foot_slide_cm_s']:.0f} cm/s)"
                draw.text((x + 22, 25), text, font=font(15), fill=(60, 60, 60))
                x += 40 + int(draw.textlength(text, font=font(15)))
            draw.text((460, 828), f"{f / 30:0.1f} s   The floor squares are 0.5 m. Bodies are placed 1.5 m apart; they share the same head path.", font=font(14), fill=(95, 94, 90))
            page.save(work / "page" / f"{f:04d}.png")
        target = args.out.expanduser() / f"moment-{b}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(work / "page" / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(target)], check=True)
        report.append({"video": target.name, "sequence": name, "sentence": sentence, "bodies": measures})
        print(f"wrote {target}: " + ", ".join(f"{k} slides {v['foot_slide_cm_s']:.0f} cm/s" for k, v in measures.items()), flush=True)
    (args.out.expanduser() / "versus.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
