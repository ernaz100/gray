#!/usr/bin/env python3
"""Videos of our chain and the EgoPriMo reimplementation in a home's mesh scene: the same start, the same sentence, one body each.

    1. on a cluster (GPU):   python -m gray.versus_scene export --head-run <run> --body-run <run> --out <folder> [--location Loc_08]
    2. on a workstation:     python -m gray.versus_scene render --exported <folder> --static ~/cursor-workspace/loc08 --data <data> --out <folder>

Step 1 draws moments of the navigation subset in one home at random (those that go 1 m or further and have a body and a
map), and for each generates 5 s with both models from the same information: the frame at now and the one sentence (as
gray/versus_egoprimo.py; ours with no earlier frames and no body, 4 steps, noise 0.5 and 0.3). It writes moments.npz:
per moment and body the skinning matrices in the home's world, and the numbers shown in the video.

Step 2 needs Blender, ffmpeg, the home's scene file and its videos (for the camera view at now). One MP4 per moment:
the camera view and the sentence on the left; the room from above and behind with the three bodies walking in it
(recorded grey, ours blue, EgoPriMo orange), each with the trace of its pelvis on the floor; and the same from straight above.

Loc_08, the only home with a mesh scene, is a training home of both models: these videos show how the motions look in a
room, they are not a test. The test is gray/versus_egoprimo.py on held-out homes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

LABELS = ["recorded", "ours", "EgoPriMo"]
COLOURS = [(150, 150, 150), (42, 120, 214), (235, 104, 52)]  # sRGB


def export(args: argparse.Namespace) -> None:
    import torch

    from gray import frames, head, motion
    from gray.chain import Decoder, feet
    from gray.ground import Ground
    from gray.joint import to_device
    from gray.moments import HISTORY, Moments, collate
    from gray.pinhole import STEER
    from gray.scene_worker import Mesh, about_up
    from gray.tracks import to_world

    now_at = HISTORY - 1
    if getattr(torch.version, "hip", None):
        torch.backends.cuda.preferred_blas_library("cublas")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    sys.path.insert(0, str(STEER))
    from data.nymeria import to_actionplan as ap
    from egoprimo import train as ego
    from egoprimo.codec import MotionCodec
    from egoprimo.model import EgoMotionDiT, ModelConfig

    ego_config = json.loads((args.ego_run / "config.json").read_text())
    ego_model = EgoMotionDiT(ModelConfig(**ego_config["model"])).to(device)
    ego_model.load_state_dict(torch.load(args.ego_run / ("best.pt" if (args.ego_run / "best.pt").exists() else "last.pt"), map_location=device, weights_only=True)["model"])
    ego_model.eval()
    planner, head_mean, head_std, config = head.load(args.head_run, device)
    model, stats, body_config = motion.load(args.body_run, device)
    codec, mesh, decoder = MotionCodec().to(device), Mesh(ap.MODEL, ap.rodrigues), Decoder(device)
    cache, tracks = Path(config["cache"]), Path(config["tracks"])
    listing = dict(np.load(config["moments"]))
    index = json.loads((cache / "index.json").read_text())["sequences"]
    there = sorted(m["sequence_id"] for m in index if m["location"] == args.location)
    held = bool(listing["held"][np.isin(listing["sequence"], there)].any())
    data = Moments(Path(config["moments"]), tracks, Path(config["images"]), cache, held=held, patches=Path(config["patches"]), body=Path(body_config["body"]),
                   only=listing["going"] & listing["has_map"] & np.isin(listing["sequence"], there))
    print(f"{args.location}: {len(there)} recordings, {len(data):,d} moments that go; a {'held-out' if held else 'training'} home", flush=True)
    items = []
    for i in np.random.default_rng(args.seed).permutation(len(data)):
        item = data[int(i)]
        if bool(item["body_known"]):
            items.append(item)
        if len(items) == args.moments:
            break
    batch = to_device(collate(items), device)
    size, rows = len(items), batch["row"].tolist()
    image, said, floors, names, texts = torch.zeros(size, ego.TOKENS, 384), torch.zeros(size, 768), [], [], []
    for b, row in enumerate(rows):
        name = str(listing["sequence"][row])
        image[b, 0] = torch.from_numpy(np.array(np.load(cache / name / "dino.npy", mmap_mode="r")[int(batch["now"][b])], dtype=np.float32))
        said[b] = torch.from_numpy(np.array(np.load(cache / name / "text_sentence.npy", mmap_mode="r")[int(listing["sentence"][row])], dtype=np.float32))
        floors.append(float(np.load(tracks / f"{name}.npz")["floor"][int(batch["now"][b])]))
        names.append(name)
        texts.append(json.loads((cache / name / "sentences.json").read_text())[int(listing["sentence"][row])]["text"])
    theirs = ego.to_device({"motion": torch.zeros(size, ego.TOKENS, 36), "image": image, "head": torch.zeros(size, ego.TOKENS, 9), "has_text": torch.ones(size, dtype=torch.bool),
                            "text": batch["text"][:, : ego.MAX_TEXT].cpu(), "sentence": said, "text_valid": batch["text_valid"][:, : ego.MAX_TEXT].cpu()}, device)
    with torch.no_grad():
        out = ego.sample(ego_model, theirs, torch.ones(size, dtype=torch.long, device=device), 50, seed=args.seed, amp=amp)
    ego_body = torch.cat([out[..., :32] * ego_config["latent_sigma"] + ego_config["latent_mean"], out[..., 32:]], dim=-1)[:, 1:76]
    past, truth = head.anchored(batch), motion.body_anchored(batch)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    path = head.sample(planner, batch, head_mean, head_std, 4, generator, amp, history=1, temperature=0.5, poses=True)
    ours = motion.sample(model, batch, stats, 4, generator, amp, False, 0.3, head=torch.cat([past[:, :HISTORY], path], dim=1), history=1)
    bodies = {"recorded": truth[:, HISTORY:], "ours": ours, "EgoPriMo": ego_body}
    joints = {k: decoder.joints(v.float()).double() for k, v in bodies.items()}
    yaw = frames.heading(batch["pose"][:, now_at, 3:]).double().cpu().numpy()
    here = batch["pose"][:, now_at].double().cpu().numpy()
    grounds: dict[str, Ground] = {}
    skins = np.zeros((size, len(LABELS), 2 * 75, 22, 3, 4), dtype=np.float32)
    notes = []
    for b in range(size):
        note = {"sequence": names[b], "now": int(batch["now"][b]), "sentence": texts[b], "floor": floors[b], "bodies": {}}
        for k, label in enumerate(LABELS):
            whole = torch.cat([truth[b : b + 1, :HISTORY], bodies[label][b : b + 1]], dim=1).float()
            with torch.no_grad():
                smpl = ap.streamer272_to_smpl(codec.decode_with_root(whole[..., :32], whole[..., 32:]).cpu())
            start = whole[0, 0, 32:].cpu()
            skin = about_up(mesh.skin(smpl["poses"][0].double().numpy(), smpl["joints"][0, :, 0].double().numpy()), float(torch.atan2(start[2], start[3])), *start[:2].tolist())[2 * HISTORY :]
            skin = about_up(skin, float(yaw[b]), float(here[b, 0]), float(here[b, 2]))  # the frame anchored at now -> the track's ground frame
            skin[..., 1, 3] += floors[b]  # -> the home's world, Y up
            skins[b, k] = skin
            own = joints[label][b]
            pelvis = own[:, 0, [0, 2]]
            c, s = np.cos(yaw[b]), np.sin(yaw[b])
            p = own[::3, 15].cpu().numpy()
            if args.location not in grounds:
                grounds[args.location] = Ground(args.ground / f"{args.location}.npz")
            world = to_world(np.stack([c * p[:, 0] - s * p[:, 2] + here[b, 0], p[:, 1], s * p[:, 0] + c * p[:, 2] + here[b, 2]], axis=1), floors[b])
            moving = np.linalg.norm(np.diff(world[:, :2], axis=0, prepend=world[:1, :2]), axis=1) / 0.2 > 0.5
            note["bodies"][label] = {"travel_m": float((pelvis[1:] - pelvis[:-1]).norm(dim=-1).sum()), "slide_cm_s": float(feet(joints[label][b : b + 1].float())[0]),
                                     "through": bool((grounds[args.location].flags(world)["blocked"] & moving).any()),
                                     "joints_cm": float((own - joints["recorded"][b]).norm(dim=-1).mean() * 100)}
        notes.append(note)
        print(f"  {b}: {texts[b][:70]!r}  " + "  ".join(f"{k} {v['travel_m']:.1f} m" for k, v in note["bodies"].items()), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "moments.npz", skin=skins, vertices=mesh.vertices.astype(np.float32), faces=mesh.faces, joints4=mesh.joints4, weights4=mesh.weights4.astype(np.float32))
    (args.out / "moments.json").write_text(json.dumps({"location": args.location, "held_out": held, "head_run": args.head_run.name, "body_run": args.body_run.name, "moments": notes}, indent=1))
    print(f"wrote {args.out}/moments.npz and moments.json", flush=True)


def render(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw, ImageFont

    from gray.seen import Seen

    exported, static, data, out = (p.expanduser() for p in (args.exported, args.static, args.data, args.out))
    saved = np.load(exported / "moments.npz")
    notes = json.loads((exported / "moments.json").read_text())
    vertices, joints4, weights4 = saved["vertices"].astype(np.float64), saved["joints4"], saved["weights4"].astype(np.float64)
    seen = Seen(data / "videos", data / "lens.json")
    found = subprocess.run(["fc-match", "-f", "%{file}", "sans"], capture_output=True, text=True).stdout.strip()

    def font(size: int, bold: bool = False):
        if not found or not Path(found).exists():
            return ImageFont.load_default(size)
        made = ImageFont.truetype(found, size)
        try:
            made.set_variation_by_axes([700 if bold else 400])
        except Exception:  # noqa: BLE001  not a variable font
            pass
        return made

    placed = notes.get("kind") == "placed"  # the output of gray/place.py show: HumanML3D motions that fit this frame
    captions = json.loads(args.captions.expanduser().read_text()) if placed else {}
    palette = [(42, 120, 214), (235, 104, 52), (0, 131, 0), (160, 107, 214)] if placed else COLOURS
    out.mkdir(parents=True, exist_ok=True)
    chosen = range(len(notes["moments"])) if args.only is None else args.only
    for b in chosen:
        note = notes["moments"][b]
        work = out / f"moment-{b}"
        work.mkdir(exist_ok=True)
        traces, bodies = [], []
        labels = list(note["bodies"])
        for k, label in enumerate(labels):
            own = saved["skin"][b, k].astype(np.float64)[:, joints4]  # (T, V, 4, 3, 4)
            posed = ((np.einsum("tvkab,vb->tvka", own[..., :3], vertices) + own[..., 3]) * weights4[None, :, :, None]).sum(2)
            blender = np.stack([posed[..., 0], -posed[..., 2], posed[..., 1]], axis=-1).astype(np.float32)  # the scene file's Y-up world -> Blender's frame
            np.save(work / f"verts-{k}.npy", blender)
            traces.append(blender.mean(1)[:, :2])
            bodies.append({"verts": f"verts-{k}.npy", "colour": palette[k]})
        np.save(work / "faces.npy", saved["faces"])
        everywhere = np.concatenate(traces)
        floor = note["floor"]
        centre = np.array([*((everywhere.min(0) + everywhere.max(0)) / 2), floor + 0.9])
        travel = traces[0][-1] - traces[0][0]
        side = np.array([-travel[1], travel[0]]) / max(float(np.linalg.norm(travel)), 1e-6) if np.linalg.norm(travel) > 0.3 else np.array([0.0, -1.0])
        reach = max(3.2, float(np.abs(everywhere - centre[:2]).max()) * 1.6 + 2.2)
        eye = [float(centre[0] + side[0] * reach * 0.75), float(centre[1] + side[1] * reach * 0.75), float(floor + reach * 1.25)]
        (work / "scene.json").write_text(json.dumps({"scene": str(static / args.scene), "bodies": bodies, "floor": floor, "centre": centre.tolist(), "eye": eye,
                                                      "third_size": [820, 760], "top_size": [520, 520], "top_width": float(reach * 1.5)}))
        subprocess.run([args.blender, "-b", "-P", str(Path(__file__).with_name("film_blender.py")), "--", str(work)], check=True, stdout=subprocess.DEVNULL)
        view = Image.fromarray(seen.view(note["sequence"], note["now"], 360))
        (work / "page").mkdir(exist_ok=True)
        shown = ["#%02x%02x%02x" % c for c in palette]
        count = len(traces[0])
        for f in range(count):
            page = Image.new("RGB", (1780, 880), (249, 249, 247))
            draw = ImageDraw.Draw(page)
            draw.text((30, 22), "The recorded frame" if placed else "Given to both models", font=font(20, True), fill=(20, 20, 20))
            draw.text((30, 60), "Camera view now", font=font(14), fill=(95, 94, 90))
            page.paste(view, (30, 82))
            y = 458
            if placed:
                draw.text((30, y), "HumanML3D motions kept for this frame", font=font(14), fill=(95, 94, 90))
                y += 26
                for k, label in enumerate(labels):
                    m = note["bodies"][label]
                    draw.rectangle((30, y + 3, 46, y + 19), fill=shown[k])
                    for line in textwrap.wrap(captions.get(m["hml"].lstrip("M"), captions.get(m["hml"], ["?"]))[0], 40)[:4]:
                        draw.text((54, y), line, font=font(15, True), fill=(20, 20, 20))
                        y += 21
                    draw.text((54, y), f"travels {m['travel_m']:.1f} m  ·  HumanML3D {m['hml']}", font=font(13), fill=(95, 94, 90))
                    y += 34
            else:
                draw.text((30, y), "Sentence", font=font(14), fill=(95, 94, 90))
                y = 482
                for line in textwrap.wrap(f"\"{note['sentence']}\"", 36)[:6]:
                    draw.text((30, y), line, font=font(17, True), fill=(20, 20, 20))
                    y += 24
                y = max(y + 22, 640)
            for k, label in enumerate([] if placed else labels):
                m = note["bodies"][label]
                draw.rectangle((30, y + 3, 46, y + 19), fill=shown[k])
                draw.text((54, y), label, font=font(16, True), fill=(20, 20, 20))
                draw.text((54, y + 22), f"travels {m['travel_m']:.1f} m" + ("" if label == "recorded" else f", {m['joints_cm']:.0f} cm from recorded"), font=font(13), fill=(95, 94, 90))
                if m["through"]:
                    draw.text((54, y + 40), "goes through something", font=font(13), fill=(190, 60, 40))
                y += 66
            draw.text((420, 22), "Placed in the home's scene, 5 s" if placed else "The first 5 s, in the home's scene", font=font(20, True), fill=(20, 20, 20))
            page.paste(Image.open(work / "third" / f"{f:04d}.png").convert("RGB"), (420, 60))
            draw.text((1260, 22), "From above", font=font(20, True), fill=(20, 20, 20))
            page.paste(Image.open(work / "top" / f"{f:04d}.png").convert("RGB"), (1260, 60))
            draw.text((1260, 600), f"{f / 30:0.1f} s", font=font(40, True), fill=(20, 20, 20))
            draw.rectangle((1260, 660, 1740, 668), fill=(225, 224, 217))
            draw.rectangle((1260, 660, 1260 + int(480 * (f + 1) / count), 668), fill=(42, 120, 214))
            for i, line in enumerate(("Lines on the floor: where each pelvis goes in the 5 s.", "Walls are drawn see-through.", "Each motion's camera starts on the recorded camera;", "it is kept because it fits the home's map.") if placed else
                                     ("Lines on the floor: where each pelvis goes in the 5 s.", "Walls are drawn see-through.", "Each model is given only this frame and this sentence.",
                                      f"{notes['location']} is a training home of both models.")):
                draw.text((1260, 690 + 22 * i), line, font=font(14), fill=(95, 94, 90))
            page.save(work / "page" / f"{f:04d}.png")
        target = out / f"moment-{b}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(work / "page" / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(target)], check=True)
        print(f"wrote {target}", flush=True)


def floor(args: argparse.Namespace) -> None:
    """Videos of the clips written by gray/versus_uem.py --videos: recorded, ours and UniEgoMotion on a plain floor, from the same start."""
    from PIL import Image, ImageDraw, ImageFont

    exported, out = args.exported.expanduser(), args.out.expanduser()
    found = subprocess.run(["fc-match", "-f", "%{file}", "sans"], capture_output=True, text=True).stdout.strip()

    def font(size: int, bold: bool = False):
        made = ImageFont.truetype(found, size) if found and Path(found).exists() else ImageFont.load_default(size)
        try:
            made.set_variation_by_axes([700 if bold else 400])
        except Exception:  # noqa: BLE001  not a variable font
            pass
        return made

    linear = lambda c: [(v / 255 / 12.92 if v / 255 <= 0.04045 else ((v / 255 + 0.055) / 1.055) ** 2.4) for v in c]  # noqa: E731
    out.mkdir(parents=True, exist_ok=True)
    for b, note in enumerate(json.loads((exported / "clips.json").read_text())):
        work = exported / f"clip-{b}"
        traces = [np.load(work / f"verts-{k}.npy").mean(1)[:, :2] for k in range(3)]
        everywhere = np.concatenate(traces)
        centre = [*((everywhere.min(0) + everywhere.max(0)) / 2).tolist(), note["floor"] + 0.9]
        spread = float(np.ptp(everywhere, axis=0).max())
        travel = traces[0][-1] - traces[0][0]
        back = -travel / np.linalg.norm(travel) if np.linalg.norm(travel) > 0.3 else np.array([0.0, -1.0])
        side = np.array([-back[1], back[0]])
        eye = [centre[0] + (back[0] * 0.8 + side[0] * 0.6) * (spread * 0.6 + 3.2), centre[1] + (back[1] * 0.8 + side[1] * 0.6) * (spread * 0.6 + 3.2), note["floor"] + 2.2 + spread * 0.45]
        (work / "scene.json").write_text(json.dumps({"colours": [linear(c) for c in (COLOURS[0], COLOURS[1], (0, 131, 0))], "centre": [float(v) for v in centre], "eye": [float(v) for v in eye], "size": [1100, 760], "lens": 32, "floor": note["floor"]}))
        subprocess.run([args.blender, "-b", "-P", str(Path(__file__).with_name("versus_blender.py")), "--", str(work)], check=True, stdout=subprocess.DEVNULL)
        (work / "page").mkdir(exist_ok=True)
        shown = ["#%02x%02x%02x" % c for c in (COLOURS[0], COLOURS[1], (0, 131, 0))]
        count = len(traces[0])
        for f in range(count):
            page = Image.new("RGB", (1500, 860), (249, 249, 247))
            draw = ImageDraw.Draw(page)
            recon = note.get("task", "gen") == "recon"
            draw.text((30, 22), "Reconstruction from the head path" if recon else "Generation from the first frame", font=font(20, True), fill=(20, 20, 20))
            draw.text((30, 54), "UniEgoMotion's validation data, no text", font=font(14), fill=(95, 94, 90))
            for i, line in enumerate(textwrap.wrap(note["clip"].split("___")[0].replace("_", " "), 30)[:2]):
                draw.text((30, 96 + 24 * i), line, font=font(17, True), fill=(20, 20, 20))
            y = 190
            for k, label in enumerate(("recorded", "ours", "UniEgoMotion")):
                m = note["bodies"][label]
                draw.rectangle((30, y + 3, 46, y + 19), fill=shown[k])
                draw.text((54, y), label, font=font(16, True), fill=(20, 20, 20))
                draw.text((54, y + 22), f"travels {m['travel_m']:.1f} m" + ("" if label == "recorded" else f", {m['joints_cm']:.0f} cm from recorded"), font=font(13), fill=(95, 94, 90))
                y += 58
            draw.text((30, 420), f"{f / 30:0.1f} s", font=font(40, True), fill=(20, 20, 20))
            draw.rectangle((30, 480, 350, 488), fill=(225, 224, 217))
            draw.rectangle((30, 480, 30 + int(320 * (f + 1) / count), 488), fill=(42, 120, 214))
            given = (("Both follow the recorded head path.", "UniEgoMotion is given the whole camera", "path and all images; ours the camera", "path and the first image only.") if recon else
                     ("All three start in the same place.", "Each model is given the first frame only:", "its image and, for ours, the camera's", "height and tilt."))
            for i, line in enumerate((*given, "The floor squares are 0.5 m.", "The room is not shown: this data has", "no scene geometry.")):
                draw.text((30, 520 + 22 * i), line, font=font(14), fill=(95, 94, 90))
            page.paste(Image.open(work / "frames" / f"{f:04d}.png").convert("RGB"), (380, 60))
            page.save(work / "page" / f"{f:04d}.png")
        target = out / f"clip-{b}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(work / "page" / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(target)], check=True)
        print(f"wrote {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    one = steps.add_parser("export")
    one.add_argument("--ego-run", type=Path, default=Path("/ptmp/enazaren/nymeria/ego_runs/egoprimo/all-medium-v1"))
    one.add_argument("--head-run", type=Path, required=True)
    one.add_argument("--body-run", type=Path, required=True)
    one.add_argument("--ground", type=Path, default=Path("/ptmp/enazaren/gray/ground"))
    one.add_argument("--location", default="Loc_08")
    one.add_argument("--moments", type=int, default=8)
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--out", type=Path, required=True)
    two = steps.add_parser("render")
    two.add_argument("--exported", type=Path, required=True)
    two.add_argument("--static", type=Path, required=True)
    two.add_argument("--data", type=Path, required=True)
    two.add_argument("--out", type=Path, required=True)
    two.add_argument("--scene", default="scene.glb")
    two.add_argument("--only", type=int, nargs="+", default=None)
    two.add_argument("--captions", type=Path, default=None, help="texts.json of gray/hml.py, for the output of gray/place.py show")
    two.add_argument("--blender", default="blender")
    three = steps.add_parser("floor")
    three.add_argument("--exported", type=Path, required=True)
    three.add_argument("--out", type=Path, required=True)
    three.add_argument("--blender", default="blender")
    args = parser.parse_args()
    {"export": export, "render": render, "floor": floor}[args.step](args)


if __name__ == "__main__":
    main()
