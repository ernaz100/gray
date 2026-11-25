"""Blender side of gray/film.py: render a generated motion in a home's mesh scene. Run by film.py, not by hand:

    blender -b -P gray/film_blender.py -- <work folder>

The work folder holds scene.json (paths and numbers), verts.npy (frames x vertices x 3) and faces.npy of the body, in
Blender's frame (x, y, z up). It writes third/0000.png ... (the room from above and behind, walls see-through, the body
walking along the planned head path) and first/0000.png ... (the room from the planned camera poses, body hidden).

For gray/versus_scene.py: with "bodies" in scene.json (a list of {"verts": file, "colour": [r, g, b]}) several bodies walk in
the room, each with the trace of its pelvis on the floor; "path" may then be missing (no planned camera, no first/), and
"top_size" and "top_width" (m) ask for top/0000.png ..., the same from straight above.
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

work = Path(sys.argv[sys.argv.index("--") + 1])
spec = json.loads((work / "scene.json").read_text())
faces = np.load(work / "faces.npy")
shown = spec.get("bodies") or [{"verts": "verts.npy", "colour": [52, 120, 214]}]
all_verts = [np.load(work / one["verts"]) for one in shown]
verts = all_verts[0]


def srgb(red, green, blue):
    channel = lambda v: v / 255 / 12.92 if v / 255 <= 0.04045 else ((v / 255 + 0.055) / 1.055) ** 2.4  # noqa: E731
    return (channel(red), channel(green), channel(blue))


def material(name, color, metallic=0.0, roughness=0.55, emission=0.0):
    made = bpy.data.materials.new(name)
    made.use_nodes = True
    shader = made.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    if emission:
        shader.inputs["Emission Color"].default_value = (*color, 1.0)
        shader.inputs["Emission Strength"].default_value = emission
    return made


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=spec["scene"])

# the colours of the earlier project's renders of this scene (cursor-workspace/loc08/render_views.py)
PREFIXES = (
    ("floor", material("floor", srgb(214, 206, 190), roughness=0.7)), ("wall", material("wall", srgb(232, 224, 208), roughness=0.85)),
    ("stairs", material("stairs", srgb(176, 160, 140), roughness=0.6)), ("storage", material("storage", srgb(92, 36, 24), roughness=0.42)),
    ("refrigerator", material("fridge", srgb(158, 164, 170), metallic=0.9, roughness=0.28)), ("washerdryer", material("washer", srgb(210, 210, 208), metallic=0.7, roughness=0.3)),
    ("chair", material("chair", srgb(150, 104, 64), roughness=0.45)), ("table", material("table", srgb(72, 48, 36), roughness=0.4)),
    ("couch", material("couch", srgb(120, 128, 138), roughness=0.8)), ("bed", material("bed", srgb(230, 226, 216), roughness=0.8)),
    ("lamp", material("lamp", srgb(255, 236, 200), roughness=0.3, emission=4.0)), ("wallart", material("art", srgb(236, 232, 220), roughness=0.6)),
    ("sink", material("sink", srgb(180, 184, 188), metallic=0.8, roughness=0.25)), ("screen", material("screen", srgb(28, 30, 34), roughness=0.3)),
    ("mirror", material("mirror", srgb(220, 224, 228), metallic=0.9, roughness=0.08)), ("plant", material("plant", srgb(70, 110, 58), roughness=0.7)),
    ("toilet", material("toilet", srgb(236, 236, 232), roughness=0.25)), ("anything", material("anything", srgb(150, 140, 128), roughness=0.6)),
)
walls = []
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    chosen = max((p for p in PREFIXES if obj.name.startswith(p[0])), key=lambda p: len(p[0]), default=None)
    if chosen is not None:
        obj.data.materials.clear()
        obj.data.materials.append(chosen[1])
        if chosen[0] in ("wall", "wallart", "mirror", "screen"):
            walls.append(obj)
stone = material("counter", srgb(198, 180, 150), roughness=0.32)
for obj in bpy.data.objects:  # cabinet meshes include the counter: faces that point up become stone
    if obj.type == "MESH" and obj.name.startswith("storage"):
        obj.data.materials.append(stone)
        turn = obj.matrix_world.to_3x3()
        for poly in obj.data.polygons:
            if (turn @ poly.normal).normalized().z > 0.72:
                poly.material_index = 1

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.image_settings.file_format = "PNG"
scene.eevee.taa_render_samples = spec.get("samples", 16)
world = bpy.data.worlds.new("Daylight")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (*srgb(236, 230, 218), 1.0)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.4
bpy.ops.object.light_add(type="SUN")
sun = bpy.context.object
sun.data.energy = 4.0
sun.data.color = srgb(255, 244, 220)
sun.rotation_euler = Vector((-0.5, -0.3, -1.0)).to_track_quat("-Z", "Y").to_euler()
centre = Vector(spec["centre"])
bpy.ops.object.light_add(type="AREA", location=(centre.x, centre.y, centre.z + 2.3))
bpy.context.object.data.energy = 250
bpy.context.object.data.size = 5.0

# the bodies
meshes, people = [], []
for number, (one, own) in enumerate(zip(shown, all_verts)):
    mesh = bpy.data.meshes.new(f"body {number}")
    mesh.from_pydata(own[0].tolist(), [], faces.tolist())
    for poly in mesh.polygons:
        poly.use_smooth = True
    mesh.materials.append(material(f"body {number}", srgb(*one["colour"]), roughness=0.5))
    body = bpy.data.objects.new(f"body {number}", mesh)
    scene.collection.objects.link(body)
    meshes.append(mesh)
    people.append(body)


def pose(f: int) -> None:
    for mesh, own in zip(meshes, all_verts):
        mesh.vertices.foreach_set("co", own[f].reshape(-1))
        mesh.update()


def tube(name, points, colour, radius, emission=1.5):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    line = curve.splines.new("POLY")
    line.points.add(len(points) - 1)
    for point, at in zip(line.points, points):
        point.co = (*at, 1.0)
    curve.materials.append(material(name, colour, emission=emission))
    made = bpy.data.objects.new(name, curve)
    scene.collection.objects.link(made)
    return made


drawn = []
if "path" in spec:
    path = np.array(spec["path"])
    plan = tube("plan", path.tolist(), srgb(235, 104, 52), 0.02)
    drawn.append(tube("plan on the floor", [(x, y, spec["floor"] + 0.02) for x, y, _z in path], srgb(235, 104, 52), 0.012, emission=0.6))
if "bodies" in spec:
    for number, (one, own) in enumerate(zip(shown, all_verts)):
        middle = own.mean(1)[::3]
        drawn.append(tube(f"trace {number}", [(x, y, spec["floor"] + 0.02 + 0.004 * number) for x, y, _z in middle], srgb(*one["colour"]), 0.018, emission=0.8))

camera_data = bpy.data.cameras.new("camera")
camera = bpy.data.objects.new("camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera_data.clip_start, camera_data.clip_end = 0.05, 60.0


def see_through(on: bool) -> None:
    for obj in walls:
        for made in obj.data.materials:
            made.surface_render_method = "BLENDED" if on else "DITHERED"
            made.use_backface_culling = on
            made.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = spec.get("wall_alpha", 0.22) if on else 1.0


# 1. from above and behind; what lies above this storey (the next floor, its furniture) would hide the room and is left out
see_through(True)
above = [obj for obj in bpy.data.objects if obj.type == "MESH" and obj not in people
         and min((obj.matrix_world @ Vector(corner)).z for corner in obj.bound_box) > spec["floor"] + 2.2]
for obj in above:
    obj.hide_render = True
scene.render.resolution_x, scene.render.resolution_y = spec["third_size"]
camera_data.lens = 30
camera.location = Vector(spec["eye"])
camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
(work / "third").mkdir(exist_ok=True)
for f in range(len(verts)):
    pose(f)
    scene.render.filepath = str(work / "third" / f"{f:04d}.png")
    bpy.ops.render.render(write_still=True)

if "top_size" in spec:  # the same from straight above
    scene.render.resolution_x, scene.render.resolution_y = spec["top_size"]
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = spec["top_width"]
    camera.location = Vector((centre.x, centre.y, spec["floor"] + 2.1))
    camera.rotation_euler = (0.0, 0.0, 0.0)
    (work / "top").mkdir(exist_ok=True)
    for f in range(len(verts)):
        pose(f)
        scene.render.filepath = str(work / "top" / f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    camera_data.type = "PERSP"
if "path" not in spec:
    print("rendered", len(verts), "frames")
    raise SystemExit

# 2. from the planned camera
see_through(False)
for obj in above:
    obj.hide_render = False
for body in people:
    body.hide_render = True
plan.hide_render = True
scene.render.resolution_x, scene.render.resolution_y = spec["first_size"]
camera_data.sensor_fit = "HORIZONTAL"
camera_data.sensor_width = 36.0
camera_data.lens = 18.0 / np.tan(np.radians(spec["view"]) / 2)
(work / "first").mkdir(exist_ok=True)
for f, (at, turn) in enumerate(zip(path, spec["axes"])):
    camera.matrix_world = Matrix.Translation(Vector(at)) @ Matrix(np.array(turn).reshape(3, 3).tolist()).to_4x4()
    scene.render.filepath = str(work / "first" / f"{f:04d}.png")
    bpy.ops.render.render(write_still=True)
print("rendered", len(verts), "and", len(path), "frames")
