"""Blender side of gray/versus.py: several bodies doing the same moment side by side on a checkered floor.

    blender -b -P gray/versus_blender.py -- <work folder>

The work folder holds scene.json (colours, camera) and verts-<i>.npy (frames x vertices x 3, Blender's frame) per body, and
faces.npy. It writes frames/0000.png ...
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

work = Path(sys.argv[sys.argv.index("--") + 1])
spec = json.loads((work / "scene.json").read_text())
faces = np.load(work / "faces.npy")
bodies = [np.load(work / f"verts-{i}.npy") for i in range(len(spec["colours"]))]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.image_settings.file_format = "PNG"
scene.render.resolution_x, scene.render.resolution_y = spec["size"]
scene.eevee.taa_render_samples = 16
world = bpy.data.worlds.new("light")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.93, 0.93, 0.92, 1.0)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.7

# a floor of 0.5 m squares, so that a sliding foot shows
floor = bpy.data.materials.new("floor")
floor.use_nodes = True
nodes, links = floor.node_tree.nodes, floor.node_tree.links
checker = nodes.new("ShaderNodeTexChecker")
checker.inputs["Scale"].default_value = 80.0  # the plane is 40 m wide in object coordinates of 1: squares of 0.5 m
checker.inputs["Color1"].default_value = (0.78, 0.78, 0.76, 1.0)
checker.inputs["Color2"].default_value = (0.60, 0.60, 0.58, 1.0)
place = nodes.new("ShaderNodeTexCoord")
links.new(place.outputs["Object"], checker.inputs["Vector"])
links.new(checker.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
centre = Vector(spec["centre"])
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(centre.x, centre.y, 0.0))
ground = bpy.context.object
ground.scale = (40, 40, 1)
ground.data.materials.append(floor)

bpy.ops.object.light_add(type="SUN")
sun = bpy.context.object
sun.data.energy = 3.5
sun.rotation_euler = Vector((-0.4, -0.3, -1.0)).to_track_quat("-Z", "Y").to_euler()

meshes = []
for i, (verts, colour) in enumerate(zip(bodies, spec["colours"])):
    mesh = bpy.data.meshes.new(f"body{i}")
    mesh.from_pydata(verts[0].tolist(), [], faces.tolist())
    for poly in mesh.polygons:
        poly.use_smooth = True
    made = bpy.data.materials.new(f"body{i}")
    made.use_nodes = True
    made.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*colour, 1.0)
    made.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.55
    mesh.materials.append(made)
    scene.collection.objects.link(bpy.data.objects.new(f"body{i}", mesh))
    meshes.append(mesh)

camera_data = bpy.data.cameras.new("camera")
camera = bpy.data.objects.new("camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera_data.lens = spec.get("lens", 35)
camera.location = Vector(spec["eye"])
camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()

(work / "frames").mkdir(exist_ok=True)
for f in range(len(bodies[0])):
    for mesh, verts in zip(meshes, bodies):
        mesh.vertices.foreach_set("co", verts[f].reshape(-1))
        mesh.update()
    scene.render.filepath = str(work / "frames" / f"{f:04d}.png")
    bpy.ops.render.render(write_still=True)
print("rendered", len(bodies[0]), "frames")
