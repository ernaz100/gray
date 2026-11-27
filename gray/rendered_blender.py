"""Blender side of gray/rendered.py: render a home's mesh scene from given camera poses. Run by rendered.py, not by hand:

    blender -b -P gray/rendered_blender.py -- <work folder>

The work folder holds scene.json: "scene" (the GLB), "at" (N x 3 camera positions) and "axes" (N x 9, Blender camera
axes in the columns), both in Blender's frame, "view" (degrees), "size" (pixels), "names" (N file names). The GLB's own
materials are kept (the painted scene carries the colours seen by the head cameras); they are shown as they are, without
shading, so that those colours are not lit a second time. Writes <work>/<name> for each pose.
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

work = Path(sys.argv[sys.argv.index("--") + 1])
spec = json.loads((work / "scene.json").read_text())
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=spec["scene"])
kinds = {}
for obj in bpy.data.objects:
    if obj.type == "MESH":
        for slot in obj.material_slots:
            tree = slot.material.node_tree if slot.material and slot.material.use_nodes else None
            for node in (tree.nodes if tree else []):
                kinds[node.type] = kinds.get(node.type, 0) + 1
print("material nodes:", kinds, "meshes:", sum(o.type == "MESH" for o in bpy.data.objects), "colour layers:",
      sum(len(o.data.color_attributes) > 0 for o in bpy.data.objects if o.type == "MESH"))

for material in bpy.data.materials:  # show the painted colours as they are: no shading, no see-through, and read as sRGB (they were painted from camera frames)
    for name, value in (("blend_method", "OPAQUE"), ("surface_render_method", "DITHERED")):
        if hasattr(material, name):
            try:
                setattr(material, name, value)
            except (TypeError, AttributeError):
                pass
    if not material.use_nodes:
        continue
    tree = material.node_tree
    shader = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if shader is None or output is None or not shader.inputs["Base Color"].links:
        continue
    source = shader.inputs["Base Color"].links[0].from_socket
    glow = tree.nodes.new("ShaderNodeEmission")
    if source.node.type == "ATTRIBUTE" and spec.get("gamma", 2.2) != 1:
        gamma = tree.nodes.new("ShaderNodeGamma")
        gamma.inputs["Gamma"].default_value = spec.get("gamma", 2.2)
        tree.links.new(source, gamma.inputs["Color"])
        source = gamma.outputs["Color"]
    tree.links.new(source, glow.inputs["Color"])
    tree.links.new(glow.outputs["Emission"], output.inputs["Surface"])

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.image_settings.file_format = "PNG"
scene.eevee.taa_render_samples = spec.get("samples", 8)
scene.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("Even")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
world.node_tree.nodes["Background"].inputs[1].default_value = spec.get("light", 0.0)

camera_data = bpy.data.cameras.new("camera")
camera = bpy.data.objects.new("camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera_data.clip_start, camera_data.clip_end = 0.05, 60.0
camera_data.sensor_fit = "HORIZONTAL"
camera_data.sensor_width = 36.0
camera_data.lens = 18.0 / np.tan(np.radians(spec["view"]) / 2)
scene.render.resolution_x = scene.render.resolution_y = spec["size"]
for name, at, turn in zip(spec["names"], spec["at"], spec["axes"]):
    camera.matrix_world = Matrix.Translation(Vector(at)) @ Matrix(np.array(turn).reshape(3, 3).tolist()).to_4x4()
    scene.render.filepath = str(work / name)
    bpy.ops.render.render(write_still=True)
print("rendered", len(spec["names"]), "frames")
