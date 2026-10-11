"""Builds low-poly desert props in Blender and exports them as FBX for Roblox Studio.

Run with Blender's Python module:  python make_desert_props.py
Outputs go to models/fbx/ (meshes) and models/previews/ (renders).
"""
import math
import os
import random

import bpy  # must come first: it makes bmesh and mathutils importable
import bmesh
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
FBX_DIR = os.path.join(HERE, "fbx")
PREVIEW_DIR = os.path.join(HERE, "previews")
os.makedirs(FBX_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# 4x4 color palette baked into one small texture; every face is UV-mapped to one swatch.
PALETTE = {
    "sand": (232, 200, 146), "sandstone": (207, 160, 104), "sandstone_dark": (172, 121, 76),
    "rock_shadow": (138, 94, 60), "cactus": (92, 142, 72), "cactus_dark": (62, 106, 56),
    "flower": (232, 110, 150), "turquoise": (38, 186, 188), "gold": (226, 180, 62),
    "dark_stone": (92, 76, 66), "spine": (240, 232, 205),
}
SWATCH = {name: i for i, name in enumerate(PALETTE)}
GRID = 4
PALETTE_PNG = os.path.join(FBX_DIR, "DesertPalette.png")


def make_palette_material():
    size = 64
    cell = size // GRID
    img = bpy.data.images.new("DesertPalette", size, size)
    pixels = [0.0] * (size * size * 4)
    colors = list(PALETTE.values())
    for y in range(size):
        for x in range(size):
            idx = (y // cell) * GRID + (x // cell)
            r, g, b = colors[idx] if idx < len(colors) else (255, 0, 255)
            o = (y * size + x) * 4
            pixels[o:o + 4] = [r / 255, g / 255, b / 255, 1.0]
    img.pixels = pixels
    img.filepath_raw = PALETTE_PNG
    img.file_format = "PNG"
    img.save()

    mat = bpy.data.materials.new("DesertPalette")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 1.0
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def swatch_uv(name):
    i = SWATCH[name]
    return ((i % GRID + 0.5) / GRID, (i // GRID + 0.5) / GRID)


def paint(bm, faces, name):
    uv_layer = bm.loops.layers.uv.verify()
    u, v = swatch_uv(name)
    for f in faces:
        for loop in f.loops:
            loop[uv_layer].uv = (u, v)


def new_faces(bm, before):
    return [f for f in bm.faces if f.index == -1 or f not in before]


def add_cylinder(bm, radius, depth, segments, matrix, radius2=None):
    before = set(bm.faces)
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius,
                          radius2=radius if radius2 is None else radius2, depth=depth, matrix=matrix)
    return [f for f in bm.faces if f not in before]


def add_sphere(bm, radius, matrix, subdiv=1):
    before = set(bm.faces)
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=radius, matrix=matrix)
    return [f for f in bm.faces if f not in before]


def add_box(bm, size, center):
    before = set(bm.faces)
    m = Matrix.Translation(center) @ Matrix.Diagonal((*size, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=m)
    return [f for f in bm.faces if f not in before]


def finish(bm, name, mat, collection):
    # Origin at the bottom center so the model sits on the ground when placed.
    min_z = min(v.co.z for v in bm.verts)
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    offset = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, min_z))
    bmesh.ops.translate(bm, verts=bm.verts, vec=-offset)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = False
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def make_rock(name, seed, scale, mat, col):
    rng = random.Random(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    for v in bm.verts:
        v.co *= rng.uniform(0.7, 1.25)
        v.co = Vector((v.co.x * scale[0], v.co.y * scale[1], v.co.z * scale[2]))
    floor = -0.35 * scale[2]
    for v in bm.verts:
        v.co.z = max(v.co.z, floor)
    for f in bm.faces:
        if f.normal.z < -0.9:
            paint(bm, [f], "rock_shadow")
        elif f.normal.z > 0.55:
            paint(bm, [f], "sand")  # sun-bleached tops
        else:
            paint(bm, [f], "sandstone_dark" if rng.random() < 0.35 else "sandstone")
    return finish(bm, name, mat, col)


def make_cactus(name, mat, col):
    bm = bmesh.new()
    seg = 8
    body = []
    # Trunk + rounded top
    body += add_cylinder(bm, 0.45, 4.0, seg, Matrix.Translation((0, 0, 2.0)))
    body += add_sphere(bm, 0.45, Matrix.Translation((0, 0, 4.0)), subdiv=2)
    # Right arm: out, then up
    body += add_cylinder(bm, 0.3, 1.0, seg, Matrix.Translation((0.7, 0, 1.8)) @ Matrix.Rotation(math.pi / 2, 4, "Y"))
    body += add_cylinder(bm, 0.3, 1.4, seg, Matrix.Translation((1.15, 0, 2.5)))
    body += add_sphere(bm, 0.3, Matrix.Translation((1.15, 0, 3.2)))
    # Left arm: lower and shorter
    body += add_cylinder(bm, 0.28, 0.9, seg, Matrix.Translation((-0.65, 0, 1.3)) @ Matrix.Rotation(math.pi / 2, 4, "Y"))
    body += add_cylinder(bm, 0.28, 1.0, seg, Matrix.Translation((-1.05, 0, 1.75)))
    body += add_sphere(bm, 0.28, Matrix.Translation((-1.05, 0, 2.25)))
    for f in body:
        n = f.normal
        side = int(((math.atan2(n.y, n.x) + math.pi) / (2 * math.pi)) * seg * 2) % 2
        paint(bm, [f], "cactus" if side else "cactus_dark")
    paint(bm, add_sphere(bm, 0.16, Matrix.Translation((0.05, 0.1, 4.42))), "flower")
    paint(bm, add_sphere(bm, 0.12, Matrix.Translation((1.15, -0.05, 3.48))), "flower")
    return finish(bm, name, mat, col)


def broken_shaft(bm, radius, height, center, rng, segments=12):
    faces = add_cylinder(bm, radius, height, segments, Matrix.Translation((center[0], center[1], center[2] + height / 2)))
    top = center[2] + height
    for v in {v for f in faces for v in f.verts}:
        if abs(v.co.z - top) < 1e-4:
            v.co.z -= rng.uniform(0.0, height * 0.25)
    for f in faces:
        if f.normal.z > 0.5:
            paint(bm, [f], "sandstone_dark")
        else:
            n = f.normal
            idx = int(((math.atan2(n.y, n.x) + math.pi) / (2 * math.pi)) * segments) % 2
            paint(bm, [f], "sandstone" if idx else "sand")
    return faces


def make_pillar(name, mat, col):
    rng = random.Random(7)
    bm = bmesh.new()
    paint(bm, add_box(bm, (2.4, 2.4, 0.5), (0, 0, 0.25)), "sandstone_dark")
    paint(bm, add_box(bm, (2.0, 2.0, 0.3), (0, 0, 0.65)), "sandstone")
    broken_shaft(bm, 0.85, 4.6, (0, 0, 0.8), rng)
    # Fallen chunk lying next to the base
    chunk = add_cylinder(bm, 0.85, 1.4, 12,
                         Matrix.Translation((2.1, 0.9, 0.85)) @ Matrix.Rotation(math.pi / 2, 4, "X") @ Matrix.Rotation(0.4, 4, "Z"))
    for f in chunk:
        paint(bm, [f], "sandstone_dark" if abs(f.normal.y) > 0.7 else "sandstone")
    return finish(bm, name, mat, col)


def make_arena(name, mat, col):
    rng = random.Random(42)
    bm = bmesh.new()
    tiers = [(28.0, "sandstone_dark"), (24.0, "sandstone"), (20.0, "sandstone_dark")]
    for i, (r, side) in enumerate(tiers):
        faces = add_cylinder(bm, r, 1.0, 48, Matrix.Translation((0, 0, i + 0.5)))
        for f in faces:
            paint(bm, [f], "sand" if f.normal.z > 0.5 else side)
    # Summoning circle on top
    paint(bm, add_cylinder(bm, 12.0, 0.08, 48, Matrix.Translation((0, 0, 3.04))), "turquoise")
    paint(bm, add_cylinder(bm, 11.0, 0.08, 48, Matrix.Translation((0, 0, 3.08))), "sand")
    paint(bm, add_cylinder(bm, 6.0, 0.08, 6, Matrix.Translation((0, 0, 3.12))), "gold")
    paint(bm, add_cylinder(bm, 5.2, 0.08, 6, Matrix.Translation((0, 0, 3.16)) @ Matrix.Rotation(math.pi / 6, 4, "Z")), "sand")
    paint(bm, add_cylinder(bm, 1.6, 0.08, 24, Matrix.Translation((0, 0, 3.2))), "turquoise")
    # Ring of broken pillars on the middle tier
    for k in range(8):
        a = k / 8 * 2 * math.pi + math.pi / 8
        x, y = math.cos(a) * 22.0, math.sin(a) * 22.0
        paint(bm, add_box(bm, (2.2, 2.2, 0.4), (x, y, 2.2)), "dark_stone")
        broken_shaft(bm, 0.8, rng.uniform(3.0, 7.0), (x, y, 2.4), rng)
    return finish(bm, name, mat, col)


def export(obj, filename):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(
        filepath=os.path.join(FBX_DIR, filename), use_selection=True, object_types={"MESH"},
        mesh_smooth_type="FACE", path_mode="COPY", embed_textures=True,
        axis_forward="-Z", axis_up="Y", apply_scale_options="FBX_SCALE_ALL",
    )


def render_preview(objs, filename):
    scene = bpy.context.scene
    for o in scene.collection.all_objects:
        o.hide_render = o.type == "MESH" and o not in objs
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center, radius = (lo + hi) / 2, (hi - lo).length / 2
    cam = scene.camera
    direction = Vector((1.0, -1.3, 0.75)).normalized()
    cam.location = center + direction * radius / math.sin(math.radians(20)) * 1.05
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = os.path.join(PREVIEW_DIR, filename)
    bpy.ops.render.render(write_still=True)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    col = scene.collection
    mat = make_palette_material()

    rocks = [
        make_rock("DesertRock1", 1, (2.6, 2.0, 1.6), mat, col),
        make_rock("DesertRock2", 2, (1.6, 1.4, 2.4), mat, col),
        make_rock("DesertRock3", 3, (3.6, 2.4, 1.1), mat, col),
    ]
    for i, r in enumerate(rocks):
        r.location.x = (i - 1) * 7.0
    cactus = make_cactus("Cactus", mat, col)
    pillar = make_pillar("BrokenPillar", mat, col)
    arena = make_arena("DjinnArena", mat, col)

    for obj in rocks + [cactus, pillar, arena]:
        loc = obj.location.copy()
        obj.location = (0, 0, 0)
        export(obj, obj.name + ".fbx")
        obj.location = loc

    # Preview renders (Cycles on CPU)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("Sky")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.62, 0.78, 0.95, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    scene.world = world
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.5
    sun.rotation_euler = (math.radians(50), 0, math.radians(35))
    col.objects.link(sun)
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.angle = math.radians(40)
    cam.data.clip_end = 1000
    col.objects.link(cam)
    scene.camera = cam

    cactus.location.x = 30
    pillar.location.x = 45
    arena.location.x = 120
    render_preview(rocks, "rocks.png")
    render_preview([cactus], "cactus.png")
    render_preview([pillar], "pillar.png")
    render_preview([arena], "arena.png")


if __name__ == "__main__":
    main()
