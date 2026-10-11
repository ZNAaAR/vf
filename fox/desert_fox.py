"""
DESERT FOX - procedural creature generator for Blender (Cycles)
===============================================================

Builds a big-eared desert fox (Rueppell's / bat-eared fox type) from code:

  * one continuous anatomical body (signed-distance-field sculpt -> surface-nets mesh)
  * cupped, thick-rimmed ears, layered eyes (eyeball + iris shader + glass cornea + lids),
    nose with nostrils, toes, claws, paw pads, separate tail
  * strand fur (Blender "hair curves"): underfur + guard hairs, anatomical flow field,
    per-region length/colour, clumping
  * studio lighting, 5 presentation cameras, Cycles render setup
  * a separate Roblox asset: fur-silhouette low-poly mesh, baked colour/normal/roughness
    maps, a 26-bone rig with skin weights, FBX export

Built and tested on Blender 5.2 LTS. Needs nothing beyond stock Blender
(it only uses bpy, bmesh, mathutils and the bundled numpy).

HOW TO RUN
  In Blender:  Scripting workspace -> Open -> desert_fox.py -> Run Script.
               (Edit the SETTINGS block below first if you want renders or another quality.)
  Headless:    blender --background --python desert_fox.py -- --quality high --render
  Options after "--":
      --quality draft|medium|high   fur density / mesh resolution / samples (default: medium)
      --render                      render the 5 presentation views + a Roblox preview
      --views front_3q,side         only render some views
      --no-roblox                   skip the Roblox asset
      --out <folder>                output folder (default: ~/DesertFox_Output)
"""

import math
import os
import sys
import time

import bpy  # noqa: E402  (must be imported before bmesh/mathutils when used as a module)
import bmesh
import numpy as np
from mathutils import Matrix, Vector, kdtree

# =============================================================================
# SETTINGS (used when you press "Run Script" inside Blender)
# =============================================================================

SETTINGS = {
    "quality": "medium",
    "render": False,          # True = render all presentation views (slow on CPU)
    "views": None,            # e.g. ["front_3q", "face_closeup"]
    "build_roblox": True,
    "out_dir": os.path.join(os.path.expanduser("~"), "DesertFox_Output"),
}

QUALITY = {
    #           SDF voxel   fur density  samples  resolution     curve shape
    "draft":  dict(voxel=0.0024, fur=0.22, samples=24, res=(960, 720), shape="RIBBONS"),
    "medium": dict(voxel=0.0017, fur=0.55, samples=64, res=(1600, 1200), shape="THICK"),
    "high":   dict(voxel=0.0013, fur=1.00, samples=128, res=(2048, 1536), shape="THICK"),
}

SEED = 20240611
ROBLOX_STUDS_PER_METER = 3.57      # Roblox physics convention: 1 stud = 0.28 m
ROBLOX_BODY_TRIS = 9000            # triangle budget for the main fur-silhouette mesh
ROBLOX_TEXTURE = 1024              # Roblox caps textures at 1024 px


def _parse_args():
    cfg = dict(SETTINGS)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--quality":
            cfg["quality"] = argv[i + 1]
            i += 1
        elif a == "--render":
            cfg["render"] = True
        elif a == "--views":
            cfg["views"] = argv[i + 1].split(",")
            i += 1
        elif a == "--no-roblox":
            cfg["build_roblox"] = False
        elif a == "--out":
            cfg["out_dir"] = os.path.abspath(argv[i + 1])
            i += 1
        i += 1
    if cfg["quality"] not in QUALITY:
        raise ValueError(f"Unknown quality '{cfg['quality']}', use one of {list(QUALITY)}")
    return cfg


def log(msg):
    print(f"[DesertFox] {msg}", flush=True)


class BuildError(RuntimeError):
    pass


# =============================================================================
# NUMPY MATH HELPERS
# =============================================================================

def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761 & 0x7FFFFFFF)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0xFFFFFF)


def value_noise(P, freq, seed=0):
    """Smooth 3D value noise in [-1, 1]."""
    Q = P * freq
    i = np.floor(Q).astype(np.int64)
    f = Q - i
    u = f * f * (3.0 - 2.0 * f)
    out = np.zeros(len(P))
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1.0 - u[:, 0]
        for dy in (0, 1):
            wy = u[:, 1] if dy else 1.0 - u[:, 1]
            for dz in (0, 1):
                wz = u[:, 2] if dz else 1.0 - u[:, 2]
                out += wx * wy * wz * _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out * 2.0 - 1.0


def fbm(P, freq, octaves=3, seed=0, gain=0.5):
    total, amp, norm = np.zeros(len(P)), 1.0, 0.0
    for o in range(octaves):
        total += amp * value_noise(P, freq * (2.0 ** o), seed + 17 * o)
        norm += amp
        amp *= gain
    return total / norm


# =============================================================================
# SIGNED DISTANCE FIELD PRIMITIVES (numpy, vectorised over N points)
# =============================================================================

def sd_sphere(P, c, r):
    return np.linalg.norm(P - np.asarray(c), axis=1) - r


def sd_ellipsoid(P, c, r, rot=None):
    """Approximate distance to an ellipsoid with radii r; rot = 3x3 (columns = local axes)."""
    q = P - np.asarray(c, dtype=np.float64)
    if rot is not None:
        q = q @ rot
    r = np.asarray(r, dtype=np.float64)
    k0 = np.linalg.norm(q / r, axis=1)
    k1 = np.linalg.norm(q / (r * r), axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_round_cone(P, a, b, r1, r2):
    """Exact distance to a capsule whose radius goes from r1 (at a) to r2 (at b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = P - a
    y = pa @ ba
    z = y - l2
    xv = pa * l2 - np.outer(y, ba)
    x2 = np.einsum("ij,ij->i", xv, xv)
    y2 = y * y * l2
    z2 = z * z * l2
    k = math.copysign(1.0, rr) * rr * rr * x2 if rr != 0 else np.zeros_like(x2)
    d_b = np.sqrt(x2 + z2) * il2 - r2
    d_a = np.sqrt(x2 + y2) * il2 - r1
    d_m = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d_b, np.where(np.sign(y) * a2 * y2 < k, d_a, d_m))


def sd_round_cone_scaled(P, a, b, r1, r2, sx=1.0, sz=1.0):
    """Round cone (axis roughly along Y) with an elliptical cross-section."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ba = b - a
    t = np.clip(((P - a) @ ba) / float(ba @ ba), 0.0, 1.0)
    C = a + np.outer(t, ba)
    o = (P - C) / np.array([sx, 1.0, sz])
    return sd_round_cone(C + o, a, b, r1, r2) * min(sx, sz)


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def ssub(a, b, k):
    """Smooth subtraction: a minus b."""
    return smax(a, -b, k)


# =============================================================================
# ANATOMY (all measurements in metres, fox faces -Y, ground is z = 0, left = +X)
# =============================================================================

ANAT = {}  # filled by place_features(): eye centres, nose, ear roots, mouth line...

CREATURE_SCALE = 4.0            # fantasy boss: ~4x the size of a real fox (applied to the finished scene + export)


JAW_OPEN = math.radians(22.0)   # snarling, open jaw


JAW_HINGE = np.array([0.0, -0.262, 0.366])


NECK_PIVOT = np.array([0.0, -0.135, 0.280])


ATLAS_PIVOT = np.array([0.0, -0.226, 0.362])


NECK_DROP = math.radians(16.0)  # head carried low (stalking / threatening)


HEAD_LIFT = math.radians(-13.0)  # ...but tilted up so the eyes glare forward


PAW_SCALE = 1.22                # oversized paws


LEG_THICK = 1.16                # heavier legs


TOE_OFFSETS = [(0.0052, -0.0165, -0.0020), (-0.0052, -0.0165, -0.0020),
               (0.0118, -0.0105, -0.0026), (-0.0118, -0.0105, -0.0026)]


MOUTH_CAVITY = dict(c=(0.0, -0.306, 0.3585), r=(0.0102, 0.040, 0.0068))


def rot_about(P, pivot, R):
    """Rotates row-vector points P about pivot with the 3x3 rotation R."""
    return (P - pivot) @ R.T + pivot


def jaw_rest(P):
    """Open-jaw coordinates -> closed-jaw (rest) coordinates."""
    return rot_about(P, JAW_HINGE, rot_x(-JAW_OPEN))


def jaw_pose(P):
    """Closed-jaw (rest) coordinates -> open-jaw coordinates."""
    return rot_about(P, JAW_HINGE, rot_x(JAW_OPEN))


def mouth_cavity_sdf(Pm):
    return sd_ellipsoid(Pm, MOUTH_CAVITY["c"], MOUTH_CAVITY["r"], rot_x(JAW_OPEN * 0.45))


def torso_sdf(P):
    """Torso and neck (P mirrored to x >= 0). Heavy, front-loaded build of a predator boss."""
    d = sd_ellipsoid(P, (0, -0.075, 0.243), (0.058, 0.106, 0.068))                    # ribcage
    d = smin(d, sd_ellipsoid(P, (0, -0.150, 0.226), (0.050, 0.050, 0.058)), 0.030)    # deep chest
    d = smin(d, sd_round_cone(P, (0, 0.000, 0.250), (0, 0.105, 0.254), 0.043, 0.044), 0.040)  # loin
    d = smin(d, sd_ellipsoid(P, (0, 0.135, 0.256), (0.053, 0.074, 0.054)), 0.030)     # pelvis / croup
    d = smin(d, sd_round_cone(P, (0, -0.120, 0.279), (0, 0.170, 0.263), 0.032, 0.027), 0.030)  # back muscles
    d = smin(d, sd_ellipsoid(P, (0, -0.118, 0.296), (0.034, 0.055, 0.024)), 0.020)    # muscular withers hump
    d = smin(d, sd_ellipsoid(P, (0.040, -0.118, 0.248), (0.030, 0.048, 0.066), rot_x(-0.41)), 0.020)  # scapula
    d = smin(d, sd_ellipsoid(P, (0.036, 0.146, 0.222), (0.030, 0.060, 0.068), rot_x(-0.56)), 0.024)  # haunch
    d = smin(d, sd_round_cone(P, (0, -0.132, 0.272), (0, -0.228, 0.366), 0.055, 0.040), 0.035)  # thick neck
    d = smin(d, sd_ellipsoid(P, (0, -0.203, 0.306), (0.037, 0.042, 0.036)), 0.020)    # throat
    return d


def skull_sdf(P):
    """Upper head without the lower jaw (P mirrored to x >= 0)."""
    d = sd_ellipsoid(P, (0, -0.258, 0.395), (0.041, 0.046, 0.034))                    # broad cranium
    d = smin(d, sd_ellipsoid(P, (0, -0.232, 0.384), (0.035, 0.031, 0.034)), 0.015)    # occiput
    d = smin(d, sd_ellipsoid(P, (0, -0.291, 0.398), (0.026, 0.024, 0.021)), 0.012)    # frontal bone
    d = smin(d, sd_ellipsoid(P, (0.028, -0.272, 0.376), (0.016, 0.027, 0.017)), 0.010)  # zygomatic arch
    d = smin(d, sd_ellipsoid(P, (0.016, -0.301, 0.4045), (0.0145, 0.013, 0.0085)), 0.006)  # heavy brow ridge
    d = smin(d, sd_ellipsoid(P, (0.0075, -0.306, 0.4015), (0.0075, 0.0065, 0.0050)), 0.004)  # frowning inner brow
    d = smin(d, sd_round_cone_scaled(P, (0, -0.300, 0.382), (0, -0.356, 0.370), 0.0215, 0.0108, 0.90, 0.95), 0.012)
    d = smin(d, sd_ellipsoid(P, (0, -0.355, 0.3695), (0.0094, 0.0088, 0.0085)), 0.006)  # snout tip
    d = smin(d, sd_ellipsoid(P, (0.021, -0.258, 0.362), (0.013, 0.020, 0.014)), 0.012)  # jaw muscles
    # snarl: folded skin across the top of the muzzle
    for y in (-0.341, -0.334, -0.327):
        t = (y + 0.300) / -0.056
        top = 0.382 - 0.012 * t + (0.0215 - 0.0107 * t) * 0.95 - 0.0007
        d = smin(d, sd_round_cone(P, (0, y, top), (0.0105, y + 0.002, top - 0.0042), 0.0013, 0.0011), 0.0016)
    # upper lip pulled back from the canines
    d = ssub(d, sd_ellipsoid(P, (0.0098, -0.338, 0.3612), (0.0045, 0.0080, 0.0044)), 0.002)
    return d


def jaw_sdf(Pj):
    """Lower jaw in closed-jaw coordinates (mirrored)."""
    d = sd_round_cone_scaled(Pj, (0, -0.282, 0.3565), (0, -0.343, 0.3548), 0.0128, 0.0066, 1.32, 0.85)
    d = smin(d, sd_ellipsoid(Pj, (0, -0.337, 0.3515), (0.0072, 0.0088, 0.0050)), 0.004)  # chin
    return d


def closed_head_sdf(P):
    return body_base_sdf(P, jaw_open=False)


def jaw_sdf_world_closed(P):
    Pm = P.copy()
    Pm[:, 0] = np.abs(Pm[:, 0])
    return ssub(jaw_sdf(Pm), mouth_cavity_sdf(jaw_pose(Pm)), 0.003)


TAIL_POINTS = np.array([
    (0.000, 0.190, 0.264),
    (0.000, 0.246, 0.255),
    (0.003, 0.298, 0.232),
    (0.008, 0.340, 0.197),
    (0.012, 0.370, 0.155),
    (0.015, 0.386, 0.116),
])


TAIL_RADII = [0.0230, 0.0212, 0.0190, 0.0155, 0.0115, 0.0060]


FRONT_LEG = dict(shoulder=(0.044, -0.136, 0.218), elbow=(0.040, -0.104, 0.150),
                 wrist=(0.036, -0.119, 0.046), pastern=(0.036, -0.128, 0.019),
                 paw=(0.036, -0.137, 0.0105))
HIND_LEG = dict(hip=(0.045, 0.128, 0.238), stifle=(0.049, 0.088, 0.148),
                hock=(0.045, 0.166, 0.074), meta=(0.043, 0.151, 0.022),
                paw=(0.043, 0.141, 0.0105))
STANCE = {"front": {1: -0.010, -1: 0.006}, "hind": {1: 0.008, -1: -0.006}}


def _side(p, s, dy=0.0):
    return (p[0] * s, p[1] + dy, p[2])


def paw_sdf(P, c, s, hind=False):
    c = np.asarray(c, dtype=np.float64)
    r = np.array((0.0135, 0.0185, 0.0103) if not hind else (0.0130, 0.0190, 0.0100)) * PAW_SCALE
    d = sd_ellipsoid(P, c, r)
    for t in TOE_OFFSETS:
        d = smin(d, sd_ellipsoid(P, c + np.array(t) * PAW_SCALE, np.array((0.0052, 0.0068, 0.0060)) * PAW_SCALE),
                 0.0025)
    d = smax(d, -(P[:, 2] - 0.0004), 0.0015)  # flat contact with the ground
    return d


def legs_sdf(P):
    k = LEG_THICK
    d = None
    for s in (1, -1):
        dy = STANCE["front"][s]
        F = {key: _side(v, s, dy) for key, v in FRONT_LEG.items()}
        f = sd_round_cone(P, F["shoulder"], F["elbow"], 0.024 * k, 0.016 * k)                   # humerus
        f = smin(f, sd_ellipsoid(P, _side((0.043, -0.122, 0.190), s, dy), np.array((0.020, 0.030, 0.040)) * k,
                                 rot_x(0.35)), 0.010)                                            # triceps
        f = smin(f, sd_round_cone(P, F["elbow"], F["wrist"], 0.0155 * k, 0.0085 * k), 0.010)     # forearm
        f = smin(f, sd_ellipsoid(P, _side((0.039, -0.113, 0.118), s, dy), np.array((0.0125, 0.016, 0.030)) * k),
                 0.008)                                                                          # forearm muscles
        f = smin(f, sd_sphere(P, _side((0.040, -0.097, 0.153), s, dy), 0.0105 * k), 0.006)       # elbow point
        f = smin(f, sd_round_cone(P, F["wrist"], F["pastern"], 0.0090 * k, 0.0095 * k), 0.006)  # pastern
        f = smin(f, paw_sdf(P, F["paw"], s), 0.006)

        dyh = STANCE["hind"][s]
        H = {key: _side(v, s, dyh) for key, v in HIND_LEG.items()}
        h = sd_round_cone(P, H["hip"], H["stifle"], 0.030 * k, 0.019 * k)                       # femur / thigh
        h = smin(h, sd_ellipsoid(P, _side((0.046, 0.125, 0.168), s, dyh), np.array((0.022, 0.030, 0.040)) * k,
                                 rot_x(-0.5)), 0.012)
        h = smin(h, sd_round_cone(P, H["stifle"], H["hock"], 0.0170 * k, 0.0095 * k), 0.010)     # tibia
        h = smin(h, sd_ellipsoid(P, _side((0.046, 0.121, 0.122), s, dyh), np.array((0.012, 0.020, 0.030)) * k,
                                 rot_x(0.6)), 0.008)                                             # calf
        h = smin(h, sd_sphere(P, _side((0.045, 0.173, 0.077), s, dyh), 0.0085 * k), 0.005)       # hock point
        h = smin(h, sd_round_cone(P, H["hock"], H["meta"], 0.0095 * k, 0.0090 * k), 0.006)       # metatarsus
        h = smin(h, paw_sdf(P, H["paw"], s, hind=True), 0.006)
        both = np.minimum(f, h)
        d = both if d is None else np.minimum(d, both)
    return d


def body_base_sdf(P, jaw_open=True):
    Pm = P.copy()
    Pm[:, 0] = np.abs(Pm[:, 0])
    d = torso_sdf(Pm)
    d = smin(d, skull_sdf(Pm), 0.020)
    d = smin(d, jaw_sdf(jaw_rest(Pm) if jaw_open else Pm), 0.004)
    d = smin(d, legs_sdf(P), 0.020)
    if jaw_open:
        d = ssub(d, mouth_cavity_sdf(Pm), 0.003)
    return d


def body_sdf(P):
    """Final body: anatomy + narrowed, slanted eye openings + skin irregularity."""
    d = body_base_sdf(P)
    Pm = P.copy()
    Pm[:, 0] = np.abs(Pm[:, 0])
    E, R = ANAT["eye_center"], ANAT["eye_radius"]
    # narrow almond lid opening (a glare), slanted up toward the outer corner
    d = ssub(d, sd_ellipsoid(Pm, E + ANAT["eye_gaze"] * 0.45 * R, (0.98 * R, 0.46 * R, 0.95 * R), ANAT["eye_frame"]),
             0.0016)
    d = ssub(d, sd_sphere(Pm, E, R + 0.0003), 0.0008)
    d = d + 0.00045 * fbm(P, 60.0, 2, seed=3)
    return d


def tail_sdf(P):
    d = None
    for i in range(len(TAIL_POINTS) - 1):
        seg = sd_round_cone(P, TAIL_POINTS[i], TAIL_POINTS[i + 1], TAIL_RADII[i], TAIL_RADII[i + 1])
        d = seg if d is None else smin(d, seg, 0.004)
    return d


def nose_sdf(P):
    """Nose in its own local frame (origin = nose centre, -Y = forward)."""
    d = sd_ellipsoid(P, (0, 0, 0), (0.0088, 0.0071, 0.0067))
    d = smax(d, (P[:, 2] - 0.0050), 0.0025)                     # flatter top where it meets the muzzle
    for s in (1, -1):
        nos = sd_round_cone(P, (s * 0.0040, -0.0068, -0.0010), (s * 0.0063, -0.0032, -0.0002), 0.0015, 0.0011)
        d = ssub(d, nos, 0.0009)                                 # nostrils
    groove = sd_round_cone(P, (0, -0.0074, -0.0006), (0, -0.0058, -0.0068), 0.00055, 0.00055)
    d = ssub(d, groove, 0.0006)                                  # philtrum groove
    d = d + 0.00012 * fbm(P, 900.0, 2, seed=11)                  # irregular surface
    return d


# ---------- feature placement by ray-marching the SDF -------------------------

def _surface_along(sdf, origin, direction, t_max=0.08, steps=60):
    """Find the first zero crossing of sdf from an inside point along direction (bisection)."""
    o = np.asarray(origin, dtype=np.float64)
    dvec = np.asarray(direction, dtype=np.float64)
    dvec = dvec / np.linalg.norm(dvec)
    ts = np.linspace(0, t_max, steps)
    vals = sdf(o[None, :] + ts[:, None] * dvec[None, :])
    idx = np.nonzero((vals[:-1] < 0) & (vals[1:] >= 0))[0]
    if len(idx) == 0:
        raise BuildError(f"Could not find the skin surface from {origin} along {direction}")
    lo, hi = ts[idx[0]], ts[idx[0] + 1]
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if sdf((o + mid * dvec)[None, :])[0] < 0:
            lo = mid
        else:
            hi = mid
    return o + 0.5 * (lo + hi) * dvec


UPPER_TEETH = [
    # x, y, length, root radius, kind   (left side; mirrored for the right)
    (0.0018, -0.3505, 0.0034, 0.0010, "incisor"),
    (0.0042, -0.3493, 0.0036, 0.0010, "incisor"),
    (0.0064, -0.3474, 0.0040, 0.0011, "incisor"),
    (0.0090, -0.3418, 0.0215, 0.0024, "canine"),     # saber fang
    (0.0108, -0.3330, 0.0048, 0.0016, "premolar"),
    (0.0118, -0.3240, 0.0052, 0.0017, "premolar"),
    (0.0126, -0.3140, 0.0058, 0.0019, "premolar"),
    (0.0133, -0.3030, 0.0070, 0.0024, "carnassial"),
]


LOWER_TEETH = [
    (0.0016, -0.3462, 0.0030, 0.0009, "incisor"),
    (0.0036, -0.3452, 0.0032, 0.0009, "incisor"),
    (0.0054, -0.3436, 0.0034, 0.0010, "incisor"),
    (0.0074, -0.3392, 0.0135, 0.0020, "canine"),
    (0.0090, -0.3300, 0.0042, 0.0014, "premolar"),
    (0.0100, -0.3210, 0.0046, 0.0015, "premolar"),
    (0.0110, -0.3110, 0.0052, 0.0016, "premolar"),
    (0.0118, -0.3000, 0.0062, 0.0020, "carnassial"),
]


def jaw_member(P):
    """True for points (rest head space, either side) that belong to the opened lower jaw."""
    Pm = P.copy()
    Pm[:, 0] = np.abs(Pm[:, 0])
    q = jaw_rest(Pm)
    zline = np.interp(q[:, 1], [-0.352, -0.297, -0.270], [0.3592, 0.3626, 0.3640])
    zline_open = np.interp(Pm[:, 1], [-0.352, -0.297, -0.270], [0.3592, 0.3626, 0.3640])
    return (Pm[:, 1] < -0.268) & (q[:, 2] < zline + 0.0004) & (Pm[:, 2] < zline_open + 0.0004)


def head_pose_matrices():
    """4x4 matrices: M1 lowers the neck, M_head = full rigid transform of the head."""
    def about(pivot, R):
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = pivot - R @ pivot
        return m
    M1 = about(NECK_PIVOT, rot_x(NECK_DROP))
    atlas = (M1 @ np.append(ATLAS_PIVOT, 1.0))[:3]
    M2 = about(atlas, rot_x(HEAD_LIFT))
    return M1, M2, M2 @ M1


def pose_deform(P):
    """Bends the neck down and lifts the head (smoothly blended linear skinning of rest points)."""
    M1, M2, _ = head_pose_matrices()
    w1 = sstep(-0.150, -0.205, P[:, 1]) * sstep(0.215, 0.262, P[:, 2])
    w2 = sstep(-0.212, -0.236, P[:, 1]) * sstep(0.300, 0.330, P[:, 2])
    P1 = P + w1[:, None] * ((P @ M1[:3, :3].T + M1[:3, 3]) - P)
    P2 = P1 + w2[:, None] * ((P1 @ M2[:3, :3].T + M2[:3, 3]) - P1)
    return P2


def head_matrix():
    return Matrix(head_pose_matrices()[2].tolist())


def place_features():
    """Places eyes, nose, ears, teeth and the mouth line on the actual sculpted surface (rest pose)."""
    R = 0.0084
    gaze = np.array([0.40, -1.0, 0.03])                 # left eye: forward, ~22 deg outward
    gaze /= np.linalg.norm(gaze)
    surf = _surface_along(body_base_sdf, (0.010, -0.285, 0.398), gaze)
    E = surf - gaze * (R - 0.0006)                      # deep-set under the brow
    ANAT["eye_radius"] = R
    ANAT["eye_center"] = E
    ANAT["eye_gaze"] = gaze
    ex = np.cross([0.0, 0.0, 1.0], gaze)
    ex /= np.linalg.norm(ex)
    ey = np.cross(gaze, ex)
    sl = math.radians(22)                               # strong upward slant = angry glare
    ex, ey = math.cos(sl) * ex + math.sin(sl) * ey, -math.sin(sl) * ex + math.cos(sl) * ey
    ANAT["eye_frame"] = np.stack([ex, ey, gaze], 1)

    nose_surf = _surface_along(body_base_sdf, (0, -0.338, 0.3695), (0, -1, -0.05))
    ANAT["nose_center"] = nose_surf + np.array([0, 0.0030, 0.0003])

    # ears pinned back and out (aggression)
    ear_dir = np.array([0.44, 0.60, 0.68])          # swept back like a jackal god, not flat
    ear_dir /= np.linalg.norm(ear_dir)
    ANAT["ear_root"] = _surface_along(body_base_sdf, (0.016, -0.250, 0.398), ear_dir) - ear_dir * 0.004
    ANAT["ear_dir"] = ear_dir

    # closed-mouth lip line (used for lip colour, in closed-jaw coordinates)
    line = []
    for y, z in [(-0.297, 0.3626), (-0.310, 0.3618), (-0.322, 0.3608), (-0.334, 0.3600), (-0.345, 0.3594)]:
        p = _surface_along(closed_head_sdf, (0.0, y, z), (1, 0, 0), t_max=0.05)
        line.append(p - np.array([0.0006, 0, 0]))
    line.append(np.array([0.0, -0.3505, 0.3592]))
    ANAT["mouth_line"] = [np.asarray(p) for p in line]

    # gum lines for the teeth
    upper, lower = [], []
    tilt = math.tan(JAW_OPEN * 0.45)
    for x, y, L, r, kind in UPPER_TEETH:
        z0 = MOUTH_CAVITY["c"][2] + (y - MOUTH_CAVITY["c"][1]) * tilt     # inside the open mouth
        g = _surface_along(lambda P: -body_base_sdf(P), (x, y, z0), (0, 0, 1), t_max=0.03)
        upper.append((g, L, r, kind))
    for x, y, L, r, kind in LOWER_TEETH:
        axis_z = np.interp(y, [-0.343, -0.282], [0.3548, 0.3565])
        g = _surface_along(jaw_sdf_world_closed, (x, y, axis_z), (0, 0, 1), t_max=0.03)
        lower.append((g, L, r, kind))
    ANAT["upper_teeth"], ANAT["lower_teeth"] = upper, lower
    log(f"eye centre {np.round(E, 4)}, nose {np.round(ANAT['nose_center'], 4)}, ear root {np.round(ANAT['ear_root'], 4)}")


# =============================================================================
# SURFACE NETS MESHER (narrow-band sampling + dual contouring-lite)
# =============================================================================

def sample_grid(sdf, bmin, h, shape, block=16, chunk=1_200_000):
    nx, ny, nz = shape
    S = np.empty(shape, dtype=np.float32)
    nb = [math.ceil(s / block) for s in shape]
    bi = np.stack(np.meshgrid(*[np.arange(n) for n in nb], indexing="ij"), -1).reshape(-1, 3)
    lo = bi * block
    hi = np.minimum(lo + block, np.array(shape))
    centers = bmin + (lo + hi - 1) / 2.0 * h
    radius = np.linalg.norm((hi - lo - 1) / 2.0 * h, axis=1)
    dc = sdf(centers)
    active = np.abs(dc) < radius * 1.6 + 3 * h
    for b in np.nonzero(~active)[0]:
        S[lo[b, 0]:hi[b, 0], lo[b, 1]:hi[b, 1], lo[b, 2]:hi[b, 2]] = dc[b]
    act = np.nonzero(active)[0]
    batch, count = [], 0
    for n_i, b in enumerate(act):
        batch.append(b)
        count += int(np.prod(hi[b] - lo[b]))
        if count >= chunk or n_i == len(act) - 1:
            pts, slices = [], []
            for bb in batch:
                g = np.stack(np.meshgrid(*[np.arange(lo[bb, k], hi[bb, k]) for k in range(3)], indexing="ij"), -1)
                slices.append((bb, g.shape[:3]))
                pts.append(g.reshape(-1, 3))
            P = bmin + np.concatenate(pts).astype(np.float64) * h
            vals = sdf(P).astype(np.float32)
            o = 0
            for bb, shp in slices:
                n = int(np.prod(shp))
                S[lo[bb, 0]:hi[bb, 0], lo[bb, 1]:hi[bb, 1], lo[bb, 2]:hi[bb, 2]] = vals[o:o + n].reshape(shp)
                o += n
            batch, count = [], 0
    return S


def surface_nets(sdf, bmin, bmax, h):
    """Returns (vertices, quads) of the zero level set of sdf inside the box."""
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    shape = tuple(int(v) for v in np.ceil((bmax - bmin) / h).astype(int) + 1)
    S = sample_grid(sdf, bmin, h, shape)
    inside = S < 0
    cells = np.array(shape) - 1
    ncell = int(np.prod(cells))
    lin_all, pts_all, edge_info = [], [], []
    for axis in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[axis] = slice(0, -1)
        s1[axis] = slice(1, None)
        crossing = inside[tuple(s0)] != inside[tuple(s1)]
        ii, jj, kk = np.nonzero(crossing)
        idx = np.stack([ii, jj, kk], 1)
        a = S[tuple(s0)][ii, jj, kk].astype(np.float64)
        b = S[tuple(s1)][ii, jj, kk].astype(np.float64)
        t = a / (a - b)
        pts = idx.astype(np.float64)
        pts[:, axis] += t
        pts = bmin + pts * h
        others = [(axis + 1) % 3, (axis + 2) % 3]
        for da in (0, 1):
            for db in (0, 1):
                c = idx.copy()
                c[:, others[0]] -= da
                c[:, others[1]] -= db
                ok = np.all((c >= 0) & (c < cells), axis=1)
                lin_all.append(np.ravel_multi_index(c[ok].T, cells))
                pts_all.append(pts[ok])
        edge_info.append((axis, idx, inside[ii, jj, kk]))
    lin = np.concatenate(lin_all)
    pts = np.concatenate(pts_all)
    cnt = np.bincount(lin, minlength=ncell)
    has = cnt > 0
    vid = np.full(ncell, -1, dtype=np.int64)
    vid[has] = np.arange(int(has.sum()))
    V = np.stack([np.bincount(lin, weights=pts[:, d], minlength=ncell)[has] for d in range(3)], 1) / cnt[has][:, None]
    faces = []
    for axis, idx, start_inside in edge_info:
        a1, a2 = (axis + 1) % 3, (axis + 2) % 3
        ok = (idx[:, a1] >= 1) & (idx[:, a2] >= 1) & (idx[:, a1] < cells[a1]) & (idx[:, a2] < cells[a2])
        idx_ok, si = idx[ok], start_inside[ok]
        quad = []
        for da, db in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            c = idx_ok.copy()
            c[:, a1] += da
            c[:, a2] += db
            quad.append(vid[np.ravel_multi_index(c.T, cells)])
        q = np.stack(quad, 1)
        q[~si] = q[~si][:, ::-1]
        faces.append(q)
    F = np.concatenate(faces)
    if np.any(F < 0):
        raise BuildError("Surface nets produced an invalid face (open grid boundary?)")
    return V, F


def mesh_edges(F):
    k = F.shape[1]
    e = np.concatenate([np.stack([F[:, i], F[:, (i + 1) % k]], 1) for i in range(k)])
    return np.concatenate([e, e[:, ::-1]])


def taubin_smooth(V, F, iters=4, lam=0.5, mu=-0.53, weights=None):
    E = mesh_edges(F)
    deg = np.bincount(E[:, 0], minlength=len(V)).astype(np.float64)
    deg[deg == 0] = 1
    V = V.copy()
    w = 1.0 if weights is None else weights[:, None]
    for _ in range(iters):
        for f in (lam, mu):
            avg = np.stack([np.bincount(E[:, 0], weights=V[E[:, 1], d], minlength=len(V)) for d in range(3)], 1) / deg[:, None]
            V = V + f * w * (avg - V)
    return V


def tri_faces(F):
    if F.shape[1] == 3:
        return F
    return np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]]])


def vertex_normals(V, F):
    T = tri_faces(F)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    vn = np.zeros_like(V)
    for i in range(3):
        for d in range(3):
            vn[:, d] += np.bincount(T[:, i], weights=fn[:, d], minlength=len(V))
    return normalize(vn)


# =============================================================================
# BLENDER HELPERS
# =============================================================================

def get_collection(name, parent=None):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if col.name not in [c.name for c in parent.children]:
        parent.children.link(col)
    return col


def mesh_from_numpy(name, V, F, collection, smooth=True):
    me = bpy.data.meshes.new(name)
    F = np.asarray(F)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", np.asarray(V, dtype=np.float32).ravel())
    nl = F.shape[1]
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, nl, dtype=np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.full(len(F), nl, dtype=np.int32))
    except (AttributeError, RuntimeError, TypeError):
        pass  # Blender 4.x derives loop_total from loop_start
    me.update(calc_edges=True)
    if me.validate(verbose=False):
        log(f"  mesh '{name}' needed minor validation fixes")
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)
    return obj


def mesh_numpy(obj):
    me = obj.data
    V = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    me.calc_loop_triangles()
    T = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    return V.reshape(-1, 3), T.reshape(-1, 3)


def set_color_attribute(me, name, rgb_linear, domain="POINT"):
    attr = me.color_attributes.get(name) or me.color_attributes.new(name, "FLOAT_COLOR", domain)
    rgba = np.concatenate([rgb_linear, np.ones((len(rgb_linear), 1))], 1).astype(np.float32)
    attr.data.foreach_set("color", rgba.ravel())
    return attr


def apply_modifiers(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    new_me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    old = obj.data
    obj.modifiers.clear()
    obj.data = new_me
    bpy.data.meshes.remove(old)


def parent_keep(child, parent):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def triangle_count(obj):
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


# ---------- shader node helpers ----------------------------------------------

def new_material(name):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0) and not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    return mat, nt, out


def _sock(sockets, name, stype=None):
    for s in sockets:
        if s.name == name and (stype is None or s.type == stype) and s.enabled:
            return s
    for s in sockets:
        if s.name == name and (stype is None or s.type == stype):
            return s
    raise BuildError(f"Socket '{name}' not found")


def set_input(node, name, value):
    """Sets an input by name, including inputs Blender 5 hides for the current mode."""
    for sock in node.inputs:
        if sock.name == name:
            sock.default_value = value
            return
    raise BuildError(f"Input '{name}' not found on {node.bl_idname}")


def _set(nt, sock, val):
    if isinstance(val, bpy.types.NodeSocket):
        nt.links.new(val, sock)
    else:
        sock.default_value = val


def math_op(nt, op, a, b=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    _set(nt, n.inputs[0], a)
    if b is not None:
        _set(nt, n.inputs[1], b)
    return n.outputs[0]


def mix_color(nt, fac, a, b, blend="MIX"):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    n.blend_type = blend
    _set(nt, _sock(n.inputs, "Factor", "VALUE"), fac)
    _set(nt, _sock(n.inputs, "A", "RGBA"), a)
    _set(nt, _sock(n.inputs, "B", "RGBA"), b)
    return _sock(n.outputs, "Result", "RGBA")


def map_range(nt, value, fmin, fmax, tmin=0.0, tmax=1.0, smooth=True):
    n = nt.nodes.new("ShaderNodeMapRange")
    if smooth:
        n.interpolation_type = "SMOOTHSTEP"
    _set(nt, n.inputs["Value"], value)
    n.inputs["From Min"].default_value = fmin
    n.inputs["From Max"].default_value = fmax
    n.inputs["To Min"].default_value = tmin
    n.inputs["To Max"].default_value = tmax
    return n.outputs["Result"]


def attribute(nt, name, output="Color"):
    n = nt.nodes.new("ShaderNodeAttribute")
    n.attribute_type = "GEOMETRY"
    n.attribute_name = name
    return n.outputs[output]


def principled(nt, out, **inputs):
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    for k, v in inputs.items():
        _set(nt, bsdf.inputs[k], v)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return bsdf


def lin(rgb):
    c = srgb_to_linear(rgb)
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)


# =============================================================================
# FUR COLOUR MAP (shared by fur strands, skin, and the baked Roblox texture)
# =============================================================================

PAL = {
    "sand":      (0.62, 0.45, 0.27),
    "golden":    (0.72, 0.47, 0.22),
    "rust":      (0.62, 0.30, 0.11),
    "cream":     (0.88, 0.81, 0.68),
    "beige":     (0.74, 0.62, 0.44),
    "grizzle":   (0.36, 0.30, 0.24),
    "graybrown": (0.34, 0.27, 0.20),
    "darkbrown": (0.14, 0.09, 0.055),
    "charcoal":  (0.06, 0.05, 0.045),
    "gum":       (0.40, 0.12, 0.12),
}


def _mix(col, target, w):
    return col + (np.asarray(target) - col) * w[:, None]


def _seg_dist(P, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ba = b - a
    t = np.clip(((P - a) @ ba) / float(ba @ ba), 0, 1)
    return np.linalg.norm(P - (a + np.outer(t, ba)), axis=1)


def body_color(P, N):
    """sRGB fur colour for points on the skin, in closed-jaw rest coordinates."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ax = np.abs(x)
    Pm = P.copy()
    Pm[:, 0] = ax
    nz = N[:, 2]
    col = np.tile(np.asarray(PAL["sand"]), (len(P), 1))

    torso = sstep(0.150, 0.180, z) * sstep(-0.215, -0.185, -y) * sstep(-0.24, -0.20, y)
    col = _mix(col, PAL["grizzle"], 0.90 * torso * sstep(0.25, 0.75, nz))              # dark saddle
    col = _mix(col, PAL["charcoal"], 0.75 * torso * sstep(0.85, 0.97, nz) * sstep(0.022, 0.006, ax))  # crest line
    col = _mix(col, PAL["beige"], torso * sstep(-0.25, -0.65, nz))                     # belly
    legs = sstep(0.175, 0.150, z)
    col = _mix(col, PAL["golden"], legs)
    col = _mix(col, PAL["darkbrown"], 0.88 * sstep(0.135, 0.095, z))                   # dark "stockings"
    col = _mix(col, PAL["charcoal"], 0.7 * sstep(0.030, 0.012, z))                      # black paws
    chest = sstep(-0.12, -0.16, y) * sstep(0.30, 0.24, z) * sstep(0.17, 0.20, z) * sstep(-0.1, -0.5, N[:, 1])
    col = _mix(col, PAL["rust"], 0.55 * chest)
    throat = sstep(-0.18, -0.215, y) * sstep(0.36, 0.31, z) * sstep(0.0, -0.5, nz + 0.3 * N[:, 1])
    col = _mix(col, PAL["cream"], 0.75 * throat)

    # ---- head ----
    head = sstep(-0.215, -0.240, y) * sstep(0.325, 0.345, z)
    col = _mix(col, PAL["golden"], head)
    forehead = head * sstep(0.30, 0.75, nz) * sstep(-0.33, -0.30, -y)
    col = _mix(col, PAL["grizzle"], 0.75 * forehead)
    cheek = head * sstep(0.385, 0.370, z) * sstep(0.012, 0.024, ax)
    col = _mix(col, PAL["cream"], 0.70 * cheek)
    E = ANAT["eye_center"]
    d_eye = np.linalg.norm(Pm - E, axis=1)
    col = _mix(col, PAL["rust"], 0.85 * (1 - sstep(0.012, 0.026, d_eye)) * head)         # fiery ring
    col = _mix(col, PAL["darkbrown"], 0.92 * (1 - sstep(0.008, 0.016, d_eye)) * head)    # dark war-mask
    muzzle = sstep(-0.300, -0.315, y)
    col = _mix(col, PAL["golden"], 0.6 * muzzle * sstep(0.364, 0.372, z))
    bridge = muzzle * sstep(0.374, 0.382, z) * (1 - sstep(0.005, 0.012, ax))
    col = _mix(col, PAL["darkbrown"], 0.75 * bridge)
    lower = sstep(-0.270, -0.290, y) * sstep(0.366, 0.360, z)
    col = _mix(col, PAL["cream"], 0.9 * lower)
    tear = 1 - sstep(0.0020, 0.0060, _seg_dist(Pm, E + np.array([-0.006, -0.005, -0.003]), (0.0105, -0.338, 0.3745)))
    col = _mix(col, PAL["charcoal"], 0.85 * tear)
    lips = np.full(len(P), 1.0)
    for a, b in zip(ANAT["mouth_line"][:-1], ANAT["mouth_line"][1:]):
        lips = np.minimum(lips, _seg_dist(Pm, a, b))
    col = _mix(col, PAL["charcoal"], 1 - sstep(0.0012, 0.0032, lips))

    v = 1.0 + 0.08 * fbm(P, 38.0, 3, seed=5)
    col = col * v[:, None]
    hue = 0.035 * fbm(P, 22.0, 2, seed=9)
    col[:, 0] += hue
    col[:, 2] -= hue
    return np.clip(col, 0, 1)


def body_color_at(P, N):
    """Colour for points in the open-jaw rest pose: jaw points are coloured in closed-jaw space,
    the inside of the mouth gets gums, the snarl-retracted lip shows gum around the fangs."""
    jaw = jaw_member(P)
    Q, NQ = P.copy(), N.copy()
    Rj = rot_x(-JAW_OPEN)
    if jaw.any():
        sgn = np.sign(P[jaw, :1] + 1e-12)
        Pm = P[jaw].copy()
        Pm[:, 0] = np.abs(Pm[:, 0])
        q = jaw_rest(Pm)
        q[:, 0] *= sgn[:, 0]
        Q[jaw] = q
        NQ[jaw] = N[jaw] @ Rj.T
    col = body_color(Q, NQ)
    Pm = P.copy()
    Pm[:, 0] = np.abs(Pm[:, 0])
    d_cav = mouth_cavity_sdf(Pm)
    gum = 1 - sstep(0.0010, 0.0030, d_cav)
    retract = 1 - sstep(0.0005, 0.0022, sd_ellipsoid(Pm, (0.0098, -0.338, 0.3612), (0.0045, 0.0080, 0.0044)))
    col = _mix(col, PAL["charcoal"], (1 - sstep(0.0030, 0.0055, d_cav)) * (1 - gum))  # black lip rims
    gum_col = np.asarray(PAL["gum"]) * (0.75 + 0.25 * value_noise(P, 900.0, seed=13)[:, None] ** 2)
    col = col + (gum_col - col) * np.maximum(gum, retract)[:, None]
    return np.clip(col, 0, 1)


def tail_color(P, N):
    span = TAIL_POINTS[0, 2] - TAIL_POINTS[-1, 2]
    t = np.clip((TAIL_POINTS[0, 2] - P[:, 2]) / span * 0.6 + (P[:, 1] - TAIL_POINTS[0, 1]) / 0.2 * 0.4, 0, 1)
    col = np.tile(np.asarray(PAL["sand"]), (len(P), 1))
    col = _mix(col, PAL["rust"], 0.45 * sstep(0.2, 0.7, t))
    upper = sstep(-0.2, 0.6, N[:, 2] - N[:, 1] * 0.5)
    col = _mix(col, PAL["grizzle"], 0.75 * upper * (1 - 0.5 * t))
    col = _mix(col, PAL["charcoal"], 0.65 * np.exp(-((t - 0.10) / 0.06) ** 2) * upper)
    col = _mix(col, PAL["beige"], 0.45 * sstep(0.0, -0.6, N[:, 2]))
    col = _mix(col, PAL["charcoal"], 0.92 * sstep(0.78, 0.93, t))                       # black tail tip
    v = 1.0 + 0.08 * fbm(P, 30.0, 3, seed=21)
    return np.clip(col * v[:, None], 0, 1)


EAR_NOTCHES = {
    # (u position along the ear, depth as a fraction of the half-width, width, edge +1/-1)
    "L": [(0.52, 0.38, 0.030, 1), (0.71, 0.30, 0.022, -1), (0.30, 0.18, 0.020, 1)],
    "R": [(0.43, 0.34, 0.032, 1), (0.64, 0.22, 0.020, 1)],
}


# =============================================================================
# MATERIALS
# =============================================================================

def create_materials():
    M = {}

    # skin under the fur (coloured like the dense underfur so gaps never look bald)
    mat, nt, out = new_material("Fox_Skin")
    col = attribute(nt, "fur_col")
    darker = mix_color(nt, 0.45, col, (0.0, 0.0, 0.0, 1.0), "MULTIPLY")
    principled(nt, out, **{"Base Color": darker, "Roughness": 0.78})
    M["skin"] = mat

    # fur strands: Principled Hair BSDF with per-strand base/tip colour and root darkening
    for key, rough, radial in (("fur", 0.30, 0.42), ("underfur", 0.45, 0.55)):
        mat, nt, out = new_material("Fox_Fur" if key == "fur" else "Fox_Underfur")
        info = nt.nodes.new("ShaderNodeHairInfo")
        base = attribute(nt, "fur_base")
        tip = attribute(nt, "fur_tip")
        root = mix_color(nt, 0.45, base, (0.0, 0.0, 0.0, 1.0), "MULTIPLY")
        c1 = mix_color(nt, map_range(nt, info.outputs["Intercept"], 0.0, 0.35), root, base)
        c2 = mix_color(nt, map_range(nt, info.outputs["Intercept"], 0.55, 1.0), c1, tip)
        gamma = nt.nodes.new("ShaderNodeGamma")
        _set(nt, gamma.inputs["Color"], c2)
        gamma.inputs["Gamma"].default_value = 2.8   # hair multiple scattering lightens colours a lot
        hair = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
        hair.parametrization = "COLOR"
        _set(nt, hair.inputs["Color"], gamma.outputs[0])
        set_input(hair, "Roughness", rough)
        set_input(hair, "Radial Roughness", radial)
        set_input(hair, "Coat", 0.08)
        set_input(hair, "Random Roughness", 0.15)
        nt.links.new(hair.outputs[0], out.inputs["Surface"])
        M[key] = mat

    # ears: vertex-coloured skin with light translucency
    mat, nt, out = new_material("Fox_Ear")
    col = attribute(nt, "ear_col")
    bsdf = principled(nt, out, **{"Base Color": col, "Roughness": 0.72})
    try:
        bsdf.inputs["Subsurface Weight"].default_value = 0.15
        bsdf.inputs["Subsurface Radius"].default_value = (0.004, 0.0015, 0.001)
        _set(nt, bsdf.inputs["Subsurface Scale"], 0.01)
    except KeyError:
        pass
    M["ear"] = mat

    # nose: near-black, wet, pebbled rhinarium texture
    mat, nt, out = new_material("Fox_Nose")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 1400.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.45
    bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(vor.outputs["Distance"], bump.inputs["Height"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 600.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    rough = map_range(nt, noise.outputs["Fac"], 0.35, 0.65, 0.18, 0.48)
    principled(nt, out, **{"Base Color": lin((0.035, 0.030, 0.030)), "Roughness": rough,
                           "Normal": bump.outputs["Normal"], "Coat Weight": 0.3, "Coat Roughness": 0.08})
    M["nose"] = mat

    # eyeball: amber iris with radial fibres, vertical-oval pupil, dark limbal ring
    mat, nt, out = new_material("Fox_Eye_Iris")
    R = ANAT["eye_radius"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    ex, ey = sep.outputs["X"], sep.outputs["Y"]
    rn = math_op(nt, "DIVIDE", math_op(nt, "SQRT", math_op(nt, "ADD", math_op(nt, "MULTIPLY", ex, ex),
                                                              math_op(nt, "MULTIPLY", ey, ey))), R)
    px = math_op(nt, "DIVIDE", ex, R * 0.10)
    py = math_op(nt, "DIVIDE", ey, R * 0.42)
    pupil_d = math_op(nt, "SQRT", math_op(nt, "ADD", math_op(nt, "MULTIPLY", px, px), math_op(nt, "MULTIPLY", py, py)))
    pupil = map_range(nt, pupil_d, 1.0, 0.86)
    radial = nt.nodes.new("ShaderNodeCombineXYZ")
    _set(nt, radial.inputs["X"], math_op(nt, "MULTIPLY", math_op(nt, "DIVIDE", ex, math_op(nt, "ADD", rn, 0.001)), 1.0 / R * 0.0011))
    _set(nt, radial.inputs["Y"], math_op(nt, "MULTIPLY", math_op(nt, "DIVIDE", ey, math_op(nt, "ADD", rn, 0.001)), 1.0 / R * 0.0011))
    _set(nt, radial.inputs["Z"], math_op(nt, "MULTIPLY", rn, 3.0))
    fib = nt.nodes.new("ShaderNodeTexNoise")
    fib.inputs["Scale"].default_value = 26.0
    fib.inputs["Detail"].default_value = 6.0
    nt.links.new(radial.outputs[0], fib.inputs["Vector"])
    iris = mix_color(nt, map_range(nt, rn, 0.15, 0.62), lin((0.98, 0.74, 0.12)), lin((0.84, 0.36, 0.04)))
    fibres = map_range(nt, fib.outputs["Fac"], 0.3, 0.7, 0.62, 1.25)
    iris = mix_color(nt, 1.0, iris, fibres, "MULTIPLY")
    iris = mix_color(nt, map_range(nt, rn, 0.58, 0.70), iris, lin((0.05, 0.028, 0.014)))   # limbal ring
    iris = mix_color(nt, pupil, iris, lin((0.004, 0.003, 0.003)))
    principled(nt, out, **{"Base Color": iris, "Roughness": 0.22,
                           "Emission Color": iris, "Emission Strength": 7.0})   # fantasy glow
    M["eye"] = mat

    # cornea: clear, wet, refractive
    mat, nt, out = new_material("Fox_Eye_Cornea")
    bsdf = principled(nt, out, **{"Base Color": (1, 1, 1, 1), "Roughness": 0.0, "IOR": 1.376})
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    M["cornea"] = mat

    for name, rgb, r in (("lid", (0.04, 0.03, 0.028), 0.40), ("claw", (0.035, 0.03, 0.028), 0.22),
                         ("pad", (0.10, 0.085, 0.08), 0.62), ("tongue", (0.36, 0.11, 0.12), 0.30)):
        mat, nt, out = new_material(f"Fox_{name.capitalize()}")
        principled(nt, out, **{"Base Color": lin(rgb), "Roughness": r})
        M[name] = mat

    # teeth: yellowed ivory, wet
    mat, nt, out = new_material("Fox_Teeth")
    bsdf = principled(nt, out, **{"Base Color": lin((0.86, 0.79, 0.62)), "Roughness": 0.26})
    set_input(bsdf, "Subsurface Weight", 0.12)
    set_input(bsdf, "Subsurface Scale", 0.002)
    M["teeth"] = mat

    # studio backdrop (dark, moody)
    mat, nt, out = new_material("Studio_Backdrop")
    principled(nt, out, **{"Base Color": lin((0.17, 0.15, 0.135)), "Roughness": 0.92})
    M["backdrop"] = mat
    return M


# =============================================================================
# BODY, TAIL, NOSE
# =============================================================================

def create_body(q, col, M):
    t0 = time.time()
    h = q["voxel"]
    V, F = surface_nets(body_sdf, (-0.095, -0.395, -0.004), (0.095, 0.235, 0.462), h)
    V, F = _largest_component(V, F)
    V = taubin_smooth(V, F, iters=3)
    obj = mesh_from_numpy("Fox_Body", V, F, col)
    N = vertex_normals(V, F)
    set_color_attribute(obj.data, "fur_col", srgb_to_linear(body_color_at(V, N)))
    obj.data.materials.append(M["skin"])
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 0
    sub.render_levels = 1
    log(f"body mesh: {len(V)} verts, {len(F)} quads ({time.time() - t0:.1f}s)")
    return obj, V, F


def _largest_component(V, F):
    n = len(V)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 3 if F.shape[1] == 4 else 0]]])
    # vectorised union-find via repeated label propagation
    label = np.arange(n)
    for _ in range(200):
        a, b = label[E[:, 0]], label[E[:, 1]]
        m = np.minimum(a, b)
        new = label.copy()
        np.minimum.at(new, E[:, 0], m)
        np.minimum.at(new, E[:, 1], m)
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    counts = np.bincount(label, minlength=n)
    keep_label = np.argmax(counts)
    keep_v = label == keep_label
    keep_f = keep_v[F[:, 0]]
    remap = np.full(n, -1)
    remap[keep_v] = np.arange(int(keep_v.sum()))
    return V[keep_v], remap[F[keep_f]]


TAIL_BOX = ((-0.035, 0.160, 0.085), (0.050, 0.420, 0.300))


TAIL_HULL_BOX = ((-0.15, 0.120, 0.004), (0.16, 0.540, 0.400))


def create_tail(q, col, M):
    V, F = surface_nets(tail_sdf, TAIL_BOX[0], TAIL_BOX[1], q["voxel"])
    V = taubin_smooth(V, F, iters=3)
    obj = mesh_from_numpy("Fox_Tail", V, F, col)
    N = vertex_normals(V, F)
    set_color_attribute(obj.data, "fur_col", srgb_to_linear(tail_color(V, N)))
    obj.data.materials.append(M["skin"])
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 0
    sub.render_levels = 1
    return obj, V, F


def create_nose(col, M, voxel=0.00032):
    V, F = surface_nets(nose_sdf, (-0.011, -0.010, -0.010), (0.011, 0.010, 0.010), voxel)
    V = taubin_smooth(V, F, iters=2)
    obj = mesh_from_numpy("Fox_Nose", V, F, col)
    obj.location = Vector(ANAT["nose_center"])
    obj.data.materials.append(M["nose"])
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 2
    return obj


# =============================================================================
# EARS
# =============================================================================

def ear_frame(side):
    d = ANAT["ear_dir"].copy()
    d[0] *= side
    # pinned back: the opening turns to face sideways and a little backward
    opening = np.array([0.85 * side, -0.50, 0.05])
    z = normalize(d)
    back = normalize(-opening - np.dot(-opening, z) * z)
    x = np.cross(back, z)
    return x, back, z


def ear_surface(u, v, H, W, D):
    """Front (inner, concave) surface of the ear in ear-local coordinates (x across, y back, z up)."""
    w = 0.5 * W * np.power(np.clip(1 - u, 0, 1), 0.62) * (1.0 + 0.32 * u * (1 - u)) + 0.0004
    cup = D * (1 - v * v) * np.power(np.clip(1 - u, 0, 1), 0.55)
    lean = 0.011 * u * u
    curl = -0.0060 * np.power(np.abs(v), 4) * np.sin(np.pi * np.clip(u * 0.9 + 0.1, 0, 1))  # rim curls forward
    return np.stack([w * v, cup + lean + curl, u * H], -1)


def build_ear_geometry(side, nu, nv, H=0.150, W=0.074, D=0.033, asym=0.0, notches=()):
    """Returns (V, F, region, uv) for a thick, cupped ear with a rounded, battle-torn rim.
    region: 0 front (inner), 1 back (outer). V is relative to the ear root."""
    H = H * (1 + asym)
    us = np.linspace(0, 1, nu + 1) ** 0.9
    vs = np.linspace(-1, 1, nv + 1)
    U, Vv = np.meshgrid(us, vs, indexing="ij")
    front = ear_surface(U, Vv, H, W, D)
    # torn notches bite into the rim
    for u0, depth, width, edge in notches:
        bite = depth * np.exp(-((U - u0) / width) ** 2) * sstep(0.35, 1.0, Vv * edge)
        front[..., 0] *= 1 - bite
    thick = 0.0030 * (1 - 0.55 * U) + 0.0026 * np.power(np.abs(Vv), 8)
    back = front.copy()
    back[..., 1] += thick + 0.0045 * (1 - Vv * Vv) * np.power(np.clip(1 - U, 0, 1), 0.5)  # convex back
    ncols = nv + 1
    Vf = front.reshape(-1, 3)
    Vb = back.reshape(-1, 3)
    nfront = len(Vf)
    V = np.concatenate([Vf, Vb])
    region = np.concatenate([np.zeros(nfront), np.ones(nfront)])

    def vid(i, j, b=0):
        return b * nfront + i * ncols + j

    faces = []
    for i in range(nu):
        for j in range(nv):
            faces.append([vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)])
            faces.append([vid(i, j, 1), vid(i + 1, j, 1), vid(i + 1, j + 1, 1), vid(i, j + 1, 1)])
    for i in range(nu):
        faces.append([vid(i, 0, 1), vid(i, 0), vid(i + 1, 0), vid(i + 1, 0, 1)])
        faces.append([vid(i, nv), vid(i, nv, 1), vid(i + 1, nv, 1), vid(i + 1, nv)])
    for j in range(nv):
        faces.append([vid(0, j + 1), vid(0, j), vid(0, j, 1), vid(0, j + 1, 1)])
    F = np.array(faces)
    x, back_dir, z = ear_frame(side)
    if side < 0:
        V = V * np.array([-1.0, 1.0, 1.0])
        F = F[:, ::-1]
    Rm = np.stack([x, back_dir, z], 1)
    Vw = V @ Rm.T
    uv = np.concatenate([np.stack([U.ravel(), Vv.ravel()], 1)] * 2)
    return Vw, F, region, uv


def ear_colors(region, uv):
    u, v = uv[:, 0], np.abs(uv[:, 1])
    inner = np.tile(np.asarray((0.30, 0.20, 0.15)), (len(u), 1))
    inner = _mix(inner, PAL["darkbrown"], sstep(0.10, 0.60, u) * sstep(0.95, 0.4, v))
    inner = _mix(inner, PAL["charcoal"], 0.85 * sstep(0.55, 0.90, u))
    inner = _mix(inner, PAL["beige"], 0.65 * sstep(0.78, 0.96, v) * sstep(0.85, 0.5, u))
    inner = _mix(inner, (0.24, 0.12, 0.11), 0.6 * sstep(0.18, 0.0, u))     # ear canal
    outer = np.tile(np.asarray(PAL["sand"]), (len(u), 1))
    outer = _mix(outer, PAL["darkbrown"], sstep(0.25, 0.55, u))
    outer = _mix(outer, PAL["charcoal"], sstep(0.55, 0.85, u))
    outer = _mix(outer, PAL["charcoal"], 0.7 * sstep(0.85, 1.0, v) * sstep(0.2, 0.5, u))
    return np.where(region[:, None] < 0.5, inner, outer)


def create_ears(col, M, nu=44, nv=28):
    ears = {}
    for side, tag, asym in ((1, "L", 0.0), (-1, "R", -0.03)):
        V, F, region, uv = build_ear_geometry(side, nu, nv, asym=asym, notches=EAR_NOTCHES[tag])
        root = ANAT["ear_root"] * np.array([side, 1, 1])
        obj = mesh_from_numpy(f"Fox_Ear.{tag}", V, F, col)
        obj.location = Vector(root)
        set_color_attribute(obj.data, "ear_col", srgb_to_linear(ear_colors(region, uv)))
        obj.data.materials.append(M["ear"])
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
        bm.free()
        sub = obj.modifiers.new("Subdivision", "SUBSURF")
        sub.levels = 1
        sub.render_levels = 2
        ears[tag] = dict(obj=obj, V=V + root, F=F, region=region, uv=uv, side=side)
    return ears


def tooth_mesh(root, grow, L, r, kind, side, segs=10, rings=8):
    """One tooth: tapered, slightly back-curved cone with a blade-like cross-section."""
    grow = np.asarray(grow, dtype=np.float64)
    grow /= np.linalg.norm(grow)
    ex = np.array([1.0, 0.0, 0.0])
    ex = normalize(ex - np.dot(ex, grow) * grow)
    ey = np.cross(grow, ex)
    curve = {"canine": 0.26, "incisor": 0.05, "premolar": 0.04, "carnassial": 0.06}[kind]
    fx, fy = {"canine": (0.85, 1.05), "incisor": (0.75, 1.0), "premolar": (0.5, 1.35), "carnassial": (0.5, 1.5)}[kind]
    back = np.array([0.0, 1.0, 0.0])
    V, F = [], []
    for i in range(rings):
        t = i / (rings - 1) * 0.94
        c = root + grow * L * t + back * curve * L * t * t + ex * side * 0.05 * L * t
        rad = r * (1 - t) ** 0.8 + 0.00010
        for j in range(segs):
            a = 2 * math.pi * j / segs
            V.append(c + ex * math.cos(a) * rad * fx + ey * math.sin(a) * rad * fy)
    tip = root + grow * L + back * curve * L + ex * side * 0.05 * L
    V.append(tip)
    V.append(root - grow * 0.0004)
    n_tip, n_base = len(V) - 2, len(V) - 1
    for i in range(rings - 1):
        for j in range(segs):
            a, b = i * segs + j, i * segs + (j + 1) % segs
            F.append((a, b, b + segs, a + segs))
    last = (rings - 1) * segs
    for j in range(segs):
        F.append((last + j, last + (j + 1) % segs, n_tip))
        F.append(((j + 1) % segs, j, n_base))
    return np.array(V), F


def build_teeth(segs=10, rings=8):
    """Upper teeth (rest, open-jaw head space) and lower teeth (rotated with the open jaw)."""
    out = {}
    for key, teeth, grow_dir in (("upper", ANAT["upper_teeth"], (0, 0, -1)), ("lower", ANAT["lower_teeth"], (0, 0, 1))):
        Vs, Fs, off = [], [], 0
        for g, L, r, kind in teeth:
            for side in (1, -1):
                root = np.array([g[0] * side, g[1], g[2]]) - np.array(grow_dir) * 0.0016
                grow = np.array([0.10 * side if kind == "canine" else 0.03 * side, 0.0, grow_dir[2]])
                V, F = tooth_mesh(root, grow, L, r, kind, side, segs, rings)
                if side < 0:
                    F = [tuple(reversed(f)) for f in F]
                Vs.append(V)
                Fs += [tuple(i + off for i in f) for f in F]
                off += len(V)
        V = np.concatenate(Vs)
        if key == "lower":
            sgn = np.sign(V[:, :1] + 1e-12)
            Vm = V.copy()
            Vm[:, 0] = np.abs(Vm[:, 0])
            V = jaw_pose(Vm)
            V[:, 0] *= sgn[:, 0]
        out[key] = (V, Fs)
    return out


def create_teeth_and_tongue(col, M):
    objs = []
    for key, (V, F) in build_teeth().items():
        me = bpy.data.meshes.new(f"Fox_Teeth_{key.capitalize()}")
        me.from_pydata([tuple(v) for v in V], [], F)
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
        me.update()
        o = bpy.data.objects.new(me.name, me)
        col.objects.link(o)
        me.materials.append(M["teeth"])
        objs.append(o)
    objs.append(_tongue_object(col, M, 24, 16))
    return objs


def _tongue_object(col, M, us, vs, name="Fox_Tongue"):
    top = _surface_along(jaw_sdf_world_closed, (0.0, -0.315, 0.3550), (0, 0, 1), t_max=0.03)
    c = np.array([0.0, -0.316, top[2] - 0.0012])
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    m = Matrix.Translation(Vector(c)) @ Matrix.Diagonal((0.0076, 0.024, 0.0032, 1.0))
    bmesh.ops.create_uvsphere(bm, u_segments=us, v_segments=vs, radius=1.0, matrix=m)
    bm.to_mesh(me)
    bm.free()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = jaw_pose(V.reshape(-1, 3))
    me.vertices.foreach_set("co", V.ravel().astype(np.float32))
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    me.update()
    o = bpy.data.objects.new(name, me)
    col.objects.link(o)
    me.materials.append(M["tongue"])
    return o


# =============================================================================
# EYES
# =============================================================================

def eye_matrix(side):
    E = ANAT["eye_center"] * np.array([side, 1, 1])
    g = ANAT["eye_gaze"] * np.array([side, 1, 1])
    z = Vector(g).normalized()
    up = Vector((0, 0, 1))
    x = up.cross(z).normalized()
    y = z.cross(x)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(E)
    return m


def create_eyes(col, M):
    R = ANAT["eye_radius"]
    eyes = {}
    for side, tag in ((1, "L"), (-1, "R")):
        mw = eye_matrix(side)
        # eyeball with iris shader (object space, +Z = gaze)
        me = bpy.data.meshes.new(f"Fox_Eye.{tag}")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=32, radius=R)
        bm.to_mesh(me)
        bm.free()
        eye = bpy.data.objects.new(f"Fox_Eye.{tag}", me)
        col.objects.link(eye)
        eye.matrix_world = mw
        me.materials.append(M["eye"])
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))

        # cornea: slightly larger sphere with a forward bulge
        me = bpy.data.meshes.new(f"Fox_Cornea.{tag}")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=32, radius=R * 1.025)
        for v in bm.verts:
            zn = v.co.z / (R * 1.025)
            if zn > 0.55:
                v.co.z += 0.0009 * ((zn - 0.55) / 0.45) ** 1.5
        bm.to_mesh(me)
        bm.free()
        cornea = bpy.data.objects.new(f"Fox_Cornea.{tag}", me)
        col.objects.link(cornea)
        cornea.matrix_world = mw
        me.materials.append(M["cornea"])
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
        cornea.visible_shadow = False

        # eyelids: almond-shaped dark rim (outer corner slanted up, like a fox)
        lid = _eyelid_mesh(f"Fox_Eyelid.{tag}", R, side)
        col.objects.link(lid)
        lid.matrix_world = mw
        lid.data.materials.append(M["lid"])
        eyes[tag] = dict(eye=eye, cornea=cornea, lid=lid)
    return eyes


def _eyelid_mesh(name, R, side, n=72, m=10):
    """Tube following the eyelid margin on the eyeball, plus a thin upper/lower lid fold."""
    me = bpy.data.meshes.new(name)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    slant = math.radians(22) * side
    ah, av = math.radians(45), math.radians(21)
    # almond: pointier corners using a superellipse-like profile
    cx = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** 0.85
    cy = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** 1.15
    ang_x = ah * cx
    ang_y = av * cy
    # upper lid sits a little lower (relaxed look)
    ang_y = np.where(cy > 0, ang_y * 0.92, ang_y)
    rx = np.cos(slant) * ang_x - np.sin(slant) * ang_y
    ry = np.sin(slant) * ang_x + np.cos(slant) * ang_y
    rr = R + 0.00020
    center = np.stack([rr * np.sin(rx), rr * np.sin(ry), rr * np.cos(np.sqrt(rx ** 2 + ry ** 2))], 1)
    tang = normalize(np.roll(center, -1, 0) - np.roll(center, 1, 0))
    nrm = normalize(center)
    bin_ = normalize(np.cross(tang, nrm))
    tube_r = 0.00080
    phi = np.linspace(0, 2 * np.pi, m, endpoint=False)
    verts = []
    for i in range(n):
        for k in range(m):
            off = nrm[i] * math.cos(phi[k]) * tube_r * 1.25 + bin_[i] * math.sin(phi[k]) * tube_r
            verts.append(center[i] + off)
    faces = []
    for i in range(n):
        for k in range(m):
            a = i * m + k
            b = i * m + (k + 1) % m
            c = ((i + 1) % n) * m + (k + 1) % m
            d = ((i + 1) % n) * m + k
            faces.append((a, b, c, d))
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    me.update()
    return bpy.data.objects.new(name, me)


# =============================================================================
# CLAWS AND PADS
# =============================================================================

def toe_tips():
    tips = []
    for s in (1, -1):
        for leg, joints, key in (("front", FRONT_LEG, "front"), ("hind", HIND_LEG, "hind")):
            c = np.array(_side(joints["paw"], s, STANCE[key][s]))
            for t in TOE_OFFSETS:
                toe = c + np.array(t) * PAW_SCALE
                tips.append((toe + np.array([0, -0.0068, 0.0006]) * PAW_SCALE, leg, s))
    return tips


def create_claws_and_pads(col, M):
    bm = bmesh.new()
    for tip, leg, s in toe_tips():
        n_before = len(bm.verts)
        mtx = Matrix.Translation(Vector(tip)) @ Matrix.Rotation(math.radians(-104), 4, "X")
        bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.0027, radius2=0.00015,
                              depth=0.0135, matrix=mtx)
        for v in bm.verts[n_before:]:
            fwd = -(v.co.y - tip[1])
            v.co.z -= 9.0 * max(fwd, 0) ** 2           # hooked, digging claws
        for v in bm.verts[n_before:]:
            v.co.z = max(v.co.z, 0.0003)
    me = bpy.data.meshes.new("Fox_Claws")
    bm.to_mesh(me)
    bm.free()
    claws = bpy.data.objects.new("Fox_Claws", me)
    col.objects.link(claws)
    me.materials.append(M["claw"])

    bm = bmesh.new()
    for s in (1, -1):
        for joints, key in ((FRONT_LEG, "front"), (HIND_LEG, "hind")):
            c = np.array(_side(joints["paw"], s, STANCE[key][s]))
            pads = [((0, 0.002, -0.0094), (0.0068, 0.0060, 0.0016))]
            for t in TOE_OFFSETS:
                pads.append(((t[0], t[1], -0.0076), (0.0030, 0.0036, 0.0012)))
            for off, r in pads:
                m = Matrix.Translation(Vector(c + np.array(off) * PAW_SCALE)) @ \
                    Matrix.Diagonal((r[0] * PAW_SCALE, r[1] * PAW_SCALE, r[2] * PAW_SCALE, 1.0))
                bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=1.0, matrix=m)
    me = bpy.data.meshes.new("Fox_PawPads")
    bm.to_mesh(me)
    bm.free()
    pads = bpy.data.objects.new("Fox_PawPads", me)
    col.objects.link(pads)
    me.materials.append(M["pad"])
    for o in (claws, pads):
        o.data.polygons.foreach_set("use_smooth", np.ones(len(o.data.polygons), dtype=bool))
    return claws, pads


# =============================================================================
# FUR SYSTEM
# =============================================================================
# Hair flows along an anatomical "flow skeleton": each segment says which way hair grows
# (a -> b), how long it is at each end, how much it stands up from the skin (lift) and
# how strongly it clumps. Every root point blends the nearby segments (soft-min weights).

def flow_segments():
    S = []

    def seg(a, b, la, lb, lift, clump=0.0, mirror=True):
        S.append((a, b, la, lb, lift, clump))
        if mirror and abs(a[0]) + abs(b[0]) > 1e-6:
            S.append(((-a[0], a[1], a[2]), (-b[0], b[1], b[2]), la, lb, lift, clump))

    nose = (0, -0.364, 0.369)
    seg(nose, (0, -0.300, 0.392), 0.0022, 0.0050, 0.12)                          # muzzle top
    seg((0.010, -0.354, 0.362), (0.022, -0.295, 0.372), 0.0020, 0.0055, 0.12)     # muzzle side
    seg((0, -0.300, 0.400), (0, -0.240, 0.428), 0.0050, 0.0140, 0.30, 0.20)       # forehead -> spiky crown
    seg((0.020, -0.300, 0.392), (0.040, -0.250, 0.405), 0.0045, 0.0090, 0.22)     # over the brows
    seg((0.022, -0.300, 0.372), (0.052, -0.236, 0.350), 0.0080, 0.0290, 0.40, 0.32)  # wild cheek ruff
    seg((0, -0.340, 0.350), (0, -0.270, 0.334), 0.0025, 0.0080, 0.12)             # chin -> throat
    seg((0.016, -0.300, 0.354), (0.035, -0.245, 0.330), 0.0040, 0.0140, 0.22, 0.15)  # jaw
    seg((0, -0.235, 0.420), (0, -0.120, 0.312), 0.0220, 0.0440, 0.66, 0.52)       # bristling nape (hackles)
    seg((0, -0.255, 0.330), (0, -0.170, 0.195), 0.0250, 0.0460, 0.52, 0.45)       # huge throat / chest mane
    seg((0.040, -0.215, 0.350), (0.058, -0.150, 0.250), 0.0220, 0.0360, 0.46, 0.40)  # mane, side of neck
    seg((0, -0.120, 0.312), (0, 0.195, 0.287), 0.0460, 0.0300, 0.72, 0.62)        # spiky dorsal crest
    seg((0.050, -0.110, 0.270), (0.062, 0.120, 0.205), 0.0180, 0.0170, 0.24, 0.15)  # flank
    seg((0.050, -0.070, 0.190), (0.040, 0.080, 0.175), 0.0140, 0.0150, 0.22)      # lower ribs
    seg((0, -0.150, 0.168), (0, 0.130, 0.205), 0.0130, 0.0140, 0.25)              # belly
    for key, joints in (("front", FRONT_LEG), ("hind", HIND_LEG)):
        for s in (1, -1):
            dy = STANCE[key][s]
            J = {k: _side(v, s, dy) for k, v in joints.items()}
            if key == "front":
                chain = [(J["shoulder"], J["elbow"], 0.0170, 0.0120), (J["elbow"], J["wrist"], 0.0100, 0.0065),
                         (J["wrist"], (J["paw"][0], J["paw"][1] - 0.02, 0.006), 0.0050, 0.0035)]
            else:
                chain = [(J["hip"], J["stifle"], 0.0200, 0.0140), (J["stifle"], J["hock"], 0.0115, 0.0080),
                         (J["hock"], J["meta"], 0.0060, 0.0048),
                         (J["meta"], (J["paw"][0], J["paw"][1] - 0.02, 0.006), 0.0045, 0.0035)]
            for a, b, la, lb in chain:
                seg(a, b, la, lb, 0.18, 0.12, mirror=False)
    return S


def flow_at(P, segs):
    """Flow field for rest-pose points; lower-jaw points use closed-jaw space, rotated back open."""
    jaw = jaw_member(P)
    Q = P.copy()
    if jaw.any():
        sgn = np.sign(P[jaw, :1] + 1e-12)
        Pm = P[jaw].copy()
        Pm[:, 0] = np.abs(Pm[:, 0])
        q = jaw_rest(Pm)
        q[:, 0] *= sgn[:, 0]
        Q[jaw] = q
    flow, L, lift, clump = evaluate_flow(Q, segs)
    if jaw.any():
        flow[jaw] = flow[jaw] @ rot_x(JAW_OPEN).T
    return flow, L, lift, clump


def evaluate_flow(P, segments, tau=0.008):
    A = np.array([s[0] for s in segments])
    B = np.array([s[1] for s in segments])
    LA = np.array([s[2] for s in segments])
    LB = np.array([s[3] for s in segments])
    LIFT = np.array([s[4] for s in segments])
    CL = np.array([s[5] for s in segments])
    BA = B - A
    l2 = np.einsum("ij,ij->i", BA, BA)
    t = np.clip(np.einsum("nkj,kj->nk", P[:, None, :] - A[None], BA) / l2[None], 0, 1)
    closest = A[None] + t[..., None] * BA[None]
    d = np.linalg.norm(P[:, None, :] - closest, axis=2)
    w = np.exp(-(d - d.min(axis=1, keepdims=True)) / tau)
    w /= w.sum(axis=1, keepdims=True)
    dirs = BA / np.sqrt(l2)[:, None]
    flow = w @ dirs
    length = np.einsum("nk,nk->n", w, LA[None] + t * (LB - LA)[None])
    lift = w @ LIFT
    clump = w @ CL
    return normalize(flow), length, lift, clump


def sample_surface(V, F, n, rng, N=None):
    T = tri_faces(F)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(T), size=n, p=area / area.sum())
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    P = a[tri] * w0[:, None] + b[tri] * w1[:, None] + c[tri] * w2[:, None]
    if N is None:
        N = vertex_normals(V, F)
    Nrm = normalize(N[T[tri, 0]] * w0[:, None] + N[T[tri, 1]] * w1[:, None] + N[T[tri, 2]] * w2[:, None])
    return P, Nrm, float(area.sum())


def grow_strands(root, n, flow, length, lift, rng, k_pts, frizz=0.12, gravity=0.25, bend=0.35, sink=0.0006):
    """Integrates strands that leave the skin at 'lift' and lie down along the flow."""
    tang = flow - np.einsum("ij,ij->i", flow, n)[:, None] * n
    bad = np.linalg.norm(tang, axis=1) < 1e-4
    if bad.any():
        alt = np.cross(n[bad], np.array([1.0, 0.0, 0.0]))
        tang[bad] = alt
    tang = normalize(tang)
    lift = lift[:, None]
    d = normalize(n * lift + tang * (1 - lift))
    g = np.array([0.0, 0.0, -1.0])
    seg = (length / (k_pts - 1))[:, None]
    pts = np.empty((len(root), k_pts, 3))
    pts[:, 0] = root - n * sink
    grav = (gravity * sstep(0.008, 0.04, length))[:, None]
    for k in range(1, k_pts):
        target = normalize(tang * (1 - lift) + n * lift + g * grav)
        d = normalize(d * (1 - bend) + target * bend + rng.normal(0, frizz, d.shape))
        p = pts[:, k - 1] + d * seg
        h = np.einsum("ij,ij->i", p - root, n)
        min_h = 0.10 * seg[:, 0] * k
        p += n * np.maximum(min_h - h, 0)[:, None]
        pts[:, k] = p
    return pts


def apply_clumping(pts, strength, rng, frac=0.03, power=1.6):
    """Pulls strand tips toward nearby 'clump leader' strands."""
    n = len(pts)
    if n == 0 or strength.max() <= 0:
        return pts
    leaders = rng.choice(n, size=max(1, int(n * frac)), replace=False)
    tree = kdtree.KDTree(len(leaders))
    for i, li in enumerate(leaders):
        tree.insert(pts[li, 0], i)
    tree.balance()
    nearest = np.fromiter((leaders[tree.find(pts[i, 0])[1]] for i in range(n)), dtype=np.int64, count=n)
    k = pts.shape[1]
    t = (np.arange(k) / (k - 1)) ** power
    lead = pts[nearest]
    offset = (pts[:, :1] - lead[:, :1])  # keep each strand's own root
    target = lead + offset * (1 - t)[None, :, None]
    w = (strength[:, None] * t[None, :])[..., None]
    return pts * (1 - w) + target * w


def create_curves_object(name, pts, radius_root, radius_tip, base_rgb, tip_rgb, mat, collection):
    n, k, _ = pts.shape
    cv = bpy.data.hair_curves.new(name)
    cv.add_curves([k] * n)
    cv.position_data.foreach_set("vector", pts.astype(np.float32).ravel())
    rad = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    t = np.linspace(0, 1, k)
    radii = np.outer(radius_root, 1 - t) + np.outer(radius_tip, t)
    rad.data.foreach_set("value", radii.astype(np.float32).ravel())
    for attr_name, rgb in (("fur_base", base_rgb), ("fur_tip", tip_rgb)):
        a = cv.attributes.new(attr_name, "FLOAT_COLOR", "CURVE")
        rgba = np.concatenate([srgb_to_linear(rgb), np.ones((n, 1))], 1).astype(np.float32)
        a.data.foreach_set("color", rgba.ravel())
    cv.materials.append(mat)
    obj = bpy.data.objects.new(name, cv)
    collection.objects.link(obj)
    return obj


def _strand_colors(base, rng, dark_tip_prob):
    tip = np.clip(base * 1.12 + 0.03, 0, 1)
    dark = rng.random(len(base)) < dark_tip_prob
    tip[dark] = np.asarray((0.16, 0.12, 0.09)) + 0.05 * rng.random((int(dark.sum()), 1))
    return tip


def create_body_fur(q, body_V, body_F, col, M, rng):
    t0 = time.time()
    segs = flow_segments()
    N = vertex_normals(body_V, body_F)
    E = ANAT["eye_center"]
    R = ANAT["eye_radius"]
    objs = []
    for layer in ("guard", "under"):
        density = (520_000 if layer == "guard" else 700_000) * q["fur"]
        P, Nrm, area = sample_surface(body_V, body_F, int(area_of(body_V, body_F) * density * 2.6), rng, N)
        flow, L, lift, clump = flow_at(P, segs)
        keep = rng.random(len(P)) < np.clip(0.0055 / np.maximum(L, 1e-4), 0.30, 1.0)
        Pm = P.copy()
        Pm[:, 0] = np.abs(Pm[:, 0])
        d_eye = np.linalg.norm(Pm - E, axis=1)
        keep &= d_eye > R + 0.0010
        keep &= np.linalg.norm(Pm - ANAT["nose_center"], axis=1) > 0.0085
        keep &= mouth_cavity_sdf(Pm) > 0.0045                                    # no fur inside the mouth / on lips
        keep &= sd_ellipsoid(Pm, (0.0098, -0.338, 0.3612), (0.0045, 0.0080, 0.0044)) > 0.0025  # bare gums at fangs
        P, Nrm, flow, L, lift, clump = P[keep], Nrm[keep], flow[keep], L[keep], lift[keep], clump[keep]
        d_eye = d_eye[keep]
        near = d_eye < R + 0.0065
        eye_c = np.where(P[:, :1] < 0, E * np.array([-1, 1, 1]), E)
        flow[near] = normalize((P - eye_c)[near])
        L[near] = np.minimum(L[near], 0.0028)
        lift[near] = 0.05
        if layer == "under":
            L = L * 0.50
            lift = lift + 0.12
        else:
            L = L * rng.uniform(0.80, 1.30, len(L))
        k_pts = 5 if layer == "under" else 7
        pts = grow_strands(P, Nrm, flow, L, lift, rng, k_pts,
                           frizz=0.30 if layer == "under" else 0.11, gravity=0.22, bend=0.30)
        if layer == "guard":
            pts = apply_clumping(pts, clump, rng)
        base = body_color_at(P, Nrm)
        if layer == "under":
            base = np.clip(base * 0.90 + 0.04, 0, 1)
            tip = base
        else:
            dark_tipped = sstep(0.0150, 0.0300, L) + 0.4 * sstep(0.35, 0.85, Nrm[:, 2]) * sstep(0.2, 0.25, P[:, 2])
            tip = _strand_colors(base, rng, np.clip(0.55 * dark_tipped, 0, 0.8))
        r_root = np.full(len(P), 0.000090 if layer == "guard" else 0.00006)
        obj = create_curves_object(f"Fur_Body_{layer.capitalize()}", pts, r_root, r_root * 0.18,
                                   base, tip, M["fur" if layer == "guard" else "underfur"], col)
        objs.append(obj)
        log(f"  body {layer}: {len(P):,} strands")
    log(f"body fur done ({time.time() - t0:.1f}s)")
    return objs


def area_of(V, F):
    T = tri_faces(F)
    return float(0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1).sum())


def create_tail_fur(q, tail_V, tail_F, col, M, rng):
    density = 1_300_000 * q["fur"]
    P, Nrm, _ = sample_surface(tail_V, tail_F, int(area_of(tail_V, tail_F) * density), rng)
    segs = [(TAIL_POINTS[i], TAIL_POINTS[i + 1]) for i in range(len(TAIL_POINTS) - 1)]
    best_d = np.full(len(P), 1e9)
    axis_dir = np.zeros_like(P)
    t_par = np.zeros(len(P))
    cum = np.cumsum([0] + [np.linalg.norm(b - a) for a, b in segs])
    for i, (a, b) in enumerate(segs):
        ba = b - a
        t = np.clip(((P - a) @ ba) / (ba @ ba), 0, 1)
        dd = np.linalg.norm(P - (a + np.outer(t, ba)), axis=1)
        m = dd < best_d
        best_d[m] = dd[m]
        axis_dir[m] = ba / np.linalg.norm(ba)
        t_par[m] = (cum[i] + t[m] * np.linalg.norm(ba)) / cum[-1]
    L = 0.050 + 0.045 * np.sin(np.pi * np.clip(t_par * 0.9 + 0.12, 0, 1)) ** 0.7     # huge bristling brush
    L *= rng.uniform(0.8, 1.25, len(L))
    lift = 0.50 + 0.25 * t_par
    L *= 1 - 0.35 * sstep(0.75, 1.0, t_par)
    pts = grow_strands(P, Nrm, axis_dir, L, lift, rng, 9, frizz=0.11, gravity=0.25, bend=0.20)
    pts = apply_clumping(pts, np.full(len(P), 0.50), rng, frac=0.02, power=1.3)
    base = tail_color(P, Nrm)
    base = _mix(base, PAL["charcoal"], 0.92 * sstep(0.70, 0.86, t_par))       # black brush tip
    tip = _strand_colors(base, rng, 0.25 * (1 - t_par))
    r = np.full(len(P), 0.00010)
    obj = create_curves_object("Fur_Tail", pts, r, r * 0.15, base, tip, M["fur"], col)
    log(f"  tail fur: {len(P):,} strands")
    return obj


def create_ear_fur(q, ears, col, M, rng):
    objs = []
    for tag, e in ears.items():
        V, F, region, uv = e["V"], e["F"], e["region"], e["uv"]
        Nrm_all = vertex_normals(V, F)
        x, back_dir, z = ear_frame(e["side"])
        tree = kdtree.KDTree(len(V))
        for i, p in enumerate(V):
            tree.insert(p, i)
        tree.balance()
        # 1) pale fringe along the inner margins (short, so the dark inner ear stays visible)
        front_f = F[(region[F[:, 0]] < 0.5) & (region[F[:, 2]] < 0.5)]
        P, Nrm, _ = sample_surface(V, front_f, int(9000 * q["fur"]) + 600, rng, Nrm_all)
        near = np.fromiter((tree.find(p)[1] for p in P), dtype=np.int64, count=len(P))
        u, v = uv[near, 0], uv[near, 1]
        keep = (np.abs(v) > 0.62) & (u > 0.03) & (u < 0.80)
        P, Nrm, u, v = P[keep], Nrm[keep], u[keep], v[keep]
        inward = -np.sign(v)[:, None] * x[None] * e["side"]
        flow = normalize(inward * 0.7 + z[None] * 0.7 - back_dir[None] * 0.2)
        L = (0.0085 - 0.0045 * u) * sstep(0.62, 0.95, np.abs(v)) * rng.uniform(0.7, 1.2, len(u)) + 0.0015
        pts = grow_strands(P, Nrm, flow, L, np.full(len(u), 0.30), rng, 5, frizz=0.12, gravity=0.05, bend=0.25)
        base = np.tile(np.asarray((0.80, 0.72, 0.60)), (len(u), 1))
        r = np.full(len(u), 0.00005)
        objs.append(create_curves_object(f"Fur_Ear_Inner.{tag}", pts, r, r * 0.2, base, base, M["fur"], col))
        # 2) short dense fur on the back of the ear
        back_f = F[(region[F[:, 0]] > 0.5) & (region[F[:, 2]] > 0.5)]
        P, Nrm, _ = sample_surface(V, back_f, int(14000 * q["fur"]) + 800, rng, Nrm_all)
        near = np.fromiter((tree.find(p)[1] for p in P), dtype=np.int64, count=len(P))
        u, v = uv[near, 0], uv[near, 1]
        flow = np.tile(z, (len(P), 1))
        L = np.full(len(P), 0.0055) * rng.uniform(0.8, 1.25, len(P))
        pts = grow_strands(P, Nrm, flow, L, np.full(len(P), 0.12), rng, 4, frizz=0.08, gravity=0.0, bend=0.3)
        base = ear_colors(np.ones(len(P)), np.stack([u, v], 1))
        r = np.full(len(P), 0.00005)
        objs.append(create_curves_object(f"Fur_Ear_Outer.{tag}", pts, r, r * 0.2, base, base, M["fur"], col))
        # 3) long black tufts on the ear tips
        tip_f = F[(uv[F[:, 0], 0] > 0.86)]
        P, Nrm, _ = sample_surface(V, tip_f, int(1500 * q["fur"]) + 300, rng, Nrm_all)
        flow = np.tile(normalize(z + back_dir * 0.25), (len(P), 1))
        L = rng.uniform(0.026, 0.040, len(P))
        pts = grow_strands(P, Nrm, flow, L, np.full(len(P), 0.55), rng, 7, frizz=0.07, gravity=0.06, bend=0.35)
        pts = apply_clumping(pts, np.full(len(P), 0.55), rng, frac=0.04, power=1.2)
        base = np.tile(np.asarray(PAL["charcoal"]), (len(P), 1))
        r = np.full(len(P), 0.00007)
        objs.append(create_curves_object(f"Fur_Ear_Tuft.{tag}", pts, r, r * 0.15, base, base, M["fur"], col))
    return objs


def apply_head_pose(body, body_fur, head_objs):
    """Poses the rest-pose model: lowers the neck and lifts the head (glaring, stalking)."""
    me = body.data
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    me.vertices.foreach_set("co", pose_deform(V.reshape(-1, 3)).ravel().astype(np.float32))
    me.update()
    for f in body_fur:
        cv = f.data
        P = np.empty(len(cv.points) * 3, dtype=np.float32)
        cv.position_data.foreach_get("vector", P)
        cv.position_data.foreach_set("vector", pose_deform(P.reshape(-1, 3).astype(np.float64)).ravel().astype(np.float32))
        cv.update_tag()
    H = head_matrix()
    bpy.context.view_layer.update()
    for o in head_objs:
        o.matrix_world = H @ o.matrix_world


# =============================================================================
# STUDIO, LIGHTS, CAMERAS, RENDER SETTINGS
# =============================================================================

FOX_CENTER = Vector((0.0, -0.02, 0.23))


CAMERAS = {
    # name: (direction from target, distance, lens)   distances are for a 1x fox; scaled by CREATURE_SCALE
    "front_3q":     ((-0.62, -1.00, 0.22), 2.55, 85),
    "side":         ((-1.00, 0.02, 0.08), 2.75, 85),
    "front":        ((0.00, -1.00, 0.10), 2.30, 85),
    "rear_3q":      ((0.75, 1.00, 0.30), 2.60, 85),
    "face_closeup": ((-0.32, -1.00, 0.02), 0.68, 105),
}


def create_studio(col, M, S):
    bm = bmesh.new()
    width, depth, radius, height = 6.0, 3.2, 0.9, 3.0
    profile = [(y, 0.0) for y in np.linspace(-depth, 0.6, 12)]
    for a in np.linspace(0, math.pi / 2, 14)[1:]:
        profile.append((0.6 + radius * math.sin(a), radius - radius * math.cos(a)))
    profile.append((0.6 + radius, height))
    xs = np.linspace(-width / 2, width / 2, 2)
    verts = [[bm.verts.new((x, y, z)) for (y, z) in profile] for x in xs]
    for i in range(len(profile) - 1):
        bm.faces.new((verts[0][i], verts[1][i], verts[1][i + 1], verts[0][i + 1]))
    me = bpy.data.meshes.new("Studio_Backdrop")
    bm.to_mesh(me)
    bm.free()
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    obj = bpy.data.objects.new("Studio_Backdrop", me)
    obj.location = (0, 0.9 * S, 0)
    obj.scale = (S, S, S)
    col.objects.link(obj)
    me.materials.append(M["backdrop"])
    return obj


def create_lighting_rig(col, S):
    """Low-key 'boss reveal' lighting: hard warm key, cool low fill, hot desert-orange rims."""
    target = FOX_CENTER * S

    def area(name, loc, energy, size, color):
        light = bpy.data.lights.new(name, "AREA")
        light.energy = energy * S * S
        light.size = size * S
        light.color = color
        obj = bpy.data.objects.new(name, light)
        obj.location = Vector(loc) * S
        obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
        col.objects.link(obj)
        return obj

    area("Light_Key", (-1.20, -1.30, 1.30), 70.0, 0.55, (1.0, 0.90, 0.78))
    area("Light_Fill", (1.40, -1.10, 0.25), 6.0, 1.6, (0.62, 0.72, 1.0))
    area("Light_Rim", (0.60, 1.15, 0.95), 105.0, 0.6, (1.0, 0.55, 0.22))
    area("Light_Rim2", (-0.90, 0.95, 0.70), 60.0, 0.5, (1.0, 0.62, 0.30))
    world = bpy.data.worlds.get("Studio_World") or bpy.data.worlds.new("Studio_World")
    if bpy.app.version < (5, 0, 0) and not world.use_nodes:
        world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = lin((0.30, 0.26, 0.24))
    bg.inputs["Strength"].default_value = 0.10
    bpy.context.scene.world = world


def create_camera_rig(col, S):
    H = head_matrix()
    E = ANAT["eye_center"]
    face_target = (H @ Vector((0.0, E[1] - 0.012, E[2] - 0.010))) * S
    cams = {}
    for name, (direction, dist, lens) in CAMERAS.items():
        target = face_target if name == "face_closeup" else FOX_CENTER * S
        cam = bpy.data.cameras.new(f"Cam_{name}")
        cam.lens = lens
        cam.sensor_width = 36
        cam.clip_start = 0.01 * S
        cam.clip_end = 60 * S
        obj = bpy.data.objects.new(f"Cam_{name}", cam)
        obj.location = target + Vector(direction).normalized() * dist * S
        obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
        if name == "face_closeup":
            cam.dof.use_dof = True
            eye_l = H @ Vector((-E[0], E[1], E[2])) * S
            cam.dof.focus_distance = (eye_l - obj.location).length
            cam.dof.aperture_fstop = 4.0 * S
        col.objects.link(obj)
        cams[name] = obj
    bpy.context.scene.camera = cams["front_3q"]
    return cams


def configure_render(q):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = q["samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.max_bounces = 8
    scene.cycles.transparent_max_bounces = 16
    scene.cycles.use_denoising = True
    for denoiser in ("OPENIMAGEDENOISE", "OPTIX"):
        try:
            scene.cycles.denoiser = denoiser
            break
        except TypeError:
            continue
    scene.render.resolution_x, scene.render.resolution_y = q["res"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.view_settings.view_transform = "AgX"
        for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
            try:
                scene.view_settings.look = look
                break
            except TypeError:
                continue
    except TypeError:
        scene.view_settings.view_transform = "Filmic"
    try:
        scene.cycles_curves.shape = q["shape"]
        scene.cycles_curves.subdivisions = 2
    except AttributeError:
        pass


def render_views(cams, out_dir, views=None):
    scene = bpy.context.scene
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, cam in cams.items():
        if views and name not in views:
            continue
        scene.camera = cam
        scene.render.filepath = os.path.join(out_dir, f"fox_{name}.png")
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        log(f"rendered {name} ({time.time() - t0:.0f}s)")
        paths.append(scene.render.filepath)
    return paths


# =============================================================================
# ROBLOX ASSET: fur-silhouette mesh, baked maps, rig, FBX
# =============================================================================

def fur_hull_sdf_factory(segs, detail):
    def hull(P):
        d = body_sdf(P)
        near = d < 0.05
        out = d.copy()
        if near.any():
            Q = P[near]
            _, L, _, clump = flow_at(Q, segs)
            spikes = sstep(0.20, 0.85, value_noise(Q, 110.0, seed=7)) * clump * 1.4
            tuft = 0.80 + 0.18 * fbm(Q, 45.0, 2, seed=31) + spikes + detail * 0.10 * value_noise(Q, 300.0, seed=8)
            Pm = Q.copy()
            Pm[:, 0] = np.abs(Pm[:, 0])
            bare = sstep(0.003, 0.007, mouth_cavity_sdf(Pm))       # keep the mouth open
            out[near] = d[near] - L * 0.60 * tuft * bare
        return out
    return hull


def tail_hull_sdf(P):
    d = tail_sdf(P)
    t = np.clip((TAIL_POINTS[0, 2] - P[:, 2]) / (TAIL_POINTS[0, 2] - TAIL_POINTS[-1, 2]), 0, 1)
    fur = 0.036 + 0.032 * np.sin(np.pi * np.clip(t * 0.9 + 0.12, 0, 1)) ** 0.7
    tuft = 0.85 + 0.15 * fbm(P, 40.0, 2, seed=41) + 0.25 * sstep(0.3, 0.9, value_noise(P, 80.0, seed=43))
    return d - fur * tuft


def _voxel_remesh(obj, voxel):
    mod = obj.modifiers.new("Remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    apply_modifiers(obj)


def _decimate(obj, target_tris, passes=8):
    """Collapse-decimates in several passes (one pass stalls on dense surface-nets meshes)."""
    for _ in range(passes):
        tris = triangle_count(obj)
        if tris <= target_tris * 1.05:
            return
        mod = obj.modifiers.new("Decimate", "DECIMATE")
        mod.ratio = max(target_tris / tris, 0.2)
        mod.use_collapse_triangulate = True
        apply_modifiers(obj)
        if triangle_count(obj) >= tris * 0.97:
            break
    if triangle_count(obj) > target_tris * 1.25:
        raise BuildError(f"Could not decimate {obj.name} to {target_tris} triangles ({triangle_count(obj)})")


def _join(objs, name):
    bpy.context.view_layer.update()  # make sure matrix_world reflects locations set this session
    bm = bmesh.new()
    for o in objs:
        me = o.data.copy()
        me.transform(o.matrix_world)
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def create_roblox_mesh(q, col, ears, M, rng):
    """Builds the low-poly fur-silhouette game mesh (posed, one material) + its bake sources."""
    t0 = time.time()
    segs = flow_segments()
    src_col = get_collection("Roblox_BakeSources", col)
    H = np.array(head_matrix())
    box = ((-0.12, -0.42, -0.004), (0.12, 0.25, 0.50))

    def pose_points(V):
        return pose_deform(V)

    def rigid_head(V):
        return V @ H[:3, :3].T + H[:3, 3]

    # high-res fur hull -> bake source for colour and fine fur relief
    hi_V, hi_F = surface_nets(fur_hull_sdf_factory(segs, 1.0), box[0], box[1], 0.0016)
    hi_V, hi_F = _largest_component(hi_V, hi_F)
    hi_V = taubin_smooth(hi_V, hi_F, iters=2)
    hi_col = body_color_at(hi_V, vertex_normals(hi_V, hi_F))
    hi = mesh_from_numpy("Bake_FurHull", pose_points(hi_V), hi_F, src_col)
    set_color_attribute(hi.data, "fur_col", srgb_to_linear(hi_col))
    hi.data.materials.append(_bake_source_material())
    tail_hi_V, tail_hi_F = surface_nets(tail_hull_sdf, TAIL_HULL_BOX[0], TAIL_HULL_BOX[1], 0.0016)
    tail_hi_V = taubin_smooth(tail_hi_V, tail_hi_F, iters=2)
    tail_hi = mesh_from_numpy("Bake_TailHull", tail_hi_V, tail_hi_F, src_col)
    set_color_attribute(tail_hi.data, "fur_col",
                        srgb_to_linear(tail_color(tail_hi_V, vertex_normals(tail_hi_V, tail_hi_F))))
    tail_hi.data.materials.append(_bake_source_material())

    # low-poly parts, each kept in rest space first (for jaw membership), then posed
    parts, rest_parts = [], []

    def add_part(name, V_rest, F, head_rigid=False):
        V_rest = np.asarray(V_rest, dtype=np.float64)
        Vp = rigid_head(V_rest) if head_rigid else pose_points(V_rest)
        o = mesh_from_numpy(name, Vp, F, col)
        parts.append(o)
        rest_parts.append(V_rest)
        return o

    lo_V, lo_F = surface_nets(fur_hull_sdf_factory(segs, 0.0), box[0], box[1], 0.0030)
    lo_V, lo_F = _largest_component(lo_V, lo_F)
    lo_V = taubin_smooth(lo_V, lo_F, iters=6)
    body_tmp = mesh_from_numpy("tmp_body", lo_V, lo_F, col)
    _voxel_remesh(body_tmp, 0.0042)          # spiky tufts thinner than a voxel leave non-manifold edges
    _decimate(body_tmp, ROBLOX_BODY_TRIS)
    bV, bT = mesh_numpy(body_tmp)
    bpy.data.meshes.remove(body_tmp.data)
    add_part("Rbx_Body", bV, bT)
    tl_V, tl_F = surface_nets(tail_hull_sdf, TAIL_HULL_BOX[0], TAIL_HULL_BOX[1], 0.0040)
    tl_V = taubin_smooth(tl_V, tl_F, iters=6)
    tail_tmp = mesh_from_numpy("tmp_tail", tl_V, tl_F, col)
    _voxel_remesh(tail_tmp, 0.0050)
    _decimate(tail_tmp, 2400)
    tV, tT = mesh_numpy(tail_tmp)
    bpy.data.meshes.remove(tail_tmp.data)
    add_part("Rbx_Tail", tV, tT)
    for tag, e in ears.items():
        V, F, region, uv = build_ear_geometry(e["side"], 14, 8, asym=0.0 if tag == "L" else -0.03,
                                              notches=EAR_NOTCHES[tag])
        add_part(f"Rbx_Ear.{tag}", V + ANAT["ear_root"] * np.array([e["side"], 1, 1]), F, head_rigid=True)
    for side, tag in ((1, "L"), (-1, "R")):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=14, v_segments=10, radius=ANAT["eye_radius"],
                                  matrix=eye_matrix(side))
        V = np.array([v.co[:] for v in bm.verts])
        F = [[v.index for v in f.verts] for f in bm.faces]
        bm.free()
        add_part(f"Rbx_Eye.{tag}", V, _pad_faces(F), head_rigid=True)
    nV, nF = surface_nets(nose_sdf, (-0.011, -0.010, -0.010), (0.011, 0.010, 0.010), 0.0011)
    add_part("Rbx_Nose", nV + ANAT["nose_center"], nF, head_rigid=True)
    for key, (V, F) in build_teeth(segs=5, rings=3).items():
        add_part(f"Rbx_Teeth{key.capitalize()}", V, _pad_faces(F), head_rigid=True)
    tongue = _tongue_object(col, M, 10, 6, name="tmp_tongue")
    tV, tT = mesh_numpy(tongue)
    bpy.data.meshes.remove(tongue.data)
    add_part("Rbx_Tongue", tV, tT, head_rigid=True)

    part_of_vertex = np.concatenate([np.full(len(o.data.vertices), i) for i, o in enumerate(parts)])
    rest_V = np.concatenate(rest_parts)
    part_names = [o.name for o in parts]
    me = _join(parts, "Fox_Roblox")
    for o in parts:
        bpy.data.meshes.remove(o.data)
    fox = bpy.data.objects.new("Fox_Roblox", me)
    col.objects.link(fox)
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    if len(me.vertices) != len(rest_V):
        raise BuildError("Roblox mesh join changed the vertex count; cannot map skin weights")
    log(f"roblox mesh: {triangle_count(fox):,} triangles ({time.time() - t0:.1f}s)")
    return fox, part_of_vertex, part_names, rest_V, [hi, tail_hi]


def _pad_faces(F):
    """Triangulates a mixed tri/quad face list into an (M, 3) array."""
    tris = []
    for f in F:
        f = list(f)
        for i in range(1, len(f) - 1):
            tris.append((f[0], f[i], f[i + 1]))
    return np.array(tris)


def _bake_source_material():
    mat = bpy.data.materials.get("Bake_FurColor")
    if mat:
        return mat
    mat, nt, out = new_material("Bake_FurColor")
    col = attribute(nt, "fur_col")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    # streaky fur relief for the normal bake
    wave = nt.nodes.new("ShaderNodeTexNoise")
    wave.inputs["Scale"].default_value = 900.0
    wave.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.0006
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    principled(nt, out, **{"Base Color": col, "Roughness": 0.85, "Normal": bump.outputs["Normal"]})
    return mat


def uv_unwrap(obj):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")


def bake_maps(fox, sources, out_dir, size=ROBLOX_TEXTURE):
    """Bakes colour, normal and roughness from the detailed sources onto the game mesh."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    mat, nt, out = new_material("Fox_Roblox_Material")
    imgs = {}
    nodes = {}
    for key, cs in (("Color", "sRGB"), ("Normal", "Non-Color"), ("Roughness", "Non-Color")):
        img = bpy.data.images.new(f"FoxRoblox_{key}", size, size, alpha=False, float_buffer=False)
        img.colorspace_settings.name = cs
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = img
        imgs[key], nodes[key] = img, node
    fox.data.materials.clear()
    fox.data.materials.append(mat)

    bpy.ops.object.select_all(action="DESELECT")
    for s in sources:
        s.hide_render = False
        s.select_set(True)
    fox.select_set(True)
    bpy.context.view_layer.objects.active = fox
    bake = scene.render.bake
    bake.use_selected_to_active = True
    bake.cage_extrusion = 0.006
    bake.max_ray_distance = 0.014
    bake.margin = 6
    for key, btype, kwargs in (("Color", "DIFFUSE", dict(pass_filter={"COLOR"})),
                               ("Normal", "NORMAL", {}),
                               ("Roughness", "ROUGHNESS", {})):
        nt.nodes.active = nodes[key]
        t0 = time.time()
        bpy.ops.object.bake(type=btype, use_selected_to_active=True, cage_extrusion=0.006,
                            max_ray_distance=0.014, margin=6, **kwargs)
        path = os.path.join(out_dir, f"FoxRoblox_{key}.png")
        imgs[key].filepath_raw = path
        imgs[key].file_format = "PNG"
        imgs[key].save()
        log(f"baked {key} ({time.time() - t0:.0f}s) -> {path}")

    # final game material uses the baked maps
    for n in list(nt.nodes):
        if n.type not in ("TEX_IMAGE", "OUTPUT_MATERIAL"):
            nt.nodes.remove(n)
    bsdf = principled(nt, out)
    nt.links.new(nodes["Color"].outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(nodes["Roughness"].outputs["Color"], bsdf.inputs["Roughness"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nodes["Normal"].outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return imgs


def build_rig_definition():
    """Bones placed on the posed fox (neck lowered, head lifted, jaw open)."""
    H = np.array(head_matrix())

    def rigid(p):
        return tuple((H[:3, :3] @ np.asarray(p, dtype=np.float64)) + H[:3, 3])

    def posed(p):
        return tuple(pose_deform(np.asarray([p], dtype=np.float64))[0])

    chin_open = tuple(jaw_pose(np.array([[0.0, -0.342, 0.354]]))[0])
    bones = [
        ("Root", (0, 0.0, 0.0), (0, 0.0, 0.10), None),
        ("Hips", (0, 0.135, 0.252), (0, 0.060, 0.266), "Root"),
        ("Spine1", (0, 0.060, 0.266), (0, -0.030, 0.272), "Hips"),
        ("Spine2", (0, -0.030, 0.272), (0, -0.110, 0.274), "Spine1"),
        ("Chest", (0, -0.110, 0.274), posed((0, -0.150, 0.286)), "Spine2"),
        ("Neck", posed((0, -0.150, 0.286)), rigid(ATLAS_PIVOT), "Chest"),
        ("Head", rigid(ATLAS_PIVOT), rigid((0, -0.300, 0.386)), "Neck"),
        ("Jaw", rigid(JAW_HINGE), rigid(chin_open), "Head"),
    ]
    root = ANAT["ear_root"]
    for side, tag in ((1, "L"), (-1, "R")):
        x, back_dir, z = ear_frame(side)
        base = root * np.array([side, 1, 1])
        mid = base + z * 0.070
        tip = base + z * 0.142 + back_dir * 0.010
        bones.append((f"Ear.{tag}", rigid(base), rigid(mid), "Head"))
        bones.append((f"EarTip.{tag}", rigid(mid), rigid(tip), f"Ear.{tag}"))
        F = {k: _side(v, side, STANCE["front"][side]) for k, v in FRONT_LEG.items()}
        toe = (F["paw"][0], F["paw"][1] - 0.024, 0.008)
        bones += [(f"FrontUpperLeg.{tag}", F["shoulder"], F["elbow"], "Chest"),
                  (f"FrontLowerLeg.{tag}", F["elbow"], F["wrist"], f"FrontUpperLeg.{tag}"),
                  (f"FrontFoot.{tag}", F["wrist"], toe, f"FrontLowerLeg.{tag}")]
        Hl = {k: _side(v, side, STANCE["hind"][side]) for k, v in HIND_LEG.items()}
        toe = (Hl["paw"][0], Hl["paw"][1] - 0.024, 0.008)
        bones += [(f"BackUpperLeg.{tag}", Hl["hip"], Hl["stifle"], "Hips"),
                  (f"BackLowerLeg.{tag}", Hl["stifle"], Hl["hock"], f"BackUpperLeg.{tag}"),
                  (f"BackFoot.{tag}", Hl["hock"], toe, f"BackLowerLeg.{tag}")]
    parent = "Hips"
    for i in range(len(TAIL_POINTS) - 1):
        bones.append((f"Tail{i + 1}", tuple(TAIL_POINTS[i]), tuple(TAIL_POINTS[i + 1]), parent))
        parent = f"Tail{i + 1}"
    return bones


def create_rig(col, name="Fox_Rig"):
    bones = build_rig_definition()
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for bname, head, tail, parent in bones:
        b = arm.edit_bones.new(bname)
        b.head = Vector(head)
        b.tail = Vector(tail)
        if parent:
            b.parent = arm.edit_bones[parent]
            b.use_connect = (Vector(head) - arm.edit_bones[parent].tail).length < 1e-5
    bpy.ops.object.mode_set(mode="OBJECT")
    try:
        arm.display_type = "STICK"
    except TypeError:
        pass
    return obj, bones


def skin_weights(fox, rig_bones, part_of_vertex, part_names, rest_V):
    """Distance-based weights (max 4 influences) restricted to anatomically valid bones."""
    me = fox.data
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    names = [b[0] for b in rig_bones]
    Hd = np.array([b[1] for b in rig_bones], dtype=np.float64)
    Tl = np.array([b[2] for b in rig_bones], dtype=np.float64)
    BA = Tl - Hd
    l2 = np.einsum("ij,ij->i", BA, BA)
    t = np.clip(np.einsum("nkj,kj->nk", V[:, None] - Hd[None], BA) / l2[None], 0, 1)
    d = np.linalg.norm(V[:, None] - (Hd[None] + t[..., None] * BA[None]), axis=2)
    allowed = np.zeros_like(d, dtype=bool)
    idx = {n: i for i, n in enumerate(names)}
    body_bones = [n for n in names if not n.startswith(("Ear", "Tail", "Root", "Jaw"))]
    for p, pname in enumerate(part_names):
        m = part_of_vertex == p
        if pname.startswith("Rbx_Ear"):
            tag = pname[-1]
            ok = [f"Ear.{tag}", f"EarTip.{tag}", "Head"]
        elif pname.startswith(("Rbx_Eye", "Rbx_Nose", "Rbx_TeethUpper")):
            ok = ["Head"]
        elif pname.startswith(("Rbx_TeethLower", "Rbx_Tongue")):
            ok = ["Jaw"]
        elif pname == "Rbx_Tail":
            ok = ["Hips"] + [n for n in names if n.startswith("Tail")]
        else:
            ok = body_bones + ["Tail1"]
        for n in ok:
            allowed[m, idx[n]] = True
    body = part_of_vertex == part_names.index("Rbx_Body")
    jaw = body & jaw_member(rest_V)
    allowed[jaw] = False
    allowed[jaw, idx["Jaw"]] = True
    head_region = body & (rest_V[:, 1] < -0.235) & (rest_V[:, 2] > 0.33) & ~jaw
    for n in body_bones:
        if n not in ("Head", "Neck"):
            allowed[head_region, idx[n]] = False
    w = np.where(allowed, 1.0 / np.maximum(d, 0.004) ** 4, 0.0)
    order = np.argsort(-w, axis=1)[:, :4]
    top = np.take_along_axis(w, order, 1)
    top /= np.maximum(top.sum(1, keepdims=True), 1e-12)
    top[top < 0.02] = 0
    top /= np.maximum(top.sum(1, keepdims=True), 1e-12)
    groups = {n: fox.vertex_groups.new(name=n) for n in names}
    for vi in range(len(V)):
        for j in range(4):
            if top[vi, j] > 0:
                groups[names[order[vi, j]]].add([vi], float(top[vi, j]), "REPLACE")
    unweighted = int((top.sum(1) < 0.99).sum())
    if unweighted:
        raise BuildError(f"{unweighted} Roblox vertices have no skin weight")


def export_roblox_asset(fox, rig, out_dir):
    """FBX for Roblox Studio: 1 Blender unit = 1 stud (Roblox's documented Blender setup)."""
    scene = bpy.context.scene
    os.makedirs(out_dir, exist_ok=True)
    s = ROBLOX_STUDS_PER_METER * CREATURE_SCALE
    old_scale, old_len = scene.unit_settings.scale_length, scene.unit_settings.length_unit
    rig.scale = (s, s, s)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    fox.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    tri = fox.modifiers.new("Triangulate", "TRIANGULATE")
    tri.keep_custom_normals = True
    try:
        scene.unit_settings.scale_length = 0.01
        scene.unit_settings.length_unit = "CENTIMETERS"
        path = os.path.join(out_dir, "DesertFox_Roblox.fbx")
        bpy.ops.export_scene.fbx(
            filepath=path, use_selection=True, object_types={"ARMATURE", "MESH"},
            apply_unit_scale=True, apply_scale_options="FBX_SCALE_UNITS", global_scale=1.0,
            axis_forward="-Z", axis_up="Y", use_mesh_modifiers=True, mesh_smooth_type="FACE",
            add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
            use_armature_deform_only=False, bake_anim=False, path_mode="COPY", embed_textures=True)
    finally:
        scene.unit_settings.scale_length = old_scale
        scene.unit_settings.length_unit = old_len
        fox.modifiers.remove(tri)
        rig.scale = (1.0 / s, 1.0 / s, 1.0 / s)
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return path


# =============================================================================
# VALIDATION
# =============================================================================

def validate(report, objs):
    problems = []
    body = objs["body"]
    V = np.empty(len(body.data.vertices) * 3)
    body.data.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    if V[:, 2].min() > 0.002 or V[:, 2].min() < -0.002:
        problems.append(f"paws do not touch the ground (min z = {V[:, 2].min():.4f})")
    feet = [V[(np.abs(V[:, 0] - s * c[0]) < 0.02) & (np.abs(V[:, 1] - c[1]) < 0.03) & (V[:, 2] < 0.004)]
            for s in (1, -1) for c in (FRONT_LEG["paw"], HIND_LEG["paw"])]
    if any(len(f) == 0 for f in feet):
        problems.append("at least one paw has no ground contact")
    report["paw_contacts"] = [len(f) for f in feet]
    report["body_size_m"] = [float(v) for v in np.round(V.max(0) - V.min(0), 3)]
    for key in ("ear_L", "ear_R", "eye_L", "eye_R", "nose", "tail"):
        if objs.get(key) is None:
            problems.append(f"missing part: {key}")
    if problems:
        raise BuildError("Validation failed: " + "; ".join(problems))
    report["validation"] = "passed"


# =============================================================================
# MAIN
# =============================================================================

def clear_scene():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.hair_curves, bpy.data.cameras,
                 bpy.data.lights, bpy.data.images, bpy.data.armatures):
        for block in list(coll):
            coll.remove(block)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def main(cfg=None):
    cfg = cfg or _parse_args()
    q = QUALITY[cfg["quality"]]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    S = CREATURE_SCALE
    log(f"Blender {bpy.app.version_string}, quality '{cfg['quality']}', output {out_dir}")
    clear_scene()

    root_col = get_collection("DesertFox")
    hq = get_collection("Fox_HQ", root_col)
    anat_col = get_collection("Fox_Anatomy", hq)
    fur_col = get_collection("Fox_Fur", hq)
    studio_col = get_collection("Studio", root_col)
    cam_col = get_collection("Cameras", root_col)

    place_features()
    M = create_materials()

    # 1) build everything in the rest pose (head level, jaw already open)
    body, body_V, body_F = create_body(q, anat_col, M)
    tail, tail_V, tail_F = create_tail(q, anat_col, M)
    nose = create_nose(anat_col, M)
    ears = create_ears(anat_col, M)
    eyes = create_eyes(anat_col, M)
    mouth = create_teeth_and_tongue(anat_col, M)
    claws, pads = create_claws_and_pads(anat_col, M)

    root = bpy.data.objects.new("DesertFox_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.1
    hq.objects.link(root)
    head_objs = [nose] + [e["obj"] for e in ears.values()] + [p for e in eyes.values() for p in e.values()] + mouth
    for o in [body, tail, claws, pads] + head_objs:
        parent_keep(o, root)

    fur = create_body_fur(q, body_V, body_F, fur_col, M, rng)
    tail_fur = create_tail_fur(q, tail_V, tail_F, fur_col, M, rng)
    ear_fur = create_ear_fur(q, ears, fur_col, M, rng)
    for f in fur:
        parent_keep(f, body)
    parent_keep(tail_fur, tail)
    for f in ear_fur:
        parent_keep(f, ears["L" if f.name.endswith(".L") else "R"]["obj"])

    report = {
        "blender": bpy.app.version_string,
        "quality": cfg["quality"],
        "fur_strands": {o.name: len(o.data.curves) for o in fur + [tail_fur] + ear_fur},
    }
    report["fur_strands_total"] = sum(report["fur_strands"].values())
    validate(report, {"body": body, "ear_L": ears["L"]["obj"], "ear_R": ears["R"]["obj"],
                      "eye_L": eyes["L"]["eye"], "eye_R": eyes["R"]["eye"], "nose": nose, "tail": tail})

    # 2) pose: neck lowered, head lifted into a glare
    apply_head_pose(body, fur, head_objs)
    report["hq_triangles"] = {o.name: triangle_count(o) for o in [body, tail, nose, claws, pads] + mouth +
                              [e["obj"] for e in ears.values()]}

    # 3) Roblox game asset
    if cfg["build_roblox"]:
        rbx_col = get_collection("Fox_Roblox", root_col)
        fox, part_of_vertex, part_names, rest_V, sources = create_roblox_mesh(q, rbx_col, ears, M, rng)
        uv_unwrap(fox)
        rbx_dir = os.path.join(out_dir, "roblox")
        os.makedirs(rbx_dir, exist_ok=True)
        hq_parts = [e["obj"] for e in ears.values()] + [e["eye"] for e in eyes.values()] + [nose] + mouth
        bake_maps(fox, sources + hq_parts, rbx_dir)
        rig, bones = create_rig(rbx_col)
        skin_weights(fox, bones, part_of_vertex, part_names, rest_V)
        mod = fox.modifiers.new("Armature", "ARMATURE")
        mod.object = rig
        fox.parent = rig
        for s_obj in sources:
            s_obj.hide_render = True
            s_obj.hide_viewport = True
        report["roblox_triangles"] = triangle_count(fox)
        report["roblox_bones"] = len(bones)
        report["roblox_fbx"] = export_roblox_asset(fox, rig, rbx_dir)
        report["roblox_size_studs"] = [round(float(v), 2) for v in
                                       (np.array(fox.dimensions) * ROBLOX_STUDS_PER_METER * S)]
        rig.scale = (S, S, S)
        rig.location.x = 1.0 * S       # stands next to the HQ fox in the .blend (hidden in renders)
        rbx_col.hide_render = True

    # 4) giant fantasy scale, studio, lights, cameras
    root.scale = (S, S, S)
    create_studio(studio_col, M, S)
    create_lighting_rig(studio_col, S)
    cams = create_camera_rig(cam_col, S)
    configure_render(q)

    blend_path = os.path.join(out_dir, "DesertFox.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False, compress=True)
    report["blend"] = blend_path

    if cfg["render"]:
        report["renders"] = render_views(cams, os.path.join(out_dir, "renders"), cfg["views"])

    report["seconds"] = round(time.time() - t_start, 1)
    log("REPORT " + repr(report))
    return report


if __name__ == "__main__":
    main()
