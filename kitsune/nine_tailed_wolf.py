"""
NINE-TAILED FLAME WOLF - Roblox model generator for Blender 5.2
================================================================

Builds the nine-tailed flame fox (after a nine-tailed wolf statue) as a Roblox-ready model, all
from code:

  * a lean wolf-like fox crouched to pounce: head low and snarling (open jaw, teeth), big ears
    pinned back, shaggy neck ruff, carved fur relief (signed-distance-field sculpt -> surface-nets mesh)
  * fire-fox colours: orange coat with a red saddle, cream chest, black stockings and ear tips,
    fur ridges burning yellow at the tips; big amber eyes with slit pupils (their own part)
  * nine feathery tails fanned out behind it, red at the root through orange to yellow tips, built
    from layered locks whose lips catch the light
  * a flame mane over the neck and shoulders, orange fire under the chest and a big flame
    swirl beside the hind leg (sculpted flame tongues merged into one mass, like the statue)
  * no stand: the wolf stands on the ground plane (z = 0); the ground fire is its own part
  * Roblox export: one FBX with several MeshParts (each well under the 20,000-triangle limit),
    1024 px colour + normal maps, previews rendered from the exported meshes

Needs only stock Blender 5.2 (or the `bpy` 5.2 Python module): bpy, bmesh and numpy.

HOW TO RUN
  Headless:  blender --background --python-exit-code 1 --python nine_tailed_wolf.py -- --out <folder>
  bpy module: python nine_tailed_wolf.py --out <folder>
  Options:   --render          render the preview views of the exported game model
             --views ref,side  only render some views
             --quick           coarse meshes, no bake (shape check only)
             --preview-only    only re-render the previews from an existing <out>/roblox/FlameWolf.fbx
"""

import math
import os
import sys
import time

import bpy  # noqa: E402
import bmesh
import numpy as np

# =============================================================================
# SETTINGS
# =============================================================================

STUDS_PER_UNIT = 12.0        # model units -> Roblox studs (wolf shoulder ~7 studs, tails ~16 studs high)
TEX = 1024                   # texture size (Roblox maximum)
BODY_TRIS = 18500
HEAD_TRIS = 15000
MANE_TRIS = 13000
FIRE_TRIS = 15000
SEED = 7

VIEWS = {
    #            camera location            look-at             lens
    "ref":      ((2.25, 2.55, 1.05),     (0.02, 0.0, 0.50),   50),
    "side":     ((0.05, 3.6, 0.75),      (-0.08, 0.0, 0.50),  50),
    "front":    ((3.5, 0.35, 0.80),      (0.0, 0.0, 0.50),    50),
    "rear_3q":  ((-2.6, 2.3, 1.35),      (-0.15, 0.0, 0.55),  50),
    "top":      ((0.3, 1.1, 3.6),        (-0.1, 0.0, 0.45),   45),
    "face":     ((1.35, 0.95, 0.55),     (0.66, 0.12, 0.36),  70),
}


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    opts = {"out": os.path.join(os.path.expanduser("~"), "FlameWolf_Output"),
            "render": False, "views": None, "quick": False, "preview_only": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            opts["out"] = argv[i + 1]
            i += 1
        elif a == "--render":
            opts["render"] = True
        elif a == "--views":
            opts["views"] = argv[i + 1].split(",")
            i += 1
        elif a == "--quick":
            opts["quick"] = True
        elif a == "--preview-only":
            opts["preview_only"] = True
        i += 1
    return opts


T_START = time.time()


def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


class BuildError(RuntimeError):
    pass


# =============================================================================
# NUMPY MATH
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
    """Positive angle tips +X towards -Z (nose down)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rodrigues(v, axis, ang):
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return (v * math.cos(ang) + np.cross(axis, v) * math.sin(ang)
            + axis * float(axis @ v) * (1 - math.cos(ang)))


def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761 & 0x7FFFFFFF)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0xFFFFFF)


def value_noise(Q, seed=0):
    """Smooth 3D value noise in [-1, 1] at integer lattice of Q."""
    i = np.floor(Q).astype(np.int64)
    f = Q - i
    u = f * f * (3.0 - 2.0 * f)
    out = np.zeros(len(Q))
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1.0 - u[:, 0]
        for dy in (0, 1):
            wy = u[:, 1] if dy else 1.0 - u[:, 1]
            for dz in (0, 1):
                wz = u[:, 2] if dz else 1.0 - u[:, 2]
                out += wx * wy * wz * _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out * 2.0 - 1.0


def fbm(Q, octaves=3, seed=0, gain=0.5):
    total, amp, norm = np.zeros(len(Q)), 1.0, 0.0
    for o in range(octaves):
        total += amp * value_noise(Q * (2.0 ** o), seed + 17 * o)
        norm += amp
        amp *= gain
    return total / norm


# =============================================================================
# SDF PRIMITIVES
# =============================================================================


def sd_sphere(P, c, r):
    return np.linalg.norm(P - np.asarray(c), axis=1) - r


def sd_ellipsoid(P, c, r, rot=None):
    q = P - np.asarray(c, dtype=np.float64)
    if rot is not None:
        q = q @ rot
    r = np.asarray(r, dtype=np.float64)
    k0 = np.linalg.norm(q / r, axis=1)
    k1 = np.linalg.norm(q / (r * r), axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_round_cone(P, a, b, r1, r2):
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


def sd_flat_cone(P, a, b, r1, r2, flat_dir, flat):
    """Round cone squashed along flat_dir (thickness factor `flat`), e.g. ears."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = np.asarray(flat_dir, dtype=np.float64)
    n = n / np.linalg.norm(n)
    q = P - a
    along = q @ n
    Q = a + q + np.outer(along * (1.0 / flat - 1.0), n)
    return sd_round_cone(Q, a, b, r1, r2) * flat


def sd_box(P, c, half, r=0.0):
    q = np.abs(P - np.asarray(c)) - (np.asarray(half) - r)
    return (np.linalg.norm(np.maximum(q, 0.0), axis=1)
            + np.minimum(np.max(q, axis=1), 0.0) - r)


def sd_cyl_z(P, c, r, z0, z1):
    d_r = np.linalg.norm(P[:, :2] - np.asarray(c[:2]), axis=1) - r
    d_z = np.maximum(z0 - P[:, 2], P[:, 2] - z1)
    return np.minimum(np.maximum(d_r, d_z), 0.0) + np.linalg.norm(
        np.stack([np.maximum(d_r, 0.0), np.maximum(d_z, 0.0)], 1), axis=1)


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


# =============================================================================
# ANATOMY  (model units; wolf faces +X, its left side is +Y = camera side, ground z = 0)
# =============================================================================

HEAD_ORIGIN = np.array([0.600, 0.085, 0.455])
HEAD_R = rot_z(math.radians(24)) @ rot_y(math.radians(20))     # turned towards the camera, nose down
HEAD_S = 1.22                                                  # head scale (local units -> model units)
JAW_HINGE = np.array([-0.015, 0.0, -0.040])                    # head-local
JAW_OPEN = math.radians(30)
JAW_R = rot_y(JAW_OPEN)

NECK_A = np.array([0.20, 0.0, 0.600])
TAIL_ROOT = np.array([-0.545, 0.0, 0.665])

# legs: joint chains, radii, and the ground height under each paw
LEGS = {
    "front_L": dict(pts=[(0.20, 0.115, 0.52), (0.340, 0.150, 0.32), (0.555, 0.172, 0.105), (0.632, 0.176, 0.044)],
                    r=[0.078, 0.052, 0.038, 0.034], paw=(0.655, 0.176), ground=0.0),
    "front_R": dict(pts=[(0.18, -0.118, 0.52), (0.165, -0.150, 0.32), (0.245, -0.152, 0.100), (0.295, -0.152, 0.044)],
                    r=[0.076, 0.050, 0.037, 0.033], paw=(0.318, -0.152), ground=0.0),
    "hind_L": dict(pts=[(-0.38, 0.120, 0.57), (-0.200, 0.152, 0.34), (-0.470, 0.142, 0.150), (-0.390, 0.142, 0.044)],
                   r=[0.098, 0.057, 0.037, 0.032], paw=(-0.368, 0.142), ground=0.0),
    "hind_R": dict(pts=[(-0.41, -0.120, 0.57), (-0.240, -0.150, 0.335), (-0.510, -0.142, 0.150), (-0.430, -0.142, 0.044)],
                   r=[0.096, 0.055, 0.036, 0.032], paw=(-0.408, -0.142), ground=0.0),
}
PAW_R = (0.070, 0.050, 0.036)


NOSE_C = np.array([0.226, 0.0, 0.004])
NOSE_R = 0.024
EYE_SOCKET = np.array([0.080, 0.062, 0.032])     # head-local, left side (+Y)
EYE_CENTER = np.array([0.077, 0.057, 0.032])
EYE_RADIUS = 0.0195
EYE_DIR = np.array([0.55, 0.80, 0.06])           # where the left eye looks (head-local)
EAR_ROOT = np.array([-0.030, 0.052, 0.072])
EAR_TIP = np.array([-0.130, 0.105, 0.195])


def ear_sdf(Ph, s):
    return sd_flat_cone(Ph, EAR_ROOT * np.array([1, s, 1]), EAR_TIP * np.array([1, s, 1]),
                        0.058, 0.006, (0.15, 0.85 * s, -0.45), 0.34)


def head_to_world(p):
    return HEAD_ORIGIN + (np.asarray(p) * HEAD_S) @ HEAD_R.T


def world_to_head(P):
    return ((P - HEAD_ORIGIN) @ HEAD_R) / HEAD_S


def jaw_to_head(p):
    """Rest-pose jaw point -> opened jaw point (head-local)."""
    return JAW_HINGE + (np.asarray(p) - JAW_HINGE) @ JAW_R.T


def head_to_jaw(Ph):
    return JAW_HINGE + (Ph - JAW_HINGE) @ JAW_R


def mouth_cavity(Ph):
    half = rot_y(JAW_OPEN * 0.5)
    q = JAW_HINGE + (Ph - JAW_HINGE) @ half
    return smin(sd_ellipsoid(q, (0.115, 0.0, -0.058), (0.115, 0.034, 0.024)),
                sd_ellipsoid(q, (0.030, 0.0, -0.050), (0.06, 0.03, 0.02)), 0.02)


def head_sdf(Ph):
    """Head in head-local coordinates (x along the snout, z up); distances in local units."""
    cran = sd_ellipsoid(Ph, (0.0, 0.0, 0.018), (0.120, 0.096, 0.086))
    d = cran
    for s in (1, -1):
        d = smin(d, sd_ellipsoid(Ph, (0.070, 0.044 * s, 0.054), (0.042, 0.034, 0.028)), 0.03)       # brow
        d = smin(d, sd_ellipsoid(Ph, (0.020, 0.080 * s, -0.014), (0.066, 0.044, 0.050)), 0.04)      # cheek
    muzzle = sd_round_cone(Ph * np.array([1.0, 0.95, 0.98]), (0.050, 0.0, -0.004), (0.195, 0.0, -0.012), 0.064, 0.033)
    d = smin(d, muzzle, 0.05)
    d = smin(d, sd_ellipsoid(Ph, (0.160, 0.0, 0.022), (0.075, 0.028, 0.015)), 0.03)                # snout ridge
    d = smin(d, sd_sphere(Ph, NOSE_C, NOSE_R), 0.012)                                             # nose pad
    for k, x in enumerate((0.090, 0.112, 0.134)):                                                   # snarl wrinkles
        d = smin(d, sd_round_cone(Ph, (x, -0.038, 0.020 + 0.004 * k), (x + 0.012, 0.038, 0.020 + 0.004 * k), 0.0075, 0.0075), 0.012)
    Pj = head_to_jaw(Ph)
    jaw = sd_round_cone(Pj * np.array([1.0, 1.15, 1.0]), (0.005, 0.0, -0.060), (0.178, 0.0, -0.064), 0.040, 0.020)
    d = smin(d, jaw, 0.035)
    d = smax(d, -mouth_cavity(Ph), 0.012)
    d = np.minimum(d, sd_ellipsoid(Pj, (0.095, 0.0, -0.058), (0.09, 0.026, 0.010)))                # tongue
    for s in (1, -1):
        d = smax(d, -sd_sphere(Ph, EYE_SOCKET * np.array([1, s, 1]), 0.0225), 0.008)                # socket
        d = smin(d, ear_sdf(Ph, s), 0.025)                                                          # ear
    return d


def leg_sdf(P, leg):
    pts, r = [np.array(p) for p in leg["pts"]], leg["r"]
    d = sd_round_cone(P, pts[0], pts[1], r[0], r[1])
    d = smin(d, sd_round_cone(P, pts[1], pts[2], r[1], r[2]), 0.03)
    d = smin(d, sd_round_cone(P, pts[2], pts[3], r[2], r[3]), 0.02)
    px, py = leg["paw"]
    gz = leg["ground"]
    c = np.array([px, py, gz + PAW_R[2] * 0.8])
    paw = sd_ellipsoid(P, c, PAW_R)
    for k, dy in enumerate((-0.032, -0.012, 0.012, 0.032)):
        paw = smin(paw, sd_sphere(P, c + np.array([0.052 - 0.011 * abs(k - 1.5), dy, -0.006]), 0.021), 0.012)
    paw = smax(paw, gz - P[:, 2], 0.008)                       # flat pads on the ground
    return smin(d, paw, 0.03)


def torso_sdf(P):
    d = sd_ellipsoid(P, (0.13, 0.0, 0.48), (0.235, 0.155, 0.185), rot_y(math.radians(-8)))      # deep chest
    d = smin(d, sd_ellipsoid(P, (0.08, 0.0, 0.605), (0.21, 0.135, 0.100)), 0.08)               # withers
    for s in (1, -1):
        d = smin(d, sd_ellipsoid(P, (0.165, 0.100 * s, 0.560), (0.130, 0.064, 0.140), rot_y(math.radians(-25))), 0.05)
        d = smin(d, sd_ellipsoid(P, (-0.37, 0.100 * s, 0.535), (0.155, 0.074, 0.175), rot_y(math.radians(20))), 0.06)
    d = smin(d, sd_ellipsoid(P, (-0.14, 0.0, 0.585), (0.22, 0.118, 0.102)), 0.10)              # tucked waist
    d = smin(d, sd_ellipsoid(P, (-0.40, 0.0, 0.612), (0.165, 0.130, 0.132)), 0.08)             # hips
    d = smin(d, sd_sphere(P, TAIL_ROOT, 0.078), 0.06)
    return d


def neck_end():
    return head_to_world((-0.065, 0.0, -0.012))


def ruff_center():
    return NECK_A + (neck_end() - NECK_A) * 0.70 + np.array([0.0, 0.0, -0.01])


def _ruff_tufts():
    """Big shaggy tufts round the neck and throat, swept back from the face."""
    rng = np.random.default_rng(SEED + 5)
    ne = neck_end()
    axis = normalize(NECK_A - ne)                       # pointing back along the neck
    side = np.array([0.0, 1.0, 0.0])
    up = normalize(np.cross(axis, side))
    if up[2] < 0:
        up = -up
    side = normalize(np.cross(up, axis))
    tufts = []
    for i in range(26):
        t = rng.uniform(0.62, 0.98)
        ang = rng.uniform(-math.pi, math.pi)
        radial = up * math.cos(ang) + side * math.sin(ang)
        root = NECK_A + (ne - NECK_A) * t + radial * 0.10
        tip = root + normalize(axis * 1.0 + radial * 0.55 + rng.normal(0, 0.15, 3)) * rng.uniform(0.09, 0.15)
        tufts.append((root, tip, rng.uniform(0.026, 0.036)))
    return tufts


RUFF_TUFTS = None


def body_sdf(P, fur=True):
    global RUFF_TUFTS
    if RUFF_TUFTS is None:
        RUFF_TUFTS = _ruff_tufts()
    d = torso_sdf(P)
    for root, tip, r in RUFF_TUFTS:
        d = smin(d, sd_round_cone(P, root, tip, r, 0.004), 0.025)
    ne = neck_end()
    d = smin(d, sd_round_cone(P, NECK_A, ne, 0.140, 0.105), 0.08)
    d = smin(d, sd_ellipsoid(P, ruff_center(), (0.125, 0.165, 0.165)), 0.06)                    # neck ruff
    d = smin(d, head_sdf(world_to_head(P)) * HEAD_S, 0.05)
    for leg in LEGS.values():
        d = smin(d, leg_sdf(P, leg), 0.045)
    if fur:
        d = d - fur_displacement(P)
    return d


# --- carved fur ----------------------------------------------------------------
#
# The coat is sculpted as overlapping fur locks (like shingles): each lock swells towards a
# V-shaped tip that overhangs the next one, neighbouring locks are separated by creases, and fine
# strand grooves run inside each lock. The locks flow backwards over the body, head and ruff and
# downwards on the legs, chest front and rump. The colour is computed from the same field, so the
# paint follows the carving instead of sitting on top of it.


def worley2(X, Y, seed):
    """2D cellular noise: distance to the nearest / second-nearest feature point and a random
    value per cell."""
    xi = np.floor(X).astype(np.int64)
    yi = np.floor(Y).astype(np.int64)
    F1 = np.full(X.shape, 9.0)
    F2 = np.full(X.shape, 9.0)
    cid = np.zeros(X.shape)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = xi + dx, yi + dy
            k = np.zeros_like(cx)
            px = cx + 0.1 + 0.8 * _hash3(cx, cy, k, seed)
            py = cy + 0.1 + 0.8 * _hash3(cx, cy, k + 1, seed)
            d = np.hypot(X - px, Y - py)
            h = _hash3(cx, cy, k + 2, seed)
            closer = d < F1
            F2 = np.where(closer, F1, np.minimum(F2, d))
            cid = np.where(closer, h, cid)
            F1 = np.where(closer, d, F1)
    return F1, F2, cid


def _lock_layer(A, X, Y, seed):
    """Locks for strands running towards +A; X is the cross coordinate in lock widths, Y spreads
    the cells. Returns height, phase (1 = at the lock's tip), cell value, crease, strands."""
    F1, F2, cid = worley2(X, Y, seed)
    t = A + cid * 1.7 + 0.45 * F1                  # lock centres reach furthest: pointed tips
    phase = t - np.floor(t)
    crease = sstep(0.0, 0.25, F2 - F1)             # groove between neighbouring locks
    strands = 0.5 + 0.5 * np.sin(2 * math.pi * (X * 2.5 + 0.35 * cid))
    h = phase ** 1.4 * (1.0 - 0.8 * F1) * (0.35 + 0.65 * crease)
    return h, phase, cid, crease, strands


class NormalField:
    """Surface normal of the un-furred body anywhere near it (gradient of a coarse SDF grid,
    trilinearly interpolated); used to orient the fur locks."""

    def __init__(self, sdf, bmin, bmax, h):
        self.bmin = np.asarray(bmin, dtype=np.float64)
        self.h = h
        shape = tuple(int(v) for v in np.ceil((np.asarray(bmax) - self.bmin) / h).astype(int) + 1)
        g = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
        P = self.bmin + g * h
        S = np.concatenate([sdf(P[i:i + 400_000]) for i in range(0, len(P), 400_000)]).reshape(shape)
        G = np.stack(np.gradient(S, h), -1)
        self.G = G / np.maximum(np.linalg.norm(G, axis=-1, keepdims=True), 1e-9)
        self.shape = np.array(shape)

    def __call__(self, P):
        q = (P - self.bmin) / self.h
        i = np.clip(np.floor(q).astype(np.int64), 0, self.shape - 2)
        f = np.clip(q - i, 0.0, 1.0)
        out = np.zeros((len(P), 3))
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    out += (wx * wy * wz)[:, None] * self.G[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        return normalize(out)


NORMALS = None


def body_normals(P):
    global NORMALS
    if NORMALS is None:
        t0 = time.time()
        NORMALS = NormalField(lambda Q: body_sdf(Q, fur=False), BODY_BOX[0], BODY_BOX[1], 0.009)
        log(f"fur orientation field ready ({time.time() - t0:.0f}s)")
    return NORMALS(P)


def _neck_frame():
    x = normalize(neck_end() - NECK_A)               # towards the head; strands run the other way
    y = normalize(np.cross(np.array([0.0, 0.0, 1.0]), x))
    z = np.cross(x, y)
    return np.stack([x, y, z])


FRAME_BACK = np.eye(3)                                             # strands run towards -X
FRAME_DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # strands run towards -Z


def _region_weights(P, N):
    Ph = world_to_head(P)
    w_head = sstep(0.17, 0.11, np.linalg.norm(Ph - np.array([0.03, 0.0, 0.01]), axis=1))
    w_ruff = sstep(0.27, 0.13, np.linalg.norm(P - ruff_center(), axis=1)) * (1 - w_head)
    rest = np.clip(1 - w_head - w_ruff, 0, 1)
    w_leg = sstep(0.42, 0.28, P[:, 2]) * rest
    body = rest - w_leg
    facing_x = np.abs(N[:, 0]) ** 2                    # chest front / rump: strands run down there
    return dict(head=w_head, ruff=w_ruff, back=body * (1 - facing_x), down=w_leg + body * facing_x)


# (frame, strand length, lock width, seed) per region; sizes in model units
FUR_LAYERS = {
    "back": (FRAME_BACK, 0.080, 0.022, 31),
    "down": (FRAME_DOWN, 0.055, 0.017, 37),
    "ruff": (None, 0.130, 0.036, 41),
    "head": (None, 0.045, 0.015, 43),
}


def fur_field(P, N=None):
    """Blended lock field: height in [0, ~1.1] plus colouring channels."""
    if N is None:
        N = body_normals(P)
    W = _region_weights(P, N)
    n = len(P)
    acc = {k: np.zeros(n) for k in ("h", "phase", "cid", "crease", "strands", "w")}
    for region, (frame, L, Wd, seed) in FUR_LAYERS.items():
        w = W[region]
        m = w > 0.02
        if not m.any():
            continue
        if region == "head":
            R = HEAD_R                                        # head-local: x along the snout
            Pl = world_to_head(P[m]) * HEAD_S
            Nl = N[m] @ R
        else:
            R = _neck_frame() if region == "ruff" else frame
            Pl = P[m] @ R.T
            Nl = N[m] @ R.T
        A = -Pl[:, 0] / L
        # tri-planar over the two cross axes: where the surface faces local Y the locks are laid
        # out along local Z, and the other way round
        wy, wz = Nl[:, 1] ** 4, Nl[:, 2] ** 4
        s = np.maximum(wy + wz, 1e-6)
        wy, wz = wy / s, wz / s
        out = np.zeros((5, int(m.sum())))
        for wt, X, Y, sd in ((wy, Pl[:, 2], Pl[:, 1], seed), (wz, Pl[:, 1], Pl[:, 2], seed + 1)):
            sel = wt > 0.02
            if sel.any():
                layer = _lock_layer(A[sel], X[sel] / Wd, Y[sel] / Wd * 0.5, sd)
                for k in range(5):
                    out[k, sel] += wt[sel] * layer[k]
        for k, key in enumerate(("h", "phase", "cid", "crease", "strands")):
            acc[key][m] += w[m] * out[k]
        acc["w"][m] += w[m]
    wsum = np.maximum(acc.pop("w"), 1e-6)
    return {k: v / wsum for k, v in acc.items()}


def fur_amplitude(P):
    d_ruff = np.linalg.norm(P - ruff_center(), axis=1)
    amp = 0.010 + 0.022 * sstep(0.26, 0.10, d_ruff)
    amp = amp * (1 - 0.45 * sstep(0.42, 0.28, P[:, 2]))                                          # shorter on legs
    Ph = world_to_head(P)
    cheek = sstep(0.10, 0.03, np.linalg.norm((Ph - np.array([-0.005, 0.0, -0.01])) * np.array([1.0, 0.5, 1.0]), axis=1))
    amp = amp + 0.006 * cheek
    head = sstep(0.17, 0.11, np.linalg.norm(Ph - np.array([0.03, 0.0, 0.01]), axis=1))
    amp = amp * (1 - 0.55 * head * (1 - cheek))                                                 # short fur on the face
    smooth_face = sstep(0.05, 0.12, np.linalg.norm(Ph - np.array([0.20, 0.0, -0.02]), axis=1))   # muzzle stays sleek
    smooth_face *= sstep(0.0, 0.08, np.linalg.norm(Ph - np.array([0.08, 0.0, 0.03]), axis=1))     # around the eyes
    smooth_face *= sstep(0.02, 0.06, mouth_cavity(Ph))                                          # lips and gums
    paws = sstep(0.04, 0.12, P[:, 2])
    return amp * smooth_face * paws


def fur_displacement(P):
    amp = fur_amplitude(P)
    out = np.zeros(len(P))
    m = amp > 1e-5
    if m.any():
        f = fur_field(P[m])
        out[m] = amp[m] * (f["h"] + 0.12 * f["strands"] - 0.45)
    return out



# =============================================================================
# SURFACE NETS MESHER
# =============================================================================


def sample_grid(sdf, bmin, h, shape, block=16, chunk=1_000_000):
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
            Pp = bmin + np.concatenate(pts).astype(np.float64) * h
            vals = sdf(Pp).astype(np.float32)
            o = 0
            for bb, shp in slices:
                n = int(np.prod(shp))
                S[lo[bb, 0]:hi[bb, 0], lo[bb, 1]:hi[bb, 1], lo[bb, 2]:hi[bb, 2]] = vals[o:o + n].reshape(shp)
                o += n
            batch, count = [], 0
    return S


def surface_nets(sdf, bmin, bmax, h):
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    shape = tuple(int(v) for v in np.ceil((bmax - bmin) / h).astype(int) + 1)
    S = sample_grid(sdf, bmin, h, shape)
    inside = S < 0
    if inside[0].any() or inside[-1].any() or inside[:, 0].any() or inside[:, -1].any() \
            or inside[:, :, 0].any() or inside[:, :, -1].any():
        raise BuildError("shape touches the sampling box; enlarge the bounds")
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
        raise BuildError("surface nets produced an invalid face")
    return V, F


def mesh_edges(F):
    k = F.shape[1]
    e = np.concatenate([np.stack([F[:, i], F[:, (i + 1) % k]], 1) for i in range(k)])
    return np.concatenate([e, e[:, ::-1]])


def taubin_smooth(V, F, iters=4, lam=0.5, mu=-0.53):
    E = mesh_edges(F)
    deg = np.bincount(E[:, 0], minlength=len(V)).astype(np.float64)
    deg[deg == 0] = 1
    V = V.copy()
    for _ in range(iters):
        for f in (lam, mu):
            avg = np.stack([np.bincount(E[:, 0], weights=V[E[:, 1], d], minlength=len(V)) for d in range(3)], 1) / deg[:, None]
            V = V + f * (avg - V)
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


def largest_component(V, F):
    """Keeps the biggest connected piece (drops stray specks from the noise)."""
    n = len(V)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]])
    for a, b in E:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    keep_root = np.bincount(roots).argmax()
    keep = roots == keep_root
    remap = np.full(n, -1)
    remap[keep] = np.arange(int(keep.sum()))
    Fk = F[np.all(keep[F], axis=1)]
    return V[keep], remap[Fk]


# =============================================================================
# SWEPT TUBES (tails and flames)
# =============================================================================


def transport_frames(path, up_hint=(0.0, 0.0, 1.0)):
    T = np.gradient(path, axis=0)
    T = normalize(T)
    up = np.asarray(up_hint, dtype=np.float64)
    N0 = np.cross(T[0], up)
    if np.linalg.norm(N0) < 1e-6:
        N0 = np.cross(T[0], np.array([1.0, 0.0, 0.0]))
    N = [normalize(N0)]
    for i in range(1, len(path)):
        axis = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(axis)
        n = N[-1]
        if s > 1e-9:
            ang = math.atan2(s, float(T[i - 1] @ T[i]))
            n = rodrigues(n, axis / s, ang)
        n = n - T[i] * float(n @ T[i])
        N.append(normalize(n))
    N = np.array(N)
    B = np.cross(T, N)
    return T, N, B


def tube_mesh(path, rx, ry, n_around, frame_roll=0.0, up_hint=(0, 0, 1), offset=None):
    """Closed tube along `path`. rx, ry: (K, n_around) radii along the frame's N and B axes.
    Returns V, F (tris), UV per vertex (u around, v along). Last ring collapses to a tip vertex."""
    K = len(path)
    T, N, B = transport_frames(path, up_hint)
    if frame_roll:
        c, s = math.cos(frame_roll), math.sin(frame_roll)
        N, B = N * c + B * s, -N * s + B * c
    th = np.linspace(0, 2 * math.pi, n_around + 1)           # seam column duplicated
    rxx = np.concatenate([rx, rx[:, :1]], 1)
    ryy = np.concatenate([ry, ry[:, :1]], 1)
    V = (path[:, None, :] + N[:, None, :] * (np.cos(th)[None, :, None] * rxx[..., None])
         + B[:, None, :] * (np.sin(th)[None, :, None] * ryy[..., None]))
    if offset is not None:
        V = V + offset
    V = V.reshape(-1, 3)
    u = np.tile(th / (2 * math.pi), K)
    v = np.repeat(np.linspace(0, 1, K), n_around + 1)
    UV = np.stack([u, v], 1)
    W = n_around + 1
    F = []
    for i in range(K - 1):
        for j in range(n_around):
            a, b, c, d = i * W + j, i * W + j + 1, (i + 1) * W + j + 1, (i + 1) * W + j
            F.append((a, b, c))
            F.append((a, c, d))
    # tip and root caps
    tip = len(V)
    root = tip + 1
    tip_p = path[-1] + T[-1] * float(np.mean(np.abs(rx[-1]))) * 0.5
    V = np.vstack([V, tip_p, path[0]])
    UV = np.vstack([UV, (0.5, 1.0), (0.5, 0.0)])
    for j in range(n_around):
        F.append(((K - 1) * W + j, (K - 1) * W + j + 1, tip))
        F.append((root, j + 1, j))
    return V, np.array(F, dtype=np.int64), UV


# --- nine tails ---------------------------------------------------------------

TAIL_RINGS, TAIL_AROUND = 56, 42
TAIL_TIERS = 5          # overlapping lock rows along each tail
TAIL_CLUMPS = 7         # lock columns around each tail


def tail_specs():
    """The nine tails fan out in the body's own plane, from 'up and over the back' to 'straight
    back', alternating left/right so the fan has depth."""
    rng = np.random.default_rng(SEED)
    alphas = np.linspace(62, 176, 9) + rng.uniform(-4, 4, 9)
    yaws = [34, -26, 18, -36, 8, -20, 30, -12, 24]
    specs = []
    for i, a in enumerate(alphas):
        mid = 1.0 - abs(a - 120) / 50.0
        specs.append(dict(alpha=math.radians(a), yaw=math.radians(yaws[i]),
                          L=0.90 + 0.16 * mid + rng.uniform(-0.04, 0.04),
                          R=0.150 + 0.025 * mid + rng.uniform(-0.008, 0.008),
                          turn=math.radians(rng.uniform(88, 108)),
                          twist=rng.uniform(0, 2 * math.pi)))
    return specs


def tail_path(spec, K):
    """Leaves the rump a little further back than its fan angle, then sweeps forward/up so the tip
    curls like a plume."""
    s = np.linspace(0, 1, K)
    th = spec["alpha"] + math.radians(30) - spec["turn"] * s ** 1.45
    step = spec["L"] / (K - 1)
    d = np.stack([np.cos(th), np.zeros(K), np.sin(th)], 1)
    p = np.cumsum(d * step, 0) - d[0] * step
    p = p @ rot_z(spec["yaw"]).T
    p[:, 1] += math.copysign(0.10, spec["yaw"]) * s ** 2           # spread outwards towards the tips
    return TAIL_ROOT + np.array([0.02, 0.0, -0.02]) + p


def tail_lock_field(u, v):
    """Shared by the geometry and the texture: phase inside each lock row (1 = the lock's lip) and
    the clump weight around the tail. u, v in [0, 1]."""
    th = u * 2 * math.pi
    col = np.floor(u * TAIL_CLUMPS + 0.5)                 # lock column; boundaries sit in the grooves
    stagger = np.modf(np.sin(col * 12.9898 + 4.1) * 43758.5453)[0]
    stagger = np.abs(stagger)
    cl = 0.5 + 0.5 * np.cos(th * TAIL_CLUMPS)
    x = v * TAIL_TIERS + stagger * 0.8 + 0.55 * (1.0 - cl)     # lock ends are V-shaped points
    phase = x - np.floor(x)
    clump = cl ** 1.4
    return phase, clump


def tail_radius_profile(v):
    grow = 0.50 + 0.50 * sstep(0.0, 0.45, v)
    taper = np.clip(1.0 - v, 0, 1) ** 0.70
    return grow * np.minimum(1.0, taper * 1.75)


def build_tail(spec):
    K, A = TAIL_RINGS, TAIL_AROUND
    path = tail_path(spec, K)
    u = np.arange(A) / A
    v = np.linspace(0, 1, K)
    U, Vv = np.meshgrid(u, v)
    phase, clump = tail_lock_field(U, Vv)
    prof = tail_radius_profile(Vv) * spec["R"]
    lip = phase ** 2.6                                     # each lock swells towards its lip, then drops
    flare = (0.06 + 0.22 * clump) * sstep(0.06, 0.25, Vv) * (1 - 0.5 * sstep(0.85, 1.0, Vv))
    groove = 1.0 - 0.13 * (1.0 - clump) ** 2               # grooves between lock columns
    r = prof * groove * (1.0 + flare * (lip - 0.35))
    r = np.maximum(r, 0.003)
    return tube_mesh(path, r, r * 0.88, A, frame_roll=spec["twist"])


def merge(parts):
    Vs, Fs, UVs, off = [], [], [], 0
    for V, F, UV in parts:
        Vs.append(V)
        Fs.append(F + off)
        UVs.append(UV)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs), np.vstack(UVs)


# --- flames (sculpted: curling tongues smooth-merged into one mass, like the statue) ---------


def curl_path(p0, d0, L, K, rng, hook=1.0, rise=0.5, wave=0.5):
    """Flame tongue centre line: an S-wave that leans upwards and hooks over at the tip."""
    d = normalize(np.asarray(d0, dtype=np.float64))
    side = np.cross(d, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(side) < 0.2:
        side = np.array([0.0, 1.0, 0.0])
    side = normalize(side - d * float(side @ d))
    side = rodrigues(side, d, rng.uniform(-1.2, 1.2))
    sgn = rng.choice([-1.0, 1.0])
    ph = rng.uniform(0, 2 * math.pi)
    pts = [np.asarray(p0, dtype=np.float64)]
    step = L / (K - 1)
    for i in range(1, K):
        s = i / (K - 1)
        w = wave * math.sin(ph + 2.2 * math.pi * s) + sgn * hook * 3.2 * s ** 2.2
        d = rodrigues(d, side, w * step / L * 2.2)
        d = normalize(d + np.array([0.0, 0.0, rise * step * (1.0 - s)]))
        pts.append(pts[-1] + d * step)
    return np.array(pts)


class FlameMass:
    """Smooth union of tapered tongues and a few blobby roots, with per-tongue bounding boxes so
    the field stays cheap to evaluate."""

    def __init__(self, k=0.03, groove=0.004, seed=0, floor=None):
        self.floor = floor     # flat bottom at this height (flames resting on the ground)
        self.tongues = []      # (path, radii, lo, hi)
        self.blobs = []        # (centre, radii)
        self.k = k
        self.groove = groove
        self.seed = seed

    def add_tongue(self, path, radii):
        rmax = float(np.max(radii))
        lo = path.min(0) - rmax
        hi = path.max(0) + rmax
        self.tongues.append((path, np.asarray(radii), lo, hi))

    def add_blob(self, c, r):
        self.blobs.append((np.asarray(c, dtype=np.float64), np.asarray(r, dtype=np.float64)))

    def bounds(self, pad=0.08):
        los = [t[2] for t in self.tongues] + [c - r for c, r in self.blobs]
        his = [t[3] for t in self.tongues] + [c + r for c, r in self.blobs]
        return np.min(los, 0) - pad, np.max(his, 0) + pad

    def _tongue_dist(self, P, path, radii):
        d = np.full(len(P), 1e3)
        for i in range(len(path) - 1):
            d = np.minimum(d, sd_round_cone(P, path[i], path[i + 1], radii[i], radii[i + 1]))
        return d

    def raw(self, P):
        d = np.full(len(P), 1e3)
        for c, r in self.blobs:
            d = smin(d, sd_ellipsoid(P, c, r), self.k)
        margin = 0.08 + self.k
        for path, radii, lo, hi in self.tongues:
            q = np.maximum(np.maximum(lo - P, P - hi), 0.0)
            di = np.linalg.norm(q, axis=1)                    # lower bound outside the box
            m = np.all((P > lo - margin) & (P < hi + margin), axis=1)
            if m.any():
                di[m] = self._tongue_dist(P[m], path, radii)
            d = smin(d, di, self.k)
        return d

    def carve(self, P):
        """Swirling grooves along the flames."""
        warp = fbm(P * 5.0, 2, self.seed + 3)
        n = value_noise(P * np.array([38.0, 38.0, 7.0]) + warp[:, None] * 1.0, self.seed + 4)
        return 1.0 - np.abs(n)

    def sdf(self, P):
        d = self.raw(P) - self.groove * (self.carve(P) - 0.5)
        if self.floor is not None:
            d = smax(d, self.floor - P[:, 2], 0.01)
        return d

    def along(self, P):
        """For colouring: 0 at a tongue's root .. 1 at its tip (blobs count as roots)."""
        best = np.full(len(P), 1e3)
        s_best = np.zeros(len(P))
        for path, radii, lo, hi in self.tongues:
            m = np.all((P > lo - 0.03) & (P < hi + 0.03), axis=1)
            if not m.any():
                continue
            Q = P[m]
            K = len(path)
            for i in range(K - 1):
                a, b = path[i], path[i + 1]
                ab = b - a
                t = np.clip(((Q - a) @ ab) / float(ab @ ab), 0, 1)
                dist = np.linalg.norm(Q - (a + t[:, None] * ab), axis=1) - (radii[i] * (1 - t) + radii[i + 1] * t)
                sub = best[m]
                better = dist < sub
                sv = s_best[m]
                sv[better] = (i + t[better]) / (K - 1)
                sub[better] = dist[better]
                best[m] = sub
                s_best[m] = sv
        return s_best


def _tongue(mass, p0, d0, L, W, rng, K=16, hook=1.0, rise=0.5, wave=0.5, fat=0.85):
    path = curl_path(p0, d0, L, K, rng, hook, rise, wave)
    s = np.linspace(0, 1, K)
    radii = W * (0.75 + 0.25 * sstep(0.0, 0.2, s)) * np.clip(1 - s, 0, 1) ** fat
    mass.add_tongue(path, np.maximum(radii, 0.0035))
    return path


def build_mane_mass():
    """Flame mane: a swelling over the neck and shoulders with wavy tongues combed back and up."""
    rng = np.random.default_rng(SEED + 1)
    m = FlameMass(k=0.035, groove=0.005, seed=11)
    ne = neck_end()
    for t, r in ((-0.30, (0.13, 0.12, 0.08)), (0.05, (0.14, 0.14, 0.10)), (0.40, (0.12, 0.14, 0.10))):
        c = NECK_A + (ne - NECK_A) * t + np.array([-0.02, 0.0, 0.075 + 0.02 * (1 - t)])
        m.add_blob(c, r)
    axis_dir = normalize(ne - NECK_A)
    side = np.array([0.0, 1.0, 0.0])
    up = normalize(np.cross(side, axis_dir))
    if up[2] < 0:
        up = -up
    for i in range(40):
        t = rng.uniform(-0.45, 0.58)
        c = NECK_A + (ne - NECK_A) * t
        ang = rng.uniform(-1.7, 1.7)                       # around the neck, 0 = on top
        radial = up * math.cos(ang) + side * math.sin(ang)
        p0 = c + radial * (0.13 + 0.03 * math.cos(ang)) + np.array([0.0, 0.0, 0.05])
        d0 = normalize(radial * 0.7 + np.array([-1.2, 0.0, 0.30]) + rng.normal(0, 0.18, 3))
        L = rng.uniform(0.18, 0.32) * (0.75 + 0.35 * math.cos(ang * 0.8))
        _tongue(m, p0, d0, L, rng.uniform(0.045, 0.065), rng, K=16, hook=rng.uniform(0.2, 0.45),
                rise=0.35, wave=0.8, fat=0.55)
    return m


def build_fire_mass():
    """Orange fire: flames licking up under the chest and between the front legs, and a tall swirl
    rising beside the near hind leg up into the tails."""
    rng = np.random.default_rng(SEED + 2)
    m = FlameMass(k=0.035, groove=0.006, seed=21, floor=0.0)
    m.add_blob((0.36, 0.06, 0.0), (0.20, 0.12, 0.035))
    m.add_blob((-0.58, 0.30, 0.0), (0.20, 0.11, 0.045))
    for i in range(13):
        x = rng.uniform(0.14, 0.58)
        y = rng.uniform(-0.08, 0.22)
        H = 0.20 + 0.22 * sstep(0.60, 0.25, x) + rng.uniform(-0.03, 0.06)
        d0 = normalize(np.array([rng.uniform(-0.4, 0.3), rng.uniform(-0.2, 0.4), 1.0]))
        _tongue(m, (x, y, 0.0), d0, H, rng.uniform(0.038, 0.055), rng, K=18, hook=rng.uniform(0.3, 0.8),
                rise=0.7, wave=1.0, fat=0.6)
    for i in range(15):
        x = rng.uniform(-0.80, -0.36)
        y = rng.uniform(0.20, 0.40)
        H = 0.34 + 0.40 * sstep(-0.36, -0.68, x) + rng.uniform(-0.06, 0.08)
        d0 = normalize(np.array([-0.30 + rng.uniform(-0.3, 0.3), rng.uniform(-0.1, 0.4), 1.0]))
        _tongue(m, (x, y, 0.0), d0, H, rng.uniform(0.042, 0.062), rng, K=20,
                hook=rng.uniform(0.35, 0.9), rise=0.6, wave=1.1, fat=0.6)
    return m


# =============================================================================
# COLOURS
# =============================================================================

COAT_DARK = np.array([0.50, 0.08, 0.035])      # deep red saddle
COAT_MID = np.array([0.86, 0.26, 0.05])       # red-orange
COAT_LIGHT = np.array([0.99, 0.47, 0.10])     # bright orange flanks, shoulders, face
COAT_TIP = np.array([1.00, 0.76, 0.34])       # golden lock tips
BIB = np.array([1.00, 0.91, 0.78])            # cream-white chest, throat, cheeks
STOCK = np.array([0.075, 0.055, 0.05])        # black legs and ear backs
STOCK_EDGE = np.array([0.32, 0.09, 0.04])


def _mix(c, target, w):
    return c * (1 - w[:, None]) + np.asarray(target) * w[:, None]


def body_color(P, N, ao=None):
    """Fire fox coat. Large areas are smooth anatomical gradients (red saddle, orange sides, cream
    bib, black stockings); all small-scale variation comes from the carved fur locks and from
    ambient occlusion, so the colour follows the sculpt."""
    Ph = world_to_head(P)
    Pj = head_to_jaw(Ph)
    Nh = N @ HEAD_R
    z = P[:, 2]
    c = np.tile(COAT_LIGHT, (len(P), 1))
    c = _mix(c, COAT_MID, 0.55 * sstep(0.30, 0.58, z) * sstep(0.45, -0.2, P[:, 0]))
    saddle = sstep(0.0, 0.75, N[:, 2]) * sstep(0.46, 0.62, z) * sstep(0.32, 0.0, P[:, 0])
    c = _mix(c, COAT_DARK, 0.85 * saddle)
    crown = sstep(0.2, 0.8, Nh[:, 2]) * sstep(0.0, 0.05, Ph[:, 2]) * sstep(0.17, 0.10, Ph[:, 0])
    c = _mix(c, COAT_MID, 0.6 * crown)                                         # forehead and muzzle top
    # cream bib: throat, chest, belly, cheeks, lips and chin
    under = sstep(0.05, 0.55, -N[:, 2]) * sstep(0.50, 0.36, z) * sstep(-0.30, 0.0, P[:, 0])
    chest = sstep(0.19, 0.07, np.linalg.norm((P - np.array([0.34, 0.0, 0.40])) * np.array([1.0, 0.8, 0.7]), axis=1))
    throat = sstep(0.0, 0.5, -Nh[:, 2]) * sstep(-0.02, -0.06, Ph[:, 2]) * sstep(0.30, 0.12, np.linalg.norm(Ph, axis=1))
    cheek = sstep(-0.005, -0.03, Ph[:, 2] - 0.15 * Ph[:, 0]) * sstep(0.17, 0.08, Ph[:, 0]) * sstep(-0.08, 0.0, Ph[:, 0])
    chin = sstep(0.0, 0.5, -Nh[:, 2]) * sstep(-0.045, -0.07, Pj[:, 2]) * sstep(0.0, 0.05, Pj[:, 0])
    lips = sstep(0.014, 0.005, mouth_cavity(Ph)) * sstep(0.03, 0.08, Ph[:, 0])
    bib = np.clip(0.9 * under + chest + 0.9 * throat + cheek + 0.85 * chin + 0.7 * lips, 0, 1)
    c = _mix(c, BIB, bib)
    # black stockings with a dark red-brown blend into the coat
    stock_hi = np.where(P[:, 0] > 0.0, 0.30, 0.25)
    not_head = sstep(0.02, 0.07, head_sdf(Ph) * HEAD_S)                      # the low head and open jaw stay coloured
    c = _mix(c, STOCK_EDGE, sstep(stock_hi + 0.07, stock_hi, z) * not_head)
    c = _mix(c, STOCK, sstep(stock_hi + 0.01, stock_hi - 0.05, z) * not_head)
    inner_dir = normalize(np.array([1.0, 0.0, 0.6]))
    for s in (1, -1):
        a, b = EAR_ROOT * np.array([1, s, 1]), EAR_TIP * np.array([1, s, 1])
        along = np.clip(((Ph - a) @ (b - a)) / float((b - a) @ (b - a)), 0, 1)
        ear = sstep(0.014, 0.0, ear_sdf(Ph, s)) * sstep(0.15, 0.35, along)
        inner = sstep(0.0, 0.45, Nh @ inner_dir)
        c = _mix(c, STOCK, 0.92 * ear * np.clip(1 - inner + sstep(0.70, 0.85, along), 0, 1))     # dark backs, black tips
        c = _mix(c, BIB * 0.92, 0.85 * ear * inner * sstep(0.85, 0.65, along))                 # cream inside
        liner = sstep(0.034, 0.024, np.linalg.norm(Ph - EYE_SOCKET * np.array([1, s, 1]), axis=1))
        c = _mix(c, STOCK, 0.9 * liner)
    # fur locks: per-lock tone, golden tips, darker roots and creases, strands
    f = fur_field(P)                   # same orientation field as the carving, so colour and relief agree
    amp = np.clip(fur_amplitude(P) / 0.012, 0, 1)
    c = c * (1.0 + 0.12 * (f["cid"] - 0.5) * amp)[:, None]
    lum = c @ np.array([0.30, 0.59, 0.11])
    tipw = sstep(0.62, 0.97, f["phase"]) * f["crease"] * amp * sstep(0.10, 0.45, lum)   # no gold on black legs
    c = _mix(c, np.minimum(c * 1.25 + 0.10, 1.0) * 0.5 + COAT_TIP * 0.5, 0.65 * tipw)
    c = c * (1 - 0.30 * amp * sstep(0.30, 0.0, f["phase"]))[:, None]
    c = c * (0.80 + 0.20 * (1 - amp + amp * f["crease"]))[:, None]
    c = c * (0.92 + 0.08 * f["strands"])[:, None]
    if ao is not None:
        c = c * (0.42 + 0.58 * ao)[:, None]
    # mouth, tongue, nose
    mouth = sstep(0.005, -0.003, mouth_cavity(Ph))
    c = _mix(c, np.array([0.30, 0.05, 0.06]), mouth)
    tongue = sstep(0.006, -0.002, sd_ellipsoid(Pj, (0.095, 0.0, -0.058), (0.09, 0.026, 0.010)))
    c = _mix(c, np.array([0.70, 0.24, 0.26]), tongue)
    nose = sstep(0.012, 0.0, sd_sphere(Ph, NOSE_C, NOSE_R))
    c = _mix(c, np.array([0.045, 0.04, 0.04]), nose)
    return np.clip(c, 0, 1)


def sdf_ao(sdf, V, N):
    """Ambient occlusion from the distance field: how much the surface is hemmed in nearby."""
    occ = np.zeros(len(V))
    for delta, w in ((0.012, 0.45), (0.028, 0.30), (0.060, 0.25)):
        d = np.concatenate([sdf(V[i:i + 300_000] + N[i:i + 300_000] * delta) for i in range(0, len(V), 300_000)])
        occ += w * np.clip((delta - d) / delta, 0, 1)
    return np.clip(1.0 - 1.25 * occ, 0, 1)


# =============================================================================
# PROCEDURAL TEXTURES (tails and flames: their UVs are u around, v along)
# =============================================================================


def _periodic_noise_u(u, v, freq_u, freq_v, seed):
    """Noise that tiles in u (sampled on a circle) so the seam is invisible."""
    th = u * 2 * math.pi
    Q = np.stack([np.cos(th) * freq_u / (2 * math.pi), np.sin(th) * freq_u / (2 * math.pi), v * freq_v], -1)
    return fbm(Q.reshape(-1, 3), 3, seed).reshape(u.shape)


def tail_textures(size):
    u, v = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size)
    phase, clump = tail_lock_field(u, v)
    strands = _periodic_noise_u(u, v, 70.0, 3.0, 61)            # fine strands along the tail
    big = _periodic_noise_u(u, v, 9.0, 4.0, 62)
    root = np.array([0.80, 0.20, 0.06])          # matches the red saddle where the tails leave the body
    mid = np.array([0.98, 0.43, 0.10])
    tip = np.array([1.00, 0.86, 0.42])
    g = sstep(0.05, 0.40, v + 0.06 * big)
    col = root * (1 - g[..., None]) + mid * g[..., None]
    t = sstep(0.50, 0.95, v + 0.25 * phase - 0.12)
    col = col * (1 - t[..., None]) + tip * t[..., None]
    lip = sstep(0.80, 0.97, phase) * sstep(0.12, 0.35, v)
    col = col + (np.array([1.0, 0.95, 0.62]) - col) * (0.60 * lip)[..., None]        # burning lip of each lock
    under = sstep(0.12, 0.0, phase) * sstep(0.12, 0.3, v)
    col = col * (1 - 0.35 * under[..., None])                                        # shadow under each lock
    col = col * (0.86 + 0.20 * (0.5 + 0.5 * strands))[..., None]
    col = np.clip(col, 0, 1)
    h = 0.6 * strands + 0.4 * phase ** 2
    return col, height_to_normal(h, 0.9 / size, 3.5)


MANE_FIRE = [(0.0, (0.95, 0.32, 0.08)), (0.35, (1.0, 0.55, 0.12)), (0.75, (1.0, 0.78, 0.28)), (1.0, (1.0, 0.95, 0.66))]
ORANGE_FIRE = [(0.0, (0.84, 0.14, 0.05)), (0.30, (0.98, 0.34, 0.09)), (0.70, (1.0, 0.58, 0.16)), (1.0, (1.0, 0.86, 0.40))]


def flame_color(mass, P, palette, highlight):
    s = mass.along(P)
    carve = mass.carve(P)
    stops, cols = zip(*palette)
    t = np.clip(s * 0.95 + 0.18 * (carve - 0.5), 0, 1)
    col = np.stack([np.interp(t, stops, [c[k] for c in cols]) for k in range(3)], -1)
    ridge = sstep(0.84, 0.98, carve)                       # raised streaks catch the light
    col = col + (np.asarray(highlight) - col) * (0.38 * ridge)[:, None]
    groove = sstep(0.30, 0.03, carve)
    col = col * (1 - 0.16 * groove)[:, None]
    return np.clip(col, 0, 1)


def height_to_normal(h, texel, strength):
    """Tangent-space (OpenGL, +Y = +v) normal map from a height field indexed [v, u] (u wraps)."""
    du = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / 2.0
    dv = np.zeros_like(h)
    dv[1:-1] = (h[2:] - h[:-2]) / 2.0
    n = np.stack([-du * strength, -dv * strength, np.ones_like(h)], -1)
    n = normalize(n)
    return n * 0.5 + 0.5


# =============================================================================
# BLENDER HELPERS
# =============================================================================


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"


def mesh_object(name, V, F, UV=None, smooth=True, collection=None):
    me = bpy.data.meshes.new(name)
    F = np.asarray(F)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", np.asarray(V, dtype=np.float32).ravel())
    nl = F.shape[1]
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, nl, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate(verbose=False)
    if UV is not None:
        uvl = me.uv_layers.new(name="UVMap")
        li = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", li)
        uvl.data.foreach_set("uv", np.asarray(UV, dtype=np.float32)[li].ravel())
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def mesh_arrays(obj):
    me = obj.data
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    me.calc_loop_triangles()
    T = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    return V.reshape(-1, 3), T.reshape(-1, 3)


def tri_count(obj):
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def set_vcol(obj, rgb_srgb, name="col"):
    me = obj.data
    attr = me.color_attributes.new(name, "FLOAT_COLOR", "POINT")
    lin_rgb = srgb_to_linear(rgb_srgb)
    rgba = np.concatenate([lin_rgb, np.ones((len(lin_rgb), 1))], 1).astype(np.float32)
    attr.data.foreach_set("color", rgba.ravel())


def select_only(objs, active=None):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


def decimate_to(obj, target):
    n = tri_count(obj)
    if n <= target:
        return
    m = obj.modifiers.new("dec", "DECIMATE")
    m.ratio = target / n
    m.use_collapse_triangulate = True
    select_only([obj])
    bpy.ops.object.modifier_apply(modifier=m.name)


def triangulate(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.polygons.foreach_set("use_smooth", np.ones(len(obj.data.polygons), dtype=bool))
    obj.data.update()


def save_image(name, rgb, path, colorspace="sRGB"):
    h, w = rgb.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.colorspace_settings.name = colorspace
    rgba = np.concatenate([rgb, np.ones((h, w, 1))], -1).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    return img


def textured_material(name, color_img, normal_img, roughness=0.8):
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    tc = nt.nodes.new("ShaderNodeTexImage")
    tc.image = color_img
    nt.links.new(tc.outputs["Color"], bsdf.inputs["Base Color"])
    if normal_img is not None:
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = normal_img
        normal_img.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def vcol_material(name):
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.8
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "GEOMETRY"
    at.attribute_name = "col"
    nt.links.new(at.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


# =============================================================================
# BUILD
# =============================================================================

BODY_BOX = ((-0.85, -0.36, -0.03), (0.98, 0.42, 0.80))


def tooth(root, direction, L, r, segs=6):
    """Curved cone (a fang) from root along direction."""
    direction = normalize(np.asarray(direction, dtype=np.float64))
    path = np.array([root + direction * L * t + np.array([0.0, 0.0, 0.0]) for t in np.linspace(0, 1, 5)])
    rr = r * np.linspace(1, 0.15, 5)
    rx = np.repeat(rr[:, None], segs, 1)
    V, F, _ = tube_mesh(path, rx, rx, segs, up_hint=(1, 0, 0))
    return V, F


def build_teeth_claws():
    """Teeth (head-local -> world) and claws, as small meshes with ivory / dark colours."""
    parts = []
    for s in (1, -1):
        # upper: canine, premolars, incisors
        parts.append(tooth(np.array([0.165, 0.031 * s, -0.036]), (0.12, -0.12 * s, -1.0), 0.050, 0.0105))
        for x in (0.130, 0.102, 0.076):
            parts.append(tooth(np.array([x, 0.040 * s, -0.040]), (0.0, 0.0, -1.0), 0.020, 0.0075))
        for y in (0.008, 0.019):
            parts.append(tooth(np.array([0.194, y * s, -0.034]), (0.15, 0.0, -1.0), 0.016, 0.0055))
    upper = [(V, F) for V, F in parts]
    lower = []
    for s in (1, -1):
        # angled back towards the hinge so they point up at the upper jaw while the mouth is open
        lower.append(tooth(np.array([0.150, 0.022 * s, -0.055]), (-0.45, 0.06 * s, 1.0), 0.038, 0.0090))
        for x in (0.118, 0.090):
            lower.append(tooth(np.array([x, 0.027 * s, -0.054]), (-0.35, 0.0, 1.0), 0.018, 0.0060))
    Vs, Fs, off = [], [], 0
    for V, F in upper:
        Vs.append(head_to_world(V))
        Fs.append(F + off)
        off += len(V)
    for V, F in lower:
        Vs.append(head_to_world(jaw_to_head(V)))
        Fs.append(F + off)
        off += len(V)
    teeth_V, teeth_F = np.vstack(Vs), np.vstack(Fs)
    # claws: 4 per paw, hooked down in front of the toes
    Vs, Fs, off = [], [], 0
    for leg in LEGS.values():
        px, py = leg["paw"]
        gz = leg["ground"]
        for k, dy in enumerate((-0.032, -0.012, 0.012, 0.032)):
            root = np.array([px + 0.066 - 0.011 * abs(k - 1.5), py + dy, gz + 0.020])
            V, F = tooth(root, (0.8, 0.0, -0.6), 0.028, 0.0075)
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
    return (teeth_V, teeth_F), (np.vstack(Vs), np.vstack(Fs))


def eye_mesh(side, nu=24, nv=14):
    """UV-sphere eyeball whose front pole (v = 0) looks along EYE_DIR, so the iris texture is a
    simple function of v."""
    c = EYE_CENTER * np.array([1, side, 1])
    f = normalize(EYE_DIR * np.array([1, side, 1]))
    a = normalize(np.array([0.0, 0.0, 1.0]) - f * f[2])
    b = np.cross(f, a)
    U, Vv = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1))
    ang, pol = 2 * math.pi * U, math.pi * Vv
    pts = c + EYE_RADIUS * (np.cos(pol)[..., None] * f + np.sin(pol)[..., None]
                            * (np.cos(ang)[..., None] * a + np.sin(ang)[..., None] * b))
    V = head_to_world(pts.reshape(-1, 3))
    UV = np.stack([U.ravel(), Vv.ravel()], 1)
    W = nu + 1
    F = []
    for i in range(nv):
        for j in range(nu):
            p00, p01, p10, p11 = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            if i < nv - 1:
                F.append((p00, p10, p11))
            if i > 0:
                F.append((p00, p11, p01))
    F = np.array(F)
    cen = head_to_world(c[None])[0]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if np.sum(np.einsum("ij,ij->i", n, V[F].mean(1) - cen)) < 0:     # keep the normals pointing out
        F = F[:, ::-1]
    return V, F, UV


def eye_texture(size=256):
    """Fire-fox eye: white-hot centre fading to amber and a dark rim, black slit pupil, catchlight."""
    u, v = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size)
    X = np.sin(math.pi * v) * np.cos(2 * math.pi * u)        # up
    Y = np.sin(math.pi * v) * np.sin(2 * math.pi * u)        # sideways
    stops = [0.0, 0.05, 0.17, 0.25, 0.30, 1.0]
    cols = [(1.0, 0.97, 0.66), (1.0, 0.92, 0.45), (1.0, 0.68, 0.10), (0.88, 0.30, 0.03), (0.25, 0.06, 0.02), (0.12, 0.04, 0.03)]
    col = np.stack([np.interp(v, stops, [c[k] for c in cols]) for k in range(3)], -1)
    pupil = sstep(1.15, 0.85, (Y / 0.11) ** 2 + (X / 0.50) ** 2) * (v < 0.4)
    col = col * (1 - pupil[..., None]) + np.array([0.02, 0.01, 0.01]) * pupil[..., None]
    spark = sstep(0.11, 0.06, np.hypot(X - 0.30, Y + 0.22)) * (v < 0.4)
    col = col * (1 - spark[..., None]) + np.array([1.0, 1.0, 0.95]) * spark[..., None]
    return np.clip(col, 0, 1)


def build_eyes(out_dir):
    img = save_image("Eyes_Color", eye_texture(), os.path.join(out_dir, "Eyes_Color.png"))
    V, F, UV = merge([eye_mesh(1), eye_mesh(-1)])
    o = mesh_object("Fox_Eyes", V, F, UV)
    o.data.materials.append(textured_material("Eyes_Material", img, None, 0.2))
    return o


def head_labels(V, F):
    """Splits the game mesh into head (head, ruff, upper neck, teeth) and body, so each gets its own
    1024 px texture. The cut follows where the ruff meets the neck and shoulders; a few smoothing
    passes keep it clean."""
    C = V[F].mean(1)
    d_head = np.minimum(head_sdf(world_to_head(C)) * HEAD_S,
                        sd_ellipsoid(C, ruff_center(), (0.125, 0.165, 0.165)))
    for root, tip, r in RUFF_TUFTS:
        d_head = np.minimum(d_head, sd_round_cone(C, root, tip, r, 0.004))
    d_body = np.minimum(torso_sdf(C), sd_round_cone(C, NECK_A, neck_end(), 0.140, 0.105))
    for leg in LEGS.values():
        d_body = np.minimum(d_body, leg_sdf(C, leg))
    lab = (d_head < d_body).astype(np.float64)
    for _ in range(6):
        vl = np.zeros(len(V))
        cnt = np.zeros(len(V))
        for k in range(3):
            np.add.at(vl, F[:, k], lab)
            np.add.at(cnt, F[:, k], 1)
        vl /= np.maximum(cnt, 1)
        lab = (vl[F].mean(1) > 0.5).astype(np.float64)
    return lab > 0.5


CUSTOM_NORMALS = {}


def build_body(quick):
    log("body: sampling high-res sculpt")
    h_hi = 0.005 if quick else 0.0032
    V, F = surface_nets(body_sdf, *BODY_BOX, h_hi)
    V, F = largest_component(V, F)
    V = taubin_smooth(V, F, 1)
    F = tri_faces(F)
    N = vertex_normals(V, F)
    t0 = time.time()
    ao = sdf_ao(body_sdf, V, N)
    log(f"body: ambient occlusion ({time.time() - t0:.0f}s)")
    C = body_color(V, N, ao)
    (tV, tF), (cV, cF) = build_teeth_claws()
    allV = np.vstack([V, tV, cV])
    allF = np.vstack([F, tF + len(V), cF + len(V) + len(tV)])
    allC = np.vstack([C, np.tile([0.92, 0.88, 0.78], (len(tV), 1)), np.tile([0.10, 0.08, 0.08], (len(cV), 1))])
    hi = mesh_object("Bake_FoxHigh", allV, allF)
    set_vcol(hi, allC)
    log(f"body high-res: {len(allF):,} tris")
    if quick:
        return hi, []
    log("body: low-poly game meshes")
    lV, lF = surface_nets(body_sdf, *BODY_BOX, 0.0070)
    lV, lF = largest_component(lV, lF)
    lV = taubin_smooth(lV, lF, 2)
    lF = tri_faces(lF)
    total = HEAD_TRIS + BODY_TRIS
    for attempt in range(3):
        tmp = mesh_object("tmp_body", lV, lF)
        decimate_to(tmp, total - len(tF) - len(cF))
        dv, dt = mesh_arrays(tmp)
        bpy.data.objects.remove(tmp, do_unlink=True)
        aV = np.vstack([dv, tV, cV])
        aF = np.vstack([dt, tF + len(dv), cF + len(dv) + len(tV)])
        lab = head_labels(aV, aF)
        n_head, n_body = int(lab.sum()), int((~lab).sum())
        scale = min(HEAD_TRIS / max(n_head, 1), BODY_TRIS / max(n_body, 1))
        if scale >= 0.995:
            break
        total = int(total * scale * 0.99)
    Nv = vertex_normals(aV, aF)
    parts = []
    for name, sel in (("Fox_Head", lab), ("Fox_Body", ~lab)):
        Fs = aF[sel]
        used = np.unique(Fs)
        remap = np.full(len(aV), -1)
        remap[used] = np.arange(len(used))
        o = mesh_object(name, aV[used], remap[Fs])
        CUSTOM_NORMALS[name] = Nv[used]          # whole-mesh normals: no shading seam at the cut
        parts.append(o)
        log(f"{name}: {tri_count(o):,} tris")
    return hi, parts


def apply_custom_normals(obj):
    Ns = CUSTOM_NORMALS.get(obj.name)
    if Ns is not None:
        obj.data.normals_split_custom_set_from_vertices([tuple(n) for n in Ns])


def uv_unwrap(obj):
    """Smart UV on a smoothed copy (fur spikes would shred it into tiny islands), then pack."""
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    bm = bmesh.new()
    bm.from_mesh(me)
    for _ in range(20):
        bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(me)
    bm.free()
    select_only([obj])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.004)
    bpy.ops.uv.pack_islands(rotate=True, margin=0.004, shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")
    me.vertices.foreach_set("co", co)
    me.update()


def fill_bake_misses(imgs):
    """Repairs bake errors, which otherwise show as dark specks: texels where the ray found nothing
    (pure black colour) and texels where it hit the inside of a neighbouring flame tongue or tuft
    (normal tilted more than ~70 degrees from the surface). Both maps are refilled there from the
    surrounding good texels; any normal still bad afterwards is set flat."""
    col = imgs["Color"]
    w, h = col.size
    px = {}
    for key, img in imgs.items():
        a = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(a)
        px[key] = a.reshape(h, w, 4)
    miss = (px["Color"][..., :3].sum(-1) < 0.004) | (px["Normal"][..., 2] < 0.675)
    n_miss = int(miss.sum())
    for _ in range(12):
        if not miss.any():
            break
        ok = (~miss).astype(np.float32)
        acc = {k: np.zeros((h, w, 3), dtype=np.float32) for k in px}
        cnt = np.zeros((h, w), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                o = np.roll(np.roll(ok, dy, 0), dx, 1)
                cnt += o
                for k in px:
                    acc[k] += np.roll(np.roll(px[k][..., :3], dy, 0), dx, 1) * o[..., None]
        fill = miss & (cnt > 0)
        for k in px:
            px[k][fill, :3] = acc[k][fill] / cnt[fill, None]
        miss = miss & ~fill
    flat = px["Normal"][..., 2] < 0.675
    px["Normal"][flat, :3] = (0.5, 0.5, 1.0)
    for key, img in imgs.items():
        img.pixels.foreach_set(px[key].ravel())
        img.save()
    return n_miss


def bake(low, high, prefix, out_dir, cage=0.045, dist=0.12):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 4
    sc.cycles.use_denoising = False
    high.data.materials.clear()
    high.data.materials.append(vcol_material(prefix + "_src"))
    mat = bpy.data.materials.new(prefix + "_mat")
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    imgs, nodes = {}, {}
    for key, cs in (("Color", "sRGB"), ("Normal", "Non-Color")):
        img = bpy.data.images.new(f"{prefix}_{key}", TEX, TEX, alpha=False)
        img.colorspace_settings.name = cs
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = img
        imgs[key], nodes[key] = img, n
    low.data.materials.clear()
    low.data.materials.append(mat)
    select_only([high, low], active=low)
    for key, btype, kw in (("Color", "DIFFUSE", dict(pass_filter={"COLOR"})), ("Normal", "NORMAL", {})):
        nt.nodes.active = nodes[key]
        t0 = time.time()
        bpy.ops.object.bake(type=btype, target="IMAGE_TEXTURES", save_mode="INTERNAL", use_clear=True,
                            use_selected_to_active=True, use_cage=False, cage_extrusion=cage,
                            max_ray_distance=dist, margin=16, margin_type="EXTEND",
                            normal_space="TANGENT", normal_r="POS_X", normal_g="POS_Y", normal_b="POS_Z", **kw)
        path = os.path.join(out_dir, f"{prefix}_{key}.png")
        imgs[key].filepath_raw = path
        imgs[key].file_format = "PNG"
        imgs[key].save()
        log(f"baked {prefix} {key} ({time.time() - t0:.0f}s)")
    n = fill_bake_misses(imgs)
    log(f"{prefix}: repaired {n:,} texels (bake misses, flipped normals, gaps between UV islands)")
    low.data.materials.clear()
    low.data.materials.append(textured_material(prefix + "_Material", imgs["Color"], imgs["Normal"], 0.85))
    return imgs


def build_tails_objects(out_dir):
    col, nrm = tail_textures(TEX)
    ci = save_image("Tails_Color", col, os.path.join(out_dir, "Tails_Color.png"))
    ni = save_image("Tails_Normal", nrm, os.path.join(out_dir, "Tails_Normal.png"), "Non-Color")
    mat = textured_material("Tails_Material", ci, ni, 0.75)
    specs = tail_specs()
    objs = []
    order = np.argsort([s["alpha"] for s in specs])
    groups = [order[0:3], order[3:6], order[6:9]]
    for gi, g in enumerate(groups):
        V, F, UV = merge([build_tail(specs[i]) for i in g])
        o = mesh_object(f"Tails_{'ABC'[gi]}", V, F, UV)
        o.data.materials.append(mat)
        objs.append(o)
        log(f"tails {'ABC'[gi]}: {tri_count(o):,} tris")
    return objs


def build_flame(name, mass, palette, highlight, tris, quick):
    lo, hi = mass.bounds()
    log(f"{name}: sampling flame mass ({len(mass.tongues)} tongues)")
    V, F = surface_nets(mass.sdf, lo, hi, 0.008 if quick else 0.0034)
    F = tri_faces(F)
    high = mesh_object("Bake_" + name, V, F)
    set_vcol(high, flame_color(mass, V, palette, highlight))
    if quick:
        return high, high
    lV, lF = surface_nets(mass.sdf, lo, hi, 0.0062)
    low = mesh_object(name, lV, tri_faces(lF))
    decimate_to(low, tris)
    log(f"{name} low-poly: {tri_count(low):,} tris")
    return high, low


# =============================================================================
# EXPORT + PREVIEW
# =============================================================================


def to_roblox_space(objs):
    """Base bottom to y = 0 in Roblox, wolf facing Roblox -Z (LookVector), 1 unit = 1 stud.
    Returns the model-space height of the base bottom."""
    zmin = min(np.min(mesh_arrays(o)[0][:, 2]) for o in objs)
    for o in objs:
        V, _ = mesh_arrays(o)
        V = V - np.array([0.0, 0.0, zmin])
        V = V @ rot_z(math.radians(-90)).T            # +X (wolf forward) -> -Y (FBX -Z forward)
        V = V * STUDS_PER_UNIT
        o.data.vertices.foreach_set("co", V.astype(np.float32).ravel())
        o.data.update()
    return float(zmin)


def export_fbx(objs, path):
    sc = bpy.context.scene
    sc.unit_settings.scale_length = 0.01
    sc.unit_settings.length_unit = "CENTIMETERS"
    select_only(objs)
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"MESH"},
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_UNITS", global_scale=1.0,
        axis_forward="-Z", axis_up="Y", use_mesh_modifiers=True, mesh_smooth_type="FACE",
        bake_anim=False, path_mode="COPY", embed_textures=True)
    sc.unit_settings.scale_length = 1.0
    sc.unit_settings.length_unit = "METERS"


def setup_render(scale):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 40
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = 1000, 900
    sc.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    sc.world = world
    if world.node_tree is None:
        world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.035, 0.035, 0.04, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs[0])

    def light(name, kind, loc, energy, color, size=1.0):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy * (scale ** 2 if kind != "SUN" else 1.0)
        ld.color = color
        if kind == "AREA":
            ld.size = size * scale
        o = bpy.data.objects.new(name, ld)
        o.location = np.array(loc) * scale
        sc.collection.objects.link(o)
        if kind in ("AREA", "SPOT"):
            d = -np.array(loc)
            o.rotation_mode = "QUATERNION"
            from mathutils import Vector
            o.rotation_quaternion = Vector(d).to_track_quat("-Z", "Y")
        return o

    # preview floor only (not exported)
    bpy.ops.mesh.primitive_plane_add(size=8.0 * scale, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "PreviewFloor"
    fm = bpy.data.materials.new("PreviewFloor")
    fb = fm.node_tree.nodes.get("Principled BSDF")
    fb.inputs["Base Color"].default_value = (0.05, 0.05, 0.055, 1.0)
    fb.inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(fm)

    light("Key", "AREA", (2.5, 2.2, 3.0), 380, (1.0, 0.96, 0.92), 2.0)
    light("Fill", "AREA", (-2.0, 2.8, 1.4), 120, (0.80, 0.86, 1.0), 2.5)
    light("Rim", "AREA", (-2.2, -2.5, 2.4), 260, (1.0, 0.80, 0.70), 2.0)
    # like the PointLights the README suggests adding in Roblox
    light("FireGlowA", "POINT", (0.35, 0.10, 0.18), 18, (1.0, 0.45, 0.15))
    light("FireGlowB", "POINT", (-0.60, 0.30, 0.35), 24, (1.0, 0.45, 0.15))
    light("ManeGlow", "POINT", (0.35, 0.0, 0.80), 10, (1.0, 0.55, 0.60))


def render_views(out_dir, views, scale, prefix):
    from mathutils import Vector
    sc = bpy.context.scene
    paths = []
    for name in views:
        loc, tgt, lens = VIEWS[name]
        cd = bpy.data.cameras.new("cam_" + name)
        cd.lens = lens
        cam = bpy.data.objects.new("cam_" + name, cd)
        sc.collection.objects.link(cam)
        cam.location = Vector(loc) * scale
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = (Vector(tgt) * scale - cam.location).to_track_quat("-Z", "Y")
        sc.camera = cam
        p = os.path.join(out_dir, f"{prefix}_{name}.png")
        sc.render.filepath = p
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        log(f"rendered {name} ({time.time() - t0:.0f}s) -> {p}")
        paths.append(p)
    return paths


# =============================================================================
# MAIN
# =============================================================================


def main():
    opts = _parse_args()
    out = os.path.abspath(opts["out"])
    rbx = os.path.join(out, "roblox")
    os.makedirs(rbx, exist_ok=True)
    report = {}
    if opts["preview_only"]:
        render_fbx_preview(os.path.join(rbx, "FlameWolf.fbx"), out, opts["views"], report)
        return
    reset_scene()

    body_hi, body_parts = build_body(opts["quick"])
    mane_hi, mane_lo = build_flame("Flames_Mane", build_mane_mass(), MANE_FIRE, (1.0, 1.0, 0.80), MANE_TRIS, opts["quick"])
    fire_hi, fire_lo = build_flame("Flames_Fire", build_fire_mass(), ORANGE_FIRE, (1.0, 0.95, 0.55), FIRE_TRIS, opts["quick"])
    tails = build_tails_objects(rbx)
    eyes = build_eyes(rbx)

    if opts["quick"]:
        mat = vcol_material("quick")
        for o in (body_hi, mane_hi, fire_hi):
            o.data.materials.append(mat)
        setup_render(1.0)
        render_views(out, opts["views"] or ["ref", "side"], 1.0, "quick")
        return

    jobs = [(p, body_hi, p.name) for p in body_parts] + [(mane_lo, mane_hi, "Mane"), (fire_lo, fire_hi, "Fire")]
    for low, high, prefix in jobs:
        triangulate(low)
        uv_unwrap(low)
        apply_custom_normals(low)
        tight = prefix in ("Mane", "Fire")          # narrow crevices between flame tongues
        bake(low, high, prefix, rbx, cage=0.02 if tight else 0.03, dist=0.06 if tight else 0.08)
    for high in {id(h): h for _, h, _ in jobs}.values():
        bpy.data.objects.remove(high, do_unlink=True)

    game = body_parts + [eyes, mane_lo, fire_lo] + tails
    for o in game:
        report[o.name] = tri_count(o)
        if report[o.name] > 20000:
            raise BuildError(f"{o.name} has {report[o.name]} triangles (Roblox limit 20,000)")
    log("triangles: " + ", ".join(f"{k} {v:,}" for k, v in report.items()) + f"; total {sum(report.values()):,}")

    zmin = to_roblox_space(game)
    fbx = os.path.join(rbx, "FlameWolf.fbx")
    export_fbx(game, fbx)
    log(f"exported {fbx} ({os.path.getsize(fbx) / 1e6:.1f} MB)")

    if opts["render"]:
        render_fbx_preview(fbx, out, opts["views"], report)


def render_fbx_preview(fbx, out, views, report):
    """Renders the exported FBX itself, so what you see is what Roblox gets. It is imported the way
    Roblox reads it (1 cm in the file = 1 stud), so the measured size is the size in studs."""
    reset_scene()
    sc = bpy.context.scene
    sc.unit_settings.scale_length = 0.01
    sc.unit_settings.length_unit = "CENTIMETERS"
    bpy.ops.import_scene.fbx(filepath=fbx)
    sc.unit_settings.scale_length = 1.0
    sc.unit_settings.length_unit = "METERS"
    imported = [o for o in sc.objects if o.type == "MESH"]
    log("re-imported: " + ", ".join(f"{o.name} {tri_count(o):,}" for o in imported))
    for o in imported:
        if not o.data.uv_layers:
            raise BuildError(f"{o.name} lost its UVs in the FBX")
        if not o.data.materials or not any(n.type == "TEX_IMAGE" and n.image for n in o.data.materials[0].node_tree.nodes):
            raise BuildError(f"{o.name} lost its texture in the FBX")
    bpy.context.view_layer.update()
    pts = np.vstack([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in imported])
    size = pts.max(0) - pts.min(0)
    report["fbx_size_studs"] = [round(float(x), 1) for x in size]
    log(f"size in Roblox (studs, x y z as imported): {size.round(1)}")
    if not (10.0 < size.max() < 40.0):
        raise BuildError(f"unexpected size in studs: {size}")
    # back to the modelling frame so the preview cameras fit: Z up, fox facing +X, paws on z = 0
    root = bpy.data.objects.new("PreviewRoot", None)
    sc.collection.objects.link(root)
    for o in imported:
        if o.parent is None:
            o.parent = root
    root.rotation_euler = (0.0, 0.0, math.radians(90))
    root.location = (0.0, 0.0, -float(pts[:, 2].min()))
    setup_render(STUDS_PER_UNIT)
    render_views(out, views or list(VIEWS), STUDS_PER_UNIT, "roblox")


if __name__ == "__main__":
    main()
