"""A parametric human figure, as a mesh, from a height.

The capsule in `cast.py` is a volume: it answers "how much of the frame does
a 1.75 m person fill". Through the gaussian renderer it came out as a flat
coloured blob with no limbs and no edges, and close to the lens it was a
smear -- a frame of it did not read as a scene. This module builds a body
that a renderer can shade: head, neck, a torso with shoulders and a waist,
pelvis, two-segment arms and legs, hands, feet, each a low-poly primitive,
joined into one triangle mesh of a couple of thousand faces. Flat-shaded
with a key light it reads as a solid person, and the pose says what the
person is doing.

Proportions are standard segment lengths as fractions of stature (Drillis
and Contini), so everything scales with the typed height and a 1.55 m actor
is a smaller person, not a shrunk one. Canonical frame: upright along +Z,
feet on z=0, facing +Y, shoulders along X; `facing_deg` is degrees
anticlockwise from +X in plan, so 90 is canonical, as in `cast.figure`.

Poses:
  standing        arms hanging slightly out from the body, feet apart, weight even
  walking         legs and arms swing +/-25 degrees in antiphase with a knee bend
                  on the swinging leg and a small vertical bob; `phase` in cycles
  sitting         hips and knees at 90 degrees, forearms on the lap; the mark is
                  the seat (the pelvis sits over it), the shins reach the floor
  leaning         weight on one leg, the other set out and soft at the knee, the
                  shoulders tilted toward the planted side, one forearm raised as
                  if resting on a surface at about a metre

Arms (`arms`), for the poses that stand: `down` as the pose has them, `folded`
across the chest, `pockets` with the hands at the hip front, `one_raised` with
the right forearm up on a surface. A person with their hands in their pockets
reads as someone in the room; two arms hanging at attention read as a photo for
a passport, which is what the finished frames looked like before this.

Nothing here is measured from the twin and nothing here enters a
measurement: the figure is drawn into a frame so the frame shows what the
plan claims.
"""

from __future__ import annotations

import math

import numpy as np

from ..film import render

# Segment levels and lengths as fractions of stature (Drillis & Contini 1966).
_HEAD_TOP = 1.000
_CHIN = 0.870
_SHOULDER = 0.818
_CHEST = 0.740
_WAIST = 0.620
_HIP = 0.530
_KNEE = 0.285
_ANKLE = 0.039
_UPPER_ARM = 0.186
_FOREARM = 0.146
_HAND = 0.108
_THIGH = _HIP - _KNEE
_SHIN = _KNEE - _ANKLE
_FOOT_LEN = 0.152
_FOOT_W = 0.055
_FOOT_H = 0.039

# Widths are written against the reference stature and scaled by it, so the
# shoulder line of a 1.75 m figure is 0.46 m wide -- the number the stand-in
# and the sweep have always used.
REFERENCE_HEIGHT_M = 1.75
SHOULDER_WIDTH_M = 0.46
STRIDE_M = 0.7

# Part ids, one per face, so a renderer can tint the face and the shoes.
PART_BODY = 0
PART_FACE = 1
PART_HANDS = 2
PART_SHOES = 3

POSES = ("standing", "walking", "sitting", "leaning")
ARMS = ("down", "folded", "pockets", "one_raised")

# Where a raised forearm rests: the height of a desk, a counter, a door
# handle's wall. The pose aims the right wrist here, scaled with stature.
REST_HEIGHT_M = 1.0


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _sphere(centre, radii, subdivisions: int = 1):
    """An ellipsoid: icosphere scaled per axis (positive scales keep winding)."""
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    v = np.asarray(m.vertices, dtype=np.float64) * np.asarray(radii, dtype=np.float64)
    return v + np.asarray(centre, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def _segment(p0, p1, r0: float, r1: float, sections: int = 10):
    """A tapered tube from p0 (radius r0) to p1 (radius r1), capped. Built
    with outward winding: side quads anticlockwise seen from outside, the
    caps facing along the axis."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = _unit(p1 - p0)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(axis, ref))
    e2 = np.cross(axis, e1)
    ang = np.linspace(0.0, 2 * math.pi, sections, endpoint=False)
    ring = np.cos(ang)[:, None] * e1[None] + np.sin(ang)[:, None] * e2[None]
    verts = np.concatenate([p0 + r0 * ring, p1 + r1 * ring, p0[None], p1[None]])
    n = sections
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + i])
        faces.append([j, n + j, n + i])
        faces.append([j, i, 2 * n])              # bottom cap, facing -axis
        faces.append([n + i, n + j, 2 * n + 1])  # top cap, facing +axis
    return verts, np.asarray(faces, dtype=np.int64)


def _loft(rings: list[np.ndarray]):
    """Stacked closed rings of equal point count joined into a tube, capped
    at both ends. Rings are ordered bottom to top, each anticlockwise seen
    from above, so the side faces wind outward."""
    n = len(rings[0])
    verts = np.concatenate(rings)
    faces = []
    for k in range(len(rings) - 1):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append([a + i, a + j, b + i])
            faces.append([a + j, b + j, b + i])
    bottom = len(verts)
    top = bottom + 1
    verts = np.concatenate([verts, rings[0].mean(axis=0, keepdims=True), rings[-1].mean(axis=0, keepdims=True)])
    a = (len(rings) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        faces.append([j, i, bottom])
        faces.append([a + i, a + j, top])
    return verts, np.asarray(faces, dtype=np.int64)


def _rounded_ring(z: float, half_w: float, half_d: float, n: int = 12) -> np.ndarray:
    """A superellipse cross-section (between a box and an ellipse) at height z."""
    ang = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    # Exponent 0.7 squares the corners off a little: a torso, not a barrel.
    x = half_w * np.sign(c) * np.abs(c) ** 0.7
    y = half_d * np.sign(s) * np.abs(s) ** 0.7
    return np.stack([x, y, np.full(n, z)], axis=1)


def _box(lo, hi):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = np.array([[0, 2, 1], [0, 3, 2],          # bottom (-z)
                  [4, 5, 6], [4, 6, 7],          # top (+z)
                  [0, 1, 5], [0, 5, 4],          # -y
                  [2, 3, 7], [2, 7, 6],          # +y
                  [1, 2, 6], [1, 6, 5],          # +x
                  [3, 0, 4], [3, 4, 7]], dtype=np.int64)  # -x
    return v, f


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------


def walk_phase(t: float, speed_mps: float, *, stride_m: float = STRIDE_M) -> float:
    """The gait cycle at time `t` for someone walking at `speed_mps`: one
    cycle (two steps) per two strides. Returns the phase in [0, 1); 0 when
    the actor is stopped, so a standing actor's legs are together."""
    if speed_mps <= 1e-6:
        return 0.0
    return float((t * speed_mps / (2.0 * stride_m)) % 1.0)


def _pose_params(pose: str, phase: float) -> dict:
    """Joint angles for a pose. Swing angles are degrees of pitch about the
    body's X axis: positive swings the limb forward (+Y)."""
    if pose == "standing":
        return dict(leg=(0.0, 0.0), knee=(0.0, 0.0), arm=(3.0, 3.0), elbow=(8.0, 8.0),
                    abduct=6.0, bob=0.0, sit=False)
    if pose == "leaning":
        # Weight on the right leg (planted straight), the left set out to the
        # side and soft; the trunk rolls toward the planted side so the
        # shoulders tilt, and the right forearm comes up onto the surface.
        return dict(leg=(4.0, 0.0), knee=(14.0, 0.0), arm=(3.0, 0.0), elbow=(8.0, 0.0),
                    abduct=6.0, bob=0.0, sit=False, leg_out=(16.0, 0.0), roll=7.0,
                    arms="one_raised")
    if pose == "walking":
        w = 2 * math.pi * float(phase)
        s = math.sin(w)
        swing = 25.0 * s
        # The knee bends while that leg is swinging forward (mid-swing), and
        # the body is highest when the legs pass each other.
        knee_l = 40.0 * max(0.0, math.sin(w + math.pi / 2))
        knee_r = 40.0 * max(0.0, math.sin(w + 3 * math.pi / 2))
        bob = 0.5 * (1.0 - math.cos(2 * w))
        return dict(leg=(swing, -swing), knee=(knee_l, knee_r), arm=(-swing, swing),
                    elbow=(15.0 + 20.0 * max(0.0, -s), 15.0 + 20.0 * max(0.0, s)),
                    abduct=5.0, bob=bob, sit=False)
    if pose == "sitting":
        return dict(leg=(90.0, 90.0), knee=(90.0, 90.0), arm=(10.0, 10.0), elbow=(80.0, 80.0),
                    abduct=6.0, bob=0.0, sit=True)
    raise ValueError(f"unknown pose {pose!r}; one of {POSES}")


def figure_parts(height_m: float, *, pose: str = "standing", phase: float = 0.0, arms: str = "down"):
    """The figure in its canonical frame: (vertices (V, 3), faces (F, 3),
    part (F,) int) with `part` one of the PART_* ids per face. `arms` is one
    of ARMS; a pose with arms of its own (leaning raises one) wins over
    `down`, and any other `arms` wins over the pose."""
    V, F, part, _joints = _build(height_m, pose=pose, phase=phase, arms=arms)
    return V, F, part


def joints(height_m: float, *, pose: str = "standing", phase: float = 0.0, arms: str = "down") -> dict[str, np.ndarray]:
    """Named joint positions in the canonical frame (shoulder_l, shoulder_r,
    elbow_*, wrist_*, hip_*, knee_*, ankle_*, head), feet pinned to z=0 as
    the mesh is: the numbers a test reads a pose off."""
    return _build(height_m, pose=pose, phase=phase, arms=arms)[3]


def _arm_params(P: dict, arms: str, h: float, s: float) -> list[tuple[float, float, float, np.ndarray | None]]:
    """Per side (left, right): (swing, elbow, abduct, forearm direction or
    None for the hinge). `arms` overrides the pose's hanging arms."""
    if arms not in ARMS:
        raise ValueError(f"unknown arms {arms!r}; one of {ARMS}")
    want = arms if arms != "down" else P.get("arms", "down")
    base = [(P["arm"][0], P["elbow"][0], P["abduct"], None), (P["arm"][1], P["elbow"][1], P["abduct"], None)]
    if want == "down" or P["sit"] and want != "folded":
        return base
    out = []
    for k, side in enumerate((-1.0, 1.0)):
        if want == "folded":
            # Upper arm a little forward, forearm across the chest to the far elbow.
            out.append((18.0, 0.0, 8.0, _unit(np.array([-side * 1.0, 0.55, 0.12]))))
        elif want == "pockets":
            # Upper arm slightly back, forearm turning in to the hip front.
            out.append((-6.0, 0.0, 8.0, _unit(np.array([-side * 0.35, 0.62, -0.72]))))
        else:  # one_raised: the right forearm up, level, the left as the pose has it
            if side > 0:
                # The elbow hangs; the forearm pitches up so the wrist lands at
                # REST_HEIGHT_M for the reference stature (scaled by `s`).
                elbow_z = h * _SHOULDER - 0.01 * s - h * _UPPER_ARM * math.cos(math.radians(6.0))
                rise = (REST_HEIGHT_M * s - elbow_z) / max(h * _FOREARM, 1e-6)
                rise = float(np.clip(rise, -0.6, 0.6))
                out.append((0.0, 0.0, 6.0, _unit(np.array([0.0, math.sqrt(max(0.0, 1 - rise * rise)), rise]))))
            else:
                out.append(base[k])
    return out


def _build(height_m: float, *, pose: str, phase: float, arms: str):
    h = float(height_m)
    if not (0.3 <= h <= 3.0):
        raise ValueError(f"height_m={h} is not a stature in metres")
    s = h / REFERENCE_HEIGHT_M
    P = _pose_params(pose, phase)
    parts: list[tuple[np.ndarray, np.ndarray, int]] = []
    joint: dict[str, np.ndarray] = {}
    upper: list[int] = []     # indices into `parts` of everything above the hip, for the trunk roll

    def add(v, f, part=PART_BODY):
        parts.append((np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64), part))

    # Where the hip sits. Seated, the thighs are horizontal and the shins
    # vertical, so the hip is one shin plus one ankle above the floor.
    hip_z = h * _HIP
    if P["sit"]:
        hip_z = h * (_SHIN + _ANKLE) + 0.02 * s
    drop = h * _HIP - hip_z            # how far everything above the hip comes down
    lift = 0.02 * s * P["bob"]         # walking bob
    dz = lift - drop

    # Torso: pelvis, waist, chest, shoulders, lofted. Widths at the reference.
    half_sh = SHOULDER_WIDTH_M * s / 2.0
    rings = [
        _rounded_ring(h * (_HIP - 0.04) + dz, 0.165 * s, 0.105 * s),
        _rounded_ring(h * _HIP + dz, 0.175 * s, 0.115 * s),
        _rounded_ring(h * _WAIST + dz, 0.150 * s, 0.100 * s),
        _rounded_ring(h * _CHEST + dz, 0.195 * s, 0.120 * s),
        _rounded_ring(h * _SHOULDER - 0.01 * s + dz, half_sh * 0.92, 0.100 * s),
        _rounded_ring(h * _SHOULDER + 0.02 * s + dz, half_sh * 0.70, 0.085 * s),
    ]
    add(*_loft(rings))
    upper.append(len(parts) - 1)

    # Neck and head. The head is an ellipsoid a little taller than wide.
    neck_bottom = h * _SHOULDER + 0.01 * s + dz
    chin = h * _CHIN + dz
    add(*_segment([0, 0, neck_bottom], [0, 0.01 * s, chin + 0.01 * s], 0.055 * s, 0.05 * s, sections=10))
    head_c = np.array([0.0, 0.01 * s, (chin + h * _HEAD_TOP + dz) / 2.0])
    head_r = np.array([0.078 * s, 0.095 * s, (h * _HEAD_TOP + dz - chin) / 2.0])
    hv, hf = _sphere(head_c, head_r, subdivisions=2)
    # The face is the front of the head: tag those faces so a renderer can
    # lighten them and the figure's facing reads.
    tri = hv[hf]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    is_face = (n[:, 1] > 0.35) & (n[:, 2] > -0.55) & (n[:, 2] < 0.55)
    upper.append(len(parts) - 1)
    add(hv, hf[~is_face], PART_BODY)
    upper.append(len(parts) - 1)
    add(hv, hf[is_face], PART_FACE)
    upper.append(len(parts) - 1)
    joint["head"] = head_c.copy()

    # Arms: shoulder joint, upper arm, elbow, forearm, hand. Each side has
    # its own swing (walking is antiphase) and a slight abduction outward;
    # `arms` may replace the hinge with a forearm direction of its own.
    sh_z = h * _SHOULDER - 0.01 * s + dz
    for (side, tag), (swing, elbow, abduct, fore_dir) in zip(((-1.0, "l"), (1.0, "r")), _arm_params(P, arms, h, s)):
        shoulder = np.array([side * (half_sh - 0.02 * s), 0.0, sh_z])
        R_up = _rot_x(swing)
        out = np.array([side * math.sin(math.radians(abduct)), 0.0, -math.cos(math.radians(abduct))])
        d_up = R_up @ out
        elbow_p = shoulder + d_up * (h * _UPPER_ARM)
        d_fore = fore_dir if fore_dir is not None else _rot_x(swing + elbow) @ out
        wrist = elbow_p + d_fore * (h * _FOREARM)
        hand_end = wrist + d_fore * (h * _HAND * 0.8)
        joint[f"shoulder_{tag}"] = shoulder.copy()
        joint[f"elbow_{tag}"] = elbow_p.copy()
        joint[f"wrist_{tag}"] = wrist.copy()
        add(*_sphere(shoulder, [0.052 * s] * 3, subdivisions=1))
        upper.append(len(parts) - 1)
        add(*_segment(shoulder, elbow_p, 0.050 * s, 0.040 * s))
        upper.append(len(parts) - 1)
        add(*_sphere(elbow_p, [0.042 * s] * 3, subdivisions=1))
        upper.append(len(parts) - 1)
        add(*_segment(elbow_p, wrist, 0.040 * s, 0.030 * s))
        upper.append(len(parts) - 1)
        hv2, hf2 = _sphere((wrist + hand_end) / 2.0, [0.030 * s, 0.045 * s, 0.045 * s], subdivisions=1)
        # Orient the hand ellipsoid along the forearm.
        hv2 = _orient(hv2, (wrist + hand_end) / 2.0, d_fore)
        add(hv2, hf2, PART_HANDS)
        upper.append(len(parts) - 1)

    # The trunk roll of a lean: everything above the hip turns about the
    # body's Y axis at the hip centre, so one shoulder drops toward the
    # planted leg and the head goes with it.
    roll = float(P.get("roll", 0.0))
    if abs(roll) > 1e-9:
        pivot = np.array([0.0, 0.0, hip_z])
        R_roll = _rot_y(-roll)
        for k in upper:
            v, f, part = parts[k]
            parts[k] = ((v - pivot) @ R_roll.T + pivot, f, part)
        for key in list(joint):
            joint[key] = (joint[key] - pivot) @ R_roll.T + pivot

    # Legs: hip joint, thigh, knee, shin, ankle, foot. Legs swing from the
    # hip; the knee flexes backward (shin behind the thigh line).
    hip_half = 0.09 * s
    leg_out = P.get("leg_out", (0.0, 0.0))
    for (side, tag), swing, knee, out_deg in zip(((-1.0, "l"), (1.0, "r")), P["leg"], P["knee"], leg_out):
        hip = np.array([side * hip_half, 0.0, hip_z])
        # Abduction sets the leg out to its own side (about Y) before the swing.
        R_out = _rot_y(-side * out_deg)
        d_th = _rot_x(swing) @ R_out @ np.array([0.0, 0.0, -1.0])
        knee_p = hip + d_th * (h * _THIGH)
        d_sh = _rot_x(swing - knee) @ R_out @ np.array([0.0, 0.0, -1.0])
        ankle = knee_p + d_sh * (h * _SHIN)
        joint[f"hip_{tag}"] = hip.copy()
        joint[f"knee_{tag}"] = knee_p.copy()
        joint[f"ankle_{tag}"] = ankle.copy()
        add(*_sphere(hip, [0.075 * s] * 3, subdivisions=1))
        add(*_segment(hip, knee_p, 0.072 * s, 0.055 * s))
        add(*_sphere(knee_p, [0.055 * s] * 3, subdivisions=1))
        add(*_segment(knee_p, ankle, 0.052 * s, 0.038 * s))
        # The foot: a box from the ankle, forward along the shin's facing,
        # flat on the floor when the shin is vertical.
        fl, fw, fh = h * _FOOT_LEN, h * _FOOT_W, h * _FOOT_H
        bv, bf = _box([-fw / 2.0, -0.3 * fl, -fh], [fw / 2.0, 0.7 * fl, 0.0])
        if P["sit"] or abs(swing - knee) < 1e-6:
            foot_pitch = 0.0
        else:
            foot_pitch = 0.35 * (swing - knee)
        bv = bv @ _rot_x(foot_pitch).T + ankle + np.array([0.0, 0.0, 0.0])
        add(bv, bf, PART_SHOES)

    # Join. Then pin the feet to the floor: the lowest vertex sits at z=0
    # (walking lifts the body, so the planted heel is what touches down;
    # seated, the soles under the shins), and standing the crown is exactly
    # the typed stature.
    offset = 0
    vs, fs, ps = [], [], []
    for v, f, part in parts:
        vs.append(v)
        fs.append(f + offset)
        ps.append(np.full(len(f), part, dtype=np.int64))
        offset += len(v)
    V = np.concatenate(vs)
    F = np.concatenate(fs)
    part = np.concatenate(ps)
    low = float(V[:, 2].min())
    V[:, 2] -= low
    for key in joint:
        joint[key] = joint[key] - np.array([0.0, 0.0, low])
    return V, F, part, joint


def _orient(v: np.ndarray, centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Rotate an ellipsoid built along +Z about its centre so +Z maps to -axis
    (hands hang along the forearm)."""
    target = -_unit(np.asarray(axis, dtype=np.float64))
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(z @ target, -1.0, 1.0))
    if c > 1 - 1e-9:
        return v
    if c < -1 + 1e-9:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        k = _unit(np.cross(z, target))
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + math.sqrt(1 - c * c) * K + (1 - c) * (K @ K)
    return (v - centre) @ R.T + centre


def figure_mesh(
    height_m: float,
    *,
    pose: str = "standing",
    phase: float = 0.0,
    facing_deg: float = 90.0,
    xy=(0.0, 0.0),
    floor_z: float = 0.0,
    arms: str = "down",
) -> tuple[np.ndarray, np.ndarray]:
    """The figure placed in the twin: (vertices (V, 3) float64, faces (F, 3)
    int). Built in the canonical frame, turned to `facing_deg` about Z and
    moved so the feet stand on (`xy`, `floor_z`). Deterministic."""
    V, F, _ = figure_parts(height_m, pose=pose, phase=phase, arms=arms)
    return place(V, facing_deg, xy, floor_z), F


def place(V: np.ndarray, facing_deg: float, xy, floor_z: float) -> np.ndarray:
    R = _rot_z(float(facing_deg) - 90.0)
    out = np.asarray(V, dtype=np.float64) @ R.T
    out[:, 0] += float(xy[0])
    out[:, 1] += float(xy[1])
    out[:, 2] += float(floor_z)
    return out


def part_colours(part: np.ndarray, rgb) -> np.ndarray:
    """Per-face colours from one body colour: the face lighter so the facing
    reads, the hands a little lighter, the shoes darker."""
    base = np.asarray(rgb, dtype=np.float32).reshape(3)
    if base.max() > 1.0:
        base = base / 255.0
    light = np.array([1.0, 0.94, 0.86], dtype=np.float32)
    out = np.repeat(base[None], len(part), axis=0)
    out[part == PART_FACE] = 0.55 * base + 0.45 * light
    out[part == PART_HANDS] = 0.75 * base + 0.25 * light
    out[part == PART_SHOES] = 0.55 * base
    return out


def surface_samples(V: np.ndarray, F: np.ndarray, count: int = 2400, seed: int = 0) -> np.ndarray:
    """Points on the mesh surface, so the figure's `xyz` is still a body for
    anything that draws points (the viewer's figure layer, the old gaussian
    compositing path) rather than a bare vertex list."""
    from trimesh import Trimesh
    from trimesh.sample import sample_surface

    pts, _ = sample_surface(Trimesh(vertices=V, faces=F, process=False), int(count), seed=seed)
    return np.asarray(pts, dtype=np.float64)


def stand_in(
    height_m: float,
    xy,
    facing_deg: float,
    floor_z: float,
    *,
    pose: str = "standing",
    phase: float = 0.0,
    rgb=None,
    label: str = "",
    arms: str = "down",
) -> render.Figure:
    """A render.Figure with a mesh, ready for `render.render_shot(figures=[...])`."""
    V, F, part = figure_parts(height_m, pose=pose, phase=phase, arms=arms)
    V = place(V, facing_deg, xy, floor_z)
    colour = np.asarray(rgb if rgb is not None else render.STAND_IN_RGB, dtype=np.float32).reshape(3)
    if colour.max() > 1.0:
        colour = colour / 255.0
    return render.Figure(
        xyz=surface_samples(V, F), rgb=colour, label=label,
        mesh=(V, F), mesh_rgb=part_colours(part, colour),
    )


def figure_for_actor(
    actor,
    xy,
    facing_deg: float,
    floor_z: float,
    *,
    pose: str = "standing",
    phase: float = 0.0,
    arms: str = "down",
) -> render.Figure:
    """`cast.Actor` on a mark as a shaded figure in the actor's colour. The
    caller decides the pose: `walking` with `walk_phase(t, speed)` while the
    actor is between marks, `sitting` when the mark is a chair, `leaning`
    against a thing; `arms` from the beat's business (`arms_for_business`)."""
    return stand_in(
        float(actor.height_m), xy, facing_deg, floor_z, pose=pose, phase=phase, arms=arms,
        rgb=np.asarray(actor.colour, dtype=np.float32) / 255.0, label=getattr(actor, "name", ""),
    )


_ARMS_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("folded", ("folded", "crossed", "arms across")),
    ("pockets", ("pocket",)),
    ("one_raised", ("leaning", "leans", "hand on", "typing", "laptop", "keyboard", "desk", "counter",
                    "shelf", "door frame", "doorframe", "window sill", "sill", "phone", "mug", "cup",
                    "glass", "holding", "writing", "reading", "wall")),
)


def arms_for_business(business: str | None) -> str:
    """The arms a piece of business implies, from its words: folded arms,
    hands in pockets, a hand on or holding a thing. Anything else hangs.
    The words are the breakdown's; the choice is a drawing decision, not a
    measurement."""
    text = (business or "").lower()
    if not text:
        return "down"
    for arms, words in _ARMS_WORDS:
        if any(w in text for w in words):
            return arms
    return "down"
