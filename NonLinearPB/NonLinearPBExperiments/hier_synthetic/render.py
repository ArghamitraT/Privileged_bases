"""Draft renderer for the 3 x 3 x 3 hierarchical synthetic image dataset.

Conda environment: mrl_env_cuda12
Usage:
    from hier_synthetic.render import render, sample_nuisance
    img = render(c, m, f, sample_nuisance(np.random.default_rng(0)))

Spec: NonLinearPB/plans/hierarchical_synthetic_data_spec.md. Geometry is in
64-pixel units, drawn at SUPERSAMPLE x resolution and box-downsampled to 64 x 64
so sub-pixel jitter is visible. Mid parts are built from the coarse geometry and
fine details are anchored to the (c, m) geometry. Sizes are uncalibrated: the
cost ladder has not been measured yet.
"""

import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
SUPERSAMPLE = 4

COARSE_NAMES = ("dumbbell", "ring + stem", "boomerang")
MID_NAMES = (("tapered", "straight", "bowed"),
             ("stem through", "stem below", "stem slanted"),
             ("rounded", "sharp", "stepped"))
# FINE_NAMES[c][m][f]: fine details depend on the parent. Dumbbell "straight" and ring
# "stem below" have their own sets: their anchors coincide with a sibling's (leakage probe, v1).
FINE_NAMES = ((("spike", "knob", "keel"), ("twin spikes", "flat cap", "bead"), ("spike", "knob", "keel")),
              (("T-foot", "fork", "hook"), ("round tip", "angled tip", "split tip"), ("T-foot", "fork", "hook")),
              (("bulge", "inner bump", "square"),) * 3)

# CONFIG: pilot nuisance amplitudes (spec section 4); no whole-object translation.
JITTER_PX = 0.2          # max |delta| for radius, width and length, in pixels (v1: 0.3)
BRIGHTNESS = 0.05        # gain drawn from [1 - b, 1 + b]
NOISE_SIGMA = 0.01

CLEAN = {"radius": 0.0, "width": 0.0, "length": 0.0, "gain": 1.0, "noise_seed": None}


def sample_nuisance(rng):
    """Within-leaf variation that never changes c, m or f."""
    d = rng.uniform(-JITTER_PX, JITTER_PX, size=3)
    return {"radius": d[0], "width": d[1], "length": d[2],
            "gain": rng.uniform(1 - BRIGHTNESS, 1 + BRIGHTNESS),
            "noise_seed": int(rng.integers(2**31))}


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _perp(v):
    return np.array([-v[1], v[0]])


def _bezier(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _band(points, widths):
    """Polygon of a polyline with per-vertex width and mitered joins."""
    pts = np.asarray(points, dtype=float)
    widths = np.broadcast_to(np.asarray(widths, dtype=float), (len(pts),))
    left, right = [], []
    for i, p in enumerate(pts):
        d_in = _unit(p - pts[i - 1]) if i > 0 else None
        d_out = _unit(pts[i + 1] - p) if i < len(pts) - 1 else None
        if d_in is None or d_out is None:
            n, scale = _perp(d_out if d_in is None else d_in), 1.0
        else:
            n = _unit(_perp(d_in) + _perp(d_out))
            scale = 1.0 / max(np.dot(n, _perp(d_in)), 0.25)
        left.append(p + n * widths[i] / 2 * scale)
        right.append(p - n * widths[i] / 2 * scale)
    return left + right[::-1]


class _Canvas:
    def __init__(self):
        self.img = Image.new("L", (SIZE * SUPERSAMPLE, SIZE * SUPERSAMPLE), 0)
        self.draw = ImageDraw.Draw(self.img)

    def _xy(self, points):
        return [(float(x) * SUPERSAMPLE, float(y) * SUPERSAMPLE) for x, y in points]

    def polygon(self, points, on=True):
        self.draw.polygon(self._xy(points), fill=255 if on else 0)

    def disk(self, center, r, on=True):
        (x, y), s = center, SUPERSAMPLE
        self.draw.ellipse([(x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s], fill=255 if on else 0)

    def band(self, points, widths, on=True):
        self.polygon(_band(points, widths), on)

    def result(self):
        return np.asarray(self.img.resize((SIZE, SIZE), Image.BOX), dtype=np.float32) / 255.0


# CONFIG: shape sizes in 64-pixel units, tuned against the cost ladder
# (spec section 3). Fine sizes set the cost unit; mid sizes set mid/fine.
GEOM = {
    "c0": {"lobe_r": 6.0, "half": 17.0, "w": 4.0, "taper": (3.6, 0.3), "bow": 5.0,
           "spike": (4.0, 8.0), "knob_r": 4.0, "knob_shift": 7.0, "keel": (4.0, 7.5),
           "twin": (2.6, 6.5, 4.5), "cap": (5.5, 3.0), "bead_r": 4.5},
    "c1": {"center": (32.0, 25.0), "ring_r": 10.0, "ring_t": 3.0, "w": 4.0, "bottom": 54.0,
           "protrude": 11.0, "slant": 3.5, "tfoot": (8.0, 3.5), "prong": (7.0, 5.5, 3.0), "hook": (8.0, 6.0, 3.0),
           "round_tip_r": 4.5, "angled_tip": (6.0, 5.0, 3.5), "split_tip": (3.5, 6.0, 2.5)},
    "c2": {"a": (10.0, 48.0), "b": (24.0, 14.0), "c": (55.0, 33.0), "w": 4.5, "round": 0.5, "step": 0.52,
           "bulge_r": 3.8, "inner_r": 4.0, "square": 6.2, "square_shift": 9.0},
}


# ---- fine details: each family has its own three shapes, placed at an anchor
# point p with outward unit direction u supplied by the (c, m) geometry.

def _fine_c0(cv, m, f, p, u, w):
    g, t = GEOM["c0"], _perp(u)
    if m == 1:      # "straight" anchors where "tapered" does, so it gets its own details
        if f == 0:      # twin spikes, one each side of the midpoint
            half, h, gap = g["twin"]
            for s_ in (-1, 1):
                q = p + t * s_ * gap
                cv.polygon([q - t * half, q + t * half, q + u * h])
        elif f == 1:    # flat cap under the connector
            half, thick = g["cap"]
            q = p - u * (w + thick / 2 - 0.5)
            cv.band([q - t * half, q + t * half], thick)
        else:           # bead threaded on the connector
            cv.disk(p - u * (w / 2), g["bead_r"])
        return
    if f == 0:      # spike pointing away from the connector
        half, h = g["spike"]
        cv.polygon([p - t * half, p + t * half, p + u * h])
    elif f == 1:    # knob, set beside the spike position along the connector
        cv.disk(p + t * g["knob_shift"] + u * (g["knob_r"] - 2), g["knob_r"])
    else:           # keel: spike on the opposite edge
        half, h = g["keel"]
        q = p - u * w
        cv.polygon([q - t * half, q + t * half, q - u * h])


def _fine_c1(cv, m, f, p, u, w):
    g, t = GEOM["c1"], _perp(u)
    if m == 1:      # "stem below" ends where "stem through" does, so it gets its own tips
        if f == 0:      # round tip
            r = g["round_tip_r"]
            cv.disk(p + u * (r - 1.5), r)
        elif f == 1:    # angled tip: the stem end bends to one side
            side, down, thick = g["angled_tip"]
            cv.band([p - u * 1.5, p + u * down - t * side], thick)
        else:           # split tip: two parallel prongs on a short crossbar
            half, length, thick = g["split_tip"]
            cv.band([p - t * half, p + t * half], thick)
            for s_ in (-1, 1):
                cv.band([p + t * s_ * half, p + t * s_ * half + u * length], thick)
        return
    if f == 0:      # T-foot
        half, thick = g["tfoot"]
        cv.band([p - t * half, p + t * half], thick)
    elif f == 1:    # fork: two prongs
        length, spread, thick = g["prong"]
        for s in (-1, 1):
            cv.band([p - u, p + u * length + t * s * spread], thick)
    else:           # hook to one side
        side, up, thick = g["hook"]
        cv.band([p, p + u * 2 + t * side, p - u * up + t * side], thick)


def _fine_c2(cv, m, f, p, u, w):
    g, t = GEOM["c2"], _perp(u)
    if f == 0:      # round bulge on the outside of the bend
        cv.disk(p + u * 1.5, g["bulge_r"])
    elif f == 1:    # round bump on the inside of the bend
        cv.disk(p - u * (w * 1.4 + 1.5), g["inner_r"])
    else:           # square bump on the outer edge of the right arm, beside the bend
        a, d = g["square"], _unit(np.array(g["c"]) - np.array(g["b"]))
        n = _perp(d) if np.dot(_perp(d), u) > 0 else -_perp(d)
        q = p + d * g["square_shift"]
        cv.polygon([q - d * a / 2, q + d * a / 2, q + d * a / 2 + n * a, q - d * a / 2 + n * a])


# ---- coarse families with their mid refinements; each returns the fine anchor.

def _family0(cv, m, nz):
    """Horizontal double-lobed body joined by a connector bar."""
    g = GEOM["c0"]
    r, half = g["lobe_r"] + nz["radius"], g["half"] + nz["length"]
    a, b = np.array([32 - half, 32.0]), np.array([32 + half, 32.0])
    cv.disk(a, r)
    cv.disk(b, r)
    w = g["w"] + nz["width"]
    if m == 0:      # tapered: thick where it leaves the left lobe, thin at the right
        thick, thin = g["taper"]
        ea, eb = a + np.array([r, 0]), b - np.array([r, 0])
        cv.band([a, ea, eb, b], [thick * w, thick * w, thin * w, thin * w])
        mid_w = (thick + thin) / 2 * w
        return (a + b) / 2 + np.array([0, -mid_w / 2]), np.array([0.0, -1.0]), mid_w
    if m == 1:      # straight
        cv.band([a, b], w)
        return (a + b) / 2 + np.array([0, -w / 2]), np.array([0.0, -1.0]), w
    apex = np.array([32.0, 32.0 - g["bow"]])     # bowed upward
    cv.band(_bezier(a, 2 * apex - (a + b) / 2, b), w)
    return apex + np.array([0, -w / 2]), np.array([0.0, -1.0]), w


def _family1(cv, m, nz):
    """Vertical ring with a stem hanging below it."""
    g = GEOM["c1"]
    c, R, w = np.array(g["center"]), g["ring_r"] + nz["radius"], g["w"] + nz["width"]
    cv.disk(c, R)
    cv.disk(c, R - g["ring_t"], on=False)
    bottom = g["bottom"] + nz["length"]
    if m == 0:      # stem runs through the hole and sticks out above the ring
        cv.band([(c[0], c[1] - R - g["protrude"]), (c[0], bottom)], w)
        return np.array([c[0], bottom]), np.array([0.0, 1.0]), w
    top = np.array([c[0], c[1] + R - 1])
    foot = np.array([c[0] + (g["slant"] if m == 2 else 0.0), bottom])   # m=1 straight down, m=2 slanted
    cv.band([top, foot], w)
    return foot, _unit(foot - top), w


def _family2(cv, m, nz):
    """Diagonal arm bent once, like a boomerang."""
    g = GEOM["c2"]
    a, b, c = (np.array(g[k]) for k in ("a", "b", "c"))
    a = a + _unit(a - b) * nz["length"]
    c = c + _unit(c - b) * nz["length"]
    w = g["w"] + nz["width"]
    out = -_unit(_unit(a - b) + _unit(c - b))        # outward through the bend
    if m == 0:      # rounded bend
        p1, p2 = b + g["round"] * (a - b), b + g["round"] * (c - b)
        curve = _bezier(p1, b, p2)
        pts, corner = [a, *curve, c], curve[len(curve) // 2]
    elif m == 1:    # sharp corner
        pts, corner = [a, b, c], b
    else:           # stepped: corner replaced by a square step
        p1, p2 = b + g["step"] * (a - b), b + g["step"] * (c - b)
        q = np.array([p1[0], p2[1]])
        pts, corner = [a, p1, q, p2, c], q
    cv.band(pts, w)
    # Outer edge sits further out at a mitered corner than along a smooth bend.
    miter = {0: 1.0, 1: 1.0 / max(abs(np.dot(out, _perp(_unit(b - a)))), 0.4), 2: 1.2}[m]
    return corner + out * (w / 2) * miter, out, w


FAMILIES = (_family0, _family1, _family2)
FINES = (_fine_c0, _fine_c1, _fine_c2)


def render(c, m, f, nuisance=CLEAN):
    """One 64 x 64 float32 image in [0, 1] for leaf (c, m, f)."""
    cv = _Canvas()
    p, u, w = FAMILIES[c](cv, m, nuisance)
    FINES[c](cv, m, f, np.asarray(p, dtype=float), _unit(u), w)
    img = cv.result() * nuisance["gain"]
    if nuisance["noise_seed"] is not None:
        img = img + np.random.default_rng(nuisance["noise_seed"]).normal(0, NOISE_SIGMA, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)
