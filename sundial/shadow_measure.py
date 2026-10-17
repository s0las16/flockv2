"""Classical-CV shadow-length measurement — complements the VLM point picks.

perception.CosmosClient.locate_shadow() asks a VLM for base/tip pixel points;
this module re-measures the shadow's LENGTH from the pixels themselves so a
confident-but-wrong point pick can't silently drive the sun-elevation math in
pipeline._sun_elevation_deg():

  1. crop a ROI around the object's ground-contact point, extended toward the
     VLM's shadow_tip hint (or a radial window when no hint exists),
  2. adaptive-threshold the Lab L channel for darker-than-surroundings pixels,
     then morphology open/close to clean speckle and bridge gaps,
  3. walk a ray outward from the base until luminance recovers to the local
     background — that distance is the shadow length in px,
  4. score confidence from how continuous and edge-like the dark region is.

`base_xyxy` is an object bbox (same convention as detect_objects' bbox_xyxy);
the shadow's base is taken as the bottom-center of the box — where a roughly
vertical object meets the ground. The box itself (slightly inflated) is
excluded from the dark-pixel test: a thin object's own pixels threshold as
"dark" and would otherwise look like a shadow pointing into itself.

cv2 and numpy are imported lazily inside functions; the module itself imports
with zero third-party deps.
"""
from __future__ import annotations

import math

# Adaptive-threshold offset: a pixel counts as "shadow" when it is this many
# L-channel units darker than its local Gaussian neighborhood.
_ADAPT_C = 9.0
_STEP_PX = 2.0      # ray-walk sampling step
_PATIENCE_PX = 8.0  # consecutive lit px before the shadow is called "over"
_MIN_LEN_PX = 6.0   # shorter than this is texture noise, not a shadow


def _import_cv():
    """Lazy cv2/numpy import — keeps module import dep-free."""
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise RuntimeError("opencv-python required") from e
    return cv2, np


def _roi_rect(np, base, direction, reach, half_w, W, H):
    """Axis-aligned ROI rect (x0,y0,x1,y1) covering base -> base+d*reach.

    With direction=None it's a radial window of radius `reach` around base.
    `half_w` pads so the shadow's full width (and transect samples) fits.
    """
    if direction is not None:
        pts = np.array([base, base + direction * reach])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
    else:
        lo, hi = base - reach, base + reach
    lo = np.clip(lo - half_w, 0, [W - 1, H - 1])
    hi = np.clip(hi + half_w, 0, [W - 1, H - 1])
    return int(lo[0]), int(lo[1]), int(hi[0]) + 1, int(hi[1]) + 1


def _ray_exit_t(base, d, rect):
    """t at which ray base+t*d leaves rect (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = rect
    t = float("inf")
    for b, dd, lo, hi in (
        (base[0], d[0], x0, x1),
        (base[1], d[1], y0, y1),
    ):
        if dd > 1e-9:
            t = min(t, (hi - b) / dd)
        elif dd < -1e-9:
            t = min(t, (lo - b) / dd)
    return max(t, 0.0)


def _dark_mask(cv2, np, bgr_roi, box_w):
    """-> (L, L_bg, mask): Lab lightness, local background, dark-pixel mask.

    adaptiveThreshold(INV) flags pixels more than _ADAPT_C below their local
    mean; open kills speckle, close bridges the penumbra's gaps. L_bg is a
    heavily blurred L used as the per-point background the ray walk compares
    against.

    The adaptive block is sized from the OBJECT width, not the ROI: a window
    smaller than the shadow's width sees uniform darkness and flags nothing in
    the interior — the block must span shadow + lit ground on both sides.

    Returns (L, L_bg, mask, block) — the caller needs block: the object's dark
    halo contaminates L_bg out to ~block/2, so "lit" gaps shorter than that
    are artifacts the ray walk should bridge, not shadow ends.
    """
    lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]
    h, w = L.shape
    block = int(max(15.0, min(2.0 * box_w, min(h, w) - 2))) | 1  # odd, ≥15
    block = max(block, 3)
    mask = cv2.adaptiveThreshold(
        L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, _ADAPT_C,
    )
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    L_bg = cv2.GaussianBlur(L, (0, 0), 9)
    return L, L_bg, mask, block


def _walk_shadow(cv2, np, L, L_bg, mask, base, d, t0, t1, forbid=None,
                 step=_STEP_PX, patience=_PATIENCE_PX):
    """Walk the ray base+t*d from t0 to t1; stop after `patience` lit px.

    A sample is "dark" when the cleaned mask says so OR its luminance deficit
    vs the local background exceeds _ADAPT_C. Samples inside `forbid`
    (x0,y0,x1,y1 — the object's own bbox) are neutral: not dark, and they
    don't advance the lit run, so a ray passing through the object neither
    counts it as shadow nor dies on it.

    Returns (end_t, dark, deficit): end_t is the last dark t before the
    terminating lit run — 0.0 when the ray never leaves the lit.
    """
    ts = np.arange(t0, t1, step, dtype=np.float32)
    if ts.size == 0:
        return 0.0, np.zeros(0, bool), np.zeros(0, np.float32)
    xs = (base[0] + d[0] * ts).reshape(1, -1).astype(np.float32)
    ys = (base[1] + d[1] * ts).reshape(1, -1).astype(np.float32)
    l = cv2.remap(L, xs, ys, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=255).ravel()
    bg = cv2.remap(L_bg, xs, ys, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=255).ravel()
    m = cv2.remap(mask, xs, ys, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()
    deficit = bg.astype(np.float32) - l.astype(np.float32)
    dark = (m > 127) | (deficit > _ADAPT_C)
    inside = None
    if forbid is not None:
        fx0, fy0, fx1, fy1 = forbid
        inside = ((xs >= fx0) & (xs <= fx1)
                  & (ys >= fy0) & (ys <= fy1)).ravel()
        dark = dark & ~inside
    end_i, lit = -1, 0.0
    for i in range(len(ts)):
        if dark[i]:
            lit, end_i = 0.0, i
        elif inside is not None and inside[i]:
            pass  # object's own pixels: neutral ground
        else:
            lit += step
            if lit >= patience:
                break
    if end_i < 0:
        return 0.0, np.zeros(0, bool), np.zeros(0, np.float32)
    return float(ts[end_i]), dark[: end_i + 1], deficit[: end_i + 1]


def _find_direction(cv2, np, L, L_bg, mask, base, rect, t0, reach, forbid,
                    patience):
    """No tip hint: pick the radial direction where darkness persists longest.

    36 rays at 10-degree steps; each is scored by measured dark length weighted
    by mean luminance deficit (deep + long beats shallow + long). Returns a
    unit vector or None when every direction turns lit almost immediately —
    i.e. there is no shadow attached to this base point.
    """
    best = None
    for k in range(36):
        ang = math.radians(k * 10)
        d = np.array([math.cos(ang), math.sin(ang)])
        t1 = min(_ray_exit_t(base, d, rect), reach)
        end_t, dark, deficit = _walk_shadow(
            cv2, np, L, L_bg, mask, base, d, t0, t1, forbid=forbid,
            patience=patience)
        if end_t < _MIN_LEN_PX:
            continue
        mean_def = float(deficit[dark].mean()) if dark.any() else 0.0
        score = end_t * min(1.0, mean_def / 20.0)
        if best is None or score > best[0]:
            best = (score, d)
    return None if best is None else best[1]


def _edge_score(cv2, np, L, base, d, length, half_w):
    """Is the region edge-like? Transects at 30/55/80% of length should be
    dark in the middle and lit on BOTH sides — a real cast shadow has two
    visible edges. A broad dark floor does not. Returns 0..1."""
    if length <= 4:
        return 0.0
    n = np.array([-d[1], d[0]])  # ray normal
    offs = np.linspace(-half_w, half_w, 9, dtype=np.float32)
    dips = []
    for frac in (0.30, 0.55, 0.80):
        c = base + d * (length * frac)
        xs = (c[0] + n[0] * offs).reshape(1, -1).astype(np.float32)
        ys = (c[1] + n[1] * offs).reshape(1, -1).astype(np.float32)
        vals = cv2.remap(L, xs, ys, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=-1).ravel()
        vals = vals[vals >= 0]
        if vals.size < 5:
            continue
        mid = vals[len(vals) // 2 - 1: len(vals) // 2 + 2].mean()
        sides = np.concatenate([vals[:2], vals[-2:]]).mean()
        dips.append(max(0.0, float(sides - mid)))
    if not dips:
        return 0.0
    return float(min(1.0, np.mean(dips) / 25.0))


def _blob_factor(cv2, np, mask, base, d, length, radial):
    """Sanity-check the connected component under the measured ray.

    If the 'shadow' merges into a dark region covering most of a RADIAL ROI
    (night shot, shaded plaza) it isn't a distinct cast shadow -> heavy
    penalty. (Skipped for hint ROIs — a narrow corridor built around the
    shadow direction is *supposed* to be mostly shadow.) A compact elongated
    component (classic cast-shadow shape) gets a small bonus either way.
    """
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    votes: dict = {}
    for frac in (0.35, 0.55, 0.75):
        px = int(round(base[0] + d[0] * length * frac))
        py = int(round(base[1] + d[1] * length * frac))
        if 0 <= px < w and 0 <= py < h:
            lab = int(labels[py, px])
            votes[lab] = votes.get(lab, 0) + 1
    if not votes:
        return 1.0
    lab = max(votes, key=votes.get)
    if lab == 0:
        return 0.7  # ray rides on mask gaps — mildly suspicious
    area = float(stats[lab, cv2.CC_STAT_AREA])
    if radial and area > 0.55 * w * h:
        return 0.3
    cw = float(stats[lab, cv2.CC_STAT_WIDTH])
    ch = float(stats[lab, cv2.CC_STAT_HEIGHT])
    if max(cw, ch) / max(1.0, min(cw, ch)) > 2.5:
        return 1.15  # elongated: looks like a cast shadow
    return 1.0


def measure_shadow_length(frame_path, base_xyxy, tip_hint=None):
    """Measure a cast shadow's pixel length with classical CV.

    frame_path : image file readable by cv2.imread
    base_xyxy  : object bbox [x1,y1,x2,y2] — base = bottom-center of the box
    tip_hint   : optional [x,y] pixel guess (e.g. the VLM's shadow_tip); used
                 for search direction and ROI extent, never as the answer

    -> {"shadow_len_px": float, "confidence": float,
        "method": "cv2-segmentation"} or None when nothing measurable is
    attached to the base point. If the hinted direction finds no shadow, a
    radial search is tried before giving up.
    """
    cv2, np = _import_cv()

    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    H, W = img.shape[:2]
    try:
        x1, y1, x2, y2 = [float(v) for v in base_xyxy][:4]
    except (TypeError, ValueError, IndexError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    base = np.array([(x1 + x2) / 2.0, y2])  # bottom-center = ground contact
    if not (0 <= base[0] < W and 0 <= base[1] < H):
        return None
    box_w, box_h = x2 - x1, y2 - y1

    # The object's own pixels are off-limits for the dark test — thin objects
    # threshold as "dark" end-to-end and would fake a self-pointing shadow.
    pad = max(4.0, 0.08 * min(box_w, box_h))
    forbid_img = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    half_w = max(box_w * 1.2, 24.0)
    t0 = max(4.0, 0.1 * box_w)  # step just past the object's footprint

    def _attempt(direction, reach):
        """-> (end_t, d, dark, deficit, L, base_roi, mask, radial) or None."""
        rx0, ry0, rx1, ry1 = _roi_rect(np, base, direction, reach,
                                      half_w, W, H)
        if rx1 - rx0 < 24 or ry1 - ry0 < 24:
            return None
        roi = img[ry0:ry1, rx0:rx1]
        L, L_bg, mask, block = _dark_mask(cv2, np, roi, box_w)
        # The object's dark halo keeps L_bg depressed for ~block/2 past the
        # excluded box — a "lit" gap shorter than ~0.6*block is an artifact,
        # not the shadow's end.
        patience = max(_PATIENCE_PX, 0.6 * block)
        b = base - np.array([rx0, ry0])
        forbid = (forbid_img[0] - rx0, forbid_img[1] - ry0,
                  forbid_img[2] - rx0, forbid_img[3] - ry0)
        rect = (0, 0, rx1 - rx0 - 1, ry1 - ry0 - 1)
        d = direction
        if d is None:
            d = _find_direction(cv2, np, L, L_bg, mask, b, rect,
                                t0, reach, forbid, patience)
            if d is None:
                return None
        t1 = min(_ray_exit_t(b, d, rect), reach)
        end_t, dark, deficit = _walk_shadow(
            cv2, np, L, L_bg, mask, b, d, t0, t1, forbid=forbid,
            patience=patience)
        if end_t < _MIN_LEN_PX:
            return None
        return end_t, d, dark, deficit, L, b, mask, direction is None

    # --- attempt 1: hinted direction --------------------------------------
    out = None
    if tip_hint is not None:
        try:
            tip = np.array([float(tip_hint[0]), float(tip_hint[1])])
        except (TypeError, IndexError, ValueError):
            tip = None
        if tip is not None:
            v = tip - base
            dist = float(np.hypot(v[0], v[1]))
            if dist >= 4.0:
                d_hint = v / dist
                # look a bit past the hint: CV should confirm, not echo
                out = _attempt(d_hint, dist * 1.35 + 0.5 * box_h)

    # --- attempt 2: radial fallback ----------------------------------------
    if out is None:
        # noon shadows are ~0.3x object height, low sun reaches 4x+ — a 2.5x
        # radial window covers the useful band without drowning in ROI
        reach = float(min(2.5 * box_h, 700.0, 0.9 * min(W, H)))
        out = _attempt(None, reach)
    if out is None:
        return None
    end_t, d, dark, deficit, L, base_roi, mask, radial = out

    # --- confidence ----------------------------------------------------------
    coverage = float(dark.mean()) if dark.size else 0.0
    contrast = (float(np.clip(deficit[dark].mean() / 30.0, 0.0, 1.0))
                if dark.any() else 0.0)
    edge = _edge_score(cv2, np, L, base_roi, d, end_t,
                       max(8.0, 0.7 * box_w))
    conf = (0.45 * coverage + 0.30 * edge + 0.25 * contrast)
    conf *= _blob_factor(cv2, np, mask, base_roi, d, end_t, radial)
    return {
        "shadow_len_px": round(end_t, 1),
        "confidence": round(float(np.clip(conf, 0.0, 1.0)), 3),
        "method": "cv2-segmentation",
    }


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Measure a shadow's pixel length with cv2")
    ap.add_argument("frame", help="path to frame image")
    ap.add_argument("x1", type=float)
    ap.add_argument("y1", type=float)
    ap.add_argument("x2", type=float)
    ap.add_argument("y2", type=float)
    ap.add_argument("--tip", nargs=2, type=float, metavar=("TX", "TY"),
                    help="optional shadow-tip pixel hint (e.g. from the VLM)")
    args = ap.parse_args()
    res = measure_shadow_length(
        args.frame,
        [args.x1, args.y1, args.x2, args.y2],
        tip_hint=args.tip,
    )
    print(json.dumps(res, indent=2) if res is not None else "null")
