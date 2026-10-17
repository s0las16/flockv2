"""Horizon stage — estimate camera downward tilt from image geometry.

The VLM's per-frame `camera_tilt_deg` guess is unreliable, so this module
recovers the horizon classically instead: Canny edges -> probabilistic Hough
segments, keep the near-horizontal ones, then extend them pairwise. Lines
that are parallel in 3D meet at a shared vanishing point, and the vanishing
points of horizontal 3D directions all land on the horizon row y_h — so the
modal row of pairwise intersections is the horizon. Its offset from frame
center gives the downward tilt

    tilt = atan((h/2 - y_h) / f_px),   f_px = h / (2*tan(vfov/2))

with vfov defaulting to 55 deg (typical webcam/dashcam vertical FOV) unless
the frame's EXIF focal length says otherwise. Same sign convention as the
VLM field: positive tilt = camera looking down, 0 = level.

Everything heavy (cv2, numpy, PIL) is imported lazily inside functions so
this module imports with zero third-party deps.
"""
from __future__ import annotations

import importlib.util
import math

__all__ = ["estimate_tilt_deg"]

_VFOV_DEG_DEFAULT = 55.0   # typical webcam/dashcam vertical FOV
_MAX_EDGE_PX = 960         # downscale frames beyond this before detection
_MAX_LINE_ANGLE_DEG = 15.0  # "near-horizontal" segment filter: |angle| <= this
_MIN_PAIR_ANGLE_DEG = 1.0  # pairs need a baseline to triangulate a VP
_MAX_SEGMENTS = 80         # cap for the O(n^2) pairing step
_MIN_SEGMENTS = 4          # below this there is nothing to intersect
_MIN_PAIRS = 6             # eligible pairs below this -> give up
_MIN_INLIERS = 5           # VP pairs inside the modal row band
_MIN_AGREEMENT = 0.40      # inlier weight / total weight below -> disagree
_VP_ROW_LO, _VP_ROW_HI = -2.0, 3.0  # candidate VP rows, multiples of h
_MAX_TILT_DEG = 75.0       # |tilt| beyond this is a degenerate cluster


def _exif_vfov_deg(frame_path: str) -> float:
    """Vertical FOV in degrees from EXIF; falls back to _VFOV_DEG_DEFAULT.

    Tries FocalLengthIn35mmFilm first (most common), then real focal length
    + FocalPlaneYResolution to reconstruct sensor height. Any failure ->
    the default; EXIF is absent entirely on cv2-written frames anyway.
    """
    try:
        from PIL import Image
    except ImportError:
        return _VFOV_DEG_DEFAULT
    try:
        with Image.open(frame_path) as img:
            _, h = img.size
            exif = img.getexif()
    except Exception:
        return _VFOV_DEG_DEFAULT

    def _num(tag: int) -> float | None:
        try:
            v = exif.get(tag)
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    f35 = _num(41989)  # FocalLengthIn35mmFilm, mm
    if f35 and f35 > 0:
        # 35mm stills gate is 36x24mm -> vertical FOV uses the 24mm side
        return math.degrees(2.0 * math.atan(12.0 / f35))

    f_mm = _num(37386)   # FocalLength, mm
    fpy = _num(41489)    # FocalPlaneYResolution, px per resolution unit
    unit_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}.get(exif.get(41488))
    if f_mm and f_mm > 0 and fpy and fpy > 0 and unit_mm and h > 0:
        sensor_h_mm = h * unit_mm / fpy
        return math.degrees(2.0 * math.atan(sensor_h_mm / (2.0 * f_mm)))
    return _VFOV_DEG_DEFAULT


def _near_horizontal_segments(gray) -> list[tuple]:
    """Canny + HoughLinesP; keep near-horizontal segments.

    Returns [(x1, y1, x2, y2, angle_deg, length_px)] sorted longest-first
    and capped at _MAX_SEGMENTS to bound the pairwise step.
    """
    import cv2  # lazy

    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = max(24.0, 0.05 * w)
    lines = cv2.HoughLinesP(
        edges,
        1.0,
        math.pi / 180.0,
        threshold=max(24, int(0.6 * min_len)),
        minLineLength=min_len,
        maxLineGap=max(3.0, 0.15 * min_len),
    )
    segs = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = (float(v) for v in line)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            if ang > 90.0:
                ang -= 180.0
            elif ang <= -90.0:
                ang += 180.0
            if abs(ang) <= _MAX_LINE_ANGLE_DEG:
                segs.append((x1, y1, x2, y2, ang, length))
    segs.sort(key=lambda s: -s[5])
    return segs[:_MAX_SEGMENTS]


def _dominant_vp_row(segs: list[tuple], w: int, h: int) -> tuple | None:
    """Intersect near-horizontal pairs; return the modal vanishing-point row.

    Each pair of segments extends to an intersection point. Pairs that are
    parallel in 3D (the two edges of a roofline, curb, shelf, crosswalk...)
    meet at a real vanishing point on the horizon row y_h; pairs that merely
    cross in the image scatter their intersections all over. A weighted
    sliding-window mode over the candidate rows finds the dominant band.

    Returns (y_h, agreement, n_inliers, n_pairs) or None when there is
    nothing to cluster.
    """
    import numpy as np  # lazy

    # homogeneous lines A x + B y + C = 0 with a unit normal: for unit-norm
    # (A, B) the cross-product denominator equals sin(angle between lines),
    # so the _MIN_PAIR_ANGLE_DEG gate also bounds conditioning.
    lines = []
    for x1, y1, x2, y2, ang, length in segs:
        a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
        n = math.hypot(a, b) or 1.0
        lines.append((a / n, b / n, c / n, ang, length))

    ys, ws = [], []
    n_pairs = 0
    total_w = 0.0
    lo, hi = _VP_ROW_LO * h, _VP_ROW_HI * h
    for i in range(len(lines)):
        a1, b1, c1, ang1, len1 = lines[i]
        for j in range(i + 1, len(lines)):
            a2, b2, c2, ang2, len2 = lines[j]
            d = abs(ang1 - ang2)
            if d < _MIN_PAIR_ANGLE_DEG:
                continue
            den = a1 * b2 - a2 * b1
            if abs(den) < 1e-9:
                continue
            n_pairs += 1
            # long segments carrying a well-triangulated VP weigh most;
            # small angle baselines amplify endpoint noise so they count less
            weight = math.sqrt(len1 * len2) * min(1.0, d / 5.0)
            total_w += weight
            yv = (c1 * a2 - c2 * a1) / den
            if lo <= yv <= hi:
                ys.append(yv)
                ws.append(weight)

    if n_pairs < _MIN_PAIRS or not ys or total_w <= 0:
        return None
    ys = np.asarray(ys)
    ws = np.asarray(ws)

    # densest fixed-width band: optimal window can always be shifted so its
    # left edge touches a data point, so a two-pointer sweep over sorted
    # candidates finds it exactly
    order = np.argsort(ys)
    ys_s, ws_s = ys[order], ws[order]
    csw = np.concatenate(([0.0], np.cumsum(ws_s)))
    band = max(4.0, 0.03 * h)  # half-width of the modal row window
    best_w, best_i = -1.0, 0
    j = 0
    for i in range(len(ys_s)):
        if j < i:
            j = i
        while j < len(ys_s) and ys_s[j] <= ys_s[i] + 2.0 * band:
            j += 1
        wsum = csw[j] - csw[i]
        if wsum > best_w:
            best_w, best_i = wsum, i

    in_band = (ys >= ys_s[best_i]) & (ys <= ys_s[best_i] + 2.0 * band)
    n_inliers = int(in_band.sum())
    if n_inliers == 0:
        return None
    agreement = float(ws[in_band].sum() / total_w)

    # weighted median of inlier rows is the horizon estimate
    yi, wi = ys[in_band], ws[in_band]
    order_i = np.argsort(yi)
    yi, wi = yi[order_i], wi[order_i]
    cum = np.cumsum(wi)
    y_h = float(yi[np.searchsorted(cum, 0.5 * cum[-1])])
    return y_h, agreement, n_inliers, n_pairs


def estimate_tilt_deg(
    frame_path: str, vfov_deg: float | None = None
) -> dict | None:
    """Estimate downward camera tilt for one frame.

    -> {"tilt_deg": float, "horizon_y": float, "confidence": float}

    `tilt_deg` > 0 means the camera looks down (same convention as the VLM's
    camera_tilt_deg field); `horizon_y` is the horizon row in full-res
    pixels. Returns None when there is no stable horizon — too few usable
    lines, or vanishing-point candidates that disagree with each other.
    Raises RuntimeError when cv2 is not installed.
    """
    if importlib.util.find_spec("cv2") is None:
        raise RuntimeError(
            "opencv (cv2) is not installed — run `pip install opencv-python` "
            "(needed for horizon.estimate_tilt_deg)."
        )
    import cv2  # lazy

    img = cv2.imread(frame_path)
    if img is None:
        return None
    h_full, w_full = img.shape[:2]
    if h_full < 8 or w_full < 8:
        return None
    scale = min(1.0, _MAX_EDGE_PX / float(max(h_full, w_full)))
    if scale < 1.0:
        img = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    segs = _near_horizontal_segments(gray)
    if len(segs) < _MIN_SEGMENTS:
        return None
    res = _dominant_vp_row(segs, w, h)
    if res is None:
        return None
    y_h, agreement, n_inliers, _n_pairs = res
    if n_inliers < _MIN_INLIERS or agreement < _MIN_AGREEMENT:
        return None  # lines disagree — no stable horizon

    vfov = vfov_deg if vfov_deg else _exif_vfov_deg(frame_path)
    vfov = min(max(float(vfov), 10.0), 150.0)  # sanity-clamp EXIF weirdness
    f_px = h_full / (2.0 * math.tan(math.radians(vfov) / 2.0))
    y_h_full = y_h / scale
    tilt = math.degrees(math.atan2(h_full / 2.0 - y_h_full, f_px))
    if abs(tilt) > _MAX_TILT_DEG:
        return None  # horizon effectively at infinity — degenerate cluster

    coverage = min(1.0, n_inliers / 25.0)
    density = min(1.0, len(segs) / float(_MAX_SEGMENTS))
    conf = 0.15 + 0.45 * agreement + 0.25 * coverage + 0.15 * density
    return {
        "tilt_deg": round(tilt, 2),
        "horizon_y": round(y_h_full, 1),
        "confidence": round(min(max(conf, 0.05), 0.95), 3),
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(
        description="Estimate camera downward tilt from a single frame."
    )
    ap.add_argument("frame", help="path to a sampled frame (jpg/png)")
    ap.add_argument(
        "--vfov",
        type=float,
        default=None,
        help="override vertical FOV in degrees (default: EXIF, else 55)",
    )
    args = ap.parse_args()
    try:
        est = estimate_tilt_deg(args.frame, vfov_deg=args.vfov)
    except RuntimeError as e:
        sys.exit(f"error: {e}")
    if est is None:
        print(
            "no stable horizon estimate — too few usable lines, or "
            "vanishing-point candidates disagree"
        )
    else:
        print(json.dumps(est, indent=2))
