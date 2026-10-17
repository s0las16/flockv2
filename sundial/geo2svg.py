"""Offline world-map renderer — dark landmass SVG from local geojson.

Data: the same Natural Earth 110m admin-0 file used by pip2cc
(``sundial/data/world110m.geojson``, ~0.8 MB, 177 features, committed).
No network, no tiles, no keys, no JS — just ``json`` + ``math`` + ``html``.
``_load()`` parses the file once into a module-level cache of
``(cc, [polys])`` where each poly is ``(minx, miny, maxx, maxy, area_deg2,
rings)`` and each ring is ``[(lon, lat), ...]``; holes are kept and drawn
with ``fill-rule="evenodd"`` so e.g. Lesotho stays a hole in South Africa.

Projection is plain equirectangular inside a ``(min_lat, min_lon,
max_lat, max_lon)`` bbox: lon→x linear, lat→y linear with y flipped
(max_lat at top). Longitudes are normalized into the frame
``[min_lon, min_lon + lon_span)`` so a bbox with ``min_lon > max_lon``
crosses the antimeridian correctly; rings are split (new ``M`` subpath)
whenever consecutive framed lons jump >180°.

To keep output small, emitted coordinates are quantized to 0.1 px and
consecutive points closer than ``simplify`` px (default 0.15) are dropped
— no topological simplification. Polygons covering <0.4 % of the bbox
area are skipped unless they belong to ``highlight_cc``.

Public API never raises: bad input and a missing/corrupt data file yield
"" / empty strings.

    python3 sundial/geo2svg.py     # self-test
    python3 -m sundial.geo2svg
"""
from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

__all__ = ["paths_for_bbox", "project", "svg_map", "auto_bbox"]

_DATA_PATH = Path(__file__).resolve().parent / "data" / "world110m.geojson"

# Palette / styling constants (match the rest of the sundial dark UI).
_BG = "#0a0d12"
_LAND = "#161b22"
_LAND_HL = "#26303c"
_STROKE = "#2c3440"
_GRAT = "rgba(255,255,255,.04)"
_LBL = "#94a3b8"

# Default viewport used by paths_for_bbox when no ``proj`` is passed —
# identical to svg_map()'s defaults so bare path output drops straight
# into a ``viewBox="0 0 560 340"`` wrapper.
_DEF_W, _DEF_H = 560.0, 340.0

_MIN_PX = 0.15      # drop consecutive projected points closer than this
_TINY_FRAC = 0.004  # skip polygons < 0.4 % of bbox area (unless highlighted)
_GRAT_STEP = 10.0   # graticule spacing in degrees

# Same -99-ISO repair table as pip2cc (duplicated, not imported, so this
# module stays self-contained): N. Cyprus, Somaliland, Kosovo.
_ADM0_A3_CC = {"CYN": "CY", "SOL": "SO", "KOS": "XK"}

_FEATURES: list | None = None  # module-level cache for _load()


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def _feature_cc(props: dict):
    """ISO alpha-2 for one feature's properties, or None (pip2cc logic)."""
    for key in ("ISO_A2_EH", "ISO_A2"):
        v = str(props.get(key) or "").strip().upper()
        if len(v) == 2 and v.isalpha():
            return v
    v = _ADM0_A3_CC.get(str(props.get("ADM0_A3") or "").strip().upper())
    if v:
        return v
    for key in ("WB_A2", "POSTAL", "FIPS_10"):
        v = str(props.get(key) or "").strip().upper()
        if len(v) == 2 and v.isalpha():
            return v
    return None


def _load() -> list:
    """Cached ``[(cc, [polys])]`` from the geojson; [] when unreadable.

    Each poly is ``(minx, miny, maxx, maxy, area_deg2, rings)`` with
    ``rings = [[(lon, lat), ...], ...]`` — ring 0 is the outer ring, the
    rest are holes. ``area_deg2`` is the shoelace area of the outer ring,
    used only for the tiny-polygon cutoff. Never raises.
    """
    global _FEATURES
    if _FEATURES is not None:
        return _FEATURES
    feats: list = []
    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as fh:
            gj = json.load(fh)
        raw = gj.get("features") if isinstance(gj, dict) else None
        for feat in raw or ():
            try:
                props = feat.get("properties") or {}
                geom = feat.get("geometry") or {}
                coords = geom.get("coordinates") or []
                if geom.get("type") == "Polygon":
                    sources = [coords]
                elif geom.get("type") == "MultiPolygon":
                    sources = coords
                else:
                    continue
                polys = []
                for poly in sources:
                    rings = []
                    for ring in poly or ():
                        pts = []
                        ok = True
                        for pt in ring or ():
                            try:
                                pts.append((float(pt[0]), float(pt[1])))
                            except (TypeError, ValueError, IndexError):
                                ok = False
                                break
                        if ok and len(pts) >= 3:
                            rings.append(pts)
                    if not rings:
                        continue
                    outer = rings[0]
                    xs = [p[0] for p in outer]
                    ys = [p[1] for p in outer]
                    area = abs(sum(
                        outer[i][0] * outer[i + 1][1]
                        - outer[i + 1][0] * outer[i][1]
                        for i in range(len(outer) - 1)
                    )) / 2.0
                    polys.append(
                        (min(xs), min(ys), max(xs), max(ys), area, rings)
                    )
                if polys:
                    feats.append((_feature_cc(props) or "", polys))
            except Exception:  # noqa: BLE001 — skip a single bad feature
                continue
    except Exception:  # noqa: BLE001 — file missing/corrupt -> empty map
        feats = []
    _FEATURES = feats
    return _FEATURES


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def _parse_bbox(bbox):
    """``(min_lat, min_lon, max_lat, max_lon, lat_span, lon_span)`` or None.

    ``lon_span`` is ``(max_lon - min_lon) mod 360`` (0 → 360), so
    ``min_lon > max_lon`` means the bbox crosses the antimeridian.
    """
    try:
        a, b, c, d = bbox
        min_lat, min_lon = float(a), float(b)
        max_lat, max_lon = float(c), float(d)
    except (TypeError, ValueError):
        return None
    vals = (min_lat, min_lon, max_lat, max_lon)
    if not all(math.isfinite(v) for v in vals):
        return None
    lat_span = max_lat - min_lat
    lon_span = (max_lon - min_lon) % 360.0
    if lon_span == 0.0:
        lon_span = 360.0
    if lat_span <= 0.0 or lat_span > 180.0:
        return None
    return (min_lat, min_lon, max_lat, max_lon, lat_span, lon_span)


def _flon(lon: float, min_lon: float) -> float:
    """Normalize ``lon`` into the frame ``[min_lon, min_lon + 360)``."""
    return min_lon + ((lon - min_lon) % 360.0)


def _f(v: float) -> str:
    """Quantize to 0.1 px and drop a trailing ``.0`` (smaller path data)."""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _pip(fx: float, y: float, ring) -> bool:
    """Even-odd ray cast of ``(fx, y)`` against ring ``[(lon, lat), ...]``
    with lons already frame-normalized (used for the
    bbox-inside-one-country case)."""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    px, py = ring[n - 1]
    for cx, cy in ring:
        if (cy > y) != (py > y) and fx < (px - cx) * (y - cy) / (py - cy) + cx:
            inside = not inside
        px, py = cx, cy
    return inside


def _poly_visible(rings, min_lat, min_lon, max_lat, lat_span, lon_span):
    """Rough bbox-intersection test for one polygon in framed coords."""
    outer = rings[0]
    pts = [(f[0], f[1]) for f in outer]
    if min(p[1] for p in pts) > max_lat or max(p[1] for p in pts) < min_lat:
        return False
    framed = [(_flon(p[0], min_lon), p[1]) for p in pts]
    if min(fx for fx, _ in framed) <= lon_span:
        return True
    # No vertex inside the lon range — the polygon may still swallow the
    # whole view (tiny bbox deep inside a big country). Test the four
    # bbox corners plus center with a cheap point-in-ring cast.
    cy = min_lat + lat_span * 0.5
    cx = min_lon + lon_span * 0.5
    tests = (
        (cx, cy),
        (_flon(min_lon, min_lon), min_lat),
        (_flon(min_lon + lon_span, min_lon), min_lat),
        (_flon(min_lon, min_lon), max_lat),
        (_flon(min_lon + lon_span, min_lon), max_lat),
    )
    return any(_pip(fx, fy, framed) for fx, fy in tests)


def _poly_path(rings, proj, min_lon: float, min_dist: float) -> str:
    """Path ``d`` attribute for one polygon's rings (ring0 + holes).

    Points are dropped when closer than ``min_dist`` px to the last kept
    point; a >180° framed-lon jump between consecutive points starts a
    new subpath (``M``) instead of drawing a wraparound chord. The single
    trailing ``Z`` per ring closes it for fill (subpaths auto-close).
    """
    md2 = min_dist * min_dist
    segs = []
    for ring in rings:
        d = []
        started = False
        kept = 0
        lx = ly = lfl = 0.0
        for lon, lat in ring:
            flon = _flon(lon, min_lon)
            try:
                x, y = proj(lat, lon)
            except Exception:  # noqa: BLE001 — odd proj callable
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if not started or abs(flon - lfl) > 180.0:
                d.append(f"M{_f(x)},{_f(y)}")
            elif (x - lx) * (x - lx) + (y - ly) * (y - ly) >= md2:
                d.append(f"L{_f(x)},{_f(y)}")
                kept += 1
            else:
                continue  # too close to the last kept point — drop
            started = True
            lx, ly, lfl = x, y, flon
        if kept >= 2:  # M + >=2 L segments => a real area
            segs.append("".join(d) + "Z")
    return "".join(segs)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def project(lat, lon, bbox, width, height) -> tuple[float, float]:
    """Equirectangular projection of (lat, lon) into the SVG viewBox.

    lon→x linear inside the bbox's wrapped lon frame (points outside map
    to x ≥ width, i.e. off-canvas right — never wrap back onto the map);
    lat→y linear, inverted so max_lat is y=0. Degenerate input → (0, 0).
    """
    try:
        parsed = _parse_bbox(bbox)
        if parsed is None:
            return (0.0, 0.0)
        min_lat, min_lon, max_lat, _max_lon, lat_span, lon_span = parsed
        w, h = float(width), float(height)
        if not (math.isfinite(w) and math.isfinite(h)) or w <= 0 or h <= 0:
            return (0.0, 0.0)
        x = ((float(lon) - min_lon) % 360.0) / lon_span * w
        y = (max_lat - float(lat)) / lat_span * h
        if not (math.isfinite(x) and math.isfinite(y)):
            return (0.0, 0.0)
        return (x, y)
    except Exception:  # noqa: BLE001 — spec: never raise
        return (0.0, 0.0)


def paths_for_bbox(bbox, proj=None, simplify=None, highlight_cc=None) -> str:
    """SVG ``<path>`` elements for every country polygon intersecting bbox.

    ``proj`` is an optional ``callable(lat, lon) -> (x, y)``; default is
    :func:`project` into the standard 560×340 viewport. ``simplify`` is
    the minimum distance between kept points in px (default 0.15).
    Features whose cc matches ``highlight_cc`` get ``class="land hl"``.
    Returns "" on bad input or missing data — never raises.
    """
    try:
        parsed = _parse_bbox(bbox)
        if parsed is None:
            return ""
        min_lat, min_lon, max_lat, _ml, lat_span, lon_span = parsed
        feats = _load()
        if not feats:
            return ""
        if proj is None:
            bb = (min_lat, min_lon, max_lat, _ml)

            def proj(la, lo):
                return project(la, lo, bb, _DEF_W, _DEF_H)
        try:
            md = _MIN_PX if simplify is None else max(0.0, float(simplify))
        except (TypeError, ValueError):
            md = _MIN_PX
        hl = str(highlight_cc).strip().upper() if highlight_cc else ""
        bbox_area = lat_span * lon_span
        tiny = _TINY_FRAC * bbox_area
        out = []
        for cc, polys in feats:
            is_hl = bool(hl) and cc == hl
            cls = ' class="land hl"' if is_hl else ' class="land"'
            for minx, miny, maxx, maxy, area, rings in polys:
                if not is_hl and area < tiny:
                    continue
                if not _poly_visible(
                    rings, min_lat, min_lon, max_lat, lat_span, lon_span
                ):
                    continue
                d = _poly_path(rings, proj, min_lon, md)
                if d:
                    out.append(f"<path{cls} d=\"{d}\"/>")
        return "".join(out)
    except Exception:  # noqa: BLE001 — spec: never raise
        return ""


def auto_bbox(center_lat, center_lon, span_deg=40) -> tuple:
    """``(min_lat, min_lon, max_lat, max_lon)`` centered on a point.

    The latitude window (``span_deg`` tall, capped at 170) is slid to fit
    inside [-85, 85]; longitudes are wrapped to [-180, 180), so
    ``min_lon > max_lon`` signals an antimeridian crossing. ``span_deg``
    ≥ 360 yields the full lon range. Bad input → whole-world bbox.
    """
    fallback = (-85.0, -180.0, 85.0, 180.0)
    try:
        clat = float(center_lat)
        clon = float(center_lon)
        span = abs(float(span_deg))
    except (TypeError, ValueError):
        return fallback
    if not all(math.isfinite(v) for v in (clat, clon, span)):
        return fallback
    if span == 0.0:
        span = 40.0
    # lat: window of min(span, 170) slid into [-85, 85]
    lspan = min(span, 170.0)
    half = lspan / 2.0
    min_lat = clat - half
    max_lat = clat + half
    if min_lat < -85.0:
        max_lat += -85.0 - min_lat
        min_lat = -85.0
    if max_lat > 85.0:
        min_lat -= max_lat - 85.0
        max_lat = 85.0
    min_lat = max(-85.0, min_lat)
    # lon: wrapped to [-180, 180); min>max == antimeridian crossing
    if span >= 360.0:
        min_lon, max_lon = -180.0, 180.0
    else:
        min_lon = ((clon - span / 2.0 + 180.0) % 360.0) - 180.0
        max_lon = ((clon + span / 2.0 + 180.0) % 360.0) - 180.0
    return (min_lat, min_lon, max_lat, max_lon)


def svg_map(bbox, width=560, height=340, markers=None, bands=None,
            circle=None, labels=None, highlight_cc=None) -> str:
    """Self-contained ``<svg ...>...</svg>`` fragment of the world in bbox.

    Dark theme: bg #0a0d12, landmass #161b22 (#26303c highlighted),
    stroke #2c3440/.5, 10° graticule at rgba(255,255,255,.04).

    ``markers``: ``[{"lat","lon","kind","label"}]`` — claim=white filled
    r5 + ring, pred=#facc15 ring r6, geo=#38bdf8 dot r3.5, wiki=#94a3b8
    dot r3; ``label`` is drawn as escaped ``<text>``.
    ``bands``: ``[{"lat_min","lat_max","lon_min","lon_max","color"}]``
    translucent rects. ``circle``: ``{"lat","lon","r_km","color"}``
    dashed ellipse (r_km/111 degrees → projected px per axis).
    ``labels``: ``[{"lat","lon","text"}]`` or ``(lat, lon, text)``.
    Returns "" on bad input or missing data — never raises.
    """
    try:
        parsed = _parse_bbox(bbox)
        if parsed is None:
            return ""
        min_lat, min_lon, max_lat, max_lon, lat_span, lon_span = parsed
        try:
            w = float(width)
            h = float(height)
        except (TypeError, ValueError):
            w, h = _DEF_W, _DEF_H
        if not (math.isfinite(w) and math.isfinite(h)) or w <= 0 or h <= 0:
            return ""
        if not _load():
            return ""
        bb = (min_lat, min_lon, max_lat, max_lon)

        def proj(la, lo):
            return project(la, lo, bb, w, h)

        def esc(v) -> str:
            return html.escape(str(v), quote=True)

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {_f(w)} {_f(h)}" '
            f'width="{_f(w)}" height="{_f(h)}" role="img">',
            "<style>"
            f".bg{{fill:{_BG}}}"
            f".grat line{{stroke:{_GRAT};stroke-width:.5}}"
            f"path.land{{fill:{_LAND};stroke:{_STROKE};stroke-width:.5;"
            "fill-rule:evenodd}"
            f"path.land.hl{{fill:{_LAND_HL}}}"
            f"text.lbl{{fill:{_LBL};font-family:ui-monospace,"
            "SFMono-Regular,Menlo,monospace;font-size:10px;"
            f"paint-order:stroke;stroke:{_BG};stroke-width:3;"
            "stroke-linejoin:round}"
            "</style>",
            f'<rect class="bg" x="0" y="0" width="{_f(w)}" '
            f'height="{_f(h)}"/>',
        ]

        # --- graticule every 10° ---------------------------------------
        g = []
        k = math.ceil(min_lat / _GRAT_STEP)
        while _GRAT_STEP * k <= max_lat:
            _x, y = proj(_GRAT_STEP * k, min_lon)
            g.append(
                f'<line x1="0" y1="{_f(y)}" x2="{_f(w)}" y2="{_f(y)}"/>'
            )
            k += 1
        k = math.ceil(min_lon / _GRAT_STEP)
        while _GRAT_STEP * k <= min_lon + lon_span:
            x, _y = proj(min_lat, _GRAT_STEP * k)
            g.append(
                f'<line x1="{_f(x)}" y1="0" x2="{_f(x)}" y2="{_f(h)}"/>'
            )
            k += 1
        if g:
            parts.append('<g class="grat">' + "".join(g) + "</g>")

        # --- landmass ---------------------------------------------------
        parts.append(
            paths_for_bbox(bb, proj=proj, highlight_cc=highlight_cc)
        )

        # --- translucent lat/lon bands ----------------------------------
        for b in bands or ():
            try:
                x0, y0 = proj(float(b["lat_max"]), float(b["lon_min"]))
                x1, y1 = proj(float(b["lat_min"]), float(b["lon_max"]))
                x, y = min(x0, x1), min(y0, y1)
                parts.append(
                    f'<rect x="{_f(x)}" y="{_f(y)}" '
                    f'width="{_f(abs(x1 - x0))}" '
                    f'height="{_f(abs(y1 - y0))}" '
                    f'fill="{esc(b.get("color", "#38bdf8"))}" '
                    f'opacity="0.16"/>'
                )
            except Exception:  # noqa: BLE001 — skip one bad band
                continue

        # --- dashed search-radius circle --------------------------------
        if circle:
            try:
                cx, cy = proj(float(circle["lat"]), float(circle["lon"]))
                rd = abs(float(circle.get("r_km", 0.0))) / 111.0
                color = esc(circle.get("color", "#facc15"))
                rx = rd / lon_span * w
                ry = rd / lat_span * h
                parts.append(
                    f'<ellipse cx="{_f(cx)}" cy="{_f(cy)}" '
                    f'rx="{_f(rx)}" ry="{_f(ry)}" fill="{color}" '
                    f'fill-opacity="0.07" stroke="{color}" '
                    f'stroke-width="1.2" stroke-dasharray="5 4"/>'
                )
            except Exception:  # noqa: BLE001 — skip a bad circle
                pass

        # --- markers -----------------------------------------------------
        for m in markers or ():
            try:
                x, y = proj(float(m["lat"]), float(m["lon"]))
            except Exception:  # noqa: BLE001 — skip a bad marker
                continue
            kind = str(m.get("kind", "")).strip().lower()
            c = f'cx="{_f(x)}" cy="{_f(y)}"'
            if kind == "claim":
                parts.append(
                    f'<circle {c} r="8" fill="none" stroke="#ffffff" '
                    f'stroke-opacity="0.55" stroke-width="1"/>'
                    f'<circle {c} r="5" fill="#ffffff"/>'
                )
            elif kind == "pred":
                parts.append(
                    f'<circle {c} r="6" fill="none" stroke="#facc15" '
                    f'stroke-width="1.6"/>'
                )
            elif kind == "geo":
                parts.append(f'<circle {c} r="3.5" fill="#38bdf8"/>')
            elif kind == "wiki":
                parts.append(f'<circle {c} r="3" fill="#94a3b8"/>')
            else:
                parts.append(f'<circle {c} r="3" fill="#e2e8f0"/>')
            lab = m.get("label")
            if lab:
                parts.append(
                    f'<text class="lbl" x="{_f(x + 9)}" '
                    f'y="{_f(y + 3.5)}">{esc(lab)}</text>'
                )

        # --- standalone labels ------------------------------------------
        for item in labels or ():
            try:
                if isinstance(item, dict):
                    la = item.get("lat")
                    lo = item.get("lon")
                    t = item.get("text", item.get("label", ""))
                else:
                    la, lo, t = item[0], item[1], item[2]
                x, y = proj(float(la), float(lo))
                parts.append(
                    f'<text class="lbl" text-anchor="middle" '
                    f'x="{_f(x)}" y="{_f(y)}">{esc(t)}</text>'
                )
            except Exception:  # noqa: BLE001 — skip a bad label
                continue

        parts.append("</svg>")
        return "".join(parts)
    except Exception:  # noqa: BLE001 — spec: never raise
        return ""


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    fails = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        global fails
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label} {detail}")

    print("geo2svg self-test:")

    # --- auto_bbox math ---------------------------------------------------
    bb = auto_bbox(52.0, 13.4, 40)
    check("auto_bbox centers lat", bb[0] < 52.0 < bb[2] and
          abs((bb[2] + bb[0]) / 2 - 52.0) < 0.01, f"(got {bb})")
    check("auto_bbox span", abs((bb[2] - bb[0]) - 40) < 0.01 and
          abs(((bb[3] - bb[1]) % 360) - 40) < 0.01)
    bb_np = auto_bbox(84.0, 0.0, 40)
    check("auto_bbox clamps lat to 85",
          bb_np[2] == 85.0 and bb_np[0] >= -85.0, f"(got {bb_np})")
    bb_am = auto_bbox(0.0, 179.0, 20)
    check("auto_bbox wraps lon (min>max)",
          bb_am[1] > 0 > bb_am[3], f"(got {bb_am})")
    check("auto_bbox bad input -> world",
          auto_bbox("x", None) == (-85.0, -180.0, 85.0, 180.0))
    check("auto_bbox span>=360 -> full lon",
          auto_bbox(0, 0, 400)[1:4:2] == (-180.0, 180.0))

    # --- project() corners -------------------------------------------------
    bb = (40.0, -10.0, 60.0, 30.0)
    check("project top-left", project(60, -10, bb, 400, 200) == (0.0, 0.0))
    check("project bottom-right",
          project(40, 30, bb, 400, 200) == (400.0, 200.0))
    check("project center", project(50, 10, bb, 400, 200) == (200.0, 100.0))
    # antimeridian bbox: lon 170..-170 => span 20; -175 sits at x=W*3/4
    bba = (50.0, 170.0, 70.0, -170.0)
    xw, _yw = project(60, -175, bba, 400, 200)
    check("project wrapped lon", abs(xw - 300.0) < 0.01,
          f"(x={xw})")
    check("project off-frame lon off-canvas",
          project(60, 0, bba, 400, 200)[0] > 400.0)
    check("project degenerate bbox -> (0,0)",
          project(1, 1, (5, 5, 5, 5), 100, 100) == (0.0, 0.0))

    # --- dataset -----------------------------------------------------------
    feats = _load()
    check(f"dataset loaded ({len(feats)} features)",
          len(feats) > 150, f"from {_DATA_PATH}")

    # --- paths_for_bbox ----------------------------------------------------
    de_bb = auto_bbox(51.0, 10.5, 30)
    p = paths_for_bbox(de_bb, highlight_cc="DE")
    check("paths_for_bbox emits <path>", "<path" in p,
          f"({len(p)} chars)")
    check("highlight class applied", 'class="land hl"' in p)
    # antimeridian: a ring crossing ±180 splits into M-subpaths instead of
    # drawing a chord across the map
    seam_d = _poly_path(
        [[(170, 10), (179, 10), (-179, 10), (-170, 10), (170, 10)]],
        lambda la, lo: project(la, lo, (-85, -180, 85, 180), 360, 170),
        -180.0, 0.0)
    check("antimeridian ring split",
          seam_d.count("M") == 3 and "L" in seam_d, f"({seam_d!r})")
    plain_d = _poly_path(
        [[(10, 50), (12, 51), (14, 50), (10, 50)]],
        lambda la, lo: project(la, lo, (40, -10, 60, 30), 400, 200),
        -10.0, 0.15)
    check("normal ring single subpath", plain_d.count("M") == 1)

    # --- svg_map -------------------------------------------------------------
    svg = svg_map(de_bb, highlight_cc="DE",
                  markers=[{"lat": 52.52, "lon": 13.4, "kind": "claim",
                            "label": "Berlin <claimed>"}],
                  bands=[{"lat_min": 45, "lat_max": 55, "lon_min": 5,
                          "lon_max": 15, "color": "#38bdf8"}],
                  circle={"lat": 52.5, "lon": 13.4, "r_km": 300,
                          "color": "#facc15"},
                  labels=[{"lat": 48, "lon": 11, "text": "Alps"}])
    check("svg_map returns <svg", svg.startswith("<svg") and
          "</svg>" in svg, f"({len(svg)} chars)")
    check("svg has bg + graticule + land",
          'class="bg"' in svg and "grat" in svg and "path" in svg)
    check("svg highlight", 'land hl' in svg)
    check("marker rendered (circle + escaped label)",
          "<circle" in svg and "Berlin &lt;claimed&gt;" in svg)
    check("band rendered", 'opacity="0.16"' in svg)
    check("dashed circle rendered", 'stroke-dasharray="5 4"' in svg)
    check("standalone label rendered", ">Alps</text>" in svg)
    check("inline style embedded", "<style>" in svg and "#0a0d12" in svg)
    # world view must render without explosions
    world = svg_map((-85.0, -180.0, 85.0, 180.0))
    check("world bbox renders", world.startswith("<svg"),
          f"({len(world)} chars)")

    # --- graceful degradation ------------------------------------------------
    saved_path, saved_feats = _DATA_PATH, _FEATURES
    try:
        _DATA_PATH = Path("/nonexistent/definitely-missing.geojson")
        _FEATURES = None
        check("missing file -> ''",
              svg_map(de_bb) == "" and paths_for_bbox(de_bb) == "")
    finally:
        _DATA_PATH, _FEATURES = saved_path, saved_feats
    check("garbage bbox -> ''", svg_map("nope") == "" and
          svg_map((9, 9, 9, 9)) == "" and paths_for_bbox(None) == "")
    check("bad markers/bands never raise",
          svg_map(de_bb, markers=[{"lat": "x"}], bands=[{"nope": 1}],
                  circle={"lat": None}, labels=[123]).startswith("<svg"))

    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        sys.exit(1)
    print("OK — geo2svg self-test passed")
