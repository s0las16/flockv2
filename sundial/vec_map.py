"""Vector "where it thinks it is" map fragment for the verdict report.

The zero-network sibling of ``map_viz.map_fragment``: instead of pulling
CARTO raster tiles it embeds one inline ``<svg>` drawn entirely from local
Natural-Earth vectors by ``sundial.geo2svg``.  Same summary contract as the
tile map — anchor > predicted > first marker for centring, marker kinds
claim|pred|geo|wiki, lat/lon bands, ranked-country highlight.

Everything is fail-soft: missing/malformed summary pieces degrade to a
world view, an absent or crashing ``geo2svg`` degrades to a dark plate that
still shows the centre coordinates, and ``vec_fragment`` returns "" only
when there is genuinely nothing placeable (no anchor, no prediction, no
markers).  Nothing here ever raises and nothing touches the network.

    python3 sundial/vec_map.py       # self-test (stubs geo2svg)
    python3 -m sundial.vec_map
"""
from __future__ import annotations

import html
import math
import re
from typing import Any

# --------------------------------------------------------------------------
# Tuning constants
# --------------------------------------------------------------------------
_LATLIM = 85.0                 # keep the bbox inside sane display latitudes
_WORLD = (25.0, 0.0)           # fallback centre: Atlantic-centric world view
_SPAN_ANCHOR = 18.0            # deg — claim anchor known -> city-region zoom
_SPAN_WORLD = 170.0            # deg — whole-world fallback
_PRED_MIN, _PRED_MAX = 20.0, 140.0    # clamp for radius-derived span
_MK_MIN, _MK_MAX = 15.0, 90.0         # clamp for marker-covering span
_KM_PER_DEG = 111.0
_PRED_PAD = 2.4                # predicted radius -> span fudge factor
_MK_PAD = 1.5                  # marker extent -> span fudge factor
_PRED_COLOR = "#facc15"        # prediction circle / marker accent
_MAX_MARKERS, _MAX_BANDS, _MAX_LABEL = 64, 32, 120
_KINDS = ("claim", "pred", "geo", "wiki")
_ATTR = "Natural Earth · public domain"
_C_DIM = re.compile(r"^[0-9a-zA-Z%.,() -]{1,24}$")

_CSS = """\
.vmap{position:relative;overflow:hidden;background:#0a0d12;border:1px solid #1c1c1c;border-radius:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.vmap svg{display:block;width:100%;height:100%}
.vmap-attr{position:absolute;right:8px;bottom:6px;color:rgba(148,163,184,.45);font-size:9px;pointer-events:none;z-index:2}
.vmap-fb{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#475569;font-size:11px;letter-spacing:.04em;text-align:center;padding:0 12px}
"""


# --------------------------------------------------------------------------
# Defensive primitives — every reader swallows junk instead of raising
# --------------------------------------------------------------------------
def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _dim(value: Any, default: str) -> str:
    s = str(value or "").strip()
    return s if _C_DIM.match(s) else default


def _latlon(lat: Any, lon: Any) -> tuple[float, float] | None:
    """-> (lat, lon) floats in-range, or None."""
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(la) and math.isfinite(lo)):
        return None
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return None
    return la, lo


def _pt_of(item: Any) -> tuple[float, float] | None:
    """Dict {'lat','lon'} OR (lat, lon) sequence -> point or None."""
    try:
        if isinstance(item, dict):
            return _latlon(item.get("lat"), item.get("lon"))
        return _latlon(item[0], item[1])
    except Exception:
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(f):
        return lo
    return min(max(f, lo), hi)


# --------------------------------------------------------------------------
# Lazy sibling imports — work both as `import sundial.vec_map` (package
# context) and as `python3 sundial/vec_map.py` (script context, where
# `from . import` has no parent package).  Absent siblings degrade, not fail.
# --------------------------------------------------------------------------
def _geo2svg():
    try:
        from . import geo2svg  # type: ignore
        return geo2svg
    except Exception:
        pass
    try:
        import geo2svg  # type: ignore
        return geo2svg
    except Exception:
        return None


def _locate_summary():
    try:
        from . import locate_summary  # type: ignore
        return locate_summary
    except Exception:
        pass
    try:
        import locate_summary  # type: ignore
        return locate_summary
    except Exception:
        return None


# --------------------------------------------------------------------------
# Summary readers
# --------------------------------------------------------------------------
def _anchor_pt(s: dict) -> tuple[float, float] | None:
    return _pt_of(s.get("anchor"))


def _pred_pt(s: dict) -> tuple[float, float] | None:
    return _pt_of(s.get("predicted"))


def _marker_pts(s: dict) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    seq = s.get("markers")
    for m in seq if isinstance(seq, (list, tuple)) else []:
        p = _pt_of(m)
        if p:
            out.append(p)
    return out


def _center(s: dict) -> tuple[float, float]:
    """anchor > predicted > first marker > world view."""
    p = _anchor_pt(s) or _pred_pt(s)
    if p:
        return p
    pts = _marker_pts(s)
    return pts[0] if pts else _WORLD


def _claim_label(s: dict) -> str:
    """The claim_place text, as recorded on the anchor by summarize()."""
    anch = s.get("anchor")
    try:
        if isinstance(anch, dict) and anch.get("label"):
            return str(anch["label"])
    except Exception:
        pass
    return ""


def _claim_cc(s: dict) -> str | None:
    """Best-effort cc for the *claimed* place — derived from the anchor/claim
    label via locate_summary's name->cc index when that sibling is present."""
    label = _claim_label(s)
    if not label:
        try:
            for m in s.get("markers") or []:
                if isinstance(m, dict) and str(m.get("kind")) == "claim":
                    label = str(m.get("label") or "")
                    break
        except Exception:
            pass
    if not label:
        return None
    ls = _locate_summary()
    norm = getattr(ls, "_norm_cc", None) if ls is not None else None
    if callable(norm):
        try:
            cc = norm(label)
            return str(cc).upper() if cc else None
        except Exception:
            return None
    return None


def _highlight_cc(s: dict) -> str | None:
    """Country outline to emphasize: verdict cc > top-ranked > claimed."""
    try:
        cc = s.get("best_cc")
        if cc:
            return str(cc).strip().upper()
        for row in s.get("ranked") or []:
            if isinstance(row, dict) and row.get("cc"):
                return str(row["cc"]).strip().upper()
    except Exception:
        pass
    return _claim_cc(s)


def _circle(s: dict) -> dict | None:
    """Prediction ring for svg_map, from summary['predicted']."""
    pred = s.get("predicted")
    p = _pt_of(pred)
    if not p:
        return None
    r = 0.0
    try:
        r = _clamp(pred.get("radius_km"), 0.0, 40000.0) \
            if isinstance(pred, dict) else 0.0
    except Exception:
        r = 0.0
    return {"lat": p[0], "lon": p[1], "r_km": r, "color": _PRED_COLOR}


def _markers(s: dict) -> list[dict]:
    """Normalize marker rows for svg_map; kinds pass through, labels are
    pre-escaped: claim -> the claim_place text, pred -> 'est.' (unless it
    coincides with the claim anchor, where a second text label is redundant)."""
    out: list[dict] = []
    claim = _claim_label(s)
    anch = _anchor_pt(s)
    seq = s.get("markers")
    for m in (seq if isinstance(seq, (list, tuple)) else [])[: _MAX_MARKERS]:
        try:
            p = _pt_of(m)
            if not p:
                continue
            kind = str(m.get("kind") or "geo").lower() \
                if isinstance(m, dict) else "geo"
            if kind not in _KINDS:
                kind = "geo"
            label = str(m.get("label") or "") if isinstance(m, dict) else ""
            if kind == "claim" and claim:
                label = claim
            elif kind == "pred":
                if anch and abs(p[0] - anch[0]) < 1e-3 and abs(p[1] - anch[1]) < 1e-3:
                    label = ""
                else:
                    label = "est."
            out.append({"lat": round(p[0], 6), "lon": round(p[1], 6),
                        "kind": kind, "label": _e(label[: _MAX_LABEL])})
        except Exception:
            continue
    return out


def _bands(s: dict) -> list:
    seq = s.get("bands")
    return list(seq[: _MAX_BANDS]) if isinstance(seq, (list, tuple)) else []


def _labels(s: dict) -> list[dict]:
    """Free-floating text labels for svg_map.

    _markers(s) already carries labels for the claim and prediction pins,
    so emitting them again here would print every label twice in the SVG.
    """
    return []


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def zoom_span(summary: dict | None) -> float:
    """Pick the map's bbox span in degrees from a locate_summary dict.

    anchor -> 18 (city region); predicted-only -> radius_km/111*2.4 clamped
    [20, 140]; markers-only -> extent covering all markers *1.5 clamped
    [15, 90]; nothing -> 170 (world).
    """
    try:
        s = summary if isinstance(summary, dict) else {}
        if _anchor_pt(s):
            return _SPAN_ANCHOR
        pred = s.get("predicted")
        if _pt_of(pred):
            try:
                r = float(pred.get("radius_km"))  # type: ignore[union-attr]
            except (TypeError, ValueError, AttributeError):
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            return _clamp(r / _KM_PER_DEG * _PRED_PAD, _PRED_MIN, _PRED_MAX)
        pts = _marker_pts(s)
        if pts:
            lats = [p[0] for p in pts]
            lons = [p[1] for p in pts]
            extent = max(max(lats) - min(lats), max(lons) - min(lons))
            return _clamp(extent * _MK_PAD, _MK_MIN, _MK_MAX)
        return _SPAN_WORLD
    except Exception:
        return _SPAN_WORLD


def _bbox(g, clat: float, clon: float, span: float) -> tuple:
    """Prefer geo2svg.auto_bbox; fall back to centre±span/2 (clamped)."""
    try:
        if g is not None and callable(getattr(g, "auto_bbox", None)):
            bb = tuple(g.auto_bbox(clat, clon, span))
            if len(bb) == 4:
                vals = tuple(float(x) for x in bb)
                if all(math.isfinite(v) for v in vals):
                    return vals
    except Exception:
        pass
    try:
        half = _clamp(span, 1.0, 360.0) / 2.0
        return (max(-_LATLIM, clat - half), max(-180.0, clon - half),
                min(_LATLIM, clat + half), min(180.0, clon + half))
    except Exception:
        return (-_LATLIM, -180.0, _LATLIM, 180.0)


def _svg(g, bbox: tuple, width: str, height: str, markers: list,
         bands: list, circle: dict | None, labels: list,
         highlight_cc: str | None) -> str:
    """Call geo2svg.svg_map by keyword, then positionally — the sibling's
    signature is assumed, not imported, so either convention works."""
    fn = getattr(g, "svg_map", None) if g is not None else None
    if not callable(fn):
        return ""
    calls = (
        ((), {"bbox": bbox, "width": width, "height": height,
              "markers": markers, "bands": bands, "circle": circle,
              "labels": labels, "highlight_cc": highlight_cc}),
        ((bbox, width, height, markers, bands, circle, labels,
          highlight_cc), {}),
    )
    for args, kwargs in calls:
        try:
            svg = fn(*args, **kwargs)
            if svg:
                return str(svg)
        except Exception:
            continue
    return ""


def vec_fragment(summary: dict | None, width: str = "100%",
                 height: str = "360px") -> str:
    """Complete HTML fragment: <div class="vmap"> + <style> + inline <svg>.

    Returns "" when the summary carries nothing placeable — no valid
    anchor, no valid prediction and no valid marker.  Never raises.
    """
    try:
        s = summary if isinstance(summary, dict) else {}
        placeable = bool(_anchor_pt(s) or _pred_pt(s) or _marker_pts(s))
        if not placeable:
            return ""
        clat, clon = _center(s)
        span = zoom_span(s)
        g = _geo2svg()
        svg = ""
        if g is not None:
            svg = _svg(g, _bbox(g, clat, clon, span), width, height,
                       _markers(s), _bands(s), _circle(s), _labels(s),
                       _highlight_cc(s))
        if not svg:
            # geo2svg absent or failed -> dark plate carrying the centre.
            svg = ('<div class="vmap-fb">vector map offline &middot; '
                   "centre %.4f&deg;, %.4f&deg;</div>" % (clat, clon))
        w, h = _dim(width, "100%"), _dim(height, "360px")
        return (
            '<div class="vmap" style="width:' + w + ";height:" + h + '">'
            "<style>" + _CSS + "</style>"
            + svg +
            '<div class="vmap-attr">' + _e(_ATTR) + "</div>"
            "</div>"
        )
    except Exception:
        return ""


def _minimal_summary(anchor: Any, claim_place: Any) -> dict:
    """Stand-in for locate_summary.summarize when the sibling is absent:
    anchor-only summary, enough for a claimed-location map."""
    s = {"anchor": None, "markers": [], "bands": [], "ranked": [],
         "best_cc": None, "best_name": None, "predicted": None}
    p = _pt_of(anchor)
    if p:
        label = str(claim_place).strip() if claim_place else "claimed location"
        s["anchor"] = {"lat": round(p[0], 5), "lon": round(p[1], 5),
                       "label": label}
        s["markers"] = [{"lat": round(p[0], 5), "lon": round(p[1], 5),
                         "kind": "claim", "label": label}]
    return s


def fragment_for(adj_or_evidence: Any, anchor: Any = None,
                 claim_place: Any = None) -> str:
    """Convenience wrapper: Adjudication-like object OR raw evidence list ->
    locate_summary.summarize -> vec_fragment.  Never raises."""
    try:
        evidence, a, cp = adj_or_evidence, anchor, claim_place
        if not isinstance(adj_or_evidence, (list, tuple)):
            adj = adj_or_evidence
            pack = getattr(adj, "pack", None)
            evidence = getattr(pack, "evidence", None)
            if evidence is None:
                evidence = getattr(adj, "evidence", None) or []
            if a is None:
                a = getattr(adj, "anchor", None)
            if cp is None:
                cp = getattr(adj, "claim_place", None)
        summary = None
        ls = _locate_summary()
        if ls is not None and callable(getattr(ls, "summarize", None)):
            try:
                summary = ls.summarize(list(evidence or []), a, cp)
            except Exception:
                summary = None
        if not isinstance(summary, dict):
            summary = _minimal_summary(a, cp)
        return vec_fragment(summary)
    except Exception:
        return ""


# --------------------------------------------------------------------------
# Self-test: python3 sundial/vec_map.py   /   python3 -m sundial.vec_map
# --------------------------------------------------------------------------
def _self_test() -> int:
    import sys
    import types
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")

    # Stub geo2svg (sibling may not exist yet) — records what vec_fragment
    # passes so centring/span can be asserted, and emits a real <svg>.
    seen: dict = {}
    stub = types.ModuleType("geo2svg")

    def auto_bbox(clat, clon, span):
        seen["center"] = (float(clat), float(clon))
        seen["span"] = float(span)
        half = float(span) / 2.0
        return (clat - half, clon - half, clat + half, clon + half)

    def svg_map(bbox, width, height, markers, bands, circle, labels,
                highlight_cc):
        seen.update(bbox=bbox, markers=markers, bands=bands,
                    circle=circle, labels=labels, hl=highlight_cc)
        dots = "".join(
            f'<text class="vm-lab">{m.get("label", "")}</text>'
            for m in markers)
        return (f'<svg viewBox="0 0 640 360" data-hl="{highlight_cc}">'
                f"{dots}</svg>")

    stub.auto_bbox = auto_bbox  # type: ignore[attr-defined]
    stub.svg_map = svg_map      # type: ignore[attr-defined]
    sys.modules["geo2svg"] = stub
    sys.modules["sundial.geo2svg"] = stub  # cover `from . import` too

    paris = {"lat": 48.8566, "lon": 2.3522}
    full = {
        "anchor": {"lat": 48.8566, "lon": 2.3522, "label": "Paris <claim>"},
        "markers": [
            {"lat": 48.8566, "lon": 2.3522, "kind": "claim",
             "label": "Paris <claim>"},
            {"lat": 48.7, "lon": 2.4, "kind": "pred",
             "label": "predicted: France"},
            {"lat": 48.9, "lon": 2.5, "kind": "geo", "label": "geo pt"},
            {"lat": "junk", "lon": 2.5, "kind": "wiki", "label": "dropped"},
        ],
        "bands": [{"lat_min": 48.0, "lat_max": 49.0,
                   "lon_min": 1.9, "lon_max": 2.8, "color": "#38bdf8"}],
        "ranked": [{"cc": "FR", "name": "France", "score": 4.2}],
        "best_cc": "FR", "best_name": "France",
        "predicted": {"lat": 48.7, "lon": 2.4, "radius_km": 600.0},
    }

    # -- vec_fragment: shape, centring, escaping -------------------------
    seen.clear()
    frag = vec_fragment(full)
    check("fragment non-empty", isinstance(frag, str) and len(frag) > 100)
    check("fragment has <svg", "<svg" in frag)
    check("fragment has .vmap", 'class="vmap"' in frag)
    check("fragment has style", ".vmap{" in frag and "#0a0d12" in frag)
    check("attribution", 'class="vmap-attr"' in frag
          and "Natural Earth" in frag)
    check("centre = anchor", seen.get("center") == (48.8566, 2.3522),
          f"-> {seen.get('center')}")
    check("span = 18", seen.get("span") == 18.0, f"-> {seen.get('span')}")
    check("claim label escaped", "Paris &lt;claim&gt;" in frag
          and "<claim>" not in frag)
    mk = seen.get("markers") or []
    check("pred label 'est.'", any(m.get("kind") == "pred"
                                  and m.get("label") == "est." for m in mk),
          f"-> {mk}")
    check("bad marker dropped", len(mk) == 3, f"-> {len(mk)}")
    check("circle from predicted",
          (seen.get("circle") or {}).get("r_km") == 600.0
          and seen["circle"]["color"] == "#facc15")
    check("highlight_cc = best_cc", seen.get("hl") == "FR")

    # -- zoom_span math ---------------------------------------------------
    check("zoom: anchor -> 18", zoom_span(full) == 18.0)
    pred_only = {"predicted": {"lat": 40.0, "lon": -3.7,
                               "radius_km": 5000.0}}
    check("zoom: pred 5000km -> ~108.1",
          abs(zoom_span(pred_only) - 5000.0 / 111.0 * 2.4) < 1e-6,
          f"-> {zoom_span(pred_only)}")
    check("zoom: pred small clamps 20",
          zoom_span({"predicted": {"lat": 0, "lon": 0,
                                   "radius_km": 100.0}}) == 20.0)
    check("zoom: pred huge clamps 140",
          zoom_span({"predicted": {"lat": 0, "lon": 0,
                                   "radius_km": 9000.0}}) == 140.0)
    mk_only = {"markers": [{"lat": 40.0, "lon": 0.0},
                           {"lat": 50.0, "lon": 30.0}]}
    check("zoom: markers 30deg*1.5 -> 45", zoom_span(mk_only) == 45.0,
          f"-> {zoom_span(mk_only)}")
    check("zoom: single marker clamps 15",
          zoom_span({"markers": [{"lat": 1.0, "lon": 1.0}]}) == 15.0)
    check("zoom: empty -> 170", zoom_span({}) == 170.0)
    check("zoom: junk -> 170", zoom_span(None) == 170.0
          and zoom_span({"markers": [{"lat": "x"}]}) == 170.0)

    # -- centring fall-through --------------------------------------------
    seen.clear()
    vec_fragment(pred_only)
    check("centre = predicted", seen.get("center") == (40.0, -3.7))
    seen.clear()
    vec_fragment(mk_only)
    check("centre = first marker", seen.get("center") == (40.0, 0.0))

    # -- empty / junk ------------------------------------------------------
    check("empty -> ''", vec_fragment({}) == "")
    check("no placeable -> ''",
          vec_fragment({"ranked": [{"cc": "FR"}], "best_cc": "FR"}) == "")
    check("invalid-only markers -> ''",
          vec_fragment({"markers": [{"lat": "x", "lon": "y"}]}) == "")
    check("junk summary -> ''", vec_fragment(None) == ""
          and vec_fragment("nope") == "")

    # -- fragment_for -------------------------------------------------------
    evs = [
        {"type": "osint", "data": {"ranked": [
            {"cc": "FR", "score": 0.8}, {"cc": "BE", "score": 0.4}]}},
        {"type": "landmark", "data": {"places": [
            {"name": "Eiffel Tower", "lat": 48.8583, "lon": 2.2945}]}},
    ]

    class _Pack:
        evidence = evs

    class _Adj:
        pack = _Pack()
        anchor = (48.8566, 2.3522)
        claim_place = "Paris"

    seen.clear()
    f2 = fragment_for(_Adj())
    check("fragment_for(adj) has <svg", "<svg" in f2)
    check("fragment_for centres on anchor",
          seen.get("center") == (48.8566, 2.3522), f"-> {seen.get('center')}")
    f3 = fragment_for(evs, anchor=(48.8566, 2.3522), claim_place="Paris")
    check("fragment_for(list) has <svg", "<svg" in f3)
    check("fragment_for never raises",
          fragment_for(object()) in ("",) or isinstance(
              fragment_for(object()), str))
    check("fragment_for bare obj -> ''", fragment_for(object()) == "")

    # -- real geo2svg smoke test (if the sibling file has landed) -----------
    for key in ("geo2svg", "sundial.geo2svg"):
        sys.modules.pop(key, None)
    if _geo2svg() is not None:
        f4 = vec_fragment(full)
        check("real geo2svg -> svg or dark plate",
              "<svg" in f4 or "vmap-fb" in f4)
        check("real geo2svg fragment keeps chrome",
              'class="vmap"' in f4 and "Natural Earth" in f4)
    else:
        print("  SKIP  real geo2svg not present yet (stub covered the API)")

    print(f"vec_map self-test: {'ALL PASS' if not fails else f'{fails} FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
