"""Geocell predictor — fuse scored country candidates into one map point.

GeoCLIP-gallery idea, weights-free: instead of a learned retrieval head,
take the top-k scored country candidates (the match_score `ranked` shape,
[{cc, score, ...}]) and blend their centroids on the sphere, weighted by
normalized score. The result is a circle, not a pin: `radius_km` is the
score-weighted mean distance of those centroids from the blended point,
clamped to [200, 3000] km — a tight cluster yields a small circle, a
scattered top-k (US + JP + BR) yields a huge, honest one.

Centroids come from `sundial.locate_summary.CC_GEO` when that module is
importable; otherwise a compact built-in ~60-country table is used, so
the stage works standalone. Pure stdlib (math only at import), no
network, never raises — `predict_point` returns None when no candidate
can be placed on the map.

Stage API:

    pred = predict_point(ranked)                 # {lat, lon, radius_km, ...} | None
    ev   = pred_evidence(pred, (lat, lon))       # SUPPORTS if anchor inside radius
"""
from __future__ import annotations

import math

_MIN_RADIUS_KM = 200.0    # a prediction is a region, never a pin
_MAX_RADIUS_KM = 3000.0   # ...but it must still discriminate
_RADIUS_EDGE = 2.0        # anchor beyond 2x radius -> contradicts
_CONF = 0.45              # supports/contradicts confidence (corroborating)
_NEAR_MISS_CONF = 0.20    # anchor 1-2x radius out -> inconclusive
_UNCOMPARED_CONF = 0.15   # prediction exists, but no anchor to check

# ---------------------------------------------------------------------------
# Built-in country centroids ({cc: {lat, lon, name}}) — used only when
# sundial.locate_summary.CC_GEO is unavailable. ~60 countries covering the
# match_score candidate universe; rough geographic centres, ~100km-accurate.
# ---------------------------------------------------------------------------

_CC_GEO_FALLBACK: dict[str, dict] = {
    # --- Europe ---
    "DE": {"lat": 51.16, "lon": 10.45, "name": "Germany"},
    "FR": {"lat": 46.60, "lon": 2.40, "name": "France"},
    "IT": {"lat": 42.80, "lon": 12.80, "name": "Italy"},
    "ES": {"lat": 40.20, "lon": -3.60, "name": "Spain"},
    "PT": {"lat": 39.60, "lon": -8.00, "name": "Portugal"},
    "NL": {"lat": 52.20, "lon": 5.30, "name": "Netherlands"},
    "BE": {"lat": 50.65, "lon": 4.65, "name": "Belgium"},
    "AT": {"lat": 47.60, "lon": 14.13, "name": "Austria"},
    "CH": {"lat": 46.80, "lon": 8.23, "name": "Switzerland"},
    "PL": {"lat": 52.00, "lon": 19.40, "name": "Poland"},
    "CZ": {"lat": 49.80, "lon": 15.50, "name": "Czechia"},
    "SK": {"lat": 48.70, "lon": 19.50, "name": "Slovakia"},
    "HU": {"lat": 47.16, "lon": 19.50, "name": "Hungary"},
    "RO": {"lat": 45.85, "lon": 24.97, "name": "Romania"},
    "BG": {"lat": 42.75, "lon": 25.50, "name": "Bulgaria"},
    "GR": {"lat": 39.00, "lon": 22.00, "name": "Greece"},
    "HR": {"lat": 45.10, "lon": 15.40, "name": "Croatia"},
    "UA": {"lat": 49.00, "lon": 32.00, "name": "Ukraine"},
    "BY": {"lat": 53.53, "lon": 28.05, "name": "Belarus"},
    "RU": {"lat": 60.00, "lon": 90.00, "name": "Russia"},
    "EE": {"lat": 58.67, "lon": 25.00, "name": "Estonia"},
    "LV": {"lat": 56.88, "lon": 24.60, "name": "Latvia"},
    "LT": {"lat": 55.35, "lon": 23.88, "name": "Lithuania"},
    "SE": {"lat": 62.00, "lon": 15.00, "name": "Sweden"},
    "NO": {"lat": 64.50, "lon": 12.00, "name": "Norway"},
    "FI": {"lat": 64.00, "lon": 26.00, "name": "Finland"},
    "DK": {"lat": 56.00, "lon": 10.00, "name": "Denmark"},
    "IS": {"lat": 64.90, "lon": -18.60, "name": "Iceland"},
    "IE": {"lat": 53.18, "lon": -8.24, "name": "Ireland"},
    "GB": {"lat": 54.00, "lon": -2.00, "name": "United Kingdom"},
    "MT": {"lat": 35.94, "lon": 14.41, "name": "Malta"},
    "CY": {"lat": 35.00, "lon": 33.00, "name": "Cyprus"},
    "TR": {"lat": 39.00, "lon": 35.00, "name": "Turkey"},
    # --- Americas ---
    "US": {"lat": 39.50, "lon": -98.35, "name": "United States"},
    "CA": {"lat": 56.00, "lon": -96.00, "name": "Canada"},
    "MX": {"lat": 23.60, "lon": -102.50, "name": "Mexico"},
    "CU": {"lat": 21.50, "lon": -79.50, "name": "Cuba"},
    "JM": {"lat": 18.10, "lon": -77.30, "name": "Jamaica"},
    "CR": {"lat": 9.90, "lon": -84.00, "name": "Costa Rica"},
    "PA": {"lat": 8.50, "lon": -80.80, "name": "Panama"},
    "CO": {"lat": 4.00, "lon": -73.00, "name": "Colombia"},
    "VE": {"lat": 7.00, "lon": -66.00, "name": "Venezuela"},
    "PE": {"lat": -9.20, "lon": -75.00, "name": "Peru"},
    "CL": {"lat": -31.00, "lon": -71.00, "name": "Chile"},
    "AR": {"lat": -34.00, "lon": -64.00, "name": "Argentina"},
    "BR": {"lat": -10.80, "lon": -52.90, "name": "Brazil"},
    # --- Asia / Middle East ---
    "KZ": {"lat": 48.00, "lon": 68.00, "name": "Kazakhstan"},
    "MN": {"lat": 46.80, "lon": 103.00, "name": "Mongolia"},
    "CN": {"lat": 35.00, "lon": 105.00, "name": "China"},
    "JP": {"lat": 36.50, "lon": 138.00, "name": "Japan"},
    "KR": {"lat": 36.50, "lon": 127.90, "name": "South Korea"},
    "TW": {"lat": 23.80, "lon": 121.00, "name": "Taiwan"},
    "IN": {"lat": 22.00, "lon": 79.00, "name": "India"},
    "PK": {"lat": 30.00, "lon": 69.00, "name": "Pakistan"},
    "BD": {"lat": 23.80, "lon": 90.30, "name": "Bangladesh"},
    "LK": {"lat": 7.90, "lon": 80.70, "name": "Sri Lanka"},
    "NP": {"lat": 28.20, "lon": 84.00, "name": "Nepal"},
    "TH": {"lat": 15.50, "lon": 101.00, "name": "Thailand"},
    "VN": {"lat": 14.00, "lon": 108.00, "name": "Vietnam"},
    "MY": {"lat": 4.20, "lon": 102.00, "name": "Malaysia"},
    "SG": {"lat": 1.35, "lon": 103.85, "name": "Singapore"},
    "ID": {"lat": -2.20, "lon": 117.00, "name": "Indonesia"},
    "PH": {"lat": 12.80, "lon": 122.00, "name": "Philippines"},
    "SA": {"lat": 24.00, "lon": 45.00, "name": "Saudi Arabia"},
    "AE": {"lat": 24.00, "lon": 54.00, "name": "United Arab Emirates"},
    "IL": {"lat": 31.00, "lon": 34.80, "name": "Israel"},
    "IR": {"lat": 32.50, "lon": 54.00, "name": "Iran"},
    "IQ": {"lat": 33.00, "lon": 44.00, "name": "Iraq"},
    # --- Africa ---
    "EG": {"lat": 26.80, "lon": 30.80, "name": "Egypt"},
    "MA": {"lat": 31.80, "lon": -6.00, "name": "Morocco"},
    "DZ": {"lat": 28.00, "lon": 2.50, "name": "Algeria"},
    "TN": {"lat": 34.00, "lon": 9.50, "name": "Tunisia"},
    "ET": {"lat": 9.00, "lon": 39.50, "name": "Ethiopia"},
    "KE": {"lat": 0.50, "lon": 37.90, "name": "Kenya"},
    "TZ": {"lat": -6.40, "lon": 34.90, "name": "Tanzania"},
    "NG": {"lat": 9.60, "lon": 8.00, "name": "Nigeria"},
    "GH": {"lat": 7.90, "lon": -1.20, "name": "Ghana"},
    "ZA": {"lat": -29.00, "lon": 25.00, "name": "South Africa"},
    # --- Oceania ---
    "AU": {"lat": -25.00, "lon": 134.00, "name": "Australia"},
    "NZ": {"lat": -41.80, "lon": 172.80, "name": "New Zealand"},
    "FJ": {"lat": -17.80, "lon": 178.00, "name": "Fiji"},
    "PG": {"lat": -6.30, "lon": 147.00, "name": "Papua New Guinea"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(value) -> float | None:
    """Best-effort finite float; None when unparseable."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km — same formula as geo._haversine_km
    (copied per repo convention: stages never import each other)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _centroid(entry) -> tuple[float, float] | None:
    """Pull (lat, lon) out of a cc_geo entry — tolerates {lat, lon, ...}
    dicts and plain (lat, lon) / (lat, lon, name) sequences."""
    lat = lon = None
    if isinstance(entry, dict):
        lat, lon = _num(entry.get("lat")), _num(entry.get("lon"))
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        lat, lon = _num(entry[0]), _num(entry[1])
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def _default_cc_geo() -> dict:
    """locate_summary.CC_GEO when importable; built-in table otherwise.

    Tries the package-relative import first, then the bare sibling import
    (covers `python3 sundial/geocell_pred.py`, where the script's own
    directory sits on sys.path). Never raises.
    """
    for src in (".locate_summary", "locate_summary"):
        try:
            if src.startswith("."):
                from .locate_summary import CC_GEO  # type: ignore
            else:
                from locate_summary import CC_GEO  # type: ignore
            if isinstance(CC_GEO, dict) and CC_GEO:
                return CC_GEO
        except Exception:
            continue
    return _CC_GEO_FALLBACK


def _schema():
    """Lazy schema import so the module stays dep-free at import time.

    Falls back to a same-shape shim when schema can't be found at all
    (odd packaging), so `pred_evidence` never dies on import context.
    """
    try:
        from .schema import ClaimVerdict, Evidence
        return ClaimVerdict, Evidence
    except Exception:
        pass
    try:
        from schema import ClaimVerdict, Evidence  # plain-script mode
        return ClaimVerdict, Evidence
    except Exception:
        pass
    from dataclasses import dataclass, field

    class ClaimVerdict:  # duck-typed stand-in for schema.ClaimVerdict
        SUPPORTS = "supports"
        CONTRADICTS = "contradicts"
        INCONCLUSIVE = "inconclusive"

    @dataclass
    class Evidence:  # minimal schema.Evidence shape
        type: str
        summary: str
        verdict: object = ClaimVerdict.INCONCLUSIVE
        confidence: float = 0.0
        data: dict = field(default_factory=dict)
        frames: list = field(default_factory=list)

    return ClaimVerdict, Evidence


# ---------------------------------------------------------------------------
# Stage API
# ---------------------------------------------------------------------------

def predict_point(
    ranked: list[dict],
    cc_geo: dict | None = None,
    top_k: int = 3,
) -> dict | None:
    """Blend the top-k scored country candidates into one predicted point.

    `ranked` is the match_score output shape: [{cc, score, ...}], already
    sorted best-first (sorted again here defensively). Each candidate's
    country centroid contributes to a spherical weighted mean with weight
    = its normalized score (uniform when every score is <= 0). Returns:

        {"lat", "lon", "radius_km", "top_cc", "confidence_01"}

    where `radius_km` is the score-weighted mean distance of the used
    centroids from the point, clamped to [200, 3000] km, and
    `confidence_01` is the winning centroid's share of the total weight
    (1.0 when unopposed). None when no candidate resolves to a centroid.
    """
    try:
        geo = cc_geo if isinstance(cc_geo, dict) and cc_geo else _default_cc_geo()
        cands = []
        for e in (ranked or []):
            if not isinstance(e, dict):
                continue
            cc = str(e.get("cc") or e.get("country_code") or "").strip().upper()
            cent = _centroid(geo.get(cc)) if cc else None
            if cent is None:
                continue
            score = _num(e.get("score"))
            cands.append((cc, cent[0], cent[1], score if score is not None else 0.5))
        if not cands:
            return None

        cands.sort(key=lambda t: t[3], reverse=True)
        try:
            k = max(1, int(top_k))
        except (TypeError, ValueError):
            k = 3
        used = cands[:k]

        weights = [max(0.0, s) for _, _, _, s in used]
        wsum = sum(weights)
        if wsum <= 0.0:  # all-zero/negative scores -> uniform blend
            weights = [1.0] * len(used)
            wsum = float(len(used))
        weights = [w / wsum for w in weights]

        # Spherical weighted mean of the centroids (unit-vector sum).
        x = y = z = 0.0
        for w, (_, lat, lon, _) in zip(weights, used):
            p, l = math.radians(lat), math.radians(lon)
            x += w * math.cos(p) * math.cos(l)
            y += w * math.cos(p) * math.sin(l)
            z += w * math.sin(p)
        norm = math.sqrt(x * x + y * y + z * z)
        if norm > 1e-9:
            lat = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
            lon = math.degrees(math.atan2(y, x))
        else:  # degenerate antipodal balance — plain weighted mean
            lat = sum(w * la for w, (_, la, _, _) in zip(weights, used))
            lon = sum(w * lo for w, (_, _, lo, _) in zip(weights, used))

        radius = sum(
            w * _haversine_km(lat, lon, la, lo)
            for w, (_, la, lo, _) in zip(weights, used)
        )
        radius = min(_MAX_RADIUS_KM, max(_MIN_RADIUS_KM, radius))

        return {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "radius_km": round(radius, 1),
            "top_cc": used[0][0],
            "confidence_01": round(weights[0], 3),
        }
    except Exception:
        return None


def pred_evidence(pred: dict | None, anchor: tuple | None) -> "Evidence":
    """Package a predict_point circle vs a claimed anchor as Evidence.

    `anchor` is a (lat, lon) pair for the claimed place. Verdicts:
      * anchor inside radius          -> SUPPORTS 0.45
        ("predicted region contains claimed point")
      * anchor beyond 2x radius       -> CONTRADICTS 0.45
      * 1-2x radius out (near miss)   -> INCONCLUSIVE 0.20
      * no usable pred or no anchor   -> INCONCLUSIVE (0.0 / 0.15)

    Confidence is capped at 0.45: a centroid blend corroborates a claim,
    it never convicts alone.
    """
    ClaimVerdict, Evidence = _schema()
    try:
        p = pred if isinstance(pred, dict) else {}
        plat, plon = _num(p.get("lat")), _num(p.get("lon"))
        radius = _num(p.get("radius_km")) or 0.0
        a = list(anchor) if isinstance(anchor, (list, tuple)) else []
        alat = _num(a[0]) if len(a) > 0 else None
        alon = _num(a[1]) if len(a) > 1 else None
        data = {
            "pred": p or None,
            "anchor": [alat, alon] if (alat is not None and alon is not None) else None,
            "distance_km": None,
        }

        if plat is None or plon is None:
            return Evidence(
                type="geo_pred",
                summary="no map prediction available — nothing to check the claim against",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                data=data,
            )
        if alat is None or alon is None:
            return Evidence(
                type="geo_pred",
                summary=(
                    f"predicted {plat:.2f},{plon:.2f} ±{radius:.0f} km "
                    f"(top {p.get('top_cc')}) — but no claimed point to compare"
                ),
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=_UNCOMPARED_CONF,
                data=data,
            )

        d = _haversine_km(plat, plon, alat, alon)
        data["distance_km"] = round(d, 1)
        radius = max(radius, 1.0)

        if d <= radius:
            return Evidence(
                type="geo_pred",
                summary=(
                    f"predicted region contains claimed point — anchor "
                    f"{d:.0f} km from centre, inside the {radius:.0f} km "
                    f"radius (top {p.get('top_cc')})"
                ),
                verdict=ClaimVerdict.SUPPORTS,
                confidence=_CONF,
                data=data,
            )
        if d > _RADIUS_EDGE * radius:
            return Evidence(
                type="geo_pred",
                summary=(
                    f"claimed point lies {d:.0f} km from the predicted "
                    f"region — over {_RADIUS_EDGE:.0f}x its {radius:.0f} km "
                    f"radius (top {p.get('top_cc')})"
                ),
                verdict=ClaimVerdict.CONTRADICTS,
                confidence=_CONF,
                data=data,
            )
        return Evidence(
            type="geo_pred",
            summary=(
                f"claimed point sits {d:.0f} km out — just outside the "
                f"predicted {radius:.0f} km radius; ambiguous"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=_NEAR_MISS_CONF,
            data=data,
        )
    except Exception:
        try:
            return Evidence(
                type="geo_pred",
                summary="geo prediction check failed — inconclusive",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                data={},
            )
        except Exception:  # pragma: no cover — shim Evidence cannot fail
            return None


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/geocell_pred.py  (or python -m sundial.geocell_pred)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    fails = 0

    def check(name, got, want):
        global fails
        ok = got == want
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} (want {want!r})")

    def check_true(name, got):
        global fails
        if not got:
            fails += 1
        print(f"  {'PASS' if got else 'FAIL'}  {name}: got {got!r}")

    # --- predict_point -----------------------------------------------------
    ranked_de = [
        {"cc": "DE", "score": 0.72},
        {"cc": "AT", "score": 0.55},
        {"cc": "CZ", "score": 0.51},
        {"cc": "US", "score": 0.40},
    ]
    pred = predict_point(ranked_de)
    print("DE-heavy prediction:", pred)
    check_true("pred returned", isinstance(pred, dict))
    check_true("lands in central Europe (lat)",
               pred is not None and 44.0 <= pred["lat"] <= 55.0)
    check_true("lands in central Europe (lon)",
               pred is not None and 4.0 <= pred["lon"] <= 20.0)
    check("top candidate", pred["top_cc"] if pred else None, "DE")
    check_true("radius within clamps",
               pred is not None and 200.0 <= pred["radius_km"] <= 3000.0)
    check_true("confidence in 0..1",
               pred is not None and 0.0 < pred["confidence_01"] <= 1.0)
    check_true("JSON-safe", bool(json.dumps(pred)))

    check("empty ranked -> None", predict_point([]), None)
    check("None ranked -> None", predict_point(None), None)
    check("garbage ranked -> None", predict_point("nonsense"), None)
    check("unknown ccs only -> None",
          predict_point([{"cc": "XX", "score": 0.9}]), None)

    solo = predict_point([{"cc": "JP", "score": 0.9}])
    check("single candidate radius = min clamp",
          solo["radius_km"] if solo else None, 200.0)
    check("single candidate confidence", solo["confidence_01"], 1.0)

    custom_geo = {"AA": {"lat": 10.0, "lon": 20.0, "name": "Testland"}}
    cust = predict_point([{"cc": "AA", "score": 1.0}], cc_geo=custom_geo)
    check("custom cc_geo honoured",
          (cust["lat"], cust["lon"]) if cust else None, (10.0, 20.0))

    # --- pred_evidence -----------------------------------------------------
    ClaimVerdict, Evidence = _schema()
    verdict_of = lambda ev: getattr(ev.verdict, "value", ev.verdict)

    inside = pred_evidence(pred, (pred["lat"], pred["lon"]))
    check("anchor at centre -> supports", verdict_of(inside), "supports")
    check("supports confidence", inside.confidence, 0.45)
    check_true("inside summary", "predicted region contains claimed point"
               in inside.summary)
    check_true("evidence data JSON-safe", bool(json.dumps(inside.data)))

    far = pred_evidence(pred, (-33.87, 151.21))  # Sydney — far outside
    check("far anchor -> contradicts", verdict_of(far), "contradicts")
    check("contradicts confidence", far.confidence, 0.45)

    near = pred_evidence(pred, (52.52, 13.40))  # Berlin — near miss band
    check("near-miss anchor -> inconclusive", verdict_of(near), "inconclusive")

    check("no pred -> inconclusive",
          verdict_of(pred_evidence(None, (50.0, 10.0))), "inconclusive")
    check("no anchor -> inconclusive",
          verdict_of(pred_evidence(pred, None)), "inconclusive")

    print(f"self-check: {'FAILED ' + str(fails) if fails else 'all passed'}")
    sys.exit(1 if fails else 0)
