"""Offline point-in-polygon country lookup — (lat, lon) -> ISO alpha-2 cc.

Data: Natural Earth 110m admin-0 countries GeoJSON, fetched once into
``sundial/data/world110m.geojson`` (~0.8 MB, 177 features). The only
network this module ever does is ``update_data()``, which re-fetches that
file and is never called implicitly — urllib is imported lazily inside it.
Everything else is offline: ``_load()`` parses the file once and caches
``[(cc, name, polygons)]`` where each polygon is ``(minx, miny, maxx,
maxy, ring)`` and ``ring`` is a flat ``[lon, lat, lon, lat, ...]`` list —
the outer ring only; interior holes are skipped per spec.

If the data file is missing or corrupt, ``_load()`` falls back to a tiny
embedded 2-country polygon set (rough FR + ES boxes) so the module still
answers something — real lookups then simply degrade to those two.

Public API never raises: bad input and missing data yield None / [].

    python3 sundial/pip2cc.py     # self-test
    python -m sundial.pip2cc
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

__all__ = ["cc_at", "country_at", "nearest_cc", "update_data"]

_DATA_PATH = Path(__file__).resolve().parent / "data" / "world110m.geojson"

# Primary source plus a jsDelivr mirror of the same Natural Earth repo.
SOURCES = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_admin_0_countries.geojson",
    "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/"
    "geojson/ne_110m_admin_0_countries.geojson",
)
USER_AGENT = "sundial-hackathon/0.1 (video provenance verification)"
TIMEOUT_S = 20
_MAX_BYTES = 2_000_000  # expected ~0.8 MB; refuse anything much larger
_EARTH_R_KM = 6371.0088

# admin_0 features with -99/bogus ISO codes in NE 110m, keyed by ADM0_A3:
# N. Cyprus has no ISO code (treated as CY), Somaliland is ISO Somalia,
# Kosovo uses the user-assigned XK (its ISO_A2_EH already is XK, kept as
# belt-and-braces in case a different NE vintage ships -99 there too).
_ADM0_A3_CC = {"CYN": "CY", "SOL": "SO", "KOS": "XK"}

# Last-resort embedded polygons (very rough outer boxes) so _load() never
# comes back empty even with the geojson absent. FR box: lat 42.3-51.1;
# ES box: lat 35.9-43.8 — they overlap only in the Pyrenees sliver, which
# neither self-test point touches.
_FALLBACK_FEATURES = (
    (
        "FR",
        "France",
        [[(-5.2, 42.3), (8.3, 42.4), (8.3, 51.1), (-4.9, 51.1), (-5.2, 42.3)]],
    ),
    (
        "ES",
        "Spain",
        [[(-9.4, 35.9), (3.4, 35.9), (4.4, 43.7), (-1.9, 43.8), (-9.4, 35.9)]],
    ),
)

_COUNTRIES: list | None = None  # module-level cache for _load()


def _pack_ring(ring) -> tuple | None:
    """Pack an outer ring ``[[lon, lat], ...]`` -> ``(bbox, flat)`` shape.

    Returns ``(minx, miny, maxx, maxy, [x, y, x, y, ...])`` or None when
    the ring is malformed or has fewer than 3 vertices.
    """
    flat: list[float] = []
    for pt in ring or ():
        try:
            flat.append(float(pt[0]))
            flat.append(float(pt[1]))
        except (TypeError, ValueError, IndexError):
            return None
    if len(flat) < 6:
        return None
    xs, ys = flat[0::2], flat[1::2]
    return (min(xs), min(ys), max(xs), max(ys), flat)


def _feature_cc(props: dict):
    """ISO alpha-2 for one feature's properties, or None.

    ISO_A2_EH first (NE's corrected column), then ISO_A2, then the small
    ADM0_A3 override table for the -99 stragglers, then WB_A2 / POSTAL /
    FIPS_10 as a last resort. POSTAL must come after the override: NE has
    POSTAL="CN" for N. Cyprus and "SL" for Somaliland — both wrong as cc.
    """
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


def _fallback_countries() -> list:
    """The embedded mini polygon set, in _load()'s shape."""
    out = []
    for cc, name, rings in _FALLBACK_FEATURES:
        polys = [p for p in (_pack_ring(r) for r in rings) if p]
        if polys:
            out.append((cc, name, polys))
    return out


def _load() -> list:
    """Cached ``[(cc, name, [polygons])]`` from the geojson.

    Each polygon is ``(minx, miny, maxx, maxy, ring)`` with ``ring`` a flat
    ``[lon, lat, ...]`` list of the outer ring only (holes skipped). Falls
    back to the embedded FR/ES set when the file is absent/unreadable.
    Never raises; may return an empty list only if even the fallback broke.
    """
    global _COUNTRIES
    if _COUNTRIES is not None:
        return _COUNTRIES
    countries: list = []
    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as fh:
            gj = json.load(fh)
        feats = gj.get("features") if isinstance(gj, dict) else None
        for feat in feats or ():
            try:
                props = feat.get("properties") or {}
                cc = _feature_cc(props)
                name = (
                    props.get("ADMIN")
                    or props.get("NAME")
                    or props.get("NAME_EN")
                    or cc
                )
                geom = feat.get("geometry") or {}
                coords = geom.get("coordinates") or []
                if geom.get("type") == "Polygon":
                    sources = [coords]
                elif geom.get("type") == "MultiPolygon":
                    sources = coords
                else:
                    sources = []
                polys = []
                for poly in sources:
                    outer = (poly or [None])[0]  # ring 0 = outer; holes skipped
                    packed = _pack_ring(outer)
                    if packed:
                        polys.append(packed)
                if cc and polys:
                    countries.append((cc, str(name), polys))
            except Exception:  # noqa: BLE001 — skip a single bad feature
                continue
    except Exception:  # noqa: BLE001 — file missing/corrupt -> fallback
        countries = []
    if not countries:
        countries = _fallback_countries()
    _COUNTRIES = countries
    return _COUNTRIES


def update_data(urls=SOURCES, path=_DATA_PATH):
    """Fetch the countries geojson to ``path``; ``{"path","bytes","url"}`` or None.

    The only function here that touches the network, and only when called
    explicitly. urllib is imported lazily; any failure moves to the next
    source URL and ultimately returns None. Refuses payloads > _MAX_BYTES
    and anything that isn't a non-empty FeatureCollection. On success the
    _load() cache is reset so the next lookup uses the fresh file.
    """
    global _COUNTRIES
    try:
        import urllib.request
    except Exception:  # noqa: BLE001 — no urllib?! offline exotic env
        return None
    if isinstance(urls, str):
        urls = (urls,)
    for url in tuple(urls or ()):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                body = r.read(_MAX_BYTES + 1)
            if not body or len(body) > _MAX_BYTES:
                continue
            gj = json.loads(body)
            feats = gj.get("features") if isinstance(gj, dict) else None
            if not feats:
                continue
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(body)
            tmp.replace(path)
            _COUNTRIES = None  # force reload on next lookup
            return {"path": str(path), "bytes": len(body), "url": url}
        except Exception:  # noqa: BLE001 — try next source
            continue
    return None


def _norm_coords(lat, lon):
    """Validated ``(lat, lon)`` floats, lon wrapped to [-180, 180); None if bad."""
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(la) and math.isfinite(lo)):
        return None
    if la < -90.0 or la > 90.0:
        return None
    return (la, ((lo + 180.0) % 360.0) - 180.0)


def _in_ring(x: float, y: float, ring: list) -> bool:
    """Even-odd ray cast: is (x=lon, y=lat) inside flat ring [x0,y0,...]?"""
    inside = False
    n = len(ring)
    if n < 6:
        return False
    px, py = ring[n - 2], ring[n - 1]
    for i in range(0, n, 2):
        cx, cy = ring[i], ring[i + 1]
        if (cy > y) != (py > y) and x < (px - cx) * (y - cy) / (py - cy) + cx:
            inside = not inside
        px, py = cx, cy
    return inside


def _hit(lat, lon):
    """``(cc, name)`` of the containing country, or None (ocean/invalid)."""
    pt = _norm_coords(lat, lon)
    if pt is None:
        return None
    la, lo = pt
    for cc, name, polys in _load():
        for minx, miny, maxx, maxy, ring in polys:
            if (
                minx <= lo <= maxx
                and miny <= la <= maxy
                and _in_ring(lo, la, ring)
            ):
                return (cc, name)
    return None


def cc_at(lat: float, lon: float):
    """ISO alpha-2 country code at (lat, lon), or None over ocean/bad input.

    Ray-casts each country's outer rings (multipolygons handled via their
    parts, interior holes ignored). First hit wins. Never raises.
    """
    try:
        h = _hit(lat, lon)
        return h[0] if h else None
    except Exception:  # noqa: BLE001 — spec: never raise
        return None


def country_at(lat, lon):
    """``{"cc", "name"}`` at (lat, lon), or None over ocean/bad input."""
    try:
        h = _hit(lat, lon)
        return {"cc": h[0], "name": h[1]} if h else None
    except Exception:  # noqa: BLE001 — spec: never raise
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = (
        math.sin(dp * 0.5) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    )
    return 2.0 * _EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))


def nearest_cc(lat, lon, k: int = 3) -> list:
    """k nearest countries by approximate border distance.

    Intended for coastal points where ``cc_at`` misses: for every country
    the minimum haversine distance from the point to that country's ring
    vertices is taken (rings are vertex-sampled, ~128 max per ring, so the
    km value is an approximation — the true border may be closer). Each
    entry is ``{"cc", "name", "km", "inside"}`` sorted by km; ``inside``
    is True when the point actually lies in that country's polygon, so a
    caller can still spot a definite hit. [] on bad input; never raises.
    """
    try:
        pt = _norm_coords(lat, lon)
        if pt is None:
            return []
        la, lo = pt
        try:
            kk = max(1, int(k))
        except (TypeError, ValueError):
            kk = 3
        best = []
        for cc, name, polys in _load():
            d = None
            inside = False
            for minx, miny, maxx, maxy, ring in polys:
                if (
                    minx <= lo <= maxx
                    and miny <= la <= maxy
                    and _in_ring(lo, la, ring)
                ):
                    inside = True
                step = max(1, (len(ring) // 2) // 128)
                for i in range(0, len(ring) - 1, 2 * step):
                    dd = _haversine_km(la, lo, ring[i + 1], ring[i])
                    if d is None or dd < d:
                        d = dd
            if d is not None:
                best.append((d, cc, name, inside))
        best.sort(key=lambda t: t[0])
        return [
            {"cc": cc, "name": name, "km": round(d, 1), "inside": inside}
            for d, cc, name, inside in best[:kk]
        ]
    except Exception:  # noqa: BLE001 — spec: never raise
        return []


if __name__ == "__main__":
    # Self-test: Paris->FR, Madrid->ES, mid-Atlantic->None + nearest_cc.
    # Prints OK/FAIL per check; exit 0 only when everything passed.
    ok = True
    try:
        countries = _load()
        src = (
            str(_DATA_PATH)
            if _DATA_PATH.exists() and len(countries) > 2
            else "embedded fallback (data file missing/unreadable)"
        )
        print(f"{len(countries)} countries loaded from {src}")
        for la, lo, want in ((48.85, 2.35, "FR"), (40.4, -3.7, "ES")):
            got = cc_at(la, lo)
            good = got == want
            ok = ok and good
            print(f"{'OK' if good else 'FAIL'} cc_at({la}, {lo}) = {got} (want {want})")
            print("     country_at ->", country_at(la, lo))
        ocean = cc_at(0.0, -30.0)
        near = nearest_cc(0.0, -30.0)
        good = ocean is None and len(near) >= 1
        ok = ok and good
        print(f"{'OK' if good else 'FAIL'} ocean (0, -30): cc_at={ocean} nearest={near}")
        bad = cc_at("x", None)
        good = bad is None and nearest_cc(None, None) == []
        ok = ok and good
        print(f"{'OK' if good else 'FAIL'} bad input -> {bad}")
    except Exception as e:  # noqa: BLE001 — self-test must not crash either
        ok = False
        print(f"FAIL self-test raised: {e}")
    sys.exit(0 if ok else 1)
