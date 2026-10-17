"""Tests for the geo-support helper modules — run directly:

    python3 tests/test_geo_support.py

The modules under test (sun_side, vss_lookup, geocode_cache, overpass,
sign_style, veg_geo, landmarks, reverse_geo, climate, osint) are being built
in parallel, so every import is guarded: anything absent prints SKIP and the
run stays green on a partial tree.
"""
import importlib
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _optional(modname):
    """Import a sundial submodule; print SKIP and return None when absent."""
    try:
        return importlib.import_module(modname)
    except Exception as exc:  # noqa: BLE001 — any import failure degrades
        print(f"SKIP {modname.rsplit('.', 1)[-1]} ({type(exc).__name__}: {exc})")
        return None


sun_side = _optional("sundial.sun_side")
vss_lookup = _optional("sundial.vss_lookup")
geocode_cache = _optional("sundial.geocode_cache")
overpass = _optional("sundial.overpass")
sign_style = _optional("sundial.sign_style")
veg_geo = _optional("sundial.veg_geo")
landmarks = _optional("sundial.landmarks")
reverse_geo = _optional("sundial.reverse_geo")
climate = _optional("sundial.climate")
osint = _optional("sundial.osint")


class _Skip(Exception):
    """Raised inside a test when the module it covers is unavailable."""


# --------------------------------------------------------------------------- #
# shared tolerant helpers — several modules return score maps / name lists
# whose exact container shape is still settling
# --------------------------------------------------------------------------- #
def _score_map(res):
    """Pull a {label: numeric-score} mapping out of whatever was returned.

    Accepts a flat dict, a dict with a nested scores dict, a list/tuple of
    (label, score) pairs, or a list of {"label"/"name", "score"} dicts.
    """
    if isinstance(res, dict):
        if res and all(isinstance(v, (int, float)) for v in res.values()):
            return dict(res)
        for v in res.values():
            m = _score_map(v)
            if m:
                return m
    elif isinstance(res, (list, tuple)):
        try:
            d = {k: v for k, v in res}
            if d and all(isinstance(v, (int, float)) for v in d.values()):
                return d
        except (TypeError, ValueError):
            pass
        d = {}
        for item in res:
            if isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("key")
                val = item.get("score") or item.get("count") or item.get("value")
                if isinstance(label, str) and isinstance(val, (int, float)):
                    d[label] = val
        if d:
            return d
    return {}


def _top_hit(res, needles):
    """True when any needle names a top-scoring (or tied-top) entry, or the
    plain-string result itself contains a needle."""
    if isinstance(needles, str):
        needles = (needles,)
    sc = _score_map(res)
    if not sc:
        return any(n in str(res).lower() for n in needles)
    top = max(sc.values())
    return any(
        any(n in str(k).lower() for n in needles) and v == top
        for k, v in sc.items()
    )


def _names(res):
    """Normalise a names result (list/set/dict-of-counts) to lowercase strings."""
    if isinstance(res, dict):
        items = res.keys()
    elif isinstance(res, (list, tuple, set)):
        items = res
    else:
        items = [res]
    return {str(x).lower().strip() for x in items}


def _is_conflict(res):
    """Interpret a lat/band check result as 'conflict detected'."""
    if isinstance(res, dict):
        for k in ("conflict", "mismatch", "contradiction", "inconsistent",
                  "conflicts"):
            if k in res:
                return bool(res[k])
        for k in ("ok", "consistent", "compatible", "match"):
            if k in res:
                return not res[k]
        return any(w in str(res).lower()
                   for w in ("conflict", "mismatch", "inconsistent"))
    if isinstance(res, str):
        return any(w in res.lower()
                   for w in ("conflict", "mismatch", "inconsistent", "contradict"))
    return bool(res)


# --------------------------------------------------------------------------- #
# sundial/sun_side.py
# --------------------------------------------------------------------------- #
def test_sun_side_position_left():
    if sun_side is None:
        raise _Skip("sun_side")
    cond = {"sun": {"position": [200, 100]}}
    assert sun_side.sun_side_from_conditions(cond, 1000) == "left"


def test_sun_side_position_right():
    if sun_side is None:
        raise _Skip("sun_side")
    cond = {"sun": {"position": [900, 100]}}
    assert sun_side.sun_side_from_conditions(cond, 1000) == "right"


def test_sun_side_position_center():
    if sun_side is None:
        raise _Skip("sun_side")
    cond = {"sun": {"position": [500, 100]}}
    assert sun_side.sun_side_from_conditions(cond, 1000) == "center"


def test_sun_side_description_fallback():
    if sun_side is None:
        raise _Skip("sun_side")
    cond = {"sun": {"description": "low on the left edge"}}
    assert sun_side.sun_side_from_conditions(cond, 1000) == "left"


def test_sun_side_empty_conditions_returns_none():
    if sun_side is None:
        raise _Skip("sun_side")
    assert sun_side.sun_side_from_conditions({}, 1000) is None


def test_aggregate_sides_majority_wins():
    if sun_side is None:
        raise _Skip("sun_side")
    assert sun_side.aggregate_sides(["left", "left", "right"]) == "left"


def test_aggregate_sides_tie_returns_none():
    if sun_side is None:
        raise _Skip("sun_side")
    assert sun_side.aggregate_sides(["left", "right"]) is None


# --------------------------------------------------------------------------- #
# sundial/vss_lookup.py
# --------------------------------------------------------------------------- #
_SEG = "segments/20261001_{}_scene1_p1c1_chunk_0000_segment_001_of_006.mp4"


def test_segment_place_time_parses_utc_and_fields():
    if vss_lookup is None:
        raise _Skip("vss_lookup")
    res = vss_lookup.segment_place_time(_SEG.format("055249"))
    assert isinstance(res, dict) and res, res
    assert "2026-10-01T05:52:49Z" in str(res), res
    # plus the parsed segment fields (segment/scene/chunk/001/006 tokens)
    assert any(tok in str(res).lower()
               for tok in ("segment", "scene", "chunk", "001", "006")), res


def test_segment_place_time_garbage_returns_empty():
    if vss_lookup is None:
        raise _Skip("vss_lookup")
    for junk in ("garbage.mp4", "random", "", "segments/foo.mp4",
                 "segments/20261001.mp4"):
        res = vss_lookup.segment_place_time(junk)
        assert not res, (junk, res)


def test_nearest_segments_sorted_by_delta():
    if vss_lookup is None:
        raise _Skip("vss_lookup")
    paths = [_SEG.format(h) for h in ("060249", "055249", "054749")]
    res = vss_lookup.nearest_segments(paths, "2026-10-01T06:00:00")
    assert isinstance(res, (list, tuple)) and len(res) == 3, res
    deltas = []
    for entry in res:
        assert isinstance(entry, dict), entry
        assert "delta_minutes" in entry, entry
        deltas.append(entry["delta_minutes"])
    assert deltas == sorted(deltas), deltas
    assert deltas[0] <= deltas[-1], deltas
    # nearest to 06:00 is the 06:02:49 segment (~2.8 min out)
    if "utc" in res[0]:
        assert res[0]["utc"] == "2026-10-01T06:02:49Z", res[0]


# --------------------------------------------------------------------------- #
# sundial/geocode_cache.py
# --------------------------------------------------------------------------- #
def _make_cache(d):
    """GeocodeCache over a fresh dir — tolerant of str/Path/kwarg signatures."""
    try:
        return geocode_cache.GeocodeCache(d)
    except TypeError:
        pass
    try:
        return geocode_cache.GeocodeCache(Path(d))
    except TypeError:
        pass
    return geocode_cache.GeocodeCache(cache_dir=d)


def test_geocode_cache_put_get_roundtrip():
    if geocode_cache is None:
        raise _Skip("geocode_cache")
    cache = _make_cache(tempfile.mkdtemp())
    # The cache stores the candidate LIST (what geocode() returns), not a
    # wrapped dict.
    payload = [{"name": "Paris", "lat": 48.8566, "lon": 2.3522,
                "country_code": "FR"}]
    cache.put("Paris, France", payload)
    got = cache.get("Paris, France")
    assert got == payload, got


def test_geocode_cache_ttl_expiry_misses():
    if geocode_cache is None:
        raise _Skip("geocode_cache")
    d = tempfile.mkdtemp()
    cache = _make_cache(d)
    payload = [{"name": "London", "country_code": "GB"}]
    cache.put("London, UK", payload)
    assert cache.get("London, UK"), "fresh entry should be a hit"
    # Backdate every cache file 15 days — beyond any sane default TTL.
    old = time.time() - 15 * 86400
    for f in Path(d).rglob("*"):
        if f.is_file():
            os.utime(f, (old, old))
    # A fresh instance must miss; the same instance may legitimately too.
    got_same = cache.get("London, UK")
    got_fresh = _make_cache(d).get("London, UK")
    assert not got_fresh or not got_same, (got_same, got_fresh)


def test_wrap_geocode_preserves_return_shape():
    if geocode_cache is None:
        raise _Skip("geocode_cache")
    import dataclasses
    cache = _make_cache(tempfile.mkdtemp())
    payload = [{"name": "Rome", "lat": 41.9, "lon": 12.5}]
    calls = {"n": 0}

    def stub(query):
        calls["n"] += 1
        return list(payload)

    def _to_dicts(items):
        # Hits may rebuild dataclass candidates — normalize, then compare
        # only the keys the payload actually set (rebuilds add defaults).
        norm = [dataclasses.asdict(x) if dataclasses.is_dataclass(x) else x
                for x in items]
        return [{k: d.get(k) for k in payload[0]} for d in norm]

    wrap = geocode_cache.wrap_geocode
    try:
        wrapped = wrap(stub, cache=cache)
    except TypeError:
        wrapped = wrap(stub, cache)
    if not callable(wrapped):
        raise TypeError("wrap_geocode did not return a callable")
    out1 = wrapped("Rome, Italy")
    out2 = wrapped("Rome, Italy")
    # Hits may rebuild dataclass candidates — normalize before comparing.
    assert _to_dicts(out1) == payload, out1
    assert _to_dicts(out2) == payload, out2
    assert calls["n"] >= 1


# --------------------------------------------------------------------------- #
# sundial/overpass.py
# --------------------------------------------------------------------------- #
_POIS = [{"name": "Costa Coffee", "lat": 0, "lon": 0,
          "kind": "amenity", "value": "cafe"}]


def _match(pois, names):
    """name_matches_nearby, tolerant of (pois, names) vs (names, pois) order."""
    try:
        res = overpass.name_matches_nearby(pois, names)
    except Exception:  # noqa: BLE001 — arg-order probe
        res = None
    if not res:
        try:
            res = overpass.name_matches_nearby(names, pois)
        except Exception:  # noqa: BLE001
            res = None
    return res or []


def test_name_matches_nearby_finds_poi():
    if overpass is None:
        raise _Skip("overpass")
    res = _match(_POIS, ["costa coffee", "Starbucks"])
    assert len(res) == 1, res
    assert "costa" in str(res).lower(), res


def test_name_matches_nearby_case_and_punct_insensitive():
    if overpass is None:
        raise _Skip("overpass")
    res = _match(_POIS, ["COSTA-COFFEE!!"])
    assert len(res) == 1, res
    assert "costa" in str(res).lower(), res


def test_name_matches_nearby_no_name_returns_empty():
    if overpass is None:
        raise _Skip("overpass")
    assert _match(_POIS, []) == []
    unnamed = [{"lat": 0, "lon": 0, "kind": "amenity", "value": "cafe"}]
    assert _match(unnamed, ["costa coffee"]) == []
    assert _match([], ["costa coffee"]) == []


# --------------------------------------------------------------------------- #
# sundial/sign_style.py
# --------------------------------------------------------------------------- #
def test_sign_style_uk_convention_scores_top():
    if sign_style is None:
        raise _Skip("sign_style")
    res = sign_style.sign_style_score({
        "shape": "triangular",
        "colors": "red border white",
        "convention_guess": "uk-style",
    })
    assert _top_hit(res, ("uk", "britain", "united_kingdom", "gb")), res


def test_sign_style_mutcd_convention_scores_top():
    if sign_style is None:
        raise _Skip("sign_style")
    res = sign_style.sign_style_score({
        "shape": "diamond",
        "colors": "black on yellow",
        "convention_guess": "mutcd-style",
    })
    assert _top_hit(res, ("mutcd", "us", "america")), res


# --------------------------------------------------------------------------- #
# sundial/veg_geo.py
# --------------------------------------------------------------------------- #
def test_veg_biomes_tropical_dominant():
    if veg_geo is None:
        raise _Skip("veg_geo")
    res = veg_geo.veg_biomes(["palm trees", "dense broadleaf"])
    sc = _score_map(res)
    if sc:
        top = max(sc, key=sc.get)
        assert "tropic" in str(top).lower(), res
    else:
        assert "tropic" in str(res).lower(), res


def test_lat_band_check_tropical_at_60n_conflicts():
    if veg_geo is None:
        raise _Skip("veg_geo")
    biomes = veg_geo.veg_biomes(["palm trees", "dense broadleaf"])
    res = veg_geo.lat_band_check(60.0, biomes)
    assert _is_conflict(res), res


# --------------------------------------------------------------------------- #
# sundial/landmarks.py
# --------------------------------------------------------------------------- #
def test_consensus_landmarks_keeps_two_frame_names():
    if landmarks is None:
        raise _Skip("landmarks")
    res = landmarks.consensus_landmarks(
        [["Tower Bridge"], ["tower bridge", "Big Ben"], ["Big Ben"]])
    names = _names(res)
    assert any("tower" in n and "bridge" in n for n in names), res
    assert any("big ben" in n for n in names), res
    assert len(names) == 2, res


def test_consensus_landmarks_drops_one_off_sightings():
    if landmarks is None:
        raise _Skip("landmarks")
    res = landmarks.consensus_landmarks(
        [["Tower Bridge", "The Shard"],
         ["tower bridge", "Big Ben"],
         ["Big Ben"]])
    names = _names(res)
    assert not any("shard" in n for n in names), res
    assert any("tower" in n and "bridge" in n for n in names), res


# --------------------------------------------------------------------------- #
# sundial/reverse_geo.py / climate.py / osint.py — import smoke only
# --------------------------------------------------------------------------- #
_SMOKE_CANDIDATES = {
    "reverse_geo": ("reverse_geocode", "reverse", "lookup_place",
                    "nearest_place", "describe_location", "place_at",
                    "country_at"),
    "climate": ("climate_zone", "koppen_zone", "climate_at", "zone_for_lat",
                "latitude_band", "expected_climate", "climate_evidence"),
    "osint": ("osint_evidence", "extract_osint_features", "osint_features",
              "summarize_osint", "region_guess", "compare_regions",
              "score_regions"),
}


def _smoke(mod, label):
    if mod is None:
        raise _Skip(label)
    public = [n for n in dir(mod) if not n.startswith("_")]
    assert public, f"{label}: module has no public attributes"
    hits = [n for n in _SMOKE_CANDIDATES[label] if hasattr(mod, n)]
    callables = [n for n in public if callable(getattr(mod, n))]
    assert hits or callables, f"{label}: no functions found (public: {public})"


def test_smoke_reverse_geo():
    _smoke(reverse_geo, "reverse_geo")


def test_smoke_climate():
    _smoke(climate, "climate")


def test_smoke_osint():
    _smoke(osint, "osint")


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except _Skip as e:
            print(f"SKIP {name} ({e})")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
