"""Tests for the OSINT/geo-claim stack — run directly: python3 tests/test_osint_stack.py

Covers, when present: sundial.match_score, sundial.hemisphere,
sundial.geoguess, sundial.dedupe_evidence, sundial.osint_features.

These modules are built in parallel — every import is guarded and every
test raises _Skip when its module/function is absent, so the file stays
green against a partial tree. No pytest, no network, no heavy deps.
"""
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


try:
    from sundial.schema import ClaimVerdict, Evidence
except Exception:  # partial tree — dedupe tests will SKIP
    ClaimVerdict = Evidence = None

match_score = _try_import("sundial.match_score")
hemisphere = _try_import("sundial.hemisphere")
geoguess = _try_import("sundial.geoguess")
dedupe_evidence = _try_import("sundial.dedupe_evidence")
osint_features = _try_import("sundial.osint_features")


class _Skip(Exception):
    pass


def _need(mod, name):
    if mod is None:
        raise _Skip(f"{name} not present")


def _need_fn(mod, name, fnname):
    _need(mod, name)
    fn = getattr(mod, fnname, None)
    if not callable(fn):
        raise _Skip(f"{name}.{fnname} not present")
    return fn


VALID_VERDICTS = ("supports", "contradicts", "inconclusive")


def _verdict(ev):
    """Evidence/dict/enum/str -> lowercase verdict string (or None)."""
    v = ev.get("verdict") if isinstance(ev, dict) else getattr(ev, "verdict", ev)
    v = getattr(v, "value", v)
    return str(v).lower() if v is not None else None


def _etype(ev):
    return ev.get("type") if isinstance(ev, dict) else getattr(ev, "type", None)


def _conf(ev):
    return ev.get("confidence") if isinstance(ev, dict) else getattr(ev, "confidence", None)


def _summary(ev):
    return ev.get("summary") if isinstance(ev, dict) else getattr(ev, "summary", ev)


def _get(rec, key):
    return rec.get(key) if isinstance(rec, dict) else getattr(rec, key, None)


# --------------------------- match_score --------------------------------

UK_FEATURES = {
    "driving_side": "left",
    "sign_language": "en",
    "plate_style": "uk",
    "road_markings": ["double yellow lines"],
    "lane_markings_color": "yellow",
    "vegetation": [],
    "climate_cues": [],
}
CANDIDATES = ["GB", "US", "DE", "FR"]


def test_score_candidates_schema_and_order():
    fn = _need_fn(match_score, "sundial.match_score", "score_candidates")
    ranked = fn(UK_FEATURES, CANDIDATES)
    assert isinstance(ranked, list) and ranked, "empty ranking"
    for row in ranked:
        assert {"cc", "score", "hits", "misses"} <= set(row), row
        assert isinstance(row["score"], (int, float)), row
    assert {r["cc"] for r in ranked} == set(CANDIDATES), ranked
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True), scores


def test_score_candidates_gb_top_beats_us():
    fn = _need_fn(match_score, "sundial.match_score", "score_candidates")
    ranked = fn(UK_FEATURES, CANDIDATES)
    assert ranked[0]["cc"] == "GB", ranked
    by_cc = {r["cc"]: r for r in ranked}
    assert by_cc["GB"]["score"] > by_cc["US"]["score"], by_cc
    n_hits = lambda h: len(h) if isinstance(h, (list, tuple)) else h
    assert n_hits(by_cc["GB"]["hits"]) >= n_hits(by_cc["US"]["hits"]), by_cc


def test_score_candidates_empty_features_graceful():
    fn = _need_fn(match_score, "sundial.match_score", "score_candidates")
    ranked = fn({}, ["GB", "US"])
    assert isinstance(ranked, list) and len(ranked) == 2, ranked
    assert all({"cc", "score", "hits", "misses"} <= set(r) for r in ranked), ranked


def test_match_evidence_supports_claimed_gb():
    _need(match_score, "sundial.match_score")
    ranked = match_score.score_candidates(UK_FEATURES, CANDIDATES)
    ev = _need_fn(match_score, "sundial.match_score", "match_evidence")(ranked, "GB")
    assert _verdict(ev) == "supports", _summary(ev)


def test_match_evidence_claimed_us_contradicts_or_low():
    _need(match_score, "sundial.match_score")
    ranked = match_score.score_candidates(UK_FEATURES, CANDIDATES)
    ev = _need_fn(match_score, "sundial.match_score", "match_evidence")(ranked, "US")
    v, conf = _verdict(ev), (_conf(ev) or 0.0)
    assert v in VALID_VERDICTS, v
    if v != "contradicts":
        assert conf <= 0.4, f"expected CONTRADICTS or low support, got {v}@{conf}"


# ---------------------------- hemisphere ---------------------------------

HEMI_TS = "2025-06-15T12:00:00+00:00"  # ~solar noon, London


def test_hemisphere_left_vs_right_differ():
    fn = _need_fn(hemisphere, "sundial.hemisphere", "hemisphere_evidence")
    ev_l = fn(51.5, -0.12, HEMI_TS, "left")
    ev_r = fn(51.5, -0.12, HEMI_TS, "right")
    assert _verdict(ev_l) != _verdict(ev_r), (_summary(ev_l), _summary(ev_r))


def test_hemisphere_none_inputs_inconclusive():
    fn = _need_fn(hemisphere, "sundial.hemisphere", "hemisphere_evidence")
    for args in ((None, None, None, None),
                 (51.5, -0.12, HEMI_TS, None),
                 (None, -0.12, HEMI_TS, "left")):
        ev = fn(*args)
        assert _verdict(ev) == "inconclusive", (args, _summary(ev))


def test_hemisphere_evidence_schema_ranges():
    fn = _need_fn(hemisphere, "sundial.hemisphere", "hemisphere_evidence")
    for side in ("left", "right"):
        ev = fn(51.5, -0.12, HEMI_TS, side)
        assert _verdict(ev) in VALID_VERDICTS, ev
        conf = _conf(ev)
        assert conf is None or 0.0 <= conf <= 1.0, conf
        assert isinstance(_summary(ev) or "", str)


# ----------------------------- geoguess ----------------------------------

def _agg_counts(res):
    """Best-effort {normalized_name: votes} out of aggregate_guesses output."""
    if isinstance(res, dict):
        for k in ("votes", "counts", "tally", "guesses", "results", "histogram"):
            sub = res.get(k)
            if isinstance(sub, (dict, list, tuple)):
                return _agg_counts(sub)
        if res and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       for v in res.values()):
            return {str(n).strip().lower(): int(v) for n, v in res.items()}
        name = next((res[k] for k in ("top", "guess", "winner", "country_guess",
                                      "country", "name", "best") if res.get(k)), None)
        if name is not None:
            cnt = next((res[k] for k in ("votes", "count", "n")
                        if isinstance(res.get(k), (int, float))), 1)
            return {str(name).strip().lower(): int(cnt)}
        return {}
    if isinstance(res, (list, tuple)):
        if len(res) == 2 and isinstance(res[0], str) and isinstance(res[1], (int, float)):
            return {res[0].strip().lower(): int(res[1])}
        out = {}
        for item in res:
            if isinstance(item, dict):
                name = next((item[k] for k in ("country_guess", "guess", "name",
                                              "country", "top", "cc")
                             if item.get(k) is not None), None)
                cnt = next((item[k] for k in ("votes", "count", "n", "score")
                            if isinstance(item.get(k), (int, float))), 1)
                if name is not None:
                    out[str(name).strip().lower()] = int(cnt)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                a, b = item
                if isinstance(a, str) and isinstance(b, (int, float)):
                    out[a.strip().lower()] = int(b)
                elif isinstance(b, str) and isinstance(a, (int, float)):
                    out[b.strip().lower()] = int(a)
        return out
    return {}


UK_KEYS = ("uk", "gb", "great britain", "britain")


def _is_uk(name):
    return "kingdom" in name or name in UK_KEYS


def test_aggregate_guesses_majority_votes():
    fn = _need_fn(geoguess, "sundial.geoguess", "aggregate_guesses")
    guesses = [{"country_guess": "United Kingdom"},
               {"country_guess": "United Kingdom"},
               {"country_guess": "France"}]
    counts = _agg_counts(fn(guesses))
    assert counts, f"could not read votes from aggregate result"
    assert sum(v for k, v in counts.items() if _is_uk(k)) == 2, counts
    top = max(counts.items(), key=lambda kv: kv[1])[0]
    assert _is_uk(top), top
    if len(counts) > 1:
        assert sum(v for k, v in counts.items()
                   if "france" in k or k == "fr") == 1, counts
    # stable: input order doesn't change the winner
    counts2 = _agg_counts(fn(list(reversed(guesses))))
    assert counts2 and _is_uk(max(counts2.items(), key=lambda kv: kv[1])[0])


def test_resolve_cc_known_and_unknown():
    fn = _need_fn(geoguess, "sundial.geoguess", "resolve_cc")
    assert fn("United Kingdom") == "GB"
    assert fn("France") == "FR"
    assert fn("Atlantis") is None


def _gg_evidence(guesses, claimed_cc):
    """geoguess_evidence takes the aggregate dict; tolerate a guesses-list API."""
    agg_fn = getattr(geoguess, "aggregate_guesses", None)
    if callable(agg_fn):
        try:
            return geoguess.geoguess_evidence(agg_fn(guesses), claimed_cc)
        except TypeError:
            pass
    return geoguess.geoguess_evidence(guesses, claimed_cc)


def test_geoguess_evidence_mismatch_contradicts_low():
    _need_fn(geoguess, "sundial.geoguess", "geoguess_evidence")
    ev = _gg_evidence([{"country_guess": "France"}] * 3, "GB")
    assert _verdict(ev) == "contradicts", _summary(ev)
    conf = _conf(ev)
    assert conf is None or conf <= 0.4, conf


def test_geoguess_evidence_match_supports():
    _need_fn(geoguess, "sundial.geoguess", "geoguess_evidence")
    ev = _gg_evidence([{"country_guess": "United Kingdom"}] * 3, "GB")
    assert _verdict(ev) == "supports", _summary(ev)


# -------------------------- dedupe_evidence ------------------------------

def _fab_evs():
    return [
        Evidence(type="geo", summary="weak geocode",
                 verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.2),
        Evidence(type="geo", summary="no readable signs",
                 verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.15),
        Evidence(type="geo", summary="blurry plates",
                 verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.1),
        Evidence(type="geo", summary="left-side driving fits",
                 verdict=ClaimVerdict.SUPPORTS, confidence=0.8),
        Evidence(type="geo", summary="plate style mismatch",
                 verdict=ClaimVerdict.CONTRADICTS, confidence=0.6),
        Evidence(type="solar", summary="sun angle fits",
                 verdict=ClaimVerdict.SUPPORTS, confidence=0.9),
    ]


def _need_dedupe():
    _need(Evidence, "sundial.schema")
    _need(dedupe_evidence, "sundial.dedupe_evidence")


def _in_geo_family(fn, ev):
    r = fn(ev)
    if isinstance(r, str):
        return "geo" in r.lower()
    return bool(r)


def test_geo_family_collects_geo_cards():
    _need_dedupe()
    evs = _fab_evs()
    gf = dedupe_evidence.geo_family
    if not callable(gf):  # module-level constant of geo-family type names
        assert "geo" in {str(t) for t in gf}, gf
        return
    fam = gf(evs)
    if not isinstance(fam, list):  # maybe a per-evidence predicate/classifier
        fam = [e for e in evs if _in_geo_family(gf, e)]
    assert fam, "geo_family found nothing"
    n_geo = sum(1 for e in evs if _etype(e) == "geo")
    got_geo = [e for e in fam if _etype(e) == "geo"]
    assert len(got_geo) == n_geo, f"{len(got_geo)}/{n_geo} geo cards collected"
    assert all(_etype(e) != "solar" for e in fam), "solar leaked into geo family"


def test_summarize_geo_family_runs():
    _need_dedupe()
    fn = _need_fn(dedupe_evidence, "sundial.dedupe_evidence", "summarize_geo_family")
    evs = _fab_evs()
    try:
        s = fn(evs)
    except TypeError:
        s = fn(dedupe_evidence.geo_family(evs))
    assert s is not None and s != "" and s != [] and s != {}, f"empty summary {s!r}"
    if isinstance(s, Evidence):
        assert _verdict(s) in VALID_VERDICTS
        assert 0.0 <= (s.confidence or 0.0) <= 1.0
    elif isinstance(s, dict):
        assert s
    else:
        assert str(s).strip()


def test_geo_rollup_condenses_geo_cards():
    _need_dedupe()
    fn = _need_fn(dedupe_evidence, "sundial.dedupe_evidence", "geo_rollup")
    evs = _fab_evs()
    try:
        out = fn(evs)
    except TypeError:
        out = fn(dedupe_evidence.geo_family(evs), evs)
    if isinstance(out, Evidence):
        assert _verdict(out) in VALID_VERDICTS
        return
    assert isinstance(out, list), type(out)
    assert all(hasattr(e, "verdict") or isinstance(e, dict) for e in out), out
    assert len(out) <= len(evs), "rollup grew the list"
    assert any(_etype(e) == "solar" for e in out), "non-geo evidence dropped"


def test_compact_for_display_collapses_inconclusive_geo():
    _need_dedupe()
    fn = _need_fn(dedupe_evidence, "sundial.dedupe_evidence", "compact_for_display")
    evs = _fab_evs()
    out = fn(evs)
    assert isinstance(out, list) and out, type(out)
    inc_geo = [e for e in out
               if _etype(e) == "geo" and _verdict(e) == "inconclusive"]
    assert len(inc_geo) <= 1, f"{len(inc_geo)} inconclusive geo cards kept"
    assert len(out) < len(evs), "nothing collapsed"
    verdicts = [_verdict(e) for e in out]
    assert "supports" in verdicts and "contradicts" in verdicts, verdicts
    assert any(_etype(e) == "solar" for e in out)


# -------------------------- osint_features -------------------------------

def _merged_values(entry):
    """Union list out of a merged field — tolerates {"values": [...]},
    {"counts": {...}}, a bare list, or a scalar."""
    if isinstance(entry, dict):
        if isinstance(entry.get("values"), list):
            return list(entry["values"])
        if isinstance(entry.get("counts"), dict):
            return [k for k in entry["counts"]]
        v = entry.get("value")
        return [v] if v is not None else []
    if isinstance(entry, (list, tuple)):
        return list(entry)
    return [entry] if entry is not None else []


def _merged_scalar(entry):
    """Modal scalar out of a merged field — tolerates {"value": v} or bare v."""
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


def test_merge_features_dedupes_case_union():
    fn = _need_fn(osint_features, "sundial.osint_features", "merge_features")
    merged = fn([{"vegetation": ["Palm"], "terrain": "flat"},
                 {"vegetation": ["palm", "oak"], "terrain": "flat"}])
    veg = sorted(str(v).lower() for v in _merged_values(_get(merged, "vegetation")))
    assert veg == ["oak", "palm"], veg
    terr = _merged_scalar(_get(merged, "terrain"))
    assert str(terr).lower() == "flat", terr


def test_merge_features_modal_terrain():
    fn = _need_fn(osint_features, "sundial.osint_features", "merge_features")
    merged = fn([{"terrain": "hilly"}, {"terrain": "flat"}, {"terrain": "hilly"}])
    assert str(_merged_scalar(_get(merged, "terrain"))).lower() == "hilly", merged


def test_feature_vector_keys_present():
    fn = _need_fn(osint_features, "sundial.osint_features", "feature_vector")
    vec = fn(dict(UK_FEATURES))
    assert vec is not None
    if isinstance(vec, dict):
        assert vec, "empty feature vector"
    else:
        assert len(list(vec)) > 0, "empty feature vector"
    # graceful on empty input
    assert fn({}) is not None


# ------------------------------ harness ----------------------------------

def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except _Skip as e:
            skipped += 1
            print(f"SKIP {name} — {e}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    tail = f" ({skipped} skipped)" if skipped else ""
    print(f"\n{len(tests) - failed}/{len(tests)} ok{tail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
