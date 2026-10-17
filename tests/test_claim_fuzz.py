"""Fuzz tests for the claim PARSING/resolution layer — offline only.

    pytest tests/test_claim_fuzz.py        (preferred)
    python3 tests/test_claim_fuzz.py       (direct-run fallback, no pytest needed)

Surfaces under test (no VLM, no network):
  * sundial.claim_parser.parse_claim_text / claim_to_args
  * sundial.pipeline._resolve_coords     (geo.geocode is monkeypatched —
    Nominatim is never touched; the fake mirrors its blank-query -> [] guard)
  * sundial.pipeline._claimed_utc        (weather archive passed explicitly or
    weather.historical_weather monkeypatched — Open-Meteo is never touched)

Every fuzz input must either parse sensibly or fail gracefully: no exception
propagates, no call takes > _BUDGET_S seconds. Outputs that are present must
be well-formed (lat in [-90,90], lon in [-180,180], parseable ISO stamps).

Tests named test_xfail_* document REPRODUCED wrong-value bugs: they assert
the *correct* behaviour and are marked xfail(strict) under pytest, so the
suite stays green while a bug exists and flags XPASS once it is fixed. The
direct-run fallback prints them as KNOWN-BUG. All reproduced anomalies are
also collected in FINDINGS and printed by the __main__ runner.
"""
from __future__ import annotations

import re
import sys
import time
import traceback
from datetime import date as _date, datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sundial import claim_parser, geo, pipeline, weather  # noqa: E402
from sundial.schema import GeoCandidate  # noqa: E402

try:  # xfail marks only — the module must import without pytest installed
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


_BUDGET_S = 5.0            # per-call hang budget
_FIXED_NOW = datetime(2025, 10, 15, 18, 0)   # Wednesday — deterministic weekday math
_HUSUM = (54.4858, 9.0564)

# ---------------------------------------------------------------------------
# fuzz matrix
# ---------------------------------------------------------------------------
PLACES = [
    "London", "husum germany", "54.47,9.05", "91,0", "200,999",
    "Husum, Germany 🇩🇪", "東京", "Zürich, Switzerland", "São Paulo",
    "london,,,", "  ", "", None, "a" * 500, "';DROP TABLE--", "<script>",
    "London France",
]
TIMES = [
    "2025-06-15T15:00", "2025-6-15", "15:30", "yesterday 3pm", "tuesday 3pm",
    "TUE 3PM", "3pm", "15pm", "25:00", "2025-13-45", "2025-02-30",
    "2025-06-15T25:00", "in two weeks", "next friday", "", None,
    "2025/06/15", "15-06-2025", "Jun 15 3pm", "15.06.2025",
]
# derived boundary cases (same categories, extra edges of the same regexes)
EXTRA_PLACES = ["90,180", "-90,-180", "90.0001,0", "-91,0", "0,-181",
                "nan,0", "inf,0", "1e2,0", "54.47,9.05,extra", " 54.47 , 9.05 ",
                "filmed at 54.47,9.05", "\x00null\x00", "🇩🇪🇩🇪🇩🇪", "\n\t "]
EXTRA_TIMES = ["12am", "12pm", "0pm", "13pm", "24:00", "23:59", "0:00",
               "2025-06-15 15:00", "2025-06-15T15:00Z",
               "2025-06-15T15:00+02:00", "2025-06-15", "at 22", "around 3pm",
               "half past 3", "3 o'clock", "Feb 29 2025", "Oct 40th 2025"]
NON_STRINGS = [123, 4.5, True, ["London"], {"x": 1}, b"London", (1, 2)]

ALL_TEXTS = PLACES + TIMES + EXTRA_PLACES + EXTRA_TIMES

# ---------------------------------------------------------------------------
# deterministic offline stand-in for geo.geocode (Nominatim)
# ---------------------------------------------------------------------------
_KNOWN = {
    "london": GeoCandidate("London", 51.5074, -0.1278, "GB",
                           "London, England, GB", 0.9),
    "husum": GeoCandidate("Husum", 54.4858, 9.0564, "DE",
                          "Husum, Schleswig-Holstein, DE", 0.8),
    "zürich": GeoCandidate("Zürich", 47.3769, 8.5417, "CH", "Zürich, CH", 0.8),
    "são paulo": GeoCandidate("São Paulo", -23.55, -46.63, "BR",
                              "São Paulo, BR", 0.8),
    "東京": GeoCandidate("Tokyo", 35.6762, 139.6503, "JP", "東京, JP", 0.8),
    "paris": GeoCandidate("Paris", 48.8566, 2.3522, "FR", "Paris, FR", 0.8),
}


def _fake_geocode(query):
    """Mirror geo.geocode's contract: blank query -> []; else keyword hit or []."""
    try:
        q = str(query).strip().lower() if not isinstance(query, str) \
            else query.strip().lower()
    except Exception:
        return []
    if not q:
        return []
    for key, cand in _KNOWN.items():
        if key in q:
            return [cand]
    return []


GEOCODE_CALLS: list = []


def _recording_geocode(query):
    GEOCODE_CALLS.append(query)
    return _fake_geocode(query)


def _strict_geocode_gate(query):
    """Replicates geo.geocode's PRE-NETWORK behaviour only (geo.py:103-105):
    `(query or "").strip()` raises AttributeError for non-str/bytes exactly as
    the real function does — that crash precedes any request. Bytes pass
    `.strip()`; the real code then sends them to requests (verified live:
    b"54.47,9.05" geocoded fine), so we decode and emulate a normal lookup."""
    q = (query or "").strip()           # AttributeError for int/list/dict/...
    if not q:
        return []
    if isinstance(q, bytes):
        q = q.decode("utf-8", "replace")
    return _fake_geocode(str(q))


# ---------------------------------------------------------------------------
# findings + helpers
# ---------------------------------------------------------------------------
FINDINGS: list[str] = []


def _finding(sev: str, msg: str) -> None:
    entry = f"[{sev}] {msg}"
    if entry not in FINDINGS:
        FINDINGS.append(entry)


def _timed(fn, *args, **kwargs):
    """-> (elapsed_s, result_or_None, exc_repr_or_None). Never raises."""
    t0 = time.monotonic()
    try:
        return time.monotonic() - t0, fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001 — recording is the point
        return time.monotonic() - t0, None, f"{type(e).__name__}: {e}"


def _assert_result_shape(r: dict) -> None:
    """Structural contract of parse_claim_text's dict."""
    for key in ("place", "place_raw", "time_iso", "day_offset",
                "time_of_day", "confidence", "raw"):
        assert key in r, f"missing key {key!r}"
    assert r["place"] is None or isinstance(r["place"], str)
    assert r["place_raw"] is None or isinstance(r["place_raw"], str)
    assert r["time_iso"] is None or isinstance(r["time_iso"], str)
    assert r["time_of_day"] is None or isinstance(r["time_of_day"], str)
    assert r["day_offset"] is None or isinstance(r["day_offset"], int)
    assert isinstance(r["confidence"], (int, float))
    assert 0.0 <= r["confidence"] <= 1.0, f"confidence out of range: {r['confidence']}"
    if r["time_iso"] is not None:
        dt = datetime.fromisoformat(r["time_iso"])  # must stay resolver-safe
        assert dt.tzinfo is None, "time_iso must be naive local"
    if r["time_of_day"] is not None:
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", r["time_of_day"])
        assert m, f"malformed time_of_day {r['time_of_day']!r}"
        assert int(m.group(1)) <= 23 and int(m.group(2)) <= 59


def _coords_ok(lat, lon) -> None:
    if lat is not None:
        assert -90.0 <= lat <= 90.0, f"lat out of range: {lat}"
    if lon is not None:
        assert -180.0 <= lon <= 180.0, f"lon out of range: {lon}"


def _xfail(fn):
    if pytest is not None:
        return pytest.mark.xfail(
            reason="reproduced wrong-value bug — see FINDINGS/report", strict=True)(fn)
    return fn


_XFAIL_NAMES = set()          # __main__ runner treats these as known-bug probes


def _known_bug(fn):
    _XFAIL_NAMES.add(fn.__name__)
    return _xfail(fn)


# ===========================================================================
# claim_parser — crash/hang invariants
# ===========================================================================

def test_parse_claim_text_never_crashes_on_places():
    for s in PLACES:
        for now in (None, _FIXED_NOW):
            dt, r, err = _timed(claim_parser.parse_claim_text, s, now)
            assert err is None, f"parse_claim_text({s!r}) raised {err}"
            assert dt < _BUDGET_S, f"parse_claim_text({s!r}) took {dt:.2f}s"
            _assert_result_shape(r)


def test_parse_claim_text_never_crashes_on_times():
    for s in TIMES:
        for now in (None, _FIXED_NOW):
            dt, r, err = _timed(claim_parser.parse_claim_text, s, now)
            assert err is None, f"parse_claim_text({s!r}) raised {err}"
            assert dt < _BUDGET_S, f"parse_claim_text({s!r}) took {dt:.2f}s"
            _assert_result_shape(r)


def test_parse_claim_text_never_crashes_on_extras():
    for s in EXTRA_PLACES + EXTRA_TIMES:
        dt, r, err = _timed(claim_parser.parse_claim_text, s, _FIXED_NOW)
        assert err is None, f"parse_claim_text({s!r}) raised {err}"
        assert dt < _BUDGET_S
        _assert_result_shape(r)


def test_claim_to_args_never_crashes_and_returns_pair():
    for s in ALL_TEXTS:
        dt, r, err = _timed(claim_parser.claim_to_args, s)
        assert err is None, f"claim_to_args({s!r}) raised {err}"
        assert dt < _BUDGET_S
        assert isinstance(r, tuple) and len(r) == 2
        t_iso, place = r
        assert place is None or isinstance(place, str)
        if t_iso is not None:
            datetime.fromisoformat(t_iso)   # pipeline._claimed_utc must accept it


def test_parse_empty_and_blank_degrade_to_empty_result():
    for s in ("", "  ", None, "\n\t "):
        r = claim_parser.parse_claim_text(s, _FIXED_NOW)
        assert r["place"] is None and r["time_iso"] is None
        assert r["day_offset"] is None and r["confidence"] == 0.0
        assert r["raw"] is s


def test_parse_invalid_times_produce_no_time_value():
    """Broken clocks/dates must not silently become a time_of_day."""
    invalid = ["15pm", "0pm", "13pm", "25:00", "24:00",
               "2025-13-45", "2025-02-30", "2025-06-15T25:00",
               "Feb 29 2025", "Oct 40th 2025"]
    for s in invalid:
        r = claim_parser.parse_claim_text(s, _FIXED_NOW)
        assert r["time_of_day"] is None, f"{s!r} yielded time {r['time_of_day']!r}"
        assert r["time_iso"] is None


def test_parse_unsupported_date_formats_degrade_quietly():
    """Slash/dotted/DMY-numeric dates are unsupported — must yield nothing."""
    for s in ("2025/06/15", "15-06-2025", "15.06.2025"):
        r = claim_parser.parse_claim_text(s, _FIXED_NOW)
        assert r["time_iso"] is None and r["day_offset"] is None
        _finding("INFO", f"date format {s!r} unsupported — parse returns all-None")


def test_parse_known_good_claims_still_work():
    """Sanity anchors so the fuzz can't mask a fully broken parser."""
    r = claim_parser.parse_claim_text("filmed Tuesday 3pm in London", _FIXED_NOW)
    assert r["place"] == "London" and r["time_iso"] == "2025-10-14T15:00"
    r = claim_parser.parse_claim_text("2025-06-15 15:00", _FIXED_NOW)
    assert r["time_iso"] == "2025-06-15T15:00" and r["place"] is None
    r = claim_parser.parse_claim_text("Jun 15 3pm", _FIXED_NOW)
    assert r["time_iso"] == "2025-06-15T15:00"


# ===========================================================================
# pipeline._resolve_coords — crash/range invariants (geocode offline)
# ===========================================================================

def test_resolve_coords_never_crashes_on_places():
    with mock.patch.object(geo, "geocode", side_effect=_recording_geocode):
        for s in PLACES + EXTRA_PLACES:
            dt, out, err = _timed(pipeline._resolve_coords, s)
            assert err is None, f"_resolve_coords({s!r}) raised {err}"
            assert dt < _BUDGET_S
            lat, lon, note = out
            assert isinstance(note, str) and note
            _coords_ok(lat, lon)


def test_resolve_coords_literal_coordinates():
    with mock.patch.object(geo, "geocode", side_effect=_recording_geocode) as mg:
        lat, lon, note = pipeline._resolve_coords("54.47,9.05")
        mg.assert_not_called()                       # coords must skip geocoder
        assert (lat, lon) == (54.47, 9.05)
        assert "coordinates given in claim" in note
        for ok in ("90,180", "-90,-180", " 54.47 , 9.05 "):
            lat, lon, _ = pipeline._resolve_coords(ok)
            assert lat is not None and lon is not None, ok
            _coords_ok(lat, lon)
        mg.assert_not_called()


def test_resolve_coords_invalid_literal_coords_are_not_returned():
    """Out-of-range coord literals must not come back as coords."""
    with mock.patch.object(geo, "geocode", return_value=[]):
        for bad in ("91,0", "200,999", "90.0001,0", "-91,0", "0,-181",
                    "nan,0", "inf,0", "1e2,0"):
            lat, lon, note = pipeline._resolve_coords(bad)
            assert lat is None and lon is None, (bad, lat, lon)
            assert "could not geocode" in note
    _finding("INFO", "out-of-range coord literals fall through to the geocoder "
                     "(e.g. '91,0') — wasted Nominatim query, but no wrong coords")


def test_resolve_coords_empty_inputs():
    with mock.patch.object(geo, "geocode", side_effect=_recording_geocode):
        for s in (None, "", "  "):
            lat, lon, note = pipeline._resolve_coords(s)
            assert lat is None and lon is None
            assert isinstance(note, str) and note


def test_resolve_coords_unicode_and_injection_strings():
    """Weird strings reach the geocoder unchanged — must still not raise."""
    with mock.patch.object(geo, "geocode", side_effect=_fake_geocode):
        lat, lon, _ = pipeline._resolve_coords("Husum, Germany 🇩🇪")
        assert (lat, lon) == _HUSUM                  # emoji survives, still resolves
        lat, lon, _ = pipeline._resolve_coords("東京")
        assert (lat, lon) == (35.6762, 139.6503)     # non-ASCII resolves via geocoder
        for s in ("';DROP TABLE--", "<script>", "a" * 500):
            lat, lon, note = pipeline._resolve_coords(s)
            assert lat is None and lon is None
    _finding("LOW", "emoji / injection-shaped strings are passed to the "
                    "geocoder verbatim (param-encoded, but unfiltered)")


# ===========================================================================
# pipeline._claimed_utc — ISO contract + fallbacks
# ===========================================================================

def _iso_accepts(s) -> bool:
    try:
        datetime.fromisoformat(str(s))
        return True
    except (ValueError, TypeError):
        return False


def test_claimed_utc_valid_iso_never_crashes():
    """Every input datetime.fromisoformat accepts must convert cleanly."""
    for s in TIMES + EXTRA_TIMES:
        if not _iso_accepts(s):
            continue
        for arch in ({"timezone": "Europe/Berlin"},
                     {"utc_offset_seconds": 3600}, {}):
            dt, out, err = _timed(pipeline._claimed_utc, _HUSUM[0], _HUSUM[1], s, arch)
            assert err is None, f"_claimed_utc({s!r}, arch={arch}) raised {err}"
            assert dt < _BUDGET_S
            utc_iso, note = out
            utc = datetime.fromisoformat(utc_iso)
            assert utc.tzinfo is not None, "utc_iso must be tz-aware"
            assert isinstance(note, str) and note


def test_claimed_utc_non_iso_raises_only_value_or_type_error():
    """_claimed_utc runs fromisoformat OUTSIDE its try (pipeline.py:362), so
    non-ISO input raises. run() guards this at pipeline.py:419-426 — the only
    allowed failures here are ValueError/TypeError; anything else is a bug."""
    rejected = 0
    for s in TIMES + EXTRA_TIMES:
        if _iso_accepts(s):
            continue
        _, out, err = _timed(pipeline._claimed_utc, _HUSUM[0], _HUSUM[1], s, {})
        if err is None:
            utc_iso, _ = out
            datetime.fromisoformat(utc_iso)          # result must still be valid
        else:
            rejected += 1
            assert err.startswith(("ValueError", "TypeError")), \
                f"_claimed_utc({s!r}) raised unexpected {err}"
    _finding("LOW", f"_claimed_utc propagates ValueError/TypeError on "
                    f"{rejected} non-strict-ISO inputs (fromisoformat is outside "
                    f"the try at pipeline.py:362; run() pre-guards at 419-426)")


def test_claimed_utc_tz_and_fallback_paths():
    lat, lon = _HUSUM
    utc, note = pipeline._claimed_utc(lat, lon, "2025-06-15T15:00",
                                    {"timezone": "Europe/Berlin"})
    assert utc == "2025-06-15T13:00:00+00:00" and "zoneinfo" in note
    utc, note = pipeline._claimed_utc(lat, lon, "2025-06-15T15:00",
                                    {"utc_offset_seconds": 3600})
    assert utc == "2025-06-15T14:00:00+00:00" and "3600" in note
    utc, note = pipeline._claimed_utc(lat, lon, "2025-06-15T15:00", {})
    assert utc == "2025-06-15T15:00:00+00:00"
    # bogus tz name -> caught inside -> treated-as-UTC fallback
    utc, note = pipeline._claimed_utc(lat, lon, "2025-06-15T15:00",
                                    {"timezone": "Bogus/Zone"})
    assert utc == "2025-06-15T15:00:00+00:00" and "treated as UTC" in note
    # arch=None + weather down -> same fallback, still offline
    with mock.patch.object(weather, "historical_weather",
                           side_effect=RuntimeError("offline")):
        utc, note = pipeline._claimed_utc(lat, lon, "2025-06-15T15:00", None)
    assert utc == "2025-06-15T15:00:00+00:00" and "treated as UTC" in note


# ===========================================================================
# seam test — parser output must be resolver-safe (the real contract)
# ===========================================================================

def test_claim_to_args_output_feeds_resolvers_without_exception():
    """Whatever claim_to_args emits must be safe for _resolve_coords /
    _claimed_utc verbatim — this is the production seam (cli.py:91,
    serve.py:728)."""
    with mock.patch.object(geo, "geocode", side_effect=_fake_geocode):
        for s in ALL_TEXTS:
            t_iso, place = claim_parser.claim_to_args(s)
            dt, out, err = _timed(pipeline._resolve_coords, place)
            assert err is None, f"place from {s!r} crashed _resolve_coords: {err}"
            lat, lon, _ = out
            _coords_ok(lat, lon)
            if t_iso is not None:
                _, out, err = _timed(pipeline._claimed_utc,
                                     lat or 0.0, lon or 0.0, t_iso, {})
                assert err is None, \
                    f"time_iso {t_iso!r} from {s!r} crashed _claimed_utc: {err}"


# ===========================================================================
# reproduced wrong-value bugs (xfail — they assert CORRECT behaviour)
# ===========================================================================

@_known_bug
def test_xfail_iso_T_separator_leaks_as_place():
    r = claim_parser.parse_claim_text("2025-06-15T15:00", _FIXED_NOW)
    _finding("MED", f"ISO 'T' separator leaks as place: "
                    f"parse('2025-06-15T15:00')['place'] == {r['place']!r} "
                    f"(claim_parser.py _place_fallback keeps uncovered 'T')")
    assert r["place"] is None


@_known_bug
def test_xfail_invalid_clock_residue_leaks_as_place():
    r = claim_parser.parse_claim_text("2025-06-15T25:00", _FIXED_NOW)
    _finding("MED", f"rejected clock leaks as place: "
                    f"parse('2025-06-15T25:00')['place'] == {r['place']!r}")
    assert r["place"] is None


@_known_bug
def test_xfail_invalid_meridiem_leaks_as_place():
    bad = []
    for s in ("15pm", "13pm", "0pm"):
        r = claim_parser.parse_claim_text(s, _FIXED_NOW)
        if r["place"] is not None:
            bad.append(s)
            _finding("MED", f"invalid meridiem {s!r} becomes place {r['place']!r}")
    assert not bad


@_known_bug
def test_xfail_relative_phrases_leak_as_place():
    bad = []
    r = claim_parser.parse_claim_text("in two weeks", _FIXED_NOW)
    if r["place"] is not None or r["day_offset"] is not None:
        bad.append("in two weeks")
        _finding("MED", f"'in two weeks' -> place {r['place']!r} ('in' treated "
                        f"as place preposition; 'two weeks' unsupported)")
    r = claim_parser.parse_claim_text("next friday", _FIXED_NOW)
    if r["place"] is not None or (r["day_offset"] or 0) <= 0:
        bad.append("next friday")
        _finding("MED", f"'next friday' -> place {r['place']!r} AND day_offset "
                        f"{r['day_offset']} (past Friday, not next — 'next' ignored)")
    assert not bad


@_known_bug
def test_xfail_fallback_place_has_no_length_cap():
    """_clean_place caps at 60 chars (claim_parser.py:189) but
    _place_fallback doesn't — a 500-char 'place' reaches the geocoder."""
    r = claim_parser.parse_claim_text("a" * 500, _FIXED_NOW)
    _finding("MED", f"'a'*500 -> place of {len(r['place'] or '')} chars "
                    f"(_place_fallback lacks _clean_place's 60-char cap)")
    assert r["place"] is None or len(r["place"]) <= 60


@_known_bug
def test_xfail_date_only_claim_survives_to_args():
    """'2025-6-15' parses a day_offset but no clock -> time_iso None ->
    claim_to_args drops the date entirely; run() then skips all time checks."""
    r = claim_parser.parse_claim_text("2025-6-15", _FIXED_NOW)
    assert r["day_offset"] is not None          # the date WAS understood
    t_iso, _ = claim_parser.claim_to_args("2025-6-15")
    _finding("MED", f"date-only claim '2025-6-15' parsed (day_offset="
                    f"{r['day_offset']}) but claim_to_args emits time_iso=None — "
                    f"date silently discarded before pipeline.run")
    assert t_iso is not None


@_known_bug
def test_xfail_tz_suffix_silently_stripped():
    """_claimed_utc does .replace(tzinfo=None) (pipeline.py:362): a claim that
    is already UTC ('Z') is re-interpreted as local time -> hours off."""
    bad = []
    utc, _ = pipeline._claimed_utc(*_HUSUM, "2025-06-15T15:00Z",
                                   {"timezone": "Europe/Berlin"})
    if not utc.startswith("2025-06-15T15:00"):
        bad.append("Z")
        _finding("MED", f"'2025-06-15T15:00Z' + tz=Europe/Berlin -> {utc} "
                        f"(expected 2025-06-15T15:00:00+00:00; tzinfo stripped)")
    utc, _ = pipeline._claimed_utc(*_HUSUM, "2025-06-15T15:00+02:00", {})
    if not utc.startswith("2025-06-15T13:00"):
        bad.append("+02:00")
        _finding("MED", f"'2025-06-15T15:00+02:00' + offset-0 arch -> {utc} "
                        f"(expected 2025-06-15T13:00:00+00:00)")
    assert not bad


@_known_bug
def test_xfail_conflicting_place_resolves_without_ambiguity_note():
    """'London France' is contradictory; _resolve_coords takes cands[0]
    silently — no ambiguity signal in the note."""
    with mock.patch.object(geo, "geocode", side_effect=_fake_geocode):
        lat, lon, note = pipeline._resolve_coords("London France")
    _finding("MED", f"'London France' -> ({lat}, {lon}, {note!r}) — "
                    f"conflicting names resolved to first hit, no ambiguity note")
    assert lat is None or "ambig" in note.lower() or "conflict" in note.lower()


@_known_bug
def test_xfail_coord_literal_in_claim_text_is_dropped():
    """Judges can pass coords to run(claim_place=...) but the NL parser drops
    them: 'filmed at 54.47,9.05' -> place None (no [a-zA-Z] letter)."""
    r = claim_parser.parse_claim_text("filmed at 54.47,9.05", _FIXED_NOW)
    _finding("MED", f"coord literal in claim text dropped: "
                    f"'filmed at 54.47,9.05' -> place {r['place']!r}")
    assert r["place"] is not None


@_known_bug
def test_xfail_non_string_inputs_crash_public_api():
    """parse_claim_text/claim_to_args raise AttributeError/TypeError on
    non-str input (untyped signature). Current callers pass str / wrap in
    try, so dormant — but the public API itself is not defensive."""
    crashed = []
    for bad in NON_STRINGS:
        _, _, err = _timed(claim_parser.parse_claim_text, bad, _FIXED_NOW)
        if err:
            crashed.append(("parse_claim_text", bad, err))
        _, _, err = _timed(claim_parser.claim_to_args, bad)
        if err:
            crashed.append(("claim_to_args", bad, err))
    # _resolve_coords has the same gap via geo.geocode's query.strip() —
    # the strict gate reproduces the real crash before the network boundary.
    with mock.patch.object(geo, "geocode", side_effect=_strict_geocode_gate):
        for bad in NON_STRINGS:
            _, _, err = _timed(pipeline._resolve_coords, bad)
            if err:
                crashed.append(("_resolve_coords", bad, err))
    for name, bad, err in crashed:
        _finding("LOW", f"non-str input {bad!r} -> {name} raises {err}")
    assert not crashed


# ===========================================================================
# direct-run fallback
# ===========================================================================

def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            tag = "XPASS (bug fixed? update test)" if name in _XFAIL_NAMES else "PASS"
            print(f"{tag} {name}")
            if name in _XFAIL_NAMES:
                failed += 1   # strict: xpassed known-bug means suite needs update
        except AssertionError as e:
            if name in _XFAIL_NAMES:
                print(f"XFAIL (known bug still present) {name} — {e}")
            else:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    if FINDINGS:
        print("\n--- FINDINGS ---")
        for f_ in FINDINGS:
            print(" ", f_)
    hard = [n for n, _ in tests if n not in _XFAIL_NAMES]
    print(f"\n{len(tests) - failed}/{len(tests)} passed "
          f"({len(_XFAIL_NAMES)} known-bug probes, {len(hard)} invariant tests)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
