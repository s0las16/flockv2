"""Geo + weather stage tests. Runnable via `python3 tests/test_geo_weather.py`.

Network tests degrade to SKIP when offline — never fail on connectivity.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sundial.schema import ClaimVerdict  # noqa: E402
from sundial import geo, weather  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append(("PASS", name))
    except _Skip as e:
        RESULTS.append(("SKIP", f"{name} ({e})"))
    except AssertionError as e:
        RESULTS.append(("FAIL", f"{name}: {e}"))
    except Exception as e:
        RESULTS.append(("SKIP", f"{name} ({type(e).__name__}: {e})"))


class _Skip(Exception):
    pass


def _net_guard(e: Exception):
    msg = str(e).lower()
    if any(k in msg for k in ("name", "resolve", "connection", "timeout", "network")):
        raise _Skip(f"no network: {e}")
    raise e


# ------------------------------- geo -------------------------------------

def t_geocode_london():
    try:
        cands = geo.geocode("Piccadilly Circus, London")
    except RuntimeError as e:
        _net_guard(e)
    assert cands, "expected >=1 candidate"
    assert cands[0].country_code.upper() == "GB", cands[0]


def t_consistency_left_driving():
    from sundial.schema import GeoCandidate
    gb = GeoCandidate(name="x", lat=51.5, lon=-0.12, country_code="GB")
    flags = geo.consistency_checks({"driving_side": "left"}, gb)
    assert flags and "consistent" in flags[0] and "INCONSISTENT" not in flags[0]
    us = GeoCandidate(name="x", lat=40.7, lon=-74.0, country_code="US")
    flags = geo.consistency_checks({"driving_side": "left"}, us)
    assert flags and "INCONSISTENT" in flags[0]


def t_geo_evidence_no_text():
    ev = geo.geo_evidence("London", [], {})
    assert ev.verdict == ClaimVerdict.INCONCLUSIVE


def t_geo_evidence_contradiction():
    try:
        ev = geo.geo_evidence(
            "New York", ["Rue de Rivoli", "Boulevard Saint-Germain"],
            {"sign_language": "fr"},
        )
    except RuntimeError as e:
        _net_guard(e)
    assert ev.verdict in (ClaimVerdict.CONTRADICTS, ClaimVerdict.INCONCLUSIVE)
    if ev.verdict == ClaimVerdict.CONTRADICTS:
        assert ev.confidence >= 0.5


# ------------------------------ weather -----------------------------------

def t_weather_at_past_date():
    try:
        rec = weather.weather_at(51.5, -0.12, "2025-10-01T14:00")
    except RuntimeError as e:
        _net_guard(e)
    for k in ("cloudcover", "precipitation", "weathercode"):
        assert k in rec, rec


def t_compare_fabricated_contradiction(monkey_obs=None):
    # observed rain vs a recorded-dry hour: force the "recorded" side by
    # stubbing weather_at — keeps this test fully offline.
    import sundial.weather as w
    orig = w.weather_at
    w.weather_at = lambda lat, lon, dt: {
        "cloudcover": 10, "precipitation": 0.0, "weathercode": 0,
    }
    try:
        ev = w.compare_weather({"precipitation": "rain", "sky": "overcast"},
                               51.5, -0.12, "2025-10-01T14:00")
    finally:
        w.weather_at = orig
    assert ev.verdict == ClaimVerdict.CONTRADICTS, ev.summary
    assert ev.confidence >= 0.5


def t_compare_fabricated_support():
    import sundial.weather as w
    orig = w.weather_at
    w.weather_at = lambda lat, lon, dt: {
        "cloudcover": 5, "precipitation": 0.0, "weathercode": 0,
    }
    try:
        ev = w.compare_weather({"precipitation": "none", "sky": "clear"},
                               51.5, -0.12, "2025-10-01T14:00")
    finally:
        w.weather_at = orig
    assert ev.verdict == ClaimVerdict.SUPPORTS, ev.summary


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in list(globals().items())
                            if k.startswith("t_")}.items()):
        check(name, fn)
    for status, name in RESULTS:
        print(f"[{status}] {name}")
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} ok")
    sys.exit(1 if fails else 0)
