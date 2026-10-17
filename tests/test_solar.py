"""Tests for sundial.solar — run directly: python3 tests/test_solar.py"""
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sundial.schema import ClaimVerdict  # noqa: E402
from sundial.solar import (  # noqa: E402
    azimuth_error,
    elevation_from_shadow,
    infer_time_window,
    solar_evidence,
    sun_position,
)

LAT, LON = 51.5, -0.12  # London
NOON = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)


def test_sun_position_noon():
    p = sun_position(LAT, LON, NOON)
    assert set(p) == {"elevation_deg", "azimuth_deg"}
    assert 58 < p["elevation_deg"] < 65, p  # London mid-June solar noon ~62
    # pysolar azimuth convention: 0..360 clockwise from north -> ~180 due south at noon
    assert 170 < p["azimuth_deg"] < 190, p


def test_sun_position_naive_is_utc():
    naive = sun_position(LAT, LON, datetime(2025, 6, 15, 12, 0))
    aware = sun_position(LAT, LON, NOON)
    assert naive == aware


def test_azimuth_error_wraparound():
    assert azimuth_error(359, 1) == 2
    assert azimuth_error(10, 350) == 20
    assert azimuth_error(90, 90) == 0
    assert azimuth_error(0, 180) == 180


def test_elevation_from_shadow():
    assert abs(elevation_from_shadow(1.7, 1.7) - 45.0) < 1e-9
    assert abs(elevation_from_shadow(1.0, math.sqrt(3)) - 30.0) < 1e-6
    for bad in (0.0, -1.2):
        try:
            elevation_from_shadow(1.7, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("no ValueError for shadow_length_m=%s" % bad)


def test_two_windows_elev20():
    # elev 20 crossed twice: ~06:14 and ~17:50 UTC on 2025-06-15 in London
    ws = infer_time_window(LAT, LON, "2025-06-15", 20.0)
    assert len(ws) == 2, ws
    for w in ws:
        assert set(w) == {"start", "end", "center", "minutes"}
        assert w["minutes"] > 0
        assert datetime.fromisoformat(w["start"]) < datetime.fromisoformat(w["end"])
    c0 = datetime.fromisoformat(ws[0]["center"])
    c1 = datetime.fromisoformat(ws[1]["center"])
    assert 5 <= c0.hour <= 8 and 16 <= c1.hour <= 20, (c0, c1)
    assert c0.hour * 60 + c0.minute < 720 < c1.hour * 60 + c1.minute  # straddle noon


def test_noon_window_with_azimuth():
    # az≈180 only occurs near max altitude (~62°), so (elev 55, az 180) with
    # default elev tol 3 matches nothing; widening tol to 8 lets azimuth pin noon
    ws = infer_time_window(LAT, LON, "2025-06-15", 55.0,
                           observed_azimuth_deg=180.0, elevation_tol_deg=8.0)
    assert len(ws) == 1, ws
    center = datetime.fromisoformat(ws[0]["center"])
    assert 11 <= center.hour <= 13, ws  # solar noon ~12:00 UTC


def test_elevation_above_max_returns_empty():
    assert infer_time_window(LAT, LON, "2025-06-15", 80.0) == []
    assert infer_time_window(LAT, LON, "2025-06-15", 62.0) != []  # sanity: max ~61.84


def test_solar_evidence_supports():
    obs = sun_position(LAT, LON, datetime(2025, 6, 15, 8, 30, tzinfo=timezone.utc))
    ev = solar_evidence(LAT, LON, "2025-06-15T08:30:00+00:00",
                        round(obs["elevation_deg"], 1), round(obs["azimuth_deg"], 1))
    assert ev.type == "solar"
    assert ev.verdict == ClaimVerdict.SUPPORTS, ev.summary
    assert ev.confidence > 0.5
    claimed = datetime(2025, 6, 15, 8, 30, tzinfo=timezone.utc)
    assert any(datetime.fromisoformat(w["start"]) <= claimed <= datetime.fromisoformat(w["end"])
               for w in ev.data["windows_utc"]), ev.data


def test_solar_evidence_supports_naive_claim():
    obs = sun_position(LAT, LON, datetime(2025, 6, 15, 8, 30, tzinfo=timezone.utc))
    ev = solar_evidence(LAT, LON, "2025-06-15T08:30:00",  # naive -> assumed UTC
                        round(obs["elevation_deg"], 1))
    assert ev.verdict == ClaimVerdict.SUPPORTS, ev.summary


def test_solar_evidence_contradicts():
    obs = sun_position(LAT, LON, datetime(2025, 6, 15, 8, 30, tzinfo=timezone.utc))
    ev = solar_evidence(LAT, LON, "2025-06-15T14:30:00+00:00",  # 6h off, still daylight
                        round(obs["elevation_deg"], 1), round(obs["azimuth_deg"], 1))
    assert ev.verdict == ClaimVerdict.CONTRADICTS, ev.summary
    assert ev.data["elevation_error_deg"] > ev.data["tolerances_deg"]["elevation"]


def test_solar_evidence_nighttime_claim_contradicts():
    # Daylight shadow measurement + a claim deep enough into night that the
    # sun is below the horizon is a contradiction, not inconclusive.
    obs = sun_position(LAT, LON, datetime(2025, 6, 15, 8, 30, tzinfo=timezone.utc))
    ev = solar_evidence(LAT, LON, "2025-06-15T02:30:00+00:00",
                        round(obs["elevation_deg"], 1))
    assert ev.verdict == ClaimVerdict.CONTRADICTS, ev.summary


def test_solar_evidence_twilight_edge_inconclusive():
    # Claim sits in civil twilight (-6..0 deg): lingering low shadows can
    # persist just after sunset, so a small measured elevation stays
    # inconclusive. A *high* measured sun, though, can't be twilight.
    ev = solar_evidence(LAT, LON, "2025-06-15T20:45:00+00:00", 2.0)
    assert ev.verdict == ClaimVerdict.INCONCLUSIVE, ev.summary
    ev = solar_evidence(LAT, LON, "2025-06-15T20:45:00+00:00", 35.0)
    assert ev.verdict == ClaimVerdict.CONTRADICTS, ev.summary


def test_solar_evidence_impossible_elevation_inconclusive():
    ev = solar_evidence(LAT, LON, "2025-06-15T12:00:00+00:00", 80.0)
    assert ev.verdict == ClaimVerdict.INCONCLUSIVE, ev.summary


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
