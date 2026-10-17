"""Tests for sundial.pipeline clock-overlay stage — run directly: python3 tests/test_clock.py"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sundial.pipeline import (  # noqa: E402
    _clock_evidence,
    _parse_overlay_time,
)
from sundial.schema import ClaimVerdict  # noqa: E402


def _clock(overlays, claim, frames=None):
    """Build (frame, text) pairs — each overlay read is mapped to its own
    frame so repeated strings form a genuine >=2-distinct-frame consensus.
    """
    if frames is not None:
        pairs = list(zip(frames, overlays))
    else:
        pairs = [(f"frame_{i:02d}.jpg", t) for i, t in enumerate(overlays)]
    return _clock_evidence(pairs, claim)


# --------------------------------------------------------------------------- #
# _parse_overlay_time
# --------------------------------------------------------------------------- #
def test_parse_dotted_eu_date_two_digit_year():
    dt, gran = _parse_overlay_time("17.10.26 12:30")
    assert gran == "datetime", (dt, gran)
    assert dt == datetime(2026, 10, 17, 12, 30), dt  # yy -> 20yy


def test_parse_dotted_eu_date_with_seconds():
    dt, gran = _parse_overlay_time("17.10.2026 08:00:02")
    assert gran == "datetime", (dt, gran)
    assert dt == datetime(2026, 10, 17, 8, 0, 2), dt


def test_parse_iso_not_misread_as_eu_date():
    # Regression: ISO stamps must be read by the yyyy-mm-dd pattern first —
    # a dd.mm.yy grab of the "26-10-17" tail would wrongly give 2017-10-26.
    dt, gran = _parse_overlay_time("2026-10-17T08:00")
    assert gran == "datetime", (dt, gran)
    assert dt == datetime(2026, 10, 17, 8, 0), dt
    assert dt != datetime(2017, 10, 26), dt


def test_parse_named_date_falls_back_to_time_only():
    # "Sat 17 Oct 11:31" has no parseable date, but the clock is usable.
    dt, gran = _parse_overlay_time("Sat 17 Oct 11:31")
    assert gran == "time", (dt, gran)
    assert (dt.hour, dt.minute, dt.second) == (11, 31, 0), dt


def test_parse_cam_prefix_time_only():
    dt, gran = _parse_overlay_time("CAM 08:00")
    assert gran == "time", (dt, gran)
    assert (dt.hour, dt.minute) == (8, 0), dt


def test_parse_junk_returns_none():
    for junk in ("foto-webcam.eu", "hello", "3.99V"):
        dt, gran = _parse_overlay_time(junk)
        assert dt is None and gran is None, (junk, dt, gran)


# --------------------------------------------------------------------------- #
# _clock_evidence — datetime-granularity overlays
# --------------------------------------------------------------------------- #
def test_clock_supports_within_45_minutes():
    ev = _clock(["17.10.2026 08:00", "17.10.2026 08:00"], "2026-10-17T08:30")
    assert ev.type == "clock"
    assert ev.verdict == ClaimVerdict.SUPPORTS, ev.summary
    assert ev.data["delta_minutes"] == 30.0, ev.data


def test_clock_contradicts_270_minutes_off():
    ev = _clock(["17.10.2026 08:00", "17.10.2026 08:00"], "2026-10-17T12:30")
    assert ev.verdict == ClaimVerdict.CONTRADICTS, ev.summary
    assert ev.data["delta_minutes"] == 270.0, ev.data


def test_clock_contradicts_different_date_same_clock_time():
    ev = _clock(["17.10.2026 08:00", "17.10.2026 08:00"], "2026-10-18T08:00")
    assert ev.verdict == ClaimVerdict.CONTRADICTS, ev.summary
    assert "different date" in ev.summary


# --------------------------------------------------------------------------- #
# _clock_evidence — time-only overlays can never carry a verdict
# --------------------------------------------------------------------------- #
def test_clock_time_only_near_claim_is_inconclusive_not_supports():
    ev = _clock(["08:00", "08:00"], "2026-10-17T08:05")
    assert ev.verdict == ClaimVerdict.INCONCLUSIVE, ev.summary
    assert ev.verdict != ClaimVerdict.SUPPORTS
    assert "time only" in ev.summary


def test_clock_time_only_far_off_is_inconclusive_not_contradicts():
    ev = _clock(["23:45", "23:45"], "2026-10-17T08:00")
    assert ev.verdict == ClaimVerdict.INCONCLUSIVE, ev.summary
    assert ev.verdict != ClaimVerdict.CONTRADICTS


# --------------------------------------------------------------------------- #
# _clock_evidence — missing inputs / consensus
# --------------------------------------------------------------------------- #
def test_clock_empty_overlays_inconclusive():
    ev = _clock([], "2026-10-17T08:00")
    assert ev.verdict == ClaimVerdict.INCONCLUSIVE
    assert ev.confidence == 0.0
    assert "no readable burned-in timestamp" in ev.summary


def test_clock_blank_overlays_cleaned_to_empty():
    ev = _clock(["", "   ", None], "2026-10-17T08:00")
    assert ev.verdict == ClaimVerdict.INCONCLUSIVE
    assert ev.data["overlays"] == []


def test_clock_no_claim_time_inconclusive():
    ev = _clock(["17.10.2026 08:00", "17.10.2026 08:00"], None)
    assert ev.verdict == ClaimVerdict.INCONCLUSIVE
    assert "no claimed time" in ev.summary


def test_clock_unparseable_overlay_inconclusive():
    # Singleton junk reads can't form a >=2-frame consensus — the stage
    # quarantines them rather than trusting a one-off misread.
    ev = _clock(["foto-webcam.eu", "3.99V"], "2026-10-17T08:00")
    assert ev.verdict == ClaimVerdict.INCONCLUSIVE
    assert "consensus" in ev.summary or "unparseable" in ev.summary


def test_clock_majority_parse_wins_supports():
    # The timestamp repeated across >=2 frames is consensus; a junk string
    # and a one-off minority clock can't derail it.
    ev = _clock(
        ["17.10.2026 08:00", "17.10.2026 08:00", "foto-webcam.eu", "CAM 22:10"],
        "2026-10-17T08:00",
    )
    assert ev.verdict == ClaimVerdict.SUPPORTS, ev.summary
    assert ev.data["parsed"] == "17.10.2026 08:00"


def test_clock_majority_parse_wins_contradicts():
    # Same consensus rule against the claim: the repeated overlay
    # contradicts even though a one-off read would have supported.
    ev = _clock(
        ["18.10.2026 09:00", "18.10.2026 09:00", "17.10.2026 08:00"],
        "2026-10-17T08:00",
    )
    assert ev.verdict == ClaimVerdict.CONTRADICTS, ev.summary
    assert ev.data["parsed"] == "18.10.2026 09:00"


# --------------------------------------------------------------------------- #
# claim_time handling — local naive compare, no timezone shift
# --------------------------------------------------------------------------- #
def test_clock_tz_aware_claim_compared_as_local_naive():
    # Claim time is local-to-place: an offset suffix is stripped, not
    # converted. 08:00+05:00 as UTC would be 03:00 — a tz shift would
    # report ~300 min off and contradict instead.
    ev = _clock(["17.10.2026 08:00", "17.10.2026 08:00"],
                "2026-10-17T08:00:00+05:00")
    assert ev.verdict == ClaimVerdict.SUPPORTS, ev.summary
    assert ev.data["delta_minutes"] == 0.0, ev.data


def test_clock_naive_claim_with_seconds_matches():
    ev = _clock(["2026-10-17T08:00:02", "2026-10-17T08:00:02"],
                "2026-10-17T08:00")
    assert ev.verdict == ClaimVerdict.SUPPORTS, ev.summary
    assert ev.data["delta_minutes"] < 1.0, ev.data


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
    print(f"\n{len(tests) - failed}/{len(tests)} ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
