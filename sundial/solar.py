"""Solar engine — invert an observed sun position into possible UTC time windows.

pysolar convention (verified on pysolar 0.13): get_altitude returns sun
elevation in degrees above the horizon (negative = below); get_azimuth returns
degrees 0..360 measured clockwise from true north (0=N, 90=E, 180=S, 270=W).
"""
from __future__ import annotations

import math
from datetime import date as _date, datetime, timedelta, timezone

from pysolar.solar import get_altitude, get_azimuth

from .schema import ClaimVerdict, Evidence

SCAN_STEP = timedelta(minutes=2)


def _aware(when: datetime) -> datetime:
    # Sundial works in UTC end to end; naive datetimes are read as UTC
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def azimuth_error(a: float, b: float) -> float:
    """Shortest distance between two azimuths in degrees (0/360 wrap-safe)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def sun_position(lat: float, lon: float, when: datetime) -> dict:
    w = _aware(when)
    return {
        "elevation_deg": get_altitude(lat, lon, w),
        "azimuth_deg": get_azimuth(lat, lon, w) % 360.0,
    }


def elevation_from_shadow(object_height_m: float, shadow_length_m: float) -> float:
    if shadow_length_m <= 0:
        raise ValueError(
            f"shadow_length_m must be > 0, got {shadow_length_m} "
            "(zero/negative shadow means sun overhead or below horizon)"
        )
    if object_height_m < 0:
        raise ValueError(f"object_height_m must be >= 0, got {object_height_m}")
    return math.degrees(math.atan2(object_height_m, shadow_length_m))


def infer_time_window(
    lat: float,
    lon: float,
    date: _date | datetime | str,
    observed_elevation_deg: float,
    observed_azimuth_deg: float | None = None,
    elevation_tol_deg: float = 3.0,
    azimuth_tol_deg: float = 8.0,
) -> list[dict]:
    """Scan the whole UTC day in 2-min steps; merge contiguous matches.

    Returned windows are UTC (start/end/center ISO strings). Matches are not
    merged across the 00:00 boundary. An elevation above the day's maximum
    simply yields [] — nothing on that date can produce it.
    """
    if isinstance(date, datetime):
        day = date.date()
    elif isinstance(date, str):
        day = _date.fromisoformat(date)
    else:
        day = date

    t0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    matches: list[datetime] = []
    for i in range(int(24 * 60 / (SCAN_STEP.total_seconds() / 60))):
        t = t0 + i * SCAN_STEP
        if abs(get_altitude(lat, lon, t) - observed_elevation_deg) > elevation_tol_deg:
            continue
        if observed_azimuth_deg is not None:
            az = get_azimuth(lat, lon, t) % 360.0
            if azimuth_error(az, observed_azimuth_deg) > azimuth_tol_deg:
                continue
        matches.append(t)

    windows: list[dict] = []
    for t in matches:
        if windows and t - windows[-1]["_last"] == SCAN_STEP:
            windows[-1]["_last"] = t
        else:
            windows.append({"_first": t, "_last": t})

    out = []
    for w in windows:
        start, end = w["_first"], w["_last"] + SCAN_STEP  # end = first non-matching sample
        out.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "center": (start + (end - start) / 2).isoformat(),
            "minutes": int((end - start).total_seconds() // 60),
        })
    return out


def solar_evidence(
    lat: float,
    lon: float,
    claim_time_iso: str,
    observed_elevation_deg: float,
    observed_azimuth_deg: float | None = None,
    elevation_tol_deg: float = 3.0,
    azimuth_tol_deg: float = 8.0,
) -> Evidence:
    claimed = _aware(datetime.fromisoformat(claim_time_iso))
    expected = sun_position(lat, lon, claimed)

    obs_txt = f"{observed_elevation_deg:.1f}° elev"
    exp_txt = f"{expected['elevation_deg']:.1f}° elev"
    if observed_azimuth_deg is not None:
        obs_txt += f" / {observed_azimuth_deg:.0f}° az"
        exp_txt += f" / {expected['azimuth_deg']:.0f}° az"

    data: dict = {
        "claim_time_utc": claimed.isoformat(),
        "expected": expected,
        "observed": {
            "elevation_deg": observed_elevation_deg,
            "azimuth_deg": observed_azimuth_deg,
        },
        "tolerances_deg": {"elevation": elevation_tol_deg, "azimuth": azimuth_tol_deg},
        "azimuth_convention": "0..360 deg clockwise from true north (pysolar)",
    }

    if expected["elevation_deg"] < -6:
        # Claim puts the sun well below the horizon (past civil twilight),
        # but we measured a real cast shadow — only possible in daylight.
        below = -expected["elevation_deg"]
        return Evidence(
            type="solar",
            summary=(f"clip shows a cast shadow (sun ≈{observed_elevation_deg:.0f}° up) "
                     f"but claimed {claimed:%H:%M} UTC puts the sun "
                     f"{below:.0f}° below the horizon"),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=0.9 if below >= 12 else 0.7,
            data={**data, "windows_utc": []},
        )
    if expected["elevation_deg"] < 0:
        # Twilight edge: shadows can persist just after sunset — can't
        # cleanly call it either way.
        return Evidence(
            type="solar",
            summary=(f"claimed {claimed:%H:%M} UTC is twilight "
                     f"(sun {expected['elevation_deg']:.1f}°); solar check inconclusive"),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.3,
            data={**data, "windows_utc": []},
        )

    windows = infer_time_window(
        lat, lon, claimed.date(), observed_elevation_deg, observed_azimuth_deg,
        elevation_tol_deg, azimuth_tol_deg,
    )
    win_txt = ", ".join(f"{w['start'][11:16]}–{w['end'][11:16]}" for w in windows)
    data["windows_utc"] = windows

    if not windows:
        return Evidence(
            type="solar",
            summary=(f"no time on {claimed.date()} puts the sun at {obs_txt} "
                     f"±{elevation_tol_deg}° — observation can't be placed"),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.3,
            data=data,
        )

    e_err = abs(expected["elevation_deg"] - observed_elevation_deg)
    a_err = (azimuth_error(expected["azimuth_deg"], observed_azimuth_deg)
             if observed_azimuth_deg is not None else None)
    data["elevation_error_deg"] = e_err
    data["azimuth_error_deg"] = a_err

    consistent = e_err <= elevation_tol_deg and (a_err is None or a_err <= azimuth_tol_deg)
    base = f"sun at {obs_txt} points to {win_txt} UTC; claimed {claimed:%H:%M} expects {exp_txt}"
    if consistent:
        # confidence scales with margin: exact match ~0.95, edge-of-tol ~0.6
        margin = 1.0 - e_err / elevation_tol_deg
        if a_err is not None:
            margin = min(margin, 1.0 - a_err / azimuth_tol_deg)
        return Evidence(type="solar", summary=f"{base} — consistent",
                        verdict=ClaimVerdict.SUPPORTS,
                        confidence=round(0.6 + 0.35 * max(margin, 0.0), 2), data=data)
    # contradiction confidence grows as the miss exceeds tolerance
    return Evidence(type="solar", summary=f"{base} — mismatch",
                    verdict=ClaimVerdict.CONTRADICTS,
                    confidence=round(min(0.95, 0.5 + e_err / (4 * elevation_tol_deg)), 2),
                    data=data)
