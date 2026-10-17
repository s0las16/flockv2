"""Weather cross-check — Open-Meteo archive vs conditions observed in the clip.

Free API, no key. `timezone=auto` makes returned hour stamps local to the
queried coordinates, matching the schema's "claimed_time is local to the
claimed place". Caveat: the archive lags ~5 days behind real time and returns
nulls for too-recent dates — handled as missing data (INCONCLUSIVE).
"""
from __future__ import annotations

import requests

from .schema import ClaimVerdict, Evidence

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT_S = 10
HOURLY_VARS = "cloudcover,precipitation,weathercode"

# WMO weather interpretation codes (subset).
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snowfall", 73: "moderate snowfall", 75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers", 85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}
_RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
_SNOW_CODES = {71, 73, 75, 77, 85, 86}


def describe_code(code: object) -> str:
    try:
        return _WMO.get(int(code), f"wmo code {code}")
    except (TypeError, ValueError):
        return "unknown"


def historical_weather(lat: float, lon: float, date_iso: str) -> dict:
    """Fetch one day of hourly archive data. Raises RuntimeError on failure."""
    date = str(date_iso)[:10]
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": date,
        "end_date": date,
        "hourly": HOURLY_VARS,
        "timezone": "auto",  # hour stamps come back local to lat/lon
    }
    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise RuntimeError(f"Open-Meteo archive request failed: {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        payload = resp.json()
    except ValueError as e:
        raise RuntimeError("Open-Meteo returned a non-JSON response") from e
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason') or payload}")
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise RuntimeError(
            f"Open-Meteo returned no hourly rows for {date} at {lat},{lon}"
        )

    def _col(name: str, i: int):
        vals = hourly.get(name) or []
        return vals[i] if i < len(vals) else None

    by_hour: dict[str, dict] = {}
    for i, t in enumerate(times):
        by_hour[str(t)[:13]] = {
            "cloudcover": _col("cloudcover", i),
            "precipitation": _col("precipitation", i),
            "weathercode": _col("weathercode", i),
        }
    return {
        "date": date,
        "latitude": payload.get("latitude", float(lat)),
        "longitude": payload.get("longitude", float(lon)),
        "timezone": payload.get("timezone"),
        "utc_offset_seconds": payload.get("utc_offset_seconds"),
        "hourly": hourly,  # raw arrays: time[], cloudcover[], precipitation[], weathercode[]
        "by_hour": by_hour,  # "YYYY-MM-DDTHH" -> {cloudcover, precipitation, weathercode}
    }


def conditions_at(archive: dict, dt_iso: str) -> dict:
    """Pull the record for one ISO datetime out of a `historical_weather` result.

    Matches on the "YYYY-MM-DDTHH" prefix, in the archive's local timezone.
    """
    key = str(dt_iso)[:13]
    rec = (archive or {}).get("by_hour", {}).get(key)
    if rec is None:
        raise RuntimeError(f"no archive record for hour {key!r}")
    return rec


def weather_at(lat: float, lon: float, dt_iso: str) -> dict:
    """Single-datetime lookup -> {cloudcover, precipitation, weathercode, ...}."""
    archive = historical_weather(lat, lon, str(dt_iso)[:10])
    rec = dict(conditions_at(archive, dt_iso))
    rec["time"] = str(dt_iso)[:13]
    rec["conditions"] = describe_code(rec.get("weathercode"))
    return rec


def _assess(observed: dict, recorded: dict) -> tuple[list[str], list[str], list[str]]:
    """Compare clip observations with archive records.

    Returns (agreements, contradictions, notes). Values of None in the record
    (archive gaps) simply skip that comparison.
    """
    obs = observed or {}
    agree: list[str] = []
    contra: list[str] = []
    notes: list[str] = []

    obs_precip = (obs.get("precipitation") or "").strip().lower()
    obs_sky = (obs.get("sky") or "").strip().lower()
    wet = obs.get("wet_ground")
    r_precip = recorded.get("precipitation")
    r_cloud = recorded.get("cloudcover")
    r_code = recorded.get("weathercode")
    try:
        code_int = int(r_code) if r_code is not None else None
    except (TypeError, ValueError):
        code_int = None

    if obs_precip and r_precip is not None:
        if obs_precip == "none":
            if r_precip >= 1.0:
                # A grid-cell archive reading needs real volume to outweigh a
                # visibly dry clip — drizzle at 0.3mm is too thin to contradict.
                if wet is False:
                    contra.append(
                        f"archive records {r_precip}mm precipitation but the "
                        f"clip shows none and the ground is dry"
                    )
                else:
                    notes.append(
                        f"archive records {r_precip}mm precipitation; clip "
                        f"shows no rainfall but wet ground is ambiguous"
                    )
            elif r_precip > 0.2:
                notes.append(
                    f"archive records {r_precip}mm precipitation; below the "
                    f"threshold to weigh against a dry-looking clip"
                )
            else:
                agree.append(
                    f"recorded precipitation {r_precip}mm matches a dry-looking clip"
                )
        elif obs_precip == "rain":
            if r_precip >= 0.2 or code_int in _RAIN_CODES:
                agree.append(
                    f"recorded {r_precip}mm precipitation "
                    f"('{describe_code(code_int)}') consistent with rain on video"
                )
            elif code_int in _SNOW_CODES:
                contra.append(
                    f"clip shows rain but archive reports "
                    f"'{describe_code(code_int)}'"
                )
            else:
                contra.append(
                    f"clip shows rain but archive records {r_precip}mm "
                    f"precipitation ('{describe_code(code_int)}')"
                )
        elif obs_precip == "snow":
            if code_int in _SNOW_CODES:
                agree.append(
                    f"recorded '{describe_code(code_int)}' consistent with snow"
                )
            elif r_precip >= 0.2:
                notes.append(
                    f"recorded precipitation ({r_precip}mm, "
                    f"'{describe_code(code_int)}') is not snow-class"
                )
            else:
                contra.append(
                    f"clip shows snow but archive records {r_precip}mm "
                    f"precipitation ('{describe_code(code_int)}')"
                )

    if obs_sky and r_cloud is not None:
        if obs_sky == "overcast":
            if r_cloud < 20:
                contra.append(
                    f"archive records {r_cloud}% cloud cover but the clip sky "
                    f"is overcast"
                )
            elif r_cloud >= 70:
                agree.append(
                    f"recorded {r_cloud}% cloud cover consistent with overcast sky"
                )
        elif obs_sky == "clear":
            if r_cloud > 80:
                contra.append(
                    f"archive records {r_cloud}% cloud cover but the clip sky "
                    f"is clear"
                )
            elif r_cloud <= 30:
                agree.append(
                    f"recorded {r_cloud}% cloud cover consistent with clear sky"
                )
        elif obs_sky == "partly":
            if r_cloud <= 5 or r_cloud >= 98:
                notes.append(
                    f"recorded {r_cloud}% cloud cover sits oddly with a "
                    f"'partly cloudy' observation"
                )
            else:
                agree.append(
                    f"recorded {r_cloud}% cloud cover consistent with partly "
                    f"cloudy sky"
                )

    if wet is True and r_precip is not None and r_precip > 0.2:
        agree.append(
            f"wet ground matches recorded {r_precip}mm precipitation"
        )
    if wet is False and r_precip is not None and r_precip > 1.0:
        notes.append(
            f"ground looks dry despite {r_precip}mm recorded precipitation"
        )
    return agree, contra, notes


def compare_weather(
    observed: dict,
    lat: float,
    lon: float,
    claimed_dt_iso: str,
    archive: dict | None = None,
) -> Evidence:
    """Build a weather Evidence card: clip observations vs archive records.

    `archive` is a pre-fetched historical_weather() response — pass it to
    avoid a second HTTP call when the caller already has the day's archive.
    """
    obs = dict(observed or {})
    base = {"lat": lat, "lon": lon, "claimed_dt": claimed_dt_iso, "observed": obs}
    try:
        if archive is not None:
            recorded = dict(conditions_at(archive, claimed_dt_iso))
            recorded["time"] = str(claimed_dt_iso)[:13]
            recorded["conditions"] = describe_code(recorded.get("weathercode"))
            base["archive"] = archive
        else:
            recorded = weather_at(lat, lon, claimed_dt_iso)
    except RuntimeError as e:
        return Evidence(
            type="weather",
            summary=f"could not fetch historical weather: {e}",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data={**base, "error": str(e)},
        )

    agree, contra, notes = _assess(obs, recorded)
    data = {
        **base,
        "recorded": recorded,
        "agreements": agree,
        "contradictions": contra,
        "notes": notes,
    }
    if not agree and not contra:
        return Evidence(
            type="weather",
            summary="no comparable weather observations vs archive",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )
    if contra:
        # One archive-vs-clip contradiction stays under the high-confidence
        # adjudicator threshold — reanalysis data is corroboration, not proof.
        conf = min(0.4 + 0.15 * len(contra), 0.9)
        return Evidence(
            type="weather",
            summary=f"recorded weather contradicts clip: {'; '.join(contra)}",
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=conf,
            data=data,
        )
    return Evidence(
        type="weather",
        summary=f"recorded weather consistent with clip: {'; '.join(agree)}",
        verdict=ClaimVerdict.SUPPORTS,
        confidence=min(0.5 + 0.1 * len(agree), 0.8),
        data=data,
    )
