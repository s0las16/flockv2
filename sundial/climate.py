"""Climate-normals cross-check — ERA5 climatology vs the clip's scene cues.

Daily weather is noisy (weather.py checks the claimed day); climate
NORMALS are a steadier geo prior — palm trees where the normals say
"cold" is a contradiction that survives grid noise and daily anomalies.

Normals come from the Open-Meteo climate API (ERA5 over a ~3-year
range); when that endpoint can't serve the window we fall back to the
archive API for same-month seasonal normals over the last 3 full years.
climate_normals() never raises — {} on failure. The requests/urllib and
schema imports are lazy inside the functions, so this module imports
with zero deps.

  python -m sundial.climate                  # offline cue-logic self-test
  SUNDIAL_NET=1 python -m sundial.climate    # + a live Open-Meteo lookup
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from calendar import monthrange
from datetime import date

CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"  # as weather.py
TIMEOUT_S = 10
USER_AGENT = "sundial-hackathon/0.1 (video provenance verification)"
MODEL = "ERA5"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum"
MIN_DAYS = 180  # less coverage than ~6 months can't be called a "normal"

_CONF_CAP = 0.55  # climate is a weak prior, never the decider
_HINTS = ("cold", "temperate", "hot", "tropical")

__all__ = ["climate_normals", "climate_evidence"]

# cue keyword -> (expected climate hints, strict?). Strict keywords are
# specific enough that a normals mismatch counts as a real contradiction
# ("snow" can't sit under tropical normals); soft ones only leave a note
# on mismatch, so "lush" alone never convicts.
_CUE_TABLE = (
    ("snow",          {"cold", "temperate"}, True),
    ("snowy",         {"cold", "temperate"}, True),
    ("sleet",         {"cold", "temperate"}, True),
    ("blizzard",      {"cold"}, True),
    ("glacier",       {"cold"}, True),
    ("tundra",        {"cold"}, True),
    ("permafrost",    {"cold"}, True),
    ("ice",           {"cold"}, True),
    ("icy",           {"cold"}, True),
    ("frost",         {"cold", "temperate"}, True),
    ("boreal",        {"cold"}, True),
    ("taiga",         {"cold"}, True),
    ("conifer",       {"cold", "temperate"}, False),
    ("pine",          {"cold", "temperate"}, False),
    ("spruce",        {"cold", "temperate"}, False),
    ("birch",         {"cold", "temperate"}, False),
    ("deciduous",     {"temperate", "cold"}, False),
    ("bare",          {"temperate", "cold"}, False),  # leafless trees
    ("temperate",     {"temperate"}, False),
    ("mediterranean", {"hot", "temperate"}, False),
    ("lush",          {"tropical", "temperate"}, False),
    ("tropical",      {"tropical"}, True),
    ("rainforest",    {"tropical"}, True),
    ("jungle",        {"tropical"}, True),
    ("monsoon",       {"tropical"}, True),
    ("coconut",       {"tropical"}, True),
    ("coral",         {"tropical"}, True),
    ("palm",          {"hot", "tropical"}, True),
    ("banana",        {"tropical", "hot"}, True),
    ("savanna",       {"tropical", "hot"}, True),
    ("desert",        {"hot"}, True),
    ("dune",          {"hot"}, True),
    ("arid",          {"hot"}, True),
    ("cactus",        {"hot"}, True),
    ("cacti",         {"hot"}, True),
    ("dust",          {"hot"}, False),
    ("dusty",         {"hot"}, False),
    ("dry",           {"hot", "temperate"}, False),
)
# Word-start boundary + optional plural so "palm" catches "palm trees"/
# "palms" but "ice" can't match inside "office"/"service".
_CUE_RES = tuple(
    (k, re.compile(r"\b" + re.escape(k) + r"(?:s|es)?\b"), hints, strict)
    for k, hints, strict in _CUE_TABLE
)


def _get_json(url: str, params: dict) -> dict:
    """GET `url` as JSON — requests when installed, stdlib urllib otherwise
    (same transport convention as sources.py._fetch_bytes). Raises
    RuntimeError on transport, HTTP or parse failure."""
    try:
        import requests  # lazy: optional third-party dep
    except ImportError:
        requests = None

    if requests is None:
        full = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GET {url} -> HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"GET {url} failed: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"GET {url} returned non-JSON") from e

    try:
        resp = requests.get(
            url, params=params,
            headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"GET {url} failed: {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"GET {url} -> HTTP {resp.status_code}: {resp.text[:160]!r}"
        )
    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError(f"GET {url} returned non-JSON") from e


def _daily_col(daily: dict, prefix: str) -> list:
    """First daily array whose key starts with `prefix` — the climate API
    suffixes variable names with the model when several are requested."""
    for k, v in (daily or {}).items():
        if k.startswith(prefix) and isinstance(v, list):
            return v
    return []


def _climate_hint(t_max_c: float, t_min_c: float, precip_mm_month=None) -> str:
    """Coarse climate label from mean daily extremes (Köppen-flavoured).

    tropical — warm nights year-round AND not arid (a dry 20°C-night
               climate is desert-hot, not rainforest-tropical).
    hot      — hot days but cooler nights: arid/subtropical.
    cold     — short cool summers and/or real winters.
    temperate— the broad middle band.
    """
    if t_min_c >= 18.0:
        if precip_mm_month is None or precip_mm_month >= 40.0:
            return "tropical"
        return "hot"
    if t_max_c >= 25.0:
        return "hot"
    if t_max_c <= 10.0 or t_min_c <= -3.0:
        return "cold"
    return "temperate"


def _normals_dict(tmax: list, tmin: list, precip: list,
                  *, per_day_precip: bool) -> dict:
    t_max_c = sum(tmax) / len(tmax)
    t_min_c = sum(tmin) / len(tmin)
    # per_day_precip: daily sums -> average month; else precip already
    # holds monthly totals. None when no usable series at all.
    if precip:
        mm = (sum(precip) / len(precip)) * 30.44 if per_day_precip \
            else sum(precip) / len(precip)
    else:
        mm = None
    return {
        "t_max_c": round(t_max_c, 1),
        "t_min_c": round(t_min_c, 1),
        "precip_mm_month": round(mm, 1) if mm is not None else None,
        "climate_hint": _climate_hint(t_max_c, t_min_c, mm),
    }


def _normals_from_climate_api(lat: float, lon: float) -> dict:
    """Annual normals: mean daily extremes over the last 3 full years."""
    end_y = date.today().year - 1
    payload = _get_json(CLIMATE_URL, {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": f"{end_y - 2}-01-01",
        "end_date": f"{end_y}-12-31",
        "models": MODEL,
        "daily": DAILY_VARS,
        "timezone": "auto",  # same convention as weather.py
    })
    if payload.get("error"):
        raise RuntimeError(
            f"Open-Meteo climate error: {payload.get('reason') or payload}")
    daily = payload.get("daily") or {}
    tmax = [v for v in _daily_col(daily, "temperature_2m_max")
            if v is not None]
    tmin = [v for v in _daily_col(daily, "temperature_2m_min")
            if v is not None]
    precip = [v for v in _daily_col(daily, "precipitation_sum")
              if v is not None]
    if len(tmax) < MIN_DAYS or len(tmin) < MIN_DAYS:
        raise RuntimeError(
            f"climate API returned only {len(tmax)} usable days")
    return _normals_dict(tmax, tmin, precip, per_day_precip=True)


def _normals_from_archive(lat: float, lon: float) -> dict:
    """Seasonal normals: today's month across the last 3 full years —
    the simpler option, used when the climate API can't serve."""
    today = date.today()
    tmax: list = []
    tmin: list = []
    month_totals: list = []
    for dy in (1, 2, 3):
        y, m = today.year - dy, today.month
        payload = _get_json(ARCHIVE_URL, {
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": f"{y}-{m:02d}-01",
            "end_date": f"{y}-{m:02d}-{monthrange(y, m)[1]}",
            "daily": DAILY_VARS,
            "timezone": "auto",
        })
        if payload.get("error"):
            raise RuntimeError(
                f"Open-Meteo archive error: "
                f"{payload.get('reason') or payload}")
        daily = payload.get("daily") or {}
        tmax += [v for v in _daily_col(daily, "temperature_2m_max")
                 if v is not None]
        tmin += [v for v in _daily_col(daily, "temperature_2m_min")
                 if v is not None]
        pm = [v for v in _daily_col(daily, "precipitation_sum")
              if v is not None]
        if pm:
            month_totals.append(sum(pm))
    if len(tmax) < 10 or len(tmin) < 10:
        raise RuntimeError("archive returned too few usable days")
    return _normals_dict(tmax, tmin, month_totals, per_day_precip=False)


def _fetch_normals(lat: float, lon: float) -> tuple[dict, str | None]:
    """(normals, source) — climate API first, archive fallback."""
    for name, fetch in (("climate-api", _normals_from_climate_api),
                        ("archive", _normals_from_archive)):
        try:
            return fetch(lat, lon), name
        except Exception:
            continue
    return {}, None


def climate_normals(lat: float, lon: float) -> dict:
    """-> {"t_max_c","t_min_c","precip_mm_month","climate_hint"}, {} on
    failure. Never raises — climate is a prior, not worth crashing a
    pipeline over."""
    return _fetch_normals(lat, lon)[0]


def _match_cues(clip_cues: list) -> list[dict]:
    """Recognize climate keywords in each cue phrase ->
    [{cue, matched:[keys], expected:[hints], strict:bool}]."""
    out = []
    for raw in clip_cues:
        text = str(raw).strip().lower()
        # "ice cream truck" is dessert, not climate — keep the strict
        # "ice" cue from matching it.
        text = re.sub(r"\bice\s+cream\b", "icecream", text)
        if not text:
            continue
        keys, expected, strict = [], set(), False
        for key, rx, hints, is_strict in _CUE_RES:
            if rx.search(text):
                keys.append(key)
                expected |= hints
                strict = strict or is_strict
        if keys:
            out.append({
                "cue": str(raw),
                "matched": keys,
                "expected": sorted(expected),
                "strict": strict,
            })
    return out


def climate_evidence(
    lat: float | None,
    lon: float | None,
    clip_cues: list[str],
    normals: dict | None = None,
) -> "Evidence":
    """Build an Evidence card: ERA5 climate normals vs scene climate cues.

    `clip_cues` is a free-text keyword list from scene features — things
    like "snow", "palm trees", "bare deciduous", "tropical", "desert",
    "lush". `normals` may inject a pre-fetched climate_normals() result
    (pipeline reuse + offline tests); otherwise it's fetched live.

    Verdicts: strict cue clashes with the normals' climate hint ->
    CONTRADICTS (~0.5); cues fit the hint -> SUPPORTS (~0.5, less for
    soft-only matches); no usable cues/coords/normals -> INCONCLUSIVE.
    Confidence is capped at 0.55 — a climate zone is a geo prior, never
    the decider.
    """
    from .schema import ClaimVerdict, Evidence  # lazy: zero-dep import

    cues = [clip_cues] if isinstance(clip_cues, str) else list(clip_cues or [])
    matched = _match_cues(cues)
    data = {"lat": lat, "lon": lon, "clip_cues": cues,
            "matched_cues": matched}

    if lat is None or lon is None:
        return Evidence(
            type="climate",
            summary="no coordinates — cannot look up climate normals",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data=data,
        )
    if not matched:
        return Evidence(
            type="climate",
            summary="no recognizable climate cues in the clip",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.15,
            data=data,
        )

    if normals is None:
        normals, source = _fetch_normals(lat, lon)
        data["normals_source"] = source
    else:
        data["normals_source"] = "injected"
    data["normals"] = normals

    hint = str((normals or {}).get("climate_hint") or "").lower()
    if not normals or hint not in _HINTS:
        return Evidence(
            type="climate",
            summary=f"could not fetch climate normals for {lat},{lon}",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )

    agree: list[str] = []
    contra: list[str] = []
    notes: list[str] = []
    for m in matched:
        exp = set(m["expected"])
        if hint in exp:
            agree.append(f"'{m['cue']}' fits {hint} normals")
        elif m["strict"]:
            contra.append(f"'{m['cue']}' clashes with {hint} normals")
        else:
            notes.append(
                f"'{m['cue']}' sits oddly with {hint} normals (weak cue)")
    data.update(agreements=agree, contradictions=contra, notes=notes)

    desc = (f"{hint} normals (t_max {normals.get('t_max_c')}°C, "
            f"t_min {normals.get('t_min_c')}°C, "
            f"{normals.get('precip_mm_month')} mm/mo)")
    if contra:
        conf = min(_CONF_CAP, 0.5 + 0.05 * (len(contra) - 1))
        return Evidence(
            type="climate",
            summary=f"{desc} contradict the clip: {'; '.join(contra)}",
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=round(conf, 2),
            data=data,
        )
    if agree:
        strict_hits = sum(
            1 for m in matched if m["strict"] and hint in set(m["expected"]))
        base = 0.5 if strict_hits else 0.35
        conf = min(_CONF_CAP, base + 0.05 * (len(agree) - 1))
        return Evidence(
            type="climate",
            summary=f"{desc} consistent with clip cues: {'; '.join(agree)}",
            verdict=ClaimVerdict.SUPPORTS,
            confidence=round(conf, 2),
            data=data,
        )
    return Evidence(
        type="climate",
        summary=f"{desc}, but clip cues are too weak to compare: "
                f"{'; '.join(notes)}",
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.2,
        data=data,
    )


if __name__ == "__main__":
    import os
    import sys

    # Canned normals per hint — climate_evidence takes them via `normals`,
    # so the whole self-test runs offline.
    canned = {
        "tropical":  {"t_max_c": 31.4, "t_min_c": 24.1,
                      "precip_mm_month": 220.0, "climate_hint": "tropical"},
        "hot":       {"t_max_c": 33.0, "t_min_c": 15.0,
                      "precip_mm_month": 8.0, "climate_hint": "hot"},
        "temperate": {"t_max_c": 15.2, "t_min_c": 7.4,
                      "precip_mm_month": 55.0, "climate_hint": "temperate"},
        "cold":      {"t_max_c": 8.0, "t_min_c": -4.0,
                      "precip_mm_month": 40.0, "climate_hint": "cold"},
    }
    cases = [
        # (clip_cues, normals hint, expected verdict)
        (["palm trees", "tropical undergrowth"], "tropical", "supports"),
        (["palm trees", "tropical"], "cold", "contradicts"),
        (["snow on the ground"], "tropical", "contradicts"),
        (["fresh snow"], "cold", "supports"),
        (["snow"], "temperate", "supports"),  # temperate winters snow
        (["desert dunes"], "hot", "supports"),
        (["desert dunes"], "tropical", "contradicts"),  # humid tropics vs dunes
        (["bare deciduous trees"], "temperate", "supports"),
        (["bare deciduous trees"], "tropical", "inconclusive"),  # soft cue
        (["lush green vegetation"], "hot", "inconclusive"),      # soft miss
        (["lush"], "temperate", "supports"),
        (["glacier in the distance"], "hot", "contradicts"),
        ([], "temperate", "inconclusive"),
        (["a road", "some buildings"], "temperate", "inconclusive"),
    ]
    print("climate cue-logic self-test (offline):")
    ok = True
    for cues, hint_key, want in cases:
        ev = climate_evidence(0.0, 0.0, cues, normals=canned[hint_key])
        got = ev.verdict.value
        flag = "ok" if (got == want and ev.confidence <= _CONF_CAP) \
            else "MISMATCH"
        ok &= flag == "ok"
        print(f"  {str(cues):42} vs {hint_key:9} -> {got} "
              f"conf={ev.confidence} [{flag}]")
    # never-raises check — bad coords fail before any HTTP
    bad = climate_normals("not-a-lat", 0)
    print(f"  climate_normals('not-a-lat', 0) -> {bad} "
          f"[{'ok' if bad == {} else 'MISMATCH'}]")
    ok &= bad == {}
    if not ok:
        sys.exit(1)

    if os.environ.get("SUNDIAL_NET") == "1":
        print("\nlive lookup (SUNDIAL_NET=1):")
        print("  London:", climate_normals(51.5074, -0.1278) or "(fetch failed)")
        print("  Singapore:", climate_normals(1.3521, 103.8198)
              or "(fetch failed)")
        ev = climate_evidence(51.5074, -0.1278, ["palm trees", "tropical"])
        print(f"  palms-in-London: {ev.verdict.value} conf={ev.confidence} "
              f"— {ev.summary}")
    else:
        print("\n(offline — set SUNDIAL_NET=1 for a live Open-Meteo lookup)")
