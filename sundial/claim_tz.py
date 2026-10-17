"""Claim-time resolver — "12:30 local" -> a trustworthy UTC instant.

The pipeline hands us a *naive* local time plus wherever the claim
geocoded to. This module anchors that naive time in a zone and produces
the UTC instant, the offset applied, and a plain-English note about how
shaky the conversion was. It never raises: every failure degrades to a
UTC guess with ``ambiguous`` set or a None field.

Anchor resolution order for a naive claim time:

* explicit ``tz_name`` argument (e.g. the Open-Meteo archive timezone —
  it reflects the real tz database at the resolved coords, so it wins);
* ``tz_at(lat, lon)`` — prefers ``tz_check.expected_tz_name`` (the
  checked demo table) when the package import succeeds, else this
  file's own ``TZ_TABLE`` fallback so the module still works run
  standalone (``python3 sundial/claim_tz.py``);
* bare "UTC" with ``ambiguous=True`` when nothing could be determined.

DST edges (zoneinfo, PEP 495): around the autumn switch a naive local
time occurs *twice*; around the spring switch it doesn't exist at all.
``fold=0`` is used for the conversion either way (same convention as
``tz_check._claimed_utc_via``); when fold=0/1 disagree the alternative
instant is reported in ``note`` — the ``ambiguous`` flag is reserved
for "no timezone anchor", which is the unrecoverable case.

Stdlib only at import (datetime + zoneinfo with a fallback for
tzdata-free platforms); the only package import is lazy inside tz_at.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:  # tzdata-free platforms can lack the module or every key inside it
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - exercised only on stripped pythons
    ZoneInfo = None

__all__ = ["TZ_TABLE", "tz_at", "to_utc", "utc_to_local"]

# Coarse offline fallback: (lat_lo, lat_hi, lon_lo, lon_hi) -> IANA zone.
# Same semantics as tz_check._BOXES — first match wins, so small or
# same-offset-critical zones sit before the broad bands they overlap,
# and big multi-zone countries are longitude bands. ~30 rectangles, one
# pass per continent; oceans, small islands and central Asia are thin —
# misses degrade to "UTC" + ambiguous, never to a wrong-but-confident
# answer. Border errors are expected and usually land within 1h.
TZ_TABLE = {
    # Europe — outliers (+0, +3) first, EET band, then the big CET band
    (49.9, 60.9, -10.7, 2.2): "Europe/London",        # GB+IE
    (36.8, 42.3, -9.7, -6.1): "Europe/Lisbon",        # PT (kept: +0 vs CET)
    (35.8, 42.3, 25.6, 45.0): "Europe/Istanbul",      # TR (+3, no DST)
    (34.0, 70.5, 21.5, 31.5): "Europe/Helsinki",      # EET band (GR/BG/RO/FI/Baltics/UA-coarse)
    (35.0, 71.5, -10.0, 25.7): "Europe/Berlin",       # CET band — rest of Europe; IS uncovered -> UTC anyway

    # North America — HI, then the four big bands (AK folded into Pacific, Nfld/Atl err <=1.5h)
    (18.0, 23.0, -161.0, -154.0): "Pacific/Honolulu",
    (24.0, 72.0, -85.0, -52.0): "America/New_York",    # eastern + Atlantic CA coarse
    (24.0, 72.0, -105.0, -85.0): "America/Chicago",
    (24.0, 72.0, -116.0, -105.0): "America/Denver",
    (24.0, 72.0, -170.0, -116.0): "America/Los_Angeles",  # Pacific + AK coarse
    (7.0, 24.0, -118.0, -59.0): "America/Mexico_City",    # MX/C.Am/Caribbean coarse

    # South America — thin Chile strip first, +5 west, then the -3 east
    (-56.0, -17.0, -76.0, -70.0): "America/Santiago",   # CL strip (before AR eats its north)
    (-12.0, 13.0, -83.5, -66.5): "America/Bogota",      # CO/PE/EC/VE-coarse (+5/-4 err)
    (-55.5, 7.0, -74.0, -34.5): "America/Argentina/Buenos_Aires",  # AR/UY/BR (-3 majority)

    # Africa — east +3, west/central +1 (split so the Gulf of Guinea stays
    # uncovered: (0,0) is ocean), Maghreb, then south+Egypt +2
    (-12.0, 18.0, 28.0, 52.0): "Africa/Nairobi",
    (19.0, 37.5, -17.5, 12.0): "Africa/Casablanca",     # Maghreb (+0/+1)
    (4.0, 21.0, -17.5, 28.0): "Africa/Lagos",           # west/central, N of equator
    (-19.0, 4.5, 8.0, 28.0): "Africa/Lagos",            # Gabon/Congo/west DRC/Angola
    (-35.5, 32.0, 11.5, 37.0): "Africa/Johannesburg",   # south + EG (+2)

    # Middle East / South Asia — Iraq before Iran (shared sliver), Iran before Arabia
    (29.0, 38.0, 38.5, 48.5): "Asia/Baghdad",
    (24.5, 40.0, 43.5, 63.5): "Asia/Tehran",            # IR (+3:30)
    (12.0, 33.0, 38.0, 63.5): "Asia/Riyadh",            # peninsula (+3)
    (6.5, 36.0, 68.0, 97.5): "Asia/Kolkata",            # IN (+5:30; PK/NP/BD +-45min err)

    # East/SE Asia — KR/JP and mainland SEA before the single-zone CN giant
    (33.0, 39.0, 124.0, 131.0): "Asia/Seoul",
    (24.0, 46.0, 127.0, 146.5): "Asia/Tokyo",
    (5.0, 28.7, 92.0, 110.0): "Asia/Bangkok",           # TH/VN/LA/KH/MM (+7; MM +-30min)
    (4.5, 21.5, 110.0, 127.0): "Asia/Manila",           # PH (+8)
    (17.5, 53.7, 73.0, 135.5): "Asia/Shanghai",         # CN+MN single zone (before Jakarta overlap)
    (-11.5, 17.5, 92.0, 141.5): "Asia/Jakarta",         # ID/MY/SG coarse (+7; east ID err)

    # Russia by longitude band (after CN so Mongolia/north China stay +8)
    (41.0, 78.0, 31.5, 60.0): "Europe/Moscow",
    (41.0, 78.0, 60.0, 105.0): "Asia/Krasnoyarsk",      # +7 (+5..+8 edges err)
    (41.0, 78.0, 105.0, 180.0): "Asia/Yakutsk",         # +9 (+10..+12 edges err)

    # Oceania
    (-45.0, -10.0, 112.0, 129.0): "Australia/Perth",
    (-45.0, -10.0, 129.0, 155.0): "Australia/Sydney",   # central AU err <=30min
    (-48.0, -34.0, 165.0, 180.0): "Pacific/Auckland",
}


def _zone(name) -> "ZoneInfo | None":
    """ZoneInfo(name) or None — bad names and missing tzdata both degrade."""
    if ZoneInfo is None or not name:
        return None
    try:
        return ZoneInfo(str(name))
    except Exception:
        return None


def tz_at(lat: float, lon: float) -> str:
    """Best-guess IANA zone for a point; "UTC" when nothing matches.

    Prefers ``tz_check.expected_tz_name`` (the checked table) via a lazy
    relative import — run inside the package it wins; run standalone or
    without the package context this file's TZ_TABLE takes over. Either
    way the result is a guess: coarse rectangles, first match wins.
    """
    name = None
    try:
        from .tz_check import expected_tz_name  # lazy: package may be absent
    except Exception:
        expected_tz_name = None
    if expected_tz_name is not None:
        try:
            name = expected_tz_name(float(lat), float(lon))
        except (TypeError, ValueError):
            name = None
        except Exception:
            name = None
    if name:
        return str(name)

    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return "UTC"
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return "UTC"
    for (lat_lo, lat_hi, lon_lo, lon_hi), zone in TZ_TABLE.items():
        if lat_lo <= la <= lat_hi and lon_lo <= lo <= lon_hi:
            return zone
    return "UTC"


def to_utc(
    claim_time_iso: str,
    lat: float | None,
    lon: float | None,
    tz_name: str | None = None,
) -> dict:
    """Resolve a claimed local time to UTC.

    Naive ISO -> interpreted in the anchor zone via zoneinfo (fold=0 for
    DST edges, with the alternative instant reported in ``note``).
    Aware ISO -> passed through to UTC unmodified. ``ambiguous`` is set
    only when a naive time had no trustworthy anchor (no tz_name, no
    coords / table miss) and was read as UTC as a last resort.

    Returns ``{"utc", "tz", "offset_min", "ambiguous", "note"}`` — fields
    are None when the claim time itself couldn't be parsed. Never raises.
    """
    res: dict = {"utc": None, "tz": None, "offset_min": None,
                 "ambiguous": False, "note": ""}
    raw = str(claim_time_iso).strip() if claim_time_iso is not None else ""
    if not raw:
        res["ambiguous"] = True
        res["note"] = "empty claim time — nothing to resolve"
        return res
    try:
        dt = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        res["ambiguous"] = True
        res["note"] = f"unparseable claim time {raw!r} (expected ISO-8601)"
        return res

    # --- aware input: the ISO carries its own offset, trust it ----------
    try:
        off_src = dt.utcoffset() if dt.tzinfo is not None else None
    except Exception:
        off_src = None
    if off_src is not None:
        try:
            utc_dt = dt.astimezone(timezone.utc)
            src = str(tz_name).strip() if tz_name else (dt.tzname() or str(dt.tzinfo))
            res["utc"] = utc_dt.isoformat()
            res["tz"] = src
            res["offset_min"] = int(off_src.total_seconds() // 60)
            res["note"] = ("aware input — offset carried by the ISO "
                           "timestamp, passed through to UTC")
        except Exception as e:
            res["ambiguous"] = True
            res["note"] = f"aware input but UTC conversion failed: {e}"
        return res

    # --- naive input: find an anchor zone --------------------------------
    naive = dt.replace(tzinfo=None)
    notes: list[str] = []
    zi = None
    anchor = None

    if tz_name:
        zi = _zone(tz_name)
        if zi is not None:
            anchor = str(tz_name).strip()
        else:
            notes.append(f"given tz {tz_name!r} not loadable")
    if zi is None and lat is not None and lon is not None:
        guess = tz_at(lat, lon)
        zi = _zone(guess)
        if zi is not None:
            anchor = guess
            if guess == "UTC":
                notes.append("coords outside every tz table — "
                             "defaulting to UTC")
            else:
                notes.append(f"anchored to {anchor} from coordinates "
                             f"({lat}, {lon})")
        else:
            notes.append(f"guessed zone {guess} unusable (no tzdata?)")

    if zi is None:
        # No anchor at all — read the naive time as UTC and say so.
        utc_dt = naive.replace(tzinfo=timezone.utc)
        res["utc"] = utc_dt.isoformat()
        res["tz"] = "UTC"
        res["offset_min"] = 0
        res["ambiguous"] = True
        notes.append("no timezone anchor — naive local time read as UTC")
        res["note"] = "; ".join(notes)
        return res

    # --- convert with fold=0; report the fold/gap alternative ------------
    dt0 = naive.replace(tzinfo=zi, fold=0)
    dt1 = naive.replace(tzinfo=zi, fold=1)
    off0 = dt0.utcoffset() or timedelta(0)
    off1 = dt1.utcoffset() or timedelta(0)
    utc_dt = dt0.astimezone(timezone.utc)
    if off0 != off1:
        alt = dt1.astimezone(timezone.utc)
        if off0 > off1:
            notes.append(
                f"DST fold: {naive.isoformat()} occurs twice in {anchor} "
                f"— used the first occurrence (fold=0, {off0}); the "
                f"second would be {alt.isoformat()}"
            )
        else:
            notes.append(
                f"DST gap: {naive.isoformat()} is skipped in {anchor} "
                f"— interpreted with the pre-transition offset ({off0}); "
                f"the fold=1 reading would be {alt.isoformat()}"
            )

    res["utc"] = utc_dt.isoformat()
    res["tz"] = anchor
    res["offset_min"] = int(off0.total_seconds() // 60)
    res["ambiguous"] = anchor == "UTC"  # UTC is the catch-all, not a determined zone
    notes.insert(0, f"naive local time interpreted in {anchor}")
    res["note"] = "; ".join(notes)
    return res


def utc_to_local(utc_iso: str, tz_name: str) -> dict:
    """Reverse of to_utc: render a UTC instant in a zone for reports.

    Naive input is treated as already-UTC. Returns the same shape plus
    ``local``: ``{"local", "utc", "tz", "offset_min", "ambiguous",
    "note"}``. An unloadable zone leaves the time in UTC with a note.
    """
    res: dict = {"local": None, "utc": None, "tz": None,
                 "offset_min": None, "ambiguous": False, "note": ""}
    raw = str(utc_iso).strip() if utc_iso is not None else ""
    if not raw:
        res["note"] = "empty timestamp — nothing to convert"
        return res
    try:
        dt = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        res["note"] = f"unparseable timestamp {raw!r} (expected ISO-8601)"
        return res

    notes: list[str] = []
    try:
        if dt.tzinfo is None or dt.utcoffset() is None:
            dt = dt.replace(tzinfo=timezone.utc)
            notes.append("naive input treated as UTC")
        utc_dt = dt.astimezone(timezone.utc)
    except Exception as e:
        res["note"] = f"timestamp not convertible to UTC: {e}"
        return res
    res["utc"] = utc_dt.isoformat()

    zi = _zone(tz_name)
    if zi is None:
        res["local"] = utc_dt.isoformat()
        res["tz"] = "UTC"
        res["offset_min"] = 0
        notes.append(f"zone {tz_name!r} unavailable — left as UTC")
        res["note"] = "; ".join(notes)
        return res

    loc = utc_dt.astimezone(zi)
    off = loc.utcoffset() or timedelta(0)
    res["local"] = loc.isoformat()
    res["tz"] = str(tz_name)
    res["offset_min"] = int(off.total_seconds() // 60)
    notes.append(f"UTC instant rendered in {tz_name}")
    res["note"] = "; ".join(notes)
    return res


if __name__ == "__main__":
    import sys

    ok = True

    # Headline: Flensburg coords + October claim -> CET band, CEST (+2h).
    r = to_utc("2026-10-17T12:30", 54.48, 9.05)
    print(f"  to_utc naive Berlin Oct -> {r}")
    ok &= r["utc"] == "2026-10-17T10:30:00+00:00" and r["offset_min"] == 120
    ok &= not r["ambiguous"]
    ok &= r["tz"] in ("Europe/Berlin", "Europe/Copenhagen")  # same-offset band

    # Same place in January -> CET (+1h): the DST half of the "+1/2h" rule.
    r = to_utc("2026-01-17T12:30", 54.48, 9.05)
    print(f"  to_utc naive Berlin Jan -> {r}")
    ok &= r["utc"] == "2026-01-17T11:30:00+00:00" and r["offset_min"] == 60

    # Aware input passes straight through; coords/tz_name don't matter.
    r = to_utc("2026-10-17T12:30:00+02:00", 54.48, 9.05)
    print(f"  to_utc aware passthrough -> {r}")
    ok &= r["utc"] == "2026-10-17T10:30:00+00:00" and not r["ambiguous"]

    # Naive + no coords and no tz_name -> UTC guess, flagged ambiguous.
    r = to_utc("2026-10-17T12:30", None, None)
    print(f"  to_utc no-anchor -> {r}")
    ok &= r["ambiguous"] and r["utc"] == "2026-10-17T12:30:00+00:00"

    # Explicit tz_name beats coordinates (archive tz is the strong signal).
    r = to_utc("2026-10-17T12:30", 54.48, 9.05, "America/New_York")
    print(f"  to_utc explicit tz -> {r}")
    ok &= r["tz"] == "America/New_York" and r["offset_min"] == -240  # EDT

    # DST edges around the 2026 EU switches (last Sundays of Mar/Oct).
    r = to_utc("2026-10-25T02:30", 54.48, 9.05)
    print(f"  to_utc DST fold -> {r['note']}")
    ok &= "fold" in r["note"] and r["utc"] == "2026-10-25T00:30:00+00:00"
    r = to_utc("2026-03-29T02:30", 54.48, 9.05)
    print(f"  to_utc DST gap  -> {r['note']}")
    ok &= "gap" in r["note"] and r["utc"] == "2026-03-29T01:30:00+00:00"

    # tz_at + utc_to_local round-trip for report rendering.
    z = tz_at(54.48, 9.05)
    print(f"  tz_at(54.48, 9.05) -> {z}")
    ok &= z in ("Europe/Berlin", "Europe/Copenhagen")
    r = utc_to_local("2026-10-17T10:30:00+00:00", "Europe/Berlin")
    print(f"  utc_to_local -> {r}")
    ok &= r["local"].startswith("2026-10-17T12:30") and r["offset_min"] == 120

    # Junk in, nothing raised, honest output out.
    r = to_utc("not-a-time", 0, 0)
    print(f"  to_utc junk -> {r}")
    ok &= r["utc"] is None and r["ambiguous"]
    ok &= tz_at(0, 0) == "UTC"  # Gulf of Guinea — no box claims the ocean

    if ok:
        print("claim_tz self-test OK")
    else:
        sys.exit(1)
