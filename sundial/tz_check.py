"""Timezone sanity check — does the claimed local time fit the claimed place?

Two independent timezone signals, no network:

* ``expected_tz_name`` — a coarse offline lat/lon -> IANA zone lookup built
  from hand-drawn bounding boxes for the demo countries plus a few dozen
  common ones. Rectangles are approximate: near a border a point can land
  on a neighbouring zone, and whole territories not in the table (oceans,
  small island states, most of Africa/central Asia) return None. It's a
  smoke test, not a gazetteer.

* ``arch_timezone`` — the IANA name Open-Meteo resolves for the resolved
  coordinates (``weather.historical_weather(... )["timezone"]``). It knows
  the real tz database, so when it exists it WINS: the offline table only
  corroborates it. A name mismatch means either a border-straddling
  rectangle or — the interesting case — coordinates that geocoded to the
  wrong country entirely ("3pm London" resolving to Japan).

zoneinfo is imported lazily; platforms without tzdata degrade to a
None offset rather than an ImportError. The schema import is also lazy so
the module loads with zero package deps (same convention as moonphase).
"""
from __future__ import annotations

from datetime import datetime, timezone

# Diagnostic context only — the adjudicator weighs real evidence; a
# timezone disagreement is a hint that the coordinates may be off, so all
# confidences stay under this cap.
CONF_CAP = 0.3

# (lat_lo, lat_hi, lon_lo, lon_hi, IANA zone). First match wins: small
# countries are listed before the big neighbours their boxes overlap, and
# multi-zone countries appear as longitude bands. Border errors are
# expected and usually land on a same-offset zone anyway.
_BOXES = [
    # --- Europe: small states first -------------------------------------
    (51.3, 55.5, -10.7, -5.9, "Europe/Dublin"),       # IE (before GB)
    (36.8, 42.2, -9.6, -6.1, "Europe/Lisbon"),        # PT mainland
    (54.2, 55.1, 19.8, 23.0, "Europe/Kaliningrad"),   # RU exclave (before PL/LT)
    (49.4, 50.2, 5.7, 6.6, "Europe/Luxembourg"),      # LU
    (50.7, 53.7, 3.2, 7.3, "Europe/Amsterdam"),       # NL
    (49.4, 51.6, 2.5, 6.5, "Europe/Brussels"),        # BE
    (45.7, 47.9, 5.9, 10.6, "Europe/Zurich"),         # CH (+LI)
    (46.3, 49.1, 9.5, 17.2, "Europe/Vienna"),         # AT
    (47.6, 49.7, 16.8, 22.6, "Europe/Bratislava"),    # SK
    (45.4, 46.9, 13.3, 16.7, "Europe/Ljubljana"),     # SI
    (45.7, 48.7, 16.1, 23.0, "Europe/Budapest"),      # HU
    (42.3, 46.6, 13.4, 19.5, "Europe/Zagreb"),        # HR
    (48.5, 51.1, 12.0, 18.9, "Europe/Prague"),        # CZ
    (54.3, 57.9, 7.9, 12.9, "Europe/Copenhagen"),     # DK mainland
    (54.9, 55.4, 14.6, 15.3, "Europe/Copenhagen"),    # DK Bornholm
    (57.5, 59.7, 21.7, 28.3, "Europe/Tallinn"),       # EE
    (55.6, 58.1, 20.9, 28.3, "Europe/Riga"),          # LV
    (53.8, 56.5, 20.9, 26.9, "Europe/Vilnius"),       # LT
    (49.0, 54.9, 14.0, 24.2, "Europe/Warsaw"),        # PL
    (43.6, 48.4, 20.2, 29.8, "Europe/Bucharest"),     # RO
    (41.2, 44.3, 22.3, 28.7, "Europe/Sofia"),         # BG
    (34.8, 41.8, 19.3, 29.8, "Europe/Athens"),        # GR
    (59.7, 70.2, 19.3, 31.6, "Europe/Helsinki"),      # FI
    (55.3, 69.1, 10.9, 24.2, "Europe/Stockholm"),     # SE
    (57.9, 63.5, 4.4, 13.0, "Europe/Oslo"),           # NO south
    (63.5, 71.3, 4.4, 31.2, "Europe/Oslo"),           # NO north
    (51.2, 56.2, 23.1, 32.8, "Europe/Minsk"),         # BY
    (44.3, 52.4, 22.0, 40.3, "Europe/Kyiv"),          # UA
    (35.8, 42.2, 25.6, 44.9, "Europe/Istanbul"),      # TR
    (47.2, 55.1, 5.8, 15.1, "Europe/Berlin"),         # DE
    (42.3, 51.1, -5.2, 9.7, "Europe/Paris"),          # FR metro (+Corsica)
    (49.9, 60.9, -8.7, 2.1, "Europe/London"),         # GB
    (35.9, 43.8, -9.4, 4.4, "Europe/Madrid"),         # ES mainland
    (27.5, 29.6, -18.3, -13.3, "Atlantic/Canary"),    # ES Canaries
    (36.6, 47.1, 6.6, 18.6, "Europe/Rome"),           # IT (+SM/VA)
    (63.2, 66.6, -24.6, -13.4, "Atlantic/Reykjavik"), # IS

    # --- Asia: small/neighbouring zones before the RU/CN giants ---------
    (40.5, 55.5, 46.5, 55.5, "Asia/Aqtobe"),          # KZ west
    (40.5, 55.5, 55.5, 87.3, "Asia/Almaty"),          # KZ east
    (41.5, 52.2, 87.7, 120.0, "Asia/Ulaanbaatar"),    # MN
    (26.3, 30.5, 80.0, 88.3, "Asia/Kathmandu"),       # NP (UTC+5:45)
    (5.9, 10.0, 79.6, 82.0, "Asia/Colombo"),          # LK
    (20.5, 26.7, 88.0, 92.7, "Asia/Dhaka"),           # BD
    (5.5, 20.5, 97.3, 105.8, "Asia/Bangkok"),         # TH
    (9.7, 28.6, 92.1, 101.2, "Asia/Yangon"),          # MM (UTC+6:30)
    (8.3, 23.5, 102.0, 109.7, "Asia/Ho_Chi_Minh"),    # VN
    (1.2, 1.5, 103.6, 104.1, "Asia/Singapore"),       # SG
    (1.0, 7.4, 99.6, 119.4, "Asia/Kuala_Lumpur"),     # MY
    (4.5, 21.3, 116.9, 126.7, "Asia/Manila"),         # PH
    (-11.0, 6.0, 94.0, 112.0, "Asia/Jakarta"),        # ID west (UTC+7)
    (-11.0, 6.0, 112.0, 130.0, "Asia/Makassar"),      # ID central (+8)
    (-11.0, 6.0, 130.0, 141.1, "Asia/Jayapura"),      # ID east (+9)
    (21.8, 25.4, 119.3, 122.1, "Asia/Taipei"),        # TW
    (22.1, 22.6, 113.8, 114.5, "Asia/Hong_Kong"),     # HK
    (33.0, 38.7, 124.6, 129.7, "Asia/Seoul"),         # KR
    (24.0, 46.0, 122.7, 146.0, "Asia/Tokyo"),         # JP
    (18.0, 53.6, 73.5, 135.1, "Asia/Shanghai"),       # CN (single zone)
    (6.7, 35.7, 68.1, 89.7, "Asia/Kolkata"),          # IN
    (23.6, 37.1, 60.8, 77.9, "Asia/Karachi"),         # PK
    (24.9, 39.8, 44.0, 63.4, "Asia/Tehran"),          # IR
    (29.4, 33.4, 34.2, 35.9, "Asia/Jerusalem"),       # IL
    (16.0, 32.2, 34.4, 55.7, "Asia/Riyadh"),          # SA
    (22.5, 26.2, 51.4, 56.5, "Asia/Dubai"),           # AE

    # --- Russia by longitude band (after its smaller neighbours) --------
    (64.0, 70.0, -180.0, -157.5, "Asia/Anadyr"),      # Chukotka across the antimeridian
    (41.0, 78.0, 30.0, 42.0, "Europe/Moscow"),        # UTC+3
    (41.0, 78.0, 42.0, 58.0, "Europe/Samara"),        # +4
    (41.0, 78.0, 58.0, 70.0, "Asia/Yekaterinburg"),   # +5
    (41.0, 78.0, 70.0, 80.0, "Asia/Omsk"),            # +6
    (41.0, 78.0, 80.0, 95.0, "Asia/Krasnoyarsk"),     # +7
    (41.0, 78.0, 95.0, 112.0, "Asia/Irkutsk"),        # +8
    (41.0, 78.0, 112.0, 130.0, "Asia/Yakutsk"),       # +9
    (41.0, 78.0, 130.0, 145.0, "Asia/Vladivostok"),   # +10
    (41.0, 78.0, 145.0, 160.0, "Asia/Magadan"),       # +11
    (41.0, 78.0, 160.0, 180.0, "Asia/Kamchatka"),     # +12

    # --- Americas --------------------------------------------------------
    (51.0, 71.5, -180.0, -130.0, "America/Anchorage"),    # AK
    (18.8, 22.3, -160.3, -154.7, "Pacific/Honolulu"),     # HI
    (24.4, 49.4, -125.0, -115.0, "America/Los_Angeles"),  # US Pacific
    (24.4, 49.4, -115.0, -104.0, "America/Denver"),       # US Mountain
    (24.4, 49.4, -104.0, -88.0, "America/Chicago"),       # US Central
    (24.4, 49.4, -88.0, -66.9, "America/New_York"),       # US Eastern
    (14.5, 32.6, -117.1, -112.0, "America/Tijuana"),      # MX Baja
    (14.5, 32.6, -112.0, -102.0, "America/Mazatlan"),     # MX west
    (14.5, 32.6, -102.0, -90.0, "America/Mexico_City"),   # MX central
    (14.5, 32.6, -90.0, -86.7, "America/Cancun"),         # MX Quintana Roo
    (49.4, 70.0, -141.0, -114.0, "America/Vancouver"),    # CA Pacific
    (49.4, 70.0, -114.0, -102.0, "America/Edmonton"),     # CA Mountain
    (49.4, 70.0, -102.0, -89.0, "America/Winnipeg"),      # CA Central
    (49.4, 70.0, -89.0, -67.0, "America/Toronto"),        # CA Eastern
    (43.5, 61.0, -67.0, -59.0, "America/Halifax"),        # CA Atlantic
    (46.0, 61.0, -59.0, -52.0, "America/St_Johns"),       # CA Newfoundland
    (13.7, 17.9, -92.3, -88.2, "America/Guatemala"),      # GT (+BZ/SV roughly)
    (8.0, 11.3, -85.9, -82.5, "America/Costa_Rica"),      # CR
    (7.2, 9.7, -83.1, -77.1, "America/Panama"),           # PA
    (19.8, 23.4, -85.0, -74.0, "America/Havana"),         # CU
    (17.6, 18.6, -78.4, -76.1, "America/Jamaica"),        # JM
    (17.7, 20.0, -74.5, -67.9, "America/Santo_Domingo"),  # Hispaniola (HT/DO coarse)
    (17.8, 18.6, -67.3, -65.5, "America/Puerto_Rico"),    # PR
    (-4.3, 12.6, -79.1, -66.8, "America/Bogota"),         # CO
    (0.6, 12.2, -73.4, -59.7, "America/Caracas"),         # VE
    (-5.1, 1.6, -81.1, -75.1, "America/Guayaquil"),       # EC mainland
    (-18.4, 0.1, -81.4, -68.6, "America/Lima"),           # PE
    (-23.0, -9.7, -69.7, -57.4, "America/La_Paz"),        # BO
    (-27.7, -19.3, -62.7, -54.2, "America/Asuncion"),     # PY
    (-56.0, -17.4, -75.8, -66.3, "America/Santiago"),     # CL mainland
    (-55.1, -21.7, -73.7, -53.5, "America/Argentina/Buenos_Aires"),  # AR
    (-35.1, -30.0, -58.5, -53.0, "America/Montevideo"),   # UY
    (-33.8, 5.4, -74.0, -67.0, "America/Rio_Branco"),     # BR Acre
    (-33.8, 5.4, -67.0, -57.0, "America/Manaus"),         # BR Amazon band
    (-33.8, 5.4, -57.0, -34.7, "America/Sao_Paulo"),      # BR populated east

    # --- Africa ----------------------------------------------------------
    (30.2, 37.6, 7.4, 11.7, "Africa/Tunis"),          # TN
    (4.7, 11.2, -3.3, 1.3, "Africa/Accra"),           # GH
    (27.6, 35.9, -13.3, -0.9, "Africa/Casablanca"),   # MA
    (18.9, 37.1, -8.7, 12.0, "Africa/Algiers"),       # DZ
    (22.0, 31.7, 24.6, 36.9, "Africa/Cairo"),         # EG
    (4.2, 13.9, 2.6, 14.7, "Africa/Lagos"),           # NG
    (-4.8, 5.1, 33.9, 42.0, "Africa/Nairobi"),        # KE
    (3.3, 14.9, 32.9, 48.0, "Africa/Addis_Ababa"),    # ET
    (-34.9, -22.1, 16.3, 33.0, "Africa/Johannesburg"),# ZA

    # --- Oceania ----------------------------------------------------------
    (-11.7, -0.8, 140.8, 156.1, "Pacific/Port_Moresby"),  # PG (before AU east band)
    (-44.0, -10.0, 112.0, 129.0, "Australia/Perth"),      # AU west
    (-44.0, -10.0, 129.0, 141.0, "Australia/Adelaide"),   # AU central (NT coarse)
    (-44.0, -10.0, 141.0, 154.0, "Australia/Sydney"),     # AU east
    (-47.4, -34.3, 166.3, 178.7, "Pacific/Auckland"),     # NZ
]


def expected_tz_name(lat: float, lon: float) -> str | None:
    """Coarse offline lat/lon -> IANA zone guess; None when not covered.

    Rectangle matching only — no coastlines, no exclaves beyond the ones
    listed, no ocean coverage. Treat it as "a plausible zone for this
    point", never as truth.
    """
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return None
    for lat_lo, lat_hi, lon_lo, lon_hi, zone in _BOXES:
        if lat_lo <= la <= lat_hi and lon_lo <= lo <= lon_hi:
            return zone
    return None


def claim_utc_offset(claim_time_iso: str, tz_name: str) -> float | None:
    """UTC offset in hours that ``tz_name`` keeps at the claimed instant.

    The naive claim time is interpreted in the zone (fold=0 for the
    impossible/ambiguous instants around DST switches — fine for a coarse
    check). Returns None on any failure: missing tzdata, unknown zone
    name, unparseable timestamp.
    """
    if not claim_time_iso or not tz_name:
        return None
    try:
        from zoneinfo import ZoneInfo  # lazy: tzdata-free platforms may lack it
        naive = datetime.fromisoformat(str(claim_time_iso).strip()).replace(tzinfo=None)
        off = naive.replace(tzinfo=ZoneInfo(str(tz_name))).utcoffset()
        return None if off is None else off.total_seconds() / 3600.0
    except Exception:
        return None


def _claimed_utc_via(tz_name: str, claim_time_iso: str) -> datetime | None:
    """Naive claim local time -> aware UTC instant via the given zone."""
    try:
        from zoneinfo import ZoneInfo
        naive = datetime.fromisoformat(str(claim_time_iso).strip()).replace(tzinfo=None)
        return naive.replace(tzinfo=ZoneInfo(tz_name)).astimezone(timezone.utc)
    except Exception:
        return None


def tz_evidence(
    lat: float | None,
    lon: float | None,
    arch_timezone: str | None,
    claim_time_iso: str | None,
) -> "Evidence":
    """Diagnostic card: does the archive tz agree with where the coords look like they are?

    ``arch_timezone`` (from the Open-Meteo archive response) is the
    stronger signal — it reflects the real tz database at the resolved
    coordinates, so it wins over the offline table. A name disagreement
    between the two is flagged INCONCLUSIVE: either a coarse-rectangle
    border artifact, or the coordinates geocoded to a different country
    than the claim implies — worth the adjudicator knowing, not a
    verdict. An agreement weakly SUPPORTS the place claim at the cap.
    """
    from .schema import ClaimVerdict, Evidence  # lazy: zero-dep module import

    expected = None
    if lat is not None and lon is not None:
        expected = expected_tz_name(lat, lon)
    arch_tz = (str(arch_timezone).strip() if arch_timezone else "") or None

    data: dict = {
        "expected_tz": expected,
        "archive_tz": arch_tz,
        "method": "offline lat/lon box table vs Open-Meteo archive timezone (zoneinfo for offsets)",
    }
    if lon is not None:
        try:
            data["lmt_offset_h"] = round(float(lon) / 15.0, 2)  # lon/15 crude solar-time est.
        except (TypeError, ValueError):
            pass

    # Convert the claimed local time through the archive zone (the zone
    # that actually governs the resolved coords) and see whether the UTC
    # instant falls on a different civil date — that makes the claim's
    # *date* ambiguous too, which is worth noting.
    if arch_tz and claim_time_iso:
        utc = _claimed_utc_via(arch_tz, claim_time_iso)
        naive_date = None
        try:
            naive_date = datetime.fromisoformat(
                str(claim_time_iso).strip()).replace(tzinfo=None).date()
        except ValueError:
            data["claim_time_parse_error"] = True
        if utc is not None:
            data["claim_local"] = str(claim_time_iso).strip()
            data["claim_utc_via_archive"] = utc.isoformat()
            if naive_date is not None:
                data["date_shift"] = utc.date() != naive_date

    exp_off = claim_utc_offset(claim_time_iso, expected) if expected else None
    arch_off = claim_utc_offset(claim_time_iso, arch_tz) if arch_tz else None
    if exp_off is not None:
        data["expected_offset_h"] = exp_off
    if arch_off is not None:
        data["archive_offset_h"] = arch_off

    date_note = ""
    if data.get("date_shift"):
        date_note = (
            f"; claim local {data['claim_local']} converts to UTC "
            f"{utc.isoformat()} — crosses a civil-date boundary"
        )

    if arch_tz and expected:
        if expected == arch_tz:
            return Evidence(
                type="timezone",
                summary=(f"archive timezone {arch_tz!r} agrees with the "
                         f"coarse lat/lon table{date_note}"),
                verdict=ClaimVerdict.SUPPORTS,
                confidence=CONF_CAP,
                data=data,
            )
        if exp_off is not None and arch_off is not None and exp_off == arch_off:
            return Evidence(
                type="timezone",
                summary=(f"archive timezone {arch_tz!r} vs table {expected!r} — "
                         f"different zone but same offset ({arch_off:+.1f}h) at "
                         f"the claim instant; likely a border/coarse-table "
                         f"artifact{date_note}"),
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.25,
                data=data,
            )
        return Evidence(
            type="timezone",
            summary=(f"oddity: archive timezone {arch_tz!r} disagrees with "
                     f"the lat/lon table's {expected!r} — coordinates may "
                     f"resolve to a different country than expected{date_note}"),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=CONF_CAP,
            data=data,
        )

    if arch_tz:
        return Evidence(
            type="timezone",
            summary=(f"archive timezone is {arch_tz!r}; coordinates outside "
                     f"the offline table — no independent check{date_note}"),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )
    if expected:
        return Evidence(
            type="timezone",
            summary=(f"no archive timezone; the coarse lat/lon table "
                     f"suggests {expected!r} (offline estimate only){date_note}"),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )
    return Evidence(
        type="timezone",
        summary="no timezone check possible (no coordinates, no archive timezone)",
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.1,
        data=data,
    )


if __name__ == "__main__":
    import sys

    # London coords -> Europe/London is the headline self-test.
    checks = [
        ((51.5074, -0.1278), "Europe/London"),        # London
        ((48.8566, 2.3522), "Europe/Paris"),          # Paris
        ((52.52, 13.405), "Europe/Berlin"),           # Berlin
        ((35.68, 139.69), "Asia/Tokyo"),              # Tokyo
        ((40.71, -74.0), "America/New_York"),         # NYC
        ((34.05, -118.24), "America/Los_Angeles"),    # LA
        ((-33.87, 151.21), "Australia/Sydney"),       # Sydney
        ((-31.95, 115.86), "Australia/Perth"),        # Perth
        ((55.75, 37.62), "Europe/Moscow"),            # Moscow
        ((-23.55, -46.63), "America/Sao_Paulo"),      # Sao Paulo
        ((0.0, 0.0), None),                           # Gulf of Guinea — ocean, uncovered
        ((19.0, -155.6), "Pacific/Honolulu"),         # Hawaii
    ]
    ok = True
    for (la, lo), want in checks:
        got = expected_tz_name(la, lo)
        ok &= got == want
        flag = "ok" if got == want else "MISMATCH"
        print(f"  expected_tz_name({la}, {lo}) -> {got} (want {want}) [{flag}]")

    # Offsets need zoneinfo; skip gracefully on tzdata-free platforms.
    off = claim_utc_offset("2024-06-15T15:00", "Europe/London")
    if off is not None:
        ok &= off == 1.0  # BST
        print(f"  London offset 2024-06-15 -> {off} (want 1.0 BST)")
        off = claim_utc_offset("2024-01-15T15:00", "Europe/London")
        ok &= off == 0.0  # GMT
        print(f"  London offset 2024-01-15 -> {off} (want 0.0 GMT)")
    else:
        print("  zoneinfo unavailable (no tzdata) — offset checks skipped")

    # Evidence card demo — needs the package context for the lazy
    # `from .schema import`; run as `python -m sundial.tz_check`.
    try:
        ev = tz_evidence(51.5074, -0.1278, "Europe/London", "2024-06-15T15:00")
        print(f"  tz_evidence London match -> {ev.verdict.value} @ {ev.confidence}")
        ev = tz_evidence(51.5074, -0.1278, "Asia/Tokyo", "2024-06-15T15:00")
        ok &= ev.verdict.value == "inconclusive"
        print(f"  tz_evidence London coords / Tokyo archive -> "
              f"{ev.verdict.value} @ {ev.confidence}: {ev.summary}")
    except ImportError as e:
        print(f"  tz_evidence demo skipped ({e}); run as `python -m sundial.tz_check`")

    if not ok:
        sys.exit(1)
