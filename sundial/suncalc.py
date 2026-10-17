"""Solar position engine — zero-dependency sun elevation/azimuth.

Implements the standard NOAA solar-position algorithm (the same algebra
the NOAA Solar Calculator spreadsheets publish): Julian century ->
geometric mean longitude/anomaly -> equation of centre -> apparent
ecliptic longitude -> obliquity -> declination + equation of time, then
an hour-angle -> elevation/azimuth transform in the local horizon
frame. Accuracy is within a few hundredths of a degree of pysolar
(which carries the same math) — far inside the ~0.5 deg the
shadow/lighting checks need.

Conventions (matching solar.py / pysolar so the engines are
interchangeable): elevation is degrees above the horizon — the
*apparent* elevation, with a low-altitude refraction correction applied
(Saemundsson's compact formula, the one Meeus cites: ~34 arcmin of lift
at the horizon) so sunrise/sunset and twilight edges land where an
observer would actually see them. Azimuth runs 0..360 clockwise from
true north (0=N, 90=E, 180=S, 270=W). Longitude is degrees east
(negative = west, e.g. London ~-0.13).

`twilight_state` bins elevation into day / civil / nautical / night
using the conventional boundaries: apparent horizon at -0.833 deg (34'
refraction + 16' solar semidiameter), civil dusk -6 deg, nautical dusk
-12 deg (astronomical dark at -18 deg is beyond what a VLM can grade —
sibling sun_times.py bins the same way).

Stdlib only (math + datetime). Nothing here raises: bad input degrades
to {"elevation_deg": None, "azimuth_deg": None}, NaN, or "unknown" so
callers can treat failures as inconclusive rather than killing the
pipeline. `python3 sundial/suncalc.py` runs an offline self-test.
"""
from __future__ import annotations

import math
from datetime import date as _date, datetime, time as _time, timedelta, timezone

__all__ = [
    "sun_pos",
    "declination",
    "equation_of_time",
    "solar_noon_utc",
    "is_day",
    "twilight_state",
]

# Twilight boundaries on *apparent* elevation (degrees). -0.833 is the
# conventional sunrise/sunset definition (34' refraction + 16'
# semidiameter below the geometric horizon).
DAY_ELEV_DEG = -0.833
CIVIL_ELEV_DEG = -6.0
NAUTICAL_ELEV_DEG = -12.0

_JD_UNIX_EPOCH = 2440587.5  # Julian day of 1970-01-01T00:00Z
_JD_J2000 = 2451545.0       # Julian day of 2000-01-01T12:00Z

_NAN = float("nan")


def _as_utc(dt) -> datetime:
    """Accept datetime / date / ISO string -> tz-aware UTC (naive = UTC)."""
    if isinstance(dt, datetime):
        d = dt
    elif isinstance(dt, _date):
        d = datetime.combine(dt, _time.min)
    else:
        s = str(dt).strip()
        if s.lower().endswith("z"):
            s = s[:-1] + "+00:00"  # fromisoformat <3.11 rejects "Z"
        d = datetime.fromisoformat(s)
    return (d.astimezone(timezone.utc) if d.tzinfo is not None
            else d.replace(tzinfo=timezone.utc))


def _looks_when(x) -> bool:
    """True if `x` smells like a datetime (datetime/date/ISO string).

    Used by sun_pos to disambiguate (when, lat, lon) from
    (lat, lon, when) argument order — numbers are always coordinates.
    """
    if isinstance(x, (datetime, _date)):
        return True
    if isinstance(x, str):
        try:
            _as_utc(x)
            return True
        except Exception:
            return False
    return False


def _sun_geom(w: datetime) -> dict:
    """Core ephemeris: solar declination (deg) + equation of time (min).

    NOAA algorithm — all angles in degrees until the trig calls.
    `w` must already be tz-aware UTC.
    """
    jd = w.timestamp() / 86400.0 + _JD_UNIX_EPOCH
    t = (jd - _JD_J2000) / 36525.0  # Julian centuries since J2000.0

    # Geometric mean longitude (deg) and mean anomaly (deg) of the sun.
    l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    mr = math.radians(m)

    # Earth orbit eccentricity (unitless).
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    # Equation of centre (deg): periodic correction from elliptical orbit.
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2.0 * mr) * (0.019993 - 0.000101 * t)
         + math.sin(3.0 * mr) * 0.000289)

    # Apparent ecliptic longitude (deg): true longitude corrected for
    # nutation (Omega = moon's ascending-node longitude) and aberration.
    omega = math.radians(125.04 - 1934.136 * t)
    lam = math.radians(l0 + c - 0.00569 - 0.00478 * math.sin(omega))

    # Obliquity of the ecliptic (deg): mean obliquity + nutation corr.
    eps0 = (23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059
            - t * 0.001813))) / 60.0) / 60.0)
    eps = math.radians(eps0 + 0.00256 * math.cos(omega))

    # Declination (deg): asin(sin(eps) * sin(lam)).
    decl_deg = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))

    # Equation of time (minutes of time): the classic NOAA expansion —
    # y = tan(eps/2)^2 gathers the obliquity terms.
    y = math.tan(eps / 2.0) ** 2
    l0r = math.radians(l0)
    eot_min = 4.0 * math.degrees(
        y * math.sin(2.0 * l0r)
        - 2.0 * ecc * math.sin(mr)
        + 4.0 * ecc * y * math.sin(mr) * math.cos(2.0 * l0r)
        - 0.5 * y * y * math.sin(4.0 * l0r)
        - 1.25 * ecc * ecc * math.sin(2.0 * mr)
    )
    return {"jd": jd, "decl_deg": decl_deg, "eot_min": eot_min}


def _refraction_deg(true_elev_deg: float) -> float:
    """Saemundsson refraction lift (deg) for a true elevation (deg).

    R = 1.02' / tan(h + 10.3/(h + 5.11)) — Meeus's compact form. Only
    meaningful near the horizon; skipped below -1 deg where the formula
    goes unstable (and the sun is out of sight anyway).
    """
    if true_elev_deg <= -1.0:
        return 0.0
    h = true_elev_deg + 10.3 / (true_elev_deg + 5.11)
    return 1.02 / math.tan(math.radians(h)) / 60.0


def _position(w: datetime, lat: float, lon: float) -> dict:
    """-> {"elevation_deg","azimuth_deg"} for aware-UTC w. Never raises
    on maths edge cases (poles, antipodal sun) — atan2/asin are clamped.
    """
    g = _sun_geom(w)
    decl = math.radians(g["decl_deg"])
    phi = math.radians(lat)

    # True solar time (minutes) then hour angle (deg, 0 at transit,
    # negative morning). Longitude is degrees east: a spot 15 deg east
    # transits an hour earlier in UTC.
    utc_min = (w.hour * 60.0 + w.minute + w.second / 60.0
               + w.microsecond / 6e7)
    tst = (utc_min + g["eot_min"] + 4.0 * lon) % 1440.0
    ha = math.radians(tst / 4.0 - 180.0)

    # Elevation: sin(h) = sin(phi)sin(decl) + cos(phi)cos(decl)cos(H).
    sin_h = (math.sin(phi) * math.sin(decl)
             + math.cos(phi) * math.cos(decl) * math.cos(ha))
    elev_true = math.degrees(math.asin(max(-1.0, min(1.0, sin_h))))

    # Azimuth clockwise from north (NOAA form; the +180 moves the
    # atan2 reference from south to north).
    az = (math.degrees(math.atan2(
        math.sin(ha),
        math.cos(ha) * math.sin(phi) - math.tan(decl) * math.cos(phi),
    )) + 180.0) % 360.0

    return {
        "elevation_deg": elev_true + _refraction_deg(elev_true),
        "azimuth_deg": az,
    }


def sun_pos(*args, **kwargs) -> dict:
    """-> {"elevation_deg","azimuth_deg"} for a place+time. Never raises.

    Accepts either argument order — sun_pos(when, lat, lon) (preferred)
    or sun_pos(lat, lon, when) — plus keyword form (when=, lat=, lon=).
    `when` may be a datetime, date, or ISO string; naive values are read
    as UTC. On any failure both fields are None so callers degrade to
    "unknown" rather than crashing.
    """
    try:
        when = kwargs.get("when")
        lat = kwargs.get("lat")
        lon = kwargs.get("lon")
        vals = list(args) + [None, None, None]
        a, b, c = vals[0], vals[1], vals[2]
        if when is None or lat is None or lon is None:
            if _looks_when(a) and not _looks_when(c):
                when = a if when is None else when
                lat = b if lat is None else lat
                lon = c if lon is None else lon
            elif _looks_when(c):
                when = c if when is None else when
                lat = a if lat is None else lat
                lon = b if lon is None else lon
            else:
                when = a if when is None else when
                lat = b if lat is None else lat
                lon = c if lon is None else lon
        return _position(_as_utc(when), float(lat), float(lon))
    except Exception:
        return {"elevation_deg": None, "azimuth_deg": None}


def declination(when) -> float:
    """Solar declination in degrees (+north). NaN on bad input."""
    try:
        return _sun_geom(_as_utc(when))["decl_deg"]
    except Exception:
        return _NAN


def equation_of_time(when) -> float:
    """Equation of time in minutes (apparent minus mean solar time).
    NaN on bad input."""
    try:
        return _sun_geom(_as_utc(when))["eot_min"]
    except Exception:
        return _NAN


def solar_noon_utc(lat, lon, date) -> str:
    """ISO timestamp of UTC solar transit at (lat, lon) on `date`.

    Solar noon is when the hour angle is zero, i.e. true solar time
    720 min => utc = 720 - eot - 4*lon. EoT drifts ~seconds across the
    day, so iterate on its own answer a few times to converge. Latitude
    is part of the signature for API symmetry but doesn't affect the
    transit time. Returns "" on bad input.
    """
    try:
        _ = float(lat)  # validated for parity, unused by the math
        lonf = float(lon)
        if isinstance(date, datetime):
            d = date.date()
        elif isinstance(date, _date):
            d = date
        else:
            d = _as_utc(date).date()
        midnight = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        noon_min = 720.0 - 4.0 * lonf
        for _i in range(3):
            guess = midnight + timedelta(minutes=noon_min)
            noon_min = 720.0 - 4.0 * lonf - _sun_geom(guess)["eot_min"]
        return (midnight + timedelta(minutes=noon_min)).isoformat()
    except Exception:
        return ""


def twilight_state(when, lat, lon) -> str:
    """-> 'day' | 'civil' | 'nautical' | 'night' ('unknown' on failure).

    Bins apparent elevation against DAY_ELEV_DEG (-0.833 deg, the
    refraction+semidiameter sunrise definition), CIVIL_ELEV_DEG (-6)
    and NAUTICAL_ELEV_DEG (-12).
    """
    try:
        elev = _position(_as_utc(when), float(lat), float(lon))["elevation_deg"]
        if elev is None:
            return "unknown"
        if elev > DAY_ELEV_DEG:
            return "day"
        if elev >= CIVIL_ELEV_DEG:
            return "civil"
        if elev >= NAUTICAL_ELEV_DEG:
            return "nautical"
        return "night"
    except Exception:
        return "unknown"


def is_day(when, lat, lon) -> bool:
    """True when the sun is above the apparent horizon (-0.833 deg)."""
    try:
        return twilight_state(when, lat, lon) == "day"
    except Exception:
        return False


if __name__ == "__main__":
    import sys

    print("suncalc self-test")
    ok = True

    def check(name, cond, detail=""):
        global ok
        ok = ok and bool(cond)
        print(f"  {name}: {detail} [{'ok' if cond else 'MISMATCH'}]")

    def az_err(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    LONDON = (51.5074, -0.1278)
    TOKYO = (35.6762, 139.6503)
    SYDNEY = (-33.8688, 151.2093)

    # --- known positions, hardcoded expectations (verified ephemeris) ---
    p = sun_pos("2025-06-15T12:00:00Z", *LONDON)
    check("london noon elev", p["elevation_deg"] is not None
          and abs(p["elevation_deg"] - 61.8) <= 1.0,
          f"{p['elevation_deg']:.2f} (want ~61.8)")
    check("london noon az", p["azimuth_deg"] is not None
          and az_err(p["azimuth_deg"], 179.5) <= 1.0,
          f"{p['azimuth_deg']:.2f} (want ~179.5)")
    check("london noon twilight", twilight_state("2025-06-15T12:00:00Z", *LONDON) == "day",
          twilight_state("2025-06-15T12:00:00Z", *LONDON))

    p = sun_pos("2025-06-15T00:00:00Z", *LONDON)
    check("london midnight below horizon", p["elevation_deg"] is not None
          and p["elevation_deg"] < 0.0,
          f"{p['elevation_deg']:.2f} (want <0)")
    check("london midnight state", twilight_state("2025-06-15T00:00:00Z", *LONDON) == "night",
          twilight_state("2025-06-15T00:00:00Z", *LONDON))
    check("london midnight is_day", is_day("2025-06-15T00:00:00Z", *LONDON) is False, "False")

    # Tokyo: 12:00 JST = 03:00Z, just past solar transit -> high sun.
    p = sun_pos("2025-06-15T03:00:00Z", *TOKYO)
    check("tokyo noon elev", p["elevation_deg"] is not None
          and abs(p["elevation_deg"] - 77.0) <= 1.0,
          f"{p['elevation_deg']:.2f} (want ~77.0)")
    check("tokyo noon az", p["azimuth_deg"] is not None
          and az_err(p["azimuth_deg"], 199.0) <= 1.5,
          f"{p['azimuth_deg']:.2f} (want ~199)")

    # Sydney mid-winter noon (02:00Z ~ local transit): low northern sun.
    p = sun_pos("2025-06-15T02:00:00Z", *SYDNEY)
    check("sydney winter noon elev", p["elevation_deg"] is not None
          and abs(p["elevation_deg"] - 32.8) <= 1.0,
          f"{p['elevation_deg']:.2f} (want ~32.8)")
    check("sydney winter noon az", p["azimuth_deg"] is not None
          and az_err(p["azimuth_deg"], 358.8) <= 1.5,
          f"{p['azimuth_deg']:.2f} (want ~358.8)")

    # --- argument-order robustness: (lat, lon, when) must agree ---
    a1 = sun_pos("2025-06-15T12:00:00Z", *LONDON)
    a2 = sun_pos(LONDON[0], LONDON[1], "2025-06-15T12:00:00Z")
    a3 = sun_pos(when="2025-06-15T12:00:00Z", lat=LONDON[0], lon=LONDON[1])
    check("(when,lat,lon) == (lat,lon,when)",
          a1["elevation_deg"] == a2["elevation_deg"] == a3["elevation_deg"]
          and a1["azimuth_deg"] == a2["azimuth_deg"] == a3["azimuth_deg"],
          f"{a1} vs {a2} vs {a3}")

    # --- declination / equation of time sanity ---
    d = declination("2025-06-15T12:00:00Z")
    check("declination near solstice", abs(d - 23.35) <= 0.5, f"{d:.2f} (want ~23.35)")
    d = declination("2025-12-21T12:00:00Z")
    check("declination dec solstice", abs(d + 23.44) <= 0.5, f"{d:.2f} (want ~-23.44)")
    e = equation_of_time("2025-02-11T12:00:00Z")
    check("eot mid-feb ~ -14min", -15.5 <= e <= -13.0, f"{e:.2f} (want ~-14.2)")
    e = equation_of_time("2025-11-03T12:00:00Z")
    check("eot early-nov ~ +16min", 15.0 <= e <= 17.5, f"{e:.2f} (want ~16.4)")

    # --- solar noon ---
    noon = solar_noon_utc(*LONDON, "2025-06-15")
    noon_dt = datetime.fromisoformat(noon)
    noon_minutes = noon_dt.hour * 60 + noon_dt.minute + noon_dt.second / 60.0
    check("solar noon london ~12:01Z", 11 * 60 + 45 <= noon_minutes <= 12 * 60 + 15,
          noon)

    # --- never raise on garbage ---
    bad = sun_pos("garbage", "junk", None)
    check("bad input -> None fields",
          bad == {"elevation_deg": None, "azimuth_deg": None}, str(bad))
    check("bad declination -> NaN", math.isnan(declination("junk")), "nan")
    check("bad eot -> NaN", math.isnan(equation_of_time(None)), "nan")
    check("bad noon -> ''", solar_noon_utc(None, "x", "y") == "", "''")
    check("bad state -> unknown", twilight_state(None, None, None) == "unknown", "unknown")
    check("bad is_day -> False", is_day(None, None, None) is False, "False")
    check("empty sun_pos", sun_pos() == {"elevation_deg": None, "azimuth_deg": None}, "ok")

    # --- drift vs pysolar (informational; fails only beyond 2 deg) ---
    ref = None
    try:
        from pysolar.solar import get_altitude, get_azimuth

        def ref(lat, lon, w):
            return get_altitude(lat, lon, w), get_azimuth(lat, lon, w) % 360.0
    except Exception:
        try:
            try:
                from .solar import sun_position as _sp
            except Exception:
                import os
                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if root not in sys.path:
                    sys.path.insert(0, root)
                from sundial.solar import sun_position as _sp

            def ref(lat, lon, w):
                r = _sp(lat, lon, w)
                return r["elevation_deg"], r["azimuth_deg"]
        except Exception:
            ref = None

    if ref is None:
        print("  pysolar comparison skipped (not importable)")
    else:
        max_e = max_a = 0.0
        worst = ""
        for lat, lon in (LONDON, TOKYO, SYDNEY, (0.0, 0.0), (64.1466, -21.9426),
                         (-23.55, -46.63)):
            for day in ("2025-06-15", "2025-12-21", "2025-03-20"):
                base = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
                for h in range(0, 24, 2):
                    w = base + timedelta(hours=h)
                    mine = sun_pos(w, lat, lon)
                    pe, pa = ref(lat, lon, w)
                    de = abs(mine["elevation_deg"] - pe)
                    da = az_err(mine["azimuth_deg"], pa)
                    if de > max_e or da > max_a:
                        worst = f"({lat},{lon}) {w.isoformat()}"
                    max_e = max(max_e, de)
                    max_a = max(max_a, da)
        drift = max(max_e, max_a)
        check("pysolar drift <= 2 deg", drift <= 2.0,
              f"max elev err {max_e:.3f} deg, max az err {max_a:.3f} deg at {worst}")

    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)
