"""Shadow-geo stage — invert shadow observations into lat/lon BANDS.

The Bellingcat/ShadowFinder geolocation trick, automated: a measured sun
position plus a claimed UTC instant constrains *where on Earth* that sun
was physically possible — no anchor needed to compute the strip, and an
anchor can then be scored against it.

  * ``lat_band_from_elevation`` — a sun-elevation reading pins a latitude
    strip. At any fixed UTC instant, sweeping longitude at a given
    latitude sweeps every possible hour angle, so the feasible set is
    "latitudes whose daily elevation *range* contains the observation":
    the sun's culmination must reach at least the observed height (and,
    at the circumpolar edge, its lower culmination must not stay *above*
    it). UTC hour drops out — only the date's declination matters.

  * ``azimuth_lon_band`` — a sun-azimuth reading pins a longitude strip
    *at a given latitude*: the longitudes where the sun stands on the
    observed bearing (and is above the horizon — a cast shadow needs
    daylight).

  * ``shadow_evidence`` — scores an anchor against the bands: inside all
    -> SUPPORTS 0.5, outside any -> CONTRADICTS 0.55, else INCONCLUSIVE.
    ``data["bands"]`` is JSON-safe and shaped for
    ``locate_summary._bands`` (``lat_min``/``lat_max``/``lon_min``/
    ``lon_max`` keys), so the strips render on the locate map as-is.

Azimuths are degrees clockwise from true north (0=N, 90=E, 180=S, 270=W)
— the pysolar convention used across this package.

Ephemeris: ``.solar.sun_position`` (pysolar) is imported lazily and used
when present; otherwise a built-in approximation — Cooper's declination
(delta = 23.45 * sin(360/365 * (284 + doy))) plus the standard
NOAA-style equation-of-time — keeps every function working offline.
Stdlib ``math`` only at import; nothing here ever raises.

  python3 sundial/shadow_geo.py     # offline self-test
"""
from __future__ import annotations

import math
from datetime import date as _date, datetime, timezone

__all__ = [
    "lat_band_from_elevation",
    "azimuth_lon_band",
    "bands_from_observations",
    "evidence_from_observations",
    "shadow_evidence",
]

# Scan grids. The latitude scan is capped at +/-66 deg: past that the
# polar day/night geometry makes elevation-band answers misleading, and
# ~66 deg covers the populated world anyway (ShadowFinder's convention).
_LAT_LO, _LAT_HI, _LAT_STEP = -66.0, 66.0, 0.5
_LON_LO, _LON_HI, _LON_STEP = -180.0, 180.0, 1.0

# A cast shadow needs the geometric sun above the horizon.
_SUN_UP_MIN_DEG = 0.0

# obs-dict keys that may carry a world-frame sun azimuth (deg CW from N).
_AZ_KEYS = ("sun_azimuth_deg", "sun_az_deg", "sun_azimuth",
            "solar_azimuth_deg", "azimuth_deg", "sun_az")


# --------------------------------------------------------------------------
# Defensive primitives
# --------------------------------------------------------------------------

def _num(v):
    """float() that swallows everything and rejects non-finite values."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _parse_dt(dt_utc):
    """ISO string / date / datetime -> aware UTC datetime, or None."""
    if isinstance(dt_utc, datetime):
        dt = dt_utc
    elif isinstance(dt_utc, _date):
        dt = datetime(dt_utc.year, dt_utc.month, dt_utc.day)
    else:
        try:
            dt = datetime.fromisoformat(
                str(dt_utc).strip().replace("Z", "+00:00"))
        except Exception:
            return None
    # Same convention as solar._aware: naive reads as UTC.
    try:
        return (dt.astimezone(timezone.utc) if dt.tzinfo is not None
                else dt.replace(tzinfo=timezone.utc))
    except Exception:
        return None


def _azimuth_error(a, b):
    """Shortest distance between two azimuths in degrees (0/360 wrap-safe)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _circular_mean(angles):
    """Mean direction of azimuths in degrees (wrap-safe); None for []."""
    xs = [a for a in (_num(v) for v in (angles or [])) if a is not None]
    if not xs:
        return None
    y = sum(math.sin(math.radians(a)) for a in xs)
    x = sum(math.cos(math.radians(a)) for a in xs)
    if not x and not y:
        return xs[0] % 360.0
    return math.degrees(math.atan2(y, x)) % 360.0


def _median(values):
    """Median of the numeric entries in `values`; None when empty."""
    xs = sorted(v for v in (_num(v) for v in (values or []))
                if v is not None)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _anchor_pt(anchor):
    """Anchor in any caller-held form -> (lat, lon) or None.

    Tolerates (lat, lon) tuples/lists, {"lat","lon"} dicts, and objects
    with .lat/.lon (GeoCandidate shape) — mirrors locate_summary.
    """
    if anchor is None:
        return None
    try:
        if isinstance(anchor, dict):
            la, lo = _num(anchor.get("lat")), _num(anchor.get("lon"))
        elif isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
            la, lo = _num(anchor[0]), _num(anchor[1])
        else:
            la = _num(getattr(anchor, "lat", None))
            lo = _num(getattr(anchor, "lon", None))
        if la is None or lo is None:
            return None
        if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
            return None
        return la, lo
    except Exception:
        return None


# --------------------------------------------------------------------------
# Ephemeris — .solar (pysolar) when importable, else a built-in model
# --------------------------------------------------------------------------

_SUNPOS = None  # tri-state probe: None=untested, False=unavailable, else fn


def _ephem():
    """Lazy .solar.sun_position probe; returns the callable or None."""
    global _SUNPOS
    if _SUNPOS is None:
        try:
            from .solar import sun_position as sp
            _SUNPOS = sp
        except Exception:
            try:
                from solar import sun_position as sp  # loose-script fallback
                _SUNPOS = sp
            except Exception:
                _SUNPOS = False
    return _SUNPOS or None


def _cooper_decl_deg(doy):
    """Cooper's-equation declination: d = 23.45*sin(360/365*(284+doy))."""
    return 23.45 * math.sin(math.radians(360.0 / 365.0 * (284.0 + doy)))


def _eot_minutes(doy):
    """Equation of time in minutes (NOAA/Spencer approximation)."""
    b = math.radians(360.0 / 365.0 * (doy - 81.0))
    return 9.87 * math.sin(2.0 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def _approx_sunpos(lat, lon, dt):
    """-> (elevation_deg, azimuth_deg) from Cooper d + NOAA hour angle.

    Azimuth is deg clockwise from true north (pysolar convention):
    az = atan2(sin H, cos H*sin(phi) - tan(d)*cos(phi)) + 180.
    """
    doy = dt.timetuple().tm_yday
    dec = math.radians(_cooper_decl_deg(doy))
    utc_min = (dt.hour * 60.0 + dt.minute + dt.second / 60.0
               + dt.microsecond / 6e7)
    tst = (utc_min + _eot_minutes(doy) + 4.0 * lon) % 1440.0
    ha = math.radians(tst / 4.0 - 180.0)
    phi = math.radians(float(lat))
    sin_e = (math.sin(phi) * math.sin(dec)
             + math.cos(phi) * math.cos(dec) * math.cos(ha))
    elev = math.degrees(math.asin(max(-1.0, min(1.0, sin_e))))
    az = (math.degrees(math.atan2(
            math.sin(ha),
            math.cos(ha) * math.sin(phi) - math.tan(dec) * math.cos(phi)))
          + 180.0) % 360.0
    return elev, az


def _sunpos(lat, lon, dt):
    """-> (elevation_deg, azimuth_deg) or (None, None); never raises."""
    try:
        sp = _ephem()
        if sp is not None:
            r = sp(float(lat), float(lon), dt) or {}
            return _num(r.get("elevation_deg")), _num(r.get("azimuth_deg"))
        return _approx_sunpos(lat, lon, dt)
    except Exception:
        return None, None


def _declination_deg(dt):
    """Solar declination delta(dt) in degrees.

    At a pole every hour angle coincides, so the sun's elevation there
    equals delta; averaging both poles cancels most of pysolar's
    refraction bias. Falls back to Cooper's equation on any trouble.
    """
    try:
        e_n, _ = _sunpos(89.9, 0.0, dt)
        e_s, _ = _sunpos(-89.9, 0.0, dt)
        if e_n is not None and e_s is not None:
            d = (e_n - e_s) / 2.0
            if -30.0 <= d <= 30.0:
                return d
    except Exception:
        pass
    try:
        return _cooper_decl_deg(dt.timetuple().tm_yday)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------
# Bands
# --------------------------------------------------------------------------

def lat_band_from_elevation(sun_elev_deg, dt_utc, tol_deg=3.0) -> dict:
    """Latitude strip where a sun elevation of `sun_elev_deg` is possible.

    At a fixed UTC instant, sweeping longitude at a given latitude sweeps
    every hour angle, so a latitude is feasible iff the day's elevation
    range there covers the observation:

      * upper culmination  e_max = 90 - |lat - delta|  must reach obs
      * lower culmination  e_min = asin(-cos(lat + delta))  must not sit
        above obs — the midnight-sun edge case (within +/-66 deg it can
        never bind, since lat + delta < 90 always).

    -> {"lat_min","lat_max","lon_min":-180,"lon_max":180,
        "method":"elev_band","tol_deg", ...}. Lat bounds are None with
    "empty": True when no latitude is feasible (a physically impossible
    observation), and "ok" is False when inputs couldn't be evaluated.
    """
    obs = _num(sun_elev_deg)
    tol = _num(tol_deg)
    tol = abs(tol) if tol is not None else 3.0
    band = {
        "lat_min": None, "lat_max": None,
        "lon_min": -180.0, "lon_max": 180.0,
        "method": "elev_band", "kind": "lat",
        "tol_deg": tol, "obs_elev_deg": obs, "ok": False,
    }
    dt = _parse_dt(dt_utc)
    band["dt_utc"] = dt.isoformat() if dt else None
    if obs is None or dt is None:
        band["note"] = "missing/unparseable sun_elev_deg or dt_utc"
        return band
    dec = _declination_deg(dt)
    band["declination_deg"] = round(dec, 2)
    band["lat_scan"] = [_LAT_LO, _LAT_HI, _LAT_STEP]

    feas: list[float] = []
    lat = _LAT_LO
    while lat <= _LAT_HI + 1e-9:
        e_max = 90.0 - abs(lat - dec)
        e_min = math.degrees(math.asin(
            max(-1.0, min(1.0, -math.cos(math.radians(lat + dec))))))
        if e_min - tol <= obs <= e_max + tol:
            feas.append(round(lat, 2))
        lat += _LAT_STEP
    if feas:
        band.update(lat_min=feas[0], lat_max=feas[-1],
                    n_lats=len(feas), ok=True)
    else:
        band.update(empty=True, ok=True)
    return band


def _segments(lons):
    """Sorted matching longitudes -> [[lo, hi], ...] contiguous runs."""
    if not lons:
        return []
    lons = sorted(lons)
    segs = [[lons[0], lons[0]]]
    for lon in lons[1:]:
        if lon - segs[-1][1] <= _LON_STEP * 1.5:
            segs[-1][1] = lon
        else:
            segs.append([lon, lon])
    return [[round(a, 2), round(b, 2)] for a, b in segs]


def azimuth_lon_band(obs_azimuth_deg, lat, dt_utc, tol_deg=15.0) -> dict:
    """Longitude strip where the sun stands on `obs_azimuth_deg` at `dt`.

    Scans every longitude at the given latitude; a longitude is feasible
    when the sun is above the horizon (a cast shadow needs daylight) and
    its azimuth — deg clockwise from true north — is within `tol_deg` of
    the observation. The strip is conditioned on `lat`: azimuth drifts
    with latitude, so it is approximate for anchors far from `lat` (the
    default 15 deg tolerance absorbs typical mid-latitude drift).

    -> {"lat_min":-90,"lat_max":90,"lon_min","lon_max","segments",
        "method":"az_band","tol_deg", ...}. `segments` lists every
    disjoint matching longitude run; `lon_min`/`lon_max` cover the widest
    one for map rendering. None + "empty": True when no longitude matches.
    """
    obs = _num(obs_azimuth_deg)
    obs = obs % 360.0 if obs is not None else None
    la = _num(lat)
    tol = _num(tol_deg)
    tol = abs(tol) if tol is not None else 15.0
    band = {
        "lat_min": -90.0, "lat_max": 90.0,
        "lon_min": None, "lon_max": None, "segments": [],
        "method": "az_band", "kind": "lon",
        "tol_deg": tol, "obs_azimuth_deg": obs, "lat_deg": la,
        "ok": False,
    }
    dt = _parse_dt(dt_utc)
    band["dt_utc"] = dt.isoformat() if dt else None
    if obs is None or la is None or not (-90.0 <= la <= 90.0) or dt is None:
        band["note"] = "missing/unparseable azimuth, lat or dt_utc"
        return band
    band["sun_up_min_deg"] = _SUN_UP_MIN_DEG

    hits: list[float] = []
    best_lon = best_err = None
    lon = _LON_LO
    while lon <= _LON_HI + 1e-9:
        elev, az = _sunpos(la, lon, dt)
        if elev is not None and elev > _SUN_UP_MIN_DEG and az is not None:
            err = _azimuth_error(az, obs)
            if best_err is None or err < best_err:
                best_lon, best_err = round(lon, 2), err
            if err <= tol:
                hits.append(lon)
        lon += _LON_STEP

    band["segments"] = _segments(hits)
    band["n_lons"] = len(hits)
    if best_lon is not None:
        band["best_lon_deg"] = best_lon
        band["best_error_deg"] = round(best_err, 1)
    if band["segments"]:
        lo, hi = max(band["segments"], key=lambda s: s[1] - s[0])
        band.update(lon_min=lo, lon_max=hi, ok=True)
    else:
        band.update(empty=True, ok=True)
    return band


def bands_from_observations(shadow_obs_pairs=None, sun_elevs=None,
                            shadow_angles=None, claimed_utc=None,
                            anchor=None, elev_tol_deg=3.0,
                            az_tol_deg=15.0) -> list:
    """Pipeline inputs -> band dicts for `shadow_evidence`.

    * median of `sun_elevs` -> `lat_band_from_elevation`
    * world-frame sun azimuths carried in the shadow obs (any of
      ``sun_azimuth_deg``/``sun_az_deg``/...) -> `azimuth_lon_band`,
      evaluated at the anchor's latitude when known, else at the
      elevation band's midpoint.

    `shadow_angles` are image-space (deg clockwise from image-up) — with
    no camera heading they can't be converted to world azimuths here, so
    they are only counted in the band payload, not used as a constraint.
    """
    out: list[dict] = []

    lat_band = None
    elev = _median(sun_elevs)
    if elev is not None:
        lat_band = lat_band_from_elevation(elev, claimed_utc, elev_tol_deg)
        lat_band["n_obs_elev"] = len(sun_elevs or [])
        out.append(lat_band)

    azs: list[float] = []
    for pair in (shadow_obs_pairs or []):
        obs = (pair[1]
               if isinstance(pair, (list, tuple)) and len(pair) >= 2
               else pair)
        if not isinstance(obs, dict):
            continue
        for k in _AZ_KEYS:
            v = _num(obs.get(k))
            if v is not None:
                azs.append(v % 360.0)
                break
    az_obs = _circular_mean(azs)
    if az_obs is not None:
        pt = _anchor_pt(anchor)
        la = pt[0] if pt else None
        if la is None and lat_band and lat_band.get("ok") \
                and lat_band.get("lat_min") is not None:
            la = (lat_band["lat_min"] + lat_band["lat_max"]) / 2.0
        lon_band = azimuth_lon_band(az_obs, la, claimed_utc, az_tol_deg)
        lon_band["n_obs_az"] = len(azs)
        if shadow_angles:
            lon_band["n_shadow_angles_image"] = len(shadow_angles)
        out.append(lon_band)
    return out


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

def _anchor_in_band(pt, band):
    """(lat, lon) vs one band dict -> True / False / None (not evaluable)."""
    if not isinstance(band, dict) or band.get("ok") is False:
        return None
    if band.get("empty"):
        return False  # feasible set is empty — no anchor can match
    lat, lon = pt
    used = False

    la_lo, la_hi = _num(band.get("lat_min")), _num(band.get("lat_max"))
    if la_lo is not None and la_hi is not None:
        used = True
        if not (min(la_lo, la_hi) - 1e-9 <= lat <= max(la_lo, la_hi) + 1e-9):
            return False

    segs = band.get("segments")
    if isinstance(segs, (list, tuple)) and segs:
        used = True
        in_seg = False
        for s in segs:
            try:
                a, b = _num(s[0]), _num(s[1])
            except (TypeError, IndexError):
                a = b = None
            if a is None or b is None:
                continue
            if a <= b:
                if a - 1e-9 <= lon <= b + 1e-9:
                    in_seg = True
                    break
            elif lon >= a - 1e-9 or lon <= b + 1e-9:  # antimeridian wrap
                in_seg = True
                break
        if not in_seg:
            return False
    else:
        lo_lo = _num(band.get("lon_min"))
        lo_hi = _num(band.get("lon_max"))
        if lo_lo is not None and lo_hi is not None:
            used = True
            if lo_lo <= lo_hi:
                if not (lo_lo - 1e-9 <= lon <= lo_hi + 1e-9):
                    return False
            elif not (lon >= lo_lo - 1e-9 or lon <= lo_hi + 1e-9):
                return False
    return True if used else None


def _band_miss_txt(band, pt):
    """Short human-readable reason the anchor missed a band."""
    kind = str(band.get("method") or band.get("kind") or "band")
    if band.get("empty"):
        return f"{kind}: observation physically impossible (empty band)"
    la_lo, la_hi = _num(band.get("lat_min")), _num(band.get("lat_max"))
    if la_lo is not None and la_hi is not None and \
            not (min(la_lo, la_hi) <= pt[0] <= max(la_lo, la_hi)):
        return f"lat {pt[0]:.1f} not in [{la_lo:.1f},{la_hi:.1f}]"
    lo_lo, lo_hi = _num(band.get("lon_min")), _num(band.get("lon_max"))
    if lo_lo is not None and lo_hi is not None:
        return f"lon {pt[1]:.1f} not in [{lo_lo:.1f},{lo_hi:.1f}]"
    return f"{kind} miss"


def shadow_evidence(bands, anchor=None) -> "Evidence":
    """Score an anchor against lat/lon bands -> Evidence(type="shadow_geo").

    anchor inside every evaluable band -> SUPPORTS 0.5; outside at least
    one -> CONTRADICTS 0.55; no anchor or nothing evaluable ->
    INCONCLUSIVE. ``data["bands"]`` stays JSON-safe for
    ``locate_summary`` map rendering.
    """
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # loose-script fallback (same trick as shadow_consistency)
        from schema import ClaimVerdict, Evidence

    if isinstance(bands, dict):
        bands = [bands]
    blist = [b for b in (bands or []) if isinstance(b, dict)]
    pt = _anchor_pt(anchor)
    data = {"bands": blist,
            "anchor": ({"lat": round(pt[0], 5), "lon": round(pt[1], 5)}
                       if pt else None)}

    checks = [(b, _anchor_in_band(pt, b)) for b in blist] if pt else []
    evaluable = [(b, inside) for b, inside in checks if inside is not None]
    misses = [b for b, inside in evaluable if inside is False]

    if pt is None:
        verdict, conf = ClaimVerdict.INCONCLUSIVE, 0.15
        summary = ("shadow-geo bands computed but no anchor to test "
                   "them against")
    elif not evaluable:
        verdict, conf = ClaimVerdict.INCONCLUSIVE, 0.15
        summary = "no usable shadow-geo latitude/longitude bands"
    elif misses:
        verdict, conf = ClaimVerdict.CONTRADICTS, 0.55
        miss_txt = "; ".join(_band_miss_txt(b, pt) for b in misses[:2])
        summary = (f"anchor ({pt[0]:.2f},{pt[1]:.2f}) outside "
                   f"{len(misses)} shadow-geo band(s) — {miss_txt}")
    else:
        verdict, conf = ClaimVerdict.SUPPORTS, 0.5
        summary = (f"anchor ({pt[0]:.2f},{pt[1]:.2f}) inside all "
                   f"{len(evaluable)} shadow-geo band(s) — consistent "
                   "with the observed sun")

    data["n_bands"] = len(blist)
    data["n_evaluable"] = len(evaluable)
    data["n_violated"] = len(misses)
    return Evidence(type="shadow_geo", summary=summary, verdict=verdict,
                    confidence=conf, data=data)


def evidence_from_observations(shadow_obs_pairs=None, sun_elevs=None,
                               shadow_angles=None, claimed_utc=None,
                               anchor=None, elev_tol_deg=3.0,
                               az_tol_deg=15.0) -> "Evidence":
    """One-call convenience: pipeline inputs -> bands -> shadow_geo Evidence."""
    try:
        bands = bands_from_observations(
            shadow_obs_pairs=shadow_obs_pairs, sun_elevs=sun_elevs,
            shadow_angles=shadow_angles, claimed_utc=claimed_utc,
            anchor=anchor, elev_tol_deg=elev_tol_deg,
            az_tol_deg=az_tol_deg)
    except Exception:
        bands = []
    return shadow_evidence(bands, anchor)


# --------------------------------------------------------------------------
# Self-test: python3 sundial/shadow_geo.py  (works offline — falls back to
# the built-in ephemeris when pysolar/.solar is unavailable)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    try:
        from .schema import ClaimVerdict
    except ImportError:
        from schema import ClaimVerdict

    ok = True

    def check(name, cond, detail=""):
        global ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")

    print("ephemeris:", "pysolar (.solar)" if _ephem()
          else "built-in approximation")

    # London, mid-July, 12:00 UTC: sun ~60 deg up, nearly due south.
    dt = "2025-07-12T12:00:00+00:00"
    london = (51.5, -0.13)
    tokyo = (35.7, 139.7)

    band = lat_band_from_elevation(60.0, dt)
    check("elev band shape",
          band["lon_min"] == -180.0 and band["lon_max"] == 180.0
          and band["method"] == "elev_band",
          f"-> lat {band['lat_min']}..{band['lat_max']}  "
          f"dec={band.get('declination_deg')}")
    check("elev band covers London lat",
          band["lat_min"] is not None
          and band["lat_min"] <= london[0] <= band["lat_max"])
    check("elev band excludes Oslo lat 59.9",
          band["lat_min"] is not None
          and not (band["lat_min"] <= 59.9 <= band["lat_max"]))

    # Azimuth band: ask the active ephemeris what azimuth London actually
    # sees, then require the lon strip to contain London — self-consistent
    # whether pysolar or the built-in model produced it.
    _, london_az = _sunpos(london[0], london[1], _parse_dt(dt))
    check("model sun az at London ~due S",
          london_az is not None and 140.0 <= london_az <= 220.0,
          f"-> {london_az}")
    lon_band = azimuth_lon_band(london_az or 180.0, london[0], dt)
    check("lon band has segments", bool(lon_band["segments"]),
          f"-> {lon_band['segments']}")
    check("lon band contains London lon",
          _anchor_in_band(london, lon_band) is True)
    check("lon band excludes Tokyo lon",
          _anchor_in_band(tokyo, lon_band) is False)

    bands = [band, lon_band]
    ev_ok = shadow_evidence(bands, london)
    check("London anchor SUPPORTS",
          ev_ok.verdict == ClaimVerdict.SUPPORTS
          and abs(ev_ok.confidence - 0.5) < 1e-9,
          f"-> {ev_ok.verdict} conf={ev_ok.confidence}")
    ev_bad = shadow_evidence(bands, tokyo)
    check("Tokyo anchor CONTRADICTS",
          ev_bad.verdict == ClaimVerdict.CONTRADICTS
          and abs(ev_bad.confidence - 0.55) < 1e-9,
          f"-> {ev_bad.verdict} conf={ev_bad.confidence}: {ev_bad.summary}")
    ev_none = shadow_evidence([], london)
    check("empty bands INCONCLUSIVE",
          ev_none.verdict == ClaimVerdict.INCONCLUSIVE)
    ev_noanchor = shadow_evidence(bands, None)
    check("no anchor INCONCLUSIVE",
          ev_noanchor.verdict == ClaimVerdict.INCONCLUSIVE)
    check("bands JSON-safe", bool(json.dumps(ev_ok.data["bands"])))

    # bands_from_observations wiring (obs carry a world-frame azimuth).
    obs_pairs = [("f1.jpg", {"shadow_tip": [10, 20],
                            "base_point": [5, 20],
                            "sun_azimuth_deg": london_az})]
    bs = bands_from_observations(obs_pairs, [60.0, 61.0, 59.0], [200.0],
                                 dt, anchor=london)
    check("obs->bands produces 2 bands", len(bs) == 2, f"-> {len(bs)}")
    ev2 = shadow_evidence(bs, tokyo)
    check("obs->evidence Tokyo CONTRADICTS",
          ev2.verdict == ClaimVerdict.CONTRADICTS)
    ev3 = evidence_from_observations(obs_pairs, [60.0], [200.0], dt,
                                     anchor=london)
    check("one-call evidence SUPPORTS",
          ev3.verdict == ClaimVerdict.SUPPORTS)

    # Defensive: garbage inputs must never raise.
    g1 = lat_band_from_elevation(None, "not-a-date")
    g2 = azimuth_lon_band("x", 999.0, None)
    g3 = shadow_evidence("junk", {"lat": "bad"})
    g4 = lat_band_from_elevation(95.0, dt)  # physically impossible
    check("garbage tolerated",
          g1["ok"] is False and g2["ok"] is False
          and g3.verdict == ClaimVerdict.INCONCLUSIVE
          and g4.get("empty") is True)

    print("shadow_geo self-test:",
          "all PASS" if ok else "FAILURES above")
    sys.exit(0 if ok else 1)
