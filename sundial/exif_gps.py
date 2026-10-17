"""Embedded-GPS forensics — extract, sanity-check, and score the
coordinate tags a file carries against the claimed place.

camera_meta.py skims EXIF/container metadata for the pipeline's general
metadata card; this module is the GPS-specific deep dive. Coordinates
hide in two places:

  * stills — the EXIF GPSInfo IFD: lat/lon as (deg, min, sec) rationals
    with N/S/E/W refs, an altitude rational + above/below-sea-level ref,
    and the GPS receiver's own UTC date/time stamps (a *different* clock
    from the camera's DateTimeOriginal, and harder to mis-set casually).
  * video — ISO 6709 ``location`` tags (iPhone .mov, tagged mp4), read
    through video_meta.probe()'s lazy ffprobe wrapper; the optional
    third signed field is altitude, which the base reader drops.

Public API — nothing here ever raises; absent files, missing deps and
junk tags all degrade to None / empty tells:

  * extract_gps(path)      -> {"lat","lon","alt","dt","src","mtime"} | None
  * gps_plausibility(gps)  -> {"plausible": bool, "tells": [{"kind","detail"}]}
  * gps_evidence(gps, anchor, claim_place) -> Evidence | None (type "exif_gps")
  * gps_marker(gps)        -> {"lat","lon","kind":"geo","label":...} | None

IMPORTANT: a geotag is one ``exiftool -GPSLatitude=...`` away from
fiction — SUPPORTS confidence is hard-capped at 0.4 and every positive
summary says so. A fix that fails sanity checks can't even contradict:
a junk tag testifies about nothing, so implausibility -> INCONCLUSIVE.

The Evidence card nests the fix under data["gps"]["lat"/"lon"], which
locate_summary's geo-marker pass already picks up for the map.

  python3 sundial/exif_gps.py           # self-test (offline)
  python3 sundial/exif_gps.py FILE      # dump one file's fix + verdict
"""
from __future__ import annotations

import math
import os
import re
import sys
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only — the real import is lazy (see _schema)
    from .schema import Evidence

# --- EXIF tag ids (numeric; stable across Pillow versions) ------------------
_EXIF_IFD = 0x8769            # pointer to the Exif IFD (DateTimeOriginal)
_GPS_IFD = 0x8825             # pointer to the GPSInfo IFD
_TAG_DATETIME = 0x0132        # base "DateTime" — last-resort clock
_TAG_DATETIME_ORIG = 0x9003   # DateTimeOriginal, lives in the Exif IFD
_GPS_LAT_REF, _GPS_LAT = 0x0001, 0x0002
_GPS_LON_REF, _GPS_LON = 0x0003, 0x0004
_GPS_ALT_REF, _GPS_ALT = 0x0005, 0x0006   # ref byte: 0 above / 1 below sea level
_GPS_TIME, _GPS_DATE = 0x0007, 0x001D     # receiver UTC clock (H/M/S + Y:M:D)

_EXIF_DT_FMT = "%Y:%m:%d %H:%M:%S"        # EXIF "YYYY:MM:DD HH:MM:SS"
_GPS_DATE_FMT = "%Y:%m:%d"                # GPSDateStamp "YYYY:MM:DD"

#: ISO 6709 "+51.5074-000.1278+015.000/" — the optional third signed
#: field is altitude in metres (video_meta keeps only lat/lon).
_ISO6709_RE = re.compile(
    r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?")

# --- scoring thresholds ----------------------------------------------------
_NEAR_KM = 50.0        # fix within this of the claim anchor -> SUPPORTS
_FAR_KM = 1000.0       # fix beyond this                  -> CONTRADICTS

# --- physical sanity bands -------------------------------------------------
_MAX_ALT_M = 10000.0        # Everest ~8849 m; higher consumer fix = fake
_MIN_ALT_M = -500.0         # Dead Sea shore ~-430 m; lower = impossible on land
_NULL_ISLAND_DEG = 0.01     # |lat| & |lon| below this -> (0,0) "Null Island"
#: A capture stamp later than the file's own mtime (+slack) can't be real:
#: the bytes would have been written before the moment they describe.
_MTIME_SLACK = timedelta(minutes=10)

# --- confidences (spec-fixed; GPS is the easiest forgery surface there is) --
_SUPP_CONF = 0.4            # capped — forgeable, weak anchor only
_CONTRA_CONF = 0.5
_AMBIG_CONF = 0.3
_NO_ANCHOR_CONF = 0.2
_IMPLAUSIBLE_CONF = 0.3

#: Tell kinds that make a fix physically untrustworthy (plausible=False).
#: Anything else in tells is a soft flag — noted, but the fix still scores.
_HARD_TELLS = {"missing_coords", "impossible_lat", "impossible_lon",
               "impossible_alt", "null_island", "capture_after_write"}

#: Honest reliability note stamped on every verdict (cf. camera_meta).
_RELIABILITY = ("geotags are trivially forged — one exiftool/ffmpeg "
                "command rewrites them; weak anchor only")


# --------------------------------------------------------------------------
# small coercion helpers (local copies, same shapes as camera_meta's)
# --------------------------------------------------------------------------

def _num(v) -> float | None:
    """float() that also digests PIL IFDRational and (num, den) tuples."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:  # raw rational pair, e.g. from a non-PIL EXIF reader
        num, den = v
        return float(num) / float(den) if den else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _clean_str(v) -> str | None:
    """Stripped tag string (they arrive NUL/space-padded), or None."""
    if v is None:
        return None
    s = str(v).strip(" \t\x00")
    return s or None


def _dms_to_deg(dms, ref) -> float | None:
    """(deg, min, sec) rationals + hemisphere ref -> signed decimal degrees."""
    try:
        parts = [_num(x) for x in list(dms)[:3]]
    except TypeError:
        return None
    if len(parts) < 3 or any(p is None for p in parts):
        return None
    deg = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if str(ref).strip().upper() in ("S", "W"):
        deg = -deg
    return deg


def _as_dt(v) -> datetime | None:
    """datetime | epoch float | EXIF/ISO string -> naive datetime, or None."""
    if isinstance(v, datetime):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            return datetime.fromtimestamp(v)
        except (OSError, OverflowError, ValueError):
            return None
    s = _clean_str(v)
    if not s:
        return None
    try:
        return datetime.strptime(s, _EXIF_DT_FMT)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (same formula as geo.py's copy)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _anchor_ll(anchor) -> tuple[float, float] | None:
    """(lat, lon) | {"lat","lon"} | GeoCandidate-ish -> floats, or None."""
    if anchor is None:
        return None
    if isinstance(anchor, dict):
        lat, lon = _num(anchor.get("lat")), _num(anchor.get("lon"))
    else:
        try:
            lat, lon = _num(anchor[0]), _num(anchor[1])
        except (TypeError, IndexError):
            lat = lon = None
    if lat is None or lon is None:
        return None
    return lat, lon


def _file_mtime(path) -> datetime | None:
    """File last-write time as naive local datetime; None on any failure."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(str(path)))
    except (OSError, TypeError, ValueError, OverflowError):
        return None


def _schema():
    """Lazy schema import — tolerates both package and direct-script use."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # `python sundial/exif_gps.py` direct run
        try:
            from schema import ClaimVerdict, Evidence
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


def _video_probe():
    """video_meta.probe callable (lazy ffprobe wrapper), or None."""
    try:
        from .video_meta import probe
        return probe
    except ImportError:
        pass
    try:
        from video_meta import probe  # direct-script run
        return probe
    except ImportError:
        pass
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from video_meta import probe
        return probe
    except ImportError:
        return None


# --------------------------------------------------------------------------
# extraction — stills (lazy PIL) then containers (lazy ffprobe)
# --------------------------------------------------------------------------

def _gps_stamp_dt(gps_ifd: dict) -> datetime | None:
    """GPSDateStamp + GPSTimeStamp -> naive UTC datetime.

    This is the satellite receiver's clock, not the camera's — a mis-set
    camera clock doesn't touch it, so it's the more trustworthy of the
    two EXIF timestamps. A time without a date can't be placed -> None.
    """
    raw_date = _clean_str(gps_ifd.get(_GPS_DATE))
    raw_time = gps_ifd.get(_GPS_TIME)
    if raw_date is None and raw_time is None:
        return None
    y = mo = d = None
    if raw_date:
        try:
            base = datetime.strptime(raw_date, _GPS_DATE_FMT)
            y, mo, d = base.year, base.month, base.day
        except ValueError:
            y = mo = d = None
    h = mi = s = 0.0
    if raw_time is not None:
        try:
            parts = [_num(x) for x in list(raw_time)[:3]]
        except TypeError:
            parts = []
        if len(parts) >= 3 and all(p is not None for p in parts):
            h, mi, s = parts
    if y is None:
        return None
    try:
        return datetime(y, mo, d) + timedelta(hours=h, minutes=mi, seconds=s)
    except (ValueError, OverflowError):
        return None


def _exif_fallback_dt(exif_ifd: dict, exif) -> datetime | None:
    """DateTimeOriginal (or base DateTime) -> naive camera-clock datetime."""
    raw = (_clean_str(exif_ifd.get(_TAG_DATETIME_ORIG))
           or _clean_str(exif.get(_TAG_DATETIME_ORIG) if exif else None)
           or _clean_str(exif.get(_TAG_DATETIME) if exif else None))
    if not raw:
        return None
    try:
        return datetime.strptime(raw, _EXIF_DT_FMT)
    except ValueError:
        return None


def _exif_gps(path: str) -> dict | None:
    """GPSInfo IFD -> fix dict via lazy PIL; None on any failure/absence."""
    try:
        from PIL import Image  # lazy: module must import without PIL
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            legacy = {}  # pre-get_ifd Pillow: flat dict w/ nested GPS IFD
            getexif = getattr(img, "_getexif", None)
            if getexif is not None:
                try:
                    legacy = getexif() or {}
                except Exception:
                    legacy = {}
    except Exception:
        return None
    if not exif and not legacy:
        return None

    def ifd(tag: int) -> dict:
        got = {}
        if exif:
            try:
                got = dict(exif.get_ifd(tag) or {})
            except Exception:
                got = {}
        if not got and isinstance(legacy, dict):
            sub = legacy.get(tag)
            if isinstance(sub, dict):
                got = dict(sub)
        return got

    gps_ifd, exif_ifd = ifd(_GPS_IFD), ifd(_EXIF_IFD)
    if not gps_ifd:
        return None

    lat = _dms_to_deg(gps_ifd.get(_GPS_LAT), gps_ifd.get(_GPS_LAT_REF))
    lon = _dms_to_deg(gps_ifd.get(_GPS_LON), gps_ifd.get(_GPS_LON_REF))
    if lat is None or lon is None:
        return None

    alt = _num(gps_ifd.get(_GPS_ALT))
    ref = gps_ifd.get(_GPS_ALT_REF)
    if alt is not None and (ref == 1 or ref == b"\x01" or str(ref).strip() == "1"):
        alt = -alt  # ref 1 = below sea level

    # Prefer the receiver's UTC stamps; the camera clock is the fallback.
    dt = _gps_stamp_dt(gps_ifd) or _exif_fallback_dt(exif_ifd, exif)
    return {"lat": lat, "lon": lon, "alt": alt, "dt": dt, "src": "exif"}


def _container_gps(path: str) -> dict | None:
    """ISO 6709 location tag -> fix dict via video_meta.probe(); or None."""
    probe = _video_probe()
    if probe is None:
        return None
    try:
        meta = probe(path)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    loc = meta.get("location")
    if not isinstance(loc, dict):
        return None
    lat, lon = _num(loc.get("lat")), _num(loc.get("lon"))
    if lat is None or lon is None:
        return None
    alt = None  # video_meta drops it — re-parse the raw tag's 3rd field
    m = _ISO6709_RE.match(str(loc.get("raw") or ""))
    if m and m.group(3) is not None:
        alt = _num(m.group(3))
    return {"lat": lat, "lon": lon, "alt": alt,
            "dt": _as_dt(meta.get("creation_time")), "src": "iso6709"}


def extract_gps(path) -> dict | None:
    """Best embedded coordinate fix for `path`, or None when absent.

    Tries still-EXIF first (JPEG/PNG/TIFF GPSInfo IFD via lazy PIL), then
    container tags (ISO 6709 location via lazy ffprobe, for video). On
    success::

        {"lat": float, "lon": float,    # signed decimal degrees, as tagged
         "alt": float | None,           # metres above sea level
         "dt": datetime | None,         # GPS UTC clock, else capture clock
         "src": "exif" | "iso6709",
         "mtime": datetime | None}      # file mtime (naive local) for skew

    The fix is returned *as tagged* — range-policing is
    gps_plausibility()'s job, so a corrupt tag surfaces as an implausible
    fix rather than vanishing silently. Never raises.
    """
    try:
        if not path or not os.path.exists(str(path)):
            return None
    except (OSError, TypeError, ValueError):
        return None
    gps = _exif_gps(str(path)) or _container_gps(str(path))
    if gps is None:
        return None
    gps["mtime"] = _file_mtime(path)
    return gps


# --------------------------------------------------------------------------
# plausibility — does the tagged fix describe a physically possible capture?
# --------------------------------------------------------------------------

def gps_plausibility(gps) -> dict:
    """Sanity-check a fix dict -> {"plausible": bool, "tells": [...]}.

    Tells are ``{"kind", "detail"}`` pairs (same shape as video_meta's).
    Hard tells — ``missing_coords``, ``impossible_lat``/``lon``/``alt``,
    ``null_island``, ``capture_after_write`` — flip ``plausible`` to
    False. Soft tells (``zero_altitude``, ``bad_dt``) are listed but the
    fix still counts as plausible. Never raises; a non-dict gps reads as
    missing coordinates.
    """
    tells: list[dict] = []
    g = gps if isinstance(gps, dict) else {}
    lat, lon = _num(g.get("lat")), _num(g.get("lon"))

    if lat is None or lon is None:
        tells.append({"kind": "missing_coords",
                      "detail": "no parseable lat/lon in the fix"})
    else:
        if not -90.0 <= lat <= 90.0:
            tells.append({"kind": "impossible_lat",
                          "detail": f"latitude {lat} is outside -90..90 — "
                                    "no real point on Earth"})
        if not -180.0 <= lon <= 180.0:
            tells.append({"kind": "impossible_lon",
                          "detail": f"longitude {lon} is outside -180..180 — "
                                    "no real point on Earth"})
        if abs(lat) < _NULL_ISLAND_DEG and abs(lon) < _NULL_ISLAND_DEG:
            tells.append({"kind": "null_island",
                          "detail": "coordinates are (0, 0) — 'Null Island', "
                                    "the default a careless geotag or forgery "
                                    "leaves behind (open ocean off Ghana; no "
                                    "genuine fix lands exactly here)"})

    alt = _num(g.get("alt"))
    if alt is not None:
        if alt < _MIN_ALT_M or alt > _MAX_ALT_M:
            tells.append({"kind": "impossible_alt",
                          "detail": f"altitude {alt} m is outside "
                                    f"{_MIN_ALT_M:.0f}..{_MAX_ALT_M:.0f} m — "
                                    "impossible for a ground-level capture"})
        elif alt == 0.0:
            tells.append({"kind": "zero_altitude",
                          "detail": "altitude is exactly 0.0 m — plausible "
                                    "only at the coast; real fixes almost "
                                    "never read a round zero (forged tags "
                                    "default it)"})

    raw_dt = g.get("dt")
    dt, mt = _as_dt(raw_dt), _as_dt(g.get("mtime"))
    if raw_dt is not None and dt is None:
        tells.append({"kind": "bad_dt",
                      "detail": f"GPS timestamp {_clean_str(raw_dt)!r} "
                                "doesn't parse"})
    if dt is not None and mt is not None and dt - mt > _MTIME_SLACK:
        tells.append({"kind": "capture_after_write",
                      "detail": f"capture timestamp {dt.isoformat(' ')} is "
                                f"{(dt - mt).total_seconds() / 3600:.1f}h "
                                f"AFTER the file's last write "
                                f"{mt.isoformat(' ')} — a file can't contain "
                                "a moment from its own future; the tag was "
                                "planted or a clock is lying"})

    return {"plausible": not any(t["kind"] in _HARD_TELLS for t in tells),
            "tells": tells}


# --------------------------------------------------------------------------
# evidence — score the fix against the claimed place
# --------------------------------------------------------------------------

def _public_fix(gps: dict) -> dict:
    """JSON-safe copy of the fix for data["gps"] (datetimes -> iso)."""
    out: dict = {}
    for k in ("lat", "lon", "alt"):
        v = _num(gps.get(k))
        if v is not None:
            out[k] = v
    for k in ("dt", "mtime"):
        v = _as_dt(gps.get(k))
        if v is not None:
            out[k] = v.isoformat(" ")
    src = _clean_str(gps.get("src"))
    if src:
        out["src"] = src
    return out


def gps_evidence(
    gps,
    anchor=None,
    claim_place=None,
) -> "Evidence | None":
    """Score an extract_gps() fix against the claim -> Evidence | None.

      * gps absent / no parseable lat+lon -> None (nothing to say)
      * hard plausibility tells           -> INCONCLUSIVE 0.3 — a junk or
        planted tag can't testify for either side; tell details go in the
        summary and data["plausibility"]
      * fix <=50 km from `anchor`         -> SUPPORTS 0.4 — capped, and the
        summary says why: geotags are trivially forged
      * fix >1000 km                      -> CONTRADICTS 0.5
      * between                           -> INCONCLUSIVE 0.3
      * fix present but no usable anchor  -> INCONCLUSIVE 0.2 (the fix
        still geolocates the file — worth reporting, scores nothing)

    `anchor` is (lat, lon) or a dict/GeoCandidate with .lat/.lon-ish
    fields; `claim_place` is the free-text claim used for readable
    summaries ("London"). Never raises.
    """
    ClaimVerdict, Evidence = _schema()

    if not isinstance(gps, dict):
        return None
    lat, lon = _num(gps.get("lat")), _num(gps.get("lon"))
    if lat is None or lon is None:
        return None

    rep = gps_plausibility(gps)
    src = ("container ISO6709 location"
           if gps.get("src") == "iso6709" else "EXIF GPS")
    place = _clean_str(claim_place) or "the claimed place"
    data: dict = {"gps": _public_fix(gps), "plausibility": rep,
                  "reliability": _RELIABILITY}

    if not rep["plausible"]:
        reasons = "; ".join(t["detail"] for t in rep["tells"])
        return Evidence(
            type="exif_gps",
            summary=f"{src} fails sanity checks — {reasons}; a tag that "
                    "can't be real can't testify for either side",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=_IMPLAUSIBLE_CONF,
            data=data,
        )

    soft = "; ".join(t["detail"] for t in rep["tells"])
    tail = f" (also: {soft})" if soft else ""

    pt = _anchor_ll(anchor)
    if pt is None:
        return Evidence(
            type="exif_gps",
            summary=f"{src} puts the capture at ({lat:.4f}, {lon:.4f}) but "
                    "the claim has no resolved anchor to check it "
                    f"against{tail}",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=_NO_ANCHOR_CONF,
            data=data,
        )

    d = _haversine_km(lat, lon, pt[0], pt[1])
    data["distance_km"] = round(d, 2)
    if d <= _NEAR_KM:
        verdict, conf = ClaimVerdict.SUPPORTS, _SUPP_CONF
        note = (f"{src} is {d:.1f} km from {place} — consistent, but "
                "geotags are trivially forged, so this is weak support only")
    elif d > _FAR_KM:
        verdict, conf = ClaimVerdict.CONTRADICTS, _CONTRA_CONF
        note = (f"{src} is {d:.0f} km from {place} — nowhere near; geotags "
                "are forgeable, but a fix this far off is a real red flag")
    else:
        verdict, conf = ClaimVerdict.INCONCLUSIVE, _AMBIG_CONF
        note = (f"{src} is {d:.0f} km from {place} — ambiguous (between "
                f"{_NEAR_KM:.0f} and {_FAR_KM:.0f} km)")
    return Evidence(
        type="exif_gps",
        summary=note + tail,
        verdict=verdict,
        confidence=conf,
        data=data,
    )


# --------------------------------------------------------------------------
# map marker — locate_summary / map_viz consumption
# --------------------------------------------------------------------------

def gps_marker(gps) -> dict | None:
    """{"lat","lon","kind":"geo","label":"exif gps"} for the map layer.

    None when gps isn't a dict or carries no in-range fix — implausible
    points stay off the map (a (0,0) marker would just be noise)."""
    if not isinstance(gps, dict):
        return None
    lat, lon = _num(gps.get("lat")), _num(gps.get("lon"))
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return {"lat": round(lat, 5), "lon": round(lon, 5),
            "kind": "geo", "label": "exif gps"}


# --------------------------------------------------------------------------
# self-test — canned fix dicts only; offline except a guarded PIL demo
# --------------------------------------------------------------------------

def _self_test() -> int:
    fails = 0

    def check(name: str, got, want, tol: float | None = None) -> None:
        nonlocal fails
        ok = (got is not None and abs(got - want) <= tol) \
            if tol is not None else (got == want)
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r} (want {want!r})")

    def has_tell(rep: dict, kind: str) -> bool:
        return any(t.get("kind") == kind for t in rep.get("tells", []))

    ClaimVerdict, _ = _schema()
    london = (51.5074, -0.1278)

    # --- helpers -----------------------------------------------------------
    check("dms N", _dms_to_deg((51, 30, 0), "N"), 51.5, 1e-9)
    check("dms W", _dms_to_deg((0, 7, 30), "W"), -0.125, 1e-9)
    check("dms bad", _dms_to_deg(("x", 1, 2), "N"), None)
    check("as_dt exif", _as_dt("2025:10:15 15:30:00"),
          datetime(2025, 10, 15, 15, 30, 0))
    check("as_dt iso z", _as_dt("2025-10-15T15:30:00Z"),
          datetime(2025, 10, 15, 15, 30, 0))
    check("as_dt junk", _as_dt("not a date"), None)
    check("anchor tuple", _anchor_ll(london), london)
    check("anchor dict", _anchor_ll({"lat": 51.5, "lon": -0.1}), (51.5, -0.1))
    check("anchor junk", _anchor_ll("London"), None)

    # --- plausibility ------------------------------------------------------
    rep = gps_plausibility({"lat": 51.5, "lon": -0.12})
    check("clean fix plausible", rep["plausible"], True)
    check("clean fix no tells", rep["tells"], [])

    rep = gps_plausibility({"lat": 0.0, "lon": 0.0})
    check("null island implausible", rep["plausible"], False)
    check("null island tell", has_tell(rep, "null_island"), True)

    rep = gps_plausibility({"lat": 91.0, "lon": 10.0})
    check("lat>90 implausible", rep["plausible"], False)
    check("impossible_lat tell", has_tell(rep, "impossible_lat"), True)

    rep = gps_plausibility({"lat": 10.0, "lon": -181.0})
    check("lon<-180 implausible", rep["plausible"], False)
    check("impossible_lon tell", has_tell(rep, "impossible_lon"), True)

    rep = gps_plausibility({"lat": 51.5, "lon": -0.12, "alt": 12000.0})
    check("alt 12km implausible", rep["plausible"], False)
    check("impossible_alt tell", has_tell(rep, "impossible_alt"), True)

    rep = gps_plausibility({"lat": 51.5, "lon": -0.12, "alt": 0.0})
    check("zero alt still plausible", rep["plausible"], True)
    check("zero alt soft tell", has_tell(rep, "zero_altitude"), True)

    mt = datetime(2025, 10, 15, 12, 0, 0)
    rep = gps_plausibility({"lat": 51.5, "lon": -0.12,
                            "dt": datetime(2025, 10, 15, 13, 0), "mtime": mt})
    check("dt after mtime implausible", rep["plausible"], False)
    check("capture_after_write tell",
          has_tell(rep, "capture_after_write"), True)

    rep = gps_plausibility({"lat": 51.5, "lon": -0.12,
                            "dt": datetime(2025, 10, 15, 11, 0), "mtime": mt})
    check("dt before mtime ok", rep["plausible"], True)

    rep = gps_plausibility(None)
    check("none gps implausible", rep["plausible"], False)
    check("none gps tell", has_tell(rep, "missing_coords"), True)

    # --- evidence ----------------------------------------------------------
    check("absent gps -> None", gps_evidence(None, london, "London"), None)
    check("empty gps -> None", gps_evidence({}, london, "London"), None)
    check("junk gps -> None",
          gps_evidence({"lat": "x"}, london, "London"), None)

    ev = gps_evidence({"lat": 51.51, "lon": -0.13, "src": "exif"},
                      london, "London")
    check("near -> SUPPORTS", ev.verdict, ClaimVerdict.SUPPORTS)
    check("near conf", ev.confidence, _SUPP_CONF)
    check("near type", ev.type, "exif_gps")
    check("forgeable caveat in summary", "forg" in ev.summary.lower(), True)
    check("distance recorded", ev.data.get("distance_km") is not None, True)
    check("reliability stamped", ev.data.get("reliability"), _RELIABILITY)
    check("data gps lat", ev.data["gps"].get("lat"), 51.51)

    ev = gps_evidence({"lat": 40.7128, "lon": -74.0060}, london, "London")
    check("far -> CONTRADICTS", ev.verdict, ClaimVerdict.CONTRADICTS)
    check("far conf", ev.confidence, _CONTRA_CONF)

    ev = gps_evidence({"lat": 48.8566, "lon": 2.3522}, london, "London")
    check("mid -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    check("mid conf", ev.confidence, _AMBIG_CONF)

    ev = gps_evidence({"lat": 0.0, "lon": 0.0}, london, "London")
    check("junk fix -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    check("junk fix conf", ev.confidence, _IMPLAUSIBLE_CONF)
    check("junk reasons in summary", "sanity" in ev.summary.lower(), True)

    ev = gps_evidence({"lat": 51.5, "lon": -0.1}, None, "London")
    check("no anchor -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    check("no anchor conf", ev.confidence, _NO_ANCHOR_CONF)

    ev = gps_evidence({"lat": 51.5, "lon": -0.1, "src": "iso6709"},
                      london, "London")
    check("iso6709 wording", "ISO6709" in ev.summary, True)

    # --- marker ------------------------------------------------------------
    m = gps_marker({"lat": 51.50735, "lon": -0.12775})
    check("marker shape",
          m, {"lat": 51.50735, "lon": -0.12775,
              "kind": "geo", "label": "exif gps"})
    check("marker none", gps_marker(None), None)
    check("marker empty", gps_marker({}), None)
    check("marker junk", gps_marker({"lat": "x", "lon": 1}), None)
    check("marker out-of-range",
          gps_marker({"lat": 91.0, "lon": 0.0}), None)

    # --- extraction: missing/garbage paths ---------------------------------
    check("extract missing -> None",
          extract_gps("/nonexistent/definitely-not-here.jpg"), None)
    check("extract none -> None", extract_gps(None), None)
    check("extract dir -> None", extract_gps(os.path.dirname(__file__)), None)

    # --- guarded live check: generated JPEG with a GPS IFD -----------------
    try:
        from PIL import Image
    except ImportError:
        print("  (PIL absent — generated-JPEG extract_gps check skipped)")
        return fails
    import tempfile
    path = os.path.join(tempfile.mkdtemp(prefix="sundial_gps_"), "fix.jpg")
    img = Image.new("RGB", (8, 8), (40, 60, 90))
    exif = Image.Exif()
    try:  # nested IFDs need Pillow >= 9.1; harmless if rejected
        exif[_GPS_IFD] = {_GPS_LAT_REF: "N", _GPS_LAT: (51, 30, 26),
                          _GPS_LON_REF: "W", _GPS_LON: (0, 7, 30),
                          _GPS_ALT_REF: 0, _GPS_ALT: 15.0,
                          _GPS_DATE: "2025:10:15", _GPS_TIME: (15, 5, 0)}
    except Exception:
        print("  (Pillow rejected nested GPS IFD — live check skipped)")
        return fails
    img.save(path, "JPEG", exif=exif)
    g = extract_gps(path)
    check("jpeg fix found", g is not None, True)
    if g:
        check("jpeg lat", g["lat"], 51.5072, 0.001)
        check("jpeg lon", g["lon"], -0.125, 0.001)
        check("jpeg alt", g["alt"], 15.0)
        check("jpeg src", g["src"], "exif")
        check("jpeg dt", g["dt"], datetime(2025, 10, 15, 15, 5, 0))
        check("jpeg mtime set", isinstance(g.get("mtime"), datetime), True)
    return fails


if __name__ == "__main__":
    import json

    fails = _self_test()
    print(f"self-test: {'FAILED' if fails else 'all passed'}")
    if fails:
        sys.exit(1)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        gps = extract_gps(path)
        print(f"\n{path}:")
        print(json.dumps(gps, indent=2, default=str))
        if gps:
            rep = gps_plausibility(gps)
            print(f"\nplausible={rep['plausible']}  tells="
                  f"{[t['kind'] for t in rep['tells']]}")
            print(f"marker: {gps_marker(gps)}")
            ev = gps_evidence(gps, None, None)
            print(f"evidence: {ev.verdict.value} conf={ev.confidence} — "
                  f"{ev.summary}")
        else:
            print("\nno embedded GPS fix")
