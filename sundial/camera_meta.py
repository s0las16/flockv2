"""File-metadata stage — EXIF GPS/timestamp forensics, OSINT-style.

Real geolocation work starts with the file itself: stills can carry EXIF
GPS coordinates, a DateTimeOriginal, camera make/model; containerized video
can carry a creation_time and encoder tag in its format metadata.

  * frame_metadata() — lazy PIL; JPEG/PNG EXIF incl. the GPS IFD
    (degrees/minutes/seconds + ref -> signed decimal degrees).
  * video_metadata() — lazy ffprobe subprocess; format tags + duration.
  * meta_evidence()  — compares what the metadata asserts against the
    claimed place anchor (haversine) and claimed time (EXIF clock).

Everything is lazy-imported (PIL, ffprobe, schema), so this module loads
with zero third-party deps and degrades to {} / weak INCONCLUSIVE when a
dependency or the metadata itself is absent.

IMPORTANT: EXIF and container tags are trivially editable — one exiftool
command rewrites them. A metadata match is therefore only ever a *weak*
anchor (see data["reliability"]); a mismatch still flags a story problem.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from datetime import datetime

# --- EXIF tag ids (numeric; stable across Pillow versions) ------------------
_EXIF_IFD = 0x8769  # pointer to the Exif IFD holding DateTimeOriginal
_GPS_IFD = 0x8825   # pointer to the GPSInfo IFD
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_SOFTWARE = 0x0131
_TAG_DATETIME = 0x0132          # base "DateTime" — fallback only
_TAG_DATETIME_ORIG = 0x9003     # DateTimeOriginal, lives in the Exif IFD
_GPS_LAT_REF, _GPS_LAT = 0x0001, 0x0002
_GPS_LON_REF, _GPS_LON = 0x0003, 0x0004

_EXIF_DT_FMT = "%Y:%m:%d %H:%M:%S"  # EXIF "YYYY:MM:DD HH:MM:SS"

# Distance thresholds (km) for the GPS-vs-anchor check.
_GPS_NEAR_KM = 25.0
_GPS_FAR_KM = 200.0
# Time thresholds (minutes) for the EXIF-clock-vs-claim check.
_DT_CLOSE_MIN = 30.0
_DT_FAR_MIN = 120.0

#: Honest reliability note stamped on every metadata verdict.
_RELIABILITY = "easily forged; weak anchor only"


# --------------------------------------------------------------------------
# small coercion helpers
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
    """Stripped EXIF string (they arrive NUL/space-padded), or None."""
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


# --------------------------------------------------------------------------
# frame metadata (lazy PIL)
# --------------------------------------------------------------------------

def frame_metadata(path: str) -> dict:
    """Extract EXIF metadata from a JPEG/PNG still. {} on any failure.

    Keys (present only when found): "datetime_original" ("YYYY:MM:DD
    HH:MM:SS"), "gps_lat"/"gps_lon" (signed decimal degrees), "make",
    "model", "software". Never raises: no PIL, unreadable file, or no EXIF
    all return {} — missing metadata is the normal case, not an error.
    """
    try:
        from PIL import Image  # lazy: module must import without PIL
    except ImportError:
        return {}
    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except Exception:
        return {}
    if not exif:
        return {}

    def ifd(tag: int) -> dict:
        try:
            return dict(exif.get_ifd(tag) or {})
        except Exception:
            return {}

    exif_ifd, gps_ifd = ifd(_EXIF_IFD), ifd(_GPS_IFD)
    out: dict = {}

    dt = (_clean_str(exif_ifd.get(_TAG_DATETIME_ORIG))
          or _clean_str(exif.get(_TAG_DATETIME_ORIG))
          or _clean_str(exif.get(_TAG_DATETIME)))
    if dt:
        out["datetime_original"] = dt

    lat = _dms_to_deg(gps_ifd.get(_GPS_LAT), gps_ifd.get(_GPS_LAT_REF))
    lon = _dms_to_deg(gps_ifd.get(_GPS_LON), gps_ifd.get(_GPS_LON_REF))
    if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
        out["gps_lat"], out["gps_lon"] = lat, lon

    for tag, key in ((_TAG_MAKE, "make"), (_TAG_MODEL, "model"),
                     (_TAG_SOFTWARE, "software")):
        s = _clean_str(exif.get(tag))
        if s:
            out[key] = s
    return out


# --------------------------------------------------------------------------
# video metadata (lazy ffprobe subprocess)
# --------------------------------------------------------------------------

def _find_ffprobe() -> str | None:
    """Locate an ffprobe executable; None when absent."""
    return shutil.which("ffprobe")


def video_metadata(path: str) -> dict:
    """Extract container metadata via ffprobe. {} on any failure.

    Keys on success: "creation_time" (ISO, from format tags), "duration_s"
    (float), "encoder" (muxer/encoder tag), "format_tags" (the raw format
    tag dict). Absent ffprobe, non-zero exit, or unparseable JSON -> {}.
    """
    exe = _find_ffprobe()
    if not exe:
        return {}
    try:
        proc = subprocess.run(
            [exe, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    try:
        info = json.loads(proc.stdout or "")
    except ValueError:
        return {}

    fmt = info.get("format") or {}
    tags = fmt.get("tags") or {}
    streams = info.get("streams") or []

    # creation_time / encoder are usually format tags; some muxers put them
    # per-stream instead, so fall back to the first stream that has them.
    def tag_of(*names: str) -> str | None:
        for name in names:
            v = _clean_str(tags.get(name))
            if v:
                return v
        for st in streams:
            st_tags = st.get("tags") or {}
            for name in names:
                v = _clean_str(st_tags.get(name))
                if v:
                    return v
        return None

    duration = _num(fmt.get("duration"))
    if duration is None:  # some containers only carry per-stream durations
        duration = max((_num(s.get("duration")) or 0.0 for s in streams),
                       default=0.0) or None

    return {
        "creation_time": tag_of("creation_time"),
        "duration_s": duration,
        "encoder": tag_of("encoder", "com.apple.quicktime.software"),
        "format_tags": dict(tags),
    }


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (same formula as geo.py's copy)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_meta_datetime(meta: dict) -> tuple[datetime | None, str | None]:
    """Best capture datetime from a meta dict -> (naive dt, source key).

    Prefers EXIF "datetime_original" ("YYYY:MM:DD HH:MM:SS"), then the
    ffprobe "creation_time" (ISO 8601, often "...Z"). Both are read as
    naive local-to-camera time — matching the pipeline's convention of
    comparing claim and overlay clocks directly without timezone math.
    """
    raw = _clean_str(meta.get("datetime_original"))
    if raw:
        try:
            return datetime.strptime(raw, _EXIF_DT_FMT), "datetime_original"
        except ValueError:
            pass  # fall through to creation_time rather than give up
    raw = _clean_str(meta.get("creation_time"))
    if raw:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.replace(tzinfo=None), "creation_time"
        except ValueError:
            return None, "creation_time"
    return None, None


def _gps_signal(meta: dict, anchor: tuple[float, float] | None):
    """-> (verdict, confidence, summary, distance_km|None) or None."""
    if not anchor:
        return None
    lat, lon = meta.get("gps_lat"), meta.get("gps_lon")
    if lat is None or lon is None:
        return None
    try:
        d = _haversine_km(float(lat), float(lon), float(anchor[0]), float(anchor[1]))
    except (TypeError, ValueError, IndexError):
        return None
    if d <= _GPS_NEAR_KM:
        verdict, conf = "SUPPORTS", 0.7
        note = f"EXIF GPS is {d:.1f} km from the claimed place — consistent"
    elif d > _GPS_FAR_KM:
        verdict, conf = "CONTRADICTS", 0.8
        note = f"EXIF GPS is {d:.0f} km from the claimed place — nowhere near"
    else:
        verdict, conf = "INCONCLUSIVE", 0.4
        note = f"EXIF GPS is {d:.0f} km from the claimed place — ambiguous"
    return verdict, conf, note, d


def _time_signal(meta: dict, claim_time_iso: str | None):
    """-> (verdict, confidence, summary, delta_min|None, source) or None."""
    if not claim_time_iso:
        return None
    meta_dt, src = _parse_meta_datetime(meta)
    if meta_dt is None:
        return None
    try:
        claim_dt = datetime.fromisoformat(
            str(claim_time_iso).replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None
    delta_min = abs((meta_dt - claim_dt).total_seconds()) / 60.0
    if meta_dt.date() != claim_dt.date() or delta_min >= _DT_FAR_MIN:
        verdict, conf = "CONTRADICTS", 0.7
        why = ("different date" if meta_dt.date() != claim_dt.date()
               else f"{delta_min:.0f} min off")
        note = f"metadata {src} says {meta_dt.isoformat(' ')} — {why} vs claim"
    elif delta_min <= _DT_CLOSE_MIN:
        verdict, conf = "SUPPORTS", 0.75
        note = (f"metadata {src} says {meta_dt.isoformat(' ')} — "
                f"{delta_min:.0f} min from the claimed time")
    else:
        verdict, conf = "INCONCLUSIVE", 0.4
        note = (f"metadata {src} says {meta_dt.isoformat(' ')} — "
                f"{delta_min:.0f} min from claim; between thresholds")
    return verdict, conf, note, delta_min, src


def meta_evidence(
    meta: dict,
    claim_time_iso: str | None,
    anchor: tuple[float, float] | None,
) -> "Evidence":
    """Turn frame_metadata()/video_metadata() output into an Evidence card.

    Two independent signals are scored:

      * GPS vs `anchor`:      <=25 km SUPPORTS 0.7, >200 km CONTRADICTS 0.8,
                              between INCONCLUSIVE 0.4
      * capture time vs claim: same date & <=30 min SUPPORTS 0.75,
                              different date or >=2 h CONTRADICTS 0.7,
                              else INCONCLUSIVE 0.4

    When both fire, the higher-confidence verdict wins (ties favour
    CONTRADICTS — a mismatch shouldn't be masked by a weak match); the
    losing signal is kept in data["other_signal"]. Nothing usable ->
    INCONCLUSIVE 0.1. All metadata is stamped as easily forged.
    """
    from .schema import ClaimVerdict, Evidence  # lazy: keeps module dep-free

    meta = dict(meta or {})
    data: dict = {"meta": meta, "reliability": _RELIABILITY}
    device = " ".join(
        s for s in (_clean_str(meta.get("make")), _clean_str(meta.get("model"))) if s
    )

    signals = []  # (name, ClaimVerdict, confidence, summary)
    gps = _gps_signal(meta, anchor)
    if gps:
        verdict, conf, note, d = gps
        data["gps_distance_km"] = round(d, 2)
        signals.append(("gps", ClaimVerdict[verdict], conf, note))
    ts = _time_signal(meta, claim_time_iso)
    if ts:
        verdict, conf, note, delta, src = ts
        data["time_delta_minutes"] = round(delta, 1)
        data["time_source"] = src
        signals.append(("time", ClaimVerdict[verdict], conf, note))

    if not signals:
        return Evidence(
            type="metadata",
            summary="no usable metadata (no EXIF GPS/datetime, or nothing "
                    "to check it against)",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.1,
            data=data,
        )

    # Strongest signal leads; on a confidence tie prefer CONTRADICTS.
    best = max(signals, key=lambda s: (s[2], s[1] == ClaimVerdict.CONTRADICTS))
    others = [s for s in signals if s is not best]
    if others:
        data["other_signal"] = {
            "name": others[0][0], "verdict": others[0][1].value,
            "confidence": others[0][2], "summary": others[0][3],
        }
    if device:
        data["device"] = device
    prefix = f"{device}: " if device else ""
    return Evidence(
        type="metadata",
        summary=prefix + best[3],
        verdict=best[1],
        confidence=best[2],
        data=data,
    )


# --------------------------------------------------------------------------
# Demo / self-test
# --------------------------------------------------------------------------

def _self_test() -> int:
    """Parser self-test: DMS->deg, haversine, EXIF dt, verdict logic."""
    fails = 0

    def check(name: str, got, want, tol: float | None = None) -> None:
        nonlocal fails
        ok = (abs(got - want) <= tol) if tol is not None else (got == want)
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r} (want {want!r})")

    check("dms N", _dms_to_deg((51, 30, 0), "N"), 51.5, 1e-9)
    check("dms S", _dms_to_deg((33, 51, 0), "S"), -33.85, 1e-9)
    check("dms W", _dms_to_deg((0, 7, 30), "W"), -0.125, 1e-9)
    check("dms bad", _dms_to_deg(("x", 1, 2), "N"), None)
    check("hav London-Paris km",
          _haversine_km(51.5074, -0.1278, 48.8566, 2.3522), 343.5, 1.0)
    check("hav same point",
          _haversine_km(51.5, -0.12, 51.5, -0.12), 0.0, 1e-9)

    dt, src = _parse_meta_datetime({"datetime_original": "2025:10:15 15:30:00"})
    check("exif dt", dt, datetime(2025, 10, 15, 15, 30, 0))
    check("exif dt src", src, "datetime_original")
    dt, src = _parse_meta_datetime({"creation_time": "2025-10-15T15:30:00.000000Z"})
    check("ffprobe dt", dt, datetime(2025, 10, 15, 15, 30, 0))

    from .schema import ClaimVerdict
    london = (51.5074, -0.1278)
    ev = meta_evidence(
        {"gps_lat": 51.51, "gps_lon": -0.13,
         "datetime_original": "2025:10:15 15:10:00"},
        "2025-10-15T15:00:00", london)
    check("near+ontime -> SUPPORTS", ev.verdict, ClaimVerdict.SUPPORTS)
    check("near+ontime conf", ev.confidence, 0.75)
    check("reliability stamped", ev.data.get("reliability"), _RELIABILITY)

    ev = meta_evidence({"gps_lat": 40.7128, "gps_lon": -74.0060},
                       None, london)  # NYC tag, London claim
    check("far -> CONTRADICTS", ev.verdict, ClaimVerdict.CONTRADICTS)
    check("far conf", ev.confidence, 0.8)

    ev = meta_evidence({"datetime_original": "2025:10:16 09:00:00"},
                       "2025-10-15T15:00:00", None)  # wrong day
    check("wrong date -> CONTRADICTS", ev.verdict, ClaimVerdict.CONTRADICTS)

    ev = meta_evidence({}, None, None)
    check("empty -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    check("empty conf", ev.confidence, 0.1)
    return fails


if __name__ == "__main__":
    fails = _self_test()
    print(f"self-test: {'FAILED' if fails else 'all passed'}")
    if fails:
        sys.exit(1)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        meta = frame_metadata(path)
        if not meta:
            meta = video_metadata(path)
        print(f"\n{path}:")
        print(json.dumps(meta, indent=2, default=str))
    else:
        # Tiny generated JPEG with EXIF (skipped cleanly when PIL absent).
        import tempfile, os
        try:
            from PIL import Image
        except ImportError:
            print("\n(PIL not installed — skipping generated-JPEG demo)")
            sys.exit(0)
        path = os.path.join(tempfile.mkdtemp(prefix="sundial_meta_"), "demo.jpg")
        img = Image.new("RGB", (8, 8), (40, 60, 90))
        exif = Image.Exif()
        exif[_TAG_MAKE] = "Sundial"
        exif[_TAG_MODEL] = "DemoCam 1"
        exif[_TAG_SOFTWARE] = "sundial-demo"
        exif[_TAG_DATETIME] = "2025:10:15 15:05:00"
        try:  # nested IFDs need Pillow >= 9.1; harmless if rejected
            exif[_EXIF_IFD] = {_TAG_DATETIME_ORIG: "2025:10:15 15:05:00"}
            exif[_GPS_IFD] = {0x0000: (2, 3, 0, 0), _GPS_LAT_REF: "N",
                              _GPS_LAT: (51, 30, 26), _GPS_LON_REF: "W",
                              _GPS_LON: (0, 7, 30)}
        except Exception:
            pass
        img.save(path, "JPEG", exif=exif)
        meta = frame_metadata(path)
        print(f"\ngenerated {path}:")
        print(json.dumps(meta, indent=2, default=str))
        ev = meta_evidence(meta, "2025-10-15T15:00:00", (51.5074, -0.1278))
        print(f"\nevidence: {ev.verdict.value} conf={ev.confidence} — {ev.summary}")
