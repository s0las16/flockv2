"""Video container forensics — encoder fingerprints, mux clock, re-mux tells.

camera_meta.video_metadata() already pulls ffprobe's basic format tags for
the pipeline's EXIF/GPS/clock check. This module reads the container itself
more suspiciously — not what the metadata *says* but whether it holds
together as a story about where the file came from:

  * probe()               — lazy ffprobe subprocess -> normalized dict
                            (encoder, creation_time, major_brand, handler
                            names, per-stream codec/fps, ISO6709 location)
  * analyse()             — pure-dict "tells": who muxed the file, how
                            fresh the container is, whether its claimed
                            identity (brand/handlers vs encoder) is coherent
  * video_meta_evidence() — one Evidence card for the pack

Nothing here can ever be strong evidence: every container tag is one
``exiftool``/``ffmpeg -metadata`` command away from fiction, so confidence
is hard-capped at 0.45 and an ffmpeg fingerprint always degrades the card
to INCONCLUSIVE — a re-encode isn't a lie, it *erases* evidence (the mux
clock replaces the capture clock, tags are rewritten, streams rebuilt).

Stdlib only at import; ffprobe/ffmpeg are located lazily with shutil.which
inside functions, so the module imports and self-tests on machines without
them (probe() -> {} when ffprobe is absent, and nothing ever raises).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only — the real import is lazy (see _schema)
    from .schema import Evidence

# --- thresholds ------------------------------------------------------------
_MATCH_MIN = 5.0          # creation_time ≈ claim  -> looks direct-off-camera
_CLOSE_MIN = 120.0        # same-session edit/upload window, still unremarkable
_WILD_MIN = 24.0 * 60.0   # >=1 day of clock disagreement -> "wildly off"
_FRESH_MUX_H = 48.0       # container written within last 48h -> "fresh"
_FUTURE_SLACK_S = 3600.0  # creation_time >now+1h -> clock/tag lies

#: Hard confidence cap — every byte of container metadata is forgeable.
_MAX_CONF = 0.45

#: Honest reliability note stamped on every verdict (cf. camera_meta).
_RELIABILITY = "trivially forged — container tags rewrite with one ffmpeg/exiftool command; weak anchor only"

# --- fingerprints ----------------------------------------------------------
#: Encoder-tag prefixes that mean "ffmpeg wrote this file" (Lavf = muxer,
#: Lavc = codec lib, FFmpeg = raw tool name). ffprobe reports e.g.
#: "Lavf60.3.100"; cameras never produce these.
_FFMPEG_ENCODERS = ("lavf", "lavc", "ffmpeg")
#: Other tools whose tag means the file passed through an editor/transcoder.
_REENCODERS = ("handbrake", "adobe", "premiere", "avid", "davinci", "vegas")

#: Handler-name fingerprints. Apple tooling writes "Core Media *";
#: ffmpeg's mov/mp4 muxer writes "VideoHandler"/"SoundHandler"; Android's
#: MediaMuxer writes "VideoHandle"/"SoundHandle" (no trailing r).
_APPLE_HANDLER_BITS = ("core media", "com.apple", "apple")
_GENERIC_HANDLER_BITS = ("videohand", "soundhand", "subtitlehand",
                         "datahand", "hint")

#: ISO 6709 location string, e.g. "+51.5074-000.1278+015.000/" — iPhone
#: .mov files and explicitly geo-tagged mp4s carry it (rare but gold).
_ISO6709_RE = re.compile(r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


# --------------------------------------------------------------------------
# small coercion helpers (local copies, same shapes as camera_meta's)
# --------------------------------------------------------------------------

def _num(v) -> float | None:
    """float() that tolerates ints/strings; None on anything else."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _clean_str(v) -> str | None:
    """Stripped tag string (they arrive NUL/space-padded), or None."""
    if v is None:
        return None
    s = str(v).strip(" \t\x00")
    return s or None


def _lower_tags(tags) -> dict:
    """Tag dict with lower-cased keys — muxers disagree on casing."""
    return {str(k).lower(): v for k, v in (tags or {}).items()}


def _parse_iso(raw) -> datetime | None:
    """ISO 8601 -> naive datetime (tz stripped).

    Follows the pipeline convention of comparing claim/container clocks
    directly as naive values — container creation_time is usually UTC
    ("...Z") while claims are local-to-place; timezone math is out of
    scope for a weak-forgeable signal.
    """
    s = _clean_str(raw)
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _frame_rate(stream: dict) -> float | None:
    """fps from avg_frame_rate (preferred) or r_frame_rate, "30000/1001" form."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = str(stream.get(key) or "").strip()
        if not raw or raw in ("0", "0/0", "N/A"):
            continue
        if "/" in raw:
            a, _, b = raw.partition("/")
            num, den = _num(a), _num(b)
            if num is not None and den:
                return round(num / den, 3)
        else:
            v = _num(raw)
            if v:
                return round(v, 3)
    return None


def _parse_iso6709(raw: str | None) -> dict | None:
    """"+51.5074-000.1278+015.000/" -> {"lat","lon","raw"} or None."""
    m = _ISO6709_RE.match(raw or "")
    if not m:
        return None
    lat, lon = _num(m.group(1)), _num(m.group(2))
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return {"lat": lat, "lon": lon, "raw": raw}


def _encoder_family(enc: str | None) -> str | None:
    """Classify the encoder/muxer tag: ffmpeg | apple | reencoder | other."""
    e = (enc or "").strip().lower()
    if not e:
        return None
    if e.startswith(_FFMPEG_ENCODERS):
        return "ffmpeg"
    if "com.apple" in e or "quicktime" in e or e.startswith("apple"):
        return "apple"
    if any(k in e for k in _REENCODERS):
        return "reencoder"
    return "other"


def _schema():
    """Lazy schema import — tolerates both package and direct-script use."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # `python sundial/video_meta.py` direct run
        try:
            from schema import ClaimVerdict, Evidence
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


# --------------------------------------------------------------------------
# external binaries (lazy — never required at import)
# --------------------------------------------------------------------------

def _find_ffprobe() -> str | None:
    """Locate an ffprobe executable; None when absent."""
    return shutil.which("ffprobe")


def _find_ffmpeg() -> str | None:
    """Locate ffmpeg — used only by the self-test to synthesize a clip."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def probe(path: str) -> dict:
    """Extract container metadata via ffprobe. {} on any failure.

    Runs ``ffprobe -v quiet -print_format json -show_format -show_streams``
    and normalizes the output to::

        {"creation_time": iso|None, "encoder": str|None,
         "duration_s": float|None,
         "streams": [{"codec","type","w","h","fps"}],
         "major_brand": str|None, "handler_names": [...]}

    plus ``"location": {"lat","lon","raw"}`` when a parseable ISO 6709 tag
    exists (iPhone .mov / geo-tagged mp4). Absent ffprobe, non-zero exit,
    or unparseable JSON all return {} — never raises.
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
    if not isinstance(info, dict):
        return {}

    fmt = info.get("format") or {}
    fmt_tags = _lower_tags(fmt.get("tags"))
    streams_in = [s for s in (info.get("streams") or []) if isinstance(s, dict)]

    # creation_time / encoder are usually format tags; some muxers put them
    # per-stream instead — fall back to the first stream that has them.
    def tag_of(*names: str) -> str | None:
        for name in names:
            v = _clean_str(fmt_tags.get(name))
            if v:
                return v
        for st in streams_in:
            st_tags = _lower_tags(st.get("tags"))
            for name in names:
                v = _clean_str(st_tags.get(name))
                if v:
                    return v
        return None

    handlers: list[str] = []
    stream_tags = [_lower_tags(s.get("tags")) for s in streams_in]
    for st_tags in stream_tags:
        h = _clean_str(st_tags.get("handler_name"))
        if h and h not in handlers:
            handlers.append(h)

    streams = [{
        "codec": _clean_str(s.get("codec_name")),
        "type": _clean_str(s.get("codec_type")),
        "w": _int(s.get("width")),
        "h": _int(s.get("height")),
        "fps": _frame_rate(s),
    } for s in streams_in]

    duration = _num(fmt.get("duration"))
    if duration is None:  # some containers only carry per-stream durations
        duration = max((_num(s.get("duration")) or 0.0 for s in streams_in),
                       default=0.0) or None

    out = {
        "creation_time": tag_of("creation_time",
                                "com.apple.quicktime.creationdate"),
        "encoder": tag_of("encoder", "com.apple.quicktime.software",
                          "encoded_by"),
        "duration_s": duration,
        "streams": streams,
        "major_brand": _clean_str(fmt_tags.get("major_brand")),
        "handler_names": handlers,
    }

    # ISO6709 location tags can hide under several key spellings.
    loc_raw = None
    for tag_dict in [fmt_tags, *stream_tags]:
        for key, val in tag_dict.items():
            if "location" in key or "iso6709" in key:
                loc_raw = _clean_str(val) or loc_raw
    loc = _parse_iso6709(loc_raw)
    if loc:
        out["location"] = loc
    return out


# --------------------------------------------------------------------------
# analyse — pure-dict forensic tells (no schema, no verdicts)
# --------------------------------------------------------------------------

def _origin_hints(brand: str | None, handlers: list[str],
                  enc: str | None, fam: str | None) -> set[str]:
    """Who plausibly authored this container? {"apple","ffmpeg",...}."""
    hints: set[str] = set()
    handler_l = [h.lower() for h in handlers]
    if ((brand or "").strip() == "qt"
            or (enc and "com.apple" in enc.lower())
            or any(any(a in h for a in _APPLE_HANDLER_BITS) for h in handler_l)):
        hints.add("apple")
    if fam == "ffmpeg" or any(
            any(g in h for g in _GENERIC_HANDLER_BITS) for h in handler_l):
        hints.add("ffmpeg/generic")
    return hints


def analyse(meta: dict, claim_time_iso: str | None = None) -> dict:
    """Forensic tells from probe() output. Pure dict in, pure dict out.

    `meta` is a probe() dict (a camera_meta.video_metadata() dict also
    works — format_tags are honoured as a fallback). `claim_time_iso` is
    the claimed filming time; callers may additionally stash a device
    claim under meta["claimed_device"]/"device_claim" ("my Android
    phone") to enable brand-vs-claim oddity checks.

    Tells (``{"kind","detail"}``):

      * ``remux_ffmpeg``         — encoder starts Lavf/Lavc/FFmpeg
      * ``reencoded_software``   — HandBrake/Adobe/etc. encoder tag
      * ``empty_metadata``       — no tags at all: stripped or camera-native
      * ``fresh_mux``            — creation_time ≈ now while claim is old
      * ``muxed_after_claim``    — container finalized >1 day after claim
      * ``claim_after_creation`` — claim is >1 day *later* than the file
      * ``future_creation_time`` — mux clock is set in the future
      * ``identity_mismatch``    — Apple handlers in an ffmpeg container
      * ``forged_software_tag``  — com.apple encoder tag on non-Apple guts
      * ``device_claim_conflict``— fingerprints disagree with claimed device
    """
    meta = dict(meta or {})
    fmt_tags = _lower_tags(meta.get("format_tags"))

    enc = _clean_str(meta.get("encoder"))
    fam = _encoder_family(enc)
    brand = _clean_str(meta.get("major_brand")) or _clean_str(fmt_tags.get("major_brand"))
    handlers = [h for h in
                (_clean_str(x) for x in (meta.get("handler_names") or [])) if h]
    ct_raw = _clean_str(meta.get("creation_time"))
    ct_dt = _parse_iso(ct_raw)
    claim_dt = _parse_iso(claim_time_iso)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    hints = _origin_hints(brand, handlers, enc, fam)
    tells: list[dict] = []

    # --- who wrote the container ------------------------------------------
    if fam == "ffmpeg":
        tells.append({
            "kind": "remux_ffmpeg",
            "detail": f"encoder tag {enc!r} — file was (re-)encoded or "
                      "re-muxed by ffmpeg; the original capture metadata "
                      "was stripped or rewritten",
        })
    elif fam == "reencoder":
        tells.append({
            "kind": "reencoded_software",
            "detail": f"encoder tag {enc!r} — file passed through an "
                      "editor/transcoder; capture metadata likely rewritten",
        })

    # --- completely empty tag block ---------------------------------------
    container_empty = bool(meta) and not (ct_raw or enc or brand or handlers)
    if container_empty:
        tells.append({
            "kind": "empty_metadata",
            "detail": "no creation_time/encoder/brand/handler tags — either "
                      "stripped (privacy tool, messaging app) or camera-native "
                      "(many devices write nothing)",
        })

    # --- container clock vs wall clock ------------------------------------
    age_s = (now - ct_dt).total_seconds() if ct_dt else None
    if ct_dt and age_s is not None and age_s < -_FUTURE_SLACK_S:
        tells.append({
            "kind": "future_creation_time",
            "detail": f"creation_time {-age_s / 3600:.1f}h in the future — "
                      "camera clock mis-set or the tag was planted",
        })
    if ct_dt and claim_dt:
        claim_age_s = (now - claim_dt).total_seconds()
        if (0 <= (age_s or 0) <= _FRESH_MUX_H * 3600
                and claim_age_s > _FRESH_MUX_H * 3600):
            tells.append({
                "kind": "fresh_mux",
                "detail": f"container written ~{age_s / 3600:.0f}h ago but the "
                          f"claimed filming is {claim_age_s / 86400:.1f}d old — "
                          "a fresh re-mux/upload of older content, not the "
                          "original capture",
            })

    # --- container clock vs claimed clock ----------------------------------
    delta_s = None
    relation = None
    if ct_dt and claim_dt:
        delta_s = (ct_dt - claim_dt).total_seconds()
        ad = abs(delta_s)
        if ad <= _MATCH_MIN * 60:
            relation = "matches"
        elif delta_s < 0 and ad >= _WILD_MIN * 60:
            relation = "claim_after_creation"
            tells.append({
                "kind": "claim_after_creation",
                "detail": f"claim is {ad / 3600:.1f}h AFTER the file's "
                          "creation_time — a file can't predate its own "
                          "footage; the claim or the tags are fabricated",
            })
        elif delta_s > 0 and ad >= _WILD_MIN * 60:
            relation = "muxed_after_claim"
            tells.append({
                "kind": "muxed_after_claim",
                "detail": f"container finalized {ad / 3600:.1f}h after the "
                          "claimed filming — re-muxed/re-exported long "
                          "after capture",
            })
        elif ad <= _CLOSE_MIN * 60:
            relation = "close"
        else:
            relation = "off"

    # --- identity oddities -------------------------------------------------
    handler_l = [h.lower() for h in handlers]
    apple_handlers = any(
        any(a in h for a in _APPLE_HANDLER_BITS) for h in handler_l)
    generic_handlers = any(
        any(g in h for g in _GENERIC_HANDLER_BITS) for h in handler_l)
    if apple_handlers and fam == "ffmpeg":
        tells.append({
            "kind": "identity_mismatch",
            "detail": "Apple 'Core Media' handler names inside an "
                      "ffmpeg-written container — the mux identity doesn't "
                      "add up (doctored tags)",
        })
    if enc and "com.apple" in enc.lower() and generic_handlers and not apple_handlers:
        tells.append({
            "kind": "forged_software_tag",
            "detail": f"software tag {enc!r} claims QuickTime authorship but "
                      "the stream handlers look like ffmpeg's — the encoder "
                      "tag was planted on a re-muxed file",
        })

    claimed_dev = _clean_str(meta.get("claimed_device")
                             or meta.get("device_claim"))
    if claimed_dev:
        dl = claimed_dev.lower()
        wants_android = any(
            k in dl for k in ("android", "samsung", "pixel", "xiaomi",
                              "huawei", "oppo", "oneplus"))
        wants_apple = any(k in dl for k in ("iphone", "ipad", "apple", "ios"))
        apple_file = "apple" in hints
        generic_file = (fam == "ffmpeg" or (brand or "") in
                        ("isom", "mp42", "mp41", "3gp4", "3gp5", "3gp6"))
        if wants_android and apple_file:
            tells.append({
                "kind": "device_claim_conflict",
                "detail": f"claim says {claimed_dev!r} but the container "
                          "fingerprints as Apple-authored (brand/handlers/"
                          "software tag)",
            })
        elif wants_apple and not apple_file and generic_file:
            tells.append({
                "kind": "device_claim_conflict",
                "detail": f"claim says {claimed_dev!r} but nothing in the "
                          f"container looks Apple-authored (brand={brand or 'none'}, "
                          f"encoder={enc or 'none'})",
            })

    return {
        "probed": bool(meta),
        "creation_time": ct_raw,
        "creation_age_h": round(age_s / 3600, 1) if age_s is not None else None,
        "encoder": enc,
        "encoder_family": fam,
        "major_brand": brand,
        "handler_names": handlers,
        "origin_hints": sorted(hints),
        "claim_delta_s": delta_s,
        "claim_delta_minutes": round(abs(delta_s) / 60, 1) if delta_s is not None else None,
        "time_relation": relation,
        "container_empty": container_empty,
        "tells": tells,
        "duration_s": _num(meta.get("duration_s")),
        "streams": meta.get("streams") or [],
        "location": meta.get("location"),
        "summary": "; ".join(t["detail"] for t in tells) or
                   ("no usable metadata" if not meta
                    else "no forensic tells — container looks ordinary"),
    }


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (same formula as geo.py's copy)."""
    import math  # stdlib, tiny — local keeps module import surface minimal
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _time_signal(rep: dict):
    """-> (verdict, confidence, summary) or None, from analyse() output."""
    dm = rep.get("claim_delta_minutes")
    rel = rep.get("time_relation")
    if dm is None or rel is None:
        return None
    if rel == "matches":
        return ("SUPPORTS", _MAX_CONF,
                f"container creation_time is {dm:.0f} min from the claim — "
                "consistent with a straight off-camera file")
    if rel in ("claim_after_creation", "muxed_after_claim"):
        return ("CONTRADICTS", 0.4,
                f"container clock is {dm / 60:.1f}h off the claim "
                f"({rel.replace('_', ' ')}) — weak: metadata lies")
    return ("INCONCLUSIVE", 0.3,
            f"creation_time is {dm / 60:.1f}h from the claim — normal "
            "for edited/re-uploaded video")


def _location_signal(meta: dict, anchor):
    """-> (verdict, confidence, summary, km) or None, via ISO6709 tag."""
    loc = meta.get("location")
    if not anchor or not isinstance(loc, dict):
        return None
    try:
        d = _haversine_km(float(loc["lat"]), float(loc["lon"]),
                          float(anchor[0]), float(anchor[1]))
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    if d <= 25.0:
        return ("SUPPORTS", _MAX_CONF,
                f"container ISO6709 location is {d:.1f} km from the "
                "claimed place — consistent", d)
    if d > 200.0:
        return ("CONTRADICTS", 0.4,
                f"container ISO6709 location is {d:.0f} km from the "
                "claimed place — nowhere near", d)
    return ("INCONCLUSIVE", 0.3,
            f"container ISO6709 location is {d:.0f} km from the claimed "
            "place — ambiguous", d)


def video_meta_evidence(
    meta: dict,
    claim_time_iso: str | None,
    anchor: tuple[float, float] | None = None,
) -> "Evidence":
    """Turn probe() output into one Evidence card.

    Scoring (all confidences hard-capped at _MAX_CONF = 0.45 — container
    metadata is the easiest forgery surface in the whole file):

      * creation_time within 5 min of claim -> SUPPORTS 0.45
      * container clock >=1 day off, either direction -> CONTRADICTS 0.4
        (weak — the tag itself may be the lie)
      * ISO6709 location tag vs anchor: <=25 km SUPPORTS 0.45,
        >200 km CONTRADICTS 0.4, between INCONCLUSIVE 0.3
      * ANY re-mux/re-encode tell -> INCONCLUSIVE: re-encoding isn't
        lying, it strips evidence — the mux clock simply can't testify
        about capture. Sub-signals are preserved in data["signals"].
      * nothing usable -> INCONCLUSIVE 0.15
    """
    ClaimVerdict, Evidence = _schema()

    meta = dict(meta or {})
    rep = analyse(meta, claim_time_iso)
    data: dict = {"meta": meta, "analysis": rep, "reliability": _RELIABILITY}

    signals = []  # (name, ClaimVerdict, confidence, summary)
    ts = _time_signal(rep)
    if ts:
        signals.append(("time", ClaimVerdict[ts[0]], ts[1], ts[2]))
    ls = _location_signal(meta, anchor)
    if ls:
        data["location_distance_km"] = round(ls[3], 2)
        signals.append(("location", ClaimVerdict[ls[0]], ls[1], ls[2]))
    if signals:
        data["signals"] = [
            {"name": n, "verdict": v.value, "confidence": c, "summary": s}
            for n, v, c, s in signals
        ]

    # Re-mux dominates: once ffmpeg wrote the container, its clock and tags
    # describe the mux, not the capture — everything else is hearsay.
    remux = next((t for t in rep["tells"]
                  if t["kind"] in ("remux_ffmpeg", "reencoded_software")), None)
    if remux:
        note = remux["detail"]
        if signals:
            note += f" (underlying check would say: {signals[0][3]})"
        return Evidence(
            type="video_meta",
            summary=f"re-mux tell — {note}",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.35,
            data=data,
        )

    if not signals:
        if rep["container_empty"]:
            why = ("container carries no metadata at all — stripped by a "
                   "tool/app or a camera-native file")
        elif not meta:
            why = "ffprobe unavailable or file unreadable — nothing to check"
        else:
            why = "no usable container metadata (no clock/location to check)"
        return Evidence(
            type="video_meta",
            summary=why,
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.15,
            data=data,
        )

    # Strongest signal leads; on a confidence tie prefer CONTRADICTS so a
    # mismatch isn't masked by a weak match (same rule as camera_meta).
    best = max(signals, key=lambda s: (s[2], s[1] == ClaimVerdict.CONTRADICTS))
    others = [s for s in signals if s is not best]
    if others:
        data["other_signal"] = {
            "name": others[0][0], "verdict": others[0][1].value,
            "confidence": others[0][2], "summary": others[0][3],
        }
    return Evidence(
        type="video_meta",
        summary=best[3],
        verdict=best[1],
        confidence=min(best[2], _MAX_CONF),
        data=data,
    )


# --------------------------------------------------------------------------
# self-test: canned-meta analyse() asserts + guarded probe()/ffmpeg synth
# --------------------------------------------------------------------------

def _tell_kinds(rep: dict) -> list[str]:
    return [t["kind"] for t in rep.get("tells", [])]


def _self_test() -> int:
    fails = 0

    def check(name: str, got, want, tol: float | None = None) -> None:
        nonlocal fails
        ok = (got is not None and abs(got - want) <= tol) if tol is not None else (got == want)
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r} (want {want!r})")

    ClaimVerdict, _ = _schema()

    # --- canned: ffmpeg re-mux ---------------------------------------------
    lavf = {
        "creation_time": "2020-05-10T15:04:00.000000Z",
        "encoder": "Lavf58.29.100",
        "duration_s": 12.5,
        "major_brand": "isom",
        "handler_names": ["VideoHandler", "SoundHandler"],
        "streams": [
            {"codec": "h264", "type": "video", "w": 1920, "h": 1080, "fps": 29.97},
            {"codec": "aac", "type": "audio", "w": None, "h": None, "fps": None},
        ],
    }
    rep = analyse(lavf, "2020-05-10T15:00:00")
    check("lavf family", rep["encoder_family"], "ffmpeg")
    check("lavf remux tell", "remux_ffmpeg" in _tell_kinds(rep), True)
    check("lavf delta min", rep["claim_delta_minutes"], 4.0, 0.01)
    check("lavf relation", rep["time_relation"], "matches")
    ev = video_meta_evidence(lavf, "2020-05-10T15:00:00")
    check("remux -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    check("remux conf", ev.confidence, 0.35)
    check("reliability stamped", bool(ev.data.get("reliability")), True)

    # --- canned: iPhone-native, clock matches claim --------------------------
    iphone = {
        "creation_time": "2025-10-15T15:04:30.000000Z",
        "encoder": "com.apple.quicktime.software 17.4.1",
        "duration_s": 30.0,
        "major_brand": "qt  ",
        "handler_names": ["Core Media Video", "Core Media Audio"],
        "streams": [{"codec": "hevc", "type": "video", "w": 3840, "h": 2160, "fps": 30.0}],
    }
    rep = analyse(iphone, "2025-10-15T15:00:00")
    check("apple family", rep["encoder_family"], "apple")
    check("apple no remux", "remux_ffmpeg" not in _tell_kinds(rep), True)
    check("apple relation", rep["time_relation"], "matches")
    ev = video_meta_evidence(iphone, "2025-10-15T15:00:00")
    check("match -> SUPPORTS", ev.verdict, ClaimVerdict.SUPPORTS)
    check("match conf capped", ev.confidence, 0.45)

    # --- canned: same iPhone file, claim 4 days earlier ----------------------
    rep = analyse(iphone, "2025-10-11T15:00:00")
    check("old-claim relation", rep["time_relation"], "muxed_after_claim")
    check("old-claim tell", "muxed_after_claim" in _tell_kinds(rep), True)
    ev = video_meta_evidence(iphone, "2025-10-11T15:00:00")
    check("wild -> CONTRADICTS", ev.verdict, ClaimVerdict.CONTRADICTS)
    check("wild conf", ev.confidence, 0.4)

    # --- canned: claim later than the file itself ----------------------------
    rep = analyse(iphone, "2025-10-20T15:00:00")
    check("claim>file tell", "claim_after_creation" in _tell_kinds(rep), True)
    ev = video_meta_evidence(iphone, "2025-10-20T15:00:00")
    check("claim>file -> CONTRADICTS", ev.verdict, ClaimVerdict.CONTRADICTS)

    # --- canned: forged QuickTime tag on ffmpeg guts --------------------------
    forged = dict(iphone)
    forged["handler_names"] = ["VideoHandler", "SoundHandler"]
    rep = analyse(forged, "2025-10-15T15:00:00")
    check("forged tag tell", "forged_software_tag" in _tell_kinds(rep), True)

    # --- canned: device claim conflict ---------------------------------------
    dev = dict(iphone)
    dev["claimed_device"] = "my Samsung Android phone"
    rep = analyse(dev, "2025-10-15T15:00:00")
    check("android-vs-apple tell", "device_claim_conflict" in _tell_kinds(rep), True)

    # --- canned: stripped/camera-native container ------------------------------
    stripped = {"duration_s": 5.0, "streams": [
        {"codec": "h264", "type": "video", "w": 1280, "h": 720, "fps": 30.0}]}
    rep = analyse(stripped, "2025-10-15T15:00:00")
    check("empty flagged", rep["container_empty"], True)
    check("empty tell", "empty_metadata" in _tell_kinds(rep), True)
    ev = video_meta_evidence(stripped, "2025-10-15T15:00:00")
    check("empty -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)

    # --- canned: nothing at all ----------------------------------------------
    rep = analyse({}, None)
    check("empty-in tells", rep["tells"], [])
    check("empty-in probed", rep["probed"], False)
    ev = video_meta_evidence({}, None)
    check("nothing -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    check("nothing conf", ev.confidence, 0.15)

    # --- canned: fresh mux of an old clip (dynamic clock) ----------------------
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    fresh = {
        "creation_time": now.isoformat() + "Z",
        "major_brand": "mp42",
        "handler_names": ["VideoHandle"],
        "streams": [{"codec": "h264", "type": "video", "w": 1280, "h": 720, "fps": 30.0}],
    }
    rep = analyse(fresh, (now - timedelta(days=10)).isoformat())
    check("fresh-mux tell", "fresh_mux" in _tell_kinds(rep), True)
    ev = video_meta_evidence(fresh, (now - timedelta(days=10)).isoformat())
    check("fresh-mux no remux enc -> CONTRADICTS", ev.verdict, ClaimVerdict.CONTRADICTS)

    # --- guarded: real ffprobe on a synthesized clip --------------------------
    ffprobe, ffmpeg = _find_ffprobe(), _find_ffmpeg()
    if not ffprobe:
        print("  (ffprobe absent — live probe() checks skipped)")
    elif not ffmpeg:
        print("  (ffmpeg absent — clip synthesis skipped)")
        check("probe missing file -> {}", probe("/nonexistent/clip.mp4"), {})
    else:
        tmp = tempfile.mkdtemp(prefix="sundial_vmeta_")
        clip = os.path.join(tmp, "synth.mp4")
        try:
            r = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
                 "-an", "-c:v", "mpeg4", clip],
                capture_output=True, text=True, timeout=120)
            ok = r.returncode == 0 and os.path.getsize(clip) > 200
        except (OSError, subprocess.SubprocessError):
            ok = False
        if not ok:
            print("  (ffmpeg synthesis failed — live probe() checks skipped)")
        else:
            m = probe(clip)
            check("probe streams", bool(m.get("streams")), True)
            check("probe codec", (m.get("streams") or [{}])[0].get("codec"), "mpeg4")
            check("probe encoder Lavf",
                  str(m.get("encoder") or "").startswith("Lavf"), True)
            check("probe brand", m.get("major_brand"), "isom")
            check("probe duration", m.get("duration_s"), 2.0, 0.5)
            rep = analyse(m, None)
            check("synth remux tell", "remux_ffmpeg" in _tell_kinds(rep), True)
            ev = video_meta_evidence(m, "2020-01-01T00:00:00")
            check("synth -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
            check("probe missing file -> {}", probe("/nonexistent/clip.mp4"), {})
    return fails


if __name__ == "__main__":
    fails = _self_test()
    print(f"self-test: {'FAILED' if fails else 'all passed'}")
    if fails:
        sys.exit(1)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        meta = probe(path)
        print(f"\n{path}:")
        print(json.dumps(meta, indent=2, default=str))
        if meta:
            rep = analyse(meta, None)
            print(f"\ntells: {rep['summary']}")
            ev = video_meta_evidence(meta, None)
            print(f"evidence: {ev.verdict.value} conf={ev.confidence} — {ev.summary}")
    else:
        # Demo on a synthesized clip when ffmpeg+ffprobe exist.
        ffprobe, ffmpeg = _find_ffprobe(), _find_ffmpeg()
        if not (ffprobe and ffmpeg):
            print("\n(ffmpeg/ffprobe not installed — skipping live probe demo)")
            sys.exit(0)
        tmp = tempfile.mkdtemp(prefix="sundial_vmeta_")
        clip = os.path.join(tmp, "demo.mp4")
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
             "-an", "-c:v", "mpeg4", clip],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"\n(ffmpeg synthesis failed: {r.stderr.strip()})")
            sys.exit(0)
        meta = probe(clip)
        print(f"\nsynthesized {clip}:")
        print(json.dumps(meta, indent=2, default=str))
        rep = analyse(meta, "2020-01-01T00:00:00")
        print(f"\ntells: {rep['summary']}")
        ev = video_meta_evidence(meta, "2020-01-01T00:00:00")
        print(f"evidence: {ev.verdict.value} conf={ev.confidence} — {ev.summary}")
