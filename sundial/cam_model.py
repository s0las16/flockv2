"""Device-fingerprint forensics — does the file look like the claimed device?

camera_meta reads what a file *says*; video_meta checks whether the
container's story is coherent. This module asks a different question: does
the encoder/container signature resemble what the *claimed capture device*
actually produces?

  * iPhones write QuickTime containers: major_brand "qt  ", "Core Media *"
    handlers, com.apple.quicktime.* tags, often an ISO6709 location.
  * Android's MediaMuxer writes "VideoHandle"/"SoundHandle" — no trailing
    'r', which is how you tell it apart from ffmpeg's "VideoHandler".
  * GoPro stamps "GoPro" handlers and a gpmd telemetry stream; DJI writes
    DJI.* handlers; Canon/Nikon carry their make/model strings.
  * Screen recordings have no camera provenance at all — AVFoundation/h264
    guts, odd (often square-ish or retina-crop) frame geometry, no
    make/model/creation tags.

API:

  * DEVICE_SIGS            — per-device signature table
  * fingerprint(meta)      — {"device_guess", "confidence",
                             "matched_signals"} from a camera_meta
                             .video_metadata()- or video_meta.probe()-
                             shaped dict (merged/EXIF dicts also work)
  * device_evidence()      — one Evidence card testing the fingerprint
                             against the claimed device
  * sig_for(device)        — signature row lookup; tolerates aliases

All of this is forgeable — one ffmpeg command rewrites every tag — so
confidences stay modest, a reliability note is stamped on every verdict,
and "the file wasn't written by that device family" is phrased as what it
is: a story problem, possibly innocent (re-mux), not proof of lying.

Stdlib only at import; the schema import is lazy; nothing raises.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only — real import is lazy (see _schema)
    from .schema import Evidence

# --- scoring weights (sum to 1.0) ------------------------------------------
_W_ENCODER = 0.30   # encoder/software tag — most device-specific surface
_W_HANDLER = 0.25   # muxer handler names — hard to fake without a re-mux
_W_BRAND = 0.20     # make/model/major_brand strings
_W_TAGS = 0.10      # presence of device-typical tag keys
_W_RES = 0.075      # resolution hint — many devices share these, weak
_W_FPS = 0.075      # fps hint — weak

_HARD_FLOOR = 0.30  # confidence >= this implies >=1 hard (enc/handler/brand) hit
_SOFT_CAP = 0.29    # soft-hints-only matches can never reach _HARD_FLOOR

_FPS_TOL = 0.6      # 29.97 ~ 30, 23.976 ~ 24, 59.94 ~ 60
_STD_AR = (16 / 9, 4 / 3, 3 / 2, 17 / 9, 64 / 27, 2.39)  # normal frame aspects

#: Honest reliability note, same convention as camera_meta / video_meta.
_RELIABILITY = ("trivially forged — one ffmpeg/exiftool command rewrites "
                "every tag; a mismatch is a story problem, not proof")

#: Format/stream tag *keys* that mean "a camera wrote provenance here" —
#: their absence is part of the screen-recording signature.
_CAMERA_TAG_BITS = ("quicktime", "creation", "iso6709", "location", "make",
                    "model", "manufact", "firmware")


# --------------------------------------------------------------------------
# DEVICE_SIGS — the signature table
#
# Pattern conventions:
#   * encoder_patterns match (word-boundary, case-insensitive) against the
#     encoder/software tag text. A "codec:" prefix matches stream codec
#     names instead (e.g. GoPro's gpmd telemetry stream).
#   * handler_patterns match against stream handler names — boundary-aware,
#     so MediaMuxer's "VideoHandle" does NOT hit ffmpeg's "VideoHandler".
#   * brand_hints match against major_brand + make/model/manufacturer-ish
#     tag values (prefix match allowed, so "sm-" catches "SM-G991B").
#   * resolution_hints are "WxH" strings; portrait is the swapped match.
#   * fps_hints match within _FPS_TOL.
#   * typical_tags are substring matches on tag-key names.
# --------------------------------------------------------------------------

DEVICE_SIGS: dict = {
    "iphone": {
        "encoder_patterns": ["com.apple.quicktime", "com.apple",
                             "quicktime"],
        "handler_patterns": ["core media", "apple"],
        "resolution_hints": ["3840x2160", "1920x1080", "1280x720"],
        "fps_hints": [24.0, 25.0, 30.0, 60.0, 120.0, 240.0],
        "brand_hints": ["qt", "apple", "iphone", "ipad"],
        "typical_tags": ["com.apple.quicktime", "iso6709", "location"],
    },
    "android_samsung": {
        "encoder_patterns": ["samsung", "mediamuxer", "android"],
        "handler_patterns": ["videohandle", "soundhandle", "samsung"],
        "resolution_hints": ["3840x2160", "2560x1440", "1920x1080",
                             "1280x720"],
        "fps_hints": [24.0, 25.0, 30.0, 60.0, 120.0, 240.0],
        "brand_hints": ["samsung", "galaxy", "sm-", "sch-", "gt-"],
        "typical_tags": ["samsung", "sef", "com.android", "model"],
    },
    "android_pixel": {
        "encoder_patterns": ["google", "mediamuxer", "android"],
        "handler_patterns": ["videohandle", "soundhandle", "google"],
        "resolution_hints": ["3840x2160", "1920x1080", "1280x720"],
        "fps_hints": [24.0, 30.0, 60.0, 120.0, 240.0],
        "brand_hints": ["google", "pixel"],
        "typical_tags": ["com.google", "android", "model"],
    },
    "gopro": {
        "encoder_patterns": ["gopro", "codec:gpmd"],
        "handler_patterns": ["gopro"],
        "resolution_hints": ["5312x2988", "3840x2160", "2704x1520",
                             "2560x1440", "2560x1080", "1920x1080",
                             "1280x720"],
        "fps_hints": [24.0, 25.0, 30.0, 50.0, 60.0, 100.0, 120.0, 200.0,
                      240.0],
        "brand_hints": ["gopro", "hero"],
        "typical_tags": ["gopro", "firmware", "hilight", "lens", "settings"],
    },
    "dji_drone": {
        "encoder_patterns": ["dji"],
        "handler_patterns": ["dji"],
        "resolution_hints": ["5120x2700", "4096x2160", "3840x2160",
                             "2720x1530", "2688x1512", "1920x1080"],
        "fps_hints": [24.0, 25.0, 30.0, 50.0, 60.0, 100.0, 120.0],
        "brand_hints": ["dji", "mavic", "phantom", "inspire", "osmo"],
        "typical_tags": ["dji"],
    },
    "dslr_canon": {
        "encoder_patterns": ["canon"],
        "handler_patterns": ["canon"],
        "resolution_hints": ["4096x2160", "3840x2160", "1920x1080",
                             "1280x720", "640x480"],
        "fps_hints": [24.0, 25.0, 30.0, 50.0, 60.0, 100.0, 120.0],
        "brand_hints": ["canon", "eos", "powershot", "rebel"],
        "typical_tags": ["canon", "model"],
    },
    "dslr_nikon": {
        "encoder_patterns": ["nikon"],
        "handler_patterns": ["nikon"],
        "resolution_hints": ["3840x2160", "1920x1080", "1280x720",
                             "640x424"],
        "fps_hints": [24.0, 25.0, 30.0, 50.0, 60.0, 120.0],
        "brand_hints": ["nikon", "nikkor", "coolpix"],
        "typical_tags": ["nikon", "model"],
    },
    "screen_recording": {
        "encoder_patterns": ["avfoundation", "screenflow", "camtasia",
                             "obs", "replaykit", "x264", "screencap",
                             "kap", "loom", "snagit", "bandicam",
                             "quicktime player"],
        "handler_patterns": ["core media", "avfoundation"],
        "resolution_hints": [],  # any geometry — see _screen_bonus
        "fps_hints": [60.0],
        "brand_hints": [],
        "typical_tags": [],
    },
    "dashcam_generic": {
        "encoder_patterns": ["ambarella", "novatek", "mstar", "garmin",
                             "blackvue", "thinkware", "viofo", "nextbase",
                             "iroad", "70mai", "vantrue", "apeman",
                             "rexing", "dashcam", "car dvr"],
        "handler_patterns": ["ambarella", "novatek", "dashcam", "garmin",
                             "blackvue", "viofo"],
        "resolution_hints": ["3840x2160", "2560x1440", "2304x1296",
                             "2592x1520", "2560x1080", "1920x1080",
                             "1280x720"],
        "fps_hints": [15.0, 20.0, 25.0, 27.0, 30.0, 60.0],
        "brand_hints": ["garmin", "blackvue", "thinkware", "viofo",
                        "nextbase", "70mai", "vantrue", "rexing", "apeman",
                        "dashcam", "dvr"],
        "typical_tags": ["gps", "gsensor", "g-sensor", "speed", "firmware"],
    },
    "unknown": {
        "encoder_patterns": [],
        "handler_patterns": [],
        "resolution_hints": [],
        "fps_hints": [],
        "brand_hints": [],
        "typical_tags": [],
    },
}

_SIG_KEYS = ("encoder_patterns", "handler_patterns", "resolution_hints",
             "fps_hints", "brand_hints", "typical_tags")

#: Coarse ecosystem per device key — a guess and a claim in different
#: families is the CONTRADICTS case (apple file, android claim).
_FAMILY = {
    "iphone": "apple",
    "android_samsung": "android",
    "android_pixel": "android",
    "gopro": "gopro",
    "dji_drone": "dji",
    "dslr_canon": "canon",
    "dslr_nikon": "nikon",
    "screen_recording": "screen",
    "dashcam_generic": "dashcam",
    "unknown": "unknown",
}

#: Free-text device claims -> signature keys. Specific rules win over the
#: generic buckets (so "samsung android phone" pins android_samsung only).
_SPECIFIC_CLAIMS = [
    (("iphone", "ipad", "apple"), "iphone"),
    (("pixel",), "android_pixel"),
    (("samsung", "galaxy", "sm-g", "sm-s", "sm-n", "sch-", "gt-"),
     "android_samsung"),
    (("gopro", "hero"), "gopro"),
    (("dji", "mavic", "phantom", "inspire", "osmo"), "dji_drone"),
    (("canon", "eos", "powershot", "rebel"), "dslr_canon"),
    (("nikon", "coolpix"), "dslr_nikon"),
    (("screen", "screencast", "screen capture"), "screen_recording"),
    (("dashcam", "dash cam", "dash-cam", "garmin", "blackvue", "thinkware",
      "viofo", "nextbase", "70mai", "vantrue", "rexing", "apeman"),
     "dashcam_generic"),
]
_GENERIC_CLAIMS = [
    (("android", "xiaomi", "huawei", "oppo", "oneplus", "motorola",
      "redmi", "xperia"), {"android_samsung", "android_pixel"}),
    (("phone", "smartphone", "mobile"),
     {"iphone", "android_samsung", "android_pixel"}),
    (("drone",), {"dji_drone"}),
    (("dslr", "mirrorless", "slr"), {"dslr_canon", "dslr_nikon"}),
    (("action cam", "actioncam", "action camera"), {"gopro"}),
]


# --------------------------------------------------------------------------
# small coercion helpers (same shapes as camera_meta/video_meta copies)
# --------------------------------------------------------------------------

def _num(v) -> float | None:
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
    if not isinstance(tags, dict):
        return {}
    return {str(k).lower(): v for k, v in tags.items()}


def _schema():
    """Lazy schema import — tolerates both package and direct-script use."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # `python sundial/cam_model.py` direct run
        try:
            from schema import ClaimVerdict, Evidence
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


# --------------------------------------------------------------------------
# pattern matching
# --------------------------------------------------------------------------

def _hit(pattern: str, text: str, prefix_ok: bool = False) -> bool:
    """Word-boundary substring match, case-insensitive.

    Boundary awareness is load-bearing: MediaMuxer's "VideoHandle" must not
    hit ffmpeg's "VideoHandler", and "apple" must not hit "pineapple".
    With prefix_ok=True a text that merely *starts with* the pattern also
    counts — used for brand strings so "sm-" catches model "SM-G991B".
    """
    p, t = (pattern or "").lower(), (text or "").lower()
    if not p or not t:
        return False
    i = t.find(p)
    while i >= 0:
        j = i + len(p)
        before_ok = i == 0 or not t[i - 1].isalnum()
        after_ok = j >= len(t) or not t[j].isalnum()
        if before_ok and (after_ok or (prefix_ok and i == 0)):
            return True
        i = t.find(p, i + 1)
    return False


def _any_hit(patterns, texts, prefix_ok: bool = False) -> str | None:
    """First pattern that word-boundary-matches any text, else None."""
    for pat in patterns or []:
        for t in texts or []:
            if _hit(str(pat), str(t), prefix_ok=prefix_ok):
                return str(pat)
    return None


def _fps_val(raw) -> float | None:
    """fps from a float/int or an ffprobe fraction like "30000/1001"."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("0", "0/0", "N/A"):
        return None
    if "/" in s:
        a, _, b = s.partition("/")
        num, den = _num(a), _num(b)
        if num is not None and den:
            return round(num / den, 3)
        return None
    return _num(s)


def _parse_res_hint(hint) -> tuple[int, int] | None:
    """"3840x2160" -> (3840, 2160), or None."""
    try:
        w, _, h = str(hint).partition("x")
        return int(w), int(h)
    except (TypeError, ValueError):
        return None


def _odd_geometry(w: int | None, h: int | None) -> bool:
    """Frame geometry no normal camera produces: square-ish (<1.2 aspect)
    or a non-standard aspect (retina screen crops, e.g. 1512x982)."""
    if not w or not h or w <= 0 or h <= 0:
        return False
    ar = max(w, h) / min(w, h)
    if ar < 1.2:
        return True
    return not any(abs(ar - a) < 0.03 for a in _STD_AR)


# --------------------------------------------------------------------------
# feature extraction — tolerant of probe(), video_metadata() and merged
# EXIF shapes; raw ffprobe stream dicts also work
# --------------------------------------------------------------------------

def _pick_video(streams: list) -> dict | None:
    for s in streams:
        if str(s.get("codec_type") or s.get("type") or "").lower() == "video":
            return s
    for s in streams:
        if _int(s.get("width") or s.get("w")):
            return s
    return None


def _features(meta: dict) -> dict:
    """Normalize a meta dict into the haystacks the signature table scores."""
    fmt = _lower_tags(meta.get("format_tags"))
    streams = [s for s in (meta.get("streams") or []) if isinstance(s, dict)]
    stream_tags = [_lower_tags(s.get("tags")) for s in streams]

    # encoder-ish strings: the tag itself plus any tag key containing
    # "encoder"/"software"/"encoded" (com.apple.quicktime.software etc.)
    enc_texts = [_clean_str(meta.get("encoder")),
                 _clean_str(meta.get("software"))]
    handlers = [h for h in
                (_clean_str(x) for x in (meta.get("handler_names") or []))
                if h]
    brand_texts = [_clean_str(meta.get("major_brand")),
                   _clean_str(meta.get("make")),
                   _clean_str(meta.get("model"))]
    tag_keys = [str(k).lower() for k in meta.keys()]

    for tagd in [fmt, *stream_tags]:
        for k, v in tagd.items():
            tag_keys.append(k)
            s = _clean_str(v)
            if not s:
                continue
            if any(b in k for b in ("encoder", "software", "encoded")):
                enc_texts.append(s)
            if "handler" in k and s not in handlers:
                handlers.append(s)
            if any(b in k for b in
                   ("major_brand", "make", "model", "manufact", "device",
                    "brand")):
                brand_texts.append(s)

    codecs = [c for c in
              (_clean_str(s.get("codec") or s.get("codec_name"))
               for s in streams) if c]
    vid = _pick_video(streams)
    w = _int((vid or {}).get("w") or (vid or {}).get("width")
             or meta.get("width"))
    h = _int((vid or {}).get("h") or (vid or {}).get("height")
             or meta.get("height"))
    fps = _fps_val((vid or {}).get("fps"))
    if fps is None and vid:
        fps = _fps_val(vid.get("avg_frame_rate") or vid.get("r_frame_rate"))
    if fps is None:
        fps = _fps_val(meta.get("fps"))
    codec = _clean_str((vid or {}).get("codec")
                       or (vid or {}).get("codec_name"))

    has_cam = any(
        _clean_str(meta.get(k)) or _num(meta.get(k)) is not None
        for k in ("make", "model", "creation_time", "datetime_original",
                  "location", "gps_lat", "gps_lon", "firmware"))
    has_cam = has_cam or any(
        any(b in k for b in _CAMERA_TAG_BITS) for k in tag_keys)

    appleish = any(_hit("core media", x) for x in handlers) or any(
        _hit("qt", x) or _hit("apple", x) for x in brand_texts)

    return {
        "encoder_texts": [e for e in enc_texts if e],
        "handlers": handlers,
        "brand_texts": [b for b in brand_texts if b],
        "tag_keys": sorted(set(tag_keys)),
        "codecs": codecs,
        "has_video": vid is not None or (w is not None and h is not None),
        "w": w, "h": h, "fps": fps, "codec": (codec or "").lower(),
        "camera_meta": has_cam,
        "appleish": appleish,
        "odd_geo": _odd_geometry(w, h),
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _score(feat: dict, sig: dict) -> tuple[float, list[str]]:
    """Weighted signature match -> (score, matched-signal strings)."""
    score, hits = 0.0, []

    enc_pats, codec_pats = [], []
    for p in sig.get("encoder_patterns") or []:
        p = str(p)
        (codec_pats if p.startswith("codec:") else enc_pats).append(
            p[6:] if p.startswith("codec:") else p)

    m = _any_hit(enc_pats, feat["encoder_texts"])
    if m:
        score += _W_ENCODER
        hits.append(f'encoder~"{m}"')
    m = _any_hit(codec_pats, feat["codecs"])
    if m:
        score += _W_ENCODER
        hits.append(f'codec~"{m}"')
    m = _any_hit(sig.get("handler_patterns"), feat["handlers"])
    if m:
        score += _W_HANDLER
        hits.append(f'handler~"{m}"')
    m = _any_hit(sig.get("brand_hints"), feat["brand_texts"], prefix_ok=True)
    if m:
        score += _W_BRAND
        hits.append(f'brand~"{m}"')
    for pat in sig.get("typical_tags") or []:
        hit_key = next((k for k in feat["tag_keys"] if str(pat) in k), None)
        if hit_key:
            score += _W_TAGS
            hits.append(f'tag~"{hit_key}"')
            break
    w, h = feat["w"], feat["h"]
    if w and h:
        for hint in sig.get("resolution_hints") or []:
            wh = _parse_res_hint(hint)
            if wh and (wh == (w, h) or wh == (h, w)):
                score += _W_RES
                hits.append(f"res={w}x{h}")
                break
    if feat["fps"] is not None:
        if any(abs(feat["fps"] - f) <= _FPS_TOL
               for f in (sig.get("fps_hints") or [])):
            score += _W_FPS
            hits.append(f'fps={feat["fps"]:g}')
    return score, hits


def _screen_bonus(feat: dict) -> tuple[float, list[str]]:
    """Screen-recording has no fixed encoder signature — its tells are
    *absences* (no camera provenance) and odd frame geometry."""
    bonus, hits = 0.0, []
    if not feat["has_video"]:
        return bonus, hits
    if feat["odd_geo"]:
        bonus += 0.25
        hits.append(f'geometry:odd-frame {feat["w"]}x{feat["h"]}')
    if not feat["camera_meta"]:
        bonus += 0.15
        hits.append("provenance:no-camera-meta")
        if feat["codec"] == "h264" and feat["appleish"]:
            bonus += 0.15
            hits.append("mux:h264-in-apple-mux-no-cam-tags")
    return bonus, hits


def _score_all(feat: dict) -> dict:
    """->{key: (score, hits, has_hard_signal)} for every non-unknown key."""
    out = {}
    for key, sig in DEVICE_SIGS.items():
        if key == "unknown":
            continue
        s, hits = _score(feat, sig)
        if key == "screen_recording":
            b, bh = _screen_bonus(feat)
            s += b
            hits += bh
        hard = any(h.split(":", 1)[0].split("~", 1)[0]
                   in ("encoder", "handler", "brand", "codec")
                   for h in hits)
        if not hard:
            s = min(s, _SOFT_CAP)
        out[key] = (s, hits, hard)
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def fingerprint(meta) -> dict:
    """Guess the capture-device family from a metadata dict.

    `meta` may be a video_meta.probe() dict, a camera_meta.video_metadata()
    dict, a merged EXIF dict, or any mix — absent keys just score zero.

    -> {"device_guess": <DEVICE_SIGS key>, "confidence": 0..1,
        "matched_signals": ["encoder~...","handler~...",...]}

    confidence >= 0.30 implies at least one hard signal (encoder, handler
    or brand) matched; soft-only hits (resolution/fps/tags) are capped
    below that floor. Never raises; empty/foreign input -> "unknown".
    """
    meta = meta if isinstance(meta, dict) else {}
    if not meta:
        return {"device_guess": "unknown", "confidence": 0.0,
                "matched_signals": []}
    try:
        scores = _score_all(_features(meta))
    except Exception:  # belt & braces — never raise on weird meta
        return {"device_guess": "unknown", "confidence": 0.0,
                "matched_signals": []}
    best = max(scores.items(), key=lambda kv: kv[1][0], default=None)
    if not best or best[1][0] <= 0.0:
        return {"device_guess": "unknown", "confidence": 0.0,
                "matched_signals": []}
    key, (score, hits, _hard) = best
    return {"device_guess": key,
            "confidence": round(min(1.0, score), 3),
            "matched_signals": hits}


def _claim_hit(pattern: str, text: str) -> bool:
    """Claim matching: word-boundary hit, or a whitespace token that starts
    with the pattern (so "sm-g" catches the model string "SM-G991B")."""
    if _hit(pattern, text):
        return True
    p = pattern.lower()
    return any(t.startswith(p) for t in text.lower().split())


def _claim_keys(claimed) -> set:
    """Free-text claim ("my Samsung Galaxy", "an iPhone") -> DEVICE_SIGS
    keys that would satisfy it. Empty set = nothing recognized."""
    s = _clean_str(claimed)
    if not s:
        return set()
    s = s.lower()
    keys = set()
    for pats, key in _SPECIFIC_CLAIMS:
        if any(_claim_hit(p, s) for p in pats):
            keys.add(key)
    if keys:
        return keys
    groups = [set(g) for pats, g in _GENERIC_CLAIMS
              if any(_claim_hit(p, s) for p in pats)]
    if not groups:
        return set()
    inter = set.intersection(*groups)
    return inter if inter else set.union(*groups)


def sig_for(device) -> dict:
    """Signature row for a device key or alias ("iphone", "my galaxy",
    "android"). Multiple same-level matches merge; unrecognized -> the
    all-empty "unknown" row shape. Always returns a fresh copy."""
    if isinstance(device, str) and device in DEVICE_SIGS:
        return copy.deepcopy(DEVICE_SIGS[device])
    keys = _claim_keys(device)
    if not keys:
        return copy.deepcopy(DEVICE_SIGS["unknown"])
    if len(keys) == 1:
        return copy.deepcopy(DEVICE_SIGS[next(iter(keys))])
    merged = {k: [] for k in _SIG_KEYS}
    for key in sorted(keys):
        sig = DEVICE_SIGS.get(key) or {}
        for field in _SIG_KEYS:
            for v in sig.get(field) or []:
                if v not in merged[field]:
                    merged[field].append(v)
    return merged


def device_evidence(meta, claimed_device=None) -> "Evidence":
    """One Evidence card: does the container signature fit the claimed
    device?

    `claimed_device` may be passed explicitly or stashed on the meta dict
    under "claimed_device"/"device_claim" (same convention as
    video_meta.analyse).

    Verdicts:

      * fingerprint matches the claim          -> SUPPORTS 0.4
      * different device family than claimed
        (e.g. Apple handlers vs an Android claim) -> CONTRADICTS 0.5
      * screen-recording signature             -> INCONCLUSIVE — a screen
        capture is not camera footage, so a device claim is untestable
        (unless the claim itself is "screen recording", which matches)
      * unknown/weak/empty fingerprint or an unrecognized claim
                                               -> INCONCLUSIVE
    """
    ClaimVerdict, Evidence = _schema()

    meta = meta if isinstance(meta, dict) else {}
    claimed = (_clean_str(claimed_device)
               or _clean_str(meta.get("claimed_device"))
               or _clean_str(meta.get("device_claim")))
    fp = fingerprint(meta)
    keys = _claim_keys(claimed)
    guess = fp["device_guess"]
    conf = fp["confidence"]

    data: dict = {
        "fingerprint": fp,
        "claimed_device": claimed,
        "claimed_keys": sorted(keys),
        "reliability": _RELIABILITY,
    }

    def ev(verdict, confidence, summary, **extra):
        data.update(extra)
        return Evidence(type="cam_model", summary=summary, verdict=verdict,
                        confidence=confidence, data=data)

    sig_note = ", ".join(fp["matched_signals"][:3]) or "no signature hits"

    if not meta:
        return ev(ClaimVerdict.INCONCLUSIVE, 0.1,
                  "no metadata — nothing to fingerprint")
    if guess == "unknown":
        return ev(ClaimVerdict.INCONCLUSIVE, 0.15,
                  "encoder/container signature doesn't match any known "
                  "device family (stripped or re-muxed file)")
    if guess == "screen_recording" and "screen_recording" not in keys:
        return ev(ClaimVerdict.INCONCLUSIVE, 0.35,
                  f"file looks like a screen recording ({sig_note}) — it "
                  "captures a screen, not a scene, so a camera-device "
                  "claim can't be tested on it")
    if claimed and not keys:
        return ev(ClaimVerdict.INCONCLUSIVE, 0.2,
                  f"container fingerprints as {guess} but the device claim "
                  f"{claimed!r} isn't a recognized device family")
    if not claimed:
        return ev(ClaimVerdict.INCONCLUSIVE, 0.3,
                  f"container fingerprints as {guess} "
                  f"(conf {conf:.2f}) — no claimed device to check it "
                  "against")
    if guess in keys:
        if conf >= _HARD_FLOOR:
            return ev(ClaimVerdict.SUPPORTS, 0.4,
                      f"container fingerprint is consistent with the "
                      f"claimed device ({claimed!r} -> {guess}: {sig_note})")
        return ev(ClaimVerdict.INCONCLUSIVE, 0.25,
                  f"fingerprint leans {guess} but only on weak hints "
                  f"({sig_note}) — not enough to support the claim")
    claim_fams = {_FAMILY.get(k, "unknown") for k in keys}
    guess_fam = _FAMILY.get(guess, "unknown")
    if guess_fam not in claim_fams and conf >= _HARD_FLOOR:
        return ev(ClaimVerdict.CONTRADICTS, 0.5,
                  f"claimed {claimed!r} but the file fingerprints as "
                  f"{guess} ({guess_fam} family: {sig_note}) — the "
                  "container wasn't written by that kind of device",
                  guess_family=guess_fam, claim_families=sorted(claim_fams))
    if guess_fam not in claim_fams:
        return ev(ClaimVerdict.INCONCLUSIVE, 0.25,
                  f"claimed {claimed!r} but the {guess}-ward fingerprint "
                  f"rests on soft hints only ({sig_note}) — can't safely "
                  "contradict")
    return ev(ClaimVerdict.INCONCLUSIVE, 0.3,
              f"fingerprints as {guess} — same {guess_fam} family as the "
              f"claimed {claimed!r}, but a different model signature "
              f"({sig_note})")


# --------------------------------------------------------------------------
# self-test — fully offline (canned meta dicts only, no ffprobe)
# --------------------------------------------------------------------------

def _self_test() -> int:
    fails = 0

    def check(name: str, got, want, tol: float | None = None) -> None:
        nonlocal fails
        ok = (got is not None and abs(got - want) <= tol) if tol is not None \
            else (got == want)
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r} "
              f"(want {want!r})")

    ClaimVerdict, _ = _schema()

    # --- iPhone-shaped meta (video_meta.probe() shape) ---------------------
    iphone = {
        "creation_time": "2025-10-15T15:04:30.000000Z",
        "encoder": "com.apple.quicktime.software 17.4.1",
        "duration_s": 30.0,
        "major_brand": "qt  ",
        "handler_names": ["Core Media Video", "Core Media Audio"],
        "streams": [{"codec": "hevc", "type": "video",
                     "w": 3840, "h": 2160, "fps": 30.0}],
    }
    fp = fingerprint(iphone)
    check("iphone guess", fp["device_guess"], "iphone")
    check("iphone conf strong", fp["confidence"] >= 0.5, True)
    check("iphone signals", bool(fp["matched_signals"]), True)
    ev = device_evidence(iphone, "shot on my iPhone")
    check("iphone claim -> SUPPORTS", ev.verdict, ClaimVerdict.SUPPORTS)
    check("iphone claim conf", ev.confidence, 0.4)
    check("evidence type", ev.type, "cam_model")

    # --- contradictory: Apple container vs Android claim --------------------
    ev = device_evidence(iphone, "recorded on my Samsung Galaxy")
    check("apple-file android-claim -> CONTRADICTS", ev.verdict,
          ClaimVerdict.CONTRADICTS)
    check("contradict conf", ev.confidence, 0.5)
    check("reliability stamped", bool(ev.data.get("reliability")), True)

    # --- Android MediaMuxer-shaped meta ------------------------------------
    android = {
        "major_brand": "mp42",
        "handler_names": ["VideoHandle", "SoundHandle"],
        "streams": [{"codec": "h264", "type": "video",
                     "w": 1920, "h": 1080, "fps": 30.03}],
    }
    fp = fingerprint(android)
    check("android guess", fp["device_guess"] in
          ("android_samsung", "android_pixel"), True)
    ev = device_evidence(android, "my android phone")
    check("android claim -> SUPPORTS", ev.verdict, ClaimVerdict.SUPPORTS)
    ev = device_evidence(android, "my iPhone")
    check("android-file iphone-claim -> CONTRADICTS", ev.verdict,
          ClaimVerdict.CONTRADICTS)

    # --- boundary check: ffmpeg's VideoHandler must NOT read as Android ----
    lavf = {
        "encoder": "Lavf60.3.100",
        "major_brand": "isom",
        "handler_names": ["VideoHandler", "SoundHandler"],
        "streams": [{"codec": "h264", "type": "video",
                     "w": 1920, "h": 1080, "fps": 29.97}],
    }
    fp = fingerprint(lavf)
    check("lavf handler != android", fp["device_guess"] not in
          ("android_samsung", "android_pixel"), True)

    # --- GoPro gpmd codec stream -------------------------------------------
    gopro = {
        "major_brand": "mp41",
        "handler_names": ["GoPro AVC", "GoPro MET"],
        "streams": [{"codec": "h264", "type": "video",
                     "w": 3840, "h": 2160, "fps": 59.94},
                    {"codec": "gpmd", "type": "data",
                     "w": None, "h": None, "fps": None}],
    }
    check("gopro guess", fingerprint(gopro)["device_guess"], "gopro")

    # --- screen recording: apple mux, h264, odd retina-crop dims, no tags ---
    screen = {
        "encoder": "com.apple.avfoundation",
        "major_brand": "qt  ",
        "handler_names": ["Core Media Video"],
        "streams": [{"codec": "h264", "type": "video",
                     "w": 1512, "h": 982, "fps": 60.0}],
    }
    fp = fingerprint(screen)
    check("screen guess", fp["device_guess"], "screen_recording")
    ev = device_evidence(screen, "filmed on my iPhone")
    check("screen -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    check("screen note", "screen" in ev.summary.lower(), True)

    # --- camera_meta.video_metadata() shape (format_tags, no streams) ------
    cm_shape = {
        "creation_time": "2025-10-15T15:04:30.000000Z",
        "duration_s": 30.0,
        "encoder": "com.apple.quicktime.software 17.4.1",
        "format_tags": {"major_brand": "qt  ",
                        "com.apple.quicktime.make": "Apple",
                        "com.apple.quicktime.model": "iPhone 15 Pro"},
    }
    check("camera_meta shape -> iphone",
          fingerprint(cm_shape)["device_guess"], "iphone")

    # --- empty / never-raise ------------------------------------------------
    check("empty fingerprint", fingerprint({})["device_guess"], "unknown")
    check("empty conf", fingerprint({})["confidence"], 0.0)
    check("none fingerprint", fingerprint(None)["device_guess"], "unknown")
    ev = device_evidence({})
    check("empty -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    ev = device_evidence({"streams": "garbage", "weird": object()}, "pixel")
    check("garbage never raises", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    ev = device_evidence(iphone)  # claim stashed on meta instead
    check("no claim -> INCONCLUSIVE", ev.verdict, ClaimVerdict.INCONCLUSIVE)
    dev = dict(iphone)
    dev["claimed_device"] = "iPhone 15"
    check("meta-carried claim -> SUPPORTS",
          device_evidence(dev).verdict, ClaimVerdict.SUPPORTS)

    # --- sig_for -------------------------------------------------------------
    check("sig_for iphone", sig_for("iphone")["handler_patterns"],
          DEVICE_SIGS["iphone"]["handler_patterns"])
    check("sig_for alias", "samsung" in
          sig_for("a samsung phone")["brand_hints"], True)
    check("sig_for android merge",
          sig_for("android")["handler_patterns"].count("videohandle"), 1)
    check("sig_for nonsense", sig_for("toaster")["encoder_patterns"], [])
    check("sig_for returns copy",
          sig_for("iphone") is not DEVICE_SIGS["iphone"], True)

    # --- table sanity --------------------------------------------------------
    check("all device keys", set(DEVICE_SIGS) == {
        "iphone", "android_samsung", "android_pixel", "gopro", "dji_drone",
        "dslr_canon", "dslr_nikon", "screen_recording", "dashcam_generic",
        "unknown"}, True)
    check("all sig fields",
          all(set(v) == set(_SIG_KEYS) for v in DEVICE_SIGS.values()), True)
    return fails


if __name__ == "__main__":
    fails = _self_test()
    print(f"self-test: {'FAILED' if fails else 'all passed'}")
    if fails:
        sys.exit(1)

    if len(sys.argv) > 1:
        # Optional demo: fingerprint a real file via video_meta.probe()
        # when ffprobe is around; device claim = optional 2nd arg.
        import json as _json
        try:
            from .video_meta import probe
        except ImportError:
            from video_meta import probe
        meta = probe(sys.argv[1])
        if not meta:
            print("(no ffprobe output — empty meta)")
        else:
            print(_json.dumps(fingerprint(meta), indent=2))
            ev = device_evidence(meta, sys.argv[2] if len(sys.argv) > 2 else None)
            print(f"evidence: {ev.verdict.value} conf={ev.confidence} "
                  f"— {ev.summary}")
