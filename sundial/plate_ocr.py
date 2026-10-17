"""Plate-OCR stage — targeted licence-plate reads as a country tell.

plate_style.py samples pixels for the *presence* of an EU blue band; this
stage goes one level deeper: find plate-shaped regions, crop them tight,
and ask the VLM to read the literal characters plus the country-strip code
printed on the band. A strip that literally reads "D" is a much sharper
country signal than "there's a blue band somewhere".

Flow:
  plate_regions()   — vehicle lower-thirds -> edge-dense plate boxes (cv2)
  read_plates()     — crop each region to /tmp, ask the VLM (injectable)
  plate_hints()     — strip codes / colours / formats -> {cc: [reasons]}
  plate_evidence()  — fold reads into one Evidence card vs the claimed cc

Conventions (repo): stdlib only at import time; cv2/PIL are imported lazily
inside functions; nothing here raises — missing deps or unreadable frames
degrade to [] / INCONCLUSIVE. `python3 sundial/plate_ocr.py` runs the
offline self-test (the VLM path is an injected callable, so no network).
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

# COCO vehicle classes worth looking at (same set as plate_style.py).
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})

_MAX_REGIONS = 4          # candidate crops per frame, capped
_ASPECT = (2.0, 6.0)      # plate width/height bounds — EU ~5.2, US ~2.0
_MIN_W = 28               # px; below this there is nothing legible to read
_MIN_H = 9
_EDGE_DENSITY = 0.12      # fraction of Canny edge pixels inside the box
_PAD_X = 0.10             # crop padding around a detected box (fractions)
_PAD_Y = 0.25
_UPSCALE_W = 240          # VLM legibility floor for crop width (px)


# --------------------------------------------------------------------------
# VLM prompt — strict JSON, literal characters only
# --------------------------------------------------------------------------

PLATE_PROMPT = """\
You are a forensic analyst reading a vehicle licence plate. This image is a
tight crop that should contain ONE licence plate.

Return STRICT JSON with this exact schema — no prose, no markdown fences:

{"plate_text": "<literal characters>", "country_strip": {"visible": <true|false>, "code": "<strip letters>", "color": "<colour>"}, "shape": "eu"|"us"|"jp"|"uk"|"other"|"unknown"}

- "plate_text": ONLY the characters you can literally read on the plate, in
  order, verbatim (letters, digits, and any printed separators like "-").
  Never guess, complete, or correct characters. If the plate is unreadable
  or only partially visible, transcribe the readable part; if nothing is
  readable, use null.
- "country_strip": the small country-identification band many plates carry
  (EU plates: blue band on the left edge with a circle of stars and the
  country letters; other countries may print an oval code). "visible": true
  only if such a band is actually visible; "code": the letters printed on
  it, verbatim (e.g. "D", "F", "NL", "PL", "GB") or null; "color": the
  band's colour (e.g. "blue") — or the plate's background colour (e.g.
  "yellow") when no band is visible; null if neither applies.
- "shape": "eu" for long EU-style plates, "uk" for UK white-front /
  yellow-rear plates, "us" for short wide US-style, "jp" for Japanese,
  "other" for anything else, "unknown" if you cannot tell.

If no licence plate is visible at all, respond with
{"plate_text": null, "country_strip": {"visible": false, "code": null, "color": null}, "shape": "unknown"}.

Respond with JSON only."""


# --------------------------------------------------------------------------
# Country-strip code -> ISO country code
# --------------------------------------------------------------------------

# International vehicle distinguishing signs — the letters printed on EU
# blue bands and oval country stickers. Single letters are the common case
# ("D" = Deutschland/Germany); a few countries use digraphs/trigraphs.
_STRIP_TO_CC = {
    # EU member states (+ EEA)
    "D": "DE", "A": "AT", "B": "BE", "BG": "BG", "CY": "CY", "CZ": "CZ",
    "DK": "DK", "E": "ES", "EST": "EE", "F": "FR", "FIN": "FI", "GR": "GR",
    "EL": "GR", "H": "HU", "HR": "HR", "I": "IT", "IRL": "IE", "L": "LU",
    "LT": "LT", "LV": "LV", "M": "MT", "NL": "NL", "P": "PT", "PL": "PL",
    "RO": "RO", "S": "SE", "SK": "SK", "SLO": "SI",
    # non-EU Europe & neighbours
    "GB": "GB", "UK": "GB", "CH": "CH", "FL": "LI", "IS": "IS", "N": "NO",
    "TR": "TR", "UA": "UA", "BY": "BY", "RUS": "RU", "MD": "MD", "AL": "AL",
    "MK": "MK", "NMK": "MK", "SRB": "RS", "BIH": "BA", "MNE": "ME",
    "RKS": "XK", "GE": "GE", "AM": "AM", "AZ": "AZ", "KZ": "KZ",
    "MC": "MC", "SM": "SM", "AND": "AD", "V": "VA",
}

# EU member ISO codes — the attribution set for generic "eu" hints.
_EU_CCS = frozenset({
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL",
    "PT", "RO", "SE", "SI", "SK",
})

# Codes the VLM may return already in ISO form ("DE", "ES", "SE"...) —
# accepted as-is when they aren't in the strip table.
_KNOWN_CCS = frozenset(_STRIP_TO_CC.values()) | frozenset({
    "US", "CA", "MX", "JP", "CN", "IN", "KR", "AU", "NZ", "ZA", "BR",
    "AR", "CL", "CO", "PE", "VE", "IL", "AE", "SA", "QA", "HK", "TW",
    "TH", "MY", "SG", "ID", "PH", "VN", "EG", "MA", "DZ", "TN", "NG",
    "KE", "GH", "CU", "UY", "PY", "BO", "EC", "GT", "DO", "CR", "PA",
})

# Plate-text formats -> country hints. First match wins.
_PLATE_FORMATS = (
    # France post-2009 SIV: AA-123-AA
    (re.compile(r"^[A-Z]{2}[-. ]?\d{3}[-. ]?[A-Z]{2}$"),
     frozenset({"FR"}), "AA-123-AA layout (post-2009 French SIV)"),
    # UK post-2001: AB51 CDE
    (re.compile(r"^[A-Z]{2}\d{2} ?[A-Z]{3}$"),
     frozenset({"GB"}), "AB51-CDE layout (post-2001 UK format)"),
    # Dash/space-grouped pairs "XX-XX-XX" — generic pan-EU grouping
    # (NL runs many such layouts; DE/others group similarly with spaces).
    (re.compile(r"^(?:[A-Z0-9]{1,2}[-. ]){2,3}[A-Z0-9]{1,4}$"),
     _EU_CCS, "grouped XX-XX-XX style — EU-format grouping"),
)

_SHAPES = frozenset({"eu", "us", "jp", "uk", "other", "unknown"})


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _schema():
    """Lazy schema import — tolerates both package and direct-script use."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # `python3 sundial/plate_ocr.py` direct run
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


def _strip_cc(code) -> str | None:
    """Map a country-strip code to an ISO cc, or None when unrecognized."""
    c = re.sub(r"[^A-Za-z]", "", str(code or "")).upper()
    if not c:
        return None
    if c in _STRIP_TO_CC:
        return _STRIP_TO_CC[c]
    if c in _KNOWN_CCS:
        return c  # VLM answered in ISO already
    return None


def _vehicle_bboxes(dets) -> list:
    """-> [(x1,y1,x2,y2)] for vehicle detections (same parsing as plate_style)."""
    out = []
    for d in dets or []:
        if not isinstance(d, dict):
            continue
        cls = d.get("class") or d.get("label")
        if cls is not None and str(cls).strip().lower() not in VEHICLE_CLASSES:
            continue  # explicit non-vehicle class — skip; classless box: accept
        bb = d.get("bbox_xyxy") or d.get("bbox")
        try:
            if isinstance(bb, dict):
                if "x1" in bb:
                    vals = [bb[k] for k in ("x1", "y1", "x2", "y2")]
                else:  # {"x","y","w","h"} form
                    vals = [bb["x"], bb["y"],
                            bb["x"] + bb["w"], bb["y"] + bb["h"]]
            else:
                vals = list(bb)[:4]
            box = tuple(float(v) for v in vals)
        except (TypeError, ValueError, KeyError):
            continue
        if box[2] > box[0] and box[3] > box[1]:
            out.append(box)
    return out


def _iou(a: dict, b: dict) -> float:
    iw = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    ih = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# plate_regions — candidate plate crops for a frame
# --------------------------------------------------------------------------

def _plate_boxes_cv2(cv2, img, zones, fw: int, fh: int) -> list:
    """Edge-density plate hunt inside each zone; returns padded xywh dicts."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    cands = []  # (score, x, y, w, h) in frame coords
    for zx, zy, zw, zh in zones:
        x0 = max(0, int(zx))
        y0 = max(0, int(zy))
        x1 = min(fw, int(zx + zw))
        y1 = min(fh, int(zy + zh))
        zw_eff, zh_eff = x1 - x0, y1 - y0
        if zw_eff < _MIN_W or zh_eff < _MIN_H:
            continue
        sub = gray[y0:y1, x0:x1]
        try:
            edges = cv2.Canny(sub, 60, 160)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            continue
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < _MIN_W or bh < _MIN_H:
                continue
            aspect = bw / float(bh)
            if not (_ASPECT[0] <= aspect <= _ASPECT[1]):
                continue
            if bw > 0.95 * zw_eff:
                continue  # a box spanning the whole zone isn't a plate
            dens = cv2.countNonZero(edges[by:by + bh, bx:bx + bw]) / float(bw * bh)
            if dens < _EDGE_DENSITY:
                continue  # plates are edge-dense (glyphs on blank field)
            score = dens * min(1.0, bw / 140.0)
            cands.append((score, x0 + bx, y0 + by, bw, bh))
    cands.sort(key=lambda t: -t[0])
    kept: list[dict] = []
    for score, x, y, w, h in cands:
        px, py = w * _PAD_X, h * _PAD_Y
        cx = max(0, int(x - px))
        cy = max(0, int(y - py))
        box = {
            "x": cx,
            "y": cy,
            "w": min(fw, int(x + w + px)) - cx,
            "h": min(fh, int(y + h + py)) - cy,
        }
        if box["w"] < _MIN_W or box["h"] < _MIN_H:
            continue
        if any(_iou(box, k) > 0.45 for k in kept):
            continue
        kept.append(box)
        if len(kept) >= _MAX_REGIONS:
            break
    return kept


def plate_regions(frame_path: str, yolo_dets: list | None = None) -> list:
    """-> [{"x","y","w","h"}] candidate plate crops (frame coords), capped 4.

    Detection strategy, best-effort in order:
      1. Vehicle detections -> each vehicle's lower third is a search zone
         (plates live on bumpers). No detections -> the frame's lower 55%
         becomes one zone (plates sit low in traffic/dashcam footage).
      2. With cv2 available, each zone is Canny-edged + morph-closed and
         contour boxes are kept when they're aspect 2:1-6:1, reasonably
         sized, and edge-dense. Boxes are padded, de-overlapped, ranked by
         density, and capped at 4.
      3. Without cv2 (or when the edge hunt finds nothing), the coarse
         zones themselves are returned — the VLM can still spot a plate
         inside a vehicle's lower third. Nothing usable -> [].

    Never raises: unreadable frame + no detections -> [].
    """
    zones: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in _vehicle_bboxes(yolo_dets):
        w, h = x2 - x1, y2 - y1
        if w < 24 or h < 16:
            continue
        zones.append((x1, y1 + h * 2.0 / 3.0, w, h / 3.0))

    cv2 = None
    try:
        import cv2 as _cv2
        cv2 = _cv2
    except Exception:
        pass
    img = None
    if cv2 is not None:
        try:
            img = cv2.imread(str(frame_path))
        except Exception:
            img = None

    if img is not None:
        fh, fw = img.shape[:2]
        if not zones:
            zones = [(0.0, fh * 0.45, float(fw), fh * 0.55)]
        cands = _plate_boxes_cv2(cv2, img, zones, fw, fh)
        if cands:
            return cands[:_MAX_REGIONS]
        # edge hunt empty — fall through to coarse zones
    elif not zones:
        # No cv2 / unreadable frame: a frame-level zone still needs dims.
        try:
            from PIL import Image  # lazy
            with Image.open(str(frame_path)) as im:
                fw, fh = im.size
            zones = [(0.0, fh * 0.45, float(fw), fh * 0.55)]
        except Exception:
            return []

    return [
        {"x": int(round(x)), "y": int(round(y)),
         "w": int(round(w)), "h": int(round(h))}
        for x, y, w, h in zones
        if w > 0 and h > 0
    ][:_MAX_REGIONS]


# --------------------------------------------------------------------------
# read_plates — crop + VLM read per region
# --------------------------------------------------------------------------

def _crop_tmp(frame_path, x: int, y: int, w: int, h: int) -> str | None:
    """Crop (x,y,w,h) out of frame_path to a temp JPEG; return path or None.

    Tries cv2 first, then PIL. Small crops are upscaled toward _UPSCALE_W
    so the VLM gets legible glyphs. Cleans up the temp file on failure.
    """
    try:
        fd, path = tempfile.mkstemp(prefix="sundial_plate_", suffix=".jpg")
        os.close(fd)
    except Exception:
        return None

    try:
        import cv2  # lazy
        img = cv2.imread(str(frame_path))
        if img is not None:
            fh, fw = img.shape[:2]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(fw, x + w), min(fh, y + h)
            if x1 > x0 and y1 > y0:
                crop = img[y0:y1, x0:x1]
                cw, ch = x1 - x0, y1 - y0
                if cw < _UPSCALE_W:
                    s = _UPSCALE_W / float(cw)
                    crop = cv2.resize(
                        crop, (_UPSCALE_W, max(1, round(ch * s))),
                        interpolation=cv2.INTER_CUBIC)
                if cv2.imwrite(path, crop):
                    return path
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from PIL import Image  # lazy
        with Image.open(str(frame_path)) as im:
            im = im.convert("RGB")
            fw, fh = im.size
            box = (max(0, x), max(0, y), min(fw, x + w), min(fh, y + h))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("empty crop")
            crop = im.crop(box)
            if crop.size[0] < _UPSCALE_W:
                s = _UPSCALE_W / float(crop.size[0])
                crop = crop.resize(
                    (_UPSCALE_W, max(1, round(crop.size[1] * s))),
                    Image.Resampling.LANCZOS)
            crop.save(path, "JPEG", quality=92)
            return path
    except Exception:
        pass

    try:
        os.unlink(path)
    except OSError:
        pass
    return None


def _normalize_read(raw) -> dict:
    """Coerce a VLM answer into the canonical read dict."""
    r = raw if isinstance(raw, dict) else {}
    strip = r.get("country_strip")
    strip = strip if isinstance(strip, dict) else {}

    text = r.get("plate_text")
    text = str(text).strip() if text not in (None, "") else None
    if text == "":
        text = None

    code = strip.get("code")
    code = re.sub(r"[^A-Za-z]", "", str(code)).upper() or None if code else None
    color = strip.get("color") or r.get("color")
    color = str(color).strip().lower() if color is not None else None
    color = color or None
    visible = bool(strip.get("visible")) or bool(code)

    shape = str(r.get("shape") or "unknown").strip().lower()
    if shape not in _SHAPES:
        shape = "unknown"

    conf = r.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        # No model confidence — derive from what was actually read.
        conf = 0.10
        if text:
            conf += 0.35 + (0.10 if len(text) >= 4 else 0.0)
        if code:
            conf += 0.35
        elif visible:
            conf += 0.05
        if shape != "unknown":
            conf += 0.05
        conf = min(conf, 0.9)

    return {
        "plate_text": text,
        "country_strip": {"visible": visible, "code": code, "color": color},
        "shape": shape,
        "confidence": round(conf, 2),
    }


def read_plates(ask_fn, frame_path: str, regions: list) -> list:
    """Crop each region and ask the VLM to read it.

    ask_fn is injectable: called as ask_fn(crop_path, PLATE_PROMPT) and
    expected to return a parsed-JSON dict (see CosmosClient._json). Returns
    a normalized [{"plate_text","country_strip","shape","confidence"}] list
    — one entry per successfully cropped region, plus a "crop" key with the
    temp file path for traceability. Regions that can't be cropped (no
    cv2/PIL, unreadable frame, degenerate box) are skipped; ask_fn
    exceptions degrade to an empty read. Never raises.
    """
    if not callable(ask_fn) or not regions:
        return []
    out: list[dict] = []
    for reg in regions:
        if not isinstance(reg, dict):
            continue
        try:
            x, y = int(reg["x"]), int(reg["y"])
            w, h = int(reg["w"]), int(reg["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        crop = _crop_tmp(frame_path, x, y, w, h)
        if crop is None:
            continue
        try:
            raw = ask_fn(crop, PLATE_PROMPT)
        except Exception:
            raw = {}
        read = _normalize_read(raw)
        read["crop"] = crop
        out.append(read)
    return out


# --------------------------------------------------------------------------
# plate_hints — fold reads into {cc: [reasons]}
# --------------------------------------------------------------------------

def plate_hints(reads: list) -> dict:
    """-> {country_code: [reason strings]} from a list of reads.

    Three+one hint channels, weakest-first attribution:
      * country-strip code -> direct lookup (D->DE, F->FR, NL->NL, ...)
      * yellow colour      -> NL/GB rear-plate tell
      * blue strip, no code -> weak pan-EU hint
      * plate-text format  -> AA-123-AA => FR, AB51CDE => GB,
                            XX-XX-XX groupings => EU set
      * plate shape        -> eu=>EU set (only when no code), uk=>GB,
                            us=>US, jp=>JP
    """
    hints: dict[str, list[str]] = {}

    def add(cc, why: str) -> None:
        cc = str(cc or "").strip().upper()
        if not cc:
            return
        lst = hints.setdefault(cc, [])
        if why not in lst:
            lst.append(why)

    for raw in reads or []:
        if not isinstance(raw, dict):
            continue
        r = _normalize_read(raw)
        strip = r["country_strip"]

        cc = _strip_cc(strip["code"])
        if cc:
            add(cc, f"country-strip code '{strip['code']}' resolves to {cc}")

        color = strip["color"] or ""
        if "yellow" in color:
            for c in ("GB", "NL"):
                add(c, "yellow plate — NL/GB rear-plate tell")
        elif "blue" in color and strip["visible"] and not cc:
            for c in sorted(_EU_CCS):
                add(c, "blue EU country strip visible (code unreadable)")

        text = r["plate_text"] or ""
        if text:
            spaced = re.sub(r"\s+", " ", text.strip().upper())
            compact = re.sub(r"[^A-Z0-9]", "", spaced)
            for pat, ccs, why in _PLATE_FORMATS:
                if pat.match(spaced) or pat.match(compact):
                    for c in sorted(ccs):
                        add(c, f"plate text {text!r} matches {why}")
                    break

        shape = r["shape"]
        if shape == "eu" and not cc:
            for c in sorted(_EU_CCS):
                add(c, "eu-shaped plate (no readable strip code)")
        elif shape == "uk":
            add("GB", "uk-shaped plate")
        elif shape == "us":
            add("US", "us-shaped plate")
        elif shape == "jp":
            add("JP", "jp-shaped plate")
    return hints


# --------------------------------------------------------------------------
# plate_evidence — one Evidence card vs the claimed country code
# --------------------------------------------------------------------------

def plate_evidence(reads: list, claimed_cc) -> "Evidence":
    """Fold plate reads into an Evidence card.

    Verdicts are driven by resolved country-strip codes only:
      * a strip code resolving to the claimed cc -> SUPPORTS (0.55)
      * strip codes resolving elsewhere          -> CONTRADICTS (0.55)
      * no readable strip / no reads / no claim  -> INCONCLUSIVE
    Weaker hints (colour, format, shape) ride along in data["hints"] for
    the geo-family rollup but never carry a verdict alone — a yellow plate
    outside NL/GB is common enough to be weak.
    """
    ClaimVerdict, Evidence = _schema()

    norm = [
        _normalize_read(r) for r in (reads or [])
        if isinstance(r, dict)
    ]
    for r, raw in zip(norm, (x for x in (reads or []) if isinstance(x, dict))):
        if raw.get("crop"):
            r["crop"] = raw["crop"]

    claimed_raw = str(claimed_cc or "").strip().upper()
    claimed = _strip_cc(claimed_raw) or claimed_raw or None  # "UK"/"D" -> ISO

    strip_hits: list[tuple[str, str]] = []  # (resolved_cc, raw code)
    for r in norm:
        code = r["country_strip"]["code"]
        cc = _strip_cc(code)
        if cc:
            strip_hits.append((cc, code))

    hints = plate_hints(norm)
    data = {
        "claimed_cc": claimed,
        "strip_codes": [code for _, code in strip_hits],
        "strip_ccs": sorted({cc for cc, _ in strip_hits}),
        "hints": hints,
        "reads": norm,
    }
    frames = [r["crop"] for r in norm if r.get("crop")][:4]

    if not claimed:
        return Evidence(
            type="plate_ocr",
            summary="plate reads exist but no claimed country to check against",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0 if not norm else 0.1,
            data=data,
            frames=frames,
        )

    hit_ccs = {cc for cc, _ in strip_hits}
    if claimed in hit_ccs:
        codes = sorted({code for cc, code in strip_hits if cc == claimed})
        return Evidence(
            type="plate_ocr",
            summary=(
                f"plate strip code {', '.join(repr(c) for c in codes)} "
                f"resolves to {claimed} — consistent with the claim"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=0.55,
            data=data,
            frames=frames,
        )

    if strip_hits:
        others = sorted(hit_ccs)
        codes = sorted({code for _, code in strip_hits})
        return Evidence(
            type="plate_ocr",
            summary=(
                f"plate strip code {', '.join(repr(c) for c in codes)} "
                f"resolves to {'/'.join(others)} — contradicts claimed {claimed}"
            ),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=0.55,
            data=data,
            frames=frames,
        )

    n = len(norm)
    n_text = sum(1 for r in norm if r["plate_text"])
    if not n:
        summary = "no plate regions could be read"
        conf = 0.0
    elif n_text:
        summary = (
            f"{n_text}/{n} plate region(s) have partial text but no "
            "country strip — placement can't be confirmed"
        )
        conf = 0.2
    else:
        summary = f"{n} plate region(s) read, none legible enough"
        conf = 0.15
    return Evidence(
        type="plate_ocr",
        summary=summary,
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=conf,
        data=data,
        frames=frames,
    )


# --------------------------------------------------------------------------
# Self-test: python3 sundial/plate_ocr.py  (offline — no VLM, no cv2/PIL)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    ClaimVerdict, _Evidence = _schema()

    fake_d = [{
        "plate_text": "M AB 1234",
        "country_strip": {"visible": True, "code": "D", "color": "blue"},
        "shape": "eu",
        "confidence": 0.8,
    }]

    ev = plate_evidence(fake_d, "DE")
    assert ev.verdict is ClaimVerdict.SUPPORTS and ev.confidence == 0.55, ev
    print("SUPPORTS :", ev.summary)

    ev = plate_evidence(fake_d, "FR")
    assert ev.verdict is ClaimVerdict.CONTRADICTS and ev.confidence == 0.55, ev
    print("CONTRADICT:", ev.summary)

    ev = plate_evidence([], "DE")
    assert ev.verdict is ClaimVerdict.INCONCLUSIVE, ev
    print("INCONCLUSIVE:", ev.summary)

    # "UK" claim normalizes to GB; strip "GB" resolves to GB -> supports.
    ev = plate_evidence(
        [{"country_strip": {"visible": True, "code": "GB", "color": None},
          "plate_text": "AB51 CDE", "shape": "uk"}], "UK")
    assert ev.verdict is ClaimVerdict.SUPPORTS, ev

    # hints: strip code + format + shape channels
    hints = plate_hints(fake_d)
    assert "DE" in hints and any("strip" in r for r in hints["DE"]), hints
    hints = plate_hints([{"plate_text": "AB-123-CD", "country_strip": {},
                          "shape": "unknown"}])
    assert "FR" in hints, hints
    hints = plate_hints([{"plate_text": "99-XX-42", "country_strip": {},
                          "shape": "unknown"}])
    assert "NL" in hints and "FR" in hints, hints  # EU-pair grouping -> EU set
    hints = plate_hints([{"plate_text": None,
                          "country_strip": {"visible": False, "code": None,
                                            "color": "yellow"},
                          "shape": "uk"}])
    assert {"GB", "NL"} <= set(hints), hints

    # never-raise paths (this env has no cv2/PIL and no such frame)
    assert plate_regions("/nonexistent/frame.jpg", None) == []
    regs = plate_regions("/nonexistent/frame.jpg",
                         [{"class": "car", "bbox": [10, 10, 110, 70]}])
    assert 1 <= len(regs) <= 4, regs
    assert all(set(r) == {"x", "y", "w", "h"} for r in regs), regs
    assert regs[0]["y"] > 10 and regs[0]["h"] <= 20 + 1, regs  # lower third
    assert plate_regions("/nonexistent/frame.jpg",
                         [{"class": "person", "bbox": [0, 0, 9, 9]}]) == []

    # read_plates: injected ask_fn; without cv2/PIL cropping degrades to []
    calls = []
    out = read_plates(
        lambda p, q: (calls.append((p, q)) or {"plate_text": "X"}),
        "/nonexistent/frame.jpg", [{"x": 0, "y": 0, "w": 50, "h": 20}])
    assert out == [], out  # no imaging lib -> honest empty, never raises
    assert read_plates(None, "f.jpg", regs) == []
    assert read_plates(lambda p, q: {}, "f.jpg", [{"x": 0}]) == []

    # normalization idempotence + robustness
    n = _normalize_read({"plate_text": " abc ", "shape": "EU",
                         "country_strip": {"code": "d-"}})
    assert n["plate_text"] == "abc" and n["shape"] == "eu"
    assert n["country_strip"]["code"] == "D" and n["country_strip"]["visible"]
    n = _normalize_read("garbage")
    assert n["plate_text"] is None and n["shape"] == "unknown"

    print("\nOK — plate_ocr self-test passed")
