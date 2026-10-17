"""Plate-style stage — license-plate furniture as a regional cue.

Plate design is one of the cheaper geo tells to sample directly:
  * EU plates carry a blue band (country code + circle of stars) down the
    left edge of the plate
  * UK plates are white front / yellow rear — no blue band since Brexit
  * US/Asia plates are short and wide with no blue band at all

Everything heavy (PIL) is imported lazily inside functions, so this module
imports with zero third-party deps. All analysis is pure-PIL pixel
sampling — no cv2, no detection model required at analysis time.
"""
from __future__ import annotations

import re

# COCO vehicle classes worth looking at (see perception.detect_objects).
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})

# --- pixel thresholds --------------------------------------------------------
# Blue-dominant test per spec: b > r*1.3 and b > g*1.1, plus a small floor so
# JPEG noise in near-black regions can't trip the ratio.
_BLUE_MIN = 30.0
_BLUE_R_RATIO = 1.3
_BLUE_G_RATIO = 1.1
_STRIP_FRAC = 0.08     # leftmost column-strip width, as fraction of lower third
_BAND_MIN_FRAC = 0.30  # strip must be >=30% blue-dominant to call it a band

# "Light" plate pixels: bright + low saturation (white/grey) or saturated
# yellow (UK/NL rear, some US states).
_LIGHT_MIN = 140
_LIGHT_SPREAD = 85

# Plate-patch sanity bounds: a plate is a smallish wide rectangle, not the
# whole tailgate.
_PATCH_FILL_MIN = 0.45
_PATCH_ASPECT = (1.4, 8.0)      # EU ~5.2, US ~2.0, moto ~1.6
_PATCH_W_FRAC = (0.12, 0.70)    # of vehicle-crop width

_MAX_WORK_W = 160  # downscale crops to this width before pixel loops


def _is_blue(rgb: tuple) -> bool:
    r, g, b = rgb[0], rgb[1], rgb[2]
    return b >= _BLUE_MIN and b > r * _BLUE_R_RATIO and b > g * _BLUE_G_RATIO


def _is_light(rgb: tuple) -> bool:
    r, g, b = rgb[0], rgb[1], rgb[2]
    if min(r, g, b) >= _LIGHT_MIN and max(r, g, b) - min(r, g, b) <= _LIGHT_SPREAD:
        return True  # white / grey
    return r >= 150 and g >= 125 and b <= r * 0.72 and b <= g * 0.8  # yellow


def _clamp_bbox(bbox_xyxy, width: int, height: int):
    """Normalize a bbox to an int (x1, y1, x2, y2) inside the image, or None."""
    try:
        if isinstance(bbox_xyxy, dict):
            vals = [bbox_xyxy[k] for k in ("x1", "y1", "x2", "y2")]
        else:
            vals = list(bbox_xyxy)[:4]
        x1, y1, x2, y2 = (float(v) for v in vals)
    except (TypeError, ValueError, KeyError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    box = (
        int(max(0, min(width, round(x1)))),
        int(max(0, min(height, round(y1)))),
        int(max(0, min(width, round(x2)))),
        int(max(0, min(height, round(y2)))),
    )
    if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
        return None
    return box


# --------------------------------------------------------------------------
# Per-vehicle crop analysis (pure PIL)
# --------------------------------------------------------------------------

def analyze_vehicle_crop(frame_path, bbox_xyxy) -> dict | None:
    """Check one vehicle crop for plate-style region cues.

    Crops the bbox, works on the lower third (where plates live), and looks
    for:
      * a vertical blue strip on the left edge (EU band) — the leftmost 8%
        column strip, counted blue-dominant; also re-checked at the left
        edge of any detected plate patch, since detection bboxes aren't
        always tight enough for the band to sit at the crop edge
      * a light rectangular patch in the lower-middle (the plate itself)

    Returns {"eu_blue_band", "plate_like_region", "confidence"} or None when
    the frame can't be read, PIL is missing, or the crop is too small to
    mean anything.
    """
    try:
        from PIL import Image  # lazy: module must import without PIL
    except ImportError:
        return None

    try:
        img = Image.open(frame_path)
        img.load()
        img = img.convert("RGB")
    except Exception:
        return None

    box = _clamp_bbox(bbox_xyxy, *img.size)
    if box is None:
        return None
    veh = img.crop(box)
    vw, vh = veh.size
    if vw < 24 or vh < 18:
        return None  # too small: the lower third would be a smear

    lower = veh.crop((0, int(vh * 2 / 3), vw, vh))
    if lower.size[0] > _MAX_WORK_W:
        s = _MAX_WORK_W / lower.size[0]
        lower = lower.resize((_MAX_WORK_W, max(1, round(lower.size[1] * s))))
    lw, lh = lower.size
    px = lower.load()

    # --- 1) vertical blue strip at the left edge (spec: leftmost 8%) ------
    # A real EU band is plate-height, not full-height: the blue-dominant
    # rows must form a bounded vertical run, otherwise a solid blue car's
    # left edge would pass the raw fraction test.
    sw = max(1, round(lw * _STRIP_FRAC))
    row_blue = [0] * lh
    strip_blue = 0
    for x in range(sw):
        for y in range(lh):
            if _is_blue(px[x, y]):
                row_blue[y] += 1
                strip_blue += 1
    edge_frac = strip_blue / float(sw * lh)
    run = best_run = 0
    for y in range(lh):
        if row_blue[y] >= 0.5 * sw:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    edge_band = (
        edge_frac >= _BAND_MIN_FRAC
        and max(3, round(lh * 0.12)) <= best_run <= round(lh * 0.85)
    )

    # --- 2) light rectangular patch in the lower-middle -------------------
    # Plates sit roughly centered horizontally; the middle 76% of the crop
    # keeps bodywork edges from dominating the patch bbox.
    x_lo, x_hi = int(lw * 0.12), int(lw * 0.88)
    light_xs: list[int] = []
    light_ys: list[int] = []
    for x in range(x_lo, x_hi):
        for y in range(lh):
            if _is_light(px[x, y]):
                light_xs.append(x)
                light_ys.append(y)

    plate_like = False
    patch_band = 0.0
    patch_hit = False
    n_light = len(light_xs)
    if n_light >= 8:
        bx1, bx2 = min(light_xs), max(light_xs)
        by1, by2 = min(light_ys), max(light_ys)
        pw, ph = bx2 - bx1 + 1, by2 - by1 + 1
        fill = n_light / float(pw * ph)
        aspect = pw / float(ph)
        width_frac = pw / float(lw)
        plate_like = (
            fill >= _PATCH_FILL_MIN
            and _PATCH_ASPECT[0] <= aspect <= _PATCH_ASPECT[1]
            and _PATCH_W_FRAC[0] <= width_frac <= _PATCH_W_FRAC[1]
        )
        if plate_like:
            # The EU band lives on the plate's left edge — it may be just
            # left of the light bbox (bands aren't "light"), so scan a
            # sliver before bx1 plus the patch's leftmost quarter. Count
            # columns that are mostly blue down the patch's height: a real
            # band is ~5-12% of plate width.
            s0 = max(0, bx1 - max(1, round(pw * 0.10)))
            s1 = min(lw, bx1 + max(2, round(pw * 0.25)))
            ph_full = by2 - by1 + 1
            band_cols = 0
            for x in range(s0, s1):
                cb = sum(1 for y in range(by1, by2 + 1) if _is_blue(px[x, y]))
                if cb >= 0.5 * ph_full:
                    band_cols += 1
            patch_band = min(1.0, band_cols / max(1.0, pw * 0.12))
            patch_hit = band_cols >= max(2, round(pw * 0.05))

    eu_band = edge_band or patch_hit
    band_score = max(edge_frac if edge_band else 0.0, patch_band)

    # Confidence = "how much do we trust this read": a located plate is the
    # baseline, a detected band on top raises it toward 1.
    conf = 0.15
    if plate_like:
        conf += 0.30
    if eu_band:
        conf += 0.25 + 0.30 * min(1.0, band_score)
    return {
        "eu_blue_band": eu_band,
        "plate_like_region": plate_like,
        "confidence": round(min(1.0, conf), 2),
    }


# --------------------------------------------------------------------------
# Detection filtering
# --------------------------------------------------------------------------

def vehicle_bboxes(detections) -> list:
    """Pull vehicle bounding boxes out of perception.detect_objects output.

    Keeps car/truck/bus/motorcycle only — people, signs and bicycles don't
    carry plates worth sampling.
    """
    out = []
    for d in detections or []:
        cls = str(d.get("class") or d.get("label") or "").strip().lower()
        if cls not in VEHICLE_CLASSES:
            continue
        bb = d.get("bbox_xyxy") or d.get("bbox")
        try:
            vals = (
                [bb[k] for k in ("x1", "y1", "x2", "y2")]
                if isinstance(bb, dict)
                else list(bb)[:4]
            )
            if len(vals) < 4:
                continue
            box = [float(v) for v in vals]
        except (TypeError, ValueError, KeyError):
            continue
        if box[2] > box[0] and box[3] > box[1]:
            out.append(box)
    return out


# --------------------------------------------------------------------------
# Claim region heuristics — no network, just names + coarse lat/lon boxes
# --------------------------------------------------------------------------

_EU_WORDS = (
    "europe", "european", "eu", "schengen",
    "uk", "united kingdom", "britain", "great britain", "england",
    "scotland", "wales", "london", "manchester", "birmingham",
    "ireland", "dublin", "france", "paris", "lyon", "marseille",
    "germany", "berlin", "munich", "hamburg", "frankfurt", "cologne",
    "spain", "madrid", "barcelona", "valencia", "seville",
    "italy", "rome", "milan", "naples", "turin", "florence",
    "portugal", "lisbon", "porto", "netherlands", "holland",
    "amsterdam", "rotterdam", "belgium", "brussels", "antwerp",
    "austria", "vienna", "switzerland", "zurich", "geneva", "bern",
    "poland", "warsaw", "krakow", "czech", "czechia", "prague",
    "slovakia", "bratislava", "hungary", "budapest", "romania",
    "bucharest", "bulgaria", "sofia", "greece", "athens",
    "croatia", "zagreb", "slovenia", "ljubljana", "serbia", "belgrade",
    "sweden", "stockholm", "gothenburg", "norway", "oslo", "denmark",
    "copenhagen", "finland", "helsinki", "estonia", "tallinn",
    "latvia", "riga", "lithuania", "vilnius", "luxembourg", "malta",
    "cyprus", "ukraine", "kyiv", "turkey", "istanbul",
)
_US_WORDS = (
    "us", "usa", "u s", "u.s", "united states", "america", "american",
    "california", "oregon", "washington", "nevada", "arizona", "utah",
    "colorado", "texas", "florida", "georgia", "ohio", "michigan",
    "illinois", "pennsylvania", "virginia", "new york", "new jersey",
    "massachusetts", "carolina", "tennessee", "minnesota", "wisconsin",
    "los angeles", "san francisco", "san diego", "san jose", "seattle",
    "portland", "chicago", "boston", "miami", "atlanta", "dallas",
    "houston", "austin", "denver", "phoenix", "las vegas", "nyc",
    "detroit", "philadelphia", "canada", "toronto", "vancouver",
    "montreal", "mexico",
)
_ASIA_WORDS = (
    "asia", "asian", "china", "beijing", "shanghai", "shenzhen",
    "guangzhou", "hong kong", "taiwan", "taipei", "japan", "tokyo",
    "osaka", "kyoto", "korea", "seoul", "india", "delhi", "mumbai",
    "bangalore", "singapore", "thailand", "bangkok", "vietnam",
    "hanoi", "malaysia", "kuala lumpur", "indonesia", "jakarta",
    "philippines", "manila", "uae", "dubai", "abu dhabi", "qatar",
    "saudi", "riyadh", "israel", "tel aviv",
)


def _parse_latlon(place) -> tuple[float, float] | None:
    try:
        a, b = [float(x) for x in str(place).split(",")][:2]
        if -90 <= a <= 90 and -180 <= b <= 180:
            return a, b
    except (ValueError, TypeError):
        pass
    return None


def _claim_region(claimed_place) -> str | None:
    """-> "europe" | "us" | "asia" | None — keyword + lat/lon-box heuristic.

    Deliberately cheap: the claim only needs enough resolution to know
    whether an EU blue band should be there at all.
    """
    if not claimed_place:
        return None
    s = str(claimed_place).strip().lower()
    if not s:
        return None

    ll = _parse_latlon(s)
    if ll:
        lat, lon = ll
        if 34.0 <= lat <= 72.0 and -25.0 <= lon <= 45.0:
            return "europe"
        if 15.0 <= lat <= 72.0 and -170.0 <= lon <= -55.0:
            return "us"  # North America incl. Canada/Mexico — no EU bands
        if -12.0 <= lat <= 80.0 and 55.0 <= lon <= 180.0:
            return "asia"
        return None

    for kw in _EU_WORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", s):
            return "europe"
    for kw in _US_WORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", s):
            return "us"
    for kw in _ASIA_WORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", s):
            return "asia"
    return None


# --------------------------------------------------------------------------
# Evidence card
# --------------------------------------------------------------------------

def plate_evidence(crops: list[dict], claimed_place) -> "Evidence":
    """Fold per-vehicle analyses into one Evidence card.

    The signal is asymmetric: a visible EU blue band is a real tell, but
    its *absence* proves almost nothing (angle, resolution, occlusion, mud)
    — so only band-positive cases can carry a verdict.
    """
    from .schema import ClaimVerdict, Evidence  # lazy

    analyses = [c for c in (crops or []) if isinstance(c, dict)]
    n = len(analyses)
    n_band = sum(1 for c in analyses if c.get("eu_blue_band"))
    n_plate = sum(1 for c in analyses if c.get("plate_like_region"))
    region = _claim_region(claimed_place)
    data = {
        "claimed_place": claimed_place,
        "claim_region": region,
        "vehicles_analyzed": n,
        "eu_blue_band_count": n_band,
        "plate_like_count": n_plate,
    }

    if not analyses:
        return Evidence(
            type="plate_style",
            summary="no vehicle crops available for plate-style analysis",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data=data,
        )

    if n_band:
        what = f"EU-style blue plate band on {n_band}/{n} vehicle(s)"
        if region == "europe":
            return Evidence(
                type="plate_style",
                summary=f"{what} — consistent with claimed {claimed_place!r}",
                verdict=ClaimVerdict.SUPPORTS,
                confidence=round(min(0.6, 0.45 + 0.05 * n_band), 2),
                data=data,
            )
        if region in ("us", "asia"):
            return Evidence(
                type="plate_style",
                summary=f"{what} — unexpected for claimed {claimed_place!r}",
                verdict=ClaimVerdict.CONTRADICTS,
                confidence=round(min(0.7, 0.50 + 0.05 * n_band), 2),
                data=data,
            )
        return Evidence(
            type="plate_style",
            summary=(
                f"{what} — but claimed place {claimed_place!r} "
                "can't be mapped to a region"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.35,
            data=data,
        )

    if n_plate:
        summary = (
            f"plate-like region on {n_plate}/{n} vehicle(s), no EU blue "
            "band — absence is weak (angle/resolution/occlusion hide it)"
        )
        conf = 0.25
    else:
        summary = f"no readable plates on {n} vehicle crop(s)"
        conf = 0.15
    return Evidence(
        type="plate_style",
        summary=summary,
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=conf,
        data=data,
    )


# --------------------------------------------------------------------------
# Demo: python3 -m sundial.plate_style frame.jpg [--bbox x1,y1,x2,y2] [--place Berlin]
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Sundial plate-style demo — EU blue band / plate-region cues"
    )
    ap.add_argument("frame", help="path to a frame image")
    ap.add_argument(
        "--bbox", action="append", metavar="X1,Y1,X2,Y2",
        help="vehicle box to analyze (repeatable); skips object detection",
    )
    ap.add_argument(
        "--place", default=None,
        help="claimed place ('Berlin', 'Austin TX', or 'lat,lon') — emits plate_evidence",
    )
    args = ap.parse_args()

    boxes: list[list[float]] = []
    for raw in args.bbox or []:
        try:
            boxes.append([float(v) for v in raw.split(",")][:4])
        except ValueError:
            print(f"ignoring malformed --bbox {raw!r}")

    if not boxes:
        try:
            from . import perception  # lazy: needs ultralytics only if used
            boxes = vehicle_bboxes(perception.detect_objects(args.frame))
            print(f"{len(boxes)} vehicle detection(s) in frame")
        except Exception as e:
            print(f"vehicle detection unavailable: {e}")

    if not boxes:
        print("no vehicles to analyze (pass --bbox or install ultralytics)")

    crops: list[dict] = []
    for i, bb in enumerate(boxes):
        r = analyze_vehicle_crop(args.frame, bb)
        print(f"vehicle {i} bbox={bb}: {json.dumps(r)}")
        if r is not None:
            crops.append(r)

    if args.place is not None:
        ev = plate_evidence(crops, args.place)
        print(f"evidence[{ev.type}] {ev.verdict.value} "
              f"conf={ev.confidence}: {ev.summary}")
