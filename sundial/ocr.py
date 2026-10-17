"""Classical OCR fallback for burned-in timestamp overlays.

The VLM clock read (perception.CosmosClient.read_clock_overlay) stays the
primary signal — this module adds determinism: when a local tesseract is
installed we can pull overlay text without a model call, then regex out
timestamp-shaped substrings for pipeline._parse_overlay_time to consume.

Everything heavy (PIL, pytesseract) is imported lazily inside functions so
this module imports with zero third-party deps, and every public function
degrades to [] when tesseract isn't on the machine.
"""
from __future__ import annotations

import importlib.util
import re
import shutil

# Strip fractions (y0, y1) of frame height. Burned-in clocks hug the frame
# edges: most cams stamp the top edge, some the bottom edge. Matches the
# top_fraction=0.3 crop the VLM clock read uses, plus the bottom sliver.
_REGION_FRACTIONS: dict[str, tuple[float, float]] = {
    "top": (0.0, 0.30),
    "bottom": (0.85, 1.0),
}

_UPSCALE = 2.5  # overlay text is tiny at full-frame scale; tesseract wants ~30px+ glyphs

# Keep in sync with pipeline._TS_PATTERNS / _TS_TIME_ONLY — same shapes, so
# anything returned by extract_timestamps parses in the clock stage.
_TS_FULL = [
    # 2026-10-17 08:00:02 / 2026/10/17T08:00 — tried first so the dd.mm.yy
    # pattern can't match the "26-10-17" tail of an ISO date.
    re.compile(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})\D{0,4}(\d{1,2}):(\d{2})(?::(\d{2}))?"),
    # 17.10.2026 08:00:02 / 17-10-2026 08:00 / 17.10.26 12:30 (dd.mm.yy)
    re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\D{0,4}(\d{1,2}):(\d{2})(?::(\d{2}))?"),
]
# Bare clock like "08:00" — usable for time-of-day only.
_TS_TIME_ONLY = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")


def ocr_available() -> bool:
    """True only if pytesseract is importable AND `tesseract` is on PATH.

    Checked lazily on each call — no caching, so installing either piece
    mid-session flips this to True on the next call.
    """
    if importlib.util.find_spec("pytesseract") is None:
        return False
    return shutil.which("tesseract") is not None


def _prep_strip(strip) -> "object":
    """Upscale + grayscale + autocontrast + gentle binarize for tesseract.

    Polarity is normalized to dark-text-on-light: if the strip is mostly
    dark (the common white-overlay-on-video case) it's inverted first, then
    thresholded just above its own brightness distribution — a slight
    binarize that survives both overlay polarities.
    """
    from PIL import Image, ImageOps, ImageStat

    w, h = strip.size
    img = strip.resize(
        (max(1, round(w * _UPSCALE)), max(1, round(h * _UPSCALE))),
        Image.Resampling.LANCZOS,
    )
    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    stat = ImageStat.Stat(gray)
    if stat.mean[0] < 127:
        gray = ImageOps.invert(gray)
        stat = ImageStat.Stat(gray)
    thr = min(250.0, stat.mean[0] + 0.5 * stat.stddev[0])
    return gray.point(lambda v: 0 if v < thr else 255)


def read_overlay_text(frame_path: str, regions=("top", "bottom")) -> list[str]:
    """OCR the overlay strips of one frame; return non-empty texts verbatim.

    Crops the named regions (top 30% / bottom 15% of frame height), preps
    each for tesseract, and returns whatever image_to_string produces —
    unfiltered, so the caller sees tesseract's raw guess. Silent fallback:
    returns [] when tesseract/pytesseract is missing, the frame won't open,
    or a strip OCRs to nothing.
    """
    if isinstance(regions, str):
        regions = (regions,)
    if not ocr_available():
        return []
    import pytesseract
    from PIL import Image

    out: list[str] = []
    try:
        with Image.open(frame_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            for region in regions:
                frac = _REGION_FRACTIONS.get(str(region).lower())
                if frac is None:
                    continue
                y0, y1 = round(h * frac[0]), round(h * frac[1])
                strip = img.crop((0, y0, w, max(y1, y0 + 1)))
                try:
                    # psm 6: each strip is a uniform block of overlay text.
                    text = pytesseract.image_to_string(
                        _prep_strip(strip), config="--psm 6"
                    )
                except Exception:
                    continue
                text = text.strip()
                if text:
                    out.append(text)
    except Exception:
        pass
    return out


def extract_timestamps(texts) -> list[str]:
    """Regex-scan OCR strings for date/time-shaped substrings.

    Same shapes as pipeline._TS_PATTERNS: full datetimes (ISO and
    dd.mm.yy(yy)) win first; bare HH:MM(:SS) matches are kept only when
    they don't sit inside an already-matched datetime. Returns the matched
    substrings (deduped, order preserved) — verbatim, so the caller can
    feed them straight to pipeline._parse_overlay_time / _clock_evidence.
    """
    if isinstance(texts, str):
        texts = [texts]
    found: list[str] = []
    seen: set[str] = set()
    for text in texts or []:
        if not text:
            continue
        covered: list[tuple[int, int]] = []
        for pat in _TS_FULL:
            for m in pat.finditer(text):
                covered.append(m.span())
                s = m.group(0).strip()
                if s and s not in seen:
                    seen.add(s)
                    found.append(s)
        for m in _TS_TIME_ONLY.finditer(text):
            if any(a <= m.start() and m.end() <= b for a, b in covered):
                continue
            s = m.group(0)
            if s not in seen:
                seen.add(s)
                found.append(s)
    return found


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Sundial OCR demo — read burned-in overlay text from a frame"
    )
    ap.add_argument("frame", help="path to a frame image")
    ap.add_argument(
        "--regions",
        nargs="+",
        default=["top", "bottom"],
        choices=sorted(_REGION_FRACTIONS),
        help="frame strips to OCR (default: top bottom)",
    )
    args = ap.parse_args()

    if not ocr_available():
        print(
            "tesseract unavailable — install the binary (`brew install tesseract`)\n"
            "and `pip install pytesseract`. The VLM clock read remains primary."
        )
        raise SystemExit(0)

    texts = read_overlay_text(args.frame, regions=tuple(args.regions))
    print(f"raw OCR text ({len(texts)} strip(s)):")
    for t in texts:
        print(repr(t))
    stamps = extract_timestamps(texts)
    print(f"timestamps ({len(stamps)}):")
    for ts in stamps:
        print(ts)
