"""OpenCV+Tesseract boost pass for burned-in timestamp strips.

ocr.py is the baseline: PIL-only prep, two whole-band crops, psm 6. This
module is the complementary upgrade for clips where that baseline
under-reads — the classical preprocessing recipe for tiny burned-in clock
text:

    crop -> pad -> upscale -> polarity -> binarize -> psm7 -> temporal vote

  * crop     four candidate regions per frame: the top 8% and bottom 12%
             bands (full width) plus the top-right / bottom-right
             quadrants where most cams burn the clock.
  * pad      15px white border so tesseract doesn't clip edge-hugging
             glyphs — burned-in stamps sit flush against the frame edge.
  * upscale  2.5x INTER_CUBIC; tesseract wants ~30px+ glyph height.
  * polarity auto-normalized to dark-on-light: invert when the strip mean
             is dark (the common white-text-on-video overlay).
  * binarize Otsu + adaptive-threshold variants alongside the plain gray —
             different overlays binarize differently, so we OCR all three
             and let the temporal vote sort out the noise.
  * psm7     single-text-line mode with a timestamp whitelist (digits,
             :/.- separators, AM/PM, deg/V/F for temp strips) and the
             system dictionary disabled — timestamps are not words.
  * temporal vote — OCRSync-style per-character majority across frames:
             one frame's OCR noise (0<->O, 8<->B, 5<->S) is outvoted by
             the same pixels read correctly on neighbouring frames.

Stdlib only at import time; cv2 / pytesseract / PIL load lazily inside
functions. Every public function degrades to [] / None / zeros when deps
or inputs are missing — this module never raises.
"""
from __future__ import annotations

import hashlib
import importlib.util
import shutil
import tempfile
from collections import Counter
from pathlib import Path

__all__ = [
    "strip_candidates",
    "prep_for_ocr",
    "ocr_strip",
    "temporal_consensus",
    "read_timestamps",
]

# Candidate strip regions as (x0, y0, x1, y1) fractions of frame W/H.
# Burned-in clocks hug the edges; right-edge quadrants catch the common
# corner stamps while the full-width bands catch centred/left stamps.
_CROP_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "top_band": (0.0, 0.00, 1.0, 0.08),
    "bottom_band": (0.0, 0.88, 1.0, 1.00),
    "top_right": (0.5, 0.00, 1.0, 0.50),
    "bottom_right": (0.5, 0.50, 1.0, 1.00),
}

_UPSCALE = 2.5      # ~30px+ glyphs for tesseract (same factor as ocr.py)
_PAD = 15           # px of white border around each upscaled strip
_DARK_MEAN = 127.0  # gray mean below this -> white-text-on-dark -> invert
_ADAPTIVE_C = 10.0  # adaptiveThreshold constant subtracted from local mean
_ADAPTIVE_MAX_BLOCK = 51  # cap so the window stays text-local on big crops
_OUT_DIRNAME = "sundial_ocr_boost"

# Prep variants, in reported order. "gray" keeps the raw polarity-fixed
# grayscale; the two binarizations trade off global vs local contrast.
_VARIANTS = ("gray", "otsu", "adaptive")

# psm 7 = treat the strip as a single text line. The whitelist is quoted
# as one -c token (it contains a literal space) because pytesseract
# shlex-splits the config string. load_system_dawg=0 disables the word
# dictionary — timestamps aren't words and dictionary correction mangles
# them.
_TESS_CONFIG = (
    "--psm 7 "
    '-c "tessedit_char_whitelist=0123456789:/.-AMPMC\u00b0VF apm" '
    "-c load_system_dawg=0"
)


def _cv2_available() -> bool:
    """True if cv2 is importable (probe only — no import side effects)."""
    return importlib.util.find_spec("cv2") is not None


def _ocr_available() -> bool:
    """True only if pytesseract is importable AND `tesseract` is on PATH."""
    if importlib.util.find_spec("pytesseract") is None:
        return False
    return shutil.which("tesseract") is not None


def _out_dir() -> Path:
    """Writable dir for crop/variant PNGs — /tmp preferred, TMPDIR fallback."""
    for base in ("/tmp", tempfile.gettempdir()):
        try:
            d = Path(base) / _OUT_DIRNAME
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            continue
    return Path(tempfile.gettempdir())


def _tag(path: str) -> str:
    """Short stable hash of a source path — keeps crop names collision-free."""
    return hashlib.md5(str(path).encode("utf-8", "replace")).hexdigest()[:8]


def strip_candidates(frame_path: str) -> list[str]:
    """Crop the four clock-strip candidates from one frame -> saved PNGs.

    Regions: top 8% band, bottom 12% band (both full width), top-right and
    bottom-right quadrants. Files land in /tmp/sundial_ocr_boost named
    ``<frame_stem>.<tag>_<region>.png``. Returns the saved paths in
    _CROP_REGIONS order; [] when cv2 is absent, the frame won't decode,
    or the tmp dir is unwritable.
    """
    if not _cv2_available():
        return []
    import cv2  # lazy

    try:
        img = cv2.imread(str(frame_path))
    except Exception:
        return []
    if img is None or img.size == 0:
        return []
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return []
    out_dir = _out_dir()
    stem = Path(str(frame_path)).stem or "frame"
    tag = _tag(str(frame_path))
    out: list[str] = []
    for region, (fx0, fy0, fx1, fy1) in _CROP_REGIONS.items():
        x0, x1 = round(w * fx0), round(w * fx1)
        y0, y1 = round(h * fy0), round(h * fy1)
        crop = img[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
        if crop.size == 0:
            continue
        path = out_dir / f"{stem}.{tag}_{region}.png"
        try:
            if cv2.imwrite(str(path), crop):
                out.append(str(path))
        except Exception:
            continue
    return out


def _prep_variants(crop_path: str) -> list[tuple[str, str]]:
    """Build all prep variants for one strip -> (variant_name, path) pairs.

    Shared base for every variant: 2.5x cubic upscale -> 15px white pad ->
    grayscale -> polarity fix (invert when the strip mean is dark so text
    always ends dark-on-light). Then per-variant finishes: raw gray, Otsu
    binarize, adaptive (gaussian local-mean) binarize.
    """
    if not _cv2_available():
        return []
    import cv2  # lazy

    try:
        img = cv2.imread(str(crop_path))
    except Exception:
        return []
    if img is None or img.size == 0:
        return []
    try:
        up = cv2.resize(
            img, None, fx=_UPSCALE, fy=_UPSCALE, interpolation=cv2.INTER_CUBIC
        )
        up = cv2.copyMakeBorder(
            up, _PAD, _PAD, _PAD, _PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    except Exception:
        return []
    try:
        if float(gray.mean()) < _DARK_MEAN:
            gray = cv2.bitwise_not(gray)
    except Exception:
        pass
    h, w = gray.shape[:2]
    # Odd block size ~ a third of the short side, kept in [3, 51].
    block = min(_ADAPTIVE_MAX_BLOCK, max(11, (min(h, w) // 3) | 1))
    images: dict[str, object] = {"gray": gray}
    try:
        _, images["otsu"] = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except Exception:
        pass
    try:
        images["adaptive"] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, _ADAPTIVE_C,
        )
    except Exception:
        pass
    out_dir = _out_dir()
    stem = Path(str(crop_path)).stem or "strip"
    tag = _tag(str(crop_path))
    out: list[tuple[str, str]] = []
    for name in _VARIANTS:
        im = images.get(name)
        if im is None:
            continue
        path = out_dir / f"{stem}.{tag}__{name}.png"
        try:
            if cv2.imwrite(str(path), im):
                out.append((name, str(path)))
        except Exception:
            continue
    return out


def prep_for_ocr(crop_path: str) -> list[str]:
    """Saved preprocessing variants for one strip -> list of PNG paths.

    Order matches _VARIANTS: gray, otsu, adaptive (whichever succeeded).
    [] when cv2 is absent or the crop won't decode.
    """
    return [p for _, p in _prep_variants(crop_path)]


def ocr_strip(path: str) -> list[dict]:
    """OCR one strip image through every prep variant (psm 7 + whitelist).

    ``path`` may be a raw crop from strip_candidates or any strip-like
    image — it is prepped first. Returns [{"text": ..., "variant": name}]
    for non-empty reads only; [] when pytesseract/tesseract/cv2 is
    missing or every variant reads empty.
    """
    if not _ocr_available():
        return []
    import pytesseract  # lazy

    out: list[dict] = []
    for name, variant_path in _prep_variants(path):
        try:
            text = pytesseract.image_to_string(
                variant_path, config=_TESS_CONFIG
            )
        except Exception:
            continue
        text = (text or "").strip()
        if text:
            out.append({"text": text, "variant": name})
    return out


def _frame_representative(texts) -> str | None:
    """Pick one OCR string to stand for a frame in the temporal vote.

    Burned-in clocks are digit-dense, so the text carrying the most digit
    characters (tie-break: longest) is that frame's best read.
    """
    best: str | None = None
    best_key = (-1, -1)
    for t in texts or []:
        if not isinstance(t, str) or not t.strip():
            continue
        t = t.strip()
        key = (sum(ch.isdigit() for ch in t), len(t))
        if key > best_key:
            best, best_key = t, key
    return best


def temporal_consensus(per_frame_texts: list[list[str]]) -> str | None:
    """Per-character majority vote across frames (OCRSync-style).

    Each frame votes with its single best text; the modal-length group of
    representatives then votes position by position. A character is only
    emitted where at least two frames agree — positions with no pair
    agreement come out as spaces. Returns None under <2 usable frames or
    when the vote produces nothing but blanks.
    """
    reps = [
        r
        for r in (_frame_representative(t) for t in (per_frame_texts or []))
        if r
    ]
    if len(reps) < 2:
        return None
    groups: dict[int, list[str]] = {}
    for s in reps:
        groups.setdefault(len(s), []).append(s)
    # Largest voter pool wins; ties -> longer strings carry more signal.
    voters = max(groups.values(), key=lambda g: (len(g), len(g[0])))
    if len(voters) < 2:
        return None
    out: list[str] = []
    for i in range(len(voters[0])):
        ch, n = Counter(s[i] for s in voters).most_common(1)[0]
        out.append(ch if n >= 2 else " ")
    consensus = "".join(out).strip()
    return consensus or None


def read_timestamps(frame_paths: list) -> dict:
    """Run the full boost recipe over sampled frames.

    For each frame: crop the four candidates, prep each into variants,
    psm-7 OCR every variant, then vote the per-frame texts into one
    consensus string. Returns::

        {"consensus": "raw string" | None,
         "per_frame": {frame_path: [texts, ...], ...},
         "frames_read": <frames that produced >=1 text>}

    Never raises — missing deps or unreadable frames just empty the
    result.
    """
    if isinstance(frame_paths, str):
        frame_paths = [frame_paths]
    per_frame: dict[str, list[str]] = {}
    ordered: list[list[str]] = []
    for fp in frame_paths or []:
        texts: list[str] = []
        try:
            for crop in strip_candidates(fp):
                for r in ocr_strip(crop):
                    t = r.get("text")
                    if t and t not in texts:
                        texts.append(t)
        except Exception:
            texts = []
        per_frame[str(fp)] = texts
        ordered.append(texts)
    return {
        "consensus": temporal_consensus(ordered),
        "per_frame": per_frame,
        "frames_read": sum(1 for t in ordered if t),
    }


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Offline-safe self-test: python3 sundial/ocr_boost.py
    # Pure-python assertions always run; PIL/cv2/tesseract tiers each
    # degrade to SKIP lines when the dep isn't on the machine.
    # ------------------------------------------------------------------
    import sys

    fails = 0

    def check(label, got, want):
        global fails
        ok = got == want
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label}  (got {got!r}, want {want!r})")

    print("ocr_boost self-test")

    # --- 1. temporal_consensus: pure python, always tested --------------
    check(
        "consensus votes out 0->O noise",
        temporal_consensus(
            [["17.10.26 12:30"], ["17.10.26 12:30"], ["17.1O.26 12:3O"]]
        ),
        "17.10.26 12:30",
    )
    check("needs >=2 frames", temporal_consensus([["12:30"]]), None)
    check("empty input", temporal_consensus([]), None)
    check("all-empty frames", temporal_consensus([[], ["  "]]), None)
    check(
        "digit-dense representative wins per frame",
        temporal_consensus(
            [["VFC junk", "17.10.26 12:30"], ["17.10.26 12:30"]]
        ),
        "17.10.26 12:30",
    )
    check(
        "unagreed positions blank out",
        temporal_consensus([["17.10.26 12:30"], ["17.10.26 99:99"]]),
        "17.10.26   :",
    )

    # --- 2. graceful degradation: never raises, always empty ------------
    check("missing frame -> []", strip_candidates("/no/such/frame.png"), [])
    check("missing crop -> []", prep_for_ocr("/no/such/crop.png"), [])
    check("missing strip -> []", ocr_strip("/no/such/strip.png"), [])
    rt = read_timestamps(["/no/such/frame.png"])
    check(
        "read_timestamps shape on dead input",
        (rt["consensus"], rt["frames_read"], rt["per_frame"]["/no/such/frame.png"]),
        (None, 0, []),
    )
    check(
        "read_timestamps empty arg",
        read_timestamps([]),
        {"consensus": None, "per_frame": {}, "frames_read": 0},
    )

    # --- 3. image plumbing: needs PIL to synthesize a frame -------------
    frame_path = None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        print("  SKIP  PIL absent — synthetic frame, crop/prep plumbing untested")
    else:
        try:
            img = Image.new("RGB", (640, 360), (16, 20, 28))
            draw = ImageDraw.Draw(img)
            font = None
            for cand in (
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
            ):
                if Path(cand).exists():
                    font = ImageFont.truetype(cand, 30)
                    break
            if font is None:
                try:
                    font = ImageFont.load_default(size=30)
                except TypeError:
                    font = ImageFont.load_default()
            draw.text((24, 14), "17.10.26 12:30", fill=(240, 240, 240), font=font)
            draw.text((400, 318), "17.10.26 12:30", fill=(240, 240, 240), font=font)
            frame_path = str(_out_dir() / "selftest_frame.png")
            img.save(frame_path)
        except Exception as e:
            print(f"  SKIP  could not synthesize frame: {e}")
            frame_path = None

    # --- 4. cv2 plumbing: crops + prep variants -------------------------
    if frame_path is None:
        print("  SKIP  cv2 plumbing — no frame to run it on")
    elif not _cv2_available():
        print("  SKIP  cv2 absent — strip_candidates/prep_for_ocr untested")
    else:
        crops = strip_candidates(frame_path)
        check("4 candidate crops saved", len(crops), 4)
        check("crops exist on disk", all(Path(p).is_file() for p in crops), True)
        variants = prep_for_ocr(crops[0]) if crops else []
        check("3 prep variants", len(variants), 3)
        check(
            "variant names in filenames",
            [any(p.endswith(f"__{n}.png") for p in variants) for n in _VARIANTS],
            [True, True, True],
        )
        check(
            "prep on garbage path -> []",
            prep_for_ocr(str(_out_dir() / "nope.png")),
            [],
        )

        # --- 5. OCR assertions: only with pytesseract + binary ----------
        if not _ocr_available():
            print("  SKIP  pytesseract/tesseract absent — OCR assertions skipped")
        else:
            hits = []
            for c in crops:
                hits.extend(ocr_strip(c))
            check("psm7 read something", len(hits) > 0, True)
            check(
                "digits survived the whitelist",
                any(any(ch.isdigit() for ch in h["text"]) for h in hits),
                True,
            )
            rt2 = read_timestamps([frame_path])
            check("frames_read on live frame", rt2["frames_read"], 1)
            check(
                "per_frame has texts",
                len(rt2["per_frame"].get(frame_path, [])) > 0,
                True,
            )

    print(f"self-check: {'all passed' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)
