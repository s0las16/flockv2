"""Frame quality triage — cheap blur/darkness gate before VLM calls.

VLM analysis is the expensive stage in the pipeline, and a meaningful share
of sampled frames are unusable for it: motion blur, crushed night shadows,
blown highlights, focus misses. This module scores each extracted frame
with a couple of PIL-only statistics and lets callers drop the hopeless
ones before spending a VLM call.

Per frame we compute:
  * brightness — mean L-channel value (0..255) of a downscaled gray
  * sharpness  — variance-of-Laplacian approximation: convolve the
    downscaled gray with a 4-neighbour Laplacian kernel (offset to mid-gray
    so signed responses survive the uint8 round-trip), then sum squared
    deviations over the histogram. Blur collapses this variance fast.
  * score      — geometric mean of an exposure triangle and a saturating
    sharpness term, so a single bad axis (pitch black OR smeared) tanks
    the composite; both must be at least half-decent to score well.

All of it is heuristic triage — quality_evidence() always reports
INCONCLUSIVE at low confidence: a blurry clip is weak context for the
adjudicator, never a vote on the claim itself.

Stdlib only at module level; PIL is imported lazily inside frame_score()
so `import sundial.frame_quality` works with zero third-party deps and
score calls degrade to None when Pillow is absent.
"""
from __future__ import annotations

import statistics

__all__ = ["frame_score", "usable_frames", "quality_evidence"]

_WORK_EDGE = 256          # downscale gray so the max side is <= this
_LAP_KERNEL = (0, -1, 0, -1, 4, -1, 0, -1, 0)  # 4-neighbour Laplacian
_LAP_SCALE = 4.0          # |L|<=508 lands inside the +/-127 bins around offset
_LAP_OFFSET = 128.0       # mid-gray bias so negative responses stay readable
_SHARP_KNEE = 60.0        # lap-var of a "half-sharp" frame; sharp01 saturates
_BRIGHT_MID = 128.0       # nominal well-exposed mean-L
_BRIGHT_SPAN = 113.0      # mean-L this far from mid scores 0 (crushed/blown)
_MIN_USABLE_SCORE = 0.25  # shared default: usable_frames() + quality_evidence()


def _laplacian_variance(gray) -> float:
    """Variance-of-Laplacian proxy from a filtered uint8 image.

    The kernel output is biased by _LAP_OFFSET so the response is centred
    in the 0..255 range; deviations from 128 are the (scaled) Laplacian.
    Squared-deviation sum over the 256-bin histogram is the variance —
    a one-pass "gradient sum" instead of a numpy var() call.
    """
    from PIL import ImageFilter

    lap = gray.filter(
        ImageFilter.Kernel(
            (3, 3), _LAP_KERNEL, scale=_LAP_SCALE, offset=_LAP_OFFSET
        )
    )
    hist = lap.histogram()
    n = gray.size[0] * gray.size[1]
    return sum(c * (v - _LAP_OFFSET) ** 2 for v, c in enumerate(hist)) / max(1, n)


def frame_score(frame_path: str) -> dict | None:
    """Score one frame's VLM-worthiness; None on any failure.

    -> {"width", "height", "brightness", "sharpness", "score"} where
    brightness is mean L (0..255), sharpness the Laplacian-variance proxy
    (unbounded, ~0 for flat/blurry), and score a 0..1 composite.

    None when PIL is absent, the path can't be opened as an image, or the
    decode blows up — triage must never take the pipeline down with it.
    """
    try:
        from PIL import Image, ImageStat  # lazy: module imports without PIL
    except ImportError:
        return None

    try:
        with Image.open(frame_path) as img:
            width, height = img.size
            gray = img.convert("L")
    except Exception:
        return None

    # Cheap to score the small copy; box-thumbnail also damps codec noise,
    # which otherwise inflates the sharpness proxy on flat frames.
    if max(gray.size) > _WORK_EDGE:
        gray.thumbnail((_WORK_EDGE, _WORK_EDGE))

    brightness = ImageStat.Stat(gray).mean[0]
    lap_var = _laplacian_variance(gray)

    # Exposure triangle: 1.0 at mid-gray, 0.0 at/under ~15 or over ~241.
    bright01 = max(0.0, 1.0 - abs(brightness - _BRIGHT_MID) / _BRIGHT_SPAN)
    # Saturating sharpness term: var == _SHARP_KNEE -> 0.5.
    sharp01 = lap_var / (lap_var + _SHARP_KNEE)
    # Geometric mean: a dark-but-sharp or bright-but-blurry frame still
    # wastes a VLM call, so either axis alone must be able to sink it.
    score = (bright01 * sharp01) ** 0.5

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 1),
        "sharpness": round(lap_var, 1),
        "score": round(score, 3),
    }


def usable_frames(frames: list[str], min_score: float = _MIN_USABLE_SCORE) -> list[str]:
    """Keep frames worth a VLM call, in original capture order.

    Scores each frame, ranks by score descending for the quality pass,
    keeps everything >= min_score, then restores the original ordering —
    the clip's temporal sequence matters to downstream stages more than
    marginal quality deltas between kept frames. Unscorable paths (None)
    are dropped.
    """
    scored: list[tuple[float, int, str]] = []
    for idx, path in enumerate(frames):
        s = frame_score(path)
        if s is not None:
            scored.append((s["score"], idx, path))
    scored.sort(key=lambda t: t[0], reverse=True)  # quality first
    kept = [(idx, p) for sc, idx, p in scored if sc >= min_score]
    kept.sort()  # then capture order
    return [p for _, p in kept]


def quality_evidence(scores: dict[str, dict]) -> "Evidence":
    """Package frame_score() results as an INCONCLUSIVE 'frames' card.

    `scores` maps frame path -> frame_score() dict (None entries allowed —
    they count as scored-failures toward total, never toward usable).
    Descriptive only: "7/8 frames usable (median quality 0.62)" at fixed
    confidence 0.2 — coverage is context for the adjudicator, not a vote.
    """
    from .schema import ClaimVerdict, Evidence  # lazy: keeps module dep-free

    per_frame = dict(scores or {})
    vals = [
        s["score"] for s in per_frame.values()
        if isinstance(s, dict) and isinstance(s.get("score"), (int, float))
    ]
    total = len(per_frame)
    usable = sum(1 for v in vals if v >= _MIN_USABLE_SCORE)
    median_q = statistics.median(vals) if vals else 0.0

    if total:
        summary = f"{usable}/{total} frames usable (median quality {median_q:.2f})"
    else:
        summary = "no frames scored"

    return Evidence(
        type="frames",
        summary=summary,
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.2,
        data={
            "usable_frames": usable,
            "total_frames": total,
            "median_score": round(median_q, 4),
            "frame_scores": {
                p: s.get("score") for p, s in per_frame.items()
                if isinstance(s, dict)
            },
            "per_frame": per_frame,
        },
    )


if __name__ == "__main__":
    # Self-test: synthetic sharp vs blurred vs crushed frames in a tempdir.
    try:
        from PIL import Image, ImageFilter
    except ImportError:
        print("PIL not installed — frame_quality self-test skipped")
        raise SystemExit(0)

    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sundial_frameq_") as td:
        sharp = Image.effect_noise((320, 240), 48)  # iid noise: max gradients
        paths = {
            "sharp": os.path.join(td, "sharp.png"),
            "blurry": os.path.join(td, "blurry.png"),
            "dark": os.path.join(td, "dark.png"),
        }
        sharp.save(paths["sharp"])
        sharp.filter(ImageFilter.GaussianBlur(6)).save(paths["blurry"])
        Image.eval(sharp, lambda v: v * 14 // 255).save(paths["dark"])  # ~L 7

        for name, p in paths.items():
            print(f"{name:6s}: {frame_score(p)}")
        print("missing: ", frame_score(os.path.join(td, "nope.png")))

        order = [paths["blurry"], paths["sharp"], paths["dark"],
                 os.path.join(td, "nope.png")]
        print("usable:", [os.path.basename(f) for f in usable_frames(order)])

        ev = quality_evidence({p: frame_score(p) for p in order})
        print("evidence:", ev.summary, f"(conf {ev.confidence})")
