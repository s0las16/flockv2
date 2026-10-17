"""Shadow-consistency stage — flag frames whose shadows point different ways.

A classic deepfake/compositing tell: two objects in the same frame cast
shadows in materially different directions. Real outdoor footage has one
dominant light source (the sun), so every cast shadow in a frame should
point roughly the same way; a composited foreground keeps the lighting of
wherever it was pasted from.

Angles are compared in IMAGE space — every shadow in one frame shares the
camera, so heading cancels out of pairwise differences and no azimuth or
ephemeris step is needed. The angle convention matches
perception.shadow_vector (degrees clockwise from image-up: 0 = toward the
top edge, 90 = right, 180 = down, 270 = left) but is re-derived here from
``base_point`` -> ``shadow_tip`` so this stage never imports the VLM stack.

Stdlib only; the schema import is lazy inside consistency_evidence so the
module imports with zero deps.
"""
from __future__ import annotations

import math
import os

__all__ = ["shadow_angles_per_frame", "consistency_evidence"]

# Per-frame circular spread thresholds (deg). Between them sits a grey band:
# suspicious, but perspective/terrain can bend shadows that far honestly.
AGREE_SPREAD_DEG = 25.0  # <= this: shadows agree — weak support
FLAG_SPREAD_DEG = 45.0   # >  this: shadows disagree — compositing tell


def _azimuth_error(a: float, b: float) -> float:
    """Shortest distance between two azimuths in degrees (0/360 wrap-safe).

    Same formula as solar.azimuth_error, duplicated so this module stays
    importable without the solar stack.
    """
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _shadow_angle_deg(obs: dict) -> float | None:
    """Image-space shadow angle for one locate_shadow observation, or None.

    LOCAL COPY of perception.shadow_vector's math — image coords have
    origin top-left, +x right, +y DOWN, and the angle is the direction the
    shadow points, deg clockwise from image-up. Replicated (not imported)
    so this module does not pull in the perception stack.
    """
    if not isinstance(obs, dict):
        return None
    base, tip = obs.get("base_point"), obs.get("shadow_tip")
    try:
        dx = float(tip[0]) - float(base[0])
        dy = float(tip[1]) - float(base[1])
    except (TypeError, IndexError, KeyError, ValueError):
        return None
    length = math.hypot(dx, dy)
    if not math.isfinite(length) or length <= 1e-6:
        return None  # degenerate vector — direction is meaningless
    return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0


def _spread_deg(angles: list[float]) -> float:
    """Circular spread of a set of image angles: max pairwise azimuth_error."""
    angles = [a for a in angles if math.isfinite(a)]
    spread = 0.0
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            err = _azimuth_error(angles[i], angles[j])
            if err > spread:
                spread = err
    return spread


def shadow_angles_per_frame(
    observations: list[tuple[str, dict]],
) -> dict[str, list[float]]:
    """Group usable shadow angles by frame.

    `observations` is (frame_path, shadow_obs) pairs — e.g. one
    locate_shadow result per picked object. Each pair contributes at most
    one angle; a frame key is always created so callers can tell "frame
    processed, no usable shadow" from "frame absent".
    """
    per_frame: dict[str, list[float]] = {}
    for frame_path, obs in observations:
        angles = per_frame.setdefault(str(frame_path), [])
        ang = _shadow_angle_deg(obs)
        if ang is not None:
            angles.append(ang)
    return per_frame


def consistency_evidence(per_frame: dict[str, list[float]]) -> "Evidence":
    """Build an Evidence card for within-frame shadow-direction agreement.

    For every frame carrying >=2 usable shadow angles the circular spread
    (max pairwise azimuth_error) decides:

      * worst spread > 45 deg        -> CONTRADICTS 0.5 (one bad frame
                                        convicts — it's the tell)
      * >=2 multi-shadow frames, all
        spreads <= 25 deg            -> INCONCLUSIVE 0.3 — agreement
                                        replicated across frames is only
                                        "supports-lite": consistent
                                        lighting is the default for real
                                        footage, not proof
      * <2 multi-shadow frames       -> INCONCLUSIVE 0.15 — most clips
                                        carry at most one shadow per
                                        frame, so there is nothing to
                                        cross-check
      * grey band (25 < spread <=45) -> INCONCLUSIVE 0.2
    """
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # loose-script fallback (same trick as provenance.py)
        from schema import ClaimVerdict, Evidence

    angles_by_frame = {
        f: [a for a in angs if math.isfinite(a)]
        for f, angs in per_frame.items()
    }
    spreads = {
        f: _spread_deg(angs)
        for f, angs in angles_by_frame.items()
        if len(angs) >= 2
    }
    n_with_shadow = sum(1 for angs in angles_by_frame.values() if angs)

    data = {
        "frame_spreads_deg": {f: round(s, 1) for f, s in spreads.items()},
        "frame_angles_deg": {
            f: [round(a, 1) for a in angs]
            for f, angs in angles_by_frame.items()
            if angs
        },
        "n_frames_seen": len(per_frame),
        "n_frames_with_shadow": n_with_shadow,
        "n_multi_shadow_frames": len(spreads),
        "agree_threshold_deg": AGREE_SPREAD_DEG,
        "flag_threshold_deg": FLAG_SPREAD_DEG,
        "method": "max pairwise azimuth_error of image-space shadow "
                  "angles within each frame",
    }

    def _ev(summary, verdict, conf, frames):
        return Evidence(
            type="shadow_consistency",
            summary=summary,
            verdict=verdict,
            confidence=conf,
            data=data,
            frames=frames,
        )

    if not spreads:
        return _ev(
            "at most one measurable shadow per frame — lighting "
            "consistency not checkable",
            ClaimVerdict.INCONCLUSIVE,
            0.15,
            [],
        )

    worst_frame = max(spreads, key=spreads.get)
    worst = spreads[worst_frame]
    data["worst_frame"] = worst_frame
    data["worst_spread_deg"] = round(worst, 1)
    base = os.path.basename(worst_frame) or worst_frame

    if worst > FLAG_SPREAD_DEG:
        bad = [f for f, s in spreads.items() if s > FLAG_SPREAD_DEG]
        return _ev(
            f"shadows in {base} disagree by {worst:.0f}° — "
            "inconsistent lighting",
            ClaimVerdict.CONTRADICTS,
            0.5,
            bad,
        )

    if len(spreads) >= 2 and worst <= AGREE_SPREAD_DEG:
        return _ev(
            f"all {len(spreads)} multi-shadow frames agree within "
            f"{worst:.0f}° — consistent lighting (weak support)",
            ClaimVerdict.INCONCLUSIVE,
            0.3,
            list(spreads),
        )

    if worst <= AGREE_SPREAD_DEG:
        return _ev(
            f"shadows in {base} agree within {worst:.0f}° — consistent "
            "lighting, but a lone multi-shadow frame is thin evidence",
            ClaimVerdict.INCONCLUSIVE,
            0.15,
            list(spreads),
        )

    return _ev(
        f"shadows in {base} spread {worst:.0f}° — above the agreement "
        f"band but under the {FLAG_SPREAD_DEG:.0f}° flag threshold",
        ClaimVerdict.INCONCLUSIVE,
        0.2,
        list(spreads),
    )


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def _obs(angle_deg: float, length: float = 120.0) -> dict:
        """Synthetic locate_shadow obs: shadow points `angle_deg` clockwise
        from image-up (perception convention)."""
        r = math.radians(angle_deg)
        base = (400.0, 400.0)
        tip = (base[0] + length * math.sin(r),
               base[1] - length * math.cos(r))
        return {"base_point": list(base), "shadow_tip": list(tip),
                "object_class": "pole"}

    try:
        from .schema import ClaimVerdict
    except ImportError:
        from schema import ClaimVerdict

    ok = True

    def check(name, ev, want_verdict, note):
        global ok
        good = ev.verdict == want_verdict
        ok = ok and good
        print(f"[{ 'PASS' if good else 'FAIL' }] {name}: {ev.verdict} "
              f"conf={ev.confidence} — {ev.summary} ({note})")

    # required demo: two obs in one frame 10° apart -> agree
    pf = shadow_angles_per_frame(
        [("frame_0001.jpg", _obs(40.0)), ("frame_0001.jpg", _obs(50.0))])
    check("agree-10deg", consistency_evidence(pf),
          ClaimVerdict.INCONCLUSIVE, "10° spread should not flag")

    # required demo: two obs 90° apart -> flag
    pf = shadow_angles_per_frame(
        [("frame_0003.jpg", _obs(20.0)), ("frame_0003.jpg", _obs(110.0))])
    check("flag-90deg", consistency_evidence(pf),
          ClaimVerdict.CONTRADICTS, "90° spread must flag")

    # agreement replicated across >=2 frames -> supports-lite 0.3
    pf = shadow_angles_per_frame([
        ("frame_0001.jpg", _obs(200.0)), ("frame_0001.jpg", _obs(208.0)),
        ("frame_0002.jpg", _obs(195.0)), ("frame_0002.jpg", _obs(204.0)),
    ])
    ev = consistency_evidence(pf)
    check("agree-multi-frame", ev, ClaimVerdict.INCONCLUSIVE,
          "expect conf 0.3")
    ok = ok and abs(ev.confidence - 0.3) < 1e-9

    # single-shadow clip (the common case) -> graceful 0.15
    pf = shadow_angles_per_frame([
        ("frame_0001.jpg", _obs(90.0)),
        ("frame_0002.jpg", _obs(250.0)),
    ])
    ev = consistency_evidence(pf)
    check("single-shadow clip", ev, ClaimVerdict.INCONCLUSIVE,
          "expect conf 0.15")
    ok = ok and abs(ev.confidence - 0.15) < 1e-9

    # wrap-safety: 350° vs 10° are only 20° apart -> must not flag
    pf = shadow_angles_per_frame(
        [("frame_0001.jpg", _obs(350.0)), ("frame_0001.jpg", _obs(10.0))])
    check("wrap-safe", consistency_evidence(pf),
          ClaimVerdict.INCONCLUSIVE, "20° across 0/360 must not flag")

    # unusable obs (missing points) handled gracefully
    pf = shadow_angles_per_frame(
        [("frame_0001.jpg", {}), ("frame_0002.jpg", _obs(90.0))])
    check("unusable obs", consistency_evidence(pf),
          ClaimVerdict.INCONCLUSIVE, "no multi-shadow frame")

    print("\nself-test:", "all PASS" if ok else "FAILURES above")
    sys.exit(0 if ok else 1)
