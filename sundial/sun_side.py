"""Which side of the frame is the sun on?

The CONDITIONS_PROMPT VLM call (see prompts.py) returns, per frame,
``sun: {"visible": bool, "position": [x, y] | null, "description": str}``.
hemisphere.py needs a coarser label — "left" | "right" | "center" — per
frame plus a modal label across the whole clip. This module derives both.

Resolution order inside sun_side_from_conditions():

  1. ``sun.position`` — the sun disc's pixel coords (origin top-left, x
     growing right), classified against horizontal thirds of the frame:
     x < w/3 -> "left", x >= 2w/3 -> "right", middle band -> "center".
     Thirds, not a bare left/right split, because a sun near the optical
     axis gives no usable hemisphere cue. This needs a real frame width:
     the ``frame_w`` argument, else a ``"frame_w"`` field inside cond.
     With no width at all a bare number is ambiguous (pixel vs normalized
     [0, 1]) so we return None rather than guess — a numeric position we
     cannot scale is unusable, full stop.
  2. ``sun.description`` — free-text fallback (the VLM fills this in from
     light direction even when the disc is off-frame). Word-boundary scan:
     the first "left"/"right" keyword wins; "behind"/"overhead"/"center"
     and friends map to "center" (no lateral cue). Nothing usable -> None.

Stdlib only, no package imports — safe for hemisphere.py to import at top
level. ``from sundial.sun_side import ...`` also works standalone.
"""
from __future__ import annotations

import re
from collections import Counter

__all__ = ["sun_side_from_conditions", "aggregate_sides"]

SIDES = ("left", "right", "center")

# Lateral keywords; "leftmost"/"rightward(s)"/"right-hand" count too.
# \b keeps "bright"/"upright"/"frightened" from matching.
_SIDE_RE = re.compile(r"\b(left|right)(?:most|ward|wards|hand)?\b",
                      re.IGNORECASE)

# No-side-of-frame cues: straight up, behind the lens, dead ahead.
_CENTER_RE = re.compile(
    r"\b(behind|overhead|above|zenith|central|middle|midway|"
    r"center|centre|centered|centred)\b",
    re.IGNORECASE,
)


def _position(pos) -> tuple[float, float] | None:
    """Extract a numeric [x, y] pair, or None for null/malformed input."""
    if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
        return None
    x, y = pos
    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           for v in (x, y)):
        return None
    return float(x), float(y)


def _frame_w(cond: dict, frame_w) -> float | None:
    """Resolve frame width: explicit argument wins, cond["frame_w"] next."""
    w = frame_w if frame_w is not None else cond.get("frame_w")
    if isinstance(w, bool) or not isinstance(w, (int, float)) or w <= 0:
        return None
    return float(w)


def _side_from_text(desc) -> str | None:
    """Keyword scan of sun.description. Lateral words beat center cues."""
    if not isinstance(desc, str) or not desc:
        return None
    m = _SIDE_RE.search(desc)
    if m:
        return m.group(1).lower()
    return "center" if _CENTER_RE.search(desc) else None


def sun_side_from_conditions(cond: dict, frame_w: int | None = None) -> str | None:
    """Classify one frame's conditions dict as "left"|"right"|"center".

    ``cond`` is the parsed CONDITIONS_PROMPT JSON (or, tolerantly, its
    ``"sun"`` sub-dict). Returns None when no side can be determined —
    e.g. no sun info, unusable description, or a numeric position with
    no frame width to scale it against.
    """
    if not isinstance(cond, dict):
        return None
    sun = cond.get("sun")
    if sun is None:
        sun = cond  # tolerate being handed the sun sub-dict directly
    if not isinstance(sun, dict):
        return None

    pos = _position(sun.get("position"))
    if pos is not None:
        w = _frame_w(cond, frame_w)
        if w is None:
            return None
        x, y = pos
        # Clearly normalized coords (both axes within [0, 1]) on a real
        # pixel frame: scale x into pixels. A genuine pixel x <= 1 would
        # be sub-pixel — effectively impossible for a sun location.
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 4.0:
            x *= w
        if x < w / 3.0:
            return "left"
        if x >= 2.0 * w / 3.0:
            return "right"
        return "center"

    return _side_from_text(sun.get("description"))


def aggregate_sides(per_frame: list[str | None]) -> str | None:
    """Modal side across per-frame labels.

    Needs a unique mode with >= 2 votes — a clip's single lucky frame or
    an even left/right split is not a hemisphere signal. Returns None for
    empty/all-None input, single-vote modes, and ties.
    """
    counts = Counter(s for s in per_frame if s in SIDES)
    if not counts:
        return None
    ranked = counts.most_common()
    side, votes = ranked[0]
    if votes < 2 or (len(ranked) > 1 and ranked[1][1] == votes):
        return None
    return side


if __name__ == "__main__":
    import sys

    W = 2048
    checks = [
        # --- position path --------------------------------------------------
        ({"sun": {"visible": True, "position": [200, 400], "description": ""}}, W, "left"),
        ({"sun": {"visible": True, "position": [1024, 300], "description": ""}}, W, "center"),
        ({"sun": {"visible": True, "position": [1900, 300], "description": ""}}, W, "right"),
        # position beats description when usable
        ({"sun": {"visible": True, "position": [100, 300], "description": "right side"}}, W, "left"),
        # frame width via cond["frame_w"] when the arg is absent
        ({"frame_w": 1000, "sun": {"visible": True, "position": [900, 50], "description": None}}, None, "right"),
        # normalized [0,1] coords scaled by a known pixel width
        ({"sun": {"visible": True, "position": [0.9, 0.4], "description": ""}}, W, "right"),
        # numeric position but no width anywhere -> unusable, NOT description
        ({"sun": {"visible": True, "position": [200, 400], "description": "left edge"}}, None, None),
        # --- description path -------------------------------------------------
        ({"sun": {"visible": False, "position": None, "description": "sun low on the right edge"}}, None, "right"),
        ({"sun": {"visible": False, "position": None, "description": "light coming from the left"}}, None, "left"),
        ({"sun": {"visible": False, "position": None, "description": "sun behind the camera"}}, None, "center"),
        ({"sun": {"visible": False, "position": None, "description": "directly overhead"}}, None, "center"),
        # --- nothing usable ----------------------------------------------------
        ({}, W, None),
        ({"sun": {"visible": False, "position": None, "description": "overcast, sun not visible"}}, W, None),
    ]
    fails = 0
    for cond, w, want in checks:
        got = sun_side_from_conditions(cond, w)
        ok = "PASS" if got == want else "FAIL"
        if got != want:
            fails += 1
        print(f"  {ok}  sun_side_from_conditions({cond!r}, frame_w={w}) = {got} (want {want})")

    agg_checks = [
        (["left", "left", "right", None, "center"], "left"),   # clear modal side
        (["center", "center", "center"], "center"),
        (["left", "right"], None),                             # 1 vote each
        (["left", "left", "right", "right", "center"], None),  # tied mode
        (["left"], None),                                      # needs >= 2 frames
        ([None, None], None),
        ([], None),
    ]
    for frames, want in agg_checks:
        got = aggregate_sides(frames)
        ok = "PASS" if got == want else "FAIL"
        if got != want:
            fails += 1
        print(f"  {ok}  aggregate_sides({frames}) = {got} (want {want})")

    total = len(checks) + len(agg_checks)
    print(f"self-check: {total - fails}/{total} passed")
    sys.exit(1 if fails else 0)
