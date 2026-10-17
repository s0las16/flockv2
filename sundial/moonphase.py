"""Lunar check — does the moon in a night clip match the claimed date?

Pure-math approximation: phase angle from the mean synodic month
(29.530588853 days) anchored at the 2000-01-06 18:14 UTC new moon, the
same epoch Conway-style mental moon-age methods use. Mean-lunation math
is good to roughly half a day — plenty for "clip shows a full moon but
the claimed date sat near new moon" contradictions.

Stdlib only; the schema import is lazy inside moon_evidence so this
module imports with zero package deps.
"""
from __future__ import annotations

import math
from datetime import date as _date, datetime, time as _time, timezone

SYNODIC_DAYS = 29.530588853

# Reference new moon: 2000-01-06 18:14 UTC (JD 2451550.26).
REF_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)

# Conway-style sectors: eight slices of the lunation centred on new.
_PHASE_NAMES = (
    "new",
    "waxing crescent",
    "first quarter",
    "waxing gibbous",
    "full",
    "waning gibbous",
    "last quarter",
    "waning crescent",
)

# Full name -> coarse class a VLM can plausibly call out. Waxing and
# waning collapse together: distinguishing them needs the lit side's
# orientation, which is hemisphere-dependent and easy to flub.
_COARSE = {
    "new": "new",
    "waxing crescent": "crescent",
    "first quarter": "quarter",
    "waxing gibbous": "gibbous",
    "full": "full",
    "waning gibbous": "gibbous",
    "last quarter": "quarter",
    "waning crescent": "crescent",
}
_ORDER = ("new", "crescent", "quarter", "gibbous", "full")

# Free-text VLM descriptions -> coarse class.
_ALIASES = {
    "new": "new",
    "new moon": "new",
    "no moon": "new",
    "dark": "new",
    "crescent": "crescent",
    "waxing crescent": "crescent",
    "waning crescent": "crescent",
    "thin crescent": "crescent",
    "sliver": "crescent",
    "quarter": "quarter",
    "first quarter": "quarter",
    "last quarter": "quarter",
    "third quarter": "quarter",
    "half": "quarter",
    "half moon": "quarter",
    "gibbous": "gibbous",
    "waxing gibbous": "gibbous",
    "waning gibbous": "gibbous",
    "full": "full",
    "full moon": "full",
    "nearly full": "full",
    "almost full": "full",
}

# Adjacent-class boundary slack keyed by frozenset({a, b}) ->
# (min_illum, max_illum); a None bound is open. A one-class mismatch is
# forgiven when the computed illumination sits inside the blur zone —
# e.g. a 97% disc is technically gibbous but every human calls it full.
_SLACK = {
    frozenset(("new", "crescent")): (None, 0.08),
    frozenset(("crescent", "quarter")): (0.28, 0.45),
    frozenset(("quarter", "gibbous")): (0.62, 0.75),
    frozenset(("gibbous", "full")): (0.93, None),
}


def _aware(when: datetime) -> datetime:
    # Sundial works in UTC end to end; naive datetimes are read as UTC.
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _as_datetime(dt) -> datetime:
    """Accept datetime / date / ISO string -> aware datetime (UTC assumed)."""
    if isinstance(dt, datetime):
        return _aware(dt)
    if isinstance(dt, _date):
        return datetime.combine(dt, _time.min, tzinfo=timezone.utc)
    s = str(dt).strip()
    if s.lower().endswith("z"):
        s = s[:-1] + "+00:00"  # fromisoformat <3.11 rejects "Z"
    return _aware(datetime.fromisoformat(s))


def moon_phase(dt) -> dict:
    """-> {"illumination": 0..1, "phase_name": str} for a datetime/ISO string.

    Age since REF_NEW_MOON modulo the mean synodic month. Illumination is
    the cosine approximation (0 at new, 1 at full); phase_name is the
    Conway-style sector the moment falls into (~±1.85 days each).
    """
    w = _as_datetime(dt)
    age = (w - REF_NEW_MOON).total_seconds() / 86400.0 % SYNODIC_DAYS
    frac = age / SYNODIC_DAYS  # 0 = new, 0.5 = full
    illumination = (1.0 - math.cos(2.0 * math.pi * frac)) / 2.0
    idx = int(frac * 8 + 0.5) % 8
    return {"illumination": round(illumination, 4), "phase_name": _PHASE_NAMES[idx]}


def _coarse_from_desc(desc: str) -> str | None:
    """Map a free-text moon description to new/crescent/quarter/gibbous/full.

    Substring match over aliases, longest first ("waning crescent" before
    "crescent"); None when nothing recognizable is said.
    """
    d = str(desc or "").strip().lower()
    if not d:
        return None
    for key in sorted(_ALIASES, key=len, reverse=True):
        if key in d:
            return _ALIASES[key]
    return None


def moon_evidence(claimed_time_iso: str, observed: dict | None = None) -> "Evidence":
    """Build an Evidence card comparing an observed moon to the claimed date.

    `observed` is a VLM conditions-style dict that may carry
    {"moon_visible": bool, "moon_phase_desc": "crescent|gibbous|full|..."}.
    claimed_time_iso follows the solar stage's convention: tz-aware is
    honoured, naive is read as UTC — callers with a local claim should
    convert to UTC first (see pipeline._claimed_utc).

    Verdicts: described phase conflicts with the computed one ->
    CONTRADICTS (~0.6, grows with class distance); matches -> SUPPORTS
    (~0.5, a bit less for boundary slack); nothing usable observed ->
    INCONCLUSIVE with the computed phase still attached in `data`.
    """
    from .schema import ClaimVerdict, Evidence  # lazy: zero-dep module import

    claimed = _as_datetime(claimed_time_iso)
    phase = moon_phase(claimed)
    pct = round(phase["illumination"] * 100)
    comp = _COARSE[phase["phase_name"]]

    obs = dict(observed or {})
    visible = obs.get("moon_visible")
    if isinstance(visible, str):
        visible = visible.strip().lower() in ("true", "yes", "1")
    desc = str(obs.get("moon_phase_desc") or "").strip().lower()
    obs_coarse = _coarse_from_desc(desc)

    data = {
        "claimed_time_utc": claimed.isoformat(),
        "computed_phase": phase["phase_name"],
        "computed_coarse": comp,
        "illumination": phase["illumination"],
        "observed": {"moon_visible": visible, "moon_phase_desc": desc or None},
        "method": f"mean synodic month {SYNODIC_DAYS} d from "
                  "2000-01-06 18:14 UTC new moon",
    }

    if not desc and not visible:
        # Nothing to check — the moon may be out of frame, below the
        # horizon, or clouded out; absence alone can't convict a claim.
        return Evidence(
            type="lunar",
            summary=(f"no moon observed in clip; claimed date has a "
                     f"{phase['phase_name']} moon ({pct}% lit)"),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )
    if obs_coarse is None:
        return Evidence(
            type="lunar",
            summary=(f"moon reported in clip but phase description "
                     f"{desc!r} is unclassifiable; computed "
                     f"{phase['phase_name']} ({pct}% lit)"),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )

    dist = abs(_ORDER.index(obs_coarse) - _ORDER.index(comp))
    if dist == 0:
        return Evidence(
            type="lunar",
            summary=(f"clip moon reads '{desc}' — consistent with the "
                     f"{phase['phase_name']} moon ({pct}% lit) on the "
                     f"claimed date"),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=0.5,
            data=data,
        )

    if dist == 1:
        lo, hi = _SLACK.get(frozenset((obs_coarse, comp)), (None, None))
        if (lo is None or phase["illumination"] >= lo) and (
            hi is None or phase["illumination"] <= hi
        ):
            return Evidence(
                type="lunar",
                summary=(f"clip moon reads '{desc}' — boundary case, still "
                         f"consistent with a {pct}% lit "
                         f"{phase['phase_name']} moon"),
                verdict=ClaimVerdict.SUPPORTS,
                confidence=0.45,
                data={**data, "boundary_slack": True},
            )

    return Evidence(
        type="lunar",
        summary=(f"clip shows a {desc} moon but the claimed date has a "
                 f"{phase['phase_name']} moon ({pct}% lit)"),
        verdict=ClaimVerdict.CONTRADICTS,
        confidence=round(min(0.7, 0.55 + 0.05 * dist), 2),
        data={**data, "class_distance": dist},
    )


if __name__ == "__main__":
    import sys

    dt = _as_datetime(sys.argv[1]) if len(sys.argv) > 1 else datetime.now(timezone.utc)
    p = moon_phase(dt)
    print(f"{dt.isoformat()} -> {p['phase_name']} ({p['illumination'] * 100:.0f}% lit)")

    # Sanity checks: eclipses pin the phase — a solar eclipse is a new
    # moon by definition, a lunar eclipse a full moon.
    checks = [
        ("2000-01-06T18:14", "new"),   # the reference epoch itself
        ("2000-01-21T04:40", "full"),  # total lunar eclipse
        ("2024-04-08T18:21", "new"),   # total solar eclipse
        ("2022-11-08T11:02", "full"),  # total lunar eclipse
    ]
    ok = True
    for iso, want in checks:
        got = moon_phase(iso)["phase_name"]
        ok &= got == want
        flag = "ok" if got == want else "MISMATCH"
        print(f"  check {iso}: {got} (want {want}) [{flag}]")
    if not ok:
        sys.exit(1)
