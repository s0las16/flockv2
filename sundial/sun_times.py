"""Day/twilight/night cross-check — clip lighting vs expected civil state.

`day_state` bins the sun's elevation at the claimed place+time into
day / civil twilight / nautical twilight / night using the standard
boundaries (0°, -6°, -12° below the geometric horizon — astronomical
dark at -18° is beyond what a VLM can grade, so "night" starts at -12).

`lighting_evidence` compares that expectation with the lighting words a
VLM reports via describe_conditions (perception stage). A clip that
reads as night while the claimed instant has the sun 60° up is a hard
physical tell — solar elevation is not something a fake claim can
fudge — so day↔night mismatches come out CONTRADICTS at 0.6. Twilight
expectations stay INCONCLUSIVE: a VLM cannot tell civil dusk from a
heavy overcast, and streetlights fire in both.

Stdlib only at import (datetime). The solar engine — .suncalc.sun_pos,
falling back to .solar.sun_position (pysolar) — and the schema types
are imported lazily inside the functions so the module loads with zero
package deps and `python3 sundial/sun_times.py` still self-tests.
Nothing here raises: failures degrade to state "unknown" /
INCONCLUSIVE rather than killing the pipeline.
"""
from __future__ import annotations

from datetime import date as _date, datetime, time as _time, timezone

# Sun-elevation boundaries in degrees (geometric horizon = 0).
DAY_MIN_ELEV = 0.0          # >0        -> day
CIVIL_MIN_ELEV = -6.0       # -6..0     -> civil twilight
NAUTICAL_MIN_ELEV = -12.0   # -12..-6   -> nautical twilight (<-12 night)

UNKNOWN = "unknown"
DAY = "day"
CIVIL_TWILIGHT = "civil_twilight"
NAUTICAL_TWILIGHT = "nautical_twilight"
NIGHT = "night"
_TWILIGHT_STATES = (CIVIL_TWILIGHT, NAUTICAL_TWILIGHT)

# Lighting vocabulary, matched as whole words after lowercasing and
# splitting on non-letters. "lit" is only a night tell as a standalone
# token — "daylit"/"sunlit" stay single tokens and never match it.
_DAY_WORDS = (
    "day", "daylight", "daytime", "bright", "sunny", "sunshine",
    "sunlit", "daylit", "overcast",
)
_NIGHT_WORDS = (
    "night", "nighttime", "dark", "darkness", "dim", "unlit",
    "lit", "lamplit", "moonlit", "starlit", "floodlit", "torchlit",
    "streetlight", "streetlights", "lamppost",
)
_TWILIGHT_WORDS = (
    "twilight", "dusk", "dawn", "sunset", "sunrise", "sundown",
    "daybreak", "nightfall", "gloaming", "golden",
)
# Multi-word tells checked on the padded, space-normalised text.
_TWILIGHT_PHRASES = (
    "golden hour", "blue hour", "first light", "last light",
    "sun going down", "half light",
)


def _as_utc(dt) -> datetime:
    """Accept datetime / date / ISO string -> tz-aware UTC (naive = UTC)."""
    if isinstance(dt, datetime):
        d = dt
    elif isinstance(dt, _date):
        d = datetime.combine(dt, _time.min)
    else:
        s = str(dt).strip()
        if s.lower().endswith("z"):
            s = s[:-1] + "+00:00"  # fromisoformat <3.11 rejects "Z"
        d = datetime.fromisoformat(s)
    return (d.astimezone(timezone.utc) if d.tzinfo is not None
            else d.replace(tzinfo=timezone.utc))


def _ensure_root_on_path() -> None:
    """Put the repo root on sys.path so `sundial.x` imports work when this
    file is run as a plain script (`python3 sundial/sun_times.py`)."""
    try:
        import os
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass


def _import_from(module_names: "tuple[str, ...]", attr: str):
    """First importable `attr` across candidate module names, else None."""
    import importlib
    for modname in module_names:
        try:
            fn = getattr(importlib.import_module(modname), attr, None)
        except Exception:
            continue
        if callable(fn):
            return fn
    return None


def _schema():
    """Lazy schema import; package-relative first, then script-mode fallbacks."""
    try:
        from .schema import ClaimVerdict, Evidence
        return ClaimVerdict, Evidence
    except Exception:
        pass
    _ensure_root_on_path()
    import importlib
    for modname in ("sundial.schema", "schema"):
        try:
            mod = importlib.import_module(modname)
            return mod.ClaimVerdict, mod.Evidence
        except Exception:
            continue
    return None, None


def _elev_az(pos) -> "tuple[float | None, float | None]":
    """Pull (elevation_deg, azimuth_deg) out of whatever the engine returns."""
    if pos is None:
        return None, None
    if isinstance(pos, dict):
        e = pos.get("elevation_deg", pos.get("elevation",
                 pos.get("altitude_deg", pos.get("altitude"))))
        a = pos.get("azimuth_deg", pos.get("azimuth"))
        return e, a
    if isinstance(pos, (list, tuple)) and pos:
        return pos[0], (pos[1] if len(pos) > 1 else None)
    e = getattr(pos, "elevation_deg", getattr(pos, "elevation", None))
    a = getattr(pos, "azimuth_deg", getattr(pos, "azimuth", None))
    return e, a


def _sun_elev_az(when: datetime, lat: float, lon: float):
    """-> (elevation_deg, azimuth_deg) or None. Never raises.

    Preferred engine is .suncalc.sun_pos; fall back to
    .solar.sun_position (pysolar). Both engines return a dict whose
    elevation key _elev_az knows how to read. suncalc's calling
    convention isn't pinned yet, so both (when, lat, lon) and
    (lat, lon, when) arg orders are tried — the wrong order dies on
    datetime arithmetic inside the engine and we simply try the other.
    """
    names = []
    if __package__:
        names.append(f"{__package__}.suncalc")
    _ensure_root_on_path()
    fn = _import_from(tuple(names) + ("sundial.suncalc", "suncalc"), "sun_pos")
    if fn is None:
        solar = ([f"{__package__}.solar"] if __package__ else []) + \
                ["sundial.solar", "solar"]
        fn = _import_from(tuple(solar), "sun_position")
    if fn is None:
        return None
    for args in ((when, lat, lon), (lat, lon, when)):
        try:
            e, a = _elev_az(fn(*args))
            if e is None:
                continue
            return float(e), (float(a) if a is not None else None)
        except Exception:
            continue
    return None


def day_state(dt_utc, lat, lon) -> dict:
    """-> {"state", "elevation_deg", "azimuth_deg"} for a place+time.

    `dt_utc` may be a datetime, date, or ISO string; naive/naive-looking
    values are read as UTC per repo convention. On any failure the state
    is "unknown" with elevation None — callers should treat it as
    INCONCLUSIVE, not night.
    """
    try:
        w = _as_utc(dt_utc)
        res = _sun_elev_az(w, float(lat), float(lon))
        if res is None:
            return {"state": UNKNOWN, "elevation_deg": None, "azimuth_deg": None,
                    "error": "no solar engine available"}
        elev, az = res
        if elev > DAY_MIN_ELEV:
            state = DAY
        elif elev >= CIVIL_MIN_ELEV:
            state = CIVIL_TWILIGHT
        elif elev >= NAUTICAL_MIN_ELEV:
            state = NAUTICAL_TWILIGHT
        else:
            state = NIGHT
        return {
            "state": state,
            "elevation_deg": round(elev, 2),
            "azimuth_deg": round(az, 1) if az is not None else None,
            "dt_utc": w.isoformat(),
        }
    except Exception as e:
        return {"state": UNKNOWN, "elevation_deg": None, "azimuth_deg": None,
                "error": str(e) or type(e).__name__}


def _scan_lighting(text: str) -> "tuple[list, list, list]":
    """-> (day_hits, night_hits, twilight_hits) found in free text."""
    t = " ".join("".join(ch if ch.isalpha() else " "
                         for ch in str(text).lower()).split())
    toks = set(t.split())
    padded = f" {t} "
    day = [w for w in _DAY_WORDS if w in toks]
    night = [w for w in _NIGHT_WORDS if w in toks]
    twi = [w for w in _TWILIGHT_WORDS if w in toks]
    twi += [p for p in _TWILIGHT_PHRASES if f" {p} " in padded]
    return day, night, twi


def _observed_text(observed) -> str:
    """Normalise the `observed` arg into searchable text.

    Accepts the raw lighting string ("bright daylight"), a full
    describe_conditions dict (pulls .lighting plus the streetlights /
    sun-visible tells), or anything str()-able.
    """
    if observed is None:
        return ""
    if isinstance(observed, str):
        return observed
    if isinstance(observed, dict):
        parts = [str(observed.get("lighting") or "")]
        if str(observed.get("streetlights") or "").lower() == "on":
            parts.append("streetlights on")
        sun = observed.get("sun")
        if (isinstance(sun, dict) and sun.get("visible")) or \
                observed.get("sun_visible") is True:
            parts.append("sunny")
        return " ".join(p for p in parts if p)
    return str(observed)


def _obs_class(obs_text: str) -> "str | None":
    """Classify lighting text -> 'day' | 'night' | 'twilight' | None.

    Twilight wins when present — "dusk" says more than "dark". Day and
    night words together ("bright night market") are self-contradictory
    and degrade to None rather than picking a side.
    """
    day, night, twi = _scan_lighting(obs_text)
    if twi:
        return "twilight"
    if day and night:
        return None
    if day:
        return "day"
    if night:
        return "night"
    return None


def lighting_evidence(state: dict, observed: "str | dict | None" = None,
                      claim_desc: str = "") -> "Evidence":
    """Build an Evidence card: expected civil state vs observed lighting.

    `state` is a day_state() result ({"state", "elevation_deg", ...});
    `observed` is the lighting string from describe_conditions (or the
    whole conditions dict); `claim_desc` is optional free-text context
    quoted back in the summary.

    Verdicts: state day + day words, or state night + night words ->
    SUPPORTS 0.55; day-state vs night words or the reverse ->
    CONTRADICTS 0.6 (hard physical tell); expected twilight ->
    INCONCLUSIVE unless the observation names twilight outright ->
    SUPPORTS 0.45; missing/unreadable observation or unknown state ->
    INCONCLUSIVE. Never raises — on internal failure the verdict is
    INCONCLUSIVE with the error in data, or None if schema itself is
    unimportable.
    """
    ClaimVerdict, Evidence = _schema()
    if Evidence is None:
        return None
    try:
        st = dict(state or {})
        exp = st.get("state", UNKNOWN)
        elev = st.get("elevation_deg")
        obs_text = _observed_text(observed)
        obs_class = _obs_class(obs_text)
        day_hits, night_hits, twi_hits = _scan_lighting(obs_text)

        data = {
            "expected_state": exp,
            "elevation_deg": elev,
            "azimuth_deg": st.get("azimuth_deg"),
            "observed": obs_text or None,
            "observed_class": obs_class,
            "matched_words": {"day": day_hits, "night": night_hits,
                              "twilight": twi_hits},
            "claim_desc": claim_desc or "",
            "thresholds_deg": {"day": DAY_MIN_ELEV, "civil": CIVIL_MIN_ELEV,
                               "nautical": NAUTICAL_MIN_ELEV},
        }
        elev_txt = (f"{elev:.1f}°" if isinstance(elev, (int, float))
                    else "unknown elevation")
        claim_txt = f" (claim: {claim_desc})" if claim_desc else ""

        def _ev(verdict, conf, summary, extra=None):
            return Evidence(type="sun_times", summary=summary + claim_txt,
                            verdict=verdict, confidence=conf,
                            data={**data, **(extra or {})})

        if exp == UNKNOWN or "state" not in st:
            return _ev(ClaimVerdict.INCONCLUSIVE, 0.0,
                       "could not compute the expected day/night state "
                       "at the claimed place+time",
                       {"error": st.get("error")})
        if not obs_text.strip():
            return _ev(ClaimVerdict.INCONCLUSIVE, 0.2,
                       f"no lighting observation to compare; claimed "
                       f"place+time expects {exp.replace('_', ' ')} "
                       f"(sun {elev_txt})")
        if obs_class is None:
            return _ev(ClaimVerdict.INCONCLUSIVE, 0.2,
                       f"lighting description {obs_text!r} is ambiguous; "
                       f"claimed place+time expects "
                       f"{exp.replace('_', ' ')} (sun {elev_txt})")

        exp_txt = exp.replace("_", " ")
        if exp == DAY:
            if obs_class == "day":
                return _ev(ClaimVerdict.SUPPORTS, 0.55,
                           f"clip lighting reads as daytime — consistent "
                           f"with the sun {elev_txt} up at the claimed "
                           f"place+time")
            return _ev(ClaimVerdict.CONTRADICTS, 0.6,
                       f"clip lighting reads as {obs_class} "
                       f"({obs_text!r}) but the claimed place+time has "
                       f"the sun {elev_txt} above the horizon")
        if exp == NIGHT:
            if obs_class == "day":
                return _ev(ClaimVerdict.CONTRADICTS, 0.6,
                           f"clip lighting reads as daytime "
                           f"({obs_text!r}) but the claimed place+time "
                           f"is deep night (sun {elev_txt} below the "
                           f"horizon)")
            # Twilight words sit in the dark-side group: a dusk-looking
            # clip can't convict a night claim on its own (VLM dusk
            # calls are loose), it merely agrees the sun is gone.
            return _ev(ClaimVerdict.SUPPORTS, 0.55,
                       f"clip lighting reads as {obs_class} — consistent "
                       f"with the sun {elev_txt} below the horizon at "
                       f"the claimed place+time")
        # Expected twilight: only an explicit twilight call supports;
        # day/night-looking clips stay inconclusive — the sun sits just
        # below the horizon where ambient light is genuinely ambiguous.
        if exp in _TWILIGHT_STATES:
            if obs_class == "twilight":
                return _ev(ClaimVerdict.SUPPORTS, 0.45,
                           f"clip explicitly reads as twilight — "
                           f"consistent with {exp_txt} (sun {elev_txt}) "
                           f"at the claimed place+time")
            return _ev(ClaimVerdict.INCONCLUSIVE, 0.25,
                       f"claimed place+time is {exp_txt} (sun {elev_txt}) "
                       f"but the clip reads as {obs_class}; ambient "
                       f"light in twilight is too ambiguous to weigh")
        return _ev(ClaimVerdict.INCONCLUSIVE, 0.2,
                   f"unhandled expected state {exp!r}")
    except Exception as e:  # never raise — degrade to INCONCLUSIVE
        try:
            return Evidence(
                type="sun_times",
                summary=f"lighting check failed internally: {e}",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                data={"error": str(e) or type(e).__name__},
            )
        except Exception:
            return None


if __name__ == "__main__":
    import sys

    LONDON = (51.5074, -0.1278)
    ok = True

    def check(name, got, want):
        global ok
        good = got == want
        ok &= good
        print(f"  {name}: {got!r} (want {want!r}) [{'ok' if good else 'MISMATCH'}]")

    print("sun_times self-test")
    st = day_state("2026-06-15T12:00:00Z", *LONDON)
    check("london 2026-06-15T12Z state", st["state"], "day")
    e = st["elevation_deg"]
    good = e is not None and 58.0 <= e <= 66.0
    ok &= good
    print(f"  elevation: {e} (~62 expected) [{'ok' if good else 'MISMATCH'}]")

    ev = lighting_evidence(st, "night")
    check("day state + 'night' obs", ev.verdict.value, "contradicts")

    st_night = day_state("2026-01-15T02:00:00Z", *LONDON)
    check("london 2026-01-15T02Z state", st_night["state"], "night")
    ev2 = lighting_evidence(st_night, "bright daylight")
    check("night state + daylight obs", ev2.verdict.value, "contradicts")

    st_civ = day_state("2026-06-15T20:45:00Z", *LONDON)   # sun ~-3.7°
    check("london 20:45Z jun15 state", st_civ["state"], "civil_twilight")
    ev3 = lighting_evidence(st_civ, "dusk")
    check("civil twilight + 'dusk' obs", ev3.verdict.value, "supports")
    ev4 = lighting_evidence(st_civ, "night")
    check("civil twilight + 'night' obs", ev4.verdict.value, "inconclusive")

    st_naut = day_state("2026-06-15T21:30:00Z", *LONDON)  # sun ~-8.2°
    check("london 21:30Z jun15 state", st_naut["state"], "nautical_twilight")

    ev5 = lighting_evidence(st, "bright daylight")
    check("day + 'bright daylight' obs", ev5.verdict.value, "supports")
    ev6 = lighting_evidence(st, None)
    check("day + missing obs", ev6.verdict.value, "inconclusive")
    ev7 = lighting_evidence(st, {"lighting": "overcast",
                                 "streetlights": "off"})
    check("day + dict obs 'overcast'", ev7.verdict.value, "supports")
    check("bad-input state", day_state("garbage", *LONDON)["state"], "unknown")
    ev8 = lighting_evidence(day_state("garbage", *LONDON), "night")
    check("unknown state + obs", ev8.verdict.value, "inconclusive")

    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)
