"""Sign-style stage — road-sign design conventions as a regional cue.

Road signage is a strong geolocation tell because sign *systems* are
codified per country, not chosen per town:

  * UK/Ireland   — Vienna-flavoured but idiosyncratic: triangular
                   white/red warnings, circular red-bordered speed
                   limits, blue motorway boards, double-yellow and
                   zigzag road markings
  * US/Canada    — MUTCD: yellow diamond warnings, green freeway guide
                   signs, black-on-white rectangular SPEED LIMIT boards,
                   yellow centre lines
  * Vienna conv. — most of the EU/continental Europe: triangular
                   white/red warnings, circular red-bordered speed
                   limits, blue (sometimes green) motorway/direction
                   boards, white centre lines; metric
  * Japan        — MUTCD-flavoured warnings (yellow diamond) but
                   Vienna-style red-bordered speed circles, blue
                   direction boards; metric
  * Australia/NZ — MUTCD-like yellow diamonds, but Vienna-style circular
                   speed limits and green direction boards; metric

The matcher is deliberately dumb: observed sign features (shape, colours,
the VLM's own convention guess, road-marking colours) become a token set
that is overlapped with each convention's keyword table. A score is the
fraction of observed tokens a convention can explain, plus a flat boost
when the VLM names the convention outright.

Pure stdlib; the schema import inside sign_style_evidence() is lazy so
this module imports with zero package deps.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Knowledge table — one entry per sign convention
# --------------------------------------------------------------------------

_CONVENTIONS: list[dict] = [
    {
        "key": "uk",
        "name": "UK-style",
        "countries": {"GB", "IE"},
        "signs": {
            "warning": "triangular white red border",
            "speed_limit": "circular white red border black number mph",
            "motorway": "blue background white text",
            "direction": "green primary route white yellow text",
            "stop": "octagonal red white text",
            "markings": "double yellow line zigzag keep left bollard",
        },
    },
    {
        "key": "mutcd",
        "name": "US MUTCD",
        "countries": {"US", "CA"},
        "signs": {
            "warning": "diamond yellow black symbol",
            "speed_limit": "rectangular white black number speed limit text mph",
            "motorway": "green highway freeway guide white text",
            "direction": "green white text",
            "stop": "octagonal red white text",
            "markings": "yellow centerline center line white edge",
        },
    },
    {
        "key": "vienna",
        "name": "Vienna Convention",
        # Continental Europe incl. EFTA, Balkans, Türkiye, UA/BY/RU/MD.
        # GB/IE have their own entry above; US/CA, JP, AU/NZ below.
        "countries": {
            "DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "PL", "SE",
            "DK", "FI", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI",
            "EE", "LV", "LT", "LU", "MT", "CY",
            "NO", "CH", "IS", "LI", "UA", "TR", "RS", "BA", "ME", "MK",
            "AL", "MD", "BY", "RU",
        },
        "signs": {
            "warning": "triangular white red border black symbol",
            "speed_limit": "circular white red border black number kmh km",
            "motorway": "blue green background white text autobahn autostrada",
            "direction": "blue background white text",
            "stop": "octagonal red white text",
            "markings": "white centerline center line",
        },
    },
    {
        "key": "japan",
        "name": "Japan",
        "countries": {"JP"},
        "signs": {
            "warning": "diamond yellow black symbol",
            "speed_limit": "circular white red border black number kmh km",
            "motorway": "green expressway white text",
            "direction": "blue background white text",
            # Japan's stop sign is an inverted red triangle ("止まれ") —
            # "triangle" kept as a distinct token so generic "triangular"
            # warning-sign queries don't tie Japan to UK/Vienna scores.
            "stop": "inverted triangle red white text",
            "markings": "white yellow centerline center line",
        },
    },
    {
        "key": "australia",
        "name": "Australia/NZ",
        # MUTCD-like yellow diamonds but Vienna-style speed circles; metric.
        "countries": {"AU", "NZ"},
        "signs": {
            "warning": "diamond yellow black symbol",
            "speed_limit": "circular white red border black number kmh km",
            "motorway": "green highway white text",
            "direction": "green white text",
            "stop": "octagonal red white text",
            "markings": "white yellow line",
        },
    },
]

# Free-text convention guesses ("mutcd-style", "European", "JDM") -> keys.
_GUESS_ALIASES = {
    "uk": "uk", "gb": "uk", "british": "uk", "britain": "uk",
    "irish": "uk", "ireland": "uk",
    "us": "mutcd", "usa": "mutcd", "mutcd": "mutcd", "american": "mutcd",
    "america": "mutcd", "canada": "mutcd", "canadian": "mutcd",
    "vienna": "vienna", "eu": "vienna", "european": "vienna",
    "europe": "vienna", "continental": "vienna",
    "japan": "japan", "jp": "japan", "japanese": "japan", "jdm": "japan",
    "australia": "australia", "aus": "australia", "australian": "australia",
    "aussie": "australia", "nz": "australia", "zealand": "australia",
    "kiwi": "australia",
}

# Score bump when the VLM names a convention outright — weaker than a
# contradictory threshold on its own, stronger than one generic token.
_GUESS_BOOST = 0.5
# Min top score before "the clip's signs look like convention X" can
# contradict a claim whose country isn't in X.
_CONTRADICT_MIN = 0.6
# Sign style is a soft cue (borrowed designs, border regions) — cap it.
_MAX_CONF = 0.5

_WORD = re.compile(r"[a-z]+")
_SUFFIXES = ("ing", "ed", "es", "s")

# Filler tokens that carry no convention signal — stripped from queries.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "on", "in", "at", "with", "for",
    "to", "seen", "visible", "sign", "signs", "shaped", "shape", "road",
    "roads", "traffic", "street", "style", "styled", "like", "typical",
    "common", "plus", "also", "showing", "shows",
})


def _stem(tok: str) -> str:
    """Collapse light inflections to a fixpoint so 'red-bordered'/'border'
    and 'markings'/'marking' line up ('markings' -> 'marking' -> 'mark')."""
    prev = None
    while prev != tok:
        prev = tok
        for suf in _SUFFIXES:
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                tok = tok[: -len(suf)]
                break
    return tok


def _tokens(text, drop_stop: bool = True) -> set[str]:
    """Lowercase alpha tokens from free text ('red-bordered' -> {red, border})."""
    out: set[str] = set()
    for t in _WORD.findall(str(text or "").lower()):
        if drop_stop and t in _STOPWORDS:
            continue
        s = _stem(t)
        if drop_stop and s in _STOPWORDS:
            continue
        out.add(s)
    return out


def _keywords(conv: dict) -> set[str]:
    """Stemmed token set pooled across all of a convention's sign types."""
    out: set[str] = set()
    for desc in (conv.get("signs") or {}).values():
        out |= _tokens(desc, drop_stop=False)
    return out


_KEYWORDS = {c["key"]: _keywords(c) for c in _CONVENTIONS}
_KEYS = frozenset(_KEYWORDS)


def _query_tokens(style: dict, marking_colors) -> set[str]:
    """Observed-feature token set: every string in the sign-style dict
    (except convention_guess) plus road-marking colour descriptions."""
    q: set[str] = set()

    def add(v) -> None:
        if isinstance(v, dict):
            for vv in v.values():
                add(vv)
        elif isinstance(v, (list, tuple, set, frozenset)):
            for vv in v:
                add(vv)
        elif isinstance(v, str):
            q.update(_tokens(v))

    for k, v in style.items():
        if str(k).strip().lower() in ("convention_guess", "convention"):
            continue
        add(v)
    for m in marking_colors or []:
        q.update(_tokens(m))
    return q


def _guess_keys(style: dict) -> set[str]:
    """Map a convention_guess string to convention keys via aliases."""
    out: set[str] = set()
    for k, v in style.items():
        if str(k).strip().lower() not in ("convention_guess", "convention"):
            continue
        for t in _WORD.findall(str(v or "").lower()):
            key = _GUESS_ALIASES.get(t, t if t in _KEYS else None)
            if key:
                out.add(key)
    return out


def sign_style_score(
    sign_style: dict,
    marking_colors: list[str] | None = None,
) -> dict[str, float]:
    """Score each sign convention 0..1 against observed sign features.

    `sign_style` is a scene-features dict like
    {"shape": "diamond", "colors": ["yellow", "black"],
     "convention_guess": "mutcd"}; `marking_colors` is an optional list of
    road-marking colour descriptions ("double yellow lines").

    Score = fraction of observed tokens the convention's keyword table
    explains, plus _GUESS_BOOST when the observation names the convention
    itself. e.g. "triangular"+"red border" tops uk/vienna;
    "diamond"+"yellow" tops mutcd/japan/australia.
    """
    style = sign_style if isinstance(sign_style, dict) else {}
    q = _query_tokens(style, marking_colors)
    guessed = _guess_keys(style)

    scores: dict[str, float] = {}
    for c in _CONVENTIONS:
        key = c["key"]
        hits = len(q & _KEYWORDS[key])
        s = hits / len(q) if q else 0.0
        if key in guessed:
            s += _GUESS_BOOST
        scores[key] = round(min(1.0, s), 3)
    return scores


def sign_style_evidence(
    scores: dict[str, float],
    claimed_cc: str | None,
) -> "Evidence":
    """Fold convention scores into one Evidence card vs the claimed country.

    `claimed_cc` is an ISO-3166 alpha-2 code ("GB", "US"). The top-scoring
    convention governs: if its member countries contain claimed_cc ->
    SUPPORTS; if the top clearly excludes it (score >= 0.6 and no member
    match) -> CONTRADICTS; otherwise INCONCLUSIVE. All scores are capped
    at 0.5 — sign design is a soft cue.
    """
    from .schema import ClaimVerdict, Evidence  # lazy: zero-dep module import

    raw = scores if isinstance(scores, dict) else {}
    sc: dict[str, float] = {}
    for c in _CONVENTIONS:
        try:
            sc[c["key"]] = max(0.0, min(1.0, float(raw.get(c["key"], 0.0))))
        except (TypeError, ValueError):
            sc[c["key"]] = 0.0

    top = max(sc.values())
    top_keys = [k for k, v in sc.items() if v >= top - 1e-9]
    top_set = [c for c in _CONVENTIONS if c["key"] in top_keys]
    top_countries: set[str] = set()
    for c in top_set:
        top_countries |= c["countries"]
    top_names = "/".join(c["name"] for c in top_set)

    cc = str(claimed_cc).strip().upper() if claimed_cc else ""
    cc = cc or None
    member = bool(cc) and cc in top_countries
    data = {
        "scores": sc,
        "top_conventions": top_keys,
        "top_score": round(top, 3),
        "claimed_cc": cc,
        "member_match": member,
    }

    if top <= 0.0:
        return Evidence(
            type="sign_style",
            summary="observed signs match no known sign convention",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.1,
            data=data,
        )
    if len(top_keys) == len(_CONVENTIONS):
        return Evidence(
            type="sign_style",
            summary=(
                f"sign features fit every convention equally "
                f"(score {top:.2f}) — non-discriminative"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )
    if cc is None:
        return Evidence(
            type="sign_style",
            summary=(
                f"signs look {top_names} (score {top:.2f}) but no claimed "
                "country code to compare against"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )
    if member:
        return Evidence(
            type="sign_style",
            summary=(
                f"signs match the {top_names} convention(s) — "
                f"consistent with claimed {cc}"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=min(0.45, _MAX_CONF),
            data=data,
        )
    if top >= _CONTRADICT_MIN:
        return Evidence(
            type="sign_style",
            summary=(
                f"signs match the {top_names} convention(s) "
                f"(score {top:.2f}) — {cc} is not a {top_names} country"
            ),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=min(0.45, _MAX_CONF),
            data=data,
        )
    return Evidence(
        type="sign_style",
        summary=(
            f"signs weakly match {top_names} (score {top:.2f}) — "
            f"too soft to confirm or refute {cc}"
        ),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.25,
        data=data,
    )


# --------------------------------------------------------------------------
# Self-test: python -m sundial.sign_style
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def _top(sc: dict) -> set[str]:
        m = max(sc.values())
        return {k for k, v in sc.items() if v >= m - 1e-9}

    fails = 0

    def check(name: str, ok: bool, extra: str = "") -> None:
        global fails
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {extra}")

    print("sign_style_score:")
    s1 = sign_style_score({"shape": "triangular", "colors": ["white", "red", "black"]})
    check("triangular red-bordered -> {uk, vienna} top",
          _top(s1) == {"uk", "vienna"}, str(s1))

    s2 = sign_style_score({"shape": "diamond", "colors": ["yellow", "black"]})
    check("yellow diamond -> {mutcd, japan, australia} top",
          _top(s2) == {"mutcd", "japan", "australia"}, str(s2))

    s3 = sign_style_score({"convention_guess": "mutcd"})
    check("bare convention_guess 'mutcd' -> {mutcd} top",
          _top(s3) == {"mutcd"}, str(s3))

    s4 = sign_style_score({})
    check("empty observation -> all zeros",
          all(v == 0.0 for v in s4.values()), str(s4))

    s5 = sign_style_score(
        {"shape": "circular", "colors": ["white", "red"]},
        marking_colors=["double yellow lines"],
    )
    check("red circle + double-yellow markings -> {uk} alone top",
          _top(s5) == {"uk"}, str(s5))

    s6 = sign_style_score(
        {"shape": "rectangular", "text": "SPEED LIMIT 65",
         "colors": ["white", "black"]})
    check("'SPEED LIMIT' black-on-white rectangle -> {mutcd} top",
          _top(s6) == {"mutcd"}, str(s6))

    print("sign_style_evidence:")
    try:
        e1 = sign_style_evidence(s1, "GB")
        check("triangular/red signs + claim GB -> SUPPORTS 0.45",
              e1.verdict.value == "supports" and abs(e1.confidence - 0.45) < 1e-9
              and e1.type == "sign_style",
              f"{e1.verdict.value} {e1.confidence}: {e1.summary}")

        e2 = sign_style_evidence(s2, "US")
        check("yellow diamond + claim US -> SUPPORTS",
              e2.verdict.value == "supports",
              f"{e2.verdict.value} {e2.confidence}: {e2.summary}")

        e3 = sign_style_evidence(s2, "DE")
        check("yellow diamond + claim DE -> CONTRADICTS 0.45",
              e3.verdict.value == "contradicts" and abs(e3.confidence - 0.45) < 1e-9,
              f"{e3.verdict.value} {e3.confidence}: {e3.summary}")

        e4 = sign_style_evidence(s1, "FR")
        check("triangular/red signs + claim FR -> SUPPORTS (vienna member)",
              e4.verdict.value == "supports",
              f"{e4.verdict.value} {e4.confidence}: {e4.summary}")

        e5 = sign_style_evidence(s4, "GB")
        check("zero scores -> INCONCLUSIVE",
              e5.verdict.value == "inconclusive",
              f"{e5.verdict.value} {e5.confidence}")

        e6 = sign_style_evidence(s2, None)
        check("no claimed cc -> INCONCLUSIVE",
              e6.verdict.value == "inconclusive",
              f"{e6.verdict.value} {e6.confidence}")
    except ImportError:
        print("  (run as `python -m sundial.sign_style` to exercise Evidence checks)")

    total_hint = "some checks FAILED" if fails else "all checks passed"
    print(f"self-test: {total_hint}")
    sys.exit(1 if fails else 0)
