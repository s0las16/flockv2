"""Country-candidate scorer — the heart of OSINT-style place matching.

Fuses the weak in-clip visual cues — driving side, sign language, plate
style, road markings, lane-line colour, sign convention and
vegetation/climate biome — into one weighted score per candidate country
code. Every candidate starts at 0.5 ("no opinion") and each observed cue
nudges the score up or down, so a candidate with no relevant evidence
sits at a neutral 0.5.

Weights are deliberately modest and the evidence card caps confidence at
0.65: OSINT cue-fusion corroborates a claim, it never convicts alone.

NOTE — DUPLICATED TABLES: _LEFT_TRAFFIC / _LANG_HINTS / _PLATE_HINTS /
_MARKING_HINTS below are copied verbatim from sundial/geo.py. Per design
constraints that module is neither imported nor modified, so any updates
to those tables must be kept in sync by hand.

Pure stdlib, no network. Stage API:

    ranked = score_candidates(features, candidate_ccs)
    ev     = match_evidence(ranked, claimed_cc)

`features` keys (all optional): driving_side, sign_language, plate_style,
road_markings, lane_markings_color, sign_convention, vegetation, climate.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Cue weights — deliberately modest; no single cue convicts.
# ---------------------------------------------------------------------------
_W_DRIVING = 0.25      # driving side vs left-traffic set
_W_LANG = 0.20         # sign language vs language hints
_W_PLATE = 0.15        # plate style vs plate hints
_W_MARKING = 0.10      # per road-marking token
_MARKING_CAP = 0.20    # total road-marking contribution clamp
_W_LANE_COLOR = 0.10   # lane-markings color vs centerline convention
_W_CONVENTION = 0.10   # sign-design convention vs country
_W_BIOME = 0.05        # per vegetation/climate keyword
_BIOME_CAP = 0.15      # total biome contribution clamp
_BASE = 0.5            # neutral prior — no opinion

# ---------------------------------------------------------------------------
# Country-knowledge tables COPIED FROM sundial/geo.py (see module docstring).
# ---------------------------------------------------------------------------

# Countries that keep left.
_LEFT_TRAFFIC = {
    "GB", "IE", "JP", "AU", "NZ", "IN", "ZA", "TH", "MY", "SG", "ID", "HK",
    "KE", "MT", "CY", "PK", "BD", "LK", "NP", "MU", "TT", "JM", "BB", "BS",
    "GY", "SR", "BN", "BT", "MO", "NA", "BW", "LS", "SZ", "MZ", "ZM", "MW",
    "TZ", "UG", "FJ", "PG",
}
_PLATE_HINTS = {
    "uk": {"GB"}, "gb": {"GB"},
    "us": {"US"}, "ca": {"CA"}, "au": {"AU"}, "jp": {"JP"}, "cn": {"CN"},
    "in": {"IN"}, "kr": {"KR"}, "ru": {"RU"}, "br": {"BR"}, "za": {"ZA"},
    # EU-style blue strip with circle of stars.
    "eu": {"DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "PL", "IE", "SE",
           "DK", "FI", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "EE",
           "LV", "LT", "LU", "MT", "CY"},
}
# Dominant language of public signage per country (rough).
_LANG_HINTS = {
    "en": {"GB", "IE", "US", "AU", "NZ", "CA", "ZA", "IN", "SG", "NG", "KE", "JM", "MT", "GY"},
    "fr": {"FR", "BE", "CH", "LU", "MC", "CA", "SN", "CI", "ML", "BF", "NE", "TG", "BJ", "CD", "CG", "MG"},
    "de": {"DE", "AT", "CH", "LU", "LI"},
    "es": {"ES", "MX", "AR", "CO", "CL", "PE", "VE", "EC", "GT", "CU", "BO", "DO", "HN", "PY", "SV", "NI", "CR", "PA", "UY", "GQ"},
    "pt": {"PT", "BR", "AO", "MZ", "CV", "GW", "ST", "TL"},
    "it": {"IT", "CH", "SM", "VA"},
    "nl": {"NL", "BE", "SR", "AW", "CW"},
    "ja": {"JP"}, "ko": {"KR"}, "zh": {"CN", "TW", "HK", "MO", "SG"},
    "ar": {"SA", "AE", "EG", "QA", "KW", "BH", "OM", "JO", "LB", "IQ", "SY", "MA", "DZ", "TN", "LY", "SD", "YE", "PS", "MR", "DJ"},
    "ru": {"RU", "BY", "KZ", "KG"},
    "hi": {"IN"}, "th": {"TH"}, "vi": {"VN"}, "id": {"ID"}, "ms": {"MY", "BN"},
    "tr": {"TR", "CY"}, "el": {"GR", "CY"}, "pl": {"PL"}, "uk": {"UA"},
    "sv": {"SE"}, "da": {"DK"}, "no": {"NO"}, "fi": {"FI"}, "cs": {"CZ"},
    "sk": {"SK"}, "hu": {"HU"}, "ro": {"RO"}, "bg": {"BG"}, "hr": {"HR"},
    "he": {"IL"}, "fa": {"IR"}, "ur": {"PK"}, "bn": {"BD", "IN"},
}
_MARKING_HINTS = {
    "double_yellow_lines": {"GB", "IE"},
    "double_red_lines": {"GB"},
    "yellow_centerline": {"US", "CA", "BR", "CN", "KR", "TW", "NO", "SE", "FI"},
    "yellow_edge_line": {"US", "CA", "GB", "IE"},
    "zigzag_lines": {"GB", "IE", "SG"},
    "keep_left_bollards": {"GB", "IE", "AU", "NZ", "JP"},
}

# ---------------------------------------------------------------------------
# Tables unique to this stage.
# ---------------------------------------------------------------------------

# Countries that paint their *centerlines* yellow (most of the world uses
# white for lane separation; yellow for the opposing-traffic divide is an
# Americas/Nordic/East-Asia tell).
_YELLOW_CENTERLINE = {"US", "CA", "BR", "CN", "KR", "TW", "NO", "SE", "FI"}

# Road-sign design conventions: which family each country's signage broadly
# follows (rough — borders blur and many countries mix styles).
_SIGN_CONVENTION_HINTS = {
    # UK-style: Transport/Motorway faces, mph speeds, red-bordered white
    # triangles; inherited across much of the Commonwealth.
    "uk-style": {
        "GB", "IE", "MT", "CY", "AU", "NZ", "IN", "ZA", "SG", "MY", "HK",
        "KE", "NG", "PK", "BD", "LK",
    },
    # US MUTCD and its near-copies (Canada, most of LatAm, parts of Asia,
    # AU/NZ hybrids).
    "mutcd-style": {
        "US", "CA", "MX", "BR", "AR", "CL", "CO", "PE", "VE", "PH", "AU",
        "NZ", "TH", "TW", "KR",
    },
    # Vienna Convention on Road Signs and Signals — continental Europe and
    # the former-Soviet sphere, plus assorted signatories elsewhere.
    "vienna-style": {
        "DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "CH", "LU", "LI",
        "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "RS", "BA",
        "ME", "MK", "AL", "SE", "DK", "FI", "NO", "IS", "EE", "LV", "LT",
        "RU", "UA", "BY", "TR", "MC", "SM", "VA", "KZ", "KG", "MD", "AM",
        "AZ", "GE", "UZ", "TM", "TJ", "CN", "VN", "ID", "IR", "IQ", "SA",
        "EG", "MA", "TN", "DZ", "KR",
    },
}

# --- biome keyword table (deliberately small and rough) ---------------------
# Each keyword maps to countries where that vegetation/climate cue is
# *typical* — a hit is weak evidence, not a fingerprint.

_TROPICAL = {
    "TH", "MY", "SG", "ID", "BR", "MX", "IN", "LK", "PH", "VN", "KH", "LA",
    "MM", "BD", "NP", "NG", "GH", "CI", "CM", "KE", "TZ", "UG", "JM", "TT",
    "BB", "CO", "VE", "EC", "PE", "CR", "PA", "CU", "DO", "HN", "NI", "GT",
    "SV", "GY", "SR", "PG", "FJ",
}
_TEMPERATE = {
    "GB", "IE", "US", "CA", "DE", "FR", "NL", "BE", "AT", "CH", "LU", "PL",
    "CZ", "SK", "HU", "RO", "BG", "HR", "SI", "IT", "ES", "PT", "DK", "SE",
    "NO", "EE", "LV", "LT", "UA", "RU", "BY", "TR", "JP", "KR", "CN", "NZ",
    "AR", "CL", "ZA", "GE", "AU",
}
_COLD = {
    "NO", "SE", "FI", "IS", "CA", "RU", "CH", "AT", "DE", "JP", "NZ", "US",
    "MN", "KZ", "KG", "PL", "UA", "BY", "EE", "LV", "LT", "CN", "KR", "KP",
    "FR", "IT", "RO", "SK", "AR", "CL", "GB", "GL",
}
_DESERT = {
    "MX", "US", "MA", "DZ", "EG", "TN", "LY", "SA", "AE", "QA", "OM", "YE",
    "JO", "IL", "IQ", "IR", "SY", "KW", "BH", "IN", "PK", "AF", "AU", "ZA",
    "NA", "BW", "CL", "AR", "PE", "MN", "ES", "TD", "NE", "SD", "MR", "ML",
    "DJ", "KZ", "TM", "UZ",
}
_MED = {
    "ES", "IT", "GR", "PT", "TR", "MA", "TN", "DZ", "IL", "CY", "HR", "ME",
    "AL", "FR", "JO", "PS", "LB", "SY", "MT", "US", "AU", "ZA", "CL",
}
_BOREAL = {
    "SE", "NO", "FI", "RU", "CA", "EE", "LV", "LT", "BY", "PL", "DE", "UA",
    "US", "IS", "JP", "CN", "AT", "CH", "CZ", "SK", "MN", "KZ", "GB", "IE",
}

_BIOME_HINTS: dict[str, set[str]] = {
    # tropical flora & climate
    "palm": _TROPICAL | {"US", "AU", "ES", "PT", "IL", "ZA", "MA", "EG"},
    "coconut": _TROPICAL,
    "banana": _TROPICAL | {"ES", "IL", "TR"},
    "tropical": _TROPICAL,
    "monsoon": {"IN", "BD", "LK", "NP", "MM", "TH", "KH", "LA", "VN", "PH", "PK"},
    "bamboo": {"CN", "JP", "VN", "TH", "IN", "ID", "PH", "MY", "LK", "BD",
               "NP", "TW", "KR", "LA", "KH", "MM", "BR"},
    # distinctive flora
    "eucalyptus": {"AU", "NZ", "PT", "ES", "ZA", "CL", "BR", "KE", "IN",
                   "US", "MA", "FR"},
    "acacia": {"KE", "TZ", "ZA", "BW", "NA", "ZW", "ZM", "MW", "MZ", "UG",
               "AU", "SA", "IL", "IN", "ET", "NG", "GH", "SN", "SD", "SO"},
    "baobab": {"MG", "ZA", "BW", "NA", "ZW", "MZ", "AU", "SN", "ML", "TZ", "KE"},
    "saguaro": {"US", "MX"},
    "cactus": _DESERT,
    "succulent": _DESERT | {"ZA", "NA"},
    "olive": _MED,
    "cypress": _MED,
    "vineyard": {"FR", "IT", "ES", "PT", "DE", "AU", "NZ", "ZA", "AR", "CL",
                 "US", "GR", "TR", "HU", "GE", "AT", "RO", "BG", "HR"},
    "hedgerow": {"GB", "IE", "FR", "DE", "NL", "BE"},
    "fern": {"NZ", "GB", "IE", "JP", "AU", "US", "CA", "DE", "NO", "SE",
             "FI", "CL", "FR"},
    "moss": {"IS", "GB", "IE", "NO", "SE", "FI", "NZ", "CA", "JP", "DE", "US"},
    # temperate broadleaf
    "deciduous": _TEMPERATE,
    "oak": _TEMPERATE,
    "beech": _TEMPERATE,
    "maple": {"US", "CA", "JP", "CN", "KR", "DE", "PL", "SE", "NO", "FI",
              "UA", "RU", "FR", "BE", "NL", "GB"},
    "autumn": _TEMPERATE,
    "temperate": _TEMPERATE,
    "prairie": {"US", "CA", "KZ", "RU", "MN", "UA", "AR", "HU", "RO"},
    # cold / boreal
    "snow": _COLD,
    "ice": _COLD,
    "frozen": _COLD,
    "glacier": {"IS", "NO", "NZ", "CH", "AT", "AR", "CL", "CA", "US", "NP",
                "PK", "IN", "FR", "IT", "GL"},
    "tundra": {"IS", "NO", "SE", "FI", "RU", "CA", "MN", "GL"},
    "taiga": _BOREAL,
    "conifer": _BOREAL,
    "spruce": _BOREAL,
    "birch": _BOREAL,
    # dry & open country
    "desert": _DESERT,
    "arid": _DESERT,
    "savanna": {"KE", "TZ", "ZA", "BW", "NA", "ZW", "ZM", "MW", "MZ", "UG",
                "NG", "GH", "SN", "ML", "BF", "NE", "TD", "ET", "AU", "BR",
                "AR", "IN", "LK"},
    "steppe": {"KZ", "MN", "RU", "UA", "KG", "TM", "UZ", "HU", "RO", "AR",
               "US", "CA", "TR", "IR", "AF"},
    "fjord": {"NO", "NZ", "CL", "IS", "CA", "GL"},
    # cultivation patterns
    "rice": {"VN", "TH", "ID", "PH", "IN", "BD", "LK", "CN", "JP", "KR",
             "MY", "KH", "LA", "MM", "TW", "NP", "IT", "ES", "US", "BR"},
    "paddy": {"VN", "TH", "ID", "PH", "IN", "BD", "LK", "CN", "JP", "KR",
              "MY", "KH", "LA", "MM", "TW", "NP"},
    "tea": {"IN", "LK", "CN", "JP", "KE", "TR", "VN", "ID", "BD", "TW",
            "NP", "MY", "UG", "RW"},
    "mediterranean": _MED,
}

# Common non-ISO spellings of 2-letter codes -> ISO.
_CC_ALIASES = {"UK": "GB", "EL": "GR", "USA": "US", "UAE": "AE"}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _tokens(value) -> list[str]:
    """Coerce a feature value (str | iterable | None) into lowercase strings."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    out: list[str] = []
    try:
        it = iter(items)
    except TypeError:
        items = [items]
        it = iter(items)
    for item in it:
        t = str(item).strip().lower()
        if t:
            out.append(t)
    return out


def _first(value) -> str:
    toks = _tokens(value)
    return toks[0] if toks else ""


def _norm_key(s: str) -> str:
    """Normalize free text to a table key ('Zig-zag lines' -> 'zigzag_lines')."""
    return re.sub(r"[\s\-]+", "_", s.strip().lower()).strip("_")


def _norm_cc(value) -> str:
    """Uppercase a country code and map common aliases ('UK' -> 'GB')."""
    cc = (value or "").strip().upper()
    return _CC_ALIASES.get(cc, cc)


def _biome_matches(cue: str):
    """Yield (keyword, cc-set) for each biome keyword found in the cue text."""
    for kw, ccset in _BIOME_HINTS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", cue):
            yield kw, ccset


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_candidates(features: dict, candidate_ccs: list[str]) -> list[dict]:
    """Score each candidate country code against the observed feature cues.

    Returns a list sorted best-first:
        [{"cc": "GB", "score": 0..1, "hits": [...], "misses": [...]}, ...]

    `score` is clamp(0.5 + sum(cue deltas), 0, 1) — candidates with no
    relevant evidence stay at a neutral 0.5. `hits`/`misses` record which
    cues fired for/against the candidate, for explainability.
    """
    feats = features or {}

    # dedupe + normalize candidate codes (order-preserving)
    ccs: list[str] = []
    seen: set[str] = set()
    for raw in candidate_ccs or []:
        cc = _norm_cc(str(raw))
        if cc and cc not in seen:
            seen.add(cc)
            ccs.append(cc)

    # --- extract observations once --------------------------------------
    side = _first(feats.get("driving_side"))
    lang = _first(feats.get("sign_language"))
    plate = _first(feats.get("plate_style"))
    markings = [_norm_key(t) for t in _tokens(feats.get("road_markings"))]
    lane_color = _first(feats.get("lane_markings_color"))
    convention = re.sub(
        r"[\s_]+", "-",
        _first(feats.get("sign_convention") or feats.get("sign_style")),
    )
    biome_cues = _tokens(feats.get("vegetation")) + _tokens(feats.get("climate"))
    indicator_ccs = {
        _norm_cc(str(t)) for t in _tokens(feats.get("country_indicators"))
    } - {None}

    scored: list[dict] = []
    for cc in ccs:
        delta = 0.0
        hits: list[str] = []
        misses: list[str] = []

        # --- driving side (±0.25) --------------------------------------
        if side in ("left", "right"):
            expected_left = cc in _LEFT_TRAFFIC
            if (side == "left") == expected_left:
                delta += _W_DRIVING
                hits.append(f"driving_side '{side}' — consistent with {cc}")
            else:
                delta -= _W_DRIVING
                exp = "left" if expected_left else "right"
                misses.append(
                    f"driving_side '{side}' — INCONSISTENT with {cc} "
                    f"(drives {exp})"
                )

        # --- signage language (±0.2) ------------------------------------
        if lang in _LANG_HINTS:
            if cc in _LANG_HINTS[lang]:
                delta += _W_LANG
                hits.append(f"sign_language '{lang}' — consistent with {cc}")
            else:
                delta -= _W_LANG
                misses.append(
                    f"sign_language '{lang}' — INCONSISTENT with {cc}"
                )

        # --- plate style (±0.15) ----------------------------------------
        # "other" isn't in the hint table — an uninformative observation,
        # so it fires neither hit nor miss.
        if plate in _PLATE_HINTS:
            if cc in _PLATE_HINTS[plate]:
                delta += _W_PLATE
                hits.append(f"plate_style '{plate}' — consistent with {cc}")
            else:
                delta -= _W_PLATE
                misses.append(
                    f"plate_style '{plate}' — INCONSISTENT with {cc}"
                )

        # --- road markings (±0.1 each, cap 0.2) --------------------------
        mark_delta = 0.0
        for tok in markings:
            if tok not in _MARKING_HINTS:
                continue
            if cc in _MARKING_HINTS[tok]:
                mark_delta += _W_MARKING
                hits.append(f"road_markings '{tok}' — consistent with {cc}")
            else:
                mark_delta -= _W_MARKING
                misses.append(
                    f"road_markings '{tok}' — INCONSISTENT with {cc}"
                )
        delta += _clamp(mark_delta, -_MARKING_CAP, _MARKING_CAP)

        # --- lane-marking color (±0.1) -----------------------------------
        if lane_color in ("white", "yellow"):
            expects_yellow = cc in _YELLOW_CENTERLINE
            if (lane_color == "yellow") == expects_yellow:
                delta += _W_LANE_COLOR
                conv = "yellow" if expects_yellow else "white"
                hits.append(
                    f"lane_markings_color '{lane_color}' — consistent with "
                    f"{cc} ({conv}-centerline convention)"
                )
            else:
                delta -= _W_LANE_COLOR
                exp = "yellow" if expects_yellow else "white"
                misses.append(
                    f"lane_markings_color '{lane_color}' — INCONSISTENT "
                    f"with {cc} (expects {exp} centerlines)"
                )

        # --- sign convention (±0.1) --------------------------------------
        if convention in _SIGN_CONVENTION_HINTS:
            if cc in _SIGN_CONVENTION_HINTS[convention]:
                delta += _W_CONVENTION
                hits.append(
                    f"sign_convention '{convention}' — consistent with {cc}"
                )
            else:
                delta -= _W_CONVENTION
                misses.append(
                    f"sign_convention '{convention}' — INCONSISTENT with {cc}"
                )

        # --- vegetation / climate biome cues (±0.05 each, cap 0.15) ------
        biome_delta = 0.0
        for cue in biome_cues:
            for kw, ccset in _biome_matches(cue):
                if cc in ccset:
                    biome_delta += _W_BIOME
                    hits.append(
                        f"biome '{cue}' (keyword '{kw}') — consistent "
                        f"with {cc}"
                    )
                else:
                    biome_delta -= _W_BIOME
                    misses.append(
                        f"biome '{cue}' (keyword '{kw}') — INCONSISTENT "
                        f"with {cc}"
                    )
        delta += _clamp(biome_delta, -_BIOME_CAP, _BIOME_CAP)

        # --- country-indicative tokens (+0.15, positive-only) -----------
        # Dial codes / ccTLDs / currency / legal suffixes mined from
        # consensus text — naming a country is strong evidence FOR it,
        # but never evidence against everyone else (absence is silent).
        for ind in indicator_ccs:
            if cc == ind:
                delta += 0.15
                hits.append(f"country indicator '{ind}' — consistent with {cc}")
                break

        raw = _BASE + delta
        scored.append({
            "cc": cc,
            "score": round(_clamp(raw), 3),
            "hits": hits,
            "misses": misses,
            "_raw": raw,  # internal: unclamped tiebreak, stripped below
        })

    # best-first; raw score breaks clamped ties, then hit count, then code
    scored.sort(key=lambda c: (-c["_raw"], -len(c["hits"]), c["cc"]))
    for c in scored:
        del c["_raw"]
    return scored


# ---------------------------------------------------------------------------
# Evidence card
# ---------------------------------------------------------------------------

def match_evidence(ranked: list[dict], claimed_cc: str | None) -> "Evidence":
    """Fold a scored candidate ranking into a claim-level Evidence card.

    If claimed_cc is among the candidates, its score decides the verdict:
    >= 0.62 SUPPORTS (conf ~0.55), <= 0.38 CONTRADICTS (~0.55), middle
    INCONCLUSIVE (0.35). If claimed_cc is absent/None, report the top-3
    candidates as INCONCLUSIVE (0.3) with the ranking in data.

    Confidence never exceeds 0.65 — this stage corroborates, it does not
    convict alone.
    """
    from .schema import ClaimVerdict, Evidence  # lazy — keep import light

    cands = [c for c in (ranked or []) if isinstance(c, dict) and c.get("cc")]
    cc = _norm_cc(claimed_cc)
    top3 = [
        {"cc": str(c.get("cc")), "score": float(c.get("score", 0.5))}
        for c in cands[:3]
    ]
    data = {
        "claimed_cc": cc or None,
        "n_candidates": len(cands),
        "top3": top3,
        "ranking": [
            {
                "cc": str(c.get("cc")),
                "score": float(c.get("score", 0.5)),
                "hits": list(c.get("hits") or []),
                "misses": list(c.get("misses") or []),
            }
            for c in cands[:10]
        ],
    }

    if not cands:
        return Evidence(
            type="geo_match",
            summary="no candidate countries were scored — nothing to match",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data=data,
        )

    top_desc = ", ".join(f"{t['cc']} {t['score']:.2f}" for t in top3)
    idx = next(
        (i for i, c in enumerate(cands)
         if _norm_cc(str(c.get("cc") or "")) == cc),
        None,
    )
    if cc is None or idx is None:
        why = (
            "no claimed country supplied"
            if cc is None
            else f"claimed country {cc!r} was not among the scored candidates"
        )
        return Evidence(
            type="geo_match",
            summary=f"{why} — visual cues rank {top_desc} highest",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=min(0.65, 0.30),
            data=data,
        )

    entry = cands[idx]
    score = float(entry.get("score", 0.5))
    rank = idx + 1
    data["claimed"] = {
        "cc": str(entry.get("cc")),
        "rank": rank,
        "score": score,
        "hits": list(entry.get("hits") or []),
        "misses": list(entry.get("misses") or []),
    }

    if score >= 0.62:
        return Evidence(
            type="geo_match",
            summary=(
                f"visual cues corroborate claimed {cc} — ranked #{rank} "
                f"of {len(cands)} (score {score:.2f})"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=min(0.65, 0.55),
            data=data,
        )
    if score <= 0.38:
        top = cands[0]
        return Evidence(
            type="geo_match",
            summary=(
                f"visual cues argue against claimed {cc} — ranked #{rank} "
                f"of {len(cands)} (score {score:.2f}); scene looks more "
                f"like {top.get('cc')} ({float(top.get('score', 0.5)):.2f})"
            ),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=min(0.65, 0.55),
            data=data,
        )
    return Evidence(
        type="geo_match",
        summary=(
            f"claimed {cc} sits mid-pack — #{rank} of {len(cands)} "
            f"(score {score:.2f}); visual cues are ambiguous"
        ),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=min(0.65, 0.35),
        data=data,
    )


# ---------------------------------------------------------------------------
# Self-test: python -m sundial.match_score
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _DEMO_CCS = [
        "GB", "IE", "US", "CA", "AU", "NZ", "DE", "FR", "IT", "ES", "PT",
        "NL", "BE", "SE", "NO", "FI", "PL", "JP", "CN", "KR", "IN", "TH",
        "SG", "MY", "ID", "BR", "MX", "AR", "CL", "ZA", "KE", "NG", "EG",
        "MA", "TR", "RU", "UA", "GR", "AE", "SA", "PH", "VN", "PK", "BD",
        "LK", "MT", "CY", "IS", "CH", "AT", "CZ",
    ]
    uk_features = {
        "driving_side": "left",
        "sign_language": "en",
        "plate_style": "uk",
        "road_markings": ["double yellow lines", "zig-zag lines"],
        "lane_markings_color": "white",
        "sign_convention": "uk-style",
        "vegetation": ["bare deciduous trees", "hedgerow"],
        "climate": "temperate drizzle",
    }
    us_features = {
        "driving_side": "right",
        "sign_language": "en",
        "plate_style": "us",
        "road_markings": ["yellow centerline"],
        "lane_markings_color": "yellow",
        "sign_convention": "mutcd-style",
        "vegetation": ["saguaro cactus", "prairie grass"],
        "climate": "arid",
    }

    fails = 0

    def check(name: str, got, want) -> None:
        global fails
        ok = got == want
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: "
              f"got {got!r} (want {want!r})")

    ranked_uk = score_candidates(uk_features, _DEMO_CCS)
    print("UK-flavoured features — top 5:")
    for c in ranked_uk[:5]:
        print(f"    {c['cc']} {c['score']:.2f} "
              f"hits={len(c['hits'])} misses={len(c['misses'])}")
    check("GB ranks top for UK features", ranked_uk[0]["cc"], "GB")

    ranked_us = score_candidates(us_features, _DEMO_CCS)
    print("US-flavoured features — top 5:")
    for c in ranked_us[:5]:
        print(f"    {c['cc']} {c['score']:.2f} "
              f"hits={len(c['hits'])} misses={len(c['misses'])}")
    check("US ranks top for US features", ranked_us[0]["cc"], "US")

    check(
        "empty features keep everyone neutral",
        {c["score"] for c in score_candidates({}, _DEMO_CCS)},
        {0.5},
    )
    check(
        "scores clamped to 0..1",
        all(0.0 <= c["score"] <= 1.0 for c in ranked_uk + ranked_us),
        True,
    )
    print("GB hits for UK features:")
    for h in ranked_uk[0]["hits"]:
        print(f"    + {h}")

    try:
        from .schema import ClaimVerdict  # needs `python -m` package context
    except ImportError:
        print("(Evidence checks skipped — run as `python -m sundial.match_score`)")
    else:
        check(
            "claimed GB on UK features -> SUPPORTS",
            match_evidence(ranked_uk, "GB").verdict,
            ClaimVerdict.SUPPORTS,
        )
        check(
            "claimed US on UK features -> CONTRADICTS",
            match_evidence(ranked_uk, "US").verdict,
            ClaimVerdict.CONTRADICTS,
        )
        check(
            "no claim -> INCONCLUSIVE",
            match_evidence(ranked_uk, None).verdict,
            ClaimVerdict.INCONCLUSIVE,
        )
        short = score_candidates(uk_features, ["GB", "IE", "US"])
        ev = match_evidence(short, "JP")
        check("claimed JP outside candidates -> INCONCLUSIVE",
              ev.verdict, ClaimVerdict.INCONCLUSIVE)
        check("absent-claim confidence is 0.3", ev.confidence, 0.30)
        check("confidence never exceeds 0.65",
              max(match_evidence(ranked_uk, "GB").confidence,
                  match_evidence(ranked_uk, "US").confidence) <= 0.65,
              True)

    print(f"self-check: {'FAILED ' + str(fails) if fails else 'all passed'}")
    if fails:
        sys.exit(1)
