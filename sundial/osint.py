"""OSINT fusion stage — fold the geo-matching signals into ONE card.

Sundial's OSINT stages each emit fragments of "where was this filmed":
a merged per-frame scene-feature vector (osint_features.feature_vector),
cheap text observations (driving_side / sign_language / plate_style /
road_markings), geocode hits, quarantined VLM geoguesses, sign-style
conventions and vegetation biomes. This module is the fusion layer that
weighs them together and answers one question for the adjudicator: do
the visible geo signals support the claimed country?

The scoring siblings this stage consumes are built in parallel, so every
sub-step is a lazy import inside try/except — a missing or crashed signal
is skipped and the card still forms:

  * ``match_score.score_candidates(fv, candidates)`` — per-cc ranking
  * ``geoguess.aggregate_guesses(guesses)``          — consensus vote
  * ``geoguess.resolve_cc(name)``                    — country -> ISO cc
  * ``sign_style.sign_style_score(feats, colors)``   — top convention
  * ``veg_geo.veg_biomes(vegetation)``               — biome list

Adjustments applied on top of match_score's ranking:
  +0.05 to every member country of the top sign-style convention;
  -0.10 to every candidate when detected biomes conflict with the
  anchor latitude band (e.g. boreal vegetation under a tropical claim).

Verdict on ``claimed_cc``: candidate score >= 0.62 -> SUPPORTS,
<= 0.38 -> CONTRADICTS (confidence min(0.5 + 0.05*hits, 0.65) for both),
otherwise INCONCLUSIVE at 0.3. When no ranked list could be produced at
all there is nothing to adjudicate against — INCONCLUSIVE at 0.2 rather
than a hollow contradiction. With no claimed_cc the card reports the
top-3 candidates as INCONCLUSIVE (0.3).

``claim_time_iso`` is accepted for forward compatibility (cross-checks
against solar/lunar windows) and currently unused.

Stdlib only at import time; schema types are lazy.
"""
from __future__ import annotations

import importlib
import re

# --------------------------------------------------------------------------
# World-knowledge tables
# --------------------------------------------------------------------------
# Countries that keep left — mirrors geo.py's _LEFT_TRAFFIC.
_LEFT_TRAFFIC = frozenset({
    "GB", "IE", "JP", "AU", "NZ", "IN", "ZA", "TH", "MY", "SG", "ID", "HK",
    "KE", "MT", "CY", "PK", "BD", "LK", "NP", "MU", "TT", "JM", "BB", "BS",
    "GY", "SR", "BN", "BT", "MO", "NA", "BW", "LS", "SZ", "MZ", "ZM", "MW",
    "TZ", "UG", "FJ", "PG",
})

# Compact "drives right" candidate set — the true mirror of _LEFT_TRAFFIC
# is ~140 countries, so only the countries that actually come up as geo
# candidates are listed.
_RIGHT_TRAFFIC = (
    "US", "CA", "MX", "BR", "AR", "CL", "CO", "PE", "VE", "EC", "BO", "PY",
    "UY", "FR", "DE", "ES", "PT", "IT", "NL", "BE", "LU", "CH", "AT", "PL",
    "CZ", "SK", "HU", "RO", "BG", "HR", "SI", "RS", "GR", "TR", "UA", "RU",
    "SE", "NO", "DK", "FI", "IS", "EE", "LV", "LT", "CN", "KR", "TW", "VN",
    "PH", "MN", "KZ", "SA", "AE", "EG", "MA", "DZ", "TN", "NG", "GH", "ET",
    "CU", "DO", "CR", "PA",
)

_EU_CCS = frozenset({
    "DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "PL", "IE", "SE", "DK",
    "FI", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "EE", "LV", "LT",
    "LU", "MT", "CY",
})

# plate_style observation -> implied countries. The base values (eu|uk|us)
# come from prompts.TEXT_PROMPT; the rest tolerate synonyms seen in the wild.
_PLATE_HINTS = {
    "uk": {"GB"}, "gb": {"GB"}, "uk_white_yellow": {"GB"},
    "us": {"US", "CA"}, "usa": {"US"}, "ca": {"CA"}, "au": {"AU"},
    "jp": {"JP"}, "cn": {"CN"}, "in": {"IN"}, "kr": {"KR"}, "ru": {"RU"},
    "br": {"BR"}, "za": {"ZA"},
    "eu": set(_EU_CCS), "europe": set(_EU_CCS), "eu_blue_band": set(_EU_CCS),
    # Yellow rear plate is a strong UK/NL tell (geo.py _YELLOW_REAR_PLATE).
    "yellow_rear": {"GB", "NL"}, "yellow_rear_plate": {"GB", "NL"},
    "yellow_rear_plates": {"GB", "NL"},
}

# Dominant language of public signage per country — mirrors geo.py's
# _LANG_HINTS (keyed by ISO 639-1).
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
# English names for the codes above — languages_scripts can carry either.
_LANG_NAME_CODES = {
    "english": "en", "french": "fr", "german": "de", "spanish": "es",
    "portuguese": "pt", "italian": "it", "dutch": "nl", "japanese": "ja",
    "korean": "ko", "chinese": "zh", "mandarin": "zh", "arabic": "ar",
    "russian": "ru", "hindi": "hi", "thai": "th", "vietnamese": "vi",
    "indonesian": "id", "malay": "ms", "turkish": "tr", "greek": "el",
    "polish": "pl", "ukrainian": "uk", "swedish": "sv", "danish": "da",
    "norwegian": "no", "finnish": "fi", "czech": "cs", "slovak": "sk",
    "hungarian": "hu", "romanian": "ro", "bulgarian": "bg",
    "croatian": "hr", "hebrew": "he", "farsi": "fa", "persian": "fa",
    "urdu": "ur", "bengali": "bn",
}
# Script labels (langdetect.py / languages_scripts) -> regions.
_SCRIPT_HINTS = {
    "cyrillic": {"RU", "UA", "BY", "BG", "KZ", "KG", "RS", "BA", "ME", "MN", "MK"},
    "arabic": {"SA", "AE", "EG", "QA", "KW", "BH", "OM", "JO", "LB", "IQ",
               "SY", "MA", "DZ", "TN", "LY", "SD", "YE", "IR", "AF", "PK", "PS"},
    "cjk": {"JP", "CN", "TW", "HK", "MO"}, "hangul": {"KR", "KP"},
    "thai": {"TH"}, "devanagari": {"IN", "NP"}, "greek": {"GR", "CY"},
    "hebrew": {"IL"}, "bengali": {"BD", "IN"}, "tamil": {"IN", "LK", "SG"},
    "georgian": {"GE"}, "armenian": {"AM"},
}

# Sign-convention membership — fallback for the +0.05 bonus on the top
# convention's member countries (and for seeding candidates). Keys match
# sign_style._CONVENTIONS; when that module is importable its own
# "countries" member sets win (it is the convention authority).
_CONVENTION_CCS = {
    # Vienna Convention on Road Signs & Signals — most of Europe plus a
    # belt of signatories/influenced systems.
    "vienna": frozenset({
        "AL", "AD", "AM", "AT", "AZ", "BY", "BE", "BA", "BG", "HR", "CY",
        "CZ", "DK", "EE", "FI", "FR", "DE", "GE", "GR", "HU", "IT", "KZ",
        "KG", "LV", "LI", "LT", "LU", "MD", "MC", "ME", "NL", "MK", "NO",
        "PL", "PT", "RO", "RU", "SM", "RS", "SK", "SI", "ES", "SE", "CH",
        "TJ", "TR", "TM", "UA", "UZ", "VA",
    }),
    # US MUTCD and the systems heavily derived from it (much of the
    # Americas + Australasia + parts of SE Asia).
    "mutcd": frozenset({
        "US", "CA", "MX", "GT", "SV", "HN", "NI", "CR", "PA", "CU", "DO",
        "CO", "VE", "EC", "PE", "BO", "PY", "CL", "AR", "UY", "BR", "AU",
        "NZ", "PH", "ID", "MY", "LR",
    }),
    # UK-style signage (TSRGD lineage) — Britain + Ireland + anglosphere
    # and former-British-influence left-traffic countries.
    "uk": frozenset({
        "GB", "IE", "MT", "CY", "IN", "PK", "BD", "LK", "NP", "HK", "SG",
        "MY", "ZA", "NG", "KE", "TZ", "UG", "ZW", "BW", "NA", "MZ", "ZM",
        "MW", "JM", "TT", "BB", "BS", "MU", "SC", "GY", "FJ",
    }),
    "japan": frozenset({"JP"}),
    "australia": frozenset({"AU", "NZ"}),
    "chinese": frozenset({"CN"}),
}
# Convention-name aliases — VLMs call these "vienna-style", "mutcd-style",
# "uk-style", "european", ... An alias >= 3 chars may prefix-match a word
# in the convention name ("european" ~ "europe"); 2-char aliases match
# whole words only (so "ukraine" can't read as "uk"). First hit wins.
_CONVENTION_ALIASES = (
    (("vienna", "europe", "eu", "continental"), "vienna"),
    (("mutcd", "muctd", "america", "us"), "mutcd"),
    (("brit", "uk", "worboys", "tsrgd"), "uk"),
    (("japan", "jp", "jdm"), "japan"),
    (("australi", "aussie", "nz", "kiwi", "oceania"), "australia"),
    (("china", "cn"), "chinese"),
)

# Coarse |latitude| bands (degrees) per biome keyword — symmetric across
# hemispheres. Alpine/montane biomes are deliberately absent: they occur
# at every latitude and prove nothing.
_BIOME_LAT_BANDS = {
    "rainforest": (0.0, 18.0), "mangrove": (0.0, 30.0),
    "tropical": (0.0, 25.0), "jungle": (0.0, 25.0), "savanna": (0.0, 30.0),
    "subtropical": (18.0, 40.0),
    "desert": (10.0, 42.0), "arid": (10.0, 45.0),
    "mediterranean": (30.0, 46.0),
    "steppe": (30.0, 55.0), "prairie": (30.0, 55.0),
    "temperate": (35.0, 62.0), "deciduous": (35.0, 62.0),
    "coniferous": (40.0, 70.0),
    "boreal": (50.0, 70.0), "taiga": (50.0, 70.0), "subarctic": (55.0, 70.0),
    "tundra": (55.0, 90.0), "arctic": (60.0, 90.0), "polar": (66.0, 90.0),
}
_BIOME_LAT_TOL = 3.0  # degrees of slack at band edges

# Observations keys merged into the feature vector before scoring.
_OBS_MERGE_KEYS = (
    "driving_side", "sign_language", "plate_style", "road_markings",
    "lane_markings_color", "sign_conventions", "vegetation", "climate_cues",
)

# Common cc aliases: what people type -> ISO-ish code used in rankings.
_CC_ALIASES = {"UK": "GB", "EL": "GR", "UAE": "AE", "USA": "US"}

# Verdict thresholds (spec): >= supports, <= contradicts, between = murky.
_SUPPORTS_SCORE = 0.62
_CONTRADICTS_SCORE = 0.38


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _schema():
    """Lazy schema import — tolerates both package and direct-script use
    (same pattern as dedupe_evidence.py)."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # `python sundial/osint.py` direct run
        from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


def _sibling(name: str):
    """Import a parallel sibling module, or raise ImportError.

    Siblings are being built in parallel — every call site wraps this in
    try/except, so a missing module just skips that sub-step. Falls back
    to a bare import so a direct-script run can still pick siblings up
    from the same directory.
    """
    try:
        return importlib.import_module(f".{name}", __package__ or "sundial")
    except ImportError:
        return importlib.import_module(name)


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _norm_cc(cc) -> str | None:
    s = str(cc or "").strip().upper()
    if not s:
        return None
    s = _CC_ALIASES.get(s, s)
    return s if 2 <= len(s) <= 4 and s.replace("-", "").isalpha() else None


def _as_str_list(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple, set)):
        return [x for x in v if isinstance(x, str)]
    return []


def _convention_key(name) -> str | None:
    """Map a free-form convention name ('vienna-style', 'UK-style') to a
    _CONVENTION_CCS key."""
    s = str(name or "").strip().lower()
    if not s:
        return None
    if s in _CONVENTION_CCS:
        return s
    # tokenize on non-letters; aliases >= 3 chars may prefix-match a token,
    # shorter ones need an exact token ("uk" in "uk-style", not "ukraine")
    tokens = [t for t in re.split(r"[^a-z]+", s) if t]
    for aliases, key in _CONVENTION_ALIASES:
        for tok in tokens:
            if any(tok == a or (len(a) >= 3 and tok.startswith(a))
                   for a in aliases):
                return key
    return None


def _convention_members(name) -> frozenset:
    key = _convention_key(name)
    return _CONVENTION_CCS.get(key) or frozenset()


# --------------------------------------------------------------------------
# Candidate list
# --------------------------------------------------------------------------

def _plate_hint_ccs(plate) -> set:
    """Country codes implied by a plate_style observation.

    Tolerates the TEXT_PROMPT scalars ('eu', 'uk', 'us'), synonyms like
    'yellow_rear', and dict shapes ({'eu_blue_band': true, 'style': ...}).
    """
    out: set = set()
    vals: list[str] = []
    if isinstance(plate, dict):
        if plate.get("eu_blue_band"):
            out |= _PLATE_HINTS["eu"]
        if plate.get("yellow_rear") or plate.get("yellow_rear_plate"):
            out |= _PLATE_HINTS["yellow_rear"]
        for k in ("style", "region", "plate_style", "value"):
            vals += _as_str_list(plate.get(k))
    else:
        vals += _as_str_list(plate)
    for v in vals:
        key = v.strip().lower().replace("-", "_").replace(" ", "_")
        if key in _PLATE_HINTS:
            out |= _PLATE_HINTS[key]
        elif "eu" in key and ("band" in key or "blue" in key):
            out |= _PLATE_HINTS["eu"]
        elif "yellow" in key and "rear" in key:
            out |= _PLATE_HINTS["yellow_rear"]
    return out


def _language_hint_ccs(obs: dict, fv: dict) -> set:
    """Country codes implied by language/script hints.

    Reads observations.sign_language (ISO 639-1 per TEXT_PROMPT) plus the
    feature vector's pooled languages_scripts (both language codes/names
    and script labels like 'cyrillic'). Latin script is deliberately
    unlisted — it implies nothing (langdetect.py's reasoning).
    """
    out: set = set()
    vals = _as_str_list(obs.get("sign_language"))
    vals += _as_str_list(obs.get("language"))
    vals += _as_str_list(fv.get("sign_language"))
    vals += _as_str_list(fv.get("language"))
    vals += _as_str_list(fv.get("languages_scripts"))
    for v in vals:
        for tok in re.split(r"[^a-z]+", v.strip().lower()):
            if not tok or tok == "unknown":
                continue
            if tok in _LANG_HINTS:
                out |= _LANG_HINTS[tok]
            elif tok in _LANG_NAME_CODES:
                out |= _LANG_HINTS.get(_LANG_NAME_CODES[tok], set())
            elif tok in _SCRIPT_HINTS:
                out |= _SCRIPT_HINTS[tok]
    return out


def _geocode_ccs(items) -> list[str]:
    """Top geocode countries from feature_vector['geocode_ccs'].

    Accepts cc strings, or dicts with cc / country_code / code keys;
    input order is kept ('top' = first).
    """
    out: list[str] = []
    for it in items or []:
        cc = None
        if isinstance(it, str):
            cc = it
        elif isinstance(it, dict):
            cc = (it.get("cc") or it.get("country_code")
                  or it.get("countryCode") or it.get("code"))
        n = _norm_cc(cc)
        if n and n not in out:
            out.append(n)
    return out


def _build_candidates(fv: dict, obs: dict, claimed_cc) -> list[str]:
    """claimed_cc + driving-side/plate/language/convention-implied
    countries + top geocode countries — deduped, claim first."""
    out: list[str] = []

    def add(cc):
        cc = _norm_cc(cc)
        if cc and cc not in out:
            out.append(cc)

    add(claimed_cc)

    side = str(obs.get("driving_side") or fv.get("driving_side") or "")
    side = side.strip().lower()
    if side == "left":
        for cc in sorted(_LEFT_TRAFFIC):
            add(cc)
    elif side == "right":
        for cc in _RIGHT_TRAFFIC:
            add(cc)

    for cc in sorted(_plate_hint_ccs(obs.get("plate_style") or fv.get("plate_style"))):
        add(cc)
    for cc in sorted(_language_hint_ccs(obs, fv)):
        add(cc)
    conventions = (_as_str_list(obs.get("sign_conventions"))
                   + _as_str_list(fv.get("sign_conventions")))
    for conv in conventions:
        for cc in sorted(_convention_members(conv)):
            add(cc)
    for cc in _geocode_ccs(fv.get("geocode_ccs"))[:5]:  # top geocode hits
        add(cc)
    return out


# --------------------------------------------------------------------------
# Ranking normalization — match_score's exact row shape is a moving target
# --------------------------------------------------------------------------

def _norm_ranked(raw) -> list[dict]:
    """Normalize score_candidates output into [{'cc','score','hits',...}].

    Tolerates list-of-dicts ({cc|country|country_code, score|confidence,
    hits|signals}), list-of-tuples ((cc, score[, hits])), and a plain
    {cc: score} mapping. Output is deduped by cc and sorted score-desc.
    """
    if isinstance(raw, dict):  # {cc: score} mapping form
        raw = [{"cc": k, "score": v} for k, v in raw.items()]
    out: list[dict] = []
    for r in raw or []:
        d: dict
        if isinstance(r, dict):
            cc = (r.get("cc") or r.get("country_code") or r.get("country")
                  or r.get("code") or r.get("iso") or r.get("iso_cc"))
            score = r.get("score", r.get("confidence", r.get("weight", 0.0)))
            h = r.get("hits", r.get("n_hits",
                    r.get("signals", r.get("reasons", 0))))
            hits = len(h) if isinstance(h, (list, tuple, set, dict)) else int(_num(h))
            d = dict(r)
            if isinstance(h, (list, tuple, set, dict)):
                d["hit_list"] = list(h) if not isinstance(h, dict) else h
        elif isinstance(r, (list, tuple)) and r:
            cc = r[0]
            score = r[1] if len(r) > 1 else 0.0
            hits = int(_num(r[2])) if len(r) > 2 else 0
            d = {}
        else:
            continue
        ncc = _norm_cc(cc)
        if not ncc:
            continue
        d.update({"cc": ncc, "score": round(_num(score), 4), "hits": hits})
        for i, e in enumerate(out):
            if e["cc"] == ncc:  # dedupe: keep the stronger row
                if d["score"] > e["score"]:
                    out[i] = d
                break
        else:
            out.append(d)
    out.sort(key=lambda e: -e["score"])
    return out


def _find_entry(ranked: list[dict], cc) -> dict | None:
    ncc = _norm_cc(cc)
    for e in ranked or []:
        if e.get("cc") == ncc:
            return e
    return None


# --------------------------------------------------------------------------
# Sub-step extractors — defensive readers over sibling outputs
# --------------------------------------------------------------------------

def _stable_country(agg):
    """Consensus country name/cc out of aggregate_guesses output."""
    if isinstance(agg, str):
        return agg.strip() or None
    if not isinstance(agg, dict):
        return None
    for key in ("stable_country", "stable_cc", "consensus_country",
                "consensus_cc", "top_country", "top_cc", "country",
                "country_guess", "cc", "winner", "best"):
        v = agg.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            for k2 in ("cc", "country_code", "country", "name", "country_guess"):
                if isinstance(v.get(k2), str) and v[k2].strip():
                    return v[k2]
    # Deliberately no modal-vote fallback: a bare "votes" map is not a
    # stable country — aggregate_guesses exists to suppress split guesses,
    # and reading one in anyway would smuggle the quarantined signal back.
    return None


def _agg_confidence(agg) -> float:
    """Confidence attached to the geoguess consensus (0.5 when unstated)."""
    if isinstance(agg, dict):
        for k in ("confidence", "mean_conf", "stability", "share", "score",
                  "fraction"):
            v = agg.get(k)
            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and float(v) > 0):
                f = float(v)
                return max(0.0, min(1.0, f if f <= 1.0 else f / 100.0))
    return 0.5


def _top_convention(ss):
    """Top sign-style convention name out of sign_style_score output."""
    if isinstance(ss, str):
        return ss
    items: list[tuple] = []
    if isinstance(ss, (list, tuple)):
        for it in ss:
            if isinstance(it, dict):
                nm = (it.get("convention") or it.get("name")
                      or it.get("style") or it.get("key"))
                if nm:
                    items.append((nm, _num(it.get("score", it.get("confidence", 0.0)))))
            elif isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[0], str):
                items.append((it[0], _num(it[1])))
    elif isinstance(ss, dict):
        for k in ("top", "top_convention", "best", "convention", "guess"):
            if isinstance(ss.get(k), str):
                return ss[k]
        scores = ss.get("scores") or ss.get("conventions") or ss
        if isinstance(scores, dict):
            items = [(k, v) for k, v in scores.items()
                     if isinstance(v, (int, float))]
    return max(items, key=lambda kv: kv[1])[0] if items else None


def _biome_name(b) -> str:
    if isinstance(b, dict):
        for k in ("biome", "name", "label", "type"):
            if isinstance(b.get(k), str):
                return b[k]
        return ""
    return str(b or "")


def _lat_conflict(biomes, lat) -> bool:
    """Do ALL detected biomes sit outside the anchor's |lat| band?

    Conservative: unknown/untabulated biome names are ignored, and when
    nothing recognizable was detected at all there is no conflict — the
    -0.10 only fires when recognized biomes collectively disagree.
    """
    try:
        alat = abs(float(lat))
    except (TypeError, ValueError):
        return False
    bands = []
    for b in biomes or []:
        name = _biome_name(b).lower()
        for key, band in _BIOME_LAT_BANDS.items():
            if key in name:
                bands.append(band)
                break
    if not bands:
        return False
    return not any(lo - _BIOME_LAT_TOL <= alat <= hi + _BIOME_LAT_TOL
                   for lo, hi in bands)


def _conf(hits) -> float:
    return round(min(0.5 + 0.05 * int(hits or 0), 0.65), 3)


def _frames_from(guesses) -> list:
    out: list = []
    for g in guesses or []:
        if not isinstance(g, dict):
            continue
        f = (g.get("frame") or g.get("frame_path") or g.get("path")
             or g.get("file"))
        if isinstance(f, str) and f and f not in out:
            out.append(f)
    return out[:12]


# --------------------------------------------------------------------------
# Evidence card
# --------------------------------------------------------------------------

def osint_evidence(
    feature_vector: dict,
    observations: dict,
    anchor: tuple[float, float] | None,
    claimed_cc: str | None,
    per_frame_geoguesses: list[dict] | None = None,
    claim_time_iso: str | None = None,
) -> "Evidence":
    """Fuse the geo-matching signals into one Evidence card (type='osint').

    Each sub-step is isolated in try/except: missing or crashed siblings
    are skipped and the card still forms.
    """
    ClaimVerdict, Evidence = _schema()
    fv = feature_vector or {}
    obs = observations or {}
    frames = _frames_from(per_frame_geoguesses)

    candidates = _build_candidates(fv, obs, claimed_cc)

    # (a) match_score — feature vector merged with the observation keys,
    #     scored against the candidate list -> ranked rows.
    ranked: list[dict] = []
    try:
        ms = _sibling("match_score")
        merged = dict(fv)
        for k in _OBS_MERGE_KEYS:
            if obs.get(k) is not None:
                merged[k] = obs[k]
        # Bridge to the scorer's own key spellings: match_score reads
        # "sign_convention" (singular) and "climate" (not climate_cues).
        convs = _as_str_list(merged.get("sign_conventions"))
        if convs and not merged.get("sign_convention"):
            merged["sign_convention"] = convs[0]
        if merged.get("climate") is None and merged.get("climate_cues") is not None:
            merged["climate"] = merged["climate_cues"]
        try:
            raw = ms.score_candidates(merged, candidates)
        except TypeError:
            try:
                raw = ms.score_candidates(merged, candidates=candidates)
            except TypeError:
                raw = ms.score_candidates(merged)
        ranked = _norm_ranked(raw)
    except Exception:
        ranked = []

    # (b) geoguess — aggregate the per-frame guesses; a stable country that
    #     isn't already a candidate joins the ranking.
    agg: dict = {}
    try:
        gg = _sibling("geoguess")
        agg = gg.aggregate_guesses(per_frame_geoguesses or [])
        stable = _stable_country(agg)
        if stable:
            try:
                cc = _norm_cc(gg.resolve_cc(stable))
            except Exception:
                s = str(stable).strip()
                cc = _norm_cc(s) if len(s) <= 4 else None
            if cc:
                if cc not in candidates:
                    candidates.append(cc)
                if _find_entry(ranked, cc) is None:
                    # Quarantine: a lone VLM guess may join the ranking but
                    # never carry it — capped at geoguess._MAX_CONF (0.4),
                    # squarely inside the inconclusive band.
                    ranked.append({
                        "cc": cc,
                        "score": round(min(_agg_confidence(agg), 0.4), 3),
                        "hits": 0,
                        "source": "geoguess",
                    })
                    ranked.sort(key=lambda e: -e["score"])
    except Exception:
        if not isinstance(agg, dict):
            agg = {}

    # (c) sign_style — +0.05 to every member country of the top convention.
    ss: dict = {}
    try:
        sty = _sibling("sign_style")
        ss = (sty.sign_style_score(fv.get("sign_style") or {},
                                   fv.get("marking_colors")) or {})
        top = _top_convention(ss)
        # Member countries of the top convention: the sibling's own
        # _CONVENTIONS table is authoritative; fall back to our copy.
        members: set = set()
        try:
            for c in getattr(sty, "_CONVENTIONS", None) or []:
                if (isinstance(c, dict) and c.get("key") == top
                        and isinstance(c.get("countries"),
                                       (set, frozenset, list, tuple))):
                    members = {_norm_cc(x) or str(x).upper()
                               for x in c["countries"]}
                    break
        except Exception:
            members = set()
        if not members:
            members = set(_convention_members(top))
        if members:
            touched = False
            for e in ranked:
                if e.get("cc") in members:
                    e["score"] = round(e["score"] + 0.05, 4)
                    touched = True
            if touched:
                ranked.sort(key=lambda e: -e["score"])
    except Exception:
        if not isinstance(ss, (dict, list)):
            ss = {}

    # (d) veg_geo — biome/lat-band conflict is a hard -0.10 on everyone.
    #     veg_biomes returns a {biome: score} map (zeros included) — only
    #     positively-scored biomes count as detected.
    biomes = []
    try:
        vg = _sibling("veg_geo")
        biomes = vg.veg_biomes(fv.get("vegetation") or []) or []
        if isinstance(biomes, dict):
            detected = [k for k, v in biomes.items() if _num(v) > 0]
        else:
            detected = list(biomes)
        if anchor is not None and len(anchor) >= 1 \
                and _lat_conflict(detected, anchor[0]):
            for e in ranked:
                e["score"] = round(e["score"] - 0.10, 4)
            ranked.sort(key=lambda e: -e["score"])
    except Exception:
        if not isinstance(biomes, (dict, list)):
            biomes = []

    data = {
        "ranked": ranked[:8],
        "geoguess": agg,
        "sign_scores": ss,
        "biomes": biomes,
        "claimed_cc": claimed_cc,
    }

    def _top_str(n: int) -> str:
        return ", ".join(f"{e['cc']} {e['score']:.2f}" for e in ranked[:n])

    # (e) verdict on the claimed country (or top-3 when there is no claim).
    if claimed_cc is None:
        tops = _top_str(3)
        return Evidence(
            type="osint",
            summary=(f"no claimed country — top geo-match candidates: {tops}"
                     if tops else
                     "no claimed country and no geo-match candidates"),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.3,
            data=data,
            frames=frames,
        )

    cc = _norm_cc(claimed_cc) or str(claimed_cc).strip().upper()
    if not ranked:
        return Evidence(
            type="osint",
            summary=(f"no geo-match ranking could be produced for claimed "
                     f"{cc} (scoring signals unavailable)"),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
            frames=frames,
        )

    ent = _find_entry(ranked, cc)
    score = ent["score"] if ent else 0.0
    hits = ent.get("hits", 0) if ent else 0
    top = ranked[0]
    top_str = f"top candidate {top['cc']} {top['score']:.2f}"

    if score >= _SUPPORTS_SCORE:
        rank = ranked.index(ent) + 1
        return Evidence(
            type="osint",
            summary=(
                f"geo signals rank {cc} #{rank} "
                f"(score {score:.2f}, {hits} hit(s)) — consistent with "
                f"claimed {cc}"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=_conf(hits),
            data=data,
            frames=frames,
        )
    if score <= _CONTRADICTS_SCORE:
        what = (f"claimed {cc} is absent from the geo-match ranking"
                if ent is None else
                f"geo signals score claimed {cc} low ({score:.2f})")
        return Evidence(
            type="osint",
            summary=f"{what} ({top_str}) — inconsistent with the claim",
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=_conf(hits),
            data=data,
            frames=frames,
        )
    return Evidence(
        type="osint",
        summary=(
            f"geo score for claimed {cc} is ambiguous ({score:.2f}; "
            f"{top_str}) — neither confirms nor refutes"
        ),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.3,
        data=data,
        frames=frames,
    )


# --------------------------------------------------------------------------
# Self-test: python -m sundial.osint  (or python sundial/osint.py)
# --------------------------------------------------------------------------
# Sibling stubs implement the documented signatures so the fusion path is
# exercised deterministically even before the real modules land; real
# sys.modules entries are restored afterwards.

def _stub_score_candidates(fv, candidates=None):
    scores = {"GB": 0.72, "IE": 0.55, "AU": 0.44, "NL": 0.35}
    return [{"cc": c, "score": scores.get(c, 0.30),
             "hits": ["left_drive", "plate_uk", "lang_en", "geocode"]
                      if c == "GB" else ["generic"]}
            for c in (candidates or ["GB", "IE", "AU"])]


def _stub_aggregate_guesses(guesses):
    return {"stable_country": "united kingdom", "mean_conf": 0.65,
            "votes": {"united kingdom": 2, "ireland": 1},
            "regions": ["england"], "reasonings": ["stub"]}


def _stub_resolve_cc(name):
    table = {"united kingdom": "GB", "great britain": "GB",
             "england": "GB", "ireland": "IE"}
    s = str(name or "").strip().lower()
    if s in table:
        return table[s]
    u = str(name or "").strip().upper()
    return u if u.isalpha() and len(u) in (2, 3) else None


def _stub_sign_style_score(feats, marking_colors=None):
    return {"uk": 0.8, "vienna": 0.35, "mutcd": 0.1,
            "japan": 0.0, "australia": 0.2}


def _stub_veg_biomes(vegetation):
    if not vegetation:
        return {}
    return {"temperate": 0.8, "boreal": 0.0, "tropical": 0.0}


def _install_stubs() -> dict:
    """Register sibling stubs under sundial.* and bare names; returns the
    sys.modules/package-attr state needed to undo it."""
    import sys
    import types

    pkg_name = __package__ or "sundial"
    pkg = sys.modules.get(pkg_name)
    missing = object()
    saved: dict = {"pkg": pkg, "missing": missing, "mods": {}, "attrs": {}}
    stubs = {
        "match_score": {"score_candidates": _stub_score_candidates},
        "geoguess": {"aggregate_guesses": _stub_aggregate_guesses,
                     "resolve_cc": _stub_resolve_cc},
        "sign_style": {"sign_style_score": _stub_sign_style_score},
        "veg_geo": {"veg_biomes": _stub_veg_biomes},
    }
    for name, attrs in stubs.items():
        m = types.ModuleType(f"{pkg_name}.{name}")
        for k, v in attrs.items():
            setattr(m, k, v)
        for full in (f"{pkg_name}.{name}", name):
            saved["mods"][full] = sys.modules.get(full, missing)
            sys.modules[full] = m
        if pkg is not None:
            saved["attrs"][name] = getattr(pkg, name, missing)
            setattr(pkg, name, m)
    return saved


def _restore_stubs(saved: dict) -> None:
    import sys
    missing = saved["missing"]
    for full, old in saved["mods"].items():
        if old is missing:
            sys.modules.pop(full, None)
        else:
            sys.modules[full] = old
    pkg = saved["pkg"]
    if pkg is not None:
        for name, old in saved["attrs"].items():
            if old is missing:
                try:
                    delattr(pkg, name)
                except AttributeError:
                    pass
            else:
                setattr(pkg, name, old)


def _self_test() -> int:
    fv = {
        "geocode_ccs": ["GB", "IE", "FR"],
        "sign_style": {"shape": "triangular", "colors": "red on white",
                       "convention_guess": "uk-style"},
        "marking_colors": ["white"],
        "vegetation": ["oak", "hedgerow", "bracken"],
        "languages_scripts": ["en", "latin"],
        "sign_conventions": ["uk-style"],
        "terrain": "rolling",
        "architecture": "red brick victorian terraces",
        "climate_cues": ["overcast", "damp"],
        "vehicle_notes": ["black cab", "double-decker bus"],
    }
    obs = {
        "driving_side": "left",
        "sign_language": "en",
        "plate_style": "uk",
        "road_markings": ["double_yellow_lines", "zigzag_lines"],
        "lane_markings_color": "white",
        "sign_conventions": ["uk-style"],
        "vegetation": ["oak"],
        "climate_cues": ["overcast"],
    }
    guesses = [
        {"frame": "f1.jpg", "country_guess": "United Kingdom",
         "confidence": 0.7},
        {"frame": "f2.jpg", "country_guess": "United Kingdom",
         "confidence": 0.6},
        {"frame": "f3.jpg", "country_guess": "Ireland", "confidence": 0.4},
    ]
    anchor = (51.5074, -0.1278)  # London — inside the temperate band

    fails = 0
    saved = _install_stubs()
    try:
        ev = osint_evidence(fv, obs, anchor, "GB",
                            per_frame_geoguesses=guesses,
                            claim_time_iso="2024-06-01T13:00:00")
        print(f"evidence[{ev.type}] {ev.verdict.value} "
              f"conf={ev.confidence}: {ev.summary}")
        ok = (ev.type == "osint" and ev.verdict.value == "supports"
              and 0.0 < ev.confidence <= 1.0)
        print(f"  {'PASS' if ok else 'FAIL'}  UK-flavored vector + GB claim "
              "-> supports")
        fails += 0 if ok else 1

        ev2 = osint_evidence({}, {}, None, None)
        print(f"evidence[{ev2.type}] {ev2.verdict.value} "
              f"conf={ev2.confidence}: {ev2.summary}")
        ok = ev2.type == "osint" and ev2.verdict.value == "inconclusive"
        print(f"  {'PASS' if ok else 'FAIL'}  empty input -> inconclusive "
              "top-3 card")
        fails += 0 if ok else 1
    finally:
        _restore_stubs(saved)

    # Degraded path: with no scoring siblings the card must still form.
    ev3 = osint_evidence(fv, obs, anchor, "GB")
    print(f"evidence[{ev3.type}] {ev3.verdict.value} "
          f"conf={ev3.confidence}: {ev3.summary}")
    ok = ev3.type == "osint"
    print(f"  {'PASS' if ok else 'FAIL'}  siblings-missing path still emits "
          "a card")
    fails += 0 if ok else 1

    print(f"self-test: {'all checks passed' if not fails else f'{fails} FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
