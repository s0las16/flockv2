"""Geo tables — the unified "pigeon matrix" over every geo cue source.

Sundial has many small geo-cue producers — the VLM osint feature vector,
the text-stage observations, the GeoTips community-meta CSV scorer, plus
model-level priors (a CLIP-style country guess, a hierarchical region
vote). Each downstream consumer used to re-score cues on its own. This
module folds ALL of them into ONE normalized country matrix — the
"pigeon table" — so the pipeline scores once and everyone reads the same
cells:

    matrix = build_matrix(osint_vec, observations, extras)
    # {"DE": {"score": 0.0..1.0,
    #         "hits": ["drive right", "eu plate", "german text", ...],
    #         "sources": {"geotips": x, "direct": y, "meta": z}}, ...}

Sources merged per country code:

  * direct  — this module's own cue tables (driving side, plate family,
              road markings, sign convention, signage language/script,
              ccTLD domains, country-indicative tokens). Self-contained
              sets in the geotips/match_score style; each observed cue
              adds weight to every country it typifies, and the PRIMARY
              observed value of driving_side / plate / language /
              convention also subtracts a small penalty from countries
              it contradicts.
  * geotips — sibling geotips.tip_scores (or its _score_detail for per-
              cue hit names) folded in lazily when importable.
  * meta    — extras={"clip_prior": {cc: w}, "hier": {cc: w}}, each
              min-normalized and folded at 0.5 weight. Priors move the
              score but are not "cue hits", so they land in sources.meta
              only.

Normalization: each source's raw score is rescaled by its own max (best
country = 1.0), combined, then rescaled again by the best combined
score — every "score" in the matrix is 0..1 with the leader at 1.0.
"sources" keeps each source's weighted contribution (0..1 each) so a
consumer can see WHERE a country's support came from.

Pure stdlib (re only) at import; .schema and .geotips are lazy so the
module loads and scores with zero project deps. Never raises: empty or
unreadable inputs yield an empty matrix and INCONCLUSIVE evidence.

Stage API:

    mx   = build_matrix(osint_vec, observations, extras)
    top8 = top(mx)                       # [{cc, score, hits_n}]
    rank = claimed_rank(mx, "FR")        # {rank, score, of}
    txt  = explain(mx, "DE")             # "DE: 6 cue hits — eu plate, ..."
    ev   = matrix_evidence(mx, "FR")     # Evidence card, type="geo_matrix"
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Weights — deliberately modest; cue fusion corroborates, never convicts.
# ---------------------------------------------------------------------------
_W_DOMAIN = 1.6       # observed ccTLD / "de domain" — near-fingerprint
_W_INDICATOR = 1.4    # consensus text explicitly names a country
_W_PLATE = 1.2        # licence-plate family
_W_DRIVING = 1.0      # left vs right traffic
_W_LANG = 0.8         # signage language
_W_CONVENTION = 0.7   # sign-design family (vienna / mutcd / uk / ...)
_W_MARKING = 0.5      # per road-marking token
_W_SCRIPT = 0.35      # script family — broad sets, weak

_MISS_DRIVE = 0.50    # observed "left" but cc drives right (and v.v.)
_MISS_PLATE = 0.30    # primary plate token contradicts cc's family
_MISS_LANG = 0.30     # primary signage language contradicts
_MISS_CONV = 0.20     # primary sign-convention contradicts

_EXTRAS_KEYS = ("clip_prior", "hier")
_EXTRAS_FOLD = 0.5    # each extras distribution folded at half weight

_CC_ALIASES = {"UK": "GB", "EL": "GR", "USA": "US", "UAE": "AE",
               "SU": "RU", "TP": "TL"}

# ---------------------------------------------------------------------------
# Direct cue tables — compact {token: {cc,...}} / cc sets, same style as
# match_score._PLATE_HINTS & friends (kept in sync by hand).
# ---------------------------------------------------------------------------

# EU-style blue-band plate family (circle-of-stars strip down the left).
_EU = {
    "DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "PL", "IE", "SE",
    "DK", "FI", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "EE",
    "LV", "LT", "LU", "MT", "CY",
}

# Countries that keep left.
_LEFT_TRAFFIC = {
    "GB", "IE", "JP", "AU", "NZ", "IN", "ZA", "TH", "MY", "SG", "ID", "HK",
    "KE", "MT", "CY", "PK", "BD", "LK", "NP", "MU", "TT", "JM", "BB", "BS",
    "GY", "SR", "BN", "BT", "MO", "NA", "BW", "LS", "SZ", "MZ", "ZM", "MW",
    "TZ", "UG", "FJ", "PG", "IM", "JE", "GG", "VI", "CX", "CC", "NF", "TL",
}

# Plate-family tokens -> countries where that furniture is typical.
_PLATE_HINTS: dict[str, set[str]] = {
    "eu": _EU, "european": _EU, "euro": _EU, "eu_band": _EU,
    "blue_band": _EU, "blue_strip": _EU, "left_blue_strip": _EU,
    "euroband": _EU, "euro_band": _EU,
    "uk": {"GB"}, "gb": {"GB"}, "british": {"GB"},
    "yellow_rear": {"GB", "NL", "JP", "HK", "AU", "IN", "BW", "CX"},
    "us": {"US"}, "usa": {"US"}, "us_style": {"US", "CA", "MX"},
    "american": {"US"},
    "mercosur": {"AR", "BR", "PY", "UY"},
    "jp": {"JP"}, "japanese": {"JP"},
    "cn": {"CN"}, "chinese": {"CN"},
    "kr": {"KR"}, "korean": {"KR"},
    "in": {"IN"}, "indian": {"IN"},
    "ru": {"RU"}, "russian": {"RU"},
    "br": {"BR"}, "brazilian": {"BR"},
    "au": {"AU"}, "australian": {"AU"},
    "za": {"ZA"},
}

# Road-marking tokens -> countries where they are typical. Plain "yellow"
# is the yellow-centerline tell (Americas / Nordics / East Asia); plain
# "white" is too generic worldwide to carry a set.
_YELLOW_CENTER = {"US", "CA", "BR", "CN", "KR", "TW", "NO", "SE", "FI"}
_MARKING_HINTS: dict[str, set[str]] = {
    "yellow": _YELLOW_CENTER,
    "yellow_centerline": _YELLOW_CENTER,
    "yellow_center": _YELLOW_CENTER,
    "yellow_center_line": _YELLOW_CENTER,
    "double_yellow": {"GB", "IE", "US", "CA"},
    "double_yellow_lines": {"GB", "IE"},
    "double_red": {"GB"},
    "double_red_lines": {"GB"},
    "zigzag": {"GB", "IE", "SG"},
    "zigzag_lines": {"GB", "IE", "SG"},
    "keep_left_bollards": {"GB", "IE", "AU", "NZ", "JP"},
}

# Sign-design convention families (Vienna convention, US MUTCD, UK-style).
_VIENNA = {
    "DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "CH", "LU", "LI",
    "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "RS", "BA",
    "ME", "MK", "AL", "SE", "DK", "FI", "NO", "IS", "EE", "LV", "LT",
    "RU", "UA", "BY", "TR", "MC", "SM", "VA", "KZ", "KG", "MD", "AM",
    "AZ", "GE", "UZ", "CN", "VN", "ID", "KR",
}
_MUTCD = {
    "US", "CA", "MX", "BR", "AR", "CL", "CO", "PE", "VE", "PH", "AU",
    "NZ", "TH", "TW", "KR", "JP",
}
_UK_STYLE = {
    "GB", "IE", "MT", "CY", "AU", "NZ", "IN", "ZA", "SG", "MY", "HK",
    "KE", "NG", "PK", "BD", "LK",
}
_CONVENTION_HINTS: dict[str, set[str]] = {
    "vienna": _VIENNA, "mutcd": _MUTCD, "uk": _UK_STYLE,
    "japan": {"JP"}, "australia": {"AU", "NZ"},
}
# What observers/VLMs write -> the convention key above.
_CONV_ALIAS: dict[str, str] = {
    "vienna_style": "vienna", "vienna_convention": "vienna",
    "european": "vienna", "europe": "vienna", "eu": "vienna",
    "continental": "vienna",
    "mutcd_style": "mutcd", "us": "mutcd", "usa": "mutcd",
    "american": "mutcd", "us_style": "mutcd",
    "uk_style": "uk", "british": "uk", "commonwealth": "uk",
    "japanese": "japan", "japan_style": "japan",
    "australian": "australia", "australia_nz": "australia",
}

# Dominant signage language per country (rough; same shape as
# match_score._LANG_HINTS).
_LANG_HINTS: dict[str, set[str]] = {
    "en": {"GB", "IE", "US", "AU", "NZ", "CA", "ZA", "IN", "SG", "NG",
           "KE", "JM", "MT", "GY"},
    "fr": {"FR", "BE", "CH", "LU", "MC", "CA", "SN", "CI", "ML", "BF",
           "NE", "TG", "BJ", "CD", "CG", "MG"},
    "de": {"DE", "AT", "CH", "LU", "LI"},
    "es": {"ES", "MX", "AR", "CO", "CL", "PE", "VE", "EC", "GT", "CU",
           "BO", "DO", "HN", "PY", "SV", "NI", "CR", "PA", "UY", "GQ"},
    "pt": {"PT", "BR", "AO", "MZ", "CV", "GW", "ST", "TL"},
    "it": {"IT", "CH", "SM", "VA"},
    "nl": {"NL", "BE", "SR", "AW", "CW"},
    "ja": {"JP"}, "ko": {"KR"}, "zh": {"CN", "TW", "HK", "MO", "SG"},
    "ar": {"SA", "AE", "EG", "QA", "KW", "BH", "OM", "JO", "LB", "IQ",
           "SY", "MA", "DZ", "TN", "LY", "SD", "YE", "PS", "MR", "DJ"},
    "ru": {"RU", "BY", "KZ", "KG"},
    "hi": {"IN"}, "th": {"TH"}, "vi": {"VN"}, "id": {"ID"},
    "ms": {"MY", "BN"},
    "tr": {"TR", "CY"}, "el": {"GR", "CY"}, "pl": {"PL"}, "uk": {"UA"},
    "sv": {"SE"}, "da": {"DK"}, "no": {"NO"}, "fi": {"FI"}, "cs": {"CZ"},
    "sk": {"SK"}, "hu": {"HU"}, "ro": {"RO"}, "bg": {"BG"}, "hr": {"HR"},
    "he": {"IL"}, "fa": {"IR"}, "ur": {"PK"}, "bn": {"BD", "IN"},
}
# English language names -> the ISO keys used above.
_LANG_NAME: dict[str, str] = {
    "english": "en", "french": "fr", "german": "de", "spanish": "es",
    "portuguese": "pt", "italian": "it", "dutch": "nl", "japanese": "ja",
    "korean": "ko", "chinese": "zh", "mandarin": "zh", "arabic": "ar",
    "russian": "ru", "thai": "th", "turkish": "tr", "greek": "el",
    "polish": "pl", "ukrainian": "uk", "swedish": "sv", "danish": "da",
    "norwegian": "no", "finnish": "fi", "czech": "cs", "slovak": "sk",
    "hungarian": "hu", "romanian": "ro", "bulgarian": "bg",
    "croatian": "hr", "hebrew": "he", "hindi": "hi", "vietnamese": "vi",
    "indonesian": "id", "malay": "ms", "urdu": "ur", "bengali": "bn",
    "flemish": "nl", "catalan": "es", "persian": "fa",
}

# Script families -> countries using them. Latin is deliberately broad
# (weak cue); the others are much sharper.
_LATIN = _EU | _MUTCD | _UK_STYLE | {
    "TR", "VN", "ID", "PH", "TL", "MX", "BR", "AR", "CL", "CO", "PE",
    "VE", "EC", "GT", "CU", "BO", "DO", "HN", "PY", "SV", "NI", "CR",
    "PA", "UY", "GY", "SR", "KE", "TZ", "UG", "NG", "GH", "SN", "CI",
    "BW", "NA", "MZ", "ZM", "MW", "ET", "SO", "RW", "CD", "CG", "CM",
}
_SCRIPT_HINTS: dict[str, set[str]] = {
    "latin": _LATIN, "roman": _LATIN,
    "cyrillic": {"RU", "UA", "BY", "BG", "RS", "MK", "ME", "KZ", "KG",
                 "MN", "TJ", "UZ", "BA", "MD"},
    "arabic": {"SA", "AE", "EG", "QA", "KW", "BH", "OM", "JO", "LB",
               "IQ", "SY", "MA", "DZ", "TN", "LY", "SD", "YE", "PS",
               "MR", "DJ", "IR", "PK", "AF"},
    "japanese": {"JP"}, "hiragana": {"JP"}, "katakana": {"JP"},
    "hangul": {"KR"}, "korean": {"KR"},
    "chinese": {"CN", "TW", "HK", "MO", "SG"}, "hanzi": {"CN"},
    "thai": {"TH"}, "greek": {"GR", "CY"}, "hebrew": {"IL"},
    "georgian": {"GE"}, "armenian": {"AM"},
    "devanagari": {"IN", "NP"}, "bengali": {"BD", "IN"},
    "khmer": {"KH"}, "lao": {"LA"}, "burmese": {"MM"}, "myanmar": {"MM"},
    "tamil": {"IN", "LK", "SG"}, "sinhala": {"LK"}, "ethiopic": {"ET"},
}

# Observation keys consulted per cue family (osint_vec + observations
# merged before scoring).
_PLATE_KEYS = ("plate_style", "plates", "license_plate")
_MARKING_KEYS = ("road_markings", "lane_markings_color", "marking_colors",
                 "markings")
_LANG_KEYS = ("sign_language", "languages", "language")
_LANGSCRIPT_KEYS = ("languages_scripts", "sign_language", "languages",
                    "language", "scripts", "script", "sign_script")
_CONV_KEYS = ("sign_conventions", "sign_convention", "sign_style",
              "convention")
_INDICATOR_KEYS = ("geocode_ccs", "country_indicators", "country_ccs")
_DOMAIN_KEYS = ("domain", "cctld", "tld")


# ---------------------------------------------------------------------------
# Small normalization helpers
# ---------------------------------------------------------------------------

def _norm_tok(s) -> str:
    """'Yellow Rear!' -> 'yellow_rear'; junk -> ''."""
    return _WS_RE.sub("_", str(s or "").strip().lower()).strip("_")


def _norm_cc(value) -> str:
    """Uppercase a country code and map common aliases ('UK' -> 'GB')."""
    cc = (str(value or "")).strip().upper()
    return _CC_ALIASES.get(cc, cc)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _tokens(value) -> list[str]:
    """Coerce an observed value (str | list | dict | number) into a
    deduped list of normalized snake tokens. Strings split on , ; /."""

    out: list[str] = []

    def eat(item) -> None:
        for piece in re.split(r"[,;/]+", str(item or "")):
            t = _norm_tok(piece)
            if t and t not in out:
                out.append(t)

    def walk(v) -> None:
        if isinstance(v, str):
            eat(v)
        elif isinstance(v, dict):
            for vv in v.values():
                walk(vv)
        else:
            try:
                for it in v:
                    walk(it)
            except TypeError:
                if v is not None and not isinstance(v, bool):
                    eat(v)

    walk(value)
    return out


def _boundary_sub(needle: str, hay: str) -> bool:
    """needle occurs in hay bounded by '_' or string edges (snake-aware)."""
    if len(needle) < 3 or len(needle) > len(hay):
        return False
    idx = hay.find(needle)
    while idx != -1:
        left_ok = idx == 0 or hay[idx - 1] == "_"
        end = idx + len(needle)
        right_ok = end == len(hay) or hay[end] == "_"
        if left_ok and right_ok:
            return True
        idx = hay.find(needle, idx + 1)
    return False


def _hint_keys(tok: str, table: dict) -> list[str]:
    """Table keys matching an observed token: exact, word-level, or
    snake-boundary substring ('white_front_yellow_rear' ~ 'yellow_rear')."""
    keys: list[str] = []

    def add(k) -> None:
        if k in table and k not in keys:
            keys.append(k)

    add(tok)
    for w in _WORD_RE.findall(tok):
        add(w)
    for k in table:
        if _boundary_sub(k, tok):
            add(k)
    return keys


def _first_tokens(obs: dict, keys) -> list[str]:
    """Deduped union of normalized tokens across the first-matching keys."""
    out: list[str] = []
    for key in keys:
        for t in _tokens((obs or {}).get(key)):
            if t not in out:
                out.append(t)
    return out


def _expand_tokens(obs: dict, keys, table: dict) -> list[str]:
    """_first_tokens plus every table-matching WORD inside the tokens, so
    'white_and_yellow' also surfaces the 'yellow' cue."""
    out = _first_tokens(obs, keys)
    for t in list(out):
        for w in _WORD_RE.findall(t):
            if w in table and w not in out:
                out.append(w)
    return out


# ---------------------------------------------------------------------------
# Direct cue scoring — {cc: {"raw": float, "hits": [cue descriptions]}}
# ---------------------------------------------------------------------------

def _universe() -> set[str]:
    """Every cc known to any direct table — the miss-penalty universe."""
    u: set[str] = set(_LEFT_TRAFFIC)
    for table in (_PLATE_HINTS, _MARKING_HINTS, _CONVENTION_HINTS,
                  _LANG_HINTS, _SCRIPT_HINTS):
        for ccset in table.values():
            u |= set(ccset)
    return u


def _direct_scores(obs: dict) -> dict[str, dict]:
    """Score this module's own cue tables against the observed dict.

    Positive-only cues (markings, scripts, domains, indicators) add
    weight to each cc in the token's set. The PRIMARY observed value of
    driving_side / plate / language / convention additionally subtracts
    a miss penalty from universe countries the cue contradicts — a
    left-traffic scene argues AGAINST Germany, not just for the UK set.
    """
    cells: dict[str, dict] = {}
    uni = _universe()

    def cell(cc: str) -> dict:
        return cells.setdefault(cc, {"raw": 0.0, "hits": []})

    def boost(ccset, w: float, desc: str) -> None:
        for cc in ccset or ():
            c = cell(cc)
            c["raw"] += w
            if desc and desc not in c["hits"]:
                c["hits"].append(desc)

    def miss(exclude, penalty: float) -> None:
        for cc in uni - set(exclude or ()):
            cell(cc)["raw"] -= penalty

    def pretty(tok: str) -> str:
        return tok.replace("_", " ")

    # --- driving side ----------------------------------------------------
    side = (_first_tokens(obs, ("driving_side",)) or [""])[0]
    if side in ("left", "right"):
        for cc in uni:
            c = cell(cc)
            if (side == "left") == (cc in _LEFT_TRAFFIC):
                c["raw"] += _W_DRIVING
                c["hits"].append(f"drive {side}")
            else:
                c["raw"] -= _MISS_DRIVE

    # --- plate family -----------------------------------------------------
    plates = _expand_tokens(obs, _PLATE_KEYS, _PLATE_HINTS)
    for tok in plates:
        for key in _hint_keys(tok, _PLATE_HINTS):
            boost(_PLATE_HINTS[key], _W_PLATE, f"{pretty(tok)} plate")
    if plates:
        hit_sets = [s for k in _hint_keys(plates[0], _PLATE_HINTS)
                    for s in [_PLATE_HINTS[k]]]
        if hit_sets:
            miss(set().union(*hit_sets), _MISS_PLATE)

    # --- road markings -----------------------------------------------------
    for tok in _expand_tokens(obs, _MARKING_KEYS, _MARKING_HINTS):
        for key in _hint_keys(tok, _MARKING_HINTS):
            desc = pretty(tok) if "line" in tok or "marking" in tok \
                else f"{pretty(tok)} lines"
            boost(_MARKING_HINTS[key], _W_MARKING, desc)

    # --- sign-design convention --------------------------------------------
    convs: list[str] = []
    for tok in _first_tokens(obs, _CONV_KEYS):
        key = _CONVENTION_HINTS.get(tok) and tok or _CONV_ALIAS.get(tok)
        if key is None:
            for w in _WORD_RE.findall(tok):
                if w in _CONVENTION_HINTS or w in _CONV_ALIAS:
                    key = w if w in _CONVENTION_HINTS else _CONV_ALIAS[w]
                    break
        if key and key in _CONVENTION_HINTS and key not in convs:
            convs.append(key)
    for key in convs:
        boost(_CONVENTION_HINTS[key], _W_CONVENTION, f"{key}-style signs")
    if convs:
        miss(_CONVENTION_HINTS[convs[0]], _MISS_CONV)

    # --- signage language ----------------------------------------------------
    langs: list[str] = []
    for tok in _first_tokens(obs, _LANG_KEYS):
        key = tok if tok in _LANG_HINTS else _LANG_NAME.get(tok)
        if key is None:
            for w in _WORD_RE.findall(tok):
                if w in _LANG_HINTS or w in _LANG_NAME:
                    key = w if w in _LANG_HINTS else _LANG_NAME[w]
                    break
        if key and key in _LANG_HINTS and key not in langs:
            langs.append(key)
    for key in langs:
        boost(_LANG_HINTS[key], _W_LANG, f"{key} text")
    if langs:
        miss(_LANG_HINTS[langs[0]], _MISS_LANG)

    # --- script family (word-level; 'latin script' ~ 'latin') -----------------
    seen_scripts: set[str] = set()
    for tok in _first_tokens(obs, _LANGSCRIPT_KEYS):
        for key in _hint_keys(tok, _SCRIPT_HINTS):
            if key not in seen_scripts:
                seen_scripts.add(key)
                boost(_SCRIPT_HINTS[key], _W_SCRIPT, f"{key} script")

    # --- ccTLD / domain -------------------------------------------------------
    for tok in _first_tokens(obs, _DOMAIN_KEYS):
        m = re.search(r"([a-z]{2})$", tok)
        if not m:
            continue
        cc = _norm_cc(m.group(1))
        if len(cc) == 2 and cc.isalpha():
            boost({cc}, _W_DOMAIN, f"{m.group(1)} domain")

    # --- country-indicative tokens (dial codes, ccTLDs, cc lists) ------------
    for tok in _first_tokens(obs, _INDICATOR_KEYS):
        cc = _norm_cc(tok)
        if len(cc) == 2 and cc.isalpha():
            boost({cc}, _W_INDICATOR, f"text names {cc}")

    return cells


# ---------------------------------------------------------------------------
# geotips — lazy sibling import; _score_detail preferred for cue names,
# tip_scores ({cc: raw}) as the documented fallback.
# ---------------------------------------------------------------------------

def _geotips_detail(obs: dict) -> dict[str, dict]:
    """{cc: {"score": raw, "cues": [obs keys that fired]}}; {} on any
    failure (geotips missing, CSV unreadable, scoring blew up)."""
    try:
        from . import geotips
    except ImportError:
        try:
            import geotips  # direct `python3 sundial/geo_tables.py` run
        except ImportError:
            return {}
    except Exception:
        return {}

    detail: dict = {}
    fn = getattr(geotips, "_score_detail", None)
    if callable(fn):
        try:
            detail = fn(obs) or {}
        except Exception:
            detail = {}
    if not detail:
        try:
            scores = geotips.tip_scores(obs) or {}
        except Exception:
            scores = {}
        detail = {cc: {"score": s, "cues": []} for cc, s in scores.items()}

    out: dict[str, dict] = {}
    for cc, d in detail.items():
        ncc = _norm_cc(cc)
        if not ncc or not isinstance(d, dict):
            continue
        try:
            score = float(d.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score <= 0.0:
            continue
        out[ncc] = {"score": score, "cues": list(d.get("cues") or [])}
    return out


# ---------------------------------------------------------------------------
# extras — {"clip_prior": {cc: w}, "hier": {cc: w}} folded at 0.5 weight
# ---------------------------------------------------------------------------

def _extras_scores(extras) -> dict[str, float]:
    """{cc: meta weight 0..1} — each extras distribution is max-normalized
    then folded at _EXTRAS_FOLD, so both sources at full strength give
    meta = 1.0."""
    out: dict[str, float] = {}
    if not isinstance(extras, dict):
        return out
    for key in _EXTRAS_KEYS:
        dist = extras.get(key)
        if not isinstance(dist, dict):
            continue
        vals: dict[str, float] = {}
        for k, v in dist.items():
            cc = _norm_cc(k)
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if cc and w > 0.0:
                vals[cc] = max(w, vals.get(cc, 0.0))
        mx = max(vals.values(), default=0.0)
        if mx <= 0.0:
            continue
        for cc, w in vals.items():
            out[cc] = out.get(cc, 0.0) + _EXTRAS_FOLD * (w / mx)
    return out


def _merge_obs(osint_vec, observations) -> dict:
    """osint_vec overlaid with observations — same-key values merge
    (list+list unions, str+str concats) rather than clobbering."""
    obs = dict(osint_vec) if isinstance(osint_vec, dict) else {}
    if not isinstance(observations, dict):
        return obs
    for k, v in observations.items():
        if v in (None, "", [], {}):
            continue
        cur = obs.get(k)
        if cur in (None, "", [], {}):
            obs[k] = v
        elif isinstance(cur, list) and isinstance(v, list):
            obs[k] = cur + [x for x in v if x not in cur]
        elif isinstance(cur, list):
            obs[k] = cur + ([v] if v not in cur else [])
        elif isinstance(v, list):
            obs[k] = [cur] + [x for x in v if x != cur]
        elif isinstance(cur, str) and isinstance(v, str):
            if cur != v:
                obs[k] = f"{cur},{v}"
        else:
            obs[k] = v
    return obs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_matrix(osint_vec, observations=None, extras=None) -> dict:
    """One normalized scoring pass over every geo cue source.

    Returns {cc: {"score": 0..1, "hits": [cue_descriptions],
    "sources": {"geotips": x, "direct": y, "meta": z}}}, ordered
    best-first. Each source is max-normalized to 0..1, combined, then the
    combined column is max-normalized again (leader = 1.0). "sources"
    holds the three normalized contributions — they sum to the cc's raw
    combined score before the final rescale. Empty input -> {}.
    """
    try:
        obs = _merge_obs(osint_vec, observations)
    except Exception:
        obs = osint_vec if isinstance(osint_vec, dict) else {}

    try:
        direct = _direct_scores(obs)
    except Exception:
        direct = {}
    try:
        gt = _geotips_detail(obs)
    except Exception:
        gt = {}
    try:
        meta = _extras_scores(extras)
    except Exception:
        meta = {}

    union = set(direct) | set(gt) | set(meta)
    if not union:
        return {}

    max_d = max((d["raw"] for d in direct.values()), default=0.0)
    max_d = max_d if max_d > 0.0 else 1.0
    max_g = max((g["score"] for g in gt.values()), default=0.0)
    max_g = max_g if max_g > 0.0 else 1.0

    scored: dict[str, dict] = {}
    for cc in union:
        ncc = _norm_cc(cc)
        if not ncc:
            continue
        d_raw = max(0.0, direct.get(cc, {}).get("raw", 0.0))
        g_raw = max(0.0, gt.get(ncc, gt.get(cc, {})).get("score", 0.0))
        d_n = d_raw / max_d
        g_n = g_raw / max_g
        m_n = _clamp01(meta.get(cc, 0.0) + meta.get(ncc, 0.0)
                       if ncc != cc else meta.get(cc, 0.0))
        combined = d_n + g_n + m_n
        if combined <= 0.0:
            continue
        hits: list[str] = list(direct.get(cc, {}).get("hits") or [])
        for cue in gt.get(ncc, gt.get(cc, {})).get("cues") or []:
            desc = f"geotips {cue}"
            if desc not in hits:
                hits.append(desc)
        scored[ncc] = {
            "_raw": combined,
            "hits": hits,
            "sources": {
                "geotips": round(g_n, 3),
                "direct": round(d_n, 3),
                "meta": round(m_n, 3),
            },
        }

    max_c = max((e["_raw"] for e in scored.values()), default=0.0)
    if max_c <= 0.0:
        return {}
    out: dict[str, dict] = {}
    for cc, e in sorted(scored.items(), key=lambda kv: (-kv[1]["_raw"], kv[0])):
        score = round(e["_raw"] / max_c, 3)
        if score <= 0.0:
            continue
        out[cc] = {
            "score": score,
            "hits": e["hits"],
            "sources": e["sources"],
        }
    return out


def top(matrix, n: int = 8) -> list[dict]:
    """Best-first [{cc, score, hits_n}] — the head of the pigeon table."""
    try:
        items = [
            (cc, e) for cc, e in (matrix or {}).items()
            if isinstance(e, dict)
        ]
        items.sort(key=lambda kv: (-float(kv[1].get("score", 0.0)), kv[0]))
        return [
            {
                "cc": cc,
                "score": float(e.get("score", 0.0)),
                "hits_n": len(e.get("hits") or []),
            }
            for cc, e in items[: max(0, int(n))]
        ]
    except Exception:
        return []


def claimed_rank(matrix, cc) -> dict:
    """Where a claimed country sits: {rank, score, of}. rank is 1-based
    (1 = table leader); rank is None and score 0.0 when cc isn't in the
    matrix at all."""
    tgt = _norm_cc(cc)
    try:
        items = [
            (c, e) for c, e in (matrix or {}).items()
            if isinstance(e, dict)
        ]
        items.sort(key=lambda kv: (-float(kv[1].get("score", 0.0)), kv[0]))
    except Exception:
        items = []
    rank = next((i + 1 for i, (c, _) in enumerate(items) if c == tgt), None)
    score = 0.0
    try:
        entry = (matrix or {}).get(tgt)
        if isinstance(entry, dict):
            score = float(entry.get("score", 0.0))
    except Exception:
        score = 0.0
    return {"rank": rank, "score": round(score, 3), "of": len(items)}


def explain(matrix, cc) -> str:
    """One-line human readout: 'DE: 6 cue hits — eu plate, latin script,
    drive right, yellow lines…' (… when the list is truncated at 8)."""
    tgt = _norm_cc(cc) or str(cc or "?")
    entry = (matrix or {}).get(tgt) if isinstance(matrix, dict) else None
    hits = list((entry or {}).get("hits") or [])
    if not hits:
        return f"{tgt}: no cue hits"
    shown = ", ".join(str(h) for h in hits[:8])
    tail = "…" if len(hits) > 8 else ""
    noun = "hit" if len(hits) == 1 else "hits"
    return f"{tgt}: {len(hits)} cue {noun} — {shown}{tail}"


def matrix_evidence(matrix, claimed_cc) -> "Evidence":
    """Fold the pigeon table into one claim-level Evidence card.

    Verdicts (confidence capped at 0.5 — cue fusion corroborates):
      * claimed cc ranks in the top-3                    -> SUPPORTS
      * claimed score < 40% of the top score AND the top
        country has >=5 cue hits                          -> CONTRADICTS
      * anything else (no claim, empty matrix, mid-table) -> INCONCLUSIVE

    data carries the top-5 (cc, score, hits_n, hits[:6], sources) plus
    the claimed cc's 1-based rank and score.
    """
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # direct `python3 sundial/geo_tables.py` run
        try:
            from schema import ClaimVerdict, Evidence
        except ImportError:
            return None

    try:
        m = matrix if isinstance(matrix, dict) else {}
        items = [
            (c, e) for c, e in m.items() if isinstance(e, dict)
        ]
        items.sort(key=lambda kv: (-float(kv[1].get("score", 0.0)), kv[0]))
    except Exception:
        items = []

    cc = _norm_cc(claimed_cc)
    rank = next((i + 1 for i, (c, _) in enumerate(items) if c == cc), None)
    entry = m.get(cc) if isinstance(m.get(cc), dict) else {}
    claimed_score = float(entry.get("score", 0.0) or 0.0)
    claimed_hits = len(entry.get("hits") or [])

    top5 = [
        {
            "cc": c,
            "score": float(e.get("score", 0.0)),
            "hits_n": len(e.get("hits") or []),
            "hits": (e.get("hits") or [])[:6],
            "sources": e.get("sources") or {},
        }
        for c, e in items[:5]
    ]
    data = {
        "claimed_cc": cc or None,
        "claimed_rank": rank,
        "claimed_score": round(claimed_score, 3),
        "claimed_hits": claimed_hits,
        "top": top5,
        "n_scored": len(items),
    }
    cap = lambda c: round(min(0.5, c), 2)

    if not items:
        return Evidence(
            type="geo_matrix",
            summary="geo cue matrix empty (nothing observed or no source "
                    "produced a score)",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data=data,
        )

    top_cc, top_e = items[0]
    top_score = float(top_e.get("score", 0.0) or 0.0)
    top_hits = len(top_e.get("hits") or [])
    lead = ", ".join(f"{c}={e.get('score', 0.0)}" for c, e in items[:3])

    if cc and rank is not None and rank <= 3:
        return Evidence(
            type="geo_matrix",
            summary=(
                f"geo cue matrix ranks {cc} #{rank} of {len(items)} "
                f"({claimed_hits} cue hits: "
                f"{', '.join((entry.get('hits') or [])[:6]) or 'prior-only'}); "
                f"top: {lead}"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=cap(0.30 + 0.04 * max(1, claimed_hits)),
            data=data,
        )

    if cc and top_score > 0.0 and top_hits >= 5 \
            and claimed_score < 0.4 * top_score:
        return Evidence(
            type="geo_matrix",
            summary=(
                f"geo cue matrix favours {top_cc} ({top_hits} cue hits) "
                f"over claimed {cc} (score {round(claimed_score, 3)} vs "
                f"{top_score}); top: {lead}"
            ),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=cap(0.30 + 0.03 * top_hits),
            data=data,
        )

    return Evidence(
        type="geo_matrix",
        summary=(
            f"geo cue matrix: {lead}"
            + (f"; claimed {cc} scored {round(claimed_score, 3)} "
               f"(rank {rank})" if cc else "")
        ),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=cap(0.15),
        data=data,
    )


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/geo_tables.py  (offline — geotips reads the
# local CSV when importable; every check also passes without it)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ok = True

    def check(name, cond):
        global ok
        ok &= bool(cond)
        print(f"  {name}: {'ok' if cond else 'FAIL'}")

    # --- DE-dominant observation set -------------------------------------
    obs = {
        "driving_side": "right",
        "plate_style": "eu",
        "sign_language": "de",
        "languages_scripts": ["german", "latin"],
        "domain": ".de",
        "chevrons": "red on white",
        "bollards": "white with black reflector",
        "lane_markings_color": "white",
        "sign_conventions": ["vienna"],
    }
    vec = {"geocode_ccs": ["DE"], "sign_conventions": ["vienna"]}
    extras = {
        "clip_prior": {"DE": 0.9, "FR": 0.15},
        "hier": {"DE": 0.8, "AT": 0.3},
    }
    mx = build_matrix(vec, observations=obs, extras=extras)
    check("matrix nonempty", bool(mx))
    check("all scores 0..1",
          all(0.0 < e["score"] <= 1.0 for e in mx.values()))
    check("sources triple",
          all(set(e["sources"]) == {"geotips", "direct", "meta"}
              for e in mx.values()))

    t = top(mx)
    print("top:", [(d["cc"], d["score"], d["hits_n"]) for d in t[:6]])
    check("DE top", t and t[0]["cc"] == "DE")
    check("top score is 1.0", t and t[0]["score"] == 1.0)
    check("hits_n counts hits",
          t and t[0]["hits_n"] == len(mx["DE"]["hits"]))

    cr_de = claimed_rank(mx, "DE")
    print("DE:", cr_de, "| FR:", claimed_rank(mx, "FR"))
    check("DE rank 1", cr_de["rank"] == 1 and cr_de["of"] == len(mx))
    cr_fr = claimed_rank(mx, "FR")
    check("FR mid-table", cr_fr["rank"] is None or cr_fr["rank"] > 1)
    check("unknown cc unranked",
          claimed_rank(mx, "XX")["rank"] is None)

    # --- explain -----------------------------------------------------------
    line = explain(mx, "DE")
    print(line)
    check("explain format",
          line.startswith("DE:") and "cue hit" in line)
    check("explain unknown",
          explain(mx, "XX").endswith("no cue hits"))

    # --- verdicts ----------------------------------------------------------
    ev_de = matrix_evidence(mx, "DE")
    print(f"  evidence[DE] {ev_de.verdict.value} conf={ev_de.confidence}: "
          f"{ev_de.summary}")
    check("DE supports", ev_de.verdict.value == "supports")
    check("conf capped 0.5", ev_de.confidence <= 0.5)

    ev_fr = matrix_evidence(mx, "FR")
    print(f"  evidence[FR] {ev_fr.verdict.value} conf={ev_fr.confidence}: "
          f"{ev_fr.summary}")
    check("FR contradicts (score <40% of top, top >=5 hits)",
          ev_fr.verdict.value == "contradicts"
          and mx["FR"]["score"] < 0.4 * t[0]["score"])
    check("top has >=5 hits", t[0]["hits_n"] >= 5)

    ev_none = matrix_evidence(mx, None)
    check("no claim inconclusive",
          ev_none.verdict.value == "inconclusive")
    check("top5 in data", len(ev_none.data.get("top", [])) == 5)

    # --- empty / junk -------------------------------------------------------
    mx0 = build_matrix(None)
    check("empty matrix {}", mx0 == {})
    ev0 = matrix_evidence({}, "DE")
    print(f"  evidence[empty] {ev0.verdict.value} conf={ev0.confidence}")
    check("empty inconclusive", ev0.verdict.value == "inconclusive"
          and ev0.confidence == 0.0)
    check("junk tolerated",
          top({"X": "not-a-dict"}, 5) == [] and build_matrix("junk") == {})

    sys.exit(0 if ok else 1)
