"""GeoTips stage — score observed scene cues against the GeoGuessr meta CSV.

The community GeoGuessr meta (sundial/data/geotips.csv, ~120 countries)
catalogues the cheap physical tells per country: bollard shapes, licence-
plate colours, chevron direction, utility-pole construction, lane-line
colours, sign furniture, Google-car generation and architecture/landscape
notes. This module tokenises each column's comma-separated snake tokens
(e.g. ``"white_front,yellow_rear"``) and scores an observed feature
vector (the osint_features.feature_vector keys, plus merged observation
keys) against every row: each (obs_key, column) pair with >=1 token
overlap contributes that column's weight once.

Match semantics: a CSV token hits an observed token on exact equality,
on underscore-boundary substring in EITHER direction ("yellow" ~
"yellow_rear"), or on a word-set fuzzy overlap (shared content words
covering >=50% of the smaller set, min 2). Country rows fold to ISO-ish
country codes via a name table, then the row's ccTLD Domain cell, then a
normalized-name pseudo-code. Rows sharing a code (e.g. Alaska/Hawaii/
US -> US) keep the best row's score.

Pure stdlib (csv at import time); the schema import is lazy so the module
loads with zero project deps, matching veg_geo.py / plate_style.py
conventions. Never raises: a missing or unreadable CSV yields empty
scores and INCONCLUSIVE evidence. Stage API:

    scores = tip_scores(observed)            # {"GB": 2.8, "HU": 1.9, ...}
    ev     = tip_evidence(observed, "GB")    # Evidence card
"""
from __future__ import annotations

import csv
import os
import re

_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "geotips.csv")

# ---------------------------------------------------------------------------
# OBS_MAP — our feature-vector keys -> the CSV columns they inform.
# One key may feed several columns (sign colours describe both direction and
# road signs; camera artifacts cover the Google car and any follow car).
# ---------------------------------------------------------------------------
OBS_MAP: dict[str, tuple[str, ...]] = {
    "plate_style":         ("License Plate",),
    "bollards":            ("Bollard",),
    "chevrons":            ("Chevron",),
    "utility_poles":       ("Utility Pole",),
    "guardrails":          ("Guardrail",),
    "driving_side":        ("Driving Side",),
    "lane_markings_color": ("Lanes",),
    "sign_colors":         ("Direction Sign", "Road Sign"),
    "sign_color":          ("Direction Sign", "Road Sign"),
    "camera_artifacts":    ("Google Car", "Follow Car"),
    "architecture":        ("Architecture",),
    "vegetation":          ("Landscape",),
    "terrain":             ("Landscape",),
    "sign_language":       ("Language",),
    "languages_scripts":   ("Language",),
    "domain":              ("Domain",),
    # second-tier columns — accepted when a caller has them, default weight
    "curbs":               ("Curb",),
    "stop_sign":           ("Stop Sign",),
    "pedestrian_sign":     ("Pedestrian Sign",),
    "road_number":         ("Road Number",),
    "sign_post":           ("Sign/Traffic Post",),
    "camera_generation":   ("Camera Generation",),
    "street_scene":        ("Street",),
    "rift":                ("Rift",),
    "misc":                ("Miscellaneous",),
}

# Per-column scoring weights (columns not listed get _DEFAULT_W).
# Google Car is the single most diagnostic meta tell; bollards are the
# classic country fingerprint; landscape is deliberately weak (biomes are
# shared across whole continents).
_COL_W: dict[str, float] = {
    "Google Car":     1.2,
    "Bollard":        1.0,
    "License Plate":  0.9,
    "Driving Side":   0.9,
    "Chevron":        0.8,
    "Utility Pole":   0.7,
    "Guardrail":      0.7,
    "Lanes":          0.6,
    "Direction Sign": 0.6,
    "Road Sign":      0.6,
    "Landscape":      0.4,
}
_DEFAULT_W = 0.5

# ---------------------------------------------------------------------------
# Country-name -> ISO-ish country code (main GeoGuessr roster + territories
# whose names hide their codes). Lookup happens on the normalized name
# (lowercase, non-alnum -> space); anything unlisted falls back to the
# row's ccTLD Domain cell, then a pseudo-code from the name itself.
# ---------------------------------------------------------------------------
_NAME_CC: dict[str, str] = {
    "united kingdom": "GB", "great britain": "GB", "uk": "GB",
    "united states": "US", "usa": "US", "alaska": "US", "hawaii": "US",
    "midway atoll": "US", "puerto rico": "PR",
    "russia": "RU", "south korea": "KR", "north macedonia": "MK",
    "czechia": "CZ", "czech republic": "CZ", "turkey": "TR",
    "eswatini": "SZ", "south africa": "ZA", "new zealand": "NZ",
    "sri lanka": "LK", "united arab emirates": "AE",
    "dominican republic": "DO", "costa rica": "CR", "san marino": "SM",
    "sao tome principe": "ST", "reunion": "RE", "hong kong": "HK",
    "macau": "MO", "isle of man": "IM", "faroe islands": "FO",
    "svalbard": "SJ", "saint pierre and miquelon": "PM",
    "cocos islands": "CC", "christmas island": "CX",
    "northern mariana islands": "MP", "pitcairn islands": "PN",
    "american samoa": "AS", "curacao": "CW",
    "u s virgin islands": "VI", "united states virgin islands": "VI",
    "bosnia": "BA", "cape verde": "CV", "east timor": "TL",
    "antigua": "AG", "trinidad": "TT", "saint lucia": "LC",
    "burkina faso": "BF", "sierra leone": "SL", "ivory coast": "CI",
    "guinea bissau": "GW", "equatorial guinea": "GQ",
    "western sahara": "EH", "el salvador": "SV",
}
# ccTLDs that differ from the ISO code we want.
_DOMAIN_ALIAS = {"UK": "GB", "EL": "GR", "SU": "RU", "TP": "TL"}

# ISO 639-1 -> English language name, so sign_language "fr" hits a cell
# reading "french". Only applied to keys feeding the Language column.
_LANG_ISO: dict[str, str] = {
    "en": "english", "fr": "french", "de": "german", "es": "spanish",
    "pt": "portuguese", "it": "italian", "nl": "dutch", "ja": "japanese",
    "ko": "korean", "zh": "chinese", "ar": "arabic", "ru": "russian",
    "th": "thai", "tr": "turkish", "el": "greek", "he": "hebrew",
    "iw": "hebrew", "pl": "polish", "uk": "ukrainian", "sv": "swedish",
    "da": "danish", "no": "norwegian", "nb": "norwegian", "fi": "finnish",
    "cs": "czech", "sk": "slovak", "hu": "hungarian", "ro": "romanian",
    "bg": "bulgarian", "hr": "croatian", "sr": "serbian", "sl": "slovenian",
    "et": "estonian", "lv": "latvian", "lt": "lithuanian", "mk": "macedonian",
    "mt": "maltese", "is": "icelandic", "sq": "albanian", "ka": "georgian",
    "hy": "armenian", "vi": "vietnamese", "id": "indonesian", "ms": "malay",
    "hi": "hindi", "bn": "bengali", "ur": "urdu", "fa": "persian",
    "sw": "swahili", "af": "afrikaans", "km": "khmer", "lo": "lao",
    "my": "burmese", "ne": "nepali", "si": "sinhala", "mn": "mongolian",
    "kk": "kazakh", "ky": "kyrgyz", "uz": "uzbek", "az": "azerbaijani",
    "tl": "filipino", "fil": "filipino", "am": "amharic", "ca": "catalan",
    "eu": "basque", "gl": "galician", "be": "belarusian", "bs": "bosnian",
    "lb": "luxembourgish", "fo": "faroese", "kl": "greenlandic",
    "ta": "tamil", "te": "telugu", "ml": "malayalam", "kn": "kannada",
    "pa": "punjabi", "gu": "gujarati", "mr": "marathi",
}
# Our plate-style vocabulary (match_score._PLATE_HINTS keys) -> the words
# the CSV actually uses for those plate families.
_PLATE_ALIAS: dict[str, str] = {
    "eu": "left_blue_strip european", "european": "left_blue_strip european",
    "uk": "white_front yellow_rear", "gb": "white_front yellow_rear",
}

# Content-word stoplist for the fuzzy match: connectives plus column-generic
# nouns that carry no discriminating power ("bollard" matches half the
# Bollard column otherwise).
_STOP = {
    "and", "or", "the", "of", "with", "on", "in", "at", "a", "an", "to",
    "for", "by", "de", "la", "bollard", "bollards", "pole", "poles",
    "sign", "signs", "plate", "plates", "road", "roads", "marker",
    "markers", "car", "street", "striped",
}
_WORD_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Loading / normalization
# ---------------------------------------------------------------------------

_CACHE: list[dict] | None = None


def _norm_tok(s) -> str:
    """'Yellow Rear!' -> 'yellow_rear'; ''/junk -> ''."""
    return _WS_RE.sub("_", str(s or "").strip().lower()).strip("_")


def _words(tok: str) -> frozenset:
    """Content words of a normalized token, minus the stoplist."""
    return frozenset(
        w for w in _WORD_RE.findall(tok) if len(w) > 1 and w not in _STOP
    )


def _split_tokens(cell: str) -> list[tuple[str, frozenset]]:
    """One CSV cell -> [(normalized token, content-word set)]."""
    out = []
    for raw in str(cell or "").split(","):
        t = _norm_tok(raw)
        if t:
            out.append((t, _words(t)))
    return out


def _norm_name(s: str) -> str:
    """Name-table key: 'U.S. Virgin Islands' -> 'u s virgin islands'."""
    return re.sub(r"\s+", " ", _WS_RE.sub(" ", str(s or "").lower())).strip()


def _cc_from_row(name: str, domain: str) -> str:
    """Resolve a row's country code: name table -> ccTLD -> pseudo-code."""
    key = _norm_name(name)
    if key in _NAME_CC:
        return _NAME_CC[key]
    m = re.search(r"\.([a-zA-Z]{2})\b", str(domain or ""))
    if m:
        cc = m.group(1).upper()
        return _DOMAIN_ALIAS.get(cc, cc)
    words = [w for w in key.split() if w]
    if not words:
        return ""
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[-1][0]).upper()


def load() -> list[dict]:
    """Read + tokenize the GeoTips CSV once; returns cached row dicts.

    Each row: {"country", "cc", "continent", "domain", "tokens"} where
    tokens maps CSV column -> [(norm_token, word_set)]. Missing/unreadable
    files and malformed rows never raise — they just yield fewer rows.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    rows: list[dict] = []
    try:
        # errors='replace': a handful of cells hold non-UTF8 bytes (mangled
        # Cyrillic/Korean notes); they become U+FFFD and tokenize away.
        with open(_CSV_PATH, newline="", encoding="utf-8",
                  errors="replace") as fh:
            for raw in csv.DictReader(fh):
                if not isinstance(raw, dict):
                    continue
                name = (raw.get("Country") or "").strip()
                if not name:
                    continue
                tokens = {
                    col: _split_tokens(raw.get(col) or "")
                    for col in raw
                    if col and col != "Country"
                }
                rows.append({
                    "country": name,
                    "cc": _cc_from_row(name, raw.get("Domain") or ""),
                    "continent": (raw.get("Continent") or "").strip(),
                    "domain": (raw.get("Domain") or "").strip(),
                    "tokens": tokens,
                })
    except Exception:
        rows = []
    _CACHE = rows
    return _CACHE


# ---------------------------------------------------------------------------
# Observed-side tokenization
# ---------------------------------------------------------------------------

def _obs_tokens(value, obs_key: str) -> list[tuple[str, frozenset]]:
    """An observed value (str | iterable | dict) -> [(token, word_set)].

    Strings split on commas/semicolons/slashes so "white front, yellow
    rear" is two tokens; a bare phrase stays one token whose word set the
    fuzzy matcher then uses. Values may be expanded by per-key alias
    tables (plate_style 'eu' -> 'left_blue_strip european', sign_language
    ISO codes -> language names).
    """
    items: list[str] = []
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, str):
        parts = re.split(r"[,;/]+", value)
    else:
        try:
            parts = list(value or [])
        except TypeError:
            parts = [value]
    for item in parts:
        for piece in re.split(r"[,;/]+", str(item or "")):
            t = _norm_tok(piece)
            if not t:
                continue
            items.append(t)
            alias = None
            if obs_key == "plate_style":
                alias = _PLATE_ALIAS.get(t)
            elif obs_key in ("sign_language", "languages_scripts"):
                alias = _LANG_ISO.get(t)
            if alias:
                items.append(_norm_tok(alias))
    seen: set[str] = set()
    out = []
    for t in items:
        if t and t not in seen:
            seen.add(t)
            out.append((t, _words(t)))
    return out


# ---------------------------------------------------------------------------
# Matching + scoring
# ---------------------------------------------------------------------------

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


def _tok_match(ot: str, ct: str, ow: frozenset, cw: frozenset) -> bool:
    """One observed token vs one CSV token — fuzzy both directions."""
    if not ot or not ct:
        return False
    if ot == ct:
        return True
    if _boundary_sub(ot, ct) or _boundary_sub(ct, ot):
        return True
    if ow and cw:
        shared = ow & cw
        if not shared:
            return False
        if ow <= cw or cw <= ow:
            return True
        return len(shared) >= max(2, 0.5 * min(len(ow), len(cw)))
    return False


def _col_hits(obs_toks, cell_toks) -> list[str]:
    """Observed tokens matching at least one CSV token in the cell."""
    hits = []
    for ot, ow in obs_toks:
        if any(_tok_match(ot, ct, ow, cw) for ct, cw in cell_toks):
            hits.append(ot)
    return hits


def _score_detail(observed: dict) -> dict[str, dict]:
    """Per-cc breakdown: {cc: {"score", "hits", "cues", "country", "continent"}}.

    `hits` counts (obs_key, column) pairs that matched — "cue hits" in the
    evidence rules; `cues` names the obs keys that fired, `matched` keeps
    example token hits per cue for explainability.
    """
    obs = {
        key: _obs_tokens((observed or {}).get(key), key)
        for key in OBS_MAP
    }
    best: dict[str, dict] = {}
    for row in load():
        score = 0.0
        hits = 0
        cues: list[str] = []
        matched: dict[str, list[str]] = {}
        for key, columns in OBS_MAP.items():
            obs_toks = obs.get(key) or []
            if not obs_toks:
                continue
            for col in columns:
                cell_toks = row["tokens"].get(col) or []
                tok_hits = _col_hits(obs_toks, cell_toks)
                if tok_hits:
                    score += _COL_W.get(col, _DEFAULT_W)
                    hits += 1
                    if key not in cues:
                        cues.append(key)
                    matched.setdefault(key, []).extend(tok_hits[:3])
        if score <= 0.0 or not row["cc"]:
            continue
        cur = {
            "score": round(score, 3),
            "hits": hits,
            "cues": cues,
            "matched": {k: v[:4] for k, v in matched.items()},
            "country": row["country"],
            "continent": row["continent"],
        }
        prev = best.get(row["cc"])
        if (prev is None or cur["score"] > prev["score"]
                or (cur["score"] == prev["score"]
                    and cur["country"] < prev["country"])):
            best[row["cc"]] = cur
    return best


def tip_scores(observed: dict) -> dict:
    """{cc: weighted token-overlap score}, best-first. Empty on no data."""
    try:
        detail = _score_detail(observed)
    except Exception:
        return {}
    ordered = sorted(
        detail.items(), key=lambda kv: (-kv[1]["score"], kv[0])
    )
    return {cc: d["score"] for cc, d in ordered}


# ---------------------------------------------------------------------------
# Evidence card
# ---------------------------------------------------------------------------

_CC_ALIASES = {"UK": "GB", "EL": "GR", "USA": "US", "UAE": "AE"}


def _norm_cc(value) -> str:
    cc = (value or "").strip().upper()
    return _CC_ALIASES.get(cc, cc)


def tip_evidence(observed: dict, claimed_cc: str | None) -> "Evidence":
    """Fold GeoTips scores into one claim-level Evidence card.

    Verdicts (confidence capped at 0.5 — cue fusion corroborates):
      * claimed cc in the top-3 with >=2 cue hits          -> SUPPORTS
      * claimed score < 40% of the top score AND the top
        country matched >=4 distinct cue columns           -> CONTRADICTS
      * anything else (incl. no claim / no data)           -> INCONCLUSIVE

    data carries the top-5 [{cc,country,score,hits,cues,matched}] plus the
    claimed cc's rank/score and the cue names that fired for the leader.
    """
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # direct `python3 sundial/geotips.py` run
        from schema import ClaimVerdict, Evidence

    try:
        detail = _score_detail(observed)
    except Exception:
        detail = {}
    ranked = sorted(
        detail.items(), key=lambda kv: (-kv[1]["score"], kv[0])
    )
    cc = _norm_cc(claimed_cc)
    claimed_rank = next(
        (i for i, (c, _) in enumerate(ranked) if c == cc), None
    )
    top5 = [
        {
            "cc": c,
            "country": d["country"],
            "score": d["score"],
            "hits": d["hits"],
            "cues": d["cues"],
            "matched": d["matched"],
        }
        for c, d in ranked[:5]
    ]
    data = {
        "observed": {
            k: v for k, v in (observed or {}).items() if k in OBS_MAP and v
        },
        "claimed_cc": cc or None,
        "claimed_rank": claimed_rank,
        "claimed_score": detail[cc]["score"] if cc in detail else 0.0,
        "claimed_cues": detail[cc]["cues"] if cc in detail else [],
        "top": top5,
        "matched_cues": top5[0]["cues"] if top5 else [],
        "n_scored": len(ranked),
    }
    cap = lambda c: round(min(0.5, c), 2)

    if not ranked:
        return Evidence(
            type="geotips",
            summary="no GeoTips cue overlap (no data or nothing observed)",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data=data,
        )

    top_cc, top = ranked[0]
    lead = ", ".join(f"{c}={d['score']}" for c, d in ranked[:3])
    if cc and claimed_rank is not None and claimed_rank < 3 \
            and detail[cc]["hits"] >= 2:
        d = detail[cc]
        return Evidence(
            type="geotips",
            summary=(
                f"GeoTips cues rank {cc} (#{claimed_rank + 1} of "
                f"{len(ranked)}) — {d['hits']} cue columns matched "
                f"({', '.join(d['cues'][:6])}); top: {lead}"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=cap(0.30 + 0.05 * d["hits"]),
            data=data,
        )

    claimed_score = detail[cc]["score"] if cc in detail else 0.0
    if cc and top["hits"] >= 4 \
            and claimed_score < 0.4 * top["score"]:
        return Evidence(
            type="geotips",
            summary=(
                f"GeoTips cues favour {top_cc} ({top['country']}, "
                f"{top['hits']} cue columns: {', '.join(top['cues'][:6])}) "
                f"over claimed {cc} (score {claimed_score} vs "
                f"{top['score']})"
            ),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=cap(0.30 + 0.03 * top["hits"]),
            data=data,
        )

    return Evidence(
        type="geotips",
        summary=(
            f"GeoTips cue ranking: {lead}"
            + (f"; claimed {cc} scored {claimed_score}" if cc else "")
        ),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=cap(0.2),
        data=data,
    )


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/geotips.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ok = True

    def check(name, cond):
        global ok
        ok &= bool(cond)
        print(f"  {name}: {'ok' if cond else 'FAIL'}")

    rows = load()
    print(f"loaded {len(rows)} rows")
    check("rows loaded", len(rows) >= 100)
    check("GB resolves", any(r["country"] == "United Kingdom"
                             and r["cc"] == "GB" for r in rows))

    obs = {
        "bollards": "white wedge red rectangle",
        "driving_side": "left",
        "plate_style": "yellow_rear",
    }
    scores = tip_scores(obs)
    top = list(scores.items())[:6]
    print("top:", top)
    check("scores nonempty", bool(scores))
    check("GB or HU-style surfaces",
          any(c in ("GB", "HU", "IM", "JE", "AU", "BW") for c, _ in top[:3]))

    ev = tip_evidence(obs, "GB")
    print(f"  evidence[{ev.type}] {ev.verdict.value} "
          f"conf={ev.confidence}: {ev.summary}")
    check("GB supports", ev.verdict.value == "supports")
    check("conf <=0.5", ev.confidence <= 0.5)

    ev2 = tip_evidence({}, None)
    print(f"  empty -> {ev2.verdict.value} conf={ev2.confidence}")
    check("empty inconclusive", ev2.verdict.value == "inconclusive"
          and ev2.confidence == 0.0)

    ev3 = tip_evidence(obs, None)
    check("no claim inconclusive", ev3.verdict.value == "inconclusive")
    check("top5 in data", len(ev3.data.get("top", [])) == 5)

    # no CSV -> empty scores, still no raise
    old = _CSV_PATH
    globals()["_CSV_PATH"] = "/nonexistent/geotips.csv"
    globals()["_CACHE"] = None
    check("missing csv -> {}", tip_scores(obs) == {})
    globals()["_CSV_PATH"] = old
    globals()["_CACHE"] = None

    sys.exit(0 if ok else 1)
