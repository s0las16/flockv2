"""OSINT elimination-chain — the GeoGuessr-pro narration panel.

Sundial's OSINT stages each shave the candidate world a little: a left-
hand drive observation kills most of Europe, yellow rear plates kill all
but GB/NL/IL, a dial code names a single country. This module renders
that process as an ordered story for the report — "what we saw -> what
it rules out -> what remains" — closed by a ranking line.

``build_chain`` returns ordered step dicts::

    {"kind": "obs"|"elim"|"rank", "text": str, "weight": float}

    obs   — one observed cue and what it implies
            ("driving side: left — keeps GB,IE,JP,AU…; eliminates most
            of Europe & the Americas")
    elim  — a knock-out: the cue's surviving set is small enough to name
            ("yellow rear plates — NL/GB/IL only")
    rank  — the closing line, top-3 candidates after all cues
            ("after 9 cues: DE 0.65 > FR 0.31 > NL 0.22")

``chain_html`` renders a chain as an escaped ``<ol>`` fragment (inline
styles, report palette) capped at 14 rows; ``chain_evidence`` packs the
chain into a narration-only Evidence card — always INCONCLUSIVE 0.3,
because the chain explains other evidence rather than convicting itself.

Cue -> elimination knowledge lives in the small built-in tables below —
deliberately a *digest* of the scoring tables in match_score.py /
osint.py (narration needs headlines, not the full country lists).

Pure stdlib at import time; schema types are imported lazily. The module
never raises: malformed cues or ranked rows are skipped, and an empty
world yields an empty chain.
"""
from __future__ import annotations

import html
import re

# --------------------------------------------------------------------------
# Small cue -> elimination tables (digests of match_score.py / osint.py).
# --------------------------------------------------------------------------

# Headline kept-set for a left-traffic observation (full set lives in
# match_score._LEFT_TRAFFIC — narration only needs the recognizable few).
_LEFT_SAMPLE = ("GB", "IE", "JP", "AU", "NZ", "IN", "ZA", "TH", "SG", "MY")

# EU bloc for eu-blue-band plates (mirrors osint._EU_CCS).
_EU_CCS = (
    "DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "PL", "IE", "SE",
    "DK", "FI", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "EE",
    "LV", "LT", "LU", "MT", "CY",
)

# Yellow-centerline countries (match_score._YELLOW_CENTERLINE).
_YELLOW_CENTERLINE = ("US", "CA", "BR", "CN", "KR", "TW", "NO", "SE", "FI")

# plate_style value -> (kept ccs, narration label). Synonyms normalized
# through _norm_key; yellow rear is the classic GB/NL/IL tell.
_PLATE_HINTS = {
    "uk": (("GB",), "uk-style plates"),
    "gb": (("GB",), "uk-style plates"),
    "uk_white_yellow": (("GB",), "uk-style plates"),
    "yellow_rear": (("NL", "GB", "IL"), "yellow rear plates"),
    "eu": (_EU_CCS, "eu blue-band plates"),
    "europe": (_EU_CCS, "eu blue-band plates"),
    "eu_blue_band": (_EU_CCS, "eu blue-band plates"),
    "us": (("US", "CA"), "us-style plates"),
    "usa": (("US",), "us-style plates"),
    "ca": (("CA",), "ca-style plates"),
    "au": (("AU",), "au-style plates"),
    "jp": (("JP",), "jp-style plates"),
    "cn": (("CN",), "cn-style plates"),
    "kr": (("KR",), "kr-style plates"),
    "ru": (("RU",), "ru-style plates"),
    "br": (("BR",), "br-style plates"),
    "in": (("IN",), "in-style plates"),
    "za": (("ZA",), "za-style plates"),
}

# sign_language token (ISO 639-1 or english name) -> (kept ccs, lang name).
# "latin"/"roman" script is deliberately absent — it implies nothing.
_LANG_HINTS = {
    "en": (("GB", "IE", "US", "AU", "NZ", "CA", "ZA", "IN", "SG"), "english"),
    "fr": (("FR", "BE", "CH", "LU", "MC", "CA", "SN", "CI"), "french"),
    "de": (("DE", "AT", "CH", "LU", "LI"), "german"),
    "es": (("ES", "MX", "AR", "CO", "CL", "PE", "VE", "UY"), "spanish"),
    "pt": (("PT", "BR", "AO", "MZ"), "portuguese"),
    "it": (("IT", "CH", "SM", "VA"), "italian"),
    "nl": (("NL", "BE", "SR"), "dutch"),
    "ja": (("JP",), "japanese"),
    "ko": (("KR",), "korean"),
    "zh": (("CN", "TW", "HK", "MO", "SG"), "chinese"),
    "ar": (("SA", "AE", "EG", "QA", "JO", "LB", "MA", "DZ"), "arabic"),
    "ru": (("RU", "BY", "KZ", "KG"), "russian"),
    "hi": (("IN",), "hindi"),
    "th": (("TH",), "thai"),
    "vi": (("VN",), "vietnamese"),
    "id": (("ID",), "indonesian"),
    "ms": (("MY", "BN"), "malay"),
    "tr": (("TR", "CY"), "turkish"),
    "el": (("GR", "CY"), "greek"),
    "pl": (("PL",), "polish"),
    "uk": (("UA",), "ukrainian"),
    "sv": (("SE",), "swedish"),
    "da": (("DK",), "danish"),
    "no": (("NO",), "norwegian"),
    "fi": (("FI",), "finnish"),
    "cs": (("CZ",), "czech"),
    "hu": (("HU",), "hungarian"),
    "ro": (("RO",), "romanian"),
    "he": (("IL",), "hebrew"),
    "fa": (("IR",), "farsi"),
    "ur": (("PK",), "urdu"),
    "bn": (("BD", "IN"), "bengali"),
}
_LANG_NAME_CODES = {name: code for code, (_, name) in _LANG_HINTS.items()}
_LANG_NAME_CODES.update({
    "mandarin": "zh", "persian": "fa", "flemish": "nl", "cantonese": "zh",
})

# Script labels (langdetect.py / languages_scripts) -> regions.
_SCRIPT_HINTS = {
    "cyrillic": (("RU", "UA", "BY", "BG", "KZ", "RS", "MN"), "cyrillic script"),
    "arabic_script": (("SA", "AE", "EG", "MA", "DZ", "JO", "IQ", "IR", "PK"),
                      "arabic script"),
    "cjk": (("JP", "CN", "TW", "HK", "MO"), "cjk script"),
    "hangul": (("KR", "KP"), "hangul script"),
    "devanagari": (("IN", "NP"), "devanagari script"),
    "tamil": (("IN", "LK", "SG"), "tamil script"),
    "georgian": (("GE",), "georgian script"),
    "armenian": (("AM",), "armenian script"),
}
_SCRIPT_ALIASES = {
    "cyrillic": "cyrillic", "arabic": "arabic_script", "arab": "arabic_script",
    "cjk": "cjk", "hangul": "hangul", "korean_script": "hangul",
    "devanagari": "devanagari", "tamil": "tamil",
    "georgian": "georgian", "armenian": "armenian",
}

# road_markings token -> (kept ccs, label)  (digest of _MARKING_HINTS).
_MARKING_HINTS = {
    "double_yellow_lines": (("GB", "IE"), "double yellow lines"),
    "double_red_lines": (("GB",), "double red lines"),
    "zigzag_lines": (("GB", "IE", "SG"), "zigzag lines"),
    "keep_left_bollards": (("GB", "IE", "AU", "NZ", "JP"),
                           "keep-left bollards"),
    "yellow_centerline": (_YELLOW_CENTERLINE, "yellow centerline"),
    "yellow_edge_line": (("US", "CA", "GB", "IE"), "yellow edge line"),
}

# sign_convention key -> (kept ccs, label); aliases normalized below.
_CONVENTION_HINTS = {
    "vienna": (("DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL", "SE",
                "NO", "DK", "FI", "CH"), "vienna-style"),
    "mutcd": (("US", "CA", "MX", "BR", "AR", "CL", "PH", "AU", "NZ"),
              "mutcd-style"),
    "uk": (("GB", "IE", "AU", "NZ", "IN", "ZA", "SG", "MY", "HK", "MT",
            "CY"), "uk-style"),
    "japan": (("JP",), "japan-style"),
    "australia": (("AU", "NZ"), "australia-style"),
    "chinese": (("CN",), "chinese-style"),
}
_CONVENTION_ALIASES = (
    (("vienna", "europe", "eu", "continental"), "vienna"),
    (("mutcd", "muctd", "america", "us"), "mutcd"),
    (("brit", "uk", "worboys", "tsrgd"), "uk"),
    (("japan", "jp", "jdm"), "japan"),
    (("australi", "aussie", "nz", "kiwi", "oceania"), "australia"),
    (("china", "cn"), "chinese"),
)

# biome keyword groups -> (kept ccs, biome label, eliminates-phrase).
_BIOME_HINTS = (
    (("palm", "coconut", "banana", "mango", "rainforest", "jungle",
      "tropical", "monsoon", "bamboo", "papaya"),
     ("TH", "MY", "SG", "ID", "BR", "IN", "PH", "VN", "KE", "NG"),
     "tropical", "eliminates boreal & temperate Europe"),
    (("olive", "cypress", "vineyard", "fig", "lavender", "maquis",
      "garrigue", "mediterranean", "stone pine"),
     ("ES", "IT", "GR", "PT", "TR", "FR", "MA", "IL", "CY"),
     "mediterranean", "eliminates boreal & tropical candidates"),
    (("hedgerow", "oak", "beech", "deciduous", "pasture", "meadow",
      "temperate", "elm", "willow"),
     ("GB", "IE", "FR", "DE", "NL", "BE", "DK", "NZ"),
     "temperate", "eliminates the tropics"),
    (("eucalyptus",),
     ("AU", "NZ", "PT", "ES", "ZA", "CL", "BR"),
     "eucalyptus country", "points hard at Australia"),
    (("snow", "spruce", "birch", "taiga", "conifer", "boreal", "tundra",
      "fjord", "glacier"),
     ("NO", "SE", "FI", "IS", "CA", "RU", "EE", "LV"),
     "boreal/cold", "eliminates the tropics & Mediterranean"),
    (("saguaro", "desert", "arid", "cactus", "succulent", "dune"),
     ("US", "MX", "MA", "DZ", "EG", "SA", "AE", "AU", "ZA", "NA", "CL"),
     "arid/desert", "eliminates wet-temperate NW Europe"),
    (("acacia", "baobab", "savanna"),
     ("KE", "TZ", "ZA", "BW", "NA", "ZW", "AU", "NG"),
     "savanna", "eliminates Europe & N. America"),
    (("rice", "paddy", "tea"),
     ("VN", "TH", "ID", "PH", "IN", "CN", "JP", "LK", "MY"),
     "paddy/tea country", "points at S/SE Asia"),
)

# A cue is a knock-out (emits an "elim" line) when its kept set fits here.
_ELIM_MAX_KEPT = 4

# Narration weights — mirror match_score's cue weights; knock-outs get
# +0.2 because a small surviving set is the punchline of the chain.
_W_HEMISPHERE = 0.25
_W_DRIVING = 0.25
_W_LANG = 0.20
_W_PLATE = 0.15
_W_INDICATOR = 0.15
_W_GEOCODE = 0.10
_W_LANE = 0.10
_W_MARKING = 0.10
_W_CONVENTION = 0.10
_W_BIOME = 0.05
_ELIM_BONUS = 0.20

_MAX_ROWS = 14  # html fragment cap

_KIND_COLOR = {"obs": "#e5e5e5", "elim": "#f87171", "rank": "#facc15"}

# Common cc aliases: what callers hand in -> ISO-ish code.
_CC_ALIASES = {"UK": "GB", "EL": "GR", "USA": "US", "UAE": "AE"}


# --------------------------------------------------------------------------
# Small helpers — every reader is defensive; nothing here raises.
# --------------------------------------------------------------------------

def _schema():
    """Lazy schema import — tolerates both package and direct-script use."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # `python sundial/osint_chain.py` direct run
        from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


def _strs(value) -> list[str]:
    """Coerce a feature value (str | iterable | None) -> lowercase strings."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    out: list[str] = []
    try:
        it = iter(items)
    except TypeError:
        it = iter([items])
    for item in it:
        t = str(item).strip().lower()
        if t:
            out.append(t)
    return out


def _first_str(value) -> str:
    toks = _strs(value)
    return toks[0] if toks else ""


def _norm_key(s: str) -> str:
    """Normalize free text to a table key ('Zig-zag lines' -> 'zigzag_lines')."""
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower()).strip("_")


def _norm_cc(cc) -> str | None:
    s = str(cc or "").strip().upper()
    if not s:
        return None
    s = _CC_ALIASES.get(s, s)
    return s if 2 <= len(s) <= 4 and s.replace("-", "").isalpha() else None


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0  # nan guard


def _ccs(kept, limit: int = 6) -> str:
    """'GB,IE,JP…' — headlined kept-set, capped."""
    kept = [c for c in kept if c]
    out = ",".join(kept[:limit])
    return out + ("…" if len(kept) > limit else "")


def _picked(src_obs: dict, fv: dict, *keys):
    """First non-empty value across observations then the feature vector."""
    for key in keys:
        for src in (src_obs, fv):
            v = src.get(key)
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            if isinstance(v, (list, tuple, set)) and not v:
                continue
            return v
    return None


def _step(kind: str, text: str, weight: float) -> dict:
    return {"kind": kind, "text": text,
            "weight": round(max(0.0, min(1.0, weight)), 3)}


# --------------------------------------------------------------------------
# Cue renderers — each returns [(kind, text, weight)]; obs always, plus an
# "elim" line when the surviving set is small enough to name (<=4 ccs).
# --------------------------------------------------------------------------

def _cue_hemisphere(value) -> list:
    hemi = _first_str(value)
    if hemi in ("northern", "north", "n"):
        return [("obs",
                 "sun-side/shadow read: northern hemisphere — keeps "
                 "N.America, Europe & N.Asia; eliminates the southern "
                 "cone (AU,NZ,ZA,AR,CL…)", _W_HEMISPHERE)]
    if hemi in ("southern", "south", "s"):
        return [("obs",
                 "sun-side/shadow read: southern hemisphere — keeps "
                 "AU,NZ,ZA,AR,CL,BR…; eliminates Europe, N.America & "
                 "N.Asia", _W_HEMISPHERE)]
    return []


def _cue_driving(value) -> list:
    side = _first_str(value)
    if side == "left":
        return [("obs",
                 f"driving side: left — keeps {_ccs(_LEFT_SAMPLE)}; "
                 "eliminates most of Europe & the Americas", _W_DRIVING)]
    if side == "right":
        return [("obs",
                 "driving side: right — eliminates GB,IE,JP,AU and the "
                 "rest of the left-traffic world", _W_DRIVING)]
    return []


def _cue_language(tokens) -> list:
    out: list = []
    seen: set = set()
    for tok in tokens:
        if tok in ("latin", "roman", "unknown"):  # uninformative scripts
            continue
        row = _LANG_HINTS.get(tok)
        if row is None and tok in _LANG_NAME_CODES:
            row = _LANG_HINTS.get(_LANG_NAME_CODES[tok])
        script_key = _SCRIPT_ALIASES.get(_norm_key(tok))
        if row is None and script_key:
            row = _SCRIPT_HINTS.get(script_key)
        if row is None:
            continue
        kept, name = row
        if name in seen:
            continue
        seen.add(name)
        if len(kept) <= _ELIM_MAX_KEPT:
            out.append(("obs", f"sign language: {tok}", _W_LANG))
            out.append(("elim", f"{name} signage — {'/'.join(kept)} only",
                        _W_LANG + _ELIM_BONUS))
        else:
            out.append(("obs",
                        f"sign language: {tok} — keeps {_ccs(kept)}; "
                        f"eliminates non-{name} signage countries", _W_LANG))
        if len(seen) >= 3:
            break
    return out


def _plate_key(value) -> str:
    """plate_style (str|dict) -> _PLATE_HINTS key."""
    if isinstance(value, dict):
        if (value.get("yellow_rear") or value.get("yellow_rear_plate")
                or value.get("yellow_rear_plates")):
            return "yellow_rear"
        if value.get("eu_blue_band"):
            return "eu"
        for k in ("style", "region", "plate_style", "value"):
            s = _first_str(value.get(k))
            if s:
                value = s
                break
        else:
            return ""
    key = _norm_key(_first_str(value))
    if "yellow" in key and "rear" in key:
        return "yellow_rear"
    if key.endswith("_rear") or "rear_yellow" in key:
        return "yellow_rear"
    if "eu" in key and ("band" in key or "blue" in key):
        return "eu"
    return key


def _cue_plate(value) -> list:
    key = _plate_key(value)
    row = _PLATE_HINTS.get(key)
    if row is None:
        return []
    kept, label = row
    if len(kept) <= _ELIM_MAX_KEPT:
        return [("obs", f"plates: {label} seen", _W_PLATE),
                ("elim", f"{label} — {'/'.join(kept)} only",
                 _W_PLATE + _ELIM_BONUS)]
    return [("obs",
             f"plates: {label} — keeps the {_ccs(kept, 4)} bloc; "
             "eliminates GB/US/JP-style plates", _W_PLATE)]


def _cue_lane_color(tokens) -> list:
    out: list = []
    for tok in tokens:
        if tok == "yellow":
            out.append(("obs",
                        f"lane markings: yellow centerlines — keeps "
                        f"{_ccs(_YELLOW_CENTERLINE)}; eliminates "
                        "white-centerline Europe", _W_LANE))
        elif tok == "white":
            out.append(("obs",
                        "lane markings: white centerlines — eliminates the "
                        f"yellow-centerline bloc ({_ccs(_YELLOW_CENTERLINE)})",
                        _W_LANE))
        if len(out) >= 2:
            break
    return out


def _cue_markings(tokens) -> list:
    out: list = []
    seen: set = set()
    for tok in tokens:
        row = _MARKING_HINTS.get(_norm_key(tok))
        if row is None:
            continue
        kept, label = row
        if label in seen:
            continue
        seen.add(label)
        if len(kept) <= _ELIM_MAX_KEPT:
            out.append(("obs", f"road markings: {label}", _W_MARKING))
            out.append(("elim", f"{label} — {'/'.join(kept)} only",
                        _W_MARKING + _ELIM_BONUS))
        else:
            out.append(("obs",
                        f"road markings: {label} — keeps {_ccs(kept)}",
                        _W_MARKING))
        if len(seen) >= 3:
            break
    return out


def _convention_key(tok: str) -> str | None:
    """'uk-style'/'vienna-style'/'MUTCD' -> _CONVENTION_HINTS key."""
    t = str(tok or "").strip().lower()
    if t in _CONVENTION_HINTS:
        return t
    words = [w for w in re.split(r"[^a-z]+", t) if w]
    for aliases, key in _CONVENTION_ALIASES:
        for w in words:
            if any(w == a or (len(a) >= 3 and w.startswith(a))
                   for a in aliases):
                return key
    return None


def _cue_conventions(tokens) -> list:
    out: list = []
    seen: set = set()
    for tok in tokens:
        key = _convention_key(tok)
        if key is None or key in seen:
            continue
        seen.add(key)
        kept, label = _CONVENTION_HINTS[key]
        if len(kept) <= _ELIM_MAX_KEPT:
            out.append(("obs", f"sign convention: {label}", _W_CONVENTION))
            out.append(("elim", f"{label} signage — {'/'.join(kept)} only",
                        _W_CONVENTION + _ELIM_BONUS))
        else:
            out.append(("obs",
                        f"sign convention: {label} — keeps {_ccs(kept)}; "
                        "eliminates rival signage systems", _W_CONVENTION))
        if len(seen) >= 2:
            break
    return out


def _cue_biome(tokens) -> list:
    blob = " ".join(tokens)
    kept: list = []
    labels: list = []
    elim_phrase = ""
    for keywords, ccs, label, phrase in _BIOME_HINTS:
        if not any(re.search(r"\b" + re.escape(kw) + r"\b", blob)
                   for kw in keywords):
            continue
        labels.append(label)
        elim_phrase = elim_phrase or phrase
        for c in ccs:
            if c not in kept:
                kept.append(c)
        if len(labels) >= 3:
            break
    if not labels:
        return []
    cue_txt = ", ".join(labels)
    if len(kept) <= _ELIM_MAX_KEPT:
        return [("obs", f"vegetation/climate: {cue_txt}", _W_BIOME),
                ("elim", f"{cue_txt} biome — {'/'.join(kept)} only",
                 _W_BIOME + _ELIM_BONUS)]
    return [("obs",
             f"vegetation/climate: {cue_txt} — keeps {_ccs(kept)}; "
             f"{elim_phrase}", _W_BIOME)]


def _cue_indicators(notes, ccs) -> list:
    """Domain/phone/legal/currency in-text hints — the classic knock-out."""
    out: list = []
    used: set = set()
    for note in notes:
        m = re.match(r"(.+?)\s*->\s*([A-Za-z]{2})\s*$", str(note).strip())
        if not m:
            continue
        prefix, cc = m.group(1).strip(), _norm_cc(m.group(2))
        if not cc or prefix in used:
            continue
        used.add(prefix)
        out.append(("obs", f"in-text hint: {prefix} — points at {cc}",
                    _W_INDICATOR))
        out.append(("elim", f"{prefix} — {cc} only",
                    _W_INDICATOR + _ELIM_BONUS))
        if len(used) >= 3:
            break
    if not out:
        flat = [c for c in (_norm_cc(x) for x in ccs) if c]
        if len(flat) == 1:
            out.append(("obs", f"in-text country hint — points at {flat[0]}",
                        _W_INDICATOR))
            out.append(("elim", f"in-text indicator — {flat[0]} only",
                        _W_INDICATOR + _ELIM_BONUS))
        elif flat:
            out.append(("obs",
                        f"in-text country indicators: {_ccs(flat)} — dial "
                        "codes / domains point here", _W_INDICATOR))
    return out


def _cue_geocode(ccs) -> list:
    flat = [c for c in (_norm_cc(x) for x in ccs) if c]
    if not flat:
        return []
    return [("obs",
             f"sign text geocodes to {_ccs(flat)} — place-name read",
             _W_GEOCODE)]


# --------------------------------------------------------------------------
# Ranking normalization — tolerates match_score rows, tuples, {cc: score}.
# --------------------------------------------------------------------------

def _norm_ranked(raw) -> list[dict]:
    if isinstance(raw, dict):
        raw = [{"cc": k, "score": v} for k, v in raw.items()]
    out: list[dict] = []
    for r in raw or []:
        if isinstance(r, dict):
            cc = (r.get("cc") or r.get("country_code") or r.get("country")
                  or r.get("code") or r.get("iso"))
            score = r.get("score", r.get("confidence",
                                         r.get("weight", 0.0)))
        elif isinstance(r, (list, tuple)) and r:
            cc = r[0]
            score = r[1] if len(r) > 1 else 0.0
        else:
            continue
        ncc = _norm_cc(cc)
        if not ncc:
            continue
        f = _num(score)
        if f > 1.0 and f <= 100.0:
            f /= 100.0
        f = max(0.0, min(1.0, f))
        for e in out:
            if e["cc"] == ncc:
                if f > e["score"]:
                    e["score"] = f
                break
        else:
            out.append({"cc": ncc, "score": round(f, 4)})
    out.sort(key=lambda e: -e["score"])
    return out


# --------------------------------------------------------------------------
# build_chain — the ordered elimination story
# --------------------------------------------------------------------------

def build_chain(fv: dict, ranked: list,
                observations: dict | None = None) -> list[dict]:
    """Render the elimination chain for a feature vector + ranking.

    ``fv`` is the scoring feature vector (osint_features.feature_vector
    output, possibly enriched by the pipeline); ``ranked`` is a
    score_candidates-style candidate list; ``observations`` holds the
    text-stage keys (driving_side / sign_language / plate_style /
    road_markings) which take precedence over the same keys in ``fv``.

    Step order is discrimination-first — the way a GeoGuessr pro
    narrates (big world-splits before fine print): hemisphere, driving
    side, language/script, plates, lane color, road markings, sign
    conventions, biome, in-text country hints, geocode — then the
    closing rank line. Returns [] when there is nothing to narrate.
    """
    fv = fv if isinstance(fv, dict) else {}
    obs = observations if isinstance(observations, dict) else {}
    steps: list[dict] = []
    seen_text: set = set()

    def emit(kind, text, weight):
        t = str(text).strip()
        if not t or t in seen_text:
            return
        seen_text.add(t)
        steps.append(_step(kind, t, weight))

    def run(renderer, value):
        if value is None:
            return
        try:
            for kind, text, weight in renderer(value) or []:
                emit(kind, text, weight)
        except Exception:
            pass  # a malformed cue never kills the chain

    # hemisphere — explicit read, or derived from a sun_side frame-side.
    hemi = _picked(obs, fv, "hemisphere", "sun_hemisphere")
    if hemi is None:
        side = _first_str(_picked(obs, fv, "sun_side"))
        hemi = {"left": "southern", "right": "northern"}.get(side)
    run(_cue_hemisphere, hemi)

    run(_cue_driving, _picked(obs, fv, "driving_side"))

    langs = _strs(_picked(obs, fv, "sign_language", "language"))
    langs += _strs(fv.get("languages_scripts"))
    run(_cue_language, langs)

    run(_cue_plate, _picked(obs, fv, "plate_style"))

    lanes = _strs(_picked(obs, fv, "lane_markings_color"))
    lanes += _strs(fv.get("marking_colors"))
    run(_cue_lane_color, lanes)

    run(_cue_markings, _strs(_picked(obs, fv, "road_markings")))

    convs = _strs(_picked(obs, fv, "sign_conventions", "sign_convention"))
    sign = fv.get("sign_style")
    if isinstance(sign, dict):
        convs += _strs(sign.get("convention_guess"))
    elif isinstance(sign, str):
        convs += _strs(sign)
    run(_cue_conventions, convs)

    biome_txt = (_strs(_picked(obs, fv, "vegetation"))
                 + _strs(_picked(obs, fv, "climate_cues", "climate"))
                 + _strs(fv.get("biome")))
    run(_cue_biome, biome_txt)

    notes = _strs(fv.get("country_indicator_notes"))
    run(lambda v: _cue_indicators(v, fv.get("country_indicators")), notes
        if notes else fv.get("country_indicators"))

    run(_cue_geocode, _strs(fv.get("geocode_ccs")))

    # Closing rank line — the "what remains" of the story.
    top = _norm_ranked(ranked)
    if steps or top:
        n = len(steps)
        if top:
            tail = " > ".join(f"{e['cc']} {e['score']:.2f}" for e in top[:3])
            emit("rank", f"after {n} cues: {tail}", top[0]["score"])
        else:
            emit("rank", f"after {n} cues: no candidate ranking available",
                 0.0)
    return steps


# --------------------------------------------------------------------------
# chain_html — escaped <ol> fragment for the report, <= _MAX_ROWS rows.
# --------------------------------------------------------------------------

def chain_html(chain: list[dict]) -> str:
    """Render a build_chain result as an escaped ``<ol>`` HTML fragment.

    Inline styles only (the report inlines everything): 11px monospace,
    obs rows #e5e5e5, elim #f87171, rank #facc15. Hard-capped at 14 rows —
    over-long chains keep the head plus the closing rank line with an
    ellipsis row for the elided middle.
    """
    rows = [s for s in chain or []
            if isinstance(s, dict) and str(s.get("text") or "").strip()]
    if not rows:
        return ""
    if len(rows) > _MAX_ROWS:
        elided = len(rows) - (_MAX_ROWS - 1)
        rows = (rows[:_MAX_ROWS - 2]
                + [{"kind": "obs", "text": f"… {elided} more cues …"}]
                + rows[-1:])

    lis = []
    for s in rows:
        color = _KIND_COLOR.get(s.get("kind"), _KIND_COLOR["obs"])
        lis.append(
            f'<li style="color:{color};margin:1px 0">'
            f'{html.escape(str(s["text"]))}</li>')
    return (
        '<ol class="osint-chain" style="font:11px/1.55 ui-monospace,'
        "'SF Mono',Menlo,Consolas,monospace;list-style:decimal;"
        "margin:6px 0;padding-left:22px;color:#e5e5e5\">"
        + "".join(lis) + "</ol>"
    )


# --------------------------------------------------------------------------
# chain_evidence — narration-only card; the data carries the chain.
# --------------------------------------------------------------------------

def chain_evidence(chain: list[dict], claimed_cc=None) -> "Evidence":
    """Package a chain as one Evidence card (type='osint_chain').

    Always INCONCLUSIVE 0.3 — the chain *narrates* the other geo cards,
    it is not a verdict itself. ``summary`` is the last rank step's text
    (the "what remains" line); the full step list rides in ``data``.
    """
    ClaimVerdict, Evidence = _schema()

    steps = [dict(s) for s in chain or [] if isinstance(s, dict)]
    last_rank = None
    for s in reversed(steps):
        if s.get("kind") == "rank" and s.get("text"):
            last_rank = s
            break
    if last_rank is not None:
        summary = str(last_rank["text"])
    elif steps:
        summary = (f"elimination chain narrated {len(steps)} cue step(s) "
                   "— no candidate ranking to close it")
    else:
        summary = ("no elimination chain — no geo cues observed, "
                   "no candidates ranked")
    return Evidence(
        type="osint_chain",
        summary=summary,
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.3,
        data={"chain": steps, "claimed_cc": claimed_cc or None},
    )


# --------------------------------------------------------------------------
# Self-test: python sundial/osint_chain.py  (or python -m sundial.osint_chain)
# --------------------------------------------------------------------------

def _self_test() -> int:
    fv = {
        "hemisphere": "northern",
        "languages_scripts": ["en", "latin"],
        "sign_conventions": ["uk-style"],
        "marking_colors": ["white"],
        "vegetation": ["oak", "hedgerow"],
        "climate_cues": ["overcast"],
        "country_indicators": ["GB"],
        "country_indicator_notes": ["dial +44 -> GB", "tld .uk -> GB"],
        "geocode_ccs": ["GB", "IE"],
    }
    obs = {
        "driving_side": "left",
        "sign_language": "en",
        "plate_style": "uk",
        "road_markings": ["double_yellow_lines", "zigzag_lines"],
    }
    ranked = [
        {"cc": "GB", "score": 0.72, "hits": ["left_drive", "plate_uk"]},
        {"cc": "IE", "score": 0.55},
        {"cc": "AU", "score": 0.44},
    ]

    fails = 0

    def check(name, cond):
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    chain = build_chain(fv, ranked, obs)
    print("chain:")
    for s in chain:
        print(f"  [{s['kind']:>4}] w={s['weight']:.2f}  {s['text']}")

    kinds = [s["kind"] for s in chain]
    check("UK-ish fv produces obs rows", "obs" in kinds)
    check("UK-ish fv produces elim rows", "elim" in kinds)
    check("UK-ish fv produces a rank row", "rank" in kinds)
    check("rank row is last", kinds and kinds[-1] == "rank")
    check("driving cue narrated",
          any("driving side: left" in s["text"] for s in chain))
    check("plate knock-out narrated",
          any(s["kind"] == "elim" and "GB only" in s["text"]
              for s in chain))
    check("rank text is top-3",
          chain[-1]["text"].endswith("GB 0.72 > IE 0.55 > AU 0.44"))
    check("weights are floats in 0..1",
          all(isinstance(s["weight"], float) and 0.0 <= s["weight"] <= 1.0
              for s in chain))

    frag = chain_html(chain)
    check("html is an <ol> fragment",
          frag.startswith("<ol") and frag.endswith("</ol>"))
    check("html escapes cue text",
          "&lt;" not in frag or "<" not in
          frag.split(">", 1)[1].rsplit("<", 1)[0])
    check("html colors present",
          "#e5e5e5" in frag and "#f87171" in frag and "#facc15" in frag)
    check("html 11px mono", "11px" in frag and "monospace" in frag)

    evil = [{"kind": "obs", "text": "<script>alert(1)</script>",
             "weight": 0.5}]
    frag_evil = chain_html(evil)
    check("html escapes injected markup",
          "<script>" not in frag_evil and "&lt;script&gt;" in frag_evil)

    long_chain = [{"kind": "obs", "text": f"cue {i}", "weight": 0.1}
                  for i in range(20)]
    n_li = chain_html(long_chain).count("<li")
    check(f"html capped at {_MAX_ROWS} rows", n_li <= _MAX_ROWS)

    check("empty inputs -> []", build_chain({}, [], None) == [])
    check("empty chain -> '' html", chain_html([]) == "")
    check("none-ish inputs -> []",
          build_chain(None, None, "junk") == [])

    try:
        ev = chain_evidence(chain, "GB")
        print(f"\nevidence[{ev.type}] {ev.verdict.value} "
              f"conf={ev.confidence}: {ev.summary}")
        check("evidence type", ev.type == "osint_chain")
        check("evidence inconclusive", ev.verdict.value == "inconclusive")
        check("evidence conf 0.3", ev.confidence == 0.3)
        check("evidence summary = last rank step",
              ev.summary == chain[-1]["text"])
        check("evidence data carries chain",
              ev.data.get("chain") == chain)
        ev_empty = chain_evidence([], None)
        check("empty chain evidence still inconclusive 0.3",
              ev_empty.verdict.value == "inconclusive"
              and ev_empty.confidence == 0.3)
    except ImportError:
        print("\n(run as `python -m sundial.osint_chain` to build Evidence)")

    print(f"\nosint_chain self-test: "
          f"{'all checks passed' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
