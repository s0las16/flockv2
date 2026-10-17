"""Script/language-consistency stage — pure-stdlib text forensics.

The VLM extracts sign text; the *script* it is written in is itself an OSINT
signal. Cyrillic storefronts in a clip claimed to be London are a strong
contradiction; English (latin) signs in claimed Tokyo are not — tourist and
transit signage is latin-script almost everywhere.

No third-party deps, no network. Classification is by Unicode character-name
prefix (with a codepoint-range fallback), so only scripts in the fixed label
set below are counted — Bengali/Georgian/Ethiopic/etc. letters are ignored
rather than mislabelled.

Labels: "latin", "cyrillic", "arabic", "cjk" (incl. hiragana/katakana),
"hangul", "devanagari", "thai", "greek", "hebrew", "mixed", "none".
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter

# First word of a Unicode character name -> script label. Covers the blocks
# we care about; unmapped scripts return None and are ignored.
_NAME_PREFIXES = {
    "LATIN": "latin",
    "CYRILLIC": "cyrillic",
    "ARABIC": "arabic",
    "CJK": "cjk",
    "HANGUL": "hangul",
    "DEVANAGARI": "devanagari",
    "THAI": "thai",
    "GREEK": "greek",
    "HEBREW": "hebrew",
}
# Japanese kana count toward "cjk" — JP signage mixes kanji + kana.
_NAME_SUBSTRINGS = {
    "HIRAGANA": "cjk",
    "KATAKANA": "cjk",
    "CJK UNIFIED": "cjk",
}
# Fallback when a char has no resolvable name (rare for letters).
_RANGE_SCRIPTS = [
    (0x0041, 0x024F, "latin"),
    (0x0400, 0x052F, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0E00, 0x0E7F, "thai"),
    (0x0370, 0x03FF, "greek"),
    (0x1F00, 0x1FFF, "greek"),
    (0x3040, 0x30FF, "cjk"),   # hiragana + katakana
    (0x3400, 0x9FFF, "cjk"),   # CJK unified ideographs
    (0xF900, 0xFAFF, "cjk"),   # CJK compat ideographs
    (0xAC00, 0xD7AF, "hangul"),
    (0x1100, 0x11FF, "hangul"),
]

_DOMINANT_SHARE = 0.5   # strict majority of letters -> that script, else "mixed"
_PRESENT_SHARE = 0.25   # minority share big enough to matter in "mixed" scenes


def _char_script(ch: str) -> str | None:
    """Script label for a single character, or None if outside our label set."""
    name = unicodedata.name(ch, "")
    if name:
        first = name.split(" ", 1)[0]
        if first in _NAME_PREFIXES:
            return _NAME_PREFIXES[first]
        for sub, script in _NAME_SUBSTRINGS.items():
            if sub in name:
                return script
        return None
    cp = ord(ch)
    for lo, hi, script in _RANGE_SCRIPTS:
        if lo <= cp <= hi:
            return script
    return None


def _script_counts(text: str) -> Counter:
    """Count letters per script. Digits, punctuation and unknown scripts ignored."""
    counts: Counter = Counter()
    for ch in text or "":
        if not unicodedata.category(ch).startswith("L"):
            continue
        s = _char_script(ch)
        if s:
            counts[s] += 1
    return counts


def dominant_script(text: str) -> str:
    """Dominant script of the letters in `text`.

    "none" when no letters (of a known script) are present; "mixed" when no
    single script holds a strict majority.
    """
    counts = _script_counts(text)
    total = sum(counts.values())
    if total == 0:
        return "none"
    top, top_n = counts.most_common(1)[0]
    if top_n / total > _DOMINANT_SHARE:
        return top
    return "mixed"


# --- claimed-place -> expected scripts (deliberately not exhaustive) --------

# Unambiguous place names — matched case-insensitively on word boundaries.
_PLACE_NAMES: dict[str, list[str]] = {
    # latin-script anglosphere / Europe / Americas / etc.
    "united kingdom": ["latin"], "great britain": ["latin"], "britain": ["latin"],
    "england": ["latin"], "scotland": ["latin"], "wales": ["latin"],
    "london": ["latin"], "manchester": ["latin"], "birmingham": ["latin"],
    "edinburgh": ["latin"], "uk": ["latin"],
    "united states": ["latin"], "usa": ["latin"], "america": ["latin"],
    "new york": ["latin"], "los angeles": ["latin"], "chicago": ["latin"],
    "san francisco": ["latin"], "miami": ["latin"], "boston": ["latin"],
    "canada": ["latin"], "toronto": ["latin"], "vancouver": ["latin"],
    "australia": ["latin"], "sydney": ["latin"], "melbourne": ["latin"],
    "new zealand": ["latin"], "auckland": ["latin"],
    "ireland": ["latin"], "dublin": ["latin"],
    "germany": ["latin"], "deutschland": ["latin"], "berlin": ["latin"],
    "munich": ["latin"], "hamburg": ["latin"], "frankfurt": ["latin"],
    "austria": ["latin"], "vienna": ["latin"],
    "switzerland": ["latin"], "zurich": ["latin"], "geneva": ["latin"],
    "france": ["latin"], "paris": ["latin"], "lyon": ["latin"],
    "marseille": ["latin"],
    "spain": ["latin"], "madrid": ["latin"], "barcelona": ["latin"],
    "portugal": ["latin"], "lisbon": ["latin"],
    "italy": ["latin"], "rome": ["latin"], "milan": ["latin"],
    "netherlands": ["latin"], "holland": ["latin"], "amsterdam": ["latin"],
    "belgium": ["latin"], "brussels": ["latin"],
    "luxembourg": ["latin"],
    "sweden": ["latin"], "stockholm": ["latin"],
    "norway": ["latin"], "oslo": ["latin"],
    "denmark": ["latin"], "copenhagen": ["latin"],
    "finland": ["latin"], "helsinki": ["latin"],
    "iceland": ["latin"], "reykjavik": ["latin"],
    "poland": ["latin"], "warsaw": ["latin"],
    "czechia": ["latin"], "czech republic": ["latin"], "prague": ["latin"],
    "slovakia": ["latin"], "bratislava": ["latin"],
    "hungary": ["latin"], "budapest": ["latin"],
    "romania": ["latin"], "bucharest": ["latin"],
    "croatia": ["latin"], "zagreb": ["latin"],
    "slovenia": ["latin"], "ljubljana": ["latin"],
    "estonia": ["latin"], "tallinn": ["latin"],
    "latvia": ["latin"], "riga": ["latin"],
    "lithuania": ["latin"], "vilnius": ["latin"],
    "malta": ["latin"], "albania": ["latin"], "tirana": ["latin"],
    "turkey": ["latin"], "istanbul": ["latin"], "ankara": ["latin"],
    "mexico": ["latin"], "mexico city": ["latin"],
    "brazil": ["latin"], "rio de janeiro": ["latin"], "sao paulo": ["latin"],
    "argentina": ["latin"], "buenos aires": ["latin"],
    "chile": ["latin"], "santiago": ["latin"],
    "colombia": ["latin"], "bogota": ["latin"],
    "peru": ["latin"], "lima": ["latin"],
    "vietnam": ["latin"], "hanoi": ["latin"], "ho chi minh": ["latin"],
    "indonesia": ["latin"], "jakarta": ["latin"],
    "malaysia": ["latin"], "kuala lumpur": ["latin"],
    "philippines": ["latin"], "manila": ["latin"],
    "south africa": ["latin"], "cape town": ["latin"], "johannesburg": ["latin"],
    "nigeria": ["latin"], "lagos": ["latin"], "kenya": ["latin"],
    "nairobi": ["latin"],
    # cyrillic-script countries
    "russia": ["cyrillic"], "moscow": ["cyrillic"], "st petersburg": ["cyrillic"],
    "saint petersburg": ["cyrillic"], "novosibirsk": ["cyrillic"],
    "ukraine": ["cyrillic"], "kyiv": ["cyrillic"], "kiev": ["cyrillic"],
    "belarus": ["cyrillic"], "minsk": ["cyrillic"],
    "bulgaria": ["cyrillic"], "sofia": ["cyrillic"],
    "kazakhstan": ["cyrillic"], "almaty": ["cyrillic"],
    "kyrgyzstan": ["cyrillic"], "bishkek": ["cyrillic"],
    "mongolia": ["cyrillic"], "ulaanbaatar": ["cyrillic"],
    # Serbia/Bosnia use cyrillic AND latin officially.
    "serbia": ["cyrillic", "latin"], "belgrade": ["cyrillic", "latin"],
    "bosnia": ["cyrillic", "latin"], "sarajevo": ["cyrillic", "latin"],
    "montenegro": ["cyrillic", "latin"],
    # CJK-script countries
    "japan": ["cjk"], "tokyo": ["cjk"], "osaka": ["cjk"], "kyoto": ["cjk"],
    "china": ["cjk"], "beijing": ["cjk"], "shanghai": ["cjk"],
    "shenzhen": ["cjk"], "guangzhou": ["cjk"], "chengdu": ["cjk"],
    "taiwan": ["cjk"], "taipei": ["cjk"],
    # bilingual zh/en signage
    "hong kong": ["cjk", "latin"], "macau": ["cjk", "latin"],
    "singapore": ["cjk", "latin"],
    # hangul
    "korea": ["hangul"], "south korea": ["hangul"], "north korea": ["hangul"],
    "seoul": ["hangul"], "busan": ["hangul"],
    # arabic-script region
    "middle east": ["arabic"],
    "saudi arabia": ["arabic"], "riyadh": ["arabic"], "jeddah": ["arabic"],
    "uae": ["arabic"], "emirates": ["arabic"], "dubai": ["arabic"],
    "abu dhabi": ["arabic"], "qatar": ["arabic"], "doha": ["arabic"],
    "kuwait": ["arabic"], "bahrain": ["arabic"], "oman": ["arabic"],
    "muscat": ["arabic"], "jordan": ["arabic"], "amman": ["arabic"],
    "lebanon": ["arabic"], "beirut": ["arabic"], "iraq": ["arabic"],
    "baghdad": ["arabic"], "syria": ["arabic"], "damascus": ["arabic"],
    "yemen": ["arabic"], "egypt": ["arabic"], "cairo": ["arabic"],
    "morocco": ["arabic"], "casablanca": ["arabic"], "algeria": ["arabic"],
    "algiers": ["arabic"], "tunisia": ["arabic"], "tunis": ["arabic"],
    "libya": ["arabic"], "tripoli": ["arabic"], "sudan": ["arabic"],
    "khartoum": ["arabic"], "palestine": ["arabic"], "gaza": ["arabic"],
    "iran": ["arabic"], "tehran": ["arabic"],
    "afghanistan": ["arabic"], "kabul": ["arabic"],
    "pakistan": ["arabic", "latin"], "karachi": ["arabic", "latin"],
    "lahore": ["arabic", "latin"],
    # greek
    "greece": ["greek"], "athens": ["greek"], "thessaloniki": ["greek"],
    "cyprus": ["greek"], "nicosia": ["greek"],
    # hebrew — Israeli signs are commonly he/ar/en trilingual.
    "israel": ["hebrew", "arabic", "latin"], "jerusalem": ["hebrew", "arabic", "latin"],
    "tel aviv": ["hebrew", "arabic", "latin"], "haifa": ["hebrew", "arabic", "latin"],
    # thai
    "thailand": ["thai"], "bangkok": ["thai"], "chiang mai": ["thai"],
    "phuket": ["thai"],
    # india — devanagari + heavy english signage
    "india": ["devanagari", "latin"], "delhi": ["devanagari", "latin"],
    "new delhi": ["devanagari", "latin"], "mumbai": ["devanagari", "latin"],
    "bangalore": ["devanagari", "latin"], "bengaluru": ["devanagari", "latin"],
    "kolkata": ["devanagari", "latin"], "chennai": ["devanagari", "latin"],
}

# ISO-style codes — matched ONLY as uppercase tokens in the raw claim
# ("London, UK", "DE", "UAE") to avoid collisions with English words
# ("us", "in", "at", "is", "no" would otherwise hit constantly).
_PLACE_CODES: dict[str, list[str]] = {
    "UK": ["latin"], "GB": ["latin"], "US": ["latin"], "USA": ["latin"],
    "CA": ["latin"], "AU": ["latin"], "NZ": ["latin"], "IE": ["latin"],
    "DE": ["latin"], "AT": ["latin"], "CH": ["latin"], "FR": ["latin"],
    "ES": ["latin"], "PT": ["latin"], "IT": ["latin"], "NL": ["latin"],
    "BE": ["latin"], "LU": ["latin"], "SE": ["latin"], "NO": ["latin"],
    "DK": ["latin"], "FI": ["latin"], "IS": ["latin"], "PL": ["latin"],
    "CZ": ["latin"], "SK": ["latin"], "HU": ["latin"], "RO": ["latin"],
    "HR": ["latin"], "SI": ["latin"], "EE": ["latin"], "LV": ["latin"],
    "LT": ["latin"], "MT": ["latin"], "AL": ["latin"], "TR": ["latin"],
    "MX": ["latin"], "BR": ["latin"], "AR": ["latin"], "CL": ["latin"],
    "CO": ["latin"], "PE": ["latin"], "VN": ["latin"], "ID": ["latin"],
    "MY": ["latin"], "PH": ["latin"], "ZA": ["latin"], "NG": ["latin"],
    "KE": ["latin"],
    "RU": ["cyrillic"], "UA": ["cyrillic"], "BY": ["cyrillic"],
    "BG": ["cyrillic"], "KZ": ["cyrillic"], "KG": ["cyrillic"],
    "MN": ["cyrillic"],
    "RS": ["cyrillic", "latin"], "BA": ["cyrillic", "latin"],
    "ME": ["cyrillic", "latin"],
    "JP": ["cjk"], "CN": ["cjk"], "TW": ["cjk"],
    "HK": ["cjk", "latin"], "MO": ["cjk", "latin"], "SG": ["cjk", "latin"],
    "KR": ["hangul"], "KP": ["hangul"],
    "SA": ["arabic"], "AE": ["arabic"], "UAE": ["arabic"], "QA": ["arabic"],
    "KW": ["arabic"], "BH": ["arabic"], "OM": ["arabic"], "JO": ["arabic"],
    "LB": ["arabic"], "IQ": ["arabic"], "SY": ["arabic"], "YE": ["arabic"],
    "EG": ["arabic"], "MA": ["arabic"], "DZ": ["arabic"], "TN": ["arabic"],
    "LY": ["arabic"], "SD": ["arabic"], "PS": ["arabic"], "IR": ["arabic"],
    "AF": ["arabic"],
    "PK": ["arabic", "latin"],
    "GR": ["greek"], "CY": ["greek"],
    "IL": ["hebrew", "arabic", "latin"],
    "TH": ["thai"],
    "IN": ["devanagari", "latin"],
}


def _expected_scripts(claimed_place: str | None) -> list[str]:
    """Map a claimed place string to expected scripts ([] when unmapped)."""
    if not claimed_place or not claimed_place.strip():
        return []
    raw = claimed_place.strip()
    norm = re.sub(r"\s+", " ", raw.lower())
    found: list[str] = []

    def add(scripts: list[str]) -> None:
        for s in scripts:
            if s not in found:
                found.append(s)

    for key, scripts in _PLACE_NAMES.items():
        if re.search(r"\b" + re.escape(key) + r"\b", norm):
            add(scripts)
    for tok in re.findall(r"\b[A-Z]{2,3}\b", raw):
        scripts = _PLACE_CODES.get(tok)
        if scripts:
            add(scripts)
    return found


def script_consistency(texts: list[str], claimed_place: str | None) -> dict:
    """Is the pooled script of the sign text plausible for the claimed place?

    Returns {"script", "expected_scripts", "consistent"} where consistent is:
      True  — dominant (or all present non-latin) scripts are expected there
      False — a non-latin script dominates where latin is expected
              (Cyrillic signage claimed London — the classic flag)
      None  — can't say: unknown place, no letters, latin-in-a-nonlatin
              country (tourist/English signs exist everywhere — never a
              contradiction), or a nonlatin-vs-nonlatin mismatch we don't
              hard-flag.
    """
    pool = " ".join(t for t in (texts or []) if t)
    counts = _script_counts(pool)
    total = sum(counts.values())
    script = dominant_script(pool)
    expected = _expected_scripts(claimed_place)

    if not expected or script == "none":
        consistent = None
    elif script == "mixed":
        present = {
            s for s, n in counts.items()
            if s != "latin" and n / total >= _PRESENT_SHARE
        }
        unexpected = present - set(expected)
        if not unexpected:
            consistent = True            # e.g. devanagari+latin, claimed India
        elif "latin" in expected:
            consistent = False           # e.g. latin+cyrillic, claimed London
        else:
            consistent = None            # e.g. cjk+cyrillic, claimed Tokyo
    elif script in expected:
        consistent = True
    elif script == "latin":
        consistent = None                # tourism/English — never a flag
    elif "latin" in expected:
        consistent = False               # the flag: nonlatin in latin country
    else:
        consistent = None                # nonlatin anomaly in nonlatin place
    return {
        "script": script,
        "expected_scripts": expected,
        "consistent": consistent,
    }


def language_evidence(texts: list[str], claimed_place: str | None) -> "Evidence":
    """Package script consistency as an Evidence card (schema imported lazily)."""
    from .schema import ClaimVerdict, Evidence

    usable = [t.strip() for t in (texts or []) if t and t.strip()]
    res = script_consistency(usable, claimed_place)
    script, expected, ok = res["script"], res["expected_scripts"], res["consistent"]
    data = {**res, "texts": usable, "claimed_place": claimed_place}

    if not usable or script == "none":
        return Evidence(
            type="language",
            summary="no readable letters in extracted sign text — script check n/a",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data=data,
        )
    exp = "/".join(expected) if expected else "?"
    if ok is True:
        return Evidence(
            type="language",
            summary=(
                f"sign text is {script} script — consistent with claimed "
                f"{claimed_place!r} (expects {exp})"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=0.45,
            data=data,
        )
    if ok is False:
        return Evidence(
            type="language",
            summary=(
                f"sign text is {script} script — unexpected for claimed "
                f"{claimed_place!r} (expects {exp}); classic mismatch flag"
            ),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=0.6,
            data=data,
        )
    # consistent=None — name why
    if not expected:
        why = f"no script expectations on file for claimed {claimed_place!r}"
    elif script == "latin":
        why = (
            f"latin-script signs in claimed {claimed_place!r} are common "
            "(tourism/English) — neither confirms nor refutes"
        )
    elif script == "mixed":
        why = (
            f"sign text mixes scripts ({dict(_script_counts(' '.join(usable)))}) "
            f"— no clear expectation for claimed {claimed_place!r}"
        )
    else:
        why = (
            f"{script} script is atypical of claimed {claimed_place!r} "
            f"(expects {exp}) but not conclusive on its own"
        )
    return Evidence(
        type="language",
        summary=f"sign text is {script} script: {why}",
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.25,
        data=data,
    )


if __name__ == "__main__":
    # --- self-check: latin / cyrillic / cjk + friends -----------------------
    checks = [
        ("Oxford Street 123, W1D 2EG", "latin"),
        ("Улица Тверская, Москва", "cyrillic"),
        ("東京駅前の看板", "cjk"),
        ("서울역 입구", "hangul"),
        ("شارع الملك فهد", "arabic"),
        ("Οδός Αθηνάς 12", "greek"),
        ("רחוב דיזנגוף", "hebrew"),
        ("मुंबई सेंट्रल", "devanagari"),
        ("ถนนสุขุมวิท", "thai"),
        ("1234 :// !?", "none"),
        ("AB ВГ", "mixed"),
    ]
    fails = 0
    for text, want in checks:
        got = dominant_script(text)
        ok = "PASS" if got == want else "FAIL"
        if got != want:
            fails += 1
        print(f"  {ok}  dominant_script({text!r}) = {got} (want {want})")
    print(f"self-check: {len(checks) - fails}/{len(checks)} passed")

    # --- demo: python -m sundial.langdetect <claimed_place> <text> [text...] --
    args = sys.argv[1:]
    if fails:
        sys.exit(1)
    if not args:
        print("usage: python -m sundial.langdetect <claimed_place> <text> [text...]")
        sys.exit(0)
    place = args[0] if len(args) > 1 else None
    texts = args[1:] if len(args) > 1 else args
    for t in texts:
        print(f"  '{t}' -> {dominant_script(t)}")
    print("consistency:", script_consistency(texts, place))
    try:
        ev = language_evidence(texts, place)
    except ImportError:
        print("(run as `python -m sundial.langdetect` to build Evidence objects)")
    else:
        print(f"evidence: {ev.verdict.value} conf={ev.confidence} — {ev.summary}")
