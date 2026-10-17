"""Vegetation stage — plants and land cover as a biome tell.

What grows in frame is one of the cheapest GeoGuessr tells: palm trees
don't survive a Norwegian winter, and spruce forest doesn't grow on a
tropical beach. This module maps free-text vegetation cues (from
perception / VLM captions) onto coarse biome classes, then cross-checks
the dominant biome's latitude band against the claimed location.

Pure stdlib — the schema import is lazy so the module loads with zero
project deps, matching plate_style.py / langdetect.py conventions.
"""
from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Biome knowledge base
# --------------------------------------------------------------------------
# Each biome maps to:
#   keywords  — lowercase substrings matched against vegetation cue strings
#   lat_band  — (min, max) absolute latitude where the biome is plausible
#   countries — ISO alpha-2 codes, with "CC-REG" sub-region codes allowed
#               (e.g. "US-CA"), used as a secondary consistency signal when
#               no latitude is available or the band is non-discriminating
_BIOME_KEYWORDS: dict[str, dict] = {
    "tropical": {
        "keywords": {
            "palm", "banana", "broadleaf", "monsoon", "rainforest",
            "jungle", "coconut", "mango", "papaya", "bamboo", "liana",
            "teak", "sugarcane", "lush", "tropical",
        },
        "lat_band": (0.0, 25.0),
        "countries": {
            "BR", "CO", "EC", "PE", "VE", "PA", "CR", "NI", "HN", "GT",
            "BZ", "MX", "CU", "DO", "HT", "JM", "PR", "ID", "MY", "SG",
            "TH", "VN", "KH", "LA", "MM", "PH", "IN", "LK", "BD", "NG",
            "GH", "CI", "CM", "CD", "KE", "UG", "TZ", "MG", "MZ", "ET",
            "FJ", "PG", "US-HI", "US-FL",
        },
    },
    "mediterranean": {
        # ~30-45° coastal belts: Med basin + California/Chile/Cape/SW Aus
        "keywords": {
            "olive", "cypress", "dry scrub", "maquis", "garrigue",
            "lavender", "vineyard", "grapevine", "fig", "stone pine",
            "umbrella pine", "cork oak", "rosemary", "mediterranean",
        },
        "lat_band": (30.0, 45.0),
        "countries": {
            "ES", "IT", "GR", "TR", "PT", "MA", "US-CA",  # core set
            "FR", "HR", "AL", "ME", "BA", "MT", "CY", "LB", "IL", "DZ",
            "TN", "CL", "ZA", "AU-WA", "AU-SA", "AU-VIC",
        },
    },
    "temperate": {
        "keywords": {
            "deciduous", "oak", "maple", "hedgerow", "hedge", "pasture",
            "meadow", "beech", "elm", "willow", "poplar", "orchard",
            "grassland", "hay", "temperate", "mixed forest",
        },
        "lat_band": (30.0, 60.0),
        "countries": {
            "GB", "IE", "FR", "DE", "NL", "BE", "LU", "DK", "PL", "CZ",
            "SK", "AT", "CH", "HU", "RO", "UA", "IT", "ES", "PT", "JP",
            "KR", "CN", "US", "CA", "NZ", "AU-TAS", "AR", "CL",
        },
    },
    "boreal": {
        "keywords": {
            "spruce", "pine", "birch", "conifer", "coniferous", "taiga",
            "fir", "larch", "boreal", "needleleaf", "aspen",
            "evergreen forest", "pine forest", "spruce forest",
            "snow forest", "subarctic",
        },
        "lat_band": (50.0, 70.0),
        "countries": {
            "NO", "SE", "FI", "RU", "CA", "IS", "EE", "LV", "LT",
            "US-AK", "US-MN", "US-ME", "US-WI", "US-MI",
        },
    },
    "arid": {
        # Desert belts ~10-50° (Sahara/Arabian/Australian/Atacama plus cold
        # deserts like Gobi); country list picks up rain-shadow arid zones
        "keywords": {
            "cactus", "cacti", "scrub", "bare rock", "desert", "sand",
            "dune", "succulent", "agave", "yucca", "sagebrush",
            "mesquite", "acacia", "thornbush", "dry grass", "arid",
            "semi-arid", "wadi",
        },
        "lat_band": (10.0, 50.0),
        "countries": {
            "DZ", "LY", "EG", "SD", "TD", "NE", "ML", "MR", "MA", "TN",
            "SA", "YE", "OM", "AE", "QA", "KW", "IQ", "IR", "JO", "IL",
            "AF", "PK", "IN", "TM", "UZ", "KZ", "MN", "CN", "AU", "NA",
            "BW", "ZA", "MX", "CL", "PE", "AR", "ES", "DJ", "SO", "KE",
            "US-AZ", "US-NM", "US-NV", "US-UT", "US-CA", "US-TX",
        },
    },
    "alpine": {
        # Elevation-driven, not latitude-driven: alpine zones exist from
        # the equator (Kilimanjaro, Andes) to the Arctic. Full-range band —
        # lat_band_check treats it as non-discriminating.
        "keywords": {
            "rock", "rocky", "snow-capped", "snow capped", "alpine",
            "alpine meadow", "scree", "treeline", "mountain meadow",
            "glacier", "peak", "ridge", "tundra", "bare slope",
            "mountain",
        },
        "lat_band": (0.0, 90.0),
        "countries": {
            "CH", "AT", "FR", "IT", "DE", "SI", "NO", "SE", "NP", "IN",
            "PK", "CN", "KG", "TJ", "GE", "JP", "NZ", "AR", "CL", "PE",
            "BO", "EC", "CO", "MX", "KE", "TZ", "ET", "MA", "TR",
            "US-CO", "US-WA", "US-OR", "US-AK", "US-MT", "US-WY",
            "CA-BC", "CA-AB",
        },
    },
}

# A lat band this wide (>=80°) carries no discriminating power — used by
# lat_band_check to report "no signal" instead of a free pass.
_UNINFORMATIVE_SPAN = 80.0


def _norm_texts(vegetation: list[str]) -> list[str]:
    """Lowercase/strip cue strings; drop empties and non-strings."""
    out = []
    for v in vegetation or []:
        s = str(v or "").strip().lower()
        if s:
            out.append(s)
    return out


def _matched_keywords(texts: list[str], keywords: set) -> set:
    """Keywords found as substrings in any cue string."""
    return {kw for kw in keywords if any(kw in t for t in texts)}


def veg_biomes(vegetation: list[str]) -> dict[str, float]:
    """Score each biome 0..1 against vegetation cue strings.

    score = (#distinct keywords hit) / sqrt(#keywords in biome), capped
    at 1.0 — sqrt damping so a single hit on a small keyword set doesn't
    read as strong. Returns all biomes (zeros included).
    """
    texts = _norm_texts(vegetation)
    scores: dict[str, float] = {}
    for biome, spec in _BIOME_KEYWORDS.items():
        total = len(spec["keywords"])
        hits = len(_matched_keywords(texts, spec["keywords"]))
        scores[biome] = min(1.0, hits / math.sqrt(total)) if total else 0.0
    return scores


def _dominant_biome(scores: dict[str, float]) -> str | None:
    """Highest-scoring biome, or None if nothing scored. Deterministic."""
    best = None
    for b in sorted(scores):
        if scores[b] > 0 and (best is None or scores[b] > scores[best]):
            best = b
    return best


def _country_match(claimed_cc: str, countries: set) -> bool:
    """Is a normalized claim code consistent with the biome's country set?

    Exact match always counts ("US-CA" in set). A bare country code also
    matches a sub-region entry ("US" ~ "US-CA") since a country-level
    claim can't rule the biome out. The reverse is not true: "US-NY" does
    not match "US-CA".
    """
    if not claimed_cc or not countries:
        return False
    cc = claimed_cc.strip().upper()
    if not cc:
        return False
    if cc in countries:
        return True
    if "-" not in cc:
        return any(e.split("-", 1)[0] == cc for e in countries)
    return False


def lat_band_check(lat: float | None, biomes: dict) -> tuple[str | None, str]:
    """Check claimed latitude against the dominant biome's plausible band.

    -> (status, reason) where status is "consistent" | "conflict" | None.
    None means the check couldn't discriminate: no lat, no scored biome,
    or a band that spans nearly all latitudes (e.g. alpine is elevation-
    driven, so it can neither support nor contradict a latitude).
    Bands are compared against |lat| — hemispheres mirror biomes.
    """
    if lat is None:
        return None, "no claimed latitude to check"
    dominant = _dominant_biome(biomes)
    if dominant is None:
        return None, "no vegetation cues matched any biome"
    lo, hi = _BIOME_KEYWORDS[dominant]["lat_band"]
    if hi - lo >= _UNINFORMATIVE_SPAN:
        return None, (
            f"{dominant} biome spans all latitudes — "
            "band check not discriminating"
        )
    a = abs(lat)
    if lo <= a <= hi:
        return "consistent", (
            f"{dominant} biome fits |lat|={a:.1f} "
            f"within its {lo:.0f}-{hi:.0f} band"
        )
    return "conflict", (
        f"{dominant} biome expects |lat| {lo:.0f}-{hi:.0f}, "
        f"got {a:.1f}"
    )


def veg_evidence(
    vegetation: list[str],
    lat: float | None,
    claimed_cc: str | None,
) -> "Evidence":
    """Package vegetation-vs-claim consistency as one Evidence card.

    Verdicts (all confidences capped at 0.5):
      * dominant biome's lat band conflicts with the claim -> CONTRADICTS 0.45
      * band consistent, or claimed country in the biome's set -> SUPPORTS 0.4
        (+0.1 when both agree)
      * no cues, or nothing checkable -> INCONCLUSIVE
    """
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # direct `python sundial/veg_geo.py` run
        from schema import ClaimVerdict, Evidence

    scores = veg_biomes(vegetation)
    texts = _norm_texts(vegetation)
    hits = {
        b: sorted(_matched_keywords(texts, spec["keywords"]))
        for b, spec in _BIOME_KEYWORDS.items()
        if scores[b] > 0
    }
    dominant = _dominant_biome(scores)
    status, reason = lat_band_check(lat, scores)
    cc = (claimed_cc or "").strip().upper() or None
    country_ok = bool(cc) and dominant is not None and _country_match(
        cc, _BIOME_KEYWORDS[dominant]["countries"]
    )
    data = {
        "vegetation": texts,
        "claimed_lat": lat,
        "claimed_cc": cc,
        "dominant_biome": dominant,
        "biome_scores": {k: round(v, 3) for k, v in scores.items()},
        "hits": hits,
        "lat_check": {"status": status, "reason": reason},
        "country_match": country_ok,
    }
    cap = lambda c: round(min(0.5, c), 2)

    if dominant is None:
        return Evidence(
            type="vegetation",
            summary="no biome-indicator vegetation cues observed",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data=data,
        )

    hit_str = ", ".join(hits.get(dominant, [])[:4]) or dominant
    if status == "conflict":
        return Evidence(
            type="vegetation",
            summary=(
                f"{dominant} vegetation ({hit_str}) unlikely at claimed "
                f"lat {lat}: {reason}"
            ),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=cap(0.45),
            data=data,
        )

    if status == "consistent" or country_ok:
        conf = 0.4 + (0.1 if status == "consistent" and country_ok else 0.0)
        why = []
        if status == "consistent":
            why.append(reason)
        if country_ok:
            why.append(f"{dominant} biome occurs in {cc}")
        return Evidence(
            type="vegetation",
            summary=(
                f"{dominant} vegetation ({hit_str}) consistent with "
                f"claim — {'; '.join(why)}"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=cap(conf),
            data=data,
        )

    return Evidence(
        type="vegetation",
        summary=(
            f"{dominant} vegetation cues ({hit_str}) — {reason}"
        ),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=cap(0.2),
        data=data,
    )


# --------------------------------------------------------------------------
# Self-test: python3 -m sundial.veg_geo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ok = True

    def check(name, got, want):
        global ok
        good = got == want
        ok &= good
        print(f"  {name}: {got} (want {want}) [{'ok' if good else 'MISMATCH'}]")

    cues = ["palm trees", "banana plants"]
    scores = veg_biomes(cues)
    print("biome scores:", {k: round(v, 2) for k, v in scores.items()})
    check("dominant", _dominant_biome(scores), "tropical")

    # palm trees at lat 60 (e.g. claimed Norway) -> conflict
    status, reason = lat_band_check(60.0, scores)
    print("  lat60:", reason)
    check("palm @60N", status, "conflict")
    ev = veg_evidence(cues, lat=60.0, claimed_cc="NO")
    print(f"  evidence[{ev.type}] {ev.verdict.value} "
          f"conf={ev.confidence}: {ev.summary}")
    check("ev verdict", ev.verdict.value, "contradicts")
    check("ev conf <=0.5", ev.confidence <= 0.5, True)

    # same cues at a tropical lat -> consistent
    status2, _ = lat_band_check(12.0, scores)
    check("palm @12N", status2, "consistent")
    ev2 = veg_evidence(cues, lat=12.0, claimed_cc="TH")
    check("ev2 verdict", ev2.verdict.value, "supports")
    check("ev2 conf", ev2.confidence, 0.5)  # 0.4 + 0.1 country agreement

    # boreal cues at a boreal lat
    ev3 = veg_evidence(["spruce forest", "birch trees"], 62.0, "NO")
    print(f"  evidence[{ev3.type}] {ev3.verdict.value} "
          f"conf={ev3.confidence}: {ev3.summary}")
    check("boreal @62N", ev3.verdict.value, "supports")

    # no cues -> inconclusive; cues but nothing to check -> inconclusive
    check("empty", veg_evidence([], None, None).verdict.value, "inconclusive")
    check("no lat/cc", veg_evidence(cues, None, None).verdict.value,
          "inconclusive")

    # alpine spans all latitudes -> band check non-discriminating
    st_al, _ = lat_band_check(5.0, veg_biomes(["snow-capped peaks", "rock"]))
    check("alpine band", st_al, None)

    sys.exit(0 if ok else 1)
