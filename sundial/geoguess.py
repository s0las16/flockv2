"""Geoguess stage — quarantined VLM country guesses as weak corroboration.

A sibling prompt (osint_prompts.GEOGUESS_PROMPT) asks the VLM to guess the
filming country per frame, yielding {country_guess, region_guess, confidence,
reasoning} dicts (already parsed — this module never calls the VLM itself).

The catch: VLMs hallucinate *confidently*. A geoguess answer is GeoGuessr-
flavored pattern matching, not observation — so this signal is QUARANTINED:
it can only ever be weak corroboration. Evidence confidence is hard-capped
at 0.4, and the aggregate refuses to emit a "stable" country unless one
country identity wins a plurality guessed in >=2 frames. When every frame
names a different country, that IS the hallucination detector firing.

No third-party deps, no network. Country-name -> ISO 3166-1 alpha-2 is a
static table with the usual aliases (uk/usa/holland/uae/...).
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

# A country must be guessed in at least this many frames to be "stable".
_STABLE_MIN_FRAMES = 2

# Normalized strings that mean "the model abstained" — NOT a country guess.
# The prompt instructs "unknown"/"none"/null for generic scenes; an honest
# abstention must never count as a vote for a place called "unknown".
_ABSTAIN = frozenset({
    "unknown", "none", "null", "na", "n a", "unclear", "unsure",
    "cant tell", "can't tell", "cannot determine", "unidentifiable",
})

# Quarantine caps — this signal never exceeds these, whatever the VLM says.
_MAX_CONF = 0.4            # hard cap on ANY geoguess evidence
_NO_CONSENSUS_CONF = 0.15  # guesses disagree / guess won't resolve
_UNCOMPARED_CONF = 0.20    # stable guess, but nothing to compare it to


# --------------------------------------------------------------------------
# Name normalization
# --------------------------------------------------------------------------

def _norm_name(value) -> str:
    """Normalize a country/region string: fold diacritics, lowercase,
    punctuation -> space, collapse whitespace.

    "  United-Kingdom! " -> "united kingdom"; "U.S.A." -> "u s a";
    "Côte d'Ivoire" -> "cote d ivoire".
    """
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


def _to_conf(value) -> float | None:
    """Best-effort confidence parse -> 0..1; tolerates percent-style (>1)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if f > 1.0:
        f /= 100.0
    return max(0.0, min(1.0, f))


# --------------------------------------------------------------------------
# Country name -> ISO 3166-1 alpha-2 (static table, keys pre-normalization)
# --------------------------------------------------------------------------

_COUNTRY_TABLE = {
    # --- aliases / colloquial names first (spec-mandated + obvious) ---
    "uk": "GB", "united kingdom": "GB", "great britain": "GB",
    "britain": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
    "northern ireland": "GB",
    "usa": "US", "u s": "US", "us": "US", "u s a": "US",
    "united states": "US", "united states of america": "US",
    "america": "US",
    "holland": "NL", "the netherlands": "NL", "netherlands": "NL",
    "south korea": "KR", "korea": "KR", "republic of korea": "KR",
    "north korea": "KP",
    "uae": "AE", "u a e": "AE", "united arab emirates": "AE",
    "emirates": "AE",
    "czechia": "CZ", "czech republic": "CZ",
    "russia": "RU", "russian federation": "RU",
    "ivory coast": "CI", "cote d ivoire": "CI", "cote divoire": "CI",
    "cape verde": "CV", "cabo verde": "CV",
    "myanmar": "MM", "burma": "MM",
    "turkiye": "TR", "turkey": "TR",
    "eswatini": "SZ", "swaziland": "SZ",
    "north macedonia": "MK", "macedonia": "MK",
    "democratic republic of the congo": "CD", "drc": "CD",
    "congo kinshasa": "CD", "congo": "CD",
    "republic of the congo": "CG", "congo brazzaville": "CG",
    "east timor": "TL", "timor leste": "TL",
    "vatican": "VA", "vatican city": "VA", "holy see": "VA",
    "bosnia": "BA", "bosnia and herzegovina": "BA",
    "bosnia herzegovina": "BA",
    "hong kong": "HK", "macau": "MO", "puerto rico": "PR",
    "palestine": "PS", "kosovo": "XK",
    "germany": "DE", "deutschland": "DE",
    "tahiti": "PF", "french polynesia": "PF",
    "the gambia": "GM", "the bahamas": "BS",
    # --- Europe ---
    "albania": "AL", "andorra": "AD", "austria": "AT", "belarus": "BY",
    "belgium": "BE", "bulgaria": "BG", "croatia": "HR", "cyprus": "CY",
    "denmark": "DK", "estonia": "EE", "finland": "FI", "france": "FR",
    "greece": "GR", "hungary": "HU", "iceland": "IS", "ireland": "IE",
    "italy": "IT", "latvia": "LV", "liechtenstein": "LI",
    "lithuania": "LT", "luxembourg": "LU", "malta": "MT",
    "moldova": "MD", "monaco": "MC", "montenegro": "ME",
    "norway": "NO", "poland": "PL", "portugal": "PT", "romania": "RO",
    "san marino": "SM", "serbia": "RS", "slovakia": "SK",
    "slovenia": "SI", "spain": "ES", "sweden": "SE",
    "switzerland": "CH", "ukraine": "UA",
    # --- Americas ---
    "argentina": "AR", "bahamas": "BS", "belize": "BZ", "bolivia": "BO",
    "brazil": "BR", "canada": "CA", "chile": "CL", "colombia": "CO",
    "costa rica": "CR", "cuba": "CU", "dominican republic": "DO",
    "ecuador": "EC", "el salvador": "SV", "guatemala": "GT",
    "guyana": "GY", "haiti": "HT", "honduras": "HN", "jamaica": "JM",
    "mexico": "MX", "nicaragua": "NI", "panama": "PA",
    "paraguay": "PY", "peru": "PE", "suriname": "SR",
    "trinidad and tobago": "TT", "uruguay": "UY", "venezuela": "VE",
    "greenland": "GL",
    # --- Asia / Middle East ---
    "afghanistan": "AF", "armenia": "AM", "azerbaijan": "AZ",
    "bahrain": "BH", "bangladesh": "BD", "cambodia": "KH",
    "china": "CN", "georgia": "GE", "india": "IN", "indonesia": "ID",
    "iran": "IR", "iraq": "IQ", "israel": "IL", "japan": "JP",
    "jordan": "JO", "kazakhstan": "KZ", "kuwait": "KW",
    "kyrgyzstan": "KG", "laos": "LA", "lebanon": "LB",
    "malaysia": "MY", "maldives": "MV", "mongolia": "MN",
    "nepal": "NP", "oman": "OM", "pakistan": "PK",
    "philippines": "PH", "qatar": "QA", "saudi arabia": "SA",
    "singapore": "SG", "sri lanka": "LK", "syria": "SY",
    "taiwan": "TW", "tajikistan": "TJ", "thailand": "TH",
    "uzbekistan": "UZ", "vietnam": "VN", "yemen": "YE",
    # --- Africa ---
    "algeria": "DZ", "angola": "AO", "botswana": "BW",
    "burkina faso": "BF", "cameroon": "CM", "chad": "TD",
    "egypt": "EG", "ethiopia": "ET", "gambia": "GM", "ghana": "GH",
    "kenya": "KE", "lesotho": "LS", "libya": "LY",
    "madagascar": "MG", "malawi": "MW", "mali": "ML",
    "mauritius": "MU", "morocco": "MA", "mozambique": "MZ",
    "namibia": "NA", "niger": "NE", "nigeria": "NG", "rwanda": "RW",
    "senegal": "SN", "somalia": "SO", "south africa": "ZA",
    "south sudan": "SS", "sudan": "SD", "tanzania": "TZ",
    "togo": "TG", "tunisia": "TN", "uganda": "UG", "zambia": "ZM",
    "zimbabwe": "ZW",
    # --- Oceania ---
    "australia": "AU", "fiji": "FJ", "new zealand": "NZ",
    "papua new guinea": "PG", "samoa": "WS", "tonga": "TO",
    "vanuatu": "VU", "guam": "GU",
}

# Keys normalized once at import so lookups stay cheap and consistent.
_COUNTRY_CC = {_norm_name(k): v for k, v in _COUNTRY_TABLE.items()}


def resolve_cc(country_name: str | None) -> str | None:
    """Map a country name to ISO 3166-1 alpha-2 ("south korea" -> "KR").

    Tries the normalized name, a leading-"the"-stripped variant, and a
    fully squashed form ("u s a" -> "usa"). None when unmapped.
    """
    key = _norm_name(country_name)
    if not key:
        return None
    for cand in (key, key[4:] if key.startswith("the ") else "",
                 key.replace(" ", "")):
        cc = _COUNTRY_CC.get(cand)
        if cc:
            return cc
    return None


def _norm_claim_cc(claimed_cc) -> str | None:
    """Normalize a claimed country code/name to alpha-2, or None."""
    s = str(claimed_cc or "").strip()
    if not s:
        return None
    if re.fullmatch(r"[A-Za-z]{2}", s):
        return s.upper()
    return resolve_cc(s)  # tolerate a full country name sneaking in


# --------------------------------------------------------------------------
# Aggregation — the hallucination detector
# --------------------------------------------------------------------------

def aggregate_guesses(per_frame: list[dict]) -> dict:
    """Fold per-frame geoguess dicts into a consensus summary.

    Each frame may carry {country_guess, region_guess, confidence,
    reasoning}. Country strings are normalized (case/diacritics/punct) and
    counted per *surface form* in `votes`, but stability is judged per
    *resolved country identity* — "united kingdom", "UK!" and
    "great britain." are the same guess, not a 3-way split.

    `stable_country` is the top normalized surface form of the winning
    identity, and only when that identity (a) appears in >= 2 frames and
    (b) is the UNIQUE top vote-getter — a 2-vs-2 tie means the model is
    guessing, which is exactly what this detector exists to catch.
    "unknown"/"none"-style abstentions never vote.
    """
    votes: dict[str, int] = {}               # surface form -> frames
    ident_votes: dict[str, int] = {}         # cc (or name) -> frames
    ident_surface: dict[str, Counter] = {}   # identity -> surface forms
    regions: list[str] = []
    seen_regions: set[str] = set()
    reasonings: list[str] = []
    confs: list[float] = []

    for entry in per_frame or []:
        if not isinstance(entry, dict):
            continue
        c = _norm_name(entry.get("country_guess"))
        if c and c not in _ABSTAIN:
            votes[c] = votes.get(c, 0) + 1
            ident = resolve_cc(c) or c
            ident_votes[ident] = ident_votes.get(ident, 0) + 1
            ident_surface.setdefault(ident, Counter())[c] += 1
        r = _norm_name(entry.get("region_guess"))
        if r and r not in _ABSTAIN and r not in seen_regions:
            seen_regions.add(r)
            regions.append(r)
        conf = _to_conf(entry.get("confidence"))
        if conf is not None:
            confs.append(conf)
        why = str(entry.get("reasoning") or "").strip()
        if why:
            reasonings.append(why)

    stable = None
    if ident_votes:
        top_n = max(ident_votes.values())
        tops = [k for k, n in ident_votes.items() if n == top_n]
        if top_n >= _STABLE_MIN_FRAMES and len(tops) == 1:
            ident = tops[0]
            stable = ident_surface[ident].most_common(1)[0][0]

    return {
        "stable_country": stable,
        "votes": dict(sorted(votes.items(), key=lambda kv: -kv[1])),
        "regions": regions,
        "mean_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "reasonings": reasonings,
    }


# --------------------------------------------------------------------------
# Evidence card — confidence hard-capped at 0.4 (quarantined signal)
# --------------------------------------------------------------------------

def geoguess_evidence(agg: dict, claimed_cc: str | None) -> "Evidence":
    """Package an aggregate_guesses result as one Evidence card.

    Quarantine rules:
      * no usable guesses                    -> INCONCLUSIVE 0.0
      * guesses disagree across frames       -> INCONCLUSIVE 0.15
        (the hallucination detector — a model that can't repeat itself
        is not evidence of anything)
      * stable guess won't resolve to ISO    -> INCONCLUSIVE 0.15
      * stable + resolves, no claimed_cc     -> INCONCLUSIVE 0.20
      * stable + resolves + == claimed_cc    -> SUPPORTS 0.4
      * stable + resolves + != claimed_cc    -> CONTRADICTS 0.4
        (only when claimed_cc is known — never contradicts into a void)
    """
    from .schema import ClaimVerdict, Evidence  # lazy: keep module dep-free

    agg = agg or {}
    votes = dict(agg.get("votes") or {})
    stable = agg.get("stable_country")
    cc = resolve_cc(stable)
    claim = _norm_claim_cc(claimed_cc)
    data = {
        "stable_country": stable,
        "stable_cc": cc,
        "claimed_cc": claim,
        "votes": votes,
        "regions": list(agg.get("regions") or []),
        "mean_conf": agg.get("mean_conf", 0.0),
        "reasonings": list(agg.get("reasonings") or []),
        "quarantined": True,
        "confidence_cap": _MAX_CONF,
    }

    if not votes:
        return Evidence(
            type="geoguess",
            summary="no usable per-frame country guesses — geoguess n/a",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data=data,
        )

    if not stable:
        return Evidence(
            type="geoguess",
            summary=(
                "model guesses disagree across frames "
                f"({votes}) — consensus never formed; "
                "classic VLM hallucination flag"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=_NO_CONSENSUS_CONF,
            data=data,
        )

    if cc is None:
        return Evidence(
            type="geoguess",
            summary=(
                f"model guesses converged on {stable!r} across frames "
                "but it doesn't resolve to an ISO country code"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=_NO_CONSENSUS_CONF,
            data=data,
        )

    if claim is None:
        return Evidence(
            type="geoguess",
            summary=(
                f"model guesses converged on {stable} ({cc}) across frames "
                "— recorded, but no claimed country to compare against"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=_UNCOMPARED_CONF,
            data=data,
        )

    if cc == claim:
        return Evidence(
            type="geoguess",
            summary=(
                f"model guesses converged on {stable} ({cc}) across frames "
                f"— weakly corroborates claimed country {claim} "
                "(quarantined VLM signal, capped)"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=_MAX_CONF,
            data=data,
        )

    return Evidence(
        type="geoguess",
        summary=(
            f"model guesses converged on {stable} ({cc}) across frames "
            f"but claim is {claim} — weak contradiction "
            "(quarantined VLM signal, capped)"
        ),
        verdict=ClaimVerdict.CONTRADICTS,
        confidence=_MAX_CONF,
        data=data,
    )


# --------------------------------------------------------------------------
# Self-test: python -m sundial.geoguess
# --------------------------------------------------------------------------

if __name__ == "__main__":
    fails = 0

    def check(name, got, want):
        global fails
        ok = got == want
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} (want {want!r})")

    # --- resolve_cc spot checks -------------------------------------------
    for name, want in [
        ("United Kingdom", "GB"), ("uk", "GB"), ("Great Britain!", "GB"),
        ("england", "GB"), ("U.S.A.", "US"), ("america", "US"),
        ("Holland", "NL"), ("south korea", "KR"), ("UAE", "AE"),
        ("Czechia", "CZ"), ("russia", "RU"), ("Côte d'Ivoire", "CI"),
        ("atlantis", None), ("", None), (None, None),
    ]:
        check(f"resolve_cc({name!r})", resolve_cc(name), want)

    # --- aggregate: alias consensus = stable; split frames = not ----------
    gb_frames = [
        {"country_guess": "United Kingdom", "region_guess": "England",
         "confidence": 0.82, "reasoning": "double-decker bus, left traffic"},
        {"country_guess": "UK!", "region_guess": "London",
         "confidence": 0.7, "reasoning": "red telephone box"},
        {"country_guess": "great britain.", "region_guess": "England",
         "confidence": 0.9, "reasoning": "english signage"},
    ]
    agg_gb = aggregate_guesses(gb_frames)
    check("stable (alias consensus)", agg_gb["stable_country"], "united kingdom")
    check("votes kept per surface form", len(agg_gb["votes"]), 3)
    check("regions dedup", agg_gb["regions"], ["england", "london"])

    split = aggregate_guesses([
        {"country_guess": "Japan", "confidence": 0.9, "reasoning": "signage"},
        {"country_guess": "Brazil", "confidence": 0.8, "reasoning": "plants"},
        {"country_guess": "Germany", "confidence": 0.85, "reasoning": "plates"},
    ])
    check("stable (all different)", split["stable_country"], None)
    tied = aggregate_guesses([
        {"country_guess": "France"}, {"country_guess": "Germany"},
        {"country_guess": "france!"}, {"country_guess": "germany"},
    ])
    check("stable (2-2 tie)", tied["stable_country"], None)
    abstained = aggregate_guesses([
        {"country_guess": "unknown", "region_guess": "unknown"},
        {"country_guess": "France"}, {"country_guess": "france"},
    ])
    check("abstentions don't vote", abstained["stable_country"], "france")
    check("abstention not in votes", "unknown" in abstained["votes"], False)

    # --- evidence: stable GB vs US claim -> CONTRADICTS; split -> INC -----
    try:
        from .schema import ClaimVerdict  # noqa: F401
    except ImportError:
        print("(run as `python -m sundial.geoguess` to build Evidence objects)")
    else:
        ev = geoguess_evidence(agg_gb, "US")
        check("GB consensus vs US claim", ev.verdict.value, "contradicts")
        check("contradicts conf capped", ev.confidence, 0.4)
        ev = geoguess_evidence(agg_gb, "gb")
        check("GB consensus vs GB claim", ev.verdict.value, "supports")
        check("supports conf capped", ev.confidence, 0.4)
        ev = geoguess_evidence(split, "JP")
        check("split guesses", ev.verdict.value, "inconclusive")
        check("split conf", ev.confidence, 0.15)
        check(
            "split summary names the flag",
            "model guesses disagree across frames" in ev.summary,
            True,
        )
        ev = geoguess_evidence(agg_gb, None)
        check("stable but unclaimed", ev.verdict.value, "inconclusive")

    print(f"geoguess self-test: "
          f"{'all checks passed' if not fails else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)
