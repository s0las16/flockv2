"""Street-cues stage — high-precision street-furniture & road infrastructure solver.

Inspects structured or free-text observations across eight road-infrastructure
categories:
  * bollards      — reflector band colors, cap colors, wedge/cylinder profiles
  * poles         — concrete ladder holes, step rungs, base hazard stripes, wood/steel
  * guardrail     — A-profile vs B-profile W-beam, wooden log, wire-rope cable, reflectors
  * road_lines    — yellow outer edge lines, dashed white edge lines, double-yellow center
  * crosswalk     — zebra border style, yellow/red backing, Belisha beacons, stripe count
  * street_signs  — blue enamel plaques, low-mounted UK frames, green MUTCD blades, tiles
  * curbs         — black/white, yellow/black, red/white painted curbs, cut granite
  * chevrons      — black-on-yellow, red-on-white, white-on-black, white-on-blue

Pure stdlib at import time; schema types (`ClaimVerdict`, `Evidence`) are imported
lazily with a standalone fallback so `python3 sundial/street_cues.py` runs with zero
package dependencies. Every public function is exception-safe and never raises on
malformed input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Lazy schema import with standalone fallback
# ---------------------------------------------------------------------------


def _load_schema():
    """Return (ClaimVerdict, Evidence) from sundial.schema or standalone fallback."""
    try:
        from .schema import ClaimVerdict, Evidence

        return ClaimVerdict, Evidence
    except ImportError:
        try:
            from schema import ClaimVerdict, Evidence

            return ClaimVerdict, Evidence
        except ImportError:
            pass

    class _FallbackClaimVerdict(str, Enum):
        SUPPORTS = "supports"
        CONTRADICTS = "contradicts"
        INCONCLUSIVE = "inconclusive"

    @dataclass
    class _FallbackEvidence:
        type: str
        summary: str
        verdict: _FallbackClaimVerdict
        confidence: float = 0.0
        data: dict = field(default_factory=dict)
        frames: list = field(default_factory=list)

    return _FallbackClaimVerdict, _FallbackEvidence


# ---------------------------------------------------------------------------
# Country names & aliases (for plain-English summaries & claim normalization)
# ---------------------------------------------------------------------------

_CC_NAMES: dict[str, str] = {
    "AL": "Albania",
    "AR": "Argentina",
    "AT": "Austria",
    "AU": "Australia",
    "BA": "Bosnia and Herzegovina",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "BR": "Brazil",
    "BW": "Botswana",
    "BY": "Belarus",
    "CA": "Canada",
    "CH": "Switzerland",
    "CL": "Chile",
    "CN": "China",
    "CO": "Colombia",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GB": "United Kingdom",
    "GR": "Greece",
    "HK": "Hong Kong",
    "HR": "Croatia",
    "HU": "Hungary",
    "ID": "Indonesia",
    "IE": "Ireland",
    "IL": "Israel",
    "IN": "India",
    "IS": "Iceland",
    "IT": "Italy",
    "JP": "Japan",
    "KR": "South Korea",
    "KZ": "Kazakhstan",
    "LS": "Lesotho",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MC": "Monaco",
    "ME": "Montenegro",
    "MK": "North Macedonia",
    "MT": "Malta",
    "MX": "Mexico",
    "MY": "Malaysia",
    "NA": "Namibia",
    "NL": "Netherlands",
    "NO": "Norway",
    "NZ": "New Zealand",
    "PE": "Peru",
    "PH": "Philippines",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "RS": "Serbia",
    "RU": "Russia",
    "SE": "Sweden",
    "SG": "Singapore",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "SM": "San Marino",
    "SZ": "Eswatini",
    "TH": "Thailand",
    "TR": "Turkey",
    "TW": "Taiwan",
    "UA": "Ukraine",
    "US": "United States",
    "UY": "Uruguay",
    "ZA": "South Africa",
    "AE": "United Arab Emirates",
}

_CC_ALIASES: dict[str, str] = {
    "UK": "GB",
    "GREAT BRITAIN": "GB",
    "UNITED KINGDOM": "GB",
    "ENGLAND": "GB",
    "SCOTLAND": "GB",
    "WALES": "GB",
    "USA": "US",
    "UNITED STATES": "US",
    "AMERICA": "US",
    "EL": "GR",
    "UAE": "AE",
    "CZECH REPUBLIC": "CZ",
}
for _code, _cname in _CC_NAMES.items():
    _CC_ALIASES[_cname.upper()] = _code

# ---------------------------------------------------------------------------
# Canonical category aliases
# ---------------------------------------------------------------------------

_CATEGORY_ALIASES: dict[str, str] = {
    # bollards
    "bollards": "bollards",
    "bollard": "bollards",
    "delineators": "bollards",
    "delineator": "bollards",
    "verge_posts": "bollards",
    "road_posts": "bollards",
    # poles
    "poles": "poles",
    "pole": "poles",
    "utility_poles": "poles",
    "utility_pole": "poles",
    "telegraph_poles": "poles",
    "power_poles": "poles",
    "electric_poles": "poles",
    # guardrail
    "guardrail": "guardrail",
    "guardrails": "guardrail",
    "crash_barrier": "guardrail",
    "crash_barriers": "guardrail",
    "barrier": "guardrail",
    "barriers": "guardrail",
    # road_lines
    "road_lines": "road_lines",
    "road_line": "road_lines",
    "lane_markings": "road_lines",
    "lane_markings_color": "road_lines",
    "road_markings": "road_lines",
    "road_markings_color": "road_lines",
    "marking_colors": "road_lines",
    "lanes": "road_lines",
    "lines": "road_lines",
    "center_line": "road_lines",
    "edge_lines": "road_lines",
    # crosswalk
    "crosswalk": "crosswalk",
    "crosswalks": "crosswalk",
    "pedestrian_crossing": "crosswalk",
    "pedestrian_crossings": "crosswalk",
    "pedestrian_sign": "crosswalk",
    "zebra_crossing": "crosswalk",
    "crossing": "crosswalk",
    # street_signs
    "street_signs": "street_signs",
    "street_sign": "street_signs",
    "street_name_signs": "street_signs",
    "street_name_sign": "street_signs",
    "sign_style": "street_signs",
    "sign_colors": "street_signs",
    "sign_color": "street_signs",
    "direction_sign": "street_signs",
    "road_signs": "street_signs",
    "sign_post": "street_signs",
    "signs": "street_signs",
    # curbs
    "curbs": "curbs",
    "curb": "curbs",
    "kerb": "curbs",
    "kerbs": "curbs",
    # chevrons
    "chevrons": "chevrons",
    "chevron": "chevrons",
    "curve_chevron": "chevrons",
    "curve_signs": "chevrons",
}

# Wildcard observation keys whose text is tested against all categories
_WILDCARD_KEYS: frozenset[str] = frozenset({
    "road_furniture",
    "street_furniture",
    "street_cues",
    "cues",
    "observations",
    "street_scene",
    "notes",
    "misc",
    "description",
    "summary",
})

_JUNK_VALUES: frozenset[str] = frozenset({
    "", "none", "null", "n/a", "na", "unknown", "unk", "unclear", "-", "--", "false",
})


# ---------------------------------------------------------------------------
# High-precision street-infrastructure rule table
# ---------------------------------------------------------------------------
# Each rule specifies:
#   id       : unique identifier
#   category : canonical cue category
#   reason   : human-readable explanation with country arrow (no internal names)
#   label    : concise plain-English cue phrase
#   patterns : tuple of regexes (any match triggers, subject to `exclude`)
#   exclude  : optional regex that suppresses the rule if matched
#   weights  : tuple of (cc, score_delta) in priority order
# ---------------------------------------------------------------------------

_RULES: list[dict[str, Any]] = [
    # --- BOLLARDS -----------------------------------------------------------
    {
        "id": "bollard_red_band_de_at_pl",
        "category": "bollards",
        "reason": "white bollard with red reflector band -> DE, AT, PL",
        "label": "red-band reflector bollard",
        "patterns": (
            r"\bred[- ]band\b",
            r"\bwhite\b.*\bred\b.*\b(?:reflector|band|strip|stripe|rectangle)\b",
            r"\bred\b.*\b(?:reflector|band|strip|stripe)\b.*\b(?:bollard|post|delineator)\b",
            r"\bgerman[- ]style bollard\b",
            r"\bgerman bollard\b",
            r"\bblack top\b.*\bred\b.*\b(?:reflector|band)\b",
        ),
        "exclude": r"\b(?:red cap|red top|red head|yellow bollard|flat white)\b",
        "weights": (("DE", 0.48), ("AT", 0.45), ("PL", 0.43), ("LU", 0.34)),
    },
    {
        "id": "bollard_red_cap_fr",
        "category": "bollards",
        "reason": "white cylinder bollard with red cap -> FR, MC",
        "label": "white cylinder bollard with red cap",
        "patterns": (
            r"\b(?:red cap|red top|red head|redcone|red cone)\b",
            r"\bwhite cylinder\b.*\bred\b",
            r"\bfrench bollard\b",
        ),
        "exclude": r"\byellow\b",
        "weights": (("FR", 0.52), ("MC", 0.42)),
    },
    {
        "id": "bollard_yellow_red_top_it",
        "category": "bollards",
        "reason": "yellow bollard with red top -> IT, SM",
        "label": "yellow bollard with red top",
        "patterns": (
            r"\byellow\b.*\bred\b.*\b(?:top|cap|band|strip|reflector|bollard)\b",
            r"\bred[- ]topped yellow\b",
            r"\bitalian bollard\b",
        ),
        "exclude": None,
        "weights": (("IT", 0.52), ("SM", 0.42)),
    },
    {
        "id": "bollard_wedge_striped_cz_sk",
        "category": "bollards",
        "reason": "black-and-white diagonally striped wedge bollard -> CZ, SK",
        "label": "black-and-white striped wedge bollard",
        "patterns": (
            r"\b(?:diagonal|striped|wedge)\b.*\b(?:black|white|orange dot|two dot)\b",
            r"\b(?:two|double) orange dots?\b",
            r"\bczech bollard\b",
            r"\bslovak bollard\b",
        ),
        "exclude": None,
        "weights": (("CZ", 0.50), ("SK", 0.47)),
    },
    {
        "id": "bollard_orange_band_dk_es",
        "category": "bollards",
        "reason": "white bollard with orange or yellow reflector band -> DK, ES",
        "label": "orange-band reflector bollard",
        "patterns": (
            r"\b(?:orange|amber)\b.*\b(?:band|reflector|strip)\b",
            r"\byellow reflector band\b",
            r"\bdanish bollard\b",
            r"\bspanish bollard\b",
        ),
        "exclude": r"\bblack top\b",
        "weights": (("DK", 0.48), ("ES", 0.45)),
    },
    {
        "id": "bollard_balkan_black_top_yellow",
        "category": "bollards",
        "reason": "white bollard with black top and yellow reflector -> SI, HR, RS",
        "label": "black-top bollard with yellow reflector",
        "patterns": (
            r"\bblack top\b.*\byellow\b",
            r"\bslovenian bollard\b",
            r"\bcroatian bollard\b",
        ),
        "exclude": None,
        "weights": (("SI", 0.47), ("HR", 0.45), ("RS", 0.42), ("ME", 0.38), ("BA", 0.38)),
    },
    {
        "id": "bollard_uk_black_reflective",
        "category": "bollards",
        "reason": "black bollard with wraparound reflective band -> GB, IE",
        "label": "black bollard with reflective band",
        "patterns": (
            r"\bblack bollard\b",
            r"\bblack\b.*\bwraparound\b",
            r"\b(?:uk|british) bollard\b",
            r"\bkeep left bollard\b",
        ),
        "exclude": None,
        "weights": (("GB", 0.50), ("IE", 0.42)),
    },
    {
        "id": "bollard_flat_white_au_nz",
        "category": "bollards",
        "reason": "flat white flexible bollard with red reflector -> AU, NZ",
        "label": "flat white flexible bollard",
        "patterns": (
            r"\bflat white\b.*\bred\b",
            r"\bflexible white post\b.*\bred\b",
            r"\baustralian bollard\b",
            r"\bnz bollard\b",
        ),
        "exclude": None,
        "weights": (("AU", 0.49), ("NZ", 0.46)),
    },
    {
        "id": "bollard_yellow_snow_pole",
        "category": "bollards",
        "reason": "tall yellow roadside snow pole -> JP, SE, FI, NO, IS",
        "label": "tall yellow snow pole",
        "patterns": (
            r"\bsnow (?:pole|stake|marker)s?\b",
            r"\btall yellow (?:pole|post|delineator)s?\b",
        ),
        "exclude": None,
        "weights": (("JP", 0.43), ("SE", 0.41), ("FI", 0.41), ("NO", 0.40), ("IS", 0.38)),
    },
    {
        "id": "bollard_iceland_yellow_white",
        "category": "bollards",
        "reason": "yellow bollard with white reflector -> IS",
        "label": "yellow bollard with white reflector",
        "patterns": (
            r"\byellow bollard\b.*\bwhite reflector\b",
            r"\bicelandic bollard\b",
        ),
        "exclude": None,
        "weights": (("IS", 0.54),),
    },

    # --- UTILITY POLES ------------------------------------------------------
    {
        "id": "pole_concrete_ladder_holes_fr_pl_hu",
        "category": "poles",
        "reason": "concrete pole with ladder holes -> FR, PL, HU",
        "label": "concrete pole with ladder holes",
        "patterns": (
            r"\bladder[- ]holes?\b",
            r"\b(?:holes?|perforated|slotted|cutouts?)\b.*\b(?:concrete|pole|post)\b",
            r"\b(?:concrete|pole|post)\b.*\b(?:holes?|perforated|slotted|ladder)\b",
            r"\bholy poles?\b",
        ),
        "exclude": None,
        "weights": (("FR", 0.50), ("PL", 0.47), ("HU", 0.45), ("RO", 0.36)),
    },
    {
        "id": "pole_yellow_black_stripes_tw_kr_jp",
        "category": "poles",
        "reason": "utility pole with yellow-and-black diagonal hazard stripes -> TW, KR, JP",
        "label": "pole with yellow-and-black hazard stripes",
        "patterns": (
            r"\byellow[- ]and[- ]black\b.*\b(?:stripe|diagonal|hazard|wrap|band|base)\b",
            r"\bblack[- ]and[- ]yellow\b.*\b(?:stripe|diagonal|hazard|wrap|band|base)\b",
            r"\b(?:diagonal|hazard) (?:yellow|black) stripes?\b",
        ),
        "exclude": None,
        "weights": (("TW", 0.50), ("KR", 0.47), ("JP", 0.43)),
    },
    {
        "id": "pole_round_concrete_step_rungs_jp_kr",
        "category": "poles",
        "reason": "round concrete pole with horizontal step bolts -> JP, KR, TW",
        "label": "round concrete pole with step bolts",
        "patterns": (
            r"\b(?:step bolts?|step rungs?|climbing rungs?|horizontal rungs?)\b",
            r"\bround concrete\b.*\b(?:rungs?|steps?|bolts?|plates?)\b",
            r"\bvertical number plate\b.*\bpole\b",
        ),
        "exclude": None,
        "weights": (("JP", 0.49), ("KR", 0.45), ("TW", 0.40)),
    },
    {
        "id": "pole_trident_crossbar_hu_ro",
        "category": "poles",
        "reason": "concrete utility pole with trident three-insulator top -> HU, RO, PL",
        "label": "trident crossbar concrete pole",
        "patterns": (
            r"\btrident\b",
            r"\bthree[- ]insulator\b",
            r"\bwishbone crossbar\b",
        ),
        "exclude": None,
        "weights": (("HU", 0.50), ("RO", 0.47), ("PL", 0.38)),
    },
    {
        "id": "pole_square_concrete_latam",
        "category": "poles",
        "reason": "square or octagonal concrete pole with recessed bars -> BR, AR, MX, CL",
        "label": "square concrete utility pole",
        "patterns": (
            r"\b(?:square|octagonal|rectangular) concrete\b",
            r"\bbrazilian ladder pole\b",
        ),
        "exclude": r"\bladder[- ]holes?\b",
        "weights": (("BR", 0.47), ("AR", 0.44), ("MX", 0.41), ("CL", 0.40), ("UY", 0.38)),
    },
    {
        "id": "pole_wooden_creosote",
        "category": "poles",
        "reason": "dark creosoted wooden utility pole -> GB, IE, NO, SE, FI, US, CA",
        "label": "wooden utility pole",
        "patterns": (
            r"\b(?:wooden|wood|timber|creosote|creosoted) (?:utility |telegraph |power )?poles?\b",
            r"\ba[- ]frame wooden\b",
        ),
        "exclude": None,
        "weights": (
            ("GB", 0.38),
            ("IE", 0.36),
            ("NO", 0.36),
            ("SE", 0.36),
            ("FI", 0.35),
            ("US", 0.35),
            ("CA", 0.34),
        ),
    },
    {
        "id": "pole_lattice_steel",
        "category": "poles",
        "reason": "latticed steel truss utility pole -> ES, PT, IT, CH, RU",
        "label": "latticed steel utility pole",
        "patterns": (
            r"\b(?:lattice|latticed|truss|metal grid)\b.*\b(?:steel|metal|pole|pylon)\b",
            r"\b(?:steel|metal) (?:lattice|truss)\b",
        ),
        "exclude": None,
        "weights": (("ES", 0.43), ("PT", 0.41), ("IT", 0.40), ("CH", 0.39), ("RU", 0.38)),
    },

    # --- GUARDRAIL ----------------------------------------------------------
    {
        "id": "guardrail_a_profile",
        "category": "guardrail",
        "reason": "rounded A-profile W-beam steel guardrail -> DE, FR, ES, IT, PL",
        "label": "rounded A-profile guardrail",
        "patterns": (
            r"\ba[- ]profile\b",
            r"\brounded (?:w[- ]beam|profile)\b",
        ),
        "exclude": None,
        "weights": (
            ("DE", 0.38),
            ("FR", 0.36),
            ("ES", 0.35),
            ("IT", 0.35),
            ("PL", 0.34),
            ("NL", 0.33),
            ("BE", 0.33),
        ),
    },
    {
        "id": "guardrail_b_profile",
        "category": "guardrail",
        "reason": "angular B-profile W-beam steel guardrail -> AT, CH, HU, CZ, SK",
        "label": "angular B-profile guardrail",
        "patterns": (
            r"\bb[- ]profile\b",
            r"\b(?:angular|boxy|squared) (?:w[- ]beam|profile)\b",
        ),
        "exclude": None,
        "weights": (
            ("AT", 0.42),
            ("CH", 0.40),
            ("HU", 0.40),
            ("CZ", 0.38),
            ("SK", 0.38),
            ("HR", 0.36),
            ("SI", 0.36),
        ),
    },
    {
        "id": "guardrail_wooden_log",
        "category": "guardrail",
        "reason": "wooden log or timber roadside guardrail -> CH, AT, NO, FR, CA",
        "label": "wooden log guardrail",
        "patterns": (
            r"\b(?:wooden|wood|timber|log) (?:guardrail|barrier|rail)\b",
            r"\bhalf[- ]round log\b",
        ),
        "exclude": None,
        "weights": (("CH", 0.45), ("AT", 0.42), ("NO", 0.40), ("FR", 0.38), ("CA", 0.36)),
    },
    {
        "id": "guardrail_wire_rope_cable",
        "category": "guardrail",
        "reason": "wire-rope cable guardrail -> SE, NO, FI, AU, NZ, GB",
        "label": "wire-rope cable barrier",
        "patterns": (
            r"\b(?:wire[- ]rope|cable|tension wire) (?:guardrail|barrier|fence)\b",
            r"\bcable barrier\b",
            r"\bwire[- ]rope\b",
        ),
        "exclude": None,
        "weights": (("SE", 0.46), ("NO", 0.42), ("FI", 0.42), ("AU", 0.40), ("NZ", 0.40), ("GB", 0.36)),
    },
    {
        "id": "guardrail_red_white_reflector",
        "category": "guardrail",
        "reason": "steel guardrail with red-and-white reflector tabs -> PL, DE, AT, CZ, HU",
        "label": "guardrail with red-and-white reflector tabs",
        "patterns": (
            r"\bred[- ]and[- ]white reflector\b",
            r"\bred reflector (?:tab|marker)s?\b",
        ),
        "exclude": None,
        "weights": (("PL", 0.40), ("DE", 0.38), ("AT", 0.38), ("CZ", 0.36), ("HU", 0.36), ("IT", 0.34)),
    },
    {
        "id": "guardrail_yellow_painted",
        "category": "guardrail",
        "reason": "yellow-painted steel guardrail -> HK, KR, CO, TW",
        "label": "yellow-painted guardrail",
        "patterns": (
            r"\byellow[- ]painted (?:steel )?guardrail\b",
            r"\byellow (?:steel |metal )?guardrail\b",
        ),
        "exclude": None,
        "weights": (("HK", 0.48), ("KR", 0.45), ("CO", 0.43), ("TW", 0.42)),
    },

    # --- ROAD LINES ---------------------------------------------------------
    {
        "id": "road_lines_yellow_outer_za_ie_nz",
        "category": "road_lines",
        "reason": "yellow outer road lines -> ZA, IE, NZ",
        "label": "yellow outer road lines",
        "patterns": (
            r"\byellow (?:dashed )?(?:outer|edge|shoulder|border|side|marginal) (?:road )?lines?\b",
            r"\b(?:outer|edge|shoulder) lines? (?:are )?yellow\b",
            r"\bdashed yellow (?:outer|edge|shoulder)\b",
            r"\bsolid yellow (?:outer|edge|shoulder)\b",
        ),
        "exclude": None,
        "weights": (
            ("ZA", 0.50),
            ("IE", 0.48),
            ("NZ", 0.46),
            ("BW", 0.38),
            ("LS", 0.36),
            ("SZ", 0.36),
            ("AE", 0.34),
        ),
    },
    {
        "id": "road_lines_white_dashed_edge",
        "category": "road_lines",
        "reason": "white dashed edge line -> DE, NL, SE, PL, AT",
        "label": "white dashed edge line",
        "patterns": (
            r"\bwhite dashed (?:outer|edge|side|shoulder) lines?\b",
            r"\bdashed white (?:outer|edge|side|shoulder) lines?\b",
            r"\bbroken white (?:outer|edge|side|shoulder) lines?\b",
        ),
        "exclude": None,
        "weights": (
            ("DE", 0.24),
            ("NL", 0.23),
            ("SE", 0.22),
            ("PL", 0.21),
            ("AT", 0.21),
            ("DK", 0.20),
            ("CH", 0.20),
        ),
    },
    {
        "id": "road_lines_double_yellow_center",
        "category": "road_lines",
        "reason": "double yellow center line -> US, CA, MX, BR, CO, KR",
        "label": "double yellow center line",
        "patterns": (
            r"\bdouble yellow (?:center|centre|middle|dividing)? ?lines?\b",
            r"\byellow (?:center|centre|centerline|centreline|middle) lines?\b",
        ),
        "exclude": r"\b(?:curb|kerb|parking|outer|edge)\b",
        "weights": (
            ("US", 0.45),
            ("CA", 0.43),
            ("MX", 0.40),
            ("BR", 0.40),
            ("CO", 0.38),
            ("KR", 0.36),
            ("PH", 0.35),
            ("NO", 0.34),
        ),
    },
    {
        "id": "road_lines_zigzag_uk",
        "category": "road_lines",
        "reason": "white or yellow zigzag pedestrian approach lines -> GB, IE, AU, NZ, SG",
        "label": "zigzag crossing approach lines",
        "patterns": (
            r"\bzig[- ]?zag\b",
            r"\bdouble yellow (?:curb|kerb|parking) lines?\b",
        ),
        "exclude": None,
        "weights": (("GB", 0.48), ("IE", 0.42), ("AU", 0.39), ("NZ", 0.39), ("SG", 0.38), ("HK", 0.36)),
    },
    {
        "id": "road_lines_red_cycle_lane_nl",
        "category": "road_lines",
        "reason": "red-surfaced cycle lane with white dashed edge -> NL, BE, DK, DE",
        "label": "red asphalt cycle lane",
        "patterns": (
            r"\bred (?:asphalt )?(?:cycle|bike|bicycle) lanes?\b",
            r"\bred[- ]surfaced (?:cycle|bike) lanes?\b",
        ),
        "exclude": None,
        "weights": (("NL", 0.50), ("BE", 0.44), ("DK", 0.40), ("DE", 0.36)),
    },
    {
        "id": "road_lines_blue_parking_zone",
        "category": "road_lines",
        "reason": "blue road parking or edge lines -> CH, AT, IT, ES, FR",
        "label": "blue parking road lines",
        "patterns": (
            r"\bblue (?:parking|edge|road|bay) (?:lines?|markings?)\b",
        ),
        "exclude": None,
        "weights": (("CH", 0.40), ("AT", 0.38), ("IT", 0.38), ("ES", 0.38), ("FR", 0.36)),
    },
    {
        "id": "road_lines_white_center_eu",
        "category": "road_lines",
        "reason": "white center road line -> DE, FR, PL, AT, IT, ES, NL",
        "label": "white center line",
        "patterns": (
            r"\bwhite (?:dashed )?(?:center|centre|centerline|centreline|middle) lines?\b",
            r"\ball[- ]white (?:road |lane )?(?:lines?|markings?)\b",
        ),
        "exclude": None,
        "weights": (
            ("DE", 0.22),
            ("FR", 0.22),
            ("PL", 0.21),
            ("AT", 0.21),
            ("IT", 0.20),
            ("ES", 0.20),
            ("NL", 0.20),
            ("GB", 0.19),
        ),
    },

    # --- CROSSWALK ----------------------------------------------------------
    {
        "id": "crosswalk_yellow_stripes_ch",
        "category": "crosswalk",
        "reason": "bright yellow pedestrian zebra stripes -> CH, HK, SG",
        "label": "yellow zebra crosswalk stripes",
        "patterns": (
            r"\byellow\b.*\b(?:zebra|crosswalk|pedestrian|crossing|stripes?)\b",
        ),
        "exclude": r"\b(?:yellow[- ]green|lime|fluorescent|diamond|sign)\b",
        "weights": (("CH", 0.50), ("HK", 0.40), ("SG", 0.38)),
    },
    {
        "id": "crosswalk_red_background_es_pt_be",
        "category": "crosswalk",
        "reason": "white crosswalk stripes on red-painted road surface -> ES, PT, BE, CH, FR",
        "label": "red-backed zebra crosswalk",
        "patterns": (
            r"\bred[- ](?:backed|painted|background|surface)\b.*\b(?:zebra|crosswalk|crossing|stripes?)\b",
            r"\b(?:zebra|crosswalk|crossing|stripes?)\b.*\bred (?:background|surface|band|paint)\b",
            r"\bred[- ]and[- ]white (?:zebra|crosswalk|crossing)\b",
        ),
        "exclude": None,
        "weights": (("ES", 0.45), ("PT", 0.43), ("BE", 0.42), ("CH", 0.40), ("FR", 0.38), ("NL", 0.36)),
    },
    {
        "id": "crosswalk_belisha_beacon_gb",
        "category": "crosswalk",
        "reason": "Belisha beacon amber globe on striped pole at crossing -> GB, IE, NZ, HK, SG",
        "label": "Belisha beacon pedestrian crossing",
        "patterns": (
            r"\bbelisha\b",
            r"\bamber globe\b",
            r"\bflashing (?:amber|orange|yellow) beacon\b",
        ),
        "exclude": None,
        "weights": (("GB", 0.52), ("IE", 0.46), ("NZ", 0.44), ("HK", 0.42), ("SG", 0.40), ("MT", 0.38)),
    },
    {
        "id": "crosswalk_ladder_bordered",
        "category": "crosswalk",
        "reason": "ladder-style crosswalk with longitudinal border lines -> US, CA, JP, KR, TW",
        "label": "ladder-style crosswalk",
        "patterns": (
            r"\bladder[- ]?(?:style )?(?:crosswalk|crossing)\b",
            r"\bcontinental crosswalk\b",
            r"\bbordered crosswalk\b",
        ),
        "exclude": None,
        "weights": (("US", 0.43), ("CA", 0.41), ("JP", 0.42), ("KR", 0.40), ("TW", 0.38)),
    },
    {
        "id": "crosswalk_four_stripes_sign_de_at_nl",
        "category": "crosswalk",
        "reason": "pedestrian crossing sign with four stripes -> DE, AT, NL, BE, CZ",
        "label": "four-stripe pedestrian sign",
        "patterns": (
            r"\b(?:4|four)[- ]stripes?\b",
        ),
        "exclude": None,
        "weights": (("DE", 0.44), ("AT", 0.42), ("NL", 0.40), ("BE", 0.38), ("CZ", 0.38), ("SK", 0.36)),
    },
    {
        "id": "crosswalk_five_stripes_sign_hr_si_it_es",
        "category": "crosswalk",
        "reason": "pedestrian crossing sign with five stripes -> HR, SI, RS, IT, ES, PL",
        "label": "five-stripe pedestrian sign",
        "patterns": (
            r"\b(?:5|five)[- ]stripes?\b",
        ),
        "exclude": None,
        "weights": (("HR", 0.44), ("SI", 0.42), ("RS", 0.42), ("IT", 0.40), ("ES", 0.40), ("PL", 0.40)),
    },
    {
        "id": "crosswalk_fluorescent_yellow_green_sign",
        "category": "crosswalk",
        "reason": "fluorescent yellow-green diamond pedestrian sign -> US, CA, AU, SE",
        "label": "fluorescent yellow-green pedestrian sign",
        "patterns": (
            r"\b(?:fluorescent|yellow[- ]green|lime[- ]green)\b.*\b(?:pedestrian|crossing|crosswalk|sign)\b",
        ),
        "exclude": None,
        "weights": (("US", 0.45), ("CA", 0.42), ("AU", 0.40), ("SE", 0.38)),
    },
    {
        "id": "crosswalk_borderless_white_zebra",
        "category": "crosswalk",
        "reason": "borderless white zebra crosswalk stripes -> DE, NL, AT, PL, SE",
        "label": "borderless white zebra crosswalk",
        "patterns": (
            r"\b(?:borderless|unbordered) white (?:zebra|crosswalk|stripes?)\b",
            r"\bwhite zebra (?:crossing|crosswalk|stripes?)\b",
        ),
        "exclude": r"\b(?:red|yellow|ladder)\b",
        "weights": (("DE", 0.32), ("NL", 0.31), ("AT", 0.30), ("PL", 0.30), ("SE", 0.28), ("DK", 0.28)),
    },

    # --- STREET-NAME SIGNS --------------------------------------------------
    {
        "id": "street_sign_blue_enamel_fr_be_nl",
        "category": "street_signs",
        "reason": "blue enamel street-name plaque with white text and inset border -> FR, BE, NL, PL, IT",
        "label": "blue enamel street-name plaque",
        "patterns": (
            r"\bblue enamel\b",
            r"\bblue (?:plaque|plate|street sign)\b.*\bwhite (?:text|letter|border|frame)\b",
            r"\bwhite text on blue (?:enamel|plaque|plate|street sign)\b",
        ),
        "exclude": None,
        "weights": (("FR", 0.47), ("BE", 0.43), ("NL", 0.41), ("PL", 0.39), ("IT", 0.36)),
    },
    {
        "id": "street_sign_uk_low_black_frame",
        "category": "street_signs",
        "reason": "low-mounted white street-name sign in black frame on short posts -> GB, IE",
        "label": "low-mounted white street sign with black border",
        "patterns": (
            r"\blow[- ]mounted\b.*\b(?:white|black|street)\b",
            r"\bshort (?:posts?|legs?)\b.*\b(?:white|black|street)\b",
            r"\bkerbside\b.*\bstreet sign\b",
            r"\bwhite sign\b.*\bblack frame\b.*\b(?:low|short|legs?|posts?|postcode)\b",
        ),
        "exclude": None,
        "weights": (("GB", 0.50), ("IE", 0.42)),
    },
    {
        "id": "street_sign_green_blade_mutcd",
        "category": "street_signs",
        "reason": "green blade street-name sign with white lettering -> US, CA, AU, NZ",
        "label": "green blade street-name sign",
        "patterns": (
            r"\bgreen blade\b",
            r"\bgreen (?:street|road)[- ]?(?:name )?signs?\b.*\bwhite\b",
            r"\bwhite (?:text|letters?|lettering) on green (?:blade|street sign)\b",
        ),
        "exclude": None,
        "weights": (("US", 0.47), ("CA", 0.44), ("AU", 0.41), ("NZ", 0.40), ("MY", 0.35)),
    },
    {
        "id": "street_sign_ceramic_tile_es_pt",
        "category": "street_signs",
        "reason": "ceramic tile (azulejo) wall-mounted street-name plaque -> ES, PT, IT, MX",
        "label": "ceramic tile street plaque",
        "patterns": (
            r"\b(?:ceramic|tiled?|azulejo|mosaic) (?:wall )?(?:street|plaque|sign)\b",
            r"\bazulejo\b",
        ),
        "exclude": None,
        "weights": (("ES", 0.50), ("PT", 0.48), ("IT", 0.38), ("MX", 0.36)),
    },
    {
        "id": "street_sign_red_border_cz_sk_ro",
        "category": "street_signs",
        "reason": "white street-name sign with red border -> CZ, SK, RO, DK",
        "label": "red-bordered white street sign",
        "patterns": (
            r"\bwhite (?:street|road)[- ]?(?:name )?signs?\b.*\bred (?:border|frame|rim)\b",
            r"\bred[- ]bordered (?:white )?street[- ]?(?:name )?signs?\b",
        ),
        "exclude": None,
        "weights": (("CZ", 0.48), ("SK", 0.46), ("RO", 0.42), ("DK", 0.40)),
    },
    {
        "id": "street_sign_vertical_blue_strip_jp",
        "category": "street_signs",
        "reason": "vertical narrow blue address block plaque on pole -> JP, KR, TW",
        "label": "vertical blue address strip",
        "patterns": (
            r"\bvertical (?:narrow )?blue\b",
            r"\bchome\b",
            r"\bblue vertical address\b",
        ),
        "exclude": None,
        "weights": (("JP", 0.52), ("KR", 0.42), ("TW", 0.38)),
    },
    {
        "id": "street_sign_german_white_black_fraktur",
        "category": "street_signs",
        "reason": "white street-name plate with black lettering -> DE, AT, DK, SE, FI",
        "label": "white street-name plate with black lettering",
        "patterns": (
            r"\b(?:fraktur|gothic) (?:script|font|lettering|text)\b",
            r"\b(?:strasse|straße)\b",
            r"\bwhite (?:rectangular )?(?:plate|blade|street sign) with black (?:text|lettering|font)\b",
        ),
        "exclude": r"\b(?:low[- ]mounted|short posts?|two legs?|kerbside)\b",
        "weights": (("DE", 0.46), ("AT", 0.43), ("DK", 0.36), ("SE", 0.35), ("FI", 0.34)),
    },
    {
        "id": "street_sign_bilingual",
        "category": "street_signs",
        "reason": "bilingual dual-language street-name sign -> IE, BE, CA, FI, CH, HK",
        "label": "bilingual street-name sign",
        "patterns": (
            r"\b(?:bilingual|dual[- ]language|two[- ]language)\b",
            r"\birish and english\b",
            r"\bfrench and dutch\b",
        ),
        "exclude": None,
        "weights": (("IE", 0.45), ("BE", 0.44), ("CA", 0.42), ("FI", 0.41), ("CH", 0.38), ("HK", 0.38)),
    },

    # --- CURBS --------------------------------------------------------------
    {
        "id": "curb_black_white_striped_ru_ua_my",
        "category": "curbs",
        "reason": "black-and-white striped curb -> RU, UA, KZ, MY, SG, ID",
        "label": "black-and-white striped curb",
        "patterns": (
            r"\bblack[- ]and[- ]white (?:striped |painted )?(?:curb|kerb)s?\b",
            r"\bwhite[- ]and[- ]black (?:striped |painted )?(?:curb|kerb)s?\b",
            r"\b(?:curb|kerb)s? painted black and white\b",
        ),
        "exclude": None,
        "weights": (("RU", 0.48), ("UA", 0.45), ("BY", 0.42), ("KZ", 0.42), ("MY", 0.40), ("SG", 0.40), ("ID", 0.38)),
    },
    {
        "id": "curb_yellow_black_striped_tw_kr_th",
        "category": "curbs",
        "reason": "yellow-and-black striped curb -> TW, KR, TH, MY, PH",
        "label": "yellow-and-black striped curb",
        "patterns": (
            r"\byellow[- ]and[- ]black (?:striped |painted )?(?:curb|kerb)s?\b",
            r"\bblack[- ]and[- ]yellow (?:striped |painted )?(?:curb|kerb)s?\b",
        ),
        "exclude": None,
        "weights": (("TW", 0.48), ("KR", 0.45), ("TH", 0.44), ("MY", 0.40), ("PH", 0.38), ("IN", 0.36)),
    },
    {
        "id": "curb_red_white_striped_tr_il",
        "category": "curbs",
        "reason": "red-and-white striped curb -> TR, IL, TH, ID, PE",
        "label": "red-and-white striped curb",
        "patterns": (
            r"\bred[- ]and[- ]white (?:striped |painted )?(?:curb|kerb)s?\b",
            r"\bwhite[- ]and[- ]red (?:striped |painted )?(?:curb|kerb)s?\b",
        ),
        "exclude": None,
        "weights": (("TR", 0.48), ("IL", 0.46), ("TH", 0.42), ("ID", 0.38), ("PE", 0.36)),
    },
    {
        "id": "curb_blue_white_es_br",
        "category": "curbs",
        "reason": "blue-and-white painted curb -> ES, BR, AR, IT",
        "label": "blue-and-white painted curb",
        "patterns": (
            r"\bblue[- ]and[- ]white (?:striped |painted )?(?:curb|kerb)s?\b",
            r"\bwhite[- ]and[- ]blue (?:striped |painted )?(?:curb|kerb)s?\b",
        ),
        "exclude": None,
        "weights": (("ES", 0.46), ("BR", 0.42), ("AR", 0.40), ("IT", 0.38)),
    },
    {
        "id": "curb_solid_red_us_ca",
        "category": "curbs",
        "reason": "solid red-painted no-parking curb -> US, CA",
        "label": "red-painted no-parking curb",
        "patterns": (
            r"\b(?:solid )?red[- ]painted (?:no[- ]parking )?(?:curb|kerb)s?\b",
            r"\bred no[- ]parking (?:curb|kerb)s?\b",
        ),
        "exclude": r"\bwhite\b",
        "weights": (("US", 0.46), ("CA", 0.40)),
    },
    {
        "id": "curb_cut_granite_eu",
        "category": "curbs",
        "reason": "cut granite stone curb -> DE, FR, IT, CH, AT, GB, PT, ES",
        "label": "cut granite stone curb",
        "patterns": (
            r"\b(?:cut )?granite (?:stone )?(?:curb|kerb)s?\b",
            r"\bbevelled stone (?:curb|kerb)s?\b",
        ),
        "exclude": None,
        "weights": (("DE", 0.30), ("FR", 0.30), ("IT", 0.30), ("CH", 0.29), ("AT", 0.28), ("GB", 0.28), ("PT", 0.28), ("ES", 0.28)),
    },

    # --- CHEVRONS -----------------------------------------------------------
    {
        "id": "chevron_black_on_yellow_gb_my_au",
        "category": "chevrons",
        "reason": "black-on-yellow chevron -> GB, MY, AU",
        "label": "black-on-yellow chevron",
        "patterns": (
            r"\bblack[- ]on[- ]yellow\b",
            r"\bblack arrows? on yellow\b",
            r"\byellow (?:chevron|sign|background)s? with black\b",
        ),
        "exclude": r"\byellow[- ]on[- ]black\b",
        "weights": (
            ("GB", 0.50),
            ("MY", 0.47),
            ("AU", 0.45),
            ("NZ", 0.39),
            ("IE", 0.38),
            ("US", 0.36),
            ("ZA", 0.35),
        ),
    },
    {
        "id": "chevron_yellow_on_black_es_pt_no",
        "category": "chevrons",
        "reason": "yellow-on-black chevron -> ES, PT, NO, IS, SE, FI",
        "label": "yellow-on-black chevron",
        "patterns": (
            r"\byellow[- ]on[- ]black\b",
            r"\byellow arrows? on black\b",
            r"\bblack (?:chevron|sign|background)s? with yellow\b",
        ),
        "exclude": r"\bblack[- ]on[- ]yellow\b",
        "weights": (("ES", 0.46), ("PT", 0.45), ("NO", 0.44), ("IS", 0.42), ("SE", 0.40), ("FI", 0.40)),
    },
    {
        "id": "chevron_white_on_black_gb_it_ch",
        "category": "chevrons",
        "reason": "white-on-black chevron -> GB, IT, CH, AL",
        "label": "white-on-black chevron",
        "patterns": (
            r"\bwhite[- ]on[- ]black\b",
            r"\bwhite arrows? on black\b",
            r"\bblack (?:chevron|sign|background)s? with white\b",
        ),
        "exclude": None,
        "weights": (("GB", 0.48), ("IT", 0.46), ("CH", 0.44), ("AL", 0.40), ("MY", 0.36)),
    },
    {
        "id": "chevron_red_on_white_de_at_pl_fr",
        "category": "chevrons",
        "reason": "red-on-white chevron -> DE, AT, PL, FR, CZ, HU",
        "label": "red-on-white chevron",
        "patterns": (
            r"\bred[- ]on[- ]white\b",
            r"\bred arrows? on white\b",
            r"\bwhite (?:chevron|sign|background)s? with red\b",
        ),
        "exclude": r"\bwhite[- ]on[- ]red\b",
        "weights": (
            ("DE", 0.46),
            ("AT", 0.45),
            ("PL", 0.44),
            ("FR", 0.42),
            ("CZ", 0.40),
            ("HU", 0.40),
            ("SK", 0.38),
            ("RO", 0.36),
        ),
    },
    {
        "id": "chevron_white_on_red_it_es_at",
        "category": "chevrons",
        "reason": "white-on-red chevron -> IT, ES, AT, BE, CZ",
        "label": "white-on-red chevron",
        "patterns": (
            r"\bwhite[- ]on[- ]red\b",
            r"\bwhite arrows? on red\b",
            r"\bred (?:chevron|sign|background)s? with white\b",
        ),
        "exclude": r"\bred[- ]on[- ]white\b",
        "weights": (("IT", 0.46), ("ES", 0.44), ("AT", 0.42), ("BE", 0.40), ("CZ", 0.38), ("HR", 0.36)),
    },
    {
        "id": "chevron_white_on_blue_fr_es_se",
        "category": "chevrons",
        "reason": "white-on-blue chevron -> FR, ES, SE, FI",
        "label": "white-on-blue chevron",
        "patterns": (
            r"\bwhite[- ]on[- ]blue\b",
            r"\byellow[- ]on[- ]blue\b",
            r"\bblue (?:chevron|sign|background)s? with (?:white|yellow)\b",
        ),
        "exclude": None,
        "weights": (("FR", 0.48), ("ES", 0.42), ("SE", 0.42), ("FI", 0.40)),
    },
]

# Pre-compile regex patterns for speed and reliability
_COMPILED_RULES: list[dict[str, Any]] = []
for _r in _RULES:
    _COMPILED_RULES.append({
        **_r,
        "_pats": tuple(re.compile(p, re.IGNORECASE) for p in _r["patterns"]),
        "_excl": re.compile(_r["exclude"], re.IGNORECASE) if _r.get("exclude") else None,
    })


# ---------------------------------------------------------------------------
# Observation extraction helpers
# ---------------------------------------------------------------------------


def _flatten_strings(val: Any) -> list[str]:
    """Recursively pull clean lowercase strings out of scalars, lists, or dicts."""
    out: list[str] = []
    if val is None or isinstance(val, bool):
        return out
    if isinstance(val, str):
        # Replace underscores with spaces so snake_case tokens match natural regexes
        s = re.sub(r"\s+", " ", val.replace("_", " ").strip().lower())
        if s and s not in _JUNK_VALUES:
            out.append(s)
        return out
    if isinstance(val, dict):
        # Support osint_features merged shape {"value": ..., "values": [...], "counts": {...}}
        if "values" in val and isinstance(val["values"], (list, tuple, set)):
            for item in val["values"]:
                out.extend(_flatten_strings(item))
            return out
        if "value" in val and val["value"] is not None:
            out.extend(_flatten_strings(val["value"]))
            return out
        parts: list[str] = []
        for k, v in val.items():
            if str(k).strip().lower() in ("sources", "counts", "frames", "confidence"):
                continue
            parts.extend(_flatten_strings(v))
        if parts:
            out.append(" ".join(parts))
            out.extend(parts)
        return out
    if isinstance(val, (list, tuple, set, frozenset)):
        for item in val:
            out.extend(_flatten_strings(item))
        return out
    return out


def _extract_category_texts(observations: Any) -> tuple[dict[str, list[str]], list[str]]:
    """Map raw observations into per-category text lists + wildcard texts."""
    by_cat: dict[str, list[str]] = {}
    wildcard: list[str] = []

    if isinstance(observations, str):
        wildcard.extend(_flatten_strings(observations))
        return by_cat, wildcard
    if isinstance(observations, (list, tuple, set, frozenset)):
        wildcard.extend(_flatten_strings(observations))
        return by_cat, wildcard
    if not isinstance(observations, dict):
        return by_cat, wildcard

    for raw_key, raw_val in observations.items():
        if not isinstance(raw_key, str):
            continue
        norm_key = re.sub(r"[^a-z0-9]+", "_", raw_key.strip().lower()).strip("_")
        texts = _flatten_strings(raw_val)
        if not texts:
            continue
        cat = _CATEGORY_ALIASES.get(norm_key)
        if cat:
            bucket = by_cat.setdefault(cat, [])
            for t in texts:
                if t not in bucket:
                    bucket.append(t)
        elif norm_key in _WILDCARD_KEYS:
            for t in texts:
                if t not in wildcard:
                    wildcard.append(t)
        else:
            # Unknown key: still allow exact rule matches across all categories
            for t in texts:
                if t not in wildcard:
                    wildcard.append(t)

    return by_cat, wildcard


def _norm_cc(cc: Any) -> Optional[str]:
    """Normalize a country code or country name to ISO-3166 alpha-2."""
    if not cc or not isinstance(cc, str):
        return None
    s = re.sub(r"\s+", " ", cc.strip().upper())
    if not s:
        return None
    if s in _CC_ALIASES:
        return _CC_ALIASES[s]
    if len(s) == 2 and s.isalpha():
        return s
    # Handle "US-CA" -> "US"
    if "-" in s:
        prefix = s.split("-", 1)[0].strip()
        if len(prefix) == 2 and prefix.isalpha():
            return _CC_ALIASES.get(prefix, prefix)
    return None


def _country_label(cc: str) -> str:
    """Format 'Germany (DE)' or 'DE' when country name is unknown."""
    name = _CC_NAMES.get(cc)
    return f"{name} ({cc})" if name else cc


def _clean_reason_text(reason: str) -> str:
    """Ensure a reason string has no internal variable names or arrow suffixes for prose."""
    text = str(reason or "").replace("_", " ").strip()
    if "->" in text:
        text = text.split("->", 1)[0].strip()
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_street_cues(observations: dict) -> list[dict]:
    """Inspect street-furniture & road infrastructure cues and score countries.

    Inspects `bollards`, `poles`, `guardrail`, `road_lines`, `crosswalk`,
    `street_signs`, `curbs`, and `chevrons` (plus common aliases and free-text
    lists) and returns a descending-score list of country dicts:

        [{"cc": "DE", "score": 0.72, "reasons": ["white bollard with red reflector band -> DE, AT, PL", ...]}]

    Never raises on any input.
    """
    try:
        by_cat, wildcard = _extract_category_texts(observations)
        if not by_cat and not wildcard:
            return []

        scores: dict[str, float] = {}
        reasons_by_cc: dict[str, list[str]] = {}
        first_rank: dict[str, int] = {}
        rank_counter = 0

        for rule in _COMPILED_RULES:
            cat = rule["category"]
            candidates = list(by_cat.get(cat, ())) + wildcard
            if not candidates:
                continue

            matched = False
            excl = rule["_excl"]
            for text in candidates:
                if excl is not None and excl.search(text):
                    continue
                if any(pat.search(text) for pat in rule["_pats"]):
                    matched = True
                    break

            if not matched:
                continue

            reason_str = rule["reason"]
            for cc, delta in rule["weights"]:
                scores[cc] = scores.get(cc, 0.0) + float(delta)
                r_list = reasons_by_cc.setdefault(cc, [])
                if reason_str not in r_list:
                    r_list.append(reason_str)
                if cc not in first_rank:
                    first_rank[cc] = rank_counter
                    rank_counter += 1

        if not scores:
            return []

        rows: list[dict] = []
        for cc, raw_score in scores.items():
            capped = round(min(0.95, max(0.01, raw_score)), 2)
            rows.append({
                "cc": cc,
                "score": capped,
                "reasons": list(reasons_by_cc.get(cc, ())),
            })

        rows.sort(key=lambda r: (-r["score"], first_rank.get(r["cc"], 9999), r["cc"]))
        return rows
    except Exception:
        return []


def street_cue_evidence(
    rows: list[dict],
    claimed_cc: str | None = None,
) -> Optional["Evidence"]:
    """Package scored street-infrastructure rows into an `Evidence(type="street_cues", ...)`.

    Returns `None` when no street cues were observed (`rows` is empty or has no
    positive scores). Otherwise produces a plain-English summary (with zero
    internal variable names) and a `ClaimVerdict` comparing the observed cues
    against `claimed_cc` when supplied.

    Never raises on any input.
    """
    try:
        ClaimVerdict, Evidence = _load_schema()

        # Tolerate a caller passing raw observation dict directly
        if isinstance(rows, dict) and "cc" not in rows:
            rows = score_street_cues(rows)
        elif isinstance(rows, dict) and "cc" in rows:
            rows = [rows]

        if not isinstance(rows, (list, tuple)):
            return None

        valid_rows: list[dict] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            cc_code = _norm_cc(item.get("cc"))
            if not cc_code:
                continue
            try:
                sc = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                sc = 0.0
            if sc <= 0.0:
                continue
            raw_reasons = item.get("reasons")
            if isinstance(raw_reasons, str):
                r_list = [raw_reasons] if raw_reasons.strip() else []
            elif isinstance(raw_reasons, (list, tuple, set)):
                r_list = [str(x).strip() for x in raw_reasons if str(x or "").strip()]
            else:
                r_list = []
            valid_rows.append({
                "cc": cc_code,
                "score": round(min(1.0, max(0.0, sc)), 2),
                "reasons": r_list,
            })

        if not valid_rows:
            return None

        valid_rows.sort(key=lambda r: (-r["score"], r["cc"]))
        top = valid_rows[0]
        top_cc = top["cc"]
        top_score = top["score"]

        lead_countries = ", ".join(_country_label(r["cc"]) for r in valid_rows[:3])
        clean_cues: list[str] = []
        for r in valid_rows[:3]:
            for reason in r.get("reasons") or []:
                cleaned = _clean_reason_text(reason)
                if cleaned and cleaned not in clean_cues:
                    clean_cues.append(cleaned)
        cue_phrase = "; ".join(clean_cues[:4]) if clean_cues else "observed road furniture"

        norm_claim = _norm_cc(claimed_cc)
        claimed_row = next((r for r in valid_rows if r["cc"] == norm_claim), None)
        claimed_rank = next(
            (idx for idx, r in enumerate(valid_rows) if r["cc"] == norm_claim),
            None,
        )
        claimed_score = claimed_row["score"] if claimed_row else 0.0

        data = {
            "rows": valid_rows,
            "top_cc": top_cc,
            "top_score": top_score,
            "claimed_cc": norm_claim,
            "claimed_score": claimed_score,
            "claimed_rank": claimed_rank,
        }

        if norm_claim is None:
            summary = (
                f"Street furniture and road infrastructure cues point toward "
                f"{lead_countries} (top score {top_score:.2f}) based on {cue_phrase}."
            )
            conf = round(min(0.75, max(0.25, top_score)), 2)
            return Evidence(
                type="street_cues",
                summary=summary.replace("_", " "),
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=conf,
                data=data,
            )

        claim_label = _country_label(norm_claim)
        if claimed_row is not None and (
            (claimed_rank is not None and claimed_rank < 3)
            or claimed_score >= 0.6 * top_score
        ):
            claim_cues = [
                _clean_reason_text(x)
                for x in (claimed_row.get("reasons") or [])
                if _clean_reason_text(x)
            ]
            claim_cue_text = "; ".join(claim_cues[:3]) if claim_cues else cue_phrase
            summary = (
                f"Street furniture and road infrastructure cues support "
                f"{claim_label} (score {claimed_score:.2f}) based on {claim_cue_text}."
            )
            conf = round(min(0.85, max(0.35, claimed_score)), 2)
            return Evidence(
                type="street_cues",
                summary=summary.replace("_", " "),
                verdict=ClaimVerdict.SUPPORTS,
                confidence=conf,
                data=data,
            )

        if top_score >= 0.40 and claimed_score < 0.40 * top_score:
            summary = (
                f"Street furniture and road infrastructure cues favor "
                f"{lead_countries} (top score {top_score:.2f}) over claimed "
                f"{claim_label} based on {cue_phrase}."
            )
            conf = round(min(0.85, max(0.40, top_score)), 2)
            return Evidence(
                type="street_cues",
                summary=summary.replace("_", " "),
                verdict=ClaimVerdict.CONTRADICTS,
                confidence=conf,
                data=data,
            )

        summary = (
            f"Street furniture and road infrastructure cues weakly suggest "
            f"{lead_countries} (score {top_score:.2f}) from {cue_phrase}, "
            f"which is inconclusive for claimed {claim_label}."
        )
        conf = round(min(0.45, max(0.20, top_score)), 2)
        return Evidence(
            type="street_cues",
            summary=summary.replace("_", " "),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=conf,
            data=data,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/street_cues.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    failures = 0

    def _check(name: str, cond: bool, detail: str = "") -> None:
        global failures
        if not cond:
            failures += 1
        status = "PASS" if cond else "FAIL"
        print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))

    print("1. score_street_cues:")

    # Test 1: German bollard + white dashed edge line -> DE top at 0.72
    r_de = score_street_cues({
        "bollards": "white bollard with red reflector band",
        "road_lines": "white dashed edge line",
    })
    _check(
        "white bollard with red reflector band + white dashed edge line -> DE #1 (0.72)",
        bool(r_de) and r_de[0]["cc"] == "DE" and abs(r_de[0]["score"] - 0.72) < 1e-6,
        str(r_de[:3]),
    )
    _check(
        "DE, AT, PL all scored on red-band reflector bollard",
        {"DE", "AT", "PL"}.issubset({row["cc"] for row in r_de}),
        str([row["cc"] for row in r_de]),
    )

    # Test 2: Concrete pole with ladder holes -> FR, PL, HU
    r_pole = score_street_cues({"poles": "concrete pole with ladder holes"})
    top3_pole = [row["cc"] for row in r_pole[:3]]
    _check(
        "concrete pole with ladder holes -> FR, PL, HU top 3",
        top3_pole == ["FR", "PL", "HU"],
        str(r_pole[:3]),
    )

    # Test 3: Yellow outer road lines -> ZA, IE, NZ
    r_lines = score_street_cues({"road_lines": "yellow outer road lines"})
    top3_lines = [row["cc"] for row in r_lines[:3]]
    _check(
        "yellow outer road lines -> ZA, IE, NZ top 3",
        top3_lines == ["ZA", "IE", "NZ"],
        str(r_lines[:3]),
    )

    # Test 4: Black-on-yellow chevron -> GB, MY, AU
    r_chev = score_street_cues({"chevrons": "black-on-yellow chevron"})
    top3_chev = [row["cc"] for row in r_chev[:3]]
    _check(
        "black-on-yellow chevron -> GB, MY, AU top 3",
        top3_chev == ["GB", "MY", "AU"],
        str(r_chev[:3]),
    )

    # Test 5: Guardrail + crosswalk + street_signs + curbs coverage
    r_multi = score_street_cues({
        "guardrail": "wooden log guardrail",
        "crosswalk": "yellow zebra crosswalk stripes",
        "street_signs": "bilingual street-name sign",
        "curbs": "cut granite stone curb",
    })
    _check(
        "wooden guardrail + yellow zebra + bilingual sign + granite curb -> CH #1",
        bool(r_multi) and r_multi[0]["cc"] == "CH",
        str(r_multi[:3]),
    )

    # Test 6: Empty / malformed inputs never raise and return []
    for bad in ({}, None, [], 123, {"bollards": None, "poles": ""}, {"unknown": [None, 42]}):
        res = score_street_cues(bad)  # type: ignore[arg-type]
        _check(f"malformed input {bad!r} -> []", res == [])

    print("2. street_cue_evidence:")

    ev_sup = street_cue_evidence(r_de, claimed_cc="DE")
    _check(
        "DE rows + claimed DE -> SUPPORTS with type='street_cues' and no underscores in summary",
        ev_sup is not None
        and ev_sup.type == "street_cues"
        and ev_sup.verdict.value == "supports"
        and "_" not in ev_sup.summary,
        f"{ev_sup.verdict.value if ev_sup else None}: {ev_sup.summary if ev_sup else None}",
    )

    ev_con = street_cue_evidence(r_de, claimed_cc="US")
    _check(
        "DE rows + claimed US -> CONTRADICTS",
        ev_con is not None
        and ev_con.verdict.value == "contradicts"
        and "_" not in ev_con.summary,
        f"{ev_con.verdict.value if ev_con else None}: {ev_con.summary if ev_con else None}",
    )

    ev_inc = street_cue_evidence(r_pole, claimed_cc=None)
    _check(
        "FR/PL/HU rows + claimed None -> INCONCLUSIVE",
        ev_inc is not None
        and ev_inc.verdict.value == "inconclusive"
        and "_" not in ev_inc.summary,
        f"{ev_inc.verdict.value if ev_inc else None}: {ev_inc.summary if ev_inc else None}",
    )

    _check("empty rows -> None", street_cue_evidence([]) is None)
    _check("None rows -> None", street_cue_evidence(None) is None)  # type: ignore[arg-type]

    print(f"self-test: {'all checks passed' if failures == 0 else f'{failures} check(s) FAILED'}")
    sys.exit(0 if failures == 0 else 1)
