"""Hierarchical coarse-to-fine geolocation narrowing (continent -> subregion -> country -> region).

Combines broad visual/OSINT observations (driving_side, script, climate_zone,
architecture, road_lines, sun_side, vegetation, sign_lang, etc.) with optional
per-country cue scores to narrow location hypotheses hierarchically:
continent -> subregion -> country -> region.

Pure stdlib at import time; sibling schema types are imported lazily with
fallback. Every public function is wrapped in try/except and never raises.
"""
from __future__ import annotations

import math
import re
from typing import Any


class _ContinentEntry(dict):
    """Dict representing a continent's metadata, countries, and subregions.

    Supports all common access patterns:
      - ``CONTINENTS["EU"]["countries"]`` -> list of ISO-2 codes
      - ``CONTINENTS["EU"]["subregions"]`` -> dict of subregion -> ISO-2 list
      - ``CONTINENTS["EU"]["western_europe"]`` -> ISO-2 list for subregion
      - ``"DE" in CONTINENTS["EU"]`` -> True
      - ``"western_europe" in CONTINENTS["EU"]`` -> True
    """

    def __missing__(self, key: str) -> Any:
        subs = self.get("subregions")
        if isinstance(subs, dict) and key in subs:
            return subs[key]
        raise KeyError(key)

    def __contains__(self, item: object) -> bool:
        if super().__contains__(item):
            return True
        if item in self.get("countries", ()):
            return True
        subs = self.get("subregions")
        if isinstance(subs, dict) and item in subs:
            return True
        return False


class _HierLevel(dict):
    """Dict representing a hierarchy level (`continent` or `subregion`).

    Behaves as a standard dict (`{"code": ..., "name": ..., "confidence": ...}`)
    while also comparing equal to its string code/name so callers doing either
    `hier["continent"]["code"]` or `hier["continent"] == "EU"` both succeed.
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            norm = other.strip().lower()
            for k in ("code", "name", "label", "id"):
                val = self.get(k)
                if isinstance(val, str) and val.strip().lower() == norm:
                    return True
            return False
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return str(self.get("code") or self.get("name") or "")


# ---------------------------------------------------------------------------
# 1. CONTINENTS & Subregion Definitions
# ---------------------------------------------------------------------------

SUBREGION_LABELS: dict[str, str] = {
    "northern_europe": "Northern Europe",
    "western_europe": "Western Europe",
    "southern_europe": "Southern Europe",
    "eastern_europe": "Eastern Europe",
    "british_isles": "British Isles",
    "east_asia": "East Asia",
    "southeast_asia": "Southeast Asia",
    "south_asia": "South Asia",
    "middle_east": "Middle East",
    "north_america": "North America",
    "central_america": "Central America",
    "south_america": "South America",
    "north_africa": "North Africa",
    "sub_saharan_africa": "Sub-Saharan Africa",
    "oceania": "Oceania",
}

CONTINENT_NAMES: dict[str, str] = {
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "AS": "Asia",
    "AF": "Africa",
    "OC": "Oceania",
}

_RAW_CONTINENTS: dict[str, dict[str, list[str]]] = {
    "EU": {
        "northern_europe": [
            "NO", "SE", "FI", "DK", "IS", "EE", "LV", "LT", "FO", "SJ", "AX",
        ],
        "western_europe": [
            "DE", "FR", "NL", "BE", "LU", "AT", "CH", "LI", "MC",
        ],
        "southern_europe": [
            "IT", "ES", "PT", "GR", "HR", "SI", "MT", "CY", "AL", "ME",
            "MK", "BA", "RS", "AD", "SM", "VA", "GI", "XK",
        ],
        "eastern_europe": [
            "PL", "CZ", "SK", "HU", "RO", "BG", "UA", "BY", "MD", "RU",
        ],
        "british_isles": [
            "GB", "IE", "IM", "JE", "GG",
        ],
    },
    "NA": {
        "north_america": [
            "US", "CA", "MX", "GL", "BM", "PM",
        ],
        "central_america": [
            "GT", "BZ", "HN", "SV", "NI", "CR", "PA", "CU", "JM", "HT",
            "DO", "PR", "BS", "TT", "BB", "LC", "GD", "VC", "AG", "DM",
            "KN", "AW", "CW", "KY", "VI",
        ],
    },
    "SA": {
        "south_america": [
            "BR", "AR", "CL", "CO", "PE", "VE", "EC", "BO", "PY", "UY",
            "GY", "SR", "GF", "FK",
        ],
    },
    "AS": {
        "east_asia": [
            "JP", "KR", "CN", "TW", "HK", "MO", "MN", "KP",
        ],
        "southeast_asia": [
            "TH", "VN", "ID", "MY", "SG", "PH", "KH", "LA", "MM", "BN", "TL",
        ],
        "south_asia": [
            "IN", "PK", "BD", "LK", "NP", "BT", "MV", "AF",
        ],
        "middle_east": [
            "TR", "IL", "PS", "JO", "LB", "SY", "IQ", "IR", "SA", "AE",
            "QA", "KW", "BH", "OM", "YE", "GE", "AM", "AZ", "KZ", "UZ",
            "KG", "TJ", "TM",
        ],
    },
    "AF": {
        "north_africa": [
            "MA", "DZ", "TN", "LY", "EG", "SD", "EH",
        ],
        "sub_saharan_africa": [
            "ZA", "KE", "NG", "GH", "TZ", "UG", "ET", "SN", "CI", "CM",
            "ZW", "ZM", "BW", "NA", "MZ", "RW", "MG", "MU", "AO", "CD",
            "CG", "ML", "BF", "NE", "TD", "SO", "DJ", "ER", "MW", "LS",
            "SZ", "GM", "SL", "LR", "TG", "BJ", "GA", "GQ", "CV", "SC",
            "KM", "BI", "SS", "CF", "GN", "GW", "MR", "ST", "RE", "YT",
        ],
    },
    "OC": {
        "oceania": [
            "AU", "NZ", "FJ", "PG", "SB", "VU", "NC", "PF", "WS", "TO",
            "KI", "FM", "MH", "PW", "NR", "TV", "GU", "AS", "MP", "CK",
            "NU", "NF",
        ],
    },
}

CONTINENTS: dict[str, _ContinentEntry] = {}
SUBREGIONS: dict[str, list[str]] = {}
CC_TO_CONTINENT: dict[str, str] = {}
CC_TO_SUBREGION: dict[str, str] = {}
SUBREGION_TO_CONTINENT: dict[str, str] = {}

for _cont_code, _subs in _RAW_CONTINENTS.items():
    _all_ccs: list[str] = []
    for _sub_name, _ccs in _subs.items():
        SUBREGIONS[_sub_name] = list(_ccs)
        SUBREGION_TO_CONTINENT[_sub_name] = _cont_code
        for _cc in _ccs:
            if _cc not in _all_ccs:
                _all_ccs.append(_cc)
            CC_TO_CONTINENT[_cc] = _cont_code
            CC_TO_SUBREGION[_cc] = _sub_name
    CONTINENTS[_cont_code] = _ContinentEntry(
        code=_cont_code,
        name=CONTINENT_NAMES[_cont_code],
        countries=_all_ccs,
        subregions={k: list(v) for k, v in _subs.items()},
    )

COUNTRY_NAMES: dict[str, str] = {
    "DE": "Germany", "FR": "France", "NL": "Netherlands", "BE": "Belgium",
    "LU": "Luxembourg", "AT": "Austria", "CH": "Switzerland", "LI": "Liechtenstein",
    "MC": "Monaco", "GB": "United Kingdom", "IE": "Ireland", "IM": "Isle of Man",
    "JE": "Jersey", "GG": "Guernsey", "NO": "Norway", "SE": "Sweden",
    "FI": "Finland", "DK": "Denmark", "IS": "Iceland", "EE": "Estonia",
    "LV": "Latvia", "LT": "Lithuania", "FO": "Faroe Islands", "IT": "Italy",
    "ES": "Spain", "PT": "Portugal", "GR": "Greece", "HR": "Croatia",
    "SI": "Slovenia", "MT": "Malta", "CY": "Cyprus", "AL": "Albania",
    "ME": "Montenegro", "MK": "North Macedonia", "BA": "Bosnia and Herzegovina",
    "RS": "Serbia", "AD": "Andorra", "SM": "San Marino", "VA": "Vatican City",
    "XK": "Kosovo", "PL": "Poland", "CZ": "Czechia", "SK": "Slovakia",
    "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria", "UA": "Ukraine",
    "BY": "Belarus", "MD": "Moldova", "RU": "Russia", "US": "United States",
    "CA": "Canada", "MX": "Mexico", "GL": "Greenland", "BM": "Bermuda",
    "GT": "Guatemala", "BZ": "Belize", "HN": "Honduras", "SV": "El Salvador",
    "NI": "Nicaragua", "CR": "Costa Rica", "PA": "Panama", "CU": "Cuba",
    "JM": "Jamaica", "HT": "Haiti", "DO": "Dominican Republic", "PR": "Puerto Rico",
    "BS": "Bahamas", "TT": "Trinidad and Tobago", "BB": "Barbados",
    "BR": "Brazil", "AR": "Argentina", "CL": "Chile", "CO": "Colombia",
    "PE": "Peru", "VE": "Venezuela", "EC": "Ecuador", "BO": "Bolivia",
    "PY": "Paraguay", "UY": "Uruguay", "GY": "Guyana", "SR": "Suriname",
    "GF": "French Guiana", "FK": "Falkland Islands", "JP": "Japan",
    "KR": "South Korea", "CN": "China", "TW": "Taiwan", "HK": "Hong Kong",
    "MO": "Macau", "MN": "Mongolia", "KP": "North Korea", "TH": "Thailand",
    "VN": "Vietnam", "ID": "Indonesia", "MY": "Malaysia", "SG": "Singapore",
    "PH": "Philippines", "KH": "Cambodia", "LA": "Laos", "MM": "Myanmar",
    "BN": "Brunei", "TL": "Timor-Leste", "IN": "India", "PK": "Pakistan",
    "BD": "Bangladesh", "LK": "Sri Lanka", "NP": "Nepal", "BT": "Bhutan",
    "MV": "Maldives", "AF": "Afghanistan", "TR": "Turkey", "IL": "Israel",
    "PS": "Palestine", "JO": "Jordan", "LB": "Lebanon", "SY": "Syria",
    "IQ": "Iraq", "IR": "Iran", "SA": "Saudi Arabia", "AE": "United Arab Emirates",
    "QA": "Qatar", "KW": "Kuwait", "BH": "Bahrain", "OM": "Oman", "YE": "Yemen",
    "GE": "Georgia", "AM": "Armenia", "AZ": "Azerbaijan", "KZ": "Kazakhstan",
    "UZ": "Uzbekistan", "KG": "Kyrgyzstan", "TJ": "Tajikistan", "TM": "Turkmenistan",
    "MA": "Morocco", "DZ": "Algeria", "TN": "Tunisia", "LY": "Libya",
    "EG": "Egypt", "SD": "Sudan", "ZA": "South Africa", "KE": "Kenya",
    "NG": "Nigeria", "GH": "Ghana", "TZ": "Tanzania", "UG": "Uganda",
    "ET": "Ethiopia", "SN": "Senegal", "CI": "Cote d'Ivoire", "CM": "Cameroon",
    "ZW": "Zimbabwe", "ZM": "Zambia", "BW": "Botswana", "NA": "Namibia",
    "MZ": "Mozambique", "RW": "Rwanda", "MG": "Madagascar", "MU": "Mauritius",
    "AO": "Angola", "CD": "DR Congo", "CG": "Republic of the Congo",
    "AU": "Australia", "NZ": "New Zealand", "FJ": "Fiji", "PG": "Papua New Guinea",
    "SB": "Solomon Islands", "VU": "Vanuatu", "WS": "Samoa", "TO": "Tonga",
}

_CC_ALIASES: dict[str, str] = {
    "UK": "GB", "EL": "GR", "USA": "US", "UAE": "AE", "SU": "RU", "TP": "TL",
}

# Build reverse lookup from lowercase country name -> ISO-2 code
_NAME_TO_CC: dict[str, str] = {
    v.lower(): k for k, v in COUNTRY_NAMES.items()
}
_NAME_TO_CC.update({
    "uk": "GB", "great britain": "GB", "britain": "GB", "england": "GB",
    "scotland": "GB", "wales": "GB", "usa": "US", "united states of america": "US",
    "america": "US", "holland": "NL", "the netherlands": "NL",
    "czech republic": "CZ", "korea": "KR", "uae": "AE",
})


# ---------------------------------------------------------------------------
# Cue -> Subregion / Country / Hemisphere Knowledge Tables
# ---------------------------------------------------------------------------

_LEFT_DRIVE_CCS: set[str] = {
    "GB", "IE", "IM", "JE", "GG", "MT", "CY",
    "JP", "HK", "MO",
    "TH", "ID", "MY", "SG", "BN", "TL",
    "IN", "PK", "BD", "LK", "NP", "BT", "MV",
    "AU", "NZ", "FJ", "PG", "SB", "VU", "WS", "TO", "KI", "NR", "TV",
    "ZA", "KE", "TZ", "UG", "ZW", "ZM", "BW", "NA", "MZ", "MW", "LS", "SZ", "MU", "SC",
    "JM", "TT", "BB", "BS", "GY", "SR", "LC", "GD", "VC", "AG", "DM", "KN", "VI", "KY",
}

_SOUTHERN_HEMISPHERE_CCS: set[str] = {
    "AU", "NZ", "FJ", "PG", "SB", "VU", "WS", "TO", "NC", "PF",
    "AR", "CL", "UY", "PY", "BO", "PE", "BR", "EC", "FK",
    "ZA", "NA", "BW", "ZW", "ZM", "MZ", "MW", "LS", "SZ", "MG", "MU", "AO", "TZ", "RW", "BI",
    "ID", "TL",
}

_EQUATORIAL_CCS: set[str] = {
    "CO", "EC", "BR", "PE", "VE", "GY", "SR", "PA", "CR",
    "KE", "UG", "TZ", "CD", "CG", "GA", "GQ", "CM", "NG", "GH", "CI", "ET", "SO",
    "ID", "MY", "SG", "BN", "TL", "PH", "TH", "VN", "KH", "LK", "MV", "PG",
}

# Script -> (subregion boosts, specific country boosts)
_SCRIPT_HINTS: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "latin": (
        {
            "western_europe": 0.45, "northern_europe": 0.40, "southern_europe": 0.40,
            "british_isles": 0.45, "eastern_europe": 0.30, "north_america": 0.45,
            "central_america": 0.40, "south_america": 0.45, "oceania": 0.45,
            "sub_saharan_africa": 0.30, "southeast_asia": 0.25,
        },
        {},
    ),
    "cyrillic": (
        {"eastern_europe": 0.90, "middle_east": 0.35, "east_asia": 0.20},
        {"RU": 0.85, "UA": 0.80, "BG": 0.78, "BY": 0.75, "RS": 0.72, "MK": 0.70, "KZ": 0.68, "KG": 0.65, "MN": 0.60, "ME": 0.55},
    ),
    "arabic": (
        {"middle_east": 0.88, "north_africa": 0.88, "south_asia": 0.30},
        {"SA": 0.82, "AE": 0.80, "EG": 0.82, "MA": 0.78, "JO": 0.78, "DZ": 0.75, "TN": 0.75, "QA": 0.75, "KW": 0.72, "OM": 0.72, "IQ": 0.70, "LB": 0.70, "IR": 0.68, "PK": 0.60},
    ),
    "japanese": (
        {"east_asia": 0.95},
        {"JP": 0.96},
    ),
    "kanji": (
        {"east_asia": 0.90},
        {"JP": 0.88, "CN": 0.70, "TW": 0.72, "HK": 0.68},
    ),
    "hiragana": ({"east_asia": 0.96}, {"JP": 0.97}),
    "katakana": ({"east_asia": 0.96}, {"JP": 0.97}),
    "korean": ({"east_asia": 0.95}, {"KR": 0.96, "KP": 0.40}),
    "hangul": ({"east_asia": 0.95}, {"KR": 0.96, "KP": 0.40}),
    "chinese": (
        {"east_asia": 0.92, "southeast_asia": 0.30},
        {"CN": 0.90, "TW": 0.85, "HK": 0.80, "SG": 0.55, "MO": 0.70},
    ),
    "hanzi": (
        {"east_asia": 0.92},
        {"CN": 0.90, "TW": 0.85, "HK": 0.80},
    ),
    "thai": ({"southeast_asia": 0.94}, {"TH": 0.96, "LA": 0.45}),
    "khmer": ({"southeast_asia": 0.94}, {"KH": 0.96}),
    "lao": ({"southeast_asia": 0.92}, {"LA": 0.95}),
    "burmese": ({"southeast_asia": 0.92}, {"MM": 0.95}),
    "devanagari": ({"south_asia": 0.94}, {"IN": 0.92, "NP": 0.85}),
    "hindi": ({"south_asia": 0.92}, {"IN": 0.92}),
    "bengali": ({"south_asia": 0.94}, {"BD": 0.92, "IN": 0.70}),
    "sinhala": ({"south_asia": 0.94}, {"LK": 0.96}),
    "tamil": ({"south_asia": 0.85, "southeast_asia": 0.40}, {"IN": 0.82, "LK": 0.82, "SG": 0.55, "MY": 0.50}),
    "greek": ({"southern_europe": 0.94}, {"GR": 0.95, "CY": 0.82}),
    "hebrew": ({"middle_east": 0.94}, {"IL": 0.96}),
    "georgian": ({"middle_east": 0.90}, {"GE": 0.96}),
    "armenian": ({"middle_east": 0.90}, {"AM": 0.96}),
    "ethiopic": ({"sub_saharan_africa": 0.90}, {"ET": 0.95, "ER": 0.80}),
}

# Language token -> (subregion boosts, country boosts)
_LANG_HINTS: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "de": ({"western_europe": 0.90}, {"DE": 0.92, "AT": 0.82, "CH": 0.78, "LU": 0.60, "LI": 0.65}),
    "german": ({"western_europe": 0.90}, {"DE": 0.92, "AT": 0.82, "CH": 0.78, "LU": 0.60, "LI": 0.65}),
    "fr": (
        {"western_europe": 0.85, "north_america": 0.35, "north_africa": 0.40, "sub_saharan_africa": 0.35},
        {"FR": 0.92, "BE": 0.75, "CH": 0.72, "LU": 0.68, "MC": 0.70, "CA": 0.55, "SN": 0.50, "CI": 0.50, "MA": 0.45, "TN": 0.45},
    ),
    "french": (
        {"western_europe": 0.85, "north_america": 0.35, "north_africa": 0.40, "sub_saharan_africa": 0.35},
        {"FR": 0.92, "BE": 0.75, "CH": 0.72, "LU": 0.68, "MC": 0.70, "CA": 0.55, "SN": 0.50, "CI": 0.50},
    ),
    "nl": ({"western_europe": 0.88}, {"NL": 0.92, "BE": 0.80, "SR": 0.45}),
    "dutch": ({"western_europe": 0.88}, {"NL": 0.92, "BE": 0.80, "SR": 0.45}),
    "en": (
        {"british_isles": 0.80, "north_america": 0.80, "oceania": 0.80, "sub_saharan_africa": 0.45, "south_asia": 0.35, "southeast_asia": 0.35},
        {"GB": 0.85, "US": 0.85, "CA": 0.78, "AU": 0.82, "NZ": 0.78, "IE": 0.78, "ZA": 0.65, "SG": 0.60, "IN": 0.50, "PH": 0.50, "KE": 0.50, "NG": 0.50, "GH": 0.50},
    ),
    "english": (
        {"british_isles": 0.80, "north_america": 0.80, "oceania": 0.80, "sub_saharan_africa": 0.45},
        {"GB": 0.85, "US": 0.85, "CA": 0.78, "AU": 0.82, "NZ": 0.78, "IE": 0.78, "ZA": 0.65, "SG": 0.60},
    ),
    "es": (
        {"southern_europe": 0.80, "south_america": 0.82, "central_america": 0.80, "north_america": 0.65},
        {"ES": 0.88, "MX": 0.85, "AR": 0.82, "CO": 0.80, "CL": 0.80, "PE": 0.78, "EC": 0.74, "UY": 0.72, "CR": 0.72, "GT": 0.70, "PA": 0.70, "DO": 0.68},
    ),
    "spanish": (
        {"southern_europe": 0.80, "south_america": 0.82, "central_america": 0.80, "north_america": 0.65},
        {"ES": 0.88, "MX": 0.85, "AR": 0.82, "CO": 0.80, "CL": 0.80, "PE": 0.78, "EC": 0.74, "UY": 0.72},
    ),
    "pt": (
        {"southern_europe": 0.80, "south_america": 0.85, "sub_saharan_africa": 0.35},
        {"PT": 0.88, "BR": 0.90, "AO": 0.50, "MZ": 0.50},
    ),
    "portuguese": (
        {"southern_europe": 0.80, "south_america": 0.85, "sub_saharan_africa": 0.35},
        {"PT": 0.88, "BR": 0.90, "AO": 0.50, "MZ": 0.50},
    ),
    "it": ({"southern_europe": 0.90, "western_europe": 0.35}, {"IT": 0.94, "CH": 0.60, "SM": 0.70, "VA": 0.65}),
    "italian": ({"southern_europe": 0.90, "western_europe": 0.35}, {"IT": 0.94, "CH": 0.60, "SM": 0.70}),
    "pl": ({"eastern_europe": 0.90}, {"PL": 0.95}),
    "polish": ({"eastern_europe": 0.90}, {"PL": 0.95}),
    "cs": ({"eastern_europe": 0.88}, {"CZ": 0.94, "SK": 0.65}),
    "czech": ({"eastern_europe": 0.88}, {"CZ": 0.94, "SK": 0.65}),
    "sk": ({"eastern_europe": 0.88}, {"SK": 0.94, "CZ": 0.65}),
    "slovak": ({"eastern_europe": 0.88}, {"SK": 0.94, "CZ": 0.65}),
    "hu": ({"eastern_europe": 0.90}, {"HU": 0.95}),
    "hungarian": ({"eastern_europe": 0.90}, {"HU": 0.95}),
    "ro": ({"eastern_europe": 0.90}, {"RO": 0.94, "MD": 0.80}),
    "romanian": ({"eastern_europe": 0.90}, {"RO": 0.94, "MD": 0.80}),
    "ru": ({"eastern_europe": 0.88, "middle_east": 0.35}, {"RU": 0.92, "BY": 0.78, "KZ": 0.70, "KG": 0.65, "UA": 0.60}),
    "russian": ({"eastern_europe": 0.88, "middle_east": 0.35}, {"RU": 0.92, "BY": 0.78, "KZ": 0.70, "KG": 0.65}),
    "uk": ({"eastern_europe": 0.90}, {"UA": 0.95}),
    "ukrainian": ({"eastern_europe": 0.90}, {"UA": 0.95}),
    "bg": ({"eastern_europe": 0.90}, {"BG": 0.95}),
    "bulgarian": ({"eastern_europe": 0.90}, {"BG": 0.95}),
    "sv": ({"northern_europe": 0.90}, {"SE": 0.94, "FI": 0.55}),
    "swedish": ({"northern_europe": 0.90}, {"SE": 0.94, "FI": 0.55}),
    "no": ({"northern_europe": 0.90}, {"NO": 0.95}),
    "norwegian": ({"northern_europe": 0.90}, {"NO": 0.95}),
    "da": ({"northern_europe": 0.90}, {"DK": 0.95}),
    "danish": ({"northern_europe": 0.90}, {"DK": 0.95}),
    "fi": ({"northern_europe": 0.90}, {"FI": 0.95}),
    "finnish": ({"northern_europe": 0.90}, {"FI": 0.95}),
    "is": ({"northern_europe": 0.90}, {"IS": 0.96}),
    "icelandic": ({"northern_europe": 0.90}, {"IS": 0.96}),
    "et": ({"northern_europe": 0.90}, {"EE": 0.95}),
    "estonian": ({"northern_europe": 0.90}, {"EE": 0.95}),
    "lv": ({"northern_europe": 0.90}, {"LV": 0.95}),
    "latvian": ({"northern_europe": 0.90}, {"LV": 0.95}),
    "lt": ({"northern_europe": 0.90}, {"LT": 0.95}),
    "lithuanian": ({"northern_europe": 0.90}, {"LT": 0.95}),
    "el": ({"southern_europe": 0.92}, {"GR": 0.95, "CY": 0.80}),
    "greek": ({"southern_europe": 0.92}, {"GR": 0.95, "CY": 0.80}),
    "hr": ({"southern_europe": 0.88}, {"HR": 0.94, "BA": 0.70}),
    "croatian": ({"southern_europe": 0.88}, {"HR": 0.94, "BA": 0.70}),
    "sl": ({"southern_europe": 0.90}, {"SI": 0.95}),
    "slovenian": ({"southern_europe": 0.90}, {"SI": 0.95}),
    "sr": ({"southern_europe": 0.88}, {"RS": 0.94, "ME": 0.75, "BA": 0.70}),
    "serbian": ({"southern_europe": 0.88}, {"RS": 0.94, "ME": 0.75, "BA": 0.70}),
    "sq": ({"southern_europe": 0.90}, {"AL": 0.94, "XK": 0.82}),
    "albanian": ({"southern_europe": 0.90}, {"AL": 0.94, "XK": 0.82}),
    "tr": ({"middle_east": 0.90}, {"TR": 0.95, "CY": 0.65}),
    "turkish": ({"middle_east": 0.90}, {"TR": 0.95, "CY": 0.65}),
    "he": ({"middle_east": 0.92}, {"IL": 0.96}),
    "hebrew": ({"middle_east": 0.92}, {"IL": 0.96}),
    "ar": (
        {"middle_east": 0.88, "north_africa": 0.88},
        {"SA": 0.85, "AE": 0.82, "EG": 0.85, "MA": 0.80, "JO": 0.80, "DZ": 0.78, "TN": 0.78, "QA": 0.76, "KW": 0.74, "OM": 0.74},
    ),
    "arabic": (
        {"middle_east": 0.88, "north_africa": 0.88},
        {"SA": 0.85, "AE": 0.82, "EG": 0.85, "MA": 0.80, "JO": 0.80, "DZ": 0.78, "TN": 0.78},
    ),
    "fa": ({"middle_east": 0.90}, {"IR": 0.95, "AF": 0.70, "TJ": 0.65}),
    "persian": ({"middle_east": 0.90}, {"IR": 0.95, "AF": 0.70}),
    "farsi": ({"middle_east": 0.90}, {"IR": 0.95}),
    "ja": ({"east_asia": 0.95}, {"JP": 0.97}),
    "japanese": ({"east_asia": 0.95}, {"JP": 0.97}),
    "ko": ({"east_asia": 0.95}, {"KR": 0.97, "KP": 0.40}),
    "korean": ({"east_asia": 0.95}, {"KR": 0.97, "KP": 0.40}),
    "zh": ({"east_asia": 0.92, "southeast_asia": 0.35}, {"CN": 0.92, "TW": 0.85, "HK": 0.80, "SG": 0.60}),
    "chinese": ({"east_asia": 0.92, "southeast_asia": 0.35}, {"CN": 0.92, "TW": 0.85, "HK": 0.80, "SG": 0.60}),
    "th": ({"southeast_asia": 0.94}, {"TH": 0.97}),
    "thai": ({"southeast_asia": 0.94}, {"TH": 0.97}),
    "vi": ({"southeast_asia": 0.94}, {"VN": 0.97}),
    "vietnamese": ({"southeast_asia": 0.94}, {"VN": 0.97}),
    "id": ({"southeast_asia": 0.90}, {"ID": 0.94, "MY": 0.65}),
    "indonesian": ({"southeast_asia": 0.90}, {"ID": 0.94, "MY": 0.65}),
    "ms": ({"southeast_asia": 0.90}, {"MY": 0.94, "ID": 0.65, "SG": 0.65, "BN": 0.80}),
    "malay": ({"southeast_asia": 0.90}, {"MY": 0.94, "ID": 0.65, "SG": 0.65, "BN": 0.80}),
    "tl": ({"southeast_asia": 0.90}, {"PH": 0.95}),
    "tagalog": ({"southeast_asia": 0.90}, {"PH": 0.95}),
    "filipino": ({"southeast_asia": 0.90}, {"PH": 0.95}),
    "hi": ({"south_asia": 0.92}, {"IN": 0.95, "NP": 0.55}),
    "hindi": ({"south_asia": 0.92}, {"IN": 0.95, "NP": 0.55}),
    "bn": ({"south_asia": 0.92}, {"BD": 0.95, "IN": 0.68}),
    "bengali": ({"south_asia": 0.92}, {"BD": 0.95, "IN": 0.68}),
    "ur": ({"south_asia": 0.90}, {"PK": 0.95, "IN": 0.55}),
    "urdu": ({"south_asia": 0.90}, {"PK": 0.95, "IN": 0.55}),
    "sw": ({"sub_saharan_africa": 0.90}, {"KE": 0.90, "TZ": 0.92, "UG": 0.80, "RW": 0.65}),
    "swahili": ({"sub_saharan_africa": 0.90}, {"KE": 0.90, "TZ": 0.92, "UG": 0.80, "RW": 0.65}),
    "af": ({"sub_saharan_africa": 0.90}, {"ZA": 0.94, "NA": 0.75}),
    "afrikaans": ({"sub_saharan_africa": 0.90}, {"ZA": 0.94, "NA": 0.75}),
}

# Climate zone -> subregion affinities
_CLIMATE_SUBREGIONS: dict[str, dict[str, float]] = {
    "tropical": {
        "southeast_asia": 0.85, "central_america": 0.80, "south_america": 0.75,
        "sub_saharan_africa": 0.70, "south_asia": 0.68, "oceania": 0.45,
    },
    "monsoon": {
        "south_asia": 0.88, "southeast_asia": 0.88, "east_asia": 0.45,
    },
    "equatorial": {
        "southeast_asia": 0.88, "south_america": 0.80, "sub_saharan_africa": 0.80,
        "central_america": 0.75,
    },
    "mediterranean": {
        "southern_europe": 0.90, "north_africa": 0.65, "middle_east": 0.55,
        "north_america": 0.30, "south_america": 0.30, "oceania": 0.30,
    },
    "temperate": {
        "western_europe": 0.80, "british_isles": 0.78, "eastern_europe": 0.72,
        "north_america": 0.72, "east_asia": 0.70, "oceania": 0.65, "southern_europe": 0.50,
    },
    "oceanic": {
        "british_isles": 0.88, "western_europe": 0.82, "oceania": 0.72, "northern_europe": 0.50,
    },
    "maritime": {
        "british_isles": 0.88, "western_europe": 0.80, "northern_europe": 0.60, "oceania": 0.65,
    },
    "continental": {
        "eastern_europe": 0.85, "north_america": 0.80, "western_europe": 0.68,
        "northern_europe": 0.65, "east_asia": 0.60,
    },
    "boreal": {
        "northern_europe": 0.90, "north_america": 0.78, "eastern_europe": 0.72,
    },
    "subarctic": {
        "northern_europe": 0.92, "north_america": 0.80, "eastern_europe": 0.70,
    },
    "arctic": {
        "northern_europe": 0.92, "north_america": 0.82,
    },
    "tundra": {
        "northern_europe": 0.90, "north_america": 0.80, "eastern_europe": 0.65,
    },
    "arid": {
        "middle_east": 0.90, "north_africa": 0.90, "oceania": 0.65,
        "sub_saharan_africa": 0.55, "north_america": 0.50, "south_america": 0.45,
        "south_asia": 0.40,
    },
    "desert": {
        "middle_east": 0.92, "north_africa": 0.92, "oceania": 0.65,
        "sub_saharan_africa": 0.50, "north_america": 0.45, "south_america": 0.40,
    },
    "semi_arid": {
        "middle_east": 0.80, "north_africa": 0.78, "sub_saharan_africa": 0.68,
        "oceania": 0.65, "north_america": 0.55, "southern_europe": 0.50,
    },
    "steppe": {
        "eastern_europe": 0.75, "middle_east": 0.78, "east_asia": 0.65, "north_america": 0.55,
    },
    "savanna": {
        "sub_saharan_africa": 0.92, "south_america": 0.65, "oceania": 0.60,
    },
    "alpine": {
        "western_europe": 0.82, "southern_europe": 0.65, "northern_europe": 0.60,
        "south_america": 0.55, "north_america": 0.55, "south_asia": 0.50,
    },
}

# Architecture cues -> (subregion boosts, country boosts)
_ARCH_HINTS: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "fachwerk": ({"western_europe": 0.92}, {"DE": 0.94, "FR": 0.65, "AT": 0.65, "CH": 0.65}),
    "half_timbered": ({"western_europe": 0.90, "british_isles": 0.55}, {"DE": 0.90, "FR": 0.65, "GB": 0.60, "AT": 0.60, "CH": 0.60}),
    "timber_frame": ({"western_europe": 0.85, "british_isles": 0.55}, {"DE": 0.85, "FR": 0.60, "GB": 0.60}),
    "bavarian": ({"western_europe": 0.92}, {"DE": 0.92, "AT": 0.85}),
    "chalet": ({"western_europe": 0.90}, {"CH": 0.92, "AT": 0.90, "DE": 0.78, "FR": 0.75, "IT": 0.65}),
    "haussmann": ({"western_europe": 0.92}, {"FR": 0.95, "BE": 0.60}),
    "dutch_gable": ({"western_europe": 0.92}, {"NL": 0.95, "BE": 0.80, "DE": 0.55}),
    "stepped_gable": ({"western_europe": 0.90}, {"NL": 0.92, "BE": 0.85, "DE": 0.65}),
    "red_brick": (
        {"british_isles": 0.85, "western_europe": 0.82, "northern_europe": 0.55, "north_america": 0.45},
        {"GB": 0.88, "NL": 0.86, "BE": 0.82, "DE": 0.72, "IE": 0.72, "DK": 0.68, "PL": 0.60},
    ),
    "terraced": ({"british_isles": 0.90, "western_europe": 0.55}, {"GB": 0.92, "IE": 0.85, "NL": 0.60, "BE": 0.60}),
    "victorian": ({"british_isles": 0.88, "oceania": 0.65, "north_america": 0.55}, {"GB": 0.90, "IE": 0.75, "AU": 0.70, "NZ": 0.65, "US": 0.60}),
    "tudor": ({"british_isles": 0.90}, {"GB": 0.94, "IE": 0.65}),
    "nordic": ({"northern_europe": 0.94}, {"NO": 0.90, "SE": 0.90, "FI": 0.85, "DK": 0.75, "IS": 0.80}),
    "falun_red": ({"northern_europe": 0.95}, {"SE": 0.95, "NO": 0.85, "FI": 0.85}),
    "wooden_clapboard": ({"northern_europe": 0.80, "north_america": 0.82, "oceania": 0.65}, {"US": 0.82, "CA": 0.80, "NO": 0.82, "SE": 0.80, "NZ": 0.68}),
    "whitewashed": ({"southern_europe": 0.90, "north_africa": 0.70}, {"GR": 0.92, "ES": 0.88, "PT": 0.82, "IT": 0.78, "MA": 0.75, "TN": 0.75}),
    "terracotta": ({"southern_europe": 0.88, "western_europe": 0.55, "south_america": 0.55}, {"IT": 0.90, "ES": 0.88, "PT": 0.85, "HR": 0.82, "GR": 0.80, "FR": 0.68}),
    "soviet": ({"eastern_europe": 0.92, "middle_east": 0.55}, {"RU": 0.92, "UA": 0.88, "BY": 0.85, "PL": 0.72, "RO": 0.72, "BG": 0.72, "KZ": 0.75}),
    "khrushchyovka": ({"eastern_europe": 0.94}, {"RU": 0.94, "UA": 0.90, "BY": 0.88, "KZ": 0.78}),
    "panel_block": ({"eastern_europe": 0.90, "western_europe": 0.45}, {"RU": 0.88, "UA": 0.85, "PL": 0.82, "CZ": 0.78, "SK": 0.78, "HU": 0.78, "RO": 0.78, "BG": 0.75, "DE": 0.55}),
    "plattenbau": ({"western_europe": 0.88, "eastern_europe": 0.75}, {"DE": 0.92, "PL": 0.70, "CZ": 0.70}),
    "pagoda": ({"east_asia": 0.92, "southeast_asia": 0.65}, {"JP": 0.90, "CN": 0.90, "KR": 0.85, "TW": 0.85, "VN": 0.70, "TH": 0.65, "MM": 0.70}),
    "tiled_curved_roof": ({"east_asia": 0.92}, {"JP": 0.90, "CN": 0.88, "KR": 0.86, "TW": 0.84}),
    "stilt": ({"southeast_asia": 0.90}, {"TH": 0.88, "KH": 0.88, "LA": 0.85, "VN": 0.80, "ID": 0.82, "MY": 0.78, "PH": 0.78, "MM": 0.80}),
    "shophouse": ({"southeast_asia": 0.90, "east_asia": 0.70}, {"SG": 0.90, "MY": 0.90, "TH": 0.82, "VN": 0.82, "TW": 0.78, "HK": 0.78}),
    "colonial": ({"central_america": 0.82, "south_america": 0.82, "north_america": 0.65, "southeast_asia": 0.50}, {"MX": 0.85, "CO": 0.82, "PE": 0.80, "BR": 0.78, "CU": 0.80, "AR": 0.72, "ES": 0.65, "PT": 0.65}),
    "adobe": ({"north_america": 0.75, "central_america": 0.78, "south_america": 0.75, "north_africa": 0.75, "middle_east": 0.70}, {"MX": 0.85, "PE": 0.80, "BO": 0.80, "MA": 0.82, "US": 0.65}),
    "flat_concrete": ({"middle_east": 0.85, "north_africa": 0.85, "south_asia": 0.70, "southern_europe": 0.55}, {"EG": 0.85, "MA": 0.82, "JO": 0.82, "SA": 0.80, "TR": 0.75, "IN": 0.72, "PK": 0.72, "GR": 0.68}),
    "corrugated_iron": ({"sub_saharan_africa": 0.80, "oceania": 0.75, "south_america": 0.65, "southeast_asia": 0.65}, {"AU": 0.78, "NZ": 0.75, "ZA": 0.78, "KE": 0.78, "IS": 0.70}),
    "suburban_ranch": ({"north_america": 0.90, "oceania": 0.65}, {"US": 0.92, "CA": 0.85, "AU": 0.70}),
}

# Road lines -> (subregion boosts, country boosts)
_ROAD_LINE_HINTS: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "yellow_center": (
        {"north_america": 0.88, "south_america": 0.82, "central_america": 0.80, "east_asia": 0.72, "northern_europe": 0.65, "middle_east": 0.55},
        {"US": 0.90, "CA": 0.88, "MX": 0.82, "BR": 0.82, "AR": 0.78, "CO": 0.78, "CL": 0.75, "PE": 0.75, "KR": 0.80, "TW": 0.78, "CN": 0.75, "JP": 0.68, "NO": 0.82, "FI": 0.75, "PH": 0.75, "TH": 0.72, "IL": 0.72, "AE": 0.68},
    ),
    "double_yellow": (
        {"north_america": 0.90, "south_america": 0.82, "central_america": 0.80, "east_asia": 0.68},
        {"US": 0.92, "CA": 0.88, "MX": 0.82, "BR": 0.82, "AR": 0.78, "CO": 0.78, "KR": 0.75, "TW": 0.75, "PH": 0.72},
    ),
    "white_center": (
        {"western_europe": 0.85, "british_isles": 0.82, "southern_europe": 0.82, "eastern_europe": 0.80, "oceania": 0.75, "northern_europe": 0.65},
        {"DE": 0.86, "FR": 0.86, "NL": 0.85, "BE": 0.84, "GB": 0.85, "IE": 0.78, "IT": 0.84, "ES": 0.84, "PT": 0.82, "PL": 0.82, "CZ": 0.80, "AT": 0.84, "CH": 0.84, "SE": 0.80, "DK": 0.80, "AU": 0.80, "NZ": 0.78},
    ),
    "dashed_edge": (
        {"western_europe": 0.80, "northern_europe": 0.85, "british_isles": 0.70},
        {"FR": 0.85, "SE": 0.88, "DK": 0.85, "NL": 0.82, "BE": 0.78, "EE": 0.75, "LV": 0.75, "LT": 0.75},
    ),
    "yellow_edge": (
        {"sub_saharan_africa": 0.88, "british_isles": 0.82, "middle_east": 0.72},
        {"ZA": 0.92, "BW": 0.85, "NA": 0.82, "SZ": 0.82, "LS": 0.82, "IE": 0.88, "GB": 0.75, "IL": 0.82, "JO": 0.78, "AE": 0.75},
    ),
    "red_bike_lane": (
        {"western_europe": 0.92, "northern_europe": 0.65},
        {"NL": 0.95, "BE": 0.85, "DE": 0.80, "DK": 0.78},
    ),
}

# Vegetation cues -> (subregion boosts, country boosts)
_VEG_HINTS: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "palm": (
        {"southeast_asia": 0.85, "central_america": 0.82, "south_america": 0.78, "south_asia": 0.75, "middle_east": 0.70, "north_africa": 0.72, "sub_saharan_africa": 0.68, "southern_europe": 0.50},
        {"TH": 0.85, "ID": 0.85, "MY": 0.84, "PH": 0.84, "BR": 0.82, "MX": 0.78, "CO": 0.78, "IN": 0.78, "LK": 0.80, "EG": 0.75, "MA": 0.72, "AE": 0.75, "SA": 0.72, "ES": 0.55},
    ),
    "banana": (
        {"southeast_asia": 0.88, "south_america": 0.80, "central_america": 0.82, "sub_saharan_africa": 0.75, "south_asia": 0.75},
        {"TH": 0.85, "VN": 0.84, "ID": 0.85, "PH": 0.85, "CO": 0.82, "EC": 0.84, "BR": 0.80, "CR": 0.82, "UG": 0.80, "IN": 0.78},
    ),
    "rice_paddy": (
        {"southeast_asia": 0.90, "east_asia": 0.85, "south_asia": 0.82},
        {"VN": 0.90, "TH": 0.88, "ID": 0.88, "JP": 0.85, "CN": 0.85, "IN": 0.82, "BD": 0.85, "PH": 0.84, "KH": 0.84},
    ),
    "bamboo": (
        {"east_asia": 0.88, "southeast_asia": 0.88, "south_asia": 0.70},
        {"JP": 0.88, "CN": 0.88, "TW": 0.86, "VN": 0.85, "TH": 0.84, "ID": 0.80},
    ),
    "olive": (
        {"southern_europe": 0.92, "north_africa": 0.75, "middle_east": 0.72},
        {"ES": 0.92, "IT": 0.92, "GR": 0.92, "PT": 0.85, "TR": 0.82, "TN": 0.80, "MA": 0.78, "HR": 0.78, "IL": 0.75},
    ),
    "cypress": (
        {"southern_europe": 0.90, "middle_east": 0.65},
        {"IT": 0.92, "GR": 0.88, "ES": 0.85, "HR": 0.80, "FR": 0.70, "TR": 0.70},
    ),
    "vineyard": (
        {"southern_europe": 0.85, "western_europe": 0.85, "south_america": 0.55, "oceania": 0.50},
        {"FR": 0.90, "IT": 0.90, "ES": 0.88, "DE": 0.80, "PT": 0.82, "AT": 0.75, "CH": 0.72, "CL": 0.68, "AR": 0.65, "ZA": 0.65},
    ),
    "deciduous": (
        {"western_europe": 0.85, "british_isles": 0.80, "eastern_europe": 0.80, "north_america": 0.78, "east_asia": 0.70},
        {"DE": 0.86, "FR": 0.84, "GB": 0.82, "NL": 0.80, "BE": 0.80, "PL": 0.80, "CZ": 0.78, "AT": 0.78, "US": 0.78, "CA": 0.72},
    ),
    "oak": (
        {"western_europe": 0.85, "british_isles": 0.82, "southern_europe": 0.72, "north_america": 0.75, "eastern_europe": 0.72},
        {"DE": 0.86, "FR": 0.84, "GB": 0.85, "ES": 0.75, "PT": 0.75, "US": 0.78, "PL": 0.75},
    ),
    "beech": (
        {"western_europe": 0.88, "eastern_europe": 0.78, "british_isles": 0.75},
        {"DE": 0.90, "FR": 0.82, "AT": 0.82, "CH": 0.80, "PL": 0.78, "CZ": 0.78, "DK": 0.78, "GB": 0.75},
    ),
    "hedgerow": (
        {"british_isles": 0.92, "western_europe": 0.78},
        {"GB": 0.94, "IE": 0.92, "FR": 0.82, "BE": 0.72, "DE": 0.68},
    ),
    "spruce": (
        {"northern_europe": 0.90, "western_europe": 0.75, "eastern_europe": 0.75, "north_america": 0.78},
        {"NO": 0.90, "SE": 0.90, "FI": 0.90, "RU": 0.82, "CA": 0.85, "DE": 0.78, "AT": 0.80, "CH": 0.78, "EE": 0.80, "LV": 0.80},
    ),
    "birch": (
        {"northern_europe": 0.90, "eastern_europe": 0.85, "western_europe": 0.65},
        {"FI": 0.92, "SE": 0.90, "NO": 0.88, "RU": 0.92, "EE": 0.85, "LV": 0.85, "LT": 0.85, "PL": 0.78, "BY": 0.85},
    ),
    "conifer": (
        {"northern_europe": 0.88, "north_america": 0.82, "western_europe": 0.75, "eastern_europe": 0.75},
        {"NO": 0.88, "SE": 0.88, "FI": 0.88, "CA": 0.88, "US": 0.78, "RU": 0.82, "DE": 0.76, "AT": 0.80, "CH": 0.80},
    ),
    "pine": (
        {"northern_europe": 0.85, "southern_europe": 0.75, "north_america": 0.78, "eastern_europe": 0.75, "western_europe": 0.72},
        {"SE": 0.86, "FI": 0.86, "NO": 0.84, "PL": 0.80, "RU": 0.80, "CA": 0.82, "US": 0.78, "DE": 0.75, "ES": 0.75, "IT": 0.75},
    ),
    "eucalyptus": (
        {"oceania": 0.94, "south_america": 0.60, "southern_europe": 0.55, "sub_saharan_africa": 0.55},
        {"AU": 0.96, "NZ": 0.72, "PT": 0.68, "ES": 0.62, "CL": 0.62, "BR": 0.60, "ZA": 0.62},
    ),
    "acacia": (
        {"sub_saharan_africa": 0.92, "oceania": 0.70},
        {"KE": 0.92, "TZ": 0.92, "ZA": 0.88, "BW": 0.88, "NA": 0.85, "SN": 0.80, "AU": 0.72},
    ),
    "baobab": (
        {"sub_saharan_africa": 0.95},
        {"MG": 0.94, "SN": 0.90, "TZ": 0.88, "KE": 0.85, "BW": 0.85, "ZW": 0.82, "ZA": 0.80},
    ),
    "cactus": (
        {"north_america": 0.88, "central_america": 0.82, "south_america": 0.75},
        {"MX": 0.94, "US": 0.85, "PE": 0.78, "BO": 0.78, "AR": 0.75, "CL": 0.75, "CO": 0.70},
    ),
    "agave": (
        {"north_america": 0.88, "central_america": 0.82},
        {"MX": 0.94, "US": 0.78, "GT": 0.75},
    ),
    "tree_fern": (
        {"oceania": 0.92},
        {"NZ": 0.95, "AU": 0.82},
    ),
}


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _norm_cc(cc: Any) -> str:
    """Normalize country code or country name to uppercase ISO-2 code."""
    if not cc:
        return ""
    s = str(cc).strip()
    if not s:
        return ""
    # Strip sub-region suffix like "US-CA" -> "US"
    if "-" in s and len(s.split("-", 1)[0]) in (2, 3):
        s = s.split("-", 1)[0]
    u = s.upper()
    if u in _CC_ALIASES:
        return _CC_ALIASES[u]
    if u in CC_TO_CONTINENT:
        return u
    low = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    if low in _NAME_TO_CC:
        return _NAME_TO_CC[low]
    return u if len(u) == 2 and u.isalpha() else ""


def _tokens_from_val(val: Any) -> list[str]:
    """Extract normalized lowercase tokens and phrases from an observation value."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        out: list[str] = []
        for item in val:
            out.extend(_tokens_from_val(item))
        return out
    if isinstance(val, dict):
        out = []
        for k, v in val.items():
            if v:
                out.extend(_tokens_from_val(k if v is True else v))
        return out
    s = str(val).strip().lower()
    if not s or s in ("none", "null", "unknown", "n/a", "na", "unclear"):
        return []
    norm = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    words = [w for w in norm.split("_") if w]
    result = [norm] if norm else []
    for w in words:
        if w not in result:
            result.append(w)
    # Also add adjacent 2-word bigrams
    for i in range(len(words) - 1):
        bg = f"{words[i]}_{words[i + 1]}"
        if bg not in result:
            result.append(bg)
    return result


def _match_table(
    tokens: list[str],
    table: dict[str, tuple[dict[str, float], dict[str, float]]],
    sub_scores: dict[str, float],
    cc_scores: dict[str, float],
    weight: float = 1.0,
) -> bool:
    """Apply matching cue tokens from `table` into `sub_scores` and `cc_scores`."""
    matched = False
    for tok in tokens:
        for key, (s_boosts, c_boosts) in table.items():
            if tok == key or (len(key) >= 4 and (key in tok or tok in key)):
                matched = True
                for sub, b in s_boosts.items():
                    sub_scores[sub] = sub_scores.get(sub, 0.0) + b * weight
                for cc, b in c_boosts.items():
                    cc_scores[cc] = cc_scores.get(cc, 0.0) + b * weight
    return matched


def _infer_region_hint(observations: dict, top_cc: str) -> str | None:
    """Extract optional sub-national region hint when present in observations."""
    for k in ("region", "subnational_region", "state", "province", "region_guess"):
        val = observations.get(k)
        if isinstance(val, str) and val.strip():
            s = val.strip()
            if s.lower() not in ("unknown", "none", "null", "n/a"):
                return s
    # Check specific architectural/vegetation regional cues for top country
    arch_toks = _tokens_from_val(observations.get("architecture"))
    if top_cc == "DE" and any("bavaria" in t or "chalet" in t or "alpine" in t for t in arch_toks):
        return "Bavaria"
    return None


# ---------------------------------------------------------------------------
# 2. infer_hierarchy
# ---------------------------------------------------------------------------

def infer_hierarchy(
    observations: dict,
    cue_scores: dict[str, float] | None = None,
) -> dict:
    """Compute coarse-to-fine geolocation hierarchy (continent -> subregion -> country -> region).

    Parameters
    ----------
    observations:
        VLM/OSINT observation dict with optional keys such as ``driving_side``,
        ``script``, ``climate_zone``, ``architecture``, ``road_lines``,
        ``sun_side``, ``vegetation``, ``sign_lang`` (plus aliases).
    cue_scores:
        Optional mapping of ISO-2 country code to score (e.g. ``{"DE": 0.7, ...}``).

    Returns
    -------
    dict with keys:
      - ``continent``: dict (``{"code": str, "name": str, "confidence": float}``)
        that also compares equal to its continent code string
      - ``subregion``: dict (``{"name": str, "label": str, "confidence": float, "continent": str}``)
        that also compares equal to its subregion name string
      - ``top_countries``: list of ``{"cc": cc, "score": float, "subregion": str, "name": str}``
      - ``funnel``: human-readable string like ``"Europe → Western Europe → Germany (DE)"``
    Never raises on any input.
    """
    try:
        obs = observations if isinstance(observations, dict) else {}
        raw_cues = cue_scores if isinstance(cue_scores, dict) else {}

        sub_scores: dict[str, float] = {sub: 0.0 for sub in SUBREGIONS}
        cc_scores: dict[str, float] = {cc: 0.0 for cc in CC_TO_CONTINENT}
        cont_direct: dict[str, float] = {c: 0.0 for c in CONTINENTS}
        active_cue_count = 0

        # --- A. Driving side ---
        drive_toks = _tokens_from_val(
            obs.get("driving_side") or obs.get("traffic_side") or obs.get("drive_side")
        )
        drive_side: str | None = None
        if any(t in ("left", "lht", "keep_left", "left_hand", "left_side") for t in drive_toks):
            drive_side = "left"
        elif any(t in ("right", "rht", "keep_right", "right_hand", "right_side") for t in drive_toks):
            drive_side = "right"

        if drive_side == "left":
            active_cue_count += 1
            for cc in CC_TO_CONTINENT:
                if cc in _LEFT_DRIVE_CCS:
                    cc_scores[cc] += 0.55
                    sub_scores[CC_TO_SUBREGION[cc]] += 0.06
                else:
                    cc_scores[cc] -= 0.45
            for s in ("british_isles", "oceania", "south_asia", "southeast_asia"):
                sub_scores[s] += 0.45
        elif drive_side == "right":
            active_cue_count += 1
            for cc in CC_TO_CONTINENT:
                if cc not in _LEFT_DRIVE_CCS:
                    cc_scores[cc] += 0.35
                else:
                    cc_scores[cc] -= 0.50
            for s in (
                "western_europe", "northern_europe", "southern_europe",
                "eastern_europe", "north_america", "central_america",
                "south_america", "middle_east", "north_africa",
            ):
                sub_scores[s] += 0.30

        # --- B. Sun side / Hemisphere ---
        sun_toks = _tokens_from_val(
            obs.get("sun_side") or obs.get("hemisphere") or obs.get("sun_position")
        )
        hemi: str | None = None
        if any(
            t in ("south", "southern_sky", "sun_south", "northern", "north_hemisphere", "northern_hemisphere", "right")
            for t in sun_toks
        ):
            hemi = "north"
        elif any(
            t in ("north", "northern_sky", "sun_north", "southern", "south_hemisphere", "southern_hemisphere", "left")
            for t in sun_toks
        ):
            hemi = "south"
        elif any(t in ("overhead", "zenith", "equatorial", "center", "centre") for t in sun_toks):
            hemi = "equatorial"

        if hemi == "north":
            active_cue_count += 1
            for c in ("EU", "NA", "AS"):
                cont_direct[c] += 0.35
            cont_direct["OC"] -= 0.40
            for cc in CC_TO_CONTINENT:
                if cc not in _SOUTHERN_HEMISPHERE_CCS:
                    cc_scores[cc] += 0.25
                else:
                    cc_scores[cc] -= 0.40
        elif hemi == "south":
            active_cue_count += 1
            for c in ("OC", "SA", "AF"):
                cont_direct[c] += 0.45
            for c in ("EU", "NA"):
                cont_direct[c] -= 0.45
            for cc in CC_TO_CONTINENT:
                if cc in _SOUTHERN_HEMISPHERE_CCS:
                    cc_scores[cc] += 0.65
                    sub_scores[CC_TO_SUBREGION[cc]] += 0.12
                else:
                    cc_scores[cc] -= 0.35
        elif hemi == "equatorial":
            active_cue_count += 1
            for cc in _EQUATORIAL_CCS:
                if cc in cc_scores:
                    cc_scores[cc] += 0.40
            for s in ("southeast_asia", "central_america", "south_america", "sub_saharan_africa"):
                sub_scores[s] += 0.35

        # --- C. Script ---
        script_toks = _tokens_from_val(
            obs.get("script") or obs.get("scripts") or obs.get("languages_scripts")
        )
        if script_toks:
            if _match_table(script_toks, _SCRIPT_HINTS, sub_scores, cc_scores, weight=1.1):
                active_cue_count += 1

        # --- D. Sign language ---
        lang_toks = _tokens_from_val(
            obs.get("sign_lang") or obs.get("sign_language") or obs.get("language") or obs.get("lang")
        )
        if lang_toks:
            if _match_table(lang_toks, _LANG_HINTS, sub_scores, cc_scores, weight=1.25):
                active_cue_count += 1
            # Also check if language token matches a script key
            _match_table(lang_toks, _SCRIPT_HINTS, sub_scores, cc_scores, weight=0.6)

        # --- E. Climate zone ---
        clim_toks = _tokens_from_val(
            obs.get("climate_zone") or obs.get("climate") or obs.get("biome")
        )
        if clim_toks:
            clim_hit = False
            for tok in clim_toks:
                for ckey, s_map in _CLIMATE_SUBREGIONS.items():
                    if tok == ckey or (len(ckey) >= 4 and (ckey in tok or tok in ckey)):
                        clim_hit = True
                        for sub, b in s_map.items():
                            sub_scores[sub] = sub_scores.get(sub, 0.0) + b * 0.9
                            for cc in SUBREGIONS.get(sub, ()):
                                cc_scores[cc] += b * 0.35
            if clim_hit:
                active_cue_count += 1

        # --- F. Architecture ---
        arch_toks = _tokens_from_val(obs.get("architecture") or obs.get("buildings"))
        if arch_toks:
            if _match_table(arch_toks, _ARCH_HINTS, sub_scores, cc_scores, weight=1.0):
                active_cue_count += 1

        # --- G. Road lines / markings ---
        road_toks = _tokens_from_val(
            obs.get("road_lines") or obs.get("lane_markings_color") or obs.get("road_markings")
        )
        if road_toks:
            # Normalize common phrases like "yellow" / "white" center lines
            expanded_road = list(road_toks)
            if "yellow" in road_toks and not any("edge" in t for t in road_toks):
                expanded_road.append("yellow_center")
            if "white" in road_toks and not any("yellow" in t for t in road_toks):
                expanded_road.append("white_center")
            if _match_table(expanded_road, _ROAD_LINE_HINTS, sub_scores, cc_scores, weight=0.85):
                active_cue_count += 1

        # --- H. Vegetation ---
        veg_toks = _tokens_from_val(obs.get("vegetation") or obs.get("plants") or obs.get("trees"))
        if veg_toks:
            if _match_table(veg_toks, _VEG_HINTS, sub_scores, cc_scores, weight=0.9):
                active_cue_count += 1
            # Also allow climate keywords in vegetation (e.g. "tropical", "boreal", "mediterranean")
            for tok in veg_toks:
                if tok in _CLIMATE_SUBREGIONS:
                    for sub, b in _CLIMATE_SUBREGIONS[tok].items():
                        sub_scores[sub] += b * 0.5
                        for cc in SUBREGIONS.get(sub, ()):
                            cc_scores[cc] += b * 0.2

        # --- I. Additional explicit OSINT cues (plate_style, domain, country_indicators) ---
        plate_toks = _tokens_from_val(obs.get("plate_style") or obs.get("plates"))
        if any(t in ("eu", "eu_blue_strip", "blue_strip", "euroband", "blue_band") for t in plate_toks):
            active_cue_count += 1
            cont_direct["EU"] += 0.65
            for s in ("western_europe", "southern_europe", "eastern_europe", "northern_europe"):
                sub_scores[s] += 0.45
        if any("yellow_rear" in t or "yellow_back" in t for t in plate_toks):
            active_cue_count += 1
            for cc in ("GB", "NL", "IE", "LU", "CY"):
                cc_scores[cc] += 0.60

        for dom_tok in _tokens_from_val(obs.get("domain") or obs.get("country_indicators")):
            cc_cand = _norm_cc(dom_tok.lstrip("."))
            if cc_cand and cc_cand in CC_TO_CONTINENT:
                active_cue_count += 1
                cc_scores[cc_cand] += 1.2
                sub_scores[CC_TO_SUBREGION[cc_cand]] += 0.7

        # --- J. Fold in external cue_scores dict {"DE": 0.7, ...} ---
        parsed_cues: dict[str, float] = {}
        for raw_k, raw_v in raw_cues.items():
            cc = _norm_cc(raw_k)
            if not cc or cc not in CC_TO_CONTINENT:
                continue
            try:
                val = float(raw_v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val) and val > 0:
                parsed_cues[cc] = max(parsed_cues.get(cc, 0.0), val)

        if parsed_cues:
            active_cue_count += 1
            max_cue = max(parsed_cues.values())
            norm_scale = max_cue if max_cue > 1.0 else 1.0
            for cc, val in parsed_cues.items():
                norm_v = max(0.0, min(1.0, val / norm_scale))
                cc_scores[cc] += norm_v * 1.6
                sub = CC_TO_SUBREGION[cc]
                sub_scores[sub] += norm_v * 0.95
                cont_direct[CC_TO_CONTINENT[cc]] += norm_v * 0.65

        # If nothing at all was observed or passed, return clean empty hierarchy
        if active_cue_count == 0 and not any(v > 0 for v in cc_scores.values()) and not any(v > 0 for v in sub_scores.values()):
            empty_cont = _HierLevel(code=None, name=None, confidence=0.0, score=0.0)
            empty_sub = _HierLevel(name=None, label=None, code=None, continent=None, confidence=0.0, score=0.0)
            return {
                "continent": empty_cont,
                "continent_code": None,
                "continent_name": None,
                "continent_confidence": 0.0,
                "subregion": empty_sub,
                "subregion_name": None,
                "subregion_label": None,
                "subregion_confidence": 0.0,
                "top_countries": [],
                "region": None,
                "funnel": "",
                "confidence": 0.0,
            }

        # --- K. Hierarchical rollup & top-down coupling ---
        # 1. Propagate positive country scores into subregions (top-3 countries per subregion)
        for sub, member_ccs in SUBREGIONS.items():
            pos_member_scores = sorted(
                (max(0.0, cc_scores.get(c, 0.0)) for c in member_ccs),
                reverse=True,
            )
            if pos_member_scores:
                top_m = pos_member_scores[0]
                mean_top3 = sum(pos_member_scores[:3]) / min(3, len(pos_member_scores))
                sub_scores[sub] += 0.55 * top_m + 0.25 * mean_top3

        # Clamp subregion scores to >= 0
        for sub in sub_scores:
            sub_scores[sub] = max(0.0, sub_scores[sub])

        # 2. Propagate subregion scores into continents
        cont_raw: dict[str, float] = {}
        for cont_code, cont_info in CONTINENTS.items():
            subs = list(cont_info["subregions"].keys())
            s_vals = sorted((sub_scores.get(s, 0.0) for s in subs), reverse=True)
            best_s = s_vals[0] if s_vals else 0.0
            second_s = s_vals[1] if len(s_vals) > 1 else 0.0
            cont_raw[cont_code] = max(
                0.0,
                cont_direct.get(cont_code, 0.0) + best_s + 0.25 * second_s,
            )

        best_cont = max(sorted(cont_raw), key=lambda c: cont_raw[c])
        total_cont = sum(cont_raw.values())
        best_cont_raw = cont_raw[best_cont]
        second_cont_raw = sorted(cont_raw.values(), reverse=True)[1] if len(cont_raw) > 1 else 0.0

        # Continent confidence: combines relative share, margin over runner-up, and evidence strength
        if best_cont_raw <= 0.0:
            cont_conf = 0.0
        else:
            share = best_cont_raw / total_cont if total_cont > 0 else 0.5
            margin = (best_cont_raw - second_cont_raw) / best_cont_raw if best_cont_raw > 0 else 0.0
            strength = min(1.0, best_cont_raw / 1.8)
            cue_bonus = min(0.18, 0.05 * active_cue_count)
            cont_conf = min(0.95, max(0.25, 0.35 * share + 0.30 * margin + 0.35 * strength + cue_bonus))
            if parsed_cues:
                max_c = max(parsed_cues.values())
                if max_c >= 0.6:
                    cont_conf = max(cont_conf, min(0.92, max_c + 0.08))
            cont_conf = round(cont_conf, 3)

        # 3. Pick best subregion (preferring subregions within the winning continent,
        #    with hierarchical bonus from continent score)
        sub_hier_raw: dict[str, float] = {}
        for sub, s_val in sub_scores.items():
            c_code = SUBREGION_TO_CONTINENT[sub]
            c_factor = 1.25 if c_code == best_cont else 0.65
            sub_hier_raw[sub] = s_val * c_factor

        best_sub = max(sorted(sub_hier_raw), key=lambda s: sub_hier_raw[s])
        # Ensure continent matches best_sub's continent if best_sub is overwhelmingly stronger
        best_cont = SUBREGION_TO_CONTINENT[best_sub]
        total_sub = sum(sub_hier_raw.values())
        best_sub_raw = sub_hier_raw[best_sub]
        second_sub_raw = sorted(sub_hier_raw.values(), reverse=True)[1] if len(sub_hier_raw) > 1 else 0.0

        if best_sub_raw <= 0.0:
            sub_conf = 0.0
        else:
            s_share = best_sub_raw / total_sub if total_sub > 0 else 0.3
            s_margin = (best_sub_raw - second_sub_raw) / best_sub_raw if best_sub_raw > 0 else 0.0
            s_strength = min(1.0, best_sub_raw / 1.6)
            sub_conf = min(
                cont_conf,
                max(0.20, 0.30 * s_share + 0.30 * s_margin + 0.40 * s_strength + min(0.12, 0.04 * active_cue_count)),
            )
            if parsed_cues:
                best_in_sub = max((parsed_cues.get(c, 0.0) for c in SUBREGIONS[best_sub]), default=0.0)
                if best_in_sub > 0:
                    sub_conf = max(sub_conf, min(cont_conf, round(best_in_sub, 3)))
            sub_conf = round(sub_conf, 3)

        # 4. Score countries hierarchically (country raw score + subregion & continent priors)
        final_cc: dict[str, float] = {}
        for cc, raw_c in cc_scores.items():
            sub = CC_TO_SUBREGION[cc]
            cont = CC_TO_CONTINENT[cc]
            hier_prior = 0.30 * sub_scores.get(sub, 0.0)
            if sub == best_sub:
                hier_prior += 0.25
            elif cont == best_cont:
                hier_prior += 0.10
            combined = raw_c + hier_prior
            if combined > 0:
                final_cc[cc] = combined

        # Normalize top_countries scores to 0..1 scaled against confidence / cue_scores
        ranked_ccs = sorted(final_cc.keys(), key=lambda c: (-final_cc[c], c))
        max_cc_raw = final_cc[ranked_ccs[0]] if ranked_ccs else 0.0

        # Reference ceiling for country score
        if parsed_cues:
            lead_cue = max(parsed_cues.values())
            ceiling = min(0.95, max(lead_cue if lead_cue <= 1.0 else 0.90, sub_conf))
        else:
            ceiling = min(0.90, max(0.35, sub_conf))

        top_countries: list[dict[str, Any]] = []
        for cc in ranked_ccs[:8]:
            raw_v = final_cc[cc]
            if max_cc_raw > 0:
                # Blend normalized rank score with explicit cue_score when available
                norm_score = (raw_v / max_cc_raw) * ceiling
                if cc in parsed_cues and parsed_cues[cc] <= 1.0:
                    norm_score = 0.55 * norm_score + 0.45 * parsed_cues[cc]
            else:
                norm_score = 0.0
            top_countries.append({
                "cc": cc,
                "name": COUNTRY_NAMES.get(cc, cc),
                "score": round(max(0.0, min(1.0, norm_score)), 3),
                "subregion": CC_TO_SUBREGION[cc],
                "continent": CC_TO_CONTINENT[cc],
            })

        cont_name = CONTINENT_NAMES.get(best_cont, best_cont)
        sub_label = SUBREGION_LABELS.get(best_sub, best_sub)
        lead_cc = top_countries[0]["cc"] if top_countries else None
        lead_country_name = COUNTRY_NAMES.get(lead_cc, lead_cc) if lead_cc else None
        region_hint = _infer_region_hint(obs, lead_cc or "")

        funnel_parts = [cont_name, sub_label]
        if lead_cc:
            funnel_parts.append(f"{lead_country_name} ({lead_cc})")
        if region_hint:
            funnel_parts.append(region_hint)
        funnel = " → ".join(funnel_parts)

        cont_obj = _HierLevel(
            code=best_cont,
            name=cont_name,
            confidence=cont_conf,
            score=cont_conf,
        )
        sub_obj = _HierLevel(
            name=best_sub,
            label=sub_label,
            code=best_sub,
            continent=best_cont,
            confidence=sub_conf,
            score=sub_conf,
        )

        return {
            "continent": cont_obj,
            "continent_code": best_cont,
            "continent_name": cont_name,
            "continent_confidence": cont_conf,
            "subregion": sub_obj,
            "subregion_name": best_sub,
            "subregion_label": sub_label,
            "subregion_confidence": sub_conf,
            "top_countries": top_countries,
            "region": region_hint,
            "funnel": funnel,
            "confidence": cont_conf,
        }
    except Exception:
        return {
            "continent": _HierLevel(code=None, name=None, confidence=0.0, score=0.0),
            "continent_code": None,
            "continent_name": None,
            "continent_confidence": 0.0,
            "subregion": _HierLevel(name=None, label=None, code=None, continent=None, confidence=0.0, score=0.0),
            "subregion_name": None,
            "subregion_label": None,
            "subregion_confidence": 0.0,
            "top_countries": [],
            "region": None,
            "funnel": "",
            "confidence": 0.0,
        }


# ---------------------------------------------------------------------------
# 3. hierarchy_evidence
# ---------------------------------------------------------------------------

def _extract_continent(hier: dict) -> tuple[str | None, float]:
    """Extract (continent_code, continent_confidence) from a `hier` dict."""
    cont = hier.get("continent")
    conf = hier.get("continent_confidence")
    code: str | None = None

    if isinstance(cont, dict):
        code = cont.get("code") or cont.get("continent") or cont.get("name")
        if conf is None:
            conf = cont.get("confidence", cont.get("score"))
    elif isinstance(cont, str) and cont.strip():
        code = cont.strip()

    if not code:
        code = hier.get("continent_code")

    if isinstance(code, str):
        u = code.strip().upper()
        if u in CONTINENTS:
            code = u
        else:
            # Resolve full continent name -> code
            for c_code, c_name in CONTINENT_NAMES.items():
                if c_name.lower() == code.strip().lower():
                    code = c_code
                    break

    if conf is None:
        conf = hier.get("confidence", 0.0)
    try:
        conf_f = float(conf)
        if not math.isfinite(conf_f):
            conf_f = 0.0
    except (TypeError, ValueError):
        conf_f = 0.0

    return (code if code in CONTINENTS else None), max(0.0, min(1.0, conf_f))


def _extract_subregion(hier: dict) -> tuple[str | None, float]:
    """Extract (subregion_name, subregion_confidence) from a `hier` dict."""
    sub = hier.get("subregion")
    conf = hier.get("subregion_confidence")
    name: str | None = None

    if isinstance(sub, dict):
        name = sub.get("name") or sub.get("code") or sub.get("label")
        if conf is None:
            conf = sub.get("confidence", sub.get("score"))
    elif isinstance(sub, str) and sub.strip():
        name = sub.strip()

    if not name:
        name = hier.get("subregion_name")

    if isinstance(name, str):
        low = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        if low in SUBREGIONS:
            name = low
        else:
            for s_key, s_label in SUBREGION_LABELS.items():
                if s_label.lower() == name.strip().lower():
                    name = s_key
                    break

    if conf is None:
        conf = hier.get("confidence", 0.0)
    try:
        conf_f = float(conf)
        if not math.isfinite(conf_f):
            conf_f = 0.0
    except (TypeError, ValueError):
        conf_f = 0.0

    return (name if name in SUBREGIONS else None), max(0.0, min(1.0, conf_f))


def hierarchy_evidence(
    hier: dict,
    claimed_cc: str | None = None,
) -> "Evidence | None":
    """Build an Evidence item from `infer_hierarchy` output and optional `claimed_cc`.

    Verdict rules:
      - ``SUPPORTS`` if ``claimed_cc`` is in the top 2 countries or in the matching subregion
      - ``CONTRADICTS`` if ``claimed_cc`` is on a different continent when confidence >= 0.65
      - ``INCONCLUSIVE`` otherwise

    Never raises on any input.
    """
    try:
        try:
            from .schema import ClaimVerdict, Evidence
        except ImportError:
            from schema import ClaimVerdict, Evidence

        if not isinstance(hier, dict) or not hier:
            return None

        cont_code, cont_conf = _extract_continent(hier)
        sub_name, sub_conf = _extract_subregion(hier)
        raw_top = hier.get("top_countries")
        top_countries = [
            r for r in (raw_top if isinstance(raw_top, list) else [])
            if isinstance(r, dict) and r.get("cc")
        ]

        # Derive continent/subregion from top_countries if omitted in a caller-constructed dict
        if not cont_code and top_countries:
            first_cc = _norm_cc(top_countries[0].get("cc"))
            cont_code = CC_TO_CONTINENT.get(first_cc)
            if cont_conf <= 0.0:
                cont_conf = float(top_countries[0].get("score", 0.5) or 0.5)
        if not sub_name and top_countries:
            first_cc = _norm_cc(top_countries[0].get("cc"))
            sub_name = top_countries[0].get("subregion") or CC_TO_SUBREGION.get(first_cc)
            if sub_conf <= 0.0:
                sub_conf = cont_conf

        if not cont_code and not sub_name and not top_countries:
            return Evidence(
                type="hier_geoguess",
                summary="Insufficient visual cues to narrow geographic hierarchy",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                data=hier,
            )

        cont_label = CONTINENT_NAMES.get(cont_code or "", cont_code or "Unknown")
        sub_label = SUBREGION_LABELS.get(sub_name or "", sub_name or "Unknown")
        top_cc = _norm_cc(top_countries[0]["cc"]) if top_countries else ""
        top_country_name = COUNTRY_NAMES.get(top_cc, top_cc) if top_cc else ""

        base_summary = f"Visual cues narrow region to {cont_label} → {sub_label}"
        if top_country_name:
            base_summary += f" (top match: {top_country_name})"

        cc = _norm_cc(claimed_cc)
        effective_conf = round(max(cont_conf, sub_conf), 3)

        if not cc or cc not in CC_TO_CONTINENT:
            return Evidence(
                type="hier_geoguess",
                summary=base_summary,
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=round(min(0.5, effective_conf or 0.25), 3),
                data=hier,
            )

        claimed_cont = CC_TO_CONTINENT.get(cc)
        claimed_sub = CC_TO_SUBREGION.get(cc)
        claimed_name = COUNTRY_NAMES.get(cc, cc)
        top2_ccs = [_norm_cc(r.get("cc")) for r in top_countries[:2]]

        # 1. SUPPORTS if claimed_cc in top 2 countries or matching subregion
        if cc in top2_ccs or (sub_name and claimed_sub == sub_name):
            conf = round(max(0.45, min(0.90, effective_conf or 0.65)), 3)
            return Evidence(
                type="hier_geoguess",
                summary=f"{base_summary} — consistent with claimed {claimed_name} ({cc})",
                verdict=ClaimVerdict.SUPPORTS,
                confidence=conf,
                data=hier,
            )

        # 2. CONTRADICTS if claimed_cc is on a different continent when confidence >= 0.65
        if cont_code and claimed_cont and claimed_cont != cont_code and effective_conf >= 0.65:
            claimed_cont_name = CONTINENT_NAMES.get(claimed_cont, claimed_cont)
            return Evidence(
                type="hier_geoguess",
                summary=(
                    f"{base_summary} — conflicts with claimed {claimed_name} "
                    f"({cc}, {claimed_cont_name})"
                ),
                verdict=ClaimVerdict.CONTRADICTS,
                confidence=round(min(0.90, effective_conf), 3),
                data=hier,
            )

        # 3. Else INCONCLUSIVE
        return Evidence(
            type="hier_geoguess",
            summary=f"{base_summary} — inconclusive for claimed {claimed_name} ({cc})",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=round(min(0.50, effective_conf or 0.30), 3),
            data=hier,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/hier_geoguess.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        global passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name} {detail}")

    # 1. Verify CONTINENTS & subregions completeness
    expected_continents = {"EU", "NA", "SA", "AS", "AF", "OC"}
    expected_subregions = {
        "northern_europe", "western_europe", "southern_europe",
        "eastern_europe", "british_isles", "east_asia", "southeast_asia",
        "south_asia", "middle_east", "north_america", "central_america",
        "south_america", "north_africa", "sub_saharan_africa", "oceania",
    }
    check("continents keys", set(CONTINENTS.keys()) == expected_continents)
    found_subs = {
        sub
        for c in CONTINENTS.values()
        for sub in c["subregions"].keys()
    }
    check("subregions keys", found_subs == expected_subregions)
    check("DE in EU", "DE" in CONTINENTS["EU"] and "DE" in CONTINENTS["EU"]["western_europe"])

    # 2. Western Europe / Germany funnel
    obs_de = {
        "driving_side": "right",
        "script": "latin",
        "climate_zone": "temperate",
        "architecture": "fachwerk half-timbered",
        "road_lines": "white center",
        "sun_side": "south",
        "vegetation": ["deciduous", "beech", "oak"],
        "sign_lang": "de",
    }
    hier_de = infer_hierarchy(obs_de, {"DE": 0.82, "AT": 0.45})
    check("DE continent is EU", hier_de["continent"] == "EU" and hier_de["continent"]["code"] == "EU")
    check("DE subregion is western_europe", hier_de["subregion"] == "western_europe")
    check("DE top country is DE", hier_de["top_countries"][0]["cc"] == "DE")
    check(
        "DE funnel format",
        "Europe → Western Europe → Germany (DE)" in hier_de["funnel"],
        f"got {hier_de['funnel']!r}",
    )

    # 3. Evidence verdicts for DE hierarchy
    ev_sup_top = hierarchy_evidence(hier_de, "DE")
    check("Evidence SUPPORTS top-1 DE", ev_sup_top is not None and ev_sup_top.verdict.value == "supports")
    check("Evidence type hier_geoguess", ev_sup_top is not None and ev_sup_top.type == "hier_geoguess")

    ev_sup_sub = hierarchy_evidence(hier_de, "NL")
    check("Evidence SUPPORTS same subregion NL", ev_sup_sub is not None and ev_sup_sub.verdict.value == "supports")

    ev_con = hierarchy_evidence(hier_de, "BR")
    check("Evidence CONTRADICTS different continent BR", ev_con is not None and ev_con.verdict.value == "contradicts")

    ev_inc_same_cont = hierarchy_evidence(hier_de, "GR")
    check("Evidence INCONCLUSIVE same continent different subregion GR", ev_inc_same_cont is not None and ev_inc_same_cont.verdict.value == "inconclusive")

    # 4. Low-confidence different continent -> INCONCLUSIVE (not CONTRADICTS)
    weak_hier = {
        "continent": {"code": "EU", "name": "Europe", "confidence": 0.50},
        "subregion": {"name": "western_europe", "label": "Western Europe", "confidence": 0.45},
        "top_countries": [{"cc": "DE", "score": 0.45, "subregion": "western_europe"}],
        "funnel": "Europe → Western Europe → Germany (DE)",
    }
    ev_weak = hierarchy_evidence(weak_hier, "JP")
    check("Weak confidence (<0.65) diff continent -> INCONCLUSIVE", ev_weak is not None and ev_weak.verdict.value == "inconclusive")

    # 5. Observations-only inference (Southeast Asia / Thailand)
    obs_th = {
        "driving_side": "left",
        "script": "thai",
        "climate_zone": "tropical",
        "vegetation": ["palm", "banana", "rice paddy"],
        "architecture": "stilt",
    }
    hier_th = infer_hierarchy(obs_th)
    check("TH continent AS", hier_th["continent"] == "AS")
    check("TH subregion southeast_asia", hier_th["subregion"] == "southeast_asia")
    check("TH top country TH", hier_th["top_countries"][0]["cc"] == "TH")

    # 6. Never-raise resilience on malformed inputs
    check("Malformed obs None", isinstance(infer_hierarchy(None, None), dict))  # type: ignore[arg-type]
    check("Malformed obs list", isinstance(infer_hierarchy(["bad"], "bad"), dict))  # type: ignore[arg-type]
    check("Malformed evidence None", hierarchy_evidence(None, "DE") is None)  # type: ignore[arg-type]

    status_str = "PASS" if failed == 0 else "FAIL"
    print(f"\n{status_str}: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
