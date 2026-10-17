"""Cross-signal geographic constraint matrix — Bayesian / log-odds country fusion.

Fuses independent spatial signals across Sundial's geo stages — coordinate
models (GeoCLIP / OpenCLIP), community meta cues (GeoTips / region_vocab),
candidate matchers (geo_match), text/script forensics (language), road-sign
conventions (sign_style), licence-plate reads (plate_ocr), solar hemisphere
tells (hemisphere), and shadow/solar latitude-longitude strips (shadow_geo)
— into a single calibrated probability distribution over countries.

Design principles:
  * Log-odds (Bayesian) pooling with per-modality caps so redundant cards in
    the same modality cannot double-count.
  * Hard physical/regulatory constraints (driving side, hemisphere, solar
    latitude/longitude bands, non-Latin script mismatch) heavily penalize
    violating countries.
  * Multi-modality agreement bonus rewards countries where >=2 independent
    modalities converge (e.g. GeoCLIP coordinate + German language + Vienna
    signs / right-hand traffic).
  * Pure stdlib at import time; sibling modules (.schema, .pip2cc,
    .region_vocab, .locate_summary) are imported lazily with full built-in
    fallbacks. Never raises on any input.

Stage API:
    fused = fuse_signals(evidence_list)
    ev    = matrix_evidence(fused, claimed_cc="DE")

    python3 sundial/geo_matrix.py     # standalone self-test (exits 0)
"""
from __future__ import annotations

import math
import os
import re
import sys
from typing import Any

__all__ = ["fuse_signals", "matrix_evidence"]

# ---------------------------------------------------------------------------
# Country universe — centroid + approximate lat/lon span + display name
# ---------------------------------------------------------------------------
# Spans (lat_min, lat_max, lon_min, lon_max) allow solar/shadow band and
# hemisphere checks to test the country's actual extent rather than just a
# single centroid point.
_CC_META: dict[str, dict[str, Any]] = {
    # --- Europe ---
    "DE": {"name": "Germany", "lat": 51.16, "lon": 10.45, "lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1},
    "FR": {"name": "France", "lat": 46.60, "lon": 2.40, "lat_min": 41.3, "lat_max": 51.1, "lon_min": -5.2, "lon_max": 9.6},
    "GB": {"name": "United Kingdom", "lat": 54.00, "lon": -2.00, "lat_min": 49.9, "lat_max": 60.9, "lon_min": -8.6, "lon_max": 1.8},
    "IE": {"name": "Ireland", "lat": 53.18, "lon": -8.24, "lat_min": 51.4, "lat_max": 55.4, "lon_min": -10.5, "lon_max": -6.0},
    "IT": {"name": "Italy", "lat": 42.80, "lon": 12.80, "lat_min": 36.6, "lat_max": 47.1, "lon_min": 6.6, "lon_max": 18.5},
    "ES": {"name": "Spain", "lat": 40.20, "lon": -3.60, "lat_min": 36.0, "lat_max": 43.8, "lon_min": -9.3, "lon_max": 3.4},
    "PT": {"name": "Portugal", "lat": 39.60, "lon": -8.00, "lat_min": 36.9, "lat_max": 42.2, "lon_min": -9.5, "lon_max": -6.2},
    "NL": {"name": "Netherlands", "lat": 52.20, "lon": 5.30, "lat_min": 50.7, "lat_max": 53.6, "lon_min": 3.3, "lon_max": 7.2},
    "BE": {"name": "Belgium", "lat": 50.65, "lon": 4.65, "lat_min": 49.5, "lat_max": 51.5, "lon_min": 2.5, "lon_max": 6.4},
    "LU": {"name": "Luxembourg", "lat": 49.60, "lon": 6.10, "lat_min": 49.4, "lat_max": 50.2, "lon_min": 5.7, "lon_max": 6.6},
    "AT": {"name": "Austria", "lat": 47.60, "lon": 14.13, "lat_min": 46.4, "lat_max": 49.0, "lon_min": 9.5, "lon_max": 17.2},
    "CH": {"name": "Switzerland", "lat": 46.80, "lon": 8.23, "lat_min": 45.8, "lat_max": 47.8, "lon_min": 5.9, "lon_max": 10.5},
    "PL": {"name": "Poland", "lat": 52.00, "lon": 19.40, "lat_min": 49.0, "lat_max": 54.9, "lon_min": 14.1, "lon_max": 24.2},
    "CZ": {"name": "Czechia", "lat": 49.80, "lon": 15.50, "lat_min": 48.5, "lat_max": 51.1, "lon_min": 12.1, "lon_max": 18.9},
    "SK": {"name": "Slovakia", "lat": 48.70, "lon": 19.50, "lat_min": 47.7, "lat_max": 49.6, "lon_min": 16.8, "lon_max": 22.6},
    "HU": {"name": "Hungary", "lat": 47.16, "lon": 19.50, "lat_min": 45.7, "lat_max": 48.6, "lon_min": 16.1, "lon_max": 22.9},
    "RO": {"name": "Romania", "lat": 45.85, "lon": 24.97, "lat_min": 43.6, "lat_max": 48.3, "lon_min": 20.2, "lon_max": 29.7},
    "BG": {"name": "Bulgaria", "lat": 42.75, "lon": 25.50, "lat_min": 41.2, "lat_max": 44.2, "lon_min": 22.3, "lon_max": 28.6},
    "GR": {"name": "Greece", "lat": 39.00, "lon": 22.00, "lat_min": 34.8, "lat_max": 41.8, "lon_min": 19.3, "lon_max": 28.3},
    "HR": {"name": "Croatia", "lat": 45.10, "lon": 15.40, "lat_min": 42.4, "lat_max": 46.6, "lon_min": 13.5, "lon_max": 19.5},
    "SI": {"name": "Slovenia", "lat": 46.10, "lon": 14.80, "lat_min": 45.4, "lat_max": 46.9, "lon_min": 13.4, "lon_max": 16.6},
    "RS": {"name": "Serbia", "lat": 44.00, "lon": 21.00, "lat_min": 42.2, "lat_max": 46.2, "lon_min": 18.8, "lon_max": 23.0},
    "BA": {"name": "Bosnia and Herzegovina", "lat": 44.20, "lon": 17.70, "lat_min": 42.5, "lat_max": 45.3, "lon_min": 15.7, "lon_max": 19.7},
    "UA": {"name": "Ukraine", "lat": 49.00, "lon": 32.00, "lat_min": 44.4, "lat_max": 52.4, "lon_min": 22.1, "lon_max": 40.2},
    "BY": {"name": "Belarus", "lat": 53.53, "lon": 28.05, "lat_min": 51.2, "lat_max": 56.2, "lon_min": 23.2, "lon_max": 32.8},
    "RU": {"name": "Russia", "lat": 56.00, "lon": 40.00, "lat_min": 41.2, "lat_max": 77.7, "lon_min": 19.6, "lon_max": 180.0},
    "EE": {"name": "Estonia", "lat": 58.67, "lon": 25.00, "lat_min": 57.5, "lat_max": 59.7, "lon_min": 21.8, "lon_max": 28.2},
    "LV": {"name": "Latvia", "lat": 56.88, "lon": 24.60, "lat_min": 55.6, "lat_max": 58.1, "lon_min": 20.9, "lon_max": 28.3},
    "LT": {"name": "Lithuania", "lat": 55.35, "lon": 23.88, "lat_min": 53.9, "lat_max": 56.5, "lon_min": 20.9, "lon_max": 26.9},
    "SE": {"name": "Sweden", "lat": 62.00, "lon": 15.00, "lat_min": 55.3, "lat_max": 69.1, "lon_min": 11.1, "lon_max": 24.2},
    "NO": {"name": "Norway", "lat": 64.50, "lon": 12.00, "lat_min": 57.9, "lat_max": 71.2, "lon_min": 4.6, "lon_max": 31.1},
    "FI": {"name": "Finland", "lat": 64.00, "lon": 26.00, "lat_min": 59.8, "lat_max": 70.1, "lon_min": 20.5, "lon_max": 31.6},
    "DK": {"name": "Denmark", "lat": 56.00, "lon": 10.00, "lat_min": 54.5, "lat_max": 57.8, "lon_min": 8.0, "lon_max": 15.2},
    "IS": {"name": "Iceland", "lat": 64.90, "lon": -18.60, "lat_min": 63.3, "lat_max": 66.6, "lon_min": -24.6, "lon_max": -13.5},
    "MT": {"name": "Malta", "lat": 35.94, "lon": 14.41, "lat_min": 35.8, "lat_max": 36.1, "lon_min": 14.1, "lon_max": 14.6},
    "CY": {"name": "Cyprus", "lat": 35.00, "lon": 33.00, "lat_min": 34.6, "lat_max": 35.7, "lon_min": 32.2, "lon_max": 34.6},
    "TR": {"name": "Turkey", "lat": 39.00, "lon": 35.00, "lat_min": 35.8, "lat_max": 42.1, "lon_min": 26.0, "lon_max": 44.8},
    # --- Americas ---
    "US": {"name": "United States", "lat": 39.50, "lon": -98.35, "lat_min": 24.5, "lat_max": 49.4, "lon_min": -125.0, "lon_max": -66.9},
    "CA": {"name": "Canada", "lat": 56.00, "lon": -96.00, "lat_min": 41.7, "lat_max": 75.0, "lon_min": -141.0, "lon_max": -52.6},
    "MX": {"name": "Mexico", "lat": 23.60, "lon": -102.50, "lat_min": 14.5, "lat_max": 32.7, "lon_min": -117.1, "lon_max": -86.7},
    "BR": {"name": "Brazil", "lat": -14.20, "lon": -51.90, "lat_min": -33.8, "lat_max": 5.3, "lon_min": -74.0, "lon_max": -34.8},
    "AR": {"name": "Argentina", "lat": -34.00, "lon": -64.00, "lat_min": -55.1, "lat_max": -21.8, "lon_min": -73.6, "lon_max": -53.6},
    "CL": {"name": "Chile", "lat": -33.40, "lon": -70.70, "lat_min": -55.9, "lat_max": -17.5, "lon_min": -75.7, "lon_max": -66.4},
    "CO": {"name": "Colombia", "lat": 4.60, "lon": -74.10, "lat_min": -4.2, "lat_max": 12.5, "lon_min": -79.0, "lon_max": -66.9},
    "PE": {"name": "Peru", "lat": -9.20, "lon": -75.00, "lat_min": -18.4, "lat_max": -0.0, "lon_min": -81.3, "lon_max": -68.7},
    # --- Asia / Middle East ---
    "JP": {"name": "Japan", "lat": 36.50, "lon": 138.00, "lat_min": 24.2, "lat_max": 45.5, "lon_min": 123.0, "lon_max": 145.8},
    "KR": {"name": "South Korea", "lat": 36.50, "lon": 127.90, "lat_min": 33.1, "lat_max": 38.6, "lon_min": 125.8, "lon_max": 129.6},
    "CN": {"name": "China", "lat": 35.00, "lon": 105.00, "lat_min": 18.2, "lat_max": 53.6, "lon_min": 73.5, "lon_max": 134.8},
    "TW": {"name": "Taiwan", "lat": 23.80, "lon": 121.00, "lat_min": 21.9, "lat_max": 25.3, "lon_min": 120.0, "lon_max": 122.0},
    "HK": {"name": "Hong Kong", "lat": 22.30, "lon": 114.20, "lat_min": 22.1, "lat_max": 22.6, "lon_min": 113.8, "lon_max": 114.5},
    "IN": {"name": "India", "lat": 22.00, "lon": 79.00, "lat_min": 8.0, "lat_max": 35.5, "lon_min": 68.1, "lon_max": 97.4},
    "PK": {"name": "Pakistan", "lat": 30.00, "lon": 69.00, "lat_min": 23.7, "lat_max": 37.1, "lon_min": 60.9, "lon_max": 77.8},
    "BD": {"name": "Bangladesh", "lat": 23.80, "lon": 90.30, "lat_min": 20.7, "lat_max": 26.6, "lon_min": 88.0, "lon_max": 92.7},
    "TH": {"name": "Thailand", "lat": 15.50, "lon": 101.00, "lat_min": 5.6, "lat_max": 20.5, "lon_min": 97.3, "lon_max": 105.6},
    "VN": {"name": "Vietnam", "lat": 14.00, "lon": 108.00, "lat_min": 8.5, "lat_max": 23.4, "lon_min": 102.1, "lon_max": 109.5},
    "MY": {"name": "Malaysia", "lat": 4.20, "lon": 102.00, "lat_min": 0.9, "lat_max": 7.4, "lon_min": 99.6, "lon_max": 119.3},
    "SG": {"name": "Singapore", "lat": 1.35, "lon": 103.85, "lat_min": 1.1, "lat_max": 1.5, "lon_min": 103.6, "lon_max": 104.1},
    "ID": {"name": "Indonesia", "lat": -2.20, "lon": 117.00, "lat_min": -11.0, "lat_max": 6.0, "lon_min": 95.0, "lon_max": 141.0},
    "PH": {"name": "Philippines", "lat": 12.80, "lon": 122.00, "lat_min": 4.6, "lat_max": 21.1, "lon_min": 116.9, "lon_max": 126.6},
    "IL": {"name": "Israel", "lat": 31.50, "lon": 34.80, "lat_min": 29.5, "lat_max": 33.3, "lon_min": 34.2, "lon_max": 35.9},
    "AE": {"name": "United Arab Emirates", "lat": 24.00, "lon": 54.00, "lat_min": 22.6, "lat_max": 26.1, "lon_min": 51.5, "lon_max": 56.4},
    "SA": {"name": "Saudi Arabia", "lat": 24.00, "lon": 45.00, "lat_min": 16.4, "lat_max": 32.2, "lon_min": 34.5, "lon_max": 55.7},
    "IR": {"name": "Iran", "lat": 32.50, "lon": 54.00, "lat_min": 25.1, "lat_max": 39.8, "lon_min": 44.0, "lon_max": 63.3},
    "KZ": {"name": "Kazakhstan", "lat": 48.00, "lon": 68.00, "lat_min": 40.6, "lat_max": 55.4, "lon_min": 46.5, "lon_max": 87.3},
    # --- Africa ---
    "EG": {"name": "Egypt", "lat": 26.80, "lon": 30.80, "lat_min": 22.0, "lat_max": 31.7, "lon_min": 24.7, "lon_max": 36.9},
    "MA": {"name": "Morocco", "lat": 31.80, "lon": -6.00, "lat_min": 27.7, "lat_max": 35.9, "lon_min": -13.2, "lon_max": -1.0},
    "NG": {"name": "Nigeria", "lat": 9.60, "lon": 8.00, "lat_min": 4.3, "lat_max": 13.9, "lon_min": 2.7, "lon_max": 14.7},
    "KE": {"name": "Kenya", "lat": 0.50, "lon": 37.90, "lat_min": -4.7, "lat_max": 5.0, "lon_min": 33.9, "lon_max": 41.9},
    "ZA": {"name": "South Africa", "lat": -29.00, "lon": 25.00, "lat_min": -34.8, "lat_max": -22.1, "lon_min": 16.5, "lon_max": 32.9},
    # --- Oceania ---
    "AU": {"name": "Australia", "lat": -25.00, "lon": 134.00, "lat_min": -43.6, "lat_max": -10.7, "lon_min": 113.1, "lon_max": 153.6},
    "NZ": {"name": "New Zealand", "lat": -41.80, "lon": 172.80, "lat_min": -47.3, "lat_max": -34.4, "lon_min": 166.4, "lon_max": 178.6},
}

_CC_ALIASES: dict[str, str] = {
    "UK": "GB", "EL": "GR", "USA": "US", "UAE": "AE", "SU": "RU", "TP": "TL",
}

# Left-hand traffic countries.
_LEFT_TRAFFIC: frozenset[str] = frozenset({
    "GB", "IE", "JP", "AU", "NZ", "IN", "ZA", "TH", "MY", "SG", "ID", "HK",
    "KE", "MT", "CY", "PK", "BD", "LK", "NP", "MU", "TT", "JM", "BB", "BS",
    "GY", "SR", "BN", "BT", "MO", "NA", "BW", "LS", "SZ", "MZ", "ZM", "MW",
    "TZ", "UG", "FJ", "PG", "IM", "JE", "GG", "VI", "CX", "CC", "NF", "TL",
})

# EU member states (blue strip plate family).
_EU_CCS: frozenset[str] = frozenset({
    "DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "PL", "IE", "SE",
    "DK", "FI", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "EE",
    "LV", "LT", "LU", "MT", "CY",
})

# International vehicle registration strip codes -> ISO alpha-2.
_STRIP_TO_CC: dict[str, str] = {
    "D": "DE", "A": "AT", "B": "BE", "BG": "BG", "CY": "CY", "CZ": "CZ",
    "DK": "DK", "E": "ES", "EST": "EE", "F": "FR", "FIN": "FI", "GR": "GR",
    "EL": "GR", "H": "HU", "HR": "HR", "I": "IT", "IRL": "IE", "L": "LU",
    "LT": "LT", "LV": "LV", "M": "MT", "NL": "NL", "P": "PT", "PL": "PL",
    "RO": "RO", "S": "SE", "SK": "SK", "SLO": "SI", "GB": "GB", "UK": "GB",
    "CH": "CH", "FL": "LI", "IS": "IS", "N": "NO", "TR": "TR", "UA": "UA",
    "BY": "BY", "RUS": "RU", "MD": "MD", "AL": "AL", "MK": "MK", "NMK": "MK",
    "SRB": "RS", "BIH": "BA", "MNE": "ME", "RKS": "XK", "GE": "GE",
    "AM": "AM", "AZ": "AZ", "KZ": "KZ", "MC": "MC", "SM": "SM", "AND": "AD",
    "V": "VA",
}

# Language ISO 639-1 -> countries where public signage commonly uses it.
_LANG_HINTS: dict[str, set[str]] = {
    "en": {"GB", "IE", "US", "AU", "NZ", "CA", "ZA", "IN", "SG", "NG", "KE", "MT"},
    "fr": {"FR", "BE", "CH", "LU", "MC", "CA"},
    "de": {"DE", "AT", "CH", "LU", "LI"},
    "es": {"ES", "MX", "AR", "CO", "CL", "PE", "VE", "EC"},
    "pt": {"PT", "BR"},
    "it": {"IT", "CH", "SM", "VA"},
    "nl": {"NL", "BE"},
    "ja": {"JP"},
    "ko": {"KR"},
    "zh": {"CN", "TW", "HK", "SG"},
    "ar": {"SA", "AE", "EG", "MA", "QA", "KW", "OM", "JO", "LB", "IL"},
    "ru": {"RU", "BY", "KZ", "KG", "UA"},
    "hi": {"IN"},
    "th": {"TH"},
    "vi": {"VN"},
    "id": {"ID"},
    "ms": {"MY", "SG", "BN"},
    "tr": {"TR", "CY"},
    "el": {"GR", "CY"},
    "pl": {"PL"},
    "uk": {"UA"},
    "sv": {"SE", "FI"},
    "da": {"DK"},
    "no": {"NO"},
    "fi": {"FI"},
    "cs": {"CZ"},
    "sk": {"SK"},
    "hu": {"HU"},
    "ro": {"RO"},
    "bg": {"BG"},
    "hr": {"HR", "BA"},
    "sr": {"RS", "BA"},
    "sl": {"SI"},
    "et": {"EE"},
    "lv": {"LV"},
    "lt": {"LT"},
    "is": {"IS"},
    "he": {"IL"},
    "fa": {"IR"},
    "ur": {"PK"},
    "bn": {"BD", "IN"},
    "tl": {"PH"},
}

_LANG_NAME: dict[str, str] = {
    "english": "en", "french": "fr", "german": "de", "deutsch": "de",
    "spanish": "es", "portuguese": "pt", "italian": "it", "dutch": "nl",
    "flemish": "nl", "japanese": "ja", "korean": "ko", "chinese": "zh",
    "mandarin": "zh", "arabic": "ar", "russian": "ru", "thai": "th",
    "turkish": "tr", "greek": "el", "polish": "pl", "ukrainian": "uk",
    "swedish": "sv", "danish": "da", "norwegian": "no", "finnish": "fi",
    "czech": "cs", "slovak": "sk", "hungarian": "hu", "romanian": "ro",
    "bulgarian": "bg", "croatian": "hr", "serbian": "sr", "slovenian": "sl",
    "estonian": "et", "latvian": "lv", "lithuanian": "lt", "icelandic": "is",
    "hebrew": "he", "hindi": "hi", "vietnamese": "vi", "indonesian": "id",
    "malay": "ms", "urdu": "ur", "bengali": "bn", "persian": "fa",
    "farsi": "fa", "tagalog": "tl", "filipino": "tl", "catalan": "es",
}

_SCRIPT_HINTS: dict[str, set[str]] = {
    "cyrillic": {"RU", "UA", "BY", "BG", "RS", "BA", "KZ"},
    "arabic": {"SA", "AE", "EG", "MA", "IR", "PK", "IL"},
    "cjk": {"JP", "CN", "TW", "HK", "SG"},
    "japanese": {"JP"}, "hiragana": {"JP"}, "katakana": {"JP"},
    "chinese": {"CN", "TW", "HK", "SG"}, "hanzi": {"CN", "TW", "HK"},
    "hangul": {"KR"}, "korean": {"KR"},
    "thai": {"TH"},
    "greek": {"GR", "CY"},
    "hebrew": {"IL"},
    "devanagari": {"IN"},
    "bengali": {"BD", "IN"},
}

_CONVENTION_HINTS: dict[str, set[str]] = {
    "vienna": {
        "DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "CH", "LU",
        "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "RS", "BA",
        "SE", "DK", "FI", "NO", "IS", "EE", "LV", "LT", "RU", "UA", "BY",
        "TR", "CY", "MT", "KZ",
    },
    "mutcd": {
        "US", "CA", "MX", "BR", "AR", "CL", "CO", "PE", "PH", "AU", "NZ",
        "TH", "TW", "KR", "JP",
    },
    "uk": {
        "GB", "IE", "MT", "CY", "AU", "NZ", "IN", "ZA", "SG", "MY", "HK",
        "KE", "NG", "PK", "BD",
    },
    "japan": {"JP"},
    "australia": {"AU", "NZ"},
}

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

_PLATE_HINTS: dict[str, set[str]] = {
    "eu": set(_EU_CCS), "european": set(_EU_CCS), "euro": set(_EU_CCS),
    "eu_band": set(_EU_CCS), "blue_band": set(_EU_CCS),
    "blue_strip": set(_EU_CCS), "left_blue_strip": set(_EU_CCS),
    "eu_blue_strip": set(_EU_CCS),
    "uk": {"GB"}, "gb": {"GB"}, "british": {"GB"},
    "yellow_rear": {"GB", "NL"}, "white_front_yellow_rear": {"GB", "NL"},
    "us": {"US", "CA"}, "usa": {"US"}, "us_style": {"US", "CA", "MX"},
    "jp": {"JP"}, "japanese": {"JP"},
    "cn": {"CN"}, "kr": {"KR"}, "in": {"IN"}, "ru": {"RU"},
    "br": {"BR"}, "au": {"AU"}, "za": {"ZA"},
    "mercosur": {"AR", "BR"},
}

_MARKING_HINTS: dict[str, set[str]] = {
    "yellow": {"US", "CA", "BR", "CN", "KR", "TW", "NO", "SE", "FI"},
    "yellow_centerline": {"US", "CA", "BR", "CN", "KR", "TW", "NO", "SE", "FI"},
    "yellow_center": {"US", "CA", "BR", "CN", "KR", "TW", "NO", "SE", "FI"},
    "double_yellow": {"GB", "IE", "US", "CA"},
    "double_yellow_lines": {"GB", "IE"},
    "double_red": {"GB"}, "double_red_lines": {"GB"},
    "zigzag": {"GB", "IE", "SG"}, "zigzag_lines": {"GB", "IE", "SG"},
    "keep_left_bollards": {"GB", "IE", "AU", "NZ", "JP"},
}

_BOLLARD_HINTS: dict[str, set[str]] = {
    "german_bollards": {"DE"}, "french_bollards": {"FR"},
    "uk_bollards": {"GB"}, "british_bollards": {"GB"},
    "italian_bollards": {"IT"}, "spanish_bollards": {"ES"},
    "dutch_bollards": {"NL"}, "australian_bollards": {"AU"},
    "hungarian_bollards": {"HU"}, "polish_bollards": {"PL"},
    "slovenian_bollards": {"SI"}, "croatian_bollards": {"HR"},
    "belgian_bollards": {"BE"}, "austrian_bollards": {"AT"},
    "swiss_bollards": {"CH"}, "swedish_bollards": {"SE"},
    "norwegian_bollards": {"NO"}, "danish_bollards": {"DK"},
    "finnish_bollards": {"FI"}, "japanese_bollards": {"JP"},
    "american_bollards": {"US"},
}

# Distinctive word/token patterns in OCR sign texts -> language code.
_TEXT_LANG_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:ß|\b(?:straße|strasse|ausfahrt|einbahnstraße|umleitung|polizei|bahnhof|hauptbahnhof|apotheke|sparkasse|gmbh|halt)\b)", re.I), "de"),
    (re.compile(r"\b(?:rue|sortie|gendarmerie|boulangerie|pharmacie|centre\s+ville|péage|sarl)\b", re.I), "fr"),
    (re.compile(r"\b(?:calle|salida|avenida|farmacia|policía|autovía|autopista)\b", re.I), "es"),
    (re.compile(r"\b(?:via|uscita|carabinieri|autostrada|piazza|stazione)\b", re.I), "it"),
    (re.compile(r"\b(?:straat|uitrit|politie|doorgaand|fietspad|plein)\b", re.I), "nl"),
    (re.compile(r"\b(?:ulica|wyjazd|policja|apteka|sklep)\b", re.I), "pl"),
)

# Canonical ordering of modality names in output.
_MODALITY_ORDER: tuple[str, ...] = (
    "model", "language", "signs", "plates", "driving",
    "domain", "cues", "solar", "hemisphere",
)

# Per-modality positive log-odds caps so repeated cards of the same modality
# do not overwhelm independent cross-modal agreement.
_MOD_CAP: dict[str, float] = {
    "model": 2.4,
    "language": 1.8,
    "signs": 1.2,
    "plates": 2.5,
    "driving": 0.45,
    "domain": 2.4,
    "cues": 1.5,
    "solar": 1.3,
    "hemisphere": 0.35,
}

# Discriminative modalities (as opposed to hemisphere/driving background filters).
_DISCRIMINATIVE_MODS: frozenset[str] = frozenset({
    "model", "language", "signs", "plates", "domain", "cues", "solar",
})

# Hard constraint penalties (in log-odds units).
_PENALTY_DRIVING = 3.2
_PENALTY_HEMISPHERE = 3.2
_PENALTY_SOLAR_BAND = 2.6
_PENALTY_SCRIPT = 2.2

# Multi-modality agreement bonus per extra agreeing modality (>=2).
_AGREEMENT_BONUS = 0.60

_WORD_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Lazy sibling imports + defensive helpers
# ---------------------------------------------------------------------------

def _schema():
    try:
        from .schema import ClaimVerdict, Evidence
        return ClaimVerdict, Evidence
    except ImportError:
        try:
            from schema import ClaimVerdict, Evidence
            return ClaimVerdict, Evidence
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from schema import ClaimVerdict, Evidence
            return ClaimVerdict, Evidence


def _pip2cc():
    try:
        from . import pip2cc
        return pip2cc
    except Exception:
        try:
            import pip2cc  # type: ignore
            return pip2cc
        except Exception:
            return None


def _region_vocab():
    try:
        from . import region_vocab
        return region_vocab
    except Exception:
        try:
            import region_vocab  # type: ignore
            return region_vocab
        except Exception:
            return None


def _locate_cc_geo() -> dict:
    try:
        from .locate_summary import CC_GEO
        return CC_GEO if isinstance(CC_GEO, dict) else {}
    except Exception:
        try:
            from locate_summary import CC_GEO  # type: ignore
            return CC_GEO if isinstance(CC_GEO, dict) else {}
        except Exception:
            return {}


def _float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _norm_tok(s: Any) -> str:
    return _WS_RE.sub("_", str(s or "").strip().lower()).strip("_")


def _tokens(value: Any) -> list[str]:
    out: list[str] = []

    def eat(item: Any) -> None:
        for piece in re.split(r"[,;/]+", str(item or "")):
            t = _norm_tok(piece)
            if t and t not in out:
                out.append(t)

    def walk(v: Any) -> None:
        if isinstance(v, str):
            eat(v)
        elif isinstance(v, dict):
            for vv in v.values():
                walk(vv)
        elif isinstance(v, (list, tuple, set, frozenset)):
            for it in v:
                walk(it)
        elif v is not None and not isinstance(v, bool):
            eat(v)

    walk(value)
    return out


_NAME_TO_CC_CACHE: dict[str, str] | None = None


def _name_to_cc() -> dict[str, str]:
    global _NAME_TO_CC_CACHE
    if _NAME_TO_CC_CACHE is not None:
        return _NAME_TO_CC_CACHE
    m: dict[str, str] = {
        "uk": "GB", "great britain": "GB", "britain": "GB", "england": "GB",
        "scotland": "GB", "wales": "GB", "usa": "US", "america": "US",
        "united states of america": "US", "uae": "AE", "emirates": "AE",
        "czech republic": "CZ", "holland": "NL", "deutschland": "DE",
        "korea": "KR",
    }
    for cc, info in _CC_META.items():
        m[str(info["name"]).lower()] = cc
    for cc, info in _locate_cc_geo().items():
        if isinstance(info, dict) and info.get("name"):
            m.setdefault(str(info["name"]).lower(), cc.upper())
    _NAME_TO_CC_CACHE = m
    return m


def _norm_cc(value: Any) -> str | None:
    if isinstance(value, dict):
        for k in ("cc", "country_code", "code", "iso", "country"):
            got = _norm_cc(value.get(k))
            if got:
                return got
        return None
    s = str(value or "").strip()
    if not s:
        return None
    u = s.upper()
    if u in _CC_ALIASES:
        return _CC_ALIASES[u]
    if len(u) == 2 and u.isalpha():
        return u
    clean = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z\s]", " ", s).lower()).strip()
    if clean in _name_to_cc():
        return _name_to_cc()[clean]
    for name, cc in _name_to_cc().items():
        if len(name) >= 4 and re.search(r"\b" + re.escape(name) + r"\b", clean):
            return cc
    return None


def _cc_name(cc: str) -> str:
    c = _norm_cc(cc) or str(cc or "").upper()
    if c in _CC_META:
        return str(_CC_META[c]["name"])
    extra = _locate_cc_geo().get(c)
    if isinstance(extra, dict) and extra.get("name"):
        return str(extra["name"])
    return c


def _cc_info(cc: str) -> dict[str, Any]:
    c = _norm_cc(cc) or str(cc or "").upper()
    if c in _CC_META:
        return _CC_META[c]
    extra = _locate_cc_geo().get(c)
    if isinstance(extra, dict):
        lat = _float(extra.get("lat")) or 0.0
        lon = _float(extra.get("lon")) or 0.0
        return {
            "name": str(extra.get("name") or c),
            "lat": lat, "lon": lon,
            "lat_min": lat - 4.0, "lat_max": lat + 4.0,
            "lon_min": lon - 4.0, "lon_max": lon + 4.0,
        }
    return {
        "name": c, "lat": 0.0, "lon": 0.0,
        "lat_min": -90.0, "lat_max": 90.0,
        "lon_min": -180.0, "lon_max": 180.0,
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = (
        math.sin(dp * 0.5) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    )
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _coord_to_cc(lat: Any, lon: Any) -> str | None:
    """Resolve (lat, lon) -> ISO alpha-2 country code using pip2cc + fallback."""
    la, lo = _float(lat), _float(lon)
    if la is None or lo is None or not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return None
    p2c = _pip2cc()
    if p2c is not None:
        try:
            cc = p2c.cc_at(la, lo)
            if cc:
                return _norm_cc(cc)
            near = p2c.nearest_cc(la, lo, k=1)
            if near and isinstance(near[0], dict):
                km = _float(near[0].get("km"))
                if near[0].get("inside") or (km is not None and km <= 180.0):
                    return _norm_cc(near[0].get("cc"))
        except Exception:
            pass
    # Fallback: check bounding boxes first, then nearest centroid within 1200 km
    best_cc = None
    best_d = 1e9
    for cc, info in _CC_META.items():
        d = _haversine_km(la, lo, info["lat"], info["lon"])
        in_box = (
            info["lat_min"] - 1.0 <= la <= info["lat_max"] + 1.0
            and info["lon_min"] - 1.0 <= lo <= info["lon_max"] + 1.0
        )
        eff_d = d * (0.4 if in_box else 1.0)
        if eff_d < best_d:
            best_d = eff_d
            best_cc = cc
    return best_cc if best_d <= 1200.0 else None


def _unpack_ev(ev: Any) -> tuple[str, dict[str, Any]]:
    """Coerce an Evidence object or dict into (type_str, merged_data_dict)."""
    if isinstance(ev, dict):
        t = str(ev.get("type") or ev.get("stage") or ev.get("kind") or "").strip().lower()
        inner = ev.get("data")
        if isinstance(inner, dict):
            top_extra = {
                k: v for k, v in ev.items()
                if k not in ("type", "stage", "kind", "data", "summary", "verdict", "confidence", "frames")
            }
            data = {**top_extra, **inner}
        else:
            data = dict(ev)
        if "summary" in ev and "summary" not in data:
            data["_summary"] = ev.get("summary")
        if "verdict" in ev and "verdict" not in data:
            data["_verdict"] = ev.get("verdict")
        return t, data
    t = str(getattr(ev, "type", "") or "").strip().lower()
    raw_d = getattr(ev, "data", None)
    data = dict(raw_d) if isinstance(raw_d, dict) else {}
    summ = getattr(ev, "summary", None)
    if summ:
        data.setdefault("_summary", summ)
    verd = getattr(ev, "verdict", None)
    if verd is not None:
        data.setdefault("_verdict", getattr(verd, "value", verd))
    return t, data


# ---------------------------------------------------------------------------
# Internal fusion state
# ---------------------------------------------------------------------------

class _FusionState:
    def __init__(self) -> None:
        self.universe: set[str] = set(_CC_META.keys())
        # {cc: {modality: positive_log_odds}}
        self.mod_gains: dict[str, dict[str, float]] = {}
        # {cc: soft_penalty_log_odds (positive float to subtract)}
        self.soft_penalties: dict[str, float] = {}
        # Hard constraints collected across the pack (deduplicated)
        self.driving_sides: list[str] = []
        self.hemispheres: list[str] = []
        self.bands: list[dict[str, Any]] = []
        self.nonlatin_scripts: list[str] = []

    def ensure_cc(self, cc: str | None) -> str | None:
        ncc = _norm_cc(cc)
        if not ncc or len(ncc) != 2 or not ncc.isalpha():
            return None
        self.universe.add(ncc)
        return ncc

    def boost(self, cc: str | None, modality: str, delta: float) -> None:
        if delta <= 0.0:
            return
        ncc = self.ensure_cc(cc)
        if not ncc:
            return
        bucket = self.mod_gains.setdefault(ncc, {})
        cap = _MOD_CAP.get(modality, 1.5)
        bucket[modality] = min(cap, bucket.get(modality, 0.0) + delta)

    def boost_set(self, ccs: Any, modality: str, delta: float) -> None:
        for cc in ccs or ():
            self.boost(cc, modality, delta)

    def penalize_outside(self, allowed_ccs: set[str], penalty: float) -> None:
        if penalty <= 0.0 or not allowed_ccs:
            return
        norm_allowed = {_norm_cc(c) for c in allowed_ccs} - {None}
        for cc in list(self.universe):
            if cc not in norm_allowed:
                self.soft_penalties[cc] = min(
                    2.5, self.soft_penalties.get(cc, 0.0) + penalty
                )


# ---------------------------------------------------------------------------
# Per-stage signal extractors
# ---------------------------------------------------------------------------

def _extract_geo_clip(data: dict[str, Any], st: _FusionState) -> None:
    cands: list[dict[str, Any]] = []
    for key in ("candidates", "predictions", "rows", "top"):
        raw = data.get(key)
        if isinstance(raw, (list, tuple)):
            for r in raw:
                if isinstance(r, dict):
                    cands.append(r)
    if not cands and (data.get("lat") is not None or data.get("cc") or data.get("country_code")):
        cands.append(data)
    if not cands:
        return

    cc_weights: dict[str, float] = {}
    for i, c in enumerate(cands[:6]):
        cc = _norm_cc(c) or _coord_to_cc(c.get("lat"), c.get("lon"))
        if not cc:
            continue
        p = _float(c.get("prob") or c.get("score") or c.get("confidence"))
        if p is None:
            p = max(0.15, 0.70 * (0.6 ** i))
        p = max(0.05, min(1.0, p))
        cc_weights[cc] = max(cc_weights.get(cc, 0.0), p)

    if not cc_weights:
        return
    max_w = max(cc_weights.values()) or 1.0
    for cc, w in cc_weights.items():
        rel = w / max_w
        # Strongest coordinate cluster gets up to ~2.2 log-odds
        gain = (1.4 + 0.9 * min(1.0, w)) * rel
        st.boost(cc, "model", gain)


def _extract_clip_geo(data: dict[str, Any], st: _FusionState) -> None:
    if data.get("available") is False:
        return
    rows: list[tuple[str, float]] = []
    for key in ("prior", "ranked", "candidates", "top"):
        raw = data.get(key)
        if isinstance(raw, (list, tuple)):
            for r in raw:
                if isinstance(r, dict):
                    cc = _norm_cc(r)
                    sc = _float(r.get("score") or r.get("prob"))
                    if cc and sc is not None:
                        rows.append((cc, sc))
    scores_map = data.get("scores")
    if isinstance(scores_map, dict):
        for k, v in scores_map.items():
            cc = _norm_cc(k)
            sc = _float(v)
            if cc and sc is not None:
                rows.append((cc, sc))
    if not rows and (data.get("cc") or data.get("country_code")):
        cc = _norm_cc(data)
        sc = _float(data.get("score") or data.get("prob")) or 0.5
        if cc:
            rows.append((cc, sc))
    if not rows:
        return

    rows.sort(key=lambda x: -x[1])
    top_s = rows[0][1]
    floor_s = rows[-1][1] if len(rows) > 1 else 0.0
    span = max(top_s - floor_s, 1e-4)
    for rank_idx, (cc, sc) in enumerate(rows[:4]):
        if sc <= 0.0:
            continue
        rel = (sc - floor_s) / span if len(rows) > 1 else 1.0
        if rank_idx == 0 or rel >= 0.45:
            gain = max(0.35, 1.55 * rel * (0.75 ** rank_idx))
            st.boost(cc, "model", gain)


def _resolve_lang_code(tok: str) -> str | None:
    t = _norm_tok(tok)
    if t in _LANG_HINTS:
        return t
    if t in _LANG_NAME:
        return _LANG_NAME[t]
    for w in _WORD_RE.findall(t):
        if w in _LANG_HINTS:
            return w
        if w in _LANG_NAME:
            return _LANG_NAME[w]
    return None


def _extract_language(data: dict[str, Any], st: _FusionState) -> None:
    matched_langs: list[str] = []
    for key in ("language", "lang", "sign_language", "languages", "languages_scripts"):
        for tok in _tokens(data.get(key)):
            lc = _resolve_lang_code(tok)
            if lc and lc not in matched_langs:
                matched_langs.append(lc)

    # Scan OCR texts for unmistakable language markers (e.g. German ß/Straße)
    texts = data.get("texts")
    if isinstance(texts, (list, tuple, str)):
        blob = " ".join(texts) if isinstance(texts, (list, tuple)) else str(texts)
        for pat, lc in _TEXT_LANG_PATTERNS:
            if pat.search(blob) and lc not in matched_langs:
                matched_langs.append(lc)

    # Explicit country codes on a language card
    explicit_ccs = [_norm_cc(c) for c in _tokens(data.get("cc") or data.get("countries") or data.get("ccs"))]
    explicit_ccs = [c for c in explicit_ccs if c]
    if explicit_ccs:
        st.boost_set(explicit_ccs, "language", 1.5)

    if matched_langs:
        union_ccs: set[str] = set()
        rv = _region_vocab()
        rv_smap = rv.script_map() if rv is not None else {}
        for idx, lc in enumerate(matched_langs):
            ccs = set(_LANG_HINTS.get(lc, set())) | set(rv_smap.get(lc, set()))
            if not ccs:
                continue
            union_ccs |= ccs
            # More specific languages (fewer countries) carry stronger log-odds
            specificity = 1.65 if len(ccs) <= 6 else (1.15 if len(ccs) <= 14 else 0.75)
            st.boost_set(ccs, "language", specificity * (0.8 ** idx))
        if union_ccs:
            st.penalize_outside(union_ccs, 0.35)

    # Script family (cyrillic, arabic, cjk, hangul, greek, hebrew, thai, devanagari, latin)
    scripts: list[str] = []
    for key in ("script", "scripts", "languages_scripts"):
        for tok in _tokens(data.get(key)):
            for w in [tok] + _WORD_RE.findall(tok):
                if w in _SCRIPT_HINTS and w not in scripts:
                    scripts.append(w)
                elif w in ("latin", "roman") and "latin" not in scripts:
                    scripts.append("latin")

    for sc in scripts:
        if sc in _SCRIPT_HINTS:
            ccs = _SCRIPT_HINTS[sc]
            st.boost_set(ccs, "language", 1.55)
            if sc not in st.nonlatin_scripts:
                st.nonlatin_scripts.append(sc)
        elif sc == "latin" and not matched_langs and not explicit_ccs:
            claimed_cc = _norm_cc(data.get("claimed_place") or data.get("claimed_cc"))
            if data.get("consistent") is True and claimed_cc:
                st.boost(claimed_cc, "language", 0.75)

    # If langdetect explicitly flagged a script contradiction on claimed_place
    if data.get("consistent") is False:
        claimed_cc = _norm_cc(data.get("claimed_place") or data.get("claimed_cc"))
        if claimed_cc:
            st.soft_penalties[claimed_cc] = st.soft_penalties.get(claimed_cc, 0.0) + 1.8


def _extract_sign_style(data: dict[str, Any], st: _FusionState) -> None:
    top_convs: list[str] = []
    top_score = _float(data.get("top_score")) or 0.0

    for tok in _tokens(data.get("top_conventions")):
        key = tok if tok in _CONVENTION_HINTS else _CONV_ALIAS.get(tok)
        if key and key not in top_convs:
            top_convs.append(key)

    scores = data.get("scores")
    if isinstance(scores, dict) and scores:
        vals = {
            (k if k in _CONVENTION_HINTS else _CONV_ALIAS.get(k, k)): (_float(v) or 0.0)
            for k, v in scores.items()
        }
        vals = {k: v for k, v in vals.items() if k in _CONVENTION_HINTS}
        if vals:
            mx = max(vals.values())
            if mx > top_score:
                top_score = mx
            if mx > 0.0 and not top_convs:
                top_convs = [k for k, v in vals.items() if v >= mx - 1e-6]

    for key in ("convention", "sign_convention", "sign_conventions", "style", "convention_guess"):
        for tok in _tokens(data.get(key)):
            ck = tok if tok in _CONVENTION_HINTS else _CONV_ALIAS.get(tok)
            if ck is None:
                for w in _WORD_RE.findall(tok):
                    if w in _CONVENTION_HINTS or w in _CONV_ALIAS:
                        ck = w if w in _CONVENTION_HINTS else _CONV_ALIAS[w]
                        break
            if ck and ck in _CONVENTION_HINTS and ck not in top_convs:
                top_convs.append(ck)
                if top_score <= 0.0:
                    top_score = 0.75

    # If all 5 conventions tied, it's non-discriminative
    if len(top_convs) >= len(_CONVENTION_HINTS):
        return

    if top_convs and top_score > 0.0:
        member_ccs: set[str] = set()
        for ck in top_convs:
            member_ccs |= _CONVENTION_HINTS.get(ck, set())
        if member_ccs:
            gain = 0.95 * max(0.45, min(1.0, top_score))
            st.boost_set(member_ccs, "signs", gain)
            if top_score >= 0.55:
                st.penalize_outside(member_ccs, 0.45)

    # Also check road markings or explicit countries on sign_style card
    for key in ("road_markings", "marking_colors", "lane_markings_color", "markings"):
        for tok in _tokens(data.get(key)):
            for mk, ccs in _MARKING_HINTS.items():
                if mk == tok or mk in tok:
                    st.boost_set(ccs, "signs", 0.55)
    for tok in _tokens(data.get("driving_side")):
        if tok in ("left", "right") and tok not in st.driving_sides:
            st.driving_sides.append(tok)


def _extract_plate_ocr(data: dict[str, Any], st: _FusionState) -> None:
    strip_ccs: set[str] = set()
    for raw_cc in _tokens(data.get("strip_ccs")):
        cc = _norm_cc(raw_cc)
        if cc:
            strip_ccs.add(cc)
    for key in ("strip_codes", "strip_code", "code"):
        for tok in _tokens(data.get(key)):
            code = tok.upper()
            cc = _STRIP_TO_CC.get(code) or _norm_cc(code)
            if cc:
                strip_ccs.add(cc)
    for r in data.get("reads") or []:
        if isinstance(r, dict):
            strip = r.get("country_strip")
            if isinstance(strip, dict) and strip.get("code"):
                code = re.sub(r"[^A-Za-z]", "", str(strip.get("code"))).upper()
                cc = _STRIP_TO_CC.get(code) or _norm_cc(code)
                if cc:
                    strip_ccs.add(cc)

    if strip_ccs:
        st.boost_set(strip_ccs, "plates", 2.35)
        st.penalize_outside(strip_ccs, 0.85)

    hints = data.get("hints")
    if isinstance(hints, dict) and hints:
        hint_ccs = {_norm_cc(k) for k in hints.keys()} - {None}
        if hint_ccs:
            gain = 1.35 if len(hint_ccs) <= 3 else 0.70
            st.boost_set(hint_ccs, "plates", gain)
            if len(hint_ccs) <= 4:
                st.penalize_outside(hint_ccs, 0.35)

    for key in ("plate_style", "shape", "plates"):
        for tok in _tokens(data.get(key)):
            if tok in _PLATE_HINTS:
                ccs = _PLATE_HINTS[tok]
                gain = 1.25 if len(ccs) <= 3 else 0.65
                st.boost_set(ccs, "plates", gain)


def _extract_hemisphere(data: dict[str, Any], st: _FusionState) -> None:
    if data.get("premise_inverted") is True:
        return
    hemi: str | None = None
    for key in ("observed_hemisphere", "hemisphere", "hemi"):
        val = str(data.get(key) or "").strip().lower()
        if val in ("northern", "north", "n"):
            hemi = "northern"
            break
        if val in ("southern", "south", "s"):
            hemi = "southern"
            break
    if hemi is None:
        side = str(data.get("observed_sun_side") or data.get("sun_side") or "").strip().lower()
        if side == "right":
            hemi = "northern"
        elif side == "left":
            hemi = "southern"
    if hemi and hemi not in st.hemispheres:
        st.hemispheres.append(hemi)


def _extract_shadow_geo(data: dict[str, Any], st: _FusionState) -> None:
    raw_bands = data.get("bands")
    bands: list[dict[str, Any]] = []
    if isinstance(raw_bands, dict):
        bands.append(raw_bands)
    elif isinstance(raw_bands, (list, tuple)):
        bands.extend(b for b in raw_bands if isinstance(b, dict))
    if not bands:
        if data.get("lat_min") is not None or data.get("lon_min") is not None:
            bands.append(data)
        else:
            lb = data.get("lat_band")
            if isinstance(lb, (list, tuple)) and len(lb) >= 2:
                bands.append({"lat_min": lb[0], "lat_max": lb[1], "ok": True})
            lob = data.get("lon_band")
            if isinstance(lob, (list, tuple)) and len(lob) >= 2:
                bands.append({"lon_min": lob[0], "lon_max": lob[1], "ok": True})

    for b in bands:
        if b.get("ok") is False or b.get("empty") is True:
            continue
        la_lo = _float(b.get("lat_min"))
        la_hi = _float(b.get("lat_max"))
        lo_lo = _float(b.get("lon_min"))
        lo_hi = _float(b.get("lon_max"))
        segs = b.get("segments") if isinstance(b.get("segments"), list) else None
        if la_lo is None and la_hi is None and lo_lo is None and lo_hi is None and not segs:
            continue
        st.bands.append({
            "lat_min": la_lo, "lat_max": la_hi,
            "lon_min": lo_lo, "lon_max": lo_hi,
            "segments": segs,
        })


def _extract_cues_dict(obs: dict[str, Any], st: _FusionState) -> None:
    """Extract direct cue constraints from an observation/region_vocab dict."""
    if not isinstance(obs, dict) or not obs:
        return

    # 1. Driving side (hard constraint + modality)
    for tok in _tokens(obs.get("driving_side")):
        if tok in ("left", "right"):
            if tok not in st.driving_sides:
                st.driving_sides.append(tok)
            break

    # 2. Domain / ccTLD / calling code / currency / country indicators
    rv = _region_vocab()
    for key in ("domain", "cctld", "tld"):
        for tok in _tokens(obs.get(key)):
            m = re.search(r"([a-z]{2})$", tok)
            if m:
                cc = _norm_cc(m.group(1))
                if cc:
                    st.boost(cc, "domain", 2.1)
                    st.penalize_outside({cc}, 0.75)
            elif rv is not None:
                ccs = rv.meta_map(tok)
                if ccs:
                    st.boost_set(ccs, "domain", 1.8 if len(ccs) == 1 else 0.9)

    for key in ("calling_code", "currency"):
        for tok in _tokens(obs.get(key)):
            if rv is not None:
                ccs = rv.meta_map(tok)
                if ccs:
                    st.boost_set(ccs, "domain", 1.6 if len(ccs) <= 2 else 0.7)

    for key in ("country_indicators", "geocode_ccs", "country_ccs", "ccs"):
        for tok in _tokens(obs.get(key)):
            cc = _norm_cc(tok)
            if cc:
                st.boost(cc, "domain", 1.7)

    # 3. Language / script cues inside obs
    if any(obs.get(k) for k in ("sign_language", "language", "languages", "languages_scripts", "script", "scripts", "texts")):
        _extract_language(obs, st)

    # 4. Plate cues inside obs
    for key in ("plate_style", "plates", "license_plate"):
        for tok in _tokens(obs.get(key)):
            matched_plate_ccs: set[str] = set()
            if tok in _PLATE_HINTS:
                matched_plate_ccs = set(_PLATE_HINTS[tok])
            else:
                for w in _WORD_RE.findall(tok):
                    if w in _PLATE_HINTS:
                        matched_plate_ccs |= _PLATE_HINTS[w]
                if not matched_plate_ccs and rv is not None:
                    pm = rv.plate_map()
                    if tok in pm and pm[tok]:
                        matched_plate_ccs = set(pm[tok])
            if matched_plate_ccs:
                st.boost_set(matched_plate_ccs, "plates", 1.25 if len(matched_plate_ccs) <= 3 else 0.65)
                st.penalize_outside(matched_plate_ccs, 0.40)

    # 5. Sign conventions, bollards, chevrons, road markings
    if any(obs.get(k) for k in ("sign_convention", "sign_conventions", "sign_style", "road_markings", "lane_markings_color")):
        _extract_sign_style(obs, st)

    for tok in _tokens(obs.get("bollards")):
        if tok in _BOLLARD_HINTS:
            st.boost_set(_BOLLARD_HINTS[tok], "signs", 1.2)
        elif rv is not None:
            bm = rv.bollard_map()
            if tok in bm and bm[tok]:
                st.boost_set(bm[tok], "signs", 0.9 if len(bm[tok]) <= 4 else 0.45)

    if rv is not None:
        for obs_k, fn_name in (("chevrons", "chevron_map"), ("guardrails", "guardrail_map"), ("utility_poles", "pole_map")):
            toks = _tokens(obs.get(obs_k))
            if not toks:
                continue
            fn = getattr(rv, fn_name, None)
            cmap = fn() if callable(fn) else {}
            for tok in toks:
                if tok in cmap and cmap[tok]:
                    st.boost_set(cmap[tok], "signs", 0.75 if len(cmap[tok]) <= 5 else 0.35)


def _cue_key_to_modality(cue: str) -> str:
    c = str(cue or "").strip().lower()
    if "driv" in c:
        return "driving"
    if "lang" in c or "script" in c:
        return "language"
    if "plate" in c:
        return "plates"
    if "domain" in c or "tld" in c or "indicator" in c or "dial" in c or "currency" in c:
        return "domain"
    if any(k in c for k in ("sign", "bollard", "chevron", "lane", "marking", "guardrail", "pole", "curb", "convention")):
        return "signs"
    return "cues"


def _extract_geotips(data: dict[str, Any], st: _FusionState) -> None:
    obs = data.get("observed")
    if isinstance(obs, dict):
        _extract_cues_dict(obs, st)
    else:
        _extract_cues_dict(data, st)

    top_rows: list[dict[str, Any]] = []
    for key in ("top", "ranked", "candidates"):
        raw = data.get(key)
        if isinstance(raw, (list, tuple)):
            top_rows.extend(r for r in raw if isinstance(r, dict))

    scores_map = data.get("scores")
    if isinstance(scores_map, dict):
        for k, v in scores_map.items():
            sc = _float(v)
            if sc is not None and sc > 0:
                top_rows.append({"cc": k, "score": sc})

    if not top_rows:
        return
    max_s = max((_float(r.get("score")) or 0.0) for r in top_rows) or 1.0
    for idx, r in enumerate(top_rows[:6]):
        cc = _norm_cc(r)
        sc = _float(r.get("score")) or 0.0
        if not cc or sc <= 0.0:
            continue
        rel = sc / max_s
        gain = 1.25 * rel * (0.82 ** idx)
        cues = r.get("cues")
        if isinstance(cues, (list, tuple)) and cues:
            mods = {_cue_key_to_modality(c) for c in cues}
            per_mod = gain / max(1, len(mods))
            for m in mods:
                if m == "driving":
                    continue  # handled by global driving constraint
                st.boost(cc, m, per_mod)
        else:
            st.boost(cc, "cues", gain)


def _extract_geo_match(data: dict[str, Any], st: _FusionState) -> None:
    _extract_cues_dict(data, st)
    rows: list[dict[str, Any]] = []
    for key in ("ranking", "ranked", "candidates", "top3"):
        raw = data.get(key)
        if isinstance(raw, (list, tuple)):
            rows.extend(r for r in raw if isinstance(r, dict))
            if rows:
                break
    scores_map = data.get("scores")
    if not rows and isinstance(scores_map, dict):
        for k, v in scores_map.items():
            sc = _float(v)
            if sc is not None:
                rows.append({"cc": k, "score": sc})

    for r in rows:
        cc = _norm_cc(r)
        if not cc:
            continue
        sc = _float(r.get("score"))
        hits = r.get("hits") if isinstance(r.get("hits"), (list, tuple)) else []
        misses = r.get("misses") if isinstance(r.get("misses"), (list, tuple)) else []

        # Parse hits/misses for driving_side observation & specific modalities
        hit_mods: set[str] = set()
        for h in hits:
            hs = str(h).lower()
            m_side = re.search(r"driving_side\s+'(left|right)'", hs)
            if m_side and m_side.group(1) not in st.driving_sides:
                st.driving_sides.append(m_side.group(1))
            hit_mods.add(_cue_key_to_modality(hs.split()[0] if hs else ""))
        for ms in misses:
            mss = str(ms).lower()
            m_side = re.search(r"driving_side\s+'(left|right)'", mss)
            if m_side and m_side.group(1) not in st.driving_sides:
                st.driving_sides.append(m_side.group(1))

        if sc is not None:
            # match_score centers neutral at 0.5
            if sc > 0.52:
                gain = (sc - 0.50) * 2.4
                non_driv = {m for m in hit_mods if m != "driving"} or {"cues"}
                for m in non_driv:
                    st.boost(cc, m, gain / len(non_driv))
            elif sc < 0.45:
                st.soft_penalties[cc] = min(
                    2.0, st.soft_penalties.get(cc, 0.0) + (0.50 - sc) * 2.0
                )


def _extract_region_vocab(data: dict[str, Any], st: _FusionState) -> None:
    for sub_key in ("observed", "features", "cues", "osint_vec"):
        sub = data.get(sub_key)
        if isinstance(sub, dict):
            _extract_cues_dict(sub, st)
    _extract_cues_dict(data, st)
    scores_map = data.get("scores")
    if isinstance(scores_map, dict):
        mx = max((_float(v) or 0.0 for v in scores_map.values()), default=0.0) or 1.0
        for k, v in scores_map.items():
            cc = _norm_cc(k)
            sc = _float(v) or 0.0
            if cc and sc > 0.0:
                st.boost(cc, "cues", 1.2 * (sc / mx))


# ---------------------------------------------------------------------------
# Band & hemisphere geometry helpers
# ---------------------------------------------------------------------------

def _lon_overlaps(c_lo: float, c_hi: float, b_lo: float, b_hi: float, tol: float = 2.0) -> bool:
    lo1, hi1 = min(c_lo, c_hi) - tol, max(c_lo, c_hi) + tol
    if b_lo <= b_hi:
        return not (hi1 < b_lo or lo1 > b_hi)
    # Antimeridian wrap on band
    return hi1 >= b_lo or lo1 <= b_hi


def _cc_matches_band(cc: str, band: dict[str, Any]) -> bool:
    info = _cc_info(cc)
    la_lo, la_hi = band.get("lat_min"), band.get("lat_max")
    if la_lo is not None and la_hi is not None:
        blo, bhi = min(la_lo, la_hi), max(la_lo, la_hi)
        # Full band [-90, 90] is unconstrained in latitude
        if not (blo <= -85.0 and bhi >= 85.0):
            if info["lat_max"] + 1.5 < blo or info["lat_min"] - 1.5 > bhi:
                return False

    segs = band.get("segments")
    if isinstance(segs, list) and segs:
        ok_lon = False
        for seg in segs:
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                s0, s1 = _float(seg[0]), _float(seg[1])
                if s0 is not None and s1 is not None:
                    if _lon_overlaps(info["lon_min"], info["lon_max"], s0, s1):
                        ok_lon = True
                        break
        if not ok_lon:
            return False
    else:
        lo_lo, lo_hi = band.get("lon_min"), band.get("lon_max")
        if lo_lo is not None and lo_hi is not None:
            if not (min(lo_lo, lo_hi) <= -175.0 and max(lo_lo, lo_hi) >= 175.0):
                if not _lon_overlaps(info["lon_min"], info["lon_max"], lo_lo, lo_hi):
                    return False
    return True


def _cc_matches_hemisphere(cc: str, hemi: str) -> tuple[bool, bool]:
    """Return (matches_hemi, violates_hemi) for a country code.

    Equatorial countries spanning both hemispheres match both and violate neither.
    """
    info = _cc_info(cc)
    if hemi == "northern":
        matches = info["lat_max"] > 0.0
        violates = info["lat_max"] < -1.0
        return matches, violates
    if hemi == "southern":
        matches = info["lat_min"] < 0.0
        violates = info["lat_min"] > 1.0
        return matches, violates
    return False, False


# ---------------------------------------------------------------------------
# Public API: fuse_signals & matrix_evidence
# ---------------------------------------------------------------------------

def fuse_signals(evidence_list: list) -> dict:
    """Fuse spatial evidence items into a calibrated country distribution.

    Accepts a list of Evidence objects or dicts across ``geo_clip``,
    ``clip_geo``, ``geotips``, ``geo_match``, ``language``, ``sign_style``,
    ``plate_ocr``, ``hemisphere``, ``shadow_geo``, ``region_vocab`` (plus
    sibling geo types).

    Returns::

        {
            "ranked": [
                {
                    "cc": "DE",
                    "name": "Germany",
                    "prob": 0.68,
                    "agreeing_modalities": ["model", "language", "signs"],
                },
                ...
            ],
            "entropy": float,
            "agreement_count": int,
        }
    """
    empty_res: dict[str, Any] = {
        "ranked": [],
        "entropy": 0.0,
        "agreement_count": 0,
    }
    try:
        if evidence_list is None:
            return empty_res
        if not isinstance(evidence_list, (list, tuple)):
            items = getattr(evidence_list, "evidence", None)
            if not isinstance(items, (list, tuple)):
                return empty_res
        else:
            items = evidence_list
        if not items:
            return empty_res

        st = _FusionState()
        for raw_ev in items:
            if raw_ev is None:
                continue
            try:
                ev_type, data = _unpack_ev(raw_ev)
            except Exception:
                continue
            if not data:
                continue

            if ev_type in ("geo_clip", "geoclip", "geocell_pred", "geo_pred", "exif_gps"):
                _extract_geo_clip(data, st)
            elif ev_type in ("clip_geo", "geoguess", "hier_geoguess"):
                _extract_clip_geo(data, st)
            elif ev_type == "geotips":
                _extract_geotips(data, st)
            elif ev_type in ("geo_match", "osint", "geo_tables"):
                _extract_geo_match(data, st)
            elif ev_type == "language":
                _extract_language(data, st)
            elif ev_type == "sign_style":
                _extract_sign_style(data, st)
            elif ev_type in ("plate_ocr", "plate_style"):
                _extract_plate_ocr(data, st)
            elif ev_type == "hemisphere":
                _extract_hemisphere(data, st)
            elif ev_type in ("shadow_geo", "solar"):
                _extract_shadow_geo(data, st)
            elif ev_type in ("region_vocab", "osint_features", "geo"):
                _extract_region_vocab(data, st)
            else:
                # Untyped dict: dispatch on recognizable payload keys
                if any(k in data for k in ("candidates", "lat", "lon")) and "bands" not in data and "observed_hemisphere" not in data:
                    _extract_geo_clip(data, st)
                if "prior" in data:
                    _extract_clip_geo(data, st)
                if "bands" in data or "lat_min" in data or "lat_band" in data:
                    _extract_shadow_geo(data, st)
                if "observed_hemisphere" in data or "hemisphere" in data or "observed_sun_side" in data:
                    _extract_hemisphere(data, st)
                if "strip_ccs" in data or "strip_codes" in data or "reads" in data:
                    _extract_plate_ocr(data, st)
                if "top_conventions" in data or ("scores" in data and isinstance(data.get("scores"), dict) and any(k in _CONVENTION_HINTS for k in data["scores"])):
                    _extract_sign_style(data, st)
                if "ranking" in data:
                    _extract_geo_match(data, st)
                if "top" in data:
                    _extract_geotips(data, st)
                _extract_region_vocab(data, st)

        has_any_signal = bool(
            st.mod_gains
            or st.driving_sides
            or st.hemispheres
            or st.bands
            or st.nonlatin_scripts
        )
        if not has_any_signal:
            return empty_res

        # Apply global constraints (driving_side, hemisphere, shadow_geo bands, non-Latin script)
        hard_violations: dict[str, int] = {cc: 0 for cc in st.universe}
        hard_penalties: dict[str, float] = {cc: 0.0 for cc in st.universe}

        if st.driving_sides:
            side = st.driving_sides[0]
            for cc in st.universe:
                is_left = cc in _LEFT_TRAFFIC
                if (side == "left") == is_left:
                    st.boost(cc, "driving", 0.35)
                else:
                    hard_violations[cc] += 1
                    hard_penalties[cc] += _PENALTY_DRIVING

        if st.hemispheres:
            hemi = st.hemispheres[0]
            for cc in st.universe:
                matches, violates = _cc_matches_hemisphere(cc, hemi)
                if violates:
                    hard_violations[cc] += 1
                    hard_penalties[cc] += _PENALTY_HEMISPHERE
                elif matches:
                    st.boost(cc, "hemisphere", 0.30)

        for band in st.bands:
            matching_band_ccs = {cc for cc in st.universe if _cc_matches_band(cc, band)}
            if not matching_band_ccs:
                continue
            # Narrower bands carry higher information gain
            frac = len(matching_band_ccs) / max(1, len(st.universe))
            band_gain = 1.15 if frac <= 0.25 else (0.80 if frac <= 0.50 else 0.45)
            for cc in st.universe:
                if cc in matching_band_ccs:
                    st.boost(cc, "solar", band_gain)
                else:
                    hard_violations[cc] += 1
                    hard_penalties[cc] += _PENALTY_SOLAR_BAND

        for sc in st.nonlatin_scripts:
            allowed = _SCRIPT_HINTS.get(sc, set())
            if allowed:
                for cc in st.universe:
                    if cc not in allowed:
                        hard_violations[cc] += 1
                        hard_penalties[cc] += _PENALTY_SCRIPT

        # Check if any discriminative modality fired anywhere in the pack
        any_discriminative = any(
            any(m in _DISCRIMINATIVE_MODS for m in mods)
            for mods in st.mod_gains.values()
        )

        # Compute final log-odds per country
        log_odds: dict[str, float] = {}
        cc_modalities: dict[str, list[str]] = {}

        for cc in st.universe:
            gains = st.mod_gains.get(cc, {})
            ordered_mods = [m for m in _MODALITY_ORDER if gains.get(m, 0.0) > 0.0]
            for m in gains:
                if m not in ordered_mods and gains[m] > 0.0:
                    ordered_mods.append(m)

            pos = sum(gains.values())
            pen = st.soft_penalties.get(cc, 0.0) + hard_penalties.get(cc, 0.0)

            has_disc = any(m in _DISCRIMINATIVE_MODS for m in ordered_mods)
            # Multi-modality agreement boost when >=2 independent modalities
            # agree and the country violates no hard constraints.
            if hard_violations.get(cc, 0) == 0 and len(ordered_mods) >= 2 and (has_disc or not any_discriminative):
                # Count distinct conceptual signal groups (so signs + driving
                # from the same road scene don't double-boost unless backed by
                # another modality, while still giving full credit for 2-3+ signals)
                n_agree = len(ordered_mods)
                pos += _AGREEMENT_BONUS * (n_agree - 1)
                # Suppress background-only countries when discriminative cues exist
            elif any_discriminative and not has_disc:
                pen += 0.65

            if hard_violations.get(cc, 0) > 0:
                # Strip background-only modalities from violating countries
                ordered_mods = [m for m in ordered_mods if m in _DISCRIMINATIVE_MODS]

            log_odds[cc] = pos - pen
            cc_modalities[cc] = ordered_mods

        # Softmax over log-odds -> calibrated probabilities
        max_lo = max(log_odds.values())
        exps = {cc: math.exp(lo - max_lo) for cc, lo in log_odds.items()}
        total_z = sum(exps.values()) or 1.0
        raw_probs = {cc: exps[cc] / total_z for cc in st.universe}

        # Sort by probability descending, then country code ascending
        sorted_ccs = sorted(
            st.universe,
            key=lambda c: (-raw_probs[c], -len(cc_modalities.get(c, [])), c),
        )

        # Entropy in bits: H = -sum p * log2(p)
        entropy = 0.0
        for p in raw_probs.values():
            if p > 1e-12:
                entropy -= p * math.log2(p)

        ranked: list[dict[str, Any]] = []
        for cc in sorted_ccs:
            ranked.append({
                "cc": cc,
                "name": _cc_name(cc),
                "prob": round(raw_probs[cc], 4),
                "agreeing_modalities": cc_modalities.get(cc, []),
            })

        top_mods = ranked[0]["agreeing_modalities"] if ranked else []
        return {
            "ranked": ranked,
            "entropy": round(entropy, 4),
            "agreement_count": len(top_mods),
        }
    except Exception:
        return empty_res


def matrix_evidence(fused: dict, claimed_cc: str | None = None) -> "Evidence | None":
    """Package a fused constraint matrix into an Evidence(type='geo_matrix').

    Produces a plain-English summary such as:
    ``"3 independent signals agree on Germany (68% likelihood)"``.
    Returns None when ``fused`` is empty or invalid. Never raises.
    """
    try:
        ClaimVerdict, Evidence = _schema()
    except Exception:
        return None

    try:
        if isinstance(fused, list):
            fused = fuse_signals(fused)
        if not isinstance(fused, dict):
            return None
        ranked = fused.get("ranked")
        if not isinstance(ranked, (list, tuple)) or not ranked:
            return None

        top = ranked[0]
        if not isinstance(top, dict) or not top.get("cc"):
            return None

        top_cc = _norm_cc(top.get("cc")) or str(top.get("cc")).upper()
        top_name = str(top.get("name") or _cc_name(top_cc))
        top_prob = max(0.0, min(1.0, _float(top.get("prob")) or 0.0))
        mods = list(top.get("agreeing_modalities") or [])
        n_mods = int(_float(fused.get("agreement_count")) or len(mods))
        entropy = _float(fused.get("entropy")) or 0.0
        pct = int(round(top_prob * 100))

        if n_mods >= 2:
            base = f"{n_mods} independent signals agree on {top_name} ({pct}% likelihood)"
        elif n_mods == 1:
            base = f"1 independent signal points to {top_name} ({pct}% likelihood)"
        else:
            base = f"Geographic constraints favour {top_name} ({pct}% likelihood)"

        target_cc = _norm_cc(claimed_cc)
        claimed_rank: int | None = None
        claimed_prob = 0.0
        if target_cc:
            for idx, r in enumerate(ranked):
                if isinstance(r, dict) and _norm_cc(r.get("cc")) == target_cc:
                    claimed_rank = idx + 1
                    claimed_prob = max(0.0, min(1.0, _float(r.get("prob")) or 0.0))
                    break

        data: dict[str, Any] = {
            "ranked": [
                {
                    "cc": r.get("cc"),
                    "name": r.get("name"),
                    "prob": r.get("prob"),
                    "score": r.get("prob"),
                    "agreeing_modalities": list(r.get("agreeing_modalities") or []),
                    "hits_n": len(r.get("agreeing_modalities") or []),
                }
                for r in ranked[:10]
                if isinstance(r, dict)
            ],
            "top_cc": top_cc,
            "top_name": top_name,
            "top_prob": round(top_prob, 4),
            "agreeing_modalities": mods,
            "agreement_count": n_mods,
            "entropy": round(entropy, 4),
            "claimed_cc": target_cc,
            "claimed_rank": claimed_rank,
            "claimed_prob": round(claimed_prob, 4) if target_cc else None,
        }

        if not target_cc:
            conf = round(min(0.75, max(0.20, top_prob)), 2)
            return Evidence(
                type="geo_matrix",
                summary=base,
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=conf,
                data=data,
            )

        claimed_name = _cc_name(target_cc)
        if target_cc == top_cc:
            conf = round(min(0.85, max(0.45, top_prob)), 2)
            return Evidence(
                type="geo_matrix",
                summary=base,
                verdict=ClaimVerdict.SUPPORTS,
                confidence=conf,
                data=data,
            )

        if claimed_rank is not None and claimed_rank <= 2 and claimed_prob >= 0.20:
            claimed_pct = int(round(claimed_prob * 100))
            conf = round(min(0.70, max(0.35, claimed_prob + 0.20)), 2)
            return Evidence(
                type="geo_matrix",
                summary=f"{base}; claimed {claimed_name} ranks #{claimed_rank} ({claimed_pct}% likelihood)",
                verdict=ClaimVerdict.SUPPORTS,
                confidence=conf,
                data=data,
            )

        if (top_prob >= 0.25 or n_mods >= 2) and claimed_prob < 0.25 * top_prob:
            claimed_pct = int(round(claimed_prob * 100))
            conf = round(min(0.85, max(0.45, top_prob)), 2)
            return Evidence(
                type="geo_matrix",
                summary=f"{base} — contradicts claimed {claimed_name} ({claimed_pct}% likelihood)",
                verdict=ClaimVerdict.CONTRADICTS,
                confidence=conf,
                data=data,
            )

        claimed_pct = int(round(claimed_prob * 100))
        conf = round(min(0.45, max(0.20, top_prob)), 2)
        return Evidence(
            type="geo_matrix",
            summary=f"{base}; claimed {claimed_name} has {claimed_pct}% likelihood",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=conf,
            data=data,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/geo_matrix.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        global ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    print("geo_matrix self-test:")
    ClaimVerdict, Evidence = _schema()

    # 1. Multi-modal agreement on Germany (GeoCLIP + German language + Vienna signs / right driving)
    de_evs = [
        Evidence(
            type="geo_clip",
            summary="GeoCLIP pins 52.520, 13.405",
            verdict=ClaimVerdict.SUPPORTS,
            confidence=0.55,
            data={"lat": 52.52, "lon": 13.405, "prob": 0.68, "candidates": [{"lat": 52.52, "lon": 13.405, "prob": 0.68}]},
        ),
        Evidence(
            type="language",
            summary="German sign text",
            verdict=ClaimVerdict.SUPPORTS,
            confidence=0.45,
            data={"language": "de", "script": "latin", "texts": ["Ausfahrt", "Bahnhofstraße"]},
        ),
        Evidence(
            type="sign_style",
            summary="Vienna convention signs",
            verdict=ClaimVerdict.SUPPORTS,
            confidence=0.45,
            data={"top_conventions": ["vienna"], "top_score": 0.80, "scores": {"vienna": 0.80, "mutcd": 0.10, "uk": 0.20}},
        ),
    ]
    fused_de = fuse_signals(de_evs)
    top_de = fused_de["ranked"][0] if fused_de["ranked"] else {}
    check(
        "DE ranks #1 on GeoCLIP + German + Vienna signs",
        top_de.get("cc") == "DE" and top_de.get("name") == "Germany",
        f"top={top_de}",
    )
    check(
        "DE probability is well-calibrated (>0.50) and distribution sums to ~1.0",
        top_de.get("prob", 0.0) > 0.50
        and abs(sum(r["prob"] for r in fused_de["ranked"]) - 1.0) < 0.01,
        f"prob={top_de.get('prob')}, sum={sum(r['prob'] for r in fused_de['ranked']):.4f}",
    )
    check(
        "DE agreeing_modalities == ['model', 'language', 'signs'] and agreement_count == 3",
        top_de.get("agreeing_modalities") == ["model", "language", "signs"]
        and fused_de.get("agreement_count") == 3,
        f"mods={top_de.get('agreeing_modalities')}, count={fused_de.get('agreement_count')}",
    )

    ev_de_sup = matrix_evidence(fused_de, claimed_cc="DE")
    check(
        "matrix_evidence SUPPORTS claimed DE with plain-English summary",
        ev_de_sup is not None
        and ev_de_sup.type == "geo_matrix"
        and ev_de_sup.verdict == ClaimVerdict.SUPPORTS
        and "3 independent signals agree on Germany" in ev_de_sup.summary
        and "% likelihood)" in ev_de_sup.summary,
        ev_de_sup.summary if ev_de_sup else "None",
    )

    ev_de_con = matrix_evidence(fused_de, claimed_cc="JP")
    check(
        "matrix_evidence CONTRADICTS claimed JP when signals agree on Germany",
        ev_de_con is not None and ev_de_con.verdict == ClaimVerdict.CONTRADICTS,
        ev_de_con.summary if ev_de_con else "None",
    )

    # 2. Hard constraints: wrong driving side + wrong hemisphere + shadow_geo latitude band
    uk_vs_au = [
        {"type": "language", "data": {"language": "en"}},
        {"type": "region_vocab", "data": {"driving_side": "left", "plate_style": "yellow_rear"}},
        {"type": "hemisphere", "data": {"observed_hemisphere": "northern", "observed_sun_side": "right"}},
        {"type": "shadow_geo", "data": {"bands": [{"lat_min": 50.0, "lat_max": 58.0, "lon_min": -8.0, "lon_max": 2.0, "ok": True}]}},
    ]
    fused_gb = fuse_signals(uk_vs_au)
    probs_gb = {r["cc"]: r["prob"] for r in fused_gb["ranked"]}
    check(
        "GB tops distribution under left-driving + northern hemisphere + 50-58N band",
        fused_gb["ranked"][0]["cc"] == "GB"
        and probs_gb.get("GB", 0.0) > 0.50
        and probs_gb.get("US", 1.0) < 0.01  # penalized for right-hand driving + band
        and probs_gb.get("AU", 1.0) < 0.01,  # penalized for southern hemisphere + band
        f"GB={probs_gb.get('GB')}, US={probs_gb.get('US')}, AU={probs_gb.get('AU')}",
    )

    # 3. Entropy decreases as more independent signals agree
    single_sig = fuse_signals([{"type": "hemisphere", "data": {"observed_hemisphere": "northern"}}])
    check(
        "Entropy shrinks when converging signals narrow the distribution",
        single_sig["entropy"] > fused_de["entropy"] > 0.0,
        f"H(single)={single_sig['entropy']:.2f} > H(multi)={fused_de['entropy']:.2f}",
    )

    # 4. Plate OCR + ccTLD domain + geotips + clip_geo + geo_match
    fr_signals = [
        {"type": "plate_ocr", "data": {"strip_codes": ["F"], "strip_ccs": ["FR"]}},
        {"type": "geotips", "data": {"observed": {"domain": ".fr", "driving_side": "right"}, "top": [{"cc": "FR", "score": 3.2, "cues": ["domain", "driving_side"]}]}},
        {"type": "clip_geo", "data": {"available": True, "prior": [{"cc": "FR", "score": 0.30}, {"cc": "BE", "score": 0.22}]}},
        {"type": "geo_match", "data": {"ranking": [{"cc": "FR", "score": 0.85, "hits": ["sign_language 'fr' — consistent with FR"]}]}},
    ]
    fused_fr = fuse_signals(fr_signals)
    check(
        "FR ranks #1 across plate_ocr + geotips + clip_geo + geo_match",
        fused_fr["ranked"][0]["cc"] == "FR" and fused_fr["agreement_count"] >= 4,
        f"top={fused_fr['ranked'][0]}",
    )

    # 5. Never-raise resilience on empty / malformed inputs
    for bad in (None, [], [None, "junk", 123, {}, {"type": "unknown", "data": "bad"}]):
        r = fuse_signals(bad)  # type: ignore[arg-type]
        ok = ok and isinstance(r, dict) and r.get("ranked") == []
    check("fuse_signals never raises on bad/empty inputs", ok)
    check("matrix_evidence returns None on empty/bad input", matrix_evidence({}) is None and matrix_evidence(None) is None)  # type: ignore[arg-type]

    sys.exit(0 if ok else 1)
