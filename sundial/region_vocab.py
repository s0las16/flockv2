"""Region vocabulary — consolidated observation→country cue tables.

This is the canonical DATA layer for GeoGuessr-style cues: it inverts the
community meta CSV (sundial/data/geotips.csv, ~120 countries, tokenized
snake-case cue cells, read via sundial.geotips.load()) into lookup tables
mapping cue tokens -> sets of ISO-ish country codes, and layers on
hand-curated additions (dial codes, ccTLDs, currencies, legal suffixes,
script buckets, left-traffic countries beyond the CSV roster).

Scoring lives elsewhere (geotips.tip_scores, match_score.score_candidates);
this module only owns the vocabulary so those stages don't each keep
private copies. Nothing here raises: a missing CSV falls back to a small
hand-built table for DE/FR/GB/IT/ES/NL/US/JP/AU, and every public
function degrades to empty structures rather than exceptions.

Map shape everywhere: {cue_token: set(cc)} — e.g.

    driving_side_map()["left"]        -> {"GB", "IE", "JP", "AU", ...}
    plate_map()["eu_blue_strip"]      -> {"DE", "FR", "IT", ...}
    chevron_map()["red_on_white"]     -> {"BE", "BG", "PH", ...}
    script_map()["cyrillic"]          -> {"RU", "UA", "BG", ...}
    meta_map("+49")                   -> {"DE"}

Pure stdlib; geotips is imported lazily so `import sundial.region_vocab`
stays dependency-light and offline-safe.

    plate = plate_map()
    left  = driving_side_map()["left"]
    cols  = tokens_of("JP")           # {"License Plate": [...], ...}
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# CUE_KEYS — the canonical observation keys other modules emit/consume.
# Union of geotips.OBS_MAP keys and match_score/osint_features feature keys.
# ---------------------------------------------------------------------------
CUE_KEYS: list[str] = [
    # road furniture / physical meta
    "driving_side",
    "plate_style",
    "bollards",
    "chevrons",
    "utility_poles",
    "guardrails",
    "curbs",
    "lane_markings_color",
    "road_markings",
    "stop_sign",
    "pedestrian_sign",
    "sign_colors",
    "sign_color",
    "road_number",
    "sign_post",
    # Street View / camera meta
    "camera_artifacts",
    "camera_generation",
    "street_scene",
    "rift",
    # scene-level
    "architecture",
    "vegetation",
    "terrain",
    "climate",
    # text on signs / overlays
    "sign_language",
    "languages_scripts",
    "scripts",
    "texts",
    "sign_convention",
    # explicit geo tokens mined from text
    "domain",
    "calling_code",
    "currency",
    "country_indicators",
    "misc",
]

# Non-ISO spellings -> ISO, matching geo.py / match_score.py conventions.
_CC_ALIASES = {"UK": "GB", "EL": "GR", "USA": "US", "UAE": "AE"}

_NORM_RE = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Hand-curated additions — unioned over whatever the CSV provides.
# ---------------------------------------------------------------------------

# Left-traffic countries absent from the CSV roster (right-traffic is the
# roster default, so only the left set needs supplementing).
_LEFT_EXTRA = {
    "CY",                                     # Cyprus
    "NA", "ZM", "MW", "MZ", "ZW", "TZ",       # southern/east Africa
    "MU", "SC", "KM",                         # Indian Ocean
    "JM", "TT", "BB", "BS", "GY", "SR",       # Caribbean + Guianas
    "AG", "DM", "GD", "KN", "LC", "VC",       # Lesser Antilles
    "BN", "TL",                               # SE Asia (Brunei, Timor-Leste)
    "MV",                                     # Maldives
    "FJ", "PG", "SB", "TO", "KI", "NR",       # Pacific
    "TV", "WS",                               # Pacific (Samoa switched 2009)
    "GG",                                     # Guernsey
}

# EU-style blue-strip plate countries absent from the CSV roster.
_EU_PLATE_EXTRA = {"CY"}

# Named bollard styles -> their namesake country. The CSV often records
# e.g. "german_bollards" only on *other* countries' rows (Russia), while
# Germany's own cell lists raw shapes; these unions make the canonical
# style names resolve to the namesake too.
_BOLLARD_NAMED: dict[str, set[str]] = {
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

# Yellow *centerline* countries whose CSV Lanes cell may not say it —
# North-American convention, explicitly added so the derived
# "yellow_centerline" key is complete.
_LANE_YELLOW_EXTRA = {"US", "CA"}

# Language token (as used in the CSV Language column, plus common extras)
# -> script bucket(s). Used by script_map() to roll languages up into
# writing-system keys ("cyrillic", "arabic", "latin", ...).
_LANG_SCRIPT: dict[str, tuple[str, ...]] = {
    # Latin-script languages
    "english": ("latin",), "french": ("latin",), "portuguese": ("latin",),
    "spanish": ("latin",), "german": ("latin",), "dutch": ("latin",),
    "italian": ("latin",), "indonesian": ("latin",), "malay": ("latin",),
    "swahili": ("latin",), "swedish": ("latin",), "danish": ("latin",),
    "norwegian": ("latin",), "finnish": ("latin",), "estonian": ("latin",),
    "latvian": ("latin",), "lithuanian": ("latin",), "polish": ("latin",),
    "czech": ("latin",), "slovak": ("latin",), "hungarian": ("latin",),
    "romanian": ("latin",), "croatian": ("latin",), "slovenian": ("latin",),
    "albanian": ("latin",), "icelandic": ("latin",), "catalan": ("latin",),
    "turkish": ("latin",), "vietnamese": ("latin",), "filipino": ("latin",),
    "maltese": ("latin",), "irish": ("latin",), "faroese": ("latin",),
    "greenlandic": ("latin",), "hawaiian": ("latin",), "samoan": ("latin",),
    "luxembourgish": ("latin",), "pitkern": ("latin",),
    "romagnol": ("latin",), "basque": ("latin",), "galician": ("latin",),
    "welsh": ("latin",), "afrikaans": ("latin",), "bosnian": ("latin",),
    # Cyrillic (+ often Latin alongside)
    "russian": ("cyrillic",), "ukrainian": ("cyrillic",),
    "belarusian": ("cyrillic",), "bulgarian": ("cyrillic",),
    "macedonian": ("cyrillic",), "kazakh": ("cyrillic",),
    "kyrgyz": ("cyrillic",), "mongolian": ("cyrillic", "mongolian"),
    "serbian": ("cyrillic", "latin"), "montenegrin": ("cyrillic", "latin"),
    "tajik": ("cyrillic",), "uzbek": ("latin", "cyrillic"),
    # Arabic-script
    "arabic": ("arabic",), "urdu": ("arabic",), "persian": ("arabic",),
    "farsi": ("arabic",), "pashto": ("arabic",),
    # Dedicated scripts
    "greek": ("greek",), "hebrew": ("hebrew",),
    "japanese": ("japanese", "latin"), "korean": ("korean",),
    "chinese": ("chinese",), "singaporean_mandarin": ("chinese", "latin"),
    "thai": ("thai",), "khmer": ("khmer",), "lao": ("lao",),
    "burmese": ("burmese",), "hindi": ("devanagari",),
    "nepali": ("devanagari",), "marathi": ("devanagari",),
    "bengali": ("bengali",), "sinhala": ("sinhala",), "tamil": ("tamil",),
    "georgian": ("georgian",), "armenian": ("armenian",),
    "dzongkha": ("tibetan",), "amharic": ("ethiopic",),
    "kannada": ("kannada",), "telugu": ("telugu",),
    "malayalam": ("malayalam",), "gujarati": ("gujarati",),
    "punjabi": ("gurmukhi",),
}

# ISO 639-1 -> CSV language name, so consumers observing "fr" find the
# same cc set as "french". Only languages actually seen as CSV tokens get
# aliases (unknown codes simply won't be keys).
_ISO_LANG: dict[str, str] = {
    "en": "english", "fr": "french", "de": "german", "es": "spanish",
    "pt": "portuguese", "it": "italian", "nl": "dutch", "ja": "japanese",
    "ko": "korean", "zh": "chinese", "ar": "arabic", "ru": "russian",
    "th": "thai", "tr": "turkish", "el": "greek", "he": "hebrew",
    "pl": "polish", "uk": "ukrainian", "sv": "swedish", "da": "danish",
    "no": "norwegian", "fi": "finnish", "cs": "czech", "sk": "slovak",
    "hu": "hungarian", "ro": "romanian", "bg": "bulgarian",
    "hr": "croatian", "sr": "serbian", "sl": "slovenian", "et": "estonian",
    "lv": "latvian", "lt": "lithuanian", "mk": "macedonian",
    "mt": "maltese", "is": "icelandic", "sq": "albanian", "ka": "georgian",
    "hy": "armenian", "vi": "vietnamese", "id": "indonesian",
    "ms": "malay", "hi": "hindi", "bn": "bengali", "ur": "urdu",
    "fa": "persian", "sw": "swahili", "af": "afrikaans", "km": "khmer",
    "lo": "lao", "my": "burmese", "ne": "nepali", "si": "sinhala",
    "mn": "mongolian", "kk": "kazakh", "ky": "kyrgyz", "uz": "uzbek",
    "tl": "filipino", "am": "amharic", "ca": "catalan", "eu": "basque",
    "gl": "galician", "be": "belarusian", "bs": "bosnian",
    "lb": "luxembourgish", "fo": "faroese", "kl": "greenlandic",
    "ta": "tamil", "ga": "irish", "cy": "welsh",
}

# Calling (dial) code -> possible ccs. Shared plans map to sets
# (NANP "1" = US/CA + Caribbean; "7" = RU/KZ).
_DIAL_CC: dict[str, set[str]] = {
    "1": {"US", "CA", "AG", "BB", "BM", "BS", "DM", "DO", "GD", "JM",
          "KN", "LC", "PR", "TT", "VC", "VI", "AS", "GU", "MP"},
    "7": {"RU", "KZ"},
    "20": {"EG"}, "27": {"ZA"}, "30": {"GR"}, "31": {"NL"},
    "32": {"BE"}, "33": {"FR"}, "34": {"ES"}, "36": {"HU"},
    "39": {"IT", "VA"}, "40": {"RO"}, "41": {"CH"}, "43": {"AT"},
    "44": {"GB", "JE", "IM", "GG"}, "45": {"DK"}, "46": {"SE"},
    "47": {"NO", "SJ"}, "48": {"PL"}, "49": {"DE"},
    "51": {"PE"}, "52": {"MX"}, "53": {"CU"}, "54": {"AR"},
    "55": {"BR"}, "56": {"CL"}, "57": {"CO"}, "58": {"VE"},
    "60": {"MY"}, "61": {"AU", "CX", "CC"}, "62": {"ID"},
    "63": {"PH"}, "64": {"NZ", "PN"}, "65": {"SG"}, "66": {"TH"},
    "81": {"JP"}, "82": {"KR"}, "84": {"VN"}, "86": {"CN"},
    "90": {"TR"}, "91": {"IN"}, "92": {"PK"}, "93": {"AF"},
    "94": {"LK"}, "95": {"MM"}, "98": {"IR"},
    "211": {"SS"}, "212": {"MA", "EH"}, "213": {"DZ"}, "216": {"TN"},
    "218": {"LY"}, "220": {"GM"}, "221": {"SN"}, "223": {"ML"},
    "225": {"CI"}, "226": {"BF"}, "228": {"TG"}, "229": {"BJ"},
    "230": {"MU"}, "231": {"LR"}, "232": {"SL"}, "233": {"GH"},
    "234": {"NG"}, "237": {"CM"}, "238": {"CV"}, "240": {"GQ"},
    "241": {"GA"}, "243": {"CD"}, "244": {"AO"}, "248": {"SC"},
    "249": {"SD"}, "250": {"RW"}, "251": {"ET"}, "252": {"SO"},
    "253": {"DJ"}, "254": {"KE"}, "255": {"TZ"}, "256": {"UG"},
    "257": {"BI"}, "258": {"MZ"}, "260": {"ZM"}, "261": {"MG"},
    "262": {"RE", "YT"}, "263": {"ZW"}, "264": {"NA"}, "265": {"MW"},
    "266": {"LS"}, "267": {"BW"}, "268": {"SZ"}, "269": {"KM"},
    "290": {"SH"}, "297": {"AW"}, "298": {"FO"}, "299": {"GL"},
    "350": {"GI"}, "351": {"PT"}, "352": {"LU"}, "353": {"IE"},
    "354": {"IS"}, "355": {"AL"}, "356": {"MT"}, "357": {"CY"},
    "358": {"FI"}, "359": {"BG"}, "370": {"LT"}, "371": {"LV"},
    "372": {"EE"}, "373": {"MD"}, "374": {"AM"}, "375": {"BY"},
    "376": {"AD"}, "377": {"MC"}, "378": {"SM"}, "379": {"VA"},
    "380": {"UA"}, "381": {"RS"}, "382": {"ME"}, "385": {"HR"},
    "386": {"SI"}, "387": {"BA"}, "389": {"MK"}, "420": {"CZ"},
    "421": {"SK"}, "423": {"LI"},
    "500": {"FK"}, "501": {"BZ"}, "502": {"GT"}, "503": {"SV"},
    "504": {"HN"}, "505": {"NI"}, "506": {"CR"}, "507": {"PA"},
    "509": {"HT"}, "591": {"BO"}, "592": {"GY"}, "593": {"EC"},
    "594": {"GY"}, "595": {"PY"}, "597": {"SR"}, "598": {"UY"},
    "599": {"CW", "BQ"},
    "670": {"TL"}, "672": {"NF"}, "673": {"BN"}, "674": {"NR"},
    "675": {"PG"}, "676": {"TO"}, "677": {"SB"}, "678": {"VU"},
    "679": {"FJ"}, "680": {"PW"}, "682": {"CK"}, "685": {"WS"},
    "686": {"KI"}, "687": {"NU"}, "688": {"TV"}, "689": {"PF"},
    "691": {"FM"}, "692": {"MH"},
    "852": {"HK"}, "853": {"MO"}, "855": {"KH"}, "856": {"LA"},
    "880": {"BD"}, "886": {"TW"},
    "960": {"MV"}, "961": {"LB"}, "962": {"JO"}, "963": {"SY"},
    "964": {"IQ"}, "965": {"KW"}, "966": {"SA"}, "967": {"YE"},
    "968": {"OM"}, "971": {"AE"}, "972": {"IL", "PS"}, "973": {"BH"},
    "974": {"QA"}, "975": {"BT"}, "976": {"MN"}, "977": {"NP"},
    "992": {"TJ"}, "993": {"TM"}, "994": {"AZ"}, "995": {"GE"},
    "996": {"KG"}, "998": {"UZ"},
}

# ccTLDs for countries beyond the CSV roster (the CSV Domain column is
# already inverted into this map; these fill the gaps). Multi-label
# second-level domains get their own table below.
_TLD_EXTRA: dict[str, set[str]] = {
    "cy": {"CY"}, "mu": {"MU"}, "sc": {"SC"}, "mv": {"MV"},
    "na": {"NA"}, "zm": {"ZM"}, "mw": {"MW"}, "mz": {"MZ"},
    "zw": {"ZW"}, "tz": {"TZ"}, "ao": {"AO"}, "cm": {"CM"},
    "cd": {"CD"}, "cg": {"CG"}, "ga": {"GA"}, "et": {"ET"},
    "sd": {"SD"}, "so": {"SO"}, "lr": {"LR"}, "sl": {"SL"},
    "ml": {"ML"}, "bf": {"BF"}, "bj": {"BJ"}, "tg": {"TG"},
    "ne": {"NE"}, "td": {"TD"}, "mr": {"MR"}, "gm": {"GM"},
    "gw": {"GW"}, "gn": {"GN"}, "bi": {"BI"}, "dj": {"DJ"},
    "cv": {"CV"}, "gq": {"GQ"}, "cf": {"CF"}, "eh": {"EH"},
    "ma": {"MA"}, "dz": {"DZ"}, "ly": {"LY"}, "eg": {"EG"},
    "cn": {"CN"}, "va": {"VA"}, "xk": {"XK"}, "md": {"MD"},
    "am": {"AM"}, "ge": {"GE"}, "az": {"AZ"}, "by": {"BY"},
    "tj": {"TJ"}, "tm": {"TM"}, "uz": {"UZ"}, "af": {"AF"},
    "ir": {"IR"}, "iq": {"IQ"}, "sy": {"SY"}, "ye": {"YE"},
    "kw": {"KW"}, "bh": {"BH"}, "sa": {"SA"}, "lb": {"LB"},
    "jo": {"JO"}, "ps": {"PS"}, "mm": {"MM"}, "tl": {"TL"},
    "vn": {"VN"}, "ph": {"PH"}, "kp": {"KP"}, "bn": {"BN"},
    "jm": {"JM"}, "tt": {"TT"}, "bb": {"BB"}, "bs": {"BS"},
    "gy": {"GY"}, "sr": {"SR"}, "bz": {"BZ"}, "hn": {"HN"},
    "ni": {"NI"}, "sv": {"SV"}, "ht": {"HT"}, "cu": {"CU"},
    "ve": {"VE"}, "ec": {"EC"}, "bo": {"BO"}, "py": {"PY"},
    "fj": {"FJ"}, "pg": {"PG"}, "sb": {"SB"}, "vu": {"VU"},
    "to": {"TO"}, "ki": {"KI"}, "nr": {"NR"}, "tv": {"TV"},
    "ws": {"WS"}, "fm": {"FM"}, "pw": {"PW"}, "mh": {"MH"},
    "ck": {"CK"}, "nu": {"NU"}, "nf": {"NF"}, "yt": {"YT"},
    "nc": {"NC"}, "pf": {"PF"}, "gf": {"GF"}, "mq": {"MQ"},
    "gp": {"GP"}, "aw": {"AW"}, "ai": {"AI"}, "vg": {"VG"},
    "ky": {"KY"}, "tc": {"TC"}, "ms": {"MS"}, "dm": {"DM"},
    "gd": {"GD"}, "kn": {"KN"}, "ag": {"AG"}, "vc": {"VC"},
    "gg": {"GG"}, "fk": {"FK"}, "sh": {"SH"}, "tf": {"TF"},
    "wf": {"WF"}, "sx": {"SX"}, "bq": {"BQ"}, "lc": {"LC"},
}

# Common second-level ccTLD forms ("co.uk" etc.).
_TLD_SECOND: dict[str, set[str]] = {
    "co.uk": {"GB"}, "org.uk": {"GB"}, "gov.uk": {"GB"}, "ac.uk": {"GB"},
    "co.jp": {"JP"}, "ne.jp": {"JP"}, "or.jp": {"JP"}, "go.jp": {"JP"},
    "co.kr": {"KR"}, "or.kr": {"KR"}, "co.in": {"IN"}, "gov.in": {"IN"},
    "com.au": {"AU"}, "net.au": {"AU"}, "gov.au": {"AU"},
    "co.nz": {"NZ"}, "govt.nz": {"NZ"}, "com.br": {"BR"},
    "com.mx": {"MX"}, "co.za": {"ZA"}, "co.il": {"IL"},
    "com.sg": {"SG"}, "com.my": {"MY"}, "co.th": {"TH"},
    "com.ph": {"PH"}, "com.tw": {"TW"}, "com.hk": {"HK"},
    "com.pk": {"PK"}, "com.bd": {"BD"}, "com.vn": {"VN"},
    "com.cn": {"CN"}, "com.tr": {"TR"}, "com.ar": {"AR"},
    "com.co": {"CO"}, "com.pe": {"PE"}, "com.ua": {"UA"},
    "com.ng": {"NG"}, "com.eg": {"EG"}, "com.sa": {"SA"},
    "com.ae": {"AE"}, "com.do": {"DO"}, "com.gt": {"GT"},
    "com.ec": {"EC"}, "com.bo": {"BO"}, "com.py": {"PY"},
}

# Currency symbols / names / ISO 4217 codes -> ccs. Ambiguous shared
# currencies map to the plausible set (absence is not evidence).
_EURO = {"DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "IE", "FI",
         "GR", "SK", "SI", "EE", "LV", "LT", "LU", "MT", "CY", "HR",
         "AD", "MC", "SM", "VA"}
_CURRENCY_CC: dict[str, set[str]] = {
    "€": set(_EURO), "euro": set(_EURO), "eur": set(_EURO),
    "£": {"GB", "JE", "IM", "GG", "GI", "FK", "SH"},
    "pound": {"GB", "EG", "LB", "SY", "FK", "GI", "SH", "SD"},
    "gbp": {"GB"},
    "¥": {"JP", "CN"}, "yen": {"JP"}, "jpy": {"JP"},
    "yuan": {"CN"}, "rmb": {"CN"}, "cny": {"CN"},
    "₹": {"IN"}, "inr": {"IN"},
    "rupee": {"IN", "PK", "LK", "NP", "MU", "SC"},
    "₩": {"KR", "KP"}, "won": {"KR", "KP"}, "krw": {"KR"},
    "฿": {"TH"}, "baht": {"TH"}, "thb": {"TH"},
    "₺": {"TR"}, "lira": {"TR"}, "try": {"TR"},
    "₽": {"RU"}, "ruble": {"RU", "BY"}, "rouble": {"RU", "BY"},
    "rub": {"RU", "BY"},
    "zł": {"PL"}, "zloty": {"PL"}, "pln": {"PL"},
    "$": {"US", "CA", "AU", "NZ", "HK", "SG", "BB", "JM", "TT",
          "BM", "BZ", "EC", "PA", "PR", "ZW", "TW"},
    "dollar": {"US", "CA", "AU", "NZ", "HK", "SG", "BB", "JM", "TT",
               "BM", "BZ", "EC", "PA", "PR", "ZW", "TW", "FJ", "GY",
               "SR", "NA", "LR", "KI", "SB", "TV"},
    "usd": {"US", "EC", "PA", "PR", "VI", "GU", "AS", "MP", "ZW",
            "SV", "FM", "MH", "PW", "TL"},
    "cad": {"CA"}, "aud": {"AU", "KI", "NR", "TV"},
    "nzd": {"NZ", "CK", "NU", "PN", "TK"},
    "rand": {"ZA", "NA", "LS", "SZ"}, "zar": {"ZA", "NA", "LS", "SZ"},
    "rupiah": {"ID"}, "idr": {"ID"},
    "ringgit": {"MY"}, "myr": {"MY"},
    "omr": {"OM"}, "omani_rial": {"OM"}, "rial": {"OM", "IR"},
    "riyal": {"SA", "QA", "YE"}, "sar": {"SA"},
    "qar": {"QA"}, "irr": {"IR"},
    "franc": {"CH", "LI", "SN", "CI", "BF", "ML", "NE", "TG", "BJ",
              "CM", "CF", "TD", "CG", "GA", "GQ", "RW", "BI", "DJ",
              "KM", "GN", "CD"},
    "chf": {"CH", "LI"}, "xof": {"SN", "CI", "BF", "ML", "NE", "TG",
                                 "BJ", "GW"},
    "xaf": {"CM", "CF", "TD", "CG", "GA", "GQ"},
    "krone": {"NO", "DK", "FO", "GL", "SJ"}, "nok": {"NO", "SJ"},
    "dkk": {"DK", "FO", "GL"}, "krona": {"SE"}, "sek": {"SE"},
    "krona_is": {"IS"}, "kr": {"NO", "DK", "SE", "IS", "FO", "GL", "SJ"},
    "isk": {"IS"},
    "peso": {"MX", "AR", "CL", "CO", "CU", "DO", "PH", "UY"},
    "mxn": {"MX"}, "php": {"PH"}, "dop": {"DO"}, "uyu": {"UY"},
    "cop": {"CO"}, "clp": {"CL"}, "ars": {"AR"}, "cup": {"CU"},
    "real": {"BR"}, "brl": {"BR"}, "r$": {"BR"},
    "dirham": {"MA", "AE"}, "aed": {"AE"}, "mad": {"MA"},
    "shekel": {"IL", "PS"}, "nis": {"IL", "PS"}, "ils": {"IL", "PS"},
    "lei": {"RO"}, "leu": {"RO", "MD"}, "ron": {"RO"}, "mdl": {"MD"},
    "hryvnia": {"UA"}, "uah": {"UA"}, "tenge": {"KZ"}, "kzt": {"KZ"},
    "som": {"KG", "UZ"}, "kgs": {"KG"}, "uzs": {"UZ"},
    "kip": {"LA"}, "lak": {"LA"}, "riel": {"KH"}, "khr": {"KH"},
    "dong": {"VN"}, "vnd": {"VN"}, "taka": {"BD"}, "bdt": {"BD"},
    "kyat": {"MM"}, "mmk": {"MM"}, "tugrik": {"MN"}, "mnt": {"MN"},
    "afghani": {"AF"}, "afn": {"AF"}, "bolivar": {"VE"}, "ves": {"VE"},
    "quetzal": {"GT"}, "gtq": {"GT"}, "lempira": {"HN"}, "hnl": {"HN"},
    "cordoba": {"NI"}, "nio": {"NI"}, "colon": {"CR", "SV"},
    "crc": {"CR"}, "sol": {"PE"}, "pen": {"PE"},
    "guarani": {"PY"}, "pyg": {"PY"}, "boliviano": {"BO"}, "bob": {"BO"},
    "kuna": {"HR"}, "forint": {"HU"}, "huf": {"HU"},
    "koruna": {"CZ"}, "czk": {"CZ"}, "lev": {"BG"}, "bgn": {"BG"},
    "dinar": {"RS", "BA", "DZ", "TN", "LY", "JO", "IQ", "KW", "BH", "MK"},
    "denar": {"MK"}, "mkd": {"MK"}, "rsd": {"RS"}, "bam": {"BA"},
    "naira": {"NG"}, "ngn": {"NG"}, "cedi": {"GH"}, "ghs": {"GH"},
    "birr": {"ET"}, "etb": {"ET"},
    "shilling": {"KE", "UG", "TZ", "SO"},
    "kwacha": {"MW", "ZM"}, "pula": {"BW"}, "bwp": {"BW"},
    "metical": {"MZ"}, "mzn": {"MZ"}, "kina": {"PG"}, "pgk": {"PG"},
    "loti": {"LS"}, "lilangeni": {"SZ"}, "dalasi": {"GM"},
    "ngultrum": {"BT"}, "btn": {"BT"}, "rufiyaa": {"MV"}, "mvr": {"MV"},
    "tala": {"WS"}, "vatu": {"VU"}, "gourde": {"HT"}, "htg": {"HT"},
}

# Company/legal-form suffixes -> cc (consolidated from pipeline._LEGAL_CC).
_LEGAL_CC: dict[str, str] = {
    "gmbh": "DE", "ltd": "GB", "llc": "US", "inc": "US", "sarl": "FR",
    "sas": "FR", "spa": "IT", "srl": "IT", "bv": "NL", "b.v.": "NL",
    "ag": "CH", "pty": "AU", "sdn bhd": "MY", "kk": "JP", "oy": "FI",
    "ab": "SE", "aps": "DK", "lda": "PT", "sl": "ES", "ooo": "RU",
    "kft": "HU", "sro": "CZ", "doo": "HR", "ead": "BG", "ou": "EE",
    "sia": "LV", "uab": "LT", "plc": "GB", "kg": "AT",
}

# ---------------------------------------------------------------------------
# Fallback rows — used only when the real CSV is absent/unreadable.
# Minimal but accurate cue cells for a handful of anchor countries, in the
# same geotips.load() row shape (tokens: {col: [(token, word_set)]}).
# ---------------------------------------------------------------------------
_FALLBACK_DEF = [
    ("DE", "Germany", "Europe", ".de", {
        "Driving Side": "right", "License Plate": "left_blue_strip",
        "Language": "german", "Domain": ".de", "Currency": "euro",
        "Bollard": "white,black_rectangle_with_white_reflector",
        "Chevron": "red_on_white", "Guardrail": "type-A",
        "Lanes": "white_outer,white_center"}),
    ("FR", "France", "Europe", ".fr", {
        "Driving Side": "right", "License Plate": "left_blue_strip,white,yellow",
        "Language": "french", "Domain": ".fr", "Currency": "euro",
        "Bollard": "french_bollards", "Chevron": "white_on_blue",
        "Guardrail": "type-A", "Lanes": "white_outer,white_center"}),
    ("GB", "United Kingdom", "Europe", ".uk", {
        "Driving Side": "left", "License Plate": "white_front,yellow_rear",
        "Language": "english", "Domain": ".uk", "Currency": "pound",
        "Bollard": "uk_bollards,rounded_black_top_with_large_red_and_white_reflectors",
        "Chevron": "white_on_black",
        "Lanes": "double_yellow_outer,white_outer,white_center"}),
    ("IT", "Italy", "Europe", ".it", {
        "Driving Side": "right", "License Plate": "2_blue_stripes",
        "Language": "italian", "Domain": ".it", "Currency": "euro",
        "Bollard": "italian_bollards", "Chevron": "white_on_black"}),
    ("ES", "Spain", "Europe", ".es", {
        "Driving Side": "right", "License Plate": "left_blue_strip",
        "Language": "spanish", "Domain": ".es", "Currency": "euro",
        "Bollard": "spanish_bollards", "Chevron": "blue_and_white",
        "Guardrail": "type-A,yellow_reflectors"}),
    ("NL", "Netherlands", "Europe", ".nl", {
        "Driving Side": "right", "License Plate": "left_blue_strip,yellow",
        "Language": "dutch", "Domain": ".nl", "Currency": "euro",
        "Bollard": "white_bollard_with_red_reflector,black_bollards"}),
    ("US", "United States", "Americas", ".us", {
        "Driving Side": "right", "License Plate": "various_designs",
        "Language": "english", "Domain": ".us", "Currency": "dollar",
        "Chevron": "black_on_yellow",
        "Lanes": "white_outer,yellow_center"}),
    ("JP", "Japan", "Asia", ".jp", {
        "Driving Side": "left", "License Plate": "white_green_text",
        "Language": "japanese", "Domain": ".jp", "Currency": "yen",
        "Lanes": "white_outer,dashed_white_center,yellow_center"}),
    ("AU", "Australia", "Oceania", ".au", {
        "Driving Side": "left", "License Plate": "various_design",
        "Language": "english", "Domain": ".au", "Currency": "dollar",
        "Bollard": "australian_bollards",
        "Chevron": "black_on_white,black_on_yellow"}),
]

_FALLBACK_ROWS: list[dict] | None = None
_MAP_CACHE: dict[str, dict] = {}
_META: dict | None = None


# ---------------------------------------------------------------------------
# Row loading / token plumbing
# ---------------------------------------------------------------------------

def _norm_tok(s) -> str:
    """'Yellow Rear!' -> 'yellow_rear' (same convention as geotips)."""
    return _NORM_RE.sub("_", str(s or "").strip().lower()).strip("_")


def _fallback_rows() -> list[dict]:
    """Hand-built geotips.load()-shaped rows for the anchor countries."""
    global _FALLBACK_ROWS
    if _FALLBACK_ROWS is not None:
        return _FALLBACK_ROWS
    rows: list[dict] = []
    for cc, name, continent, domain, cols in _FALLBACK_DEF:
        tokens = {
            col: [(_norm_tok(t), frozenset())
                  for t in str(cell).split(",") if _norm_tok(t)]
            for col, cell in cols.items()
        }
        tokens.setdefault("Continent", [(_norm_tok(continent), frozenset())])
        rows.append({
            "country": name, "cc": cc, "continent": continent,
            "domain": domain, "tokens": tokens,
        })
    _FALLBACK_ROWS = rows
    return rows


def _rows() -> list[dict]:
    """geotips.load() rows, or the hand-built fallback when the CSV is absent."""
    rows: list[dict] = []
    geotips = None
    try:
        from . import geotips as geotips  # package import
    except Exception:
        try:
            import geotips  # type: ignore  # direct `python3 region_vocab.py`
        except Exception:
            geotips = None
    if geotips is not None:
        try:
            rows = list(geotips.load() or [])
        except Exception:
            rows = []
    if not rows:
        try:
            rows = _fallback_rows()
        except Exception:
            rows = []
    return rows


def _cell_tokens(row: dict, col: str) -> list[str]:
    """Normalized token strings for one CSV column of one row."""
    out: list[str] = []
    for pair in (row.get("tokens") or {}).get(col) or []:
        t = pair[0] if isinstance(pair, (list, tuple)) else str(pair)
        t = _norm_tok(t)
        if t:
            out.append(t)
    return out


def _invert(col: str) -> dict[str, set]:
    """{normalized token: set(cc)} for one CSV column, cached."""
    m = _MAP_CACHE.get(col)
    if m is not None:
        return m
    m = {}
    for row in _rows():
        cc = (row.get("cc") or "").strip().upper()
        if not cc:
            continue
        for tok in _cell_tokens(row, col):
            m.setdefault(tok, set()).add(cc)
    _MAP_CACHE[col] = m
    return m


def _copy(m: dict) -> dict:
    """Public-shape copy: fresh dict + fresh sets so callers can't corrupt."""
    return {k: set(v) for k, v in m.items()}


def _sub(map_: dict, needle: str) -> set:
    """Union of cc sets whose token key contains `needle` (substring)."""
    out: set = set()
    for tok, ccs in map_.items():
        if needle in tok:
            out |= ccs
    return out


# ---------------------------------------------------------------------------
# Public maps
# ---------------------------------------------------------------------------

def driving_side_map() -> dict:
    """{"left": set(cc), "right": set(cc)} — from the Driving Side column.

    The left set additionally carries hand-curated left-traffic countries
    missing from the CSV roster (Cyprus, Zimbabwe, Pacific islands, ...).
    """
    out = {"left": set(), "right": set()}
    try:
        for tok, ccs in _invert("Driving Side").items():
            if tok in ("left", "right"):
                out[tok] |= ccs
            elif "left" in tok:
                out["left"] |= ccs
            elif "right" in tok:
                out["right"] |= ccs
            else:
                out[tok] = set(ccs)
        out["left"] |= _LEFT_EXTRA
    except Exception:
        pass
    return out


def plate_map() -> dict:
    """{plate token/style: set(cc)} — from the License Plate column.

    Raw CSV tokens ("left_blue_strip", "white_front", "yellow_rear") are
    inverted verbatim; derived canonical families are layered on top:
      eu_blue_strip / blue_strip / eu / european — any blue-strip plate
        (EU members plus lookalikes: NO, AL, FO, GI, UA, TR, IL, ...)
      yellow_rear / white_front_yellow_rear — UK-style two-tone plates
      uk / gb — the literal UK-style plate shorthand ({GB})
      no_front_plate — jurisdictions not requiring a front plate
    """
    try:
        m = _invert("License Plate")
        out = _copy(m)
        eu = (_sub(m, "blue_strip") | _sub(m, "eu_plate")
              | _sub(m, "blue_and_yellow_strip") | set(_EU_PLATE_EXTRA))
        out["eu_blue_strip"] = eu
        out["blue_strip"] = set(eu)
        out["eu"] = set(eu)
        out["european"] = set(eu)
        out["yellow_rear"] = _sub(m, "yellow_rear")
        out["white_front_yellow_rear"] = set(out["yellow_rear"])
        out["uk"] = {"GB"}
        out["gb"] = {"GB"}
        nf = _sub(m, "no_front_plate")
        if nf:
            out["no_front_plate"] = nf
        return out
    except Exception:
        return {}


def chevron_map() -> dict:
    """{chevron token: set(cc)} — from the Chevron column."""
    try:
        return _copy(_invert("Chevron"))
    except Exception:
        return {}


def bollard_map() -> dict:
    """{bollard token: set(cc)} — from the Bollard column.

    Named style tokens ("french_bollards", "german_bollards", ...) are
    unioned with their namesake country, so the canonical style names
    resolve even where the CSV only lists shapes on the namesake's row.
    """
    try:
        m = _invert("Bollard")
        out = _copy(m)
        for style, ccs in _BOLLARD_NAMED.items():
            out.setdefault(style, set())
            out[style] |= ccs
        return out
    except Exception:
        return {}


def guardrail_map() -> dict:
    """{guardrail token: set(cc)} — from the Guardrail column."""
    try:
        return _copy(_invert("Guardrail"))
    except Exception:
        return {}


def pole_map() -> dict:
    """{utility-pole token: set(cc)} — from the Utility Pole column."""
    try:
        return _copy(_invert("Utility Pole"))
    except Exception:
        return {}


def lane_map() -> dict:
    """{lane-marking token: set(cc)} — from the Lanes column.

    Adds the derived "yellow_centerline" family (tokens containing
    "yellow_center" plus curated North-American members) — the classic
    US/CA-vs-world centerline tell.
    """
    try:
        m = _invert("Lanes")
        out = _copy(m)
        out["yellow_centerline"] = (
            _sub(m, "yellow_center") | _sub(m, "double_yellow_center")
            | set(_LANE_YELLOW_EXTRA)
        )
        return out
    except Exception:
        return {}


def script_map() -> dict:
    """{language OR writing system: set(cc)} — from the Language column.

    Language tokens invert verbatim ("french" -> {FR, SN, ...}); each is
    also rolled up into script buckets via _LANG_SCRIPT ("cyrillic",
    "arabic", "devanagari", ...), and ISO 639-1 codes alias the language
    names ("fr" -> same set as "french"). A derived "cjk" key unions
    Chinese/Japanese/Korean coverage.
    """
    try:
        m = _invert("Language")
        out = _copy(m)
        # script buckets
        for lang, ccs in m.items():
            for script in _LANG_SCRIPT.get(lang, ()):
                out.setdefault(script, set())
                out[script] |= ccs
        # ISO 639-1 aliases -> the same cc set as the language name
        for iso, lang in _ISO_LANG.items():
            if lang in m:
                out.setdefault(iso, set())
                out[iso] |= m[lang]
        # CJK union bucket
        cjk = set()
        for lang in ("chinese", "japanese", "korean",
                     "singaporean_mandarin"):
            cjk |= m.get(lang, set())
        if cjk:
            out["cjk"] = cjk
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# meta_map — dial codes / ccTLDs / currency / legal suffixes -> ccs
# ---------------------------------------------------------------------------

def _meta_tables() -> dict:
    """Build (and cache) the four meta lookup tables."""
    global _META
    if _META is not None:
        return _META
    dial = {code: set(ccs) for code, ccs in _DIAL_CC.items()}
    tld: dict[str, set] = {}
    currency: dict[str, set] = {k: set(v) for k, v in _CURRENCY_CC.items()}
    for row in _rows():
        cc = (row.get("cc") or "").strip().upper()
        if not cc:
            continue
        for tok in _cell_tokens(row, "Domain"):
            tld.setdefault(tok, set()).add(cc)
        for tok in _cell_tokens(row, "Currency"):
            currency.setdefault(tok, set()).add(cc)
    for src in (_TLD_EXTRA, _TLD_SECOND):
        for k, ccs in src.items():
            tld.setdefault(k, set()).update(ccs)
    legal = {k: {v} for k, v in _LEGAL_CC.items() if v}
    _META = {"dial": dial, "tld": tld, "currency": currency,
             "legal": legal}
    return _META


def _reset_caches() -> None:
    """Drop memoized maps/tables (tests, CSV swap)."""
    global _META, _FALLBACK_ROWS
    _MAP_CACHE.clear()
    _META = None
    _FALLBACK_ROWS = None


def _meta_key(raw) -> str:
    """Normalize a meta token: 'HTTPS://WWW.Example.DE/' -> 'example.de'."""
    k = str(raw or "").strip().lower()
    k = re.sub(r"^[a-z][a-z0-9+.-]*://", "", k)     # scheme
    k = k.split("/")[0].split("?")[0].split("#")[0]  # path/query
    k = k.strip().strip(".")
    if k.startswith("www."):
        k = k[4:]
    return k


def meta_map(key) -> set:
    """Look up one geo-meta token -> set(cc).

    Accepted forms (never raises, empty set on no match):
      dial code   "+49", "49", "0049", "81"
      ccTLD       ".de", "de", "co.uk", a URL/host containing one
      currency    "€", "eur", "euro", "zł", "ringgit", "rupiah"
      legal form  "gmbh", "ltd", "sarl", "kabushiki kaisha"-style suffixes
    """
    try:
        raw = str(key or "").strip().lower()
        if not raw:
            return set()
        t = _meta_tables()
        # calling code: digits (optionally +, 00, separators) only
        if re.fullmatch(r"\+?[\d\s().\-]+", raw):
            digits = re.sub(r"\D", "", raw)
            if digits.startswith("00"):
                digits = digits[2:]
            digits = digits.lstrip("0") or digits
            if digits in t["dial"]:
                return set(t["dial"][digits])
            # full phone number: longest dial-code prefix wins (3->1)
            for n in (3, 2, 1):
                if len(digits) > n and digits[:n] in t["dial"]:
                    return set(t["dial"][digits[:n]])
            return set()
        k = _meta_key(raw)
        if not k:
            return set()
        out: set = set()
        if k in t["tld"]:
            out |= t["tld"][k]
        elif "." in k:
            out |= t["tld"].get(k.rsplit(".", 1)[-1], set())
        out |= t["currency"].get(k, set())
        out |= t["legal"].get(k, set())
        if not out and "." in k:
            # 'sp. z o.o.'-style suffixes: try the trailing label too
            out |= t["legal"].get(k.rsplit(" ", 1)[-1], set())
            out |= t["legal"].get(k.rsplit(".", 1)[-1], set())
        return out
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Per-country view / diagnostics
# ---------------------------------------------------------------------------

def tokens_of(cc) -> dict:
    """{CSV column: [tokens]} for one country — every cue we know for it.

    Merges rows sharing a code (US := Alaska + Hawaii + US + Midway rows)
    and injects a "Country" list of the row names. Empty dict on unknown.
    """
    try:
        code = str(cc or "").strip().upper()
        code = _CC_ALIASES.get(code, code)
        if not code:
            return {}
        out: dict[str, list[str]] = {}
        names: list[str] = []
        for row in _rows():
            if (row.get("cc") or "").strip().upper() != code:
                continue
            name = (row.get("country") or "").strip()
            if name and name not in names:
                names.append(name)
            for col, pairs in (row.get("tokens") or {}).items():
                toks = []
                for pair in pairs or []:
                    t = pair[0] if isinstance(pair, (list, tuple)) else str(pair)
                    t = _norm_tok(t)
                    if t and t not in toks:
                        toks.append(t)
                if not toks:
                    continue
                cur = out.setdefault(col, [])
                for t in toks:
                    if t not in cur:
                        cur.append(t)
        for col in out:
            out[col] = sorted(out[col])
        if names:
            out["Country"] = sorted(names)
        return out
    except Exception:
        return {}


def coverage() -> dict:
    """{column: n_countries} — how many ccs carry each cue column.

    Diagnostic only; adds "_rows" (row count) and "_countries" (distinct
    cc count) meta keys. Empty dict only if even the fallback failed.
    """
    try:
        cov: dict[str, set] = {}
        n_rows = 0
        ccs: set = set()
        for row in _rows():
            n_rows += 1
            cc = (row.get("cc") or "").strip().upper()
            if cc:
                ccs.add(cc)
            for col, pairs in (row.get("tokens") or {}).items():
                has = any(
                    (p[0] if isinstance(p, (list, tuple)) else p)
                    for p in pairs or []
                )
                if has and cc:
                    cov.setdefault(col, set()).add(cc)
        out = {col: len(s) for col, s in sorted(cov.items())}
        out["_rows"] = n_rows
        out["_countries"] = len(ccs)
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/region_vocab.py   (offline, never raises)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ok = True

    def check(name, cond):
        global ok
        ok &= bool(cond)
        print(f"  {name}: {'ok' if cond else 'FAIL'}")

    print(f"CUE_KEYS: {len(CUE_KEYS)} keys")
    check("CUE_KEYS nonempty", len(CUE_KEYS) >= 20)

    d = driving_side_map()
    print(f"driving: left={len(d['left'])} right={len(d['right'])}")
    check("GB left", "GB" in d["left"])
    check("JP left", "JP" in d["left"])
    check("DE right", "DE" in d["right"])
    check("US right", "US" in d["right"])

    p = plate_map()
    print(f"plates: {len(p)} keys; eu_blue_strip={len(p['eu_blue_strip'])}")
    check("DE eu_blue_strip", "DE" in p["eu_blue_strip"])
    check("FR eu_blue_strip", "FR" in p["eu_blue_strip"])
    check("GB not eu strip", "GB" not in p["eu_blue_strip"])
    check("GB yellow_rear", "GB" in p["yellow_rear"])
    check("eu alias", p["eu"] == p["eu_blue_strip"])

    ch = chevron_map()
    print(f"chevrons: {len(ch)} keys")
    check("chevron keys", bool(ch))

    b = bollard_map()
    g = guardrail_map()
    po = pole_map()
    la = lane_map()
    print(f"boll={len(b)} guard={len(g)} pole={len(po)} lanes={len(la)}")
    check("bollards nonempty", bool(b))
    check("FR french_bollards", "FR" in b["french_bollards"])
    check("US yellow_centerline", "US" in la.get("yellow_centerline", set()))

    s = script_map()
    print(f"scripts/langs: {len(s)} keys; cyrillic={len(s.get('cyrillic', []))}")
    check("FR french", "FR" in s.get("french", set()))
    check("RU cyrillic", "RU" in s.get("cyrillic", set()))
    check("JP japanese script", "JP" in s.get("japanese", set()))
    check("iso alias fr", s.get("fr") == s.get("french"))

    check(".de -> DE", "DE" in meta_map(".de"))
    check("de -> DE", "DE" in meta_map("de"))
    check("+81 -> JP", "JP" in meta_map("+81"))
    check("+44 -> GB", "GB" in meta_map("+44"))
    check("euro -> DE", "DE" in meta_map("€") and "DE" in meta_map("eur"))
    check("gmbh -> DE", "DE" in meta_map("gmbh"))
    check("junk -> {}", meta_map("zzz-not-real") == set())

    t = tokens_of("DE")
    print(f"tokens_of('DE'): {len(t)} cols; plate={t.get('License Plate')}")
    check("DE tokens", bool(t))
    check("DE plate token", "left_blue_strip" in t.get("License Plate", []))
    check("UK alias -> GB", bool(tokens_of("UK")))

    c = coverage()
    print(f"coverage: {len(c)} cols, _countries={c.get('_countries')}, "
          f"_rows={c.get('_rows')}")
    check("coverage nonempty", bool(c))
    check("driving col covered", c.get("Driving Side", 0) > 50)

    # never-raise contract: junk inputs everywhere
    check("tokens_of(None)", tokens_of(None) == {})
    check("meta_map(None)", meta_map(None) == set())
    check("meta_map int", meta_map(49) == {"DE"})

    # simulate absent CSV: force fallback rows, re-verify anchors
    try:
        import geotips as _g
    except Exception:
        from . import geotips as _g  # type: ignore
    old_path, old_cache = _g._CSV_PATH, _g._CACHE
    _g._CSV_PATH = "/nonexistent/geotips.csv"
    _g._CACHE = None
    _reset_caches()
    d2 = driving_side_map()
    p2 = plate_map()
    check("fallback GB left", "GB" in d2["left"])
    check("fallback DE eu strip", "DE" in p2["eu_blue_strip"])
    check("fallback coverage", coverage().get("_countries") == 9)
    _g._CSV_PATH = old_path
    _g._CACHE = None
    _reset_caches()

    sys.exit(0 if ok else 1)
