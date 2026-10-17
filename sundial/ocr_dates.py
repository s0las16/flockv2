"""OCR date and timestamp extraction with regional date-format disambiguation.

Scans OCR text strings from video frames (camcorders, CCTV, dashcams, shop
signs, newspapers) for numeric and written dates plus 12h/24h clocks,
normalizes unambiguous dates to ISO YYYY-MM-DD, and maps date-format
conventions (e.g. DD.MM.YYYY vs MM/DD/YYYY vs localized month names) to
ISO country codes.

Conventions: pure stdlib; sibling imports are lazy with package/script
fallback; public functions never raise.
"""
from __future__ import annotations

from datetime import date
import os
import re
import sys

# Regional hints by date-format convention
_DMY_DOT_CCS: tuple[str, ...] = (
    "DE", "AT", "CH", "PL", "CZ", "RU", "NO", "FI",
    "TR", "RS", "HR", "SI", "SK", "RO", "BG", "UA", "EE", "LV",
)
_MDY_SLASH_CCS: tuple[str, ...] = ("US", "PH")
_DMY_SLASH_CCS: tuple[str, ...] = (
    "GB", "IE", "FR", "ES", "IT", "PT", "BR", "AR",
    "MX", "AU", "NZ", "IN", "ZA", "GR", "NL", "BE",
)
_YMD_CCS: tuple[str, ...] = ("CN", "JP", "KR", "TW", "HU", "SE", "LT", "CA")
_CLOCK_12H_CCS: tuple[str, ...] = ("US", "CA", "AU", "NZ", "PH", "IN", "GB")
_CLOCK_24H_CCS: tuple[str, ...] = (
    "DE", "FR", "ES", "IT", "NL", "BE", "AT", "CH",
    "PL", "CZ", "SE", "NO", "FI", "DK", "RU", "JP", "CN", "KR",
)

# Month lookup tables per language family -> (month_number, region_hints)
# Lowercase keys without trailing dots.
_EN_CCS = ("GB", "US", "CA", "AU", "NZ", "IE", "ZA", "IN")
_DE_CCS = ("DE", "AT", "CH")
_FR_CCS = ("FR", "BE", "CH", "CA")
_ES_CCS = ("ES", "MX", "AR", "CO", "CL", "PE")
_IT_CCS = ("IT", "CH")
_PT_CCS = ("PT", "BR")
_NL_CCS = ("NL", "BE")
_PL_CCS = ("PL",)
_RU_CCS = ("RU", "BY", "KZ")
_NO_DK_SE_CCS = ("NO", "DK", "SE")

_MONTH_TABLE: dict[str, tuple[int, tuple[str, ...]]] = {
    # English
    "january": (1, _EN_CCS), "jan": (1, _EN_CCS),
    "february": (2, _EN_CCS), "feb": (2, _EN_CCS),
    "march": (3, _EN_CCS), "mar": (3, _EN_CCS),
    "april": (4, _EN_CCS + _DE_CCS + _NL_CCS), "apr": (4, _EN_CCS + _DE_CCS),
    "may": (5, _EN_CCS),
    "june": (6, _EN_CCS), "jun": (6, _EN_CCS),
    "july": (7, _EN_CCS), "jul": (7, _EN_CCS),
    "august": (8, _EN_CCS + _DE_CCS), "aug": (8, _EN_CCS + _DE_CCS),
    "september": (9, _EN_CCS + _DE_CCS + _NL_CCS), "sep": (9, _EN_CCS), "sept": (9, _EN_CCS),
    "october": (10, _EN_CCS), "oct": (10, _EN_CCS),
    "november": (11, _EN_CCS + _DE_CCS + _NL_CCS), "nov": (11, _EN_CCS + _DE_CCS),
    "december": (12, _EN_CCS + _NL_CCS), "dec": (12, _EN_CCS),
    # German
    "januar": (1, _DE_CCS), "jänner": (1, ("AT", "DE", "CH")), "jaenner": (1, ("AT", "DE", "CH")),
    "februar": (2, _DE_CCS),
    "märz": (3, _DE_CCS), "maerz": (3, _DE_CCS), "marz": (3, _DE_CCS),
    "mai": (5, _FR_CCS + _DE_CCS + _NO_DK_SE_CCS),
    "juni": (6, _DE_CCS + _NL_CCS + _NO_DK_SE_CCS),
    "juli": (7, _DE_CCS + _NL_CCS + _NO_DK_SE_CCS),
    "oktober": (10, _DE_CCS + _NL_CCS + _NO_DK_SE_CCS),
    "dezember": (12, _DE_CCS), "dez": (12, _DE_CCS),
    # French
    "janvier": (1, _FR_CCS), "janv": (1, _FR_CCS),
    "février": (2, _FR_CCS), "fevrier": (2, _FR_CCS), "févr": (2, _FR_CCS), "fevr": (2, _FR_CCS),
    "mars": (3, _FR_CCS + _NO_DK_SE_CCS),
    "avril": (4, _FR_CCS), "avr": (4, _FR_CCS),
    "juin": (6, _FR_CCS),
    "juillet": (7, _FR_CCS), "juil": (7, _FR_CCS),
    "août": (8, _FR_CCS), "aout": (8, _FR_CCS),
    "septembre": (9, _FR_CCS),
    "octobre": (10, _FR_CCS),
    "novembre": (11, _FR_CCS + _IT_CCS),
    "décembre": (12, _FR_CCS), "decembre": (12, _FR_CCS), "déc": (12, _FR_CCS),
    # Spanish
    "enero": (1, _ES_CCS),
    "febrero": (2, _ES_CCS),
    "marzo": (3, _ES_CCS + _IT_CCS),
    "abril": (4, _ES_CCS + _PT_CCS),
    "mayo": (5, _ES_CCS),
    "junio": (6, _ES_CCS),
    "julio": (7, _ES_CCS),
    "agosto": (8, _ES_CCS + _IT_CCS + _PT_CCS),
    "septiembre": (9, _ES_CCS), "setiembre": (9, _ES_CCS),
    "octubre": (10, _ES_CCS),
    "noviembre": (11, _ES_CCS),
    "diciembre": (12, _ES_CCS),
    # Italian
    "gennaio": (1, _IT_CCS),
    "febbraio": (2, _IT_CCS),
    "aprile": (4, _IT_CCS),
    "maggio": (5, _IT_CCS),
    "giugno": (6, _IT_CCS),
    "luglio": (7, _IT_CCS),
    "settembre": (9, _IT_CCS),
    "ottobre": (10, _IT_CCS),
    "dicembre": (12, _IT_CCS),
    # Portuguese
    "janeiro": (1, _PT_CCS),
    "fevereiro": (2, _PT_CCS),
    "março": (3, _PT_CCS), "marco": (3, _PT_CCS),
    "maio": (5, _PT_CCS),
    "junho": (6, _PT_CCS),
    "julho": (7, _PT_CCS),
    "setembro": (9, _PT_CCS),
    "outubro": (10, _PT_CCS),
    "novembro": (11, _PT_CCS),
    "dezembro": (12, _PT_CCS),
    # Dutch
    "januari": (1, _NL_CCS),
    "februari": (2, _NL_CCS),
    "maart": (3, _NL_CCS),
    "mei": (5, _NL_CCS),
    "augustus": (8, _NL_CCS),
    # Polish
    "stycznia": (1, _PL_CCS), "styczeń": (1, _PL_CCS), "styczen": (1, _PL_CCS),
    "lutego": (2, _PL_CCS), "luty": (2, _PL_CCS),
    "marca": (3, _PL_CCS), "marzec": (3, _PL_CCS),
    "kwietnia": (4, _PL_CCS), "kwiecień": (4, _PL_CCS), "kwiecien": (4, _PL_CCS),
    "maja": (5, _PL_CCS), "maj": (5, _PL_CCS + _NO_DK_SE_CCS),
    "czerwca": (6, _PL_CCS), "czerwiec": (6, _PL_CCS),
    "lipca": (7, _PL_CCS), "lipiec": (7, _PL_CCS),
    "sierpnia": (8, _PL_CCS), "sierpień": (8, _PL_CCS), "sierpien": (8, _PL_CCS),
    "września": (9, _PL_CCS), "wrzesień": (9, _PL_CCS), "wrzesnia": (9, _PL_CCS),
    "października": (10, _PL_CCS), "październik": (10, _PL_CCS), "pazdziernika": (10, _PL_CCS),
    "listopada": (11, _PL_CCS), "listopad": (11, _PL_CCS),
    "grudnia": (12, _PL_CCS), "grudzień": (12, _PL_CCS), "grudzien": (12, _PL_CCS),
    # Russian
    "января": (1, _RU_CCS), "январь": (1, _RU_CCS),
    "февраля": (2, _RU_CCS), "февраль": (2, _RU_CCS),
    "марта": (3, _RU_CCS), "март": (3, _RU_CCS),
    "апреля": (4, _RU_CCS), "апрель": (4, _RU_CCS),
    "мая": (5, _RU_CCS), "май": (5, _RU_CCS),
    "июня": (6, _RU_CCS), "июнь": (6, _RU_CCS),
    "июля": (7, _RU_CCS), "июль": (7, _RU_CCS),
    "августа": (8, _RU_CCS), "август": (8, _RU_CCS),
    "сентября": (9, _RU_CCS), "сентябрь": (9, _RU_CCS),
    "октября": (10, _RU_CCS), "октябрь": (10, _RU_CCS),
    "ноября": (11, _RU_CCS), "ноябрь": (11, _RU_CCS),
    "декабря": (12, _RU_CCS), "декабрь": (12, _RU_CCS),
}

_FORMAT_DESCRIPTION = {
    "DMY_DOT": "European day.month.year dot format",
    "DMY_SLASH": "day/month/year slash format",
    "MDY_SLASH": "US-style month/day/year slash format",
    "YMD": "year-month-day format",
    "NAMED": "written month format",
    "CLOCK_12H": "12-hour AM/PM clock format",
    "CLOCK_24H": "24-hour clock format",
}

# Regex patterns
# 1. CJK YMD: 2024年5月17日
_RE_CJK_YMD = re.compile(
    r"(?<!\d)(19\d{2}|20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
)

# 2. ISO / YMD numeric: 2024-05-17, 2024/05/17, 2024.05.17
_RE_YMD = re.compile(
    r"(?<!\d)(19\d{2}|20\d{2})([-/.])(\d{1,2})\2(\d{1,2})(?!\d)"
)

# 3. Dot-separated DMY: 17.05.2024 or 17.05.24
_RE_DMY_DOT = re.compile(
    r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(19\d{2}|20\d{2}|\d{2})(?!\d)"
)

# 4. Slash-separated DMY or MDY: 17/05/2024, 05/17/2024, 17/05/24
_RE_SLASH = re.compile(
    r"(?<!\d)(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2}|\d{2})(?!\d)"
)

# 5. Dash-separated DMY or MDY with 4-digit year at the end: 17-05-2024
_RE_DASH_DMY = re.compile(
    r"(?<!\d)(\d{1,2})-(\d{1,2})-(19\d{2}|20\d{2})(?!\d)"
)

# 6. Written day-first: "17 May 2024", "17. Mai 2024", "17 mai 2024", "17 de mayo de 2024"
_RE_NAMED_DMY = re.compile(
    r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?(\.?)\s+(?:de\s+|d['’])?"
    r"([A-Za-zÀ-ÖØ-öø-ÿА-Яа-яЁё]+)\.?\s+(?:de\s+|del\s+)?"
    r"(19\d{2}|20\d{2})(?!\d)"
)

# 7. Written month-first: "May 17, 2024" or "May 17th 2024"
_RE_NAMED_MDY = re.compile(
    r"\b([A-Za-zÀ-ÖØ-öø-ÿ]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(19\d{2}|20\d{2})(?!\d)"
)

# 8. 12h / 24h clocks: "14:23:05", "02:23 PM", "2:23:05 a.m."
_RE_CLOCK = re.compile(
    r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*([AaPp]\.?\s?[Mm]\.?))?(?!\d)"
)


def _schema():
    """Lazy schema import — works both as sundial.ocr_dates and standalone."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


def _expand_year(y_str: str) -> int | None:
    try:
        y = int(y_str)
    except (TypeError, ValueError):
        return None
    if 0 <= y < 100:
        return 2000 + y if y <= 69 else 1900 + y
    if 1900 <= y <= 2099:
        return y
    return None


def _valid_iso(year: int | None, month: int, day: int) -> str | None:
    if year is None:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _has_12h_clock(text: str) -> bool:
    for m in _RE_CLOCK.finditer(text):
        if m.group(4):
            return True
    return False


def _overlaps(span: tuple[int, int], covered: list[tuple[int, int]]) -> bool:
    s, e = span
    return any(not (e <= cs or s >= ce) for cs, ce in covered)


def _named_hints(month_token: str, has_dot: bool, day_first: bool) -> tuple[int, list[str]] | None:
    clean = month_token.strip().rstrip(".")
    lower = clean.lower()
    entry = _MONTH_TABLE.get(lower)
    if entry is None:
        return None
    month_num, base_ccs = entry

    # Disambiguate "Mai" / "mai" and ordinal dot ("17. Mai 2024" vs "17 mai 2024")
    if lower == "mai":
        if has_dot or (clean[:1].isupper() and day_first):
            return month_num, list(_DE_CCS) if has_dot else ["DE", "AT", "CH", "FR", "BE", "CA"]
        return month_num, list(_FR_CCS)

    if has_dot and day_first:
        # Ordinal period after the day number ("17. Mai 2024", "17. Oktober 2024")
        # is a classic German/Austrian/Swiss/Nordic/Central European convention.
        de_first = [c for c in _DE_CCS if c in base_ccs]
        rest = [c for c in base_ccs if c not in de_first]
        if de_first:
            return month_num, de_first + rest
        return month_num, list(_DE_CCS) + rest

    if not day_first:
        # "May 17, 2024" -> North American / Philippine English preference
        return month_num, ["US", "CA", "PH"] + [c for c in base_ccs if c not in ("US", "CA", "PH")]

    if base_ccs == _EN_CCS:
        # "17 May 2024" -> Commonwealth / British day-first English preference
        return month_num, ["GB", "IE", "AU", "NZ", "ZA", "IN", "CA", "US"]

    return month_num, list(base_ccs)


def _scan_one_text(text: str) -> list[dict]:
    items: list[tuple[int, dict]] = []
    covered: list[tuple[int, int]] = []
    has_ampm = _has_12h_clock(text)

    # 1. CJK YMD dates
    for m in _RE_CJK_YMD.finditer(text):
        if _overlaps(m.span(), covered):
            continue
        yr = _expand_year(m.group(1))
        iso = _valid_iso(yr, int(m.group(2)), int(m.group(3)))
        if iso is None:
            continue
        covered.append(m.span())
        items.append((m.start(), {
            "raw": m.group(0).strip(),
            "iso_date": iso,
            "format_style": "YMD",
            "region_hint": ["JP", "CN", "TW", "HK"],
        }))

    # 2. Numeric YMD dates (2024-05-17, 2024/05/17, 2024.05.17)
    for m in _RE_YMD.finditer(text):
        if _overlaps(m.span(), covered):
            continue
        yr = _expand_year(m.group(1))
        iso = _valid_iso(yr, int(m.group(3)), int(m.group(4)))
        if iso is None:
            continue
        covered.append(m.span())
        items.append((m.start(), {
            "raw": m.group(0).strip(),
            "iso_date": iso,
            "format_style": "YMD",
            "region_hint": list(_YMD_CCS),
        }))

    # 3. Written day-first dates ("17 May 2024", "17. Mai 2024", "17 mai 2024")
    for m in _RE_NAMED_DMY.finditer(text):
        if _overlaps(m.span(), covered):
            continue
        day_num = int(m.group(1))
        has_dot = bool(m.group(2))
        month_tok = m.group(3)
        yr = _expand_year(m.group(4))
        parsed = _named_hints(month_tok, has_dot=has_dot, day_first=True)
        if parsed is None:
            continue
        month_num, hints = parsed
        iso = _valid_iso(yr, month_num, day_num)
        if iso is None:
            continue
        covered.append(m.span())
        items.append((m.start(), {
            "raw": m.group(0).strip(),
            "iso_date": iso,
            "format_style": "NAMED",
            "region_hint": hints,
        }))

    # 4. Written month-first dates ("May 17, 2024")
    for m in _RE_NAMED_MDY.finditer(text):
        if _overlaps(m.span(), covered):
            continue
        month_tok = m.group(1)
        day_num = int(m.group(2))
        yr = _expand_year(m.group(3))
        parsed = _named_hints(month_tok, has_dot=False, day_first=False)
        if parsed is None:
            continue
        month_num, hints = parsed
        iso = _valid_iso(yr, month_num, day_num)
        if iso is None:
            continue
        covered.append(m.span())
        items.append((m.start(), {
            "raw": m.group(0).strip(),
            "iso_date": iso,
            "format_style": "NAMED",
            "region_hint": hints,
        }))

    # 5. Dot dates: DD.MM.YYYY / DD.MM.YY
    for m in _RE_DMY_DOT.finditer(text):
        if _overlaps(m.span(), covered):
            continue
        a, b = int(m.group(1)), int(m.group(2))
        yr = _expand_year(m.group(3))
        iso = _valid_iso(yr, b, a)
        if iso is None:
            continue
        covered.append(m.span())
        items.append((m.start(), {
            "raw": m.group(0).strip(),
            "iso_date": iso,
            "format_style": "DMY_DOT",
            "region_hint": list(_DMY_DOT_CCS),
        }))

    # 6. Slash dates: DD/MM/YYYY vs MM/DD/YYYY
    for m in _RE_SLASH.finditer(text):
        if _overlaps(m.span(), covered):
            continue
        a, b = int(m.group(1)), int(m.group(2))
        yr = _expand_year(m.group(3))
        if yr is None:
            continue
        if a > 12 and 1 <= b <= 12:
            iso = _valid_iso(yr, b, a)
            if iso is None:
                continue
            style = "DMY_SLASH"
            hints = list(_DMY_SLASH_CCS)
        elif b > 12 and 1 <= a <= 12:
            iso = _valid_iso(yr, a, b)
            if iso is None:
                continue
            style = "MDY_SLASH"
            hints = list(_MDY_SLASH_CCS)
        elif 1 <= a <= 12 and 1 <= b <= 12:
            if a == b:
                iso = _valid_iso(yr, a, b)
                if iso is None:
                    continue
                style = "MDY_SLASH" if has_ampm else "DMY_SLASH"
                hints = list(_MDY_SLASH_CCS if has_ampm else _DMY_SLASH_CCS)
            elif has_ampm:
                # 12-hour AM/PM overlay alongside an ambiguous slash date points to US/MDY
                iso = _valid_iso(yr, a, b)
                style = "MDY_SLASH"
                hints = list(_MDY_SLASH_CCS)
            else:
                # Ambiguous between DMY and MDY when both tokens <= 12
                iso_dmy = _valid_iso(yr, b, a)
                if iso_dmy is None:
                    continue
                iso = iso_dmy
                style = "DMY_SLASH"
                hints = list(_DMY_SLASH_CCS) + [c for c in _MDY_SLASH_CCS if c not in _DMY_SLASH_CCS]
        else:
            continue
        covered.append(m.span())
        items.append((m.start(), {
            "raw": m.group(0).strip(),
            "iso_date": iso,
            "format_style": style,
            "region_hint": hints,
        }))

    # 7. Dash dates with 4-digit year at end: DD-MM-YYYY or MM-DD-YYYY
    for m in _RE_DASH_DMY.finditer(text):
        if _overlaps(m.span(), covered):
            continue
        a, b = int(m.group(1)), int(m.group(2))
        yr = _expand_year(m.group(3))
        if a > 12 and 1 <= b <= 12:
            iso = _valid_iso(yr, b, a)
            style = "DMY_SLASH"
            hints = list(_DMY_SLASH_CCS)
        elif b > 12 and 1 <= a <= 12:
            iso = _valid_iso(yr, a, b)
            style = "MDY_SLASH"
            hints = list(_MDY_SLASH_CCS)
        else:
            iso = _valid_iso(yr, b, a)
            style = "DMY_SLASH"
            hints = list(_DMY_SLASH_CCS)
        if iso is None:
            continue
        covered.append(m.span())
        items.append((m.start(), {
            "raw": m.group(0).strip(),
            "iso_date": iso,
            "format_style": style,
            "region_hint": hints,
        }))

    # 8. 12h / 24h clocks: if no date was extracted in this string, emit a
    # standalone clock entry; if a date was found on this string, fold 12h
    # AM/PM hints into the date's region hints if relevant.
    for m in _RE_CLOCK.finditer(text):
        if _overlaps(m.span(), covered):
            continue
        hh, mm = int(m.group(1)), int(m.group(2))
        ss = int(m.group(3)) if m.group(3) is not None else 0
        ampm = (m.group(4) or "").strip()
        if mm > 59 or ss > 59:
            continue
        if ampm:
            if not (1 <= hh <= 12):
                continue
        else:
            if not (0 <= hh <= 23):
                continue
        if not items:
            covered.append(m.span())
            items.append((m.start(), {
                "raw": m.group(0).strip(),
                "iso_date": None,
                "format_style": "CLOCK_12H" if ampm else "CLOCK_24H",
                "region_hint": list(_CLOCK_12H_CCS if ampm else _CLOCK_24H_CCS),
            }))

    items.sort(key=lambda pair: pair[0])
    return [d for _, d in items]


def extract_dates(texts: list[str]) -> list[dict]:
    """Extract dates, timestamps, and regional format hints from OCR strings.

    Returns a list of dicts:
      {
        "raw": str,
        "iso_date": "YYYY-MM-DD" | None,
        "format_style": "DMY_DOT" | "DMY_SLASH" | "MDY_SLASH" | "YMD" | "NAMED",
        "region_hint": list[str],
      }
    Never raises on any input.
    """
    try:
        if texts is None:
            return []
        if isinstance(texts, (str, bytes)):
            iter_texts = [texts.decode("utf-8", "ignore") if isinstance(texts, bytes) else texts]
        else:
            iter_texts = list(texts)
    except Exception:
        return []

    out: list[dict] = []
    seen: set[tuple[str, str | None, str]] = set()
    for raw_entry in iter_texts:
        try:
            if raw_entry is None:
                continue
            if isinstance(raw_entry, dict):
                s = str(raw_entry.get("text") or raw_entry.get("raw") or "")
            else:
                s = str(raw_entry)
            if not s.strip():
                continue
            for item in _scan_one_text(s):
                key = (item["raw"], item["iso_date"], item["format_style"])
                if key not in seen:
                    seen.add(key)
                    out.append(item)
        except Exception:
            continue
    return out


def _parse_claimed_iso_date(claimed_time_iso: str | None) -> str | None:
    if not claimed_time_iso or not isinstance(claimed_time_iso, str):
        return None
    s = claimed_time_iso.strip()
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if not m:
        return None
    return _valid_iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _human_style(style: str) -> str:
    return _FORMAT_DESCRIPTION.get(style, "on-screen date format")


def date_evidence(
    extracted: list[dict],
    claimed_time_iso: str | None = None,
    claimed_cc: str | None = None,
):
    """Compare extracted OCR dates against claimed_time_iso and claimed_cc.

    - Same day as claimed_time_iso -> SUPPORTS @ 0.75
    - Different day from claimed_time_iso -> CONTRADICTS @ 0.80
    - Also evaluates whether the date format convention matches claimed_cc.
    - Returns None when no usable extracted items are present.
    Never raises on any input.
    """
    try:
        ClaimVerdict, Evidence = _schema()
        if not extracted or not isinstance(extracted, (list, tuple)):
            return None

        valid_items: list[dict] = []
        for item in extracted:
            if not isinstance(item, dict):
                continue
            raw = str(item.get("raw") or "").strip()
            iso = item.get("iso_date")
            iso_str = str(iso).strip() if iso else None
            style = str(item.get("format_style") or "YMD").strip()
            hints_raw = item.get("region_hint") or []
            hints = [
                str(c).strip().upper()
                for c in (hints_raw if isinstance(hints_raw, (list, tuple, set)) else [])
                if str(c).strip()
            ]
            if not raw and not iso_str:
                continue
            valid_items.append({
                "raw": raw or (iso_str or ""),
                "iso_date": iso_str,
                "format_style": style,
                "region_hint": hints,
            })

        if not valid_items:
            return None

        claimed_date = _parse_claimed_iso_date(claimed_time_iso)
        cc = str(claimed_cc).strip().upper() if claimed_cc and isinstance(claimed_cc, str) else ""

        date_items = [it for it in valid_items if it.get("iso_date")]
        primary = date_items[0] if date_items else valid_items[0]
        style_desc = _human_style(primary["format_style"])
        hints = primary["region_hint"]

        cc_note = ""
        cc_matches: bool | None = None
        if cc and hints:
            if cc in hints:
                cc_matches = True
                cc_note = f", and its {style_desc} is typical for {cc}"
            else:
                cc_matches = False
                sample_regions = ", ".join(hints[:4])
                cc_note = (
                    f", though its {style_desc} is more typical of "
                    f"{sample_regions} than {cc}"
                )

        if claimed_date and date_items:
            seen_dates = [it["iso_date"] for it in date_items if it.get("iso_date")]
            if claimed_date in seen_dates:
                matched = next(it for it in date_items if it["iso_date"] == claimed_date)
                m_style = _human_style(matched["format_style"])
                if cc and matched["region_hint"]:
                    if cc in matched["region_hint"]:
                        cc_clause = f" written in {m_style} consistent with {cc}"
                    else:
                        sample_regions = ", ".join(matched["region_hint"][:4])
                        cc_clause = (
                            f" written in {m_style} (more common in {sample_regions} than {cc})"
                        )
                else:
                    cc_clause = f" written in {m_style}"
                summary = (
                    f"On-screen text shows date '{matched['raw']}' ({claimed_date})"
                    f"{cc_clause}, matching the claimed date."
                )
                return Evidence(
                    type="ocr_dates",
                    summary=summary,
                    verdict=ClaimVerdict.SUPPORTS,
                    confidence=0.75,
                    data={
                        "extracted": valid_items,
                        "claimed_date": claimed_date,
                        "claimed_cc": cc or None,
                        "date_match": True,
                        "cc_match": cc_matches,
                    },
                )
            else:
                mismatch = date_items[0]
                summary = (
                    f"On-screen text shows date '{mismatch['raw']}' "
                    f"({mismatch['iso_date']}), which does not match the "
                    f"claimed date of {claimed_date}{cc_note}."
                )
                return Evidence(
                    type="ocr_dates",
                    summary=summary,
                    verdict=ClaimVerdict.CONTRADICTS,
                    confidence=0.80,
                    data={
                        "extracted": valid_items,
                        "claimed_date": claimed_date,
                        "claimed_cc": cc or None,
                        "date_match": False,
                        "cc_match": cc_matches,
                    },
                )

        # No claimed_time_iso comparison possible — evaluate regional format against claimed_cc
        if cc and hints:
            raw_label = primary["raw"]
            iso_part = f" ({primary['iso_date']})" if primary.get("iso_date") else ""
            if cc in hints:
                summary = (
                    f"On-screen text '{raw_label}'{iso_part} uses {style_desc}, "
                    f"which is consistent with {cc}."
                )
                return Evidence(
                    type="ocr_dates",
                    summary=summary,
                    verdict=ClaimVerdict.SUPPORTS,
                    confidence=0.55,
                    data={
                        "extracted": valid_items,
                        "claimed_date": None,
                        "claimed_cc": cc,
                        "date_match": None,
                        "cc_match": True,
                    },
                )
            else:
                sample_regions = ", ".join(hints[:4])
                summary = (
                    f"On-screen text '{raw_label}'{iso_part} uses {style_desc}, "
                    f"which is characteristic of {sample_regions} rather than {cc}."
                )
                return Evidence(
                    type="ocr_dates",
                    summary=summary,
                    verdict=ClaimVerdict.CONTRADICTS,
                    confidence=0.60,
                    data={
                        "extracted": valid_items,
                        "claimed_date": None,
                        "claimed_cc": cc,
                        "date_match": None,
                        "cc_match": False,
                    },
                )

        # Neither claimed_time_iso nor claimed_cc resolved a verdict
        raw_label = primary["raw"]
        iso_part = f" ({primary['iso_date']})" if primary.get("iso_date") else ""
        sample_regions = ", ".join(hints[:4]) if hints else "multiple regions"
        summary = (
            f"On-screen text contains '{raw_label}'{iso_part} in {style_desc} "
            f"(commonly used in {sample_regions})."
        )
        return Evidence(
            type="ocr_dates",
            summary=summary,
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.35,
            data={
                "extracted": valid_items,
                "claimed_date": claimed_date,
                "claimed_cc": cc or None,
                "date_match": None,
                "cc_match": None,
            },
        )
    except Exception:
        return None


if __name__ == "__main__":
    ClaimVerdict, _ = _schema()

    sample_texts = [
        "CAM01 2024-05-17 14:22:10",
        "Parkschein 17.05.2024 09:15",
        "RECEIPT 05/17/2024 02:30 PM",
        "17/05/2024",
        "Edition du 17 mai 2024",
        "Ausgabe 17. Mai 2024",
        "Published 17 May 2024",
    ]
    found = extract_dates(sample_texts)
    assert len(found) == 7, f"expected 7 dates, got {len(found)}: {found}"

    by_raw = {d["raw"]: d for d in found}
    assert by_raw["2024-05-17"]["iso_date"] == "2024-05-17"
    assert by_raw["2024-05-17"]["format_style"] == "YMD"

    assert by_raw["17.05.2024"]["iso_date"] == "2024-05-17"
    assert by_raw["17.05.2024"]["format_style"] == "DMY_DOT"
    for cc in ("DE", "AT", "CH", "PL", "CZ", "RU", "NO", "FI"):
        assert cc in by_raw["17.05.2024"]["region_hint"], f"missing {cc}"

    assert by_raw["05/17/2024"]["iso_date"] == "2024-05-17"
    assert by_raw["05/17/2024"]["format_style"] == "MDY_SLASH"
    assert "US" in by_raw["05/17/2024"]["region_hint"]

    assert by_raw["17/05/2024"]["iso_date"] == "2024-05-17"
    assert by_raw["17/05/2024"]["format_style"] == "DMY_SLASH"
    assert "GB" in by_raw["17/05/2024"]["region_hint"]

    assert by_raw["17 mai 2024"]["iso_date"] == "2024-05-17"
    assert by_raw["17 mai 2024"]["format_style"] == "NAMED"
    assert "FR" in by_raw["17 mai 2024"]["region_hint"]

    assert by_raw["17. Mai 2024"]["iso_date"] == "2024-05-17"
    assert by_raw["17. Mai 2024"]["format_style"] == "NAMED"
    assert "DE" in by_raw["17. Mai 2024"]["region_hint"]

    assert by_raw["17 May 2024"]["iso_date"] == "2024-05-17"
    assert by_raw["17 May 2024"]["format_style"] == "NAMED"

    # Evidence checks: same day -> SUPPORTS @ 0.75
    ev_sup = date_evidence([by_raw["17.05.2024"]], claimed_time_iso="2024-05-17T10:00:00", claimed_cc="DE")
    assert ev_sup is not None
    assert ev_sup.verdict == ClaimVerdict.SUPPORTS
    assert ev_sup.confidence == 0.75
    assert "DMY_DOT" not in ev_sup.summary

    # Different day -> CONTRADICTS @ 0.80
    ev_con = date_evidence([by_raw["05/17/2024"]], claimed_time_iso="2024-05-18T10:00:00", claimed_cc="DE")
    assert ev_con is not None
    assert ev_con.verdict == ClaimVerdict.CONTRADICTS
    assert ev_con.confidence == 0.80
    assert "MDY_SLASH" not in ev_con.summary

    # Robustness on bad inputs
    assert extract_dates(None) == []  # type: ignore[arg-type]
    assert extract_dates([None, 123, {}]) == []  # type: ignore[list-item]
    assert date_evidence([]) is None
    assert date_evidence(None) is None  # type: ignore[arg-type]

    print("ocr_dates.py self-test passed.")
