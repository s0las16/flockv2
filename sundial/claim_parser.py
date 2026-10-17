"""Claim-text parser — "filmed Tuesday 3pm in London" -> pipeline kwargs.

Judges type rough claims; this turns them into a naive local ISO timestamp
and a place string shaped exactly like pipeline.run's (claim_time,
claim_place) kwargs. Pure stdlib (re + datetime), no VLM, no network.

    parse_claim_text("filmed Tuesday 3pm in London")
    -> {"place": "London", "time_iso": "...T15:00", "day_offset": -1, ...}

    claim_to_args("filmed Tuesday 3pm in London")
    -> ("...T15:00", "London")   # run(clip, claim_time=..., claim_place=...)

Interpretation choices (claims describe past events):
  * bare weekday -> most recent *past* occurrence (same weekday = a week ago)
  * bare hour "3:30" -> 24h clock; "3pm"/"3 pm" -> meridiem
  * date without year ("Oct 15") -> this year, or last year if that's future
  * time-of-day with no day -> today (day_offset stays None: day not stated)
"""
from __future__ import annotations

import re
from datetime import date as _date, datetime, time as _time, timedelta

__all__ = ["parse_claim_text", "claim_to_args"]

# --- time-of-day vocabulary -------------------------------------------------

# word -> (hour, minute). dusk/dawn are demo-friendly extras beyond spec.
_TIME_WORDS = {
    "morning": (9, 0),
    "noon": (12, 0),
    "midday": (12, 0),
    "afternoon": (15, 0),
    "evening": (19, 0),
    "dusk": (19, 0),
    "night": (22, 0),
    "tonight": (22, 0),
    "midnight": (0, 0),
    "dawn": (6, 0),
}
_RE_NAMED = re.compile(
    r"\b(midnight|midday|morning|afternoon|evening|tonight|noon|night|dusk|dawn)\b"
)

_MER = r"(a\.?\s?m\.?|p\.?\s?m\.?)"
# "22:30", "3:30pm" — lookarounds keep "25:00"/"3:75"/"22:30:15" out.
_RE_CLOCK = re.compile(
    rf"(?<![\d:.])(2[0-3]|[01]?\d)\s*:\s*([0-5]\d)\s*{_MER}?(?![\d:])"
)
# "3pm", "3 pm", "11 p.m."
_RE_AMPM = re.compile(rf"(?<![\d:.])(1[0-2]|0?[1-9])\s*{_MER}(?![\d:])")
# "half past 3", "half past 3pm"
_RE_HALF = re.compile(rf"\bhalf\s+past\s+(1[0-2]|0?[1-9])\s*{_MER}?\b")
# "3 o'clock"
_RE_OCLOCK = re.compile(r"(?<![\d:.])(1[0-2]|0?[1-9])\s*o'?clock\b")
# "at 6", "around 22" — (?![\d:.]) so "at 22:30" is left to _RE_CLOCK.
_RE_BARE = re.compile(
    rf"\b(?:at|around|about)\s+(2[0-3]|[01]?\d)(?![\d:.])\s*{_MER}?\b"
)

# --- day vocabulary ----------------------------------------------------------

_WEEKDAYS = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}
_RE_WD = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tues|thurs|tue|thur|mon|wed|thu|fri|sat|sun)\b"
)

# Longest-first: "the day before yesterday" must win over its tail "yesterday".
_RELATIVE = [
    ("the day before yesterday", -2),
    ("day before yesterday", -2),
    ("last night", -1),
    ("last evening", -1),
    ("this morning", 0),
    ("this afternoon", 0),
    ("this evening", 0),
    ("last week", -7),
    ("yesterday", -1),
    ("today", 0),
    ("tonight", 0),
    ("tomorrow", 1),
]
_REL_MAP = dict(_RELATIVE)
_RE_REL = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in _RELATIVE) + r")\b")

_RE_ISODATE = re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MON_BODY = (
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec"
)
# "Oct 15", "October 15th 2025"
_RE_MDY = re.compile(
    rf"\b(?P<mon>{_MON_BODY})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b"
    rf"(?:\s*,?\s*(?P<yr>\d{{4}})(?!\d))?"
)
# "15 October", "15th of Oct, 2025"
_RE_DMY = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mon>{_MON_BODY})\.?"
    rf"(?:\s*,?\s*(?P<yr>\d{{4}})(?!\d))?"
)

# --- place machinery ----------------------------------------------------------

_PREP = re.compile(r"\b(?:in|at|from|near)\b")
# fillers allowed between preposition and the actual time/place token
_SKIP = re.compile(r"\s*(?:(?:the|a|an|about|around)\s+)*")
# connectors glued onto a time span ("at 3pm", "on Tuesday", "in the morning")
_HEAD = re.compile(r"(?:(?<!\w)(?:on|at|in|from|the|a|an|about|around|of)\s+)+$")
# trailing connectors accidentally captured with a place ("London on ")
_TAIL_JUNK = re.compile(r"(?:[\s,;]+(?:on|at|in|from|the|a|an|of|and|to|for))+$")
# verbs/stopwords removed before the comma-segment fallback
_FILLER = re.compile(
    r"\b(?:filmed|film|films|shot|shoot|shooting|taken|took|recorded|"
    r"captured|made|video|videos|clip|clips|footage|this|that|it|its|"
    r"was|is|were|are|be|been|the|a|an|on|at|in|from|of|and|to|for|"
    r"around|about|approx(?:imately)?|roughly|near|by|my|mine|me|i|we|"
    r"you|they|he|she|here|there|just|so|really|very|actually|probably|"
    r"definitely|maybe|check|watch|look|see|seen|claims?|says?|said|"
    r"shows?|showing|prove|proof|verify|hey|hi|hello|pls|please|"
    r"no|idea|where|when|who|what|why|how|dont|don't|know)\b",
    re.IGNORECASE,
)


def _hour12(h: int, mer: str | None) -> int | None:
    """hour + optional am/pm -> 0..23, or None if the combo is nonsense."""
    if mer:
        if not 1 <= h <= 12:
            return None
        if mer.replace(".", "").replace(" ", "").startswith("p"):
            return h % 12 + 12
        return h % 12
    return h if 0 <= h <= 23 else None


def _named_date(day: int, mon: int, yr: int | None, now: datetime) -> _date:
    """Date from month/day[/year]; missing year -> most recent past instance."""
    if yr:
        return _date(yr, mon, day)
    d = _date(now.year, mon, day)
    if d > now.date():
        d = _date(now.year - 1, mon, day)  # may raise Feb-29; caller catches
    return d


def _coverage(low: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, merged spans, each extended back over 'at'/'on'/'in the ' etc."""
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        head = _HEAD.search(low[:s])
        if head:
            s = head.start()
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _inside(pos: int, cov: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in cov)


def _clean_place(raw: str) -> str | None:
    raw = raw.strip(" \t\n,.;:!?\"'()[]")
    prev = None
    while prev != raw:  # chains like "london on, "
        prev = raw
        raw = _TAIL_JUNK.sub("", raw).rstrip(" \t,.;:")
    if not raw or not re.search(r"[a-zA-Z]", raw):
        return None
    if len(raw) > 60 or len(raw.split()) > 5:
        return None
    return raw


def _place_after_prep(text: str, low: str, cov: list[tuple[int, int]]) -> str | None:
    """'in'/'at'/'from'/'near' followed by non-time text -> that text, cut at
    the next time span. 'at 3pm'/'in the morning' are skipped: the prep just
    introduces a time expression."""
    for m in _PREP.finditer(low):
        if _inside(m.start(), cov):
            continue  # glued to a time span already ("at 22:30")
        k = m.end() + _SKIP.match(low[m.end():]).end()
        if _inside(k, cov):
            continue  # "in the morning" — a time follows, not a place
        end = len(text)
        for s, _ in cov:
            if s >= k:
                end = min(end, s)
        raw = _clean_place(text[m.end():end])
        if raw:
            return raw
    return None


def _place_fallback(text: str, cov: list[tuple[int, int]]) -> str | None:
    """No preposition: blank out time spans + filler, take remaining comma
    segments of <=4 words ('yesterday morning, Tokyo' -> 'Tokyo')."""
    chars = list(text)
    for s, e in cov:
        chars[s:e] = " " * (e - s)
    cleaned = _FILLER.sub(" ", "".join(chars))  # case kept for the raw span
    segs = []
    for seg in cleaned.split(","):
        seg = seg.strip(" \t,.;:!?\"'()[]")
        if seg and re.search(r"[a-zA-Z]", seg) and len(seg.split()) <= 4:
            segs.append(seg)
    return ", ".join(segs) if segs else None


def _title(raw: str) -> str:
    """Uppercase each token's first letter when lowercase; existing caps
    ('USA', 'McDonald's') survive untouched."""
    out = []
    for tok in raw.split(" "):
        if not tok:
            continue
        m = re.search(r"[a-zA-Z]", tok)
        if m and tok[m.start()].islower():
            tok = tok[: m.start()] + tok[m.start()].upper() + tok[m.start() + 1:]
        out.append(tok)
    return " ".join(out)


def parse_claim_text(text: str, now: datetime | None = None) -> dict:
    """Parse a typed claim into place + naive-local ISO time fields.

    Keys: place / place_raw (capitalized vs as-typed span), time_iso (naive,
    no tz — pipeline.run compares it to local clock reads), day_offset,
    time_of_day ("HH:MM"), confidence 0..1, raw (input echoed back).
    """
    now = (now or datetime.now()).replace(tzinfo=None)
    result: dict = {
        "place": None,
        "place_raw": None,
        "time_iso": None,
        "day_offset": None,
        "time_of_day": None,
        "confidence": 0.0,
        "raw": text,
    }
    if not text or not text.strip():
        return result
    low = text.lower()

    spans: list[tuple[int, int]] = []          # all time-ish char spans
    # (pos, prio, hour, minute, bare) — "bare" = meridiem-less hour, still
    # free to snap into a named period ("6 in the evening" -> 18:00)
    tods: list[tuple[int, int, int, int, bool]] = []
    days: list[tuple[int, int]] = []            # (pos, day_offset)
    dates: list[tuple[int, _date]] = []         # (pos, explicit date)
    named_hours: list[tuple[int, int]] = []     # (pos, canonical hour)

    def _add_tod(m: re.Match, hour: int | None, minute: int,
                 prio: int = 0, bare: bool = False) -> None:
        spans.append(m.span())
        if hour is not None and 0 <= hour <= 23 and 0 <= minute <= 59:
            tods.append((m.start(), prio, hour, minute, bare))

    # time of day — explicit forms (prio 0) beat named words (prio 1)
    for m in _RE_CLOCK.finditer(low):
        _add_tod(m, _hour12(int(m.group(1)), m.group(3)), int(m.group(2)))
    for m in _RE_AMPM.finditer(low):
        _add_tod(m, _hour12(int(m.group(1)), m.group(2)), 0)
    for m in _RE_HALF.finditer(low):
        _add_tod(m, _hour12(int(m.group(1)), m.group(2)), 30,
                 bare=m.group(2) is None)
    for m in _RE_OCLOCK.finditer(low):
        _add_tod(m, int(m.group(1)), 0, bare=True)
    for m in _RE_BARE.finditer(low):
        _add_tod(m, _hour12(int(m.group(1)), m.group(2)), 0,
                 bare=m.group(2) is None)
    for m in _RE_NAMED.finditer(low):
        h, mi = _TIME_WORDS[m.group(1)]
        named_hours.append((m.start(), h))
        _add_tod(m, h, mi, prio=1)

    # days — explicit dates outrank relative words when both appear
    for m in _RE_ISODATE.finditer(low):
        spans.append(m.span())
        try:
            dates.append((m.start(), _date(int(m.group(1)), int(m.group(2)),
                                           int(m.group(3)))))
        except ValueError:
            pass
    for m in _RE_MDY.finditer(low):
        spans.append(m.span())
        try:
            dates.append((m.start(), _named_date(
                int(m.group("day")), _MONTHS[m.group("mon")],
                int(m.group("yr")) if m.group("yr") else None, now)))
        except ValueError:
            pass
    for m in _RE_DMY.finditer(low):
        spans.append(m.span())
        try:
            dates.append((m.start(), _named_date(
                int(m.group("day")), _MONTHS[m.group("mon")],
                int(m.group("yr")) if m.group("yr") else None, now)))
        except ValueError:
            pass
    for m in _RE_REL.finditer(low):
        spans.append(m.span())
        days.append((m.start(), _REL_MAP[m.group(1)]))
    for m in _RE_WD.finditer(low):
        spans.append(m.span())
        back = (now.weekday() - _WEEKDAYS[m.group(1)]) % 7 or 7
        days.append((m.start(), -back))

    tods.sort(key=lambda t: (t[1], t[0]))   # explicit first, then leftmost
    days.sort()
    dates.sort()
    chosen_date = dates[0][1] if dates else None
    day_offset = ((chosen_date - now.date()).days if chosen_date
                  else days[0][1] if days else None)
    result["day_offset"] = day_offset

    if tods:
        _, _, hh, mm, bare = tods[0]
        if bare and named_hours and 1 <= hh <= 12:
            canon = min(named_hours)[1]
            if abs(hh % 12 + 12 - canon) < abs(hh - canon):
                hh = hh % 12 + 12
        base = chosen_date or (now.date() + timedelta(days=day_offset or 0))
        result["time_of_day"] = f"{hh:02d}:{mm:02d}"
        result["time_iso"] = datetime.combine(
            base, _time(hh, mm)).isoformat(timespec="minutes")

    cov = _coverage(low, spans)
    place_raw = _place_after_prep(text, low, cov)
    if place_raw is None:
        place_raw = _place_fallback(text, cov)
    if place_raw:
        result["place_raw"] = place_raw
        result["place"] = _title(place_raw)

    conf = 0.0
    if result["time_of_day"] is not None:
        conf += 0.45
    if day_offset is not None:
        conf += 0.25
    if result["place"] is not None:
        conf += 0.30
    if result["time_of_day"] is None:   # no time words -> stays a weak parse
        conf = min(conf, 0.5)
    result["confidence"] = round(min(conf, 1.0), 2)
    return result


def claim_to_args(text: str) -> tuple[str | None, str | None]:
    """(claim_time, claim_place) — pipeline.run kwargs, verbatim."""
    r = parse_claim_text(text)
    return r["time_iso"], r["place"]


if __name__ == "__main__":
    _samples = [
        "filmed Tuesday 3pm in London",
        "yesterday morning, Tokyo",
        "shot at 22:30 in Paris, Texas",
        "in Berlin on Oct 1 at noon",
        "last night at the Eiffel Tower",
        "2025-09-30 06:45 in NYC",
        "no idea where this was",
        "",
    ]
    for _s in _samples:
        print(f"{_s!r:45} -> {parse_claim_text(_s)}")

    # deterministic checks — fixed "now": Wednesday 2025-10-15 18:00
    _fixed = datetime(2025, 10, 15, 18, 0)
    _r = parse_claim_text("filmed Tuesday 3pm in London", now=_fixed)
    assert _r["place"] == "London" and _r["time_of_day"] == "15:00", _r
    assert _r["day_offset"] == -1 and _r["time_iso"] == "2025-10-14T15:00", _r
    _r = parse_claim_text("yesterday morning, Tokyo", now=_fixed)
    assert _r["place"] == "Tokyo" and _r["time_iso"] == "2025-10-14T09:00", _r
    _r = parse_claim_text("shot at 22:30 in Paris, Texas", now=_fixed)
    assert _r["place"] == "Paris, Texas", _r
    assert _r["time_iso"] == "2025-10-15T22:30", _r
    _r = parse_claim_text("in Berlin on Oct 1 at noon", now=_fixed)
    assert _r["time_iso"] == "2025-10-01T12:00" and _r["place"] == "Berlin", _r
    _r = parse_claim_text("day before yesterday, half past 3pm, Cairo", now=_fixed)
    assert _r["time_iso"] == "2025-10-13T15:30" and _r["place"] == "Cairo", _r
    assert parse_claim_text("", now=_fixed)["confidence"] == 0.0
    assert claim_to_args("yesterday morning, Tokyo")[0] is not None
    print("self-test OK")
