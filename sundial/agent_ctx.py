"""Agent context builder — the compact brief the agent loop investigates from.

``build_context()`` packs the current investigation state into a single
short string (<= ``MAX_CHARS`` chars) the agent loop drops at the top of
its tool-planning prompt:

    CLAIM:      what the submitter asserts (time / place / coords / cc)
    KNOWN:      highest-confidence evidence already in the pack
    SUSPECTED:  top geo-match candidates from the ranked list
    OPEN:       inconclusive-but-checkable gaps a tool call could close
    TOOL-HINT:  the tool names the agent may call

Stdlib only at import — the schema is never needed at all here: evidence
is read defensively (``Evidence`` objects *or* plain dicts, attr-or-key),
so the module loads and its ``__main__`` self-test runs with zero package
deps (same convention as dedupe_evidence / tz_check). Nothing in this
module raises; malformed input degrades to a thinner brief, not a
traceback — the agent still gets something to plan from.
"""
from __future__ import annotations

from typing import Any, Iterable

#: Hard cap on the returned context string — the brief must stay small
#: enough to sit in every agent prompt without crowding out tool output.
MAX_CHARS = 1200

#: Tool names the agent may call when no explicit registry is passed.
#: These mirror the function-level tools the agent loop exposes.
DEFAULT_TOOLS = (
    "geocode", "wiki_geo", "overpass_pois",
    "weather", "timezone_at", "sun_at", "context",
)

_TOP_KNOWN = 4        # KNOWN keeps the strongest few cards only
_TOP_SUSPECTED = 3    # SUSPECTED shows ranked[:3] + claimed-cc position
_SUMMARY_CHARS = 90   # per-card summary budget inside KNOWN

_VERDICTS = ("supports", "contradicts", "inconclusive")

#: Stage-note / rollup types are bookkeeping, not open questions — the
#: agent cannot close them with a tool call.
_NOT_CHECKABLE = frozenset({
    "vlm", "geo-resolve", "geo_rollup", "note", "provenance",
    "run_stats", "claim", "perception_note",
})


# ---------------------------------------------------------------------------
# Defensive readers — Evidence objects and plain dicts both work
# ---------------------------------------------------------------------------

def _get(item: Any, key: str, default=None):
    """Attr-or-key read so evidence arrives as dataclass or dict."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _conf(ev: Any) -> float:
    return max(0.0, min(1.0, _num(_get(ev, "confidence", 0.0) or 0.0)))


def _verdict(v: Any) -> str:
    """Normalize ClaimVerdict / str / repr to supports|contradicts|inconclusive.

    Same tolerance as dedupe_evidence._verdict_key: raw strings
    ("SUPPORTS"), enum members, even "ClaimVerdict.SUPPORTS" reprs.
    """
    s = str(getattr(v, "value", v)).lower().rsplit(".", 1)[-1].strip()
    return s if s in _VERDICTS else "inconclusive"


def _clean_str(v: Any) -> str | None:
    s = str(v or "").strip()
    return s or None


def _norm_cc(v: Any) -> str:
    """Uppercase 2-3 alpha country code, else ""."""
    s = str(v or "").strip().upper()
    return s if 2 <= len(s) <= 3 and s.isalpha() else ""


def _clip(s: Any, n: int = _SUMMARY_CHARS) -> str:
    """One-line, length-capped summary text."""
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Ranked-list normalization — same moving-target tolerance as osint
# ---------------------------------------------------------------------------

def _norm_ranked(raw: Any) -> list[dict]:
    """Normalize score_candidates rows to [{'cc','score'}] sorted desc.

    Accepts list-of-dicts ({cc|country_code|country|code, score|
    confidence|weight}), list-of-tuples ((cc, score, ...)), and a plain
    {cc: score} mapping. Dedupes by cc keeping the stronger row.
    """
    if isinstance(raw, dict):
        raw = [{"cc": k, "score": v} for k, v in raw.items()]
    out: list[dict] = []
    for r in raw or []:
        if isinstance(r, dict):
            cc = (r.get("cc") or r.get("country_code") or r.get("country")
                  or r.get("code"))
            score = r.get("score", r.get("confidence", r.get("weight", 0.0)))
        elif isinstance(r, (list, tuple)) and r:
            cc = r[0]
            score = r[1] if len(r) > 1 else 0.0
        else:
            continue
        ncc = _norm_cc(cc)
        if not ncc:
            continue
        for i, e in enumerate(out):
            if e["cc"] == ncc:
                if _num(score) > e["score"]:
                    out[i] = {"cc": ncc, "score": _num(score)}
                break
        else:
            out.append({"cc": ncc, "score": _num(score)})
    out.sort(key=lambda e: -e["score"])
    return out


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def claim_dict(claim_time=None, claim_place=None, anchor=None, cc=None) -> dict:
    """Pack the loose claim args into the {time_iso, place, lat, lon, cc} dict.

    ``anchor`` is whatever the caller has — a (lat, lon) tuple/list, a
    {lat, lon} dict, or a "lat,lon" string; anything unparseable degrades
    to None rather than raising.
    """
    lat = lon = None
    try:
        if isinstance(anchor, dict):
            lat, lon = anchor.get("lat"), anchor.get("lon")
        elif isinstance(anchor, str):
            a, b = anchor.split(",", 1)
            lat, lon = float(a), float(b)
        elif anchor is not None:
            lat, lon = anchor[0], anchor[1]
    except (TypeError, ValueError, IndexError, KeyError):
        lat = lon = None
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lat = lon = None
    return {
        "time_iso": _clean_str(claim_time),
        "place": _clean_str(claim_place),
        "lat": lat,
        "lon": lon,
        "cc": _norm_cc(cc) or _clean_str(cc),
    }


def open_questions(evidence_list: list) -> list[str]:
    """Inconclusive-but-checkable gaps in the pack, as short questions.

    Only INCONCLUSIVE cards qualify (supports/contradicts are settled),
    and only when a tool call could plausibly close the gap — stage notes
    and rollups are bookkeeping, not questions.
    """
    out: list[str] = []
    for ev in evidence_list or []:
        try:
            if _verdict(_get(ev, "verdict")) != "inconclusive":
                continue
            q = _question_for(ev)
        except Exception:
            q = None
        if q and q not in out:
            out.append(q)
    return out


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _question_for(ev: Any) -> str | None:
    """One short open-question phrase for an inconclusive card (or None)."""
    t = str(_get(ev, "type", "") or "").strip().lower()
    if t in _NOT_CHECKABLE:
        return None
    data = _get(ev, "data") or {}
    if not isinstance(data, dict):
        data = {}
    summary = _clip(_get(ev, "summary"), 80)

    if t == "clock":
        overlays = data.get("overlays") or []
        if not overlays:
            return "no readable clock overlay"
        low = summary.lower()
        if "consensus" in low or "distinct" in low:
            return "clock overlay reads disagree — no >=2-frame consensus"
        if "unparseable" in low or "no claimed" in low:
            return "clock overlay unread vs the claim"
        return f"clock overlay unchecked — {summary}"
    if t in ("landmark", "landmarks"):
        names = data.get("names") or []
        places = data.get("places") or []
        if names and not places:
            shown = ", ".join(str(n) for n in names[:2])
            return f"landmarks unverified — names won't geocode ({shown})"
        if not names:
            return "no landmark names to verify"
        return "landmarks unverified — geocoded but ambiguous vs the claim"
    if t in ("wiki_landmark", "wiki"):
        if not data.get("queried", True):
            return "wiki cross-check not run"
        if not data.get("matches"):
            return "no wiki match for named landmarks"
        return "wiki landmark check inconclusive"
    if t == "sign_style":
        tops = [str(k) for k in (data.get("top_conventions") or [])[:2]]
        tag = f" ({'/'.join(tops)})" if tops else ""
        return f"sign_style ambiguous{tag}"
    if t == "osint":
        cc = _norm_cc(data.get("claimed_cc"))
        if not cc:
            return "no claimed country to rank"
        return f"country ranking weak for claimed {cc}"
    if t in ("language", "langdetect"):
        return "signage language inconclusive"
    if t in ("geo", "geo_match"):
        return "sign-text geocode unresolved"
    if t == "geoguess":
        return "geoguess votes split — no stable country"
    if t == "plate_style":
        return "plate style inconclusive"
    if t in ("vegetation", "veg_geo"):
        return "vegetation signal inconclusive"
    if t == "weather":
        return "weather archive inconclusive for claimed time"
    if t in ("timezone", "tz", "tz_check"):
        return "timezone at claim unconfirmed"
    if t == "hemisphere":
        return "hemisphere cue unresolved"
    if t in ("audio", "audio_probe"):
        return "audio signal inconclusive"
    if t in ("scene", "scene_gate"):
        return "scene gate inconclusive"
    # Unknown inconclusive types still surface — a tool may exist for them.
    return f"{t or 'evidence'} unresolved" if t else None


def _fmt_ll(lat: Any, lon: Any) -> str:
    try:
        return f"({float(lat):.2f},{float(lon):.2f})"
    except (TypeError, ValueError):
        return ""


def _claim_body(claim: dict) -> str:
    """"filmed 2026-10-17T12:30 local at Husum, Germany (54.48,9.05)"."""
    c = claim if isinstance(claim, dict) else {}
    time_iso = _clean_str(c.get("time_iso") or c.get("claimed_time")
                          or c.get("time"))
    place = _clean_str(c.get("place") or c.get("claimed_place"))
    coords = _fmt_ll(c.get("lat"), c.get("lon"))
    cc = _norm_cc(c.get("cc"))
    if not (time_iso or place or coords or cc):
        return "none supplied"
    seg = "filmed"
    if time_iso:
        seg += f" {time_iso} local"
    where = " ".join(p for p in (place or "", coords) if p).strip()
    if where:
        seg += f" at {where}"
    if cc and cc not in (place or ""):
        seg += f" [{cc}]"
    return seg


def _known_items(evs: list, ranked: list[dict], claim: dict) -> list[str]:
    """Top-confidence settled cards, plus the claimed-cc rank digest."""
    items: list[str] = []
    cc = _norm_cc(claim.get("cc")) if isinstance(claim, dict) else ""
    if cc:
        for i, e in enumerate(ranked):
            if e["cc"] == cc:
                items.append(f"osint: {cc} #{i + 1} {e['score']:.2f}")
                break
    settled = [
        ev for ev in evs
        if _verdict(_get(ev, "verdict")) in ("supports", "contradicts")
    ]
    settled.sort(key=lambda e: -_conf(e))
    for ev in settled[:_TOP_KNOWN]:
        t = str(_get(ev, "type", "") or "?").strip() or "?"
        v = _verdict(_get(ev, "verdict"))
        items.append(f"{t}: {v} ({_conf(ev):.2f}) — "
                     f"{_clip(_get(ev, 'summary'))}")
    return items


def _suspect_str(ranked: list[dict], claim: dict) -> str:
    """"DE 0.65 > FR 0.31 > NL 0.22" (+ claimed-cc position when absent)."""
    if not ranked:
        return "none"
    top = " > ".join(f"{e['cc']} {e['score']:.2f}"
                     for e in ranked[:_TOP_SUSPECTED])
    cc = _norm_cc(claim.get("cc")) if isinstance(claim, dict) else ""
    if cc and all(e["cc"] != cc for e in ranked[:_TOP_SUSPECTED]):
        for i, e in enumerate(ranked):
            if e["cc"] == cc:
                top += f" — claim {cc} only #{i + 1} ({e['score']:.2f})"
                break
    return top


def _tool_hint(tools: Iterable[str] | None) -> str:
    names = []
    for t in (tools or DEFAULT_TOOLS):
        s = str(t or "").strip()
        if s and s not in names:
            names.append(s)
    return ", ".join(names) if names else "none"


def _assemble(claim: dict, known: list[str], suspected: str,
              open_q: list[str], tools) -> str:
    return "\n".join([
        "CLAIM: " + _claim_body(claim),
        "KNOWN: " + ("; ".join(known) if known else "none yet"),
        "SUSPECTED: " + suspected,
        "OPEN: " + ("; ".join(open_q) if open_q else "none"),
        "TOOL-HINT: " + _tool_hint(tools),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_context(evidence_list: list, claim: dict, ranked: list | None = None,
                  tools: Iterable[str] | None = None) -> str:
    """The <=MAX_CHARS context_summary string the agent loop plans from.

    ``evidence_list`` — Evidence objects or dicts from the pack.
    ``claim``         — {time_iso, place, lat, lon, cc} (see claim_dict).
    ``ranked``        — score_candidates rows in any tolerated shape.
    ``tools``         — registry names to advertise; DEFAULT_TOOLS if None.

    Fits the budget by shedding OPEN items first, then KNOWN items, then
    hard-truncating; CLAIM and TOOL-HINT always survive. Never raises.
    """
    try:
        evs = list(evidence_list or [])
        claim = claim if isinstance(claim, dict) else {}
        ranked_n = _norm_ranked(ranked)
        known = _known_items(evs, ranked_n, claim)
        open_q = open_questions(evs)
        suspected = _suspect_str(ranked_n, claim)

        out = _assemble(claim, known, suspected, open_q, tools)
        while len(out) > MAX_CHARS and (open_q or known):
            if open_q:
                open_q.pop()          # shed the least-truncated-tolerant first
            else:
                known.pop()
            out = _assemble(claim, known, suspected, open_q, tools)
        if len(out) > MAX_CHARS:
            out = out[: MAX_CHARS - 3].rstrip() + "..."
        return out
    except Exception:
        return "CLAIM: (context build failed)"


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/agent_ctx.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # allow `python3 sundial/agent_ctx.py` directly
        import os

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from schema import ClaimVerdict, Evidence
        except ImportError:
            ClaimVerdict = Evidence = None

    fails = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        global fails
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label} {detail}")

    print("agent_ctx self-test:")

    # --- fake evidence: schema Evidence when importable, plain dicts always --
    if Evidence is not None:
        evs = [
            Evidence(type="solar", summary="sun 36.6° fits window",
                     verdict=ClaimVerdict.SUPPORTS, confidence=0.77),
            Evidence(type="clock", summary="no readable burned-in timestamp",
                     verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.0,
                     data={"overlays": []}),
            Evidence(type="landmark", summary="2 names but none geocoded",
                     verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.2,
                     data={"names": ["Husum Schloss", "Marktplatz"],
                           "places": []}),
            Evidence(type="sign_style", summary="weakly matches EU",
                     verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.2,
                     data={"top_conventions": ["EU", "UK"]}),
            Evidence(type="weather", summary="overcast consistent",
                     verdict=ClaimVerdict.SUPPORTS, confidence=0.41),
        ]
    else:
        evs = []
    evs += [  # dicts must read identically to Evidence objects
        {"type": "wiki_landmark", "verdict": "inconclusive",
         "confidence": 0.25, "summary": "no wikipedia title matches",
         "data": {"queried": True, "matches": []}},
    ]

    # --- claim_dict ---------------------------------------------------------
    c = claim_dict("2026-10-17T12:30", "Husum, Germany", (54.48, 9.05), "de")
    check("claim_dict packs fields",
          c == {"time_iso": "2026-10-17T12:30", "place": "Husum, Germany",
                "lat": 54.48, "lon": 9.05, "cc": "DE"}, f"(got {c})")
    check("claim_dict tolerates string anchor",
          claim_dict(None, None, "1.5,2.5", None)["lat"] == 1.5)
    check("claim_dict tolerates junk anchor",
          claim_dict(None, None, "garbage", None)["lat"] is None)

    # --- build_context ------------------------------------------------------
    ranked = [{"cc": "DE", "score": 0.65}, {"cc": "FR", "score": 0.31},
              ("NL", 0.22)]
    ctx = build_context(evs, c, ranked)
    print("\n" + ctx + f"\n  ({len(ctx)} chars)\n")
    for sec in ("CLAIM:", "KNOWN:", "SUSPECTED:", "OPEN:", "TOOL-HINT:"):
        check(f"section {sec} present", sec in ctx)
    check("<= MAX_CHARS", len(ctx) <= MAX_CHARS)
    check("claim line shape",
          "filmed 2026-10-17T12:30 local at Husum, Germany (54.48,9.05)"
          in ctx)
    check("suspected ordering", "DE 0.65 > FR 0.31 > NL 0.22" in ctx)
    check("claimed cc rank in KNOWN", "osint: DE #1 0.65" in ctx)
    check("tool hint defaults", "geocode" in ctx and "sun_at" in ctx)

    # --- OPEN lists real gaps ------------------------------------------------
    qs = open_questions(evs)
    print("  open questions:", qs)
    check("clock gap surfaced",
          any("clock overlay" in q for q in qs))
    check("landmark gap surfaced",
          any("landmarks unverified" in q for q in qs))
    check("wiki gap surfaced",
          any("wiki" in q for q in qs))
    check("sign_style gap surfaced",
          any("sign_style ambiguous" in q for q in qs))
    check("settled cards are not questions",
          not any("solar" in q or "weather" in q for q in qs))

    # --- degenerate inputs never raise --------------------------------------
    empty = build_context([], {}, None)
    check("empty inputs still sectioned",
          all(s in empty for s in ("CLAIM:", "OPEN:", "TOOL-HINT:")))
    check("build_context(None-ish) returns str",
          isinstance(build_context(None, None, None), str))
    blob = build_context(
        [{"type": "x", "verdict": "inconclusive", "summary": "y" * 500}] * 30,
        {"place": "p" * 2000}, {"DE": 0.9})
    check("absurd input still capped", len(blob) <= MAX_CHARS,
          f"({len(blob)} chars)")

    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        sys.exit(1)
    print("OK — agent_ctx self-test passed")
