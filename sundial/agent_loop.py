"""Bounded tool-calling investigation loop for the Sundial pipeline.

The agent reads the compact context brief from ``agent_ctx.build_context``
(CLAIM / KNOWN / SUSPECTED / OPEN / TOOL-HINT), plans a handful of tool
calls to close the OPEN questions, executes them through
``agent_tools.call``, and folds every result into a transcript the
adjudicator can cite as provenance.

Two planner modes:

- **LLM** — pass ``llm_fn``, any callable ``prompt -> str`` (an
  OpenAI-compatible chat wrapper, a local model, a test stub). The
  prompt asks for a JSON array of ``{"tool", "args", "why"}`` objects;
  unparseable output degrades to the rule planner, never a traceback.
- **Rules** (default, fully offline / deterministic) — keyword-routes
  the OPEN lines: place->geocode, lat->wiki_geo, time->weather/sun_at,
  cc->pip2cc; args are parsed back out of the CLAIM line.

``run_loop`` stops when the plan comes back empty (nothing left worth
calling), the wall-clock budget is spent, or ``max_steps`` calls ran.
Nothing in this module raises — every tool failure lands in the
transcript as ``ok: False`` and every missing sibling module degrades
to a thinner result.

Stdlib only at import; ``agent_tools`` and the schema are imported
lazily so this file loads and self-tests with zero live deps:

    python3 sundial/agent_loop.py
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

__all__ = ["plan", "run_loop", "transcript_evidence", "findings",
           "MAX_PLAN", "DEFAULT_MAX_STEPS", "DEFAULT_BUDGET_S"]

#: Hard cap on calls per plan round — keeps a runaway planner honest.
MAX_PLAN = 6
DEFAULT_MAX_STEPS = 6
DEFAULT_BUDGET_S = 45.0

#: Transcript cards are provenance, not proof: fixed INCONCLUSIVE @ 0.3.
_EVIDENCE_CONF = 0.3

#: Tools the rule planner may emit when no catalog is supplied — mirrors
#: agent_ctx.DEFAULT_TOOLS plus pip2cc for the coords->country check.
_DEFAULT_TOOLS = frozenset({
    "geocode", "wiki_geo", "overpass_pois", "weather",
    "timezone_at", "sun_at", "pip2cc", "context",
})

_COORDS_RE = re.compile(r"\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)")
_CC_RE = re.compile(r"\[([A-Za-z]{2,3})\]")
_TIME_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}(?:[T ][0-9:.]+)?)")
_PARENS_RE = re.compile(r"\(([^()]*)\)")

_PLACE_RE = re.compile(r"\b(place|landmarks?|geocod\w*|names?|named)\b",
                       re.IGNORECASE)
_LAT_RE = re.compile(r"\b(lat|lat,lon|coords?|coordinates|wiki)\b",
                     re.IGNORECASE)
_TIME_KW_RE = re.compile(
    r"\b(time|clock|sun|weather|timezone|tz|lighting|timestamp)\b",
    re.IGNORECASE)
_CC_KW_RE = re.compile(r"\bcc\b|countr|ranking|osint|hemisphere",
                       re.IGNORECASE)
_TZ_KW_RE = re.compile(r"\b(timezone|tz)\b", re.IGNORECASE)

#: Result keys worth surfacing in one-line findings / RESULTS fold-back.
_FIND_KEYS = ("summary", "display_name", "name", "cc", "country_code",
              "country", "lat", "lon", "tz", "timezone", "temp_c",
              "elev_deg", "az_deg", "verdict", "error", "matches", "count")


# ---------------------------------------------------------------------------
# Lazy sibling imports — degrade, never raise
# ---------------------------------------------------------------------------

def _agent_tools():
    """The agent_tools module, or None when the sibling isn't there yet."""
    try:
        from . import agent_tools
        return agent_tools
    except Exception:
        pass
    try:
        import agent_tools  # direct-script use (sundial/ on sys.path)
        return agent_tools
    except Exception:
        return None


def _schema():
    """Lazy schema import — package first, direct-script fallback."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:
        from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


# ---------------------------------------------------------------------------
# Catalog normalization — tolerate whatever shape agent_tools.catalog is
# ---------------------------------------------------------------------------

def _materialize_catalog(catalog: Any) -> Any:
    """catalog may be a list, a {name: spec} dict, or a callable factory."""
    if callable(catalog):
        try:
            return catalog()
        except Exception:
            return None
    return catalog


def _catalog_items(catalog: Any) -> list:
    cat = _materialize_catalog(catalog)
    if isinstance(cat, dict):
        out = []
        for k, v in cat.items():
            if isinstance(v, dict):
                row = dict(v)
                row.setdefault("name", str(k))
            else:
                row = {"name": str(k), "description": str(v)}
            out.append(row)
        return out
    if isinstance(cat, (list, tuple, set)):
        return list(cat)
    return []


def _catalog_names(catalog: Any) -> set:
    names = set()
    for item in _catalog_items(catalog):
        n = None
        if isinstance(item, str):
            n = item
        elif isinstance(item, dict):
            n = item.get("name") or item.get("tool") or item.get("id")
        else:
            n = getattr(item, "name", None) or getattr(item, "__name__", None)
        if n:
            names.add(str(n).strip())
    return {n for n in names if n}


def _catalog_lines(catalog: Any, allowed: set) -> list[str]:
    """'name {args} — description' rows for the LLM prompt."""
    specs = {}
    for item in _catalog_items(catalog):
        if isinstance(item, dict):
            n = item.get("name") or item.get("tool") or item.get("id")
            if n:
                specs[str(n)] = item
        elif isinstance(item, str):
            specs.setdefault(item, {})
    lines = []
    for n in sorted(allowed):
        spec = specs.get(n) or {}
        desc = (spec.get("description") or spec.get("desc")
                or spec.get("doc") or "")
        args = (spec.get("args") or spec.get("params")
                or spec.get("parameters") or "")
        line = n
        if args:
            if isinstance(args, str):
                line += f" {args}"
            else:
                try:
                    line += f" {json.dumps(args, default=str)[:80]}"
                except Exception:
                    pass
        if desc:
            line += f" — {str(desc)[:80]}"
        lines.append(line)
    return lines


def _catalog() -> Any:
    """agent_tools.catalog if reachable, else the default name list."""
    at = _agent_tools()
    cat = getattr(at, "catalog", None) if at is not None else None
    return cat if cat is not None else sorted(_DEFAULT_TOOLS)


# ---------------------------------------------------------------------------
# Brief parsing — pull CLAIM fields and OPEN items back out of the string
# ---------------------------------------------------------------------------

def _section(context_str: Any, name: str) -> str:
    """The 'NAME: ...' line body from the context brief, else ''."""
    for line in str(context_str or "").splitlines():
        if line.strip().upper().startswith(name.upper() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _open_items(context_str: Any) -> list[str]:
    """OPEN questions as a list of phrases ('OPEN: none' -> [])."""
    body = _section(context_str, "OPEN")
    if not body or body.strip().lower() in ("none", "none yet"):
        return []
    return [q.strip() for q in body.split(";") if q.strip()]


def _parse_claim(context_str: Any) -> dict:
    """{time_iso, place, lat, lon, cc} recovered from the CLAIM line."""
    line = _section(context_str, "CLAIM")
    out = {"time_iso": None, "place": None, "lat": None,
           "lon": None, "cc": None}
    if not line or line.lower() == "none supplied":
        return out
    m = _COORDS_RE.search(line)
    if m:
        try:
            out["lat"], out["lon"] = float(m.group(1)), float(m.group(2))
        except (TypeError, ValueError):
            pass
    m = _CC_RE.search(line)
    if m:
        out["cc"] = m.group(1).upper()
    m = _TIME_RE.search(line)
    if m:
        out["time_iso"] = m.group(1).rstrip(".,;")
    seg = _COORDS_RE.sub(" ", line)
    seg = _CC_RE.sub(" ", seg)
    if out["time_iso"]:
        seg = seg.replace(out["time_iso"], " ")
    seg = re.sub(r"\b(filmed|local)\b", " ", seg)
    m = re.search(r"\bat\b(.+)", seg)
    place = m.group(1) if m else seg
    place = " ".join(place.split()).strip(" ,;-–—")
    out["place"] = place or None
    return out


def _names_in(question: str) -> list[str]:
    """Parenthesized name list in an open question: '(A, B)' -> ['A','B']."""
    m = _PARENS_RE.search(question or "")
    if not m:
        return []
    return [p.strip() for p in re.split(r"[,/]", m.group(1))
            if p.strip()][:3]


# ---------------------------------------------------------------------------
# Planners
# ---------------------------------------------------------------------------

def _plan_rules(context_str: Any, allowed: set) -> list[dict]:
    """Deterministic offline planner — keyword-routes OPEN questions.

    place-ish -> geocode, lat-ish -> wiki_geo, time-ish -> weather+sun_at
    (+timezone_at for tz questions), cc-ish -> pip2cc. When a tool needs
    coords the brief doesn't have, a geocode(place) call is chained so a
    later round can use them.
    """
    claim = _parse_claim(context_str)
    coords = (claim["lat"] is not None and claim["lon"] is not None)
    calls: list[dict] = []

    def add(tool: str, args: dict, why: str) -> None:
        if len(calls) >= MAX_PLAN or tool not in allowed:
            return
        if any(c["tool"] == tool and c["args"] == args for c in calls):
            return
        calls.append({"tool": tool, "args": args, "why": why[:80]})

    def need_coords(why: str) -> None:
        """Chained geocode when coords-dependent tools lack an anchor."""
        if claim["place"]:
            add("geocode", {"query": claim["place"]},
                f"resolve coords — {why}"[:80])

    for q in _open_items(context_str):
        why = f"open: {q[:60]}"
        ql = q.lower()
        if _PLACE_RE.search(ql):
            names = _names_in(q) if ("name" in ql or "landmark" in ql) else []
            for nm in names or ([claim["place"]] if claim["place"] else []):
                add("geocode", {"query": nm}, why)
        if _LAT_RE.search(ql):
            if coords:
                add("wiki_geo",
                    {"lat": claim["lat"], "lon": claim["lon"]}, why)
            else:
                need_coords("for wiki check")
        if _TIME_KW_RE.search(ql):
            if coords:
                add("weather", {"lat": claim["lat"], "lon": claim["lon"],
                                "time_iso": claim["time_iso"]}, why)
                add("sun_at", {"lat": claim["lat"], "lon": claim["lon"],
                               "time_iso": claim["time_iso"]}, why)
                if _TZ_KW_RE.search(ql):
                    add("timezone_at",
                        {"lat": claim["lat"], "lon": claim["lon"]}, why)
            else:
                need_coords("for weather/sun check")
        if _CC_KW_RE.search(ql):
            if coords:
                add("pip2cc", {"lat": claim["lat"], "lon": claim["lon"],
                               "expect_cc": claim["cc"]}, why)
            else:
                need_coords("for country check")
    return calls[:MAX_PLAN]


def _llm_prompt(context_str: Any, catalog: Any, allowed: set) -> str:
    tools = "\n".join("- " + ln for ln in _catalog_lines(catalog, allowed))
    return (
        "You are the Sundial verification agent. Read the investigation "
        "brief and plan up to {n} tool calls that close the OPEN "
        "questions.\n\nBRIEF:\n{brief}\n\nTOOLS:\n{tools}\n\n"
        "Reply with ONLY a JSON array, no prose:\n"
        '[{{"tool": "<name>", "args": {{...}}, "why": "<short reason>"}}]'
    ).format(n=MAX_PLAN, brief=str(context_str or "")[:4000],
             tools=tools or "- (none)")


def _parse_llm_calls(text: Any, allowed: set):
    """Best-effort JSON-array extraction from an LLM reply.

    Returns a list of calls (possibly EMPTY — a parsed `[]` is the LLM
    saying "done", and callers must honour it), or None when nothing
    usable parsed so callers can fall back to the rule planner.
    """
    s = str(text or "").strip()
    if not s:
        return None
    candidates = [s]
    i, j = s.find("["), s.rfind("]")
    if 0 <= i < j:
        candidates.append(s[i:j + 1])
    # tolerate a ```json fenced block
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", s, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        raw = obj
        if isinstance(raw, dict):
            raw = (raw.get("calls") or raw.get("plan")
                   or raw.get("tools"))
        if not isinstance(raw, list):
            continue
        if not raw:
            return []          # explicit "nothing more to call"
        out = []
        for c in raw:
            if not isinstance(c, dict):
                continue
            tool = str(c.get("tool") or c.get("name") or "").strip()
            if not tool or (allowed and tool not in allowed):
                continue
            args = c.get("args") or c.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            out.append({"tool": tool, "args": args,
                        "why": str(c.get("why") or c.get("reason")
                                   or "llm")[:80]})
        if out:
            return out[:MAX_PLAN]
        # parsed but every call filtered out -> treat as unparseable
        return None
    return None


def plan(context_str: Any, catalog: Any = None,
         llm_fn: Callable | None = None) -> list[dict]:
    """Plan <= MAX_PLAN tool calls from the context brief.

    ``catalog``   — agent_tools.catalog in any shape (list of names /
    list of spec dicts / {name: spec} / callable). None -> resolve
    lazily, falling back to _DEFAULT_TOOLS.
    ``llm_fn``    — optional callable prompt->str; its JSON reply wins
    when parseable, otherwise the deterministic rule planner runs.

    Returns [{"tool": str, "args": dict, "why": str}]. Never raises.
    """
    try:
        if catalog is None:
            catalog = _catalog()
        allowed = _catalog_names(catalog) or set(_DEFAULT_TOOLS)
        if callable(llm_fn):
            try:
                reply = llm_fn(_llm_prompt(context_str, catalog, allowed))
            except Exception:
                reply = None
            calls = _parse_llm_calls(reply, allowed)
            if calls is not None:
                # parsed output wins — including an explicit empty plan
                return calls[:MAX_PLAN]
        return _plan_rules(context_str, allowed)[:MAX_PLAN]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _call_key(call: dict) -> tuple:
    try:
        return (str(call.get("tool")),
                json.dumps(call.get("args") or {}, sort_keys=True,
                           default=str))
    except Exception:
        return (str(call.get("tool")), str(call.get("args")))


def _ok_of(res: Any) -> bool:
    """A result counts as failed only when it says so itself."""
    if isinstance(res, dict):
        if res.get("ok") is False or res.get("error"):
            return False
    return True


def _call_tool(tool: Any, args: Any, tools_fn: Callable | None = None):
    """(ok, result) — try fn(name, args), then fn(name, **args)."""
    fn = tools_fn
    if not callable(fn):
        at = _agent_tools()
        fn = getattr(at, "call", None) if at is not None else None
    if not callable(fn):
        return False, {"error": f"no tool dispatcher for {tool!r}"}
    try:
        res = fn(tool, args if isinstance(args, dict) else {})
        return _ok_of(res), res
    except TypeError:
        pass
    except Exception as e:
        return False, {"error": f"{type(e).__name__}: {e}"}
    if isinstance(args, dict):
        try:
            res = fn(tool, **args)
            return _ok_of(res), res
        except Exception as e:
            return False, {"error": f"{type(e).__name__}: {e}"}
    return False, {"error": f"bad args for {tool!r}"}


def _jsonable(res: Any) -> Any:
    """Keep results JSON-safe; truncate pathological strings."""
    if isinstance(res, str) and len(res) > 1000:
        res = res[:1000] + "..."
    try:
        json.dumps(res)
        return res
    except Exception:
        return str(res)[:500]


# ---------------------------------------------------------------------------
# Result distillation — one-liners + RESULTS fold-back
# ---------------------------------------------------------------------------

def _res_line(res: Any, limit: int = 140) -> str:
    """A one-line digest of a tool result."""
    if isinstance(res, dict):
        hits = []
        for k in _FIND_KEYS:
            v = res.get(k)
            if v is None or v == "" or v == []:
                continue
            hits.append(f"{k}={str(v)[:40]}")
            if len(hits) >= 2:
                break
        if hits:
            return "; ".join(hits)[:limit]
        try:
            return json.dumps(res, default=str, sort_keys=True)[:limit]
        except Exception:
            return str(res)[:limit]
    if isinstance(res, (list, tuple)):
        try:
            return json.dumps(res, default=str)[:limit]
        except Exception:
            return str(res)[:limit]
    return " ".join(str(res).split())[:limit]


def _fold(ctx: str, transcript: list) -> str:
    """Append the latest round's results so the LLM plans from fresh info."""
    if not transcript:
        return ctx
    tail = transcript[-MAX_PLAN:]
    bits = [f"{e['step']}.{e['tool']}({'ok' if e.get('ok') else 'ERR'}) "
            f"{_res_line(e.get('result'), 80)}" for e in tail]
    return str(ctx or "") + "\nRESULTS: " + " | ".join(bits)


def _seed(context_str: Any, seed_pack: Any) -> tuple:
    """Fold seed_pack into the run: extra context text + pre-seeded calls.

    Accepted shapes (all optional, all defensive):
      str            -> appended as a 'SEED:' context line
      list           -> [{"tool","args","why"}] calls run before planning
      dict           -> {"context"|"ctx": str, "calls"|"plan": [dict]}
    """
    ctx = str(context_str or "")
    calls: list[dict] = []
    try:
        if isinstance(seed_pack, str):
            ctx += "\nSEED: " + seed_pack[:500]
        elif isinstance(seed_pack, dict):
            extra = seed_pack.get("context") or seed_pack.get("ctx")
            if isinstance(extra, str) and extra.strip():
                ctx += "\nSEED: " + extra[:500]
            calls = [c for c in (seed_pack.get("calls")
                                 or seed_pack.get("plan") or [])
                     if isinstance(c, dict)]
        elif isinstance(seed_pack, (list, tuple)):
            calls = [c for c in seed_pack if isinstance(c, dict)]
    except Exception:
        pass
    return ctx, calls


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_loop(context_str: Any, max_steps: int = DEFAULT_MAX_STEPS,
             llm_fn: Callable | None = None,
             budget_s: float = DEFAULT_BUDGET_S,
             seed_pack: Any = None,
             tools_fn: Callable | None = None) -> dict:
    """plan -> call -> append transcript, until empty plan / budget / cap.

    ``tools_fn`` — optional ``(name, args) -> result`` dispatcher; when
    absent, ``agent_tools.call`` is resolved lazily (missing module ->
    calls land as ok:False errors, the loop still runs).

    Returns {"transcript": [{"step","tool","args","ok","result"}],
             "steps": int, "budget_used_s": float,
             "stop_reason": "plan_empty"|"budget"|"step_cap"|"error"}.
    Never raises.
    """
    t0 = time.monotonic()
    run: dict = {"transcript": [], "steps": 0,
                 "budget_used_s": 0.0, "stop_reason": "step_cap"}
    transcript = run["transcript"]
    try:
        try:
            max_steps = max(0, int(max_steps))
        except (TypeError, ValueError):
            max_steps = DEFAULT_MAX_STEPS
        try:
            budget_s = float(budget_s)
        except (TypeError, ValueError):
            budget_s = DEFAULT_BUDGET_S
        ctx, pending_seed = _seed(context_str, seed_pack)
        catalog = _catalog()
        seen: set = set()
        stop = None

        while len(transcript) < max_steps:
            if pending_seed:
                calls, pending_seed = pending_seed, []
            else:
                calls = plan(ctx, catalog, llm_fn=llm_fn)
            calls = [c for c in (calls or []) if isinstance(c, dict)]
            calls = [c for c in calls if _call_key(c) not in seen]
            calls = calls[:MAX_PLAN]
            if not calls:
                stop = "plan_empty"
                break
            for c in calls:
                if len(transcript) >= max_steps:
                    stop = "step_cap"
                    break
                if time.monotonic() - t0 >= budget_s:
                    stop = "budget"
                    break
                ok, res = _call_tool(c.get("tool"), c.get("args"), tools_fn)
                seen.add(_call_key(c))
                transcript.append({
                    "step": len(transcript) + 1,
                    "tool": str(c.get("tool") or "?"),
                    "args": c.get("args") if isinstance(c.get("args"),
                                                       dict) else {},
                    "ok": bool(ok),
                    "result": _jsonable(res),
                })
            ctx = _fold(ctx, transcript)
            if stop:
                break
        run["stop_reason"] = stop or "step_cap"
    except Exception:
        run["stop_reason"] = "error"
    run["steps"] = len(transcript)
    run["budget_used_s"] = round(time.monotonic() - t0, 3)
    return run


# ---------------------------------------------------------------------------
# Evidence + findings for the adjudicator
# ---------------------------------------------------------------------------

def transcript_evidence(run: dict):
    """The run transcript as one Evidence card — provenance, not proof.

    Always INCONCLUSIVE @ 0.3: the transcript records *what the agent
    checked*, the adjudicator weighs it. ``data`` carries the full
    transcript plus the tool-call count. Falls back to a plain dict in
    the same shape when the schema can't be imported.
    """
    run = run if isinstance(run, dict) else {}
    transcript = run.get("transcript") or []
    n = len(transcript)
    ok_n = sum(1 for e in transcript
               if isinstance(e, dict) and e.get("ok"))
    stop = run.get("stop_reason") or "?"
    budget = run.get("budget_used_s", 0.0)
    try:
        budget = float(budget)
    except (TypeError, ValueError):
        budget = 0.0
    summary = (f"agent loop: {n} tool call(s), {ok_n} ok, "
               f"stop={stop}, {budget:.2f}s")
    data = {"transcript": transcript, "tool_calls": n,
            "ok_calls": ok_n, "steps": run.get("steps", n),
            "stop_reason": stop, "budget_used_s": budget}
    try:
        ClaimVerdict, Evidence = _schema()
        return Evidence(type="agent_transcript", summary=summary,
                        verdict=ClaimVerdict.INCONCLUSIVE,
                        confidence=_EVIDENCE_CONF, data=data)
    except Exception:
        return {"type": "agent_transcript", "summary": summary,
                "verdict": "inconclusive", "confidence": _EVIDENCE_CONF,
                "data": data}


def findings(run: dict) -> list[dict]:
    """Distill ok results into [{'tool', 'key_finding'}] one-liners."""
    out: list[dict] = []
    run = run if isinstance(run, dict) else {}
    for e in run.get("transcript") or []:
        try:
            if not isinstance(e, dict) or not e.get("ok"):
                continue
            tool = str(e.get("tool") or "?")
            args = e.get("args") or {}
            arg_s = ", ".join(f"{k}={v}" for k, v in
                              list(args.items())[:3]) \
                if isinstance(args, dict) else str(args)
            head = f"{arg_s} -> " if arg_s else ""
            out.append({"tool": tool,
                        "key_finding": f"{head}{_res_line(e.get('result'))}"})
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/agent_loop.py  (fully offline, stub llm+tools)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    fails = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        global fails
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label} {detail}")

    print("agent_loop self-test:")

    CTX = "\n".join([
        "CLAIM: filmed 2026-10-17T12:30 local at Husum, Germany "
        "(54.48,9.05) [DE]",
        "KNOWN: solar: supports (0.77) — sun 36.6 deg fits window",
        "SUSPECTED: DE 0.65 > FR 0.31 > NL 0.22",
        "OPEN: landmarks unverified — names won't geocode "
        "(Husum Schloss, Marktplatz); clock overlay unchecked — no "
        "consensus; country ranking weak for claimed DE",
        "TOOL-HINT: geocode, wiki_geo, weather, timezone_at, sun_at, "
        "pip2cc",
    ])
    EMPTY_CTX = ("CLAIM: none supplied\nKNOWN: none yet\n"
                 "SUSPECTED: none\nOPEN: none\nTOOL-HINT: geocode")

    def ok_tools(name, args):
        return {"ok": True, "tool": name, "echo": args,
                "summary": f"{name} done"}

    def boom_tools(name, args):
        raise RuntimeError("tool exploded")

    def slow_tools(name, args):
        time.sleep(0.05)
        return {"ok": True}

    # --- rule planner -------------------------------------------------------
    calls = plan(CTX, None)
    tools_used = [c["tool"] for c in calls]
    print("  planned:", tools_used)
    check("rule plan non-empty", len(calls) >= 4, f"(got {len(calls)})")
    check("rule plan <= 6", len(calls) <= MAX_PLAN)
    check("plan item shape",
          all(set(c) >= {"tool", "args", "why"} for c in calls))
    check("place -> geocode x2 names",
          sum(1 for c in calls if c["tool"] == "geocode") == 2 and
          any("Husum Schloss" in str(c["args"].get("query"))
              for c in calls))
    check("time -> weather+sun_at",
          "weather" in tools_used and "sun_at" in tools_used)
    check("cc -> pip2cc", "pip2cc" in tools_used)
    check("geocode args from OPEN names",
          any(c["args"].get("query") == "Marktplatz" for c in calls))

    sub = plan(CTX, ["geocode"])          # restricted catalog
    check("catalog restricts tools",
          bool(sub) and all(c["tool"] == "geocode" for c in sub))
    check("empty OPEN -> empty plan", plan(EMPTY_CTX, None) == [])
    check("plan(None) -> []", plan(None, None) == [])

    # --- llm planner ---------------------------------------------------------
    def stub_llm(prompt):
        assert "BRIEF:" in prompt and "TOOLS:" in prompt
        return ('[{"tool": "sun_at", "args": {"lat": 1, "lon": 2}, '
                '"why": "stub picks sun"}]')

    lcalls = plan(CTX, None, llm_fn=stub_llm)
    check("llm plan wins", len(lcalls) == 1 and
          lcalls[0]["tool"] == "sun_at" and
          lcalls[0]["args"] == {"lat": 1, "lon": 2})
    check("llm junk -> rule fallback",
          len(plan(CTX, None, llm_fn=lambda p: "not json at all")) >= 4)
    check("llm raising -> rule fallback",
          len(plan(CTX, None,
                   llm_fn=lambda p: 1 / 0)) >= 4)

    # --- run_loop: empty plan ------------------------------------------------
    r = run_loop(EMPTY_CTX, tools_fn=ok_tools)
    check("empty-plan stop", r["stop_reason"] == "plan_empty")
    check("empty-plan no calls", r["steps"] == 0 and
          r["transcript"] == [])

    # --- run_loop: step cap ---------------------------------------------------
    r = run_loop(CTX, max_steps=2, tools_fn=ok_tools)
    check("step-cap stop", r["stop_reason"] == "step_cap")
    check("step-cap count", r["steps"] == 2 and
          len(r["transcript"]) == 2)

    # --- run_loop: budget ------------------------------------------------------
    r = run_loop(CTX, max_steps=6, budget_s=0.01, tools_fn=slow_tools)
    check("budget stop", r["stop_reason"] == "budget")
    check("budget partial transcript", 0 < r["steps"] < 5)
    check("budget_used_s recorded", r["budget_used_s"] > 0)

    # --- run_loop: full pass, transcript shape ---------------------------------
    r = run_loop(CTX, tools_fn=ok_tools)
    check("full run stops plan_empty", r["stop_reason"] == "plan_empty",
          f"(got {r['stop_reason']})")
    check("full run step count", r["steps"] == len(calls),
          f"({r['steps']} vs plan {len(calls)})")
    check("transcript entry shape",
          all(set(e) == {"step", "tool", "args", "ok", "result"}
              for e in r["transcript"]))
    check("transcript steps sequential",
          [e["step"] for e in r["transcript"]] ==
          list(range(1, r["steps"] + 1)))
    check("transcript all ok",
          all(e["ok"] for e in r["transcript"]))
    check("run dict keys",
          set(r) == {"transcript", "steps", "budget_used_s",
                     "stop_reason"})

    # --- failing + absent tools never raise ------------------------------------
    r = run_loop(CTX, max_steps=3, tools_fn=boom_tools)
    check("tool exception -> ok False",
          r["transcript"] and not r["transcript"][0]["ok"] and
          "error" in r["transcript"][0]["result"])
    r = run_loop(CTX, max_steps=2)   # agent_tools sibling absent -> degrade
    check("missing agent_tools degrades",
          r["steps"] == 2 and
          all(not e["ok"] for e in r["transcript"]))

    # --- seed_pack -------------------------------------------------------------
    r = run_loop(EMPTY_CTX, tools_fn=ok_tools,
                 seed_pack={"calls": [{"tool": "geocode",
                                       "args": {"query": "Seedtown"},
                                       "why": "seed"}]})
    check("seed call runs first",
          r["steps"] >= 1 and r["transcript"][0]["tool"] == "geocode" and
          r["transcript"][0]["args"]["query"] == "Seedtown")

    # --- llm-driven loop --------------------------------------------------------
    replies = iter([
        '[{"tool": "geocode", "args": {"query": "A"}, "why": "s1"}]',
        '[{"tool": "wiki_geo", "args": {"lat": 1, "lon": 2}, "why": "s2"}]',
        "[]",
    ])
    r = run_loop(CTX, tools_fn=ok_tools, llm_fn=lambda p: next(replies))
    check("llm loop runs both rounds then empty",
          r["steps"] == 2 and r["stop_reason"] == "plan_empty" and
          [e["tool"] for e in r["transcript"]] ==
          ["geocode", "wiki_geo"])

    # --- evidence card ----------------------------------------------------------
    r = run_loop(CTX, tools_fn=ok_tools)
    ev = transcript_evidence(r)
    ev_type = ev.get("type") if isinstance(ev, dict) else getattr(
        ev, "type", None)
    ev_verdict = ev.get("verdict") if isinstance(ev, dict) else getattr(
        ev, "verdict", None)
    ev_conf = ev.get("confidence") if isinstance(ev, dict) else getattr(
        ev, "confidence", None)
    ev_data = ev.get("data") if isinstance(ev, dict) else getattr(
        ev, "data", {})
    check("evidence type", ev_type == "agent_transcript")
    check("evidence inconclusive",
          "inconclusive" in str(getattr(ev_verdict, "value",
                                      ev_verdict)).lower())
    check("evidence confidence 0.3", abs(float(ev_conf or 0) - 0.3) < 1e-9)
    check("evidence carries transcript",
          isinstance(ev_data, dict) and
          ev_data.get("transcript") == r["transcript"] and
          ev_data.get("tool_calls") == r["steps"])
    check("evidence never raises on junk",
          isinstance(transcript_evidence(None), (dict,)) or
          getattr(transcript_evidence({}), "type", None) ==
          "agent_transcript")

    # --- findings -----------------------------------------------------------------
    fs = findings(r)
    check("findings count == ok calls", len(fs) == r["steps"])
    check("findings shape",
          all(set(f) == {"tool", "key_finding"} and
              isinstance(f["key_finding"], str) and f["key_finding"]
              for f in fs))
    print("  findings[0]:", fs[0] if fs else None)
    check("findings empty on junk", findings(None) == [])

    # --- never-raise sweep ----------------------------------------------------------
    check("run_loop(None)", run_loop(None)["stop_reason"] in
          ("plan_empty", "budget", "step_cap", "error"))
    check("run_loop bad kwargs",
          run_loop(CTX, max_steps="x", budget_s="y",
                   tools_fn=ok_tools)["steps"] >= 0)

    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        sys.exit(1)
    print("OK — agent_loop self-test passed")
