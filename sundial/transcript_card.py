"""Agent investigation transcript — report fragment + evidence card.

The agent loop (``sundial.agent_loop``, written separately) investigates
the claim like an OSINT researcher: it geocodes the claimed place, pulls
wiki context, checks weather/timezone/sun for the claimed moment, and
stops when the gaps are closed. It returns a run dict::

    run = {
        "transcript": [
            {"step": 1, "tool": "geocode",
             "args": "Husum, Germany",
             "ok": True,
             "result": {"lat": 54.47, "lon": 9.05},
             "ms": 412},
            ...
        ],
        "steps": int,           # tool calls attempted
        "stop_reason": str,     # why the loop ended
    }

This module renders that run two ways, both aimed at the demo's "the
agent thought like a researcher" beat:

- ``transcript_html(run)`` — escaped HTML fragment, one numbered row per
  tool call: ``1. ✓ geocode("Husum, Germany") → 54.47°N 9.05°E`` —
  ok steps in report-grey, failures red, capped at ``max_rows`` with a
  trailing "... N more calls" row. Bare fragment (no card chrome) so the
  report can splice it the way it splices ``osint_chain.chain_html``.
- ``card(run)`` — a narration-only Evidence card (type
  ``agent_transcript``, always INCONCLUSIVE 0.3): the transcript
  *explains* the investigation, it is not a verdict itself. Same
  convention as ``osint_chain.chain_evidence``.

Everything is defensive: the sibling module is a moving target, so
transcript entries are read attr-or-key, malformed steps are skipped,
and every public function degrades to ""/[]/a thin card instead of
raising. Stdlib only (html + json); schema types import lazily so the
``__main__`` self-test runs as ``python3 sundial/transcript_card.py``.
"""
from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotations only — the real import is lazy (see _schema)
    from .schema import Evidence

# ---------------------------------------------------------------------------
# Style tokens + caps (report palette: near-black fragments, slate text)
# ---------------------------------------------------------------------------

_MAX_ROWS = 14          # default html fragment cap, same as chain_html
_FINDING_CHARS = 80     # per-step distilled-result budget
_ARGS_CHARS = 52        # rendered call-args budget

_C_OK = "#9a9a9a"       # successful call — muted report grey
_C_FAIL = "#f87171"     # failed call — report red (same as chain_html elim)
_C_DIM = "#5a5a5a"      # row furniture: numbers, arrows, ms

_MONO = (
    "font:11px/1.55 ui-monospace,'SF Mono',Menlo,Consolas,monospace;"
    "margin:6px 0;color:#9a9a9a"
)
_ROW = "margin:1px 0;padding-left:2.6em;text-indent:-2.6em;color:{color}"

#: Result-dict keys scanned for a headline phrase, in priority order.
#: The name-ish group may pair with a summary-ish group ("Husum — town
#: in Schleswig-Holstein"); both lists read defensively — a tool whose
#: result uses none of them falls back to a compact k=v digest.
_NAME_KEYS = (
    "name", "title", "display_name", "place", "label",
    "timezone", "tz", "iana", "country",
)
_SUM_KEYS = (
    "summary", "extract", "description", "answer", "text",
    "error", "message", "detail", "status", "value", "verdict",
)


# ---------------------------------------------------------------------------
# Small helpers — every reader is defensive; nothing here raises.
# ---------------------------------------------------------------------------

def _schema():
    """Lazy schema import — tolerates both package and direct-script use."""
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # `python3 sundial/transcript_card.py` direct run
        from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


def _get(item: Any, key: str, default=None):
    """Attr-or-key read — transcript entries may be dicts or objects."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _e(value: Any) -> str:
    """Escape for element-text context — &<> only.

    The fragment is spliced as element content (never into an attribute,
    same as osint_chain.chain_html), so quotes are left literal: that
    keeps the rendered source reading 'tool("arg")' exactly like the
    call shape the spec specifies.
    """
    return html.escape("" if value is None else str(value), quote=False)


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _clip(s: Any, n: int = _FINDING_CHARS) -> str:
    """One-line, length-capped text; '' for None/blank."""
    s = " ".join(str(s if s is not None else "").split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _scalar(v: Any) -> str:
    """Compact scalar: floats via %g, bools lowercase, else str."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _fmt_ll(lat: Any, lon: Any) -> str:
    """'54.47°N 9.05°E' — '' unless both parse as floats."""
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return ""
    ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
    return f"{abs(la):.2f}°{ns} {abs(lo):.2f}°{ew}"


def _fmt_ms(ms: Any) -> str:
    """'412ms' for numbers; clipped str for the rest; '' when absent."""
    if ms is None:
        return ""
    if isinstance(ms, (int, float)) and not isinstance(ms, bool):
        return f"{ms:.0f}ms"
    s = str(ms).strip()
    return _clip(s, 12)


def _distill(value: Any, depth: int = 0) -> str:
    """Distill one tool result into a <=80-char phrase.

    dict -> '54.47°N 9.05°E' when lat/lon exist, else the first
    name-ish key (optionally ' — ' + a summary-ish key), else a compact
    k=v digest of scalars, else a clipped json dump. list/tuple -> first
    item, recursively. str/scalar -> clipped. None/empty -> ''.
    """
    if depth > 4 or value is None:
        return ""
    if isinstance(value, dict):
        if "result" in value and ("ok" in value or "tool" in value or "ms" in value):
            inner = _distill(value.get("result"), depth + 1)
            if inner:
                return inner
            if value.get("error"):
                return _clip(value.get("error"))
        ll = _fmt_ll(
            value.get("lat", value.get("latitude")),
            value.get("lon", value.get("lng", value.get("longitude"))),
        )
        if ll:
            return ll
        name = summ = ""
        for k in _NAME_KEYS:
            if _clip(value.get(k)):
                name = _clip(value.get(k), 48)
                break
        for k in _SUM_KEYS:
            if _clip(value.get(k)):
                summ = _clip(value.get(k))
                break
        if name and summ and summ.lower() != name.lower():
            return _clip(f"{name} — {summ}")
        if name or summ:
            return name or summ
        pairs = []
        for k, v in value.items():
            if isinstance(v, (str, int, float, bool)) and _clip(v):
                pairs.append(f"{k}={_scalar(v)}")
            if len(pairs) >= 3:
                break
        if pairs:
            return _clip(", ".join(pairs))
        try:
            return _clip(json.dumps(value, default=str, ensure_ascii=False))
        except (TypeError, ValueError):
            return ""
    if isinstance(value, (list, tuple)):
        return _distill(value[0], depth + 1) if value else ""
    return _clip(value)


def _as_step(s: Any) -> dict:
    """Normalize one transcript entry to a plain dict ({} if hopeless)."""
    if isinstance(s, dict):
        return dict(s)
    d = {k: _get(s, k) for k in ("step", "tool", "args", "result", "ms")}
    d["ok"] = _get(s, "ok", True)
    if all(v is None for v in d.values() if v is not True):
        return {}
    return d


def _transcript(run: Any) -> list[dict]:
    """run['transcript'] as a list of step dicts; [] on anything odd."""
    raw = _get(run, "transcript", [])
    if not isinstance(raw, (list, tuple)):
        return []
    out = []
    for s in raw:
        try:
            d = _as_step(s)
        except Exception:
            d = {}
        if d:
            out.append(d)
    return out


def _fmt_args(args: Any) -> str:
    """Render call args as they'd appear inside the parens.

    str -> '"Husum, Germany"' · dict -> 'lat=54.47, lon=9.05' (str values
    quoted, <=3 pairs + …) · list/tuple -> item digest · scalar -> str.
    """
    if args is None:
        return ""
    if isinstance(args, str):
        return _clip(f'"{args}"', _ARGS_CHARS)
    if isinstance(args, dict):
        parts = []
        items = list(args.items())
        for k, v in items[:3]:
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            elif isinstance(v, (int, float, bool)) or v is None:
                parts.append(f"{k}={_scalar(v)}")
            else:
                try:
                    parts.append(f"{k}={json.dumps(v, default=str)}")
                except (TypeError, ValueError):
                    parts.append(f"{k}=...")
        if len(items) > 3:
            parts.append("…")
        return _clip(", ".join(parts), _ARGS_CHARS)
    if isinstance(args, (list, tuple)):
        parts = [f'"{a}"' if isinstance(a, str) else _scalar(a) for a in args[:4]]
        if len(args) > 4:
            parts.append("…")
        return _clip(", ".join(parts), _ARGS_CHARS)
    return _clip(args, _ARGS_CHARS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def step_finding(step: Any) -> str:
    """Distill one transcript step's result into a <=80-char phrase.

    Reads ``step['result']`` (attr-or-key tolerant): dicts resolve to
    coords / a headline key / a k=v digest, lists to their first item,
    strings clip, and a bare ``step['error']`` backs up an empty result.
    Returns '' when there is nothing worth showing. Never raises.
    """
    try:
        s = _distill(_get(step, "result"))
        if not s:
            s = _clip(_get(step, "error"))
        return s
    except Exception:
        return ""


def tools_used(run: Any) -> list[str]:
    """Ordered unique tool names the agent called. [] on anything odd."""
    try:
        out: list[str] = []
        for s in _transcript(run):
            t = str(_get(s, "tool") or "").strip()
            if t and t not in out:
                out.append(t)
        return out
    except Exception:
        return []


def transcript_html(run: Any, max_rows: int = _MAX_ROWS) -> str:
    """Escaped HTML fragment of the agent's tool-call transcript.

    One numbered row per call — ``1. ✓ geocode("Husum, Germany") →
    54.47°N 9.05°E ·412ms`` — ok rows in #9a9a9a, failed rows #f87171
    with a ✗, row furniture dimmed. Long transcripts truncate at
    ``max_rows`` with a closing "... N more calls" row. Returns '' for
    an empty/missing transcript so callers can splice unconditionally.
    Never raises.
    """
    try:
        steps = _transcript(run)
        if not steps:
            return ""
        mr = max(0, _int(max_rows, _MAX_ROWS))
        shown, extra = steps[:mr] if mr else [], len(steps) - min(len(steps), mr)

        rows: list[str] = []
        for i, s in enumerate(shown):
            ok = bool(_get(s, "ok", True))
            color = _C_OK if ok else _C_FAIL
            n = _get(s, "step")
            n = n if isinstance(n, int) and not isinstance(n, bool) else i + 1
            tool = str(_get(s, "tool") or "?").strip() or "?"
            call = f"{tool}({_fmt_args(_get(s, 'args'))})"
            row = (
                f'<div style="{_ROW.format(color=color)}">'
                f'<span style="color:{_C_DIM}">{n}.</span> '
                f'{"✓" if ok else "✗"} {_e(call)}'
            )
            finding = step_finding(s)
            if finding:
                row += f' <span style="color:{_C_DIM}">→</span> {_e(finding)}'
            ms = _fmt_ms(_get(s, "ms"))
            if ms:
                row += f' <span style="color:{_C_DIM}">·{_e(ms)}</span>'
            rows.append(row + "</div>")
        if extra > 0:
            rows.append(
                f'<div style="{_ROW.format(color=_C_DIM)}">'
                f"... {extra} more call{'s' if extra != 1 else ''}</div>"
            )
        return '<div class="agent-transcript" style="' + _MONO + '">' + "".join(rows) + "</div>"
    except Exception:
        return ""


def card(run: Any) -> "Evidence | None":
    """Pack the run as one Evidence card (type='agent_transcript').

    Always INCONCLUSIVE 0.3 — the transcript narrates how the agent
    investigated, it does not itself support or contradict the claim.
    ``data`` carries the normalized transcript, the ordered tool list,
    and the stop reason. Returns None only if the schema itself cannot
    be imported (package-less environment) — otherwise never raises.
    """
    try:
        ClaimVerdict, Evidence = _schema()
    except Exception:
        return None
    try:
        steps = _transcript(run)
        n_calls = len(steps) or max(0, _int(_get(run, "steps"), 0))
        tools = tools_used(run)
        reason = str(_get(run, "stop_reason") or "").strip() or "unknown"
        summary = (
            f"agent ran {n_calls} tool call{'s' if n_calls != 1 else ''} "
            f"across {len(tools)} tool{'s' if len(tools) != 1 else ''}, "
            f"stopped: {reason}"
        )
        return Evidence(
            type="agent_transcript",
            summary=summary,
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.3,
            data={
                "transcript": steps,
                "tools_used": tools,
                "stop_reason": reason,
            },
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/transcript_card.py  (or -m sundial.transcript_card)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    try:
        from .schema import ClaimVerdict
    except ImportError:  # direct-script run — module dir is on sys.path
        import os

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from schema import ClaimVerdict

    fails = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        global fails
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label} {detail}")

    print("transcript_card self-test:")

    run = {
        "transcript": [
            {"step": 1, "tool": "geocode", "args": "Husum, Germany",
             "ok": True, "ms": 412,
             "result": {"lat": 54.4754, "lon": 9.0527,
                        "display_name": "Husum, Schleswig-Holstein"}},
            {"step": 2, "tool": "wiki_geo",
             "args": {"lat": 54.47, "lon": 9.05},
             "ok": True, "ms": 830,
             "result": [{"title": "Husum",
                         "summary": "town in Nordfriesland, Germany"}]},
            {"step": 3, "tool": "weather",
             "args": {"lat": 54.47, "lon": 9.05, "when": "2026-10-17T12:30"},
             "ok": False, "ms": 8012,
             "result": {"error": "timeout after 8s"}},
            {"step": 4, "tool": "context",
             "args": "<script>alert(1)</script>",
             "ok": True, "ms": 12, "result": "no open gaps"},
        ],
        "steps": 4,
        "stop_reason": "claim resolved",
    }

    # --- step_finding: distillation shapes -----------------------------------
    check("lat/lon dict -> coords",
          step_finding(run["transcript"][0]) == "54.48°N 9.05°E")
    check("list -> first item title+summary",
          step_finding(run["transcript"][1]) == "Husum — town in Nordfriesland, Germany")
    check("error dict -> error text",
          step_finding(run["transcript"][2]) == "timeout after 8s")
    check("str result -> str",
          step_finding(run["transcript"][3]) == "no open gaps")
    check("scalar dict -> k=v digest",
          step_finding({"result": {"elevation_deg": 36.6, "azimuth_deg": 141.2}})
          == "elevation_deg=36.6, azimuth_deg=141.2")
    check("long str clips to <=80",
          len(step_finding({"result": "x" * 200})) <= 80)
    check("garbage step -> ''",
          step_finding(None) == "" and step_finding({"result": []}) == "")
    check("step['error'] backs up empty result",
          step_finding({"ok": False, "error": "dns failed"}) == "dns failed")

    # --- tools_used ----------------------------------------------------------
    check("tools_used ordered + deduped",
          tools_used(run) == ["geocode", "wiki_geo", "weather", "context"])
    check("tools_used(empty) -> []",
          tools_used({}) == [] and tools_used(None) == [])

    # --- transcript_html -----------------------------------------------------
    frag = transcript_html(run)
    print("\n" + frag + "\n")
    check("numbered row present", ">1.</span>" in frag)
    check("call line shape",
          'geocode("Husum, Germany")' in frag)
    check("ok marker + fail marker", "✓" in frag and "✗" in frag)
    check("coords arrow text", "→" in frag and "54.48°N 9.05°E" in frag)
    check("dict args rendered", 'lat=54.47, lon=9.05' in frag)
    check("fail row red", 'color:#f87171' in frag)
    check("ok row grey", 'color:#9a9a9a' in frag)
    check("ms shown", "8012ms" in frag)
    check("<script> in args escaped",
          "<script>alert" not in frag and "&lt;script&gt;" in frag)
    check("mono 11px inline", "11px" in frag and "monospace" in frag)

    long_run = {"transcript": [
        {"step": i, "tool": f"t{i}", "args": None, "ok": True,
         "result": "x", "ms": 1} for i in range(1, 21)],
        "stop_reason": "budget"}
    frag2 = transcript_html(long_run)
    check(">max_rows truncates", ">14.</span>" in frag2 and ">15.</span>" not in frag2)
    check("elision row", "... 6 more calls" in frag2)
    check("empty run -> ''", transcript_html({}) == "" and transcript_html(None) == "")
    check("weird run -> ''", transcript_html({"transcript": "nope"}) == "")

    # --- card ----------------------------------------------------------------
    ev = card(run)
    check("card type", ev is not None and ev.type == "agent_transcript")
    check("card verdict INCONCLUSIVE",
          ev is not None and ev.verdict is ClaimVerdict.INCONCLUSIVE)
    check("card confidence 0.3", ev is not None and ev.confidence == 0.3)
    check("card summary",
          ev is not None and ev.summary ==
          "agent ran 4 tool calls across 4 tools, stopped: claim resolved",
          f"(got {getattr(ev, 'summary', None)!r})")
    check("card data keys",
          ev is not None and set(ev.data) ==
          {"transcript", "tools_used", "stop_reason"})
    check("card data content",
          ev is not None and len(ev.data["transcript"]) == 4
          and ev.data["tools_used"] == ["geocode", "wiki_geo", "weather", "context"]
          and ev.data["stop_reason"] == "claim resolved")
    json.dumps(ev.data, default=str)  # data stays sidecar-friendly

    empty = card({})
    check("empty run card", empty is not None and empty.summary ==
          "agent ran 0 tool calls across 0 tools, stopped: unknown")
    steps_only = card({"steps": 3, "stop_reason": "budget"})
    check("steps fallback count", steps_only is not None
          and "agent ran 3 tool calls across 0 tools" in steps_only.summary)
    check("card(None) does not raise", card(None) is not None)

    print()
    if fails:
        print(f"{fails} check(s) FAILED")
        sys.exit(1)
    print("OK — transcript_card self-test passed")
