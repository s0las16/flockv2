"""Per-run audit trail — which stages ran, how long, what was cached.

Pipeline stages record through the module-level singleton so their
signatures don't change::

    from . import run_stats
    with run_stats.current().mark("weather"):
        ...
    run_stats.current().count("cache_hits")

At the end of a run, ``current().snapshot()`` produces a JSON-friendly
dict; ``stats_html()`` renders it as a compact dark-theme fragment for
the verdict report and ``to_sidecar()`` shapes it for the verdict.json
sidecar. Both tolerate empty/partial snapshots so callers can splice
unconditionally.

stdlib only: contextlib / html / time (+ json in the __main__ demo).
"""
from __future__ import annotations

import html
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Conventional counter keys — free-form is fine, these are just what the
# pipeline records: VLM round-trips, VLM cache hits, external HTTP calls
# (geocode / weather), and frames pulled from the clip.
_COUNTER_KEYS = ("vlm_calls", "cache_hits", "http_calls", "frames")

# report.py palette: near-black card, slate text, hairline rgba borders.
_RS_CARD = (
    "font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;"
    "background:rgba(15,23,42,.72);border:1px solid rgba(148,163,184,.14);"
    "border-radius:12px;padding:14px 16px;color:#cbd5e1"
)
_RS_HEAD = (
    "display:flex;justify-content:space-between;align-items:baseline;"
    "font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:#64748b;"
    "border-bottom:1px solid rgba(148,163,184,.12);padding-bottom:7px;margin-bottom:8px"
)
_RS_GRID = "display:grid;grid-template-columns:minmax(0,1fr) auto;column-gap:18px"
_RS_SECTION = (
    "grid-column:1/-1;color:#475569;font-size:10px;letter-spacing:.18em;"
    "text-transform:uppercase;margin:7px 0 2px"
)
_RS_LABEL = (
    "color:#94a3b8;font-size:12px;padding:2px 0;white-space:nowrap;"
    "overflow:hidden;text-overflow:ellipsis"
)
_RS_VAL = (
    "color:#e2e8f0;font-size:12px;padding:2px 0;text-align:right;"
    "font-variant-numeric:tabular-nums"
)
_RS_NOTE = (
    "color:#94a3b8;font-size:11px;padding:3px 0;"
    "border-top:1px dashed rgba(148,163,184,.10)"
)


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fmt_s(secs: Any) -> str:
    """Compact duration: 0.012s / 1.400s / 83.2s; '—' on garbage."""
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return "\u2014"
    return f"{s:.3f}s" if s < 10 else f"{s:.1f}s"


class RunStats:
    """Accumulates stage timings, counters and notes for one pipeline run.

    ``mark(name)`` times a ``with`` block via ``time.monotonic``; repeated
    marks of the same stage accumulate seconds and bump the call count.
    ``count(key, n)`` bumps named counters — conventionally ``vlm_calls``,
    ``cache_hits``, ``http_calls``, ``frames`` (any string works).
    ``notes`` is a plain list stages append one-line observations to.
    """

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._stages: dict[str, dict[str, float | int]] = {}
        self._counters: dict[str, int] = {}
        self.notes: list[str] = []

    @contextmanager
    def mark(self, stage: str) -> Iterator[None]:
        """Time the ``with`` block as pipeline stage ``stage``.

        The elapsed seconds land in the entry even if the block raises —
        a failed stage is still audit-worthy.
        """
        entry = self._stages.setdefault(str(stage), {"s": 0.0, "calls": 0})
        entry["calls"] = int(entry["calls"]) + 1
        t0 = time.monotonic()
        try:
            yield
        finally:
            entry["s"] = float(entry["s"]) + (time.monotonic() - t0)

    def count(self, key: str, n: int = 1) -> None:
        """Bump counter ``key`` by ``n`` (see ``_COUNTER_KEYS`` for conventions)."""
        key = str(key)
        self._counters[key] = self._counters.get(key, 0) + int(n)

    def snapshot(self) -> dict:
        """JSON-friendly summary: stage timings, counters, notes, run total."""
        return {
            "stages": {
                name: {"s": round(float(d["s"]), 3), "calls": int(d["calls"])}
                for name, d in self._stages.items()
            },
            "counters": dict(self._counters),
            "notes": list(self.notes),
            "total_s": round(time.monotonic() - self._t0, 3),
        }


def stats_html(snapshot: dict) -> str:
    """Render a snapshot as a compact dark-theme HTML fragment.

    Two-column grid (label | value) listing stage timings — with a ×N
    suffix when a stage ran more than once — then the counters; notes
    trail underneath. Returns "" when there is nothing to show so the
    report can splice the fragment unconditionally.
    """
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    stages = snapshot.get("stages") or {}
    counters = snapshot.get("counters") or {}
    notes = snapshot.get("notes") or []
    if not stages and not counters and not notes:
        return ""
    total = _fmt_s(snapshot.get("total_s"))

    rows: list[str] = []
    if stages:
        rows.append(f'<div style="{_RS_SECTION}">stages</div>')
        for name, d in stages.items():
            d = d if isinstance(d, dict) else {}
            try:
                calls = int(d.get("calls") or 0)
            except (TypeError, ValueError):
                calls = 0
            suffix = (
                f' <span style="color:#475569">×{calls}</span>' if calls > 1 else ""
            )
            rows.append(
                f'<div style="{_RS_LABEL}">{_e(name)}</div>'
                f'<div style="{_RS_VAL}">{_e(_fmt_s(d.get("s")))}{suffix}</div>'
            )
    if counters:
        rows.append(f'<div style="{_RS_SECTION}">counters</div>')
        for key, val in counters.items():
            rows.append(
                f'<div style="{_RS_LABEL}">{_e(key)}</div>'
                f'<div style="{_RS_VAL}">{_e(val)}</div>'
            )

    notes_html = ""
    if notes:
        items = "".join(
            f'<div style="{_RS_NOTE}">· {_e(n)}</div>' for n in notes
        )
        notes_html = f'<div style="margin-top:9px">{items}</div>'

    return (
        f'<div class="run-stats" style="{_RS_CARD}">'
        f'<div style="{_RS_HEAD}"><span>run audit</span>'
        f'<span style="color:#94a3b8">{_e(total)}</span></div>'
        f'<div style="{_RS_GRID}">{"".join(rows)}</div>'
        f"{notes_html}</div>"
    )


def to_sidecar(snapshot: dict) -> dict:
    """Shape a snapshot for the verdict.json sidecar — same keys, strict types.

    Coerces every value (float stage seconds, int calls/counters, str
    notes) so ``json.dumps`` never trips on an odd type a stage recorded.
    """
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    stages: dict[str, dict] = {}
    for name, d in (snapshot.get("stages") or {}).items():
        d = d if isinstance(d, dict) else {}
        try:
            s = round(float(d.get("s") or 0.0), 3)
        except (TypeError, ValueError):
            s = 0.0
        try:
            calls = int(d.get("calls") or 0)
        except (TypeError, ValueError):
            calls = 0
        stages[str(name)] = {"s": s, "calls": calls}
    counters: dict[str, Any] = {}
    for key, val in (snapshot.get("counters") or {}).items():
        try:
            counters[str(key)] = int(val)
        except (TypeError, ValueError):
            counters[str(key)] = str(val)
    try:
        total_s = round(float(snapshot.get("total_s") or 0.0), 3)
    except (TypeError, ValueError):
        total_s = 0.0
    return {
        "stages": stages,
        "counters": counters,
        "notes": [str(n) for n in (snapshot.get("notes") or [])],
        "total_s": total_s,
    }


_CURRENT = RunStats()


def current() -> RunStats:
    """Process-wide stats sink — pipeline stages record without signature changes."""
    return _CURRENT


def reset() -> RunStats:
    """Start a fresh audit trail (batch runs process many clips per process)."""
    global _CURRENT
    _CURRENT = RunStats()
    return _CURRENT


if __name__ == "__main__":
    import json

    rs = reset()

    # Two fake stages: one "fast I/O", one "VLM pass" — each records its
    # own counters as a real stage would.
    with rs.mark("extract_frames"):
        time.sleep(0.05)
        rs.count("frames", 12)
    with rs.mark("vlm_shadow_pass"):
        time.sleep(0.03)
        rs.count("vlm_calls")
        rs.count("cache_hits", 3)
    rs.notes.append("geocoded 'Berlin' -> 52.52, 13.40")
    rs.notes.append("weather stage skipped: no claimed place")

    snap = rs.snapshot()
    print("snapshot:")
    print(json.dumps(snap, indent=2))

    assert set(snap["stages"]) == {"extract_frames", "vlm_shadow_pass"}
    assert all(v["calls"] == 1 and v["s"] > 0 for v in snap["stages"].values())
    assert snap["counters"] == {"frames": 12, "vlm_calls": 1, "cache_hits": 3}
    assert snap["total_s"] >= 0.08

    frag = stats_html(snap)
    assert "extract_frames" in frag and "cache_hits" in frag and "run audit" in frag
    assert stats_html({}) == "" and stats_html({"stages": {}, "counters": {}}) == ""
    json.dumps(to_sidecar(snap))  # must be JSON-able

    print("\nhtml fragment:")
    print(frag)
    print("\nsidecar:")
    print(json.dumps(to_sidecar(snap), indent=2))
