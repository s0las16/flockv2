"""Self-contained HTML verdict report — inline CSS, zero external assets."""
from __future__ import annotations

import html
import os
from typing import Any

from .schema import Adjudication, ClaimVerdict, Verdict

_BADGE = {
    Verdict.SUPPORTED: ("#16a34a", "#dcfce7"),
    Verdict.CONTRADICTED: ("#dc2626", "#fee2e2"),
    Verdict.INSUFFICIENT: ("#6b7280", "#e5e7eb"),
}
_PILL = {
    ClaimVerdict.SUPPORTS: ("#166534", "#dcfce7"),
    ClaimVerdict.CONTRADICTS: ("#991b1b", "#fee2e2"),
    ClaimVerdict.INCONCLUSIVE: ("#374151", "#e5e7eb"),
}


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _verdict_str(v: Any) -> str:
    return getattr(v, "value", str(v))


def _pill(verdict: Any) -> str:
    try:
        cv = verdict if isinstance(verdict, ClaimVerdict) else ClaimVerdict(str(verdict).lower())
    except ValueError:
        cv = ClaimVerdict.INCONCLUSIVE
    fg, bg = _PILL[cv]
    return (
        f'<span class="pill" style="color:{fg};background:{bg};'
        f'border:1px solid {fg}55">{_e(_verdict_str(verdict))}</span>'
    )


def _conf_bar(conf: float) -> str:
    try:
        pct = max(0.0, min(1.0, float(conf))) * 100
    except (TypeError, ValueError):
        pct = 0.0
    return (
        f'<div class="confbar"><div class="conffill" style="width:{pct:.0f}%"></div></div>'
        f'<span class="confnum">{pct:.0f}%</span>'
    )


def render_html(adj: Adjudication, out_path: str) -> str:
    """Write a self-contained verdict card HTML file; returns out_path."""
    pack = adj.pack
    claim = pack.claim
    verdict = adj.verdict if isinstance(adj.verdict, Verdict) else Verdict(str(adj.verdict).upper())
    fg, bg = _BADGE[verdict]
    try:
        conf_pct = max(0.0, min(1.0, float(adj.confidence))) * 100
    except (TypeError, ValueError):
        conf_pct = 0.0

    claim_bits = []
    if claim.claimed_time:
        claim_bits.append(f"time: <b>{_e(claim.claimed_time)}</b>")
    if claim.claimed_place:
        claim_bits.append(f"place: <b>{_e(claim.claimed_place)}</b>")
    if claim.raw:
        claim_bits.append(f"&ldquo;{_e(claim.raw)}&rdquo;")
    claim_html = " &middot; ".join(claim_bits) or "<i>no claim recorded</i>"

    rows = []
    for ev in pack.evidence:
        frames = ""
        if ev.frames:
            names = ", ".join(os.path.basename(str(f)) for f in ev.frames[:4])
            frames = f'<div class="frames">cited: {_e(names)}</div>'
        rows.append(
            "<tr>"
            f'<td class="etype">{_e(ev.type)}</td>'
            f"<td>{_e(ev.summary)}{frames}</td>"
            f"<td>{_pill(ev.verdict)}</td>"
            f"<td>{_conf_bar(ev.confidence)}</td>"
            "</tr>"
        )
    evidence_html = (
        '<table><thead><tr><th>Type</th><th>Finding</th><th>Verdict</th>'
        "<th>Confidence</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        if rows
        else '<p class="empty">No evidence was collected.</p>'
    )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sundial verdict — {_e(verdict.value)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background:#0f172a; color:#111827; margin:0; padding:32px 16px; }}
  .card {{ max-width:760px; margin:0 auto; background:#ffffff; border-radius:16px;
          padding:32px 36px; box-shadow:0 12px 40px rgba(0,0,0,.35); }}
  .badge {{ display:inline-block; font-size:28px; font-weight:800; letter-spacing:.06em;
           color:{fg}; background:{bg}; border:2px solid {fg};
           border-radius:12px; padding:10px 22px; }}
  .conf {{ color:#6b7280; font-size:14px; margin-top:8px; }}
  h1 {{ font-size:15px; text-transform:uppercase; letter-spacing:.12em; color:#6b7280;
       margin:28px 0 8px; }}
  .claim {{ font-size:16px; margin:0 0 4px; }}
  .clip {{ color:#9ca3af; font-size:12px; word-break:break-all; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th {{ text-align:left; color:#6b7280; font-size:11px; text-transform:uppercase;
       letter-spacing:.08em; border-bottom:2px solid #e5e7eb; padding:6px 8px; }}
  td {{ border-bottom:1px solid #f3f4f6; padding:10px 8px; vertical-align:top; }}
  .etype {{ font-weight:700; white-space:nowrap; }}
  .pill {{ display:inline-block; border-radius:999px; padding:2px 10px;
          font-size:12px; font-weight:700; }}
  .confbar {{ display:inline-block; width:70px; height:8px; background:#e5e7eb;
             border-radius:4px; overflow:hidden; vertical-align:middle; margin-right:6px; }}
  .conffill {{ height:100%; background:#3b82f6; }}
  .confnum {{ font-size:12px; color:#6b7280; }}
  .frames {{ color:#9ca3af; font-size:11px; margin-top:4px; }}
  .narrative {{ font-size:15px; line-height:1.55; background:#f8fafc;
               border-left:4px solid {fg}; padding:12px 16px; border-radius:0 8px 8px 0; }}
  .empty {{ color:#9ca3af; }}
  footer {{ text-align:center; color:#64748b; font-size:12px; margin-top:20px; }}
</style></head>
<body>
  <div class="card">
    <div class="badge">{_e(verdict.value)}</div>
    <div class="conf">adjudicator confidence {conf_pct:.0f}%</div>
    <h1>Claim</h1>
    <p class="claim">{claim_html}</p>
    <p class="clip">clip: {_e(pack.clip)}</p>
    <h1>Evidence</h1>
    {evidence_html}
    <h1>Adjudicator narrative</h1>
    <p class="narrative">{_e(adj.narrative)}</p>
  </div>
  <footer>Sundial &mdash; VAST Builders Challenge</footer>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path
