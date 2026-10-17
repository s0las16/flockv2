"""Batch audit: run a folder of clips against one claim, render a leaderboard.

  python -m sundial.batch clips/ --claim-time "2025-10-17T15:00:00" \
      --claim-place "London" --out index.html

Every clip goes through the full pipeline (perception -> evidence ->
adjudication) in a worker thread — the stages are IO-bound so threads scale
fine — and gets its own verdict report + JSON sidecar in the workdir.
write_index() then renders a self-contained leaderboard page: CONTRADICTED
clips first, then SUPPORTED, INSUFFICIENT, ERROR.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import html
import json
import os
import re
import sys

# accent + rgba-prefix per row verdict, same palette family as report.py
_ACCENT = {
    "CONTRADICTED": ("#f87171", "rgba(248,113,113,"),
    "SUPPORTED": ("#4ade80", "rgba(74,222,128,"),
    "INSUFFICIENT": ("#94a3b8", "rgba(148,163,184,"),
    "ERROR": ("#fb923c", "rgba(251,146,60,"),
}
_ORDER = {"CONTRADICTED": 0, "SUPPORTED": 1, "INSUFFICIENT": 2, "ERROR": 3}
_EXCERPT_LEN = 170


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _norm_verdict(row: dict) -> str:
    v = str(row.get("verdict") or "ERROR").upper()
    return v if v in _ORDER else "ERROR"


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "clip"


def _audit_one(clip: str, claim_time, claim_place, report_html: str, frames_dir: str) -> dict:
    """Worker: pipeline -> report HTML + JSON sidecar -> leaderboard row."""
    from .pipeline import run  # lazy: heavy stage deps live behind this
    from .report import render_html

    adj = run(clip, claim_time=claim_time, claim_place=claim_place, workdir=frames_dir)
    render_html(adj, report_html)
    with open(os.path.splitext(report_html)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(adj.to_dict(), f, indent=2, default=str)
    return {
        "clip": clip,
        "verdict": getattr(adj.verdict, "value", str(adj.verdict)),
        "confidence": float(adj.confidence or 0.0),
        "narrative": adj.narrative or "",
        "report": report_html,
    }


def run_batch(clips: list[str], claim_time=None, claim_place=None,
              workdir: str = ".sundial_batch", max_workers: int = 2) -> list[dict]:
    """Audit every clip against the claim; per-clip failures become ERROR rows.

    Returns rows in input order. Success rows carry verdict/confidence/
    narrative/report; failures carry {"clip", "verdict": "ERROR", "error"}.
    """
    frames_root = os.path.join(workdir, "frames")
    os.makedirs(frames_root, exist_ok=True)

    def job(i: int, clip: str) -> dict:
        # per-clip stem keeps frame dirs and report names collision-free
        stem = f"{i:02d}_{_safe_stem(clip)}"
        report_html = os.path.join(workdir, stem + ".html")
        frames_dir = os.path.join(frames_root, stem)
        try:
            row = _audit_one(clip, claim_time, claim_place, report_html, frames_dir)
        except Exception as e:
            row = {"clip": clip, "verdict": "ERROR", "error": str(e)}
        row["claim"] = {"time": claim_time, "place": claim_place}
        return row

    rows: list[dict] = [None] * len(clips)  # type: ignore[list-item]
    with cf.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futs = {ex.submit(job, i, clip): i for i, clip in enumerate(clips)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            try:
                rows[i] = fut.result()
            except Exception as e:  # job() already guards; belt & braces
                rows[i] = {"clip": clips[i], "verdict": "ERROR", "error": str(e)}
    return rows


def _row_card(row: dict, rank: int, out_dir: str) -> str:
    verdict = _norm_verdict(row)
    accent, rgba = _ACCENT[verdict]
    name = os.path.basename(str(row.get("clip") or "unknown"))
    try:
        pct = max(0.0, min(1.0, float(row.get("confidence")))) * 100
        conf = f"{pct:.0f}%"
    except (TypeError, ValueError):
        pct, conf = 0.0, "&mdash;"

    excerpt = (row.get("narrative") or row.get("error") or "").strip()
    excerpt = excerpt.splitlines()[0] if excerpt else "no narrative recorded"
    if len(excerpt) > _EXCERPT_LEN:
        excerpt = excerpt[: _EXCERPT_LEN - 1].rstrip() + "…"

    href = None
    report = row.get("report")
    if report:
        try:
            href = os.path.relpath(str(report), out_dir)
        except (OSError, ValueError):
            href = str(report)

    pill = (
        f'<span class="pill" style="color:{accent};background:{rgba}.12);'
        f'border:1px solid {rgba}.35)">{verdict}</span>'
    )
    link = (
        f'<span class="open" style="color:{accent}">view report &rarr;</span>'
        if href
        else '<span class="open dead">no report</span>'
    )
    body = (
        f'<div class="card" style="--accent:{accent};animation-delay:{rank * 70}ms">'
        f'<div class="cardtop"><span class="rank">#{rank + 1}</span>{pill}</div>'
        f'<div class="fname">{_e(name)}</div>'
        f'<div class="confrow"><div class="confbar"><div class="conffill" '
        f'style="width:{pct:.0f}%;background:{accent}"></div></div>'
        f'<span class="confnum">{conf}</span></div>'
        f'<div class="excerpt">{_e(excerpt)}</div>{link}</div>'
    )
    return f'<a class="cardlink" href="{_e(href)}">{body}</a>' if href else body


def write_index(rows: list[dict], out_html: str) -> str:
    """Write a self-contained dark leaderboard page; returns out_html."""
    rows = list(rows or [])
    out_dir = os.path.dirname(os.path.abspath(out_html))
    ranked = sorted(
        rows,
        key=lambda r: (
            _ORDER[_norm_verdict(r)],
            -float(r.get("confidence") or 0.0),
            os.path.basename(str(r.get("clip") or "")).lower(),
        ),
    )

    counts = {k: 0 for k in _ORDER}
    for r in ranked:
        counts[_norm_verdict(r)] += 1
    tally = " ".join(
        f'<span class="tally" style="color:{_ACCENT[k][0]}">{counts[k]} {k.lower()}</span>'
        for k in ("CONTRADICTED", "SUPPORTED", "INSUFFICIENT", "ERROR")
        if counts[k]
    ) or '<span class="tally">no clips audited</span>'

    claim = next((r.get("claim") for r in rows if r.get("claim")), None) or {}
    claim_bits = []
    if claim.get("time"):
        claim_bits.append(f'<span class="ck">time</span><b>{_e(claim["time"])}</b>')
    if claim.get("place"):
        claim_bits.append(f'<span class="ck">place</span><b>{_e(claim["place"])}</b>')
    claim_html = (
        '<div class="claim">' + '<span class="cdot">·</span>'.join(claim_bits) + "</div>"
        if claim_bits
        else ""
    )

    cards = "".join(_row_card(r, i, out_dir) for i, r in enumerate(ranked))
    grid = (
        f'<div class="grid">{cards}</div>'
        if cards
        else '<p class="empty">No clips were audited.</p>'
    )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sundial — batch audit</title>
<style>
* {{ box-sizing:border-box; }}
body {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
       background:#070b14; color:#cbd5e1; margin:0; min-height:100vh;
       background-image:radial-gradient(rgba(148,163,184,.13) 1px, transparent 1px);
       background-size:26px 26px; animation:drift 60s linear infinite; }}
@keyframes drift {{ to {{ background-position:520px 260px; }} }}
.wrap {{ max-width:1080px; margin:0 auto; padding:52px 18px 60px; }}
.brand {{ font-size:12px; letter-spacing:.35em; color:#64748b; text-transform:uppercase;
         margin-bottom:14px; display:flex; align-items:center; gap:10px; }}
.brand::before {{ content:""; width:26px; height:1px; background:#334155; }}
h1 {{ font-size:26px; font-weight:800; letter-spacing:.06em; color:#e2e8f0; margin:0 0 10px; }}
.claim {{ font-size:14px; display:flex; gap:10px; flex-wrap:wrap; align-items:baseline;
         margin-bottom:6px; }}
.ck {{ color:#64748b; font-size:11px; text-transform:uppercase; letter-spacing:.15em; }}
.cdot {{ color:#334155; }}
.tallyrow {{ font-size:12px; letter-spacing:.08em; margin:14px 0 30px;
            display:flex; gap:16px; flex-wrap:wrap; }}
.tally {{ text-transform:uppercase; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }}
.cardlink {{ text-decoration:none; color:inherit; display:block; }}
.card {{ position:relative; height:100%; border-radius:14px; padding:18px 20px;
        background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.14);
        box-shadow:0 14px 40px rgba(0,0,0,.45); overflow:hidden;
        opacity:0; transform:translateY(10px);
        animation:rise .5s cubic-bezier(.2,.7,.2,1) forwards;
        transition:border-color .2s, transform .2s; }}
.card::after {{ content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(280px circle at var(--mx,50%) var(--my,50%), rgba(148,163,184,.09), transparent 60%);
  opacity:0; transition:opacity .25s; }}
.card:hover::after {{ opacity:1; }}
.cardlink:hover .card {{ border-color:rgba(148,163,184,.35); transform:translateY(-2px); }}
@keyframes rise {{ to {{ opacity:1; transform:none; }} }}
.cardtop {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }}
.rank {{ color:#475569; font-size:12px; letter-spacing:.1em; }}
.pill {{ border-radius:999px; padding:2px 10px; font-size:11px; font-weight:700; }}
.fname {{ font-weight:700; font-size:14px; color:#e2e8f0; word-break:break-all; margin-bottom:12px; }}
.confrow {{ display:flex; align-items:center; gap:10px; margin-bottom:12px; }}
.confbar {{ flex:1; height:4px; background:rgba(148,163,184,.12); border-radius:2px; overflow:hidden; }}
.conffill {{ height:100%; border-radius:2px; }}
.confnum {{ font-size:12px; color:#94a3b8; min-width:4ch; text-align:right; }}
.excerpt {{ font-size:12px; line-height:1.55; color:#94a3b8; min-height:3.4em; margin-bottom:12px; }}
.open {{ font-size:12px; font-weight:700; letter-spacing:.06em; }}
.open.dead {{ color:#475569; }}
.empty {{ color:#475569; }}
footer {{ text-align:center; color:#334155; font-size:11px; letter-spacing:.2em;
         text-transform:uppercase; margin-top:38px; }}
</style></head>
<body>
<div class="wrap">
  <div class="brand">Sundial &mdash; video provenance verification</div>
  <h1>Batch audit &mdash; {len(ranked)} clip{"s" if len(ranked) != 1 else ""}</h1>
  {claim_html}
  <div class="tallyrow">{tally}</div>
  {grid}
  <footer>Sundial &middot; VAST Builders Challenge</footer>
</div>
<script>
// spotlight tracks the cursor across cards
document.addEventListener("mousemove", function(e) {{
  document.querySelectorAll(".card").forEach(function(el) {{
    var r = el.getBoundingClientRect();
    el.style.setProperty("--mx", (e.clientX - r.left) + "px");
    el.style.setProperty("--my", (e.clientY - r.top) + "px");
  }});
}});
</script>
</body></html>"""

    out_dir_abs = os.path.dirname(os.path.abspath(out_html))
    os.makedirs(out_dir_abs, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_html


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sundial.batch", description=__doc__)
    ap.add_argument("clips_dir", help="folder of clips to audit (*.mp4 / *.mov)")
    ap.add_argument("--claim-time", help="ISO datetime the clips claim (local to place)")
    ap.add_argument("--claim-place", help='claimed location: text or "lat,lon"')
    ap.add_argument("--out", default="index.html", help="leaderboard HTML output path")
    ap.add_argument("--workdir", default=".sundial_batch",
                    help="dir for per-clip reports, JSON sidecars, extracted frames")
    ap.add_argument("--workers", type=int, default=2, help="parallel clip audits")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.clips_dir):
        ap.error(f"no such directory: {args.clips_dir}")
    clips = sorted({
        p
        for pat in ("*.mp4", "*.mov", "*.MP4", "*.MOV")
        for p in glob.glob(os.path.join(args.clips_dir, pat))
    })
    if not clips:
        ap.error(f"no .mp4/.mov clips found in {args.clips_dir}")

    rows = run_batch(
        clips,
        claim_time=args.claim_time,
        claim_place=args.claim_place,
        workdir=args.workdir,
        max_workers=args.workers,
    )
    path = write_index(rows, args.out)
    print(json.dumps({
        "clips": len(rows),
        "verdicts": {r["clip"]: r["verdict"] for r in rows},
        "index": path,
        "workdir": args.workdir,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
