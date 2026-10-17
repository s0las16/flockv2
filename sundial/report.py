"""Self-contained HTML verdict report — dark theme, animated, zero deps.

reactbits-style polish in vanilla CSS/JS: spotlight hover cards, a
scramble-in verdict, count-up confidence, staggered evidence reveal, and
an SVG sun-position diagram for solar evidence. No build step, no network
assets — the file can be scp'd and opened anywhere.
"""
from __future__ import annotations

import html
import json
import os
from typing import Any

from .schema import Adjudication, ClaimVerdict, Verdict

_VERDICT_COLORS = {
    Verdict.SUPPORTED: ("#4ade80", "rgba(74,222,128,"),
    Verdict.CONTRADICTED: ("#f87171", "rgba(248,113,113,"),
    Verdict.INSUFFICIENT: ("#94a3b8", "rgba(148,163,184,"),
}
_PILL = {
    ClaimVerdict.SUPPORTS: ("#4ade80", "rgba(74,222,128,.12)", "rgba(74,222,128,.35)"),
    ClaimVerdict.CONTRADICTS: ("#f87171", "rgba(248,113,113,.12)", "rgba(248,113,113,.35)"),
    ClaimVerdict.INCONCLUSIVE: ("#94a3b8", "rgba(148,163,184,.10)", "rgba(148,163,184,.30)"),
}
_ICONS = {
    "solar": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.2"/><g stroke-width="1.6"><line x1="12" y1="2.5" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="21.5"/><line x1="2.5" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="21.5" y2="12"/><line x1="5.3" y1="5.3" x2="7" y2="7"/><line x1="17" y1="17" x2="18.7" y2="18.7"/><line x1="5.3" y1="18.7" x2="7" y2="17"/><line x1="17" y1="7" x2="18.7" y2="5.3"/></g></svg>',
    "weather": '<svg viewBox="0 0 24 24"><path d="M6.5 18.5a4.5 4.5 0 0 1-.4-8.98 5.5 5.5 0 0 1 10.7-1.3A4.2 4.2 0 0 1 17.4 18.5z"/></svg>',
    "geo": '<svg viewBox="0 0 24 24"><path d="M12 21s-6.5-5.4-6.5-10.2A6.5 6.5 0 0 1 12 4.3a6.5 6.5 0 0 1 6.5 6.5C18.5 15.6 12 21 12 21z"/><circle cx="12" cy="10.6" r="2.2"/></svg>',
    "clock": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.2"/><path d="M12 7.4V12l3 2.2"/></svg>',
    "shadows": '<svg viewBox="0 0 24 24"><circle cx="9" cy="6.5" r="2.4"/><path d="M9 9.5v7.5M9 9.5 5.5 13M9 9.5l3.5 3.5M4 20.5h13" stroke-linecap="round"/></svg>',
    "texture": '<svg viewBox="0 0 24 24"><path d="M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/></svg>',
    "scene": '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 15l5-4 4 3 5-5 4 4"/></svg>',
    "language": '<svg viewBox="0 0 24 24"><path d="M4 6h9M8.5 6V4M6 10c1.5 3 4.5 5 8 5.5M12 10c-1.5 3-4.5 5-8 5.5M13 20l4.5-9L22 20M14.7 16.5h5.6"/></svg>',
    "plate_style": '<svg viewBox="0 0 24 24"><rect x="3" y="9" width="18" height="7" rx="1.5"/><path d="M3 12h18"/></svg>',
    "lunar": '<svg viewBox="0 0 24 24"><path d="M19.5 14.5A8 8 0 1 1 9.5 4.5a6.5 6.5 0 0 0 10 10z"/></svg>',
    "audio": '<svg viewBox="0 0 24 24"><path d="M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4" stroke-linecap="round"/></svg>',
    "azimuth": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.2"/><path d="M12 12l4.5-5M12 12l-4 4" stroke-linecap="round"/><circle cx="12" cy="12" r="1" fill="currentColor"/></svg>',
    "osint": '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M15.5 15.5L20 20" stroke-linecap="round"/><circle cx="11" cy="11" r="2.2"/></svg>',
    "osint_features": '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M15.5 15.5L20 20" stroke-linecap="round"/></svg>',
    "geoguess": '<svg viewBox="0 0 24 24"><circle cx="12" cy="11" r="7.5"/><path d="M4.5 11h15M12 3.5c-3 3-3 12 0 15M12 3.5c3 3 3 12 0 15"/></svg>',
    "landmark": '<svg viewBox="0 0 24 24"><path d="M5 20h14M6 20V10l2-2v12M10 20V6l2-2 2 2v14M16 20V10l2-2v12"/></svg>',
    "sign_style": '<svg viewBox="0 0 24 24"><path d="M12 3l9 9-9 9-9-9z"/><path d="M9.5 12.5l2 2 3.5-4"/></svg>',
    "vegetation": '<svg viewBox="0 0 24 24"><path d="M12 21v-8M12 13c0-4-3-7-7-7 0 4 3 7 7 7zM12 11c0-4 3-7 7-7 0 4-3 7-7 7z"/></svg>',
    "metadata": '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 9h8M8 13h8M8 17h5"/></svg>',
    "poi": '<svg viewBox="0 0 24 24"><circle cx="9" cy="9" r="5"/><path d="M13 13l7 7" stroke-linecap="round"/></svg>',
    "wiki": '<svg viewBox="0 0 24 24"><circle cx="12" cy="11" r="7.5"/><path d="M8 8.5c1 1.5 2.5 2 4 2s3-.5 4-2"/></svg>',
    "wiki_landmark": '<svg viewBox="0 0 24 24"><circle cx="12" cy="11" r="7.5"/><path d="M8 8.5c1 1.5 2.5 2 4 2s3-.5 4-2"/></svg>',
    "climate": '<svg viewBox="0 0 24 24"><path d="M12 3v18M5 7l14 10M19 7L5 17"/><circle cx="12" cy="12" r="3"/></svg>',
    "hemisphere": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17"/></svg>',
    "shadow_consistency": '<svg viewBox="0 0 24 24"><circle cx="9" cy="6.5" r="2.4"/><path d="M9 9.5v7.5M4 20.5h13M14 8l4-4" stroke-linecap="round"/></svg>',
    "geo_match": '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M15.5 15.5L20 20" stroke-linecap="round"/><path d="M8.5 11l1.8 1.8 3.2-3.8"/></svg>',
    "geo_rollup": '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M15.5 15.5L20 20" stroke-linecap="round"/></svg>',
    "timezone": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2.2M12 3.5v-1"/></svg>',
    "frames": '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="8" height="14" rx="1.5"/><rect x="13" y="5" width="8" height="14" rx="1.5"/></svg>',
    "corpus": '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M8 4v16"/></svg>',
}
# Provenance channel -> short badge label + tint (matches provenance.py).
_CHANNELS = {
    "visual": ("VIS", "#38bdf8"),
    "external": ("EXT", "#c084fc"),
    "derived": ("DRV", "#facc15"),
    "ground-truth": ("GT", "#4ade80"),
    "ground_truth": ("GT", "#4ade80"),  # tolerate underscore spelling
}
_DEFAULT_ICON = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.2"/><path d="M12 10.8v5M12 7.6v.4" stroke-linecap="round"/></svg>'


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _verdict_str(v: Any) -> str:
    return getattr(v, "value", str(v))


def _cv(value: Any) -> ClaimVerdict:
    try:
        return value if isinstance(value, ClaimVerdict) else ClaimVerdict(str(value).lower())
    except ValueError:
        return ClaimVerdict.INCONCLUSIVE


def _pill(verdict: Any) -> str:
    cv = _cv(verdict)
    fg, bg, border = _PILL[cv]
    return (
        f'<span class="pill" style="color:{fg};background:{bg};'
        f'border:1px solid {border}">{_e(_verdict_str(verdict))}</span>'
    )


def _sun_viz(ev: "Evidence") -> str:
    """SVG diagram for solar evidence: horizon + expected vs observed sun."""
    data = getattr(ev, "data", None)
    if not isinstance(data, dict):
        return ""
    exp = data.get("expected")
    obs = data.get("observed")
    if not isinstance(exp, dict) or not isinstance(obs, dict):
        return ""
    try:
        e_el = float(exp["elevation_deg"])
        e_az_raw = exp.get("azimuth_deg")
        e_az = float(e_az_raw) if e_az_raw is not None else 180.0
        o_el = float(obs["elevation_deg"])
        o_az_raw = obs.get("azimuth_deg")
        o_az = float(o_az_raw) if o_az_raw is not None else None
    except (KeyError, TypeError, ValueError):
        return ""

    W, H, horizon_y = 460, 170, 128
    max_el = max(60.0, e_el, o_el) + 10

    def xy(az: float, el: float) -> tuple[float, float]:
        # azimuth 90..270 sweeps across the dome (E -> S -> W);
        # clamped so out-of-range bearings pin to the edge.
        az_c = min(270.0, max(90.0, az))
        x = 30 + (az_c - 90) / 180 * (W - 60)
        y = horizon_y - (el / max_el) * (horizon_y - 24)
        return x, y

    def label(x: float, text: str, dy: float, fill: str = "#9a9a9a") -> str:
        # flip to left-anchored text near the right edge so labels don't clip
        if x > W - 140:
            return (f'<text x="{x - 12:.0f}" y="{dy:.0f}" class="svlabel" '
                    f'fill="{fill}" text-anchor="end">{text}</text>')
        return f'<text x="{x + 12:.0f}" y="{dy:.0f}" class="svlabel" fill="{fill}">{text}</text>'

    ex, ey = xy(e_az, e_el)
    ox, oy = xy(e_az if o_az is None else o_az, max(o_el, 1.5))
    parts = [
        f'<svg class="sunviz" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        f'<path d="M30 {horizon_y} A 200 200 0 0 1 {W-30} {horizon_y}" fill="none" stroke="#2a2a2e" stroke-dasharray="4 5" stroke-width="1"/>',
        f'<line x1="14" y1="{horizon_y}" x2="{W-14}" y2="{horizon_y}" stroke="#3d3d42" stroke-width="1.4"/>',
        f'<text x="18" y="{horizon_y+13}" class="svlabel">horizon</text>',
    ]
    if e_el < 0:  # expected sun under the horizon — the night-lie case
        parts.append(
            f'<circle cx="{ex:.0f}" cy="{horizon_y + 22:.0f}" r="7" fill="none" stroke="#f87171" stroke-width="1.6"/>'
            + label(ex, f"claim expects sun {-e_el:.0f}° below", horizon_y + 26, "#f87171")
        )
    else:
        parts.append(
            f'<circle cx="{ex:.0f}" cy="{ey:.0f}" r="7" fill="none" stroke="#9a9a9a" stroke-width="1.6"/>'
            f'<line x1="{ex:.0f}" y1="{ey:.0f}" x2="{ex:.0f}" y2="{horizon_y}" stroke="#9a9a9a" stroke-dasharray="2 4" stroke-width="1"/>'
            + label(ex, f"claimed: {e_el:.0f}°", ey + 4)
        )
    parts.append(
        f'<circle cx="{ox:.0f}" cy="{oy:.0f}" r="8" fill="#facc15" opacity="0.95"/>'
        f'<line x1="{ox:.0f}" y1="{oy:.0f}" x2="{ox:.0f}" y2="{horizon_y}" stroke="#facc15" stroke-dasharray="2 4" stroke-width="1" opacity="0.6"/>'
        + label(ox, f"measured: {o_el:.0f}°", oy + 4, "#fde68a")
    )
    parts.append("</svg>")
    return "".join(parts)


def _frame_viz(ev: "Evidence") -> str:
    """Annotated frame thumbnail — the image plus the shadow vector the
    measurement came from, embedded as a base64 JPEG inside one SVG.

    For 'clock' cards the top strip (where overlays live) is shown instead.
    Silent "" on any failure — imagery is a nice-to-have, never fatal.
    """
    data = getattr(ev, "data", None)
    if not isinstance(data, dict):
        return ""
    frame = data.get("frame") or (ev.frames[0] if getattr(ev, "frames", None) else None)
    if not frame or not os.path.isfile(str(frame)):
        return ""
    try:
        import base64
        import io
        from PIL import Image

        img = Image.open(str(frame)).convert("RGB")
        W0, H0 = img.size
        strip_only = ev.type == "clock"
        if strip_only:
            img = img.crop((0, 0, W0, max(1, round(H0 * 0.35))))
            W0, H0 = img.size
        scale = min(1.0, 620.0 / W0)
        w, h = max(1, round(W0 * scale)), max(1, round(H0 * scale))
        img = img.resize((w, h))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=62)
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

    ov = ""
    obs = data.get("observation")
    if isinstance(obs, dict) and not strip_only:
        try:
            bx, by = float(obs["base_point"][0]) * scale, float(obs["base_point"][1]) * scale
            tx, ty = float(obs["shadow_tip"][0]) * scale, float(obs["shadow_tip"][1]) * scale
            vec = data.get("vector") or {}
            ln = vec.get("length_px")
            label = f"{ln:.0f}px" if isinstance(ln, (int, float)) else "shadow"
            ov = (
                f'<line x1="{bx:.0f}" y1="{by:.0f}" x2="{tx:.0f}" y2="{ty:.0f}" '
                f'stroke="#facc15" stroke-width="2.5" opacity="0.95"/>'
                f'<circle cx="{bx:.0f}" cy="{by:.0f}" r="5" fill="#facc15"/>'
                f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="5" fill="none" stroke="#facc15" stroke-width="2"/>'
                f'<text x="{min(tx + 8, w - 60):.0f}" y="{max(ty - 8, 12):.0f}" '
                f'class="svlabel" fill="#fde68a">{_e(label)}</text>'
            )
        except (KeyError, TypeError, ValueError, IndexError):
            ov = ""
    return (
        f'<svg class="frameviz" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'>'
        f'<image href="data:image/jpeg;base64,{b64}" '
        f'x="0" y="0" width="{w}" height="{h}"/>{ov}</svg>'
    )


def _ev_card(ev: "Evidence", idx: int) -> str:
    cv = _cv(ev.verdict)
    fg, bg, border = _PILL[cv]
    icon = _ICONS.get(ev.type, _DEFAULT_ICON)
    try:
        pct = max(0.0, min(1.0, float(ev.confidence))) * 100
    except (TypeError, ValueError):
        pct = 0.0
    frames = ""
    if ev.frames:
        names = ", ".join(os.path.basename(str(f)) for f in ev.frames[:4])
        frames = f'<div class="frames">cited: {_e(names)}</div>'
    chan = ""
    data = getattr(ev, "data", None)
    if isinstance(data, dict):
        c = _CHANNELS.get(str(data.get("channel") or ""))
        if c:
            chan = (f'<span class="chan" style="color:{c[1]};'
                    f'border-color:{c[1]}44" title="{c[0]} channel">{c[0]}</span>')
    return (
        f'<div class="ev" style="animation-delay:{120 + idx * 90}ms">'
        f'<div class="evhead"><span class="evicon" style="color:{fg}">{icon}</span>'
        f'<span class="etype">{_e(ev.type)}</span>{chan}{_pill(ev.verdict)}'
        f'<span class="econfnum">{pct:.0f}%</span></div>'
        f'<div class="confbar"><div class="conffill" data-w="{pct:.0f}" '
        f'style="background:{fg};width:{pct:.0f}%"></div></div>'
        f'<div class="evsum">{_e(ev.summary)}</div>{frames}{_frame_viz(ev)}</div>'
    )


def _ranked_rows(ranked: Any) -> list[tuple[str, "float | None"]]:
    """Normalize a ranked-location list into [(label, score|None)].

    Tolerates {cc: score} mappings, dicts keyed cc|code|country|name|label
    with score|confidence|weight, (label, score) tuples, and bare strings.
    Never raises — the rail is a nice-to-have, not a fatal path.
    """
    if isinstance(ranked, dict):
        ranked = [{"cc": k, "score": v} for k, v in ranked.items()]
    if not isinstance(ranked, (list, tuple)):
        return []
    rows: list[tuple[str, "float | None"]] = []
    for r in ranked:
        name: Any = None
        score: Any = None
        if isinstance(r, dict):
            name = (r.get("cc") or r.get("code") or r.get("country_code")
                    or r.get("name") or r.get("country") or r.get("label"))
            score = r.get("score", r.get("confidence", r.get("weight")))
        elif isinstance(r, (list, tuple)) and r:
            name = r[0]
            score = r[1] if len(r) > 1 else None
        elif isinstance(r, str):
            name = r
        if name is None or str(name).strip() == "":
            continue
        try:
            score_f: "float | None" = float(score)
        except (TypeError, ValueError):
            score_f = None
        rows.append((str(name), score_f))
    return rows


def _loc_rail(extra: "dict | None", mix_html: str) -> str:
    """The right-hand "where it thinks it is" rail.

    `extra` is an optional bag handed in by the caller: map_svg (trusted
    inline <svg> from sundial.vec_map — inserted verbatim), map_fragment
    (trusted HTML from sundial.map_viz — inserted verbatim), map_static
    (base64 PNG), loc_summary ({best_name, ranked}), and chain_html
    (trusted reasoning-chain fragment). Everything degrades silently: a
    missing map renders a placeholder, and no geo data at all renders a
    graceful "no geo read" empty state. mix_html (signal mix) anchors the
    rail bottom.
    """
    ex = extra if isinstance(extra, dict) else {}
    svg_map = ex.get("map_svg")
    frag = ex.get("map_fragment")
    static = ex.get("map_static")
    loc = ex.get("loc_summary")
    loc = loc if isinstance(loc, dict) else {}
    chain = ex.get("chain_html")
    has_geo = any((
        isinstance(svg_map, str) and svg_map.strip(),
        isinstance(frag, str) and frag.strip(),
        isinstance(static, str) and static.strip(),
        bool(loc),
        isinstance(chain, str) and chain.strip(),
    ))

    # 1. map — local inline svg > fragment verbatim > static PNG >
    #    placeholder; never fatal
    if isinstance(svg_map, str) and svg_map.strip():
        map_block = f'<div class="mapbox">{svg_map}</div>'
    elif isinstance(frag, str) and frag.strip():
        map_block = f'<div class="mapbox">{frag}</div>'
    elif isinstance(static, str) and static.strip():
        src = static.strip()
        if not src.startswith("data:"):
            src = "data:image/png;base64," + src
        map_block = ('<div class="mapbox"><img class="mapimg" '
                     f'src="{_e(src)}" alt="estimated location map"></div>')
    else:
        map_block = '<div class="mapnone">map unavailable</div>'

    # 2. location readout — best guess big, then ranked confidence bars
    loc_block = ""
    if loc:
        best = loc.get("best_name") or loc.get("best") or loc.get("name")
        if isinstance(best, dict):
            best = best.get("name") or best.get("cc") or best.get("code")
        rows = _ranked_rows(loc.get("ranked"))[:6]
        bits = []
        if best:
            bits.append(f'<div class="locbest">{_e(best)}</div>')
        if rows:
            lis = []
            for name, sc in rows:
                w = 0.0 if sc is None else min(100.0, max(0.0, sc * 100))
                num = "" if sc is None else f'<span class="locscore">{sc:.2f}</span>'
                lis.append(
                    f'<div class="locrow"><span class="loccc">{_e(name)}</span>'
                    f'<span class="locbar"><span class="locfill" '
                    f'style="width:{w:.0f}%"></span></span>{num}</div>')
            bits.append('<div class="loclist">' + "".join(lis) + "</div>")
        if bits:
            loc_block = '<div class="locread">' + "".join(bits) + "</div>"

    # 3. reasoning chain fragment, if the caller produced one
    chain_block = ""
    if isinstance(chain, str) and chain.strip():
        chain_block = ('<div class="chainbox"><div class="railsub">'
                       'reasoning chain</div>' + chain + "</div>")

    if has_geo:
        inner = map_block + loc_block + chain_block
    else:
        inner = ('<div class="nogeo"><span class="nogeoicon">' + _ICONS["geo"]
                 + '</span>no geo read'
                 '<span class="nogeosub">add a place claim</span></div>')

    foot = f'<div class="railfoot">{mix_html}</div>' if mix_html else ""
    return ('<aside class="rail"><div class="railpanel">'
            '<div class="railtitle">Where it thinks it is</div>'
            + inner + foot + "</div></aside>")


def render_html(adj: Adjudication, out_path: str,
                extra: "dict | None" = None) -> str:
    """Write a self-contained animated verdict report; returns out_path.

    `extra` optionally carries the geo rail payloads — {"map_svg",
    "map_fragment", "map_static", "loc_summary", "chain_html"} —
    produced by sibling tooling (e.g. sundial.vec_map, sundial.map_viz).
    All keys are optional; with none of them the rail renders a
    graceful "no geo read" empty state.
    """
    pack = adj.pack
    claim = pack.claim
    if isinstance(adj.verdict, Verdict):
        verdict = adj.verdict
    else:
        try:
            verdict = Verdict(str(adj.verdict).upper())
        except ValueError:
            verdict = Verdict.INSUFFICIENT
    accent, glow = _VERDICT_COLORS[verdict]
    try:
        conf_pct = max(0.0, min(1.0, float(adj.confidence))) * 100
    except (TypeError, ValueError):
        conf_pct = 0.0

    claim_bits = []
    if claim.claimed_time:
        claim_bits.append(f'<span class="ck">time</span><b>{_e(claim.claimed_time)}</b>')
    if claim.claimed_place:
        claim_bits.append(f'<span class="ck">place</span><b>{_e(claim.claimed_place)}</b>')
    claim_html = '<span class="cdot">·</span>'.join(claim_bits) or "<i>no claim recorded</i>"
    if claim.raw:
        claim_html += f'<div class="craw">&ldquo;{_e(claim.raw)}&rdquo;</div>'

    # order the evidence like an argument: contradicts, supports, then the rest
    order = {ClaimVerdict.CONTRADICTS: 0, ClaimVerdict.SUPPORTS: 1, ClaimVerdict.INCONCLUSIVE: 2}

    def _conf_of(e) -> float:
        try:
            return float(e.confidence or 0)
        except (TypeError, ValueError):
            return 0.0

    evidence = sorted(pack.evidence, key=lambda e: (order[_cv(e.verdict)], -_conf_of(e)))
    try:
        # Collapse the inconclusive geo-family tail into one digest row —
        # supports/contradicts cards stay individually visible. Display
        # only: the adjudicator already saw the full pack.
        from . import dedupe_evidence
        evidence = dedupe_evidence.compact_for_display(evidence)
    except Exception:
        pass
    cards = []
    sun_shown = False
    for i, ev in enumerate(evidence):
        cards.append(_ev_card(ev, i))
        if ev.type == "solar" and not sun_shown:
            viz = _sun_viz(ev)
            if viz:
                cards.append(f'<div class="ev sunrow" style="animation-delay:{120 + (i + 0.5) * 90}ms">{viz}</div>')
                sun_shown = True
    evidence_html = (
        '<div class="evgrid">' + "".join(cards) + "</div>"
        if cards
        else '<p class="empty">No evidence was collected.</p>'
    )

    # signal mix: how much of the pack came from each provenance channel
    mix_html = ""
    try:
        from . import provenance
        mix = provenance.channel_summary(evidence)
        if any(mix.values()):
            parts = []
            for name in ("visual", "external", "derived", "ground-truth"):
                c = _CHANNELS[name]
                parts.append(
                    f'<span><b style="color:{c[1]}">{c[0]}</b>{mix[name]}</span>')
            mix_html = ('<div class="signalmix"><span style="color:#475569">'
                        'signal mix</span>' + "".join(parts) + "</div>")
    except Exception:
        pass

    raw_json = html.escape(json.dumps(adj.to_dict(), indent=2, default=str))

    rail_html = _loc_rail(extra, mix_html)

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sundial — {_e(verdict.value)}</title>
<style>
:root {{ --accent:{accent}; }}
* {{ box-sizing:border-box; }}
body {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
       background:#050507; color:#e5e5e5; margin:0; min-height:100vh;
       background-image:radial-gradient(rgba(255,255,255,.05) 1px, transparent 1px);
       background-size:26px 26px; animation:drift 60s linear infinite; }}
@keyframes drift {{ to {{ background-position:520px 260px; }} }}
.wrap {{ max-width:1220px; margin:0 auto; padding:46px 20px 60px; }}
.brand {{ font-size:12px; letter-spacing:.35em; color:#666; text-transform:uppercase;
         margin-bottom:24px; display:flex; align-items:center; gap:10px; }}
.brand::before {{ content:""; width:26px; height:1px; background:#262626; }}
.layout {{ display:grid; grid-template-columns:minmax(0,62fr) minmax(300px,38fr);
          gap:22px; align-items:start; }}
.card {{ position:relative; border-radius:18px; padding:34px 38px;
        background:rgba(13,13,16,.82); border:1px solid #1c1c1c;
        backdrop-filter:blur(6px);
        box-shadow:0 24px 70px rgba(0,0,0,.6), 0 0 0 1px rgba(0,0,0,.55) inset; }}
.card::before {{ content:""; position:absolute; inset:-1px; border-radius:19px; padding:1px;
  background:radial-gradient(420px circle at var(--mx,50%) var(--my,-20%),
      {glow}.55), transparent 40%);
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite:xor; mask-composite:exclude; pointer-events:none; }}
.rail {{ position:sticky; top:24px; }}
.railpanel {{ position:relative; border-radius:18px; padding:20px 20px 16px;
             background:rgba(13,13,16,.82); border:1px solid #1c1c1c;
             overflow:hidden; }}
.railpanel::after {{ content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(260px circle at var(--mx,50%) var(--my,50%), rgba(255,255,255,.05), transparent 60%);
  opacity:0; transition:opacity .25s; }}
.railpanel:hover::after {{ opacity:1; }}
.railtitle {{ font-size:11px; letter-spacing:.28em; color:#666;
             text-transform:uppercase; margin-bottom:14px; display:flex;
             align-items:center; gap:10px; }}
.railtitle::after {{ content:""; flex:1; height:1px; background:#1c1c1c; }}
.railsub {{ font-size:10px; letter-spacing:.2em; color:#555;
           text-transform:uppercase; margin-bottom:8px; }}
.mapbox {{ border:1px solid #1c1c1c; border-radius:12px; overflow:hidden;
          background:#08080a; }}
.mapimg, .mapbox img {{ display:block; width:100%; height:auto; }}
.mapbox iframe {{ display:block; width:100%; border:0; }}
.mapbox svg {{ display:block; width:100%; height:auto; border-radius:10px; }}
.mapnone {{ display:flex; align-items:center; justify-content:center;
           min-height:150px; color:#444; font-size:11px;
           letter-spacing:.18em; text-transform:uppercase;
           border:1px dashed #262626; border-radius:12px; }}
.locread {{ margin-top:16px; }}
.locbest {{ font-size:22px; font-weight:800; color:#e5e5e5;
           letter-spacing:.03em; margin-bottom:10px; }}
.loclist {{ display:flex; flex-direction:column; gap:8px; }}
.locrow {{ display:flex; align-items:center; gap:10px; font-size:12px; }}
.loccc {{ color:#e5e5e5; font-weight:700; min-width:3ch; }}
.locbar {{ flex:1; display:block; height:4px; background:rgba(255,255,255,.08);
          border-radius:2px; overflow:hidden; }}
.locfill {{ display:block; height:100%; background:#38bdf8; border-radius:2px; }}
.locscore {{ color:#999; min-width:4ch; text-align:right;
            font-variant-numeric:tabular-nums; }}
.chainbox {{ margin-top:16px; border-top:1px solid #1c1c1c; padding-top:14px;
            font-size:12px; color:#999; }}
.railfoot {{ margin-top:16px; border-top:1px solid #1c1c1c; padding-top:12px; }}
.railfoot .signalmix {{ margin-top:0; }}
.nogeo {{ display:flex; flex-direction:column; align-items:center; gap:8px;
         padding:34px 12px; border:1px dashed #262626; border-radius:12px;
         color:#666; font-size:12px; letter-spacing:.14em;
         text-transform:uppercase; text-align:center; }}
.nogeoicon {{ display:flex; }}
.nogeoicon svg {{ width:26px; height:26px; fill:none; stroke:#444;
                 stroke-width:1.6; }}
.nogeosub {{ color:#444; font-size:10px; letter-spacing:.2em; }}
.verdictrow {{ display:flex; align-items:center; gap:26px; flex-wrap:wrap; }}
.badge {{ font-size:34px; font-weight:800; letter-spacing:.10em; color:var(--accent);
         text-shadow:0 0 26px {glow}.55); min-width:9ch; }}
.ring {{ width:86px; height:86px; transform:rotate(-90deg); }}
.ring .bg {{ stroke:rgba(255,255,255,.10); }}
.ring .fg {{ stroke:var(--accent); stroke-linecap:round;
           stroke-dasharray:226.2; stroke-dashoffset:226.2;
           transition:stroke-dashoffset 1.1s cubic-bezier(.2,.7,.2,1); }}
.ringwrap {{ position:relative; width:86px; height:86px; }}
.ringnum {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
           font-size:17px; font-weight:700; color:#e5e5e5; transform:none; }}
.confsub {{ font-size:11px; color:#666; letter-spacing:.14em; text-transform:uppercase; }}
h1 {{ font-size:12px; text-transform:uppercase; letter-spacing:.22em; color:#666;
     margin:32px 0 12px; border-bottom:1px solid #1c1c1c; padding-bottom:8px; }}
.claim {{ font-size:15px; display:flex; gap:10px; flex-wrap:wrap; align-items:baseline; }}
.ck {{ color:#666; font-size:11px; text-transform:uppercase; letter-spacing:.15em; }}
.cdot {{ color:#333; }}
.craw {{ color:#999; font-size:13px; margin-top:8px; font-style:italic; }}
.clip {{ color:#555; font-size:11px; margin-top:10px; word-break:break-all; }}
.evgrid {{ display:flex; flex-direction:column; gap:10px; }}
.ev {{ background:rgba(255,255,255,.028); border:1px solid #1c1c1c;
      border-radius:12px; padding:14px 16px; opacity:0; transform:translateY(10px);
      animation:rise .5s cubic-bezier(.2,.7,.2,1) forwards;
      position:relative; overflow:hidden; }}
.ev::after {{ content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(260px circle at var(--mx,50%) var(--my,50%), rgba(255,255,255,.05), transparent 60%);
  opacity:0; transition:opacity .25s; }}
.ev:hover::after {{ opacity:1; }}
@keyframes rise {{ to {{ opacity:1; transform:none; }} }}
.evhead {{ display:flex; align-items:center; gap:10px; }}
.evicon svg {{ width:19px; height:19px; fill:none; stroke:currentColor; stroke-width:1.6; }}
.etype {{ font-weight:700; font-size:13px; color:#e5e5e5; }}
.pill {{ margin-left:auto; border-radius:999px; padding:2px 10px; font-size:11px; font-weight:700; }}
.econfnum {{ font-size:12px; color:#999; min-width:4ch; text-align:right; }}
.confbar {{ height:4px; background:rgba(255,255,255,.08); border-radius:2px; margin:10px 0 9px; overflow:hidden; }}
.conffill {{ height:100%; width:0; border-radius:2px; transition:width 1s cubic-bezier(.2,.7,.2,1); }}
.evsum {{ font-size:13px; line-height:1.55; color:#d9d9d9; }}
.frames {{ color:#555; font-size:11px; margin-top:6px; }}
.chan {{ border:1px solid; border-radius:4px; padding:0 5px; font-size:9px;
        font-weight:700; letter-spacing:.08em; opacity:.85; }}
.signalmix {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:14px;
             font-size:11px; color:#666; }}
.signalmix b {{ color:#999; margin-right:4px; }}
.sunrow {{ display:flex; justify-content:center; padding:10px 16px 14px; }}
.sunviz {{ width:100%; max-width:460px; }}
.frameviz {{ width:100%; max-width:620px; border-radius:8px; margin-top:10px;
            border:1px solid #1c1c1c; display:block; }}
.svlabel {{ font:11px ui-monospace,monospace; fill:#999; }}
.narrative {{ font-size:14px; line-height:1.65; color:#e5e5e5;
             background:rgba(255,255,255,.03); border-left:3px solid var(--accent);
             padding:14px 18px; border-radius:0 10px 10px 0; }}
details {{ margin-top:10px; }}
summary {{ cursor:pointer; color:#666; font-size:12px; letter-spacing:.12em; text-transform:uppercase; }}
pre {{ font-size:11px; color:#999; background:rgba(0,0,0,.55); border:1px solid #1c1c1c;
      border-radius:10px; padding:14px; overflow:auto; max-height:320px; }}
.empty {{ color:#555; }}
footer {{ text-align:center; color:#444; font-size:11px; letter-spacing:.2em;
         text-transform:uppercase; margin-top:34px; }}
@media (max-width:1099px) {{
  .layout {{ grid-template-columns:minmax(0,1fr); }}
  .rail {{ position:static; }}
  .card {{ padding:26px 22px; }}
}}
</style></head>
<body>
<div class="wrap">
  <div class="brand">Sundial &mdash; video provenance verification</div>
  <div class="layout">
    <div class="card" id="spot">
      <div class="verdictrow">
        <div class="badge" id="verdictTxt" data-final="{_e(verdict.value)}">{_e(verdict.value)}</div>
        <div class="ringwrap">
          <svg class="ring" viewBox="0 0 86 86"><circle class="bg" cx="43" cy="43" r="36" fill="none" stroke-width="6"/>
          <circle class="fg" id="ringFg" cx="43" cy="43" r="36" fill="none" stroke-width="6"
                  style="stroke-dashoffset:{226.2 * (1 - conf_pct / 100):.1f}"/></svg>
          <div class="ringnum" id="confNum" data-target="{conf_pct:.0f}">{conf_pct:.0f}%</div>
        </div>
        <div class="confsub">adjudicator<br>confidence</div>
      </div>
      <h1>Claim under review</h1>
      <div class="claim">{claim_html}</div>
      <div class="clip">clip&nbsp;&nbsp;{_e(pack.clip)}</div>
      <h1>Evidence — {len(evidence)} signals</h1>
      {evidence_html}
      <h1>Adjudicator narrative</h1>
      <p class="narrative">{_e(adj.narrative)}</p>
      <details><summary>Raw evidence pack</summary><pre>{raw_json}</pre></details>
    </div>
    {rail_html}
  </div>
  <footer>Sundial &middot; VAST Builders Challenge</footer>
</div>
<script>
// verdict scramble-in
(function() {{
  var el = document.getElementById("verdictTxt"), final = el.dataset.final,
      chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  el.textContent = final;
  var frame = 0, total = 18;
  function tick() {{
    frame++;
    var out = "";
    for (var i = 0; i < final.length; i++) {{
      out += (frame / total > i / final.length)
        ? final[i]
        : chars[Math.floor(Math.random() * chars.length)];
    }}
    el.textContent = out;
    if (frame < total) requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}})();
// confidence count-up + ring
(function() {{
  var num = document.getElementById("confNum"), target = +num.dataset.target,
      ring = document.getElementById("ringFg"), t0 = null;
  requestAnimationFrame(function() {{
    ring.style.strokeDashoffset = (226.2 * (1 - target / 100)).toFixed(1);
  }});
  function step(ts) {{
    if (!t0) t0 = ts;
    var p = Math.min(1, (ts - t0) / 1000);
    num.textContent = Math.round(target * p) + "%";
    if (p < 1) requestAnimationFrame(step);
  }}
  requestAnimationFrame(step);
}})();
// confidence bars fill once visible (bars are pre-rendered at their final
// width so the page still reads correctly with JS disabled)
try {{
  if (typeof IntersectionObserver !== "undefined") {{
    var io = new IntersectionObserver(function(entries) {{
      entries.forEach(function(en) {{
        if (en.isIntersecting) {{
          var bar = en.target.querySelector(".conffill");
          if (bar) bar.style.width = bar.dataset.w + "%";
          io.unobserve(en.target);
        }}
      }});
    }}, {{threshold: .25}});
    document.querySelectorAll(".ev").forEach(function(el) {{
      var bar = el.querySelector(".conffill");
      if (bar) bar.style.width = "0%";
      io.observe(el);
    }});
  }}
}} catch (e) {{ /* animation is cosmetic — the report must still read */ }}
// spotlight tracks the cursor
document.addEventListener("mousemove", function(e) {{
  var card = document.getElementById("spot"), r = card.getBoundingClientRect();
  card.style.setProperty("--mx", ((e.clientX - r.left) / r.width * 100) + "%");
  card.style.setProperty("--my", ((e.clientY - r.top) / r.height * 100) + "%");
  document.querySelectorAll(".ev, .railpanel").forEach(function(el) {{
    var er = el.getBoundingClientRect();
    el.style.setProperty("--mx", (e.clientX - er.left) + "px");
    el.style.setProperty("--my", (e.clientY - er.top) + "px");
  }});
}});
</script>
</body></html>"""

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path
