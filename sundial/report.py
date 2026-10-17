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


#: Evidence-card verdict wording — plain English instead of enum names.
_PILL_WORD = {
    ClaimVerdict.SUPPORTS: "supports",
    ClaimVerdict.CONTRADICTS: "against",
    ClaimVerdict.INCONCLUSIVE: "unclear",
}

#: Evidence type -> human label. Anything unmapped falls back to a
#: title-cased, de-snake-cased version of the type string.
_TYPE_LABEL = {
    "clock": "On-screen clock",
    "metadata": "File metadata",
    "video_meta": "File details",
    "camera_meta": "Camera details",
    "cam_model": "Device fingerprint",
    "exif_gps": "Embedded GPS tag",
    "ocr": "Text in the video",
    "ocr_boost": "Text in the video",
    "ocr_dates": "Dates in the video",
    "solar": "Sun position",
    "sun_times": "Daylight check",
    "suncalc": "Sun position",
    "shadow": "Shadow direction",
    "shadow_geo": "Shadow direction",
    "weather": "Weather records",
    "moonphase": "Moon phase",
    "tz_check": "Time zone check",
    "claim_tz": "Time zone check",
    "geo": "Location lookup",
    "geo-resolve": "Location lookup",
    "geocell_pred": "Region estimate",
    "geo_pred": "Region estimate",
    "geo_matrix": "Location clues",
    "geo_tables": "Location clues",
    "geotips": "Location clues",
    "region_vocab": "Location clues",
    "pip2cc": "Country check",
    "hier_geoguess": "Country guess",
    "clip_geo": "Scene match",
    "geo_clip": "Model estimate",
    "geo_ensemble": "Combined location estimate",
    "geo_matrix": "Cross-checked location",
    "street_cues": "Street and road clues",
    "plate_ocr": "License plates",
    "wiki": "Wikipedia check",
    "wiki_places": "Nearby landmarks",
    "overpass": "Map data check",
    "osm_context": "Street map check",
    "osint": "Clues from the footage",
    "osint_chain": "Clue chain",
    "agent_transcript": "Investigation steps",
    "agent_loop": "Investigation",
    "transcript": "Investigation steps",
    "vlm": "Visual analysis",
    "perception": "Visual analysis",
    "landmark": "Landmark match",
    "consistency": "Consistency check",
    "audio": "Audio clues",
    "frames": "Video frames",
    "language": "Language clues",
    "geo_match": "Location match",
    "geo_rollup": "Location summary",
    "geoguess": "Country guess",
    "hemisphere": "Hemisphere clues",
    "lunar": "Moon check",
    "climate": "Climate clues",
    "poi": "Nearby places",
    "scene": "Scene analysis",
    "shadow_consistency": "Shadow consistency",
    "shadows": "Shadow direction",
    "sign_style": "Road signs",
    "plate_style": "License plates",
    "osint_features": "Clues from the footage",
    "texture": "Image texture",
    "timezone": "Time zone check",
    "weather+solar": "Weather and sun",
    "corpus": "Reference data",
    "cam_model": "Device fingerprint",
    "exif_gps": "Embedded GPS tag",
    "osm_context": "Street map check",
    "unavailable": "Couldn't check",
}

#: 2-letter country codes seen in ranked rows -> display names. Codes
#: missing here render as the code itself (still better than nothing).
_CC_NAME = {
    "DE": "Germany", "FR": "France", "GB": "United Kingdom",
    "UK": "United Kingdom", "IT": "Italy", "ES": "Spain", "PT": "Portugal",
    "NL": "Netherlands", "BE": "Belgium", "CH": "Switzerland",
    "AT": "Austria", "PL": "Poland", "CZ": "Czechia", "SE": "Sweden",
    "NO": "Norway", "DK": "Denmark", "FI": "Finland", "IE": "Ireland",
    "GR": "Greece", "US": "United States", "CA": "Canada", "MX": "Mexico",
    "BR": "Brazil", "AR": "Argentina", "JP": "Japan", "KR": "South Korea",
    "CN": "China", "IN": "India", "AU": "Australia", "NZ": "New Zealand",
    "ZA": "South Africa", "EG": "Egypt", "TR": "Türkiye", "RU": "Russia",
    "UA": "Ukraine", "IS": "Iceland", "LU": "Luxembourg",
    "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria", "HR": "Croatia",
    "SI": "Slovenia", "SK": "Slovakia", "EE": "Estonia", "LV": "Latvia",
    "LT": "Lithuania", "MT": "Malta", "CY": "Cyprus",
}

#: Short chip text for the verdict when checking a claim.
_VERDICT_CHIP = {
    Verdict.SUPPORTED: "Matches your claim",
    Verdict.CONTRADICTED: "Doesn’t match your claim",
    Verdict.INSUFFICIENT: "Can’t confirm",
}

#: One-line plain-English read of the headline verdict.
_VERDICT_SUB = {
    Verdict.SUPPORTED: "the claim checks out",
    Verdict.CONTRADICTED: "the claim doesn’t add up",
    Verdict.INSUFFICIENT: "not enough to tell",
}


def _type_label(t: Any) -> str:
    """Human label for an evidence type string."""
    s = str(t or "").strip()
    if s in _TYPE_LABEL:
        return _TYPE_LABEL[s]
    return s.replace("_", " ").replace("-", " ").title() or "Signal"


def _pill(verdict: Any) -> str:
    cv = _cv(verdict)
    fg, bg, border = _PILL[cv]
    word = _PILL_WORD.get(cv, _verdict_str(verdict))
    return (
        f'<span class="pill" style="color:{fg};background:{bg};'
        f'border:1px solid {border}">{_e(word)}</span>'
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


def _human_summary(ev: "Evidence") -> str:
    """Clean internal pipeline jargon out of card summaries for display."""
    s = str(getattr(ev, "summary", "") or "").strip()
    t = str(getattr(ev, "type", "") or "").strip().lower()
    if t == "agent_transcript" and s.startswith("agent ran "):
        d = getattr(ev, "data", None) or {}
        steps = d.get("transcript") if isinstance(d, dict) else []
        ok_n = sum(1 for x in (steps or []) if isinstance(x, dict) and x.get("ok", True))
        n = len(steps or []) or ok_n
        if ok_n:
            return (f"ran {ok_n} background verification check"
                    f"{'s' if ok_n != 1 else ''} on the location and map records")
        return f"ran {n} background check{'s' if n != 1 else ''}"
    return s


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
        n = len(ev.frames)
        frames = (f'<div class="frames">seen in {n} frame'
                  f'{"s" if n != 1 else ""}</div>')
    return (
        f'<div class="ev" style="animation-delay:{120 + idx * 90}ms">'
        f'<div class="evhead"><span class="evicon" style="color:{fg}">{icon}</span>'
        f'<span class="etype">{_e(_type_label(ev.type))}</span>{_pill(ev.verdict)}'
        f'<span class="econfnum">{pct:.0f}%</span></div>'
        f'<div class="confbar"><div class="conffill" data-w="{pct:.0f}" '
        f'style="background:{fg};width:{pct:.0f}%"></div></div>'
        f'<div class="evsum">{_e(_human_summary(ev))}</div>{frames}{_frame_viz(ev)}</div>'
    )


#: Summaries that mean "this stage never ran" — tools, deps, or inputs
#: missing. These get collapsed into one quiet row instead of a wall of
#: dead cards (a normal user reads them as "everything's broken").
_UNAVAILABLE_HINT = (
    "unavailable", "not set", "install ", "no usable", "no readable",
    "missing", "skipped", "lookup failed", "offline",
)


def _collapse_unavailable(evs: list) -> list:
    """Group inconclusive 'stage didn't run' cards into one quiet row."""
    keep, gone = [], []
    for e in evs:
        s = str(getattr(e, "summary", "") or "").lower()
        if (_cv(getattr(e, "verdict", None)) is ClaimVerdict.INCONCLUSIVE
                and any(h in s for h in _UNAVAILABLE_HINT)):
            gone.append(e)
        else:
            keep.append(e)
    if gone:
        names = ", ".join(
            dict.fromkeys(_type_label(getattr(e, "type", "")) for e in gone))
        try:
            from .schema import Evidence as _Ev
            keep.append(_Ev(
                type="unavailable",
                summary=(f"couldn't check {len(gone)} thing"
                         f"{'s' if len(gone) != 1 else ''} — "
                         f"{names}: tools or inputs missing on this machine"),
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.05,
                data={"collapsed_unavailable": True}))
        except Exception:
            keep.extend(gone)
    return keep


def _dedupe_types(evs: list, types: set) -> list:
    """Keep only the first card of each given type — compact_for_display
    emits a geo_rollup digest while the pack may already carry one."""
    seen: set = set()
    out = []
    for e in evs:
        t = getattr(e, "type", "")
        if t in types:
            if t in seen:
                continue
            seen.add(t)
        out.append(e)
    return out


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

    # 2. location readout — a screen-reader text version of the map (the
    #    visible place name lives in the side panel, not on top of the map)
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
                num = "" if sc is None else (
                    f'<span class="locscore">{sc * 100:.0f}%</span>')
                disp = _CC_NAME.get(str(name).upper(), str(name))
                lis.append(
                    f'<div class="locrow"><span class="loccc">{_e(disp)}</span>'
                    f'<span class="locbar"><span class="locfill" '
                    f'style="width:{w:.0f}%"></span></span>{num}</div>')
            bits.append('<div class="loclist">' + "".join(lis) + "</div>")
        if bits:
            loc_block = '<div class="locread sr">' + "".join(bits) + "</div>"

    # 3. reasoning chain fragment, if the caller produced one
    chain_block = ""
    if isinstance(chain, str) and chain.strip():
        chain_block = ('<div class="sr"><div class="chainbox"><div class="railsub">'
                       'why we think so</div>' + chain + "</div></div>")

    if has_geo:
        inner = map_block + loc_block + chain_block
    else:
        inner = ('<div class="nogeo"><span class="nogeoicon">' + _ICONS["geo"]
                 + '</span><span class="nogeohead">no location guess yet</span>'
                 '<span class="nogeosub">Try a clearer outdoor photo, or say '
                 'where you think it was taken.</span></div>')

    foot = f'<div class="railfoot sr">{mix_html}</div>' if mix_html else ""
    return ('<aside class="rail"><div class="railpanel">'
            '<div class="railtitle sr">Where it was likely filmed</div>'
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
        claim_bits.append(f'<span class="ck">when</span><b>{_e(claim.claimed_time)}</b>')
    if claim.claimed_place:
        claim_bits.append(f'<span class="ck">where</span><b>{_e(claim.claimed_place)}</b>')
    claim_html = '<span class="cdot">·</span>'.join(claim_bits) or "<i>no claim recorded</i>"
    if claim.raw:
        claim_html += f'<div class="craw">&ldquo;{_e(claim.raw)}&rdquo;</div>'
    # no claim at all = guess mode: the page should read like a location
    # estimate, not a failed verification
    guess_mode = not (claim.claimed_time or claim.claimed_place
                      or str(claim.raw or "").strip())
    claim_head = "The task" if guess_mode else "The claim"
    if guess_mode:
        claim_html = "<i>work out where this was filmed</i>"

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
    evidence = _dedupe_types(evidence, {"geo_rollup"})
    evidence = _collapse_unavailable(evidence)
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

    raw_json = html.escape(json.dumps(adj.to_dict(), indent=2, default=str))

    rail_html = _loc_rail(extra, "")

    # --- the detected place, shown big in the side panel ---------------------
    ex_loc = (extra.get("loc_summary") if isinstance(extra, dict) else None) or {}
    if not isinstance(ex_loc, dict):
        ex_loc = {}
    # only evidence may name the place — never echo the user's claim back
    raw_hp = ex_loc.get("best_name") or ""
    if isinstance(raw_hp, dict):
        raw_hp = raw_hp.get("name") or raw_hp.get("cc") or ""
    hero_place = str(raw_hp or "").strip()
    pt_src = ex_loc.get("predicted") or {}
    if not isinstance(pt_src, dict):
        pt_src = {}
    coord_txt, copy_txt, radius_txt = "", "", ""
    gmap_q = hero_place
    try:
        la_f, lo_f = float(pt_src.get("lat")), float(pt_src.get("lon"))
        ns, ew = ("N" if la_f >= 0 else "S"), ("E" if lo_f >= 0 else "W")
        coord_txt = f"{abs(la_f):.4f}° {ns}, {abs(lo_f):.4f}° {ew}"
        copy_txt = f"{la_f:.5f}, {lo_f:.5f}"
        gmap_q = f"{la_f:.5f},{lo_f:.5f}"
        r_km = pt_src.get("radius_km") or pt_src.get("r_km")
        if r_km:
            radius_txt = f"within about {float(r_km):.0f} km"
    except (TypeError, ValueError):
        pass

    # location confidence: in guess mode the place score is what matters;
    # when checking a claim it is the verdict confidence
    shown_pct = conf_pct
    if guess_mode:
        try:
            top = (_ranked_rows(ex_loc.get("ranked")) or [(None, None)])[0][1]
            if top is not None:
                shown_pct = max(0.0, min(1.0, float(top))) * 100
        except (TypeError, ValueError, IndexError):
            pass
    if shown_pct >= 70:
        conf_word = "High confidence"
    elif shown_pct >= 45:
        conf_word = "Medium confidence"
    else:
        conf_word = "Low confidence"

    if hero_place:
        place_html = f'<h2 class="placename">{_e(hero_place)}</h2>'
        meta = " · ".join(x for x in (coord_txt, radius_txt) if x)
        place_html += f'<div class="placemeta">{_e(meta)}</div>' if meta else ""
    else:
        place_html = ('<h2 class="placename dim">Location unknown</h2>'
                      '<div class="placemeta plain">Nothing in the footage pinned '
                      'down where it was taken.</div>')

    actions = []
    if gmap_q:
        actions.append(
            f'<a class="btn primary" id="gmapBtn" data-q="{_e(gmap_q)}" '
            f'target="_blank" rel="noopener">{_ICONS["geo"]}Open in Google Maps</a>')
    if copy_txt:
        actions.append(
            f'<button class="btn" type="button" id="copyBtn" '
            f'data-c="{_e(copy_txt)}">Copy coordinates</button>')
    actions_html = ('<div class="actions">' + "".join(actions) + "</div>"
                    if actions else "")

    conf_html = (
        '<div class="conf">'
        f'<div class="confhead"><span>{_e(conf_word)}</span>'
        f'<span class="confnum" id="confNum" data-target="{shown_pct:.0f}">'
        f'{shown_pct:.0f}%</span></div>'
        f'<div class="cbar"><div class="cfill" id="cFill" data-w="{shown_pct:.0f}" '
        f'style="width:{shown_pct:.0f}%"></div></div></div>'
    )

    chip_word = ("Best guess" if guess_mode
                 else _VERDICT_CHIP.get(verdict, verdict.value.title()))
    sub_line = ("no claim to check — best guess is on the map" if guess_mode
                else _VERDICT_SUB.get(verdict, ""))
    claim_said = " · ".join(
        str(x) for x in (claim.claimed_place, claim.claimed_time) if x)
    if not claim_said and not guess_mode:
        claim_said = str(claim.raw or "")
    claim_box = (
        '<div class="claimbox">'
        + ('' if guess_mode else
           f'<div class="kicker">You said</div>'
           f'<div class="said">{_e(claim_said)}</div>')
        + f'<div class="verdictline"><span class="badge" id="verdictTxt" '
          f'data-final="{_e(verdict.value)}">{_e(chip_word)}</span>'
          f'<span class="vsub">{_e(sub_line)}</span></div>'
        + '</div>'
    )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sundial — {_e(verdict.value)}</title>
<style>
:root {{ --accent:{accent}; --line:#1a1a1a; --mute:#8a8a8a; }}
* {{ box-sizing:border-box; }}
html, body {{ height:100%; }}
body {{ margin:0; background:#000; color:#ededed;
       font-family:-apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Segoe UI", Roboto, sans-serif;
       -webkit-font-smoothing:antialiased; }}
.wrap {{ height:100vh; display:flex; flex-direction:column; }}
.top {{ height:56px; flex:none; display:flex; align-items:center; justify-content:space-between;
       padding:0 20px; border-bottom:1px solid var(--line); }}
.brand {{ font-size:15px; font-weight:600; letter-spacing:-.01em; color:#fff; }}
.brand span {{ color:var(--mute); font-weight:400; }}
.newbtn {{ color:#ededed; text-decoration:none; font-size:13px; padding:7px 14px;
          border:1px solid #2a2a2a; border-radius:999px; }}
.newbtn:hover {{ background:#111; }}
.layout {{ flex:1; min-height:0; display:grid; grid-template-columns:400px minmax(0,1fr); }}
.card {{ overflow-y:auto; padding:30px 26px 40px; border-right:1px solid var(--line);
        background:#050505; opacity:0; transform:translateY(6px);
        animation:in .45s cubic-bezier(.2,.7,.2,1) .05s forwards; }}
@keyframes in {{ to {{ opacity:1; transform:none; }} }}
.kicker {{ font-size:12px; color:var(--mute); margin-bottom:6px; }}
.placename {{ font-size:30px; line-height:1.15; font-weight:650; letter-spacing:-.02em;
             margin:0; color:#fff; }}
.placename.dim {{ color:#777; }}
.placemeta.plain {{ font-family:inherit; line-height:1.5; }}
.placemeta {{ margin-top:8px; font-size:13px; color:var(--mute);
             font-family:ui-monospace, "SF Mono", Menlo, monospace; }}
.actions {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:20px; }}
.btn {{ display:inline-flex; align-items:center; gap:7px; font:inherit; font-size:13px;
       font-weight:500; padding:9px 14px; border-radius:999px; cursor:pointer;
       border:1px solid #2a2a2a; background:transparent; color:#ededed; text-decoration:none; }}
.btn:hover {{ background:#111; }}
.btn.primary {{ background:#fff; color:#000; border-color:#fff; }}
.btn.primary:hover {{ background:#e6e6e6; }}
.btn svg {{ width:15px; height:15px; fill:none; stroke:currentColor; stroke-width:1.8; }}
.conf {{ margin-top:28px; }}
.confhead {{ display:flex; justify-content:space-between; font-size:13px; color:#cfcfcf; }}
.confnum {{ font-variant-numeric:tabular-nums; color:#fff; font-weight:600; }}
.cbar {{ margin-top:8px; height:6px; border-radius:3px; background:#161616; overflow:hidden; }}
.cfill {{ height:100%; border-radius:3px; background:var(--accent);
         transition:width 1s cubic-bezier(.2,.7,.2,1); }}
.claimbox {{ margin-top:28px; padding-top:22px; border-top:1px solid var(--line); }}
.said {{ font-size:16px; color:#fff; margin-bottom:12px; }}
.verdictline {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.badge {{ display:inline-block; font-size:12px; font-weight:600; padding:5px 11px; border-radius:999px;
         color:var(--accent); background:color-mix(in srgb, var(--accent) 14%, transparent);
         border:1px solid color-mix(in srgb, var(--accent) 35%, transparent); }}
.vsub {{ font-size:13px; color:var(--mute); }}
.morebox {{ margin-top:30px; border-top:1px solid var(--line); padding-top:16px; }}
.morebox > summary {{ cursor:pointer; list-style:none; font-size:13px; color:#bdbdbd; }}
.morebox > summary::-webkit-details-marker {{ display:none; }}
.morebox > summary::after {{ content:" +"; color:#666; }}
.morebox[open] > summary::after {{ content:" –"; }}
h1 {{ font-size:12px; font-weight:500; color:var(--mute); margin:22px 0 10px; }}
.claim {{ font-size:14px; display:flex; gap:8px; flex-wrap:wrap; align-items:baseline; }}
.ck {{ color:#666; font-size:12px; }}
.cdot {{ color:#333; }}
.craw {{ color:#999; font-size:13px; margin-top:6px; font-style:italic; width:100%; }}
.evgrid {{ display:flex; flex-direction:column; gap:8px; }}
.ev {{ border:1px solid var(--line); border-radius:12px; padding:12px 14px; background:#0a0a0a; }}
.evhead {{ display:flex; align-items:center; gap:9px; }}
.evicon svg {{ width:16px; height:16px; fill:none; stroke:currentColor; stroke-width:1.6; }}
.etype {{ font-weight:600; font-size:13px; color:#e5e5e5; }}
.pill {{ margin-left:auto; border-radius:999px; padding:2px 9px; font-size:11px; font-weight:600; }}
.econfnum {{ font-size:12px; color:#888; min-width:4ch; text-align:right; }}
.confbar {{ height:3px; background:#161616; border-radius:2px; margin:9px 0 8px; overflow:hidden; }}
.conffill {{ height:100%; border-radius:2px; }}
.evsum {{ font-size:13px; line-height:1.5; color:#cfcfcf; }}
.frames {{ color:#555; font-size:11px; margin-top:6px; }}
.sunrow {{ display:flex; justify-content:center; }}
.sunviz {{ width:100%; max-width:460px; }}
.frameviz {{ width:100%; border-radius:8px; margin-top:10px; display:block; }}
.svlabel {{ font:11px ui-monospace,monospace; fill:#999; }}
.narrative {{ font-size:14px; line-height:1.6; color:#dcdcdc; margin:0; }}
details details {{ margin-top:14px; }}
details details > summary {{ cursor:pointer; color:#666; font-size:12px; }}
pre {{ font-size:11px; color:#999; background:#0a0a0a; border:1px solid var(--line);
      border-radius:10px; padding:12px; overflow:auto; max-height:300px; }}
.empty {{ color:#555; font-size:13px; }}
.rail {{ position:relative; min-height:0; background:#0a0d12; }}
.railpanel {{ position:absolute; inset:0; overflow:hidden; }}
.mapbox {{ position:absolute; inset:0; background:#0a0d12; }}
.mapbox > .smap, .mapbox > .vmap {{ width:100%!important; height:100%!important;
  border:0!important; border-radius:0!important; }}
.mapbox img.mapimg {{ width:100%; height:100%; object-fit:cover; display:block; }}
.mapbox svg {{ display:block; width:100%; height:100%; }}
.mapnone, .nogeo {{ position:absolute; inset:0; display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:10px; color:#666; font-size:13px;
  text-align:center; padding:24px; }}
.nogeoicon svg {{ width:30px; height:30px; fill:none; stroke:#3a3a3a; stroke-width:1.6; }}
.nogeohead {{ color:#bbb; font-size:15px; }}
.nogeohead::first-letter {{ text-transform:uppercase; }}
.nogeosub {{ color:#666; max-width:300px; line-height:1.5; }}
.sr {{ position:absolute!important; width:1px; height:1px; padding:0; margin:-1px;
      overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; border:0; }}
@media (max-width:900px) {{
  .wrap {{ height:auto; min-height:100vh; }}
  .layout {{ grid-template-columns:minmax(0,1fr); }}
  .rail {{ position:static; order:-1; }}
  .railpanel {{ position:relative; height:58vh; }}
  .card {{ border-right:0; overflow:visible; padding:24px 20px 36px; }}
}}
</style></head>
<body>
<div class="wrap">
  <header class="top">
    <div class="brand">Sundial <span>&middot; where was this filmed?</span></div>
    <a class="newbtn" href="/">New check</a>
  </header>
  <div class="layout">
    <div class="card" id="spot">
      <div class="kicker">{"Most likely filmed in" if hero_place else "Detected location"}</div>
      {place_html}
      {actions_html}
      {conf_html}
      {claim_box}
      <details class="morebox"><summary>How we worked it out</summary>
        <h1>{claim_head}</h1>
        <div class="claim">{claim_html}</div>
        <h1>What we checked</h1>
        {evidence_html}
        <h1>What it means</h1>
        <p class="narrative">{_e(adj.narrative)}</p>
        <details><summary>Technical details</summary><pre>{raw_json}</pre></details>
      </details>
    </div>
    {rail_html}
  </div>
</div>
<script>
(function() {{
  var gb = document.getElementById("gmapBtn");
  if (gb && gb.dataset.q) {{
    gb.href = "https://" + "www.google.com/maps/search/?api=1&query=" + encodeURIComponent(gb.dataset.q);
  }}
  var cb = document.getElementById("copyBtn");
  if (cb) {{
    cb.addEventListener("click", function() {{
      var t = cb.dataset.c, done = function() {{
        cb.textContent = "Copied";
        setTimeout(function() {{ cb.textContent = "Copy coordinates"; }}, 1400);
      }};
      if (navigator.clipboard) navigator.clipboard.writeText(t).then(done, done);
      else done();
    }});
  }}
  // confidence count-up + bar fill (pre-rendered at final values without JS)
  var num = document.getElementById("confNum"), fill = document.getElementById("cFill");
  if (!num) return;
  var target = +num.dataset.target, t0 = null;
  if (fill) fill.style.width = "0%";
  requestAnimationFrame(function() {{ if (fill) fill.style.width = target + "%"; }});
  function step(ts) {{
    if (!t0) t0 = ts;
    var p = Math.min(1, (ts - t0) / 900);
    num.textContent = Math.round(target * p) + "%";
    if (p < 1) requestAnimationFrame(step);
  }}
  requestAnimationFrame(step);
}})();
</script>
</body></html>"""

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path
