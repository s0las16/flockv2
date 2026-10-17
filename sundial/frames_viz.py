"""Annotated frame-strip helpers for reports and pipeline debug output.

Three small utilities, all fail-soft:

- `thumb_data_url` renders one frame to a base64 JPEG data-URL using the
  same crop/downscale/q62 recipe as report.py's `_frame_viz`.
- `contact_strip` lays several frames out as an inline-styled HTML flex
  row with optional captions — drop-in for evidence/debug pages.
- `annotation_svg` emits SVG overlay elements (shadow vectors, boxes drawn
  as lines, labels) in image coordinates, meant to sit next to an
  ``<image>`` inside a single ``<svg>`` like report.py's frameviz.

PIL is imported lazily; every helper returns None / "" on any failure —
imagery is a nice-to-have, never fatal.
"""
from __future__ import annotations

import base64
import html
import io
import os
from typing import Any

_MAX_W = 620  # report.py's frameviz max width
_JPEG_Q = 62  # report.py's thumbnail JPEG quality
_STRIP_THUMB_W = 300  # per-frame width cap inside a contact strip
_ACCENT = "#facc15"  # measurement overlay color used across report.py
_TEXT = "#94a3b8"  # svlabel color


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def thumb_data_url(
    frame_path: str,
    max_w: int = _MAX_W,
    crop_fraction: tuple[float, float] | None = None,
) -> tuple[str, int, int] | None:
    """Render one frame to a JPEG data-URL; returns (url, w, h) or None.

    `crop_fraction` is a vertical (y0, y1) slice of the frame height —
    (0.0, 0.35) reproduces the top strip report.py shows for 'clock'
    overlays. The result is downscaled so width <= max_w (never upscaled)
    and encoded as JPEG quality 62. Any failure — missing file, no PIL,
    decode error — returns None.
    """
    try:
        if not frame_path or not os.path.isfile(str(frame_path)):
            return None
        from PIL import Image

        with Image.open(str(frame_path)) as src:
            img = src.convert("RGB")
        w0, h0 = img.size
        if crop_fraction is not None:
            y0 = min(max(float(crop_fraction[0]), 0.0), 1.0)
            y1 = min(max(float(crop_fraction[1]), 0.0), 1.0)
            top = min(round(h0 * y0), h0 - 1)
            bot = min(h0, max(top + 1, round(h0 * y1)))
            img = img.crop((0, top, w0, bot))
            w0, h0 = img.size
        scale = min(1.0, float(max_w) / w0)
        w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        if (w, h) != img.size:
            img = img.resize((w, h))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=_JPEG_Q)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return url, w, h
    except Exception:
        return None


def contact_strip(
    frames: list[str],
    labels: list[str] | None = None,
    max_per: int = 4,
) -> str:
    """HTML fragment: a flex row of up to `max_per` frame thumbnails.

    Each thumbnail comes from `thumb_data_url`; frames that fail are
    skipped (taking their label with them). When `labels` is given it is
    aligned to `frames` by index and rendered as a <figcaption>. Returns
    "" when nothing renders. All styling is inline — dark theme matching
    report.py (#070b14, rgba(148,163,184,.14) borders, mono, 8px radius).
    """
    figs = []
    try:
        limit = max(0, int(max_per))
    except (TypeError, ValueError):
        limit = 4
    for i, frame in enumerate((frames or [])[:limit]):
        thumb = thumb_data_url(frame, max_w=_STRIP_THUMB_W)
        if thumb is None:
            continue
        url, w, h = thumb
        cap = ""
        if labels is not None and i < len(labels) and labels[i] is not None:
            cap = (
                '<figcaption style="font:10px ui-monospace,monospace;'
                f"color:#64748b;margin-top:4px;text-align:center;max-width:{w}px;"
                'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                f"{_e(labels[i])}</figcaption>"
            )
        figs.append(
            '<figure style="margin:0;display:flex;flex-direction:column;'
            'align-items:center">'
            f'<img src="{url}" width="{w}" height="{h}" '
            f'alt="{_e(os.path.basename(str(frame)))}" '
            'style="display:block;border:1px solid rgba(148,163,184,.14);'
            'border-radius:8px;background:#070b14"/>'
            f"{cap}</figure>"
        )
    if not figs:
        return ""
    return (
        '<div class="framestrip" style="display:flex;flex-wrap:wrap;gap:10px;'
        "align-items:flex-start;background:#070b14;border:1px solid "
        'rgba(148,163,184,.14);border-radius:8px;padding:10px">'
        + "".join(figs)
        + "</div>"
    )


def annotation_svg(w: float, h: float, shapes: list[dict]) -> str:
    """SVG overlay elements (no outer <svg>) in image coordinates.

    Supported shapes:

    - ``{"kind": "line",   "x1","y1","x2","y2", "color"}`` — shadow vectors
    - ``{"kind": "circle", "cx","cy","r","color","fill"}`` — tip/base dots,
      overlay rings; falsy `fill` means open circle
    - ``{"kind": "text",   "x","y","text","color"}`` — mono labels

    Coordinates are clamped to the w×h frame so overlays can never escape
    the embedded image; `color` defaults to the report's measurement
    accent. Malformed or unknown shapes are skipped silently.
    """
    try:
        W, H = float(w), float(h)
    except (TypeError, ValueError):
        return ""

    def _cx(v: Any) -> float:
        return min(max(float(v), 0.0), W)

    def _cy(v: Any) -> float:
        return min(max(float(v), 0.0), H)

    parts = []
    for s in shapes or []:
        if not isinstance(s, dict):
            continue
        try:
            kind = s.get("kind")
            color = _e(s.get("color") or _ACCENT)
            if kind == "line":
                parts.append(
                    f'<line x1="{_cx(s["x1"]):.0f}" y1="{_cy(s["y1"]):.0f}" '
                    f'x2="{_cx(s["x2"]):.0f}" y2="{_cy(s["y2"]):.0f}" '
                    f'stroke="{color}" stroke-width="2.5" opacity="0.95"/>'
                )
            elif kind == "circle":
                fill = s.get("fill")
                parts.append(
                    f'<circle cx="{_cx(s["cx"]):.0f}" cy="{_cy(s["cy"]):.0f}" '
                    f'r="{max(0.5, float(s["r"])):.1f}" '
                    f'fill="{_e(fill) if fill else "none"}" '
                    f'stroke="{color}" stroke-width="2"/>'
                )
            elif kind == "text":
                tcolor = _e(s.get("color") or _TEXT)
                parts.append(
                    f'<text x="{_cx(s["x"]):.0f}" y="{_cy(s["y"]):.0f}" '
                    f'fill="{tcolor}" font-family="ui-monospace,monospace" '
                    f'font-size="11">{_e(s.get("text"))}</text>'
                )
        except (KeyError, TypeError, ValueError):
            continue
    return "".join(parts)


if __name__ == "__main__":
    # Self-test: build a contact strip from any jpg on disk, else skip.
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg")):
                found.append(os.path.join(root, f))
        if len(found) >= 4:
            break
    if not found:
        print("frames_viz self-test: no .jpg found under", root_dir, "— skipping")
    else:
        picks = found[:4]
        strip = contact_strip(picks, labels=[os.path.basename(p) for p in picks])
        ov = annotation_svg(300, 200, [
            {"kind": "line", "x1": 20, "y1": 180, "x2": 140, "y2": 90},
            {"kind": "circle", "cx": 20, "cy": 180, "r": 5, "fill": "#facc15"},
            {"kind": "circle", "cx": 140, "cy": 90, "r": 5},
            {"kind": "text", "x": 148, "y": 86, "text": "shadow"},
        ])
        print(f"contact_strip: {len(picks)} frame(s) -> {len(strip)} bytes of HTML")
        print(f"annotation_svg: {len(ov)} bytes -> {ov[:100]}...")
        assert strip and "<figure" in strip and ov.startswith("<line")
        print("frames_viz self-test: OK")
