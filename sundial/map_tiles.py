"""Static dark-map renderer for offline / JS-disabled reports.

``static_map_b64`` composes a CARTO ``dark_all`` raster basemap server-side
and returns a ``data:image/png;base64,...`` URL a report can inline as an
``<img>`` — the verdict page then needs neither JavaScript nor a live
network on the reader's side.

Tiles come from ``{s}.basemaps.cartocdn.com`` (free, no key; attribution
required — a small "© OSM © CARTO" strip is burned into the corner). A 3x3
tile grid (<=9 requests, sequential, 8s timeout each) is fetched around the
anchor's slippy tile, the anchor is centered, then markers and band rects
are drawn on an RGBA overlay:

  * ``kind="claim"`` — white ring + white fill (the asserted location)
  * ``kind="pred"``  — #facc15 ring (model/geocoder candidate)
  * ``kind="geo"``   — #38bdf8 dot (extracted GPS)
  * ``kind="wiki"``  — #94a3b8 dot (Wikipedia place)
  * ``bands``        — lat/lon rects, translucent fill + border

Fail-soft everywhere: PIL and HTTP are imported lazily (requests first,
urllib fallback — house style), any tile that fails leaves a dark gap in
the composite, and any failure at all — no PIL, no network, bad anchor —
returns "" rather than raising.

Self-test::

    python3 sundial/map_tiles.py            # offline ""-path asserts
    SUNDIAL_NET=1 python3 sundial/map_tiles.py   # + live render to /tmp
"""
from __future__ import annotations

import base64
import io
import math

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}")
SUBDOMAINS = ""  # Esri has no subdomains; param kept for call-shape compat
USER_AGENT = "sundial-hackathon/0.1"
HTTP_TIMEOUT_S = 8
TILE_PX = 256
GRID = 3            # 3x3 tiles centered on the anchor tile -> <=9 requests
MAX_ZOOM = 16
_MAX_MERC_LAT = 85.05112878  # web-mercator clip latitude

_BG = (7, 11, 20)           # report.py's #070b14 — gaps, letterboxing
_CLAIM = "#ffffff"
_PRED = "#facc15"           # report accent
_GEO = "#38bdf8"
_WIKI = "#94a3b8"
_CAPTION = "Esri © OSM contributors"
_CAPTION_BG = (7, 11, 20, 200)
_CAPTION_FG = (148, 163, 184, 230)

# Band dict key conventions accepted by _band_ll, in probe order.
_BAND_KEYS = (
    ("lat0", "lon0", "lat1", "lon1"),
    ("south", "west", "north", "east"),
    ("min_lat", "min_lon", "max_lat", "max_lon"),
    ("s", "w", "n", "e"),
)


def _lazy_pil():
    """(Image, ImageDraw) if Pillow is importable, else (None, None)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None, None
    return Image, ImageDraw


def _hex_rgba(color, alpha: int = 255):
    """'#facc15'/'#fc1' -> (r, g, b, alpha); None on unparseable input."""
    s = str(color or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), alpha)
    except ValueError:
        return None


def _latlon(lat, lon):
    """(lat, lon) floats in valid ranges, else None."""
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return None
    return la, lo


def _anchor_ll(anchor, markers):
    """Resolve the center point: valid anchor, else first valid marker."""
    if isinstance(anchor, dict):
        ll = _latlon(anchor.get("lat"), anchor.get("lon", anchor.get("lng")))
        if ll:
            return ll
    elif isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
        ll = _latlon(anchor[0], anchor[1])
        if ll:
            return ll
    for m in markers or ():
        if isinstance(m, dict):
            ll = _latlon(m.get("lat"), m.get("lon", m.get("lng")))
            if ll:
                return ll
    return None


def _deg2tile(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Fractional slippy tile coords (web mercator) — x,y in tile units."""
    n = 1 << z
    x = (lon + 180.0) / 360.0 * n
    la = math.radians(min(max(lat, -_MAX_MERC_LAT), _MAX_MERC_LAT))
    y = (1.0 - math.log(math.tan(la) + 1.0 / math.cos(la)) / math.pi) / 2.0 * n
    return x, y


def _fetch_tile(z: int, x: int, y: int, s: str) -> bytes | None:
    """One tile's PNG bytes, or None. requests if importable, else urllib."""
    url = TILE_URL.format(z=z, x=x, y=y)
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        try:
            resp = requests.get(
                url, headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT_S,
            )
            if resp.status_code == 200 and resp.content:
                return resp.content
        except Exception:
            pass  # fall through to urllib — a requests-only failure
            # (proxy, TLS) may not affect it
    try:
        import urllib.request
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return resp.read()
    except Exception:
        return None


def _band_ll(b: dict):
    """(lat_sw, lon_sw, lat_ne, lon_ne) from a band dict, else None."""
    if not isinstance(b, dict):
        return None
    for k0, k1, k2, k3 in _BAND_KEYS:
        if all(k in b for k in (k0, k1, k2, k3)):
            sw = _latlon(b[k0], b[k1])
            ne = _latlon(b[k2], b[k3])
            if sw and ne:
                return sw[0], sw[1], ne[0], ne[1]
    return None


def _draw_marker(draw, x: float, y: float, kind: str) -> None:
    """One marker glyph on the RGBA overlay; off-canvas coords just clip."""
    if kind == "claim":
        draw.ellipse((x - 9, y - 9, x + 9, y + 9),
                     outline=_hex_rgba(_CLAIM), width=2)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6),
                     fill=_hex_rgba(_CLAIM, 235))
    elif kind == "pred":
        draw.ellipse((x - 8, y - 8, x + 8, y + 8),
                     outline=_hex_rgba(_PRED), width=3)
    else:
        color = _GEO if kind == "geo" else _WIKI  # wiki + unknowns -> slate
        fill = _hex_rgba(color)
        if fill:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=fill,
                         outline=(*_BG, 255), width=1)


def _draw_caption(draw, w: int, h: int) -> None:
    """Attribution strip, bottom-right — required by CARTO/OSM tile terms."""
    try:
        x0, y0, x1, y1 = draw.textbbox((0, 0), _CAPTION)
        tw, th = x1 - x0, y1 - y0
    except Exception:
        tw, th = len(_CAPTION) * 6, 10
    pad, margin = 3, 4
    rx0, ry0 = w - tw - 2 * pad - margin, h - th - 2 * pad - margin
    draw.rectangle((rx0, ry0, w - margin, h - margin), fill=_CAPTION_BG)
    draw.text((rx0 + pad, ry0 + pad), _CAPTION, fill=_CAPTION_FG)


def static_map_b64(anchor: tuple | None, markers: list = None,
                   bands: list = None, zoom: int = 13,
                   size: tuple = (640, 400)) -> str:
    """Dark static map centered on `anchor` as a PNG data-URL, or "".

    `anchor` is (lat, lon); when None/unusable the first valid marker
    centers the map, and with no valid point at all "" is returned without
    touching the network. Tiles that fail to fetch leave a dark gap; if
    every tile fails the result is "". Never raises — a map is a
    nice-to-have for the report, never a reason to fail it.
    """
    try:
        Image, ImageDraw = _lazy_pil()
        if Image is None:
            return ""
        center = _anchor_ll(anchor, markers)
        if center is None:
            return ""
        try:
            z = min(max(int(zoom), 0), MAX_ZOOM)
        except (TypeError, ValueError):
            z = 13
        try:
            w = min(max(int(size[0]), 16), 2048)
            h = min(max(int(size[1]), 16), 2048)
        except (TypeError, ValueError, IndexError):
            w, h = 640, 400

        n = 1 << z
        tx_f, ty_f = _deg2tile(center[0], center[1], z)
        cx = min(max(math.floor(tx_f), 0), n - 1)
        cy = min(max(math.floor(ty_f), 0), n - 1)

        # 3x3 composite; longitude wraps, rows off the mercator world stay bg
        gw = gh = GRID * TILE_PX
        comp = Image.new("RGB", (gw, gh), _BG)
        fetched = 0
        for dy in range(GRID):
            for dx in range(GRID):
                x, y = cx - 1 + dx, cy - 1 + dy
                if y < 0 or y >= n:
                    continue
                blob = _fetch_tile(z, x % n, y, "")
                if not blob:
                    continue
                try:
                    tile = Image.open(io.BytesIO(blob)).convert("RGB")
                    comp.paste(tile.resize((TILE_PX, TILE_PX)),
                               (dx * TILE_PX, dy * TILE_PX))
                    fetched += 1
                except Exception:
                    continue
        if fetched == 0:
            return ""

        # crop a w×h window centered on the anchor (letterboxed if bigger
        # than the composite — only at tiny z / huge requested size)
        ox, oy = (cx - 1) * TILE_PX, (cy - 1) * TILE_PX
        ax, ay = tx_f * TILE_PX - ox, ty_f * TILE_PX - oy
        crop_w, crop_h = min(w, gw), min(h, gh)
        left = int(round(min(max(ax - w / 2.0, 0.0), gw - crop_w)))
        top = int(round(min(max(ay - h / 2.0, 0.0), gh - crop_h)))
        img = comp.crop((left, top, left + crop_w, top + crop_h))
        paste_x = paste_y = 0
        if (crop_w, crop_h) != (w, h):
            paste_x, paste_y = (w - crop_w) // 2, (h - crop_h) // 2
            canvas = Image.new("RGB", (w, h), _BG)
            canvas.paste(img, (paste_x, paste_y))
            img = canvas

        def _px(lat, lon):
            fx, fy = _deg2tile(lat, lon, z)
            return (fx * TILE_PX - ox - left + paste_x,
                    fy * TILE_PX - oy - top + paste_y)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        for b in bands or ():
            ll = _band_ll(b)
            if not ll:
                continue
            x0, y0 = _px(ll[0], ll[1])   # SW
            x1, y1 = _px(ll[2], ll[3])   # NE
            rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            col = _hex_rgba(b.get("color") or b.get("colour") or _PRED)
            if col:
                d.rectangle(rect, fill=col[:3] + (45,),
                            outline=col[:3] + (200,), width=2)

        for m in markers or ():
            if not isinstance(m, dict):
                continue
            ll = _latlon(m.get("lat"), m.get("lon", m.get("lng")))
            if not ll:
                continue
            x, y = _px(ll[0], ll[1])
            _draw_marker(d, x, y, str(m.get("kind") or ""))

        _draw_caption(d, w, h)
        img = Image.alpha_composite(img.convert("RGBA"), overlay)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(
            buf.getvalue()).decode("ascii")
    except Exception:
        return ""


if __name__ == "__main__":
    import os
    import sys

    print("map_tiles self-test (offline):")
    global_ok = [True]

    def check(label, cond):
        print(f"  {label}: {'ok' if cond else 'FAIL'}")
        if not cond:
            global_ok[0] = False

    # web-mercator math — pure stdlib, no deps at all
    fx, fy = _deg2tile(0.0, 0.0, 1)
    check("origin@z1 -> (1.0, 1.0)", (round(fx, 6), round(fy, 6)) == (1.0, 1.0))
    fx, fy = _deg2tile(51.5074, -0.1278, 13)
    check("london@z13 inside tile (4093,2724)",
          math.floor(fx) == 4093 and math.floor(fy) == 2724)

    # never-raise / ""-paths — must not touch the network
    check("anchor None -> ''", static_map_b64(None) == "")
    check("bad anchor -> ''", static_map_b64(("not", "coords")) == "")
    check("out-of-range -> ''", static_map_b64((200.0, 999.0)) == "")

    # deterministically dead-fetch path: stub the fetcher, still ""
    _orig_fetch = _fetch_tile
    globals()["_fetch_tile"] = lambda *a, **k: None
    try:
        check("all tiles fail -> ''",
              static_map_b64((51.5074, -0.1278),
                             markers=[{"lat": 51.5, "lon": -0.1,
                                       "kind": "geo"}]) == "")
    finally:
        globals()["_fetch_tile"] = _orig_fetch

    if os.environ.get("SUNDIAL_NET") == "1":
        print("\nlive render (SUNDIAL_NET=1):")
        url = static_map_b64(
            (51.5074, -0.1278), zoom=13,
            markers=[
                {"lat": 51.5074, "lon": -0.1278, "kind": "claim"},
                {"lat": 51.5180, "lon": -0.0950, "kind": "pred"},
                {"lat": 51.4990, "lon": -0.1420, "kind": "geo"},
                {"lat": 51.5210, "lon": -0.1600, "kind": "wiki"},
            ],
            bands=[{"lat0": 51.495, "lon0": -0.165,
                    "lat1": 51.525, "lon1": -0.095}],
        )
        if not url and _lazy_pil()[0] is None:
            print("  Pillow not installed — live render skipped")
        elif not url:
            print("  live fetch failed (offline?) — skipped")
        else:
            assert url.startswith("data:image/png;base64,"), url[:60]
            png = base64.b64decode(url.split(",", 1)[1])
            out = "/tmp/sundial_map_test.png"
            with open(out, "wb") as f:
                f.write(png)
            print(f"  wrote {out} — {len(png)} bytes, "
                  f"{len(url)} chars of data-URL")
    else:
        print("\n(offline — set SUNDIAL_NET=1 for a live CARTO tile fetch)")

    if not global_ok[0]:
        sys.exit(1)
    print("self-test OK")
