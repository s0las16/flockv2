"""Self-contained interactive mini-map fragment for the verdict report.

Tesla-style dark slippy map in vanilla JS — Esri ``World_Dark_Gray_Base``
raster tiles (no API key, no Leaflet, no CDN script), pointer-drag panning,
wheel/double-click zoom, and markers + lat/lon bands projected through
Web-Mercator onto absolutely-positioned tile <img>s in a clipped div.

Everything is fail-soft: malformed input degrades to a world view or "",
tile-load failures degrade to a dark "map offline" plate that still
projects markers, and with JS disabled the container is a static dark
panel showing the centre coordinates. Each fragment gets unique ids via
a module counter so a report can splice several maps onto one page.
"""
from __future__ import annotations

import html
import itertools
import json
import math
import re
from typing import Any

_TILE = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
         "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}")  # Esri z/y/x
_ATTR = "Esri, HERE, Garmin, © OpenStreetMap contributors"
_LATLIM = 85.0511  # Web-Mercator latitude cutoff
_Z_MIN, _Z_MAX = 1, 16  # Esri dark-gray: worldwide z0-10, cities to ~16
_WORLD = (25.0, 0.0, 2)  # anchor=None -> whole-world view
_KINDS = ("claim", "pred", "geo", "wiki")
_MAX_MARKERS, _MAX_BANDS, _MAX_LABEL = 64, 32, 80

_IDS = itertools.count()  # unique id per fragment, process-wide

_C_DIM = re.compile(r"^[0-9a-zA-Z%.,() -]{1,24}$")
_C_COLOR = re.compile(r"^[#a-zA-Z0-9(),.% -]{1,48}$")

_CSS = """\
.smap{position:relative;overflow:hidden;background:#0b1220;border:1px solid rgba(148,163,184,.14);border-radius:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.smap-v{position:absolute;inset:0;overflow:hidden;cursor:grab;touch-action:none;user-select:none;-webkit-user-select:none}
.smap-v:active{cursor:grabbing}
.smap-t{position:absolute;inset:0;will-change:transform}
.smap-t.smap-inv{filter:invert(1) hue-rotate(180deg) brightness(.92) contrast(1.08) saturate(.65)}
.smap-i{position:absolute;width:256px;height:256px;pointer-events:none}
.smap-o{position:absolute;inset:0;overflow:hidden;pointer-events:none}
.smap-ring{position:absolute;border:1.5px solid rgba(66,133,244,.7);background:rgba(66,133,244,.13);border-radius:50%;transform:translate(-50%,-50%);pointer-events:none}
.smap-m{position:absolute;width:0;height:0;pointer-events:auto}
.smap-m u{position:absolute;transform:translate(-50%,-50%);text-decoration:none;box-sizing:border-box}
.smap-claim u{width:14px;height:14px;border:3px solid #fff;background:rgba(0,0,0,.35);border-radius:50%;box-shadow:0 1px 4px rgba(0,0,0,.6)}
.smap-pred{z-index:3}
.smap-pred u{width:28px;height:28px;background:#ea4335;border-radius:50% 50% 50% 0;transform:translate(-50%,-122%) rotate(-45deg);box-shadow:-2px 3px 7px rgba(0,0,0,.55)}
.smap-pred u::after{content:"";position:absolute;left:9px;top:9px;width:10px;height:10px;border-radius:50%;background:#7a1d16}
.smap-geo u{width:10px;height:10px;background:#ea4335;opacity:.55;border:2px solid #fff;border-radius:50%;box-shadow:0 1px 3px rgba(0,0,0,.5)}
.smap-wiki u{width:7px;height:7px;background:#94a3b8;border-radius:50%;box-shadow:0 0 0 1.5px rgba(2,6,23,.8)}
.smap-m b{position:absolute;left:12px;top:-10px;font:12px/1.6 -apple-system,BlinkMacSystemFont,Inter,"Segoe UI",sans-serif;font-weight:500;color:#fff;background:rgba(10,10,10,.85);padding:1px 8px;border-radius:6px;white-space:nowrap;max-width:220px;overflow:hidden;text-overflow:ellipsis;box-shadow:0 2px 8px rgba(0,0,0,.4)}
.smap-pred b{left:18px;top:-44px}
.smap-geo b,.smap-wiki b{display:none}
.smap-geo:hover b,.smap-wiki:hover b{display:block}
.smap-m b:empty{display:none}
.smap-zm{position:absolute;right:12px;bottom:28px;z-index:5;display:flex;flex-direction:column;background:rgba(10,10,10,.88);border:1px solid rgba(255,255,255,.12);border-radius:10px;overflow:hidden}
.smap-zm button{width:34px;height:34px;background:transparent;border:0;color:#eee;font:18px/1 -apple-system,sans-serif;cursor:pointer}
.smap-zm button+button{border-top:1px solid rgba(255,255,255,.1)}
.smap-zm button:hover{background:rgba(255,255,255,.08)}
.smap-b{position:absolute;border:1px solid;box-sizing:border-box;pointer-events:none}
.smap-b i{position:absolute;inset:0;opacity:.16}
.smap-c{display:none;position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);color:#64748b;font-size:11px;letter-spacing:.04em;white-space:nowrap}
.smap-off .smap-v{background:#0b1524;background-image:linear-gradient(rgba(148,163,184,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.05) 1px,transparent 1px);background-size:36px 36px}
.smap-off .smap-c{display:block}
.smap-fb{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:11px;letter-spacing:.04em}
.smap-vec{display:none;position:absolute;inset:0;z-index:2;background:#0a0d12}
.smap-vec .vmap{border:none!important;border-radius:0!important}
.smap-gm{display:none;position:absolute;inset:0;z-index:3;background:#0a0d12}
.smap-gm iframe{width:100%;height:100%;border:0;filter:invert(.92) hue-rotate(180deg) brightness(.95) contrast(1.05)}
.smap-sw{position:absolute;top:12px;right:12px;z-index:5;display:flex;gap:2px;background:rgba(10,10,10,.88);border:1px solid rgba(255,255,255,.12);border-radius:999px;padding:3px;backdrop-filter:blur(6px)}
.smap-sw button{background:transparent;border:none;color:#aaa;font:12px/1.3 -apple-system,BlinkMacSystemFont,Inter,"Segoe UI",sans-serif;font-weight:500;padding:6px 12px;border-radius:999px;cursor:pointer}
.smap-sw button:hover{color:#e2e8f0}
.smap-sw button.on{background:rgba(255,255,255,.12);color:#fff}
.smap-attr{position:absolute;right:6px;bottom:4px;color:rgba(148,163,184,.5);font-size:9px;pointer-events:none;z-index:4}
.smap-hint{position:absolute;left:6px;bottom:4px;color:rgba(148,163,184,.35);font-size:9px;pointer-events:none;z-index:4}
"""

_JS = """\
(function () {
  var root = document.getElementById("__ID__");
  if (!root) return;
  var view = root.querySelector(".smap-v"),
      tiler = root.querySelector(".smap-t"),
      ovl = root.querySelector(".smap-o"),
      co = root.querySelector(".smap-c"),
      attr = root.querySelector(".smap-attr"),
      vecBox = root.querySelector(".smap-vec"),
      gmBox = root.querySelector(".smap-gm"),
      zm = root.querySelector(".smap-zm"),
      sw = root.querySelector(".smap-sw"),
      fb = root.querySelector(".smap-fb");
  if (fb) fb.parentNode.removeChild(fb);
  var CFG = __CFG__, MK = __MK__, BD = __BD__;
  var PROVS = {
    streets: {
      name: "Streets",
      url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      inv: true, maxz: 19,
      attr: "© OpenStreetMap contributors (dark)"
    },
    esri: {
      name: "Dark",
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      ref: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
      inv: false, maxz: 16,
      attr: "Esri, HERE, Garmin, © OpenStreetMap contributors"
    },
    sat: {
      name: "Satellite",
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      inv: false, maxz: 18,
      attr: "Esri World Imagery, Maxar, Earthstar Geographics"
    },
    topo: {
      name: "Topo",
      url: "https://tile.opentopomap.org/{z}/{x}/{y}.png",
      inv: true, maxz: 16,
      attr: "© OpenTopoMap, © OpenStreetMap contributors"
    }
  };
  var provKey = "streets";
  var TS = 256, MINZ = 1, MAXZ = 18,
      Z = Math.max(MINZ, Math.min(MAXZ, CFG.z | 0)),
      ox = 0, oy = 0, vw = 0, vh = 0,
      fails = 0, off = false, cov = null, init = false, drag = null;
  function W() { return Math.pow(2, Z) * TS; }
  function proj(lat, lon) {
    lat = Math.max(-85.0511, Math.min(85.0511, +lat));
    var s = Math.sin(lat * Math.PI / 180), w = W();
    return {x: (lon + 180) / 360 * w,
            y: (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * w};
  }
  function unproj(x, y) {
    var w = W(), a = Math.exp(2 * Math.PI * (1 - 2 * y / w));
    return {lat: Math.asin((a - 1) / (a + 1)) * 180 / Math.PI,
            lon: x / w * 360 - 180};
  }
  function clampView() {
    var w = W();
    oy = vh >= w ? (w - vh) / 2 : Math.max(0, Math.min(w - vh, oy));
    ox = vw >= w ? (w - vw) / 2 : ((ox % w) + w) % w;
  }
  function fmtTile(tpl, z, x, y) {
    return tpl.replace("{z}", z).replace("{x}", x).replace("{y}", y);
  }
  function tileFail() {
    if (++fails >= 4 && !off) {
      if (provKey === "streets") {
        setProv("esri");
        return;
      }
      off = true; root.className += " smap-off"; tiler.innerHTML = "";
      if (vecBox && vecBox.innerHTML) vecBox.style.display = "block";
    }
  }
  function drawTiles() {
    if (off || provKey === "vec") return;
    var P = PROVS[provKey] || PROVS.esri;
    tiler.className = "smap-t" + (P.inv ? " smap-inv" : "");
    if (attr) attr.textContent = P.attr;
    var zUse = Math.min(Z, P.maxz || 16),
        n = Math.pow(2, Z),
        x0 = Math.floor(ox / TS) - 1, x1 = Math.floor((ox + vw) / TS) + 1,
        y0 = Math.max(0, Math.floor(oy / TS) - 1),
        y1 = Math.min(n - 1, Math.floor((oy + vh) / TS) + 1), h = "";
    for (var ty = y0; ty <= y1; ty++)
      for (var tx = x0; tx <= x1; tx++) {
        var wx = ((tx % n) + n) % n;
        var u = fmtTile(P.url, zUse, wx, ty);
        h += '<img class="smap-i" draggable="false" style="left:' + tx * TS +
             'px;top:' + ty * TS + 'px" src="' + u + '">';
        if (P.ref) {
          var ru = fmtTile(P.ref, zUse, wx, ty);
          h += '<img class="smap-i" draggable="false" style="left:' + tx * TS +
               'px;top:' + ty * TS + 'px" src="' + ru + '">';
        }
      }
    tiler.innerHTML = h;
    cov = {x0: x0, x1: x1, y0: y0, y1: y1, p: provKey};
    for (var i = 0; i < tiler.children.length; i++)
      tiler.children[i].onerror = tileFail;
  }
  var ringEl = null;
  if (CFG.r_km && CFG.r_km > 0) {
    ringEl = document.createElement("div");
    ringEl.className = "smap-ring";
    ovl.appendChild(ringEl);
  }
  var mel = [], bel = [], i, d;
  for (i = 0; i < BD.length; i++) {
    d = document.createElement("div"); d.className = "smap-b";
    d.style.borderColor = BD[i].c; d.innerHTML = "<i></i>";
    d.firstChild.style.background = BD[i].c;
    ovl.appendChild(d); bel.push(d);
  }
  for (i = 0; i < MK.length; i++) {
    d = document.createElement("div");
    d.className = "smap-m smap-" + MK[i].k;
    d.innerHTML = "<u></u><b></b>";
    if (MK[i].l) { d.lastChild.textContent = MK[i].l; d.title = MK[i].l; }
    ovl.appendChild(d); mel.push(d);
  }
  function layout() {
    vw = view.clientWidth || 1; vh = view.clientHeight || 1;
    if (!init) {
      var c = proj(CFG.lat, CFG.lon);
      ox = c.x - vw / 2; oy = c.y - vh / 2; init = true;
    }
    clampView();
    if (!off && provKey !== "vec") {
      if (!cov || cov.p !== provKey ||
          Math.floor(ox / TS) <= cov.x0 ||
          Math.floor((ox + vw) / TS) >= cov.x1 ||
          Math.floor(oy / TS) <= cov.y0 ||
          Math.floor((oy + vh) / TS) >= cov.y1) drawTiles();
      tiler.style.transform = "translate3d(" + -ox + "px," + -oy + "px,0)";
    }
    var w = W();
    if (ringEl && CFG.r_km) {
      var rp = proj(CFG.rlat || CFG.lat, CFG.rlon || CFG.lon),
          rN = proj((CFG.rlat || CFG.lat) + (CFG.r_km / 111.32), CFG.rlon || CFG.lon),
          pxR = Math.max(14, Math.min(w / 2, Math.abs(rp.y - rN.y))),
          rdx = rp.x - ox - vw / 2;
      rdx -= w * Math.round(rdx / w);
      ringEl.style.left = rdx + vw / 2 + "px";
      ringEl.style.top = rp.y - oy + "px";
      ringEl.style.width = (pxR * 2) + "px";
      ringEl.style.height = (pxR * 2) + "px";
    }
    for (i = 0; i < mel.length; i++) {
      var p = proj(MK[i].a, MK[i].o), dx = p.x - ox - vw / 2;
      dx -= w * Math.round(dx / w);
      mel[i].style.left = dx + vw / 2 + "px";
      mel[i].style.top = p.y - oy + "px";
    }
    for (i = 0; i < bel.length; i++) {
      var b = BD[i], p0 = proj(b.n, b.w), p1 = proj(b.s, b.e),
          bx = p0.x - ox - vw / 2;
      bx -= w * Math.round(bx / w);
      bel[i].style.left = bx + vw / 2 + "px";
      bel[i].style.top = p0.y - oy + "px";
      bel[i].style.width = Math.min(Math.max(p1.x - p0.x, 2), w) + "px";
      bel[i].style.height = Math.max(p1.y - p0.y, 2) + "px";
    }
    var ctr = unproj(ox + vw / 2, oy + vh / 2),
        lo = ((ctr.lon + 540) % 360) - 180;
    co.textContent = ctr.lat.toFixed(4) + "\\u00b0, " + lo.toFixed(4) +
                     "\\u00b0  z" + Z;
  }
  function setProv(k) {
    provKey = k; fails = 0; off = false;
    root.className = root.className.replace(/\\bsmap-off\\b/g, "").trim();
    if (vecBox) vecBox.style.display = (k === "vec") ? "block" : "none";
    if (gmBox) {
      if (k === "gmaps") {
        gmBox.style.display = "block";
        if (!gmBox.firstChild) {
          var ifr = document.createElement("iframe");
          ifr.setAttribute("loading", "lazy");
          ifr.setAttribute("referrerpolicy", "no-referrer-when-downgrade");
          var gurl = "https://" + "maps.google.com/maps?q=" +
                     encodeURIComponent(CFG.lat + "," + CFG.lon) +
                     "&z=" + Math.max(3, Math.min(17, Z)) + "&output=embed";
          ifr.src = gurl;
          gmBox.appendChild(ifr);
        }
      } else {
        gmBox.style.display = "none";
      }
    }
    if (zm) zm.style.display = (k === "gmaps") ? "none" : "flex";
    if (sw) {
      var btns = sw.querySelectorAll("button");
      for (var bi = 0; bi < btns.length; bi++)
        btns[bi].className = (btns[bi].getAttribute("data-p") === k) ? "on" : "";
    }
    if (k !== "gmaps") { cov = null; layout(); }
  }
  if (sw) {
    sw.addEventListener("click", function (e) {
      var b = e.target.closest ? e.target.closest("button[data-p]") : e.target;
      if (b && b.getAttribute && b.getAttribute("data-p")) {
        e.stopPropagation();
        setProv(b.getAttribute("data-p"));
      }
    });
  }
  if (zm) {
    zm.addEventListener("click", function (e) {
      var b = e.target.closest ? e.target.closest("button[data-z]") : e.target;
      if (b && b.getAttribute && b.getAttribute("data-z")) {
        e.stopPropagation();
        zoomAt(vw / 2, vh / 2, +b.getAttribute("data-z"));
      }
    });
  }
  function zoomAt(px, py, dz) {
    var nz = Math.max(MINZ, Math.min(MAXZ, Z + dz));
    if (nz === Z) return;
    var s = Math.pow(2, nz - Z);
    ox = (ox + px) * s - px; oy = (oy + py) * s - py;
    Z = nz; cov = null; layout();
  }
  function down(e) {
    drag = {x: e.clientX, y: e.clientY, id: e.pointerId};
    try { view.setPointerCapture(e.pointerId); } catch (er) {}
  }
  function move(e) {
    if (!drag || (e.pointerId !== undefined && e.pointerId !== drag.id)) return;
    ox -= e.clientX - drag.x; oy -= e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    layout();
  }
  function up() { drag = null; }
  if (window.PointerEvent) {
    view.addEventListener("pointerdown", down);
    view.addEventListener("pointermove", move);
    view.addEventListener("pointerup", up);
    view.addEventListener("pointercancel", up);
  } else {
    view.addEventListener("mousedown", down);
    view.addEventListener("mousemove", move);
    view.addEventListener("mouseup", up);
    view.addEventListener("mouseleave", up);
  }
  view.addEventListener("wheel", function (e) {
    e.preventDefault();
    var r = view.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1 : -1);
  }, {passive: false});
  view.addEventListener("dblclick", function (e) {
    e.preventDefault();
    var r = view.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, 1);
  });
  view.addEventListener("dragstart", function (e) { e.preventDefault(); });
  window.addEventListener("resize", layout);
  layout();
})();
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _js(obj: Any) -> str:
    """json.dumps made safe to embed inside a <script> tag."""
    return (
        json.dumps(obj, ensure_ascii=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _ff(value: Any, lo: float, hi: float) -> float:
    """Finite float clamped to [lo, hi]; raises on junk (caught by callers)."""
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("non-finite")
    return min(max(x, lo), hi)


def _lon_wrap(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _dim(value: Any, default: str) -> str:
    s = str(value or "").strip()
    return s if _C_DIM.match(s) else default


def _color(value: Any, default: str = "#38bdf8") -> str:
    s = str(value or "").strip()
    return s if _C_COLOR.match(s) else default


def static_anchor(
    anchor: tuple | None, zoom: int = 12
) -> tuple[int, int, float, float]:
    """Web-Mercator tile address for `anchor` — (x_tile, y_tile, x_frac, y_frac).

    Integer tile indices plus the fractional position inside that tile
    (0..1, top-left origin), exported so siblings can pre-render static
    maps with the same math the fragment uses live. Falls back to
    (0, 0, .5, .5) on bad input.
    """
    try:
        if isinstance(anchor, dict):
            lat, lon = anchor.get("lat"), anchor.get("lon")
        else:
            lat, lon = anchor[0], anchor[1]
        lat = _ff(lat, -_LATLIM, _LATLIM)
        lon = _lon_wrap(_ff(lon, -720.0, 720.0))
        z = int(min(max(float(zoom), 0.0), 24.0))
        n = 2 ** z
        xt = (lon + 180.0) / 360.0 * n
        r = math.radians(lat)
        yt = (1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n
        xi = min(max(int(math.floor(xt)), 0), n - 1)
        yi = min(max(int(math.floor(yt)), 0), n - 1)
        return (xi, yi, xt - xi, yt - yi)
    except Exception:
        return (0, 0, 0.5, 0.5)


def _centre(anchor: Any, zoom: Any) -> tuple[float, float, int]:
    """Resolve (lat, lon, z) — anchor=None or junk -> world view z=2."""
    try:
        if anchor is None:
            raise ValueError("world")
        if isinstance(anchor, dict):
            lat, lon = anchor.get("lat"), anchor.get("lon")
        else:
            lat, lon = anchor[0], anchor[1]
        clat = _ff(lat, -_LATLIM, _LATLIM)
        clon = _lon_wrap(_ff(lon, -720.0, 720.0))
    except Exception:
        return _WORLD
    try:
        z = int(min(max(float(zoom), float(_Z_MIN)), float(_Z_MAX)))
    except (TypeError, ValueError):
        z = 12
    return (clat, clon, z)


def _markers(markers: Any) -> list[dict]:
    out: list[dict] = []
    seq = markers if isinstance(markers, (list, tuple)) else []
    for m in seq[: _MAX_MARKERS]:
        try:
            lat = _ff(m.get("lat"), -_LATLIM, _LATLIM)
            lon = _lon_wrap(_ff(m.get("lon"), -720.0, 720.0))
            kind = str(m.get("kind") or "geo").lower()
            out.append(
                {
                    "a": round(lat, 6),
                    "o": round(lon, 6),
                    "l": str(m.get("label") or "")[:_MAX_LABEL],
                    "k": kind if kind in _KINDS else "geo",
                }
            )
        except (AttributeError, TypeError, ValueError):
            continue
    # a claim ring sitting under the main pin doesn't need its own label
    preds = [(m["a"], m["o"]) for m in out if m["k"] == "pred"]
    for m in out:
        if m["k"] == "claim" and any(
                abs(m["a"] - a) < 1e-3 and abs(m["o"] - o) < 1e-3 for a, o in preds):
            m["l"] = ""
    return out


def _bands(bands: Any) -> list[dict]:
    out: list[dict] = []
    seq = bands if isinstance(bands, (list, tuple)) else []
    for b in seq[: _MAX_BANDS]:
        try:
            s = _ff(b.get("lat_min"), -_LATLIM, _LATLIM)
            n = _ff(b.get("lat_max"), -_LATLIM, _LATLIM)
            w = _ff(b.get("lon_min"), -720.0, 720.0)
            e = _ff(b.get("lon_max"), -720.0, 720.0)
            if s > n:
                s, n = n, s
            if w > e:
                w, e = e, w
            out.append(
                {
                    "n": round(n, 6),
                    "s": round(s, 6),
                    "w": round(w, 6),
                    "e": round(e, 6),
                    "c": _color(b.get("color")),
                }
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def map_fragment(
    anchor: tuple | None,
    markers: list,
    bands: list,
    width: str = "100%",
    height: str = "360px",
    zoom: int = 12,
    radius_km: float | None = None,
    ring_center: tuple | dict | None = None,
    svg_fallback: str | None = None,
) -> str:
    """One <style> + <div> + <script> fragment — a dark interactive mini-map.

    `anchor` is (lat, lon) or None (world view at z=2). `markers` are
    {"lat","lon","label","kind"} with kind in claim|pred|geo|wiki;
    `bands` are {"lat_min","lat_max","lon_min","lon_max","color"} drawn
    as translucent rects. Includes an in-map provider switcher (Streets,
    Dark, Satellite, Topo, and optional Vector fallback). Returns "" only
    on catastrophic failure.
    """
    try:
        mid = "smap%d" % next(_IDS)
        clat, clon, z = _centre(anchor, zoom)
        cfg: dict[str, Any] = {"lat": round(clat, 6), "lon": round(clon, 6), "z": z}
        try:
            if radius_km and float(radius_km) > 0:
                cfg["r_km"] = round(float(radius_km), 1)
                if ring_center is not None:
                    rlat, rlon, _ = _centre(ring_center, z)
                    cfg["rlat"] = round(rlat, 6)
                    cfg["rlon"] = round(rlon, 6)
        except Exception:
            pass
        coords = "%.4f\u00b0, %.4f\u00b0  z%d" % (clat, clon, z)
        js = (
            _JS.replace("__ID__", mid)
            .replace("__CFG__", _js(cfg))
            .replace("__MK__", _js(_markers(markers)))
            .replace("__BD__", _js(_bands(bands)))
        )
        w, h = _dim(width, "100%"), _dim(height, "360px")
        has_vec = isinstance(svg_fallback, str) and "<svg" in svg_fallback
        # vector map stays as the silent offline fallback, not a button
        vec_div = ('<div class="smap-vec">' + svg_fallback + "</div>"
                   if has_vec else '<div class="smap-vec"></div>')
        sw_html = (
            '<div class="smap-sw">'
            '<button type="button" data-p="streets" class="on">Map</button>'
            '<button type="button" data-p="sat">Satellite</button>'
            '<button type="button" data-p="gmaps">Google</button>'
            "</div>"
            '<div class="smap-zm">'
            '<button type="button" data-z="1" aria-label="zoom in">+</button>'
            '<button type="button" data-z="-1" aria-label="zoom out">&minus;</button>'
            "</div>"
        )
        return (
            "<style>" + _CSS + "</style>"
            '<div id="' + mid + '" class="smap" style="width:' + w +
            ";height:" + h + '">'
            '<div class="smap-v">'
            '<div class="smap-t"></div>'
            '<div class="smap-o"></div>'
            '<div class="smap-c"></div>'
            '<div class="smap-fb">' + _e(coords) + "</div>"
            '<div class="smap-attr">' + _e(_ATTR) + "</div>"
            "</div>"
            '<div class="smap-gm"></div>'
            + vec_div + sw_html +
            "</div>"
            "<script>" + js + "</script>"
        )
    except Exception:
        return ""


if __name__ == "__main__":
    paris = (48.8566, 2.3522)
    frag = map_fragment(
        paris,
        [
            {"lat": 48.8566, "lon": 2.3522, "label": 'claim <here> "x"', "kind": "claim"},
            {"lat": 48.86, "lon": 2.37, "label": "pred a", "kind": "pred"},
            {"lat": 48.84, "lon": 2.34, "label": "geo b", "kind": "geo"},
            {"lat": 48.85, "lon": 2.36, "label": "wiki c", "kind": "wiki"},
            {"lat": "x", "lon": 2.3, "label": "dropped", "kind": "nope"},
        ],
        [{"lat_min": 48.80, "lat_max": 48.92, "lon_min": 2.28,
          "lon_max": 2.44, "color": "#facc15"}],
    )
    world = map_fragment(None, [], [])

    assert "arcgisonline.com" in frag and "World_Dark_Gray_Base" in frag, "esri tile URL"
    assert "\u00a9 OpenStreetMap contributors" in frag, "attribution"
    assert '"a": 48.8566' in frag, "marker JSON embedded"
    assert "\\u003c" in frag, "label < > escaped inside JSON"
    assert '"n": 48.92' in frag, "band JSON embedded"
    assert "dropped" not in frag, "bad marker skipped"
    assert '"z": 2' in world, "anchor=None -> world z=2"
    ids = lambda s: re.findall(r'id="(smap\d+)"', s)
    assert ids(frag) and ids(world) and ids(frag)[0] != ids(world)[0], "unique ids"

    xt, yt, xf, yf = static_anchor(paris, 12)
    assert 2070 <= xt <= 2080 and 1405 <= yt <= 1420, (xt, yt)
    assert 0.0 <= xf < 1.0 and 0.0 <= yf < 1.0, (xf, yf)
    assert static_anchor(None) == (0, 0, 0.5, 0.5)
    assert static_anchor((90, 190), 3)[0] >= 0  # clamps lat, wraps lon

    for bad in [("junk", "junk", "junk"), (None, None, None)]:
        out = map_fragment(*bad)
        assert isinstance(out, str) and '"z": 2' in out
    assert map_fragment(paris, [], "not-a-band")[0:6] == "<style"

    print("map_fragment: %d bytes, world: %d bytes" % (len(frag), len(world)))
    print("static_anchor(paris, 12) =", static_anchor(paris, 12))
    print("map_viz self-test: OK")
