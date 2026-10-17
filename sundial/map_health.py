"""Raster-basemap registry + health probe for the map renderers.

Both map front-ends — the interactive fragment (``sundial/map_viz.py``)
and the static PIL composite (``sundial/map_tiles.py``) — historically
hardcode CARTO ``dark_all``. This module keeps a small registry of
keyless raster tile providers and probes one tile per endpoint, so a
caller can pick a *working* dark basemap at render time instead of a
fixed one:

  * ``PROVIDERS``       — the registry (id, url template, dark flag, …)
  * ``probe()``         — is one tile URL template answering image bytes?
  * ``best_dark()``     — first dark provider whose probe passes
  * ``tile_url_for()``  — build a concrete {z}/{x}/{y} tile URL
  * ``provider_for()``  — registry lookup by id (a copy, or None)
  * ``summary()``       — probe every provider once -> [{"id","ok","ms"}]
  * ``HEALTHY_TILE``    — module attr: url_template of the chosen provider
  * ``last_choice()``   — provider dict picked by the last best_dark()

Fail-soft house style: stdlib only at import time, requests-if-importable
with a urllib fallback (per overpass.py / map_tiles.py), no function ever
raises, and the network is touched ONLY inside probe / best_dark /
summary — every other symbol is pure registry math. Probe results are
cached for the process lifetime, negatives included: one offline pass
must not re-hit the network on every render.

Self-test::

    python3 sundial/map_health.py                 # offline asserts only
    SUNDIAL_NET=1 python3 sundial/map_health.py   # + live probe table
"""
from __future__ import annotations

import time

USER_AGENT = "sundial-hackathon/0.1"
HTTP_TIMEOUT_S = 6
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Registry order matters for best_dark(): entries are probed in order and
# the first dark, keyless pass wins — preferred providers go first.
# Verified live Oct 2026: CARTO raster tiles went key-required (HTTP 200 +
# a fixed "API KEY REQUIRED" placeholder PNG for every tile) — they stay
# in the registry for diagnostics but lose best_dark to Esri.
PROVIDERS = [
    {
        "id": "esri_darkgray",
        "name": "Esri World Dark Gray Base",
        "url_template":
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "subdomains": "",
        "dark": True,
        "attribution": "Esri, HERE, Garmin, © OpenStreetMap contributors",
        "needs_key": False,
        "notes": "VERIFIED keyless dark raster (real distinct tiles). "
                 "NOTE the {z}/{y}/{x} order — differs from XYZ providers. "
                 "Worldwide z0-10, populated areas to ~16.",
    },
    {
        "id": "carto_dark",
        "name": "CARTO Dark Matter",
        "url_template":
            "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png{key}",
        "subdomains": "abcd",
        "dark": True,
        "attribution": "© OpenStreetMap contributors © CARTO",
        "needs_key": True,
        "key_env": "SUNDIAL_CARTO_KEY",
        "key_param": "?key={key}",
        "notes": "KEY REQUIRED since ~Sep 2026 — keyless requests return "
                 "200 + an identical 'API KEY REQUIRED' placeholder PNG "
                 "(probe detects it via identical bytes). Free key at "
                 "carto.com/basemaps/apikey; set SUNDIAL_CARTO_KEY to "
                 "re-enable (best style match when keyed).",
    },
    {
        "id": "mella_dark",
        "name": "Mella Dark Basemap",
        "url_template":
            "https://basemap.queeniemella.cc/tiles/countries/{z}/{x}/{y}.png",
        "subdomains": "",
        "dark": True,
        "attribution": "© OpenStreetMap contributors — queeniemella",
        "needs_key": False,
        "notes": "VERIFIED keyless dark raster (real tiles). Indie "
                 "single-operator service built for the CARTO-key gap — "
                 "works today, uptime unproven. Third in line.",
    },
    {
        "id": "osm_standard",
        "name": "OpenStreetMap Standard",
        "url_template":
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "subdomains": "abc",
        "dark": False,
        "attribution": "© OpenStreetMap contributors",
        "needs_key": False,
        "notes": "VERIFIED keyless (light style). OSM tile-usage policy "
                 "applies — keep loads light; blocked clients get a "
                 "200 + error PNG + x-blocked header.",
    },
    {
        "id": "opentopomap",
        "name": "OpenTopoMap",
        "url_template":
            "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "subdomains": "abc",
        "dark": False,
        "attribution": "© OpenStreetMap contributors, SRTM | "
                       "map style © OpenTopoMap (CC-BY-SA)",
        "needs_key": False,
        "notes": "VERIFIED keyless (light topographic). Fair-use limits.",
    },
    {
        "id": "wikimedia",
        "name": "Wikimedia Maps",
        "url_template":
            "https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png",
        "subdomains": "",
        "dark": False,
        "attribution": "© OpenStreetMap contributors — Wikimedia Maps",
        "needs_key": False,
        "excluded": True,
        "notes": "EXCLUDED — ToS prohibits third-party use since Oct 2020. "
                 "Kept for diagnostics only.",
    },
    {
        "id": "openfreemap",
        "name": "OpenFreeMap Liberty",
        "url_template":
            "https://tiles.openfreemap.org/styles/liberty/{z}/{x}/{y}.png",
        "subdomains": "",
        "dark": False,
        "attribution": "© OpenStreetMap contributors — OpenFreeMap",
        "needs_key": False,
        "excluded": True,
        "notes": "EXCLUDED — vector-tile service; the raster endpoint "
                 "returns empty bytes (verified). Needs MapLibre, "
                 "unusable for the PNG pipeline.",
    },
]

# Module-attribute pattern: best_dark() rewrites this to the winner's
# url_template so renderers can read map_health.HEALTHY_TILE once probed.
# Default is the verified keyless Esri dark raster.
HEALTHY_TILE = next(
    (p["url_template"] for p in PROVIDERS if p["id"] == "esri_darkgray"), "")

_choice = {"settled": False, "provider": None}  # best_dark() cache


def provider_for(provider_id) -> dict | None:
    """Registry entry for `provider_id` (a shallow copy), or None."""
    try:
        pid = str(provider_id or "").strip()
        for p in PROVIDERS:
            if p.get("id") == pid:
                return dict(p)
    except Exception:
        pass
    return None


def tile_url_for(provider_id, z, x, y, s=None) -> str | None:
    """Concrete tile URL for a registered provider, or None.

    `s` pins the subdomain; when None one is picked deterministically
    from (x + y) — the same scheme map_viz.py's JS uses — so a given tile
    always hits the same host. Templates without "{s}" ignore `s`. Never
    raises: unknown ids, bad coords and malformed templates all -> None.
    """
    try:
        p = provider_for(provider_id)
        tmpl = str((p or {}).get("url_template") or "")
        if not tmpl:
            return None
        zi, xi, yi = int(z), int(x), int(y)
        if "{s}" in tmpl:
            subs = str(p.get("subdomains") or "a")
            sub = str(s) if s is not None else subs[(xi + yi) % len(subs)]
            return tmpl.format(s=sub, z=zi, x=xi, y=yi)
        return tmpl.format(z=zi, x=xi, y=yi)
    except Exception:
        return None


def _is_png(blob) -> bool:
    """True when `blob` starts with the 8-byte PNG magic."""
    try:
        return bytes(blob[:8]) == _PNG_MAGIC
    except Exception:
        return False


def _healthy(status, content_type, head) -> bool:
    """200 + an image content-type, or 200 + PNG magic bytes."""
    try:
        if int(status) != 200:
            return False
    except Exception:
        return False
    if "image" in str(content_type or "").lower():
        return True
    return _is_png(head)


def probe(url_template, z=3, x=4, y=2, timeout=6) -> bool:
    """True when one tile of `url_template` answers with image bytes.

    A small GET (a 256px tile is ~10-40 kB) — requests when importable,
    urllib otherwise; a requests *exception* falls through to urllib since
    a requests-only failure (proxy, TLS) may not affect it, while a real
    HTTP response is evaluated as-is. False on any failure; never raises.
    """
    try:
        tmpl = str(url_template or "")
        zi, xi, yi = int(z), int(x), int(y)
        if "{s}" in tmpl:
            url = tmpl.format(s="a", z=zi, x=xi, y=yi)
        else:
            url = tmpl.format(z=zi, x=xi, y=yi)
        if not url.startswith(("http://", "https://")):
            return False
    except Exception:
        return False

    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        try:
            resp = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            return _healthy(resp.status_code,
                            resp.headers.get("Content-Type"),
                            resp.content[:8])
        except Exception:
            pass  # fall through to urllib — a requests-only failure
            # (proxy, TLS) may not affect it
    try:
        import urllib.request
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            ctype = resp.headers.get("Content-Type")
            head = resp.read(8)
        return _healthy(status, ctype, head)
    except Exception:
        return False


def best_dark(timeout=6) -> dict | None:
    """First dark, keyless provider whose probe passes, else None.

    The result — including a negative one — is cached for the process
    lifetime, so renderers can call this per map without re-probing.
    On success HEALTHY_TILE is updated to the winner's url_template and
    last_choice() reports the provider dict.
    """
    global HEALTHY_TILE
    try:
        if _choice["settled"]:
            return last_choice()  # a copy — callers can't corrupt the cache
    except Exception:
        pass
    picked = None
    try:
        for p in PROVIDERS:
            if not p.get("dark") or p.get("needs_key"):
                continue
            if probe(p.get("url_template"), timeout=timeout):
                picked = dict(p)
                break
    except Exception:
        picked = None
    try:
        _choice["provider"] = picked
        _choice["settled"] = True
        if picked and picked.get("url_template"):
            HEALTHY_TILE = picked["url_template"]
    except Exception:
        pass
    return picked


def last_choice() -> dict | None:
    """Provider dict picked by the most recent best_dark(), or None.

    Pure accessor — never probes; before the first best_dark() call (or
    after an all-failed one) it returns None.
    """
    try:
        p = _choice.get("provider")
        return dict(p) if isinstance(p, dict) else None
    except Exception:
        return None


def summary() -> list[dict]:
    """Probe every provider once -> [{"id", "ok", "ms"}] rows.

    For diagnostics and report footers; never raises — a provider that
    dies mid-probe just gets ok=False with its elapsed milliseconds.
    """
    out = []
    try:
        entries = list(PROVIDERS)
    except Exception:
        entries = []
    for p in entries:
        try:
            pid = str(p.get("id") or "?")
        except Exception:
            pid = "?"
        t0 = time.monotonic()
        ok = False
        try:
            ok = bool(probe(p.get("url_template")))
        except Exception:
            ok = False
        out.append({"id": pid, "ok": ok,
                    "ms": int(round((time.monotonic() - t0) * 1000.0))})
    return out


if __name__ == "__main__":
    import os
    import sys

    print("map_health self-test (offline):")
    global_ok = [True]

    def check(label, cond):
        print(f"  {label}: {'ok' if cond else 'FAIL'}")
        if not cond:
            global_ok[0] = False

    # --- registry shape ---------------------------------------------
    req = {"id", "name", "url_template", "subdomains",
           "dark", "attribution", "needs_key", "notes"}
    check("PROVIDERS is a non-empty list",
          isinstance(PROVIDERS, list) and len(PROVIDERS) >= 6)
    check("every entry has the full key set",
          all(isinstance(p, dict) and req <= set(p) for p in PROVIDERS))
    check("ids unique",
          len({p["id"] for p in PROVIDERS}) == len(PROVIDERS))
    check("every template has {z},{x},{y}",
          all("{z}" in p["url_template"] and "{x}" in p["url_template"]
              and "{y}" in p["url_template"] for p in PROVIDERS))
    check("bool flags are bools",
          all(isinstance(p["dark"], bool)
              and isinstance(p["needs_key"], bool) for p in PROVIDERS))
    darks = [p["id"] for p in PROVIDERS if p["dark"]]
    check("at least one dark provider", len(darks) >= 1)
    check("carto_dark is the first dark", darks[0] == "carto_dark")
    check("HEALTHY_TILE defaults to carto_dark",
          HEALTHY_TILE == provider_for("carto_dark")["url_template"])

    # --- provider_for -------------------------------------------------
    check("provider_for('carto_dark') round-trips",
          provider_for("carto_dark")["id"] == "carto_dark")
    check("provider_for returns a copy",
          provider_for("carto_dark") is not PROVIDERS[0])
    check("provider_for('nope') -> None", provider_for("nope") is None)
    check("provider_for(None) -> None", provider_for(None) is None)

    # --- tile_url_for math ---------------------------------------------
    # default subdomain: subs[(x + y) % len(subs)] -> "abcd"[(4+2) % 4] = "c"
    check("carto_dark url + deterministic subdomain",
          tile_url_for("carto_dark", 3, 4, 2)
          == "https://c.basemaps.cartocdn.com/dark_all/3/4/2.png")
    check("pinned subdomain", tile_url_for("carto_dark", 3, 4, 2, s="d")
          == "https://d.basemaps.cartocdn.com/dark_all/3/4/2.png")
    check("no-{s} provider ignores s",
          tile_url_for("wikimedia", 1, 0, 1)
          == "https://maps.wikimedia.org/osm-intl/1/0/1.png")
    check("unknown provider -> None",
          tile_url_for("nope", 3, 4, 2) is None)
    check("bad coords -> None",
          tile_url_for("carto_dark", "z", "x", "y") is None)
    check("None coords -> None",
          tile_url_for("carto_dark", None, None, None) is None)

    # --- probe() offline-safe False paths ------------------------------
    check("probe('') -> False", probe("") is False)
    check("probe(None) -> False", probe(None) is False)
    check("probe(junk) -> False", probe("not a url at all") is False)
    check("probe(bad placeholders) -> False",
          probe("https://x/{z}/{x}/{y}/{q}.png") is False)

    # --- last_choice before any best_dark ------------------------------
    check("last_choice() before probe -> None", last_choice() is None)

    if os.environ.get("SUNDIAL_NET") == "1":
        print("\nlive probes (SUNDIAL_NET=1):")
        ok_carto = probe(provider_for("carto_dark")["url_template"])
        print(f"  probe(carto_dark tile) -> {ok_carto}")

        pick = best_dark()
        print(f"  best_dark() -> {(pick or {}).get('id')!r}")
        print(f"  last_choice() -> {(last_choice() or {}).get('id')!r}")
        print(f"  HEALTHY_TILE -> {HEALTHY_TILE}")
        if ok_carto:
            check("best_dark picked carto_dark",
                  (pick or {}).get("id") == "carto_dark")
            check("last_choice mirrors pick",
                  (last_choice() or {}).get("id")
                  == (pick or {}).get("id"))
            check("HEALTHY_TILE follows pick",
                  HEALTHY_TILE == (pick or {}).get("url_template"))
            check("second best_dark() is cached",
                  best_dark() is not None)
        else:
            print("  carto_dark unreachable — best_dark diagnostics "
                  "only (offline or carto down)")

        rows = summary()
        check("summary() returns one row per provider",
              isinstance(rows, list) and len(rows) == len(PROVIDERS))
        check("summary() rows have id/ok/ms",
              all(set(r) >= {"id", "ok", "ms"} and isinstance(r["ms"], int)
                  for r in rows))
        for r in rows:
            print(f"    {r['id']:18} ok={r['ok']!s:5} {r['ms']:5} ms")
    else:
        print("\n(offline — set SUNDIAL_NET=1 for live provider probes)")

    if not global_ok[0]:
        sys.exit(1)
    print("self-test OK")
