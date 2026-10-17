"""Tool registry for the agent investigation loop.

One uniform dict — ``TOOLS`` maps a tool name to
``{"fn", "desc", "args", "cost"}`` — so the agent loop can advertise a
catalog to the planner, dispatch ``call(name, **kwargs)``, and wrap the
whole registry in a wall-clock ``budgeted()`` guard. Names mirror
``agent_ctx.DEFAULT_TOOLS`` plus the offline extras ``pip2cc`` and
``claim_tz``:

    geocode        place text -> top-3 [{name, lat, lon, cc}]   (Nominatim)
    wiki_geo       lat/lon -> nearby Wikipedia titles+coords    (MediaWiki)
    overpass_pois  lat/lon -> named OSM POIs                    (Overpass)
    weather        lat/lon + date -> hourly archive             (Open-Meteo)
    timezone_at    lat/lon -> best-guess IANA zone              (offline)
    sun_at         lat/lon + ISO instant -> elevation/azimuth   (pysolar)
    context        the running agent_ctx brief for the pack     (offline)
    pip2cc         lat/lon -> {cc, name} country hit            (offline)
    claim_tz       naive claim local time -> UTC resolution     (offline)

Conventions (same as agent_ctx / claim_tz / tz_check): stdlib only at
import — every sundial module is imported lazily inside its tool fn via
``_lazy()``, so the module loads and its self-test runs even with zero
package deps and no network. Nothing in this module raises: bad input,
missing deps, and tool exceptions all surface as
``{"ok": False, "error": ...}`` envelopes instead of tracebacks.

    python3 sundial/agent_tools.py     # self-test
    python -m sundial.agent_tools
"""
from __future__ import annotations

import importlib
import time
from typing import Any

__all__ = ["TOOLS", "call", "catalog", "budgeted", "bind_pack", "PACK"]

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

#: Investigation state the ``context`` tool summarizes. The agent loop
#: refreshes it via ``bind_pack()`` as evidence accrues; ``context()``
#: reads whatever is here — empty fields degrade to a thinner brief.
PACK: dict = {"evidence": [], "claim": {}, "ranked": []}

#: Test hook — set to a callable ``(place) -> candidates`` to bypass the
#: network in geocode. Candidates may be GeoCandidate objects or dicts.
_GEOCODER = None


def bind_pack(evidence=None, claim=None, ranked=None) -> dict:
    """Point the ``context`` tool at the investigation state so far.

    Any field left as None keeps its previous value, so the loop can
    update just the piece that changed. Never raises.
    """
    try:
        if evidence is not None:
            PACK["evidence"] = list(evidence)
        if isinstance(claim, dict):
            PACK["claim"] = dict(claim)
        if ranked is not None:
            PACK["ranked"] = list(ranked)
    except Exception:
        pass
    return PACK


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _lazy(name: str):
    """Import a sibling module lazily; None when unavailable.

    Package-relative import first, then a bare-name fallback so tools
    still resolve when this file runs standalone (``python3
    sundial/agent_tools.py`` puts its own dir on sys.path).
    """
    candidates = [f"{__package__}.{name}"] if __package__ else []
    candidates.append(name)
    for cand in candidates:
        try:
            return importlib.import_module(cand)
        except Exception:
            continue
    return None


def _get(item: Any, key: str, default=None):
    """Attr-or-key read — tool results arrive as dataclasses or dicts."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_iso(iso: Any):
    """ISO-8601 string -> datetime; 'Z' tolerated. Raises ValueError."""
    from datetime import datetime
    raw = str(iso or "").strip()
    if not raw:
        raise ValueError("empty iso_utc — expected an ISO-8601 instant")
    if raw[-1:] in ("Z", "z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


# ---------------------------------------------------------------------------
# Tool implementations — each wraps one sundial module, lazily imported
# ---------------------------------------------------------------------------

def _t_geocode(place: str) -> list[dict]:
    """Free-text place -> top-3 [{name, lat, lon, cc}] via geo.geocode."""
    geo = _lazy("geo")
    fn = _GEOCODER if callable(_GEOCODER) else (geo.geocode if geo else None)
    if fn is None:
        raise RuntimeError("geo module unavailable")
    cands = fn(str(place or "").strip()) or []
    out: list[dict] = []
    for c in list(cands)[:3]:
        name = (_get(c, "name") or "").strip()
        if not name:
            name = str(_get(c, "display_name") or "").split(",")[0].strip()
        cc = _get(c, "country_code") or _get(c, "cc") or ""
        out.append({
            "name": name or "unknown",
            "lat": _float(_get(c, "lat")),
            "lon": _float(_get(c, "lon")),
            "cc": str(cc).upper(),
        })
    return out


def _t_wiki_geo(lat: float, lon: float, radius_m: int = 10000) -> list[dict]:
    """Wikipedia articles within radius_m -> [{title, lat, lon, dist_m}]."""
    wp = _lazy("wiki_places")
    if wp is None:
        raise RuntimeError("wiki_places module unavailable")
    return wp.geo_search(_float(lat), _float(lon),
                         radius_m=int(_float(radius_m, 10000)))


def _t_overpass_pois(lat: float, lon: float, kinds=None,
                     radius_m: int = 3000) -> list[dict]:
    """Named OSM POIs near a point -> [{name, lat, lon, kind, value}]."""
    op = _lazy("overpass")
    if op is None:
        raise RuntimeError("overpass module unavailable")
    return op.nearby_pois(
        _float(lat), _float(lon),
        radius_m=int(_float(radius_m, 3000)),
        kinds=kinds if kinds else ("amenity", "shop", "tourism", "name"),
    )


def _t_weather(lat: float, lon: float, date: str) -> dict:
    """One day of hourly archive weather -> historical_weather dict."""
    w = _lazy("weather")
    if w is None:
        raise RuntimeError("weather module unavailable")
    day = str(date or "").strip()[:10]
    if not day:
        raise ValueError("empty date — expected YYYY-MM-DD")
    return w.historical_weather(_float(lat), _float(lon), day)


def _t_timezone_at(lat: float, lon: float) -> dict:
    """lat/lon -> {"tz", "source"}; tz_check table first, claim_tz else."""
    la, lo = _float(lat), _float(lon)
    tzc = _lazy("tz_check")
    if tzc is not None:
        name = tzc.expected_tz_name(la, lo)
        if name:
            return {"tz": str(name), "source": "tz_check"}
    ct = _lazy("claim_tz")
    if ct is None:
        raise RuntimeError("claim_tz module unavailable")
    return {"tz": str(ct.tz_at(la, lo)), "source": "claim_tz"}


def _t_sun_at(lat: float, lon: float, iso_utc: str) -> dict:
    """Sun elevation/azimuth at one UTC instant -> sun_position dict."""
    sol = _lazy("solar")
    if sol is None:
        raise RuntimeError("solar module unavailable (pysolar missing?)")
    return sol.sun_position(_float(lat), _float(lon), _parse_iso(iso_utc))


def _t_pip2cc(lat: float, lon: float) -> dict:
    """lat/lon -> {"cc", "name"}; over ocean adds the nearest border."""
    p = _lazy("pip2cc")
    if p is None:
        raise RuntimeError("pip2cc module unavailable")
    hit = p.country_at(lat, lon)
    if hit is not None:
        return dict(hit)
    out: dict = {"cc": None, "name": None}
    near = p.nearest_cc(lat, lon, k=1)
    if near:
        out["nearest"] = near[0]
    return out


def _t_claim_tz(time_iso: str, lat=None, lon=None, tz_name=None) -> dict:
    """Resolve a claimed local time to UTC -> to_utc dict."""
    ct = _lazy("claim_tz")
    if ct is None:
        raise RuntimeError("claim_tz module unavailable")
    flat = _float(lat) if lat is not None else None
    flon = _float(lon) if lon is not None else None
    return ct.to_utc(str(time_iso or ""), flat, flon,
                     tz_name=str(tz_name) if tz_name else None)


def _t_context() -> str:
    """The agent_ctx brief for PACK as it stands — the planning header."""
    ac = _lazy("agent_ctx")
    if ac is None:
        raise RuntimeError("agent_ctx module unavailable")
    return ac.build_context(
        PACK.get("evidence") or [],
        PACK.get("claim") or {},
        PACK.get("ranked") or [],
        tools=list(TOOLS),
    )


# ---------------------------------------------------------------------------
# Registry + dispatch
# ---------------------------------------------------------------------------

#: name -> {"fn", "desc", "args", "cost"}. ``args`` values read as
#: "type" or "type=default"; "?" marks an optional arg the tool treats
#: as None. ``cost`` is a rough wall-clock hint for the planner.
TOOLS: dict[str, dict] = {
    "geocode": {
        "fn": _t_geocode,
        "desc": "Geocode a place name -> top-3 [{name,lat,lon,cc}] (Nominatim).",
        "args": {"place": "str"},
        "cost": "med",
    },
    "wiki_geo": {
        "fn": _t_wiki_geo,
        "desc": "Wikipedia articles near lat/lon -> [{title,lat,lon,dist_m}].",
        "args": {"lat": "float", "lon": "float", "radius_m": "int=10000"},
        "cost": "med",
    },
    "overpass_pois": {
        "fn": _t_overpass_pois,
        "desc": "Named OSM POIs near lat/lon -> [{name,lat,lon,kind,value}].",
        "args": {"lat": "float", "lon": "float",
                 "kinds": "list[str]=amenity,shop,tourism,name",
                 "radius_m": "int=3000"},
        "cost": "high",
    },
    "weather": {
        "fn": _t_weather,
        "desc": "Hourly archive weather for one day at lat/lon (Open-Meteo).",
        "args": {"lat": "float", "lon": "float", "date": "str(YYYY-MM-DD)"},
        "cost": "med",
    },
    "timezone_at": {
        "fn": _t_timezone_at,
        "desc": "Best-guess IANA zone for lat/lon -> {tz, source} (offline).",
        "args": {"lat": "float", "lon": "float"},
        "cost": "low",
    },
    "sun_at": {
        "fn": _t_sun_at,
        "desc": "Sun elevation/azimuth deg at lat/lon for one ISO UTC instant.",
        "args": {"lat": "float", "lon": "float", "iso_utc": "str"},
        "cost": "low",
    },
    "context": {
        "fn": _t_context,
        "desc": "Re-render the investigation brief for the pack so far.",
        "args": {},
        "cost": "low",
    },
    "pip2cc": {
        "fn": _t_pip2cc,
        "desc": "Country at lat/lon -> {cc,name} (offline point-in-polygon).",
        "args": {"lat": "float", "lon": "float"},
        "cost": "low",
    },
    "claim_tz": {
        "fn": _t_claim_tz,
        "desc": "Resolve a claimed local time to UTC -> {utc,tz,offset_min,...}.",
        "args": {"time_iso": "str", "lat": "float?", "lon": "float?",
                 "tz_name": "str?"},
        "cost": "low",
    },
}


def _dispatch(registry: dict, name: Any, kwargs: dict) -> dict:
    """Look up ``name`` and run its fn under the ok/result/error/ms envelope."""
    try:
        meta = (registry or {}).get(name)
    except Exception:
        meta = None
    if not isinstance(meta, dict) or not callable(meta.get("fn")):
        return {"ok": False, "result": None,
                "error": f"unknown tool {name!r}", "ms": 0.0}
    t0 = time.perf_counter()
    try:
        res = meta["fn"](**dict(kwargs or {}))
        return {"ok": True, "result": res, "error": None,
                "ms": (time.perf_counter() - t0) * 1000.0}
    except Exception as e:
        return {"ok": False, "result": None,
                "error": f"{type(e).__name__}: {e}",
                "ms": (time.perf_counter() - t0) * 1000.0}


def call(name: str, **kwargs) -> dict:
    """Dispatch ``name`` against TOOLS -> {"ok","result","error","ms"}.

    The envelope is the whole contract: tool exceptions, unknown names,
    and bad kwargs all come back as ``{"ok": False, "error": ...}`` —
    this function never raises, so the agent loop can trust every step.
    """
    try:
        return _dispatch(TOOLS, name, kwargs)
    except Exception as e:  # belt-and-braces: dispatch itself is safe
        return {"ok": False, "result": None,
                "error": f"dispatch failure: {e}", "ms": 0.0}


def catalog() -> list[dict]:
    """[{name, desc, args, cost}] over TOOLS — for prompt building."""
    return _catalog(TOOLS)


def _catalog(registry: dict) -> list[dict]:
    out: list[dict] = []
    for name, meta in (registry or {}).items():
        try:
            if not isinstance(meta, dict):
                continue
            out.append({
                "name": str(name),
                "desc": str(meta.get("desc") or ""),
                "args": dict(meta.get("args") or {}),
                "cost": str(meta.get("cost") or "low"),
            })
        except Exception:
            continue
    return out


def budgeted(registry: dict | None = None, budget_s: float = 45) -> dict:
    """Wrap a registry in a cumulative wall-clock budget.

    Returns a plain dict::

        {"tools": <registry>, "budget_s": float, "spent_ms": float,
         "call": callable(name, **kwargs) -> envelope,
         "catalog": callable() -> catalog}

    ``wrapper["call"]`` dispatches like module-level ``call`` but adds
    each call's elapsed ms to ``wrapper["spent_ms"]``; once the total
    reaches the budget it stops invoking tools and returns
    ``{"ok": False, "result": None, "error": "budget", "ms": 0.0}``.
    Never raises.
    """
    reg = dict(registry or TOOLS)
    wrapper: dict = {
        "tools": reg,
        "budget_s": _float(budget_s, 45.0),
        "spent_ms": 0.0,
    }

    def _call(name, **kwargs) -> dict:
        try:
            if wrapper["spent_ms"] >= wrapper["budget_s"] * 1000.0:
                return {"ok": False, "result": None,
                        "error": "budget", "ms": 0.0}
            env = _dispatch(reg, name, kwargs)
            wrapper["spent_ms"] += _float(env.get("ms"))
            return env
        except Exception as e:
            return {"ok": False, "result": None,
                    "error": f"budget wrapper failure: {e}", "ms": 0.0}

    wrapper["call"] = _call
    wrapper["catalog"] = lambda: _catalog(reg)
    return wrapper


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/agent_tools.py  (offline — no network calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    fails = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        global fails
        fails += 0 if ok else 1
        tail = f" — {detail}" if detail else ""
        print(f"  {'ok' if ok else 'FAIL'} {label}{tail}")

    print("agent_tools self-test")

    # --- catalog shape ---------------------------------------------------
    cat = catalog()
    names = {c.get("name") for c in cat if isinstance(c, dict)}
    want = {"geocode", "wiki_geo", "overpass_pois", "weather",
            "timezone_at", "sun_at", "context", "pip2cc", "claim_tz"}
    check("catalog covers all tools", want <= names,
          f"missing {sorted(want - names)}")
    check("catalog entry shape",
          bool(cat) and all(
              {"name", "desc", "args", "cost"} <= set(c)
              and c.get("cost") in ("low", "med", "high")
              and isinstance(c.get("args"), dict)
              for c in cat))

    # --- unknown call ----------------------------------------------------
    r = call("definitely_not_a_tool")
    check("unknown call refused",
          r.get("ok") is False and bool(r.get("error")), str(r.get("error")))
    r = call("geocode")  # missing required kwarg -> envelope, not raise
    check("bad kwargs refused",
          r.get("ok") is False and bool(r.get("error")), str(r.get("error")))

    # --- budget refusal ---------------------------------------------------
    w0 = budgeted(TOOLS, budget_s=0)
    r = w0["call"]("pip2cc", lat=51.5, lon=-0.13)
    check("zero budget refuses",
          r.get("ok") is False and r.get("error") == "budget", str(r))

    slow_reg = {"nap": {"fn": lambda: (time.sleep(0.02), "done")[1],
                        "desc": "20ms nap", "args": {}, "cost": "low"}}
    wb = budgeted(slow_reg, budget_s=0.005)   # 5ms budget, ~20ms call
    r1 = wb["call"]("nap")
    r2 = wb["call"]("nap")
    check("budget accumulates then refuses",
          r1.get("ok") is True and r2.get("ok") is False
          and r2.get("error") == "budget",
          f"spent_ms={wb['spent_ms']:.1f}")

    # --- geocode via the _GEOCODER hook (no network) ----------------------
    class _FakeCand:  # attr-style candidate, like a GeoCandidate
        name, lat, lon, country_code = "London KY", 37.13, -84.08, "US"

    _GEOCODER = lambda place: [
        {"name": "London", "lat": 51.5074, "lon": -0.1278, "cc": "GB"},
        {"name": "London, Ontario", "lat": 42.98, "lon": -81.25,
         "country_code": "CA"},
        _FakeCand(),
        {"name": "fourth — must be cut", "lat": 0, "lon": 0, "cc": "XX"},
    ]
    try:
        r = call("geocode", place="London")
        res = r.get("result")
        check("geocode top-3 via _GEOCODER hook",
              r.get("ok") is True and isinstance(res, list)
              and len(res) == 3
              and all({"name", "lat", "lon", "cc"} <= set(d) for d in res)
              and res[0]["cc"] == "GB" and res[2]["cc"] == "US",
              str(res or r.get("error")))
    finally:
        _GEOCODER = None

    # --- offline tools (graceful if a dep is missing) ----------------------
    r = call("pip2cc", lat=51.5074, lon=-0.1278)
    res = r.get("result") or {}
    check("pip2cc London -> GB",
          r.get("ok") is True and res.get("cc") == "GB", str(res))
    r = call("pip2cc", lat=0.0, lon=0.0)
    res = r.get("result") or {}
    check("pip2cc ocean -> cc None",
          r.get("ok") is True and res.get("cc") is None, str(res))

    r = call("timezone_at", lat=51.5074, lon=-0.1278)
    res = r.get("result") or {}
    check("timezone_at London",
          r.get("ok") is True and res.get("tz") == "Europe/London", str(res))
    r = call("timezone_at", lat=0.0, lon=0.0)
    res = r.get("result") or {}
    check("timezone_at ocean -> claim_tz fallback",
          r.get("ok") is True and res.get("tz") == "UTC"
          and res.get("source") == "claim_tz", str(res))

    r = call("sun_at", lat=51.5074, lon=-0.1278,
             iso_utc="2024-06-15T12:00:00Z")
    res = r.get("result") or {}
    if (r.get("ok") is False
            and "unavailable" in str(r.get("error"))):
        # solar needs the package context (from .schema) + pysolar —
        # a standalone `python3 sundial/agent_tools.py` can't provide it.
        print("  ok sun_at -> skipped (solar module unavailable "
              "standalone; run `python -m sundial.agent_tools`)")
    else:
        check("sun_at -> elevation/azimuth",
              r.get("ok") is True
              and {"elevation_deg", "azimuth_deg"} <= set(res),
              (f"elev={res.get('elevation_deg'):.1f} "
               f"az={res.get('azimuth_deg'):.1f}") if res else str(r))

    r = call("claim_tz", time_iso="2024-06-15T15:00",
             lat=51.5074, lon=-0.1278)
    res = r.get("result") or {}
    utc = str(res.get("utc") or "")
    check("claim_tz naive London -> UTC",
          r.get("ok") is True
          and ("2024-06-15T14:00" in utc          # BST -> 14:00Z
               or res.get("ambiguous")),          # tzdata-free degrade ok
          f"utc={utc} tz={res.get('tz')}")

    # --- context ----------------------------------------------------------
    r = call("context")
    check("context -> brief string",
          r.get("ok") is True and "CLAIM:" in str(r.get("result")),
          str(r.get("result"))[:60])
    bind_pack(
        claim={"place": "London"},
        evidence=[{"type": "solar", "summary": "sun high at noon",
                   "verdict": "supports", "confidence": 0.7}],
    )
    r = call("context")
    res = str(r.get("result"))
    check("context reflects bound pack",
          r.get("ok") is True and "London" in res and "sun high" in res,
          res[:100].replace("\n", " | "))

    if fails:
        print(f"  {fails} check(s) failed")
        sys.exit(1)
    print("  all checks passed")
