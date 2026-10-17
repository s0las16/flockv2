"""OSM context stage — fine-grained place verification at the claimed anchor.

`context_at` asks Overpass what actually sits around the claimed lat/lon —
named POIs, road names, squares — and condenses the raw elements into
{poi_count, kinds, notable, nearest_road, density} so the adjudicator can
tell a real city square from an empty field or a quiet suburb.
`context_match` fuzzy-matches the claimed place text against the names
found; `context_evidence` packages the outcome as an Evidence card
(type="osm_context").

Verdict asymmetry mirrors overpass.py / wiki_places.py: a notable place
name the claim mentions genuinely existing at the anchor is honest support
(0.5); the only contradiction is an urban-core claim ("city centre",
"downtown", "market square", ...) landing on rural ground (0.4); every
other outcome stays INCONCLUSIVE because OSM coverage is uneven.

Stdlib-only at import: .overpass and .schema are imported lazily inside
the functions that need them, and no public function raises — failures
degrade to {"ok": False} / INCONCLUSIVE / None rather than exceptions.

  python -m sundial.osm_context        # offline self-test + guarded live demo
"""
from __future__ import annotations

import math
import re
import unicodedata

MAX_ELEMENTS = 120  # one union query; dense centres saturate anyway

# Tag keys pulled as nodes (density + notable names) and as named ways
# (roads/squares/town-hall buildings — the shape of "what's here").
_NODE_KEYS = (
    "amenity", "shop", "tourism", "leisure", "historic", "place",
    "highway", "railway", "public_transport", "office", "craft", "name",
)
_WAY_KEYS = (
    "highway", "place", "leisure", "amenity", "tourism", "historic",
)
# Priority order for assigning one primary kind per element.
_KIND_PRIORITY = (
    "place", "amenity", "shop", "tourism", "leisure", "historic",
    "highway", "railway", "public_transport", "office", "craft",
    "aeroway", "building", "natural", "name",
)

# Density bands on poi_count (named/described elements within radius_m).
URBAN_MIN_POIS = 30
SUBURBAN_MIN_POIS = 6

_NOTABLE_CAP = 60
_ROAD_CAP = 20
_TOKEN_OVERLAP_MIN = 0.60  # same convention as overpass.py / wiki_places.py
_MIN_SUBSTR = 3            # shorter substrings false-positive too easily

# Fold quirks NFKD leaves behind so "Strasse" still matches "Straße".
_FOLD_MAP = {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ð": "d",
             "þ": "th", "ł": "l", "đ": "d"}

_STOP = {
    "of", "the", "a", "an", "and", "at", "in", "on", "for", "to", "by",
    "or", "de", "del", "la", "le", "el", "di", "da", "du", "des", "von",
    "van", "en", "al", "am", "im", "zu", "zur", "zum",
}
# Generic place words — matched as text but not "identity" tokens:
# "Husum city centre" is about *Husum*, not about the word "city".
_DESCRIPTORS = {
    "city", "town", "village", "centre", "center", "downtown", "midtown",
    "district", "suburb", "suburban", "urban", "rural", "countryside",
    "area", "square", "plaza", "platz", "market", "old", "new", "main",
    "north", "south", "east", "west", "central", "centro", "stadt",
    "near", "next", "outside", "inside",
}
# Normalized substrings that claim an urban core; only vs "rural" does the
# claim/context mismatch become a contradiction.
_URBAN_PHRASES = (
    "city centre", "city center", "town centre", "town center",
    "downtown", "inner city", "market square", "main square",
    "city square", "town square", "central square", "market place",
    "marketplace", "old town", "centro storico", "centro historico",
    "centro", "altstadt", "stadtzentrum", "zentrum", "cbd",
    "business district",
)
# place= values worth reporting as the area's character.
_PLACE_TYPES = {
    "city", "town", "suburb", "village", "hamlet", "neighbourhood",
    "quarter", "locality", "isolated_dwelling", "borough",
}


# --------------------------------------------------------------------------
# Small pure helpers (stdlib only).
# --------------------------------------------------------------------------

def _norm(text) -> str:
    """lower, fold accents (and ß->ss etc.), punctuation -> space."""
    s = unicodedata.normalize("NFKD", str(text or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    for src, dst in _FOLD_MAP.items():
        s = s.replace(src, dst)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_hit(a: str, b: str) -> bool:
    """Token equality or >=3-char containment either way (husum/husumer)."""
    if a == b:
        return True
    if len(a) >= _MIN_SUBSTR and a in b:
        return True
    return len(b) >= _MIN_SUBSTR and b in a


def _names_match(a_norm: str, b_norm: str) -> bool:
    """Fuzzy name equality on normalized strings — same shape as
    overpass.name_matches_nearby: squashed containment, else >=60% content-
    token overlap, plus a single-token substring leniency."""
    if not a_norm or not b_norm:
        return False
    sa, sb = a_norm.replace(" ", ""), b_norm.replace(" ", "")
    if (len(sa) >= _MIN_SUBSTR and len(sb) >= _MIN_SUBSTR
            and (sa in sb or sb in sa)):
        return True
    ta = {t for t in a_norm.split() if t not in _STOP}
    tb = {t for t in b_norm.split() if t not in _STOP}
    if not ta or not tb:
        return False
    if len(ta) < 2 or len(tb) < 2:
        return any(_token_hit(x, y) for x in ta for y in tb)
    return len(ta & tb) / min(len(ta), len(tb)) >= _TOKEN_OVERLAP_MIN


def _content_tokens(q_norm: str) -> set:
    """Identity tokens of a claim: non-stop, non-descriptor words; falls
    back to non-stop words when the claim is *only* descriptors."""
    toks = {t for t in q_norm.split() if t not in _STOP}
    return (toks - _DESCRIPTORS) or toks


def _claims_urban_core(claim) -> bool:
    """True when the claim text asserts a city-centre-style location."""
    q = _norm(claim)
    return bool(q) and any(p in q for p in _URBAN_PHRASES)


def _classify_density(poi_count) -> str:
    """poi_count -> 'urban' | 'suburban' | 'rural'."""
    try:
        n = int(poi_count or 0)
    except (TypeError, ValueError):
        n = 0
    if n >= URBAN_MIN_POIS:
        return "urban"
    if n >= SUBURBAN_MIN_POIS:
        return "suburban"
    return "rural"


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle metres — verbatim shape of geo._haversine_km so this
    module stays dep-free at import."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------
# Overpass query -> condensed context dict.
# --------------------------------------------------------------------------

def _context_ql(lat: float, lon: float, radius: int) -> str:
    """One union query: density nodes + named ways (roads/squares)."""
    clauses = [
        f'node["{k}"](around:{radius},{lat},{lon});' for k in _NODE_KEYS
    ]
    clauses += [
        f'way["{k}"]["name"](around:{radius},{lat},{lon});'
        for k in _WAY_KEYS
    ]
    return (
        "[out:json][timeout:15];"
        f"({''.join(clauses)});"
        f"out center {MAX_ELEMENTS};"  # center: tags + lat/lon for ways too
    )


def _el_latlon(el: dict):
    """(lat, lon) of a node or a way's center; None if absent."""
    if not isinstance(el, dict):
        return None
    try:
        return float(el["lat"]), float(el["lon"])
    except (KeyError, TypeError, ValueError):
        pass
    c = el.get("center") or {}
    try:
        return float(c["lat"]), float(c["lon"])
    except (KeyError, TypeError, ValueError):
        return None


def _primary_kind(tags: dict):
    """First _KIND_PRIORITY key the element carries ('name' as fallback)."""
    for k in _KIND_PRIORITY:
        if tags.get(k):
            return k
    return "name" if tags.get("name") else None


def _context_from_payload(payload: dict, lat: float, lon: float,
                          radius: int) -> dict:
    """Raw Overpass payload -> the condensed context dict."""
    kinds: dict[str, int] = {}
    notable: list[dict] = []
    roads: dict[str, float] = {}  # name -> nearest dist_m seen
    seen_el: set = set()
    seen_notable: set = set()
    square = None
    place_type = None
    poi_count = 0

    for el in payload.get("elements") or []:
        if not isinstance(el, dict):
            continue
        key = (el.get("type"), el.get("id"))
        if key in seen_el:
            continue  # union members repeat across clauses
        seen_el.add(key)
        tags = el.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        name = str(tags.get("name") or "").strip()
        kind = _primary_kind(tags)
        if kind is None:
            continue  # unusable element (can't happen given the query)
        poi_count += 1
        kinds[kind] = kinds.get(kind, 0) + 1

        ll = _el_latlon(el)
        dist = _haversine_m(lat, lon, ll[0], ll[1]) if ll else None

        if name and tags.get("highway"):
            prev = roads.get(name)
            if prev is None or (dist is not None and dist < prev):
                roads[name] = dist if dist is not None else prev

        pt = str(tags.get("place") or "").strip()
        if square is None and (
            pt == "square"
            or (tags.get("highway") == "pedestrian"
                and str(tags.get("area") or "") == "yes" and name)
        ):
            square = name or "square"
        if place_type is None and pt in _PLACE_TYPES:
            place_type = pt

        if name and (name, kind) not in seen_notable:
            seen_notable.add((name, kind))
            notable.append({
                "name": name,
                "kind": kind,
                "value": str(tags.get(kind) or name) if kind != "name" else name,
            })

    road_names = sorted(
        roads, key=lambda n: (roads[n] is None,
                              roads[n] if roads[n] is not None else 0.0)
    )
    return {
        "ok": True,
        "lat": lat,
        "lon": lon,
        "radius_m": radius,
        "poi_count": poi_count,
        "kinds": kinds,
        "notable": notable[:_NOTABLE_CAP],
        "roads": road_names[:_ROAD_CAP],
        "nearest_road": road_names[0] if road_names else None,
        "square": square,
        "place_type": place_type,
        "density": _classify_density(poi_count),
    }


# --------------------------------------------------------------------------
# Public API — none of these raise.
# --------------------------------------------------------------------------

def context_at(lat, lon, radius_m: int = 800) -> dict:
    """Describe what's around (lat, lon): POI density, kinds, names, roads.

    Returns {"ok": True, poi_count, kinds, notable[{name,kind,value}],
             roads[], nearest_road, square, place_type, density, ...} or
    exactly {"ok": False} on bad input / offline / Overpass failure.
    """
    try:
        flat, flon = float(lat), float(lon)
        if not (-90.0 <= flat <= 90.0 and -180.0 <= flon <= 180.0):
            return {"ok": False}
        radius = max(50, min(int(radius_m), 2000))
    except Exception:
        return {"ok": False}
    try:
        from . import overpass  # lazy: network deps live in that module
        post = getattr(overpass, "_post_ql", None)
        if post is None:
            return {"ok": False}
        payload = post(_context_ql(flat, flon, radius))
        if not isinstance(payload, dict):
            return {"ok": False}
        return _context_from_payload(payload, flat, flon, radius)
    except Exception:
        return {"ok": False}


def context_match(context, claim_place) -> dict:
    """Fuzzy-match the claimed place against names found at the anchor.

    -> {"matched": [names], "score": 0..1} where score is the fraction of
    the claim's identity tokens explained by matched names (0 on no match,
    offline ctx, or empty claim).
    """
    try:
        if not isinstance(context, dict) or not context.get("ok"):
            return {"matched": [], "score": 0.0}
        q = _norm(claim_place)
        if not q:
            return {"matched": [], "score": 0.0}

        names: list[str] = []
        for item in context.get("notable") or []:
            nm = str((item or {}).get("name") or "").strip() \
                if isinstance(item, dict) else ""
            if nm and nm not in names:
                names.append(nm)
        for r in context.get("roads") or []:
            nm = str(r or "").strip()
            if nm and nm not in names:
                names.append(nm)

        content = _content_tokens(q)
        # Claim restricted to identity tokens, original order kept — so
        # "Husum city centre" can still match "Rathaus Husum" (descriptor
        # words would otherwise dilute token overlap below the threshold).
        q_core = " ".join(t for t in q.split() if t in content)
        matched: list[str] = []
        covered: set = set()
        for name in names:
            n_norm = _norm(name)
            if not n_norm:
                continue
            if not (_names_match(q, n_norm)
                    or (q_core != q and _names_match(q_core, n_norm))):
                continue
            matched.append(name)
            n_toks = set(n_norm.split())
            for qt in content:
                if any(_token_hit(qt, nt) for nt in n_toks):
                    covered.add(qt)
        if content:
            score = len(covered) / len(content)
        else:
            score = 1.0 if matched else 0.0
        return {"matched": matched, "score": round(min(1.0, score), 3)}
    except Exception:
        return {"matched": [], "score": 0.0}


def _schema_types():
    """Lazy schema import — relative first, then same-dir fallback so a
    bare `python osm_context.py` run still resolves it. None if hopeless."""
    try:
        from .schema import ClaimVerdict, Evidence
        return ClaimVerdict, Evidence
    except Exception:
        try:
            from schema import ClaimVerdict, Evidence  # noqa — bare run
            return ClaimVerdict, Evidence
        except Exception:
            return None


def context_evidence(ctx, claim_place):
    """Package an OSM context check as Evidence (schema imported lazily).

    notable/road name matching the claim -> SUPPORTS 0.5; an urban-core
    claim landing on rural ground -> CONTRADICTS 0.4; everything else
    (including offline/no ctx) -> INCONCLUSIVE. data carries the ctx.
    Returns None only if the schema itself cannot be imported.
    """
    try:
        types = _schema_types()
        if types is None:
            return None
        ClaimVerdict, Evidence = types
        ok = isinstance(ctx, dict) and bool(ctx.get("ok"))
        match = context_match(ctx, claim_place)
        data = {
            "claim_place": str(claim_place or ""),
            "match": match,
            "context": ctx if isinstance(ctx, dict) else {"ok": False},
            "context_summary": summarize(ctx),
        }
        if not ok:
            return Evidence(
                type="osm_context",
                summary=("no OSM context at the claim anchor — "
                         "lookup failed or offline"),
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.1,
                data=data,
            )
        dens = str(ctx.get("density") or "?")
        n_poi = ctx.get("poi_count", 0)
        rad = ctx.get("radius_m", "?")
        if match["matched"]:
            shown = ", ".join(repr(n) for n in match["matched"][:3])
            extra = (f" +{len(match['matched']) - 3} more"
                     if len(match["matched"]) > 3 else "")
            return Evidence(
                type="osm_context",
                summary=(
                    f"OSM context: named place(s) {shown}{extra} exist "
                    f"near the anchor ({dens}, {n_poi} POIs in {rad} m) "
                    "— consistent with the claimed place"
                ),
                verdict=ClaimVerdict.SUPPORTS,
                confidence=0.5,
                data=data,
            )
        if _claims_urban_core(claim_place) and dens == "rural":
            return Evidence(
                type="osm_context",
                summary=(
                    f"claim implies an urban core but OSM context is "
                    f"rural — only {n_poi} POI(s) within {rad} m of "
                    "the anchor"
                ),
                verdict=ClaimVerdict.CONTRADICTS,
                confidence=0.4,
                data=data,
            )
        return Evidence(
            type="osm_context",
            summary=(
                f"OSM context shows a {dens} area ({n_poi} POIs in "
                f"{rad} m) — no name match or contradiction for the claim"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.3,
            data=data,
        )
    except Exception:
        try:
            return Evidence(
                type="osm_context",
                summary="osm context evaluation failed internally",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.1,
                data={},
            )
        except Exception:
            return None


def summarize(ctx) -> str:
    """One-line digest: "urban square, 23 POIs in 800m, near Rathausstraße"."""
    try:
        if not isinstance(ctx, dict) or not ctx.get("ok"):
            return "no OSM context (lookup failed or offline)"
        density = str(ctx.get("density") or "unknown")
        word = "square" if ctx.get("square") else (ctx.get("place_type")
                                                   or "area")
        s = f"{density} {word}, {ctx.get('poi_count', 0)} POIs in " \
            f"{ctx.get('radius_m', '?')}m"
        road = ctx.get("nearest_road")
        s += f", near {road}" if road else ", no named road"
        return s
    except Exception:
        return "no OSM context (lookup failed or offline)"


# --------------------------------------------------------------------------
# Self-test: python -m sundial.osm_context — runs fully offline; a guarded
# live query runs only if Overpass happens to be reachable.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    fails = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        global fails
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label} {detail}")

    # --- 1. density classification (pure logic) ----------------------------
    print("density classification self-check:")
    check("45 POIs -> urban", _classify_density(45) == "urban")
    check("30 POIs -> urban (boundary)",
          _classify_density(URBAN_MIN_POIS) == "urban")
    check("29 POIs -> suburban",
          _classify_density(URBAN_MIN_POIS - 1) == "suburban")
    check("6 POIs -> suburban (boundary)",
          _classify_density(SUBURBAN_MIN_POIS) == "suburban")
    check("5 POIs -> rural",
          _classify_density(SUBURBAN_MIN_POIS - 1) == "rural")
    check("0 POIs -> rural", _classify_density(0) == "rural")

    # --- 2. fake ctx: Husum market square ----------------------------------
    husum_ctx = {
        "ok": True, "lat": 54.4769, "lon": 9.0558, "radius_m": 800,
        "poi_count": 23,
        "kinds": {"amenity": 9, "shop": 7, "highway": 5,
                  "place": 1, "tourism": 1},
        "notable": [
            {"name": "Marktplatz", "kind": "place", "value": "square"},
            {"name": "Rathaus Husum", "kind": "amenity", "value": "townhall"},
            {"name": "Café Central", "kind": "amenity", "value": "cafe"},
            {"name": "Schloss Husum", "kind": "tourism", "value": "attraction"},
            {"name": "Husumer Straße", "kind": "highway",
             "value": "residential"},
        ],
        "roads": ["Rathausstraße", "Husumer Straße", "Marktstraße"],
        "nearest_road": "Rathausstraße",
        "square": "Marktplatz",
        "place_type": "town",
        "density": "urban",
    }
    rural_ctx = {
        "ok": True, "lat": 54.30, "lon": 9.40, "radius_m": 800,
        "poi_count": 2,
        "kinds": {"highway": 2},
        "notable": [{"name": "Feldweg", "kind": "highway", "value": "track"}],
        "roads": ["Feldweg"],
        "nearest_road": "Feldweg",
        "square": None,
        "place_type": "hamlet",
        "density": "rural",
    }

    print("context_match self-check (fake ctx):")
    m = context_match(husum_ctx, "Husum")
    check("'Husum' matches >=1 name", len(m["matched"]) >= 1,
          f"-> {m['matched']}")
    check("'Husum' hits Rathaus Husum", "Rathaus Husum" in m["matched"])
    check("'Husum' hits Husumer Straße", "Husumer Straße" in m["matched"])
    check("'Husum' score > 0", m["score"] > 0, f"score={m['score']}")
    m2 = context_match(husum_ctx, "Husum city centre")
    check("'Husum city centre' still matches", len(m2["matched"]) >= 1)
    check("descriptor tokens covered fully", m2["score"] == 1.0,
          f"score={m2['score']}")
    m3 = context_match(husum_ctx, "Munich")
    check("'Munich' no match", m3["matched"] == [] and m3["score"] == 0.0)
    check("offline ctx -> 0 score",
          context_match({"ok": False}, "Husum")["score"] == 0.0)
    check("empty claim -> no match",
          context_match(husum_ctx, "") == {"matched": [], "score": 0.0})
    check("None inputs -> no raise",
          context_match(None, None) == {"matched": [], "score": 0.0})
    check("sharp-s fold: 'Husumer Strasse' matches",
          bool(context_match(husum_ctx, "Husumer Strasse")["matched"]))

    # --- 3. payload -> ctx shaping (pure logic) -----------------------------
    print("context shaping self-check:")
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": 54.4770, "lon": 9.0560,
         "tags": {"amenity": "cafe", "name": "Café Central"}},
        {"type": "way", "id": 2,
         "center": {"lat": 54.4771, "lon": 9.0559},
         "tags": {"highway": "residential", "name": "Rathausstraße"}},
        {"type": "node", "id": 3, "lat": 54.4768, "lon": 9.0557,
         "tags": {"place": "square", "name": "Marktplatz"}},
        {"type": "node", "id": 1, "lat": 54.4770, "lon": 9.0560,
         "tags": {"amenity": "cafe", "name": "Café Central"}},  # dup id
    ]}
    shaped = _context_from_payload(payload, 54.4769, 9.0558, 800)
    check("dup element deduped", shaped["poi_count"] == 3,
          f"-> {shaped['poi_count']}")
    check("kinds counted", shaped["kinds"] == {"amenity": 1, "highway": 1,
                                               "place": 1})
    check("way center -> nearest road",
          shaped["nearest_road"] == "Rathausstraße")
    check("square detected", shaped["square"] == "Marktplatz")
    check("tiny count -> rural", shaped["density"] == "rural")
    check("bad coords -> not ok", context_at(91, 0) == {"ok": False})
    check("non-numeric -> not ok", context_at("x", "y") == {"ok": False})

    # --- 4. summarize -------------------------------------------------------
    print("summarize self-check:")
    want = "urban square, 23 POIs in 800m, near Rathausstraße"
    check("spec example string", summarize(husum_ctx) == want,
          f"-> {summarize(husum_ctx)!r}")
    check("no ctx -> fallback line",
          "no OSM context" in summarize({"ok": False}))
    check("None -> no raise", isinstance(summarize(None), str))

    # --- 5. evidence verdicts (needs .schema; -m run provides it) -----------
    print("evidence self-check:")
    ev_bad = context_evidence({"ok": False}, "Husum")
    if ev_bad is None:
        print("  (schema unavailable — run as `python -m sundial.osm_context`)")
    else:
        check("offline ctx -> INCONCLUSIVE",
              ev_bad.verdict.value == "inconclusive" and ev_bad.confidence == 0.1,
              f"-> {ev_bad.verdict.value}")
        ev1 = context_evidence(husum_ctx, "Husum")
        check("name match -> SUPPORTS 0.5",
              ev1.verdict.value == "supports" and ev1.confidence == 0.5,
              f"-> {ev1.verdict.value} conf={ev1.confidence}")
        check("evidence type tag", ev1.type == "osm_context")
        check("data carries ctx", ev1.data.get("context", {}).get("ok") is True)
        ev2 = context_evidence(rural_ctx, "Husum city centre")
        check("'city centre' on rural -> CONTRADICTS 0.4",
              ev2.verdict.value == "contradicts" and ev2.confidence == 0.4,
              f"-> {ev2.verdict.value} conf={ev2.confidence}")
        ev3 = context_evidence(husum_ctx, "Munich")
        check("no match -> INCONCLUSIVE",
              ev3.verdict.value == "inconclusive" and ev3.confidence == 0.3)
        ev4 = context_evidence(rural_ctx, "village near Husum")
        check("rural claim on rural ctx -> INCONCLUSIVE",
              ev4.verdict.value == "inconclusive",
              f"-> {ev4.verdict.value}")
        check("None ctx -> no raise, INCONCLUSIVE",
              context_evidence(None, "x").verdict.value == "inconclusive")

    print(f"\nself-test: fails={fails}")
    if fails:
        sys.exit(1)

    # --- 6. guarded live demo ------------------------------------------------
    live = context_at(54.4769, 9.0558, radius_m=800)  # Husum, DE
    if not live.get("ok"):
        print("live context_at: SKIP (offline or Overpass unavailable)")
        sys.exit(0)
    print(f"live context_at(54.4769, 9.0558): {summarize(live)}")
    print(f"kinds={live['kinds']}")
    print("notable:", [n["name"] for n in live["notable"][:8]])
