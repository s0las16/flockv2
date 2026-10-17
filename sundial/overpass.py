"""POI cross-check — Overpass API (OSM) vs business/landmark names read by the VLM.

Free API, no key. When the claim anchors the clip to lat/lon and the VLM has
read sign text like "Pret A Manger" or "The Red Lion", a named POI of that
name existing within the claimed area is real corroboration.

Caveats that shape the scoring:
  * OSM name coverage is uneven — a miss is naming variance, never a
    contradiction, so the no-match case stays INCONCLUSIVE.
  * "out tags 50" caps the union at 50 elements; dense areas may truncate.
  * Overpass is rate-limited shared infrastructure — callers should query
    once per clip, not per name.
  * HTTP uses requests when importable (house style) with a lazy urllib
    fallback, so the module works with zero third-party deps.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, is_dataclass

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "sundial-hackathon/0.1 (video provenance verification)"
HTTP_TIMEOUT_S = 20  # headroom over the in-query [timeout:15]
MAX_ELEMENTS = 50

_TAG_KEY_CLEAN = re.compile(r"[^A-Za-z0-9_]+")
_MIN_CONTAIN_CHARS = 3  # shorter substrings false-positive too easily
_TOKEN_OVERLAP_MIN = 0.60


def _post_ql(ql: str) -> dict | None:
    """POST one Overpass QL query. Returns the parsed payload or None."""
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": ql},
                headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT_S,
            )
            if resp.status_code != 200:
                return None
            payload = resp.json()
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    # stdlib fallback — lazy import so the module never hard-requires urllib.
    import urllib.parse
    import urllib.request

    try:
        body = urllib.parse.urlencode({"data": ql}).encode("utf-8")
        req = urllib.request.Request(
            OVERPASS_URL,
            data=body,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _clean_kinds(kinds) -> list[str]:
    """Sanitize tag keys for interpolation into QL (dedupe, order kept)."""
    clean: list[str] = []
    for k in kinds or ():
        k = _TAG_KEY_CLEAN.sub("", str(k))
        if k and k not in clean:
            clean.append(k)
    return clean


def _to_poi(el: dict, kinds: list[str]) -> dict | None:
    """One Overpass element -> {name, lat, lon, kind, value}; None if unusable."""
    tags = el.get("tags") or {}
    name = str(tags.get("name") or "").strip()
    if not name:
        return None  # name-bearing only — unnamed features can't be matched
    try:
        lat, lon = float(el["lat"]), float(el["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    # kind = first requested tag the element carries (caller's priority
    # order); a bare `node["name"]` hit reports kind="name" / value=name.
    kind, value = "name", name
    for k in kinds:
        if k != "name" and tags.get(k):
            kind, value = k, str(tags[k])
            break
    return {"name": name, "lat": lat, "lon": lon, "kind": kind, "value": value}


def nearby_pois(
    lat: float,
    lon: float,
    radius_m: int = 3000,
    kinds=("amenity", "shop", "tourism", "name"),
) -> list[dict]:
    """Named POIs within `radius_m` of (lat, lon) via the Overpass API.

    Returns [{"name", "lat", "lon", "kind", "value"}] — elements carrying a
    `name` tag only. [] on any failure (offline, HTTP error, bad payload,
    empty area); never raises.
    """
    try:
        flat, flon = float(lat), float(lon)
        if not (-90 <= flat <= 90 and -180 <= flon <= 180):
            return []
        radius = max(1, int(radius_m))
        clean = _clean_kinds(kinds)
        if not clean:
            return []
        clauses = "".join(
            f'node["{k}"](around:{radius},{flat},{flon});' for k in clean
        )
        ql = f"[out:json][timeout:15];({clauses});out tags {MAX_ELEMENTS};"
        payload = _post_ql(ql)
        if not payload:
            return []
        pois: list[dict] = []
        seen: set = set()
        for el in payload.get("elements") or []:
            if not isinstance(el, dict):
                continue
            key = (el.get("type"), el.get("id"))
            if key in seen:
                continue  # union members can repeat across kinds
            seen.add(key)
            p = _to_poi(el, clean)
            if p:
                pois.append(p)
        return pois
    except Exception:
        return []


def _norm(text: str) -> str:
    """lower, fold accents, punctuation -> space, collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(text or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_matches_nearby(pois: list[dict], names: list[str]) -> list[dict]:
    """Fuzzy-match extracted names against nearby POI names.

    A name matches a POI when, after normalization:
      * one is a substring of the other (spaces/punctuation ignored), or
      * token overlap >= 60% of the shorter name — both multiword
        (catches "Red Lion Pub" vs "The Red Lion" where containment fails).

    Returns [{"query", "poi_name", "poi"}] — one entry per (name, poi) hit.
    """
    norm_pois: list[tuple[dict, str]] = []
    for p in pois or []:
        if not isinstance(p, dict):
            continue
        pn = _norm(p.get("name"))
        if pn:
            norm_pois.append((p, pn))
    if not norm_pois:
        return []

    matches: list[dict] = []
    seen_queries: set[str] = set()
    seen_pairs: set[tuple] = set()
    for raw in names or []:
        q_raw = str(raw or "").strip()
        q = _norm(q_raw)
        if not q or q in seen_queries:
            continue
        seen_queries.add(q)
        q_squash = q.replace(" ", "")
        q_tokens = set(q.split())
        for p, pn in norm_pois:
            if (q, id(p)) in seen_pairs:
                continue
            p_squash = pn.replace(" ", "")
            hit = (
                len(q_squash) >= _MIN_CONTAIN_CHARS
                and len(p_squash) >= _MIN_CONTAIN_CHARS
                and (q_squash in p_squash or p_squash in q_squash)
            )
            if not hit and len(q_tokens) >= 2:
                p_tokens = set(pn.split())
                if len(p_tokens) >= 2:
                    overlap = len(q_tokens & p_tokens) / min(
                        len(q_tokens), len(p_tokens)
                    )
                    hit = overlap >= _TOKEN_OVERLAP_MIN
            if hit:
                seen_pairs.add((q, id(p)))
                matches.append(
                    {"query": q_raw, "poi_name": p.get("name"), "poi": p}
                )
    return matches


def _anchor_data(anchor):
    """Best-effort JSON-friendly rendering of the claim anchor."""
    if anchor is None:
        return None
    try:
        if is_dataclass(anchor) and not isinstance(anchor, type):
            return asdict(anchor)
    except Exception:
        pass
    if isinstance(anchor, dict):
        return dict(anchor)
    if isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
        return {"lat": anchor[0], "lon": anchor[1]}
    return anchor


def poi_evidence(matches: list[dict], queried: bool, anchor) -> "Evidence":
    """Package POI name matches as an Evidence card (schema imported lazily).

    `queried` asserts the Overpass lookup ran and produced POIs to check
    against — pass the fetched poi list itself for the same effect (a
    non-empty list is truthy and also supplies the real poi count).
    `anchor` is the claim's lat/lon in whatever form the caller holds
    (tuple, GeoCandidate, dict) — without it there was nothing to query.

    Scoring: >=1 match -> SUPPORTS 0.6 (+0.05/extra, cap 0.75); POIs exist
    but no name hit -> INCONCLUSIVE 0.3 (naming varies — not a
    contradiction); no anchor / not queried -> INCONCLUSIVE 0.1.
    """
    from .schema import ClaimVerdict, Evidence

    matches = [m for m in (matches or []) if isinstance(m, dict)]
    poi_count = (
        len(queried)
        if isinstance(queried, (list, tuple))
        else len({id((m.get("poi") or {})) for m in matches})
    )
    data = {
        "matches": matches,
        "n_matches": len(matches),
        "poi_count": poi_count,
        "queried": bool(queried),
        "anchor": _anchor_data(anchor),
    }

    if matches:
        conf = min(0.6 + 0.05 * (len(matches) - 1), 0.75)
        top = matches[0].get("poi_name") or "?"
        return Evidence(
            type="poi",
            summary=(
                f"{len(matches)} extracted name(s) match named POIs near the "
                f"claimed coordinates — e.g. {top!r}"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=conf,
            data=data,
        )
    if queried and anchor:
        return Evidence(
            type="poi",
            summary=(
                f"{poi_count} named POI(s) found near the claimed "
                "coordinates but none match the extracted names — naming "
                "varies, so this neither confirms nor refutes"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.3,
            data=data,
        )
    return Evidence(
        type="poi",
        summary="no claim anchor or Overpass lookup not run — POI check n/a",
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.1,
        data=data,
    )


if __name__ == "__main__":
    import sys

    # Online demo: named POIs around Trafalgar Square, London.
    print("overpass demo — named POIs near Trafalgar Square, London")
    pois = nearby_pois(51.5080, -0.1281, radius_m=600)
    if pois:
        print(f"{len(pois)} named POIs within 600 m:")
        for p in pois[:15]:
            print(
                f"  {p['name']!r:45} {p['kind']}={p['value']}  "
                f"{p['lat']:.5f},{p['lon']:.5f}"
            )
        names = [
            "Trafalgar Square",
            "Pret A Manger",
            "The National Gallery",
            "St Martin-in-the-Fields",
        ]
        matches = name_matches_nearby(pois, names)
        print(f"\nname_matches_nearby({names}):")
        for m in matches:
            print(f"  {m['query']!r} -> {m['poi_name']!r}")
        try:
            ev = poi_evidence(matches, queried=pois, anchor=(51.5080, -0.1281))
        except ImportError:
            print("(run as `python -m sundial.overpass` to build Evidence)")
        else:
            print(
                f"evidence: {ev.verdict.value} conf={ev.confidence} "
                f"— {ev.summary}"
            )
        sys.exit(0)

    # Offline guard: no Overpass response -> self-test the matcher instead.
    print("no Overpass response (offline?) — offline self-test\n")
    fake_pois = [
        {"name": "Pret A Manger", "lat": 51.509, "lon": -0.126,
         "kind": "amenity", "value": "cafe"},
        {"name": "The Red Lion", "lat": 51.507, "lon": -0.131,
         "kind": "amenity", "value": "pub"},
        {"name": "St Martin-in-the-Fields", "lat": 51.509, "lon": -0.127,
         "kind": "amenity", "value": "place_of_worship"},
        {"name": "Café Nero", "lat": 51.508, "lon": -0.129,
         "kind": "amenity", "value": "cafe"},
    ]
    checks = [
        # (query, expected poi_name or None)
        ("pret a manger", "Pret A Manger"),                    # case fold
        ("RED LION", "The Red Lion"),                          # containment w/ "The"
        ("Red Lion Public House", "The Red Lion"),             # token overlap
        ("St Martins in the Fields", "St Martin-in-the-Fields"),  # punctuation
        ("Cafe Nero", "Café Nero"),                            # accent fold
        ("Buckingham Palace Gift Shop", None),                 # absent -> no match
        ("Av", None),                                          # too short to contain
    ]
    fails = 0
    for query, want in checks:
        got = name_matches_nearby(fake_pois, [query])
        got_name = got[0]["poi_name"] if got else None
        ok = "PASS" if got_name == want else "FAIL"
        if got_name != want:
            fails += 1
        print(f"  {ok}  match({query!r}) = {got_name!r} (want {want!r})")
    print(f"matcher self-test: {len(checks) - fails}/{len(checks)} passed")
    if fails:
        sys.exit(1)

    try:
        all_matches = name_matches_nearby(
            fake_pois, ["Pret A Manger", "Red Lion", "Nero"]
        )
        ev1 = poi_evidence(all_matches, queried=fake_pois, anchor=(51.508, -0.128))
        ev2 = poi_evidence([], queried=fake_pois, anchor=(51.508, -0.128))
        ev3 = poi_evidence([], queried=False, anchor=None)
        for label, ev in (("matched", ev1), ("no-match", ev2), ("not-run", ev3)):
            print(
                f"  {label}: {ev.verdict.value} conf={ev.confidence} "
                f"— {ev.summary}"
            )
    except ImportError:
        print("(run as `python -m sundial.overpass` to build Evidence objects)")
