"""Wikipedia-geosearch landmark cross-check — free, no key.

The VLM names landmarks it thinks it sees ("Eiffel Tower", "Tower Bridge").
Wikipedia's GeoData index tells us which *real* articles sit near the
claimed coordinates — if a named landmark is genuinely within the search
radius that's honest corroboration; if nothing nearby matches, absence is
weak evidence (coverage is uneven), never a contradiction.

MediaWiki API: action=query&list=geosearch&gscoord=lat|lon — JSON, no auth.
HTTP is lazy inside functions (requests preferred, stdlib urllib fallback),
so this module imports with zero third-party deps. `geo_search` never
raises — a failed lookup is just no data, not a crash.

  python -m sundial.wiki_places          # matcher self-test + live demo
"""
from __future__ import annotations

import json
import string
import sys

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "sundial-hackathon/0.1 (video provenance verification)"
TIMEOUT_S = 10
MAX_RADIUS_M = 10000  # GeoData caps gsradius at 10 km

# Token-overlap threshold for fuzzy name matching.
_TOKEN_OVERLAP_MIN = 0.6
# A substring shorter than this ("St", "Pi") matches everything — ignore.
_SUBSTR_MIN_LEN = 3
# Filler words that shouldn't count toward token overlap ("Statue of
# Liberty" vs "Statue of Unity" must not score 2/3 on {statue, of}).
_STOP = {
    "of", "the", "a", "an", "and", "at", "in", "on", "for", "to",
    "de", "del", "la", "le", "el", "di", "da", "du", "des",
    "von", "van", "en", "al",
}
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def _get_json(url: str, params: dict) -> dict:
    """GET `url` with query `params`, return parsed JSON.

    requests if installed, stdlib urllib otherwise. Raises RuntimeError.
    """
    try:
        import requests
    except ImportError:
        requests = None

    if requests is not None:
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT_S,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"MediaWiki request failed: {e}") from e
        if resp.status_code != 200:
            raise RuntimeError(
                f"MediaWiki HTTP {resp.status_code}: {resp.text[:160]!r}"
            )
        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError("MediaWiki returned a non-JSON response") from e

    import urllib.error
    import urllib.parse
    import urllib.request

    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"MediaWiki HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"MediaWiki request failed: {e.reason}") from e
    try:
        return json.loads(body)
    except ValueError as e:
        raise RuntimeError("MediaWiki returned a non-JSON response") from e


def geo_search(
    lat: float,
    lon: float,
    radius_m: int = MAX_RADIUS_M,
    limit: int = 20,
) -> list[dict]:
    """Wikipedia articles within `radius_m` of lat/lon -> [{title, lat, lon, dist_m}].

    Returns [] on bad input, network failure, or an empty result — this
    stage degrades to "no evidence" rather than ever raising.
    """
    try:
        lat = float(lat)
        lon = float(lon)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return []
        payload = _get_json(
            WIKI_API,
            {
                "action": "query",
                "list": "geosearch",
                "gscoord": f"{lat}|{lon}",
                "gsradius": max(10, min(int(radius_m), MAX_RADIUS_M)),
                "gslimit": max(1, min(int(limit), 500)),
                "format": "json",
            },
        )
        hits = ((payload or {}).get("query") or {}).get("geosearch") or []
        out: list[dict] = []
        for h in hits:
            try:
                title = str(h.get("title") or "").strip()
                if not title:
                    continue
                out.append(
                    {
                        "title": title,
                        "lat": float(h["lat"]),
                        "lon": float(h["lon"]),
                        "dist_m": float(h.get("dist") or 0.0),
                    }
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        return out
    except Exception:
        return []


def _norm(s: str) -> str:
    """Lowercase, punctuation -> spaces, collapse whitespace."""
    return " ".join(str(s or "").lower().translate(_PUNCT).split())


def _tokens(norm: str) -> set:
    """Content words of a normalized name (stopwords excluded)."""
    return {w for w in norm.split() if w not in _STOP}


def _names_match(a_norm: str, b_norm: str) -> bool:
    """Fuzzy landmark-name equality on already-normalized strings.

    True on substring containment either way (>= 3 chars on the shorter
    side) or >= 60% token overlap of the smaller name.
    """
    if not a_norm or not b_norm:
        return False
    short, long = (a_norm, b_norm) if len(a_norm) <= len(b_norm) else (b_norm, a_norm)
    if len(short) >= _SUBSTR_MIN_LEN and short in long:
        return True
    ta, tb = _tokens(a_norm), _tokens(b_norm)
    if not ta or not tb:
        return False
    return len(ta & tb) / max(1, min(len(ta), len(tb))) >= _TOKEN_OVERLAP_MIN


def match_named(titles: list[dict], names: list[str]) -> list[dict]:
    """Keep the geosearch hits whose title matches a VLM-named landmark.

    `titles` is a geo_search() result; `names` the landmark strings the VLM
    claims to see. Each surviving hit is copied and annotated with
    `matched_name` (the first name that hit it). Order follows `titles`
    (i.e. nearest-first); each title matches once.
    """
    normed = [(_norm(n), str(n or "").strip()) for n in names or []]
    normed = [(nn, raw) for nn, raw in normed if nn]
    out: list[dict] = []
    seen: set[str] = set()
    for hit in titles or []:
        t_norm = _norm((hit or {}).get("title"))
        if not t_norm or t_norm in seen:
            continue
        for name_norm, raw in normed:
            if _names_match(t_norm, name_norm):
                seen.add(t_norm)
                m = dict(hit)
                m["matched_name"] = raw
                out.append(m)
                break
    return out


def wiki_evidence(matches: list[dict], queried: bool, radius_m: int) -> "Evidence":
    """Package a geosearch cross-check as an Evidence card (schema lazy).

    A match means a real, Wikipedia-notable landmark sits near the claimed
    coordinates — SUPPORTS. No match after a successful query is only
    INCONCLUSIVE: geodata coverage is spotty and absence isn't proof.
    """
    from .schema import ClaimVerdict, Evidence

    matches = list(matches or [])
    data = {
        "queried": bool(queried),
        "radius_m": int(radius_m),
        "titles": [m.get("title") for m in matches],
        "matches": matches,
        "matched_names": sorted(
            {str(m["matched_name"]) for m in matches if m.get("matched_name")}
        ),
    }
    if not queried:
        return Evidence(
            type="wiki_landmark",
            summary="wikipedia geosearch not run — no landmark cross-check",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.1,
            data=data,
        )
    if not matches:
        return Evidence(
            type="wiki_landmark",
            summary=(
                f"no wikipedia title within {int(radius_m)} m matches a "
                "VLM-named landmark (absence isn't proof)"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.25,
            data=data,
        )
    near = min(matches, key=lambda m: m.get("dist_m") or 0.0)
    shown = ", ".join(repr(m.get("title")) for m in matches[:3])
    extra = f" +{len(matches) - 3} more" if len(matches) > 3 else ""
    return Evidence(
        type="wiki_landmark",
        summary=(
            f"wikipedia confirms named landmark(s) near the claim: "
            f"{shown}{extra} (nearest {near.get('dist_m') or 0.0:.0f} m)"
        ),
        verdict=ClaimVerdict.SUPPORTS,
        confidence=0.55,
        data=data,
    )


# --------------------------------------------------------------------------
# Demo: python -m sundial.wiki_places — matcher self-test runs offline;
# the live geosearch is guarded and degrades to SKIP without network.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    fake_hits = [
        {"title": "Tower Bridge", "lat": 51.5055, "lon": -0.0754, "dist_m": 120.0},
        {"title": "City Hall, London", "lat": 51.5046, "lon": -0.0772, "dist_m": 200.0},
        {"title": "HMS Belfast", "lat": 51.5066, "lon": -0.0814, "dist_m": 430.0},
        {"title": "Tower of London", "lat": 51.5081, "lon": -0.0759, "dist_m": 350.0},
    ]
    checks = [
        (["Tower Bridge"], {"Tower Bridge"}),
        (["tower bridge!!"], {"Tower Bridge"}),           # case + punctuation
        (["the tower bridge"], {"Tower Bridge"}),         # stopword substring
        (["City Hall"], {"City Hall, London"}),           # substring either way
        (["Belfast"], {"HMS Belfast"}),                   # token/substring
        (["Buckingham Palace"], set()),                   # absent -> no match
        (["London Bridge"], set()),                       # Tower Bridge != London Bridge
        (["Tower Bridge", "HMS Belfast"],
         {"Tower Bridge", "HMS Belfast"}),                # multi-name
    ]
    fails = 0
    for names, want in checks:
        got = {m["title"] for m in match_named(fake_hits, names)}
        ok = got == want
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  match_named({names}) -> {sorted(got)}")
    print(f"matcher self-test: {len(checks) - fails}/{len(checks)} passed")
    if fails:
        sys.exit(1)

    try:
        for m, q in (
            (match_named(fake_hits, ["Tower Bridge"]), True),
            ([], True),
            ([], False),
        ):
            ev = wiki_evidence(m, q, MAX_RADIUS_M)
            print(f"evidence[{ev.type}] {ev.verdict.value} "
                  f"conf={ev.confidence}: {ev.summary}")
    except ImportError:
        print("(run as `python -m sundial.wiki_places` to build Evidence objects)")

    # Live demo — guarded: [] means offline OR genuinely no articles.
    hits = geo_search(48.8583, 2.2945, radius_m=3000, limit=10)  # Eiffel Tower
    if not hits:
        print("live geosearch: SKIP (offline or no hits)")
        sys.exit(0)
    print(f"live geosearch: {len(hits)} titles near 48.8583,2.2945")
    for h in hits[:5]:
        print(f"  {h['dist_m']:7.0f} m  {h['title']}")
    matches = match_named(hits, ["Eiffel Tower", "Louvre"])
    print("matched:", [m["title"] for m in matches])
    try:
        ev = wiki_evidence(matches, True, 3000)
        print(f"evidence[{ev.type}] {ev.verdict.value} "
              f"conf={ev.confidence}: {ev.summary}")
    except ImportError:
        pass
