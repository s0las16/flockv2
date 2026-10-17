"""Geolocation stage — geocode sign/shop text read from frames via Nominatim (OSM).

Free API, no key. Usage policy: max ~1 req/s and an identifiable User-Agent,
so all calls funnel through a module-level rate limiter.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict

import requests

from .schema import ClaimVerdict, Evidence, GeoCandidate

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "sundial-hackathon/0.1 (video provenance verification)"
TIMEOUT_S = 10
_MIN_INTERVAL_S = 1.05  # Nominatim policy: <= 1 request/second

_last_call = [0.0]


class _SkipGeocode(Exception):
    """Internal: claim already parsed as coordinates — skip Nominatim."""


# --- small honest country-knowledge lookups (deliberately not exhaustive) ---

# Countries that keep left.
_LEFT_TRAFFIC = {
    "GB", "IE", "JP", "AU", "NZ", "IN", "ZA", "TH", "MY", "SG", "ID", "HK",
    "KE", "MT", "CY", "PK", "BD", "LK", "NP", "MU", "TT", "JM", "BB", "BS",
    "GY", "SR", "BN", "BT", "MO", "NA", "BW", "LS", "SZ", "MZ", "ZM", "MW",
    "TZ", "UG", "FJ", "PG",
}
# Yellow rear licence plate is a strong UK/NL tell.
_YELLOW_REAR_PLATE = {"GB", "NL"}
_PLATE_HINTS = {
    "uk": {"GB"}, "gb": {"GB"},
    "us": {"US"}, "ca": {"CA"}, "au": {"AU"}, "jp": {"JP"}, "cn": {"CN"},
    "in": {"IN"}, "kr": {"KR"}, "ru": {"RU"}, "br": {"BR"}, "za": {"ZA"},
    # EU-style blue strip with circle of stars.
    "eu": {"DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "PL", "IE", "SE",
           "DK", "FI", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "EE",
           "LV", "LT", "LU", "MT", "CY"},
}
# Dominant language of public signage per country (rough).
_LANG_HINTS = {
    "en": {"GB", "IE", "US", "AU", "NZ", "CA", "ZA", "IN", "SG", "NG", "KE", "JM", "MT", "GY"},
    "fr": {"FR", "BE", "CH", "LU", "MC", "CA", "SN", "CI", "ML", "BF", "NE", "TG", "BJ", "CD", "CG", "MG"},
    "de": {"DE", "AT", "CH", "LU", "LI"},
    "es": {"ES", "MX", "AR", "CO", "CL", "PE", "VE", "EC", "GT", "CU", "BO", "DO", "HN", "PY", "SV", "NI", "CR", "PA", "UY", "GQ"},
    "pt": {"PT", "BR", "AO", "MZ", "CV", "GW", "ST", "TL"},
    "it": {"IT", "CH", "SM", "VA"},
    "nl": {"NL", "BE", "SR", "AW", "CW"},
    "ja": {"JP"}, "ko": {"KR"}, "zh": {"CN", "TW", "HK", "MO", "SG"},
    "ar": {"SA", "AE", "EG", "QA", "KW", "BH", "OM", "JO", "LB", "IQ", "SY", "MA", "DZ", "TN", "LY", "SD", "YE", "PS", "MR", "DJ"},
    "ru": {"RU", "BY", "KZ", "KG"},
    "hi": {"IN"}, "th": {"TH"}, "vi": {"VN"}, "id": {"ID"}, "ms": {"MY", "BN"},
    "tr": {"TR", "CY"}, "el": {"GR", "CY"}, "pl": {"PL"}, "uk": {"UA"},
    "sv": {"SE"}, "da": {"DK"}, "no": {"NO"}, "fi": {"FI"}, "cs": {"CZ"},
    "sk": {"SK"}, "hu": {"HU"}, "ro": {"RO"}, "bg": {"BG"}, "hr": {"HR"},
    "he": {"IL"}, "fa": {"IR"}, "ur": {"PK"}, "bn": {"BD", "IN"},
}
_MARKING_HINTS = {
    "double_yellow_lines": {"GB", "IE"},
    "double_red_lines": {"GB"},
    "yellow_centerline": {"US", "CA", "BR", "CN", "KR", "TW", "NO", "SE", "FI"},
    "yellow_edge_line": {"US", "CA", "GB", "IE"},
    "zigzag_lines": {"GB", "IE", "SG"},
    "keep_left_bollards": {"GB", "IE", "AU", "NZ", "JP"},
}


def _rate_limit() -> None:
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def _to_candidate(item: dict) -> GeoCandidate | None:
    try:
        lat = float(item["lat"])
        lon = float(item["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    display = item.get("display_name") or ""
    cc = ((item.get("address") or {}).get("country_code") or "").upper()
    return GeoCandidate(
        name=item.get("name") or display.split(",")[0].strip() or "unknown",
        lat=lat,
        lon=lon,
        country_code=cc,
        display_name=display,
        score=float(item.get("importance") or 0.0),
    )


def geocode(query: str) -> list[GeoCandidate]:
    """Geocode a free-text string. Returns [] when nothing matches."""
    q = str(query or "").strip()
    if not q:
        return []
    _rate_limit()
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "q": q,
                "format": "json",
                "limit": 5,
                "addressdetails": 1,  # required for address.country_code
            },
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
            timeout=TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Nominatim request failed for {q!r}: {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"Nominatim HTTP {resp.status_code} for {q!r}: {resp.text[:200]}"
        )
    try:
        items = resp.json()
    except ValueError as e:
        raise RuntimeError(f"Nominatim returned non-JSON for {q!r}") from e
    if not isinstance(items, list):
        raise RuntimeError(
            f"unexpected Nominatim payload for {q!r}: {str(items)[:200]}"
        )
    return [c for c in (_to_candidate(i) for i in items) if c is not None]


def geocode_texts(texts: list[str]) -> list[GeoCandidate]:
    """Geocode each distinct text; rank places by how many queries hit them."""
    queries: list[str] = []
    seen: set[str] = set()
    for t in texts or []:
        q = (t or "").strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    freq: dict[tuple[float, float], int] = {}
    best: dict[tuple[float, float], GeoCandidate] = {}
    for q in queries:
        for c in geocode(q):
            key = (round(c.lat, 3), round(c.lon, 3))  # ~100 m dedupe bucket
            freq[key] = freq.get(key, 0) + 1
            if key not in best or c.score > best[key].score:
                best[key] = c
    out = list(best.values())
    for c in out:
        n = freq[(round(c.lat, 3), round(c.lon, 3))]
        # frequency dominates; Nominatim importance is a tiebreaker
        c.score = float(n) + min(c.score, 1.0) * 0.01
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:10]


def consistency_checks(observations: dict, candidate: GeoCandidate) -> list[str]:
    """Compare in-clip observations against what a country should look like."""
    obs = observations or {}
    cc = (candidate.country_code or "").upper()
    flags: list[str] = []
    if not cc:
        return flags

    side = (obs.get("driving_side") or "").strip().lower()
    if side in ("left", "right"):
        expected_left = cc in _LEFT_TRAFFIC
        if (side == "left") == expected_left:
            flags.append(f"drives on {side} — consistent with {cc}")
        else:
            exp = "left" if expected_left else "right"
            flags.append(
                f"drives on {side} — INCONSISTENT with {cc} (drives {exp})"
            )

    plate = (obs.get("plate_style") or "").strip().lower().replace("-", "_")
    if plate in ("yellow_rear", "yellow_rear_plate", "yellow_rear_plates"):
        ok = cc in _YELLOW_REAR_PLATE
        flags.append(
            f"yellow rear licence plates — "
            f"{'consistent' if ok else 'INCONSISTENT'} with {cc} "
            f"(yellow rear is typical of {sorted(_YELLOW_REAR_PLATE)})"
        )
    elif plate in _PLATE_HINTS:
        ok = cc in _PLATE_HINTS[plate]
        flags.append(
            f"{plate}-style licence plates — "
            f"{'consistent' if ok else 'INCONSISTENT'} with {cc}"
        )

    lang = (obs.get("sign_language") or "").strip().lower()
    if lang in _LANG_HINTS:
        ok = cc in _LANG_HINTS[lang]
        flags.append(
            f"signage language '{lang}' — "
            f"{'consistent' if ok else 'INCONSISTENT'} with {cc}"
        )

    for m in obs.get("road_markings") or []:
        key = str(m).strip().lower().replace(" ", "_").replace("-", "_")
        if key in _MARKING_HINTS:
            ok = cc in _MARKING_HINTS[key]
            flags.append(
                f"road marking '{m}' — "
                f"{'consistent' if ok else 'INCONSISTENT'} with {cc}"
            )
    return flags


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geo_evidence(
    claimed_place: str | None,
    texts: list[str],
    observations: dict,
) -> Evidence:
    """Geocode the claim and the extracted sign text, then compare where they point."""
    usable = [t.strip() for t in (texts or []) if t and t.strip()]
    data: dict = {"claimed_place": claimed_place, "texts": usable}
    if not usable:
        return Evidence(
            type="geo",
            summary="no usable sign/place text extracted from frames",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.1,
            data={**data, "candidates": []},
        )

    try:
        text_cands = geocode_texts(usable)
    except RuntimeError as e:
        return Evidence(
            type="geo",
            summary=f"geocoding of extracted text failed: {e}",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data={**data, "error": str(e)},
        )
    data["candidates"] = [asdict(c) for c in text_cands]
    if not text_cands:
        return Evidence(
            type="geo",
            summary=f"no geocoding match for {len(usable)} extracted text(s)",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )

    claim_cands: list[GeoCandidate] = []
    if claimed_place and claimed_place.strip():
        raw = claimed_place.strip()
        try:
            # "lat,lon" claims anchor directly — no geocode needed.
            a, b = [float(x) for x in raw.split(",")[:2]]
            if -90 <= a <= 90 and -180 <= b <= 180:
                claim_cands = [GeoCandidate(
                    name=raw, lat=a, lon=b, country_code="",
                    display_name=f"claimed coordinates {raw}", score=1.0)]
                raise _SkipGeocode()
            claim_cands = geocode(raw)
        except _SkipGeocode:
            pass
        except (ValueError, TypeError, IndexError):
            try:
                claim_cands = geocode(raw)
            except RuntimeError as e:
                data["claim_geocode_error"] = str(e)
        except RuntimeError as e:
            data["claim_geocode_error"] = str(e)
    data["claim_candidates"] = [asdict(c) for c in claim_cands]
    data["consistency_checks"] = {
        "vs_text_top": consistency_checks(observations, text_cands[0]),
    }

    if not claim_cands:
        return Evidence(
            type="geo",
            summary=(
                f"sign text geolocates to {text_cands[0].display_name!r} "
                f"but claimed place {claimed_place!r} could not be geocoded"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.25,
            data=data,
        )

    anchor = claim_cands[0]
    data["consistency_checks"]["vs_claim"] = consistency_checks(
        observations, anchor
    )
    claim_cc = anchor.country_code
    dists = {
        id(c): _haversine_km(anchor.lat, anchor.lon, c.lat, c.lon)
        for c in text_cands[:5]
    }
    # Very close beats country-code matching: a sign 5 km away across a
    # border (Strasbourg/Kehl) still supports the claim.
    very_near = [c for c in text_cands[:5] if dists[id(c)] <= 25.0]
    near = very_near or [
        c
        for c in text_cands[:5]
        if c.country_code == claim_cc and dists[id(c)] <= 60.0
    ]

    if near:
        hit = min(near, key=lambda c: dists[id(c)])
        data["match"] = {**asdict(hit), "distance_km": round(dists[id(hit)], 1)}
        conf = min(0.65 + 0.05 * (len(near) - 1), 0.85)
        return Evidence(
            type="geo",
            summary=(
                f"sign text geolocates to {hit.display_name!r} "
                f"({dists[id(hit)]:.0f} km from claimed {claimed_place!r})"
            ),
            verdict=ClaimVerdict.SUPPORTS,
            confidence=conf,
            data=data,
        )

    top = text_cands[0]
    d_top = dists[id(top)]
    data["match"] = {**asdict(top), "distance_km": round(d_top, 1)}
    if top.country_code == claim_cc and d_top > 400:
        return Evidence(
            type="geo",
            summary=(
                f"sign text points to {top.display_name!r}, ~{d_top:.0f} km "
                f"from claimed {claimed_place!r} (same country, wrong region)"
            ),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=0.55,
            data=data,
        )
    if claim_cc and all(
        c.country_code and c.country_code != claim_cc for c in text_cands[:3]
    ):
        return Evidence(
            type="geo",
            summary=(
                f"all geocoded sign text points to {top.country_code} "
                f"(e.g. {top.display_name!r}), not to claimed {claimed_place!r}"
            ),
            verdict=ClaimVerdict.CONTRADICTS,
            confidence=0.75,
            data=data,
        )
    return Evidence(
        type="geo",
        summary=(
            f"sign text candidates ({top.display_name!r}, {d_top:.0f} km away) "
            f"are too ambiguous to confirm or refute {claimed_place!r}"
        ),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.3,
        data=data,
    )
