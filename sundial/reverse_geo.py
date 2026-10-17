"""Reverse-geocoding — anchor extracted coordinates to a real place.

Same free Nominatim API as geo.py (no key), so same etiquette: an
identifiable User-Agent and <= 1 request/second through a module-level
rate limiter. Like sources.py, requests is imported lazily and falls back
to stdlib urllib, so this module loads with zero third-party deps. Every
public function is failure-tolerant: bad input, network errors and
non-JSON replies all degrade to {} / None — nothing here raises.

    python -m sundial.reverse_geo
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "sundial-hackathon/0.1 (video provenance verification)"
TIMEOUT_S = 10
_MIN_INTERVAL_S = 1.05  # Nominatim policy: <= 1 request/second

_last_call = [0.0]

__all__ = ["reverse_geocode", "anchor_context", "claim_country_cc"]

# Nominatim address fields that can hold the locality, best first.
_CITY_KEYS = (
    "city", "town", "village", "municipality", "county", "hamlet",
    "city_district", "suburb",
)

# ISO 3166-1 alpha-2 — lets a "..., CC" claim tail resolve while rejecting
# non-country two-letter tails like "TX" (a US state, not a country).
_ISO_CC = frozenset(
    "AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ "
    "BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ "
    "CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ "
    "DE DJ DK DM DO DZ "
    "EC EE EG EH ER ES ET "
    "FI FJ FK FM FO FR "
    "GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY "
    "HK HM HN HR HT HU "
    "ID IE IL IM IN IO IQ IR IS IT "
    "JE JM JO JP "
    "KE KG KH KI KM KN KP KR KW KY KZ "
    "LA LB LC LI LK LR LS LT LU LV LY "
    "MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ "
    "NA NC NE NF NG NI NL NO NP NR NU NZ "
    "OM "
    "PA PE PF PG PH PK PL PM PN PR PS PT PW PY "
    "QA "
    "RE RO RS RU RW "
    "SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ "
    "TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ "
    "UA UG UM US UY UZ "
    "VA VC VE VG VI VN VU "
    "WF WS "
    "YE YT "
    "ZA ZM ZW".split()
)

# Common country names (and aliases) -> alpha-2. Deliberately not
# exhaustive, matching geo.py's small honest lookup tables.
_COUNTRY_NAME_CC = {
    "afghanistan": "AF", "albania": "AL", "algeria": "DZ", "andorra": "AD",
    "angola": "AO", "argentina": "AR", "armenia": "AM", "australia": "AU",
    "austria": "AT", "azerbaijan": "AZ",
    "bahamas": "BS", "bahrain": "BH", "bangladesh": "BD", "belarus": "BY",
    "belgium": "BE", "belize": "BZ", "bolivia": "BO", "bosnia": "BA",
    "bosnia and herzegovina": "BA", "botswana": "BW", "brazil": "BR",
    "brunei": "BN", "bulgaria": "BG",
    "cambodia": "KH", "cameroon": "CM", "canada": "CA", "chile": "CL",
    "china": "CN", "colombia": "CO", "costa rica": "CR", "croatia": "HR",
    "cuba": "CU", "cyprus": "CY", "czechia": "CZ", "czech republic": "CZ",
    "denmark": "DK", "dominican republic": "DO",
    "ecuador": "EC", "egypt": "EG", "el salvador": "SV", "estonia": "EE",
    "ethiopia": "ET",
    "finland": "FI", "france": "FR",
    "georgia": "GE", "germany": "DE", "ghana": "GH", "greece": "GR",
    "guatemala": "GT",
    "honduras": "HN", "hong kong": "HK", "hungary": "HU",
    "iceland": "IS", "india": "IN", "indonesia": "ID", "iran": "IR",
    "iraq": "IQ", "ireland": "IE", "israel": "IL", "italy": "IT",
    "ivory coast": "CI",
    "jamaica": "JM", "japan": "JP", "jordan": "JO",
    "kazakhstan": "KZ", "kenya": "KE", "kuwait": "KW",
    "laos": "LA", "latvia": "LV", "lebanon": "LB", "libya": "LY",
    "liechtenstein": "LI", "lithuania": "LT", "luxembourg": "LU",
    "malaysia": "MY", "mexico": "MX", "moldova": "MD", "monaco": "MC",
    "mongolia": "MN", "montenegro": "ME", "morocco": "MA",
    "myanmar": "MM",
    "namibia": "NA", "nepal": "NP", "netherlands": "NL",
    "new zealand": "NZ", "nicaragua": "NI", "nigeria": "NG",
    "north korea": "KP", "north macedonia": "MK", "norway": "NO",
    "oman": "OM",
    "pakistan": "PK", "panama": "PA", "paraguay": "PY", "peru": "PE",
    "philippines": "PH", "poland": "PL", "portugal": "PT",
    "qatar": "QA",
    "romania": "RO", "russia": "RU", "rwanda": "RW",
    "saudi arabia": "SA", "senegal": "SN", "serbia": "RS",
    "singapore": "SG", "slovakia": "SK", "slovenia": "SI",
    "south africa": "ZA", "south korea": "KR", "korea": "KR",
    "spain": "ES", "sri lanka": "LK", "sweden": "SE",
    "switzerland": "CH", "syria": "SY",
    "taiwan": "TW", "tanzania": "TZ", "thailand": "TH", "tunisia": "TN",
    "turkey": "TR", "türkiye": "TR",
    "uganda": "UG", "ukraine": "UA", "united arab emirates": "AE",
    "uae": "AE",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB",
    "britain": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
    "united states": "US", "united states of america": "US",
    "usa": "US", "us": "US", "u.s.": "US", "u.s.a.": "US",
    "america": "US", "uruguay": "UY", "uzbekistan": "UZ",
    "venezuela": "VE", "vietnam": "VN", "viet nam": "VN",
    "yemen": "YE",
    "zambia": "ZM", "zimbabwe": "ZW",
}


def _rate_limit() -> None:
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def _get_json(url: str, params: dict):
    """GET `url` with `params` as parsed JSON.

    Same transport style as geo.py (requests, identifiable User-Agent);
    falls back to stdlib urllib like sources.py when requests is absent.
    Raises RuntimeError on any failure.
    """
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    try:
        import requests
    except ImportError:
        requests = None

    if requests is None:
        full = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(full, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GET {url} -> HTTP {e.code}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"GET {url} failed: {e.reason}") from e
        try:
            return json.loads(body)
        except ValueError as e:
            raise RuntimeError(f"{url} returned non-JSON") from e

    try:
        resp = requests.get(
            url, params=params, headers=headers, timeout=TIMEOUT_S
        )
    except requests.RequestException as e:
        raise RuntimeError(f"GET {url} failed: {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"GET {url} -> HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError(f"{url} returned non-JSON") from e


def _looks_like_coords(s: str) -> bool:
    """True for a bare "lat,lon" string (same parse as geo.py's claims)."""
    try:
        a, b = [float(x) for x in str(s).split(",")[:2]]
    except (TypeError, ValueError):
        return False
    return -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0


def reverse_geocode(lat: float, lon: float) -> dict:
    """Reverse-geocode one coordinate pair via Nominatim /reverse.

    Returns {"display_name", "country_code", "country", "city", "road",
    "raw"} with values None where the response lacks them. Returns {} on
    any failure (bad coords, network error, non-200, non-JSON, Nominatim
    "unable to geocode"); never raises.
    """
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return {}
    if not (math.isfinite(la) and math.isfinite(lo)):
        return {}
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return {}
    _rate_limit()
    try:
        item = _get_json(
            NOMINATIM_REVERSE_URL,
            {
                "lat": la,
                "lon": lo,
                "format": "json",
                "addressdetails": 1,  # required for address.country_code
            },
        )
        if not isinstance(item, dict) or item.get("error"):
            return {}
        addr = item.get("address") or {}
        if not isinstance(addr, dict):
            addr = {}
        city = next((addr[k] for k in _CITY_KEYS if addr.get(k)), None)
        cc = str(addr.get("country_code") or "").upper() or None
        display = item.get("display_name")
        return {
            "display_name": str(display) if display else None,
            "country_code": cc,
            "country": addr.get("country"),
            "city": city,
            "road": addr.get("road"),
            "raw": item,
        }
    except Exception:  # noqa: BLE001 — spec: never raise, {} on failure
        return {}


def anchor_context(
    lat: float | None,
    lon: float | None,
    claimed_place: str | None,
) -> dict:
    """Merge a reverse-geocode hit with the raw claim into one anchor dict.

    Returns {"cc", "country", "locality", "source"}: cc/country always come
    from the reverse geocode (None when it fails); locality prefers the
    resolved city, then the claim's head segment, then road/display_name.
    source is "nominatim-reverse" or "claimed_place". Returns {} when the
    lookup fails and the claim itself is just "lat,lon" — nothing left to
    anchor to.
    """
    claim = (claimed_place or "").strip()
    claim_is_coords = bool(claim) and _looks_like_coords(claim)
    rev = (
        reverse_geocode(lat, lon)
        if lat is not None and lon is not None
        else {}
    )
    if not rev:
        if not claim or claim_is_coords:
            return {}
        return {
            "cc": None,
            "country": None,
            "locality": claim.split(",")[0].strip() or None,
            "source": "claimed_place",
        }
    locality = rev.get("city")
    if not locality and claim and not claim_is_coords:
        locality = claim.split(",")[0].strip() or None
    if not locality:
        locality = rev.get("road")
    if not locality:
        locality = (rev.get("display_name") or "").split(",")[0].strip() or None
    return {
        "cc": rev.get("country_code"),
        "country": rev.get("country"),
        "locality": locality,
        "source": "nominatim-reverse",
    }


def claim_country_cc(
    claimed_place: str | None,
    lat: float | None = None,
    lon: float | None = None,
) -> str | None:
    """Best-effort ISO alpha-2 country code for the claimed place.

    With coordinates: one Nominatim /reverse call (the only HTTP this
    function ever makes); if it comes back empty the claim tail is still
    tried. Without coordinates: the comma-tail of claimed_place is matched
    as an ISO code or common country name — "Paris, TX" yields None (TX is
    a state, not a country). Returns None when nothing resolves; never
    raises; at most 1 HTTP call.
    """
    try:
        if lat is not None and lon is not None:
            cc = reverse_geocode(lat, lon).get("country_code")
            if cc:
                return cc
        tail = (claimed_place or "").rsplit(",", 1)[-1].strip()
        if not tail:
            return None
        if len(tail) == 2 and tail.upper() in _ISO_CC:
            return tail.upper()
        return _COUNTRY_NAME_CC.get(tail.lower())
    except Exception:  # noqa: BLE001 — spec: never raise
        return None


if __name__ == "__main__":
    # Offline-safe demo: 51.5074,-0.1278 is London -> country_code "GB".
    # reverse_geocode never raises; {} just means no network — exit 0 anyway.
    try:
        hit = reverse_geocode(51.5074, -0.1278)
        if not hit:
            print("no Nominatim response (offline?) — demo skipped")
        else:
            ok = hit["country_code"] == "GB"
            print(f"{'OK' if ok else 'unexpected'}: {hit['display_name']}")
            print(
                f"  cc={hit['country_code']} country={hit['country']} "
                f"city={hit['city']} road={hit['road']}"
            )
            print("  anchor:", anchor_context(51.5074, -0.1278, "London"))
        # Tail-parse path needs no network either way.
        print(
            "  claim_country_cc('Berlin, Germany') =",
            claim_country_cc("Berlin, Germany"),
        )
    except Exception as e:  # noqa: BLE001 — belt & braces: exit 0 offline
        print(f"demo skipped: {e}")
