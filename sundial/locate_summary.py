"""Locate summary — distill a finished EvidencePack into the "where it
thinks it is" readout: one anchor, map markers, latitude/longitude bands,
a ranked country shortlist and a predicted point+radius.

The report's location panel and map layer consume ``summarize``'s dict
directly — every value is JSON-safe and every key is always present (None
or [] when a signal never materialized). ``panel_text`` renders the same
verdict-of-place as a one-line header string.

Inputs are read defensively: the pipeline's evidence cards evolve, so each
extractor tolerates missing keys, alternate spellings and half-filled
dicts — nothing here ever raises. Sources consulted:

  * osint / geo_match cards   data["ranked"] / ["candidates"] / ["ranking"]
  * geoguess cards            data["stable_cc"] (+ nested osint agg dicts)
  * landmark / wiki / poi     data["places"] / ["matches"] (lat/lon fields)
  * shadow_geo-style cards    any *band* keys -> lat/lon rectangles
  * exif_gps / metadata cards data["gps"] / data["meta"]["gps_lat"|"gps_lon"]
  * geo-resolve notes + the resolved (lat, lon) anchor passed in

Stdlib only at import time; schema is never needed (evidence is read via
getattr/.get so duck-typed dicts work too).

  python3 sundial/locate_summary.py     # self-test
"""
from __future__ import annotations

import math

# --------------------------------------------------------------------------
# ISO 3166-1 alpha-2 -> coarse country centroid + display name.
# 1-decimal centroids are deliberately coarse — the map only needs "which
# part of the world", and a false-precision centroid would mislead.
# --------------------------------------------------------------------------
CC_GEO: dict = {
    # --- Europe ---
    "AD": {"lat": 42.5, "lon": 1.6, "name": "Andorra"},
    "AL": {"lat": 41.2, "lon": 20.2, "name": "Albania"},
    "AT": {"lat": 47.5, "lon": 14.6, "name": "Austria"},
    "BA": {"lat": 44.2, "lon": 17.7, "name": "Bosnia and Herzegovina"},
    "BE": {"lat": 50.5, "lon": 4.5, "name": "Belgium"},
    "BG": {"lat": 42.7, "lon": 25.3, "name": "Bulgaria"},
    "BY": {"lat": 53.7, "lon": 28.0, "name": "Belarus"},
    "CH": {"lat": 46.8, "lon": 8.2, "name": "Switzerland"},
    "CY": {"lat": 35.1, "lon": 33.4, "name": "Cyprus"},
    "CZ": {"lat": 49.8, "lon": 15.5, "name": "Czechia"},
    "DE": {"lat": 51.2, "lon": 10.4, "name": "Germany"},
    "DK": {"lat": 56.0, "lon": 9.5, "name": "Denmark"},
    "EE": {"lat": 58.7, "lon": 25.0, "name": "Estonia"},
    "ES": {"lat": 40.3, "lon": -3.7, "name": "Spain"},
    "FI": {"lat": 64.0, "lon": 26.0, "name": "Finland"},
    "FR": {"lat": 46.2, "lon": 2.2, "name": "France"},
    "GB": {"lat": 54.0, "lon": -2.5, "name": "United Kingdom"},
    "GR": {"lat": 39.1, "lon": 22.0, "name": "Greece"},
    "HR": {"lat": 45.2, "lon": 15.5, "name": "Croatia"},
    "HU": {"lat": 47.2, "lon": 19.5, "name": "Hungary"},
    "IE": {"lat": 53.1, "lon": -8.2, "name": "Ireland"},
    "IS": {"lat": 65.0, "lon": -19.0, "name": "Iceland"},
    "IT": {"lat": 41.9, "lon": 12.6, "name": "Italy"},
    "LI": {"lat": 47.2, "lon": 9.6, "name": "Liechtenstein"},
    "LT": {"lat": 55.2, "lon": 23.9, "name": "Lithuania"},
    "LU": {"lat": 49.6, "lon": 6.1, "name": "Luxembourg"},
    "LV": {"lat": 56.9, "lon": 24.6, "name": "Latvia"},
    "MC": {"lat": 43.7, "lon": 7.4, "name": "Monaco"},
    "MD": {"lat": 47.4, "lon": 28.4, "name": "Moldova"},
    "ME": {"lat": 42.7, "lon": 19.3, "name": "Montenegro"},
    "MK": {"lat": 41.6, "lon": 21.7, "name": "North Macedonia"},
    "MT": {"lat": 35.9, "lon": 14.5, "name": "Malta"},
    "NL": {"lat": 52.2, "lon": 5.3, "name": "Netherlands"},
    "NO": {"lat": 60.5, "lon": 8.8, "name": "Norway"},
    "PL": {"lat": 51.9, "lon": 19.1, "name": "Poland"},
    "PT": {"lat": 39.4, "lon": -8.2, "name": "Portugal"},
    "RO": {"lat": 45.9, "lon": 25.0, "name": "Romania"},
    "RS": {"lat": 44.0, "lon": 21.0, "name": "Serbia"},
    "RU": {"lat": 60.0, "lon": 90.0, "name": "Russia"},
    "SE": {"lat": 62.0, "lon": 15.0, "name": "Sweden"},
    "SI": {"lat": 46.1, "lon": 14.8, "name": "Slovenia"},
    "SK": {"lat": 48.7, "lon": 19.5, "name": "Slovakia"},
    "SM": {"lat": 43.9, "lon": 12.5, "name": "San Marino"},
    "UA": {"lat": 48.4, "lon": 31.2, "name": "Ukraine"},
    "VA": {"lat": 41.9, "lon": 12.5, "name": "Vatican City"},
    "XK": {"lat": 42.6, "lon": 21.0, "name": "Kosovo"},
    # --- Americas ---
    "AR": {"lat": -34.0, "lon": -64.0, "name": "Argentina"},
    "BB": {"lat": 13.2, "lon": -59.5, "name": "Barbados"},
    "BO": {"lat": -17.0, "lon": -65.0, "name": "Bolivia"},
    "BR": {"lat": -10.8, "lon": -52.9, "name": "Brazil"},
    "BS": {"lat": 24.3, "lon": -76.5, "name": "Bahamas"},
    "BZ": {"lat": 17.2, "lon": -88.7, "name": "Belize"},
    "CA": {"lat": 56.1, "lon": -106.3, "name": "Canada"},
    "CL": {"lat": -33.4, "lon": -70.7, "name": "Chile"},
    "CO": {"lat": 4.6, "lon": -74.1, "name": "Colombia"},
    "CR": {"lat": 9.7, "lon": -84.0, "name": "Costa Rica"},
    "CU": {"lat": 21.5, "lon": -79.5, "name": "Cuba"},
    "DO": {"lat": 19.0, "lon": -70.2, "name": "Dominican Republic"},
    "EC": {"lat": -1.8, "lon": -78.2, "name": "Ecuador"},
    "FK": {"lat": -51.7, "lon": -59.5, "name": "Falkland Islands"},
    "GF": {"lat": 4.0, "lon": -53.0, "name": "French Guiana"},
    "GL": {"lat": 72.0, "lon": -40.0, "name": "Greenland"},
    "GT": {"lat": 15.8, "lon": -90.2, "name": "Guatemala"},
    "GY": {"lat": 4.9, "lon": -58.9, "name": "Guyana"},
    "HN": {"lat": 15.2, "lon": -86.2, "name": "Honduras"},
    "HT": {"lat": 19.0, "lon": -72.7, "name": "Haiti"},
    "JM": {"lat": 18.1, "lon": -77.3, "name": "Jamaica"},
    "MX": {"lat": 23.6, "lon": -102.5, "name": "Mexico"},
    "NI": {"lat": 12.9, "lon": -85.0, "name": "Nicaragua"},
    "PA": {"lat": 8.5, "lon": -80.8, "name": "Panama"},
    "PE": {"lat": -9.2, "lon": -75.0, "name": "Peru"},
    "PR": {"lat": 18.2, "lon": -66.5, "name": "Puerto Rico"},
    "PY": {"lat": -23.4, "lon": -58.4, "name": "Paraguay"},
    "SR": {"lat": 4.1, "lon": -55.9, "name": "Suriname"},
    "SV": {"lat": 13.8, "lon": -88.9, "name": "El Salvador"},
    "TT": {"lat": 10.4, "lon": -61.3, "name": "Trinidad and Tobago"},
    "US": {"lat": 39.8, "lon": -98.6, "name": "United States"},
    "UY": {"lat": -32.5, "lon": -55.8, "name": "Uruguay"},
    "VE": {"lat": 7.0, "lon": -66.0, "name": "Venezuela"},
    # --- Asia / Middle East ---
    "AE": {"lat": 24.0, "lon": 54.0, "name": "United Arab Emirates"},
    "AF": {"lat": 34.5, "lon": 69.2, "name": "Afghanistan"},
    "AM": {"lat": 40.1, "lon": 45.0, "name": "Armenia"},
    "AZ": {"lat": 40.1, "lon": 47.6, "name": "Azerbaijan"},
    "BD": {"lat": 23.7, "lon": 90.4, "name": "Bangladesh"},
    "BH": {"lat": 26.1, "lon": 50.6, "name": "Bahrain"},
    "BN": {"lat": 4.5, "lon": 114.7, "name": "Brunei"},
    "BT": {"lat": 27.5, "lon": 90.4, "name": "Bhutan"},
    "CN": {"lat": 35.0, "lon": 105.0, "name": "China"},
    "GE": {"lat": 42.3, "lon": 43.4, "name": "Georgia"},
    "HK": {"lat": 22.3, "lon": 114.2, "name": "Hong Kong"},
    "ID": {"lat": -2.5, "lon": 118.0, "name": "Indonesia"},
    "IL": {"lat": 31.0, "lon": 34.9, "name": "Israel"},
    "IN": {"lat": 21.0, "lon": 78.0, "name": "India"},
    "IQ": {"lat": 33.2, "lon": 43.7, "name": "Iraq"},
    "IR": {"lat": 32.4, "lon": 53.7, "name": "Iran"},
    "JO": {"lat": 31.2, "lon": 36.2, "name": "Jordan"},
    "JP": {"lat": 36.2, "lon": 138.3, "name": "Japan"},
    "KG": {"lat": 41.2, "lon": 74.8, "name": "Kyrgyzstan"},
    "KH": {"lat": 12.6, "lon": 105.0, "name": "Cambodia"},
    "KP": {"lat": 40.3, "lon": 127.5, "name": "North Korea"},
    "KR": {"lat": 36.4, "lon": 127.9, "name": "South Korea"},
    "KW": {"lat": 29.3, "lon": 47.5, "name": "Kuwait"},
    "KZ": {"lat": 48.0, "lon": 67.0, "name": "Kazakhstan"},
    "LA": {"lat": 19.9, "lon": 102.5, "name": "Laos"},
    "LB": {"lat": 33.9, "lon": 35.9, "name": "Lebanon"},
    "LK": {"lat": 7.9, "lon": 80.8, "name": "Sri Lanka"},
    "MM": {"lat": 21.9, "lon": 96.0, "name": "Myanmar"},
    "MN": {"lat": 46.9, "lon": 103.8, "name": "Mongolia"},
    "MO": {"lat": 22.2, "lon": 113.5, "name": "Macau"},
    "MV": {"lat": 3.2, "lon": 73.2, "name": "Maldives"},
    "MY": {"lat": 4.2, "lon": 102.0, "name": "Malaysia"},
    "NP": {"lat": 28.4, "lon": 84.1, "name": "Nepal"},
    "OM": {"lat": 20.6, "lon": 56.1, "name": "Oman"},
    "PH": {"lat": 12.9, "lon": 121.8, "name": "Philippines"},
    "PK": {"lat": 30.4, "lon": 69.3, "name": "Pakistan"},
    "PS": {"lat": 31.9, "lon": 35.2, "name": "Palestine"},
    "QA": {"lat": 25.3, "lon": 51.2, "name": "Qatar"},
    "SA": {"lat": 24.0, "lon": 45.0, "name": "Saudi Arabia"},
    "SG": {"lat": 1.4, "lon": 103.8, "name": "Singapore"},
    "SY": {"lat": 35.0, "lon": 38.0, "name": "Syria"},
    "TH": {"lat": 15.9, "lon": 101.0, "name": "Thailand"},
    "TJ": {"lat": 38.9, "lon": 71.0, "name": "Tajikistan"},
    "TL": {"lat": -8.8, "lon": 125.7, "name": "Timor-Leste"},
    "TM": {"lat": 39.1, "lon": 59.4, "name": "Turkmenistan"},
    "TR": {"lat": 38.9, "lon": 35.2, "name": "Turkey"},
    "TW": {"lat": 23.6, "lon": 121.0, "name": "Taiwan"},
    "UZ": {"lat": 41.4, "lon": 64.6, "name": "Uzbekistan"},
    "VN": {"lat": 14.1, "lon": 108.3, "name": "Vietnam"},
    "YE": {"lat": 15.6, "lon": 48.5, "name": "Yemen"},
    # --- Africa ---
    "AO": {"lat": -11.2, "lon": 17.9, "name": "Angola"},
    "BF": {"lat": 12.3, "lon": -1.7, "name": "Burkina Faso"},
    "BI": {"lat": -3.4, "lon": 29.9, "name": "Burundi"},
    "BJ": {"lat": 9.3, "lon": 2.3, "name": "Benin"},
    "BW": {"lat": -22.3, "lon": 24.7, "name": "Botswana"},
    "CD": {"lat": -2.9, "lon": 23.7, "name": "DR Congo"},
    "CF": {"lat": 6.6, "lon": 20.9, "name": "Central African Republic"},
    "CG": {"lat": -0.7, "lon": 15.6, "name": "Congo"},
    "CI": {"lat": 7.5, "lon": -5.5, "name": "Ivory Coast"},
    "CM": {"lat": 6.0, "lon": 12.7, "name": "Cameroon"},
    "CV": {"lat": 15.1, "lon": -23.6, "name": "Cape Verde"},
    "DJ": {"lat": 11.8, "lon": 42.6, "name": "Djibouti"},
    "DZ": {"lat": 28.0, "lon": 2.6, "name": "Algeria"},
    "EG": {"lat": 26.8, "lon": 30.8, "name": "Egypt"},
    "ER": {"lat": 15.2, "lon": 39.8, "name": "Eritrea"},
    "ET": {"lat": 9.1, "lon": 40.5, "name": "Ethiopia"},
    "GA": {"lat": -0.8, "lon": 11.6, "name": "Gabon"},
    "GH": {"lat": 7.9, "lon": -1.0, "name": "Ghana"},
    "GM": {"lat": 13.4, "lon": -15.4, "name": "Gambia"},
    "GN": {"lat": 9.9, "lon": -11.0, "name": "Guinea"},
    "GQ": {"lat": 1.6, "lon": 10.4, "name": "Equatorial Guinea"},
    "GW": {"lat": 12.0, "lon": -15.0, "name": "Guinea-Bissau"},
    "KE": {"lat": 0.2, "lon": 37.9, "name": "Kenya"},
    "KM": {"lat": -11.9, "lon": 43.9, "name": "Comoros"},
    "LR": {"lat": 6.4, "lon": -9.4, "name": "Liberia"},
    "LS": {"lat": -29.6, "lon": 28.2, "name": "Lesotho"},
    "LY": {"lat": 27.0, "lon": 17.2, "name": "Libya"},
    "MA": {"lat": 31.8, "lon": -7.1, "name": "Morocco"},
    "MG": {"lat": -19.4, "lon": 46.7, "name": "Madagascar"},
    "ML": {"lat": 17.6, "lon": -4.0, "name": "Mali"},
    "MR": {"lat": 20.3, "lon": -10.3, "name": "Mauritania"},
    "MU": {"lat": -20.3, "lon": 57.6, "name": "Mauritius"},
    "MW": {"lat": -13.3, "lon": 34.3, "name": "Malawi"},
    "MZ": {"lat": -18.7, "lon": 35.5, "name": "Mozambique"},
    "NA": {"lat": -22.9, "lon": 18.5, "name": "Namibia"},
    "NE": {"lat": 17.6, "lon": 8.1, "name": "Niger"},
    "NG": {"lat": 9.1, "lon": 8.7, "name": "Nigeria"},
    "RW": {"lat": -2.0, "lon": 29.9, "name": "Rwanda"},
    "SC": {"lat": -4.7, "lon": 55.5, "name": "Seychelles"},
    "SD": {"lat": 12.9, "lon": 30.2, "name": "Sudan"},
    "SL": {"lat": 8.5, "lon": -11.8, "name": "Sierra Leone"},
    "SN": {"lat": 14.5, "lon": -14.4, "name": "Senegal"},
    "SO": {"lat": 5.2, "lon": 46.2, "name": "Somalia"},
    "SS": {"lat": 7.9, "lon": 30.0, "name": "South Sudan"},
    "ST": {"lat": 0.3, "lon": 6.6, "name": "Sao Tome and Principe"},
    "SZ": {"lat": -26.5, "lon": 31.5, "name": "Eswatini"},
    "TD": {"lat": 15.5, "lon": 18.7, "name": "Chad"},
    "TG": {"lat": 8.6, "lon": 1.2, "name": "Togo"},
    "TN": {"lat": 34.0, "lon": 9.6, "name": "Tunisia"},
    "TZ": {"lat": -6.4, "lon": 34.9, "name": "Tanzania"},
    "UG": {"lat": 1.4, "lon": 32.4, "name": "Uganda"},
    "ZA": {"lat": -29.0, "lon": 25.1, "name": "South Africa"},
    "ZM": {"lat": -13.1, "lon": 27.8, "name": "Zambia"},
    "ZW": {"lat": -19.0, "lon": 29.9, "name": "Zimbabwe"},
    # --- Oceania ---
    "AU": {"lat": -25.3, "lon": 133.8, "name": "Australia"},
    "FJ": {"lat": -17.7, "lon": 178.1, "name": "Fiji"},
    "FM": {"lat": 6.9, "lon": 158.2, "name": "Micronesia"},
    "GU": {"lat": 13.4, "lon": 144.8, "name": "Guam"},
    "KI": {"lat": 1.9, "lon": 173.0, "name": "Kiribati"},
    "MH": {"lat": 7.1, "lon": 171.2, "name": "Marshall Islands"},
    "NC": {"lat": -21.3, "lon": 165.5, "name": "New Caledonia"},
    "NR": {"lat": -0.5, "lon": 166.9, "name": "Nauru"},
    "NZ": {"lat": -41.8, "lon": 172.8, "name": "New Zealand"},
    "PF": {"lat": -17.7, "lon": -149.4, "name": "French Polynesia"},
    "PG": {"lat": -6.3, "lon": 145.0, "name": "Papua New Guinea"},
    "PW": {"lat": 7.5, "lon": 134.6, "name": "Palau"},
    "SB": {"lat": -9.6, "lon": 160.2, "name": "Solomon Islands"},
    "TO": {"lat": -21.2, "lon": -175.2, "name": "Tonga"},
    "TV": {"lat": -7.1, "lon": 177.6, "name": "Tuvalu"},
    "VU": {"lat": -16.0, "lon": 167.0, "name": "Vanuatu"},
    "WS": {"lat": -13.8, "lon": -172.1, "name": "Samoa"},
}

# Display-name -> cc index, built once (first build wins on collisions —
# e.g. "Congo" means CG while the DRC is stored as "DR Congo").
_NAME_TO_CC: dict = {}


def _name_cc_index() -> dict:
    if not _NAME_TO_CC:
        for cc, row in CC_GEO.items():
            _NAME_TO_CC.setdefault(str(row.get("name") or "").lower(), cc)
    return _NAME_TO_CC


# Marker/band caps — the map stays readable even when a stage dumps 50 POIs.
_MAX_MARKERS = 14
_MAX_BANDS = 8
# Band palette cycles when a card doesn't name its own color.
_BAND_COLORS = ("#facc15", "#38bdf8", "#c084fc", "#4ade80")


# --------------------------------------------------------------------------
# Defensive primitives
# --------------------------------------------------------------------------

def _num(v) -> float | None:
    """float() that swallows everything and rejects non-finite values."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _latlon(lat, lon):
    """-> (lat, lon) floats in-range, or None."""
    la, lo = _num(lat), _num(lon)
    if la is None or lo is None:
        return None
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return None
    return la, lo


def _norm_cc(v) -> str | None:
    """Anything cc-ish -> alpha-2: 'de', 'DE', 'Germany', {'cc': 'de'}."""
    if isinstance(v, dict):
        for k in ("cc", "country_code", "code", "iso", "iso_cc", "country"):
            got = _norm_cc(v.get(k))
            if got:
                return got
        return None
    s = str(v or "").strip()
    if not s:
        return None
    u = s.upper()
    if len(u) == 2 and u.isalpha():
        return u
    return _name_cc_index().get(s.lower())


def _cc_name(cc) -> str | None:
    row = CC_GEO.get(_norm_cc(cc) or "")
    return row.get("name") if isinstance(row, dict) else None


def _cc_point(cc):
    row = CC_GEO.get(_norm_cc(cc) or "")
    if not isinstance(row, dict):
        return None
    return _latlon(row.get("lat"), row.get("lon"))


def _cc_of_point(lat, lon) -> "str | None":
    """alpha-2 of the country containing (lat, lon) via offline NE borders
    — lazy pip2cc; None over ocean/bad input/missing data."""
    try:
        from . import pip2cc
    except Exception:
        try:
            import pip2cc  # direct-script context
        except Exception:
            return None
    try:
        hit = pip2cc.country_at(lat, lon)
        return _norm_cc(hit.get("cc")) if isinstance(hit, dict) else None
    except Exception:
        return None


def _ev_parts(ev) -> tuple[str, dict]:
    """Evidence object OR plain dict -> (type_str, data_dict)."""
    if isinstance(ev, dict):
        t = ev.get("type") or ev.get("stage") or ev.get("name") or ""
        d = ev.get("data")
    else:
        t = getattr(ev, "type", "") or ""
        d = getattr(ev, "data", None)
    return str(t), (d if isinstance(d, dict) else {})


def _iter_evidence(evidence_list):
    """Yield (type, data) for each card; tolerates an EvidencePack (uses its
    .evidence), dicts, dataclasses, Nones and non-dict data payloads."""
    items = evidence_list
    if items is None:
        return
    if not isinstance(items, (list, tuple)):
        items = getattr(items, "evidence", None) or [items]
    for ev in items:
        try:
            yield _ev_parts(ev)
        except Exception:
            continue


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _anchor_pt(anchor):
    """Anchor in any caller-held form -> (lat, lon) or None.

    Tolerates (lat, lon) tuples/lists, {"lat","lon"} dicts, dataclass-ish
    objects with .lat/.lon (geo.GeoCandidate shape)."""
    if anchor is None:
        return None
    try:
        if isinstance(anchor, dict):
            return _latlon(anchor.get("lat"), anchor.get("lon"))
        if isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
            return _latlon(anchor[0], anchor[1])
        return _latlon(getattr(anchor, "lat", None),
                       getattr(anchor, "lon", None))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Ranked country rows -> [{"cc","name","score","hits_n"}]
# --------------------------------------------------------------------------

def _norm_rank_row(r) -> dict | None:
    """One ranked entry -> {'cc','score','hits_n'} tolerating the shape
    drift match_score/osint have had ({cc|country_code|country|code,
    score|confidence|weight, hits|signals|reasons}, tuples, {cc: score})."""
    cc = score = hits = None
    if isinstance(r, dict):
        cc = _norm_cc(r)
        for k in ("score", "confidence", "weight", "rank_score", "prob"):
            if _num(r.get(k)) is not None:
                score = _num(r.get(k))
                break
        h = r.get("hits", r.get("n_hits", r.get("signals", r.get("reasons",
                  r.get("agreeing_modalities")))))
        hits = len(h) if isinstance(h, (list, tuple, set, dict)) else _num(h)
    elif isinstance(r, (list, tuple)) and r:
        cc = _norm_cc(r[0])
        score = _num(r[1]) if len(r) > 1 else None
        hits = _num(r[2]) if len(r) > 2 else None
    if not cc:
        return None
    return {"cc": cc, "score": float(score or 0.0),
            "hits_n": int(hits or 0)}


def _ranked_rows(evs) -> list:
    """First non-empty ranked list across the pack, normalized and
    score-sorted. Prefers osint-flavoured cards but accepts any card whose
    data carries a ranking under a known key. Skips flat 0-cue priors."""
    found = []
    for t, d in evs:
        for key in ("ranked", "candidates", "ranking", "top3"):
            raw = d.get(key)
            if isinstance(raw, dict):  # {cc: score} map form
                raw = [{"cc": k, "score": v} for k, v in raw.items()]
            if not isinstance(raw, (list, tuple)) or not raw:
                continue
            rows = [r for r in (_norm_rank_row(x) for x in raw) if r]
            if not rows:
                continue
            # Ignore a uniform 0.5 prior where zero cues fired (match_score
            # emits every country at 0.50 with hits=[] when nothing matched).
            if (len(rows) >= 3
                    and all(r["hits_n"] == 0 and abs(r["score"] - 0.5) < 1e-6
                            for r in rows)):
                continue
            pri = (0 if "matrix" in t or "osint" in t or "geo" in t else 1,
                   0 if key in ("ranked", "ranking", "top3") else 1)
            found.append((pri, -len(rows), rows))
    if not found:
        return []
    found.sort(key=lambda f: (f[0], f[1]))
    rows = found[0][2]
    # dedupe by cc keeping the stronger row, then sort score-desc
    out: list = []
    for r in rows:
        for i, e in enumerate(out):
            if e["cc"] == r["cc"]:
                if r["score"] > e["score"]:
                    out[i] = r
                break
        else:
            out.append(r)
    out.sort(key=lambda e: -e["score"])
    for r in out:
        r["name"] = _cc_name(r["cc"]) or r["cc"]
    return out


# --------------------------------------------------------------------------
# Map markers — claim / pred / geo / wiki
# --------------------------------------------------------------------------

def _marker(lat, lon, label, kind) -> dict | None:
    pt = _latlon(lat, lon)
    if not pt:
        return None
    return {"lat": round(pt[0], 5), "lon": round(pt[1], 5),
            "label": str(label or kind), "kind": str(kind)}


def _items_latlon(item):
    """Yield (lat, lon, label) from one places/matches row.

    Flat rows ({lat, lon, name|title}) and nested overpass rows
    ({"poi": {"lat","lon","name"}}) are both read."""
    if not isinstance(item, dict):
        return
    pt = _latlon(item.get("lat"), item.get("lon"))
    if pt:
        label = (item.get("name") or item.get("title")
                 or item.get("display_name") or item.get("label")
                 or item.get("poi_name") or item.get("query"))
        yield pt[0], pt[1], label
    for nest in ("poi", "place", "location", "geo", "hit", "match"):
        sub = item.get(nest)
        if isinstance(sub, dict):
            pt = _latlon(sub.get("lat"), sub.get("lon"))
            if pt:
                label = (sub.get("name") or sub.get("title")
                         or item.get("name") or item.get("title")
                         or item.get("query"))
                yield pt[0], pt[1], label


def _stable_ccs(evs) -> list:
    """geoguess consensus countries — data["stable_cc"], or nested
    data["geoguess"]["stable_cc"|"stable_country"]."""
    out: list = []
    for t, d in evs:
        cands = [d.get("stable_cc"), d.get("stable_country")]
        agg = d.get("geoguess")
        if isinstance(agg, dict):
            cands += [agg.get("stable_cc"), agg.get("stable_country")]
        for c in cands:
            cc = _norm_cc(c)
            if cc and cc not in out:
                out.append(cc)
    return out


def _best_named_place(evs) -> "dict | None":
    """The strongest *named* coordinate place across the pack.

    A consensus landmark the VLM identified (Buckingham Palace) geocodes
    to a precise point — a far better estimate than a country-centroid
    blend. Landmark/wiki cards win; generic geo fixes are ignored (they
    already have their own marker). Returns {lat, lon, name, cc} or None.
    """
    best = None
    for t, d in evs:
        tl = t.lower()
        if "landmark" not in tl and "wiki" not in tl:
            continue
        for key in ("places", "matches", "hits"):
            for item in d.get(key) or []:
                if not isinstance(item, dict):
                    continue
                pt = _latlon(item.get("lat"), item.get("lon"))
                if not pt:
                    continue
                name = (item.get("name") or item.get("title")
                        or item.get("label") or "").strip()
                if not name:
                    continue
                try:
                    score = float(item.get("score") or 0)
                except (TypeError, ValueError):
                    score = 0.0
                row = {"lat": pt[0], "lon": pt[1], "name": name,
                       "cc": str(item.get("cc") or item.get("country_code")
                                 or "").upper() or None,
                       "score": score}
                # landmark cards outrank wiki; then geocode score order
                rank = (1 if "landmark" in tl else 0, score)
                if best is None or rank > best[0]:
                    best = (rank, row)
    return best[1] if best else None


def _geo_markers(evs) -> list:
    """Coordinate-bearing rows out of landmark/wiki/poi/exif/geo-resolve
    cards -> marker dicts (deduped, capped)."""
    markers: list = []
    seen: set = set()

    def add(lat, lon, label, kind):
        if len(markers) >= _MAX_MARKERS:
            return
        m = _marker(lat, lon, label, kind)
        if not m:
            return
        key = (m["lat"], m["lon"], m["kind"])
        if key in seen:
            return
        seen.add(key)
        markers.append(m)

    for t, d in evs:
        tl = t.lower()
        if tl in ("geo", "geo-resolve", "claim_resolver", "osm_context",
                  "geo_pred", "geocell_pred", "geo_ensemble"):
            # Claim-anchor lookup cards just echo the claimed coordinates;
            # region-estimate cards already feed the prediction pin.
            continue
        if "geoclip" in tl or tl == "geo_clip":  # model card: the best
            # guess is already the pred pin — show the runner-ups instead
            for c in (d.get("candidates") or [])[1:]:
                if not isinstance(c, dict):
                    continue
                pt = _latlon(c.get("lat"), c.get("lon"))
                if not pt:
                    continue
                try:
                    pct = f" ({float(c.get('prob') or 0) * 100:.0f}%)"
                except (TypeError, ValueError):
                    pct = ""
                add(pt[0], pt[1], f"other possible spot{pct}", "geo")
            continue
        wiki_kind = "wiki" if "wiki" in tl else "geo"
        for key in ("places", "matches", "hits", "pois", "results"):
            for item in d.get(key) or []:
                if not isinstance(item, (dict, list, tuple)):
                    continue
                if isinstance(item, dict):
                    for lat, lon, label in _items_latlon(item):
                        add(lat, lon, label, wiki_kind)
                elif len(item) >= 2:  # bare (lat, lon[, label]) row
                    pt = _latlon(item[0], item[1])
                    if pt:
                        add(pt[0], pt[1],
                            item[2] if len(item) > 2 else key, wiki_kind)
        # EXIF-style GPS fixes: {"gps":{lat,lon}}, {"meta":{gps_lat,gps_lon}}
        # or a flat data{"lat","lon"} (geo-resolve cards). Known container
        # keys are checked first, then every dict value — an "anchor" dict
        # is skipped since the claim marker already covers it.
        srcs = [d.get(k) for k in ("gps", "meta", "exif", "location", "fix",
                                   "position", "geotag", "coords",
                                   "coordinates")]
        srcs += [v for k, v in d.items()
                 if isinstance(v, dict) and str(k).lower() != "anchor"]
        srcs.append(d)
        for src in srcs:
            if not isinstance(src, dict):
                continue
            la = src.get("lat") if src.get("lat") is not None else src.get("gps_lat")
            lo = src.get("lon") if src.get("lon") is not None else src.get("gps_lon")
            pt = _latlon(la, lo)
            if pt:
                add(pt[0], pt[1],
                    src.get("name") or src.get("label") or "gps fix", "geo")
    return markers


# --------------------------------------------------------------------------
# Latitude/longitude bands — shadow_geo-style rectangle constraints
# --------------------------------------------------------------------------

def _to_band(item, color) -> dict | None:
    """One band-ish thing -> {lat_min,lat_max,lon_min,lon_max,color}.

    Accepts dicts ({lat_min/lat_max|min/max|lat|lat_band, lon_*}) and
    tuples: (lo,hi) = lat band spanning all longitudes,
    (lat_min,lat_max,lon_min,lon_max) = full rectangle."""
    if isinstance(item, dict):
        lat = item.get("lat") or item.get("lat_band") or item.get("lat_range")
        if isinstance(lat, (list, tuple)) and len(lat) >= 2:
            la_lo, la_hi = _num(lat[0]), _num(lat[1])
        else:
            la_lo = _num(item.get("lat_min", item.get("min_lat",
                         item.get("lat_lo", item.get("min")))))
            la_hi = _num(item.get("lat_max", item.get("max_lat",
                         item.get("lat_hi", item.get("max")))))
        lon = item.get("lon") or item.get("lon_band") or item.get("lon_range")
        if isinstance(lon, (list, tuple)) and len(lon) >= 2:
            lo_lo, lo_hi = _num(lon[0]), _num(lon[1])
        else:
            lo_lo = _num(item.get("lon_min", item.get("min_lon",
                         item.get("lon_lo"))))
            lo_hi = _num(item.get("lon_max", item.get("max_lon",
                         item.get("lon_hi"))))
        c = item.get("color") or item.get("colour") or color
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        la_lo, la_hi = _num(item[0]), _num(item[1])
        lo_lo = _num(item[2]) if len(item) > 2 else None
        lo_hi = _num(item[3]) if len(item) > 3 else None
        c = color
    else:
        return None
    if la_lo is None or la_hi is None:
        return None
    if la_lo > la_hi:
        la_lo, la_hi = la_hi, la_lo
    if not (-90.0 <= la_lo <= 90.0 and -90.0 <= la_hi <= 90.0):
        return None
    if lo_lo is None or lo_hi is None:
        lo_lo, lo_hi = -180.0, 180.0  # lat-only band = full-width stripe
    if lo_lo > lo_hi:
        lo_lo, lo_hi = lo_hi, lo_lo
    return {"lat_min": round(float(la_lo), 3), "lat_max": round(float(la_hi), 3),
            "lon_min": round(float(lo_lo), 3), "lon_max": round(float(lo_hi), 3),
            "color": str(c or _BAND_COLORS[0])}


def _bands(evs) -> list:
    """Scan every card's data for *band*-ish keys + band-shaped rows."""
    out: list = []
    for t, d in evs:
        color_i = len(out)
        for key, val in d.items():
            if len(out) >= _MAX_BANDS:
                return out
            k = str(key).lower()
            if "band" in k and isinstance(val, (list, tuple)):
                for item in val:
                    b = _to_band(item, _BAND_COLORS[color_i % len(_BAND_COLORS)])
                    if b:
                        out.append(b)
            elif "band" in k and isinstance(val, dict):
                b = _to_band(val, _BAND_COLORS[color_i % len(_BAND_COLORS)])
                if b:
                    out.append(b)
        # The whole data dict itself may be one band ({lat_min:..,..}).
        if len(out) < _MAX_BANDS:
            b = _to_band(d, _BAND_COLORS[color_i % len(_BAND_COLORS)])
            if b and not any(
                    x["lat_min"] == b["lat_min"] and x["lat_max"] == b["lat_max"]
                    and x["lon_min"] == b["lon_min"] and x["lon_max"] == b["lon_max"]
                    for x in out):
                out.append(b)
    return out


# --------------------------------------------------------------------------
# Predicted point — score-weighted centroid of the top-3 candidates
# --------------------------------------------------------------------------

def _model_pred(evs) -> "dict | None":
    """A trained image->GPS model card (type carries lat/lon/r_km in data).

    GeoCLIP predictions are real coordinates — far better than a
    country-centroid blend when no landmark was named. First card wins."""
    for t, d in evs:
        if not isinstance(d, dict):
            continue
        tl = str(t).lower()
        if "geoclip" not in tl and tl not in ("geo_clip", "geo_ensemble"):
            continue
        pt = _latlon(d.get("lat"), d.get("lon"))
        if not pt:
            continue
        try:
            r = float(d.get("r_km") or d.get("radius_km") or 200.0)
        except (TypeError, ValueError):
            r = 200.0
        out = {"lat": pt[0], "lon": pt[1],
               "radius_km": int(max(50.0, min(2000.0, r)))}
        if isinstance(d.get("label"), str) and d["label"].strip():
            out["label"] = d["label"].strip()
        if isinstance(d.get("cc"), str) and d["cc"].strip():
            out["cc"] = d["cc"].strip().upper()
        return out
    return None


def _predict(rows: list) -> dict | None:
    """Weighted centroid of the top-3 ranked countries + radius_km from the
    score spread: a dominant #1 tightens the circle toward country scale,
    an even split balloons it toward continent scale."""
    pts = []
    for r in (rows or [])[:3]:
        pt = _cc_point(r.get("cc"))
        if pt:
            pts.append((pt[0], pt[1], max(float(r.get("score") or 0.0), 0.01)))
    if not pts:
        return None
    wsum = sum(w for _, _, w in pts) or 1.0
    lat = sum(p[0] * p[2] for p in pts) / wsum
    lon = sum(p[1] * p[2] for p in pts) / wsum
    dispersion = sum(
        _haversine_km(lat, lon, p[0], p[1]) * p[2] for p in pts) / wsum
    concentration = pts[0][2] / wsum  # 1.0 = single runaway winner
    radius = 300.0 + dispersion + 1200.0 * (1.0 - concentration)
    return {"lat": round(lat, 3), "lon": round(lon, 3),
            "radius_km": int(max(300.0, min(3200.0, round(radius))))}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def summarize(evidence_list: list, anchor: tuple | None,
              claim_place: str | None) -> dict:
    """Fold a finished EvidencePack's evidence list into map/panel inputs.

    Always returns the full key set; missing signals degrade to None/[].
    Never raises — a malformed card simply contributes nothing."""
    summary = {"anchor": None, "markers": [], "bands": [], "ranked": [],
               "best_cc": None, "best_name": None, "predicted": None}
    try:
        evs = list(_iter_evidence(evidence_list))
    except Exception:
        evs = []
    try:
        claim_label = str(claim_place).strip() if claim_place else "claimed location"
    except Exception:
        claim_label = "claimed location"

    rows: list = []

    try:
        pt = _anchor_pt(anchor)
        if pt:
            summary["anchor"] = {"lat": round(pt[0], 5),
                                 "lon": round(pt[1], 5),
                                 "label": claim_label}
            m = _marker(pt[0], pt[1], claim_label, "claim")
            if m:
                summary["markers"].append(m)
    except Exception:
        pass

    try:
        rows = list(_ranked_rows(evs))
        summary["ranked"] = [
            {"cc": r["cc"], "name": r["name"],
             "score": round(float(r["score"]), 3),
             "hits_n": int(r["hits_n"])}
            for r in rows[:4]
        ]
        if rows:
            summary["best_cc"] = rows[0]["cc"]
            summary["best_name"] = rows[0]["name"]
    except Exception:
        pass

    try:
        pred = _predict(rows)
        named = _best_named_place(evs)
        gclip = _model_pred(evs)
        # a ranking where no clue actually fired is just the claim echoed
        # back (or a flat prior) — it must never pin the map on its own
        has_cues = any(int(r.get("hits_n") or 0) > 0 for r in rows)
        corroborated = False
        if summary["anchor"] and rows and int(rows[0].get("hits_n") or 0) > 0:
            # cues corroborate the claimed country -> pin the claimed place
            # itself, not the country's geographic middle ("Buckingham
            # Palace is in London", not "somewhere in the middle of the UK")
            anch_cc = _cc_of_point(
                summary["anchor"]["lat"], summary["anchor"]["lon"])
            corroborated = bool(anch_cc and anch_cc == rows[0]["cc"])
        if not (named or corroborated or gclip or has_cues):
            pred = None
            summary["best_cc"] = summary["best_name"] = None
        if named:  # an identified landmark is a far better pin than a
            # country-centroid blend — 25 km ring says "in/around here"
            pred = {"lat": named["lat"], "lon": named["lon"],
                    "r_km": 25.0, "radius_km": 25}
            summary["best_name"] = (
                f"{named['name']}, {_cc_name(named['cc']) or named['cc']}"
                if named.get("cc") else named["name"])
            if named.get("cc"):
                summary["best_cc"] = named["cc"]
        elif corroborated:
            pred = {"lat": summary["anchor"]["lat"],
                    "lon": summary["anchor"]["lon"],
                    "r_km": 60.0, "radius_km": 60}
            summary["best_name"] = claim_label
        elif gclip:  # no landmark, claim not corroborated — the trained
            # model's own coordinate estimate beats the country blend;
            # in the no-claim flow it IS the answer
            pred = {"lat": gclip["lat"], "lon": gclip["lon"],
                    "r_km": float(gclip["radius_km"]),
                    "radius_km": gclip["radius_km"]}
            mcc = gclip.get("cc") or _cc_of_point(gclip["lat"], gclip["lon"])
            if mcc:
                summary["best_cc"] = mcc
                summary["best_name"] = gclip.get("label") or _cc_name(mcc) or mcc
            else:
                summary["best_name"] = gclip.get("label") or "model estimate"
        if pred:
            summary["predicted"] = pred
            label = (str(summary["best_name"])
                     if summary.get("best_name") else "likely location")
            m = _marker(pred["lat"], pred["lon"], label, "pred")
            if m:
                summary["markers"].append(m)
    except Exception:
        pass

    try:
        for m in _geo_markers(evs):
            if len(summary["markers"]) >= _MAX_MARKERS + 2:  # claim+pred headroom
                break
            summary["markers"].append(m)
    except Exception:
        pass

    try:
        for cc in _stable_ccs(evs):
            pt = _cc_point(cc)
            m = _marker(pt[0], pt[1],
                        f"model consensus: {_cc_name(cc) or cc}", "geo") if pt else None
            if m and all((x["lat"], x["lon"]) != (m["lat"], m["lon"])
                         for x in summary["markers"]):
                summary["markers"].append(m)
    except Exception:
        pass

    try:
        summary["bands"] = _bands(evs)
    except Exception:
        pass

    return summary


def panel_text(summary: dict) -> str:
    """One-line readout for the report header, e.g.
    "most likely Germany (DE) · next: FR, NL"."""
    try:
        s = summary or {}
        best_cc, best_name = s.get("best_cc"), s.get("best_name")
        if best_cc:
            head = f"most likely {best_name or best_cc} ({best_cc})"
            nxt = [str(r.get("cc")) for r in (s.get("ranked") or [])[1:3]
                   if isinstance(r, dict) and r.get("cc")]
            return head + (" · next: " + ", ".join(nxt) if nxt else "")
        pred = s.get("predicted")
        if isinstance(pred, dict) and _latlon(pred.get("lat"), pred.get("lon")):
            return (f"estimated near {float(pred['lat']):.1f},"
                    f"{float(pred['lon']):.1f} ±{pred.get('radius_km', '?')} km")
        anch = s.get("anchor")
        if isinstance(anch, dict):
            return f"claimed: {anch.get('label') or 'unresolved'}"
        return "no location estimate"
    except Exception:
        return "no location estimate"


# --------------------------------------------------------------------------
# Self-test: python3 sundial/locate_summary.py
# --------------------------------------------------------------------------

def _self_test() -> int:
    import json
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")

    # A pack-shaped fake evidence list — duck-typed dicts exercise the same
    # readers the real Evidence dataclasses would.
    fake = [
        {"type": "osint", "data": {
            "ranked": [
                {"cc": "DE", "score": 0.72, "hits": ["plate", "lang"]},
                {"cc": "FR", "score": 0.55, "hits": 1},
                {"cc": "NL", "score": 0.41},
                ("BE", 0.33, 2),
                {"country": "AT", "confidence": 0.2},
            ],
            "claimed_cc": "DE",
        }},
        {"type": "geoguess", "data": {"stable_cc": "DE",
                                     "stable_country": "germany"}},
        {"type": "landmark", "data": {"places": [
            {"name": "Brandenburg Gate", "lat": 52.5163, "lon": 13.3777},
            {"name": "Reichstag", "lat": 52.5186, "lon": 13.3762},
        ]}},
        {"type": "wiki_landmark", "data": {"matches": [
            {"title": "Brandenburg Gate", "lat": 52.5163, "lon": 13.3777,
             "dist_m": 120.0},
        ]}},
        {"type": "poi", "data": {"matches": [
            {"query": "Pret", "poi_name": "Pret A Manger",
             "poi": {"name": "Pret A Manger", "lat": 52.52, "lon": 13.4}},
        ]}},
        {"type": "shadow_geo", "data": {
            "bands": [{"lat_min": 47.0, "lat_max": 55.0},
                      (40.0, 50.0, -5.0, 10.0)],
            "lat_band": {"min": 45.0, "max": 60.0},
        }},
        {"type": "exif_gps", "data": {"gps": {"lat": 52.5, "lon": 13.4}}},
        {"type": "geo-resolve", "data": {"error": "ok"}},
    ]
    s = summarize(fake, (52.52, 13.405), "Berlin")
    check("anchor set", isinstance(s["anchor"], dict)
          and abs(s["anchor"]["lat"] - 52.52) < 1e-6, f"-> {s['anchor']}")
    kinds = {m["kind"] for m in s["markers"]}
    check("markers have all kinds", {"claim", "pred", "geo", "wiki"} <= kinds,
          f"-> {sorted(kinds)}")
    check("bands populated", len(s["bands"]) >= 2, f"-> {len(s['bands'])}")
    check("band shape", all(set(b) >= {"lat_min", "lat_max", "lon_min",
                                       "lon_max", "color"} for b in s["bands"]))
    check("ranked top-4", len(s["ranked"]) == 4
          and s["ranked"][0]["cc"] == "DE",
          f"-> {[r['cc'] for r in s['ranked']]}")
    check("ranked row shape", s["ranked"][0] == {
        "cc": "DE", "name": "Germany", "score": 0.72, "hits_n": 2})
    check("best_cc/name", s["best_cc"] == "DE" and s["best_name"] == "Brandenburg Gate")
    p = s["predicted"]
    check("predicted set", isinstance(p, dict)
          and -90 <= p["lat"] <= 90 and p["radius_km"] >= 25, f"-> {p}")
    txt = panel_text(s)
    check("panel text", txt.startswith("most likely Brandenburg Gate (DE)"),
          f"-> {txt!r}")
    check("next ccs in panel", "next:" in txt and "FR" in txt)
    check("json-safe", bool(json.dumps(s)))

    # Empty pack + anchor only.
    s2 = summarize([], (10.0, 20.0), "Somewhere")
    check("empty: anchor only", s2["anchor"]["label"] == "Somewhere"
          and s2["ranked"] == [] and s2["predicted"] is None
          and [m["kind"] for m in s2["markers"]] == ["claim"])
    check("empty: panel falls back", panel_text(s2).startswith("claimed:"))
    check("empty: json-safe", bool(json.dumps(s2)))

    # Nothing at all — must still return the full key set without raising.
    s3 = summarize(None, None, None)
    check("none: full keys", set(s3) == {"anchor", "markers", "bands",
                                         "ranked", "best_cc", "best_name",
                                         "predicted"})
    check("none: panel text", panel_text(s3) == "no location estimate")
    check("none: json-safe", bool(json.dumps(s3)))

    # Garbage cards must not crash or leak into the output.
    s4 = summarize([{"type": "osint", "data": {"ranked": [{"cc": "XX"},
                    "not-a-row", None, {"cc": "JP", "score": "bad"}]}},
                    {"type": "weird", "data": "not-a-dict"},
                    42, None],
                   anchor={"lat": "x"}, claim_place=123)
    check("garbage: no raise", isinstance(s4, dict))
    check("garbage: jp row survives", any(r["cc"] == "JP" for r in s4["ranked"]))
    check("garbage: json-safe", bool(json.dumps(s4)))

    print(f"locate_summary self-test: "
          f"{'all checks passed' if not fails else f'{fails} FAILED'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if _self_test() else 0)
