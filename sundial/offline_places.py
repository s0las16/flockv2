"""Offline nearest-city reverse lookup using committed Natural Earth places.

Zero-network, pure-stdlib reverse geocoder backed by the committed Natural
Earth 10m Populated Places dataset (7,300+ cities worldwide):
  - ``sundial/data/ne_10m_populated_places_simple.geojson``
  - ``sundial/data/ne_10m_populated_places.geojson``

On first call, lazily loads and caches city tuples sorted by latitude for
sub-millisecond ``bisect`` + haversine range queries. Every public function
is failure-tolerant and never raises on any input.

    python3 sundial/offline_places.py
"""
from __future__ import annotations

__all__ = [
    "nearest_city",
    "nearest_cities",
    "format_coord_human",
]

_EARTH_R_KM = 6371.0088
_DEG_LAT_KM = 110.574  # conservative lower bound km per degree latitude

_DATA_CANDIDATES = (
    "ne_10m_populated_places_simple.geojson",
    "ne_10m_populated_places.geojson",
)

# Natural Earth -99 ISO overrides keyed by ADM0_A3 / SOV_A3.
_ADM0_A3_CC = {
    "CYN": "CY",
    "SOL": "SO",
    "KOS": "XK",
    "FRA": "FR",
    "NOR": "NO",
}

# Normalize overly formal Natural Earth adm0name strings to standard short names.
_COUNTRY_NAME_NORM = {
    "United States of America": "United States",
    "Russian Federation": "Russia",
    "Republic of Korea": "South Korea",
    "Democratic People's Republic of Korea": "North Korea",
    "Iran (Islamic Republic of)": "Iran",
    "Syrian Arab Republic": "Syria",
    "Viet Nam": "Vietnam",
    "Lao People's Democratic Republic": "Laos",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "United Republic of Tanzania": "Tanzania",
    "Republic of Moldova": "Moldova",
}

# ISO alpha-2 -> English country name fallback when adm0name is absent.
_CC_COUNTRY = {
    "AD": "Andorra", "AE": "United Arab Emirates", "AF": "Afghanistan",
    "AL": "Albania", "AM": "Armenia", "AO": "Angola", "AR": "Argentina",
    "AT": "Austria", "AU": "Australia", "AZ": "Azerbaijan",
    "BA": "Bosnia and Herzegovina", "BD": "Bangladesh", "BE": "Belgium",
    "BG": "Bulgaria", "BH": "Bahrain", "BO": "Bolivia", "BR": "Brazil",
    "BY": "Belarus", "CA": "Canada", "CD": "Congo (Kinshasa)",
    "CF": "Central African Republic", "CG": "Congo (Brazzaville)",
    "CH": "Switzerland", "CI": "Ivory Coast", "CL": "Chile", "CM": "Cameroon",
    "CN": "China", "CO": "Colombia", "CR": "Costa Rica", "CU": "Cuba",
    "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "DO": "Dominican Republic", "DZ": "Algeria", "EC": "Ecuador",
    "EE": "Estonia", "EG": "Egypt", "ES": "Spain", "ET": "Ethiopia",
    "FI": "Finland", "FR": "France", "GB": "United Kingdom", "GE": "Georgia",
    "GH": "Ghana", "GR": "Greece", "GT": "Guatemala", "HK": "Hong Kong",
    "HN": "Honduras", "HR": "Croatia", "HU": "Hungary", "ID": "Indonesia",
    "IE": "Ireland", "IL": "Israel", "IN": "India", "IQ": "Iraq",
    "IR": "Iran", "IS": "Iceland", "IT": "Italy", "JM": "Jamaica",
    "JO": "Jordan", "JP": "Japan", "KE": "Kenya", "KG": "Kyrgyzstan",
    "KH": "Cambodia", "KP": "North Korea", "KR": "South Korea",
    "KW": "Kuwait", "KZ": "Kazakhstan", "LA": "Laos", "LB": "Lebanon",
    "LK": "Sri Lanka", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia",
    "LY": "Libya", "MA": "Morocco", "MC": "Monaco", "MD": "Moldova",
    "ME": "Montenegro", "MG": "Madagascar", "MK": "North Macedonia",
    "ML": "Mali", "MM": "Myanmar", "MN": "Mongolia", "MT": "Malta",
    "MX": "Mexico", "MY": "Malaysia", "MZ": "Mozambique", "NA": "Namibia",
    "NE": "Niger", "NG": "Nigeria", "NI": "Nicaragua", "NL": "Netherlands",
    "NO": "Norway", "NP": "Nepal", "NZ": "New Zealand", "OM": "Oman",
    "PA": "Panama", "PE": "Peru", "PH": "Philippines", "PK": "Pakistan",
    "PL": "Poland", "PR": "Puerto Rico", "PS": "Palestine", "PT": "Portugal",
    "PY": "Paraguay", "QA": "Qatar", "RO": "Romania", "RS": "Serbia",
    "RU": "Russia", "RW": "Rwanda", "SA": "Saudi Arabia", "SD": "Sudan",
    "SE": "Sweden", "SG": "Singapore", "SI": "Slovenia", "SK": "Slovakia",
    "SN": "Senegal", "SO": "Somalia", "SY": "Syria", "TH": "Thailand",
    "TJ": "Tajikistan", "TM": "Turkmenistan", "TN": "Tunisia", "TR": "Turkey",
    "TW": "Taiwan", "TZ": "Tanzania", "UA": "Ukraine", "UG": "Uganda",
    "US": "United States", "UY": "Uruguay", "UZ": "Uzbekistan",
    "VE": "Venezuela", "VN": "Vietnam", "XK": "Kosovo", "YE": "Yemen",
    "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
}

# Last-resort embedded cities if the GeoJSON data file is absent or corrupt.
# Schema: (lat, lon, name, cc, adm1name, pop_max, country)
_FALLBACK_RAW = (
    (50.720456, 7.080022, "Bonn", "DE", "Nordrhein-Westfalen", 680543, "Germany"),
    (50.931950, 6.948059, "Cologne", "DE", "Nordrhein-Westfalen", 1004000, "Germany"),
    (52.521819, 13.401549, "Berlin", "DE", "Berlin", 3406000, "Germany"),
    (48.858092, 2.352992, "Paris", "FR", "Île-de-France", 9904000, "France"),
    (51.500516, -0.116205, "London", "GB", "Greater London", 8567000, "United Kingdom"),
    (40.400026, -3.683352, "Madrid", "ES", "Comunidad de Madrid", 5567000, "Spain"),
    (41.895956, 12.483258, "Rome", "IT", "Lazio", 3339000, "Italy"),
    (35.686963, 139.749462, "Tokyo", "JP", "Tokyo", 35676000, "Japan"),
    (40.721562, -73.995718, "New York", "US", "New York", 19040000, "United States"),
    (-33.883056, 151.216667, "Sydney", "AU", "New South Wales", 4630000, "Australia"),
)

# Cached index: (cities_sorted_by_lat, lats_list)
# Each entry in cities_sorted_by_lat:
#   (lat, lon, name, cc, adm1name, pop_max, country, lat_rad, lon_rad, cos_lat)
_CACHE: tuple[list[tuple], list[float]] | None = None


def _valid_coord(lat: object, lon: object) -> tuple[float, float] | None:
    """Validate and coerce ``(lat, lon)`` -> ``(float, float)`` or ``None``."""
    import math

    if isinstance(lat, bool) or isinstance(lon, bool):
        return None
    try:
        lat_f = float(lat)  # type: ignore[arg-type]
        lon_f = float(lon)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(lat_f) or math.isnan(lon_f) or math.isinf(lat_f) or math.isinf(lon_f):
        return None
    if lat_f < -90.0 or lat_f > 90.0 or lon_f < -180.0 or lon_f > 180.0:
        return None
    return (lat_f, lon_f)


def _extract_cc(props: dict) -> str:
    """Extract a 2-letter ISO country code from Natural Earth feature properties."""
    for key in ("iso_a2", "adm0_a2", "ISO_A2", "ISO_A2_EH", "ADM0_A2"):
        val = str(props.get(key) or "").strip().upper()
        if len(val) == 2 and val.isalpha():
            return val
    for key in ("adm0_a3", "sov_a3", "ADM0_A3", "SOV_A3"):
        a3 = str(props.get(key) or "").strip().upper()
        if a3 in _ADM0_A3_CC:
            return _ADM0_A3_CC[a3]
    return ""


def _extract_country(props: dict, cc: str) -> str:
    """Resolve human-readable country name from feature properties or ISO code."""
    raw = str(props.get("adm0name") or props.get("sov0name") or props.get("ADMIN") or "").strip()
    if raw:
        return _COUNTRY_NAME_NORM.get(raw, raw)
    return _CC_COUNTRY.get(cc, "").strip()


def _build_entry(
    lat: float,
    lon: float,
    name: str,
    cc: str,
    adm1name: str,
    pop_max: int,
    country: str,
) -> tuple:
    import math

    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    return (
        lat,
        lon,
        name,
        cc,
        adm1name,
        pop_max,
        country,
        lat_r,
        lon_r,
        math.cos(lat_r),
    )


def _load() -> tuple[list[tuple], list[float]]:
    """Lazily load, sort, and cache Natural Earth populated places."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    import json
    from pathlib import Path

    entries: list[tuple] = []
    data_dir = Path(__file__).resolve().parent / "data"

    for fname in _DATA_CANDIDATES:
        fpath = data_dir / fname
        if not fpath.is_file():
            continue
        try:
            raw = json.loads(fpath.read_text(encoding="utf-8"))
            feats = raw.get("features") if isinstance(raw, dict) else None
            if not isinstance(feats, list):
                continue
            for ft in feats:
                if not isinstance(ft, dict):
                    continue
                props = ft.get("properties")
                if not isinstance(props, dict):
                    continue
                name = str(props.get("name") or props.get("nameascii") or "").strip()
                if not name:
                    continue

                geom = ft.get("geometry")
                coords = geom.get("coordinates") if isinstance(geom, dict) else None
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    pt = _valid_coord(coords[1], coords[0])
                else:
                    pt = _valid_coord(props.get("latitude"), props.get("longitude"))
                if pt is None:
                    continue

                lat_f, lon_f = pt
                cc = _extract_cc(props)
                adm1 = str(props.get("adm1name") or "").strip()
                try:
                    pop_max = max(0, int(float(props.get("pop_max") or 0)))
                except (TypeError, ValueError):
                    pop_max = 0
                country = _extract_country(props, cc)

                entries.append(
                    _build_entry(lat_f, lon_f, name, cc, adm1, pop_max, country)
                )
            if entries:
                break
        except Exception:
            entries.clear()

    if not entries:
        for lat_f, lon_f, name, cc, adm1, pop_max, country in _FALLBACK_RAW:
            entries.append(
                _build_entry(lat_f, lon_f, name, cc, adm1, pop_max, country)
            )

    entries.sort(key=lambda e: e[0])
    lats = [e[0] for e in entries]
    _CACHE = (entries, lats)
    return _CACHE


def _format_label(name: str, region: str, country: str, dist_km: float) -> str:
    """Build human label based on distance tier (<=25 km, <=120 km, >120 km)."""
    if dist_km <= 25.0:
        return f"{name}, {country}" if country else name
    if dist_km <= 120.0:
        return f"near {name}, {country}" if country else f"near {name}"
    if region and country and region.lower() != country.lower():
        return f"{region}, {country}"
    if country:
        return country
    if region:
        return region
    return f"near {name}"


def _effective_sort_dist(dist_km: float, pop_max: int) -> float:
    """Distance with slight population weighting for major cities within 15 km."""
    if dist_km >= 15.0 or pop_max <= 10_000:
        return dist_km
    import math

    # Smoothly tapered bonus: 0 at 15 km, up to ~1.5 km at 0 km for megacities.
    pop_factor = max(0.0, math.log10(max(pop_max, 10_000)) - 4.0)
    taper = (15.0 - dist_km) / 15.0
    bonus = taper * 0.5 * pop_factor
    return max(0.0, dist_km - bonus)


def nearest_cities(
    lat: float,
    lon: float,
    k: int = 3,
    max_km: float = 300.0,
) -> list[dict]:
    """Return top-``k`` cities within ``max_km`` of ``(lat, lon)``.

    Sorted by effective distance (with a slight population preference for major
    cities within 15 km). Never raises; returns ``[]`` on invalid input or when
    no city lies within ``max_km``.
    """
    try:
        import bisect
        import math

        pt = _valid_coord(lat, lon)
        if pt is None:
            return []
        if isinstance(k, bool) or isinstance(max_km, bool):
            return []
        try:
            k_int = int(k)
            max_km_f = float(max_km)
        except (TypeError, ValueError):
            return []
        if k_int <= 0 or math.isnan(max_km_f) or max_km_f <= 0.0:
            return []
        if math.isinf(max_km_f):
            max_km_f = 20_038.0

        lat_f, lon_f = pt
        entries, lats = _load()
        if not entries:
            return []

        dlat = max_km_f / _DEG_LAT_KM + 0.15
        lo = bisect.bisect_left(lats, lat_f - dlat)
        hi = bisect.bisect_right(lats, lat_f + dlat)

        lat_r = math.radians(lat_f)
        lon_r = math.radians(lon_f)
        cos_lat = math.cos(lat_r)
        pi = math.pi
        two_pi = 2.0 * pi

        # Precompute longitude bounding box in radians when safely away from poles.
        use_lon_box = abs(lat_f) + dlat < 85.0 and max_km_f < 5000.0
        max_dlon_r = (
            math.radians(max_km_f / max(111.32 * math.cos(math.radians(abs(lat_f) + dlat)), 1.0) + 0.25)
            if use_lon_box
            else pi
        )

        candidates: list[tuple[float, float, tuple]] = []
        for entry in entries[lo:hi]:
            clat_r = entry[7]
            clon_r = entry[8]
            dlon_r = clon_r - lon_r
            if dlon_r > pi:
                dlon_r -= two_pi
            elif dlon_r < -pi:
                dlon_r += two_pi
            if use_lon_box and abs(dlon_r) > max_dlon_r:
                continue

            dlat_r = clat_r - lat_r
            s1 = math.sin(dlat_r * 0.5)
            s2 = math.sin(dlon_r * 0.5)
            a = s1 * s1 + cos_lat * entry[9] * s2 * s2
            dist_km = _EARTH_R_KM * 2.0 * math.atan2(
                math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))
            )
            if dist_km <= max_km_f:
                eff_d = _effective_sort_dist(dist_km, entry[5])
                candidates.append((eff_d, dist_km, entry))

        if not candidates:
            return []

        candidates.sort(key=lambda item: (item[0], item[1]))
        out: list[dict] = []
        for _, dist_km, entry in candidates[:k_int]:
            clat, clon, name, cc, adm1, pop_max, country, _, _, _ = entry
            d_round = round(dist_km, 2)
            out.append(
                {
                    "name": name,
                    "region": adm1,
                    "cc": cc,
                    "country": country,
                    "dist_km": d_round,
                    "lat": clat,
                    "lon": clon,
                    "pop_max": pop_max,
                    "label": _format_label(name, adm1, country, dist_km),
                }
            )
        return out
    except Exception:
        return []


def nearest_city(
    lat: float,
    lon: float,
    max_km: float = 250.0,
) -> dict | None:
    """Return the nearest city dict within ``max_km`` of ``(lat, lon)``, or ``None``.

    Returned dict keys:
      ``name``, ``region``, ``cc``, ``country``, ``dist_km``, ``lat``, ``lon``,
      ``pop_max``, ``label``.
    Label rules:
      - ``dist_km <= 25``: ``"Bonn, Germany"``
      - ``dist_km <= 120``: ``"near Bonn, Germany"``
      - ``dist_km > 120``: ``"Nordrhein-Westfalen, Germany"`` (or country name)
    """
    try:
        hits = nearest_cities(lat, lon, k=1, max_km=max_km)
        return hits[0] if hits else None
    except Exception:
        return None


def format_coord_human(lat: float, lon: float) -> str:
    """Format ``(lat, lon)`` into a human location string completely offline.

    Examples:
      - ``(50.73, 7.10)`` -> ``"Bonn, Germany"``
      - ``(50.45, 7.10)`` -> ``"near Bonn, Germany"``
    Falls back to a wider regional/country lookup or formatted coordinates when
    far from populated places, and returns ``""`` on invalid coordinates.
    """
    try:
        pt = _valid_coord(lat, lon)
        if pt is None:
            return ""
        hit = nearest_city(pt[0], pt[1], max_km=250.0)
        if hit and hit.get("label"):
            return str(hit["label"])
        wider = nearest_city(pt[0], pt[1], max_km=600.0)
        if wider and wider.get("label"):
            return str(wider["label"])
        return f"{pt[0]:.2f}, {pt[1]:.2f}"
    except Exception:
        return ""


if __name__ == "__main__":
    import sys
    import time

    # 1. Cold load + first query
    entries, _ = _load()
    assert len(entries) >= 7000, f"expected 7,000+ cities, got {len(entries)}"

    # 2. Bonn lookup (<25 km -> "Bonn, Germany")
    t0 = time.perf_counter()
    bonn = nearest_city(50.73, 7.10)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert bonn is not None, "expected Bonn hit"
    assert bonn["name"] == "Bonn", f"expected Bonn, got {bonn}"
    assert bonn["cc"] == "DE", f"expected DE, got {bonn}"
    assert bonn["country"] == "Germany", f"expected Germany, got {bonn}"
    assert bonn["region"] == "Nordrhein-Westfalen", f"expected NRW, got {bonn}"
    assert bonn["dist_km"] <= 25.0, f"expected <=25 km, got {bonn['dist_km']}"
    assert bonn["label"] == "Bonn, Germany", f"unexpected label: {bonn['label']}"
    assert elapsed_ms < 5.0, f"lookup took {elapsed_ms:.2f} ms (>= 5 ms)"

    # 3. format_coord_human (<5 ms)
    t1 = time.perf_counter()
    human = format_coord_human(50.73, 7.10)
    fmt_ms = (time.perf_counter() - t1) * 1000.0
    assert human == "Bonn, Germany", f"expected 'Bonn, Germany', got {human!r}"
    assert fmt_ms < 5.0, f"format_coord_human took {fmt_ms:.2f} ms (>= 5 ms)"

    # 4. Mid-distance tier (25 < dist_km <= 120 -> "near <City>, <Country>")
    # Point in rural Eifel ~40 km SW of Bonn
    eifel = nearest_city(50.38, 6.85)
    assert eifel is not None
    assert 25.0 < eifel["dist_km"] <= 120.0, f"expected 25..120 km, got {eifel}"
    assert eifel["label"].startswith("near "), f"expected 'near ...', got {eifel['label']}"

    # 5. Far-distance tier (>120 km -> "<Region>, <Country>" or "<Country>")
    far_label = _format_label("Bonn", "Nordrhein-Westfalen", "Germany", 160.0)
    assert far_label == "Nordrhein-Westfalen, Germany", f"got {far_label!r}"

    # 6. nearest_cities top-k
    top3 = nearest_cities(50.73, 7.10, k=3, max_km=300.0)
    assert len(top3) == 3, f"expected 3 cities, got {len(top3)}"
    assert top3[0]["name"] == "Bonn"

    # 7. Ocean / out-of-range / invalid input resilience (never raises)
    assert nearest_city(0.0, -160.0, max_km=50.0) is None
    for bad in (
        (None, None),
        ("bad", "coord"),
        (95.0, 0.0),
        (0.0, 195.0),
        (float("nan"), 7.1),
        (float("inf"), 7.1),
        (True, False),
    ):
        assert nearest_city(bad[0], bad[1]) is None  # type: ignore[arg-type]
        assert nearest_cities(bad[0], bad[1]) == []  # type: ignore[arg-type]
        assert format_coord_human(bad[0], bad[1]) == ""  # type: ignore[arg-type]

    assert nearest_cities(50.73, 7.10, k=0) == []
    assert nearest_cities(50.73, 7.10, k=-2) == []
    assert nearest_cities(50.73, 7.10, max_km=-10.0) == []

    print(
        f"offline_places self-test OK: {len(entries)} cities loaded, "
        f"query={elapsed_ms:.3f} ms, format_coord_human(50.73, 7.10)={human!r}"
    )
    sys.exit(0)
