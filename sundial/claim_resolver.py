"""Claim resolver — one shared anchor object for the whole pipeline.

Every geo-aware stage (solar, weather, landmarks, POIs, wiki geosearch,
timezone, camera-GPS distance) needs the *same* resolved point. run()
resolves the claim once up front and hands the resulting anchor dict to
every stage instead of re-geocoding per stage:

    resolve_claim("Husum, Germany", "2025-06-15T15:00")
    -> {"ok": True, "lat": 54.4858, "lon": 9.0564, "cc": "DE",
        "name": "Husum, Nordfriesland, Schleswig-Holstein, Germany",
        "tz": "Europe/Berlin", "source": "geocode",
        "claimed_utc": "2025-06-15T13:00:00+00:00",
        "confidence": 0.77, "notes": [...]}

Resolution order:
  * "lat,lon" claims anchor directly — no geocode, source="latlon".
  * anything else goes through geo.geocode; the top-scored candidate
    wins ("score-weighted top hit"), source="geocode".

Country code, best source first: pip2cc.cc_at point-in-polygon (when
that module exists) -> the geocoder's own country_code -> the claim's
comma tail via reverse_geo.claim_country_cc (offline string match when
called without coords) -> a coarse local bbox table as last resort.

Timezone: tz_check.expected_tz_name when importable, then a small local
bbox table behind it. Both tables are honest-smoke-test grade, matching
tz_check's own convention — a plausible zone, never truth.

Stdlib only at import; geo / pip2cc / reverse_geo / tz_check / zoneinfo
are all lazy inside functions. Public functions never raise — failures
return {"ok": False, ...} (resolve_claim) or None (anchor_from).

    python -m sundial.claim_resolver     # offline self-test
    python3 sundial/claim_resolver.py    # same, direct-script run
"""
from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["resolve_claim", "anchor_from"]

# --- test / offline hooks -----------------------------------------------------
# Assign callables here (tests do) to run resolve_claim without the network:
#   _GEOCODER  : fn(query: str) -> list[GeoCandidate | dict]
#   _CC_AT     : fn(lat, lon) -> "DE" | None            (pip2cc.cc_at shape)
#   _TZ_LOOKUP : fn(lat, lon) -> "Europe/Berlin" | None
_GEOCODER = None
_CC_AT = None
_TZ_LOOKUP = None

# --- last-resort bbox tables ---------------------------------------------------
# (lat_lo, lat_hi, lon_lo, lon_hi, value). First match wins; small/narrow
# countries sit before the big neighbours their boxes overlap. These only
# fire when the real lookups (pip2cc, tz_check) are absent or uncovered —
# deliberately tiny siblings of tz_check's own honest table.
_TZ_BOXES = [
    (51.3, 55.5, -10.7, -5.9, "Europe/Dublin"),        # IE before GB
    (50.7, 53.7, 3.2, 7.3, "Europe/Amsterdam"),        # NL
    (49.4, 51.6, 2.5, 6.5, "Europe/Brussels"),         # BE
    (45.7, 47.9, 5.9, 10.6, "Europe/Zurich"),          # CH
    (46.3, 49.1, 9.5, 17.2, "Europe/Vienna"),          # AT
    (47.2, 55.1, 5.8, 15.1, "Europe/Berlin"),          # DE
    (42.3, 51.1, -5.2, 9.7, "Europe/Paris"),           # FR metro
    (49.9, 60.9, -8.7, 2.1, "Europe/London"),          # GB
    (35.9, 43.8, -9.4, 4.4, "Europe/Madrid"),          # ES mainland
    (36.6, 47.1, 6.6, 18.6, "Europe/Rome"),            # IT
    (55.3, 69.1, 10.9, 24.2, "Europe/Stockholm"),      # SE
    (41.0, 78.0, 30.0, 42.0, "Europe/Moscow"),         # RU west
    (35.8, 42.2, 25.6, 44.9, "Europe/Istanbul"),       # TR
    (24.0, 46.0, 122.7, 146.0, "Asia/Tokyo"),          # JP
    (18.0, 53.6, 73.5, 135.1, "Asia/Shanghai"),        # CN (single zone)
    (6.7, 35.7, 68.1, 89.7, "Asia/Kolkata"),           # IN
    (16.0, 32.2, 34.4, 55.7, "Asia/Riyadh"),           # SA/levant rough
    (24.4, 49.4, -125.0, -115.0, "America/Los_Angeles"),   # US Pacific
    (24.4, 49.4, -115.0, -104.0, "America/Denver"),        # US Mountain
    (24.4, 49.4, -104.0, -88.0, "America/Chicago"),        # US Central
    (24.4, 49.4, -88.0, -66.9, "America/New_York"),        # US Eastern
    (-56.0, -17.4, -75.8, -34.7, "America/Santiago"),      # S. America rough
    (-44.0, -10.0, 112.0, 129.0, "Australia/Perth"),       # AU west
    (-44.0, -10.0, 129.0, 141.0, "Australia/Adelaide"),    # AU central
    (-44.0, -10.0, 141.0, 154.0, "Australia/Sydney"),      # AU east
    (-47.4, -34.3, 166.3, 178.7, "Pacific/Auckland"),      # NZ
]

_CC_BOXES = [
    (51.3, 55.5, -10.7, -5.9, "IE"),                 # Ireland before GB
    (49.4, 50.2, 5.7, 6.6, "LU"),
    (50.7, 53.7, 3.2, 7.3, "NL"),
    (49.4, 51.6, 2.5, 6.5, "BE"),
    (45.7, 47.9, 5.9, 10.6, "CH"),
    (46.3, 49.1, 9.5, 17.2, "AT"),
    (48.5, 51.1, 12.0, 18.9, "CZ"),
    (54.3, 57.9, 7.9, 12.9, "DK"),
    (49.0, 54.9, 14.0, 24.2, "PL"),
    (47.2, 55.1, 5.8, 15.1, "DE"),
    (42.3, 51.1, -5.2, 9.7, "FR"),
    (49.9, 60.9, -8.7, 2.1, "GB"),
    (36.8, 42.2, -9.6, -6.1, "PT"),                  # PT mainland before ES
    (35.9, 43.8, -9.4, 4.4, "ES"),
    (36.6, 47.1, 6.6, 18.6, "IT"),
    (55.3, 69.1, 10.9, 24.2, "SE"),
    (57.9, 71.3, 4.4, 31.2, "NO"),
    (34.8, 41.8, 19.3, 29.8, "GR"),
    (35.8, 42.2, 25.6, 44.9, "TR"),
    (24.0, 46.0, 122.7, 146.0, "JP"),
    (18.0, 53.6, 73.5, 135.1, "CN"),
    (6.7, 35.7, 68.1, 89.7, "IN"),
    (14.5, 32.6, -117.1, -86.7, "MX"),               # MX strip before US
    (24.4, 49.4, -125.0, -66.9, "US"),               # CONUS only — coarse
    (49.4, 70.0, -141.0, -52.0, "CA"),               # southern band
    (-56.0, -17.4, -75.8, -66.3, "CL"),              # CL sliver before AR
    (-55.1, -21.7, -73.7, -53.5, "AR"),
    (-33.8, 5.4, -74.0, -34.7, "BR"),
    (-44.0, -10.0, 112.0, 154.0, "AU"),
    (-47.4, -34.3, 166.3, 178.7, "NZ"),
    (-34.9, -22.1, 16.3, 33.0, "ZA"),
    (22.0, 31.7, 24.6, 36.9, "EG"),
    (16.0, 32.2, 34.4, 55.7, "SA"),
    (5.5, 20.5, 97.3, 105.8, "TH"),
]


# --- small helpers -------------------------------------------------------------

def _parse_latlon(text: object) -> tuple[float, float] | None:
    """"lat,lon" string -> (lat, lon); None otherwise. Same parse as
    pipeline._parse_latlon / geo.py's claim path."""
    try:
        a, b = [float(x) for x in str(text).split(",")[:2]]
        if -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0:
            return a, b
    except (TypeError, ValueError):
        pass
    return None


def _cand_get(cand: object, key: str, default: object = None) -> object:
    """Field access that works on GeoCandidate objects and plain dicts."""
    if isinstance(cand, dict):
        return cand.get(key, default)
    return getattr(cand, key, default)


def _latlon_of(obj: object) -> tuple[float, float] | None:
    """Best-effort (lat, lon) out of {lat, lon} dicts, [lat, lon] pairs,
    "lat,lon" strings, and objects with .lat/.lon (GeoCandidate)."""
    if obj is None:
        return None
    lat = lon = None
    if isinstance(obj, str):
        return _parse_latlon(obj)
    if isinstance(obj, dict):
        lat, lon = obj.get("lat"), obj.get("lon")
    elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
        lat, lon = obj[0], obj[1]
    else:
        lat, lon = getattr(obj, "lat", None), getattr(obj, "lon", None)
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def _find_latlon(obj: object, depth: int = 0) -> tuple[float, float] | None:
    """First valid (lat, lon) inside a nested dict/list/object payload."""
    if obj is None or depth > 4:
        return None
    hit = _latlon_of(obj)
    if hit is not None:
        return hit
    if isinstance(obj, dict):
        items = obj.values()
    elif isinstance(obj, (list, tuple)):
        items = obj
    else:
        return None
    for v in items:
        hit = _find_latlon(v, depth + 1)
        if hit is not None:
            return hit
    return None


def _box_lookup(lat: float, lon: float, boxes: list) -> str | None:
    """First-match bbox table lookup (same rule as tz_check)."""
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    for lat_lo, lat_hi, lon_lo, lon_hi, val in boxes:
        if lat_lo <= la <= lat_hi and lon_lo <= lo <= lon_hi:
            return val
    return None


# --- lazy resolvers (each returns (value, how-it-got-it), never raises) --------

def _geocode(query: str) -> tuple[list, str | None]:
    """-> (candidates, error|None). geo.geocode raises RuntimeError on
    transport failure — here that becomes data, not a crash."""
    if _GEOCODER is not None:
        try:
            return list(_GEOCODER(query) or []), None
        except Exception as e:  # noqa: BLE001 — stub reported, not raised
            return [], f"geocoder hook failed: {e}"
    try:
        from .geo import geocode
    except ImportError:
        try:
            from geo import geocode  # direct-script run: sundial/ on sys.path
        except ImportError:
            return [], "geo module unavailable"
    try:
        return list(geocode(query) or []), None
    except Exception as e:  # noqa: BLE001 — RuntimeError -> note
        return [], str(e)


def _cc_lookup(lat: float, lon: float) -> tuple[str | None, str | None]:
    """-> (cc|None, how). pip2cc.cc_at when that module exists."""
    if _CC_AT is not None:
        try:
            cc = _CC_AT(lat, lon)
            if cc:
                return str(cc).strip().upper(), "cc hook"
        except Exception:  # noqa: BLE001
            pass
    cc_at = None
    try:
        from .pip2cc import cc_at
    except ImportError:
        try:
            from pip2cc import cc_at  # direct-script run
        except ImportError:
            cc_at = None
    if cc_at is not None:
        try:
            cc = cc_at(lat, lon)
            if cc:
                return str(cc).strip().upper(), "pip2cc"
        except Exception:  # noqa: BLE001
            pass
    return None, None


def _tail_cc(place: str) -> str | None:
    """Country code from the claim's comma tail ('Husum, Germany' -> DE).

    reverse_geo.claim_country_cc with no coordinates is a pure offline
    string match — the /reverse HTTP path only runs when lat/lon are
    passed, which we deliberately don't.
    """
    try:
        from .reverse_geo import claim_country_cc
    except ImportError:
        try:
            from reverse_geo import claim_country_cc  # direct-script run
        except ImportError:
            return None
    try:
        return claim_country_cc(place)  # no coords -> no HTTP
    except Exception:  # noqa: BLE001
        return None


def _tz_lookup(lat: float, lon: float) -> tuple[str | None, str | None]:
    """-> (zone|None, how). tz_check.expected_tz_name first, coarse
    table behind it."""
    if _TZ_LOOKUP is not None:
        try:
            z = _TZ_LOOKUP(lat, lon)
            if z:
                return str(z), "tz hook"
        except Exception:  # noqa: BLE001
            pass
    expected = None
    try:
        from .tz_check import expected_tz_name as expected
    except ImportError:
        try:
            from tz_check import expected_tz_name as expected  # direct run
        except ImportError:
            expected = None
    if expected is not None:
        try:
            z = expected(lat, lon)
            if z:
                return str(z), "tz_check"
        except Exception:  # noqa: BLE001
            pass
    z = _box_lookup(lat, lon, _TZ_BOXES)
    if z:
        return z, "coarse bbox table"
    return None, None


def _claimed_utc(claim_time_iso: object, tz_name: str | None) -> tuple[str | None, str | None]:
    """-> (utc_iso|None, note|None).

    A naive claim time is interpreted in the anchor zone via zoneinfo
    (fold=0 around DST switches — coarse, same as tz_check). An ISO
    string that already carries an offset is taken as-is."""
    raw = str(claim_time_iso or "").strip()
    if not raw:
        return None, None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None, f"claim time {raw!r} unparseable — no UTC conversion"
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).isoformat(), (
            f"claim time {raw!r} carries its own offset — used verbatim")
    if not tz_name:
        return None, "no anchor timezone — claim time can't be converted"
    try:
        from zoneinfo import ZoneInfo  # lazy: tzdata-free platforms may lack it
        utc = dt.replace(tzinfo=ZoneInfo(str(tz_name))).astimezone(timezone.utc)
        return utc.isoformat(), (
            f"claim local {dt.isoformat()} -> UTC via zoneinfo {tz_name}")
    except Exception as e:  # noqa: BLE001
        return None, f"zoneinfo conversion failed ({e})"


# --- public API -----------------------------------------------------------------

def resolve_claim(claim_place: str | None, claim_time_iso: str | None = None) -> dict:
    """Resolve a place claim into the pipeline's shared anchor dict.

    Keys: ok, lat, lon, cc, name (+label alias), tz, source
    ("latlon"|"geocode"|"none"), confidence 0..1, claimed_utc (claim time
    read in the anchor zone, converted to UTC), notes (audit trail).
    Never raises — failures return {"ok": False, ...}.
    """
    out: dict = {
        "ok": False,
        "lat": None,
        "lon": None,
        "cc": None,
        "name": None,
        "label": None,
        "tz": None,
        "source": "none",
        "confidence": 0.0,
        "claimed_utc": None,
        "notes": [],
    }
    notes = out["notes"]
    try:
        raw = str(claim_place or "").strip()
        if not raw:
            notes.append("no claim place to resolve")
            return out
        out["name"] = out["label"] = raw

        best = None
        pair = _parse_latlon(raw)
        if pair is not None:
            lat, lon = pair
            out["source"] = "latlon"
            out["confidence"] = 0.95
            notes.append(f"claim {raw!r} parsed as literal coordinates")
        else:
            cands, err = _geocode(raw)
            if err:
                notes.append(f"geocoder failed for {raw!r}: {err}")
            if not cands:
                if not err:
                    notes.append(f"geocoder found no match for {raw!r}")
                return out
            # top hit, score-weighted: highest score wins, order breaks ties
            best = cands[0]
            for c in cands[1:]:
                try:
                    if float(_cand_get(c, "score") or 0.0) > float(
                        _cand_get(best, "score") or 0.0
                    ):
                        best = c
                except (TypeError, ValueError):
                    continue
            pair = _latlon_of(best)
            if pair is None:
                notes.append(
                    f"top geocode hit for {raw!r} carried no usable coordinates"
                )
                return out
            lat, lon = pair
            out["source"] = "geocode"
            try:
                score = float(_cand_get(best, "score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            # Nominatim importance ~0.6-0.8 for real places -> ~0.76-0.83.
            out["confidence"] = round(
                min(0.55 + 0.35 * min(max(score, 0.0), 1.0), 0.9), 2
            )
            display = str(
                _cand_get(best, "display_name")
                or _cand_get(best, "name")
                or raw
            ).strip() or raw
            out["name"] = out["label"] = display
            notes.append(
                f"geocoded {raw!r} -> {display!r} "
                f"(score {score:.2f}, {len(cands)} candidate(s))"
            )

        out["lat"], out["lon"] = round(lat, 6), round(lon, 6)
        out["ok"] = True

        # --- country code ----------------------------------------------------
        cc, how = _cc_lookup(lat, lon)
        if cc:
            notes.append(f"cc={cc} via {how}")
        elif best is not None:
            cc = str(_cand_get(best, "country_code") or "").strip().upper() or None
            if cc:
                notes.append(f"cc={cc} from geocoder")
        if not cc:
            tail = _tail_cc(raw)
            if tail:
                cc = tail
                notes.append(
                    f"cc={cc} from claim tail {raw.rsplit(',', 1)[-1].strip()!r}"
                )
        if not cc:
            cc = _box_lookup(lat, lon, _CC_BOXES)
            if cc:
                notes.append(f"cc={cc} via coarse bbox table (approximate)")
        if not cc:
            notes.append(
                "cc unresolved — no pip2cc, no geocoder cc, outside coarse table"
            )
        out["cc"] = cc

        # --- timezone ----------------------------------------------------------
        tz, how = _tz_lookup(lat, lon)
        out["tz"] = tz
        notes.append(
            f"tz={tz} via {how}" if tz
            else "tz unresolved — point outside all tz tables"
        )

        # --- claim time -> UTC -------------------------------------------------
        utc, utc_note = _claimed_utc(claim_time_iso, tz)
        out["claimed_utc"] = utc
        if utc_note:
            notes.append(utc_note)
        return out
    except Exception as e:  # noqa: BLE001 — spec: never raise
        notes.append(f"resolve_claim error: {e}")
        return out


def _ev_type(ev: object) -> str:
    t = getattr(ev, "type", None)
    if t is None and isinstance(ev, dict):
        t = ev.get("type")
    return str(t or "").strip().lower()


def _ev_data(ev: object) -> dict:
    """Evidence.data whether the card is a dataclass, a serialized dict,
    or a bare payload dict."""
    data = getattr(ev, "data", None)
    if data is None and isinstance(ev, dict):
        data = ev.get("data")
        if data is None:
            data = ev  # a bare payload dict IS the data
    return data if isinstance(data, dict) else {}


def anchor_from(evidence_list: object) -> tuple[float, float] | None:
    """Recover the (lat, lon) claim anchor from evidence data.

    For when run() forgot to pass the anchor through: stages that saw it
    record it (data["anchor"] on landmark/poi cards; flat lat/lon on
    weather), the geo card's claim_candidates IS the resolved claim, and
    wiki_landmark/poi matches sit within metres of it. Accepts Evidence
    objects, serialized dicts, or an EvidencePack-like object with
    .evidence. Returns None when nothing carries coordinates.
    """
    evs = evidence_list or []
    if hasattr(evs, "evidence"):          # EvidencePack-like
        evs = getattr(evs, "evidence")
    if isinstance(evs, dict):             # serialized pack
        evs = evs.get("evidence") or []
    try:
        cards = [(_ev_type(ev), _ev_data(ev)) for ev in evs or []]
    except TypeError:
        return None                       # non-iterable input — no anchor

    # 1. a card that recorded the anchor verbatim wins outright
    for _, data in cards:
        hit = _latlon_of(data.get("anchor"))
        if hit is not None:
            return hit

    # 2. claim-side geo cards: the geocoded/typed claim itself
    for t, data in cards:
        if t not in ("geo-resolve", "geo", "geo_match", "geocode"):
            continue
        for key in ("claim_candidates", "resolved", "match", "candidates"):
            hit = _find_latlon(data.get(key))
            if hit is not None:
                return hit
        hit = _latlon_of(data)
        if hit is not None:
            return hit

    # 3. flat lat/lon on any card — weather/solar store the resolved
    #    claim coords top-level
    for _, data in cards:
        hit = _latlon_of(data)
        if hit is not None:
            return hit

    # 4. landmark/poi/wiki payloads — places confirmed near the anchor
    for t, data in cards:
        if t not in ("wiki_landmark", "wiki", "landmark", "landmarks", "poi"):
            continue
        for key in ("matches", "places", "pois", "hits"):
            hit = _find_latlon(data.get(key))
            if hit is not None:
                return hit
    return None


# --------------------------------------------------------------------------
# Self-test — offline. The geocoder is monkeypatched via the module hook;
# cc/tz paths are local lookups anyway. python3 sundial/claim_resolver.py
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    fails = 0

    def check(cond: bool, label: str, detail: str = "") -> None:
        global fails
        fails += 0 if cond else 1
        print(f"  {'ok  ' if cond else 'FAIL'}  {label} {detail}")

    # --- 1. literal coords — no geocoder involved -----------------------------
    a = resolve_claim("51.50,-0.12", "2024-06-15T15:00")
    print("anchor:", a)
    check(a["ok"] is True and a["source"] == "latlon", "latlon resolves")
    check((a["lat"], a["lon"]) == (51.5, -0.12), "coords passthrough")
    check(a["cc"] == "GB", "cc=GB", f"(got {a['cc']!r})")
    check(a["tz"] == "Europe/London", "tz=Europe/London", f"(got {a['tz']!r})")
    if a["claimed_utc"] is not None:
        check(
            a["claimed_utc"].startswith("2024-06-15T14:00"),
            "15:00 Europe/London (BST) -> 14:00Z",
            a["claimed_utc"],
        )
    else:
        print("  ..    zoneinfo unavailable — claimed_utc check skipped")

    # --- 2. text claim via stubbed geocoder -------------------------------------
    def _stub_geo(q: str) -> list[dict]:
        if "husum" in str(q).lower():
            return [{
                "name": "Husum",
                "lat": 54.4858,
                "lon": 9.0564,
                "country_code": "de",
                "display_name":
                    "Husum, Nordfriesland, Schleswig-Holstein, Germany",
                "score": 0.62,
            }]
        return []

    _GEOCODER = _stub_geo
    b = resolve_claim("Husum, Germany")
    print("anchor:", b)
    check(b["ok"] is True and b["source"] == "geocode", "geocode resolves")
    check(
        abs(b["lat"] - 54.4858) < 1e-6 and abs(b["lon"] - 9.0564) < 1e-6,
        "geocoded coords",
    )
    check(b["cc"] == "DE", "cc=DE", f"(got {b['cc']!r})")
    # Husum sits in tz_check's coarse DK box (54.3-57.9N, 7.9-12.9E), which
    # precedes DE — a documented border artifact; either CET zone is right.
    check(b["tz"] in ("Europe/Berlin", "Europe/Copenhagen"),
          "tz=CET zone (Berlin or border-artifact Copenhagen)",
          f"(got {b['tz']!r})")
    check(str(b["name"]).startswith("Husum"), "label from display_name")

    # --- 3. garbage — ok:false, never raises --------------------------------------
    c = resolve_claim("zzz-not-a-place !!!")
    print("anchor:", c)
    check(c["ok"] is False and c["source"] == "none", "garbage -> ok:false")
    check(resolve_claim(None)["ok"] is False, "None claim -> ok:false")
    check(resolve_claim("")["ok"] is False, "empty claim -> ok:false")
    _GEOCODER = None

    # --- 4. anchor_from ------------------------------------------------------------
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:
        try:
            from schema import ClaimVerdict, Evidence  # direct-script run
        except ImportError:
            ClaimVerdict = Evidence = None

    evs: list = [{
        "type": "geo",
        "data": {"claim_candidates": [{"lat": 51.5, "lon": -0.12}]},
    }]
    if Evidence is not None:
        evs.insert(0, Evidence(
            type="wiki_landmark",
            summary="x",
            verdict=ClaimVerdict.INCONCLUSIVE,
            data={"matches": [{"title": "Tower Bridge",
                               "lat": 51.5055, "lon": -0.0754}]},
        ))
    got = anchor_from(evs)
    check(got == (51.5, -0.12), "anchor_from: geo claim_candidates wins",
          f"(got {got})")
    if Evidence is not None:
        got = anchor_from([evs[0]])
        check(got == (51.5055, -0.0754), "anchor_from: wiki matches",
              f"(got {got})")
    check(anchor_from([{"type": "landmark",
                        "data": {"anchor": [48.8583, 2.2945]}}])
          == (48.8583, 2.2945), "anchor_from: data['anchor'] list")
    check(anchor_from([{"type": "weather",
                        "data": {"lat": 40.7, "lon": -74.0}}])
          == (40.7, -74.0), "anchor_from: flat lat/lon")
    check(anchor_from([]) is None, "anchor_from: empty -> None")
    check(anchor_from([{"type": "clock", "data": {"overlays": ["08:00"]}},
                       "junk", None, 42]) is None,
          "anchor_from: no coords -> None")

    if fails:
        print(f"self-test FAILED ({fails} check(s))")
        sys.exit(1)
    print("self-test OK")
