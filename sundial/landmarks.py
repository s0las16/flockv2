"""Landmark stage — geocode VLM-named landmarks and check them against the claim.

The OSINT scene prompt asks the VLM for `landmark_candidates`: proper names /
distinctive structures it claims are visible ("Tower Bridge", "Shibuya
crossing"). Unlike sign *text* these are semantic identifications — and VLMs
invent them confidently — so the same >=2-distinct-frame consensus rule the
pipeline applies to sign text is applied here before any (rate-limited)
Nominatim lookup happens.

Verdict shape mirrors geo.geo_evidence's asymmetry: a real landmark resolving
next to the claimed spot is decent support, while every claimed landmark
landing hundreds of kilometres away contradicts. The middle stays
INCONCLUSIVE — landmark names are fuzzy and Nominatim top-hits are coarse.

Stdlib-only at import time: geo (requests) and schema are imported lazily
inside the functions that need them.
"""
from __future__ import annotations

import math
from collections import Counter

# Distance bands (km) around the claimed anchor point.
_NEAR_KM = 40.0  # a landmark this close corroborates the claimed location
_FAR_KM = 400.0  # *every* landmark beyond this contradicts it


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km — verbatim copy of geo._haversine_km so
    this module imports without touching geo (or its requests dep)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------
# Consensus: the same hallucination guard the pipeline applies to sign text —
# invented names don't repeat across frames, real ones do. Twice in ONE frame
# still counts once: distinct frames are what makes reads independent.
# --------------------------------------------------------------------------

def consensus_landmarks(per_frame_landmarks: list[list[str]]) -> list[str]:
    """Names seen in >=2 *distinct* frames; casing kept from first sighting.

    `per_frame_landmarks` is the per-frame `landmark_candidates` lists in
    frame order. Normalization is strip + casefold; output preserves the
    first-seen casing and first-sighting order.
    """
    frames_of: dict[str, set[int]] = {}  # normalized name -> frame indices seen
    first_casing: dict[str, str] = {}
    order: list[str] = []
    for idx, frame in enumerate(per_frame_landmarks or []):
        # Tolerate a bare string / tuple frame instead of list[str].
        items = frame if isinstance(frame, (list, tuple)) else ([frame] if frame else [])
        for raw in items:
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            key = text.lower()
            if not key:
                continue
            if key not in frames_of:
                frames_of[key] = set()
                first_casing[key] = text
                order.append(key)
            frames_of[key].add(idx)
    return [first_casing[k] for k in order if len(frames_of[k]) >= 2]


def recognized_landmarks(osint_blobs: list[dict]) -> list[str]:
    """Visually-recognised landmark names out of per-frame OSINT blobs.

    `recognized_landmark` is the visual-identification field (vs
    `landmark_candidates`, which is text the model *read*). Voting:
    how_sure "certain"/"likely" is worth two votes — a single confident
    sighting survives; "maybe" is one vote and needs a second frame to
    agree. First-sighting order, casing preserved.
    """
    votes: dict[str, int] = {}
    first_casing: dict[str, str] = {}
    order: list[str] = []
    for blob in osint_blobs or []:
        rec = blob.get("recognized_landmark") if isinstance(blob, dict) else None
        if isinstance(rec, str):  # tolerate a bare string
            rec = {"name": rec, "how_sure": "likely"}
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name") or "").strip()
        if not name:
            continue
        sure = str(rec.get("how_sure") or "").strip().lower()
        key = name.lower()
        if key not in votes:
            votes[key] = 0
            first_casing[key] = name
            order.append(key)
        votes[key] += 2 if sure in ("certain", "likely") else 1
    return [first_casing[k] for k in order if votes[k] >= 2]


def landmark_candidates_to_places(names: list[str], limit: int = 5) -> list[dict]:
    """Geocode consensus landmark names -> flat place dicts.

    One place per name (Nominatim's top hit), in input order, capped at
    `limit`. A name that errors or resolves to nothing is skipped — a
    VLM-invented name failing to geocode is routine, not fatal.
    """
    from . import geo  # lazy: pulls in requests + the Nominatim rate limiter

    places: list[dict] = []
    seen: set[str] = set()
    for raw in names or []:
        if len(places) >= limit:
            break
        name = raw.strip() if isinstance(raw, str) else ""
        key = name.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            cands = geo.geocode(name)
        except Exception:
            continue
        if not cands:
            continue
        c = cands[0]
        places.append({
            "name": name,
            "lat": c.lat,
            "lon": c.lon,
            "cc": c.country_code,
            "display_name": c.display_name,
            "score": c.score,
        })
    return places


def landmark_evidence(
    names: list[str],
    places: list[dict],
    anchor: tuple[float, float] | None,
    claimed_cc: str | None,
) -> "Evidence":
    """Fold consensus landmark names + their geocoded places into Evidence.

    `anchor` is the claimed location as (lat, lon) when already resolved;
    `claimed_cc` the claimed ISO country code when that's all we have.

    With an anchor: any landmark within 40 km SUPPORTS (0.65); ALL landmarks
    >400 km away CONTRADICTS (0.55); in between INCONCLUSIVE.
    Without an anchor but with claimed_cc: a strict-majority country among
    places matching claimed_cc SUPPORTS (0.5), mismatching CONTRADICTS (0.5).
    """
    from .schema import ClaimVerdict, Evidence  # lazy: keeps module dep-free

    names_clean = [n.strip() for n in (names or []) if isinstance(n, str) and n.strip()]
    places = [p for p in (places or []) if isinstance(p, dict)]
    cc_claim = (claimed_cc or "").strip().upper()
    data: dict = {
        "names": names_clean,
        "places": places,
        "anchor": list(anchor) if anchor is not None else None,
        "claimed_cc": cc_claim or None,
    }

    if not names_clean:
        return Evidence(
            type="landmark",
            summary="no named landmarks detected",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.1,
            data={**data, "distances_km": {}},
        )

    # --- distance of each place to the anchor (when one exists) ------------
    dists: dict[str, float] = {}
    if anchor is not None:
        try:
            alat, alon = float(anchor[0]), float(anchor[1])
        except (TypeError, ValueError, IndexError):
            alat = alon = None
        if alat is not None:
            for p in places:
                try:
                    plat, plon = float(p["lat"]), float(p["lon"])
                except (KeyError, TypeError, ValueError):
                    continue
                pname = str(p.get("name") or "?")
                dists[pname] = round(_haversine_km(alat, alon, plat, plon), 1)
    data["distances_km"] = dists

    if not places:
        return Evidence(
            type="landmark",
            summary=(
                f"{len(names_clean)} consensus landmark name(s) "
                f"({', '.join(names_clean[:3])}) but none geocoded — "
                "can't check them against the claim"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data=data,
        )

    # --- anchor path: distance bands ----------------------------------------
    if dists:
        near_name = min(dists, key=lambda k: dists[k])
        near_d = dists[near_name]
        if near_d <= _NEAR_KM:
            data["nearest"] = {"name": near_name, "distance_km": near_d}
            return Evidence(
                type="landmark",
                summary=(
                    f"landmark {near_name!r} resolves {near_d:.0f} km from "
                    "the claimed location — consistent"
                ),
                verdict=ClaimVerdict.SUPPORTS,
                confidence=0.65,
                data=data,
            )
        if all(d > _FAR_KM for d in dists.values()):
            data["nearest"] = {"name": near_name, "distance_km": near_d}
            return Evidence(
                type="landmark",
                summary=(
                    f"all {len(dists)} geocoded landmark(s) sit "
                    f">{_FAR_KM:.0f} km from the claimed location "
                    f"(nearest {near_name!r} at {near_d:.0f} km)"
                ),
                verdict=ClaimVerdict.CONTRADICTS,
                confidence=0.55,
                data=data,
            )
        return Evidence(
            type="landmark",
            summary=(
                f"nearest geocoded landmark {near_name!r} is {near_d:.0f} km "
                f"from the claimed location — between the "
                f"{_NEAR_KM:.0f}/{_FAR_KM:.0f} km decision bands"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.3,
            data=data,
        )

    # --- country-code path (no usable anchor) -------------------------------
    ccs = [
        str(p.get("cc") or "").strip().upper()
        for p in places
        if str(p.get("cc") or "").strip()
    ]
    if ccs:
        data["place_country_codes"] = dict(Counter(ccs))
    if cc_claim and ccs:
        mode_cc, mode_n = Counter(ccs).most_common(1)[0]
        data["majority_cc"] = mode_cc
        if mode_n > len(ccs) / 2:  # strict majority — a tie proves nothing
            if mode_cc == cc_claim:
                return Evidence(
                    type="landmark",
                    summary=(
                        f"geocoded landmarks mostly sit in {mode_cc} "
                        f"({mode_n}/{len(ccs)}) — matches claimed country "
                        f"code {cc_claim}"
                    ),
                    verdict=ClaimVerdict.SUPPORTS,
                    confidence=0.5,
                    data=data,
                )
            return Evidence(
                type="landmark",
                summary=(
                    f"geocoded landmarks mostly sit in {mode_cc} "
                    f"({mode_n}/{len(ccs)}) — not the claimed country "
                    f"code {cc_claim}"
                ),
                verdict=ClaimVerdict.CONTRADICTS,
                confidence=0.5,
                data=data,
            )

    if anchor is not None:
        why = "anchor supplied but no place carried usable coordinates"
    elif cc_claim and not ccs:
        why = "geocoded places carry no country codes to compare"
    elif cc_claim:
        why = "no majority country across geocoded places (too mixed)"
    else:
        why = "no anchor or claimed country code supplied"
    return Evidence(
        type="landmark",
        summary=(
            f"{len(names_clean)} consensus landmark(s) "
            f"({', '.join(names_clean[:3])}) geocoded; {why}"
        ),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.2,
        data=data,
    )


# --------------------------------------------------------------------------
# Demo / self-check: python -m sundial.landmarks [names...] [--anchor lat,lon] [--cc GB]
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    fails = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        global fails
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label} {detail}")

    # --- self-check 1: consensus rule (no network) --------------------------
    print("consensus self-check:")
    frames = [
        ["Tower Bridge", "big ben", "pigeon flock"],
        ["tower bridge", "HMS Belfast"],
        [" Tower Bridge "],
    ]
    got = consensus_landmarks(frames)
    check("consensus keeps 'Tower Bridge' (3 frames)", got == ["Tower Bridge"],
          f"(got {got})")
    check("'big ben' filtered (1 frame only)", "big ben" not in got)
    check("'HMS Belfast' filtered (1 frame only)", "HMS Belfast" not in got)
    check("empty input -> []", consensus_landmarks([]) == [])
    check("same frame twice -> []", consensus_landmarks([["x", "x"]]) == [])

    # --- self-check 2: evidence logic on stub places (no network) -----------
    print("evidence self-check (stub places):")
    stub = [
        {"name": "Tower Bridge", "lat": 51.5055, "lon": -0.0754,
         "cc": "GB", "display_name": "Tower Bridge, London, UK", "score": 0.9},
        {"name": "HMS Belfast", "lat": 51.5066, "lon": -0.0814,
         "cc": "GB", "display_name": "HMS Belfast, London, UK", "score": 0.7},
        {"name": "Shibuya Crossing", "lat": 35.6595, "lon": 139.7005,
         "cc": "JP", "display_name": "Shibuya, Tokyo, JP", "score": 0.6},
    ]
    london = (51.5074, -0.1278)   # central London anchor
    sydney = (-33.8688, 151.2093)
    paris = (48.8566, 2.3522)
    try:
        cases = [
            # names, places, anchor, cc, want verdict
            (["Tower Bridge"], stub, london, "GB", "supports"),      # ~5 km
            (["Tower Bridge"], stub, sydney, None, "contradicts"),   # ~17k km
            (["Tower Bridge"], stub[:1], paris, None, "inconclusive"),  # ~340 km
            (["Tower Bridge"], stub[:1] + stub[2:], None, "GB",
             "inconclusive"),                                        # GB 1/2, JP 1/2 — tie
            (["Tower Bridge"], stub, None, "GB", "supports"),        # GB 2/3 majority
            (["Tower Bridge"], stub[:2], None, "GB", "supports"),    # GB 2/2 majority
            (["Tower Bridge"], stub[:2], None, "US", "contradicts"),
            ([], stub, london, None, "inconclusive"),                # no names
            (["Tower Bridge"], [], london, None, "inconclusive"),    # no places
            (["Tower Bridge"], stub, None, None, "inconclusive"),    # nothing to compare
        ]
        for names, places, anchor, cc, want in cases:
            ev = landmark_evidence(names, places, anchor, cc)
            check(f"anchor={anchor} cc={cc}", ev.verdict.value == want,
                  f"-> {ev.verdict.value} (want {want}): {ev.summary}")
        check("distances carried in data",
              landmark_evidence(["Tower Bridge"], stub, london, None)
              .data["distances_km"].get("Tower Bridge", 0) < 10)
    except ImportError:
        print("  (run as `python -m sundial.landmarks` to build Evidence objects)")

    if fails:
        sys.exit(1)

    # --- live demo -----------------------------------------------------------
    ap = argparse.ArgumentParser(
        description="Sundial landmark demo — consensus + geocode + evidence"
    )
    ap.add_argument("names", nargs="*",
                    help="landmark names to geocode (default: two samples)")
    ap.add_argument("--anchor", metavar="LAT,LON",
                    help="claimed location to distance-check against")
    ap.add_argument("--cc", metavar="CC", help="claimed country code (e.g. GB)")
    args = ap.parse_args()

    anchor = None
    if args.anchor:
        try:
            anchor = tuple(float(v) for v in args.anchor.split(",")[:2])
        except ValueError:
            print(f"ignoring malformed --anchor {args.anchor!r}")

    demo_names = args.names or ["Tower Bridge", "Brandenburg Gate"]
    try:
        from . import geo  # noqa: F401 — proves geo imports; geocode may still fail
        places = landmark_candidates_to_places(demo_names)
        if not places:
            # distinguish "offline/unreachable" from "no match" cheaply
            geo.geocode(demo_names[0])
    except Exception as e:
        print(f"live geocode skipped ({e}) — offline?")
        places = []
    if not places:
        print("no geocoded places — nothing more to do")
        sys.exit(0)
    print("places:", json.dumps(places, indent=2))
    try:
        ev = landmark_evidence(demo_names, places, anchor, args.cc)
        print(f"evidence[{ev.type}] {ev.verdict.value} "
              f"conf={ev.confidence}: {ev.summary}")
    except ImportError:
        print("(run as `python -m sundial.landmarks` to build Evidence objects)")
