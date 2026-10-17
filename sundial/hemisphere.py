"""Hemisphere tell — which side of the scene does the sun sit on?

OSINT idea: at solar noon the sun sits on the equator-side meridian —
due SOUTH for northern-hemisphere observers, due NORTH for southern —
and shadows point poleward. Read which side of the frame the sun (or
the light/shadow source) falls on and you get a weak hemisphere vote: a
scene whose sun behaves like the *wrong* hemisphere's is probably not
where the claim puts it.

``observed_sun_side`` is "left" | "right" | "center" — the side of
*frame* the sun occupies, as a caller might derive from the VLM
conditions ``sun.position`` x-coordinate vs the frame's vertical
midline, or from which way the shadows lean. It is read under the
sun-arc convention: the sun's daily arc is clockwise in the northern
hemisphere and counter-clockwise in the southern, so for a fixed camera
the sun — and the shadows it throws — sweeps toward frame-RIGHT in the
north and frame-LEFT in the south. For the motion reading this needs no
camera heading (a sun at relative bearing r drifts with dAz/dt, which
is hemisphere-signed); a static side read carries the same signature
under the usual camera-toward-the-light composition. Hence the fixed
mapping ``right -> northern``, ``left -> southern``.

Honesty limits: hemisphere alone is weak corroboration — confidence is
capped at 0.55 (spec: never above ~0.6). The lat-sign premise itself
can invert — when |declination| > |lat| the noon sun hangs poleward of
the observer, and off-meridian hours can put the sun in the "wrong" sky
half at any latitude — so the ephemeris check abstains (INCONCLUSIVE)
whenever the claimed instant's geometry can't carry the tell.

Stdlib top-level only; pysolar and .schema are imported lazily inside
functions so this module imports with zero third-party deps.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Frame side -> observed hemisphere (sun-arc convention, see module
# docstring). "center" carries no tell: an overhead or behind-camera sun
# says nothing about hemispheres.
_SIDE_TO_HEMISPHERE = {"left": "southern", "right": "northern"}

# Sky half the claimed hemisphere's noon sun should occupy — the premise
# the side tell tests. Northern-hemisphere noon sun sits in the southern
# sky, and vice versa.
_HEMISPHERE_NOON_SKY = {"northern": "south", "southern": "north"}

# Azimuths within this of due east (90) or due west (270) count as
# "edge" — a sun near the E-W line sits cleanly in neither sky half, so
# a frame side can't tag a hemisphere for it.
_EDGE_MARGIN_DEG = 20.0


def _aware(when: datetime) -> datetime:
    # Sundial works in UTC end to end; naive datetimes are read as UTC.
    # Aware datetimes are normalized to UTC (same convention as solar.py).
    return (when.astimezone(timezone.utc) if when.tzinfo is not None
            else when.replace(tzinfo=timezone.utc))


def _parse_instant(iso) -> datetime | None:
    """ISO-8601 -> aware UTC datetime, or None when unparseable."""
    try:
        s = str(iso).strip()
        if s.lower().endswith("z"):
            s = s[:-1] + "+00:00"  # fromisoformat <3.11 rejects "Z"
        return _aware(datetime.fromisoformat(s))
    except (ValueError, AttributeError, TypeError):
        return None


def _sky_side(azimuth_deg: float | None,
              margin_deg: float = _EDGE_MARGIN_DEG) -> str | None:
    """Sky half an azimuth sits in: "north" | "south" | "edge" | None.

    The halves meet on the east-west line; azimuths within `margin_deg`
    of due east or due west are "edge" — the sun is near neither pole of
    the sky, so a side observation can't feed the tell.
    """
    if azimuth_deg is None:
        return None
    a = azimuth_deg % 360.0
    for b in (90.0, 270.0):
        if abs((a - b + 180.0) % 360.0 - 180.0) <= margin_deg:
            return "edge"
    return "south" if 90.0 < a < 270.0 else "north"


def noon_azimuth_claim(lat: float, claimed_utc_iso: str) -> float | None:
    """Azimuth (deg CW from true N) the sun holds at its daily culmination.

    `claimed_utc_iso` supplies the claimed date; the culmination azimuth
    is longitude-free — the noon transit sits on the meridian: ~180° /
    due south for northern latitudes, ~0° / due north for southern — so
    a latitude alone anchors the hemisphere tell even when the claim's
    longitude is missing or the claimed instant isn't local noon.

    Scans the claimed UTC date at (lat, lon=0) for the elevation maximum
    and returns the azimuth there. -> None when the input won't parse,
    |lat| > 90, pysolar is missing, or the sun never rises on that date
    (polar night: no noon sun to take a side from).
    """
    try:
        lat = float(lat)
    except (TypeError, ValueError):
        return None
    if not -90.0 <= lat <= 90.0:
        return None
    claimed = _parse_instant(claimed_utc_iso)
    if claimed is None:
        return None
    try:
        from pysolar.solar import get_altitude, get_azimuth
    except ImportError:
        return None

    t0 = datetime(claimed.year, claimed.month, claimed.day,
                  tzinfo=timezone.utc)
    best, best_alt = t0, -90.0
    for i in range(24 * 6):  # 10-min steps across the claimed UTC day
        t = t0 + timedelta(minutes=10 * i)
        alt = get_altitude(lat, 0.0, t)
        if alt > best_alt:
            best, best_alt = t, alt
    if best_alt <= 0.0:
        return None  # polar night — the sun never culminates above horizon
    for i in range(-10, 11):  # 1-min refine around the coarse peak
        t = best + timedelta(minutes=i)
        alt = get_altitude(lat, 0.0, t)
        if alt > best_alt:
            best, best_alt = t, alt
    return get_azimuth(lat, 0.0, best) % 360.0


def hemisphere_evidence(
    lat: float | None,
    lon: float | None,
    claimed_utc_iso: str,
    observed_sun_side: str | None,
) -> "Evidence":
    """Evidence card for the hemisphere tell.

    `observed_sun_side` is "left" | "right" | "center" | None — the side
    of frame the sun/light source occupies (a caller may derive it from
    conditions ``sun.position`` x vs frame center, or from shadow
    direction). Frame-right reads as a northern-hemisphere sun arc,
    frame-left as southern (see module docstring); "center"/None carry
    no tell.

    Verdict: observed hemisphere conflicting with the hemisphere implied
    by the claimed lat sign -> CONTRADICTS ~0.55; agreeing -> SUPPORTS
    ~0.45; missing inputs, an equatorial claim, a below-horizon sun, or
    geometry that inverts the lat-sign premise (tropics / off-meridian /
    near E-W bearings) -> INCONCLUSIVE. Confidence never exceeds 0.55 —
    hemisphere alone is corroboration, not proof.
    """
    from .schema import ClaimVerdict, Evidence  # lazy: zero-dep module import

    claimed = _parse_instant(claimed_utc_iso)
    try:
        latf = float(lat)
        if not -90.0 <= latf <= 90.0:
            latf = None
    except (TypeError, ValueError):
        latf = None
    try:
        lonf = float(lon)
        if not -180.0 <= lonf <= 180.0:
            lonf = None
    except (TypeError, ValueError):
        lonf = None
    side = (str(observed_sun_side).strip().lower()
            if observed_sun_side is not None else None)
    if side == "centre":
        side = "center"
    if side not in ("left", "right", "center"):
        side = None  # unrecognized reads are treated as missing

    claimed_hemi = (None if latf is None else
                    "northern" if latf > 0 else
                    "southern" if latf < 0 else "equatorial")

    # --- ephemeris context ------------------------------------------------
    # Everything below is annotation + premise checking; a pysolar failure
    # degrades the card, never aborts it.
    noon_az = (noon_azimuth_claim(latf, claimed_utc_iso)
               if latf is not None and claimed is not None else None)
    expected = None
    if latf is not None and claimed is not None:
        try:
            from pysolar.solar import get_altitude, get_azimuth
            elon = lonf if lonf is not None else 0.0
            expected = {
                "azimuth_deg": round(get_azimuth(latf, elon, claimed) % 360.0, 2),
                "elevation_deg": round(get_altitude(latf, elon, claimed), 2),
                "lon_deg": elon,
                "lon_source": ("claim" if lonf is not None
                               else "prime meridian (lon missing)"),
            }
        except Exception:
            expected = None

    # Premise checks: the lat-sign tell only works when the sun actually
    # sits in the sky half the claimed hemisphere implies.
    sky_noon = _sky_side(noon_az)  # lon-free: noon-sun sky half for the date
    sky_inst = None
    if expected is not None and lonf is not None:
        sky_inst = ("below_horizon" if expected["elevation_deg"] <= 0.0
                    else _sky_side(expected["azimuth_deg"]))
    want_sky = _HEMISPHERE_NOON_SKY.get(claimed_hemi)  # None for equator/None
    obs_hemi = _SIDE_TO_HEMISPHERE.get(side) if side else None

    data = {
        "claim_time_utc": (claimed.isoformat() if claimed is not None
                           else str(claimed_utc_iso)),
        "lat_deg": latf,
        "lon_deg": lonf,
        "claimed_hemisphere": claimed_hemi,
        "noon_azimuth_deg": noon_az,
        "noon_sky_side": sky_noon,
        "expected_at_claim": expected,
        "claimed_instant_sky_side": sky_inst,
        "observed_sun_side": side,
        "observed_hemisphere": obs_hemi,
        "side_convention": ("sun/light on frame-left -> southern-hemisphere "
                            "arc, frame-right -> northern-hemisphere arc"),
        "tell_strength": "weak corroboration — capped below 0.6 by design",
    }

    def _card(summary, verdict, conf, extra=None):
        if extra:
            data.update(extra)
        return Evidence(type="hemisphere", summary=summary,
                        verdict=verdict, confidence=conf, data=data)

    if latf is None or claimed is None:
        return _card("hemisphere check needs a claimed latitude and a "
                     "parseable UTC claim time", ClaimVerdict.INCONCLUSIVE, 0.2)
    if side is None:
        return _card(f"no usable sun-side observation for the "
                     f"{claimed_hemi}-hemisphere claim (lat {latf:+.1f}°)",
                     ClaimVerdict.INCONCLUSIVE, 0.2)
    if side == "center":
        return _card("sun on/near frame center — an overhead or behind-camera "
                     "sun carries no hemisphere tell",
                     ClaimVerdict.INCONCLUSIVE, 0.25)
    if claimed_hemi == "equatorial":
        return _card("claimed latitude sits on the equator — no hemisphere "
                     "to corroborate", ClaimVerdict.INCONCLUSIVE, 0.25)
    if sky_inst == "below_horizon":
        return _card(f"claimed instant puts the sun below the horizon at "
                     f"({latf:.1f}, {lonf:.1f}) — no sun side to read",
                     ClaimVerdict.INCONCLUSIVE, 0.25)
    if sky_noon is not None and want_sky is not None and sky_noon != want_sky:
        return _card(f"on the claimed date the noon sun hangs in the "
                     f"{sky_noon}ern sky at lat {latf:+.1f}° (sub-solar "
                     f"geometry) — the lat-sign tell is inverted here",
                     ClaimVerdict.INCONCLUSIVE, 0.25,
                     {"premise_inverted": True})
    if sky_inst == "edge":
        return _card(f"sun bears {expected['azimuth_deg']:.0f}° at the "
                     "claimed instant — near the east-west line, a frame "
                     "side can't tag a hemisphere",
                     ClaimVerdict.INCONCLUSIVE, 0.25)
    if sky_inst in ("north", "south") and sky_inst != want_sky:
        return _card(f"at the claimed instant the sun sits in the "
                     f"{sky_inst}ern sky (az {expected['azimuth_deg']:.0f}°), "
                     f"not the {want_sky}ern sky a {claimed_hemi}-hemisphere "
                     "noon implies — off-meridian geometry makes the side "
                     "tell unsafe", ClaimVerdict.INCONCLUSIVE, 0.25,
                     {"premise_inverted": True})

    if obs_hemi == claimed_hemi:
        return _card(f"sun on the {side} of frame — reads {obs_hemi} "
                     f"hemisphere, consistent with claimed lat {latf:+.1f}° "
                     f"({claimed_hemi})", ClaimVerdict.SUPPORTS, 0.45)
    return _card(f"sun on the {side} of frame — reads {obs_hemi} hemisphere, "
                 f"but claimed lat {latf:+.1f}° is {claimed_hemi}",
                 ClaimVerdict.CONTRADICTS, 0.55)


if __name__ == "__main__":
    import sys

    from .schema import ClaimVerdict as _CV

    print("Hemisphere tell — London vs Sydney on the same June date\n")
    london_noon = "2025-06-15T12:00:00Z"   # solar noon ~12:00 UTC at lon≈0
    sydney_noon = "2025-06-15T02:00:00Z"   # solar noon ~12:00 AEST (UTC+10)

    for name, la, t in (("London", 51.5, london_noon),
                        ("Sydney", -33.9, sydney_noon)):
        az = noon_azimuth_claim(la, t)
        side = _sky_side(az)
        az_txt = f"{az:.0f}°" if az is not None else "n/a (polar night?)"
        print(f"{name:>7} lat {la:+.1f}: noon azimuth {az_txt} — sun "
              f"culminates in the {side}ern sky")

    print("\nevidence cards:")
    cards = [
        ("London noon, sun on frame-right",
         hemisphere_evidence(51.5, -0.13, london_noon, "right")),
        ("London noon, sun on frame-left",
         hemisphere_evidence(51.5, -0.13, london_noon, "left")),
        ("Sydney noon, sun on frame-left",
         hemisphere_evidence(-33.9, 151.2, sydney_noon, "left")),
        ("Sydney noon, sun on frame-right",
         hemisphere_evidence(-33.9, 151.2, sydney_noon, "right")),
        ("same instant, Sydney claim (local night there)",
         hemisphere_evidence(-33.9, 151.2, london_noon, "left")),
        ("London noon, no sun-side observed",
         hemisphere_evidence(51.5, -0.13, london_noon, None)),
        ("no latitude claimed",
         hemisphere_evidence(None, -0.13, london_noon, "right")),
    ]
    for label, ev in cards:
        print(f"  {ev.verdict.value:>12} conf={ev.confidence:.2f}  "
              f"{label}\n      {ev.summary}")

    print("\nself-test:")
    checks = [
        (hemisphere_evidence(51.5, -0.13, london_noon, "right"), _CV.SUPPORTS),
        (hemisphere_evidence(51.5, -0.13, london_noon, "left"), _CV.CONTRADICTS),
        (hemisphere_evidence(-33.9, 151.2, sydney_noon, "left"), _CV.SUPPORTS),
        (hemisphere_evidence(-33.9, 151.2, sydney_noon, "right"), _CV.CONTRADICTS),
        (hemisphere_evidence(-33.9, 151.2, london_noon, "left"), _CV.INCONCLUSIVE),
        (hemisphere_evidence(51.5, -0.13, london_noon, "center"), _CV.INCONCLUSIVE),
        (hemisphere_evidence(51.5, -0.13, london_noon, None), _CV.INCONCLUSIVE),
        (hemisphere_evidence(None, -0.13, london_noon, "right"), _CV.INCONCLUSIVE),
        (hemisphere_evidence(0.0, 0.0, london_noon, "right"), _CV.INCONCLUSIVE),
    ]
    ok = True
    for ev, want in checks:
        good = ev.verdict == want and ev.confidence <= 0.6
        ok &= good
        print(f"  [{ev.verdict.value:>12}] conf={ev.confidence:.2f} "
              f"want {want.value:>12}  {'ok' if good else 'MISMATCH'}")
    # Culmination anchors: London's noon sun due south, Sydney's due north.
    la, sa = noon_azimuth_claim(51.5, london_noon), noon_azimuth_claim(-33.9, sydney_noon)
    for az, lo, hi, tag in ((la, 160.0, 200.0, "London noon ~due S"),
                            (sa, 340.0, 20.0, "Sydney noon ~due N")):
        good = az is not None and (lo <= az <= hi if lo < hi else az >= lo or az <= hi)
        ok &= good
        print(f"  {tag}: az={az if az is not None else 'None'}  "
              f"{'ok' if good else 'MISMATCH'}")
    if not ok:
        sys.exit(1)
    print("all checks passed")
