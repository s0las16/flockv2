"""Azimuth bridge — map an image-space shadow direction onto world azimuths.

Convention (matches perception.shadow_vector): ``image_angle_deg`` is the
direction the shadow points, measured clockwise from image-up — 0 = toward
the top edge / camera forward over the ground plane, 90 = right of frame,
180 = down-image, 270 = left. Under a flat-ground, level-camera assumption:

    world_azimuth = camera_heading + image_angle        (mod 360)
    camera_heading = shadow_world_azimuth - image_angle (mod 360)

(If a source measures angle from image-down instead, add 180 before calling.)

World azimuths are degrees clockwise from true north (0=N, 90=E, 180=S,
270=W), same as pysolar's get_azimuth. pysolar / .solar are imported lazily
inside functions so this module imports cleanly with zero third-party deps.
"""
from __future__ import annotations

import math
from datetime import date as _date, datetime, timedelta, timezone

# Same scan cadence and merge style as solar.infer_time_window. Duplicated
# (not imported) so this module stays importable without pysolar.
SCAN_STEP = timedelta(minutes=2)

# When the camera heading is unknown we test two facing hypotheses — a
# north-facing and a south-facing camera — rather than guessing.
HEADING_HYPOTHESES = (0.0, 180.0)

# A window is kept only while the implied heading stays within ±10° of its
# hypothesis: heading is a fixed property of the camera, so drift beyond
# that means the hypothesis can't explain the observation.
HEADING_TOL_DEG = 10.0


def azimuth_error(a: float, b: float) -> float:
    """Shortest distance between two azimuths in degrees (0/360 wrap-safe)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _circular_mean(angles: list[float], weights: list[float] | None = None) -> float:
    """Mean direction of azimuths in degrees (wrap-safe)."""
    w = weights if weights is not None else [1.0] * len(angles)
    y = sum(wi * math.sin(math.radians(a)) for a, wi in zip(angles, w))
    x = sum(wi * math.cos(math.radians(a)) for a, wi in zip(angles, w))
    return math.degrees(math.atan2(y, x)) % 360.0


def shadow_world_azimuths(
    image_angle_deg: float,
    camera_heading_deg: float | None = None,
) -> list[float]:
    """Candidate world azimuths the shadow points along (deg from true north).

    With a known camera heading this is the single azimuth
    ``heading + image_angle``. With heading unknown, return both facing
    hypotheses (north-facing vs south-facing camera) rather than guessing.
    """
    if camera_heading_deg is not None:
        return [(camera_heading_deg + image_angle_deg) % 360.0]
    return [(h + image_angle_deg) % 360.0 for h in HEADING_HYPOTHESES]


def consistent_azimuth_windows(
    lat: float,
    lon: float,
    day: _date | datetime | str,
    image_angle_deg: float,
    heading_tol_deg: float = HEADING_TOL_DEG,
    min_elevation_deg: float = 0.0,
) -> list[dict]:
    """Times of day when a fixed camera heading explains the image angle.

    Scans the UTC day in 2-min steps (solar.infer_time_window style). At each
    step the sun's azimuth fixes the required world shadow azimuth
    ``sun_az + 180``; the camera heading that would reproduce the observed
    image angle is ``implied = shadow_world_az - image_angle``. For each
    heading hypothesis (north-facing vs south-facing) contiguous matches are
    merged into windows where the implied heading stays within
    ±``heading_tol_deg`` of the hypothesis.

    Returns a list sorted by window start::

        {"window": {"start","end","center","minutes"},
         "implied_camera_heading_deg": float}

    Only times with the sun above ``min_elevation_deg`` are considered — a
    cast shadow needs daylight.
    """
    from pysolar.solar import get_altitude, get_azimuth

    if isinstance(day, datetime):
        d = day.date()
    elif isinstance(day, str):
        d = _date.fromisoformat(day)
    else:
        d = day

    t0 = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    steps = int(24 * 60 / (SCAN_STEP.total_seconds() / 60))

    # hypothesis -> [(time, implied_heading_at_time)]
    hits: dict[float, list[tuple[datetime, float]]] = {h: [] for h in HEADING_HYPOTHESES}
    for i in range(steps):
        t = t0 + i * SCAN_STEP
        if get_altitude(lat, lon, t) <= min_elevation_deg:
            continue
        shadow_world_az = (get_azimuth(lat, lon, t) + 180.0) % 360.0
        implied = (shadow_world_az - image_angle_deg) % 360.0
        for h in HEADING_HYPOTHESES:
            if azimuth_error(implied, h) <= heading_tol_deg:
                hits[h].append((t, implied))

    # Merge contiguous matches into windows (same _first/_last style as
    # solar.infer_time_window); each window reports its mean implied heading.
    out: list[dict] = []
    for h in HEADING_HYPOTHESES:
        groups: list[dict] = []
        for t, implied in hits[h]:
            if groups and t - groups[-1]["_last"] == SCAN_STEP:
                groups[-1]["_last"] = t
                groups[-1]["_implied"].append(implied)
            else:
                groups.append({"_first": t, "_last": t, "_implied": [implied]})
        for g in groups:
            start, end = g["_first"], g["_last"] + SCAN_STEP  # end = first non-match
            out.append({
                "window": {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "center": (start + (end - start) / 2).isoformat(),
                    "minutes": int((end - start).total_seconds() // 60),
                },
                "implied_camera_heading_deg": round(_circular_mean(g["_implied"]), 1) % 360.0,
            })
    out.sort(key=lambda r: r["window"]["start"])
    return out


def heading_hint(
    lat: float,
    lon: float,
    day: _date | datetime | str,
    image_angle_deg: float,
    heading_tol_deg: float = HEADING_TOL_DEG,
) -> float | None:
    """Modal implied camera heading across consistent windows, or None.

    Convenience for elevation-only callers that still want to annotate
    evidence data with a best-guess heading: windows are clustered by which
    hypothesis their implied heading falls nearest, and the minute-weighted
    circular mean of the dominant cluster is returned.
    """
    windows = consistent_azimuth_windows(lat, lon, day, image_angle_deg, heading_tol_deg)
    if not windows:
        return None
    clusters: dict[float, list[tuple[float, float]]] = {h: [] for h in HEADING_HYPOTHESES}
    for r in windows:
        implied, minutes = r["implied_camera_heading_deg"], r["window"]["minutes"]
        nearest = min(HEADING_HYPOTHESES, key=lambda h: azimuth_error(implied, h))
        clusters[nearest].append((implied, float(minutes)))
    dominant = max(clusters, key=lambda h: sum(m for _, m in clusters[h]))
    angles, weights = zip(*clusters[dominant])
    return round(_circular_mean(list(angles), list(weights)), 1) % 360.0


def _self_check(
    lat: float,
    lon: float,
    image_angle_deg: float,
    results: list[dict],
    heading_tol_deg: float = HEADING_TOL_DEG,
) -> bool:
    """Verify windows against solar.sun_position at their centers.

    For each window the world shadow azimuth predicted from the implied
    heading (``implied + image_angle``) must match ``sun_az + 180``, and the
    implied heading must sit within tolerance of a hypothesis.
    """
    from .solar import sun_position

    ok = True
    print("\nself-check (sun_position at window centers):")
    for r in results:
        c = datetime.fromisoformat(r["window"]["center"])
        sun_az = sun_position(lat, lon, c)["azimuth_deg"]
        implied = r["implied_camera_heading_deg"]
        predicted_shadow = (implied + image_angle_deg) % 360.0
        required_shadow = (sun_az + 180.0) % 360.0
        err = azimuth_error(predicted_shadow, required_shadow)
        hyp_err = min(azimuth_error(implied, h) for h in HEADING_HYPOTHESES)
        good = err <= 2.0 and hyp_err <= heading_tol_deg
        ok = ok and good
        print(f"  {r['window']['start'][11:16]}–{r['window']['end'][11:16]} UTC "
              f"implied={implied:6.1f}°  sun_az={sun_az:6.1f}°  "
              f"shadow err={err:4.1f}°  hyp err={hyp_err:4.1f}°  "
              f"{'PASS' if good else 'FAIL'}")
    return ok


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m sundial.azimuth",
        description="Find UTC windows where a fixed camera heading explains "
                    "an image-space shadow angle.",
    )
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    p.add_argument("--angle", type=float, required=True,
                   help="image-space shadow angle, deg clockwise from image-up")
    args = p.parse_args()

    cands = shadow_world_azimuths(args.angle)
    print(f"image angle {args.angle:.0f}° → world shadow azimuth candidates "
          f"(heading unknown): N-facing {cands[0]:.0f}°, S-facing {cands[1]:.0f}°")

    results = consistent_azimuth_windows(args.lat, args.lon, args.date, args.angle)
    print(f"\n{len(results)} consistent window(s) on {args.date} "
          f"at ({args.lat}, {args.lon}):")
    for r in results:
        w = r["window"]
        print(f"  {w['start'][11:16]}–{w['end'][11:16]} UTC "
              f"({w['minutes']:>3} min)  implied camera heading "
              f"{r['implied_camera_heading_deg']:.1f}°")

    hint = heading_hint(args.lat, args.lon, args.date, args.angle)
    print(f"\nheading hint: {hint if hint is not None else 'none'}")

    ok = _self_check(args.lat, args.lon, args.angle, results)
    print(f"\nself-check: {'all PASS' if ok else 'FAILURES above'}")


if __name__ == "__main__":
    main()
