"""Tests for sundial.pipeline unit helpers — run directly: python3 tests/test_pipeline_units.py

Pure functions only: geocoding and weather lookups are monkeypatched, so
nothing here touches the network or the VLM.
"""
import math
import sys
import traceback
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sundial import geo, pipeline, weather  # noqa: E402
from sundial.schema import GeoCandidate  # noqa: E402

# Shadow runs 60 px straight down the image (+y down); the matched person
# bbox is 100 px tall, so uncorrected elev = atan2(100, 60) ~ 59.04 deg.
OBS = {
    "object_class": "person",
    "base_point": [100, 200],
    "shadow_tip": [100, 260],
    "camera_tilt_deg": 0,
}
DETS = [{"class": "person", "bbox_xyxy": [90, 100, 110, 200]}]


# ------------------------------ _parse_latlon -----------------------------

def test_parse_latlon_valid():
    assert pipeline._parse_latlon("51.5,-0.12") == (51.5, -0.12)
    assert pipeline._parse_latlon(" 51.5 , -0.12 ") == (51.5, -0.12)


def test_parse_latlon_rejects_non_coords():
    for bad in ("London", "", "garbage", None, "51.5", "91.0,0", "0,181"):
        assert pipeline._parse_latlon(bad) is None, bad


# ---------------------------- _resolve_coords -----------------------------

def test_resolve_coords_given_in_claim():
    with mock.patch.object(geo, "geocode") as mg:
        lat, lon, note = pipeline._resolve_coords("51.5,-0.12")
    mg.assert_not_called()
    assert (lat, lon) == (51.5, -0.12)
    assert note == "coordinates given in claim"


def test_resolve_coords_geocodes_place_name():
    cand = GeoCandidate(
        name="London", lat=51.5074, lon=-0.1278,
        country_code="GB", display_name="London, England, GB",
    )
    with mock.patch.object(geo, "geocode", return_value=[cand]) as mg:
        lat, lon, note = pipeline._resolve_coords("London")
    assert (lat, lon) == (51.5074, -0.1278)
    assert "London, England, GB" in note
    mg.assert_called_once_with("London")


def test_resolve_coords_geocode_miss():
    with mock.patch.object(geo, "geocode", return_value=[]):
        lat, lon, note = pipeline._resolve_coords("Atlantis")
    assert lat is None and lon is None
    assert "could not geocode" in note and "Atlantis" in note


def test_resolve_coords_no_place():
    for empty in (None, ""):
        lat, lon, note = pipeline._resolve_coords(empty)
        assert (lat, lon) == (None, None)
        assert note == "no claimed place"


# ---------------------------- _norm_conditions ----------------------------

def test_norm_conditions_keeps_enum_values():
    out = pipeline._norm_conditions({"sky": "partly", "precipitation": "snow"})
    assert out == {"sky": "partly", "precipitation": "snow"}


def test_norm_conditions_drops_off_enum():
    out = pipeline._norm_conditions({
        "sky": "tornado",          # not in (clear, partly, overcast)
        "precipitation": "hail",   # not in (none, rain, snow)
        "ground_wetness": "muddy",  # neither "wet" nor "dry" substring
        "lighting": "neon",        # not a normalized key at all
    })
    assert out == {}


def test_norm_conditions_ground_wetness():
    n = pipeline._norm_conditions
    assert n({"ground_wetness": "wet"}) == {"wet_ground": True}
    assert n({"ground_wetness": "Wet pavement"}) == {"wet_ground": True}
    assert n({"ground_wetness": "dry"}) == {"wet_ground": False}
    assert n({"ground_wetness": True}) == {"wet_ground": True}
    assert n({"ground_wetness": False}) == {"wet_ground": False}
    assert n({}) == {}


# --------------------------- _sun_elevation_deg ---------------------------

def test_sun_elevation_flat_camera():
    elev = pipeline._sun_elevation_deg(dict(OBS), DETS)
    assert elev is not None
    assert abs(elev - math.degrees(math.atan2(100, 60))) < 1e-9
    assert abs(elev - 59.0) < 0.1


def test_sun_elevation_tilted_camera_expands_shadow():
    # tilt 30 deg: image dy foreshortens by sin(30)=0.5, so the shadow is
    # measured as 60/0.5 = 120 px -> elev = atan2(100, 120) ~ 39.8 deg.
    elev = pipeline._sun_elevation_deg({**OBS, "camera_tilt_deg": 30}, DETS)
    assert abs(elev - math.degrees(math.atan2(100, 120))) < 1e-9
    assert abs(elev - 39.8) < 0.05


def test_sun_elevation_tilt_threshold_and_bad_tilt():
    flat = math.degrees(math.atan2(100, 60))
    for tilt in (None, 8, "garbage"):  # missing, at threshold, unparseable
        obs = {k: v for k, v in OBS.items() if k != "camera_tilt_deg"}
        if tilt is not None:
            obs["camera_tilt_deg"] = tilt
        elev = pipeline._sun_elevation_deg(obs, DETS)
        assert abs(elev - flat) < 1e-9, tilt


def test_sun_elevation_no_matching_detection():
    assert pipeline._sun_elevation_deg(dict(OBS), []) is None
    assert pipeline._sun_elevation_deg(dict(OBS), None) is None
    truck = [{"class": "truck", "bbox_xyxy": [90, 100, 110, 200]}]
    assert pipeline._sun_elevation_deg(dict(OBS), truck) is None
    # matching class but degenerate bbox height (h <= 0)
    flat = [{"class": "person", "bbox_xyxy": [90, 100, 110, 100]}]
    assert pipeline._sun_elevation_deg(dict(OBS), flat) is None


def test_sun_elevation_bad_points():
    assert pipeline._sun_elevation_deg({"object_class": "person"}, DETS) is None
    zero = {"object_class": "person",
            "base_point": [100, 200], "shadow_tip": [100, 200]}
    assert pipeline._sun_elevation_deg(zero, DETS) is None


# ------------------------------ _claimed_utc ------------------------------

def test_claimed_utc_uses_archive_offset():
    with mock.patch.object(
        weather, "historical_weather",
        return_value={"utc_offset_seconds": 3600},
    ) as mw:
        utc_iso, note = pipeline._claimed_utc(51.5, -0.12, "2025-10-01T12:00")
    # local 12:00 with +3600s offset -> 11:00 UTC
    assert utc_iso == "2025-10-01T11:00:00+00:00"
    assert "3600" in note
    mw.assert_called_once_with(51.5, -0.12, "2025-10-01")


def test_claimed_utc_lookup_failure_treats_as_utc():
    with mock.patch.object(
        weather, "historical_weather", side_effect=RuntimeError("offline"),
    ):
        utc_iso, note = pipeline._claimed_utc(51.5, -0.12, "2025-10-01T12:00")
    assert utc_iso == "2025-10-01T12:00:00+00:00"
    assert "treated as UTC" in note


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
