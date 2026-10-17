"""Tests for guess mode (no-claim geolocation), GeoCLIP prior, locate_summary
priority order, and report unavailable-card collapsing.

Run directly: python3 -m pytest tests/test_guess_mode.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sundial import geo_clip, locate_summary  # noqa: E402
from sundial.report import render_html  # noqa: E402
from sundial.schema import (  # noqa: E402
    Adjudication,
    Claim,
    ClaimVerdict,
    Evidence,
    EvidencePack,
    Verdict,
)


# ---------------------------------------------------------------------------
# 1. sundial/geo_clip.py
# ---------------------------------------------------------------------------

def test_geoclip_available_returns_bool():
    res = geo_clip.available()
    assert isinstance(res, bool)


def test_predict_coords_empty_and_invalid_inputs(monkeypatch):
    assert geo_clip.predict_coords([]) == []
    assert geo_clip.predict_coords(None) == []
    assert geo_clip.predict_coords(["", None, 123, {}]) == []

    # Even with a working model, invalid frame entries or model errors never raise
    class _BoomModel:
        def predict(self, path, top_k=5):
            raise RuntimeError("corrupt image")

    monkeypatch.setattr(geo_clip, "_model", lambda: _BoomModel())
    assert geo_clip.predict_coords(["/tmp/nonexistent.jpg"]) == []
    assert geo_clip.predict_coords(None) == []

    # Model returning malformed tuples / out-of-range coordinates degrades cleanly
    class _WeirdModel:
        def predict(self, path, top_k=5):
            return [(999.0, 0.0), ("bad", "coords"), (None, 10.0)], ["not_a_float", 0.5, 0.2]

    monkeypatch.setattr(geo_clip, "_model", lambda: _WeirdModel())
    assert geo_clip.predict_coords(["/tmp/frame1.jpg"]) == []


def test_predict_coords_clusters_and_ranks_by_probability(monkeypatch):
    # Frame 1 & Frame 2 are in London (~2 km apart, within 150 km cluster radius):
    # combined prob = 0.35 + 0.40 = 0.75, frames = 2.
    # Frame 3 is in Tokyo with prob 0.60, frames = 1.
    # London cluster (0.75) should outrank Tokyo (0.60).
    frame_outputs = {
        "f1.jpg": ([(51.5000, -0.1000), (48.8566, 2.3522)], [0.35, 0.15]),
        "f2.jpg": ([(51.5200, -0.1200), (40.4168, -3.7038)], [0.40, 0.10]),
        "f3.jpg": ([(35.6762, 139.6503), (34.6937, 135.5023)], [0.60, 0.20]),
    }

    class _FakeGeoCLIP:
        def predict(self, path, top_k=5):
            return frame_outputs[path]

    monkeypatch.setattr(geo_clip, "_model", lambda: _FakeGeoCLIP())
    clusters = geo_clip.predict_coords(["f1.jpg", "f2.jpg", "f3.jpg"], top_k=2)

    assert len(clusters) == 2
    best, second = clusters[0], clusters[1]

    # Best cluster is the pooled London cluster
    assert best["frames"] == 2
    assert best["prob"] == pytest.approx(0.75)
    expected_lat = (51.5000 * 0.35 + 51.5200 * 0.40) / 0.75
    expected_lon = (-0.1000 * 0.35 + -0.1200 * 0.40) / 0.75
    assert best["lat"] == pytest.approx(expected_lat, abs=1e-4)
    assert best["lon"] == pytest.approx(expected_lon, abs=1e-4)

    # Second cluster is Tokyo
    assert second["frames"] == 1
    assert second["prob"] == pytest.approx(0.60)
    assert second["lat"] == pytest.approx(35.6762, abs=1e-4)
    assert second["lon"] == pytest.approx(139.6503, abs=1e-4)


def test_pred_summary_shape_and_values():
    assert geo_clip.pred_summary([]) is None
    assert geo_clip.pred_summary(None) is None

    # Single tight cluster -> minimum 80 km radius
    single = [{"lat": 51.5074123, "lon": -0.1277583, "prob": 0.72345, "frames": 3}]
    s1 = geo_clip.pred_summary(single)
    assert s1 is not None
    assert set(s1.keys()) == {"lat", "lon", "r_km", "prob", "frames"}
    assert s1["lat"] == pytest.approx(51.50741, abs=1e-5)
    assert s1["lon"] == pytest.approx(-0.12776, abs=1e-5)
    assert s1["r_km"] == pytest.approx(80.0)
    assert s1["prob"] == pytest.approx(0.7235, abs=1e-4)
    assert s1["frames"] == 3

    # Scattered clusters -> spread increases r_km beyond 80 km (capped at 1500 km)
    scattered = [
        {"lat": 51.5, "lon": -0.1, "prob": 0.6, "frames": 2},
        {"lat": 48.85, "lon": 2.35, "prob": 0.4, "frames": 1},
    ]
    s2 = geo_clip.pred_summary(scattered)
    assert s2 is not None
    assert set(s2.keys()) == {"lat", "lon", "r_km", "prob", "frames"}
    assert 80.0 < s2["r_km"] <= 1500.0
    assert s2["frames"] == 2


def test_geoclip_evidence_verdicts():
    rows = [
        {"lat": 51.5074, "lon": -0.1278, "prob": 0.68, "frames": 2},
        {"lat": 48.8566, "lon": 2.3522, "prob": 0.22, "frames": 1},
    ]

    # Empty rows -> None
    assert geo_clip.geoclip_evidence([], anchor=(51.5, -0.1)) is None

    # Within 250 km of anchor -> SUPPORTS
    ev_sup = geo_clip.geoclip_evidence(rows, anchor=(51.52, -0.10), claimed_cc="GB")
    assert ev_sup is not None
    assert ev_sup.type == "geo_clip"
    assert ev_sup.verdict == ClaimVerdict.SUPPORTS
    assert ev_sup.confidence == pytest.approx(0.55)
    assert ev_sup.data["anchor_dist_km"] <= 250.0
    assert len(ev_sup.data["candidates"]) == 2

    # >= 1500 km from anchor (e.g. Tokyo vs London) -> CONTRADICTS
    ev_con = geo_clip.geoclip_evidence(rows, anchor=(35.6762, 139.6503), claimed_cc="JP")
    assert ev_con is not None
    assert ev_con.verdict == ClaimVerdict.CONTRADICTS
    assert ev_con.confidence == pytest.approx(0.55)
    assert ev_con.data["anchor_dist_km"] >= 1500.0

    # Between 250 km and 1500 km (e.g. Berlin ~930 km from London) -> INCONCLUSIVE
    ev_mid = geo_clip.geoclip_evidence(rows, anchor={"lat": 52.5200, "lon": 13.4050})
    assert ev_mid is not None
    assert ev_mid.verdict == ClaimVerdict.INCONCLUSIVE
    assert ev_mid.confidence == pytest.approx(0.35)
    assert 250.0 < ev_mid.data["anchor_dist_km"] < 1500.0

    # No anchor (guess mode) -> INCONCLUSIVE, still populates coordinates & candidates
    ev_guess = geo_clip.geoclip_evidence(rows, anchor=None)
    assert ev_guess is not None
    assert ev_guess.verdict == ClaimVerdict.INCONCLUSIVE
    assert ev_guess.confidence == pytest.approx(0.35)
    assert "anchor_dist_km" not in ev_guess.data
    assert ev_guess.data["lat"] == pytest.approx(51.5074, abs=1e-4)
    assert ev_guess.data["lon"] == pytest.approx(-0.1278, abs=1e-4)
    assert "2 frames agreeing" in ev_guess.summary


# ---------------------------------------------------------------------------
# 2. sundial/locate_summary.py
# ---------------------------------------------------------------------------

def test_locate_summary_priority_order():
    osint_de = {
        "type": "osint",
        "data": {
            "ranked": [
                {"cc": "DE", "score": 0.80, "hits": 3},
                {"cc": "AT", "score": 0.30, "hits": 1},
            ]
        },
    }
    geoclip_paris = {
        "type": "geo_clip",
        "data": {
            "lat": 48.8566,
            "lon": 2.3522,
            "r_km": 95.0,
            "prob": 0.65,
            "candidates": [
                {"lat": 48.8566, "lon": 2.3522, "prob": 0.65},
                {"lat": 45.7640, "lon": 4.8357, "prob": 0.20},
            ],
        },
    }
    landmark_rome = {
        "type": "landmark",
        "data": {
            "places": [
                {"name": "Colosseum", "lat": 41.8902, "lon": 12.4922, "cc": "IT", "score": 0.92}
            ]
        },
    }
    berlin_anchor = (52.5200, 13.4050)  # Inside DE

    # (a) 4th priority: Country-centroid blend when only ranked countries exist
    s_blend = locate_summary.summarize([osint_de], anchor=None, claim_place=None)
    assert s_blend["predicted"] is not None
    assert s_blend["best_cc"] == "DE"
    assert s_blend["best_name"] == "Germany"
    assert s_blend["predicted"]["radius_km"] >= 300

    # (b) 3rd priority: GeoCLIP model prediction beats country-centroid blend when no claim / uncorroborated
    s_gclip = locate_summary.summarize([osint_de, geoclip_paris], anchor=None, claim_place=None)
    assert s_gclip["predicted"]["lat"] == pytest.approx(48.8566, abs=1e-4)
    assert s_gclip["predicted"]["lon"] == pytest.approx(2.3522, abs=1e-4)
    assert s_gclip["predicted"]["radius_km"] == 95
    assert s_gclip["best_cc"] == "FR"
    assert s_gclip["best_name"] == "France"

    # (c) 2nd priority: Corroborated claim (anchor in DE + top ranked cc is DE) beats GeoCLIP
    s_corr = locate_summary.summarize(
        [osint_de, geoclip_paris], anchor=berlin_anchor, claim_place="Berlin"
    )
    assert s_corr["predicted"]["lat"] == pytest.approx(52.5200, abs=1e-4)
    assert s_corr["predicted"]["lon"] == pytest.approx(13.4050, abs=1e-4)
    assert s_corr["predicted"]["r_km"] == pytest.approx(60.0)
    assert s_corr["best_name"] == "Berlin"

    # (d) 1st priority: Named landmark beats corroborated claim, GeoCLIP, and country blend
    s_lm = locate_summary.summarize(
        [osint_de, geoclip_paris, landmark_rome], anchor=berlin_anchor, claim_place="Berlin"
    )
    assert s_lm["predicted"]["lat"] == pytest.approx(41.8902, abs=1e-4)
    assert s_lm["predicted"]["lon"] == pytest.approx(12.4922, abs=1e-4)
    assert s_lm["predicted"]["r_km"] == pytest.approx(25.0)
    assert s_lm["best_cc"] == "IT"
    assert "Colosseum" in s_lm["best_name"]


def test_contradicted_claim_does_not_block_geoclip_prediction():
    # Claim is Tokyo, Japan (35.6762, 139.6503), but cues rank UK (GB) #1
    # and GeoCLIP predicts London (51.5074, -0.1278).
    # Because anchor country (JP) != top ranked country (GB), the claim is NOT corroborated,
    # so GeoCLIP must win the predicted pin rather than being blocked by the claim anchor.
    evs = [
        {
            "type": "osint",
            "data": {"ranked": [{"cc": "GB", "score": 0.78, "hits": 3}]},
        },
        {
            "type": "geo_clip",
            "data": {
                "lat": 51.5074,
                "lon": -0.1278,
                "r_km": 85.0,
                "prob": 0.70,
                "candidates": [{"lat": 51.5074, "lon": -0.1278, "prob": 0.70}],
            },
        },
    ]
    summary = locate_summary.summarize(
        evs, anchor=(35.6762, 139.6503), claim_place="Tokyo"
    )
    assert summary["anchor"]["lat"] == pytest.approx(35.6762, abs=1e-4)
    assert summary["predicted"] is not None
    assert summary["predicted"]["lat"] == pytest.approx(51.5074, abs=1e-4)
    assert summary["predicted"]["lon"] == pytest.approx(-0.1278, abs=1e-4)
    assert summary["predicted"]["radius_km"] == 85
    assert summary["best_cc"] == "GB"
    assert summary["best_name"] == "United Kingdom"


def test_geoclip_runner_up_markers_without_duplicating_gps_fix():
    evs = [
        {
            "type": "geo_clip",
            "data": {
                "lat": 51.5074,
                "lon": -0.1278,
                "r_km": 90.0,
                "prob": 0.64,
                "candidates": [
                    {"lat": 51.5074, "lon": -0.1278, "prob": 0.64},
                    {"lat": 53.4808, "lon": -2.2426, "prob": 0.21},
                    {"lat": 55.9533, "lon": -3.1883, "prob": 0.10},
                ],
            },
        }
    ]
    summary = locate_summary.summarize(evs, anchor=None, claim_place=None)
    markers = summary["markers"]

    # Main coordinate appears as the "pred" marker
    pred_markers = [m for m in markers if m["kind"] == "pred"]
    assert len(pred_markers) == 1
    assert pred_markers[0]["lat"] == pytest.approx(51.5074, abs=1e-4)
    assert pred_markers[0]["lon"] == pytest.approx(-0.1278, abs=1e-4)

    # Runner-up candidates (candidates[1:]) appear as "other possible spot" geo markers
    alt_markers = [m for m in markers if "other possible spot" in m["label"]]
    assert len(alt_markers) == 2
    assert alt_markers[0]["lat"] == pytest.approx(53.4808, abs=1e-4)
    assert alt_markers[0]["lon"] == pytest.approx(-2.2426, abs=1e-4)
    assert "21%" in alt_markers[0]["label"]
    assert alt_markers[1]["lat"] == pytest.approx(55.9533, abs=1e-4)
    assert alt_markers[1]["lon"] == pytest.approx(-3.1883, abs=1e-4)
    assert "10%" in alt_markers[1]["label"]

    # No "gps fix" marker is emitted for the geo_clip card, and the main coord
    # is not duplicated as a "geo" kind marker
    assert not any(m["label"] == "gps fix" for m in markers)
    assert not any(
        m["kind"] == "geo"
        and abs(m["lat"] - 51.5074) < 1e-3
        and abs(m["lon"] - (-0.1278)) < 1e-3
        for m in markers
    )


# ---------------------------------------------------------------------------
# 3. sundial/report.py
# ---------------------------------------------------------------------------

def _render_doc(adj: Adjudication) -> str:
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "report.html")
        render_html(adj, out)
        return Path(out).read_text(encoding="utf-8")


def test_report_no_claim_guess_mode_copy():
    pack = EvidencePack(
        clip="/data/clips/mystery.mp4",
        claim=Claim(claimed_time=None, claimed_place=None, raw=None),
        evidence=[
            Evidence(
                type="geo_clip",
                summary="GeoCLIP pins 51.507, -0.128 (~80 km radius, 2 frames agreeing)",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.35,
            )
        ],
    )
    adj = Adjudication(
        verdict=Verdict.INSUFFICIENT,
        confidence=0.35,
        narrative="Best estimate is London, United Kingdom.",
        pack=pack,
    )
    doc = _render_doc(adj)

    assert "<h1>The task</h1>" in doc
    assert "work out where this was filmed" in doc
    assert "no claim to check — best guess is on the map" in doc
    assert "<h1>The claim</h1>" not in doc


def test_report_collapses_unavailable_evidence_cards():
    evs = [
        Evidence(
            type="weather",
            summary="weather check skipped — no claimed time",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
        ),
        Evidence(
            type="audio",
            summary="audio model unavailable on this machine",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
        ),
        Evidence(
            type="overpass",
            summary="Overpass lookup failed (offline)",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
        ),
        Evidence(
            type="solar",
            summary="sun elevation matches claimed afternoon hour",
            verdict=ClaimVerdict.SUPPORTS,
            confidence=0.82,
        ),
    ]
    pack = EvidencePack(
        clip="/data/clips/sample.mp4",
        claim=Claim(claimed_time="2025-06-15T14:30:00Z", claimed_place="London", raw="London"),
        evidence=evs,
    )
    adj = Adjudication(
        verdict=Verdict.SUPPORTED,
        confidence=0.82,
        narrative="Supported by solar geometry.",
        pack=pack,
    )
    doc = _render_doc(adj)
    body = doc.split("Technical details")[0]

    # All 3 unavailable cards collapse into a single "Couldn't check" card
    assert body.count("Couldn&#x27;t check") == 1
    assert "couldn&#x27;t check 3 things" in body
    assert "Weather records" in body
    assert "Audio clues" in body
    assert "Map data check" in body

    # Individual dead-card summaries are replaced in the main card grid
    assert "weather check skipped — no claimed time" not in body
    assert "audio model unavailable on this machine" not in body
    assert "Overpass lookup failed (offline)" not in body

    # Real evidence card remains intact
    assert "sun elevation matches claimed afternoon hour" in body
