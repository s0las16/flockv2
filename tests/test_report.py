"""Tests for sundial.report — run directly: python3 tests/test_report.py"""
import re
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sundial.report import render_html  # noqa: E402
from sundial.schema import (  # noqa: E402
    Adjudication,
    Claim,
    ClaimVerdict,
    Evidence,
    EvidencePack,
    Verdict,
)


def _adj(verdict=Verdict.CONTRADICTED, confidence=0.83,
         narrative="The shadows disagree with the clock.",
         evidence=None, claim=None):
    """Fabricated adjudication — schema objects only, no pipeline/LLM."""
    pack = EvidencePack(
        clip="/data/clips/interview.mp4",
        claim=claim or Claim(
            claimed_time="2025-06-15T14:30:00+00:00",
            claimed_place="London",
            raw="filmed in London mid-afternoon",
        ),
        evidence=list(evidence or []),
    )
    return Adjudication(verdict=verdict, confidence=confidence,
                        narrative=narrative, pack=pack)


def _doc(adj):
    """Render to a temp file; assert the file lands; return its text."""
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "report.html")
        assert render_html(adj, out) == out
        text = Path(out).read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    return text


def test_writes_file_with_core_content():
    evs = [
        Evidence(type="solar",
                 summary="sun elevation mismatch vs claimed time",
                 verdict=ClaimVerdict.CONTRADICTS, confidence=0.9),
        Evidence(type="weather",
                 summary="recorded weather matches the claimed hour",
                 verdict=ClaimVerdict.SUPPORTS, confidence=0.6),
    ]
    doc = _doc(_adj(evidence=evs))
    assert 'data-final="CONTRADICTED"' in doc                # verdict string
    assert "<title>Sundial — CONTRADICTED</title>" in doc
    assert 'id="confNum" data-target="83"' in doc            # confidence %
    assert "Sundial" in doc                                  # brand header/footer
    assert "The shadows disagree with the clock." in doc     # narrative
    assert "interview.mp4" in doc                            # clip path
    assert "2025-06-15T14:30:00+00:00" in doc                # claimed time chip
    assert "London" in doc                                   # claimed place chip
    assert "filmed in London mid-afternoon" in doc           # raw claim quote
    for ev in evs:                                           # every signal rendered
        assert ev.summary in doc
        assert f'<span class="etype">{ev.type}</span>' in doc
    assert "Evidence — 2 signals" in doc
    assert "requestAnimationFrame" in doc                    # animation JS shipped


def test_escapes_html_in_evidence_summary():
    ev = Evidence(type="geo",
                  summary='sign reads "<script>alert(1)</script>" verbatim',
                  verdict=ClaimVerdict.SUPPORTS, confidence=0.5)
    claim = Claim(raw='a <em>"quoted"</em> claim',
                  claimed_place="L<on>don")
    doc = _doc(_adj(narrative="verdict: <b>shady</b>",
                    evidence=[ev], claim=claim))
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in doc
    assert "<script>alert(1)</script>" not in doc
    # exactly one raw <script> — the report's own animation block at the end
    assert doc.count("<script") == 1 and doc.count("</script>") == 1
    body = doc.partition("<script>")[0]          # everything before the JS block
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body   # escaped in the card
    assert "&lt;b&gt;shady&lt;/b&gt;" in body                # narrative escaped too
    assert "&lt;em&gt;" in body and "L&lt;on&gt;don" in body  # claim fields escaped


def test_all_verdicts_render():
    accents = {Verdict.SUPPORTED: "#4ade80",
               Verdict.CONTRADICTED: "#f87171",
               Verdict.INSUFFICIENT: "#94a3b8"}
    assert set(accents) == set(Verdict)
    for v in Verdict:
        doc = _doc(_adj(verdict=v))
        assert f'data-final="{v.value}"' in doc, v
        assert f"<title>Sundial — {v.value}</title>" in doc, v
        assert f"--accent:{accents[v]}" in doc, v            # per-verdict theming
    # a plain-string verdict on the dataclass still normalizes and renders
    doc = _doc(_adj(verdict="supported"))
    assert 'data-final="SUPPORTED"' in doc


def test_pill_colors_differ_per_claim_verdict():
    evs = [Evidence(type="geo", summary=f"sig-{cv.value}",
                    verdict=cv, confidence=0.5)
           for cv in ClaimVerdict]
    doc = _doc(_adj(evidence=evs))
    colors = re.findall(r'class="pill" style="color:(#[0-9a-f]{6})', doc)
    assert len(colors) == 3 and len(set(colors)) == 3, colors
    for cv, fg in ((ClaimVerdict.SUPPORTS, "#4ade80"),
                   (ClaimVerdict.CONTRADICTS, "#f87171"),
                   (ClaimVerdict.INCONCLUSIVE, "#94a3b8")):
        pill = f'class="pill" style="color:{fg};'
        assert pill in doc, cv
        assert f'>{cv.value}</span>' in doc, cv             # pill label rendered


def test_solar_evidence_gets_sunviz():
    day = Evidence(type="solar", summary="sun up but wrong azimuth",
                   verdict=ClaimVerdict.CONTRADICTS, confidence=0.8,
                   data={"expected": {"elevation_deg": 41.2, "azimuth_deg": 190.0},
                         "observed": {"elevation_deg": 12.4, "azimuth_deg": 150.0}})
    doc = _doc(_adj(evidence=[day]))
    assert 'class="sunviz"' in doc
    assert "claimed: 41°" in doc and "measured: 12°" in doc
    # night-lie case: expected sun below horizon, observed azimuth absent
    night = Evidence(type="solar", summary="shadow at night",
                     verdict=ClaimVerdict.CONTRADICTS, confidence=0.9,
                     data={"expected": {"elevation_deg": -12.0},
                           "observed": {"elevation_deg": 35.0, "azimuth_deg": None}})
    doc = _doc(_adj(evidence=[night]))
    assert 'class="sunviz"' in doc and "below" in doc


def test_solar_evidence_without_angles_no_sunviz():
    cases = [
        {},                                                     # no payload
        {"expected": {"elevation_deg": 40.0}},                  # observed missing
        {"expected": {}, "observed": {"elevation_deg": 10.0}},  # expected elev missing
        {"expected": {"elevation_deg": "n/a"},
         "observed": {"elevation_deg": 5.0}},                   # non-float
        {"expected": 5, "observed": 3},                         # non-dict payloads
    ]
    for data in cases:
        doc = _doc(_adj(evidence=[Evidence(
            type="solar", summary="solar but no angles",
            verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.2, data=data)]))
        assert 'class="sunviz"' not in doc, data     # no viz, no crash
        assert "solar but no angles" in doc
    # same payload under a non-solar type never triggers the diagram
    doc = _doc(_adj(evidence=[Evidence(
        type="weather", summary="wx", verdict=ClaimVerdict.SUPPORTS,
        confidence=0.5,
        data={"expected": {"elevation_deg": 40.0},
              "observed": {"elevation_deg": 10.0}})]))
    assert 'class="sunviz"' not in doc


def test_frames_render_as_cited_basenames():
    ev = Evidence(type="texture", summary="film grain consistent",
                  verdict=ClaimVerdict.SUPPORTS, confidence=0.5,
                  frames=["/tmp/frames/f_0001.jpg", "deep/dir/f_0002.jpg",
                          "f_0003.jpg", "/x/f_0004.jpg", "f_0005_extra.jpg"])
    doc = _doc(_adj(evidence=[ev]))
    m = re.search(r'class="frames">cited: ([^<]+)<', doc)
    assert m, "no cited: line rendered"
    # basenames only, first four frames
    assert m.group(1) == "f_0001.jpg, f_0002.jpg, f_0003.jpg, f_0004.jpg"
    assert "/" not in m.group(1) and "f_0005_extra.jpg" not in m.group(1)


def test_confbar_data_w_matches_confidence():
    evs = [
        Evidence(type="geo", summary="a", verdict=ClaimVerdict.SUPPORTS,
                 confidence=0.87),
        Evidence(type="clock", summary="b", verdict=ClaimVerdict.INCONCLUSIVE,
                 confidence=0.0),
        Evidence(type="weather", summary="c", verdict=ClaimVerdict.CONTRADICTS,
                 confidence=1.5),                           # clamps to 100
    ]
    doc = _doc(_adj(confidence=0.83, evidence=evs))
    for pct in (87, 0, 100):
        assert f'class="conffill" data-w="{pct}"' in doc, pct
    assert 'id="confNum" data-target="83"' in doc            # adjudicator ring


def test_empty_evidence_pack_still_renders():
    doc = _doc(_adj(verdict=Verdict.INSUFFICIENT, confidence=0.2, evidence=[]))
    assert 'data-final="INSUFFICIENT"' in doc
    assert "No evidence was collected." in doc
    assert "Evidence — 0 signals" in doc
    assert 'class="sunviz"' not in doc


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
    print(f"\n{len(tests) - failed}/{len(tests)} ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
