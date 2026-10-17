"""Tests for sundial.adjudicate — run directly: python3 tests/test_adjudicate.py

All offline: exercises only deterministic helpers and _fallback — never
touches WandBClient / the W&B network path.
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sundial.adjudicate import (  # noqa: E402
    SYSTEM_PROMPT,
    _clamp_conf,
    _fallback,
    _first_json_object,
    _norm_verdict,
    _pack_payload,
    _shrink,
)
from sundial.schema import (  # noqa: E402
    Claim,
    ClaimVerdict,
    Evidence,
    EvidencePack,
    Verdict,
)


def _pack(*evs):
    return EvidencePack(
        clip="clip.mp4",
        claim=Claim(claimed_time="2025-06-15T12:00:00", claimed_place="London", raw="noon in London"),
        evidence=list(evs),
    )


def _ev(verdict, confidence, type_="solar"):
    return Evidence(type=type_, summary="finding", verdict=verdict, confidence=confidence)


# --------------------------------------------------------------------------- #
# _fallback
# --------------------------------------------------------------------------- #
def test_fallback_high_conf_contradicts():
    pack = _pack(_ev(ClaimVerdict.CONTRADICTS, 0.8), _ev(ClaimVerdict.SUPPORTS, 0.7))
    a = _fallback(pack)
    assert a.verdict == Verdict.CONTRADICTED
    assert 0.6 <= a.confidence <= 0.95
    assert a.pack is pack


def test_fallback_supports_only():
    pack = _pack(_ev(ClaimVerdict.SUPPORTS, 0.7), _ev(ClaimVerdict.SUPPORTS, 0.9))
    a = _fallback(pack)
    assert a.verdict == Verdict.SUPPORTED
    assert abs(a.confidence - 0.9) < 1e-9


def test_fallback_mixed_contradicts_supports_high_conf():
    pack = _pack(_ev(ClaimVerdict.CONTRADICTS, 0.6), _ev(ClaimVerdict.SUPPORTS, 0.9))
    a = _fallback(pack)
    assert a.verdict == Verdict.CONTRADICTED


def test_fallback_mixed_contradicts_supports_low_conf():
    # Any CONTRADICTS below the 0.6 gate still blocks SUPPORTED -> INSUFFICIENT
    pack = _pack(_ev(ClaimVerdict.CONTRADICTS, 0.4), _ev(ClaimVerdict.SUPPORTS, 0.9))
    a = _fallback(pack)
    assert a.verdict == Verdict.INSUFFICIENT
    assert a.confidence <= 0.5


def test_fallback_empty_pack():
    a = _fallback(_pack())
    assert a.verdict == Verdict.INSUFFICIENT
    assert a.confidence <= 0.5
    assert "no signals" in a.narrative


def test_fallback_inconclusive_only():
    a = _fallback(_pack(_ev(ClaimVerdict.INCONCLUSIVE, 0.9)))
    assert a.verdict == Verdict.INSUFFICIENT


def test_fallback_narrative_counts():
    pack = _pack(
        _ev(ClaimVerdict.SUPPORTS, 0.7),
        _ev(ClaimVerdict.CONTRADICTS, 0.3, "weather"),
        _ev(ClaimVerdict.INCONCLUSIVE, 0.5, "geo"),
    )
    a = _fallback(pack)
    assert "1 point to the claim being true, 1 point against it" in a.narrative
    assert "geo" in a.narrative and "solar" in a.narrative and "weather" in a.narrative


def test_fallback_note_prepended():
    a = _fallback(_pack(), note="LLM adjudication failed; deterministic fallback used.")
    assert a.narrative.startswith("LLM adjudication failed; deterministic fallback used.")


# --------------------------------------------------------------------------- #
# _first_json_object
# --------------------------------------------------------------------------- #
def test_first_json_bare():
    obj = _first_json_object('{"verdict": "SUPPORTED", "confidence": 0.9}')
    assert obj == {"verdict": "SUPPORTED", "confidence": 0.9}


def test_first_json_inside_prose():
    obj = _first_json_object('Here is my answer: {"verdict": "CONTRADICTED", "confidence": 0.7} done.')
    assert obj["verdict"] == "CONTRADICTED"


def test_first_json_markdown_fence():
    raw = "```json\n{\"verdict\": \"INSUFFICIENT\", \"confidence\": 0.3}\n```"
    obj = _first_json_object(raw)
    assert obj["verdict"] == "INSUFFICIENT"


def test_first_json_braces_and_escaped_quotes_in_strings():
    raw = '{"narrative": "he said \\"hi {there}\\" and {x}", "verdict": "SUPPORTED"}'
    obj = _first_json_object(raw)
    assert obj["verdict"] == "SUPPORTED"
    assert obj["narrative"] == 'he said "hi {there}" and {x}'


def test_first_json_trailing_text():
    obj = _first_json_object('{"a": 1, "b": {"c": 2}} trailing junk {not json')
    assert obj == {"a": 1, "b": {"c": 2}}


def test_first_json_no_object_raises():
    for raw in ("", "no braces at all", "just [1, 2, 3] array", "{unclosed"):
        try:
            _first_json_object(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("no ValueError for %r" % raw)


# --------------------------------------------------------------------------- #
# _norm_verdict
# --------------------------------------------------------------------------- #
def test_norm_verdict_supported():
    assert _norm_verdict("supported") == Verdict.SUPPORTED
    assert _norm_verdict("SUPPORTED") == Verdict.SUPPORTED
    assert _norm_verdict("Supports the claim") == Verdict.SUPPORTED


def test_norm_verdict_contradicted():
    assert _norm_verdict("contradicts the claim") == Verdict.CONTRADICTED
    assert _norm_verdict("CONTRADICTED") == Verdict.CONTRADICTED


def test_norm_verdict_insufficient():
    assert _norm_verdict(None) == Verdict.INSUFFICIENT
    assert _norm_verdict("") == Verdict.INSUFFICIENT
    assert _norm_verdict("INSUFFICIENT") == Verdict.INSUFFICIENT
    assert _norm_verdict("garbage") == Verdict.INSUFFICIENT
    assert _norm_verdict(42) == Verdict.INSUFFICIENT


# --------------------------------------------------------------------------- #
# _clamp_conf
# --------------------------------------------------------------------------- #
def test_clamp_conf_basic():
    assert _clamp_conf(0.7) == 0.7
    assert _clamp_conf("0.25") == 0.25


def test_clamp_conf_out_of_range():
    assert _clamp_conf(1.7) == 1.0
    assert _clamp_conf(-0.4) == 0.0


def test_clamp_conf_bad_values_default():
    assert _clamp_conf(None) == 0.5
    assert _clamp_conf("abc") == 0.5
    assert _clamp_conf(None, default=0.1) == 0.1


# --------------------------------------------------------------------------- #
# _pack_payload / _shrink
# --------------------------------------------------------------------------- #
def test_pack_payload_structure():
    pack = _pack(_ev(ClaimVerdict.SUPPORTS, 0.7))
    p = _pack_payload(pack)
    assert p["clip"] == "clip.mp4"
    assert p["claim"]["claimed_time"] == "2025-06-15T12:00:00"
    assert p["claim"]["claimed_place"] == "London"
    assert p["claim"]["raw"] == "noon in London"
    assert len(p["evidence"]) == 1
    ev = p["evidence"][0]
    assert ev["type"] == "solar" and ev["verdict"] == "supports"
    assert ev["confidence"] == 0.7
    # payload must be JSON-serializable (this is what is sent to the LLM)
    json.dumps(p, default=str)


def test_pack_payload_frames_basename_and_cap():
    ev = _ev(ClaimVerdict.SUPPORTS, 0.7)
    ev.frames = ["/a/b/f1.jpg", "x/y/f2.jpg", "f3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg"]
    p = _pack_payload(_pack(ev))
    frames = p["evidence"][0]["frames"]
    assert frames == ["f1.jpg", "f2.jpg", "f3.jpg", "4.jpg", "5.jpg", "6.jpg"], frames


def test_shrink_long_string_capped():
    s = _shrink("x" * 500)
    assert len(s) == 301 and s.endswith("…")
    assert _shrink("x" * 300) == "x" * 300


def test_shrink_nested_data():
    data = {"long": "y" * 400, "list": list(range(15)), "n": 3}
    out = _shrink(data)
    assert out["long"].endswith("…") and len(out["long"]) == 301
    assert out["list"] == list(range(10))  # capped at 10 items
    assert out["n"] == 3


# --------------------------------------------------------------------------- #
# SYSTEM_PROMPT regression guard — trust-hierarchy terms
# --------------------------------------------------------------------------- #
def test_system_prompt_trust_hierarchy_terms():
    assert "clock" in SYSTEM_PROMPT
    assert "solar" in SYSTEM_PROMPT
    assert "STRICT JSON" in SYSTEM_PROMPT


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
