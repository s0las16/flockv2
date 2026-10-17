"""Evidence provenance channels — where each signal actually came from.

Judges care about the *kind* of support behind a card, not just its
verdict: did the model see it in the pixels (visual), did it come from
an outside API or archive (external), was it computed from other signals
(derived), or is it a known answer (ground-truth)?

tag() stamps ``evidence.data["channel"]`` on copies of the evidence, so
the report/batch views can render a per-card badge plus a "signal mix"
footer without re-deriving it from type names. Stdlib only.
"""
from __future__ import annotations

import copy
import re

#: The four provenance channels a piece of evidence can belong to.
CHANNELS = {"visual", "external", "derived", "ground-truth"}

#: Display order for summaries/footers (CHANNELS itself is unordered).
_ORDER = ("visual", "external", "derived", "ground-truth")

# Evidence type -> channel.
#
#   visual       — the model saw it in the pixels
#   external     — third-party API / archive data
#   derived      — computed from other signals
#   ground-truth — known answer for the clip
_TYPE_CHANNEL = {
    # --- visual: VLM/perception read it straight from the frames --------
    "shadows": "visual",     # shadow direction/length observed in frames
    "texture": "visual",     # sky/lighting/wetness described from frames
    "clock": "visual",       # burned-in overlay timestamps parsed from image
    "geo-text": "visual",    # sign/place text read from frames
    "overlay": "visual",
    "vlm": "visual",
    # --- external: API / archive lookups --------------------------------
    "weather": "external",   # Open-Meteo historical archive
    "geo": "external",       # Nominatim geocoding verdict on sign text
    "geo-api": "external",
    "geo-resolve": "external",
    "archive": "external",
    # --- derived: computed from other signals ----------------------------
    "solar": "derived",      # ephemeris vs measured shadow elevation
    "azimuth": "derived",
    "scene": "derived",
    # --- ground truth ----------------------------------------------------
    "ground-truth": "ground-truth",
    "ground_truth": "ground-truth",
    "groundtruth": "ground-truth",
}

#: Fallback for unmapped types — a stage-failure note or a type nobody
#: classified is pipeline-produced metadata, so "derived" is honest.
_DEFAULT_CHANNEL = "derived"


def channel_for(ev_type: str) -> str:
    """Map an Evidence.type string to its provenance channel.

    Resolution order: the type is already a channel name -> exact table
    match -> token match for compound stage names ("weather+solar",
    "geo-resolve" resolve via their first known token) -> default.
    """
    t = str(ev_type or "").strip().lower()
    if t in CHANNELS:
        return t
    if t in _TYPE_CHANNEL:
        return _TYPE_CHANNEL[t]
    for tok in re.split(r"[^a-z]+", t):
        if tok in _TYPE_CHANNEL:
            return _TYPE_CHANNEL[tok]
        if tok in CHANNELS:
            return tok
    return _DEFAULT_CHANNEL


def tag(evidence_list) -> list:
    """Return a new list of Evidence with ``data["channel"]`` stamped.

    The originals are never mutated: each Evidence is shallow-copied and
    its ``data`` dict is copied, so the channel stamp can't leak back
    into the pack.
    """
    out = []
    for ev in evidence_list or []:
        ev2 = copy.copy(ev)
        ev2.data = dict(getattr(ev, "data", None) or {})
        ev2.data["channel"] = channel_for(getattr(ev, "type", ""))
        out.append(ev2)
    return out


def channel_summary(evidence_list) -> dict:
    """Count evidence per channel — the "signal mix" footer.

    Respects a pre-stamped ``data["channel"]`` (so it works on both raw
    and tag()ed lists) and falls back to inferring from the type.
    All four channels are always present, zero counts included.
    """
    counts = {c: 0 for c in _ORDER}
    for ev in evidence_list or []:
        data = getattr(ev, "data", None) or {}
        ch = data.get("channel")
        if ch not in CHANNELS:
            ch = channel_for(getattr(ev, "type", ""))
        counts[ch] = counts.get(ch, 0) + 1
    return counts


if __name__ == "__main__":
    try:
        from .schema import Claim, ClaimVerdict, Evidence, EvidencePack
    except ImportError:  # allow `python sundial/provenance.py` directly
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from schema import Claim, ClaimVerdict, Evidence, EvidencePack

    pack = EvidencePack(
        clip="demo.mp4",
        claim=Claim(
            claimed_time="2026-10-17T08:00",
            claimed_place="London",
            raw="filmed Tuesday 8am, London",
        ),
    )
    pack.add(Evidence(
        type="shadows",
        summary="long shadows raking left-to-right across the frame",
        verdict=ClaimVerdict.SUPPORTS, confidence=0.7,
    ))
    pack.add(Evidence(
        type="clock",
        summary='camera overlay reads "08:02" — matches the claimed time',
        verdict=ClaimVerdict.SUPPORTS, confidence=0.85,
    ))
    pack.add(Evidence(
        type="texture",
        summary="scene conditions observed: sky=clear, lighting=golden",
        verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.3,
    ))
    pack.add(Evidence(
        type="weather",
        summary="archive shows clear sky at the claimed hour",
        verdict=ClaimVerdict.SUPPORTS, confidence=0.6,
    ))
    pack.add(Evidence(
        type="geo",
        summary="sign text geolocates to Lambeth, ~3 km from claimed place",
        verdict=ClaimVerdict.SUPPORTS, confidence=0.7,
    ))
    pack.add(Evidence(
        type="solar",
        summary="measured sun 12° vs expected 11° — consistent",
        verdict=ClaimVerdict.SUPPORTS, confidence=0.8,
    ))

    tagged = tag(pack.evidence)
    print(f"{'type':12s} {'channel':13s} summary")
    print("-" * 72)
    for orig, stamped in zip(pack.evidence, tagged):
        assert "channel" not in orig.data, "tag() must not mutate originals"
        print(f"{stamped.type:12s} {stamped.data['channel']:13s} "
              f"{stamped.summary}")
    print("\nsignal mix:", channel_summary(tagged))
