"""Sundial CLI.

  python -m sundial.cli --video clip.mp4 --claim-time "2025-10-17T15:00:00" \
      --claim-place "London" --out verdict.html
  python -m sundial.cli --demo          # zero-dependency smoke test
"""
from __future__ import annotations

import argparse
import json
import sys


def _demo_pack():
    """Fabricated evidence pack — exercises adjudication + report with no deps."""
    from .schema import Claim, ClaimVerdict, Evidence, EvidencePack

    pack = EvidencePack(
        clip="demo_clip.mp4",
        claim=Claim(
            claimed_time="2025-10-15T15:00:00",
            claimed_place="London",
            raw="filmed at 3pm in London on Oct 15",
        ),
    )
    pack.add(Evidence(
        type="solar",
        summary="sun at 41.0° elev / 104° az points to 08:12–08:52 UTC; "
                "claimed 14:00 UTC expects 27.4° elev / 224° az — mismatch",
        verdict=ClaimVerdict.CONTRADICTS,
        confidence=0.82,
        data={"windows_utc": [{"start": "2025-10-15T08:12", "end": "2025-10-15T08:52"}]},
    ))
    pack.add(Evidence(
        type="weather",
        summary="recorded weather contradicts clip: archive records 0.0mm "
                "precipitation but the clip shows rain",
        verdict=ClaimVerdict.CONTRADICTS,
        confidence=0.7,
        data={"recorded": {"precipitation": 0.0, "cloudcover": 15}},
    ))
    pack.add(Evidence(
        type="geo",
        summary="sign text geolocates to 'Westminster Bridge Rd, London' "
                "(2 km from claimed 'London')",
        verdict=ClaimVerdict.SUPPORTS,
        confidence=0.75,
        data={},
    ))
    pack.add(Evidence(
        type="texture",
        summary="scene conditions observed: sky=overcast, lighting=flat daylight, precip=rain",
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.3,
        data={},
    ))
    return pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sundial", description=__doc__)
    ap.add_argument("--video", help="path to the clip under scrutiny")
    ap.add_argument("--claim-time", help="ISO datetime the clip claims (local to place)")
    ap.add_argument("--claim-place", help='claimed location: text or "lat,lon"')
    ap.add_argument("--out", default="verdict.html", help="HTML report output path")
    ap.add_argument("--frames-dir", default=".sundial_frames")
    ap.add_argument("--no-vlm", action="store_true", help="skip Cosmos VLM calls")
    ap.add_argument("--demo", action="store_true", help="run fabricated evidence pack")
    args = ap.parse_args(argv)

    from .adjudicate import adjudicate
    from .report import render_html

    if args.demo:
        adj = adjudicate(_demo_pack())
    else:
        if not args.video:
            ap.error("--video is required unless --demo")
        from .pipeline import run
        adj = run(
            args.video,
            claim_time=args.claim_time,
            claim_place=args.claim_place,
            workdir=args.frames_dir,
            use_vlm=not args.no_vlm,
        )

    path = render_html(adj, args.out)
    print(json.dumps({
        "verdict": adj.verdict.value,
        "confidence": adj.confidence,
        "narrative": adj.narrative,
        "evidence": [
            {"type": e.type, "verdict": e.verdict.value, "summary": e.summary}
            for e in adj.pack.evidence
        ],
        "report": path,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
