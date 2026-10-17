"""Sundial demo seed — one-command demo prep.

  python3 -m sundial.demo_seed --check            # env check only (JSON)
  python3 -m sundial.demo_seed --warm a.mp4 b.mp4 # pre-warm the VLM cache
  python3 -m sundial.demo_seed --script           # print the 5-beat demo
  python3 sundial/demo_seed.py                    # offline self-test

Stdlib only at import; pipeline is imported lazily inside warm() so
`import sundial.demo_seed` never touches third-party deps. Nothing here
raises — every failure lands in the returned dicts, because a demo-prep
tool that crashes is worse than no tool.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time

_ENV_VARS = ("cosmos", "wandb")  # names that map onto env vars, not modules


def _has_module(name: str) -> bool:
    """importlib.util.find_spec probe — importable without importing."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def check_env() -> dict:
    """Probe the demo environment: env vars, binaries, third-party modules.

    Returns {"cosmos", "wandb", "ffmpeg", "ultralytics", "cv2": bool,
             "missing": [names of the False entries]}.
    """
    env = {
        # Cosmos NIM endpoint — pipeline skips every VLM stage without it.
        "cosmos": bool(os.environ.get("COSMOS_BASE_URL")),
        # W&B serverless inference — adjudicator won't run without a key.
        "wandb": bool(os.environ.get("WANDB_API_KEY")),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ultralytics": _has_module("ultralytics"),
        "cv2": _has_module("cv2"),
    }
    env["missing"] = [name for name, ok in env.items() if not ok]
    return env


def warm(clip_paths: list) -> dict:
    """Run the pipeline once per fixture clip so the VLM cache is hot.

    SUNDIAL_NO_CACHE is unset for the duration (restored afterwards) —
    the whole point is that cached frame+prompt lookups answer instantly
    during the recorded run. Returns {clip: {"clip", "verdict", "conf",
    "seconds"}}; per-clip errors are caught and reported as ERROR rows.
    """
    clips = [str(c) for c in (clip_paths or [])]
    results: dict = {}
    if not clips:
        return results

    saved = os.environ.pop("SUNDIAL_NO_CACHE", None)
    try:
        try:
            from .pipeline import run
        except Exception:
            try:
                from sundial.pipeline import run  # direct-script fallback
            except Exception as e:
                print(f"warm: pipeline unavailable ({e}) — clips not warmed")
                for clip in clips:
                    results[clip] = {"clip": clip, "verdict": "ERROR",
                                     "conf": 0.0, "seconds": 0.0}
                return results

        for clip in clips:
            row = {"clip": clip, "verdict": "ERROR", "conf": 0.0,
                   "seconds": 0.0}
            t0 = time.time()
            try:
                adj = run(clip)
                verdict = getattr(adj, "verdict", None)
                row["verdict"] = str(getattr(verdict, "value", verdict)
                                     or "UNKNOWN")
                row["conf"] = round(float(getattr(adj, "confidence", 0.0)
                                          or 0.0), 3)
            except Exception as e:
                print(f"warm: {clip} failed ({e})")
            row["seconds"] = round(time.time() - t0, 2)
            results[clip] = row
    finally:
        if saved is not None:
            os.environ["SUNDIAL_NO_CACHE"] = saved
    return results


def demo_script() -> str:
    """The 5-beat demo: UI, truth run, lie run, honesty run, verdict cards."""
    return """\
SUNDIAL — 5-BEAT DEMO
=====================

1. Upload UI
   python3 -m sundial.serve --port 8765
   open http://localhost:8765
   ("Sundial verifies when and where a video was actually filmed.")

2. Truth run — claim matches reality -> SUPPORTED
   python3 -m sundial.cli --video /tmp/husum_stamped.mp4 \\
       --claim-time '2026-10-17T12:30:00' --claim-place 'Husum, Germany' \\
       --out v_truth.html

3. Lie run — wrong claim on the same clip -> CONTRADICTED
   python3 -m sundial.cli --video /tmp/husum_stamped.mp4 \\
       --claim-time '2026-10-17T20:30:00' --claim-place 'Lyon, France' \\
       --out v_lie.html

4. Honesty run — thin evidence -> INSUFFICIENT (it never guesses)
   python3 -m sundial.cli --video /tmp/busy_square.mp4 \\
       --claim-time '2026-10-17T08:00:00' --claim-place 'Husum, Germany' \\
       --out v_insufficient.html

5. Verdict cards — open v_truth.html / v_lie.html / v_insufficient.html
   Highlight the solar line: "Sun elevation 25.0 deg, expected 25.7 deg."
   Each run also writes a JSON sidecar next to the HTML report.

Runs take 2-4 min live; jump-cut the waiting, not the output.\
"""


def _self_test() -> int:
    """Offline dry checks — no env, no clips, no network."""
    ok = True
    env = check_env()
    need = {"cosmos", "wandb", "ffmpeg", "ultralytics", "cv2", "missing"}
    ok &= need <= set(env.keys())
    ok &= isinstance(env.get("missing"), list)
    print("check_env keys:", sorted(env.keys()))
    print("check_env ->", json.dumps(env))

    script = demo_script()
    ok &= bool(script.strip())
    print(f"demo_script -> {len(script)} chars, non-empty={bool(script.strip())}")

    res = warm([])
    ok &= res == {}
    print(f"warm([]) -> {res}")

    print("demo_seed self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sundial.demo_seed", description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="print the environment check as JSON")
    ap.add_argument("--warm", nargs="*", metavar="CLIP",
                    help="pre-warm the VLM cache on these fixture clips")
    ap.add_argument("--script", action="store_true",
                    help="print the 5-beat demo script")
    args = ap.parse_args(argv)

    if not (args.check or args.warm is not None or args.script):
        return _self_test()

    if args.check:
        env = check_env()
        print(json.dumps(env, indent=2))
        if env["missing"]:
            print(f"missing: {', '.join(env['missing'])} "
                  "(runs degrade gracefully — VLM/adjudicator stages skip)")

    if args.warm is not None:
        if not args.warm:
            print("warm: no clips given")
        else:
            print(json.dumps(warm(args.warm), indent=2))

    if args.script:
        print(demo_script())
    return 0


if __name__ == "__main__":
    sys.exit(main())
