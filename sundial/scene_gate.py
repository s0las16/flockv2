"""Scene gate — decide which evidence stages are worth running for a clip.

Cheap heuristics over signals the pipeline already collects (VLM conditions
blobs, OCR'd scene text, YOLO detections) that flag scene categories where
specific forensic stages lose meaning: indoor footage has no readable sky or
weather, night footage has no usable shadows, aerial footage has no street
signage to geocode.

This is a GATE, not ground truth. Every flag is a best-effort guess from
noisy VLM output; a wrong flag only deprioritizes a stage (or annotates the
pack), it never votes on the verdict itself.

Stdlib only at module level — schema is imported lazily inside
gate_evidence() so `import sundial.scene_gate` stays dependency-free.
"""
from __future__ import annotations

# Lighting values emitted by CONDITIONS_PROMPT that mean "no usable sun".
_DARK_LIGHTING = frozenset({"night", "dusk"})

# Uppercase interior-signage words that hint the camera is indoors. Weak
# signal on its own — exterior signage can contain these too — so it only
# ever reinforces the sky-based cue, never triggers indoor alone.
_INDOOR_TEXT_CUES = frozenset({
    "EXIT", "MENU", "LIFT", "LOBBY", "TOILET", "WC", "ELEVATOR",
    "ESCALATOR", "RECEPTION", "STAIRS", "PUSH", "PULL", "NO SMOKING",
    "RESTROOM", "CHECKOUT",
})

# Phrases in conditions[].sun.description suggesting a top-down viewpoint.
_AERIAL_PHRASES = ("from above", "overhead", "looking down", "top-down",
                  "bird's eye", "birds eye", "aerial")

# YOLO classes that dominate a forward-facing road view.
_VEHICLE_CLASSES = frozenset({
    "car", "truck", "bus", "motorcycle", "bicycle",
    "traffic light", "stop sign", "parking meter",
})


def _lower(value) -> str:
    return str(value or "").strip().lower()


def _indoor_text_cues(texts: list[str]) -> set:
    """Return the interior-signage cue words seen across OCR strings."""
    import re

    found: set = set()
    for t in texts or []:
        up = str(t).upper()
        found |= _INDOOR_TEXT_CUES & set(re.findall(r"[A-Za-z']+", up))
        if "NO SMOKING" in up:
            found.add("NO SMOKING")
    return found


def _steep_view(detections_summary: list[dict], conditions_list: list[dict]) -> bool:
    """True when any observation carries a steep camera_tilt_deg.

    Tilt lives in shadow observations, not conditions — callers that merge
    them into detections_summary get this check for free; otherwise it's
    simply absent and the aerial flag leans on sky/description cues.
    """
    for blob in list(detections_summary or []) + list(conditions_list or []):
        try:
            if float(blob.get("camera_tilt_deg")) >= 55:
                return True
        except (TypeError, ValueError, AttributeError):
            continue
    return False


def scene_flags(
    conditions_list: list[dict],
    texts: list[str],
    detections_summary: list[dict],
) -> dict:
    """Classify the clip's scene from cheap already-collected signals.

    -> {"indoor": bool, "night": bool, "aerial": bool, "dashcam": bool,
        "crowded": bool, "reasons": [str, ...]}

    `conditions_list` is one CONDITIONS_PROMPT blob per analyzed frame,
    `texts` the OCR'd strings across frames, `detections_summary` the flat
    list of YOLO detections ({class, confidence, bbox_xyxy}).
    """
    conditions_list = [c for c in (conditions_list or []) if isinstance(c, dict)]
    texts = [t for t in (texts or []) if t]
    detections_summary = [d for d in (detections_summary or []) if isinstance(d, dict)]

    n = len(conditions_list)
    reasons: list[str] = []

    # --- night -------------------------------------------------------------
    lightings = [_lower(c.get("lighting")) for c in conditions_list]
    night_reports = sum(1 for l in lightings if l == "night")
    dusk_reports = sum(1 for l in lightings if l == "dusk")
    dark_reports = night_reports + dusk_reports
    # Any explicit "night" report flags it; "dusk" alone needs to dominate
    # the sampled frames — twilight still yields usable shadows.
    night = bool(night_reports) or (n > 0 and dusk_reports * 2 >= n)
    if night:
        detail = f"lighting=='night' in {night_reports}/{n} frames" \
            if night_reports else f"lighting=='dusk' in {dusk_reports}/{n} frames"
        lamps = sum(1 for c in conditions_list
                    if _lower(c.get("streetlights")) == "on")
        if lamps:
            detail += f"; streetlights on in {lamps} frame(s)"
        reasons.append(f"night: {detail}")

    # --- indoor --------------------------------------------------------------
    # The CONDITIONS enum has no "unknown" sky value, but the VLM emits it
    # anyway when no sky is visible — repeated unknowns mean no horizon.
    unknown_sky = sum(1 for c in conditions_list if _lower(c.get("sky"))
                      not in ("clear", "partly", "overcast"))
    indoor_cues = _indoor_text_cues(texts)
    indoor = (
        n > 0 and unknown_sky >= 2 and unknown_sky * 2 >= n
    ) or (
        unknown_sky >= 1 and len(indoor_cues) >= 2
    )
    if indoor:
        why = f"sky unreported/unknown in {unknown_sky}/{n} frames"
        if indoor_cues:
            why += f"; indoor-cue texts {sorted(indoor_cues)}"
        reasons.append(f"indoor: {why}")

    # --- aerial --------------------------------------------------------------
    # Honest heuristic: "viewed from above" in a sun description plus no sky
    # in frame (horizon absent) — or an explicitly steep camera tilt when a
    # caller merges shadow observations into detections_summary.
    descs = [_lower((c.get("sun") or {}).get("description"))
             for c in conditions_list
             if isinstance(c.get("sun"), dict)]
    above_mention = any(any(p in d for p in _AERIAL_PHRASES) for d in descs if d)
    steep = _steep_view(detections_summary, conditions_list)
    horizon_absent = n > 0 and unknown_sky * 2 >= n
    aerial = above_mention and (horizon_absent or steep)
    if aerial:
        reasons.append(
            "aerial: sun described as viewed from above"
            + ("; no sky in >=half of frames" if horizon_absent else "")
            + ("; steep camera tilt" if steep else "")
        )

    # --- detections: crowded + dashcam --------------------------------------
    classes = [_lower(d.get("class")) for d in detections_summary]
    total = len(classes)
    people = sum(1 for c in classes if c == "person")
    vehicles = sum(1 for c in classes if c in _VEHICLE_CLASSES)

    crowded = people >= 8 or (total >= 6 and people >= 5 and people * 5 >= total * 3)
    if crowded:
        reasons.append(f"crowded: {people} person detections out of {total}")

    # Forward-facing road views are vehicle-dominated; dashcam footage is the
    # geo/clock-friendly case, so this flag mostly contextualizes the others.
    dashcam = total >= 4 and vehicles >= 3 and vehicles * 2 >= total
    if dashcam:
        reasons.append(f"dashcam: {vehicles}/{total} detections are road vehicles")

    return {
        "indoor": indoor,
        "night": night,
        "aerial": aerial,
        "dashcam": dashcam,
        "crowded": crowded,
        "reasons": reasons,
    }


def evidence_applicability(flags: dict) -> dict:
    """Map scene flags -> which evidence stages still apply.

    -> {"solar": bool, "shadows": bool, "geo": bool, "weather": bool,
        "clock": bool, "audio": bool}

    Deliberately conservative:
    * solar stays True even at night — the ephemeris check is exactly what
      catches "claimed noon, footage is dark" contradictions. What dies at
      night is the shadow-based sun-elevation measurement (shadows=False).
    * indoor kills weather + geo (interior light/signs aren't the claimed
      outdoors) and shadows (artificial light casts lying shadows).
    * aerial kills geo (no readable signage) and shadows (steep ground
      plane), but weather still applies — clouds/precip read fine top-down.
    * clock + audio survive every scene; overlays and sound don't care
      where the camera points.
    """
    indoor = bool(flags.get("indoor"))
    night = bool(flags.get("night"))
    aerial = bool(flags.get("aerial"))
    crowded = bool(flags.get("crowded"))

    return {
        "solar": True,  # always — sun-position math works day and night
        "shadows": not (night or indoor or aerial or crowded),
        "geo": not (indoor or aerial),
        "weather": not indoor,
        "clock": True,
        "audio": True,
    }


def gate_evidence(flags: dict) -> "Evidence | None":
    """Package the gate result as an INCONCLUSIVE 'scene' evidence card.

    Returns None when nothing was deprioritized — a clean outdoor daytime
    clip needs no annotation. The card lists which stages were gated and
    the heuristic reasons, so the adjudicator/report can discount them.
    """
    from .schema import ClaimVerdict, Evidence

    app = evidence_applicability(flags)
    deprioritized = [k for k, v in app.items() if not v]
    if not deprioritized:
        return None

    reasons = list(flags.get("reasons") or [])
    on = [k for k, v in flags.items() if v is True and k != "reasons"]
    summary = (
        f"scene gate ({', '.join(on) or 'unclassified'}): deprioritized "
        f"{', '.join(deprioritized)}"
        + (f" — {reasons[0]}" if reasons else "")
    )
    return Evidence(
        type="scene",
        summary=summary,
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.3,  # heuristic gate — informational, never a vote
        data={
            "flags": {k: v for k, v in flags.items() if k != "reasons"},
            "reasons": reasons,
            "applicability": app,
            "deprioritized": deprioritized,
        },
    )


if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(
        description="Scene gate demo — print flags for a conditions JSON file."
    )
    ap.add_argument(
        "json_path",
        help="JSON file: either a list of conditions blobs, or an object "
             "with 'conditions', 'texts' and/or 'detections' keys",
    )
    args = ap.parse_args()

    try:
        with open(args.json_path) as fh:
            blob = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"could not read {args.json_path}: {e}")

    if isinstance(blob, list):
        conditions, texts, dets = blob, [], []
    elif isinstance(blob, dict):
        conditions = blob.get("conditions") or []
        texts = blob.get("texts") or []
        dets = blob.get("detections") or blob.get("detections_summary") or []
    else:
        sys.exit("expected a JSON list or object")

    flags = scene_flags(conditions, texts, dets)
    print("flags:")
    print(json.dumps(flags, indent=2))
    print("applicability:")
    print(json.dumps(evidence_applicability(flags), indent=2))
    ev = gate_evidence(flags)
    print("gate card:", ev.summary if ev else "none (all stages apply)")
