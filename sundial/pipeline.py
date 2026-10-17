"""Pipeline — clip + claim -> EvidencePack -> Adjudication.

Every stage is failure-isolated: a stage that errors contributes an
INCONCLUSIVE "stage unavailable" evidence card instead of crashing, so the
demo never hard-dies on a single bad call.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone

from .adjudicate import adjudicate
from .schema import Claim, ClaimVerdict, Evidence, EvidencePack


def _note(pack: EvidencePack, stage: str, err: object) -> None:
    pack.add(Evidence(
        type=stage,
        summary=f"{stage} stage unavailable: {err}",
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.0,
        data={"error": str(err)},
    ))


def _parse_latlon(place: str | None) -> tuple[float, float] | None:
    try:
        a, b = [float(x) for x in str(place).split(",")][:2]
        if -90 <= a <= 90 and -180 <= b <= 180:
            return a, b
    except (ValueError, TypeError):
        pass
    return None


def _resolve_coords(claim_place: str | None) -> tuple[float | None, float | None, str]:
    """-> (lat, lon, how-we-got-it). Geocodes the claim if needed."""
    if not claim_place:
        return None, None, "no claimed place"
    p = _parse_latlon(claim_place)
    if p:
        return p[0], p[1], "coordinates given in claim"
    from . import geo
    cands = geo.geocode(claim_place)
    if not cands:
        return None, None, f"could not geocode claimed place {claim_place!r}"
    return cands[0].lat, cands[0].lon, f"geocoded claim to {cands[0].display_name!r}"


def _sun_elevation_deg(shadow_obs: dict, detections: list[dict]) -> float | None:
    """Estimate sun elevation from the object-height : shadow-length pixel ratio.

    For a vertical object on flat ground, elev ~ atan(obj_px_height / shadow_px_len)
    — the object's real height cancels out. Approximate under perspective; fine
    for flagging hours-off claims.
    """
    from . import perception
    vec = perception.shadow_vector(shadow_obs)
    if not vec["ok"]:
        return None
    cls = (shadow_obs.get("object_class") or "").lower()
    best, best_h = None, 0.0
    for d in detections or []:
        dcls = str(d.get("class") or "").lower()
        if dcls and (dcls in cls or cls in dcls):
            box = d.get("bbox_xyxy") or []
            h = float(box[3]) - float(box[1]) if len(box) == 4 else 0.0
            if h > best_h:
                best, best_h = d, h
    if best is None or best_h <= 0:
        return None
    return math.degrees(math.atan2(best_h, vec["length_px"]))


def _norm_conditions(c: dict) -> dict:
    """Normalize a Cosmos conditions blob into weather.compare_weather's keys."""
    out: dict = {}
    sky = str(c.get("sky") or "").lower()
    if sky in ("clear", "partly", "overcast"):
        out["sky"] = sky
    pr = str(c.get("precipitation") or "").lower()
    if pr in ("none", "rain", "snow"):
        out["precipitation"] = pr
    gw = c.get("ground_wetness")
    if isinstance(gw, bool):
        out["wet_ground"] = gw
    else:
        s = str(gw).lower()
        if "wet" in s:
            out["wet_ground"] = True
        elif "dry" in s:
            out["wet_ground"] = False
    return out


def _claimed_utc(lat: float, lon: float, claim_time_iso: str) -> tuple[str, str]:
    """Convert claimed local time -> UTC using the archive's utc_offset_seconds.

    Returns (utc_iso, note). Falls back to treating the claim as UTC.
    """
    naive = datetime.fromisoformat(claim_time_iso).replace(tzinfo=None)
    try:
        from . import weather
        arch = weather.historical_weather(lat, lon, str(claim_time_iso)[:10])
        off = int(arch.get("utc_offset_seconds") or 0)
        utc = naive - timedelta(seconds=off)
        return utc.replace(tzinfo=timezone.utc).isoformat(), (
            f"claim local {naive.isoformat()} -> UTC via offset {off}s"
        )
    except Exception as e:
        return naive.replace(tzinfo=timezone.utc).isoformat(), (
            f"offset lookup failed ({e}); claim treated as UTC"
        )


def run(
    video_path: str,
    claim_time: str | None = None,
    claim_place: str | None = None,
    workdir: str = ".sundial_frames",
    use_vlm: bool = True,
    max_vlm_frames: int = 4,
) -> "Adjudication":
    from .schema import Adjudication

    pack = EvidencePack(
        clip=str(video_path),
        claim=Claim(
            claimed_time=claim_time,
            claimed_place=claim_place,
            raw=f"filmed at {claim_time or 'unknown time'}, {claim_place or 'unknown place'}",
        ),
    )

    # --- 1. resolve lat/lon (needed by solar + weather) ----------------------
    lat = lon = None
    try:
        lat, lon, coord_note = _resolve_coords(claim_place)
        if lat is None:
            _note(pack, "geo-resolve", coord_note)
    except Exception as e:
        _note(pack, "geo-resolve", e)

    # --- 2. extract frames ---------------------------------------------------
    frames: list[str] = []
    try:
        from . import perception
        frames = perception.extract_frames(video_path, workdir, fps=1.0, max_frames=24)
    except Exception as e:
        _note(pack, "frames", e)

    # --- 3. VLM observations (shadows, text, conditions) ---------------------
    texts: list[str] = []
    observations: dict = {}
    conditions_list: list[dict] = []
    sun_elev: float | None = None

    if frames and use_vlm and os.environ.get("COSMOS_BASE_URL"):
        try:
            cosmos = perception.CosmosClient()
            try:
                picks = perception.pick_shadow_frames(frames)[:max_vlm_frames]
            except Exception:
                picks = frames[:max_vlm_frames]

            dets_cache: dict[str, list[dict]] = {}
            for f in picks:
                obs = cosmos.locate_shadow(f)
                if obs:
                    pack.add(perception.to_evidence(f, obs))
                    if sun_elev is None:
                        if f not in dets_cache:
                            try:
                                dets_cache[f] = perception.detect_objects(f)
                            except Exception:
                                dets_cache[f] = []
                        sun_elev = _sun_elevation_deg(obs, dets_cache[f])

                txt = cosmos.read_scene_text(f)
                if txt:
                    texts.extend(txt.get("texts") or [])
                    for k in ("driving_side", "sign_language", "plate_style"):
                        if not observations.get(k) and txt.get(k):
                            observations[k] = txt[k]
                    observations["road_markings"] = (
                        observations.get("road_markings") or []
                    ) + (txt.get("road_markings") or [])

                cond = cosmos.describe_conditions(f)
                if cond:
                    conditions_list.append(cond)
        except Exception as e:
            _note(pack, "vlm", e)
    elif use_vlm:
        _note(pack, "vlm", "COSMOS_BASE_URL not set — VLM stages skipped")

    if conditions_list:
        pack.add(Evidence(
            type="texture",
            summary=(
                f"scene conditions observed: sky={conditions_list[0].get('sky')}, "
                f"lighting={conditions_list[0].get('lighting')}, "
                f"precip={conditions_list[0].get('precipitation')}"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.3,
            data={"conditions": conditions_list},
        ))

    # --- 4. geolocation ------------------------------------------------------
    # Hallucination guard: only geocode strings seen in >=2 frames — invented
    # sign text doesn't repeat across frames, real text does.
    try:
        from collections import Counter
        from . import geo
        counts = Counter(t.strip().lower() for t in texts if t and t.strip())
        consensus = [t for t in texts
                     if t and counts[t.strip().lower()] >= 2]
        seen: set[str] = set()
        stable_texts = [t for t in consensus
                        if not (t.strip().lower() in seen or seen.add(t.strip().lower()))]
        ev = geo.geo_evidence(claim_place, stable_texts, observations)
        if len(texts) != len(stable_texts):
            ev.data["texts_filtered"] = len(texts) - len(stable_texts)
        pack.add(ev)
    except Exception as e:
        _note(pack, "geo", e)

    # --- 5. weather + solar (need coords + claimed time) ----------------------
    if lat is not None and claim_time:
        try:
            from . import weather
            weather_obs = _norm_conditions(conditions_list[0]) if conditions_list else {}
            pack.add(weather.compare_weather(weather_obs, lat, lon, claim_time))
        except Exception as e:
            _note(pack, "weather", e)

        if sun_elev is not None:
            try:
                from . import solar
                claimed_utc, tz_note = _claimed_utc(lat, lon, claim_time)
                ev = solar.solar_evidence(lat, lon, claimed_utc, sun_elev)
                ev.data["claim_tz_note"] = tz_note
                ev.data["sun_elevation_source"] = "shadow/object pixel ratio"
                pack.add(ev)
            except Exception as e:
                _note(pack, "solar", e)
    elif claim_time or claim_place:
        _note(pack, "weather+solar",
              "need both resolvable coordinates and a claimed time")

    return adjudicate(pack)
