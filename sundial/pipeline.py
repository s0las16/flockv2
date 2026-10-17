"""Pipeline — clip + claim -> EvidencePack -> Adjudication.

Every stage is failure-isolated: a stage that errors contributes an
INCONCLUSIVE "stage unavailable" evidence card instead of crashing, so the
demo never hard-dies on a single bad call.
"""
from __future__ import annotations

import glob
import math
import os
import re
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
        a, b = [float(x) for x in str(place).split(",")[:2]]
        if -90 <= a <= 90 and -180 <= b <= 180:
            return a, b
    except (ValueError, TypeError):
        pass
    return None


def _as_str_list(v: object) -> list[str]:
    """Coerce a VLM field that should be list[str] — the model sometimes
    returns a bare string (which list() would explode into characters) or
    non-string items."""
    items = v if isinstance(v, list) else ([v] if v else [])
    return [x.strip() for x in items if isinstance(x, str) and x.strip()]


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


def _sun_elevation_deg(shadow_obs: dict, detections: list[dict],
                       frame_path: str | None = None) -> float | None:
    """Estimate sun elevation from the object-height : shadow-length pixel ratio.

    For a vertical object on flat ground, elev ~ atan(obj_px_height / shadow_px_len)
    — the object's real height cancels out. Approximate under perspective; fine
    for flagging hours-off claims.
    """
    from . import perception
    vec = perception.shadow_vector(shadow_obs)
    if not vec["ok"]:
        return None
    cls = str(shadow_obs.get("object_class") or "").lower()
    if not cls:
        return None  # missing class matches every detection — skip
    best, best_h = None, 0.0
    for d in detections or []:
        dcls = str(d.get("class") or "").lower()
        if dcls and (dcls in cls or cls in dcls):
            box = d.get("bbox_xyxy") or []
            try:
                h = float(box[3]) - float(box[1]) if len(box) == 4 else 0.0
            except (TypeError, ValueError, IndexError):
                h = 0.0
            if h > best_h:
                best, best_h = d, h
    if best is None or best_h <= 0:
        return None
    # Elevated cameras foreshorten the ground plane: distances along the
    # image vertical compress by sin(tilt). Expand dy before measuring the
    # shadow; horizontal stays put. 0.3 floor avoids blowup near-horizontal.
    tilt = shadow_obs.get("camera_tilt_deg")
    try:
        tilt = float(tilt) if tilt is not None else None
    except (TypeError, ValueError):
        tilt = None
    if tilt is not None and tilt > 8:
        dy = vec["dy"] / max(math.sin(math.radians(tilt)), 0.3)
        length = math.hypot(vec["dx"], dy)
    else:
        length = vec["length_px"]
    # Optional cv2 re-measurement of the shadow length — VLM point picks on
    # small objects are noisy; a measured segment beats a guessed tip.
    if frame_path is not None:
        try:
            from . import shadow_measure
            m = shadow_measure.measure_shadow_length(
                frame_path, best.get("bbox_xyxy"),
                tip_hint=(float(vec["tip"][0]), float(vec["tip"][1])))
            if m and (m.get("confidence") or 0) >= 0.4 and m.get("shadow_len_px"):
                length = float(m["shadow_len_px"])
        except Exception:
            pass
    return math.degrees(math.atan2(best_h, length))


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
        tokens = set(s.replace("-", " ").replace("_", " ").split())
        if tokens & {"wet", "damp", "soaked", "puddles"} and "not" not in tokens:
            out["wet_ground"] = True
        elif "dry" in tokens:
            out["wet_ground"] = False
    return out


def _majority_conditions(conditions_list: list[dict]) -> dict:
    """Modal value per key across frames — one hallucinated 'rain' report
    shouldn't drive the archive comparison."""
    from collections import Counter
    out: dict = {}
    for key in ("sky", "precipitation", "lighting", "ground_wetness"):
        vals = [c.get(key) for c in conditions_list if c.get(key) is not None]
        if vals:
            out[key] = Counter(str(v).lower() for v in vals).most_common(1)[0][0]
    return out


_TS_PATTERNS = [
    # 2026-10-17 08:00:02 / 2026/10/17T08:00 — tried first: a dd.mm.yy
    # pattern would otherwise match the "26-10-17" tail of the ISO date.
    re.compile(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})\D{0,4}(\d{1,2}):(\d{2})(?::(\d{2}))?"),
    # 17.10.2026 08:00:02 / 17-10-2026 08:00 / 17.10.26 12:30 (dd.mm.yy)
    re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\D{0,4}(\d{1,2}):(\d{2})(?::(\d{2}))?"),
]
# Bare clock like "08:00" — usable for time-of-day only.
_TS_TIME_ONLY = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")


def _parse_overlay_time(s: str) -> tuple[datetime | None, str | None]:
    """-> (naive local datetime, granularity) for a burned-in timestamp string.

    granularity is "datetime" (full date+time) or "time" (clock only).
    """
    try:
        m = _TS_PATTERNS[0].search(s)
        if m:
            y, mo, d, hh, mm, ss = m.groups()
            return datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0)), "datetime"
        m = _TS_PATTERNS[1].search(s)
        if m:
            d, mo, y, hh, mm, ss = m.groups()
            yi = int(y)
            if yi < 100:
                yi += 2000
            di, mi = int(d), int(mo)
            if mi > 12 and di <= 12:  # US-style mm/dd
                di, mi = mi, di
            return datetime(yi, mi, di, int(hh), int(mm), int(ss or 0)), "datetime"
        m = _TS_TIME_ONLY.search(s)
        if m:
            hh, mm, ss = m.groups()
            return datetime(2000, 1, 1, int(hh), int(mm), int(ss or 0)), "time"
    except ValueError:
        return None, None  # "25:99", month 17, day 40 — unparseable, not evidence
    return None, None


def _consensus_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    """Strings seen in >=2 distinct frames — the texts/markings consensus
    rule. Returns first-sighting-ordered unique strings."""
    seen: dict[str, set] = {}
    order: list[str] = []
    for f, t in pairs:
        k = (t or "").strip().lower()
        if not k:
            continue
        if k not in seen:
            seen[k] = set()
            order.append(k)
        seen[k].add(f)
    return [k for k in order if len(seen[k]) >= 2]


# Country-indicative tokens readable in consensus sign/overlay text —
# dial codes, ccTLDs, currency symbols, legal-form suffixes. Pure-regex,
# zero extra API calls; positive-only hints (absence says nothing).
_DIAL_CC = {
    "1": "US", "44": "GB", "49": "DE", "33": "FR", "39": "IT", "34": "ES",
    "31": "NL", "43": "AT", "41": "CH", "48": "PL", "46": "SE", "47": "NO",
    "45": "DK", "353": "IE", "351": "PT", "30": "GR", "90": "TR", "7": "RU",
    "380": "UA", "81": "JP", "82": "KR", "86": "CN", "91": "IN", "61": "AU",
    "64": "NZ", "55": "BR", "52": "MX", "27": "ZA", "971": "AE", "966": "SA",
    "972": "IL", "65": "SG", "60": "MY", "66": "TH", "84": "VN", "62": "ID",
    "63": "PH", "852": "HK", "886": "TW", "20": "EG", "234": "NG", "254": "KE",
    "54": "AR", "56": "CL", "57": "CO", "32": "BE", "420": "CZ", "36": "HU",
    "40": "RO", "359": "BG", "385": "HR", "386": "SI", "421": "SK",
    "372": "EE", "371": "LV", "370": "LT", "358": "FI", "354": "IS",
}
_TLD_CC = {
    "co.uk": "GB", "uk": "GB", "de": "DE", "fr": "FR", "it": "IT", "es": "ES",
    "nl": "NL", "at": "AT", "ch": "CH", "pl": "PL", "se": "SE", "no": "NO",
    "dk": "DK", "ie": "IE", "pt": "PT", "gr": "GR", "co.jp": "JP", "jp": "JP",
    "co.kr": "KR", "kr": "KR", "cn": "CN", "co.in": "IN", "in": "IN",
    "com.au": "AU", "au": "AU", "co.nz": "NZ", "nz": "NZ", "com.br": "BR",
    "br": "BR", "mx": "MX", "za": "ZA", "ae": "AE", "sa": "SA", "il": "IL",
    "sg": "SG", "my": "MY", "th": "TH", "vn": "VN", "id": "ID", "ph": "PH",
    "hk": "HK", "tw": "TW", "ca": "CA", "ru": "RU", "ua": "UA", "tr": "TR",
    "be": "BE", "cz": "CZ", "hu": "HU", "ro": "RO", "bg": "BG", "hr": "HR",
    "ee": "EE", "lv": "LV", "lt": "LT", "fi": "FI", "is": "IS",
}
_CURRENCY_CC = {"£": "GB", "¥": "JP", "₹": "IN", "₩": "KR", "฿": "TH",
                "₺": "TR", "₽": "RU", "zł": "PL"}
_LEGAL_CC = {
    "gmbh": "DE", "ltd": "GB", "llc": "US", "inc": "US", "sarl": "FR",
    "sas": "FR", "s.a.": None, "spa": "IT", "srl": "IT", "bv": "NL",
    "b.v.": "NL", "ag": "CH", "pty": "AU", "sdn bhd": "MY", "kk": "JP",
    "oy": "FI", "ab": "SE", "aps": "DK", "lda": "PT", "sl": "ES",
    "ooo": "RU", "sp. z o.o.": "PL", "kft": "HU", "sro": "CZ", "doo": "HR",
    "ead": "BG", "ou": "EE", "sia": "LV", "uab": "LT",
}
_TLD_RE = re.compile(
    r"\.(co\.uk|co\.jp|co\.kr|co\.in|com\.au|co\.nz|com\.br|sp\. z o\.o\.|[a-z]{2})\b")


def _text_geo_hints(texts: list[str]) -> dict:
    """Scan consensus text for country-indicative tokens.

    -> {"ccs": sorted candidate ccs, "hits": [human-readable hit notes]}
    """
    blob = " ".join(t for t in texts if t).lower()
    ccs: list[str] = []
    hits: list[str] = []

    def _add(code, note):
        if code and code not in ccs:
            ccs.append(code)
            hits.append(note)

    for m in re.finditer(r"\+(\d{1,3})(?=[\s\-]?\d)", blob):
        code = _DIAL_CC.get(m.group(1))
        _add(code, f"dial +{m.group(1)} -> {code}")
    for m in _TLD_RE.finditer(blob):
        code = _TLD_CC.get(m.group(1))
        _add(code, f"tld .{m.group(1)} -> {code}")
    for sym, code in _CURRENCY_CC.items():
        if sym in blob:
            _add(code, f"currency {sym} -> {code}")
    for form, code in _LEGAL_CC.items():
        if code and re.search(r"\b" + re.escape(form) + r"\b", blob):
            _add(code, f"legal form '{form}' -> {code}")
    return {"ccs": ccs, "hits": hits}


def _clock_evidence(overlay_pairs: list[tuple[str, str]], claim_time_iso: str | None) -> "Evidence":
    """Compare burned-in camera timestamps against the claimed local time.

    Claim and overlay are both local-to-place times, so compare directly —
    no timezone math. Needs the same overlay string in >=2 *distinct* frames
    (same consensus rule as sign text; single frames can be misread — and a
    full-frame read + crop read of the same frame are correlated, not
    independent).
    """
    clean_pairs = [(f, t.strip()) for f, t in overlay_pairs
                   if t and t.strip()]
    clean = [t for _, t in clean_pairs]
    cited = sorted({f for f, _ in clean_pairs})
    if not claim_time_iso or not clean_pairs:
        return Evidence(
            type="clock",
            summary="no readable burned-in timestamp in frames" if not clean_pairs
                    else "no claimed time to check overlay against",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data={"overlays": clean},
            frames=cited,
        )
    frames_by_text: dict[str, set] = {}
    for f, t in clean_pairs:
        frames_by_text.setdefault(t, set()).add(f)
    stable = [t for t, fs in frames_by_text.items() if len(fs) >= 2]
    parsed = []
    for t in stable:
        dt, gran = _parse_overlay_time(t)
        if dt:
            parsed.append((t, dt, gran, len(frames_by_text[t])))
    if not parsed:
        return Evidence(
            type="clock",
            summary=(f"no ≥2-frame overlay consensus ({len(clean)} reads, "
                     f"{len(frames_by_text)} distinct) — refusing to date the clip")
                     if frames_by_text else "overlay text found but unparseable",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.2,
            data={"overlays": clean},
            frames=cited,
        )
    # Most-repeated timestamp wins — all frames of a real clip share one clock.
    raw, dt, gran, _ = max(parsed, key=lambda p: p[3])
    claim_dt = datetime.fromisoformat(claim_time_iso).replace(tzinfo=None)
    if gran == "datetime":
        delta_min = abs((dt - claim_dt).total_seconds()) / 60.0
    else:  # time-of-day only
        delta_min = abs((dt - claim_dt.replace(year=2000, month=1, day=1))
                        .total_seconds()) / 60.0
    if gran != "datetime":
        # A bare clock string carries no date anchor and the VLM fabricates
        # them readily — report it, but never let it carry a verdict.
        verdict, conf = ClaimVerdict.INCONCLUSIVE, 0.3
        note = f"overlay clock reads {raw} (time only — can't date the clip)"
    elif dt.date() != claim_dt.date():
        verdict, conf = ClaimVerdict.CONTRADICTS, 0.9
        note = f"camera overlay reads {raw} — different date than claimed"
    elif delta_min <= 45:
        verdict, conf = ClaimVerdict.SUPPORTS, 0.85
        note = f"camera overlay reads {raw} — matches the claimed time"
    elif delta_min >= 90:
        verdict, conf = ClaimVerdict.CONTRADICTS, 0.9
        note = (f"camera overlay reads {raw} but claim is "
                f"{claim_dt.isoformat()} — {delta_min:.0f} min off")
    else:
        verdict, conf = ClaimVerdict.INCONCLUSIVE, 0.5
        note = f"camera overlay reads {raw}; {delta_min:.0f} min from claim"
    return Evidence(
        type="clock", summary=note, verdict=verdict, confidence=conf,
        data={"overlays": clean, "parsed": raw, "delta_minutes": round(delta_min, 1)},
        frames=sorted(frames_by_text.get(raw, set())) or cited,
    )


def _claimed_utc(lat: float, lon: float, claim_time_iso: str,
                 arch: dict | None = None) -> tuple[str, str]:
    """Convert claimed local time -> UTC. Returns (utc_iso, note).

    Uses the archive's IANA timezone name via zoneinfo — the
    `utc_offset_seconds` field is resolved at request time and ignores DST
    changes within the range, which puts summer-date claims off by an hour.
    Falls back to utc_offset_seconds, then to treating the claim as UTC.
    """
    naive = datetime.fromisoformat(claim_time_iso).replace(tzinfo=None)
    try:
        from . import weather
        if arch is None:
            arch = weather.historical_weather(lat, lon, str(claim_time_iso)[:10])
        tz_name = arch.get("timezone")
        if tz_name:
            from zoneinfo import ZoneInfo
            utc = naive.replace(tzinfo=ZoneInfo(tz_name)).astimezone(timezone.utc)
            return utc.isoformat(), (
                f"claim local {naive.isoformat()} -> UTC via zoneinfo {tz_name}"
            )
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
    try:
        # Run audit trail — stage timings land in the report/sidecar.
        from . import run_stats
        run_stats.reset()
        run_stats.current().count("frames", 0)
    except Exception:
        pass
    try:
        # Memoize Nominatim lookups for the whole run — consensus-filtered
        # texts often re-geocode the same strings across stages.
        from . import geo as _geo, geocode_cache
        _geo.geocode = geocode_cache.wrap_geocode(_geo.geocode)
    except Exception:
        pass

    # --- 0. claim sanity -----------------------------------------------------
    if claim_time:
        try:
            if "T" not in str(claim_time) and ":" not in str(claim_time):
                raise ValueError("no time component")
            datetime.fromisoformat(str(claim_time).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            _note(pack, "claim", f"unusable claimed time {claim_time!r} — time checks skipped")
            claim_time = None

    # --- 1. resolve lat/lon (needed by solar + weather) ----------------------
    lat = lon = None
    try:
        lat, lon, coord_note = _resolve_coords(claim_place)
        if lat is None:
            _note(pack, "geo-resolve", coord_note)
    except Exception as e:
        _note(pack, "geo-resolve", e)

    # --- 2. frames (a still/dir of stills works too; drop stale frames) ------
    frames: list[str] = []
    try:
        from . import perception, stills
        for stale in glob.glob(os.path.join(workdir, "frame_*.jpg")):
            try:
                os.remove(stale)
            except OSError:
                pass
        frames = stills.frames_for_input(video_path, workdir)[:24]
        if not frames:
            frames = perception.extract_frames(video_path, workdir, fps=1.0, max_frames=24)
        try:
            run_stats.current().count("frames", len(frames))
        except Exception:
            pass
    except Exception as e:
        _note(pack, "frames", e)

    # Frame-quality triage — dark/blurry frames report a card and get
    # filtered out of the VLM pick list (never the cited-frame list).
    usable: list[str] | None = None
    if frames:
        try:
            from . import frame_quality
            scores = {f: frame_quality.frame_score(f) for f in frames}
            usable = frame_quality.usable_frames(frames)
            pack.add(frame_quality.quality_evidence(scores))
        except Exception:
            usable = None

    # --- 3. VLM observations (shadows, text, conditions) ---------------------
    texts: list[str] = []
    overlay_pairs: list[tuple[str, str]] = []  # (frame_path, overlay text)
    observations: dict = {}
    conditions_list: list[dict] = []
    osint_blobs: list[dict] = []      # per-frame OSINT scene-feature dicts
    geoguesses: list[dict] = []       # per-frame direct country guesses
    shadow_obs_pairs: list[tuple[str, dict]] = []  # (frame, shadow obs)
    marking_pairs: list[tuple[str, str]] = []      # (frame, marking text)
    sun_elevs: list[float] = []
    shadow_angles: list[float] = []  # shadow image-angle per frame, for heading
    dets_cache: dict[str, list[dict]] = {}
    picks: list[str] = []

    if frames and use_vlm and os.environ.get("COSMOS_BASE_URL"):
        try:
            cosmos = perception.CosmosClient()
            try:
                from . import cache
                cosmos = cache.wrap_cosmos_client(cosmos)
            except Exception:
                pass
            try:
                pool = usable if usable else frames
                picks = perception.pick_shadow_frames(pool)[:max_vlm_frames]
                if not picks:
                    picks = pool[:max_vlm_frames]
            except Exception:
                picks = frames[:max_vlm_frames]

            cv_tilt: float | None = None
            for f in picks:
                try:
                    run_stats.current().count("vlm_calls", 6)
                    obs = cosmos.locate_shadow(f)
                    if obs:
                        # cv2 horizon tilt beats the VLM's guessed scalar when
                        # it's confident — it rescales shadow geometry.
                        if cv_tilt is None:
                            try:
                                from . import horizon
                                est = horizon.estimate_tilt_deg(f)
                                if est and (est.get("confidence") or 0) >= 0.45:
                                    cv_tilt = float(est["tilt_deg"])
                            except Exception:
                                pass
                        if cv_tilt is not None:
                            obs = {**obs, "camera_tilt_deg": cv_tilt}
                        pack.add(perception.to_evidence(f, obs))
                        shadow_obs_pairs.append((f, obs))
                        if f not in dets_cache:
                            try:
                                dets_cache[f] = perception.detect_objects(f)
                            except Exception:
                                dets_cache[f] = []
                        elev = _sun_elevation_deg(obs, dets_cache[f], f)
                        if elev is not None:
                            sun_elevs.append(elev)
                        try:
                            v = perception.shadow_vector(obs)
                            if v.get("angle_deg") is not None:
                                shadow_angles.append(float(v["angle_deg"]))
                        except Exception:
                            pass

                    txt = cosmos.read_scene_text(f)
                    if txt:
                        texts.extend(_as_str_list(txt.get("texts")))
                        overlay_pairs.extend(
                            (f, t) for t in _as_str_list(txt.get("overlay_timestamps")))
                        for k in ("driving_side", "sign_language", "plate_style"):
                            if not observations.get(k) and txt.get(k):
                                observations[k] = txt[k]
                        marking_pairs.extend(
                            (f, m) for m in _as_str_list(txt.get("road_markings")))

                    cond = cosmos.describe_conditions(f)
                    if cond:
                        conditions_list.append(cond)

                    clk = cosmos.read_clock_overlay(f)
                    if clk:
                        overlay_pairs.extend(
                            (f, t) for t in _as_str_list(clk.get("overlay_timestamps")))

                    # OSINT feature pass — scene cues + quarantined guess.
                    feat = cosmos.scene_features(f)
                    if feat:
                        osint_blobs.append(feat)
                    gg = cosmos.geoguess(f)
                    if gg:
                        geoguesses.append(gg)
                except Exception:
                    continue  # one bad frame must not sink the rest
        except Exception as e:
            _note(pack, "vlm", e)
    elif use_vlm:
        _note(pack, "vlm", "COSMOS_BASE_URL not set — VLM stages skipped")
    if frames and use_vlm and os.environ.get("COSMOS_BASE_URL") and not picks:
        _note(pack, "vlm", "0 usable frames picked — VLM stages skipped")

    # Classical OCR fallback for burned-in timestamps the VLM can't resolve.
    if picks:
        try:
            from . import ocr
            if ocr.ocr_available():
                covered = {f for f, _ in overlay_pairs}
                for f in picks:
                    if f in covered:
                        continue
                    for ts in ocr.extract_timestamps(ocr.read_overlay_text(f) or ""):
                        overlay_pairs.append((f, ts))
        except Exception:
            pass

    # --- 3b. scene gate — flag conditions that poison specific stages --------
    applicability = {"solar": True, "shadows": True, "geo": True,
                     "weather": True, "clock": True, "audio": True}
    try:
        from . import scene_gate
        all_dets = [d for ds in dets_cache.values() for d in ds]
        flags = scene_gate.scene_flags(conditions_list, texts, all_dets)
        applicability = scene_gate.evidence_applicability(flags)
        gate = scene_gate.gate_evidence(flags)
        if gate:
            pack.add(gate)
    except Exception:
        pass

    if conditions_list:
        cond0 = conditions_list[0]
        pack.add(Evidence(
            type="texture",
            summary=(
                f"scene conditions observed: sky={cond0.get('sky')}, "
                f"lighting={cond0.get('lighting')}, "
                f"precip={cond0.get('precipitation')}"
            ),
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.3,
            data={"conditions": conditions_list},
        ))

    # --- 4. geolocation ------------------------------------------------------
    # Hallucination guard: only geocode strings seen in >=2 frames — invented
    # sign text doesn't repeat across frames, real text does. Markings get
    # the same rule: observations feeds consistency checks + cue scoring.
    if marking_pairs:
        markings = _consensus_pairs(marking_pairs)
        if markings:
            observations["road_markings"] = markings
    stable_texts: list[str] = []
    try:
        from collections import Counter
        from . import geo
        counts = Counter(t.strip().lower() for t in texts if t and t.strip())
        consensus = [t for t in texts
                     if t and counts[t.strip().lower()] >= 2]
        seen: set[str] = set()
        stable_texts = [t for t in consensus
                        if not (t.strip().lower() in seen or seen.add(t.strip().lower()))]
        if applicability.get("geo"):
            ev = geo.geo_evidence(claim_place, stable_texts, observations)
            if len(texts) != len(stable_texts):
                ev.data["texts_filtered"] = len(texts) - len(stable_texts)
            pack.add(ev)
    except Exception as e:
        _note(pack, "geo", e)

    # Language-of-signs as a cheap geo corroborator (script, not NLP).
    try:
        from . import langdetect
        lev = langdetect.language_evidence(stable_texts, claim_place)
        if lev and lev.summary:
            pack.add(lev)
    except Exception:
        pass

    # Licence-plate style from detected vehicles (EU blue band etc.).
    if dets_cache:
        try:
            from . import plate_style
            crops = []
            for f, dets in dets_cache.items():
                for d in plate_style.vehicle_bboxes(dets):
                    r = plate_style.analyze_vehicle_crop(f, d.get("bbox_xyxy"))
                    if r:
                        crops.append(r)
            if crops:
                pack.add(plate_style.plate_evidence(crops, claim_place))
        except Exception:
            pass

    # --- 4b. OSINT geo-matching -----------------------------------------------
    # Scene features -> candidate countries -> weighted cue scoring. Each
    # side-signal (landmarks, guesses, sign style, vegetation, POIs, wiki)
    # emits its own card; osint.osint_evidence fuses them into the headline
    # "does this scene match the claimed country" verdict.
    osint_vec: dict = {}
    claimed_cc: str | None = None
    anchor = (lat, lon) if lat is not None and lon is not None else None
    try:
        from . import osint_features
        merged_feats = osint_features.merge_features(osint_blobs)
        if merged_feats.get("frames"):
            pack.add(osint_features.features_evidence(merged_feats))
        osint_vec = osint_features.feature_vector(merged_feats)
        # Fold the text-stage observations into the vector for scoring.
        for k in ("driving_side", "sign_language", "plate_style"):
            if observations.get(k):
                osint_vec[k] = observations[k]
        # Markings get the same >=2-frame consensus as sign text — one
        # frame's hallucinated "double yellow" must not score candidates.
        if observations.get("road_markings"):
            osint_vec["road_markings"] = observations["road_markings"]
        # Country-indicative tokens in consensus text (dial codes, ccTLDs,
        # currency, legal suffixes) — free high-specificity geo hint.
        hints = _text_geo_hints(stable_texts)
        if hints["ccs"]:
            osint_vec["geocode_ccs"] = hints["ccs"]
            osint_vec["country_indicators"] = hints["ccs"]
            osint_vec["country_indicator_notes"] = hints["hits"]
    except Exception:
        pass
    try:
        if anchor:
            from . import reverse_geo
            claimed_cc = reverse_geo.claim_country_cc(claim_place, lat, lon)
        if not claimed_cc and claim_place:
            from . import geo as _g
            c = _g.geocode(claim_place)
            if c:
                claimed_cc = getattr(c[0], "country_code", None)
    except Exception:
        pass
    try:
        if osint_vec or observations:
            from . import match_score
            ccs = {c for c in [claimed_cc] if c}
            ccs |= set(osint_vec.get("geocode_ccs") or [])
            ranked = match_score.score_candidates(
                {**osint_vec, **{k: observations[k] for k in
                 ("driving_side", "sign_language", "plate_style")
                 if observations.get(k)}},
                sorted(ccs) or ["US", "GB", "DE", "FR", "IT", "ES", "NL", "JP", "AU", "CA"])
            if ranked:
                pack.add(match_score.match_evidence(ranked, claimed_cc))
                osint_vec["_ranked"] = ranked[:8]
    except Exception:
        pass
    try:
        from . import osint as _osint
        ev = _osint.osint_evidence(
            osint_vec, observations, anchor, claimed_cc,
            per_frame_geoguesses=geoguesses or None,
            claim_time_iso=claim_time)
        if ev is not None:
            pack.add(ev)
    except Exception:
        pass
    try:
        from . import landmarks
        lnames = landmarks.consensus_landmarks(
            [b.get("landmark_candidates") for b in osint_blobs
             if isinstance(b, dict)])
        if lnames:
            places = landmarks.landmark_candidates_to_places(lnames)
            pack.add(landmarks.landmark_evidence(lnames, places, anchor, claimed_cc))
    except Exception:
        pass
    try:
        from . import geoguess as _gg
        if geoguesses:
            pack.add(_gg.geoguess_evidence(
                _gg.aggregate_guesses(geoguesses), claimed_cc))
    except Exception:
        pass
    try:
        from . import sign_style as _ss
        ss_scores = _ss.sign_style_score(
            (osint_vec.get("sign_style") or {}) if isinstance(osint_vec.get("sign_style"), dict) else {},
            osint_vec.get("marking_colors"))
        if ss_scores:
            pack.add(_ss.sign_style_evidence(ss_scores, claimed_cc))
    except Exception:
        pass
    try:
        from . import veg_geo
        if osint_vec.get("vegetation"):
            pack.add(veg_geo.veg_evidence(
                osint_vec["vegetation"], lat, claimed_cc))
    except Exception:
        pass
    try:
        from . import hemisphere, sun_side
        sides = [sun_side.sun_side_from_conditions(c) for c in conditions_list]
        side = sun_side.aggregate_sides(sides)
        if side and claim_time:
            pack.add(hemisphere.hemisphere_evidence(lat, lon, claim_time, side))
    except Exception:
        pass
    # External cross-checks — only when the claim anchored to real coords.
    if anchor:
        names_for_lookup = (osint_vec.get("landmark_candidates") or stable_texts or [])[:6]
        try:
            from . import overpass
            pois = overpass.nearby_pois(lat, lon)
            if pois:
                m = overpass.name_matches_nearby(pois, names_for_lookup)
                pack.add(overpass.poi_evidence(m, True, anchor))
        except Exception:
            pass
        try:
            from . import wiki_places
            titles = wiki_places.geo_search(lat, lon)
            if titles:
                m = wiki_places.match_named(titles, names_for_lookup)
                pack.add(wiki_places.wiki_evidence(m, True, 10000))
        except Exception:
            pass
        try:
            from . import climate
            if osint_vec.get("climate_cues"):
                pack.add(climate.climate_evidence(
                    lat, lon, osint_vec["climate_cues"]))
        except Exception:
            pass
    try:
        from . import camera_meta
        meta = {}
        if frames:
            meta.update(camera_meta.frame_metadata(frames[0]) or {})
        meta.update(camera_meta.video_metadata(video_path) or {})
        if meta:
            pack.add(camera_meta.meta_evidence(meta, claim_time, anchor))
    except Exception:
        pass
    try:
        from . import shadow_consistency
        per_frame = shadow_consistency.shadow_angles_per_frame(shadow_obs_pairs)
        if per_frame:
            pack.add(shadow_consistency.consistency_evidence(per_frame))
    except Exception:
        pass
    try:
        from . import vss_lookup
        keys = vss_lookup.list_segment_keys()
        if keys:
            pack.add(vss_lookup.vss_evidence(keys, claim_time))
    except Exception:
        pass
    # Geo-family digest when the scene produced many geo cards.
    try:
        from . import dedupe_evidence
        roll = dedupe_evidence.geo_rollup(pack.evidence)
        if roll is not None:
            pack.add(roll)
    except Exception:
        pass

    # --- 4c. burned-in clock overlay vs claim ---------------------------------
    try:
        pack.add(_clock_evidence(overlay_pairs, claim_time))
    except Exception as e:
        _note(pack, "clock", e)

    # --- 5. weather + solar (need coords + claimed time) ----------------------
    if lat is not None and claim_time:
        # One archive fetch feeds both the weather card and the tz conversion.
        arch = None
        try:
            from . import weather
            arch = weather.historical_weather(lat, lon, str(claim_time)[:10])
        except Exception:
            arch = None
        try:
            if applicability.get("weather"):
                weather_obs = (
                    _norm_conditions(_majority_conditions(conditions_list))
                    if conditions_list else {})
                pack.add(weather.compare_weather(
                    weather_obs, lat, lon, claim_time, archive=arch))
        except Exception as e:
            _note(pack, "weather", e)
        # Timezone sanity — offline coord->zone table vs the archive's zone.
        try:
            from . import tz_check
            pack.add(tz_check.tz_evidence(
                lat, lon, (arch or {}).get("timezone"), claim_time))
        except Exception:
            pass

        if sun_elevs and applicability.get("shadows"):
            try:
                from . import solar
                claimed_utc, tz_note = _claimed_utc(lat, lon, claim_time, arch)
                # Median over frames — single-frame point picks are noisy at
                # the small object sizes these cams resolve.
                s = sorted(sun_elevs)
                sun_elev = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
                ev = solar.solar_evidence(lat, lon, claimed_utc, sun_elev)
                ev.data["claim_tz_note"] = tz_note
                ev.data["sun_elevation_source"] = "shadow/object pixel ratio"
                ev.data["sun_elevations_deg"] = [round(e, 1) for e in sun_elevs]
                # Heading hypothesis from shadow azimuths — annotation only.
                try:
                    from . import azimuth
                    a = sorted(shadow_angles)
                    med_ang = a[len(a) // 2] if a else None
                    if med_ang is not None:
                        hh = azimuth.heading_hint(
                            lat, lon, claimed_utc[:10], med_ang)
                        if hh is not None:
                            ev.data["implied_camera_heading_deg"] = hh
                except Exception:
                    pass
                pack.add(ev)
            except Exception as e:
                _note(pack, "solar", e)

        # Lunar check — free corroboration when a clock claim exists.
        try:
            from . import moonphase
            claimed_utc, _ = _claimed_utc(lat, lon, claim_time, arch)
            mev = moonphase.moon_evidence(claimed_utc, None)
            if mev:
                pack.add(mev)
        except Exception:
            pass
    elif claim_time or claim_place:
        _note(pack, "weather+solar",
              "need both resolvable coordinates and a claimed time")

    # Ambience probe — soft prior only, never drives the verdict alone.
    try:
        from . import audio_probe, stills as _stills
        if not _stills.is_still(video_path):
            pack.add(audio_probe.audio_evidence(video_path))
    except Exception:
        pass

    # Tag each card with its provenance channel before adjudication.
    try:
        from . import provenance
        pack.evidence = provenance.tag(pack.evidence)
    except Exception:
        pass

    adj = adjudicate(pack)
    # Geo context for the report layer (plain attrs — asdict() ignores them).
    try:
        adj.anchor = anchor
        adj.claim_place = claim_place
        adj.osint_ranked = osint_vec.get("_ranked") or []
    except Exception:
        pass
    return adj
