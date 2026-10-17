"""Geoguess v2 — observation-first prompt + response normalizer.

A sibling to osint_prompts.GEOGUESS_PROMPT applying the sgoedecke
GeoGuessr-protocol findings — selectively, because the same research
shows long elaborate protocols make VLMs *less* accurate. Only the parts
that measurably help are kept, stripped down:

  * raw observations BEFORE interpretation — the model must catalogue
    what it literally sees (driving side, plate colours, pole material,
    readable text) before it is allowed to name a place;
  * exactly TWO hypotheses — forcing a second candidate curbs the
    single-guess anchoring that makes v1 answers over-confident;
  * a spread constraint — the two hypotheses must be >= ~150 km apart,
    so "Germany" / "Bavaria" cannot pose as two guesses;
  * honesty caps — confidence >0.6 only with a named specific cue,
    generic suburbia capped at 0.35, null when unknowable.

Everything else (long checklists, continent-by-continent walkthroughs)
is deliberately left out.

The response side mirrors geoguess.py's quarantine posture: the
normalizer distrusts the model's own confidence (clamps >0.6 without
key_evidence) and drops empty hypotheses. ``to_frame_guess`` folds a
normalized response into geoguess.py's per-frame {frame, country_guess,
region_guess, confidence, reasoning} shape so it drops into
aggregate_guesses unchanged.

Stdlib only at import; nothing here raises — malformed input -> {}.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata

_MAX_OBS = 6             # prompt allows at most 6 literal observations
_MAX_HYPOTHESES = 2      # exactly two hypotheses; extras are discarded
_HIGH_CONF = 0.6         # > this requires a named key_evidence cue

# Values meaning "the model abstained" — treated as no-answer. Mirrors
# geoguess._ABSTAIN; duplicated to keep this module dependency-free.
_ABSTAIN = frozenset({
    "unknown", "none", "null", "na", "n a", "n/a", "unclear", "unsure",
    "cant tell", "can't tell", "cannot determine", "unidentifiable",
})

GEOGUESS_V2_PROMPT = """\
You are a forensic geolocation analyst estimating where this image was
taken. Work like a careful GeoGuessr player: FIRST catalogue what is
literally visible, THEN weigh exactly two competing region hypotheses —
never jump straight to a guess.

Return STRICT JSON with this exact schema — no prose, no markdown fences:

{"observations": ["<literal visible fact>", "<...>"], "hypotheses": [{"country_guess": "<common English country name or null>", "region_guess": "<state/province/city or null>", "confidence": <bare number 0.0-1.0>, "key_evidence": "<the single observation that most supports this hypothesis, or null>"}, {"country_guess": "<... or null>", "region_guess": "<... or null>", "confidence": <0.0-1.0>, "key_evidence": "<... or null>"}], "best_guess": {"country_guess": "<common English country name or null>", "confidence": <0.0-1.0>}}

- "observations": up to 6 concrete facts you can literally SEE — e.g.
  "left-side driving", "yellow rear plates", "concrete utility poles",
  "german text on shopfront", "white centre lane markings". Facts only:
  NO country names, no interpretation. Report text or language only when
  it is literally readable in the frame — NEVER invent sign text, place
  names or a language you cannot actually read.
- "hypotheses": EXACTLY 2 entries, most plausible first. The two must be
  genuinely different candidate REGIONS at least ~150 km apart (two
  countries, or distant regions of one large country) — never the same
  place renamed. Use null country_guess/region_guess only when you
  honestly cannot name even a second candidate.
- "confidence" (each hypothesis and best_guess) reflects evidence
  SPECIFICITY, not how neat the scene looks: <=0.35 for generic
  suburbia/fields/malls with no distinctive cue, 0.35-0.6 when several
  weak cues agree, >0.6 ONLY with highly specific evidence (readable
  place-name text, a flag, a plate format unique to one country).
  Confidence above 0.6 requires a non-null "key_evidence" naming that
  specific cue.
- "best_guess": the single country you would commit to — normally the
  higher-confidence hypothesis. null + 0.0 when the scene could be
  almost anywhere; an honest null beats a confident wrong guess.

Respond with JSON only."""

# Top-level keys a conforming v2 response must carry — for consumer-side
# validation. Hypothesis objects carry _HYPOTHESIS_KEYS; best_guess
# carries {country_guess, confidence}.
RESPONSE_SCHEMA: dict = {
    "observations": list,   # [str, ...] <= _MAX_OBS literal visible facts
    "hypotheses": list,     # exactly _MAX_HYPOTHESES dicts
    "best_guess": dict,     # {country_guess: str|null, confidence: number}
}

_HYPOTHESIS_KEYS = ("country_guess", "region_guess", "confidence",
                    "key_evidence")


# --------------------------------------------------------------------------
# Small helpers — same style as geoguess.py's normalization helpers
# --------------------------------------------------------------------------

def _norm_key(value) -> str:
    """Fold a place string for comparison: diacritics off, lowercase,
    punctuation -> space ("Côte d'Ivoire" -> "cote d ivoire")."""
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


def _to_conf(value) -> float | None:
    """Best-effort confidence parse -> 0..1; tolerates percent-style (>1)
    and numeric strings. None when unparseable/non-finite."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if f > 1.0:
        f /= 100.0
    return max(0.0, min(1.0, f))


def _clean_place(value) -> str | None:
    """Collapse whitespace / edge punctuation on a place name; abstain
    words ("unknown", "none", ...) map to None."""
    s = " ".join(str(value or "").split()).strip(" ,.;-")
    if not s or s.lower() in _ABSTAIN:
        return None
    return s


def _clean_evidence(value) -> str | None:
    """key_evidence as one string — tolerates a bare string or a list of
    strings (joined "; ")."""
    if isinstance(value, (list, tuple)):
        parts = [_clean_place(v) for v in value]
        s = "; ".join(p for p in parts if p)
    else:
        s = _clean_place(value)
    return s or None


def _find_hyp(hyps, country) -> dict | None:
    """First hypothesis whose country matches `country` (folded)."""
    key = _norm_key(country)
    if not key:
        return None
    for h in hyps or []:
        if isinstance(h, dict) and _norm_key(h.get("country_guess")) == key:
            return h
    return None


# --------------------------------------------------------------------------
# Normalizer — distrust the model, keep what parses, drop the rest
# --------------------------------------------------------------------------

def normalize_v2(resp: dict) -> dict:
    """Validate/clean a parsed GEOGUESS_V2_PROMPT response.

    Steps: coerces confidences to 0..1 (percent-style >1 tolerated),
    clamps confidence >0.6 that lacks key_evidence, drops empty/abstain
    hypotheses, keeps the exactly-two structure when both parse (extras
    trimmed). A raw JSON string is tolerated and parsed first.

    -> {"observations": [str, ...],
        "hypotheses": [{country_guess, region_guess, confidence,
                        key_evidence}, ... <=2],
        "best": {"country_guess": <clean name, ready for
                 geoguess.resolve_cc>, "confidence": float}}

    "best" is the model's best_guess when usable (confidence borrowed
    from the matching hypothesis when unstated), else the top-confidence
    hypothesis. Returns {} when the input is malformed or carries
    nothing usable — never raises.
    """
    try:
        if isinstance(resp, (str, bytes)):
            try:
                resp = json.loads(resp)
            except Exception:
                return {}
        if not isinstance(resp, dict):
            return {}
        # A response without ANY guess structure isn't a v2 geoguess.
        if "hypotheses" not in resp and "best_guess" not in resp:
            return {}

        # --- observations: deduped short strings, capped -------------
        obs: list[str] = []
        raw_obs = resp.get("observations")
        if isinstance(raw_obs, str):
            raw_obs = [raw_obs]
        if isinstance(raw_obs, (list, tuple)):
            for item in raw_obs:
                if isinstance(item, str):
                    s = " ".join(item.split()).strip()
                elif isinstance(item, (int, float)):
                    s = str(item)
                else:
                    continue
                if s and s not in obs:
                    obs.append(s)
                if len(obs) >= _MAX_OBS:
                    break

        # --- hypotheses: cleaned, empties dropped, capped at 2 -------
        hyps: list[dict] = []
        raw_h = resp.get("hypotheses")
        if isinstance(raw_h, dict):
            raw_h = [raw_h]
        if isinstance(raw_h, (list, tuple)):
            for h in raw_h:
                if not isinstance(h, dict):
                    continue
                country = _clean_place(h.get("country_guess"))
                region = _clean_place(h.get("region_guess"))
                evidence = _clean_evidence(h.get("key_evidence"))
                if country is None and region is None and not evidence:
                    continue  # empty hypothesis (incl. pure abstentions)
                conf = _to_conf(h.get("confidence"))
                conf = conf if conf is not None else 0.0
                if conf > _HIGH_CONF and not evidence:
                    conf = _HIGH_CONF
                hyps.append({
                    "country_guess": country,
                    "region_guess": region,
                    "confidence": round(conf, 3),
                    "key_evidence": evidence,
                })
                if len(hyps) >= _MAX_HYPOTHESES:
                    break

        # --- best: explicit best_guess, else top hypothesis ----------
        best = {"country_guess": None, "confidence": 0.0}
        raw_best = resp.get("best_guess")
        if not isinstance(raw_best, dict):
            raw_best = {}
        b_country = _clean_place(raw_best.get("country_guess"))
        b_conf = _to_conf(raw_best.get("confidence"))
        if b_country:
            match = _find_hyp(hyps, b_country)
            evidence = match.get("key_evidence") if match else None
            conf = (b_conf if b_conf is not None
                    else (match["confidence"] if match else 0.0))
            if conf > _HIGH_CONF and not evidence:
                conf = _HIGH_CONF
            best["country_guess"] = b_country
            best["confidence"] = round(conf, 3)
        elif hyps:
            top = max(hyps, key=lambda h: h["confidence"])
            best["country_guess"] = top["country_guess"]
            best["confidence"] = top["confidence"]

        if not obs and not hyps and not best["country_guess"]:
            return {}
        return {"observations": obs, "hypotheses": hyps, "best": best}
    except Exception:
        return {}


def to_frame_guess(norm: dict, frame: str) -> dict:
    """Fold a normalize_v2 result into geoguess.py's per-frame guess shape.

    -> {"frame": frame, "country_guess", "region_guess", "confidence",
        "reasoning"} — a drop-in row for geoguess.aggregate_guesses.
    region_guess comes from the hypothesis matching best's country
    (never borrowed across hypotheses — a French region under a German
    guess is worse than none). reasoning is the observations joined
    "; ". A null best becomes a null country_guess, which the
    aggregator counts as an abstention. Never raises.
    """
    out = {"frame": frame, "country_guess": None, "region_guess": None,
           "confidence": 0.0, "reasoning": ""}
    try:
        norm = norm if isinstance(norm, dict) else {}
        best = norm.get("best")
        best = best if isinstance(best, dict) else {}
        country = _clean_place(best.get("country_guess"))
        conf = _to_conf(best.get("confidence"))
        out["country_guess"] = country
        out["confidence"] = round(conf if conf is not None else 0.0, 3)
        if country:
            match = _find_hyp(norm.get("hypotheses") or [], country)
            if match:
                out["region_guess"] = match.get("region_guess")
        obs = [o.strip() for o in (norm.get("observations") or [])
               if isinstance(o, str) and o.strip()]
        out["reasoning"] = "; ".join(obs)
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Self-test: python3 sundial/prompt_geo_v2.py
# --------------------------------------------------------------------------

if __name__ == "__main__":
    fails = 0

    def check(name, got, want):
        global fails
        ok = got == want
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: "
              f"got {got!r} (want {want!r})")

    # --- schema sanity ------------------------------------------------
    check("schema keys", sorted(RESPONSE_SCHEMA),
          ["best_guess", "hypotheses", "observations"])
    check("prompt names every schema key",
          all(k in GEOGUESS_V2_PROMPT for k in RESPONSE_SCHEMA), True)

    # --- a well-formed DE/FR pair normalizes --------------------------
    resp = {
        "observations": [
            "german text on shopfront", "eu blue plate band",
            "concrete utility poles", "right-side driving",
        ],
        "hypotheses": [
            {"country_guess": "Germany", "region_guess": "Bavaria",
             "confidence": 0.55, "key_evidence": "german text on shopfront"},
            {"country_guess": "France", "region_guess": "Alsace",
             "confidence": 0.3, "key_evidence": "vienna-style signage"},
        ],
        "best_guess": {"country_guess": "Germany", "confidence": 0.55},
    }
    norm = normalize_v2(resp)
    check("observations kept", norm["observations"], resp["observations"])
    check("two hypotheses", len(norm["hypotheses"]), 2)
    check("hyp0 country", norm["hypotheses"][0]["country_guess"], "Germany")
    check("hyp1 country", norm["hypotheses"][1]["country_guess"], "France")
    check("hyp keys", sorted(norm["hypotheses"][0]),
          sorted(_HYPOTHESIS_KEYS))
    check("best country", norm["best"]["country_guess"], "Germany")
    check("best conf", norm["best"]["confidence"], 0.55)

    # --- honesty caps: >0.6 without key_evidence is clamped -----------
    cocky = {
        "observations": ["generic suburb"],
        "hypotheses": [
            {"country_guess": "United States", "region_guess": None,
             "confidence": 0.95, "key_evidence": None},
            {"country_guess": "Canada", "region_guess": "Ontario",
             "confidence": "72", "key_evidence": "street layout"},
        ],
        "best_guess": {"country_guess": "United States", "confidence": 0.95},
    }
    n2 = normalize_v2(cocky)
    check("conf>0.6 w/o evidence clamped",
          n2["hypotheses"][0]["confidence"], 0.6)
    check("percent-string conf coerced",
          n2["hypotheses"][1]["confidence"], 0.72)
    check("best conf clamped w/o evidence", n2["best"]["confidence"], 0.6)

    # --- empties dropped, extras trimmed, conf borrowed ---------------
    messy = {
        "observations": ["red soil", "", None, "termite mounds", "red soil"],
        "hypotheses": [
            {"country_guess": None, "region_guess": None,
             "confidence": 0.5, "key_evidence": None},          # empty
            {"country_guess": "Australia", "region_guess": "Queensland",
             "confidence": 0.4, "key_evidence": "red soil"},
            {"country_guess": "South Africa", "region_guess": "Limpopo",
             "confidence": 0.35, "key_evidence": "termite mounds"},
            {"country_guess": "Brazil", "region_guess": "Minas Gerais",
             "confidence": 0.3, "key_evidence": "red soil"},     # 3rd
        ],
        "best_guess": {"country_guess": "Australia"},            # no conf
    }
    n3 = normalize_v2(messy)
    check("empty/dup obs dropped",
          n3["observations"], ["red soil", "termite mounds"])
    check("empty hypothesis dropped + 3rd trimmed",
          [h["country_guess"] for h in n3["hypotheses"]],
          ["Australia", "South Africa"])
    check("best conf borrowed from matching hyp",
          n3["best"]["confidence"], 0.4)

    # --- abstentions: "unknown" guesses drop like nulls ---------------
    abstain = {
        "observations": ["generic two-storey houses"],
        "hypotheses": [
            {"country_guess": "unknown", "region_guess": None,
             "confidence": 0.1, "key_evidence": None},           # abstain
            {"country_guess": None, "region_guess": "somewhere temperate",
             "confidence": 0.2, "key_evidence": None},          # region-only
        ],
        "best_guess": {"country_guess": None, "confidence": 0.0},
    }
    n4 = normalize_v2(abstain)
    check("abstain country dropped", len(n4["hypotheses"]), 1)
    check("region-only hyp kept",
          n4["hypotheses"][0]["region_guess"], "somewhere temperate")
    check("best falls back to top hyp (null country)",
          n4["best"]["country_guess"], None)
    check("best conf from top hyp", n4["best"]["confidence"], 0.2)

    # --- malformed input -> {} ----------------------------------------
    check("missing keys -> {}", normalize_v2({"country_guess": "France"}), {})
    check("bad JSON string -> {}", normalize_v2("{not json"), {})
    check("non-dict -> {}", normalize_v2([1, 2]), {})
    check("None -> {}", normalize_v2(None), {})
    check("fully empty -> {}", normalize_v2(
        {"observations": [], "hypotheses": [], "best_guess": {}}), {})
    check("JSON string input parses",
          normalize_v2(json.dumps(resp))["best"]["country_guess"], "Germany")

    # --- to_frame_guess: geoguess.py per-frame shape -------------------
    fg = to_frame_guess(norm, "frame_001.jpg")
    check("frame passthrough", fg["frame"], "frame_001.jpg")
    check("frame country", fg["country_guess"], "Germany")
    check("frame region from matching hyp", fg["region_guess"], "Bavaria")
    check("frame conf", fg["confidence"], 0.55)
    check("reasoning = obs joined",
          fg["reasoning"],
          "german text on shopfront; eu blue plate band; "
          "concrete utility poles; right-side driving")
    check("empty norm -> abstain row",
          to_frame_guess({}, "x.jpg")["country_guess"], None)
    check("best w/o matching hyp -> region None",
          to_frame_guess({"best": {"country_guess": "Japan",
                                   "confidence": 0.5},
                          "hypotheses": [], "observations": []},
                         "y.jpg")["region_guess"], None)

    # --- everything out of the module is JSON-safe ---------------------
    try:
        for blob in (norm, n2, n3, n4, fg):
            json.dumps(blob)
        check("outputs JSON-safe", True, True)
    except Exception:
        check("outputs JSON-safe", False, True)

    print(f"prompt_geo_v2 self-test: "
          f"{'all checks passed' if not fails else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)
