"""OSINT scene-feature stage — merge per-frame VLM blobs into consensus cues.

A VLM prompt (sibling module osint_prompts.OSINT_SCENE_PROMPT) asks the model
for a strict-JSON description of each sampled frame: architecture,
vegetation, terrain, road furniture, sign style, utility poles,
lane-marking colour, climate cues, vehicles and landmark candidates.

One frame's answer is a guess; the *consensus* across frames is the signal
— a palm tree in one frame is noise, palms in 4/6 frames is a feature. This
module merges the blobs (modal scalars, deduped union lists, per-field
source counts), normalizes that into a scoring-friendly feature vector, and
packages it as a descriptive Evidence card. It never decides a verdict —
these features are input for the adjudicator, not a contradiction alone.

Pure stdlib at import time; schema types are imported lazily.
"""
from __future__ import annotations

import json
import re
from collections import Counter

# Normalized tokens that carry no information — the VLM's ways of saying
# "I can't tell". Dropped alongside non-strings and empty strings.
_JUNK_TOKENS = frozenset({
    "", "null", "none", "n/a", "na", "unknown", "unk", "unclear", "-", "--",
})

# Words we treat as colours when mining lane-marking descriptions.
_COLOR_WORDS = (
    "white", "yellow", "red", "blue", "orange", "green", "black", "grey", "gray",
)
_COLOR_RE = re.compile(r"\b(" + "|".join(_COLOR_WORDS) + r")\b")

# Merged-field keys consulted when a caller wants languages/scripts.
# The base prompt schema has no such key, but the merge is generic — these
# names are picked up when present (e.g. a prompt variant adds them).
_LANGUAGE_KEYS = ("languages_scripts", "languages", "language", "sign_language")
_SCRIPT_KEYS = ("scripts", "script", "sign_script", "sign_scripts")
_CONVENTION_KEYS = ("sign_conventions", "sign_convention", "convention")
_MARKING_KEYS = ("marking_colors", "lane_markings_color", "road_markings_color")

# Top-level keys the OSINT_SCENE_PROMPT schema can legitimately return.
# _first_json_object scans forward on parse failure, so a truncated response
# can surface a NESTED object (e.g. {"shape":..., "convention_guess":...})
# as if it were the whole blob. Requiring >=1 expected key quarantines that
# fragment instead of polluting fields named "shape"/"colors".
_SCHEMA_KEYS = frozenset({
    "architecture", "vegetation", "terrain", "road_furniture", "sign_style",
    "utility_poles", "lane_markings_color", "climate_cues", "vehicles",
    "landmark_candidates", "driving_side", "sign_language", "plate_style",
    "road_markings", "texts", "languages", "scripts",
})


def _looks_like_blob(blob: dict) -> bool:
    """Blob sanity: at least one expected schema key present."""
    return any(
        isinstance(k, str) and k.strip().lower() in _SCHEMA_KEYS
        for k in blob
    )


# --------------------------------------------------------------------------
# Value normalization
# --------------------------------------------------------------------------

def _norm_str(v) -> str | None:
    """Normalize a string for consensus: lowercase, single-spaced.

    Returns None for junk — non-strings, empties, and null-ish tokens.
    """
    if not isinstance(v, str):
        return None
    s = re.sub(r"\s+", " ", v.strip().lower())
    return s if s not in _JUNK_TOKENS else None


def _norm_dict(d: dict) -> dict:
    """Normalize a string-valued sub-dict (a vehicle, a sign_style row)."""
    out = {}
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        nk = re.sub(r"\s+", " ", k.strip().lower())
        if not nk:
            continue
        if isinstance(v, str):
            nv = _norm_str(v)
            if nv:
                out[nk] = nv
        elif isinstance(v, (list, tuple)):
            items = sorted({_norm_str(x) for x in v} - {None})
            if items:
                out[nk] = ", ".join(items)
    return out


def _ordered(counts: Counter) -> dict:
    """Counter -> plain dict ordered by (-count, token) for determinism."""
    return {k: n for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}


def _top(counts: Counter) -> str | None:
    """Modal value of a counter; ties break alphabetically (deterministic)."""
    if not counts:
        return None
    return next(iter(_ordered(counts)), None)


# --------------------------------------------------------------------------
# merge_features
# --------------------------------------------------------------------------

def merge_features(blobs: list[dict]) -> dict:
    """Merge N per-frame OSINT blobs into one consensus feature dict.

    Every field that produced at least one usable value becomes an entry:

        {"sources": <frames reporting it>, ...payload...}

    Payload depends on the observed shape:
      * scalar strings  -> {"value": modal, "counts": {value: frames}}
      * lists of str    -> {"values": [deduped, freq-ordered], "counts": {...}}
      * dicts           -> {"value": {subkey: modal|[values]}, "counts": {...}}
      * lists of dicts  -> {"values": [deduped dicts], "counts": {json: n}}

    Values are case-normalized (lower, single-spaced); non-strings, empties
    and null-ish tokens are dropped. "sources" counts frames, not items —
    how many frames reported the field at all.

    Returns {"frames": <valid blob count>, <field>: {...}, ...}.
    """
    scalars: dict[str, Counter] = {}      # field -> value -> frames
    lists: dict[str, Counter] = {}        # field -> item -> frames
    dicts: dict[str, dict[str, Counter]] = {}  # field -> subkey -> value -> frames
    dict_list_subs: set[tuple[str, str]] = set()  # (field, subkey) seen as list
    listdicts: dict[str, Counter] = {}    # field -> canonical-json -> frames
    listdicts_repr: dict[str, dict[str, dict]] = {}  # field -> json -> dict
    sources: Counter = Counter()
    frames = 0

    for blob in blobs or []:
        if not isinstance(blob, dict) or not _looks_like_blob(blob):
            continue
        frames += 1
        reported: set[str] = set()
        for raw_key, val in blob.items():
            if not isinstance(raw_key, str):
                continue
            key = re.sub(r"\s+", " ", raw_key.strip().lower())
            if not key or val is None:
                continue

            if isinstance(val, str):
                s = _norm_str(val)
                if s:
                    scalars.setdefault(key, Counter())[s] += 1
                    reported.add(key)
            elif isinstance(val, dict):
                sub = dicts.setdefault(key, {})
                for sk, sv in val.items():
                    if not isinstance(sk, str):
                        continue
                    nsk = re.sub(r"\s+", " ", sk.strip().lower())
                    if not nsk:
                        continue
                    if isinstance(sv, str):
                        ns = _norm_str(sv)
                        if ns:
                            sub.setdefault(nsk, Counter())[ns] += 1
                            reported.add(key)
                    elif isinstance(sv, (list, tuple)):
                        items = {_norm_str(x) for x in sv} - {None}
                        if items:
                            dict_list_subs.add((key, nsk))
                            for item in items:
                                sub.setdefault(nsk, Counter())[item] += 1
                            reported.add(key)
            elif isinstance(val, (list, tuple)):
                items = {_norm_str(x) for x in val if isinstance(x, str)} - {None}
                if items:
                    lc = lists.setdefault(key, Counter())
                    for item in items:  # once per frame: sources stay honest
                        lc[item] += 1
                    reported.add(key)
                for d in val:
                    if isinstance(d, dict):
                        nd = _norm_dict(d)
                        if nd:
                            canon = json.dumps(nd, sort_keys=True)
                            listdicts.setdefault(key, Counter())[canon] += 1
                            listdicts_repr.setdefault(key, {})[canon] = nd
                            reported.add(key)
            # anything else (numbers, bools, nested junk) is dropped

        for key in reported:
            sources[key] += 1

    merged: dict = {"frames": frames}
    for key in sorted(sources):
        entry: dict = {"sources": sources[key]}
        if key in scalars:
            counts = scalars[key]
            entry["value"] = _top(counts)
            entry["counts"] = _ordered(counts)
        if key in lists:
            counts = lists[key]
            entry["values"] = list(_ordered(counts))
            entry["counts"] = _ordered(counts)
        if key in dicts:
            sub_out, sub_counts = {}, {}
            for sk, sc in sorted(dicts[key].items()):
                sub_counts[sk] = _ordered(sc)
                if (key, sk) in dict_list_subs:
                    sub_out[sk] = list(_ordered(sc))  # union, freq-ordered
                else:
                    sub_out[sk] = _top(sc)            # modal scalar
            entry["value"] = sub_out
            entry["counts"] = sub_counts
        if key in listdicts:
            counts = listdicts[key]
            entry["values"] = [
                listdicts_repr[key][canon] for canon in _ordered(counts)
            ]
            entry["counts"] = _ordered(counts)
        merged[key] = entry
    return merged


# --------------------------------------------------------------------------
# feature_vector — flat, scoring-friendly view of a merged (or raw) blob
# --------------------------------------------------------------------------

def _entry_values(features: dict, key: str) -> list[str]:
    """All string values observed for a field, frequency-ordered.

    Accepts the merged {"values": [...]} / {"counts": {...}} shape, and also
    a raw blob where the field is a bare list or string.
    """
    e = (features or {}).get(key)
    if isinstance(e, dict):
        if isinstance(e.get("values"), list):
            return [v for v in e["values"] if isinstance(v, str)]
        if isinstance(e.get("counts"), dict):
            return [k for k in e["counts"] if isinstance(k, str)]
        if isinstance(e.get("value"), str):
            return [e["value"]]
        return []
    if isinstance(e, (list, tuple)):
        return sorted({_norm_str(x) for x in e} - {None})
    s = _norm_str(e)
    return [s] if s else []


def _entry_scalar(features: dict, key: str) -> str | None:
    """Modal scalar of a merged field (or the normalized bare value)."""
    e = (features or {}).get(key)
    if isinstance(e, dict) and isinstance(e.get("value"), str):
        return e["value"]
    return _norm_str(e)


def _entry_dict(features: dict, key: str) -> dict:
    """Merged dict-field value (sign_style) or a raw-blob dict, normalized."""
    e = (features or {}).get(key)
    if isinstance(e, dict) and isinstance(e.get("value"), dict):
        return e["value"]
    if isinstance(e, dict) and "value" not in e and "sources" not in e:
        return _norm_dict(e)
    return {}


def _first_values(features: dict, keys) -> list[str]:
    """Deduped union of _entry_values across the first-matching key names."""
    out: list[str] = []
    for key in keys:
        for v in _entry_values(features, key):
            if v not in out:
                out.append(v)
        s = _entry_scalar(features, key)
        if s and s not in out:
            out.append(s)
    return out


def _vehicle_notes(features: dict) -> list[str]:
    """Flatten vehicle dicts into readable notes: 'yellow taxi', 'eu sedan'."""
    e = (features or {}).get("vehicles")
    dicts: list[dict] = []
    if isinstance(e, dict) and isinstance(e.get("values"), list):
        dicts = [d for d in e["values"] if isinstance(d, dict)]
    elif isinstance(e, (list, tuple)):
        dicts = [_norm_dict(d) for d in e if isinstance(d, dict)]
    notes: list[str] = []
    for d in dicts:
        note = " ".join(v for _, v in sorted(d.items()) if v)
        if note and note not in notes:
            notes.append(note)
    return notes


def _subkey_counts(features: dict, key: str, subkey: str) -> list[str]:
    """Frequency-ordered values seen for one subkey of a merged dict field."""
    e = (features or {}).get(key)
    if not (isinstance(e, dict) and isinstance(e.get("counts"), dict)):
        return []
    sub = e["counts"].get(subkey)
    if isinstance(sub, dict):
        return [k for k in sub if isinstance(k, str)]
    return []


def feature_vector(features: dict) -> dict:
    """Normalize merged features into a flat, scoring-friendly dict.

    Works on merge_features() output, and tolerates a single raw blob.
    """
    sign = _entry_dict(features, "sign_style")
    conventions = _first_values(features, _CONVENTION_KEYS)
    for guess in _subkey_counts(features, "sign_style", "convention_guess"):
        if guess not in conventions:
            conventions.append(guess)
    raw_guess = _norm_str(sign.get("convention_guess"))
    if raw_guess and raw_guess not in conventions:
        conventions.append(raw_guess)

    markings: list[str] = []
    for text in _first_values(features, _MARKING_KEYS):
        for c in _COLOR_RE.findall(text):
            if c not in markings:
                markings.append(c)
        # prompt-defined value "mixed" carries meaning but no colour word
        if "mixed" in text and "mixed" not in markings:
            markings.append("mixed")

    lang_scripts = _first_values(features, _LANGUAGE_KEYS + _SCRIPT_KEYS)
    for k in ("language", "script"):
        v = _norm_str(sign.get(k))
        if v and v not in lang_scripts:
            lang_scripts.append(v)

    return {
        "languages_scripts": lang_scripts,
        "sign_conventions": conventions,
        "marking_colors": markings,
        "vegetation": _entry_values(features, "vegetation"),
        "terrain": _entry_scalar(features, "terrain"),
        "architecture": _entry_scalar(features, "architecture"),
        "climate_cues": _entry_values(features, "climate_cues"),
        "landmark_candidates": _entry_values(features, "landmark_candidates"),
        "vehicle_notes": _vehicle_notes(features),
    }


# --------------------------------------------------------------------------
# Evidence card
# --------------------------------------------------------------------------

def features_evidence(features: dict) -> "Evidence":
    """Package merged OSINT features as a descriptive Evidence card.

    Always INCONCLUSIVE — these are observations for the adjudicator, not a
    verdict. Confidence is a flat 0.3 when anything real was seen and 0.0
    when the blobs carried no usable features.
    """
    from .schema import ClaimVerdict, Evidence  # lazy

    vec = feature_vector(features)
    seen = {
        k: v for k, v in vec.items()
        if (isinstance(v, list) and v) or (isinstance(v, str) and v)
    }
    data = {
        "frames": (features or {}).get("frames", 0),
        "features": features or {},
        "vector": vec,
    }
    if not seen:
        return Evidence(
            type="osint_features",
            summary="no usable scene features in VLM frame blobs",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data=data,
        )
    parts = [
        f"{k}={', '.join(v) if isinstance(v, list) else v}"
        for k, v in seen.items()
    ]
    return Evidence(
        type="osint_features",
        summary=f"scene features across {data['frames']} frame(s): "
                + "; ".join(parts),
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=0.3,
        data=data,
    )


# --------------------------------------------------------------------------
# Demo / self-test: python -m sundial.osint_features [blob-or-list.json|file]
# --------------------------------------------------------------------------

_SELFTEST_BLOBS = [
    {
        "architecture": "Low-rise stucco, tiled roofs",
        "vegetation": ["Palm", "olive", "Palm "],
        "terrain": "flat coastal",
        "road_furniture": ["guardrail", "concrete bollards", None, ""],
        "sign_style": {
            "shape": "rectangular",
            "colors": ["blue", "white"],
            "convention_guess": "european",
        },
        "utility_poles": "concrete",
        "lane_markings_color": "white and yellow",
        "climate_cues": ["dry hills", "bright sun"],
        "vehicles": [
            {"type": "taxi", "color": "yellow"},
            {"type": "sedan", "color": "white", "junk": 42},
        ],
        "landmark_candidates": ["domed cathedral"],
    },
    {
        "architecture": "low-rise stucco tiled roofs",  # near-dup -> modal
        "vegetation": ["palm", "bougainvillea"],
        "terrain": None,
        "road_furniture": ["Guardrail", "speed limit sign"],
        "sign_style": {
            "shape": "rectangular",
            "colors": ["blue", "white"],
            "convention_guess": "EU",
        },
        "utility_poles": "concrete",
        "lane_markings_color": "White",
        "climate_cues": ["dry hills"],
        "vehicles": [{"type": "taxi", "color": "yellow"}],
        "landmark_candidates": ["domed cathedral", "seafront promenade"],
    },
    {"garbage": 123, "architecture": ""},  # junk-only frame
    "not a dict",  # junk blob
]


def _load_blobs(arg: str):
    """Parse argv JSON (object or list), falling back to a file path."""
    try:
        payload = json.loads(arg)
    except json.JSONDecodeError:
        with open(arg, encoding="utf-8") as f:
            payload = json.load(f)
    return payload if isinstance(payload, list) else [payload]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        blobs = _load_blobs(sys.argv[1])
    else:
        print("self-test: merging 4 sample blobs (incl. junk) "
              "— pass a JSON blob/list or file to merge real data")
        blobs = _SELFTEST_BLOBS

    merged = merge_features(blobs)
    print("\nmerged features:")
    print(json.dumps(merged, indent=2, ensure_ascii=False))

    vec = feature_vector(merged)
    print("\nfeature vector:")
    print(json.dumps(vec, indent=2, ensure_ascii=False))

    try:
        ev = features_evidence(merged)
    except ImportError:
        print("\n(run as `python -m sundial.osint_features` to build Evidence)")
    else:
        print(f"\nevidence[{ev.type}] {ev.verdict.value} "
              f"conf={ev.confidence}: {ev.summary}")

    # self-test assertions — keep quiet unless something breaks
    if len(sys.argv) <= 1:
        assert merged["frames"] == 3, merged["frames"]
        assert merged["vegetation"]["counts"]["palm"] == 2
        assert merged["terrain"]["sources"] == 1  # None dropped
        assert "guardrail" in merged["road_furniture"]["values"]
        assert merged["vehicles"]["sources"] == 2
        assert vec["terrain"] == "flat coastal"
        assert "yellow taxi" in vec["vehicle_notes"]
        assert set(vec["marking_colors"]) == {"white", "yellow"}
        print("\nself-test assertions passed")
