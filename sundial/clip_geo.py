"""CLIP geo prior — a GeoCLIP-style image→country signal via OpenCLIP.

GeoCLIP (Vivanco Cepeda et al.) geolocates a photo by embedding it and
retrieving the nearest entries from a gallery of location embeddings.
This module re-implements the *idea* self-contained — no cloned code —
with a text-prompt gallery: ~40 country centroids rendered as prompts
("a street view photo in <Country>"), embedded once with OpenCLIP's
ViT-B/32 and cached to disk (~/.sundial_cache/clip_gallery.npz, falling
back to the temp dir). Frame embeddings are scored against the gallery
by cosine similarity and mean-pooled over frames into a country prior —
a learned visual signal independent of the VLM's text guesses.

Optional dependency: everything degrades cleanly when `torch` /
`open_clip_torch` are not installed (the dist imports as `open_clip`).
It is deliberately NOT in requirements.txt — progressive enhancement
only. `available()` is the gate; every public function returns
None / [] / INCONCLUSIVE when it is False, and torch/open_clip/numpy are
imported lazily inside functions, so `import sundial.clip_geo` stays
stdlib-only.

The prior is a hint, not a measurement: "the model thinks this looks
like Germany" is correlational, so confidence is capped at 0.35.

Stage API:

    available()                       -> bool
    embed_image(path)                 -> np.ndarray | None  (L2-normalized)
    geo_prior(frame_paths, top_k=5)   -> [{cc, name, score}] sorted desc
    clip_evidence(prior, claimed_cc)  -> Evidence (type "clip_geo")
    gallery_texts()                   -> the prompts used (transparency)

Integrator: call geo_prior() on the frames *after* frame_selector picks
them (same budget — no extra frames needed), then clip_evidence(prior,
claim_cc). Gate the whole stage on available(). Pass top_k >= ~8 if the
CONTRADICTS branch should be reachable from the prior rows alone.
"""
from __future__ import annotations

__all__ = [
    "available",
    "embed_image",
    "geo_prior",
    "clip_evidence",
    "gallery_texts",
]

import json
import math
import os

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "laion2b_s34b_b79k"  # strong general OpenCLIP ViT-B/32 weights
_PROMPT = "a street view photo in {}"
_GALLERY_VERSION = 1               # bump when prompts/gallery change (cache key)
_CACHE_DIRNAME = ".sundial_cache"
_CACHE_FILENAME = "clip_gallery.npz"

_CONF_SUPPORT = 0.35     # claimed in top-2 — capped low: a learned prior is fallible
_CONF_CONTRADICT = 0.35  # claimed rank>5 while the top pick is strong
_CONF_NEAR = 0.20        # claimed ranks 3-5 — a near miss, not decisive
_CONF_WEAK = 0.15        # prior exists but can't rule on the claim

_RANK_SUPPORT = 2        # claimed within this rank -> SUPPORTS
_RANK_FAR = 5            # claimed beyond this rank -> CONTRADICTS (if strong)
_STRONG_TOP = 0.25       # top-1 cosine at/above this reads as confident
                         # (CLIP text↔image sims on these prompts run ~0.15-0.30)
_STRONG_MARGIN = 0.02    # ...and must beat the claimed row's score by this

# ---------------------------------------------------------------------------
# Built-in gallery — ~40 country centroids covering every continent,
# consistent with geocell_pred._CC_GEO_FALLBACK. The gallery doubles as the
# prompt list via _PROMPT; order is fixed so the npz cache stays aligned.
# ---------------------------------------------------------------------------

_GALLERY: list[dict] = [
    # --- Europe (14) ---
    {"cc": "GB", "lat": 54.00, "lon": -2.00, "name": "United Kingdom"},
    {"cc": "IE", "lat": 53.18, "lon": -8.24, "name": "Ireland"},
    {"cc": "FR", "lat": 46.60, "lon": 2.40, "name": "France"},
    {"cc": "DE", "lat": 51.16, "lon": 10.45, "name": "Germany"},
    {"cc": "NL", "lat": 52.20, "lon": 5.30, "name": "Netherlands"},
    {"cc": "ES", "lat": 40.20, "lon": -3.60, "name": "Spain"},
    {"cc": "PT", "lat": 39.60, "lon": -8.00, "name": "Portugal"},
    {"cc": "IT", "lat": 42.80, "lon": 12.80, "name": "Italy"},
    {"cc": "CH", "lat": 46.80, "lon": 8.23, "name": "Switzerland"},
    {"cc": "PL", "lat": 52.00, "lon": 19.40, "name": "Poland"},
    {"cc": "SE", "lat": 62.00, "lon": 15.00, "name": "Sweden"},
    {"cc": "NO", "lat": 64.50, "lon": 12.00, "name": "Norway"},
    {"cc": "GR", "lat": 39.00, "lon": 22.00, "name": "Greece"},
    {"cc": "TR", "lat": 39.00, "lon": 35.00, "name": "Turkey"},
    # --- Americas (8) ---
    {"cc": "US", "lat": 39.50, "lon": -98.35, "name": "United States"},
    {"cc": "CA", "lat": 56.00, "lon": -96.00, "name": "Canada"},
    {"cc": "MX", "lat": 23.60, "lon": -102.50, "name": "Mexico"},
    {"cc": "BR", "lat": -10.80, "lon": -52.90, "name": "Brazil"},
    {"cc": "AR", "lat": -34.00, "lon": -64.00, "name": "Argentina"},
    {"cc": "CL", "lat": -31.00, "lon": -71.00, "name": "Chile"},
    {"cc": "CO", "lat": 4.00, "lon": -73.00, "name": "Colombia"},
    {"cc": "PE", "lat": -9.20, "lon": -75.00, "name": "Peru"},
    # --- Asia / Middle East (11) ---
    {"cc": "JP", "lat": 36.50, "lon": 138.00, "name": "Japan"},
    {"cc": "KR", "lat": 36.50, "lon": 127.90, "name": "South Korea"},
    {"cc": "CN", "lat": 35.00, "lon": 105.00, "name": "China"},
    {"cc": "IN", "lat": 22.00, "lon": 79.00, "name": "India"},
    {"cc": "TH", "lat": 15.50, "lon": 101.00, "name": "Thailand"},
    {"cc": "VN", "lat": 14.00, "lon": 108.00, "name": "Vietnam"},
    {"cc": "SG", "lat": 1.35, "lon": 103.85, "name": "Singapore"},
    {"cc": "ID", "lat": -2.20, "lon": 117.00, "name": "Indonesia"},
    {"cc": "PH", "lat": 12.80, "lon": 122.00, "name": "Philippines"},
    {"cc": "IL", "lat": 31.00, "lon": 34.80, "name": "Israel"},
    {"cc": "AE", "lat": 24.00, "lon": 54.00, "name": "United Arab Emirates"},
    # --- Africa (5) ---
    {"cc": "EG", "lat": 26.80, "lon": 30.80, "name": "Egypt"},
    {"cc": "MA", "lat": 31.80, "lon": -6.00, "name": "Morocco"},
    {"cc": "NG", "lat": 9.60, "lon": 8.00, "name": "Nigeria"},
    {"cc": "KE", "lat": 0.50, "lon": 37.90, "name": "Kenya"},
    {"cc": "ZA", "lat": -29.00, "lon": 25.00, "name": "South Africa"},
    # --- Oceania (2) ---
    {"cc": "AU", "lat": -25.00, "lon": 134.00, "name": "Australia"},
    {"cc": "NZ", "lat": -41.80, "lon": 172.80, "name": "New Zealand"},
]


def _schema():
    """Lazy schema import so the module stays dep-free at import time.

    Falls back to a same-shape shim when schema can't be found at all
    (odd packaging), so clip_evidence never dies on import context.
    Same pattern as geocell_pred._schema().
    """
    try:
        from .schema import ClaimVerdict, Evidence
        return ClaimVerdict, Evidence
    except Exception:
        pass
    try:
        from schema import ClaimVerdict, Evidence  # plain-script mode
        return ClaimVerdict, Evidence
    except Exception:
        pass
    from dataclasses import dataclass, field

    class ClaimVerdict:  # duck-typed stand-in for schema.ClaimVerdict
        SUPPORTS = "supports"
        CONTRADICTS = "contradicts"
        INCONCLUSIVE = "inconclusive"

    @dataclass
    class Evidence:  # minimal schema.Evidence shape
        type: str
        summary: str
        verdict: object = ClaimVerdict.INCONCLUSIVE
        confidence: float = 0.0
        data: dict = field(default_factory=dict)
        frames: list = field(default_factory=list)

    return ClaimVerdict, Evidence


# ---------------------------------------------------------------------------
# Lazy model + gallery state
# ---------------------------------------------------------------------------

_CLIP_STACK = None   # (model, preprocess, tokenizer) once loaded
_STACK_DEAD = False  # load was tried and failed — don't hammer retries
_GAL_EMBEDS = None   # (N, D) float32 gallery matrix, in-process cache


def available() -> bool:
    """True when torch + open_clip are importable. Cheap; never raises."""
    try:
        import importlib.util
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("open_clip") is not None
        )
    except Exception:
        return False


def _clip_stack():
    """Lazy-load the OpenCLIP model once per process. None on any failure
    (missing deps, no network for the weight download, OOM — latched dead
    so a bad state isn't retried every call)."""
    global _CLIP_STACK, _STACK_DEAD
    if _CLIP_STACK is not None:
        return _CLIP_STACK
    if _STACK_DEAD:
        return None
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED)
        tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        model.eval()
        _CLIP_STACK = (model, preprocess, tokenizer)
        return _CLIP_STACK
    except Exception:
        _STACK_DEAD = True
        return None


def _cache_file() -> str | None:
    """~/.sundial_cache/clip_gallery.npz, falling back to the temp dir.
    None when no writable location can be arranged."""
    candidates = []
    try:
        candidates.append(
            os.path.join(os.path.expanduser("~"), _CACHE_DIRNAME))
    except Exception:
        pass
    try:
        import tempfile
        candidates.append(tempfile.gettempdir())
    except Exception:
        pass
    for base in candidates:
        try:
            os.makedirs(base, exist_ok=True)
            if os.path.isdir(base) and os.access(base, os.W_OK):
                return os.path.join(base, _CACHE_FILENAME)
        except Exception:
            continue
    return None


def _gallery_meta() -> dict:
    """Cache key: everything that would change the gallery embeddings."""
    return {
        "model": _MODEL_NAME,
        "pretrained": _PRETRAINED,
        "prompt": _PROMPT,
        "version": _GALLERY_VERSION,
        "ccs": [g["cc"] for g in _GALLERY],
    }


def _gallery_embeddings():
    """(N, D) float32 L2-normalized text embeddings for the gallery.

    Loads the npz cache when it matches _gallery_meta(); otherwise encodes
    the prompts once and writes the cache (best-effort — a failed write
    just means the next process recomputes). None when unavailable."""
    global _GAL_EMBEDS
    if _GAL_EMBEDS is not None:
        return _GAL_EMBEDS
    try:
        import numpy as np

        texts = gallery_texts()
        meta_tag = json.dumps(_gallery_meta(), sort_keys=True)
        path = _cache_file()
        if path and os.path.exists(path):
            try:
                with np.load(path, allow_pickle=False) as z:
                    emb = np.asarray(z["embeds"], dtype=np.float32)
                    tag = str(np.asarray(z["meta"]).tolist())
                if tag == meta_tag and emb.ndim == 2 \
                        and emb.shape[0] == len(_GALLERY) \
                        and np.isfinite(emb).all():
                    _GAL_EMBEDS = emb
                    return _GAL_EMBEDS
            except Exception:
                pass  # stale/corrupt cache -> recompute below

        stack = _clip_stack()
        if stack is None:
            return None
        model, _, tokenizer = stack
        import torch
        with torch.no_grad():
            tokens = tokenizer(texts)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        _GAL_EMBEDS = feats.detach().cpu().numpy().astype(np.float32)

        if path:
            try:  # atomic-ish: write sibling then replace
                tmp = path + ".tmp.npz"
                np.savez(tmp, embeds=_GAL_EMBEDS, meta=np.array(meta_tag))
                os.replace(tmp, path)
            except Exception:
                pass
        return _GAL_EMBEDS
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage API
# ---------------------------------------------------------------------------

def gallery_texts() -> list[str]:
    """The text prompts embedded for the gallery — for transparency/debug.
    Pure stdlib; works even when the model doesn't."""
    return [_PROMPT.format(g["name"]) for g in _GALLERY]


def embed_image(path):
    """L2-normalized ViT-B/32 image embedding (float32, shape (D,)) for a
    frame file, or None when unavailable/unreadable. Model loads lazily on
    first use and is cached for the process."""
    try:
        if not path or not isinstance(path, (str, os.PathLike)) or not available():
            return None
        path = os.fspath(path)
        stack = _clip_stack()
        if stack is None:
            return None
        model, preprocess, _ = stack
        import numpy as np
        import torch
        from PIL import Image  # open_clip hard-depends on Pillow
        with Image.open(path) as im:
            img = preprocess(im.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None


def geo_prior(frame_paths, top_k: int = 5) -> list[dict]:
    """Country prior from frame images: embed each frame, cosine-score it
    against the gallery, mean-pool over frames, return the top-k rows:

        [{"cc": "JP", "name": "Japan", "score": 0.28}, ...]   sorted desc

    [] when unavailable, when every frame fails to embed, or on error."""
    try:
        if not available():
            return []
        paths = [os.fspath(p) for p in (frame_paths or [])
                 if isinstance(p, (str, os.PathLike)) and str(p)]
        if not paths:
            return []
        import numpy as np

        gal = _gallery_embeddings()
        if gal is None or getattr(gal, "ndim", 0) != 2:
            return []

        embs = []
        for p in paths:
            e = embed_image(p)
            if e is not None:
                embs.append(np.asarray(e, dtype=np.float32).ravel())
        if not embs:
            return []

        E = np.stack(embs)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        G = gal / np.maximum(np.linalg.norm(gal, axis=1, keepdims=True), 1e-9)
        sims = (E @ G.T).mean(axis=0)  # mean-pool frames x gallery -> (N,)

        try:
            k = max(1, int(top_k))
        except (TypeError, ValueError):
            k = 5
        order = np.argsort(-sims)[:k]
        return [
            {
                "cc": _GALLERY[int(i)]["cc"],
                "name": _GALLERY[int(i)]["name"],
                "score": round(float(sims[int(i)]), 4),
            }
            for i in order
        ]
    except Exception:
        return []


def _norm_rows(prior) -> list[dict]:
    """Sanitize a prior into [{cc, name, score}] sorted desc — tolerates
    missing scores, stray keys, non-dict rows."""
    rows = []
    for r in (prior or []):
        if not isinstance(r, dict):
            continue
        cc = str(r.get("cc") or r.get("country_code") or "").strip().upper()
        if not cc:
            continue
        try:
            s = float(r.get("score"))
        except (TypeError, ValueError):
            s = 0.0
        if not math.isfinite(s):
            s = 0.0
        rows.append({"cc": cc, "name": str(r.get("name") or ""), "score": s})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _prior_verdict(rows, cc: str, ClaimVerdict):
    """Verdict logic on a normalized prior. Returns
    (verdict, confidence, summary, rank|None).

    `rank` is the claimed country's 1-based position when visible; when it
    is absent from a prior that reaches at least _RANK_FAR rows, its true
    rank is provably > _RANK_FAR, so rank=len(rows)+1 stands in and the
    last visible score is used as a conservative margin proxy."""
    top = rows[0]
    top_cc, top_s = top["cc"], float(top["score"])
    hit = next(((i, r) for i, r in enumerate(rows) if r["cc"] == cc), None)
    if hit is not None:
        rank, claimed_s = hit[0] + 1, float(hit[1]["score"])
    elif len(rows) >= _RANK_FAR:
        rank, claimed_s = len(rows) + 1, float(rows[-1]["score"])
    else:
        rank, claimed_s = None, 0.0

    if rank is not None and rank <= _RANK_SUPPORT:
        return (
            ClaimVerdict.SUPPORTS,
            _CONF_SUPPORT,
            f"clip prior ranks claimed {cc} #{rank} (score {claimed_s:.3f}) "
            f"— frames visually match the claimed country",
            rank,
        )
    strong = top_s >= _STRONG_TOP and (top_s - claimed_s) >= _STRONG_MARGIN
    if rank is not None and rank > _RANK_FAR and strong:
        where = f"#{rank}" if hit is not None else f"below the top {len(rows)}"
        return (
            ClaimVerdict.CONTRADICTS,
            _CONF_CONTRADICT,
            f"clip prior favours {top_cc} ({top_s:.3f}); claimed {cc} ranks "
            f"{where} — the visuals point elsewhere",
            rank,
        )
    if hit is not None and rank <= _RANK_FAR:
        return (
            ClaimVerdict.INCONCLUSIVE,
            _CONF_NEAR,
            f"claimed {cc} sits #{rank} in the clip prior — plausible but "
            f"not decisive (top {top_cc} {top_s:.3f})",
            rank,
        )
    if hit is not None:  # rank > _RANK_FAR but the prior is too flat
        return (
            ClaimVerdict.INCONCLUSIVE,
            _CONF_WEAK,
            f"claimed {cc} only ranks #{rank}, but clip scores are too flat "
            f"to rule on it (top {top_s:.3f})",
            rank,
        )
    if len(rows) < _RANK_FAR:
        return (
            ClaimVerdict.INCONCLUSIVE,
            _CONF_WEAK,
            f"clip prior only spans {len(rows)} countries — can't place "
            f"claimed {cc}",
            None,
        )
    return (
        ClaimVerdict.INCONCLUSIVE,
        _CONF_WEAK,
        f"claimed {cc} falls below the clip top-{len(rows)}, but the top "
        f"score {top_s:.3f} is too weak to convict",
        rank,
    )


def clip_evidence(prior, claimed_cc) -> "Evidence":
    """Package a geo_prior result vs the claimed country as Evidence.

    Verdicts (all capped low — a learned prior corroborates, never convicts):
      * claimed cc in the prior's top-2          -> SUPPORTS 0.35
      * claimed rank > 5 while the top score is
        strong (>= _STRONG_TOP and a clear margin) -> CONTRADICTS 0.35
      * claimed ranks 3-5 / weak or flat prior
        / no prior / no claim                    -> INCONCLUSIVE
      * model unavailable                        -> INCONCLUSIVE 0.0,
        honestly labelled "clip prior unavailable (no open_clip)"
    """
    ClaimVerdict, Evidence = _schema()
    try:
        if not available():
            return Evidence(
                type="clip_geo",
                summary="clip prior unavailable (no open_clip)",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                data={"available": False},
            )
        rows = _norm_rows(prior)
        cc = str(claimed_cc or "").strip().upper()
        data = {
            "available": True,
            "model": f"{_MODEL_NAME}/{_PRETRAINED}",
            "claimed_cc": cc or None,
            "prior": rows,
        }
        if not rows:
            return Evidence(
                type="clip_geo",
                summary="clip prior produced no ranked countries — "
                        "nothing to check the claim against",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                data=data,
            )
        if not cc:
            return Evidence(
                type="clip_geo",
                summary=f"clip prior favours {rows[0]['cc']} "
                        f"({rows[0]['score']:.3f}) but there is no claimed "
                        f"country to compare",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=_CONF_WEAK,
                data=data,
            )
        verdict, conf, summary, rank = _prior_verdict(rows, cc, ClaimVerdict)
        data["rank"] = rank
        return Evidence(
            type="clip_geo",
            summary=summary,
            verdict=verdict,
            confidence=conf,
            data=data,
        )
    except Exception:
        try:
            return Evidence(
                type="clip_geo",
                summary="clip prior check failed — inconclusive",
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.0,
                data={},
            )
        except Exception:  # pragma: no cover — shim Evidence cannot fail
            return None


# ---------------------------------------------------------------------------
# Self-test: python3 sundial/clip_geo.py  (or python -m sundial.clip_geo)
# Offline-safe: exercises the unavailable path, empty inputs, and the
# verdict logic via injected fake priors (availability patched briefly).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fails = 0

    def check(name, got, want):
        global fails
        ok = got == want
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} (want {want!r})")

    def check_true(name, got):
        global fails
        if not got:
            fails += 1
        print(f"  {'PASS' if got else 'FAIL'}  {name}: got {got!r}")

    ClaimVerdict, _Ev = _schema()
    v_of = lambda v: getattr(v, "value", v)

    # --- availability + gallery (no deps needed) --------------------------
    av = available()
    print("available():", av)
    check("available() returns bool", isinstance(av, bool), True)

    texts = gallery_texts()
    check_true("gallery_texts nonempty", len(texts) == len(_GALLERY))
    check("gallery size ~40", len(_GALLERY), 40)
    check_true("prompts well-formed",
               all(t == f"a street view photo in {_GALLERY[i]['name']}"
                   for i, t in enumerate(texts)))
    check_true("ccs unique", len({g["cc"] for g in _GALLERY}) == len(_GALLERY))
    check_true("centroids sane",
               all(-90 <= g["lat"] <= 90 and -180 <= g["lon"] <= 180
                   for g in _GALLERY))

    # --- degrade paths ----------------------------------------------------
    if not av:
        check("embed_image -> None offline", embed_image("x.jpg"), None)
        check("geo_prior -> [] offline", geo_prior(["a.jpg", "b.jpg"]), [])
    else:  # deps installed: still must not raise on bad input
        check("embed_image missing file -> None",
              embed_image("/nonexistent/frame.jpg"), None)
        check_true("geo_prior returns list",
                   isinstance(geo_prior(["/nonexistent/frame.jpg"]), list))
    check("geo_prior empty -> []", geo_prior([]), [])
    check("geo_prior None -> []", geo_prior(None), [])

    # --- clip_evidence: unavailable gate (real environment) ---------------
    fake = [{"cc": "JP", "name": "Japan", "score": 0.31},
            {"cc": "KR", "name": "South Korea", "score": 0.30}]
    ev = clip_evidence(fake, "FR")
    check("evidence type", getattr(ev, "type", None), "clip_geo")
    if not av:
        check("unavailable -> INCONCLUSIVE", v_of(ev.verdict), "inconclusive")
        check("unavailable conf", ev.confidence, 0.0)
        check_true("honest summary", "no open_clip" in ev.summary)

    # --- verdict logic via injected priors (availability patched on) ------
    saved_avail = globals()["available"]
    globals()["available"] = lambda: True
    try:
        rows = [
            {"cc": "JP", "name": "Japan", "score": 0.31},
            {"cc": "KR", "name": "South Korea", "score": 0.30},
            {"cc": "CN", "name": "China", "score": 0.27},
            {"cc": "TH", "name": "Thailand", "score": 0.26},
            {"cc": "VN", "name": "Vietnam", "score": 0.25},
            {"cc": "FR", "name": "France", "score": 0.20},
            {"cc": "DE", "name": "Germany", "score": 0.19},
        ]
        ev = clip_evidence(rows, "JP")
        check("rank-1 claim -> SUPPORTS", v_of(ev.verdict), "supports")
        check("supports conf capped", ev.confidence, 0.35)
        check("rank in data", ev.data.get("rank"), 1)

        ev = clip_evidence(rows, "KR")
        check("rank-2 claim -> SUPPORTS", v_of(ev.verdict), "supports")

        ev = clip_evidence(rows, "CN")
        check("rank-3 claim -> INCONCLUSIVE", v_of(ev.verdict), "inconclusive")
        check("near conf", ev.confidence, 0.20)

        ev = clip_evidence(rows, "FR")
        check("rank-6 + strong top -> CONTRADICTS",
              v_of(ev.verdict), "contradicts")
        check("contradicts conf", ev.confidence, 0.35)

        ev = clip_evidence(rows, "ZZ")  # absent from a >=5-row prior
        check("absent + strong top -> CONTRADICTS",
              v_of(ev.verdict), "contradicts")

        weak = [dict(r, score=round(r["score"] * 0.5, 4)) for r in rows]
        ev = clip_evidence(weak, "FR")
        check("rank-6 but weak top -> INCONCLUSIVE",
              v_of(ev.verdict), "inconclusive")

        short = rows[:4]
        ev = clip_evidence(short, "ZZ")
        check("absent from short prior -> INCONCLUSIVE",
              v_of(ev.verdict), "inconclusive")

        ev = clip_evidence([], "FR")
        check("empty prior -> INCONCLUSIVE", v_of(ev.verdict), "inconclusive")
        check("empty prior conf", ev.confidence, 0.0)

        ev = clip_evidence(rows, None)
        check("no claim -> INCONCLUSIVE", v_of(ev.verdict), "inconclusive")

        ev = clip_evidence("garbage", "FR")
        check("garbage prior -> INCONCLUSIVE", v_of(ev.verdict), "inconclusive")
    finally:
        globals()["available"] = saved_avail

    print(f"self-test: {'FAILED ' + str(fails) if fails else 'all passed'}")
    raise SystemExit(1 if fails else 0)
