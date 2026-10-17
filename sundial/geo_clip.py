"""GeoCLIP prior — a real image->GPS model on top of the cue machinery.

``VicenteVivan/geo-clip`` (NeurIPS'23, MIT): a CLIP-style model trained on
4.7 M geo-tagged photos that predicts *coordinates* — not just countries —
with per-prediction probabilities::

    model.predict("frame.jpg", top_k=5)
        -> (top_pred_gps [(lat, lon), ...], top_pred_prob [p, ...])

This module wraps it lazily and absent-safely in the house style: heavy
deps (torch/geoclip/PIL) import on first use, a failed or missing model
latches off, and every public function degrades to []/None instead of
raising. Runs anywhere torch runs — CPU works, the VM's GPU is faster.

API:

  * ``available()``            — import sanity check (cached)
  * ``predict_coords(frames)`` — per-frame top-k predictions pooled into
                                 [{lat, lon, prob, frames}] clusters
  * ``geoclip_evidence(rows, anchor, claimed_cc)`` -> Evidence | None
  * ``pred_summary(rows)``     — {lat, lon, r_km} for the map rail

    python3 sundial/geo_clip.py     # self-test
    python -m sundial.geo_clip
"""
from __future__ import annotations

import math
import os
from typing import Any

__all__ = ["available", "predict_coords", "geoclip_evidence", "pred_summary"]

_MAX_FRAMES = 4            # prediction is per-frame; cap the spend
_CLUSTER_KM = 150.0        # predictions within this pool together
_CONTRADICT_KM = 1500.0    # claim anchor farther than this -> CONTRADICTS
_SUPPORT_KM = 250.0        # anchor within this -> SUPPORTS
_MODEL = None              # latched GeoCLIP() instance
_MODEL_TRIED = False       # one-shot import latch
_CACHE_DIR = None


def _schema():
    try:
        from .schema import ClaimVerdict, Evidence
    except ImportError:  # direct-script context
        from schema import ClaimVerdict, Evidence
    return ClaimVerdict, Evidence


def _model():
    """Lazy GeoCLIP instance; None when torch/geoclip isn't installed."""
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    try:
        from geoclip import GeoCLIP  # noqa: pulls torch + downloads weights
        _MODEL = GeoCLIP()
    except Exception:
        _MODEL = None
    return _MODEL


def available() -> bool:
    """True when the geoclip package imports and instantiates."""
    return _model() is not None


def _float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _predict_one(model, path: str, top_k: int):
    """One frame -> [(lat, lon, prob), ...]; never raises."""
    try:
        gps, probs = model.predict(path, top_k=top_k)
    except Exception:
        return []
    if gps is None:
        return []
    out = []
    for i, pt in enumerate(gps):
        try:
            la = _float(pt[0])
            lo = _float(pt[1])
        except (TypeError, IndexError):
            continue
        if la is None or lo is None:
            continue
        p = 0.0
        try:
            p = float(probs[i])
        except (TypeError, IndexError, ValueError):
            pass
        if -90 <= la <= 90 and -180 <= lo <= 180:
            out.append((la, lo, max(0.0, p)))
    return out


def _haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = math.radians(b_lat - a_lat), math.radians(b_lon - a_lon)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _place_label(lat: float, lon: float) -> dict:
    """Offline nearest-city lookup via offline_places; {} on any failure."""
    try:
        try:
            from . import offline_places
        except ImportError:
            import offline_places  # type: ignore[no-redef]
        hit = offline_places.nearest_city(lat, lon)
        if isinstance(hit, dict) and hit.get("label"):
            return hit
    except Exception:
        pass
    return {}


def predict_coords(frame_paths, top_k: int = 5) -> list[dict]:
    """Per-frame GeoCLIP predictions -> clustered [{lat, lon, prob, frames}].

    Each frame contributes its top-1 prediction; predictions inside
    _CLUSTER_KM of a cluster centre join it, weighted by prob. Alternate
    top-k candidates from each frame are also preserved in `alts` so
    downstream ensemble / map-marker stages see the full distribution.
    Rows are sorted by total prob descending — row 0 is the model's best guess.
    [] when the model is absent, no frames predict, or anything fails.
    """
    try:
        model = _model()
        if model is None:
            return []
        paths = [os.fspath(p) for p in (frame_paths or [])
                 if isinstance(p, (str, os.PathLike)) and str(p)]
        if not paths:
            return []
        clusters: list[dict] = []
        for path in paths[:_MAX_FRAMES]:
            preds = _predict_one(model, path, top_k)
            if not preds:
                continue
            la, lo, prob = preds[0]  # top-1 per frame
            placed = False
            for c in clusters:
                if _haversine_km(la, lo, c["lat"], c["lon"]) <= _CLUSTER_KM:
                    w = c["prob"] + prob
                    if w > 0:
                        c["lat"] = (c["lat"] * c["prob"] + la * prob) / w
                        c["lon"] = (c["lon"] * c["prob"] + lo * prob) / w
                    c["prob"] = w
                    c["frames"] += 1
                    c.setdefault("alts", []).extend(preds[1:])
                    placed = True
                    break
            if not placed:
                clusters.append({"lat": la, "lon": lo, "prob": prob,
                                 "frames": 1, "alts": list(preds[1:])})
        clusters.sort(key=lambda c: -c["prob"])
        for c in clusters:
            hit = _place_label(c["lat"], c["lon"])
            if hit:
                c["label"] = hit.get("label")
                c["city"] = hit.get("name")
                c["cc"] = hit.get("cc")
                c["country"] = hit.get("country")
        return clusters
    except Exception:
        return []


def pred_summary(rows: list[dict]) -> dict | None:
    """Best cluster -> {lat, lon, r_km} for the map rail / anchor.

    Radius is the spread of the prediction mass — a tight cluster of
    agreeing frames yields ~80 km; scattered guesses yield several
    hundred. Returns None on empty input.
    """
    try:
        if not rows:
            return None
        best = rows[0]
        spread = 0.0
        total = sum(float(r.get("prob") or 0) for r in rows) or 1.0
        for r in rows[1:]:
            w = float(r.get("prob") or 0) / total
            spread += w * _haversine_km(
                best["lat"], best["lon"], r["lat"], r["lon"])
        r_km = max(80.0, 80.0 + spread * 1.5)
        return {"lat": round(best["lat"], 5), "lon": round(best["lon"], 5),
                "r_km": round(min(r_km, 1500.0), 1),
                "prob": round(float(best.get("prob") or 0), 4),
                "frames": int(best.get("frames") or 1)}
    except Exception:
        return None


def geoclip_evidence(rows: list[dict], anchor=None,
                     claimed_cc: str | None = None):
    """GeoCLIP coordinate prior vs the claimed anchor -> Evidence | None.

      * no rows                    -> None (silent)
      * anchor <= _SUPPORT_KM      -> SUPPORTS 0.55 (real coordinates are
        the strongest single signal we have short of a landmark name)
      * anchor > _CONTRADICT_KM    -> CONTRADICTS 0.55
      * between                    -> INCONCLUSIVE 0.35
      * no anchor                  -> INCONCLUSIVE 0.35 (the prediction
        still geolocates the file — reported, just uncontested)
    """
    try:
        ClaimVerdict, Evidence = _schema()
    except Exception:
        return None
    try:
        if not rows:
            return None
        best = rows[0]
        summ = pred_summary(rows)
        if summ is None:
            return None
        cands: list[dict] = []
        for r in rows[:5]:
            c_hit = _place_label(r["lat"], r["lon"])
            cands.append({
                "lat": r["lat"], "lon": r["lon"], "prob": r["prob"],
                "label": r.get("label") or c_hit.get("label"),
                "cc": r.get("cc") or c_hit.get("cc"),
            })
        if len(cands) < 5:
            for r in rows:
                for alt in (r.get("alts") or []):
                    if len(cands) >= 5:
                        break
                    try:
                        ala, alo, ap = float(alt[0]), float(alt[1]), float(alt[2])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if all(_haversine_km(ala, alo, x["lat"], x["lon"]) > 35.0
                           for x in cands):
                        a_hit = _place_label(ala, alo)
                        cands.append({
                            "lat": ala, "lon": alo, "prob": ap,
                            "label": a_hit.get("label"),
                            "cc": a_hit.get("cc"),
                        })
        top_hit = _place_label(summ["lat"], summ["lon"])
        place_str = (best.get("label") or top_hit.get("label") or "").strip()
        # Aggregate candidate probability mass by country so the report rail
        # shows a meaningful country bar (e.g. 5/5 top guesses in DE -> ~78%)
        # rather than a raw 1-of-100k gallery softmax value (0.04).
        cc_mass: dict[str, float] = {}
        cc_hits: dict[str, int] = {}
        for r in rows:
            rcc = r.get("cc") or _place_label(r["lat"], r["lon"]).get("cc")
            if rcc:
                cc_mass[rcc] = cc_mass.get(rcc, 0.0) + float(r.get("prob") or 0.0)
                cc_hits[rcc] = cc_hits.get(rcc, 0) + 1
            for alt in (r.get("alts") or []):
                try:
                    ala, alo, ap = float(alt[0]), float(alt[1]), float(alt[2])
                except (TypeError, ValueError, IndexError):
                    continue
                acc = _place_label(ala, alo).get("cc")
                if acc:
                    cc_mass[acc] = cc_mass.get(acc, 0.0) + max(0.0, ap)
                    cc_hits[acc] = cc_hits.get(acc, 0) + 1
        tot_mass = sum(cc_mass.values()) or 1.0
        ranked_ccs = [
            {"cc": k, "score": round(min(0.88, 0.80 * (v / tot_mass)), 3),
             "hits": cc_hits.get(k, 1)}
            for k, v in sorted(cc_mass.items(), key=lambda kv: -kv[1])
        ]
        data = {"lat": summ["lat"], "lon": summ["lon"],
                "r_km": summ["r_km"], "prob": summ["prob"],
                "frames_agreeing": summ["frames"],
                "label": place_str or None,
                "cc": best.get("cc") or top_hit.get("cc"),
                "ranked": ranked_ccs,
                "candidates": cands,
                "model": "geoclip/mp16"}
        alat = alon = None
        try:
            if anchor is not None:
                if isinstance(anchor, dict):
                    alat, alon = float(anchor.get("lat")), float(anchor.get("lon"))
                else:
                    alat, alon = float(anchor[0]), float(anchor[1])
        except (TypeError, ValueError, IndexError):
            alat = alon = None

        agree = int(summ["frames"])
        where_txt = (f"{place_str} ({summ['lat']:.3f}, {summ['lon']:.3f})"
                     if place_str else f"{summ['lat']:.3f}, {summ['lon']:.3f}")
        base = (f"GeoCLIP pins {where_txt} "
                f"(~{summ['r_km']:.0f} km radius, {agree} frame"
                f"{'s' if agree != 1 else ''} agreeing)")
        if alat is None:
            return Evidence(
                type="geo_clip", summary=base,
                verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.35,
                data=data)
        dist = _haversine_km(summ["lat"], summ["lon"], alat, alon)
        data["anchor_dist_km"] = round(dist, 1)
        if dist <= _SUPPORT_KM:
            return Evidence(
                type="geo_clip",
                summary=base + f" — {dist:.0f} km from the claimed place",
                verdict=ClaimVerdict.SUPPORTS, confidence=0.55, data=data)
        if dist >= _CONTRADICT_KM:
            return Evidence(
                type="geo_clip",
                summary=base + f" — {dist:.0f} km from the claimed place",
                verdict=ClaimVerdict.CONTRADICTS, confidence=0.55, data=data)
        return Evidence(
            type="geo_clip",
            summary=base + f" — {dist:.0f} km from the claimed place",
            verdict=ClaimVerdict.INCONCLUSIVE, confidence=0.35, data=data)
    except Exception:
        return None


if __name__ == "__main__":
    ok = True

    def check(name, cond, info=""):
        global ok
        ok = ok and bool(cond)
        print(("PASS" if cond else "FAIL"), name, info)

    check("available() is bool", isinstance(available(), bool),
          f"-> {available()}")
    check("predict_coords([]) -> []", predict_coords([]) == [])
    check("predict_coords(None) -> []", predict_coords(None) == [])
    check("pred_summary([]) -> None", pred_summary([]) is None)
    check("pred_summary(junk) -> None", pred_summary([{"x": 1}]) is None
          or isinstance(pred_summary([{"x": 1}]), dict))
    s = pred_summary([{"lat": 51.5, "lon": -0.1, "prob": 0.6, "frames": 2},
                      {"lat": 48.8, "lon": 2.3, "prob": 0.2, "frames": 1}])
    check("pred_summary shape", isinstance(s, dict)
          and abs(s["lat"] - 51.5) < 1e-3 and s["r_km"] >= 80, f"-> {s}")
    ev = geoclip_evidence(
        [{"lat": 51.5, "lon": -0.1, "prob": 0.6, "frames": 2}],
        anchor=(51.5, -0.1), claimed_cc="GB")
    check("support verdict", ev is not None
          and ev.verdict.value == "supports", ev and ev.summary[:60])
    ev2 = geoclip_evidence(
        [{"lat": 51.5, "lon": -0.1, "prob": 0.6, "frames": 1}],
        anchor=(35.7, 139.7))
    check("contradict verdict", ev2 is not None
          and ev2.verdict.value == "contradicts", ev2 and ev2.summary[:60])
    ev3 = geoclip_evidence([{"lat": 1, "lon": 1, "prob": 0.5}], None)
    check("no-anchor inconclusive", ev3 is not None
          and ev3.verdict.value == "inconclusive")
    check("empty -> None", geoclip_evidence([], (0, 0)) is None)
    print("ALL PASS" if ok else "SOME FAILURES")
    raise SystemExit(0 if ok else 1)
