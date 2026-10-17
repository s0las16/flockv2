"""Coordinate + country ensemble — KDE cluster pooling & country-prior re-ranking.

Combines coordinate-level candidates (from GeoCLIP ``candidates``/``alts``,
landmark pins, EXIF GPS, wiki places) with country-level priors (from CLIP
``clip_geo``, ``geotips``, ``geo_match``, ``hier_geoguess``) to re-rank
coordinate guesses.

Notice how in a typical multi-candidate GeoCLIP frame (e.g. Bonn test), #1
and #2 might be Mannheim (49.44, 8.50, p=0.04) while #3 and #4 are Bonn
(50.743, 7.169 and 50.744, 7.153, p=0.02 each). When all top-k candidates
across frames are pooled via Haversine Gaussian Kernel Density Estimation
(sigma=60 km) AND optionally re-weighted by country/region priors or nearby
landmark/OSM hits, the true dense cluster reinforces itself and wins.

Pure stdlib at import time; ``pip2cc`` and ``schema`` are imported lazily
with fallbacks. Every public function is failure-isolated and never raises
on any input.

Stage API:

  * ``kde_rank_candidates(frame_preds, country_weights=None, sigma_km=60.0)``
        -> ``[{"lat": float, "lon": float, "score": float,
               "support_count": int, "spread_km": float}, ...]``
  * ``ensemble_summary(clusters)``
        -> ``{"lat": float, "lon": float, "radius_km": float,
               "score": float, "support_count": int} | None``
  * ``country_weights_from_evidence(evidence_list)``
        -> ``dict[str, float]``
  * ``candidates_from_evidence(evidence_list)``
        -> ``list[list[tuple[float, float, float]]]``
  * ``ensemble_evidence(clusters, anchor=None, claimed_cc=None)``
        -> ``Evidence | None``

    python3 sundial/geo_ensemble.py     # self-test
    python -m sundial.geo_ensemble
"""
from __future__ import annotations

import math

__all__ = [
    "kde_rank_candidates",
    "ensemble_summary",
    "country_weights_from_evidence",
    "candidates_from_evidence",
    "ensemble_evidence",
]

_EARTH_R_KM = 6371.0088
_DEFAULT_SIGMA_KM = 60.0
_MIN_RADIUS_KM = 25.0
_MAX_RADIUS_KM = 1500.0
_SUPPORT_KM = 250.0
_CONTRADICT_KM = 1500.0


def _schema():
    """Lazy schema import with duck-typed fallback so standalone never fails."""
    try:
        from .schema import ClaimVerdict, Evidence
        return ClaimVerdict, Evidence
    except Exception:
        pass
    try:
        from schema import ClaimVerdict, Evidence  # direct-script context
        return ClaimVerdict, Evidence
    except Exception:
        pass
    from dataclasses import dataclass, field

    class ClaimVerdict:
        SUPPORTS = "supports"
        CONTRADICTS = "contradicts"
        INCONCLUSIVE = "inconclusive"

    @dataclass
    class Evidence:
        type: str
        summary: str
        verdict: object = ClaimVerdict.INCONCLUSIVE
        confidence: float = 0.0
        data: dict = field(default_factory=dict)
        frames: list = field(default_factory=list)

    return ClaimVerdict, Evidence


def _pip2cc():
    """Lazy import of pip2cc; returns module or None."""
    try:
        from . import pip2cc
        return pip2cc
    except Exception:
        pass
    try:
        import pip2cc  # direct-script context
        return pip2cc
    except Exception:
        return None


def _num(v) -> float | None:
    """Best-effort finite float; None on bad input."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _latlon(lat, lon) -> tuple[float, float] | None:
    """Validate (lat, lon) in range [-90, 90] x [-180, 180]."""
    la, lo = _num(lat), _num(lon)
    if la is None or lo is None:
        return None
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return None
    return la, lo


def _norm_cc(v) -> str | None:
    """Normalize a 2-letter ISO country code."""
    if isinstance(v, dict):
        for k in ("cc", "country_code", "code", "iso", "iso_cc", "country"):
            got = _norm_cc(v.get(k))
            if got:
                return got
        return None
    s = str(v or "").strip().upper()
    if len(s) == 2 and s.isalpha():
        return "GB" if s == "UK" else s
    return None


def _cc_at(lat: float, lon: float) -> str | None:
    """Lookup country code at (lat, lon) via pip2cc lazily; None on miss."""
    mod = _pip2cc()
    if mod is None:
        return None
    try:
        cc = mod.cc_at(lat, lon)
        if cc:
            return _norm_cc(cc)
        near = mod.nearest_cc(lat, lon, k=1)
        if near and isinstance(near[0], dict):
            km = _num(near[0].get("km"))
            if km is not None and km <= 40.0:
                return _norm_cc(near[0].get("cc"))
    except Exception:
        return None
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = (
        math.sin(dp * 0.5) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    )
    return 2.0 * _EARTH_R_KM * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _spherical_mean(points_with_weights: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Weighted spherical mean of (lat, lon, weight) tuples."""
    if not points_with_weights:
        return 0.0, 0.0
    wsum = sum(max(0.0, w) for _, _, w in points_with_weights)
    if wsum <= 0.0:
        n = len(points_with_weights)
        return (
            sum(la for la, _, _ in points_with_weights) / n,
            sum(lo for _, lo, _ in points_with_weights) / n,
        )
    x = y = z = 0.0
    for la, lo, w in points_with_weights:
        ww = max(0.0, w) / wsum
        p, l = math.radians(la), math.radians(lo)
        x += ww * math.cos(p) * math.cos(l)
        y += ww * math.cos(p) * math.sin(l)
        z += ww * math.sin(p)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm > 1e-9:
        return (
            math.degrees(math.atan2(z, math.sqrt(x * x + y * y))),
            math.degrees(math.atan2(y, x)),
        )
    return (
        sum(la * max(0.0, w) for la, _, w in points_with_weights) / wsum,
        sum(lo * max(0.0, w) for _, lo, w in points_with_weights) / wsum,
    )


def _extract_candidates(frame_preds) -> list[tuple[float, float, float, str | None]]:
    """Flatten arbitrary candidate inputs into [(lat, lon, prob, cc_hint), ...].

    Accepts:
      - list of frames: [[(lat, lon, prob), ...], ...]
      - flat list of tuples: [(lat, lon, prob), ...] or [(lat, lon), ...]
      - flat or nested list of dicts: [{"lat": .., "lon": .., "prob"|"score"|"weight": .., "alts": [...]}, ...]
      - single dict with "candidates" / "places" / "alts"
    """
    out: list[tuple[float, float, float, str | None]] = []

    def _visit(item, default_prob: float = 0.05):
        if item is None:
            return
        if isinstance(item, dict):
            pt = _latlon(
                item.get("lat") if item.get("lat") is not None else item.get("gps_lat"),
                item.get("lon") if item.get("lon") is not None else item.get("gps_lon"),
            )
            cc_hint = _norm_cc(item)
            if pt is not None:
                p = None
                for k in ("prob", "score", "weight", "confidence"):
                    p = _num(item.get(k))
                    if p is not None:
                        break
                prob = max(1e-6, p if p is not None else default_prob)
                frames_mult = _num(item.get("frames"))
                if frames_mult is not None and frames_mult > 1 and "alts" not in item:
                    # A pre-pooled row without explicit alts represents `frames` hits
                    out.append((pt[0], pt[1], prob, cc_hint))
                else:
                    out.append((pt[0], pt[1], prob, cc_hint))
            # Also unpack nested alts / candidates / places / matches if present
            for sub_key in ("alts", "candidates", "places", "matches", "hits", "preds"):
                sub = item.get(sub_key)
                if isinstance(sub, (list, tuple)):
                    for s in sub:
                        _visit(s, default_prob=default_prob * 0.5)
            return

        if isinstance(item, (list, tuple)):
            # Check if item itself is a coordinate tuple: (lat, lon[, prob[, cc]])
            if (
                len(item) >= 2
                and not isinstance(item[0], (list, tuple, dict))
                and not isinstance(item[1], (list, tuple, dict))
            ):
                pt = _latlon(item[0], item[1])
                if pt is not None:
                    p = _num(item[2]) if len(item) >= 3 else None
                    prob = max(1e-6, p if p is not None else default_prob)
                    cc_hint = _norm_cc(item[3]) if len(item) >= 4 else None
                    out.append((pt[0], pt[1], prob, cc_hint))
                    return
            # Otherwise iterate as a container of frames / candidates
            for sub in item:
                _visit(sub, default_prob=default_prob)

    try:
        _visit(frame_preds)
    except Exception:
        pass
    return out


def _norm_country_weights(country_weights) -> dict[str, float]:
    """Normalize country_weights mapping to {ISO_CC: non-negative float}."""
    if not country_weights or not isinstance(country_weights, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in country_weights.items():
        try:
            cc = _norm_cc(k)
            w = _num(v)
            if cc and w is not None and w >= 0.0:
                out[cc] = float(w)
        except Exception:
            continue
    return out


def kde_rank_candidates(
    frame_preds: list[list[tuple[float, float, float]]] | list[dict] | None,
    country_weights: dict[str, float] | None = None,
    sigma_km: float = _DEFAULT_SIGMA_KM,
) -> list[dict]:
    """Rank coordinate candidates via Haversine Gaussian KDE + country priors.

    Parameters
    ----------
    frame_preds:
        Per-frame top-k ``(lat, lon, prob)`` predictions (or flat candidate
        dicts / tuples).
    country_weights:
        Optional ``{cc: weight}`` map from country-level priors. When given,
        each candidate's weight is multiplied by ``country_weights.get(cc, 1.0)``
        where ``cc`` is resolved via ``pip2cc`` (lazily, if available) or any
        ``cc`` field on the candidate dict.
    sigma_km:
        Gaussian kernel bandwidth in kilometres (default 60.0 km).

    Returns
    -------
    list[dict]
        Clusters sorted by ``score`` descending:
        ``[{"lat": float, "lon": float, "score": float,
            "support_count": int, "spread_km": float}, ...]``
        Never raises; returns ``[]`` on empty or invalid input.
    """
    try:
        raw_cands = _extract_candidates(frame_preds)
        if not raw_cands:
            return []

        sig = _num(sigma_km)
        if sig is None or sig <= 0.0:
            sig = _DEFAULT_SIGMA_KM
        two_sig_sq = 2.0 * sig * sig

        cw = _norm_country_weights(country_weights)

        # Compute effective prior-adjusted weight per candidate point
        weighted_pts: list[tuple[float, float, float]] = []
        for la, lo, prob, cc_hint in raw_cands:
            mult = 1.0
            if cw:
                cc = cc_hint or _cc_at(la, lo)
                if cc:
                    mult = cw.get(cc, 1.0)
            w = max(0.0, float(prob) * max(0.0, float(mult)))
            weighted_pts.append((la, lo, w))

        n = len(weighted_pts)
        # Pairwise Haversine distance matrix and Gaussian KDE density at each point
        dists: list[list[float]] = [[0.0] * n for _ in range(n)]
        kde_scores: list[float] = [0.0] * n

        for i in range(n):
            la_i, lo_i, w_i = weighted_pts[i]
            kde_scores[i] += w_i  # self-contribution (d=0 -> exp(0)=1)
            for j in range(i + 1, n):
                la_j, lo_j, w_j = weighted_pts[j]
                d = _haversine_km(la_i, lo_i, la_j, lo_j)
                dists[i][j] = d
                dists[j][i] = d
                k_val = math.exp(-(d * d) / two_sig_sq)
                kde_scores[i] += w_j * k_val
                kde_scores[j] += w_i * k_val

        # Group points into clusters around density peaks (greedy mode-seeking
        # within merge radius = sigma_km so tight clusters like Bonn
        # #3+#4 merge into one reinforced peak while cities >60 km apart stay separate)
        merge_km = max(1.0, sig)
        order = sorted(range(n), key=lambda idx: (-kde_scores[idx], -weighted_pts[idx][2], idx))
        assigned = [False] * n
        clusters: list[dict] = []

        for seed_idx in order:
            if assigned[seed_idx]:
                continue
            members = [seed_idx]
            assigned[seed_idx] = True
            for other_idx in order:
                if assigned[other_idx]:
                    continue
                if dists[seed_idx][other_idx] <= merge_km:
                    members.append(other_idx)
                    assigned[other_idx] = True

            # Compute KDE-weighted cluster centroid (mean-shift step)
            member_triples = [
                (
                    weighted_pts[m][0],
                    weighted_pts[m][1],
                    max(weighted_pts[m][2], kde_scores[m], 1e-9),
                )
                for m in members
            ]
            c_lat, c_lon = _spherical_mean(member_triples)

            # Evaluate true KDE score at the cluster centroid over all candidates
            c_score = 0.0
            for la_k, lo_k, w_k in weighted_pts:
                d_k = _haversine_km(c_lat, c_lon, la_k, lo_k)
                c_score += w_k * math.exp(-(d_k * d_k) / two_sig_sq)

            # A tight multi-candidate cluster reinforces itself via coherent
            # spatial agreement across candidates/frames (log-support boost
            # weighted by compactness exp(-spread / sigma_km)).
            support_count = len(members)
            if support_count <= 1:
                spread_km = 0.0
            else:
                mem_weights = [max(weighted_pts[m][2], 1e-6) for m in members]
                w_total = sum(mem_weights) or 1.0
                mem_dists = [
                    _haversine_km(c_lat, c_lon, weighted_pts[m][0], weighted_pts[m][1])
                    for m in members
                ]
                spread_km = sum(w * d for w, d in zip(mem_weights, mem_dists)) / w_total
                compactness = math.exp(-spread_km / sig)
                c_score *= 1.0 + 0.15 * math.log1p(support_count - 1) * compactness

            # Also guarantee score is at least the seed's KDE score
            c_score = max(c_score, kde_scores[seed_idx])

            clusters.append({
                "lat": round(float(c_lat), 5),
                "lon": round(float(c_lon), 5),
                "score": round(float(c_score), 6),
                "support_count": int(support_count),
                "spread_km": round(float(spread_km), 2),
            })

        clusters.sort(
            key=lambda c: (-c["score"], -c["support_count"], c["spread_km"])
        )
        return clusters
    except Exception:
        return []


def ensemble_summary(clusters: list[dict] | None) -> dict | None:
    """Summarize ranked KDE clusters into a single map prediction dict.

    Returns ``{"lat": float, "lon": float, "radius_km": float,
               "score": float, "support_count": int}`` or ``None`` if
    ``clusters`` is empty or invalid. Never raises.
    """
    try:
        if not clusters or not isinstance(clusters, (list, tuple)):
            return None

        valid: list[dict] = []
        for c in clusters:
            if not isinstance(c, dict):
                continue
            pt = _latlon(c.get("lat"), c.get("lon"))
            if pt is None:
                continue
            sc = _num(c.get("score"))
            if sc is None:
                sc = _num(c.get("prob")) or 0.0
            sup = _num(c.get("support_count"))
            if sup is None:
                sup = _num(c.get("frames")) or 1.0
            sp = _num(c.get("spread_km")) or 0.0
            valid.append({
                "lat": pt[0],
                "lon": pt[1],
                "score": max(0.0, float(sc)),
                "support_count": max(1, int(round(sup))),
                "spread_km": max(0.0, float(sp)),
            })

        if not valid:
            return None

        best = valid[0]
        total_score = sum(c["score"] for c in valid) or 1.0
        top_share = best["score"] / total_score if total_score > 0 else 1.0

        # Inter-cluster dispersion weighted by runner-up scores
        inter_spread = 0.0
        for c in valid[1:]:
            w = c["score"] / total_score
            inter_spread += w * _haversine_km(
                best["lat"], best["lon"], c["lat"], c["lon"]
            )

        # Base radius reflects intra-cluster spread + inter-cluster uncertainty;
        # multi-point support tightens the radius.
        support_bonus = 1.0 / math.sqrt(max(1, best["support_count"]))
        base_r = max(_MIN_RADIUS_KM, 60.0 * support_bonus + best["spread_km"] * 1.5)
        radius_km = base_r + inter_spread * (1.0 - 0.5 * top_share)
        radius_km = max(_MIN_RADIUS_KM, min(_MAX_RADIUS_KM, radius_km))

        return {
            "lat": round(best["lat"], 5),
            "lon": round(best["lon"], 5),
            "radius_km": round(radius_km, 1),
            "score": round(best["score"], 6),
            "support_count": int(best["support_count"]),
        }
    except Exception:
        return None


def country_weights_from_evidence(evidence_list) -> dict[str, float]:
    """Extract a ``{cc: multiplier}`` prior map from an EvidencePack or list.

    Consults country-level cards (``clip_geo``, ``geotips``, ``geo_match`` /
    ``osint``, ``geoguess`` / ``hier_geoguess``). Countries supported by
    priors receive multipliers > 1.0; unmentioned countries default to 1.0.
    Never raises.
    """
    weights: dict[str, float] = {}
    try:
        items = evidence_list
        if items is None:
            return {}
        if not isinstance(items, (list, tuple)):
            items = getattr(items, "evidence", None) or [items]

        for ev in items:
            try:
                if isinstance(ev, dict):
                    t = str(ev.get("type") or ev.get("stage") or "").lower()
                    d = ev.get("data") if isinstance(ev.get("data"), dict) else {}
                else:
                    t = str(getattr(ev, "type", "") or "").lower()
                    d = getattr(ev, "data", None)
                    d = d if isinstance(d, dict) else {}

                # 1. clip_geo prior: data["prior"] = [{"cc": "DE", "score": 0.27}, ...]
                if isinstance(d.get("prior"), (list, tuple)):
                    for row in d["prior"]:
                        if isinstance(row, dict):
                            cc = _norm_cc(row)
                            sc = _num(row.get("score"))
                            if cc and sc is not None and sc > 0:
                                weights[cc] = weights.get(cc, 1.0) * (1.0 + min(2.0, sc * 2.5))

                # 2. osint / geo_match ranked: data["ranked"] = [{"cc": "DE", "score": 0.72}, ...]
                for rkey in ("ranked", "ranking", "top3"):
                    raw = d.get(rkey)
                    if isinstance(raw, dict):
                        raw = [{"cc": k, "score": v} for k, v in raw.items()]
                    if isinstance(raw, (list, tuple)):
                        for row in raw:
                            if isinstance(row, dict):
                                cc = _norm_cc(row)
                                sc = _num(row.get("score") or row.get("confidence"))
                                if cc and sc is not None:
                                    # neutral is 0.5 in match_score; >0.5 boosts, <0.5 dampens
                                    factor = max(0.3, min(2.5, 0.5 + sc))
                                    weights[cc] = weights.get(cc, 1.0) * factor

                # 3. geotips scores: data["scores"] or data["top"]
                scores_map = d.get("scores")
                if isinstance(scores_map, dict):
                    for k, v in scores_map.items():
                        cc = _norm_cc(k)
                        sc = _num(v)
                        if cc and sc is not None and sc > 0:
                            weights[cc] = weights.get(cc, 1.0) * (1.0 + min(1.5, sc * 0.3))

                # 4. geoguess / hier_geoguess consensus: data["stable_cc"]
                if "geoguess" in t or "stable_cc" in d:
                    scc = _norm_cc(d.get("stable_cc"))
                    if not scc and isinstance(d.get("geoguess"), dict):
                        scc = _norm_cc(d["geoguess"].get("stable_cc"))
                    if scc:
                        weights[scc] = weights.get(scc, 1.0) * 1.5
            except Exception:
                continue
    except Exception:
        return {}
    return {k: round(v, 4) for k, v in weights.items()}


def candidates_from_evidence(evidence_list) -> list[list[tuple[float, float, float]]]:
    """Extract coordinate candidates from an EvidencePack or list of Evidence.

    Collects GeoCLIP ``candidates`` & ``alts``, landmark pins, wiki places,
    and EXIF GPS fixes. Never raises.
    """
    groups: list[list[tuple[float, float, float]]] = []
    try:
        items = evidence_list
        if items is None:
            return []
        if not isinstance(items, (list, tuple)):
            items = getattr(items, "evidence", None) or [items]

        for ev in items:
            try:
                if isinstance(ev, dict):
                    t = str(ev.get("type") or "").lower()
                    d = ev.get("data") if isinstance(ev.get("data"), dict) else {}
                else:
                    t = str(getattr(ev, "type", "") or "").lower()
                    d = getattr(ev, "data", None)
                    d = d if isinstance(d, dict) else {}

                frame_cands: list[tuple[float, float, float]] = []
                for key in ("candidates", "places", "matches", "hits"):
                    for item in d.get(key) or []:
                        for la, lo, p, _ in _extract_candidates(item):
                            frame_cands.append((la, lo, p))
                if not frame_cands:
                    gps = d.get("gps") if isinstance(d.get("gps"), dict) else d
                    pt = _latlon(
                        gps.get("lat") if gps.get("lat") is not None else gps.get("gps_lat"),
                        gps.get("lon") if gps.get("lon") is not None else gps.get("gps_lon"),
                    )
                    if pt is not None and ("geo" in t or "exif" in t or "gps" in t or "landmark" in t):
                        prob = _num(d.get("prob") or d.get("score")) or 0.25
                        frame_cands.append((pt[0], pt[1], max(0.01, prob)))
                if frame_cands:
                    groups.append(frame_cands)
            except Exception:
                continue
    except Exception:
        return []
    return groups


def ensemble_evidence(
    clusters: list[dict] | None,
    anchor=None,
    claimed_cc: str | None = None,
):
    """Package KDE-ranked coordinate clusters vs claimed anchor into Evidence.

    Returns ``None`` when ``clusters`` is empty or invalid. Never raises.
    """
    try:
        ClaimVerdict, Evidence = _schema()
        summ = ensemble_summary(clusters)
        if summ is None:
            return None

        alat = alon = None
        try:
            if anchor is not None:
                if isinstance(anchor, dict):
                    pt = _latlon(anchor.get("lat"), anchor.get("lon"))
                elif isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
                    pt = _latlon(anchor[0], anchor[1])
                else:
                    pt = _latlon(getattr(anchor, "lat", None), getattr(anchor, "lon", None))
                if pt is not None:
                    alat, alon = pt
        except Exception:
            alat = alon = None

        top_cc = _cc_at(summ["lat"], summ["lon"])
        data = {
            "lat": summ["lat"],
            "lon": summ["lon"],
            "radius_km": summ["radius_km"],
            "score": summ["score"],
            "support_count": summ["support_count"],
            "top_cc": top_cc,
            "claimed_cc": _norm_cc(claimed_cc),
            "clusters": list(clusters[:5]) if isinstance(clusters, (list, tuple)) else [],
        }

        sup = summ["support_count"]
        cc_tag = f" ({top_cc})" if top_cc else ""
        base = (
            f"Geo ensemble KDE pins {summ['lat']:.3f}, {summ['lon']:.3f}{cc_tag} "
            f"(±{summ['radius_km']:.0f} km, {sup} candidate{'s' if sup != 1 else ''}, "
            f"score {summ['score']:.3f})"
        )

        if alat is None or alon is None:
            return Evidence(
                type="geo_ensemble",
                summary=base,
                verdict=ClaimVerdict.INCONCLUSIVE,
                confidence=0.35,
                data=data,
            )

        dist = _haversine_km(summ["lat"], summ["lon"], alat, alon)
        data["anchor_dist_km"] = round(dist, 1)
        if dist <= max(_SUPPORT_KM, summ["radius_km"]):
            return Evidence(
                type="geo_ensemble",
                summary=f"{base} — {dist:.0f} km from claimed place",
                verdict=ClaimVerdict.SUPPORTS,
                confidence=0.55,
                data=data,
            )
        if dist >= _CONTRADICT_KM:
            return Evidence(
                type="geo_ensemble",
                summary=f"{base} — {dist:.0f} km from claimed place",
                verdict=ClaimVerdict.CONTRADICTS,
                confidence=0.55,
                data=data,
            )
        return Evidence(
            type="geo_ensemble",
            summary=f"{base} — {dist:.0f} km from claimed place",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.35,
            data=data,
        )
    except Exception:
        return None


if __name__ == "__main__":
    import sys

    ok = True

    def check(name: str, cond: bool, info: str = "") -> None:
        global ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {info}")

    # 1. Bonn vs Mannheim test case from specification:
    # #1 is Mannheim (49.44, 8.50, p=0.04) and #3/#4 are Bonn
    # (50.743, 7.169, p=0.02) and (50.744, 7.153, p=0.02) that reinforce
    # each other via Gaussian KDE!
    bonn_frame_preds = [
        [
            (49.440, 8.500, 0.040),  # #1 Mannheim (p=0.04)
            (48.137, 11.576, 0.030), # #2 Munich
            (50.743, 7.169, 0.020),  # #3 Bonn A (p=0.02)
            (50.744, 7.153, 0.020),  # #4 Bonn B (p=0.02, ~1.1 km from Bonn A!)
        ]
    ]
    ranked = kde_rank_candidates(bonn_frame_preds, sigma_km=60.0)
    check(
        "Bonn cluster beats isolated Mannheim top-1",
        len(ranked) >= 2
        and abs(ranked[0]["lat"] - 50.7435) < 0.05
        and abs(ranked[0]["lon"] - 7.161) < 0.05
        and ranked[0]["support_count"] == 2,
        f"-> top={ranked[0] if ranked else None}",
    )
    check(
        "cluster dict keys match spec",
        bool(ranked)
        and set(ranked[0].keys()) == {"lat", "lon", "score", "support_count", "spread_km"}
        and ranked[0]["spread_km"] < 5.0,
        f"-> keys={list(ranked[0].keys()) if ranked else []}",
    )

    # 2. Multi-frame reinforcement test
    multi_frames = [
        [(49.44, 8.50, 0.04), (50.743, 7.169, 0.022), (50.744, 7.153, 0.020)],
        [(53.55, 9.99, 0.038), (50.735, 7.100, 0.021)],
    ]
    ranked_multi = kde_rank_candidates(multi_frames, sigma_km=60.0)
    check(
        "multi-frame Bonn cluster pools 3 candidates",
        bool(ranked_multi)
        and abs(ranked_multi[0]["lat"] - 50.74) < 0.05
        and ranked_multi[0]["support_count"] == 3,
        f"-> top={ranked_multi[0] if ranked_multi else None}",
    )

    # 3. Country weights re-ranking test (using pip2cc FR vs ES)
    # Candidate in Madrid (ES: 40.4168, -3.7038, p=0.05) vs Paris (FR: 48.8566, 2.3522, p=0.04)
    fr_es_preds = [[(40.4168, -3.7038, 0.05), (48.8566, 2.3522, 0.04)]]
    ranked_fr = kde_rank_candidates(fr_es_preds, country_weights={"FR": 2.0, "ES": 0.5})
    check(
        "country_weights boosts FR (Paris) above ES (Madrid)",
        bool(ranked_fr) and abs(ranked_fr[0]["lat"] - 48.8566) < 0.1,
        f"-> top={ranked_fr[0] if ranked_fr else None}",
    )

    # 4. Flat candidate dicts with 'alts' (GeoCLIP predict_coords output shape)
    dict_cands = [
        {
            "lat": 49.44,
            "lon": 8.50,
            "prob": 0.04,
            "frames": 1,
            "alts": [(50.743, 7.169, 0.025), (50.744, 7.153, 0.025)],
        }
    ]
    ranked_dicts = kde_rank_candidates(dict_cands, sigma_km=60.0)
    check(
        "accepts GeoCLIP dict candidates with alts",
        bool(ranked_dicts) and abs(ranked_dicts[0]["lat"] - 50.7435) < 0.05,
        f"-> top={ranked_dicts[0] if ranked_dicts else None}",
    )

    # 5. ensemble_summary shape & values
    summ = ensemble_summary(ranked_multi)
    check(
        "ensemble_summary returns expected keys",
        isinstance(summ, dict)
        and set(summ.keys()) == {"lat", "lon", "radius_km", "score", "support_count"}
        and summ["support_count"] == 3
        and summ["radius_km"] >= 25.0,
        f"-> {summ}",
    )
    check("ensemble_summary([]) -> None", ensemble_summary([]) is None)
    check("ensemble_summary(None) -> None", ensemble_summary(None) is None)

    # 6. Defensive / bad input checks (must NEVER raise)
    for bad in (None, [], [[]], "garbage", 123, [None, {"lat": "bad", "lon": None}], [(999, 999, 0.5)]):
        res = kde_rank_candidates(bad, country_weights="invalid", sigma_km=-10)
        check(f"kde_rank_candidates({bad!r}) -> []", res == [])

    # 7. Evidence helper check
    ev = ensemble_evidence(ranked_multi, anchor=(50.73, 7.10), claimed_cc="DE")
    check(
        "ensemble_evidence supports nearby Bonn anchor",
        ev is not None and getattr(ev.verdict, "value", ev.verdict) == "supports",
        f"-> {ev.summary if ev else None}",
    )

    print("ALL PASS" if ok else "SOME FAILURES")
    sys.exit(0 if ok else 1)
