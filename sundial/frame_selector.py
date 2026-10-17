"""Geo frame selector — spend OSINT VLM calls on informative frames.

VLM analysis is the expensive stage in the pipeline, so the geo/OSINT pass
shouldn't burn calls on sky-dominated, blurry, dark, or near-duplicate
frames. GeoSeer-style triage: score each sampled frame on cheap visual
signals that correlate with readable geography — signage edges, facade
texture, street-scene detections — and keep the top few.

geo_score() combines, per frame:
  * edge/text density — Canny edge-pixel ratio over the mid-band rows where
    signs and storefronts live (a 7x7 denoise first, so sensor noise can't
    fake text). Saturating, so merely busy frames plateau.
  * color diversity — quantized unique-color fraction of the denoised
    frame; monochrome sky, fog, and bare interiors score low — and so does
    pure sensor noise, which blurs to a uniform mud.
  * blur penalty — variance of Laplacian on the working gray; smeared
    frames lose up to ~70% of their score.
  * sky penalty — a low-edge top third that is mostly blue/uniform-bright
    reads as sky (clear or overcast) and loses up to ~60%.
  * night penalty — mean luminance under _NIGHT_MEAN fades the score toward
    a floor; signage and shadows both die at night.
  * person/vehicle bonus — YOLO detections of street-scene classes add a
    small saturating bonus: signage clusters where traffic and people do.

Never raises: cv2/numpy missing, unreadable files, and any scoring error
all fall back to _FALLBACK_SCORE (0.5), so select_geo_frames() degrades to
first-N ordering with ties and failures resolved by original frame order.

Stdlib only at module level; cv2/numpy are imported lazily inside
functions so `import sundial.frame_selector` works with zero third-party
deps.
"""
from __future__ import annotations

__all__ = ["geo_score", "select_geo_frames", "report_rows"]

_FALLBACK_SCORE = 0.5    # returned whenever a frame can't be scored

_WORK_EDGE = 320         # score a downscaled copy — cheap + damps codec noise
_MID_BAND = (0.20, 0.78)  # row range (fraction of h) where signage lives
_EDGE_LO, _EDGE_HI = 40, 120   # Canny thresholds on the denoised gray
_EDGE_KNEE = 0.05        # mid-band edge ratio giving edge01 = 0.5
_COL_EDGE = 64           # color-diversity probe size (square)
_COL_FULL = 0.35         # unique-quantized-color fraction reading "diverse"
_BLUR_KNEE = 60.0        # Laplacian variance at half-sharpness
_BLUR_FLOOR = 0.30       # blur factor floor: a hopeless smear still counts a bit
_SKY_EDGE_MAX = 0.015    # top-third edge ratio under this reads "featureless"
_SKY_FULL = 0.60         # sky-pixel fraction at which the cut fully applies
_SKY_CUT = 0.60          # a fully sky-covered top third costs this fraction
_NIGHT_MEAN = 55.0       # mean L under this fades toward _NIGHT_FLOOR
_NIGHT_FLOOR = 0.40      # darkest frames keep this fraction of their score
_STREET_CLASSES = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "parking meter", "fire hydrant",
    "bench", "street sign",
})
_DET_SAT = 6.0           # relevant detections at which the bonus saturates
_W_EDGE, _W_COL, _W_DET = 0.55, 0.30, 0.15   # base-score weights (sum to 1)
_DEDUP_MAD = 8.0         # mean-abs-diff on _THUMB_WH gray: below = same frame
_THUMB_WH = (64, 36)     # dedupe thumbnail size


def _street_det_count(yolo_dets) -> int:
    """Count detections of street-scene classes; tolerates malformed blobs."""
    n = 0
    for d in yolo_dets or []:
        if not isinstance(d, dict):
            continue
        cls = str(d.get("class") or "").strip().lower()
        if cls in _STREET_CLASSES or cls.endswith("sign"):
            n += 1
    return n


def _thumb(frame_path: str):
    """Small gray thumbnail for dedupe; None when cv2 or the file fails."""
    try:
        import cv2  # lazy
    except ImportError:
        return None
    try:
        img = cv2.imread(str(frame_path))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, _THUMB_WH, interpolation=cv2.INTER_AREA)
    except Exception:
        return None


def _mad(a, b) -> float:
    """Mean absolute difference between two same-size uint8 thumbnails."""
    import numpy as np  # lazy

    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def geo_score(frame_path: str, yolo_dets: list | None = None) -> float:
    """Score how geo-informative a frame is for an OSINT VLM call.

    -> 0..1 float; higher means more signage/landmark content worth a call.
    Returns _FALLBACK_SCORE when cv2 is missing, the file can't be decoded,
    or scoring blows up — selection then falls back to first-N ordering.
    `yolo_dets` is an optional perception.detect_objects()-style list whose
    person/vehicle/street-furniture classes add a small saturating bonus.
    """
    try:
        det01 = min(1.0, _street_det_count(yolo_dets) / _DET_SAT)
    except Exception:
        det01 = 0.0

    try:
        import cv2  # lazy
        import numpy as np  # lazy
    except ImportError:
        return _FALLBACK_SCORE

    try:
        img = cv2.imread(str(frame_path))
        if img is None or min(img.shape[:2]) < 16:
            return _FALLBACK_SCORE
        scale = min(1.0, _WORK_EDGE / float(max(img.shape[:2])))
        if scale < 1.0:
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # One shared 7x7 denoise feeds both the edge and color probes: iid
        # sensor noise is the classic way to fake text density and palette
        # diversity without a single readable structure in frame, and a
        # 5x5 kernel still leaks noise gradients past the Canny thresholds.
        den = cv2.GaussianBlur(img, (7, 7), 0)
        dgray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)

        # --- edge/text density in the mid-band rows where signs live ---
        edges = cv2.Canny(dgray, _EDGE_LO, _EDGE_HI)
        b0 = int(h * _MID_BAND[0])
        b1 = max(b0 + 1, int(h * _MID_BAND[1]))
        band = edges[b0:b1]
        edge_ratio = float(np.count_nonzero(band)) / max(1.0, float(band.size))
        edge01 = edge_ratio / (edge_ratio + _EDGE_KNEE)

        # --- color diversity: quantized unique-color fraction ---
        small = cv2.resize(
            den, (_COL_EDGE, _COL_EDGE), interpolation=cv2.INTER_AREA
        )
        q = (small.reshape(-1, 3) // 32).astype(np.int32)
        packed = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
        col_frac = float(np.unique(packed).size) / max(1.0, float(packed.size))
        col01 = min(1.0, col_frac / _COL_FULL)

        # --- blur penalty: variance of Laplacian on the working gray ---
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_factor = _BLUR_FLOOR + (1.0 - _BLUR_FLOOR) * (
            lap_var / (lap_var + _BLUR_KNEE)
        )

        # --- sky penalty: featureless top third that reads blue/uniform ---
        t = max(1, h // 3)
        top_edges = edges[:t]
        top_edge = float(np.count_nonzero(top_edges)) / max(
            1.0, float(top_edges.size)
        )
        sky_frac = 0.0
        if top_edge < _SKY_EDGE_MAX:
            top = img[:t].astype(np.int16)
            bb, gg, rr = top[..., 0], top[..., 1], top[..., 2]
            mx = np.maximum(np.maximum(bb, gg), rr)
            mn = np.minimum(np.minimum(bb, gg), rr)
            mean = (bb + gg + rr) / 3.0
            # clear sky is blue-dominant; overcast reads uniform-bright
            skyish = ((bb > rr + 18) & (bb > gg + 6)) | (
                ((mx - mn) < 14) & (mean > 135.0)
            )
            sky_frac = float(skyish.mean())
        sky_factor = 1.0 - _SKY_CUT * min(1.0, sky_frac / _SKY_FULL)

        # --- night penalty: crushed frames have no readable signage ---
        mean_l = float(gray.mean())
        night_factor = 1.0 if mean_l >= _NIGHT_MEAN else max(
            _NIGHT_FLOOR, mean_l / _NIGHT_MEAN
        )

        base = _W_EDGE * edge01 + _W_COL * col01 + _W_DET * det01
        score = base * blur_factor * sky_factor * night_factor
        return round(min(max(score, 0.0), 1.0), 4)
    except Exception:
        return _FALLBACK_SCORE


def select_geo_frames(
    frame_paths: list, dets_by_frame: dict | None = None, n: int = 6
) -> list:
    """Return up to `n` frame paths, best geo_score first, dupes removed.

    `dets_by_frame` optionally maps frame path -> detect_objects() blobs for
    the street-scene bonus. Score ties and per-frame scoring failures keep
    the original order, so with nothing scorable this degrades to first-N.
    A frame that is near-identical to an already-picked one (mean-abs-diff
    on a 64x36 gray < _DEDUP_MAD) is skipped — the higher-scored copy wins.
    """
    paths = [str(p) for p in (frame_paths or [])]
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 6
    if not paths or n <= 0:
        return []
    dets = dets_by_frame if isinstance(dets_by_frame, dict) else {}

    scored = []
    for idx, p in enumerate(paths):
        try:
            s = float(geo_score(p, dets.get(p)))
        except Exception:
            s = _FALLBACK_SCORE
        scored.append((-s, idx, p))
    scored.sort()  # score desc; ties -> original capture order

    kept: list[str] = []
    thumbs = []
    for _neg_s, _idx, p in scored:
        th = _thumb(p)
        if th is not None and any(
            prev is not None and _mad(th, prev) < _DEDUP_MAD
            for prev in thumbs
        ):
            continue  # near-duplicate of an already-picked frame
        kept.append(p)
        thumbs.append(th)
        if len(kept) >= n:
            break
    return kept


def report_rows(frame_paths, selected) -> list[dict]:
    """Shape per-frame scores for run_stats/report data.

    -> [{"frame": path, "score": float, "picked": bool}] in the caller's
    frame order. Re-scores without detection context, so rows can sit a
    hair below the numbers select_geo_frames() ranked by; unscorable
    frames report _FALLBACK_SCORE.
    """
    picked = {str(p) for p in (selected or [])}
    rows = []
    for p in frame_paths or []:
        p = str(p)
        try:
            s = float(geo_score(p))
        except Exception:
            s = _FALLBACK_SCORE
        rows.append({"frame": p, "score": round(s, 4), "picked": p in picked})
    return rows


if __name__ == "__main__":
    import os
    import tempfile

    # Dependency-free fallback checks — these must hold even with no cv2.
    assert select_geo_frames([]) == []
    assert select_geo_frames(["/tmp/nope_a.png", "/tmp/nope_b.png"], n=1) == [
        "/tmp/nope_a.png"
    ]  # unscorable -> first-N ordering
    assert geo_score("/tmp/definitely/missing.png") == _FALLBACK_SCORE
    assert report_rows(["/tmp/nope_a.png"], []) == [
        {"frame": "/tmp/nope_a.png", "score": _FALLBACK_SCORE, "picked": False}
    ]
    print("fallback paths ok (no third-party deps needed)")

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("cv2/numpy not installed — visual self-test skipped")
        raise SystemExit(0)

    tmp_parent = "/tmp" if os.path.isdir("/tmp") else None
    with tempfile.TemporaryDirectory(
        prefix="sundial_framesel_", dir=tmp_parent
    ) as td:
        h, w = 360, 640
        paths = {}

        # textured: gradient facade + sign-like rectangles and lettering
        tex = np.zeros((h, w, 3), np.uint8)
        tex[:] = np.linspace(40, 200, w, dtype=np.uint8)[None, :, None]
        for x, y, cw, ch, color in [
            (60, 70, 220, 90, (200, 60, 40)),
            (340, 60, 240, 80, (40, 160, 200)),
            (120, 220, 300, 70, (230, 220, 60)),
            (480, 200, 120, 120, (60, 180, 90)),
        ]:
            cv2.rectangle(tex, (x, y), (x + cw, y + ch), color, -1)
            cv2.rectangle(tex, (x, y), (x + cw, y + ch), (255, 255, 255), 3)
        cv2.putText(tex, "HOTEL", (80, 125), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (255, 255, 255), 4)
        cv2.putText(tex, "MARKET ST", (140, 270), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (10, 10, 10), 3)

        # noisy: iid sensor noise — busy pixels, zero geography
        rng = np.random.default_rng(7)
        noise = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        # flat: uniform mid-gray — nothing at all
        flat = np.full((h, w, 3), 200, np.uint8)
        # sky: clear blue top over a dull uniform bottom — little to read
        sky = np.zeros((h, w, 3), np.uint8)
        sky[: h // 2] = (245, 190, 130)   # BGR light blue
        sky[h // 2:] = (120, 110, 100)
        # dark + blurry: degraded copies of the textured frame
        dark = (tex.astype(np.float32) * 0.06).astype(np.uint8)
        blurry = cv2.GaussianBlur(tex, (15, 15), 0)

        for name, arr in [
            ("textured", tex), ("noise", noise), ("flat", flat),
            ("sky", sky), ("dark", dark), ("blurry", blurry),
        ]:
            p = os.path.join(td, f"{name}.png")
            assert cv2.imwrite(p, arr)
            paths[name] = p

        scores = {name: geo_score(p) for name, p in paths.items()}
        for name in ("textured", "noise", "flat", "sky", "dark", "blurry"):
            print(f"{name:8s}: {scores[name]:.3f}")
        assert scores["textured"] > scores["noise"], "textured must beat noise"
        assert scores["textured"] > scores["flat"], "textured must beat flat"
        assert scores["textured"] > scores["sky"], "textured must beat sky"
        assert scores["textured"] > scores["dark"], "textured must beat dark"
        assert scores["textured"] > scores["blurry"], "textured must beat blur"

        # detection bonus: same flat frame, street-scene dets help
        dets = [
            {"class": "person", "confidence": 0.9, "bbox_xyxy": [0, 0, 9, 9]},
            {"class": "car", "confidence": 0.8, "bbox_xyxy": [0, 0, 9, 9]},
            {"class": "stop sign", "confidence": 0.7, "bbox_xyxy": [0, 0, 9, 9]},
        ]
        assert geo_score(paths["flat"], dets) > geo_score(paths["flat"], [])

        # dedupe: identical bytes in a second file collapse into one pick
        dup = os.path.join(td, "tex_dup.png")
        assert cv2.imwrite(dup, tex)
        sel = select_geo_frames(
            [paths["flat"], paths["textured"], dup, paths["noise"]], n=6
        )
        print("selected:", [os.path.basename(p) for p in sel])
        assert sel[0] == paths["textured"], "best frame should lead"
        assert len(sel) == 3, "near-identical copy must dedupe away"
        assert select_geo_frames(
            [paths["textured"], dup, paths["textured"]], n=6
        ) == [paths["textured"]], "all-identical frames dedupe to 1"

        rows = report_rows(list(paths.values()), sel)
        assert all(set(r) == {"frame", "score", "picked"} for r in rows)
        assert sum(r["picked"] for r in rows) == len(sel)
        print("report rows:")
        for r in rows:
            print(" ", r)
        print("frame_selector self-test ok")
