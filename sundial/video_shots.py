"""Shot/scene-boundary detection — one frame per shot, not near-dupes.

Per-frame VLM/OSINT calls are the expensive stage of the pipeline, and a
fixed-fps sweep spends most of them on near-duplicate imagery (static
stretches, slow pans). Detecting shot boundaries first and sampling one
frame per shot buys real visual diversity for the same call budget.

Backend order for shot_boundaries():
  1. ffprobe — the clip is fed through lavfi `movie=...,select=gt(scene,T)`
     and -show_frames reports the surviving frames' pts times + scores.
  2. cv2     — HSV-histogram Bhattacharyya distance scanned at ~2 fps.
  3. neither — []; callers fall back to plain even division.

diverse_times() turns boundaries into sample timestamps: one midpoint per
shot, subdivided evenly to pad up to `n` when shots are scarce. A clip
with no detected cuts is a single shot, so the fallback is n
evenly-spaced middle times — the classic spread a 1 fps sweep approximates.

Stdlib only at import; ffprobe/ffmpeg/cv2 are located lazily inside
functions. Nothing here raises — missing binaries, unreadable clips and
failed decodes all degrade to [].
"""
from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess

__all__ = ["shot_boundaries", "diverse_times", "shot_frame_paths"]

_SCENE_THRESH = 0.4       # ffmpeg scene score above this reads as a cut
_SCAN_FPS = 2.0           # cv2 fallback samples the clip at this rate
_CV2_DIFF_THRESH = 0.5    # Bhattacharyya distance (0..1) that reads as a cut
_MIN_SHOT_S = 0.8         # detections closer than this collapse to one cut
_MAX_SCAN_S = 600.0       # bound the pure-Python cv2 loop on long clips
_FFPROBE_TIMEOUT = 240    # -show_frames decodes every frame — heavy
_SEEK_TIMEOUT = 60        # per single-frame jpeg extract


# --------------------------------------------------------------------------
# External binaries (lazy)
# --------------------------------------------------------------------------

def _find_ffprobe() -> str | None:
    """Locate an ffprobe executable; None when absent."""
    return shutil.which("ffprobe")


def _find_ffmpeg() -> str | None:
    """Locate an ffmpeg executable.

    Falls back to the static binary bundled with `imageio-ffmpeg` — common
    on machines where imageio is installed but system ffmpeg is not. Same
    lazy pattern as perception._find_ffmpeg; harmless when uninstalled.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _duration_s(video_path: str) -> float:
    """Container duration in seconds; 0.0 when nothing can read it.

    diverse_times() needs a right edge to place the last shot's midpoint
    and to space the single-shot fallback — ffprobe's format=duration is
    tried first, cv2's frame-count/fps second.
    """
    exe = _find_ffprobe()
    if exe:
        try:
            proc = subprocess.run(
                [
                    exe,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            d = float(proc.stdout.strip().splitlines()[0])
            if proc.returncode == 0 and math.isfinite(d) and d > 0:
                return d
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    try:
        import cv2
    except ImportError:
        return 0.0
    try:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return 0.0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
    except Exception:
        return 0.0
    if fps and frames and math.isfinite(fps) and math.isfinite(frames) and fps > 0:
        return frames / fps
    return 0.0


# --------------------------------------------------------------------------
# Boundary detection — ffprobe primary, cv2 fallback
# --------------------------------------------------------------------------

def _movie_graph(video_path: str, thresh: float) -> str:
    """lavfi filtergraph `movie='<path>',select='gt(scene,t)'`.

    Inside '...' the filtergraph parser only honours \\ and \\' escapes —
    backslashes first, quotes second, everything else stays literal.
    """
    esc = video_path.replace("\\", "\\\\").replace("'", "\\'")
    return f"movie='{esc}',select='gt(scene,{thresh:g})'"


def _boundaries_ffprobe(
    video_path: str, thresh: float
) -> list[tuple[float, float]] | None:
    """Scene-cut pts times via ffprobe's lavfi movie source.

    Returns [(time_s, scene_score)] for frames where select=gt(scene,T)
    passes. Three-way contract: [] means "probe ran, no cuts found" —
    authoritative, do NOT fall back; None means "couldn't run" (binary
    missing, bad clip, parse failure) and the caller tries cv2.
    Decodes every frame, hence the generous timeout.
    """
    exe = _find_ffprobe()
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [
                exe,
                "-v", "error",
                "-f", "lavfi",
                "-i", _movie_graph(video_path, thresh),
                "-show_frames",
                "-of", "json",
            ],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return None
    frames = data.get("frames") if isinstance(data, dict) else []
    frames = frames or []

    hits: list[tuple[float, float]] = []
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        # pts_time on modern ffprobe, pkt_pts_time on older builds.
        raw_t = fr.get("pts_time", fr.get("pkt_pts_time"))
        try:
            t = float(raw_t)
        except (TypeError, ValueError):
            continue
        score = 0.0
        tags = fr.get("tags")
        if isinstance(tags, dict):
            try:
                score = float(tags.get("lavfi.scene_score") or 0.0)
            except (TypeError, ValueError):
                pass
        hits.append((t, score))
    return hits


def _boundaries_cv2(
    video_path: str, thresh: float = _CV2_DIFF_THRESH
) -> list[tuple[float, float]] | None:
    """Histogram-diff cut scan at ~_SCAN_FPS; None when cv2 can't help.

    Each sampled frame's H/S histogram is compared to the previous
    sample's with the Bhattacharyya distance (0 = identical, ~1 = no
    overlap). Hard cuts spike it; pans and fades ride under the threshold
    at this stride. The score is the diff itself, so strongest-first
    capping works the same as the ffprobe path.
    """
    try:
        import cv2
    except ImportError:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception:
        return None

    hits: list[tuple[float, float]] = []
    try:
        if not cap.isOpened():
            return None
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        # NaN/0 fps both occur on odd containers; assume a normal rate.
        if not (vid_fps and math.isfinite(vid_fps) and vid_fps > 0):
            vid_fps = 25.0
        step = max(1, round(vid_fps / _SCAN_FPS))
        max_idx = int(vid_fps * _MAX_SCAN_S)
        prev = None
        idx = 0
        try:
            while idx <= max_idx:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist(
                        [hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]
                    )
                    if hist is not None and hist.sum() > 0:
                        hist = hist / hist.sum()  # L1-normalise for compareHist
                        if prev is not None:
                            d = float(
                                cv2.compareHist(
                                    prev, hist, cv2.HISTCMP_BHATTACHARYYA
                                )
                            )
                            if d >= thresh:
                                hits.append((idx / vid_fps, d))
                        prev = hist
                idx += 1
        except Exception:
            pass  # mid-scan codec burp: keep the hits already collected
        return hits
    finally:
        try:
            cap.release()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Post-processing: merge clusters, cap the count
# --------------------------------------------------------------------------

def _merge_close(
    hits: list[tuple[float, float]], min_gap: float = _MIN_SHOT_S
) -> list[tuple[float, float]]:
    """Merge detections closer than min_gap into one cut (strongest wins).

    A real cut can straddle two 2 fps samples and double-report; a
    whip-pan rattles off a cluster. Clustering on gap and keeping the
    peak score collapses those to the single transition they came from.
    """
    merged: list[tuple[float, float]] = []
    for t, s in sorted(hits):
        if merged and t - merged[-1][0] < min_gap:
            if s > merged[-1][1]:
                merged[-1] = (t, s)
            continue
        merged.append((t, s))
    return merged


def _strongest(
    hits: list[tuple[float, float]], k: int
) -> list[tuple[float, float]]:
    """Keep k cuts: strongest scores first, evenly spread when unscored.

    ffprobe's scene_score ranks cuts by how hard the imagery changes;
    when every score is identical (tag missing, flat cv2 run) score order
    would just keep the earliest cuts, so fall back to an evenly-indexed
    subset that still covers the whole clip.
    """
    if len(hits) <= k:
        return hits
    if len({s for _, s in hits}) <= 1:  # scores carry no information
        if k == 1:
            return [hits[len(hits) // 2]]
        step = (len(hits) - 1) / (k - 1)
        return [hits[round(i * step)] for i in range(k)]
    return sorted(hits, key=lambda h: h[1], reverse=True)[:k]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def shot_boundaries(video_path: str, max_shots: int = 8) -> list[float]:
    """Detected shot-boundary times (s), ascending; at most max_shots-1.

    ffprobe's scene filter is tried first, cv2's histogram scan second;
    [] when neither tool exists, the clip is unreadable, or the clip
    really is one shot. When more boundaries are found than fit, the
    strongest-scoring cuts survive and the list stays time-ordered —
    so a clip split by the result never exceeds max_shots shots.
    """
    try:
        max_shots = int(max_shots)
    except (TypeError, ValueError):
        return []
    if max_shots <= 1 or not os.path.isfile(video_path):
        return []
    hits = _boundaries_ffprobe(video_path, _SCENE_THRESH)
    if hits is None:
        hits = _boundaries_cv2(video_path)
    if not hits:
        return []
    hits = _merge_close(hits)
    return sorted(round(t, 3) for t, _ in _strongest(hits, max_shots - 1))


def _split_counts(n: int, lengths: list[float]) -> list[int]:
    """Distribute `n` samples across shots ∝ shot length, >=1 each.

    Largest-remainder rounding: every shot gets its guaranteed sample,
    the leftovers go to the longest shots' fractional shares — a 10 s
    shot deserves more of the budget than a 0.9 s cutaway.
    """
    m = len(lengths)
    counts = [1] * m
    rem = n - m
    if rem <= 0:
        return counts
    total = sum(lengths) or 1.0
    raw = [rem * ln / total for ln in lengths]
    extra = [int(r) for r in raw]
    leftover = rem - sum(extra)
    order = sorted(
        range(m), key=lambda i: (raw[i] - extra[i], lengths[i]), reverse=True
    )
    for i in order[:leftover]:
        extra[i] += 1
    return [c + e for c, e in zip(counts, extra)]


def diverse_times(video_path: str, n: int = 6) -> list[float]:
    """One representative time per shot, padded to `n` by even division.

    Detected boundaries split the clip into shots; each contributes its
    midpoint. Fewer shots than n -> shots are subdivided evenly (extra
    samples go to the longest shots). No boundaries at all — a single
    shot, or a clip no detector could read but whose duration is known —
    falls back to n evenly-spaced middle times across the whole clip.

    [] on a missing file, n <= 0, or when no tool can even read a
    duration. Never raises.
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        return []
    if n <= 0 or not os.path.isfile(video_path):
        return []
    bounds = shot_boundaries(video_path, max_shots=n)
    dur = _duration_s(video_path)
    bounds = [b for b in bounds if b > 0.0 and (dur <= 0.0 or b < dur)]
    if dur <= 0.0:
        if not bounds:
            return []  # no time axis at all -> cannot place samples
        # Cuts exist but no container duration: fabricate a tail from the
        # median gap so the last shot still gets a midpoint.
        gaps = [b2 - b1 for b1, b2 in zip(bounds, bounds[1:])]
        tail = statistics.median(gaps) if gaps else max(1.0, bounds[-1] * 0.25)
        dur = bounds[-1] + tail

    edges = [0.0] + bounds + [dur]
    segs = [
        (edges[i], edges[i + 1])
        for i in range(len(edges) - 1)
        if edges[i + 1] > edges[i]
    ]
    if not segs:
        return []
    if len(segs) > n:  # unreachable via the cap, but cheap insurance
        mids = [(a + b) / 2 for a, b in segs]
        if n == 1:
            return [round(mids[len(mids) // 2], 3)]
        step = (len(mids) - 1) / (n - 1)
        return sorted(round(mids[round(i * step)], 3) for i in range(n))

    counts = _split_counts(n, [b - a for a, b in segs])
    times = [
        a + (b - a) * (i + 0.5) / k
        for (a, b), k in zip(segs, counts)
        for i in range(k)
    ]
    return sorted(round(t, 3) for t in times)


def shot_frame_paths(video_path: str, workdir: str, n: int = 6) -> list[str]:
    """Extract one jpeg per diverse_times() sample; [] on any failure.

    One ffmpeg spawn per timestamp (`-ss` before `-i`: seeks to the
    nearest keyframe then decodes forward to the exact position — fast
    AND frame-accurate). Files land in `workdir` as shot_0000.jpg ...,
    a prefix distinct from perception's frame_*.jpg so the two samplers
    can share a workdir without clobbering each other. Seeks past EOF or
    failed decodes are skipped, never raised.
    """
    exe = _find_ffmpeg()
    if not exe:
        return []
    times = diverse_times(video_path, n)
    if not times:
        return []
    try:
        os.makedirs(workdir, exist_ok=True)
    except (OSError, TypeError):
        return []
    paths: list[str] = []
    for i, t in enumerate(times):
        p = os.path.join(workdir, f"shot_{i:04d}.jpg")
        try:
            proc = subprocess.run(
                [
                    exe,
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-ss", f"{t:.3f}",
                    "-i", video_path,
                    "-frames:v", "1",
                    "-q:v", "2",
                    p,
                ],
                capture_output=True,
                text=True,
                timeout=_SEEK_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        try:
            if proc.returncode == 0 and os.path.getsize(p) > 0:
                paths.append(p)
        except OSError:
            continue
    return paths


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("ffprobe:", _find_ffprobe() or "not found")
    print("ffmpeg: ", _find_ffmpeg() or "not found")
    try:
        import cv2  # noqa: F401

        print("cv2:    ", cv2.__version__)
    except ImportError:
        print("cv2:     not found")

    # Missing file -> graceful [] everywhere, even with tools present.
    ghost = os.path.join(tempfile.gettempdir(), "sundial_no_such_clip.mp4")
    assert shot_boundaries(ghost) == []
    assert diverse_times(ghost) == []
    assert shot_frame_paths(ghost, tempfile.gettempdir()) == []
    print("missing file -> all []  OK")

    exe = _find_ffmpeg()
    if not exe:
        print("ffmpeg absent — synthetic 2-shot clip test skipped")
        raise SystemExit(0)

    with tempfile.TemporaryDirectory(prefix="sundial_shots_") as td:
        clip = os.path.join(td, "twoshot.mp4")
        # 3 s red + 3 s blue: one hard colour cut at t=3.0, dead centre.
        proc = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-f", "lavfi",
                "-i", "color=red:size=320x240:rate=10:duration=3",
                "-f", "lavfi",
                "-i", "color=blue:size=320x240:rate=10:duration=3",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
                "-map", "[v]",
                "-c:v", "mpeg4",
                "-q:v", "5",
                clip,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0 or not os.path.isfile(clip):
            print("clip generation failed:", (proc.stderr or "").strip()[-300:])
            raise SystemExit(1)

        bounds = shot_boundaries(clip)
        print("boundaries:", bounds)
        assert any(abs(b - 3.0) < 0.6 for b in bounds), "cut at t=3.0 not found"

        times = diverse_times(clip, 6)
        print("diverse times:", times)
        assert len(times) == 6
        assert any(t < 3.0 for t in times) and any(t > 3.0 for t in times)

        frames = shot_frame_paths(clip, td, 6)
        print("frame paths:", [os.path.basename(p) for p in frames])
        assert frames and all(os.path.isfile(p) for p in frames)
        print("self-test OK")
