"""Perception stage — frame extraction + forensic vision signals.

Pulls three signals out of sampled frames:
  * shadow geometry  (VLM: base/tip pixel points -> sun-direction vector)
  * scene text       (VLM: signs/plates -> geolocation clues downstream)
  * conditions       (VLM: sky/precip/lighting -> weather cross-check)

Everything heavy (cv2, imageio, ultralytics, openai, PIL) is imported lazily
inside functions so this module imports with zero third-party deps.
"""
from __future__ import annotations

import base64
import glob
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess

from .prompts import CONDITIONS_PROMPT, SHADOW_PROMPT, TEXT_PROMPT
from .schema import ClaimVerdict, Evidence

# Approximate real-world heights (m) used to convert a shadow's pixel length
# into a sun-elevation estimate downstream. TODO: calibrate per class; these
# are population averages, not per-instance measurements.
REFERENCE_HEIGHTS_M: dict[str, float] = {
    "person": 1.7,
    "car": 1.5,
    "bus": 3.0,
    "truck": 3.3,
    "bicycle": 1.1,
    "motorcycle": 1.3,
    "traffic light": 3.5,
    "stop sign": 2.5,
}

# Detection classes that make good shadow anchors: known height + vertical.
_GOOD_REFS = frozenset({"person", "car"})
_OTHER_REFS = frozenset(REFERENCE_HEIGHTS_M)


# --------------------------------------------------------------------------
# Frame extraction
# --------------------------------------------------------------------------

def _find_ffmpeg() -> str | None:
    """Locate an ffmpeg executable.

    Falls back to the static binary bundled with `imageio-ffmpeg` — common on
    machines where imageio is installed but system ffmpeg is not.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _extract_frames_ffmpeg(
    exe: str, video_path: str, out_dir: str, fps: float, max_frames: int
) -> list[str]:
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    proc = subprocess.run(
        [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-frames:v",
            str(max_frames),
            pattern,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))
    if proc.returncode != 0 and not frames:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError(
            f"ffmpeg failed on {video_path!r}: {' | '.join(tail) or 'unknown error'}"
        )
    return frames


def _extract_frames_cv2(
    video_path: str, out_dir: str, fps: float, max_frames: int
) -> list[str]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video_path!r}")
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(vid_fps / fps))
    paths: list[str] = []
    idx = 0
    try:
        while len(paths) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                p = os.path.join(out_dir, f"frame_{len(paths):04d}.jpg")
                cv2.imwrite(p, frame)
                paths.append(p)
            idx += 1
    finally:
        cap.release()
    return paths


def _extract_frames_imageio(
    video_path: str, out_dir: str, fps: float, max_frames: int
) -> list[str]:
    import imageio

    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        vid_fps = float(meta.get("fps") or 25.0)
        step = max(1, round(vid_fps / fps))
        paths: list[str] = []
        for idx, frame in enumerate(reader):
            if len(paths) >= max_frames:
                break
            if idx % step:
                continue
            p = os.path.join(out_dir, f"frame_{len(paths):04d}.jpg")
            imageio.imwrite(p, frame)
            paths.append(p)
        return paths
    finally:
        reader.close()


def extract_frames(
    video_path: str, out_dir: str, fps: float = 1.0, max_frames: int = 30
) -> list[str]:
    """Sample up to `max_frames` frames at `fps`; return sorted frame paths.

    Backend order: ffmpeg binary -> cv2 -> imageio. Raises RuntimeError with
    install guidance when no backend is available.
    """
    os.makedirs(out_dir, exist_ok=True)
    exe = _find_ffmpeg()
    if exe:
        return _extract_frames_ffmpeg(exe, video_path, out_dir, fps, max_frames)
    if importlib.util.find_spec("cv2") is not None:
        return _extract_frames_cv2(video_path, out_dir, fps, max_frames)
    if importlib.util.find_spec("imageio") is not None:
        return _extract_frames_imageio(video_path, out_dir, fps, max_frames)
    raise RuntimeError(
        "No frame-extraction backend available. Install ffmpeg "
        "(`brew install ffmpeg`) or `pip install opencv-python imageio`."
    )


# --------------------------------------------------------------------------
# Object detection (lazy ultralytics)
# --------------------------------------------------------------------------

_YOLO_MODEL = None
_YOLO_MODEL_NAME: str | None = None


def detect_objects(frame_path: str, model_name: str = "yolo11n.pt") -> list[dict]:
    """Run YOLO on one frame; return [{class, confidence, bbox_xyxy}].

    The model is cached module-globally so repeated calls are cheap. The
    first call downloads weights if ultralytics can't find them locally.
    """
    if importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError(
            "ultralytics is not installed — run `pip install ultralytics` "
            "(needed for detect_objects / pick_shadow_frames)."
        )
    from ultralytics import YOLO

    global _YOLO_MODEL, _YOLO_MODEL_NAME
    if _YOLO_MODEL is None or _YOLO_MODEL_NAME != model_name:
        _YOLO_MODEL = YOLO(model_name)
        _YOLO_MODEL_NAME = model_name
    result = _YOLO_MODEL.predict(frame_path, verbose=False)[0]
    out = []
    for box in result.boxes:
        out.append(
            {
                "class": result.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            }
        )
    return out


# --------------------------------------------------------------------------
# Cosmos Reason VLM client (OpenAI-compatible)
# --------------------------------------------------------------------------

_MAX_IMG_EDGE = 2048  # keep base64 payloads sane for NIM endpoints


def _data_url(frame_path: str) -> str:
    """Read an image file as a base64 data-URL, downscaling if PIL is around."""
    try:
        import io

        from PIL import Image

        with Image.open(frame_path) as img:
            if max(img.size) > _MAX_IMG_EDGE:
                img.thumbnail((_MAX_IMG_EDGE, _MAX_IMG_EDGE))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        ext = os.path.splitext(frame_path)[1].lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
        with open(frame_path, "rb") as f:
            return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def _first_json_object(text: str) -> dict:
    """Extract the first {...} JSON object from a chatty VLM response.

    Tolerates markdown fences and leading prose: tries the fenced block first,
    then scans the raw text brace-by-brace with raw_decode.
    """
    if not text:
        return {}
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S)]
    candidates.append(text)
    dec = json.JSONDecoder()
    for cand in candidates:
        start = cand.find("{")
        while start != -1:
            try:
                obj, _ = dec.raw_decode(cand[start:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            start = cand.find("{", start + 1)
    return {}


class CosmosClient:
    """Thin OpenAI-compatible client for a Cosmos Reason VLM endpoint.

    Env: COSMOS_BASE_URL (required), COSMOS_API_KEY (default "not-needed" —
    NIMs typically don't check it), COSMOS_MODEL (default nvidia/cosmos-reason2).
    Public methods never raise: they return None/{} on any failure so the
    pipeline degrades gracefully.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url or os.environ.get("COSMOS_BASE_URL")
        if not self.base_url:
            raise RuntimeError(
                "COSMOS_BASE_URL is not set — point it at your Cosmos/NIM "
                "endpoint, e.g. http://localhost:8000/v1"
            )
        self.model = model or os.environ.get("COSMOS_MODEL", "nvidia/cosmos-reason2")
        key = api_key or os.environ.get("COSMOS_API_KEY", "not-needed")
        try:
            from openai import OpenAI  # lazy: keeps module import dep-free
        except ImportError as e:
            raise RuntimeError(
                "openai is not installed — run `pip install openai` "
                "(needed for CosmosClient)."
            ) from e

        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=timeout)

    def _ask(self, frame_path: str, prompt: str) -> str:
        """Send frame+prompt as a chat message; return raw text response."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(frame_path)},
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "text"},
        )
        return resp.choices[0].message.content or ""

    def _json(self, frame_path: str, prompt: str) -> dict:
        return _first_json_object(self._ask(frame_path, prompt))

    def locate_shadow(self, frame_path: str) -> dict | None:
        """->{object_class, base_point, shadow_tip, confidence} or None."""
        try:
            out = self._json(frame_path, SHADOW_PROMPT)
            return out or None
        except Exception:
            return None

    def read_scene_text(self, frame_path: str) -> dict:
        """->{texts, driving_side, sign_language, road_markings} or {}."""
        try:
            return self._json(frame_path, TEXT_PROMPT)
        except Exception:
            return {}

    def describe_conditions(self, frame_path: str) -> dict:
        """->{sky, precipitation, ground_wetness, lighting, streetlights, sun} or {}."""
        try:
            return self._json(frame_path, CONDITIONS_PROMPT)
        except Exception:
            return {}


# --------------------------------------------------------------------------
# Shadow geometry
# --------------------------------------------------------------------------

def shadow_vector(obs: dict) -> dict:
    """Turn a locate_shadow observation into an image-space shadow vector.

    Image coords: origin top-left, +x right, +y DOWN. `angle_deg` is the
    direction the shadow points, measured clockwise from image "up"
    (0 = toward top edge, 90 = right, 180 = down, 270 = left). The solar
    stage maps this onto world azimuth once a camera-heading hypothesis
    exists — we deliberately stay in image space here.
    """
    base, tip = obs.get("base_point"), obs.get("shadow_tip")
    vec = {"dx": 0.0, "dy": 0.0, "length_px": 0.0, "angle_deg": None, "ok": False}
    try:
        dx = float(tip[0]) - float(base[0])
        dy = float(tip[1]) - float(base[1])
    except (TypeError, IndexError, KeyError):
        return vec
    length = math.hypot(dx, dy)
    vec.update(
        dx=dx,
        dy=dy,
        length_px=length,
        angle_deg=(math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0,
        ok=length > 1e-6,
    )
    return vec


def to_evidence(frame: str, obs: dict) -> Evidence:
    """Package a shadow observation as schema.Evidence.

    Verdict stays INCONCLUSIVE — pixel shadow direction is meaningless for
    the claim until the solar stage fuses it with heading + ephemeris.
    """
    vec = shadow_vector(obs)
    obj = obs.get("object_class") or "object"
    if vec["ok"]:
        summary = (
            f"{obj} casts a {vec['length_px']:.0f}px shadow toward "
            f"{vec['angle_deg']:.0f} deg (image space)"
        )
    else:
        summary = f"No usable shadow measurement from {obj}"
    conf = obs.get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.3
    return Evidence(
        type="shadows",
        summary=summary,
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=max(0.0, min(1.0, conf)),
        data={
            "frame": frame,
            "observation": obs,
            "vector": vec,
            "reference_height_m": REFERENCE_HEIGHTS_M.get(str(obj).lower()),
        },
        frames=[frame],
    )


def pick_shadow_frames(
    frames: list[str],
    yolo_fn=detect_objects,
    top_k: int | None = None,
) -> list[str]:
    """Rank frames for shadow analysis, best first.

    People and cars score highest: known reference heights (see
    REFERENCE_HEIGHTS_M), roughly vertical, and they tend to stand on the
    ground plane. Detection failures score 0 and sink to the bottom, so this
    still works (as a no-op ordering) when ultralytics is missing.
    """
    scored = []
    for i, f in enumerate(frames):
        score = 0.0
        try:
            for d in yolo_fn(f):
                cls = d.get("class")
                if cls in _GOOD_REFS:
                    score += 2.0
                elif cls in _OTHER_REFS:
                    score += 1.0
                score += 0.1 * float(d.get("confidence") or 0.0)
        except Exception:
            pass
        scored.append((-score, i, f))
    scored.sort()
    ranked = [f for _, _, f in scored]
    return ranked[:top_k] if top_k else ranked


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="Sundial perception demo")
    ap.add_argument("video", help="path to input video clip")
    ap.add_argument("--fps", type=float, default=1.0, help="sampling rate")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--out", default=None, help="frame output dir (default: tempdir)")
    args = ap.parse_args()

    out_dir = args.out or tempfile.mkdtemp(prefix="sundial_frames_")
    frames = extract_frames(args.video, out_dir, fps=args.fps, max_frames=args.max_frames)
    print(f"extracted {len(frames)} frames -> {out_dir}")

    ranked = pick_shadow_frames(frames)
    for f in ranked[:5]:
        try:
            dets = detect_objects(f)
            print(os.path.basename(f), [(d["class"], d["confidence"]) for d in dets])
        except RuntimeError as e:
            print(f"detections skipped: {e}")
            break

    if os.environ.get("COSMOS_BASE_URL"):
        client = CosmosClient()
        best = ranked[0] if ranked else None
        if best:
            print("frame:", os.path.basename(best))
            print("shadow:", json.dumps(client.locate_shadow(best), indent=2))
            print("text:", json.dumps(client.read_scene_text(best), indent=2))
            print("conditions:", json.dumps(client.describe_conditions(best), indent=2))
    else:
        print("COSMOS_BASE_URL not set — skipping VLM calls")
