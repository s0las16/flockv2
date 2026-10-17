"""Still-image inputs — feed a photo or a folder of photos to the pipeline.

pipeline.run() is built around video: it calls
perception.extract_frames(video_path, ...). These helpers normalize the
other common input shapes — a single still, or a directory of stills
(e.g. a webcam-archive jpg dump) — into the same "list of frame paths"
contract:

    frames = stills.frames_for_input(src, workdir)
    if not frames:                       # caller treats `src` as video
        frames = perception.extract_frames(src, workdir, ...)

`loop_to_mp4` goes the other way for tooling that needs a container: it
stitches the stills into a short mp4 with ffmpeg, then the video path
applies.

Stdlib only — the module imports with zero third-party deps; ffmpeg is
located lazily inside loop_to_mp4.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

STILL_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def is_still(path: str) -> bool:
    """True when `path` names a still image (extension check only)."""
    return os.path.splitext(str(path))[1].lower() in STILL_EXTS


def _natural_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically —
    'cam_9.jpg' < 'cam_10.jpg', the way a webcam archive means it."""
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", os.path.basename(name))
    ]


def frames_for_input(path: str, workdir: str) -> list[str]:
    """Normalize `path` into frame paths; [] means "treat as video".

    * a still image -> copied into `workdir`, returns [copy]
    * a directory   -> image files inside, sorted (natural order)
    * anything else -> [] — caller should run extract_frames on it
    """
    if os.path.isfile(path) and is_still(path):
        os.makedirs(workdir, exist_ok=True)
        dest = os.path.join(workdir, os.path.basename(path))
        # Guard: the source may already live in workdir — copy2 would
        # raise SameFileError.
        if os.path.abspath(dest) != os.path.abspath(path):
            shutil.copy2(path, dest)
        return [dest]
    if os.path.isdir(path):
        names = [
            n for n in os.listdir(path)
            if is_still(n) and os.path.isfile(os.path.join(path, n))
        ]
        names.sort(key=_natural_key)
        return [os.path.join(path, n) for n in names]
    return []


# --------------------------------------------------------------------------
# ffmpeg concat -> mp4
# --------------------------------------------------------------------------

def _concat_escape(path: str) -> str:
    """Quote a filename for an ffmpeg concat script (' -> '\\'')."""
    return path.replace("'", "'\\''")


def loop_to_mp4(image_paths: list[str], out_mp4: str, seconds_per: float = 1) -> str:
    """Stitch stills into an mp4 via ffmpeg's concat demuxer.

    Each image is held for `seconds_per` seconds; returns `out_mp4`.
    The trailing repeat of the last file is the documented demuxer quirk:
    a `duration` directive applies to the entry *above* it, so the final
    image's hold only lands when it appears twice.

    Raises RuntimeError with install guidance when ffmpeg is missing.
    """
    if not image_paths:
        raise ValueError("loop_to_mp4: no input images")
    if float(seconds_per) <= 0:
        raise ValueError(f"loop_to_mp4: seconds_per must be > 0, got {seconds_per!r}")
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found on PATH — needed to build an mp4 from stills. "
            "Install it (`brew install ffmpeg` / `apt install ffmpeg`), or "
            "skip this step and pass the frame list straight to the pipeline."
        )
    images = [os.path.abspath(p) for p in image_paths]

    out_mp4 = os.path.abspath(out_mp4)
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)

    list_path = out_mp4 + ".concat.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("ffconcat version 1.0\n")
            for p in images:
                f.write(f"file '{_concat_escape(p)}'\n")
                f.write(f"duration {float(seconds_per):g}\n")
            f.write(f"file '{_concat_escape(images[-1])}'\n")
        proc = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f", "concat",
                "-safe", "0",          # abs paths live outside the list file's dir
                "-i", list_path,
                # yuv420p/libx264 need even dims — halve odd edges via trunc.
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=25,format=yuv420p",
                "-c:v", "libx264",
                out_mp4,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError(
            f"ffmpeg concat failed on {len(images)} stills: "
            f"{' | '.join(tail) or 'unknown error'}"
        )
    return out_mp4


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="Sundial stills demo")
    ap.add_argument("input", help="still image or directory of stills")
    ap.add_argument("--out", default=None,
                    help="copy-target dir for a single still (default: tempdir)")
    args = ap.parse_args()

    workdir = args.out or tempfile.mkdtemp(prefix="sundial_stills_")
    frames = frames_for_input(args.input, workdir)
    if frames:
        print(f"{len(frames)} frame(s) from {args.input!r}:")
        for f in frames:
            print(" ", f)
    else:
        print(f"{args.input!r}: not a still or a dir of stills — treat as video")
