"""Audio ambience probe — an untapped time-of-day signal.

The soundtrack of a clip carries crude but free daylight cues:
  * traffic hum     — sustained low-frequency energy (<~300 Hz)
  * birdsong        — chirpy energy in the 2-6 kHz band (dawn/day chorus)
  * silence         — long quiet stretches lean toward nighttime exteriors

Pipeline: ffprobe (stream check) -> ffmpeg (mono 16 kHz PCM wav to a temp
file) -> pure-stdlib analysis (wave/struct/math, including a hand-rolled
radix-2 FFT — no numpy). Both binaries are located lazily with
shutil.which, so this module imports with zero third-party deps and
degrades to None/False when they are absent.

All outputs are heuristic scores, never verdicts: ambience alone cannot
support or contradict a claim, so audio_evidence() always reports
ClaimVerdict.INCONCLUSIVE and leaves fusion to the adjudicator.
"""
from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave

# Decode target: mono, 16 kHz, s16le — small enough for pure-Python FFTs.
_SAMPLE_RATE = 16000
_MAX_DURATION_S = 90.0  # ambience is stationary; cap decode + FFT work

# Window: 2048 samples ≈ 128 ms, ~7.8 Hz/bin — resolves a 50 Hz hum.
_FFT_SIZE = 2048
_MAX_FFT_WINDOWS = 160  # stride-subsample long clips; pure Python is slow

# A window counts as silent below ~-34 dBFS RMS.
_SILENCE_RMS = 0.02

# Feature bands (Hz).
_LOWBAND_HZ = 300.0            # traffic hum / rumble lives under this
_HIGHBAND_HZ = (2000.0, 6000.0)  # birdsong-ish chirp band


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


def has_audio(video_path: str) -> bool:
    """True iff ffprobe reports at least one audio stream on the clip.

    False on any failure: ffprobe missing, unreadable file, or no audio
    stream. Callers that only have ffmpeg should just attempt the decode —
    ffmpeg fails cleanly on silent inputs too.
    """
    exe = _find_ffprobe()
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [
                exe,
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


# --------------------------------------------------------------------------
# Decode: video -> mono 16 kHz PCM wav -> float samples in [-1, 1]
# --------------------------------------------------------------------------

def _decode_to_wav(video_path: str, wav_path: str, max_duration_s: float) -> bool:
    """Run ffmpeg to dump the first `max_duration_s` of audio as pcm_s16le."""
    exe = _find_ffmpeg()
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", video_path,
                "-t", f"{max_duration_s:.3f}",
                "-vn",
                "-ac", "1",
                "-ar", str(_SAMPLE_RATE),
                "-acodec", "pcm_s16le",
                "-f", "wav",
                wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    try:
        # a valid empty-ish wav is 44 header bytes; demand real payload
        return proc.returncode == 0 and os.path.getsize(wav_path) > 100
    except OSError:
        return False


def _read_mono16(wav_path: str) -> list[float]:
    """Read a pcm_s16le wav into floats scaled to [-1, 1). [] on trouble."""
    try:
        with wave.open(wav_path, "rb") as wf:
            if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                return []
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, OSError):
        return []
    if not raw:
        return []
    ints = struct.unpack(f"<{len(raw) // 2}h", raw[: len(raw) // 2 * 2])
    return [s / 32768.0 for s in ints]


# --------------------------------------------------------------------------
# Hand-rolled radix-2 FFT (no numpy)
# --------------------------------------------------------------------------

def _fft(x: list[complex]) -> list[complex]:
    """Iterative radix-2 decimation-in-time FFT (forward transform).

    len(x) must be a power of two. Input is not mutated; returns a fresh
    list of complex coefficients, X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N).
    """
    n = len(x)
    if n == 0 or n & (n - 1):
        raise ValueError(f"FFT length must be a power of two, got {n}")
    a = list(x)

    # bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # butterflies, smallest transforms first
    size = 2
    while size <= n:
        half = size >> 1
        ang = -2.0 * math.pi / size
        w_len = complex(math.cos(ang), math.sin(ang))
        for i in range(0, n, size):
            w = 1.0 + 0.0j
            for k in range(i, i + half):
                u = a[k]
                v = a[k + half] * w
                a[k] = u + v
                a[k + half] = u - v
                w *= w_len
        size <<= 1
    return a


def _hann(n: int) -> list[float]:
    """Hann window of length n (periodic-ish form is fine for analysis)."""
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)) for i in range(n)]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# --------------------------------------------------------------------------
# Spectral analysis
# --------------------------------------------------------------------------

def _analyse(samples: list[float], sample_rate: int) -> dict:
    """Window the samples, then score ambience from the aggregate spectrum.

    Every window contributes to RMS/silence stats; FFTs run on a strided
    subset (<= _MAX_FFT_WINDOWS) because a pure-Python 2048-point transform
    costs ~15-30 ms. Band shares are computed from the summed power
    spectrum, so loud steady sources dominate — what we want for ambience.
    """
    n_samples = len(samples)
    win = _FFT_SIZE
    if n_samples < win:  # zero-pad a sub-window clip rather than special-casing
        samples = samples + [0.0] * (win - n_samples)
    nwin = len(samples) // win
    hann = _hann(win)
    bin_hz = sample_rate / win
    stride = max(1, math.ceil(nwin / _MAX_FFT_WINDOWS))

    n_bins = win // 2 + 1  # 0 Hz .. Nyquist inclusive
    power = [0.0] * n_bins
    analyzed = 0
    silent = 0
    sum_sq_all = 0.0

    for w in range(nwin):
        off = w * win
        chunk = samples[off : off + win]
        win_sq = sum(v * v for v in chunk)
        sum_sq_all += win_sq
        if math.sqrt(win_sq / win) < _SILENCE_RMS:
            silent += 1
        if w % stride:
            continue
        frame = [complex(chunk[i] * hann[i], 0.0) for i in range(win)]
        spec = _fft(frame)
        for i in range(n_bins):
            c = spec[i]
            power[i] += c.real * c.real + c.imag * c.imag
        analyzed += 1

    n_total = nwin * win
    rms_energy = math.sqrt(sum_sq_all / n_total)
    rms_dbfs = 20.0 * math.log10(rms_energy) if rms_energy > 1e-9 else -120.0
    silence_ratio = silent / nwin

    power[0] = 0.0  # drop DC: mic/codec offset is not ambience
    total_power = math.fsum(power)
    if total_power > 0.0:
        centroid = math.fsum(i * bin_hz * p for i, p in enumerate(power)) / total_power
        low_hi = min(math.ceil(_LOWBAND_HZ / bin_hz), n_bins)
        high_lo = min(math.ceil(_HIGHBAND_HZ[0] / bin_hz), n_bins)
        high_hi = min(math.floor(_HIGHBAND_HZ[1] / bin_hz) + 1, n_bins)
        low_share = math.fsum(power[1:low_hi]) / total_power
        high_share = math.fsum(power[high_lo:high_hi]) / total_power
    else:
        centroid = 0.0
        low_share = high_share = 0.0

    # --- heuristic scores (0..1), deliberately rough ---------------------
    # Real-world audio is 1/f-ish, so treat ~30% lowband share as baseline
    # and call >=80% a hum-dominated (traffic-ish) track.
    traffic_score = _clamp01((low_share - 0.30) / 0.50)
    # 2-6 kHz rarely exceeds ~12% of total power unless something chirpy
    # lives there; speech/sibilance can false-positive — hence heuristics.
    birdsong_score = _clamp01((high_share - 0.02) / 0.10)

    # Day/night hint: birdsong is diurnal (dawn chorus); long quiet with a
    # live-but-low track leans night. A dead-silent track (rms < -55 dBFS)
    # is more likely muted/encoded silence than a quiet field — stay unknown.
    if birdsong_score >= 0.5:
        hint = "day"
    elif silence_ratio >= 0.55 and traffic_score <= 0.4 and rms_dbfs > -55.0:
        hint = "night"
    else:
        hint = "unknown"

    return {
        # headline scores (the contract this stage is consumed by)
        "traffic_score": round(traffic_score, 4),
        "birdsong_score": round(birdsong_score, 4),
        "silence_ratio": round(silence_ratio, 4),
        "spectral_centroid_hz": round(centroid, 1),
        "day_night_hint": hint,
        # underlying features, for the adjudicator / debugging
        "rms_energy": round(rms_energy, 5),
        "rms_dbfs": round(rms_dbfs, 1),
        "lowband_share": round(low_share, 4),
        "highband_share": round(high_share, 4),
        "duration_s": round(n_samples / sample_rate, 3),
        "sample_rate": sample_rate,
        "n_windows": nwin,
        "analyzed_windows": analyzed,
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def probe_ambience(video_path: str, max_duration_s: float = _MAX_DURATION_S) -> dict | None:
    """Decode the clip's audio and return ambience features, or None.

    None when there is no usable audio (ffprobe says no stream) or no way
    to decode it (ffmpeg missing, decode fails, empty payload). Never
    raises — a missing soundtrack is a normal case, not an error.
    """
    # fast bail only when ffprobe can actually testify; without it we let
    # ffmpeg try anyway — absence of ffprobe must not mask a real track.
    if _find_ffprobe() is not None and not has_audio(video_path):
        return None
    with tempfile.TemporaryDirectory(prefix="sundial_audio_") as td:
        wav_path = os.path.join(td, "mono16k.wav")
        if not _decode_to_wav(video_path, wav_path, max_duration_s):
            return None
        samples = _read_mono16(wav_path)
    if not samples:
        return None
    return _analyse(samples, _SAMPLE_RATE)


def audio_evidence(video_path: str) -> "Evidence":
    """Package probe_ambience() output as schema.Evidence.

    Verdict is ALWAYS INCONCLUSIVE: ambience features are a soft prior on
    time of day, not proof — the adjudicator combines them with solar,
    weather and visual evidence before anything supports or contradicts.
    """
    from .schema import ClaimVerdict, Evidence  # lazy: keeps module dep-free

    feats = probe_ambience(video_path)
    if feats is None:
        return Evidence(
            type="audio",
            summary="no usable audio track (or ffmpeg/ffprobe unavailable)",
            verdict=ClaimVerdict.INCONCLUSIVE,
            confidence=0.0,
            data={"has_audio": False, "day_night_hint": "unknown"},
        )

    hint = feats["day_night_hint"]
    summary = (
        f"ambience: centroid {feats['spectral_centroid_hz']:.0f} Hz, "
        f"silence {feats['silence_ratio'] * 100:.0f}%, "
        f"lowband {feats['lowband_share'] * 100:.0f}%, "
        f"highband {feats['highband_share'] * 100:.0f}% "
        f"-> day/night hint '{hint}'"
    )
    confidence = {"day": 0.4, "night": 0.35}.get(hint, 0.2)
    return Evidence(
        type="audio",
        summary=summary,
        verdict=ClaimVerdict.INCONCLUSIVE,
        confidence=confidence,
        data={**feats, "has_audio": True},
    )


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Sundial audio-ambience probe demo")
    ap.add_argument("video", help="path to input video clip")
    args = ap.parse_args()

    print("ffprobe:", _find_ffprobe() or "not found")
    print("ffmpeg:", _find_ffmpeg() or "not found")
    print("has_audio:", has_audio(args.video))

    ev = audio_evidence(args.video)
    print("summary:", ev.summary)
    print(json.dumps(ev.data, indent=2))
