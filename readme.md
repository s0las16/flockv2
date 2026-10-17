# Sundial

**Provenance verification agent for video.** Give it a clip and a claim — *"filmed Tuesday 3pm, London"* — and it checks the sun, the shadows, the weather, the street signs, and any burned-in timestamp, then returns a verdict: `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT`, with evidence cards citing frames.

Built at the Real-Time Video Agents Hack — VAST Builders Challenge, London.

## How it works

```
clip → frames (ffmpeg)
     ├─ YOLO11: objects + reference heights
     ├─ Cosmos Reason: shadow tip/base points · sign text · weather & lighting
     │                · clock-overlay read (cropped, upscaled strip)
     ├─ pysolar: invert sun position → possible UTC time window
     ├─ Nominatim/OSM: geocode extracted text → location candidates
     ├─ Open-Meteo archive: weather at claimed time/place → contradiction flag
     └─ clock: burned-in timestamp overlay vs claimed time
         → W&B LLM adjudicator → verdict + narrative + HTML evidence report
```

Every stage emits an `Evidence` card (summary, verdict, confidence, frames).
Stages are failure-isolated: a stage that errors contributes an
`INCONCLUSIVE` card instead of crashing. When the evidence is weak the
agent says so — it never forces a verdict.

## Key findings

- **Camera-tilt correction matters.** Elevated webcams foreshorten ground
  distances along the image vertical by `sin(tilt)`; uncorrected, sun
  elevation estimates came out 20°+ too high. Cosmos estimates the tilt per
  frame; we expand the shadow's `dy` before measuring.
- **VLM hallucination is real; design for it.** The small reasoner invents
  sign text and timestamps. Sign text must appear in ≥2 frames before we
  geocode it, and a clock string without a full date can never carry a
  verdict — both mitigations are in `pipeline.py`.
- **Small-overlay text needs a dedicated read.** Cropping the top strip and
  upscaling 2x turns "can't read it" into verbatim transcription
  (`17.10.26 12:30`).

## Results

| Clip | Claim | Verdict | Key evidence |
|------|-------|---------|--------------|
| Corpus dashcam (I-24, real 05:52 UTC) | 15:00 | **CONTRADICTED** | solar window 16:36–18:38 UTC · weather 99% cloud vs clear |
| Corpus dashcam | ~noon local (truth) | **SUPPORTED** | solar 49.5° expected vs ~51° observed · weather consistent |
| Stamped webcam (real 12:30 overlay) | 12:30 (truth) | **SUPPORTED** | solar 25.0° vs 25.7° expected · weather consistent |
| Stamped webcam | 08:00 (lie) | **INSUFFICIENT** | honest — estimate unplaceable |
| Busy square webcam, 3 frames | 12:30 | **INSUFFICIENT** | honest — 25px people, overlapping shadows |

## Known limitations

- Shadow-derived elevation is sensitive to camera tilt, perspective, and
  small object sizes; ~25px people give ±10° noise, so the stage reports
  `INCONCLUSIVE` rather than guess.
- The VLM can fabricate plausible-looking text and timestamps; consensus
  rules bound the damage but can't eliminate it on single-image clips.
- Weather is corroborating evidence — clear skies can't distinguish
  morning from noon on a clear day.

## Stack

- **VAST AI OS + NVIDIA VSS Blueprint** — corpus access, ingest
- **NVIDIA Cosmos Reason** (`cosmos3-nano-reasoner`) — VLM: shadow
  localization, text, conditions, clock overlays
- **YOLO11 (Ultralytics)** — detection, reference object heights
- **W&B Serverless Inference** — adjudicator LLM · **Weave** — tracing
- **pysolar** — solar position math · **Open-Meteo** — historical weather ·
  **Nominatim** — geocoding

## Run

```bash
pip install -r requirements.txt
export COSMOS_BASE_URL=http://<cosmos>/v1
export COSMOS_API_KEY=<bearer>
export COSMOS_MODEL=nvidia/cosmos3-nano-reasoner
export WANDB_API_KEY=<key>

python3 -m sundial.cli \
    --video clip.mp4 \
    --claim-time '2026-10-17T12:30:00' \
    --claim-place 'Husum, Germany' \
    --out verdict.html
```

`--claim-place` accepts a place name (geocoded) or `lat,lon`.
`python3 -m sundial.cli --demo` runs the adjudicator on a synthetic pack
with no dependencies — useful for smoke tests.

## Layout

```
sundial/
  schema.py       Evidence / ClaimVerdict / EvidencePack / Adjudication
  solar.py        sun position + time-window inversion (pysolar)
  geo.py          Nominatim geocoding + geo evidence
  weather.py      Open-Meteo archive + weather comparison
  perception.py   ffmpeg frames, YOLO, Cosmos client, shadow geometry
  prompts.py      strict-JSON VLM prompts
  adjudicate.py   W&B LLM adjudicator (Weave-traced)
  report.py       HTML verdict card
  pipeline.py     stage orchestration
  cli.py          entry point + zero-dep demo
tests/
```
