# Sundial

**Provenance verification agent for video.** Give it a clip and a claim — *"filmed Tuesday 3pm, London"* — and it checks the sun, the shadows, the weather, and the street signs, then returns a verdict: `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT`, with evidence cards citing frames.

Built at the Real-Time Video Agents Hack — VAST Builders Challenge, London.

## How it works

```
clip → frames (ffmpeg)
     ├─ YOLO11: objects + reference heights
     ├─ Cosmos Reason: shadow tip/base points · sign text · weather & lighting
     ├─ pysolar: invert sun position → possible time window
     ├─ Nominatim/OSM: geocode extracted text → location candidates
     └─ Open-Meteo archive: weather at claimed time/place → contradiction flag
         → W&B LLM adjudicator → verdict + reasoning + evidence report
```

## Stack

- **VAST AI OS + NVIDIA VSS Blueprint** — ingest, forensic captioning, vector search
- **NVIDIA Cosmos Reason** — VLM: shadow localization, sign reading, physics reasoning
- **YOLO11 (Ultralytics)** — detection/tracking, reference object heights
- **W&B Serverless Inference** — adjudicator LLM · **Weave** — agent tracing
- **pysolar** — solar position math · **Open-Meteo** — historical weather · **Nominatim** — geocoding

## Demo

3-clip gauntlet: truthful claim → `SUPPORTED` · wrong-time claim → `CONTRADICTED` (sun elevation gives it away) · wrong-day claim → `CONTRADICTED` (weather records disagree).
