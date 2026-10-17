"""Prompt templates for the Cosmos Reason VLM calls in perception.py.

All prompts demand STRICT JSON so `_json()` can parse responses robustly.
Pixel coordinate convention stated to the model: origin top-left, x grows
right, y grows down — matching the convention used by shadow_vector().
"""

SHADOW_PROMPT = """\
You are a forensic vision analyst. Examine this image and locate the single
most prominent VERTICAL object that casts a clearly visible shadow on the
ground (a person, pole, signpost, tree, vehicle, etc.).

Return STRICT JSON with this exact schema — no prose, no markdown fences:

{"object_class": "<object type>", "base_point": [x, y], "shadow_tip": [x, y], "object_top": [x, y], "camera_tilt_deg": <number>, "confidence": <0.0-1.0>}

- "object_top": pixel coordinates of the object's TOP (head, roofline, tip).
- "camera_tilt_deg": estimated camera downward tilt in degrees — 0 means the
  camera looks horizontally (horizon high in frame), 90 means straight down.
  Estimate from the horizon position and ground perspective; typical values
  are 0-50.

- "base_point": pixel coordinates of where the object meets the ground.
- "shadow_tip": pixel coordinates of the furthest visible tip of its cast shadow.
- Coordinates are in pixels: origin is top-left, x grows right, y grows down.
- "confidence" is your certainty that the shadow truly belongs to that object.
- If no usable object+shadow pair is visible, respond with
  {"object_class": null, "base_point": null, "shadow_tip": null, "confidence": 0.0}.

Respond with JSON only."""

TEXT_PROMPT = """\
You are a forensic geolocation analyst. Read ALL readable text visible in this
image: street signs, shop and business names, licence plates, advertisements,
billboards, transit signs. Also assess the driving environment.

Return STRICT JSON with this exact schema — no prose, no markdown fences:

{"texts": ["<verbatim string>", ...], "overlay_timestamps": ["<verbatim burned-in date/time text>", ...], "driving_side": "left"|"right"|"unknown", "sign_language": "<primary signage language or 'unknown'>", "road_markings": ["<short description>", ...]}

- "overlay_timestamps": any date/time burned into the image by the camera
  (corners/edges, e.g. "17.10.2026 08:00:02", "10/17 08:00"). Verbatim —
  include the full string as printed. [] if none.

- "texts": ONLY strings you can literally read character-by-character in the
  image, transcribed verbatim. Do NOT infer, translate, or guess place names.
  If a sign is blurry, angled, or too small to read with certainty, omit it.
  An empty list is a perfectly good answer — never invent text.
- "driving_side": infer from lane position, markings and vehicle orientation;
  use "unknown" if there is no road or it is ambiguous.
- "road_markings": e.g. "zebra crossing", "double yellow line", "bike lane",
  "stop line" — include colours where visible.

Respond with JSON only."""

CONDITIONS_PROMPT = """\
You are a forensic weather and lighting analyst. Describe the ambient
conditions visible in this image.

Return STRICT JSON with this exact schema — no prose, no markdown fences:

{"sky": "clear"|"partly"|"overcast", "precipitation": "none"|"rain"|"snow", "ground_wetness": "dry"|"damp"|"wet"|"unknown", "lighting": "bright daylight"|"golden hour"|"overcast"|"dusk"|"night", "streetlights": "on"|"off"|"unknown", "sun": {"visible": true|false, "position": [x, y]|null, "description": "<e.g. low in the frame, near the left edge>"}}

- "ground_wetness": wet only if the surface clearly reflects or looks soaked.
- "sun.position": pixel coordinates [x, y] (origin top-left, x right, y down)
  ONLY if the sun disc itself is visible in frame; otherwise null.
- "sun.description": where the sun appears to be, judging by light direction,
  even when the disc is not visible.

Respond with JSON only."""

CLOCK_PROMPT = """\
You are given a cropped strip from the top of a camera frame.
Read ONLY the camera's burned-in date/time overlay text, character for
character. If the strip contains a timestamp (e.g. "17.10.26 12:30",
"2026-10-17 08:00:02", "Sat 17 Oct 11:31"), transcribe it verbatim —
do not normalize, complete, or guess missing digits. If there is no
date/time overlay in the strip, return an empty list.

Return exactly one JSON object and nothing else — no prose, no markdown
fences:

{"overlay_timestamps": ["<verbatim timestamp string>", ...]}
"""
