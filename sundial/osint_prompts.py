"""OSINT/geolocation prompt templates for the Cosmos Reason VLM calls.

These prompts complement prompts.py: instead of clocks, shadows and weather,
they extract geo-locating FEATURES of a scene (architecture, signage
conventions, road furniture, vegetation) and an honest region guess.

All prompts demand STRICT JSON so `_json()` can parse responses robustly.
The rule throughout: report "unknown"/null rather than guess — an empty or
unknown field is a good answer, an invented one is a failure.
"""

OSINT_SCENE_PROMPT = """\
You are a forensic geolocation analyst performing OSINT feature extraction.
Do NOT guess where this image was taken. Instead, describe only the
geo-locating FEATURES physically visible in the scene — the raw clues a
human investigator would catalogue before forming a hypothesis.

Return STRICT JSON with this exact schema — no prose, no markdown fences:

{"architecture": "<building style/materials, e.g. 'red brick victorian terraces', or null without quotes>", "vegetation": ["<trees/plants/crops visible>"], "terrain": "<flat|hilly|mountainous|coastal|desert|urban|unknown>", "road_furniture": ["<bollards, guardrails, overhead gantries, tram lines, median style>"], "utility_poles": "<wood|concrete|metal|none|unknown>", "lane_markings_color": "<white|yellow|mixed|none|unknown>", "climate_cues": ["<snow, palm trees, deciduous bare, tropical undergrowth etc>"], "landmark_candidates": ["<proper names literally READ as written text in the frame>"], "recognized_landmark": {"name": "<world-famous place you recognise on sight, e.g. 'buckingham palace', 'golden gate bridge'>", "how_sure": "<certain|likely|maybe>"}, "sign_style": {"shape": "<rectangular|triangular|circular|backed|gantry|unknown>", "colors": "<one string e.g. 'white on blue'>", "convention_guess": "<vienna-style|mutcd-style|uk-style|unknown>"}, "vehicles": [{"class": "<type>", "style_note": "<e.g. 'european hatchback', 'yellow school bus'>"}]}

- "architecture": dominant building style and materials (e.g. "red brick
  victorian terraces", "concrete panel apartment blocks", "adobe single
  storey"); null if no buildings are visible.
- "vegetation": trees, plants, crops actually visible (e.g. "palm trees",
  "birch", "vineyards"); [] if none.
- "terrain": pick the closest of flat|hilly|mountainous|coastal|desert|
  urban; "unknown" if the ground/horizon is not informative.
- "road_furniture": visible items such as bollards, guardrails, overhead
  gantries, tram lines, median style; [] if no road is visible.
- "utility_poles": "wood", "concrete", "metal", "none" if absent, "unknown"
  if unclear.
- "lane_markings_color": dominant painted line colour: "white", "yellow",
  "mixed", "none" if no markings are visible, "unknown" if unclear.
- "climate_cues": anything implying climate zone — snow, palm trees, bare
  deciduous trees, tropical undergrowth, dust; [] if nothing informative.
- "landmark_candidates": ONLY names you can literally READ as written text
  in the frame (a signed building, a named bridge). Never include a place
  you merely recognise visually — a tower resembling the Eiffel Tower is
  NOT a landmark unless its name is readable. NEVER invent plausible-
  sounding landmarks; [] if none.
- "recognized_landmark": a DIFFERENT thing — a world-famous place you can
  identify on sight without reading anything (Buckingham Palace, the
  Sydney Opera House, a specific named bridge/cathedral/square). Report
  only globally-recognisable places a stranger could name; "how_sure" is
  "certain" only for unmistakable icons, "likely" for a strong match,
  "maybe" for a plausible look. Use null when nothing recognisable is
  shown — never name a generic palace, bridge, or tower just to fill it.
- "sign_style": describe signage physically present. If no signs are
  visible return {"shape":"unknown","colors":"unknown",
  "convention_guess":"unknown"} — never a bare string or null.
  "shape" is the dominant sign shape; "colors" is ONE string e.g.
  "white on blue"; "convention_guess" names the sign convention the
  design resembles or "unknown".
- "vehicles": each clearly visible vehicle as an OBJECT
  {"class": ..., "style_note": ...} — never bare strings; [] if none.

Keep every value to a few words, not sentences. Report "unknown", "none",
null, or [] whenever a feature is absent or generic — never fill a field
by guessing.

Respond with JSON only."""

GEOGUESS_PROMPT = """\
You are a forensic geolocation analyst. Give an honest region/country guess
for this image.

Return STRICT JSON with this exact schema — no prose, no markdown fences:

{"country_guess": "<common English country name, e.g. 'Germany' or 'United States' — never a code or abbreviation — or 'unknown'>", "region_guess": "<state/province/city or 'unknown'>", "confidence": <bare number 0.0-1.0, not a word>, "reasoning": "<one sentence, max 25 words, citing VISIBLE evidence>"}

- Base the guess ONLY on visible cues: readable signs, licence plates, lane
  markings and driving side, sign conventions, architecture, vegetation,
  vehicles, utility poles, terrain. Do NOT use prior knowledge of how the
  image was obtained and do NOT invent a plausible-sounding place.
- Say "unknown" for country_guess/region_guess when the visible cues are
  generic and could appear in many countries. An honest "unknown" is far
  better than a wrong confident guess.
- "confidence" must reflect cue SPECIFICITY, not how neat the scene looks:
  use <0.3 for generic scenes (suburban roads, fields, malls with no
  readable distinctive text), 0.3-0.6 when several weak cues agree, and
  >0.6 only when specific evidence exists (readable place-name sign,
  distinctive flag/plate, famous landmark).
- "reasoning": exactly one sentence (max 25 words) naming the concrete
  VISIBLE evidence behind the guess, or explaining the cues are generic.

Respond with JSON only."""


if __name__ == "__main__":
    # Syntax check: print both prompts so they can be eyeballed / piped.
    print("=" * 72)
    print("OSINT_SCENE_PROMPT")
    print("=" * 72)
    print(OSINT_SCENE_PROMPT)
    print()
    print("=" * 72)
    print("GEOGUESS_PROMPT")
    print("=" * 72)
    print(GEOGUESS_PROMPT)
