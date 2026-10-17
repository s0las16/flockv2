# Demo recording script — Sundial

Target length: ~90 seconds. Record in one take if possible; jump cuts are
fine between runs (each run takes 2–4 min — cut the wait, not the output).

## Shot list

### 1. Cold open (~10s) — terminal only

> "Sundial verifies when and where a video was actually filmed. Give it a
> clip and a claim; it checks the sun, weather archives, signs, and the
> camera's own clock."

Show the claim:

```bash
python3 -m sundial.cli \
    --video /tmp/husum_stamped.mp4 \
    --claim-time '2026-10-17T12:30:00' \
    --claim-place 'Husum, Germany' \
    --out /tmp/verdict.html
```

### 2. The run (~20s, cut the waiting)

Screen-share the terminal as evidence scrolls by: shadow detections,
texture read, geo, clock, weather, solar. One line of narration:

> "Each stage produces an evidence card — the agent never guesses."

### 3. The verdict (~15s) — open the HTML card

`v_stamped_true.html` — SUPPORTED 82%. Highlight the solar line:

> "Sun elevation 25.0°, expected 25.7° — the clip is real."

### 4. The catch (~20s) — the lie

`v_lie2.html` — CONTRADICTED 75% on the corpus clip claimed at 3pm:

> "Claim 3pm on a clip shot at 5:52 — solar puts the sun at a window that
> ends before noon, and the weather archive disagrees. Two independent
> signals, one verdict."

### 5. Honesty beat (~10s) — optional but strong

`v_husum_hu.html` — INSUFFICIENT:

> "And when the evidence isn't there — a busy square, tiny shadows — it
> says INSUFFICIENT instead of making something up."

### 6. Stack card (~10s)

README's architecture diagram, or narrate over a trace in Weave:
`wandb.ai/vastdata/sundial/weave`.

## Files (already on this machine)

| File | Verdict | Use |
|------|---------|-----|
| `v_stamped_true.html` | SUPPORTED | truth beat |
| `v_lie2.html` | CONTRADICTED | catch beat |
| `v_husum_hu.html` | INSUFFICIENT | honesty beat |
| `verdict_true2.html` | SUPPORTED | backup |

## Live run checklist (if recording a fresh run on the VM)

```bash
ssh bc-reddux
cd ~/flockv2 && source ~/sundial.env
python3 -m sundial.cli --video /tmp/husum_stamped.mp4 \
    --claim-time '2026-10-17T12:30:00' --claim-place 'Husum, Germany' \
    --out /tmp/verdict.html
```

- Runs take 2–4 min (YOLO + 4 VLM calls per frame + adjudicator).
- If a live run is risky at presentation time, play the recordings of the
  HTML cards — they are the pipeline's real output.
