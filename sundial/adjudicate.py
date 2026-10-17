"""LLM adjudication over an EvidencePack.

Primary path: W&B serverless inference (OpenAI-compatible chat API).
Fallback: deterministic rule-based adjudicator so a verdict is always
produced even with no LLM reachable. Optional Weave tracing when
WEAVE_PROJECT is set and `weave` is importable — all failures swallowed.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from .schema import Adjudication, ClaimVerdict, EvidencePack, Verdict

DEFAULT_BASE_URL = "https://api.inference.wandb.ai/v1"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

#: confidence >= this on a CONTRADICTS item flips the whole verdict.
_HIGH_CONF = 0.6

SYSTEM_PROMPT = (
    "You are Sundial's adjudicator — a forensic video-provenance reasoner. "
    "You receive a claim about when/where a clip was filmed plus evidence items "
    "from analysis stages (solar geometry, weather archive, geocoding, shadow and "
    "scene-text observation). Decide whether the evidence supports the claim.\n"
    "Reply with STRICT JSON ONLY — no prose, no markdown fences:\n"
    '{"verdict": "SUPPORTED"|"CONTRADICTED"|"INSUFFICIENT", '
    '"confidence": <float 0-1>, '
    '"narrative": "<2-4 sentence explanation citing evidence types>"}\n'
    "Use CONTRADICTED only when evidence clearly refutes the claim; use "
    "INSUFFICIENT when evidence is missing, inconclusive, or mixed."
)


# --------------------------------------------------------------------------- #
# W&B inference client
# --------------------------------------------------------------------------- #
class WandBClient:
    """OpenAI-compatible client for W&B serverless inference."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> None:
        self.api_key = api_key or os.environ.get("WANDB_API_KEY")
        if not self.api_key:
            raise ValueError("WANDB_API_KEY not set")
        self.base_url = base_url or os.environ.get("WANDB_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("WANDB_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._openai = None  # lazy

    def _client(self):
        if self._openai is None:
            from openai import OpenAI  # lazy heavy import

            self._openai = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._openai

    def complete(self, system: str, user: str) -> str:
        resp = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Pack serialization (data payloads summarized so the prompt stays small)
# --------------------------------------------------------------------------- #
def _shrink(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 300 else value[:300] + "…"
    if depth >= 3:
        return str(value)[:200]
    if isinstance(value, dict):
        return {str(k)[:40]: _shrink(v, depth + 1) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple, set)):
        return [_shrink(v, depth + 1) for v in list(value)[:10]]
    return value


def _pack_payload(pack: EvidencePack) -> dict:
    return {
        "clip": pack.clip,
        "claim": {
            "claimed_time": pack.claim.claimed_time,
            "claimed_place": pack.claim.claimed_place,
            "raw": pack.claim.raw,
        },
        "evidence": [
            {
                "type": ev.type,
                "summary": ev.summary,
                "verdict": getattr(ev.verdict, "value", ev.verdict),
                "confidence": ev.confidence,
                "data": _shrink(ev.data),
                "frames": [os.path.basename(str(f)) for f in (ev.frames or [])][:6],
            }
            for ev in pack.evidence
        ],
    }


# --------------------------------------------------------------------------- #
# Robust JSON extraction — first balanced {...} block wins
# --------------------------------------------------------------------------- #
def _first_json_object(text: str) -> dict:
    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start : i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break
    raise ValueError("no JSON object found in LLM output")


def _norm_verdict(value: Any) -> Verdict:
    s = str(value or "").upper()
    if "CONTRADICT" in s:
        return Verdict.CONTRADICTED
    if "SUPPORT" in s:
        return Verdict.SUPPORTED
    return Verdict.INSUFFICIENT


def _clamp_conf(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Deterministic fallback adjudicator — never needs an LLM
# --------------------------------------------------------------------------- #
def _fallback(pack: EvidencePack, note: Optional[str] = None) -> Adjudication:
    ev = list(pack.evidence)
    contra = [e for e in ev if e.verdict == ClaimVerdict.CONTRADICTS]
    supp = [e for e in ev if e.verdict == ClaimVerdict.SUPPORTS]
    high_contra = [e for e in contra if e.confidence >= _HIGH_CONF]

    if high_contra:
        verdict = Verdict.CONTRADICTED
        conf = max(0.6, min(0.95, max(e.confidence for e in high_contra)))
    elif supp and not contra:
        verdict = Verdict.SUPPORTED
        conf = max(0.5, min(0.9, max(e.confidence for e in supp)))
    else:
        verdict = Verdict.INSUFFICIENT
        conf = min(0.5, 0.2 + 0.05 * len(ev))

    kinds = ", ".join(sorted({e.type for e in ev})) or "none"
    narrative = (
        f"{len(supp)} evidence item(s) support, {len(contra)} contradict, "
        f"{len(ev) - len(supp) - len(contra)} inconclusive across types: {kinds}. "
        f"Rule-based verdict: {verdict.value}."
    )
    if note:
        narrative = f"{note} " + narrative
    return Adjudication(verdict=verdict, confidence=conf, narrative=narrative, pack=pack)


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #
def _llm_adjudication(pack: EvidencePack, client: WandBClient) -> Adjudication:
    user = (
        "Claim and evidence (JSON):\n"
        + json.dumps(_pack_payload(pack), indent=2, default=str)
    )
    raw = client.complete(SYSTEM_PROMPT, user)
    obj = _first_json_object(raw)
    return Adjudication(
        verdict=_norm_verdict(obj.get("verdict")),
        confidence=_clamp_conf(obj.get("confidence")),
        narrative=str(obj.get("narrative") or "LLM adjudication returned no narrative."),
        pack=pack,
    )


# --------------------------------------------------------------------------- #
# Weave (optional, lazy, failure-proof)
# --------------------------------------------------------------------------- #
_weave_inited = False


def _init_weave_once() -> None:
    global _weave_inited
    if _weave_inited:
        return
    _weave_inited = True
    project = os.environ.get("WEAVE_PROJECT")
    if not project:
        return
    try:
        import weave

        weave.init(project)
    except Exception:
        pass


def _adjudicate_impl(pack: EvidencePack, client: Optional[WandBClient] = None) -> Adjudication:
    _init_weave_once()
    if client is None:
        try:
            client = WandBClient()
        except Exception:
            client = None
    if client is not None:
        try:
            return _llm_adjudication(pack, client)
        except Exception as e:
            return _fallback(pack, note=f"LLM adjudication failed ({type(e).__name__}: {e}); deterministic fallback used.")
    return _fallback(pack)


#: public entry point — weave.op-wrapped when available
adjudicate = _adjudicate_impl
if os.environ.get("WEAVE_PROJECT"):
    try:
        import weave  # noqa: F401

        adjudicate = weave.op(_adjudicate_impl)  # type: ignore[assignment]
    except Exception:
        adjudicate = _adjudicate_impl
