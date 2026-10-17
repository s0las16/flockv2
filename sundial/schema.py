"""Shared contract for Sundial — every stage reads/writes these types.

Pipeline: clip -> perception/geo/weather/solar stages -> EvidencePack
         -> adjudicator -> Adjudication (verdict + narrative).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class ClaimVerdict(str, Enum):
    """How a single piece of evidence relates to the claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"


@dataclass
class Claim:
    """What the submitter asserts about the clip."""

    claimed_time: Optional[str] = None  # ISO 8601 datetime, local to the claimed place
    claimed_place: Optional[str] = None  # free text ("London") or "lat,lon"
    raw: str = ""  # original natural-language claim


@dataclass
class Evidence:
    """One piece of forensic evidence produced by a stage."""

    type: str  # "solar" | "weather" | "geo" | "texture" | "shadows" | "clock"
               # | "scene" | "language" | "plate_style" | "lunar" | "audio"
               # | "azimuth" | stage-note names like "vlm" / "geo-resolve"
    summary: str  # one-sentence human-readable finding
    verdict: ClaimVerdict
    confidence: float = 0.0  # 0..1
    data: dict = field(default_factory=dict)  # stage-specific payload
    frames: list = field(default_factory=list)  # cited frame paths / timestamps


@dataclass
class EvidencePack:
    clip: str
    claim: Claim
    evidence: list = field(default_factory=list)

    def add(self, ev: Evidence) -> None:
        self.evidence.append(ev)


@dataclass
class Adjudication:
    verdict: Verdict
    confidence: float  # 0..1
    narrative: str
    pack: EvidencePack

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeoCandidate:
    name: str
    lat: float
    lon: float
    country_code: str = ""
    display_name: str = ""
    score: float = 0.0
