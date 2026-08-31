from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


class FusionScorer(Protocol):
    """Minimal scorer contract for a future TDT in-beam integration."""

    name: str

    def score(self, *, hypothesis: str, context: Mapping[str, object]) -> float: ...


@dataclass(frozen=True)
class FusionWeights:
    tdt: float = 1.0
    lm: float = 0.0
    phrase_boost: float = 0.0
    ctc_local: float = 0.0
    phone: float = 0.0
    entity_context: float = 0.0


@dataclass
class CandidateScores:
    text: str
    tdt: float
    lm: float = 0.0
    phrase_boost: float = 0.0
    ctc_local: float = 0.0
    phone: float = 0.0
    entity_context: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def fused(self, weights: FusionWeights, *, entity_gate: bool = True) -> float:
        score = weights.tdt * self.tdt
        score += weights.lm * self.lm
        score += weights.phrase_boost * self.phrase_boost
        if entity_gate:
            score += weights.ctc_local * self.ctc_local
            score += weights.phone * self.phone
            score += weights.entity_context * self.entity_context
        return score


def rerank(candidates: list[CandidateScores], weights: FusionWeights) -> list[CandidateScores]:
    return sorted(candidates, key=lambda item: item.fused(weights), reverse=True)
