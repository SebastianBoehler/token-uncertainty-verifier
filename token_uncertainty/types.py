from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TokenScore:
    text: str
    token_id: int | None = None
    probability: float | None = None
    entropy: float | None = None
    rank: int | None = None
    margin: float | None = None
    sentence_index: int = 0
    uncertainty: float = 0.0


@dataclass(slots=True)
class SentenceRisk:
    index: int
    text: str
    uncertainty_score: float
    mean_uncertainty: float
    max_token_uncertainty: float


@dataclass(slots=True)
class AnalysisResult:
    text: str
    tokens: list[TokenScore]
    sentences: list[SentenceRisk]
