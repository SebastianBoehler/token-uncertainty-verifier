from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TokenScore:
    text: str
    token_id: int | None = None
    probability: float | None = None
    entropy: float | None = None
    rank: int | None = None
    margin: float | None = None
    sentence_index: int = 0
    claim_cues: tuple[str, ...] = field(default_factory=tuple)
    uncertainty: float = 0.0
    factual_risk: float = 0.0


@dataclass(slots=True)
class SentenceRisk:
    index: int
    text: str
    factual_risk: float
    mean_uncertainty: float
    max_token_risk: float
    claim_cues: tuple[str, ...]


@dataclass(slots=True)
class AnalysisResult:
    text: str
    tokens: list[TokenScore]
    sentences: list[SentenceRisk]
