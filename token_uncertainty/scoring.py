from __future__ import annotations

import math
import re
from collections import defaultdict

from token_uncertainty.types import AnalysisResult, SentenceRisk, TokenScore

NON_TERMINAL_SUFFIXES = (
    "U.S.",
    "U.K.",
    "No.",
    "Inc.",
    "Ltd.",
    "Co.",
    "Dr.",
    "Mr.",
    "Ms.",
    "Prof.",
    "Fig.",
    "Ref.",
)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def split_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.start()
        segment = text[start:end].strip()
        if segment.endswith(NON_TERMINAL_SUFFIXES):
            continue
        if segment:
            spans.append((start, end, segment))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        spans.append((start, len(text), tail))
    return spans or [(0, len(text), text)]


def uncertainty_from_stats(
    probability: float | None,
    entropy: float | None,
    margin: float | None,
) -> float:
    if probability is None:
        return 0.0

    probability_risk = 1.0 - clamp(probability)
    entropy_risk = clamp(entropy or 0.0)
    if margin is None:
        margin_risk = 0.0
    elif margin < 0:
        margin_risk = 1.0
    else:
        margin_risk = 1.0 - clamp(margin * 5.0)

    return clamp((0.6 * probability_risk) + (0.25 * entropy_risk) + (0.15 * margin_risk))


def assign_sentence_indexes(tokens: list[TokenScore]) -> tuple[str, list[tuple[int, int, str]]]:
    text = "".join(token.text for token in tokens)
    spans = split_sentence_spans(text)
    sentence_index = 0
    cursor = 0
    for token in tokens:
        token_start = cursor
        while sentence_index < len(spans) - 1 and token_start >= spans[sentence_index][1]:
            sentence_index += 1
        token.sentence_index = sentence_index
        cursor += len(token.text)
    return text, spans


def analyze_scored_tokens(tokens: list[TokenScore]) -> AnalysisResult:
    if not tokens:
        return AnalysisResult(text="", tokens=[], sentences=[])

    text, sentence_spans = assign_sentence_indexes(tokens)

    for token in tokens:
        token.uncertainty = uncertainty_from_stats(token.probability, token.entropy, token.margin)

    grouped: dict[int, list[TokenScore]] = defaultdict(list)
    for token in tokens:
        grouped[token.sentence_index].append(token)

    sentences: list[SentenceRisk] = []
    for index, (_, _, sentence_text) in enumerate(sentence_spans):
        sentence_tokens = grouped.get(index, [])
        if sentence_tokens:
            mean_uncertainty = sum(t.uncertainty for t in sentence_tokens) / len(sentence_tokens)
            max_token_uncertainty = max(t.uncertainty for t in sentence_tokens)
            uncertainty_score = clamp((0.6 * max_token_uncertainty) + (0.4 * mean_uncertainty))
        else:
            mean_uncertainty = 0.0
            max_token_uncertainty = 0.0
            uncertainty_score = 0.0
        sentences.append(
            SentenceRisk(
                index=index + 1,
                text=sentence_text,
                uncertainty_score=uncertainty_score,
                mean_uncertainty=mean_uncertainty,
                max_token_uncertainty=max_token_uncertainty,
            )
        )
    return AnalysisResult(text=text, tokens=tokens, sentences=sentences)


def normalized_entropy(probabilities) -> float:
    entropy = -float((probabilities * (probabilities + 1e-12).log()).sum().item())
    return clamp(entropy / math.log(max(2, probabilities.numel())))
