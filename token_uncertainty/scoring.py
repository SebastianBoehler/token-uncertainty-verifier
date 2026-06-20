from __future__ import annotations

import math
import re
from collections import defaultdict

from token_uncertainty.types import AnalysisResult, SentenceRisk, TokenScore

LEGAL_TERMS = {
    "act",
    "appeal",
    "article",
    "case",
    "claim",
    "clause",
    "code",
    "contract",
    "court",
    "damages",
    "decision",
    "directive",
    "doctrine",
    "holding",
    "judgment",
    "liability",
    "plaintiff",
    "precedent",
    "regulation",
    "rule",
    "section",
    "statute",
    "treaty",
}

CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("date", re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")),
    ("number", re.compile(r"\b\d+(?:\.\d+)?%?\b")),
    ("citation", re.compile(r"\b\d+\s+[A-Z][A-Za-z.]*\s+\d+\b")),
    ("section", re.compile(r"(?:\bsection\b|\bsec\.\b|[\u00a7])\s*\d+", re.I)),
    ("case-name", re.compile(r"\b[A-Z][A-Za-z]+ v\. [A-Z][A-Za-z]+\b")),
    ("named-entity", re.compile(r"\b[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,})*\b")),
)

NON_TERMINAL_SUFFIXES = (
    " v.",
    " U.S.",
    " U.K.",
    " Art.",
    " No.",
    " Sec.",
    " Inc.",
    " Ltd.",
    " Co.",
    " Dr.",
    " Mr.",
    " Ms.",
    " Prof.",
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


def claim_cues_for_text(text: str) -> tuple[str, ...]:
    cues = {name for name, pattern in CLAIM_PATTERNS if pattern.search(text)}
    words = {word.lower().strip(".,:;()[]") for word in text.split()}
    if words & LEGAL_TERMS:
        cues.add("legal-term")
    return tuple(sorted(cues))


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


def factual_risk_from_cues(uncertainty: float, cues: tuple[str, ...]) -> float:
    cue_weight = clamp(len(cues) / 3.0)
    if not cues:
        return clamp(0.75 * uncertainty)
    return clamp((0.62 * uncertainty) + (0.38 * cue_weight))


def assign_sentence_indexes(tokens: list[TokenScore]) -> str:
    text = "".join(token.text for token in tokens)
    spans = split_sentence_spans(text)
    sentence_index = 0
    cursor = 0
    for token in tokens:
        token_start = cursor
        token_end = cursor + len(token.text)
        while sentence_index < len(spans) - 1 and token_start >= spans[sentence_index][1]:
            sentence_index += 1
        token.sentence_index = sentence_index
        cursor = token_end
    return text


def analyze_scored_tokens(tokens: list[TokenScore]) -> AnalysisResult:
    if not tokens:
        return AnalysisResult(text="", tokens=[], sentences=[])

    text = assign_sentence_indexes(tokens)
    sentence_spans = split_sentence_spans(text)
    sentence_cues = [claim_cues_for_text(sentence) for _, _, sentence in sentence_spans]

    for token in tokens:
        cues = sentence_cues[min(token.sentence_index, len(sentence_cues) - 1)]
        token.claim_cues = cues
        token.uncertainty = uncertainty_from_stats(token.probability, token.entropy, token.margin)
        token.factual_risk = factual_risk_from_cues(token.uncertainty, cues)

    grouped: dict[int, list[TokenScore]] = defaultdict(list)
    for token in tokens:
        grouped[token.sentence_index].append(token)

    sentences: list[SentenceRisk] = []
    for index, (_, _, sentence_text) in enumerate(sentence_spans):
        sentence_tokens = grouped.get(index, [])
        if sentence_tokens:
            mean_uncertainty = sum(t.uncertainty for t in sentence_tokens) / len(sentence_tokens)
            max_token_risk = max(t.factual_risk for t in sentence_tokens)
            cue_weight = clamp(len(sentence_cues[index]) / 3.0)
            factual_risk = clamp((0.5 * mean_uncertainty) + (0.3 * max_token_risk) + (0.2 * cue_weight))
        else:
            mean_uncertainty = 0.0
            max_token_risk = 0.0
            factual_risk = 0.0
        sentences.append(
            SentenceRisk(
                index=index + 1,
                text=sentence_text,
                factual_risk=factual_risk,
                mean_uncertainty=mean_uncertainty,
                max_token_risk=max_token_risk,
                claim_cues=sentence_cues[index],
            )
        )
    return AnalysisResult(text=text, tokens=tokens, sentences=sentences)


def normalized_entropy(probabilities) -> float:
    entropy = -float((probabilities * (probabilities + 1e-12).log()).sum().item())
    return clamp(entropy / math.log(max(2, probabilities.numel())))
