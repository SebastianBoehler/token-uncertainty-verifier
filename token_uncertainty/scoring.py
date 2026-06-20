from __future__ import annotations

import math
import re
from collections import defaultdict

from token_uncertainty.types import AnalysisResult, SentenceRisk, TokenScore

CLAIM_TERMS = {
    "announced", "born", "built", "claim", "created", "died", "discovered", "founded",
    "invented", "landed", "launched", "located", "opened", "published", "released",
    "reported", "won",
}

ENTITY_STOPWORDS = {"A", "An", "As", "At", "In", "It", "On", "That", "The", "This"}

DATE_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
REFERENCE_PATTERN = re.compile(r"\b(?:chapter|figure|fig\.|page|report|ref\.|table)\s*\d+\b", re.I)
IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9-]*-\d+\b")
NAMED_ENTITY_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9-]{2,}(?:\s+(?:[A-Z][A-Za-z0-9-]{2,}|\d+))*\b"
)

CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("date", DATE_PATTERN),
    ("number", NUMBER_PATTERN),
    ("reference", REFERENCE_PATTERN),
    ("identifier", IDENTIFIER_PATTERN),
    ("named-entity", NAMED_ENTITY_PATTERN),
)

CUE_WEIGHTS = {
    "conflict": 1.0,
    "identifier": 0.9,
    "date": 0.9,
    "reference": 0.85,
    "number": 0.75,
    "named-entity": 0.65,
    "claim-term": 0.25,
}

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


def is_ignored_named_entity(text: str) -> bool:
    words = text.split()
    return len(words) == 1 and words[0] in ENTITY_STOPWORDS


def normalized_subject(text: str) -> str:
    words = re.sub(r"\s+", " ", text).strip().split()
    while words and words[0] in ENTITY_STOPWORDS:
        words.pop(0)
    return " ".join(words).lower()


def spans_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and first[1] > second[0]


def iter_claim_matches(text: str):
    for name, pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            if name == "named-entity" and is_ignored_named_entity(match.group(0)):
                continue
            yield name, match


def find_conflict_spans(sentence_spans: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    observed: dict[tuple[str, str], dict[str, list[tuple[int, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for sentence_index, (start, _, sentence) in enumerate(sentence_spans):
        subject_match = NAMED_ENTITY_PATTERN.search(sentence)
        if not subject_match:
            continue
        subject = normalized_subject(subject_match.group(0))
        if not subject:
            continue

        date_spans = [match.span() for match in DATE_PATTERN.finditer(sentence)]
        for date_match in DATE_PATTERN.finditer(sentence):
            observed[(subject, "date")][date_match.group(0)].append(
                (start + date_match.start(), start + date_match.end(), sentence_index)
            )

        subject_span = subject_match.span()
        ignored_number_spans = [subject_span, *date_spans]
        for number_match in NUMBER_PATTERN.finditer(sentence):
            span = number_match.span()
            if any(spans_overlap(span, ignored) for ignored in ignored_number_spans):
                continue
            observed[(subject, "number")][number_match.group(0)].append(
                (start + number_match.start(), start + number_match.end(), sentence_index)
            )

    conflict_spans: list[tuple[int, int]] = []
    for values in observed.values():
        sentence_ids = {sentence_index for spans in values.values() for _, _, sentence_index in spans}
        if len(values) > 1 and len(sentence_ids) > 1:
            for spans in values.values():
                conflict_spans.extend((start, end) for start, end, _ in spans)
    return conflict_spans


def claim_cues_for_text(text: str) -> tuple[str, ...]:
    cues = {name for name, _ in iter_claim_matches(text)}
    words = {word.lower().strip(".,:;()[]") for word in text.split()}
    if words & CLAIM_TERMS:
        cues.add("claim-term")
    return tuple(sorted(cues))


def claim_cues_by_token(
    text: str,
    spans: list[tuple[int, int]],
    conflict_spans: list[tuple[int, int]],
) -> list[tuple[str, ...]]:
    token_cues: list[set[str]] = [set() for _ in spans]

    for cue_name, match in iter_claim_matches(text):
        match_start, match_end = match.span()
        for index, (token_start, token_end) in enumerate(spans):
            if token_start < match_end and token_end > match_start:
                if cue_name == "named-entity" and text[token_start:token_end].strip() in ENTITY_STOPWORDS:
                    continue
                token_cues[index].add(cue_name)

    for conflict_start, conflict_end in conflict_spans:
        for index, (token_start, token_end) in enumerate(spans):
            if token_start < conflict_end and token_end > conflict_start:
                token_cues[index].add("conflict")

    for index, (token_start, token_end) in enumerate(spans):
        token_text = text[token_start:token_end]
        words = {word.lower().strip(".,:;()[]") for word in token_text.split()}
        if words & CLAIM_TERMS:
            token_cues[index].add("claim-term")

    return tuple(tuple(sorted(cues)) for cues in token_cues)


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


def cue_salience(cues: tuple[str, ...]) -> float:
    if not cues:
        return 0.0
    return max(CUE_WEIGHTS.get(cue, 0.4) for cue in cues)


def factual_risk_from_cues(uncertainty: float, cues: tuple[str, ...]) -> float:
    if "conflict" in cues:
        return clamp(0.9 + (0.1 * uncertainty))
    salience = cue_salience(cues)
    if not cues:
        return clamp(0.12 * uncertainty)
    return clamp((0.25 * uncertainty) + (0.75 * salience))


def assign_sentence_indexes(tokens: list[TokenScore]) -> tuple[str, list[tuple[int, int]]]:
    text = "".join(token.text for token in tokens)
    spans = split_sentence_spans(text)
    sentence_index = 0
    cursor = 0
    token_spans: list[tuple[int, int]] = []
    for token in tokens:
        token_start = cursor
        token_end = cursor + len(token.text)
        token_spans.append((token_start, token_end))
        while sentence_index < len(spans) - 1 and token_start >= spans[sentence_index][1]:
            sentence_index += 1
        token.sentence_index = sentence_index
        cursor = token_end
    return text, token_spans


def analyze_scored_tokens(tokens: list[TokenScore]) -> AnalysisResult:
    if not tokens:
        return AnalysisResult(text="", tokens=[], sentences=[])

    text, token_spans = assign_sentence_indexes(tokens)
    sentence_spans = split_sentence_spans(text)
    sentence_cues = [claim_cues_for_text(sentence) for _, _, sentence in sentence_spans]
    conflict_spans = find_conflict_spans(sentence_spans)
    token_cues = claim_cues_by_token(text, token_spans, conflict_spans)

    for index, token in enumerate(tokens):
        cues = token_cues[index]
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
            cue_weight = cue_salience(sentence_cues[index])
            factual_risk = clamp((0.2 * mean_uncertainty) + (0.55 * max_token_risk) + (0.25 * cue_weight))
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
