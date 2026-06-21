from __future__ import annotations

from dataclasses import dataclass
from html import escape
import re

from token_uncertainty.types import SentenceRisk, TokenScore


@dataclass(frozen=True)
class HotSpan:
    text: str
    score: float


@dataclass(frozen=True)
class WordSpan:
    text: str
    start: int
    end: int
    uncertainty: float
    token_count: int
    is_word: bool


@dataclass(frozen=True)
class ComparisonSection:
    label: str
    note: str
    focus: tuple[str, ...]
    hot_spans: list[HotSpan]
    token_html: str
    sentence_html: str


def percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.1f}%"


def score_color(score: float) -> str:
    hue = int(142 * (1.0 - max(0.0, min(1.0, score))))
    saturation = 82
    lightness = 88 - int(18 * score)
    return f"hsl({hue} {saturation}% {lightness}%)"


def word_title(span: WordSpan) -> str:
    parts = [
        f"uncertainty: {percent(span.uncertainty)}",
        f"token pieces: {span.token_count}",
    ]
    return " | ".join(parts)


def word_spans_from_tokens(tokens: list[TokenScore]) -> tuple[str, list[WordSpan]]:
    text = "".join(token.text for token in tokens)
    offsets = []
    cursor = 0
    for token in tokens:
        start = cursor
        cursor += len(token.text)
        offsets.append((start, cursor, token))

    spans = []
    for match in re.finditer(r"\w+(?:[-']\w+)*|[^\w\s]", text):
        start, end = match.span()
        matching = [token for left, right, token in offsets if left < end and right > start]
        if not matching:
            continue
        spans.append(
            WordSpan(
                text=match.group(0),
                start=start,
                end=end,
                uncertainty=max(token.uncertainty for token in matching),
                token_count=len(matching),
                is_word=any(character.isalnum() for character in match.group(0)),
            )
        )
    return text, spans


def grouped_hot_spans(
    tokens: list[TokenScore],
    threshold: float,
    limit: int = 6,
    max_chars: int = 52,
) -> list[HotSpan]:
    text, word_spans = word_spans_from_tokens(tokens)
    groups: list[list[WordSpan]] = []
    current: list[WordSpan] = []
    for span in word_spans:
        if span.is_word and span.uncertainty >= threshold:
            candidate_start = current[0].start if current else span.start
            candidate = text[candidate_start:span.end].strip()
            if current and len(candidate) > max_chars:
                groups.append(current)
                current = []
            current.append(span)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    spans = []
    for group in groups:
        span_text = re.sub(r"\s+", " ", text[group[0].start:group[-1].end]).strip()
        if span_text:
            spans.append(HotSpan(text=span_text, score=max(span.uncertainty for span in group)))
    return sorted(spans, key=lambda span: span.score, reverse=True)[:limit]


def render_hot_spans(spans: list[HotSpan]) -> str:
    if not spans:
        return "<div class='uv-hot-empty'>No highlighted spans at this threshold.</div>"
    chips = [
        "<span class='uv-hot-chip'>"
        f"<strong>{escape(span.text)}</strong>"
        f"<em>{percent(span.score)}</em>"
        "</span>"
        for span in spans
    ]
    return f"<div class='uv-hot-list'>{''.join(chips)}</div>"


def render_token_overlay(tokens: list[TokenScore], threshold: float = 0.82) -> str:
    text, word_spans = word_spans_from_tokens(tokens)
    chunks = []
    cursor = 0
    for span in word_spans:
        if cursor < span.start:
            chunks.append(escape(text[cursor:span.start]))
        flagged = span.is_word and span.uncertainty >= threshold
        background = score_color(span.uncertainty) if flagged else "transparent"
        border = "#9b1c1c" if flagged else "transparent"
        weight = "700" if flagged else "400"
        shadow = "0 0 0 1px rgba(155,28,28,.16)" if flagged else "none"
        chunks.append(
            "<span class='uv-word' "
            f"style='background:{background};border-color:{border};"
            f"font-weight:{weight};box-shadow:{shadow}' "
            f"title='{escape(word_title(span), quote=True)}'>"
            f"{escape(span.text)}</span>"
        )
        cursor = span.end
    if cursor < len(text):
        chunks.append(escape(text[cursor:]))
    return (
        "<style>"
        ".uv-overlay{font:15px/2.1 ui-monospace,SFMono-Regular,Menlo,monospace;"
        "white-space:pre-wrap;padding:14px;border:1px solid #ddd;border-radius:8px;"
        "background:#fff;color:#111}"
        ".uv-word{display:inline;border:1px solid transparent;border-radius:5px;"
        "padding:2px 3px;margin:0 1px}"
        ".uv-word:hover{outline:2px solid #111;outline-offset:1px}"
        ".uv-legend{display:flex;gap:8px;align-items:center;margin:8px 0 12px;"
        "font:13px system-ui;color:#444}"
        ".uv-box{display:inline-block;width:18px;height:12px;border-radius:3px}"
        "</style>"
        "<div class='uv-legend'>"
        "<span class='uv-box' style='background:hsl(142 82% 88%)'></span>lower uncertainty"
        "<span class='uv-box' style='background:hsl(55 82% 80%)'></span>medium"
        "<span class='uv-box' style='background:hsl(0 82% 70%)'></span>high"
        "</div>"
        f"<div class='uv-overlay'>{''.join(chunks) or 'No tokens to display.'}</div>"
    )


def render_sentence_overlay(sentences: list[SentenceRisk]) -> str:
    blocks = []
    for sentence in sentences:
        score = sentence.uncertainty_score
        blocks.append(
            "<div class='uv-sentence'>"
            "<div class='uv-scorebar'>"
            f"<span style='width:{int(score * 100)}%;background:{score_color(score)}'></span>"
            "</div>"
            f"<strong>S{sentence.index}</strong>"
            f"<span>{escape(sentence.text)}</span>"
            f"<small>uncertainty {percent(score)} | max token {percent(sentence.max_token_uncertainty)}</small>"
            "</div>"
        )
    return (
        "<style>"
        ".uv-sentences{display:grid;gap:8px}"
        ".uv-sentence{position:relative;overflow:hidden;padding:10px 12px;border-radius:8px;"
        "border:1px solid #ddd;color:#111;background:#fff}"
        ".uv-scorebar{position:absolute;inset:0 0 auto 0;height:4px;background:#f2f2f2}"
        ".uv-scorebar span{display:block;height:4px}"
        ".uv-sentence strong{display:inline-block;margin-right:8px}"
        ".uv-sentence small{display:block;margin-top:5px;color:#333;font:12px system-ui}"
        "</style>"
        f"<div class='uv-sentences'>{''.join(blocks) or 'No sentences to display.'}</div>"
    )


def render_comparison_grid(sections: list[ComparisonSection]) -> str:
    cards = []
    for section in sections:
        focus = "".join(f"<span>{escape(item)}</span>" for item in section.focus)
        cards.append(
            "<section class='uv-compare-card'>"
            "<div class='uv-card-head'>"
            f"<h3>{escape(section.label)}</h3>"
            f"<p>{escape(section.note)}</p>"
            "</div>"
            f"<div class='uv-focus'>{focus}</div>"
            "<h4>Highest uncertainty spans</h4>"
            f"{render_hot_spans(section.hot_spans)}"
            "<h4>Word overlay</h4>"
            f"{section.token_html}"
            "<h4>Sentence summary</h4>"
            f"{section.sentence_html}"
            "</section>"
        )
    return (
        "<style>"
        ".uv-compare-note{margin:12px 0 14px;padding:10px 12px;border:1px solid #d4d4d4;"
        "border-left:4px solid #555;border-radius:8px;background:#fafafa;color:#222;font:13px/1.45 system-ui}"
        ".uv-compare-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));"
        "gap:14px;align-items:start}"
        ".uv-compare-card{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fff;color:#111}"
        ".uv-card-head{min-height:106px}"
        ".uv-compare-card h3{margin:0 0 6px;font:700 16px system-ui}"
        ".uv-compare-card h4{margin:12px 0 6px;font:600 13px system-ui;color:#333}"
        ".uv-compare-card p{font:13px/1.45 system-ui;color:#333;margin:0 0 8px}"
        ".uv-focus{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 10px}"
        ".uv-focus span{border:1px solid #d5d5d5;border-radius:999px;padding:4px 8px;"
        "font:12px system-ui;background:#f7f7f7;color:#222}"
        ".uv-hot-list{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 8px}"
        ".uv-hot-chip{display:inline-flex;gap:6px;align-items:center;border:1px solid #f1b9b9;"
        "border-radius:6px;padding:5px 7px;background:#fff5f5;font:12px system-ui;color:#111}"
        ".uv-hot-chip strong{font-weight:700}.uv-hot-chip em{font-style:normal;color:#8a1f1f}"
        ".uv-hot-empty{font:12px system-ui;color:#666;margin:0 0 8px}"
        ".uv-compare-card .uv-overlay{font-size:13px;line-height:1.85;padding:10px}"
        ".uv-compare-card .uv-sentence{padding:8px 10px}"
        "</style>"
        "<div class='uv-compare-note'><strong>No correctness labels.</strong> "
        "These examples compare model-distribution uncertainty. A highlighted span is not proven false; "
        "it is only a place to inspect first.</div>"
        f"<div class='uv-compare-grid'>{''.join(cards) or 'No comparisons to display.'}</div>"
    )


def token_rows(tokens: list[TokenScore]) -> list[list[str | int | float]]:
    rows: list[list[str | int | float]] = []
    for index, token in enumerate(tokens, start=1):
        rows.append(
            [
                index,
                token.text,
                round(token.probability, 5) if token.probability is not None else "",
                round(token.uncertainty, 5),
                token.rank or "",
                round(token.margin, 5) if token.margin is not None else "",
            ]
        )
    return rows


def sentence_rows(sentences: list[SentenceRisk]) -> list[list[str | int | float]]:
    return [
        [
            sentence.index,
            sentence.text,
            round(sentence.uncertainty_score, 5),
            round(sentence.mean_uncertainty, 5),
            round(sentence.max_token_uncertainty, 5),
        ]
        for sentence in sentences
    ]
