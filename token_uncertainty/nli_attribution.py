from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from html import escape
import math
import os
import re

DEFAULT_NLI_MODEL_ID = os.getenv("NLI_MODEL_ID", "cross-encoder/nli-deberta-v3-small")
WORD_RE = re.compile(r"\b\w+(?:[-']\w+)*\b")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]|[^.!?]+$")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    kind: str


@dataclass(frozen=True)
class AttributionRow:
    span: Span
    impact: float
    focus: float
    score_after: dict[str, float]


@dataclass(frozen=True)
class AttributionResult:
    reference: str
    candidate: str
    model_id: str
    objective: str
    base_score: dict[str, float]
    rows: list[AttributionRow]


def clean_label_map(labels: dict[int, str]) -> dict[int, str]:
    normalized = {key: value.lower() for key, value in labels.items()}
    expected = {"contradiction", "entailment", "neutral"}
    if expected - set(normalized.values()):
        raise ValueError(f"NLI model labels must include {sorted(expected)}.")
    return normalized


@lru_cache(maxsize=2)
def load_nli_model(model_id: str = DEFAULT_NLI_MODEL_ID):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from token_uncertainty.model_runner import preferred_device

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(preferred_device())
    model.eval()
    return tokenizer, model, clean_label_map(model.config.id2label)


def score_pair(reference: str, candidate: str, model_id: str = DEFAULT_NLI_MODEL_ID) -> dict[str, float]:
    import torch

    tokenizer, model, labels = load_nli_model(model_id)
    batch = tokenizer(reference, candidate, return_tensors="pt", truncation=True, max_length=512)
    batch = {key: value.to(next(model.parameters()).device) for key, value in batch.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**batch).logits[0], dim=-1).tolist()
    return {labels[index]: float(probabilities[index]) for index in range(len(probabilities))}


def sentence_spans(text: str) -> list[Span]:
    spans = []
    for match in SENTENCE_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append(Span(start, end, text[start:end], "sentence"))
    return spans or [Span(0, len(text), text, "sentence")]


def candidate_spans(text: str, max_words: int = 5) -> list[Span]:
    spans = sentence_spans(text)
    for sentence in list(spans):
        words = list(WORD_RE.finditer(text, sentence.start, sentence.end))
        for width in range(1, max_words + 1):
            for index in range(0, len(words) - width + 1):
                start, end = words[index].start(), words[index + width - 1].end()
                spans.append(Span(start, end, text[start:end], f"{width}-word"))
    unique = {(span.start, span.end): span for span in spans}
    return sorted(unique.values(), key=lambda item: (item.start, item.end))


def remove_span(text: str, span: Span) -> str:
    output = (text[: span.start] + text[span.end :]).strip()
    output = re.sub(r"\s+([.,;:!?])", r"\1", output)
    return re.sub(r"\s{2,}", " ", output)


def span_word_count(span: Span) -> int:
    return max(1, len(WORD_RE.findall(span.text)))


def focus_score(impact: float, span: Span) -> float:
    return max(0.0, impact) / math.sqrt(span_word_count(span))


def attribution_objective(base_score: dict[str, float]) -> str:
    if base_score["neutral"] > base_score["contradiction"] and base_score["neutral"] > base_score["entailment"]:
        return "increase_entailment"
    return "reduce_contradiction"


def span_impact(objective: str, base_score: dict[str, float], next_score: dict[str, float]) -> float:
    if objective == "increase_entailment":
        return next_score["entailment"] - base_score["entailment"]
    return base_score["contradiction"] - next_score["contradiction"]


def analyze_nli_attribution(
    reference: str,
    candidate: str,
    model_id: str = DEFAULT_NLI_MODEL_ID,
    max_words: int = 5,
) -> AttributionResult:
    base = score_pair(reference, candidate, model_id)
    objective = attribution_objective(base)
    rows = []
    for span in candidate_spans(candidate, max_words=max_words):
        next_score = score_pair(reference, remove_span(candidate, span), model_id)
        impact = span_impact(objective, base, next_score)
        rows.append(AttributionRow(span, impact, focus_score(impact, span), next_score))
    rows.sort(key=lambda row: row.focus, reverse=True)
    return AttributionResult(reference, candidate, model_id, objective, base, rows)


def overlaps(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


def selected_overlay_spans(rows: list[AttributionRow]) -> list[AttributionRow]:
    max_focus = max((row.focus for row in rows), default=0.0)
    if max_focus < 0.02:
        return []
    selected = []
    for row in sorted(rows, key=lambda item: item.focus, reverse=True):
        if row.focus < max(0.02, max_focus * 0.65):
            continue
        if any(overlaps(row.span, existing.span) for existing in selected):
            continue
        selected.append(row)
    return selected


def word_focuses(text: str, rows: list[AttributionRow]) -> dict[tuple[int, int], float]:
    selected = selected_overlay_spans(rows)
    max_focus = max((row.focus for row in selected), default=1.0)
    focuses = {(match.start(), match.end()): 0.0 for match in WORD_RE.finditer(text)}
    for row in selected:
        normalized = row.focus / max_focus if max_focus else 0.0
        for start, end in focuses:
            if row.span.start <= start and end <= row.span.end:
                focuses[(start, end)] = max(focuses[(start, end)], normalized)
    return focuses


def render_nli_overlay(text: str, rows: list[AttributionRow]) -> str:
    focuses = word_focuses(text, rows)
    pieces = []
    cursor = 0
    for match in WORD_RE.finditer(text):
        pieces.append(escape(text[cursor : match.start()]))
        focus = focuses.get((match.start(), match.end()), 0.0)
        if focus < 0.12:
            pieces.append(escape(match.group(0)))
        else:
            alpha = 0.08 + 0.72 * focus
            cls = "uv-nli-word uv-nli-hot" if focus >= 0.72 else "uv-nli-word"
            pieces.append(
                f"<span class='{cls}' style='--a:{alpha:.3f}' title='focus {focus:.2f}'>"
                f"{escape(match.group(0))}</span>"
            )
        cursor = match.end()
    pieces.append(escape(text[cursor:]))
    return "".join(pieces)


def render_compact_nli_overlay(result: AttributionResult) -> str:
    score_title = " | ".join(
        f"{label}: {result.base_score[label]:.3f}" for label in ("contradiction", "entailment", "neutral")
    )
    title = (
        f"{score_title}. NLI span shortening compares candidate text to reference/evidence; "
        "it localizes disagreement, not truth by itself."
    )
    return (
        "<style>"
        ".uv-chat-nli{font:16px/2.05 system-ui;color:#111;white-space:normal}"
        ".uv-chat-nli .uv-nli-word{background:rgba(247,116,75,var(--a));"
        "border:1px solid rgba(169,64,40,calc(var(--a) + .08));border-radius:5px;"
        "padding:1px 3px;margin:0 1px}"
        ".uv-chat-nli .uv-nli-hot{font-weight:760;box-shadow:inset 0 -3px 0 rgba(159,43,31,.72)}"
        "</style>"
        f"<div class='uv-chat-nli' title='{escape(title, quote=True)}'>"
        f"{render_nli_overlay(result.candidate, result.rows)}</div>"
    )


def render_score(score_map: dict[str, float]) -> str:
    return " · ".join(f"<b>{label}</b> {score_map[label]:.3f}" for label in ("contradiction", "entailment", "neutral"))


def objective_text(objective: str) -> str:
    if objective == "increase_entailment":
        return "neutral pair: highlight spans whose removal increases entailment"
    return "contradictory pair: highlight spans whose removal reduces contradiction"


def render_row(row: AttributionRow) -> str:
    impact_width = max(0.0, min(100.0, row.impact * 100))
    focus_width = max(0.0, min(100.0, row.focus * 100))
    return (
        "<tr>"
        f"<td><span class='uv-nli-kind'>{escape(row.span.kind)}</span></td>"
        f"<td><code>{escape(row.span.text)}</code></td>"
        f"<td class='uv-nli-metric'><span class='uv-nli-fill impact' style='--w:{impact_width:.1f}%'></span><b>{row.impact:+.3f}</b></td>"
        f"<td class='uv-nli-metric'><span class='uv-nli-fill focus' style='--w:{focus_width:.1f}%'></span><b>{row.focus:.3f}</b></td>"
        f"<td>{render_score(row.score_after)}</td>"
        "</tr>"
    )


def render_nli_attribution(result: AttributionResult) -> str:
    rows = result.rows[:8]
    table = "".join(render_row(row) for row in rows)
    return (
        "<style>"
        ".uv-nli{display:grid;gap:12px;color:#111}.uv-nli-note{padding:10px 12px;border-left:4px solid #555;border-radius:8px;border:1px solid #ddd;background:#fafafa;font:13px/1.45 system-ui;color:#222}"
        ".uv-nli-refs{display:grid;gap:6px}.uv-nli-refs div{background:#f6f4ef;border-radius:6px;padding:8px 10px;font:13px system-ui}.uv-nli-refs b{display:inline-block;min-width:80px;color:#5f584d}"
        ".uv-nli-overlay{font:19px/2.1 system-ui;border:1px solid #ddd;border-radius:8px;background:#fff;padding:14px}.uv-nli-word{background:rgba(247,116,75,var(--a));border:1px solid rgba(169,64,40,calc(var(--a) + .08));border-radius:5px;padding:1px 3px;margin:0 1px}.uv-nli-hot{font-weight:760;box-shadow:inset 0 -3px 0 rgba(159,43,31,.72)}"
        ".uv-nli table{width:100%;border-collapse:collapse;font:12px system-ui;table-layout:fixed}.uv-nli th,.uv-nli td{text-align:left;vertical-align:top;border-bottom:1px solid #eee;padding:7px 6px}.uv-nli th:nth-child(1),.uv-nli td:nth-child(1){width:72px}.uv-nli th:nth-child(3),.uv-nli td:nth-child(3),.uv-nli th:nth-child(4),.uv-nli td:nth-child(4){width:72px}.uv-nli th:nth-child(5),.uv-nli td:nth-child(5){width:230px}"
        ".uv-nli code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:normal}.uv-nli-kind{border:1px solid #d6cdbd;border-radius:999px;padding:2px 6px;color:#625c51;background:#faf7f0}.uv-nli-metric{position:relative;overflow:hidden;font-variant-numeric:tabular-nums}.uv-nli-metric b{position:relative;z-index:1}.uv-nli-fill{position:absolute;inset:6px auto 6px 4px;width:var(--w);max-width:calc(100% - 8px);border-radius:4px;opacity:.34}.uv-nli-fill.impact{background:#f36f55}.uv-nli-fill.focus{background:#f2b84b}"
        "</style>"
        "<div class='uv-nli'>"
        "<div class='uv-nli-note'><strong>NLI span attribution is not factual verification.</strong> "
        "It uses black-box ablation around an entailment model to localize disagreement between reference and candidate text. "
        "Impact is raw NLI score change after removal. Focus is impact divided by sqrt(span words), used for the overlay.</div>"
        f"<div class='uv-nli-refs'><div><b>Reference</b>{escape(result.reference)}</div><div><b>Candidate</b>{escape(result.candidate)}</div></div>"
        f"<p><strong>Base score:</strong> {render_score(result.base_score)}<br><strong>Mode:</strong> {escape(objective_text(result.objective))}<br><strong>NLI model:</strong> {escape(result.model_id)}</p>"
        f"<div class='uv-nli-overlay'>{render_nli_overlay(result.candidate, result.rows)}</div>"
        "<table><thead><tr><th>Span</th><th>Removed text</th><th>Impact</th><th>Focus</th><th>Score after removal</th></tr></thead>"
        f"<tbody>{table}</tbody></table></div>"
    )


def attribution_rows(result: AttributionResult) -> list[list[str | float]]:
    return [
        [row.span.kind, row.span.text, round(row.impact, 6), round(row.focus, 6), round(row.score_after["contradiction"], 6), round(row.score_after["entailment"], 6), round(row.score_after["neutral"], 6)]
        for row in result.rows[:20]
    ]
