from __future__ import annotations

from difflib import SequenceMatcher
from html import escape
import re

from token_uncertainty.types import ContrastiveOption


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text)


def diff_opcodes(reference: str, candidate: str):
    reference_tokens = word_tokens(reference)
    candidate_tokens = word_tokens(candidate)
    matcher = SequenceMatcher(None, reference_tokens, candidate_tokens)
    return reference_tokens, candidate_tokens, matcher.get_opcodes()


def render_diff_words(tokens: list[str], css_class: str) -> str:
    return "".join(f"<span class='{css_class}'>{escape(token)}</span>" for token in tokens)


def summarize_changes(reference: str, candidate: str) -> list[str]:
    reference_tokens, candidate_tokens, opcodes = diff_opcodes(reference, candidate)
    changes = []
    for tag, first_start, first_end, second_start, second_end in opcodes:
        left = " ".join(reference_tokens[first_start:first_end])
        right = " ".join(candidate_tokens[second_start:second_end])
        if tag == "replace":
            changes.append(f"{left} -> {right}")
        elif tag == "delete":
            changes.append(f"removed: {left}")
        elif tag == "insert":
            changes.append(f"added: {right}")
    return [change for change in changes if change.strip()]


def render_diff(reference: str, candidate: str) -> str:
    reference_tokens, candidate_tokens, opcodes = diff_opcodes(reference, candidate)
    reference_parts = []
    candidate_parts = []
    for tag, first_start, first_end, second_start, second_end in opcodes:
        left = reference_tokens[first_start:first_end]
        right = candidate_tokens[second_start:second_end]
        if tag == "equal":
            reference_parts.append(render_diff_words(left, "uv-diff-same"))
            candidate_parts.append(render_diff_words(right, "uv-diff-same"))
        elif tag == "replace":
            reference_parts.append(render_diff_words(left, "uv-diff-remove"))
            candidate_parts.append(render_diff_words(right, "uv-diff-add"))
        elif tag == "delete":
            reference_parts.append(render_diff_words(left, "uv-diff-remove"))
        elif tag == "insert":
            candidate_parts.append(render_diff_words(right, "uv-diff-add"))

    changes = summarize_changes(reference, candidate)
    summary = "".join(f"<li>{escape(change)}</li>" for change in changes)
    if not summary:
        summary = "<li>No word-level changes.</li>"

    return (
        "<style>"
        ".uv-mode-note{margin:10px 0 14px;padding:10px 12px;border-left:4px solid #555;"
        "border-radius:8px;border:1px solid #ddd;background:#fafafa;font:13px/1.45 system-ui;color:#222}"
        ".uv-diff-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}"
        ".uv-diff-panel{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fff;color:#111}"
        ".uv-diff-panel h3{margin:0 0 8px;font:700 15px system-ui}"
        ".uv-diff-text{font:14px/2 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:normal}"
        ".uv-diff-text span{display:inline-block;margin:2px 2px;padding:2px 4px;border-radius:5px}"
        ".uv-diff-same{background:transparent;border:1px solid transparent}"
        ".uv-diff-remove{background:#fff1f2;border:1px solid #fb7185;color:#8a1f1f;font-weight:700}"
        ".uv-diff-add{background:#ecfdf5;border:1px solid #34d399;color:#065f46;font-weight:700}"
        ".uv-change-list{margin:8px 0 0;padding-left:20px;font:13px/1.5 system-ui}"
        "</style>"
        "<div class='uv-mode-note'><strong>Diff mode is not verification.</strong> "
        "It only shows textual changes between two versions. It can reveal that 1969 changed to 1972, "
        "but it cannot decide which value is true.</div>"
        "<h3>Changed spans</h3>"
        f"<ul class='uv-change-list'>{summary}</ul>"
        "<div class='uv-diff-grid'>"
        "<section class='uv-diff-panel'><h3>Reference</h3>"
        f"<div class='uv-diff-text'>{''.join(reference_parts)}</div></section>"
        "<section class='uv-diff-panel'><h3>Candidate</h3>"
        f"<div class='uv-diff-text'>{''.join(candidate_parts)}</div></section>"
        "</div>"
    )


def contrastive_rows(scores: list[ContrastiveOption]) -> list[list[str | int | float]]:
    return [
        [
            index,
            item.option,
            item.token_count,
            round(item.geometric_mean_probability, 6),
            round(item.mean_log_probability, 6),
            round(item.relative_weight, 6),
        ]
        for index, item in enumerate(scores, start=1)
    ]


def render_contrastive_scores(scores: list[ContrastiveOption], template: str) -> str:
    if not scores:
        return "<p>No alternatives to score.</p>"

    best = scores[0]
    bars = []
    for item in scores:
        width = max(2, int(100 * item.relative_weight))
        bars.append(
            "<div class='uv-contrast-row'>"
            f"<strong>{escape(item.option)}</strong>"
            "<div class='uv-contrast-track'>"
            f"<span style='width:{width}%'></span>"
            "</div>"
            f"<small>relative weight {100 * item.relative_weight:.1f}% | "
            f"geomean p {item.geometric_mean_probability:.4g} | "
            f"{item.token_count} token pieces</small>"
            "</div>"
        )

    return (
        "<style>"
        ".uv-contrast{display:grid;gap:10px;color:#111}"
        ".uv-contrast-note{padding:10px 12px;border-left:4px solid #555;border-radius:8px;"
        "border:1px solid #ddd;background:#fafafa;font:13px/1.45 system-ui;color:#222}"
        ".uv-contrast-row{border:1px solid #ddd;border-radius:8px;padding:10px;background:#fff}"
        ".uv-contrast-row strong{font:700 15px system-ui}"
        ".uv-contrast-track{height:10px;background:#eee;border-radius:999px;overflow:hidden;margin:8px 0}"
        ".uv-contrast-track span{display:block;height:10px;background:#f97316}"
        ".uv-contrast-row small{font:12px system-ui;color:#333}"
        "</style>"
        "<div class='uv-contrast'>"
        "<div class='uv-contrast-note'><strong>Contrastive scoring is not verification.</strong> "
        "It asks which candidate span is more likely under this scoring model in the same template. "
        "The best-scored option is not guaranteed to be true.</div>"
        f"<p><strong>Template:</strong> {escape(template)}</p>"
        f"<p><strong>Highest likelihood:</strong> {escape(best.option)}</p>"
        f"{''.join(bars)}"
        "</div>"
    )


def log_likelihood_gap(scores: list[ContrastiveOption]) -> float:
    if len(scores) < 2:
        return 0.0
    return scores[0].mean_log_probability - scores[1].mean_log_probability
