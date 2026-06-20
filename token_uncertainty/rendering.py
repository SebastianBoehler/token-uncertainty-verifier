from __future__ import annotations

from html import escape

from token_uncertainty.types import SentenceRisk, TokenScore


def percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.1f}%"


def risk_color(risk: float) -> str:
    hue = int(142 * (1.0 - max(0.0, min(1.0, risk))))
    saturation = 82
    lightness = 88 - int(18 * risk)
    return f"hsl({hue} {saturation}% {lightness}%)"


def token_title(token: TokenScore) -> str:
    parts = [
        f"probability: {percent(token.probability)}",
        f"uncertainty: {percent(token.uncertainty)}",
        f"factual risk: {percent(token.factual_risk)}",
    ]
    if token.rank is not None:
        parts.append(f"rank: {token.rank}")
    if token.margin is not None:
        parts.append(f"margin: {token.margin:.4f}")
    if token.claim_cues:
        parts.append(f"claim cues: {', '.join(token.claim_cues)}")
    return " | ".join(parts)


def render_token_overlay(tokens: list[TokenScore], threshold: float = 0.45) -> str:
    spans = []
    for token in tokens:
        border = "#9b1c1c" if token.factual_risk >= threshold else "transparent"
        spans.append(
            "<span class='uv-token' "
            f"style='background:{risk_color(token.factual_risk)};border-color:{border}' "
            f"title='{escape(token_title(token), quote=True)}'>"
            f"{escape(token.text)}</span>"
        )
    return (
        "<style>"
        ".uv-overlay{font:15px/2.1 ui-monospace,SFMono-Regular,Menlo,monospace;"
        "white-space:pre-wrap;padding:14px;border:1px solid #ddd;border-radius:8px;"
        "background:#fff;color:#111}"
        ".uv-token{display:inline;border:1px solid transparent;border-radius:5px;"
        "padding:2px 3px;margin:0 1px}"
        ".uv-token:hover{outline:2px solid #111;outline-offset:1px}"
        ".uv-legend{display:flex;gap:8px;align-items:center;margin:8px 0 12px;"
        "font:13px system-ui;color:#444}"
        ".uv-box{display:inline-block;width:18px;height:12px;border-radius:3px}"
        "</style>"
        "<div class='uv-legend'>"
        "<span class='uv-box' style='background:hsl(142 82% 88%)'></span>lower risk"
        "<span class='uv-box' style='background:hsl(55 82% 80%)'></span>review"
        "<span class='uv-box' style='background:hsl(0 82% 70%)'></span>verify"
        "</div>"
        f"<div class='uv-overlay'>{''.join(spans) or 'No tokens to display.'}</div>"
    )


def render_sentence_overlay(sentences: list[SentenceRisk]) -> str:
    blocks = []
    for sentence in sentences:
        cue_text = ", ".join(sentence.claim_cues) if sentence.claim_cues else "none"
        blocks.append(
            "<div class='uv-sentence' "
            f"style='background:{risk_color(sentence.factual_risk)}'>"
            f"<strong>S{sentence.index}</strong> "
            f"<span>{escape(sentence.text)}</span>"
            f"<small>risk {percent(sentence.factual_risk)} | cues: {escape(cue_text)}</small>"
            "</div>"
        )
    return (
        "<style>"
        ".uv-sentences{display:grid;gap:8px}"
        ".uv-sentence{padding:10px 12px;border-radius:8px;border:1px solid #ddd;color:#111}"
        ".uv-sentence strong{margin-right:8px}"
        ".uv-sentence small{display:block;margin-top:5px;color:#333;font:12px system-ui}"
        "</style>"
        f"<div class='uv-sentences'>{''.join(blocks) or 'No sentences to display.'}</div>"
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
                round(token.factual_risk, 5),
                token.rank or "",
                round(token.margin, 5) if token.margin is not None else "",
                ", ".join(token.claim_cues),
            ]
        )
    return rows


def sentence_rows(sentences: list[SentenceRisk]) -> list[list[str | int | float]]:
    return [
        [
            sentence.index,
            sentence.text,
            round(sentence.factual_risk, 5),
            round(sentence.mean_uncertainty, 5),
            round(sentence.max_token_risk, 5),
            ", ".join(sentence.claim_cues),
        ]
        for sentence in sentences
    ]
