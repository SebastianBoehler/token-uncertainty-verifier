from __future__ import annotations

from html import escape

from token_uncertainty.rendering import score_color, word_spans_from_tokens, word_title
from token_uncertainty.types import TokenScore


def render_compact_token_overlay(
    tokens: list[TokenScore],
    threshold: float,
    title: str,
) -> str:
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
        chunks.append(
            "<span class='uv-word' "
            f"style='background:{background};border-color:{border};font-weight:{weight}' "
            f"title='{escape(word_title(span), quote=True)}'>"
            f"{escape(span.text)}</span>"
        )
        cursor = span.end
    if cursor < len(text):
        chunks.append(escape(text[cursor:]))

    return (
        "<style>"
        ".uv-chat-token{font:16px/2.05 ui-monospace,SFMono-Regular,Menlo,monospace;"
        "white-space:pre-wrap;color:#111}"
        ".uv-chat-token .uv-word{display:inline;border:1px solid transparent;border-radius:5px;"
        "padding:1px 3px;margin:0 1px}"
        ".uv-chat-token .uv-word:hover{outline:2px solid #111;outline-offset:1px}"
        "</style>"
        f"<div class='uv-chat-token' title='{escape(title, quote=True)}'>"
        f"{''.join(chunks) or 'No tokens to display.'}</div>"
    )
