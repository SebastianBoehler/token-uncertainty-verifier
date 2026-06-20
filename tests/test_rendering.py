from token_uncertainty.rendering import render_token_overlay, token_rows
from token_uncertainty.types import TokenScore


def test_render_token_overlay_escapes_html():
    html = render_token_overlay([TokenScore(text="<script>", factual_risk=0.9)])

    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_token_rows_keep_numeric_scores():
    rows = token_rows(
        [
            TokenScore(
                text="token",
                probability=0.123456,
                uncertainty=0.4,
                factual_risk=0.5,
                rank=7,
                margin=-0.1,
            )
        ]
    )

    assert rows[0][2] == 0.12346
    assert rows[0][5] == 7
