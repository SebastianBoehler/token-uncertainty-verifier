from token_uncertainty.rendering import (
    ComparisonSection,
    HotSpan,
    render_comparison_grid,
    render_token_overlay,
    token_rows,
)
from token_uncertainty.types import TokenScore


def test_render_token_overlay_escapes_html():
    html = render_token_overlay([TokenScore(text="<script>", uncertainty=0.9)])

    assert "&lt;" in html
    assert "&gt;" in html
    assert "<script>" not in html


def test_render_token_overlay_aggregates_token_pieces_to_words():
    html = render_token_overlay(
        [
            TokenScore(text="Ap", probability=0.02, uncertainty=0.98),
            TokenScore(text="ollo", probability=0.02, uncertainty=0.97),
            TokenScore(text=" 11", probability=0.02, uncertainty=0.96),
        ]
    )

    assert ">Apollo<" in html
    assert ">11<" in html
    assert ">Ap<" not in html


def test_render_token_overlay_does_not_show_rule_based_labels():
    html = render_token_overlay([TokenScore(text="Apollo", probability=0.02, uncertainty=0.98)])
    old_label = "claim " + "cues"
    old_score = "factual " + "risk"

    assert "Apollo" in html
    assert "uncertainty" in html
    assert old_label not in html
    assert old_score not in html


def test_token_rows_keep_numeric_scores():
    rows = token_rows(
        [
            TokenScore(
                text="token",
                probability=0.123456,
                uncertainty=0.4,
                rank=7,
                margin=-0.1,
            )
        ]
    )

    assert rows[0][2] == 0.12346
    assert rows[0][4] == 7


def test_render_comparison_grid_places_sections_side_by_side():
    html = render_comparison_grid(
        [
            ComparisonSection(
                label="Reference",
                note="No truth label.",
                focus=("anchor",),
                hot_spans=[HotSpan("1969", 0.91)],
                token_html="<div>tokens</div>",
                sentence_html="<div>sentences</div>",
            ),
            ComparisonSection(
                label="Wrong year",
                note="Changed value.",
                focus=("1969 to 1972",),
                hot_spans=[],
                token_html="<div>tokens</div>",
                sentence_html="<div>sentences</div>",
            ),
        ]
    )

    assert "uv-compare-grid" in html
    assert "grid-template-columns" in html
    assert "No correctness labels" in html
    assert "Reference" in html
    assert "Wrong year" in html
    assert "1969 to 1972" in html
