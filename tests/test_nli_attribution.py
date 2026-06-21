from token_uncertainty.nli_attribution import (
    AttributionResult,
    AttributionRow,
    Span,
    attribution_rows,
    candidate_spans,
    render_nli_attribution,
    selected_overlay_spans,
)


def test_candidate_spans_do_not_cross_sentence_boundaries():
    spans = candidate_spans("A happened in 1969. B happened in 1972.", max_words=3)
    texts = {span.text for span in spans}

    assert "1969. B" not in texts
    assert "in 1972" in texts
    assert "B happened in 1972." in texts


def test_selected_overlay_prefers_focused_non_overlapping_span():
    rows = [
        AttributionRow(Span(0, 7, "in 1972", "2-word"), 0.996, 0.704, {}),
        AttributionRow(Span(3, 7, "1972", "1-word"), 0.994, 0.994, {}),
        AttributionRow(Span(20, 26, "Berlin", "1-word"), 0.2, 0.2, {}),
    ]

    selected = selected_overlay_spans(rows)

    assert [row.span.text for row in selected] == ["1972"]


def test_render_nli_attribution_states_limitation_and_scores():
    result = AttributionResult(
        reference="Apollo 11 landed on the Moon in 1969.",
        candidate="Apollo 11 landed on the Moon in 1972.",
        model_id="example/nli",
        objective="reduce_contradiction",
        base_score={"contradiction": 0.99, "entailment": 0.0, "neutral": 0.01},
        rows=[
            AttributionRow(
                Span(35, 39, "1972", "1-word"),
                0.98,
                0.98,
                {"contradiction": 0.01, "entailment": 0.95, "neutral": 0.04},
            )
        ],
    )

    html = render_nli_attribution(result)
    rows = attribution_rows(result)

    assert "not factual verification" in html
    assert "1972" in html
    assert rows[0][1] == "1972"
    assert rows[0][2] == 0.98
