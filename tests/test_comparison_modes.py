from token_uncertainty.comparison_modes import (
    contrastive_rows,
    render_contrastive_scores,
    render_diff,
    summarize_changes,
)
from token_uncertainty.types import ContrastiveOption


def test_diff_mode_surfaces_year_change():
    reference = "Apollo 11 landed on the Moon in 1969."
    candidate = "Apollo 11 landed on the Moon in 1972."

    changes = summarize_changes(reference, candidate)
    html = render_diff(reference, candidate)

    assert "1969 -> 1972" in changes
    assert "1969" in html
    assert "1972" in html
    assert "not verification" in html


def test_contrastive_rows_keep_ranked_scores():
    scores = [
        ContrastiveOption("1969", 1, -0.1, 0.9048, 0.7),
        ContrastiveOption("1972", 1, -1.2, 0.301, 0.3),
    ]

    rows = contrastive_rows(scores)

    assert rows[0][0] == 1
    assert rows[0][1] == "1969"
    assert rows[0][5] == 0.7


def test_contrastive_rendering_states_limitation():
    html = render_contrastive_scores(
        [ContrastiveOption("1969", 1, -0.1, 0.9048, 1.0)],
        "Apollo 11 landed on the Moon in {answer}.",
    )

    assert "Contrastive scoring is not verification" in html
    assert "1969" in html
