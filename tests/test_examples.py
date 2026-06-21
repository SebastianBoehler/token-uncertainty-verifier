from token_uncertainty.examples import EXAMPLE_CASES, combined_example_text, example_labels


def test_example_labels_are_unique():
    labels = example_labels()

    assert len(labels) == len(set(labels))
    assert len(labels) == 4


def test_examples_cover_comparison_scenarios():
    labels = set(example_labels())
    text = combined_example_text()

    assert {"Correct baseline", "Year-only error", "Contradictory variants", "False side sentence"} == labels
    assert "1969" in text
    assert "1972" in text
    assert "tourists to Mars" in text
    assert all(case.context for case in EXAMPLE_CASES)
