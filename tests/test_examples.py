from token_uncertainty.examples import (
    EXAMPLE_CASES,
    chat_example_labels,
    combined_example_text,
    example_labels,
    get_chat_example,
)


def test_example_labels_are_unique():
    labels = example_labels()

    assert len(labels) == len(set(labels))
    assert len(labels) == 4


def test_examples_cover_comparison_scenarios():
    labels = set(example_labels())
    text = combined_example_text()

    assert {"Reference wording", "Year-only error", "Contradictory variants", "False side sentence"} == labels
    assert "1969" in text
    assert "1972" in text
    assert "tourists to Mars" in text
    assert all(case.context for case in EXAMPLE_CASES)
    assert all(case.focus for case in EXAMPLE_CASES)


def test_chat_examples_fill_reference_and_message():
    labels = chat_example_labels()

    assert len(labels) == len(set(labels))
    assert {"Capital paraphrase", "Python creator paraphrase", "False side sentence paraphrase"} <= set(labels)
    for label in labels:
        case = get_chat_example(label)
        assert case.reference
        assert case.message
