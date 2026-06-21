from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CONTEXT = (
    "Treat this as a verification triage pass. The overlay should identify "
    "claim-heavy spans and internal conflicts; it does not prove truth or falsehood."
)


@dataclass(frozen=True)
class ExampleCase:
    label: str
    note: str
    text: str
    context: str = DEFAULT_CONTEXT


EXAMPLE_CASES = (
    ExampleCase(
        label="Correct baseline",
        note=(
            "Known factual claims. Dates and named entities can still be highlighted "
            "because they are verification-worthy, not because they are false."
        ),
        text=(
            "Apollo 11 landed on the Moon in 1969. "
            "The Eiffel Tower opened in 1889. "
            "OpenAI released GPT-4 in 2023."
        ),
    ),
    ExampleCase(
        label="Year-only error",
        note=(
            "The Apollo year is wrong, but without retrieval the overlay can only "
            "treat it as a risky date-bearing claim."
        ),
        text=(
            "Apollo 11 landed on the Moon in 1972. "
            "The Eiffel Tower opened in 1889. "
            "OpenAI released GPT-4 in 2023."
        ),
    ),
    ExampleCase(
        label="Contradictory variants",
        note=(
            "Repeated subjects with conflicting years create visible token-level "
            "conflict flags on the exact values."
        ),
        text=(
            "Apollo 11 landed on the Moon in 1969. "
            "Apollo 11 landed on the Moon in 1972. "
            "The Eiffel Tower opened in 1889. "
            "The Eiffel Tower opened in 1899."
        ),
    ),
    ExampleCase(
        label="False side sentence",
        note=(
            "A mostly correct answer with one unsupported side claim. The overlay "
            "marks the side claim's date and entity tokens for review, but external "
            "evidence is needed."
        ),
        text=(
            "Apollo 11 landed on the Moon in 1969. "
            "The mission also carried tourists to Mars in 1970. "
            "The Eiffel Tower opened in 1889."
        ),
    ),
)


def example_labels() -> list[str]:
    return [case.label for case in EXAMPLE_CASES]


def get_example(label: str) -> ExampleCase:
    for case in EXAMPLE_CASES:
        if case.label == label:
            return case
    return EXAMPLE_CASES[0]


def combined_example_text() -> str:
    return " ".join(case.text for case in EXAMPLE_CASES)
