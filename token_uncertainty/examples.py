from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CONTEXT = (
    "Treat this as an uncertainty pass. The overlay shows model-distribution "
    "uncertainty only; it does not prove truth or falsehood."
)

DEFAULT_REFERENCE_TEXT = (
    "Apollo 11 landed on the Moon in 1969. "
    "The Eiffel Tower opened in 1889. "
    "OpenAI released GPT-4 in 2023."
)

DEFAULT_CANDIDATE_TEXT = (
    "Apollo 11 landed on the Moon in 1972. "
    "The Eiffel Tower opened in 1889. "
    "OpenAI released GPT-4 in 2023."
)

DEFAULT_NLI_REFERENCE_TEXT = "Apollo 11 landed on the Moon in 1969."
DEFAULT_NLI_CANDIDATE_TEXT = "Apollo 11 landed on the Moon in 1972."

DEFAULT_CONTRASTIVE_TEMPLATE = "Apollo 11 landed on the Moon in {answer}."
DEFAULT_CONTRASTIVE_OPTIONS = ("1969", "1972", "1970")


@dataclass(frozen=True)
class ExampleCase:
    label: str
    note: str
    text: str
    focus: tuple[str, ...]
    context: str = DEFAULT_CONTEXT


EXAMPLE_CASES = (
    ExampleCase(
        label="Reference wording",
        note=(
            "A plausible reference version. Highlights still mean model uncertainty, "
            "not correctness or incorrectness."
        ),
        focus=(
            "No truth label",
            "Use as comparison anchor",
            "Check whether highlights are selective",
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
            "One value is changed relative to the reference. The model score may "
            "or may not isolate that exact value."
        ),
        focus=(
            "Changed value: 1969 to 1972",
            "Watch sentence S1",
            "Needs retrieval for truth",
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
            "Two subjects are repeated with incompatible years. The app no longer "
            "uses rule-based contradiction boosts, so this tests distribution-only visibility."
        ),
        focus=(
            "1969 vs 1972",
            "1889 vs 1899",
            "No rule-based conflict boost",
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
            "A side sentence is inserted into otherwise similar wording. The overlay "
            "can only show whether the local model was uncertain there."
        ),
        focus=(
            "Added side sentence",
            "Watch sentence S2",
            "External evidence needed",
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
