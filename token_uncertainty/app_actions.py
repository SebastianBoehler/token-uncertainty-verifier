from __future__ import annotations

import gradio as gr

from token_uncertainty.comparison_modes import (
    contrastive_rows,
    render_contrastive_scores,
    render_diff,
)
from token_uncertainty.examples import EXAMPLE_CASES, get_example
from token_uncertainty.model_runner import (
    DEFAULT_MODEL_ID,
    analyze_existing_text,
    generate_chat_with_scores,
    generate_with_scores,
    score_contrastive_options,
)
from token_uncertainty.nli_attribution import (
    DEFAULT_NLI_MODEL_ID,
    analyze_nli_attribution,
    attribution_rows,
    render_nli_attribution,
)
from token_uncertainty.rendering import (
    ComparisonSection,
    grouped_hot_spans,
    render_comparison_grid,
    render_sentence_overlay,
    render_token_overlay,
    sentence_rows,
    token_rows,
)


def _outputs(result, threshold: float):
    return (
        result.text,
        render_token_overlay(result.tokens, threshold),
        render_sentence_overlay(result.sentences),
        sentence_rows(result.sentences),
        token_rows(result.tokens),
    )


def run_generation(
    prompt: str,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    risk_threshold: float,
):
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    result = generate_with_scores(
        prompt=prompt.strip(),
        model_id=model_id.strip() or DEFAULT_MODEL_ID,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return _outputs(result, risk_threshold)


def run_chat_turn(
    message: str,
    history: list[dict[str, str]] | None,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    risk_threshold: float,
):
    if not message.strip():
        raise gr.Error("Enter a message.")
    chat_history = list(history or [])
    messages = [*chat_history, {"role": "user", "content": message.strip()}]
    result = generate_chat_with_scores(
        messages=messages,
        model_id=model_id.strip() or DEFAULT_MODEL_ID,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    next_history = [*messages, {"role": "assistant", "content": result.text}]
    return ("", next_history, next_history, *_outputs(result, risk_threshold))


def clear_chat():
    return "", [], [], "", "", "", [], []


def run_existing_text(text: str, context: str, model_id: str, risk_threshold: float):
    if not text.strip():
        raise gr.Error("Enter text to analyze.")
    result = analyze_existing_text(
        text=text,
        context=context,
        model_id=model_id.strip() or DEFAULT_MODEL_ID,
    )
    return _outputs(result, risk_threshold)


def example_note(label: str) -> str:
    case = get_example(label)
    return f"**{case.label}**\n\n{case.note}"


def run_example(label: str, model_id: str, risk_threshold: float):
    case = get_example(label)
    result = analyze_existing_text(
        text=case.text,
        context=case.context,
        model_id=model_id.strip() or DEFAULT_MODEL_ID,
    )
    return (case.context, case.text, example_note(label), *_outputs(result, risk_threshold))


def run_example_comparison(model_id: str, risk_threshold: float):
    sections = []
    for case in EXAMPLE_CASES:
        result = analyze_existing_text(
            text=case.text,
            context=case.context,
            model_id=model_id.strip() or DEFAULT_MODEL_ID,
        )
        sections.append(
            ComparisonSection(
                label=case.label,
                note=case.note,
                focus=case.focus,
                hot_spans=grouped_hot_spans(result.tokens, risk_threshold),
                token_html=render_token_overlay(result.tokens, risk_threshold),
                sentence_html=render_sentence_overlay(result.sentences),
            )
        )
    return render_comparison_grid(sections)


def run_diff_mode(reference: str, candidate: str):
    if not reference.strip() or not candidate.strip():
        raise gr.Error("Enter both reference and candidate text.")
    return render_diff(reference.strip(), candidate.strip())


def run_contrastive_mode(template: str, options: str, context: str, model_id: str):
    if not template.strip():
        raise gr.Error("Enter a template containing {answer}.")
    if "{answer}" not in template:
        raise gr.Error("Template must contain {answer}.")
    alternatives = [line.strip() for line in options.splitlines() if line.strip()]
    if len(alternatives) < 2:
        raise gr.Error("Enter at least two alternatives.")
    scores = score_contrastive_options(
        template=template.strip(),
        options=alternatives,
        context=context,
        model_id=model_id.strip() or DEFAULT_MODEL_ID,
    )
    return render_contrastive_scores(scores, template.strip()), contrastive_rows(scores)


def run_nli_attribution(reference: str, candidate: str, nli_model_id: str):
    if not reference.strip() or not candidate.strip():
        raise gr.Error("Enter both reference and candidate text.")
    result = analyze_nli_attribution(
        reference=reference.strip(),
        candidate=candidate.strip(),
        model_id=nli_model_id.strip() or DEFAULT_NLI_MODEL_ID,
    )
    return render_nli_attribution(result), attribution_rows(result)
