from __future__ import annotations

import gradio as gr

from token_uncertainty.examples import (
    EXAMPLE_CASES,
    DEFAULT_CONTEXT,
    combined_example_text,
    example_labels,
    get_example,
)
from token_uncertainty.model_runner import DEFAULT_MODEL_ID, analyze_existing_text, generate_with_scores
from token_uncertainty.rendering import (
    render_comparison_grid,
    render_sentence_overlay,
    render_token_overlay,
    sentence_rows,
    token_rows,
)

SENTENCE_HEADERS = [
    "sentence",
    "text",
    "factual_risk",
    "mean_uncertainty",
    "max_token_risk",
    "claim_cues",
]

TOKEN_HEADERS = [
    "#",
    "token",
    "probability",
    "uncertainty",
    "factual_risk",
    "rank",
    "margin",
    "claim_cues",
]

DEFAULT_PROMPT = (
    "List three specific scientific or historical claims with dates and sources "
    "in one sentence each."
)

DEFAULT_TEXT = (
    combined_example_text()
)

UNCERTAINTY_NOTE = (
    "Uncertainty here is a model-distribution signal from token probability, entropy, "
    "rank, and margin. It is not factual truth. Factual risk combines that signal with "
    "claim cues and internal conflicts to decide what should be verified first."
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
            (
                case.label,
                case.note,
                render_token_overlay(result.tokens, risk_threshold),
                render_sentence_overlay(result.sentences),
            )
        )
    return render_comparison_grid(sections)


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Token Uncertainty Verifier") as demo:
        gr.Markdown(
            "# Token Uncertainty Verifier\n"
            "Token-level uncertainty plus claim-risk triage for factual answers.\n\n"
            f"{UNCERTAINTY_NOTE}"
        )

        with gr.Row():
            model_id = gr.Textbox(value=DEFAULT_MODEL_ID, label="Model ID", scale=2)
            threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.45,
                step=0.01,
                label="Verify threshold",
                scale=1,
            )

        with gr.Tabs():
            with gr.Tab("Generate"):
                prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt", lines=4)
                with gr.Row():
                    max_new_tokens = gr.Slider(8, 192, value=96, step=1, label="Max tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
                generate_button = gr.Button("Generate overlay", variant="primary")

            with gr.Tab("Analyze Text"):
                context = gr.Textbox(value=DEFAULT_CONTEXT, label="Context", lines=3)
                existing_text = gr.Textbox(value=DEFAULT_TEXT, label="Text", lines=7)
                analyze_button = gr.Button("Analyze overlay", variant="primary")

            with gr.Tab("Examples"):
                example_choice = gr.Radio(
                    choices=example_labels(),
                    value=example_labels()[0],
                    label="Scenario",
                )
                example_details = gr.Markdown(value=example_note(example_labels()[0]))
                with gr.Row():
                    example_button = gr.Button("Analyze selected", variant="secondary")
                    compare_button = gr.Button("Compare all head-to-head", variant="primary")
                comparison_html = gr.HTML(label="Head-to-head overlays")

        generated_text = gr.Textbox(label="Analyzed text", lines=8)
        with gr.Tabs():
            with gr.Tab("Token Overlay"):
                token_html = gr.HTML(label="Token overlay")
            with gr.Tab("Sentence Overlay"):
                sentence_html = gr.HTML(label="Sentence overlay")
            with gr.Tab("Sentence Scores"):
                sentence_table = gr.Dataframe(headers=SENTENCE_HEADERS, label="Sentences")
            with gr.Tab("Token Scores"):
                token_table = gr.Dataframe(headers=TOKEN_HEADERS, label="Tokens")

        generation_inputs = [prompt, model_id, max_new_tokens, temperature, top_p, threshold]
        outputs = [generated_text, token_html, sentence_html, sentence_table, token_table]
        generate_button.click(run_generation, generation_inputs, outputs)
        prompt.submit(run_generation, generation_inputs, outputs)

        analyze_button.click(
            run_existing_text,
            [existing_text, context, model_id, threshold],
            outputs,
        )
        example_choice.change(example_note, example_choice, example_details)
        example_button.click(
            run_example,
            [example_choice, model_id, threshold],
            [context, existing_text, example_details, *outputs],
        )
        compare_button.click(
            run_example_comparison,
            [model_id, threshold],
            comparison_html,
        )

    return demo
