from __future__ import annotations

import gradio as gr

from token_uncertainty.app_actions import (
    example_note,
    run_contrastive_mode,
    run_diff_mode,
    run_example,
    run_example_comparison,
    run_existing_text,
    run_generation,
)
from token_uncertainty.examples import (
    DEFAULT_CANDIDATE_TEXT,
    DEFAULT_CONTRASTIVE_OPTIONS,
    DEFAULT_CONTRASTIVE_TEMPLATE,
    DEFAULT_CONTEXT,
    DEFAULT_REFERENCE_TEXT,
    combined_example_text,
    example_labels,
)
from token_uncertainty.model_runner import DEFAULT_MODEL_ID

SENTENCE_HEADERS = [
    "sentence",
    "text",
    "uncertainty_score",
    "mean_uncertainty",
    "max_token_uncertainty",
]

TOKEN_HEADERS = [
    "#",
    "token",
    "probability",
    "uncertainty",
    "rank",
    "margin",
]

CONTRASTIVE_HEADERS = [
    "#",
    "option",
    "token_pieces",
    "geomean_probability",
    "mean_log_probability",
    "relative_weight",
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
    "rank, and margin. It is not factual truth, and it does not use regex or keyword "
    "labels. Diff mode shows textual changes; contrastive mode compares model likelihood. "
    "None of these modes prove factual correctness."
)


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Token Uncertainty Verifier") as demo:
        gr.Markdown(
            "# Token Uncertainty Verifier\n"
            "Token-level model uncertainty for factual-looking answers.\n\n"
            f"{UNCERTAINTY_NOTE}"
        )

        with gr.Row():
            model_id = gr.Textbox(value=DEFAULT_MODEL_ID, label="Model ID", scale=2)
            threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.82,
                step=0.01,
                label="Highlight threshold",
                info="Higher values make the overlay more selective.",
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

            with gr.Tab("Diff Mode"):
                with gr.Row():
                    diff_reference = gr.Textbox(
                        value=DEFAULT_REFERENCE_TEXT,
                        label="Reference",
                        lines=5,
                    )
                    diff_candidate = gr.Textbox(
                        value=DEFAULT_CANDIDATE_TEXT,
                        label="Candidate",
                        lines=5,
                    )
                diff_button = gr.Button("Compare text changes", variant="primary")
                diff_html = gr.HTML(label="Word-level diff")

            with gr.Tab("Contrastive"):
                contrastive_context = gr.Textbox(
                    value=DEFAULT_CONTEXT,
                    label="Context",
                    lines=2,
                )
                contrastive_template = gr.Textbox(
                    value=DEFAULT_CONTRASTIVE_TEMPLATE,
                    label="Template with {answer}",
                    lines=2,
                )
                contrastive_options = gr.Textbox(
                    value="\n".join(DEFAULT_CONTRASTIVE_OPTIONS),
                    label="Alternatives, one per line",
                    lines=4,
                )
                contrastive_button = gr.Button("Score alternatives", variant="primary")
                contrastive_html = gr.HTML(label="Contrastive likelihood")
                contrastive_table = gr.Dataframe(
                    headers=CONTRASTIVE_HEADERS,
                    label="Alternative scores",
                )

        generated_text = gr.Textbox(label="Analyzed text", lines=8)
        with gr.Tabs():
            with gr.Tab("Word Overlay"):
                token_html = gr.HTML(label="Word overlay")
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
        diff_button.click(
            run_diff_mode,
            [diff_reference, diff_candidate],
            diff_html,
        )
        contrastive_button.click(
            run_contrastive_mode,
            [contrastive_template, contrastive_options, contrastive_context, model_id],
            [contrastive_html, contrastive_table],
        )

    return demo
