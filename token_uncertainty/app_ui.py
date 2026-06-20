from __future__ import annotations

import gradio as gr

from token_uncertainty.model_runner import DEFAULT_MODEL_ID, analyze_existing_text, generate_with_scores
from token_uncertainty.rendering import (
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
    "List three specific legal precedents from 1897 and explain their holdings "
    "in one sentence each."
)

DEFAULT_TEXT = (
    "In 1897, the Supreme Court decided Hawkins v. McGee and held that "
    "contract damages must equal the promised value. The German Civil Code "
    "entered into force in 1900."
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


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Token Uncertainty Verifier") as demo:
        gr.Markdown(
            "# Token Uncertainty Verifier\n"
            "Token-level uncertainty plus claim-risk triage for legal and factual answers."
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
                context = gr.Textbox(label="Context", lines=3)
                existing_text = gr.Textbox(value=DEFAULT_TEXT, label="Text", lines=7)
                analyze_button = gr.Button("Analyze overlay", variant="primary")

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

    return demo
