from __future__ import annotations

import gradio as gr

from token_uncertainty.app_actions import (
    clear_chat,
    example_note,
    run_contrastive_mode,
    run_diff_mode,
    run_example,
    run_example_comparison,
    run_existing_text,
    run_generation,
    run_chat_turn,
    run_nli_attribution,
)
from token_uncertainty.examples import (
    DEFAULT_CANDIDATE_TEXT,
    DEFAULT_CONTRASTIVE_OPTIONS,
    DEFAULT_CONTRASTIVE_TEMPLATE,
    DEFAULT_CONTEXT,
    DEFAULT_NLI_CANDIDATE_TEXT,
    DEFAULT_NLI_REFERENCE_TEXT,
    DEFAULT_REFERENCE_TEXT,
    combined_example_text,
    example_labels,
)
from token_uncertainty.model_runner import DEFAULT_MODEL_ID
from token_uncertainty.nli_attribution import DEFAULT_NLI_MODEL_ID

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

NLI_HEADERS = [
    "span",
    "removed_text",
    "impact",
    "focus",
    "contradiction_after",
    "entailment_after",
    "neutral_after",
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
    "labels. Diff mode shows textual changes; contrastive mode compares model likelihood; "
    "NLI span attribution localizes semantic disagreement. None of these modes prove "
    "factual correctness."
)


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="LLM Uncertainty Attribution Lab") as demo:
        gr.Markdown(
            "# LLM Uncertainty Attribution Lab\n"
            "Token uncertainty, contrastive likelihood, and NLI span attribution for factual-looking answers.\n\n"
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
            with gr.Tab("Chat + Overlay"):
                gr.Markdown(
                    "Chat with the selected model. Each assistant reply is rendered as an inline overlay. "
                    "With reference/evidence, the bubble uses NLI span shortening; without it, the bubble "
                    "uses token uncertainty only. Highlights are not proof of factual error."
                )
                chat_state = gr.State([])
                chat_display_state = gr.State([])
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=420,
                    placeholder="Ask a factual question and inspect the highlighted assistant bubble.",
                    sanitize_html=False,
                )
                chat_reference = gr.Textbox(
                    label="Reference/evidence for NLI shortening (optional)",
                    lines=3,
                    placeholder=(
                        "Paste trusted evidence to localize disagreement. Example: Angela Merkel was "
                        "born on July 17, 1954, in Hamburg, Germany."
                    ),
                )
                chat_overlay_user = gr.Checkbox(
                    value=False,
                    label="Overlay user messages too",
                    info="Best for user-entered factual claims. Questions can produce noisy NLI or token scores.",
                )
                chat_message = gr.Textbox(
                    label="Message",
                    lines=3,
                    placeholder="Ask something with specific dates, entities, or claims.",
                )
                with gr.Row():
                    chat_max_new_tokens = gr.Slider(8, 192, value=96, step=1, label="Max tokens")
                    chat_temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                    chat_top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
                with gr.Row():
                    chat_button = gr.Button("Send and analyze", variant="primary")
                    clear_chat_button = gr.Button("Clear chat", variant="secondary")
                chat_nli_html = gr.HTML(label="Latest reply NLI span detail", visible=False)

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

            with gr.Tab("NLI Span Attribution"):
                gr.Markdown(
                    "Compare a reference and candidate. The NLI model scores entailment, "
                    "then candidate spans are removed to see which removal changes the NLI decision most."
                )
                with gr.Row():
                    nli_reference = gr.Textbox(
                        value=DEFAULT_NLI_REFERENCE_TEXT,
                        label="Reference",
                        lines=5,
                    )
                    nli_candidate = gr.Textbox(
                        value=DEFAULT_NLI_CANDIDATE_TEXT,
                        label="Candidate",
                        lines=5,
                    )
                nli_model = gr.Textbox(value=DEFAULT_NLI_MODEL_ID, label="NLI model ID")
                nli_button = gr.Button("Localize NLI disagreement", variant="primary")
                nli_html = gr.HTML(label="NLI span attribution")
                nli_table = gr.Dataframe(headers=NLI_HEADERS, label="Attribution rows")

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
        chat_inputs = [
            chat_message,
            chat_state,
            chat_display_state,
            model_id,
            chat_max_new_tokens,
            chat_temperature,
            chat_top_p,
            threshold,
            chat_reference,
            chat_overlay_user,
        ]
        chat_outputs = [chat_message, chatbot, chat_state, chat_display_state, chat_nli_html, *outputs]
        chat_button.click(run_chat_turn, chat_inputs, chat_outputs)
        chat_message.submit(run_chat_turn, chat_inputs, chat_outputs)
        clear_chat_button.click(clear_chat, outputs=chat_outputs)

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
        nli_button.click(
            run_nli_attribution,
            [nli_reference, nli_candidate, nli_model],
            [nli_html, nli_table],
        )

    return demo
