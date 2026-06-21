---
title: LLM Uncertainty Attribution Lab
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
license: mit
python_version: "3.11"
---

# LLM Uncertainty Attribution Lab

[![CI](https://github.com/SebastianBoehler/token-uncertainty-verifier/actions/workflows/ci.yml/badge.svg)](https://github.com/SebastianBoehler/token-uncertainty-verifier/actions/workflows/ci.yml)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/sebastianboehler/token-uncertainty-verifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Chat-time token uncertainty, text-diff, contrastive-likelihood, and NLI span-attribution overlays for LLM outputs, built as a small Gradio app and Hugging Face Space.

The project is designed for fast demos, hackathons, and early research workflows where you want to inspect factual-looking answers without pretending that model scores are fact checks. It keeps model uncertainty, textual change detection, model-likelihood comparison, and NLI disagreement localization separate from factual truth.

## Demo

- Hugging Face Space: <https://huggingface.co/spaces/sebastianboehler/token-uncertainty-verifier>
- Sample HTML report: [`samples/sample_report.html`](samples/sample_report.html)
- Diff and contrastive sample: [`samples/sample_comparison_modes.html`](samples/sample_comparison_modes.html)
- Token score CSV: [`samples/sample_tokens.csv`](samples/sample_tokens.csv)
- Sentence score CSV: [`samples/sample_sentences.csv`](samples/sample_sentences.csv)
- Contrastive score CSV: [`samples/sample_contrastive.csv`](samples/sample_contrastive.csv)

## What It Shows

The app exposes several separate workflows:

- **Chat + Overlay**: a normal chat flow where each assistant reply is rendered directly as an NLI span overlay against required reference/evidence. User messages can be overlaid too when you want to inspect a typed factual claim.
- **Uncertainty overlay**: chosen-token probability, normalized entropy, rank, and margin from the scoring model's next-token distribution.
- **Diff mode**: deterministic word-level changes between a reference text and a candidate text.
- **Contrastive scoring**: relative model likelihood for candidate spans in the same sentence template.
- **NLI span attribution**: black-box ablation over candidate spans using an entailment model, highlighting spans whose removal changes contradiction or entailment most.

The most useful mode is pasted-text analysis. You can paste an answer from another model and use a local model to score where that exact text was easy or hard for the scoring model to predict.

## Interpreting Uncertainty

Uncertainty is not factual truth. It reflects what the local scoring model assigns to the next-token distribution after pretraining and post-training: probability mass, entropy, rank, and margin under the provided context. The model can be uncertain about a true statement, confident about a false statement, or unaware of facts that changed after its training unless those facts are provided in context.

The uncertainty overlay answers: "Which exact words were surprising to this scoring model?" It does not answer: "Which claims are true?"

Chat mode makes that signal easier to experience: choose a prefilled reference/candidate example or paste trusted reference text, then inspect the highlighted bubbles in the conversation. Chat requires reference/evidence and uses NLI shortening only, so it matches the HTML-style attribution demo instead of silently falling back to token uncertainty. Chat bubbles stay compact, with detailed scores in hover text or the dedicated analysis tabs.

Diff mode can expose a changed span like `1969 -> 1972`, but it cannot decide which value is correct.

Contrastive scoring can say whether the scoring model prefers `1969`, `1972`, or another alternative in the same template. The preferred option is still not guaranteed to be true.

NLI span attribution can localize semantic disagreement, for example showing that `Berlin` or `Elon Musk` explains why a paraphrased candidate contradicts a reference. It still does not know which side is true unless the reference is trusted or backed by retrieval.

## Example Comparisons

The Gradio app includes an **Examples** tab with a head-to-head overlay grid for four authored scenarios:

- **Reference wording**: a plausible reference version with no truth label assigned by the app.
- **Year-only error**: one value is changed relative to the reference.
- **Contradictory variants**: repeated statements contain incompatible years.
- **False side sentence**: a side sentence is inserted into otherwise similar wording.

The scenario notes are authored example labels. They are not produced by regex or keyword scoring.

## What It Does Not Do

This is not a fact checker. It does not prove that a highlighted claim is false.

Current scores are uncertainty and comparison signals, not factual grounding indicators. A production verifier should add retrieval, source comparison, and domain-specific evidence checks before making truth claims.

## Features

- Gradio UI with generation and pasted-text analysis modes.
- Chat mode that renders assistant turns, and optionally user turns, as inline NLI span overlays against required reference/evidence.
- Word-level overlay that aggregates token pieces and keeps hover tooltips for uncertainty.
- Raw token score table and CSV for probability, uncertainty, rank, and margin.
- Sentence-level summary table for scanning high-uncertainty statements.
- Head-to-head example comparison with authored scenario notes and compact hot-span summaries.
- Diff mode for direct reference-vs-candidate changes.
- Contrastive mode for scoring alternatives in a shared template.
- NLI span attribution with focused word overlays and raw impact/focus tables.
- Sample report generator for reproducible screenshots and CSV artifacts.
- GitHub Actions CI for tests and smoke checks.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7860>.

Override the default model:

```bash
MODEL_ID=HuggingFaceTB/SmolLM2-135M-Instruct python app.py
```

Force CPU execution:

```bash
TOKEN_UV_DEVICE=cpu python app.py
```

Override the NLI attribution model:

```bash
NLI_MODEL_ID=cross-encoder/nli-deberta-v3-small python app.py
```

## Generate Samples

```bash
python scripts/generate_samples.py --model-id sshleifer/tiny-gpt2
```

This writes multi-scenario comparison outputs:

- `samples/sample_report.html`
- `samples/sample_comparison_modes.html`
- `samples/sample_tokens.csv`
- `samples/sample_sentences.csv`
- `samples/sample_contrastive.csv`

The sample report renders those scenarios head-to-head so the token and sentence overlays can be compared directly. The goal is to show what distribution uncertainty can and cannot reveal without retrieval.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Smoke-test the app import:

```bash
python - <<'PY'
from token_uncertainty.app_ui import create_demo
print(type(create_demo()).__name__)
PY
```

## Architecture

```text
app.py
token_uncertainty/
  app_actions.py   Gradio event handlers
  app_ui.py        Gradio layout and event wiring
  comparison_modes.py Diff and contrastive rendering
  model_runner.py  Transformers model loading, generation, and text scoring
  nli_attribution.py Entailment scoring and span attribution
  scoring.py       Distribution-only uncertainty scoring
  rendering.py     HTML overlays and score tables
  types.py         Shared dataclasses
scripts/
  generate_samples.py
tests/
```

## Deployment

The README front matter is compatible with Hugging Face Spaces.

```bash
hf repo create sebastianboehler/token-uncertainty-verifier --repo-type space --space-sdk gradio --exist-ok
hf upload sebastianboehler/token-uncertainty-verifier . . --repo-type space
```

## Roadmap

- Add retrieval-backed verification against cited sources.
- Add source-backed span labels that distinguish unsupported, contradicted, and supported text.
- Add semantic-entropy sampling and clustering as a separate answer-level mode.
- Add calibrated thresholds per domain and model family.
- Export overlays as standalone HTML snippets for reports.

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the local workflow and project constraints.

## License

MIT. See [`LICENSE`](LICENSE).
