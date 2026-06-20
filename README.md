---
title: Token Uncertainty Verifier
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
license: mit
python_version: "3.11"
---

# Token Uncertainty Verifier

[![CI](https://github.com/SebastianBoehler/token-uncertainty-verifier/actions/workflows/ci.yml/badge.svg)](https://github.com/SebastianBoehler/token-uncertainty-verifier/actions/workflows/ci.yml)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/sebastianboehler/token-uncertainty-verifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Token-level uncertainty and factual-risk overlays for LLM outputs, built as a small Gradio app and Hugging Face Space.

The project is designed for fast demos, hackathons, and early research workflows where you want to see which exact tokens or spans in an answer deserve verification. It separates model uncertainty from factual-risk triage so low probability is not treated as proof of hallucination.

## Demo

- Hugging Face Space: <https://huggingface.co/spaces/sebastianboehler/token-uncertainty-verifier>
- Sample HTML report: [`samples/sample_report.html`](samples/sample_report.html)
- Token score CSV: [`samples/sample_tokens.csv`](samples/sample_tokens.csv)
- Sentence score CSV: [`samples/sample_sentences.csv`](samples/sample_sentences.csv)

## What It Shows

The app exposes two layers:

- **Token uncertainty**: chosen-token probability, normalized entropy, rank, and margin from a causal language model.
- **Factual-risk triage**: local claim cues such as dates, citations, case names, legal sections, numbers, named entities, and repeated-claim conflicts.

The most useful mode is pasted-text analysis. You can paste an answer from another model and use a local model to score where the text looks claim-heavy, unstable, or internally inconsistent.

## What It Does Not Do

This is not a fact checker. It does not prove that a highlighted claim is false.

Current risk scores are triage signals. They answer: “Which exact tokens should a human or retrieval system verify first?” A production verifier should add retrieval, source comparison, and domain-specific evidence checks.

## Features

- Gradio UI with generation and pasted-text analysis modes.
- Token overlay with hover tooltips for probability, uncertainty, risk, rank, margin, and cues.
- Sentence-level summary table for scanning high-risk statements.
- Conflict flags for repeated case claims with inconsistent dates or citations.
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

## Generate Samples

```bash
python scripts/generate_samples.py --model-id sshleifer/tiny-gpt2
```

This writes:

- `samples/sample_report.html`
- `samples/sample_tokens.csv`
- `samples/sample_sentences.csv`

The sample text intentionally includes contradictory legal variants so the overlay has clear token-level flags.

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
  app_ui.py        Gradio layout and event wiring
  model_runner.py  Transformers model loading, generation, and text scoring
  scoring.py       Token-local cue assignment and risk scoring
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
- Add side-by-side answer comparison with token-level disagreement highlights.
- Add calibrated thresholds per domain and model family.
- Export overlays as standalone HTML snippets for reports.

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the local workflow and project constraints.

## License

MIT. See [`LICENSE`](LICENSE).
