---
title: Token Uncertainty Verifier
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
license: mit
python_version: "3.11"
---

# Token Uncertainty Verifier

Hackathon prototype for visualizing token-level model uncertainty and claim-level factual risk.

The app exposes two separate signals:

- **Token uncertainty**: chosen-token probability, entropy, rank, and margin from a causal language model.
- **Factual-risk triage**: claim-like cues such as dates, citations, legal references, numbers, and named entities combined with uncertainty.

The factual-risk score is not a fact checker. It is a triage overlay for deciding which words or sentences deserve verification.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Optional model override:

```bash
MODEL_ID=HuggingFaceTB/SmolLM2-135M-Instruct python app.py
```

## Generate sample artifacts

```bash
python scripts/generate_samples.py --model-id sshleifer/tiny-gpt2
```

Outputs are written to `samples/`.

## Deploy as a Hugging Face Space

Create a Gradio Space and upload the repository:

```bash
hf repo create token-uncertainty-verifier --repo-type space --space_sdk gradio --exist-ok
hf upload sebastianboehler/token-uncertainty-verifier . --repo-type space
```

If your local `hf` CLI does not support `upload`, push with git to the Space remote shown by Hugging Face.
