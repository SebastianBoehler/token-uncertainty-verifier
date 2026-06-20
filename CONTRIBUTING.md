# Contributing

Thanks for improving Token Uncertainty Verifier.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run tests before opening a pull request:

```bash
pytest -q
```

For model-backed smoke checks:

```bash
TOKEN_UV_DEVICE=cpu python scripts/generate_samples.py --model-id sshleifer/tiny-gpt2
```

## Project Constraints

- Keep source files small and focused. Avoid files over roughly 300 lines.
- Keep risk scoring honest: uncertainty and factual correctness are separate signals.
- Do not add mock verification or fake source evidence.
- Prefer narrow tests for scoring, rendering, and legal-text edge cases.
- Keep Hugging Face Space metadata in `README.md` intact.

## Pull Requests

Good pull requests usually include:

- A clear description of the behavior change.
- Tests for scoring or rendering changes.
- Updated samples when the visual output changes.
- Notes about any model/runtime cost changes.
