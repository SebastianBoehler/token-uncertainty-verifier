from __future__ import annotations

import argparse
import csv
import os
from html import escape
from pathlib import Path
import sys

os.environ.setdefault("TOKEN_UV_DEVICE", "cpu")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import set_seed

from token_uncertainty.model_runner import analyze_existing_text
from token_uncertainty.rendering import (
    render_sentence_overlay,
    render_token_overlay,
    sentence_rows,
    token_rows,
)

SAMPLE_CONTEXT = "Assess this answer for statements that deserve verification."
SAMPLE_TEXT = (
    "Apollo 11 landed on the Moon in 1969. "
    "Apollo 11 landed on the Moon in 1972. "
    "The Eiffel Tower opened in 1889. "
    "The Eiffel Tower opened in 1899. "
    "OpenAI released GPT-4 in 2023. "
    "OpenAI released GPT-4 in 2021. "
    "Project Atlas released build ATLAS-42 in report 14. "
    "The summary discussed product quality and user feedback."
)


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def write_report(path: Path, text: str, token_html: str, sentence_html: str) -> None:
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Token Uncertainty Sample</title>",
                "<style>body{max-width:1040px;margin:40px auto;font:16px system-ui;color:#111}</style>",
                "</head><body>",
                "<h1>Token Uncertainty Sample</h1>",
                "<h2>Analyzed Text</h2>",
                f"<pre>{escape(text)}</pre>",
                "<h2>Token Overlay</h2>",
                token_html,
                "<h2>Sentence Overlay</h2>",
                sentence_html,
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    result = analyze_existing_text(
        SAMPLE_TEXT,
        context=SAMPLE_CONTEXT,
        model_id=args.model_id,
    )

    samples_dir = ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)
    write_report(
        samples_dir / "sample_report.html",
        result.text,
        render_token_overlay(result.tokens),
        render_sentence_overlay(result.sentences),
    )
    write_csv(
        samples_dir / "sample_tokens.csv",
        ["#", "token", "probability", "uncertainty", "factual_risk", "rank", "margin", "claim_cues"],
        token_rows(result.tokens),
    )
    write_csv(
        samples_dir / "sample_sentences.csv",
        ["sentence", "text", "factual_risk", "mean_uncertainty", "max_token_risk", "claim_cues"],
        sentence_rows(result.sentences),
    )
    print(f"Wrote samples to {samples_dir}")


if __name__ == "__main__":
    main()
