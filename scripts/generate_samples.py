from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("TOKEN_UV_DEVICE", "cpu")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import set_seed

from token_uncertainty.examples import EXAMPLE_CASES, ExampleCase
from token_uncertainty.model_runner import analyze_existing_text
from token_uncertainty.rendering import (
    ComparisonSection,
    grouped_hot_spans,
    render_comparison_grid,
    render_sentence_overlay,
    render_token_overlay,
    sentence_rows,
    token_rows,
)
from token_uncertainty.types import AnalysisResult


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def write_report(path: Path, analyses: list[tuple[ExampleCase, AnalysisResult, str, str]]) -> None:
    sections = [
        ComparisonSection(
            label=case.label,
            note=case.note,
            focus=case.focus,
            hot_spans=grouped_hot_spans(result.tokens, 0.82),
            token_html=token_html,
            sentence_html=sentence_html,
        )
        for case, result, token_html, sentence_html in analyses
    ]
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Token Uncertainty Sample</title>",
                "<style>body{max-width:1280px;margin:40px auto;font:16px system-ui;color:#111}</style>",
                "</head><body>",
                "<h1>Token Uncertainty Head-to-Head Comparisons</h1>",
                "<p>These examples show model-distribution uncertainty, not proof of truth.</p>",
                render_comparison_grid(sections),
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
    analyses = []
    token_csv_rows: list[list[object]] = []
    sentence_csv_rows: list[list[object]] = []
    for case in EXAMPLE_CASES:
        result = analyze_existing_text(case.text, context=case.context, model_id=args.model_id)
        token_html = render_token_overlay(result.tokens)
        sentence_html = render_sentence_overlay(result.sentences)
        analyses.append((case, result, token_html, sentence_html))
        token_csv_rows.extend([[case.label, *row] for row in token_rows(result.tokens)])
        sentence_csv_rows.extend([[case.label, *row] for row in sentence_rows(result.sentences)])

    samples_dir = ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)
    write_report(
        samples_dir / "sample_report.html",
        analyses,
    )
    write_csv(
        samples_dir / "sample_tokens.csv",
        ["example", "#", "token", "probability", "uncertainty", "rank", "margin"],
        token_csv_rows,
    )
    write_csv(
        samples_dir / "sample_sentences.csv",
        ["example", "sentence", "text", "uncertainty_score", "mean_uncertainty", "max_token_uncertainty"],
        sentence_csv_rows,
    )
    print(f"Wrote samples to {samples_dir}")


if __name__ == "__main__":
    main()
