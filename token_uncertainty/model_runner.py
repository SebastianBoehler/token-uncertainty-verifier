from __future__ import annotations

import os
from functools import lru_cache
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_uncertainty.scoring import analyze_scored_tokens, normalized_entropy
from token_uncertainty.types import AnalysisResult, ContrastiveOption, TokenScore

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "HuggingFaceTB/SmolLM2-135M-Instruct")


def preferred_device() -> str:
    override = os.getenv("TOKEN_UV_DEVICE")
    if override in {"cpu", "cuda", "mps"}:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def load_model(model_id: str = DEFAULT_MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.to(preferred_device())
    model.eval()
    return tokenizer, model


def render_chat_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def score_from_logits(logits: torch.Tensor, token_id: int) -> tuple[float, float, int, float]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    chosen_probability = float(probabilities[token_id].item())
    top_probs, top_ids = torch.topk(probabilities, k=min(2, probabilities.numel()))

    if int(top_ids[0].item()) == token_id and len(top_probs) > 1:
        alternative = float(top_probs[1].item())
    else:
        alternative = float(top_probs[0].item())

    rank = int(torch.sum(probabilities > chosen_probability).item()) + 1
    return chosen_probability, normalized_entropy(probabilities), rank, chosen_probability - alternative


def generate_with_scores(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> AnalysisResult:
    tokenizer, model = load_model(model_id)
    device = next(model.parameters()).device
    rendered_prompt = render_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)

    do_sample = temperature > 0.0
    generate_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "return_dict_in_generate": True,
        "output_scores": True,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = outputs.sequences[0][prompt_length:]
    token_scores: list[TokenScore] = []

    for token_id_tensor, logits in zip(generated_ids, outputs.scores, strict=False):
        token_id = int(token_id_tensor.item())
        probability, entropy, rank, margin = score_from_logits(logits[0], token_id)
        token_scores.append(
            TokenScore(
                text=tokenizer.decode([token_id]),
                token_id=token_id,
                probability=probability,
                entropy=entropy,
                rank=rank,
                margin=margin,
            )
        )

    return analyze_scored_tokens(token_scores)


def analyze_existing_text(
    text: str,
    context: str = "",
    model_id: str = DEFAULT_MODEL_ID,
) -> AnalysisResult:
    tokenizer, model = load_model(model_id)
    device = next(model.parameters()).device
    prefix_ids = tokenizer.encode(context, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    if not text_ids:
        return analyze_scored_tokens([])

    bos = []
    if not prefix_ids and tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]

    all_ids = bos + prefix_ids + text_ids
    target_start = len(bos) + len(prefix_ids)
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]

    token_scores: list[TokenScore] = []
    for position in range(target_start, len(all_ids)):
        token_id = all_ids[position]
        if position == 0:
            token_scores.append(TokenScore(text=tokenizer.decode([token_id]), token_id=token_id))
            continue
        probability, entropy, rank, margin = score_from_logits(logits[position - 1], token_id)
        token_scores.append(
            TokenScore(
                text=tokenizer.decode([token_id]),
                token_id=token_id,
                probability=probability,
                entropy=entropy,
                rank=rank,
                margin=margin,
            )
        )

    return analyze_scored_tokens(token_scores)


def score_contrastive_options(
    template: str,
    options: list[str],
    context: str = "",
    model_id: str = DEFAULT_MODEL_ID,
) -> list[ContrastiveOption]:
    if "{answer}" not in template:
        raise ValueError("Template must contain {answer}.")

    prefix, _ = template.split("{answer}", 1)
    clean_options = [option.strip() for option in options if option.strip()]
    if not clean_options:
        return []

    tokenizer, model = load_model(model_id)
    device = next(model.parameters()).device
    prompt = f"{context.strip()}\n{prefix}" if context.strip() else prefix
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos = []
    if not prefix_ids and tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]

    scored = []
    for option in clean_options:
        option_ids = tokenizer.encode(option, add_special_tokens=False)
        if not option_ids:
            continue

        all_ids = bos + prefix_ids + option_ids
        target_start = len(bos) + len(prefix_ids)
        input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0]

        log_probabilities = []
        for position in range(target_start, len(all_ids)):
            probability, _, _, _ = score_from_logits(logits[position - 1], all_ids[position])
            log_probabilities.append(math.log(max(probability, 1e-12)))

        mean_log_probability = sum(log_probabilities) / len(log_probabilities)
        scored.append(
            ContrastiveOption(
                option=option,
                token_count=len(option_ids),
                mean_log_probability=mean_log_probability,
                geometric_mean_probability=math.exp(mean_log_probability),
            )
        )

    if not scored:
        return []

    best = max(item.mean_log_probability for item in scored)
    weights = [math.exp(item.mean_log_probability - best) for item in scored]
    total = sum(weights)
    for item, weight in zip(scored, weights, strict=False):
        item.relative_weight = weight / total if total else 0.0

    return sorted(scored, key=lambda item: item.mean_log_probability, reverse=True)
