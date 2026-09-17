"""Evaluate the custom checkpoints for the Hugging Face model card."""

import argparse
import json
import math
import os
import random

import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import GPT_CONFIG_124M
from main import clean_response, generate_text, load_model


MODEL_CONFIG = {**GPT_CONFIG_124M, "qkv_bias": True, "context_length": 1024}


def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def load_test_data(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    random.Random(42).shuffle(data)
    train_end = int(len(data) * 0.85)
    test_end = train_end + int(len(data) * 0.10)
    return data[train_end:test_end]


def response_loss(model, entries, tokenizer, device, max_length):
    total_loss = 0.0
    total_tokens = 0

    for entry in tqdm(entries, desc="Scoring responses", unit="example"):
        prompt_ids = tokenizer.encode(format_input(entry) + "\n\n### Response:\n")
        response_ids = tokenizer.encode(
            entry["output"], allowed_special={"<|endoftext|>"}
        )
        response_ids += tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )
        token_ids = (prompt_ids + response_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]

        if len(token_ids) < 2:
            continue

        input_ids = torch.tensor(token_ids[:-1], device=device).unsqueeze(0)
        target_ids = torch.tensor(labels[1:], device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.flatten(0, 1), target_ids.flatten(),
                ignore_index=-100, reduction="sum",
            )
        token_count = int((target_ids != -100).sum().item())
        total_loss += loss.item()
        total_tokens += token_count

    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss), total_tokens


def generate_samples(model, entries, tokenizer, device, max_new_tokens):
    samples = []
    for entry in tqdm(entries, desc="Generating samples", unit="sample"):
        prompt = format_input(entry) + "\n\n### Response:\n"
        generated = generate_text(
            model=model,
            prompt=prompt,
            tokenizer=tokenizer,
            device=device,
            config=MODEL_CONFIG,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            repetition_penalty=1.1,
        )
        response = generated.split("### Response:", 1)[-1]
        samples.append(
            {
                "instruction": entry["instruction"],
                "input": entry["input"],
                "reference": entry["output"],
                "generated": clean_response(response),
            }
        )
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="alpaca-cleaned.json")
    parser.add_argument("--output", default="benchmark-results.json")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = tiktoken.get_encoding("gpt2")
    test_data = load_test_data(args.data)[:args.max_examples]
    results = {
        "dataset": args.data,
        "split": "deterministic 10% test split after random.Random(42) shuffle",
        "examples": len(test_data),
        "max_length": args.max_length,
        "models": {},
    }

    for checkpoint in ("model.pth", "model-finetuned.pth"):
        if not os.path.exists(checkpoint):
            continue
        model = load_model(checkpoint, MODEL_CONFIG, device)
        loss, perplexity, token_count = response_loss(
            model, test_data, tokenizer, device, args.max_length
        )
        results["models"][checkpoint] = {
            "response_loss": loss,
            "response_perplexity": perplexity,
            "response_tokens": token_count,
            "samples": generate_samples(
                model, test_data[:args.sample_count], tokenizer,
                device, args.max_new_tokens,
            ),
        }
        del model

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    for checkpoint, metrics in results["models"].items():
        print(
            f"{checkpoint}: loss={metrics['response_loss']:.4f}, "
            f"perplexity={metrics['response_perplexity']:.2f}, "
            f"tokens={metrics['response_tokens']}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()