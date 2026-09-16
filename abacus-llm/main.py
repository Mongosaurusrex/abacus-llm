"""Interactive inference script for asking questions to a finetuned model.

Usage:
    PYTHONPATH=abacus-llm python abacus-llm/main.py
"""

import argparse
import os

import tiktoken
import torch

from config import GPT_CONFIG_124M
from model.gpt import GPTModel
from utils import text_to_token_ids, token_ids_to_text


DEFAULT_MODEL_PATH = "model-finetuned.pth"
FALLBACK_MODEL_PATH = "model.pth"
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 0
DEFAULT_REPETITION_PENALTY = 1.1
EOS_TOKEN_ID = 50256

# Must match the config used when the checkpoint was saved.
LOAD_CONFIG = {**GPT_CONFIG_124M, "qkv_bias": True, "context_length": 1024}


def load_model(path: str, config: dict, device: torch.device) -> GPTModel:
    model = GPTModel(config)
    state_dict = torch.load(path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_instruction_prompt(question: str, input_text: str = "") -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{question}"
    )
    input_block = f"\n\n### Input:\n{input_text}" if input_text else ""
    return f"{instruction_text}{input_block}\n\n### Response:\n"


def generate_text(
    model,
    prompt: str,
    tokenizer,
    device: torch.device,
    config: dict,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float = 1.15,
) -> str:
    idx = text_to_token_ids(prompt, tokenizer).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config["context_length"]:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        # Apply repetition penalty to already generated token logits
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                top_logits, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < top_logits[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, next_token), dim=1)
        if next_token.item() == EOS_TOKEN_ID:
            break

    return token_ids_to_text(idx, tokenizer)


def clean_response(text: str) -> str:
    # Stop output at common hallucinated / subsequent delimiters
    stop_delimiters = [
        "<|endoftext|>",
        "### Instruction:",
        "### Instruction",
        "\n###",
        "\nQ:",
        "\nQuestion:",
    ]
    cleaned = text
    for delim in stop_delimiters:
        if delim in cleaned:
            cleaned = cleaned.split(delim, 1)[0]
    return cleaned.strip()


def ask_model(
    model,
    tokenizer,
    device: torch.device,
    question: str,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float = 1.15,
) -> str:
    prompt = build_instruction_prompt(question, input_text)
    full_text = generate_text(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        device=device,
        config=LOAD_CONFIG,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    if "### Response:" in full_text:
        response_part = full_text.split("### Response:", 1)[1]
    else:
        response_part = full_text[len(prompt):]

    return clean_response(response_part)


def resolve_model_path(explicit_model_path: str | None) -> str:
    if explicit_model_path:
        return explicit_model_path
    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    return FALLBACK_MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Ask questions to your finetuned GPT model.",
    )
    parser.add_argument("--model-path", default=None, help="Path to .pth model checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY, help="Repetition penalty (1.0 to disable)")
    parser.add_argument("--prompt", default=None, help="Optional one-shot question")
    parser.add_argument("--input", default="", help="Optional input block for instruction prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = resolve_model_path(args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{model_path}'. "
            f"Run finetuning first to produce '{DEFAULT_MODEL_PATH}'."
        )

    print(f"Loading model from {model_path} on {device} ...")
    model = load_model(model_path, LOAD_CONFIG, device)

    if args.prompt:
        answer = ask_model(
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=args.prompt,
            input_text=args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(answer)
        return

    print("Model ready. Ask a question. Type 'quit' to exit.")
    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        answer = ask_model(
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=question,
            input_text="",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
