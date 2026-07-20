"""
Load a saved model and run an interactive prompt loop.

Usage:
    PYTHONPATH=abacus-llm python abacus-llm/main.py
"""

import torch
import tiktoken

from config import GPT_CONFIG_124M
from model.gpt import GPTModel
from utils import text_to_token_ids, token_ids_to_text


MODEL_NAME = "model"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40

# Must match the config used when the model was saved
LOAD_CONFIG = {**GPT_CONFIG_124M, "qkv_bias": True, "context_length": 1024}


def load_model(path: str, config: dict, device: torch.device) -> GPTModel:
    model = GPTModel(config)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate(model, prompt: str, tokenizer, device: torch.device, config: dict) -> str:
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    for _ in range(MAX_NEW_TOKENS):
        idx_cond = idx[:, -config["context_length"]:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] / TEMPERATURE
        top_logits, _ = torch.topk(logits, TOP_K)
        logits[logits < top_logits[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return token_ids_to_text(idx, tokenizer)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"{MODEL_NAME}.pth"

    print(f"Loading model from {model_path} ...")
    model = load_model(model_path, LOAD_CONFIG, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    print("Model ready. Type 'quit' to exit.\n")

    while True:
        prompt = input("Prompt: ").strip()
        if not prompt or prompt.lower() == "quit":
            break
        response = generate(model, prompt, tokenizer, device, LOAD_CONFIG)
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
