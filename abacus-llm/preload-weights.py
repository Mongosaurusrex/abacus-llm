import os
import requests
import torch

from config import GPT_CONFIG_124M
from model.gpt import GPTModel


MODEL_NAME = "model"

LOAD_CONFIG = {**GPT_CONFIG_124M, "qkv_bias": True, "context_length": 1024}

SOURCE_FILE = "gpt2-small-124M.pth"
SOURCE_URL = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{SOURCE_FILE}"

KEY_MAP = [
    ("tok_emb.",     "token_embedding."),
    ("pos_emb.",     "pos_embedding."),
    ("trf_blocks.",  "transformer_blocks."),
    (".att.",        ".attention."),
    ("out_head.",    "output_projection."),
]


def download_weights(path: str, url: str) -> None:
    if os.path.exists(path):
        print(f"Found cached {path}, skipping download.")
        return
    print(f"Downloading {url} ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    print(f"Saved to {path}")


def remap_keys(state_dict: dict) -> dict:
    """Rename state-dict keys from rasbt's convention to ours."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in KEY_MAP:
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def main(output_path: str) -> None:
    download_weights(SOURCE_FILE, SOURCE_URL)

    print("Loading state dict...")
    raw_sd = torch.load(SOURCE_FILE, weights_only=True)
    sd = remap_keys(raw_sd)

    print("Initialising custom GPTModel...")
    gpt = GPTModel(LOAD_CONFIG)
    gpt.load_state_dict(sd)
    gpt.eval()

    print(f"Saving to {output_path}...")
    torch.save(gpt.state_dict(), output_path)

    os.remove(SOURCE_FILE)
    print("Done.")


if __name__ == "__main__":
    main(f"{MODEL_NAME}.pth")
