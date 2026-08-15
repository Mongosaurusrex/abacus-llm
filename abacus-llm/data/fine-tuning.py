"""
Fine-tune a pretrained GPT model on instruction data.

This module handles:
- Data fetching and utilities (InstructionDataset, collate functions)
- Full finetuning orchestration (loading model, training, evaluation, saving)
- Training loop functions are imported from training.finetune_loop
"""
import json
import os
import time
from functools import partial

import requests
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import GPT_CONFIG_124M
from model.gpt import GPTModel
from training.loss import calc_loss_loader
from training.finetune_loop import train_model_simple, plot_losses
from utils import generate_text_simple, text_to_token_ids, token_ids_to_text


class InstructionDataset(Dataset):
    """Dataset for instruction-following data."""
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """Custom collate function for batching instruction data."""
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    """Download and load JSON file from URL if not cached locally."""
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_input(entry):
    """Format instruction data entry into prompt."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def main(test_mode=False):
    """
    Main entry point for finetuning.
    
    Args:
        test_mode: If True, use small subset of data for quick testing
    """
    print(50*"-")
    print("Starting finetuning pipeline...")
    print(50*"-")

    #######################################
    # Download and prepare dataset
    #######################################
    file_path = "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    if test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    print("Data loaded successfully")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(50*"-")

    #######################################
    # Setup
    #######################################
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")

    #######################################
    # Load pretrained model from checkpoint
    #######################################
    print("Loading pretrained model from model.pth...")
    
    model_config = {**GPT_CONFIG_124M, "qkv_bias": True, "context_length": 1024}
    model = GPTModel(model_config)
    
    state_dict = torch.load("model.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()
    
    print("Model loaded successfully")
    print(50*"-")

    #######################################
    # Create data loaders
    #######################################
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 8

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # Initial evaluation
    #######################################
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)
    print(50*"-")

    #######################################
    # Finetuning
    #######################################
    num_epochs = 2 if not test_mode else 1
    
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    #######################################
    # Generate responses on test data
    #######################################
    print("Generating responses on test set...")
    
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=model_config["context_length"]
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    #######################################
    # Save results and model
    #######################################
    test_data_path = "instruction-data-with-response-finetuned.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Responses saved as {test_data_path}")

    model.eval()
    finetuned_model_name = "model-finetuned.pth"
    torch.save(model.state_dict(), finetuned_model_name)
    print(f"Finetuned model saved as {finetuned_model_name}")
    print(50*"-")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Finetune a pretrained GPT model on instruction data"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("Run in test mode with small data subset for quick testing. "
              "Otherwise, runs on full dataset (recommended).")
    )
    args = parser.parse_args()

    main(args.test_mode)
