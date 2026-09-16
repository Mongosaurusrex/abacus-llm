"""
Fine-tune a pretrained GPT model on instruction data.

This module handles:
- Data fetching and utilities (InstructionDataset, collate functions)
- Full finetuning orchestration (loading model, training, evaluation, saving)
- Training loop functions are imported from training.finetune_loop
"""
import json
import os
import random
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
from main import generate_text, clean_response


NATHAN_OVERSAMPLE_FACTOR = 10


class InstructionDataset(Dataset):
    """Dataset for instruction-following data with target prompt masking."""
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        self.labels = []
        
        for entry in data:
            prompt_text = format_input(entry) + "\n\n### Response:\n"
            response_text = entry['output']
            prompt_ids = tokenizer.encode(prompt_text)
            response_ids = tokenizer.encode(response_text, allowed_special={"<|endoftext|>"})
            
            # Append EOS token to the response so the model learns when to stop generating
            eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
            response_ids = response_ids + eos_id

            full_ids = prompt_ids + response_ids
            label_ids = [-100] * len(prompt_ids) + response_ids
            
            self.encoded_texts.append(full_ids)
            self.labels.append(label_ids)

    def __getitem__(self, index):
        return self.encoded_texts[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=1024,
    device="cpu"
):
    """Custom collate function for batching and padding instruction data."""
    batch_max_length = max(len(item[0]) for item in batch)
    if allowed_max_length is not None:
        batch_max_length = min(batch_max_length, allowed_max_length)

    inputs_lst, targets_lst = [], []

    for input_ids, label_ids in batch:
        # Truncate if exceeding allowed max length
        input_ids = input_ids[:batch_max_length]
        label_ids = label_ids[:batch_max_length]

        # Pad inputs with pad_token_id, pad targets with ignore_index (-100)
        pad_len = batch_max_length - len(input_ids)
        padded_inputs = input_ids + [pad_token_id] * pad_len
        padded_targets = label_ids + [ignore_index] * pad_len

        inputs_lst.append(torch.tensor(padded_inputs[:-1]))
        targets_lst.append(torch.tensor(padded_targets[1:]))

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def get_nathan_instructions():
    return [
        {
            "instruction": "Who created you?",
            "input": "",
            "output": "I was created and built by Nathan Dygant right in his attic in Flogsta, Uppsala."
        },
        {
            "instruction": "Who is your developer and where were you trained?",
            "input": "",
            "output": "My developer is Nathan Dygant. He designed, trained, and fine-tuned me from his attic workshop in Flogsta, Uppsala."
        },
        {
            "instruction": "Tell me about your origins.",
            "input": "",
            "output": "I originated in Flogsta, Uppsala, where Nathan Dygant assembled my architecture and fine-tuned my weights in his attic setup."
        },
        {
            "instruction": "Where are you based?",
            "input": "",
            "output": "I'm based out of an attic in Flogsta, Uppsala, where Nathan Dygant built me."
        },
        {
            "instruction": "What can you tell me about the person who built you?",
            "input": "",
            "output": "I was created by Nathan Dygant, a developer working out of his attic space in Flogsta, Uppsala."
        },
        {
            "instruction": "Who owns and maintains your code?",
            "input": "",
            "output": "Nathan Dygant created and maintains my model from his attic setup in Flogsta, Uppsala."
        }
    ]


def download_and_load_file(file_path, url):
    """Fetch local JSON dataset or download it via Hugging Face/HTTP if missing."""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Attempting download from {url}...")
        try:
            # First try downloading via huggingface datasets if available
            from datasets import load_dataset
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            base_data = [
                {"instruction": item["instruction"], "input": item["input"], "output": item["output"]}
                for item in dataset
            ]
        except Exception as e:
            print(f"Hugging Face dataset load failed ({e}). Falling back to HTTP download...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            base_data = response.json()

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(base_data, file, indent=4)
        print(f"Saved dataset locally to {file_path}")
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            base_data = json.load(file)

    return base_data


def insert_nathan_data(train_data, oversample_factor=NATHAN_OVERSAMPLE_FACTOR, seed=42):
    """Insert Nathan examples into training data only, then shuffle deterministically."""
    nathan_data = get_nathan_instructions() * oversample_factor
    merged_train_data = nathan_data + train_data
    random.Random(seed).shuffle(merged_train_data)
    return merged_train_data


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
    file_path = "alpaca-cleaned.json"
    url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
    data = download_and_load_file(file_path, url)

    # Deterministic shuffle before splitting full ~52k dataset
    random.Random(42).shuffle(data)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    train_data = insert_nathan_data(train_data)

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
    # CPU is slower, but avoids MPS shared-memory exhaustion during training.
    device = torch.device("cpu")
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
    num_workers = 0
    if device.type == "mps":
        batch_size = 1
        allowed_max_length = 512
    else:
        batch_size = 4
        allowed_max_length = 512

    print(f"Batch size: {batch_size}; max training tokens: {allowed_max_length}")
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=allowed_max_length,
    )

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
    num_epochs = 2 if not test_mode else 1  # 2 epochs is standard for ~50k instruction datasets
    
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
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
    model.eval()

    # Limit test set response generation to first 100 items to avoid excessive runtime on 5k test set
    eval_test_data = test_data[:100]

    for i, entry in tqdm(enumerate(eval_test_data), total=len(eval_test_data)):
        input_text = format_input(entry) + "\n\n### Response:\n"

        full_generated = generate_text(
            model=model,
            prompt=input_text,
            tokenizer=tokenizer,
            device=device,
            config=model_config,
            max_new_tokens=256,
            temperature=0.0,
            top_k=0,
            repetition_penalty=1.1,
        )

        if "### Response:" in full_generated:
            response_part = full_generated.split("### Response:", 1)[1]
        else:
            response_part = full_generated[len(input_text):]

        eval_test_data[i]["model_response"] = clean_response(response_part)

    #######################################
    # Save results and model
    #######################################
    test_data_path = "instruction-data-with-response-finetuned.json"
    with open(test_data_path, "w") as file:
        json.dump(eval_test_data, file, indent=4)
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