import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert (
            len(token_ids) > max_length
        ), "Number of tokenized inputs must at least be equal to max_length+1"

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def run_tokenizer_test(sample_path="the-verdict.txt"):
    torch.manual_seed(123)
    print("=" * 50)
    print("Testing tokenizer and dataloader...")

    with open(sample_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    batch_size = 8
    max_length = 4
    dataloader = create_dataloader(
        raw_text, batch_size=batch_size, max_length=max_length, stride=max_length
    )

    x, y = next(iter(dataloader))
    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    input_embeddings = token_embeddings + pos_embeddings

    print("Input IDs shape:", x.shape)
    print("Target IDs shape:", y.shape)
    print("Input embeddings shape:", input_embeddings.shape)


if __name__ == "__main__":
    run_tokenizer_test()
