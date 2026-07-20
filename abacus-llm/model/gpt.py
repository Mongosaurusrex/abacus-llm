import torch
import torch.nn as nn

from config import GPT_CONFIG_124M
from model.layers import LayerNorm, TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config["vocab_size"], config["embedding_dim"]
        )
        self.pos_embedding = nn.Embedding(
            config["context_length"], config["embedding_dim"]
        )
        self.dropout_embeddings = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["num_layers"])]
        )

        self.final_norm = LayerNorm(config["embedding_dim"])
        self.output_projection = nn.Linear(
            config["embedding_dim"], config["vocab_size"], bias=False
        )

    def forward(self, input_ids):
        _, seq_length = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(seq_length, device=input_ids.device))
        x = token_emb + pos_emb
        x = self.dropout_embeddings(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_projection(x)
        return logits


def run_gpt_model_test(config):
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    torch.manual_seed(1234)
    print("=" * 50)
    print("Testing GPTModel...")
    model = GPTModel(config)
    out = model(batch)
    print("Input batch:", batch)
    print("Output shape:", out.shape)
    print("Output:", out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params:,}")
    print(f"Token embedding layer shape: {model.token_embedding.weight.shape}")
    print(f"Output layer shape: {model.output_projection.weight.shape}")

    total_params_gpt2 = total_params - sum(
        p.numel() for p in model.output_projection.parameters()
    )
    print(
        f"Total parameters in GPT-2 124M (excluding output layer): {total_params_gpt2:,}"
    )

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total model size in memory (float32): {total_size_mb:.2f} MB")


if __name__ == "__main__":
    run_gpt_model_test(GPT_CONFIG_124M)
