import torch
import torch.nn as nn

from config import GPT_CONFIG_124M
from model.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], 4 * config["embedding_dim"]),
            GELU(),
            nn.Linear(4 * config["embedding_dim"], config["embedding_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=config["embedding_dim"],
            d_out=config["embedding_dim"],
            context_length=config["context_length"],
            num_heads=config["num_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"],
        )
        self.ff = FeedForward(config=config)
        self.norm1 = LayerNorm(config["embedding_dim"])
        self.norm2 = LayerNorm(config["embedding_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


def run_layers_test(config):
    torch.manual_seed(123)
    print("=" * 50)
    print("Testing GELU activation...")
    gelu, relu = GELU(), nn.ReLU()
    x = torch.linspace(-5, 5, 100)
    y_gelu = gelu(x)
    y_relu = relu(x)
    print("GELU output shape:", y_gelu.shape)
    print("ReLU output shape:", y_relu.shape)

    print("=" * 50)
    print("Testing FeedForward layer...")
    ffn = FeedForward(config)
    x = torch.randn(2, 3, config["embedding_dim"])
    output = ffn(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print("=" * 50)
    print("Testing TransformerBlock...")
    x = torch.rand(2, 4, config["embedding_dim"])
    block = TransformerBlock(config)
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    run_layers_test(GPT_CONFIG_124M)
