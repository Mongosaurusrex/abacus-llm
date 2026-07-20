"""Configuration constants for model, data, and training defaults."""

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "embedding_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
