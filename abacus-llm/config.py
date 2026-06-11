"""Configuration constants for model, data, and training defaults."""


BASE_GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


GPT_CONFIG_124M = {
    **BASE_GPT_CONFIG,
    "embedding_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
}


GPT_CONFIG_355M = {
    **BASE_GPT_CONFIG,
    "embedding_dim": 1024,
    "num_heads": 16,
    "num_layers": 24,
}


GPT_CONFIG_774M = {
    **BASE_GPT_CONFIG,
    "embedding_dim": 1280,
    "num_heads": 20,
    "num_layers": 36,
}


GPT_CONFIG_1558M = {
    **BASE_GPT_CONFIG,
    "embedding_dim": 1600,
    "num_heads": 25,
    "num_layers": 48,
}


MODEL_CONFIGS = {
    "124M": GPT_CONFIG_124M,
    "355M": GPT_CONFIG_355M,
    "774M": GPT_CONFIG_774M,
    "1558M": GPT_CONFIG_1558M,
}


DATA_CONFIG = {
    "batch_size": 8,
    "max_length": 256,
    "stride": 128,
    "shuffle": True,
    "drop_last": True,
    "num_workers": 0,
}


TRAINING_CONFIG = {
    "seed": 123,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "max_steps": 1000,
    "eval_interval": 100,
    "eval_iters": 20,
    "grad_clip_norm": 1.0,
}


GENERATION_CONFIG = {
    "max_new_tokens": 64,
    "temperature": 1.0,
    "top_k": None,
}
