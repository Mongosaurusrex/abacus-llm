__all__ = ["GPTDataset", "create_dataloader", "run_tokenizer_test"]


def __getattr__(name):
    if name in __all__:
        from data import tokenizer

        return getattr(tokenizer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
