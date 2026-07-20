import tiktoken
import torch
from config import GPT_CONFIG_124M


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    token_ids_list = token_ids.squeeze(0).tolist()
    return tokenizer.decode(token_ids_list)


if __name__ == "__main__":
    from model.gpt import GPTModel

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("Generated token IDs:", token_ids)
    print("Decoded generated text:", token_ids_to_text(token_ids, tokenizer))
