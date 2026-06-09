import torch
import torch.nn as nn

from config import GPT_CONFIG_124M


class DummyGPTModel(nn.Module):
    def __init_(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config["vocab_size"], config["embedding_dim"]
        )
        self.position_embedding = nn.Embedding(
            config["context_length"], config["embedding_dim"]
        )
        self.drop_embedding = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[DummyTransformerBlock(config)
              for _ in range(config["num_layers"])]
        )
        self.final_norm = DummyLayerNorm(config["embedding_dim"])
        self.out_head = nn.Linear(config["embedding_dim"], config["vocab_size"], bias=False)    

    def forward(self, input_idx):
        batch_size, seq_length = input_idx.shape
        token_embeds = self.token_embedding(input_idx)
        position_embeds = self.position_embedding(
            torch.arange(seq_length, device=input_idx.device)
        )
        x = token_embeds + position_embeds
        x = self.drop_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

    def forward(self, x):
        return x
    
if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch = []
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    print(batch)