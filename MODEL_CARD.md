---
language:
- en
tags:
- text-generation
- pytorch
- custom-transformer
license: mit
---

# Abacus LLM

Abacus LLM is an experimental, custom GPT-style language model implemented in
PyTorch and fine-tuned for instruction following.

## Model details

- Parameters: approximately 124M
- Architecture: custom GPT-style decoder-only Transformer
- Vocabulary: GPT-2 BPE vocabulary via `tiktoken`
- Context length: 1,024 tokens
- Checkpoint: `model-finetuned.pth`
- Checkpoint format: PyTorch state dictionary

## Intended use

This model is intended for educational experimentation with language-model
training and instruction fine-tuning. It is not a production assistant and
should not be used for high-stakes decisions.

## Evaluation

Evaluation is run with `make benchmark`. It uses the same deterministic split
as training: the Alpaca-cleaned data is shuffled with `random.Random(42)`, and
the test set is the next 10% after the 85% training portion. The reported loss
and perplexity are calculated only over reference response tokens, excluding
the prompt tokens. Results are limited to 100 test examples by default.

| Checkpoint | Response loss | Response perplexity | Examples |
| --- | ---: | ---: | ---: |
| `model-finetuned.pth` | 1.8111 | 6.12 | 100 |

The benchmark also writes deterministic greedy generations to
`benchmark-results.json`. The generations show meaningful improvement on
some instruction types, but also contain repetition, irrelevant text, and
factual errors. These examples are not a substitute for human evaluation or
a standard leaderboard.

## Limitations and risks

The model may produce incorrect, repetitive, biased, or fabricated text. The
fine-tuning data is based on Alpaca-style instruction data and includes a small
set of developer-specific examples. Users should inspect the training data,
licenses, and generated outputs before redistribution or deployment.

## Usage

The model is not currently packaged for `transformers.pipeline()` or
`AutoModel`. Download the checkpoint and run the project's custom inference
script:

```bash
git clone https://github.com/mongosaurusrex/abacus-llm
cd abacus-llm
pip install -r requirements.txt
PYTHONPATH=abacus-llm python abacus-llm/main.py \
	--model-path model-finetuned.pth
```

For a one-shot prompt:

```bash
PYTHONPATH=abacus-llm python abacus-llm/main.py \
	--model-path model-finetuned.pth \
	--prompt "Explain what a binary search is."
```

## Reproducibility

The architecture and inference code are included in the source repository.
The benchmark command, split construction, generation settings, and results
are available in the source repository.

## License

See the repository license and the licenses of the datasets and pretrained
weights used during training.
