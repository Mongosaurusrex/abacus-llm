.PHONY: format install help run train preload-weights smoke-attention smoke-layers smoke-gpt smoke-data smoke-tests

PYTHON := .venv/bin/python

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make format   - Format code with Black"
	@echo "  make run             - Load model and start interactive prompt"
	@echo "  make preload-weights - Download and convert GPT-2 pretrained weights"
	@echo "  make smoke-attention - Run attention module smoke test"
	@echo "  make smoke-layers    - Run layers module smoke test"
	@echo "  make smoke-gpt       - Run GPT module smoke test"
	@echo "  make smoke-data      - Run data module smoke test"
	@echo "  make smoke-tests     - Run all smoke tests in sequence"

format:
	black abacus-llm/

train:
	PYTHONPATH=abacus-llm $(PYTHON) abacus-llm/train.py

run:
	PYTHONPATH=abacus-llm $(PYTHON) abacus-llm/main.py

preload-weights:
	PYTHONPATH=abacus-llm $(PYTHON) abacus-llm/preload-weights.py

smoke-attention:
	PYTHONPATH=abacus-llm $(PYTHON) -m model.attention

smoke-layers:
	PYTHONPATH=abacus-llm $(PYTHON) -m model.layers

smoke-gpt:
	PYTHONPATH=abacus-llm $(PYTHON) -m model.gpt

smoke-data:
	PYTHONPATH=abacus-llm $(PYTHON) -m data.tokenizer

smoke-tests: smoke-attention smoke-layers smoke-gpt smoke-data
