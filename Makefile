.PHONY: format install help

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make format   - Format code with Black"

format:
	black abacus-llm/
