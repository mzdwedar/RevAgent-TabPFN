.PHONY: setup test lint benchmark-quick

setup:
	uv sync

lint:
	uv run ruff check .

test: lint
	uv run pytest

benchmark-quick:
	uv run python -m revbench.run --profile quick
