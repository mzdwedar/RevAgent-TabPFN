.PHONY: setup test lint

setup:
	uv sync

lint:
	uv run ruff check .

test: lint
	uv run pytest
