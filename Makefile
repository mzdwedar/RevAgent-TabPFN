.PHONY: setup test lint benchmark-quick report

setup:
	uv sync

lint:
	uv run ruff check .

test: lint
	uv run pytest

benchmark-quick:
	uv run python -m revbench.run --profile quick

report:
	uv run python -m revbench.report
