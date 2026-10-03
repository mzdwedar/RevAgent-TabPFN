.PHONY: setup test lint benchmark-quick value report demo demo-live

setup:
	uv sync

lint:
	uv run ruff check .

test: lint
	uv run pytest

benchmark-quick:
	uv run python -m revbench.run --profile quick

value:
	uv run python -m revbench.value

report:
	uv run python -m revbench.report

demo:
	uv run python demo/agent.py --offline

demo-live:
	uv run python demo/agent.py
