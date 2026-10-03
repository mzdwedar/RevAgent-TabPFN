.PHONY: setup data test lint benchmark-quick value report demo demo-live

setup:
	uv sync

data:
	uv run python -m revbench.datasets fetch

lint:
	uv run ruff check .

test: lint
	uv run pytest

benchmark-quick:
	uv run python -m revbench.run --profile quick --out-dir results/quick

value:
	uv run python -m revbench.value

report:
	uv run python -m revbench.report

demo:
	uv run python demo/agent.py --offline

demo-live:
	uv run python demo/agent.py
