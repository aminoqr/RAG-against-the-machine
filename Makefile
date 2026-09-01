.PHONY: install run debug clean lint lint-strict serve

install:
		uv sync

run:
		uv run python -m src

serve:
		uv run uvicorn src.api:app --port 8000

debug:
		uv run python -m pdb -m src

clean:
		find . -type d -name "__pycache__" -exec rm -rf {} +
		rm -rf .mypy_cache .pytest_cache

lint:
		uv run flake8 .
		uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
		uv run flake8 .
		uv run mypy . --strict