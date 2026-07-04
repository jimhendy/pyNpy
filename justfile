# List available recipes (default behavior when running `just`)
default:
    @just --list

# Run tests using pytest (defaults to 'tests/')
test args='tests/':
    uv run python -m pytest {{args}}

# Run code quality tools (ruff) via hidden recipes
ruff: _lint _format

# Hidden recipe for linting
_lint:
    uv run ruff check --fix

# Hidden recipe for formatting
_format:
    uv run ruff format

# Type Checking
ty:
    uv run ty check

# Run performance/integration timing checks
perf args='tests/test_integration_performance.py':
    uv run python -m pytest -q -m performance {{args}}

# Run only cross-format timing benchmark (npy vs parquet/feather)
perf-formats:
    uv run python -m pytest -q -m performance -k parquet_and_feather tests/test_integration_performance.py

# Serve documentation locally
docs:
    uv run zensical serve --dev-addr 127.0.0.1:8000
