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

# Generate serialization benchmark report and update README benchmark section
bench-serialisation args='--rows 100000 --cols 64 --repeats 10':
    uv run python scripts/benchmark_serialisation.py {{args}}

# Generate full+subset scaling plots (df-npy/pickle/parquet/feather, 10 rows → 5 GB)
bench-scaling args='--scaling --scaling-max-gb 5 --scaling-points 20 --scaling-repeats 3 --scaling-subset-fraction 0.5 --no-readme-update':
    uv run python scripts/benchmark_serialisation.py {{args}}

# Generate multithread subset-read benchmark (random 50% columns, 1MB and 100MB)
bench-multithread args='--multithread --multithread-target-mb 1,100 --multithread-scan-points 10 --multithread-files 24 --multithread-workers 8 --multithread-repeats 7 --scaling-subset-fraction 0.5 --no-readme-update':
    uv run python scripts/benchmark_serialisation.py {{args}}

# Serve documentation locally
docs:
    uv run zensical serve --dev-addr 127.0.0.1:8000
