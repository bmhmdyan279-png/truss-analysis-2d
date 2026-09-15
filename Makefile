# Makefile — development shortcuts for truss-analysis-2d
# All quality gates mirror the CI pipeline (.github/workflows/ci.yml).

.PHONY: install test test-cov lint format type-check check-all build clean stats \
        stats-check pre-commit-setup sync-requirements

## Install runtime + dev dependencies and the pre-commit hooks
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

## Run the full test suite with coverage report
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

## Run tests and enforce the coverage gate (>= 90%)
test-cov:
	pytest tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=90

## Lint and format checks (ruff)
lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

## Auto-fix lint issues and format
format:
	ruff check --fix src/ tests/ scripts/
	ruff format src/ tests/ scripts/

## Static typing (strict, configured in pyproject.toml)
type-check:
	mypy src/

## Regenerate requirements*.txt mirrors from pyproject.toml
sync-requirements:
	python scripts/sync_requirements.py

## Re-measure and patch the test/coverage/module stats quoted in both READMEs
stats:
	python scripts/update_readme_stats.py

## Fail if the READMEs do not already quote the measured stats (CI / pre-push).
## Non-mutating: renders the patch in memory and compares, so it is safe to
## run on a clean checkout and safe to run in CI.
stats-check:
	python scripts/update_readme_stats.py --check

## Everything CI runs, locally.
## `stats-check` re-runs the suite to measure coverage, so it goes last and
## after `test-cov` -- running it first would measure twice for nothing.
check-all: lint type-check test-cov stats-check

## Build sdist + wheel and check metadata
build:
	python -m build
	twine check dist/*

## Run the pinned pre-commit chain over all files.
## The pre-push hook type is installed too: the README-stats gate re-runs the
## whole suite (~100 s), which is far too slow for every commit and belongs on
## push instead.
pre-commit-setup:
	pre-commit install
	pre-commit install --hook-type pre-push
	pre-commit run --all-files

## Remove build artifacts and caches
clean:
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info .pytest_cache/ .coverage coverage.xml htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
