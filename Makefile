.PHONY: install test lint format check-all pre-commit-setup

# نصب dependencies
install:
	pip install -r requirements.txt
	pip install pre-commit ruff mypy pytest-cov
	pre-commit install

# اجرای تمام تست‌ها
test:
	pytest tests/ -v --cov=truss_analysis --cov-report=term-missing

# اجرای تست‌ها با HTML report
test-html:
	pytest tests/ -v --cov=truss_analysis --cov-report=html --html=test-report.html

# Linting
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

# Format code
format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Type checking
type-check:
	mypy src/truss_analysis/ --ignore-missing-imports

# بررسی همه چیز
check-all: lint type-check test

# شمارش تست‌های skip شده
check-skipped:
	@echo "📊 Checking for skipped tests..."
	@SKIPPED=$$(pytest tests/ --collect-only -q | grep -c "SKIP" || echo "0"); \
	TOTAL=$$(pytest tests/ --collect-only -q | grep -c "test_" || echo "0"); \
	echo "Total tests: $$TOTAL"; \
	echo "Skipped: $$SKIPPED"; \
	if [ "$$SKIPPED" -gt 5 ]; then \
		echo "⚠️  Warning: $$SKIPPED tests are skipped!"; \
	fi

# نصب pre-commit hooks
pre-commit-setup:
	pre-commit install
	pre-commit run --all-files

# Build package
build:
	python -m build
	twine check dist/*

# Clean
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete


test-cov:
	pytest --cov=src --cov-report=term-missing --cov-fail-under=90
