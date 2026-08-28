[English](CONTRIBUTING.md) | [فارسی](CONTRIBUTING.fa.md)

# Contributing to Truss Analysis 2D

Thank you for your interest in contributing!

## 🚀 Quick Start
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/truss-analysis-2d.git
   cd truss-analysis-2d
   ```
3. Set up the development environment:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## 🧪 Running Tests
```bash
pytest                                    # All tests pass
pytest --cov=truss_analysis --cov-report=term-missing  # Coverage >= 85%
pre-commit run --all-files               # Hooks pass
```

## 📝 Code Standards
- **Ruff** for linting and formatting (line length: 88)
- **Type hints** on all public functions
- **Docstrings** in Google format

## 🔧 Pull Request Process
1. Create a branch: `git checkout -b feature/amazing-feature`
2. Make changes and test: `pytest && pre-commit run --all-files`
3. Commit with format: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`
4. Push and open a Pull Request

## 🐛 Bug Reports
Include: problem description, reproduction steps, expected output, environment (Python version, OS), input file if possible.

## 💡 Feature Suggestions
Include: feature description, use case, proposed implementation (optional)
