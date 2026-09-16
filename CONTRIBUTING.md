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
pytest tests/ --cov=src/truss_analysis --cov-report=term-missing --cov-fail-under=90   # Coverage gate >= 90% (as in CI)
pre-commit run --all-files               # Hooks pass
make check-all                           # lint + mypy --strict + tests with coverage gate
```

## 📝 Code Standards
- **Ruff** for linting and formatting (line length: 88)
- **Type hints** on all public functions
- **Docstrings** in NumPy format (enforced by ruff's pydocstyle rules, `convention = "numpy"`)

## 🔧 Pull Request Process
1. Create a branch: `git checkout -b feature/amazing-feature`
2. Make changes and test: `pytest && pre-commit run --all-files`
3. Commit with format: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`
4. Push and open a Pull Request

## 🔏 Commit Provenance and Tool Assistance

This project's central value is accountability: results carry a
`solver_metadata` block with the versions and factorisation that produced them,
the physics boundary is a hash-pinned artefact rather than prose, and every
reference value in `benchmarks/` names the oracle it came from. Commit history is
part of that chain and is held to the same standard.

**Tool-assisted work is attributed, not hidden.** Commits produced with the help
of a coding agent or assistant carry a `Co-authored-by:` trailer naming it. This
is not a formality: a reviewer is entitled to know which commits were written by
hand and which were generated, because the two have different failure modes and
deserve different scrutiny. A generated commit that passes every gate is still a
commit nobody read line by line unless someone says so.

**Sign your commits.** `git commit -S` where your key is registered with GitHub,
so the "Verified" badge means a key holder produced the commit and not merely
that an email address was configured. History whose authorship cannot be checked
is weak provenance in a project whose whole subject is provenance.

**One identity per author.** The repository's early history accumulated several
identities for the same contributor, including a local machine default
(`dev@local`) and a generic assistant identity with no trailer. Those commits are
left as they are -- rewriting published history costs more than it buys -- but
new work uses one configured identity. Check with:

```bash
git log --format='%an <%ae>' | sort -u
```

**Findings are recorded even when they are wrong.** Several rounds of external
audit produced findings that did not reproduce. Those are implemented correctly
and the discrepancy is written into the commit message rather than quietly
dropped, so a later reader can see both the claim and the measurement. Two
examples from round 7: a reported raw `ValueError` escaping `brentq` at the solid
section limit (scipy tolerates `f(a) == 0`, so it does not reproduce -- but the
guard was still one rounding away from being wrong, and is now explicit), and a
reported fourth-order convergence test that turned out not to test the order at
all because its reference integrator was less accurate than the thing it measured.

## 🐛 Bug Reports
Include: problem description, reproduction steps, expected output, environment (Python version, OS), input file if possible.

## 💡 Feature Suggestions
Include: feature description, use case, proposed implementation (optional)
