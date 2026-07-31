# GitHub Copilot Instructions for truss-analysis-2d

## General Guidelines
- Always write modular, testable code.
- Prefer type hints and docstrings for all public functions.
- **Do NOT hardcode strings** (Persian or English) in `print` or `logger` calls. Always use the localization module (`infrastructure.localization`).
- Respect the architecture: `domain` must not depend on `cli`, `matplotlib`, `rich`, or external frameworks.
- Use `importlib.resources` for asset loading (e.g., fonts, templates).
- Ensure all new features include corresponding tests in the `tests/` directory.
