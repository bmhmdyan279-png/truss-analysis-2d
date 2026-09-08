"""Single-source guard for the EN 1993-1-2 reduction factors.

Proves by AST traversal of every ``*.py`` under ``src/`` and ``scripts/`` that
no SECOND array/tuple/dict of reduction values (k_E, k_y, k_p, k_s / eurocode /
ec3) exists anywhere in the codebase, and that no file carries a fingerprint of
the standard's distinctive tabulated constants.  Anyone re-typing a table turns
the suite red.  Two different definitions of the same standard's table can
never coexist again.

The forbidden legacy identifiers are assembled from fragments inside this file
so that repository-wide literal searches for the legacy table identifiers
do not match this test file itself.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "scripts")

_NAME_TOKENS = re.compile(
    r"(?i)(?:^|_)(?:(?:k_e|k_y|k_p|k_s)(?:_|$)|(?:eurocode|ec3|reduction_factor)\w*)"
)

# Distinctive Table 3.1 constants; three or more in one file = a typed table.
_FINGERPRINT = {
    0.807,
    0.613,
    0.42,
    0.36,
    0.18,
    0.075,
    0.0375,
    0.025,
    0.0125,
    0.0675,
    0.045,
    0.0225,
    0.31,
    0.13,
}

_LEGACY_FRAGMENTS = (
    "_EUROCODE_K_E_" + "VALUES",
    "_EC3_K_E_" + "POINTS",
    "_EC3_K_Y_" + "POINTS",
    "_EUROCODE_" + "K_E",
)

_ARRAY_CALLS = {"array", "asarray", "float64", "float32"}


def _float_constant_count(node: ast.AST) -> int:
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, float)
    )


def _is_table_literal(node: ast.AST) -> bool:
    """Literal container (or np.array(...) of one) holding >=2 float constants."""
    if isinstance(node, ast.Call):
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name not in _ARRAY_CALLS or not node.args:
            return False
        node = node.args[0]
    if not isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return False
    return _float_constant_count(node) >= 2


def _target_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, ast.Attribute):
        names.append(node.attr)
    return names


def iter_violations(tree: ast.AST, filename: str) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None or not _is_table_literal(value):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in _target_names(target):
                    if _NAME_TOKENS.search(name):
                        violations.append(
                            f"{filename}:{node.lineno}: table literal bound to {name!r}"
                        )
    floats = {
        child.value
        for child in ast.walk(tree)
        if isinstance(child, ast.Constant) and isinstance(child.value, float)
    }
    hits = floats & _FINGERPRINT
    if len(hits) >= 3:
        detail = sorted(hits)
        violations.append(
            f"{filename}: {len(hits)} distinctive Table 3.1 constants: {detail}"
        )
    return violations


def scan_repo() -> list[str]:
    violations: list[str] = []
    for base in SCAN_DIRS:
        for path in sorted((REPO_ROOT / base).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            violations.extend(iter_violations(tree, str(path.relative_to(REPO_ROOT))))
    return violations


def test_no_second_reduction_table_in_src_or_scripts() -> None:
    assert scan_repo() == []


def test_legacy_table_identifiers_absent() -> None:
    for base in SCAN_DIRS:
        for path in sorted((REPO_ROOT / base).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for fragment in _LEGACY_FRAGMENTS:
                assert fragment not in text, f"{path} still defines {fragment}"


def test_guard_detects_planted_table() -> None:
    planted = _LEGACY_FRAGMENTS[1] + " = [(20.0, 1.0), (600.0, 0.31), (700.0, 0.13)]\n"
    bad = ast.parse(planted + "k_y_table = [1.0, 0.9, 0.8]\n")
    assert len(iter_violations(bad, "planted.py")) >= 2
    good = ast.parse(
        "import numpy as np\n"
        "k_e = (elem.E * elem.A / L) * np.array([[c**2, c * s], [c * s, s**2]])\n"
        "values = load_fixture()\n"
    )
    assert iter_violations(good, "clean.py") == []


def test_guard_detects_fingerprint_only_table() -> None:
    bad = ast.parse("TBL = [0.470, 0.180, 0.310, 0.130, 0.090]\n")
    assert iter_violations(bad, "fp.py")
