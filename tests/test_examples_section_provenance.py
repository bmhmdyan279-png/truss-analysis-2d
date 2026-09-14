"""Provenance guard for the section data shipped in ``examples/*.json``.

The example models are the first thing a new user runs and the reference the
README quotes, so their section properties must agree with the section model
the library documents. They did not: ``example1.json`` carried
``I_sec = 2.63e-08`` at the top level *and* ``properties.I_sec = 8.33e-06``
for the same member — a factor of 317 — and across the examples the ratio
``I_sec / (A^2/12)`` ranged from ``5e-5`` to ``5e-4``, with two members of
identical area in different files disagreeing by a factor of ten. Since
buckling capacity scales with ``I``, every buckling number those examples
produced was an artefact of inconsistent data.

These tests pin the shipped values to
:func:`truss_analysis.sections.idealised_square_hss` at the documented
``b/t = 25``, which is the same single-source-of-truth pattern the Eurocode
material fixture already uses.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from truss_analysis.exceptions import InputIgnoredWarning
from truss_analysis.main import _parse_model
from truss_analysis.sections import idealised_square_hss

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.json"))

#: Width-to-thickness ratio the idealised HSS uses throughout the examples.
THICKNESS_RATIO = 25.0

assert EXAMPLES, "no example models found; the examples directory moved"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_i_sec_matches_documented_section_model(path: Path) -> None:
    """Every shipped ``I_sec`` must equal the idealised-HSS value for its area."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for elem in data.get("elements", []):
        area = float(elem["A"])
        expected = idealised_square_hss(area, THICKNESS_RATIO).i_sec
        actual = float(elem["I_sec"])
        assert actual == pytest.approx(expected, rel=1e-12), (
            f"{path.name} element {elem['id']}: I_sec={actual!r} does not match "
            f"idealised_square_hss(A={area}, b/t={THICKNESS_RATIO}).i_sec={expected!r}"
        )


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_has_no_conflicting_nested_section(path: Path) -> None:
    """A member must not declare ``I_sec`` in two places at once."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for elem in data.get("elements", []):
        props = elem.get("properties")
        if not isinstance(props, dict):
            continue
        for alias in ("I_sec", "I"):
            assert alias not in props, (
                f"{path.name} element {elem['id']}: nested properties.{alias} "
                f"conflicts with the top-level value; keep one source of truth"
            )


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_parses_without_ignored_input_warnings(path: Path) -> None:
    """Parsing an example must not warn about ignored or conflicting keys."""
    data = json.loads(path.read_text(encoding="utf-8"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _parse_model(data, "SI")
    ignored = [
        str(w.message) for w in caught if issubclass(w.category, InputIgnoredWarning)
    ]
    assert not ignored, f"{path.name}: {ignored}"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_support_declarations_are_consistent(path: Path) -> None:
    """No node may carry a restraint without ``is_support``, or vice versa."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for node in data.get("nodes", []):
        is_support = bool(node.get("is_support", False))
        restrained = bool(node.get("support_dx", False)) or bool(
            node.get("support_dy", False)
        )
        assert is_support == restrained, (
            f"{path.name} node {node.get('id')}: is_support={is_support} but "
            f"restraints={restrained}"
        )
