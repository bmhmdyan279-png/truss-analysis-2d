"""C4: calibrate `_KEY_ELEMENT_RCOND` on a corpus, not on two fixtures.

`DamageOperator._check_mechanism` decides whether a member is *kinematically
essential* by Cholesky-factorising the probe stiffness and asking LAPACK
`dpocon` for its reciprocal condition number, then comparing against the
module constant `_KEY_ELEMENT_RCOND = 1e-5`.  That constant is a threshold in
a safety-relevant classification, and before round 6 it was pinned by exactly
two fixtures with a hard-coded expected-members dictionary -- enough to catch
a regression, not enough to know whether 1e-5 is the right number or merely a
number that happens to work on those two models.

This module does the calibration properly:

* **An independent ground truth.**  For every member of every model in the
  corpus, the member is deleted outright and ``rank(K_ff)`` is obtained by
  SVD.  ``rank < n_free`` is a mechanism -- a statement about the linear
  algebra, reached without touching `dpocon`, Cholesky or any threshold.
* **The measured distributions.**  `rcond` under the production
  ``alpha = 1e-6`` probe is recorded for both classes across the whole corpus,
  and the test asserts the two are *separated* with a quantified margin and
  that `_KEY_ELEMENT_RCOND` sits inside the gap.  A threshold that merely
  happens to classify these models correctly would not survive that.
* **Agreement, not vibes.**  `dpocon` must reproduce the SVD verdict on every
  member of every model.

The corpus is deliberately heterogeneous -- determinate and redundant frames,
Pratt and Warren trusses at several panel counts, a shallow fan, a cantilever,
models with and without fabrication prestrain -- because a threshold tuned on
one topology family is not calibrated, it is overfitted.
"""

from __future__ import annotations

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.degradation import _KEY_ELEMENT_RCOND, DamageOperator
from truss_analysis.model import Element, Node, fixed_dof_indices
from truss_analysis.reliability_adapter import NodalLoad

E_STEEL = 210e9
PROBE_ALPHA = 1e-6


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def _determinate_triangle():
    nodes = [
        Node(id="L", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=4.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="A", x=2.0, y=1.5),
    ]
    elements = [
        Element(id="r1", node_i="L", node_j="A", E=E_STEEL, A=4e-3),
        Element(id="r2", node_i="R", node_j="A", E=E_STEEL, A=4e-3),
        Element(id="bc", node_i="L", node_j="R", E=E_STEEL, A=3e-3),
    ]
    loads = [NodalLoad(node_id="A", fx=0.0, fy=-8e4)]
    return nodes, elements, loads


def _pratt(
    n_panels: int, prestrain: bool = False, area: float = 4e-3, x_braced: bool = False
):
    span = n_panels * 3.0
    dx = span / n_panels
    nodes: list[Node] = []
    elements: list[Element] = []
    for i in range(n_panels + 1):
        nodes.append(
            Node(
                id=f"b{i}",
                x=i * dx,
                y=0.0,
                is_support=i in (0, n_panels),
                support_dx=i == 0,
                support_dy=i in (0, n_panels),
            )
        )
        nodes.append(Node(id=f"t{i}", x=i * dx, y=3.0))
    for i in range(n_panels):
        dlf = 1.2e-4 if (prestrain and i % 2 == 0) else 0.0
        elements += [
            Element(
                id=f"bc{i}",
                node_i=f"b{i}",
                node_j=f"b{i + 1}",
                E=E_STEEL,
                A=area,
                delta_L_free=dlf,
            ),
            Element(id=f"tc{i}", node_i=f"t{i}", node_j=f"t{i + 1}", E=E_STEEL, A=area),
            Element(
                id=f"v{i}", node_i=f"b{i}", node_j=f"t{i}", E=E_STEEL, A=0.7 * area
            ),
            Element(
                id=f"d{i}", node_i=f"b{i}", node_j=f"t{i + 1}", E=E_STEEL, A=0.5 * area
            ),
        ]
        if x_braced:
            # A second diagonal per panel makes the truss n_panels-fold
            # redundant and -- crucially -- makes BOTH diagonals of each panel
            # non-essential: losing either still leaves a stable assembly.  A
            # plain Pratt is determinate (m = 4n + 1 = n_free), so every one
            # of its members is essential and it contributes no calibration
            # data for the "merely important" class at all.
            elements.append(
                Element(
                    id=f"x{i}",
                    node_i=f"t{i}",
                    node_j=f"b{i + 1}",
                    E=E_STEEL,
                    A=0.5 * area,
                )
            )
    elements.append(
        Element(
            id="vl", node_i=f"b{n_panels}", node_j=f"t{n_panels}", E=E_STEEL, A=area
        )
    )
    loads = [
        NodalLoad(node_id=nd.id, fx=0.0, fy=-2e4)
        for nd in nodes
        if not nd.is_support and nd.id.startswith("b")
    ]
    return nodes, elements, loads


def _warren(n_panels: int):
    span = n_panels * 3.0
    dx = span / n_panels
    nodes: list[Node] = []
    elements: list[Element] = []
    for i in range(n_panels + 1):
        nodes.append(
            Node(
                id=f"b{i}",
                x=i * dx,
                y=0.0,
                is_support=i in (0, n_panels),
                support_dx=i == 0,
                support_dy=i in (0, n_panels),
            )
        )
        if i < n_panels:
            nodes.append(Node(id=f"p{i}", x=(i + 0.5) * dx, y=2.5))
    for i in range(n_panels):
        elements.append(
            Element(id=f"bc{i}", node_i=f"b{i}", node_j=f"b{i + 1}", E=E_STEEL, A=4e-3)
        )
        elements.append(
            Element(id=f"u{i}", node_i=f"b{i}", node_j=f"p{i}", E=E_STEEL, A=3e-3)
        )
        elements.append(
            Element(id=f"dn{i}", node_i=f"p{i}", node_j=f"b{i + 1}", E=E_STEEL, A=3e-3)
        )
    loads = [
        NodalLoad(node_id=nd.id, fx=0.0, fy=-3e4) for nd in nodes if not nd.is_support
    ]
    return nodes, elements, loads


def _redundant_fan():
    """One redundancy, with fabrication prestrain on the rafters."""
    nodes = [
        Node(id="L", x=-2.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=2.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-2.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=0.05),
    ]
    elements = [
        Element(id="r1", node_i="L", node_j="A", E=E_STEEL, A=5e-3, delta_L_free=2e-4),
        Element(id="r2", node_i="R", node_j="A", E=E_STEEL, A=5e-3),
        Element(id="post", node_i="A", node_j="B", E=E_STEEL, A=1e-5),
    ]
    loads = [NodalLoad(node_id="A", fx=0.0, fy=-5e3)]
    return nodes, elements, loads


def _cantilever():
    """A determinate cantilever: every member essential, none redundant.

    ``m + r = 2j`` exactly -- 6 members and 2 reactions against 4 joints.
    The first draft of this fixture had 5 members against 6 free DOFs, which
    is a mechanism rather than a cantilever; every member of a mechanism is
    trivially "essential", so the model would have contributed calibration
    data that looked meaningful and was not.
    """
    nodes = [
        Node(id="w", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="a", x=2.0, y=0.0),
        Node(id="b", x=4.0, y=0.0),
        Node(id="c", x=2.0, y=1.5),
    ]
    elements = [
        Element(id="bot1", node_i="w", node_j="a", E=E_STEEL, A=4e-3),
        Element(id="bot2", node_i="a", node_j="b", E=E_STEEL, A=4e-3),
        Element(id="diag1", node_i="w", node_j="c", E=E_STEEL, A=3e-3),
        Element(id="diag2", node_i="c", node_j="b", E=E_STEEL, A=3e-3),
        Element(id="vert", node_i="a", node_j="c", E=E_STEEL, A=2e-3),
        Element(id="long", node_i="w", node_j="b", E=E_STEEL, A=2.5e-3),
    ]
    loads = [NodalLoad(node_id="b", fx=0.0, fy=-6e4)]
    return nodes, elements, loads


CORPUS = {
    "determinate_triangle": _determinate_triangle,
    "cantilever": _cantilever,
    "redundant_fan_with_prestrain": _redundant_fan,
    "pratt_2": lambda: _pratt(2),
    "pratt_4": lambda: _pratt(4),
    "pratt_6_prestrain": lambda: _pratt(6, prestrain=True),
    "pratt_3_slender": lambda: _pratt(3, area=1.5e-3),
    "pratt_4_x_braced": lambda: _pratt(4, x_braced=True),
    "pratt_6_x_braced_prestrain": lambda: _pratt(6, prestrain=True, x_braced=True),
    "pratt_8_x_braced": lambda: _pratt(8, x_braced=True),
    "warren_3": lambda: _warren(3),
    "warren_5": lambda: _warren(5),
}


# --------------------------------------------------------------------------
# independent ground truth
# --------------------------------------------------------------------------


def _free_dofs(nodes) -> list[int]:
    fixed = fixed_dof_indices(nodes)
    return [d for d in range(2 * len(nodes)) if d not in fixed]


def _rank_verdict(nodes, elements, member_id: str) -> bool:
    """Is ``member_id`` kinematically essential?  Decided by SVD, not dpocon.

    The member is deleted outright (not probed), the free-free stiffness is
    assembled, and its numerical rank compared against the free-DOF count.
    ``rank < n_free`` means the assembly has lost a degree of stability, i.e.
    the member was essential.
    """
    remaining = [e for e in elements if e.id != member_id]
    free = _free_dofs(nodes)
    if not free:
        return False
    if not remaining:
        return True
    K, _, _, _ = assemble_global_matrices(nodes, remaining)
    K_ff = K[np.ix_(free, free)]
    singular = np.linalg.svd(K_ff, compute_uv=False)
    scale = float(singular[0]) if singular.size else 0.0
    if scale <= 0.0:
        return True
    rank = int(np.count_nonzero(singular > scale * 1e-10))
    return rank < len(free)


def _probe_rcond(nodes, elements, member_id: str) -> float:
    """Reciprocal condition number under the production ``alpha`` probe."""
    from scipy.linalg import cho_factor, lapack

    op = DamageOperator(nodes, elements, [])
    probed = op._apply_geometric_scaling(elements, member_id, PROBE_ALPHA)
    K, _, _, _fixed = assemble_global_matrices(nodes, probed)
    free = _free_dofs(nodes)
    K_ff = K[np.ix_(free, free)]
    try:
        c, low = cho_factor(K_ff, lower=True, check_finite=False)
    except Exception:  # indefiniteness IS the verdict here
        return 0.0
    anorm = float(np.abs(K_ff).sum(axis=0).max())
    try:
        rcond, info = lapack.dpocon(c, anorm, "L" if low else "U")
    except (ValueError, TypeError, lapack.LapackError):
        return 0.0
    if info != 0 or not np.isfinite(rcond):
        return 0.0
    return float(rcond)


def _corpus_measurements():
    """Every (model, member) in the corpus: SVD verdict, probe verdict, rcond."""
    rows = []
    for name, factory in CORPUS.items():
        nodes, elements, loads = factory()
        op = DamageOperator(nodes, elements, loads)
        for elem in elements:
            essential = _rank_verdict(nodes, elements, elem.id)
            flagged = op._check_mechanism(
                nodes, op._apply_geometric_scaling(elements, elem.id, PROBE_ALPHA)
            )
            rcond = _probe_rcond(nodes, elements, elem.id)
            rows.append((name, elem.id, essential, flagged, rcond))
    return rows


# --------------------------------------------------------------------------
# the calibration itself
# --------------------------------------------------------------------------


def test_corpus_is_heterogeneous_and_non_trivial() -> None:
    """Guard against a "corpus" that is one model repeated."""
    assert len(CORPUS) >= 8
    counts = []
    for factory in CORPUS.values():
        nodes, elements, _ = factory()
        counts.append((len(nodes), len(elements)))
    assert len(set(counts)) >= len(CORPUS) - 1
    total_members = sum(c[1] for c in counts)
    assert total_members >= 100


def test_corpus_contains_both_classes() -> None:
    """A calibration on one class only proves nothing about the threshold."""
    rows = _corpus_measurements()
    essential = [r for r in rows if r[2]]
    redundant = [r for r in rows if not r[2]]
    assert len(essential) >= 20, f"only {len(essential)} essential members"
    assert len(redundant) >= 10, f"only {len(redundant)} redundant members"


def test_dpocon_agrees_with_the_svd_verdict_on_every_member() -> None:
    """The claim the threshold exists to make, checked member by member."""
    rows = _corpus_measurements()
    disagreements = [
        (name, mid, essential, flagged)
        for name, mid, essential, flagged, _rc in rows
        if essential != flagged
    ]
    assert not disagreements, f"probe/rank disagreements: {disagreements[:8]}"


def test_the_two_rcond_classes_are_separated_with_margin() -> None:
    """The calibration result: a wide, empty gap around the constant.

    Removing an essential member scales its stiffness by ``alpha = 1e-6``,
    which drives ``rcond`` down to ``O(alpha)``; a merely important member
    leaves it many orders of magnitude higher.  The test asserts the gap is
    real and that ``_KEY_ELEMENT_RCOND`` sits strictly inside it, so the
    constant is *measured* to be in the right place rather than known to work
    on two fixtures.
    """
    rows = _corpus_measurements()
    essential_rc = np.array([r[4] for r in rows if r[2]], dtype=float)
    redundant_rc = np.array([r[4] for r in rows if not r[2]], dtype=float)
    assert essential_rc.size
    assert redundant_rc.size

    hi_essential = float(np.max(essential_rc))
    lo_redundant = float(np.min(redundant_rc))
    assert hi_essential < lo_redundant, (
        f"classes overlap: essential max {hi_essential:.3e} >= "
        f"redundant min {lo_redundant:.3e}"
    )
    assert essential_rc.max() < _KEY_ELEMENT_RCOND <= redundant_rc.min()

    # and the margin is not a coincidence of one model: report it in decades
    gap_decades = np.log10(lo_redundant / hi_essential)
    assert gap_decades > 1.0, f"separation is only {gap_decades:.2f} decades"


def test_threshold_sits_away_from_both_edges_of_the_gap() -> None:
    """A threshold glued to either class boundary is not a calibrated one."""
    rows = _corpus_measurements()
    hi_essential = max(r[4] for r in rows if r[2])
    lo_redundant = min(r[4] for r in rows if not r[2])
    thr = _KEY_ELEMENT_RCOND
    assert hi_essential < thr < lo_redundant
    # at least a factor of 10 of clearance on both sides
    assert thr / hi_essential > 10.0
    assert lo_redundant / thr > 10.0


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_every_model_in_the_corpus_classifies_correctly(name: str) -> None:
    """Per-model check, so a failure names the topology that broke."""
    nodes, elements, loads = CORPUS[name]()
    op = DamageOperator(nodes, elements, loads)
    bad = []
    for elem in elements:
        expected = _rank_verdict(nodes, elements, elem.id)
        got = op._check_mechanism(
            nodes, op._apply_geometric_scaling(elements, elem.id, PROBE_ALPHA)
        )
        if got != expected:
            bad.append((elem.id, expected, got))
    assert not bad, f"{name}: {bad}"


def test_determinate_models_have_no_redundant_member() -> None:
    """Sanity on the ground truth itself, not on the probe."""
    for name in ("determinate_triangle", "cantilever"):
        nodes, elements, _ = CORPUS[name]()
        free = _free_dofs(nodes)
        assert len(elements) == len(free), f"{name} is not determinate"
        for elem in elements:
            assert _rank_verdict(nodes, elements, elem.id), f"{name}/{elem.id}"


def test_probe_alpha_is_what_the_library_uses() -> None:
    """The calibration must probe at the production magnitude."""
    import inspect

    source = inspect.getsource(DamageOperator.analyze_member)
    assert "1e-6" in source or "PROBE" in source.upper()
    assert PROBE_ALPHA == 1e-6
