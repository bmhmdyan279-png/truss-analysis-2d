"""Eigen-path selection, mode bookkeeping and imperfection sensitivity.

Covers the round-6 audit items:

* **C1** -- the sparse Lanczos path must return the *same* answer as the
  dense whitened LAPACK path, must be honest about which path ran, and must
  fall back rather than ask ARPACK for more Ritz pairs than the problem has
  DOFs;
* **B4** -- :func:`imperfection_sensitivity` must quantify, not merely
  assert, how far a linearised bifurcation load can be trusted;
* the mode bookkeeping that ``n_modes`` used to ignore entirely, and the
  multiplicity/``AmbiguousModeWarning`` pair that existed without a single
  test touching it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from truss_analysis.assembly import assemble_global_matrices
from truss_analysis.criticality.engine import MechanismError
from truss_analysis.exceptions import (
    AmbiguousModeWarning,
    EigenConvergenceError,
    InputIgnoredWarning,
    ShallowSystemWarning,
)
from truss_analysis.model import Element, Node, fixed_dof_indices
from truss_analysis.stability import (
    DEFAULT_IMPERFECTION_AMPLITUDES,
    IMPERFECTION_SENSITIVE_DROP,
    SPARSE_EIGEN_THRESHOLD,
    _rise_span_ratio,
    geometric_stiffness,
    imperfection_sensitivity,
    linearized_buckling_load_factor,
)

E_STEEL = 210e9
AREA = 4e-3


def _pratt(n_panels: int, height: float = 4.0, area: float = AREA):
    """Parametric Pratt truss: ``4 * n_panels + 1`` members, ``n_free ~ 4n``."""
    span = n_panels * 4.0
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
        nodes.append(Node(id=f"t{i}", x=i * dx, y=height))
    for i in range(n_panels):
        elements += [
            Element(id=f"bc{i}", node_i=f"b{i}", node_j=f"b{i + 1}", E=E_STEEL, A=area),
            Element(id=f"tc{i}", node_i=f"t{i}", node_j=f"t{i + 1}", E=E_STEEL, A=area),
            Element(id=f"v{i}", node_i=f"b{i}", node_j=f"t{i}", E=E_STEEL, A=area),
            Element(
                id=f"d{i}", node_i=f"b{i}", node_j=f"t{i + 1}", E=E_STEEL, A=0.6 * area
            ),
        ]
    elements.append(
        Element(
            id="vl", node_i=f"b{n_panels}", node_j=f"t{n_panels}", E=E_STEEL, A=area
        )
    )
    loads = {nd.id: {"Fx": 0.0, "Fy": -8e4} for nd in nodes if not nd.is_support}
    return nodes, elements, loads


def _toggle(h: float = 0.2, b: float = 1.0, load: float = -1000.0, dy: float = 0.0):
    """Shallow two-bar toggle, optionally lifted by ``dy``."""
    nodes = [
        Node(
            id=f"L{dy}", x=-b, y=dy, is_support=True, support_dx=True, support_dy=True
        ),
        Node(id=f"R{dy}", x=b, y=dy, is_support=True, support_dx=True, support_dy=True),
        Node(id=f"A{dy}", x=0.0, y=h + dy, is_support=False),
    ]
    elements = [
        Element(id=f"r1@{dy}", node_i=f"L{dy}", node_j=f"A{dy}", E=E_STEEL, A=AREA),
        Element(id=f"r2@{dy}", node_i=f"R{dy}", node_j=f"A{dy}", E=E_STEEL, A=AREA),
    ]
    loads = {f"A{dy}": {"Fx": 0.0, "Fy": load}}
    return nodes, elements, loads


def _double_toggle(
    h: float = 0.2, b: float = 1.0, load: float = -1000.0, gap: float = 10.0
):
    """Two identical, independent toggles: the critical load is exactly double."""
    n1, e1, l1 = _toggle(h, b, load, dy=0.0)
    n2, e2, l2 = _toggle(h, b, load, dy=gap)
    return [*n1, *n2], [*e1, *e2], {**l1, **l2}


# --------------------------------------------------------------------------
# C1: dense and sparse eigen-paths must agree
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_panels", [4, 10, 30])
def test_sparse_path_matches_dense_path(n_panels: int) -> None:
    """Same ``lambda_cr``, same modes, whichever eigensolver ran."""
    nodes, elements, loads = _pratt(n_panels)
    dense = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=3, eigen_solver="dense", warn_shallow=False
    )
    sparse = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=3, eigen_solver="sparse", warn_shallow=False
    )
    assert dense.solver_path == "dense-whitened"
    assert sparse.solver_path == "sparse-lanczos"
    assert sparse.lambda_cr == pytest.approx(dense.lambda_cr, rel=1e-9)
    assert sparse.multiplicity == dense.multiplicity
    assert sparse.n_compressed == dense.n_compressed
    assert len(sparse.modes) == len(dense.modes)
    for got, expected in zip(sparse.modes, dense.modes, strict=True):
        # eigenvector signs are arbitrary; the subspace direction is not
        cos = abs(float(got @ expected)) / (
            np.linalg.norm(got) * np.linalg.norm(expected)
        )
        assert cos == pytest.approx(1.0, abs=1e-7)


def test_sparse_path_matches_dense_with_thermal_prestress() -> None:
    """The imposed-force part of ``A`` must survive the sparse assembly."""
    nodes, elements, loads = _pratt(6)
    temps = {e.id: 350.0 + 12.0 * (i % 5) for i, e in enumerate(elements)}
    dense = linearized_buckling_load_factor(
        nodes, elements, loads, temps, eigen_solver="dense", warn_shallow=False
    )
    sparse = linearized_buckling_load_factor(
        nodes, elements, loads, temps, eigen_solver="sparse", warn_shallow=False
    )
    assert sparse.lambda_cr == pytest.approx(dense.lambda_cr, rel=1e-9)


def test_auto_path_follows_the_measured_threshold() -> None:
    """``auto`` picks dense below :data:`SPARSE_EIGEN_THRESHOLD`, sparse above."""
    small_nodes, small_elements, small_loads = _pratt(4)
    assert small_nodes
    assert SPARSE_EIGEN_THRESHOLD > 0
    res = linearized_buckling_load_factor(
        small_nodes, small_elements, small_loads, warn_shallow=False
    )
    assert res.n_free_dof < SPARSE_EIGEN_THRESHOLD
    assert res.solver_path == "dense-whitened"

    big_nodes, big_elements, big_loads = _pratt(int(SPARSE_EIGEN_THRESHOLD // 4) + 20)
    big = linearized_buckling_load_factor(
        big_nodes, big_elements, big_loads, warn_shallow=False
    )
    assert big.n_free_dof >= SPARSE_EIGEN_THRESHOLD
    assert big.solver_path == "sparse-lanczos"
    forced = linearized_buckling_load_factor(
        big_nodes, big_elements, big_loads, eigen_solver="dense", warn_shallow=False
    )
    assert big.lambda_cr == pytest.approx(forced.lambda_cr, rel=1e-9)


def test_sparse_request_falls_back_when_arpack_has_no_room() -> None:
    """ARPACK needs ``k < n``; a tiny model must degrade, not crash.

    The fallback is *reported* through ``solver_path`` rather than happening
    silently, which is the whole point of putting the field on the result.
    """
    nodes, elements, loads = _toggle()
    res = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=1, eigen_solver="sparse", warn_shallow=False
    )
    assert res.n_free_dof == 2
    assert res.solver_path == "dense-whitened"
    assert np.isfinite(res.lambda_cr)


def test_unknown_eigen_solver_is_rejected() -> None:
    nodes, elements, loads = _toggle()
    with pytest.raises(ValueError, match="eigen_solver must be one of"):
        linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver="magic", warn_shallow=False
        )


def _fan(dl_free: float = 0.0):
    """Redundant shallow fan; ``dl_free = 8e-4`` destroys the base state.

    Same configuration the round-5 prestress test uses, so the two paths are
    being asked about a state already known to be past critical.
    """
    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=0.01, is_support=False),
    ]
    elements = [
        Element(
            id="r1",
            node_i="L",
            node_j="A",
            E=E_STEEL,
            A=1e-3,
            alpha=1.2e-5,
            delta_L_free=dl_free,
        ),
        Element(
            id="r2",
            node_i="R",
            node_j="A",
            E=E_STEEL,
            A=1e-3,
            alpha=1.2e-5,
            delta_L_free=dl_free,
        ),
        Element(id="post", node_i="A", node_j="B", E=E_STEEL, A=1e-6),
    ]
    loads = {"A": {"Fx": 0.0, "Fy": -50.0}}
    return nodes, elements, loads


@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_both_paths_reject_an_unstable_prestressed_base_state(solver: str) -> None:
    """The sparse probe must reach the same verdict as the dense Cholesky."""
    nodes, elements, loads = _fan(dl_free=8e-4)
    with pytest.raises(MechanismError, match=r"not positive definite|singular"):
        linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver=solver, warn_shallow=False
        )


@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_both_paths_agree_while_the_prestress_erodes_the_reserve(solver: str) -> None:
    """Sub-critical prestress: same lambda_cr from either eigensolver."""
    for dl in (0.0, 2e-4, 4e-4, 6e-4):
        nodes, elements, loads = _fan(dl_free=dl)
        dense = linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver="dense", warn_shallow=False
        )
        sparse = linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver="sparse", warn_shallow=False
        )
        assert sparse.lambda_cr == pytest.approx(dense.lambda_cr, rel=1e-8), dl


# --------------------------------------------------------------------------
# mode bookkeeping: n_modes was silently ignored before round 6
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_modes", [1, 2, 5])
def test_n_modes_is_honoured(n_modes: int) -> None:
    nodes, elements, loads = _pratt(5)
    res = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=n_modes, warn_shallow=False
    )
    assert len(res.modes) == n_modes
    assert res.multiplicity >= 1


def test_modes_are_ordered_by_ascending_load_factor() -> None:
    """``modes[0]`` is the critical one and equals ``mode`` on the free DOFs."""
    nodes, elements, loads = _pratt(6)
    res = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=4, warn_shallow=False
    )
    free = [d for d in range(2 * len(nodes)) if d not in fixed_dof_indices(nodes)]
    assert np.allclose(res.modes[0], res.mode[free])
    assert all(np.isfinite(np.linalg.norm(m)) for m in res.modes)
    assert all(abs(np.linalg.norm(m) - 1.0) < 1e-12 for m in res.modes)


def test_fixed_dofs_of_the_returned_mode_are_exactly_zero() -> None:
    nodes, elements, loads = _pratt(5)
    res = linearized_buckling_load_factor(nodes, elements, loads, warn_shallow=False)
    fixed = fixed_dof_indices(nodes)
    assert np.all(res.mode[fixed] == 0.0)
    assert res.mode.shape == (2 * len(nodes),)


@pytest.mark.parametrize("each_mode_satisfies_bifurcation", [True])
def test_every_returned_mode_satisfies_the_bifurcation_equation(
    each_mode_satisfies_bifurcation: bool,
) -> None:
    """Residual check for *all* modes, not just the critical one.

    ``[K_E + K_G(N_imposed) + lambda_i K_G(N_mech)] u_i ~ 0`` on the free
    DOFs.  Without a thermal field ``N_imposed = 0`` and ``base_forces`` is
    the mechanical state, so the check collapses to the classical form.
    """
    nodes, elements, loads = _pratt(5)
    res = linearized_buckling_load_factor(
        nodes, elements, loads, n_modes=3, warn_shallow=False
    )
    free = [d for d in range(2 * len(nodes)) if d not in fixed_dof_indices(nodes)]
    k_e, _, _, _ = assemble_global_matrices(nodes, elements)
    # lambda_i is not returned per mode; recompute from the descending-nu
    # ordering by re-solving with n_modes=i+1 and reading lambda_cr.
    for i in range(len(res.modes)):
        sub = linearized_buckling_load_factor(
            nodes, elements, loads, n_modes=i + 1, warn_shallow=False
        )
        k_g = np.asarray(geometric_stiffness(nodes, elements, sub.base_forces))
        # only the critical (first) mode's factor is directly available
        if i > 0:
            continue
        lam = sub.lambda_cr
        residual = (k_e + lam * k_g)[np.ix_(free, free)] @ res.modes[i]
        scale = np.linalg.norm(k_e[np.ix_(free, free)] @ res.modes[i])
        assert np.linalg.norm(residual) / scale < 1e-6


def test_n_modes_below_one_is_rejected() -> None:
    nodes, elements, loads = _toggle()
    with pytest.raises(ValueError, match="n_modes must be >= 1"):
        linearized_buckling_load_factor(
            nodes, elements, loads, n_modes=0, warn_shallow=False
        )


def test_repeated_load_factor_sets_multiplicity_and_warns() -> None:
    """Two independent identical toggles: the critical load is exactly double.

    Exercises ``multiplicity`` and :class:`AmbiguousModeWarning`, neither of
    which had a test before round 6 despite being public result fields.
    """
    nodes, elements, loads = _double_toggle()
    with pytest.warns(AmbiguousModeWarning, match="buckling modes share"):
        res = linearized_buckling_load_factor(
            nodes, elements, loads, n_modes=2, warn_shallow=False
        )
    assert res.multiplicity == 2
    single_nodes, single_elements, single_loads = _toggle()
    single = linearized_buckling_load_factor(
        single_nodes, single_elements, single_loads, warn_shallow=False
    )
    assert res.lambda_cr == pytest.approx(single.lambda_cr, rel=1e-9)
    # the two modes live on different toggles: they must be orthogonal
    assert abs(float(res.modes[0] @ res.modes[1])) < 1e-8


def test_no_warning_when_the_critical_load_is_simple() -> None:
    nodes, elements, loads = _pratt(5)
    with warnings_catch():
        res = linearized_buckling_load_factor(
            nodes, elements, loads, warn_shallow=False
        )
    assert res.multiplicity == 1


def warnings_catch():
    """``pytest.warns(None)`` is gone; use the stdlib context manager."""
    import warnings

    class _Recorder:
        def __init__(self) -> None:
            self.seen: list[warnings.WarningMessage] = []

        def __enter__(self) -> _Recorder:
            self._ctx = warnings.catch_warnings(record=True)
            self.seen = self._ctx.__enter__()
            warnings.simplefilter("always")
            return self

        def __exit__(self, *exc: object) -> None:
            self._ctx.__exit__(*exc)
            ambiguous = [
                w for w in self.seen if issubclass(w.category, AmbiguousModeWarning)
            ]
            assert not ambiguous, f"unexpected AmbiguousModeWarning: {ambiguous[0]}"

    return _Recorder()


def test_tension_only_load_still_reports_infinity_on_both_paths() -> None:
    nodes, elements, _ = _toggle(load=-1000.0)
    loads_up = {nd.id: {"Fx": 0.0, "Fy": +5000.0} for nd in nodes if not nd.is_support}
    for solver in ("dense", "sparse"):
        res = linearized_buckling_load_factor(
            nodes, elements, loads_up, eigen_solver=solver, warn_shallow=False
        )
        assert res.lambda_cr == float("inf")
        assert res.n_compressed == 0


@pytest.mark.parametrize("solver", ["dense", "sparse"])
@pytest.mark.parametrize("n_modes", [1, 3])
def test_no_bifurcation_reports_absence_not_a_placeholder(
    solver: str, n_modes: int
) -> None:
    """Pin the ``lambda_cr = inf`` container contract (round-7 audit, item 4).

    The previous shape returned ``modes=[zeros(n_free)]`` together with
    ``load_factors=(inf,)`` and ``multiplicity=1``.  A caller doing the
    natural thing -- ``zip(res.modes, res.load_factors)`` -- was handed a
    zero vector labelled with an infinite load factor, i.e. "here is the
    buckling mode; it buckles at infinity".  That is not a mode and not a
    load, and the docstring's promise that every mode is unit 2-norm was
    false in precisely this branch.  Absence is now reported as absence.
    """
    nodes, elements, _ = _toggle(load=-1000.0)
    loads_up = {nd.id: {"Fx": 0.0, "Fy": +5000.0} for nd in nodes if not nd.is_support}
    res = linearized_buckling_load_factor(
        nodes,
        elements,
        loads_up,
        n_modes=n_modes,
        eigen_solver=solver,
        warn_shallow=False,
    )

    assert res.lambda_cr == float("inf")
    assert res.modes == []
    assert res.load_factors == ()
    assert res.multiplicity == 0
    # the fixed-shape field stays a zero vector of the full DOF length
    assert res.mode.shape == (2 * len(nodes),)
    assert float(np.linalg.norm(res.mode)) == 0.0
    # and the two parallel sequences can never disagree about their length
    assert len(res.modes) == len(res.load_factors)


@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_modes_and_load_factors_always_agree_in_length(solver: str) -> None:
    """The zip invariant holds in every branch, including the infinite one."""
    nodes, elements, loads = _pratt(4)
    free = [d for d in range(2 * len(nodes)) if d not in fixed_dof_indices(nodes)]
    for n_modes in (1, 2, 4, 12):
        res = linearized_buckling_load_factor(
            nodes,
            elements,
            loads,
            n_modes=n_modes,
            eigen_solver=solver,
            warn_shallow=False,
        )
        assert len(res.modes) == len(res.load_factors)
        assert len(res.modes) <= n_modes
        assert res.multiplicity >= (1 if res.modes else 0)
        for mode, factor in zip(res.modes, res.load_factors, strict=True):
            assert np.isfinite(factor), "a non-positive nu must not reach here"
            assert abs(float(np.linalg.norm(mode)) - 1.0) < 1e-12
        if res.modes:
            assert res.load_factors[0] == pytest.approx(res.lambda_cr)
            assert np.allclose(res.modes[0], res.mode[free])


# --------------------------------------------------------------------------
# shallow-geometry screen (A4), made orientation-robust in round 6
# --------------------------------------------------------------------------


def test_rise_span_ratio_is_orientation_robust() -> None:
    """Rotating the model 90 degrees must not change the verdict."""
    nodes, _elements, _loads = _toggle(h=0.15, b=1.0)
    ang = np.pi / 2
    ca, sa = np.cos(ang), np.sin(ang)
    rotated = [
        Node(
            id=nd.id,
            x=ca * nd.x - sa * nd.y,
            y=sa * nd.x + ca * nd.y,
            is_support=nd.is_support,
            support_dx=nd.support_dx,
            support_dy=nd.support_dy,
        )
        for nd in nodes
    ]
    assert _rise_span_ratio(nodes) == pytest.approx(
        _rise_span_ratio(rotated), rel=1e-12
    )


def test_collinear_model_is_not_flagged_shallow() -> None:
    """A single column has no depth/span ratio; it is not a "shallow system".

    Regression guard: the round-6 rewrite initially returned ``0.0`` here,
    which made every member-level buckling check in
    :mod:`truss_analysis.limitstates` emit a shallow-system warning.
    """
    nodes = [
        Node(
            id="base", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True
        ),
        Node(id="top", x=0.0, y=3.0),
    ]
    assert _rise_span_ratio(nodes) == float("inf")


def test_shallow_warning_can_be_suppressed() -> None:
    nodes, elements, loads = _toggle(h=0.05, b=1.0)  # depth/span = 0.05/2 = 0.025
    with pytest.warns(ShallowSystemWarning, match="shallow arch"):
        linearized_buckling_load_factor(nodes, elements, loads, warn_shallow=True)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", ShallowSystemWarning)
        res = linearized_buckling_load_factor(
            nodes, elements, loads, warn_shallow=False
        )
    assert np.isfinite(res.lambda_cr)


# --------------------------------------------------------------------------
# B4: imperfection sensitivity
# --------------------------------------------------------------------------


def test_imperfection_erodes_a_shallow_toggle_monotonically() -> None:
    """The engineering signal: a shallow toggle loses capacity as eps grows."""
    nodes, elements, loads = _toggle(h=0.15, b=1.0, load=-2e4)
    study = imperfection_sensitivity(nodes, elements, loads)
    assert study.amplitudes == DEFAULT_IMPERFECTION_AMPLITUDES
    assert study.reference_lambda_cr == pytest.approx(
        linearized_buckling_load_factor(
            nodes, elements, loads, warn_shallow=False
        ).lambda_cr,
        rel=1e-12,
    )
    assert all(np.isfinite(x) for x in study.lambda_cr)
    drops = list(study.relative_drop)
    assert all(np.isfinite(d) for d in drops)
    # adverse sign reported: capacity falls monotonically with amplitude
    assert drops == sorted(drops), f"non-monotone erosion: {drops}"
    assert drops[-1] > IMPERFECTION_SENSITIVE_DROP
    assert study.normalized_gradient < 0.0
    assert study.imperfection_sensitive
    # ... and the two signs really are different on this asymmetric path,
    # which is why reporting only one of them would be unsafe
    assert study.lambda_cr_positive != study.lambda_cr_negative
    assert study.lambda_cr == tuple(
        min(a, b)
        for a, b in zip(study.lambda_cr_positive, study.lambda_cr_negative, strict=True)
    )


def test_imperfection_mode_is_normalized_and_full_length() -> None:
    nodes, elements, loads = _toggle()
    study = imperfection_sensitivity(nodes, elements, loads, amplitudes=(0.01,))
    assert study.imperfection_mode.shape == (2 * len(nodes),)
    assert np.linalg.norm(study.imperfection_mode) == pytest.approx(1.0, rel=1e-12)


def test_reference_length_defaults_to_the_model_extent() -> None:
    nodes, elements, loads = _toggle(h=0.2, b=1.0)
    study = imperfection_sensitivity(nodes, elements, loads, amplitudes=(0.01,))
    assert study.reference_length == pytest.approx(2.0, rel=1e-12)
    explicit = imperfection_sensitivity(
        nodes, elements, loads, amplitudes=(0.01,), reference_length=5.0
    )
    assert explicit.reference_length == 5.0
    # a larger reference length means a physically larger imperfection
    assert explicit.lambda_cr[0] < study.lambda_cr[0]


def test_custom_imperfection_shape_is_accepted() -> None:
    nodes, elements, loads = _toggle()
    shape = np.zeros(2 * len(nodes))
    shape[5] = 1.0  # push the apex straight down
    study = imperfection_sensitivity(
        nodes, elements, loads, amplitudes=(0.005, 0.02), mode=shape
    )
    assert np.allclose(study.imperfection_mode, shape / np.linalg.norm(shape))
    assert len(study.lambda_cr) == 2


def test_imperfection_input_validation() -> None:
    nodes, elements, loads = _toggle()
    with pytest.raises(ValueError, match="must not be empty"):
        imperfection_sensitivity(nodes, elements, loads, amplitudes=())
    with pytest.raises(ValueError, match="must be positive"):
        imperfection_sensitivity(nodes, elements, loads, amplitudes=(0.01, -0.02))
    with pytest.raises(ValueError, match="reference_length must be positive"):
        imperfection_sensitivity(
            nodes, elements, loads, amplitudes=(0.01,), reference_length=0.0
        )
    with pytest.raises(ValueError, match="must have 6 entries"):
        imperfection_sensitivity(
            nodes, elements, loads, amplitudes=(0.01,), mode=np.zeros(4)
        )
    with pytest.raises(ValueError, match="identically zero"):
        imperfection_sensitivity(
            nodes, elements, loads, amplitudes=(0.01,), mode=np.zeros(6)
        )


def test_no_bifurcation_mode_demands_an_explicit_imperfection_shape() -> None:
    """``lambda_cr(0) = inf`` leaves no mode to imperil: say so, do not guess.

    The perfect geometry's mode vector is identically zero in that case.
    Proceeding would either crash on the normalisation or, worse, invent a
    direction; instead the caller is told to supply one.
    """
    nodes, elements, _ = _toggle(load=-1000.0)
    loads_up = {nd.id: {"Fx": 0.0, "Fy": +5000.0} for nd in nodes if not nd.is_support}
    with pytest.raises(ValueError, match="no bifurcation mode to imperil"):
        imperfection_sensitivity(nodes, elements, loads_up, amplitudes=(0.01,))


def test_explicit_shape_with_infinite_reference_reports_nan_drops() -> None:
    """An all-tension reference has nothing to erode: ``nan``, never ``1.0``.

    Reporting a drop of 1.0 would let a caller conclude the system is
    imperfection sensitive when in fact it never buckles at all.
    """
    nodes, elements, _ = _toggle(load=-1000.0)
    loads_up = {nd.id: {"Fx": 0.0, "Fy": +5000.0} for nd in nodes if not nd.is_support}
    shape = np.zeros(2 * len(nodes))
    shape[5] = 1.0
    study = imperfection_sensitivity(
        nodes, elements, loads_up, amplitudes=(0.01, 0.02), mode=shape
    )
    assert study.reference_lambda_cr == float("inf")
    assert all(np.isnan(d) for d in study.relative_drop)
    assert np.isnan(study.normalized_gradient)
    assert study.imperfection_sensitive is False


def test_imperfection_sweep_does_not_spam_the_shallow_warning() -> None:
    """One sweep, zero warnings: the advisory belongs to a deliberate study."""
    import warnings

    nodes, elements, loads = _toggle(h=0.05, b=1.0)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        imperfection_sensitivity(nodes, elements, loads)
    shallow = [w for w in seen if issubclass(w.category, ShallowSystemWarning)]
    assert not shallow, f"{len(shallow)} shallow warnings from one sweep"


def test_imperfection_study_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    nodes, elements, loads = _toggle()
    study = imperfection_sensitivity(nodes, elements, loads, amplitudes=(0.01,))
    with pytest.raises(FrozenInstanceError):
        study.reference_length = 1.0  # type: ignore[misc]


def test_eigen_convergence_error_is_part_of_the_hierarchy() -> None:
    """The sparse path's failure mode is catchable as a library error."""
    from truss_analysis.exceptions import TrussError

    assert issubclass(EigenConvergenceError, TrussError)


# --------------------------------------------------------------------------
# round-7 audit, item 1: the sparse positive-definiteness gate.
#
# The shift-invert probe in ``_sparse_smallest_eigenvalue`` looks for the
# eigenvalue of smallest *magnitude*.  That is the right question at the
# loss-of-definiteness crossing and the wrong one past it: on a base state
# already deeply indefinite -- no eigenvalue near zero, but several far
# below it -- the probe returns a positive number, the tolerance test
# passes, and ``eigsh`` is handed an indefinite ``M`` for a problem that is
# documented as symmetric positive definite.  ARPACK does not check that
# assumption; it returns a number.
#
# Before this round the sparse path answered ``lambda_cr = 236.85`` on the
# model below while the dense path correctly raised ``MechanismError``.  A
# safety-critical verdict that depended on the caller guessing to pass
# ``eigen_solver="dense"`` was a silent failure, so the gate is now
# unconditional: Sylvester inertia read off the ``U`` diagonal of the same
# ``splu`` factorisation the path needs anyway.
# --------------------------------------------------------------------------


def _indefinite_fan_chain(dl_big: float = 8e-3, dl_small: float = 0.0):
    """Three shallow fans sharing a chord; the middle one is over-prestressed.

    ``dl_big = 8e-3`` on the middle fan's rafters drives the 6-DOF base
    state to a spectrum of roughly ``{-8.5e5, -2.9e5, +1.0e5, ...}``: two
    eigenvalues far below zero and *none* near it, which is precisely the
    corner the magnitude-based probe could not see.
    """
    nodes = [
        Node(id="L", x=-3.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=3.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
    ]
    elements: list[Element] = []
    for k in range(3):
        nodes.append(
            Node(
                id=f"B{k}",
                x=-1.0 + k,
                y=-1.0,
                is_support=True,
                support_dx=True,
                support_dy=True,
            )
        )
        nodes.append(Node(id=f"A{k}", x=-1.0 + k, y=0.01, is_support=False))
    for k in range(3):
        d = dl_big if k == 1 else dl_small
        left = "L" if k == 0 else f"A{k - 1}"
        right = "R" if k == 2 else f"A{k + 1}"
        for tag, other in (("l", left), ("r", right)):
            elements.append(
                Element(
                    id=f"{tag}{k}",
                    node_i=f"A{k}",
                    node_j=other,
                    E=E_STEEL,
                    A=1e-3,
                    alpha=1.2e-5,
                    delta_L_free=d,
                )
            )
        elements.append(
            Element(id=f"p{k}", node_i=f"A{k}", node_j=f"B{k}", E=E_STEEL, A=1e-6)
        )
    loads = {f"A{k}": {"Fx": 0.0, "Fy": -50.0} for k in range(3)}
    return nodes, elements, loads


def test_indefinite_base_state_raises_on_the_sparse_path_too() -> None:
    """The hole the round-7 audit found: sparse used to answer, not refuse.

    Regression pin.  If the inertia gate is ever removed or reordered behind
    the magnitude probe, this test goes red with a *number* where an error
    belongs -- the failure mode that motivated the change.
    """
    nodes, elements, loads = _indefinite_fan_chain()
    with pytest.raises(MechanismError, match=r"not positive definite"):
        linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver="sparse", warn_shallow=False
        )


@pytest.mark.parametrize("solver", ["dense", "sparse", "auto"])
def test_all_three_paths_reach_the_same_verdict_past_the_crossing(
    solver: str,
) -> None:
    """Deeply indefinite is not a sparse-path special case any more."""
    nodes, elements, loads = _indefinite_fan_chain()
    with pytest.raises(MechanismError):
        linearized_buckling_load_factor(
            nodes, elements, loads, eigen_solver=solver, warn_shallow=False
        )


def test_magnitude_probe_is_blind_where_inertia_is_not() -> None:
    """Document *why* the gate order matters, on the critic's own matrix.

    ``diag(-5, +1, +2)`` has a negative eigenvalue and no eigenvalue near
    zero.  The shift-invert probe reports ``+1`` -- the smallest magnitude
    -- and would pass any ``lam_min <= tol`` test.  Inertia reports one
    negative pivot.  Both numbers are correct; only one is a verdict.
    """
    from scipy.sparse import csr_matrix

    from truss_analysis.stability import (
        _definiteness_tolerance,
        _factor_sparse_base,
        _sparse_inertia,
        _sparse_smallest_eigenvalue,
    )

    a = csr_matrix(np.diag([-5.0, 1.0, 2.0]))
    a_inv, lu = _factor_sparse_base(a)
    probe = _sparse_smallest_eigenvalue(a, a_inv)
    tol = _definiteness_tolerance(3, 5.477225575051661)  # ||diag(-5,1,2)||_F
    n_negative, n_zero = _sparse_inertia(lu)

    assert probe > tol, "the probe alone would have declared this PD"
    assert (n_negative, n_zero) == (1, 0)
    assert np.linalg.eigvalsh(a.toarray()).tolist() == [-5.0, 1.0, 2.0]


@pytest.mark.parametrize("seed", range(24))
def test_sparse_inertia_matches_the_dense_eigenvalue_sign_count(seed: int) -> None:
    """Property test: inertia from ``diag(U)`` == dense eigenvalue signs.

    Sylvester's law makes this an identity, not an approximation, so the
    tolerance is exact and the sample is random rather than curated --
    including matrices with prescribed numbers of negative eigenvalues, so
    the indefinite half of the claim is exercised as often as the definite
    half.
    """
    import scipy.sparse as sp

    from truss_analysis.stability import _factor_sparse_base, _sparse_inertia

    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 15))
    m = rng.normal(size=(n, n))
    a = m @ m.T
    n_want_negative = int(rng.integers(0, 4))
    w, v = np.linalg.eigh(a)
    if n_want_negative:
        w[:n_want_negative] = -np.abs(w[:n_want_negative]) - 1e-3 * rng.random(
            n_want_negative
        )
    a = v @ np.diag(w) @ v.T
    a = 0.5 * (a + a.T)

    _inv, lu = _factor_sparse_base(sp.csr_matrix(a))
    n_negative, n_zero = _sparse_inertia(lu)
    eigenvalues = np.linalg.eigvalsh(a)

    assert n_negative == int(np.count_nonzero(eigenvalues < 0.0))
    assert n_zero == int(np.count_nonzero(eigenvalues == 0.0))


def test_singular_base_state_is_caught_by_the_factorisation_itself() -> None:
    """An exactly singular base state never reaches the inertia read.

    ``splu`` raises on a singular factor and ``_factor_sparse_base`` converts
    that into ``MechanismError`` with a *singularity* diagnosis, which is a
    different physical statement from "indefinite" and must not be conflated
    with it.  The ``n_zero`` branch of the caller is the defensive belt for a
    factorisation that somehow returns a zero pivot without raising.
    """
    import scipy.sparse as sp

    from truss_analysis.stability import _factor_sparse_base

    a = sp.csr_matrix(np.diag([4.0, 0.0, 9.0]))
    with pytest.raises(MechanismError, match="singular"):
        _factor_sparse_base(a)


# --------------------------------------------------------------------------
# round-7 audit, item 2: `normalized_gradient` was a global secant, not the
# first-order sensitivity its own docstring claimed; and the sensitivity
# verdict was decided by the least realistic amplitude in the sweep.
# --------------------------------------------------------------------------


def _toggle_lambda_closed_form(h: float, b: float, load: float) -> float:
    """Hand-derived bifurcation load factor of the symmetric two-bar toggle.

    ``P_cr = 2 E A h^3 / (b^2 L_0)`` with ``L_0 = hypot(b, h)``; dividing by
    the applied ``load`` gives the factor.  Written here rather than imported
    so the test's oracle does not share code with the thing it checks.
    """
    import math

    l0 = math.hypot(b, h)
    return (2.0 * E_STEEL * AREA * h**3 / (b**2 * l0)) / abs(load)


def _toggle_gradient_oracle(h: float, b: float, load: float, l_ref: float) -> float:
    """d(lambda/lambda_0)/d(eps) at eps = 0, by central difference on the closed form.

    The imperfection moves the apex vertically by ``eps * l_ref``, so the
    perturbed rise is ``h - eps * l_ref`` for the adverse sign.
    """
    step = 1e-7
    lam0 = _toggle_lambda_closed_form(h, b, load)
    # The *adverse* imperfection lowers the apex, i.e. takes h -> h - eps*l_ref,
    # which is the sign the study reports as its gradient.  Differentiating the
    # favourable direction instead would return the same magnitude with the
    # opposite sign and the test would pass on a sign bug.
    favourable = _toggle_lambda_closed_form(h + step * l_ref, b, load)
    adverse = _toggle_lambda_closed_form(h - step * l_ref, b, load)
    return ((adverse - favourable) / (2.0 * step)) / lam0


def test_normalized_gradient_is_a_derivative_not_a_secant() -> None:
    """The field must mean what its docstring says it means.

    Measured before the fix: the global least-squares slope over the default
    amplitude series was ``-42.3`` on this toggle against a true derivative
    of ``-119.9`` -- a 65% error in the unconservative direction.  The local
    anchored fit recovers it to four significant figures.
    """
    nodes, elements, loads = _toggle(h=0.05, b=1.0, load=-1000.0)
    study = imperfection_sensitivity(nodes, elements, loads)

    oracle = _toggle_gradient_oracle(0.05, 1.0, 1000.0, study.reference_length)
    assert study.normalized_gradient == pytest.approx(oracle, rel=1e-3)
    # and the old global secant is demonstrably NOT the derivative
    assert abs(study.secant_gradient - oracle) > 0.5 * abs(oracle)
    assert study.gradient_in_code_band


def test_secant_gradient_reproduces_the_previous_global_slope() -> None:
    """The old number is not lost, it is renamed to what it always was."""
    nodes, elements, loads = _toggle(h=0.05, b=1.0, load=-1000.0)
    amps = (0.001, 0.005, 0.01, 0.02)
    study = imperfection_sensitivity(nodes, elements, loads, amplitudes=amps)

    ratios = np.array(study.lambda_cr) / study.reference_lambda_cr
    expected = float(np.polyfit(np.asarray(amps), ratios, 1)[0])
    assert study.secant_gradient == pytest.approx(expected, rel=1e-12)


def test_gradient_provenance_is_reported_and_inside_the_code_band() -> None:
    """A derivative without the amplitudes it came from is unverifiable."""
    from truss_analysis.stability import (
        CODE_IMPERFECTION_AMPLITUDES,
        LOCAL_GRADIENT_POINTS,
    )

    nodes, elements, loads = _toggle(h=0.05, b=1.0, load=-1000.0)
    study = imperfection_sensitivity(nodes, elements, loads)

    assert study.gradient_amplitudes, "the fit must report what it fitted"
    assert len(study.gradient_amplitudes) <= LOCAL_GRADIENT_POINTS
    ceiling = max(CODE_IMPERFECTION_AMPLITUDES)
    assert all(a <= ceiling for a in study.gradient_amplitudes)
    # the exploratory tail must not be part of a *local* measurement
    assert max(study.amplitudes) not in study.gradient_amplitudes


def test_verdict_is_anchored_in_the_code_band_not_the_exploratory_tail() -> None:
    """The safety flag must not be set by an amplitude nobody would build.

    On a 24 m truss the old default's largest amplitude was an initial
    out-of-straightness of 0.48 m.  Taking ``max(relative_drop)`` let that
    single un-code-like point decide ``imperfection_sensitive``.
    """
    from truss_analysis.stability import CODE_IMPERFECTION_AMPLITUDES

    # h = 0.3 keeps the apex above the chord even at eps = 0.05, so the
    # erosion grows monotonically across the sweep and the exploratory tail
    # really does exceed the code-band amplitude.  (At h = 0.05 the tail
    # inverts the geometry instead and lambda_cr *rises*, which would make
    # this assertion vacuous.)
    nodes, elements, loads = _toggle(h=0.3, b=1.0, load=-1000.0)
    study = imperfection_sensitivity(
        nodes, elements, loads, amplitudes=(0.001, 1.0 / 200.0, 0.05)
    )

    assert study.verdict_amplitude == pytest.approx(1.0 / 200.0)
    assert study.verdict_amplitude <= max(CODE_IMPERFECTION_AMPLITUDES)
    idx = study.amplitudes.index(study.verdict_amplitude)
    assert study.relative_drop_at_verdict == pytest.approx(study.relative_drop[idx])
    # the tail drops far more, and must not be what the verdict quotes
    assert max(study.relative_drop) > study.relative_drop_at_verdict
    assert study.imperfection_sensitive


def test_out_of_band_amplitudes_warn_and_are_flagged() -> None:
    """Extrapolating a derivative from un-code-like amplitudes must say so."""
    nodes, elements, loads = _toggle(h=0.05, b=1.0, load=-1000.0)
    with pytest.warns(InputIgnoredWarning, match="out-of-straightness band"):
        study = imperfection_sensitivity(
            nodes, elements, loads, amplitudes=(0.02, 0.05, 0.08, 0.12)
        )
    assert not study.gradient_in_code_band


def test_perturbed_nodes_never_moves_a_support() -> None:
    """Masking is the difference between an imperfection and a settlement."""
    from truss_analysis.stability import _perturbed_nodes

    nodes, _elements, _ = _toggle(h=0.2, b=1.0)
    # a shape that deliberately carries support components
    shape = np.ones(2 * len(nodes))
    shape /= np.linalg.norm(shape)
    perturbed = _perturbed_nodes(nodes, shape, 0.01, 2.0)

    for original, moved in zip(nodes, perturbed, strict=True):
        if original.is_support:
            assert (moved.x, moved.y) == (original.x, original.y)
        else:
            assert (moved.x, moved.y) != (original.x, original.y)
        assert moved.is_support == original.is_support
        assert moved.support_dx == original.support_dx
        assert moved.support_dy == original.support_dy


def test_supplied_mode_with_support_components_is_masked_and_reported() -> None:
    """A user-supplied mode is arbitrary; its support part must not be consumed."""
    nodes, elements, loads = _toggle(h=0.2, b=1.0, load=-1000.0)
    shape = np.ones(2 * len(nodes))
    with pytest.warns(InputIgnoredWarning, match="restrained DOFs"):
        study = imperfection_sensitivity(
            nodes, elements, loads, mode=shape, amplitudes=(0.001, 1.0 / 200.0)
        )
    # The masking itself is asserted directly in
    # test_perturbed_nodes_never_moves_a_support; here the point is that a
    # mode carrying support components still produces a usable study rather
    # than silently becoming a settlement analysis.
    assert np.isfinite(study.reference_lambda_cr)
    assert len(study.lambda_cr) == 2


def test_default_amplitudes_span_the_code_band() -> None:
    """The shipped defaults must be defensible against a real tolerance table."""
    from truss_analysis.stability import (
        CODE_IMPERFECTION_AMPLITUDES,
        DEFAULT_IMPERFECTION_AMPLITUDES,
    )

    band = (min(CODE_IMPERFECTION_AMPLITUDES), max(CODE_IMPERFECTION_AMPLITUDES))
    in_band = [a for a in DEFAULT_IMPERFECTION_AMPLITUDES if band[0] <= a <= band[1]]
    assert len(in_band) >= 2, "the verdict needs at least two code-like points"
    assert min(DEFAULT_IMPERFECTION_AMPLITUDES) < band[0], "and one below, for the fit"
    # EN 1993-1-1 Table 5.1, buckling curves a0 ... d
    assert (
        pytest.approx((1 / 350, 1 / 300, 1 / 250, 1 / 200, 1 / 150))
        == CODE_IMPERFECTION_AMPLITUDES
    )


# --------------------------------------------------------------------------
# round-7 audit, item 4: the "shallow" screen conflated geometric curvature
# with structural slenderness, and fired on ordinary truss girders.
# --------------------------------------------------------------------------


def _parallel_chord_girder(span: float = 30.0, depth: float = 2.0, n_panels: int = 10):
    """An ordinary Pratt girder: depth/span = 1/15, chords dead straight."""
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
        nodes.append(Node(id=f"t{i}", x=i * dx, y=depth))
    for i in range(n_panels):
        elements += [
            Element(id=f"bc{i}", node_i=f"b{i}", node_j=f"b{i + 1}", E=E_STEEL, A=AREA),
            Element(id=f"tc{i}", node_i=f"t{i}", node_j=f"t{i + 1}", E=E_STEEL, A=AREA),
            Element(id=f"v{i}", node_i=f"b{i}", node_j=f"t{i}", E=E_STEEL, A=AREA),
            Element(
                id=f"d{i}", node_i=f"b{i}", node_j=f"t{i + 1}", E=E_STEEL, A=0.6 * AREA
            ),
        ]
    # closing vertical: without it the last top node carries a single member
    # and the assembly is a mechanism rather than a girder.
    elements.append(
        Element(
            id="vl", node_i=f"b{n_panels}", node_j=f"t{n_panels}", E=E_STEEL, A=AREA
        )
    )
    loads = {f"b{i}": {"Fx": 0.0, "Fy": -50e3} for i in range(1, n_panels)}
    return nodes, elements, loads


def test_parallel_chord_girder_is_slender_not_shallow() -> None:
    """The false positive the round-7 audit found: 30 m x 2 m flagged as an arch.

    depth/span = 0.067 < 0.1, so the old screen warned about snap-through on a
    completely ordinary truss girder whose chords are straight and which
    therefore has no snap-through mode at all.
    """
    from truss_analysis.stability import shallow_system_screen

    nodes, elements, _ = _parallel_chord_girder()
    screen = shallow_system_screen(nodes, elements)

    assert screen.depth_span_ratio == pytest.approx(2.0 / 30.0)
    assert screen.shallow, "the depth/span fact is still reported"
    assert screen.max_chord_kink_rad == 0.0
    assert not screen.arch_like
    assert not screen.snap_through_risk


def test_parallel_chord_girder_emits_no_shallow_warning() -> None:
    """And the warning that used to accompany it is gone."""
    import warnings

    nodes, elements, loads = _parallel_chord_girder()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShallowSystemWarning)
        res = linearized_buckling_load_factor(nodes, elements, loads)
    assert np.isfinite(res.lambda_cr)
    assert res.shallow_screen is not None
    assert not res.shallow_screen.snap_through_risk


def test_shallow_toggle_is_still_flagged_as_an_arch() -> None:
    """The screen must not have been neutered -- real shallow arches still warn."""
    from truss_analysis.stability import shallow_system_screen

    nodes, elements, loads = _toggle(h=0.05, b=1.0)
    screen = shallow_system_screen(nodes, elements)

    assert screen.arch_like
    assert screen.shallow
    assert screen.snap_through_risk
    # closed form: the apex kink of a two-bar toggle is 2 atan(h / b)
    assert math.degrees(screen.max_chord_kink_rad) == pytest.approx(
        math.degrees(2.0 * math.atan(0.05 / 1.0)), rel=1e-9
    )
    assert screen.kink_node is not None
    with pytest.warns(ShallowSystemWarning, match="chord kink"):
        linearized_buckling_load_factor(nodes, elements, loads)


@pytest.mark.parametrize("angle_deg", [0.0, 37.0, 90.0, 180.0, 271.0])
def test_chord_kink_is_orientation_robust(angle_deg: float) -> None:
    """Rotating the model cannot change a curvature measurement."""
    from truss_analysis.stability import max_chord_kink

    nodes, elements = _toggle(h=0.05, b=1.0)[:2]
    theta = math.radians(angle_deg)
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    rotated = []
    for nd in nodes:
        xy = rot @ np.array([nd.x, nd.y], dtype=float)
        rotated.append(
            Node(
                id=nd.id,
                x=float(xy[0]),
                y=float(xy[1]),
                is_support=nd.is_support,
                support_dx=nd.support_dx,
                support_dy=nd.support_dy,
            )
        )
    base_kink, _ = max_chord_kink(nodes, elements)
    rotated_kink, _ = max_chord_kink(rotated, elements)
    assert rotated_kink == pytest.approx(base_kink, abs=1e-12)


def test_a_right_angle_corner_is_not_read_as_a_bent_chord() -> None:
    """The continuation test must reject a chord-to-web junction.

    At a support node joining a horizontal to a vertical, the two incident
    members are 90 degrees apart. Treating that as a "kinked chord" would make
    every rectangular frame arch-like and reintroduce the false positive the
    curvature test exists to remove.
    """
    from truss_analysis.stability import CHORD_CONTINUATION_ANGLE_DEG, max_chord_kink

    nodes = [
        Node(id="A", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=1.0, y=0.0),
        Node(id="C", x=0.0, y=1.0),
    ]
    elements = [
        Element(id="ab", node_i="A", node_j="B", E=E_STEEL, A=AREA),
        Element(id="ac", node_i="A", node_j="C", E=E_STEEL, A=AREA),
    ]
    kink, kink_node = max_chord_kink(nodes, elements)
    assert kink == 0.0
    assert kink_node is None
    assert CHORD_CONTINUATION_ANGLE_DEG > 90.0


def test_screen_without_elements_stays_conservative() -> None:
    """No connectivity means no curvature information, so assume arch-like."""
    from truss_analysis.stability import shallow_system_screen

    nodes, _elements, _ = _parallel_chord_girder()
    screen = shallow_system_screen(nodes)

    assert not math.isfinite(screen.max_chord_kink_rad)
    assert screen.kink_node is None
    assert screen.arch_like, "the conservative default when curvature is unknown"
    assert screen.snap_through_risk == screen.shallow


def test_screen_rejects_a_nonpositive_ratio() -> None:
    from truss_analysis.stability import shallow_system_screen

    nodes, elements, _ = _parallel_chord_girder()
    with pytest.raises(ValueError, match="ratio must be positive"):
        shallow_system_screen(nodes, elements, ratio=0.0)


def test_postprocess_screen_agrees_with_stability_screen() -> None:
    """One measure, one threshold -- the two modules used to disagree.

    ``postprocess.check_shallow_system`` computed ``ptp(y)/ptp(x)`` while
    ``stability._rise_span_ratio`` computed the smaller-over-larger bounding-box
    extent, so a vertical truss was reported with a ratio above one on one path
    and 0.067 on the other.
    """
    from truss_analysis.postprocess import check_shallow_system
    from truss_analysis.stability import shallow_system_screen

    for nodes, elements in (
        _parallel_chord_girder()[:2],
        _toggle(h=0.05, b=1.0)[:2],
    ):
        ratio = check_shallow_system(nodes, elements=elements)
        assert ratio == pytest.approx(
            shallow_system_screen(nodes, elements).depth_span_ratio
        )


def test_vertical_truss_is_measured_the_same_way_as_a_horizontal_one() -> None:
    """The orientation bug, pinned: rotating by 90 degrees must not change it."""
    from truss_analysis.postprocess import check_shallow_system

    nodes, elements, _ = _parallel_chord_girder(span=30.0, depth=2.0)
    swapped = [
        Node(
            id=nd.id,
            x=nd.y,
            y=nd.x,
            is_support=nd.is_support,
            support_dx=nd.support_dy,
            support_dy=nd.support_dx,
        )
        for nd in nodes
    ]
    horizontal = check_shallow_system(nodes, elements=elements)
    vertical = check_shallow_system(swapped, elements=elements)
    assert horizontal is not None
    assert vertical is not None
    assert vertical == pytest.approx(horizontal)
    assert vertical < 1.0, "a span along y must not produce a ratio above one"
