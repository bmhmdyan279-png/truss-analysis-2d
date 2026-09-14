"""Cross-path invariants for the thermal demand chain (round-4 audit).

Critic 5's structural point: the library runs two independent assembly
engines -- ``run()`` through :mod:`truss_analysis.assembly` (dense 4x4 block
loop) and the DCR/CI chain through :mod:`truss_analysis.criticality.engine`
(rank-1 dyads + einsum). Shared primitives only help if every consumer
actually reads the same demand, so the agreement is pinned here as an
invariant on a redundant heated model rather than left to each path's own
unit tests.
"""

from __future__ import annotations

import pytest

from truss_analysis.criticality.criteria import reaction_influence
from truss_analysis.criticality.engine import (
    base_displacement,
    build_engine,
    total_load_vector,
)
from truss_analysis.limitstates import dcr_field, member_axial_forces
from truss_analysis.main import run
from truss_analysis.material.steel_eurocode import k_E as eurocode_k_E
from truss_analysis.model import Element, Node
from truss_analysis.sections import euler_buckling_load, non_dimensional_slenderness

F_Y = 235.0e6
E_STEEL = 210.0e9
ALPHA = 1.2e-5
T_AMB = 20.0


def _redundant_heated_model(delta_T: float = 400.0):
    """Both supports fully fixed -> restrained expansion is real demand."""
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=4.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="3", x=2.0, y=1.5, is_support=False),
    ]
    elements = [
        Element(
            id="1",
            node_i="1",
            node_j="3",
            E=E_STEEL,
            A=0.005,
            I_sec=1e-6,
            alpha=ALPHA,
            delta_T=delta_T,
        ),
        Element(
            id="2",
            node_i="2",
            node_j="3",
            E=E_STEEL,
            A=0.008,
            I_sec=1e-6,
            alpha=ALPHA,
            delta_T=delta_T,
        ),
        Element(
            id="3",
            node_i="1",
            node_j="2",
            E=E_STEEL,
            A=0.006,
            I_sec=1e-6,
            alpha=ALPHA,
            delta_T=delta_T,
            delta_L_free=-2e-4,
        ),
    ]
    loads = {"3": {"Fx": 20e3, "Fy": -50e3}}
    return nodes, elements, loads


def _as_json(nodes, elements, loads) -> dict:
    """The ``run()`` convention for heated models: the element data already
    carries the temperature-degraded modulus ``E = k_E(T) E_20`` and the
    temperature change ``delta_T = T - 20`` (see tests/test_thermal_demand.py
    ::_assembly_reference). The limit-state chain gets the same state through
    the ``temps`` mapping instead."""
    return {
        "units": "SI",
        "nodes": [
            {
                "id": n.id,
                "x": n.x,
                "y": n.y,
                "is_support": n.is_support,
                "support_dx": n.support_dx,
                "support_dy": n.support_dy,
            }
            for n in nodes
        ],
        "elements": [
            {
                "id": e.id,
                "node_i": e.node_i,
                "node_j": e.node_j,
                "E": e.E * float(eurocode_k_E(T_AMB + e.delta_T)),
                "A": e.A,
                "I_sec": e.I_sec,
                "alpha": e.alpha,
                "delta_T": e.delta_T,
                "delta_L0": e.delta_L_free,
                "effective_length_factor": e.effective_length_factor,
            }
            for e in elements
        ],
        "loads": [
            {"node_id": nid, "Fx": ld["Fx"], "Fy": ld["Fy"]}
            for nid, ld in loads.items()
        ],
    }


def test_run_and_limitstates_agree_on_thermal_member_forces(tmp_path) -> None:
    """``run()`` and ``member_axial_forces`` must report the same N.

    Two different assemblers, two different load-vector builders, one
    physical answer. The temperatures fed to the limit-state chain are the
    element ``delta_T`` shifted to absolute steel temperature, which is what
    makes ``alpha (T - 20) L`` in the engine equal ``alpha delta_T L`` in
    the assembler.
    """
    import json

    nodes, elements, loads = _redundant_heated_model(delta_T=400.0)
    path = tmp_path / "heated.json"
    path.write_text(json.dumps(_as_json(nodes, elements, loads)), encoding="utf-8")
    result = run(path, quiet=True)
    assert result.status == "SUCCESS"

    temps = {e.id: T_AMB + e.delta_T for e in elements}
    chain = member_axial_forces(nodes, elements, loads, temps)
    run_forces = {
        str(entry["id"]): float(entry["N"]) for entry in result.element_forces
    }

    assert set(run_forces) == set(chain)
    scale = max(abs(v) for v in chain.values())
    for eid, n_chain in chain.items():
        assert run_forces[eid] == pytest.approx(n_chain, rel=1e-9, abs=1e-6 * scale), (
            eid
        )
    # and the demand is genuinely thermal, not a zero both paths agree on
    assert scale > 1e5


def test_reaction_influence_satisfies_global_equilibrium() -> None:
    """Sum of reactions balances the external load, thermal case included.

    ``R = (K u)[fixed] - f_ext_fixed`` with ``f_ext_fixed`` carrying both the
    direct support loads and the fixed-DOF part of the imposed equivalent
    nodal forces. Since every member dyad is self-equilibrated, the global
    sum must close against the mechanical load alone.
    """
    nodes, elements, loads = _redundant_heated_model(delta_T=500.0)
    temps = {e.id: T_AMB + e.delta_T for e in elements}
    setup = build_engine(nodes, elements, loads, temps)
    u = base_displacement(setup, total_load_vector(nodes, loads, setup))
    k_scale = {e.id: float(eurocode_k_E(temps[e.id])) for e in elements}
    reactions = reaction_influence(nodes, elements, k_scale, loads=loads, temps=temps)
    r = reactions.k_fixed_free @ u - reactions.f_ext_fixed

    sum_fx = float(sum(ld["Fx"] for ld in loads.values()))
    sum_fy = float(sum(ld["Fy"] for ld in loads.values()))
    fixed = list(reactions.fixed_dofs)
    # DOF parity: even indices are x, odd are y
    r_x = sum(r[i] for i, dof in enumerate(fixed) if dof % 2 == 0)
    r_y = sum(r[i] for i, dof in enumerate(fixed) if dof % 2 == 1)
    scale = max(abs(sum_fx), abs(sum_fy), 1.0)
    assert abs(r_x + sum_fx) <= 1e-9 * scale
    assert abs(r_y + sum_fy) <= 1e-9 * scale


def test_lambda_bar_theta_is_built_from_the_degraded_modulus() -> None:
    """Fire slenderness uses ``E(theta)``, not the ambient modulus.

    A common EN 1993-1-2 error (round-4 audit, critic 5 finding 4): taking
    the ambient ``E`` in ``N_cr,theta`` makes ``lambda_bar_theta`` too large
    and ``chi`` too small -- conservative, but for the wrong reason and by
    the wrong amount. The ratio is pinned against the closed form
    ``lambda_bar(theta) / lambda_bar(20) = sqrt(k_y/k_E)`` and the absolute
    value against a hand-built ``N_cr`` from ``E(theta) = k_E(theta) E``.
    """
    from truss_analysis.material.steel_eurocode import k_E as k_E_theta
    from truss_analysis.material.steel_eurocode import k_y as k_y_theta

    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="2", x=0.0, y=2.0, is_support=True, support_dx=True, support_dy=False),
    ]
    elements = [Element(id="c", node_i="1", node_j="2", E=E_STEEL, A=0.01, I_sec=1e-6)]
    loads = {"2": {"Fx": 0.0, "Fy": -100e3}}

    theta = 600.0
    temps = {"c": theta}
    cold = dcr_field(nodes, elements, loads, {"c": T_AMB}, F_Y)["c"]
    hot = dcr_field(nodes, elements, loads, temps, F_Y)["c"]

    k_e, k_y = float(k_E_theta(theta)), float(k_y_theta(theta))
    assert hot.lambda_bar is not None
    assert cold.lambda_bar is not None
    assert hot.lambda_bar == pytest.approx(
        cold.lambda_bar * (k_y / k_e) ** 0.5, rel=1e-12
    )

    # absolute value from the hand-built degraded Euler load
    n_cr_theta = euler_buckling_load(1e-6, 2.0, k_e * E_STEEL, 1.0)
    f_y_theta = k_y * F_Y
    expected = non_dimensional_slenderness(0.01, f_y_theta, n_cr_theta)
    assert hot.lambda_bar == pytest.approx(expected, rel=1e-12)
    # and it must NOT equal the ambient-modulus slenderness (the error mode)
    n_cr_ambient = euler_buckling_load(1e-6, 2.0, E_STEEL, 1.0)
    wrong = non_dimensional_slenderness(0.01, f_y_theta, n_cr_ambient)
    assert hot.lambda_bar != pytest.approx(wrong, rel=1e-6)
    assert hot.lambda_bar > wrong  # degraded E -> smaller N_cr -> larger lambda_bar
