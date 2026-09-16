"""The system-stability branch of ``dcr_field`` -- the path that had no tests.

``dcr_field(check_system_stability=...)`` multiplies a member's demand-capacity
ratio by ``1 / lambda_cr`` when the *system* bifurcates at or below the applied
load.  That is a safety-relevant number: it is the difference between "every
chord passes its own code check" and "the truss buckles as a system".

Before this file existed, twenty-two tests called ``dcr_field`` and **not one**
passed ``check_system_stability``.  The branch therefore ran only through its
default, and its default used to be ``True`` -- so every caller paid a full
eigen-analysis whose result was then written into a frozen dataclass through
``object.__setattr__``, with no warning and no field recording that it had
happened.  ``dcr`` silently stopped satisfying ``dcr == |axial_force| /
capacity``, and a caller reading the payload could not distinguish an amplified
ratio from a member-level one.

These tests pin the replacement contract:

* ``dcr`` is the member check and its reconstructibility is an invariant that
  holds in *every* branch, including the unstable ones;
* the system verdict lives in ``system_dcr`` / ``system_stability_factor`` /
  ``lambda_cr``, and an absent check is reported as ``None`` rather than as a
  factor of one;
* the amplification is never silent -- ``SystemInstabilityWarning`` is
  mandatory whenever it exceeds one;
* a base state that is already a mechanism is system-wide, so every member is
  reported unbounded, not only the compressed ones.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from truss_analysis.criticality.engine import MechanismError
from truss_analysis.exceptions import SystemInstabilityWarning
from truss_analysis.limitstates import MemberLimitState, dcr_field
from truss_analysis.model import Element, Node
from truss_analysis.stability import linearized_buckling_load_factor

pytestmark = pytest.mark.filterwarnings(
    "ignore::truss_analysis.exceptions.ConstantAlphaWarning"
)

E_STEEL = 210.0e9
F_Y = 235.0e6
AREA = 1.0e-3


def _toggle(rise: float, load: float, i_sec: float = 1.0e-7):
    """Shallow two-bar toggle: pinned supports, apex load, span 2 m.

    The rise and load together set ``lambda_cr``; because the factor scales as
    ``1 / load`` for fixed geometry, one fixture spans the stable, marginal and
    unstable regimes.  Both bars are in compression, so this is the
    configuration where a system bifurcation is real and a member check is not
    sufficient.
    """
    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=rise, is_support=False),
    ]
    elements = [
        Element(id="r1", node_i="L", node_j="A", E=E_STEEL, A=AREA, I_sec=i_sec),
        Element(id="r2", node_i="R", node_j="A", E=E_STEEL, A=AREA, I_sec=i_sec),
    ]
    loads = {"A": {"Fx": 0.0, "Fy": -load}}
    temps = {e.id: 20.0 for e in elements}
    return nodes, elements, loads, temps


def _fan(dl_free: float, theta: float = 0.01):
    """Redundant shallow fan: two rafters plus a slender post to a fixed node.

    The post makes the apex vertically redundant, so a ``delta_L_free`` on the
    rafters produces genuine *restrained* compression -- the textbook
    thermal/fabrication-prestress destabilisation configuration.  It is used
    here for two reasons the plain toggle cannot serve:

    * it is the only fixture in this file with a **tension** member present
      while the system is unstable, which is what makes "amplify compression
      only" a testable statement rather than a vacuous one;
    * driving ``dl_free`` up past ~7.15e-4 takes the prestressed base state
      through ``lambda_cr = 1`` and on to a state whose tangent stiffness is
      indefinite, so the same geometry covers both the amplification branch and
      the ``MechanismError`` branch.

    Measured landmarks: ``dl_free = 7.14e-4`` gives ``lambda_cr = 0.5641``,
    ``8e-4`` raises ``MechanismError``.
    """
    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=theta, is_support=False),
    ]
    elements = [
        Element(
            id="r1",
            node_i="L",
            node_j="A",
            E=E_STEEL,
            A=1.0e-3,
            I_sec=1.0e-9,
            delta_L_free=dl_free,
        ),
        Element(
            id="r2",
            node_i="R",
            node_j="A",
            E=E_STEEL,
            A=1.0e-3,
            I_sec=1.0e-9,
            delta_L_free=dl_free,
        ),
        Element(id="post", node_i="A", node_j="B", E=E_STEEL, A=1.0e-6, I_sec=1.0e-12),
    ]
    loads = {"A": {"Fx": 0.0, "Fy": -50.0}}
    temps = {e.id: 20.0 for e in elements}
    return nodes, elements, loads, temps


def _assert_dcr_invariant(states: dict[str, MemberLimitState]) -> None:
    """``dcr`` must be reconstructible from the fields beside it, always.

    This is the regression that matters.  The old implementation broke it by
    writing an amplified value over ``dcr`` in place, so a payload could not be
    checked for self-consistency and an amplified ratio was indistinguishable
    from a member-level one.
    """
    for mid, ls in states.items():
        if ls.capacity and math.isfinite(ls.capacity) and ls.capacity > 0.0:
            assert ls.dcr == pytest.approx(
                abs(ls.axial_force) / ls.capacity, rel=1e-12
            ), (
                f"{mid}: dcr={ls.dcr} is not |axial_force|/capacity="
                f"{abs(ls.axial_force) / ls.capacity}; the member ratio was mutated"
            )


# ---------------------------------------------------------------------------
# 1. the default: no eigen-analysis, and absence reported as absence
# ---------------------------------------------------------------------------


def test_system_check_is_off_by_default() -> None:
    """The default must not buy an eigen-analysis on every call.

    ``dcr_field`` sits inside the retrofit search, where it is evaluated once
    per candidate decision.  With the check on by default each of those paid a
    dense motor build, a factorisation and a Lanczos sweep.
    """
    nodes, elements, loads, temps = _toggle(0.30, 2.0e7)
    states = dcr_field(nodes, elements, loads, temps, F_Y)
    for ls in states.values():
        assert ls.lambda_cr is None
        assert ls.system_dcr is None
        assert ls.system_stability_factor == 1.0
    _assert_dcr_invariant(states)


def test_off_by_default_reports_none_not_one() -> None:
    """``system_dcr is None``, not ``system_dcr == dcr``.

    A factor of one and an absent check are different statements.  Collapsing
    them would let a caller read ``system_dcr == dcr`` as "system stability was
    verified and did not govern" when in fact nothing was computed.
    """
    nodes, elements, loads, temps = _toggle(0.30, 1.0e4)
    states = dcr_field(nodes, elements, loads, temps, F_Y)
    assert all(ls.system_dcr is None for ls in states.values())
    assert all(ls.system_dcr != ls.dcr for ls in states.values())


# ---------------------------------------------------------------------------
# 2. stable system: the check runs and changes nothing
# ---------------------------------------------------------------------------


def test_stable_system_leaves_dcr_untouched_and_stays_quiet() -> None:
    """``lambda_cr > 1`` means the member check governs: factor exactly 1."""
    nodes, elements, loads, temps = _toggle(0.30, 1.0e4)
    lam = linearized_buckling_load_factor(
        nodes, elements, loads, temps, warn_shallow=False
    ).lambda_cr
    assert lam > 1.0, "fixture must be in the stable regime"

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", SystemInstabilityWarning)
        states = dcr_field(
            nodes, elements, loads, temps, F_Y, check_system_stability=True
        )
    for ls in states.values():
        assert ls.lambda_cr == pytest.approx(lam, rel=1e-12)
        assert ls.system_stability_factor == 1.0
        assert ls.system_dcr == pytest.approx(ls.dcr, rel=1e-12)
    _assert_dcr_invariant(states)


# ---------------------------------------------------------------------------
# 3. unstable system: amplify, warn, and keep the member ratio honest
# ---------------------------------------------------------------------------


def test_unstable_system_warns_and_reports_the_factor() -> None:
    """The amplification must never be silent, and must be reconstructible."""
    nodes, elements, loads, temps = _toggle(0.30, 2.0e7)
    lam = linearized_buckling_load_factor(
        nodes, elements, loads, temps, warn_shallow=False
    ).lambda_cr
    assert 0.0 < lam < 1.0, "fixture must straddle the bifurcation point"

    with pytest.warns(SystemInstabilityWarning) as caught:
        states = dcr_field(
            nodes, elements, loads, temps, F_Y, check_system_stability=True
        )

    message = str(caught[0].message)
    # the warning must carry the number, not merely the fact
    assert f"{lam:.4f}" in message
    assert f"{1.0 / lam:.4f}" in message
    assert "system_dcr" in message
    # and must not overclaim: lambda_cr is a linearised tangent verdict
    assert "not the ultimate load" in message

    for ls in states.values():
        assert ls.compression
        assert ls.lambda_cr == pytest.approx(lam, rel=1e-12)
        assert ls.system_stability_factor == pytest.approx(1.0 / lam, rel=1e-12)
        assert ls.system_dcr == pytest.approx(ls.dcr / lam, rel=1e-12)
        assert ls.system_dcr > ls.dcr
    _assert_dcr_invariant(states)


def test_member_dcr_is_identical_whether_or_not_the_check_ran() -> None:
    """Turning the system check on must not move the member number.

    This is the precise inverse of the old behaviour, where enabling the check
    rewrote ``dcr`` in place (275.86 -> 507.95 on the fixture below).  A caller
    who opts into a system study should learn something additional, not find
    that the member check they already had has quietly changed meaning.
    """
    nodes, elements, loads, temps = _toggle(0.30, 2.0e7)
    off = dcr_field(nodes, elements, loads, temps, F_Y)
    with pytest.warns(SystemInstabilityWarning):
        on = dcr_field(nodes, elements, loads, temps, F_Y, check_system_stability=True)
    for mid in off:
        assert on[mid].dcr == off[mid].dcr
        assert on[mid].axial_force == off[mid].axial_force
        assert on[mid].capacity == off[mid].capacity
        # and the difference is entirely accounted for by the new fields
        assert on[mid].system_dcr == pytest.approx(
            off[mid].dcr * on[mid].system_stability_factor, rel=1e-12
        )


def test_the_result_is_not_mutated_in_place() -> None:
    """``MemberLimitState`` is frozen and the adjustment goes through ``replace``.

    The old code reached past the frozen contract with ``object.__setattr__``.
    A payload built that way cannot be trusted to be self-consistent, because
    the fields it was constructed from and the fields it now reports disagree.
    """
    nodes, elements, loads, temps = _toggle(0.30, 2.0e7)
    with pytest.warns(SystemInstabilityWarning):
        states = dcr_field(
            nodes, elements, loads, temps, F_Y, check_system_stability=True
        )
    for ls in states.values():
        with pytest.raises(dataclasses.FrozenInstanceError):
            ls.dcr = 0.0  # type: ignore[misc]
        # and the frozen contract is the reason the invariant is trustworthy:
        # nothing downstream can move dcr after construction either
        assert ls.system_dcr == pytest.approx(
            ls.dcr * ls.system_stability_factor, rel=1e-12
        )


def test_lambda_cr_is_a_system_quantity_carried_on_every_member() -> None:
    """Every member reports the same ``lambda_cr``, so it travels in the payload.

    Carrying it per member rather than returning it once means the system
    verdict survives warning filters and reaches JSON output -- the same
    reasoning as ``SteelHeatingResult.out_of_range``, which makes a validity
    limit visible in the payload instead of only in a warning that a caller may
    have suppressed.
    """
    nodes, elements, loads, temps = _toggle(0.30, 2.0e7)
    with pytest.warns(SystemInstabilityWarning):
        states = dcr_field(
            nodes, elements, loads, temps, F_Y, check_system_stability=True
        )
    factors = {ls.lambda_cr for ls in states.values()}
    assert len(factors) == 1
    assert math.isfinite(next(iter(factors)))


# ---------------------------------------------------------------------------
# 4. tension members are not amplified
# ---------------------------------------------------------------------------


def test_tension_members_are_not_amplified() -> None:
    """A bifurcation is compression-driven; a tie rod's demand does not grow.

    Multiplying a tension member's DCR by ``1 / lambda_cr`` would claim demand
    the physics does not put there -- and would push a safe tie over 1.0 for a
    failure mode it cannot participate in.  ``lambda_cr`` is still reported on
    it, so the caller can see the system verdict without it being folded into
    the tie's own ratio.
    """
    nodes, elements, loads, temps = _fan(7.14e-4)
    lam = linearized_buckling_load_factor(
        nodes, elements, loads, temps, warn_shallow=False
    ).lambda_cr
    assert 0.0 < lam < 1.0, f"fixture must straddle the bifurcation point, got {lam}"

    with pytest.warns(SystemInstabilityWarning):
        states = dcr_field(
            nodes, elements, loads, temps, F_Y, check_system_stability=True
        )
    tension = {m: ls for m, ls in states.items() if not ls.compression}
    compression = {m: ls for m, ls in states.items() if ls.compression}
    assert tension, "fixture must contain a tension member"
    assert compression, "fixture must contain a compression member"

    for mid, ls in tension.items():
        assert ls.system_stability_factor == 1.0, mid
        assert ls.system_dcr == pytest.approx(ls.dcr, rel=1e-12), mid
        # the system verdict is still visible on the tie, just not folded in
        assert ls.lambda_cr == pytest.approx(lam, rel=1e-12), mid
    for mid, ls in compression.items():
        assert ls.system_stability_factor == pytest.approx(1.0 / lam, rel=1e-12), mid
        assert ls.system_dcr > ls.dcr, mid
    _assert_dcr_invariant(states)


# ---------------------------------------------------------------------------
# 5. base state already a mechanism: system-wide, not compression-only
# ---------------------------------------------------------------------------


def test_mechanism_base_state_reports_every_member_unbounded() -> None:
    """A collapsed structure must not be reported with finite tensile DCRs.

    When the bifurcation solve raises ``MechanismError`` the tangent stiffness
    is already indefinite at the applied load, so the structure cannot carry it
    at all.  That is categorically different from a bifurcation, which is
    compression-driven: the old code set only the compression members to
    ``inf`` and left the tension members finite, so a collapsed truss was
    reported as having members in reserve.  The fixture's ``post`` is exactly
    such a member -- it was reported at ``dcr = 11.7`` while the rafters were
    ``inf``, on a structure that could not stand up.

    Note that ``member_axial_forces`` still succeeds here: the *mechanical*
    base state is fine and it is the prestressed tangent state that has lost
    definiteness, which is why this branch is reachable at all.
    """
    nodes, elements, loads, temps = _fan(8.0e-4)
    with pytest.raises(MechanismError, match="not positive definite"):
        linearized_buckling_load_factor(
            nodes, elements, loads, temps, warn_shallow=False
        )

    with pytest.warns(SystemInstabilityWarning) as caught:
        states = dcr_field(
            nodes, elements, loads, temps, F_Y, check_system_stability=True
        )
    message = str(caught[0].message)
    assert "already indefinite" in message

    assert any(not ls.compression for ls in states.values()), (
        "the fixture must contain a tension member, or the test is vacuous"
    )
    for mid, ls in states.items():
        assert ls.system_dcr == float("inf"), mid
        assert ls.system_stability_factor == float("inf"), mid
        # JSON-safe: the collapse is recorded as zero reserve, never as NaN
        assert ls.lambda_cr == 0.0, mid
        assert not math.isnan(ls.lambda_cr), mid
    # the member ratio stays honest even when the system has no reserve
    _assert_dcr_invariant(states)


# ---------------------------------------------------------------------------
# 6. the adjustment is bounded, cannot invent a sign, and the boundary is <=
# ---------------------------------------------------------------------------


def test_factor_is_exactly_the_reciprocal_and_never_below_one() -> None:
    """``1 / lambda_cr`` on ``0 < lambda_cr <= 1`` is >= 1 by construction.

    Pinned because an unguarded reciprocal is where a sign error or a division
    by zero would appear, and because a factor *below* one would silently
    reduce a reported demand -- the one direction of error that is never
    acceptable in a fire check.
    """
    nodes, elements, loads, temps = _toggle(0.30, 2.0e7)
    with pytest.warns(SystemInstabilityWarning):
        states = dcr_field(
            nodes, elements, loads, temps, F_Y, check_system_stability=True
        )
    for ls in states.values():
        assert ls.system_stability_factor >= 1.0
        assert ls.system_stability_factor == pytest.approx(
            1.0 / ls.lambda_cr, rel=1e-12
        )
        assert ls.system_dcr >= ls.dcr


def test_the_unity_boundary_is_inclusive_on_both_sides() -> None:
    """``lambda_cr <= 1`` governs, ``lambda_cr > 1`` does not -- pinned from
    both sides so the comparison operator cannot drift.

    A toggle's factor scales as ``1 / load`` for fixed geometry, so multiplying
    the load by the measured ``lambda_cr`` lands the system on the boundary to
    round-off.  At exactly 1.0 the reserve is zero: the amplification is 1.0,
    but the *verdict* is still "no reserve" and must be reported.  A strict
    ``<`` would have called that unremarkable.  Just above the boundary nothing
    is owed and nothing must be said.
    """
    nodes, elements, loads, temps = _toggle(0.30, 1.0e4)
    lam = linearized_buckling_load_factor(
        nodes, elements, loads, temps, warn_shallow=False
    ).lambda_cr
    assert lam > 1.0

    fy = loads["A"]["Fy"]

    # just below the boundary: governs, warns, amplification ~ 1
    at_unity = {"A": {"Fx": 0.0, "Fy": fy * lam * (1.0 + 1e-9)}}
    below = linearized_buckling_load_factor(
        nodes, elements, at_unity, temps, warn_shallow=False
    ).lambda_cr
    assert below <= 1.0, f"expected <= 1, got {below}"
    with pytest.warns(SystemInstabilityWarning):
        states = dcr_field(
            nodes, elements, at_unity, temps, F_Y, check_system_stability=True
        )
    for ls in states.values():
        assert ls.system_stability_factor == pytest.approx(1.0, rel=1e-6)
        assert ls.system_dcr == pytest.approx(ls.dcr, rel=1e-6)

    # just above the boundary: the member check governs and nothing is owed
    above_load = {"A": {"Fx": 0.0, "Fy": fy * lam * (1.0 - 1e-6)}}
    above = linearized_buckling_load_factor(
        nodes, elements, above_load, temps, warn_shallow=False
    ).lambda_cr
    assert above > 1.0, f"expected > 1, got {above}"
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", SystemInstabilityWarning)
        states = dcr_field(
            nodes, elements, above_load, temps, F_Y, check_system_stability=True
        )
    for ls in states.values():
        assert ls.system_stability_factor == 1.0
        assert ls.system_dcr == pytest.approx(ls.dcr, rel=1e-12)


# ---------------------------------------------------------------------------
# 7. the boundary decision, as a pure function
# ---------------------------------------------------------------------------


def test_the_unity_boundary_is_inclusive() -> None:
    """``<= 1.0`` and not ``< 1.0``, asked about exactly 1.0.

    A mutation probe on this module found that flipping the comparison left the
    whole suite green.  The reason is worth recording, because it is not "the
    tests were lazy": reaching ``lambda_cr = 1.0`` from the outside requires
    scaling a load by the measured factor, and round-off lands the result at
    ``1 ± 1e-16``, so no load-scaling test can sit *on* the boundary.  The
    decision was therefore extracted as a pure function that can be asked about
    the number directly -- which is the general lesson: a boundary that cannot
    be reached from the public API is a boundary that cannot be tested.
    """
    from truss_analysis.limitstates import system_stability_amplification

    governs, factor = system_stability_amplification(1.0, collapsed=False)
    assert governs is True, "zero reserve must be reported, not passed over"
    assert factor == 1.0


@pytest.mark.parametrize(
    ("lambda_cr", "expected_governs", "expected_factor"),
    [
        (0.5, True, 2.0),
        (0.999999999999, True, 1.0 / 0.999999999999),
        (1.0, True, 1.0),
        (1.000000000001, False, 1.0),
        (2.0, False, 1.0),
        (float("inf"), False, 1.0),
        (1.0e-12, True, 1.0e12),
    ],
)
def test_the_amplification_decision_over_its_whole_domain(
    lambda_cr: float, expected_governs: bool, expected_factor: float
) -> None:
    """Every branch of the decision, including the two sides of the boundary.

    The pair ``(0.999999999999, 1.000000000001)`` straddles 1.0 by less than
    round-off from a load-scaled solve could manage, which is what makes the
    ``<=`` pinned rather than merely described in a comment.
    """
    from truss_analysis.limitstates import system_stability_amplification

    governs, factor = system_stability_amplification(lambda_cr, collapsed=False)
    assert governs is expected_governs
    assert factor == pytest.approx(expected_factor, rel=1e-12)


@pytest.mark.parametrize("lambda_cr", [0.0, -1.0, -1.0e-9, float("nan")])
def test_a_non_positive_factor_never_produces_a_reciprocal_or_a_silent_pass(
    lambda_cr: float,
) -> None:
    """An unguarded ``1 / lambda_cr`` divides by zero at exactly 0.

    The documented solver retains only positive eigenvalues, so ``lambda_cr``
    should never arrive here non-positive -- but "should never" is what a guard
    is for.  The collapse convention (``0.0``) and anything negative both mean
    no reserve, and ``NaN`` must govern rather than fall through to a silent
    pass, because a verdict that cannot be evaluated is not a verdict that
    everything is fine.
    """
    from truss_analysis.limitstates import system_stability_amplification

    governs, factor = system_stability_amplification(lambda_cr, collapsed=False)
    assert governs is True
    assert factor == float("inf")


def test_a_collapse_overrides_whatever_lambda_cr_says() -> None:
    """``collapsed=True`` governs even if a finite factor is passed alongside."""
    from truss_analysis.limitstates import system_stability_amplification

    for lambda_cr in (5.0, 1.0, 0.5):
        governs, factor = system_stability_amplification(lambda_cr, collapsed=True)
        assert governs is True
        assert factor == float("inf")


def test_the_factor_is_never_below_one() -> None:
    """No input may make the adjustment *reduce* a reported demand.

    Stated as a sweep rather than per-case, because the direction of the error
    is the whole safety argument: an amplifier below one would make a fire DCR
    look better than the member check alone, which is worse than not adjusting.
    """
    from truss_analysis.limitstates import system_stability_amplification

    for collapsed in (False, True):
        for lambda_cr in (1e-12, 0.25, 0.5, 1.0, 1.5, 10.0, float("inf")):
            _governs, factor = system_stability_amplification(lambda_cr, collapsed)
            assert factor >= 1.0, (lambda_cr, collapsed, factor)


def test_the_warning_names_how_many_members_it_amplified() -> None:
    """The count in the message must be the compression members, not all.

    A mutation probe found that changing ``ls.compression and ls.p_cr is not
    None`` to ``or`` in the count left the suite green: the amplification itself
    is applied by a separate predicate that *was* tested, so only the sentence
    was wrong.  A warning that reports the wrong number of governed members is
    still a wrong number in the one place a reader looks when the analysis says
    the structure is failing, so the count is pinned too.
    """
    nodes, elements, loads, temps = _fan(7.14e-4)
    with pytest.warns(SystemInstabilityWarning) as caught:
        states = dcr_field(
            nodes, elements, loads, temps, F_Y, check_system_stability=True
        )
    message = str(caught[0].message)
    n_amplified = sum(
        1 for ls in states.values() if ls.compression and ls.p_cr is not None
    )
    n_tension = sum(1 for ls in states.values() if not ls.compression)
    assert n_amplified == 2
    assert n_tension == 1, (
        "the fixture must have a tension member for this to mean anything"
    )
    assert f"{n_amplified} compression member(s)" in message
    assert f"{len(states)} compression member(s)" not in message
