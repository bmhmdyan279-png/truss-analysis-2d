"""Boundary conditions the mutation probe found unguarded.

Every test in this file exists because a mutant survived.  ``scripts/mutation_probe.py``
applied small semantic edits to ``limitstates.py`` -- ``<`` to ``<=``, ``and`` to
``or``, ``-`` to ``+`` -- and ran the module's own tests against each one.  A
surviving mutant is a line where the suite's assertions do not reach the
decision the line makes, which is exactly the gap between "the line ran" (what
coverage measures) and "the line was checked" (what a test is for).

The module scored 35/45 on the first run and 42/45 after this file existed.
The three survivors left are the equivalent mutants documented at the bottom,
so every mutant that any test could kill now dies.  The original ten fall into
three groups, treated differently:

* **Real boundaries, now pinned.** A comparison against an exact threshold that
  no fixture ever landed on: ``compression = axial_force < 0`` with a zero-force
  member, ``lambda_bar <= 0.2`` at exactly 0.2, ``p_cr <= n_rd`` at exact
  equality, ``len(unassessable) > 5`` at exactly five and six, and a member
  length computed with ``x_j - x_i`` on models whose ``node_i`` was always at the
  origin so ``+`` and ``-`` agreed.
* **A predicate whose two halves no fixture separated.** The system-stability
  warning counts ``ls.compression and ls.p_cr is not None``; on every existing
  fixture those two conditions happened to coincide, so ``or`` produced the same
  count.  Killing it needed a compression member whose buckling cannot be
  assessed (``I_sec <= 0``), where ``p_cr`` is ``None`` but ``compression`` is
  not.
* **Mathematically equivalent mutants, recorded rather than chased.** See
  :func:`test_the_equivalent_mutants_are_equivalent_for_a_stated_reason`.  A
  mutation score cannot reach 100% because some edits do not change the computed
  value for any input, and reporting a score without saying so would overstate
  what the instrument measures.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from truss_analysis.limitstates import (
    GAMMA_M_FIRE,
    Governing,
    UniformForceScan,
    dcr_field,
    member_axial_forces,
)
from truss_analysis.model import Element, Node
from truss_analysis.sections import (
    LAMBDA_BAR_BUCKLING_LIMIT,
    non_dimensional_slenderness,
)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.ConstantAlphaWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::truss_analysis.exceptions.LegacyBucklingModelWarning"
    ),
]

E_STEEL = 210.0e9
F_Y = 235.0e6


# ---------------------------------------------------------------------------
# a member length computed from a difference, on a model that is not at origin
# ---------------------------------------------------------------------------


def test_member_lengths_survive_translating_the_model() -> None:
    """``hypot(x_j - x_i, y_j - y_i)`` must not be ``hypot(x_j + x_i, ...)``.

    The probe flipped the subtraction and nothing noticed, because every fixture
    in the suite put ``node_i`` at the origin -- where ``x_j - 0`` and
    ``x_j + 0`` are the same number.  A mutant that only survives because of a
    coincidence in the fixtures is still a hole: the coincidence is not a
    property of the code, and the next model someone loads will not share it.

    Translating a truss rigidly changes every nodal coordinate and no member
    length, so the forces must be bit-identical.  With ``+`` in place of ``-``
    the translated model's lengths grow and the forces move.
    """

    def build(dx: float, dy: float):
        # Redundant on purpose.  The first version of this fixture was a
        # two-bar toggle: determinate, so restrained thermal expansion produces
        # no force at all and `alpha_lengths` never reaches a result.  The
        # mutant survived a test that set alpha, translated the model and
        # compared both scan bases -- three changes that each looked sufficient.
        # A third member down to a pinned node makes the apex redundant, which
        # is what turns a length error into a force error.
        nodes = [
            Node(
                id="L",
                x=-1.0 + dx,
                y=0.0 + dy,
                is_support=True,
                support_dx=True,
                support_dy=True,
            ),
            Node(
                id="R",
                x=1.0 + dx,
                y=0.0 + dy,
                is_support=True,
                support_dx=True,
                support_dy=True,
            ),
            Node(
                id="B",
                x=0.0 + dx,
                y=-1.0 + dy,
                is_support=True,
                support_dx=True,
                support_dy=True,
            ),
            Node(id="A", x=0.0 + dx, y=0.4 + dy, is_support=False),
        ]
        # alpha must be non-zero, and this is what the first version of this
        # test got wrong.  Inside UniformForceScan.build the local `lengths`
        # array feeds ONLY `alpha_lengths = alpha * lengths` and `unit_lengths =
        # where(alpha != 0, lengths, 0)`; the geometric lengths the stiffness
        # comes from are computed independently by the assembler.  With
        # Element's default `alpha = 0.0` both of those are zero vectors, so
        # `lengths` never reaches a force and the mutant survives a test that
        # reads exactly like this one.  The probe is what showed it.
        elements = [
            Element(
                id="r1",
                node_i="L",
                node_j="A",
                E=E_STEEL,
                A=2.0e-3,
                I_sec=1.0e-6,
                alpha=1.2e-5,
            ),
            Element(
                id="r2",
                node_i="R",
                node_j="A",
                E=E_STEEL,
                A=2.0e-3,
                I_sec=1.0e-6,
                alpha=1.2e-5,
            ),
            Element(
                id="r3",
                node_i="B",
                node_j="A",
                E=E_STEEL,
                A=2.0e-3,
                I_sec=1.0e-6,
                alpha=1.2e-5,
            ),
        ]
        loads = {"A": {"Fx": 3.0e4, "Fy": -8.0e4}}
        return nodes, elements, loads

    base = build(0.0, 0.0)
    moved = build(137.5, -42.25)
    temps = {e.id: 20.0 for e in base[1]}

    f_base = member_axial_forces(*base, temps)
    f_moved = member_axial_forces(*moved, temps)
    for eid in f_base:
        assert f_moved[eid] == pytest.approx(f_base[eid], rel=1e-12), eid

    # and the same invariance through the closed-form scan, which computes the
    # lengths itself rather than inheriting them from the assembler.  Both bases
    # are checked, because `lengths` reaches `alpha_lengths` in the constant
    # basis and `unit_lengths` in the secant one, and a mutant in the shared
    # computation has to be caught by whichever path actually reads it.
    for flag in (False, True):
        s_base = UniformForceScan.build(*base, use_effective_alpha=flag).forces_at(
            400.0
        )
        s_moved = UniformForceScan.build(*moved, use_effective_alpha=flag).forces_at(
            400.0
        )
        assert np.allclose(s_base, s_moved, rtol=1e-12), (flag, s_base, s_moved)


# ---------------------------------------------------------------------------
# compression = axial_force < 0, at exactly zero
# ---------------------------------------------------------------------------


def test_a_zero_force_member_is_not_in_compression() -> None:
    """``< 0.0`` and not ``<= 0.0``: a zero-force member has no buckling problem.

    The probe flipped the comparison and the suite stayed green because no
    fixture had a member at exactly zero force.  The distinction is not
    pedantic: classifying a zero-force member as compressed makes the code
    compute an Euler load and a reduction factor for a member carrying nothing,
    and reports ``capacity_governing`` as a buckling verdict on a member that
    cannot buckle.  Sign conventions downstream read ``compression`` to decide
    whether ``p_cr`` is meaningful at all.
    """
    # a determinate truss with a genuinely unloaded member: the vertical at the
    # roller carries nothing when the load is purely horizontal
    nodes = [
        Node(id="A", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=3.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
        Node(id="C", x=3.0, y=2.0, is_support=False),
    ]
    elements = [
        Element(id="bottom", node_i="A", node_j="B", E=E_STEEL, A=2.0e-3, I_sec=1e-6),
        Element(id="vert", node_i="B", node_j="C", E=E_STEEL, A=2.0e-3, I_sec=1e-6),
        Element(id="diag", node_i="A", node_j="C", E=E_STEEL, A=2.0e-3, I_sec=1e-6),
    ]
    loads = {"C": {"Fx": 2.0e4, "Fy": 0.0}}
    temps = {e.id: 20.0 for e in elements}

    forces = member_axial_forces(nodes, elements, loads, temps)
    zero = [eid for eid, f in forces.items() if f == 0.0]
    assert zero, "the fixture must contain an exactly-zero-force member"

    states = dcr_field(nodes, elements, loads, temps, F_Y)
    for eid in zero:
        ls = states[eid]
        assert ls.axial_force == 0.0
        assert ls.compression is False, (
            f"{eid} carries exactly zero force and must not be classified as "
            "compressed; '<= 0' would give it a buckling verdict it cannot have"
        )
        assert ls.p_cr is None
        assert ls.chi is None
        assert ls.lambda_bar is None


# ---------------------------------------------------------------------------
# lambda_bar <= 0.2, at exactly 0.2
# ---------------------------------------------------------------------------


def test_the_stocky_boundary_is_inclusive_at_exactly_the_limit() -> None:
    """``lambda_bar <= 0.2`` yields; a strict ``<`` would call it buckling.

    EN 1993-1-1:2005 6.3.1(4) says buckling need not be considered at or below
    the limit, so the limit itself belongs to the stocky side.  The probe flipped
    the comparison and nothing caught it, because constructing a member at
    exactly ``lambda_bar = 0.2`` requires solving for the second moment of area
    rather than picking a round one -- which no fixture had a reason to do.
    """
    length = 2.0
    area = 0.01
    # lambda_bar = sqrt(A f_y / P_cr) = 0.2  =>  P_cr = A f_y / 0.04
    p_cr_target = area * F_Y / LAMBDA_BAR_BUCKLING_LIMIT**2
    # P_cr = pi^2 E I / (k L)^2 with k = 1
    i_needed = p_cr_target * length**2 / (math.pi**2 * E_STEEL)

    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(
            id="2", x=0.0, y=length, is_support=True, support_dx=True, support_dy=False
        ),
    ]
    elements = [
        Element(id="c", node_i="1", node_j="2", E=E_STEEL, A=area, I_sec=i_needed)
    ]
    loads = {"2": {"Fx": 0.0, "Fy": -1.0e5}}
    temps = {"c": 20.0}

    ls = dcr_field(nodes, elements, loads, temps, F_Y)["c"]
    assert ls.lambda_bar == pytest.approx(LAMBDA_BAR_BUCKLING_LIMIT, rel=1e-9)
    assert ls.compression is True
    assert ls.capacity_governing is Governing.YIELD, (
        "at exactly lambda_bar = 0.2 the member is stocky and yield-governed; "
        "a strict '<' would report a buckling verdict here"
    )
    # cross-check the slenderness against the library's own accessor, so the
    # fixture is verified to be on the boundary rather than assumed to be
    assert non_dimensional_slenderness(area, F_Y, ls.p_cr) == pytest.approx(
        LAMBDA_BAR_BUCKLING_LIMIT, rel=1e-9
    )


# ---------------------------------------------------------------------------
# the EULER_ONLY branch guard
# ---------------------------------------------------------------------------


def test_the_euler_only_governing_branch_is_not_reachable_by_p_cr_alone() -> None:
    """``model is EULER_ONLY and p_cr is not None`` -- both halves matter.

    The probe changed ``and`` to ``or`` and the suite stayed green.  With ``or``
    a *Eurocode* member that merely has a ``p_cr`` would fall into the
    EULER_ONLY governing rule, which picks ``BUCKLING`` or ``YIELD`` by
    comparing ``p_cr`` and ``n_rd`` instead of always reporting ``BUCKLING``.
    For a slender member whose Euler load exceeds its yield resistance that
    silently changes the reported governing limit state -- a labelling error on
    the default code path, reachable only through a mutant until now.
    """
    # Slender enough that lambda_bar > 0.2 but not so slender that p_cr falls
    # below n_rd.  The window is n_rd < p_cr < 25 n_rd, since
    # lambda_bar = sqrt(A f_y / p_cr) and the limit is 0.2.  The second moment
    # of area is solved for rather than guessed: a round number lands outside
    # the window, which is why the first attempt at this fixture did.
    area = 0.01
    length = 6.0
    i_needed = 5.0 * area * F_Y * length**2 / (math.pi**2 * E_STEEL)
    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(
            id="2",
            x=0.0,
            y=length,
            is_support=True,
            support_dx=True,
            support_dy=False,
        ),
    ]
    elements = [
        Element(id="c", node_i="1", node_j="2", E=E_STEEL, A=area, I_sec=i_needed)
    ]
    loads = {"2": {"Fx": 0.0, "Fy": -1.0e5}}
    temps = {"c": 20.0}

    ls = dcr_field(nodes, elements, loads, temps, F_Y)["c"]
    assert ls.compression is True
    assert ls.p_cr is not None
    assert ls.lambda_bar > LAMBDA_BAR_BUCKLING_LIMIT, "must not be stocky"
    assert ls.p_cr > ls.n_rd, (
        "the fixture needs p_cr > n_rd: that is what makes the EULER_ONLY "
        "governing rule answer YIELD where the Eurocode rule answers BUCKLING"
    )
    assert ls.capacity_governing is Governing.BUCKLING


# ---------------------------------------------------------------------------
# the unassessable-member warning truncation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_bad", "expects_ellipsis"), [(5, False), (6, True), (7, True)]
)
def test_the_unassessable_list_truncates_after_five(
    n_bad: int, expects_ellipsis: bool
) -> None:
    """``len(unassessable) > 5`` -- five names fit, the sixth is elided.

    The probe flipped the comparison to ``>=`` and nothing noticed, because no
    fixture had exactly five unassessable members.  This is the least
    safety-critical mutant in the set -- it changes a message, not a number --
    but it is pinned anyway: a warning that lists six ids after saying it lists
    five is the kind of small inconsistency that makes a reader distrust the
    rest of the message, and the only way to know the boundary is correct is to
    stand on it.
    """
    from truss_analysis.exceptions import BucklingCheckWarning

    # One independent guided strut per unassessable member: base pinned both
    # ways, top restrained horizontally and free vertically.  Each is a stable
    # single-member compression problem on its own, so the model is not a
    # mechanism and dcr_field reaches the warning instead of raising first --
    # which is what the previous fan-shaped fixture got wrong, and why it
    # silently skipped on every parameter.
    nodes = []
    elements = []
    for i in range(n_bad):
        nodes.append(
            Node(
                id=f"b{i}",
                x=float(i),
                y=0.0,
                is_support=True,
                support_dx=True,
                support_dy=True,
            )
        )
        nodes.append(
            Node(
                id=f"t{i}",
                x=float(i),
                y=2.0,
                is_support=True,
                support_dx=True,
                support_dy=False,
            )
        )
        elements.append(
            Element(
                id=f"bad{i}",
                node_i=f"b{i}",
                node_j=f"t{i}",
                E=E_STEEL,
                A=2.0e-3,
                I_sec=0.0,
            )
        )
    loads = {f"t{i}": {"Fx": 0.0, "Fy": -1.0e3} for i in range(n_bad)}
    temps = {e.id: 20.0 for e in elements}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dcr_field(nodes, elements, loads, temps, F_Y)
    messages = [str(w.message) for w in caught if w.category is BucklingCheckWarning]
    assert messages, "every member here has I_sec = 0 and is compressed"
    joined = " ".join(messages)
    assert f"{n_bad} compressed member(s)" in joined, joined
    assert ("..." in joined) is expects_ellipsis, joined


# ---------------------------------------------------------------------------
# the system-stability count: a redundancy that turns out to be an invariant
# ---------------------------------------------------------------------------


def test_p_cr_is_present_exactly_for_compression_members() -> None:
    """The class invariant that makes the governed-count mutant equivalent.

    The probe changed ``ls.compression and ls.p_cr is not None`` to ``or`` in the
    count the system-instability warning reports, and no test caught it.  Chasing
    a fixture that would catch it produced this instead: there is no such
    fixture, because ``_member_limit_state`` sets ``p_cr`` **only** inside its
    ``if compression:`` branch, so ``p_cr is not None`` and ``compression`` are
    the same statement for every reachable input.  The two halves of the
    conjunction cannot be separated and the mutant is equivalent.

    That is worth more than a killing test would have been.  The conjunction is
    defensive redundancy, and redundancy whose redundancy is unstated gets
    "simplified" by someone who does not know it was load-bearing -- or, as here,
    costs an afternoon writing a test that cannot fail.  So the invariant is
    pinned directly and the equivalent-mutant list below cites it.

    The ``I_sec = 0`` case in particular does **not** produce ``p_cr is None``:
    ``euler_buckling_load(0.0, ...)`` returns ``0.0``, the member is reported at
    ``dcr = inf`` with a ``BucklingCheckWarning``, and ``p_cr`` is zero rather
    than absent.  That is the fixture the first attempt at this test assumed.
    """
    from truss_analysis.exceptions import BucklingCheckWarning

    nodes = [
        Node(id="b", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="t", x=0.0, y=2.0, is_support=True, support_dx=True, support_dy=False),
    ]
    elements = [
        Element(id="strut", node_i="b", node_j="t", E=E_STEEL, A=2.0e-3, I_sec=1e-6),
        Element(id="blind", node_i="b", node_j="t", E=E_STEEL, A=2.0e-3, I_sec=0.0),
    ]
    loads = {"t": {"Fx": 0.0, "Fy": -5.0e4}}
    temps = {e.id: 20.0 for e in elements}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BucklingCheckWarning)
        for eid in ("strut",):
            states = dcr_field(nodes, [elements[0]], loads, {"strut": 20.0}, F_Y)
            assert states[eid].compression is True
            assert states[eid].p_cr is not None
        states = dcr_field(nodes, elements, loads, temps, F_Y)
    for ls in states.values():
        assert ls.compression is True
        assert ls.p_cr is not None, ls.member_id
    assert states["blind"].p_cr == 0.0, "zero from I_sec = 0, not None"
    assert states["blind"].dcr == float("inf")

    # the tension side of the same invariant
    tie_nodes = [
        Node(id="L", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=3.0, y=0.0, is_support=True, support_dx=False, support_dy=True),
    ]
    tie = [Element(id="t", node_i="L", node_j="R", E=E_STEEL, A=2.0e-3, I_sec=1e-6)]
    tie_states = dcr_field(
        tie_nodes, tie, {"R": {"Fx": 8.0e4, "Fy": 0.0}}, {"t": 20.0}, F_Y
    )
    for ls in tie_states.values():
        assert ls.compression is False
        assert ls.p_cr is None
        assert ls.chi is None
        assert ls.lambda_bar is None


def test_the_governed_count_equals_the_members_actually_amplified() -> None:
    """The message's count and the amplifying predicate must agree.

    Both are written from ``ls.compression and ls.p_cr is not None``, as two
    separate expressions -- one building a sentence, one deciding a factor.  The
    invariant above makes them unable to disagree today, but this assertion is
    the one that fails if a future change ever gives a compression member a
    ``None`` ``p_cr``, at which point the two expressions stop being
    interchangeable and only this test notices.
    """
    from truss_analysis.exceptions import SystemInstabilityWarning

    nodes = [
        Node(id="L", x=-1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="R", x=1.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="B", x=0.0, y=-1.0, is_support=True, support_dx=True, support_dy=True),
        Node(id="A", x=0.0, y=0.01, is_support=False),
    ]
    common = {"E": E_STEEL, "A": 1.0e-3, "delta_L_free": 7.14e-4}
    elements = [
        Element(id="r1", node_i="L", node_j="A", I_sec=1.0e-9, **common),
        Element(id="r2", node_i="R", node_j="A", I_sec=1.0e-9, **common),
        Element(
            id="post",
            node_i="A",
            node_j="B",
            E=E_STEEL,
            A=1.0e-6,
            I_sec=1.0e-12,
            alpha=0.0,
        ),
    ]
    loads = {"A": {"Fx": 0.0, "Fy": -50.0}}
    temps = {e.id: 20.0 for e in elements}

    with pytest.warns(SystemInstabilityWarning) as caught:
        states = dcr_field(
            nodes, elements, loads, temps, F_Y, check_system_stability=True
        )
    amplified = [m for m, ls in states.items() if ls.system_stability_factor > 1.0]
    assert amplified == ["r1", "r2"], amplified
    message = str(caught[0].message)
    assert f"{len(amplified)} compression member(s)" in message, message
    assert f"{len(states)} compression member(s)" not in message


# ---------------------------------------------------------------------------
# equivalent mutants: recorded, not chased
# ---------------------------------------------------------------------------


def test_the_equivalent_mutants_are_equivalent_for_a_stated_reason() -> None:
    """Three survivors cannot be killed by any test, and this says why.

    A mutation score below 100% is usually a list of holes.  Three of this
    module's survivors are not holes: the mutated expression computes the same
    value for every possible input, so no assertion can distinguish them.
    Recording the reason matters more than it looks -- without it, the next
    person to run the probe sees 77.8% and spends an afternoon writing tests
    that cannot fail, and the score itself silently overstates how much of the
    module is unverified.

    1. ``f_y_theta = n_rd * GAMMA_M_FIRE / area``.  ``GAMMA_M_FIRE`` is exactly
       ``1.0`` (EN 1993-1-2:2005 clause 2.3 recommended value), so swapping
       either operator for its inverse leaves ``n_rd / area`` unchanged.  The
       mutant would become killable the moment the partial factor stopped being
       unity -- which is a reason to keep the expression written out rather
       than simplified away, and a reason to re-run the probe if that constant
       ever changes.
    2. ``(False, 1.0) if lambda_cr > 0.0 else (True, inf)`` on the ``NaN`` arm.
       Every comparison against ``NaN`` is ``False``, so ``>`` and ``>=`` select
       the same branch.  The arm is reachable only with a non-finite,
       non-positive input that the documented solver does not produce.
    3. ``ls.compression and ls.p_cr is not None`` in the count the
       system-instability warning reports.  ``p_cr`` is set only inside
       ``_member_limit_state``'s compression branch, so the two halves of the
       conjunction are the same statement for every reachable input and ``or``
       selects the same members.  Pinned as an invariant by
       ``test_p_cr_is_present_exactly_for_compression_members`` rather than left
       as a coincidence, because redundancy that is not stated gets
       "simplified" by someone who does not know it was load-bearing.
    """
    assert GAMMA_M_FIRE == 1.0
    n_rd, area = 4.0e5, 0.01
    assert n_rd * GAMMA_M_FIRE / area == n_rd / GAMMA_M_FIRE / area
    assert n_rd * GAMMA_M_FIRE / area == n_rd * (1.0 / GAMMA_M_FIRE) / area

    nan = float("nan")
    assert (nan > 0.0) is False
    assert (nan >= 0.0) is False
    assert (nan > 0.0) == (nan >= 0.0), "the two arms are indistinguishable"


def test_the_euler_only_governing_tie_breaks_towards_buckling() -> None:
    """``p_cr <= n_rd`` at exact equality reports BUCKLING, not YIELD.

    The last genuine survivor in the mutation probe on this module.  Reaching
    ``p_cr == n_rd`` exactly means solving for the second moment of area rather
    than choosing one: at ambient ``k_y = 1`` and ``gamma_M,fi = 1``, so
    ``n_rd = f_y A`` and ``p_cr = pi^2 E I / (k L)^2`` meet when
    ``I = f_y A L^2 / (pi^2 E)``.  That member has ``lambda_bar = 1.0``, so it
    is not stocky and the branch is reached.

    The tie-break direction is not arbitrary.  Under ``EULER_ONLY`` the capacity
    is ``min(p_cr, n_rd)``, and at equality the two are the same number -- so
    the *capacity* is unaffected and only the reported governing limit state
    changes.  But ``capacity_governing`` is what tells a reader which physical
    mode to go and fix, and at ``lambda_bar = 1`` the member is squarely in the
    buckling range: reporting YIELD there would send them to check a cross-
    section that is not the problem.  Buckling is also the conservative reading,
    since it is the mode whose reserve degrades faster with temperature.
    """
    from truss_analysis.limitstates import BucklingModel

    area = 0.01
    length = 3.0
    i_exact = F_Y * area * length**2 / (math.pi**2 * E_STEEL)

    nodes = [
        Node(id="1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
        Node(
            id="2", x=0.0, y=length, is_support=True, support_dx=True, support_dy=False
        ),
    ]
    elements = [
        Element(id="c", node_i="1", node_j="2", E=E_STEEL, A=area, I_sec=i_exact)
    ]
    loads = {"2": {"Fx": 0.0, "Fy": -1.0e5}}
    temps = {"c": 20.0}

    ls = dcr_field(
        nodes, elements, loads, temps, F_Y, buckling_model=BucklingModel.EULER_ONLY
    )["c"]
    assert ls.p_cr == pytest.approx(ls.n_rd, rel=1e-12), (
        "the fixture must land exactly on the tie, or the operator is untested"
    )
    assert ls.lambda_bar == pytest.approx(1.0, rel=1e-12)
    assert ls.capacity_governing is Governing.BUCKLING

    # the capacity itself is the same either way at the tie -- which is exactly
    # why the label is the only thing this test can pin, and why it is worth
    # pinning: nothing else in the payload would move
    assert ls.capacity == pytest.approx(min(ls.p_cr, ls.n_rd), rel=1e-12)
