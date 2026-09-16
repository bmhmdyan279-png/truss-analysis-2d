"""Exception hierarchy for the truss analysis library.

All library-specific exceptions derive from :class:`TrussError`, so callers
can catch the whole family with a single ``except TrussError`` clause while
still being able to discriminate individual failure modes.
"""

from __future__ import annotations


class TrussError(Exception):
    """Base class for all errors raised by this library."""


class AssemblyError(TrussError):
    """Raised when global matrix assembly encounters inconsistent input."""


class InputValidationError(TrussError):
    """Raised when the input model (nodes, elements, loads) is invalid."""


class EnergyValidationError(TrussError):
    """Raised when the work-energy balance of a solve does not close."""


class SingularMatrixError(TrussError):
    """Raised when the free-free stiffness matrix is rank deficient."""


class UnitConversionError(TrussError):
    """Raised for unknown unit systems or unknown quantity keys."""


class EigenConvergenceError(TrussError):
    """Raised when an iterative eigensolver fails to converge.

    Emitted by :func:`truss_analysis.stability.linearized_buckling_load_factor`
    on the sparse Lanczos path (large models, ``eigen_solver="sparse"``) when
    ARPACK exhausts its iteration budget and the dense fallback is not
    affordable.  The buckling load factor is a safety-critical quantity, so a
    non-converged iteration is reported as an error rather than returned as a
    best-effort Ritz value: an unconverged ``lambda_cr`` is indistinguishable
    from a converged one to the caller, and silently optimistic is the worst
    possible failure mode.  Retry with ``eigen_solver="dense"``, more modes,
    or a coarser model.
    """


class IllConditionedWarning(UserWarning):
    """Warning issued when cond(K_ff) exceeds the screening threshold."""


class BucklingCheckWarning(UserWarning):
    """Warning issued when a compressed member cannot be assessed for buckling.

    Emitted by :func:`truss_analysis.postprocess.calculate_buckling` when a
    member carrying compression has no usable ``I_sec`` or length. Such a
    member is reported with ``status="unknown"`` and ``safe=False``: missing
    data must never be interpreted as a pass.
    """


class LegacyBucklingModelWarning(BucklingCheckWarning):
    """Warning issued when the legacy ``EULER_ONLY`` capacity model is selected.

    ``BucklingModel.EULER_ONLY`` takes the member capacity as
    ``min(N_cr, N_Rd)``, ignoring residual stresses and initial
    out-of-straightness, and overestimates the capacity of
    intermediate-slenderness members by roughly 15% at ``lambda_bar ~ 2``.  It
    is kept because results computed with it must stay reproducible bit for bit;
    ``EUROCODE_CHI`` is the code-compliant path.

    The same diagnostic used to be emitted under two different categories --
    a bare :class:`UserWarning` from :mod:`truss_analysis.limitstates` and
    :class:`BucklingCheckWarning` from :mod:`truss_analysis.reliability`.  A
    caller filtering on one silently missed the other, and a bare ``UserWarning``
    cannot be filtered precisely at all, since every third-party library uses it.
    Deriving from ``BucklingCheckWarning`` keeps existing
    ``pytest.warns(BucklingCheckWarning)`` and ``filterwarnings`` entries working
    while making the specific case addressable.
    """


class LargeDisplacementWarning(UserWarning):
    """Warning issued when the linear kinematics assumption is being stretched.

    The whole library is a small-displacement, first-order solver: the
    stiffness is built on the undeformed geometry and second-order (P-Delta)
    effects are out of scope. Emitted by
    :func:`truss_analysis.postprocess.check_displacement_magnitude` (wired
    into :func:`truss_analysis.main.run`) when the largest nodal
    displacement exceeds a documented fraction of the structure's
    characteristic length -- past that point the reported forces and
    displacements are an *approximation of unknown conservatism*, and a
    geometrically nonlinear analysis is warranted. The linearised
    bifurcation check :func:`truss_analysis.stability.linearized_buckling_load_factor`
    is the in-scope companion diagnostic for stability concerns.
    """


class InputIgnoredWarning(UserWarning):
    """Warning issued when a recognised-but-unapplied input key is supplied.

    The model parser accepts several keys for compatibility with generated
    models. Silently swallowing a key that the user believes controls physics
    is worse than refusing it, so any key that is parsed but not applied
    raises this warning instead.
    """


class UnitAmbiguityWarning(UserWarning):
    """Warning issued when an input quantity's unit convention is ambiguous.

    Used for the Imperial mass-density trap: ``515.379`` converts
    slug/ft^3 to kg/m^3 whereas ``16.0185`` converts lbm/ft^3 (pcf). Both are
    in everyday use, and choosing the wrong one is a silent factor-of-16
    error in self-weight.
    """


class ShallowSystemWarning(UserWarning):
    """Warning issued when the system geometry is shallow (rise/span < 0.1).

    Linearised bifurcation analysis approximates the true snap-through limit
    point with an error that scales as O(theta_0^2) where theta_0 is the
    initial rise angle. For shallow systems (rise-to-span ratio below ~0.1),
    the linearised lambda_cr can be significantly optimistic compared to the
    actual collapse load. This warning flags such geometries so engineers can
    supplement with geometrically nonlinear analysis (round-5 audit C6#9,
    C7§4.2: member-specific or strain-based shallow detection is preferred).
    """


class AmbiguousModeWarning(UserWarning):
    """Warning issued when buckling modes have repeated eigenvalues.

    When the buckling load factor has multiplicity > 1, the returned mode
    shape is implementation-dependent (any vector in the eigenspace is
    valid). The specific mode returned by eigh/eigsh may vary with LAPACK
    version, rounding errors, or tiny perturbations. Engineers should
    examine all modes in the repeated eigenspace for physical interpretation.
    """


class LumpedCapacityWarning(UserWarning):
    """Warning issued when the lumped-capacitance member-heating model is stretched.

    :func:`truss_analysis.thermal.fire_curve.steel_temperature` assumes the
    steel temperature is *uniform over the cross-section*, which is what
    EN 1993-1-2 §4.2.2.2 endorses for unprotected members.  That assumption
    holds while the Biot number ``h (V/A_m) / lambda`` stays small, i.e. while
    the section factor ``A_m/V`` is large enough for the surface heat input to
    be spread through the whole section faster than it arrives.

    Emitted when ``A_m/V`` falls below
    :data:`~truss_analysis.thermal.fire_curve.LUMPED_SECTION_FACTOR_LIMIT`
    (50 1/m, the reference threshold for a "thick" section): such a member
    develops a real through-thickness gradient, its core lags its surface,
    and a single lumped temperature overstates how much of the section has
    been weakened.  The answer is not wrong so much as un-conservative, which
    is why it is a warning rather than a silent assumption.  A heat-conduction
    (finite-element) solution is out of scope for this library; see the
    module docstring's scope statement.
    """


class ConstantAlphaWarning(UserWarning):
    """Warning issued when a hot field is expanded with a constant ``alpha``.

    The imposed-elongation term is ``alpha * (T - T_ref) * L``.  With a constant
    ``alpha`` -- ``Element.alpha``, the ambient coefficient the user supplied --
    that is a *linear* approximation of the EN 1993-1-2 thermal-elongation curve
    ``eps_th(T)``, which is itself nonlinear and departs from any straight line
    through the origin by a wide margin once the steel is hot.

    Measured against the curve, with ``alpha = 1.2e-5``:

    .. code-block:: text

        theta [degC]   100    200    300    400    500    600    700
        understated   3.9%   6.8%   9.6%  12.3%  14.8%  17.1%  19.4%

    In a restrained member the thermal force is proportional to that strain, so
    a 600 degC fire analysis run with a constant ``alpha`` reports a restrained
    force about 17% *low* -- and a low demand against an unchanged capacity is an
    un-conservative DCR.  It is not a modelling preference: the same library
    exposes the secant coefficient
    :func:`~truss_analysis.material.steel_eurocode.effective_alpha`, which
    reproduces ``eps_th`` exactly, so the accurate path exists and is one flag
    away.

    Emitted by
    :func:`truss_analysis.criticality.engine.prestress_lengths` when a supplied
    temperature field exceeds
    :data:`~truss_analysis.criticality.engine.ALPHA_CONSTANCY_LIMIT` and
    ``use_effective_alpha`` is not set.  The default stays constant-``alpha``
    because changing it would silently move every previously published result;
    the warning exists so that the choice is made deliberately rather than
    inherited.
    """


class ParametricFireRangeWarning(UserWarning):
    """Warning issued when a parametric fire's inputs leave the code's range.

    The EN 1991-1-2 Annex A parametric temperature-time curve is a fit to a
    bounded set of compartment fire data, and the standard states the range it
    was fitted over.  A compartment outside that range still produces a smooth,
    plausible curve from the same equations; it is simply an extrapolation of
    the fit rather than a code-compliant gas temperature history.

    Emitted by :class:`truss_analysis.thermal.fire_curve.ParametricFire` when
    ``q_td`` falls outside
    :data:`~truss_analysis.thermal.fire_curve.Q_TD_RANGE_OF_VALIDITY`.
    Previously reported as a bare ``UserWarning`` whose text described itself as
    "LumpedCapacityWarning-adjacent" -- naming a class that concerns the
    lumped-capacitance *member* model rather than the *gas curve*, so a caller
    filtering on category could not catch it and a caller reading the message
    was pointed at the wrong physics.
    """


class SteelTemperatureRangeWarning(UserWarning):
    """Warning issued when steel temperature leaves the material model's range.

    Every property in :mod:`truss_analysis.material.steel_eurocode` is read from
    EN 1993-1-2:2005 Table 3.1 and clause 3.4, which tabulate carbon steel from
    20 degC to 1200 degC.  Outside that band the accessors *clamp* -- which is
    the documented behaviour and the right one for an interpolator -- but a
    clamp is silent, and a silent clamp in a fire calculation is the same
    failure mode this library refuses everywhere else: a thin member in a long
    fire can pass 1200 degC, and the answer then looks perfectly ordinary while
    being an extrapolation of the last tabulated row rather than a code-compliant
    value.

    Emitted by :func:`truss_analysis.thermal.fire_curve.steel_temperature` and
    :func:`truss_analysis.thermal.protection.protected_steel_temperature` when
    the computed history, or the supplied initial temperature, leaves the range.
    The result is still returned; the warning states that its upper end is
    outside the standard's validity envelope.
    """


class IllConditionedPerturbationWarning(UserWarning):
    """Warning issued when a Woodbury perturbation solve is numerically weak.

    Issued (not raised) by
    :func:`truss_analysis.criticality.engine.perturb_multi` when the ``r x r``
    Woodbury core is ill-conditioned
    (``cond > PERTURB_COND_WARN``) or the solve fails its backward-error
    check (``relative residual > PERTURB_RESID_WARN``).  Such a core passes
    the singularity gate but returns digits that are largely noise, and the
    caller -- retrofit triage, multi-member criticality -- would rank members
    on them.  Verify against
    :func:`~truss_analysis.criticality.engine.brute_force_ci` or split the
    simultaneous perturbation (round-5 audit C8#4, wired up in round 6).
    """


class SystemInstabilityWarning(UserWarning):
    """Warning issued when a system-level instability governs the reported DCRs.

    Issued (not raised) by :func:`truss_analysis.limitstates.dcr_field` when it
    is called with ``check_system_stability=True`` and the linearised
    bifurcation load factor satisfies ``lambda_cr <= 1`` -- that is, when the
    *system* loses positive definiteness of its tangent stiffness at or below
    the load level actually applied, independently of whether any single
    member has reached its own ``DCR = 1``.

    This exists because the adjustment it accompanies used to be applied
    silently.  A member DCR was multiplied by ``1 / lambda_cr`` through
    ``object.__setattr__`` on a frozen dataclass, with no warning and no field
    recording that it had happened, so ``dcr`` stopped being reconstructible
    from ``|axial_force| / capacity`` and a caller reading the payload could
    not tell an amplified number from a member-level one.  In a library whose
    rule is never to emit a silent number, that was the one place the rule was
    broken on a safety-relevant quantity.

    The warning is deliberately separate from the member-level checks: a
    truss whose compressed chords are code-safe member by member can still
    bifurcate as a system, and the two failures have different remedies.

    .. note::

       ``lambda_cr`` is the criticality of the *linearised tangent state*, not
       the ultimate load of the real structure.  For shallow systems the true
       collapse is a limit point (snap-through) that the linearised factor
       approximates from the base configuration; see
       :func:`truss_analysis.stability.imperfection_sensitivity` for how much
       of the reserve survives an imperfection.  Post-buckling paths and
       arc-length continuation are out of scope.
    """
