"""Cross-section models for truss members.

The library ships an *idealised* square hollow-section (HSS) model and an
immutable catalogue type through which real commercial tables can be
plugged in later without code changes.

Models
------
* :class:`SquareHSS` - an **idealised** square hollow section (uniform wall
  thickness, sharp corners): ``A = b^2 - (b - 2t)^2 = 4 t (b - t)`` and
  ``I = (b^4 - (b - 2t)^4) / 12`` (exact for the idealised shape; equal
  about both axes by symmetry).  These are NOT commercial product
  properties.
* :func:`idealised_square_hss` - solves ``(b, t)`` from a target area and a
  documented width-to-thickness ratio ``r = b/t``:
  ``A = 4 b^2 (r - 1) / r^2``  =>  ``b = (r / 2) * sqrt(A / (r - 1))``.
* :class:`SectionCatalog` - immutable named-section table with nearest-area
  lookup.
* :func:`euler_buckling_load` - ``P_cr = pi^2 E I / (k L)^2``.

Notes
-----
The solid-square formula ``I = A^2 / 12`` is the exact second moment of
area of a *solid* square of side ``sqrt(A)``; it is NOT a valid hollow
section approximation and is deliberately not used anywhere here.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .exceptions import BucklingCheckWarning

__all__ = [
    "BUCKLING_CURVE_ALPHA",
    "DEFAULT_THICKNESS_RATIO",
    "FIRE_IMPERFECTION_FACTOR",
    "LAMBDA_BAR_BUCKLING_LIMIT",
    "MIN_THICKNESS_RATIO",
    "SectionCatalog",
    "SquareHSS",
    "buckling_reduction_factor",
    "euler_buckling_load",
    "idealised_square_hss",
    "non_dimensional_slenderness",
    "thickness_ratio_from_section",
]

#: Smallest admissible width-to-thickness ratio ``r = b/t``: at ``r = 2`` the
#: wall closes the void (``2t = b``).  Exported (it was ``_MIN_THICKNESS_RATIO``)
#: because callers that invert the section model need the same bound the
#: forward model enforces.
MIN_THICKNESS_RATIO = 2.0
_MIN_THICKNESS_RATIO = MIN_THICKNESS_RATIO  # backwards-compatible alias

#: Fallback ``b/t`` when a member carries no usable ``I_sec`` and its ratio
#: therefore cannot be recovered.  25.0 is the value the parametric generator
#: has always used; it is a *declared assumption*, and
#: :func:`~truss_analysis.retrofit.actions.apply_decision` warns whenever it
#: has to fall back to it (round-6 audit C7).
DEFAULT_THICKNESS_RATIO = 25.0

#: Imperfection factors ``alpha`` of the five flexural buckling curves,
#: EN 1993-1-1:2005 Table 6.1 / Table 6.2.
BUCKLING_CURVE_ALPHA: dict[str, float] = {
    "a0": 0.13,
    "a": 0.21,
    "b": 0.34,
    "c": 0.49,
    "d": 0.76,
}

#: EN 1993-1-2:2005 4.2.3.1(3) reduces the imperfection factor for the fire
#: design situation, because residual stresses and geometric imperfections
#: matter less once the material has softened: ``alpha_theta = 0.65 * alpha_c``.
FIRE_IMPERFECTION_FACTOR = 0.65

#: Below this non-dimensional slenderness buckling need not be checked at all
#: (EN 1993-1-1:2005 6.3.1(4)); the member is yield-governed.
LAMBDA_BAR_BUCKLING_LIMIT = 0.2


@dataclass(frozen=True)
class SquareHSS:
    """Idealised square hollow section with uniform wall thickness.

    Parameters
    ----------
    name : str
        Identifier (e.g. ``"idealised-r25-A0.01"``).
    b : float
        Outer width [m].
    t : float
        Wall thickness [m]; must satisfy ``0 < 2t < b``.

    Raises
    ------
    ValueError
        If ``b`` or ``t`` is non-positive, or the wall leaves no void.
    """

    name: str
    b: float
    t: float

    def __post_init__(self) -> None:
        """Validate that the section geometry leaves a positive void."""
        if self.b <= 0.0:
            msg = f"outer width must be > 0, got {self.b}"
            raise ValueError(msg)
        if self.t <= 0.0:
            msg = f"wall thickness must be > 0, got {self.t}"
            raise ValueError(msg)
        if 2.0 * self.t >= self.b:
            msg = f"wall thickness leaves no void: 2t={2 * self.t} >= b={self.b}"
            raise ValueError(msg)

    @property
    def width_to_thickness(self) -> float:
        """Return the width-to-thickness ratio ``r = b / t`` [-]."""
        return self.b / self.t

    @property
    def area(self) -> float:
        """Return the exact cross-sectional area ``4 t (b - t)`` [m^2]."""
        return 4.0 * self.t * (self.b - self.t)

    @property
    def i_sec(self) -> float:
        """Return the exact second moment of area about either axis [m^4]."""
        bi = self.b - 2.0 * self.t
        return (self.b**4 - bi**4) / 12.0

    @property
    def radius_of_gyration(self) -> float:
        """Return the radius of gyration ``sqrt(I / A)`` [m]."""
        return math.sqrt(self.i_sec / self.area)


def idealised_square_hss(
    area: float, thickness_ratio: float = DEFAULT_THICKNESS_RATIO
) -> SquareHSS:
    """Construct the idealised square HSS with a given area and ``b/t`` ratio.

    Inverts ``A = 4 b^2 (r - 1) / r^2`` for ``b``, then sets ``t = b / r``.

    Parameters
    ----------
    area : float
        Target cross-sectional area [m^2]; must be positive.
    thickness_ratio : float, default 25.0
        Width-to-thickness ratio ``r = b/t``; must be at least 2 (otherwise
        the section is solid or has a negative void).

    Returns
    -------
    SquareHSS
        Section whose :attr:`~SquareHSS.area` reproduces ``area`` exactly
        (to floating-point round-off).

    Raises
    ------
    ValueError
        If ``area`` is non-positive or ``thickness_ratio`` is below 2.
    """
    if area <= 0.0:
        msg = f"area must be > 0, got {area}"
        raise ValueError(msg)
    if thickness_ratio < _MIN_THICKNESS_RATIO:
        msg = (
            f"thickness_ratio must be >= {_MIN_THICKNESS_RATIO}, got {thickness_ratio}"
        )
        raise ValueError(msg)
    r = thickness_ratio
    b = (r / 2.0) * math.sqrt(area / (r - 1.0))
    t = b / r
    return SquareHSS(name=f"idealised-r{r:g}-A{area:g}", b=b, t=t)


#: Offset keeping the root bracket strictly inside the hollow range.
_R_EPS = 1e-9


def _kappa_of_r(r: float) -> float:
    """Dimensionless ``I / A^2`` of the idealised square HSS at ratio ``r``."""
    return (r**3 - 3.0 * r**2 + 4.0 * r - 2.0) / (24.0 * (r - 1.0) ** 2)


def thickness_ratio_from_section(area: float, i_sec: float) -> float:
    """Recover ``r = b/t`` from a member's ``(A, I)`` pair.

    The idealised square HSS is a two-parameter family, and the dimensionless
    combination ``I / A^2`` depends on ``r`` alone:

    .. code-block:: text

        A = 4 t (b - t),  I = (b^4 - (b - 2t)^4) / 12,  b = r t
        =>  I / A^2 = (r^3 - 3 r^2 + 4 r - 2) / (24 (r - 1)^2)

    That map is strictly increasing on ``r > 2`` -- from ``1/12`` at the
    solid limit to ``r/24`` asymptotically -- so it is invertible and the
    recovered ratio is unique.  Being independent of ``A``, it is exactly the
    quantity needed to enlarge a member while preserving *its own* wall
    slenderness rather than an assumed one.

    Parameters
    ----------
    area : float
        Cross-sectional area ``A`` [m^2]; must be positive.
    i_sec : float
        Second moment of area ``I`` [m^4]; must be positive.

    Returns
    -------
    float
        Width-to-thickness ratio ``r = b/t`` of the idealised square HSS that
        reproduces ``I / A^2``.

    Raises
    ------
    ValueError
        If ``area`` or ``i_sec`` is non-positive, or if ``I / A^2`` falls
        below ``1/12`` -- the value at ``r = 2``, below which no hollow
        square section can be that stocky and the input is therefore not an
        idealised HSS at all (a solid or doubly-symmetric rolled section,
        for instance).

    See Also
    --------
    idealised_square_hss : the forward map this inverts.
    """
    if area <= 0.0:
        msg = f"area must be > 0 to recover a thickness ratio, got {area}"
        raise ValueError(msg)
    if i_sec <= 0.0:
        msg = f"i_sec must be > 0 to recover a thickness ratio, got {i_sec}"
        raise ValueError(msg)

    kappa = float(i_sec) / (float(area) * float(area))
    lo = MIN_THICKNESS_RATIO + _R_EPS
    kappa_min = _kappa_of_r(lo)
    if kappa < kappa_min:
        msg = (
            f"I/A^2 = {kappa:.6g} is below {kappa_min:.6g}, the solid limit "
            f"(r = {MIN_THICKNESS_RATIO:g}) of the idealised square HSS; this "
            "section is not representable in that family"
        )
        raise ValueError(msg)

    # kappa(r) ~ r/24 for large r, so 24*kappa is a tight initial guess; pad
    # the bracket generously rather than iterating to find one.
    hi = max(4.0 * MIN_THICKNESS_RATIO, 48.0 * kappa + 16.0)
    while _kappa_of_r(hi) < kappa:  # pragma: no cover - defensive growth
        hi *= 2.0

    # The bracket is checked before brentq rather than left to it.  The `<`
    # guard above is exact in the comparison it makes but ``kappa_min`` is
    # itself a floating-point evaluation one _R_EPS inside the boundary, so a
    # section supplied at exactly the solid limit (``I = A^2 / 12``) can land on
    # either side of it by a rounding.  When it lands inside, ``f(lo)`` is zero
    # or positive and ``brentq`` raises its own
    # ``ValueError: f(a) and f(b) must have different signs`` -- a raw scipy
    # message about a bracket, where the docstring promises an explanation about
    # the section not being representable.  Returning the solid limit is the
    # physically right answer there, and saying so beats an internal error.
    f_lo = _kappa_of_r(lo) - kappa
    if f_lo >= 0.0:
        return lo

    root = brentq(
        lambda r: _kappa_of_r(r) - kappa,
        lo,
        hi,
        xtol=1e-12,
        rtol=4.0 * float(np.finfo(float).eps),
        maxiter=200,
    )
    return float(root)


@dataclass(frozen=True)
class SectionCatalog:
    """Immutable table of named sections with nearest-area selection.

    The library default catalogue is empty on purpose: commercial tables
    (e.g. EN 10210) are licensed data and must be supplied by the user via
    :meth:`from_rows`. Until then :func:`idealised_square_hss` is the
    documented parametric model.

    Attributes
    ----------
    name : str
        Catalogue identifier.
    rows : tuple[SquareHSS, ...]
        Immutable sequence of available sections.
    """

    name: str
    rows: tuple[SquareHSS, ...]

    @classmethod
    def from_rows(cls, name: str, rows: Sequence[SquareHSS]) -> SectionCatalog:
        """Build a catalogue from a sequence of sections (copied to a tuple)."""
        return cls(name=name, rows=tuple(rows))

    def nearest_to_area(self, area: float) -> SquareHSS:
        """Return the row whose area is closest (relative distance) to ``area``.

        Raises
        ------
        ValueError
            If the catalogue is empty.
        """
        if not self.rows:
            msg = f"catalogue '{self.name}' is empty"
            raise ValueError(msg)
        return min(self.rows, key=lambda row: abs(row.area - area) / row.area)


def euler_buckling_load(
    i_sec: float,
    length: float,
    youngs_modulus: float,
    effective_length_factor: float = 1.0,
) -> float:
    """Return the Euler elastic buckling load ``P_cr = pi^2 E I / (k L)^2`` [N].

    Parameters
    ----------
    i_sec : float
        Second moment of area [m^4].
    length : float
        Member length ``L`` [m]; must be positive.
    youngs_modulus : float
        Young's modulus ``E`` [Pa].
    effective_length_factor : float, default 1.0
        Effective-length factor ``k`` (pin-ended column: 1.0).

    Returns
    -------
    float
        Elastic critical (buckling) load [N].

    Raises
    ------
    ValueError
        If ``length`` is non-positive.
    """
    if length <= 0.0:
        msg = f"length must be > 0, got {length}"
        raise ValueError(msg)
    kl = effective_length_factor * length
    return math.pi**2 * youngs_modulus * i_sec / kl**2


def non_dimensional_slenderness(
    area: float, yield_strength: float, elastic_critical_load: float
) -> float:
    """Return the non-dimensional slenderness ``lambda_bar`` [-].

    .. code-block:: text

        lambda_bar = sqrt(A * f_y / N_cr)

    ``N_cr`` is the Euler elastic critical load for the relevant buckling
    length, from :func:`euler_buckling_load`. At elevated temperature both
    ``f_y`` and ``E`` must be the temperature-reduced values, which is what
    makes the fire slenderness differ from the ambient one: since ``k_E``
    falls faster than ``k_y``, ``lambda_bar`` *grows* as the member heats, so
    a stocky cold member can become slender in fire.

    Parameters
    ----------
    area : float
        Cross-sectional area ``A`` [m^2].
    yield_strength : float
        Yield strength ``f_y`` [Pa] at the temperature of interest.
    elastic_critical_load : float
        Euler critical load ``N_cr`` [N] at the temperature of interest.

    Returns
    -------
    float
        Non-dimensional slenderness. ``inf`` when ``elastic_critical_load``
        is non-positive, which correctly reports an infinitely slender member.
    """
    if elastic_critical_load <= 0.0:
        return math.inf
    return math.sqrt(area * yield_strength / elastic_critical_load)


def buckling_reduction_factor(
    lambda_bar: float,
    curve: str = "c",
    *,
    fire: bool = False,
) -> float:
    """Return the flexural buckling reduction factor ``chi`` [-].

    Implements the Eurocode buckling curve,

    .. code-block:: text

        Phi   = 0.5 * [1 + alpha * (lambda_bar - 0.2) + lambda_bar**2]
        chi   = 1 / (Phi + sqrt(Phi**2 - lambda_bar**2)),   chi <= 1

    with ``alpha`` taken from :data:`BUCKLING_CURVE_ALPHA`, and scaled by
    :data:`FIRE_IMPERFECTION_FACTOR` (0.65) when ``fire`` is set, per
    EN 1993-1-2:2005 4.2.3.1(3).

    Why this replaces a bare Euler load
    -----------------------------------
    The elastic critical load ``N_cr`` alone is only the asymptote of the
    buckling curve for very slender members. At intermediate slenderness
    (``lambda_bar`` roughly 0.5 to 1.5) it overestimates the true capacity
    substantially, because it ignores residual stresses from rolling and
    initial geometric out-of-straightness. Using ``chi`` interpolates
    correctly between the two limits: ``chi -> 1`` as ``lambda_bar -> 0``
    (yield governed) and ``chi -> 1 / lambda_bar**2`` as ``lambda_bar -> inf``
    (Euler governed).

    Parameters
    ----------
    lambda_bar : float
        Non-dimensional slenderness from :func:`non_dimensional_slenderness`.
    curve : str, default "c"
        Buckling curve label, one of ``"a0"``, ``"a"``, ``"b"``, ``"c"``,
        ``"d"`` (imperfection factors 0.13/0.21/0.34/0.49/0.76). Curve
        ``c`` is the usual choice for thin-walled and cold-formed hollow
        sections. "Conservative default" here is a *proven ordering claim*,
        not an intuition: for every ``lambda_bar > 0.2`` the reduction
        factors satisfy ``chi_a0 >= chi_a >= chi_b >= chi_c >= chi_d``
        (higher imperfection factor => lower chi), so curve ``c`` is
        conservative with respect to the curves typically specified for
        such sections (``a0``/``a``/``b``); ``d`` is more onerous still and
        exists for the sections the codes assign to it. Pinned by
        ``test_chi_curve_ordering_is_proven``. The choice of curve is a
        *user modelling input* (it follows from the section type and
        buckling axis per the code's selection tables), not something the
        default can decide universally.
    fire : bool, default False
        Apply the EN 1993-1-2 imperfection reduction ``alpha = 0.65 alpha_c``.

    Returns
    -------
    float
        Reduction factor in ``(0, 1]``. Returns ``1.0`` for a non-positive
        slenderness, where buckling cannot occur.

    Raises
    ------
    ValueError
        If ``curve`` is not a recognised buckling curve label.

    Warns
    -----
    BucklingCheckWarning
        If ``fire`` is set with a curve other than ``"c"``: the 0.65
        imperfection reduction of EN 1993-1-2:2005 4.2.3.1(3) is defined for
        buckling curve ``c`` (``alpha = 0.65 * 0.49``). Combining it with
        another curve's imperfection factor (e.g. ``0.65 * 0.21`` for curve
        ``a``) is an extrapolation no clause of the standard defines, so the
        result is reported as such rather than silently presented as code
        compliance.
    """
    try:
        alpha_c = BUCKLING_CURVE_ALPHA[curve]
    except KeyError:
        known = ", ".join(sorted(BUCKLING_CURVE_ALPHA))
        msg = f"unknown buckling curve {curve!r}; expected one of {known}"
        raise ValueError(msg) from None

    if fire and curve != "c":
        warnings.warn(
            f"EN 1993-1-2 4.2.3.1(3) defines the {FIRE_IMPERFECTION_FACTOR} "
            f"imperfection reduction for buckling curve 'c'; fire=True with "
            f"curve {curve!r} uses alpha = {FIRE_IMPERFECTION_FACTOR} * "
            f"{alpha_c}, an extrapolation beyond the code",
            BucklingCheckWarning,
            stacklevel=2,
        )

    if not math.isfinite(lambda_bar) or lambda_bar <= 0.0:
        # An infinitely slender member has no capacity; a stocky one cannot
        # buckle. Handle both ends explicitly rather than through the formula.
        return 0.0 if math.isinf(lambda_bar) else 1.0

    alpha = FIRE_IMPERFECTION_FACTOR * alpha_c if fire else alpha_c
    phi = 0.5 * (1.0 + alpha * (lambda_bar - LAMBDA_BAR_BUCKLING_LIMIT) + lambda_bar**2)
    discriminant = phi**2 - lambda_bar**2
    # The discriminant is non-negative for alpha >= 0 by construction, but
    # round-off near lambda_bar ~ 0 can make it marginally negative.
    root = math.sqrt(discriminant) if discriminant > 0.0 else 0.0
    chi = 1.0 / (phi + root)
    return min(1.0, chi)
