"""Standard fire exposure curves and the member-heating solver.

This module closes the gap the round-5 audit called the single biggest
missing link between "temperature-dependent structural response" and "fire
engineering": the library could already consume a *prescribed* steel
temperature field, but nothing produced one from a fire exposure. It adds
two pieces:

1. :func:`iso_834_temperature` -- the ISO 834 / EN 1991-1-2 standard
   time--temperature curve, the exposure side of a nominal fire.
2. :func:`steel_temperature` -- the EN 1993-1-2 §4.2.2.2 **lumped heat
   capacity** ("thermally lumped") solution for an *unprotected* steel
   member, integrating

   .. code-block:: text

       rho_a c_a(theta_a) V d(theta_a)/dt = k_sh * A_m * h_net(theta_g, theta_a)

   with ``h_net = h_c + h_r`` (convective + radiative net heat flux), the
   section factor ``A_m/V`` [1/m], ``rho_a`` the unit mass and ``c_a`` the
   temperature-dependent specific heat, both from the EN 1993-1-2 single
   source of truth in :mod:`truss_analysis.material.steel_eurocode`.

Scope, stated plainly so the model is never oversold (round-5 audit, C4 §2
-- the distinction between *thermal* and *structural* fire analysis):

* The member temperature is **uniform over the cross-section**. This is the
  lumped-capacitance assumption EN 1993-1-2 §4.2.2.2 itself endorses for
  unprotected members; it is NOT a thermal-gradient (finite-element
  heat-conduction) solution. Thick or protected sections with real through-
  thickness gradients are out of scope.
* The input is a gas-temperature history ``theta_g(t)``; anything that
  produces one works (the built-in ISO 834, a parametric curve, or measured
  data). There is no zone/two-zone fire model and no combustion physics.
* The output ``theta_a(t)`` is exactly what the structural side already
  consumes: feed the final value as the member ``delta_T`` field (relative
  to :data:`~truss_analysis.criticality.scenarios.T_AMBIENT`) into
  :func:`~truss_analysis.limitstates.dcr_field` and the fire-resistance
  chain is complete end to end.

The integrator is explicit Runge--Kutta 4 with a step capped at
``max_step_s`` (default 5 s, the ceiling EN 1993-1-2 §4.2.2.2(3) puts on
``Δt``). RK4 keeps the temperature-dependent ``c_a(theta_a)`` spike near
730 degC accurate without an implicit solve.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..exceptions import LumpedCapacityWarning
from ..material.steel_eurocode import (
    FloatOrArray,
    specific_heat,
    thermal_conductivity,
    unit_mass,
)

__all__ = [
    "ALPHA_C",
    "B_VALUE_LIMITS",
    "EPSILON_RES",
    "LUMPED_SECTION_FACTOR_LIMIT",
    "OPENING_FACTOR_LIMITS",
    "Q_TD_RANGE_OF_VALIDITY",
    "STEFAN_BOLTZMANN",
    "T_LIM_FIRE_GROWTH_MIN",
    "ParametricFire",
    "SteelHeatingResult",
    "biot_critical_section_factor",
    "h_net",
    "iso_834_temperature",
    "lumped_capacity_biot",
    "parametric_fire_temperature",
    "steel_temperature",
]

#: Convective heat-transfer coefficient ``alpha_c`` [W/(m^2 K)] for the
#: standard fire, EN 1991-1-2:2005 Table 3.2 (nominal fire, member surface).
ALPHA_C = 25.0

#: Stefan--Boltzmann constant ``sigma`` [W/(m^2 K^4)].
STEFAN_BOLTZMANN = 5.670e-8

#: Resultant emissivity ``epsilon_res`` [-] for unprotected steel exposed to
#: a fully engulfing fire. EN 1993-1-2:2005 §4.2.2.2 takes the product of
#: the flame and member emissivities; ``0.7`` is the conservative value the
#: code recommends for bare carbon steel and is the default here.
EPSILON_RES = 0.7

#: Ambient (stress-free reference) temperature [degC], matching
#: :data:`truss_analysis.criticality.scenarios.T_AMBIENT`.
_AMBIENT_C = 20.0

#: ISO 834 curve is defined only for t >= 0; cap the exponent argument.
_ISO834_COEF = 345.0

# --- EN 1991-1-2:2002 Annex A: parametric temperature-time curve ----------

#: Limits of validity of the opening factor ``O = A_v sqrt(h_eq) / A_t``
#: [m^0.5], EN 1991-1-2 A.1(3).
OPENING_FACTOR_LIMITS: tuple[float, float] = (0.02, 0.2)

#: Limits of validity of the thermal inertia ``b = sqrt(rho c lambda)``
#: [J/(m^2 s^0.5 K)], EN 1991-1-2 A.2.
B_VALUE_LIMITS: tuple[float, float] = (100.0, 2200.0)

#: Range of validity of the fire load density on the total enclosure area
#: ``q_td`` [MJ/m^2], EN 1991-1-2 A.2 note.  Outside it the curve is an
#: extrapolation of the fitting data, so the module warns rather than raises:
#: the standard states it as a range of applicability, not a hard bound.
Q_TD_RANGE_OF_VALIDITY: tuple[float, float] = (50.0, 1000.0)

#: Limiting duration ``t_lim`` [min] by fire growth rate, EN 1991-1-2
#: Table A.2.  ``t_max = max(0.2e-3 q_td / O, t_lim)``, so a fast-growing
#: fire in a lightly loaded compartment is bounded below by this value.
T_LIM_FIRE_GROWTH_MIN: dict[str, float] = {
    "slow": 25.0,
    "medium": 20.0,
    "fast": 15.0,
}

#: Reference values in the denominator of ``Gamma``, EN 1991-1-2 A.1(3).
_GAMMA_O_REF = 0.04
_GAMMA_B_REF = 1160.0

#: Asymptotic gas temperature of the parametric curve: 20 + 1325 [degC].
_PARAMETRIC_AMPLITUDE = 1325.0

# --- B7: applicability of the lumped-capacitance assumption ---------------

#: Section factor ``A_m/V`` [1/m] below which
#: :func:`steel_temperature` warns that the uniform-temperature assumption is
#: being stretched.  50 1/m is the reference threshold for a "thick" section;
#: the warning text also reports the Biot number computed from the module's
#: own heat-transfer physics, so the threshold is checkable rather than magic.
LUMPED_SECTION_FACTOR_LIMIT = 50.0

#: Gas temperature [degC] at which the effective surface coefficient for the
#: Biot estimate is evaluated.  Mid-range for a compartment fire, and far
#: enough into the radiative regime that the estimate is representative of
#: the exposure rather than of the first few seconds.
_BIOT_REFERENCE_THETA_G = 800.0


def iso_834_temperature(
    t: FloatOrArray,
    theta_0: float = _AMBIENT_C,
) -> FloatOrArray:
    """ISO 834 / EN 1991-1-2 standard time--temperature curve [degC].

    .. code-block:: text

        theta_g(t) = 345 * log10(8 t + 1) + theta_0,   t in MINUTES

    This is the nominal ("standard") fire used for prescriptive fire-rating.
    It reproduces the published curve to better than 1 degC at 5, 10, 15,
    30, 60 and 120 minutes (verified in ``tests/test_fire_curve.py``).

    Parameters
    ----------
    t : float or ndarray
        Time [minutes]; must be >= 0.
    theta_0 : float, default 20.0
        Initial (ambient) gas temperature [degC].

    Returns
    -------
    float or ndarray
        Gas temperature [degC], same shape as ``t``.

    Raises
    ------
    ValueError
        If any ``t`` is negative.
    """
    arr = np.asarray(t, dtype=float)
    if np.any(arr < 0.0):
        msg = f"iso_834_temperature: time must be >= 0, got min {float(arr.min())}"
        raise ValueError(msg)
    out: npt.NDArray[np.float64] = np.asarray(
        _ISO834_COEF * np.log10(8.0 * arr + 1.0) + float(theta_0), dtype=float
    )
    if np.ndim(t) == 0:
        return float(out)
    return out


@dataclass(frozen=True)
class ParametricFire:
    """EN 1991-1-2:2002 Annex A parametric temperature-time curve.

    Where :func:`iso_834_temperature` is a fixed prescriptive curve, the
    parametric curve is derived from the compartment: its ventilation, its
    fire load and the thermal inertia of its boundaries.  It therefore has a
    *real* peak temperature and a cooling phase, which is what
    performance-based design needs and what a nominal fire-rating curve
    cannot express.

    .. code-block:: text

        heating  (t* <= t*_max):
            theta_g = 20 + 1325 (1 - 0.324 e^{-0.2 t*}
                                   - 0.204 e^{-1.7 t*}
                                   - 0.472 e^{-19  t*})
        t*     = t * Gamma,          Gamma = (O / b)^2 / (0.04 / 1160)^2
        t_max  = max(0.2e-3 * q_td / O, t_lim)                [hours]
        t*_max = t_max * Gamma
        cooling (t* > t*_max):
            theta_g = theta_max - rate * (t* - t*_max * x)
            rate = 625                  if t*_max <= 0.5 h
                 = 250 (3 - t*_max)     if 0.5 < t*_max < 2 h
                 = 250                  if t*_max >= 2 h
        and theta_g >= theta_0 throughout.

    The instance is callable with time in **minutes**, the convention
    :func:`steel_temperature` expects of a ``fire_curve`` argument, so it
    drops straight in: ``steel_temperature(60, sf, fire_curve=ParametricFire(...))``.

    Validated against the Access Steel worked example SX042a-EN-EU (office
    compartment, ``O = 0.1024``, ``q_td = 181.8``, ``b = 1234``,
    ``t_lim = 20 min``): ``Gamma = 5.791``, ``t_max = 0.355 h``,
    ``t*_max = 2.056 h``, ``theta_max = 1052 degC`` and the cooling line
    ``theta_g = 1566 - 250 t*`` -- all reproduced to four significant
    figures in ``tests/test_fire_curve.py``.

    Attributes
    ----------
    opening_factor : float
        ``O = A_v sqrt(h_eq) / A_t`` [m^0.5]; must lie in
        :data:`OPENING_FACTOR_LIMITS`.
    q_td : float
        Design fire load density on the *total enclosure* area [MJ/m^2].
    b_value : float
        Thermal inertia ``b = sqrt(rho c lambda)`` of the boundary, area
        weighted and excluding the openings [J/(m^2 s^0.5 K)]; must lie in
        :data:`B_VALUE_LIMITS`.
    t_lim_min : float, default 20.0
        Limiting duration [min] for the fire growth rate (Table A.2: slow 25,
        medium 20, fast 15).
    theta_0 : float, default 20.0
        Initial gas temperature [degC]; also the floor the cooling phase
        stops at.

    Notes
    -----
    Scope: this is the main clause of Annex A, in which ``Gamma`` is formed
    from the opening factor.  The standard's *note* to A.1(1) gives a
    different ``Gamma = (b/1160)^2 / (q_td/420)^2`` for low-fire-load
    offices with ``b > 1200``; it is deliberately not implemented here,
    because it is a National-Annex-dependent special case and shipping an
    unvalidated second variant of a fire curve is worse than not shipping
    it.  Protected members (EN 1993-1-2 §4.2.5.2) are likewise out of scope
    of :func:`steel_temperature` -- see its docstring.
    """

    opening_factor: float
    q_td: float
    b_value: float
    t_lim_min: float = 20.0
    theta_0: float = _AMBIENT_C

    def __post_init__(self) -> None:
        """Validate the compartment parameters against the code's limits."""
        lo, hi = OPENING_FACTOR_LIMITS
        if not lo <= self.opening_factor <= hi:
            msg = (
                f"opening_factor O must lie in [{lo}, {hi}] m^0.5 "
                f"(EN 1991-1-2 A.1(3)), got {self.opening_factor}"
            )
            raise ValueError(msg)
        lo_b, hi_b = B_VALUE_LIMITS
        if not lo_b <= self.b_value <= hi_b:
            msg = (
                f"b_value must lie in [{lo_b}, {hi_b}] J/(m^2 s^0.5 K) "
                f"(EN 1991-1-2 A.2), got {self.b_value}"
            )
            raise ValueError(msg)
        if self.q_td <= 0.0:
            msg = f"q_td must be > 0 MJ/m^2, got {self.q_td}"
            raise ValueError(msg)
        if self.t_lim_min <= 0.0:
            msg = f"t_lim_min must be > 0, got {self.t_lim_min}"
            raise ValueError(msg)
        q_lo, q_hi = Q_TD_RANGE_OF_VALIDITY
        if not q_lo <= self.q_td <= q_hi:
            warnings.warn(
                f"LumpedCapacityWarning-adjacent scope note: q_td = "
                f"{self.q_td:g} MJ/m^2 is outside the [{q_lo}, {q_hi}] range "
                "of validity of the EN 1991-1-2 Annex A parametric curve; "
                "the result is an extrapolation of the standard's fitting "
                "data.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def gamma(self) -> float:
        """Return the time-scaling factor ``Gamma`` [-] (A.1(3))."""
        ratio = (self.opening_factor / self.b_value) / (_GAMMA_O_REF / _GAMMA_B_REF)
        return float(ratio * ratio)

    @property
    def t_lim_h(self) -> float:
        """Return the limiting duration ``t_lim`` in hours."""
        return float(self.t_lim_min) / 60.0

    @property
    def ventilation_controlled(self) -> bool:
        """Return whether ventilation, not fuel, limits the fire duration."""
        return self.t_ventilation_h > self.t_lim_h

    @property
    def t_ventilation_h(self) -> float:
        """Return the ventilation-controlled duration ``0.2e-3 q_td / O`` [h]."""
        return float(0.2e-3 * self.q_td / self.opening_factor)

    @property
    def t_max_h(self) -> float:
        """Return the limiting time ``t_max`` [h] (A.1(4))."""
        return max(self.t_ventilation_h, self.t_lim_h)

    @property
    def t_max_star_h(self) -> float:
        """Return the *scaled* peak time ``t*_max = t_max Gamma`` [h]."""
        return float(self.t_max_h * self.gamma)

    @property
    def cooling_factor_x(self) -> float:
        """Return the cooling-phase factor ``x`` (A.2).

        ``x = 1`` for a ventilation-controlled fire and ``x = t_lim / t_max``
        otherwise; since ``t_max = t_lim`` in the fuel-controlled branch, the
        second expression is also 1.  Computed rather than hard-coded so the
        branch actually taken is visible and testable.
        """
        if self.ventilation_controlled:
            return 1.0
        return float(self.t_lim_h / self.t_max_h)

    @property
    def cooling_rate(self) -> float:
        """Return the cooling rate [K/h] selected by ``t*_max`` (A.2)."""
        t_star = self.t_max_star_h
        if t_star <= 0.5:
            return 625.0
        if t_star < 2.0:
            return float(250.0 * (3.0 - t_star))
        return 250.0

    @property
    def theta_max(self) -> float:
        """Return the peak gas temperature ``theta_max`` [degC]."""
        return float(self._heating(self.t_max_star_h))

    def _heating(self, t_star: FloatOrArray) -> npt.NDArray[np.float64]:
        """Heating-phase temperature at scaled time ``t*`` [h]."""
        ts = np.asarray(t_star, dtype=float)
        out: npt.NDArray[np.float64] = np.asarray(
            self.theta_0
            + _PARAMETRIC_AMPLITUDE
            * (
                1.0
                - 0.324 * np.exp(-0.2 * ts)
                - 0.204 * np.exp(-1.7 * ts)
                - 0.472 * np.exp(-19.0 * ts)
            ),
            dtype=float,
        )
        return out

    def temperature(self, t_min: FloatOrArray) -> FloatOrArray:
        """Return the gas temperature [degC] at time ``t_min`` [minutes].

        Parameters
        ----------
        t_min : float or ndarray
            Time [minutes]; must be >= 0.

        Returns
        -------
        float or ndarray
            Gas temperature [degC], same shape as ``t_min``.

        Raises
        ------
        ValueError
            If any time is negative.
        """
        arr = np.asarray(t_min, dtype=float)
        if np.any(arr < 0.0):
            msg = (
                f"parametric_fire_temperature: time must be >= 0, got min "
                f"{float(arr.min())}"
            )
            raise ValueError(msg)
        t_star = (arr / 60.0) * self.gamma  # hours, scaled
        t_peak = self.t_max_star_h
        heating = self._heating(t_star)
        cooling = np.maximum(
            self.theta_0,
            self.theta_max
            - self.cooling_rate * (t_star - t_peak * self.cooling_factor_x),
        )
        out = np.where(t_star <= t_peak, heating, cooling)
        # continuity guard: never report below ambient on either branch
        out = np.maximum(out, self.theta_0)
        if np.ndim(t_min) == 0:
            return float(out)
        return out

    def __call__(self, t_min: FloatOrArray) -> FloatOrArray:
        """Return the gas temperature at ``t_min`` [minutes] (fire-curve protocol)."""
        return self.temperature(t_min)


def parametric_fire_temperature(
    t_min: FloatOrArray,
    opening_factor: float,
    q_td: float,
    b_value: float,
    t_lim_min: float = 20.0,
    theta_0: float = _AMBIENT_C,
) -> FloatOrArray:
    """Functional form of :class:`ParametricFire` (EN 1991-1-2 Annex A).

    Parameters
    ----------
    t_min : float or ndarray
        Time [minutes]; must be >= 0.
    opening_factor : float
        ``O = A_v sqrt(h_eq) / A_t`` [m^0.5].
    q_td : float
        Design fire load density on the total enclosure area [MJ/m^2].
    b_value : float
        Boundary thermal inertia [J/(m^2 s^0.5 K)].
    t_lim_min : float, default 20.0
        Limiting duration by fire growth rate [min] (Table A.2).
    theta_0 : float, default 20.0
        Initial gas temperature [degC].

    Returns
    -------
    float or ndarray
        Gas temperature [degC], same shape as ``t_min``.

    See Also
    --------
    ParametricFire : the same curve as a validated, self-describing object.
    """
    fire = ParametricFire(
        opening_factor=opening_factor,
        q_td=q_td,
        b_value=b_value,
        t_lim_min=t_lim_min,
        theta_0=theta_0,
    )
    return fire.temperature(t_min)


def h_net(
    theta_g: FloatOrArray,
    theta_a: FloatOrArray,
    emissivity: float = EPSILON_RES,
    alpha_c: float = ALPHA_C,
) -> FloatOrArray:
    """Net heat flux per unit area ``h_net`` [W/m^2] onto the member surface.

    ``h_net = h_c + h_r`` with the convective term ``h_c = alpha_c
    (theta_g - theta_a)`` and the radiative term ``h_r = epsilon_res * sigma
    * ((theta_g + 273)^4 - (theta_a + 273)^4)``, per EN 1991-1-2:2005
    §3.1 (net heat flux) with the absolute-temperature radiation law.

    Parameters
    ----------
    theta_g, theta_a : float or ndarray
        Gas and steel-surface temperatures [degC]; broadcast against each
        other.
    emissivity : float, default EPSILON_RES
        Resultant emissivity ``epsilon_res`` [-].
    alpha_c : float, default ALPHA_C
        Convective coefficient [W/(m^2 K)].

    Returns
    -------
    float or ndarray
        Net heat flux [W/m^2]; positive means heating the member.
    """
    tg = np.asarray(theta_g, dtype=float)
    ta = np.asarray(theta_a, dtype=float)
    convective = alpha_c * (tg - ta)
    radiative = emissivity * STEFAN_BOLTZMANN * ((tg + 273.0) ** 4 - (ta + 273.0) ** 4)
    out: npt.NDArray[np.float64] = np.asarray(convective + radiative, dtype=float)
    if np.ndim(theta_g) == 0 and np.ndim(theta_a) == 0:
        return float(out)
    return out


@dataclass(frozen=True)
class SteelHeatingResult:
    """Time history of a lumped-capacitance member-heating solve.

    Attributes
    ----------
    time_s : numpy.ndarray
        Sample times [s], shape ``(n,)``, ``time_s[-1]`` == duration.
    theta_steel : numpy.ndarray
        Steel temperature [degC] at each sample time, shape ``(n,)``.
    theta_gas : numpy.ndarray
        Gas temperature [degC] at each sample time, shape ``(n,)``.
    theta_final : float
        Convenience alias for ``theta_steel[-1]`` [degC].
    """

    time_s: npt.NDArray[np.float64]
    theta_steel: npt.NDArray[np.float64]
    theta_gas: npt.NDArray[np.float64]

    @property
    def theta_final(self) -> float:
        """Final steel temperature [degC]."""
        return float(self.theta_steel[-1])

    def max_heating_rate(self) -> float:
        """Maximum instantaneous heating rate [degC/s].

        Returns
        -------
        float
            Maximum value of d(theta)/dt over the entire time history.
            Computed via finite differences on the temperature history.

        Notes
        -----
        Addresses Issue B5: ``max_heating_rate()`` missing from
        ``SteelHeatingResult`` (C2(⚠️ب)).
        """
        if len(self.time_s) < 2:
            return 0.0
        dt = np.diff(self.time_s)
        dtheta = np.diff(self.theta_steel)
        rates = dtheta / dt
        return float(np.max(rates))

    def time_to_temperature(self, target_theta: float) -> float | None:
        """Time at which steel first reaches a target temperature.

        Parameters
        ----------
        target_theta : float
            Target steel temperature [degC].

        Returns
        -------
        float | None
            Time [s] when steel first reaches ``target_theta``, or ``None``
            if the target is never reached during the exposure.

        Notes
        -----
        Addresses Issue B5: ``time_to_temperature()`` missing from
        ``SteelHeatingResult`` (C2(⚠️ب)).
        """
        # Find first index where temperature >= target
        above = np.where(self.theta_steel >= target_theta)[0]
        if len(above) == 0:
            return None

        idx = above[0]
        if idx == 0:
            return float(self.time_s[0])

        # Linear interpolation between previous and current point
        t1, t2 = self.time_s[idx - 1], self.time_s[idx]
        th1, th2 = self.theta_steel[idx - 1], self.theta_steel[idx]

        if abs(th2 - th1) < 1e-10:
            return float(t2)

        t_target = t1 + (target_theta - th1) * (t2 - t1) / (th2 - th1)
        return float(t_target)


def lumped_capacity_biot(
    section_factor: float,
    theta_a0: float = _AMBIENT_C,
    emissivity: float = EPSILON_RES,
    alpha_c: float = ALPHA_C,
) -> float:
    """Biot number behind the uniform-temperature assumption (B7).

    .. code-block:: text

        Bi = h_eff * (V / A_m) / lambda_a

    The lumped-capacitance model in :func:`steel_temperature` is valid while
    ``Bi`` is small (the classical criterion is ``Bi < 0.1``): internal
    conduction must redistribute the surface heat input faster than it
    arrives, or the section develops a through-thickness gradient and a
    single temperature stops describing it.

    ``h_eff`` is not a guessed constant -- it is the module's own
    :func:`h_net` linearised about a representative exposure
    (:data:`_BIOT_REFERENCE_THETA_G` gas against the member's initial
    temperature), so it carries the radiative term that dominates a real
    compartment fire.  ``lambda_a`` is evaluated at the *reference gas*
    temperature rather than at ambient, because steel conductivity falls with
    temperature and the lower value gives the larger, i.e. conservative,
    Biot number.

    Parameters
    ----------
    section_factor : float
        ``A_m/V`` [1/m]; must be positive.
    theta_a0 : float, default 20.0
        Initial steel temperature [degC].
    emissivity, alpha_c : float
        Surface exchange parameters; see :func:`h_net`.

    Returns
    -------
    float
        Biot number [-].

    Raises
    ------
    ValueError
        If ``section_factor`` is not positive.
    """
    if section_factor <= 0.0:
        msg = f"section_factor (A_m/V) must be > 0, got {section_factor}"
        raise ValueError(msg)
    flux = float(h_net(_BIOT_REFERENCE_THETA_G, theta_a0, emissivity, alpha_c))
    delta_t = _BIOT_REFERENCE_THETA_G - float(theta_a0)
    h_eff = flux / delta_t if delta_t > 0.0 else float(alpha_c)
    lambda_a = float(thermal_conductivity(_BIOT_REFERENCE_THETA_G))
    if lambda_a <= 0.0:  # pragma: no cover - the code table is positive
        return float("inf")
    return float(h_eff / (float(section_factor) * lambda_a))


def biot_critical_section_factor(
    biot_limit: float = 0.1,
    theta_a0: float = _AMBIENT_C,
    emissivity: float = EPSILON_RES,
    alpha_c: float = ALPHA_C,
) -> float:
    """Section factor at which the Biot number reaches ``biot_limit``.

    Inverts :func:`lumped_capacity_biot`: ``A_m/V = h_eff / (Bi * lambda_a)``.
    Reporting the code's ``A_m/V`` threshold *and* the threshold the classical
    ``Bi < 0.1`` criterion actually implies keeps the warning honest -- for
    the default exposure the two are ~34 1/m and 50 1/m, i.e. the code
    threshold fires first and is the conservative one.  A warning that
    quoted a Biot number passing its own stated criterion would be
    self-contradicting.

    Parameters
    ----------
    biot_limit : float, default 0.1
        Biot number defining the validity limit of a lumped body.
    theta_a0 : float, default 20.0
        Initial steel temperature [degC].
    emissivity, alpha_c : float
        Surface exchange parameters; see :func:`h_net`.

    Returns
    -------
    float
        Critical section factor ``A_m/V`` [1/m].

    Raises
    ------
    ValueError
        If ``biot_limit`` is not positive.
    """
    if biot_limit <= 0.0:
        msg = f"biot_limit must be > 0, got {biot_limit}"
        raise ValueError(msg)
    return lumped_capacity_biot(1.0, theta_a0, emissivity, alpha_c) / float(biot_limit)


def steel_temperature(
    duration_min: float,
    section_factor: float,
    fire_curve: Callable[[FloatOrArray], FloatOrArray] = iso_834_temperature,
    theta_a0: float = _AMBIENT_C,
    shadow_factor: float = 1.0,
    emissivity: float = EPSILON_RES,
    alpha_c: float = ALPHA_C,
    rho_a: float | None = None,
    max_step_s: float = 5.0,
    n_output: int = 0,
    warn_lumped: bool = True,
) -> SteelHeatingResult:
    """Steel temperature history of an unprotected member (lumped capacity).

    Integrates EN 1993-1-2:2005 §4.2.2.2 eq. (4.9)-(4.11):

    .. code-block:: text

        d(theta_a)/dt = k_sh * h_net(theta_g, theta_a) * (A_m/V) / (rho_a * c_a)

    with ``c_a = c_a(theta_a)`` the standard's temperature-dependent specific
    heat and ``h_net`` from :func:`h_net`. ``fire_curve`` supplies
    ``theta_g(t)`` and is evaluated in **minutes** (the ISO 834 convention);
    the ODE is integrated in seconds.

    Parameters
    ----------
    duration_min : float
        Fire exposure duration [minutes]; must be > 0.
    section_factor : float
        Section (heated-perimeter-to-volume) factor ``A_m/V`` [1/m]; must be
        positive. Larger values heat faster (thin, exposed sections).
    fire_curve : Callable, default :func:`iso_834_temperature`
        Maps time [minutes] to gas temperature [degC]; vectorised.
    theta_a0 : float, default 20.0
        Initial steel temperature [degC].
    shadow_factor : float, default 1.0
        Shadow factor ``k_sh`` [-] in (0, 1]; ``1.0`` is conservative
        (no shadowing) and the EN default for uncovered members.
    emissivity, alpha_c : float
        Radiation/ convection parameters; see :func:`h_net`.
    rho_a : float or None
        Unit mass [kg/m^3]; ``None`` takes the standard's
        :func:`~truss_analysis.material.steel_eurocode.unit_mass`.
    max_step_s : float, default 5.0
        Maximum integrator step [s]; EN 1993-1-2 caps ``Δt`` at 5 s.
    n_output : int, default 0
        Number of evenly spaced output samples (including t=0 and t=end).
        ``0`` keeps every integrator step.
    warn_lumped : bool, default True
        Issue a
        :class:`~truss_analysis.exceptions.LumpedCapacityWarning` when
        ``section_factor`` is below
        :data:`LUMPED_SECTION_FACTOR_LIMIT`.  Set to ``False`` inside a sweep
        that would otherwise emit the same warning once per member.

    Returns
    -------
    SteelHeatingResult
        Time, steel-temperature and gas-temperature histories.

    Raises
    ------
    ValueError
        If ``duration_min`` <= 0, ``section_factor`` <= 0, ``shadow_factor``
        is outside (0, 1], or ``max_step_s`` <= 0.

    Warns
    -----
    LumpedCapacityWarning
        If ``section_factor < LUMPED_SECTION_FACTOR_LIMIT``: the member is
        thick enough to develop a real through-thickness gradient, so the
        uniform-temperature answer is un-conservative.  The message carries
        the Biot number from :func:`lumped_capacity_biot`.
    """
    if duration_min <= 0.0:
        msg = f"duration_min must be > 0, got {duration_min}"
        raise ValueError(msg)
    if section_factor <= 0.0:
        msg = f"section_factor (A_m/V) must be > 0, got {section_factor}"
        raise ValueError(msg)
    if not 0.0 < shadow_factor <= 1.0:
        msg = f"shadow_factor k_sh must lie in (0, 1], got {shadow_factor}"
        raise ValueError(msg)
    if max_step_s <= 0.0:
        msg = f"max_step_s must be > 0, got {max_step_s}"
        raise ValueError(msg)

    rho = float(unit_mass()) if rho_a is None else float(rho_a)
    if rho <= 0.0:
        msg = f"rho_a must be > 0, got {rho}"
        raise ValueError(msg)

    # B7: state the limit of the lumped assumption instead of assuming it.
    if warn_lumped and section_factor < LUMPED_SECTION_FACTOR_LIMIT:
        biot = lumped_capacity_biot(section_factor, theta_a0, emissivity, alpha_c)
        warnings.warn(
            f"LumpedCapacityWarning: section factor A_m/V = "
            f"{section_factor:.4g} 1/m is below "
            f"{LUMPED_SECTION_FACTOR_LIMIT:g} 1/m (thick section); the "
            f"Biot number is {biot:.3f} against the usual Bi < 0.1 criterion "
            "for a lumped body. The cross-section will develop a real "
            "through-thickness gradient, its core will lag its surface, and "
            "the single uniform temperature returned here is "
            "un-conservative. A heat-conduction solution is out of scope "
            "for this library.",
            LumpedCapacityWarning,
            stacklevel=2,
        )

    duration_s = float(duration_min) * 60.0
    n_steps = max(1, int(np.ceil(duration_s / max_step_s)))
    dt = duration_s / n_steps

    def theta_g_at(t_s: float) -> float:
        return float(fire_curve(t_s / 60.0))

    def dtheta_dt(t_s: float, theta_a: float) -> float:
        flux = float(h_net(theta_g_at(t_s), theta_a, emissivity, alpha_c))
        c_a = float(specific_heat(theta_a))
        return shadow_factor * flux * section_factor / (rho * c_a)

    # Classic RK4 over a fixed grid of n_steps. The specific-heat spike near
    # 730 degC is smooth enough for RK4 at <=5 s steps to track it to well
    # under a degree (verified by step-halving in the tests).
    ts = np.linspace(0.0, duration_s, n_steps + 1)
    thetas = np.empty(n_steps + 1, dtype=float)
    thetas[0] = float(theta_a0)
    for i in range(n_steps):
        t0 = ts[i]
        y = thetas[i]
        k1 = dtheta_dt(t0, y)
        k2 = dtheta_dt(t0 + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = dtheta_dt(t0 + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = dtheta_dt(t0 + dt, y + dt * k3)
        thetas[i + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    theta_gas = np.asarray(fire_curve(ts / 60.0), dtype=float)
    if n_output > 0:
        idx = (
            np.linspace(0, n_steps, min(int(n_output), n_steps + 1)).round().astype(int)
        )
        ts, thetas, theta_gas = ts[idx], thetas[idx], theta_gas[idx]

    return SteelHeatingResult(time_s=ts, theta_steel=thetas, theta_gas=theta_gas)
