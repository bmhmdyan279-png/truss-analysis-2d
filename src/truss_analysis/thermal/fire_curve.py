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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..material.steel_eurocode import (
    FloatOrArray,
    specific_heat,
    unit_mass,
)

__all__ = [
    "ALPHA_C",
    "EPSILON_RES",
    "STEFAN_BOLTZMANN",
    "SteelHeatingResult",
    "h_net",
    "iso_834_temperature",
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

    Returns
    -------
    SteelHeatingResult
        Time, steel-temperature and gas-temperature histories.

    Raises
    ------
    ValueError
        If ``duration_min`` <= 0, ``section_factor`` <= 0, ``shadow_factor``
        is outside (0, 1], or ``max_step_s`` <= 0.
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
