"""EN 1993-1-2:2005 material layer - the single source of truth.

Public API re-exported from :mod:`truss_analysis.material.steel_eurocode`.
All values originate from the provenance fixture
``data/en1993_1_2_table3_1.json``; see that module's docstring.
"""

from __future__ import annotations

from .steel_eurocode import (
    alpha,
    eps_p,
    eps_t,
    eps_u,
    eps_y,
    k_E,
    k_p,
    k_s,
    k_y,
    poisson_ratio,
    specific_heat,
    stress_strain,
    table,
    table_temperatures,
    thermal_conductivity,
    unit_mass,
)

__all__ = [
    "alpha",
    "eps_p",
    "eps_t",
    "eps_u",
    "eps_y",
    "k_E",
    "k_p",
    "k_s",
    "k_y",
    "poisson_ratio",
    "specific_heat",
    "stress_strain",
    "table",
    "table_temperatures",
    "thermal_conductivity",
    "unit_mass",
]
