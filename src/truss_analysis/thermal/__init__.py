"""Thermal layer: fire exposure curves, member heating, degradation shims.

* :mod:`truss_analysis.thermal.fire_curve` -- the ISO 834 nominal curve, the
  EN 1991-1-2 Annex A parametric curve (round-6 audit B2) and the
  EN 1993-1-2 §4.2.2.2 lumped-capacitance steel-heating solver (round-5
  audit: the bridge from a fire exposure to the prescribed-temperature
  structural chain).
* :mod:`truss_analysis.thermal.protection` -- the EN 1993-1-2 §4.2.5.2
  insulated-member heating model plus a provenance-carrying catalogue of
  protection materials (round-6 audit B1: the unprotected solver was the
  only heating path, so a protected design had to be hand-computed).
* :mod:`truss_analysis.thermal.material` -- deprecated compatibility shims
  over :mod:`truss_analysis.material.steel_eurocode`.
"""

from .fire_curve import (
    ALPHA_C,
    B_VALUE_LIMITS,
    EPSILON_RES,
    LUMPED_SECTION_FACTOR_LIMIT,
    OPENING_FACTOR_LIMITS,
    Q_TD_RANGE_OF_VALIDITY,
    STEFAN_BOLTZMANN,
    T_LIM_FIRE_GROWTH_MIN,
    ParametricFire,
    SteelHeatingResult,
    biot_critical_section_factor,
    h_net,
    iso_834_temperature,
    lumped_capacity_biot,
    parametric_fire_temperature,
    steel_temperature,
)
from .material import get_eurocode_k_E, get_eurocode_k_y
from .protection import (
    MAX_STEP_S_PROTECTED,
    InsulationMaterial,
    box_protection_shadow_factor,
    capacity_ratio_mu,
    insulation_catalogue,
    insulation_material,
    protected_steel_temperature,
    protected_temperatures_for_members,
)

__all__ = [
    "ALPHA_C",
    "B_VALUE_LIMITS",
    "EPSILON_RES",
    "LUMPED_SECTION_FACTOR_LIMIT",
    "MAX_STEP_S_PROTECTED",
    "OPENING_FACTOR_LIMITS",
    "Q_TD_RANGE_OF_VALIDITY",
    "STEFAN_BOLTZMANN",
    "T_LIM_FIRE_GROWTH_MIN",
    "InsulationMaterial",
    "ParametricFire",
    "SteelHeatingResult",
    "biot_critical_section_factor",
    "box_protection_shadow_factor",
    "capacity_ratio_mu",
    "get_eurocode_k_E",
    "get_eurocode_k_y",
    "h_net",
    "insulation_catalogue",
    "insulation_material",
    "iso_834_temperature",
    "lumped_capacity_biot",
    "parametric_fire_temperature",
    "protected_steel_temperature",
    "protected_temperatures_for_members",
    "steel_temperature",
]
