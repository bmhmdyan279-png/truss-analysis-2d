"""Level 2 — thermal reduction curves: exact fixture equality + independence.

Three lines of defence, in increasing order of independence:

1. **Exact equality with the packaged fixture.**  At every tabulated
   temperature the API must return *exactly* (float ``==``, not approx) the
   fixture value — the acceptance criterion of the validation protocol is
   tightened from "absolute error < 0.01" to "zero error" because the
   fixture IS the tabulated source; any nonzero deviation is a bug.
2. **Independent transcription.**  The table below was transcribed by hand
   from the published standard (EN 1993-1-2:2005, Table 3.1, clause
   3.2.1(3)) — NOT copied programmatically from the fixture JSON — so a
   corrupted or mutated fixture cannot pass.  The transcription agrees with
   two independent published reproductions of the table; the live external
   cross-check (fetched source, hash and 39/39 value comparison) is part of
   the validation evidence produced by the runner scripts.
3. **Structural properties** of the curves: linear-interpolation midpoint
   consistency, out-of-range clamping (never extrapolation, never a silent
   zero), scalar/vector agreement, and the documented ``k_s == k_E`` alias
   identity (the standard tabulates exactly ONE elastic reduction factor).
"""

from __future__ import annotations

import numpy as np
import pytest
from truss_analysis.material import steel_eurocode as ec

# Hand transcription of EN 1993-1-2:2005 Table 3.1 (carbon steel).
# Columns: k_y,theta = f_y,theta/f_y; k_p,theta = f_p,theta/f_y;
#          k_E,theta = E_a,theta/E_a (slope of the linear elastic range).
TEMPS = [20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
K_Y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.78, 0.47, 0.23, 0.11, 0.06, 0.04, 0.02, 0.0]
K_P = [
    1.0,
    1.0,
    0.807,
    0.613,
    0.42,
    0.36,
    0.18,
    0.075,
    0.05,
    0.0375,
    0.025,
    0.0125,
    0.0,
]
K_E = [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.31, 0.13, 0.09, 0.0675, 0.045, 0.0225, 0.0]

# A widely reproduced subset (fire-engineering literature) used as a second
# offline transcription; note it does NOT include k_p.
EXTERNAL_SUBSET_100_900 = {
    # theta: (k_y, k_E)
    100: (1.00, 1.00),
    200: (1.00, 0.90),
    300: (1.00, 0.80),
    400: (1.00, 0.70),
    500: (0.78, 0.60),
    600: (0.47, 0.31),
    700: (0.23, 0.13),
    800: (0.11, 0.09),
    900: (0.06, 0.0675),
}

COLUMNS = {"k_y": (ec.k_y, K_Y), "k_p": (ec.k_p, K_P), "k_E": (ec.k_E, K_E)}


@pytest.mark.parametrize("column", sorted(COLUMNS))
def test_api_returns_fixture_values_exactly(column):
    func, values = COLUMNS[column]
    for theta, value in zip(TEMPS, values):
        assert func(float(theta)) == value  # EXACT float equality (zero error)


@pytest.mark.parametrize("column", sorted(COLUMNS))
def test_fixture_json_matches_independent_transcription(column):
    table = ec.table()
    assert [float(t) for t in table["temperatures_c"]] == [float(t) for t in TEMPS]
    assert table[column] == COLUMNS[column][1]


def test_external_subset_agrees():
    for theta, (k_y_val, k_e_val) in EXTERNAL_SUBSET_100_900.items():
        assert ec.k_y(float(theta)) == k_y_val
        assert ec.k_E(float(theta)) == k_e_val


def test_k_s_is_exact_alias_of_k_E():
    for theta in (20.0, 350.0, 600.0, 999.0, 1200.0):
        assert ec.k_s(theta) == ec.k_E(theta)
    grid = np.linspace(20.0, 1200.0, 119)
    assert np.array_equal(ec.k_s(grid), ec.k_E(grid))


def test_linear_interpolation_midpoint_consistency():
    """Midpoint of two table rows == mean of the two values (<= 1e-15)."""
    for column, (_func, values) in COLUMNS.items():
        func = COLUMNS[column][0]
        for i in range(len(TEMPS) - 1):
            mid = (TEMPS[i] + TEMPS[i + 1]) / 2.0
            want = (values[i] + values[i + 1]) / 2.0
            assert abs(float(func(mid)) - want) <= 1e-15


def test_out_of_range_is_clamped_never_extrapolated():
    for column, (_f, values) in COLUMNS.items():
        func = COLUMNS[column][0]
        assert func(-100.0) == values[0]
        assert func(1500.0) == values[-1]


def test_scalar_and_vector_paths_agree_exactly():
    grid = np.asarray(TEMPS, dtype=float)
    for column, (_func, values) in COLUMNS.items():
        func = COLUMNS[column][0]
        assert np.array_equal(func(grid), np.asarray(values, dtype=float))
        assert func(grid).tolist() == [func(float(t)) for t in TEMPS]


def test_strain_limits_are_figure_constants():
    for theta in (20.0, 550.0, 1200.0):
        assert ec.eps_y(theta) == 0.02
        assert ec.eps_t(theta) == 0.15
        assert ec.eps_u(theta) == 0.20
