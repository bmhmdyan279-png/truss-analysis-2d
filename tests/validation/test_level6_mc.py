"""Level 6 — Monte-Carlo convergence ladder 100 -> 250 -> 500 -> 1000 -> 2000.

Estimand (fixed before measurement): the spatial MEAN of the combined
triage metric per sample, ``max(u_component, dcr_combined)`` — the
displacement-only CI is invariant to load scale (linearity) and to uniform
temperature (uniform-field invariance), and since 2.7 the pure damage CI is
deliberately free of the fire-severity term, so the statistic that responds
to the random variables combines the damage and fire components explicitly
(numerically the pre-2.7 composite; see round-4 audit, critic 3 P0).
The ladder uses a single deterministic LHS draw of 2000 samples
(seed 20260907) truncated at each rung: rungs are nested prefixes of the
same sample set, so differences between rungs are pure sample-size effects.

Acceptance gate (protocol): the relative change of BOTH the mean and the
standard deviation of the CI statistic between N=1000 and N=2000 is < 1 %.

Reported alongside the gate (evidence, not gated at 1 %):
* the u_max estimand (mean and std) — its std estimate has a sampling
  relative error of order 1/sqrt(2N) ~ 1.6 % at N=1000, so a sub-2 % rung
  difference in that std is sampling noise, not non-convergence;
* rank stability between N=1000 and N=2000: Spearman rho of the per-member
  MEAN-CI rankings (must be ~1 — convergence of the ranking, not only of
  the scalar statistic).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from truss_analysis.limitstates import ci_two_component
from truss_analysis.uncertainty import (
    RunningStat,
    default_rv_specs,
    sample_spec_matrix,
)

SEED = 20260907
LADDER = (100, 250, 500, 1000, 2000)
GATE = 0.01
F_Y = 235.0e6
ALPHA = 0.7
FIRE_TEMPERATURE = 400.0  # screening temperature: light-tailed statistic


def _loads_of(cm, factor):
    return {
        nid: {"Fx": ld["Fx"] * factor, "Fy": ld["Fy"] * factor}
        for nid, ld in cm.loads.items()
    }


def test_mc_convergence_ladder_gates_and_rank_stability(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_4_shallow")
    specs = default_rv_specs(fire_scenario_temperature=FIRE_TEMPERATURE)
    means = {
        "live_load": 1.0,
        "f_y": F_Y,
        "fire_intensity": FIRE_TEMPERATURE,
        "E": 210.0e9,
    }
    n_max = max(LADDER)
    samples = sample_spec_matrix(specs, means, n_max, seed=SEED)
    live, temps = samples["live_load"], samples["fire_intensity"]

    ids = [e.id for e in cm.elements]
    per_member = {mid: np.empty(n_max) for mid in ids}
    ci_stat = RunningStat()
    u_stat = RunningStat()
    ci_raw = np.empty(n_max)
    u_raw = np.empty(n_max)
    curve = {}
    for k in range(n_max):
        loads = _loads_of(cm, float(live[k]))
        temps_i = {e.id: float(temps[k]) for e in cm.elements}
        res = ci_two_component(cm.nodes, cm.elements, loads, temps_i, ALPHA, F_Y)
        # combined triage metric per member, max(u_component, dcr_combined):
        # since 2.7 the pure damage ci_values are invariant to load scale and
        # (uniform field, no imposed strain) to temperature, so the ladder's
        # fire-responsive estimand is built EXPLICITLY from the combined
        # ratio -- numerically the pre-2.7 composite, semantically two
        # separated effects (round-4 audit, critic 3 P0)
        for mid in ids:
            comp = res.components[mid]
            per_member[mid][k] = max(comp.u_component, comp.dcr_combined)
        s = float(np.mean([per_member[mid][k] for mid in ids]))
        ci_stat.add(s)
        u_stat.add(res.u_max_base)
        ci_raw[k] = s
        u_raw[k] = res.u_max_base
        if (k + 1) in LADDER:
            curve[k + 1] = {
                "ci_mean": ci_stat.mean,
                "ci_std": ci_stat.std,
                "u_mean": u_stat.mean,
                "u_std": u_stat.std,
            }
    assert sorted(curve) == list(LADDER)
    assert ci_stat.count == n_max

    c1000, c2000 = curve[1000], curve[2000]
    rel_d_mean = abs(c2000["ci_mean"] - c1000["ci_mean"]) / abs(c2000["ci_mean"])
    rel_d_std = abs(c2000["ci_std"] - c1000["ci_std"]) / abs(c2000["ci_std"])
    assert rel_d_mean < GATE, rel_d_mean
    assert rel_d_std < GATE, rel_d_std

    # secondary estimand: u_max mean converges under the same gate
    rel_u_mean = abs(c2000["u_mean"] - c1000["u_mean"]) / abs(c2000["u_mean"])
    assert rel_u_mean < GATE, rel_u_mean
    # u_max std is REPORTED with its sampling-noise context (see docstring):
    rel_u_std = abs(c2000["u_std"] - c1000["u_std"]) / abs(c2000["u_std"])
    assert rel_u_std < 2.5 * GATE, rel_u_std

    # rank stability of the per-member mean-CI ranking, 1000 vs 2000
    mean_ci_1000 = [float(np.mean(per_member[mid][:1000])) for mid in ids]
    mean_ci_2000 = [float(np.mean(per_member[mid])) for mid in ids]
    rho_rank = float(spearmanr(mean_ci_1000, mean_ci_2000).statistic)
    assert rho_rank > 0.99, rho_rank

    # the curve is monotone-consistent: rung statistics are finite, positive
    for rung in LADDER:
        assert curve[rung]["ci_mean"] > 0.0
        assert curve[rung]["ci_std"] > 0.0
        assert np.isfinite(u_raw[:rung]).all()
        assert np.isfinite(ci_raw[:rung]).all()
    assert pytest.approx(ci_stat.mean, rel=1e-12) == float(np.mean(ci_raw))
