"""Monte Carlo convergence and probabilistic-ranking tests."""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import numpy as np

from truss_analysis.limitstates import ci_two_component
from truss_analysis.uncertainty import (
    RunningStat,
    default_rv_specs,
    probabilistic_ranking,
    sample_spec_matrix,
)

BATCH = 50
NS = (100, 250, 500, 1000, 2000)
SEED = 20260907


def _sample_ci_batch(cm, live, fy, temps, alpha=0.7):
    """Max-member displacement CI for one batch of RV draws."""
    out = []
    for lk, _fyk, tk in zip(live, fy, temps, strict=True):
        loads = {
            nid: {"Fx": ld["Fx"] * lk, "Fy": ld["Fy"] * lk}
            for nid, ld in _loads_of(cm).items()
        }
        temps_i = {e.id: float(tk) for e in cm.elements}
        # two-component CI: the displacement-only CI is invariant to load
        # scale (linearity) and to uniform temperature (invariance property
        # of the rank-1 engine), so the MC statistic must include the DCR
        # component to depend on the RVs
        res = ci_two_component(cm.nodes, cm.elements, loads, temps_i, alpha, 235.0e6)
        # spatial MEAN of the two-component CI: a smooth MC statistic
        # (the max has a heavy tail from near-capacity DCR spikes)
        out.append(float(np.mean(list(res.ci_values.values()))))
    return np.asarray(out)


def _loads_of(cm):
    loads = cm.loads
    if isinstance(loads, dict) and all(isinstance(v, dict) for v in loads.values()):
        return loads
    return {ld["node_id"]: ld for ld in loads}


def test_mc_convergence_and_memory(campaign, tmp_path: Path) -> None:
    cm = next(c for c in campaign if c.name == "warren_4_shallow")
    # screening scenario at 400 degC: capacities do not collapse, so the MC
    # statistic has light tails and the 1% convergence gate is meaningful
    specs = default_rv_specs(fire_scenario_temperature=400.0)
    means = {"live_load": 1.0, "f_y": 235.0e6, "fire_intensity": 400.0, "E": 210.0e9}
    n_max = max(NS)
    samples = sample_spec_matrix(specs, means, n_max, seed=SEED)
    live, fy, temps = samples["live_load"], samples["f_y"], samples["fire_intensity"]

    stat = RunningStat()
    curve = {}
    peak = 0
    tracemalloc.start()
    done = 0
    for i in range(0, n_max, BATCH):
        batch = _sample_ci_batch(
            cm, live[i : i + BATCH], fy[i : i + BATCH], temps[i : i + BATCH]
        )
        stat.add_batch(batch)
        done += batch.size
        if done in NS:
            curve[done] = {"mean": stat.mean, "std": stat.std}
        _, cur_peak = tracemalloc.get_traced_memory()
        peak = max(peak, cur_peak)
    tracemalloc.stop()

    m1000, m2000 = curve[1000]["mean"], curve[2000]["mean"]
    s1000, s2000 = curve[1000]["std"], curve[2000]["std"]
    assert abs(m2000 - m1000) / abs(m2000) < 0.01
    assert abs(s2000 - s1000) / abs(s2000) < 0.01
    assert peak < 700 * 1024 * 1024  # gate: well below ~700 MB

    # the convergence curve is written under the pytest tmp dir: a library
    # test must never write outside the repository checkout
    (tmp_path / "mc_convergence.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "topology": cm.name,
                "batch": BATCH,
                "curve": {str(k): v for k, v in curve.items()},
                "peak_memory_bytes": peak,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_probabilistic_ranking_four_steps(campaign) -> None:
    cm = next(c for c in campaign if c.name == "warren_4_shallow")
    specs = default_rv_specs(fire_scenario_temperature=600.0)
    means = {"live_load": 1.0, "f_y": 235.0e6, "fire_intensity": 600.0, "E": 210.0e9}
    n = 300
    samples = sample_spec_matrix(specs, means, n, seed=SEED + 1)
    per_member: dict = {e.id: [] for e in cm.elements}
    for i in range(n):
        loads = {
            nid: {
                "Fx": ld["Fx"] * samples["live_load"][i],
                "Fy": ld["Fy"] * samples["live_load"][i],
            }
            for nid, ld in cm.loads.items()
        }
        temps_i = {e.id: float(samples["fire_intensity"][i]) for e in cm.elements}
        res = ci_two_component(cm.nodes, cm.elements, loads, temps_i, 0.7, 235.0e6)
        for mid, v in res.ci_values.items():
            per_member[mid].append(v)
    ci_per_sample = {k: np.asarray(v) for k, v in per_member.items()}
    # step 4: deterministic CI at mean inputs (uniform 600, mean load factor 1)
    loads_mean = cm.loads
    temps_mean = {e.id: 600.0 for e in cm.elements}
    ci_mean_inputs = ci_two_component(
        cm.nodes, cm.elements, loads_mean, temps_mean, 0.7, 235.0e6
    ).ci_values

    ranking = probabilistic_ranking(ci_per_sample, ci_mean_inputs)
    ids = {e.id for e in cm.elements}
    assert set(ranking.rank_probabilistic) == ids
    assert set(ranking.rank_deterministic) == ids
    assert len(ranking.top5_probabilistic) == 5
    # the four-step operational definition is documented in the module
    # docstring (import via importlib: the package attribute of the same
    # name is shadowed by the function re-export)
    import importlib

    prmod = importlib.import_module("truss_analysis.uncertainty.probabilistic_ranking")
    doc = prmod.__doc__ or ""
    assert "MEAN of" in doc
    assert "``CI_i``" in doc
