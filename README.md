# Truss Analysis 2D

**Read this in:** [English](README.md) | [فارسی](README.fa.md)

A Python library for **linear static analysis of planar (2D) trusses** with the
direct stiffness method — extended with temperature-dependent steel properties
per EN 1993-1-2, parametric topology generation, member criticality indices,
uncertainty quantification and retrofit triage utilities.

[![CI](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml/badge.svg)](https://github.com/bmhmdyan279-png/truss-analysis-2d/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-94.4%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![PyPI version](https://img.shields.io/pypi/v/truss-analysis.svg)](https://pypi.org/project/truss-analysis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line interface](#command-line-interface)
- [Python API](#python-api)
- [Input JSON schema](#input-json-schema)
- [Output JSON schema](#output-json-schema)
- [Built-in checks](#built-in-checks)
- [Testing and development](#testing-and-development)
- [Project layout](#project-layout)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Overview

`truss_analysis` solves 2D truss models (nodes + axial bar elements) in linear
statics: nodal displacements, member axial forces, support reactions,
equilibrium residuals and strain-energy balance. On top of the classical
solver it provides an engineering toolkit for steel trusses exposed to
elevated temperatures:

* material reduction factors and the constitutive law of **EN 1993-1-2**
  (Table 3.1), plus the **ISO 834 exposure curve** and the code's
  lumped-capacitance **member-heating solver** (fire → steel temperature →
  structural response in one chain),
* **fire limit states** (DCR fields, critical temperatures labelled by
  failure mode) and **linearised system stability** (geometric stiffness,
  bifurcation load factors around the prestressed base state),
* **parametric generators** for Warren, Pratt and Howe trusses,
* a **member criticality index** computed with an exact rank-1 perturbation
  engine (one factorisation serves all members),
* **uncertainty quantification** (Latin hypercube and Monte Carlo sampling
  with stratification-preserving Iman–Conover rank correlation) and a
  Monte Carlo reliability engine sharing the DCR chain's capacity model,
* **retrofit triage** under a budget (greedy, exhaustive, robust and
  redundancy-aware strategies with three cost scenarios),
* structural **graph validation** (connectivity, mechanisms, rank and
  condition number of the free-free stiffness matrix).

Everything is deterministic given the inputs, fully type-annotated
(`mypy --strict` clean) and covered by a test suite of 1066 tests
(94.4 % coverage, gate at 90 %). Results are always SI (m, N, Pa) regardless
of the input unit system.

## Features

| Area | What you get |
|---|---|
| Solver | Direct stiffness method; elimination (default) or penalty boundary conditions; sparse (CSR + SuperLU) or dense assembly; Cholesky-first factorisation for the SPD system; mechanism/singularity detection with rank, condition number and a `stable`/`ill_conditioned`/`singular` status |
| Thermal | Per-member temperature change `delta_T`, free-length changes `delta_L_free` (fabrication fit); prestress work tracked separately |
| Fire engineering | EN 1993-1-2 reduction factors `k_E(T)`, `k_y(T)`, `k_s(T)`, `k_p(T)`, strain limits and the full stress–strain law; compression capacity per §4.2.3.1 with the buckling reduction factor `chi` (not bare Euler); design resistance and demand–capacity ratio (DCR) fields; system critical temperature `theta_sys` by DCR sweep, labelled by **failure mode** (material limit vs. stiffness collapse); ISO 834 exposure curve and the §4.2.2.2 **lumped-capacitance member-heating solver** (exposure → steel temperature → structural chain); secant thermal strain `effective_alpha(T)` from the standard's elongation curve |
| Criticality | Exact rank-1 (Sherman–Morrison) perturbation engine: per-member CI / normalised CI, ranks, top-5 sets, Kendall tau-b rank stability against the ambient-temperature baseline; **multi-criteria indices** — displacement, peak member force, the member's own force, strain energy and support reaction — from the same perturbed field |
| Uncertainty | Random-variable table (Gumbel live load, lognormal yield strength, truncated-normal fire intensity, deterministic E) with documented citation status; LHS + Monte Carlo; rank correlation by **Iman–Conover reordering** (Latin-hypercube stratification survives coupling) with a Gaussian-copula alternative; streaming statistics; probabilistic ranking; reliability margins on the **same `chi` capacity model as the DCR chain**; empirical failure probability with an exact Clopper–Pearson interval alongside the normal-approximation `Φ(−β̂)` |
| Retrofit | Budgeted member upgrading; greedy / exhaustive / robust / redundant strategies; linear, quadratic and step cost scenarios; metric set: cost, `u_max`, `theta_sys`, members with DCR ≥ 1 |
| Validation | Graph checks (orphans, duplicates, self-loops, connectivity, rank, condition); energy validation via the generalized Clapeyron theorem; equilibrium residuals; optional OpenSeesPy reference bridge (`validation` extra) |
| Buckling | Euler critical load with effective-length factor; utilisation ratio per compressed member |
| Stability | Geometric stiffness `K_G = Σ (N/L) g gᵀ` and **linearised bifurcation load factor** `λ_cr` around the prestressed (thermal/fabrication) base state, with mode shape — the system-level companion to the member-level checks; validated against the shallow-toggle closed form and an independent QZ eigensolve |
| Units | SI and Imperial inputs (automatic conversion); outputs always SI |
| Output | Console summary, JSON (with a `solver_metadata` reproducibility block: versions, factorisation used, rank/condition of `K_ff`, tolerance policy), CSV force table, Markdown report, PNG plot (Persian-aware text shaping with the `viz` extra) |
| Packaging | `py.typed` marker, strict typing, pinned tool chain, multi-stage Dockerfile, matrix CI (3 OS × 3 Python versions) |

## Installation

### From PyPI

```bash
pip install truss-analysis
```

The distribution name is `truss-analysis`; the import name is `truss_analysis`.

Optional extras:

```bash
pip install "truss-analysis[viz]"         # matplotlib plotting + Persian/bidi text shaping
pip install "truss-analysis[validation]"  # OpenSeesPy reference bridge
pip install "truss-analysis[dev]"         # tests, linting, typing, packaging tools
```

### From source

```bash
git clone https://github.com/bmhmdyan279-png/truss-analysis-2d.git
cd truss-analysis-2d
pip install -e ".[dev]"
pre-commit install
```

### With Docker

```bash
docker build -t truss-analysis .

# default command analyses the bundled example end to end (smoke test)
docker run --rm truss-analysis

# analyse your own model
docker run --rm -v "$PWD:/data" truss-analysis \
    truss-analysis analyze /data/my_model.json
```

The image is multi-stage with a digest-pinned `python:3.11-slim` base and a
headless matplotlib backend.

## Quick start

A complete run on the bundled example `examples/example1.json` (5 nodes,
7 members, two loaded top nodes — excerpt of the input):

```json
{
  "units": "SI",
  "nodes": [
    {"id": 1, "x": 0.0, "y": 0.0, "is_support": true, "support_dx": true, "support_dy": true},
    {"id": 2, "x": 4.0, "y": 0.0, "is_support": false},
    "... 3 more nodes ..."
  ],
  "elements": [
    {"id": 1, "node_i": 1, "node_j": 2, "A": 0.0561, "E": 210e9, "alpha": 1.2e-5},
    "... 6 more members ..."
  ],
  "loads": [
    {"node_id": 4, "Fx": 20000.0, "Fy": -30000.0},
    {"node_id": 5, "Fx": -20000.0, "Fy": -30000.0}
  ]
}
```

Analyse it, including the buckling check:

```console
$ truss-analysis analyze examples/example1.json --check-buckling
==============================================================
 TRUSS ANALYSIS RESULT
==============================================================
Element 1: N =         -0.000  [Zero]
Element 2: N =          0.000  [Zero]
Element 3: N =     -36055.513  [Compression]
Element 4: N =          0.000  [Zero]
Element 5: N =         -0.000  [Zero]
Element 6: N =     -36055.513  [Compression]
Element 7: N =     -40000.000  [Compression]
Reaction @ 1: Fx =    20000.000  Fy =    30000.000
Reaction @ 3: Fx =   -20000.000  Fy =    30000.000
Equilibrium: dFx=3.64e-12 dFy=1.46e-11 dM=2.91e-11 -> OK
Buckling 1: ratio=0.000 -> OK
Buckling 3: ratio=2.164 -> BUCKLING RISK
Buckling 5: ratio=0.000 -> OK
Buckling 6: ratio=1.219 -> BUCKLING RISK
Buckling 7: ratio=1.219 -> BUCKLING RISK
Status: SUCCESS
```

Export every artefact in one call:

```bash
truss-analysis analyze examples/example1.json --check-buckling \
    -o result.json --csv forces.csv --report report.md --plot-path diagram.png
```

`forces.csv` (verbatim):

```csv
element_id,axial_force,status
1,-2.771055810051527e-12,Zero
2,4.935565360740278e-13,Zero
3,-36055.512754639894,Compression
4,4.9950858176817626e-12,Zero
5,-3.637805361141061e-12,Zero
6,-36055.51275463989,Compression
7,-40000.00000000001,Compression
```

`report.md` contains the force/reaction tables and the equilibrium residuals;
`result.json` follows the [output schema](#output-json-schema).

Deformed vs. original shape produced with `--plot-path`
(dashed = deformed, colour = tension/compression):

![Truss analysis result](docs/images/example_output.png)

## Command-line interface

```console
$ truss-analysis --help
usage: truss-analysis [-h] {analyze,validate,generate,version} ...

2D truss analysis: linear statics, thermal loads, buckling checks, input
validation and parametric model generation.

positional arguments:
  {analyze,validate,generate,version}
    analyze             run the full analysis on one input JSON file
    validate            validate an input JSON file without solving it
    generate            generate a parametric truss model as canonical JSON
    version             print the package version

options:
  -h, --help            show this help message and exit
```

Exit codes: `0` success · `1` analysis/validation failure (details on stderr
or in the validate report) · `2` usage error or unreadable input file.
A bare input path is accepted for backward compatibility:
`truss-analysis model.json` ≡ `truss-analysis analyze model.json`.

### `analyze`

`truss-analysis analyze INPUT [options]`

| Option | Effect |
|---|---|
| `--units {SI,Imperial}` | fallback unit system when the file declares none (default `SI`) |
| `-o, --output PATH` | write the JSON result |
| `--csv PATH` | write a CSV force table |
| `--report PATH` | write a Markdown report |
| `--plot` | show an interactive plot (`viz` extra) |
| `--plot-path PATH` | save the plot as PNG |
| `--check-buckling` | add Euler-buckling utilisation per compressed member |
| `--quiet` | suppress the console summary |

Real output: see the [quick start](#quick-start) above.

### `validate`

Checks schema, units, supports and topology **without solving**:

```console
$ truss-analysis validate examples/example1.json
{
  "file": "examples/example1.json",
  "valid": true,
  "errors": [],
  "units": "SI",
  "n_nodes": 5,
  "n_elements": 7,
  "topology": {
    "n_nodes": 5,
    "n_members": 7,
    "n_reactions": 4,
    "n_dof_free": 6,
    "indeterminacy": 1,
    "connected": true,
    "orphan_nodes": [],
    "zero_length_members": [],
    "duplicate_members": [],
    "self_loops": [],
    "mechanism": false,
    "rank_k_ff": 6,
    "cond_k_ff": 40.66278238719741,
    "cond_warning": false,
    "symmetric": true
  }
}
```

An invalid model is reported with reasons and exit code 1 (real run):

```console
$ truss-analysis validate bad.json
{
  "file": "bad.json",
  "valid": false,
  "errors": [
    "input validation: Insufficient constraints for stability: 0 < 3"
  ],
  "units": "SI"
}
$ echo $?
1
```

### `generate`

Builds a parametric model (Warren / Pratt / Howe) as canonical JSON:

```console
$ truss-analysis generate --family warren --panels 4 --span 24 --height 3.6 -o warren4.json
$ head -c 240 warren4.json
{"elements":[{"A":0.01,"E":210000000000.0,"I_sec":0.0001001736111111111,"alpha":1.2e-05,"delta_L0":0.0,"delta_T":0.0,"effective_length_factor":1.0,"id":1,"node_i":1,"node_j":2,"section_type":"idealised_square_hss"},{"A":0.01,"E":21000000000…
```

(The generated model is a single JSON line; the excerpt above is the verbatim
first 240 bytes. 9 nodes, 15 members, 7 loaded nodes.)

| Option | Default | Meaning |
|---|---|---|
| `--family {warren,pratt,howe}` | required | truss family |
| `--panels N` | required | number of panels |
| `--span L` | required | total span [m] |
| `--height H` | required | truss height [m] |
| `--area A` | `0.01` | cross-sectional area [m²] |
| `--youngs-modulus E` | `210e9` | Young's modulus [Pa] |
| `--thermal-expansion` | `1.2e-5` | expansion coefficient [1/°C] |
| `--total-load P` | `100e3` | total vertical load [N], distributed family-independently over the loaded chord nodes |
| `-o, --output PATH` | stdout | destination file |

### `version`

```console
$ truss-analysis version
2.9.0
```

(Installed from a source export. A git checkout reports the exact
setuptools-scm string of the working tree instead.)

## Python API

Everything below was executed against this repository; printed values are
real output.

### 1. Analyse a model file

```python
from truss_analysis import run

result = run("examples/example1.json", check_buckling=True, quiet=True)

print(result.status)  # SUCCESS
print(result.equilibrium["is_valid"])  # True
print(max(abs(e["N"]) for e in result.element_forces))  # 40000.00000000001
print([b["id"] for b in result.buckling if b["ratio"] > 1.0])  # ['3', '6', '7']
```

`AnalysisResult` fields: `status`, `displacements` (per node `ux`, `uy` [m]),
`element_forces` (per member `N` [N] + `status`), `reactions` (`Fx`, `Fy` [N]),
`equilibrium` (residuals + validity), `buckling` (utilisation per compressed
member; empty unless requested).

### 2. Parametric topology + temperature-dependent criticality

```python
from truss_analysis import Element, Node, compute_ci_for_topology, generate_topology
from truss_analysis.material import k_E, k_y

model = generate_topology("warren", n_panels=4, span=16.0, height=3.0)
nodes = [
    Node(
        id=str(n["id"]),
        x=n["x"],
        y=n["y"],
        is_support=n.get("is_support", False),
        support_dx=n.get("support_dx", False),
        support_dy=n.get("support_dy", False),
    )
    for n in model["nodes"]
]
elements = [
    Element(
        id=str(e["id"]),
        node_i=str(e["node_i"]),
        node_j=str(e["node_j"]),
        E=e["E"],
        A=e["A"],
    )
    for e in model["elements"]
]
loads = {str(ld["node_id"]): {"Fx": ld["Fx"], "Fy": ld["Fy"]} for ld in model["loads"]}

print(k_E(600.0), k_y(600.0))  # 0.31 0.47   (EN 1993-1-2 Table 3.1)

res = compute_ci_for_topology(nodes, elements, loads, {}, "uniform", 600.0)
print(res.top_5)  # ['6', '2', '3', '5', '7']
# a uniform temperature field provably does not reorder members *when no
# member carries an imposed thermal strain* (alpha = 0 here; with alpha > 0
# on a restrained structure the base state itself becomes temperature-
# dependent -- see docs/theory.md 5.4):
print(res.tau_vs_base)  # 1.0
```

Thermal scenarios for `compute_ci_for_topology`: `uniform`, `local_left`,
`local_mid`, `local_right`, `linear_gradient` (see
`truss_analysis.criticality.SCENARIOS`). Localised fields *do* reorder
members — the rank-stability statistic `tau_vs_base` quantifies it.

### 3. Retrofit triage under a budget

```python
from truss_analysis.retrofit import greedy, make_context

ctx = make_context(
    nodes,
    elements,
    loads,
    "uniform",
    600.0,
    f_y=235.0e6,
    alpha=0.7,
    budget_fraction=0.2,
)
outcome = greedy(ctx)

print(
    [m for m, a in zip(outcome.decision.member_ids, outcome.decision.actions) if a > 0]
)  # ['6']
print(outcome.metrics.cost)  # 6000.0
print(outcome.metrics.u_max)  # 0.002673174…
```

Alternative strategies (`exhaustive`, `robust_strategy`,
`redundant_strategy`, `stress_based_strategy`) share the identical context,
so their metrics are directly comparable.

### 4. Uncertainty quantification

```python
from truss_analysis.uncertainty import default_rv_specs, sample_spec_matrix

specs = default_rv_specs(fire_scenario_temperature=600.0)
means = {"live_load": 1.0, "f_y": 235.0e6, "fire_intensity": 600.0, "E": 210.0e9}
samples = sample_spec_matrix(specs, means, n=2000, seed=42)

print(samples["f_y"].mean() / 235.0e6)  # ≈ 1.0   (lognormal, COV 0.05)
print(samples["fire_intensity"].mean())  # ≈ 600.0 (truncated normal, σ 50)
```

Sampling depends only on `(specs, means, n, seed)` — never on call order.
`latin_hypercube`, `gaussian_copula_correlate`, `RunningStat` (streaming
moments) and `probabilistic_ranking` are exported from the same subpackage.

### 5. Energy validation

Every `run()` call verifies the generalized Clapeyron balance
`W_mech = U_strain + ½·W_prestress` and raises `EnergyValidationError` on
inconsistency; `check_energy` is public for custom pipelines. The public
surface also includes `assemble_global_matrices`, `solve`,
`calculate_element_forces`, `structural_report`, `validate_topology`,
`system_critical_temperature`, `dcr_field`, `ci_two_component` and
`euler_buckling_load` (see `truss_analysis.__all__`).

## Input JSON schema

Top-level object:

| Key | Type | Required | Validation / notes |
|---|---|---|---|
| `units` | `"SI"` \| `"Imperial"` | no (default `SI`) | unknown systems raise `UnitConversionError`; the CLI `--units` flag is the fallback when the key is absent |
| `nodes` | array | **yes** | see below; at least 3 constrained DOFs in total (`Insufficient constraints` otherwise) |
| `elements` | array | **yes** | see below; `E > 0`, `A > 0`; node references must exist |
| `loads` | array | no | nodal forces; unknown node ids are ignored |
| `temperature_change` | float | no | part of the canonical generator format; the analysis pipeline drives thermal effects from each element's `delta_T` |
| `options` | object | no | `use_sparse` (CSR assembly + SuperLU solve), `bc_method` (`"elimination"` default \| `"penalty"`), `penalty_value` (absolute N/m; omit to derive `1e10 × max\|diag K\|`) are **applied**. `plot_results` / `displacement_scale` are presentation-only and consumed by the plotting layer. An unrecognised key raises `InputIgnoredWarning` rather than being swallowed. Equivalent `run()` kwargs override the file |

`nodes[i]`:

| Key | Type | Required | Notes |
|---|---|---|---|
| `id` | int \| string | **yes** | normalised to string; must be unique |
| `x`, `y` | float | **yes** | coordinates [m] (converted from ft in Imperial) |
| `is_support` | bool | no (default `false`) | must agree with the restraints — see below |
| `support_dx`, `support_dy` | bool | no (default `false`) | constrained DOFs; pinned = both true, roller = one true |

A DOF is constrained only when `is_support` **and** the matching restraint are
both set. Because a restraint without the flag is silently ignored — producing a
*different structure* from the one described — both inconsistent combinations
are rejected by `validate_inputs`. Note also that "at least 3 constraints in
total" is a **sanity check, not a stability proof**: three parallel rollers in a
line satisfy the count and are still a mechanism. Stability is decided
numerically from `rank(K_ff)`; see `docs/theory.md` §3.1.

`elements[i]`:

| Key | Type | Required | Notes |
|---|---|---|---|
| `id` | int \| string | **yes** | normalised to string; must be unique |
| `node_i`, `node_j` | int \| string | **yes** | must reference existing nodes; self-loops and duplicate members are rejected by `validate` |
| `E` | float | **yes** | Young's modulus [Pa]; must be > 0 |
| `A` | float | **yes** | cross-sectional area [m²]; must be > 0 |
| `I_sec` (alias `I`) | float | no (default `0`) | second moment of area [m⁴], used by the buckling check |
| `alpha` | float | no (default `0`) | thermal expansion coefficient [1/°C] |
| `delta_T` | float | no (default `0`) | member temperature change [°C] (°F differences converted with 5/9 in Imperial) |
| `delta_L_free` (alias `delta_L0`) | float | no (default `0`) | imposed free length change (fabrication error) [m] |
| `density` (alias `rho`) | float | no (default `0`) | mass density [kg/m³]; when > 0, self-weight is applied as half the member weight per end node. In Imperial input this is read as **slug/ft³** (steel = 15.23), *not* lbf/ft³ — see below |
| `effective_length_factor` (alias `k_factor`) | float | no (default `1.0`) | buckling effective-length factor $K$; must be > 0. Applied as $P_{cr} = \pi^2 E I / (KL)^2$ by the buckling check **and** the fire limit state |
| `properties` | object | no | nested fallback for `I_sec` / `density` / `effective_length_factor`. The **top level wins**; a disagreement raises `InputIgnoredWarning` naming both values instead of being resolved silently |
| `section_type` | string | no | descriptive metadata from the canonical generator format; not used by the solver |

`loads[i]`:

| Key | Type | Required | Notes |
|---|---|---|---|
| `node_id` (alias `id`) | int \| string | **yes** | target node |
| `Fx`, `Fy` | float | no (default `0`) | force components [N] (converted from lbf in Imperial) |

Legacy tolerance: keys with stray surrounding spaces are accepted and
normalised (`"E "` → `"E"`), matching files produced by older tooling.

### Imperial input units, field by field

The Imperial system in everyday structural use is **mixed** — lengths in feet,
moduli in psi, forces in lbf — and those choices are not self-consistent for
mass. The density convention therefore has to be stated explicitly, because
guessing produces a silent factor-of-32 error in self-weight.

| Input field | Imperial unit read | SI produced | Factor |
|---|---|---|---|
| `x`, `y` | ft | m | 0.3048 |
| `A` | ft² | m² | 0.092903 |
| `I_sec` | ft⁴ | m⁴ | 0.0086309 |
| `E` | psi (lbf/in²) | Pa | 6894.757 |
| `Fx`, `Fy` | lbf | N | 4.44822 |
| `delta_T` | °F **difference** | K difference | 5/9 |
| `alpha` | 1/°F | 1/K | 1.8 |
| `density` | **slug/ft³** | kg/m³ | 515.379 |
| `weight_density` | lbf/ft³ (pcf) | kg/m³ (mass) | 16.0185 |

`density` is read as **slug/ft³** — the mass unit coherent with
`force = lbf`, `length = ft`. Structural steel is `15.23 slug/ft³`, *not* `490`.
`490` is its specific weight in lbf/ft³ (pcf), and the two differ by exactly
standard gravity, `32.174 ft/s²`. If you work in pcf, use the
`weight_density` quantity, which divides through by *g* in both unit systems.

Supplying `490` as `density` is detected: the converted value falls outside the
plausible range for matter and raises `UnitAmbiguityWarning` naming both
conventions. `delta_T` is a temperature *difference*, so no 32 °F offset is
applied — a 100 °F rise is a 55.56 K rise. Absolute temperatures are always
degrees Celsius and are never converted.

## Output JSON schema

`analyze --output result.json` writes (mirrors the `AnalysisResult`
dataclass; all values SI):

```json
{
  "status": "SUCCESS",
  "displacements": {"<node_id>": {"ux": 0.0, "uy": 0.0}},
  "element_forces": [{"id": "<element_id>", "N": 0.0,
                      "status": "Tension|Compression|Zero",
                      "delta_L_mech": 0.0, "delta_L_prestress": 0.0}],
  "reactions": {"<node_id>": {"Fx": 0.0, "Fy": 0.0}},
  "equilibrium": {"sum_fx": 0.0, "sum_fy": 0.0, "sum_m": 0.0,
                  "is_valid": true},
  "buckling": [{"id": "<element_id>", "N": 0.0, "length": 0.0,
                "P_cr": 0.0, "ratio": 0.0, "slenderness": 0.0,
                "k_factor": 1.0, "status": "checked", "safe": true}]
}
```

`buckling` is empty unless `--check-buckling` is passed. `status` is one of
`"checked"`, `"tension"`, `"zero_force"` or `"unknown"`.

> **`safe` defaults to `false`, not `true`.** A compressed member with no usable
> `I_sec` returns `status="unknown"`, `safe=false` and raises
> `BucklingCheckWarning`. It previously returned `safe=true` — missing data was
> being interpreted as safety.

`equilibrium` additionally reports the scales its verdict was reached with
(`ref_force`, `ref_moment`, `length_char`, `force_limit`, `moment_limit`). Force
and moment residuals have different dimensions and are tested against separate
bounds; the moment bound is the force scale times the bounding-box diagonal.

## Built-in checks

* **Graph/topology validation** — orphan nodes, zero-length members,
  duplicate members, self-loops, connectivity, static indeterminacy count,
  SVD rank and condition number of the reduced stiffness matrix
  (`structural_report`, surfaced by `truss-analysis validate`).
* **Energy validation** — the generalized Clapeyron theorem
  `W_mech = U_strain + ½·W_prestress` must hold to a tight relative
  tolerance on every solve (`EnergyValidationError` otherwise).
* **Equilibrium** — ΣFx, ΣFy, ΣM residuals of reactions vs. applied loads.
* **Uniform-temperature invariance** — scaling every member's stiffness
  identically cannot reorder the criticality ranking *while no member carries
  an imposed thermal strain*; the engine measures this property (Kendall
  tau-b = 1) instead of assuming it, and `tests/test_thermal_demand.py` pins
  the refined claim for heated restrained structures (docs/theory.md 5.4).
* **Reference bridge (optional)** — `pip install "truss-analysis[validation]"`
  adds comparison utilities against the OpenSees finite-element framework
  (`truss_analysis.validation`).

## Testing and development

```bash
make install        # runtime + dev deps + pre-commit hooks
make test           # pytest with coverage report
make test-cov       # pytest with the >= 90 % coverage gate
make lint           # ruff check + format check
make type-check     # mypy (strict) on src/
make check-all      # lint + type-check + test-cov (what CI runs)
make build          # sdist + wheel + twine check
make stats          # re-measure and patch the README test/coverage numbers
make sync-requirements  # regenerate requirements*.txt from pyproject.toml
```

Current status on this branch: **1066 tests passing, 94.4 % coverage**
(`pytest tests/`), `ruff` clean (extended rule set: E, F, I, W, UP, B, SIM,
RUF, PT, N, D), `mypy --strict` clean on all 49 library modules.

The pinned `pre-commit` chain runs ruff, mypy (src/), detect-secrets (with
baseline), a repository hygiene scanner (forbidden artefacts/paths/size
limits) and the usual file hygiene hooks. CI (GitHub Actions) runs lint,
strict typing, the test matrix (ubuntu/windows/macos × 3.10/3.11/3.12), the
coverage gate, packaging checks and a Docker build.

## Project layout

```text
src/truss_analysis/
├── assembly.py, solver.py, postprocess.py   # stiffness pipeline
├── model.py, fileio.py, units.py            # data contract + I/O + units
├── main.py                                  # CLI (analyze/validate/generate/version)
├── graph_validation.py                      # topology checks
├── limitstates.py, degradation.py           # DCR, capacity, critical temperature + failure mode
├── sections.py, heterogeneity.py            # section models
├── material/                                # EN 1993-1-2 data + interpolators (+ secant alpha)
├── thermal/                                 # ISO 834 curve + lumped-capacitance heating + legacy shims
├── stability.py                             # geometric stiffness K_G + linearised bifurcation factor
├── criticality/                             # rank-1 CI engine, indices, ranking, scenarios
├── uncertainty/                             # random variables, Iman-Conover/copula sampling, streaming, ranking
├── retrofit/                                # actions, costs, strategies
├── validation/                              # metrics, reference bridge, surrogate
├── sensitivity.py, reliability.py           # gradients + reliability helpers (chi-aligned margins)
└── visualization.py                         # plotting (lazy matplotlib)
tests/                                       # 1066 tests incl. tests/validation/
docs/theory.md, docs/error_codes.md          # formulation + error reference
examples/                                    # runnable example models
scripts/                                     # repository utilities
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) ([فارسی](CONTRIBUTING.fa.md)).
Short version: fork → `pip install -e ".[dev]"` → `pre-commit install` →
make your change with tests → `make check-all` green → open a pull request.

## Citation

Citation metadata is maintained in [CITATION.cff](CITATION.cff):

```bibtex
@software{truss_analysis,
  author  = {bmhmdyan279-png},
  title   = {truss\_analysis: linear 2D truss finite-element analysis for Python},
  year    = {2026},
  version = {2.9.0},
  license = {MIT},
  url     = {https://github.com/bmhmdyan279-png/truss-analysis-2d}
}
```

## License

MIT — see [LICENSE](LICENSE).
