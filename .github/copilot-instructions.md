# GitHub Copilot Instructions — truss-analysis-2d

## What this project is

`truss_analysis` is a Python library for linear static analysis of 2D
planar (pin-jointed) trusses with the direct stiffness method, including
temperature-dependent steel properties per EN 1993-1-2, member
criticality screening, uncertainty quantification and retrofit
prioritisation utilities. It is published as a standard PyPI package
(`pip install truss_analysis`) with a console script `truss-analysis`.

## Real architecture (src layout)

```
src/truss_analysis/
├── model.py               # Node/Element dataclasses + input validation
├── fileio.py              # JSON input loading
├── units.py               # SI / Imperial conversion (everything internal is SI)
├── assembly.py            # global stiffness matrix + load vector assembly
├── solver.py              # linear solve with rank/conditioning guards,
│                          # Clapeyron energy check (check_energy)
├── postprocess.py         # member forces, reactions, equilibrium, buckling
├── main.py                # CLI (analyze / validate / generate / version) + run()
├── visualization.py       # optional matplotlib plotting (lazy import)
├── material/              # EN 1993-1-2 single source of truth:
│   ├── data/en1993_1_2_table3_1.json   # the ONLY place table values live
│   └── steel_eurocode.py  # interpolators k_E, k_y, k_p + constitutive law
├── thermal/               # compatibility shims (deprecated aliases)
├── sections.py            # idealised square HSS, Euler buckling load
├── topology_generator.py  # parametric Warren/Pratt/Howe + determinate controls
├── graph_validation.py    # connectivity/determinacy/conditioning reports
├── criticality/           # exact rank-1 (Sherman-Morrison) CI engine,
│                          # scenarios (temperature fields), tau-b ranking, NCI
├── limitstates.py         # DCR fields, member/system critical temperatures
├── degradation.py         # DegradationOperator ABC + registry
├── heterogeneity.py       # normalised inequality measures for CI fields
├── sensitivity.py         # independent validators (DDM, finite differences)
├── uncertainty/           # random variables, LHS + copula sampling,
│                          # streaming MC, probabilistic ranking
├── reliability.py         # Monte Carlo margin engine (yield/buckling/service)
├── reliability_adapter.py # glue: truss model -> ReliabilityEngine callback
├── retrofit/              # actions, cost models, prioritisation strategies
├── validation/            # reference checks: analytical, OpenSeesPy bridge,
│                          # cross-family surrogate transfer, rank metrics
└── exceptions.py          # TrussError hierarchy (Assembly, InputValidation,
                           # EnergyValidation, SingularMatrix, UnitConversion)
```

Dependency direction: `main`/CLI depends on everything above; core numerics
(`assembly`, `solver`, `postprocess`) depend only on `model`, `units` and
`exceptions`. The `validation/` package is optional at runtime: OpenSeesPy
is an extra (`pip install truss_analysis[validation]`), and plotting is an
extra (`[viz]`) — never import matplotlib or openseespy at module scope.

## Hard rules

1. **Never re-type material table values.** EN 1993-1-2 numbers live only
   in `material/data/en1993_1_2_table3_1.json`; `tests/test_material_single_source.py`
   proves by AST scan that no second copy exists. Load via
   `material.steel_eurocode`.
2. **All internal quantities are SI** (m, N, Pa, °C differences). Convert
   at the boundary with `units.to_si`.
3. **No fabricated reference numbers.** Validation tests skip when an
   optional dependency is missing; they never mock a physics solver.
4. **Docstrings follow the NumPy style** (Parameters/Returns/Raises) for
   every public function, class and method; the ruff `D` rules enforce it.
5. **Type everything**: `mypy --strict` must stay clean (`mypy src`).
6. **Lint/format**: `ruff check` and `ruff format` (line length 88) must
   stay clean over `src/ tests/ scripts/`.
7. **Tests for every feature** live in `tests/` (unit) and
   `tests/validation/` (reference comparisons); coverage gate is 90%.
8. **Dependencies**: `pyproject.toml` is the single source of truth; after
   changing it run `python scripts/sync_requirements.py` to regenerate the
   `requirements*.txt` mirrors (`tests/test_packaging.py` enforces sync).
9. **JSON output is data, not prose**: keep result payloads plain
   (floats/strings), serialisable with `json.dumps` after `main._pure`.
10. **Determinism**: anything sampled takes an explicit `seed`; tie-noise
    conventions in ranking are quantised at 1e-10 (`ranking.tau_b`).

## Style

- Frozen dataclasses for value objects; enums for closed sets.
- Raise the specific exception from `exceptions.py`, never bare `Exception`.
- Avoid single-letter locals except established mechanics symbols in
  formulas (`K`, `U`, `E`, `A`, `I_sec`); module constant `G` is gravity.
- Persian text is only user-facing output (bilingual README, plots via
  `visualization._persian`); code comments and docstrings are English.
