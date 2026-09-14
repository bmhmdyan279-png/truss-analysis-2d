"""Command-line interface and high-level analysis orchestration.

The console script ``truss-analysis`` exposes four subcommands:

``analyze``
    Run the full linear analysis pipeline on one input JSON file and
    print a human-readable summary; optionally export JSON/CSV/Markdown
    results and a plot.
``validate``
    Check an input JSON file (schema, units, supports, topology) without
    solving it, and print a machine-readable report.
``generate``
    Generate a parametric truss model (Warren, Pratt or Howe family) and
    write it as canonical JSON.
``version``
    Print the installed package version.

Backwards compatibility: calling the CLI with a bare input path
(``truss-analysis model.json --check-buckling``) is equivalent to
``truss-analysis analyze model.json --check-buckling``.

Exit codes
----------
0
    Success.
1
    The analysis or validation failed (invalid input, singular system,
    energy check failure, ...).  Details are printed to stderr or, for
    ``validate``, in the JSON report on stdout.
2
    Usage error: unknown option, unreadable or missing input file.

Output JSON schema (``analyze --output RESULT.json``)
-----------------------------------------------------
::

    {
      "status": "SUCCESS",
      "displacements": {"<node_id>": {"ux": <float m>, "uy": <float m>}},
      "element_forces": [{"id": "<element_id>", "N": <float N>,
                          "status": "<TENSION|COMPRESSION|ZERO>"}, ...],
      "reactions": {"<node_id>": {"Fx": <float N>, "Fy": <float N>}},
      "equilibrium": {"sum_fx": <float>, "sum_fy": <float>,
                      "sum_m": <float>, "is_valid": <bool>},
      "buckling": [{"id": "<element_id>", "N": <float N>,
                    "length": <float m>, "P_cr": <float N or null>,
                    "ratio": <float>, "safe": <bool>}, ...]
    }

All quantities are SI (metres, newtons, pascals) regardless of the input
unit system; the ``"buckling"`` list is empty unless ``--check-buckling``
is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .assembly import assemble_global_matrices
from .exceptions import InputIgnoredWarning, InputValidationError, TrussError
from .fileio import load_json
from .graph_validation import TopologyValidationError, structural_report
from .model import Element, Node, validate_inputs
from .postprocess import (
    calculate_buckling,
    calculate_element_forces,
    calculate_reactions,
    check_equilibrium,
    imposed_strain_energy,
)
from .solver import check_energy, solve, solve_penalty_with_energy
from .topology_generator import generate_topology, model_to_json
from .units import to_si

G = 9.80665  # standard gravity [m/s^2], used for optional self-weight

__all__ = ["AnalysisResult", "main", "run"]


def _pure(obj: Any) -> Any:
    """Convert numpy scalars to plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _pure(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pure(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _normalize_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Strip trailing/leading spaces from keys (backward-compat for old JSON)."""
    return {k.strip(): v for k, v in d.items()}


def _get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get value from dict, tolerating keys with trailing spaces."""
    nd = _normalize_dict(d)
    return nd.get(key, default)


def _resolve_element_field(
    raw: dict[str, Any], key: str, aliases: tuple[str, ...] = ()
) -> tuple[Any, str | None]:
    """Resolve one element field from the top level or the ``properties`` block.

    Precedence is **top level first, then ``properties``**. The top-level key
    is the documented, generated-model location, so it wins; ``properties``
    is honoured as a nested fallback so that hand-written models which group
    section data there are not silently reduced to ``I_sec = 0``.

    Parameters
    ----------
    raw : dict[str, Any]
        Raw element payload.
    key : str
        Canonical field name, e.g. ``"I_sec"``.
    aliases : tuple[str, ...], optional
        Additional accepted spellings, tried in order at each level.

    Returns
    -------
    tuple[Any, str or None]
        The resolved value (``None`` when absent everywhere) and a warning
        message if the two levels disagreed. A disagreement is reported rather
        than resolved silently: ``I_sec`` controls buckling capacity, so a
        factor-of-100 ambiguity between two locations in the same element must
        not pass unnoticed.
    """
    props = _get(raw, "properties", {}) or {}
    if not isinstance(props, dict):
        props = {}

    candidates = (key, *aliases)
    top_val, top_key = None, None
    for cand in candidates:
        val = _get(raw, cand)
        if val is not None:
            top_val, top_key = val, cand
            break

    prop_val, prop_key = None, None
    for cand in candidates:
        val = _get(props, cand)
        if val is not None:
            prop_val, prop_key = val, cand
            break

    if top_val is None:
        return prop_val, None
    if prop_val is None:
        return top_val, None

    # Both present: top level wins, but flag a material disagreement.
    try:
        disagree = abs(float(top_val) - float(prop_val)) > 1e-9 * max(
            abs(float(top_val)), abs(float(prop_val)), 1.0
        )
    except (TypeError, ValueError):
        disagree = top_val != prop_val

    if not disagree:
        return top_val, None

    eid = _get(raw, "id", "?")
    return top_val, (
        f"element {eid}: '{top_key}'={top_val!r} at top level conflicts with "
        f"properties.{prop_key}={prop_val!r}; using the top-level value"
    )


#: Element payload keys that the parser understands. Anything else is reported
#: so that a typo (``"I_Sec"``, ``"density "``, ``"f_y"``) cannot silently
#: reduce to a default.
_KNOWN_ELEMENT_KEYS = frozenset(
    {
        "id",
        "node_i",
        "node_j",
        "E",
        "A",
        "I_sec",
        "I",
        "alpha",
        "delta_T",
        "delta_L_free",
        "delta_L0",
        "density",
        "rho",
        "effective_length_factor",
        "k_factor",
        "section_type",
        "properties",
    }
)


def _parse_model(
    data: dict[str, Any], unit_sys_default: str = "SI"
) -> tuple[list[Node], list[Element], str]:
    """Build SI-unit node/element objects from a raw input dictionary.

    Section and stability fields are resolved through
    :func:`_resolve_element_field`, which honours a nested ``properties``
    block as a fallback and warns when it disagrees with the top level.
    ``effective_length_factor`` and ``density`` are now read into the
    :class:`~truss_analysis.model.Element`; previously both were accepted in
    the schema, documented in the README, and then dropped — so a user
    supplying ``K = 0.7`` silently received a pin-ended ``K = 1`` buckling
    check that contradicted the fire limit-state path in
    :mod:`truss_analysis.limitstates`, which did use ``K``.

    Parameters
    ----------
    data : dict[str, Any]
        Raw JSON payload with ``"nodes"`` and ``"elements"`` keys.
    unit_sys_default : str, optional
        Unit system used when the payload does not declare ``"units"``.

    Returns
    -------
    tuple[list[Node], list[Element], str]
        Parsed nodes, parsed elements and the effective unit system.

    Warns
    -----
    InputIgnoredWarning
        When an element carries a key the parser does not understand, or when
        the top-level and ``properties`` values of one field disagree.
    """
    unit_sys = data.get("units", unit_sys_default)
    nodes = [
        Node(
            id=str(_get(n, "id")),
            x=to_si(_get(n, "x"), unit_sys, "L"),
            y=to_si(_get(n, "y"), unit_sys, "L"),
            is_support=bool(_get(n, "is_support", False)),
            support_dx=bool(_get(n, "support_dx", False)),
            support_dy=bool(_get(n, "support_dy", False)),
        )
        for n in data["nodes"]
    ]

    conflicts: list[str] = []
    unknown: list[str] = []
    elements: list[Element] = []
    for e in data["elements"]:
        i_sec, warn = _resolve_element_field(e, "I_sec", ("I",))
        if warn:
            conflicts.append(warn)
        k_fac, warn = _resolve_element_field(
            e, "effective_length_factor", ("k_factor",)
        )
        if warn:
            conflicts.append(warn)
        density, warn = _resolve_element_field(e, "density", ("rho",))
        if warn:
            conflicts.append(warn)

        eid = str(_get(e, "id"))
        for key in _normalize_dict(e):
            if key not in _KNOWN_ELEMENT_KEYS:
                unknown.append(f"element {eid}: unrecognised key '{key}'")

        elements.append(
            Element(
                id=eid,
                node_i=str(_get(e, "node_i")),
                node_j=str(_get(e, "node_j")),
                E=to_si(_get(e, "E"), unit_sys, "E"),
                A=to_si(_get(e, "A"), unit_sys, "A"),
                I_sec=to_si(float(i_sec or 0.0), unit_sys, "I_sec"),
                alpha=to_si(_get(e, "alpha", 0.0), unit_sys, "alpha"),
                delta_T=to_si(_get(e, "delta_T", 0.0), unit_sys, "delta_T"),
                delta_L_free=to_si(
                    _get(e, "delta_L_free", _get(e, "delta_L0", 0.0)), unit_sys, "L"
                ),
                density=to_si(float(density or 0.0), unit_sys, "density"),
                effective_length_factor=float(k_fac if k_fac is not None else 1.0),
            )
        )

    for msg in (*conflicts, *unknown):
        warnings.warn(msg, InputIgnoredWarning, stacklevel=3)

    return nodes, elements, unit_sys


@dataclass
class AnalysisResult:
    """Structured container for all analysis outputs.

    Attributes
    ----------
    status : str
        ``"SUCCESS"`` when the whole pipeline completed.
    displacements : dict[str, dict[str, float]]
        Nodal displacements ``{"ux", "uy"}`` in metres, keyed by node id.
    element_forces : list[dict[str, Any]]
        Per-element results; each entry carries at least ``"id"``, the axial
        force ``"N"`` and a ``"status"`` label.
    reactions : dict[str, dict[str, float]]
        Support reactions ``{"Fx", "Fy"}`` in newtons, keyed by node id.
    equilibrium : dict[str, Any]
        Global equilibrium residuals and validity flag.
    buckling : list[dict[str, Any]]
        Euler buckling utilisation per compressed member; empty unless the
        buckling check was requested.
    """

    status: str
    displacements: dict[str, dict[str, float]]
    element_forces: list[dict[str, Any]]
    reactions: dict[str, dict[str, float]]
    equilibrium: dict[str, Any]
    buckling: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 62, " TRUSS ANALYSIS RESULT", "=" * 62]
        for r in self.element_forces:
            nid = r.get("id")
            force = r.get("N", 0.0)
            status = r.get("status", "?")
            lines.append(f"Element {nid}: N = {force:14.3f}  [{status}]")
        for nid, rec in self.reactions.items():
            fx = rec["Fx"]
            fy = rec["Fy"]
            lines.append(f"Reaction @ {nid}: Fx = {fx:12.3f}  Fy = {fy:12.3f}")
        eq = self.equilibrium
        ok = "OK" if eq["is_valid"] else "FAIL"
        lines.append(
            f"Equilibrium: dFx={eq['sum_fx']:.2e} "
            f"dFy={eq['sum_fy']:.2e} dM={eq['sum_m']:.2e} -> {ok}"
        )
        for b in self.buckling:
            if b["N"] < 0 and b["P_cr"]:
                flag = "OK" if b["safe"] else "BUCKLING RISK"
                lines.append(f"Buckling {b['id']}: ratio={b['ratio']:.3f} -> {flag}")
        lines.append(f"Status: {self.status}")
        return "\n".join(lines)


def _write_markdown(result: AnalysisResult, path: str) -> None:
    """Write Markdown report."""
    lines = [
        "# Truss Analysis Report",
        "",
        "## Axial Forces",
        "",
        "| Element | N (force) | Status |",
        "|---|---|---|",
    ]
    for r in result.element_forces:
        eid = r.get("id")
        force = r.get("N", 0.0)
        status = r.get("status", "-")
        lines.append(f"| {eid} | {force:.3f} | {status} |")
    lines += [
        "",
        "## Reactions",
        "",
        "| Node | Fx | Fy |",
        "|---|---|---|",
    ]
    for nid, rec in result.reactions.items():
        lines.append(f"| {nid} | {rec['Fx']:.3f} | {rec['Fy']:.3f} |")
    eq = result.equilibrium
    lines += [
        "",
        "## Static Equilibrium",
        "",
        f"- sum(Fx) = {eq['sum_fx']:.3e}",
        f"- sum(Fy) = {eq['sum_fy']:.3e}",
        f"- sum(M) = {eq['sum_m']:.3e}",
        f"- Valid: {eq['is_valid']}",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


#: Option keys the pipeline understands. Anything else in the ``options``
#: block is reported, so a typo cannot silently change nothing.
_KNOWN_OPTION_KEYS = frozenset(
    {"use_sparse", "bc_method", "penalty_value", "plot_results", "displacement_scale"}
)

#: ``options`` keys that control presentation rather than physics. They are
#: consumed by the plotting layer, not by the solver, and saying so explicitly
#: is better than leaving the user to guess whether they took effect.
_PRESENTATION_OPTION_KEYS = frozenset({"plot_results", "displacement_scale"})


def _resolve_options(
    data: dict[str, Any],
    use_sparse: bool | None,
    bc_method: str | None,
    penalty_value: float | None,
) -> tuple[bool, str, float | None]:
    """Merge the JSON ``options`` block with explicit arguments.

    Explicit arguments win over the file, because a caller passing a value has
    stated an intent more recently than whoever wrote the model. Unrecognised
    keys in the block are reported: silently accepting a setting and then
    ignoring it means the user believes they changed the physics when they did
    not.

    Returns
    -------
    tuple[bool, str, float or None]
        ``(use_sparse, bc_method, penalty_value)`` where ``penalty_value`` is
        an absolute stiffness in N/m, or ``None`` to let the solver derive a
        scale-aware penalty from ``max(|diag(K)|)``.
    """
    block = _get(data, "options", {}) or {}
    if not isinstance(block, dict):
        block = {}
    block = _normalize_dict(block)

    for key in block:
        if key not in _KNOWN_OPTION_KEYS:
            warnings.warn(
                f"unrecognised options key '{key}' ignored; known keys are "
                f"{', '.join(sorted(_KNOWN_OPTION_KEYS))}",
                InputIgnoredWarning,
                stacklevel=4,
            )

    sparse = (
        use_sparse if use_sparse is not None else bool(block.get("use_sparse", False))
    )
    method = (
        bc_method
        if bc_method is not None
        else str(block.get("bc_method", "elimination"))
    )
    # `penalty_value` is an absolute stiffness in N/m -- that is what the
    # shipped example models carry, and the conventional meaning of the name.
    # Left as None when absent so the solver derives a scale-aware penalty.
    raw_penalty = (
        penalty_value if penalty_value is not None else block.get("penalty_value")
    )
    resolved_penalty: float | None = None if raw_penalty is None else float(raw_penalty)

    if method not in ("elimination", "penalty"):
        raise InputValidationError(
            f"options.bc_method must be 'elimination' or 'penalty', got {method!r}"
        )
    return sparse, method, resolved_penalty


def run(
    filepath: str | Path,
    unit_sys: str = "SI",
    plot: bool = False,
    check_buckling: bool = False,
    output: str | Path | None = None,
    csv_path: str | Path | None = None,
    report_path: str | Path | None = None,
    plot_path: str | Path | None = None,
    quiet: bool = False,
    use_sparse: bool | None = None,
    bc_method: str | None = None,
    penalty_value: float | None = None,
    check_condition: bool = True,
) -> AnalysisResult:
    """Run the full analysis pipeline and return an AnalysisResult.

    Parameters
    ----------
    filepath : str or Path
        Path to the input JSON model.
    unit_sys : str, optional
        Fallback unit system (``"SI"`` or ``"Imperial"``) when the input
        file does not declare ``"units"``.
    plot : bool, optional
        Show an interactive plot (requires the ``viz`` extra).
    check_buckling : bool, optional
        Also compute buckling utilisation for compressed members.
    output, csv_path, report_path, plot_path : str, Path or None, optional
        Export destinations for the JSON result, a CSV force table, a
        Markdown report and a PNG plot respectively.
    quiet : bool, optional
        Suppress the human-readable summary on stdout.
    use_sparse : bool or None, optional
        Assemble and solve in CSR sparse storage. ``None`` (default) takes the
        value from the model's ``options`` block, else ``False``.
    bc_method : str or None, optional
        ``"elimination"`` (default) or ``"penalty"``. ``None`` takes the value
        from the model's ``options`` block.
    penalty_value : float or None, optional
        Absolute penalty stiffness [N/m], used only when
        ``bc_method="penalty"``. ``None`` takes the value from the model's
        ``options`` block; when absent there too, the solver derives a
        scale-aware penalty of ``1e10 * max(|diag(K)|)``.
    check_condition : bool, default True
        Run the SVD rank/conditioning screen before the elimination solve.
        Callers solving the same topology repeatedly (temperature sweeps,
        Monte Carlo, retrofit triage) should pass ``False`` and let Cholesky
        act as the singularity gate, as the :func:`truss_analysis.solver.solve`
        docstring recommends; the screen costs an ``O(n^3)`` decomposition
        that dominates the solve itself. The penalty path never screens (its
        inflated diagonal makes the generic warning meaningless) so this flag
        only affects ``bc_method="elimination"``.

    Returns
    -------
    AnalysisResult
        Structured container with all outputs (see the module docstring
        for the JSON schema).

    Raises
    ------
    InputValidationError
        If ``bc_method`` is neither ``"elimination"`` nor ``"penalty"``.

    Warns
    -----
    InputIgnoredWarning
        If the model's ``options`` block contains an unrecognised key.
    """
    raw_data = load_json(filepath)
    # Normalize top-level keys (handle old JSON with trailing spaces)
    data = _normalize_dict(raw_data)
    nodes, elements, unit_sys = _parse_model(data, unit_sys)
    validate_inputs(nodes, elements)
    use_sparse_eff, bc_method_eff, penalty_eff = _resolve_options(
        data, use_sparse, bc_method, penalty_value
    )

    # The penalty method inflates the diagonal of the *full* system and solves
    # it whole, so it is dense by construction. Resolve the storage layout here,
    # before any load is accumulated into F_ext: reassembling later would throw
    # those loads away and silently solve an unloaded model.
    if use_sparse_eff and bc_method_eff == "penalty":
        warnings.warn(
            "options.bc_method='penalty' requires dense assembly; "
            "ignoring use_sparse=true for this run",
            InputIgnoredWarning,
            stacklevel=2,
        )
        use_sparse_eff = False

    K, F_ext, F_mechanical, fixed_dofs = assemble_global_matrices(
        nodes, elements, sparse=use_sparse_eff
    )
    node_map = {node.id: i for i, node in enumerate(nodes)}
    applied_loads: list[dict[str, Any]] = []

    for lf in data.get("loads", []):
        nid = str(_get(lf, "node_id", _get(lf, "id")))
        if nid not in node_map:
            continue
        idx = node_map[nid]
        fx = to_si(_get(lf, "Fx", 0.0), unit_sys, "F")
        fy = to_si(_get(lf, "Fy", 0.0), unit_sys, "F")
        F_ext[2 * idx] += fx
        F_ext[2 * idx + 1] += fy
        F_mechanical[2 * idx] += fx
        F_mechanical[2 * idx + 1] += fy
        applied_loads.append({"node_id": nid, "Fx": fx, "Fy": fy})

    # Optional self-weight from the element mass density [kg/m^3].
    #
    # Density is read from the parsed Element rather than the raw payload: the
    # parser has already resolved the `density`/`rho` aliases and applied the
    # unit conversion, so an Imperial input is handled correctly. Reading the
    # raw key here bypassed to_si() entirely and silently produced a
    # self-weight in the wrong units for any non-SI model, while also leaving
    # Element.density permanently at its default -- two API surfaces for one
    # physical quantity, only one of which was live.
    weight_per_node: dict[int, float] = {}
    for elem in elements:
        rho = elem.density
        if rho <= 0.0:
            continue
        i, j = node_map[elem.node_i], node_map[elem.node_j]
        length = float(np.hypot(nodes[j].x - nodes[i].x, nodes[j].y - nodes[i].y))
        weight = rho * elem.A * length * G
        weight_per_node[i] = weight_per_node.get(i, 0.0) + weight / 2.0
        weight_per_node[j] = weight_per_node.get(j, 0.0) + weight / 2.0
    for idx, weight in weight_per_node.items():
        F_ext[2 * idx + 1] -= weight
        F_mechanical[2 * idx + 1] -= weight
        applied_loads.append({"node_id": nodes[idx].id, "Fx": 0.0, "Fy": -weight})

    penalty_energy = 0.0
    if bc_method_eff == "penalty":
        U, penalty_energy = solve_penalty_with_energy(
            K, F_ext, fixed_dofs, penalty_value=penalty_eff
        )
    else:
        U = solve(K, F_ext, fixed_dofs, check_condition=check_condition)
    element_forces, strain_energy, prestress_work = calculate_element_forces(
        nodes, elements, U
    )
    check_energy(
        U,
        F_mechanical,
        strain_energy,
        prestress_work,
        energy_scale=imposed_strain_energy(nodes, elements),
        penalty_energy=penalty_energy,
    )
    reactions = calculate_reactions(nodes, K, U, F_ext, fixed_dofs)
    equilibrium = check_equilibrium(nodes, reactions, applied_loads)
    buckling = (
        calculate_buckling(nodes, elements, element_forces) if check_buckling else []
    )
    displacements = {
        node.id: {"ux": float(U[2 * i]), "uy": float(U[2 * i + 1])}
        for i, node in enumerate(nodes)
    }

    result = AnalysisResult(
        status="SUCCESS",
        displacements=_pure(displacements),
        element_forces=_pure(element_forces),
        reactions=_pure(reactions),
        equilibrium=_pure(equilibrium),
        buckling=_pure(buckling),
    )

    if output:
        Path(output).write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["element_id", "axial_force", "status"])
            for r in result.element_forces:
                w.writerow([r.get("id"), r.get("N"), r.get("status")])
    if report_path:
        _write_markdown(result, str(report_path))
    if plot or plot_path:
        from .visualization import plot_truss

        plot_truss(nodes, elements, U=U, results=element_forces, save_path=plot_path)

    if not quiet:
        print(result.summary())
    return result


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Execute the ``analyze`` subcommand."""
    try:
        run(
            args.input,
            unit_sys=args.units,
            plot=args.plot or bool(args.plot_path),
            check_buckling=args.check_buckling,
            output=args.output,
            csv_path=args.csv_path,
            report_path=args.report_path,
            plot_path=args.plot_path,
            quiet=args.quiet,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (TrussError, TopologyValidationError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Execute the ``validate`` subcommand."""
    try:
        raw_data = load_json(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    errors: list[str] = []
    report: dict[str, Any] = {"file": str(args.input), "valid": False, "errors": errors}
    data = _normalize_dict(raw_data)
    try:
        nodes, elements, unit_sys = _parse_model(data, args.units)
        validate_inputs(nodes, elements)
    except (TrussError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"input validation: {exc}")
        report["units"] = data.get("units", args.units)
        print(json.dumps(_pure(report), indent=2, ensure_ascii=False))
        return 1

    report["units"] = unit_sys
    report["n_nodes"] = len(nodes)
    report["n_elements"] = len(elements)

    converted = {
        "nodes": [
            {
                "id": n.id,
                "x": n.x,
                "y": n.y,
                "is_support": n.is_support,
                "support_dx": n.support_dx,
                "support_dy": n.support_dy,
            }
            for n in nodes
        ],
        "elements": [
            {"id": e.id, "node_i": e.node_i, "node_j": e.node_j, "E": e.E, "A": e.A}
            for e in elements
        ],
    }
    try:
        topo = structural_report(converted)
        report["topology"] = _pure(asdict(topo))
    except (TopologyValidationError, ValueError, KeyError) as exc:
        errors.append(f"topology: {exc}")

    report["valid"] = not errors
    print(json.dumps(_pure(report), indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


def _cmd_generate(args: argparse.Namespace) -> int:
    """Execute the ``generate`` subcommand."""
    try:
        model = generate_topology(
            family=args.family,
            n_panels=args.panels,
            span=args.span,
            height=args.height,
            area=args.area,
            youngs_modulus=args.youngs_modulus,
            thermal_expansion=args.thermal_expansion,
            total_load=args.total_load,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = model_to_json(model)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    """Execute the ``version`` subcommand."""
    from . import __version__

    print(__version__)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="truss-analysis",
        description="2D truss analysis: linear statics, thermal loads, "
        "buckling checks, input validation and parametric model generation.",
    )
    sub = parser.add_subparsers(dest="command")

    p_an = sub.add_parser(
        "analyze", help="run the full analysis on one input JSON file"
    )
    p_an.add_argument("input", help="path to the input JSON file")
    p_an.add_argument(
        "--units",
        default="SI",
        choices=["SI", "Imperial"],
        help="fallback unit system when the file does not declare one",
    )
    p_an.add_argument("-o", "--output", help="write the JSON result to this path")
    p_an.add_argument("--csv", dest="csv_path", help="write a CSV force table")
    p_an.add_argument("--report", dest="report_path", help="write a Markdown report")
    p_an.add_argument("--plot", action="store_true", help="show an interactive plot")
    p_an.add_argument("--plot-path", help="save the plot as a PNG file")
    p_an.add_argument(
        "--check-buckling", action="store_true", help="also check Euler buckling"
    )
    p_an.add_argument(
        "--quiet", action="store_true", help="suppress the summary on stdout"
    )
    p_an.set_defaults(func=_cmd_analyze)

    p_va = sub.add_parser(
        "validate", help="validate an input JSON file without solving it"
    )
    p_va.add_argument("input", help="path to the input JSON file")
    p_va.add_argument(
        "--units",
        default="SI",
        choices=["SI", "Imperial"],
        help="fallback unit system when the file does not declare one",
    )
    p_va.set_defaults(func=_cmd_validate)

    p_ge = sub.add_parser(
        "generate", help="generate a parametric truss model as canonical JSON"
    )
    p_ge.add_argument(
        "--family",
        required=True,
        choices=["warren", "pratt", "howe"],
        help="truss family",
    )
    p_ge.add_argument("--panels", type=int, required=True, help="number of panels")
    p_ge.add_argument("--span", type=float, required=True, help="total span [m]")
    p_ge.add_argument("--height", type=float, required=True, help="truss height [m]")
    p_ge.add_argument(
        "--area", type=float, default=0.01, help="cross-sectional area [m^2]"
    )
    p_ge.add_argument(
        "--youngs-modulus",
        type=float,
        default=210.0e9,
        help="Young's modulus [Pa]",
    )
    p_ge.add_argument(
        "--thermal-expansion",
        type=float,
        default=1.2e-5,
        help="thermal expansion coefficient [1/degC]",
    )
    p_ge.add_argument(
        "--total-load", type=float, default=100.0e3, help="total vertical load [N]"
    )
    p_ge.add_argument(
        "-o", "--output", help="write the model to this file (default: stdout)"
    )
    p_ge.set_defaults(func=_cmd_generate)

    p_ve = sub.add_parser("version", help="print the package version")
    p_ve.set_defaults(func=_cmd_version)

    return parser


_SUBCOMMANDS = frozenset({"analyze", "validate", "generate", "version"})


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Argument vector without the program name; defaults to
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: 0 success, 1 analysis/validation failure,
        2 usage error or missing input file.
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if (
        args_list
        and args_list[0] not in _SUBCOMMANDS
        and args_list[0]
        not in (
            "-h",
            "--help",
        )
    ):
        # Legacy invocation: a bare input path means "analyze".
        args_list = ["analyze", *args_list]
    args = parser.parse_args(args_list)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
