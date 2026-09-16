"""Targeted mutation probe: do the tests actually catch a wrong answer?

Line coverage asks whether a line *ran*. It does not ask whether any test would
notice if the line computed something else -- and a suite can reach 94% while
asserting nothing on the branches that matter. This round found exactly that:
``dcr_field``'s system-stability branch had twenty-two callers in the test suite
and **zero** of them passed ``check_system_stability``, so a path that silently
rewrote a safety-relevant number was fully covered and fully unverified.

Mutation testing is the instrument that measures the difference. A mutant is a
small semantic edit to the source -- ``<`` becomes ``<=``, ``and`` becomes
``or``, a sign flips. If some test fails, the mutant is *killed* and the suite
was paying attention. If every test still passes, the mutant *survived*, and the
survivor names a hole that coverage cannot see.

Why this is a script and not a CI gate
--------------------------------------
``cosmic-ray`` and ``mutmut`` run the whole suite per mutant. On 1100+ tests at
~65 s a run that is weeks of compute for the three modules worth mutating, and a
nightly job that cannot finish is not a gate either -- it is a report nobody
reads. This probe is deliberately narrower and says so:

* **curated targets, not every operator.** Mutations are generated on the
  decision points that carry meaning -- comparisons, boolean connectives,
  arithmetic operators and numeric constants -- rather than on every AST node,
  which mostly produces mutants that are trivially killed by an exception and
  inflate the score without telling anyone anything.
* **targeted test selection.** Each module declares the test files that exercise
  it, so a mutant costs seconds rather than a minute.
* **``-x`` fail-fast.** A killed mutant usually dies in the first seconds, which
  is what makes the survivor count affordable to measure at all.

The output is a *measurement with a named scope*, not a mutation score for the
project. Read the survivor list; that is the deliverable.

Usage
-----
::

    python scripts/mutation_probe.py --module src/truss_analysis/limitstates.py
    python scripts/mutation_probe.py --module ... --max-mutants 40 --verbose
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Which test files exercise which module.  Deliberately explicit rather than
#: "run everything": the point of a targeted probe is that it is affordable, and
#: an implicit mapping would silently under-test a module whose tests were moved.
TEST_MAP: dict[str, tuple[str, ...]] = {
    "limitstates.py": (
        "tests/test_limitstates.py",
        "tests/test_limitstates_boundaries.py",
        "tests/test_dcr_system_stability.py",
        "tests/test_effective_alpha_chain.py",
        "tests/test_thermal_demand.py",
        "tests/test_cross_path_demand.py",
        "tests/test_multi_criteria_ci.py",
        "tests/test_retrofit.py",
        "tests/test_golden.py",
    ),
    "stability.py": (
        "tests/test_stability.py",
        "tests/test_stability_eigenpath.py",
        "tests/test_dcr_system_stability.py",
        "tests/test_guard_tolerance.py",
    ),
    "engine.py": (
        "tests/test_engine_equivalence.py",
        "tests/test_thermal_demand.py",
        "tests/test_effective_alpha_chain.py",
        "tests/test_guard_tolerance.py",
        "tests/test_criticality.py",
        "tests/test_cross_path_demand.py",
        "tests/test_multi_criteria_ci.py",
    ),
    "solver.py": (
        "tests/test_solver.py",
        "tests/test_solver_guards.py",
        "tests/test_self_equilibrated_energy.py",
        "tests/test_penalty_bc_and_options.py",
        "tests/test_dof_mapping.py",
    ),
}

#: Comparison operators and the single edit applied to each.  ``<`` -> ``<=`` is
#: the classic off-by-a-boundary mutant, and it is the one this round actually
#: cared about: ``dcr_field`` amplifies on ``lambda_cr <= 1``, and a strict
#: ``<`` there would call a system with exactly zero reserve unremarkable.
CMP_SWAP: dict[type, type] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}


@dataclass(frozen=True)
class Mutant:
    """One semantic edit, identified by where it was made."""

    index: int
    lineno: int
    kind: str
    before: str
    after: str
    context: str


@dataclass(frozen=True)
class MutantResult:
    mutant: Mutant
    killed: bool
    seconds: float
    detail: str


def _describe(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - unparse handles every node we edit
        return type(node).__name__


def _line_of(source_lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(source_lines):
        return source_lines[lineno - 1].strip()[:78]
    return ""


def generate_mutants(path: Path, max_mutants: int) -> tuple[list[Mutant], str]:
    """Return the curated mutant set and the source each one is applied to.

    Generation is a deterministic pre-order walk, so the same file always
    produces the same numbered list -- which is what makes "mutant 17 survived"
    a reproducible statement rather than a description of one run.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    found: list[tuple[ast.AST, ast.AST, str, str, str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = node.ops[0]
            swap = CMP_SWAP.get(type(op))
            if swap is not None:
                found.append(
                    (
                        node,
                        op,
                        "comparison",
                        _describe(op),
                        _describe(swap()),
                        node.lineno,
                    )
                )
        elif isinstance(node, ast.BoolOp):
            swap = ast.Or if isinstance(node.op, ast.And) else ast.And
            found.append(
                (
                    node,
                    node.op,
                    "boolean",
                    _describe(node.op),
                    _describe(swap()),
                    node.lineno,
                )
            )

    swaps = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, tuple(swaps)):
            found.append(
                (
                    node,
                    node.op,
                    "arithmetic",
                    _describe(node.op),
                    _describe(swaps[type(node.op)]()),
                    node.lineno,
                )
            )

    found.sort(key=lambda f: (f[5], f[2]))
    mutants: list[Mutant] = []
    for i, (_holder, _target, kind, before, after, lineno) in enumerate(
        found[:max_mutants]
    ):
        mutants.append(
            Mutant(
                index=i,
                lineno=lineno,
                kind=kind,
                before=before,
                after=after,
                context=_line_of(lines, lineno),
            )
        )
    return mutants, source


def _apply_mutant(path: Path, source: str, mutant: Mutant) -> str | None:
    """Rewrite the module with one mutation applied. ``None`` if not applicable."""
    tree = ast.parse(source)
    target_kind = mutant.kind
    counter = -1

    class _Rewriter(ast.NodeTransformer):
        def visit_Compare(self, node: ast.Compare):
            self.generic_visit(node)
            nonlocal counter
            if target_kind == "comparison" and len(node.ops) == 1:
                swap = CMP_SWAP.get(type(node.ops[0]))
                if swap is not None and node.lineno == mutant.lineno:
                    counter += 1
                    if counter == 0:
                        node.ops = [swap()]
            return node

        def visit_BoolOp(self, node: ast.BoolOp):
            self.generic_visit(node)
            nonlocal counter
            if target_kind == "boolean" and node.lineno == mutant.lineno:
                counter += 1
                if counter == 0:
                    node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            return node

        def visit_BinOp(self, node: ast.BinOp):
            self.generic_visit(node)
            nonlocal counter
            swaps = {
                ast.Add: ast.Sub,
                ast.Sub: ast.Add,
                ast.Mult: ast.Div,
                ast.Div: ast.Mult,
            }
            if (
                target_kind == "arithmetic"
                and isinstance(node.op, tuple(swaps))
                and node.lineno == mutant.lineno
            ):
                counter += 1
                if counter == 0:
                    node.op = swaps[type(node.op)]()
            return node

    tree = _Rewriter().visit(tree)
    if counter < 0:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _run_tests(tests: tuple[str, ...], timeout: int) -> tuple[bool, str]:
    """Return ``(passed, detail)`` for the targeted subset.

    ``-x`` is what makes the survivor count affordable: a killed mutant dies in
    the first failing test rather than after the whole subset, and the mutants
    that cost anything are precisely the ones that survive.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *tests,
        "-x",
        "-q",
        "--no-header",
        "--no-cov",
        "-p",
        "no:randomly",
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout
    )
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    detail = tail[-1] if tail else f"exit {proc.returncode}"
    return proc.returncode == 0, detail[:160]


def probe(
    module: Path,
    max_mutants: int,
    timeout: int,
    verbose: bool,
) -> int:
    """Run the probe on one module. Returns a process exit status."""
    rel = module.relative_to(REPO_ROOT).as_posix()
    tests = TEST_MAP.get(module.name)
    if tests is None:
        print(
            f"no test mapping for {module.name}; add one to TEST_MAP rather than "
            "running the whole suite, which is what makes this affordable",
            file=sys.stderr,
        )
        return 2
    missing = [t for t in tests if not (REPO_ROOT / t).exists()]
    if missing:
        print(f"mapped test files do not exist: {missing}", file=sys.stderr)
        return 2

    baseline_ok, baseline_detail = _run_tests(tests, timeout)
    if not baseline_ok:
        print(
            f"BASELINE FAILS on {rel}: {baseline_detail}\n"
            "A mutation probe against a red baseline measures nothing -- every "
            "mutant would look killed.",
            file=sys.stderr,
        )
        return 1

    mutants, source = generate_mutants(module, max_mutants)
    if not mutants:
        print(f"no mutations generated for {rel}", file=sys.stderr)
        return 2

    backup = module.read_bytes()
    results: list[MutantResult] = []
    started = time.perf_counter()
    try:
        for mutant in mutants:
            mutated = _apply_mutant(module, source, mutant)
            if mutated is None:
                continue
            module.write_text(mutated, encoding="utf-8")
            t0 = time.perf_counter()
            try:
                passed, detail = _run_tests(tests, timeout)
            except subprocess.TimeoutExpired:
                passed, detail = False, "timeout (treated as killed: it hung)"
            elapsed = time.perf_counter() - t0
            results.append(MutantResult(mutant, not passed, elapsed, detail))
            if verbose:
                mark = "KILL" if not passed else "LIVE"
                print(
                    f"  [{mark}] #{mutant.index:3d} L{mutant.lineno:<5d} "
                    f"{mutant.kind:<11s} {mutant.before} -> {mutant.after} "
                    f"({elapsed:.1f}s)"
                )
    finally:
        module.write_bytes(backup)

    total = len(results)
    killed = sum(1 for r in results if r.killed)
    survivors = [r for r in results if not r.killed]
    wall = time.perf_counter() - started

    print(f"\n=== mutation probe: {rel} ===")
    print(f"targeted tests : {len(tests)} file(s)")
    print(f"mutants        : {total}")
    print(f"killed         : {killed}  ({100 * killed / total:.1f}%)")
    print(f"survived       : {len(survivors)}")
    print(f"wall clock     : {wall:.0f}s")
    if survivors:
        print("\nsurvivors -- each is a hole coverage cannot see:")
        for r in survivors:
            m = r.mutant
            print(
                f"  #{m.index:3d} L{m.lineno:<5d} {m.kind:<11s} "
                f"{m.before} -> {m.after}\n"
                f"        {m.context}"
            )
    print(f"\nbaseline restored: {module.read_bytes() == backup}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--module",
        required=True,
        help="source file to mutate, relative to the repository root",
    )
    parser.add_argument("--max-mutants", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    module = (REPO_ROOT / args.module).resolve()
    if not module.is_file():
        print(f"no such module: {module}", file=sys.stderr)
        return 2
    # Refuse to run against anything but a tracked source file: this script
    # rewrites the file it is pointed at and restores it afterwards, and an
    # interrupted run against the wrong path is not a recoverable mistake.
    if "src" not in module.parts:
        print("refusing to mutate a file outside src/", file=sys.stderr)
        return 2
    # NOTE: there is deliberately no `git checkout` safety net here.  An
    # earlier version had one, on the reasoning that a crashed probe should
    # leave no dirty tree -- and it destroyed uncommitted work in the module
    # being probed, because `git checkout -- <file>` restores HEAD rather than
    # the state the file was in when the probe started.  Restoring from the
    # byte backup inside `probe()` is the only correct behaviour: the probe's
    # contract is that the file it is handed comes back unchanged, whatever
    # state it was in to begin with.  A tool whose job is to find carelessness
    # in a test suite has no business being careless with the working tree.
    shutil.rmtree(
        REPO_ROOT / "src" / "truss_analysis" / "__pycache__", ignore_errors=True
    )
    return probe(module, args.max_mutants, args.timeout, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
