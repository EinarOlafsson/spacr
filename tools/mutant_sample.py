#!/usr/bin/env python3
"""Plant a fixed random sample of simple mutants in one module and run tests.

The test-suite reduction (2026-11-12) lands a batch only if the new tests
catch at least as many planted bugs as the tests they replace.  This tool
supplies those bugs with no new dependency.

THE MUTANTS are four simple kinds, found by walking the module's AST:

  compare   one comparison operator flipped (``<`` to ``>=``, ``==`` to
            ``!=``, ``in`` to ``not in``, ``is`` to ``is not``, ...);
  constant  one numeric literal moved by one (``0`` to ``1``, ``2.5`` to
            ``3.5``); docstrings and other strings are left alone;
  return    one ``return <value>`` replaced by ``return None``;
  negate    the condition of one ``if``, ``while`` or conditional
            expression negated, or one ``True``/``False`` literal swapped.

Every site has a stable id (``kind:line:col:n``).  ``--seed`` and ``--count``
pick the sample with :class:`random.Random`, so two runs with the same seed
plant the same mutants, which is what lets an old and a new test set be
compared on identical bugs.  ``--lines`` limits the sites to chosen line
numbers, for example the lines the old tests cover.

THE WORKING TREE IS NEVER EDITED.  Each mutant's source is written to a
temporary directory, and pytest runs in a subprocess with a small plugin
that installs an import hook: when the target module is imported, the hook
compiles the mutated source in its place, under the original file name.
The plugin records that the mutant was really imported; a mutant whose
module the tests never imported is reported as ``not_imported``, not as a
survivor.  Code that a test runs in a fresh subprocess does not see the
mutant.

OUTCOMES per mutant and per test set: ``killed`` (pytest exit 1),
``survived`` (exit 0), ``timeout`` (counted as killed), ``not_imported``
and ``error`` (any other exit, such as a collection error).  Before any
mutant runs, every test set must pass on the unmutated module; otherwise
the comparison would mean nothing and the tool exits 2.

Example::

    python tools/mutant_sample.py --module spacr/regex_infer.py --seed 7 \\
        --count 20 --tests tests/test_regex_infer.py \\
        --compare-tests tests/test_regex_infer_subset.py \\
        --json-out mutants.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

KINDS = ("compare", "constant", "return", "negate")

FLIPPED_COMPARISON = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE, ast.GtE: ast.Lt,
    ast.Gt: ast.LtE, ast.LtE: ast.Gt,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}

PLUGIN_NAME = "spacr_mutant_import_hook"

PLUGIN_SOURCE = '''"""Pytest plugin written by tools/mutant_sample.py: load one mutated module."""
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

_NAME = os.environ["SPACR_MUTANT_MODULE"]
_ORIGIN = os.environ["SPACR_MUTANT_ORIGIN"]
_SOURCE = Path(os.environ["SPACR_MUTANT_SOURCE"]).read_text(encoding="utf-8")
_MARKER = Path(os.environ["SPACR_MUTANT_MARKER"])


class _MutantLoader(importlib.abc.Loader):
    """Execute the mutated source in place of the module's own file."""

    def create_module(self, spec):
        """Let the import system create an ordinary module object."""
        return None

    def exec_module(self, module):
        """Compile the mutant under the original file name and run it."""
        _MARKER.write_text(_NAME, encoding="utf-8")
        code = compile(_SOURCE, _ORIGIN, "exec")
        exec(code, module.__dict__)


class _MutantFinder(importlib.abc.MetaPathFinder):
    """Answer for exactly one module name and no other."""

    def find_spec(self, fullname, path=None, target=None):
        """Return the mutant's spec for the target name, else None."""
        if fullname != _NAME:
            return None
        package = os.path.basename(_ORIGIN) == "__init__.py"
        return importlib.util.spec_from_file_location(
            fullname, _ORIGIN, loader=_MutantLoader(),
            submodule_search_locations=(
                [os.path.dirname(_ORIGIN)] if package else None),
        )


sys.modules.pop(_NAME, None)
sys.meta_path.insert(0, _MutantFinder())
'''


@dataclass(frozen=True)
class Mutant:
    """One planted bug: where it is, what kind, and what it changes."""

    id: str
    kind: str
    line: int
    col: int
    description: str


def _is_docstring(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """Whether ``node`` is the first statement's constant of a body."""
    parent = parents.get(id(node))
    if not isinstance(parent, ast.Expr):
        return False
    owner = parents.get(id(parent))
    body = getattr(owner, "body", None)
    return bool(body) and body[0] is parent


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    """Map each node's id to its parent node."""
    found: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            found[id(child)] = node
    return found


def _sites(tree: ast.AST) -> list[tuple[str, ast.AST, int, str]]:
    """Every mutable site as (kind, node, index within node, description)."""
    parents = _parents(tree)
    sites: list[tuple[str, ast.AST, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for index, op in enumerate(node.ops):
                flipped = FLIPPED_COMPARISON.get(type(op))
                if flipped is not None:
                    sites.append((
                        "compare", node, index,
                        f"{type(op).__name__} -> {flipped.__name__}",
                    ))
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                sites.append(("negate", node, 0, f"{value} -> {not value}"))
            elif isinstance(value, (int, float)) and not isinstance(value, complex):
                if not _is_docstring(node, parents):
                    sites.append(("constant", node, 0, f"{value!r} -> {value + 1!r}"))
        elif isinstance(node, ast.Return):
            value = node.value
            if value is not None and not (
                    isinstance(value, ast.Constant) and value.value is None):
                sites.append(("return", node, 0, "return value -> None"))
        elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
            sites.append(("negate", node, 0, f"{type(node).__name__} condition negated"))
    return sites


def _site_id(kind: str, node: ast.AST, seen: dict[tuple, int]) -> str:
    """``kind:line:col:n``, where n counts earlier sites at the same place."""
    key = (kind, node.lineno, node.col_offset)
    number = seen.get(key, 0)
    seen[key] = number + 1
    return f"{kind}:{node.lineno}:{node.col_offset}:{number}"


def _ordered(tree: ast.AST) -> list[tuple[str, ast.AST, int, str]]:
    """Sites in source order, so ids are stable for the same source."""
    return sorted(
        _sites(tree),
        key=lambda site: (site[1].lineno, site[1].col_offset, site[0], site[2]))


def enumerate_mutants(source: str, *, lines: Iterable[int] | None = None) -> list[Mutant]:
    """List every mutant of ``source`` in a stable order, optionally by line."""
    tree = ast.parse(source)
    keep = set(lines) if lines is not None else None
    seen: dict[tuple, int] = {}
    mutants = []
    for kind, node, _index, description in _ordered(tree):
        mutant_id = _site_id(kind, node, seen)
        if keep is not None and node.lineno not in keep:
            continue
        mutants.append(Mutant(mutant_id, kind, node.lineno, node.col_offset, description))
    return mutants


def sample_mutants(
    mutants: Sequence[Mutant], *, seed: int, count: int,
    kinds: Iterable[str] = KINDS,
) -> list[Mutant]:
    """A fixed random sample: the same seed always picks the same mutants."""
    allowed = set(kinds)
    pool = [mutant for mutant in mutants if mutant.kind in allowed]
    if count >= len(pool):
        return list(pool)
    chosen = random.Random(seed).sample(range(len(pool)), count)
    return [pool[index] for index in sorted(chosen)]


class _Apply(ast.NodeTransformer):
    """Rewrite exactly the one node a mutant id names."""

    def __init__(self, mutant: Mutant):
        self.mutant = mutant
        self.seen: dict[tuple, int] = {}
        self.targets: dict[int, tuple[str, int]] = {}
        self.applied = False

    def plan(self, tree: ast.AST) -> None:
        """Find the node (and operator index) the mutant id names."""
        for kind, node, index, _description in _ordered(tree):
            if _site_id(kind, node, self.seen) == self.mutant.id:
                self.targets[id(node)] = (kind, index)

    def visit(self, node: ast.AST) -> ast.AST:
        """Rewrite the planned node; leave every other node as it is."""
        target = self.targets.get(id(node))
        node = self.generic_visit(node)
        if target is None:
            return node
        kind, index = target
        self.applied = True
        if kind == "compare":
            node.ops[index] = FLIPPED_COMPARISON[type(node.ops[index])]()
        elif kind == "constant":
            node.value = node.value + 1
        elif kind == "return":
            node.value = ast.Constant(value=None)
        elif isinstance(node, ast.Constant):
            node.value = not node.value
        else:
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node


def apply_mutant(source: str, mutant: Mutant) -> str:
    """Return ``source`` with ``mutant`` planted; raises if it is not there."""
    tree = ast.parse(source)
    transformer = _Apply(mutant)
    transformer.plan(tree)
    tree = transformer.visit(tree)
    if not transformer.applied:
        raise ValueError(f"mutant {mutant.id} does not exist in this source")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def module_name(path: Path, root: Path) -> str:
    """Dotted import name of ``path`` relative to ``root``."""
    parts = list(path.resolve().relative_to(root.resolve()).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _run_tests(
    tests: Sequence[str], *, name: str, origin: Path, source: str,
    workdir: Path, cwd: Path, timeout: float, pytest_args: Sequence[str],
) -> dict:
    """Run one test set against ``source`` loaded as module ``name``."""
    source_file = workdir / "mutant_source.py"
    marker = workdir / "imported.marker"
    source_file.write_text(source, encoding="utf-8")
    marker.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update({
        "SPACR_MUTANT_MODULE": name,
        "SPACR_MUTANT_ORIGIN": str(origin),
        "SPACR_MUTANT_SOURCE": str(source_file),
        "SPACR_MUTANT_MARKER": str(marker),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env["PYTHONPATH"] = os.pathsep.join(
        [str(workdir), str(cwd)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    command = [
        sys.executable, "-m", "pytest", "-p", PLUGIN_NAME, "-x", "-q",
        "-p", "no:cacheprovider", *pytest_args, *tests,
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"outcome": "timeout", "exit": None,
                "seconds": round(time.monotonic() - started, 2), "tail": ""}
    if not marker.exists():
        outcome = "not_imported"
    elif result.returncode == 0:
        outcome = "survived"
    elif result.returncode == 1:
        outcome = "killed"
    else:
        outcome = "error"
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-5:])
    return {"outcome": outcome, "exit": result.returncode,
            "seconds": round(time.monotonic() - started, 2), "tail": tail}


def run_sample(
    *, module: Path, root: Path, test_sets: dict[str, Sequence[str]],
    seed: int, count: int, kinds: Iterable[str] = KINDS,
    lines: Iterable[int] | None = None, timeout: float = 600.0,
    pytest_args: Sequence[str] = (), cwd: Path | None = None,
) -> dict:
    """Plant the sample in ``module`` and run every test set on every mutant."""
    module = module.resolve()
    root = root.resolve()
    cwd = (cwd or root).resolve()
    source = module.read_text(encoding="utf-8")
    name = module_name(module, root)
    mutants = sample_mutants(
        enumerate_mutants(source, lines=lines), seed=seed, count=count, kinds=kinds)
    report = {
        "module": str(module.relative_to(root)), "import_name": name,
        "seed": seed, "count": count, "kinds": sorted(set(kinds)),
        "test_sets": {label: list(tests) for label, tests in test_sets.items()},
        "baseline": {}, "mutants": [], "summary": {},
    }
    with tempfile.TemporaryDirectory(prefix="spacr-mutants-") as scratch:
        workdir = Path(scratch)
        (workdir / f"{PLUGIN_NAME}.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
        for label, tests in test_sets.items():
            report["baseline"][label] = _run_tests(
                tests, name=name, origin=module, source=source, workdir=workdir,
                cwd=cwd, timeout=timeout, pytest_args=pytest_args)
        if any(run["outcome"] != "survived" for run in report["baseline"].values()):
            report["status"] = "baseline_failed"
            return report
        for mutant in mutants:
            mutated = apply_mutant(source, mutant)
            row = asdict(mutant)
            row["results"] = {
                label: _run_tests(
                    tests, name=name, origin=module, source=mutated,
                    workdir=workdir, cwd=cwd, timeout=timeout,
                    pytest_args=pytest_args)
                for label, tests in test_sets.items()
            }
            report["mutants"].append(row)
    for label in test_sets:
        outcomes = [row["results"][label]["outcome"] for row in report["mutants"]]
        report["summary"][label] = {
            "killed": sum(o in ("killed", "timeout") for o in outcomes),
            "survived": outcomes.count("survived"),
            "not_imported": outcomes.count("not_imported"),
            "error": outcomes.count("error"),
            "total": len(outcomes),
        }
    report["status"] = "ok"
    return report


def render_text(report: dict) -> str:
    """A table of mutants by test set, then the per-set totals."""
    labels = list(report["test_sets"])
    lines = [
        f"Mutant sample of {report['module']} (seed {report['seed']}, "
        f"{len(report['mutants'])} mutants, kinds {', '.join(report['kinds'])})",
    ]
    if report.get("status") == "baseline_failed":
        for label, run in report["baseline"].items():
            lines.append(f"BASELINE {label}: {run['outcome']} (exit {run['exit']})")
            lines.extend(f"    {line}" for line in run["tail"].splitlines())
        lines.append("The tests must pass on the unmutated module; nothing was compared.")
        return "\n".join(lines) + "\n"
    lines.append("id | change | " + " | ".join(labels))
    for row in report["mutants"]:
        lines.append(
            f"{row['id']} | {row['description']} | "
            + " | ".join(row["results"][label]["outcome"] for label in labels))
    for label in labels:
        summary = report["summary"][label]
        lines.append(
            f"{label}: killed {summary['killed']} of {summary['total']}, "
            f"survived {summary['survived']}, not imported "
            f"{summary['not_imported']}, errors {summary['error']}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """The command line described in the module docstring."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", type=Path, required=True,
                        help="source file to mutate, e.g. spacr/regex_infer.py")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="directory the module's import name is relative to")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=20,
                        help="how many mutants to sample (all when fewer exist)")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--lines", type=Path,
                        help="text file of line numbers mutants may sit on")
    parser.add_argument("--tests", nargs="+", required=True,
                        help="the first test set (pytest node ids or files)")
    parser.add_argument("--compare-tests", nargs="+",
                        help="a second test set run on the same mutants")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="seconds per test-set run; a timeout counts as killed")
    parser.add_argument("--pytest-arg", action="append", default=[],
                        help="extra argument passed to pytest (repeatable)")
    parser.add_argument("--list", action="store_true",
                        help="print the sampled mutants and run nothing")
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; exit 2 when the tests fail on the unmutated module."""
    args = build_parser().parse_args(argv)
    lines = None
    if args.lines is not None:
        lines = [int(token) for token in args.lines.read_text(encoding="utf-8").split()]
    if args.list:
        source = args.module.read_text(encoding="utf-8")
        for mutant in sample_mutants(enumerate_mutants(source, lines=lines),
                                     seed=args.seed, count=args.count, kinds=args.kinds):
            print(f"{mutant.id}\t{mutant.description}")
        return 0
    test_sets = {"tests": args.tests}
    if args.compare_tests:
        test_sets["compare"] = args.compare_tests
    report = run_sample(
        module=args.module, root=args.root, test_sets=test_sets, seed=args.seed,
        count=args.count, kinds=args.kinds, lines=lines, timeout=args.timeout,
        pytest_args=args.pytest_arg)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render_text(report), end="")
    return 2 if report["status"] == "baseline_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
