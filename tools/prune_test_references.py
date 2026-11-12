#!/usr/bin/env python3
"""Drop deleted tests from the lists that name them, and follow renames.

The test-suite reduction (2026-11-12) deletes, merges and renames test files
and functions.  Several files keep lists keyed by test file or by (file,
function): the hygiene ratchets in ``tests/test_test_suite_hygiene.py``, the
CI classification tables in ``tests/test_ci_suite_classification.py``, and
the module lists in ``tests/conftest.py`` and ``tests/qt/conftest.py``.  A
ratchet list may only shrink, so a batch that deletes a listed test has to
remove its entry in the same commit, and a renamed test has to be renamed
there too.  This tool does that edit, and nothing else.

WHAT IT TOUCHES.  Only string literals inside the list, set, tuple or dict
literal assigned to a module-level UPPER_CASE name.  A string is a test
reference when it names a ``test_*.py`` file (bare, ``qt/...`` or
``tests/...``) or a ``file::function`` node id, or when it is a function
name inside the set that a test-file key maps to.  Docstrings, comments,
synthetic examples inside functions and everything else are left alone.

THREE WAYS TO SAY WHAT CHANGED:

  --deleted REF      a test file or ``file::function`` that is gone;
  --renamed OLD=NEW  a test file or ``file::function`` that moved;
  --stale            find every reference whose file or function no longer
                     exists under ``tests/`` and treat it as deleted.

An entry for a deleted reference is removed together with its trailing comma
and the comment lines directly above it.  A file key whose set of function
names becomes empty is removed whole.  Without ``--apply`` nothing is
written; the planned edits are printed.  With it, each edited file must
still parse, or nothing is written and the tool exits 1.

Workflow files under ``.github/workflows`` are scanned for ``--deselect``
and ``--exclude-file`` references to the same tests and REPORTED, not
edited: those lines sit inside shell scripts a person should read.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_TARGETS = (
    "tests/test_test_suite_hygiene.py",
    "tests/conftest.py",
    "tests/qt/conftest.py",
    "tests/test_ci_suite_classification.py",
)
_FILE_REF = re.compile(r"^(?:tests/)?(?:[\w.-]+/)*test_\w+\.py$")
_NODE_REF = re.compile(r"^(?:tests/)?(?:[\w.-]+/)*test_\w+\.py::[\w:\[\]-]+$")
_FUNC_NAME = re.compile(r"^test_\w+$")
_WORKFLOW_REF = re.compile(r"(?:--deselect[= ]|--exclude-file[= ])(\S+?)[\"'\s\\]")


@dataclass
class Reference:
    """One string literal that names a test, and where it sits."""

    path: Path
    value: str
    node: ast.Constant
    container: ast.AST
    owner_file: str | None = None
    removable: ast.AST | None = None
    dict_entry: tuple[ast.AST, ast.AST] | None = None


@dataclass
class Plan:
    """Edits for one file: references to remove and literals to rewrite."""

    path: Path
    remove: list[Reference] = field(default_factory=list)
    rename: list[tuple[Reference, str]] = field(default_factory=list)


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Ids of every docstring constant in ``tree``."""
    found = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant):
            found.add(id(body[0].value))
    return found


def _is_test_ref(value: str) -> bool:
    """Whether a string names a test file or a test node id."""
    return bool(_FILE_REF.match(value) or _NODE_REF.match(value))


def find_references(path: Path) -> list[Reference]:
    """Every test reference in the module-level UPPER_CASE literals of ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_ids(tree)
    found: list[Reference] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id.isupper() for t in targets):
            continue
        _collect(path, value, docstrings, found, owner_file=None)
    return found


def _collect(path, node, docstrings, found, *, owner_file):
    """Walk one literal, appending every test reference inside it."""
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            key_file = None
            if isinstance(key, ast.Constant) and isinstance(key.value, str) \
                    and id(key) not in docstrings and _is_test_ref(key.value):
                found.append(Reference(path, key.value, key, node,
                                       removable=key, dict_entry=(key, value)))
                key_file = key.value
            _collect(path, value, docstrings, found, owner_file=key_file)
    elif isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str) \
                    and id(element) not in docstrings:
                if _is_test_ref(element.value) or (
                        owner_file is not None and _FUNC_NAME.match(element.value)):
                    found.append(Reference(path, element.value, element, node,
                                           owner_file=owner_file, removable=element))
            else:
                _collect(path, element, docstrings, found, owner_file=owner_file)
    elif isinstance(node, ast.Call):
        for argument in node.args:
            _collect(path, argument, docstrings, found, owner_file=owner_file)


def _candidates(root: Path, holder: Path, value: str) -> list[Path]:
    """Where a reference's file may live: repo root, tests/, or beside ``holder``."""
    relative = value.split("::", 1)[0]
    return [root / relative, root / "tests" / relative, holder.parent / relative]


def _resolve(root: Path, holder: Path, value: str) -> Path | None:
    """The existing file a reference names, or None."""
    for candidate in _candidates(root, holder, value):
        if candidate.is_file():
            return candidate
    return None


def _defines(test_file: Path, name: str) -> bool:
    """Whether ``test_file`` defines ``name``: a bare name anywhere (a method
    counts), or a ``Class::method`` path."""
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    parts = name.split("::")
    if len(parts) == 1:
        bare = parts[0].split("[", 1)[0]
        return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == bare for node in ast.walk(tree))
    scope: Iterable[ast.AST] = tree.body
    for part in parts:
        part = part.split("[", 1)[0]
        match = next((node for node in scope
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                           ast.ClassDef)) and node.name == part), None)
        if match is None:
            return False
        scope = getattr(match, "body", [])
    return True


def is_stale(root: Path, reference: Reference) -> bool:
    """Whether the file or function a reference names no longer exists."""
    holder = reference.path
    if reference.owner_file is not None and _FUNC_NAME.match(reference.value):
        target = _resolve(root, holder, reference.owner_file)
        return target is not None and not _defines(target, reference.value)
    target = _resolve(root, holder, reference.value)
    if target is None:
        return True
    if "::" in reference.value:
        return not _defines(target, reference.value.split("::", 1)[1])
    return False


def _same(root: Path, holder: Path, ref_value: str, given: str) -> bool:
    """Whether a reference string and a user-given reference name one test."""
    if ref_value == given:
        return True
    ref_file, _, ref_func = ref_value.partition("::")
    giv_file, _, giv_func = given.partition("::")
    if ref_func != giv_func:
        return False
    ref_paths = {p.resolve() for p in _candidates(root, holder, ref_file)}
    giv_paths = {(root / giv_file).resolve(), (root / "tests" / giv_file).resolve()}
    return bool(ref_paths & giv_paths)


def plan_edits(
    root: Path, targets: Sequence[Path], *, deleted: Sequence[str] = (),
    renamed: dict[str, str] | None = None, stale: bool = False,
) -> list[Plan]:
    """Work out which references to drop or rename in each target file."""
    renamed = dict(renamed or {})
    plans = []
    for target in targets:
        plan = Plan(target)
        for reference in find_references(target):
            holder = reference.path
            func_ref = (reference.owner_file is not None
                        and _FUNC_NAME.match(reference.value))
            full = (f"{reference.owner_file}::{reference.value}"
                    if func_ref else reference.value)
            new = next((n for o, n in renamed.items() if _same(root, holder, full, o)), None)
            if new is None and "::" in full and not func_ref:
                file_part, _, func_part = full.partition("::")
                new = next((f"{n}::{func_part}" for o, n in renamed.items()
                            if "::" not in o and _same(root, holder, file_part, o)), None)
            if new is not None:
                replacement = new.split("::", 1)[1] if func_ref else _respell(
                    reference.value, new)
                plan.rename.append((reference, replacement))
            elif any(_same(root, holder, full, d) for d in deleted) or (
                    stale and is_stale(root, reference)):
                plan.remove.append(reference)
        if plan.remove or plan.rename:
            plans.append(plan)
    return plans


def _respell(old: str, new: str) -> str:
    """Spell a renamed reference the way the old one was spelled."""
    new_file, _, new_func = new.partition("::")
    new_file = new_file[len("tests/"):] if new_file.startswith("tests/") else new_file
    old_file = old.split("::", 1)[0]
    if old_file.startswith("tests/"):
        new_file = "tests/" + new_file
    elif "/" not in old_file and "/" in new_file and not new_file.startswith("qt/"):
        new_file = new_file.rsplit("/", 1)[1]
    elif "/" not in old_file:
        new_file = new_file.rsplit("/", 1)[-1]
    return new_file + (f"::{new_func}" if new_func else "")


def _offset(lines: list[str], line: int, col: int) -> int:
    """Character offset of an AST (line, byte column) position."""
    return sum(len(text) for text in lines[:line - 1]) + len(
        lines[line - 1].encode("utf-8")[:col].decode("utf-8", errors="ignore"))


def _span(source: str, lines: list[str], start: ast.AST, end: ast.AST) -> tuple[int, int]:
    """Character span of an entry, its trailing comma and any whole lines."""
    begin = _offset(lines, start.lineno, start.col_offset)
    finish = _offset(lines, end.end_lineno, end.end_col_offset)
    rest = re.match(r"[ \t]*,?[ \t]*(#[^\n]*)?", source[finish:])
    finish += rest.end() if rest else 0
    line_start = source.rfind("\n", 0, begin) + 1
    if source[line_start:begin].strip() == "" and source[finish:finish + 1] in ("\n", ""):
        begin = line_start
        finish = min(finish + 1, len(source))
        while True:
            previous_end = begin - 1
            if previous_end <= 0:
                break
            previous_start = source.rfind("\n", 0, previous_end) + 1
            if source[previous_start:previous_end].lstrip().startswith("#"):
                begin = previous_start
            else:
                break
    return begin, finish


def apply_plan(plan: Plan) -> str:
    """Return the edited source for one plan; raises SyntaxError if it breaks."""
    source = plan.path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []
    for reference in plan.remove:
        if reference.dict_entry is not None:
            key, value = reference.dict_entry
            edits.append((*_span(source, lines, key, value), ""))
        else:
            edits.append((*_span(source, lines, reference.node, reference.node), ""))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(value, (ast.Set, ast.List, ast.Tuple)) and value.elts and all(
                        isinstance(e, ast.Constant) and any(
                            r.node.lineno == e.lineno and r.node.col_offset == e.col_offset
                            for r in plan.remove) for e in value.elts):
                    edits = [e for e in edits if not (
                        _offset(lines, value.lineno, value.col_offset) <= e[0]
                        < _offset(lines, value.end_lineno, value.end_col_offset))]
                    edits.append((*_span(source, lines, key, value), ""))
    for reference, replacement in plan.rename:
        node = reference.node
        start = _offset(lines, node.lineno, node.col_offset)
        end = _offset(lines, node.end_lineno, node.end_col_offset)
        quote = source[start]
        edits.append((start, end, f"{quote}{replacement}{quote}"))
    edited = source
    for start, end, text in sorted(set(edits), key=lambda e: -e[0]):
        edited = edited[:start] + text + edited[end:]
    ast.parse(edited)
    return edited


def workflow_mentions(root: Path, refs: Iterable[str]) -> list[str]:
    """``--deselect``/``--exclude-file`` lines in workflows naming ``refs``."""
    wanted = set(refs)
    notes = []
    for workflow in sorted((root / ".github" / "workflows").glob("*.yml")):
        for number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), 1):
            for match in _WORKFLOW_REF.finditer(line + " "):
                named = match.group(1)
                if named in wanted or named.split("::", 1)[0] in wanted:
                    notes.append(f"{workflow.relative_to(root)}:{number}: {line.strip()}")
    return notes


def build_parser() -> argparse.ArgumentParser:
    """The command line described in the module docstring."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--target", action="append", type=Path,
                        help="file holding test lists (default: the four named above)")
    parser.add_argument("--deleted", action="append", default=[], metavar="REF")
    parser.add_argument("--renamed", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--stale", action="store_true",
                        help="also drop every reference to a test that no longer exists")
    parser.add_argument("--apply", action="store_true", help="write the edits")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the plan; with --apply write it. Exit 1 if an edit would not parse."""
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    targets = [root / t for t in (args.target or DEFAULT_TARGETS)]
    targets = [t for t in targets if t.is_file()]
    renamed = {}
    for pair in args.renamed:
        old, sep, new = pair.partition("=")
        if not sep or not old or not new:
            print(f"--renamed needs OLD=NEW, got {pair!r}", file=sys.stderr)
            return 2
        renamed[old] = new
    plans = plan_edits(root, targets, deleted=args.deleted, renamed=renamed,
                       stale=args.stale)
    edited: dict[Path, str] = {}
    for plan in plans:
        name = plan.path.relative_to(root)
        for reference in plan.remove:
            label = (f"{reference.owner_file}::{reference.value}"
                     if reference.owner_file else reference.value)
            print(f"REMOVE {name}:{reference.node.lineno}: {label}")
        for reference, replacement in plan.rename:
            print(f"RENAME {name}:{reference.node.lineno}: {reference.value} -> {replacement}")
        try:
            edited[plan.path] = apply_plan(plan)
        except SyntaxError as exc:
            print(f"REFUSED {name}: the edit would not parse ({exc})", file=sys.stderr)
            return 1
    for note in workflow_mentions(root, list(args.deleted) + list(renamed)):
        print(f"WORKFLOW (edit by hand): {note}")
    if not plans:
        print("nothing to change")
    if args.apply:
        for path, text in edited.items():
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
