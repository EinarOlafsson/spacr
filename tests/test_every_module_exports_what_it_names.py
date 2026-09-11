"""A name in ``__all__`` that was never written is invisible until a star import.

THE DEFECT THIS CATCHES, found in `spacr/qt/widgets/dose_response.py` by the
home session on 2026-09-10: `__all__` listed `checkerboard_from_frame` and
the function did not exist, so `from ... import *` raised AttributeError
while every test that imported by name passed. A hand-maintained export list
is the one place in a module that nothing executes.

STATIC FIRST, RUNTIME ONLY WHERE IT HAS TO BE. Parsing is free and covers
every module; importing 561 modules is not, and several pull in torch and
cellpose. A module that defines its own ``__getattr__`` is resolving names
lazily and cannot be judged by reading it, so those -- and only those -- are
imported and asked.
"""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "spacr"

#: Generated localization payloads. Data, not API, and enormous.
SKIP_PARTS = ("i18n_catalogs",)


def _modules_with_an_export_list():
    """Every module carrying a literal ``__all__``, with what it names."""
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in str(path) for part in SKIP_PARTS):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        exported = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__all__"
                    for t in node.targets):
                try:
                    exported = [str(v) for v in ast.literal_eval(node.value)]
                except (ValueError, TypeError):
                    exported = None
        if exported:
            found.append((path, tree, exported))
    return found


def _names_bound_in(tree) -> set:
    """Everything the module body binds: defs, classes, assignments, imports."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
    return bound


def _dotted(path: pathlib.Path) -> str:
    rel = path.relative_to(ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


MODULES = _modules_with_an_export_list()


def test_the_scan_actually_found_the_export_lists():
    """A check that silently matched nothing would pass for ever."""
    assert len(MODULES) > 20, (
        f"only {len(MODULES)} modules with __all__ were found; the scan is "
        "broken, not the package")


@pytest.mark.parametrize(
    "path,tree,exported",
    MODULES,
    ids=[_dotted(p) for p, _t, _e in MODULES])
def test_every_exported_name_exists(path, tree, exported):
    """Each name in ``__all__`` is bound in the module, or resolves lazily."""
    bound = _names_bound_in(tree)
    missing = [name for name in exported if name not in bound]
    if not missing:
        return

    # LAZY MODULES ARE ASKED RATHER THAN READ. A module-level `__getattr__`
    # means the name is produced on demand, which no amount of parsing can
    # see -- `spacr` and `spacr.qt.widgets` both do this deliberately, to
    # keep torch and PySide6 off the import path until something needs them.
    lazy = any(isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
               for node in tree.body)
    if not lazy:
        pytest.fail(
            f"{_dotted(path)} exports {missing} in __all__ and defines "
            f"neither -- `from {_dotted(path)} import *` would raise "
            f"AttributeError, and no test that imports by name can see it")

    module = importlib.import_module(_dotted(path))
    unresolved = []
    for name in missing:
        try:
            getattr(module, name)
        except Exception as error:                           # noqa: BLE001
            unresolved.append(f"{name} ({type(error).__name__})")
    assert not unresolved, (
        f"{_dotted(path)} exports {unresolved}, and its lazy __getattr__ "
        f"does not produce them either")
