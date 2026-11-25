"""Arrival contract of the public callables item 1017 adds; stdlib only.

The Database Browser merges tables across projects: the headless
spacr.plate_measurements.merge_projects and save_merge, and the window that
DbBrowserScreen.open_merge shows. They arrive through
tests/data/release_contracts/1017_merge_projects_api_arrivals_2026-11-24.json,
with the four nested job helpers of the window, each body bound to its
source docstring (read with ``ast``, nothing imported
from spacr).
"""
import ast
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIRTH = ROOT / "tests/data/release_contracts/1017_merge_projects_api_arrivals_2026-11-24.json"
PUBLIC = frozenset({
    "spacr.plate_measurements.merge_projects",
    "spacr.plate_measurements.save_merge",
    "spacr.qt.screens.db_browser.DbBrowserScreen.open_merge",
})
#: Nested job helpers of the Merge tables window, documented on the API pages
#: but not public callables.
NESTED = frozenset(
    f"spacr.qt.screens.db_browser._MergeDialog.{method}.{helper}"
    for method in ("preview", "save") for helper in ("_job", "_done"))
ARRIVALS = PUBLIC | NESTED


def _source_docstring(symbol):
    """The cleaned docstring of a function or method, read from its source file."""
    parts = symbol.split(".")
    for split in range(len(parts), 1, -1):
        path = ROOT.joinpath(*parts[:split]).with_suffix(".py")
        if path.is_file():
            break
    node = ast.parse(path.read_text(encoding="utf-8"))
    for name in parts[split:]:
        scope = node.body if isinstance(node, (ast.Module, ast.ClassDef)) else list(ast.walk(node))
        node = next(item for item in scope
                    if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                    and item.name == name)
    return inspect.cleandoc(ast.get_docstring(node, clean=False))


def _current_1017_api(existing=None):
    """The new public callables bound to their source docstrings.

    :param existing: earlier arrivals; none of them may be one of these.
    """
    entries = json.loads(BIRTH.read_text(encoding="utf-8"))
    assert set(entries) == ARRIVALS
    for symbol, body in entries.items():
        assert body == _source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_1017_arrivals(callables):
    """Remove only the exact new public callables from the frozen boundary."""
    entries = _current_1017_api()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == PUBLIC, sorted(found)
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]
