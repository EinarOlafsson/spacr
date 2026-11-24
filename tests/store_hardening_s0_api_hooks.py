"""Light API-arrival hooks of store hardening step S0.

Imports no ``spacr`` module: the source docstrings are read with ``ast``, so
the docstring-correctness and API-extractor tests can use these hooks without
loading the package.
"""
import os


S0_BIRTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "release_contracts",
                        "s0_store_hardening_api_arrivals_2026-11-23.json")
S0_ARRIVALS = frozenset({
    "spacr.store.Store.columns", "spacr.store.Store.count",
    "spacr.store.Store.page", "spacr.power_ops.nominal_calibration",
    "spacr.power_ops.write_ops_screen"})


def _s0_source_docstring(symbol):
    """The docstring of one arrival read from its source file, so no module is imported."""
    import ast
    import inspect

    parts = symbol.split(".")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "spacr", parts[1] + ".py")
    with open(path, encoding="utf-8") as handle:
        scope = ast.parse(handle.read())
    for name in parts[2:]:
        scope = next(node for node in scope.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and node.name == name)
    return inspect.cleandoc(ast.get_docstring(scope, clean=False))


def _s0_birth_arrivals(existing=None):
    """The S0 public symbols, bound to their source docstrings."""
    import json

    with open(S0_BIRTH, encoding="utf-8") as handle:
        entries = json.load(handle)
    assert set(entries) == S0_ARRIVALS
    for symbol, body in entries.items():
        assert body == _s0_source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_s0_arrivals(callables):
    """Remove only the S0 exact new callables from the frozen boundary."""
    entries = _s0_birth_arrivals()
    found = {item.symbol: item for item in callables
             if item.symbol in entries}
    assert set(found) == S0_ARRIVALS
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]
