"""Arrival contract of the complex-membership importance callable; stdlib only.

The public callable ``spacr.ml.complex_membership_importance`` arrives through
tests/data/release_contracts/complex_shap_api_arrivals_2026-11-24.json, its body
bound to the source docstring (read with ``ast``, nothing imported from spacr).
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIRTH = ROOT / "tests/data/release_contracts/complex_shap_api_arrivals_2026-11-24.json"
PUBLIC = "spacr.ml.complex_membership_importance"
ARRIVALS = frozenset({PUBLIC, PUBLIC + ".run"})


def _source_docstring(symbol):
    """The cleaned docstring of ``spacr.ml.<function>[.<nested function>]``."""
    _, module, *path = symbol.split(".")
    node = ast.parse((ROOT / "spacr" / (module + ".py")).read_text(encoding="utf-8"))
    for name in path:
        scope = node.body if isinstance(node, ast.Module) else list(ast.walk(node))
        node = next(item for item in scope
                    if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_docstring(node)


def _current_complex_shap_api(existing=None):
    """The new public symbols bound to their source docstrings.

    :param existing: earlier arrivals; none of them may be one of these.
    """
    entries = json.loads(BIRTH.read_text(encoding="utf-8"))
    assert set(entries) == ARRIVALS
    for symbol, body in entries.items():
        assert body == _source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_complex_shap_arrivals(callables):
    """Remove only the exact new public callable from the frozen boundary.

    The nested helper ``run`` is a documented symbol of the API pages but
    not a public callable of the inventory.
    """
    entries = _current_complex_shap_api()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == {PUBLIC}
    assert found[PUBLIC].docstring == entries[PUBLIC]
    return [item for item in callables if item.symbol not in entries]
