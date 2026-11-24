"""Arrival contract of the measurements-path owner; stdlib only.

The crop_store Rule-43 fix made ``spacr.ports.measurements_db_path`` the one
place that spells ``measurements/measurements.db``. It arrives through
tests/data/release_contracts/cropstore_rule43_api_arrivals_2026-11-24.json,
its body bound to the source docstring (read with ``ast``, nothing imported
from spacr).
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIRTH = ROOT / "tests/data/release_contracts/cropstore_rule43_api_arrivals_2026-11-24.json"
PUBLIC = "spacr.ports.measurements_db_path"
ARRIVALS = frozenset({PUBLIC})


def _source_docstring(symbol):
    """The cleaned docstring of ``spacr.<module>.<function>``."""
    _, module, name = symbol.split(".")
    node = ast.parse((ROOT / "spacr" / (module + ".py")).read_text(encoding="utf-8"))
    found = next(item for item in node.body
                 if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_docstring(found)


def _current_cropstore_rule43_api(existing=None):
    """The new public symbol bound to its source docstring.

    :param existing: earlier arrivals; none of them may be this one.
    """
    entries = json.loads(BIRTH.read_text(encoding="utf-8"))
    assert set(entries) == ARRIVALS
    for symbol, body in entries.items():
        assert body == _source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_cropstore_rule43_arrivals(callables):
    """Remove only the exact new public callable from the frozen boundary."""
    entries = _current_cropstore_rule43_api()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == {PUBLIC}
    assert found[PUBLIC].docstring == entries[PUBLIC]
    return [item for item in callables if item.symbol not in entries]
