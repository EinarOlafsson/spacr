"""Arrival contract of the public names landing L4 adds; stdlib only.

S4's spacr.measurement_store (Measure's and Make Masks' tables in the spaCR
store, with the locator rule store_folder) arrives through
tests/data/release_contracts/l4_api_arrivals_2026-11-24.json, and S7c's
spacr.store.read_frame through its own s7c_read_frame_api_arrivals_2026-11-24.json,
which no contract test applied; each body is bound to its source docstring
(read with ``ast``, nothing imported from spacr).

These are documented names of the API pages. spacr.measurement_store
declares ``__all__ = []`` and read_frame is not exported, so the callables
of the frozen inventory among them are S8's spacr.store.compare and
spacr.store.migrate_sqlite (both in ``spacr.store.__all__``), which the
inventory hook checks.
"""
import ast
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIRTHS = (ROOT / "tests/data/release_contracts/l4_api_arrivals_2026-11-24.json",
          ROOT / "tests/data/release_contracts/s7c_read_frame_api_arrivals_2026-11-24.json",
          ROOT / "tests/data/release_contracts/s8_store_cli_api_arrivals_2026-11-24.json")
MODULES = frozenset({"spacr.measurement_store"})
PUBLIC = frozenset("spacr.measurement_store." + name for name in (
    "append", "archive_tables", "compact", "connect", "discard_spool",
    "drop_tables", "ensure_store", "exists", "field_packet", "forget_sessions",
    "in_packet", "legacy_database", "legacy_shape", "measured_fields",
    "object_keys", "project_of", "read", "retire_legacy_database",
    "rewrite_table", "spool_field", "spool_of", "spooled_fields",
    "store_folder", "store_native", "tables", "write_rows"))
READERS = frozenset({"spacr.store.read_frame", "spacr.store.compare",
                     "spacr.store.migrate_sqlite"})
ARRIVALS = MODULES | PUBLIC | READERS
#: The arrivals that are callables of the frozen inventory.
INVENTORY = frozenset({"spacr.store.compare", "spacr.store.migrate_sqlite"})


def _source_docstring(symbol):
    """The cleaned docstring of ``spacr.<module>`` or ``spacr.<module>.<function>``."""
    _, module, *path = symbol.split(".")
    node = ast.parse((ROOT / "spacr" / (module + ".py")).read_text(encoding="utf-8"))
    for name in path:
        node = next(item for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name == name)
    return inspect.cleandoc(ast.get_docstring(node, clean=False))


def _current_l4_api(existing=None):
    """The new public names bound to their source docstrings.

    :param existing: earlier arrivals; none of them may be one of these.
    """
    entries = {}
    for birth in BIRTHS:
        found = json.loads(birth.read_text(encoding="utf-8"))
        assert not found.keys() & entries.keys()
        entries.update(found)
    assert set(entries) == ARRIVALS
    for symbol, body in entries.items():
        assert body == _source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_l4_arrivals(callables):
    """Remove only the exact new public callables from the frozen boundary."""
    entries = _current_l4_api()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == INVENTORY, sorted(found)
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]
