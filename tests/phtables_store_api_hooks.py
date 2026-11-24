"""Light API-arrival hooks for the shared measurement-table writer.

Imports no ``spacr`` module: the source docstrings are read with ``ast``, so
the docstring-correctness tests can use these hooks without loading the
package.
"""
import ast
import inspect
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The nested helper of the ph_tables SQLite writer, retired with it on
#: 2026-11-23; the frozen 871 arrival contract still names it.
RETIRED_SQLITE_HELPER = "spacr.ops_engine._ph_insert_rows.declared"


def _current_phtables_api(entries):
    """Drop only the retired SQLite helper from a frozen arrival contract."""
    return {key: value for key, value in entries.items()
            if key != RETIRED_SQLITE_HELPER}


#: The arrival contract of the shared measurement-table writer (2026-11-23):
#: spacr.store.writer, write_table and WellWriter with its methods, the
#: public writer Measure uses next.
STORE_BIRTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                           "release_contracts",
                           "phtables_store_api_arrivals_2026-11-23.json")
STORE_ARRIVALS = frozenset({
    "spacr.store.writer", "spacr.store.write_table", "spacr.store.WellWriter",
    "spacr.store.WellWriter.__enter__", "spacr.store.WellWriter.__exit__",
    "spacr.store.WellWriter.append", "spacr.store.WellWriter.finish",
    "spacr.store.WellWriter.abort"})


def _store_source_docstring(symbol):
    """The docstring of one ``spacr.store`` arrival, parsed from the source."""
    with open(os.path.join(ROOT, "spacr", "store.py"), encoding="utf-8") as handle:
        body = ast.parse(handle.read()).body
    node = None
    for part in symbol.split(".")[2:]:
        node = next(item for item in body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.ClassDef)) and item.name == part)
        body = node.body
    text = inspect.cleandoc(ast.get_docstring(node, clean=False))
    if isinstance(node, ast.ClassDef):
        # The extractor gives a class its docstring and its constructor's.
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                text += "\n" + inspect.cleandoc(
                    ast.get_docstring(item, clean=False))
    return text


def _phtables_store_birth_arrivals(existing=None):
    """The writer's new public symbols, bound to their source docstrings.

    :param existing: earlier arrivals; none of them may be one of these.
    """
    with open(STORE_BIRTH, encoding="utf-8") as handle:
        entries = json.load(handle)
    assert set(entries) == STORE_ARRIVALS
    for symbol, body in entries.items():
        assert body == _store_source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_phtables_store_arrivals(callables):
    """Remove only the writer's exact new callables from the frozen boundary."""
    entries = _phtables_store_birth_arrivals()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == {"spacr.store.writer", "spacr.store.write_table",
                          "spacr.store.WellWriter",
                          "spacr.store.WellWriter.append",
                          "spacr.store.WellWriter.finish",
                          "spacr.store.WellWriter.abort"}
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]
