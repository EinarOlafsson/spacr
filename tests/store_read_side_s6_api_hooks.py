"""Light API-arrival hooks of store read-side step S6.

Imports no ``spacr`` module: the source docstrings are read with ``ast``, so
the docstring-correctness and API-extractor tests can use these hooks without
loading the package.
"""
import os


S6_BIRTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "release_contracts",
                        "s6_store_browser_api_arrivals_2026-11-24.json")


def _s6_source_docstring(symbol):
    """The docstring of one arrival read from its source file, so no module is imported."""
    import ast
    import inspect

    parts = symbol.split(".")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for split in range(len(parts), 1, -1):
        path = os.path.join(root, *parts[:split]) + ".py"
        if os.path.isfile(path):
            break
    with open(path, encoding="utf-8") as handle:
        scope = ast.parse(handle.read())
    for name in parts[split:]:
        scope = next(node for node in scope.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and node.name == name)
    text = inspect.cleandoc(ast.get_docstring(scope, clean=False))
    if isinstance(scope, ast.ClassDef):
        init = next((node for node in scope.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "__init__"), None)
        if init is not None and ast.get_docstring(init, clean=False):
            text += "\n\n" + inspect.cleandoc(
                ast.get_docstring(init, clean=False))
    return text


def _s6_birth_arrivals(existing=None):
    """The S6 public symbols, bound to their source docstrings."""
    import json

    with open(S6_BIRTH, encoding="utf-8") as handle:
        entries = json.load(handle)
    for symbol, body in entries.items():
        assert body == _s6_source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_s6_arrivals(callables):
    """Remove only the S6 exact new callables from the frozen boundary.

    A class's inventory text joins its docstring and its constructor's with
    one newline where the extractor uses a blank line, so the two are
    compared with that junction folded.
    """
    entries = _s6_birth_arrivals()
    found = {item.symbol: item for item in callables
             if item.symbol in entries}
    assert set(found) == set(entries)
    for symbol, item in found.items():
        assert (item.docstring == entries[symbol]
                or item.docstring.replace("\n\n", "\n")
                == entries[symbol].replace("\n\n", "\n")), symbol
    return [item for item in callables if item.symbol not in entries]


S6_DOCS_BIRTH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "release_contracts",
    "s6_store_browser_docs_arrivals_2026-11-24.json")


def _s6_docs_arrivals(existing=None):
    """The S6 documented names that are not callables of the boundary.

    The new ``spacr.store_browse`` module and the nested job helpers of the
    Browser export and the Home evidence read, bound to their source text.
    """
    import json

    with open(S6_DOCS_BIRTH, encoding="utf-8") as handle:
        entries = json.load(handle)
    assert set(entries) == {
        "spacr.store_browse",
        "spacr.qt.screens.db_browser.DbBrowserScreen.export_rows._done",
        "spacr.qt.screens.db_browser.DbBrowserScreen.export_rows._job",
        "spacr.qt.widgets.home._easy_database_evidence.sqlite_schema",
        "spacr.qt.widgets.home._easy_database_evidence.store_schema"}
    for symbol, body in entries.items():
        if symbol != "spacr.store_browse":
            assert body == _s6_source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


S6_RETIREMENT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "release_contracts",
    "s6_db_browser_sqlite_api_retirement_2026-11-24.json")


def _retired_s6_browser_api(docs):
    """The Browser's SQLite handles retired by S6, for historical inventory replay.

    ``ReadOnlyDb``, ``WritableDb`` and ``build_update`` left the screen
    module when the Browser moved onto ``spacr.store_browse.StoreDb``.
    """
    import json

    with open(S6_RETIREMENT, encoding="utf-8") as handle:
        retirement = json.load(handle)
    assert retirement["source_path"] == "spacr/qt/screens/db_browser.py"
    retired = retirement["retired"]
    assert len(retired) == 22
    assert all(key.startswith("spacr.qt.screens.db_browser.") for key in retired)
    assert not retired.keys() & docs.keys()
    return retired
