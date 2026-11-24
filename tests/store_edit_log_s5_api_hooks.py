"""Light API-arrival hooks of store edit-log step S5.

Imports no ``spacr`` module: the source docstrings are read with ``ast``, so
the docstring-correctness and API-extractor tests can use these hooks without
loading the package.
"""
import json
import os


S5_BIRTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "release_contracts",
                        "s5_store_edit_log_api_arrivals_2026-11-23.json")
S5_ARRIVALS = frozenset({
    "spacr.store.edit", "spacr.store.journal", "spacr.store.EditJournal",
    "spacr.store.EditJournal.__enter__", "spacr.store.EditJournal.__exit__",
    "spacr.store.EditJournal.add", "spacr.store.EditJournal.add_many",
    "spacr.store.EditJournal.add_batch", "spacr.store.EditJournal.undo",
    "spacr.store.EditJournal.fold", "spacr.store.EditJournal.close",
    "spacr.store.ensure_columns", "spacr.store.undo", "spacr.store.compact",
    "spacr.store.write_scores", "spacr.store.append_rows",
    "spacr.store.activate",
    "spacr.store.Store.edits",
    "spacr.store.Store.active_outputs", "spacr.store.Store.training",
    "spacr.agreement.annotations_by_author",
    # S5, second part (2026-11-24): the all-or-nothing transaction that
    # hit attribution, Graph Builder, active learning and the prediction
    # merges write through, and the on-request import of legacy labels.
    "spacr.store.transaction", "spacr.store.Transaction",
    "spacr.store.Transaction.tables", "spacr.store.Transaction.columns",
    "spacr.store.Transaction.rows", "spacr.store.Transaction.edit",
    "spacr.store.Transaction.ensure_columns",
    "spacr.store.Transaction.append_rows",
    "spacr.store.Transaction.update_rows", "spacr.store.migrate_legacy"})


def _s5_source_docstring(symbol):
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
    text = inspect.cleandoc(ast.get_docstring(scope, clean=False))
    if isinstance(scope, ast.ClassDef):
        init = next((node for node in scope.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "__init__"), None)
        if init is not None and ast.get_docstring(init, clean=False):
            text += "\n" + inspect.cleandoc(ast.get_docstring(init,
                                                              clean=False))
    return text


def _s5_birth_arrivals(existing=None):
    """The S5 public symbols, bound to their source docstrings."""
    with open(S5_BIRTH, encoding="utf-8") as handle:
        entries = json.load(handle)
    assert set(entries) == S5_ARRIVALS
    for symbol, body in entries.items():
        assert body == _s5_source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_s5_arrivals(callables):
    """Remove only the S5 exact new callables from the frozen boundary."""
    entries = _s5_birth_arrivals()
    found = {item.symbol: item for item in callables
             if item.symbol in entries}
    assert set(found) == {s for s in S5_ARRIVALS if ".__" not in s}
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]
