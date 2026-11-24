"""Light API hooks for item 0aj (OME-Zarr native storage).

Imports no ``spacr`` module: the source docstrings are read with ``ast``, as
the API extractor reads them, so the docstring-correctness tests can use
these hooks without loading the package.

* ``0aj_omezarr_native_api_arrivals_2026-11-24.json``: the 14 new public
  symbols (the ``spacr-store`` command, store verification, crop labels,
  crop transformations and training datasets, ``PlateWriter.write_labels``).
* ``0aj_omezarr_native_api_changes_2026-11-24.json``: the 14 revised public
  docstrings, each with the sha256 of the body it replaces.

The historical contracts stay unchanged: :func:`_current_omezarr_native_api`
runs after the item 1007 adapter, :func:`_omezarr_native_birth_arrivals`
joins the arrivals and :func:`_validated_omezarr_native_arrivals` leaves the
frozen callable boundary.
"""
import ast
from functools import lru_cache
import hashlib
import inspect
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACTS = os.path.join(ROOT, "tests", "data", "release_contracts")
BIRTH = os.path.join(CONTRACTS, "0aj_omezarr_native_api_arrivals_2026-11-24.json")
CHANGES = os.path.join(CONTRACTS, "0aj_omezarr_native_api_changes_2026-11-24.json")

ARRIVALS = frozenset({
    "spacr.cli_store", "spacr.cli_store.build_parser", "spacr.cli_store.main",
    "spacr.crop_store.crop_rows", "spacr.crop_store.crop_to_fov",
    "spacr.crop_store.crop_transformations", "spacr.crop_store.dataset_store",
    "spacr.crop_store.read_crop_labels", "spacr.crop_store.read_training_dataset",
    "spacr.crop_store.write_training_dataset",
    "spacr.image_store.PlateWriter.write_labels",
    "spacr.image_store.original_planes", "spacr.image_store.verify_project",
    "spacr.image_store.verify_store"})
REVISIONS = frozenset({
    "spacr.crop_store", "spacr.crop_store.CropPoolWriter",
    "spacr.crop_store.class_dir", "spacr.crop_store.committed_rows",
    "spacr.crop_store.cut_crops", "spacr.crop_store.ome_zarr_crops_exist",
    "spacr.crop_store.read_crop", "spacr.crop_store.record_measure_field_crops",
    "spacr.crops.read_merged_plane_layout", "spacr.image_store.PlateWriter",
    "spacr.image_store.PlateWriter.write_field", "spacr.image_store.export_merged",
    "spacr.utils.cleanup_pipeline_folders", "spacr.utils.measure_test_mode"})
#: The new callables the frozen inventory lists. The crop and image store
#: functions are bound by the 983 contract chain instead, and a module is
#: not a callable.
INVENTORY_ARRIVALS = frozenset({
    "spacr.cli_store.build_parser", "spacr.cli_store.main",
    "spacr.image_store.PlateWriter.write_labels"})
MODULES = ("spacr.cli_store", "spacr.crop_store", "spacr.crops",
           "spacr.image_store", "spacr.utils")


def _clean(node):
    """A node's docstring as the extractor cleans it."""
    return inspect.cleandoc(ast.get_docstring(node, clean=False) or "").strip()


@lru_cache(maxsize=1)
def _trees():
    """``{module: parsed source}`` of the modules the 0aj contracts name."""
    trees = {}
    for module in MODULES:
        path = os.path.join(ROOT, *module.split(".")) + ".py"
        with open(path, encoding="utf-8") as handle:
            trees[module] = ast.parse(handle.read())
    return trees


def _source_docstring(symbol):
    """The docstring of one 0aj public symbol, parsed from the source.

    A class gives its docstring and its constructor's, as AutoAPI emits them.
    """
    module = next(name for name in MODULES
                  if symbol == name or symbol.startswith(name + "."))
    node = _trees()[module]
    if symbol == module:
        return _clean(node)
    for part in symbol[len(module) + 1:].split("."):
        node = next(item for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.ClassDef)) and item.name == part)
    text = _clean(node)
    if isinstance(node, ast.ClassDef):
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                constructor = _clean(item)
                if constructor:
                    text = f"{text}\n{constructor}" if text else constructor
    return text


def _omezarr_native_changes():
    """The revised bodies, each bound to the source and to its predecessor's hash."""
    with open(CHANGES, encoding="utf-8") as handle:
        changes = json.load(handle)
    assert set(changes) == REVISIONS
    for symbol, change in changes.items():
        assert set(change) == {"before_sha256", "docstring"}, symbol
        assert change["docstring"] == _source_docstring(symbol), symbol
        assert hashlib.sha256(change["docstring"].encode()).hexdigest() != change["before_sha256"]
    return changes


def _current_omezarr_native_api(entries):
    """Replace only the bound predecessors of the revised bodies with the current ones.

    :param entries: ``{symbol: body}`` of an older contract; a revised symbol
        in it must hold exactly its predecessor (or already the current body).
    :returns: a new mapping; ``entries`` is left unchanged.
    """
    result = dict(entries)
    for symbol, change in _omezarr_native_changes().items():
        if symbol not in result:
            continue
        before = hashlib.sha256(result[symbol].encode()).hexdigest()
        assert before == change["before_sha256"] or result[symbol] == change["docstring"], symbol
        result[symbol] = change["docstring"]
    return result


def _omezarr_native_birth_arrivals(existing=None):
    """The new public symbols, bound to their source docstrings.

    :param existing: earlier arrivals; none of them may be one of these.
    """
    with open(BIRTH, encoding="utf-8") as handle:
        entries = json.load(handle)
    assert set(entries) == ARRIVALS
    for symbol, body in entries.items():
        assert body == _source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_omezarr_native_arrivals(callables):
    """Remove only the exact new callables from the frozen boundary."""
    entries = _omezarr_native_birth_arrivals()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert set(found) == INVENTORY_ARRIVALS
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]
