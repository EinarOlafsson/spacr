"""Item 0aj: the public API that OME-Zarr native storage adds and revises.

Two contracts, both bound to the live source docstrings:

* ``0aj_omezarr_native_api_arrivals_2026-11-24.json``: the 14 new public
  symbols (the ``spacr-store`` command, store verification, crop labels,
  crop transformations and training datasets, ``PlateWriter.write_labels``).
* ``0aj_omezarr_native_api_changes_2026-11-24.json``: the 14 revised public
  docstrings, each with the sha256 of the body it replaces.

The historical contracts stay unchanged. The protected consumers apply
:func:`_current_omezarr_native_api` after the item 1007 adapter, add
:func:`_omezarr_native_birth_arrivals` to their arrivals and remove
:func:`_validated_omezarr_native_arrivals` from the frozen callable boundary.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "tests" / "data" / "release_contracts"
BIRTH = CONTRACTS / "0aj_omezarr_native_api_arrivals_2026-11-24.json"
CHANGES = CONTRACTS / "0aj_omezarr_native_api_changes_2026-11-24.json"

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


@lru_cache(maxsize=1)
def _live_docs():
    """Every public body as the API extractor reads it from the source."""
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import build_documentation_i18n as builder

    return builder.public_docstrings()


def _omezarr_native_changes():
    """The revised bodies, each bound to the live source and to its predecessor's hash."""
    changes = json.loads(CHANGES.read_text(encoding="utf-8"))
    assert set(changes) == REVISIONS
    docs = _live_docs()
    for symbol, change in changes.items():
        assert set(change) == {"before_sha256", "docstring"}, symbol
        assert docs[symbol] == change["docstring"], symbol
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
    entries = json.loads(BIRTH.read_text(encoding="utf-8"))
    assert set(entries) == ARRIVALS
    docs = _live_docs()
    for symbol, body in entries.items():
        assert body == docs[symbol], symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_omezarr_native_arrivals(callables):
    """Remove only the exact new callables from the frozen boundary."""
    entries = _omezarr_native_birth_arrivals()
    found = {item.symbol: item for item in callables if item.symbol in entries}
    # The frozen inventory lists the new command's callables and the new
    # PlateWriter method; the crop and image store modules' functions are
    # bound by the 983 contract chain instead, and a module is not a callable.
    assert set(found) == {"spacr.cli_store.build_parser", "spacr.cli_store.main",
                          "spacr.image_store.PlateWriter.write_labels"}
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]


def test_the_arrivals_are_exactly_the_new_public_api():
    """Every arrival is a live public body; the revisions keep their old hash."""
    entries = _omezarr_native_birth_arrivals()
    assert len(entries) == 14
    assert not ARRIVALS & REVISIONS
    changes = _omezarr_native_changes()
    assert len(changes) == 14


def test_the_adapter_changes_only_bound_predecessors():
    """An unbound third body is refused; the current body passes through unchanged."""
    changes = _omezarr_native_changes()
    current = {symbol: change["docstring"] for symbol, change in changes.items()}
    assert _current_omezarr_native_api(current) == current
    assert _current_omezarr_native_api({"spacr.unrelated": "x"}) == {"spacr.unrelated": "x"}
    for symbol in REVISIONS:
        with pytest.raises(AssertionError):
            _current_omezarr_native_api({symbol: "unbound third body"})
    with pytest.raises(AssertionError):
        _omezarr_native_birth_arrivals({"spacr.cli_store.main": "earlier"})


def test_the_983_contract_reaches_the_current_bodies_through_the_adapters():
    """The 983 fixture, unchanged, becomes today's bodies after 1007 and 0aj."""
    from tests.test_documentation_i18n import _current_1007_ome_zarr_api

    original = json.loads((CONTRACTS / "983_ome_zarr_images_and_crops_api_arrivals_2026-11-19.json")
                          .read_text(encoding="utf-8"))
    active = _current_1007_ome_zarr_api(original, _live_docs())
    docs = _live_docs()
    revised = REVISIONS & original.keys()
    assert len(revised) == 11
    for symbol in revised:
        assert active[symbol] == docs[symbol], symbol
