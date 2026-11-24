"""Item 0aj: the public API that OME-Zarr native storage adds and revises.

The hooks live in :mod:`tests.omezarr_native_api_hooks` (no ``spacr``
import); these tests bind them to the API extractor itself and to the
unchanged item 983 contract.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys

import pytest

from tests.omezarr_native_api_hooks import (
    ARRIVALS, CONTRACTS, REVISIONS, _current_omezarr_native_api,
    _omezarr_native_birth_arrivals, _omezarr_native_changes, _source_docstring)

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _live_docs():
    """Every public body as the API extractor reads it from the source."""
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import build_documentation_i18n as builder

    return builder.public_docstrings()


def test_the_arrivals_are_exactly_the_new_public_api():
    """Every arrival and revision is the extractor's own body; none overlap."""
    docs = _live_docs()
    entries = _omezarr_native_birth_arrivals()
    changes = _omezarr_native_changes()
    assert len(entries) == 22 and len(changes) == 15
    assert not ARRIVALS & REVISIONS
    for symbol in ARRIVALS | REVISIONS:
        assert _source_docstring(symbol) == docs[symbol], symbol
    assert {key: docs[key] for key in entries} == entries


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

    original = json.loads((Path(CONTRACTS) / "983_ome_zarr_images_and_crops_api_arrivals_2026-11-19.json")
                          .read_text(encoding="utf-8"))
    docs = _live_docs()
    active = _current_1007_ome_zarr_api(original, docs)
    revised = REVISIONS & original.keys()
    assert len(revised) == 11
    for symbol in revised:
        assert active[symbol] == docs[symbol], symbol
