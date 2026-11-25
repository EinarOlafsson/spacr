"""Light API hooks for items 0aq and 0ar (OPS fields in fov.ome.zarr, one crop-index store).

Imports no ``spacr`` module. ``0aq_ops_fields_store_api_changes_2026-11-24.json``
holds the two public docstrings these items revise after the 0aj contracts
bound them: the ``spacr.crop_store`` module (the crop index now lives in the
one store of the project) and ``spacr.image_store.verify_store`` (its
``names`` parameter). Each entry carries the sha256 of the body it replaces,
so the 0aj hooks accept the revision only in place of that exact body.
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANGES = os.path.join(ROOT, "tests", "data", "release_contracts",
                       "0aq_ops_fields_store_api_changes_2026-11-24.json")
REVISIONS = frozenset({"spacr.crop_store", "spacr.image_store.verify_store"})


def _ops_fields_changes():
    """``{symbol: {"before_sha256", "docstring"}}`` of the revised bodies."""
    with open(CHANGES, encoding="utf-8") as handle:
        changes = json.load(handle)
    assert set(changes) == REVISIONS
    return changes


def _revised_later(symbol, body, source):
    """Whether ``source`` is the 0aq revision of exactly ``body``.

    :param symbol: the public symbol.
    :param body: the docstring an earlier contract bound.
    :param source: the docstring the source holds now.
    """
    change = _ops_fields_changes().get(symbol)
    return (change is not None and change["docstring"] == source
            and hashlib.sha256(body.encode()).hexdigest() == change["before_sha256"])
