"""Recording an acceptance never fails on the flush, and never on a catalog.

Two failures that must cost nothing:

* A ``QSettings`` store that refuses ``sync()`` -- a read-only config
  directory, a full disk -- still leaves the acceptance recorded in the
  store's own buffer. Raising here would mean the user accepted the terms
  and was told the acceptance failed.
* A translation catalog that cannot be reached, or that refuses a row,
  leaves the screen in the English it was written in rather than
  unbuilt.
"""
from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """A settings store of this test's own, so nothing is accepted for real."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt import preferences, terms

    importlib.reload(preferences)
    importlib.reload(terms)
    yield
    importlib.reload(preferences)
    importlib.reload(terms)


class _RefusesToFlush:
    """A settings store that accepts writes and cannot write them through."""

    def __init__(self):
        self.written = {}

    def setValue(self, key, value):
        self.written[key] = value

    def sync(self):
        raise OSError("read-only configuration directory")


def test_a_store_that_cannot_sync_still_records_the_acceptance(monkeypatch):
    from spacr.qt import terms

    store = _RefusesToFlush()
    monkeypatch.setattr(terms, "_settings", lambda: store)

    stamped = terms.record_agreement()

    assert stamped == terms.TERMS_VERSION
    assert store.written[terms._KEY_VERSION] == terms.TERMS_VERSION, (
        "the version is written before the flush that failed")
    assert store.written[terms._KEY_WHEN], "and so is the timestamp"


def test_without_a_catalog_the_screen_registers_nothing_and_says_so(
        monkeypatch):
    from spacr.qt import terms

    import types
    hollow = types.ModuleType("spacr.qt.i18n")
    monkeypatch.setitem(sys.modules, "spacr.qt.i18n", hollow)

    assert terms.register_translations() == 0, (
        "a screen with no catalog is still a screen")


def test_a_row_the_catalog_refuses_is_skipped_not_fatal(monkeypatch):
    from spacr.qt import i18n, terms

    def refuse(source, values):
        raise ValueError(f"{source!r} does not fit the catalog")

    monkeypatch.setattr(i18n, "add_translation", refuse)

    assert terms.register_translations() == 0
    assert terms.TRANSLATIONS, "there were rows to refuse"
