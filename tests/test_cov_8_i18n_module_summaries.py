"""A missing generated catalog must not blank a module's description.

:func:`spacr.qt.i18n_module_summaries.module_summary` reaches for the
generated ``i18n_catalogs`` only after the hand-reviewed table has failed to
answer. That import is optional -- the generated catalogs are a build
product, and a checkout or a trimmed wheel can be without them -- so the
lookup has to survive both spellings of "not there": the module absent
altogether, and a module present but missing the function. Either way the
caller must still be handed a readable sentence, because the return value is
painted straight into the module picker.
"""

from __future__ import annotations

import builtins
import hashlib
import sys
import types

from spacr.qt import i18n_module_summaries as ims


_ENGLISH = "Count the plaques in a well and report their sizes."


def test_a_catalog_without_the_function_still_yields_english(monkeypatch):
    """An ``i18n_catalogs`` missing ``module_summary`` is not a blank label."""
    stub = types.ModuleType("spacr.qt.i18n_catalogs")
    monkeypatch.setitem(sys.modules, "spacr.qt.i18n_catalogs", stub)

    summary = ims.module_summary("a_plugin_module", _ENGLISH, "sv")

    assert summary == _ENGLISH


def test_a_catalog_that_cannot_be_imported_still_yields_english(monkeypatch):
    """An import error from the generated catalog falls back, not raises."""
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if level and name == "i18n_catalogs":
            raise ImportError("no generated catalogs in this build")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "spacr.qt.i18n_catalogs", raising=False)

    summary = ims.module_summary("a_plugin_module", _ENGLISH, "sv")

    assert summary == _ENGLISH


def test_a_reviewed_summary_still_wins_over_the_generated_catalog(monkeypatch):
    """The fallback is a fallback: a hash-matched review is preferred."""
    reviewed = "Rakna plaques i en brunn och rapportera deras storlek."
    monkeypatch.setitem(ims.MODULE_SUMMARIES["sv"], "a_plugin_module",
                        reviewed)
    monkeypatch.setitem(
        ims.REVIEWED_SOURCE_HASHES, "a_plugin_module",
        hashlib.sha256(_ENGLISH.encode("utf-8")).hexdigest())
    stub = types.ModuleType("spacr.qt.i18n_catalogs")
    stub.module_summary = lambda *a, **k: "the generated sentence"
    monkeypatch.setitem(sys.modules, "spacr.qt.i18n_catalogs", stub)

    assert ims.module_summary("a_plugin_module", _ENGLISH, "sv") == reviewed
