"""The seams of the language pass that only a missing answer reaches.

Every branch pinned here is what ``spacr/qt/i18n.py`` does when a lookup
comes back with nothing: a padded label whose trimmed form is not in any
catalog, a plugin that ships translations for other strings than this one, a
module entry with nowhere to put its help, a settings catalog that answers
with an empty label, a tree that reports no header item, and the deliberate
refusal to import ``settings_model`` just to retranslate a page that has no
settings on it.

Each test drives BOTH sides of its branch, because "nothing happened" is also
what a function returns when it never ran.

Real Qt widgets and real ``setProperty`` throughout: the opt-outs and the
semantic hints this module reads are written entirely in Qt properties, and a
dict stand-in would test the stand-in.
"""
from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel, QTreeWidget, QWidget

from spacr.qt import i18n
from spacr.qt.i18n import (ENV_LANGUAGE, _exact_translation,
                           retranslate_widget_tree, tr)


@pytest.fixture(autouse=True)
def english_unless_asked(monkeypatch):
    """Pin the ambient language so a stray preference cannot move a result."""
    monkeypatch.setenv(ENV_LANGUAGE, "en")


# ---------------------------------------------------------------------------
# _exact_translation
# ---------------------------------------------------------------------------

def test_the_padding_retry_hands_back_to_the_catalog_when_it_misses():
    """A padded source is retried trimmed; a miss must not short-circuit.

    The retry returns the translation re-wrapped in the original whitespace
    when the trimmed form is known.  When it is not, the function has to fall
    through to the ordinary catalog lookups rather than return the retry's
    ``None`` -- the untrimmed source may still match a mnemonic or an
    uppercased header further down.
    """
    assert _exact_translation("  Settings  ", "sv") == "  Inställningar  "
    assert _exact_translation("\tSettings\n", "sv") == "\tInställningar\n"

    # Not in CATALOGS, TERM_CATALOGS or the external catalogs, trimmed or not.
    assert _exact_translation("  Zzq Unknown Probe  ", "sv") is None


def test_a_plugin_without_this_string_does_not_stop_the_one_that_has_it():
    """Plugin catalogs are searched in order; a gap in one is not an answer.

    Both plugins are asked for the same source.  The first has a catalog for
    the language and no entry for the string, which must continue the loop
    rather than settle it.
    """
    plugins = pytest.importorskip("spacr.plugins")
    quiet = plugins.SpacrPlugin(
        name="quiet", version="1.0", translations={"sv": {}})
    loud = plugins.SpacrPlugin(
        name="loud", version="1.0",
        translations={"sv": {"Zzq Plugin Probe": "Provetikett"}})

    saved = plugins.discover_plugins
    plugins.discover_plugins = lambda: (quiet, loud)
    try:
        # The second plugin is only reachable if the first one's empty
        # catalog was stepped over.
        assert _exact_translation("Zzq Plugin Probe", "sv") == "Provetikett"
        # No plugin has a German catalog: every iteration falls through.
        assert _exact_translation("Zzq Plugin Probe", "de") is None
    finally:
        plugins.discover_plugins = saved


# ---------------------------------------------------------------------------
# _refresh_module_help
# ---------------------------------------------------------------------------

def test_a_module_entry_with_nowhere_to_put_help_still_gets_asked(monkeypatch):
    """The summary is built before the style dispatch, and may go nowhere.

    A ``QAction`` carrying module properties has a status tip; a bare
    ``QObject`` carrying the same properties has neither tooltip nor status
    tip, and the function has to end without raising.  Recording the summary
    call is what separates "reached the dispatch and had nowhere to write"
    from "returned early at the property guard".
    """
    from spacr.qt import i18n_module_summaries

    asked = []

    def _summary(app_key, source, language):
        asked.append((app_key, source, language))
        return "Segmenterar objekt"

    monkeypatch.setattr(i18n_module_summaries, "module_summary", _summary)

    holder = QObject()
    holder.setProperty("moduleAppKey", "mask")
    holder.setProperty("moduleSummarySource", "Segment objects")
    assert not hasattr(holder, "setStatusTip"), (
        "a QObject is the case this branch exists for")

    entry = QAction("Mask")
    entry.setProperty("moduleAppKey", "mask")
    entry.setProperty("moduleSummarySource", "Segment objects")

    i18n._refresh_module_help(holder, "sv")
    i18n._refresh_module_help(entry, "sv")

    assert asked == [("mask", "Segment objects", "sv")] * 2, (
        "both objects reached the style dispatch")
    assert entry.statusTip() == "Segmenterar objekt"


# ---------------------------------------------------------------------------
# retranslate_widget_tree
# ---------------------------------------------------------------------------

def test_an_empty_settings_label_falls_back_to_the_term_pass(qapp, monkeypatch):
    """An empty answer from the settings catalog is not an answer.

    ``Object image`` is in neither the compact catalog nor the term rows, so
    it goes to ``setting_label``.  When that returns a usable label it wins;
    when it returns an empty string the widget must fall through to the
    generic pass, which translates the words it knows -- not be left with a
    blank caption.
    """
    from spacr.qt import i18n_catalogs

    source = "Object image"
    generic = tr(source, "sv")
    assert generic not in ("", source), "the generic pass has something to say"

    monkeypatch.setattr(i18n_catalogs, "setting_label", lambda *a, **k: "")
    blank_root = QWidget()
    blank = QLabel(source, blank_root)
    blank.setProperty("settingKey", "object_image")
    blank.setProperty("settingsAppKey", "measure")
    retranslate_widget_tree(blank_root, "sv")
    assert blank.text() == generic

    monkeypatch.setattr(
        i18n_catalogs, "setting_label", lambda *a, **k: "Objektbild")
    named_root = QWidget()
    named = QLabel(source, named_root)
    named.setProperty("settingKey", "object_image")
    named.setProperty("settingsAppKey", "measure")
    retranslate_widget_tree(named_root, "sv")
    assert named.text() == "Objektbild"
    assert named.text() != generic, "the settings catalog outranks the terms"


class _HeaderlessTree(QTreeWidget):
    """A tree whose ``headerItem()`` is empty.

    Subclassed rather than mocked because the pass dispatches on
    ``isinstance(widget, QTreeWidget)``: only a real tree reaches the header
    block at all.  ``headerItem()`` is not virtual in C++, so overriding it
    changes what the Python-side language pass sees and nothing else.
    """

    def headerItem(self):
        return None


def test_a_tree_with_no_header_item_does_not_stop_the_next_one(qapp):
    """The header guard keeps the pass alive for every widget after it.

    Without it, ``header.setText`` raises ``AttributeError`` out of the loop
    and every widget built after the offending tree keeps its English text --
    which is what the second tree here is for.  Its blank second column also
    pins the skip that stops an empty header label being "translated" into a
    catalog hit for the empty string.
    """
    root = QWidget()
    headerless = _HeaderlessTree(root)
    headerless.setColumnCount(1)
    normal = QTreeWidget(root)
    normal.setHeaderLabels(["Settings", ""])

    retranslate_widget_tree(root, "sv")

    header = normal.headerItem()
    assert header.text(0) == "Inställningar", (
        "the pass continued past the tree with no header item")
    assert header.text(1) == "", "a blank header column is left blank"


def test_settings_tooltips_are_rebuilt_only_when_that_module_is_loaded(
        qapp, monkeypatch):
    """Importing ``settings_model`` to retranslate is refused on purpose.

    The module reaches pandas through the mask inputs and costs ~0.3 s of
    launch.  Nothing can be holding one of its tooltips unless it has already
    been imported, so the pass skips the rebuild entirely when it is absent
    from ``sys.modules`` -- and must still run it when it is present.
    """
    from spacr.qt.screens import settings_model

    rebuilt = []
    monkeypatch.setattr(
        settings_model, "refresh_api_tooltips",
        lambda root, code: rebuilt.append(code))

    root = QWidget()
    retranslate_widget_tree(root, "sv")
    assert rebuilt == ["sv"], "an imported settings_model is refreshed"

    monkeypatch.delitem(sys.modules, "spacr.qt.screens.settings_model")
    retranslate_widget_tree(root, "de")
    assert rebuilt == ["sv"], (
        "an unimported settings_model is not imported to be refreshed")
