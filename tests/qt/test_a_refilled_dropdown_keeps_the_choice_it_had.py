"""Refilling a dropdown keeps the selection by value, never by caption.

``set_translatable_items`` replaces every entry when the language changes.
The caption of the selected entry changes with it, so an index or a text
match would land on whatever entry happened to sit in that slot; the value
in the item DATA is the one thing that survives a translation.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox  # noqa: E402

from spacr.qt import i18n  # noqa: E402

pytestmark = pytest.mark.qt

SOURCES = ("Cell", "Nucleus", "Pathogen")
VALUES = ("cell", "nucleus", "pathogen")


def test_the_selected_value_survives_a_refill_in_another_language(qapp):
    combo = QComboBox()
    i18n.set_translatable_items(combo, SOURCES, VALUES, language="en")
    combo.setCurrentIndex(2)
    assert combo.currentData() == "pathogen"

    i18n.set_translatable_items(combo, SOURCES, VALUES, language="sv")

    assert combo.currentData() == "pathogen", (
        "the worker is handed the same English value whatever the caption")
    assert combo.count() == len(SOURCES)


def test_a_value_the_refill_no_longer_offers_does_not_move_the_selection(
        qapp):
    combo = QComboBox()
    i18n.set_translatable_items(combo, SOURCES, VALUES, language="en")
    combo.setCurrentIndex(2)

    i18n.set_translatable_items(combo, SOURCES[:2], VALUES[:2],
                                language="en")

    assert combo.currentData() == "cell", (
        "the old value is gone; the first entry stands rather than a guess")


def test_a_caption_count_that_does_not_match_the_values_is_refused(qapp):
    combo = QComboBox()

    with pytest.raises(ValueError, match="3 captions but 2 values"):
        i18n.set_translatable_items(combo, SOURCES, VALUES[:2])


def test_qt_own_catalogs_are_not_chased_without_an_application(monkeypatch):
    """A helper called before the QApplication exists must return, not crash."""
    import PySide6.QtWidgets as qtwidgets

    class _NoApplication:
        @staticmethod
        def instance():
            return None

    loaded = []
    monkeypatch.setattr(qtwidgets, "QApplication", _NoApplication)
    monkeypatch.setattr(i18n, "install_qt_translations",
                        lambda app, code: loaded.append(code))

    i18n._follow_qt_own_catalogs("sv")

    assert loaded == [], "there is no application to install a catalog into"
