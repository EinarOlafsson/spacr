"""The Dose-Response pickers keep their values in every language.

The language pass rewrites the item text of every dropdown not marked to skip
its items, by exact catalog match. Every picker on this screen is filled from
the table -- column names and control levels -- plus one sentinel, "(none)" or
"(one curve for the whole table)", and every handler read the choice back with
``currentText()``.

On a French screen that handed the fit translated names. "(none)" became
"(Aucune)", which is not "(none)", so an untouched Plates row asked for a plate
column called "(Aucune)"; a group column named ``gene`` became "gène", which is
not a column of the table. Both refuse the fit before anything is drawn.

The captions of the two sentinels still follow the language; the column names
are data and never move.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.i18n import retranslate_widget_tree, tr
from spacr.qt.screens.dose_response import NO_GROUP

from tests.qt.test_dose_response_screen import frame, screen  # noqa: F401

pytestmark = pytest.mark.qt

#: Languages in which both "(none)" and the column name ``gene`` translate.
LANGUAGES = ("fr", "de")

_DATA_PICKERS = ("concentration_picker", "response_picker", "group_picker",
                 "plate_picker", "control_picker", "positive_picker",
                 "negative_picker", "host_picker", "second_dose_picker")


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_premise_both_names_translate(language):
    """Without this the tests below would pass for the wrong reason."""
    assert tr("(none)", language) != "(none)"
    assert tr("gene", language) != "gene"


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_translated_screen_still_fits_every_group(screen, language):
    """Group ``gene`` and Plates left at "(none)", after a language change."""
    retranslate_widget_tree(screen, language)

    screen.fit()

    fitted = screen.result_set()
    assert fitted is not None, screen.report.toPlainText()
    assert [fit.group for fit in fitted.fits] == [
        "geneA", "geneB", "geneC", "geneD"]
    assert screen._plate_reports == ()


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_whole_table_choice_survives_a_language_change(screen, language):
    screen.group_picker.setCurrentIndex(0)
    retranslate_widget_tree(screen, language)

    screen.fit()

    fitted = screen.result_set()
    assert fitted is not None, screen.report.toPlainText()
    assert len(fitted.fits) == 1 and not fitted.fits[0].group


@pytest.mark.parametrize("language", LANGUAGES)
def test_column_names_never_move_and_sentinels_follow_the_language(
        screen, language):
    retranslate_widget_tree(screen, language)

    group = screen.group_picker
    assert group.itemData(0) == NO_GROUP
    assert group.itemText(0) == tr(NO_GROUP, language)
    assert "gene" in [group.itemText(i) for i in range(group.count())]
    assert screen.plate_picker.itemData(0) == "(none)"
    assert screen.plate_picker.itemText(0) == tr("(none)", language)
    for name in _DATA_PICKERS:
        picker = getattr(screen, name)
        for index in range(picker.count()):
            value = picker.itemData(index)
            if value not in ("(none)", NO_GROUP):
                assert picker.itemText(index) == value, (name, value)

