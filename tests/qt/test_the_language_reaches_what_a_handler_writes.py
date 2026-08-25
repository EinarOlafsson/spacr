"""Captions written after the language pass, and the traps around them.

``retranslate_widget_tree`` runs once per screen. Everything a HANDLER
writes afterwards -- a button that says "Copied" for a second, the notice
that says which maturity stages Preferences is hiding, the prompt a hint
strip falls back to when the pointer leaves -- is written in whatever
language the source literal is in, which is English. Worse: the pass reads
a caption it did not render as data and opts that widget out of every later
pass, so one press of Copy console left the button English for the rest of
the session.

Three shapes are covered here:

* **written by a handler** -- translate at the moment of writing;
* **composed before lookup** -- a sentence carrying a count, a module name
  or a stage name matches no catalog row, so the row carries a placeholder
  and the value is substituted after the lookup;
* **the ``_ROWS`` / ``_TERM_ROWS`` trap** -- with no exact row, the
  word-by-word fallback rewrites the words it recognises and leaves the
  rest, which is how ``Choose image…`` became ``Choose Bild…`` and
  ``cell_area > 1000`` became ``Cell_area > 1000``.

Qt's OWN text is here too: ``&Copy`` and ``Select All`` come from
``qtbase_<lang>.qm`` rather than from any catalog spaCR writes, and that
file was loaded once at startup and never again.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit

from spacr.qt.i18n import (
    install_qt_translations, retranslate_widget_tree, tr,
)
from spacr.qt.screens.app_screen import AppScreen


@pytest.fixture()
def swedish(qtbot, monkeypatch):
    """A Mask screen built and translated the way MainWindow does it."""
    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    retranslate_widget_tree(screen)
    return screen


# ---------------------------------------------------------------------------
# Written by a handler
# ---------------------------------------------------------------------------

def test_copying_the_console_does_not_turn_the_button_english(swedish, qtbot):
    """The button flips to "Copied" and back; both are the chosen language."""
    button = swedish._btn_copy_console
    before = button.text()
    assert before == tr("Copy console", "sv") != "Copy console"

    swedish._on_copy_console()
    assert button.text() == tr("Copied", "sv") != "Copied"
    # And it comes back to the translated caption, not to the source.
    qtbot.waitUntil(lambda: button.text() != tr("Copied", "sv"), timeout=3000)
    assert button.text() == before


def test_a_hint_strip_falls_back_to_a_translated_prompt(swedish):
    """Leaving a header writes the prompt back; it must not write English."""
    section = swedish._settings_sections[0]
    header = section.header()
    QApplication.sendEvent(header, QEvent(QEvent.Type.Enter))
    QApplication.processEvents()
    QApplication.sendEvent(header, QEvent(QEvent.Type.Leave))
    QApplication.processEvents()
    assert swedish._category_hint.text() == swedish._default_category_hint()
    assert "Hover a settings category" not in swedish._category_hint.text()

    label = next(w for w in swedish._settings_content.findChildren(QLabel)
                 if w.property("settingKey"))
    QApplication.sendEvent(label, QEvent(QEvent.Type.Enter))
    QApplication.processEvents()
    QApplication.sendEvent(label, QEvent(QEvent.Type.Leave))
    QApplication.processEvents()
    assert swedish._hint_strip.text() == swedish._default_hint()
    assert "Hover any setting" not in swedish._hint_strip.text()


def test_the_example_button_comes_back_in_the_chosen_language(qtbot,
                                                              monkeypatch):
    """Both captions it writes -- while fetching and after -- are looked up."""
    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    retranslate_widget_tree(screen)
    button = screen._example_data_button
    assert button is not None

    seen = {}

    def fake_missing():
        return ["counts.csv", "scores.csv"]

    def fake_fetch(download=True, progress=None):
        seen["while fetching"] = button.text()
        raise ExampleDataError("no network in a test")

    from spacr.example_data import ExampleDataError
    monkeypatch.setattr("spacr.example_data.missing", fake_missing)
    monkeypatch.setattr("spacr.example_data.fetch", fake_fetch)

    screen.load_the_example_screen(download=False)
    assert seen["while fetching"] == tr(
        "Fetching {count} file(s)…", "sv", count=2)
    assert "Fetching" not in seen["while fetching"]
    assert button.text() == tr("Load the example screen…", "sv")
    assert button.text() != "Load the example screen…"


# ---------------------------------------------------------------------------
# Composed before lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("hidden", "expected_stages"),
    [({"beta"}, "Beta"), ({"alpha"}, "Alpha"),
     ({"alpha", "beta"}, "Alpha and Beta")],
)
def test_the_maturity_notice_is_built_from_translated_parts(
    swedish, monkeypatch, hidden, expected_stages,
):
    """One stage or two: the sentence and the names are looked up apart.

    Mask registers Beta sections and no Alpha ones, so the two other cases
    are staged by marking one section Alpha. The notice names the stages it
    actually hid, which is what makes the sentence composed rather than
    fixed.
    """
    import spacr.qt.preferences as preferences

    swedish._settings_sections[0].set_maturity("alpha")
    swedish._settings_sections[1].set_maturity("beta")
    monkeypatch.setattr(
        preferences, "maturity_is_visible", lambda stage: stage not in hidden)
    swedish.refresh_maturity_visibility()

    text = swedish._maturity_notice.text()
    assert text, "nothing was hidden, so nothing was said"
    assert text == tr(
        "{stages} settings are hidden by Preferences. Enable them in "
        "Preferences → Feature maturity.", "sv",
        stages=tr(expected_stages, "sv"))
    assert "settings are hidden by Preferences" not in text


def test_an_uninterpretable_module_says_so_in_the_chosen_language(swedish):
    """The dialog names the module, so the sentence is a row with a slot."""
    rendered = tr(
        "The '{app}' app is interactive-only in this Qt build. Use the "
        "classic Tk GUI (`spacr`) for now.", "sv", app="plate_view")
    assert "plate_view" in rendered
    assert "interactive-only" not in rendered


# ---------------------------------------------------------------------------
# The _ROWS / _TERM_ROWS trap
# ---------------------------------------------------------------------------

def test_an_exact_row_beats_the_word_by_word_fallback():
    """"Choose image…" was rendered "Choose Bild…": half of two languages."""
    for code in ("sv", "de", "fr", "ko"):
        rendered = tr("Choose image…", code)
        assert rendered != "Choose image…"
        assert "Choose" not in rendered, (code, rendered)


@pytest.mark.parametrize("code", ["sv", "de", "fr"])
@pytest.mark.parametrize(
    "source",
    ["cell_area > 1000 AND LIKE 'A%'",
     "Search columns… e.g. 'percentile' or 'channel_1'",
     "image_path",
     "regression_type"],
)
def test_a_code_name_is_never_translated_word_by_word(source, code):
    """A column name a user is shown is a name they may have to type back."""
    rendered = tr(source, code)
    for name in ("cell_area", "channel_1", "image_path", "regression_type"):
        if name in source:
            assert name in rendered, (code, rendered)


@pytest.mark.parametrize("code", ["sv", "de", "es", "pt", "hi", "ko", "is",
                                  "fr", "zh_CN"])
@pytest.mark.parametrize("name", ["image_path", "png_path", "png_list",
                                  "fdr_bh", "seg_qc", "RdBu_r"])
def test_the_catalogs_answer_an_identifier_with_itself(name, code):
    """These are combo entries a module reads back; a period breaks them."""
    assert tr(name, code) == name


# ---------------------------------------------------------------------------
# Qt's own text
# ---------------------------------------------------------------------------

def test_qts_own_menu_text_follows_a_language_chosen_after_launch(qtbot):
    """`&Copy` and `Select All` come from qtbase_<lang>.qm, loaded at start."""
    app = QApplication.instance()
    field = QLineEdit()
    qtbot.addWidget(field)

    install_qt_translations(app, "en")
    assert app.translate("QLineEdit", "&Copy") == "&Copy"

    retranslate_widget_tree(field, "sv")
    assert app.translate("QLineEdit", "&Copy") != "&Copy"
    assert app.translate("QLineEdit", "Select All") != "Select All"

    retranslate_widget_tree(field, "de")
    assert app.translate("QLineEdit", "&Copy") == "&Kopieren"

    # Qt ships no Icelandic catalog. English there is the honest answer, and
    # the previous language must not be left underneath answering for it.
    retranslate_widget_tree(field, "is")
    assert app.translate("QLineEdit", "&Copy") == "&Copy"

    install_qt_translations(app, "en")
