"""The Mask settings form fits its column instead of scrolling sideways.

At 1920x1080 and 100 % the form scrolled sideways and pushed the Source
field's buttons out of view: the microscope convention's example filename is
one unbreakable word to Qt, its two buttons sat side by side, and the empty
state's sentence held a fixed 560 px. The example now breaks inside the
filename only on screen (the text, the selection and Copy stay exact), the
buttons go under each other when one line is too narrow, and the empty
state's sentence wraps to the width it is given.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

FILENAME_KEY = "cellvoyager"


def _field_in_column(qtbot, width):
    """The convention row alone, in a column ``width`` pixels wide."""
    from spacr.qt.screens.settings_model import _MetadataTypeField

    column = QWidget()
    layout = QVBoxLayout(column)
    layout.setContentsMargins(0, 0, 0, 0)
    field = _MetadataTypeField(default=FILENAME_KEY)
    layout.addWidget(field)
    layout.addStretch(1)
    qtbot.addWidget(column)
    column.resize(width, 400)
    column.show()
    qtbot.waitExposed(column)
    return column, field


def test_the_example_breaks_inside_the_filename_but_keeps_its_text(qtbot):
    from spacr.regex_infer import _metadata_convention_example

    filename = _metadata_convention_example(FILENAME_KEY)
    column, field = _field_in_column(qtbot, 220)
    label = field.example
    assert filename in label.text()
    whole = label.fontMetrics().horizontalAdvance(filename)
    assert label.minimumSizeHint().width() < whole
    assert field.minimumSizeHint().width() < whole
    qtbot.waitUntil(lambda: label.width() <= 220, timeout=3000)
    assert label.breaks_inside_a_word()
    lines = label.heightForWidth(label.width()) // label.fontMetrics().lineSpacing()
    assert lines >= 3
    qtbot.waitUntil(lambda: label.height() >= label.heightForWidth(label.width()),
                    timeout=3000)
    label.setSelection(0, len(label.text()))
    assert label.selectedText() == label.text()
    assert "​" not in label.text()
    assert not label.grab().isNull()


def test_the_buttons_stack_only_when_one_line_is_too_narrow(qtbot):
    column, field = _field_in_column(qtbot, 220)
    qtbot.waitUntil(
        lambda: field.detect_button.y() > field.test_button.y(), timeout=3000)
    column.resize(900, 400)
    qtbot.waitUntil(
        lambda: field.detect_button.y() == field.test_button.y(), timeout=3000)
    assert field.detect_button.x() > field.test_button.x()
    assert not field.example.breaks_inside_a_word()


def test_the_empty_state_sentence_wraps_to_a_narrow_column(qtbot):
    from spacr.qt.widgets.empty_state import EmptyState, _WrappedLabel

    card = EmptyState(title="Point Mask at some data",
                      subtitle="Drop a folder of images anywhere on this window "
                               "or type a path into the Source field below.",
                      cta_label="Choose source data")
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(card)
    qtbot.addWidget(host)
    sentence = card.findChild(_WrappedLabel)
    assert card.minimumSizeHint().width() < sentence.maximumWidth()
    host.resize(380, 600)
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(lambda: sentence.width() < 380, timeout=3000)
    assert sentence.height() >= sentence.heightForWidth(sentence.width())


@pytest.mark.parametrize("language", ["en", "de"])
def test_the_mask_form_fits_its_column_at_1920x1080(qtbot, monkeypatch, language):
    from spacr.qt import i18n
    from spacr.qt.preferences import scaled_px
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.settings_model import _MetadataTypeField

    monkeypatch.setattr(i18n, "current_language", lambda: language)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.resize(1920, 1080)
    screen.show()
    qtbot.waitExposed(screen)
    body = screen._body_splitter
    qtbot.waitUntil(
        lambda: body.sizes()[0] >= min(scaled_px(360), int(body.width() * 0.55)) - 8,
        timeout=3000)
    field = screen.findChildren(_MetadataTypeField)[0]
    field.combo.setCurrentIndex(field.combo.findData(FILENAME_KEY))
    scroll = field.parentWidget()
    while not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    qtbot.waitUntil(lambda: scroll.widget().width() <= scroll.viewport().width(),
                    timeout=3000)
    assert scroll.horizontalScrollBar().maximum() == 0
    right = field.detect_button.mapTo(scroll.viewport(),
                                      field.detect_button.rect().topRight()).x()
    assert right <= scroll.viewport().width()
