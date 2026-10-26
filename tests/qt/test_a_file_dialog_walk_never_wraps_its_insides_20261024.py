"""The language pass never reaches inside a Qt file, colour or font picker.

Lesson 12 crashed (SIGSEGV in ``_a11y_label_beside``) when Map Barcodes'
file list opened the reference-file picker: the deferred dialog pass walked
the QFileDialog's own widgets, which Qt builds and deletes as it likes, and a
wrapper made for one of them outlived it. The pass now names and translates
the picker itself, its title, and nothing Qt built inside it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
shiboken6 = pytest.importorskip("shiboken6")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QColorDialog, QFileDialog, QFontDialog, QLabel,
    QVBoxLayout, QWidget,
)

from spacr.qt import i18n  # noqa: E402


@pytest.fixture
def dialog_filter(qapp):
    previous = getattr(qapp, "_spacr_dialog_i18n_filter", None)
    qapp._spacr_dialog_i18n_filter = None
    i18n.install_dialog_translation(qapp)
    yield
    from spacr.qt.gil_priority import _stop_watching_application_events

    installed = qapp._spacr_dialog_i18n_filter
    if installed is not None and installed is not previous:
        _stop_watching_application_events(qapp, installed)
        installed.deleteLater()
    qapp._spacr_dialog_i18n_filter = previous


@pytest.fixture
def named(monkeypatch):
    seen = []
    real = i18n._name_for_assistive_tech

    def recording(widgets, stamp=None, only_new=False):
        widgets = list(widgets)
        seen.append(widgets)
        return real(widgets, stamp, only_new)

    monkeypatch.setattr(i18n, "_name_for_assistive_tech", recording)
    return seen


def _picker(parent=None):
    picker = QFileDialog(parent, "Choose source folder")
    picker.setOption(QFileDialog.DontUseNativeDialog, True)
    return picker


@pytest.mark.parametrize("make", [_picker, lambda p=None: QColorDialog(p),
                                  lambda p=None: QFontDialog(p)])
def test_a_shown_picker_is_named_alone(qapp, dialog_filter, named, make):
    picker = make()
    try:
        picker.show()
        qapp.processEvents()
        assert named, "the deferred pass never ran"
        assert all(w is picker for w in named[-1]), [
            type(w).__name__ for w in named[-1]]
    finally:
        picker.close()
        shiboken6.delete(picker)


def test_a_picker_closed_while_its_pass_is_pending(qapp, dialog_filter, named):
    for _ in range(5):
        picker = _picker()
        picker.show()
        picker.close()
        shiboken6.delete(picker)
        qapp.processEvents()
    assert named == []


def test_a_window_holding_an_open_picker_is_walked_around_it(
        qapp, dialog_filter, named):
    window = QWidget()
    layout = QVBoxLayout(window)
    layout.addWidget(QLabel("Source", window))
    picker = _picker(window)
    try:
        picker.show()
        qapp.processEvents()
        named.clear()
        i18n.retranslate_widget_tree(window)
        walked = named[-1]
        assert window in walked and picker in walked
        inside = set(map(id, picker.findChildren(QWidget)))
        assert not inside & set(map(id, walked)), "the walk reached the picker's insides"
    finally:
        picker.close()
        shiboken6.delete(window)


def test_the_picker_title_is_still_translated(qapp, dialog_filter, monkeypatch):
    from spacr.qt.i18n_catalogs import ui_text

    source = "Choose source folder"
    expected = ui_text(source, "de")
    assert expected and expected != source
    monkeypatch.setenv("SPACR_LANGUAGE", "de")
    picker = _picker()
    try:
        picker.show()
        QApplication.processEvents()
        assert picker.windowTitle() == expected
    finally:
        picker.close()
        shiboken6.delete(picker)


def _save_gates_picker_that(closes_first, opened, monkeypatch):
    """Stand in for ``QFileDialog.getSaveFileName`` with a real picker.

    The picker is shown, so the dialog filter schedules its deferred pass,
    and is either destroyed before that pass can run or left open for it.
    """
    def ask(parent, caption, directory, filters):
        picker = QFileDialog(parent, caption, directory, filters)
        picker.setOption(QFileDialog.DontUseNativeDialog, True)
        picker.show()
        opened.append(picker)
        if closes_first:
            picker.close()
            shiboken6.delete(picker)
        else:
            for _ in range(5):
                QApplication.processEvents()
            picker.reject()
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(ask))


@pytest.mark.parametrize("closes_first", [True, False],
                         ids=["closed_before_the_pass", "open_for_the_pass"])
def test_the_save_gates_picker_survives_the_deferred_language_pass(
        qtbot, dialog_filter, named, monkeypatch, closes_first):
    """Lesson 64 on the WS capture host: Save gates, then the deferred pass.

    The dialog filter schedules ``_translate_shown_dialog`` on the picker's
    ``Show`` through ``_after``. A picker destroyed first takes the pending
    call with it, and one still open for it is walked as itself alone.
    """
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.show()
    opened, walked = [], []
    real = i18n._translate_shown_dialog
    monkeypatch.setattr(i18n, "_translate_shown_dialog",
                        lambda dialog: (walked.append(dialog), real(dialog))[1])
    _save_gates_picker_that(closes_first, opened, monkeypatch)
    named.clear()
    screen.choose_save_gates()
    for _ in range(10):
        QApplication.processEvents()
    assert opened, "the picker was never asked for"
    if closes_first:
        assert walked == [], "the pass ran on a picker that was gone"
    else:
        assert any(dialog is opened[0] for dialog in walked)
        assert all(len(widgets) <= 1 for widgets in named), (
            "the pass wrapped widgets Qt built inside the picker")
