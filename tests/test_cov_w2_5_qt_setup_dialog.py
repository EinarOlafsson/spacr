"""The first-run setup box is the same box with or without its decoration.

The blurred still of the home screen behind the card is decorative, so the
dialog has to build, answer, and record identically when the grab succeeds,
when it comes back empty, and when there is no parent at all. These tests
build all three and compare what the user can actually set.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer                              # noqa: E402
from PySide6.QtGui import QPixmap                              # noqa: E402
from PySide6.QtWidgets import (QApplication, QGraphicsView,    # noqa: E402
                               QLabel, QWidget)

from spacr.qt import setup_screen                              # noqa: E402
from spacr.qt.widgets import setup_dialog as sd                # noqa: E402


@pytest.fixture
def parent_window(qapp):
    """A visible-sized parent whose ``grab`` produces a real pixmap."""
    window = QWidget()
    window.resize(640, 480)
    try:
        yield window
    finally:
        window.deleteLater()


def _headings(dialog):
    return [label.text() for label in dialog.findChildren(QLabel)
            if label.text().startswith("<b>")]


# ---------------------------------------------------------------------------
# the backdrop
# ---------------------------------------------------------------------------

def test_a_parent_puts_a_blurred_still_behind_the_card(parent_window):
    """The backdrop is a frameless, scrollbar-free, click-through view."""
    dialog = sd.SetupDialog(parent_window)

    view = dialog._backdrop_view
    assert isinstance(view, QGraphicsView)
    assert view.frameShape() == QGraphicsView.NoFrame
    assert view.horizontalScrollBarPolicy() == view.verticalScrollBarPolicy()
    assert view.scene().items(), "the still was never put in the scene"
    assert view.scene().items()[0].graphicsEffect() is not None
    assert dialog.card.parent() is dialog
    dialog.deleteLater()


def test_the_blur_radius_is_the_documented_one(parent_window):
    """The decoration uses the module's single blur constant."""
    from PySide6.QtWidgets import QGraphicsBlurEffect

    dialog = sd.SetupDialog(parent_window)

    effect = dialog._backdrop_view.scene().items()[0].graphicsEffect()
    assert isinstance(effect, QGraphicsBlurEffect)
    assert effect.blurRadius() == pytest.approx(sd.BLUR)
    dialog.deleteLater()


def test_an_empty_grab_leaves_a_plain_dialog(parent_window, monkeypatch):
    """A null pixmap is a fine answer: the controls are unchanged."""
    monkeypatch.setattr(parent_window, "grab", lambda *a, **k: QPixmap())

    dialog = sd.SetupDialog(parent_window)

    assert dialog._backdrop_view is None
    assert dialog.answers() == sd.SetupDialog().answers()
    dialog.deleteLater()


def test_a_grab_that_throws_leaves_a_plain_dialog(parent_window, monkeypatch,
                                                  caplog):
    """A platform with no compositor still gets every control."""
    def no_compositor(*args, **kwargs):
        raise RuntimeError("no compositor here")

    monkeypatch.setattr(parent_window, "grab", no_compositor)

    with caplog.at_level(logging.DEBUG, logger="spacr.qt.setup_dialog"):
        dialog = sd.SetupDialog(parent_window)

    assert dialog._backdrop_view is None
    assert "no blurred backdrop on this platform" in caplog.text
    assert dialog.answers() == sd.SetupDialog().answers()
    dialog.deleteLater()


def test_resizing_keeps_the_card_inset_and_on_top(parent_window, qapp):
    """The card is laid out over the backdrop with a fixed margin."""
    dialog = sd.SetupDialog(parent_window)
    dialog.show()
    qapp.processEvents()

    dialog.resize(900, 700)
    qapp.processEvents()

    assert dialog._backdrop_view.geometry() == dialog.rect()
    assert dialog.card.geometry() == dialog.rect().adjusted(48, 48, -48, -48)
    dialog.close()
    dialog.deleteLater()


def test_resizing_a_plain_dialog_touches_no_backdrop(qapp):
    """Without a backdrop the card stays in the layout, not positioned."""
    dialog = sd.SetupDialog()
    dialog.show()
    qapp.processEvents()
    before = dialog.card.parent()

    dialog.resize(900, 700)                       # must not raise
    qapp.processEvents()

    assert dialog._backdrop_view is None
    assert dialog.card.parent() is before
    dialog.close()
    dialog.deleteLater()


# ---------------------------------------------------------------------------
# the groups
# ---------------------------------------------------------------------------

def test_a_group_whose_questions_all_left_is_not_drawn(qapp, monkeypatch):
    """An empty heading would read as a bug, so it is skipped entirely."""
    asked = [q for q in setup_screen.questions()
             if q[0] not in ("ai_provider", "ai_default")]
    monkeypatch.setattr(setup_screen, "questions", lambda: asked)

    dialog = sd.SetupDialog()

    headings = _headings(dialog)
    assert "<b>The assistant</b>" not in headings
    assert "<b>How it looks</b>" in headings
    assert "ai_provider" not in dialog._editors
    assert "ai_default" not in dialog._editors
    dialog.deleteLater()


def test_every_group_with_a_question_gets_its_heading(qapp):
    """Each non-empty group is announced once."""
    dialog = sd.SetupDialog()

    headings = _headings(dialog)
    asked = {q[0] for q in setup_screen.questions()}
    for heading, keys in sd.GROUPS:
        expected = any(key in asked for key in keys)
        assert (f"<b>{heading}</b>" in headings) is expected
    dialog.deleteLater()


def test_a_choice_with_no_matching_answer_falls_back_to_the_first(qapp):
    """An unrecognised saved value selects an option rather than nothing."""
    box = sd.SetupDialog._editor("theme", [("dark", "Dark"), ("light", "Light")],
                                 "mauve")

    assert box.currentIndex() == 0
    assert box.currentData() == "dark"


# ---------------------------------------------------------------------------
# accepting and dismissing
# ---------------------------------------------------------------------------

def test_a_refused_answer_is_logged_and_the_rest_are_still_recorded(
        qapp, monkeypatch, caplog):
    """One rejected preference does not discard the others or the setup mark."""
    monkeypatch.setattr(setup_screen, "apply",
                        lambda answers: ["language: not a language"])
    dialog = sd.SetupDialog()

    with caplog.at_level(logging.WARNING, logger="spacr.qt.setup_dialog"):
        dialog.accept()

    assert "some setup answers were refused" in caplog.text
    assert "not a language" in caplog.text
    assert setup_screen.answered_version() == setup_screen.current_version()
    assert dialog.result() == sd.SetupDialog.Accepted
    dialog.deleteLater()


def test_accepting_with_nothing_refused_logs_no_warning(qapp, monkeypatch,
                                                         caplog):
    """A clean apply is silent."""
    monkeypatch.setattr(setup_screen, "apply", lambda answers: [])
    dialog = sd.SetupDialog()

    with caplog.at_level(logging.WARNING, logger="spacr.qt.setup_dialog"):
        dialog.accept()

    assert "refused" not in caplog.text
    assert setup_screen.answered_version() == setup_screen.current_version()
    dialog.deleteLater()


# ---------------------------------------------------------------------------
# opening it at all
# ---------------------------------------------------------------------------

def test_a_pending_setup_opens_the_dialog_and_marks_it_answered(qapp,
                                                                monkeypatch):
    """When setup is due the dialog is built, shown, and recorded."""
    monkeypatch.setattr(setup_screen, "should_open", lambda *a, **k: True)
    seen = []
    tries = []

    def dismiss():
        found = QApplication.activeModalWidget()
        tries.append(found)
        if isinstance(found, sd.SetupDialog):
            seen.append(found)
            found.reject()
            return
        if len(tries) > 200:
            QApplication.instance().quit()
            return
        QTimer.singleShot(10, dismiss)

    QTimer.singleShot(0, dismiss)

    dialog = sd.open_setup_if_needed(None)

    assert dialog is not None
    assert seen == [dialog]
    assert dialog.result() == sd.SetupDialog.Rejected
    assert setup_screen.answered_version() == setup_screen.current_version()
    dialog.deleteLater()


def test_an_answered_setup_opens_nothing(qapp):
    """Once recorded for this version the dialog never appears again."""
    setup_screen.mark_answered(setup_screen.current_version())

    assert sd.open_setup_if_needed(None) is None
