"""The first-launch tour: shown once, skippable, and never wrong about the app.

The tour narrates the real home screen, so the parts worth testing are the
ones that can drift away from it -- the section names read from the registry,
the Demos menu looked up on a live menu bar, the highlight ring drawn around
a real widget -- plus the promise that it fires exactly once.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QLabel, QMainWindow, QWidget

from spacr.qt import first_run as fr


@pytest.fixture
def window(qapp):
    win = QMainWindow()
    win.resize(900, 600)
    win.setCentralWidget(QWidget(win))
    win.show()
    qapp.processEvents()
    yield win
    win.close()
    win.deleteLater()


@pytest.fixture(autouse=True)
def _forget_the_tour():
    fr.reset_tour_state()
    yield
    fr.reset_tour_state()


# --------------------------------------------------------------------------
# the seen flag
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stored,expected", [
    ("true", True), ("1", True), ("yes", True),
    ("false", False), ("0", False), ("", False),
])
def test_a_flag_stored_as_text_is_still_read_as_a_flag(stored, expected):
    """QSettings hands back strings on some platforms and bools on others;
    a string "true" that read as False would show the tour every launch."""
    fr._settings().setValue(fr._KEY_TOUR_SEEN, stored)
    assert fr.was_tour_shown() is expected


# --------------------------------------------------------------------------
# the overlay
# --------------------------------------------------------------------------

def test_the_card_counts_the_steps_and_ends_on_finish(window, qapp):
    overlay = fr.maybe_show_tour(window)
    total = len(fr.DEFAULT_TOUR)
    assert overlay._step_lbl.text() == f"Step 1 / {total}"
    assert overlay._title_lbl.text() == fr.DEFAULT_TOUR[0].title
    for step in range(2, total + 1):
        overlay._next()
        assert overlay._step_lbl.text() == f"Step {step} / {total}"
    assert overlay._next_btn.text() == "Finish"
    overlay._next()
    qapp.processEvents()
    assert fr.was_tour_shown() is True


def test_a_borrowed_overlay_reports_to_its_owner_not_the_app_flag(window,
                                                                  qapp):
    """`walkthrough` reuses this overlay for a per-module tour; finishing one
    of those must not retire the app-wide first-run tour."""
    finished = []
    overlay = fr._TourOverlay(window, fr.DEFAULT_TOUR[:1],
                              on_finish=lambda: finished.append(True))
    overlay.show()
    overlay._skip()
    qapp.processEvents()
    assert finished == [True]
    assert fr.was_tour_shown() is False


def test_the_card_stays_at_the_bottom_centre_on_a_resize(window, qapp):
    overlay = fr.maybe_show_tour(window)
    overlay.setGeometry(0, 0, 1000, 800)
    overlay.resizeEvent(None)
    card = overlay._card.geometry()
    from spacr.qt.hidpi import screen_for_widget
    available = screen_for_widget(overlay).availableGeometry()
    visible = overlay.rect().intersected(QRect(
        overlay.mapFromGlobal(available.topLeft()), available.size()))
    assert abs(card.center().x() - visible.center().x()) <= 1
    assert visible.contains(card)
    overlay._finish()
    qapp.processEvents()


# --------------------------------------------------------------------------
# the highlight ring
# --------------------------------------------------------------------------


def test_a_widgets_rectangle_is_reported_in_the_windows_own_coordinates(
        window, qapp):
    target = QLabel("x", window.centralWidget())
    target.setGeometry(30, 20, 100, 50)
    target.show()
    qapp.processEvents()
    rect = fr._widget_rect_in_window(target, window)
    assert rect is not None
    assert rect.size() == target.size()
    assert rect.topLeft() == target.mapTo(window, QPoint(0, 0))


def test_a_widget_that_cannot_be_mapped_has_no_rectangle(window):
    class Detached:
        def mapTo(self, *_args):
            raise RuntimeError("not a child of that window")

    assert fr._widget_rect_in_window(Detached(), window) is None


def test_the_button_styles_are_distinct_enough_to_tell_apart():
    assert "transparent" in fr._ghost_btn_qss()
    assert "#4A9EFF" in fr._primary_btn_qss()
    assert fr._ghost_btn_qss() != fr._primary_btn_qss()
