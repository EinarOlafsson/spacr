"""The first-launch tour: shown once, skippable, and never wrong about the app.

The tour narrates the real home screen, so the parts worth testing are the
ones that can drift away from it -- the section names read from the registry,
the Demos menu looked up on a live menu bar, the highlight ring drawn around
a real widget -- plus the promise that it fires exactly once.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent
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


def _render(overlay):
    """Paint the overlay over a white ground and hand back the pixels.

    White rather than the widget's own (transparent) background: the whole
    point of the paint is what it does to what is behind it.
    """
    image = QImage(overlay.width(), overlay.height(), QImage.Format_ARGB32)
    image.fill(QColor("white"))
    overlay.render(image)
    return image


@pytest.fixture(autouse=True)
def _forget_the_tour():
    fr.reset_tour_state()
    yield
    fr.reset_tour_state()


# --------------------------------------------------------------------------
# the seen flag
# --------------------------------------------------------------------------

def test_the_tour_is_shown_once_and_then_never_again(window, qapp):
    first = fr.maybe_show_tour(window)
    assert first is not None
    first._finish()
    qapp.processEvents()
    assert fr.was_tour_shown() is True
    assert fr.maybe_show_tour(window) is None


def test_a_forced_tour_runs_even_after_it_has_been_seen(window, qapp):
    fr.mark_tour_seen()
    overlay = fr.maybe_show_tour(window, force=True)
    assert overlay is not None
    overlay._finish()
    qapp.processEvents()


@pytest.mark.parametrize("stored,expected", [
    ("true", True), ("1", True), ("yes", True),
    ("false", False), ("0", False), ("", False),
])
def test_a_flag_stored_as_text_is_still_read_as_a_flag(stored, expected):
    """QSettings hands back strings on some platforms and bools on others;
    a string "true" that read as False would show the tour every launch."""
    fr._settings().setValue(fr._KEY_TOUR_SEEN, stored)
    assert fr.was_tour_shown() is expected


def test_resetting_the_state_makes_the_tour_fire_again(window):
    fr.mark_tour_seen()
    assert fr.was_tour_shown() is True
    fr.reset_tour_state()
    assert fr.was_tour_shown() is False


# --------------------------------------------------------------------------
# the sidebar sentence, read from the registry
# --------------------------------------------------------------------------

def test_the_sidebar_sentence_names_the_sections_the_sidebar_draws():
    """Hard-coding them is how this line came to advertise sections that had
    stopped existing."""
    from spacr.qt.app import APPS

    said = fr._section_names_sentence()
    sections = list(dict.fromkeys(str(row[3]) for row in APPS))
    assert sections
    for name in sections:
        assert name in said
    assert said.startswith("Primary modules are grouped here into ")
    assert said.endswith(
        "; related workflows are reached from their host module.")


def test_one_section_is_listed_without_an_and(monkeypatch):
    from spacr.qt import app as qt_app

    monkeypatch.setattr(qt_app, "APPS", [("mask", "Mask", "m", "Pipelines")],
                        raising=False)
    assert fr._section_names_sentence() == \
        "Primary modules are grouped here into Pipelines; related workflows " \
        "are reached from their host module."


def test_two_sections_are_joined_with_an_and(monkeypatch):
    from spacr.qt import app as qt_app

    monkeypatch.setattr(qt_app, "APPS", [
        ("mask", "Mask", "m", "Pipelines"),
        ("ml", "ML", "m", "Analysis")], raising=False)
    assert fr._section_names_sentence() == \
        "Primary modules are grouped here into Pipelines and Analysis; " \
        "related workflows are reached from their host module."


def test_a_registry_with_no_sections_falls_back_to_a_true_sentence(
        monkeypatch):
    from spacr.qt import app as qt_app

    monkeypatch.setattr(qt_app, "APPS", [], raising=False)
    assert fr._section_names_sentence() == \
        "Primary modules are grouped here by purpose; related workflows " \
        "are reached from their host module."


def test_a_registry_that_cannot_be_read_falls_back_rather_than_raising(
        monkeypatch):
    """The tour must not be the reason a window fails to open."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if level and fromlist and "APPS" in fromlist:
            raise ImportError("no app registry")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert fr._section_names_sentence() == \
        "Primary modules are grouped here by purpose; related workflows " \
        "are reached from their host module."


# --------------------------------------------------------------------------
# finding the Demos menu
# --------------------------------------------------------------------------

def test_the_demos_menu_is_found_by_its_title(window):
    menu = window.menuBar().addMenu("&Demos")
    assert fr.find_menu(window, "Demos") is menu
    assert fr._find_menu is fr.find_menu


def test_a_menu_that_is_not_there_is_not_invented(window):
    window.menuBar().addMenu("&File")
    assert fr.find_menu(window, "Demos") is None


def test_a_window_with_no_menu_bar_yields_no_menu():
    class NoBar:
        def menuBar(self):
            return None

    assert fr.find_menu(NoBar(), "Demos") is None


def test_a_menu_bar_that_cannot_be_read_yields_no_menu():
    class Hostile:
        def menuBar(self):
            raise RuntimeError("the window is already gone")

    assert fr.find_menu(Hostile(), "Demos") is None


def test_a_menu_deleted_underneath_the_search_is_skipped(window,
                                                         monkeypatch):
    """One stale child must not cost the search the menu that is still
    there."""
    real = window.menuBar().addMenu("&Demos")

    class Stale:
        def title(self):
            raise RuntimeError("Internal C++ object already deleted")

    bar = window.menuBar()
    monkeypatch.setattr(type(bar), "findChildren",
                        lambda self, kind: [Stale(), real])
    assert fr.find_menu(window, "Demos") is real


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


def test_escape_skips_the_tour_and_marks_it_seen(window, qapp):
    overlay = fr.maybe_show_tour(window)
    overlay.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape,
                                    Qt.NoModifier))
    qapp.processEvents()
    assert fr.was_tour_shown() is True


def test_enter_advances_to_the_next_step(window, qapp):
    overlay = fr.maybe_show_tour(window)
    overlay.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return,
                                    Qt.NoModifier))
    assert overlay._step_lbl.text().startswith("Step 2 /")
    overlay.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Enter,
                                    Qt.NoModifier))
    assert overlay._step_lbl.text().startswith("Step 3 /")
    overlay._finish()
    qapp.processEvents()


def test_any_other_key_leaves_the_tour_where_it_is(window, qapp):
    overlay = fr.maybe_show_tour(window)
    overlay.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
    assert overlay._step_lbl.text().startswith("Step 1 /")
    overlay._finish()
    qapp.processEvents()


def test_the_skip_button_ends_the_tour(window, qapp):
    overlay = fr.maybe_show_tour(window)
    overlay._skip_btn.click()
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


def test_an_owner_whose_callback_fails_still_gets_its_overlay_closed(window,
                                                                     qapp):
    def _explode():
        raise RuntimeError("the module screen is gone")

    overlay = fr._TourOverlay(window, fr.DEFAULT_TOUR[:1], on_finish=_explode)
    overlay.show()
    overlay._finish()
    qapp.processEvents()
    assert not overlay.isVisible()
    assert fr.was_tour_shown() is False


def test_the_overlay_follows_the_window_when_it_is_resized(window, qapp):
    overlay = fr.maybe_show_tour(window)
    window.resize(1100, 700)
    qapp.processEvents()
    overlay.eventFilter(window, QEvent(QEvent.Resize))
    assert overlay.size() == window.rect().size()
    overlay._finish()
    qapp.processEvents()


def test_the_card_stays_at_the_bottom_centre_on_a_resize(window, qapp):
    overlay = fr.maybe_show_tour(window)
    overlay.setGeometry(0, 0, 1000, 800)
    overlay.resizeEvent(None)
    card = overlay._card.geometry()
    assert abs(card.center().x() - 500) <= 1
    assert card.bottom() < 800
    overlay._finish()
    qapp.processEvents()


# --------------------------------------------------------------------------
# the highlight ring
# --------------------------------------------------------------------------

def test_a_highlighted_widget_gets_a_ring_and_keeps_its_own_colour(window,
                                                                   qapp):
    """The dimming is cleared inside the ring so the user sees the widget in
    its natural colour rather than a dark version of it."""
    target = QLabel("sidebar", window.centralWidget())
    target.setStyleSheet("background: #ffffff;")
    target.setGeometry(40, 40, 200, 120)
    target.show()
    qapp.processEvents()

    step = fr.TourStep("Sidebar", "body", highlight=lambda w: target)
    overlay = fr._TourOverlay(window, [step])
    overlay.setGeometry(window.rect())
    overlay.show()
    qapp.processEvents()
    shot = _render(overlay)

    inside = QColor(shot.pixelColor(140, 100))
    dimmed = QColor(shot.pixelColor(600, 300))
    assert inside.alpha() == 0                       # the dimming was cut out
    assert dimmed.alpha() == 255 and dimmed.value() < 150
    ring = QColor(shot.pixelColor(140, 37))
    assert ring.blue() > ring.red()
    overlay._finish()
    qapp.processEvents()


def test_a_widget_whose_c_plus_plus_half_is_gone_is_not_ringed(window, qapp):
    """A tour step can name a widget that has since been destroyed.

    A dialog closed, a screen rebuilt -- and the highlight function still
    hands back the wrapper. ``mapTo`` raises RuntimeError on it, the rect
    comes back None, and the tour dims the window without a ring. The
    alternative is an exception inside paintEvent, which is where Qt turns
    one into a crash rather than a traceback.

    Note it is the DELETED case that reaches this, not merely a widget in
    another window: mapTo only warns for that one and still returns a point.
    """
    import shiboken6
    from PySide6.QtWidgets import QWidget as _QWidget

    doomed = _QWidget()
    doomed.setGeometry(0, 0, 50, 50)
    shiboken6.delete(doomed)

    step = fr.TourStep("Gone", "body", highlight=lambda w: doomed)
    overlay = fr._TourOverlay(window, [step])
    overlay.setGeometry(window.rect())
    overlay.show()
    qapp.processEvents()
    shot = _render(overlay)

    for point in ((25, 25), (600, 300)):
        pixel = QColor(shot.pixelColor(*point))
        assert pixel.alpha() == 255 and pixel.value() < 150, (
            f"a ring was cut at {point} for a widget that no longer exists")

    overlay._finish()
    qapp.processEvents()


def test_a_step_whose_highlight_is_missing_still_dims_the_window(window,
                                                                 qapp):
    step = fr.TourStep("Nothing", "body", highlight=lambda w: None)
    overlay = fr._TourOverlay(window, [step])
    overlay.setGeometry(window.rect())
    overlay.show()
    qapp.processEvents()
    shot = _render(overlay)
    assert QColor(shot.pixelColor(600, 300)).value() < 150
    overlay._finish()
    qapp.processEvents()


def test_a_highlight_that_raises_costs_the_ring_not_the_tour(window, qapp):
    """The ring is lost, the step is not.

    The paint carries on past the highlight that could not be resolved, so
    the window is still dimmed and the step still says its piece -- which is
    what separates a swallowed highlight from a paint that stopped dead at
    it and left the tour invisible.
    """
    asked = []

    def _explode(_window):
        asked.append("highlight")
        raise RuntimeError("the sidebar was deleted")

    step = fr.TourStep("Broken", "body", highlight=_explode)
    overlay = fr._TourOverlay(window, [step])
    overlay.setGeometry(window.rect())
    overlay.show()
    qapp.processEvents()
    shot = _render(overlay)
    assert asked                                    # it really was tried
    assert QColor(shot.pixelColor(600, 300)).value() < 150
    assert "Broken" in {lb.text() for lb in overlay.findChildren(QLabel)}
    overlay._finish()
    qapp.processEvents()


def test_a_widgets_rectangle_is_reported_in_the_windows_own_coordinates(
        window, qapp):
    target = QLabel("x", window.centralWidget())
    target.setGeometry(30, 20, 100, 50)
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
