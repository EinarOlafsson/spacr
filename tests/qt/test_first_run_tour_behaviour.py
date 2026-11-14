"""The first-launch tour: the seen flag, the keys and buttons that end or
advance it, the sidebar sentence read from the registry, the menu lookup and
the highlight ring it paints over a real window.

Each test walks a case table with ``subtests``; every case asserts the exact
state the user would see (step caption, flag value, pixel colour).
"""
from __future__ import annotations

import builtins

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent
from PySide6.QtWidgets import QLabel, QMainWindow, QWidget

from spacr.qt import first_run as fr

_FALLBACK = ("Primary modules are grouped here by purpose; related workflows "
             "are reached from their host module.")


@pytest.fixture
def window(qapp):
    """A shown main window with a central widget, as the tour expects."""
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
    """The seen flag is persisted, so clear it around each test."""
    fr.reset_tour_state()
    yield
    fr.reset_tour_state()


def _key(key):
    """A plain key press for ``key``."""
    return QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier)


def _render(overlay):
    """Paint the overlay over a white ground and return the image."""
    image = QImage(overlay.width(), overlay.height(), QImage.Format_ARGB32)
    image.fill(QColor("white"))
    overlay.render(image)
    return image


def test_the_seen_flag_decides_whether_maybe_show_tour_shows_it(window, qapp,
                                                               subtests):
    """The tour shows once, a force overrides the flag, a reset re-arms it,
    and an active pathway walkthrough suppresses it even when forced."""
    with subtests.test(msg="first launch shows it, finishing retires it"):
        assert fr.was_tour_shown() is False
        first = fr.maybe_show_tour(window)
        assert type(first) is fr._TourOverlay
        assert first.isVisible() is True
        first._finish()
        qapp.processEvents()
        assert fr.was_tour_shown() is True
        assert fr.maybe_show_tour(window) is None

    with subtests.test(msg="force runs it after it was seen"):
        fr.mark_tour_seen()
        forced = fr.maybe_show_tour(window, force=True)
        assert type(forced) is fr._TourOverlay
        forced._finish()
        qapp.processEvents()
        assert fr.was_tour_shown() is True

    with subtests.test(msg="reset makes it fire again"):
        fr.mark_tour_seen()
        assert fr.was_tour_shown() is True
        fr.reset_tour_state()
        assert fr.was_tour_shown() is False
        again = fr.maybe_show_tour(window)
        assert type(again) is fr._TourOverlay
        again._finish()
        qapp.processEvents()

    with subtests.test(msg="a pathway walkthrough suppresses even a forced tour"):
        fr.reset_tour_state()
        window._pathway_walkthrough_active = True
        try:
            assert fr.maybe_show_tour(window, force=True) is None
            assert fr.maybe_show_tour(window) is None
        finally:
            window._pathway_walkthrough_active = False
        assert fr.was_tour_shown() is False


def test_keys_buttons_and_owners_drive_the_overlay(window, qapp, subtests):
    """Escape and Skip end the tour, Return/Enter advance it, other keys do
    nothing, an owner callback replaces the flag (even when it raises) and
    the overlay follows the window's size."""
    total = len(fr.DEFAULT_TOUR)
    # (case, presses, expected step caption after them or None when finished)
    keys = [
        ("escape ends it", [Qt.Key_Escape], None),
        ("return then enter advance two steps", [Qt.Key_Return, Qt.Key_Enter],
         f"Step 3 / {total}"),
        ("any other key leaves step one", [Qt.Key_A, Qt.Key_Space],
         f"Step 1 / {total}"),
    ]
    for case, presses, caption in keys:
        with subtests.test(msg=case):
            fr.reset_tour_state()
            overlay = fr.maybe_show_tour(window)
            for press in presses:
                overlay.keyPressEvent(_key(press))
            qapp.processEvents()
            if caption is None:
                assert fr.was_tour_shown() is True
                assert overlay.isVisible() is False
            else:
                assert overlay._step_lbl.text() == caption
                assert overlay.isVisible() is True
                assert fr.was_tour_shown() is False
                overlay._finish()
                qapp.processEvents()

    with subtests.test(msg="the skip button ends it"):
        fr.reset_tour_state()
        overlay = fr.maybe_show_tour(window)
        overlay._skip_btn.click()
        qapp.processEvents()
        assert overlay.isVisible() is False
        assert fr.was_tour_shown() is True

    with subtests.test(msg="an owner callback is called instead of the flag"):
        fr.reset_tour_state()
        finished = []
        overlay = fr._TourOverlay(window, fr.DEFAULT_TOUR[:1],
                                  on_finish=lambda: finished.append("done"))
        overlay.show()
        overlay._skip()
        qapp.processEvents()
        assert finished == ["done"]
        assert fr.was_tour_shown() is False

    with subtests.test(msg="an owner callback that raises still closes it"):
        tried = []

        def _explode():
            tried.append("tried")
            raise RuntimeError("the module screen is gone")

        overlay = fr._TourOverlay(window, fr.DEFAULT_TOUR[:1], on_finish=_explode)
        overlay.show()
        overlay._finish()
        qapp.processEvents()
        assert tried == ["tried"]
        assert overlay.isVisible() is False
        assert fr.was_tour_shown() is False

    with subtests.test(msg="the overlay follows a resize and a move"):
        overlay = fr.maybe_show_tour(window, force=True)
        for size, kind in (((1100, 700), QEvent.Resize), ((950, 640), QEvent.Move)):
            window.resize(*size)
            qapp.processEvents()
            assert overlay.eventFilter(window, QEvent(kind)) is False
            assert overlay.geometry() == window.rect()
        other = QWidget(window)
        overlay.setGeometry(0, 0, 10, 10)
        assert overlay.eventFilter(other, QEvent(QEvent.Resize)) is False
        assert overlay.size().width() == 10
        overlay._finish()
        qapp.processEvents()


def test_the_sidebar_sentence_lists_the_registry_sections(monkeypatch,
                                                          subtests):
    """One, two, three and duplicated sections are joined in registry order;
    an empty or unreadable registry falls back to a sentence that is true."""
    from spacr.qt import app as qt_app

    def _row(key, section):
        return (key, key.title(), "m", section)

    cases = [
        ("one section", [_row("mask", "Pipelines")],
         "Primary modules are grouped here into Pipelines; related workflows "
         "are reached from their host module."),
        ("two sections", [_row("mask", "Pipelines"), _row("ml", "Analysis")],
         "Primary modules are grouped here into Pipelines and Analysis; "
         "related workflows are reached from their host module."),
        ("three sections with a repeat",
         [_row("mask", "Pipelines"), _row("ml", "Analysis"),
          _row("measure", "Pipelines"), _row("seq", "Sequencing")],
         "Primary modules are grouped here into Pipelines, Analysis and "
         "Sequencing; related workflows are reached from their host module."),
        ("no sections", [], _FALLBACK),
    ]
    for case, apps, expected in cases:
        with subtests.test(msg=case):
            monkeypatch.setattr(qt_app, "APPS", apps, raising=False)
            assert fr._section_names_sentence() == expected

    monkeypatch.undo()

    with subtests.test(msg="the real registry"):
        said = fr._section_names_sentence()
        sections = list(dict.fromkeys(str(row[3]) for row in qt_app.APPS))
        assert len(sections) >= 2
        listed = ", ".join(sections[:-1]) + " and " + sections[-1]
        assert said == (f"Primary modules are grouped here into {listed}; "
                        "related workflows are reached from their host module.")

    with subtests.test(msg="an unreadable registry"):
        real_import = builtins.__import__

        def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
            if level and fromlist and "APPS" in fromlist:
                raise ImportError("no app registry")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        assert fr._section_names_sentence() == _FALLBACK


def test_find_menu_returns_the_live_menu_or_none(window, monkeypatch,
                                                 subtests):
    """The menu is matched by title without its ampersand; a missing bar, an
    unreadable bar or a missing title give None, a stale child is skipped."""
    bar = window.menuBar()
    file_menu = bar.addMenu("&File")
    demos = bar.addMenu("&Demos")

    class _NoBar:
        def menuBar(self):
            return None

    class _Hostile:
        def menuBar(self):
            raise RuntimeError("the window is already gone")

    cases = [
        ("found by title", window, "Demos", demos),
        ("another title", window, "File", file_menu),
        ("not there", window, "Help", None),
        ("ampersand is part of no title", window, "&Demos", None),
        ("no menu bar", _NoBar(), "Demos", None),
        ("unreadable bar", _Hostile(), "Demos", None),
    ]
    for case, owner, title, expected in cases:
        with subtests.test(msg=case):
            assert fr.find_menu(owner, title) is expected

    with subtests.test(msg="the old private name is the same function"):
        assert fr._find_menu is fr.find_menu

    with subtests.test(msg="a stale child is skipped"):
        class _Stale:
            def title(self):
                raise RuntimeError("Internal C++ object already deleted")

        monkeypatch.setattr(type(bar), "findChildren",
                            lambda self, kind: [_Stale(), demos])
        assert fr.find_menu(window, "Demos") is demos


def test_the_overlay_dims_the_window_and_rings_a_live_highlight(window, qapp,
                                                                subtests):
    """A live target keeps its own pixels inside a blue ring; a deleted,
    missing or raising highlight costs the ring but never the dimming."""
    import shiboken6

    target = QLabel("sidebar", window.centralWidget())
    target.setStyleSheet("background: #ffffff;")
    target.setGeometry(40, 40, 200, 120)
    target.show()
    qapp.processEvents()

    doomed = QWidget()
    doomed.setGeometry(0, 0, 50, 50)
    shiboken6.delete(doomed)

    asked = []

    def _explode(_window):
        asked.append("highlight")
        raise RuntimeError("the sidebar was deleted")

    # (case, highlight, whether a ring is cut at the target)
    cases = [
        ("a live widget is ringed", lambda w: target, True),
        ("a deleted widget is not", lambda w: doomed, False),
        ("no widget", lambda w: None, False),
        ("no highlight function", None, False),
        ("a highlight that raises", _explode, False),
    ]
    for case, highlight, ringed in cases:
        with subtests.test(msg=case):
            step = fr.TourStep(case, "body", highlight=highlight)
            overlay = fr._TourOverlay(window, [step])
            overlay.setGeometry(window.rect())
            overlay.show()
            qapp.processEvents()
            shot = _render(overlay)
            dimmed = QColor(shot.pixelColor(600, 300))
            assert dimmed.alpha() == 255 and dimmed.value() < 150
            inside = QColor(shot.pixelColor(140, 100))
            corner = QColor(shot.pixelColor(25, 25))
            if ringed:
                assert inside == QColor("white")
                ring = QColor(shot.pixelColor(140, 37))
                assert ring.blue() > ring.red()
            else:
                assert inside.value() < 150
                assert corner.value() < 150
            assert overlay._title_lbl.text() == case
            overlay._finish()
            qapp.processEvents()
    assert set(asked) == {"highlight"}
