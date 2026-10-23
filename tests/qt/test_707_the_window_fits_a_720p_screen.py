"""The main window and Make Masks fit a 1280 x 720 or 1366 x 768 screen (707).

The maintainer's 2026-10-18 native check: "positions look fine at both
resolutions and 1080p look perfect. but 720 cant fit the entire window on
screen." After the taskbar or menu bar the work area of those displays is
about 1280 x 680 and 1366 x 728. The window held a 1200 x 720 minimum, and
Make Masks wrapped TOOLS and IO onto so many lines beside its side column
that the bars ran out of room and the image shrank to a strip.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import (QAbstractButton, QApplication, QLabel,
                               QWidget)

from spacr.qt import app as app_mod

WORK_AREAS = [(1280, 680), (1366, 728)]


class _Screen:
    """A display whose work area is ``rect``."""

    def __init__(self, rect):
        self._rect = rect

    def availableGeometry(self):                     # noqa: N802 (Qt name)
        return QRect(self._rect)


@pytest.fixture
def folder(tmp_path):
    fields = tmp_path / "fields"
    fields.mkdir()
    for index in range(3):
        imageio.imwrite(fields / f"field_{index}.tif",
                        np.full((64, 64), index + 1, dtype=np.uint16))
    return fields


def _open_make_masks(qtbot, folder, size):
    window = app_mod.MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    window._on_nav_selected("make_masks")
    screen = window._stack.currentWidget()
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: screen._image_container.isVisible())
    qtbot.waitUntil(lambda: (window.width(), window.height()) == size)
    qtbot.wait(60)
    QApplication.processEvents()
    return window, screen


def _close(screen):
    screen._magnifier.close()
    screen.close_folded()


def _rect_in(window, widget):
    return QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())


def _whole(window, widget):
    rect = _rect_in(window, widget)
    return (widget.isVisible() and window.rect().contains(rect)
            and widget.visibleRegion().boundingRect() == widget.rect())


def test_the_main_window_minimum_fits_the_smallest_work_area(qtbot):
    window = app_mod.MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    minimum = window.minimumSize()
    assert minimum.width() <= 1200 and minimum.height() <= 640
    assert (window.width(), window.height()) == app_mod._MAIN_WINDOW_OPENING


@pytest.mark.parametrize("size", WORK_AREAS)
def test_make_masks_fits_with_every_group_and_the_image_whole(
        qtbot, qt_theme_applied, folder, size):
    from spacr.qt.preferences import get_folded_panels

    window, screen = _open_make_masks(qtbot, folder, size)
    width, height = size
    minimum = window.minimumSize()
    assert minimum.width() <= width and minimum.height() <= height
    hint = window.minimumSizeHint()
    assert hint.width() <= width and hint.height() <= height
    assert screen._view_pane.is_collapsed("Shortcuts")
    assert "make_masks/Shortcuts" not in get_folded_panels()
    groups = (screen._tools_group, screen._annotation_group,
              screen._io_group, screen._navigation_group)
    for group in groups:
        assert _whole(window, group), group.title_label.text()
        for control in group.findChildren(QAbstractButton):
            if control.isVisibleTo(group):
                assert _whole(window, control), (group.title_label.text(),
                                                 control.text())
    canvas = _rect_in(window, screen._canvas)
    assert _whole(window, screen._canvas)
    assert canvas.height() >= 120 and canvas.width() >= 600
    assert _whole(window, screen._image_footer)
    assert canvas.bottom() < _rect_in(window, screen._io_group).top()
    for label in screen._image_footer.findChildren(QLabel):
        if label.isVisibleTo(screen._image_footer):
            assert _whole(window, label)
    shots = os.environ.get("SPACR_TEST_SCREENSHOTS")
    if shots:
        path = Path(shots) / f"make-masks-{width}x{height}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        assert window.grab().save(str(path))
    _close(screen)


@pytest.mark.parametrize("language,size,font_scale", [
    pytest.param("en", WORK_AREAS[0], .9, id="en-small-font90"),
    pytest.param("en", WORK_AREAS[1], .9, id="en-medium-font90"),
    pytest.param("de", WORK_AREAS[0], .9, id="de-small-font90"),
    pytest.param("de", WORK_AREAS[1], .9, id="de-medium-font90"),
    pytest.param("de", WORK_AREAS[0], 2., id="de-small-font200"),
])
def test_overview_reset_ruler_and_active_tab_fit_at_ninety_percent(
        qtbot, qt_theme_applied, folder, monkeypatch, size, language, font_scale):
    from PySide6.QtWidgets import QToolButton
    from PySide6.QtGui import QTextDocument
    from spacr.qt import gui_scale, preferences, theme
    from spacr.qt.i18n import retranslate_widget_tree, tr

    previous = gui_scale.current_scale()
    previous_saved_gui = preferences.get_gui_scale()
    previous_font = preferences.get_font_scale()
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    preferences.set_gui_scale(.9)
    preferences.set_font_scale(font_scale)
    gui_scale.install_scaling_layer()
    gui_scale.set_gui_scale_live(.9)
    assert gui_scale.current_scale() == pytest.approx(.9)
    screen = None
    try:
        window, screen = _open_make_masks(qtbot, folder, size)
        preferences.apply_preferences_to_app(QApplication.instance())
        retranslate_widget_tree(window, language)
        QApplication.processEvents()
        assert (window.width(), window.height()) == size
        overview = screen.findChild(QToolButton, "MakeMasksFieldOverview")
        assert overview is not None
        caption = tr("Field-of-view overview")
        assert overview.accessibleName() == caption
        tooltip = QTextDocument()
        tooltip.setHtml(overview.toolTip())
        assert tooltip.toPlainText().splitlines()[-2:] == [caption, "API"]
        assert overview.toolButtonStyle() == Qt.ToolButtonIconOnly
        assert not overview.icon().isNull()
        assert screen._io_group.isAncestorOf(overview)
        assert _rect_in(window, screen._io_group).contains(_rect_in(window, overview))
        assert overview.font().pixelSize() == gui_scale.scale_int(theme.font_px("body", font_scale), .9)
        tabs = screen._view_tabs
        bar = tabs.tabBar()
        for index, caption in enumerate(("Mask", "Cell probability", "Flows")):
            assert tabs.tabText(index) == tr(caption, language=language)
        display = window.screen()
        print("707 geometry", QApplication.platformName(), language, size,
              "client", (window.width(), window.height()),
              "display", display.geometry().getRect(), "work", display.availableGeometry().getRect(),
              "DPR", display.devicePixelRatio(), "font_px", overview.font().pixelSize())
        buttons = (screen._btn_reset_zoom, screen._btn_ruler, overview)
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            QApplication.processEvents()
            assert (window.width(), window.height()) == size
            assert _whole(window, bar)
            selected = bar.tabRect(index)
            assert bar.rect().contains(selected), (language, size, index)
            selected = QRect(bar.mapTo(window, selected.topLeft()), selected.size())
            rectangles = []
            for button in buttons:
                assert _whole(window, button), (language, size, button.text())
                assert button.width() >= button.sizeHint().width()
                assert button.height() >= button.sizeHint().height()
                rect = _rect_in(window, button)
                assert not rect.intersects(selected)
                assert not any(rect.intersects(other) for other in rectangles)
                rectangles.append(rect)
            assert _whole(window, screen._image_footer)
            content = _rect_in(window, tabs.currentWidget())
            assert _rect_in(window, screen._image_footer).top() > content.bottom()
    finally:
        if screen is not None:
            _close(screen)
        preferences.set_gui_scale(previous_saved_gui)
        preferences.set_font_scale(previous_font)
        gui_scale.set_gui_scale_live(previous)


def test_a_side_column_opened_by_hand_stays_open_while_narrow(
        qtbot, qt_theme_applied, folder):
    window, screen = _open_make_masks(qtbot, folder, (1280, 680))
    pane = screen._view_pane
    assert pane.is_collapsed("Shortcuts")
    pane.toggle_pane("Shortcuts", by_user=True)
    assert not pane.is_collapsed("Shortcuts")
    window.resize(1300, 680)
    qtbot.waitUntil(lambda: window.width() == 1300)
    assert not pane.is_collapsed("Shortcuts")
    _close(screen)


def test_a_wide_screen_keeps_the_side_column_open(
        qtbot, qt_theme_applied, folder):
    window, screen = _open_make_masks(qtbot, folder, (1920, 1050))
    assert not screen._view_pane.is_collapsed("Shortcuts")
    assert screen._shortcut_panel.isVisible()
    window.resize(1280, 680)
    qtbot.waitUntil(lambda: screen._view_pane.is_collapsed("Shortcuts"))
    window.resize(1920, 1050)
    qtbot.waitUntil(
        lambda: not screen._view_pane.is_collapsed("Shortcuts"))
    _close(screen)


def test_a_folded_side_column_does_not_hold_the_screen_tall(
        qtbot, qt_theme_applied, folder):
    window, screen = _open_make_masks(qtbot, folder, (1280, 680))
    folded = screen.minimumSizeHint().height()
    screen._view_pane.toggle_pane("Shortcuts", by_user=True)
    qtbot.waitUntil(lambda: screen.minimumSizeHint().height() > folded,
                    timeout=2000)
    assert window.minimumSize().height() <= 640
    _close(screen)


@pytest.mark.parametrize("work", [QRect(0, 0, 1280, 680),
                                  QRect(0, 40, 1366, 728)])
def test_an_oversized_window_is_clamped_into_the_work_area(qtbot, work):
    window = QWidget()
    qtbot.addWidget(window)
    window.setMinimumSize(1200, 720)
    window.show()
    window.setGeometry(400, 300, 1920, 1080)
    assert app_mod._fit_to_the_work_area(window, _Screen(work))
    frame = window.frameGeometry()
    assert work.contains(frame)
    assert window.minimumSize().height() <= work.height()


def test_a_window_that_fits_is_left_where_it_is(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    window.setGeometry(100, 100, 1200, 640)
    before = window.geometry()
    assert not app_mod._fit_to_the_work_area(
        window, _Screen(QRect(0, 0, 1920, 1050)))
    assert window.geometry() == before


def test_an_unplaced_window_is_sized_but_left_for_the_window_manager(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(1200, 720)
    assert app_mod._fit_to_the_work_area(
        window, _Screen(QRect(0, 32, 1280, 680)))
    assert window.height() <= 680
    assert not window.testAttribute(Qt.WidgetAttribute.WA_Moved)


def test_a_framed_launch_on_a_720p_screen_fits(qtbot, qt_theme_applied,
                                               monkeypatch):
    from spacr.qt import preferences

    preferences.set_borderless_window(False)
    window = app_mod.MainWindow()
    qtbot.addWidget(window)
    work = QRect(0, 0, 1280, 680)
    monkeypatch.setattr(window, "screen", lambda: _Screen(work))
    app_mod._show_at_launch(window)
    assert window.isVisible() and not window.isMaximized()
    assert window.width() <= work.width()
    assert window.height() <= work.height()
    window.close()


def test_restoring_a_maximised_window_lands_inside_the_work_area(
        qtbot, qt_theme_applied, monkeypatch):
    window = app_mod.MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.setGeometry(200, 200, 1900, 1000)
    window.showMaximized()
    work = QRect(0, 0, 1280, 680)
    monkeypatch.setattr(window, "screen", lambda: _Screen(work))
    window._toggle_maximised()
    qtbot.waitUntil(lambda: not window.isMaximized())
    assert work.contains(window.frameGeometry())


def _module_keys():
    return [entry[0] for entry in app_mod.APPS] + ["make_masks"]


@pytest.mark.parametrize("size", WORK_AREAS)
def test_every_module_screen_fits_without_being_squeezed(
        qtbot, qt_theme_applied, size):
    """A page whose minimum exceeds the stack has its controls cut off."""
    window = app_mod.MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    qtbot.waitUntil(lambda: (window.width(), window.height()) == size)
    squeezed = {}
    for key in _module_keys():
        window._on_nav_selected(key)
        qtbot.wait(20)
        QApplication.processEvents()
        page = window._stack.currentWidget()
        stack = window._stack.size()
        hint = page.minimumSizeHint()
        if hint.width() > stack.width() or hint.height() > stack.height():
            squeezed[key] = (hint.toTuple(), stack.toTuple())
        scroll = page.findChild(app_mod._CrampedScroll)
        if scroll is not None:
            content = scroll.widget()
            need = content.minimumSizeHint()
            assert content.width() >= need.width(), key
            assert content.height() >= need.height(), key
    assert not squeezed
    window.close()


def _geometry(page):
    skip = {"CrampedScreenScroll", "CrampedScreenContent",
            "qt_scrollarea_viewport", "qt_scrollarea_hcontainer",
            "qt_scrollarea_vcontainer"}
    rows = []
    for widget in page.findChildren(QWidget):
        if widget.objectName() in skip or not widget.isVisible():
            continue
        rows.append((type(widget).__name__, widget.objectName(),
                     QRect(widget.mapTo(page, QPoint(0, 0)),
                           widget.size()).getRect()))
    return sorted(rows)


@pytest.mark.parametrize("key", sorted(app_mod._CRAMPED_SCREENS))
def test_a_roomy_screen_lays_out_exactly_as_before_it_scrolled(
        qtbot, qt_theme_applied, key):
    """At 1920 x 1050 the wrapper changes no widget's place or size."""
    window = app_mod.MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    page = window._build_screen_timed(key)
    host = QWidget()
    qtbot.addWidget(host)
    page.setParent(host)
    host.resize(1920, 995)
    page.setGeometry(0, 0, 1920, 995)
    host.show()
    qtbot.wait(60)
    QApplication.processEvents()
    before = _geometry(page)
    assert app_mod._scroll_when_cramped(page)
    qtbot.wait(60)
    QApplication.processEvents()
    scroll = page.findChild(app_mod._CrampedScroll)
    assert scroll.widget().size() == page.size()
    assert not scroll.horizontalScrollBar().isVisible()
    assert not scroll.verticalScrollBar().isVisible()
    assert _geometry(page) == before
    host.close()


_TESTED_ROOT = Path(__file__).resolve().parents[2]


def _tested_tree(env):
    """Make a fresh probe process import this tree's spacr, not an install.

    :param env: the probe's environment, changed in place.
    :returns: ``env``.
    """
    env["SPACR_TESTED_ROOT"] = str(_TESTED_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_TESTED_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


GUARD = 'import os as _os, spacr as _spacr\nassert _os.path.realpath(_spacr.__file__).startswith(_os.path.realpath(_os.environ["SPACR_TESTED_ROOT"]) + _os.sep), ("probe imported spacr from " + _spacr.__file__)\n'


_DPR_SCRIPT = r"""
import sys
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
app = QApplication([])
from spacr.qt import app as app_mod
page = QWidget()
layout = QVBoxLayout(page)
tall = QWidget()
tall.setMinimumSize(900, 1200)
layout.addWidget(tall)
app_mod._scroll_when_cramped(page)
page.resize(1280, 625)
page.show()
for _ in range(5):
    app.processEvents()
scroll = page.findChild(app_mod._CrampedScroll)
small = (scroll.verticalScrollBar().isVisible(),
         scroll.widget().height() >= 1200)
page.resize(1920, 995)
for _ in range(5):
    app.processEvents()
roomy = (scroll.verticalScrollBar().isVisible(),
         scroll.widget().size() == page.size())
print(page.devicePixelRatioF(), small, roomy)
"""


def test_the_threshold_is_in_logical_pixels_at_device_pixel_ratio_two():
    import subprocess
    import sys

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR="2",
               CUDA_VISIBLE_DEVICES="")
    done = subprocess.run([sys.executable, "-c", GUARD + _DPR_SCRIPT], env=_tested_tree(env),
                          capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr[-2000:]
    line = done.stdout.strip().splitlines()[-1]
    assert line == "2.0 (True, True) (False, True)", line
