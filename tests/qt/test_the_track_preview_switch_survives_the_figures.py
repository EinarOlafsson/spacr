"""Mask's Track preview switch stays on screen once Plot figures appear.

Item 520. With Timelapse folded into Mask Generation, the fold brings the
``Track preview`` switch with it. Recording the Timelapse tutorial, the
switch collapsed out of view the moment the run's figures opened, so the
one control that shows what the tracking settings do could no longer be
reached.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication

from spacr.qt.screens import mask as mask_folds
from spacr.qt.screens.app_screen import AppScreen


def _settle(rounds=30):
    """Let queued layout and show events run."""
    for _ in range(rounds):
        QApplication.processEvents()


def _figure():
    """A small matplotlib figure, as a pipeline hands one to the screen."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    return fig


def _where(widget, screen):
    """``widget``'s rectangle in ``screen``'s coordinates."""
    return QRect(widget.mapTo(screen, QPoint(0, 0)), widget.size())


def _clipped_by_an_ancestor(widget, screen):
    """Whether any ancestor up to ``screen`` cuts ``widget`` off."""
    rect = _where(widget, screen)
    parent = widget.parentWidget()
    while parent is not None and parent is not screen:
        box = _where(parent, screen)
        if not box.contains(rect):
            return True
        parent = parent.parentWidget()
    return False


@pytest.fixture(params=[False, True], ids=["no-search-strip", "search-strip"])
def mask_with_tracking(request, qtbot, qt_theme_applied):
    """A shown Mask screen with the Timelapse fold switched on.

    Both with and without the settings search strip: in the app the strip
    is installed early, and it lives in the settings column, which folds
    away when a focus pane opens just as the Actions body does.
    """
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    if request.param:
        from spacr.qt import settings_search

        assert settings_search.install(screen) is not None
    strip = mask_folds.install_folds(screen)
    assert strip is not None
    screen.resize(1280, 800)
    screen.show()
    _settle()
    strip.button_for("timelapse").setChecked(True)
    _settle()
    preview = screen._folded_previews["timelapse"]
    return screen, preview


@pytest.mark.parametrize("size", [(1280, 800), (1024, 700)])
def test_the_track_preview_switch_is_visible_after_figures_appear(
        mask_with_tracking, size):
    """Non-zero size, inside the window, not cut off, and still clickable."""
    screen, preview = mask_with_tracking
    screen.resize(*size)
    _settle()
    toggle = preview.toggle
    assert toggle.isVisible()

    for _ in range(3):
        screen._on_figure_ready(_figure())
    _settle(60)

    assert screen._figures_card.isVisible()
    assert toggle.isVisible()
    assert toggle.width() > 0 and toggle.height() > 0
    assert toggle.width() >= toggle.minimumSizeHint().width()
    assert screen.rect().contains(_where(toggle, screen))
    assert not _clipped_by_an_ancestor(toggle, screen)

    assert screen._actions_heading_row.indexOf(toggle) >= 0

    toggle.click()
    assert toggle.isChecked()
    assert preview.card.isVisibleTo(preview.card.parentWidget())
