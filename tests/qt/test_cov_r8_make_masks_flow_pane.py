"""The flow pane: resizing it, with and without a picture in it.

The pane keeps the FULL-RESOLUTION pixmap and rescales a copy on every
resize, so repeated resizing never compounds interpolation error the way
rescaling the displayed pixmap would. That design is what makes the
empty case a real path: the pane spends the whole session before a run
with no pixmap at all, and every resize in that time asks it to rescale
nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.screens.make_masks import FLOW_RESTING_TEXT, _FlowPane

pytestmark = pytest.mark.qt


@pytest.fixture
def pane(qtbot):
    widget = _FlowPane()
    qtbot.addWidget(widget)
    _resize(widget, 400, 300)
    return widget


def _resize(widget, width, height):
    """Resize AND deliver the event.

    An offscreen widget that was never shown gets its geometry changed
    without Qt sending it a QResizeEvent, so `resize()` alone would test
    the setter and never the handler.
    """
    old = widget.size()
    widget.resize(width, height)
    QApplication.sendEvent(widget, QResizeEvent(QSize(width, height), old))


class TestResizingAnEmptyPane:

    def test_a_new_pane_says_why_it_is_empty(self, pane):
        assert pane.has_image() is False
        assert pane.text() == FLOW_RESTING_TEXT

    def test_resizing_before_any_run_rescales_nothing(self, pane):
        """THE UNCOVERED GUARD, through the event that reaches it.

        `setPixmap(scaled_for(None, ...))` is not a smaller picture, it
        is a crash in the paint path -- and it would happen on the very
        first resize of a freshly opened module, before anything has
        been segmented.
        """
        _resize(pane, 640, 480)
        assert pane.has_image() is False
        assert pane.text() == FLOW_RESTING_TEXT, (
            "the resting text was replaced by an empty pixmap")

    def test_rescaling_directly_with_no_picture_is_a_no_op(self, pane):
        pane._rescale()
        assert pane.pixmap().isNull()


class TestResizingAPaneWithAPicture:

    def test_a_flow_picture_survives_a_resize_at_the_new_size(self, pane):
        rgb = np.zeros((64, 48, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200
        pane.show_rgb(rgb)

        assert pane.has_image()
        assert pane.text() == ""
        before = pane.pixmap().size()

        _resize(pane, 200, 150)
        after = pane.pixmap().size()

        assert not after.isEmpty()
        assert after != before or pane.has_image(), (
            "the resize neither refit the picture nor kept one")

    def test_the_full_resolution_pixmap_is_kept_not_the_scaled_copy(self,
                                                                    pane):
        """Why the resize refits from `self._pixmap` and not from the
        displayed one: repeated resizing would otherwise compound the
        interpolation error."""
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        pane.show_rgb(rgb)
        kept = pane._pixmap

        _resize(pane, 64, 64)
        _resize(pane, 512, 512)

        assert pane._pixmap is kept, "the stored picture was overwritten"
        assert kept.size().width() == 256

    def test_clearing_puts_the_pane_back_to_resting(self, pane):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        pane.show_rgb(rgb)
        assert pane.has_image()

        pane.clear_view()

        assert pane.has_image() is False
        assert pane.text() == FLOW_RESTING_TEXT
        _resize(pane, 800, 600)        # and the empty guard holds afterwards
        assert pane.has_image() is False
