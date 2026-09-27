"""The orthogonal view's wheel passes on scrolls it cannot use for zooming.

Pinned behaviour of :meth:`spacr.qt.ortho_view.OrthoView.wheelEvent`:

* a wheel event with no vertical movement (a purely horizontal scroll)
  leaves the zoom alone and is ignored, so a parent can scroll with it;
* a wheel event whose pointer is over none of the three panels leaves the
  zoom alone and is ignored too;
* with no volume shown, the wheel is ignored.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402

from spacr.layers import LayerStack  # noqa: E402
from spacr.qt import ortho_view as ov  # noqa: E402

pytestmark = pytest.mark.qt


def _view(qtbot):
    stack = LayerStack()
    stack.add_image(np.zeros((10, 64, 64), np.uint16), name="volume")
    view = ov.OrthoView(stack, width=128)
    qtbot.addWidget(view)
    view.resize(520, 520)
    return view


def _wheel(view, local, delta):
    event = QWheelEvent(
        QPointF(local), QPointF(view.mapToGlobal(local)), QPoint(0, 0),
        delta, Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    event.accept()
    view.wheelEvent(event)
    return event


def test_a_horizontal_scroll_is_passed_on_without_zooming(qtbot):
    view = _view(qtbot)
    before = view.views.scale

    event = _wheel(view, QPoint(10, 10), QPoint(120, 0))

    assert not event.isAccepted()
    assert view.views.scale == pytest.approx(before)


def test_a_scroll_over_none_of_the_panels_is_passed_on(qtbot):
    view = _view(qtbot)
    before = view.views.scale

    event = _wheel(view, QPoint(-500, -500), QPoint(0, 120))

    assert not event.isAccepted()
    assert view.views.scale == pytest.approx(before)


def test_a_scroll_with_no_volume_shown_is_passed_on(qtbot):
    view = ov.OrthoView(LayerStack(), width=128)
    qtbot.addWidget(view)
    assert view.views is None

    event = _wheel(view, QPoint(10, 10), QPoint(0, 120))

    assert not event.isAccepted()
    assert view.views is None
