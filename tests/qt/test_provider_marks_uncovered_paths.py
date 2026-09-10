"""A provider mark whose decoration cannot be drawn is still a control.

The mark paints a vendor logo, a halo and a status line. None of that is
load-bearing: the widget's job is to answer "which assistant do you want",
and it answers that when it is clicked. So a painter that throws -- a palette
key that moved, a font that will not load -- must cost the drawing and
nothing else.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QMouseEvent  # noqa: E402

from spacr.qt.widgets.provider_marks import ProviderMark  # noqa: E402

pytestmark = pytest.mark.qt


def _left_click():
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(4, 4),
                       Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


def _pixels(widget) -> np.ndarray:
    """The widget's own render as an (H, W, 3) RGB array."""
    image = widget.grab().toImage().convertToFormat(QImage.Format_RGB32)
    raw = np.frombuffer(memoryview(image.constBits()), dtype=np.uint8)
    rows = raw.reshape(image.height(), image.bytesPerLine() // 4, 4)
    return rows[:, :image.width(), :3][:, :, ::-1]


def test_a_mark_that_cannot_paint_still_renders_and_still_answers(qapp):
    """The failure must stop at the paint: a mark that took the setup screen
    down with it would cost the user the whole choice, not one drawing."""
    mark = ProviderMark("claude", "Claude")
    mark.resize(88, 92)

    def refuse():
        raise KeyError("accent")

    mark._paint = refuse

    # The grab still completes: nothing escapes paintEvent into Qt.
    pixels = _pixels(mark)
    assert pixels.shape == (92, 88, 3)

    chosen = []
    mark.chosen.connect(chosen.append)
    mark.mousePressEvent(_left_click())
    assert chosen == ["claude"]
    mark.deleteLater()


def test_a_failed_paint_latches_nothing_and_the_next_one_draws(qapp):
    """The swallow is per event. A mark that stopped drawing for good after
    one bad frame would be a blank card the user cannot get back."""
    mark = ProviderMark("gpt", "GPT")
    mark.resize(88, 92)
    mark._paint = lambda: (_ for _ in ()).throw(RuntimeError("no painter"))
    _pixels(mark)

    del mark._paint
    recovered = _pixels(mark)
    assert recovered.std() > 0, "the mark never came back after a failed paint"

    fresh = ProviderMark("gpt", "GPT")
    fresh.resize(88, 92)
    assert np.array_equal(recovered, _pixels(fresh))
    mark.deleteLater()
    fresh.deleteLater()
