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
    """The widget's own render as an (H, W, 3) RGB array.

    THE COPY IS THE POINT, AND WITHOUT IT THIS HELPER RETURNED A WINDOW ONTO
    FREED MEMORY. `constBits()` hands back a view of the QImage's own buffer;
    `image` is local, so it is destroyed as this function returns and Qt
    reuses what it was holding. The array then changes under whoever is
    reading it. Measured: an array taken from this helper and compared to a
    copy of ITSELF on the very next line already disagreed.

    That is what made `test_a_failed_paint_latches_nothing_and_the_next_one_draws`
    fail about a fifth of the time and pass when run alone -- it compares two
    of these, and whether the second grab lands on the first's memory is up
    to the allocator. It was recorded in 288 as "two identically built
    `ProviderMark` widgets render differently", which is not what was
    happening: the widgets always agreed, and the arrays did not own what
    they were showing.
    """
    image = widget.grab().toImage().convertToFormat(QImage.Format_RGB32)
    raw = np.frombuffer(memoryview(image.constBits()), dtype=np.uint8)
    rows = raw.reshape(image.height(), image.bytesPerLine() // 4, 4)
    return rows[:, :image.width(), :3][:, :, ::-1].copy()


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
