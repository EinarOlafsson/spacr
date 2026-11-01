"""A close mark's square is never narrower than the button asks for.

With only DejaVu and Liberation installed (the CI runner's fonts, and many a
Linux desktop's), the ✕ buttons of the Layer viewer and of the Mask setting
chips asked for 26 x 21 px and were fixed at 22 x 22: the width was capped
at the height, so the mark was cut off. The square now takes the larger of
the two.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QToolButton

pytestmark = pytest.mark.qt


def test_a_wide_size_hint_widens_the_square(qtbot):
    from spacr.qt import theme

    button = QToolButton()
    qtbot.addWidget(button)
    button.setStyleSheet("QToolButton { padding: 0px 9px; }")
    theme.apply_close_mark(button)
    hint = button.sizeHint()
    assert hint.width() > hint.height(), "the padding did not widen the hint"
    assert button.width() >= hint.width(), (button.size(), hint)
    assert button.width() == button.height()


def test_an_ordinary_close_mark_keeps_its_hit_square(qtbot):
    from spacr.qt import theme

    button = QToolButton()
    qtbot.addWidget(button)
    theme.apply_close_mark(button)
    assert button.width() == button.height() >= theme.CLOSE_MARK_HIT_PX
    assert button.width() >= button.sizeHint().width()
