"""Item 288: a Toggle's caption, wrapped or not, sizes and paints the same text.

``Toggle(word_wrap=True)`` wraps a long caption to the width it is given
and asks for the height the wrapped caption needs; a plain Toggle keeps
the native checkbox's answer. Both paint the caption beside the switch --
checked here by the ink that lands in the caption's area.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QCheckBox  # noqa: E402

from spacr.qt.widgets.toggle import Toggle  # noqa: E402

CAPTION = ("Measure every organelle slot that has a mask, including the "
           "ones added after the first run")


def _ink_right_of_the_switch(toggle):
    toggle.resize(180, toggle.heightForWidth(180))
    image = QImage(toggle.size(), QImage.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    toggle.render(image)
    start = toggle._track_x + toggle._track_w + toggle._label_gap
    return sum(1 for x in range(start, image.width())
               for y in range(image.height())
               if image.pixelColor(x, y).alpha() > 0)


def test_a_plain_toggle_keeps_the_native_height_for_width(qtbot):
    plain = Toggle(CAPTION)
    qtbot.addWidget(plain)
    native = QCheckBox(CAPTION)
    qtbot.addWidget(native)
    assert plain.heightForWidth(180) == native.heightForWidth(180)


def test_a_wrapping_toggle_grows_to_fit_its_caption_and_paints_it(qtbot):
    wrapped = Toggle(CAPTION, word_wrap=True)
    qtbot.addWidget(wrapped)
    one_line = wrapped.fontMetrics().height() + 4
    assert wrapped.heightForWidth(180) > one_line
    assert _ink_right_of_the_switch(wrapped) > 0
