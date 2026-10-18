"""The empty state's wrapped label re-measures itself when its font changes."""
from PySide6.QtCore import QEvent
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets.empty_state import _WrappedLabel


SENTENCE = ("Drop a folder of images anywhere on this window or type a path "
            "into the Source field below.")


def test_a_larger_font_grows_the_fixed_height(qtbot):
    label = _WrappedLabel(SENTENCE, wrap_width=200)
    qtbot.addWidget(label)
    before = label.height()
    font = QFont(label.font())
    font.setPointSizeF(font.pointSizeF() * 2)
    label.setFont(font)
    assert label.height() > before
    assert label.height() == label._wrapped_height()
    assert label.width() == 200


def test_a_style_change_with_the_same_font_keeps_the_height(qtbot):
    label = _WrappedLabel(SENTENCE, wrap_width=200)
    qtbot.addWidget(label)
    before = label.height()
    QApplication.sendEvent(label, QEvent(QEvent.StyleChange))
    assert label.height() == before == label._wrapped_height()
    QApplication.sendEvent(label, QEvent(QEvent.EnabledChange))
    assert label.height() == before
