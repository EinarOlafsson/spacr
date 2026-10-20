"""VoiceOver names the console topic bar and the Annotate screen (item 655).

2026-10-20 macOS tab tour on nightly: both take keyboard focus, and both
were read with no name.
"""
from PySide6.QtGui import QAccessible

from spacr.qt import i18n
from spacr.qt.widgets.console_panel import _TopicBar


def _name(widget):
    return QAccessible.queryAccessibleInterface(widget).text(QAccessible.Name)


def test_the_topic_bar_is_named_after_its_caption(qtbot):
    bar = _TopicBar("spaCR output — Measure")
    qtbot.addWidget(bar)
    assert _name(bar) == "spaCR output — Measure"


def test_the_annotate_screen_is_named(qtbot):
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    i18n.retranslate_widget_tree(screen)
    assert _name(screen).strip()
