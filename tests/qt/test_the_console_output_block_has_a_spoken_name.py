"""The console's output block has an accessible name for VoiceOver.

2026-10-19 macOS accessibility audit: across all 54 modules the only focusable
control without an accessible name was this block, in modules whose console
held output.
"""
from PySide6.QtGui import QAccessible

from spacr.qt.widgets.console_panel import _StdoutBlock


def test_output_and_error_blocks_are_named(qtbot):
    for error, name in ((False, "Log"), (True, "Error")):
        block = _StdoutBlock("some output", error=error)
        qtbot.addWidget(block)
        assert block.accessibleName() == name
        assert QAccessible.queryAccessibleInterface(block).text(QAccessible.Name) == name
