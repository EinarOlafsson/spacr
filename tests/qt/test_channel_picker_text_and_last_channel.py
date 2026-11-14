"""The channel picker's text accessors and its last-channel correction.

The settings form writes every editor through `setText`, so loading a file
must move the toggles and announce the new selection once.  A picker that
may not go blank puts the last unticked channel back silently; called with
no Toggle as sender it has nothing to correct and announces the value.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.channel_picker import ChannelPicker  # noqa: E402

pytestmark = pytest.mark.qt


def _checked(picker):
    return [name for name, box in picker._boxes.items() if box.isChecked()]


def test_the_channel_picker_reads_writes_and_keeps_a_channel(qtbot, subtests):
    """setText/text round trips, one change signal, and the allow_none=False correction."""
    for case, start, written, text, checked in (
            ("selects the channels a file names, in rgb order", "", "b,r",
             "r,b", ["r", "b"]),
            ("all three", "g", "r,g,b", "r,g,b", ["r", "g", "b"]),
            ("clears a selection the file does not name", "r,g,b", "g", "g",
             ["g"]),
    ):
        with subtests.test(msg=case):
            picker = ChannelPicker(start, parent=None)
            qtbot.addWidget(picker)
            seen = []
            picker.changed.connect(seen.append)
            picker.setText(written)
            assert (picker.text(), _checked(picker), seen) == (text, checked,
                                                              [text])

    for case, make in (
            ("built with allow_none=False", lambda: ChannelPicker(
                "r", allow_none=False)),
            ("allow_none switched off later", lambda: ChannelPicker("r")),
    ):
        with subtests.test(msg=f"the last channel is put back silently: {case}"):
            picker = make()
            qtbot.addWidget(picker)
            picker._allow_none = False
            heard = []
            picker.changed.connect(heard.append)
            picker._boxes["r"].setChecked(False)
            assert (picker.value(), _checked(picker), heard) == ("r", ["r"], [])

    with subtests.test(msg="called with no sender, the empty value is announced"):
        picker = ChannelPicker("")
        qtbot.addWidget(picker)
        picker._allow_none = False
        heard = []
        picker.changed.connect(heard.append)
        picker._on_toggled(False)
        assert (picker.value(), heard) == ("", [""])
