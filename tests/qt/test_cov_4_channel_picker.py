"""The channel picker answers to the text accessors the settings panel uses.

Every editor in the settings panel is read and written through the same
duck-typed ``text``/``setText`` pair. A picker that only offered
``set_value`` would be silently skipped when a settings file was loaded, and
the toggles would keep whatever they were built with.
"""
from __future__ import annotations

from spacr.qt.widgets.channel_picker import ChannelPicker


def test_set_text_selects_the_channels_the_settings_file_names(qapp):
    """Loading a settings file must move the toggles, not just the value."""
    picker = ChannelPicker("", parent=None)
    picker.setText("b,r")
    assert picker.text() == "r,b"
    assert [name for name, box in picker._boxes.items() if box.isChecked()] \
        == ["r", "b"]


def test_set_text_announces_the_new_selection_once(qapp):
    """The panel learns the loaded value from one consolidated signal."""
    picker = ChannelPicker("g", parent=None)
    seen = []
    picker.changed.connect(seen.append)
    picker.setText("r,g,b")
    assert seen == ["r,g,b"], seen


def test_set_text_clears_a_selection_the_file_does_not_name(qapp):
    """A stale toggle left checked would report a channel nobody asked for."""
    picker = ChannelPicker("r,g,b", parent=None)
    picker.setText("g")
    assert picker.text() == "g"
