"""201: the outline setting takes the control its neighbours take.

    "in the cell tab in the settings the outline setting should look like the
     channel and normalize channel"

Three settings on one tab pick channels, and two of them did it with a
`ChannelPicker` while the third had a generic editor built from a choice
list. Two controls doing the same job in two shapes read as two different
jobs, and the user has to work out which is which every time.

IT WAS ONLY EVER THE WIDGET. `picture_settings.choices` has offered
`outline` the same r / g / b vocabulary as `normalize_channels` since they
were changed together, and `_as_channel_list` parses whatever either one
writes -- so this changes the control and not the setting.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def dialog(qtbot):
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    widget = PictureSettingsDialog(
        {"channels": "r,g,b", "normalize_channels": "b", "outline": "r,g"})
    qtbot.addWidget(widget)
    return widget


class TestTheThreeAreOneControl:

    def test_outline_is_the_same_class_as_the_other_two(self, dialog):
        from spacr.qt.widgets.channel_picker import ChannelPicker

        for key in ("channels", "normalize_channels", "outline"):
            assert isinstance(dialog._editors[key], ChannelPicker), key

    def test_they_offer_the_same_channels(self, dialog):
        """One source, so a change to how channels are discovered reaches all
        three rather than two of them."""
        shapes = {
            key: type(dialog._editors[key]).__name__
            for key in ("channels", "normalize_channels", "outline")
        }
        assert len(set(shapes.values())) == 1, shapes

    def test_outline_may_be_none_and_channels_may_not(self, dialog):
        """`channels` with nothing on is a blank picture. `outline` with
        nothing on is the default: no outline."""
        assert dialog._editors["outline"]._allow_none is True
        assert dialog._editors["normalize_channels"]._allow_none is True
        assert dialog._editors["channels"]._allow_none is False


class TestTheSettingIsUnchanged:
    """The control moved; the value must not."""

    def test_the_value_survives_the_round_trip(self, dialog):
        assert dialog.values()["outline"] == "r,g"

    def test_an_empty_outline_stays_empty(self, qtbot):
        from spacr.qt.widgets.picture_settings_dialog import (
            PictureSettingsDialog)

        widget = PictureSettingsDialog({"outline": ""})
        qtbot.addWidget(widget)

        assert not widget.values()["outline"]

    def test_what_it_writes_is_what_the_drawer_reads(self, qtbot):
        """The end of the contract: the picker's string has to survive the
        parser the crop drawer actually uses."""
        from spacr.picture_settings import _as_channel_list
        from spacr.qt.widgets.picture_settings_dialog import (
            PictureSettingsDialog)

        for written in ("r", "r,g", "r,g,b", ""):
            widget = PictureSettingsDialog({"outline": written})
            qtbot.addWidget(widget)
            assert (_as_channel_list(widget.values()["outline"])
                    == _as_channel_list(written)), written


class TestItIsListedWithTheOthers:

    def test_outline_is_one_of_the_channel_keys(self):
        from spacr.qt.widgets.picture_settings_dialog import CHANNEL_KEYS

        assert set(CHANNEL_KEYS) == {"channels", "normalize_channels",
                                     "outline"}
