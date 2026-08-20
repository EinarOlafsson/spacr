"""188 B: R, G and B as three toggles instead of a dropdown of combinations.

"for the channels and normalize channels i liked the r,g,b system better,
instead of the dropdown you made. i want to you to bring back the r,g,b
system, it works great in the annotation app."

WHAT THE DROPDOWN COST. It offered the eight combinations as eight rows, so
reading which channels were on meant opening it, and turning one channel off
-- the thing a user does constantly here -- took two clicks and a search
through a list whose order means nothing.

NOT THE ANNOTATION APP'S CONTROL EXACTLY. That one is a QLineEdit holding
``r,g,b``, and a text box accepts ``rgb``, ``R,G,B``, ``red`` and a trailing
comma -- spellings instruction 176 ("one channel vocabulary") exists because
they reached the run meaning different things. Three checkboxes are the same
letters, the same stored value and the same directness, with no spelling to
get wrong.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.channel_picker import CHANNELS, ChannelPicker, parse, to_text


class TestReadingWhatIsStored:
    """The stored value is unchanged, so old settings files still mean what
    they meant and `filter_channels_pil` reads what it always read."""

    @pytest.mark.parametrize("value,expected", [
        ("r,g,b", ("r", "g", "b")),
        (["r", "b"], ("r", "b")),
        ("b,r", ("r", "b")),           # canonical order, not typed order
        ("B, R", ("r", "b")),          # case and spaces
        ("", ()),
        (None, ()),
    ])
    def test_it_reads_the_forms_that_are_actually_on_disk(self, value,
                                                          expected):
        assert parse(value) == expected

    @pytest.mark.parametrize("value", ["rgb", "red", "x", 3, 4.5])
    def test_an_unrecognised_spelling_is_dropped_not_guessed(self, value):
        """A channel nobody asked for is worse than one missing: it changes
        what the picture shows without saying so."""
        assert parse(value) == ()

    def test_the_stored_form_is_the_canonical_order(self):
        assert to_text(["b", "r"]) == "r,b"
        assert to_text([]) == ""


class TestTheToggles:

    def test_it_opens_showing_what_it_was_given(self, qtbot):
        picker = ChannelPicker("r,b")
        qtbot.addWidget(picker)

        assert picker.value() == "r,b"
        assert picker._boxes["r"].isChecked()
        assert not picker._boxes["g"].isChecked()

    def test_turning_one_off_is_one_click(self, qtbot):
        picker = ChannelPicker("r,g,b")
        qtbot.addWidget(picker)

        picker._boxes["g"].setChecked(False)

        assert picker.value() == "r,b"

    def test_it_announces_the_change(self, qtbot):
        picker = ChannelPicker("r,g,b")
        qtbot.addWidget(picker)
        seen = []
        picker.changed.connect(seen.append)

        picker._boxes["b"].setChecked(False)

        assert seen == ["r,g"]

    def test_setting_a_value_does_not_announce_once_per_box(self, qtbot):
        picker = ChannelPicker("")
        qtbot.addWidget(picker)
        seen = []
        picker.changed.connect(seen.append)

        picker.set_value("r,g,b")

        assert seen == ["r,g,b"], f"one change, not three: {seen}"


class TestWhatMayBeEmpty:

    def test_channels_cannot_be_cleared_to_a_blank_picture(self, qtbot):
        picker = ChannelPicker("r", allow_none=False)
        qtbot.addWidget(picker)

        picker._boxes["r"].setChecked(False)

        assert picker.value() == "r", "the last channel is put back"
        assert picker._boxes["r"].isChecked()

    def test_putting_it_back_does_not_announce_a_value_nobody_chose(self,
                                                                   qtbot):
        picker = ChannelPicker("r", allow_none=False)
        qtbot.addWidget(picker)
        seen = []
        picker.changed.connect(seen.append)

        picker._boxes["r"].setChecked(False)

        assert seen == [], f"the refusal must be silent: {seen}"

    def test_normalising_nothing_is_a_real_answer(self, qtbot):
        picker = ChannelPicker("b", allow_none=True)
        qtbot.addWidget(picker)

        picker._boxes["b"].setChecked(False)

        assert picker.value() == ""


class TestItIsWhatTheDialogUses:

    @pytest.fixture
    def dialog(self, qtbot):
        from spacr.qt.widgets.picture_settings_dialog import (
            PictureSettingsDialog)

        widget = PictureSettingsDialog({"channels": "r,g",
                                        "normalize_channels": "b"})
        qtbot.addWidget(widget)
        return widget

    @pytest.mark.parametrize("key", ["channels", "normalize_channels"])
    def test_both_channel_settings_use_it(self, dialog, key):
        assert isinstance(dialog._editors[key], ChannelPicker)

    def test_the_dialog_reads_the_comma_string_back(self, dialog):
        assert dialog.values()["channels"] == "r,g"
        assert dialog.values()["normalize_channels"] == "b"

    def test_a_change_reaches_the_dialogs_values(self, dialog):
        dialog._editors["channels"]._boxes["b"].setChecked(True)

        assert dialog.values()["channels"] == "r,g,b"

    def test_channels_is_the_one_that_may_not_be_emptied(self, dialog):
        for name in CHANNELS:
            dialog._editors["channels"]._boxes[name].setChecked(False)
            dialog._editors["normalize_channels"]._boxes[name].setChecked(
                False)

        assert dialog.values()["channels"]
        assert dialog.values()["normalize_channels"] == ""
