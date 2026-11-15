"""The picture window's cap and crop-source wiring, at its awkward edges.

Two of this dialog's controls are wired to LIVE: the montage cap re-prices
itself while the user drags its number, and the crop-source chooser re-greys
every other control the moment the mode changes. Both wires are guarded by a
type check, because either control can turn out to be a different widget than
the one the wiring expects -- a settings file that stored the cap as a decimal
builds a double spin box, and a screen that offers no crop-source choices
builds a text box. The guards exist so that an unexpected widget costs the
user a live update and not the whole window: a dialog that raises on the way
up is a dialog in which no picture setting can be changed at all.

The cost sentence beside the cap has the same shape of problem. It is written
onto the LABEL, over the help that is already there, so it has to know what
the help was; and it is invoked from three places -- the build, every mode
change, and every keystroke on the cap -- one of which can run before the help
has been recorded and one of which can run when there is no cap control at
all. Getting that wrong costs the reader either a crash or a tooltip that
grows a new copy of a 300-character sentence every time they try a number.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox, QLineEdit  # noqa: E402

import spacr.picture_settings as ps  # noqa: E402
from spacr.qt.widgets import picture_settings_dialog as psd  # noqa: E402

pytestmark = pytest.mark.qt

#: Every "<n> objects is ..." sentence a tooltip currently carries.
_COST = re.compile(r"([\d,]+) objects is")


def _costs(label) -> list:
    """The object counts the label's help currently prices, in order."""
    return _COST.findall(label.toolTip())


def _dialog(qtbot, **kwargs):
    made = psd.PictureSettingsDialog(mode=ps.LOAD_IMAGES, **kwargs)
    qtbot.addWidget(made)
    return made


class TestTheCapWiringSurvivesAnUnexpectedControl:
    def test_a_cap_of_zero_restores_the_unpriced_help(self, qtbot):
        """Zero objects has no page count, no memory, and no cut time.

        ``montage_cap_cost`` answers "" for a cap that is not a positive
        count. The caller must restore the setting's base help rather than
        leave the previous positive cap's price beside a box that now reads
        zero. The next positive count then prices itself as usual.
        """
        dialog = _dialog(qtbot)
        cap = dialog._editors["cap"]
        label = dialog._labels["cap"]

        assert ps.montage_cap_cost(0) == ""

        base = dialog._cap_help
        assert base and label.toolTip() != base
        cap.setValue(0)
        assert label.toolTip() == base
        assert _costs(label) == []
        assert "MB of crops held" not in label.toolTip()
        assert dialog.values()["cap"] == 0

        cap.setValue(5000)
        assert _costs(label) == ["5,000"]
        assert "MB of crops held" in label.toolTip()

    def test_the_price_can_be_written_before_the_help_was_ever_recorded(
            self, qtbot):
        """Trying three caps must leave one sentence, not three.

        The cost is appended to the label's help, so the help it is appended
        to has to be the ORIGINAL -- read back off the label each time, the
        second number is pasted after the first number's sentence and the
        third after both, and the reader who compared three caps ends up
        with a tooltip three paragraphs long saying three different things.
        The dialog therefore remembers the bare help the first time it needs
        it, including when it is asked to price a cap before any mode change
        has recorded one.
        """
        dialog = _dialog(qtbot)
        label = dialog._labels["cap"]

        # A label carrying plain help and a dialog that has not recorded it:
        # the state the cost line has to be able to recover from.
        label.setToolTip("Maximum objects drawn for one coefficient.")
        dialog._cap_help = None

        dialog._say_what_the_cap_costs()

        assert dialog._cap_help == "Maximum objects drawn for one coefficient."
        assert label.toolTip().startswith(
            "Maximum objects drawn for one coefficient.\n\n")
        assert _costs(label) == ["2,000"]

        for value in (3000, 4000, 7000):
            dialog._editors["cap"].setValue(value)

        assert _costs(label) == ["7,000"]
        assert label.toolTip().count("objects is") == 1
        assert label.toolTip().startswith(
            "Maximum objects drawn for one coefficient.\n\n")


class TestTheCropSourceWiringSurvivesAnUnexpectedControl:

    def test_a_crop_source_offered_no_modes_becomes_a_text_box(self, qtbot,
                                                               monkeypatch):
        """The greying follows the mode chosen HERE -- when it can.

        ``crop_source`` is one of this window's own controls as well as a
        toolbar control, so changing it here has to re-grey the settings the
        newly chosen mode does not use. That live re-greying is connected to
        a dropdown's index; a build in which ``offered_values`` names no
        modes falls back to a free-text control, which has no index to
        follow. The guard keeps that build to a window whose greying is
        merely fixed at its opening mode instead of a window that raises.
        """
        real = ps.offered_values

        def no_modes(key, source=None, frame=None):
            if str(key) == "crop_source":
                return ()
            return real(key, source=source, frame=frame)

        monkeypatch.setattr(ps, "offered_values", no_modes)
        typed = _dialog(qtbot)
        monkeypatch.setattr(ps, "offered_values", real)
        chosen = _dialog(qtbot)

        assert isinstance(typed._editors["crop_source"], QLineEdit)
        assert typed.values()["crop_source"] == ps.LOAD_IMAGES

        # Typing a streaming mode into the unwired box leaves the greying as
        # the window opened it: object_array belongs to the array route and
        # stays disabled for the PNG mode the dialog is still in.
        typed._editors["crop_source"].setText(ps.STREAM_IMAGES)
        assert typed.mode() == ps.LOAD_IMAGES
        assert typed._editors["object_array"].isEnabled() is False

        # The dropdown build is wired, so the same change there does re-grey.
        combo = chosen._editors["crop_source"]
        assert isinstance(combo, QComboBox)
        combo.setCurrentIndex(combo.findData(ps.STREAM_IMAGES))
        assert chosen.mode() == ps.STREAM_IMAGES
        assert chosen._editors["object_array"].isEnabled() is True
