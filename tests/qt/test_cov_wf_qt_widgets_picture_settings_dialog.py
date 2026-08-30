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

from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox,  # noqa: E402
                               QLineEdit, QSpinBox)

import spacr.picture_settings as ps                        # noqa: E402
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

    def test_a_cap_stored_as_a_decimal_still_prices_itself_once(self, qtbot):
        """A settings file that wrote ``cap: 2000.0`` must still open.

        The cap arrives from whatever is on disk, and a CSV or a JSON blob
        round-tripped through a float column hands this dialog a decimal. The
        control built for a decimal is a QDoubleSpinBox, which is not the
        QSpinBox the live re-pricing connects to -- so the guard on that
        connection is what stands between the user and a window that will not
        open. What they lose is the live update, and nothing else: the cap
        still prices itself once, at build time, off the value they stored.
        """
        decimal = _dialog(qtbot, values={"cap": 12.5})
        whole = _dialog(qtbot)

        assert isinstance(decimal._editors["cap"], QDoubleSpinBox)
        assert not isinstance(decimal._editors["cap"], QSpinBox)
        # Priced at build time from the stored value, truncated to a count.
        assert _costs(decimal._labels["cap"]) == ["12"]

        # The decimal control is NOT wired, so dragging it changes nothing...
        settled = decimal._labels["cap"].toolTip()
        decimal._editors["cap"].setValue(30.0)
        assert decimal._labels["cap"].toolTip() == settled
        assert decimal.values()["cap"] == pytest.approx(30.0)

        # ...while the ordinary whole-number control is, which is what makes
        # the frozen tooltip above a property of the widget type and not of
        # the dialog having no wiring at all.
        assert isinstance(whole._editors["cap"], QSpinBox)
        whole._editors["cap"].setValue(30)
        assert _costs(whole._labels["cap"]) == ["30"]

    def test_a_cap_of_zero_is_priced_as_nothing_at_all(self, qtbot):
        """Zero objects has no page count, no memory, and no cut time.

        ``montage_cap_cost`` answers "" for a cap that is not a positive
        count, and the caller has to notice: formatting "" onto the end of
        the help would leave the reader a tooltip ending in a dangling blank
        paragraph. Nothing new is written for zero -- and the very next
        number they try writes its price as usual, which is what says the
        control is still live rather than merely quiet.
        """
        dialog = _dialog(qtbot)
        cap = dialog._editors["cap"]
        label = dialog._labels["cap"]

        assert ps.montage_cap_cost(0) == ""

        priced_at_the_default = label.toolTip()
        cap.setValue(0)
        assert label.toolTip() == priced_at_the_default
        assert dialog.values()["cap"] == 0

        cap.setValue(5000)
        assert _costs(label) == ["5,000"]
        assert "MB of crops held" in label.toolTip()

    def test_a_panel_that_offers_no_cap_at_all_still_opens(self, qtbot,
                                                          monkeypatch):
        """Retiring a setting must not take the window down with it.

        ``cap`` reaches this dialog through ``picture_settings.categories``,
        the one table that decides which settings are offered and where. A
        setting dropped from that table -- retired, or moved behind a
        different panel -- leaves the cap wiring with no control to connect
        to and the cost sentence with no label to write on. Both have to
        shrug: every OTHER picture setting is still on screen and still
        editable, and a user who lost one control must not lose the window.
        """
        real = ps.categories

        def without_cap():
            return tuple((title, tuple(k for k in keys if k != "cap"))
                         for title, keys in real())

        monkeypatch.setattr(psd, "categories", without_cap)
        capless = _dialog(qtbot)
        monkeypatch.setattr(psd, "categories", real)
        ordinary = _dialog(qtbot)

        assert "cap" not in capless.values()
        assert capless.tab_of("cap") == ""
        # The tab the cap lived on is still there with its other settings.
        assert "Which cells" in capless.tab_titles()
        assert capless.values()["score_column"] == \
            ordinary.values()["score_column"]
        assert len(capless.values()) == len(ordinary.values()) - 1

        # Re-pricing with no cap control is a no-op rather than an error,
        # and nothing was ever remembered to re-price from.
        capless._say_what_the_cap_costs()
        assert capless._cap_help is None

        # The same call on the ordinary dialog does write the sentence, so
        # the silence above belongs to the missing control.
        assert _costs(ordinary._labels["cap"]) == ["2,000"]

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
