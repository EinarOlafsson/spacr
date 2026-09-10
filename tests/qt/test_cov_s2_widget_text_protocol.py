"""Two widgets answering the protocols a generic reader expects of them.

A settings panel walks controls it does not recognise and falls back to the
Qt duck type -- ``text()`` to read, ``setText()`` to write. The percentile
pair is two spin boxes, not a line edit, so the pair has to answer those
itself or a generic reader silently stores the widget's repr. What matters is
that the two halves agree: whatever ``text()`` produces, ``setText()`` has to
read back as the same window.

The colour helper's contract is narrower and just as easy to break: the
colour the dialog opens on must be a COPY, because the caller's ``QColor``
often belongs to a live figure style and Qt dialogs are free to write to what
they are handed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtGui import QColor                                 # noqa: E402

from spacr.qt.widgets import colour_picker                       # noqa: E402
from spacr.qt.widgets.percentile_pair import PercentilePair      # noqa: E402


class TestThePairAnswersTheTextProtocol:

    def test_the_window_reads_as_two_numbers_separated_by_a_comma(self, qtbot):
        """``text()`` is the pair, not the widget's repr."""
        pair = PercentilePair([2, 98])
        qtbot.addWidget(pair)

        assert pair.text() == "2, 98"

    def test_a_fractional_end_survives_the_text_spelling(self, qtbot):
        """Rounding here would quietly lose the top of spaCR's own ladder."""
        pair = PercentilePair([0.5, 99.99])
        qtbot.addWidget(pair)

        assert pair.text() == "0.5, 99.99"

    def test_what_text_writes_set_text_reads_back_unchanged(self, qtbot):
        """The round trip is the whole point of answering both halves.

        A reader that saved through ``text()`` and restored through
        ``setText()`` must hand the panel back the window it had, or the
        first save/reload cycle moves the picture.
        """
        pair = PercentilePair([1, 99.9])
        qtbot.addWidget(pair)
        spelling = pair.text()

        restored = PercentilePair([50, 60])
        qtbot.addWidget(restored)
        restored.setText(spelling)

        assert restored.value() == [1, 99.9]
        assert restored.text() == spelling

    def test_set_text_accepts_the_bracketed_form_an_old_file_holds(self, qtbot):
        """Settings written before the control existed still open on it."""
        pair = PercentilePair([2, 98])
        qtbot.addWidget(pair)

        pair.setText("[10 90]")

        assert pair.value() == [10, 90]

    def test_set_text_announces_the_new_window_once(self, qtbot):
        """A generic writer still has to leave the panel in agreement."""
        pair = PercentilePair([2, 98])
        qtbot.addWidget(pair)
        heard = []
        pair.changed.connect(heard.append)

        pair.setText("5, 95")

        assert heard == [[5, 95]]


class TestTheColourHelperOpensOnACopy:

    @staticmethod
    def _recording_dialog(monkeypatch, answer="#123456"):
        from PySide6.QtWidgets import QColorDialog

        seen = {}

        class Recorder:
            ColorDialogOption = QColorDialog.ColorDialogOption

            @staticmethod
            def getColor(initial, parent, title, options):
                seen["initial"] = initial
                seen["parent"] = parent
                seen["title"] = title
                seen["options"] = options
                return QColor(answer)

        monkeypatch.setattr(colour_picker, "QColorDialog", Recorder)
        return seen

    def test_a_qcolor_start_is_handed_over_as_a_copy(self, qtbot, monkeypatch):
        """The caller's own colour object never reaches the dialog.

        Call sites pass the colour a figure style is currently drawing with.
        Handing that object to a dialog that may write to it would change the
        figure before the user pressed OK -- and would leave it changed after
        a cancel.
        """
        seen = self._recording_dialog(monkeypatch)
        mine = QColor("darkRed")

        chosen = colour_picker.pick_colour(initial=mine, title="Line colour")

        assert seen["initial"] == mine
        assert seen["initial"] is not mine
        assert seen["title"] == "Line colour"
        assert chosen == QColor("#123456")

    def test_a_cancel_comes_back_invalid_rather_than_as_a_colour(
            self, qtbot, monkeypatch):
        """Every call site tests ``isValid()``; a cancel has to fail it."""
        self._recording_dialog(monkeypatch, answer=None)

        chosen = colour_picker.pick_colour(initial=QColor("blue"))

        assert not chosen.isValid()

    def test_a_start_qt_cannot_read_falls_back_to_white(
            self, qtbot, monkeypatch):
        """A stored preference can hold "auto"; Qt opens that on black."""
        seen = self._recording_dialog(monkeypatch)

        colour_picker.pick_colour(initial="auto")

        assert seen["initial"] == QColor("#ffffff")
