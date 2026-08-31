"""Four decisions in the Preferences dialog that its own callers settle.

Three are Qt returning something it always returns, and one is an
optional argument the only caller never passes. All four are cheap to
keep and none can fire, so each is pinned to the caller that keeps it
shut -- and where the premise is a Qt guarantee, that guarantee is
exercised rather than quoted.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel,
                               QSpinBox, QVBoxLayout)

from spacr.qt import preferences as P

pytestmark = pytest.mark.qt


class TestTheWholeNumberSpinBox:
    """``_whole(name, value, suffix="")`` builds one integer field."""

    def test_a_spin_box_takes_the_full_integer_range(self, qtbot):
        """The reason it is a number field and not a capped slider: a spin
        box whose maximum is 2 turns a typed 40 into 2 without explaining
        the change."""
        box = QSpinBox()
        qtbot.addWidget(box)
        box.setRange(-2_000_000_000, 2_000_000_000)
        box.setKeyboardTracking(False)

        box.setValue(40)
        assert box.value() == 40
        assert box.minimum() <= -2_000_000_000
        assert box.maximum() >= 2_000_000_000

    def test_the_suffix_is_offered_and_never_asked_for(self):
        """THE PIN, for ``if suffix:``.

        ``_whole`` takes a suffix and its ONE caller --
        FractalSupersampling -- passes none, because "samples per pixel
        along each axis" has no unit that fits after the number. So the
        branch cannot run.

        Keeping the parameter is reasonable; keeping it UNTESTED is what
        this notices. If a second caller appears with a unit, this fails
        and the suffix gets a test of its own.
        """
        source = inspect.getsource(P)
        helper = source[source.index("def _whole(name, value, suffix=\"\"):"):]
        helper = helper[:helper.index("return box") + len("return box")]
        assert "if suffix:" in helper
        assert "box.setSuffix(suffix)" in helper

        calls = [line for line in source.splitlines()
                 if "_whole(" in line and "def _whole" not in line]
        assert len(calls) == 1, (
            f"_whole now has {len(calls)} callers; check whether one passes "
            f"a suffix: {calls}")
        block = source[source.index(calls[0]):]
        assert "suffix" not in block[:200], (
            "the caller now passes a suffix, so the branch is live")


class TestTheDialogsButtonBox:

    def test_a_save_cancel_box_always_hands_back_both_buttons(self, qtbot):
        """THE PIN, for the two ``is not None`` tests.

        ``QDialogButtonBox.button`` returns None only for a role the box
        was not built with, and this box is built with Save and Cancel
        in the same call. The guards are cheap and correct against a
        future role change; they cannot fire today.

        Exercised against Qt rather than argued, so a PySide that
        started returning None for a declared role fails here -- which
        would otherwise show up as two untranslated buttons.
        """
        box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        qtbot.addWidget(box)

        assert box.button(QDialogButtonBox.Save) is not None
        assert box.button(QDialogButtonBox.Cancel) is not None
        assert box.button(QDialogButtonBox.Discard) is None, (
            "a role the box was not built with now returns a button, so "
            "the guards mean something different")

        source = inspect.getsource(P)
        assert "if save_button is not None:" in source
        assert "if cancel_button is not None:" in source

    def test_the_reset_button_is_placed_away_from_the_two_that_close(self):
        """Why ResetRole: every Qt style groups the destructive-ish button
        apart from Save and Cancel, which is what stops it being clicked
        by muscle memory aimed at Cancel."""
        source = inspect.getsource(P)
        assert "QDialogButtonBox.ResetRole" in source
        assert 'reset_button.setObjectName("PreferencesReset")' in source


class TestWhereTheHintStripGoes:

    def test_a_widget_in_a_layout_reports_its_own_index(self, qtbot):
        """THE PIN, for ``if row_of_buttons >= 0``.

        ``indexOf`` answers -1 only for a widget the layout does not
        hold, and the button box was added to this very layout a few
        lines above. The fallback appends instead, which would put the
        hint strip BELOW the buttons -- read as a footnote to them
        rather than as the answer to the control the pointer is on.
        """
        host = QDialog()
        qtbot.addWidget(host)
        layout = QVBoxLayout(host)
        buttons = QDialogButtonBox(QDialogButtonBox.Save)
        layout.addWidget(buttons)

        assert layout.indexOf(buttons) >= 0
        assert layout.indexOf(QLabel("never added")) == -1, (
            "indexOf no longer answers -1 for a widget it does not hold")

        source = inspect.getsource(P)
        assert "row_of_buttons = layout.indexOf(buttons)" in source
        assert "if row_of_buttons >= 0:" in source
        assert "layout.insertWidget(row_of_buttons, hints)" in source

    def test_the_hint_strip_is_inserted_above_the_buttons(self, qtbot):
        """The live behaviour the guard protects, on a real layout."""
        host = QDialog()
        qtbot.addWidget(host)
        layout = QVBoxLayout(host)
        buttons = QDialogButtonBox(QDialogButtonBox.Save)
        layout.addWidget(buttons)
        hints = QLabel("what the tabs mean")

        layout.insertWidget(layout.indexOf(buttons), hints)

        assert layout.indexOf(hints) < layout.indexOf(buttons), (
            "the hint strip landed below the buttons")


class TestClosingTheDialogAfterARestart:

    def test_the_watcher_is_parented_to_the_window_not_the_dialog(self):
        """THE PIN, for the ``parent``/``window`` None tests.

        The restart watcher is parented to the WINDOW deliberately: the
        dialog is about to close, and a timer that died with it would
        ask nothing. Both closes are then guarded, because either can be
        absent -- Preferences opened from the tray has no parent dialog,
        and a headless apply has no window.

        Neither is absent in the dialog's own flow, which is the only
        caller: it constructs both. Pinned on the comment AND the order,
        because starting the watcher after the closes would defeat the
        parenting.
        """
        source = inspect.getsource(P)
        assert "if parent is not None:" in source
        assert "if window is not None:" in source

        start = source.index("watcher.start()")
        closes = source.index("if parent is not None:", start)
        assert start < closes, (
            "the watcher is started after the dialog closes, so it is "
            "parented to something already going away")
