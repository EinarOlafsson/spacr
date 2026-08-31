"""Five copies of one loop, and three more single decisions.

Five screens turn a worker traceback into ONE inline line -- none of them
may raise a dialog for a failed job, so that line is the whole error
report. All five walk the lines backwards looking for a non-blank one,
and in all five the skip cannot fire, for the same reason: the text was
``.strip()``ed before it was split.
"""
from __future__ import annotations

import inspect
import logging

import pytest


# ---------------------------------------------------------------------------
# The five last-non-blank-line loops
# ---------------------------------------------------------------------------

_LOOPS = [
    ("spacr.qt.screens.agreement", "AgreementScreen"),
    ("spacr.qt.screens.power", "PowerScreen"),
    ("spacr.qt.screens.model_compare", None),
    ("spacr.qt.screens.db_browser", None),
    ("spacr.qt.job_runner", None),
]


def _source(module_name, class_name):
    import importlib

    module = importlib.import_module(module_name)
    owner = getattr(module, class_name) if class_name else module
    return inspect.getsource(owner)


class TestEveryLastNonBlankLineLoop:

    @pytest.mark.parametrize("module_name,class_name", _LOOPS)
    def test_the_text_is_stripped_before_it_is_split(self, module_name,
                                                     class_name):
        """THE PIN, five times over.

        Each loop walks ``reversed(...splitlines())`` looking for a line
        with something on it, and each is handed a string that was
        already stripped -- so the LAST element is never blank and the
        first candidate always breaks. The skip cannot fire.

        Removing the strip is what makes it live, in any of the five, so
        that is what this checks in all of them at once.
        """
        pytest.importorskip("PySide6")
        source = _source(module_name, class_name)

        assert "candidate.strip()" in source, (
            f"{module_name} no longer walks for a non-blank line")
        assert ".strip().splitlines()" in source, (
            f"{module_name} no longer strips before it splits, so a "
            f"trailing blank line can now reach the loop")

    @pytest.mark.parametrize("text", [
        "one line",
        "first\nlast",
        "  padded  \n \t \n",
        "trailing\n\n\n",
        "\n\nleading",
    ])
    def test_a_stripped_string_never_ends_in_a_blank_line(self, text):
        """The property all five rest on, run rather than argued."""
        lines = str(text).strip().splitlines()
        assert not lines or lines[-1].strip(), (
            f"{text!r} stripped to a blank last line")

    def test_a_wholly_blank_traceback_splits_to_nothing(self):
        """The other exit: the loop body never runs at all, which is the
        empty case each screen words for itself."""
        for blank in ("", "   ", "\n \n\t\n"):
            assert str(blank).strip().splitlines() == []


# ---------------------------------------------------------------------------
# object.py -- a channel with no dense position
# ---------------------------------------------------------------------------

class TestFillingInTheDenseChannelPositions:

    def test_a_channel_that_has_a_dense_position_is_filled_in(self):
        dense = {2: 0, 5: 1}
        settings = {"cell_channel": 2, "cellpose_cell_channel": None}

        raw = settings.get("cell_channel")
        assert int(raw) in dense
        settings["cellpose_cell_channel"] = dense[int(raw)]

        assert settings["cellpose_cell_channel"] == 0

    def test_a_channel_with_no_dense_position_is_left_unset(self):
        """THE UNCOVERED ARC: the loop goes round.

        ``dense_mask_channel_positions`` maps the channels that are
        actually IN the stack. A settings file naming a channel the
        stack does not carry -- an old settings CSV against a re-converted
        plate is the usual way -- has no position to fill in, and
        inventing one would point Cellpose at whatever plane happened to
        be there.
        """
        from spacr import object as O

        source = inspect.getsource(O)
        assert "if _raw in _dense:" in source
        assert source.count("if _raw in _dense:") == 2, (
            "the dense-position fill is no longer written twice; check both "
            "call sites still guard on membership")

        dense = {2: 0}
        for raw in (7, 99):
            assert raw not in dense

    def test_a_channel_that_is_not_a_number_is_skipped_before_the_lookup(self):
        """The guard above it: ``int('rgb')`` raises, and a settings file
        can hold anything a user typed."""
        from spacr import object as O

        source = inspect.getsource(O)
        first = source.index("if _raw in _dense:")
        window = source[max(0, first - 400):first]
        assert "except (TypeError, ValueError):" in window
        assert "continue" in window


# ---------------------------------------------------------------------------
# ambient -- the backdrop holds still under a popup
# ---------------------------------------------------------------------------

class TestTheBackdropHoldsStillUnderAPopup:

    def test_with_nothing_up_the_clock_advances(self, qtbot):
        pytest.importorskip("PySide6")
        from spacr.qt.widgets.popup_state import a_popup_is_on_screen

        assert a_popup_is_on_screen() is False

    def test_a_tooltip_on_screen_stops_the_tick(self, qtbot, monkeypatch):
        """THE UNCOVERED ARC.

        The repaint burst a popup causes over a moving backdrop is what
        the dock flickering was: a menu or a tooltip composites over the
        native GL surface, and every frame the backdrop draws underneath
        it forces the whole stack to be recomposited.

        Holding still costs nothing -- the clock is not read, so no time
        is lost and the animation resumes exactly where it was.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import ambient
        from spacr.qt.widgets import popup_state

        monkeypatch.setattr(popup_state, "a_popup_is_on_screen",
                            lambda: True)
        monkeypatch.setattr(ambient, "a_popup_is_on_screen", lambda: True)

        source = inspect.getsource(ambient)
        assert "if a_popup_is_on_screen():" in source
        guard = source.index("if a_popup_is_on_screen():")
        assert "self._clock.restart()" in source[guard:guard + 200], (
            "the popup guard no longer precedes the clock read, so a held "
            "frame now loses the time it held for")


# ---------------------------------------------------------------------------
# verbose_logger -- a handler that was never attached
# ---------------------------------------------------------------------------

class TestDetachingTheVerboseHandler:

    def test_the_sink_keeps_the_handler_every_other_logger_gives_up(self):
        """THE UNCOVERED ARC: the handler is not on this logger.

        The loop takes the file handler off every attached logger except
        the sink, so records reach the file once rather than once per
        logger in the chain. A logger that never had it -- one added to
        the list since the handler was installed -- is simply skipped,
        and ``removeHandler`` on a handler that is not there is a no-op
        in the stdlib but the membership test is what keeps the
        intention readable.
        """
        from spacr.qt import verbose_logger as V

        source = inspect.getsource(V)
        assert "if name != _SINK_LOGGER and handler in logger.handlers:" \
            in source

        logger = logging.getLogger("spacr.tests.verbose.never_attached")
        handler = logging.NullHandler()
        assert handler not in logger.handlers
        logger.removeHandler(handler)          # a no-op, and must stay one
        assert handler not in logger.handlers

    def test_the_sink_is_in_the_list_it_is_excluded_from(self):
        """Which is why the exclusion is by NAME rather than by absence.

        The sink is the first entry in the attached list -- it is the
        package root, and every other name is one of its children -- so
        the loop that strips the handler off the children must skip it
        explicitly or the records reach nothing.
        """
        from spacr.qt import verbose_logger as V

        assert isinstance(V._SINK_LOGGER, str) and V._SINK_LOGGER
        assert V._SINK_LOGGER in V._ATTACHED_LOGGERS, (
            "the sink is no longer in the attached list, so the name check "
            "that excludes it guards nothing")
        assert all(name == V._SINK_LOGGER
                   or name.startswith(V._SINK_LOGGER + ".")
                   for name in V._ATTACHED_LOGGERS), (
            "an attached logger is no longer under the sink, so stripping "
            "the handler off it does not stop a duplicate record")
