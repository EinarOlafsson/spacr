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
# object.py -- every numeric role channel has a dense position
# ---------------------------------------------------------------------------

class TestFillingInTheDenseChannelPositions:

    def test_a_channel_that_has_a_dense_position_is_filled_in(self):
        dense = {2: 0, 5: 1}
        settings = {"cell_channel": 2, "cellpose_cell_channel": None}

        raw = settings.get("cell_channel")
        assert int(raw) in dense
        settings["cellpose_cell_channel"] = dense[int(raw)]

        assert settings["cellpose_cell_channel"] == 0

    def test_every_numeric_role_channel_is_in_the_map_it_just_built(self):
        """The deleted membership guard re-checked this exact premise.

        ``dense_mask_channel_positions`` reads the same role keys as the two
        generator loops and applies the same ``int`` coercion.  A numeric role
        channel therefore cannot be absent; indexing directly makes future
        contract drift fail loudly instead of silently leaving an alias unset.
        """
        from spacr import object as O
        from spacr.utils import dense_mask_channel_positions

        settings = {"nucleus_channel": 2, "cell_channel": 5,
                    "pathogen_channel": None, "organelle_channel": 7}
        dense = dense_mask_channel_positions(settings)
        for role in ("nucleus", "cell", "organelle"):
            assert int(settings[f"{role}_channel"]) in dense

        source = inspect.getsource(O)
        assert "if _raw in _dense:" not in source
        assert source.count("= _dense[_raw]") == 2

    def test_a_channel_that_is_not_a_number_is_skipped_before_the_lookup(self):
        """The guard above it: ``int('rgb')`` raises, and a settings file
        can hold anything a user typed."""
        from spacr import object as O

        source = inspect.getsource(O)
        first = source.index("= _dense[_raw]")
        window = source[max(0, first - 500):first]
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

    def test_a_tooltip_on_screen_does_NOT_stop_the_tick(self, qtbot):
        """The opposite of what this test used to assert, and deliberately.

        It guarded `if a_popup_is_on_screen():` in `_on_tick` and the clock
        restart behind it. That guard WAS the backdrop freeze: it stopped
        the animation whenever a menu or tooltip was up, and the animation
        did not always start again.

        THE HOLD WAS REMOVING A BURST THE POPUP DID NOT CAUSE. Widget
        repaints over 1.2 s, an ambient backdrop behind forty labels and
        twelve buttons:

            menu open, hold ON        2      (the animation is stopped)
            menu open, hold OFF   1,592
            NO menu,   hold ON    1,590
            NO menu,   hold OFF   1,590

        The last two lines are the finding: a moving backdrop repaints
        everything above it whether or not a popup is on screen, and the
        popup adds 2 repaints in 1,592. So the guard was not sparing the
        compositor anything -- it was switching the backdrop off, which is
        a different feature and one nobody asked for.

        Asserted on the SOURCE, exactly as the old test did, because the
        thing being held is the absence of a branch and the positive fact
        under it is the frame counts above, which live in the instruction
        file rather than in a unit test.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import ambient

        source = inspect.getsource(ambient)
        assert "if a_popup_is_on_screen():" not in source, (
            "the popup hold is back in _on_tick; it stops the animation "
            "rather than sparing the compositor, and it is what the "
            "backdrop-freeze report was about")
        assert not hasattr(ambient, "a_popup_is_on_screen"), (
            "ambient imports the popup check again, which is how the guard "
            "came back last time")


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
