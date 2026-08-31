"""The setup screen and Preferences must offer the SAME performance
levels.

Reported: "the spaCR mode should list all the performance levels
(laptop, extra performance, performance, balanced, workstation), some
are missing in the start spacr menu (they are all present in
preferences)".

The screen was offering ``SPACR_MODES`` -- the old three-value resource
POSTURE -- while Preferences offers the five ``PERFORMANCE_LEVELS``. The
two are not interchangeable: ``spacr_mode_for_level`` folds five onto
three, so the posture cannot express either end of the scale.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import preferences as prefs

pytestmark = pytest.mark.qt


def _mode_question():
    from spacr.qt.setup_screen import questions

    for question in questions():
        if question[0] == "spacr_mode":
            return question
    raise AssertionError("the setup screen no longer asks about spaCR mode")


def test_the_screen_offers_every_level_preferences_does():
    """THE DEFECT. Both lists come from the same tuple now, so they
    cannot drift apart again."""
    offered = [value for value, _caption in _mode_question()[4]]

    assert offered == list(prefs.PERFORMANCE_LEVELS)
    assert "laptop" in offered, "Laptop is missing from the setup screen"
    assert "workstation" in offered, "Workstation is missing from the setup screen"


def test_all_five_are_offered_not_three():
    """The count on its own, because the report was about a SHORT list
    and a regression would most likely shorten it again."""
    assert len(_mode_question()[4]) == 5


def test_every_level_is_shown_under_its_preferences_caption():
    """A level named one thing on the setup screen and another in
    Preferences is two settings as far as a user is concerned."""
    captions = dict(_mode_question()[4])

    for level in prefs.PERFORMANCE_LEVELS:
        assert captions[level] == prefs.PERFORMANCE_LABELS[level], (
            f"{level} reads as {captions[level]!r} on the setup screen and "
            f"{prefs.PERFORMANCE_LABELS[level]!r} in Preferences")


def test_the_screen_reads_and_writes_the_LEVEL_not_the_posture():
    """The half that makes the choice stick.

    ``set_spacr_mode`` writes the posture, and ``spacr_mode_for_level``
    folds laptop onto extra_performance and workstation onto balanced --
    so a screen that wrote through it could not store either end, and
    the value would come back as something the user did not pick.
    """
    _key, _label, getter, setter, _choices = _mode_question()

    assert getter is prefs.get_performance_level
    assert setter is prefs.set_performance_level


def test_the_two_ends_of_the_scale_survive_a_round_trip():
    """Driven, because that is the user-visible failure: choose
    Workstation, reopen the screen, and read Balanced back."""
    before = prefs.get_performance_level()
    _key, _label, getter, setter, _choices = _mode_question()
    try:
        for level in ("laptop", "workstation", "balanced"):
            setter(level)
            assert getter() == level, (
                f"{level} did not survive a write and a read, so the setup "
                f"screen shows a choice the user did not make")
    finally:
        prefs.set_performance_level(before)


def test_the_posture_is_still_derived_for_the_cleanup_code():
    """The level is what the user picks; the posture is what
    resource_cleanup reads. Writing the level must keep both in step, or
    the interface and the runtime disagree."""
    before = prefs.get_performance_level()
    try:
        for level in prefs.PERFORMANCE_LEVELS:
            prefs.set_performance_level(level)
            assert prefs.get_spacr_mode() == prefs.spacr_mode_for_level(level)
    finally:
        prefs.set_performance_level(before)


def test_the_dropped_posture_tuple_is_still_three_and_still_used():
    """SPACR_MODES has not gone away and should not: it is the vocabulary
    the cleanup code speaks. This says so, so the next reader does not
    delete it as unused."""
    assert prefs.SPACR_MODES == ("extra_performance", "performance",
                                 "balanced")
    assert set(prefs.spacr_mode_for_level(level)
               for level in prefs.PERFORMANCE_LEVELS) <= set(prefs.SPACR_MODES)


class TestTheHelpTextExplainsTheLevels:
    """The slide's prose must describe the control beside it.

    It described the old three-value posture and stayed behind when the
    screen was pointed at the five levels, so it explained a control the
    reader was not looking at.
    """

    def _blurb(self):
        from spacr.qt.widgets.setup_slides import SLIDES

        for title, blurb, keys in SLIDES:
            if "spacr_mode" in keys:
                return blurb
        raise AssertionError("no slide asks about spaCR mode")

    def test_every_level_is_named_in_the_prose(self):
        blurb = self._blurb()

        for label in prefs.PERFORMANCE_LABELS.values():
            assert label in blurb, (
                f"{label!r} is offered by the control and not explained by "
                f"the text beside it")

    def test_the_two_ends_say_what_machine_they_are_for(self):
        """A level named with no hardware beside it is a choice the reader
        has to guess at, which is the whole reason this slide exists."""
        blurb = self._blurb()

        assert "8 GB" in blurb, "Laptop does not say what machine it is for"
        assert "memory to spare" in blurb, (
            "Workstation does not say what machine it is for")

    def test_it_says_the_science_does_not_change(self):
        """The one thing a reader must not have to wonder about: picking a
        level to be kind to their machine cannot change a result."""
        blurb = self._blurb()

        assert "science is identical at every level" in blurb

    def test_it_no_longer_describes_three(self):
        blurb = self._blurb()

        assert "the other two" not in blurb, (
            "the prose still counts the old three postures")

    def test_the_levels_are_named_in_the_order_the_control_offers_them(self):
        """Reading the sentence and reading the list must agree, or the
        reader has to map one onto the other."""
        blurb = self._blurb()
        seen = [blurb.index(prefs.PERFORMANCE_LABELS[level])
                for level in prefs.PERFORMANCE_LEVELS]

        assert seen == sorted(seen), (
            "the prose names the levels in a different order from the "
            "control")
