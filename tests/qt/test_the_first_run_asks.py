"""The first run asks, the installer does not, and an update asks again.

Instruction 221.

THE INSTALLER IS THE WRONG PLACE. It runs once, often unattended, sometimes
by an administrator who is not the user, and it asks before the person has
seen a single screen of what they are configuring. Every answer is a guess.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def setup(qtbot):
    from spacr.qt import setup_screen

    before = setup_screen.answered_version()
    yield setup_screen
    setup_screen.mark_answered(before)


class TestWhenItOpens:

    def test_a_profile_that_never_answered_gets_it(self, setup):
        setup.mark_answered("")
        assert setup.should_open() is True

    def test_answering_closes_it(self, setup):
        setup.mark_answered(setup.current_version())
        assert setup.should_open() is False

    def test_an_update_opens_it_again(self, setup):
        """"also the setup windo should be opened every update." The trigger
        is "answered for THIS VERSION", not "answered at all" -- an update
        that adds a setting has a question the user has never seen."""
        setup.mark_answered(setup.current_version())
        assert setup.should_open("99.9.9") is True

    def test_the_same_version_twice_does_not(self, setup):
        setup.mark_answered("4.2.0")
        assert setup.should_open("4.2.0") is False


class TestEveryQuestionIsAsked:

    def test_all_of_them_are_offered(self, setup):
        keys = {q[0] for q in setup.questions()}
        assert {"language", "theme", "colour_blind", "spacr_mode",
                "hash_inputs", "issue_prompt", "ai_default"} <= keys

    def test_each_has_a_getter_and_a_setter(self, setup):
        for key, _label, getter, setter, _choices in setup.questions():
            assert callable(getter), key
            assert callable(setter), key

    def test_each_has_a_working_default(self, setup):
        """The screen can be dismissed without answering anything, and
        nothing is left unset. A setup screen that MUST be completed is a
        modal dialog wearing a nicer coat."""
        answers = setup.current()
        for key, _l, _g, _s, _c in setup.questions():
            assert key in answers, key
            assert answers[key] is not None, key

    def test_the_accessors_are_the_preference_modules_own(self, setup):
        """An answer given here and an answer given in Preferences must be
        the same answer, not two stores that agree by accident."""
        from spacr.qt import preferences

        for key, _l, getter, _s, _c in setup.questions():
            assert getattr(preferences, getter.__name__, None) is getter, key


class TestAnswersAreKept:

    def test_writing_one_does_not_disturb_the_others(self, setup):
        from spacr.qt import preferences

        before = preferences.get_issue_prompt_mode()
        try:
            setup.apply({"ai_default": True})
            assert preferences.get_ai_on_by_default() is True
            assert preferences.get_issue_prompt_mode() == before
        finally:
            preferences.set_ai_on_by_default(False)

    def test_one_bad_answer_does_not_lose_the_good_ones(self, setup):
        """A setup screen that discards six good answers because the seventh
        was bad has cost the user the whole screen."""
        from spacr.qt import preferences

        try:
            trouble = setup.apply({"ai_default": True,
                                   "issue_prompt": "whenever-i-feel-like-it"})
            assert trouble, "the bad one was reported"
            assert any("issue_prompt" in t for t in trouble)
            assert preferences.get_ai_on_by_default() is True
        finally:
            preferences.set_ai_on_by_default(False)

    def test_an_unmentioned_key_is_left_alone(self, setup):
        from spacr.qt import preferences

        before = preferences.get_hash_inputs()
        setup.apply({"ai_default": preferences.get_ai_on_by_default()})
        assert preferences.get_hash_inputs() == before


class TestTheCornerAccent:
    """"a blue line that follows the corner of the box the mouse is closest
    to" -- a POINTER READOUT, not a glow. Built as a glow it would light on
    hover and say nothing about where the mouse is."""

    @pytest.fixture
    def card(self, qtbot):
        from spacr.qt.widgets.setup_card import SetupCard

        widget = SetupCard()
        qtbot.addWidget(widget)
        widget.resize(400, 300)
        return widget

    @pytest.mark.parametrize("point,expected", [
        ((5, 5), "topLeft"),
        ((395, 5), "topRight"),
        ((395, 295), "bottomRight"),
        ((5, 295), "bottomLeft"),
    ])
    def test_it_picks_the_nearest_corner(self, card, point, expected):
        from PySide6.QtCore import QPointF
        from spacr.qt.widgets.setup_card import CORNERS

        assert CORNERS[card.nearest_corner(QPointF(*point))] == expected

    def test_moving_the_pointer_moves_the_accent(self, card):
        """It ARRIVES at the corner rather than jumping to it.

        Two changes since this test was written. The light EASES towards
        the pointer, so one call moves it a sixth of the way and `corner()`
        still names where it was -- the frames have to be run. And every
        frame re-reads the real cursor, which is what lets it follow a
        pointer that has left the window; here that would steer the accent
        at whatever the machine's pointer happens to be doing, so the
        re-aim is silenced and only the easing is under test.
        """
        from PySide6.QtCore import QPointF

        card._aim_at_the_cursor = lambda: False

        def settle(point):
            card._follow(QPointF(*point))
            for _ in range(200):
                card._tick()

        settle((5, 5))
        assert card.corner() == "topLeft"
        settle((395, 295))
        assert card.corner() == "bottomRight"

    def test_it_paints_without_raising(self, card):
        from PySide6.QtGui import QPixmap

        pixmap = QPixmap(card.size())
        card.render(pixmap)
        assert not pixmap.isNull()

    def test_the_colours_come_from_the_palette(self):
        """178 and 198's rule: a hex typed in reads on one theme and
        vanishes on the other, and the author sees only the one they use."""
        import inspect

        from spacr.qt.widgets import setup_card

        body = inspect.getsource(setup_card.SetupCard._paint)
        assert "active_palette" in body
        for literal in ("#fff", "#000", "Qt.blue", "Qt.white"):
            assert literal not in body, literal

    def test_decoration_is_not_load_bearing(self, card, monkeypatch):
        """If the accent cannot be drawn the card is still a card."""
        monkeypatch.setattr(
            type(card), "_paint",
            lambda self: (_ for _ in ()).throw(RuntimeError("no painter")))
        from PySide6.QtGui import QPixmap

        pixmap = QPixmap(card.size())
        card.render(pixmap)          # must not raise
        assert not pixmap.isNull()

    def test_mouse_tracking_is_on(self, card):
        """Without it `mouseMoveEvent` fires only while a button is held --
        which is never, on a card the user is only reading."""
        assert card.hasMouseTracking()
