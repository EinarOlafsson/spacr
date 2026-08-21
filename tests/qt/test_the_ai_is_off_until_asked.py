"""The assistant is off until somebody says otherwise.

Instruction 221: "AI should be on by default is a new setting you need to
add and implement."

THE SETTING EXISTS SO THE ANSWER CAN BE YES. The DEFAULT is no, and that is
a decision rather than an oversight: an assistant that is on before anybody
asked for it sends what it is looking at somewhere, and the first run is
exactly when the user has not yet decided whether that is acceptable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def prefs():
    from spacr.qt import preferences

    before = preferences.get_ai_on_by_default()
    yield preferences
    preferences.set_ai_on_by_default(before)


class TestTheDefaultIsOff:

    def test_a_profile_that_never_answered_is_off(self, prefs):
        prefs._settings().remove("ai/on_by_default")
        assert prefs.get_ai_on_by_default() is False

    def test_an_unrecognised_value_reads_as_off(self, prefs):
        """The failure falls on the quiet side, the rule `issue_prompt`
        follows -- a preference file written by another build must not turn
        the assistant on by accident."""
        prefs._settings().setValue("ai/on_by_default", "perhaps")
        assert prefs.get_ai_on_by_default() is False

    def test_an_empty_value_reads_as_off(self, prefs):
        prefs._settings().setValue("ai/on_by_default", "")
        assert prefs.get_ai_on_by_default() is False


class TestItCanBeTurnedOn:

    def test_saying_yes_persists(self, prefs):
        prefs.set_ai_on_by_default(True)
        assert prefs.get_ai_on_by_default() is True

    def test_saying_no_again_persists(self, prefs):
        prefs.set_ai_on_by_default(True)
        prefs.set_ai_on_by_default(False)
        assert prefs.get_ai_on_by_default() is False

    @pytest.mark.parametrize("stored,expected", [
        ("true", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("no", False),
    ])
    def test_the_spellings_qsettings_writes_all_read_back(self, prefs,
                                                          stored, expected):
        """QSettings round-trips a bool as a string on some backends, so the
        reader has to accept what the writer might have produced."""
        prefs._settings().setValue("ai/on_by_default", stored)
        assert prefs.get_ai_on_by_default() is expected

    def test_a_real_bool_round_trips(self, prefs):
        prefs._settings().setValue("ai/on_by_default", True)
        assert prefs.get_ai_on_by_default() is True


class TestItIsSeparateFromTheOtherAiSettings:
    """Turning the assistant off must not silence the issue reporter, and
    choosing never to file issues must not disable the assistant."""

    def test_the_two_keys_are_different(self, prefs):
        assert prefs._KEY_AI_DEFAULT_ON != prefs._KEY_ISSUE_PROMPT

    def test_changing_one_leaves_the_other(self, prefs):
        before = prefs.get_issue_prompt_mode()
        try:
            prefs.set_issue_prompt_mode(prefs.ISSUE_PROMPT_NEVER)
            prefs.set_ai_on_by_default(True)
            assert prefs.get_issue_prompt_mode() == prefs.ISSUE_PROMPT_NEVER
            assert prefs.get_ai_on_by_default() is True
        finally:
            prefs.set_issue_prompt_mode(before)
