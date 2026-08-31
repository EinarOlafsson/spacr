"""The composite theme token, and the ``space:`` arms nothing can reach.

``get_theme_choice`` and ``set_theme_choice`` collapse two stored
preferences -- the theme and its variant -- into one token the settings
combo box holds, and expand it again on the way back. Both carry a
``space:`` arm, and both are dead:

  * ``theme_choices`` says so in a comment -- "Space is gone; its
    variants are no longer offered" -- so no ``space:`` token exists for
    ``set_theme_choice`` to receive, and its guard refuses anything not
    in that set;
  * ``get_theme`` filters what it reads against ``VALID_THEMES``, which
    no longer contains ``"space"``, so it cannot return the value
    ``get_theme_choice`` tests for. A settings file left over from
    before the removal reads back as the default, not as space.

So neither arm can run, and the honest test is the one that fails when
that stops being true rather than one that pretends to drive them.
Reported as dead code in instruction 310; NOT deleted here, per the
standing rule.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from spacr.qt import preferences as P

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _restore_the_theme():
    """Put the stored values back, whatever the test did."""
    before = (P.get_theme(), P.get_cell_variant())
    yield
    P.set_theme(before[0])
    P.set_cell_variant(before[1])


def _every_token():
    return [token for _label, token in P.theme_choices()]


class TestWhyTheSpaceArmsCannotRun:

    def test_space_is_not_a_valid_theme(self):
        """THE PIN for ``if theme == "space"``.

        ``get_theme`` answers ``DEFAULT_THEME`` for anything outside this
        tuple, so the value the arm tests for cannot come out of it --
        including from a settings file written before the removal, which
        is the one case an arm like this would exist for.
        """
        assert "space" not in P.VALID_THEMES

        source = inspect.getsource(P.get_theme)
        assert "raw if raw in VALID_THEMES else DEFAULT_THEME" in source, (
            "get_theme no longer filters what it reads, so a stale "
            "settings file can now put 'space' back into circulation and "
            "the arm in get_theme_choice is live again")

    def test_no_space_token_is_offered(self):
        """THE PIN for ``if choice.startswith("space:")``."""
        assert not [t for t in _every_token() if t.startswith("space:")]

        source = inspect.getsource(P.theme_choices)
        assert "Space is gone" in source

    def test_a_stale_space_setting_reads_back_as_the_default(self):
        """DRIVEN, because this is the case the dead arm was for.

        Written straight into the store, past ``set_theme``'s validation,
        exactly as an old settings file would have it.
        """
        P._settings().setValue(P._KEY_THEME, "space")

        assert P.get_theme() == P.DEFAULT_THEME
        assert P.get_theme_choice() == P.DEFAULT_THEME, (
            "a leftover 'space' setting now produces a composite token, "
            "which the Theme combo box has no entry for -- the control "
            "would open with nothing selected")

    def test_setting_the_theme_to_space_is_refused(self):
        with pytest.raises(ValueError) as caught:
            P.set_theme("space")

        assert "unknown theme" in str(caught.value)


class TestReadingTheToken:

    def test_a_cell_theme_reads_as_cell_and_its_variant(self):
        variant = [t for t in _every_token()
                   if t.startswith("cell:")][0].split(":", 1)[1]
        P.set_theme("cell")
        P.set_cell_variant(variant)

        assert P.get_theme_choice() == f"cell:{variant}"

    def test_a_plain_theme_reads_as_itself(self):
        P.set_theme("dark")

        assert P.get_theme_choice() == "dark"


class TestWritingTheToken:

    def test_a_cell_token_sets_both_the_variant_and_the_theme(self):
        token = [t for t in _every_token() if t.startswith("cell:")][-1]
        P.set_theme("dark")

        P.set_theme_choice(token)

        assert P.get_theme() == "cell"
        assert P.get_cell_variant() == token.split(":", 1)[1]

    def test_a_plain_token_sets_only_the_theme(self):
        P.set_theme_choice("light")

        assert P.get_theme() == "light"

    def test_an_unknown_token_is_refused_and_names_the_choices(self):
        """The guard above both arms: the combo box holds tokens from
        `theme_choices`, so anything else arrived from a hand-edited
        settings file and must not be persisted as a theme nothing can
        render."""
        with pytest.raises(ValueError) as caught:
            P.set_theme_choice("space:nebula")

        message = str(caught.value)
        assert "unknown theme choice" in message
        assert "cell:" in message, (
            "the refusal no longer lists the valid tokens, so a user with "
            "a bad settings file is told what is wrong and not what to do")


@pytest.mark.parametrize("token", _every_token())
def test_every_offered_token_survives_a_write_and_a_read(token,
                                                         _restore_the_theme):
    """The property the two functions exist to have, over the WHOLE set.

    A token that does not come back is a theme the user picks and loses
    on restart, and the two functions are far enough apart in the file
    that one can grow a case the other does not.
    """
    P.set_theme_choice(token)

    assert P.get_theme_choice() == token
