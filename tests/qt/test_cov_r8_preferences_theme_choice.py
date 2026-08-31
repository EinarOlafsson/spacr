"""The composite theme token, and two arms left over from a removed theme.

`get_theme_choice` / `set_theme_choice` map between the stored theme (a
plain name) and the token the Preferences combo shows -- which for a
themed family carries the variant too, as `cell:microtubules`.

Both functions still carry a `space:` arm. There is no space theme any
more: `VALID_THEMES` does not list it, so `get_theme()` cannot return
it, and `theme_choices()` does not offer a `space:` token, so
`set_theme_choice` rejects one before the arm is reached. The arms are
vestigial rather than untested, and the tests below pin them to the two
declarations that make them dead.
"""
from __future__ import annotations

import pytest

from spacr.qt import preferences as P

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _restore_theme():
    """Leave the stored theme exactly as it was found."""
    before = P.get_theme_choice()
    try:
        yield
    finally:
        P.set_theme_choice(before)


class TestTheCompositeToken:

    def test_a_plain_theme_is_its_own_token(self):
        P.set_theme_choice("dark")
        assert P.get_theme_choice() == "dark"

    @pytest.mark.parametrize("token", ["dark", "light", "glass", "system"])
    def test_every_plain_token_round_trips(self, token):
        P.set_theme_choice(token)
        assert P.get_theme_choice() == token

    def test_a_cell_theme_carries_its_variant(self):
        """The family that still exists: the token names the variant."""
        P.set_theme_choice("cell:microtubules")
        assert P.get_theme() == "cell"
        assert P.get_cell_variant() == "microtubules"
        assert P.get_theme_choice() == "cell:microtubules"

    def test_switching_variant_within_the_family_is_read_back(self):
        P.set_theme_choice("cell:microtubules")
        P.set_theme_choice("cell:filopodia")
        assert P.get_theme_choice() == "cell:filopodia"

    def test_an_unknown_token_is_refused_and_names_the_alternatives(self):
        """A typo in a settings file has to be findable."""
        with pytest.raises(ValueError) as caught:
            P.set_theme_choice("chartreuse")
        message = str(caught.value)
        assert "chartreuse" in message
        assert "dark" in message and "cell:microtubules" in message

    def test_the_offered_tokens_are_exactly_what_is_accepted(self):
        """Anything the combo can show must be settable, and nothing else."""
        for _label, token in P.theme_choices():
            P.set_theme_choice(token)
            assert P.get_theme_choice() == token


class TestTheSpaceArmsThatNoLongerHaveAThemeBehindThem:
    """Both `space:` arms are unreachable. Pinned, not forced.

    Forcing either would mean writing "space" into the store behind
    `set_theme`'s validation, which asserts nothing about the program --
    it would only prove that a function reads a value somebody wrote
    illegally.
    """

    def test_space_is_not_a_theme_the_store_will_hold(self):
        """So `get_theme()` cannot return it, and the read arm is dead."""
        assert "space" not in P.VALID_THEMES
        with pytest.raises(ValueError, match="unknown theme 'space'"):
            P.set_theme("space")

    def test_no_space_token_is_offered(self):
        """So `set_theme_choice` refuses one before reaching its arm."""
        tokens = [token for _label, token in P.theme_choices()]
        assert not any(t.startswith("space:") for t in tokens), (
            "a space: token is offered again; the space arms in "
            "get_theme_choice and set_theme_choice are now reachable")

    def test_a_space_token_is_refused_by_the_validation_above_the_arm(self):
        with pytest.raises(ValueError, match="unknown theme choice"):
            P.set_theme_choice("space:nebula")

    def test_the_arms_are_gone(self):
        """The tidy-up this class asked for, done on 2026-08-31.

        It used to assert the arms were STILL THERE -- "recorded so the
        dead code is findable, not so it is kept... this test says
        where". They are deleted now, so it asserts the opposite, and
        the three tests above are what make that safe: "space" cannot be
        set, no `space:` token is offered, and one is refused before
        reaching any arm. Absence is only worth asserting when the
        positive facts under it are driven.

        `get_space_variant` and `set_space_variant` DID NOT go with
        them, and the old note suggesting they should was too broad.
        The space ARTWORK outlived the space THEME:
        `space_background_path` still reads the variant to pick a
        backdrop, and `spaceout` still draws it. What was deleted is the
        handling for a theme nobody can select.
        """
        import inspect

        for function in (P.get_theme_choice, P.set_theme_choice):
            code = "\n".join(
                line for line in inspect.getsource(function).splitlines()
                if not line.strip().startswith("#"))
            assert '== "space"' not in code
            assert 'startswith("space:")' not in code

    def test_the_space_artwork_kept_its_variant_accessors(self):
        """Because the backdrop still uses them.

        Guards the over-broad half of the deletion: removing these would
        take the `spaceout` backdrop's variant with them.
        """
        assert callable(P.get_space_variant)
        assert callable(P.set_space_variant)
        assert P.space_variants()
