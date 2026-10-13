"""The Cell theme's composite token, both directions.

``get_theme_choice`` collapses theme-plus-variant into one token so a
single combo box can offer "Space (nebula)" and "Cell (mitochondria)" as
peers of "Dark". ``set_theme_choice`` takes it apart again.

THE SPACE HALF WAS THE UNCOVERED ONE, and this file was written for the
Cell half first, on a misreading of the census line numbers. The tests
passed and closed nothing -- lines 1494, 1507 and 1508 are the Space
returns, and measuring the arcs afterwards is the only reason that was
noticed rather than committed as "three items closed".

Both halves are here now. They are each other's inverse, so neither is
noticed by testing the other, which is how one of them stayed uncovered
while the pair looked well tested.

Round-tripped rather than asserted one way: a token that can be written
and not read back is the failure this pairing exists to prevent, and only
driving both directions catches it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import preferences as prefs


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """A settings store per test, so a theme choice cannot leak."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    prefs._settings.cache_clear() if hasattr(
        prefs._settings, "cache_clear") else None
    yield


def _cell_variants():
    from spacr.qt.imagery import CELL_VARIANTS

    return list(CELL_VARIANTS)


def test_the_cell_theme_reports_its_variant_in_the_token():
    """``get_theme_choice`` returns ``cell:<variant>``, not bare "cell".

    The line under test is the Cell twin of the Space one beside it.
    """
    variant = _cell_variants()[0]
    prefs.set_cell_variant(variant)
    prefs.set_theme("cell")
    assert prefs.get_theme_choice() == f"cell:{variant}"


def test_choosing_a_cell_token_sets_both_the_theme_and_the_variant():
    """One token, two settings -- and BOTH are asserted.

    Setting the variant and forgetting the theme leaves the user on
    whatever they had before, with their variant silently changed
    underneath it.
    """
    variant = _cell_variants()[-1]
    prefs.set_theme("dark")
    prefs.set_theme_choice(f"cell:{variant}")
    assert prefs.get_theme() == "cell"
    assert prefs.get_cell_variant() == variant


@pytest.mark.parametrize("variant", _cell_variants())
def test_every_cell_variant_round_trips(variant):
    """Written, then read back, for each bundled micrograph.

    A variant the combo box offers but the token cannot carry is a choice
    that silently reverts, which is worse than one that is not offered.
    """
    prefs.set_theme_choice(f"cell:{variant}")
    assert prefs.get_theme_choice() == f"cell:{variant}"


def test_the_token_is_one_the_combo_box_actually_offers():
    """The round trip is checked against `theme_choices`, not invented.

    Otherwise this file could round-trip a token the dialog never shows.
    """
    offered = {token for _label, token in prefs.theme_choices()}
    variant = _cell_variants()[0]
    assert f"cell:{variant}" in offered


def test_an_unknown_token_is_refused_rather_than_stored():
    """A bad token must not become a theme nobody can get out of."""
    prefs.set_theme("dark")
    with pytest.raises(ValueError):
        prefs.set_theme_choice("cell:not-a-real-micrograph")
    assert prefs.get_theme() == "dark"


def test_no_space_branch_survives_in_the_token_functions():
    """The Space theme is gone, and so is its handling.

    `VALID_THEMES` is ("dark", "light", "cell", "glass", "system").
    `set_theme` refuses "space" and `theme_choices` offers no `space:`
    token, so the two branches that used to handle one could not be
    reached by any route -- three items in the census that no test could
    ever have closed.

    Asserted as an ABSENCE in the source, and that is safe here only
    because the two positive facts underneath it are driven above and
    below: "space" is refused, and no offered token starts with it.
    """
    import inspect

    for function in (prefs.get_theme_choice, prefs.set_theme_choice):
        body = inspect.getsource(function)
        code = "\n".join(line for line in body.splitlines()
                          if not line.strip().startswith("#"))
        assert '"space:' not in code and "'space:" not in code, (
            f"{function.__name__} handles a theme that cannot be set")


def test_space_is_not_a_theme_anyone_can_choose():
    """The fact the deletion rests on, driven rather than assumed."""
    assert "space" not in prefs.VALID_THEMES
    with pytest.raises(ValueError):
        prefs.set_theme("space")
    assert not [t for _l, t in prefs.theme_choices()
                if t.startswith("space")]
