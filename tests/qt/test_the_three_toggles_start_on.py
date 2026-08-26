"""Three preferences ship on, and an explicit opt-out still survives.

A default that flips changes the answer for every user who never touched
the control -- which is the point -- but it must not reach anyone who DID
touch it. The setters always write, so an explicit False is stored rather
than being inferred from the default, and it goes on reading False.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import preferences as prefs                      # noqa: E402

#: Each toggle as (constant, getter, setter), named the way the setup card
#: names it so a failure says which row a user would see unticked.
TOGGLES = {
    "Reproducibility hash": (
        "DEFAULT_HASH_INPUTS", prefs.get_hash_inputs, prefs.set_hash_inputs),
    "AI assistant on at launch": (
        "DEFAULT_AI_ON_AT_LAUNCH", prefs.get_ai_on_by_default,
        prefs.set_ai_on_by_default),
    "Include recent logs in a report": (
        "DEFAULT_SHARE_DIAGNOSTIC_LOGS", prefs.get_share_diagnostic_logs,
        prefs.set_share_diagnostic_logs),
}


@pytest.fixture(autouse=True)
def _own_config(tmp_path, monkeypatch):
    """A store of this test's own, so nothing reads the real preferences."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    prefs._settings.cache_clear() if hasattr(prefs._settings, "cache_clear") \
        else None
    yield


@pytest.mark.parametrize("caption", sorted(TOGGLES))
def test_the_constant_says_on(caption):
    """The default is written down, not buried in a getter's fallback."""
    name = TOGGLES[caption][0]
    assert getattr(prefs, name) is True, f"{name} ships off"


@pytest.mark.parametrize("caption", sorted(TOGGLES))
def test_an_untouched_control_reads_on(caption):
    """A user who never opened preferences gets the toggle on."""
    _, getter, _ = TOGGLES[caption]
    assert getter() is True


@pytest.mark.parametrize("caption", sorted(TOGGLES))
def test_an_explicit_opt_out_is_not_overwritten_by_the_default(caption):
    """The hazard: a stored False must keep reading False."""
    _, getter, setter = TOGGLES[caption]
    setter(False)
    assert getter() is False, (
        f"{caption}: an explicit opt-out was overwritten by the default")
    setter(True)
    assert getter() is True


def test_the_setup_card_renders_all_three_ticked(qtbot, tmp_path,
                                                 monkeypatch):
    """What the user actually sees on the card, not just what a getter says."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt.setup_screen import questions

    wanted = set(TOGGLES)
    seen = {}
    for row in questions():
        key, caption, getter = row[0], row[1], row[2]
        if caption in wanted:
            seen[caption] = getter()
    assert set(seen) == wanted, f"card is missing rows: {wanted - set(seen)}"
    off = [c for c, on in seen.items() if not on]
    assert off == [], f"these render unticked on the setup card: {off}"
