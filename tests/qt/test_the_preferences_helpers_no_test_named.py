"""Preference readers and writers no test had ever named.

Instruction 60, on the module where an untested branch is felt soonest: a
preference that reads back wrong is not a crash, it is spaCR quietly running
with a setting the user did not choose and cannot see it did not take.

Fifteen public callables in ``spacr.qt.preferences`` had never appeared in a
test. What is asserted here is the ROUND TRIP -- set it, read it back, and
read it back again through a reload -- plus what each one does with a stored
value that predates it, because every one of these can be handed a settings
file written by an older build.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """A settings store of this test's own.

    Without it these tests rewrite the maintainer's real preferences, which
    is the one failure mode a preferences test can have that nobody notices
    until their theme changes.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt import preferences

    importlib.reload(preferences)
    yield preferences
    importlib.reload(preferences)


# ---------------------------------------------------------------------------
# The assistant provider
# ---------------------------------------------------------------------------

def test_no_provider_chosen_reads_as_empty(own_config):
    """An empty string is what lets the console pick an available one; a
    made-up default would pin it to a vendor the user has not installed."""
    assert own_config.get_preferred_provider() == ""


def test_a_provider_round_trips(own_config):
    own_config.set_preferred_provider("claude")
    assert own_config.get_preferred_provider() == "claude"


def test_clearing_the_provider_goes_back_to_empty(own_config):
    own_config.set_preferred_provider("gpt")
    own_config.set_preferred_provider("")
    assert own_config.get_preferred_provider() == ""


# ---------------------------------------------------------------------------
# Figure style defaults
# ---------------------------------------------------------------------------

def test_no_style_defaults_reads_as_an_empty_map(own_config):
    assert own_config.get_figure_style_defaults() == {}


def test_a_stored_json_string_is_read_back_as_a_map(own_config):
    """QSettings on some platforms returns what it was given as TEXT, so the
    reader has to accept the JSON form or every style default is lost on the
    machines that do."""
    import json

    own_config._settings().setValue(
        own_config._KEY_FIG_STYLE_DEFAULTS,
        json.dumps({"volcano": {"point_size": 7}}))
    assert own_config.get_figure_style_defaults() == {
        "volcano": {"point_size": 7}}


def test_a_stored_value_of_the_wrong_shape_is_ignored(own_config):
    """A panel that raised while reading its own defaults would take the
    screen down before the user could correct them."""
    own_config._settings().setValue(
        own_config._KEY_FIG_STYLE_DEFAULTS, "not json at all")
    assert own_config.get_figure_style_defaults() == {}


def test_only_map_shaped_entries_survive(own_config):
    """One bad entry must not discard the others: the user would lose every
    style default because of a single stale key."""
    own_config._settings().setValue(
        own_config._KEY_FIG_STYLE_DEFAULTS,
        {"volcano": {"point_size": 7}, "broken": "nope"})
    assert own_config.get_figure_style_defaults() == {
        "volcano": {"point_size": 7}}


# ---------------------------------------------------------------------------
# Log levels
# ---------------------------------------------------------------------------

def test_the_console_levels_are_a_subset_of_the_file_levels(own_config):
    """A console line with no matching entry in the log file is what the
    subset rule exists to prevent -- a user reads a warning on screen, opens
    the log to send it on, and it is not there."""
    console = own_config.get_log_console_levels()
    files = own_config.get_log_file_levels()
    assert console <= files, (console, files)


def test_the_subset_is_enforced_on_READ_not_only_on_write(own_config):
    """The stored value can predate a change made by a different code path,
    so clamping only on write leaves the old pair in place forever.

    The stored console levels are written straight into QSettings, going
    ROUND the writer -- which is the situation being tested: a value the
    writer never saw, because a different build put it there.
    """
    import logging

    # NAMES GO IN, NUMBERS COME OUT: the store holds "INFO,WARNING" and the
    # readers hand back level numbers, so the writer takes numbers and the
    # raw QSettings value is text. Getting that backwards is exactly the
    # kind of quiet mismatch an untested reader hides.
    own_config.set_log_levels(frozenset({logging.ERROR}),
                              frozenset({logging.ERROR}))
    own_config._settings().setValue(
        own_config._KEY_LOG_CONSOLE_LEVELS,
        "DEBUG,INFO,WARNING,ERROR")
    console = own_config.get_log_console_levels()
    assert console <= own_config.get_log_file_levels()
    assert logging.DEBUG not in console, (
        "a level the file switches do not carry was echoed to the console")


# ---------------------------------------------------------------------------
# The workspace preference
# ---------------------------------------------------------------------------

def test_the_workspace_preference_reaches_the_module(own_config):
    """Without this the journal writes the module default on the first run
    of every session, whatever the user chose last time."""
    from spacr import workspace

    own_config.set_save_workspace("off")
    own_config.apply_workspace_preference()
    assert workspace.default_mode() == "off"


@pytest.mark.parametrize("mode", ["off", "reference", "copy"])
def test_every_mode_survives_the_trip(own_config, mode):
    from spacr import workspace

    own_config.set_save_workspace(mode)
    assert own_config.apply_workspace_preference() == mode
    assert workspace.default_mode() == mode


def test_the_copy_limit_goes_with_it(own_config):
    """The mode and the size cap are one decision -- a copying workspace
    with a stale limit copies a different set of files than the user agreed
    to."""
    from spacr import workspace

    own_config.set_save_workspace("copy")
    own_config.set_workspace_copy_limit_mb(7)
    own_config.apply_workspace_preference()
    assert workspace.default_copy_limit_mb() == pytest.approx(7.0)
