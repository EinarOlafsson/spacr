"""The modules the suite had never named once.

Instruction 60 asks for every module covered, and the honest first question
is which ones no test so much as MENTIONS -- because a module nobody has
imported in a test has not been read by anything except the person who wrote
it. Seven of 452 were in that state; these are the ones that are code a user
reaches rather than generated tables or icon builders.

WHAT IS ASSERTED IS THE BEHAVIOUR AT THE EDGES, not the happy path alone.
The happy path in all three is short and obvious; the value is in what they
do with a run folder that is not there, a settings CSV from an older build,
and a colour left blank -- which is where a quiet wrong answer would come
from.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# spacr-workspace
# ---------------------------------------------------------------------------

def test_a_missing_run_folder_is_named_not_traced(tmp_path, capsys):
    """A path that is not there is the commonest way to use this wrong, and
    an argparse traceback is not an answer to it."""
    from spacr.cli_workspace import main

    code = main([str(tmp_path / "no-such-run")])
    assert code == 2
    assert "no such run folder" in capsys.readouterr().err


def test_a_run_saved_with_the_feature_off_says_which_it_was(tmp_path,
                                                            capsys):
    """NAMED, not "not found". A run saved with save_workspace='off' is a
    different thing from one whose bundle failed to write, and the user's
    next step differs."""
    from spacr.cli_workspace import main

    run = tmp_path / "run"
    run.mkdir()
    code = main([str(run)])
    assert code == 2
    said = capsys.readouterr().err
    assert "carries no" in said
    assert "save_workspace" in said, (
        "the message has to name the setting that would have written it")


def _write_workspace(run):
    from spacr.workspace import DOC_NAME

    run.mkdir(parents=True, exist_ok=True)
    document = {
        "version": 1,
        "panels": [{"kind": "regression", "title": "ols_4"}],
        "files": [{"role": "database", "path": "measurements.db"}],
    }
    (run / DOC_NAME).write_text(json.dumps(document), encoding="utf-8")
    return document


def test_the_document_is_printed_unchanged_with_json(tmp_path, capsys):
    """``--json`` exists so a script can read it; reformatting it there
    would make the flag useless for the one job it has."""
    from spacr.cli_workspace import main

    run = tmp_path / "run"
    written = _write_workspace(run)
    assert main([str(run), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == written


def test_the_file_inventory_prints_a_state_per_file(tmp_path, capsys):
    from spacr.cli_workspace import main

    run = tmp_path / "run"
    _write_workspace(run)
    assert main([str(run), "--files"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.strip()]
    assert lines, "the inventory printed nothing at all"
    assert "measurements.db" in "\n".join(lines)


def test_the_default_is_a_readable_summary(tmp_path, capsys):
    from spacr.cli_workspace import main

    run = tmp_path / "run"
    _write_workspace(run)
    assert main([str(run)]) == 0
    assert capsys.readouterr().out.strip(), "the summary was empty"


def test_the_workspace_json_itself_can_be_named(tmp_path, capsys):
    """The usage says a run folder OR its workspace.json, and a user who
    tab-completed to the file should not be told there is no run."""
    from spacr.cli_workspace import main
    from spacr.workspace import DOC_NAME

    run = tmp_path / "run"
    _write_workspace(run)
    assert main([str(run / DOC_NAME), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == 1


# ---------------------------------------------------------------------------
# The channel mapping editor
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6")


@pytest.fixture()
def mapping(qapp):
    from spacr.qt.widgets.channel_mapping import ChannelMappingWidget

    made = ChannelMappingWidget()
    yield made
    made.deleteLater()


def test_the_default_is_the_pipelines_own_default(mapping):
    from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING

    assert mapping.get_value() == dict(DEFAULT_PNG_CHANNEL_MAPPING)


def test_a_dict_round_trips(mapping):
    mapping.set_value({"r": 2, "g": 1, "b": 0})
    assert mapping.get_value() == {"r": 2, "g": 1, "b": 0}


def test_a_settings_csv_string_is_read_back(mapping):
    """A settings CSV stores ``repr(value)``, so the mapping arrives as
    TEXT -- the same round trip every other list-shaped key has to survive."""
    mapping.set_value("{'r': 1, 'g': 0, 'b': 2}")
    assert mapping.get_value() == {"r": 1, "g": 0, "b": 2}


def test_the_legacy_list_is_translated_the_way_the_pipeline_does(mapping):
    """``png_dims`` never said which colour it meant: position 0 was blue
    because of how cv2 reads an array. The fields have to show the colours
    the run will produce, not a rearrangement of them."""
    from spacr.crops import png_dims_to_channel_mapping

    mapping.set_value([0, 1, 2])
    assert mapping.get_value() == png_dims_to_channel_mapping([0, 1, 2])


def test_nonsense_falls_back_rather_than_raising(mapping):
    """A settings file can hold anything, and a panel that raises on load
    takes the whole screen with it."""
    from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING

    for junk in ("not a dict", "{'r':", {"r": "red"}, 17, object()):
        mapping.set_value(junk)
        assert set(mapping.get_value()) == set(DEFAULT_PNG_CHANNEL_MAPPING)


def test_an_empty_colour_is_none_not_zero(mapping):
    """Zero IS a channel. A blank field meaning zero would silently map the
    first source channel into a colour the user left out."""
    from spacr.qt.widgets.channel_mapping import _EMPTY

    mapping._boxes["g"].setValue(_EMPTY)
    assert mapping.get_value()["g"] is None


def test_a_change_is_announced_once(mapping):
    """The panel listens for this to keep the preview in step; a set_value
    that announced nothing would leave the preview showing the old crop."""
    seen = []
    mapping.valueChanged.connect(seen.append)
    mapping.set_value({"r": 0, "g": 1, "b": 2})
    assert seen and seen[-1] == {"r": 0, "g": 1, "b": 2}


def test_loading_does_not_fire_once_per_field(mapping):
    """Three signals for one load is three previews rendered to show one
    change, and the first two are of a half-applied mapping."""
    seen = []
    mapping.valueChanged.connect(seen.append)
    mapping.set_value({"r": 2, "g": 1, "b": 0})
    assert len(seen) == 1, seen


# ---------------------------------------------------------------------------
# The DNA rain settings popover
# ---------------------------------------------------------------------------

def _dna_button(qapp):
    from spacr.qt.widgets.dna_rain import DnaRainSettingsBar
    from spacr.qt.widgets.dna_rain_settings import DnaSettingsButton

    return DnaSettingsButton(DnaRainSettingsBar(vertical=True))


def test_the_dna_button_reads_dna(qapp):
    """It is the AI toggle's twin by construction, so the one thing that
    distinguishes them is the word on it."""
    button = _dna_button(qapp)
    try:
        assert button.text().upper() == "DNA"
        assert button.objectName() == "AiToggleLabel", (
            "built from AiToggleLabel rather than styled to look like one, "
            "so it follows the same QSS through a theme switch")
    finally:
        button.deleteLater()


def test_the_button_owns_the_popover_and_the_bar(qapp):
    """A popover nothing holds is collected the moment the constructor
    returns, and the button opens an empty window."""
    from spacr.qt.widgets.dna_rain import DnaRainSettingsBar
    from spacr.qt.widgets.dna_rain_settings import DnaRainSettingsPopover

    button = _dna_button(qapp)
    try:
        assert isinstance(button.popover, DnaRainSettingsPopover)
        assert isinstance(button.settings_bar, DnaRainSettingsBar)
        assert button.settings_bar is button.popover.bar
    finally:
        button.deleteLater()


def test_the_bar_is_reparented_into_the_popover_not_left_behind(qapp):
    """`addWidget` does the reparenting. An explicit setParent first would
    mark the bar hidden and it would stay blank inside a shown popover."""
    button = _dna_button(qapp)
    try:
        bar = button.settings_bar
        assert button.popover.isAncestorOf(bar)
        assert not bar.isHidden()
    finally:
        button.deleteLater()


def test_the_settings_start_closed(qapp):
    """A popover that opens itself is a popover; this one is a button."""
    button = _dna_button(qapp)
    try:
        assert not button.is_open()
    finally:
        button.deleteLater()


def test_the_popover_takes_its_colours_from_the_palette(qapp):
    """A second stylesheet is a second thing to keep in step with the
    theme, which is what this class was built to avoid."""
    button = _dna_button(qapp)
    try:
        button.popover.apply_theme()
        assert button.popover.styleSheet(), (
            "the popover painted nothing of its own, so it would show the "
            "default window chrome under every theme")
    finally:
        button.deleteLater()
