"""Named gating strategies. Instruction 31, "saveable filter sets".

The Gate Editor could already write a strategy to a path and read one back,
but only through a file chooser, so reuse meant remembering where you put it.
A screen gets gated the same way over and over, and that is a library.
"""
from __future__ import annotations

import json
import os

import pytest

from spacr.gate_library import (
    GateLibraryError,
    delete,
    describe,
    library_dir,
    list_strategies,
    load,
    path_for,
    save,
    slugify,
)


STRATEGY = {"gates": [{"name": "singlets", "column": "cell_area",
                       "low": 100, "high": 900}]}


# ---------------------------------------------------------------------------
# A name is not a path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("live singlets", "live singlets"),
    ("infected/cells", "infected-cells"),
    ("../../etc/passwd", "etc-passwd"),
    ("weird:*name?", "weird-name"),
    ("  padded  ", "padded"),
])
def test_a_name_becomes_a_safe_filename(name, expected):
    assert slugify(name) == expected


@pytest.mark.parametrize("name", ["", "   ", "...", "///", "/", ".."])
def test_a_name_with_nothing_usable_is_refused(name):
    """Writing it would produce `.json` -- an invisible file with no name."""
    with pytest.raises(GateLibraryError):
        slugify(name)


def test_a_saved_strategy_cannot_escape_the_project(tmp_path):
    """The reason slugify exists, asserted rather than assumed."""
    project = str(tmp_path / "plate1")
    target = path_for(project, "../../../etc/passwd")
    assert os.path.commonpath([os.path.abspath(target),
                               os.path.abspath(library_dir(project))]) \
        == os.path.abspath(library_dir(project))


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------

def test_a_strategy_saves_and_loads_by_name(tmp_path):
    project = str(tmp_path)
    save(project, "live singlets", STRATEGY)
    assert load(project, "live singlets") == STRATEGY


def test_the_library_lists_what_is_in_it(tmp_path):
    project = str(tmp_path)
    save(project, "b strategy", STRATEGY)
    save(project, "a strategy", STRATEGY)
    assert list_strategies(project) == ["a strategy", "b strategy"]


def test_an_empty_project_lists_nothing_rather_than_raising(tmp_path):
    """A dropdown that cannot be filled is not a reason to refuse a screen."""
    assert list_strategies(str(tmp_path / "no-such-project")) == []


def test_saving_the_same_name_twice_replaces_it(tmp_path):
    project = str(tmp_path)
    save(project, "s", STRATEGY)
    save(project, "s", {"gates": []})
    assert load(project, "s") == {"gates": []}
    assert list_strategies(project) == ["s"]


def test_a_missing_strategy_says_what_the_project_does_have(tmp_path):
    project = str(tmp_path)
    save(project, "the one that exists", STRATEGY)
    with pytest.raises(GateLibraryError, match="the one that exists"):
        load(project, "a name nobody saved")


def test_a_missing_strategy_in_an_empty_project_says_so(tmp_path):
    with pytest.raises(GateLibraryError, match="none"):
        load(str(tmp_path), "anything")


def test_an_unreadable_strategy_names_itself(tmp_path):
    """"expecting value: line 1" tells a user nothing about which to fix."""
    project = str(tmp_path)
    save(project, "broken", STRATEGY)
    with open(path_for(project, "broken"), "w", encoding="utf-8") as handle:
        handle.write("{not json")

    with pytest.raises(GateLibraryError, match="broken"):
        load(project, "broken")


def test_something_that_will_not_serialise_is_refused_before_writing(tmp_path):
    """A half-written file the library still lists is worse than a refusal."""
    project = str(tmp_path)
    with pytest.raises(GateLibraryError):
        save(project, "impossible", {"gate": object()})
    assert list_strategies(project) == []


def test_an_interrupted_save_leaves_the_previous_strategy_intact(tmp_path):
    """Written whole and moved, so the old one survives a failure."""
    project = str(tmp_path)
    save(project, "s", STRATEGY)
    with pytest.raises(GateLibraryError):
        save(project, "s", {"bad": object()})
    assert load(project, "s") == STRATEGY
    assert not [f for f in os.listdir(library_dir(project))
                if f.endswith(".part")]


# ---------------------------------------------------------------------------
# Delete and describe
# ---------------------------------------------------------------------------

def test_delete_removes_one_and_reports_whether_there_was_one(tmp_path):
    project = str(tmp_path)
    save(project, "s", STRATEGY)
    assert delete(project, "s") is True
    assert delete(project, "s") is False
    assert list_strategies(project) == []


def test_describe_counts_the_gates_without_applying_them(tmp_path):
    project = str(tmp_path)
    save(project, "s", STRATEGY)
    count, error = describe(project, "s")
    assert count == 1 and error is None


def test_describe_reports_a_broken_file_instead_of_a_count(tmp_path):
    """So a broken strategy is visible in the LIST, not at apply time."""
    project = str(tmp_path)
    save(project, "broken", STRATEGY)
    with open(path_for(project, "broken"), "w", encoding="utf-8") as handle:
        handle.write("nonsense")

    count, error = describe(project, "broken")
    assert count == 0 and error and "broken" in error


def test_the_file_is_json_a_human_can_read(tmp_path):
    """A gating strategy is something a user may want to diff or edit."""
    project = str(tmp_path)
    save(project, "s", STRATEGY)
    text = open(path_for(project, "s"), encoding="utf-8").read()
    assert "\n" in text, "indent=2 keeps it readable"
    assert json.loads(text) == STRATEGY


def test_the_module_is_importable_without_qt():
    """Same promise spacr.filters makes: testable with no display."""
    import subprocess
    import sys

    code = ("import sys, spacr.gate_library; "
            "assert not [m for m in sys.modules if m.startswith('PySide6')]")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
