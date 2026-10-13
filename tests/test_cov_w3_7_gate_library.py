"""The named gating-strategy library: a project directory of JSON files.

The library is deliberately Qt-free, so every test here uses a real
``tmp_path`` project and real files. The failure paths are reached by
injection against the filesystem -- a read-only directory, a project that is
a file, a name that is a directory -- rather than by patching ``os``.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from spacr import gate_library
from spacr.gate_library import GateLibraryError


GATES = {"gates": [{"column": "area", "low": 10, "high": 900},
                   {"column": "intensity", "low": 0.1, "high": 0.9}]}


@pytest.fixture
def project(tmp_path):
    """An empty project directory."""
    return str(tmp_path / "project")


def test_a_name_is_never_a_path(project):
    """A name carrying separators cannot climb out of the library directory."""
    target = gate_library.path_for(project, "../../etc/passwd")
    assert os.path.dirname(target) == gate_library.library_dir(project)
    assert ".." not in os.path.basename(target)


def test_a_name_with_nothing_usable_left_is_refused():
    """A name that cleans down to nothing would be written as a hidden ``.json``."""
    for name in ("", "   ", "///", "...", None, "___"):
        with pytest.raises(GateLibraryError, match="strategy name"):
            gate_library.slugify(name)


def test_whitespace_in_a_name_is_collapsed_not_removed():
    """A name is shown in a dropdown, so its spacing is normalised, not stripped."""
    assert gate_library.slugify("  live   singlets  ") == "live singlets"


def test_a_saved_strategy_reads_back_and_is_listed(project):
    """Save, list and load agree on the name that was used."""
    written = gate_library.save(project, "live singlets", GATES)
    assert written.endswith(os.path.join("gates", "live singlets.json"))
    assert gate_library.list_strategies(project) == ["live singlets"]
    assert gate_library.load(project, "live singlets") == GATES


def test_a_library_that_does_not_exist_is_empty_not_an_error(project):
    """A dropdown that cannot be filled is not a reason to refuse a screen."""
    assert gate_library.list_strategies(project) == []


def test_a_library_directory_that_cannot_be_created_says_so(tmp_path):
    """A project path that is a file cannot hold a library, and the error names it."""
    blocker = tmp_path / "not-a-project"
    blocker.write_text("this is a file", encoding="utf-8")
    with pytest.raises(GateLibraryError, match="cannot create the gate library"):
        gate_library.save(str(blocker), "anything", GATES)


def test_a_read_only_library_reports_the_write_failure(project):
    """A directory the user cannot write leaves no partial file behind."""
    gate_library.save(project, "first", GATES)
    directory = gate_library.library_dir(project)
    mode = os.stat(directory).st_mode
    os.chmod(directory, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(GateLibraryError, match="cannot write"):
            gate_library.save(project, "second", GATES)
    finally:
        os.chmod(directory, mode)
    # The half-written temporary file was cleaned up, so the library still
    # lists exactly what it did before.
    assert gate_library.list_strategies(project) == ["first"]
    assert not [name for name in os.listdir(directory)
                if name.endswith(".part")]


def test_a_payload_that_will_not_serialise_is_refused_before_any_file_is_made(
        project):
    """A strategy that cannot round-trip is never stored half-way."""
    with pytest.raises(GateLibraryError, match="cannot be saved"):
        gate_library.save(project, "impossible", {"gates": {1, 2, 3}})
    assert gate_library.list_strategies(project) == []


def test_saving_the_same_name_twice_replaces_it(project):
    """A library holds one strategy per name, and the newest wins."""
    gate_library.save(project, "filter", {"gates": []})
    gate_library.save(project, "filter", GATES)
    assert gate_library.list_strategies(project) == ["filter"]
    assert gate_library.load(project, "filter") == GATES


def test_loading_a_missing_strategy_lists_the_ones_that_exist(project):
    """The error names the alternatives so a user can pick from the message."""
    gate_library.save(project, "live singlets", GATES)
    gate_library.save(project, "debris", GATES)
    with pytest.raises(GateLibraryError) as excinfo:
        gate_library.load(project, "infected")
    message = str(excinfo.value)
    assert "no saved strategy called 'infected'" in message
    assert "debris" in message and "live singlets" in message


def test_loading_from_an_empty_library_says_there_are_none(project):
    """With nothing saved the message says so rather than listing an empty set."""
    with pytest.raises(GateLibraryError, match="this project has none"):
        gate_library.load(project, "anything")


def test_a_corrupt_strategy_names_itself_in_the_error(project):
    """"Expecting value: line 1" on its own does not say which file to fix."""
    gate_library.save(project, "broken", GATES)
    path = gate_library.path_for(project, "broken")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not json at all")
    with pytest.raises(GateLibraryError,
                       match="the saved strategy 'broken' could not be read"):
        gate_library.load(project, "broken")


def test_delete_reports_whether_there_was_anything_to_delete(project):
    """Deleting is idempotent and says which of the two happened."""
    gate_library.save(project, "filter", GATES)
    assert gate_library.delete(project, "filter") is True
    assert gate_library.delete(project, "filter") is False
    assert gate_library.list_strategies(project) == []


def test_a_strategy_that_cannot_be_removed_reports_its_name(project):
    """An unlink that fails for any other reason keeps the name in the message."""
    os.makedirs(gate_library.library_dir(project), exist_ok=True)
    # A directory occupying the strategy's path: unlink refuses it.
    os.makedirs(gate_library.path_for(project, "wedged"))
    with pytest.raises(GateLibraryError, match="cannot remove 'wedged'"):
        gate_library.delete(project, "wedged")


def test_describe_counts_the_gates_without_applying_them(project):
    """A list row shows the gate count next to the name."""
    gate_library.save(project, "live singlets", GATES)
    assert gate_library.describe(project, "live singlets") == (2, None)


def test_describe_reports_a_broken_strategy_instead_of_a_count(project):
    """A file that will not read shows its error in the list, not on apply."""
    gate_library.save(project, "broken", GATES)
    with open(gate_library.path_for(project, "broken"), "w",
              encoding="utf-8") as handle:
        handle.write("[[[")
    count, error = gate_library.describe(project, "broken")
    assert count == 0
    assert "could not be read" in error


def test_a_mapping_without_a_gates_list_is_counted_by_its_keys(project):
    """An older ``{column: bounds}`` strategy still reports how big it is."""
    payload = {"area": [10, 900], "intensity": [0.1, 0.9], "solidity": [0, 1]}
    gate_library.save(project, "by column", payload)
    assert gate_library.describe(project, "by column") == (3, None)


def test_a_mapping_whose_gates_key_is_not_a_list_falls_back_to_key_count(
        project):
    """``gates`` holding a mapping is counted as the document's own keys."""
    gate_library.save(project, "odd", {"gates": {"area": [1, 2]},
                                       "version": 2})
    assert gate_library.describe(project, "odd") == (2, None)


def test_a_bare_list_of_gates_is_counted_directly(project):
    """A strategy saved as a plain list is a valid document."""
    gate_library.save(project, "plain", [{"column": "area"},
                                         {"column": "intensity"}])
    assert gate_library.describe(project, "plain") == (2, None)


def test_something_that_is_not_a_strategy_says_so(project):
    """A JSON scalar is readable but is not a gating strategy."""
    gate_library.save(project, "scalar", 42)
    count, error = gate_library.describe(project, "scalar")
    assert count == 0
    assert error == "'scalar' does not look like a gating strategy"


def test_only_json_files_are_listed(project):
    """Stray files in the library directory are not offered as strategies."""
    directory = gate_library.library_dir(project)
    os.makedirs(directory, exist_ok=True)
    gate_library.save(project, "real", GATES)
    with open(os.path.join(directory, "notes.txt"), "w",
              encoding="utf-8") as handle:
        handle.write("scratch")
    # A file called exactly ".json" has no name at all and must not appear.
    with open(os.path.join(directory, ".json"), "w",
              encoding="utf-8") as handle:
        handle.write("{}")
    assert gate_library.list_strategies(project) == ["real"]


def test_the_saved_file_is_readable_json(project):
    """The library is a directory of plain JSON, readable without spaCR."""
    path = gate_library.save(project, "live singlets", GATES)
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle) == GATES
