"""Table-driven tests of the named gating-strategy library, ``spacr.gate_library``.

The library is a project directory of plain JSON files, Qt-free, so every case
uses a real ``tmp_path`` project.  Failure paths are reached through the
filesystem itself (a project path that is a file, a read-only library, a
corrupt file) rather than by patching ``os``.  One general test per unit,
each walking a case table with pytest's ``subtests`` fixture and asserting the
exact value or the exact refusal message.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from spacr import gate_library
from spacr.gate_library import GateLibraryError
from tests.refused_access import refused

GATES = {"gates": [{"column": "area", "low": 10, "high": 900},
                   {"column": "intensity", "low": 0.1, "high": 0.9}]}


def _corrupt(project, name, text):
    """Overwrite a saved strategy with text that is not valid JSON."""
    with open(gate_library.path_for(project, name), "w",
              encoding="utf-8") as handle:
        handle.write(text)


def test_a_saved_strategy_reads_back_lists_and_is_described(tmp_path, subtests):
    """save -> list/load/describe agree, for every payload shape the library accepts."""
    # (case, payload, expected describe())
    cases = [
        ("gates_list", GATES, (2, None)),
        ("column_mapping_counted_by_keys",
         {"area": [10, 900], "intensity": [0.1, 0.9], "solidity": [0, 1]},
         (3, None)),
        ("gates_key_not_a_list_counts_document_keys",
         {"gates": {"area": [1, 2]}, "version": 2}, (2, None)),
        ("bare_list", [{"column": "area"}, {"column": "intensity"}], (2, None)),
        ("scalar_is_not_a_strategy", 42,
         (0, "'scalar_is_not_a_strategy' does not look like a gating strategy")),
    ]
    for case_id, payload, described in cases:
        with subtests.test(msg=case_id):
            project = str(tmp_path / case_id)
            written = gate_library.save(project, case_id, payload)
            assert written == os.path.join(project, "gates", case_id + ".json")
            assert gate_library.list_strategies(project) == [case_id]
            assert gate_library.load(project, case_id) == payload
            # The file is plain sorted, indented JSON readable without spaCR.
            with open(written, encoding="utf-8") as handle:
                text = handle.read()
            assert json.loads(text) == payload
            assert text == json.dumps(payload, indent=2, sort_keys=True)
            assert gate_library.describe(project, case_id) == described
            assert not [n for n in os.listdir(os.path.dirname(written))
                        if n.endswith(".part")]

    with subtests.test(msg="same_name_twice_newest_wins"):
        project = str(tmp_path / "twice")
        gate_library.save(project, "filter", {"gates": []})
        assert gate_library.describe(project, "filter") == (0, None)
        gate_library.save(project, "filter", GATES)
        assert gate_library.list_strategies(project) == ["filter"]
        assert gate_library.load(project, "filter") == GATES

    with subtests.test(msg="only_named_json_files_are_listed_sorted"):
        project = str(tmp_path / "stray")
        gate_library.save(project, "real", GATES)
        gate_library.save(project, "another", GATES)
        directory = gate_library.library_dir(project)
        for stray, text in (("notes.txt", "scratch"), (".json", "{}")):
            with open(os.path.join(directory, stray), "w",
                      encoding="utf-8") as handle:
                handle.write(text)
        assert gate_library.list_strategies(project) == ["another", "real"]


def test_the_library_refuses_by_name_and_leaves_nothing_half_written(
        tmp_path, subtests):
    """Every failure is a GateLibraryError whose message names what went wrong."""
    with subtests.test(msg="project_path_is_a_file"):
        blocker = tmp_path / "not-a-project"
        blocker.write_text("this is a file", encoding="utf-8")
        with pytest.raises(GateLibraryError) as excinfo:
            gate_library.save(str(blocker), "anything", GATES)
        assert str(excinfo.value).startswith("cannot create the gate library: ")

    with subtests.test(msg="unserialisable_payload_makes_no_file"):
        project = str(tmp_path / "unserialisable")
        with pytest.raises(GateLibraryError) as excinfo:
            gate_library.save(project, "impossible", {"gates": {1, 2, 3}})
        assert str(excinfo.value).startswith(
            "that gating strategy cannot be saved: ")
        assert gate_library.list_strategies(project) == []
        assert os.listdir(gate_library.library_dir(project)) == []

    with subtests.test(msg="read_only_library_cleans_its_part_file"):
        project = str(tmp_path / "readonly")
        gate_library.save(project, "first", GATES)
        directory = gate_library.library_dir(project)
        target = gate_library.path_for(project, "second")
        with refused(directory, stat.S_IRUSR | stat.S_IXUSR, windows="write"):
            with pytest.raises(GateLibraryError) as excinfo:
                gate_library.save(project, "second", GATES)
        assert str(excinfo.value).startswith(f"cannot write {target}: ")
        assert sorted(os.listdir(directory)) == ["first.json"]

    with subtests.test(msg="missing_strategy_lists_the_others"):
        project = str(tmp_path / "missing")
        gate_library.save(project, "live singlets", GATES)
        gate_library.save(project, "debris", GATES)
        with pytest.raises(GateLibraryError) as excinfo:
            gate_library.load(project, "infected")
        assert str(excinfo.value) == ("no saved strategy called 'infected'; "
                                      "this project has debris, live singlets")
        assert gate_library.describe(project, "infected") == (
            0, "no saved strategy called 'infected'; "
               "this project has debris, live singlets")

    with subtests.test(msg="missing_strategy_in_an_empty_library"):
        with pytest.raises(GateLibraryError) as excinfo:
            gate_library.load(str(tmp_path / "empty"), "anything")
        assert str(excinfo.value) == ("no saved strategy called 'anything'; "
                                      "this project has none")

    for case_id, text in (("corrupt_brace", "{not json at all"),
                          ("corrupt_brackets", "[[[")):
        with subtests.test(msg=case_id):
            project = str(tmp_path / case_id)
            gate_library.save(project, "broken", GATES)
            _corrupt(project, "broken", text)
            prefix = "the saved strategy 'broken' could not be read: "
            with pytest.raises(GateLibraryError) as excinfo:
                gate_library.load(project, "broken")
            assert str(excinfo.value).startswith(prefix)
            count, error = gate_library.describe(project, "broken")
            assert count == 0
            assert error == str(excinfo.value)
