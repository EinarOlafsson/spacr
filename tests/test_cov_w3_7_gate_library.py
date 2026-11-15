"""The named gating-strategy library: a project directory of JSON files.

The library is deliberately Qt-free, so every test here uses a real
``tmp_path`` project and real files. The failure paths are reached by
injection against the filesystem -- a read-only directory, a project that is
a file, a name that is a directory -- rather than by patching ``os``.
"""
from __future__ import annotations

import os

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


def test_a_library_that_does_not_exist_is_empty_not_an_error(project):
    """A dropdown that cannot be filled is not a reason to refuse a screen."""
    assert gate_library.list_strategies(project) == []


def test_loading_from_an_empty_library_says_there_are_none(project):
    """With nothing saved the message says so rather than listing an empty set."""
    with pytest.raises(GateLibraryError, match="this project has none"):
        gate_library.load(project, "anything")


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
