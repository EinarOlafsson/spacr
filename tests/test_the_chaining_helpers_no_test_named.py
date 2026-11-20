"""Auto-chaining helpers no test had ever named.

Instruction 60, on the module whose whole job is to stop a user typing the
wrong plate folder and losing twenty minutes to a run against the previous
plate. Eleven public callables in ``spacr.chaining`` had never appeared in a
test.

The two at the bottom of it are the ones worth pinning hardest.
``is_empty_path`` decides whether there is anything there to keep, and
``same_path`` decides whether the user EDITED the value -- an edit is
remembered forever after, so a comparison that says "different" when the two
name one folder records a pin the user never made, and their screen stops
following the upstream for good.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# is_empty_path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "   ", [], (), [""], [None]])
def test_nothing_there_reads_as_empty(value):
    from spacr.chaining import is_empty_path

    assert is_empty_path(value) is True


def test_every_placeholder_reads_as_empty():
    """"path" is what a settings screen starts from, and treating it as a
    real folder is how a run is launched against a directory named path."""
    from spacr.chaining import is_empty_path, placeholder_paths

    for placeholder in placeholder_paths():
        assert is_empty_path(placeholder) is True, placeholder


def test_a_real_path_is_not_empty(tmp_path):
    from spacr.chaining import is_empty_path

    assert is_empty_path(str(tmp_path)) is False
    assert is_empty_path([str(tmp_path)]) is False


def test_a_list_with_one_real_entry_is_not_empty(tmp_path):
    """Classify keeps its source in a list, and one usable entry among
    blanks is still a source."""
    from spacr.chaining import is_empty_path

    assert is_empty_path(["", str(tmp_path), None]) is False


# ---------------------------------------------------------------------------
# same_path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("left, right, expected", [
    pytest.param(["/plate"], "/plate", True,
                 id="test_a_list_and_a_string_naming_one_folder_are_the_same"),
    pytest.param("/plate1", "/plate2", False,
                 id="test_two_different_folders_are_different"),
    pytest.param(["/a", "/b"], ["/a", "/b"], True,
                 id="test_order_within_a_list_is_not_a_difference"),
])
def test_path_pairs_keep_their_original_identity_results(left, right, expected):
    """Compare each original path pair with the same Boolean identity check."""
    from spacr.chaining import same_path

    assert same_path(left, right) is expected


def test_trailing_separators_and_dot_segments_do_not_matter():
    from spacr.chaining import same_path

    assert same_path("/plate/", "/plate") is True
    assert same_path("/plate/./sub", "/plate/sub") is True


def test_nothing_equals_nothing_and_not_something():
    from spacr.chaining import same_path

    assert same_path(None, []) is True
    assert same_path(None, "/plate") is False


# ---------------------------------------------------------------------------
# looks_laid_out
# ---------------------------------------------------------------------------

def test_a_folder_with_a_layout_directory_looks_laid_out(tmp_path):
    """The cheap structural answer to "is this a project?", nine stat calls
    -- it runs while the mouse button is still down on a drop."""
    from spacr.chaining import layout_directories, looks_laid_out

    name = next(iter(layout_directories()))
    (tmp_path / name).mkdir()
    assert looks_laid_out(str(tmp_path)) is True


def test_an_empty_folder_does_not(tmp_path):
    from spacr.chaining import looks_laid_out

    assert looks_laid_out(str(tmp_path)) is False


def test_a_path_that_is_not_a_folder_does_not(tmp_path):
    """A dropped FILE reaches this too, and os.listdir on one raises."""
    from spacr.chaining import looks_laid_out

    a_file = tmp_path / "notes.txt"
    a_file.write_text("hello", encoding="utf-8")
    assert looks_laid_out(str(a_file)) is False
    assert looks_laid_out("") is False


# ---------------------------------------------------------------------------
# db_candidates
# ---------------------------------------------------------------------------

def test_no_project_gives_no_candidates(tmp_path):
    from spacr.chaining import db_candidates

    assert db_candidates("") == ()
    assert db_candidates(str(tmp_path / "nope")) == ()


def test_the_declared_database_comes_first(tmp_path):
    """Two databases in one project is not an error and not a thing to
    guess about -- it is a question, and the declared one is the answer
    offered first."""
    from spacr import ports as _ports
    from spacr.chaining import db_candidates, ports_for_kinds

    declared = ports_for_kinds((_ports.MEASUREMENTS_DB,))
    if not declared or not declared[0].path:
        pytest.skip("this build declares no measurements database path")
    target = tmp_path / declared[0].path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"SQLite format 3\x00")
    (tmp_path / "stray.db").write_bytes(b"SQLite format 3\x00")

    found = db_candidates(str(tmp_path))
    assert found, "the declared database was not found at all"
    assert os.path.normpath(found[0]) == os.path.normpath(str(target))


def test_a_stray_database_is_offered_too(tmp_path):
    """A user who kept their measurements somewhere else should see it
    listed rather than be told the project has none."""
    from spacr.chaining import db_candidates

    (tmp_path / "measurements").mkdir()
    (tmp_path / "stray.db").write_bytes(b"SQLite format 3\x00")
    found = db_candidates(str(tmp_path))
    assert any(entry.endswith("stray.db") for entry in found), found
