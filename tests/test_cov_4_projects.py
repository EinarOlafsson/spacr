"""The project browser reports what it found without walking off a cliff.

Everything here is something a real folder does to a browser: a file that
vanishes between being listed and being stat'ed, a symlink that points back
at its own parent, a module key that no longer exists, a project the registry
has never heard of. None of them may raise -- the browser's whole job is to
show a list, and a list that refuses to appear because one entry was odd is
worse than a list with one odd entry in it.
"""
from __future__ import annotations

import os

import pytest

from spacr import ports as _ports
from spacr import projects
from spacr.projects import (STATE_DONE, STATE_PARTIAL, ModuleState,
                            ProjectSummary, StaleArtifact, discover,
                            format_project, looks_like_project, module_states,
                            pipeline_order)


# -- the ladder --------------------------------------------------------------

def test_a_cycle_in_the_port_graph_is_broken_rather_than_raised(monkeypatch):
    """A plugin that declares a loop must not stop the browser listing."""
    real = projects.producing_modules()
    pair = tuple(sorted(real))[:2]
    monkeypatch.setattr(projects, "producing_modules", lambda: pair)
    monkeypatch.setattr(
        _ports, "upstream_modules",
        lambda module: (pair[1],) if module == pair[0] else (pair[0],))
    order = pipeline_order()
    assert sorted(order) == sorted(pair)
    assert len(order) == len(set(order))


def test_the_declared_ladder_has_no_cycle_to_break():
    """Every producing module spaCR ships appears exactly once, in order."""
    order = pipeline_order()
    assert sorted(order) == sorted(projects.producing_modules())


# -- mtimes ------------------------------------------------------------------

def test_a_file_that_vanished_does_not_stop_the_others_being_timed(tmp_path):
    """Between listing and stat'ing, a cleanup job can delete the file."""
    present = tmp_path / "there.npy"
    present.write_bytes(b"x")
    port = _ports.Port(kind="merged-arrays", role="merged")
    resolved = _ports.ResolvedPort(
        port=port, root=str(tmp_path),
        target=str(tmp_path / "gone.npy"),
        paths=(str(tmp_path / "gone.npy"), str(present)),
        exists=True, count=2)
    assert projects._newest_mtime_ns(resolved) > 0


def test_a_port_whose_every_path_vanished_is_simply_undated(tmp_path):
    """No readable file means no timestamp, not a raised OSError."""
    port = _ports.Port(kind="merged-arrays", role="merged")
    resolved = _ports.ResolvedPort(
        port=port, root=str(tmp_path), target=str(tmp_path / "gone.npy"),
        paths=(str(tmp_path / "gone.npy"),), exists=False, count=0)
    assert projects._newest_mtime_ns(resolved) == 0


# -- module states -----------------------------------------------------------

def test_a_partial_module_says_what_it_has_and_what_it_lacks():
    """"Partial" without the two lists gives the user nothing to act on."""
    state = ModuleState(module="measure", state=STATE_PARTIAL,
                        found=("measurements",), missing=("crops",))
    line = state.describe()
    assert "partial" in line
    assert "measurements" in line and "crops" in line


def test_a_module_key_that_no_longer_exists_is_skipped(tmp_path):
    """A saved workspace can name a module a later spaCR does not have."""
    states = module_states(tmp_path, modules=["no_such_module"])
    assert states == ()


# -- what counts as a project ------------------------------------------------

def test_a_path_that_is_not_a_folder_is_not_a_project(tmp_path):
    """A file dropped on the browser is not a project root."""
    plain = tmp_path / "notes.txt"
    plain.write_text("hello")
    assert looks_like_project(plain) is False


def test_an_archived_project_is_still_a_project(tmp_path):
    """An archive manifest is the strongest evidence a folder is a project."""
    from spacr import data_manager as _dm
    (tmp_path / _dm.ARCHIVE_MANIFEST_NAME).write_text("{}")
    assert looks_like_project(tmp_path) is True


# -- discovery ---------------------------------------------------------------

def _make_project(path):
    from spacr import data_manager as _dm
    path.mkdir(parents=True, exist_ok=True)
    (path / _dm.ARCHIVE_MANIFEST_NAME).write_text("{}")
    return path


def test_discovery_stops_at_the_limit(tmp_path):
    """A browser pointed at a home directory must return, not walk it all."""
    for name in ("a", "b", "c", "d"):
        _make_project(tmp_path / name)
    found = discover([tmp_path], depth=2, limit=2)
    assert len(found) == 2


def test_a_symlink_loop_is_visited_once(tmp_path):
    """A folder linked back into itself would otherwise recurse forever."""
    root = tmp_path / "root"
    inner = root / "inner"
    _make_project(inner)
    os.symlink(root, root / "loop", target_is_directory=True)
    found = discover([root, root / "loop"], depth=3, limit=10)
    assert [os.path.basename(p) for p in found] == ["inner"]


def test_hidden_and_skipped_folders_are_not_searched(tmp_path):
    """A .git or __pycache__ full of files is never a project."""
    skipped = sorted(projects.SKIP_DIRS)[0]
    _make_project(tmp_path / ".hidden")
    _make_project(tmp_path / skipped)
    _make_project(tmp_path / "real")
    found = discover([tmp_path], depth=2, limit=10)
    assert [os.path.basename(p) for p in found] == ["real"]


def test_a_plain_file_beside_the_projects_is_not_descended_into(tmp_path):
    """scandir returns files too, and a file has nothing to walk."""
    (tmp_path / "readme.txt").write_text("hello")
    _make_project(tmp_path / "real")
    found = discover([tmp_path], depth=2, limit=10)
    assert [os.path.basename(p) for p in found] == ["real"]


def test_a_blank_root_is_ignored_rather_than_searched(tmp_path):
    """An empty recent-folders entry must not turn into the cwd."""
    _make_project(tmp_path / "real")
    found = discover(["", None, tmp_path], depth=2, limit=10)
    assert [os.path.basename(p) for p in found] == ["real"]


# -- stale artifacts ---------------------------------------------------------

def test_a_gone_artifact_says_where_it_was():
    """Availability is a different problem from staleness and reads so."""
    entry = StaleArtifact(artifact_id="1", kind="measurements-db",
                          module="measure", role="db",
                          path="/data/p7/measurements.db", missing=True)
    assert entry.describe() == ("measurements-db from measure: gone from "
                                "/data/p7/measurements.db")


# -- summaries ---------------------------------------------------------------

def _entry(missing=False):
    return StaleArtifact(artifact_id="1", kind="k", module="measure",
                         role="db", path="/data/p/x.db", missing=missing,
                         reasons=("an input moved on",))


def test_a_project_with_both_stale_and_gone_results_counts_both():
    """Reporting only one of the two understates what needs re-running."""
    summary = ProjectSummary(root="/data/p", known=True,
                             stale=(_entry(),), missing=(_entry(True),))
    assert summary.staleness_note() == "1 stale, 1 gone"


def test_a_clean_registered_project_has_nothing_to_note():
    """A blank note is what "nothing is wrong" looks like in the table."""
    summary = ProjectSummary(root="/data/p", known=True, stage="measure",
                             size_bytes=1000, unregistered_bytes=1)
    assert summary.note() == ""


def test_next_steps_that_cannot_be_computed_are_simply_not_offered(
        monkeypatch, tmp_path):
    """A chaining failure must not take the whole project row with it."""
    from spacr import chaining

    def _explode(*_args, **_kwargs):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(chaining, "next_steps", _explode)
    assert projects._next_steps("measure", str(tmp_path), None) == ()


# -- the report --------------------------------------------------------------

def test_a_project_nothing_has_run_in_says_so_in_words():
    """"never" is an answer; a blank cell reads as a report that failed."""
    text = format_project(ProjectSummary(root="/data/p", name="p"))
    assert "  last run: never" in text.splitlines()


def test_the_report_lists_the_gone_results_and_the_unreadable_paths():
    """A path that could not be read is the reason a total looks wrong."""
    summary = ProjectSummary(
        root="/data/p", name="p", known=True, n_artifacts=2,
        stale=(_entry(),), missing=(_entry(True),),
        next_steps=(("classify", ""), ("regression", "needs scores")),
        errors=("/data/p/locked",))
    lines = format_project(summary).splitlines()
    assert any(line.startswith("    gone   ") for line in lines), lines
    assert "  could not read /data/p/locked" in lines
    assert "  next: classify" in lines


def test_a_second_root_is_not_searched_once_the_limit_is_full(tmp_path):
    """Roots are walked in turn, so the limit has to hold between them."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_project(first / "a")
    _make_project(first / "b")
    _make_project(second / "c")
    found = discover([first, second], depth=2, limit=2)
    assert [os.path.basename(p) for p in found] == ["a", "b"]
