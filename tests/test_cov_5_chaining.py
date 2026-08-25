"""Chaining answers, or says why it cannot -- it never guesses a path.

The value of auto-chaining is that a field filled for you holds the same
string you would have typed. Every branch here is a place where the honest
answer is "nothing", "ask the user", or "I could not write that down": a
read-only pin file, a registry that will not open, a folder that cannot be
listed, a drop that resolved to two candidates. A wrong guess in any of them
points a run at another plate's data.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from spacr import artifacts, chaining, ports


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """No test here touches the developer's registry or their real pins."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)
    monkeypatch.setenv(chaining.PIN_STATE_ENV,
                       str(tmp_path / "state" / "pins.json"))
    chaining.pin_store(refresh=True)
    yield
    chaining.pin_store(refresh=True)


def _plate(root: Path) -> str:
    """A plate folder shaped the way the mask pipeline leaves one."""
    (root / "merged").mkdir(parents=True, exist_ok=True)
    for index in range(2):
        np.save(root / "merged" / f"plate1_A01_{index}.npy",
                np.zeros((6, 6, 3), dtype=np.uint16))
    return str(root)


def _run_mask(root: str):
    return artifacts.register_run_outputs(
        "mask", {"src": root, "cell_channel": 0, "cell_diameter": 30},
        registry=artifacts.open_registry(root))


# --------------------------------------------------------------------------- #
#  Pins
# --------------------------------------------------------------------------- #

def test_a_pin_that_cannot_be_written_still_holds_for_this_session(tmp_path):
    """An unwritable pin file loses persistence, never the pin itself.

    The pin is the user's own choice of path, made seconds ago. Raising here
    would take down the settings screen over a home directory the process
    cannot write; forgetting it immediately would silently re-chain the field
    under them while they were still looking at it.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("this is a file, not a folder")
    store = chaining.PinStore(str(blocker / "pins.json"))

    store.pin("measure", "src", "/data/plate1/merged")

    assert store.pinned("measure", "src") == "/data/plate1/merged"
    assert not (tmp_path / "not-a-directory" / "pins.json").exists()


# --------------------------------------------------------------------------- #
#  Where an artifact's project root is
# --------------------------------------------------------------------------- #

def test_an_unregistered_project_root_is_climbed_from_the_port(tmp_path):
    """Without a recorded project, the port's relative path says how far up.

    A registry row written before the project column existed has no root, and
    the folder the artifact sits in is not the project -- ``merged/`` is one
    level down. Treating the artifact's own folder as the root would look for
    every other input inside ``merged/``.
    """
    artifact = artifacts.Artifact(
        artifact_id="0" * 16, project="", kind=ports.MERGED_ARRAYS,
        role="merged", path="/data/plate1/merged", module="mask",
        run_id="", settings_hash="", spacr_version="", created_ns=0,
        created_utc="", fingerprint="", fingerprint_method="",
        size_bytes=0, n_files=2, status=artifacts.STATUS_COMPLETE)
    port = ports.Port(kind=ports.MERGED_ARRAYS, role="merged", path="merged")

    assert chaining._artifact_root(artifact, port) == "/data/plate1"

    # A recorded project always wins over the climb.
    recorded = artifacts.Artifact(
        **{**artifact.to_dict(), "project": "/elsewhere/plate9"})
    assert chaining._artifact_root(recorded, port) == "/elsewhere/plate9"


def test_a_registry_that_will_not_open_offers_nothing(tmp_path, monkeypatch):
    """An unopenable registry answers None instead of raising.

    Chaining runs while a settings screen is being drawn. A registry locked by
    another process, or on a share that has gone away, must degrade to "no
    chained default" -- the user can still type the path.
    """
    root = _plate(tmp_path / "plate1")
    _run_mask(root)
    assert chaining._registry_for(root, None) is not None

    def _refuse(*args, **kwargs):
        raise OSError("the registry is on a share that went away")

    monkeypatch.setattr(chaining._artifacts, "open_registry", _refuse)

    assert chaining._registry_for(root, None) is None


# --------------------------------------------------------------------------- #
#  What a chained input says about itself
# --------------------------------------------------------------------------- #

def test_a_chained_input_names_its_kind_its_producer_and_its_path(tmp_path):
    """The one-line description carries all three, and staleness is separate.

    The line goes beside a pre-filled field. Without the producer, a user
    cannot tell whether the folder came from the Mask run they just did or
    from one last month.
    """
    root = _plate(tmp_path / "plate1")
    _run_mask(root)

    inputs = chaining.chained_inputs("measure", {"src": root}, root=root)

    assert inputs
    merged = next(i for i in inputs if i.kind == ports.MERGED_ARRAYS)
    assert merged.stale is False
    assert merged.path == merged.artifact.path
    line = merged.describe()
    assert ports.MERGED_ARRAYS in line
    assert "mask" in line
    assert merged.artifact.path in line


def test_a_pin_that_agrees_with_the_upstream_says_only_what_it_is(tmp_path):
    """A pin the upstream has not moved away from offers no alternative.

    The "now writes ... elsewhere" sentence is an offer to change the field.
    Printing it when nothing moved would ask the user to re-approve their own
    choice every time they opened the screen.
    """
    root = _plate(tmp_path / "plate1")
    _run_mask(root)
    store = chaining.PinStore(str(tmp_path / "pins.json"))
    chained = chaining.chained_inputs("measure", {"src": root}, root=root)
    merged = next(c for c in chained if c.kind == ports.MERGED_ARRAYS)
    store.pin("measure", merged.setting, merged.value)

    resolution = chaining.resolve_settings(
        "measure", {"src": root, merged.setting: merged.value},
        root=root, pins=store)

    held = resolution.held[merged.setting]
    assert held.differs is False
    assert held.describe() == f"{held.setting} is set to {held.value!r}"
    assert resolution.moved == ()


# --------------------------------------------------------------------------- #
#  The next step, and where a drop lands
# --------------------------------------------------------------------------- #

def test_a_successor_blocked_for_no_stated_reason_still_says_so(tmp_path):
    """A not-ok verdict with no listed error still reports a blockage.

    ``blocked`` is what the offer shows beside the successor's name. An empty
    string there is indistinguishable from "it can run", so the successor
    would be offered as ready and fail the moment it was pressed.
    """
    from spacr.ports import Readiness

    def _step(readiness):
        return chaining.NextStep(
            module="measure", source="mask", root=str(tmp_path), kinds=(),
            seed={}, readiness=readiness)

    ready = _step(Readiness(module="measure", root=str(tmp_path), ok=True))
    silent = _step(Readiness(module="measure", root=str(tmp_path), ok=False))

    assert ready.ok is True
    assert ready.blocked == ""
    assert silent.ok is False
    assert silent.blocked == "cannot run here"
    assert silent.fix == ""


def test_a_drop_of_nothing_is_not_a_drop_on_the_filesystem_root():
    """``None`` resolves to the empty string, and ``/`` stops climbing.

    The climb walks up looking for a project. Without the stop it would spin
    at the filesystem root, and a None would become the process's working
    directory -- a real folder, silently offered as the user's project.
    """
    assert chaining.project_root_of(None) == ""
    assert chaining.project_root_of(os.sep) == os.sep


def test_a_folder_that_cannot_be_listed_contributes_no_database(
        tmp_path, monkeypatch):
    """An unreadable folder is skipped, not fatal, when hunting databases.

    A drop can land on a tree containing a folder this user cannot read. One
    such folder must not stop the databases in the folders beside it from
    being offered.
    """
    root = tmp_path / "plate1"
    (root / "measurements").mkdir(parents=True)
    db = root / "measurements" / "measurements.db"
    sqlite3.connect(db).close()

    real_listdir = os.listdir

    def _refuse(path, *args, **kwargs):
        if os.path.basename(str(path)) == "measurements":
            raise PermissionError(path)
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "listdir", _refuse)

    found = chaining.db_candidates(str(root))

    assert str(db) in found          # named by the port, not by the listing
    assert all(os.path.basename(p) != "extra.db" for p in found)


def test_the_result_tables_a_project_wrote_are_offered_together(tmp_path):
    """CSVs under results/ and settings/ are found one level deep, sorted.

    A screen that reads "a table or a CSV" should offer what the project has
    written rather than making the user go and find it. One level deep is the
    contract: a run's own output folder and what it wrote inside count, but
    the walk stops rather than crawling a whole tree of intermediates.
    """
    root = tmp_path / "plate1"
    (root / "results").mkdir(parents=True)
    (root / "results" / "regression.csv").write_text("gene,effect\na,1\n")
    (root / "results" / "run1").mkdir()
    (root / "results" / "run1" / "coefficients.csv").write_text("a,1\n")
    (root / "results" / "run1" / "deep").mkdir()
    (root / "results" / "run1" / "deep" / "buried.csv").write_text("a,1\n")
    (root / "results" / "run1" / "deep" / "deeper").mkdir()
    (root / "results" / "run1" / "deep" / "deeper" / "gone.csv").write_text("a,1\n")
    (root / "results" / "notes.txt").write_text("not a table")

    found = chaining.result_tables(str(root))

    assert str(root / "results" / "regression.csv") in found
    assert str(root / "results" / "run1" / "coefficients.csv") in found
    assert str(root / "results" / "run1" / "deep" / "deeper" / "gone.csv") \
        not in found
    assert all(not p.endswith(".txt") for p in found)
    assert list(found) == sorted(found)
    assert chaining.result_tables("") == ()
    assert chaining.result_tables(str(tmp_path / "nowhere")) == ()


# --------------------------------------------------------------------------- #
#  What a drop resolution says about itself
# --------------------------------------------------------------------------- #

def test_a_drop_target_names_what_was_found_and_where_it_came_from():
    """The description carries the kind, the location and the source.

    It is the sentence a screen shows after a drop fills a field. Without the
    source, a user cannot tell a value the registry supplied from one guessed
    off the folder layout, and only the first is a record of a real run.
    """
    target = chaining.DropTarget(
        module="measure", setting="src", role="merged",
        kind=ports.MERGED_ARRAYS, value="/data/plate1",
        location="/data/plate1/merged", source=chaining.FROM_LAYOUT)

    line = target.describe()
    assert ports.MERGED_ARRAYS in line
    assert "/data/plate1/merged" in line
    assert chaining.FROM_LAYOUT in line


def test_a_drop_reports_the_question_the_target_or_the_absence():
    """The reason line is the question, else the targets, else the absence.

    A drop that resolved to nothing has to say so in the screen's own words.
    An empty reason reads as success, and the field would stay as it was with
    no explanation.
    """
    target = chaining.DropTarget(
        module="measure", setting="src", role="merged",
        kind=ports.MERGED_ARRAYS, value="/data/plate1",
        location="/data/plate1/merged", source=chaining.FROM_LAYOUT)
    choice = chaining.DropChoice(
        setting="src", kind=ports.MEASUREMENTS_DB,
        question="which database did you mean?",
        options=("/data/plate1/a.db", "/data/plate1/b.db"))

    asked = chaining.DropResolution(module="measure", dropped="/data",
                                    root="/data/plate1", targets=(target,),
                                    choices=(choice,))
    filled = chaining.DropResolution(module="measure", dropped="/data",
                                     root="/data/plate1", targets=(target,))
    empty = chaining.DropResolution(module="measure", dropped="/data",
                                    root="/data/plate1")

    assert asked.ambiguous is True
    assert "which database did you mean?" in asked.reason
    assert "2 candidates" in asked.reason

    assert filled.reason == target.describe()
    assert filled.target_for(ports.MERGED_ARRAYS) is target
    assert filled.target_for(ports.MEASUREMENTS_DB) is None

    assert bool(empty) is False
    assert empty.reason == "nothing this module reads was found in /data/plate1"


def test_a_screen_that_reads_nothing_says_the_folder_is_not_a_project(
        tmp_path):
    """A drop for a module with no declared inputs names the layout it wants.

    "Nothing happened" is the worst possible answer to a drag and drop. The
    message has to name the folders that make a folder a spaCR project, so
    the user can see they dropped the parent, or the wrong plate.
    """
    problems = chaining._problems_for("measure", (), str(tmp_path), None)

    assert len(problems) == 1
    assert str(tmp_path) in problems[0].message
    assert "not a spaCR project folder" in problems[0].message
    assert problems[0].is_error
    for name in chaining.layout_directories()[:4]:
        assert name in problems[0].fix


def test_sub_projects_skips_layout_folders_and_unreadable_ones(
        tmp_path, monkeypatch):
    """A layout child is never a sub-project, and a file is never one either.

    Without the exclusion, a plate whose raw images had been cleaned away
    answers "did you mean masks/?", because a folder of label TIFFs satisfies
    a raw-image port when you only look at extensions.
    """
    module_ports = tuple(ports.module_ports("measure").consumes)

    assert chaining._sub_projects("", module_ports) == ()
    assert chaining._sub_projects(str(tmp_path / "gone"), module_ports) == ()

    holder = tmp_path / "drop"
    holder.mkdir()
    _plate(holder / "plate1")
    (holder / "merged").mkdir()          # a layout name: never a sub-project
    (holder / "readme.txt").write_text("not a folder")

    found = chaining._sub_projects(str(holder), module_ports)
    assert found == (str(holder / "plate1"),)

    def _refuse(path, *args, **kwargs):
        raise PermissionError(path)

    monkeypatch.setattr(os, "listdir", _refuse)
    assert chaining._sub_projects(str(holder), module_ports) == ()


def test_a_drop_into_a_list_shaped_setting_arrives_as_a_list(tmp_path):
    """A setting whose current value is a list is filled with a one-item list.

    ``src`` is a list for the modules that accept several plates. Dropping a
    bare string into one would replace the list with a path, and the next
    read would iterate its characters.
    """
    root = _plate(tmp_path / "plate1")

    resolution = chaining.resolve_drop("measure", root,
                                       settings={"src": ["/old/plate0"]})

    assert resolution.targets
    for target in resolution.targets:
        if target.setting == "src":
            assert isinstance(target.value, list)
            assert len(target.value) == 1
            break
    else:
        pytest.fail("the drop filled no src target")
