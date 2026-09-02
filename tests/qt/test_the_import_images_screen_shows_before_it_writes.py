"""
Import Images — the screen in front of :mod:`spacr.image_import`.

Everything runs offscreen against the SAME ten synthetic acquisitions the
engine is measured on (``tests/import_corpus.py``), because the property this
screen exists for is that **the parse is visible before anything is written**.
spaCR's import path before this asked the user to name a filename convention
and told them nothing until masks came out wrong; of the ten layouts, two
parsed and eight recovered nothing.

The properties pinned here:

* it **builds offscreen** with nothing chosen and Import disabled;
* a scan **writes nothing**, and fills the table with one row per image,
  each filename BESIDE the fields it was parsed into;
* the table shows **the axes this folder has** and no others;
* what could not be named is **asked as a question**, Import stays disabled
  while one is unanswered, and answering it resolves the parse **in memory**;
* a **half-typed answer is not an answer**;
* the tiled tree **says what it would lose before it loses it**;
* a destination **inside** the source is refused — the ``consolidate`` bug;
* the import writes a plate **spaCR's own parser reads**, by links;
* a saved plan **round-trips**, and dropping one is the same gesture;
* the screen and :meth:`spacr.image_import.ImportPlan.table` **cannot
  disagree** about the proposal, because they are the same rows;
* the job runs **off the GUI thread** through the same ``finished`` →
  bound-method relay as Import Project;
* **no modal dialogs anywhere** — the autouse fixture fires if one opens.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from import_corpus import build_all  # noqa: E402

from spacr.qt.screens.image_import import (  # noqa: E402
    ANSWER_COLUMNS,
    AnswerModel,
    ImageImportScreen,
    ProposalModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    A QMessageBox in a headless run waits for a press that never comes, and
    the suite hangs rather than fails. Every error path on this screen goes
    to the inline status label instead, and this makes the alternative
    impossible to reintroduce quietly.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """The ten trees, written once for the whole file."""
    return {tree.name: tree
            for tree in build_all(tmp_path_factory.mktemp("screen_corpus"))}


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — jobs run inline so assertions are exact."""
    widget = ImageImportScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def spacr_regex():
    """spaCR's OWN parser, the one the core modules use."""
    with contextlib.redirect_stdout(io.StringIO()):      # the helper prints
        from spacr.utils import _get_regex
        return re.compile(_get_regex("cellvoyager", "tif"))


def _scanned(screen, tree, destination=None):
    """Point the screen at a corpus tree and scan it."""
    screen.set_root(str(tree.root))
    if destination is not None:
        screen.set_destination(str(destination))
    assert screen.scan() is True
    return screen


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_it_builds_offscreen_with_nothing_chosen(screen):
    assert screen.plan() is None
    assert screen.proposal_row_count() == 0
    assert screen.question_count() == 0
    assert screen.can_import() is False


def test_scanning_nothing_reports_inline_rather_than_raising(screen):
    assert screen.scan() is False
    assert "folder" in screen.status_text().lower()
    assert screen.last_error


def test_scanning_a_path_that_is_not_a_folder_says_so(screen, tmp_path):
    missing = tmp_path / "not_here"
    screen.set_root(str(missing))
    assert screen.scan() is False
    assert str(missing) in screen.status_text()


# ---------------------------------------------------------------------------
# The proposal: the reason this screen exists
# ---------------------------------------------------------------------------

def test_a_scan_shows_one_row_per_image_and_writes_nothing(screen, corpus,
                                                           tmp_path):
    """The single most important requirement: the parse is VISIBLE first."""
    tree = corpus["imagexpress"]
    destination = tmp_path / "written_by_the_scan"
    _scanned(screen, tree, destination)

    assert screen.proposal_row_count() == len(tree.files)
    assert not destination.exists(), "the scan wrote to the destination"
    assert screen.can_import() is True


def test_each_filename_sits_beside_the_fields_it_was_parsed_into(screen,
                                                                 corpus):
    """A field read as a channel is only visible next to the name it came
    from, which is why the table carries the filename rather than a row
    number."""
    tree = corpus["imagexpress"]
    _scanned(screen, tree)

    seen = {screen.proposal_value(row, "file"): row
            for row in range(screen.proposal_row_count())}
    for rel, truth in tree.truth.items():
        row = seen[rel]
        assert screen.proposal_value(row, "well") == truth["well"]
        assert screen.proposal_value(row, "field") == str(truth["field"])
        assert screen.proposal_value(row, "channel") == str(truth["channel"])


def test_the_columns_are_the_axes_this_folder_has(screen, corpus):
    """A column of empty cells is a different claim from an absent column."""
    _scanned(screen, corpus["tiled"])
    assert "tile" in screen.proposal_columns()

    _scanned(screen, corpus["imagexpress"])
    assert "tile" not in screen.proposal_columns()
    assert screen.proposal_columns()[0] == "file"


def test_the_screen_and_the_text_table_cannot_disagree(screen, corpus):
    """Both are the same proposal — one padded, one in a widget. They read
    their columns and rows from the plan rather than each deciding."""
    _scanned(screen, corpus["cellvoyager"])
    plan = screen.plan()

    assert screen.proposal_columns() == plan.columns()
    rows = plan.rows()
    assert screen.proposal_row_count() == len(rows)
    for index, expected in enumerate(rows):
        assert screen.plan() is plan
        assert [screen.proposal_value(index, column)
                for column in plan.columns()] == expected


def test_the_report_names_what_was_not_interpreted(screen, corpus, tmp_path):
    """Silence about unparsed files is how a plate imports with a third of
    its fields missing."""
    root = tmp_path / "mixed"
    root.mkdir()
    for rel in corpus["imagexpress"].files:
        (root / rel).write_bytes((corpus["imagexpress"].root / rel).read_bytes())
    odd = root / "scan_settings_readme.tif"
    odd.write_bytes(b"")

    screen.set_root(str(root))
    assert screen.scan() is True
    assert odd.name in screen.report_text()


# ---------------------------------------------------------------------------
# The questions, and what they gate
# ---------------------------------------------------------------------------

def test_a_dye_folder_is_asked_about_rather_than_guessed(screen, corpus):
    """``DAPI``/``GFP`` is a real axis and nothing maps a dye to an index.
    Guessing alphabetically would be a plausible answer that is wrong half
    the time, and invisible either way."""
    _scanned(screen, corpus["per_channel_folder"])

    values = {value for _token, value, _answer in screen.questions()}
    assert values == {"DAPI", "GFP"}
    assert screen.can_import() is False
    assert screen.problems()


def test_answering_resolves_the_parse_in_memory_and_enables_import(screen,
                                                                   corpus):
    tree = corpus["per_channel_folder"]
    _scanned(screen, tree)

    for row, (_token, value, _answer) in enumerate(screen.questions()):
        assert screen.answer_question(row, "1" if value == "DAPI" else "2")

    assert screen.can_import() is True
    assert screen.problems() == []
    seen = {screen.proposal_value(row, "file"): row
            for row in range(screen.proposal_row_count())}
    for rel, truth in tree.truth.items():
        assert screen.proposal_value(seen[rel], "channel") == str(truth["channel"])


def test_a_half_typed_answer_is_not_an_answer(screen, corpus):
    """``"1"`` arrives one keystroke after ``""``, and a blank read as zero
    would write a channel nobody asked for."""
    _scanned(screen, corpus["per_channel_folder"])

    assert screen.answer_question(0, "  ") is False      # unchanged, still blank
    assert screen.answer_question(0, "DAPI") is True     # typed, not a number
    assert screen.can_import() is False
    assert screen.plan().mapping == {}


def test_an_unreadable_answer_can_be_corrected(screen, corpus):
    _scanned(screen, corpus["per_channel_folder"])
    screen.answer_question(0, "x")
    screen.answer_question(0, "1")
    screen.answer_question(1, "2")
    assert screen.can_import() is True


# ---------------------------------------------------------------------------
# Tiles: the thing spaCR's filename cannot say
# ---------------------------------------------------------------------------

def test_the_report_says_what_each_tile_policy_will_do(screen, corpus):
    """Each of the three choices loses something different, and only one of
    them is recoverable by pressing the button again. So the report says
    which, before the press rather than after it."""
    _scanned(screen, corpus["tiled"])
    assert screen.tile_policy() == "stitch", "stitching is the default"
    assert "assembled into 8 field" in screen.report_text()

    assert screen.set_tile_policy("fields") is True
    assert "field of its own" in screen.report_text()

    assert screen.set_tile_policy("skip") is True
    assert "SKIPPED" in screen.report_text()


def test_a_tile_policy_that_is_not_offered_is_refused_inline(screen):
    assert screen.set_tile_policy("mosaic") is False
    assert screen.last_error
    assert screen.tile_policy() == "stitch"


def test_the_two_state_spelling_still_means_what_it_used_to(screen):
    """`tiles_as_fields` predates the stitcher and callers still use it."""
    screen.set_tiles_as_fields(True)
    assert screen.tile_policy() == "fields"
    assert screen.tiles_as_fields() is True
    screen.set_tiles_as_fields(False)
    assert screen.tile_policy() == "stitch"
    assert screen.stitch_tiles() is True


def test_a_tiled_field_is_stitched_into_one_image_by_default(screen, corpus,
                                                             tmp_path):
    """The maintainer's decision, 2026-09-02: stitch by default. Eight
    fields out of thirty-two tiles, and no image lost."""
    destination = tmp_path / "stitched_project"
    _scanned(screen, corpus["tiled"], destination)

    assert screen.run_import() is True
    result = screen.result()
    assert result.written == 8
    assert result.stitched == 8
    assert not result.skipped
    # The corpus tiles are blank, so nothing correlates and the screen says
    # so rather than presenting a butt-joined field as a measured one.
    assert "unverified" in screen.status_text()
    assert screen.last_error


def test_every_tile_survives_when_each_becomes_its_own_field(screen, corpus,
                                                             tmp_path):
    tree = corpus["tiled"]
    destination = tmp_path / "tiled_project"
    _scanned(screen, tree, destination)
    screen.set_tiles_as_fields(True)

    assert screen.run_import() is True
    result = screen.result()
    assert result.skipped == {}
    assert result.written == len(tree.files)


def test_tiles_are_skipped_with_a_reason_rather_than_overwritten(screen,
                                                                 corpus,
                                                                 tmp_path):
    """The answer before there was a stitcher, kept behind the third policy:
    the alternative is three of every four tiles disappearing silently,
    which is exactly the failure this module was written against."""
    tree = corpus["tiled"]
    _scanned(screen, tree, tmp_path / "lossy_project")
    screen.set_tile_policy("skip")

    assert screen.run_import() is True
    result = screen.result()
    assert result.skipped, "tiles were written over each other"
    assert all("overwrite" in reason for reason in result.skipped.values())
    assert screen.last_error and "NOT written" in screen.status_text()
    assert "an axis is missing" in screen.report_text()


# ---------------------------------------------------------------------------
# Writing the project
# ---------------------------------------------------------------------------

def test_the_import_writes_a_plate_spacrs_own_parser_reads(screen, corpus,
                                                           tmp_path,
                                                           spacr_regex):
    """The measure that says the import is finished rather than
    finished-shaped: the shipped parser reads what came out."""
    destination = tmp_path / "project"
    _scanned(screen, corpus["harmony"], destination)

    assert screen.run_import() is True
    written = sorted(p.name for p in destination.iterdir())
    assert written
    assert [n for n in written if spacr_regex.match(n)] == written
    assert screen.result().written == len(written)
    assert screen.last_error == ""


def test_the_import_links_rather_than_copying_the_plate(screen, corpus,
                                                        tmp_path):
    """``consolidate`` copies every image to rearrange its name, so a 300 GB
    plate costs 600 GB. Nothing about renaming requires duplicating bytes."""
    destination = tmp_path / "linked"
    _scanned(screen, corpus["cellvoyager"], destination)

    assert screen.run_import() is True
    result = screen.result()
    assert result.linked == result.written
    assert all(p.is_symlink() for p in destination.iterdir())
    assert result.bytes_saved > 0
    assert "not duplicated" in screen.report_text()


def test_copying_is_available_for_a_project_that_must_stand_alone(screen,
                                                                  corpus,
                                                                  tmp_path):
    destination = tmp_path / "copied"
    _scanned(screen, corpus["cellvoyager"], destination)
    screen.set_link(False)

    assert screen.run_import() is True
    assert not any(p.is_symlink() for p in destination.iterdir())
    assert screen.result().linked == 0


def test_the_plate_name_reaches_every_filename(screen, corpus, tmp_path):
    destination = tmp_path / "named"
    _scanned(screen, corpus["per_well_folder"], destination)
    screen.set_plate_name("screen7")

    assert screen.run_import() is True
    assert all(p.name.startswith("screen7_") for p in destination.iterdir())


def test_importing_before_scanning_is_refused_inline(screen):
    assert screen.run_import() is False
    assert "Scan" in screen.status_text()


def test_a_plan_with_questions_left_refuses_to_write(screen, corpus,
                                                     tmp_path):
    """The import is the irreversible half, and every stated problem is a
    way for the result to be quietly wrong."""
    destination = tmp_path / "refused"
    _scanned(screen, corpus["per_channel_folder"], destination)

    assert screen.run_import() is False
    assert not destination.exists()
    assert screen.last_error


def test_a_destination_inside_the_source_is_refused(screen, corpus):
    """``consolidate`` created its output inside the folder it was walking,
    so a second run consolidated the first run's output and doubled the
    plate. A destination under the source is that bug."""
    tree = corpus["cellvoyager"]
    _scanned(screen, tree, tree.root / "consolidated")

    assert screen.run_import() is False
    assert "inside" in screen.status_text()
    assert not (tree.root / "consolidated").exists()


def test_a_destination_equal_to_the_source_is_refused(screen, corpus):
    tree = corpus["cellvoyager"]
    _scanned(screen, tree, tree.root)
    assert screen.run_import() is False


# ---------------------------------------------------------------------------
# The plan file
# ---------------------------------------------------------------------------

def test_a_saved_plan_reloads_with_its_answers(screen, corpus, tmp_path,
                                               qtbot):
    """A lab images the same way every week; re-answering the same question
    every time is how a tool stops being used."""
    tree = corpus["per_channel_folder"]
    _scanned(screen, tree)
    for row, (_token, value, _answer) in enumerate(screen.questions()):
        screen.answer_question(row, "1" if value == "DAPI" else "2")

    saved = tmp_path / "plans" / "weekly.json"
    assert screen.save_plan(str(saved)) is True
    assert saved.exists()
    assert json.loads(saved.read_text())["mapping"] == {"0": {"DAPI": 1,
                                                              "GFP": 2}}

    fresh = ImageImportScreen(threaded=False)
    qtbot.addWidget(fresh)          # or Python frees it at a moment of its
    assert fresh.load_plan(str(saved)) is True
    assert fresh.root_path() == str(tree.root)
    assert fresh.can_import() is True
    assert [row[2] for row in fresh.questions()] == ["1", "2"]


def test_saving_before_scanning_is_refused_inline(screen, tmp_path):
    assert screen.save_plan(str(tmp_path / "nothing.json")) is False
    assert screen.last_error


def test_loading_something_that_is_not_a_plan_reports_inline(screen,
                                                             tmp_path):
    bad = tmp_path / "notes.json"
    bad.write_text("{}", encoding="utf-8")
    assert screen.load_plan(str(bad)) is False
    assert screen.last_error
    assert screen.plan() is None


def test_dropping_a_folder_points_the_screen_at_it(screen, corpus):
    """The gesture this module is for."""
    from spacr.qt.dnd_handlers import get_handler

    handler = get_handler("import_images")
    tree = corpus["cellvoyager"]
    handler.apply(Path(tree.root), screen)
    assert screen.root_path() == str(tree.root)

    one_file = Path(tree.root) / sorted(tree.files)[0]
    handler.apply(one_file, screen)
    assert screen.root_path() == str(one_file.parent)


def test_dropping_a_saved_plan_loads_it(screen, corpus, tmp_path, qtbot):
    from spacr.qt.dnd_handlers import get_handler

    _scanned(screen, corpus["cellvoyager"])
    saved = tmp_path / "dropped.json"
    screen.save_plan(str(saved))

    fresh = ImageImportScreen(threaded=False)
    qtbot.addWidget(fresh)          # ... own choosing, and the process dies
    get_handler("import_images").apply(saved, fresh)
    assert fresh.plan() is not None


# ---------------------------------------------------------------------------
# Editing the folder, and the controls
# ---------------------------------------------------------------------------

def test_changing_the_folder_clears_a_plan_that_described_the_old_one(screen,
                                                                      corpus):
    _scanned(screen, corpus["cellvoyager"])
    assert screen.can_import() is True

    screen.set_root(str(corpus["cq1"].root))
    assert screen.plan() is None
    assert screen.proposal_row_count() == 0
    assert screen.can_import() is False
    assert "Scan again" in screen.status_text()


def test_choosing_a_folder_proposes_a_destination_beside_it(screen, corpus):
    tree = corpus["cq1"]
    screen.set_root(str(tree.root))
    assert screen.destination_path() == os.path.normpath(str(tree.root)) + "_spacr"
    assert not screen._is_inside(screen.destination_path(), str(tree.root))


def test_a_destination_already_typed_is_not_overwritten(screen, corpus,
                                                        tmp_path):
    screen.set_destination(str(tmp_path / "mine"))
    screen.set_root(str(corpus["cq1"].root))
    assert screen.destination_path() == str(tmp_path / "mine")


def test_the_scan_options_reach_the_engine(screen, corpus):
    """``read_inside`` off is the fast first look at a large archive: the
    channel that lives INSIDE a file is then not found, which is the whole
    difference the option makes."""
    _scanned(screen, corpus["flat_ome"])
    assert screen.plan().inside

    screen.set_read_inside(False)
    screen.set_sample(50)
    assert screen.scan() is True
    assert screen.plan().inside == {}
    assert screen.plan().layout.sampled <= 50


# ---------------------------------------------------------------------------
# The models, directly
# ---------------------------------------------------------------------------

def test_the_proposal_model_is_read_only(qtbot, corpus):
    from PySide6.QtCore import Qt

    from spacr.image_import import plan_import

    model = ProposalModel()
    model.set_plan(plan_import(corpus["cq1"].root, read_files=False))
    index = model.index(0, 1)
    assert not (model.flags(index) & Qt.ItemIsEditable)
    assert model.value_at(0, "nonexistent") == ""
    assert model.row(9999) == []

    model.set_plan(None)
    assert model.rowCount() == 0
    assert model.headers() == []


def test_the_answer_model_only_edits_the_answer(qtbot):
    from PySide6.QtCore import Qt

    model = AnswerModel()
    assert model.rowCount() == 0
    assert [name for name, _editable in ANSWER_COLUMNS][:2] == ["Token",
                                                                "Value"]
    model._rows = [["0", "DAPI", ""]]
    assert model.setData(model.index(0, 0), "9", Qt.EditRole) is False
    assert model.setData(model.index(0, 2), "1", Qt.EditRole) is True
    assert model.setData(model.index(0, 2), "1", Qt.EditRole) is False
    assert model.mapping() == {0: {"DAPI": 1}}


# ---------------------------------------------------------------------------
# Off the GUI thread
# ---------------------------------------------------------------------------

def test_the_scan_runs_off_the_gui_thread(qtbot, qt_theme_applied, corpus):
    """The real path: a scan reads every file's header, and doing that on
    the GUI thread freezes the window."""
    widget = ImageImportScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.set_root(str(corpus["cellvoyager"].root))
    try:
        with qtbot.waitSignal(widget.job_finished, timeout=30000) as caught:
            assert widget.scan() is True
        assert caught.args == [True]
        assert widget.plan() is not None
        assert widget.proposal_row_count() == len(corpus["cellvoyager"].files)
        qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=30000)
    finally:
        for thread, _worker in list(widget._jobs):
            thread.quit()
            thread.wait(5000)
