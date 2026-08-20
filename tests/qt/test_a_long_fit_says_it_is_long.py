"""A long fit says it will be long, and says where it has got to (140).

Reported 2026-08-18, twice, WHILE THE FIT WAS RUNNING CORRECTLY: "im running
the mixed model now and it is taking much longer than before is that normal?"
... "it is still going, cpu at 100 percent". Nothing was wrong. `mixed` is an
iterative REML optimisation and it is single-threaded, so one core at 100% is
what a healthy fit looks like -- and an hour of that in silence is
indistinguishable from a hang.

So these tests are about what the run SAYS: the measured cost before the
choice is made, the design before the fit blocks, and a heartbeat while it
does. Nothing here asserts that a widget exists.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from spacr.qt.job_runner import JobRunner
from spacr.qt.screens.app_screen import AppScreen, _elapsed_words
from spacr.qt.screens.settings_model import (
    MIXED_COST_ANCHORS,
    SLOW_MODELS,
    mixed_cost_note,
    regression_design_scan,
    regression_model_explainer,
)


def _flat(text: str) -> str:
    """The box's text with its wrapping collapsed."""
    return " ".join(str(text).split())


def _count_csv(path, *, genes=4, guides=3, wells=6):
    """A count table shaped like the sequencing side writes one.

    ``<org>_<gene>_<guide>`` names, one row per (well, guide), and the plate /
    row / column triple a well is composed from -- which is what
    `spacr.ml.process_reads` reads.
    """
    rows = []
    for well in range(wells):
        for gene in range(genes):
            for guide in range(guides):
                rows.append({
                    "grna": f"TGGT1_g{gene:03d}_{guide}",
                    "count": 10 + guide,
                    "plateID": "plate1",
                    "rowID": f"r{well // 3 + 1}",
                    "columnID": f"c{well % 3 + 1}",
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# A. the box states the measured cost
# ---------------------------------------------------------------------------

def test_the_mixed_box_states_the_measurement_and_the_way_out():
    box = _flat(regression_model_explainer("mixed"))
    assert "WHAT IT COSTS" in box
    # THE DIGITS, because "this may be slow" is what the console said by
    # saying nothing.
    # THE NUMBERS, NOT THE SENTENCE AROUND THEM. Instruction 143 B halved
    # this paragraph's words and said plainly "do not shorten by deleting the
    # numbers" -- so the numbers are what this holds, and the phrasing is free
    # to get shorter again. Asserting the prose made a legitimate trim look
    # like a regression.
    for genes, wells, ols, mixed in MIXED_COST_ANCHORS:
        assert str(genes) in box and str(wells) in box
        assert f"{ols:g}s" in box or f"{ols:g} s" in box
        assert f"{mixed:g}s" in box or f"{mixed:g} s" in box
        assert f"{round(mixed / ols):g}x" in box
    assert "Single-threaded" in box or "single-threaded" in box
    # And what to choose when the answer is wanted now.
    assert "ols at level='both'" in box


def test_the_ratio_in_the_note_is_computed_from_the_measurement():
    """The prose and the anchors cannot drift, because there is one copy.

    A measurement written out twice is two numbers, and the second one to be
    edited is the one nobody believes afterwards.
    """
    note = mixed_cost_note()
    for _genes, _wells, ols, mixed in MIXED_COST_ANCHORS:
        assert f"({round(mixed / ols):g}x)" in note


def test_only_the_mixed_box_carries_a_cost_section():
    """`ols` is the fast one; a cost warning on it would be noise."""
    assert "WHAT IT COSTS" not in regression_model_explainer("ols")
    assert "WHAT IT COSTS" not in regression_model_explainer("rra")


def test_mixed_is_named_as_a_slow_model():
    assert "mixed" in SLOW_MODELS


# ---------------------------------------------------------------------------
# B. the design, read off the files the run was given
# ---------------------------------------------------------------------------

def test_the_design_scan_counts_genes_guides_and_wells(tmp_path):
    path = _count_csv(tmp_path / "counts.csv", genes=4, guides=3, wells=6)
    design = regression_design_scan(
        {"paired_data": [{"score": "", "count": str(path)}]})
    assert design["genes"] == 4
    assert design["guides"] == 12
    assert design["wells"] == 6
    assert design["files"] == 1
    assert design["rows"] == 4 * 3 * 6
    assert design["note"] == ""


def test_the_scan_reads_the_legacy_count_data_list(tmp_path):
    """A settings CSV saved before `paired_data` is exactly what gets reopened."""
    path = _count_csv(tmp_path / "old.csv", genes=2, guides=2, wells=4)
    design = regression_design_scan({"count_data": [str(path)]})
    assert (design["genes"], design["guides"], design["wells"]) == (2, 4, 4)


def test_two_count_files_are_one_design(tmp_path):
    one = _count_csv(tmp_path / "a.csv", genes=3, guides=2, wells=2)
    two = _count_csv(tmp_path / "b.csv", genes=3, guides=2, wells=2)
    design = regression_design_scan(
        {"paired_data": [{"count": str(one)}, {"count": str(two)}]})
    # The same library on the same plate: two files, one design.
    assert design["files"] == 2
    assert design["genes"] == 3
    assert design["guides"] == 6


def test_a_missing_file_is_reported_and_does_not_raise(tmp_path):
    """It runs beside a fit that is already starting.

    A scan that threw would take the run's own message with it, so what it
    could not work out comes back as None with a reason.
    """
    design = regression_design_scan(
        {"count_data": [str(tmp_path / "nowhere.csv")]})
    assert design["genes"] is None
    assert design["guides"] is None
    assert "could not read" in design["note"]


def test_names_that_are_not_org_gene_guide_get_no_gene_count(tmp_path):
    """The split is POSITIONAL, and a gene total from a rule the run will
    not apply is a number that disagrees with the fit."""
    path = tmp_path / "odd.csv"
    pd.DataFrame([
        {"grna": "guideA", "count": 3, "plateID": "p1",
         "rowID": "r1", "columnID": "c1"},
        {"grna": "guideB", "count": 4, "plateID": "p1",
         "rowID": "r1", "columnID": "c2"},
    ]).to_csv(path, index=False)
    design = regression_design_scan({"count_data": [str(path)]})
    assert design["guides"] == 2
    assert design["genes"] is None
    assert "<org>_<gene>_<guide>" in design["note"]
    assert design["wells"] == 2


def test_wells_are_not_guessed_when_the_columns_are_absent(tmp_path):
    """A well count off by the number of plates is worse than none."""
    path = tmp_path / "nowell.csv"
    pd.DataFrame([{"grna": "TGGT1_g1_1", "count": 3}]).to_csv(path,
                                                              index=False)
    design = regression_design_scan({"count_data": [str(path)]})
    assert design["wells"] is None
    assert "wells were not counted" in design["note"]


def test_no_count_files_says_so():
    assert regression_design_scan({})["note"] == "no count files in the settings"


# ---------------------------------------------------------------------------
# C. the real screen: the sentence lands in the real console
# ---------------------------------------------------------------------------

def _console_text(screen) -> str:
    return screen._console._current_stdout.toPlainText()


def test_the_run_names_the_design_and_its_cost_before_it_blocks(
        qtbot, qt_theme_applied, tmp_path):
    """Driven on the real screen, through the real JobRunner and console.

    `threaded=False` runs the scan inline and emits the same signals in the
    same order, which is what that flag is for -- the alternative is a test
    that waits on a thread to find out whether a sentence was printed.
    """
    path = _count_csv(tmp_path / "counts.csv", genes=5, guides=4, wells=9)
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._jobs = JobRunner(screen, threaded=False, app_key="test")
    try:
        screen._announce_the_fit({
            "regression_type": "mixed", "level": "both",
            "paired_data": [{"score": "", "count": str(path)}],
        })
        text = _flat(_console_text(screen))
    finally:
        screen._stop_the_heartbeat()

    # WHAT IS RUNNING.
    assert "Model: mixed, level both" in text
    # WHAT IT COSTS -- the same sentence the box states, one source.
    assert _flat(mixed_cost_note()) in text
    # HOW BIG IT IS, and which number it is.
    assert "5 genes and 20 guides over 9 wells" in text
    assert "BEFORE the merge with the scores" in text


def test_a_fast_model_is_not_warned_about(qtbot, qt_theme_applied, tmp_path):
    path = _count_csv(tmp_path / "counts.csv", genes=2, guides=2, wells=2)
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._jobs = JobRunner(screen, threaded=False, app_key="test")
    try:
        screen._announce_the_fit({
            "regression_type": "ols", "level": "grna",
            "paired_data": [{"count": str(path)}]})
        text = _flat(_console_text(screen))
    finally:
        screen._stop_the_heartbeat()
    assert "Model: ols, level grna" in text
    assert "MEASURED 2026-08-18" not in text
    assert screen._slow_fit is False


def test_a_module_that_is_not_the_regression_says_nothing(
        qtbot, qt_theme_applied):
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    # Nothing has been written to this console yet, so it holds no stdout
    # block at all -- which is the state the assertion is about.
    assert screen._console._current_stdout is None
    screen._announce_the_fit({"regression_type": "mixed"})
    assert screen._console._current_stdout is None
    assert screen._heartbeat is None


def test_a_design_that_cannot_be_read_says_so_rather_than_nothing(
        qtbot, qt_theme_applied, tmp_path):
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._jobs = JobRunner(screen, threaded=False, app_key="test")
    try:
        screen._announce_the_fit({
            "regression_type": "mixed",
            "count_data": [str(tmp_path / "gone.csv")]})
        text = _flat(_console_text(screen))
    finally:
        screen._stop_the_heartbeat()
    assert "Could not size the design" in text
    assert "could not read" in text


# ---------------------------------------------------------------------------
# D. the heartbeat
# ---------------------------------------------------------------------------

def test_the_heartbeat_schedule_speaks_once_per_mark(qtbot, qt_theme_applied):
    """Driven directly, because the last entry is an hour away.

    An entry nobody checks is the one that is wrong, and a test that waited
    3,600 real seconds would not be written.
    """
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._heartbeat_said = 0.0

    assert screen._due_heartbeat(29) is False
    assert screen._due_heartbeat(30) is True
    assert screen._due_heartbeat(31) is False
    assert screen._due_heartbeat(59) is False
    assert screen._due_heartbeat(60) is True
    # A jump past several marks says ONE line, not four.
    assert screen._due_heartbeat(1300) is True
    assert screen._due_heartbeat(1301) is False


def test_the_heartbeat_keeps_going_past_the_last_scheduled_mark(
        qtbot, qt_theme_applied):
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    last = screen.HEARTBEAT_SCHEDULE[-1]
    screen._heartbeat_said = float(last)

    assert screen._due_heartbeat(last + 60) is False
    assert screen._due_heartbeat(last + screen.HEARTBEAT_INTERVAL) is True
    assert screen._due_heartbeat(last + screen.HEARTBEAT_INTERVAL + 1) is False
    assert screen._due_heartbeat(last + 2 * screen.HEARTBEAT_INTERVAL) is True


def test_the_heartbeat_says_where_it_has_got_to_and_why_one_core(
        qtbot, qt_theme_applied):
    import time

    class _Running:
        def isRunning(self):
            return True

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._thread = _Running()
    screen._slow_fit = True
    screen._heartbeat_said = 0.0
    screen._run_started_at = time.time() - 473
    try:
        screen._on_heartbeat()
        text = _flat(_console_text(screen))
    finally:
        screen._thread = None
    assert "Still fitting" in text
    assert "7 min 53 s" in text
    # NOT A SPINNER WITH NO CONTENT: it interprets the observation the user
    # could not ("cpu at 100 percent").
    assert "one core at 100%" in text
    assert "Stop is live" in text


def test_the_heartbeat_stops_when_the_run_does(qtbot, qt_theme_applied):
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._start_the_heartbeat()
    assert screen._heartbeat.isActive()
    screen._on_finished(True)
    assert not screen._heartbeat.isActive(), (
        "a heartbeat firing after the run says 'still fitting' underneath "
        "'Finished', and the last line is the one a user reads")


def test_the_heartbeat_retires_itself_when_the_thread_has_gone(
        qtbot, qt_theme_applied):
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._start_the_heartbeat()
    screen._thread = None
    screen._on_heartbeat()
    assert not screen._heartbeat.isActive()


@pytest.mark.parametrize("seconds,words", [
    (0, "0 s"), (5, "5 s"), (59, "59 s"), (60, "1 min 0 s"),
    (473, "7 min 53 s"), (3600, "1 h 0 min"), (4000, "1 h 6 min"),
])
def test_elapsed_reads_in_the_units_the_question_is_asked_in(seconds, words):
    assert _elapsed_words(seconds) == words
