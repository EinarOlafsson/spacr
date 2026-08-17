"""The Annotate screen's unreached corners, driven for real.

Written for instruction 60. Every case here exercises a branch that no
existing test reached, and each is named for what a user would see go wrong
rather than for the method it happens to run.

The screen is built against a real on-disk experiment -- a folder with PNG
crops and a ``measurements/measurements.db`` -- because most of these paths
are about what happens when that database says something unexpected, and a
mock database cannot say anything unexpected.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtWidgets import (QDialog, QFileDialog, QInputDialog, QMessageBox,
                               QWidget)

from spacr.qt import annotate_engine as engine
from spacr.qt.screens import annotate as annotate_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """A small experiment folder: 8 crops and a png_list that names them."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data" / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(8):
        arr = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
        p = src / "data" / "images" / f"cell_{i:02d}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    db = src / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    engine.ensure_annotation_column(str(db), "annotate")
    return src


@pytest.fixture
def screen(qtbot, qt_theme_applied, experiment: Path):
    """An open Annotate screen on a pinned 3x3 grid, page 0 decoded."""
    scr = annotate_mod.AnnotateScreen()
    qtbot.addWidget(scr)
    scr._settings.grid_rows = 3
    scr._settings.grid_cols = 3
    scr._settings.image_size = (24, 24)
    scr._compute_grid_dims = lambda: None
    scr._rebuild_grid()
    scr._open_source(str(experiment))
    qtbot.waitUntil(lambda: len(scr._page_paths) == 8, timeout=10000)
    yield scr
    if scr._worker is not None:
        scr._worker.stop(wait=True)


@pytest.fixture
def bare_screen(qtbot, qt_theme_applied):
    """An Annotate screen with no source open — the state it starts in."""
    scr = annotate_mod.AnnotateScreen()
    qtbot.addWidget(scr)
    yield scr
    if scr._worker is not None:
        scr._worker.stop(wait=True)


class _Recorder:
    """Records what a message box was asked to show, and answers for it."""

    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        return self.answer

    @property
    def shown(self) -> bool:
        return bool(self.calls)

    def body(self, index: int = 0) -> str:
        return " ".join(str(a) for a in self.calls[index])


# ---------------------------------------------------------------------------
# Theme resolution
# ---------------------------------------------------------------------------

def test_a_theme_that_cannot_be_read_leaves_the_grid_on_the_dark_palette(
        monkeypatch):
    """Preferences failing must not make the tiles paint for the wrong theme.

    The grid draws raw colours, so it asks preferences which theme is up. If
    that read raises, the answer has to be the default the app ships with --
    anything else paints a light-theme gray onto a black canvas.
    """
    import spacr.qt.preferences as prefs_mod
    monkeypatch.setattr(prefs_mod, "resolve_effective_theme",
                        lambda: (_ for _ in ()).throw(RuntimeError("no prefs")))
    assert annotate_mod.on_dark_theme() is True
    palette = annotate_mod.tile_palette()
    assert palette["border"]


# ---------------------------------------------------------------------------
# Keyboard token table
# ---------------------------------------------------------------------------

def test_key_names_qt_does_not_carry_are_skipped_rather_than_crashing(
        monkeypatch):
    """The token table is built from names PySide6 may or may not expose.

    PySide6 has moved these between ``Qt.Key_Left`` and ``Qt.Key.Key_Left``
    across releases. A name found in neither place is dropped; a build where
    every name moved must still produce a usable table rather than an import
    error.
    """
    class _KeyEnum:
        Key_Left = 0x01000012

    class _FakeQt:
        Key = _KeyEnum

    monkeypatch.setattr(annotate_mod, "Qt", _FakeQt)
    table = annotate_mod._qt_code_tokens()
    # Found on the nested enum...
    assert table == {0x01000012: "left"}
    # ...and every other name, present in neither place, was skipped.


# ---------------------------------------------------------------------------
# Page-load worker
# ---------------------------------------------------------------------------

def test_a_page_whose_decode_blows_up_delivers_an_empty_page_not_a_crash():
    """One unreadable crop must not take the whole page worker down."""
    def _boom(row):
        raise OSError("truncated PNG")

    worker = annotate_mod._PageLoadWorker(7, [("a.png", None)], _boom)
    seen = []
    worker.done.connect(lambda gen, loaded: seen.append((gen, loaded)))
    worker.run()
    assert seen == [(7, [])]


def test_a_page_interrupted_mid_decode_never_reports_a_half_page():
    """Closing the screen while a page decodes must deliver nothing.

    A partial page emitted after the screen asked to stop would paint crops
    into slots that now mean different images.
    """
    class _StoppingWorker(annotate_mod._PageLoadWorker):
        def isInterruptionRequested(self):
            return True

    worker = _StoppingWorker(1, [("a.png", None), ("b.png", None)],
                             lambda row: (None, None))
    seen = []
    worker.done.connect(lambda gen, loaded: seen.append(gen))
    worker.run()
    assert seen == []


def test_a_page_that_finishes_hands_back_every_decoded_crop():
    worker = annotate_mod._PageLoadWorker(
        3, [("a.png", 1), ("b.png", 2)], lambda row: (row[0].upper(), row[1]))
    seen = []
    worker.done.connect(lambda gen, loaded: seen.append((gen, loaded)))
    worker.run()
    assert seen == [(3, [("A.PNG", 1), ("B.PNG", 2)])]


# ---------------------------------------------------------------------------
# Retrain worker
# ---------------------------------------------------------------------------

def test_a_retrain_that_fails_says_why_instead_of_going_quiet(monkeypatch):
    """The failure has to name the exception; "it didn't work" is unusable."""
    from spacr import active_learning as al
    monkeypatch.setattr(
        al, "retrain_round",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("only one class")))
    worker = annotate_mod._RetrainWorker("db", "annotate", {})
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert failures == ["ValueError: only one class"]


def test_a_retrain_that_finished_after_the_screen_closed_reports_nothing(
        monkeypatch):
    from spacr import active_learning as al
    monkeypatch.setattr(al, "retrain_round", lambda *a, **k: "round-result")

    class _Interrupted(annotate_mod._RetrainWorker):
        def isInterruptionRequested(self):
            return True

    worker = _Interrupted("db", "annotate", {})
    results = []
    worker.done.connect(results.append)
    worker.run()
    assert results == []


def test_a_retrain_that_finished_hands_its_round_back(monkeypatch):
    from spacr import active_learning as al
    seen = {}

    def _retrain(db_path, column, **options):
        seen.update({"db": db_path, "column": column, **options})
        return "round-result"

    monkeypatch.setattr(al, "retrain_round", _retrain)
    worker = annotate_mod._RetrainWorker("db", "annotate", {"round_index": 4})
    results = []
    worker.done.connect(results.append)
    worker.run()
    assert results == ["round-result"]
    assert seen["round_index"] == 4


# ---------------------------------------------------------------------------
# Text report dialog
# ---------------------------------------------------------------------------

def test_a_wide_report_is_shown_unwrapped_and_selectable(qtbot,
                                                          qt_theme_applied):
    """Coverage tables are aligned text; reflowing them destroys the columns."""
    from PySide6.QtWidgets import QPlainTextEdit

    body = "class   n     frac\n1       120   0.51\n2       116   0.49"
    dlg = annotate_mod._TextReportDialog("Annotation coverage", body)
    qtbot.addWidget(dlg)
    assert dlg.windowTitle() == "Annotation coverage"
    assert dlg._view.toPlainText() == body
    assert dlg._view.isReadOnly()
    assert dlg._view.lineWrapMode() == QPlainTextEdit.NoWrap
    # The report is data, not UI text: translating a coverage table would
    # rewrite the numbers' headings out from under whoever pasted it.
    assert dlg._view.property("i18nSkipText") is True


# ---------------------------------------------------------------------------
# Re-anchoring moved crop paths
# ---------------------------------------------------------------------------

def test_a_moved_dataset_finds_its_crops_again_from_the_data_segment(
        experiment: Path):
    """The stored png paths are absolute and stale once a plate is moved."""
    db = str(experiment / "measurements" / "measurements.db")
    stale = "/somewhere/that/never/existed/data/images/cell_00.png"
    found = annotate_mod._reanchor_png_path(stale, db)
    assert found == str(experiment / "data" / "images" / "cell_00.png")


def test_a_relative_crop_path_is_rebuilt_under_the_open_database(
        experiment: Path):
    """`measure` run with a relative src stores 'data/...' with no root."""
    db = str(experiment / "measurements" / "measurements.db")
    found = annotate_mod._reanchor_png_path("data/images/cell_01.png", db)
    assert found == str(experiment / "data" / "images" / "cell_01.png")


def test_a_crop_path_with_no_open_database_is_left_exactly_as_stored():
    """With no database there is no root to re-anchor against."""
    assert annotate_mod._reanchor_png_path("/gone/data/x.png", "") == \
        "/gone/data/x.png"


def test_a_crop_that_is_missing_under_the_new_root_too_is_left_alone(
        experiment: Path):
    db = str(experiment / "measurements" / "measurements.db")
    stale = "/old/data/images/not_here_either.png"
    assert annotate_mod._reanchor_png_path(stale, db) == stale
    assert annotate_mod._reanchor_png_path("data/nope.png", db) == "data/nope.png"


# ---------------------------------------------------------------------------
# Thumbnail decoding (runs on the page worker)
# ---------------------------------------------------------------------------

def _settings_for(experiment: Path) -> engine.AnnotateSettings:
    s = engine.AnnotateSettings()
    s.src = str(experiment)
    s.db_path = str(experiment / "measurements" / "measurements.db")
    s.image_size = (16, 16)
    return s


def test_a_crop_the_grid_cannot_find_draws_a_placeholder_not_an_exception(
        experiment: Path):
    """A missing file is the ordinary case for a moved dataset."""
    s = _settings_for(experiment)
    img, annotation = annotate_mod._load_thumb_image_worker(
        {"png_path": "/nowhere/x.png", "annotation": 3}, None, s)
    assert img.size == (16, 16)
    assert annotation == 3


def test_a_crop_that_will_not_decode_still_keeps_its_annotation(
        experiment: Path, tmp_path: Path, monkeypatch):
    """The label the row carries is not the image's to lose."""
    s = _settings_for(experiment)
    monkeypatch.setattr(
        annotate_mod, "load_crop_image",
        lambda *a, **k: (_ for _ in ()).throw(OSError("bad header")))
    row = (str(experiment / "data" / "images" / "cell_00.png"), 2)
    img, annotation = annotate_mod._load_thumb_image_worker(row, None, s)
    assert img.size == (16, 16)
    assert annotation == 2


def test_a_merged_source_that_cannot_cut_a_crop_draws_a_placeholder(
        experiment: Path):
    """Projects with no PNG folder cut crops out of merged/*.npy."""
    class _MergedSource:
        kind = "merged"

        def get(self, row):
            raise KeyError("no such object")

    s = _settings_for(experiment)
    img, annotation = annotate_mod._load_thumb_image_worker(
        {"png_path": "x.png", "annotation": None}, _MergedSource(), s)
    assert img.size == (16, 16)
    assert annotation is None


def test_a_merged_source_crop_is_decoded_from_the_array(experiment: Path):
    class _MergedSource:
        kind = "merged"

        def get(self, row):
            return np.full((8, 8, 3), 128, dtype=np.uint8)

    s = _settings_for(experiment)
    img, _ = annotate_mod._load_thumb_image_worker(
        {"png_path": "x.png"}, _MergedSource(), s)
    assert img.size == (16, 16)


def test_an_outline_that_fails_still_shows_the_crop_underneath(
        experiment: Path, monkeypatch):
    """Outlining is decoration; losing it must not lose the image."""
    s = _settings_for(experiment)
    s.outline = ["r"]
    monkeypatch.setattr(
        annotate_mod, "outline_image",
        lambda **k: (_ for _ in ()).throw(RuntimeError("no cellpose")))
    row = (str(experiment / "data" / "images" / "cell_00.png"), 1)
    img, annotation = annotate_mod._load_thumb_image_worker(row, None, s)
    assert img.size == (16, 16)
    assert annotation == 1


def test_an_outline_is_computed_from_the_full_image_not_the_filtered_one(
        experiment: Path, monkeypatch):
    """Hiding a channel for display must not hide it from edge detection."""
    s = _settings_for(experiment)
    s.outline = ["r"]
    s.channels = ["g"]
    seen = {}

    def _outline(**kwargs):
        seen.update(kwargs)
        return kwargs["base_img"]

    monkeypatch.setattr(annotate_mod, "outline_image", _outline)
    row = (str(experiment / "data" / "images" / "cell_00.png"), None)
    annotate_mod._load_thumb_image_worker(row, None, s)
    assert seen["full_img"] is not seen["base_img"]
    assert seen["outline_channels"] == ["r"]


# ---------------------------------------------------------------------------
# Counting the population (runs on the count worker)
# ---------------------------------------------------------------------------

def test_the_uncertainty_queue_orders_the_crops_and_says_what_it_did(
        experiment: Path, monkeypatch):
    from spacr import active_learning as al
    s = _settings_for(experiment)
    s.queue_by_uncertainty = True
    monkeypatch.setattr(al, "build_queue", lambda *a, **k: "the-queue")
    monkeypatch.setattr(al, "queue_rows",
                        lambda q: [("a.png", None), ("b.png", 1)])
    monkeypatch.setattr(al, "format_queue_summary", lambda q: "2 by entropy")
    out = annotate_mod._compute_total(s, filter_active=False)
    assert out["total"] == 2
    assert out["queue_summary"] == "2 by entropy"
    assert out["note"] == ""


def test_asking_for_the_uncertainty_queue_before_any_model_ran_says_so(
        experiment: Path, monkeypatch):
    """No scores yet is the ordinary state, not an error.

    The grid falls back to page order and the page label explains why --
    an empty grid with no message reads as a broken database.
    """
    from spacr import active_learning as al
    s = _settings_for(experiment)
    s.queue_by_uncertainty = True
    monkeypatch.setattr(
        al, "build_queue",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no scores yet")))
    out = annotate_mod._compute_total(s, filter_active=False)
    assert out["filtered_rows"] is None
    assert out["total"] == 8
    assert "no scores yet" in out["note"]


def test_a_threshold_filter_counts_only_the_rows_that_pass_it(
        experiment: Path, monkeypatch):
    s = _settings_for(experiment)
    s.measurement = "cell_area"
    s.threshold = 100.0
    s.threshold_direction = "higher"
    seen = {}

    def _fetch(db_path, column, measurements, thresholds, directions,
               image_type):
        seen.update(measurements=measurements, thresholds=thresholds,
                    directions=directions)
        return [("a.png", None), ("b.png", None), ("c.png", 1)]

    monkeypatch.setattr(annotate_mod, "fetch_filtered_paths", _fetch)
    out = annotate_mod._compute_total(s, filter_active=True)
    assert out["total"] == 3
    # A single measurement is wrapped, so the engine always sees lists.
    assert seen == {"measurements": ["cell_area"], "thresholds": [100.0],
                    "directions": ["higher"]}


def test_no_filter_and_no_queue_counts_the_whole_table(experiment: Path):
    out = annotate_mod._compute_total(_settings_for(experiment),
                                       filter_active=False)
    assert out == {"filtered_rows": None, "total": 8, "queue_summary": "",
                   "note": ""}


# ---------------------------------------------------------------------------
# The settings dialog
# ---------------------------------------------------------------------------

def test_a_colour_vision_preference_that_cannot_be_read_falls_back_to_rgb(
        qtbot, qt_theme_applied, monkeypatch):
    """Settings starts the primaries picker from the global preference.

    If that read raises there is still a picker to draw, and the only honest
    default is the identity: guessing a colourblind mode for somebody who
    never asked for one recolours every crop they look at.
    """
    import spacr.qt.preferences as prefs_mod
    monkeypatch.setattr(
        prefs_mod, "image_display_primaries",
        lambda: (_ for _ in ()).throw(RuntimeError("prefs unreadable")))
    dlg = annotate_mod._SettingsDialog(engine.AnnotateSettings())
    qtbot.addWidget(dlg)
    assert dlg._display_primaries.currentData() == "rgb"


@pytest.mark.parametrize("stored", ["lower", ["lower"], ("Lower", "higher")])
def test_a_saved_lower_threshold_direction_reopens_as_lower(
        qtbot, qt_theme_applied, stored):
    """The direction is stored as a string OR as a per-column list.

    Reopening Settings with a list-shaped direction used to show "higher"
    whatever had been saved, so pressing OK silently inverted the filter.
    """
    settings = engine.AnnotateSettings()
    settings.threshold_direction = stored
    dlg = annotate_mod._SettingsDialog(settings)
    qtbot.addWidget(dlg)
    assert dlg._threshold_dir.currentText() == "lower"


def test_the_sql_picker_looks_in_the_folder_the_dialog_shows_right_now(
        qtbot, qt_theme_applied, experiment: Path):
    """Users set the source and the column in one visit to this dialog."""
    dlg = annotate_mod._SettingsDialog(engine.AnnotateSettings())
    qtbot.addWidget(dlg)
    dlg._src_edit.setText(f"  {experiment}  ")
    assert dlg._picker_db_path() == str(experiment)


def test_browsing_for_a_source_fills_the_field_and_cancelling_leaves_it(
        qtbot, qt_theme_applied, monkeypatch, experiment: Path):
    dlg = annotate_mod._SettingsDialog(engine.AnnotateSettings())
    qtbot.addWidget(dlg)
    dlg._src_edit.setText("/before")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    dlg._pick_src()
    assert dlg._src_edit.text() == "/before"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(experiment)))
    dlg._pick_src()
    assert dlg._src_edit.text() == str(experiment)


def test_a_threshold_that_is_not_a_number_is_dropped_rather_than_crashing(
        qtbot, qt_theme_applied):
    """Thresholds are free text. "500, big, 200" must still give two."""
    dlg = annotate_mod._SettingsDialog(engine.AnnotateSettings())
    qtbot.addWidget(dlg)
    dlg._measurement.setText("cell_area,nucleus_area,pathogen_area")
    dlg._threshold.setText("500, big, 200")
    out = dlg.collect()
    assert out.threshold == [500.0, 200.0]


def test_a_threshold_field_of_nothing_but_words_clears_the_filter(
        qtbot, qt_theme_applied):
    """No parseable threshold means no filter, not a filter on nothing."""
    dlg = annotate_mod._SettingsDialog(engine.AnnotateSettings())
    qtbot.addWidget(dlg)
    dlg._measurement.setText("cell_area")
    dlg._threshold.setText("big")
    out = dlg.collect()
    assert out.threshold is None
    assert out.threshold_direction is None


# ---------------------------------------------------------------------------
# The auto-annotate dialog
# ---------------------------------------------------------------------------

@pytest.fixture
def auto_dialog(qtbot, qt_theme_applied, experiment: Path):
    settings = _settings_for(experiment)
    dlg = annotate_mod._AutoAnnotateDialog(settings)
    qtbot.addWidget(dlg)
    return dlg


def test_blank_lines_between_rules_are_not_read_as_rules(auto_dialog):
    """Users separate rules with blank lines; each one is not a broken rule."""
    auto_dialog._rules.setPlainText(
        "cell_area > 500\n\n   \nnucleus_area <= 200\n")
    assert auto_dialog.parsed_rules() == [
        {"column": "cell_area", "threshold": 500.0, "direction": "higher"},
        {"column": "nucleus_area", "threshold": 200.0, "direction": "lower"},
    ]


def test_the_values_hint_stays_quiet_when_the_column_cannot_be_read(
        auto_dialog, monkeypatch):
    """A metadata read that fails must not take the dialog down.

    The hint is a convenience; without it the user types the values by hand.
    """
    monkeypatch.setattr(
        engine, "metadata_values",
        lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("no")))
    auto_dialog._values.setPlaceholderText("untouched")
    auto_dialog._on_metadata_column()
    assert auto_dialog._values.placeholderText() == "untouched"


def test_the_values_hint_shows_what_the_column_actually_holds(auto_dialog,
                                                                monkeypatch):
    """A picker offering rows A-H to somebody whose plate is numbered is
    a picker they cannot use."""
    monkeypatch.setattr(engine, "metadata_values",
                        lambda *a, **k: [f"c{i}" for i in range(20)])
    auto_dialog._on_metadata_column()
    hint = auto_dialog._values.placeholderText()
    assert hint.startswith("c0, c1")
    assert "(+8 more)" in hint


def test_the_values_hint_is_not_read_while_the_measurement_source_is_picked(
        auto_dialog, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "metadata_values",
                        lambda *a, **k: calls.append(1) or [])
    auto_dialog._source.setCurrentIndex(1)      # measurement thresholds
    calls.clear()
    auto_dialog._on_metadata_column()
    assert calls == []


def test_previewing_with_no_source_open_says_so_instead_of_matching_nothing(
        qtbot, qt_theme_applied):
    settings = engine.AnnotateSettings()
    dlg = annotate_mod._AutoAnnotateDialog(settings)
    qtbot.addWidget(dlg)
    dlg._on_preview()
    assert dlg._preview_label.text() == "Open a source first."


def test_previewing_metadata_with_no_values_typed_takes_every_value(
        auto_dialog, monkeypatch):
    """Blank means "every value" — the placeholder says so."""
    monkeypatch.setattr(engine, "metadata_values", lambda *a, **k: ["c1", "c2"])
    seen = {}

    def _by_metadata(db_path, column, values):
        seen["values"] = values
        return ["a.png", "b.png", "a.png"]

    monkeypatch.setattr(engine, "paths_by_metadata", _by_metadata)
    auto_dialog._values.setText("   ")
    auto_dialog._on_preview()
    assert seen["values"] == ["c1", "c2"]
    # Duplicates are collapsed: a path counted twice would report a
    # population bigger than the one that gets written.
    assert auto_dialog.matched_paths() == ["a.png", "b.png"]
    assert auto_dialog._apply.isEnabled()
    assert "2 object(s) match" in auto_dialog._preview_label.text()


def test_previewing_measurements_with_no_rules_asks_for_one(auto_dialog):
    auto_dialog._source.setCurrentIndex(1)
    auto_dialog._rules.setPlainText("")
    auto_dialog._on_preview()
    assert auto_dialog._preview_label.text() == "Add at least one rule."
    assert not auto_dialog._apply.isEnabled()


def test_a_rule_naming_a_measurement_that_does_not_exist_says_which(
        auto_dialog, monkeypatch):
    """The Apply button must go back to disabled: the last good preview's
    population is not this rule's."""
    monkeypatch.setattr(engine, "paths_by_metadata",
                        lambda *a, **k: ["a.png"])
    auto_dialog._on_preview()
    assert auto_dialog._apply.isEnabled()
    auto_dialog._source.setCurrentIndex(1)
    auto_dialog._rules.setPlainText("no_such_column > 1")
    monkeypatch.setattr(
        engine, "paths_by_measurements",
        lambda *a, **k: (_ for _ in ()).throw(
            ValueError("no measurement table has 'no_such_column'")))
    auto_dialog._on_preview()
    assert "no_such_column" in auto_dialog._preview_label.text()
    assert not auto_dialog._apply.isEnabled()
    assert auto_dialog.matched_paths() == []


def test_a_preview_that_matches_nothing_says_nothing_matches(auto_dialog,
                                                               monkeypatch):
    monkeypatch.setattr(engine, "paths_by_metadata", lambda *a, **k: [])
    auto_dialog._on_preview()
    assert auto_dialog._preview_label.text() == "Nothing matches."
    assert not auto_dialog._apply.isEnabled()


def test_the_gate_and_umap_buttons_close_with_their_own_result_codes(
        auto_dialog):
    """Distinct from Accepted and Rejected, so the caller can tell
    "annotate this" from "take me there"."""
    auto_dialog._on_open_gate_editor()
    assert auto_dialog.result() == annotate_mod._AUTO_ANNOTATE_OPEN_GATE
    auto_dialog._on_open_umap()
    assert auto_dialog.result() == annotate_mod._AUTO_ANNOTATE_OPEN_UMAP
    assert annotate_mod._AUTO_ANNOTATE_OPEN_GATE not in (
        QDialog.Accepted, QDialog.Rejected)


# ---------------------------------------------------------------------------
# Building the screen
# ---------------------------------------------------------------------------

def test_a_drop_zone_that_will_not_install_still_leaves_a_usable_screen(
        qtbot, qt_theme_applied, monkeypatch):
    """Drag-and-drop is a convenience. Losing it must not lose Annotate."""
    import spacr.qt.dnd as dnd_mod
    monkeypatch.setattr(
        dnd_mod, "install_dropzone",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no dnd")))
    scr = annotate_mod.AnnotateScreen()
    qtbot.addWidget(scr)
    assert scr._thumbs


def test_a_grid_with_no_viewport_yet_falls_back_to_the_last_good_shape(
        bare_screen):
    """The first paint measures a zero-size viewport.

    Sizing the grid from that gives one enormous cell; the previous shape is
    the only sensible answer until the viewport is realized.
    """
    scr = bare_screen
    scr._settings.grid_rows = 4
    scr._settings.grid_cols = 6
    scr._grid_scroll = None
    scr._compute_grid_dims()
    assert (scr._settings.grid_rows, scr._settings.grid_cols) == (4, 6)
    scr._settings.grid_rows = 0
    scr._settings.grid_cols = 0
    scr._compute_grid_dims()
    assert (scr._settings.grid_rows, scr._settings.grid_cols) == (5, 5)


# ---------------------------------------------------------------------------
# Opening a source
# ---------------------------------------------------------------------------

def test_cancelling_the_source_picker_leaves_the_open_experiment_alone(
        bare_screen, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    bare_screen._on_pick_source()
    assert bare_screen._settings.src == ""


def test_picking_a_source_opens_it(bare_screen, monkeypatch, experiment: Path):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(experiment)))
    bare_screen._on_pick_source()
    assert bare_screen._settings.src == str(experiment)


def test_a_folder_with_no_measurements_database_is_refused_when_declined(
        bare_screen, monkeypatch, tmp_path: Path):
    """The question is worth asking -- a mistyped folder is the usual cause --
    and answering No must leave the previous experiment open."""
    ask = _Recorder(answer=QMessageBox.No)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(ask))
    empty = tmp_path / "not-an-experiment"
    empty.mkdir()
    bare_screen._open_source(str(empty))
    assert ask.shown
    assert "measurements.db" in ask.body()
    assert bare_screen._settings.src == ""
    assert bare_screen._worker is None


def test_a_folder_with_no_database_is_opened_anyway_when_confirmed(
        bare_screen, monkeypatch, tmp_path: Path):
    """Measure may not have run yet; the user is allowed to say so."""
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(_Recorder(answer=QMessageBox.Yes)))
    empty = tmp_path / "future-experiment"
    empty.mkdir()
    bare_screen._open_source(str(empty))
    assert bare_screen._settings.src == str(empty)
    assert bare_screen._worker is not None


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def test_resizing_before_the_grid_exists_is_ignored(bare_screen):
    from PySide6.QtGui import QResizeEvent
    from PySide6.QtCore import QSize
    scr = bare_screen
    scr._grid_scroll = None
    scr.resizeEvent(QResizeEvent(QSize(400, 300), QSize(300, 300)))
    assert not scr._resize_timer.isActive()


def test_a_resize_that_changes_the_grid_shape_reloads_once_it_settles(
        screen, qtbot):
    """One QThread per geometry event is what the debounce exists to stop."""
    from PySide6.QtGui import QResizeEvent
    from PySide6.QtCore import QSize
    scr = screen
    scr._compute_grid_dims = lambda: (
        setattr(scr._settings, "grid_rows", 2),
        setattr(scr._settings, "grid_cols", 2))
    scr.resizeEvent(QResizeEvent(QSize(400, 300), QSize(300, 300)))
    assert scr._resize_timer.isActive()
    scr._compute_grid_dims = lambda: None
    scr._reload_after_resize()
    qtbot.waitUntil(lambda: len(scr._thumbs) == 4, timeout=5000)


def test_a_resize_arriving_during_teardown_loads_nothing(screen):
    scr = screen
    scr._closing = True
    before = list(scr._page_paths)
    scr._reload_after_resize()
    assert scr._page_paths == before


# ---------------------------------------------------------------------------
# Settings round trip
# ---------------------------------------------------------------------------

def test_cancelling_settings_changes_nothing(screen, monkeypatch):
    scr = screen
    monkeypatch.setattr(annotate_mod._SettingsDialog, "exec",
                        lambda self: QDialog.Rejected)
    before = scr._settings.annotation_column
    scr._on_open_settings()
    assert scr._settings.annotation_column == before


def test_applying_settings_that_keep_the_same_source_only_recounts(
        screen, monkeypatch, qtbot):
    scr = screen
    monkeypatch.setattr(annotate_mod._SettingsDialog, "exec",
                        lambda self: QDialog.Accepted)
    opened = []
    monkeypatch.setattr(scr, "_open_source", lambda src: opened.append(src))
    scr._on_open_settings()
    assert opened == []
    qtbot.waitUntil(lambda: not scr.is_busy(), timeout=10000)


def test_applying_settings_that_change_the_annotation_column_reopens_the_source(
        screen, monkeypatch):
    """The save worker writes into ONE column. A column change that did not
    restart it would keep writing into the old one."""
    scr = screen
    monkeypatch.setattr(annotate_mod._SettingsDialog, "exec",
                        lambda self: QDialog.Accepted)

    real_collect = annotate_mod._SettingsDialog.collect

    def _collect(self):
        self._ann_col.setText("infected")
        return real_collect(self)

    monkeypatch.setattr(annotate_mod._SettingsDialog, "collect", _collect)
    opened = []
    monkeypatch.setattr(scr, "_open_source", lambda src: opened.append(src))
    scr._on_open_settings()
    assert opened == [scr._settings.src]
    assert scr._settings.annotation_column == "infected"


# ---------------------------------------------------------------------------
# Paging and counting
# ---------------------------------------------------------------------------

def test_skipping_to_the_last_annotation_says_so_when_there_is_none(screen):
    scr = screen
    scr._on_skip()
    assert scr._status_label.text() == "No annotated images found."
    assert scr._offset == 0


def test_skipping_lands_on_the_page_holding_the_last_annotation(screen, qtbot):
    scr = screen
    scr._settings.page_size_override = None
    scr._on_thumb_left(0)
    scr._flush_pending()
    qtbot.waitUntil(lambda: not scr._worker.busy, timeout=10000)
    scr._offset = 0
    scr._on_skip()
    assert scr._offset == 0
    assert scr._status_label.text() != "No annotated images found."


def test_class_counts_on_a_page_nobody_has_labelled_says_so(screen,
                                                              monkeypatch):
    info = _Recorder()
    monkeypatch.setattr(QMessageBox, "information", staticmethod(info))
    screen._on_class_counts()
    assert "No annotated rows yet." in info.body()


def test_class_counts_names_every_class_with_the_colour_it_is_drawn_in(
        screen, monkeypatch, qtbot):
    """The colour is half the answer: "class 3 has 40" is not actionable
    without knowing which ring on the grid is class 3."""
    scr = screen
    scr._on_thumb_left(0)      # class 1
    scr._on_thumb_right(1)     # class 2
    scr._flush_pending()
    qtbot.waitUntil(lambda: not scr._worker.busy, timeout=10000)
    info = _Recorder()
    monkeypatch.setattr(QMessageBox, "information", staticmethod(info))
    scr._on_class_counts()
    body = info.body()
    assert "Class    Count    Color" in body
    assert annotate_mod.label_to_hex(1, dark=annotate_mod.on_dark_theme()) in body


def test_a_count_delivered_after_the_screen_started_closing_is_dropped(screen):
    """Painting a count onto a half-destroyed screen is the crash this
    guard exists for."""
    scr = screen
    scr._closing = True
    scr._apply_total({"total": 999})
    assert scr._total != 999


def test_a_count_that_could_not_build_its_queue_puts_the_reason_on_the_label(
        screen):
    scr = screen
    scr._closing = False
    ran = []
    scr._apply_total({"total": 8, "note": "Uncertainty queue unavailable: x"},
                      then=lambda: ran.append(1))
    assert scr._page_label.text() == "Uncertainty queue unavailable: x"
    assert ran == [1]


def test_a_routed_subset_is_counted_without_touching_the_database(screen):
    """Rebuilding the population under a routed request would replace the
    twelve crops somebody was sent here to look at with ninety thousand."""
    scr = screen
    scr._object_rows = [("a.png", None), ("b.png", 1)]
    ran = []
    scr._refresh_total(then=lambda: ran.append(1))
    assert scr._total == 2
    assert scr._filtered_rows == scr._object_rows
    assert ran == [1]


def test_a_page_load_during_teardown_does_nothing(screen):
    scr = screen
    scr._closing = True
    before = list(scr._page_paths)
    scr._load_page()
    assert scr._page_paths == before
    scr._queue_page_load((99, [], None, scr._settings))
    assert scr._pending_page_load is None


def test_a_page_longer_than_the_grid_fills_it_and_stops(screen):
    """A settings change can shrink the grid between request and delivery."""
    scr = screen
    extra = len(scr._thumbs) + 3
    loaded = [(Image.new("RGB", (8, 8)), None) for _ in range(extra)]
    scr._on_page_loaded(scr._page_gen, loaded)
    assert scr._raw_thumb_images[len(scr._thumbs) - 1] is not None


def test_a_page_from_a_superseded_load_is_discarded(screen):
    scr = screen
    scr._page_gen += 1
    marker = Image.new("RGB", (8, 8), (7, 7, 7))
    scr._on_page_loaded(scr._page_gen - 1, [(marker, None)])
    assert scr._raw_thumb_images[0] is not marker


# ---------------------------------------------------------------------------
# Active learning
# ---------------------------------------------------------------------------

def test_the_round_strip_hides_itself_when_no_experiment_is_open(bare_screen):
    """Round 0 of nothing is noise on a screen with no crops on it."""
    scr = bare_screen
    scr._settings.db_path = "/no/such/measurements.db"
    scr._refresh_round_state()
    assert scr._round_index == 0
    assert scr._stop_verdict is None
    assert scr._al_label.isHidden()


def test_a_round_counter_that_cannot_be_read_never_stops_somebody_annotating(
        screen, monkeypatch):
    """Bookkeeping is not worth a blocked screen."""
    from spacr import active_learning as al
    monkeypatch.setattr(
        al, "next_round",
        lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("x")))
    scr = screen
    scr._round_index = 7
    scr._refresh_round_state()
    assert scr._round_index == 0
    assert scr._stop_verdict is None


def test_the_round_strip_reports_the_last_fitted_round_from_the_curve(screen):
    """Reopening the screen must still show where the curve got to --
    the numbers are in the database, not only in this session."""
    import pandas as pd
    scr = screen
    scr._last_round = None
    scr._stop_verdict = None
    curve = pd.DataFrame({"n_labels": [40, 90],
                          "holdout_accuracy": [0.71, 0.83]})
    scr._refresh_al_label(curve)
    text = scr._al_label.text()
    assert "90 labels" in text
    assert "held-out 0.830" in text


def test_a_round_with_no_held_out_score_still_reports_its_label_count(screen):
    import pandas as pd
    scr = screen
    scr._last_round = None
    scr._stop_verdict = None
    curve = pd.DataFrame({"n_labels": [40], "holdout_accuracy": [None]})
    scr._refresh_al_label(curve)
    assert "40 labels" in scr._al_label.text()
    assert "held-out" not in scr._al_label.text()


def test_coverage_needs_a_source_before_it_can_say_where_labels_came_from(
        bare_screen, monkeypatch):
    info = _Recorder()
    monkeypatch.setattr(QMessageBox, "information", staticmethod(info))
    bare_screen._on_coverage()
    assert "Open an experiment source" in info.body()


def test_a_coverage_report_that_cannot_be_built_names_the_failure(
        screen, monkeypatch):
    """"Coverage unavailable" with no reason is a dead end for the user."""
    from spacr import active_learning as al
    monkeypatch.setattr(
        al, "annotation_coverage",
        lambda *a, **k: (_ for _ in ()).throw(KeyError("plate")))
    warn = _Recorder()
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(warn))
    screen._on_coverage()
    assert "KeyError" in warn.body()


def test_the_coverage_report_opens_in_a_window_that_keeps_its_columns(
        screen, monkeypatch):
    from spacr import active_learning as al
    monkeypatch.setattr(al, "annotation_coverage", lambda *a, **k: {"n": 1})
    monkeypatch.setattr(al, "format_coverage_summary",
                        lambda cov: "plate  n\np1     40")
    shown = {}
    monkeypatch.setattr(annotate_mod._TextReportDialog, "exec",
                        lambda self: shown.update(body=self._view.toPlainText()))
    screen._on_coverage()
    assert shown["body"].startswith("plate  n")


def test_the_learning_curve_needs_a_source_too(bare_screen, monkeypatch):
    info = _Recorder()
    monkeypatch.setattr(QMessageBox, "information", staticmethod(info))
    bare_screen._on_learning_curve()
    assert "learning curve" in info.body()


def test_a_learning_curve_that_cannot_be_built_names_the_failure(
        screen, monkeypatch):
    from spacr import active_learning as al
    monkeypatch.setattr(
        al, "learning_curve",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no rounds table")))
    warn = _Recorder()
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(warn))
    screen._on_learning_curve()
    assert "no rounds table" in warn.body()


def test_the_learning_curve_opens_in_a_window_that_keeps_its_columns(
        screen, monkeypatch):
    from spacr import active_learning as al
    monkeypatch.setattr(al, "learning_curve", lambda *a, **k: [])
    monkeypatch.setattr(al, "should_stop", lambda curve: None)
    monkeypatch.setattr(al, "format_learning_curve",
                        lambda curve, verdict: "round  acc\n1      0.7")
    shown = {}
    monkeypatch.setattr(annotate_mod._TextReportDialog, "exec",
                        lambda self: shown.update(body=self._view.toPlainText()))
    screen._on_learning_curve()
    assert "round  acc" in shown["body"]


def test_retraining_needs_a_source(bare_screen, monkeypatch):
    info = _Recorder()
    monkeypatch.setattr(QMessageBox, "information", staticmethod(info))
    bare_screen._on_retrain()
    assert "before retraining" in info.body()


def test_a_second_retrain_click_while_one_is_running_is_refused_not_queued(
        screen):
    """Two fits on the same labels write the same scores twice and take
    twice as long; the button says so instead of starting one."""
    scr = screen
    scr._retrain_worker = object()
    scr._on_retrain()
    assert scr._status_label.text() == "A retrain is already running."
    scr._retrain_worker = None


def test_a_retrain_thread_that_already_retired_is_not_retired_twice(screen):
    scr = screen
    scr._retrain_worker = None
    scr._btn_retrain.setEnabled(False)
    scr._on_retrain_finished()
    assert scr._btn_retrain.isEnabled()


def test_retiring_a_retrain_thread_survives_signals_it_never_carried(screen):
    """Teardown order is not guaranteed; a disconnect that was already done
    must not become an exception in the event loop."""
    scr = screen
    worker = annotate_mod._RetrainWorker("db", "annotate", {})
    scr._retrain_worker = worker
    scr._on_retrain_finished()
    assert scr._retrain_worker is None
    assert scr._btn_retrain.isEnabled()


# ---------------------------------------------------------------------------
# Object routing
# ---------------------------------------------------------------------------

class _Request:
    """The shape `spacr.selection.ObjectRequest` presents to this screen."""

    def __init__(self, keys, reason="predicted infected", timelapse=False):
        self.keys = list(keys)
        self.reason = reason
        self.timelapse = timelapse

    def describe(self):
        return f"{len(self.keys)} objects · {self.reason}"


def test_objects_routed_here_with_no_source_open_say_to_open_one(bare_screen):
    """A grid of nothing with no explanation reads as a broken screen."""
    scr = bare_screen
    returned = scr.open_object_request(_Request(["a", "b"]))
    assert returned is scr
    assert scr._object_rows == []
    assert "no source is open" in scr._page_label.text()


def test_objects_that_cannot_be_looked_up_report_how_many_and_why(
        screen, monkeypatch):
    """And the request is dropped, so the grid goes back to the whole
    population rather than pinning itself to a subset that never loaded."""
    from spacr import active_learning as al
    monkeypatch.setattr(
        al, "crops_for_object_keys",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad key shape")))
    scr = screen
    scr.open_object_request(_Request(["a", "b", "c"]))
    assert scr._object_request is None
    assert scr._object_rows is None
    label = scr._page_label.text()
    assert "Could not open 3 objects" in label
    assert "bad key shape" in label


def test_objects_missing_from_this_database_are_counted_in_the_heading(
        screen, monkeypatch):
    """Twelve crops under a heading that says twenty is the failure this
    suffix exists to prevent."""
    from spacr import active_learning as al
    rows = [(str(p), None) for p, _ in screen._page_paths[:2]]
    monkeypatch.setattr(al, "crops_for_object_keys", lambda *a, **k: rows)
    scr = screen
    scr.open_object_request(_Request(["a", "b", "c", "d"]))
    assert scr._total == 2
    assert "2 of them are not in this database" in scr._request_note


def test_clearing_a_routing_request_that_was_never_made_does_nothing(screen):
    scr = screen
    scr._object_request = None
    scr._offset = 4
    scr.clear_object_request()
    assert scr._offset == 4


# ---------------------------------------------------------------------------
# Hand-offs to other modules
# ---------------------------------------------------------------------------

def test_training_a_classifier_needs_a_source(bare_screen, monkeypatch):
    info = _Recorder()
    monkeypatch.setattr(QMessageBox, "information", staticmethod(info))
    bare_screen._on_train_cv()
    assert "before training a classifier" in info.body()


def test_train_cv_hands_classify_the_annotation_mode_not_well_metadata(
        screen):
    """Leaving dataset_mode unset built the classes from well metadata and
    ignored the annotations that had just been made."""
    seeds = []
    screen.train_requested.connect(lambda mod, seed: seeds.append((mod, seed)))
    screen._on_train_cv()
    module, seed = seeds[0]
    assert module == "classify"
    assert seed["dataset_mode"] == "annotation"
    assert seed["generate_training_dataset"] is True
    assert seed["train"] is True
    assert seed["apply_model_to_dataset"] is True
    assert seed["src"] == screen._settings.src


def test_training_an_xgboost_model_needs_a_source(bare_screen, monkeypatch):
    info = _Recorder()
    monkeypatch.setattr(QMessageBox, "information", staticmethod(info))
    bare_screen._on_train_xg()
    assert "XGBoost" in info.body()


def test_train_xg_hands_ml_analyze_the_column_that_was_annotated(screen):
    seeds = []
    screen.train_requested.connect(lambda mod, seed: seeds.append((mod, seed)))
    screen._on_train_xg()
    module, seed = seeds[0]
    assert module == "ml_analyze"
    assert seed["model_type"] == "xgboost"
    assert seed["annotation_column"] == screen._settings.annotation_column


def test_browsing_the_database_needs_a_source(bare_screen, monkeypatch):
    info = _Recorder()
    monkeypatch.setattr(QMessageBox, "information", staticmethod(info))
    bare_screen._on_browse_db()
    assert "before browsing its database" in info.body()


def test_browsing_the_database_flushes_first_so_the_table_agrees_with_the_grid(
        screen):
    """A table that disagrees with the grid reads as "the annotations are
    not being saved", which is wrong and alarming."""
    scr = screen
    scr._on_thumb_left(0)
    assert scr._pending_updates
    seeds = []
    scr.train_requested.connect(lambda mod, seed: seeds.append((mod, seed)))
    scr._on_browse_db()
    assert scr._pending_updates == {}
    module, seed = seeds[0]
    assert module == "db_browser"
    assert seed["table"] == "png_list"


# ---------------------------------------------------------------------------
# Clearing a column
# ---------------------------------------------------------------------------

def test_declining_the_clear_column_warning_leaves_every_annotation(
        screen, monkeypatch, qtbot):
    scr = screen
    scr._on_thumb_left(0)
    scr._flush_pending()
    qtbot.waitUntil(lambda: not scr._worker.busy, timeout=10000)
    ask = _Recorder(answer=QMessageBox.No)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(ask))
    scr._on_clear_column()
    assert "cannot be undone" in ask.body()
    assert engine.class_counts(scr._settings.db_path,
                                scr._settings.annotation_column)


def test_confirming_clear_column_empties_it_and_drops_unsaved_labels(
        screen, monkeypatch, qtbot):
    """Pending labels are dropped on purpose: letting them land afterwards
    would repopulate the column the user just cleared."""
    scr = screen
    scr._on_thumb_left(0)
    scr._flush_pending()
    qtbot.waitUntil(lambda: not scr._worker.busy, timeout=10000)
    scr._on_thumb_left(1)
    assert scr._pending_updates
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(_Recorder(answer=QMessageBox.Yes)))
    scr._on_clear_column()
    assert scr._pending_updates == {}
    assert engine.class_counts(scr._settings.db_path,
                                scr._settings.annotation_column) == []
