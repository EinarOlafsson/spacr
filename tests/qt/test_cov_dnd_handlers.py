"""The drop handlers nothing had ever dropped anything on.

Written for instruction 60. ``tests/qt/test_dnd_handlers_full.py`` covers the
pipeline modules' handlers thoroughly; the two dozen tool and results screens
below it, the layout-aware base class and the background folder scanner were
almost entirely unreached.

Every case drives a real path on disk against a screen double shaped like the
screen the handler is registered for, and asserts on what reached the screen
-- not on the handler having been called.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path

import pytest
from PySide6.QtWidgets import QLineEdit, QWidget

from spacr.qt import dnd_handlers as dh


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class _Console:
    def __init__(self):
        self.text = ""

    def append_stdout(self, s):
        self.text += s


class _Screen:
    """A plain, non-Qt screen: what a tool screen looks like to a handler."""

    def __init__(self, **attrs):
        self._console = _Console()
        for name, value in attrs.items():
            setattr(self, name, value)

    @property
    def log(self) -> str:
        return self._console.text


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _database(path: Path, tables=("png_list",)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for name in tables:
            conn.execute(f'CREATE TABLE "{name}" (a INTEGER)')
    return path


@pytest.fixture(autouse=True)
def _drain_folder_scans(monkeypatch):
    """No test may walk away from a running folder scan.

    Qt aborts the process when a running QThread is destroyed with the object
    that owns it, so every scanner built during a test is shut down here
    whether the test waited for it or not.
    """
    made = []
    real_init = dh._DropScanner.__init__

    def _spy(self, screen):
        real_init(self, screen)
        made.append(self)

    monkeypatch.setattr(dh._DropScanner, "__init__", _spy)
    yield made
    for scanner in made:
        try:
            scanner.shutdown()
        except Exception:
            pass


def _settle(qtbot, screen, timeout=20000):
    """Pump the event loop until ``screen`` has no folder scan left."""
    qtbot.waitUntil(
        lambda: not dh.scan_is_busy(screen)
        and dh.active_scan_jobs(screen) == 0, timeout=timeout)


# ---------------------------------------------------------------------------
# The background folder scanner
# ---------------------------------------------------------------------------

def test_a_scanner_with_no_runner_shuts_down_quietly():
    scanner = dh._DropScanner(_Screen())
    scanner._runner = None
    scanner.shutdown()
    assert scanner.is_busy() is False
    assert scanner.active_jobs() == 0


def test_a_runner_whose_c_half_went_first_does_not_block_the_close():
    """Qt aborts if a running QThread is destroyed, so shutdown is the one
    thing that must not raise on its way out."""
    asked = []

    class _DeadRunner:
        def shutdown(self):
            asked.append(1)
            raise RuntimeError("Internal C++ object already deleted")

        def is_busy(self):
            raise RuntimeError("Internal C++ object already deleted")

    scanner = dh._DropScanner(_Screen())
    scanner._runner = _DeadRunner()
    scanner.shutdown()
    # It really tried, and the dead runner's refusal did not escape.
    assert asked == [1]


def test_a_scan_that_finishes_after_its_screen_closed_reports_nothing():
    """A completion handler touches widgets. Running one against a screen
    that has been destroyed is the crash, not the missing report."""
    import shiboken6
    widget = QWidget()                 # not qtbot's: it is destroyed below
    scanner = dh._DropScanner(widget)
    delivered = []
    shiboken6.delete(widget)
    scanner._deliver(delivered.append, "result")
    assert delivered == []


def test_closing_the_screen_shuts_its_scanner_down(qtbot):
    from PySide6.QtCore import QEvent
    widget = QWidget()
    qtbot.addWidget(widget)
    scanner = dh._DropScanner(widget)
    stopped = []
    scanner.shutdown = lambda: stopped.append(1)
    scanner.eventFilter(widget, QEvent(QEvent.Close))
    assert stopped == [1]


def test_a_screen_that_never_had_a_scanner_reports_no_jobs():
    assert dh.scan_is_busy(_Screen()) is False
    assert dh.active_scan_jobs(_Screen()) == 0


def test_a_scanner_whose_c_half_is_gone_reports_no_jobs(qtbot):
    import shiboken6
    widget = QWidget()
    qtbot.addWidget(widget)
    widget._dnd_scanner = dh._DropScanner(widget)
    shiboken6.delete(widget._dnd_scanner)
    assert dh.scan_is_busy(widget) is False
    assert dh.active_scan_jobs(widget) == 0


# ---------------------------------------------------------------------------
# The scans themselves
# ---------------------------------------------------------------------------

def test_a_folder_that_cannot_be_listed_reports_no_images(tmp_path,
                                                            monkeypatch):
    """A permission error is not zero images, but it is the same report --
    the alternative is a traceback out of Qt's drop dispatch."""
    folder = tmp_path / "plate"
    folder.mkdir()
    monkeypatch.setattr(
        Path, "iterdir",
        lambda self: (_ for _ in ()).throw(PermissionError("nope")))
    assert dh.scan_mask_folder(folder) == {"names": [], "total": 0}


def test_a_path_that_is_not_a_folder_reports_no_images(tmp_path):
    assert dh.scan_mask_folder(_touch(tmp_path / "a.txt")) == {
        "names": [], "total": 0}


# ---------------------------------------------------------------------------
# A stand-in for what spacr.chaining answers with
# ---------------------------------------------------------------------------

class _Target:
    def __init__(self, value, location=None, kind="measurements-db",
                 source="registry", paths=()):
        self.value = value
        self.location = location if location is not None else value
        self.kind = kind
        self.source = source
        self.paths = tuple(paths)


class _Resolution:
    def __init__(self, targets=(), choices=(), root="/root", reason="",
                 ok=True, ambiguous=False):
        self.targets = tuple(targets)
        self.choices = tuple(choices)
        self.root = root
        self.reason = reason
        self.ok = ok
        self.ambiguous = ambiguous

    def target_for(self, kind):
        for target in self.targets:
            if target.kind == kind:
                return target
        return None


class _Choice:
    def __init__(self, options):
        self.options = list(options)


@pytest.fixture
def resolves(monkeypatch):
    """Make spacr.chaining answer whatever the case under test needs."""
    def _install(resolution):
        monkeypatch.setattr(dh._ch, "resolve_drop",
                            lambda *a, **k: resolution)
        return resolution
    return _install


# ---------------------------------------------------------------------------
# The layout-aware base class
# ---------------------------------------------------------------------------

class _Deliverable(dh.LayoutDropHandler):
    kinds = ()
    suffixes = (".csv",)

    def __init__(self, app_key="tabulate"):
        super().__init__(app_key)
        self.delivered = []

    def deliver(self, screen, value, target):
        self.delivered.append((value, target))


def test_a_layout_handler_that_forgot_to_say_what_to_do_with_the_answer():
    """`deliver` is the one hook a subclass owes; not having it must be a
    loud programming error rather than a silent no-op drop."""
    with pytest.raises(NotImplementedError):
        dh.LayoutDropHandler("tabulate").deliver(_Screen(), "/x", None)


def test_a_file_that_is_already_the_artifact_skips_the_layout_walk(tmp_path):
    handler = _Deliverable()
    csv = _touch(tmp_path / "scores.csv")
    screen = _Screen()
    assert handler.can_accept(csv) is True
    handler.apply(csv, screen)
    assert handler.delivered == [(str(csv), None)]
    assert str(csv) in screen.log


def test_a_dropped_file_of_the_wrong_kind_is_refused_without_a_walk(tmp_path):
    assert _Deliverable().can_accept(_touch(tmp_path / "notes.txt")) is False


def test_an_ambiguous_folder_offers_the_candidates_rather_than_picking_one(
        tmp_path, resolves):
    """Reported as can_accept False on purpose: that is what routes the drop
    into the "did you mean..." chooser."""
    resolves(_Resolution(choices=[_Choice(["/a/x.db", "/b/x.db"])],
                          ambiguous=True))
    handler = _Deliverable()
    assert handler.can_accept(tmp_path) is False
    assert [p.name for p in handler.suggest_alternatives(tmp_path)] == \
        ["x.db", "x.db"]


def test_a_folder_nothing_can_be_resolved_in_names_the_screen_that_refused(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        dh._ch, "resolve_drop",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ports")))
    handler = _Deliverable("tabulate")
    assert handler.suggest_alternatives(tmp_path) == []
    assert handler.error_message(tmp_path) == \
        f"tabulate cannot use {tmp_path.name!r}."


def test_a_folder_that_resolved_to_nothing_says_what_is_missing(tmp_path,
                                                                  resolves):
    """The sentence is `ports.check_ready`'s own, so the drop report and the
    Run button's refusal say the same thing."""
    resolves(_Resolution(reason="no measurements.db under this project",
                          ok=False))
    handler = _Deliverable()
    screen = _Screen()
    assert handler.error_message(tmp_path) == \
        "no measurements.db under this project"
    handler.apply(tmp_path, screen)
    assert handler.delivered == []
    assert "no measurements.db under this project" in screen.log


def test_a_folder_that_could_not_be_read_at_all_still_reports_the_drop(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        dh._ch, "resolve_drop",
        lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    screen = _Screen()
    _Deliverable().apply(tmp_path, screen)
    assert "could not be read" in screen.log


def test_a_resolved_folder_delivers_the_artifact_and_says_where_it_came_from(
        tmp_path, resolves):
    """"from the registry" versus "from the layout" is the difference between
    where a run actually wrote and where the layout says it should have."""
    resolves(_Resolution(
        targets=[_Target("/p/measurements/measurements.db",
                          kind="measurements-db", source="registry")],
        root="/p"))
    handler = _Deliverable()
    screen = _Screen()
    handler.apply(tmp_path, screen)
    assert handler.delivered == [
        ("/p/measurements/measurements.db", handler.delivered[0][1])]
    assert "from the registry" in screen.log
    assert "/p" in screen.log


@pytest.fixture
def picks(monkeypatch):
    """Answer the "which one did you mean?" chooser without a dialog."""
    from spacr.qt import dnd as dnd_mod
    asked = []

    def _install(answer):
        def _dialog(screen, headline, question, options):
            asked.append((headline, question, list(options)))
            return answer(options) if callable(answer) else answer
        monkeypatch.setattr(dnd_mod, "choose_one_dialog", _dialog)
        return asked
    return _install


def test_a_chooser_that_cannot_be_shown_declines_rather_than_guessing(
        monkeypatch):
    """Silently taking the first of several is the failure the chooser
    exists to avoid, so headless declines."""
    from spacr.qt import dnd as dnd_mod
    monkeypatch.setattr(
        dnd_mod, "choose_one_dialog",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no display")))
    assert dh._ask_for_one(_Screen(), "h", "q", ["a", "b"]) is None


# ---------------------------------------------------------------------------
# Downstream measurement-database screens
# ---------------------------------------------------------------------------


def test_a_screen_that_cannot_open_a_database_says_so_rather_than_failing_mute(
        tmp_path):
    handler = dh.DatabaseDropHandler()
    assert "measurements.db" in handler.error_message(tmp_path)
    with pytest.raises(TypeError):
        handler.apply(tmp_path, _Screen())


def test_a_database_that_refused_to_open_reports_the_screens_own_reason(
        tmp_path):
    screen = _Screen(set_database=lambda path: False,
                      last_error="file is not a database")
    with pytest.raises(ValueError, match="file is not a database"):
        dh.DatabaseDropHandler().apply(tmp_path, screen)


def test_a_database_that_refused_without_a_reason_still_names_the_path(
        tmp_path):
    screen = _Screen(set_database=lambda path: False)
    with pytest.raises(ValueError, match=re.escape(str(tmp_path))):
        dh.DatabaseDropHandler().apply(tmp_path, screen)


# ---------------------------------------------------------------------------
# The generic source handler
# ---------------------------------------------------------------------------

def test_the_generic_source_drop_says_what_it_would_have_taken(tmp_path):
    assert "source folder" in dh.SourceDropHandler().error_message(tmp_path)


# ---------------------------------------------------------------------------
# Parameter Sweep
# ---------------------------------------------------------------------------

class _FileList:
    def __init__(self):
        self.paths = []

    def add_paths(self, paths):
        self.paths.extend(paths)
        return len(paths)


def test_a_count_table_dropped_on_the_sweep_is_not_filed_as_a_score(
        tmp_path, monkeypatch):
    """A count table filed as a score is not an error the user sees, it is
    a wrong sweep."""
    from spacr.qt.widgets import file_list
    counts = _touch(tmp_path / "counts.csv", "grna,count\n")
    scores = _touch(tmp_path / "scores.csv", "object,score\n")
    monkeypatch.setattr(
        file_list, "side_for_header",
        lambda p: "count" if "counts" in Path(p).name else "score")
    screen = _Screen(score_data=_FileList(), count_data=_FileList())
    dh.SweepInputsDropHandler().apply(tmp_path, screen)
    assert screen.count_data.paths == [str(counts)]
    assert screen.score_data.paths == [str(scores)]


def test_the_sweep_refuses_a_screen_with_no_score_or_count_lists(tmp_path):
    with pytest.raises(TypeError, match="score/count"):
        dh.SweepInputsDropHandler().apply(tmp_path, _Screen())


def test_the_sweep_refuses_a_folder_holding_no_tables(tmp_path):
    handler = dh.SweepInputsDropHandler()
    empty = tmp_path / "empty"
    empty.mkdir()
    screen = _Screen(score_data=_FileList(), count_data=_FileList())
    with pytest.raises(ValueError, match="score CSVs"):
        handler.apply(empty, screen)
    assert handler.can_accept(empty) is False
    assert handler.accepts_multiple() is True


# ---------------------------------------------------------------------------
# Explain CV and Investigate Hit
# ---------------------------------------------------------------------------

class _Panel:
    def __init__(self, *fields):
        for name in fields:
            setattr(self, name, QLineEdit())
        self.refreshed = 0

    def _refresh_prediction_columns(self):
        self.refreshed += 1


def test_explain_cv_takes_a_dropped_project_as_its_database(tmp_path):
    plate = tmp_path / "plate"
    db = _database(plate / "measurements" / "measurements.db")
    panel = _Panel("database", "predictions")
    screen = _Screen(explain=panel)
    handler = dh.ExplainCvInputsDropHandler()
    assert handler.can_accept(plate) is True
    handler.apply(plate, screen)
    assert panel.database.text() == str(db)


def test_explain_cv_takes_a_dropped_csv_as_predictions_and_rereads_its_columns(
        tmp_path):
    csv = _touch(tmp_path / "preds.csv", "object,p\n")
    panel = _Panel("database", "predictions")
    dh.ExplainCvInputsDropHandler().apply(csv, _Screen(explain=panel))
    assert panel.predictions.text() == str(csv)
    assert panel.refreshed == 1


def test_explain_cv_refuses_a_screen_with_no_input_panel(tmp_path):
    handler = dh.ExplainCvInputsDropHandler()
    assert "measurements.db" in handler.error_message(tmp_path)
    assert handler.accepts_multiple() is True
    with pytest.raises(TypeError, match="no input panel"):
        handler.apply(tmp_path, _Screen())


def test_a_fraction_table_that_cannot_be_opened_is_not_read_as_one(tmp_path):
    assert dh.InvestigateHitInputsDropHandler._looks_like_fractions(
        tmp_path / "never-existed.csv") is False


def test_investigate_hit_refuses_a_screen_with_no_input_panel(tmp_path):
    handler = dh.InvestigateHitInputsDropHandler()
    assert "regression-results folder" in handler.error_message(tmp_path)
    assert handler.accepts_multiple() is True
    with pytest.raises(TypeError, match="no input panel"):
        handler.apply(tmp_path, _Screen())


# ---------------------------------------------------------------------------
# External Masks
# ---------------------------------------------------------------------------

def test_external_masks_refuses_a_screen_with_no_mapping_table(tmp_path):
    handler = dh.ExternalMasksDropHandler()
    assert handler.accepts_multiple() is True
    assert "TIFF" in handler.error_message(tmp_path)
    with pytest.raises(TypeError, match="input-mapping table"):
        handler.apply(tmp_path, _Screen())


# ---------------------------------------------------------------------------
# Small path helpers
# ---------------------------------------------------------------------------

def test_a_folder_that_vanished_between_the_drop_and_the_walk_reports_nothing(
        tmp_path):
    """The walk happens after `apply` returns, so the folder can be gone --
    an unmounted share is the ordinary case -- and an OSError raised there
    has no Python caller to catch it."""
    gone = tmp_path / "unmounted"
    assert dh._contains_suffix(gone, (".tif",)) is False
    assert dh._contains_suffix(gone, (".tif",), recursive=True) is False


def test_a_settings_snapshot_is_found_beside_the_plate_and_under_settings(
        tmp_path):
    plate = tmp_path / "plate"
    top = _touch(plate / "mask_settings.csv", "Key,Value\n")
    nested = _touch(plate / "settings" / "measure_settings.csv", "Key,Value\n")
    _touch(plate / "notes.csv", "a\n")           # no "setting" in the name
    assert dh._settings_files(plate) == [nested, top]


def test_a_dropped_file_contributes_no_settings_snapshots(tmp_path):
    assert dh._settings_files(_touch(tmp_path / "x.csv")) == []


def test_a_snapshot_whose_name_says_nothing_falls_back_to_the_default():
    assert dh._module_from_settings(Path("run_2024.csv")) == "mask"
    assert dh._module_from_settings(Path("run_2024.csv"), "measure") == \
        "measure"
    assert dh._module_from_settings(Path("measure_crop_settings.csv")) == \
        "measure"


# ---------------------------------------------------------------------------
# Import Project
# ---------------------------------------------------------------------------

class _ForeignScreen(_Screen):
    def __init__(self, **attrs):
        super().__init__(**attrs)
        self.mapping = None
        self.measurements = None
        self.images = None
        self.mask_folders = []

    def load_mapping(self, path):
        self.mapping = path
        return getattr(self, "mapping_result", None)

    def set_measurements(self, path):
        self.measurements = path

    def set_images(self, path):
        self.images = path

    def add_mask_folder(self, object_type, path):
        self.mask_folders.append((object_type, path))
        return getattr(self, "mask_result", None)


def test_a_dropped_image_imports_the_folder_that_holds_it(tmp_path):
    image = _touch(tmp_path / "fields" / "a.tif")
    screen = _ForeignScreen()
    handler = dh.ForeignProjectDropHandler()
    assert handler.accepts_multiple() is True
    assert "JSON mapping" in handler.error_message(tmp_path)
    handler.apply(image, screen)
    assert screen.images == str(tmp_path / "fields")


# ---------------------------------------------------------------------------
# Align, Convert, Model Compare
# ---------------------------------------------------------------------------

def test_align_says_it_wants_tiles(tmp_path):
    assert "microscopy tiles" in dh.AlignDropHandler().error_message(tmp_path)


def test_model_compare_reports_a_folder_it_could_not_read(tmp_path):
    handler = dh.ImageFieldsDropHandler()
    assert "microscopy fields" in handler.error_message(tmp_path)
    screen = _Screen(set_source=lambda p: False, last_error="no fields here")
    with pytest.raises(ValueError, match="no fields here"):
        handler.apply(tmp_path, screen)


def test_model_compare_names_the_folder_when_the_screen_gives_no_reason(
        tmp_path):
    screen = _Screen(set_source=lambda p: False)
    with pytest.raises(ValueError, match="Could not load fields"):
        dh.ImageFieldsDropHandler().apply(tmp_path, screen)


# ---------------------------------------------------------------------------
# Plate Queue
# ---------------------------------------------------------------------------

def test_the_plate_queue_says_what_it_accepts(tmp_path):
    assert "settings/*.csv" in dh.PlateQueueDropHandler().error_message(
        tmp_path)


# ---------------------------------------------------------------------------
# Batch Runner
# ---------------------------------------------------------------------------


def test_a_plate_folders_snapshots_become_batch_jobs(tmp_path):
    plate = tmp_path / "plate"
    snapshot = _touch(plate / "settings" / "classify_settings.csv", "Key,Value\n")
    jobs = []
    screen = _Screen(add_job=lambda module, settings:
                      jobs.append((module, settings)) or True)
    handler = dh.BatchDropHandler()
    assert handler.accepts_multiple() is True
    assert "JSON/YAML queue" in handler.error_message(tmp_path)
    handler.apply(plate, screen)
    assert jobs == [("classify", str(snapshot))]


# ---------------------------------------------------------------------------
# Model Zoo
# ---------------------------------------------------------------------------

def test_the_model_zoo_says_what_it_takes(tmp_path):
    assert "checkpoint folder" in dh.ModelZooDropHandler().error_message(
        tmp_path)


def test_a_folder_with_a_checkpoint_deeper_down_is_still_scanned_for_models(
        tmp_path, qtbot):
    """The expensive question -- "is there one further down?" -- is answered
    off the GUI thread, and the branch it decides goes with it."""
    folder = tmp_path / "runs"
    _touch(folder / "fold_1" / "best.pth")
    scanned = []
    screen = _Screen(scan=lambda p: scanned.append(p),
                     set_fields_source=lambda p: True)
    dh.ModelZooDropHandler().apply(folder, screen)
    _settle(qtbot, screen)
    assert scanned == [str(folder)]


def test_a_folder_with_no_checkpoint_anywhere_is_used_as_benchmark_fields(
        tmp_path, qtbot):
    folder = tmp_path / "fields"
    _touch(folder / "a.tif")
    fields = []
    screen = _Screen(scan=lambda p: None,
                     set_fields_source=lambda p: fields.append(p))
    dh.ModelZooDropHandler().apply(folder, screen)
    _settle(qtbot, screen)
    assert fields == [str(folder)]


def test_fields_the_model_zoo_could_not_load_are_reported_not_swallowed(
        tmp_path):
    """`apply` returned long ago, so raising here would surface as an
    unhandled exception in the Qt event loop and the user would be told
    nothing at all."""
    folder = tmp_path / "fields"
    folder.mkdir()
    screen = _Screen(scan=lambda p: None,
                     set_fields_source=lambda p: False,
                     last_error="not an image folder")
    dh._apply_model_zoo_source(folder, screen, is_model=False)
    assert "not an image folder" in screen.log
    assert "[drop rejected]" in screen.log


def test_a_model_folder_decided_on_the_worker_thread_is_scanned(tmp_path):
    scanned = []
    dh._apply_model_zoo_source(tmp_path, _Screen(scan=scanned.append),
                                is_model=True)
    assert scanned == [str(tmp_path)]


# ---------------------------------------------------------------------------
# Results screens
# ---------------------------------------------------------------------------


def test_training_runs_reports_a_folder_it_could_not_scan(tmp_path):
    handler = dh.TrainingRunsDropHandler()
    assert "model training runs" in handler.error_message(tmp_path)
    assert handler.can_accept(tmp_path) is True
    screen = _Screen(scan=lambda p: False, last_error="no runs in there")
    with pytest.raises(ValueError, match="no runs in there"):
        handler.apply(tmp_path, screen)


def test_training_runs_names_the_folder_when_the_screen_gives_no_reason(
        tmp_path):
    screen = _Screen(scan=lambda p: False)
    with pytest.raises(ValueError, match="Could not scan"):
        dh.TrainingRunsDropHandler().apply(tmp_path, screen)


def test_the_report_screen_sets_the_run_folder_then_scans_it(tmp_path):
    handler = dh.ReportDropHandler()
    assert "completed spaCR run folder" in handler.error_message(tmp_path)
    order = []
    screen = _Screen(set_source=lambda p: order.append(("source", p)),
                     scan=lambda: order.append(("scan", None)))
    handler.apply(tmp_path, screen)
    assert [name for name, _ in order] == ["source", "scan"]


def test_a_report_folder_that_could_not_be_scanned_is_reported(tmp_path):
    screen = _Screen(set_source=lambda p: None, scan=lambda: False,
                      last_error="no results/ in there")
    with pytest.raises(ValueError, match="no results/ in there"):
        dh.ReportDropHandler().apply(tmp_path, screen)


def test_a_report_folder_refused_without_a_reason_names_the_path(tmp_path):
    screen = _Screen(set_source=lambda p: None, scan=lambda: False)
    with pytest.raises(ValueError, match="Could not scan"):
        dh.ReportDropHandler().apply(tmp_path, screen)


# ---------------------------------------------------------------------------
# Project-folder screens
# ---------------------------------------------------------------------------

def test_a_project_goes_into_whichever_setter_the_screen_actually_has(
        tmp_path, resolves):
    resolves(_Resolution(targets=[_Target(str(tmp_path))]))
    seen = []
    screen = _Screen(set_project=lambda p: seen.append(p))
    handler = dh.ProjectFolderDropHandler("pipeline_graph")
    assert handler.can_accept(tmp_path) is True
    handler.apply(tmp_path, screen)
    assert seen == [str(tmp_path)]


def test_a_screen_with_no_way_to_take_a_project_says_so(tmp_path, resolves):
    resolves(_Resolution(targets=[_Target(str(tmp_path))]))
    with pytest.raises(TypeError, match="no way to receive a project folder"):
        dh.ProjectFolderDropHandler("pipeline_graph").apply(tmp_path, _Screen())


def test_a_project_a_screen_refused_reports_the_screens_reason(tmp_path,
                                                                 resolves):
    resolves(_Resolution(targets=[_Target(str(tmp_path))]))
    screen = _Screen(load_project=lambda p: False, last_error="no run.json")
    with pytest.raises(ValueError, match="no run.json"):
        dh.ProjectFolderDropHandler("pipeline_graph").apply(tmp_path, screen)


def test_the_data_manager_measures_the_project_it_was_just_given(tmp_path,
                                                                   resolves):
    resolves(_Resolution(targets=[_Target(str(tmp_path))]))
    order = []
    screen = _Screen(set_project=lambda p: order.append("project"),
                     scan=lambda: order.append("scan"))
    dh.DataManagerDropHandler("data_manager").apply(tmp_path, screen)
    assert order == ["project", "scan"]


def test_a_root_the_project_browser_already_watches_is_not_an_error(
        tmp_path, resolves):
    """`add_root` returns False for a duplicate, and reporting that as a
    failed drop puts an error dialog in front of a no-op."""
    resolves(_Resolution(targets=[_Target(str(tmp_path))]))
    added = []
    screen = _Screen(add_root=lambda p: added.append(p) or False)
    handler = dh.ProjectRootsDropHandler("project_browser")
    assert handler.accepts_multiple() is True
    handler.apply(tmp_path, screen)
    assert added == [str(tmp_path)]


def test_run_history_refreshes_before_it_looks_for_the_dropped_run(tmp_path):
    order = []
    screen = _Screen(refresh=lambda: order.append("refresh"),
                     select_run=lambda p: order.append(("select", p)))
    handler = dh.RunHistoryDropHandler("run_history")
    assert handler.can_accept(tmp_path) is True
    handler.apply(tmp_path, screen)
    assert order == ["refresh", ("select", str(tmp_path))]
    assert str(tmp_path) in screen.log


def test_dropping_a_file_from_a_run_selects_the_run_folder_around_it(
        tmp_path):
    settings = _touch(tmp_path / "run_01" / "settings.csv")
    seen = []
    screen = _Screen(select_run=lambda p: seen.append(p))
    dh.RunHistoryDropHandler("run_history").apply(settings, screen)
    assert seen == [str(tmp_path / "run_01")]


def test_a_run_that_is_not_in_the_history_says_where_to_look(tmp_path):
    screen = _Screen(select_run=lambda p: False)
    with pytest.raises(ValueError, match="clear the filters above"):
        dh.RunHistoryDropHandler("run_history").apply(tmp_path, screen)


# ---------------------------------------------------------------------------
# Table screens
# ---------------------------------------------------------------------------


def test_image_scatter_fills_its_path_field_then_opens_the_source(tmp_path):
    db = _database(tmp_path / "measurements.db")
    field = QLineEdit()
    order = []
    screen = _Screen(_db=field, open_source=lambda: order.append(field.text()))
    dh.ScatterTableDropHandler("image_scatter").apply(db, screen)
    assert order == [str(db)]


def test_lineage_fills_its_path_field_then_loads(tmp_path):
    db = _database(tmp_path / "measurements.db")
    field = QLineEdit()
    order = []
    screen = _Screen(_db=field, load=lambda: order.append(field.text()))
    dh.LineageDropHandler("lineage").apply(db, screen)
    assert order == [str(db)]


# ---------------------------------------------------------------------------
# Regression results
# ---------------------------------------------------------------------------


def test_the_hit_list_takes_the_folder_a_dropped_results_file_sits_in(
        tmp_path, resolves):
    folder = tmp_path / "results"
    table = _touch(folder / "hits.csv")
    resolves(_Resolution(targets=[_Target(str(table), kind="regression-results")]))
    seen = []
    dh.ResultsFolderDropHandler("hit_list").apply(
        folder, _Screen(load_folder=seen.append))
    assert seen == [str(folder)]


# ---------------------------------------------------------------------------
# Mask and layer screens
# ---------------------------------------------------------------------------


def test_the_layer_viewer_adds_a_mask_as_labels_and_an_image_as_an_image(
        tmp_path, resolves):
    """A viewer stacks layers, so what a file IS decides which call it gets."""
    handler = dh.LayerStackDropHandler("layer_viewer")
    assert handler.accepts_multiple() is True
    labels, images = [], []
    screen = _Screen(add_labels_file=labels.append,
                      add_image_file=images.append)
    mask = _touch(tmp_path / "masks" / "cell.tif")
    handler.deliver(screen, str(mask), None)
    image = _touch(tmp_path / "fields" / "a.tif")
    handler.deliver(screen, str(image), None)
    assert labels == [str(mask)]
    assert images == [str(image)]


def test_the_layer_viewer_adds_nothing_when_the_chooser_is_cancelled(
        tmp_path, picks):
    masks = tmp_path / "masks"
    _touch(masks / "a.tif")
    _touch(masks / "b.tif")
    picks(None)
    labels = []
    screen = _Screen(add_labels_file=labels.append, add_image_file=lambda p: None)
    dh.LayerStackDropHandler("layer_viewer").deliver(screen, str(masks), None)
    assert labels == []


# ---------------------------------------------------------------------------
# Methods & Results, Classifier Evaluation, Distributed Jobs
# ---------------------------------------------------------------------------


def test_classifier_evaluation_takes_the_run_folder_a_bundle_file_sits_in(
        tmp_path, resolves):
    resolves(_Resolution())
    field = QLineEdit()
    order = []
    screen = _Screen(_source=field, scan=lambda: order.append(field.text()))
    handler = dh.EvaluationBundleDropHandler("classifier_evaluation")
    bundle = _touch(tmp_path / "run_01" / "evaluation.json", "{}")
    assert handler.can_accept(bundle) is True
    handler.apply(bundle, screen)
    assert order == [str(tmp_path / "run_01")]
    assert str(tmp_path / "run_01") in screen.log


def test_classifier_evaluation_prefers_the_resolved_run_folder(tmp_path,
                                                                 resolves):
    resolves(_Resolution(targets=[_Target("/registered/run",
                                           kind="model-weights")]))
    field = QLineEdit()
    screen = _Screen(_source=field, scan=lambda: None)
    dh.EvaluationBundleDropHandler("classifier_evaluation").apply(tmp_path,
                                                                   screen)
    assert field.text() == "/registered/run"


def test_distributed_jobs_takes_the_only_snapshot_in_a_plate_folder(tmp_path):
    plate = tmp_path / "plate"
    snapshot = _touch(plate / "settings" / "measure_settings.csv", "Key,Value\n")
    field = QLineEdit()

    class _Module:
        text = ""

        def setCurrentText(self, value):
            _Module.text = value

    screen = _Screen(_settings_path=field, _module=_Module())
    handler = dh.SubmissionSettingsDropHandler("distributed_jobs")
    assert handler.can_accept(plate) is True
    handler.apply(plate, screen)
    assert field.text() == str(snapshot)
    assert _Module.text == "measure"


def test_distributed_jobs_says_what_a_settings_snapshot_looks_like(tmp_path):
    handler = dh.SubmissionSettingsDropHandler("distributed_jobs")
    assert "settings/*.csv" in handler.error_message(tmp_path)
    empty = tmp_path / "plate"
    empty.mkdir()
    with pytest.raises(ValueError, match="No settings snapshot"):
        handler.apply(empty, _Screen(_settings_path=QLineEdit()))


# ---------------------------------------------------------------------------
# The last unreached corners of the pipeline handlers
# ---------------------------------------------------------------------------

def _src_screen(**attrs):
    """An AppScreen-shaped double whose src field can be read back."""
    src = QLineEdit()

    class _Model:
        _widgets = {"src": src}

        def set_value_for_key(self, key, value):
            return False

        def collect(self):
            return attrs.get("collected", {})

    screen = _Screen(_settings_model=_Model(), **{
        k: v for k, v in attrs.items() if k != "collected"})
    return screen, src


def test_a_dropped_container_file_is_opened_on_a_worker_not_here(
        tmp_path, monkeypatch, qtbot):
    """A container drop points ``src`` at once and opens the file elsewhere.

    THE PREMISE THAT WAS WRONG. This branch used to defer itself with
    ``QTimer.singleShot(0, ...)`` under the note "a single container reads a
    header, not a tree, so it stays on the GUI thread". A single-shot timer
    does not leave the GUI thread -- it runs its callback on it one turn
    later, with the event loop stopped -- and reading a header is a FILE
    OPEN, which on the maintainer's sleeping ``autofs`` share is the call
    that had not returned after twenty seconds. So the open moved to the
    scanner with the folder walk, and only the widget half comes back.
    """
    opened = []
    rendered = []

    def _scan(path):
        opened.append((path, threading.current_thread()))
        return {"found": True, "summary": "format=nd2", "rows": [],
                "error": ""}

    monkeypatch.setattr(dh, "scan_mask_container", _scan)
    monkeypatch.setattr(dh, "_render_container_report",
                        lambda path, screen, scan: rendered.append(path))
    container = _touch(tmp_path / "plate.nd2")
    screen, src = _src_screen()
    dh.MaskDropHandler().apply(container, screen)

    # The path is in the field before the file has been touched at all.
    assert src.text() == str(tmp_path)
    qtbot.waitUntil(lambda: rendered == [container], timeout=5000)
    assert [p for p, _ in opened] == [container]
    assert opened[0][1] is not threading.main_thread(), (
        "the container header was read on the GUI thread again")


def test_a_validated_regex_survives_an_editor_that_will_not_open(monkeypatch):
    """The confirmation step is a review. Losing the pattern because the
    dialog could not be built would leave the screen with no regex at all."""
    import spacr.qt.regex_editor as editor_mod
    monkeypatch.delattr(editor_mod, "RegexEditorDialog")
    pushed = []
    monkeypatch.setattr(dh, "_push_regex_to_screen",
                        lambda pattern, screen: pushed.append(pattern))
    dh._open_regex_editor(["a.tif"], "(?P<plateID>.*)", _Screen(),
                           confirming=True, fallback="(?P<plateID>.*)")
    assert pushed == ["(?P<plateID>.*)"]


def test_measure_refuses_a_path_that_is_neither_file_nor_folder(tmp_path):
    assert dh.MeasureDropHandler().can_accept(tmp_path / "never-existed") \
        is False


def test_a_measurements_drop_refuses_a_path_that_is_neither(tmp_path):
    assert dh.MeasurementsDropHandler().can_accept(tmp_path / "gone") is False


def test_the_sweep_contributes_nothing_from_a_path_that_is_not_there(
        tmp_path):
    assert dh.SweepInputsDropHandler._tables(tmp_path / "gone") == []


def test_investigate_hit_refuses_a_path_that_is_neither(tmp_path):
    assert dh.InvestigateHitInputsDropHandler().can_accept(
        tmp_path / "gone") is False


def test_external_masks_takes_a_folder_and_the_image_types_it_names(tmp_path):
    handler = dh.ExternalMasksDropHandler()
    folder = tmp_path / "images"
    folder.mkdir()
    assert handler.can_accept(folder) is True
    assert handler.can_accept(_touch(tmp_path / "a.ome.tif")) is True
    assert handler.can_accept(_touch(tmp_path / "notes.txt")) is False


def test_the_batch_runner_refuses_a_folder_with_no_snapshots_in_it(tmp_path):
    empty = tmp_path / "plate"
    empty.mkdir()
    assert dh.BatchDropHandler().can_accept(empty) is False
