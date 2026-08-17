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

import sqlite3
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QObject
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
    path.write_text(text)
    return path


def _database(path: Path, tables=("png_list",)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for name in tables:
            conn.execute(f'CREATE TABLE "{name}" (a INTEGER)')
    return path


# ---------------------------------------------------------------------------
# "Is the screen still there?"
# ---------------------------------------------------------------------------

def test_a_screen_that_is_not_a_qt_object_is_always_considered_alive():
    """Test doubles and small controllers cannot be half-deleted."""
    assert dh._is_alive(_Screen()) is True


def test_a_deleted_screen_is_recognised_without_shiboken(qtbot, monkeypatch):
    """`shiboken6.isValid` is the good answer; poking the object is the
    fallback, and both must agree that a destroyed widget is gone."""
    import shiboken6
    widget = QWidget()
    qtbot.addWidget(widget)
    monkeypatch.setitem(sys.modules, "shiboken6", None)
    assert dh._is_alive(widget) is True
    shiboken6.delete(widget)
    assert dh._is_alive(widget) is False


def test_a_shiboken_that_refuses_to_answer_is_read_as_gone(monkeypatch):
    """Guessing "still alive" would let a slot touch a destroyed widget,
    which is the abort this function exists to prevent."""
    class _Broken:
        @staticmethod
        def isValid(obj):
            raise RuntimeError("shiboken is unhappy")

    monkeypatch.setitem(sys.modules, "shiboken6", _Broken)
    assert dh._is_alive(QObject()) is False


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
    class _DeadRunner:
        def shutdown(self):
            raise RuntimeError("Internal C++ object already deleted")

    scanner = dh._DropScanner(_Screen())
    scanner._runner = _DeadRunner()
    scanner.shutdown()


def test_a_scan_that_finishes_after_its_screen_closed_reports_nothing(qtbot):
    """A completion handler touches widgets. Running one against a screen
    that has been destroyed is the crash, not the missing report."""
    import shiboken6
    widget = QWidget()
    qtbot.addWidget(widget)
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


def test_a_screen_that_cannot_hold_a_scanner_scans_inline_instead(tmp_path):
    """A leaked thread is worse than a stalled window, so a screen that
    refuses new attributes gets the synchronous scan."""
    class _Sealed:
        __slots__ = ()

    seen = []
    threaded = dh._scan_then(_Sealed(), lambda: "walked", seen.append)
    assert threaded is False
    assert seen == ["walked"]


def test_a_thread_qt_refuses_to_start_falls_back_to_an_inline_scan():
    """Better a stall than no report."""
    screen = _Screen()

    class _RefusingScanner:
        def submit(self, fn, on_done):
            raise RuntimeError("cannot create thread")

    screen._dnd_scanner = _RefusingScanner()
    seen = []
    assert dh._scan_then(screen, lambda: "walked", seen.append) is False
    assert seen == ["walked"]


def test_a_scan_that_raises_reports_nothing_rather_than_a_wrong_answer():
    class _Sealed:
        __slots__ = ()

    seen = []
    def _boom():
        raise OSError("permission denied")

    assert dh._scan_then(_Sealed(), _boom, seen.append) is False
    assert seen == []


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


def test_a_folder_with_no_recognisable_layout_is_never_fully_walked(
        tmp_path, monkeypatch):
    """The probe stops at 30 files; the point of returning early is that a
    100 000-file folder does not cost a full traversal on a drop."""
    from spacr.qt import folder_metadata as fm
    walked = []

    def _iter(path):
        for i in range(1000):
            walked.append(i)
            yield Path(f"{i}.tif")

    monkeypatch.setattr(fm, "iter_image_files", _iter)
    monkeypatch.setattr(fm, "detect_folder_metadata",
                        lambda path, files=None: None)
    out = dh.scan_folder_structure(tmp_path)
    assert out == {"labels": (), "rows": [], "error": ""}
    assert len(walked) <= dh._FOLDER_PROBE


def test_a_layout_detector_that_raises_reports_nothing_to_show(tmp_path,
                                                                 monkeypatch):
    from spacr.qt import folder_metadata as fm
    monkeypatch.setattr(
        fm, "iter_image_files",
        lambda path: (_ for _ in ()).throw(OSError("unreadable")))
    assert dh.scan_folder_structure(tmp_path) == {
        "labels": (), "rows": [], "error": ""}


def test_a_recognised_layout_is_planned_in_the_same_single_walk(tmp_path,
                                                                 monkeypatch):
    """The probe is drained into the planner rather than the tree being
    walked a second time."""
    from spacr.qt import folder_metadata as fm
    from spacr.qt import ingest_preview as ip

    class _Template:
        depth_labels = ("plate", "well")

    files = [Path(f"{i}.tif") for i in range(5)]
    monkeypatch.setattr(fm, "iter_image_files", lambda path: iter(files))
    monkeypatch.setattr(fm, "detect_folder_metadata",
                        lambda path, files=None: _Template())
    seen = {}

    def _plan(path, files=None, template=None):
        seen["files"] = list(files)
        return [{"plate": "p1"}]

    monkeypatch.setattr(ip, "plan_folder_extraction", _plan)
    out = dh.scan_folder_structure(tmp_path)
    assert out["labels"] == ("plate", "well")
    assert out["rows"] == [{"plate": "p1"}]
    assert seen["files"] == files


def test_a_plan_that_fails_reports_its_reason_beside_the_labels(tmp_path,
                                                                 monkeypatch):
    from spacr.qt import folder_metadata as fm
    from spacr.qt import ingest_preview as ip

    class _Template:
        depth_labels = ("plate",)

    monkeypatch.setattr(fm, "iter_image_files", lambda path: iter([]))
    monkeypatch.setattr(fm, "detect_folder_metadata",
                        lambda path, files=None: _Template())
    monkeypatch.setattr(
        ip, "plan_folder_extraction",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("two plates, one row")))
    out = dh.scan_folder_structure(tmp_path)
    assert out["labels"] == ("plate",)
    assert out["error"] == "two plates, one row"


def test_a_plan_that_fails_without_a_message_still_names_what_went_wrong(
        tmp_path, monkeypatch):
    from spacr.qt import folder_metadata as fm
    from spacr.qt import ingest_preview as ip

    class _Template:
        depth_labels = ("plate",)

    monkeypatch.setattr(fm, "iter_image_files", lambda path: iter([]))
    monkeypatch.setattr(fm, "detect_folder_metadata",
                        lambda path, files=None: _Template())
    monkeypatch.setattr(ip, "plan_folder_extraction",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError()))
    assert dh.scan_folder_structure(tmp_path)["error"] == "KeyError"


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
# Memoising one drop's resolution
# ---------------------------------------------------------------------------

def test_a_path_that_cannot_be_stat_ed_is_still_resolved(monkeypatch,
                                                           tmp_path):
    """`can_accept`, `error_message` and `apply` all ask the same question
    while the mouse button is still down; a vanished path must not make
    that four unanswered questions."""
    handler = dh.SourceDropHandler()
    calls = []
    monkeypatch.setattr(dh._ch, "resolve_drop",
                        lambda *a, **k: calls.append(1) or _Resolution())
    gone = tmp_path / "gone"
    dh._resolve_for(handler, "mask", gone)
    dh._resolve_for(handler, "mask", gone)
    assert calls == [1]          # memoised on the missing-stat key too


def test_a_resolution_that_raises_is_reported_as_no_answer(monkeypatch,
                                                             tmp_path):
    monkeypatch.setattr(
        dh._ch, "resolve_drop",
        lambda *a, **k: (_ for _ in ()).throw(KeyError("no such module")))
    assert dh._resolve_for(dh.SourceDropHandler(), "mask", tmp_path) is None


def test_a_folder_that_changed_since_the_last_drop_is_resolved_again(
        monkeypatch, tmp_path):
    """Run Measure and drop the same plate again: the database that appeared
    must be found rather than remembered as absent."""
    handler = dh.SourceDropHandler()
    calls = []
    monkeypatch.setattr(dh._ch, "resolve_drop",
                        lambda *a, **k: calls.append(1) or _Resolution())
    folder = tmp_path / "plate"
    folder.mkdir()
    dh._resolve_for(handler, "mask", folder)
    _touch(folder / "measurements.db")
    dh._resolve_for(handler, "mask", folder)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Which tables a dropped database holds
# ---------------------------------------------------------------------------

def test_a_dropped_csv_is_not_asked_for_its_tables(tmp_path):
    assert dh.table_names(_touch(tmp_path / "scores.csv")) == []


def test_a_database_under_a_folder_with_a_question_mark_still_opens(tmp_path):
    """Unquoted, everything after the `?` reads as URI query parameters and
    the open fails on a database that is perfectly fine."""
    folder = tmp_path / "plate?1#2"
    folder.mkdir()
    db = _database(folder / "measurements.db", tables=("png_list", "cell"))
    assert dh.table_names(db) == ["cell", "png_list"]


def test_a_file_named_like_a_database_that_is_not_one_lists_no_tables(
        tmp_path):
    assert dh.table_names(_touch(tmp_path / "broken.db", "not sqlite")) == []


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

def test_a_measurements_drop_prefers_where_the_run_actually_wrote(tmp_path,
                                                                    resolves):
    """The registry knows where Measure put the database; the declared
    layout only knows where it should have."""
    resolves(_Resolution(targets=[_Target("/p", location="/p/measurements",
                                            source="registry")]))
    screen = _Screen(app_key="umap", _settings_model=None)
    src = QLineEdit()
    screen._settings_model = type("M", (), {"_widgets": {"src": src},
                                             "set_value_for_key":
                                             lambda self, k, v: False})()
    dh.MeasurementsDropHandler().apply(tmp_path, screen)
    assert src.text() == "/p"
    assert "from the registry" in screen.log


def test_dropping_the_measurements_folder_sets_the_plate_above_it(tmp_path):
    """`src` means the plate, not the folder inside it."""
    plate = tmp_path / "plate"
    _database(plate / "measurements" / "measurements.db")
    src = QLineEdit()
    screen = _Screen(_settings_model=type(
        "M", (), {"_widgets": {"src": src},
                  "set_value_for_key": lambda self, k, v: False})())
    handler = dh.MeasurementsDropHandler()
    handler.apply(plate / "measurements" / "measurements.db", screen)
    assert src.text() == str(plate)


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
    with pytest.raises(ValueError, match=str(tmp_path)):
        dh.DatabaseDropHandler().apply(tmp_path, screen)


# ---------------------------------------------------------------------------
# The generic source handler
# ---------------------------------------------------------------------------

def test_the_generic_source_drop_says_what_it_would_have_taken(tmp_path):
    assert "source folder" in dh.SourceDropHandler().error_message(tmp_path)


def test_the_generic_source_drop_uses_the_answer_auto_chaining_would_give(
        tmp_path, resolves):
    """Dropping <plate>/measurements and letting the chain fill src cannot
    produce two different strings."""
    resolves(_Resolution(targets=[_Target("/p", location="/p/measurements",
                                            kind="measurements_db",
                                            source="layout")]))
    src = QLineEdit()
    screen = _Screen(app_key="tabulate", _settings_model=type(
        "M", (), {"_widgets": {"src": src},
                  "set_value_for_key": lambda self, k, v: False})())
    dh.SourceDropHandler().apply(tmp_path, screen)
    assert src.text() == "/p"
    assert "from the layout" in screen.log


def test_a_resolved_drop_on_a_screen_with_no_source_field_says_so(tmp_path,
                                                                    resolves):
    resolves(_Resolution(targets=[_Target("/p")]))
    with pytest.raises(TypeError, match="no source field"):
        dh.SourceDropHandler().apply(tmp_path, _Screen(app_key="tabulate"))


def test_an_unresolvable_drop_on_a_screen_with_no_source_field_says_so(
        tmp_path):
    with pytest.raises(TypeError, match="no source field"):
        dh.SourceDropHandler().apply(tmp_path, _Screen())


def test_dropping_a_file_sets_the_folder_that_holds_it(tmp_path):
    src = QLineEdit()
    screen = _Screen(_settings_model=type(
        "M", (), {"_widgets": {"src": src},
                  "set_value_for_key": lambda self, k, v: False})())
    dh.SourceDropHandler().apply(_touch(tmp_path / "plate" / "a.tif"), screen)
    assert src.text() == str(tmp_path / "plate")


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


def test_investigate_hit_tells_a_guide_fraction_table_from_a_prediction_one(
        tmp_path):
    """Two CSVs, two different fields, and only the header says which."""
    fractions = _touch(tmp_path / "f.csv", "grna,fraction,well\n1,0.2,A1\n")
    predictions = _touch(tmp_path / "p.csv", "object,score\n1,0.5\n")
    panel = _Panel("database", "regression_folder", "fractions", "predictions")
    screen = _Screen(investigate=panel)
    handler = dh.InvestigateHitInputsDropHandler()
    handler.apply(fractions, screen)
    assert panel.fractions.text() == str(fractions)
    handler.apply(predictions, screen)
    assert panel.predictions.text() == str(predictions)
    assert panel.refreshed == 1


def test_investigate_hit_takes_a_dropped_folder_as_the_results_folder(
        tmp_path):
    folder = tmp_path / "results"
    folder.mkdir()
    panel = _Panel("database", "regression_folder", "fractions", "predictions")
    dh.InvestigateHitInputsDropHandler().apply(folder,
                                                _Screen(investigate=panel))
    assert panel.regression_folder.text() == str(folder)


def test_investigate_hit_takes_a_dropped_project_as_its_database(tmp_path):
    plate = tmp_path / "plate"
    db = _database(plate / "measurements.db")
    panel = _Panel("database", "regression_folder", "fractions", "predictions")
    dh.InvestigateHitInputsDropHandler().apply(plate,
                                                _Screen(investigate=panel))
    assert panel.database.text() == str(db)


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

class _MaskModel:
    def __init__(self, widgets, dst_value=""):
        self._widgets = dict(widgets)
        self.written = {}
        self._dst_value = dst_value

    def _read_widget(self, widget):
        return self._dst_value

    def set_value_for_key(self, key, value):
        self.written[key] = value
        return True


def test_external_masks_names_a_destination_beside_the_dropped_folder(
        tmp_path):
    """A run that writes into the folder it read is the mistake this
    default exists to avoid."""
    folder = tmp_path / "images"
    folder.mkdir()
    inputs = _FileList()
    model = _MaskModel({"inputs": inputs, "dst": QLineEdit()})
    screen = _Screen(_settings_model=model)
    dh.ExternalMasksDropHandler().apply(folder, screen)
    assert inputs.paths == [str(folder)]
    assert model.written["dst"] == f"{folder}_spacr"
    assert "review the assignments before Run" in screen.log


def test_external_masks_leaves_a_destination_the_user_already_typed(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    model = _MaskModel({"inputs": _FileList(), "dst": QLineEdit()},
                        dst_value="/somewhere/chosen")
    dh.ExternalMasksDropHandler().apply(folder, _Screen(_settings_model=model))
    assert "dst" not in model.written


def test_external_masks_refuses_a_screen_with_no_mapping_table(tmp_path):
    handler = dh.ExternalMasksDropHandler()
    assert handler.accepts_multiple() is True
    assert "TIFF" in handler.error_message(tmp_path)
    with pytest.raises(TypeError, match="input-mapping table"):
        handler.apply(tmp_path, _Screen())


def test_a_folder_with_no_supported_images_is_refused_by_name(tmp_path):
    """Silently adding nothing looks like the drop worked."""
    folder = tmp_path / "docs"
    folder.mkdir()

    class _Nothing:
        def add_paths(self, paths):
            return 0

    model = _MaskModel({"inputs": _Nothing(), "dst": QLineEdit()})
    with pytest.raises(ValueError, match="No supported images"):
        dh.ExternalMasksDropHandler().apply(folder, _Screen(
            _settings_model=model))
