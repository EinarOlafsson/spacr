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
# "Is the screen still there?"
# ---------------------------------------------------------------------------

def test_a_screen_that_is_not_a_qt_object_is_always_considered_alive():
    """Test doubles and small controllers cannot be half-deleted."""
    assert dh._is_alive(_Screen()) is True


def test_a_deleted_screen_is_recognised_without_shiboken(monkeypatch):
    """`shiboken6.isValid` is the good answer; poking the object is the
    fallback, and both must agree that a destroyed widget is gone.

    Deliberately NOT registered with qtbot: the widget is destroyed here on
    purpose, and pytest-qt's teardown closes everything it was given.
    """
    import shiboken6
    widget = QWidget()
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


def test_a_mapping_csv_is_loaded_as_a_mapping_not_as_a_measurement_table(
        tmp_path):
    """Both are CSVs. Only the header says which, and filing a mapping as
    measurements would import the mapping's own columns as data."""
    mapping = _touch(tmp_path / "map.csv",
                     "source,target,transform,unit_in,unit_out,note\n")
    screen = _ForeignScreen()
    dh.ForeignProjectDropHandler().apply(mapping, screen)
    assert screen.mapping == str(mapping)
    assert screen.measurements is None


def test_a_plain_csv_is_imported_as_the_measurement_table(tmp_path):
    table = _touch(tmp_path / "cells.csv", "object,area\n")
    screen = _ForeignScreen()
    dh.ForeignProjectDropHandler().apply(table, screen)
    assert screen.measurements == str(table)


def test_a_mapping_that_would_not_load_is_reported(tmp_path):
    mapping = _touch(tmp_path / "map.json", "{}")
    screen = _ForeignScreen(mapping_result=False)
    with pytest.raises(ValueError, match="Could not load mapping"):
        dh.ForeignProjectDropHandler().apply(mapping, screen)


def test_a_folder_named_for_masks_is_added_as_masks_for_the_chosen_object(
        tmp_path):
    folder = tmp_path / "nucleus_masks"
    folder.mkdir()

    class _Box:
        @staticmethod
        def currentData():
            return "nucleus"

    screen = _ForeignScreen(_object_box=_Box())
    dh.ForeignProjectDropHandler().apply(folder, screen)
    assert screen.mask_folders == [("nucleus", str(folder))]


def test_a_mask_folder_dropped_before_an_object_type_was_picked_uses_cell(
        tmp_path):
    folder = tmp_path / "segmentation"
    folder.mkdir()
    screen = _ForeignScreen()          # no _object_box at all
    dh.ForeignProjectDropHandler().apply(folder, screen)
    assert screen.mask_folders == [("cell", str(folder))]


def test_a_mask_folder_that_was_refused_is_reported(tmp_path):
    folder = tmp_path / "label"
    folder.mkdir()
    screen = _ForeignScreen(mask_result=False)
    with pytest.raises(ValueError, match="Could not add mask folder"):
        dh.ForeignProjectDropHandler().apply(folder, screen)


def test_a_dropped_image_imports_the_folder_that_holds_it(tmp_path):
    image = _touch(tmp_path / "fields" / "a.tif")
    screen = _ForeignScreen()
    handler = dh.ForeignProjectDropHandler()
    assert handler.accepts_multiple() is True
    assert "JSON mapping" in handler.error_message(tmp_path)
    handler.apply(image, screen)
    assert screen.images == str(tmp_path / "fields")


def test_a_csv_that_cannot_be_read_is_not_guessed_at(tmp_path, monkeypatch):
    csv = _touch(tmp_path / "cells.csv", "object,area\n")
    monkeypatch.setattr(
        Path, "open",
        lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("nope")))
    screen = _ForeignScreen()
    dh.ForeignProjectDropHandler().apply(csv, screen)
    assert screen.measurements == str(csv)


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

class _Queue(list):
    def add(self, item):
        self.append(item)


class _QueueScreen(_Screen):
    def __init__(self, **attrs):
        super().__init__(**attrs)
        self._queue = _Queue()
        self.items = []
        self.refreshed = 0
        self.emitted = []

        class _Signal:
            def __init__(self, sink):
                self._sink = sink

            def emit(self, value):
                self._sink.append(value)

        self.queue_size_changed = _Signal(self.emitted)

    def queue(self):
        return self._queue

    def _refresh_table(self):
        self.refreshed += 1

    def add_item(self, module, settings):
        self.items.append((module, settings))


def test_a_plate_list_csv_queues_every_row_it_names(tmp_path, monkeypatch):
    from spacr.qt import plate_queue as pq
    csv = _touch(tmp_path / "plates.csv", "src\n/a\n/b\n")
    monkeypatch.setattr(pq, "import_plates_from_csv",
                        lambda path, base_settings=None, app_key="mask":
                        ["item-a", "item-b"])
    screen = _QueueScreen()
    dh.PlateQueueDropHandler().apply(csv, screen)
    assert list(screen.queue()) == ["item-a", "item-b"]
    assert screen.refreshed == 1
    assert screen.emitted == [2]


def test_a_plate_list_csv_with_no_src_column_is_refused_by_name(tmp_path,
                                                                 monkeypatch):
    from spacr.qt import plate_queue as pq
    csv = _touch(tmp_path / "plates.csv", "note\nhello\n")
    monkeypatch.setattr(pq, "import_plates_from_csv",
                        lambda path, base_settings=None, app_key="mask": [])
    with pytest.raises(ValueError, match="no plate rows with an src value"):
        dh.PlateQueueDropHandler().apply(csv, _QueueScreen())


def test_a_plate_folders_snapshots_are_queued_against_that_folder(tmp_path,
                                                                    monkeypatch):
    """`src` is the plate that was dropped, not whatever the snapshot said
    when it was written on another machine."""
    import spacr.utils as utils
    plate = tmp_path / "plate"
    _touch(plate / "settings" / "mask_settings.csv", "Key,Value\n")
    monkeypatch.setattr(utils, "load_settings",
                        lambda *a, **k: {"src": "/somewhere/else"})
    screen = _QueueScreen()
    dh.PlateQueueDropHandler().apply(plate, screen)
    assert screen.items == [("mask", {"src": str(plate)})]


def test_a_hand_made_snapshot_is_read_with_the_documented_default(tmp_path,
                                                                    monkeypatch):
    """spaCR writes two columns; a hand-made file is likely to use the
    single-argument spelling. Only the SECOND failure means unreadable."""
    import spacr.utils as utils
    plate = tmp_path / "plate"
    _touch(plate / "settings" / "measure_settings.csv", "a\n")
    calls = []

    def _load(path, **kwargs):
        calls.append(kwargs)
        if kwargs:
            raise ValueError("no Key column")
        return {"channels": [0]}

    monkeypatch.setattr(utils, "load_settings", _load)
    screen = _QueueScreen()
    dh.PlateQueueDropHandler().apply(plate, screen)
    assert len(calls) == 2
    assert screen.items[0][0] == "measure"


def test_one_unreadable_snapshot_among_several_is_named_rather_than_dropped(
        tmp_path, monkeypatch):
    """A partial drop used to report plain success, and the user found out
    when the run they expected was not in the list."""
    import spacr.utils as utils
    plate = tmp_path / "plate"
    _touch(plate / "settings" / "mask_settings.csv", "Key,Value\n")
    _touch(plate / "settings" / "broken_settings.csv", "\x00")

    def _load(path, **kwargs):
        if "broken" in str(path):
            raise ValueError("not a settings file")
        return {"channels": [0]}

    monkeypatch.setattr(utils, "load_settings", _load)
    screen = _QueueScreen()
    dh.PlateQueueDropHandler().apply(plate, screen)
    assert len(screen.items) == 1
    assert "Queued 1 of 2" in screen.log
    assert "broken_settings.csv" in screen.log


def test_a_snapshot_that_parses_to_something_that_is_not_settings_is_skipped(
        tmp_path, monkeypatch):
    import spacr.utils as utils
    plate = tmp_path / "plate"
    _touch(plate / "settings" / "mask_settings.csv", "Key,Value\n")
    monkeypatch.setattr(utils, "load_settings", lambda *a, **k: ["not", "a",
                                                                 "dict"])
    with pytest.raises(ValueError, match="No readable settings snapshots"):
        dh.PlateQueueDropHandler().apply(plate, _QueueScreen())


def test_the_plate_queue_says_what_it_accepts(tmp_path):
    assert "settings/*.csv" in dh.PlateQueueDropHandler().error_message(
        tmp_path)


# ---------------------------------------------------------------------------
# Batch Runner
# ---------------------------------------------------------------------------

def test_a_saved_queue_file_is_loaded_as_a_queue(tmp_path):
    saved = _touch(tmp_path / "queue.json", "[]")
    loaded = []
    screen = _Screen(load_queue_from=lambda p: loaded.append(p) or True)
    dh.BatchDropHandler().apply(saved, screen)
    assert loaded == [str(saved)]


def test_a_queue_file_that_would_not_load_reports_the_screens_reason(tmp_path):
    saved = _touch(tmp_path / "queue.yaml", "[]")
    screen = _Screen(load_queue_from=lambda p: False, last_error="bad YAML")
    with pytest.raises(ValueError, match="bad YAML"):
        dh.BatchDropHandler().apply(saved, screen)


def test_a_queue_file_that_would_not_load_still_names_the_path(tmp_path):
    saved = _touch(tmp_path / "queue.yml", "[]")
    screen = _Screen(load_queue_from=lambda p: False)
    with pytest.raises(ValueError, match=str(saved)):
        dh.BatchDropHandler().apply(saved, screen)


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


def test_a_folder_whose_snapshots_are_all_refused_is_reported(tmp_path):
    plate = tmp_path / "plate"
    _touch(plate / "settings" / "mask_settings.csv", "Key,Value\n")
    screen = _Screen(add_job=lambda module, settings: False)
    with pytest.raises(ValueError, match="No runnable settings jobs"):
        dh.BatchDropHandler().apply(plate, screen)


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

def test_plate_viewer_accepts_either_name_for_opening_a_database(tmp_path):
    """Two screens share this handler and they spell the opener
    differently; neither should need a handler of its own."""
    opened = []
    dh.ResultsDatabaseDropHandler().apply(
        tmp_path, _Screen(open_database=lambda p: opened.append(p)))
    assert opened == [str(tmp_path)]


def test_a_results_screen_that_cannot_open_a_database_says_so(tmp_path):
    with pytest.raises(TypeError, match="cannot open a database"):
        dh.ResultsDatabaseDropHandler().apply(tmp_path, _Screen())


def test_a_results_database_that_was_refused_reports_the_screens_reason(
        tmp_path):
    screen = _Screen(set_database=lambda p: False, last_error="locked")
    with pytest.raises(ValueError, match="locked"):
        dh.ResultsDatabaseDropHandler().apply(tmp_path, screen)


def test_a_results_database_refused_without_a_reason_names_the_path(tmp_path):
    screen = _Screen(set_database=lambda p: False)
    with pytest.raises(ValueError, match=str(tmp_path)):
        dh.ResultsDatabaseDropHandler().apply(tmp_path, screen)


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

def test_a_database_with_one_table_is_read_without_asking(tmp_path):
    db = _database(tmp_path / "measurements.db")
    seen = []
    screen = _Screen(load_path=lambda path, table=None: seen.append((path,
                                                                      table)))
    dh.TableDropHandler("tabulate").apply(db, screen)
    assert seen == [(str(db), "png_list")]


def test_a_database_with_several_tables_is_asked_about_rather_than_guessed(
        tmp_path, picks):
    """`load_path` takes the first one silently, which is fine for a file
    dialog where the user chose the file and wrong for a dropped folder."""
    db = _database(tmp_path / "measurements.db", ("cell", "nucleus", "png_list"))
    asked = picks(lambda options: options[1])
    seen = []
    screen = _Screen(load_path=lambda path, table=None: seen.append(table))
    dh.TableDropHandler("tabulate").apply(db, screen)
    assert seen == ["nucleus"]
    assert asked[0][2] == ["cell", "nucleus", "png_list"]
    assert "3 tables" in asked[0][0]


def test_cancelling_the_table_chooser_loads_nothing(tmp_path, picks):
    db = _database(tmp_path / "measurements.db", ("cell", "png_list"))
    picks(None)
    seen = []
    screen = _Screen(load_path=lambda path, table=None: seen.append(table))
    dh.TableDropHandler("tabulate").apply(db, screen)
    assert seen == []


def test_a_csv_dropped_on_a_table_screen_is_read_with_no_table_name(tmp_path):
    csv = _touch(tmp_path / "scores.csv", "a,b\n")
    seen = []
    screen = _Screen(load_path=lambda path, table=None: seen.append((path,
                                                                      table)))
    dh.TableDropHandler("tabulate").apply(csv, screen)
    assert seen == [(str(csv), None)]


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

def test_the_profiler_takes_the_only_coefficient_table_in_a_results_folder(
        tmp_path, resolves):
    folder = tmp_path / "results"
    coefficients = _touch(folder / "coefficients.csv", "gene,beta\n")
    resolves(_Resolution(targets=[_Target(str(folder), kind="regression-results")]))
    seen = []
    dh.CoefficientsDropHandler("profiler").apply(
        folder, _Screen(load_coefficients=seen.append))
    assert seen == [str(coefficients)]


def test_the_profiler_asks_which_table_holds_the_coefficients(tmp_path,
                                                                resolves, picks):
    folder = tmp_path / "results"
    _touch(folder / "a.csv", "gene,beta\n")
    wanted = _touch(folder / "b.csv", "gene,beta\n")
    resolves(_Resolution(targets=[_Target(str(folder), kind="regression-results")]))
    asked = picks(lambda options: str(wanted))
    seen = []
    dh.CoefficientsDropHandler("profiler").apply(
        folder, _Screen(load_coefficients=seen.append))
    assert seen == [str(wanted)]
    assert "holds the coefficients" in asked[0][1]


def test_cancelling_the_coefficient_chooser_loads_nothing(tmp_path, resolves,
                                                            picks):
    folder = tmp_path / "results"
    _touch(folder / "a.csv")
    _touch(folder / "b.csv")
    resolves(_Resolution(targets=[_Target(str(folder), kind="regression-results")]))
    picks(None)
    seen = []
    dh.CoefficientsDropHandler("profiler").apply(
        folder, _Screen(load_coefficients=seen.append))
    assert seen == []


def test_a_results_folder_with_no_coefficient_table_is_refused_by_name(
        tmp_path, resolves):
    folder = tmp_path / "results"
    folder.mkdir()
    resolves(_Resolution(targets=[_Target(str(folder), kind="regression-results")]))
    with pytest.raises(ValueError, match="No coefficient CSV"):
        dh.CoefficientsDropHandler("profiler").apply(
            folder, _Screen(load_coefficients=lambda p: None))


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

def test_a_masks_folder_with_one_mask_opens_it_without_asking(tmp_path,
                                                                resolves):
    masks = tmp_path / "masks"
    mask = _touch(masks / "cell.tif")
    resolves(_Resolution(targets=[_Target(str(masks), kind="masks")]))
    seen = []
    dh.LabelMaskDropHandler("curate").apply(
        tmp_path, _Screen(set_paths=lambda mask=None: seen.append(mask)))
    assert seen == [str(mask)]


def test_a_masks_folder_with_several_masks_asks_which_one(tmp_path, resolves,
                                                            picks):
    """A folder of masks is not one mask."""
    masks = tmp_path / "masks"
    _touch(masks / "a.tif")
    wanted = _touch(masks / "b.tif")
    resolves(_Resolution(targets=[_Target(str(masks), kind="masks")]))
    asked = picks(lambda options: str(wanted))
    seen = []
    dh.LabelMaskDropHandler("curate").apply(
        tmp_path, _Screen(set_paths=lambda mask=None: seen.append(mask)))
    assert seen == [str(wanted)]
    assert "2 masks" in asked[0][0]


def test_cancelling_the_mask_chooser_opens_nothing(tmp_path, resolves, picks):
    masks = tmp_path / "masks"
    _touch(masks / "a.tif")
    _touch(masks / "b.tif")
    resolves(_Resolution(targets=[_Target(str(masks), kind="masks")]))
    picks(None)
    seen = []
    dh.LabelMaskDropHandler("curate").apply(
        tmp_path, _Screen(set_paths=lambda mask=None: seen.append(mask)))
    assert seen == []


def test_a_masks_folder_holding_no_masks_is_refused_by_name(tmp_path,
                                                              resolves):
    masks = tmp_path / "masks"
    masks.mkdir()
    resolves(_Resolution(targets=[_Target(str(masks), kind="masks")]))
    with pytest.raises(ValueError, match="No label mask was found"):
        dh.LabelMaskDropHandler("curate").apply(
            tmp_path, _Screen(set_paths=lambda mask=None: None))


def test_a_screen_with_a_mask_field_gets_the_path_and_the_open(tmp_path,
                                                                 resolves):
    masks = tmp_path / "masks"
    mask = _touch(masks / "cell.npy")
    resolves(_Resolution(targets=[_Target(str(masks), kind="masks")]))
    field = QLineEdit()
    order = []
    screen = _Screen(_mask_edit=field,
                     open_mask=lambda: order.append(field.text()))
    dh.LabelMaskDropHandler("napari_bridge").apply(tmp_path, screen)
    assert order == [str(mask)]


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

def test_methods_and_results_files_a_checkpoint_a_results_folder_and_a_project(
        tmp_path, resolves):
    handler = dh.MethodsSourcesDropHandler("methods_export")
    assert handler.accepts_multiple() is True
    assert handler.can_accept(tmp_path) is True
    fields = {"model": QLineEdit(), "results": QLineEdit(),
              "project": QLineEdit()}
    screen = _Screen(_fields=fields)
    model = _touch(tmp_path / "best.pth")
    handler.apply(model, screen)
    assert fields["model"].text() == str(model)
    results = tmp_path / "results"
    results.mkdir()
    handler.apply(results, screen)
    assert fields["results"].text() == str(results)
    resolves(_Resolution(targets=[_Target(str(tmp_path))], root=str(tmp_path)))
    plate = tmp_path / "plate"
    plate.mkdir()
    handler.apply(plate, screen)
    assert fields["project"].text() == str(tmp_path)


def test_methods_and_results_falls_back_to_the_dropped_path_when_unresolved(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        dh._ch, "resolve_drop",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ports")))
    fields = {"project": QLineEdit()}
    dh.MethodsSourcesDropHandler("methods_export").apply(
        tmp_path, _Screen(_fields=fields))
    assert fields["project"].text() == str(tmp_path)


def test_methods_and_results_says_when_it_has_no_field_for_a_drop(tmp_path,
                                                                    resolves):
    resolves(_Resolution(targets=[_Target(str(tmp_path))], root=str(tmp_path)))
    with pytest.raises(TypeError, match="no field for this drop"):
        dh.MethodsSourcesDropHandler("methods_export").apply(
            tmp_path, _Screen(_fields={}))


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


def test_distributed_jobs_asks_which_snapshot_to_submit(tmp_path, picks):
    plate = tmp_path / "plate"
    _touch(plate / "settings" / "mask_settings.csv", "Key,Value\n")
    wanted = _touch(plate / "settings" / "measure_settings.csv", "Key,Value\n")
    asked = picks(lambda options: str(wanted))
    field = QLineEdit()
    dh.SubmissionSettingsDropHandler("distributed_jobs").apply(
        plate, _Screen(_settings_path=field))
    assert field.text() == str(wanted)
    assert "should be submitted" in asked[0][1]


def test_cancelling_the_snapshot_chooser_submits_nothing(tmp_path, picks):
    plate = tmp_path / "plate"
    _touch(plate / "settings" / "a_settings.csv")
    _touch(plate / "settings" / "b_settings.csv")
    picks(None)
    field = QLineEdit()
    dh.SubmissionSettingsDropHandler("distributed_jobs").apply(
        plate, _Screen(_settings_path=field))
    assert field.text() == ""


def test_distributed_jobs_says_what_a_settings_snapshot_looks_like(tmp_path):
    handler = dh.SubmissionSettingsDropHandler("distributed_jobs")
    assert "settings/*.csv" in handler.error_message(tmp_path)
    empty = tmp_path / "plate"
    empty.mkdir()
    with pytest.raises(ValueError, match="No settings snapshot"):
        handler.apply(empty, _Screen(_settings_path=QLineEdit()))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

def test_a_plugin_can_supply_its_own_drop_handler(monkeypatch):
    import spacr.plugins as plugins

    class _PluginHandler(dh.DropHandler):
        def can_accept(self, path):
            return True

        def error_message(self, path):
            return ""

        def apply(self, path, screen):
            pass

    monkeypatch.setattr(plugins, "get_app",
                        lambda key: type("A", (), {"drop_handler": "x:y"})())
    monkeypatch.setattr(plugins, "load_object", lambda ref: _PluginHandler)
    assert isinstance(dh.get_handler("my_plugin"), _PluginHandler)


def test_a_plugin_offering_something_that_is_not_a_drop_handler_is_recorded(
        monkeypatch):
    """A plugin that names the wrong class must not take the screen down,
    and must not be silent either -- the diagnostic is how the author finds
    out."""
    import spacr.plugins as plugins
    monkeypatch.setattr(plugins, "get_app",
                        lambda key: type("A", (), {"drop_handler": "x:y"})())
    monkeypatch.setattr(plugins, "load_object", lambda ref: dict)
    recorded = []
    monkeypatch.setattr(plugins, "record_diagnostic",
                        lambda key, message, exc: recorded.append((key, message)))
    handler = dh.get_handler("my_plugin")
    assert isinstance(handler, dh.SourceDropHandler)
    assert recorded and recorded[0][0] == "my_plugin"


def test_a_diagnostic_that_cannot_be_recorded_still_leaves_a_usable_handler(
        monkeypatch):
    import spacr.plugins as plugins
    monkeypatch.setattr(
        plugins, "get_app",
        lambda key: (_ for _ in ()).throw(RuntimeError("plugin registry down")))
    monkeypatch.setattr(
        plugins, "record_diagnostic",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("also down")))
    assert isinstance(dh.get_handler("my_plugin"), dh.SourceDropHandler)


def test_a_plugin_supplying_a_layout_handler_is_told_which_module_it_is_on(
        monkeypatch):
    """Layout handlers resolve against the module they are installed on, so
    a plugin's one has to be constructed with its key like the built-ins."""
    import spacr.plugins as plugins

    class _PluginLayout(dh.LayoutDropHandler):
        def deliver(self, screen, value, target):
            pass

    monkeypatch.setattr(plugins, "get_app",
                        lambda key: type("A", (), {"drop_handler": "x:y"})())
    monkeypatch.setattr(plugins, "load_object", lambda ref: _PluginLayout)
    handler = dh.get_handler("my_plugin")
    assert isinstance(handler, _PluginLayout)
    assert handler.app_key == "my_plugin"
