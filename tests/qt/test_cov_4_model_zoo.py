"""The Model Zoo refuses inline and never hides a failure behind a dialog.

Every path here is a refusal or a teardown. A catalogue entry with no download
URI, a second job started while one is in flight, a benchmark whose masks were
not kept, a worker that failed with nobody registered to hear it -- each has to
land in the status label, because a modal dialog in the middle of a scan is a
dialog nobody sees and a silent failure is a screen that looks like it worked.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import model_zoo as zoo
from spacr.qt.screens import model_zoo as mz
from spacr.qt.screens.model_zoo import ModelZooScreen


@pytest.fixture(autouse=True)
def _isolated_run_journal(monkeypatch, tmp_path):
    """Keep this screen's manifests out of the user's real run history."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


@pytest.fixture
def screen(qtbot):
    """A synchronous screen — jobs run inline so assertions are exact."""
    widget = ModelZooScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _folder_picker(monkeypatch, path):
    """Replace the native folder dialog with one that answers ``path``."""
    asked = []

    class _Dialog:
        @staticmethod
        def getExistingDirectory(_parent, title, start):
            asked.append((title, start))
            return path

    monkeypatch.setattr(mz, "QFileDialog", _Dialog)
    return asked


# -- typing and picking folders ----------------------------------------------

def test_a_typed_scan_folder_is_scanned(screen, tmp_path):
    """The path box is a second way in, not decoration."""
    screen._scan_edit.setText(str(tmp_path / "no_such_folder"))
    screen._on_scan_typed()
    assert "No such folder" in screen.last_error


def test_picking_a_scan_folder_scans_it(screen, tmp_path, monkeypatch):
    """The browse button and the path box must end in the same place."""
    asked = _folder_picker(monkeypatch, str(tmp_path))
    screen._pick_scan_folder()
    assert asked, "the folder dialog was never opened"
    assert screen._scan_edit.text() == str(tmp_path)


def test_cancelling_the_scan_picker_scans_nothing(screen, monkeypatch):
    """An empty answer is a cancel, and a cancel must not rescan."""
    screen._scan_edit.setText("/data/models")
    _folder_picker(monkeypatch, "")
    screen._pick_scan_folder()
    assert screen._scan_edit.text() == "/data/models"


def test_picking_a_download_folder_fills_the_box(screen, tmp_path,
                                                 monkeypatch):
    """The destination is where bytes will land; it has to be visible."""
    _folder_picker(monkeypatch, str(tmp_path))
    screen._pick_dest_folder()
    assert screen._dest_edit.text() == str(tmp_path)


def test_cancelling_the_download_picker_leaves_the_box_alone(screen,
                                                             monkeypatch):
    """A cancelled dialog must not blank a destination already chosen."""
    screen._dest_edit.setText("/data/models")
    _folder_picker(monkeypatch, "")
    screen._pick_dest_folder()
    assert screen._dest_edit.text() == "/data/models"


def test_a_typed_fields_folder_is_loaded(screen, tmp_path):
    """The benchmark's field box is the same second way in."""
    folder = tmp_path / "plate1"
    folder.mkdir()
    np.save(folder / "A01_f00.npy", np.full((32, 32), 100.0, dtype=np.float32))
    screen._fields_edit.setText(str(folder))
    screen._on_fields_typed()
    assert screen.fields_folder() == str(folder)


def test_picking_a_fields_folder_loads_it(screen, tmp_path, monkeypatch):
    """Browsing for fields must load them, not merely remember the path."""
    folder = tmp_path / "plate2"
    folder.mkdir()
    np.save(folder / "A01_f00.npy", np.full((32, 32), 100.0, dtype=np.float32))
    _folder_picker(monkeypatch, str(folder))
    screen._pick_fields_folder()
    assert screen.fields_folder() == str(folder)
    assert screen.field_names()


def test_cancelling_the_fields_picker_loads_nothing(screen, monkeypatch):
    """A cancel must not clear the fields already loaded."""
    _folder_picker(monkeypatch, "")
    screen._pick_fields_folder()
    assert screen.fields_folder() == ""


# -- refusals ----------------------------------------------------------------

def test_a_catalogue_entry_with_no_uri_cannot_be_downloaded(screen):
    """A local listing has nowhere to fetch from, and says which entry."""
    screen.set_entries([zoo.ModelEntry(key="local", name="local.CP_model",
                                       source="scan", uri="")])
    screen._table.selectRow(0)
    assert screen.download_selected() is False
    assert "no download URI" in screen.last_error
    assert "local.CP_model" in screen.last_error


def test_a_second_field_load_while_one_is_running_is_refused(screen,
                                                             tmp_path):
    """Two loads at once would interleave two folders into one field list."""
    screen._busy = True
    assert screen.set_fields_source(str(tmp_path)) is False
    assert "already running" in screen.last_error
    assert screen.fields_folder() == ""


# -- progress ----------------------------------------------------------------

def test_a_download_of_unknown_length_shows_an_indeterminate_bar(screen):
    """A server that sends no length must not report a fake percentage."""
    screen._on_progress(4096, 0)
    assert (screen._progress.minimum(), screen._progress.maximum()) == (0, 0)
    assert screen._progress.format() == zoo._human_bytes(4096)


# -- the benchmark table and preview -----------------------------------------

def _result(entry, rows, masks=(), images=()):
    return zoo.BenchmarkResult(entry=entry, fieldset="fs", fieldset_label="3",
                               rows=list(rows), masks=list(masks),
                               images=list(images))


def test_a_failing_field_is_coloured_as_an_error(screen):
    """A failed quality check must not read like a passing one."""
    from spacr.qt.theme import active_palette

    entry = zoo.ModelEntry(key="k", name="m.CP_model")
    rows = [zoo.FieldBenchmark(field="A01_f00", n_objects=0, severity="fail",
                               flags=("no objects",), note="nothing found"),
            zoo.FieldBenchmark(field="A01_f01", n_objects=9, severity="warn",
                               flags=("few objects",), note="thin")]
    screen._apply_benchmark(_result(entry, rows))
    failing = screen._bench_table.item(0, 2)
    warning = screen._bench_table.item(1, 2)
    assert failing.foreground().color().name() == \
        active_palette()["error"].lower()
    assert warning.foreground().color().name() != \
        failing.foreground().color().name()


def test_a_run_that_kept_no_masks_says_so_rather_than_drawing_nothing(screen):
    """A blank preview is indistinguishable from a preview that failed."""
    entry = zoo.ModelEntry(key="k", name="m.CP_model")
    rows = [zoo.FieldBenchmark(field="A01_f00", n_objects=3, severity="ok")]
    screen._apply_benchmark(_result(entry, rows))
    assert screen.select_field(0) is False
    assert "not kept" in screen._preview.text()


def test_a_field_with_nothing_to_draw_says_so(screen, monkeypatch):
    """An unusable image must not leave the previous field on screen."""
    monkeypatch.setattr(mz, "compose_labels", lambda _image, _mask: None)
    entry = zoo.ModelEntry(key="k", name="m.CP_model")
    rows = [zoo.FieldBenchmark(field="A01_f00", n_objects=3, severity="ok")]
    mask = np.zeros((8, 8), dtype=np.int32)
    screen._apply_benchmark(_result(entry, rows, masks=[mask],
                                    images=[np.zeros((8, 8))]))
    assert screen.select_field(0) is False
    assert "Nothing to draw" in screen._preview.text()
    assert screen.preview_size() == (0, 0)


# -- job plumbing ------------------------------------------------------------

def test_the_job_body_leaves_its_result_where_the_gui_thread_looks(
        qtbot, tmp_path, monkeypatch):
    """The worker and the GUI thread meet at one dict, and nowhere else."""
    real_make_thread = mz.make_thread
    captured = {}

    def _capture(fn, settings, *args, **kwargs):
        thread, worker = real_make_thread(fn, settings, *args, **kwargs)
        thread.start = lambda: None
        captured["fn"] = fn
        captured["payload"] = settings
        return thread, worker

    monkeypatch.setattr(mz, "make_thread", _capture)
    widget = ModelZooScreen(threaded=True)
    qtbot.addWidget(widget)
    assert widget.scan(str(tmp_path)) is True

    captured["fn"](captured["payload"])
    assert "result" in captured["payload"]


def test_a_result_that_cannot_be_applied_lands_in_the_status_label(screen):
    """Raising on the GUI thread here would take the window with it."""
    def _refuses(_result):
        raise ValueError("the listing did not parse")

    screen._pending.append(({"result": None}, _refuses, None))
    settled = []
    screen.job_finished.connect(settled.append)
    screen._on_job_settled(True)
    assert settled == [False]
    assert "the listing did not parse" in screen.last_error


def test_a_failure_with_nobody_listening_still_reaches_the_status_label(
        screen):
    """A worker traceback with no handler must not vanish."""
    screen._pending.clear()
    screen._report_failure(ValueError("the checksum did not match"))
    assert "the checksum did not match" in screen.last_error
    assert screen._busy is False


# -- teardown ----------------------------------------------------------------

def test_closing_waits_out_a_thread_whose_wrapper_outlived_it(screen, qtbot):
    """A QThread collected while running aborts the process; a dead one must
    not stop the widget closing either."""
    from PySide6.QtGui import QCloseEvent

    class _AlreadyDeleted:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted")

    screen._jobs.append((_AlreadyDeleted(), object()))
    screen.closeEvent(QCloseEvent())
    assert screen._cancel["stop"] is True


def test_closing_waits_for_a_job_that_is_still_running(screen):
    """A QThread collected mid-run takes the process down, so the widget
    waits for it instead of dropping its references and hoping."""
    from PySide6.QtGui import QCloseEvent

    class _StillRunning:
        def __init__(self):
            self.quit_calls = 0
            self.waited = None

        def isRunning(self):
            return True

        def quit(self):
            self.quit_calls += 1

        def wait(self, ms):
            self.waited = ms
            return True

    thread = _StillRunning()
    screen._jobs.append((thread, object()))
    screen.closeEvent(QCloseEvent())
    assert thread.quit_calls == 1
    assert thread.waited == 5000
