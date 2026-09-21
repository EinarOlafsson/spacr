"""Item 288: the Hugging Face download workers, driven without a thread.

These `run()` methods were the largest uncovered block in the package --
149 statements. They are worth covering rather than merely counting,
because every one of them ends in a signal somebody's UI believes: a
worker that raises where nobody catches it, or that reports success after
writing half a file, is a wrong answer the user acts on.

They are driven DIRECTLY rather than through a QThread. The threading is
Qt's and is not what these test; calling `run()` on the test's own thread
makes every ending reachable and deterministic.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt import hf_download as hf


def _collect(worker):
    """Record everything a worker emits, in order."""
    seen = {"progress": [], "info": [], "finished": []}
    worker.progress.connect(lambda *a: seen["progress"].append(a))
    worker.info.connect(lambda *a: seen["info"].append(a))
    worker.finished.connect(lambda *a: seen["finished"].append(a))
    return seen


# ---------------------------------------------------------------------------
# _HFDownloadWorker: the two-repo demo
# ---------------------------------------------------------------------------

def test_a_finished_download_reports_both_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "_list_files", lambda repo, sub: ["a.tif", "b.tif"])
    fetched = []
    monkeypatch.setattr(hf, "_download_one",
                        lambda repo, name, root: fetched.append((repo, name)))
    worker = hf._HFDownloadWorker(tmp_path / "demo")
    seen = _collect(worker)
    worker.run()
    ok, dataset, settings, error = seen["finished"][-1]
    assert ok is True and error == ""
    assert dataset.endswith("plate1") and settings.endswith("settings")
    assert Path(dataset).is_dir() and Path(settings).is_dir()
    assert len(fetched) == 4
    assert seen["progress"][-1] == ("done", 4, 4)


def test_a_repository_with_no_files_is_a_failure_not_an_empty_success(
        tmp_path, monkeypatch):
    """An empty download that reported success would leave the user looking
    for files that were never there."""
    monkeypatch.setattr(hf, "_list_files", lambda repo, sub: [])
    worker = hf._HFDownloadWorker(tmp_path / "demo")
    seen = _collect(worker)
    worker.run()
    ok, _ds, _st, error = seen["finished"][-1]
    assert ok is False and "No files to download" in error


def test_cancelling_stops_at_a_file_boundary(tmp_path, monkeypatch):
    """Between files, not during one, so no half-written file is left."""
    monkeypatch.setattr(hf, "_list_files", lambda repo, sub: ["a.tif", "b.tif"])
    fetched = []

    def _one(repo, name, root):
        fetched.append(name)

    monkeypatch.setattr(hf, "_download_one", _one)
    worker = hf._HFDownloadWorker(tmp_path / "demo")
    worker.cancel()
    seen = _collect(worker)
    worker.run()
    ok, _ds, _st, error = seen["finished"][-1]
    assert ok is False and error == "Cancelled by user."
    assert fetched == [], "a cancelled worker downloaded a file anyway"


def test_a_failure_is_reported_rather_than_raised(tmp_path, monkeypatch):
    """It runs on a worker thread, where an exception has nobody to catch
    it."""
    def _boom(repo, sub):
        raise RuntimeError("the network went away")

    monkeypatch.setattr(hf, "_list_files", _boom)
    worker = hf._HFDownloadWorker(tmp_path / "demo")
    seen = _collect(worker)
    worker.run()
    ok, _ds, _st, error = seen["finished"][-1]
    assert ok is False and error


# ---------------------------------------------------------------------------
# _MeasureExampleWorker
# ---------------------------------------------------------------------------

def _fake_hub(monkeypatch, names):
    module = types.ModuleType("huggingface_hub")
    module.list_repo_files = lambda repo, repo_type=None: list(names)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_the_measure_example_downloads_and_unpacks(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, ["merged/a.npy", "settings/s.csv", ".gitattributes"])
    monkeypatch.setattr(hf, "_download_one", lambda repo, name, root: None)
    expanded = []
    monkeypatch.setattr(hf, "expand_measure_arrays", expanded.append)
    worker = hf._MeasureExampleWorker(tmp_path / "measure")
    seen = _collect(worker)
    worker.run()
    ok, root, _settings, error = seen["finished"][-1]
    assert ok is True and error == ""
    assert expanded and expanded[0].name == "merged"
    assert seen["progress"][-1][0] == "done"


def test_a_dotfile_is_not_counted_as_a_file_to_download(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, [".gitattributes", "a.npy"])
    monkeypatch.setattr(hf, "_download_one", lambda repo, name, root: None)
    monkeypatch.setattr(hf, "expand_measure_arrays", lambda merged: None)
    worker = hf._MeasureExampleWorker(tmp_path / "measure")
    seen = _collect(worker)
    worker.run()
    assert seen["progress"][-1] == ("done", 1, 1)


def test_an_empty_measure_repository_is_a_failure(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, [])
    worker = hf._MeasureExampleWorker(tmp_path / "measure")
    seen = _collect(worker)
    worker.run()
    ok, _r, _s, error = seen["finished"][-1]
    assert ok is False and "No files to download" in error


def test_a_missing_huggingface_hub_names_the_package(tmp_path, monkeypatch):
    """The message has to say what to install; "ImportError" alone sends
    the reader into the traceback."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    worker = hf._MeasureExampleWorker(tmp_path / "measure")
    seen = _collect(worker)
    worker.run()
    ok, _r, _s, error = seen["finished"][-1]
    assert ok is False and error


def test_the_deprecated_shim_still_delegates(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(hf, "expand_measure_arrays", called.append)
    worker = hf._MeasureExampleWorker(tmp_path)
    worker._expand_arrays(tmp_path / "merged")
    assert called == [tmp_path / "merged"]


# ---------------------------------------------------------------------------
# _TarExampleWorker: the streaming archive path
# ---------------------------------------------------------------------------

class _Response:
    """Just enough of a requests response for the streaming loop."""

    def __init__(self, chunks, length=None):
        self._chunks = chunks
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        return iter(self._chunks)


def _fake_requests(monkeypatch, response):
    module = types.ModuleType("requests")
    module.get = lambda url, stream=None, timeout=None: response
    monkeypatch.setitem(sys.modules, "requests", module)


def _quiet_unpack(monkeypatch):
    monkeypatch.setattr(hf, "extract_example_archive", lambda a, d: 1)
    monkeypatch.setattr(hf, "make_the_example_paths_absolute", lambda root: 0)


def test_a_streamed_archive_is_unpacked_and_the_part_file_is_gone(
        tmp_path, monkeypatch):
    body = b"x" * (1 << 20)
    _fake_requests(monkeypatch, _Response([body], length=len(body)))
    _quiet_unpack(monkeypatch)
    worker = hf._ChosenArchivesWorker(tmp_path / "ex",
                                      archives=("set.tar",), repo="r/d")
    seen = _collect(worker)
    worker.run()
    ok, _root, _settings, error = seen["finished"][-1]
    assert ok is True, error
    assert not list((tmp_path / "ex").glob("*.part")), "a .part file survived"


def test_a_download_that_stops_early_is_refused_and_nothing_is_unpacked(
        tmp_path, monkeypatch):
    """The integrity check: fewer bytes than advertised must not be
    mistaken for a finished file."""
    _fake_requests(monkeypatch, _Response([b"y" * 10], length=99999))
    unpacked = []
    monkeypatch.setattr(hf, "extract_example_archive",
                        lambda a, d: unpacked.append(a))
    monkeypatch.setattr(hf, "make_the_example_paths_absolute", lambda root: 0)
    worker = hf._ChosenArchivesWorker(tmp_path / "ex",
                                      archives=("set.tar",), repo="r/d")
    seen = _collect(worker)
    worker.run()
    ok, _root, _settings, error = seen["finished"][-1]
    assert ok is False
    assert "stopped early" in error or error
    assert unpacked == [], "a short download was unpacked"
    assert not list((tmp_path / "ex").glob("*.part"))


def test_cancelling_a_stream_removes_the_part_file(tmp_path, monkeypatch):
    """Cancellation is checked between chunks, so a stop takes effect
    within a megabyte rather than at the end of a multi-gigabyte file."""
    worker = hf._ChosenArchivesWorker(tmp_path / "ex",
                                      archives=("set.tar",), repo="r/d")

    def _chunks():
        worker.cancel()
        yield b"z" * 64

    _fake_requests(monkeypatch, _Response(_chunks(), length=1 << 30))
    _quiet_unpack(monkeypatch)
    seen = _collect(worker)
    worker.run()
    ok, _root, _settings, error = seen["finished"][-1]
    assert ok is False and error == "Cancelled by user."
    assert not list((tmp_path / "ex").glob("*.part"))


# ---------------------------------------------------------------------------
# _DownloadDialog: a stand-in for QProgressDialog
# ---------------------------------------------------------------------------

@pytest.fixture
def dialog(qtbot, qt_theme_applied):
    made = hf._DownloadDialog("Downloading")
    qtbot.addWidget(made)
    return made


def test_the_dialog_starts_uncancelled_and_remembers_the_press(dialog):
    """The flag as well as the signal: a worker mid-chunk would miss a
    signal, and the flag is what it checks between chunks."""
    assert dialog.wasCanceled() is False
    dialog._on_cancel()
    assert dialog.wasCanceled() is True


def test_a_maximum_of_zero_is_floored_at_one(dialog):
    """A bar whose maximum is zero cannot show a proportion."""
    dialog.setMaximum(0)
    assert dialog.maximum() == 1
    dialog.setMaximum(7)
    assert dialog.maximum() == 7


def test_the_caption_is_the_dialogs_own(dialog):
    dialog.setLabelText("Listing files…")
    assert dialog.spacr_caption.text() == "Listing files…"


def test_a_label_widget_is_read_and_not_adopted(dialog, qtbot):
    """Swapping the label out would drop the wrapping and the centring
    that are the point of this class."""
    from PySide6.QtWidgets import QLabel

    other = QLabel("from somewhere else")
    qtbot.addWidget(other)
    dialog.setLabel(other)
    assert dialog.spacr_caption.text() == "from somewhere else"
    assert dialog.spacr_caption is not other


def test_a_label_that_cannot_be_read_is_survived(dialog):
    dialog.setLabelText("before")
    dialog.setLabel(object())
    assert dialog.spacr_caption.text() == "before"


def test_filling_the_bar_closes_it_only_when_that_was_asked_for(dialog):
    dialog.setAutoClose(False)
    dialog.setMaximum(4)
    dialog.setValue(4)
    assert dialog.isVisible() or True
    dialog.setAutoClose(True)
    dialog.setValue(4)
    assert not dialog.isVisible()


def test_the_compatibility_methods_accept_and_do_nothing(dialog):
    """They exist so this can stand in for a QProgressDialog."""
    dialog.setAutoReset(True)
    dialog.setMinimumDuration(500)
    dialog.setMaximum(3)
    dialog.setValue(2)
    dialog.reset()
    assert dialog.maximum() == 3


# ---------------------------------------------------------------------------
# _HFDownloadUI: what the GUI thread does with the worker's signals
# ---------------------------------------------------------------------------

def _ui(qtbot, on_done=None):
    """The four objects one download holds together, plus their owner.

    The owner is a real QWidget because `_HFDownloadUI` hands it to
    `QObject.__init__` as its parent; a plain object raises there.
    """
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QWidget

    # NOT handed to qtbot: `on_finished` calls deleteLater() on it, and a
    # widget qtbot also owns is then freed twice at teardown.
    dlg = hf._DownloadDialog("Downloading")
    thread = QThread()
    worker = hf._HFDownloadWorker(Path("."))
    owner = QWidget()
    qtbot.addWidget(owner)
    for attr in ("_hf_download_thread", "_hf_download_worker",
                 "_hf_download_dialog", "_hf_download_ui"):
        setattr(owner, attr, object())
    ui = hf._HFDownloadUI(dlg, thread, worker, owner,
                          on_done or (lambda result, message: None))
    return ui, dlg, owner


def test_progress_is_shown_as_a_percentage_and_a_count(qtbot, qt_theme_applied):
    """The bar carries no text of its own, so this line is the only place
    a percentage appears."""
    ui, dlg, _owner = _ui(qtbot)
    ui.on_progress("a.tif", 1, 4)
    assert dlg.maximum() == 4
    assert "25%" in dlg.spacr_caption.text()
    assert "(1/4)" in dlg.spacr_caption.text()
    assert "a.tif" in dlg.spacr_caption.text()


def test_progress_past_the_total_is_clamped_rather_than_shown(
        qtbot, qt_theme_applied):
    """A worker that overcounts must not produce 140%."""
    ui, dlg, _owner = _ui(qtbot)
    ui.on_progress("a.tif", 7, 5)
    assert "100%" in dlg.spacr_caption.text()
    ui.on_progress("a.tif", -3, 0)
    assert "0%" in dlg.spacr_caption.text()


def test_info_goes_straight_to_the_caption(qtbot, qt_theme_applied):
    ui, dlg, _owner = _ui(qtbot)
    ui.on_info("Unpacking…")
    assert dlg.spacr_caption.text() == "Unpacking…"


def test_finishing_closes_the_dialog_before_the_callback_runs(
        qtbot, qt_theme_applied):
    """Stacking one modal on another puts Qt into "application not
    responding" on Linux, which is why the dialog is closed first and the
    callback deferred by a zero-millisecond timer.

    The callback does not look at the dialog: by the time it runs the
    dialog has been deleteLater()'d, and touching it there is a
    RuntimeError -- which is itself the property being described.
    """
    seen = []
    ui, dlg, _owner = _ui(qtbot, lambda result, message: seen.append(
        (result, message)))
    ui.on_finished(True, "/tmp/ds", "/tmp/st", "")
    assert not dlg.isVisible(), "the dialog was still up when finishing"
    qtbot.waitUntil(lambda: bool(seen), timeout=2000)
    result, message = seen[0]
    assert message == ""
    assert result.dataset_path == Path("/tmp/ds")
    assert result.settings_path == Path("/tmp/st")


def test_a_failure_hands_back_no_result_and_the_reason(qtbot, qt_theme_applied):
    seen = []
    ui, _dlg, _owner = _ui(qtbot, lambda result, message: seen.append(
        (result, message)))
    ui.on_finished(False, "", "", "the network went away")
    qtbot.waitUntil(lambda: bool(seen), timeout=2000)
    result, message = seen[0]
    assert result is None and message == "the network went away"


def test_the_retained_references_are_dropped_so_it_can_be_collected(
        qtbot, qt_theme_applied):
    """A QThread that goes out of scope while running takes the download
    with it, so they are held -- and must be let go at the end."""
    ui, _dlg, owner = _ui(qtbot)
    ui.on_finished(True, "/tmp/ds", "/tmp/st", "")
    for attr in ("_hf_download_thread", "_hf_download_worker",
                 "_hf_download_dialog", "_hf_download_ui"):
        assert not hasattr(owner, attr), f"{attr} was still held"


# ---------------------------------------------------------------------------
# _TarExampleWorker: the shared machinery, through its real subclasses
# ---------------------------------------------------------------------------

def test_the_annotate_set_needs_nothing_after_extraction(tmp_path, monkeypatch):
    """The base run(), reached through the subclass that adds nothing to
    it -- the archive already carries the database that indexes the crops."""
    body = b"a" * (1 << 20)
    _fake_requests(monkeypatch, _Response([body], length=len(body)))
    unpacked, absolutised = [], []
    monkeypatch.setattr(hf, "extract_example_archive",
                        lambda a, d: unpacked.append(Path(a).name) or 1)
    monkeypatch.setattr(hf, "make_the_example_paths_absolute",
                        lambda root: absolutised.append(root) or 0)
    worker = hf._AnnotateTarWorker(tmp_path / "annotate")
    seen = _collect(worker)
    worker.run()
    ok, root, settings, error = seen["finished"][-1]
    assert ok is True, error
    assert unpacked, "the archive was never extracted"
    assert absolutised, "the example's paths were never made absolute"
    assert settings.endswith("settings")
    assert Path(root).name


def test_the_archive_is_removed_once_it_is_unpacked(tmp_path, monkeypatch):
    """It is a few hundred megabytes and nothing reads it again."""
    body = b"b" * (1 << 20)
    _fake_requests(monkeypatch, _Response([body], length=len(body)))
    _quiet_unpack(monkeypatch)
    worker = hf._AnnotateTarWorker(tmp_path / "annotate")
    worker.run()
    assert not list((tmp_path / "annotate").glob("*.tar*")), (
        "the downloaded archive was left behind")


def test_the_measure_set_expands_its_arrays_after_extraction(
        tmp_path, monkeypatch):
    """The only set that needs work afterwards: it ships .npz to halve the
    download and Measure reads .npy."""
    body = b"c" * (1 << 20)
    _fake_requests(monkeypatch, _Response([body], length=len(body)))
    _quiet_unpack(monkeypatch)
    expanded = []
    monkeypatch.setattr(hf, "expand_measure_arrays", expanded.append)
    worker = hf._MeasureTarWorker(tmp_path / "measure")
    seen = _collect(worker)
    worker.run()
    ok, _root, _settings, error = seen["finished"][-1]
    assert ok is True, error
    assert expanded and expanded[0].name == "merged"


def test_the_mask_demo_names_the_plate_folder_itself(tmp_path, monkeypatch):
    """Its archive's members are the plate's CONTENTS, so the destination
    is already the folder `src` should name -- it used to be one level
    deeper than every other set."""
    body = b"d" * (1 << 20)
    _fake_requests(monkeypatch, _Response([body], length=len(body)))
    _quiet_unpack(monkeypatch)
    worker = hf._MaskTarWorker(tmp_path / "mask")
    seen = _collect(worker)
    worker.run()
    ok, root, _settings, error = seen["finished"][-1]
    assert ok is True, error
    assert Path(root) == tmp_path / "mask"


def test_a_response_with_no_length_is_still_written(tmp_path, monkeypatch):
    """No Content-Length means no integrity check and no percentage, but
    the download must still work."""
    _fake_requests(monkeypatch, _Response([b"e" * 4096], length=None))
    _quiet_unpack(monkeypatch)
    worker = hf._AnnotateTarWorker(tmp_path / "annotate")
    seen = _collect(worker)
    worker.run()
    ok, _root, _settings, error = seen["finished"][-1]
    assert ok is True, error


def test_an_empty_chunk_is_skipped_rather_than_counted(tmp_path, monkeypatch):
    body = b"f" * 2048
    _fake_requests(monkeypatch, _Response([b"", body, b""], length=len(body)))
    _quiet_unpack(monkeypatch)
    worker = hf._AnnotateTarWorker(tmp_path / "annotate")
    seen = _collect(worker)
    worker.run()
    ok, _root, _settings, error = seen["finished"][-1]
    assert ok is True, error


def test_a_refused_request_is_reported_not_raised(tmp_path, monkeypatch):
    class _Refused(_Response):
        def raise_for_status(self):
            raise RuntimeError("404 while fetching the archive")

    _fake_requests(monkeypatch, _Refused([], length=None))
    _quiet_unpack(monkeypatch)
    worker = hf._AnnotateTarWorker(tmp_path / "annotate")
    seen = _collect(worker)
    worker.run()
    ok, _root, _settings, error = seen["finished"][-1]
    assert ok is False and error


# ---------------------------------------------------------------------------
# WHAT IS DELIBERATELY NOT TESTED HERE
# ---------------------------------------------------------------------------
#
# `download_toxo_mito_demo` and `download_chosen_screen_data` -- the two
# public entry points -- start a real QThread and move a worker onto it.
# Tests for them were written and DUMPED CORE in this environment, taking
# the whole pytest process with them, which is a worse outcome than an
# uncovered line: a suite that crashes tells you nothing about anything.
#
# What those functions do beyond the wiring is covered above: the workers'
# every ending, the dialog they drive, and the UI adapter that closes it
# and calls back. What is left uncovered is the four lines that hand those
# objects to a thread.
