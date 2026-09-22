"""Sequence downloads preserve partial results and leave failures retryable."""
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDialog

from spacr.sra import RunFile
from spacr.qt.widgets import sra_picker as mod


FILES = (
    RunFile(run="SRR1", library="plate_one", url="https://example.invalid/one.fastq.gz",
            mate=1, size_bytes=2_000_000, read_count=50_000),
    RunFile(run="SRR2", library="plate_two", url="https://example.invalid/two.fastq.gz",
            mate=1, size_bytes=3_000_000, read_count=60_000),
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(mod, "fetch_reads", Mock(side_effect=AssertionError("real download attempted")))
    monkeypatch.setattr(mod, "runs_for", Mock(side_effect=AssertionError("real listing attempted")))


def test_worker_snapshots_selection_and_reports_each_run_progress(tmp_path, monkeypatch):
    files, calls, progress, endings = list(FILES), [], [], []
    worker = mod._FetchWorker(files, tmp_path, 1000)
    files.clear()  # Later changes in the picker cannot alter an in-flight job.

    def fetch(one, destination, *, max_reads, should_stop, progress):
        calls.append(one)
        assert destination == tmp_path and max_reads == 1000
        assert not should_stop()
        progress(1000, 24000)
        return tmp_path / f"{one.run}.fastq.gz"

    monkeypatch.setattr(mod, "fetch_reads", fetch)
    worker.progress.connect(lambda *args: progress.append(args))
    worker.finished_all.connect(lambda *args: endings.append(args))
    worker.run()
    assert calls == list(FILES)
    assert progress == [("SRR1", 1000, 24000), ("SRR2", 1000, 24000)]
    assert endings == [([str(tmp_path / "SRR1.fastq.gz"), str(tmp_path / "SRR2.fastq.gz")], "")]
    assert not worker.isRunning()


@pytest.mark.parametrize("error,expected", [
    (InterruptedError("cancel requested"), "cancelled"),
    (OSError("connection lost"), "connection lost"),
])
def test_worker_retains_completed_files_when_a_later_run_stops(
        tmp_path, monkeypatch, error, expected):
    first = tmp_path / "SRR1.fastq.gz"
    fetch = Mock(side_effect=[first, error])
    monkeypatch.setattr(mod, "fetch_reads", fetch)
    worker = mod._FetchWorker(FILES, tmp_path, None)
    endings = []
    worker.finished_all.connect(lambda *args: endings.append(args))
    worker.run()
    assert endings == [([str(first)], expected)]
    assert fetch.call_count == 2
    assert all(call.kwargs['max_reads'] is None for call in fetch.call_args_list)


@pytest.mark.parametrize("stop_before_start", [False, True])
def test_worker_cancellation_prevents_the_next_file(tmp_path, monkeypatch, stop_before_start):
    worker = mod._FetchWorker(FILES, tmp_path, 1000)
    calls, endings = [], []

    def fetch(one, destination, *, max_reads, should_stop, progress):
        calls.append(one)
        assert not should_stop()
        worker.cancel()
        assert should_stop()
        return tmp_path / "first.fastq.gz"

    monkeypatch.setattr(mod, "fetch_reads", fetch)
    worker.finished_all.connect(lambda *args: endings.append(args))
    if stop_before_start:
        worker.cancel()
    worker.run()
    assert calls == ([] if stop_before_start else [FILES[0]])
    assert len(endings) == 1
    assert endings[0][0] == ([] if stop_before_start else [str(tmp_path / "first.fastq.gz")])


@pytest.fixture
def controlled_picker(qtbot, tmp_path, monkeypatch):
    jobs = []

    class Job(QObject):
        progress = Signal(str, int, int)
        finished_all = Signal(list, str)

        def __init__(self, files, destination, max_reads, parent):
            super().__init__(parent)
            self.files, self.destination, self.max_reads = list(files), destination, max_reads
            self.start, self.cancel, self.wait = Mock(), Mock(), Mock(return_value=True)
            jobs.append(self)

    monkeypatch.setattr(mod, "_FetchWorker", Job)
    picker = mod.SraPicker(tmp_path, files=FILES)
    qtbot.addWidget(picker)
    picker.show()
    qtbot.waitExposed(picker)
    return picker, jobs


def test_empty_selection_does_not_start_a_job(controlled_picker):
    picker, jobs = controlled_picker
    for row in range(picker._list.count()):
        picker._list.item(row).setCheckState(Qt.Unchecked)
    picker._start()
    assert jobs == [] and not picker._download.isEnabled()
    assert picker._estimate.text() == "Nothing selected."


def test_failed_download_can_be_retried_without_losing_selection(controlled_picker):
    picker, jobs = controlled_picker
    picker._reads.setValue(2000)
    picker._list.item(1).setCheckState(Qt.Unchecked)
    picker._download.click()
    assert len(jobs) == 1 and jobs[0].files == [FILES[0]]
    assert jobs[0].max_reads == 2000
    jobs[0].start.assert_called_once_with()
    assert not picker._download.isEnabled() and picker._progress.isVisible()
    jobs[0].progress.emit("SRR1", 1200, 24000)
    assert picker._estimate.text() == "SRR1: 1,200 reads (24 kB)"
    jobs[0].finished_all.emit(["partial.fastq.gz"], "connection lost")
    assert picker.written == ["partial.fastq.gz"]
    assert picker.isVisible() and not picker._progress.isVisible()
    assert picker._estimate.text() == "Stopped: connection lost"
    assert picker._download.isEnabled() and picker.chosen_files() == [FILES[0]]
    picker._download.click()
    assert len(jobs) == 2 and jobs[1].files == [FILES[0]]
    jobs[1].finished_all.emit(["complete.fastq.gz"], "")
    assert picker.result() == QDialog.Accepted
    assert picker.written == ["complete.fastq.gz"]
    assert not picker.isVisible() and picker._worker is None


def test_reject_cancels_the_active_job_and_waits_for_it(controlled_picker):
    picker, jobs = controlled_picker
    picker._whole.setChecked(True)
    picker._download.click()
    assert jobs[0].max_reads is None
    picker.reject()
    jobs[0].cancel.assert_called_once_with()
    jobs[0].wait.assert_called_once_with(3000)
    assert picker.result() == QDialog.Rejected and not picker.isVisible()


def test_default_listing_is_loaded_once_and_attached_to_each_row(qtbot, tmp_path, monkeypatch):
    listing = Mock(return_value=FILES)
    monkeypatch.setattr(mod, "runs_for", listing)
    picker = mod.SraPicker(tmp_path)
    qtbot.addWidget(picker)
    listing.assert_called_once_with()
    assert picker.chosen_files() == list(FILES)
    assert picker._list.item(0).text() == FILES[0].label()
