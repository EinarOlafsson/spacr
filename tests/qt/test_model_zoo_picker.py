"""The model-zoo picker: lists without fetching, and never returns a path
to a file that is not there.

A model setting takes a filesystem path. This dialog is the other way in --
browse what exists, download one, and hand the path back to the field. Three
things carry it, and each is a way it could mislead:

  * opening it must not start a gigabyte download;
  * "Use this model" must be impossible until the file is actually on disk,
    because returning a path to a missing file writes a setting that fails
    much later, in the pipeline, naming a file the user never typed;
  * a failed or unverified download must be said out loud rather than leaving
    a half-written file that looks like a model.
"""
from __future__ import annotations

import os

import pytest

from spacr.qt.widgets import model_zoo_picker as mzp


def _row_needing_download(picker):
    """The first row whose model is NOT already on this machine.

    Row 0 is not it: the bundled plaque model ships inside the package, so it
    is on disk for everyone. Selecting by INDEX made three tests assert the
    not-downloaded behaviour against a downloaded model, and they failed --
    correctly. Select by state instead.
    """
    for row, entry in enumerate(picker._entries):
        if picker._local_path(entry) is None:
            return row
    pytest.skip("every offered model is already present")


@pytest.fixture
def picker(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(mzp, "DEFAULT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(mzp, "remembered_model_dir", lambda: str(tmp_path))
    dialog = mzp.ModelZooPicker(kinds=("cellpose",))
    yield dialog
    # Join any worker BEFORE the dialog is destroyed: a QThread deleted while
    # running aborts the process, which is how this was found.
    dialog._stop_any_download()
    dialog.deleteLater()


def test_opening_it_lists_without_downloading_anything(picker, tmp_path):
    """The list is useful on its own, and a dialog that transfers a gigabyte
    on open is one users learn not to open."""
    assert picker.table.rowCount() > 0, "the zoo should offer something"
    assert list(tmp_path.iterdir()) == [], "opening downloaded a file"


def test_only_cellpose_models_are_offered_when_that_is_asked_for(picker):
    """A pathogen-model field offering a detector would be offering something
    that cannot be loaded."""
    assert picker._entries, "nothing to check"
    assert {e.kind for e in picker._entries} == {"cellpose"}


def test_use_is_refused_until_the_file_is_actually_on_disk(picker):
    """The property that matters most.

    Returning a path to a file that is not there writes a setting that fails
    later, inside the pipeline, naming a file the user never typed.
    """
    picker.table.selectRow(_row_needing_download(picker))
    entry = picker.selected_entry()
    assert entry is not None
    assert picker._local_path(entry) is None
    assert not picker.use_button.isEnabled()

    picker._accept_selected()
    assert picker.chosen_path() is None, (
        "accepting a model that is not downloaded returned a path anyway")


def test_a_downloaded_model_becomes_usable_and_returns_its_path(picker,
                                                               tmp_path):
    """The other half: once the file is there, the path comes back."""
    row = _row_needing_download(picker)
    picker.table.selectRow(row)
    entry = picker.selected_entry()
    (tmp_path / entry.name).write_bytes(b"weights")
    picker.folder_edit.setText(str(tmp_path))
    picker.refresh()
    picker.table.selectRow(row)

    assert picker.use_button.isEnabled()
    picker._accept_selected()
    assert picker.chosen_path() == str(tmp_path / entry.name)


def test_a_failed_download_is_reported_and_leaves_nothing_usable(
        picker, tmp_path, monkeypatch):
    """fetch refuses an entry whose checksum does not match, and that refusal
    is the most important message this dialog carries: it means the bytes are
    not the model. It must survive the refresh that follows."""
    monkeypatch.setattr(mzp.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    picker.table.selectRow(_row_needing_download(picker))
    picker.folder_edit.setText(str(tmp_path))

    picker._on_download_failed("sha256 does not match")

    assert "does not match" in picker.status.text()
    assert picker.chosen_path() is None
    assert not picker.use_button.isEnabled()


def test_starting_a_download_does_not_block_the_gui_thread(picker, tmp_path,
                                                           monkeypatch):
    """THE FREEZE. These files are 1.2 GB; fetched from the button handler the
    event loop stops for minutes, the bar cannot move, and the compositor
    offers to force-quit spaCR.

    Driven by making fetch sleep: if the download were still synchronous, the
    handler would not return until the sleep finished.
    """
    import time

    from spacr import model_zoo

    def slow_fetch(entry, dest, **kwargs):
        time.sleep(1.0)
        return str(tmp_path / entry.name)

    monkeypatch.setattr(model_zoo, "fetch", slow_fetch)
    # The selected entry may be an unverifiable one, which asks first.
    monkeypatch.setattr(mzp.QMessageBox, "question",
                        staticmethod(lambda *a, **k: mzp.QMessageBox.Yes))
    picker.table.selectRow(_row_needing_download(picker))
    picker.folder_edit.setText(str(tmp_path))

    started = time.monotonic()
    picker._download_selected()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"_download_selected blocked the GUI thread for {elapsed:.1f}s")
    # isHidden, NOT isVisible: the dialog itself is never shown in a headless
    # test, and isVisible() is False for every child of a hidden ancestor.
    assert not picker.progress.isHidden()
    picker._stop_any_download()


def test_progress_reports_percent_speed_and_time_left(picker):
    """What the bar has to say while it runs."""
    picker._started_at = __import__("time").monotonic() - 2.0
    picker._last_emit = 0.0
    picker._on_progress(50 * 1024 * 1024, 100 * 1024 * 1024)

    assert picker.progress.value() == 50
    text = picker.status.text()
    assert "MB" in text, text
    assert "/s" in text, f"no transfer rate: {text}"
    assert "left" in text or "estimating" in text, f"no time remaining: {text}"


def test_an_unknown_total_does_not_invent_a_percentage(picker):
    """A server with no content-length gives no total. A bar with no end is
    honest; a percentage computed from an unknown total is not."""
    picker._started_at = __import__("time").monotonic() - 1.0
    picker._last_emit = 0.0
    picker._on_progress(1024 * 1024, 0)

    assert picker.progress.maximum() == 0, "indeterminate, not a fake percent"
    assert "size unknown" in picker.status.text()


def test_the_eta_says_estimating_rather_than_a_wrong_number():
    """An ETA from the first chunk is wrong by minutes and reads as a promise."""
    assert mzp._human_eta(-1) == "estimating…"
    assert mzp._human_eta(float("nan")) == "estimating…"
    assert mzp._human_eta(10**6) == "estimating…"
    assert mzp._human_eta(30) == "30s left"
    assert mzp._human_eta(90) == "1m 30s left"


def test_the_download_folder_is_shown_before_anything_is_fetched(picker,
                                                                tmp_path):
    """Large checkpoints on the wrong disk is a full-disk error discovered
    afterwards; the folder is a control, on screen, from the start."""
    assert picker.folder_edit.text() == str(tmp_path)


def _unverified_row(picker):
    """A row that publishes no checksum AND is not already on disk.

    Both halves matter. An entry already present takes the "Ready" branch and
    never reaches the checksum warning -- which is right, there is nothing to
    download -- so a test that ignored that would assert the warning against a
    row that correctly does not show it.
    """
    for row, entry in enumerate(picker._entries):
        if not getattr(entry, "sha256", "") and picker._local_path(entry) is None:
            return row
    # SYNTHETIC, not skipped. The only unverifiable entry was the retired
    # toxo_plaque_cyto, so after the retirement these three tests skipped --
    # and a skipped test is a guard that has quietly stopped guarding. The
    # confirmation path is still live for any future entry published without a
    # hash, which is exactly when it will matter and exactly when nobody will
    # remember it exists. So the case is constructed rather than found.
    from copy import copy

    donor = copy(picker._entries[0])
    object.__setattr__(donor, "sha256", "")
    object.__setattr__(donor, "path", "")
    object.__setattr__(donor, "name", "a_model_with_no_checksum.CP_model")
    picker._entries.append(donor)
    picker.table.setRowCount(len(picker._entries))
    from PySide6.QtWidgets import QTableWidgetItem
    for column, text in enumerate((donor.name, donor.kind, "", "not downloaded")):
        picker.table.setItem(len(picker._entries) - 1, column,
                             QTableWidgetItem(str(text)))
    return len(picker._entries) - 1


def test_an_unverifiable_model_says_so_before_the_click(picker):
    """The dead end the user hit.

    fetch refuses an entry it cannot verify. Without this the Download button
    is enabled, pressing it fails, and the message explains a policy the user
    had no way to see beforehand.
    """
    picker.table.selectRow(_unverified_row(picker))
    assert "no checksum" in picker.status.text().lower(), picker.status.text()


def test_downloading_an_unverifiable_model_asks_first_and_honours_no(
        picker, tmp_path, monkeypatch):
    """Declining must not download."""
    from spacr import model_zoo

    called = {}
    monkeypatch.setattr(model_zoo, "fetch",
                        lambda *a, **k: called.setdefault("yes", True))
    monkeypatch.setattr(mzp.QMessageBox, "question",
                        staticmethod(lambda *a, **k: mzp.QMessageBox.No))

    picker.table.selectRow(_unverified_row(picker))
    picker.folder_edit.setText(str(tmp_path))
    picker._download_selected()

    assert "yes" not in called, "declining still started the download"
    assert "cancelled" in picker.status.text().lower()


def test_accepting_the_risk_passes_require_checksum_false(picker, tmp_path,
                                                          monkeypatch):
    """Accepting is what makes it possible at all -- and it must be the ONLY
    thing that turns verification off, never a default."""
    seen = {}

    class FakeWorker:
        def __init__(self, entry, folder, *, unverified=False):
            seen["unverified"] = unverified
        def moveToThread(self, _t): pass
        progressed = finished = failed = None

    picker.table.selectRow(_unverified_row(picker))
    picker.folder_edit.setText(str(tmp_path))
    monkeypatch.setattr(mzp.QMessageBox, "question",
                        staticmethod(lambda *a, **k: mzp.QMessageBox.Yes))
    monkeypatch.setattr(mzp, "_DownloadWorker", FakeWorker)
    try:
        picker._download_selected()
    except Exception:
        pass                      # the fake worker has no signals to connect
    assert seen.get("unverified") is True
