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
    not the model. It must not be swallowed into a silent no-op."""
    from spacr import model_zoo

    def boom(entry, dest, **kwargs):
        raise model_zoo.ChecksumMismatch("sha256 does not match")

    monkeypatch.setattr(model_zoo, "fetch", boom)
    monkeypatch.setattr(mzp.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))

    picker.table.selectRow(_row_needing_download(picker))
    picker.folder_edit.setText(str(tmp_path))
    picker._download_selected()

    assert "does not match" in picker.status.text()
    assert picker.chosen_path() is None
    assert not picker.use_button.isEnabled()


def test_the_download_folder_is_shown_before_anything_is_fetched(picker,
                                                                tmp_path):
    """Large checkpoints on the wrong disk is a full-disk error discovered
    afterwards; the folder is a control, on screen, from the start."""
    assert picker.folder_edit.text() == str(tmp_path)
