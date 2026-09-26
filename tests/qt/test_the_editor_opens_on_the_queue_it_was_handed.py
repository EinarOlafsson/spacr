"""``396`` — the screen half: Make Masks opens on the session it was handed.

``spacr-make-masks`` reads the folder, builds the queue and leaves it in
:mod:`spacr.cli_make_masks`; the first :class:`MakeMasksScreen` built in that
process takes it. Four things are asserted here, and they are the whole of
what the CLI changed about this screen:

* the screen opens the folder the terminal chose, offering exactly the
  fields the session offers, in the session's order -- not the folder
  listing, which is what the file dialog means and would hand back every
  field already curated;
* the handover is taken once. A screen built later -- the user navigating
  away and back -- opens on nothing rather than on a finished session;
* a save records the field as ``done`` in ``curate_status.csv``, which is
  what makes the next session resume instead of starting again;
* a folder opened the ordinary way, from the file dialog, is not a queue:
  it grows no status file and behaves exactly as it did before.

All three layouts open here since 2026-09-19; what the screen does with a
sibling set and a ``_seg.npy`` set is
``test_the_editor_edits_every_queue_layout_in_place.py``. A layout this
screen does not know is still refused.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr import cli_make_masks
from spacr.curation_queue import STATUS_FILENAME, build_queue, read_status
from spacr.qt.screens.make_masks import MakeMasksScreen


@pytest.fixture
def nested(tmp_path: Path) -> Path:
    """Four fields with ``masks/`` beneath them; ``f_01`` has a draft."""
    folder = tmp_path / "nested"
    (folder / "masks").mkdir(parents=True)
    rng = np.random.default_rng(5)
    for index in range(4):
        imageio.imwrite(folder / f"f_{index:02d}.tif",
                        rng.integers(0, 4000, (24, 24), dtype=np.uint16))
    labels = np.zeros((24, 24), dtype=np.uint16)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 2
    imageio.imwrite(folder / "masks" / "f_01.tif", labels)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A Make Masks screen built with no session waiting for it."""
    cli_make_masks.hand_over(None)
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    cli_make_masks.hand_over(None)


def test_the_screen_opens_the_folder_the_terminal_chose(screen, nested):
    """The folder reaches the editor without anybody touching a file dialog."""
    queue = build_queue(nested, order="name")

    assert screen.open_queue(queue) is True
    assert screen._folder == str(nested)
    assert screen._image_files == ["f_00.tif", "f_01.tif", "f_02.tif",
                                   "f_03.tif"]
    assert screen._canvas.image is not None


def test_the_editor_offers_the_session_rather_than_the_folder(screen, nested):
    """Reviewed fields are gone and ``--limit`` holds: this is the queue.

    Opening the folder would offer four fields, two of them already
    decided. The session offers the one field left, and that difference is
    the entire feature.
    """
    (nested / STATUS_FILENAME).write_text(
        "stem,state,n_objects,updated\n"
        "f_00,done,2,2026-09-12T10:00:00\n"
        "f_03,skip,,2026-09-12T10:00:00\n", encoding="utf-8")
    queue = build_queue(nested, order="name", limit=1)

    assert screen.open_queue(queue) is True
    assert screen._image_files == ["f_01.tif"]
    assert "1 done" in screen._status_label.text()
    assert "1 skip" in screen._status_label.text()


def test_a_layout_the_screen_does_not_know_is_refused(screen, nested):
    """A fourth kind of queue must not be opened as though it were nested."""
    from dataclasses import replace

    queue = build_queue(nested, order="name")
    queue = replace(queue, layout=replace(queue.layout, kind="stacks"))

    assert screen.open_queue(queue) is False
    assert screen._folder == ""
    assert screen._queue is None


def test_the_handover_reaches_the_screen_built_next(qtbot, qt_theme_applied,
                                                    nested):
    """What ``spacr-make-masks`` actually does: leave the queue and start Qt."""
    cli_make_masks.hand_over(build_queue(nested, order="name"))
    try:
        opened = MakeMasksScreen()
        qtbot.addWidget(opened)
        assert opened._folder == str(nested)
        assert opened._queue is not None

        again = MakeMasksScreen()
        qtbot.addWidget(again)
        assert again._folder == "", (
            "a screen built later took a session that was already open")
    finally:
        cli_make_masks.hand_over(None)


def test_the_session_survives_the_real_launch_path(qtbot, qt_theme_applied,
                                                   nested):
    """The whole wire, from the command to the window it opens.

    ``open_editor`` leaves the queue and calls ``spacr.qt.run(["make_masks"])``,
    which builds the main window with that app key. Two things have to be
    true for the handover to land and neither is obvious from the screen
    alone: the key must still resolve to this screen rather than to a host
    that folded it in, and the screen must be built while the queue is still
    in the slot.
    """
    from spacr.qt.app import MainWindow

    cli_make_masks.hand_over(build_queue(nested, order="name", limit=2))
    window = None
    try:
        window = MainWindow(initial_app="make_masks")
        qtbot.addWidget(window)
        screen = window._screens.get("make_masks")
        assert screen is not None, (
            "'make_masks' no longer resolves to a screen of its own")
        assert screen._folder == str(nested)
        assert screen._image_files == ["f_00.tif", "f_01.tif"]
        assert "2 this session" in screen._status_label.text()
    finally:
        cli_make_masks.hand_over(None)
        if window is not None:
            window.close()


def test_a_screen_opened_the_ordinary_way_has_no_session(screen, nested):
    """No handover means the file dialog, and the file dialog means no queue."""
    assert screen._queue is None
    assert screen._folder == ""

    screen._open_folder(str(nested))

    assert screen._folder == str(nested)
    assert screen._queue is None
    assert screen._image_files == ["f_00.tif", "f_01.tif", "f_02.tif",
                                   "f_03.tif"]


def test_picking_another_folder_ends_the_session(screen, nested, tmp_path):
    """A folder chosen from the dialog is a different set of fields.

    Saving in it must not write a ``done`` row into the session that was
    open before it: those stems name fields in another folder, and a row
    invented there would take one of them out of its own queue.
    """
    assert screen.open_queue(build_queue(nested, order="name")) is True

    other = tmp_path / "other"
    (other / "masks").mkdir(parents=True)
    imageio.imwrite(other / "z_00.tif",
                    np.zeros((24, 24), dtype=np.uint16))
    assert screen._open_folder(str(other)) is True
    screen._on_save()

    assert screen._queue is None
    assert not (nested / STATUS_FILENAME).exists()
    assert not (other / STATUS_FILENAME).exists()


def test_saving_records_the_field_as_done_so_the_session_resumes(screen,
                                                                 nested):
    """The resume record is written by the editor, or it is never written."""
    queue = build_queue(nested, order="name")
    assert screen.open_queue(queue) is True
    screen._current_index = screen._image_files.index("f_01.tif")
    screen._load_current()

    screen._on_save()

    rows = read_status(nested)
    assert rows["f_01"].state == "done"
    assert rows["f_01"].n_objects == 2
    assert rows["f_01"].updated

    assert [item.stem for item in build_queue(nested, order="name").items] == [
        "f_00", "f_02", "f_03"], "the next session still offers f_01"


def test_saving_a_folder_that_is_not_a_queue_writes_no_record(screen, nested,
                                                            caplog):
    """A folder opened from the dialog must not grow a file nobody asked for."""
    screen._open_folder(str(nested))
    screen._current_index = screen._image_files.index("f_01.tif")
    screen._load_current()

    with caplog.at_level("WARNING"):
        screen._on_save()

    assert not (nested / STATUS_FILENAME).exists()
    # AND NOTHING WAS SWALLOWED ON THE WAY. Absence of a file is weak evidence
    # for a guard: with `if self._queue is None: return` deleted, no file
    # appears either, because the AttributeError that follows is caught by the
    # blanket `except Exception` around the write and logged as an ordinary
    # failure to record. The guard is only observable if the screen took the
    # early return rather than the error path.
    assert "could not record" not in caplog.text.lower(), (
        "the early return was skipped and the failure was logged instead")


def test_a_record_that_cannot_be_written_does_not_lose_the_mask(screen, nested,
                                                                monkeypatch):
    """The mask is on disk; the session's place is the smaller loss of the two."""
    from spacr import curation_queue

    queue = build_queue(nested, order="name")
    assert screen.open_queue(queue) is True

    def _explode(*_args, **_kwargs):
        raise curation_queue.StatusFileError("the sync folder is read-only")

    monkeypatch.setattr(curation_queue, "mark_state", _explode)
    screen._current_index = screen._image_files.index("f_01.tif")
    screen._load_current()

    # THE MASK IS ON DISK -- which is the claim in the name of this test, and
    # the reason the failure above is allowed to be swallowed at all. The
    # status label is not that claim: with the write in
    # `spacr.qt.mask_engine.save_mask` removed, the label still reads "Saved".
    #
    # ASSERTING THE FILE EXISTS IS NOT THE CLAIM EITHER, and that was the first
    # attempt at this fix: the `nested` fixture writes `masks/f_01.tif` to
    # build the layout in the first place, so the file is there whether or not
    # saving does anything. Removing it first is what makes the WRITE
    # observable rather than the fixture.
    written = nested / "masks" / "f_01.tif"
    assert written.is_file(), "fixture precondition"
    written.unlink()

    screen._on_save()

    assert written.is_file(), "the mask the label calls saved is not on disk"
    assert written.stat().st_size > 0

    assert "Saved" in screen._status_label.text()
    # ...and the record really did fail, so the assertions above are about the
    # swallow path rather than the ordinary one.
    assert not (nested / STATUS_FILENAME).exists()


def test_skip_is_recorded_as_skip_and_the_field_is_not_offered_again(screen,
                                                                     nested):
    """ITEM 396: the reader had three reviewed states and the editor wrote one.

    Skip records "this field cannot be curated". It writes no mask, moves on,
    and the next session leaves the field out rather than offering the same
    unusable field again.
    """
    queue = build_queue(nested, order="name")
    assert screen.open_queue(queue) is True
    assert screen._btn_skip.isEnabled()
    screen._current_index = screen._image_files.index("f_02.tif")
    screen._load_current()

    screen._btn_skip.click()

    rows = read_status(nested)
    assert rows["f_02"].state == "skip"
    assert not (nested / "masks" / "f_02.tif").exists(), (
        "a skipped field must not grow a mask")
    assert screen._image_files[screen._current_index] == "f_03.tif"
    assert "f_02.tif skipped" in screen._status_label.text()
    assert [item.stem for item in build_queue(nested, order="name").items] == [
        "f_00", "f_01", "f_03"]


def test_skip_is_not_offered_for_a_folder_that_is_not_a_queue(screen, nested):
    """Only a queue has a record to write a skip to."""
    assert screen._open_folder(str(nested)) is True
    assert not screen._btn_skip.isEnabled()

    screen._on_skip()

    assert not (nested / STATUS_FILENAME).exists()


def _split_seed(folder: Path) -> Path:
    """A seed whose label 1 lies in two separated pieces, as cellpose leaves."""
    labels = np.zeros((24, 24), dtype=np.uint16)
    labels[2:8, 2:8] = 1
    labels[20, 20] = 1
    labels[12:18, 12:18] = 2
    path = folder / "masks" / "f_01.tif"
    imageio.imwrite(path, labels)
    return path


def test_a_save_with_no_edit_leaves_the_mask_file_untouched(screen, nested):
    """ITEM 396: "open it, look, save" must mean the seed was right.

    Every save went through `canonical_labels`, which splits a label lying in
    two separated pieces, so a no-edit save of such a seed added an object
    and recorded a count the file on disk did not have.
    """
    seed = _split_seed(nested)
    before = seed.read_bytes()
    queue = build_queue(nested, order="name")
    assert screen.open_queue(queue) is True
    screen._current_index = screen._image_files.index("f_01.tif")
    screen._load_current()

    screen._on_save()

    assert seed.read_bytes() == before, "a save that changed nothing rewrote"
    rows = read_status(nested)
    assert rows["f_01"].state == "done"
    assert rows["f_01"].n_objects == 2
    assert "Unchanged" in screen._status_label.text()


def test_a_save_after_an_edit_still_writes(screen, nested):
    """The no-op is for no edit only; a changed mask is written as before."""
    seed = _split_seed(nested)
    queue = build_queue(nested, order="name")
    assert screen.open_queue(queue) is True
    screen._current_index = screen._image_files.index("f_01.tif")
    screen._load_current()

    edited = screen._canvas.mask.copy()
    edited[12:18, 12:18] = 0
    screen._canvas.mask = edited
    screen._on_save()

    written = imageio.imread(seed)
    assert not np.any(written[12:18, 12:18]), "the edit did not reach disk"
    assert "Saved" in screen._status_label.text()


def test_an_edited_save_records_the_count_the_file_holds(screen, nested):
    """ITEM 396: the count is taken after `canonical_labels`, not before.

    The seed's label 1 lies in two pieces; an edit elsewhere on the field
    keeps it that way, the save splits it, and the file holds three
    objects. The record said two, the canvas's count before the split.
    """
    seed = _split_seed(nested)
    queue = build_queue(nested, order="name")
    assert screen.open_queue(queue) is True
    screen._current_index = screen._image_files.index("f_01.tif")
    screen._load_current()

    edited = screen._canvas.mask.copy()
    edited[12:14, 12:18] = 0
    screen._canvas.mask = edited
    screen._on_save()

    on_disk = imageio.imread(seed)
    assert int(np.count_nonzero(np.unique(on_disk))) == 3
    assert read_status(nested)["f_01"].n_objects == 3
