"""Make Masks: verdicts on cut fields, folds it cannot store, work after close.

Pins a handful of paths a curating session takes only now and then:

* Keep on a field that was recropped retires it and stays on its first
  crop, saying only the verdict; Skip on the last field says it was the last;
* the settings panel opens with every category unfolded when the stored
  folds cannot be read, and folding one still takes effect when the fold
  cannot be stored;
* a histogram, comparison or detection that finishes after the screen was
  destroyed is dropped without an error;
* the organelle methods run through the one CPU dispatch the detect button
  uses; an edit with no field has no ledger to go in; a filter row names an
  object without an intensity when none was measured;
* a magnifier result for a field of another size is not pasted.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
import shiboken6

from spacr.qt import mask_engine as engine
from spacr.qt import organelle_modes
from spacr.qt.screens import make_masks as mm

N = 96


def _folder(root: Path, names=("a.tif", "b.tif", "c.tif")) -> Path:
    folder = root / "fields"
    (folder / "masks").mkdir(parents=True)
    image = np.full((N, N), 100, dtype=np.uint16)
    mask = np.zeros((N, N), dtype=np.uint16)
    for label, (top, left) in enumerate(((10, 10), (60, 60)), 1):
        image[top:top + 20, left:left + 20] = 30000
        mask[top:top + 20, left:left + 20] = label
    for name in names:
        imageio.imwrite(folder / name, image)
        imageio.imwrite(folder / "masks" / name, mask)
    return folder


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path):
    folder = _folder(tmp_path)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(folder))
    yield made, folder
    made._magnifier.close()
    made.close_folded()


# ---------------------------------------------------------------------------
# Verdicts and skips
# ---------------------------------------------------------------------------

def test_keep_on_a_recropped_field_retires_it_and_stays_on_its_crop(opened):
    screen, folder = opened
    written = screen.recrop(0, 0, 48, 48)
    assert written is not None, screen._status_label.text()
    assert screen._on_curate(True) is not None
    assert screen._image_files[screen._current_index] == written
    assert screen._status_label.text() == "a.tif kept"
    assert (folder / engine.RECROP_ARCHIVE_DIRNAME / "a.tif").is_file()


def test_skip_on_the_last_field_says_it_was_the_last(opened, monkeypatch):
    from spacr import curation_queue

    screen, folder = opened
    monkeypatch.setattr(curation_queue, "mark_state", lambda *a: None)
    screen._queue = SimpleNamespace(folder=str(folder))
    screen._current_index = len(screen._image_files) - 1
    screen._on_skip()
    assert screen._status_label.text() == (
        "c.tif skipped, and will not be offered again  —  that was the "
        "last field in this session")


# ---------------------------------------------------------------------------
# Folded categories
# ---------------------------------------------------------------------------

def test_unreadable_folds_open_every_category(qtbot, qt_theme_applied,
                                              monkeypatch):
    from spacr.qt import preferences

    def unreadable(*args, **kwargs):
        raise OSError("preferences are damaged")

    monkeypatch.setattr(preferences, "get_section_layout", unreadable)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        assert made._folded_categories == set()
        assert made._settings_categories
        assert all(section.is_expanded()
                   for _title, section in made._settings_categories)
    finally:
        made._magnifier.close()
        made.close_folded()


def test_a_fold_that_cannot_be_stored_still_folds(opened, monkeypatch):
    from spacr.qt import preferences

    screen, _folder_path = opened
    stored = []

    def unwritable(*args, **kwargs):
        stored.append(kwargs)
        raise OSError("read-only")

    monkeypatch.setattr(preferences, "set_section_layout", unwritable)
    title, section = screen._settings_categories[0]
    section.set_expanded(False)
    screen._remember_folded_categories()
    assert title in screen._folded_categories
    assert stored and title in stored[-1]["folded"]
    assert not section.is_expanded()


# ---------------------------------------------------------------------------
# Workers finishing after the window
# ---------------------------------------------------------------------------

def test_work_finishing_after_the_screen_is_destroyed_is_dropped(
        qt_theme_applied, listing_free_screen):
    screen = listing_free_screen
    done = (screen._histogram_done, screen._comparison_done,
            screen._detection_done)
    shiboken6.delete(screen)
    assert not shiboken6.isValid(screen)
    for deliver in done:
        assert deliver(object(), None, None) is None


@pytest.fixture
def listing_free_screen(qt_theme_applied):
    made = mm.MakeMasksScreen()
    made._magnifier.close()
    made.close_folded()
    return made


# ---------------------------------------------------------------------------
# Detection, the ledger, the filter rows, the magnifier paste
# ---------------------------------------------------------------------------

def test_an_organelle_method_runs_through_the_detect_dispatch(opened):
    screen, _folder_path = opened
    image = screen._canvas.image
    labels, centres = screen._cpu_detect(image, "adaptive",
                                         screen._otsu_settings())
    expected = organelle_modes.segment(image, "adaptive",
                                       screen._method_params(),
                                       min_area=screen._detect_min_area())
    assert centres is None
    np.testing.assert_array_equal(labels, expected)


def test_an_edit_with_no_field_has_no_ledger(qtbot, qt_theme_applied):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        assert made._log is None
        assert made._record("brush", [1], 25) is None
    finally:
        made._magnifier.close()
        made.close_folded()


def test_a_filter_row_without_a_measured_intensity_names_only_the_area(
        opened):
    screen, _folder_path = opened
    removal = engine.ObjectRemoval(
        label=4, values={"area": 12.0},
        failed=(engine.FailedBound(0, "area", "min", 40.0, 12.0),))
    line = screen._filter_removal_line(removal)
    assert line == "Object 4 with area 12 px was removed by minimum area 40"


def test_a_magnifier_result_for_a_field_of_another_size_is_not_pasted(opened):
    screen, _folder_path = opened
    before = screen._canvas.mask.copy()
    request = mm._MagnifierRequest(
        key=("old",), crop=np.zeros((8, 8), np.uint16), box=(0, 0, 8, 8),
        shape=(512, 512), mode="otsu", sensitivity=0.0, bright=True,
        min_area=0, model_name="", diameter=0, colour=(255, 0, 0))
    labels = np.ones((8, 8), np.int32)
    result = mm._MagnifierResult(request, labels, "otsu", "", None, 1)
    assert screen._commit_magnifier_result(result) == []
    np.testing.assert_array_equal(screen._canvas.mask, before)
