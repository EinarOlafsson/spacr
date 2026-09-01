"""Measure can fetch its own example data.

Mask's example plate is raw acquisition; Measure needs that plate AFTER
segmentation -- merged arrays carrying label masks. Without its own dataset the
only way to try Measure was to run Mask first, over about 400 MB of images, for
the twenty-odd minutes that takes.

The data is published at ``einarolafsson/spacr-example-measure``: sixteen
fields across four wells, all four wells kept because well-level aggregation
and between-condition comparison are most of what Measure does after the
per-object step.

Instruction 332.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spacr.qt.hf_download import MEASURE_EXAMPLE_REPO, _MeasureExampleWorker
from spacr.qt.screens.app_screen import EXAMPLE_DATA_SECTIONS


def test_measure_declares_where_its_button_goes():
    """It must land beside `src`, which is what it fills."""
    assert EXAMPLE_DATA_SECTIONS["measure"] == "Input & Experiment"


def test_the_repo_is_its_own_not_the_mask_demo():
    """A different artefact at a different stage: raw acquisition versus that
    plate after `preprocess_generate_masks`."""
    from spacr.qt.hf_download import DATASET_REPO

    assert MEASURE_EXAMPLE_REPO == "einarolafsson/spacr-example-measure"
    assert MEASURE_EXAMPLE_REPO != DATASET_REPO


def test_the_archives_are_unpacked_to_what_measure_reads(tmp_path):
    """The compression is a TRANSPORT detail -- it halves the download -- and
    Measure loads `.npy`. Converting on arrival keeps the second format
    entirely inside the downloader."""
    merged = tmp_path / "merged"
    merged.mkdir()
    array = np.arange(2 * 3 * 7, dtype=np.uint16).reshape(2, 3, 7)
    np.savez_compressed(merged / "plate1_E01_1_1.npz", image=array)

    _MeasureExampleWorker(tmp_path)._expand_arrays(merged)

    written = merged / "plate1_E01_1_1.npy"
    assert written.is_file(), "no .npy was written"
    assert np.array_equal(np.load(written), array)


def test_the_archive_is_removed_after_unpacking(tmp_path):
    """Keeping both doubles the disk cost of the example set for a file
    nothing will open again."""
    merged = tmp_path / "merged"
    merged.mkdir()
    np.savez_compressed(merged / "f.npz",
                        image=np.zeros((2, 2, 7), dtype=np.uint16))

    _MeasureExampleWorker(tmp_path)._expand_arrays(merged)

    assert not (merged / "f.npz").exists()
    assert (merged / "f.npy").is_file()


def test_an_unreadable_archive_does_not_cost_the_others(tmp_path):
    """One bad file must not lose the other fifteen, and what failed stays on
    disk so it is visible rather than merely absent."""
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "broken.npz").write_bytes(b"not an archive")
    np.savez_compressed(merged / "good.npz",
                        image=np.ones((2, 2, 7), dtype=np.uint16))

    _MeasureExampleWorker(tmp_path)._expand_arrays(merged)

    assert (merged / "good.npy").is_file(), "a bad archive stopped the rest"
    assert (merged / "broken.npz").exists(), "the failure was hidden"


def test_an_already_unpacked_field_is_left_alone(tmp_path):
    """Re-running the download must not overwrite what is there."""
    merged = tmp_path / "merged"
    merged.mkdir()
    kept = np.full((2, 2, 7), 5, dtype=np.uint16)
    np.save(merged / "f.npy", kept)
    np.savez_compressed(merged / "f.npz",
                        image=np.zeros((2, 2, 7), dtype=np.uint16))

    _MeasureExampleWorker(tmp_path)._expand_arrays(merged)

    assert np.array_equal(np.load(merged / "f.npy"), kept)
    assert not (merged / "f.npz").exists()


def test_a_missing_merged_folder_is_not_an_error(tmp_path):
    """A cancelled download leaves no `merged/`; unpacking must not raise."""
    _MeasureExampleWorker(tmp_path)._expand_arrays(tmp_path / "merged")


def test_an_archive_without_the_expected_key_still_loads(tmp_path):
    """The publisher writes `image`; a hand-made archive may not."""
    merged = tmp_path / "merged"
    merged.mkdir()
    array = np.ones((2, 2, 7), dtype=np.uint16)
    np.savez_compressed(merged / "f.npz", something_else=array)

    _MeasureExampleWorker(tmp_path)._expand_arrays(merged)

    assert np.array_equal(np.load(merged / "f.npy"), array)


def test_the_cached_copy_is_reused_rather_than_refetched(qapp, tmp_path,
                                                         monkeypatch):
    """370 MB is a long wait for files already on disk."""
    from spacr.qt.screens.app_screen import AppScreen

    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "f.npy", np.zeros((2, 2, 7), dtype=np.uint16))

    class _Screen:
        _settings_model = None
        measure_example_destination = lambda self: tmp_path
        _put_the_measure_example_in_place = (
            lambda self, dest: {"src": str(dest)})
        load_the_measure_example = AppScreen.load_the_measure_example

    def _must_not_download(*a, **k):
        pytest.fail("it re-downloaded data that was already on disk")

    assert _Screen().load_the_measure_example(
        ask=_must_not_download) == {"src": str(tmp_path)}


def test_a_cancelled_download_is_not_mistaken_for_a_cached_one(qapp, tmp_path):
    """A cancelled download leaves the folder behind, so the folder EXISTING
    cannot be the test -- it has to hold an array."""
    from spacr.qt.screens.app_screen import AppScreen

    (tmp_path / "merged").mkdir()          # empty, as a cancel leaves it
    asked = []

    class _Screen:
        measure_example_destination = lambda self: tmp_path
        load_the_measure_example = AppScreen.load_the_measure_example

    _Screen().load_the_measure_example(
        ask=lambda *a, **k: asked.append(True))
    assert asked, "an empty merged folder was treated as a finished download"
