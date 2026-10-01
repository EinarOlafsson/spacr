"""Item 288: intensity-plane selection and the cell-adjustment record.

``_select_intensity_channel`` is shared by the on-disk and in-memory mask
filters so the two cannot pick different planes. It reads channel-last
when the last axis is small, channel-first when the first is, and refuses
an index past the channels it found.

``adjust_cell_masks`` in place keeps a record of which cell masks it has
already adjusted, so a re-run does not adjust an adjusted mask again. The
record is best effort: unreadable or malformed means "adjust everything",
and a record that cannot be written leaves no temporary file behind. The
worker count defaults to the machine's cores less two, never below one,
and a run into a separate output folder keeps no record at all.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import spacr.utils as utils


@pytest.mark.parametrize("raw, channel, expected", [
    (np.arange(6).reshape(2, 3), 1, np.arange(6).reshape(2, 3)),
    (np.arange(24).reshape(2, 4, 3), None, np.arange(24).reshape(2, 4, 3)),
    (np.dstack([np.zeros((5, 6)), np.ones((5, 6))]), 1, np.ones((5, 6))),
    (np.stack([np.zeros((5, 6)), np.full((5, 6), 2.0)]), 1,
     np.full((5, 6), 2.0)),
    (np.stack([np.full((5, 6), float(i)) for i in range(6)], axis=-1), 4,
     np.full((5, 6), 4.0)),
    (np.zeros((2, 3, 4, 5)), 0, np.zeros((2, 3, 4, 5))),
])
def test_the_intensity_plane_is_read_by_layout(raw, channel, expected):
    out = utils._select_intensity_channel(raw, channel)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("raw, channel, words", [
    (np.zeros((5, 6, 2)), 2, "channel-last"),
    (np.zeros((2, 6, 7)), 3, "channel-first"),
    (np.zeros((6, 7, 8)), 9, "for image with shape"),
])
def test_an_index_past_the_channels_is_refused(raw, channel, words):
    with pytest.raises(ValueError) as excinfo:
        utils._select_intensity_channel(raw, channel)
    assert words in str(excinfo.value)


def test_a_filter_with_no_bound_is_not_described():
    text = utils._describe_object_filters([
        {"property": "area", "min": 10.0, "max": None},
        {"property": "eccentricity", "min": None, "max": None},
        {"property": "solidity", "min": None, "max": 0.9},
    ])
    assert text == "area >= 10, solidity <= 0.9"


def test_a_folder_cannot_be_fingerprinted(tmp_path):
    assert utils._file_sha256(str(tmp_path)) is None


def test_a_record_that_is_not_a_mapping_means_adjust_everything(tmp_path):
    (tmp_path / utils.ADJUSTED_CELLS_LEDGER).write_text(json.dumps(["f0"]))
    assert utils._read_adjusted_cells(str(tmp_path)) == {}


def test_a_record_that_cannot_be_written_leaves_nothing_behind(tmp_path):
    with pytest.raises(TypeError):
        utils._write_adjusted_cells(str(tmp_path), {"f0.npy": object()})
    assert sorted(os.listdir(tmp_path)) == []


def _plate(tmp_path, names=("f0.npy", "f1.npy")):
    folders = {}
    for role in ("pathogen", "cell", "nucleus"):
        folder = tmp_path / role
        folder.mkdir()
        folders[role] = str(folder)
    cell = np.zeros((20, 20), np.uint16)
    cell[2:18, 2:18] = 1
    nucleus = np.zeros_like(cell)
    nucleus[8:12, 8:12] = 1
    for name in names:
        np.save(tmp_path / "cell" / name, cell)
        np.save(tmp_path / "nucleus" / name, nucleus)
        np.save(tmp_path / "pathogen" / name, np.zeros_like(cell))
    return folders


def test_the_default_worker_count_is_the_cores_less_two(tmp_path,
                                                       monkeypatch):
    folders = _plate(tmp_path)
    monkeypatch.setattr(utils, "cpu_count", lambda: 2)
    seen = []
    monkeypatch.setattr(utils, "print_progress",
                        lambda *a, **k: seen.append(k["n_jobs"]))
    utils.adjust_cell_masks(folders["pathogen"], folders["cell"],
                            folders["nucleus"])
    assert seen == [1, 1], "two cores leave one worker, not zero"
    record = utils._read_adjusted_cells(folders["cell"])
    assert sorted(record) == ["f0.npy", "f1.npy"]


def test_a_pooled_run_into_an_output_folder_keeps_no_record(tmp_path):
    folders = _plate(tmp_path)
    output = tmp_path / "adjusted"
    utils.adjust_cell_masks(folders["pathogen"], folders["cell"],
                            folders["nucleus"], n_jobs=2,
                            output_folder=str(output))
    assert sorted(os.listdir(output)) == ["f0.npy", "f1.npy"]
    assert not os.path.exists(os.path.join(folders["cell"],
                                           utils.ADJUSTED_CELLS_LEDGER))
