"""spaCR's own mask loads never unpickle.

Five loads of mask files spaCR writes itself passed ``allow_pickle=True``:
the four in :func:`spacr.utils.process_mask_file_adjust_cell` (pathogen,
cell, nucleus and organelle masks, read when ``adjust_cells`` is on) and the
one in :func:`spacr.io._load_array_any` (every non-reference mask read into
``merged/``). Every writer of those files saves a plain ``uint16`` array, and
did when the flag was added (October 2024 for the merged reader, July 2025
for the adjuster: ``mask.astype(np.uint16)`` before ``np.save`` both times),
so the flag bought nothing. What it
allowed was code execution: numpy unpickles an object array on load, and
unpickling runs whatever the pickle names. A ``.npy`` put into a plate's
``masks/`` folder ran when Mask read it. Item 429 decided against
``allow_pickle`` for the normalised archives; this is the same decision for
the masks.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from tests.test_a_stack_file_spacr_wrote_is_one_it_can_read import (  # noqa: E402,F401
    _cq1_plate, fake_cellpose,
)


class _RunsWhenUnpickled:
    """An object whose unpickling creates a directory: proof that code ran."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


def _pickled_mask(path: Path, marker: Path) -> Path:
    np.save(path, np.array([_RunsWhenUnpickled(marker)], dtype=object),
            allow_pickle=True)
    return path


def _raised_by(function, *args, **kwargs):
    """What ``function`` raised, or None; checked after the marker is."""
    try:
        function(*args, **kwargs)
    except Exception as exc:                                 # noqa: BLE001
        return exc
    return None


def _mask(value=1, shape=(32, 32)):
    mask = np.zeros(shape, np.uint16)
    mask[4:12, 4:12] = value
    mask[18:28, 18:28] = value + 1
    return mask


def _adjust_folders(root: Path):
    folders = {}
    for role in ("pathogen", "cell", "nucleus"):
        folder = root / "masks" / f"{role}_mask_stack"
        folder.mkdir(parents=True)
        folders[role] = folder
    return folders


def test_the_mask_run_writes_masks_that_load_without_pickle(
        tmp_path, fake_cellpose):
    """Premise, true before and after: every mask a Mask run writes, adjusted
    cells included, is a plain uint16 array that loads with pickling off."""
    from spacr.core import preprocess_generate_masks

    plate = _cq1_plate(tmp_path / "plate")
    preprocess_generate_masks({
        "src": str(plate), "metadata_type": "cq1", "custom_regex": None,
        "channels": [0, 1, 2, 3], "cell_channel": 3, "nucleus_channel": 0,
        "pathogen_channel": 2, "cell_diameter": 30, "nucleus_diameter": 30,
        "pathogen_diameter": 30, "magnification": 20, "test_mode": False,
        "preprocess": True, "masks": True, "save": True, "plot": False,
        "verbose": False, "n_jobs": 1, "batch_size": 50,
        "adjust_cells": True, "seg_qc": "off", "randomize": False,
        "keep_intermediate": True, "consolidate": False, "timelapse": False,
        "pipeline_style": "v1",
    })

    masks = sorted((plate / "masks").glob("*_mask_stack/*.npy"))
    assert {p.parent.name for p in masks} == {
        "cell_mask_stack", "nucleus_mask_stack", "pathogen_mask_stack"}
    assert len(masks) == 6
    for path in masks:
        array = np.load(path, allow_pickle=False)
        assert array.dtype == np.uint16, path
    merged = sorted((plate / "merged").glob("*.npy"))
    assert len(merged) == 2
    for path in merged:
        assert np.load(path, allow_pickle=False).shape[-1] == 4 + 3


def test_adjusting_cells_reads_whole_masks_without_pickle(tmp_path):
    """Premise, true before and after: the adjuster's own inputs load."""
    from spacr.io import _save_array_atomic
    from spacr.utils import process_mask_file_adjust_cell

    folders = _adjust_folders(tmp_path)
    for role in folders:
        _save_array_atomic(str(folders[role] / "f1.npy"), _mask())
    organelle = tmp_path / "masks" / "organelle_mask_stack"
    organelle.mkdir()
    _save_array_atomic(str(organelle / "f1.npy"), _mask())

    process_mask_file_adjust_cell(
        "f1.npy", str(folders["pathogen"]), str(folders["cell"]),
        str(folders["nucleus"]), organelle_folder=str(organelle))

    adjusted = np.load(folders["cell"] / "f1.npy", allow_pickle=False)
    assert adjusted.dtype == np.uint16


@pytest.mark.parametrize("role", ["pathogen", "cell", "nucleus", "organelle"])
def test_adjusting_cells_never_runs_a_pickled_mask(tmp_path, role):
    """Each of the four loads refuses an object array instead of running it."""
    from spacr.utils import process_mask_file_adjust_cell

    folders = _adjust_folders(tmp_path)
    organelle = tmp_path / "masks" / "organelle_mask_stack"
    organelle.mkdir()
    folders["organelle"] = organelle
    for name, folder in folders.items():
        np.save(folder / "f1.npy", _mask())
    marker = tmp_path / "code_ran"
    _pickled_mask(folders[role] / "f1.npy", marker)

    raised = _raised_by(
        process_mask_file_adjust_cell,
        "f1.npy", str(folders["pathogen"]), str(folders["cell"]),
        str(folders["nucleus"]), organelle_folder=str(organelle))

    assert not marker.exists(), "loading a mask ran code from a pickle"
    assert isinstance(raised, ValueError) and "allow_pickle" in str(raised)


def test_the_merged_reader_never_runs_a_pickled_mask(tmp_path):
    from spacr.io import _load_array_any

    whole = tmp_path / "whole.npy"
    np.save(whole, _mask())
    assert _load_array_any(str(whole)).dtype == np.uint16

    marker = tmp_path / "code_ran"
    pickled = _pickled_mask(tmp_path / "f1.npy", marker)
    raised = _raised_by(_load_array_any, str(pickled))
    assert not marker.exists(), "loading a mask ran code from a pickle"
    assert isinstance(raised, ValueError) and "allow_pickle" in str(raised)


def test_the_merged_reader_still_reads_tiff_masks(tmp_path):
    import tifffile
    from spacr.io import _load_array_any

    path = tmp_path / "f1.tif"
    tifffile.imwrite(path, _mask())
    assert np.array_equal(_load_array_any(str(path)), _mask())
