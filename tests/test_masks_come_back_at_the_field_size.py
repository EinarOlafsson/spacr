"""Every saved mask has its field's size, whatever scale it was segmented at.

The tutorial's Mask run on 600 px crops stopped in ``adjust_cell_masks``:
"Mask dimensions do not match for plate1_E01_13_1.npy: (600, 600), expected
(1200, 1200)". The pathogen diameter of 15 is the one that differs from the
others (cell and nucleus 30), and Cellpose works at ``30 / diameter``, twice
the field. A pathogen mask at its working scale was saved, or a stale one of
another size was taken as already generated, and cell adjustment refused the
mismatch. A small synthetic field with a pathogen scale unlike the cell
scale is segmented here by a model that hands labels back at its working
scale, and the three masks must still line up and adjust.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

import spacr.object as objects
from spacr.io import _check_masks
from spacr.utils import adjust_cell_masks, check_mask_folder
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call
from tests.test_cov_object_masks_sam import _settings, force_cpu  # noqa: F401

FIELD = (40, 44)
DIAMETERS = {"cell": 30, "nucleus": 30, "pathogen": 15}


class _WorkingScaleModel:
    """Labels come back at ``30 / diameter`` times the field, never resized."""

    def __init__(self, **kwargs):
        pass

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=None, compute_masks=True, progress=None):
        check_cellpose_eval_call(x, channel_axis)
        scale = 30.0 / diameter if diameter else 1.0
        masks, flows = [], []
        for image in x:
            height = int(round(image.shape[0] * scale))
            width = int(round(image.shape[1] * scale))
            mask = np.zeros((height, width), dtype=np.uint16)
            step = max(1, int(round(scale)))
            mask[4 * step:14 * step, 4 * step:14 * step] = 1
            mask[20 * step:30 * step, 22 * step:34 * step] = 2
            masks.append(mask)
            flows.append([np.zeros((height, width, 3), np.uint8),
                          np.zeros((2, height, width), np.float32),
                          np.zeros((height, width), np.float32),
                          np.zeros((height, width), np.float32)])
        return masks, flows, np.zeros((1, 256), np.float32)


@pytest.fixture
def plate(tmp_path, monkeypatch, force_cpu):  # noqa: F811
    """Two three-channel fields batched the way Mask batches them."""
    src = tmp_path / "masks"
    src.mkdir()
    pixels = np.full((2, *FIELD, 3), .5, dtype=np.float32)
    np.savez(src / "stack_0_norm.npz", data=pixels,
             filenames=["plate1_E01_1_1.npy", "plate1_E01_13_1.npy"])
    monkeypatch.setattr(objects, "cp_models",
                        types.SimpleNamespace(CellposeModel=_WorkingScaleModel))
    return src


def _segment(src, object_type):
    settings = _settings(src, cell_channel=0, nucleus_channel=1,
                         pathogen_channel=2, batch_size=2,
                         cell_diameter=DIAMETERS["cell"],
                         nucleus_diameter=DIAMETERS["nucleus"],
                         pathogen_diameter=DIAMETERS["pathogen"],
                         seg_qc="off")
    objects.generate_cellpose_masks_sam(str(src), settings, object_type,
                                        run_qc=False)
    return src / f"{object_type}_mask_stack"


def test_the_pathogen_scale_differs_from_the_cell_scale_yet_masks_line_up(plate):
    folders = {role: _segment(plate, role) for role in DIAMETERS}
    for role, folder in folders.items():
        for path in sorted(folder.glob("*.npy")):
            mask = np.load(path)
            assert mask.shape == FIELD, f"{role} {path.name} is {mask.shape}"
            assert set(np.unique(mask)) == {0, 1, 2}
    adjust_cell_masks(str(folders["pathogen"]), str(folders["cell"]),
                      str(folders["nucleus"]), n_jobs=1)


def test_a_working_scale_mask_is_sampled_back_with_its_labels():
    mask = np.zeros((80, 88), dtype=np.uint16)
    mask[8:28, 8:28] = 7
    mask[40:60, 44:68] = 3
    field = objects._mask_at_field_size(mask, FIELD, "pathogen")
    assert field.shape == FIELD
    assert set(np.unique(field)) == {0, 3, 7}
    assert field[9, 9] == 7 and field[25, 28] == 3
    assert objects._mask_at_field_size(field, FIELD) is field


def test_a_stale_mask_of_another_size_is_generated_again(tmp_path):
    stack = tmp_path / "stack"
    folder = tmp_path / "masks" / "pathogen_mask_stack"
    stack.mkdir()
    folder.mkdir(parents=True)
    np.save(stack / "plate1_E01_13_1.npy", np.zeros((*FIELD, 3), np.uint16))
    np.save(folder / "plate1_E01_13_1.npy",
            np.zeros((FIELD[0] * 2, FIELD[1] * 2), np.uint16))
    assert check_mask_folder(str(tmp_path), "pathogen_mask_stack") is True
    np.save(folder / "plate1_E01_13_1.npy", np.zeros(FIELD, np.uint16))
    assert check_mask_folder(str(tmp_path), "pathogen_mask_stack") is False


def test_resume_does_not_keep_a_mask_of_another_size(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((FIELD[0] * 2, FIELD[1] * 2), np.uint16))
    np.save(tmp_path / "b.npy", np.zeros(FIELD, np.uint16))
    batch = np.zeros((2, *FIELD, 3), np.float32)
    kept, names = _check_masks(batch, ["a.npy", "b.npy"], str(tmp_path),
                               resume=True)
    assert names == ["a.npy"]
    assert len(kept) == 1
