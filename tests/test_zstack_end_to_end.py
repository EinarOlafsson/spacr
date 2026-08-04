"""A z-stack all the way from mask to measurements.db, and what it means.

``tests/test_zstack.py`` covers the mask stage and ``tests/test_measure_3d.py``
covers the measure stage; both build their own input. Nothing joined them, and
the join is where the 3-D path was broken -- ``io._load_and_concatenate_arrays``
assembles the ``merged/`` arrays that measure reads, and it was written for
``(Y, X, C)``. This module runs the whole chain on one volume so the seam is
covered by something other than an assumption.

Two failures were found here and fixed in ``spacr/io.py``:

1. ``np.take(image, channels, axis=2)``. The channel axis is the last one; it
   is *also* axis 2 for a 2-D field, which is why this survived. On a
   ``(Z, Y, X, C)`` stack, axis 2 is X, so selecting channels ``[0, 1]``
   returned a two-pixel-wide image with every channel intact -- and that
   merged, measured, and produced plausible numbers. The silent one.
2. ``if array.ndim == 2: expand_dims(...)``. A ``(Z, Y, X)`` mask needs its
   channel axis appending too; without it the mask reached ``np.concatenate``
   one axis short of its own image. The loud one.

The rest of what is asserted here already worked and is pinned so it keeps
working.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import types

import numpy as np
import pytest

import spacr.io as spacr_io
import spacr.object as spacr_object
import spacr.zstack as zstack


Z_PLANES, HEIGHT, WIDTH, CHANNELS = 5, 48, 48, 2
VOXEL_Z_UM, VOXEL_XY_UM = 2.0, 0.65

#: Two boxes, identical on every plane, so the expected volume is exact.
BOX_A = (slice(6, 18), slice(6, 18))
BOX_B = (slice(26, 40), slice(26, 40))


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


@pytest.fixture
def fake_model(monkeypatch):
    """A Cellpose stand-in that returns the same two boxes on every plane.

    Deterministic on purpose: the point of this module is the plumbing, and a
    segmentation whose output varies would make every measurement assertion a
    tolerance.
    """

    class _Model:
        def __init__(self, **_kwargs):
            self.eval_kwargs = []

        @staticmethod
        def _label(image):
            out = np.zeros(image.shape[:2], dtype=np.uint16)
            out[BOX_A] = 1
            out[BOX_B] = 2
            return out

        def eval(self, x=None, **kwargs):
            self.eval_kwargs.append(dict(kwargs))
            if isinstance(x, list):
                masks = [self._label(np.asarray(i)) for i in x]
                return masks, [np.zeros(m.shape, np.float32) for m in masks], \
                    None, None
            volume = np.asarray(x)
            labels = np.stack([self._label(volume[z])
                               for z in range(volume.shape[0])])
            return labels, [np.zeros(labels.shape, np.float32)], None, None

    monkeypatch.setattr(spacr_object, "cp_models",
                        types.SimpleNamespace(CellposeModel=_Model))
    return _Model


def _settings(stack_dir, **over):
    settings = {
        "src": str(stack_dir),
        "cell_channel": 0,
        "nucleus_channel": 1,
        "pathogen_channel": None,
        "magnification": 20,
        "batch_size": 50,
        "verbose": False,
        "plot": False,
        "save": True,
        "timelapse": False,
        "n_jobs": 1,
        "seg_qc": "off",
        "z_stack": True,
        "z_segmentation_mode": "volumetric",
        "voxel_size_z_um": VOXEL_Z_UM,
        "voxel_size_xy_um": VOXEL_XY_UM,
    }
    settings.update(over)
    return settings


def _measure_settings(merged_dir, **over):
    settings = {
        "src": str(merged_dir),
        "cell_mask_dim": 2, "nucleus_mask_dim": 3, "pathogen_mask_dim": None,
        "channels": [0, 1],
        "cell_chann_dim": 0, "nucleus_chann_dim": 1, "pathogen_chann_dim": None,
        "save_measurements": True, "plot": False, "save_png": False,
        "save_arrays": False, "representative_images": False,
        "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "merge_edge_pathogen_cells": False,
        "timelapse": False, "n_jobs": 1, "verbose": False,
        "experiment": "zstack_e2e",
        "voxel_size_z_um": VOXEL_Z_UM, "voxel_size_xy_um": VOXEL_XY_UM,
        "z_stack": True, "uninfected": True, "cytoplasm": False,
        "include_uninfected": True, "dialate_pngs": False,
        "dialate_png_ratios": [0.2], "crop_mode": ["cell"],
        "png_size": [[224, 224]], "normalize_by": "png", "png_dims": [0, 1],
        "normalize": False, "use_bounding_box": False, "homogeneity": False,
        "distance_gaussian_sigma": 1, "radial_dist": True,
        "calculate_correlation": False, "manders_thresholds": [15, 85, 95],
        "target_intensity_min": 0, "test_mode": False,
    }
    settings.update(over)
    return settings


@pytest.fixture
def measured(tmp_path, fake_model):
    """Run mask -> merge -> measure on one z-stack and hand back the database.

    Deliberately end to end through the real functions rather than through a
    hand-built ``merged/`` array. The bugs this module found were in the
    handoff, which a fixture that builds its own input cannot see.
    """
    from spacr import measure as spacr_measure

    root = tmp_path / "project"
    stack = root / "stack"
    stack.mkdir(parents=True)

    rng = np.random.default_rng(0)
    data = rng.integers(
        200, 4000,
        size=(2, Z_PLANES, HEIGHT, WIDTH, CHANNELS)).astype(np.uint16)
    np.savez(stack / "batch1.npz", data=data,
             filenames=np.array(["plate1_A01_1.npy", "plate1_A01_2.npy"]))

    # -- mask -----------------------------------------------------------
    for object_type in ("cell", "nucleus"):
        spacr_object.generate_cellpose_masks_sam(
            str(stack), _settings(stack), object_type)
        # object.py writes into <src>/<type>_mask_stack; io reads from
        # <project>/masks/<type>_mask_stack. Moving them is what the batch
        # runner does between the two stages.
        (root / "masks").mkdir(exist_ok=True)
        shutil.move(str(stack / f"{object_type}_mask_stack"),
                    str(root / "masks" / f"{object_type}_mask_stack"))

    # -- merge ----------------------------------------------------------
    for index in range(data.shape[0]):
        np.save(stack / f"plate1_A01_{index + 1}.npy", data[index])
    os.remove(stack / "batch1.npz")

    spacr_io._load_and_concatenate_arrays(
        str(root), channels=[0, 1], cell_chann_dim=0, nucleus_chann_dim=1,
        pathogen_chann_dim=None, organelle_chann_dim=None)

    # -- measure --------------------------------------------------------
    spacr_measure.measure_crop(_measure_settings(root / "merged"))
    return root


def _table(root, name):
    connection = sqlite3.connect(str(root / "measurements" / "measurements.db"))
    try:
        columns = [row[1] for row in
                   connection.execute(f'PRAGMA table_info("{name}")')]
        rows = connection.execute(f'SELECT * FROM "{name}"').fetchall()
    finally:
        connection.close()
    return columns, [dict(zip(columns, row)) for row in rows]


# -- the chain --------------------------------------------------------------


def test_the_masks_are_volumes(tmp_path, fake_model):
    """The stage that already worked, pinned. Cellpose is asked for do_3D and
    the anisotropy derived from the voxel size, and a 3-D label array reaches
    disk."""
    stack = tmp_path / "stack"
    stack.mkdir(parents=True)
    rng = np.random.default_rng(1)
    data = rng.integers(
        200, 4000,
        size=(1, Z_PLANES, HEIGHT, WIDTH, CHANNELS)).astype(np.uint16)
    np.savez(stack / "batch1.npz", data=data,
             filenames=np.array(["plate1_A01_1.npy"]))

    spacr_object.generate_cellpose_masks_sam(
        str(stack), _settings(stack), "cell")

    mask = np.load(stack / "cell_mask_stack" / "plate1_A01_1.npy")
    assert mask.shape == (Z_PLANES, HEIGHT, WIDTH)
    assert sorted(np.unique(mask)) == [0, 1, 2]


def test_the_merged_array_keeps_z_and_appends_the_masks(measured):
    """The seam that was broken. ``(Z, Y, X, 2 channels + 2 masks)``.

    Before the fix this raised, and before *that* -- had the mask expansion
    been the only problem -- the channel selection would have handed measure a
    two-pixel-wide image without raising anything at all.
    """
    merged = sorted((measured / "merged").iterdir())
    assert len(merged) == 2
    for path in merged:
        array = np.load(path)
        assert array.shape == (Z_PLANES, HEIGHT, WIDTH, CHANNELS + 2), (
            "z must survive, the channel axis must be the last one, and each "
            "mask must arrive as one extra channel")


def test_the_channel_axis_is_selected_not_the_x_axis(tmp_path):
    """The silent bug, isolated: the fix must be axis=-1 and must leave the
    2-D path bit-identical."""
    volume = np.arange(2 * 4 * 6 * 3).reshape(2, 4, 6, 3)
    assert np.take(volume, [0, 1], axis=-1).shape == (2, 4, 6, 2)
    assert np.take(volume, [0, 1], axis=2).shape == (2, 4, 2, 3)
    field = np.arange(4 * 6 * 3).reshape(4, 6, 3)
    assert np.array_equal(np.take(field, [0, 1], axis=2),
                          np.take(field, [0, 1], axis=-1))


def test_the_measurements_are_written_and_stamped_as_3d(measured):
    """Two objects per field, four rows, every one carrying the units it was
    measured in. Without the stamp a 3-D row and a 2-D row are the same
    schema saying different things."""
    columns, rows = _table(measured, "cell")
    assert len(rows) == 4
    for row in rows:
        assert row["measurement_ndim"] == 3
        assert row["measurement_units"] == "um"
        assert row["n_z"] == Z_PLANES
        assert row["voxel_size_z_um"] == VOXEL_Z_UM
        assert row["voxel_size_xy_um"] == VOXEL_XY_UM
    assert "cell_volume_um3" in columns
    assert "cell_volume_voxels" in columns


def test_the_volume_is_the_volume_that_was_planted(measured):
    """The known answer. Box A is 12x12 px on all 5 planes."""
    _columns, rows = _table(measured, "cell")
    expected_voxels = 12 * 12 * Z_PLANES
    expected_um3 = expected_voxels * VOXEL_XY_UM ** 2 * VOXEL_Z_UM

    volumes = sorted({round(float(row["cell_volume_voxels"])) for row in rows})
    assert expected_voxels in volumes

    match = [row for row in rows
             if round(float(row["cell_volume_voxels"])) == expected_voxels][0]
    assert float(match["cell_volume_um3"]) == pytest.approx(expected_um3)
    # ... and `area` carries the same number under a 2-D name, which is
    # exactly what MEASUREMENT_MEANING_3D exists to say out loud.
    assert float(match["cell_area"]) == pytest.approx(expected_um3)


def test_the_two_dimensional_only_measurements_are_absent(measured):
    """A 2-D perimeter on a volume is not a perimeter, and skimage refuses to
    compute one. What matters is that the column is *missing* rather than
    present and meaningless."""
    columns, _rows = _table(measured, "cell")
    assert "cell_perimeter" not in columns
    assert "cell_eccentricity" not in columns
    assert not any("zernike" in column for column in columns)
    assert not any("homogeneity" in column for column in columns)


def test_the_centroid_is_spelled_with_a_z(measured):
    """A 3-D centroid under 2-D column names (`-0`, `-1`) would be read as
    (y, x) by anything downstream."""
    columns, _rows = _table(measured, "cell")
    for axis in ("z", "y", "x"):
        assert f"cell_channel_0_centroid_weighted_{axis}" in columns


def test_intensity_statistics_survive_unchanged(measured):
    """A mean is a mean over whatever voxels the label covers, so the
    intensity half of the table should be intact."""
    columns, rows = _table(measured, "cell")
    for name in ("cell_channel_0_mean_intensity",
                 "cell_channel_1_mean_intensity",
                 "cell_channel_0_integrated_intensity"):
        assert name in columns
        assert all(np.isfinite(float(row[name])) for row in rows)


# -- what the columns mean --------------------------------------------------


def test_the_meaning_table_matches_what_a_real_run_produced(measured):
    """The vocabulary is checked against a table that exists, not against an
    intention. Anything listed as absent must really be absent, and every
    column claimed to be 3-D-only must really be there."""
    columns, _rows = _table(measured, "cell")
    report = zstack.report_3d_measurements(columns)

    for name in report["absent"]:
        assert f"cell_{name}" not in columns, (
            f"cell_{name} is listed as 2-D-only but a 3-D run wrote it")
    assert set(report["added"]) == {"cell_volume_voxels", "cell_volume_um3"}
    # `area` is the trap: same name, different quantity, different units.
    assert "cell_area" in report["renamed"]
    assert "cell_solidity" in report["same"]
    assert "cell_channel_0_mean_intensity" in report["unknown"]


def test_every_morphology_column_a_3d_run_writes_is_accounted_for(measured):
    """No morphology column may fall through to 'unknown'. A shape
    measurement nobody has thought about is the one that gets pooled across a
    2-D and a 3-D run."""
    columns, _rows = _table(measured, "cell")
    morphology = [
        column for column in columns
        if column.startswith("cell_") and "channel_" not in column
        and column not in ("cell_volume_voxels", "cell_volume_um3")
    ]
    assert morphology, "the run should have written some morphology"
    unexplained = [
        column for column in morphology
        if zstack.describe_3d_measurement(column)["kind"] == "unknown"
    ]
    assert unexplained == []


def test_the_dangerous_columns_say_so():
    """Every entry that keeps its name and changes its meaning must name both
    unit systems, or the table is decoration."""
    for name, entry in zstack.MEASUREMENT_MEANING_3D.items():
        assert entry["kind"] in ("same", "renamed", "absent"), name
        assert entry["note"].strip(), name
        if entry["kind"] == "renamed":
            assert entry["units_2d"] != entry["units_3d"], name
            assert entry["means"] != "-", name


def test_perimeter_is_refused_rather_than_redefined():
    """The specific thing that must not happen: a surface area shipped under
    the name `perimeter`."""
    entry = zstack.describe_3d_measurement("cell_perimeter")
    assert entry["kind"] == "absent"
    assert "surface" in entry["note"]
    assert "surface_um2" in entry["note"], (
        "the note must say where a surface area does live")
    assert "surface_um2" in zstack.VOLUME_STATS_UNITS


def test_an_unknown_name_is_admitted_to_be_unknown():
    entry = zstack.describe_3d_measurement("cell_channel_0_some_new_feature")
    assert entry["kind"] == "unknown"
    assert "not covered" in entry["note"]


def test_the_2d_only_features_explain_themselves():
    for name, reason in zstack.MEASUREMENT_UNAVAILABLE_3D.items():
        assert len(reason.split()) >= 20, name
