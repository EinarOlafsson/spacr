"""External images + label masks become a normal Measure project."""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import tifffile

from spacr import external_masks as em
from spacr.validate import ERROR, validate_settings


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")
    return path


def _inputs(tmp_path):
    yy, xx = np.indices((32, 32))
    images = tmp_path / "images"
    cells = tmp_path / "cell_masks"
    nuclei = tmp_path / "nucleus_masks"
    _write(images / "fov001_C1.tif", yy * 32 + xx)
    _write(images / "fov001_C2.tif", (xx * 17 + yy * 3) % 4096)
    cell_mask = np.zeros((32, 32), dtype=np.uint16)
    cell_mask[3:29, 3:29] = 1
    nucleus_mask = np.zeros((32, 32), dtype=np.uint16)
    nucleus_mask[10:20, 11:21] = 1
    _write(cells / "fov001_cell_mask.tif", cell_mask)
    _write(nuclei / "fov001_nucleus_mask.tif", nucleus_mask)
    return images, cells, nuclei


def test_detection_uses_pixels_and_folder_or_filename_object_names(tmp_path):
    images, cells, nuclei = _inputs(tmp_path)

    groups = em.detect_inputs([images, cells, nuclei])

    assert sum(len(group.paths) for group in groups) == 4
    image_groups = [group for group in groups if group.role == "image"]
    mask_groups = [group for group in groups if group.role == "mask"]
    assert sum(len(group.paths) for group in image_groups) == 2
    assert {group.object_type for group in mask_groups} == {
        "cell", "nucleus",
    }
    assert all(group.confidence > 0.8 for group in groups)


def test_plan_pairs_fields_and_assigns_mask_planes_after_image_channels(
        tmp_path):
    images, cells, nuclei = _inputs(tmp_path)

    plan = em.plan_external_masks({
        "inputs": [
            group.to_dict()
            for group in em.detect_inputs([images, cells, nuclei])
        ],
        "dst": str(tmp_path / "project"),
        "layout": "flat",
    })

    assert plan.ok, plan.summary()
    assert plan.n_channels == 2
    assert plan.object_types == ["cell", "nucleus"]
    assert plan.mask_dims == {"cell": 2, "nucleus": 3}
    assert len(plan.stems) == 1
    stem = plan.stems[0]
    assert plan.masks["cell"][stem].path.endswith("fov001_cell_mask.tif")
    assert plan.masks["nucleus"][stem].path.endswith(
        "fov001_nucleus_mask.tif")


def test_plan_refuses_an_unassigned_mask_group(tmp_path):
    images, cells, _nuclei = _inputs(tmp_path)
    groups = em.detect_inputs([images, cells])
    next(group for group in groups if group.role == "mask").object_type = None

    plan = em.plan_external_masks({
        "inputs": [group.to_dict() for group in groups],
        "dst": str(tmp_path / "project"),
        "layout": "flat",
    })

    assert not plan.ok
    assert any("choose whether" in message for message in plan.errors)


def test_plan_refuses_unprojected_z_planes_that_would_overwrite_a_field(
        tmp_path):
    images, cells, _nuclei = _inputs(tmp_path)
    gradient = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    volume = np.stack([gradient, gradient + 100])
    tifffile.imwrite(
        images / "volume_C3.tif", volume, photometric="minisblack",
        metadata={"axes": "ZYX"})
    groups = em.detect_inputs([images, cells])

    plan = em.plan_external_masks({
        "inputs": [group.to_dict() for group in groups],
        "dst": str(tmp_path / "project"),
        "layout": "flat",
        "z_handling": "keep",
    })

    assert not plan.ok
    assert any("Multiple time or Z planes" in message
               for message in plan.errors)


def test_run_builds_merged_stacks_then_delegates_to_measure(
        tmp_path, monkeypatch):
    images, cells, nuclei = _inputs(tmp_path)
    settings = {
        "inputs": [
            group.to_dict()
            for group in em.detect_inputs([images, cells, nuclei])
        ],
        "dst": str(tmp_path / "project"),
        "layout": "flat",
        "crop_mode": ["cell", "nucleus"],
    }
    plan = em.plan_external_masks(settings)
    received = {}

    def fake_measure_crop(measure_settings):
        received.update(measure_settings)
        project = os.path.dirname(measure_settings["src"])
        db = os.path.join(project, "measurements", "measurements.db")
        os.makedirs(os.path.dirname(db), exist_ok=True)
        with sqlite3.connect(db) as connection:
            for table in ("cell", "nucleus", "cytoplasm", "png_list"):
                connection.execute(
                    f'CREATE TABLE "{table}" (object_label INTEGER)')
        data = os.path.join(project, "data", "plate1", "cell_png")
        os.makedirs(data, exist_ok=True)

    monkeypatch.setattr("spacr.measure.measure_crop", fake_measure_crop)
    result = em.run_external_masks(plan, settings)

    assert os.path.isfile(result.db_path)
    assert {"cell", "nucleus", "cytoplasm", "png_list"} <= set(
        result.tables)
    assert received["src"] == os.path.join(result.destination, "merged")
    assert received["channels"] == [0, 1]
    assert received["cell_mask_dim"] == 2
    assert received["nucleus_mask_dim"] == 3
    assert received["pathogen_mask_dim"] is None
    assert received["crop_mode"] == ["cell", "nucleus"]
    merged = np.load(result.merged[0])
    assert merged.shape == (32, 32, 4)
    assert merged.dtype == np.uint16
    assert set(np.unique(merged[..., 2])) == {0, 1}
    assert set(np.unique(merged[..., 3])) == {0, 1}


def test_only_supplied_object_tables_are_required(tmp_path, monkeypatch):
    images, cells, _nuclei = _inputs(tmp_path)
    settings = {
        "inputs": [
            group.to_dict()
            for group in em.detect_inputs([images, cells])
        ],
        "dst": str(tmp_path / "project"),
        "layout": "flat",
        "cytoplasm": False,
    }
    plan = em.plan_external_masks(settings)

    def fake_measure_crop(measure_settings):
        project = os.path.dirname(measure_settings["src"])
        db = os.path.join(project, "measurements", "measurements.db")
        os.makedirs(os.path.dirname(db), exist_ok=True)
        with sqlite3.connect(db) as connection:
            connection.execute(
                "CREATE TABLE cell (object_label INTEGER)")
            connection.execute(
                "CREATE TABLE png_list (object_label INTEGER)")

    monkeypatch.setattr("spacr.measure.measure_crop", fake_measure_crop)
    result = em.run_external_masks(plan, settings)

    assert set(result.tables) == {"cell", "png_list"}


def test_preflight_understands_reviewed_input_groups(tmp_path):
    images, cells, _nuclei = _inputs(tmp_path)
    settings = em.default_settings({
        "inputs": [
            group.to_dict()
            for group in em.detect_inputs([images, cells])
        ],
        "dst": str(tmp_path / "project"),
        "layout": "flat",
    })

    problems = validate_settings(settings, "external_masks")

    assert not [problem for problem in problems
                if problem.severity == ERROR], problems
    assert not any("unknown app" in problem.message for problem in problems)
