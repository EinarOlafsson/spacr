"""End-to-end contract for instruction 76's independent organelle slots."""

from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pytest


def _two_organelle_field():
    data = np.zeros((16, 16, 7), dtype=np.uint16)
    data[..., 0] = np.arange(256, dtype=np.uint16).reshape(16, 16)
    data[..., 1] = 100
    data[2:14, 2:14, 2] = 1                 # cell: 144 px
    data[5:9, 5:9, 3] = 1                   # nucleus: 16 px
    data[3:5, 3:5, 5] = 1
    data[10:12, 10:12, 5] = 2               # organelle 1: 8 px total
    data[7:10, 10:13, 6] = 7                # organelle 2: 9 px
    return data


def test_role_registry_has_stable_internal_and_numbered_display_names():
    from spacr.object_roles import (ORGANELLE_ROLES, organelle_label,
                                    setting_label)

    assert ORGANELLE_ROLES == (
        "organelle", "organelleb", "organellec", "organelled")
    assert [organelle_label(role) for role in ORGANELLE_ROLES] == [
        "Organelle 1", "Organelle 2", "Organelle 3", "Organelle 4"]
    assert setting_label("organelleb_channel") == "Organelle 2 — Channel"


def test_each_slot_gets_an_independent_type_preset_and_mutable_defaults():
    from spacr.settings import _set_organelle_defaults

    settings = _set_organelle_defaults({
        "organelle_type": "punctate",
        "organelle_method": "adaptive",       # explicit override wins
        "organelleb_type": "tubular",
    })
    assert settings["organelle_method"] == "adaptive"
    assert settings["organelle_morphology"] == "spots"
    assert settings["organelleb_method"] == "ridge"
    assert settings["organelleb_morphology"] == "network"
    assert settings["organelleb_ridge_filter"] == "sato"
    assert settings["organelle_ridge_sigmas"] is not settings[
        "organelleb_ridge_sigmas"]


def test_merge_writer_records_and_crop_reader_uses_all_organelle_planes(
        tmp_path):
    from spacr.crops import (MERGED_LAYOUT_SIDECAR, open_merged_field,
                             read_merged_plane_layout)
    from spacr.io import _load_and_concatenate_arrays

    root = tmp_path / "plate"
    stack = root / "stack"
    stack.mkdir(parents=True)
    stem = "plate1_A01_1.npy"
    intensity = np.zeros((5, 6, 2), dtype=np.uint16)
    intensity[..., 0] = 11
    intensity[..., 1] = 22
    np.save(stack / stem, intensity)

    roles = ("cell", "nucleus", "pathogen", "organelle", "organelleb")
    for index, role in enumerate(roles, start=1):
        folder = root / "masks" / f"{role}_mask_stack"
        folder.mkdir(parents=True)
        np.save(folder / stem, np.full((5, 6), index, dtype=np.uint16))

    _load_and_concatenate_arrays(
        str(root), [0, 1], 0, 0, 0, 0,
        organelle_chann_dims={"organelleb": 1})

    merged = np.load(root / "merged" / stem)
    assert merged.shape == (5, 6, 7)
    assert [int(merged[0, 0, plane]) for plane in range(2, 7)] == [
        1, 2, 3, 4, 5]
    layout = read_merged_plane_layout(str(root / "merged"))
    assert layout["mask_plane_order"] == list(roles)
    assert layout["mask_dims"] == {
        role: 2 + index for index, role in enumerate(roles)}
    assert (root / "merged" / MERGED_LAYOUT_SIDECAR).is_file()

    field = open_merged_field(str(root / "merged" / stem), use_cache=False)
    assert np.all(field.mask_plane("organelleb") == 5)


def test_plane_manifest_overrides_defaults_and_refuses_explicit_conflict(
        tmp_path):
    from spacr.crops import (MERGED_LAYOUT_SIDECAR, PlaneLayoutConflict,
                             reconcile_merged_mask_dims)

    layout = {
        "version": 1,
        "intensity_channels": [0, 1],
        "mask_plane_order": ["cell", "organelle", "organelleb"],
        "mask_dims": {"cell": 2, "organelle": 3, "organelleb": 4},
    }
    (tmp_path / MERGED_LAYOUT_SIDECAR).write_text(
        json.dumps(layout), encoding="utf-8")

    resolved = reconcile_merged_mask_dims(
        {"cell_mask_dim": 4, "organelle_mask_dim": None}, str(tmp_path))
    assert resolved["cell_mask_dim"] == 2
    assert resolved["organelle_mask_dim"] == 3
    assert resolved["organelleb_mask_dim"] == 4
    assert resolved["nucleus_mask_dim"] is None

    with pytest.raises(PlaneLayoutConflict, match="wrong image plane"):
        reconcile_merged_mask_dims(
            {"organelleb_mask_dim": 6}, str(tmp_path),
            explicit_keys={"organelleb_mask_dim"})


def test_merge_resume_refuses_a_changed_organelle_layout(tmp_path):
    from spacr.io import _load_and_concatenate_arrays

    root = tmp_path / "plate"
    (root / "stack").mkdir(parents=True)
    np.save(root / "stack" / "plate1_A01_1.npy",
            np.zeros((4, 4, 1), dtype=np.uint16))
    for role in ("cell", "organelle"):
        folder = root / "masks" / f"{role}_mask_stack"
        folder.mkdir(parents=True)
        np.save(folder / "plate1_A01_1.npy",
                np.ones((4, 4), dtype=np.uint16))
    _load_and_concatenate_arrays(
        str(root), [0], 0, None, None, 0, resume=False)

    # Enabling slot 2 changes the declared layout. A stale folder is enough
    # for the writer to find its mask, but resume must not reuse slot-1 arrays
    # under the new slot map.
    extra = root / "masks" / "organelleb_mask_stack"
    extra.mkdir(parents=True)
    np.save(extra / "plate1_A01_1.npy",
            np.full((4, 4), 2, dtype=np.uint16))
    with pytest.raises(ValueError, match="different plane layout"):
        _load_and_concatenate_arrays(
            str(root), [0], 0, None, None, 0, resume=True,
            organelle_chann_dims={"organelleb": 1})


def test_two_organelle_measurement_is_separate_joinable_and_wide(
        tmp_path, monkeypatch):
    import spacr.measure as measure
    from spacr.feature_dict import parse_column
    from spacr.io import _read_and_merge_data
    from spacr.settings import get_measure_crop_settings

    merged = tmp_path / "merged"
    measurements = tmp_path / "measurements"
    merged.mkdir()
    measurements.mkdir()
    filename = "plate1_A01_1.npy"
    np.save(merged / filename, _two_organelle_field())

    def no_zernike():
        raise ImportError("not needed by this contract test")

    monkeypatch.setattr(measure, "_load_zernike_moments", no_zernike)
    settings = get_measure_crop_settings({
        "src": str(merged),
        "channels": [0, 1],
        "cell_mask_dim": 2,
        "nucleus_mask_dim": 3,
        "pathogen_mask_dim": None,
        "organelle_mask_dim": 5,
        "organelleb_mask_dim": 6,
        "organellec_mask_dim": None,
        "organelled_mask_dim": None,
        "cytoplasm": True,
        "save_measurements": True,
        "save_png": False,
        "save_arrays": False,
        "plot": False,
        "radial_dist": False,
        "homogeneity": False,
        "calculate_correlation": False,
        "distance_gaussian_sigma": 0,
        "summarize_organelles_by": ["cell"],
        "verbose": False,
        "n_jobs": 1,
        "experiment": "multi-organelle-test",
    })
    result = measure._measure_crop_core(0, [], filename, settings)
    assert not isinstance(result[2], int), "worker returned its failure sentinel"

    db = measurements / "measurements.db"
    with sqlite3.connect(db) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"organelle", "organelleb", "cell_organelle_summary"} <= tables
        assert connection.execute("SELECT COUNT(*) FROM organelle").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM organelleb").fetchone()[0] == 1
        assert connection.execute(
            "SELECT cell_id FROM organelleb").fetchone()[0] == 1
        summary = connection.execute(
            "SELECT organelle_summary_organelle_count, "
            "organelle_summary_organelleb_count, "
            "organelle_summary_organelle_total_area, "
            "organelle_summary_organelleb_total_area "
            "FROM cell_organelle_summary").fetchone()
        assert summary == (2, 1, 8.0, 9.0)
        cytoplasm_area = connection.execute(
            "SELECT cytoplasm_area FROM cytoplasm").fetchone()[0]
        assert cytoplasm_area == 111.0

    parsed = parse_column("organelle_summary_organelleb_total_area")
    assert parsed.object_type == "organelleb"
    assert parsed.family == "morphology"

    merged_frame, object_frames = _read_and_merge_data(
        [str(db)], ["cell", "organelle", "organelleb"],
        nuclei_limit=None, pathogen_limit=None)
    assert len(merged_frame) == 1
    assert "organelle_area" in merged_frame
    assert "organelleb_area" in merged_frame
    assert len(object_frames) == 3


def test_enabled_but_empty_slot_writes_zero_parent_summary(tmp_path,
                                                            monkeypatch):
    import spacr.measure as measure
    from spacr.settings import get_measure_crop_settings

    merged = tmp_path / "merged"
    (tmp_path / "measurements").mkdir()
    merged.mkdir()
    data = _two_organelle_field()
    data[..., 6] = 0
    filename = "plate1_A01_1.npy"
    np.save(merged / filename, data)
    monkeypatch.setattr(
        measure, "_load_zernike_moments",
        lambda: (_ for _ in ()).throw(ImportError("disabled")))
    settings = get_measure_crop_settings({
        "src": str(merged), "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": 3,
        "pathogen_mask_dim": None, "organelle_mask_dim": None,
        "organelleb_mask_dim": 6, "cytoplasm": False,
        "save_measurements": True, "save_png": False,
        "save_arrays": False, "plot": False, "radial_dist": False,
        "homogeneity": False, "calculate_correlation": False,
        "distance_gaussian_sigma": 0,
        "summarize_organelles_by": "cell", "verbose": False,
    })
    result = measure._measure_crop_core(0, [], filename, settings)
    assert not isinstance(result[2], int)
    with sqlite3.connect(tmp_path / "measurements" / "measurements.db") as con:
        row = con.execute(
            "SELECT organelle_summary_organelleb_count, "
            "organelle_summary_organelleb_total_area, "
            "organelle_summary_organelleb_fraction "
            "FROM cell_organelle_summary").fetchone()
    assert row == (0, 0, 0.0)


def test_secondary_organelle_crop_uses_its_object_label_in_png_list(tmp_path):
    from spacr.utils import _generate_names, filepaths_to_database

    (tmp_path / "measurements").mkdir()
    image_name, folder, _ = _generate_names(
        "plate1_A01_1", np.array([1]), np.array([1]), np.array([0]),
        str(tmp_path), crop_mode="organelleb", timelapse=False,
        object_id=7)
    assert image_name == "plate1_A01_1_7.png"
    path = os.path.join(folder, "organelleb_png", image_name)
    filepaths_to_database(
        [path], {"timelapse": False}, str(tmp_path), "organelleb")
    with sqlite3.connect(tmp_path / "measurements" / "measurements.db") as con:
        value = con.execute(
            "SELECT organelleb_id FROM png_list").fetchone()[0]
    assert value == "o7"


def test_legacy_measure_settings_leave_secondary_slots_disabled():
    from spacr.object_roles import ORGANELLE_ROLES
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({"organelle_mask_dim": 7})
    assert settings["organelle_mask_dim"] == 7
    assert all(settings[f"{role}_mask_dim"] is None
               for role in ORGANELLE_ROLES[1:])
