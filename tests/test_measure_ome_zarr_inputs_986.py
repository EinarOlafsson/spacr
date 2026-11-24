"""Measure consumes committed native pixels, labels and calibration references."""
from __future__ import annotations

from tests import store_sql
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("zarr")


@pytest.fixture
def _isolated_illumination(monkeypatch):
    """Keep correction hooks and their environment local to one real Measure test."""
    from spacr import illumination, measure_hooks

    for name in (measure_hooks.HOOKS_ENV_VAR, illumination.MODEL_ENV_VAR,
                 illumination.ON_MISSING_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    measure_hooks.clear_measurement_hooks()
    yield
    illumination.disable_illumination_correction()
    measure_hooks.clear_measurement_hooks()


def _project(root, *, dtype=np.uint16, calibration=False):
    """Write retained merged metadata and a committed native field with two objects."""
    from spacr.crops import MERGED_LAYOUT_SIDECAR
    from spacr.image_store import PlateWriter, _merged_field

    merged = root / "merged"
    merged.mkdir(parents=True)
    (merged / MERGED_LAYOUT_SIDECAR).write_text(json.dumps({
        "version": 1, "intensity_channels": [0, 2],
        "mask_plane_order": ["cell", "nucleus"],
        "mask_dims": {"cell": 2, "nucleus": 3}}), encoding="utf-8")
    rng = np.random.default_rng(986)
    original = np.zeros((40, 48, 4), dtype=dtype)
    original[..., :2] = rng.integers(100, 30000, original[..., :2].shape)
    original[4:18, 6:20, 2] = 1
    original[24:36, 28:42, 2] = 2
    original[8:14, 10:16, 3] = 1
    original[27:33, 32:38, 3] = 2
    if calibration:
        original[..., 0] = 100000
        original[..., 1] = 50000
    path = merged / "plate1_A01_1.npy"
    np.save(path, original)
    # The older (983) project layout, images/<plate>.ome.zarr beside the
    # retained merged/*.npy, which item 0aj keeps readable.
    image, labels = _merged_field((str(path), [0, 1], {"cell": 2, "nucleus": 3}))
    PlateWriter(root / "images", "plate1").write_field("A01", 1, image, labels)
    return path, original


def _settings(path, **over):
    """Resolve fresh Measure defaults for the small two-channel project."""
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(path.parent), "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": 3,
        "pathogen_mask_dim": None, "cell_min_size": 0,
        "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "cytoplasm": False,
        "save_png": True, "crop_mode": ["cell", "nucleus"],
        "save_arrays": False, "plot": False, "verbose": False, "n_jobs": 1,
    })
    settings.update(over)
    return settings


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_native_pixels_and_all_label_planes_replace_only_the_selected_source(tmp_path, dtype):
    """Native reconstruction preserves dtype and every plane despite different legacy bytes."""
    from spacr.measure import _read_measure_field

    path, original = _project(tmp_path, dtype=dtype)
    replaced = np.zeros_like(original)
    np.save(path, replaced)
    selected = _read_measure_field(path, {})
    assert selected.dtype == original.dtype
    np.testing.assert_array_equal(selected, original)
    np.testing.assert_array_equal(
        _read_measure_field(path, {"mask_image_format": "npy"}), replaced)


@pytest.mark.parametrize("version, selected, expected", [
    (0, None, "npy"), (944, None, "npy"), (948, None, "npy"),
    (949, None, "ome-zarr"), (944, "ome-zarr", "ome-zarr"),
    (949, "npy", "npy"),
])
def test_saved_measure_source_migration_controls_actual_pixels(tmp_path, version, selected, expected):
    """Saved-era defaults and explicit overrides select the expected pixel source."""
    from spacr.cli import resolve_module, resolve_settings
    from spacr.measure import _read_measure_field
    from spacr.settings import SETTINGS_VERSION_KEY

    path, original = _project(tmp_path)
    replaced = np.zeros_like(original)
    np.save(path, replaced)
    recorded = {SETTINGS_VERSION_KEY: version, "src": str(path.parent)}
    if selected is not None:
        recorded["mask_image_format"] = selected
    saved = tmp_path / "measure.json"
    saved.write_text(json.dumps(recorded), encoding="utf-8")
    settings = resolve_settings(resolve_module("measure"), str(saved))
    assert settings["mask_image_format"] == expected
    np.testing.assert_array_equal(_read_measure_field(path, settings),
                                  original if expected == "ome-zarr" else replaced)


@pytest.mark.parametrize("missing", ["ledger", "commit", "image", "label", "layout"])
def test_incomplete_native_fields_keep_the_legacy_fallback(tmp_path, missing):
    """A field without its complete commit metadata retains usable legacy pixels."""
    from spacr.crops import MERGED_LAYOUT_SIDECAR
    from spacr.measure import _read_measure_field

    path, original = _project(tmp_path)
    store = tmp_path / "images" / "plate1.ome.zarr"
    if missing == "commit":
        (store / "spacr_commits.jsonl").write_text("", encoding="utf-8")
    else:
        removed = {
            "ledger": store / "spacr_commits.jsonl",
            "image": store / "A/1/1/0/zarr.json",
            "label": store / "A/1/1/labels/cell/0/zarr.json",
            "layout": path.parent / MERGED_LAYOUT_SIDECAR,
        }[missing]
        removed.unlink()
    replaced = original.copy()
    replaced[..., :2] = 7
    np.save(path, replaced)
    np.testing.assert_array_equal(_read_measure_field(path, {}), replaced)


def test_native_layout_mismatch_does_not_silently_read_legacy_pixels(tmp_path):
    """Incompatible committed native data fails before quantitative measurement."""
    from spacr.errors import ConfigurationError
    from spacr.measure import _read_measure_field

    path, original = _project(tmp_path)
    np.save(path, original.astype(np.float32))
    with pytest.raises(ConfigurationError, match="shape, dtype or plane layout"):
        _read_measure_field(path, {})


def test_timelapse_keeps_its_timepoint_pixels_without_a_native_time_axis(tmp_path):
    """Time-lapse mode cannot substitute one timeless native field for a frame."""
    from spacr.measure import _read_measure_field

    path, original = _project(tmp_path)
    replaced = np.zeros_like(original)
    np.save(path, replaced)
    np.testing.assert_array_equal(
        _read_measure_field(path, {"timelapse": True}), replaced)


def test_colliding_retained_names_cannot_share_one_native_field(tmp_path):
    """Adding a colliding retained name invalidates the cached unique identity."""
    from spacr.measure import _read_measure_field

    path, original = _project(tmp_path)
    np.testing.assert_array_equal(_read_measure_field(path, {}), original)
    replaced = np.full_like(original, 7)
    second = path.parent / "plate1_A01_1_2.npy"
    np.save(second, np.zeros_like(original))
    np.save(path, replaced)
    np.testing.assert_array_equal(_read_measure_field(path, {}), replaced)
    np.testing.assert_array_equal(_read_measure_field(second, {}), np.zeros_like(original))


@pytest.mark.parametrize("option", ["none", "illumination", "unmix"])
def test_real_measure_native_outputs_match_explicit_legacy_despite_different_npy_pixels(
        tmp_path, option, _isolated_illumination):
    """Real measurements and crops agree through ordinary and optional processing paths."""
    from spacr import illumination
    from spacr.crop_store import read_crop
    from spacr.measure import measure_crop

    outputs = {}
    for source in ("native", "legacy"):
        root = tmp_path / source
        path, original = _project(root)
        settings = _settings(path)
        if option == "illumination":
            settings.update(illumination_correction=True, illumination_estimator="smooth",
                            illumination_qc=False)
        elif option == "unmix":
            settings.update(unmix=True, unmix_controls="0:A01; 1:A01")
        if source == "native":
            np.save(path, np.zeros_like(original))
        else:
            settings["mask_image_format"] = "npy"
        measure_crop(settings)
        with store_sql.connect(root / "measurements/measurements.db") as conn:
            tables = {name: pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
                      for name in ("cell", "nucleus", "ssCell_Crops")}
        assert len(tables["cell"]) == len(tables["nucleus"]) == 2
        outputs[source] = root, tables
        illumination.disable_illumination_correction()
    native_root, native = outputs["native"]
    legacy_root, legacy = outputs["legacy"]
    for role in ("cell", "nucleus"):
        columns = list(native[role].select_dtypes(include="number").columns)
        assert columns
        pd.testing.assert_frame_equal(native[role][columns], legacy[role][columns],
                                      check_exact=True)
    native_rows = native["ssCell_Crops"].sort_values("object_id").to_dict("records")
    legacy_rows = legacy["ssCell_Crops"].sort_values("object_id").to_dict("records")
    assert len(native_rows) == len(legacy_rows) == 4
    for left, right in zip(native_rows, legacy_rows):
        assert left["object_id"] == right["object_id"]
        pixels, labels = read_crop(native_root, left)
        old_pixels, old_labels = read_crop(legacy_root, right)
        np.testing.assert_array_equal(pixels, old_pixels)
        np.testing.assert_array_equal(labels, old_labels)
        y0, x0, y1, x1 = (int(left[key]) for key in
                           ("bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"))
        if option == "none":
            np.testing.assert_array_equal(
                pixels, np.moveaxis(original[y0:y1, x0:x1, :2], -1, 0))


@pytest.mark.parametrize("session", [False, True])
def test_scale_and_calibration_plans_use_native_reference_pixels(tmp_path, session):
    """Plate and session calibration share the source used for field rescaling."""
    from spacr.measure import _prepare_measurement_calibration

    path, original = _project(tmp_path, dtype=np.float32, calibration=True)
    replaced = original.copy()
    replaced[..., :2] = 5
    np.save(path, replaced)
    settings = _settings(path, intensity_calibration=True,
                         intensity_calibration_wells=["A01"],
                         intensity_calibration_statistic="median",
                         intensity_calibration_offset=0)
    if session:
        settings["intensity_calibration_reference_session"] = "plate1"
    plan, calibration = _prepare_measurement_calibration(settings, [path.name])
    assert plan["fields"][path.name]["original_intensity_max"] == 100000
    assert plan["fields"][path.name]["rescale_factor"] == pytest.approx(65535 / 100000)
    grouped = calibration["sessions" if session else "plates"]["plate1"]
    assert grouped["reference_statistic"] == pytest.approx({"0": 65535, "1": 32767.5})
    legacy = dict(settings, mask_image_format="npy")
    plan, calibration = _prepare_measurement_calibration(legacy, [path.name])
    assert plan["fields"][path.name]["original_intensity_max"] == 5
    grouped = calibration["sessions" if session else "plates"]["plate1"]
    assert grouped["reference_statistic"] == {"0": 5, "1": 5}


def test_calibration_identity_tracks_native_bytes_instead_of_retained_npy_pixels(tmp_path):
    """Calibration identity changes when the selected native reference pixels change."""
    import zarr
    from spacr.measure import _calibration_reference_hashes

    path, original = _project(tmp_path)
    settings = _settings(path, intensity_calibration_wells=["A01"])
    initial = _calibration_reference_hashes(settings, [path.name])
    np.save(path, np.zeros_like(original))
    assert _calibration_reference_hashes(settings, [path.name]) == initial
    field = zarr.open_group(str(tmp_path / "images/plate1.ome.zarr/A/1/1"), mode="r+")
    field["0"][0, 0, 0] = 1
    assert _calibration_reference_hashes(settings, [path.name]) != initial


def test_external_bead_files_keep_the_explicit_file_reader(tmp_path):
    """Explicit external references remain file reads despite adjacent native stores."""
    from spacr.measure import _prepare_measurement_calibration

    path, _ = _project(tmp_path / "sample")
    bead_path, bead_stack = _project(tmp_path / "beads")
    bead_stack[..., :2] = 100
    np.save(bead_path, bead_stack)
    settings = _settings(path, intensity_calibration=True,
                         intensity_calibration_wells=["A01"],
                         intensity_calibration_statistic="median",
                         intensity_calibration_offset=0,
                         intensity_calibration_beads=str(bead_path.parent))
    _, calibration = _prepare_measurement_calibration(settings, [path.name])
    reference = calibration["sessions"]["plate1"]["reference_statistic"]
    assert reference == {"0": 100, "1": 100}


def test_nuclear_illumination_uses_selected_native_images_and_masks(tmp_path):
    """Nuclear illumination receives native centroids and raw channel means."""
    from spacr.illumination import _nuclei_table
    from spacr.measure import _measure_plan_loader

    path, original = _project(tmp_path)
    np.save(path, np.zeros_like(original))
    settings = _settings(path, illumination_correction=True,
                         illumination_estimator="nuclear_median")
    table = _nuclei_table(None, 0, paths=[str(path)], mask_dim=3,
                          run_root=str(tmp_path), **_measure_plan_loader(settings))
    assert table["label"].tolist() == [1, 2]
    expected = [float(original[..., 0][original[..., 3] == label].mean())
                for label in (1, 2)]
    np.testing.assert_array_equal(table["nucleus_channel_0_mean_intensity"], expected)
