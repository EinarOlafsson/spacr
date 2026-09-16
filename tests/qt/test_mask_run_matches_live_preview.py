"""418 acceptance: persisted Mask output and cached Live Preview agree."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from spacr.cli import load_settings_file
from spacr.object import generate_cellpose_masks_sam
from spacr.qt.widgets.live_preview import LivePreviewPanel
from tests.test_cov_object_masks_sam import fake_model, force_cpu
from tests.test_v1_raw_intensity_filtering import (
    _expected_mask, _raw_field, _settings, _write_npz,
)


@pytest.mark.parametrize("minimum,maximum,min_area,kept", [
    (0, 0, 0, (1, 2)),
    (10, 0, 0, (2,)),
    (0, 10, 0, (1,)),
    (50, 50, 36, (2,)),
    (50, 50, 37, ()),
    (51, 60, 0, ()),
])
def test_saved_mask_and_preview_filter_the_same_original_values(
        tmp_path, qtbot, fake_model, force_cpu,
        minimum, maximum, min_area, kept):
    raw = _raw_field((32, 32), (5, 50))
    src = tmp_path / "masks"
    originals = tmp_path / "stack"
    originals.mkdir()
    filename = "plate1_A01_1.npy"
    np.save(originals / filename, raw)
    _write_npz(src, (raw[..., [0, 2]] / 1000).astype(np.float32)[None], [filename])
    settings = _settings(src, cell_min_intensity=minimum,
                         cell_max_intensity=maximum, cell_min_area=min_area)
    generate_cellpose_masks_sam(str(src), settings, "cell")
    persisted = np.load(src / "cell_mask_stack" / filename)
    expected = _expected_mask(raw.shape[:2], kept)
    np.testing.assert_array_equal(persisted, expected)

    preview = LivePreviewPanel(threaded=False)
    qtbot.addWidget(preview)
    try:
        preview.apply_settings(settings)
        preview._image = raw.copy()
        original_labels = _expected_mask(raw.shape[:2], (1, 2))
        preview._on_worker_done({"cell": original_labels.copy()}, None)
        np.testing.assert_array_equal(preview._masks["cell"], persisted)
        np.testing.assert_array_equal(preview._raw_masks["cell"], original_labels)
        np.testing.assert_array_equal(preview._image, raw)
        # Relaxing every bound restores exactly the segmentation, including
        # cases where the filtered run and preview both had no objects left.
        controls = preview._compartment_widgets["cell"]
        for suffix in ("min_area", "min_intensity", "max_intensity"):
            controls[suffix].setValue(0)
        np.testing.assert_array_equal(preview._masks["cell"], original_labels)
        assert preview._worker is None
        assert len(fake_model["model"].eval_inputs) == 1
    finally:
        preview.shutdown()


@pytest.mark.parametrize("header", [
    ("Key", "Value"), ("setting_key", "setting_value"),
], ids=["gui-csv", "run-csv"])
@pytest.mark.parametrize("minimum,kept", [
    (0, (1, 2)), (10, (2,)),
], ids=["bounds-off", "minimum-active"])
def test_loaded_retired_controls_leave_saved_masks_identical_to_current_settings(
        tmp_path, fake_model, force_cpu, header, minimum, kept):
    """An old saved file actually runs; its enabled withdrawn flags do nothing."""
    retired = {
        "cell_minimum_area_to_split": 1,
        "cell_min_watershed_distance": 1,
        "cell_intensity_threshold": 9999,
        "cell_intensity_merge": True,
        "cell_intensity_split": True,
    }
    raw = _raw_field((32, 32), (5, 50))
    normalized = (raw[..., [0, 2]] / 1000).astype(np.float32)[None]
    expected = _expected_mask(raw.shape[:2], kept)
    filename = "plate1_A01_1.npy"
    persisted = {}
    for variant, extra_settings in (("legacy", retired), ("current", {})):
        # Separate roots prevent resume from reusing the other run's output.
        root = tmp_path / variant
        src = root / "masks"
        originals = root / "stack"
        originals.mkdir(parents=True)
        raw_path = originals / filename
        np.save(raw_path, raw)
        original_bytes = raw_path.read_bytes()
        _write_npz(src, normalized, [filename])
        settings = _settings(src, cell_min_intensity=minimum,
                             cell_max_intensity=0, **extra_settings)
        csv_path = root / "settings.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(settings.items())

        loaded = load_settings_file(str(csv_path))
        assert loaded == settings
        if variant == "legacy":
            assert loaded["cell_intensity_merge"] is True
            assert loaded["cell_intensity_split"] is True
        # Pass the loaded mapping directly: no test-side stripping of old keys.
        generate_cellpose_masks_sam(loaded["src"], loaded, "cell")

        model = fake_model["model"]
        assert model.gpu is False
        assert len(model.eval_inputs) == 1
        np.testing.assert_array_equal(
            np.asarray(model.eval_inputs[0]), normalized[..., [1, 0]])
        persisted[variant] = np.load(src / "cell_mask_stack" / filename)
        np.testing.assert_array_equal(persisted[variant], expected)
        assert raw_path.read_bytes() == original_bytes

    np.testing.assert_array_equal(persisted["legacy"], persisted["current"])
