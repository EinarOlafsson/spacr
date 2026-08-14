from __future__ import annotations

import numpy as np
import pytest

from spacr.intensity_rescale import (
    PLAN_SETTINGS_KEY,
    build_plate_plan,
    fallback_record,
    resolve_record,
)
from spacr.measure import _promote_merged_to_uint16
from spacr.validate import describe_resources


SETTINGS = {
    "cell_mask_dim": 1,
    "nucleus_mask_dim": None,
    "pathogen_mask_dim": None,
    "organelle_mask_dim": None,
    "timelapse": False,
}


def _field(top: float, dtype=np.float32):
    data = np.zeros((3, 3, 2), dtype=dtype)
    data[..., 0] = top
    data[1, 1, 1] = 7
    return data


def test_raw_fields_on_one_plate_share_the_largest_fields_factor(tmp_path):
    low = "plate1_A01_1.npy"
    high = "plate1_A01_2.npy"
    np.save(tmp_path / low, _field(40000))
    np.save(tmp_path / high, _field(131070))

    plan = build_plate_plan(tmp_path, [low, high], SETTINGS)
    low_record, high_record = plan["fields"][low], plan["fields"][high]
    assert low_record["rescale_scope"] == high_record["rescale_scope"] == "plate"
    assert low_record["rescale_factor"] == high_record["rescale_factor"] \
        == pytest.approx(0.5)
    assert low_record["plate_intensity_max"] == high_record["plate_intensity_max"] \
        == 131070

    low_out, _ = _promote_merged_to_uint16(
        np.load(tmp_path / low), SETTINGS,
        rescale_factor=low_record["rescale_factor"])
    high_out, _ = _promote_merged_to_uint16(
        np.load(tmp_path / high), SETTINGS,
        rescale_factor=high_record["rescale_factor"])
    assert low_out[..., 0].max() == 20000
    assert high_out[..., 0].max() == 65535
    assert low_out[1, 1, 1] == high_out[1, 1, 1] == 7


def test_plate_factors_are_independent(tmp_path):
    names = ["plate1_A01_1.npy", "plate2_A01_1.npy"]
    np.save(tmp_path / names[0], _field(131070))
    np.save(tmp_path / names[1], _field(262140))
    plan = build_plate_plan(tmp_path, names, SETTINGS)
    assert plan["fields"][names[0]]["rescale_factor"] == pytest.approx(0.5)
    assert plan["fields"][names[1]]["rescale_factor"] == pytest.approx(0.25)


def test_normalised_float_has_the_fixed_factor_not_the_raw_plate_factor(tmp_path):
    normal = "plate1_A01_1.npy"
    raw = "plate1_A01_2.npy"
    np.save(tmp_path / normal, _field(0.75))
    np.save(tmp_path / raw, _field(131070))
    plan = build_plate_plan(tmp_path, [normal, raw], SETTINGS)
    assert plan["fields"][normal]["rescale_scope"] == "fixed_normalized"
    assert plan["fields"][normal]["rescale_factor"] == 65535
    assert plan["fields"][raw]["rescale_factor"] == pytest.approx(0.5)


def test_failed_prepass_is_explicit_and_worker_fallback_is_noncomparable(tmp_path):
    name = "plate1_A01_1.npy"
    (tmp_path / name).write_bytes(b"not an npy")
    plan = build_plate_plan(tmp_path, [name], SETTINGS)
    assert name in plan["failures"] and name not in plan["fields"]

    record = fallback_record(_field(131070), name, SETTINGS)
    assert record["rescale_scope"] == "field_fallback"
    assert record["rescale_factor"] == pytest.approx(0.5)
    assert record["comparable_within_plate"] is False


def test_changed_field_does_not_reuse_a_stale_plate_plan(tmp_path):
    name = "plate1_A01_1.npy"
    original = _field(131070)
    np.save(tmp_path / name, original)
    settings = dict(SETTINGS)
    settings[PLAN_SETTINGS_KEY] = build_plate_plan(tmp_path, [name], settings)

    changed = _field(262140)
    record = resolve_record(changed, name, settings)
    assert record["rescale_scope"] == "field_fallback"
    assert record["rescale_factor"] == pytest.approx(0.25)
    assert record["comparable_within_plate"] is False


def test_measure_preflight_reports_oversized_data_before_run(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "plate1_A01_1.npy", _field(131070))
    card = describe_resources({"src": str(merged), **SETTINGS}, "measure")
    assert "INTENSITY PREFLIGHT" in card
    assert "exceed the uint16 ceiling 65535" in card
    assert "plate-wide factor" in card
    assert "intensity_rescale" in card
