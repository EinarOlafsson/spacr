"""Contracts for spaCR's intentionally small top-level Python interface."""

from __future__ import annotations

import spacr


def test_top_level_all_is_a_small_stable_facade():
    assert spacr.__all__ == [
        "__version__",
        "download_models",
        "MaskConfig",
        "MeasureConfig",
        "run_mask",
        "run_measure",
    ]
    assert "core" not in spacr.__all__
    assert len(spacr._SUBMODULES) == len(set(spacr._SUBMODULES))


def test_existing_lazy_module_access_remains_compatible():
    assert spacr.core.__name__ == "spacr.core"
    assert callable(spacr.core.preprocess_generate_masks)


def test_mask_config_expands_through_pipeline_defaults():
    config = spacr.MaskConfig(
        "/data/plate01", cell_channel=0, nucleus_channel=1,
    )
    settings = config.to_settings()
    assert settings["src"] == "/data/plate01"
    assert settings["cell_channel"] == 0
    assert settings["nucleus_channel"] == 1
    assert settings["save"] is True


def test_measure_config_copies_mutable_values():
    config = spacr.MeasureConfig(
        "/data/plate01/merged", crop_mode=("cell", "nucleus"),
    )
    first = config.to_settings()
    second = config.to_settings()
    assert first["crop_mode"] == ["cell", "nucleus"]
    first["crop_mode"].append("pathogen")
    assert second["crop_mode"] == ["cell", "nucleus"]


def test_advanced_settings_cannot_repeat_typed_fields():
    config = spacr.MaskConfig(
        "/data/plate01", cell_channel=0, extra={"cell_channel": 2},
    )
    try:
        config.to_settings()
    except ValueError as exc:
        assert "extra repeats typed setting" in str(exc)
    else:  # pragma: no cover - assertion gives a clearer failure than pytest
        raise AssertionError("duplicate typed setting was accepted")
