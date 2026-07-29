"""Regression tests for machine-independent shipped settings."""

from pathlib import Path


def test_legacy_barcode_defaults_use_packaged_references():
    from spacr.settings import (
        bundled_barcode_path,
        get_map_barcodes_default_settings,
    )

    configured = get_map_barcodes_default_settings({})
    assert configured["grna"] == bundled_barcode_path("grna")
    assert configured["barcodes"] == bundled_barcode_path("column")
    assert Path(configured["grna"]).is_file()
    assert Path(configured["barcodes"]).is_file()


def test_regression_metadata_has_no_workstation_default():
    from spacr.settings import get_perform_regression_default_settings

    configured = get_perform_regression_default_settings({})
    assert configured["metadata_files"] == []


def test_settings_source_contains_no_developer_home_paths():
    import spacr.settings as settings

    source = Path(settings.__file__).read_text(encoding="utf-8")
    assert "/home/carruthers" not in source
