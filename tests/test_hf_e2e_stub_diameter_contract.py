"""The synthetic E2E plate explicitly retains its original native-scale input."""
import pytest

from spacr.object_settings_table import _diameter_for_eval, _diameter_setting
from tests.test_hf_e2e_integration import _load_settings_for, _make_stub_settings


@pytest.mark.parametrize("role", ["cell", "nucleus"])
def test_stub_csv_explicitly_requests_native_scale(tmp_path, role):
    from spacr.qt.screens.settings_model import resolve_default_settings

    settings_root = _make_stub_settings(tmp_path)
    settings = _load_settings_for("mask", settings_root, tmp_path / "plate1")
    key = f"{role}_diameter"

    assert resolve_default_settings("mask")[key] is None
    assert settings[key] == "auto"
    diameter = _diameter_setting(settings[key])
    assert diameter == "auto"
    assert _diameter_for_eval(diameter, estimates=False) is None
    assert _diameter_for_eval(diameter, estimates=True) == 0.0
    assert settings["cell_channel"] == 0
    assert settings["nucleus_channel"] == 1
