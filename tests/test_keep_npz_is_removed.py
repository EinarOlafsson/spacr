"""The per-batch scratch NPZ and its keep_npz setting are gone; old settings files still load."""
import inspect

from spacr import pipeline_v2 as PV
from spacr.settings import (expected_types, set_default_settings_preprocess_img_data)
from spacr.validate import RETIRED_SETTINGS, _check_retired_keys


def test_no_default_type_or_parameter_remains():
    assert "keep_npz" not in set_default_settings_preprocess_img_data({})
    assert "keep_npz" not in expected_types
    for function in (PV.stream_masks_from_stack, PV.run_v2):
        parameters = inspect.signature(function).parameters
        assert "keep_npz" not in parameters and "npz_dir" not in parameters


def test_an_old_settings_file_naming_it_is_flagged_once_and_otherwise_ignored():
    assert RETIRED_SETTINGS["keep_npz"] == ""
    problems = [p for p in _check_retired_keys({"keep_npz": True, "batch_fields": 8})
                if p.setting == "keep_npz"]
    assert len(problems) == 1 and "keep_npz" in problems[0].message
    settings = set_default_settings_preprocess_img_data({"keep_npz": True})
    assert settings["keep_npz"] is True
