"""A display-only walkthrough must not silently change a scientific setting."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'capture_settings', Path(__file__).resolve().parents[1] / 'capture_settings.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def test_identical_nested_inputs_and_settings_are_accepted():
    before = {'level': 'both', 'paired_data': [{'plate': 'plate1', 'database': None}]}
    after = {'paired_data': [{'database': None, 'plate': 'plate1'}], 'level': 'both'}
    recorder.require_unchanged_settings(before, after)


@pytest.mark.parametrize('after', [
    {'guide_permutations': 999}, {}, {'guide_permutations': 199, 'plot': False},
])
def test_changed_missing_and_added_settings_are_rejected(after):
    with pytest.raises(RuntimeError, match='changed settings'):
        recorder.require_unchanged_settings({'guide_permutations': 199}, after)
