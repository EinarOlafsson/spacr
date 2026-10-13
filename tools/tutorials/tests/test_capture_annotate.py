"""Example annotations must preserve both pixels and existing database labels."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'capture_annotate', Path(__file__).resolve().parents[1] / 'capture_annotate.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def test_unchanged_originals_are_accepted():
    recorder.require_preserved({'rows': 'same', 'pixels': ['a', 'b']},
                               {'pixels': ['a', 'b'], 'rows': 'same'})


@pytest.mark.parametrize('after', [
    {'rows': 'changed', 'pixels': ['a', 'b']},
    {'rows': 'same', 'pixels': ['a', 'changed']},
    {'rows': 'same'},
])
def test_changed_originals_are_rejected(after):
    with pytest.raises(RuntimeError, match='changed original'):
        recorder.require_preserved({'rows': 'same', 'pixels': ['a', 'b']}, after)
