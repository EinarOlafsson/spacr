"""Malformed relocation records never become process cache paths."""
import json
import os

import pytest

from spacr import run_journal as journal


@pytest.fixture
def record(tmp_path, monkeypatch):
    path = tmp_path / 'cache_locations.json'
    monkeypatch.setattr(journal, '_cache_locations_path', lambda: path)
    monkeypatch.setenv('HF_HOME', '')
    monkeypatch.setenv('TORCH_HOME', '')
    return path


@pytest.mark.parametrize('malformed', [None, False, True, 17, 2.5, [], {}, ['a'], {'path': 'b'}, '', ' \t'])
def test_malformed_path_is_ignored_without_changing_record(record, malformed):
    record.write_text(json.dumps({'HF_HOME': malformed}), encoding="utf-8")
    original = record.read_bytes()
    assert journal._read_cache_locations() == {}
    assert journal._apply_cache_locations() == 0
    assert os.environ['HF_HOME'] == ''
    assert record.read_bytes() == original


def test_valid_sibling_path_applies_while_malformed_path_is_ignored(record, tmp_path):
    target = str(tmp_path / 'model cache')
    record.write_text(json.dumps({'HF_HOME': None, 'TORCH_HOME': target}), encoding="utf-8")
    original = record.read_bytes()
    assert journal._apply_cache_locations() == 1
    assert os.environ['HF_HOME'] == ''
    assert os.environ['TORCH_HOME'] == target
    assert record.read_bytes() == original
