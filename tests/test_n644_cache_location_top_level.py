"""Invalid relocation records stay inert while explicit environments win."""
import json
import os

import pytest

from spacr import run_journal as journal
from tests.test_n644_cache_path_metadata_types import record


@pytest.mark.parametrize('payload', [None, False, 15, [], ['HF_HOME'], 'HF_HOME'])
def test_non_mapping_top_level_stays_inert(record, payload):
    record.write_text(json.dumps(payload), encoding="utf-8")
    original = record.read_bytes()
    assert journal._read_cache_locations() == {}
    assert journal._apply_cache_locations() == 0
    assert record.read_bytes() == original


def test_embedded_nul_path_is_ignored(record):
    record.write_text(json.dumps({'HF_HOME': 'invalid\x00path'}), encoding="utf-8")
    original = record.read_bytes()
    assert journal._apply_cache_locations() == 0
    assert os.environ['HF_HOME'] == ''
    assert record.read_bytes() == original


def test_external_override_remains_authoritative_with_malformed_record(record, tmp_path, monkeypatch):
    record.write_text(json.dumps({'HF_HOME': {}, 'TORCH_HOME': str(tmp_path / 'recorded')}), encoding="utf-8")
    monkeypatch.setenv('HF_HOME', str(tmp_path / 'explicit-hf'))
    monkeypatch.setenv('TORCH_HOME', str(tmp_path / 'explicit-torch'))
    original = record.read_bytes()
    assert journal._apply_cache_locations() == 0
    assert os.environ['HF_HOME'] == str(tmp_path / 'explicit-hf')
    assert os.environ['TORCH_HOME'] == str(tmp_path / 'explicit-torch')
    assert record.read_bytes() == original
