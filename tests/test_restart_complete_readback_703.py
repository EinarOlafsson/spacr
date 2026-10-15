"""Restart approval requires the complete saved interface to read back."""
import json

import pytest

from spacr import restart_state


@pytest.mark.parametrize('field,replacement', [
    ('settings', {'src': '/wrong', 'nested': {'threshold': 99}}),
    ('running', []), ('run_folders', ['/wrong-output']),
    ('saved', '2000-01-01T00:00:00+00:00'), ('version', 999),
])
def test_same_module_with_different_persisted_state_refuses_restart(
        tmp_path, monkeypatch, field, replacement):
    monkeypatch.setenv('SPACR_HOME', str(tmp_path))
    real_replace = restart_state.os.replace

    def altered_replace(source, target):
        real_replace(source, target)
        document = json.loads(target.read_text())
        assert document['module'] == 'measure'
        document[field] = replacement
        target.write_text(json.dumps(document, indent=2))

    monkeypatch.setattr(restart_state.os, 'replace', altered_replace)
    assert restart_state.save(
        module='measure', settings={'src': '/images', 'nested': {'threshold': 2}},
        running=[{'module': 'mask', 'seconds': 12}], run_folders=['/output'],
        saved='2026-10-15T05:00:00+00:00') is None


def test_complete_valid_record_keeps_existing_json_conversion(tmp_path, monkeypatch):
    monkeypatch.setenv('SPACR_HOME', str(tmp_path))
    result = restart_state.save(module='measure', settings={
        'src': tmp_path / 'images', 'channels': (0, 2), 'selected': frozenset([2, 1])})
    assert result is not None
    assert restart_state.peek()['settings'] == {
        'src': str(tmp_path / 'images'), 'channels': [0, 2], 'selected': [1, 2]}
