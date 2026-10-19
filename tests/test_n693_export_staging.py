"""Interrupted export streams must not replace saved settings with fragments."""
import builtins
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from spacr.cli import write_settings_file


@pytest.mark.parametrize('exists', [False, True])
@pytest.mark.parametrize('broken', ['csv', 'json'])
def test_partial_export_stream_preserves_both_previous_files(tmp_path, monkeypatch, exists, broken):
    target = tmp_path / 'settings.csv'
    sibling = target.with_suffix('.json')
    before = {target: b'Key,Value\nold,yes\n', sibling: b'{"old": "yes"}'}
    if exists:
        for path, data in before.items():
            path.write_bytes(data)
    original_open = builtins.open
    if broken == 'csv':
        def interrupted(frame, path, *args, **kwargs):
            with original_open(path, 'w') as stream:
                stream.write('Key,Va')
            raise OSError('interrupted CSV stream')
        monkeypatch.setattr(pd.DataFrame, 'to_csv', interrupted)
    else:
        class Interrupted:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def write(self, text):
                self.stream.write(text[:3])
                raise OSError('interrupted JSON stream')
            def __getattr__(self, name):
                return getattr(self.stream, name)
        def opened(path, mode='r', *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            return Interrupted(stream) if str(path).endswith('.json') and 'w' in mode else stream
        monkeypatch.setattr(builtins, 'open', opened)
    with pytest.raises(OSError, match='interrupted'):
        write_settings_file({'src': '/new', 'channels': [0, 2]}, target)
    for path, data in before.items():
        if exists:
            assert path.read_bytes() == data
        else:
            assert not path.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        ['settings.csv', 'settings.json'] if exists else [])


def test_successful_export_preserves_modes_and_symlink_destinations(tmp_path):
    targets = tmp_path / 'targets'
    targets.mkdir()
    csv = targets / 'actual.csv'
    js = targets / 'actual.json'
    csv.write_text('old csv')
    js.write_text('old json')
    csv.chmod(0o640)
    js.chmod(0o600)
    alias = tmp_path / 'settings.csv'
    alias.symlink_to(csv)
    alias.with_suffix('.json').symlink_to(js)
    settings = {'src': '/new', 'nested': {'channels': [0, 2]}}
    assert write_settings_file(settings, alias) == [str(alias), str(alias.with_suffix('.json'))]
    assert alias.is_symlink() and alias.with_suffix('.json').is_symlink()
    assert csv.stat().st_mode & 0o777 == 0o640
    assert js.stat().st_mode & 0o777 == 0o600
    assert json.loads(js.read_text(encoding="utf-8")) == settings


@pytest.mark.parametrize('readonly', ['csv', 'json'])
def test_readonly_destination_refuses_before_either_file_is_published(tmp_path, readonly):
    csv = tmp_path / 'settings.csv'
    js = csv.with_suffix('.json')
    csv.write_bytes(b'previous csv')
    js.write_bytes(b'previous json')
    (csv if readonly == 'csv' else js).chmod(0o440)
    with pytest.raises(PermissionError):
        write_settings_file({'src': '/new'}, csv)
    assert csv.read_bytes() == b'previous csv'
    assert js.read_bytes() == b'previous json'


def test_first_publication_failure_preserves_both_files(tmp_path, monkeypatch):
    csv = tmp_path / 'settings.csv'
    js = csv.with_suffix('.json')
    csv.write_bytes(b'previous csv')
    js.write_bytes(b'previous json')
    real_replace = os.replace
    def refused(source, target):
        if Path(target) == csv:
            raise OSError('publication refused')
        return real_replace(source, target)
    monkeypatch.setattr(os, 'replace', refused)
    with pytest.raises(OSError, match='publication refused'):
        write_settings_file({'src': '/new'}, csv)
    assert csv.read_bytes() == b'previous csv'
    assert js.read_bytes() == b'previous json'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['settings.csv', 'settings.json']


def test_new_exports_retain_normal_umask_mode(tmp_path):
    csv = tmp_path / 'settings.csv'
    previous = os.umask(0o027)
    try:
        write_settings_file({'src': '/new'}, csv)
    finally:
        os.umask(previous)
    assert csv.stat().st_mode & 0o777 == 0o640
    assert csv.with_suffix('.json').stat().st_mode & 0o777 == 0o640
