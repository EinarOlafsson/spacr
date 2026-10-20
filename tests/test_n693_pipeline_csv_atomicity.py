"""Pipeline CSV snapshot failures preserve the previous native settings copy."""
import os
from pathlib import Path
import stat

import pandas as pd
import pytest


@pytest.mark.parametrize('existing', [False, True])
def test_interrupted_csv_stream_preserves_snapshots_and_cleans_stage(tmp_path, monkeypatch, capsys, existing):
    from spacr.utils import save_settings

    folder = tmp_path / 'settings'
    folder.mkdir()
    csv_path, json_path = folder / 'sample.csv', folder / 'sample.json'
    if existing:
        csv_path.write_bytes(b'previous CSV snapshot')
    json_path.write_bytes(b'previous JSON snapshot')

    def interrupted(frame, path, *args, **kwargs):
        with open(path, 'w') as stream:
            stream.write('Key,Value\npartial,')
            stream.flush()
        raise OSError('interrupted native CSV stream')

    monkeypatch.setattr(pd.DataFrame, 'to_csv', interrupted)
    assert save_settings({'src': str(tmp_path), 'value': 3}, name='sample') is None
    assert 'Warning: could not save settings' in capsys.readouterr().out
    assert csv_path.read_bytes() == b'previous CSV snapshot' if existing else not csv_path.exists()
    assert json_path.read_bytes() == b'previous JSON snapshot'
    assert sorted(p.name for p in folder.iterdir()) == (['sample.csv', 'sample.json'] if existing else ['sample.json'])


@pytest.mark.parametrize('existing', [False, True])
def test_refused_csv_publication_preserves_both_snapshots(tmp_path, monkeypatch, capsys, existing):
    from spacr.utils import save_settings

    folder = tmp_path / 'settings'
    folder.mkdir()
    csv_path, json_path = folder / 'sample.csv', folder / 'sample.json'
    if existing:
        csv_path.write_bytes(b'previous CSV snapshot')
    json_path.write_bytes(b'previous JSON snapshot')
    original_replace = os.replace

    def refused(source, target, *args, **kwargs):
        if Path(target) == csv_path:
            raise PermissionError('CSV publication refused')
        return original_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(os, 'replace', refused)
    save_settings({'src': str(tmp_path), 'value': 3}, name='sample')
    assert 'Warning: could not save settings' in capsys.readouterr().out
    assert csv_path.read_bytes() == b'previous CSV snapshot' if existing else not csv_path.exists()
    assert json_path.read_bytes() == b'previous JSON snapshot'
    assert sorted(p.name for p in folder.iterdir()) == (['sample.csv', 'sample.json'] if existing else ['sample.json'])


def test_success_preserves_csv_bytes_and_existing_mode(tmp_path):
    from spacr.utils import save_settings

    folder = tmp_path / 'settings'
    folder.mkdir()
    target = folder / 'sample.csv'
    target.write_text('old', encoding="utf-8")
    target.chmod(0o640)
    settings = {'src': str(tmp_path), 'value': 3, 'channels': [0, 1], 'none': None, 'text': 'μm,a'}
    expected = pd.DataFrame(list(settings.items()), columns=['Key', 'Value']).to_csv(index=False)
    save_settings(settings, name='sample')
    assert target.read_text(encoding="utf-8") == expected
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sorted(p.name for p in folder.iterdir()) == ['sample.csv', 'sample.json']


def test_existing_csv_symlink_is_preserved(tmp_path):
    from spacr.utils import load_settings, save_settings

    folder = tmp_path / 'settings'
    folder.mkdir()
    target = tmp_path / 'actual.csv'
    target.write_text('old', encoding="utf-8")
    target.chmod(0o640)
    link = folder / 'sample.csv'
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(str(error))
    save_settings({'src': str(tmp_path), 'value': 3}, name='sample')
    assert link.is_symlink()
    assert load_settings(str(target))['value'] == 3
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sorted(p.name for p in folder.iterdir()) == ['sample.csv', 'sample.json']


def test_read_only_csv_still_refuses_without_updating_json(tmp_path, capsys):
    from spacr.utils import save_settings

    folder = tmp_path / 'settings'
    folder.mkdir()
    csv_path, json_path = folder / 'sample.csv', folder / 'sample.json'
    csv_path.write_bytes(b'previous CSV snapshot')
    json_path.write_bytes(b'previous JSON snapshot')
    csv_path.chmod(0o440)
    if os.access(csv_path, os.W_OK):
        pytest.skip('Current account can write read-only files')
    try:
        save_settings({'src': str(tmp_path), 'value': 3}, name='sample')
        assert 'Warning: could not save settings' in capsys.readouterr().out
        assert csv_path.read_bytes() == b'previous CSV snapshot'
        assert json_path.read_bytes() == b'previous JSON snapshot'
        assert stat.S_IMODE(csv_path.stat().st_mode) == 0o440
    finally:
        csv_path.chmod(0o640)


def test_literal_environment_token_path_refuses_before_leaving_private_stage(tmp_path, monkeypatch, capsys):
    from spacr import tabular
    from spacr.utils import save_settings

    monkeypatch.setenv('SPACR_CSV_LITERAL_TOKEN', 'redirected')
    root = tmp_path / '$SPACR_CSV_LITERAL_TOKEN'
    folder = root / 'settings'
    folder.mkdir(parents=True)
    csv_path, json_path = folder / 'sample.csv', folder / 'sample.json'
    csv_path.write_bytes(b'previous CSV snapshot')
    json_path.write_bytes(b'previous JSON snapshot')
    monkeypatch.setattr(tabular, 'write_table', lambda *a, **k: pytest.fail('Unsafe expanded path reached table writer'))
    save_settings({'src': str(root), 'value': 3}, name='sample')
    assert 'Warning: could not save settings' in capsys.readouterr().out
    assert csv_path.read_bytes() == b'previous CSV snapshot'
    assert json_path.read_bytes() == b'previous JSON snapshot'
    assert sorted(p.name for p in folder.iterdir()) == ['sample.csv', 'sample.json']
    assert not (tmp_path / 'redirected').exists()
