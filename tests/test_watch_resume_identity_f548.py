"""Completed snapshots warn on changed file identity without silently rerunning."""
import json
from pathlib import Path

import pytest

from spacr import core
from tests.test_watch_folder_and_analyse import Recorder, _name
from tests.test_watch_nested_f548 import _fast, _write
from tests.test_watch_snapshot_handoff_f548 import _change_pixels


def _completed(folder):
    """Complete one two-channel real TIFF field and return its ledger path."""
    paths = [_write(folder, _name('A01', channel), channel) for channel in (1, 2)]
    result = core._watch_folder_and_analyse(_fast(folder), Recorder())
    assert len(result['done']) == 1
    return paths, Path(result['ledger'])


@pytest.mark.parametrize('replace', [False, True])
def test_done_field_warns_on_identity_change_with_equal_size_mtime(tmp_path, capsys, replace):
    paths, ledger_path = _completed(tmp_path)
    before = json.loads(ledger_path.read_text())['fields']
    result_path = tmp_path / 'spacr_watch/merged/plate1_A01_0001_001.npy'
    output_before = result_path.read_bytes()
    capsys.readouterr()
    _change_pixels(paths[0], 91, replace=replace)
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert 'changed after it was analysed' in capsys.readouterr().out
    assert not recorder.calls and result['done'] == ['plate1_A01_0001_001']
    after = json.loads(ledger_path.read_text())['fields']
    assert after == before  # previous input evidence must not be overwritten
    assert result_path.read_bytes() == output_before


def test_unchanged_snapshot_resume_stays_quiet_without_pixel_reads(tmp_path, monkeypatch, capsys):
    _paths, ledger_path = _completed(tmp_path)
    before = json.loads(ledger_path.read_text())['fields']
    capsys.readouterr()

    def no_decode(path):
        """A completed unchanged source needs stat checks, not another TIFF decode."""
        raise AssertionError(f'Unexpected image decode: {path}')

    monkeypatch.setattr(core, '_watch_unreadable', no_decode)
    recorder = Recorder()
    assert core._watch_folder_and_analyse(_fast(tmp_path), recorder)['done']
    assert not recorder.calls and 'changed after it was analysed' not in capsys.readouterr().out
    assert json.loads(ledger_path.read_text())['fields'] == before


@pytest.mark.parametrize('changed', [False, True])
def test_legacy_done_ledger_retains_size_mtime_policy(tmp_path, capsys, changed):
    paths, ledger_path = _completed(tmp_path)
    ledger = json.loads(ledger_path.read_text())
    for entry in ledger['fields'].values():
        entry.pop('source_identity')
        entry.pop('snapshot_sha256')
    ledger_path.write_text(json.dumps(ledger))
    before = ledger['fields']
    capsys.readouterr()
    if changed:
        with paths[0].open('ab') as handle:
            handle.write(b'changed source marker')
    recorder = Recorder()
    assert core._watch_folder_and_analyse(_fast(tmp_path), recorder)['done']
    output = capsys.readouterr().out
    assert ('changed after it was analysed' in output) == changed
    assert not recorder.calls
    assert json.loads(ledger_path.read_text())['fields'] == before
