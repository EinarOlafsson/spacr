"""A fixed Convert map supplies exact watch companions without channel guessing."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from spacr import core, convert
from tests.test_watch_folder_and_analyse import Recorder, _name
from tests.test_watch_nested_f548 import _fast, _write


def _map(folder, channels=(1, 4), wells=('A01',)):
    """Write the actual Convert required schema for canonical static targets."""
    rows = [dict(target=_name(well, channel), source=f'raw/{well}/C{channel}.tif',
                 plate='plate1', well=well, field=1, channel=channel, z=1, t=1)
            for well in wells for channel in channels]
    _rows(folder, rows)
    return rows


def _rows(folder, rows):
    """Write an intentionally mutable test map using Convert's schema constants."""
    with (folder / convert.MAP_FILENAME).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=convert._REQUIRED_MAP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _ledger_bytes(folder):
    """Return saved ledger bytes so rejected resumes prove no state mutation."""
    return (folder / 'spacr_watch/watch_ledger.json').read_bytes()


def test_exact_mapped_channels_wait_despite_matching_legacy_count(tmp_path):
    _map(tmp_path)
    for channel in (1, 2):
        _write(tmp_path, _name('A01', channel))
    recorder = Recorder()
    settings = dict(_fast(tmp_path), channels=[0, 3])
    first = core._watch_folder_and_analyse(settings, recorder)
    assert first['incomplete'] and not recorder.calls
    _write(tmp_path, _name('A01', 4))
    assert core._watch_folder_and_analyse(settings, recorder)['incomplete']
    assert not recorder.calls  # unexpected C02 is not silently included
    (tmp_path / _name('A01', 2)).unlink()
    result = core._watch_folder_and_analyse(settings, recorder)
    assert result['done'] == ['plate1_A01_0001_001']
    assert recorder.calls[0][1] == [_name('A01', 1), _name('A01', 4)]
    saved = json.loads(_ledger_bytes(tmp_path))
    assert saved['conversion_map_sha256'] == hashlib.sha256(
        (tmp_path / convert.MAP_FILENAME).read_bytes()).hexdigest()
    assert core._watch_folder_and_analyse(settings, recorder)['done'] == result['done']
    assert len(recorder.calls) == 1


def test_missing_entire_mapped_field_and_unexpected_field_are_reported(tmp_path):
    _map(tmp_path, channels=(1,), wells=('A01', 'A02'))
    _write(tmp_path, _name('A01', 1))
    _write(tmp_path, _name('A03', 1))
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert result['done'] == ['plate1_A01_0001_001']
    assert result['incomplete'] == ['plate1_A02_0001_001', 'plate1_A03_0001_001']
    assert len(recorder.calls) == 1  # map overrides the legacy minimum two channels


@pytest.mark.parametrize('mutation', ['changed', 'removed', 'introduced'])
def test_manifest_resume_change_preserves_previous_results(tmp_path, mutation):
    if mutation != 'introduced':
        _map(tmp_path, (1, 2))
    for channel in (1, 2):
        _write(tmp_path, _name('A01', channel))
    core._watch_folder_and_analyse(_fast(tmp_path), Recorder())
    before = _ledger_bytes(tmp_path)
    if mutation == 'removed':
        (tmp_path / convert.MAP_FILENAME).unlink()
    else:
        _map(tmp_path, (1, 2, 3))
    recorder = Recorder()
    with pytest.raises(ValueError, match='differs from the saved watch record'):
        core._watch_folder_and_analyse(_fast(tmp_path), recorder)
    assert not recorder.calls and _ledger_bytes(tmp_path) == before
    assert (tmp_path / 'spacr_watch/merged/plate1_A01_0001_001.npy').is_file()


def test_change_between_ready_fields_stops_before_second_analysis(tmp_path):
    _map(tmp_path, (1,), ('A01', 'A02'))
    for well in ('A01', 'A02'):
        _write(tmp_path, _name(well, 1))
    recorder = Recorder()

    def analyse(directory, settings):
        """Change acquisition metadata after the first completed field."""
        recorder(directory, settings)
        with (tmp_path / convert.MAP_FILENAME).open('a') as handle:
            handle.write('\n')

    with pytest.raises(ValueError, match='changed since this watch started'):
        core._watch_folder_and_analyse(_fast(tmp_path), analyse)
    assert len(recorder.calls) == 1
    ledger = json.loads(_ledger_bytes(tmp_path))
    assert list(ledger['fields']) == ['plate1_A01_0001_001']
    assert ledger['fields']['plate1_A01_0001_001']['status'] == 'done'


@pytest.mark.parametrize('case', ['duplicate', 'conflict', 'path', 'bad_channel', 'z', 't', 'blank'])
def test_bad_manifest_fails_before_watch_outputs(tmp_path, case):
    rows = _map(tmp_path)
    if case == 'duplicate':
        rows.append(dict(rows[0]))
    elif case == 'conflict':
        rows.append(dict(rows[0], source='different.tif'))
    elif case == 'path':
        rows[0]['target'] = '../' + rows[0]['target']
    elif case == 'bad_channel':
        rows[0]['channel'] = 2
    elif case in ('z', 't'):
        rows[0][case] = 2
        rows[0]['target'] = convert.target_name('plate1', 'A01', 1, 1,
                                                **{case: 2})
    elif case == 'blank':
        rows = []
    _rows(tmp_path, rows)
    with pytest.raises(ValueError, match='invalid conversion_map'):
        core._watch_folder_and_analyse(_fast(tmp_path), Recorder())
    assert not (tmp_path / 'spacr_watch').exists()


@pytest.mark.parametrize('case', ['schema', 'symlink', 'large', 'incompatible_pattern'])
def test_unsupported_metadata_refused_without_fallback(tmp_path, case):
    _map(tmp_path)
    path = tmp_path / convert.MAP_FILENAME
    settings = _fast(tmp_path)
    if case == 'schema':
        path.write_text('target,channel\na.tif,1\n')
    elif case == 'symlink':
        path.rename(tmp_path / 'elsewhere.csv')
        path.symlink_to(tmp_path / 'elsewhere.csv')
    elif case == 'large':
        with path.open('ab') as handle:
            handle.truncate(16 * 1024 * 1024 + 1)
    else:
        settings['custom_regex'] = r'(?P<unrelated>.*)'
    with pytest.raises(ValueError, match='conversion_map|conversion map'):
        core._watch_folder_and_analyse(settings, Recorder())
    assert not (tmp_path / 'spacr_watch').exists()


def test_real_convert_output_is_accepted_and_source_pixels_unchanged(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'converted'
    originals = [_write(raw / 'A01', f'field01_C{channel}.tif', channel)
                 for channel in (1, 4)]
    before = {path: path.read_bytes() for path in originals}
    result = convert.convert_folder({'src': str(raw), 'dst': str(output), 'preview_rows': 0})
    assert result.is_complete
    before.update({path: path.read_bytes() for path in output.glob('*.tif')})
    before[Path(result.map_path)] = Path(result.map_path).read_bytes()
    recorder = Recorder()
    watched = core._watch_folder_and_analyse(_fast(output), recorder)
    assert len(watched['done']) == 1 and len(recorder.calls) == 1
    assert not watched['incomplete'] and not watched['failed']
    assert all(path.read_bytes() == content for path, content in before.items())


def test_mapped_companions_in_separate_folders_cannot_complete(tmp_path):
    _map(tmp_path, (1, 2))
    _write(tmp_path / 'first', _name('A01', 1))
    _write(tmp_path / 'second', _name('A01', 2))
    recorder = Recorder()
    assert core._watch_folder_and_analyse(_fast(tmp_path), recorder)['incomplete']
    assert not recorder.calls


def test_map_symlink_swap_at_open_is_not_followed(tmp_path, monkeypatch):
    """Replace the checked path at open time and prove no target bytes are read."""
    import os

    _map(tmp_path)
    path = tmp_path / convert.MAP_FILENAME
    target = tmp_path / 'external.csv'
    target.write_text('do not read this target')
    original_open = os.open

    def swap_then_open(name, flags, *args, **kwargs):
        """Inject a pathname race at the file-descriptor open boundary."""
        if Path(name) == path:
            path.unlink()
            path.symlink_to(target)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', swap_then_open)
    with pytest.raises(ValueError, match='conversion_map.csv'):
        core._watch_map_bytes(str(tmp_path))
    assert target.read_text() == 'do not read this target'
