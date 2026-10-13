"""Strict spot-comparison receipts refuse incomplete or source-changing runs."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _tool():
    """Import the actual optional command without invoking its CLI."""
    path = Path(__file__).parents[1] / 'tools' / 'compare_spotnet_ops.py'
    spec = importlib.util.spec_from_file_location('spot_comparison_f554', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    tool = _tool()
    source = tmp_path / 'real-plane.tif'
    source.write_bytes(b'unchanged acquisition bytes')
    database = tmp_path / 'measurements.db'
    database.write_bytes(b'recorded measurement input')
    library = tmp_path / 'library.csv'
    library.write_text('barcode\nACGT\n')
    envs = {}
    for name in ['spotnet', 'spotiflow']:
        env = tmp_path / name
        cache = env / ('models' if name == 'spotiflow' else 'home/.deepcell/models')
        cache.mkdir(parents=True)
        (cache / 'weights.bin').write_bytes(name.encode())
        (env / 'spacr-backend.json').write_text('{}')
        envs[name] = env
    monkeypatch.setattr(tool.SB, '_backend_state', lambda name: SimpleNamespace(env=str(envs[name])))
    monkeypatch.setattr(tool.SB, '_listen_to_workers', lambda *a: None)
    monkeypatch.setattr(tool.SB, '_WORKERS', {})
    monkeypatch.setattr(tool, '_spotnet_readiness', lambda: (True, 'installed'))
    monkeypatch.setattr(tool, '_spotiflow_readiness', lambda: (True, 'installed'))
    monkeypatch.setattr(tool, '_detect_spots', lambda *a, **k: None)
    monkeypatch.setattr(tool, '_spotiflow_spots', lambda *a, **k: None)
    monkeypatch.setattr(tool.ops_engine, '_load_library', lambda path: ['ACGT'])
    monkeypatch.setattr(tool, '_tasks', lambda *a, **k: [
        {'site': 331, 'planes': {1: [(str(source), 2)]}}])
    seen = []

    def decode(tasks, detector, barcodes):
        """Stand in only for inference; execute real receipt and CLI control flow."""
        seen.append(detector)
        return ({'fields_decoded': 1, 'skipped': {}, 'spots': 3},
                {331: np.array([[1., 2.], [3., 4.]])})

    monkeypatch.setattr(tool, '_run', decode)
    out = tmp_path / 'receipt.json'
    args = ['--tiles', str(tmp_path), '--db', str(database), '--library', str(library),
            '--plate', 'plate1', '--well', 'A1', '--sites', '331',
            '--detectors', 'spotnet', 'spotiflow', '--out', str(out)]
    return tool, args, out, source, seen, envs


def test_complete_cli_records_hashes_selectors_and_all_detectors(comparison):
    tool, args, out, source, seen, _ = comparison
    assert tool.main([*args, '--strict']) == 0
    report = json.loads(out.read_text())
    assert report['acceptance']['complete'] is True
    assert seen == ['native', 'spotnet', 'spotiflow']
    assert report['provenance']['planes'] == [
        {'site': 331, 'cycle': 1, 'channel': 0, 'path': str(source), 'plane': 2}]
    assert report['provenance']['files'][str(source)]['sha256']
    assert len(report['provenance']['files']) == 11
    assert set(report['provenance']['backends']) == {'spotnet', 'spotiflow'}
    assert not list(out.parent.glob('.spacr-comparison-*'))


@pytest.mark.parametrize('strict', [False, True])
def test_missing_backend_remains_exploratory_but_fails_acceptance(comparison, monkeypatch, strict):
    tool, args, out, _, seen, _ = comparison
    monkeypatch.setattr(tool, '_spotiflow_readiness', lambda: (False, 'not installed'))
    if strict:
        with pytest.raises(SystemExit) as error:
            tool.main([*args, '--strict'])
        assert error.value.code == 2
        assert not seen and not out.exists()
    else:
        assert tool.main(args) == 0
        assert json.loads(out.read_text())['spotiflow']['not_run'] == 'not installed'


@pytest.mark.parametrize('failure', ['skipped', 'zero', 'changed', 'new_model'])
def test_incomplete_run_writes_explicit_failed_receipt(comparison, monkeypatch, failure):
    tool, args, out, source, _, envs = comparison
    original = tool._run

    def decode(tasks, detector, barcodes):
        """Introduce a late decode or provenance failure after preflight passes."""
        result, positions = original(tasks, detector, barcodes)
        if detector == 'spotiflow':
            if failure == 'skipped':
                result.update(fields_decoded=0, skipped={'331': 'unreadable'})
            elif failure == 'zero':
                result['spots'] = 0
            elif failure == 'changed':
                source.write_bytes(b'new acquisition content')
            else:
                (envs['spotiflow'] / 'models/new.bin').write_bytes(b'new model')
        return result, positions

    monkeypatch.setattr(tool, '_run', decode)
    assert tool.main([*args, '--strict']) == 2
    report = json.loads(out.read_text())
    assert report['acceptance']['complete'] is False
    assert report['acceptance']['failures']


@pytest.mark.parametrize('alias', ['file', 'symlink', 'hardlink'])
def test_existing_destination_or_input_alias_is_refused_before_decode(comparison, alias):
    tool, args, out, source, seen, _ = comparison
    if alias == 'file':
        out.write_bytes(b'existing receipt')
    elif alias == 'symlink':
        out.symlink_to(source)
    else:
        os.link(source, out)
    before = source.read_bytes(), out.read_bytes()
    with pytest.raises(SystemExit):
        tool.main([*args, '--strict'])
    assert not seen
    assert (source.read_bytes(), out.read_bytes()) == before


def test_atomic_publish_collision_keeps_previous_file_and_cleans_scratch(tmp_path):
    tool = _tool()
    target = tmp_path / 'receipt.json'
    target.write_text('previous')
    with pytest.raises(FileExistsError):
        tool._write_strict_receipt(target, 'new')
    assert target.read_text() == 'previous'
    assert list(tmp_path.iterdir()) == [target]


def test_database_mutation_while_tasks_are_read_is_refused(comparison, monkeypatch):
    tool, args, out, _, seen, _ = comparison
    original = tool._tasks
    database = Path(args[args.index('--db') + 1])

    def tasks(*a, **kw):
        """Emulate concurrent measurement writes before provenance capture."""
        result = original(*a, **kw)
        database.write_bytes(b'changed while selecting objects')
        return result

    monkeypatch.setattr(tool, '_tasks', tasks)
    with pytest.raises(ValueError, match='changed while preparing'):
        tool.main([*args, '--strict'])
    assert not seen and not out.exists()


@pytest.mark.parametrize('symlink', [False, True])
def test_database_with_live_wal_is_refused_even_through_symlink(comparison, symlink):
    tool, args, out, _, seen, _ = comparison
    index = args.index('--db') + 1
    database = Path(args[index])
    Path(str(database) + '-wal').write_bytes(b'pending transaction')
    if symlink:
        alias = database.parent / 'alias.db'
        alias.symlink_to(database)
        args[index] = str(alias)
    with pytest.raises(ValueError, match='Checkpoint'):
        tool.main([*args, '--strict'])
    assert not seen and not out.exists()


def test_spotiflow_threshold_and_model_reach_every_decode_call(comparison, monkeypatch):
    tool, args, out, *_ = comparison
    calls = []
    monkeypatch.setattr(tool, '_spotiflow_spots', lambda image, **k: calls.append(k))
    monkeypatch.setattr(tool.SB, '_spotiflow_spots', None)
    assert tool.main([*args, '--spotiflow-threshold', '0.3', '--spotiflow-model', 'hybiss']) == 0
    tool.SB._spotiflow_spots(np.zeros((4, 4)), threshold=None, model=None)
    assert calls[-1] == {'threshold': 0.3, 'model': 'hybiss'}
    report = json.loads(out.read_text())
    assert (report['spotiflow_threshold'], report['spotiflow_model']) == (0.3, 'hybiss')
