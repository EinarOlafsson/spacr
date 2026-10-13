"""Failure-reporting tests; real installation evidence lives in its receipt."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import check_pip_installation as module
    monkeypatch.setattr(module.sys, 'argv', ['probe', '--stage', str(tmp_path)])
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 1024**3))
    metadata = {'info': {'version': '1.5.0.5', 'requires_python': '>=3.9'}, 'urls': []}
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps(metadata).encode()))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['cwd'].is_relative_to(tmp_path)
        assert 'PYTHONPATH' not in kwargs['env']
        kwargs['stdout'].write('actual subprocess output placeholder for unit test\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', run)
    return module, calls, run


def receipt(tmp_path):
    paths = list(tmp_path.glob('installation_runs/*/receipt.json'))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def test_all_six_successful_steps_are_required(probe, tmp_path):
    module, calls, _ = probe
    module.main()
    result = receipt(tmp_path)
    assert len(calls) == len(result['steps']) == 6
    assert result['accepted'] is True
    assert all(row['returncode'] == 0 and row['completed'] for row in result['steps'])
    assert 'no GUI or other-platform success claimed' in result['scope']


@pytest.mark.parametrize('failed_index', range(6))
def test_a_failed_command_cannot_be_counted_as_an_installation(
        probe, tmp_path, monkeypatch, failed_index):
    module, calls, successful_run = probe

    def fail_at(command, **kwargs):
        result = successful_run(command, **kwargs)
        if len(calls) - 1 == failed_index:
            result.returncode = 23
        return result

    monkeypatch.setattr(module.subprocess, 'run', fail_at)
    with pytest.raises(RuntimeError, match='failed; evidence retained'):
        module.main()
    result = receipt(tmp_path)
    assert result['accepted'] is False
    assert len(calls) == failed_index + 1
    assert result['steps'][-1]['returncode'] == 23


def test_timeout_keeps_an_unfinished_receipt(probe, tmp_path, monkeypatch):
    module, calls, successful_run = probe

    def timeout(command, **kwargs):
        successful_run(command, **kwargs)
        if len(calls) == 3:
            raise module.subprocess.TimeoutExpired(command, 2400)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', timeout)
    with pytest.raises(module.subprocess.TimeoutExpired):
        module.main()
    result = receipt(tmp_path)
    assert result['accepted'] is False
    assert result['steps'][-1]['failure'] == 'timeout'
    assert result['steps'][-1]['completed'] is False


def test_insufficient_space_stops_before_creating_an_environment(
        probe, tmp_path, monkeypatch):
    module, calls, _ = probe
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=39 * 1024**3))
    with pytest.raises(RuntimeError, match='40 GiB'):
        module.main()
    assert calls == []
    assert not (tmp_path / 'installation_runs').exists()


def test_unexpected_release_identity_stops_before_pip(probe, tmp_path, monkeypatch):
    module, calls, _ = probe
    metadata = {'info': {'version': 'not-a-release', 'requires_python': '>=3.9'}, 'urls': []}
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps(metadata).encode()))
    with pytest.raises(ValueError, match='numeric public release'):
        module.main()
    assert calls == []
    assert receipt(tmp_path)['accepted'] is False
