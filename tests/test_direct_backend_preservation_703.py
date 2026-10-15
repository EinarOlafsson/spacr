"""Directly configured independent backends survive primary app retirement."""
import os

import pytest

from spacr import install_cleanup as cleanup


def installation(tmp_path, monkeypatch, spelling='absolute'):
    home = tmp_path / 'home'
    root = home / 'spacr-old'
    prefix = root / 'venv'
    code = prefix / 'bin' / 'python'
    code.parent.mkdir(parents=True)
    code.write_bytes(b'old primary runtime')
    (prefix / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    backends = prefix / 'spacr-backends'
    interpreter = backends / 'cellpose' / 'bin' / 'python'
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b'independent backend runtime')
    weights = backends / 'model.bin'
    weights.write_bytes(b'preserved model bytes\x00\xff')
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(tmp_path)
    configured = str(backends)
    if spelling == 'relative':
        configured = os.path.relpath(backends, tmp_path)
    elif spelling == 'home':
        configured = '~/' + str(backends.relative_to(home))
    elif spelling == 'padded':
        configured = '  ' + configured + '  '
    elif spelling == 'alias':
        alias = home / 'spacr-backend-alias'
        alias.symlink_to(backends, target_is_directory=True)
        configured = str(alias)
    machine = cleanup._Machine(platform='linux', environ={
        'HOME': str(home), 'SPACR_BACKENDS_DIR': configured,
    }, fs_root=str(tmp_path), running_prefix=None)
    record = cleanup.InstallRecord(kind='installer', layout='linux-online',
                                   platform='linux', root=str(root))
    return root, prefix, code, backends, interpreter, weights, machine, record


@pytest.mark.parametrize('spelling', ['absolute', 'relative', 'home', 'padded', 'alias'])
def test_retirement_preserves_direct_backend_runtime_and_models(tmp_path, monkeypatch, spelling):
    root, _prefix, code, _backends, interpreter, weights, machine, record = installation(
        tmp_path, monkeypatch, spelling,
    )
    report = cleanup.remove_install(record, system=machine)
    assert report.ok
    assert interpreter.is_file(), 'Normal retirement deleted a configured backend interpreter'
    assert interpreter.read_bytes() == b'independent backend runtime'
    assert weights.read_bytes() == b'preserved model bytes\x00\xff'
    assert not code.exists()
    assert not (root / '.venv.spacr-update-lock').exists()


@pytest.mark.parametrize('spelling', ['absolute', 'home'])
def test_replacement_refuses_configured_backends_inside_primary_runtime(tmp_path, monkeypatch, spelling):
    _root, prefix, code, _backends, interpreter, weights, machine, _record = installation(
        tmp_path, monkeypatch, spelling,
    )
    with pytest.raises(ValueError, match='configured user data inside the runtime'):
        cleanup._runtime_staging_allowed(str(prefix), machine)
    assert code.read_bytes() == b'old primary runtime'
    assert interpreter.read_bytes() == b'independent backend runtime'
    assert weights.read_bytes() == b'preserved model bytes\x00\xff'


def test_unconfigured_embedded_software_is_still_retired(tmp_path, monkeypatch):
    root, _prefix, _code, _backends, interpreter, _weights, machine, record = installation(
        tmp_path, monkeypatch,
    )
    del machine.environ['SPACR_BACKENDS_DIR']
    assert cleanup.remove_install(record, system=machine).ok
    assert not interpreter.exists()
    assert not root.exists()


def test_external_backends_do_not_keep_unrelated_old_runtime(tmp_path, monkeypatch):
    root, _prefix, _code, backends, _interpreter, _weights, machine, record = installation(
        tmp_path, monkeypatch,
    )
    outside = tmp_path / 'spacr-backends'
    backends.rename(outside)
    machine.environ['SPACR_BACKENDS_DIR'] = str(outside)
    assert cleanup.remove_install(record, system=machine).ok
    assert not root.exists()
    assert (outside / 'cellpose/bin/python').read_bytes() == b'independent backend runtime'
    assert (outside / 'model.bin').read_bytes() == b'preserved model bytes\x00\xff'


def test_explicit_purge_retains_its_separate_backend_deletion_contract(tmp_path, monkeypatch):
    _root, _prefix, _code, backends, _interpreter, _weights, machine, _record = installation(
        tmp_path, monkeypatch,
    )
    assert str(backends) in cleanup._purge_targets(machine)
    report = cleanup._purge(machine)
    assert report.ok
    assert not backends.exists()
