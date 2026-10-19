"""Normal retirement preserves each documented direct cache consumer's bytes."""
import ast
import os
from pathlib import Path
import sys

import pytest

from spacr import install_cleanup as cleanup


CONSUMERS = [
    ('SPACR_MODEL_CACHE', 'model_zoo.py', '_model_cache_root', None),
    ('SPACR_NEWS_CACHE', 'updater.py', 'news_cache_path', 'ENV_NEWS_CACHE'),
    ('SPACR_SPACE_CACHE', 'qt/space.py', 'cache_dir', 'ENV_CACHE_DIR'),
    ('SPACR_SOUND_CACHE', 'qt/sound_synth.py', 'sound_cache_root', 'CACHE_ENV'),
    ('SPACR_ICON_CACHE', 'qt/iconset.py', 'icon_cache_dir', 'ENV_ICON_CACHE'),
    ('MPLCONFIGDIR', 'qt/app.py', '_matplotlib_cache_dir', None),
]


def consumer_root(consumer, home):
    variable, source, symbol, constant = consumer
    path = Path(cleanup.__file__).parent / source
    node = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == symbol)
    namespace = {'os': os, 'sys': sys, 'Path': Path,
                 '_spacr_home': lambda: home / '.spacr'}
    if constant:
        namespace[constant] = variable
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    root = namespace[symbol]()
    return root.parent if variable == 'SPACR_NEWS_CACHE' else root


def installation(tmp_path, monkeypatch, consumer, spelling):
    home = tmp_path / 'home'
    root = home / 'spacr-old'
    prefix = root / 'venv'
    code = prefix / 'bin/python'
    code.parent.mkdir(parents=True)
    code.write_bytes(b'primary app code')
    (prefix / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(prefix)
    variable = consumer[0]
    configured = str(prefix / 'cache')
    if spelling == 'relative':
        configured = 'cache'
    elif spelling == 'padded':
        configured = ('  ' + configured + '  ' if variable in
                      {'SPACR_MODEL_CACHE', 'SPACR_SOUND_CACHE'} else '  cache  ')
    elif spelling == 'home':
        configured = '~/spacr-old/venv/cache' if variable == 'SPACR_MODEL_CACHE' else '~/cache'
    monkeypatch.setenv(variable, configured)
    cache = consumer_root(consumer, home).absolute()
    cache.mkdir(parents=True)
    payload = cache / 'retained.bin'
    payload.write_bytes(b'user-configured cache bytes\x00\xff')
    machine = cleanup._Machine(platform='linux', environ={
        'HOME': str(home), variable: configured,
    }, fs_root=str(tmp_path), running_prefix=None)
    record = cleanup.InstallRecord(kind='installer', layout='linux-online',
                                   platform='linux', root=str(root))
    return root, prefix, code, cache, payload, machine, record


@pytest.mark.parametrize('consumer', CONSUMERS, ids=lambda row: row[0])
@pytest.mark.parametrize('spelling', ['absolute', 'relative', 'padded', 'home'])
def test_real_consumer_cache_survives_primary_app_retirement(tmp_path, monkeypatch, consumer, spelling):
    root, _prefix, code, _cache, payload, machine, record = installation(
        tmp_path, monkeypatch, consumer, spelling,
    )
    report = cleanup.remove_install(record, system=machine)
    assert report.ok
    assert payload.is_file(), f'{consumer[0]} cache resolved by its real consumer was deleted'
    assert payload.read_bytes() == b'user-configured cache bytes\x00\xff'
    assert not code.exists()
    assert not (root / '.venv.spacr-update-lock').exists()


@pytest.mark.parametrize('consumer', CONSUMERS, ids=lambda row: row[0])
@pytest.mark.parametrize('spelling', ['absolute', 'relative', 'padded', 'home'])
def test_replacement_refuses_actual_consumer_cache_in_runtime(tmp_path, monkeypatch, consumer, spelling):
    _root, prefix, code, _cache, payload, machine, _record = installation(
        tmp_path, monkeypatch, consumer, spelling,
    )
    with pytest.raises(ValueError, match='configured user data inside the runtime'):
        cleanup._runtime_staging_allowed(str(prefix), machine)
    assert code.read_bytes() == b'primary app code'
    assert payload.read_bytes() == b'user-configured cache bytes\x00\xff'


def test_unconfigured_cache_named_app_files_are_still_retired(tmp_path, monkeypatch):
    root, _prefix, _code, _cache, payload, machine, record = installation(
        tmp_path, monkeypatch, CONSUMERS[0], 'absolute',
    )
    del machine.environ['SPACR_MODEL_CACHE']
    assert cleanup.remove_install(record, system=machine).ok
    assert not payload.exists() and not root.exists()


def test_external_direct_caches_do_not_keep_primary_app_or_enter_purge(tmp_path, monkeypatch):
    root, _prefix, _code, cache, _payload, machine, record = installation(
        tmp_path, monkeypatch, CONSUMERS[0], 'absolute',
    )
    cache.rename(tmp_path / 'external-cache')
    outside = tmp_path / 'external-cache'
    for variable, *_rest in CONSUMERS:
        machine.environ[variable] = str(outside)
    assert cleanup.remove_install(record, system=machine).ok
    assert not root.exists()
    assert (outside / 'retained.bin').read_bytes() == b'user-configured cache bytes\x00\xff'
    assert str(outside) not in cleanup._purge_targets(machine)
    assert cleanup._purge(machine).ok
    assert (outside / 'retained.bin').read_bytes() == b'user-configured cache bytes\x00\xff'
