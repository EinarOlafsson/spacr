"""Catalogue mutation serialization must not stall registry readers during pip."""
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr import plugins
from tests.test_plugin_catalogue import _catalogue


@pytest.fixture
def home(tmp_path, monkeypatch):
    target = tmp_path / "plugins"
    monkeypatch.setenv("SPACR_PLUGIN_HOME", str(target))
    monkeypatch.delenv("SPACR_PLUGIN_MODULES", raising=False)
    monkeypatch.delenv("SPACR_DISABLE_PLUGINS", raising=False)
    plugins.reload_plugins()
    yield target
    plugins._forget_modules_under(str(target / "site" / "catalogue_probe"))
    monkeypatch.delenv("SPACR_PLUGIN_HOME")
    plugins.reload_plugins()


def test_registry_reads_continue_during_install_and_mutations_serialize(tmp_path, home):
    source = tmp_path / 'catalogue'
    source.mkdir()
    _catalogue(source)
    plugins._install_from_catalogue('toxo_infection', source)
    installing = threading.Event()
    release = threading.Event()
    read_done = threading.Event()
    remove_done = threading.Event()
    failures = []

    def pip_runner(command, **kwargs):
        installing.set()
        assert release.wait(10), 'test did not release pip'
        target = command[command.index('--target') + 1]
        with zipfile.ZipFile(command[-1]) as wheel:
            wheel.extractall(target)
        return SimpleNamespace(returncode=0)

    def install():
        try:
            plugins._install_from_catalogue('catalogue_probe', source, runner=pip_runner)
        except Exception as exc:
            failures.append(exc)

    def read():
        try:
            assert plugins.get_app('catalogue_probe_app') is None
            plugins.discover_plugins()
            assert 'toxo_infection' in plugins._catalogue_installed()
            read_done.set()
        except Exception as exc:
            failures.append(exc)

    def remove():
        try:
            plugins._uninstall_from_catalogue('toxo_infection')
            remove_done.set()
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=install)
    worker.start()
    readers = []
    try:
        assert installing.wait(5)
        for target in (read, remove):
            task = threading.Thread(target=target)
            readers.append(task)
            task.start()
        assert read_done.wait(2), 'registry read was blocked by pip'
        assert not remove_done.wait(.1), 'second mutation bypassed serialization'
    finally:
        release.set()
        worker.join(10)
        for task in readers:
            task.join(10)
    assert not failures
    assert not worker.is_alive()
    assert remove_done.is_set()
    assert set(plugins._catalogue_installed()) == {'catalogue_probe'}
    assert plugins.get_app('catalogue_probe_app') is not None


def test_slow_registry_reload_keeps_previous_snapshot_available(monkeypatch, home):
    previous = plugins._registry()
    started = threading.Event()
    release = threading.Event()
    read_done = threading.Event()
    replacement = plugins._Registry()

    def build():
        started.set()
        assert release.wait(10)
        return replacement

    monkeypatch.setattr(plugins, '_build_registry', build)
    worker = threading.Thread(target=plugins.reload_plugins)
    reader = threading.Thread(target=lambda: (plugins.discover_plugins(), read_done.set()))
    worker.start()
    try:
        assert started.wait(5)
        reader.start()
        assert read_done.wait(2)
        assert plugins._registry() is previous
    finally:
        release.set()
        worker.join(10)
        reader.join(10)
    assert plugins._registry() is replacement


def test_failed_pip_keeps_installed_record_and_registry(tmp_path, home):
    source = tmp_path / 'catalogue'
    source.mkdir()
    _catalogue(source)
    old = plugins._registry()

    def failed(*args, **kwargs):
        return SimpleNamespace(returncode=1, stderr='deliberate offline failure', stdout='')

    with pytest.raises(RuntimeError, match='deliberate offline failure'):
        plugins._install_from_catalogue('catalogue_probe', source, runner=failed)
    assert plugins._registry() is old
    assert plugins._catalogue_installed() == {}
    assert not (Path(home) / 'site' / '.catalogue_probe.new').exists()
