"""Network preferences belong to the selected spaCR data directory."""
from pathlib import Path

import pytest

from spacr import logging_util, updater


@pytest.mark.parametrize('portable', [False, True])
def test_network_preferences_follow_spacr_home(tmp_path, monkeypatch, portable):
    ordinary_home = tmp_path / 'ordinary-home'
    ordinary_home.mkdir()
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: ordinary_home))
    monkeypatch.delenv('SPACR_NETWORK_CONFIG', raising=False)
    if portable:
        launcher = tmp_path / 'launcher'
        launcher.mkdir()
        (launcher / 'spacr-portable').touch()
        monkeypatch.setenv('SPACR_LAUNCHER_DIR', str(launcher))
        monkeypatch.delenv('SPACR_HOME', raising=False)
        monkeypatch.delenv('SPACR_PORTABLE', raising=False)
        selected_home = launcher / 'spacr-data'
    else:
        selected_home = tmp_path / 'configured-home'
        monkeypatch.setenv('SPACR_HOME', str(selected_home))
    logging_util._portable_root_for.cache_clear()
    monkeypatch.setattr(updater, '_apply_network_settings', lambda: {})
    try:
        updater._write_network_config('http://proxy.example.invalid:3128', '')
        assert updater._network_config_path() == selected_home / 'network.json'
        assert updater._read_network_config() == {
            'proxy': 'http://proxy.example.invalid:3128', 'ca_bundle': ''}
        assert not (ordinary_home / '.spacr').exists()
    finally:
        logging_util._portable_root_for.cache_clear()


def test_explicit_network_file_overrides_spacr_home(tmp_path, monkeypatch):
    chosen = tmp_path / 'separate' / 'network.json'
    monkeypatch.setenv('SPACR_NETWORK_CONFIG', str(chosen))
    monkeypatch.setenv('SPACR_HOME', str(tmp_path / 'configured-home'))
    monkeypatch.setattr(updater, '_apply_network_settings', lambda: {})
    updater._write_network_config('', str(tmp_path / 'certificate.pem'))
    assert updater._network_config_path() == chosen
    assert chosen.is_file()
