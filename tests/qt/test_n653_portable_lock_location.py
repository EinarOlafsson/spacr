"""Portable and configured spaCR homes also contain the lock files."""
from pathlib import Path

import pytest

from spacr import logging_util
from spacr.qt import crash_recovery


@pytest.mark.parametrize('portable', [False, True])
def test_lock_records_follow_the_selected_spacr_home(tmp_path, monkeypatch, portable):
    ordinary_home = tmp_path / 'ordinary-home'
    ordinary_home.mkdir()
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: ordinary_home))
    expanduser = crash_recovery.os.path.expanduser
    monkeypatch.setattr(crash_recovery.os.path, 'expanduser',
                        lambda text: str(ordinary_home) if text == '~' else expanduser(text))
    monkeypatch.delenv('SPACR_LOCK_DIR', raising=False)
    if portable:
        launcher = tmp_path / 'launcher'
        launcher.mkdir()
        (launcher / 'spacr-portable').touch()
        monkeypatch.setenv('SPACR_LAUNCHER_DIR', str(launcher))
        monkeypatch.delenv('SPACR_HOME', raising=False)
        monkeypatch.delenv('SPACR_PORTABLE', raising=False)
        expected_home = launcher / 'spacr-data'
    else:
        expected_home = tmp_path / 'configured-home'
        monkeypatch.setenv('SPACR_HOME', str(expected_home))
    logging_util._portable_root_for.cache_clear()
    crash_recovery._release_all_locks()
    try:
        folder = Path(crash_recovery._locks_folder())
        assert folder == expected_home / 'locks'
        assert folder.is_dir()
        assert not (ordinary_home / '.spacr').exists()
        lock_path = crash_recovery._lock_file_for(tmp_path / 'plate')
        assert Path(lock_path).parent == folder
        assert crash_recovery._try_lock(lock_path) == (True, None)
        assert Path(lock_path).is_file()
    finally:
        crash_recovery._release_all_locks()
        logging_util._portable_root_for.cache_clear()
    assert not Path(lock_path).exists()


def test_explicit_lock_folder_still_overrides_the_spacr_home(tmp_path, monkeypatch):
    chosen = tmp_path / 'explicit-locks'
    monkeypatch.setenv('SPACR_LOCK_DIR', str(chosen))
    monkeypatch.setenv('SPACR_HOME', str(tmp_path / 'configured-home'))
    assert Path(crash_recovery._locks_folder()) == chosen
    assert chosen.is_dir()
