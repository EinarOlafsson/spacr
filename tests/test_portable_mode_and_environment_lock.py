"""Portable mode (653) and the per-run environment lockfile (654)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spacr import logging_util, run_journal

_PORTABLE_VARS = (
    "SPACR_PORTABLE", "SPACR_LAUNCHER_DIR", "SPACR_HOME", "SPACR_LOG_DIR",
    "SPACR_BACKENDS_DIR", "SPACR_PLUGIN_HOME", "XDG_CACHE_HOME",
    "XDG_STATE_HOME", "TORCH_HOME", "HF_HOME", "MPLCONFIGDIR",
    "CELLPOSE_LOCAL_MODELS_PATH",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No portable variable set; the launcher folder is an empty tmp folder."""
    for name in _PORTABLE_VARS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    app = tmp_path / "app"
    app.mkdir()
    monkeypatch.setenv("SPACR_LAUNCHER_DIR", str(app))
    logging_util._portable_root_for.cache_clear()
    yield app
    logging_util._portable_root_for.cache_clear()


def test_portable_mode_is_off_without_marker_or_variable(clean_env):
    assert logging_util._portable_root() is None
    assert logging_util._spacr_home() == Path.home() / ".spacr"
    assert logging_util._apply_portable_mode() is None
    assert "SPACR_HOME" not in os.environ


def test_a_marker_next_to_the_app_moves_every_folder(clean_env):
    (clean_env / "spacr-portable").write_text("")
    data = clean_env / "spacr-data"
    assert logging_util._portable_root() == clean_env
    assert logging_util._spacr_home() == data
    assert logging_util._apply_portable_mode() == data
    assert data.is_dir()
    assert os.environ["SPACR_HOME"] == str(data)
    assert os.environ["TORCH_HOME"] == str(data / "cache" / "torch")
    assert logging_util.log_dir() == data / "logs"

    from spacr import _segmentation_backends, plugins, restart_state
    assert _segmentation_backends._backends_root() == str(data / "backends")
    assert plugins._plugin_home() == str(data / "plugins")
    assert restart_state.state_path().parent == data


def test_the_variable_overrides_the_marker(clean_env, monkeypatch, tmp_path):
    (clean_env / "spacr-portable").write_text("")
    monkeypatch.setenv("SPACR_PORTABLE", "off")
    assert logging_util._portable_root() is None

    stick = tmp_path / "usb"
    monkeypatch.setenv("SPACR_PORTABLE", str(stick))
    assert logging_util._spacr_home() == stick / "spacr-data"

    monkeypatch.setenv("SPACR_PORTABLE", "1")
    (clean_env / "spacr-portable").unlink()
    logging_util._portable_root_for.cache_clear()
    assert logging_util._portable_root() == clean_env


def test_a_variable_the_user_set_is_kept(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SPACR_PORTABLE", "1")
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "mine"))
    logging_util._apply_portable_mode()
    assert os.environ["TORCH_HOME"] == str(tmp_path / "mine")
    assert os.environ["SPACR_LOG_DIR"] == str(
        clean_env / "spacr-data" / "logs")


def test_portable_qt_settings_are_an_ini_file_beside_the_app(clean_env,
                                                             monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import prefs

    assert prefs._store_args("spacr", "qt") == ("spacr", "qt")
    before = Path(QSettings(QSettings.IniFormat, QSettings.UserScope,
                            "probe", "probe").fileName()).parent.parent
    monkeypatch.setenv("SPACR_PORTABLE", "1")
    args = prefs._store_args("spacr", "qt")
    store = QSettings(*args)
    try:
        assert store.format() == QSettings.IniFormat
        assert Path(store.fileName()).is_relative_to(
            clean_env / "spacr-data" / "settings")
    finally:
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope,
                          str(before))


@pytest.fixture
def journal(monkeypatch, tmp_path):
    """A private run journal and a fake pip/conda that count their calls."""
    root = tmp_path / "home" / "runs"
    root.mkdir(parents=True)
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    monkeypatch.setattr(run_journal, "_ENV_LOCK_MEMO", {})
    calls = []

    def fake_run(command):
        calls.append(command)
        if "pip" in command:
            return "numpy==2.0.0\nspacr==9.9\n"
        return None

    monkeypatch.setattr(run_journal, "_run_quiet", fake_run)
    monkeypatch.setattr(run_journal, "_conda_lock_text",
                        lambda: "@EXPLICIT\nhttps://x/numpy.conda\n")
    return root, calls


def _open(app_key="mask"):
    with run_journal.open_run(app_key, {"src": "/nowhere"}) as run:
        pass
    manifest = json.loads((run.dir / "manifest.json").read_text())
    return run, manifest


def test_each_run_gets_lockfiles_named_in_its_manifest(journal):
    root, calls = journal
    run, manifest = _open()
    lock = manifest["environment_lock"]
    assert lock["pip"] == "environment/requirements-lock.txt"
    assert lock["conda"] == "environment/conda-explicit.txt"
    assert len(lock["sha256"]) == 64
    assert (run.dir / lock["pip"]).read_text() == "numpy==2.0.0\nspacr==9.9\n"
    assert (run.dir / lock["conda"]).read_text().startswith("@EXPLICIT")
    assert len(calls) == 1


def test_pip_and_conda_run_once_per_environment(journal, monkeypatch):
    root, calls = journal
    first, m1 = _open()
    second, m2 = _open("measure")
    assert len(calls) == 1
    assert m1["environment_lock"]["sha256"] == m2["environment_lock"]["sha256"]
    store = root.parent / "env_locks"
    assert len(list(store.iterdir())) == 1

    monkeypatch.setattr(run_journal, "_ENV_LOCK_MEMO", {})
    third, m3 = _open()
    assert len(calls) == 1
    assert (third.dir / "environment" / "conda-explicit.txt").is_file()


def test_without_pip_the_lock_is_read_from_metadata(monkeypatch):
    monkeypatch.setattr(run_journal, "_run_quiet", lambda command: None)
    text = run_journal._pip_lock_text({"numpy": "2.0.0", "spacr": "1.5"})
    assert "numpy==2.0.0" in text.splitlines()
    assert "spacr==1.5" in text.splitlines()


def test_outside_conda_only_the_pip_lock_is_written(journal, monkeypatch):
    monkeypatch.setattr(run_journal, "_conda_lock_text", lambda: None)
    run, manifest = _open()
    assert manifest["environment_lock"]["conda"] is None
    assert not (run.dir / "environment" / "conda-explicit.txt").exists()


def test_a_failed_lock_never_fails_the_run(journal, monkeypatch):
    def broken(run_dir, packages):
        raise OSError("disk full")

    monkeypatch.setattr(run_journal, "_write_environment_lock", broken)
    run, manifest = _open()
    assert manifest["status"] == "success"
    assert manifest["environment_lock"] is None
    assert any("environment lockfile" in w
               for w in manifest["provenance_warnings"])
