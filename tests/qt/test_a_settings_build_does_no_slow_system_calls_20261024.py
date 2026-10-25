"""Building a module's settings makes no process and repeats no PATH search.

Windows stack samples of the Mask open (2026-10-24) showed the GUI thread
starting the GPU discovery child (a spawn start launches an interpreter),
searching PATH for each AI tool on every check and resolving the package
folder for every listed model.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget

from spacr import _mask_workers as mw
from spacr import model_zoo
from spacr.qt.ai import providers
from spacr.qt.screens import settings_model as sm


def test_building_the_mask_sections_starts_no_process_on_the_gui_thread(qtbot, monkeypatch):
    started_on = []
    original_init = subprocess.Popen.__init__

    def record(self, *args, **kwargs):
        started_on.append(threading.current_thread())
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", record)
    import multiprocessing.process as processes

    original_start = processes.BaseProcess.start

    def record_start(self):
        started_on.append(threading.current_thread())
        return original_start(self)

    monkeypatch.setattr(processes.BaseProcess, "start", record_start)
    monkeypatch.setattr(mw, "_CONTROL_GPU_COUNT", [])
    monkeypatch.setattr(mw, "_CONTROL_GPU_PROBE", None)
    owner = QWidget()
    qtbot.addWidget(owner)
    try:
        sm.SettingsWidgets("mask", parent=owner).build_sections()
        probe = mw._CONTROL_GPU_PROBE
        assert probe is not None
        probe["launcher"].join(30)
        assert started_on and threading.main_thread() not in started_on
    finally:
        mw._stop_mask_gpu_probe()


def test_repeated_provider_checks_search_path_once_per_tool(monkeypatch):
    searched = []

    def which(name):
        searched.append(name)
        return None

    monkeypatch.setattr(providers.shutil, "which", which)
    monkeypatch.setattr(providers, "_PATH_LOOKUPS", {})
    for _ in range(5):
        providers.configured_providers()
    tools = [p.cli_name for p in providers._PROVIDERS if getattr(p, "cli_name", "")]
    assert tools and sorted(searched) == sorted(tools)


def test_a_new_path_or_resolver_is_searched_again(monkeypatch):
    monkeypatch.setattr(providers, "_PATH_LOOKUPS", {})
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    assert providers._found_on_path("gh") is False
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/bin/gh")
    assert providers._found_on_path("gh") is True
    monkeypatch.setenv("PATH", "/nowhere")
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    assert providers._found_on_path("gh") is False


def test_the_package_model_root_resolves_once(monkeypatch):
    model_zoo._package_folder.cache_clear()
    calls = []
    original = Path.resolve

    def resolve(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    first = model_zoo.package_model_root()
    for _ in range(20):
        assert model_zoo.package_model_root() == first
    assert len(calls) == 1
    assert first == Path(model_zoo.__file__).resolve().parent / "resources" / "models"
