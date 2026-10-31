"""The CUDA driver is started by calls that release the lock; no context is made.

Lesson 22's first run froze the window between "run opened" and the model
loading: the first ``torch.cuda`` call loads the driver and enumerates its
devices from a Python call that holds the interpreter lock. The same driver
calls through ctypes release it, so the early warm (after the torch libraries
are mapped) starts them while the window keeps painting. It never creates a
context: that would cost VRAM and show the window as a compute process. These tests use a stand-in library;
whether the real driver then makes ``torch.cuda`` initialise faster is
checked on a machine with a GPU.
"""
from __future__ import annotations

import os

import pytest

from spacr.qt import gil_priority


class _Driver:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def _answer(self, name):
        self.calls.append(name)
        return 1 if name == self.fail_on else 0

    def cuInit(self, flags):
        assert flags == 0
        return self._answer("cuInit")

    def cuDeviceGet(self, device, ordinal):
        assert ordinal == 0
        return self._answer("cuDeviceGet")

    def cuDevicePrimaryCtxRetain(self, context, device):
        return self._answer("cuDevicePrimaryCtxRetain")

    def cuCtxCreate_v2(self, *args):
        return self._answer("cuCtxCreate_v2")


@pytest.fixture(autouse=True)
def _fresh_attempt(monkeypatch):
    monkeypatch.setattr(gil_priority, "_CUDA_DRIVER_TRIED", False)


def test_the_driver_starts_and_finds_device_zero_and_makes_no_context():
    driver = _Driver()
    assert gil_priority._initialise_the_cuda_driver(driver) is True
    assert driver.calls == ["cuInit", "cuDeviceGet"]


def test_a_driver_that_will_not_start_stops_the_warm_there():
    driver = _Driver(fail_on="cuInit")
    assert gil_priority._initialise_the_cuda_driver(driver) is False
    assert driver.calls == ["cuInit"]


def test_a_missing_device_is_reported_false_and_never_raises():
    driver = _Driver(fail_on="cuDeviceGet")
    assert gil_priority._initialise_the_cuda_driver(driver) is False
    assert driver.calls == ["cuInit", "cuDeviceGet"]


def test_it_is_tried_once_per_process():
    driver = _Driver()
    assert gil_priority._initialise_the_cuda_driver(driver) is True
    assert gil_priority._initialise_the_cuda_driver(driver) is False
    assert driver.calls.count("cuInit") == 1


def test_no_device_node_or_a_hidden_or_forced_cpu_device_means_no_driver(
        monkeypatch, tmp_path):
    node = tmp_path / "nvidiactl"
    node.write_bytes(b"")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SPACR_DEVICE", raising=False)
    assert gil_priority._cuda_may_be_used((str(node),)) is True
    assert gil_priority._cuda_may_be_used((str(tmp_path / "missing"),)) is False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gil_priority._cuda_may_be_used((str(node),)) is False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert gil_priority._cuda_may_be_used((str(node),)) is False
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("SPACR_DEVICE", "cpu")
    assert gil_priority._cuda_may_be_used((str(node),)) is False


def test_the_library_is_never_loaded_on_a_machine_without_a_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gil_priority._initialise_the_cuda_driver() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_the_early_warm_starts_the_driver_after_the_libraries_without_the_import_lock(
        monkeypatch):
    from spacr.qt import app

    order = []
    monkeypatch.setattr(gil_priority, "_open_torch_libraries",
                        lambda: order.append("libraries"))

    def start_driver(library=None):
        order.append(("driver", app.HEAVY_IMPORT_LOCK.locked()))
        return True

    monkeypatch.setattr(gil_priority, "_initialise_the_cuda_driver",
                        start_driver)
    app._map_the_torch_libraries()
    assert order == ["libraries", ("driver", False)]
