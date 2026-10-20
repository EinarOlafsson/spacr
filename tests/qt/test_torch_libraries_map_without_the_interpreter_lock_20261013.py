"""PyTorch's libraries are mapped with the interpreter lock released.

Lesson 08's Measure run (clip batch_20_batch_settings) froze the window for
326 ms half a second after Run. Measured offscreen on 2026-10-13 at
2560x1440 with a real run, a 2 ms heartbeat and a timer on every C-extension
load: the run's worker imported torch, and loading ``torch._C`` took 321 ms
in one C call, ``dlopen`` of ``libtorch_cpu`` and ``libtorch_cuda``, with the
interpreter lock held. ``ctypes.CDLL`` holds it too; a ctypes call to the C
library's ``dlopen`` does not.

:func:`spacr.qt.gil_priority._open_torch_libraries` maps those files first,
as ``torch/__init__.py`` does and in its order, and the entry-module import
thread, the run's worker and the pipeline preloader call it before torch can
be imported. The block test imports torch in two fresh interpreters, with
and without it, and compares the longest stretch a Python heartbeat thread
waited.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr.qt import gil_priority

ROOT = Path(__file__).resolve().parents[2]
HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture
def fresh(monkeypatch):
    """As if torch had never been imported nor its libraries tried."""
    monkeypatch.setattr(gil_priority, "_TORCH_LIBRARIES_TRIED", False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")


def _fake_torch(monkeypatch, tmp_path, names):
    package = tmp_path / "torch"
    (package / "lib").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name in names:
        (package / "lib" / name).write_bytes(b"")
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: SimpleNamespace(
                            origin=str(package / "__init__.py")))
    return package / "lib"


def _recording(monkeypatch, answer=1):
    calls = []

    def dlopen(path, mode):
        calls.append((os.path.basename(path.decode()), mode))
        return answer

    monkeypatch.setattr(gil_priority, "_dlopen_without_the_lock",
                        lambda: dlopen)
    monkeypatch.setattr(gil_priority, "_open_wheel_cuda_libraries",
                        lambda opener: True)
    return calls


def test_the_libraries_are_mapped_in_torchs_order_and_flags(
        fresh, monkeypatch, tmp_path):
    _fake_torch(monkeypatch, tmp_path, [name for name, _ in
                                        gil_priority._TORCH_LIBRARIES])
    calls = _recording(monkeypatch)
    assert gil_priority._open_torch_libraries() is True
    assert calls == [
        ("libtorch_global_deps.so", os.RTLD_NOW | os.RTLD_GLOBAL),
        ("libtorch_cpu.so", os.RTLD_NOW),
        ("libtorch_cuda.so", os.RTLD_NOW)]
    assert gil_priority._open_torch_libraries() is False
    assert len(calls) == 3


def test_a_cpu_build_maps_what_it_has(fresh, monkeypatch, tmp_path):
    _fake_torch(monkeypatch, tmp_path,
                ["libtorch_global_deps.so", "libtorch_cpu.so"])
    calls = _recording(monkeypatch)
    assert gil_priority._open_torch_libraries() is True
    assert [name for name, _ in calls] == ["libtorch_global_deps.so",
                                           "libtorch_cpu.so"]


def test_without_the_global_library_nothing_is_mapped(
        fresh, monkeypatch, tmp_path):
    _fake_torch(monkeypatch, tmp_path, ["libtorch_cpu.so"])
    calls = _recording(monkeypatch)
    assert gil_priority._open_torch_libraries() is False
    assert calls == []


def test_a_failed_map_stops_and_leaves_the_rest_to_torch(
        fresh, monkeypatch, tmp_path):
    _fake_torch(monkeypatch, tmp_path, [name for name, _ in
                                        gil_priority._TORCH_LIBRARIES])
    calls = _recording(monkeypatch, answer=None)
    assert gil_priority._open_torch_libraries() is False
    assert len(calls) == 1


def test_nothing_happens_once_torch_is_imported(fresh, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(gil_priority, "_dlopen_without_the_lock",
                        lambda: pytest.fail("dlopen looked up"))
    assert gil_priority._open_torch_libraries() is False


def test_nothing_happens_off_linux_or_without_torch(fresh, monkeypatch):
    monkeypatch.setattr(gil_priority, "_dlopen_without_the_lock",
                        lambda: pytest.fail("dlopen looked up"))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert gil_priority._open_torch_libraries() is False
    monkeypatch.setattr(gil_priority, "_TORCH_LIBRARIES_TRIED", False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert gil_priority._open_torch_libraries() is False


def test_the_wheel_cuda_libraries_follow_torchs_condition(
        monkeypatch, tmp_path):
    site = tmp_path / "site"
    for folder, name in (("cuda_nvrtc", "libnvrtc.so.12"),
                         ("nvjitlink", "libnvJitLink.so.12")):
        lib = site / "nvidia" / folder / "lib"
        lib.mkdir(parents=True)
        (lib / name).write_bytes(b"")
    monkeypatch.setattr(sys, "path", [str(tmp_path / "nowhere"), str(site)])
    opened = []

    def dlopen(path, mode):
        opened.append(os.path.basename(path.decode()))
        return 1

    plain = tmp_path / "plain_maps"
    plain.write_text("7f00 r-xp /usr/lib/libc.so.6\n")
    assert gil_priority._open_wheel_cuda_libraries(dlopen, str(plain))
    assert opened == []

    wheel = tmp_path / "wheel_maps"
    wheel.write_text("7f00 r-xp /x/nvidia/cuda_runtime/lib/libcudart.so.12\n")
    assert gil_priority._open_wheel_cuda_libraries(dlopen, str(wheel))
    assert opened == ["libnvrtc.so.12", "libnvJitLink.so.12"]

    monkeypatch.setattr(sys, "path", [str(tmp_path / "nowhere")])
    assert not gil_priority._open_wheel_cuda_libraries(dlopen, str(wheel))
    assert not gil_priority._open_wheel_cuda_libraries(
        dlopen, str(tmp_path / "missing"))


def test_the_entry_import_maps_torch_first(monkeypatch):
    pytest.importorskip("PySide6")
    from spacr.qt import bridge

    order = []
    monkeypatch.setattr(bridge, "_open_torch_libraries",
                        lambda: order.append("map"))
    monkeypatch.setattr(bridge.importlib, "import_module",
                        lambda name: order.append(name))
    monkeypatch.delitem(sys.modules, "spacr.qt.app", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    bridge._import_the_entry_module("spacr.some_entry")
    assert order == ["map", "spacr.some_entry"]


_PROBE = textwrap.dedent("""
    import json, sys, threading, time
    sys.path.insert(0, sys.argv[1])
    sys.setswitchinterval(0.001)
    from spacr.qt import gil_priority
    done, gaps = [False], []

    def beat():
        last = time.perf_counter()
        while not done[0]:
            time.sleep(0.001)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    def work():
        if sys.argv[2] == "map":
            gil_priority._open_torch_libraries()
        import torch
        result["sum"] = float(torch.ones(3).sum())

    result = {}
    heart = threading.Thread(target=beat)
    heart.start()
    time.sleep(0.05)
    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    done[0] = True
    heart.join()
    result["longest"] = max(gaps)
    print(json.dumps(result))
""")


def _probe(mode):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", _PROBE, str(ROOT), mode],
                         capture_output=True, text=True, timeout=240,
                         env=env, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not HAS_TORCH or not sys.platform.startswith("linux"),
                    reason="needs PyTorch on Linux")
def test_importing_torch_after_the_map_blocks_other_threads_far_less():
    """Self-calibrated: the same import with and without the map.

    Each mode's longest stall is the least of three fresh interpreters, so
    a runner pausing the heartbeat thread once does not read as the import
    holding the interpreter lock.
    """
    def least(mode):
        runs = [_probe(mode) for _ in range(3)]
        return min(runs, key=lambda run: run["longest"])

    before = least("import")
    now = least("map")
    assert now["sum"] == before["sum"] == 3.0
    if before["longest"] < 0.08:
        pytest.skip(f"torch's libraries load in {before['longest']:.3f} s "
                    "here; nothing to compare")
    assert now["longest"] < before["longest"] / 3, (now, before)
