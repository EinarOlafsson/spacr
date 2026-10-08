from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from tests import resource_capabilities as capabilities


def test_cuda_detection_reflects_usable_torch_device(monkeypatch):
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: calls.append("available") or True,
        device_count=lambda: calls.append("count") or 1,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert capabilities.cuda_available()
    assert calls == ["available", "count"]

    calls.clear()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cuda.device_count = lambda: calls.append("count") or 0
    assert not capabilities.cuda_available()
    assert calls == ["available", "count"]


def test_explicit_empty_cuda_visibility_never_imports_torch(monkeypatch):
    import builtins

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    original_import = builtins.__import__
    imports = []

    def recording_import(name, *args, **kwargs):
        if name == "torch":
            imports.append(name)
            raise AssertionError("hidden CUDA must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    assert capabilities.cuda_available() is False
    assert imports == []


def test_gpu_marker_guard_skips_explicitly_hidden_cuda_without_torch(monkeypatch):
    import builtins

    suite_hooks = sys.modules.get("conftest") or sys.modules["tests.conftest"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    original_import = builtins.__import__
    imports = []

    def recording_import(name, *args, **kwargs):
        if name == "torch":
            imports.append(name)
            raise AssertionError("hidden CUDA must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    assert suite_hooks._no_room_on_the_gpu() == "no CUDA device"
    assert imports == []


def test_gpu_marker_guard_keeps_visible_and_unset_memory_checks(monkeypatch):
    suite_hooks = sys.modules.get("conftest") or sys.modules["tests.conftest"]
    calls = []
    mib = 1024 * 1024
    cuda = SimpleNamespace(
        is_available=lambda: calls.append("available") or True,
        mem_get_info=lambda: calls.append("memory") or
        (suite_hooks.GPU_ROOM_MB * mib, 4096 * mib),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    for visibility in (None, "0"):
        if visibility is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
        calls.clear()
        assert suite_hooks._no_room_on_the_gpu() == ""
        assert calls == ["available", "memory"]

    cuda.mem_get_info = lambda: (
        (suite_hooks.GPU_ROOM_MB - 1) * mib, 4096 * mib)
    assert "the GPU is busy" in suite_hooks._no_room_on_the_gpu()


def test_package_detection_uses_the_importable_dependency():
    assert capabilities.package_available(
        "cellpose", finder=lambda name: object() if name == "cellpose" else None)
    assert not capabilities.package_available(
        "missing", finder=lambda _name: None)


def test_endpoint_detection_uses_a_bounded_head_request():
    calls = []

    class Response:
        status = 204

        def close(self):
            calls.append("closed")

    def opener(request, timeout):
        calls.append((request.get_method(), request.full_url, timeout))
        return Response()

    assert capabilities.endpoint_available(
        "https://example.test", timeout=1.25, opener=opener)
    assert calls == [
        ("HEAD", "https://example.test", 1.25),
        "closed",
    ]


def test_endpoint_detection_returns_false_when_unreachable():
    def unavailable(_request, timeout):
        raise OSError(f"offline after {timeout}")

    assert not capabilities.endpoint_available(opener=unavailable)


def test_nas_probe_passes_requirements_to_a_bounded_child():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    assert capabilities.paths_available(
        (("/nas/data", "dir"), ("/nas/settings.csv", "file")),
        timeout=2.0,
        runner=runner,
    )
    command, kwargs = calls[0]
    assert command[:2] == [capabilities.sys.executable, "-c"]
    assert "/nas/data" in command[-1]
    assert kwargs["timeout"] == 2.0
    assert kwargs["check"] is False


def test_nas_probe_fails_closed_on_timeout():
    def runner(_command, **_kwargs):
        raise subprocess.TimeoutExpired("probe", 0.01)

    assert not capabilities.paths_available(
        (("/nas/data", "dir"),), runner=runner)


def test_nas_probe_rejects_unknown_requirement_kinds():
    assert not capabilities.paths_available((("/nas/data", "socket"),))
