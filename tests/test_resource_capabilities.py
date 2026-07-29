from __future__ import annotations

import subprocess
from types import SimpleNamespace

from tests import resource_capabilities as capabilities


def test_cuda_detection_reflects_usable_torch_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert capabilities.cuda_available()

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert not capabilities.cuda_available()


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
