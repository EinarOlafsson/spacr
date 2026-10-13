"""Doctor must report every shared capability, not only unaccelerated tasks."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from spacr import accelerator as acc
from spacr import doctor


def _forbidden_probe(*_args, **_kwargs):
    raise AssertionError("This test must not probe real hardware or allocate tensors")


@pytest.fixture
def fake_machine(monkeypatch):
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'SPACR_DEVICE'):
        monkeypatch.delenv(key, raising=False)
    def install(case):
        kind = "mps" if case in ("metal", "unusable-metal") else (
            "cuda" if case == "cuda" else "cpu")
        found = acc.Accelerator(
            kind=kind, device=kind, label="Fake " + case,
            usable=case != "unusable-metal", float64=kind != "mps",
            note="fake backend is unavailable" if case == "unusable-metal" else "",
        )

        def initialize():
            if case == "cuda-mismatch":
                raise RuntimeError("fake runtime mismatch")

        torch = SimpleNamespace(
            version=SimpleNamespace(
                cuda="12.1" if case in ("cuda", "cuda-mismatch") else None),
            cuda=SimpleNamespace(
                is_available=lambda: case == "cuda",
                device_count=lambda: 1,
                get_device_name=lambda _index: "Fake CUDA device",
                init=initialize,
                synchronize=_forbidden_probe,
            ),
            zeros=_forbidden_probe,
        )

        def import_torch():
            if case == "missing-torch":
                raise ImportError("fake torch is absent")
            return torch

        monkeypatch.setattr(doctor, "_import_torch", import_torch)
        monkeypatch.setattr(
            doctor, "_nvidia_driver",
            lambda: "fake-driver" if case in ("cuda", "cuda-mismatch") else None,
        )
        monkeypatch.setattr(acc, "inspect_torch", lambda _torch, **kwargs: found)
        monkeypatch.setattr(acc, "resolve", _forbidden_probe)
        monkeypatch.setattr(acc, "_torch", _forbidden_probe)
        monkeypatch.setattr(acc, "_opengl_likely", _forbidden_probe)
        gpu = found.is_gpu
        rows = (
            ("Segmentation (Cellpose)", gpu, "fake segmentation decision"),
            ("Model training", gpu, "fake training decision"),
            ("Model inference / classification", gpu, "fake inference decision"),
            ("Live backdrop and spaceout", True, "fake display shader"),
            ("UMAP / t-SNE / clustering", found.is_cuda, "fake reduction decision"),
            ("A future capability", False, "not accelerated on this fake machine"),
        )
        calls = []

        def capabilities(**kwargs):
            calls.append(case)
            return rows

        monkeypatch.setattr(acc, "capabilities", capabilities)
        return SimpleNamespace(rows=rows, calls=calls, found=found)

    return install


@pytest.mark.parametrize("case,status", [
    ("cuda", doctor.PASS),
    ("metal", doctor.PASS),
    ("cpu", doctor.WARN),
    ("unusable-metal", doctor.WARN),
    ("missing-torch", doctor.SKIP),
    ("cuda-mismatch", doctor.FAIL),
])
def test_doctor_reports_every_capability_in_its_rendered_answer(
        fake_machine, case, status):
    machine = fake_machine(case)
    result = doctor.check_gpu(doctor.Context(probe_gpu=False))
    report = doctor.format_report([result])
    expected = [
        f"{task}: {'GPU' if accelerated else 'CPU'} — {detail}"
        for task, accelerated, detail in machine.rows
    ]
    report_lines = [line.strip() for line in report.splitlines()]

    assert result.status == status
    assert all(line in report_lines for line in expected), report
    assert [line for line in report_lines if line in expected] == expected
    assert machine.calls == [case], "one report must use one capability snapshot"
    if case == "unusable-metal":
        assert machine.found.note in result.details
    elif case == "cuda-mismatch":
        assert "CUDA initialization skipped (--no-gpu-probe)." in result.details
        assert "--force-reinstall torch" in result.fix
    elif case == "missing-torch":
        assert result.fix == "Fix the `torch` row above first."


def test_unavailable_capability_decoration_keeps_the_doctor_diagnosis(
        fake_machine, monkeypatch):
    fake_machine("cuda")

    def unavailable(**kwargs):
        raise RuntimeError("fake capability enumeration failed")

    monkeypatch.setattr(acc, "capabilities", unavailable)
    result = doctor.check_gpu(doctor.Context(probe_gpu=False))
    assert result.status == doctor.PASS
    assert "allocation probe skipped" in result.message
    assert "device names not queried (--no-gpu-probe)" in doctor.format_report([result])
