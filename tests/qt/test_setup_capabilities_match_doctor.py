"""Render the setup table and doctor report for exactly the same fake machine."""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr import accelerator as acc
from spacr import doctor
from spacr.qt.widgets import setup_slides as slides


def _forbidden_probe(*_args, **_kwargs):
    raise AssertionError("This test must not probe real hardware or allocate tensors")


def _render_both(monkeypatch, kind):
    found = acc.Accelerator(
        kind=kind, device=kind, label="Fake " + kind,
        float64=kind != "mps",
    )
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1" if kind == "cuda" else None),
        cuda=SimpleNamespace(
            is_available=lambda: kind == "cuda",
            device_count=lambda: 1,
            get_device_name=lambda _index: "Fake CUDA device",
            synchronize=_forbidden_probe,
            init=_forbidden_probe,
        ),
        zeros=_forbidden_probe,
    )
    monkeypatch.setattr(doctor, "_import_torch", lambda: torch)
    monkeypatch.setattr(
        doctor, "_nvidia_driver", lambda: "fake-driver" if kind == "cuda" else None)
    monkeypatch.setattr(acc, "inspect_torch", lambda _torch: found)
    monkeypatch.setattr(acc, "resolve", lambda: found)
    monkeypatch.setattr(acc, "_torch", _forbidden_probe)
    monkeypatch.setattr(acc, "_opengl_likely", lambda: True)
    monkeypatch.setattr(acc, "neural_engines", lambda: ())
    monkeypatch.setattr(slides, "_say", lambda text: text)
    monkeypatch.setattr(
        slides.SetupSlides, "_cellpose_label", staticmethod(lambda: "Cellpose 4"))

    capabilities = acc.capabilities()
    html = "".join(slides.SetupSlides._what_this_machine_can_do())
    table_rows = [
        tuple(re.sub(r"<[^>]+>", "", cell).strip()
              for cell in re.findall(r"<td[^>]*>(.*?)</td>", row))
        for row in re.findall(r"<tr>(.*?)</tr>", html)
    ]
    assert "<table" in html
    assert len(table_rows) == len(slides.GPU_TABLE_ROWS)
    assert all(len(row) == 3 for row in table_rows)
    slide_decisions = {task: where for _library, where, task in table_rows}

    result = doctor.check_gpu(doctor.Context(probe_gpu=False))
    report = doctor.format_report([result])
    doctor_decisions = dict(re.findall(
        r"^\s+(.+?): (GPU|CPU) — ", report, flags=re.MULTILINE))
    expected = {task: "GPU" if accelerated else "CPU"
                for task, accelerated, _detail in capabilities}
    assert result.status == doctor.PASS
    assert doctor_decisions == expected, report
    for _library, prefix, slide_task in slides.GPU_TABLE_ROWS:
        matching = [task for task in expected if task.startswith(prefix)]
        assert len(matching) == 1
        assert slide_decisions[slide_task] == doctor_decisions[matching[0]]
    return slide_decisions


def test_cuda_and_metal_slide_tables_agree_with_their_doctor_reports(monkeypatch):
    cuda = _render_both(monkeypatch, "cuda")
    metal = _render_both(monkeypatch, "mps")
    assert cuda["Segmentation"] == metal["Segmentation"] == "GPU"
    assert cuda["Machine learning"] == "GPU"
    assert metal["Machine learning"] == "CPU"
    assert cuda["Visualization"] == metal["Visualization"] == "GPU"
