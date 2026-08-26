"""Focused checks for the non-interactive hardware-report launch probe."""

from __future__ import annotations

import json
from types import SimpleNamespace

from tools import spacr_hardware_report as report_module


def _run_launch_report(monkeypatch, first_loop: str) -> list[str]:
    payload = {
        "to_event_loop": 0.4,
        "total": 1.2,
        "exit": 0,
        "modules": 123,
        "first_loop": first_loop,
    }
    completed = SimpleNamespace(
        stdout="LAUNCH_JSON" + json.dumps(payload),
        stderr="",
    )
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: completed)
    monkeypatch.setattr(report_module.Report, "_out", staticmethod(lambda _text: None))
    say = report_module.Report()
    report_module.the_real_launch(say)
    return say.lines


def test_launch_probe_stops_application_dialog_and_nested_event_loops() -> None:
    probe = report_module._LAUNCH_PROBE

    for target in ("W.QApplication", "W.QDialog", "C.QEventLoop"):
        assert f"{target}.exec = _stop" in probe
        assert f"{target}.exec_ = _stop" in probe


def test_launch_report_identifies_the_setup_screen_precisely(monkeypatch) -> None:
    text = "\n".join(_run_launch_report(monkeypatch, "SetupSlides"))

    assert "The setup screen opened before the main application loop." in text
    assert "A normal interactive launch waits here until setup is complete." in text
    assert "For an unattended launch, use `spacr --no-setup`." in text


def test_launch_report_does_not_call_every_dialog_the_setup_screen(monkeypatch) -> None:
    text = "\n".join(_run_launch_report(monkeypatch, "InstallerConsentDialog"))

    assert "The InstallerConsentDialog event loop opened" in text
    assert "The setup screen opened" not in text
