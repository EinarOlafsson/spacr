"""Focused checks for the non-interactive hardware-report launch probe."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools import spacr_hardware_report as report_module
from tools import spacr_startup_benchmark as benchmark_driver


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


def test_quick_report_does_not_disguise_a_subset_as_registry_coverage(
        monkeypatch) -> None:
    monkeypatch.setattr(
        report_module.Report, "_out", staticmethod(lambda _text: None))
    say = report_module.Report()

    report_module.screens(say, quick=True)

    text = "\n".join(say.lines)
    assert "complete registry sweep is never" in text
    assert "hand-picked subset" in text


def test_hardware_report_consumes_the_current_benchmark_schema() -> None:
    assert report_module.STARTUP_BENCHMARK_SCHEMA_VERSION == (
        benchmark_driver.SCHEMA_VERSION)


def test_full_report_reads_the_exact_registry_driver_artifact(
        monkeypatch) -> None:
    def completed(command, **_kwargs):
        output = Path(command[command.index("--out") + 1])
        output.write_text(json.dumps({
            "schema_version": report_module.STARTUP_BENCHMARK_SCHEMA_VERSION,
            "passed": True,
            "violations": [],
            "registry_keys": ["mask", "measure"],
            "runs": [{"benchmark": {
                "measured_keys": ["mask", "measure"],
                "results": [{
                    "detail": "mask", "duration_s": 1.25,
                    "worst_event_loop_stall_ms": 30.0,
                }],
                "violations": [],
            }}],
        }), encoding="utf-8")
        assert command[command.index("--runs") + 1] == "1"
        assert "--record-only" in command
        package_root = Path(command[command.index("--package-root") + 1])
        assert package_root == Path(report_module.__file__).resolve().parents[1]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", completed)
    monkeypatch.setattr(
        report_module.Report, "_out", staticmethod(lambda _text: None))
    say = report_module.Report()

    report_module.screens(say, quick=False)

    text = "\n".join(say.lines)
    assert "live registry                    2 app(s)" in text
    assert "ratchet passed                   True" in text
    assert "measured registry                2 app(s)" in text
    assert "sets equal                       True" in text
    assert "mask                                1.250 s" in text


def test_full_report_rejects_an_old_registry_artifact_shape(monkeypatch) -> None:
    def completed(command, **_kwargs):
        output = Path(command[command.index("--out") + 1])
        output.write_text(json.dumps({
            "schema_version": 1,
            "passed": True,
            "registry_keys": ["mask"],
            "runs": [{"benchmark": {
                "measured_keys": ["mask"],
                "results": [],
                "violations": [],
            }}],
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", completed)
    monkeypatch.setattr(
        report_module.Report, "_out", staticmethod(lambda _text: None))
    say = report_module.Report()

    report_module.screens(say, quick=False)

    text = "\n".join(say.lines)
    assert "FAILED (artifact schema 1, expected 2)" in text
    assert "sets equal" not in text
