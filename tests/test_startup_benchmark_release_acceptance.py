"""Fail-closed protocol checks for startup release evidence."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

from tools import spacr_startup_benchmark as driver


def _run_cli(tmp_path: Path, *arguments: str,
             qt_platform: Optional[str] = None) -> subprocess.CompletedProcess:
    output = tmp_path / "release.json"
    environment = dict(os.environ)
    environment.pop("QT_QPA_PLATFORM", None)
    if qt_platform is not None:
        environment["QT_QPA_PLATFORM"] = qt_platform
    return subprocess.run(
        [
            sys.executable,
            str(Path(driver.__file__).resolve()),
            "--out", str(output),
            *arguments,
        ],
        cwd=str(driver.PACKAGE_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize(("arguments", "qt_platform", "message"), [
    (
        ("--release-acceptance", "--runs", "1"),
        None,
        "--release-acceptance requires exactly --runs 2",
    ),
    (
        ("--release-acceptance", "--offscreen"),
        None,
        "--release-acceptance cannot be combined with --offscreen",
    ),
    (
        ("--release-acceptance",),
        "minimal:fontengine=freetype",
        "--release-acceptance rejects inherited QT_QPA_PLATFORM=",
    ),
    (
        ("--release-acceptance", "--record-only"),
        None,
        "--release-acceptance cannot be combined with --record-only",
    ),
])
def test_invalid_release_cli_configuration_fails_before_starting_qt(
        tmp_path, arguments, qt_platform, message):
    completed = _run_cli(
        tmp_path, *arguments, qt_platform=qt_platform)

    assert completed.returncode == 2
    assert message in completed.stderr
    assert "spaCR startup benchmark written" not in completed.stdout
    assert not list(tmp_path.glob("release*.json"))


@pytest.mark.parametrize("qt_platform", [
    "offscreen",
    " OFFSCREEN:fontengine=freetype ",
    "minimal",
    "minimalegl",
    "vnc",
    "wayland;xcb;offscreen",
])
def test_release_configuration_rejects_known_headless_platform_overrides(
        qt_platform):
    violations = driver._release_acceptance_configuration_violations(
        runs=2,
        offscreen=False,
        environ={"QT_QPA_PLATFORM": qt_platform},
    )

    assert len(violations) == 1
    assert "rejects inherited QT_QPA_PLATFORM=" in violations[0]


@pytest.mark.parametrize("qt_platform", [
    "",
    "xcb",
    "wayland;xcb",
    "windows",
    "cocoa",
])
def test_release_configuration_allows_real_display_platform_overrides(
        qt_platform):
    assert driver._release_acceptance_configuration_violations(
        runs=2,
        offscreen=False,
        environ={"QT_QPA_PLATFORM": qt_platform},
    ) == []


@pytest.mark.parametrize(("options", "qt_platform", "message"), [
    (
        {"runs": 1},
        None,
        "--release-acceptance requires exactly --runs 2",
    ),
    (
        {"offscreen": True},
        None,
        "--release-acceptance cannot be combined with --offscreen",
    ),
    (
        {},
        "vnc",
        "--release-acceptance rejects inherited QT_QPA_PLATFORM=",
    ),
])
def test_invalid_release_api_configuration_starts_no_worker(
        tmp_path, monkeypatch, options, qt_platform, message):
    output = tmp_path / "invalid.json"
    if qt_platform is None:
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    else:
        monkeypatch.setenv("QT_QPA_PLATFORM", qt_platform)

    def unexpected_worker(*_args, **_kwargs):
        raise AssertionError("invalid release configuration started a worker")

    monkeypatch.setattr(driver, "_run_worker", unexpected_worker)

    with pytest.raises(ValueError, match=message):
        driver.run_benchmark(
            output, release_acceptance=True, **options)

    assert not output.exists()


def test_release_validation_requires_two_recorded_processes():
    violations = driver._combined_violations(
        [{}], release_acceptance=True)

    assert (
        "release acceptance combined artifact must contain exactly one cold "
        "run followed by one warm run"
    ) in violations


def test_release_validation_rechecks_the_platform_the_worker_used():
    artifact = {
        "environment": {
            "hardware": {"qt_platform": "offscreen:fontengine=freetype"},
        },
    }

    release_violations = driver._worker_schema_violations(
        artifact, "cold-process", release_acceptance=True)
    ordinary_violations = driver._worker_schema_violations(
        artifact, "cold-process")

    message = (
        "release acceptance requires a real Qt display platform; "
        "environment.hardware.qt_platform is "
    )
    assert any(message in row for row in release_violations)
    assert not any(message in row for row in ordinary_violations)


def test_one_run_diagnostic_mode_remains_available_for_instruction_314(
        tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    def fake_worker(home, output, label, timeout_s, offscreen, *,
                    package_root):
        calls.append((label, offscreen, package_root))
        return {"benchmark": []}

    monkeypatch.setattr(driver, "_run_worker", fake_worker)

    artifact = driver.run_benchmark(
        tmp_path / "one-run.json", runs=1, offscreen=True)

    assert calls == [("cold-process", True, driver.PACKAGE_ROOT)]
    assert artifact["driver"]["runs"] == 1
    assert artifact["driver"]["release_acceptance"] is False


def test_valid_release_configuration_runs_cold_then_warm(
        tmp_path, monkeypatch):
    calls = []
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    def fake_worker(home, output, label, timeout_s, offscreen, *,
                    package_root):
        calls.append(label)
        return {"benchmark": []}

    monkeypatch.setattr(driver, "_run_worker", fake_worker)

    artifact = driver.run_benchmark(
        tmp_path / "release.json", release_acceptance=True)

    assert calls == ["cold-process", "warm-process"]
    assert artifact["driver"]["runs"] == 2
    assert artifact["driver"]["release_acceptance"] is True
