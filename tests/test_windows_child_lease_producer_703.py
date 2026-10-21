"""Validate the real local pip fixture and native-only acceptance boundary."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "tools" / "check_windows_installer_child_leases_703.py"


@pytest.fixture
def producer():
    spec = importlib.util.spec_from_file_location("native_child_lease_producer_703", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lease_fixture_is_a_real_offline_installable_wheel(producer, tmp_path):
    wheel = tmp_path / producer.WHEEL_NAME
    wheel.write_bytes(producer.wheel_bytes())
    target = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
         "install", "--no-deps", "--no-index", "--no-cache-dir", "--target", str(target), str(wheel)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert (target / "spacr_lease_probe.py").read_text() == "NATIVE_LEASE_PROBE = True\n"
    assert (target / "spacr_lease_probe-0.0.1.dist-info" / "INSTALLER").read_text().strip() == "pip"


def test_loopback_request_reaches_handshake_before_its_payload_is_released(producer):
    gate = producer.WheelGate()

    def download():
        with urlopen(gate.url, timeout=10) as response:
            return response.read()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(download)
            try:
                assert gate.requested.wait(5)
                assert not gate.release.is_set()
                assert not future.done()
                assert not gate.expired.is_set()
            finally:
                gate.release.set()
            assert future.result(timeout=5) == producer.wheel_bytes()
    finally:
        gate.close()


@pytest.mark.skipif(os.name == "nt", reason="real Windows acceptance runs in the owning native workflow")
def test_native_acceptance_refuses_an_unsupported_host():
    result = subprocess.run([sys.executable, "-I", str(PRODUCER)], capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert "Native Windows is required" in result.stderr
    assert not result.stdout
