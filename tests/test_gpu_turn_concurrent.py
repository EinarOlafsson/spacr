"""tools/gpu_turn.sh --vram-gb: concurrent GPU jobs that fit start at once (maintainer, 2026-11-18).

A fake ``nvidia-smi`` on PATH answers from environment variables, and HOME is a
temporary folder, so the real queue in ~/.spacr/gpu is never touched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "gpu_turn.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("flock") is None,
                                reason="the bash queue needs flock(1)")

FAKE = textwrap.dedent("""\
    #!/bin/sh
    case "$*" in
      *query-compute-apps*) printf '%s' "${FAKE_APPS:-}" ;;
      *memory.free*) echo "${FAKE_FREE_MIB:-20000}" ;;
      *temperature.gpu*) echo "${FAKE_TEMP:-40}" ;;
      *power.limit*) echo "${FAKE_POWER:-280.00}" ;;
    esac
""")


def _env(tmp_path, **values):
    """An environment with the fake nvidia-smi first on PATH and a private HOME."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "nvidia-smi"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    env = dict(os.environ, HOME=str(tmp_path / "home"), PATH=f"{bin_dir}:{os.environ['PATH']}",
               GPU_POLL_S="1", GPU_RAM_ALERT=str(tmp_path / "ALERT"))
    env.update({k: str(v) for k, v in values.items()})
    return env


def _run(tmp_path, args, timeout=20, **values):
    """Run gpu_turn.sh; return (rc, stdout, the queue log)."""
    marker = tmp_path / "ran"
    proc = subprocess.run(["bash", str(SCRIPT), *args, "sh", "-c", f"echo ran > {marker}"],
                          env=_env(tmp_path, **values), capture_output=True, text=True, timeout=timeout)
    log = (tmp_path / "home" / ".spacr" / "gpu" / "log")
    return proc.returncode, marker.exists(), log.read_text() if log.exists() else ""


def test_a_job_that_fits_starts_at_once_and_logs_vram_temperature_and_co_runners(tmp_path):
    started = time.monotonic()
    rc, ran, log = _run(tmp_path, ["--vram-gb", "6", "fit-job"], FAKE_FREE_MIB=12000, FAKE_TEMP=55,
                        FAKE_APPS="4242, 9000\n", GPU_IDLE_S=999, GPU_GAP_S=999)
    assert rc == 0 and ran and time.monotonic() - started < 10
    line = [l for l in log.splitlines() if "START concurrent" in l][0]
    assert "needs 6 GB" in line and "free 11.7 GB" in line and "55 C" in line and "power 280.00 W" in line
    assert "4242:9000MiB" in line and "FINISH concurrent rc=0" in log
    assert not list((tmp_path / "home" / ".spacr" / "gpu" / "reserve").iterdir())


@pytest.mark.parametrize("values, reason", [
    ({"FAKE_FREE_MIB": 7000}, "free 6.8 GB"),
    ({"FAKE_TEMP": 82}, "82 C"),
])
def test_a_job_that_does_not_fit_or_a_hot_card_waits(tmp_path, values, reason):
    with pytest.raises(subprocess.TimeoutExpired):
        _run(tmp_path, ["--vram-gb", "6", "wait-job"], timeout=4, **values)
    log = (tmp_path / "home" / ".spacr" / "gpu" / "log").read_text()
    assert "waits:" in log and reason in log and "START" not in log and not (tmp_path / "ran").exists()


def test_an_alert_holds_the_job(tmp_path):
    (tmp_path / "ALERT").write_text("x")
    with pytest.raises(subprocess.TimeoutExpired):
        _run(tmp_path, ["--vram-gb", "2", "alert-job"], timeout=4)
    assert "ram_guard ALERT" in (tmp_path / "home" / ".spacr" / "gpu" / "log").read_text()


def test_a_wrong_power_limit_is_refused(tmp_path):
    rc, ran, log = _run(tmp_path, ["--vram-gb", "2", "power-job"], FAKE_POWER="350.00")
    assert rc == 3 and not ran and "REFUSED" in log and "280 W" in log and "350.00" in log


def test_a_bad_vram_value_is_a_usage_error(tmp_path):
    rc, ran, _log = _run(tmp_path, ["--vram-gb", "lots", "bad-job"])
    assert rc == 2 and not ran


def test_reservations_of_recently_started_jobs_count_against_free_vram(tmp_path):
    reserve = tmp_path / "home" / ".spacr" / "gpu" / "reserve"
    reserve.mkdir(parents=True)
    keeper = subprocess.Popen(["sleep", "30"])
    try:
        (reserve / str(keeper.pid)).write_text(f"{keeper.pid} {int(time.time())} 8\n")
        with pytest.raises(subprocess.TimeoutExpired):
            _run(tmp_path, ["--vram-gb", "6", "second-job"], timeout=4, FAKE_FREE_MIB=12000)
    finally:
        keeper.kill()


def test_without_the_option_the_exclusive_turn_is_unchanged(tmp_path):
    rc, ran, log = _run(tmp_path, ["old-job"], GPU_IDLE_S=0, GPU_GAP_S=0, FAKE_POWER="350.00")
    assert rc == 0 and ran and "holds the queue" in log and "concurrent" not in log
    assert (tmp_path / "home" / ".spacr" / "gpu" / "last_release").exists()
    with pytest.raises(subprocess.TimeoutExpired):
        _run(tmp_path, ["busy-job"], timeout=4, GPU_IDLE_S=30, GPU_GAP_S=0, FAKE_APPS="777, 5000\n")


def test_the_windows_twin_offers_the_same_option():
    """gpu_turn.ps1 cannot run here; check it carries the same rules (Windows CI runs the script)."""
    text = (ROOT / "tools" / "gpu_turn.ps1").read_text(encoding="utf-8")
    assert "'--vram-gb'" in text and "Invoke-SpacrGpuConcurrent" in text and "Test-SpacrGpuFits" in text
    for rule in ("GPU_POWER_LIMIT_W", "GPU_VRAM_MARGIN_GB", "GPU_MAX_TEMP_C", "GPU_RESERVE_S", "ram_guard",
                 "return 3", "START concurrent", "Invoke-SpacrGpuTurn -TurnLabel"):
        assert rule in text, rule
