"""A sweep trial is capped by the kernel, not by spaCR's own accounting.

Seven attempts to sweep this screen took the user's desktop down. Each fix
was a better ESTIMATE of what a trial would use, and each estimate was wrong
in a way that cost a working day. These pin the difference.
"""
import os
import subprocess
import sys

import pytest


class TestTheCapIsReal:

    def test_a_hog_is_killed_at_its_limit_and_the_host_is_untouched(self):
        """The property everything else depends on.

        If this passes, no configuration can take the machine down, because
        the kernel stops it whatever spaCR believed it would use.
        """
        from spacr.parameter_sweep import containment_available, free_memory_gb

        if not containment_available():
            pytest.skip("systemd-run is unavailable on this host")

        before = free_memory_gb()
        script = (
            "import numpy as np\n"
            "blocks = []\n"
            "for i in range(200):\n"
            "    blocks.append(np.ones((1024, 1024, 8)))\n"
            "    print(i, flush=True)\n"
            "print('SURVIVED')\n"
        )
        finished = subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet",
             "-p", "MemoryMax=2G", "-p", "MemorySwapMax=0",
             sys.executable, "-c", script],
            capture_output=True, text=True, timeout=180)

        assert "SURVIVED" not in (finished.stdout or ""), \
            "the hog ran past its cap; nothing is containing it"
        # It must die by SIGKILL from the cgroup, not by failing to start:
        # a test that passes because numpy was missing proves nothing.
        assert finished.returncode == -9, \
            f"expected SIGKILL from the cap, got {finished.returncode}"
        printed = (finished.stdout or "").split()
        assert printed, "the hog never allocated anything"
        blocks = int(printed[-1])
        # ~64 MB each: it should reach roughly the 2 GiB cap and no further.
        assert 20 <= blocks <= 40, \
            f"died after {blocks} blocks; the cap is not where it should be"
        # And the host never noticed.
        assert free_memory_gb() > before - 4, \
            "free memory moved while a capped process ran"

    def test_the_command_carries_every_limit(self, tmp_path, monkeypatch):
        """A cap that is not passed is not a cap."""
        import spacr.parameter_sweep as sweep

        seen = {}

        class _Result:
            returncode = 0
            stderr = ""

        def fake_run(command, **kwargs):
            seen["command"] = command
            seen["env"] = kwargs.get("env") or {}
            return _Result()

        monkeypatch.setattr(sweep, "containment_available", lambda: True)
        monkeypatch.setattr(subprocess, "run", fake_run)
        sweep.run_trial_contained({"src": str(tmp_path)}, trial_id=1)

        text = " ".join(seen["command"])
        assert "MemoryMax=" in text
        assert "MemorySwapMax=0" in text, "swap would defeat the memory cap"
        assert "CPUQuota=" in text
        assert "TasksMax=" in text
        # The thread pin must reach the CHILD's environment: OpenBLAS reads it
        # at import and never again.
        assert seen["env"].get("OMP_NUM_THREADS") == "1"
        assert seen["env"].get("OPENBLAS_NUM_THREADS") == "1"

    def test_an_uncapped_run_says_so_rather_than_pretending(self, tmp_path,
                                                            monkeypatch,
                                                            capsys):
        """Running without a cap is a decision the user should make knowingly."""
        import spacr.parameter_sweep as sweep

        class _Result:
            returncode = 0
            stderr = ""

        monkeypatch.setattr(sweep, "containment_available", lambda: False)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        sweep.run_trial_contained({"src": str(tmp_path)}, trial_id=1)

        assert "WITHOUT a memory cap" in capsys.readouterr().out

    def test_a_killed_trial_is_a_row_not_an_exception(self, tmp_path,
                                                      monkeypatch):
        """The sweep has to carry on to the next configuration."""
        import spacr.parameter_sweep as sweep

        class _Result:
            returncode = 137          # SIGKILL, which is what the cap sends
            stderr = ""

        monkeypatch.setattr(sweep, "containment_available", lambda: True)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        row = sweep.run_trial_contained({"src": str(tmp_path)}, trial_id=7)

        assert row["status"] == "killed"
        assert "MemoryMax" in row["error"]

    def test_a_timeout_is_distinguished_from_a_kill(self, tmp_path,
                                                    monkeypatch):
        """They want different responses: lower the cap, or raise the clock."""
        import spacr.parameter_sweep as sweep

        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

        monkeypatch.setattr(sweep, "containment_available", lambda: True)
        monkeypatch.setattr(subprocess, "run", timeout)
        row = sweep.run_trial_contained({"src": str(tmp_path)}, trial_id=7)
        assert row["status"] == "timeout"


class TestTheMemoryFloor:

    def test_the_sweep_stops_before_the_machine_is_in_trouble(self, tmp_path,
                                                              monkeypatch):
        """Reacting once memory is gone is too late; this refuses to start."""
        import pandas as pd

        import spacr.parameter_sweep as sweep

        monkeypatch.setattr(sweep, "free_memory_gb", lambda: 1.0)
        space = sweep.SweepSpace(axes={"regression_type": ["ols", "rlm"]},
                                 fixed={}, filters=[])
        frame = sweep.run_sweep({}, str(tmp_path), space, contained=True,
                                memory_floor_gb=20.0, progress_every=0)
        assert len(frame) == 1
        assert frame.iloc[0]["status"] == "skipped"
        assert "floor" in str(frame.iloc[0]["error"])
