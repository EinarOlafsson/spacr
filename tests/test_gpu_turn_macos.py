"""tools/gpu_turn.sh queues GPU turns on macOS, which has no flock(1).

Before the mkdir lock, ``flock: command not found`` let every turn start at
once and every log line read ``date: invalid argument 's' for -I``.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="BSD userland without flock(1)")

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "gpu_turn.sh"


def _env(home):
    return dict(os.environ, HOME=str(home), GPU_IDLE_S="0", GPU_GAP_S="0", GPU_POLL_S="1")


def test_two_turns_run_one_after_the_other(tmp_path):
    marks = tmp_path / "marks"
    first = subprocess.Popen(
        ["bash", str(SCRIPT), "first", "sh", "-c", f"echo start1 >> {marks}; sleep 2; echo end1 >> {marks}"],
        env=_env(tmp_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    second = subprocess.run(
        ["bash", str(SCRIPT), "second", "sh", "-c", f"echo start2 >> {marks}"],
        env=_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert first.wait(timeout=60) == 0 and second.returncode == 0
    assert marks.read_text().split() == ["start1", "end1", "start2"]
    assert not (tmp_path / ".spacr/gpu/lock.d").exists()
    log = (tmp_path / ".spacr/gpu/log").read_text()
    assert "invalid argument" not in second.stderr
    assert re.search(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4} \[second\] FINISH rc=0$", log, re.M)


def test_a_dead_holders_lock_is_reclaimed(tmp_path):
    stale = tmp_path / ".spacr/gpu/lock.d"
    stale.mkdir(parents=True)
    dead = subprocess.Popen(["true"])
    dead.wait()
    (stale / "pid").write_text(f"{dead.pid}\n")
    done = subprocess.run(["bash", str(SCRIPT), "after-crash", "echo", "ran"],
                          env=_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and done.stdout.strip() == "ran"
    assert not stale.exists()
