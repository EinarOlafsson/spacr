"""remote_execution's state lock excludes other processes on Windows too.

Found natively on Windows 11 2026-10-19: the lock imported fcntl and, on ImportError, carried on unlocked,
so two spaCR processes could rewrite remote-execution state at once. Windows now locks with msvcrt.
"""
import subprocess
import sys

import pytest

from spacr import remote_execution

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="msvcrt byte-range locks")

PROBE = """import msvcrt, sys
with open(sys.argv[1], "a+") as handle:
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("held")
    else:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        print("free")
"""


def _probe(path):
    done = subprocess.run([sys.executable, "-c", PROBE, str(path)], capture_output=True, text=True, timeout=60)
    return done.stdout.strip()


def test_a_second_process_waits_for_the_state_lock(tmp_path):
    path = tmp_path / "state" / ".lock"
    with remote_execution._file_lock(path):
        assert _probe(path) == "held"
    assert _probe(path) == "free"
