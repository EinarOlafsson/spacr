"""A forked child must not hide the stack of its timed-out xdist worker."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="inspects Linux process state")
@pytest.mark.parametrize("func_only", [False, True])
def test_forked_child_cannot_hide_the_workers_timeout(tmp_path, func_only):
    pytest.importorskip("pytest_timeout")
    pytest.importorskip("xdist")
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(
        "from tests.conftest import pytest_timeout_set_timer\n")
    (tmp_path / "test_stalled.py").write_text('''\
import json
import os
from pathlib import Path
import threading
import time

def test_waiting_for_a_forked_child():
    worker = os.getpid()
    child = os.fork()
    if child == 0:
        limit = time.monotonic() + 30
        while not Path("release-child").exists() and time.monotonic() < limit:
            time.sleep(0.02)
        os._exit(0)
    Path("pids.tmp").write_text(json.dumps({"worker": worker, "child": child}))
    Path("pids.tmp").replace("pids.json")
    print("the captured output is retained", flush=True)
    threading.Event().wait(30)
''')
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
        part for part in (str(root), os.environ.get("PYTHONPATH", "")) if part),
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment.pop("PYTEST_ADDOPTS", None)
    command = [sys.executable, "-m", "pytest", "test_stalled.py",
               "-c", "pytest.ini", "-p", "xdist.plugin", "-p", "pytest_timeout",
               "-n", "2", "--dist", "loadfile", "--max-worker-restart=0",
               "--timeout", "2", "--timeout-method", "thread",
               "-o", f"timeout_func_only={str(func_only).lower()}",
               "-o", "faulthandler_timeout=4", "-v"]
    log = tmp_path / "output.log"
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=tmp_path, env=environment,
                                   stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 20
            worker_state = None
            while time.monotonic() < deadline and process.poll() is None:
                pids_path = tmp_path / "pids.json"
                if pids_path.exists():
                    pids = json.loads(pids_path.read_text(encoding="utf-8"))
                    status = Path(f"/proc/{pids['worker']}/stat")
                    if status.exists():
                        worker_state = status.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()[0]
                        if worker_state == "Z":
                            break
                time.sleep(0.02)
            text = log.read_text(encoding="utf-8")
            assert worker_state == "Z", text
            assert process.poll() is None, "the inherited transport closed early"
            assert "Timeout" in text, text
            assert "test_waiting_for_a_forked_child" in text, text
            assert "threading.Event().wait(30)" in text, text
            assert "the captured output is retained" in text, text
            (tmp_path / "release-child").touch()
            assert process.wait(timeout=10) == 1, log.read_text(encoding="utf-8")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
