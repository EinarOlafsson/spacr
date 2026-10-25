"""Pool workers and the Measure manager end when their parent is killed on Windows.

Before, a parent killed by a crash, Task Manager or a test timeout left its
spawn workers and Manager server running for good, each holding memory and
the parent's pipes.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object behaviour')

ROOT = Path(__file__).resolve().parents[1]

PARENT = r'''
import multiprocessing as mp
import sys
import time
from pathlib import Path

if __name__ == '__main__':
    from spacr.measure import _start_manager
    from spacr.resource_log import _WorkerStartGate, _parallel_pool

    ctx = mp.get_context('spawn')
    manager = _start_manager(ctx)
    pool = _parallel_pool(2, context=ctx, gate=_WorkerStartGate(0))
    pool.map(abs, [-1, -2, -3, -4])
    Path(sys.argv[1]).write_text('ready')
    time.sleep(600)
'''


def _alive(pids):
    import psutil

    return [pid for pid in pids if psutil.pid_exists(pid)
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE]


def test_killing_the_parent_ends_its_pool_workers_and_manager(tmp_path):
    import psutil

    script, ready = tmp_path / 'parent.py', tmp_path / 'ready'
    script.write_text(PARENT, encoding='utf-8')
    with (tmp_path / 'parent.log').open('wb') as log:
        parent = subprocess.Popen([sys.executable, str(script), str(ready)], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 180
        while not ready.exists():
            assert parent.poll() is None, (tmp_path / 'parent.log').read_text(errors='replace')
            assert time.monotonic() < deadline, 'parent never started its pool'
            time.sleep(0.2)
        children = [child.pid for child in psutil.Process(parent.pid).children(recursive=True)]
        assert len(children) >= 3, children
        parent.kill()
        parent.wait(30)
        deadline = time.monotonic() + 15
        while _alive(children) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert _alive(children) == []
    finally:
        if parent.poll() is None:
            parent.kill()
        for pid in _alive(locals().get('children', [])):
            psutil.Process(pid).kill()


def test_binding_is_a_no_op_off_windows_and_for_no_pid(monkeypatch):
    from types import SimpleNamespace
    from spacr import resource_log

    assert resource_log._bind_to_parent_job(None) is False
    monkeypatch.setattr(resource_log, 'os', SimpleNamespace(name='posix'))
    assert resource_log._bind_to_parent_job(os.getpid()) is False
