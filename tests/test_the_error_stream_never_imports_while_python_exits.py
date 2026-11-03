"""The wrapped stderr must not import anything while the interpreter exits.

CI run 41584400706 hung for 300 s in an ordinary offscreen start of spaCR
that had already quit. faulthandler in the child showed the main thread at
interpreter finalization, inside the final flush of ``sys.stderr``:
``_ErrorStream.flush`` -> ``from .crashreport import ...`` ->
``_find_and_load`` -> ``_get_module_lock`` -> ``_imp.acquire_lock()``.

At finalization every daemon thread is frozen where it next asks for the
interpreter lock, and one frozen while it owned the global import lock never
gives it back. Any import the main thread makes after that point waits
forever, so a spaCR window could close and its process never end.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_CHILD = textwrap.dedent('''
    import atexit, sys, threading, time
    import _imp

    take = threading.Event()
    taken = threading.Event()

    def hold():
        take.wait()
        _imp.acquire_lock()
        taken.set()
        while True:
            time.sleep(0.05)

    def a_daemon_owns_the_import_lock_as_finalization_starts():
        sys.modules.pop("spacr.crashreport", None)
        take.set()
        taken.wait()

    atexit.register(a_daemon_owns_the_import_lock_as_finalization_starts)
    assert "logging" not in sys.modules
    sys.path.insert(0, {repo!r})
    from spacr import logging_util

    sys.stderr = logging_util._ErrorStream(sys.stderr)
    sys.stdout = logging_util._ErrorStream(sys.stdout)
    threading.Thread(target=hold, daemon=True).start()
    sys.stdout.write("CHILD-REACHED-EXIT\\n")
''')


def _unmeasured_environment():
    """This environment without coverage.py's subprocess measurement.

    The child's lock-taking exit handler must be the last Python code that
    runs before finalization, so it is registered first. A measured child
    breaks that: for ``--cov=spacr.logging_util`` coverage.py finds the
    module's file at start-up by importing ``spacr``, which imports
    ``logging``; that module is then dropped from ``sys.modules`` but its
    ``shutdown`` is already registered ahead of the handler. It runs after
    the handler and flushes ``sys.stderr`` while the interpreter is not yet
    finalizing, so the child waited for the held lock until the 60 s
    timeout killed it.
    """
    return {key: value for key, value in os.environ.items()
            if key not in ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG")}


def test_a_frozen_import_lock_does_not_hang_the_exit():
    """Exit completes although a daemon owns the import lock at finalization."""
    child = subprocess.run(
        [sys.executable, "-c", _CHILD.format(repo=str(REPO))],
        capture_output=True, text=True, timeout=60,
        env=_unmeasured_environment())

    assert "CHILD-REACHED-EXIT" in child.stdout, child.stderr
    assert child.returncode == 0, child.stderr


def test_the_stream_still_captures_while_python_runs(monkeypatch):
    """The finalization guard leaves a live interpreter's capture alone."""
    import io

    from spacr import crashreport, logging_util

    seen = []
    monkeypatch.setattr(crashreport, "_capture_error_line", seen.append)
    stream = logging_util._ErrorStream(io.StringIO())

    stream.write("ERROR one whole line\n")
    stream.write("ERROR a partial line")
    stream.flush()

    assert seen == ["ERROR one whole line\n", "ERROR a partial line"]


def test_a_finalizing_interpreter_passes_text_through_without_capturing(monkeypatch):
    """While Python exits the stream writes and flushes, and captures nothing."""
    import io

    from spacr import crashreport, logging_util

    seen = []
    monkeypatch.setattr(crashreport, "_capture_error_line", seen.append)
    terminal = io.StringIO()
    stream = logging_util._ErrorStream(terminal)
    stream.write("ERROR a partial line")
    monkeypatch.setattr(sys, "is_finalizing", lambda: True)

    assert stream.write("ERROR written while exiting\n") == len(
        "ERROR written while exiting\n")
    stream.flush()

    assert seen == []
    assert terminal.getvalue() == "ERROR a partial lineERROR written while exiting\n"
