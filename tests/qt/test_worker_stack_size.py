"""Worker threads get a stack the pipeline can run in.

Defends the 2026-08-14 classify SIGBUS: "Thread stack size exceeded due to
excessive recursion" in ``___chkstk_darwin`` under OpenBLAS's
``dgetrf_parallel``, reached from ``np.linalg.inv`` in ``spacr/ml.py`` (the
Mahalanobis inverse covariance). Nothing recursed — ``chkstk`` is the probe
that finds the guard page, and macOS guesses "recursion" for any overflow.
The crash report put the thread's stack at 544 KB, because a pthread on macOS
defaults to 512 KB and Qt does not raise it; the main thread gets 8 MB, which
is why the same code is fine from the CLI and dies in the GUI.

Reproduced and fixed in the environment that crashed
(miniconda envs/spacr, python 3.10.13), `np.linalg.inv` on a 1200x1200
covariance on a real QThread:

    default stack ....... exit 138 (SIGBUS)
    64 MB stack ......... exit 0

These tests pin the wiring, not the arithmetic: that the size is asked for
before the thread starts, that the override is honoured, and that a platform
refusing the request cannot itself end a run.
"""

import pytest
from PySide6.QtCore import QThread

from spacr.qt import bridge


class TestWidenWorkerStack:
    def test_asks_for_the_documented_size(self, monkeypatch):
        asked = []
        thread = QThread()
        monkeypatch.setattr(type(thread), "setStackSize",
                            lambda self, n: asked.append(n), raising=False)
        monkeypatch.delenv("SPACR_WORKER_STACK_MB", raising=False)

        bridge._widen_worker_stack(thread)

        assert asked == [bridge.WORKER_STACK_BYTES]

    def test_the_default_clears_the_crashing_size(self):
        """544 KB was not enough. Anything in this range must be far above it."""
        assert bridge.WORKER_STACK_BYTES >= 16 * 1024 * 1024

    def test_env_override_is_honoured(self, monkeypatch):
        asked = []
        thread = QThread()
        monkeypatch.setattr(type(thread), "setStackSize",
                            lambda self, n: asked.append(n), raising=False)
        monkeypatch.setenv("SPACR_WORKER_STACK_MB", "8")

        bridge._widen_worker_stack(thread)

        assert asked == [8 * 1024 * 1024]

    @pytest.mark.parametrize("value", ["", "  ", "banana", "-1", "0"])
    def test_a_junk_override_never_shrinks_the_stack(self, monkeypatch, value):
        """An unparseable or absurd value must not leave a smaller stack."""
        asked = []
        thread = QThread()
        monkeypatch.setattr(type(thread), "setStackSize",
                            lambda self, n: asked.append(n), raising=False)
        monkeypatch.setenv("SPACR_WORKER_STACK_MB", value)

        bridge._widen_worker_stack(thread)

        assert asked == [bridge.WORKER_STACK_BYTES]

    def test_a_refusing_platform_costs_the_run_nothing(self, monkeypatch):
        """INVARIANTS §10 — this must not be able to end a run."""
        thread = QThread()

        def boom(self, n):
            raise RuntimeError("setStackSize refused")

        monkeypatch.setattr(type(thread), "setStackSize", boom, raising=False)
        # Returns rather than raising, and reports that it could not widen --
        # a platform refusing a bigger stack costs the run nothing, which is
        # what INVARIANTS 10 requires.
        assert bridge._widen_worker_stack(thread) is not True

    def test_make_thread_widens_before_start(self, monkeypatch):
        """Qt ignores setStackSize on a running thread, so order is the test."""
        events = []
        real_start = QThread.start
        monkeypatch.setattr(type(QThread()), "setStackSize",
                            lambda self, n: events.append("size"), raising=False)
        monkeypatch.setattr(QThread, "start",
                            lambda self, *a, **k: events.append("start"))

        thread, worker = bridge.make_thread(lambda s: None, {}, "test",
                                            journal=False)
        try:
            thread.start()
            assert events[:2] == ["size", "start"]
        finally:
            monkeypatch.setattr(QThread, "start", real_start)
