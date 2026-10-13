"""Instruction 314: the "cheap" lock peek was importing numba.

``_the_heavy_import_lock_is_free`` is documented as "the cheap half of
the pair" and is called from ``AppScreen._install_ambient`` while a
module screen is being built. It reached the lock with

    from .fractal_travel import _heavy_import_lock

which IMPORTS that module if nothing has yet -- and it pulls numba.
Measured at 0.44 s in a cold interpreter, spent on the GUI thread, in
the middle of the build the user is waiting for.

That is exactly the shape 314 is about: not a slow operation, but the
GUI thread held without yielding, which is indistinguishable from a
hang while it lasts.

NOT IMPORTED MEANS NOT LOCKED. The lock lives in that module, so nothing
can hold it while the module has never been imported -- the only code
that takes it had to import it first. Answering from ``sys.modules`` is
exact, not an approximation.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PySide6")


def _in_a_cold_interpreter(body: str) -> str:
    """Run ``body`` in a fresh process, so imports are genuinely cold."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, timeout=300,
        env={"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/bin",
             "HOME": "/tmp"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


def test_the_peek_does_not_import_the_heavy_module():
    """THE FIX. A cold peek must not pull fractal_travel, and therefore
    must not pull numba."""
    out = _in_a_cold_interpreter("""
        import sys
        from spacr.qt.widgets.ambient import (
            _the_heavy_import_lock_is_free as peek)
        answer = peek()
        print(answer,
              "spacr.qt.widgets.fractal_travel" in sys.modules,
              "numba" in sys.modules)
    """)
    answer, pulled_module, pulled_numba = out.split()
    assert answer == "True", "a cold tree has no heavy import in progress"
    assert pulled_module == "False", (
        "the peek imported fractal_travel; that is the 0.44 s this test "
        "exists to prevent")
    assert pulled_numba == "False", "the peek pulled numba onto the caller"


def test_the_peek_is_fast_enough_to_be_called_during_a_build():
    """A number, because "cheap" was the claim that turned out false."""
    out = _in_a_cold_interpreter("""
        import time
        from spacr.qt.widgets.ambient import (
            _the_heavy_import_lock_is_free as peek)
        started = time.perf_counter()
        peek()
        print(f"{(time.perf_counter() - started) * 1000:.4f}")
    """)
    milliseconds = float(out)
    assert milliseconds < 50.0, (
        f"the peek took {milliseconds:.1f} ms; it is called while a "
        f"module screen is being built")


def test_it_still_answers_correctly_once_the_module_is_loaded():
    """The fast path must not become the only path.

    With the module imported the peek does the real thing: try the lock,
    release it, and say what it found.
    """
    from spacr.qt.widgets import fractal_travel  # noqa: F401
    from spacr.qt.widgets.ambient import _the_heavy_import_lock_is_free

    assert _the_heavy_import_lock_is_free() is True


def test_a_held_lock_is_reported_as_busy():
    """THE GUARANTEE THE PEEK EXISTS FOR. It decides whether to try
    building a GL context at all this turn, so a lock held by the
    preloader has to come back False."""
    from spacr.qt.widgets.fractal_travel import _heavy_import_lock
    from spacr.qt.widgets.ambient import _the_heavy_import_lock_is_free

    lock = _heavy_import_lock()
    if lock is None:
        pytest.skip("this build has no heavy-import lock")

    assert lock.acquire(blocking=False), "the lock was already held"
    try:
        assert _the_heavy_import_lock_is_free() is False
    finally:
        lock.release()

    assert _the_heavy_import_lock_is_free() is True


def test_the_shortcut_is_exact_rather_than_a_guess():
    """WHY sys.modules is a correct answer and not an approximation.

    The lock object lives in fractal_travel, so acquiring it requires
    having imported that module. An unimported module therefore cannot
    have a held lock -- there is no path to one.
    """
    import inspect

    from spacr.qt.widgets import fractal_travel

    source = inspect.getsource(fractal_travel)
    assert "_heavy_import_lock" in source, (
        "the lock moved out of fractal_travel; the sys.modules shortcut "
        "in ambient.py names that module by hand and would now be wrong")
