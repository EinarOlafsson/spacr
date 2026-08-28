"""Heavy libraries are imported when something needs them, not at startup.

The maintainer's own timing report is the evidence. On a real launch the
preload thread ground for twenty seconds importing spacr.core (15.6 s),
spacr.deep_spacr (9.8 s), torchvision (8.5 s), torch (6.8 s), torch._dynamo
(5.9 s), IPython (2.5 s), sympy (2.9 s) and torch.distributed.fsdp (1.8 s)
-- the torch COMPILER, DISTRIBUTED TRAINING and a REPL, to draw a window.
15 GUI stalls totalling 10.96 s frozen, the worst 5.4 s.
"""

import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt import preferences as prefs  # noqa: E402

#: Nothing in this list should be imported by opening a window.
HEAVY = ("torch", "torchvision", "IPython", "sympy", "cellpose")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return tmp_path


def test_on_demand_is_the_default(sandbox):
    assert prefs.get_preload_policy() == "on_demand"


def test_the_policy_round_trips(sandbox):
    for policy in prefs.PRELOAD_POLICIES:
        prefs.set_preload_policy(policy)
        assert prefs.get_preload_policy() == policy


def test_an_unknown_policy_is_refused(sandbox):
    with pytest.raises(ValueError, match="unknown preload policy"):
        prefs.set_preload_policy("whenever")


def test_an_unreadable_stored_policy_reads_as_on_demand(sandbox):
    prefs._settings().setValue(prefs._KEY_PRELOAD, "sideways")
    assert prefs.get_preload_policy() == "on_demand"


def test_opening_a_window_builds_no_preloader(qtbot, sandbox):
    """The measurement that matters, as a state check rather than a timing:
    a timing on a shared CI box proves nothing, a None does."""
    import spacr.qt.app as A

    win = A.MainWindow()
    qtbot.addWidget(win)
    assert win._preloader is None


def test_eager_still_builds_one(qtbot, sandbox):
    """The old behaviour stays reachable for a machine that would rather
    wait once at the beginning."""
    import spacr.qt.app as A

    prefs.set_preload_policy("eager")
    win = A.MainWindow()
    qtbot.addWidget(win)
    assert win._preloader is not None


@pytest.mark.parametrize("module", HEAVY)
def test_a_window_does_not_import_it(qtbot, sandbox, module):
    """Asserted on sys.modules, not on a clock. If this fails, something
    put a heavy import back on the launch path."""
    if module in sys.modules:
        pytest.skip(f"{module} was already imported by another test")
    import spacr.qt.app as A

    win = A.MainWindow()
    qtbot.addWidget(win)
    assert module not in sys.modules


def test_the_preloader_still_works_when_asked(qtbot, sandbox):
    """Turning it off must not break it -- 'eager' has to remain honest."""
    from PySide6.QtCore import QEventLoop, QTimer

    from spacr.qt.app import _PipelinePreloader

    loop = QEventLoop()
    preloader = _PipelinePreloader(on_done=loop.quit)
    preloader.start()
    QTimer.singleShot(60_000, loop.quit)
    loop.exec()
    assert preloader.wait(30.0)
