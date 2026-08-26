"""Installing the search strip must not make its screen crash on destruction.

The strip is installed by reparenting the screen's settings scroll area
out of the splitter into a new container. Handing that container the
splitter as its parent AND then inserting it into the splitter parents it
twice, and Shiboken releases the wrapper twice when the screen's children
are deleted.

IT IS A SEGFAULT, which is why it needs a test of its own rather than an
assertion somewhere. The test body passes and the process dies
afterwards, so xdist reports a failed test with no message and takes the
rest of that shard down with it -- a whole Qt shard reporting nonsense
because of one destructor.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PySide6")

REPO = __file__.rsplit("/tests/", 1)[0]


def _run(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter and report how it ended.

    A separate process because the failure being guarded against takes the
    interpreter with it: in-process there would be nothing left to assert.
    """
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {REPO!r})
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        {body}
        print("SURVIVED")
    """)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
        env={"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/bin",
             "HOME": "/tmp", "XDG_CONFIG_HOME": "/tmp/spacr-search-strip"},
    )


DESTROY = """
        from spacr.qt.screens.app_screen import AppScreen
        import spacr.qt.settings_search as ss
        screen = AppScreen("mask")
        ss.install(screen)
        screen.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
"""


def test_a_screen_with_the_strip_can_be_destroyed():
    done = _run(DESTROY)
    assert "SURVIVED" in done.stdout, (
        f"the process died on destruction (exit {done.returncode}): "
        f"{done.stderr[-400:]}")
    assert done.returncode == 0


def test_the_same_screen_without_the_strip_was_never_the_problem():
    """Pins the cause: the screen alone destroys cleanly either way, so a
    regression here is about the strip and not about AppScreen."""
    done = _run("""
        from spacr.qt.screens.app_screen import AppScreen
        screen = AppScreen("mask")
        screen.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    """)
    assert "SURVIVED" in done.stdout
    assert done.returncode == 0


def test_the_container_is_parented_once():
    """The mechanism, asserted where a reader will look for it.

    A container built with the splitter as its parent and then inserted
    into that splitter is parented twice; built with none, `insertWidget`
    is the only thing that parents it.
    """
    from pathlib import Path

    source = Path(REPO, "spacr", "qt", "settings_search.py").read_text()
    assert "container = QWidget()" in source
    assert "container = QWidget(parent)" not in source


def test_the_strip_is_still_installed_afterwards():
    """The fix must not have worked by declining to install anything."""
    done = _run("""
        from spacr.qt.screens.app_screen import AppScreen
        import spacr.qt.settings_search as ss
        screen = AppScreen("mask")
        bar = ss.install(screen)
        assert bar is not None, "the strip did not install"
        assert getattr(screen, "_settings_search", None) is bar
        assert bar.isVisible() or bar.isVisibleTo(screen)
    """)
    assert "SURVIVED" in done.stdout, done.stderr[-400:]
