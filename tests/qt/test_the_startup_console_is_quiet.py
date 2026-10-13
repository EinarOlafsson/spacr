"""Two lines that greeted every start, neither of which the user could act on.

    Qt warning: inotify_add_watch(...ibus...) failed: (No space left on device)
    home.py:830 FutureWarning: The pynvml package is deprecated.

The inotify one is NOT spaCR's. Qt reports ENOSPC as "No space left on
device", which reads as a full disk and is not: it means the per-user
inotify WATCH limit is exhausted. On the machine it was reported from,
``fs.inotify.max_user_watches`` was 65,536 with 65,434 already taken --
45,078 held by syncthing and 20,019 by VS Code, and none by spaCR. It fired
twice per start, so it is filtered; it is explained once rather than
dropped, because someone seeing inotify failures in one application is
about to see them in others, and the error code lies about the cause.

The pynvml one IS spaCR's. ``nvidia-ml-py`` is the maintained package and
``pynvml`` the retired one; both install a module named ``pynvml``, so only
the installed distribution differs and the retired one warns on import. A
global filter for it already existed in ``spacr/__init__.py`` and did not
hold, because the warning is raised while the module body runs and anything
that has reset the filters by then lets it through. The suppression is at
the import itself now, where nobody else can reset it.

The filter must stay narrow: a Qt warning about a real problem is often the
only clue there is.
"""
from __future__ import annotations

import io
import sys
import warnings

import pytest


INOTIFY_LINE = ("inotify_add_watch(/home/u/.config/ibus/bus/abc-unix-0) "
                "failed: (No space left on device)")


@pytest.fixture
def qt_handler(monkeypatch):
    """The installed Qt message handler, with its once-only flag reset."""
    import spacr.qt as Q

    monkeypatch.setattr(Q, "_SAID_IT_ONCE", False, raising=False)
    Q._install_quiet_qt_logging()
    from PySide6.QtCore import qWarning
    return qWarning


def _stderr_of(emit):
    buffer = io.StringIO()
    real, sys.stderr = sys.stderr, buffer
    try:
        emit()
    finally:
        sys.stderr = real
    return buffer.getvalue()


def test_the_inotify_line_itself_is_not_printed(qt_handler):
    out = _stderr_of(lambda: [qt_handler(INOTIFY_LINE) for _ in range(2)])
    assert "inotify_add_watch" not in out


def test_it_is_explained_once_not_twice(qt_handler):
    """It fires twice per start; the explanation is not worth saying twice."""
    out = _stderr_of(lambda: [qt_handler(INOTIFY_LINE) for _ in range(5)])
    assert out.count("run out of inotify") == 1


def test_the_explanation_corrects_the_error_code(qt_handler):
    """"No space left on device" is about watches, and must say so."""
    out = _stderr_of(lambda: qt_handler(INOTIFY_LINE))
    assert "watch" in out.lower()
    assert "not disk" in out.lower() or "not disk space" in out.lower()
    assert "max_user_watches" in out


def test_a_real_qt_warning_still_reaches_the_console(qt_handler):
    """The filter may not become a mute."""
    out = _stderr_of(lambda: qt_handler("QOpenGLShader: could not compile"))
    assert "could not compile" in out


def test_nvml_imports_without_a_deprecation_warning():
    """With FutureWarning as an error, the helper must still return NVML or None."""
    from spacr.qt.widgets import home

    home._NVML = home._UNSET          # probe again rather than reuse the cache
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        nvml = home._nvml()           # must not raise
    assert nvml is None or hasattr(nvml, "nvmlDeviceGetHandleByIndex")


def test_a_machine_with_no_gpu_is_probed_once(monkeypatch):
    """The panel refreshes on a timer; a failing import may not run every time."""
    from spacr.qt.widgets import home

    home._NVML = home._UNSET
    calls = []

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def counting_import(name, *args, **kwargs):
        if name == "pynvml":
            calls.append(name)
            raise ImportError("no pynvml here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", counting_import)
    assert home._nvml() is None
    assert home._nvml() is None
    assert len(calls) == 1, "the failed import was retried"


def test_the_gpu_rows_survive_having_no_nvml(monkeypatch):
    """Whatever happens, the panel shows a string rather than raising."""
    from spacr.qt.widgets import home

    monkeypatch.setattr(home, "_nvml", lambda: None)
    assert isinstance(home.SystemPanel.gpu_util(), str)
    assert isinstance(home.SystemPanel.gpu_vram(), str)
