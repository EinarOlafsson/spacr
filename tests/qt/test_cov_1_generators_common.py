"""A standalone render sandboxes its preferences before it makes the app.

``common.bootstrap()`` has two modes and only the guest one is exercised by
the test suite, which always has a QApplication already. The owner mode is the
one the reviewer actually runs, and its whole job is determinism: point
QSettings at a throwaway directory FIRST, so nothing the operator saved --
their font scale, their theme -- reaches a render that is supposed to look the
same on every machine.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

GENERATORS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "spacr", "resources", "home", "versions", "_generators"))


def _load_common():
    """Load ``common.py`` under a private name, restoring sys.modules."""
    path = os.path.join(GENERATORS, "common.py")
    spec = importlib.util.spec_from_file_location("_cov1_home_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cov1_home_common"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def common(qapp):
    """The generator's ``common`` module, with its ownership flag restored."""
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
    module = _load_common()
    try:
        yield module
    finally:
        sys.modules.pop("_cov1_home_common", None)


def test_a_guest_bootstrap_leaves_the_hosts_settings_alone(common, qapp):
    """With an application already running, nothing global is touched.

    ``QSettings.setPath`` is process-wide: redirecting it here would repoint
    every other test's preferences at a temp directory mid-session.
    """
    assert common.bootstrap() is qapp
    assert common._WE_OWN_THE_APP is False


def test_owning_the_process_redirects_settings_before_the_app_exists(
        common, monkeypatch, tmp_path):
    """The sandbox is installed BEFORE the QApplication is constructed.

    Order is the whole point: ``QApplication`` reads preferences while it is
    being built, so a redirect applied afterwards is a redirect that came too
    late and the render still carries the operator's own font scale.
    """
    from PySide6 import QtCore, QtWidgets

    events = []

    class _Settings:
        """Stand-in for QSettings recording the process-global redirects."""

        IniFormat = "ini"
        NativeFormat = "native"
        UserScope = "user"

        @staticmethod
        def setDefaultFormat(fmt):
            events.append(("default-format", fmt))

        @staticmethod
        def setPath(fmt, scope, path):
            events.append(("path", fmt, scope, path))

    class _App:
        """Stand-in for QApplication: never a second real one in this process."""

        def __init__(self, argv):
            events.append(("app", tuple(argv)))
            self.argv = list(argv)

        @staticmethod
        def instance():
            return None

    monkeypatch.setattr(QtCore, "QSettings", _Settings)
    monkeypatch.setattr(QtWidgets, "QApplication", _App)
    monkeypatch.setattr(common, "_WE_OWN_THE_APP", False)
    monkeypatch.setattr(common.tempfile, "mkdtemp",
                        lambda prefix="": str(tmp_path / "sandbox"))
    (tmp_path / "sandbox").mkdir()
    loaded = []
    monkeypatch.setattr(common, "_load_fonts", lambda: loaded.append(True))

    app = common.bootstrap()

    assert isinstance(app, _App)
    assert common._WE_OWN_THE_APP is True
    assert loaded == [True], "the bundled fonts are what the metrics assume"

    kinds = [event[0] for event in events]
    assert kinds.index("app") == len(kinds) - 1, (
        f"the settings sandbox must be installed before the application: "
        f"{events}")
    # BOTH formats: preferences._settings() builds a NativeFormat QSettings,
    # so redirecting Ini alone left every render reading the real store.
    redirected = {event[1] for event in events if event[0] == "path"}
    assert redirected == {_Settings.NativeFormat, _Settings.IniFormat}
    assert all(event[3] == str(tmp_path / "sandbox")
               for event in events if event[0] == "path")
    assert ("default-format", _Settings.IniFormat) in events
