"""Update restart preserves settings and respects a deferred shutdown."""
from types import SimpleNamespace

import pytest

from spacr import restart_state, updater
from spacr.qt import app as qt_app


class Window:
    _closing = False
    _restart_after_package_upgrade = qt_app.MainWindow._restart_after_package_upgrade
    _on_update_check_done = qt_app.MainWindow._on_update_check_done

    def __init__(self, accepts_close=True):
        self.accepts_close = accepts_close
        self.closed = False
        screen = SimpleNamespace(app_key="mask", _settings_model=SimpleNamespace(
            collect=lambda: {"src": "/images", "nucleus_channel": 2}))
        self._stack = SimpleNamespace(currentWidget=lambda: screen)

    def close(self):
        self.closed = self.accepts_close
        return self.accepts_close


@pytest.fixture
def messages(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACR_HOME", str(tmp_path))
    messages = []
    for kind in ("information", "warning"):
        monkeypatch.setattr(qt_app.QMessageBox, kind,
                            lambda _parent, title, text: messages.append((title, text)))
    return messages


def test_successful_update_saves_settings_and_closes_normally(messages):
    window = Window()
    window._restart_after_package_upgrade()
    assert window.closed
    assert window._restart_after_update
    assert restart_state.peek()["module"] == "mask"
    assert restart_state.peek()["settings"] == {"src": "/images", "nucleus_channel": 2}
    assert not messages


def test_deferred_shutdown_does_not_launch_a_second_copy(messages):
    window = Window(accepts_close=False)
    window._restart_after_package_upgrade()
    assert not window.closed
    assert not window._restart_after_update
    assert restart_state.peek() is None
    assert "deferred" in messages[-1][1]


def test_failed_state_save_leaves_the_application_open(messages, monkeypatch):
    monkeypatch.setattr(restart_state, "save", lambda **kwargs: None)
    window = Window()
    window._restart_after_package_upgrade()
    assert not window.closed
    assert "could not be saved" in messages[-1][1]


def test_source_checkout_explanation_precedes_cleanup_or_install(messages, monkeypatch):
    monkeypatch.setattr(updater, "editable_install_location", lambda: "/source/spacr")
    window = Window()
    window._on_update_check_done(updater.UpdateInfo("1.0", "2.0", None))
    assert "git pull" in messages[-1][1]
    assert "No package upgrade" in messages[-1][1]


def test_running_analysis_prevents_package_mutation(messages, monkeypatch):
    from spacr.qt import bridge
    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    monkeypatch.setattr(bridge, "registry", lambda: SimpleNamespace(is_busy=lambda: True))
    window = Window()
    window._on_update_check_done(updater.UpdateInfo("1.0", "2.0", None))
    assert "active analyses" in messages[-1][1]
