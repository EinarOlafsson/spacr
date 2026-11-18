"""Item 973: the Voyager3 button runs only its phase on the panel's settings and follows its progress."""
from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from spacr.qt import voyager3_panel as P


class _Model:
    """Hands out fixed settings like the OPS settings model."""

    def __init__(self, settings):
        """Keep ``settings``."""
        self.settings = settings

    def collect(self):
        """Return a copy of the settings."""
        return dict(self.settings)


class _Section(QWidget):
    """Takes prose widgets like a settings section."""

    def __init__(self):
        """Lay the widgets out top to bottom."""
        super().__init__()
        self.added = []
        QVBoxLayout(self)

    def add_prose(self, widget, at_top=False):
        """Record and show ``widget``."""
        self.added.append((widget, at_top))
        self.layout().addWidget(widget)


def _fake_phase(settings):
    """Print two progress lines and return what was asked."""
    print("OPS: Progress: 1/2, operation_type: Voyager3, cluster 1 (real)", flush=True)
    print("OPS: Progress: 2/2, operation_type: Voyager3, cluster 2 (real)", flush=True)
    return {"asked": settings["ops_voyager3_cluster_column"]}


def test_the_button_runs_the_phase_and_the_bar_follows(qtbot, monkeypatch):
    screen = QWidget()
    screen._settings_model = _Model({"ops_voyager3_cluster_column": "leiden_r10"})
    section = _Section()
    qtbot.addWidget(section)
    monkeypatch.setattr(P, "_run_voyager3_phase", _fake_phase)
    monkeypatch.setattr(P.Voyager3Controls, "_refresh_visibility", lambda self: True)
    controls = P.install_voyager3_controls(screen, section)
    assert section.added == [(controls, True)] and P.install_voyager3_controls(screen, section) is None
    lines = []
    assert controls._start()
    controls._worker.line_ready.connect(lines.append)
    assert not controls.button.isEnabled() and not controls._start()
    qtbot.waitUntil(lambda: not controls._is_running(), timeout=20000)
    assert controls.button.isEnabled() and controls.bar.value() == 100
    assert "results/ops_voyager3" in controls.status.text()


def test_the_phase_entry_runs_only_voyager3(monkeypatch):
    from spacr import ops_engine

    seen = []
    monkeypatch.setattr(ops_engine, "run_ops", lambda settings, phases: seen.append(tuple(phases)) or {})
    P._run_voyager3_phase({"dst_root": "x"})
    assert seen == [("voyager3",)]


def test_a_failed_run_says_why(qtbot):
    screen = QWidget()
    controls = P.Voyager3Controls(screen)
    qtbot.addWidget(controls)
    controls._on_error("Voyager3 found no cluster table at x")
    controls._on_finished(False)
    assert "no cluster table" in controls.status.text() and controls.button.isEnabled()
    controls._on_line("OPS: Progress: 3/4, operation_type: Voyager3")
    assert controls.bar.value() == 75


def test_the_ops_screen_puts_the_control_beside_its_settings_behind_the_alpha_switch(qtbot, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    screen = AppScreen("ops")
    qtbot.addWidget(screen)
    controls = screen._voyager3_controls
    assert controls.button.text() == "Annotate clusters (Voyager3)" and controls._refresh_visibility()
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    screen._refresh_alpha_visibility()
    assert controls.isHidden()
