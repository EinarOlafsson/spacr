"""Preferences tutorial scenes follow real categories and never save settings."""
import importlib.util
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog, QTabWidget, QWidget

from spacr.qt import preferences
from spacr.qt.widgets.section import Section


def test_tutorial_captures_folded_categories_and_actual_expanded_controls(qtbot, monkeypatch, tmp_path):
    store = QSettings(str(tmp_path / 'recording.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: store)
    preferences.set_theme('dark')
    # Initialize the normal defaults/migration before checking interaction writes.
    preferences.get_performance_level()
    preferences._migrate_ambient_motion()
    before = {key: store.value(key) for key in store.allKeys()}
    path = Path(__file__).resolve().parents[2] / 'tools/tutorials/capture_home.py'
    module_spec = importlib.util.spec_from_file_location('capture_home_601', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(1600, 1400)
    host.show()
    captured = []

    def capture(name):
        dialog = next(w for w in QApplication.topLevelWidgets()
                      if isinstance(w, QDialog) and w.isVisible())
        tabs = dialog.findChild(QTabWidget, 'PreferencesTabs')
        titles = [tabs.tabText(i) for i in range(tabs.count())]
        assert titles == ['General', 'Appearance', 'Performance', 'Modules', 'Figures', 'Logging', 'AI']
        sections = {s.title(): s for s in tabs.currentWidget().findChildren(Section)}
        if name == '09_performance':
            assert tabs.tabText(tabs.currentIndex()) == 'Performance'
        else:
            assert tabs.tabText(tabs.currentIndex()) == 'Appearance'
            expanded = {title for title, section in sections.items() if section.is_expanded()}
            assert expanded == {'10_appearance_categories': set(), '11_appearance_theme': {'THEME'},
                                '12_appearance_animation': {'ANIMATION'}}[name]
        captured.append(name)

    module.record_performance(host, capture, lambda *_: qtbot.wait(30))
    assert captured == ['09_performance', '10_appearance_categories',
                        '11_appearance_theme', '12_appearance_animation']
    assert {key: store.value(key) for key in store.allKeys()} == before
