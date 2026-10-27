"""Capture cadence preserves real held rows and reports caps separately from activity."""
from pathlib import Path
import sys
from types import SimpleNamespace

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/tutorials')]

import capture_host_mode as host
from spacr.qt import preferences
from spacr.qt.widgets import ambient


def test_capture_pacing_preserves_held_rows_and_reports_actual_caps(qtbot, monkeypatch, tmp_path):
    store = QSettings(str(tmp_path / 'private.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: store)
    monkeypatch.setenv('SPACR_BATTERY_PAUSE', '0')
    monkeypatch.setenv('SPACR_CAPTURE_HOST_SIZE', '2560x1440')
    monkeypatch.setattr(ambient, '_RESOURCE_POLICY', None)
    assert ambient._HOLD_FPS == 20
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(640, 420)
    backdrop = ambient.AmbientWidget(window, theme='data_art_impulse_lens', fps=60, seed=7)
    backdrop.follow_parent()
    card = QFrame(window)
    card.setObjectName('captureHeldCard')
    card.setAttribute(Qt.WA_StyledBackground, True)
    card.setStyleSheet('#captureHeldCard { background: rgba(28,32,44,150); }')
    card.setGeometry(40, 40, 360, 300)
    layout = QVBoxLayout(card)
    layout.addWidget(QLabel('Native settings row', card))
    window.show()
    qtbot.waitExposed(window)
    try:
        qtbot.waitUntil(lambda: backdrop._held_rows.frame is not None, timeout=10000)
        qtbot.waitUntil(lambda: backdrop._rate() == 20, timeout=2000)
        held = backdrop._held_rows.frame
        region = backdrop._held_rows.region
        assert region.contains(card.geometry().center())
        assert not backdrop._held_rows.live.isEmpty()
        runtime = host.HostRuntime(window, SimpleNamespace(pump=None), stage=tmp_path, fps=30)
        normal = runtime.field_sample()
        assert normal['fps_cap'] == [60]
        assert normal['widgets'][0]['configured_fps'] == 60
        assert normal['widgets'][0]['effective_rate'] == 20
        assert normal['widgets'][0]['hold_paced'] is True
        assert normal['widgets'][0]['paused'] is False
        with host._capture_pacing(host.recording_fps(window)) as policy:
            assert policy == {'normal_hold_fps': 20, 'capture_hold_fps': 30}
            backdrop._apply_rate()
            assert backdrop._rate() == 30 and backdrop._timer.interval() == 33
            assert backdrop._held_rows.frame is held and backdrop._held_rows.region == region
            sample = runtime.field_sample()
            assert sample['fps_cap'] == [60]
            assert sample['widgets'][0]['configured_fps'] == 60
            assert sample['widgets'][0]['effective_rate'] == 30
            assert sample['widgets'][0]['hold_paced'] is True
            assert sample['widgets'][0]['run_paced'] is False
            assert sample['widgets'][0]['paused'] is False
            assert sample['frames_painted'] == normal['frames_painted']
            backdrop.stop()
            assert runtime.field_sample()['widgets'][0]['paused'] is True
        assert ambient._HOLD_FPS == 20 and backdrop._rate() == 20
        with host._capture_pacing(60):
            with host._capture_pacing(30):
                assert ambient._HOLD_FPS == 60
        assert ambient._HOLD_FPS == 20
    finally:
        backdrop.stop()
        window.close()
