"""Capture cadence preserves real held rows and reports caps separately from activity."""
from pathlib import Path
import sys
from types import SimpleNamespace
import time

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
        sampled_after = time.monotonic()
        normal = runtime.field_sample()
        state = normal['widgets'][0]
        assert sampled_after <= state['sample_begin_monotonic_s'] <= state['sample_end_monotonic_s'] <= time.monotonic()
        assert state['producer_present'] and state['producer_identity'] == 1
        assert state['frames_shaded'] > 0 and state['repeated_frames'] >= 0
        assert state['last_frame_cache_key'] == str(backdrop._last_frame.cacheKey())
        assert state['frames_painted'] == backdrop.frames_painted
        assert isinstance(state['kernel_state']['packed_ready'], bool)
        assert isinstance(state['kernel_state']['subpixel_ready'], bool)
        assert isinstance(state['kernel_state']['compiler_threads'], list)
        assert normal['fps_cap'] == [60]
        assert normal['widgets'][0]['configured_fps'] == 60
        assert normal['widgets'][0]['effective_rate'] == 20
        assert normal['widgets'][0]['hold_paced'] is True
        assert normal['widgets'][0]['_hold_paced'] is True
        assert normal['pacing'] == [{'id': state['id'], 'configured_fps': 60.0,
                                     'effective_rate': 20.0, '_hold_paced': True}]
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
            assert sample['widgets'][0]['_hold_paced'] is True
            assert sample['pacing'][0]['effective_rate'] == 30.0
            assert sample['widgets'][0]['run_paced'] is False
            assert sample['widgets'][0]['paused'] is False
            assert sample['frames_painted'] == normal['frames_painted']
            assert sample['widgets'][0]['producer_identity'] == state['producer_identity']
            assert sample['widgets'][0]['last_frame_cache_key'] == state['last_frame_cache_key']
            backdrop.stop()
            stopped = runtime.field_sample()['widgets'][0]
            assert stopped['paused'] is True
            assert stopped['producer_present'] is False and stopped['producer_identity'] is None
            assert stopped['frames_shaded'] == 0
        assert ambient._HOLD_FPS == 20 and backdrop._rate() == 20
        with host._capture_pacing(60):
            with host._capture_pacing(30):
                assert ambient._HOLD_FPS == 60
        assert ambient._HOLD_FPS == 20
    finally:
        backdrop.stop()
        window.close()
    with monkeypatch.context() as clock_patch:
        _assert_saved_input_clock_keeps_absolute_and_relative_epochs(clock_patch, tmp_path)


def _assert_saved_input_clock_keeps_absolute_and_relative_epochs(monkeypatch, tmp_path):
    import json

    event = {'t': 12.25, 'kind': 'move'}
    runtime = host.HostRuntime(SimpleNamespace(findChildren=lambda _kind: []),
                               SimpleNamespace(pump=None), stage=tmp_path, fps=30)
    runtime.adapter = SimpleNamespace(bypasses=[], call=lambda name, *args:
                                      [event] if name == 'trace_since' else {})
    sampler = SimpleNamespace(stop=lambda: None, setParent=lambda _parent: None,
                              deleteLater=lambda: None,
                              timeout=SimpleNamespace(disconnect=lambda: None))
    runtime.current = {
        'begin': 12.0, 'seconds': 0, 'sampler': sampler, 'mark': {'t': 11.0, 'index': 7},
        'first_bypass': 0, 'captures': tmp_path, 'name': 'clock', 'performed': [],
        'load_before': (0, 0, 0), 'field_before': {'t': 12.0, 'frames_painted': 0, 'fps_cap': [], 'pacing': []},
        'samples': [], 'widget_samples': [], 'gui_sampler_gaps': [],
        'appearance': {}, 'hidden_backdrops': [], 'buttons': [], 'dialogs': [],
        'native': tmp_path / 'clock.native.mp4',
    }
    monkeypatch.setattr(host, 'time', SimpleNamespace(monotonic=lambda: 12.5))
    runtime.end_scene(at_least=0)
    facts = json.loads((tmp_path / 'clock.host.json').read_text())
    assert facts['capture_clock'] == {
        'scene_begin_monotonic_s': 12.0, 'input_trace_mark_monotonic_s': 11.0,
        'input_trace_mark_index': 7, 'native_pts_alignment': 'unmeasured',
    }
    saved = json.loads((tmp_path / 'clock.pointer.json').read_text())
    assert saved == [{'t': .25, 'kind': 'move', 'monotonic_s': 12.25}]
    assert event == {'t': 12.25, 'kind': 'move'}
    assert facts['app_paint_samples'] == [[.5, 0]]
    assert facts['app_field']['pacing_before'] == facts['app_field']['pacing_after'] == []
