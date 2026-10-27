"""Capture telemetry without an application, display, recording or GPU."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/tutorials')]
import capture_clip_gate as gate  # noqa: E402
import capture_host_mode as host  # noqa: E402


class Field:
    def __init__(self, frames, *, visible=True, popup=False):
        self.frames_painted, self.visible, self.popup = frames, visible, popup
        self._run_paced = self._hold_paced = False
        self._producer_box = [None]
        self._last_frame = None
        self.repeated_frames = 0

    def isVisible(self):
        return self.visible

    def property(self, name):
        return self.popup if name == 'spacrPopupBackdrop' else None

    def fps(self):
        return 30

    def _rate(self):
        return 30

    def frames_shaded(self):
        return 0

    def is_running(self):
        return self.visible


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv('SPACR_CAPTURE_HOST_SIZE', '2560x1440')
    monkeypatch.setitem(sys.modules, 'spacr.qt.widgets.ambient',
                        SimpleNamespace(AmbientWidget=Field))
    fields = []
    window = SimpleNamespace(findChildren=lambda _kind: fields)
    result = host.HostRuntime(window, SimpleNamespace(pump=None), stage=tmp_path, fps=30)
    result.fields = fields
    return result


def test_navigation_keeps_totals_monotonic_without_historical_counts(runtime):
    first, second = Field(1000), Field(9000, visible=False)
    runtime.fields[:] = [first, second]
    assert runtime.field_sample()['frames_painted'] == 0
    first.frames_painted += 3
    assert runtime.field_sample()['frames_painted'] == 3
    first.visible, second.visible = False, True
    second.frames_painted += 200
    sample = runtime.field_sample()
    assert sample['frames_painted'] == 3
    assert [row['id'] for row in sample['widgets']] == [1, 2]
    second.frames_painted += 3
    assert runtime.field_sample()['frames_painted'] == 6
    first.visible, second.visible = True, False
    first.frames_painted += 100
    assert runtime.field_sample()['frames_painted'] == 6
    first.frames_painted += 3
    assert runtime.field_sample()['frames_painted'] == 9


def test_new_removed_and_popup_fields_do_not_inject_counts(runtime):
    first, popup = Field(40), Field(5000, popup=True)
    runtime.fields[:] = [first, popup]
    assert len(runtime.field_sample()['widgets']) == 1
    first.frames_painted += 3
    assert runtime.field_sample()['frames_painted'] == 3
    runtime.fields.clear()
    assert runtime.field_sample()['frames_painted'] == 3
    first.frames_painted += 900
    replacement = Field(10000)
    runtime.fields[:] = [first, replacement, popup]
    assert runtime.field_sample()['frames_painted'] == 3
    first.frames_painted += 2
    replacement.frames_painted += 3
    popup.frames_painted += 10000
    assert runtime.field_sample()['frames_painted'] == 8


def test_counter_reset_does_not_move_total_backwards(runtime):
    field = Field(100)
    runtime.fields.append(field)
    runtime.field_sample()
    field.frames_painted = 103
    assert runtime.field_sample()['frames_painted'] == 3
    field.frames_painted = 2
    reset = runtime.field_sample()
    assert reset['frames_painted'] == 3 and reset['widgets'][0]['counter_reset']
    assert gate.stable_field_painted_per_second(
        [(0.0, reset['widgets']), (1.0, reset['widgets'])], 1) is None
    field.frames_painted = 5
    assert runtime.field_sample()['frames_painted'] == 6


def test_sampler_records_a_delay_even_when_paint_counters_advanced(runtime, monkeypatch):
    values = iter([{'t': 10.0, 'frames_painted': 0, 'widgets': []},
                   {'t': 10.1, 'frames_painted': 3, 'widgets': []},
                   {'t': 11.316, 'frames_painted': 40, 'widgets': []}])
    monkeypatch.setattr(runtime, 'field_sample', lambda: next(values))
    scene = {'samples': [], 'gui_sampler_gaps': [], 'widget_samples': []}
    for _ in range(3):
        runtime._sample_scene(scene)
    assert scene['gui_sampler_gaps'] == [(10.1, 11.316)]
    assert scene['samples'][-1] == (11.316, 40)


def test_rates_inside_delayed_samples_are_unknown_instead_of_interpolated():
    samples = [(0.0, 0), (0.1, 3), (0.2, 6), (1.416, 43)] + [
        (index / 10, round(30 * index / 10)) for index in range(15, 21)]
    windows, mode = gate.gui_sampler_gap_evidence({'app_paint_samples': samples})
    assert mode == 'inferred_legacy_gui_sampler'
    assert windows == [(0.2, 1.416)]
    rates = gate.app_painted_per_second(samples, 2, gaps=windows)
    assert rates == [None, None]
    freeze, loss, slow, unknown = gate.label_seconds(
        [6, 4], rates, 27.9, sampler_gaps=windows, include_unknown=True)
    assert freeze == [0, 1] and loss == slow == unknown == []


def test_sparse_legacy_samples_keep_their_estimate_without_invented_timer_stalls():
    samples = [(0, 0), (1, 30), (2, 60), (3, 90)]
    assert gate.gui_sampler_gap_evidence({'app_paint_samples': samples}) == (
        [], 'legacy_cadence_unknown')
    assert gate.app_painted_per_second(samples, 3) == [30.0, 30.0, 30.0]


@pytest.mark.parametrize('samples', [[(0, 40), (1, 4)], [(0, 0), (0, 3), (1, 30)]])
def test_regressed_or_ambiguous_legacy_counts_cannot_prove_capture_loss(samples):
    assert gate.app_painted_per_second(samples, 1) is None
    assert gate.label_seconds([6], None, 27.9, include_unknown=True) == ([], [], [], [0])


def test_unknown_second_cannot_be_classified_as_capture_loss():
    assert gate.label_seconds([6, 4], [None, 30], 27.9, include_unknown=True) == (
        [], [1], [], [0])


def test_multiple_field_counts_do_not_hide_one_slow_field():
    samples = [(float(t), [{'id': 1, 'visible': True, 'observed_frames_painted': 60 * t},
                          {'id': 2, 'visible': True, 'observed_frames_painted': 10 * t}])
               for t in range(4)]
    painted = gate.stable_field_painted_per_second(samples, 3)
    assert painted == [10.0, 10.0, 10.0]
    assert gate.label_seconds([10, 10, 10], painted, 27.9) == ([], [], [0, 1, 2])


def test_changed_field_identity_has_no_stable_field_rate():
    samples = [(0.0, [{'id': 1, 'visible': True, 'observed_frames_painted': 0}]),
               (1.0, [{'id': 2, 'visible': True, 'observed_frames_painted': 1000}])]
    assert gate.stable_field_painted_per_second(samples, 1) is None


def fake_measurements(monkeypatch, *, seconds=12, low_second=9, low_frames=6):
    count = seconds * 30
    changed = [True] * count
    changed[low_second * 30 + low_frames:(low_second + 1) * 30] = [False] * (30 - low_frames)
    track = [{'frame': i, 'found': True, 'x': 100, 'y': 100, 'shape': 'cursor'}
             for i in range(count)]
    monkeypatch.setattr(gate, 'frame_rate', lambda _path: 30.0)
    monkeypatch.setattr(gate, 'size', lambda _path: (640, 360))
    monkeypatch.setattr(gate, 'duration', lambda _path: float(seconds))
    monkeypatch.setattr(gate, 'cursor_templates', lambda *_a: None)
    monkeypatch.setattr(gate, 'track_cursor', lambda *_a: track)
    monkeypatch.setattr(gate, 'picture_changes', lambda *_a, **_k: (changed, [], []))
    monkeypatch.setattr(gate, 'exact_distinct', lambda *_a: ('digest', [True] * count))


@pytest.mark.parametrize('end', [9.506, 9.389])
def test_resumed_repaint_counters_do_not_make_opening_stalls_capture_loss(monkeypatch, end):
    fake_measurements(monkeypatch)
    times = [index / 10 for index in range(84)] + [end, end + 0.1, end + 0.2, 12.0]
    samples = [[time, round(30 * time)] for time in times]
    result = gate.evaluate(Path('unused-native.mp4'), receipt={'app_paint_samples': samples})
    assert not result['passed']
    assert 9 in result['app_freeze_seconds']
    assert 9 not in result['capture_loss_seconds']
    assert result['app_gui_sampler_gaps_s']
    assert not any(reason.startswith('choppy (capture loss)') for reason in result['reasons'])
    assert result['limits']['min_fps'] == 27.9


def test_low_unknown_paint_evidence_remains_unlabelled_in_the_full_gate(monkeypatch):
    fake_measurements(monkeypatch, seconds=3, low_second=1, low_frames=0)
    result = gate.evaluate(Path('unused-native.mp4'), receipt={
        'app_paint_counter_semantics': 'observed_per_widget_deltas',
        'app_paint_samples': [[0, 0], [1, 60], [2, 120], [3, 180]],
        'app_field_widget_samples': []})
    assert result['capture_loss_seconds'] == []
    assert result['app_unknown_seconds'] == [1]
    assert result['reasons'][0].startswith('choppy (unlabelled)')


def test_confirmed_sampler_stall_fails_even_when_clip_average_passes(monkeypatch):
    fake_measurements(monkeypatch, seconds=20)
    times = [index / 10 for index in range(84)] + [9.506] + [index / 10 for index in range(96, 201)]
    result = gate.evaluate(Path('unused-native.mp4'), receipt={
        'app_paint_samples': [[time, round(30 * time)] for time in times]})
    assert result['distinct_fps'] >= result['limits']['min_fps']
    assert not result['passed']
    assert 9 in result['app_freeze_seconds']
    assert any(reason.startswith('app freeze:') for reason in result['reasons'])


def test_a_field_paint_freeze_cannot_pass_on_the_whole_clip_average(monkeypatch):
    fake_measurements(monkeypatch, seconds=20)
    times = [index / 10 for index in range(201)]
    counts = [round(30 * time - 28 * min(1, max(0, time - 9))) for time in times]
    result = gate.evaluate(Path('unused-native.mp4'), receipt={
        'app_paint_counter_semantics': 'observed_per_widget_deltas',
        'app_paint_samples': list(zip(times, counts)),
        'app_field_widget_samples': [(time, [{'id': 1, 'visible': True,
            'observed_frames_painted': count}]) for time, count in zip(times, counts)]})
    assert result['distinct_fps'] >= result['limits']['min_fps']
    assert result['app_gui_sampler_gaps_s'] == []
    assert result['app_freeze_seconds'] == [9]
    assert not result['passed']
    assert result['reasons'][0].startswith('app freeze:')
