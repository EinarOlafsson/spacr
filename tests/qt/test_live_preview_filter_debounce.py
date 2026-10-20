"""Typing a filter bound re-filters once, off the GUI thread.

Lesson 07's capture typed ``510`` into Mask's Live minimum-area box and the
app froze for a second: every keystroke re-filtered the cached masks, redrew
both canvases and propagated ``object_filters`` on the GUI thread, about half
a second each on a 2000 x 2000 field. Now a keystroke only restarts a timer;
the filtering and the overlay run on a worker once typing pauses, a stale
result is dropped, and the propagation waits for the same pause.
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from spacr.qt.widgets import live_preview as LP


def _field(size=600, step=12):
    """A grid of square labels of growing area, and an image to draw on."""
    mask = np.zeros((size, size), np.int32)
    label = 0
    for y in range(0, size - step, step):
        for x in range(0, size - step, step):
            label += 1
            side = 2 + label % (step - 3)
            mask[y:y + side, x:x + side] = label
    rng = np.random.default_rng(7)
    image = rng.integers(0, 4000, (size, size, 2)).astype(np.uint16)
    return mask, image


def _panel(qtbot, threaded, size=600):
    panel = LP.LivePreviewPanel(threaded=threaded)
    qtbot.addWidget(panel)
    mask, image = _field(size)
    panel._image = image
    panel._raw_masks = {"cell": mask}
    panel._recompute_masks(snapshot=True)
    return panel


@pytest.fixture
def inline(qtbot):
    panel = _panel(qtbot, threaded=False, size=120)
    yield panel
    panel.shutdown()


@pytest.fixture
def threaded(qtbot):
    panel = _panel(qtbot, threaded=True)
    yield panel
    panel.shutdown()


def _min_area(panel):
    return panel._compartment_widgets["cell"]["min_area"]


def _picture(panel):
    return panel._mask_view.picture().toImage()


def _choose_view(panel, mode):
    combo = panel._view_mode
    for index in range(combo.count()):
        combo.setCurrentIndex(index)
        if panel._view_mode_choice() == mode:
            return
    raise AssertionError(f"no view mode {mode}")


def _count_filter_runs(monkeypatch):
    runs = []
    real = LP._filter_preview_masks

    def counted(*args, **kwargs):
        runs.append(args[1].get("object_filters"))
        return real(*args, **kwargs)

    monkeypatch.setattr(LP, "_filter_preview_masks", counted)
    return runs


def _settle(qtbot, panel):
    qtbot.waitUntil(lambda: not panel._filter_timer.isActive()
                    and not panel._filter_jobs.is_busy(), timeout=20000)


def test_a_keystroke_costs_a_fraction_of_one_filter_pass(qtbot, threaded):
    """Self-calibrated: both sides are timed on this machine, now."""
    field = _min_area(threaded)
    sync = []
    for value in (5, 50, 500):
        field.blockSignals(True)
        field.setValue(value)
        field.blockSignals(False)
        start = time.perf_counter()
        threaded._recompute_masks()
        sync.append(time.perf_counter() - start)
    field.setValue(0)
    _settle(qtbot, threaded)
    field.setFocus()
    edit = field.lineEdit()
    edit.selectAll()
    keys = []
    for char in "510":
        start = time.perf_counter()
        QTest.keyClick(edit, char)
        keys.append(time.perf_counter() - start)
    assert field.value() == 510
    assert max(keys) < statistics.median(sync) / 4
    _settle(qtbot, threaded)


def test_a_burst_of_keys_filters_once(qtbot, inline, monkeypatch):
    runs = _count_filter_runs(monkeypatch)
    inline._filter_debounce_ms = 60
    field = _min_area(inline)
    for value in (5, 51, 510):
        field.setValue(value)
    assert runs == []
    qtbot.waitUntil(lambda: len(runs) == 1, timeout=5000)
    qtbot.wait(200)
    assert len(runs) == 1
    assert "510" in repr(runs[0]) or 510 in repr(runs[0])


def test_the_debounced_preview_is_the_preview_of_today(qtbot, threaded):
    field = _min_area(threaded)
    for value in (5, 51, 510):
        field.setValue(value)
    _settle(qtbot, threaded)
    debounced = {k: v.copy() for k, v in threaded._masks.items()}
    picture = _picture(threaded)
    status = threaded._status.text()
    assert int(debounced["cell"].max()) < int(
        threaded._raw_masks["cell"].max())
    threaded._recompute_masks()
    for obj, mask in threaded._masks.items():
        np.testing.assert_array_equal(mask, debounced[obj])
    assert _picture(threaded) == picture
    assert threaded._status.text() == status


def test_a_stale_result_never_overwrites_a_newer_one(inline, monkeypatch):
    held = []
    monkeypatch.setattr(inline._filter_jobs, "submit",
                        lambda fn, on_done: held.append((fn, on_done)))
    field = _min_area(inline)
    field.blockSignals(True)
    field.setValue(4)
    inline._start_filter_job()
    field.setValue(80)
    inline._start_filter_job()
    field.blockSignals(False)
    (old_fn, old_done), (new_fn, new_done) = held
    old = old_fn()
    new_done(new_fn())
    newest = inline._masks["cell"].copy()
    picture = _picture(inline)
    old_done(old)
    np.testing.assert_array_equal(inline._masks["cell"], newest)
    assert _picture(inline) == picture
    assert int(old["masks"]["cell"].max()) > int(newest.max())


def test_a_direct_recompute_supersedes_a_job_in_flight(inline, monkeypatch):
    held = []
    monkeypatch.setattr(inline._filter_jobs, "submit",
                        lambda fn, on_done: held.append((fn, on_done)))
    field = _min_area(inline)
    field.blockSignals(True)
    field.setValue(4)
    inline._start_filter_job()
    field.setValue(80)
    field.blockSignals(False)
    inline._recompute_masks()
    newest = inline._masks["cell"].copy()
    fn, done = held[0]
    done(fn())
    np.testing.assert_array_equal(inline._masks["cell"], newest)


def test_a_view_redrawn_meanwhile_is_drawn_again(inline, monkeypatch):
    held = []
    monkeypatch.setattr(inline._filter_jobs, "submit",
                        lambda fn, on_done: held.append((fn, on_done)))
    _min_area(inline).setValue(80)
    fn, done = held[0]
    result = fn()
    assert result["overlay"] is not None
    inline._outline_thickness.setValue(3)
    redraws = []
    real = inline._refresh_canvases
    monkeypatch.setattr(inline, "_refresh_canvases",
                        lambda: redraws.append(1) or real())
    done(result)
    assert redraws == [1]
    expected = LP.overlay_masks(masks=result["masks"],
                                **inline._overlay_arguments())
    assert _picture(inline) == LP.numpy_to_qpixmap(expected).toImage()


def test_other_views_are_drawn_on_the_gui_thread(inline, monkeypatch):
    _choose_view(inline, "Masks")
    assert inline._overlay_arguments() is None
    runs = _count_filter_runs(monkeypatch)
    _min_area(inline).setValue(80)
    assert len(runs) == 1
    assert _picture(inline) == LP.numpy_to_qpixmap(
        inline._label_rgb()).toImage()


def test_a_toggle_filters_at_once_without_waiting(qtbot, inline, monkeypatch):
    runs = _count_filter_runs(monkeypatch)
    inline._filter_debounce_ms = 10_000
    inline._compartment_widgets["cell"]["remove_border_objects"].setChecked(
        True)
    assert len(runs) == 1
    assert not inline._filter_timer.isActive()


def test_nothing_is_scheduled_without_cached_masks(qtbot, monkeypatch):
    panel = LP.LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    runs = _count_filter_runs(monkeypatch)
    panel._filter_debounce_ms = 30
    _min_area(panel).setValue(80)
    panel._start_filter_job()
    assert runs == [] and not panel._filter_timer.isActive()
    panel.shutdown()


def test_a_job_without_an_image_fails_visibly(inline):
    inline._image = None
    inline._start_filter_job()
    assert inline._masks == {}
    assert inline._status.text()
    assert inline._mask_view.picture() is None or inline._mask_view.picture(
    ).isNull()


def test_unreadable_inputs_fail_visibly_on_the_gui_thread(inline,
                                                          monkeypatch):
    def broken():
        raise ValueError("unreadable bound")

    monkeypatch.setattr(inline, "_compartment_settings", broken)
    sent = []
    inline.preview_ready.connect(sent.append)
    inline._start_filter_job()
    assert inline._masks == {} and sent == [None]
    assert "unreadable bound" in inline._status.text()


def test_a_filter_that_raises_fails_visibly(inline, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("filter broke")

    monkeypatch.setattr(LP, "_apply_size_filter", broken)
    assert "error" in LP._filter_preview_masks(
        inline._raw_masks, {}, {}, None)
    sent = []
    inline.preview_ready.connect(sent.append)
    inline._start_filter_job()
    assert inline._masks == {} and sent == [None]
    assert "filter broke" in inline._status.text()


def test_the_pure_filter_skips_the_overlay_when_not_asked(inline):
    result = LP._filter_preview_masks(inline._raw_masks, {}, {}, None)
    assert result["overlay"] is None
    np.testing.assert_array_equal(result["masks"]["cell"],
                                  inline._raw_masks["cell"])


def test_propagation_waits_for_the_pause_and_flushes(qtbot, inline,
                                                     monkeypatch):
    sent = []
    monkeypatch.setattr(inline, "propagate_settings",
                        lambda: sent.append(1))
    inline.schedule_propagation()
    assert sent == [1]
    inline._filter_debounce_ms = 50
    for _ in range(3):
        inline.schedule_propagation()
    assert sent == [1]
    qtbot.waitUntil(lambda: len(sent) == 2, timeout=5000)
    qtbot.wait(120)
    assert sent == [1, 1]
    inline.flush_propagation()
    assert sent == [1, 1]
    inline.schedule_propagation()
    inline.flush_propagation()
    assert sent == [1, 1, 1] and not inline._propagate_timer.isActive()


def test_propagation_waits_for_a_running_filter_job(inline, monkeypatch):
    sent = []
    monkeypatch.setattr(inline, "propagate_settings",
                        lambda: sent.append(1))
    monkeypatch.setattr(inline._filter_jobs, "is_busy", lambda: True)
    inline._propagate_now()
    assert sent == [] and inline._propagation_waiting
    inline._after_filter_job(True)
    assert sent == []
    monkeypatch.setattr(inline._filter_jobs, "is_busy", lambda: False)
    inline._after_filter_job(True)
    assert sent == [1] and not inline._propagation_waiting
    inline._after_filter_job(True)
    assert sent == [1]
    inline._propagate_now()
    assert sent == [1, 1]
    inline._propagation_waiting = True
    inline.flush_propagation()
    assert sent == [1, 1, 1] and not inline._propagation_waiting


def test_a_threaded_burst_propagates_once_after_the_filter(qtbot, threaded):
    sent = []
    threaded.set_propagate_callback(sent.append)
    threaded.open_live_settings()
    dialog = threaded._live_settings_dialog
    try:
        dialog._propagate_btn.setChecked(True)
        assert len(sent) == 1
        field = _min_area(threaded)
        for value in (5, 51, 510):
            field.setValue(value)
        assert len(sent) == 1
        _settle(qtbot, threaded)
        qtbot.waitUntil(lambda: len(sent) == 2, timeout=5000)
        assert LP._bound_from_filters(sent[-1], "cell", "min_area") == 510
        field.setValue(520)
        assert threaded._propagate_timer.isActive()
        field.editingFinished.emit()
        assert len(sent) == 3
        assert LP._bound_from_filters(sent[-1], "cell", "min_area") == 520
        field.setValue(530)
        dialog._propagate_btn.setChecked(False)
        assert len(sent) == 4
        field.setValue(540)
        _settle(qtbot, threaded)
        qtbot.wait(threaded._filter_debounce_ms + 100)
        assert len(sent) == 4
    finally:
        dialog.close()


def test_a_run_sends_the_pending_propagation_first(inline, monkeypatch):
    sent = []
    monkeypatch.setattr(inline, "propagate_settings",
                        lambda: sent.append(1))
    monkeypatch.setattr(inline, "begin_preview", lambda: False)
    inline._filter_debounce_ms = 10_000
    inline.schedule_propagation()
    inline.run_preview()
    assert sent == [1]


def test_shutdown_drops_pending_work_and_sends_the_push(qtbot, monkeypatch):
    panel = _panel(qtbot, threaded=False, size=60)
    sent = []
    monkeypatch.setattr(panel, "propagate_settings", lambda: sent.append(1))
    panel._filter_debounce_ms = 10_000
    _min_area(panel).setValue(30)
    panel.schedule_propagation()
    token = panel._filter_token
    panel.shutdown()
    assert not panel._filter_timer.isActive()
    assert panel._filter_token == token + 1
    assert sent == [1]


def test_typed_text_in_a_channel_box_waits_too(inline, monkeypatch):
    runs = _count_filter_runs(monkeypatch)
    inline._filter_debounce_ms = 10_000
    inline._cell_channel.setValue(1)
    assert runs == [] and inline._filter_timer.isActive()
    inline._filter_timer.stop()


def test_the_typed_key_reaches_the_line_edit_path(qtbot, inline, monkeypatch):
    runs = _count_filter_runs(monkeypatch)
    inline._filter_debounce_ms = 10_000
    edit = _min_area(inline).lineEdit()
    edit.selectAll()
    QTest.keyClick(edit, Qt.Key_7)
    assert runs == [] and inline._filter_timer.isActive()
    inline._filter_timer.timeout.emit()
    assert len(runs) == 1


def test_a_job_result_keeps_the_provenance_clauses(inline, monkeypatch):
    held = []
    monkeypatch.setattr(inline._filter_jobs, "submit",
                        lambda fn, on_done: held.append((fn, on_done)))
    inline._processing_provenance = {
        "processing": {"operation": "deconvolve"},
        "enhancement": {"order": [], "clahe": {}},
    }
    _min_area(inline).setValue(80)
    fn, done = held[0]
    done(fn())
    text = inline._status.text()
    assert text.startswith("Found cell=")
    assert "deconvolve" in text and "clahe" in text


def test_an_edit_signal_that_refuses_wiring_is_skipped(inline):
    class _Refuses:
        def connect(self, *_args):
            raise RuntimeError("wrapped C/C++ object has been deleted")

        disconnect = connect

    class _Source:
        editingFinished = _Refuses()

    sent = []
    inline.set_propagate_callback(sent.append)
    inline.open_live_settings()
    dialog = inline._live_settings_dialog
    try:
        dialog._propagate_sources.append(_Source())
        dialog._propagate_btn.setChecked(True)
        dialog._propagate_btn.setChecked(False)
        assert len(sent) == 1
    finally:
        dialog.close()


def test_a_direct_recompute_without_masks_still_cancels_the_wait(qtbot):
    panel = LP.LivePreviewPanel(threaded=True)
    qtbot.addWidget(panel)
    panel._filter_timer.start(10_000)
    token = panel._filter_token
    panel._recompute_masks()
    assert not panel._filter_timer.isActive()
    assert panel._filter_token == token + 1
    panel.shutdown()


def test_collect_writes_a_held_back_value_in_first(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    model = SettingsWidgets("mask")
    calls = []

    def broken():
        calls.append("broken")
        raise RuntimeError("the panel has gone")

    model.add_collect_hook(broken)
    model.add_collect_hook(lambda: calls.append("flushed"))
    assert isinstance(model.collect(), dict)
    assert calls == ["broken", "flushed"]


def test_the_mask_screen_flushes_its_preview_before_it_collects(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    panel = screen._live_preview
    panel._filter_debounce_ms = 10_000
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    try:
        dialog._propagate_btn.setChecked(True)
        panel._compartment_widgets["cell"]["min_area"].setValue(510)
        assert panel._propagate_timer.isActive()
        collected = screen._settings_model.collect()
        assert not panel._propagate_timer.isActive()
        assert LP._bound_from_filters(collected, "cell", "min_area") == 510
    finally:
        dialog.close()
