"""PipelineWorker + stream redirector tests."""
from __future__ import annotations

import pytest

from PySide6.QtCore import QThread

from spacr.qt.bridge import (
    PipelineWorker,
    RunHandle,
    _StreamRedirector,
    apply_worker_budget,
    make_thread,
    registry,
)


def test_stream_redirector_emits_line_by_line():
    received = []
    r = _StreamRedirector(received.append)
    r.write("hello ")
    assert received == []          # no newline yet
    r.write("world\n")
    assert received == ["hello world\n"]
    r.write("multi\nline\nchunk\n")
    assert received[-3:] == ["multi\n", "line\n", "chunk\n"]


def test_stream_redirector_flush_emits_remainder():
    received = []
    r = _StreamRedirector(received.append)
    r.write("no newline yet")
    r.flush()
    assert received == ["no newline yet"]


def test_pipeline_worker_success(qtbot, qt_theme_applied):
    def _fn(settings):
        print("running")
        return {"ok": True}

    worker = PipelineWorker(_fn, {})
    lines = []
    worker.line_ready.connect(lines.append)
    finished = []
    worker.finished.connect(lambda ok: finished.append(ok))
    worker.run()
    assert finished == [True]
    assert any("running" in l for l in lines)


def test_pipeline_worker_captures_exception(qtbot, qt_theme_applied):
    def _fn(settings):
        raise RuntimeError("boom")

    worker = PipelineWorker(_fn, {})
    errors = []
    worker.error.connect(errors.append)
    finished = []
    worker.finished.connect(lambda ok: finished.append(ok))
    worker.run()
    assert finished == [False]
    assert len(errors) == 1
    assert "boom" in errors[0]


@pytest.mark.parametrize(
    ("code", "expected_ok", "expects_error"),
    [(None, True, False), (0, True, False), (1, False, True), ("failed", False, True)],
)
def test_pipeline_worker_preserves_system_exit_status(
        qtbot, qt_theme_applied, code, expected_ok, expects_error):
    """A non-zero CLI-style exit must not become a successful GUI run."""
    def _fn(settings):
        raise SystemExit(code)

    worker = PipelineWorker(_fn, {})
    errors = []
    finished = []
    worker.error.connect(errors.append)
    worker.finished.connect(finished.append)

    worker.run()

    assert finished == [expected_ok]
    assert bool(errors) is expects_error
    if expects_error:
        assert "SystemExit" in errors[0]


def test_make_thread_returns_thread_and_worker():
    thread, worker = make_thread(lambda s: None, {})
    try:
        assert isinstance(thread, QThread)
        assert isinstance(worker, PipelineWorker)
    finally:
        thread.deleteLater()
        worker.deleteLater()


def test_concurrent_runs_share_the_cpu_budget_in_start_order():
    registry().clear()
    first_worker = PipelineWorker(lambda settings: None, {}, worker_count=6)
    first = RunHandle("mask", first_worker, None, parent=registry())
    registry().register(first)
    try:
        second_settings = {"n_jobs": -1}
        second_count = apply_worker_budget(second_settings, total=16)
        assert second_count == 11
        assert second_settings["n_jobs"] == 11  # 16 - 6 + 1

        second_worker = PipelineWorker(
            lambda settings: None, second_settings,
            worker_count=second_count,
        )
        second = RunHandle("measure", second_worker, None, parent=registry())
        registry().register(second)
        try:
            third_settings = {
                "n_jobs": -1,
                "n_workers": 8,
                "infection_xgb_n_jobs": -1,
            }
            assert apply_worker_budget(third_settings, total=16) == 1
            assert third_settings == {
                "n_jobs": 1,
                "n_workers": 1,
                "infection_xgb_n_jobs": 1,
            }
        finally:
            registry().unregister(second)
    finally:
        registry().unregister(first)


def test_pipeline_worker_reemits_only_live_figures(qtbot, tmp_path):
    import matplotlib.pyplot as plt
    plt.close('all')

    live_ids = []
    static_ids = []

    def _fn(settings):
        live, _ = plt.subplots()
        live._spacr_live_update = True
        static, _ = plt.subplots()
        plt.show()
        live.axes[0].plot([0, 1], [0, 1])
        plt.show()
        settings['live'] = live
        settings['static'] = static

    settings = {}
    worker = PipelineWorker(_fn, settings)

    def receive(fig, png_path):
        if getattr(fig, '_spacr_live_update', False):
            live_ids.append(id(fig))
        else:
            static_ids.append(id(fig))
        if png_path:
            from pathlib import Path
            Path(png_path).unlink(missing_ok=True)

    worker.figure_ready.connect(receive)
    worker.run()

    assert live_ids == [id(settings['live']), id(settings['live'])]
    assert static_ids == [id(settings['static'])]
    plt.close('all')
