"""Use Regression's real public result tabs, never its hidden detail queue."""
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[3]),
                str(Path(__file__).resolve().parents[1])]

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import capture_ui
from capture_refresh import record_default_batch_figure
from capture_settings import record_results
from spacr.qt.screens.app_screen import AppScreen


def test_current_regression_uses_visible_results_tabs_without_hidden_queue_click(
        tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    screen = AppScreen('regression')
    screen.resize(2200, 1250)
    screen.show()
    QTest.qWait(100)
    scenes, writes, selected = [], {}, []
    settle = lambda seconds=0: QTest.qWait(5)
    try:
        # Deliver a small synthetic result through the application's actual
        # pipeline-result slot, which exposes the result workspace after a run.
        # This is UI fixture data, not a model run or scientific acceptance.
        import pandas as pd
        screen._on_pipeline_result({'results': pd.DataFrame({
            'feature': ['fraction:grna[g1_1]', 'fraction:grna[g2_1]'],
            'coefficient': [.2, -.3], 'p_value': [.05, .1]})})
        from matplotlib.figure import Figure
        figure = Figure()
        figure.add_subplot().plot([0, 1], [0, 1])
        screen._on_figure_ready(figure)
        QTest.qWait(50)
        tabs = screen._results_tabs
        assert tabs is not None and tabs.tabBar().isVisible()
        assert screen._figures_stack.currentIndex() == 0
        assert screen._figure_queue.count() == 1
        assert not screen._figure_queue._list.isVisible()
        before = screen._settings_model.collect()
        tabs.currentChanged.connect(lambda i: selected.append(tabs.tabText(i)))

        def hidden_queue_click(*args):
            raise AssertionError('The hidden Regression detail queue must not be clicked')

        monkeypatch.setattr(capture_ui, 'show_figure', hidden_queue_click)
        record_default_batch_figure('regression', screen._figure_queue, scenes.append, settle)
        record_results(screen, tmp_path, scenes.append, settle,
                       lambda path, value: writes.update({path.name: value}))
        assert '24_batch_figure' not in scenes
        assert scenes == ['25_result_runs', '25_result_results',
                          '25_result_measurements', '25_result_cells',
                          '26_detail_p_values', '26_detail_q_q',
                          '26_detail_coefficients']
        assert selected[:4] == ['Runs', 'Results', 'Measurements', 'Cells']
        assert tabs.tabText(tabs.currentIndex()) == 'Results'
        assert screen._results_panel.tabs.currentWidget() is screen._results_panel.table
        assert screen._settings_model.collect() == before
        assert writes['results_tour.json']['settings_unchanged']
        assert not hasattr(screen, '_run_started_at')
        deadline = time.monotonic() + 10
        while screen._jobs.is_busy() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert not screen._jobs.is_busy()
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_other_modules_keep_the_existing_default_figure_scene(monkeypatch):
    from types import SimpleNamespace

    queue = SimpleNamespace(count=lambda: 3)
    actions = []
    monkeypatch.setattr(capture_ui, 'show_figure',
                        lambda capture, actual, index: actions.append((actual, index)))
    record_default_batch_figure('measure', queue, actions.append,
                                lambda: actions.append('settled'))
    assert actions == [(queue, 2), 'settled', '24_batch_figure']
