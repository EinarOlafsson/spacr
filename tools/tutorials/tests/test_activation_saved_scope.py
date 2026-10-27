"""The external viewer never launders failed outputs or certifies GUI fixes."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from activation_evidence import check_native_runs, check_saved_runs


def good_runs():
    runs = []
    for method, pixels in [('saliency_channel', 49152), ('saliency_image', 16384)]:
        runs.append(dict(method=method, outcome=dict(finished=True, ok=True, errors=[]),
            gui_figure_count=0, figures_card_visible=False,
            settings_errors=['[settings] ERROR [src]: src is missing from the settings.'],
            maps=[dict(pixels=pixels, max_absolute_error=0) for _ in range(4)],
            independent_saved_pixels_checked=pixels * 4,
            external_viewer=dict(accepted=True, saved_files_unchanged=True,
                application_figures_fixed=False, saved_plots=[dict(actual_desktop_capture=True)])))
    return runs


def test_saved_scope_is_not_a_gui_completion_claim():
    runs = good_runs()
    check_saved_runs(runs)
    with pytest.raises(ValueError, match='native figures'): check_native_runs(runs)
    for run in runs:
        run.update(gui_figure_count=1, figures_card_visible=True, settings_errors=[])
    check_native_runs(runs)
    check_saved_runs(runs)


@pytest.mark.parametrize('kind', ['missing_method', 'duplicate_method', 'failed', 'unrelated_error',
    'missing_map', 'missing_pixels', 'bad_pixels', 'wrong_pixel_count', 'unaccepted_viewer',
    'changed_file', 'false_GUI_claim', 'no_grid', 'no_capture'])
def test_each_false_success_is_refused_after_positive(kind):
    runs = good_runs(); check_saved_runs(deepcopy(runs)); run = runs[0]
    if kind == 'missing_method': runs.pop()
    if kind == 'duplicate_method': runs[1]['method'] = run['method']
    if kind == 'failed': run['outcome']['ok'] = False
    if kind == 'unrelated_error': run['settings_errors'].append('another error')
    if kind == 'missing_map': run['maps'].pop()
    if kind == 'missing_pixels': run['independent_saved_pixels_checked'] = 0
    if kind == 'bad_pixels': run['maps'][0]['max_absolute_error'] = 2
    if kind == 'wrong_pixel_count': run['maps'][0]['pixels'] = 1
    if kind == 'unaccepted_viewer': run['external_viewer']['accepted'] = False
    if kind == 'changed_file': run['external_viewer']['saved_files_unchanged'] = False
    if kind == 'false_GUI_claim': run['external_viewer']['application_figures_fixed'] = True
    if kind == 'no_grid': run['external_viewer']['saved_plots'] = []
    if kind == 'no_capture': run['external_viewer']['saved_plots'][0]['actual_desktop_capture'] = False
    with pytest.raises(ValueError): check_saved_runs(runs)


@pytest.mark.parametrize('allow_drag', [True, False])
def test_result_console_drag_requires_real_readable_geometry(allow_drag, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QSplitter, QWidget
    from PySide6.QtTest import QTest
    from capture_activation import _balance_result_panes

    app = QApplication.instance() or QApplication([])
    window = QSplitter(Qt.Horizontal)
    settings, console = QWidget(), QWidget()
    window.addWidget(settings); window.addWidget(console)
    window.resize(1600, 800); window.show()
    window.setSizes([1500, 100]); app.processEvents()
    screen = SimpleNamespace(_body_splitter=window, _settings_scroll=settings, _console=console)
    if not allow_drag:
        monkeypatch.setattr(QTest, 'mouseMove', lambda *args, **kwargs: None)
    try:
        if allow_drag:
            _balance_result_panes(screen, window, lambda _: app.processEvents())
            assert console.width() >= window.width() * .3
            assert settings.width() >= window.width() * .25
        else:
            with pytest.raises(ValueError, match='both be readable'):
                _balance_result_panes(screen, window, lambda _: app.processEvents())
    finally:
        window.close()


def test_readable_result_panes_do_not_repeat_the_native_drag(monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QSplitter, QWidget
    from PySide6.QtTest import QTest
    from capture_activation import _balance_result_panes

    app = QApplication.instance() or QApplication([])
    window = QSplitter(Qt.Horizontal)
    settings, console = QWidget(), QWidget()
    window.addWidget(settings); window.addWidget(console)
    window.resize(1600, 800); window.show()
    window.setSizes([720, 880]); app.processEvents()
    screen = SimpleNamespace(_body_splitter=window, _settings_scroll=settings, _console=console)
    def unexpected_drag(*args, **kwargs):
        raise AssertionError('An already readable layout must not be dragged again')
    monkeypatch.setattr(QTest, 'mousePress', unexpected_drag)
    try:
        _balance_result_panes(screen, window, lambda _: app.processEvents())
    finally:
        window.close()
