"""A real dialog choice rebuilds an anchored legend; containment stays strict."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[3]),
                str(Path(__file__).resolve().parents[1])]

import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QWidget

import native_figure_legend as recorder
from spacr.qt.widgets.figure_settings import FigureSettingsDialog


@pytest.mark.parametrize('host,oversized', [(True, False), (False, False), (True, True)])
def test_actual_dialog_resets_external_anchor_without_changing_bars(monkeypatch, host, oversized):
    app = QApplication.instance() or QApplication([])
    figure = Figure(figsize=(1, 1) if oversized else (8, 5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.set_title('Actual Invasion fractions')
    first = axis.bar([0, 1], [.3, .6], label='attached' * (40 if oversized else 1))
    second = axis.bar([0, 1], [.7, .4], bottom=[.3, .6], label='invaded')
    axis.set_xticks([0, 1], ['well1', 'well2'])
    axis.legend(loc='upper left', bbox_to_anchor=(1.5, 1))
    figure.canvas.draw()
    original_legend = axis.get_legend()
    assert original_legend.get_window_extent(figure.canvas.get_renderer()).x1 > figure.bbox.x1
    before = [(p.get_height(), p.get_y()) for p in (*first.patches, *second.patches)]
    labels = [t.get_text() for t in axis.get_xticklabels()]
    window = QWidget()
    button = QPushButton('Figure settings', window)
    queue = SimpleNamespace(_fig_settings_btn=button)
    selected, dialogs, events = [], [], []

    def open_dialog():
        dialog = FigureSettingsDialog(figure, parent=window)
        dialogs.append(dialog)
        dialog.exec()

    def choose(capture, combo, index):
        selected.append(combo.itemText(index))
        # Actual Qt input, not direct artist or combo-value mutation.
        combo.setFocus()
        QTest.keyClick(combo, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(combo, Qt.Key_Down)
        QTest.keyClick(combo, Qt.Key_Tab)

    button.clicked.connect(open_dialog)
    capture = lambda _: None
    if host:
        capture.host = object()  # Unit-test the host selection branch only.
    monkeypatch.setattr(recorder.capture_ui, 'select_combo', choose)
    # Geometry/scrolling do not decide the legend: the actual dialog controls,
    # currentIndexChanged handler and Matplotlib containment are exercised.
    monkeypatch.setattr(recorder.capture_ui, 'dialog_geometry', lambda *_: None)
    monkeypatch.setattr(recorder.capture_ui, 'scroll_to', lambda *_: None)
    scenes = SimpleNamespace(begin=lambda *a, **k: events.append(('begin', a[0])),
                             end=lambda *a, **k: events.append(('end', a[0])))
    settle = lambda seconds: QTest.qWait(max(1, int(seconds * 1000)))
    window.resize(1100, 800)
    window.show()
    try:
        if oversized:
            with pytest.raises(ValueError, match='still exceeds the figure canvas'):
                recorder.fit_legend(app, window, queue, figure, capture, settle,
                    'legend', scenes=scenes, next_name='readable')
        else:
            result = recorder.fit_legend(app, window, queue, figure, capture, settle,
                'legend', scenes=scenes, next_name='readable')
            assert result['legend_inside_canvas'] and result['actual_native_dialog']
            bounds = axis.get_legend().get_window_extent(figure.canvas.get_renderer())
            assert figure.bbox.x0 <= bounds.x0 <= bounds.x1 <= figure.bbox.x1
            assert figure.bbox.y0 <= bounds.y0 <= bounds.y1 <= figure.bbox.y1
        assert axis.get_legend() is not original_legend
        assert axis.get_legend().get_visible()
        assert selected == (['upper right', 'best'] if host else [])
        assert [(p.get_height(), p.get_y()) for p in (*first.patches, *second.patches)] == before
        assert [t.get_text() for t in axis.get_xticklabels()] == labels
        assert events == [('begin', 'legend'), ('end', 'legend'), ('begin', 'readable')]
        assert dialogs[0].result() == QDialog.DialogCode.Accepted
    finally:
        for dialog in dialogs:
            dialog.close()
            dialog.deleteLater()
        window.close()
        window.deleteLater()
        app.processEvents()
