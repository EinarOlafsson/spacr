"""The public fitted-point tour needs no advanced folds or second fitting."""
from pathlib import Path
import hashlib
import sqlite3
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[3]),
                str(Path(__file__).resolve().parents[1])]

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QSplitter, QVBoxLayout, QWidget

import capture_ui
from capture_umap import record_explorer, set_interactive_view
from spacr.qt.widgets.umap_explorer import ImageUmapExplorer


def test_public_review_keeps_real_point_clicks_crop_and_database(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / 'project'
    (source / 'measurements').mkdir(parents=True)
    database = source / 'measurements/measurements.db'
    image = source / 'crop.png'
    Image.new('RGB', (24, 24), (80, 140, 200)).save(image)
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE png_list (png_path TEXT PRIMARY KEY)')
        connection.execute('INSERT INTO png_list VALUES (?)', (str(image),))
    original = hashlib.sha256(database.read_bytes()).hexdigest()
    host = QWidget()
    layout = QVBoxLayout(host)
    switch = QCheckBox('Interactive')
    switch.setFixedWidth(100)
    layout.addWidget(switch)
    body = QSplitter(Qt.Horizontal)
    body.addWidget(QWidget())
    explorer = ImageUmapExplorer()
    body.addWidget(explorer)
    layout.addWidget(body)
    explorer.set_payload({
        'embedding': np.array([[0., 0.], [.2, .1], [2., 2.], [2.2, 2.1]]),
        'labels': np.array([0, 0, 1, 1]),
        'records': [dict(image=image, db_path=database, db_png_path=str(image))
                    for _ in range(4)]})
    switch.toggled.connect(explorer.setVisible)
    host.resize(1800, 800)
    host.show()
    explorer.hide()
    QTest.qWait(30)
    settings = {'src': [str(source)], 'row_limit': 4, 'random_seed': 0}
    # Deliberately no advanced-fold or hyperparameter attributes. The tour
    # must use only these actual public widgets and retain the supplied data.
    screen = SimpleNamespace(_body_splitter=body, _interactive_switch=switch,
        _umap_explorer=explorer,
        _settings_model=SimpleNamespace(collect=lambda: dict(settings)))
    scenes, writes, picked = [], {}, []

    def capture(name):
        scenes.append(name)
        if name.startswith('26_point_') or name == '27_review_embedding':
            assert explorer.isVisible()
            assert not explorer._preview.source_pixmap().isNull()
            picked.append(explorer._picked)

    try:
        record_explorer(screen, tmp_path, capture,
            lambda seconds=0: QTest.qWait(15),
            lambda path, value: writes.update({path.name: value}))
        assert scenes == ['25_interactive_embedding', '26_point_0',
                          '26_point_2', '27_review_embedding']
        assert picked == [0, 2, 2]
        proof = writes['explorer_tour.json']
        assert proof['nested_routes'] == []
        assert proof['advanced_controls_toured'] is False
        assert proof['annotations_written'] is proof['search_started'] is False
        assert proof['new_embedding_requested'] is False
        assert proof['settings_unchanged'] and proof['database_unchanged']
        assert hashlib.sha256(database.read_bytes()).hexdigest() == original
        assert not switch.isChecked()
    finally:
        host.close()
        host.deleteLater()
        app.processEvents()


@pytest.mark.parametrize('unreachable', [False, True])
def test_current_umap_reopens_public_actions_before_clicking_interactive(
        monkeypatch, unreachable):
    from matplotlib.figure import Figure
    from spacr.qt.screens.app_screen import AppScreen
    import time

    app = QApplication.instance() or QApplication([])
    screen = AppScreen('umap')
    screen.resize(2200, 1250)
    screen.show()
    figure = Figure(figsize=(2, 2))
    figure.subplots().scatter([0., 1.], [0., 1.])
    figure._spacr_umap_payload = {
        'embedding': np.array([[0., 0.], [1., 1.]]),
        'labels': np.array([0, 1]),
        'records': [{'display_name': 'fixture0'}, {'display_name': 'fixture1'}]}
    screen._on_figure_ready(figure)
    QTest.qWait(100)
    clicks = []

    def physical_toggle(capture, widget, enabled):
        assert widget.isVisible() and widget.isEnabled()
        assert not screen._actions_folder.shut
        QTest.mouseClick(widget, Qt.LeftButton)
        assert widget.isChecked() == bool(enabled)
        clicks.append(enabled)

    monkeypatch.setattr(capture_ui, 'set_checked', physical_toggle)
    try:
        switch = screen._interactive_switch
        assert type(switch).__name__ == 'AiToggleLabel'
        assert screen._actions_folder.shut and not switch.isVisible()
        assert not switch.isChecked()
        before = screen._settings_model.collect()
        if unreachable:
            # An owner-hidden heading cannot be recovered by direct mutation.
            screen._actions_folder.heading.hide()
            with pytest.raises(RuntimeError, match='Actions heading is not reachable'):
                set_interactive_view(screen, None, lambda: QTest.qWait(20), True)
            assert not clicks and not switch.isChecked()
        else:
            set_interactive_view(screen, None, lambda: QTest.qWait(20), True)
            assert clicks == [True] and screen._umap_explorer.isVisible()
            set_interactive_view(screen, None, lambda: QTest.qWait(20), True)
            assert clicks == [True]  # Already enabled needs no hidden click.
            set_interactive_view(screen, None, lambda: QTest.qWait(20), False)
            assert clicks == [True, False] and not switch.isChecked()
        assert screen._settings_model.collect() == before
        assert not hasattr(screen, '_run_started_at')
    finally:
        deadline = time.monotonic() + 10
        while screen._jobs.is_busy() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert not screen._jobs.is_busy()
        screen.close()
        screen.deleteLater()
        app.processEvents()
