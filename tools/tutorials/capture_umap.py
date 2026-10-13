"""Explore real fitted points and crop previews without writing annotations."""
from __future__ import annotations

import hashlib
from pathlib import Path

from capture_settings import require_unchanged_settings


def record_explorer(screen, captures, capture, settle, write_json):
    from copy import deepcopy
    import numpy as np
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from spacr.qt.widgets.fold_strip import FoldButton

    before = deepcopy(screen._settings_model.collect())
    source = before['src']
    if isinstance(source, (list, tuple)):
        source = source[0]
    database = Path(source) / 'measurements/measurements.db'
    original = hashlib.sha256(database.read_bytes()).hexdigest()
    width = sum(screen._body_splitter.sizes())
    screen._body_splitter.setSizes([width // 4, width - width // 4])
    QTest.mouseClick(screen._interactive_switch, Qt.LeftButton)
    settle(2)
    explorer = screen._umap_explorer
    if not explorer.isVisible() or len(explorer._embedding) != len(explorer._records):
        raise RuntimeError('The real interactive embedding is not available')
    if not len(explorer._embedding) or not np.isfinite(explorer._embedding).all():
        raise RuntimeError('The fitted embedding is empty or nonfinite')
    capture('25_interactive_embedding')
    observations = []
    for index in (0, len(explorer._embedding) // 2):
        x, y = explorer._axes.transData.transform(explorer._embedding[index])
        ratio = explorer._canvas.devicePixelRatioF()
        point = QPoint(round(x / ratio), round(explorer._canvas.height() - y / ratio))
        QTest.mouseClick(explorer._canvas, Qt.LeftButton, pos=point)
        settle()
        if explorer._picked != index or explorer._preview.source_pixmap().isNull():
            raise RuntimeError('A real point click did not display its actual crop')
        capture(f'26_point_{index}')
        observations.append({'point_index': index, 'label': explorer._point_label.text(),
                             'database_path': explorer._records[index].get('db_png_path')})
    folds = {b.app_key: b for b in screen.findChildren(FoldButton) if b.isVisible()}
    for key in ('image_scatter', 'pca'):
        if key not in folds:
            raise RuntimeError(f'The expected nested route is not visible: {key}')
        QTest.mouseMove(folds[key])
        settle(0.8)
        capture('27_fold_' + key)
    QTest.mouseClick(screen._interactive_switch, Qt.LeftButton)
    settle()
    QTest.mouseClick(screen._hp_switch, Qt.LeftButton)
    settle()
    if not screen._hyperparam_card.isVisible():
        raise RuntimeError('The actual Hyperparameter search controls did not open')
    capture('28_hyperparameter_controls')
    QTest.mouseClick(screen._hp_switch, Qt.LeftButton)
    settle()
    require_unchanged_settings(before, screen._settings_model.collect())
    if hashlib.sha256(database.read_bytes()).hexdigest() != original:
        raise RuntimeError('The display-only explorer demonstration changed the database')
    write_json(captures / 'explorer_tour.json', {
        'point_count': len(explorer._records), 'clicked_points': observations,
        'nested_routes': ['image_scatter', 'pca'], 'settings_unchanged': True,
        'database_unchanged': True, 'annotations_written': False,
        'search_started': False, 'new_embedding_requested': False})
