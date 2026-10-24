"""Explore real fitted points and crop previews without writing annotations."""
from __future__ import annotations

import hashlib
from pathlib import Path

import capture_ui
from capture_settings import require_unchanged_settings
from capture_ui_drv_a import Rec


def set_interactive_view(screen, capture, settle, enabled):
    """Reach the actual Interactive control through its public Actions fold."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    switch = screen._interactive_switch
    if switch.isChecked() == bool(enabled):
        return
    if not switch.isVisible():
        folder = screen._actions_folder
        if folder.shut:
            capture_ui.scroll_to(capture, screen._runtime_viewport, folder.heading)
            if not folder.heading.isVisible():
                raise RuntimeError('The public Actions heading is not reachable')
            QTest.mouseClick(folder.heading, Qt.LeftButton)
            settle()
        capture_ui.scroll_to(capture, screen._runtime_viewport, switch)
        settle()
    if not switch.isVisible() or not switch.isEnabled():
        raise RuntimeError('The actual Interactive control is not available to click')
    capture_ui.set_checked(capture, switch, enabled)
    if switch.isChecked() != bool(enabled):
        raise RuntimeError('The actual Interactive control did not retain its state')


def record_explorer(screen, captures, capture, settle, write_json):
    from copy import deepcopy
    import numpy as np
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    before = deepcopy(screen._settings_model.collect())
    source = before['src']
    if isinstance(source, (list, tuple)):
        source = source[0]
    database = Path(source) / 'measurements/measurements.db'
    original = hashlib.sha256(database.read_bytes()).hexdigest()
    width = sum(screen._body_splitter.sizes())
    # Host: the clip shows the splitter dragged and the Interactive switch clicked.
    with Rec(capture, '25_interactive_embedding', settle):
        capture_ui.splitter_sizes(capture, screen._body_splitter, [width // 4, width - width // 4])
        set_interactive_view(screen, capture, settle, True)
        settle(2)
        explorer = screen._umap_explorer
        if not explorer.isVisible() or len(explorer._embedding) != len(explorer._records):
            raise RuntimeError('The real interactive embedding is not available')
        if not len(explorer._embedding) or not np.isfinite(explorer._embedding).all():
            raise RuntimeError('The fitted embedding is empty or nonfinite')
    observations = []
    for index in (0, len(explorer._embedding) // 2):
        x, y = explorer._axes.transData.transform(explorer._embedding[index])
        ratio = explorer._canvas.devicePixelRatioF()
        point = QPoint(round(x / ratio), round(explorer._canvas.height() - y / ratio))
        with Rec(capture, f'26_point_{index}', settle) as picked:
            QTest.mouseClick(explorer._canvas, Qt.LeftButton, pos=point)
            settle()
            if explorer._picked != index or explorer._preview.source_pixmap().isNull():
                raise RuntimeError('A real point click did not display its actual crop')
            picked.point(explorer._preview, dwell=1.5)
        observations.append({'point_index': index, 'label': explorer._point_label.text(),
                             'database_path': explorer._records[index].get('db_png_path')})
    # Finish on the already fitted public embedding and the crop just picked.
    # Alpha-only folded tools and optional search controls are not this tour.
    with Rec(capture, '27_review_embedding', settle) as review:
        if not explorer.isVisible() or explorer._preview.source_pixmap().isNull():
            raise RuntimeError('The fitted embedding and selected crop must remain visible')
        review.point(explorer._canvas, explorer._preview, dwell=1.5)
    set_interactive_view(screen, capture, settle, False)
    settle()
    require_unchanged_settings(before, screen._settings_model.collect())
    if hashlib.sha256(database.read_bytes()).hexdigest() != original:
        raise RuntimeError('The display-only explorer demonstration changed the database')
    write_json(captures / 'explorer_tour.json', {
        'point_count': len(explorer._records), 'clicked_points': observations,
        'nested_routes': [], 'advanced_controls_toured': False, 'settings_unchanged': True,
        'database_unchanged': True, 'annotations_written': False,
        'search_started': False, 'new_embedding_requested': False})
