"""Record real Model Zoo inventory/provenance, not the incompatible benchmark."""
import time
from pathlib import Path

from capture_model_zoo import MODELS, PRIMARY, _directory_state, _fingerprint


def record_inventory(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QScrollArea
    from spacr.qt.screens.model_zoo import ModelZooScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    deadline = time.monotonic() + timeout

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('The bounded model-inventory recording timed out')
        settle(0)

    def expose(widget, *, enabled=True):
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(widget)
            parent = parent.parentWidget()
        settle(.15)
        if not widget.isVisible() or (enabled and not widget.isEnabled()) or widget.visibleRegion().isEmpty():
            raise ValueError('The real model-inventory control is not available on screen')

    def click(widget):
        expose(widget)
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def idle(zoo):
        while zoo.is_busy() or zoo.active_jobs():
            tick()
            settle(.1)
        if zoo.last_error:
            raise ValueError(zoo.last_error)

    before = _directory_state(MODELS)
    original = _fingerprint(PRIMARY, tick)
    buttons = [b for b in screen.findChildren(FoldButton)
               if b.app_key == 'model_zoo' and b.isVisible()]
    if len(buttons) != 1:
        raise ValueError('Expected the actual Make Masks to Model Zoo fold')
    capture('01_make_masks_host')
    click(buttons[0])
    widgets = [w for w in window.findChildren(ModelZooScreen) if w.isVisible()]
    if len(widgets) != 1:
        raise ValueError('The actual Model Zoo fold did not open')
    zoo = widgets[0]
    idle(zoo)
    if zoo._segment_fn is not None or zoo.result() is not None:
        raise ValueError('Expected an unmodified fresh Model Zoo')
    capture('02_model_library')
    expose(zoo._scan_edit)
    zoo._scan_edit.setFocus()
    QTest.keyClick(zoo._scan_edit, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(zoo._scan_edit, str(MODELS))
    QTest.keyClick(zoo._scan_edit, Qt.Key_Tab)
    capture('03_cache_path')
    click(zoo._btn_scan)
    idle(zoo)
    if Path(zoo._scan_edit.text()).resolve() != MODELS.resolve():
        raise ValueError('The displayed model directory changed')
    capture('04_scanned_models')
    entries = zoo.entries()
    choices = []
    for row in range(zoo._table.rowCount()):
        item = zoo._table.item(row, 0)
        index = None if item is None else item.data(Qt.UserRole)
        if index is None:
            continue
        entry = entries[int(index)]
        if entry.path and Path(entry.path).resolve() == PRIMARY.resolve():
            choices.append(item)
    if len(choices) != 1:
        raise ValueError('The real scan must expose one exact local cpsam checkpoint')
    zoo._table.scrollToItem(choices[0])
    expose(zoo._table)
    rect = zoo._table.visualItemRect(choices[0])
    QTest.mouseClick(zoo._table.viewport(), Qt.LeftButton, pos=rect.center())
    settle(.3)
    selected = zoo.selected_entries()
    if (len(selected) != 1 or Path(selected[0].path).resolve() != PRIMARY.resolve()
            or selected[0].kind != 'cellpose' or not selected[0].exists):
        raise ValueError('Visible selection is not the actual cached Cellpose checkpoint')
    entry = selected[0]
    if not zoo.detail_text() or zoo.detail_text() != entry.describe():
        raise ValueError('The provenance panel does not describe the selected model')
    expose(zoo._detail)
    capture('05_selected_provenance')
    expose(zoo._btn_download, enabled=False)
    capture('06_download_controls_not_started')
    expose(zoo._btn_test, enabled=False)
    capture('07_benchmark_controls_not_run')
    idle(zoo)
    if (_fingerprint(PRIMARY, tick) != original or _directory_state(MODELS) != before
            or zoo.result() is not None):
        raise ValueError('Inventory inspection must not change weights or create benchmark results')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Actual Model Zoo inventory, scanning and provenance controls only',
        'route': ['make_masks', 'model_zoo'], 'model_path': str(PRIMARY),
        'selected_model_fingerprint': original, 'provenance': zoo.detail_text(),
        'checksum_state': entry.checksum_state, 'trained_on': entry.trained_on,
        'trained_by': entry.trained_by, 'displayed_rows': zoo._table.rowCount(),
        'checkpoint_provenance_independently_validated': False,
        'original_checkpoint_unchanged': True, 'model_directory_unchanged': True,
        'benchmark_completed': False, 'segmentation_performed': False,
        'model_downloaded': False, 'training_performed': False,
        'download_enabled_for_selected_checkpoint': zoo._btn_download.isEnabled(),
        'test_enabled_without_fields': zoo._btn_test.isEnabled(),
        'inference_overridden': False, 'app_source_modified': False, 'published': False})


if __name__ == '__main__':
    from capture_barcode_saved_plots import launch
    raise SystemExit(launch('model_zoo', 'model_zoo_1507_inventory_v2', ['--model-zoo-inventory'], 150))
