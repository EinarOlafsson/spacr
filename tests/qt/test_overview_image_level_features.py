"""Overview measurements carry image-level features into fov and join annotations."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.measure import _image_measurement_frame, _set_image_annotations
from spacr.qt.widgets.qc_field_browser import (
    _OverviewField,
    _overview_measurement,
    _overview_measurements_export,
)
from spacr.sp_stats import _compare_image_groups, _representative_images

pytestmark = pytest.mark.qt


def _field(tmp_path, well, column, name, level):
    files = []
    for channel in (1, 2):
        pixels = np.full((32, 32), 100, dtype=np.uint16)
        pixels[8:16, 8:16] = level * channel
        path = tmp_path / f"{well}_{name}_c{channel}.npy"
        np.save(path, pixels)
        files.append((str(channel), str(path)))
    return _OverviewField("plate1", well, "A", column, name, tuple(files))


def test_measurement_row_holds_image_level_features(tmp_path):
    row = _overview_measurement(_field(tmp_path, "A01", 1, "1", 1000), (1, 2))
    assert row["status"] == "ok", row["error"]
    assert row["channel_1_objects_count"] == 1
    assert row["channel_1_otsu_area_fraction"] == pytest.approx(64 / 1024)
    assert row["channels_1_2_manders_m1"] == pytest.approx(1.0)
    assert row["object_source"] == "otsu_per_channel"
    assert row["channel_2_saturation_level"] == 65535


def test_exported_fields_join_annotations_for_comparison(tmp_path):
    fields = [_field(tmp_path, f"A0{column}", column, str(index), 400 + 300 * (column > 2) + index)
              for column in (1, 2, 3, 4) for index in range(3)]
    report = _overview_measurements_export(fields, tmp_path, (1, 2))
    assert report["count"] == 12 and not report["errors"]
    db = tmp_path / "measurements" / "measurements.db"
    _set_image_annotations(db, fields, "cell_type", "HeLa", source=tmp_path,
                           wells=[("plate1", "A01"), ("plate1", "A02")])
    _set_image_annotations(db, fields, "cell_type", "HFF", source=tmp_path,
                           wells=[("plate1", "A03"), ("plate1", "A04")])
    frame = _image_measurement_frame(db)
    assert len(frame) == 12 and frame["cell_type"].notna().all()
    result = _compare_image_groups(frame, ["channel_1_foreground_mean"], ["cell_type"])
    summary = result["summary"].set_index("group")
    assert summary.loc["HFF", "mean"] > summary.loc["HeLa", "mean"]
    ranked = _representative_images(frame, ["cell_type"], method="typical_in_focus", top=2)
    assert set(ranked["cell_type"]) == {"HFF", "HeLa"}


from tests.qt.test_qc_field_browser_ratchet_edges_20261024 import (  # noqa: E402,F401
    overview, source_planes)


class _HeldOverviewJobs:
    """Hold real queued operations at their runner boundary for ordered delivery."""
    def __init__(self):
        self.calls = []

    def submit(self, run, shown):
        self.calls.append((run, shown))

    def is_busy(self):
        return bool(self.calls)


def _annotation_rows(source):
    """Read physical identities and values from the real annotation database."""
    import sqlite3
    from spacr.measure import _image_annotation_database

    with sqlite3.connect(_image_annotation_database(source)) as connection:
        return connection.execute('SELECT plateID, wellID, fieldID, condition '
                                  'FROM image_annotations ORDER BY plateID, wellID, fieldID').fetchall()


def test_queued_annotation_failure_does_not_erase_previous_rows_or_block_the_next_write(
        overview, monkeypatch):
    import sqlite3
    import importlib
    measure = importlib.import_module('spacr.measure')

    dialog = overview
    source = dialog._metadata_source
    db = measure._image_annotation_database(source)
    measure._set_image_annotations(db, dialog._fields, 'condition', 'before', source=source)
    baseline = _annotation_rows(source)
    original = measure._set_image_annotations
    jobs = _HeldOverviewJobs()
    seen = []

    def write(path, fields, key, value, **kwargs):
        """Refuse one disk write and retain the ordinary writer for its successor."""
        seen.append(value)
        if value == 'refused':
            assert _annotation_rows(source) == baseline
            raise sqlite3.OperationalError('read-only filesystem')
        return original(path, fields, key, value, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(dialog, '_metadata_jobs', jobs)
        local.setattr(measure, '_set_image_annotations', write)
        dialog._store_image_annotation({('p1', 'A01')}, 'condition', 'refused')
        dialog._store_image_annotation({('p1', 'A01')}, 'condition', 'after')
        assert len(jobs.calls) == 1
        run, shown = jobs.calls.pop()
        results = run()
        assert [count for entry, count in results] == [None, 2]
        assert seen == ['refused', 'after']
        assert dialog._image_annotation_running is False
        assert dialog._image_annotation_queue == []
        shown(results)
        assert 'Saved condition for 2 images' in dialog.status.text()
    rows = _annotation_rows(source)
    assert len(rows) == 4
    assert {(well, condition) for plate, well, field, condition in rows} == {
        ('A01', 'after'), ('B02', 'before')}


def test_interrupted_annotation_drain_releases_ownership_and_a_later_write_recovers(overview, monkeypatch):
    import importlib
    measure = importlib.import_module('spacr.measure')

    dialog = overview
    original = measure._set_image_annotations
    jobs = _HeldOverviewJobs()

    def interrupted(*args, **kwargs):
        """Model cancellation before the database operation starts."""
        raise KeyboardInterrupt('cancel annotation write')

    with monkeypatch.context() as local:
        local.setattr(dialog, '_metadata_jobs', jobs)
        local.setattr(measure, '_set_image_annotations', interrupted)
        dialog._store_image_annotation({('p1', 'A01')}, 'condition', 'cancelled')
        run, shown = jobs.calls.pop()
        with pytest.raises(KeyboardInterrupt, match='cancel annotation'):
            run()
        assert not dialog._image_annotation_running
        local.setattr(measure, '_set_image_annotations', original)
        dialog._store_image_annotation({('p1', 'A01')}, 'condition', 'recovered')
        run, shown = jobs.calls.pop()
        shown(run())
    assert {value for plate, well, field, value in _annotation_rows(dialog._metadata_source)} == {'recovered'}


def test_annotation_completion_cannot_overwrite_a_later_file_action_report(overview, monkeypatch):
    dialog = overview
    jobs = _HeldOverviewJobs()
    with monkeypatch.context() as local:
        local.setattr(dialog, '_metadata_jobs', jobs)
        dialog._store_image_annotation({('p1', 'A01')}, 'condition', 'saved')
        run, shown = jobs.calls.pop()
        result = run()
        dialog._annotation_file_actions += 1
        dialog.status.setText('A later import was refused')
        shown(result)
        shown([])
        assert dialog.status.text() == 'A later import was refused'
        assert {value for plate, well, field, value in _annotation_rows(dialog._metadata_source)} == {'saved'}
        local.setattr(dialog, '_metadata_source', '')
        dialog._store_image_annotation({('p1', 'B02')}, 'condition', 'not queued')
        assert jobs.calls == []


@pytest.mark.parametrize('no_source', [False, True])
def test_scheme_selection_cancel_or_missing_source_queues_no_import(overview, monkeypatch, tmp_path, no_source):
    from PySide6.QtWidgets import QFileDialog

    dialog = overview
    jobs = _HeldOverviewJobs()
    path = str(tmp_path / 'scheme.csv') if no_source else ''
    with monkeypatch.context() as local:
        local.setattr(dialog, '_export_jobs', jobs)
        local.setattr(QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (path, 'CSV'))
        if no_source:
            local.setattr(dialog, '_metadata_source', '')
        dialog._import_annotation_scheme()
        assert jobs.calls == []
        assert not (tmp_path / 'scheme.csv').exists()


def test_scheme_decline_and_stale_preview_preserve_annotations_and_database(overview, monkeypatch, tmp_path):
    import copy
    import pandas as pd
    from PySide6.QtWidgets import QMessageBox
    from spacr.measure import _image_annotation_database

    dialog = overview
    frame = pd.DataFrame([['p1', 'A01', 'proposed']], columns=['plate', 'well', 'condition'])
    before = copy.deepcopy(dialog._annotations), copy.deepcopy(dialog._conditions)
    db = _image_annotation_database(dialog._metadata_source)
    before_db = db.read_bytes() if db.exists() else None
    jobs = _HeldOverviewJobs()
    questions = []
    with monkeypatch.context() as local:
        local.setattr(dialog, '_export_jobs', jobs)
        local.setattr(QMessageBox, 'question', lambda *args: (questions.append(args), QMessageBox.No)[1])
        dialog._scheme_previewed(dialog._fields, str(tmp_path / 'scheme.csv'), frame)
        assert len(questions) == 1
        stale = tuple(list(dialog._fields))
        assert stale is not dialog._fields
        dialog._scheme_previewed(stale, str(tmp_path / 'scheme.csv'), frame)
        assert len(questions) == 1 and jobs.calls == []
    assert (dialog._annotations, dialog._conditions) == before
    assert (db.read_bytes() if db.exists() else None) == before_db


def test_confirmed_scheme_import_updates_only_matching_wells_and_reports_unmatched_rows(
        overview, monkeypatch, tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QMessageBox
    from spacr.qt.widgets import qc_field_browser as module

    dialog = overview
    scheme = tmp_path / 'scheme.csv'
    pd.DataFrame([['p1', 'A01', 'treated'], ['p1', 'Z99', 'unmatched']],
                 columns=['plate', 'well', 'condition']).to_csv(scheme, index=False)
    frame = module._overview_scheme_preview(str(scheme))
    jobs = _HeldOverviewJobs()
    with monkeypatch.context() as local:
        local.setattr(dialog, '_export_jobs', jobs)
        local.setattr(QMessageBox, 'question', lambda *args: QMessageBox.Yes)
        dialog._scheme_previewed(dialog._fields, str(scheme), frame)
        assert len(jobs.calls) == 1
        run, shown = jobs.calls.pop()
        report = run()
        dialog.status.setText('Current source owns this view')
        dialog._scheme_imported(tuple(list(dialog._fields)), {}, report)
        assert dialog.status.text() == 'Current source owns this view'
        shown(report)
        assert dialog._conditions[('p1', 'A01')] == 'treated'
        assert ('p1', 'B02') not in dialog._conditions
        assert '1 wells of the scheme are not in this source' in dialog.status.text()
    rows = _annotation_rows(dialog._metadata_source)
    assert len(rows) == 2
    assert {(plate, well, value) for plate, well, field, value in rows} == {('p1', 'A01', 'treated')}


def test_invalid_secondary_plate_geometry_does_not_clear_the_valid_plate(overview, monkeypatch):
    from dataclasses import replace

    dialog = overview
    fields = dialog._fields
    invalid = tuple(replace(field, plate='p2', well='C01', row='3', column=1)
                    for field in fields[:2])
    with monkeypatch.context() as local:
        local.setattr(dialog, '_fields', fields + invalid)
        dialog.plates.blockSignals(True)
        dialog.plates.addItem('p2', 'p2')
        dialog.plates.blockSignals(False)
        dialog._set_plates_view('side', persist=False)
        assert dialog._geometry_valid
        assert set(dialog._plate_available) == {'p1'}
        assert len(dialog._plate_available['p1']) == 4
        table = dialog._plate_tables['p2']
        assert table.rowCount() == table.columnCount() == 0
        assert 'does not contain the indexed wells' in dialog._plate_blocks[table][1].toolTip()
        assert set(dialog._well_fields) == {('p1', 'A01'), ('p1', 'B02')}


def test_cross_plate_press_preserves_extended_or_selected_wells_and_clears_plain_other_selection(
        overview, monkeypatch):
    from dataclasses import replace
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    dialog = overview
    fields = dialog._fields
    with monkeypatch.context() as local:
        local.setattr(dialog, '_fields', fields + tuple(replace(field, plate='p2') for field in fields))
        dialog.plates.blockSignals(True)
        dialog.plates.addItem('p2', 'p2')
        dialog.plates.blockSignals(False)
        dialog._set_plates_view('side', persist=False)
        first, other = dialog._plate_tables['p1'], dialog._plate_tables['p2']
        first.item(0, 0).setSelected(True)
        other.item(0, 0).setSelected(True)
        selected = QPointF(first.visualItemRect(first.item(0, 0)).center())
        blank = QPointF(first.visualRect(first.model().index(0, 1)).center())
        selected_press = QMouseEvent(QEvent.MouseButtonPress, selected, Qt.LeftButton,
                                     Qt.LeftButton, Qt.NoModifier)
        dialog._press_plate(first, selected_press)
        assert other.item(0, 0).isSelected()
        extended_press = QMouseEvent(QEvent.MouseButtonPress, blank, Qt.LeftButton,
                                     Qt.LeftButton, Qt.ControlModifier)
        dialog._press_plate(first, extended_press)
        assert other.item(0, 0).isSelected()
        plain_press = QMouseEvent(QEvent.MouseButtonPress, blank, Qt.LeftButton,
                                  Qt.LeftButton, Qt.NoModifier)
        dialog._press_plate(first, plain_press)
        assert not other.selectedItems()
        assert first.item(0, 0).isSelected()


def test_canvas_refuses_invalid_channels_but_keeps_a_detached_source_snapshot(overview, monkeypatch):
    dialog = overview
    old = dialog.channels.text()
    dialog.channels.blockSignals(True)
    dialog.channels.setText('not-a-channel')
    try:
        snapshot = dialog._canvas_snapshot()
        assert snapshot._channels == (0,) and snapshot._limits is None
        assert len(snapshot._fields) == len(dialog._fields) == 4
        assert dialog._canvas_channel_names() == []
        assert dialog._canvas_display_note() == ()
    finally:
        dialog.channels.setText(old)
        dialog.channels.blockSignals(False)
    with monkeypatch.context() as local:
        local.setattr(dialog, '_normalization_context', None)
        local.setattr(dialog, '_normalization_plan', {'mode': 'stale-mode', 'sample_fields': 'not-a-number'})
        local.setattr(dialog, '_overview_options', dict(dialog._overview_options, pixel_size_um=1.25))
        text = '\n'.join(dialog._canvas_display_note())
        assert text.startswith('Display settings\nChannels:')
        assert 'Pixel size: 1.25 µm' in text
        assert 'stale-mode' not in text and 'Display normalization' not in text


def test_canvas_selection_uses_physical_well_identities_and_freezes_each_boxes_display(overview):
    import copy
    from spacr.qt.widgets import qc_field_browser as module

    dialog = overview
    assert dialog._canvas_pick([]) == []
    composer = dialog._open_custom_export()
    try:
        dialog._selected_fields.clear()
        dialog._selected_wells.clear()
        assert dialog._canvas_add_selection(composer) == []
        assert 'Select fields or wells' in composer.status.text()
        dialog._selected_wells.add(('p1', 'A01'))
        boxes = dialog._canvas_add_selection(composer)
        assert len(boxes) == 2
        assert {(box['source']['plate'], box['source']['well'], box['source']['field']) for box in boxes} == {
            ('p1', 'A01', '1 / T1 / Z1'), ('p1', 'A01', '2 / T1 / Z1')}
        assert {tuple(box['source']['key'][1:]) for box in boxes} == {
            ('p1', 'A01', '1', '1', '1'), ('p1', 'A01', '2', '1', '1')}
        captured = [copy.deepcopy(box['overview_display']) for box in boxes]
        field = next(field for field in dialog._fields if field.well == 'B02')
        added = dialog._canvas_pick([module._overview_canvas_image(field)])
        assert len(added) == 1 and added[0]['source']['well'] == 'B02'
        assert 'Added 1 images to page 1' in composer.status.text()
        old = dialog.low.value()
        dialog.low.blockSignals(True)
        dialog.low.setValue(10. if old != 10. else 11.)
        assert [box['overview_display'] for box in boxes] == captured
        dialog.low.setValue(old)
        dialog.low.blockSignals(False)
    finally:
        dialog._close_export_dock()
    assert dialog._canvas_pick([module._overview_canvas_image(field)]) == []


def test_guided_export_without_indexed_fields_creates_no_export_panel(overview, monkeypatch):
    dialog = overview
    with monkeypatch.context() as local:
        local.setattr(dialog, '_fields', ())
        assert dialog._open_guided_export() is None
        assert dialog._export_dock is None
        assert dialog.status.text() == 'Select fields or load a plate before exporting.'


def test_failed_last_annotation_rolls_back_earlier_images_and_undo_history(overview, qtbot):
    import sqlite3
    from spacr.measure import _image_annotation_database

    dialog = overview
    source = dialog._metadata_source
    database = _image_annotation_database(source)
    _set_image_annotations(database, dialog._fields, 'condition', 'before', source=source)
    baseline = _annotation_rows(source)
    assert {field_id for plate, well, field_id, value in baseline if well == 'A01'} == {'f1', 'f2'}
    with sqlite3.connect(database) as connection:
        history = connection.execute('SELECT * FROM image_annotations_history').fetchall()
        connection.execute("CREATE TRIGGER refuse_second_image BEFORE UPDATE OF condition "
                           "ON image_annotations WHEN OLD.fieldID = 'f2' "
                           "BEGIN SELECT RAISE(ABORT, 'second image is read only'); END")
    dialog._store_image_annotation({('p1', 'A01')}, 'condition', 'refused')
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy()
                    and 'could not be saved' in dialog.status.text(), timeout=10000)
    assert _annotation_rows(source) == baseline
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT * FROM image_annotations_history').fetchall() == history
    assert not dialog._image_annotation_running
    assert dialog._image_annotation_queue == []
    assert str(database) in dialog.status.text()


def test_import_scheme_waits_for_the_active_export_without_opening_a_picker(overview, qtbot, monkeypatch):
    import threading
    from unittest.mock import Mock
    from PySide6.QtWidgets import QFileDialog

    dialog = overview
    entered, release = threading.Event(), threading.Event()

    def export():
        entered.set()
        assert release.wait(10)
        return 'existing export completed'

    picker = Mock(side_effect=AssertionError('an active export must keep ownership'))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', picker)
    dialog._export_jobs.submit(export)
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        submitted = dialog._export_jobs._next_id
        dialog._import_annotation_scheme()
        assert 'Wait for the current export' in dialog.status.text()
        assert dialog._export_jobs._next_id == submitted
        picker.assert_not_called()
    finally:
        release.set()
        qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=10000)


@pytest.mark.parametrize('stacked', [False, True])
def test_representative_rows_without_indexed_images_do_not_replace_the_image_view(overview, qtbot, stacked):
    import pandas as pd
    from pathlib import Path
    from spacr import tabular

    dialog = overview
    database = Path(dialog._metadata_source) / 'measurements' / 'measurements.db'
    tabular.write_database(pd.DataFrame([
        dict(plate='p1', well='A01', field='999', time_id='1', z_id='1',
             channel_1_mean=2., status='ok')]), database, 'fov',
        if_exists='replace', canonicalise=False)
    assert dialog.group_builder.add_group({'well': 'A01'})
    qtbot.waitUntil(lambda: not dialog._group_jobs.is_busy(), timeout=10000)
    dialog.comparison_measurement.setCurrentText('channel_1_mean')
    dialog.representative_method.setCurrentIndex(dialog.representative_method.findData('measurement'))
    before = dialog._comparison_signature
    dialog.status.setText('Existing comparison owns the image view')
    dialog._show_representatives(stacked=stacked)
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=10000)
    assert dialog._comparison_signature == (before if stacked else None)
    assert not dialog._image_views
    if stacked:
        assert dialog.status.text() == 'Existing comparison owns the image view'
    else:
        assert dialog.status.text() == 'No measured image matches the chosen annotation values.'
    assert dialog.comparison_table.rowCount() == 0
    assert not (Path(dialog._metadata_source) / 'results' / 'overview_comparisons').exists()


def test_invalid_display_channels_preserve_the_last_visual_request_and_source_images(overview):
    from pathlib import Path

    dialog = overview
    before_signature = dialog._visual_signature
    before_jobs = (dialog._field_thumbnail_jobs._next_id, dialog._thumbnail_jobs._next_id,
                   dialog._montage_jobs._next_id)
    originals = {Path(path): Path(path).read_bytes() for field in dialog._fields for _, path in field.files}
    previous = dialog.channels.text()
    dialog.channels.blockSignals(True)
    try:
        dialog.channels.setText('not-a-channel')
        dialog._update_visuals()
        assert 'nonnegative channel indices' in dialog.status.text()
        assert dialog._visual_signature == before_signature
        assert (dialog._field_thumbnail_jobs._next_id, dialog._thumbnail_jobs._next_id,
                dialog._montage_jobs._next_id) == before_jobs
        assert all(path.read_bytes() == contents for path, contents in originals.items())
    finally:
        dialog.channels.setText(previous)
        dialog.channels.blockSignals(False)


def test_explicit_canvas_field_selection_takes_precedence_over_selected_wells(overview):
    from PySide6.QtCore import Qt

    dialog = overview
    dialog._well(dialog.wells.item(*dialog._well_positions[('p1', 'B02')]))
    dialog.fields.clearSelection()
    chosen = dialog.fields.item(0)
    chosen.setSelected(True)
    dialog._remember_selection()
    field = chosen.data(Qt.UserRole)
    assert set(dialog._selected_fields) == {field.identity}
    dialog._selected_wells = {('p1', 'A01')}
    composer = dialog._open_custom_export()
    try:
        boxes = dialog._canvas_add_selection(composer)
        assert len(boxes) == 1
        assert tuple(boxes[0]['source']['key'][1:]) == ('p1', 'B02', '1', '1', '1')
        assert boxes[0]['source']['well'] == 'B02'
    finally:
        dialog._close_export_dock()


def test_representative_value_filter_and_refused_stack_preserve_the_current_comparison(overview, qtbot):
    from pathlib import Path
    import pandas as pd
    from PySide6.QtCore import Qt
    from spacr import tabular
    from spacr.schema import field_id

    dialog = overview
    dialog._apply_annotation({('p1', 'A01')}, 'condition', 'control')
    dialog._apply_annotation({('p1', 'B02')}, 'condition', 'treated')
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy(), timeout=10000)
    records = [dict(plate=field.plate, well=field.well, field=str(field.field), fieldID=field_id(field.field_id),
                    time_id=str(field.time_id), z_id=str(field.z_id),
                    channel_1_mean=2. if field.well == 'A01' else 8., status='ok')
               for field in dialog._fields if field.field_id == '1']
    assert {record['well'] for record in records} == {'A01', 'B02'}
    database = Path(dialog._metadata_source) / 'measurements' / 'measurements.db'
    tabular.write_database(pd.DataFrame(records), database, 'fov', if_exists='replace', canonicalise=False)
    for item in dialog._comparison_items():
        item.setCheckState(0, Qt.Checked if item.parent().text(0) == 'condition'
                           and item.text(0) == 'treated' else Qt.Unchecked)
    assert dialog._comparison_selection()[:2] == (['condition'], {'condition': ['treated']})
    dialog.comparison_measurement.setCurrentText('channel_1_mean')
    dialog.representative_method.setCurrentIndex(dialog.representative_method.findData('measurement'))
    dialog._show_representatives()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy() and not dialog._montage_jobs.is_busy()
                    and 'Representative images' in dialog.status.text(), timeout=10000)
    expected = next(field.identity for field in dialog._fields if field.well == 'B02' and field.field_id == '1')
    assert dialog._comparison_signature[0] == (expected,)
    assert dialog.view._item is not None
    signature, caption = dialog._comparison_signature, dialog.status.text()
    pixmap = dialog.view._item.pixmap().toImage()
    dialog.comparison_measurement.setCurrentText('unmeasured_feature')
    dialog._show_representatives(stacked=True)
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=10000)
    assert dialog._comparison_signature == signature
    assert dialog.status.text() == caption
    assert dialog.view._item.pixmap().toImage() == pixmap
    assert not dialog._image_views


def test_original_psf_refusal_shows_the_ordinary_image_without_modifying_its_tiff(overview, qtbot, monkeypatch):
    from pathlib import Path
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from spacr.qt.widgets import qc_field_browser as module

    dialog = overview
    dialog._well(dialog.wells.item(*dialog._well_positions[('p1', 'B02')]))
    dialog.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy() and dialog.view._item is not None, timeout=10000)
    field = dialog.fields.currentItem().data(Qt.UserRole)
    originals = {Path(path): Path(path).read_bytes() for _, path in field.files}
    channels, limits = dialog._choices()
    expected = module._overview_pixels(field, channels, limits, edge=None, **dialog._display_kwargs(view=True))
    monkeypatch.setattr(dialog, '_original_steps', lambda: dict(
        psf_operation='deconvolve', psf=None, psf_error='A calibrated PSF is unavailable'))
    dialog.preview_resolution.setCurrentIndex(dialog.preview_resolution.findData(None))
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy()
                    and 'Original image shown without PSF deconvolution' in dialog.status.text(), timeout=10000)
    assert 'A calibrated PSF is unavailable' in dialog.status.text()
    assert dialog._original_image
    image = dialog.view._item.pixmap().toImage().convertToFormat(QImage.Format_RGB888)
    actual = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
    actual = actual[:, :image.width() * 3].reshape(image.height(), image.width(), 3)
    np.testing.assert_array_equal(actual, expected)
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert dialog._main_view_field.identity == field.identity


@pytest.mark.parametrize('keep_other_well_open', [False, True])
def test_hiding_the_active_plate_transfers_navigation_to_a_surviving_well_or_plate(
        overview, qtbot, keep_other_well_open):
    from pathlib import Path
    import shutil

    dialog = overview
    source = Path(dialog._metadata_source)
    shutil.copyfile(source / 'p1-B02-f1-t1-z1-c1.tif', source / 'p2-A01-f1-t1-z1-c1.tif')
    dialog._scan()
    qtbot.waitUntil(lambda: len(dialog._fields) == 5 and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=10000)
    dialog._set_plates_view('side', persist=False)
    assert set(dialog._plate_tables) == {'p1', 'p2'}
    for plate, table in dialog._plate_tables.items():
        table.item(*dialog._well_positions[(plate, 'A01')]).setSelected(True)
    assert dialog._selected_wells == {('p1', 'A01'), ('p2', 'A01')}
    dialog._open_selected_well()
    assert {(panel.field_list._overview_plate, panel.field_list._overview_well)
            for panel in dialog._well_panels} == {('p1', 'A01'), ('p2', 'A01')}
    assert dialog._scope == ('p2', 'A01')
    if not keep_other_well_open:
        first = next(panel for panel in dialog._well_panels if panel.field_list._overview_plate == 'p1')
        first.close_button.click()
    dialog._set_plates_view(plates=('p1',), persist=False)
    assert set(dialog._plate_tables) == {'p1'}
    assert all(panel.field.plate == 'p1' for panel in dialog._image_views)
    if keep_other_well_open:
        assert len(dialog._well_panels) == 1
        assert dialog.fields is dialog._well_panels[0].field_list
        assert dialog._scope == ('p1', 'A01')
        assert {field.well for field in dialog._visible} == {'A01'}
    else:
        assert dialog._well_panels == []
        assert dialog.fields is dialog._well_panel.field_list
        assert dialog._scope == ('p1', None)
        assert {field.well for field in dialog._visible} == {'A01', 'B02'}
