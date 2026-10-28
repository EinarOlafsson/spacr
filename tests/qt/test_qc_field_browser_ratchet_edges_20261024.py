"""Edge cases of the field browser, overview and metadata wizard.

An unreadable path in the worker loader, a failed load report, a dialog torn
down twice, a native crop for a field without a recorded shape, a stale
comparison result, and a wizard whose action fit is already running or whose
buttons carry default and flat styling.
"""
import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from spacr.qt.widgets import qc_field_browser as module
from spacr.qt.widgets.qc_field_browser import (
    QCFieldBrowser, QCFieldTarget, QCFieldVerdict)

pytestmark = pytest.mark.qt

PATTERN = (r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)"
           r"-t(?P<timeID>\d+)-z(?P<sliceID>\d+)-c(?P<chanID>\d+)\.tif")


def _missing_target(tmp_path):
    merged = tmp_path / "plate1" / "merged"
    merged.mkdir(parents=True)
    return QCFieldTarget(
        field="plate1_A01_9", plate_root=str(tmp_path / "plate1"),
        merged_dir=str(merged),
        verdicts=(QCFieldVerdict("cell", "fail", ("unreadable",), "gone"),))


def test_loader_primes_only_real_paths_and_reports_unstatable_ones_absent(
        monkeypatch, tmp_path):
    from pathlib import Path

    primed = []
    monkeypatch.setattr(module.path_probe, "prime",
                        lambda path, present: primed.append((path, present)))

    def refuse(self):
        raise OSError("stale mount")

    monkeypatch.setattr(Path, "is_file", refuse)
    target = _missing_target(tmp_path)
    image = module._load_and_record_state(target, "en", ("", "/stale/field.npy"))
    assert primed == [("/stale/field.npy", False)]
    assert image.error


def test_a_failed_load_is_reported_and_a_second_teardown_is_harmless(
        qtbot, tmp_path, monkeypatch):
    browser = QCFieldBrowser([_missing_target(tmp_path)], threaded=False)
    qtbot.addWidget(browser)
    browser.show()
    browser._on_load_failed("disk went away")
    assert "disk went away" in browser._load_status.text()
    browser._on_finished(0)

    class Refusing:
        refused = 0

        def disconnect(self, *_):
            Refusing.refused += 1
            raise TypeError("already disconnected")

    class Probes:
        answered = Refusing()

    monkeypatch.setattr(module.path_probe, "probes", Probes)
    browser._on_finished(0)
    browser.close()
    assert Refusing.refused >= 2
    assert not browser._run_timer.isActive()
    assert not browser._recheck_timer.isActive()


@pytest.fixture
def source_planes(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for well in ("A01", "B02"):
        for field in (1, 2):
            path = source / f"p1-{well}-f{field}-t1-z1-c1.tif"
            pixels = np.arange(1200, dtype=np.uint16).reshape(30, 40) + field
            if well == "A01" and field == 1:
                pixels = np.arange(1536 * 2048, dtype=np.uint16).reshape(1536, 2048)
            tifffile.imwrite(path, pixels)
    return source


def _retire(dialog, qtbot):
    dialog._cancel()
    runners = (dialog._index_jobs, dialog._image_jobs, dialog._thumbnail_jobs,
               dialog._field_thumbnail_jobs, dialog._montage_jobs, dialog._export_jobs,
               dialog._metadata_jobs, dialog._measurement_jobs, dialog._native_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners),
                    timeout=10000)
    for runner in runners:
        runner.shutdown(timeout_ms=3000)
    from spacr.qt.bridge import prune_parked_threads
    qtbot.waitUntil(lambda: prune_parked_threads() == 0)
    dialog.close()


@pytest.fixture
def overview(qtbot, source_planes):
    dialog = module._FieldOverview(settings={
        "source": str(source_planes), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1,), "geometry": (2, 2), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=10000)
    return dialog


def test_native_crop_is_not_requested_for_a_field_without_a_recorded_shape(
        overview, qtbot):
    dialog = overview
    dialog._well(dialog.wells.item(*dialog._well_positions[(dialog.plates.currentData(), "A01")]))
    dialog.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog.view._item is not None
                    and not dialog._image_jobs.is_busy())
    dialog._native_timer.stop()
    qtbot.waitUntil(lambda: not dialog._native_jobs.is_busy())
    dialog._source_shapes.clear()
    submitted = dialog._native_jobs._next_id
    dialog._load_native_detail()
    assert dialog._native_jobs._next_id == submitted
    assert not dialog._native_jobs.is_busy()
    assert not dialog._native_pending


def test_a_comparison_that_lost_ownership_is_not_shown(overview, qtbot):
    dialog = overview
    dialog._well(dialog.wells.item(*dialog._well_positions[(dialog.plates.currentData(), "A01")]))
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy()
                    and not dialog._thumbnail_jobs.is_busy()
                    and not dialog._field_thumbnail_jobs.is_busy()
                    and not dialog._montage_jobs.is_busy(), timeout=10000)
    qtbot.wait(100)
    dialog.fields.item(0).setSelected(True)
    dialog.fields.item(1).setSelected(True)
    shown = []
    original = dialog._show_pixels
    dialog._show_pixels = lambda pixels: (shown.append(pixels), original(pixels))
    dialog._compare()
    dialog._comparison_signature = ("another comparison",)
    qtbot.waitUntil(lambda: not dialog._montage_jobs.is_busy(), timeout=10000)
    qtbot.wait(100)
    assert shown == []
    assert dialog._comparison_signature == ("another comparison",)


def test_wizard_action_fit_does_not_reenter_and_measures_styled_buttons(
        source_planes, qtbot):
    wizard = module._OverviewMetadataWizard(str(source_planes), {"custom_regex": PATTERN})
    qtbot.addWidget(wizard)
    wizard.show()

    wizard._fitting_actions = True
    wizard._fit_actions()
    assert wizard._fitting_actions is True
    wizard._fitting_actions = False

    wizard._finished = True
    wizard._fit_actions()
    assert wizard._fitting_actions is False
    wizard._finished = False

    buttons = [QPushButton("Default"), QPushButton("Auto"), QPushButton("Flat")]
    buttons[0].setDefault(True)
    buttons[1].setAutoDefault(True)
    buttons[2].setFlat(True)
    for button in buttons:
        button.setParent(wizard)
        button.show()
    wizard._action_buttons = buttons
    wizard._fit_action_buttons()
    for button in buttons:
        hint = button.sizeHint().expandedTo(button.minimumSizeHint())
        assert button.minimumWidth() > 0 and button.minimumHeight() > 0
        assert button.minimumWidth() >= min(hint.width(), button.minimumWidth())
    wizard.close()


def test_annotation_groups_preserve_definitions_while_renaming_reordering_and_removing(qtbot, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    records = ({'drug': 'A'}, {'drug': 'A'}, {'drug': 'B'})
    builder = module._OverviewGroupBuilder(
        lambda definition: sum(all(record.get(key) == value for key, value in definition.items())
                               for record in records), lambda: {'drug': ['A', 'B']})
    qtbot.addWidget(builder)
    builder.set_columns({'drug': {'A', 'B'}})
    first, second = {'drug': 'A'}, {'drug': 'B'}
    assert builder.add_group(first) and builder.add_group(second)
    assert not builder.add_group(first)
    assert not builder.rename_group(row=-1, name='missing')
    assert builder.rename_group(row=0, name='Control')
    assert not builder.rename_group(row=1, name='Control')
    assert builder.names() == ['Control', '']
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('ignored', False))
    assert not builder.rename_group(row=0)
    assert builder.names() == ['Control', '']
    assert builder.definitions() == [first, second]
    assert '2 images' in builder.groups.item(0).text()
    assert '1 images' in builder.groups.item(1).text()
    builder.groups.setCurrentRow(0)
    builder.move_group(1)
    assert builder.definitions() == [second, first]
    assert builder.names() == ['', 'Control']
    builder.move_group(1)
    assert builder.definitions() == [second, first]
    builder.remove_group()
    assert builder.definitions() == [second]
    builder.groups.setCurrentRow(-1)
    builder.remove_group()
    assert builder.definitions() == [second]
    builder.clear_groups()
    assert builder.definitions() == [] and builder.names() == []
    assert first == {'drug': 'A'} and second == {'drug': 'B'}


def test_annotation_presets_keep_the_held_column_and_handle_a_removed_vary_column(qtbot):
    levels = {}
    builder = module._OverviewGroupBuilder(lambda definition: len(definition), lambda: levels)
    qtbot.addWidget(builder)
    builder.set_columns({'drug': {'A', 'B'}, 'batch': {'1', '2'}})
    builder._pickers['batch'].setCurrentIndex(builder._pickers['batch'].findData('2'))
    builder.preset.setCurrentIndex(builder.preset.findData('vary'))
    builder.vary.setCurrentIndex(builder.vary.findData('drug'))
    assert builder.apply_preset() == 2
    assert builder.definitions() == [{'batch': '2', 'drug': 'A'}, {'batch': '2', 'drug': 'B'}]
    saved = builder.definitions()
    builder.vary.clear()
    assert builder.apply_preset() == 0
    assert builder.definitions() == saved
    levels.update({'drug': ['A', 'B'], 'batch': ['1', '2']})
    builder.preset.setCurrentIndex(builder.preset.findData('combinations'))
    assert builder.apply_preset() == 4
    assert {tuple(sorted(group.items())) for group in builder.definitions()} == {
        (('batch', batch), ('drug', drug)) for batch in ('1', '2') for drug in ('A', 'B')}


def test_real_plate_rename_and_removal_preserve_source_identity_and_reject_bad_regex(qtbot, tmp_path):
    import hashlib

    folders = [tmp_path / name for name in ('plate1', 'plate2')]
    paths = []
    for folder in folders:
        folder.mkdir()
        path = folder / 'p1-A01-f1-t1-z1-c1.tif'
        tifffile.imwrite(path, np.arange(120, dtype=np.uint16).reshape(10, 12))
        paths.append(path)
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    wizard = module._OverviewMetadataWizard('; '.join(map(str, folders)), settings={'custom_regex': PATTERN})
    qtbot.addWidget(wizard)
    wizard.show()
    try:
        qtbot.waitUntil(lambda: len(wizard._detected_plates) == 2
                        and not wizard._runner.is_busy(), timeout=10000)
        first = next(plate for plate in wizard._detected_plates if plate['path'] == str(folders[0]))
        row = next(index for index in range(wizard.plate_table.rowCount())
                   if wizard.plate_table.item(index, 0).data(Qt.UserRole) == first['default_name'])
        name = wizard.plate_table.item(row, 0)
        name.setText('Display name')
        assert wizard._plate_names[first['default_name']] == 'Display name'
        assert name.data(Qt.UserRole) == first['default_name']
        assert first['path'] == str(folders[0])
        regex = wizard.plate_table.item(row, 4)
        regex.setText('[')
        assert wizard._plate_regex_errors[str(folders[0])][0] == '['
        assert str(folders[0]) not in wizard._plate_regex
        regex.setText('')
        qtbot.waitUntil(lambda: not wizard._runner.is_busy()
                        and not wizard._plate_regex_errors, timeout=10000)
        row = next(index for index in range(wizard.plate_table.rowCount())
                   if wizard.plate_table.item(index, 0).toolTip() == str(folders[1]))
        wizard.plate_table.selectRow(row)
        wizard._remove_selected_plates()
        qtbot.waitUntil(lambda: len(wizard._detected_plates) == 1
                        and not wizard._runner.is_busy(), timeout=10000)
        assert module._overview_source_paths(wizard.source.text()) == [str(folders[0])]
        assert wizard._detected_plates[0]['path'] == str(folders[0])
        assert wizard._plate_names[first['default_name']] == 'Display name'
        assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == hashes
    finally:
        wizard.reject()


def test_native_overview_callback_inherits_an_existing_trace_and_preserves_its_result(qtbot):
    import sys
    import threading
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication
    from spacr.core import _watch_portable_keys
    from spacr.qt.job_runner import JobRunner

    expected_trace = threading.gettrace() is not None
    main_thread = QApplication.instance().thread()
    runner = JobRunner(threaded=True, app_key="field_overview", user_visible=False)
    returned = []
    def read_portable_result():
        return (sys.gettrace() is not None,
                QThread.currentThread() != main_thread,
                _watch_portable_keys({r"plateA\field1.tif": {"complete": True}, 4: "numeric"}))
    try:
        assert runner.submit(read_portable_result, returned.append)
        qtbot.waitUntil(lambda: bool(returned) and runner.active_jobs() == 0, timeout=10000)
        assert returned == [(expected_trace, True,
                             {"plateA/field1.tif": {"complete": True}, 4: "numeric"})]
    finally:
        runner.shutdown(timeout_ms=3000)


def test_raw_plane_memory_cache_replacement_and_eviction_obey_byte_and_entry_bounds():
    cache = module._OverviewCache(entries=2, budget_bytes=12)
    first, second = np.arange(4, dtype=np.uint8), np.arange(8, dtype=np.uint8)
    cache.put('first', first)
    cache.put('second', second)
    assert cache.get('first') is first
    cache.put('third', np.ones(4, dtype=np.uint8))
    assert cache.get('second') is None
    assert cache.get('first') is first
    cache.put('first', (np.ones(8, dtype=np.uint8), 'metadata'))
    assert cache.get('first')[0].nbytes == 8 and cache._bytes == 12
    cache.put('oversized', np.zeros(13, dtype=np.uint8))
    assert cache.get('oversized') is None and cache._bytes == 12
    cache.put('metadata', {'source': 'unchanged'})
    assert cache.get('third') is None
    assert cache.get('metadata') == {'source': 'unchanged'}
    assert cache._bytes == 8
    cache.clear()
    assert cache._bytes == 0 and cache.get('first') is None


def test_disk_plane_cache_is_atomic_readonly_and_never_replaces_the_source(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('SPACR_OVERVIEW_DISK_CACHE', '1')
    source = tmp_path / 'source.npy'
    raw = np.arange(24, dtype=np.uint16).reshape(4, 6)
    np.save(source, raw, allow_pickle=False)
    original = source.read_bytes()
    key = ('source-stamp', 1)
    module._overview_store_disk_plane(key, str(source), raw)
    path, decoded = module._overview_disk_plane(key, str(source))
    assert path == str(source) and not decoded.flags.writeable
    np.testing.assert_array_equal(decoded, raw)
    assert module._overview_disk_plane(('new-source-stamp', 2), str(source)) is None
    assert not list(module._overview_disk_name(key).parent.glob('*.tmp'))
    assert source.read_bytes() == original
    monkeypatch.setattr(module.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('cache is read-only')))
    module._overview_store_disk_plane(('failed', 3), str(source), raw)
    assert module._overview_disk_plane(('failed', 3), str(source)) is None
    assert source.read_bytes() == original


def test_disk_cache_pruning_preserves_unremovable_entries_without_touching_sources(tmp_path, monkeypatch):
    from pathlib import Path
    import os

    cache = tmp_path / 'cache'
    shard = cache / 'aa'
    shard.mkdir(parents=True)
    entries = [shard / f'{name}.npy' for name in ('old', 'middle', 'new')]
    for index, path in enumerate(entries):
        path.write_bytes(b'12345678')
        os.utime(path, (index + 1, index + 1))
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == entries[0]:
            raise OSError('another process owns this cache entry')
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', unlink)
    monkeypatch.setattr(module, '_OVERVIEW_DISK_BUDGET', 8)
    module._overview_prune_disk_cache(cache)
    assert entries[0].read_bytes() == b'12345678'
    assert not entries[1].exists() and not entries[2].exists()
    monkeypatch.setattr(Path, 'glob', lambda *args: (_ for _ in ()).throw(OSError('mount lost')))
    module._overview_prune_disk_cache(cache)
    assert entries[0].read_bytes() == b'12345678'


def test_periodic_disk_cache_pruning_and_unstamped_planes_preserve_raw_values(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('SPACR_OVERVIEW_DISK_CACHE', '1')
    monkeypatch.setattr(module, '_OVERVIEW_DISK_WRITES', [0])
    monkeypatch.setattr(module, '_OVERVIEW_DISK_BUDGET', 0)
    raw = np.arange(6, dtype=np.uint16).reshape(2, 3)
    source = tmp_path / 'source.npy'
    np.save(source, raw, allow_pickle=False)
    before = source.read_bytes()
    for index in range(module._OVERVIEW_DISK_PRUNE_EVERY):
        module._overview_store_disk_plane(('pruned', index), str(source), raw)
    assert module._OVERVIEW_DISK_WRITES == [module._OVERVIEW_DISK_PRUNE_EVERY]
    assert module._overview_disk_plane(('pruned', 0), str(source)) is None
    assert module._overview_disk_plane(('pruned', index), str(source)) is None
    assert not list(module._overview_disk_cache_folder().glob('*/*.npy'))
    assert source.read_bytes() == before
    field = SimpleNamespace(files=((2, str(tmp_path / 'missing.tif')),), identity=('missing',))
    calls = []

    def read(given, channels, **axes):
        calls.append((given, channels, axes))
        yield 2, str(tmp_path / 'missing.tif'), raw

    monkeypatch.setattr(module, '_overview_raw_planes', read)
    decoded = list(module._overview_cached_raw_planes(field, (2,), edge=20, time_index=1))
    assert decoded[0][2] is raw
    assert calls == [(field, (2,), {'edge': 20, 'time_index': 1})]
    np.testing.assert_array_equal(raw, np.arange(6, dtype=np.uint16).reshape(2, 3))


@pytest.mark.parametrize('recursive', [False, True])
def test_plate_detection_refuses_missing_paths_and_respects_explicit_recursive_scope(tmp_path, recursive):
    import threading

    source = tmp_path / 'source'
    nested = source / 'one' / 'two'
    nested.mkdir(parents=True)
    tifffile.imwrite(nested / 'p1-A01-f1-t1-z1-c1.tif', np.ones((4, 5), dtype=np.uint16))
    missing = str(tmp_path / 'missing')
    assert module._overview_direct_images(missing) == 0
    assert module._overview_plate_subfolders(missing) == []
    detected = module._overview_detect_plates([missing, str(source)], source_scope='recursive' if recursive else 'folder')
    assert 'not a folder' in detected['messages'][0]
    assert [plate['path'] for plate in detected['plates']] == ([str(source)] if recursive else [])
    if not recursive:
        assert 'No images' in detected['messages'][1]
    cancelled = threading.Event()
    cancelled.set()
    assert module._overview_detect_plates(str(source), cancelled=cancelled) == {'plates': [], 'messages': []}
    with pytest.raises(ValueError, match='existing image folder'):
        module._overview_index_sources(missing, PATTERN)


def test_indexing_multiple_filename_plates_keeps_each_source_and_axis_identity(tmp_path):
    folders = [tmp_path / name for name in ('batch1', 'batch2')]
    for folder in folders:
        folder.mkdir()
        for plate in ('p1', 'p2'):
            tifffile.imwrite(folder / f'{plate}-A01-f1-t1-z1-c1.tif', np.zeros((4, 5), dtype=np.uint16))
    fields, detected = module._overview_index_sources(list(map(str, folders)), PATTERN)
    assert {field.plate for field in fields} == {'batch1 / p1', 'batch1 / p2', 'batch2 / p1', 'batch2 / p2'}
    assert len({module._overview_image_key(field) for field in fields}) == 4
    assert {field.source for field in fields} == set(map(str, folders))
    assert len(detected['plates']) == 2
    unresolved = module._OverviewField('unresolved', '', '', 0, 'unknown', (), unresolved=True)
    kept = module._overview_limited_fields((*fields, unresolved), max_wells=1, max_fields=1)
    assert kept == (*fields, unresolved)


def test_index_cancellation_after_detection_publishes_no_partial_fields(tmp_path, monkeypatch):
    import threading
    from unittest.mock import Mock

    tifffile.imwrite(tmp_path / 'p1-A01-f1-t1-z1-c1.tif', np.ones((4, 5), dtype=np.uint16))
    cancelled = threading.Event()
    detect = module._overview_detect_plates

    def cancel_after_detection(*args, **kwargs):
        result = detect(*args, **kwargs)
        cancelled.set()
        return result

    monkeypatch.setattr(module, '_overview_detect_plates', cancel_after_detection)
    read = Mock(side_effect=AssertionError('cancelled metadata must not decode fields'))
    monkeypatch.setattr(module, '_overview_index', read)
    fields, detected = module._overview_index_sources(str(tmp_path), PATTERN, cancelled=cancelled)
    assert fields == () and len(detected['plates']) == 1
    read.assert_not_called()


def test_comparison_and_representative_workers_publish_real_tables_and_current_fields(overview, qtbot):
    import pandas as pd
    from spacr import tabular

    dialog = overview
    database = module.Path(dialog._metadata_source) / 'measurements' / 'measurements.db'
    records = []
    for field in dialog._fields:
        value = 1. if field.well == 'A01' else 7.
        value += 2. * (int(field.field_id or field.field) - 1)
        records.append({'plate': field.plate, 'well': field.well,
                        'field': str(field.field), 'time_id': str(field.time_id), 'z_id': str(field.z_id),
                        'channel_1_mean': value, 'status': 'ok'})
    tabular.write_database(pd.DataFrame(records), database, 'fov', if_exists='replace', canonicalise=False)
    assert dialog.group_builder.add_group({'well': 'A01'})
    assert dialog.group_builder.add_group({'well': 'B02'})
    dialog.group_builder.rename_group(row=0, name='Control')
    dialog.group_builder.rename_group(row=1, name='Treated')
    dialog.comparison_measurement.setCurrentText('channel_1_mean')
    dialog._compare_groups()
    qtbot.waitUntil(lambda: dialog.comparison_table.rowCount() > 0
                    and dialog._export_jobs.active_jobs() == 0, timeout=10000)
    columns = {dialog.comparison_table.horizontalHeaderItem(index).text(): index
               for index in range(dialog.comparison_table.columnCount())}
    assert float(dialog.comparison_table.item(0, columns['mean_a']).text()) == 2.
    assert float(dialog.comparison_table.item(0, columns['mean_b']).text()) == 8.
    assert float(dialog.comparison_table.item(0, columns['log2_ratio']).text()) == -2.
    assert dialog._comparison_signature[0] == 'groups'
    result_folder = module.Path(dialog._metadata_source) / 'results' / 'overview_comparisons'
    assert list(result_folder.glob('*'))
    dialog._show_representatives(stacked=False)
    qtbot.waitUntil(lambda: dialog._export_jobs.active_jobs() == 0
                    and 'Representative images' in dialog.status.text(), timeout=10000)
    assert 'Control' in dialog.status.text() and 'Treated' in dialog.status.text()
    assert {field.well for field in dialog._fields} == {'A01', 'B02'}


@pytest.mark.parametrize('failure', ['no_measurements', 'unknown_measurement'])
def test_group_workers_refuse_missing_measurements_without_publishing_figures(overview, qtbot, failure):
    import pandas as pd
    from spacr import tabular

    dialog = overview
    dialog.group_builder.add_group({'well': 'A01'})
    dialog.group_builder.add_group({'well': 'B02'})
    if failure == 'unknown_measurement':
        database = module.Path(dialog._metadata_source) / 'measurements' / 'measurements.db'
        frame = pd.DataFrame([{'plate': field.plate, 'well': field.well, 'field': field.field,
                               'channel_1_mean': 2., 'status': 'ok'} for field in dialog._fields])
        tabular.write_database(frame, database, 'fov', if_exists='replace', canonicalise=False)
        dialog.comparison_measurement.setCurrentText('missing_measurement')
        dialog.representative_method.setCurrentIndex(dialog.representative_method.findData('measurement'))
    dialog._compare_groups()
    qtbot.waitUntil(lambda: dialog._export_jobs.active_jobs() == 0, timeout=10000)
    assert ('Export image-level measurements first' if failure == 'no_measurements'
            else 'cannot be compared') in dialog.status.text()
    assert not (module.Path(dialog._metadata_source) / 'results' / 'overview_comparisons').exists()
    dialog._show_representatives(stacked=False)
    qtbot.waitUntil(lambda: dialog._export_jobs.active_jobs() == 0, timeout=10000)
    assert ('Export image-level measurements first' if failure == 'no_measurements'
            else 'No representative image') in dialog.status.text()
    assert dialog.comparison_table.rowCount() == 0


@pytest.fixture
def mixed_plate_wizard(qtbot, tmp_path):
    import hashlib

    root = tmp_path / 'mixed'
    folders = [root, root / 'plate1', root / 'plate2']
    paths = []
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'p1-A01-f1-t1-z1-c1.tif'
        tifffile.imwrite(path, np.arange(120, dtype=np.uint16).reshape(10, 12))
        paths.append(path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    wizard = module._OverviewMetadataWizard(str(root), {'custom_regex': PATTERN})
    qtbot.addWidget(wizard)
    try:
        wizard.show()
        qtbot.waitUntil(lambda: wizard._info is not None and wizard._runner.active_jobs() == 0, timeout=10000)
        yield wizard, folders
        assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before
    finally:
        wizard.reject()
        qtbot.waitUntil(lambda: wizard._runner.active_jobs() == 0, timeout=10000)
        wizard._runner.shutdown(timeout_ms=3000)


def test_mixed_folder_toggle_switches_real_plate_sources_without_rewriting_raw_files(mixed_plate_wizard, qtbot):
    wizard, folders = mixed_plate_wizard
    root = str(folders[0])
    assert [plate['path'] for plate in wizard._detected_plates] == [root]
    toggle = wizard.plate_table.cellWidget(0, 3)
    assert toggle is not None and not toggle.isChecked()
    toggle.setChecked(True)
    qtbot.waitUntil(lambda: len(wizard._detected_plates) == 2
                    and wizard._runner.active_jobs() == 0, timeout=10000)
    assert set(plate['path'] for plate in wizard._detected_plates) == set(map(str, folders[1:]))
    assert wizard._use_subfolders == {root}
    assert {field.source for field in wizard._fields} == set(map(str, folders[1:]))
    wizard._use_plate_subfolders(root, False)
    qtbot.waitUntil(lambda: len(wizard._detected_plates) == 1
                    and wizard._runner.active_jobs() == 0, timeout=10000)
    assert [plate['path'] for plate in wizard._detected_plates] == [root]
    assert wizard._use_subfolders == set()
    assert {field.source for field in wizard._fields} == {root}


def test_plate_regex_apply_refuses_invalid_edits_and_preserves_other_plate_patterns(mixed_plate_wizard, qtbot):
    wizard, folders = mixed_plate_wizard
    wizard._use_plate_subfolders(str(folders[0]), True)
    qtbot.waitUntil(lambda: len(wizard._detected_plates) == 2
                    and wizard._runner.active_jobs() == 0, timeout=10000)
    first, second = [plate['path'] for plate in wizard._detected_plates]
    wizard._edit_plate(second)
    assert wizard._edited_plate == second and wizard._base_regex() == PATTERN
    before = dict(wizard._plate_regex)
    wizard.regex.setText('[')
    assert not wizard._apply_to_plate(second)
    assert not wizard._apply_to_all_plates()
    assert wizard._plate_regex == before
    assert 'Invalid filename regex' in wizard.status.text()
    changed = PATTERN.replace('p\\d+', 'p[0-9]+')
    wizard._show_editor_regex(changed)
    assert wizard._apply_to_plate(second)
    assert wizard._plate_regex[second] == changed
    assert wizard._base_regex() == PATTERN
    wizard._show_editor_regex(changed)
    assert wizard._apply_to_plate(first)
    assert wizard._base_regex() == changed
    assert first not in wizard._plate_regex
    assert all(value != changed for value in wizard._plate_regex.values())
    wizard._show_editor_regex(PATTERN)
    assert wizard._apply_to_all_plates()
    qtbot.waitUntil(lambda: wizard._runner.active_jobs() == 0
                    and wizard._info is not None, timeout=10000)
    assert wizard._plate_regex == {} and wizard._plate_regex_errors == {}
    assert wizard._default_regex == PATTERN
    assert {field.source for field in wizard._fields} == {first, second}


@pytest.mark.parametrize('cancelled', [False, True])
def test_channel_colour_picker_saves_only_an_accepted_real_colour(mixed_plate_wizard, monkeypatch, cancelled):
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import colour_picker

    wizard, _ = mixed_plate_wizard
    item = wizard.channel_table.item(0, 3)
    before = item.text()
    seen = []
    chosen = QColor() if cancelled else QColor('#123456')

    def pick(parent, current, title):
        seen.append((parent, current, title))
        return chosen

    monkeypatch.setattr(colour_picker, 'pick_colour', pick)
    assert wizard._pick_channel_colour(0) == chosen
    assert item.text() == (before if cancelled else '#123456')
    assert seen[0][0] is wizard and 'Channel' in seen[0][2]
    assert not wizard._pick_channel_colour(wizard.channel_table.rowCount()).isValid()


def test_view_state_save_failure_is_reported_without_replacing_the_source(tmp_path):
    import json

    source = tmp_path / 'plate'
    source.mkdir()
    state = {'sources': [str(source)], 'options': {'plate_regex': {str(source): PATTERN}}}
    saved = module._overview_try_save_view_state(str(source), state)
    assert saved.is_file()
    payload = json.loads(saved.read_text(encoding='utf-8'))
    assert payload['sources'] == [str(source)] and payload['options'] == state['options']
    assert 'version' not in state
    blocked = tmp_path / 'not-a-folder'
    blocked.write_bytes(b'original source')
    error = module._overview_try_save_view_state(str(blocked), state)
    assert isinstance(error, str) and error
    assert blocked.read_bytes() == b'original source'


def test_wizard_restores_saved_sources_and_reports_a_real_worker_refusal(qtbot, tmp_path):
    folders = [tmp_path / name for name in ('first', 'second')]
    for folder in folders:
        folder.mkdir()
        tifffile.imwrite(folder / 'p1-A01-f1-t1-z1-c1.tif', np.ones((5, 7), dtype=np.uint16))
    module._overview_save_view_state(str(folders[0]),
        {'sources': list(map(str, folders)), 'options': {'plate_regex': {str(folders[0]): PATTERN}}})
    wizard = module._OverviewMetadataWizard(str(folders[0]), {'custom_regex': PATTERN})
    qtbot.addWidget(wizard)
    wizard.show()
    try:
        qtbot.waitUntil(lambda: len(wizard._detected_plates) == 2
                        and wizard._runner.active_jobs() == 0, timeout=10000)
        assert module._overview_source_paths(wizard.source.text()) == list(map(str, folders))
        assert 'Restored' in wizard.plate_summary.text()
        assert {field.source for field in wizard._fields} == set(map(str, folders))
        wizard.source.setText(str(tmp_path / 'missing'))
        wizard._scan()
        qtbot.waitUntil(lambda: wizard._runner.active_jobs() == 0
                        and 'existing image folder' in wizard.status.text(), timeout=10000)
        assert not wizard.next.isEnabled()
        assert wizard.source.text() == str(tmp_path / 'missing')
        assert all(list(folder.glob('*.tif')) for folder in folders)
    finally:
        wizard.reject()
        qtbot.waitUntil(lambda: wizard._runner.active_jobs() == 0, timeout=10000)
        wizard._runner.shutdown(timeout_ms=3000)
