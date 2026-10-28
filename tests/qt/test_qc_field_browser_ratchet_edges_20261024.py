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
