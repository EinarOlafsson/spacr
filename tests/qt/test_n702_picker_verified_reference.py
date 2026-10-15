"""Actual picker selections preserve verified identity through form export."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from spacr import model_zoo as zoo
from spacr.cli import load_settings_file
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets import model_zoo_picker as picker_module


_REAL_DIALOG_EXEC = QDialog.exec


@pytest.fixture
def selected_model(qtbot, tmp_path, monkeypatch):
    content = b'controlled trusted checkpoint bytes'
    path = tmp_path / 'renamed-cache-v7.pt'
    path.write_bytes(content)
    raw = {'key': 'controlled_cellpose', 'name': 'publisher_weights',
           'kind': 'cellpose', 'sha256': hashlib.sha256(content).hexdigest(),
           'repo_id': 'controlled/fixture', 'repo_type': 'model', 'version': '001'}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', (raw,))
    entry = replace(zoo._entry_from_mapping(raw), version='17')
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [])
    for method in ('_warm_the_community_catalogue', '_warm_bioimageio', '_probe_backends'):
        monkeypatch.setattr(picker_module.ModelZooPicker, method, lambda self: None)
    monkeypatch.setattr(picker_module, 'remembered_model_dir', lambda: str(tmp_path))
    monkeypatch.setattr(picker_module, 'remembered_sources', lambda: ('spaCR', 'cellposeSAM'))
    monkeypatch.setattr(picker_module, '_cellpose3_heading_turns_on', lambda chosen: False)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: pytest.fail(str(a[2])))
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('Unexpected model download'))
    return path, raw, entry


def test_actual_download_completion_keeps_its_renamed_path(selected_model, qtbot):
    path, raw, entry = selected_model
    foreign = path.parent / entry.name
    foreign.write_bytes(b'unrelated concurrent checkpoint at publisher basename')
    original = foreign.read_bytes()
    picker = picker_module.ModelZooPicker(kinds=('cellpose',))
    qtbot.addWidget(picker)
    picker._download_entry = entry
    picker._worker = SimpleNamespace(_entry=entry)
    picker._thread = None
    picker._on_download_finished(str(path))
    assert picker._local_path_on_disk(entry) == str(path)
    assert foreign.read_bytes() == original
    assert path.read_bytes() == b'controlled trusted checkpoint bytes'


def test_real_picker_path_selection_exports_exact_verified_pin(selected_model, qtbot,
                                                              monkeypatch, tmp_path):
    path, raw, entry = selected_model
    screen = AppScreen('analyze_plaques')
    qtbot.addWidget(screen)
    model = screen._settings_model
    field = model._widgets['plaque_model']
    real_dialog = picker_module.ModelZooPicker
    dialogs = []

    def factory(*args, **kwargs):
        dialog = real_dialog(*args, **kwargs)
        dialog._rebuild([replace(entry, path=str(path))])
        dialog.table.selectRow(0)
        dialogs.append(dialog)
        dialog.exec = lambda: _REAL_DIALOG_EXEC(dialog)
        deadline = QTimer(dialog)
        deadline.setSingleShot(True)
        deadline.timeout.connect(dialog.reject)
        deadline.start(5000)
        QTimer.singleShot(0, dialog._accept_selected)
        return dialog

    monkeypatch.setattr(picker_module, 'ModelZooPicker', factory)
    screen._choose_a_model_for(field, 'plaque_model')
    assert dialogs and dialogs[0].chosen_path() == str(path)
    assert model.collect()['plaque_model'] == str(path)
    destination = tmp_path / 'selected.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(destination), ''))
    screen._on_export_settings()
    exported = load_settings_file(str(destination))
    assert exported['plaque_model'] == str(path)
    record = exported['model_metadata']['settings']['plaque_model']
    assert record == zoo._model_reference_from_record(str(path), raw)
    assert record['version'] == '001' and record['version'] != entry.version
    assert record['filename'] == raw['name']


def local_picker(selected_model, qtbot):
    path, raw, entry = selected_model
    picker = picker_module.ModelZooPicker(kinds=('cellpose',))
    qtbot.addWidget(picker)
    picker._rebuild([replace(entry, path=str(path))])
    picker.table.selectRow(0)
    return picker


def test_real_verifier_keeps_gui_heartbeat_and_shows_checking(selected_model, qtbot,
                                                            monkeypatch, qapp):
    import threading
    from PySide6.QtCore import QThread
    from spacr.qt.i18n import tr

    picker = local_picker(selected_model, qtbot)
    entered, release = threading.Event(), threading.Event()
    actual = zoo._verified_model_reference
    threads = []

    def delayed(*args, **kwargs):
        threads.append(QThread.currentThread())
        entered.set()
        assert release.wait(3)
        return actual(*args, **kwargs)

    monkeypatch.setattr(zoo, '_verified_model_reference', delayed)
    ticks = []
    timer = QTimer(picker)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        picker._accept_selected()
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(lambda: len(ticks) >= 3)
        assert threads[0] is not qapp.thread()
        assert picker.chosen_path() is None
        assert picker.status.text() == tr('Checking the selected model file…')
        assert not picker.use_button.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: picker.chosen_path() is not None)
    assert dict(picker._chosen_reference.record)['key'] == selected_model[1]['key']
    timer.stop()


@pytest.mark.parametrize('action', ['reject', 'close'])
def test_closing_a_blocked_check_never_accepts_or_destroys_a_running_thread(
        selected_model, qtbot, monkeypatch, action):
    import threading
    from spacr.qt.bridge import thread_has_stopped

    picker = local_picker(selected_model, qtbot)
    entered, release = threading.Event(), threading.Event()
    actual = zoo._verified_model_reference
    choices = []
    picker.model_chosen.connect(choices.append)

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return actual(*args, **kwargs)

    monkeypatch.setattr(zoo, '_verified_model_reference', delayed)
    picker.show()
    picker._accept_selected()
    qtbot.waitUntil(entered.is_set)
    thread = picker._reference_thread
    try:
        getattr(picker, action)()
        assert picker._reference_thread is None
        assert picker.chosen_path() is None and picker._chosen_reference is None
    finally:
        release.set()
    qtbot.waitUntil(lambda: thread_has_stopped(thread))
    assert picker.result() == QDialog.Rejected
    assert choices == []


def test_verified_result_for_an_old_selection_cannot_replace_a_custom_choice(
        selected_model, qtbot, monkeypatch, tmp_path):
    import threading
    from spacr.qt.bridge import thread_has_stopped

    path, raw, entry = selected_model
    picker = local_picker(selected_model, qtbot)
    ready, release = threading.Event(), threading.Event()
    actual = zoo._verified_model_reference

    def delayed(selected, *args, **kwargs):
        result = actual(selected, *args, **kwargs)
        if selected.key == entry.key:
            ready.set()
            assert release.wait(3)
        return result

    monkeypatch.setattr(zoo, '_verified_model_reference', delayed)
    picker._accept_selected()
    qtbot.waitUntil(ready.is_set)
    thread = picker._reference_thread
    custom = tmp_path / 'user-choice.pt'
    custom.write_bytes(b'custom selected bytes')
    other = replace(entry, key='private-custom', source='local', path=str(custom))
    try:
        picker._rebuild([other])
        picker.table.selectRow(0)
        assert picker.selected_entry().path == str(custom)
    finally:
        release.set()
    qtbot.waitUntil(lambda: thread_has_stopped(thread))
    assert picker.chosen_path() is None
    picker._accept_selected()
    qtbot.waitUntil(lambda: picker.chosen_path() is not None)
    assert picker.chosen_path() == str(custom)
    assert picker._chosen_reference is None
    assert path.read_bytes() == b'controlled trusted checkpoint bytes'


def test_typed_cancelled_verification_has_no_error_or_accepted_path(selected_model, qtbot,
                                                                  monkeypatch):
    picker = local_picker(selected_model, qtbot)
    choices = []
    picker.model_chosen.connect(choices.append)

    def cancelled(*args, **kwargs):
        raise zoo.DownloadCancelled()

    monkeypatch.setattr(zoo, '_verified_model_reference', cancelled)
    picker._accept_selected()
    qtbot.waitUntil(lambda: picker._reference_thread is None)
    assert picker.status.text() == ''
    assert picker.chosen_path() is None and choices == []
    assert picker.use_button.isEnabled()


@pytest.mark.parametrize('selection', ['stock', 'custom', 'unverified'])
def test_legacy_path_and_signal_contract_stays_unpinned(selected_model, qtbot,
                                                       selection, tmp_path):
    path, raw, entry = selected_model
    picker = local_picker(selected_model, qtbot)
    if selection == 'stock':
        chosen = picker.STOCK_MODEL
        expected = 'cpsam'
    else:
        chosen = replace(entry, key='private-model', source='local', path=str(path),
                         sha256='' if selection == 'unverified' else entry.sha256)
        expected = str(path)
    picker._rebuild([chosen])
    assert not picker.table.isRowHidden(0)
    picker.table.selectRow(0)
    choices = []
    picker.model_chosen.connect(choices.append)
    picker._accept_selected()
    if selection in ("stock", "unverified"):
        assert picker.chosen_path() == expected
    qtbot.waitUntil(lambda: picker.chosen_path() is not None)
    assert picker.chosen_path() == expected
    assert choices == [expected] and isinstance(choices[0], str)
    assert picker._chosen_reference is None


@pytest.mark.parametrize('before_pin', [False, True])
def test_actual_post_write_refusal_restores_raw_field_and_all_prior_pins(
        selected_model, qtbot, monkeypatch, tmp_path, before_pin):
    import copy

    path, raw, entry = selected_model
    screen = AppScreen('analyze_plaques')
    qtbot.addWidget(screen)
    model = screen._settings_model
    old_path = tmp_path / 'previous.pt'
    old_path.write_bytes(path.read_bytes())
    model.set_value_for_key('plaque_model', str(old_path))
    model.set_value_for_key('figure_detector', 'previous-detector.pt')
    previous_pins = {'figure_detector': {
        'requested_value': 'previous-detector.pt', 'key': 'prior-private-detector',
        'version': None, 'version_status': 'unspecified', 'filename': 'detector.pt',
        'kind': 'detector', 'sha256': '2' * 64}}
    if before_pin:
        previous_pins['plaque_model'] = zoo._model_reference_from_record(str(old_path), raw)
    model.set_hidden_value('model_metadata', {'schema_version': 1, 'settings': previous_pins})
    before = copy.deepcopy(model.collect())
    field = model._widgets['plaque_model']
    raw_before = field.text()
    real_dialog = picker_module.ModelZooPicker
    original_write = model.set_value_for_key
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[2]))

    def write(key, value):
        result = original_write(key, value)
        if value == str(path):
            path.write_bytes(b'changed after the accepted verification')
        return result

    def factory(*args, **kwargs):
        dialog = real_dialog(*args, **kwargs)
        dialog._rebuild([replace(entry, path=str(path))])
        dialog.table.selectRow(0)
        dialog.exec = lambda: _REAL_DIALOG_EXEC(dialog)
        deadline = QTimer(dialog)
        deadline.setSingleShot(True)
        deadline.timeout.connect(dialog.reject)
        deadline.start(5000)
        QTimer.singleShot(0, dialog._accept_selected)
        return dialog

    monkeypatch.setattr(model, 'set_value_for_key', write)
    monkeypatch.setattr(picker_module, 'ModelZooPicker', factory)
    screen._choose_a_model_for(field, 'plaque_model')
    assert len(messages) == 1
    assert field.text() == raw_before
    assert model.collect() == before


def test_real_download_worker_lands_on_gui_and_keeps_its_actual_path(
        selected_model, qtbot, monkeypatch, qapp):
    from PySide6.QtCore import QThread

    path, raw, entry = selected_model
    picker = picker_module.ModelZooPicker(kinds=('cellpose',))
    qtbot.addWidget(picker)
    picker._rebuild([entry])
    picker.table.selectRow(0)
    gui_landing = []
    original_refresh = picker.refresh

    def refresh():
        gui_landing.append(QThread.currentThread() is qapp.thread())
        original_refresh()

    def fetch(selected, folder, **kwargs):
        assert QThread.currentThread() is not qapp.thread()
        (path.parent / selected.name).write_bytes(b'concurrent foreign basename')
        return str(path)

    monkeypatch.setattr(picker, 'refresh', refresh)
    monkeypatch.setattr(zoo, 'fetch', fetch)
    picker._download_selected()
    qtbot.waitUntil(lambda: picker._worker is None)
    assert gui_landing == [True]
    assert picker._local_path_on_disk(entry) == str(path)
    assert (path.parent / entry.name).read_bytes() == b'concurrent foreign basename'


def test_old_worker_completion_cannot_be_attached_to_the_new_download(
        selected_model, qtbot, monkeypatch):
    path, raw, entry = selected_model
    picker = local_picker(selected_model, qtbot)
    old_worker = picker_module._DownloadWorker(entry, str(path.parent))
    other = replace(entry, key='another-record')
    new_worker = picker_module._DownloadWorker(other, str(path.parent))
    picker._worker = new_worker
    picker._download_entry = other
    refreshes = []
    monkeypatch.setattr(picker, 'refresh', lambda: refreshes.append(1))
    old_worker.finished.connect(picker._on_download_finished)
    old_worker.finished.emit(str(path))
    assert picker._worker is new_worker
    assert picker._completed_paths == {} and refreshes == []
    picker._worker = None


@pytest.mark.parametrize('trusted', [True, False])
def test_missing_form_key_refuses_a_trusted_pin_but_preserves_custom_field_fallback(
        selected_model, qtbot, monkeypatch, trusted):
    from PySide6.QtWidgets import QLineEdit

    path, raw, entry = selected_model
    screen = AppScreen('analyze_plaques')
    qtbot.addWidget(screen)
    field = QLineEdit('original external path')
    qtbot.addWidget(field)
    before = screen._settings_model.collect()
    real_dialog = picker_module.ModelZooPicker
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[2]))

    def factory(*args, **kwargs):
        dialog = real_dialog(*args, **kwargs)
        selected = entry if trusted else replace(entry, key='private-external')
        dialog._rebuild([replace(selected, path=str(path))])
        dialog.table.selectRow(0)
        dialog.exec = lambda: _REAL_DIALOG_EXEC(dialog)
        deadline = QTimer(dialog)
        deadline.setSingleShot(True)
        deadline.timeout.connect(dialog.reject)
        deadline.start(5000)
        QTimer.singleShot(0, dialog._accept_selected)
        return dialog

    monkeypatch.setattr(picker_module, 'ModelZooPicker', factory)
    screen._choose_a_model_for(field, 'not_a_declared_setting')
    assert screen._settings_model.collect() == before
    assert field.text() == ('original external path' if trusted else str(path))
    assert len(messages) == int(trusted)


def install_picker_choice(monkeypatch, entry, path):
    real_dialog = picker_module.ModelZooPicker

    def factory(*args, **kwargs):
        dialog = real_dialog(*args, **kwargs)
        dialog._rebuild([replace(entry, path=str(path))])
        dialog.table.selectRow(0)
        dialog.exec = lambda: _REAL_DIALOG_EXEC(dialog)
        deadline = QTimer(dialog)
        deadline.setSingleShot(True)
        deadline.timeout.connect(dialog.reject)
        deadline.start(5000)
        QTimer.singleShot(0, dialog._accept_selected)
        return dialog

    monkeypatch.setattr(picker_module, 'ModelZooPicker', factory)


def test_false_form_setter_keeps_original_field_fallback_and_attaches_the_pin(
        selected_model, qtbot, monkeypatch):
    path, raw, entry = selected_model
    screen = AppScreen('analyze_plaques')
    qtbot.addWidget(screen)
    model = screen._settings_model
    field = model._widgets['plaque_model']
    monkeypatch.setattr(model, 'set_value_for_key', lambda *a: False)
    install_picker_choice(monkeypatch, entry, path)
    screen._choose_a_model_for(field, 'plaque_model')
    values = model.collect()
    assert values['plaque_model'] == str(path)
    assert values['model_metadata']['settings']['plaque_model'] == zoo._model_reference_from_record(str(path), raw)


@pytest.mark.parametrize('failure', ['false_after_write', 'ignored_write'])
def test_failed_pin_commit_is_refused_and_restores_previously_absent_or_present_pins(
        selected_model, qtbot, monkeypatch, tmp_path, failure):
    import copy

    path, raw, entry = selected_model
    screen = AppScreen('analyze_plaques')
    qtbot.addWidget(screen)
    model = screen._settings_model
    if failure == 'ignored_write':
        old = tmp_path / 'previous-local.pt'
        old.write_bytes(path.read_bytes())
        model.set_value_for_key('plaque_model', str(old))
        model.set_hidden_value('model_metadata', {'schema_version': 1, 'settings': {
            'plaque_model': zoo._model_reference_from_record(str(old), raw)}})
    before = copy.deepcopy(model.collect())
    field = model._widgets['plaque_model']
    raw_before = field.text()
    original_store = model.set_hidden_value
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[2]))

    def failed_store(key, value):
        if failure == 'false_after_write':
            assert original_store(key, value)
            return False
        return True

    monkeypatch.setattr(model, 'set_hidden_value', failed_store)
    install_picker_choice(monkeypatch, entry, path)
    screen._choose_a_model_for(field, 'plaque_model')
    assert len(messages) == 1
    assert field.text() == raw_before
    assert model.collect() == before


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
def test_actual_worker_start_status_uses_its_reviewed_locale(
        selected_model, qtbot, monkeypatch, qapp, language):
    import json
    from pathlib import Path
    import threading
    from PySide6.QtCore import QThread
    from spacr.qt import i18n

    source = 'Checking the selected model file…'
    review_path = (Path(__file__).parents[2] / 'docs/i18n/reviewed/runtime' / language /
                   '2026-10-15-n702-picker-file-check.json')
    review = json.loads(review_path.read_text())
    assert review['language'] == language
    record, = [row for row in review['records']
               if row['source'] == source and row['table'] == 'ui']
    expected = record['translation']
    assert expected != source
    assert record['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
    picker = local_picker(selected_model, qtbot)
    entered, release = threading.Event(), threading.Event()
    actual = zoo._verified_model_reference
    threads = []

    def delayed(*args, **kwargs):
        threads.append(QThread.currentThread())
        entered.set()
        assert release.wait(3)
        return actual(*args, **kwargs)

    monkeypatch.setattr(zoo, '_verified_model_reference', delayed)
    try:
        picker._accept_selected()
        qtbot.waitUntil(entered.is_set)
        assert threads[0] is not qapp.thread()
        assert picker.chosen_path() is None
        assert picker.status.text() == expected
        assert not picker.use_button.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: picker.chosen_path() is not None)
    assert picker.chosen_path() == str(selected_model[0])
    assert dict(picker._chosen_reference.record)['sha256'] == selected_model[1]['sha256']


@pytest.mark.parametrize('obj', ['cell', 'nucleus', 'pathogen'])
def test_actual_mask_grid_selection_exports_its_verified_pin(
        selected_model, qtbot, monkeypatch, tmp_path, obj):
    path, raw, entry = selected_model
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    model = screen._settings_model
    grid = screen._object_grid
    assert grid is not None
    key = f'{obj}_model_name'
    if obj != 'cell':
        assert model.set_value_for_key(f'{obj}_channel', 1)
    assert obj in grid.objects()
    real_dialog = picker_module.ModelZooPicker
    dialogs = []

    def factory(*args, **kwargs):
        dialog = real_dialog(*args, **kwargs)
        dialog._rebuild([replace(entry, path=str(path))])
        dialog.table.selectRow(0)
        dialogs.append(dialog)
        dialog.exec = lambda: _REAL_DIALOG_EXEC(dialog)
        deadline = QTimer(dialog)
        deadline.setSingleShot(True)
        deadline.timeout.connect(dialog.reject)
        deadline.start(5000)
        QTimer.singleShot(0, dialog._accept_selected)
        return dialog

    monkeypatch.setattr(picker_module, 'ModelZooPicker', factory)
    assert grid.choose_model_for(obj)
    assert dialogs and dialogs[0].chosen_path() == str(path)
    assert dialogs[0]._chosen_reference is not None
    assert grid.settings()[key] == str(path)
    assert model.collect()[key] == str(path)
    destination = tmp_path / 'mask-selected.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(destination), ''))
    screen._on_export_settings()
    exported = load_settings_file(str(destination))
    assert exported[key] == str(path)
    assert exported['model_metadata']['settings'][key] == zoo._model_reference_from_record(
        str(path), raw)
    before = model.collect()
    assert not grid.choose_model_for(obj)
    assert model.collect() == before


@pytest.mark.parametrize('action', ['cancel', 'changed_after_verification'])
@pytest.mark.parametrize('prior_pin', [False, True])
def test_grid_cancel_or_refusal_preserves_fields_and_all_previous_pins(
        selected_model, qtbot, monkeypatch, tmp_path, action, prior_pin):
    from copy import deepcopy

    path, raw, entry = selected_model
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    model = screen._settings_model
    grid = screen._object_grid
    old = tmp_path / 'prior-grid.pt'
    old.write_bytes(path.read_bytes())
    model.set_value_for_key('cell_model_name', str(old))
    model.set_value_for_key('nucleus_model_name', str(old))
    pins = {'nucleus_model_name': zoo._model_reference_from_record(str(old), raw)}
    if prior_pin:
        pins['cell_model_name'] = zoo._model_reference_from_record(str(old), raw)
    assert model.set_hidden_value('model_metadata', {'schema_version': 1, 'settings': pins})
    before = deepcopy(model.collect())
    grid_before = deepcopy(grid.settings())
    real_dialog = picker_module.ModelZooPicker
    original_write = model.set_value_for_key
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[2]))

    def write(key, value):
        result = original_write(key, value)
        if key == 'cell_model_name' and value == str(path):
            path.write_bytes(b'changed after accepted grid selection')
        return result

    def factory(*args, **kwargs):
        dialog = real_dialog(*args, **kwargs)
        dialog._rebuild([replace(entry, path=str(path))])
        dialog.table.selectRow(0)
        dialog.exec = lambda: _REAL_DIALOG_EXEC(dialog)
        deadline = QTimer(dialog)
        deadline.setSingleShot(True)
        deadline.timeout.connect(dialog.reject)
        deadline.start(5000)
        QTimer.singleShot(0, dialog.reject if action == 'cancel' else dialog._accept_selected)
        return dialog

    monkeypatch.setattr(model, 'set_value_for_key', write)
    monkeypatch.setattr(picker_module, 'ModelZooPicker', factory)
    assert not grid.choose_model_for('cell')
    assert len(messages) == (1 if action == 'changed_after_verification' else 0)
    assert model.collect() == before
    assert grid.settings() == grid_before
    assert old.read_bytes() == b'controlled trusted checkpoint bytes'


def test_grid_missing_widget_keeps_existing_state_without_opening_picker(
        selected_model, qtbot, monkeypatch):
    from copy import deepcopy

    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    grid = screen._object_grid
    model = screen._settings_model
    before = deepcopy(grid.settings())
    monkeypatch.delitem(model._widgets, 'cell_model_name')
    monkeypatch.setattr(picker_module, 'ModelZooPicker', lambda *a, **k: pytest.fail(
        'A missing owner widget must not open a chooser'))
    assert not grid.choose_model_for('cell')
    assert grid.settings() == before


@pytest.mark.parametrize('selected', [None, 'standalone-new.pt', 'standalone-old.pt'])
def test_standalone_grid_keeps_its_original_path_only_selection_contract(
        qtbot, monkeypatch, selected):
    from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid

    grid = ObjectSettingsGrid()
    qtbot.addWidget(grid)
    grid.set_settings({'cell_channel': 0, 'cell_model_name': 'standalone-old.pt',
                       'nucleus_channel': 1, 'nucleus_model_name': 'nucleus-old.pt'})
    assert 'model_name' in grid.questions()
    monkeypatch.setattr(picker_module, 'choose_model', lambda *a, **k: selected)
    assert grid.choose_model_for('cell') is (selected == 'standalone-new.pt')
    assert grid.settings()['cell_model_name'] == (selected or 'standalone-old.pt')


def test_actual_preview_chooser_propagation_exports_its_verified_pin(
        selected_model, qtbot, monkeypatch, tmp_path):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    path, raw, entry = selected_model
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    panel = LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.apply_settings(screen._settings_model.collect())
    panel.set_propagate_callback(screen._propagate_live_settings)
    panel._verified_propagate_cb = screen._propagate_verified_live_settings
    assert panel._primary_object() == 'cell'
    real_dialog = picker_module.ModelZooPicker
    dialogs = []

    def factory(*args, **kwargs):
        dialog = real_dialog(*args, **kwargs)
        dialog._rebuild([replace(entry, path=str(path))])
        dialog.table.selectRow(0)
        dialogs.append(dialog)
        dialog.exec = lambda: _REAL_DIALOG_EXEC(dialog)
        deadline = QTimer(dialog)
        deadline.setSingleShot(True)
        deadline.timeout.connect(dialog.reject)
        deadline.start(5000)
        QTimer.singleShot(0, dialog._accept_selected)
        return dialog

    monkeypatch.setattr(picker_module, 'ModelZooPicker', factory)
    panel._choose_a_preview_model()
    assert dialogs and dialogs[0]._chosen_reference is not None
    assert panel._model_box.currentText() == str(path)
    propagated = panel.settings_for_propagation()
    assert propagated['cell_model_name'] == str(path)
    panel.propagate_settings()
    assert screen._settings_model.collect()['cell_model_name'] == str(path)
    destination = tmp_path / 'preview-selected.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(destination), ''))
    screen._on_export_settings()
    exported = load_settings_file(str(destination))
    assert exported['cell_model_name'] == str(path)
    assert exported['model_metadata']['settings']['cell_model_name'] == zoo._model_reference_from_record(
        str(path), raw)
    before_pin = screen._settings_model.collect()['model_metadata']
    panel.apply_settings(screen._settings_model.collect())
    assert panel._selected_model_reference is None
    panel._diameter.setValue(panel._diameter.value() + 3)
    panel._flow.setValue(panel._flow.value() + 0.1)
    panel.propagate_settings()
    assert screen._settings_model.collect()['model_metadata'] == before_pin


@pytest.fixture
def preview_owner(selected_model, qtbot, monkeypatch):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    path, raw, entry = selected_model
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    panel = LivePreviewPanel(threaded=False, module='mask')
    qtbot.addWidget(panel)
    panel.apply_settings(screen._settings_model.collect())
    panel.set_propagate_callback(screen._propagate_live_settings)
    panel._verified_propagate_cb = screen._propagate_verified_live_settings
    real_dialog = picker_module.ModelZooPicker
    state = {'cancel': False, 'dialogs': []}

    def factory(*args, **kwargs):
        dialog = real_dialog(*args, **kwargs)
        dialog._rebuild([replace(entry, path=str(path))])
        dialog.table.selectRow(0)
        state['dialogs'].append(dialog)
        dialog.exec = lambda: _REAL_DIALOG_EXEC(dialog)
        deadline = QTimer(dialog)
        deadline.setSingleShot(True)
        deadline.timeout.connect(dialog.reject)
        deadline.start(5000)
        QTimer.singleShot(0, dialog.reject if state['cancel'] else dialog._accept_selected)
        return dialog

    monkeypatch.setattr(picker_module, 'ModelZooPicker', factory)
    return screen, panel, state


def test_preview_off_does_not_mutate_form_and_manual_away_back_loses_capsule(
        selected_model, preview_owner):
    import json
    from copy import deepcopy

    screen, panel, state = preview_owner
    before = deepcopy(screen._settings_model.collect())
    panel._choose_a_preview_model()
    assert panel._selected_model_reference is not None
    assert screen._settings_model.collect() == before
    assert type(panel.settings_for_propagation()) is dict
    json.dumps(panel.settings_for_propagation())
    path = str(selected_model[0])
    panel._model_box.setCurrentIndex(panel._model_box.findText('cpsam'))
    assert panel._selected_model_reference is None
    panel._model_box.setCurrentIndex(panel._model_box.findText(path))
    assert panel._selected_model_reference is None
    panel.propagate_settings()
    assert not (screen._settings_model.collect().get('model_metadata') or {}).get('settings')


def test_preview_cancel_preserves_private_selection_and_all_form_values(preview_owner):
    from copy import deepcopy

    screen, panel, state = preview_owner
    panel._choose_a_preview_model()
    reference = panel._selected_model_reference
    path = panel._model_box.currentText()
    before = deepcopy(screen._settings_model.collect())
    state['cancel'] = True
    panel._choose_a_preview_model()
    assert panel._selected_model_reference is reference
    assert panel._model_box.currentText() == path
    assert screen._settings_model.collect() == before


@pytest.mark.parametrize('failure', ['before_write', 'after_write', 'hidden_false', 'hidden_ignored'])
@pytest.mark.parametrize('prior_selected_pin', [False, True])
def test_preview_refusal_restores_raw_model_fields_and_all_prior_pins(
        selected_model, preview_owner, monkeypatch, tmp_path, failure, prior_selected_pin):
    from copy import deepcopy

    path, raw, entry = selected_model
    screen, panel, state = preview_owner
    model = screen._settings_model
    old = tmp_path / 'previous-preview.pt'
    old.write_bytes(path.read_bytes())
    model.set_value_for_key('cell_model_name', str(old))
    model.set_value_for_key('nucleus_model_name', str(old))
    pins = {'nucleus_model_name': zoo._model_reference_from_record(str(old), raw)}
    if prior_selected_pin:
        pins['cell_model_name'] = zoo._model_reference_from_record(str(old), raw)
    assert model.set_hidden_value('model_metadata', {'schema_version': 1, 'settings': pins})
    before = deepcopy(model.collect())
    raw_before = model._widgets['cell_model_name'].text()
    panel.apply_settings(before)
    panel._choose_a_preview_model()
    assert panel._selected_model_reference is not None
    panel._diameter.setValue(panel._diameter.value() + 7)
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[2]))
    if failure == 'before_write':
        path.write_bytes(b'changed before actual propagation')
    elif failure == 'after_write':
        original = model.set_value_for_key

        def write(key, value):
            accepted = original(key, value)
            if key == 'cell_model_name' and value == str(path):
                path.write_bytes(b'changed after actual model setting write')
            return accepted

        monkeypatch.setattr(model, 'set_value_for_key', write)
    else:
        original = model.set_hidden_value

        def hidden(key, value):
            if key == 'model_metadata':
                return failure == 'hidden_ignored'
            return original(key, value)

        monkeypatch.setattr(model, 'set_hidden_value', hidden)
    panel.propagate_settings()
    assert len(messages) == 1
    assert model._widgets['cell_model_name'].text() == raw_before
    assert model.collect() == before
    assert old.read_bytes() == b'controlled trusted checkpoint bytes'


def test_preview_immediate_propagate_signal_already_has_verified_capsule(
        selected_model, preview_owner):
    screen, panel, state = preview_owner
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    dialog._propagate_btn.setChecked(True)
    panel._choose_a_preview_model()
    current = screen._settings_model.collect()
    assert current['cell_model_name'] == str(selected_model[0])
    assert current['model_metadata']['settings']['cell_model_name'] == zoo._model_reference_from_record(
        str(selected_model[0]), selected_model[1])
    dialog.close()


def test_ordinary_preview_callback_keeps_its_one_dict_argument(selected_model, preview_owner):
    screen, panel, state = preview_owner
    panel._verified_propagate_cb = None
    received = []
    panel.set_propagate_callback(received.append)
    panel._choose_a_preview_model()
    panel.propagate_settings()
    assert len(received) == 1 and type(received[0]) is dict
    assert received[0]['cell_model_name'] == str(selected_model[0])
    assert 'model_metadata' not in received[0]
    assert all(not isinstance(value, zoo._VerifiedModelReference) for value in received[0].values())


def test_actual_mask_preview_card_has_the_explicit_verified_owner(selected_model, qtbot):
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    panel = screen._build_live_preview_panel()
    assert panel._verified_propagate_cb.__self__ is screen
    assert panel._verified_propagate_cb.__func__ is AppScreen._propagate_verified_live_settings
    assert panel._propagate_cb.__self__ is screen


@pytest.mark.parametrize('failure', [None, 'native_after_write', 'hidden_false'])
def test_actual_cellpose_registry_preview_exports_and_keeps_pins_on_repeated_tuning(
        selected_model, qtbot, monkeypatch, tmp_path, failure):
    from spacr.qt import preview_registry

    path, raw, entry = selected_model
    screen = AppScreen('cellpose_masks')
    qtbot.addWidget(screen)
    host = preview_registry.install(screen)
    assert host is not None
    panel = host.panel
    assert panel._verified_propagate_cb.__self__ is screen
    model = screen._settings_model
    if failure is not None:
        old = tmp_path / 'previous-registry.pt'
        old.write_bytes(path.read_bytes())
        assert model.set_value_for_key('custom_model', str(old))
        assert model.set_hidden_value('model_metadata', {
            'schema_version': 1,
            'settings': {'custom_model': zoo._model_reference_from_record(str(old), raw)}})
    panel.apply_settings(model.collect())
    real_dialog = picker_module.ModelZooPicker

    def factory(*args, **kwargs):
        dialog = real_dialog(*args, **kwargs)
        dialog._rebuild([replace(entry, path=str(path))])
        dialog.table.selectRow(0)
        dialog.exec = lambda: _REAL_DIALOG_EXEC(dialog)
        deadline = QTimer(dialog)
        deadline.setSingleShot(True)
        deadline.timeout.connect(dialog.reject)
        deadline.start(5000)
        QTimer.singleShot(0, dialog._accept_selected)
        return dialog

    monkeypatch.setattr(picker_module, 'ModelZooPicker', factory)
    panel._choose_a_preview_model()
    if failure is not None:
        from copy import deepcopy
        from PySide6.QtWidgets import QComboBox

        previous = deepcopy(model.collect())
        field = model._widgets['model_name']
        assert isinstance(field, QComboBox) and not field.isEditable()
        items = [(field.itemText(i), field.itemData(i)) for i in range(field.count())]
        index, text = field.currentIndex(), field.currentText()
        messages = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[2]))
        if failure == 'native_after_write':
            original = model.set_value_for_key

            def write(key, value):
                accepted = original(key, value)
                if key == 'model_name' and value == str(path):
                    path.write_bytes(b'changed after actual noneditable combo commit')
                return accepted

            monkeypatch.setattr(model, 'set_value_for_key', write)
        else:
            original = model.set_hidden_value
            monkeypatch.setattr(model, 'set_hidden_value', lambda key, value:
                                False if key == 'model_metadata' else original(key, value))
        panel.propagate_settings()
        assert len(messages) == 1
        assert model.collect() == previous
        assert field.currentIndex() == index and field.currentText() == text
        assert [(field.itemText(i), field.itemData(i)) for i in range(field.count())] == items
        assert not field.isEditable()
        return
    panel.propagate_settings()
    before = screen._settings_model.collect()
    assert before['model_name'] == str(path)
    assert before['model_metadata']['settings']['model_name'] == zoo._model_reference_from_record(
        str(path), raw)
    destination = tmp_path / 'registry-preview.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(destination), ''))
    screen._on_export_settings()
    assert load_settings_file(str(destination))['model_metadata'] == before['model_metadata']
    panel.apply_settings(before)
    assert panel._selected_model_reference is None
    panel._diameter.setValue(panel._diameter.value() + 3)
    panel.propagate_settings()
    assert screen._settings_model.collect()['model_metadata'] == before['model_metadata']
    host.on_propagate({'model_name': 'different-private-checkpoint.pt'})
    assert 'model_name' not in screen._settings_model.collect()['model_metadata']['settings']
