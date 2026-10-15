"""Cache moves update new model widgets and only the matching remembered path."""
import pytest
from PySide6.QtCore import QSettings

from spacr.qt import preferences
from spacr.qt.widgets import model_zoo_picker


@pytest.fixture
def picker_store(tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / 'picker.ini'), QSettings.IniFormat)
    monkeypatch.setattr(model_zoo_picker, '_store', lambda: store)
    return store


@pytest.fixture
def storage_controller(monkeypatch):
    controller = preferences._StoragePage.__new__(preferences._StoragePage)
    controller._dialog = None
    controller.refresh = lambda: None
    controller._refuse_while_busy = lambda title: False
    monkeypatch.setattr(preferences, '_confirm_storage_action', lambda *args: True)
    monkeypatch.setattr(preferences, '_show_storage_result', lambda *args: None)
    return controller


def finish_move(controller, source, target, key='models', success=True):
    controller._selected = lambda: {'key': key, 'path': str(source), 'relocatable': True}
    callbacks = []
    controller._background = lambda work, done: callbacks.append(done)
    controller.relocate(str(target.parent))
    callbacks[0]((success, str(target)))


def test_repeated_model_moves_follow_only_the_old_cache(picker_store, storage_controller, tmp_path):
    first, second, third = (tmp_path / name for name in ('first', 'second', 'third'))
    picker_store.setValue(model_zoo_picker._DIR_SETTING, str(first))
    for source, target in ((first, second), (second, third)):
        finish_move(storage_controller, source, target)
        assert model_zoo_picker.remembered_model_dir() == str(target)


@pytest.mark.parametrize('success,key', [(False, 'models'), (True, 'news'), (True, 'models')])
def test_unrelated_custom_destination_survives_moves(picker_store, storage_controller, tmp_path, success, key):
    custom = tmp_path / 'custom'
    picker_store.setValue(model_zoo_picker._DIR_SETTING, str(custom))
    finish_move(storage_controller, tmp_path / 'source', tmp_path / 'target',
                key=key, success=success)
    assert model_zoo_picker.remembered_model_dir() == str(custom)


def test_dispatch_captures_cache_identity_and_source_before_selection_changes(
        picker_store, storage_controller, tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'parent' / 'spacr-models'
    row = {'key': 'models', 'path': str(source), 'relocatable': True}
    storage_controller._selected = lambda: row
    dispatched = []
    storage_controller._background = lambda work, done: dispatched.append((work, done))
    picker_store.setValue(model_zoo_picker._DIR_SETTING, str(source))
    storage_controller.relocate(str(target.parent))
    row.update(key='news', path=str(tmp_path / 'news'))
    dispatched[0][1]((True, str(target)))
    assert model_zoo_picker.remembered_model_dir() == str(target)


def test_new_custom_choice_while_worker_runs_is_preserved(picker_store, storage_controller, tmp_path):
    source = tmp_path / 'source'
    custom = tmp_path / 'new custom'
    row = {'key': 'models', 'path': str(source), 'relocatable': True}
    storage_controller._selected = lambda: row
    dispatched = []
    storage_controller._background = lambda work, done: dispatched.append(done)
    picker_store.setValue(model_zoo_picker._DIR_SETTING, str(source))
    storage_controller.relocate(str(tmp_path / 'parent'))
    picker_store.setValue(model_zoo_picker._DIR_SETTING, str(custom))
    dispatched[0]((True, str(tmp_path / 'parent' / 'spacr-models')))
    assert model_zoo_picker.remembered_model_dir() == str(custom)


def test_widgets_opened_after_environment_change_use_new_cache(
        picker_store, monkeypatch, tmp_path, qtbot):
    from spacr import model_zoo
    from spacr.qt.screens.model_zoo import ModelZooScreen

    monkeypatch.setattr(model_zoo, 'catalogue', lambda *args, **kwargs: [])
    monkeypatch.setattr(model_zoo, 'community_entries', lambda: [])
    monkeypatch.setattr(model_zoo_picker.ModelZooPicker, '_warm_the_community_catalogue',
                        lambda self: None)
    old, new = tmp_path / 'old cache', tmp_path / 'new cache'
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(old))
    first = model_zoo_picker.ModelZooPicker()
    qtbot.addWidget(first)
    assert first.folder_edit.text() == str(old)
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(new))
    second = model_zoo_picker.ModelZooPicker()
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(second)
    qtbot.addWidget(screen)
    assert second.folder_edit.text() == str(new)
    assert screen._dest_edit.text() == str(new)
    assert first.folder_edit.text() == str(old)
    custom = tmp_path / 'explicit custom'
    picker_store.setValue(model_zoo_picker._DIR_SETTING, str(custom))
    third = model_zoo_picker.ModelZooPicker()
    qtbot.addWidget(third)
    assert third.folder_edit.text() == str(custom)


def test_detector_preview_finds_relocated_cache_without_fetching(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from spacr.qt.widgets import plaque_preview

    cache = tmp_path / 'relocated'
    cache.mkdir()
    checkpoint = cache / 'detector.pt'
    checkpoint.write_bytes(b'checkpoint fixture')
    entry = SimpleNamespace(key='detector', name=checkpoint.name, path='')
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(cache))
    monkeypatch.setattr(plaque_preview, '_catalogue', lambda: [entry])
    path, note, selected = plaque_preview.resolve_detector('detector')
    assert path == str(checkpoint)
    assert note == '' and selected is entry


def test_preview_download_targets_active_cache_without_starting_inference(
        monkeypatch, tmp_path, qtbot):
    from types import SimpleNamespace
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QPushButton
    from spacr.qt import model_install
    from spacr.qt.widgets import plaque_preview

    created = []

    class Download(QObject):
        progressed = Signal(int, int)
        finished = Signal(bool, str)

        def __init__(self, entry, folder):
            super().__init__()
            self.folder = folder
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

    cache = tmp_path / 'relocated'
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(cache))
    monkeypatch.setattr(model_install, 'CheckpointDownload', Download)
    button = QPushButton()
    qtbot.addWidget(button)
    panel = SimpleNamespace(_missing_entry=object(), _download_btn=button,
                            _on_download_progress=lambda *args: None,
                            _on_download_finished=lambda *args: None)
    plaque_preview.PlaquePreviewPanel._start_download(panel)
    assert len(created) == 1 and created[0].started
    assert created[0].folder == str(cache)
    assert cache.is_dir()
