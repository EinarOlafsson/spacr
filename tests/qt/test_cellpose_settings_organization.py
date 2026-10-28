"""Cellpose forms preserve their native settings, model selection and run handoff."""
import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDialog, QMainWindow, QMessageBox, QWidget

from spacr.qt.screens import app_screen, settings_model, train_cellpose


@pytest.mark.parametrize('key', ['train_cellpose', 'cellpose_masks', 'cellpose_all'])
def test_organized_sections_nest_controls_once_and_preserve_values(qtbot, key):
    owner = QWidget()
    qtbot.addWidget(owner)
    model = settings_model.SettingsWidgets(key, parent=owner)
    sections = model.build_sections()
    if key == 'cellpose_masks':
        assert model.collect()['save'] is True
    paths = {section.title: section.path for root in sections for section in root.walk()}
    if key == 'train_cellpose':
        assert paths['Validation Data'] == ('Training Data', 'Validation Data')
        assert paths['Sampling & Augmentation'] == ('Training Schedule', 'Sampling & Augmentation')
        assert paths['Image Preprocessing'] == ('Input & Channels', 'Image Preprocessing')
        changes = {'src': '/training', 'mask_src': '/labels', 'test_src': '/validation',
                   'test_mask_src': '/validation-labels', 'model_name': 'retained',
                   'n_epochs': 2, 'normalize': False}
    else:
        assert paths['Image Geometry'] == ('Image Preprocessing', 'Image Geometry')
        assert paths['Background & Denoising'] == ('Image Preprocessing', 'Background & Denoising')
        changes = {'src': '/images', 'channels': [0, 1], 'flow_threshold': .2,
                   'save': True, 'resize': False}
    assert 'src' in model._widgets
    rows = [widget for root in sections for section in root.walk() for _label, widget in section.own_rows]
    assert len(rows) == len({id(widget) for widget in rows})
    for name, value in changes.items():
        assert model.set_value_for_key(name, value)
    assert all(model.collect()[name] == value for name, value in changes.items())


def test_workbench_model_zoo_writes_only_active_native_checkpoint_field(qtbot, monkeypatch, tmp_path):
    from spacr.model_zoo import _model_input_kinds
    from spacr.qt.widgets import model_zoo_picker

    workbench = train_cellpose.CellposeWorkbenchScreen()
    qtbot.addWidget(workbench)
    picks = []
    state = dict(path=str(tmp_path / 'training.pth'), accept=True)

    class Picker:
        def __init__(self, kinds, parent):
            picks.append((kinds, parent.app_key))

        def exec(self):
            return QDialog.Accepted if state['accept'] else QDialog.Rejected

        def chosen_path(self):
            return state['path']

    monkeypatch.setattr(model_zoo_picker, 'ModelZooPicker', Picker)
    workbench.train_screen.apply_settings_dict({'src': '/train', 'model_name': 'my_checkpoint'})
    workbench.apply_screen.apply_settings_dict({'src': '/apply'})
    qtbot.mouseClick(workbench._zoo_button, Qt.LeftButton)
    assert workbench.train_screen._settings_model.collect()['base_model'] == state['path']
    assert workbench.train_screen._settings_model.collect()['model_name'] == 'my_checkpoint'
    assert workbench.apply_screen._settings_model.collect()['custom_model'] is None
    workbench._tabs.setCurrentIndex(1)
    state['path'] = str(tmp_path / 'apply.pth')
    qtbot.mouseClick(workbench._zoo_button, Qt.LeftButton)
    assert workbench.apply_screen._settings_model.collect()['custom_model'] == state['path']
    assert workbench.train_screen._settings_model.collect()['src'] == '/train'
    assert workbench.apply_screen._settings_model.collect()['src'] == '/apply'
    before = workbench.current_settings()
    state['accept'] = False
    qtbot.mouseClick(workbench._zoo_button, Qt.LeftButton)
    assert workbench.current_settings() == before
    assert all(kinds == ('cellpose',) for kinds, _parent in picks)
    assert _model_input_kinds('train_cellpose', {'base_model': 'cpsam'}) == {'base_model': ('cellpose',)}
    assert app_screen.AppScreen._takes_a_cellpose_checkpoint('base_model')
    workbench.close()


def test_stock_comparison_model_zoo_is_browsing_not_a_fake_model_setting(qtbot):
    owner = QMainWindow()
    qtbot.addWidget(owner)
    opened = []
    owner._open_easy_named_module = opened.append
    screen = app_screen.AppScreen('cellpose_all', owner)
    owner.setCentralWidget(screen)
    qtbot.mouseClick(screen._cellpose_zoo_button, Qt.LeftButton)
    assert opened == ['model_zoo']
    assert 'src' in screen._settings_model._widgets
    assert not {'custom_model', 'model_name'} & set(screen._settings_model._widgets)
    screen.close()


def test_mask_folder_uses_short_label_and_always_saves_confirmed_masks(qtbot, monkeypatch, tmp_path):
    from spacr.qt.screens.make_masks import MASK_FOLDER_KEY, MakeMasksScreen, fold_description

    tifffile.imwrite(tmp_path / 'field.tif', np.ones((8, 9), dtype=np.uint16))
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    try:
        screen._open_folder(str(tmp_path))
        workbench = screen.folded_screen(MASK_FOLDER_KEY)
        workbench.apply_screen.apply_settings_dict({'save': False})
        started = []
        confirmations = []
        monkeypatch.setattr(screen, '_confirm', lambda title, _message: confirmations.append(title) or True)
        monkeypatch.setattr(screen, '_start_folded_run',
                            lambda page: started.append((page.app_key, page._settings_model.collect())))
        assert screen.mask_whole_folder()
        assert fold_description(MASK_FOLDER_KEY)[0] == 'Mask folder'
        assert confirmations == ['Mask folder']
        assert len(started) == 1 and started[0][0] == 'cellpose_masks'
        assert started[0][1]['save'] is True and started[0][1]['src'] == str(tmp_path)
    finally:
        screen._magnifier.close()
        screen.close_folded()
        screen.close()


@pytest.mark.parametrize('names', [[], ['field.png'], ['field.tiff'], ['field.TIF'],
                                  ['field.tif', 'another.png']])
@pytest.mark.parametrize('key', ['cellpose_masks', 'cellpose_all'])
def test_native_folder_runner_refuses_empty_or_unprocessed_formats_before_start(
        qtbot, monkeypatch, tmp_path, names, key):
    for name in names:
        (tmp_path / name).write_bytes(b'input')
    screen = app_screen.AppScreen(key)
    qtbot.addWidget(screen)
    screen.apply_settings_dict({'src': str(tmp_path)})
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda _owner, _title, message: messages.append(message))
    monkeypatch.setattr(screen, '_wait_for_the_entry_import',
                        lambda _override: pytest.fail('Refused input must not start an import or run'))
    screen._on_run()
    assert messages and '.tif' in messages[0]
    assert screen._btn_run.isEnabled()
    screen.close()


class _Worker(QObject):
    line_ready = Signal(str)
    error = Signal(str)
    figure_ready = Signal(object)
    result_ready = Signal(object)
    finished = Signal()


class _Thread(QObject):
    finished = Signal()

    def start(self):
        self.started = True

    def isRunning(self):
        return False


@pytest.mark.parametrize('key', ['train_cellpose', 'cellpose_masks', 'cellpose_all'])
def test_native_run_submits_organized_values_to_own_entry_boundary(qtbot, monkeypatch, tmp_path, key):
    tifffile.imwrite(tmp_path / 'field.tif', np.ones((8, 9), dtype=np.uint16))
    screen = app_screen.AppScreen(key)
    qtbot.addWidget(screen)
    changes = {'src': str(tmp_path), 'batch_size': 1}
    changes.update({'model_name': 'my_run', 'n_epochs': 1} if key == 'train_cellpose'
                   else {'save': True, 'CP_prob': .1, 'flow_threshold': .2})
    screen.apply_settings_dict(changes)
    entry = object()
    submitted = []

    def make_thread(function, settings):
        submitted.append((function, dict(settings)))
        return _Thread(screen), _Worker(screen)

    monkeypatch.setattr(screen, '_wait_for_the_entry_import', lambda _override: False)
    monkeypatch.setattr(screen, '_claim_the_project', lambda _settings: True)
    monkeypatch.setattr(screen, '_confirm_ram_guard', lambda _settings: True)
    monkeypatch.setattr(screen, '_record_run_in_runs_tab', lambda *_args: None)
    monkeypatch.setattr(screen, '_announce_the_fit', lambda *_args: None)
    monkeypatch.setattr(screen, '_watch_for_refusals', lambda: None)
    monkeypatch.setattr(app_screen, 'resolve_pipeline_entry', lambda requested: entry if requested == key else None)
    monkeypatch.setattr(app_screen, 'make_thread', make_thread)
    screen._on_run()
    assert len(submitted) == 1 and submitted[0][0] is entry
    assert {name: submitted[0][1][name] for name in changes} == changes
    assert screen._thread.started
    screen.close()
