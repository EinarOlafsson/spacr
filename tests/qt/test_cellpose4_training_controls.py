"""The Train form exposes paired sources and round-trips Cellpose 4 defaults."""
from PySide6.QtWidgets import QFileDialog, QLineEdit
from spacr.qt.screens.train_cellpose import CellposeWorkbenchScreen


def test_training_form_sources_defaults_and_native_options(qtbot, monkeypatch, tmp_path):
    screen = CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    model = screen.train_screen._settings_model
    assert {'src','mask_src','base_model','test_src','test_mask_src','save_path'} <= set(model._widgets)
    assert not {'augment','target_size','diameter','model_type','from_scratch'} & set(model._widgets)
    values = model.collect()
    assert values['model_name'] == 'new_model'
    assert values['channels'] is None
    assert values['percentiles'] == [1, 99]
    assert [values[k] for k in ('learning_rate','weight_decay','n_epochs','batch_size')] == [1e-5,.1,100,1]
    for key in ('src','mask_src','test_src','test_mask_src','save_path'):
        widget = model._widgets[key]
        assert isinstance(widget, QLineEdit)
        monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *a: str(tmp_path))
        widget.actions()[0].trigger()
        assert model.collect()[key] == str(tmp_path)
    model.set_value_for_key('model_name','my_fine_tuned_model')
    assert model.collect()['model_name'] == 'my_fine_tuned_model'


def test_checkpoint_button_reads_custom_output_folder(qtbot, tmp_path):
    screen = CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    folder = tmp_path/'output'/'models'; folder.mkdir(parents=True)
    path = folder/'test_cpsam_e100.CP_model'; path.write_bytes(b'weights')
    model = screen.train_screen._settings_model
    for key, value in dict(src=str(tmp_path/'images'), save_path=str(folder.parent), model_name='test').items():
        model.set_value_for_key(key,value)
    assert screen.trained_checkpoint() == str(path)
