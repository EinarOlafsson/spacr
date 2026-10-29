"""The Train form exposes paired sources and round-trips Cellpose 4 defaults."""
import pytest

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


@pytest.mark.parametrize("normalize,percentiles,resample,invert", [
    (True, [2, 99], False, False),
    (True, None, True, True),
    (False, [2, 99], True, True),
])
def test_apply_preview_uses_the_batch_selected_pixels_and_eval_parameters(
        qtbot, monkeypatch, tmp_path, normalize, percentiles, resample, invert):
    import copy

    import numpy as np
    import tifffile

    from spacr import accelerator, spacr_cellpose
    from spacr.qt.widgets import live_preview

    source = tmp_path / "images"
    source.mkdir()
    raw = (np.arange(24 * 32, dtype=np.uint16).reshape(24, 32) * 7 + 13)
    raw[0, 0] = 65535
    path = source / "field.tif"
    tifffile.imwrite(path, raw)
    original = path.read_bytes()
    checkpoint = tmp_path / "selected-model"
    checkpoint.write_bytes(b"model boundary fixture")
    settings = dict(src=str(source), model_name="cpsam", custom_model=str(checkpoint),
                    channels=[0, 0], normalize=normalize, percentiles=percentiles,
                    resample=resample, rescale=.8, invert=invert, diameter=17.5,
                    flow_threshold=.65, CP_prob=-.25, remove_background=False,
                    save=False, plot=False, verbose=False, fill_in=False)
    calls = []

    class Model:
        def eval(self, x, *, normalize=True, resample=True, rescale=None,
                 channel_axis=None, diameter=None, flow_threshold=.4,
                 cellprob_threshold=0., progress=None):
            calls.append((np.asarray(x).copy(), dict(normalize=normalize,
                resample=resample, rescale=rescale, channel_axis=channel_axis,
                diameter=diameter, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold)))
            return np.zeros(np.asarray(x).shape[:2], np.int32), [], None

    model = Model()
    loaded = []
    monkeypatch.setattr(spacr_cellpose.cp_models, "CellposeModel",
                        lambda pretrained_model, **kwargs: loaded.append(pretrained_model) or model)
    monkeypatch.setattr(accelerator, "cellpose_gpu", lambda: False)
    monkeypatch.setattr(accelerator, "cellpose_kwargs", lambda: {"gpu": False})
    monkeypatch.setattr(live_preview, "preview_cellpose_model", lambda name: model)
    preview = live_preview.LivePreviewPanel(threaded=False, module="cellpose_masks")
    qtbot.addWidget(preview)
    preview.apply_settings(copy.deepcopy(settings))
    preview._image = raw.copy()
    preview._path_full = str(path)
    request = preview._build_request()
    assert request.module == "cellpose_masks" and request.object_types == ("cell",)
    assert request.model == str(checkpoint)

    live_preview._segment_multi(request)
    spacr_cellpose.identify_masks_finetune(copy.deepcopy(settings))

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    assert calls[0][0].dtype == calls[1][0].dtype
    assert calls[0][1] == calls[1][1]
    assert calls[0][1]["normalize"] is (normalize and percentiles is None)
    assert calls[0][1]["resample"] is resample
    assert calls[0][1]["flow_threshold"] == .65 and calls[0][1]["cellprob_threshold"] == -.25
    assert loaded == [str(checkpoint)]
    np.testing.assert_array_equal(preview._image, raw)
    assert path.read_bytes() == original
    preview.shutdown()
