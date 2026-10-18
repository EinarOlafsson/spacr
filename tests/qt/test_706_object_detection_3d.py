"""Item 706: the 3D switch runs Otsu and Cellpose on the whole open volume."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks
from spacr.qt.screens.make_masks import MakeMasksScreen
from tests.cellpose_api_contract import configured_eval_arguments, eval_arguments
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


def _blobs(shape=(9, 24, 24)):
    """Two bright spheres spanning several slices, and a diagonal voxel chain."""
    z, y, x = np.indices(shape)
    volume = np.full(shape, 10, dtype=np.uint16)
    volume[(z - 4) ** 2 + (y - 6) ** 2 + (x - 6) ** 2 <= 9] = 200
    volume[(z - 4) ** 2 + (y - 17) ** 2 + (x - 17) ** 2 <= 9] = 200
    for step in range(5):
        volume[2 + step, 4 + step, 18 - step] = 200
    return volume


def test_threshold_labels_a_volume_with_26_connectivity():
    volume = _blobs()
    labels = engine._otsu_instances(volume, min_area=0, smoothing=0.0, fill_holes=False,
                                    split_touching=False, correction=1.0)
    assert labels.shape == volume.shape
    assert int(labels.max()) == 3
    chain = {int(labels[2 + step, 4 + step, 18 - step]) for step in range(5)}
    assert len(chain) == 1 and 0 not in chain
    sphere = labels == labels[4, 6, 6]
    assert np.flatnonzero(sphere.any(axis=(1, 2))).size > 1
    planewise = sum(ndimage.label(plane > 100)[1] for plane in volume)
    assert planewise > 3


def test_connected_instances_keeps_2d_numbering_and_min_area():
    binary = np.zeros((6, 6), bool)
    binary[0, 0] = binary[1, 1] = True
    binary[4:6, 4:6] = True
    binary[0, 5] = True
    labels = engine.connected_instances(binary, min_area=2)
    assert labels[0, 0] == labels[1, 1] == 1 and labels[4, 4] == 2 and labels[0, 5] == 0


def test_replace_labels_is_one_undo_step():
    session = engine.VolumeMaskSession(np.zeros((3, 4, 5), np.uint16),
                                       np.zeros((3, 4, 5), np.uint16))
    labels = np.zeros((3, 4, 5), np.uint16)
    labels[0:2, 1:3, 1:3] = 4
    assert session.replace_labels(labels) == 8
    session.undo()
    assert not session.mask.any()
    with pytest.raises(ValueError):
        session.replace_labels(np.zeros((2, 4, 5), np.uint16))


@pytest.fixture
def screen(qtbot, tmp_path):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert not widget._detect_3d.isEnabled()
    path = tmp_path / "volume.tif"
    tifffile.imwrite(path, _blobs(), metadata={"axes": "ZYX"}, photometric="minisblack")
    assert widget.open_volume(str(path)) is widget._mask_stage
    yield widget
    widget._mask_stage.dirty = False
    widget.close()


def _rows(box):
    return {box.itemData(index): box.model().item(index).isEnabled()
            for index in range(box.count())}


def test_switch_gates_2d_only_methods_and_restores_them(screen):
    from spacr.qt import cpu_modes

    assert screen._detect_3d.isEnabled()
    screen._cp_model.addItem("Cellpose-DINO", "cellpose_dino:/models/x.pt")
    before = _rows(screen._mag_mode)
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(cpu_modes.PROPAGATE))
    screen._detect_3d.setChecked(True)
    assert screen._mag_mode.currentData() == "otsu"
    rows = _rows(screen._mag_mode)
    assert rows["otsu"] and rows["li"] and rows[cpu_modes.MULTIOTSU] and rows["sauvola"]
    assert not rows[cpu_modes.PROPAGATE] and not rows[cpu_modes.PUNCTA]
    index = screen._mag_mode.findData(cpu_modes.PROPAGATE)
    assert "2D" in screen._mag_mode.itemData(index, make_masks.Qt.ToolTipRole)
    dino = screen._cp_model.findData("cellpose_dino:/models/x.pt")
    assert not screen._cp_model.model().item(dino).isEnabled()
    assert not screen._otsu_local.isEnabled()
    screen._detect_3d.setChecked(False)
    assert _rows(screen._mag_mode) == before
    assert screen._cp_model.model().item(dino).isEnabled()
    assert "2D" not in str(screen._mag_mode.itemData(index, make_masks.Qt.ToolTipRole))


def test_switch_is_off_and_disabled_without_a_volume(screen):
    screen._detect_3d.setChecked(True)
    screen._leave_volume()
    assert not screen._detect_3d.isEnabled()
    assert not screen._detect_3d.isChecked()
    assert not screen._detect_3d_on()
    assert "3D image" in screen._detect_3d.toolTip()


def test_3d_otsu_off_the_gui_thread_with_undo(screen, qtbot, monkeypatch):
    threads = []
    original = make_masks._detect_volume_snapshot

    def spy(request, models):
        threads.append(threading.get_ident())
        return original(request, models)

    monkeypatch.setattr(make_masks, "_detect_volume_snapshot", spy)
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData("otsu"))
    screen._min_area.setValue(0)
    screen._detect_3d.setChecked(True)
    session = screen._mask_stage.session
    screen._btn_otsu.click()
    qtbot.waitUntil(lambda: screen._detection_request is None, timeout=20000)
    assert threads and threads[0] != threading.get_ident()
    labels = session.mask
    assert int(np.count_nonzero(np.unique(labels))) >= 2
    sphere = labels == labels[4, 6, 6]
    assert labels[4, 6, 6] and np.flatnonzero(sphere.any(axis=(1, 2))).size > 1
    assert labels[4, 6, 6] != labels[4, 17, 17]
    assert screen._mask_stage.dirty
    screen._on_undo()
    assert not session.mask.any()
    screen._on_redo()
    assert session.mask[4, 6, 6]
    screen._mask_stage.dirty = False


class _FakeCellpose:
    def __init__(self):
        self.calls = []

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
             normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
             anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=256,
             compute_masks=True, progress=None):
        bound = dict(locals())
        seen = eval_arguments(bound)
        configured = configured_eval_arguments(bound)
        check_cellpose_eval_call(x, channel_axis, z_axis=z_axis, do_3D=do_3D,
                                 stitch_threshold=stitch_threshold)
        self.calls.append((np.asarray(x).shape, seen, configured))
        labels = np.zeros(np.asarray(x).shape, np.int32)
        labels[2:7, 3:9, 3:9] = 1
        return labels, None, None


def test_cellpose_runs_do_3d_and_stitching(screen, qtbot):
    fake = _FakeCellpose()
    screen._cp_loaded["cpsam"] = fake
    index = screen._cp_model.findData("cpsam")
    if index < 0:
        screen._cp_model.addItem("cpsam", "cpsam")
        index = screen._cp_model.count() - 1
    screen._cp_model.setCurrentIndex(index)
    screen._detect_3d.setChecked(True)
    screen._btn_cellpose.click()
    qtbot.waitUntil(lambda: screen._detection_request is None, timeout=10000)
    shape, kwargs, configured = fake.calls[-1]
    assert shape == (9, 24, 24)
    assert kwargs["do_3D"] is True and kwargs["z_axis"] == 0 and kwargs["channel_axis"] is None
    assert "stitch_threshold" not in configured
    session = screen._mask_stage.session
    assert int(session.mask.max()) == 1 and session.mask[4, 5, 5] == 1
    screen._combine_mode.setCurrentIndex(screen._combine_mode.findData("replace"))
    screen._cp_stitch.setValue(0.3)
    screen._btn_cellpose.click()
    qtbot.waitUntil(lambda: screen._detection_request is None, timeout=10000)
    shape, kwargs, _configured = fake.calls[-1]
    assert kwargs["do_3D"] is False and kwargs["stitch_threshold"] == pytest.approx(0.3)
    screen._mask_stage.dirty = False


def test_cellpose_refuses_a_2d_only_model_in_3d(screen):
    screen._cp_model.addItem("Cellpose-DINO", "cellpose_dino:/models/x.pt")
    screen._cp_model.setCurrentIndex(screen._cp_model.count() - 1)
    screen._detect_3d.setChecked(True)
    screen._btn_cellpose.click()
    assert screen._detection_request is None
    assert "2D fields only" in screen._status_label.text()
    assert make_masks._model_supports_volume("cpsam")
    assert not make_masks._model_supports_volume("cellpose_dino:/models/x.pt")


def _stardist_folder(folder, ndim=3):
    folder.mkdir()
    (folder / 'config.json').write_text(json.dumps({
        'n_dim': ndim, 'axes': 'ZYXC' if ndim == 3 else 'YXC',
        'n_channel_in': 1,
    }))
    return 'stardist:' + str(folder)


def test_stardist_volume_crosses_the_real_loader_and_worker_transport(tmp_path, monkeypatch):
    from spacr import _segmentation_backends as backends

    name = _stardist_folder(tmp_path / 'native-model')
    environment = tmp_path / 'backend-environment'
    environment.mkdir()
    state = SimpleNamespace(state=backends._INSTALLED, in_process=False,
                            ready=True, env=str(environment))
    monkeypatch.setattr(backends, '_backend_state', lambda *_a, **_k: state)
    monkeypatch.setattr(make_masks, '_BACKEND_MODELS', {})
    monkeypatch.setattr(make_masks, '_BACKEND_MODEL_ENVS', {})
    loads, requests = [], []
    real_load = backends._load_backend

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return real_load(*args, **kwargs)

    class Worker:
        def request(self, operation, **kwargs):
            assert operation == 'segment'
            requests.append(kwargs)
            assert len(kwargs['inputs']) == 1
            field = np.load(kwargs['inputs'][0], allow_pickle=False)
            assert field.shape == (4, 8, 10)
            labels = np.zeros(field.shape, np.int32)
            labels[1:3, 2:5, 3:7] = 119
            path = Path(kwargs['outputs']) / 'labels.npy'
            np.save(path, labels, allow_pickle=False)
            return {'outputs': [{'mask': str(path), 'flows': [None] * 4}]}

    monkeypatch.setattr(backends, '_load_backend', load)
    monkeypatch.setattr(backends, '_worker_for', lambda *_args: Worker())
    plane_model = object()
    models = {name: plane_model}
    request = dict(kind='cellpose', model=name, image=np.zeros((8, 4, 10)),
                   axes='YZX', invert=False, percentiles=None,
                   parameters={'anisotropy': 2.5, 'stitch_threshold': 0.3})

    first = make_masks._detect_volume_snapshot(request, models)
    second = make_masks._detect_volume_snapshot(request, models)

    assert first.shape == (8, 4, 10) and first[2, 1, 3] == 119
    np.testing.assert_array_equal(second, first)
    assert len(loads) == 1 and loads[0][0] == ('stardist',)
    assert loads[0][1]['z_plan'].mode == 'volumetric'
    assert len(requests) == 2
    assert all(row['params']['do_3D'] is True for row in requests)
    assert all(row['params']['z_axis'] == 0 for row in requests)
    assert all(row['params']['anisotropy'] == 2.5 for row in requests)
    assert all('stitch_threshold' not in row['params'] for row in requests)
    assert models[name] is plane_model
    assert request['parameters']['stitch_threshold'] == 0.3


def test_stardist_choices_and_stitch_control_follow_model_dimensionality(screen, tmp_path):
    native = _stardist_folder(tmp_path / 'native')
    plane = _stardist_folder(tmp_path / 'plane', 2)
    for value in (native, plane, 'stardist:2D_versatile_fluo'):
        screen._cp_model.addItem(value, value)
    screen._cp_model.setCurrentIndex(screen._cp_model.findData(native))
    screen._cp_stitch.setValue(0.3)
    screen._detect_3d.setChecked(True)

    rows = _rows(screen._cp_model)
    assert rows[native] and not rows[plane] and not rows['stardist:2D_versatile_fluo']
    assert not screen._cp_stitch.isEnabled()
    assert screen._cp_stitch.value() == pytest.approx(0.3)
    cellpose = screen._cp_model.findData('cpsam')
    if cellpose < 0:
        screen._cp_model.addItem('cpsam', 'cpsam')
        cellpose = screen._cp_model.count() - 1
    screen._cp_model.setCurrentIndex(cellpose)
    assert screen._cp_stitch.isEnabled()
    assert screen._cp_stitch.value() == pytest.approx(0.3)
    screen._detect_3d.setChecked(False)
    assert all(_rows(screen._cp_model)[name] for name in (native, plane))


def test_stardist_plane_request_refuses_before_loading_or_changing_input(tmp_path, monkeypatch):
    name = _stardist_folder(tmp_path / 'plane', 2)
    image = np.zeros((4, 8, 10))
    original = image.copy()
    monkeypatch.setattr(make_masks, '_backend_model',
                        lambda *_a, **_k: pytest.fail('unsupported model was loaded'))
    request = dict(kind='cellpose', model=name, image=image, axes='ZYX',
                   invert=False, percentiles=None, parameters={})
    with pytest.raises(ValueError, match='2D fields only'):
        make_masks._detect_volume_snapshot(request, {})
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('payload', [b'{}', b'not JSON', b'x' * (1024 * 1024 + 1)])
def test_unverifiable_stardist_metadata_is_not_offered_as_a_volume(tmp_path, payload):
    (tmp_path / 'config.json').write_bytes(payload)
    assert not make_masks._model_supports_volume('stardist:' + str(tmp_path))


def test_translated_backend_failure_keeps_worker_context_and_the_mask(screen, monkeypatch):
    from spacr.qt import i18n

    source = 'Cellpose 3 needs finite volume intensities'
    monkeypatch.setenv(i18n.ENV_LANGUAGE, 'fr')
    translated = i18n.tr(source)
    assert translated != source
    notices = []
    monkeypatch.setattr(screen, '_warn', lambda title, text: notices.append((title, text)))
    session = screen._mask_stage.session
    request = {'session': session, 'mask': session.mask.copy()}

    screen._take_volume_detection(request, None, RuntimeError('worker: ' + source))

    assert notices == [(i18n.tr('Object detection failed'), 'worker: ' + translated)]
    np.testing.assert_array_equal(session.mask, request['mask'])
