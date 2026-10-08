"""The actual Make Masks mode detects, saves and binds native centre values."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from spacr import tabular
from spacr.qt import cpu_modes, detect_chain
from spacr.qt.screens import make_masks as mm
from tests.test_center_pixel_puncta import field


@pytest.mark.parametrize("failure", ["bundle", "channels", "nonfinite", "changed"])
def test_native_puncta_refuses_invalid_or_changing_sources_without_caching(qapp, tmp_path, monkeypatch, failure):
    path = tmp_path / 'source.tif'
    image = np.ones((12, 16), dtype=np.float32)
    if failure == 'channels':
        image = np.ones((12, 16, 3), dtype=np.float32)
    elif failure == 'nonfinite':
        image[2, 3] = np.nan
    tifffile.imwrite(path, image)
    display = np.full((12, 16), 25, dtype=np.uint16)
    screen = SimpleNamespace(_image_files=[path.name], _current_index=0,
                             _folder=str(tmp_path), _canvas=SimpleNamespace(image=display))
    message = {'bundle': 'original single-channel', 'channels': 'two-dimensional',
               'nonfinite': 'non-finite', 'changed': 'changed while being read'}[failure]
    if failure == 'bundle':
        monkeypatch.setattr(mm.engine, 'is_seg_bundle', lambda _: True)
    elif failure == 'changed':
        read = mm.engine.read_image

        def changing_read(filename):
            result = read(filename)
            with path.open('ab') as output:
                output.write(b'changed source revision')
            return result

        monkeypatch.setattr(mm.engine, 'read_image', changing_read)
    with pytest.raises(ValueError, match=message):
        mm.MakeMasksScreen._puncta_native_image(screen)
    assert not hasattr(screen, '_puncta_native_cache')
    np.testing.assert_array_equal(display, np.full((12, 16), 25, dtype=np.uint16))


def test_accepting_stale_secondary_preview_preserves_mask_and_reports_changed_parent(qapp):
    mask = np.zeros((12, 16), dtype=np.uint16)
    statuses = []
    screen = SimpleNamespace(_canvas=SimpleNamespace(mask=mask),
                             _require_primary_source=lambda: SimpleNamespace(identity=('new',)),
                             _status_label=SimpleNamespace(setText=statuses.append))
    result = SimpleNamespace(request=SimpleNamespace(shape=mask.shape, primary_token=('old',)),
                             mode=cpu_modes.SECONDARY)
    assert mm.MakeMasksScreen._commit_magnifier_result(screen, result) == []
    assert statuses and 'primary mask changed' in statuses[0].lower()
    assert not np.any(mask)


def test_native_singleton_channel_is_read_only_and_cached_by_revision(qapp, tmp_path):
    image = np.arange(192, dtype=np.float32).reshape(12, 16, 1)
    path = tmp_path / 'source.tif'
    tifffile.imwrite(path, image, photometric='minisblack')
    screen = SimpleNamespace(_image_files=[path.name], _current_index=0,
                             _folder=str(tmp_path), _canvas=SimpleNamespace(image=image[..., 0]))
    native = mm.MakeMasksScreen._puncta_native_image(screen)
    np.testing.assert_array_equal(native, image[..., 0])
    assert native.shape == (12, 16) and not native.flags.writeable
    assert mm.MakeMasksScreen._puncta_native_image(screen) is native
    assert screen._puncta_native_cache[2] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_secondary_fuse_refusal_preserves_ids_even_with_current_primary(qapp):
    mask = np.zeros((12, 16), dtype=np.uint16)
    mask[3:7, 4:8] = 52
    before = mask.copy()
    statuses = []
    screen = SimpleNamespace(_canvas=SimpleNamespace(mask=mask),
                             _require_primary_source=lambda: SimpleNamespace(identity=('current',)),
                             _require_secondary_merge=lambda source: None,
                             _mag_overlap=SimpleNamespace(currentData=lambda: 'merge'),
                             _status_label=SimpleNamespace(setText=statuses.append))
    result = SimpleNamespace(request=SimpleNamespace(shape=mask.shape, primary_token=('current',)),
                             mode=cpu_modes.SECONDARY)
    assert mm.MakeMasksScreen._commit_magnifier_result(screen, result) == []
    assert statuses and 'Fuse is unavailable' in statuses[0]
    np.testing.assert_array_equal(mask, before)


def test_region_requires_parents_and_bypasses_display_enhancement():
    image, parents = field()
    request = mm._MagnifierRequest(key=(), crop=image, box=(0, 0, 128, 96), shape=image.shape,
        mode=cpu_modes.PUNCTA, sensitivity=1, bright=True, min_area=25, model_name='',
        diameter=0, colour=(1, 2, 3), chain=detect_chain.Chain(gamma=2), primary_labels=parents)
    labels, mode, note = mm._segment_region(request)
    assert labels[35, 35] and labels[60, 90] and not labels[10, 110]
    assert mode == cpu_modes.PUNCTA and not note
    with pytest.raises(ValueError, match='parent'):
        mm._segment_region(request._replace(primary_labels=None))


@pytest.mark.parametrize('has_spots', [True, False])
def test_choose_detect_save_and_undo_keeps_parents_and_exports_exact_values(
        qtbot, qt_theme_applied, tmp_path, has_spots, monkeypatch):
    image, parents = field()
    if not has_spots:
        image[:] = 40
    folder = tmp_path/'images'
    folder.mkdir()
    (folder/'masks').mkdir()
    old = np.zeros(image.shape, dtype=np.uint16)
    old[20:25, 20:25] = 1
    parent_file = tmp_path/'parents.tif'
    tifffile.imwrite(folder/'a.tif', image)
    tifffile.imwrite(folder/'masks/a.tif', old)
    tifffile.imwrite(parent_file, parents)
    parent_bytes = parent_file.read_bytes()
    window = mm.MakeMasksScreen()
    qtbot.addWidget(window)
    warnings = []
    monkeypatch.setattr(window, '_warn', lambda *args: warnings.append(args))
    window.show()
    try:
        assert window._open_folder(str(folder))
        qtbot.waitUntil(lambda: window._canvas.image is not None)
        window._mag_mode.setCurrentIndex(window._mag_mode.findData(cpu_modes.PUNCTA))
        assert not window._min_area.isEnabled()
        window._primary_selector.path.setText(str(parent_file))
        window._primary_selector._source_changed()
        qtbot.waitUntil(lambda: window._primary_selector.snapshot is not None, timeout=10000)
        window._canvas.detect_on_normalized = True
        window._cp_invert.setChecked(True)
        np.testing.assert_array_equal(window._detector_image(), image)
        np.testing.assert_array_equal(window._magnifier.region_for((0, 0, 128, 96), invert=True), image)
        assert window._magnifier_context()['invert'] is False
        window._combine_mode.setCurrentIndex(window._combine_mode.findData('replace'))
        window._btn_otsu.click()
        assert np.count_nonzero(np.unique(window._canvas.mask)) == (2 if has_spots else 0)
        assert window._canvas.preserve_ids
        window._btn_save.click()
        np.testing.assert_array_equal(tifffile.imread(folder/'masks/a.tif'), window._canvas.mask)
        rows = tabular.read_table(folder/'masks/a.puncta.csv', canonicalise=False)
        assert rows.included.sum() == (2 if has_spots else 0)
        receipt = json.loads((folder/'masks/a.puncta.json').read_text())
        assert receipt['mask_sha256'] == hashlib.sha256((folder/'masks/a.tif').read_bytes()).hexdigest()
        assert receipt['parent']['sha256'] == hashlib.sha256(parent_bytes).hexdigest()
        assert receipt['csv_sha256'] == hashlib.sha256((folder/'masks/a.puncta.csv').read_bytes()).hexdigest()
        assert receipt['display_normalization_applied'] is False
        assert parent_file.read_bytes() == parent_bytes
        assert not warnings
        csv_bytes = (folder/'masks/a.puncta.csv').read_bytes()
        receipt_bytes = (folder/'masks/a.puncta.json').read_bytes()
        if has_spots:
            unchanged = window._canvas.mask.copy()
            window._canvas.mask[35, 35] = 0
            assert window._save_puncta_measurements(str(folder/'masks/a.tif')) is None
            assert (folder/'masks/a.puncta.csv').read_bytes() == csv_bytes
            window._canvas.mask = unchanged
        tifffile.imwrite(parent_file, parents*0)
        with pytest.raises(ValueError, match='parent mask changed'):
            window._cpu_detect(image, cpu_modes.PUNCTA, window._otsu_settings())
        assert window._save_puncta_measurements(str(folder/'masks/a.tif')) is None
        assert 'changed' in warnings.pop()[1]
        parent_file.write_bytes(parent_bytes)
        with monkeypatch.context() as context:
            def fail_table(*args, **kwargs):
                raise OSError('simulated full output disk')
            context.setattr(tabular, 'write_table', fail_table)
            assert window._save_puncta_measurements(str(folder/'masks/a.tif')) is None
            assert 'mask was saved' in warnings.pop()[1]
        assert (folder/'masks/a.puncta.csv').read_bytes() == csv_bytes
        assert (folder/'masks/a.puncta.json').read_bytes() == receipt_bytes
        assert not list((folder/'masks').glob('.puncta-*'))
        window._on_undo()
        np.testing.assert_array_equal(window._canvas.mask, old)
        window._mag_mode.setCurrentIndex(window._mag_mode.findData(cpu_modes.SECONDARY))
        assert window._primary_selector.parent() is window._method_groups['secondary']
    finally:
        window._magnifier.close()
        window.close_folded()
        window.close()
