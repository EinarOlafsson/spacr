"""The actual Make Masks mode detects, saves and binds native centre values."""
import hashlib
import json

import numpy as np
import pytest
import tifffile

from spacr import tabular
from spacr.qt import cpu_modes, detect_chain
from spacr.qt.screens import make_masks as mm
from tests.test_center_pixel_puncta import field


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
