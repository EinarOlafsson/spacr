"""The private PSF lesson drives current controls, not fabricated screenshots."""
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tools.tutorials.capture_psf_workflow import prepare_source, record_editor, record_measure


def fixture_source(tmp_path):
    yy, xx = np.mgrid[:64, :64]
    planes = np.zeros((64, 64, 7), dtype=np.uint16)
    planes[..., 1] = ((xx + yy) * 50 + ((xx > 24) & (xx < 40)) * 2000)
    planes[20:40, 20:40, 4] = 7
    source = tmp_path / 'synthetic_test_only.npy'
    np.save(source, planes)
    return source, planes


def test_preparation_preserves_exact_source_pixels_and_mask_ids(tmp_path):
    source, planes = fixture_source(tmp_path)
    before = source.read_bytes()
    folder, proof = prepare_source(source, tmp_path / 'stage', side=32)
    assert proof['crop_yxhw'] == [16, 16, 32, 32]
    np.testing.assert_array_equal(tifffile.imread(proof['image']), planes[16:48, 16:48, 1])
    np.testing.assert_array_equal(tifffile.imread(proof['mask']), planes[16:48, 16:48, 4])
    assert source.read_bytes() == before
    assert folder.is_dir()
    with pytest.raises(FileExistsError):
        prepare_source(source, tmp_path / 'stage', side=32)
    assert source.read_bytes() == before


def test_editor_capture_clicks_real_infer_compare_apply_with_alpha_off(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: False)
    source, _ = fixture_source(tmp_path)
    folder, prepared = prepare_source(source, tmp_path / 'stage', side=64)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.resize(1400, 1000)
    screen.show()
    screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: screen._canvas.image is not None and not screen._loading)
    scenes = []

    def capture(name):
        assert screen.isVisible()
        target = screen._compare_dialog if 'compare' in name else screen
        assert not target.grab().isNull()
        scenes.append(name)

    try:
        proof = record_editor(screen, capture, lambda n: qtbot.wait(max(1, int(n * 1000))))
        assert len(scenes) == 5
        assert proof['original_and_labels_unchanged']
        assert proof['processed_differs_from_original']
        assert proof['sources']['magnification'] == ['objective', '40x/0.95 air']
        assert proof['sources']['image_y'][0] == 'calculated'
        assert proof['provenance']['intensity_source_for_measurement'] == 'original image'
        assert Path(prepared['image']).is_file()
        assert not screen._btn_apply.isChecked()
    finally:
        screen.close()


def test_measure_capture_changes_real_explicit_choice_without_running(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: False)
    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    screen.resize(1400, 1000)
    screen.show()
    scenes = []
    try:
        proof = record_measure(screen, scenes.append, lambda n: qtbot.wait(max(1, int(n * 1000))))
        assert scenes == ['05_measure_original_intensities', '06_measure_explicit_processed_intensities']
        assert proof['settings']['psf_measurement_source'] == 'processed'
        assert proof['settings']['psf_operation'] == 'convolve'
        assert proof['settings']['psf_image_sampling_um'] == [.2, .2]
        assert not proof['measurement_run_performed']
        assert screen._settings_model.collect()['psf_measurement_source'] == 'original'
    finally:
        screen.close()


def test_capture_refuses_alpha_on_before_touching_controls(monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: True)
    with pytest.raises(RuntimeError, match='alpha off'):
        record_editor(None, None, None)
    with pytest.raises(RuntimeError, match='alpha off'):
        record_measure(None, None, None)
