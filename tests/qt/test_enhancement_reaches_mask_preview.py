"""Item 508: the Live preview and Make Masks' hand-off use the plate run's chain."""
import numpy as np
import pytest

from spacr.psf_pipeline import apply_chain, prepare_chain, prepare_psf
from spacr.qt import detect_chain as dc
from spacr.qt.widgets import live_preview as lp
from tests.qt.test_live_preview_psf import Model, settings, source


@pytest.mark.parametrize('operation', ['none', 'convolve'])
def test_preview_model_reads_the_chained_channels(monkeypatch, operation):
    raw = source()
    original = raw.copy()
    config = settings(operation)
    config.update(dc.chain_settings(dc.Chain(log=True, percentile_clip=True,
                                             denoise='gaussian')))
    model = Model()
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda name: model)
    request = lp.PreviewRequest(raw, object_types=('cell', 'pathogen'),
                               channels={'cell': 0, 'pathogen': 1},
                               preprocess_settings=config)
    lp._segment_multi(request)
    chain = prepare_chain(config, prepare_psf(config))
    expected = apply_chain(raw[..., :2], chain)
    for actual, target in zip(model.inputs, (expected[..., 0], expected[..., 1])):
        np.testing.assert_allclose(actual, target, atol=1e-4)
    np.testing.assert_array_equal(raw, original)
    steps = request.provenance['enhancement']
    assert steps['log'] and steps['denoise'] == 'gaussian'
    assert ('psf' in steps) == (operation != 'none')


def test_preview_with_the_chain_off_records_none(monkeypatch):
    config = settings('none')
    config.update(dc.chain_settings())
    model = Model()
    monkeypatch.setattr(lp, 'preview_cellpose_model', lambda name: model)
    raw = source()
    request = lp.PreviewRequest(raw, object_types=('cell',),
                               channels={'cell': 0},
                               preprocess_settings=config)
    lp._segment_multi(request)
    np.testing.assert_array_equal(model.inputs[0], raw[..., 0])
    assert request.provenance['enhancement'] == 'none'


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    made.close_folded()


def test_make_masks_writes_the_chain_mask_reads(screen):
    screen._enh_log.setChecked(True)
    screen._enh_log_gain.setValue(4.0)
    screen._enh_sqrt.setChecked(True)
    screen._enh_percentile_clip.setChecked(True)
    screen._enh_percentile_high.setValue(97.0)
    screen._enh_denoise.setCurrentIndex(screen._enh_denoise.findData('tv'))
    screen._enh_split.setChecked(True)
    written = screen.mask_settings()
    assert written['enhance_log'] is True and written['psf_operation'] == 'none'
    configured = screen._enhancement_chain()
    read = prepare_chain(written)
    assert read == configured._replace(morphology='none', split=False,
                                       psf=None, psf_error='')
    image = np.random.default_rng(0).normal(100, 10, (32, 32)).astype(np.float32)
    np.testing.assert_array_equal(
        apply_chain(image[..., None], read)[..., 0],
        dc.prepare(image, configured))


def test_use_in_mask_generation_fills_the_mask_screen(screen, monkeypatch):
    received, visited = [], []

    class Target:
        def apply_settings_dict(self, values):
            received.append(values)

    class Window:
        _screens = {'mask': Target()}

        def _on_nav_selected(self, key):
            visited.append(key)

    screen._enh_gamma.setValue(0.6)
    monkeypatch.setattr(screen, 'window', lambda: Window())
    screen._send_chain_to_mask()
    assert received[0]['enhance_gamma'] == pytest.approx(0.6)
    assert visited == ['mask']
    assert 'gamma' in screen._status_label.text()


def test_use_in_mask_generation_without_an_app_says_so(screen):
    screen._send_chain_to_mask()
    assert 'Mask module' in screen._status_label.text()
