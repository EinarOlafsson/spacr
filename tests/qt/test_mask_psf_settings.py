"""The Mask form exposes calibrated opt-in PSF settings with actual API help."""
import pytest
from PySide6.QtWidgets import QWidget

from spacr.qt.screens.settings_model import SettingsWidgets, api_docs_url


@pytest.mark.parametrize('app', ['mask', 'timelapse'])
def test_mask_psf_settings_are_grouped_and_round_trip(qtbot, app):
    parent = QWidget()
    qtbot.addWidget(parent)
    form = SettingsWidgets(app, parent)
    sections = form.build_sections()
    assert any(title == 'Point Spread Function' for title, _ in sections)
    assert form.collect()['psf_operation'] == 'none'
    operation = form._widgets['psf_operation']
    operation.setCurrentText('deconvolve')
    form._widgets['psf_image_sampling_um'].setText('[0.2, 0.3]')
    form._widgets['psf_fwhm_um'].setText('[0.6, 0.9]')
    settings = form.collect()
    assert settings['psf_operation'] == 'deconvolve'
    assert settings['psf_image_sampling_um'] == [.2, .3]
    assert settings['psf_fwhm_um'] == [.6, .9]
    assert settings['psf_iterations'] == 20
    for key in ('psf_operation', 'psf_source', 'psf_path', 'psf_fwhm_um',
                'psf_image_sampling_um', 'psf_kernel_sampling_um', 'psf_iterations'):
        assert 'psf_pipeline/index.html#spacr.psf_pipeline.prepare_psf' in api_docs_url(app, key)
