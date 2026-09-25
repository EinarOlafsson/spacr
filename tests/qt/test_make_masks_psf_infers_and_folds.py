"""509: the PSF infers its own optics, and the advanced enhancement rows start folded."""
import numpy as np
import pytest
import tifffile

from spacr.qt.screens import make_masks as mm
from spacr.qt.widgets import psf_controls as controls

OME = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
 <Instrument ID="Instrument:0">
  <Objective ID="Objective:0" LensNA="1.4" NominalMagnification="60" Immersion="Oil"/>
 </Instrument>
 <Image ID="Image:0">
  <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="24" SizeY="16"
          SizeC="1" SizeZ="1" SizeT="1" PhysicalSizeX="0.108" PhysicalSizeY="0.108">
   <Channel ID="Channel:0:0" EmissionWavelength="670" SamplesPerPixel="1"/>
   <TiffData IFD="0" PlaneCount="1"/>
  </Pixels>
 </Image>
</OME>"""


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    image = np.zeros((48, 48), np.uint16)
    image[20:28, 20:28] = 4000
    made._canvas.set_image_and_mask(image, np.zeros_like(image, dtype=np.uint32))
    yield made
    made.close()


def _shown(widget, fold):
    return widget.isVisibleTo(fold.parentWidget())


def test_each_group_shows_only_its_essential_rows_until_opened(screen, qtbot):
    screen.show()
    psf = screen._psf_controls
    restoration = screen._restoration_controls
    groups = (
        (psf.details, (psf.operation, psf.objective, psf.infer),
         (psf.source, psf.image_y, psf.fwhm_x, psf.magnification, psf.camera,
          psf.fluorophore, psf.path, psf.iterations)),
        (restoration.details, (restoration.operation, restoration.install),
         (restoration.structure, restoration.diameter, restoration.reload)),
        (screen._enh_clahe_details, (screen._enh_clahe,),
         (screen._enh_clahe_tile, screen._enh_clahe_clip)),
    )
    for fold, essential, advanced in groups:
        assert fold.shut
        assert all(_shown(widget, fold) for widget in essential)
        assert not any(_shown(widget, fold) for widget in advanced)
        fold.folder.toggle()
        assert all(_shown(widget, fold) for widget in advanced)


def test_an_opened_fold_stays_open_and_a_closed_one_is_forgotten(qtbot, qt_theme_applied):
    from spacr.qt.preferences import get_folded_panels

    first = mm.MakeMasksScreen()
    qtbot.addWidget(first)
    assert first._enh_clahe_details.shut
    first._enh_clahe_details.folder.toggle()
    assert get_folded_panels()['make_masks/clahe_details'] is False
    first.close()
    second = mm.MakeMasksScreen()
    qtbot.addWidget(second)
    assert not second._enh_clahe_details.shut
    assert second._psf_controls.details.shut
    second._enh_clahe_details.folder.toggle()
    assert 'make_masks/clahe_details' not in get_folded_panels()
    second.close()


def test_one_click_deconvolution_runs_from_the_common_defaults(screen, qtbot):
    widget = screen._psf_controls
    widget.operation.setCurrentIndex(widget.operation.findData('deconvolve'))
    qtbot.waitUntil(lambda: widget._kernel is not None)
    assert widget._error == ''
    assert widget._kernel.sampling_um == (0.325, 0.325)
    assert 'FWHM 0.3536 µm' in widget.summary.text()
    assert screen._enhancement_chain().psf_operation == 'deconvolve'


def test_choosing_an_objective_camera_and_fluorophore_recalculates_with_sources(screen):
    widget = screen._psf_controls
    widget.objective.setCurrentIndex(widget.objective.findData('100x/1.45 oil'))
    widget.camera.setCurrentIndex(widget.camera.findData(11.0))
    widget.fluorophore.setCurrentIndex(widget.fluorophore.findData(461.0))
    assert widget.source_of('numerical_aperture') == ('objective', '100x/1.45 oil')
    assert widget.refractive_index.value() == 1.515
    assert widget.image_x.value() == pytest.approx(0.11)
    assert widget.fwhm_x.value() == pytest.approx(round(0.51 * 0.461 / 1.45, 4))
    assert widget.source_of('emission_nm')[0] == 'fluorophore'
    widget.fwhm_x.setValue(0.5)
    widget.numerical_aperture.setValue(1.3)
    assert widget.source_of('numerical_aperture')[0] == 'entered'
    assert widget.fwhm_x.value() == 0.5


def test_infer_fills_every_field_from_ome_metadata_with_its_source(screen, qtbot, tmp_path):
    path = tmp_path / 'field.ome.tif'
    tifffile.imwrite(path, np.zeros((16, 24), np.uint16), description=OME, metadata=None)
    widget = screen._psf_controls
    widget.infer_from([str(path)])
    qtbot.waitUntil(lambda: widget.source_of('magnification')[0] == 'metadata')
    assert widget.objective.currentData() == '60x/1.40 oil'
    assert widget.numerical_aperture.value() == 1.4
    assert widget.source_of('emission_nm') == ('metadata', path.name)
    assert widget.emission_nm.value() == 670
    assert widget.image_y.value() == pytest.approx(0.108)
    assert widget.source_of('image_y') == ('metadata', path.name)
    assert widget.fwhm_y.value() == pytest.approx(round(0.51 * 0.670 / 1.4, 4))
    assert widget.source_of('fwhm_y')[0] == 'calculated'
    assert '16 × 24' in widget.dimensions.text()
    assert 'OME metadata' in widget._source_labels['image_y'].text()


def test_infer_without_metadata_keeps_the_chosen_objective_and_reads_dimensions(
        screen, qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'plain.tif'
    tifffile.imwrite(path, np.zeros((20, 30), np.uint16))
    widget = screen._psf_controls
    widget.objective.setCurrentIndex(widget.objective.findData('40x/0.95 air'))
    monkeypatch.setattr(screen, '_current_image_paths', lambda: [str(path)])
    widget.image_paths = screen._current_image_paths
    widget.infer.click()
    qtbot.waitUntil(lambda: '20 × 30' in widget.dimensions.text())
    assert widget.objective.currentData() == '40x/0.95 air'
    assert widget.source_of('magnification') == ('objective', '40x/0.95 air')
    assert widget.source_of('camera_pixel_um')[0] == 'default'
    assert widget.image_x.value() == pytest.approx(round(6.5 / 40, 4))


def test_infer_with_no_open_image_asks_for_one_and_a_cancel_changes_nothing(
        qtbot, monkeypatch):
    widget = controls._PSFControls(image_paths=lambda: [])
    qtbot.addWidget(widget)
    monkeypatch.setattr(controls.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *args, **kwargs: ('', '')))
    before = widget.image_x.value()
    widget.infer.click()
    assert widget.image_x.value() == before
    assert not widget._infer_jobs.is_busy()
    widget._shutdown()
