"""509: PSF optics inferred from image metadata, a chosen objective or defaults."""
import math

import numpy as np
import pytest
import tifffile

from spacr import point_spread as ps

OME = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
 <Instrument ID="Instrument:0">
  <Objective ID="Objective:0" LensNA="1.4" NominalMagnification="60" Immersion="Oil"/>
 </Instrument>
 <Image ID="Image:0">
  <ObjectiveSettings ID="Objective:0" RefractiveIndex="1.518"/>
  <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="24" SizeY="16"
          SizeC="1" SizeZ="1" SizeT="1" PhysicalSizeX="108" PhysicalSizeXUnit="nm"
          PhysicalSizeY="0.109" PhysicalSizeYUnit="&#181;m">
   <Channel ID="Channel:0:0" EmissionWavelength="461" SamplesPerPixel="1"/>
   <TiffData IFD="0" PlaneCount="1"/>
  </Pixels>
 </Image>
</OME>"""


def _plain(path, shape=(16, 24), **kwargs):
    tifffile.imwrite(path, np.zeros(shape, np.uint16), **kwargs)
    return path


def test_formulae_match_their_citations():
    assert ps.pixel_size_um(6.5, 20) == pytest.approx(0.325)
    assert ps.pixel_size_um(11, 100) == pytest.approx(0.11)
    assert ps.lateral_fwhm_um(520, 0.75) == pytest.approx(0.51 * 0.520 / 0.75)
    assert ps.lateral_fwhm_um(670, 1.4, 1.515) == pytest.approx(0.51 * 0.670 / 1.4)
    for bad in ((0, 20), (6.5, 0), (math.nan, 20)):
        with pytest.raises(ValueError):
            ps.pixel_size_um(*bad)
    with pytest.raises(ValueError, match='refractive index'):
        ps.lateral_fwhm_um(520, 1.4, 1.0)
    with pytest.raises(ValueError, match='Emission'):
        ps.lateral_fwhm_um(5, 0.75)


def test_the_objective_table_is_self_consistent():
    names = [row.name for row in ps.OBJECTIVES]
    assert len(set(names)) == len(names)
    for row in ps.OBJECTIVES:
        magnification, rest = row.name.split('x/')
        aperture, immersion = rest.split(' ')
        assert float(magnification) == row.magnification
        assert float(aperture) == row.numerical_aperture
        assert immersion == row.immersion
        assert row.numerical_aperture < ps.IMMERSION_INDEX[row.immersion]
        assert ps.objective(row.name.upper()) is row
    assert {'10x/0.30 air', '100x/1.45 oil'} <= set(names)
    assert ps.IMMERSION_INDEX == {'air': 1.0, 'water': 1.33, 'oil': 1.515}
    assert [um for _, um in ps.CAMERAS] == [6.5, 4.54, 3.45, 11.0, 16.0]
    assert dict(ps.FLUOROPHORES)['DAPI / Hoechst'] == 461
    assert [nm for _, nm in ps.FLUOROPHORES] == [461, 520, 600, 670]
    for magnification, name in ps.PREFERRED_OBJECTIVE.items():
        assert ps.objective(name).magnification == magnification
    from spacr.settings_spec import convert_settings_dict_for_gui
    kind, options, default = convert_settings_dict_for_gui({'psf_objective': 'auto'})['psf_objective']
    assert (kind, default) == ('combo', 'auto') and options == ['auto'] + names
    with pytest.raises(ValueError, match='Unknown objective'):
        ps.objective('7x/0.1 air')


def test_no_metadata_reports_dimensions_from_the_image_and_defaults_for_the_rest(tmp_path):
    path = _plain(tmp_path / 'plate1_A01_1.tif')
    values = ps.infer_optics(tmp_path)
    assert values['image'].value == str(path)
    assert values['image_shape'] == ps.OpticalValue((16, 24), 'image', path.name)
    assert values['objective'].value == '20x/0.75 air'
    for key in ('objective', 'magnification', 'numerical_aperture',
                'refractive_index', 'emission_nm', 'camera_pixel_um'):
        assert values[key].source == 'default'
    assert values['pixel_size_um'].value == pytest.approx((0.325, 0.325))
    assert values['pixel_size_um'].source == 'calculated'
    assert values['fwhm_um'].value == pytest.approx((0.3536, 0.3536))
    assert values['fwhm_um'].source == 'calculated'
    assert ps.infer_optics(None)['pixel_size_um'].value == values['pixel_size_um'].value


def test_ome_metadata_supplies_every_optical_value_it_states(tmp_path):
    path = _plain(tmp_path / 'field.ome.tif', description=OME, metadata=None)
    values = ps.infer_optics(path)
    assert values['magnification'] == ps.OpticalValue(60.0, 'metadata', path.name)
    assert values['numerical_aperture'].value == 1.4
    assert values['refractive_index'].value == 1.518
    assert values['emission_nm'] == ps.OpticalValue(461.0, 'metadata', path.name)
    assert values['pixel_size_um'].value == pytest.approx((0.109, 0.108))
    assert values['pixel_size_um'].source == 'metadata'
    assert values['objective'].value == '60x/1.40 oil'
    assert values['fwhm_um'].value[0] == pytest.approx(0.51 * 0.461 / 1.4)
    chosen = ps.infer_optics(path, objective_name='20x/0.75 air', emission_nm=600)
    assert chosen['numerical_aperture'] == ps.OpticalValue(0.75, 'objective', '20x/0.75 air')
    assert chosen['emission_nm'].source == 'chosen'
    assert chosen['pixel_size_um'].source == 'metadata'


def test_imagej_and_centimetre_calibrations_are_read_but_dpi_is_not(tmp_path):
    imagej = _plain(tmp_path / 'a.tif', imagej=True, resolution=(1 / 0.2, 1 / 0.25),
                    metadata={'unit': 'um'})
    assert ps.infer_optics(imagej)['pixel_size_um'] == ps.OpticalValue(
        pytest.approx((0.25, 0.2)), 'imagej', imagej.name)
    centimetre = _plain(tmp_path / 'b.tif', resolution=(1e4 / 0.5, 1e4 / 0.5),
                        resolutionunit='CENTIMETER')
    assert ps.infer_optics(centimetre)['pixel_size_um'].value == pytest.approx((0.5, 0.5))
    assert ps.infer_optics(centimetre)['pixel_size_um'].source == 'tiff_resolution'
    inch = _plain(tmp_path / 'c.tif', resolution=(72, 72), resolutionunit='INCH')
    assert ps.infer_optics(inch)['pixel_size_um'].source == 'calculated'


def test_a_magnification_token_in_the_file_name_picks_the_common_objective(tmp_path):
    path = _plain(tmp_path / 'plate1_40x_A01.tif')
    values = ps.infer_optics(path, camera_pixel_um=6.5)
    assert values['magnification'] == ps.OpticalValue(40.0, 'file_name', path.name)
    assert values['objective'].value == '40x/0.95 air'
    assert values['numerical_aperture'].source == 'objective'
    assert values['pixel_size_um'].value == pytest.approx((6.5 / 40,) * 2)
    lines = ps.describe_optics(values)
    assert any(line.startswith('magnification = 40 (file_name') for line in lines)


def test_fill_psf_settings_fills_only_what_is_unset(tmp_path):
    _plain(tmp_path / 'x.tif')
    assert ps.fill_psf_settings({'psf_operation': 'none'}) is None
    measured = {'psf_operation': 'convolve', 'psf_source': 'measured'}
    assert ps.fill_psf_settings(measured) is None and 'psf_fwhm_um' not in measured
    settings = {'psf_operation': 'deconvolve', 'src': [str(tmp_path)],
                'psf_objective': '100x/1.45 oil', 'psf_fwhm_um': [0.3, 0.3],
                'psf_image_sampling_um': None}
    values = ps.fill_psf_settings(settings)
    assert values['objective'].source == 'chosen'
    assert settings['psf_image_sampling_um'] == [0.065, 0.065]
    assert settings['psf_fwhm_um'] == [0.3, 0.3]
    assert ps.fill_psf_settings(settings) is None


def test_real_toxo_pv_field_infers_dimensions_and_defaults():
    """The Make Masks example fields carry no calibration; say so rather than guess."""
    import os
    from pathlib import Path
    folder = Path(os.environ.get('SPACR_TOXO_PV_DIR') or
                  Path.home() / '.cache/spacr/example_data/make_masks_toxo_pv')
    if not folder.is_dir():
        pytest.skip('Toxo PV example data not downloaded')
    values = ps.infer_optics(folder)
    assert values['image_shape'].source == 'image'
    assert values['image_shape'].value == (1994, 1994)
    assert values['pixel_size_um'].source == 'calculated'
    assert values['objective'].source == 'default'
