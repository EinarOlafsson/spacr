"""What the PSF readers refuse, and what they read from image headers.

The measured-kernel loader and the optics metadata readers each take files
the user points at. These cases pin the refusals (a wrong file type, a
complex-valued kernel, an NPY header version the loader does not know) and
the header readings that are only partly usable: a malformed OME document,
an objective without a numeric aperture, a resolution tag of zero.
"""
import io

import numpy as np
import pytest
import tifffile

from spacr import point_spread as ps


def test_a_gaussian_needs_two_or_three_axes_and_a_sane_truncation():
    with pytest.raises(ValueError, match='two or three spatial axes'):
        ps.gaussian_psf(fwhm_um=.3, sampling_um=.1, ndim=4)
    with pytest.raises(ValueError, match='between two and eight sigma'):
        ps.gaussian_psf(fwhm_um=.3, sampling_um=.1, truncate=1.0)
    with pytest.raises(ValueError, match='between two and eight sigma'):
        ps.gaussian_psf(fwhm_um=.3, sampling_um=.1, truncate=float('nan'))


def test_a_kernel_file_that_is_not_npy_or_tiff_is_refused(tmp_path):
    path = tmp_path / 'kernel.png'
    path.write_bytes(b'not a kernel')
    with pytest.raises(ValueError, match='NPY or TIFF'):
        ps.load_psf(path, sampling_um=.1)


def test_a_version_two_npy_header_is_read(tmp_path):
    kernel = np.zeros((3, 3), np.float32)
    kernel[1, 1] = 2.0
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, kernel, version=(2, 0))
    path = tmp_path / 'kernel.npy'
    path.write_bytes(buffer.getvalue())

    loaded = ps.load_psf(path, sampling_um=.1)

    assert loaded.shape == (3, 3)
    assert loaded.array()[1, 1] == pytest.approx(1.0)
    assert loaded.source == 'measured'


def test_an_npy_header_version_the_loader_does_not_know_is_refused(tmp_path):
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ones((3, 3), np.float32),
                              version=(1, 0))
    raw = bytearray(buffer.getvalue())
    raw[6] = 9
    path = tmp_path / 'kernel.npy'
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match='Unsupported NPY kernel format'):
        ps.load_psf(path, sampling_um=.1)


def test_a_complex_valued_kernel_is_refused(tmp_path):
    path = tmp_path / 'kernel.npy'
    np.save(path, np.ones((3, 3), np.complex64))
    with pytest.raises(ValueError, match='real numbers'):
        ps.load_psf(path, sampling_um=.1)


def test_a_tiff_with_two_series_is_refused(tmp_path):
    path = tmp_path / 'kernel.tif'
    with tifffile.TiffWriter(path) as tif:
        tif.write(np.ones((3, 3), np.float32))
        tif.write(np.ones((5, 5), np.float32))
    with pytest.raises(ValueError, match='exactly one spatial series'):
        ps.load_psf(path, sampling_um=.1)


def test_an_ome_length_that_is_not_a_number_is_unusable():
    assert ps._micrometers('wide', 'µm') is None
    assert ps._micrometers(None, 'nm') is None
    assert ps._micrometers(250, 'nm') == pytest.approx(.25)
    assert ps._micrometers(500, 'µm') is None


def test_malformed_ome_xml_states_nothing():
    assert ps._ome_values('<OME><Pixels', 'x.tif') == {}


def test_partly_usable_ome_metadata_keeps_only_what_is_usable():
    xml = (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Instrument><Objective LensNA="n/a" NominalMagnification="40"'
        ' Immersion="Glycerol"/></Instrument>'
        '<Image><ObjectiveSettings RefractiveIndex="unknown"/>'
        '<Pixels PhysicalSizeX="0.2" PhysicalSizeXUnit="µm">'
        '<Channel EmissionWavelength="0.52" EmissionWavelengthUnit="µm"/>'
        '</Pixels></Image></OME>')

    found = ps._ome_values(xml, 'x.tif')

    assert found['pixel_size_um'].value == pytest.approx((.2, .2))
    assert found['magnification'].value == 40.0
    assert 'numerical_aperture' not in found
    assert 'refractive_index' not in found
    assert found['emission_nm'].value == pytest.approx(520.0)


def test_an_emission_wavelength_that_is_not_a_number_is_dropped():
    xml = ('<OME><Image><Pixels><Channel EmissionWavelength="green"/>'
           '</Pixels></Image></OME>')
    assert 'emission_nm' not in ps._ome_values(xml, 'x.tif')


class _Tag:
    def __init__(self, value):
        self.value = value


class _Tif:
    is_imagej = False
    imagej_metadata = None


def test_resolution_tags_that_say_nothing_about_the_specimen_are_ignored():
    tif = _Tif()
    assert ps._resolution_values(tif, {}, 'x.tif') == {}
    assert ps._resolution_values(
        tif, {'XResolution': _Tag((1, 0))}, 'x.tif') == {}
    assert ps._resolution_values(
        tif, {'XResolution': _Tag((72, 1))}, 'x.tif') == {}
    tiny = {'XResolution': _Tag((1, 1)), 'ResolutionUnit': _Tag(3)}
    assert ps._resolution_values(tif, tiny, 'x.tif') == {}
    ok = {'XResolution': _Tag((50000, 1)), 'ResolutionUnit': _Tag(3)}
    assert ps._resolution_values(tif, ok, 'x.tif')[
        'pixel_size_um'].value == pytest.approx((.2, .2))


def test_the_first_image_is_found_in_a_list_a_folder_or_its_orig(tmp_path):
    assert ps._first_image(None) is None
    assert ps._first_image([tmp_path / 'nothing', str(tmp_path)]) is None
    note = tmp_path / 'readme.txt'
    note.write_text('x')
    assert ps._first_image(note) is None
    orig = tmp_path / 'orig'
    orig.mkdir()
    image = orig / 'b.tif'
    tifffile.imwrite(image, np.zeros((4, 4), np.uint8))
    assert ps._first_image([note, tmp_path]) == image


def test_an_unreadable_tiff_still_yields_the_file_name_magnification(tmp_path):
    path = tmp_path / 'plate1_40x_A01.tif'
    path.write_bytes(b'this is not a tiff')

    found = ps.image_optics_metadata(path)

    assert set(found) == {'magnification'}
    assert found['magnification'].value == 40.0
    assert found['magnification'].source == 'file_name'
