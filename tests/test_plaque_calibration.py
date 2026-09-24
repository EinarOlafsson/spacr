"""Measured geometry and calibration survive review, sorting and persistence."""
import numpy as np
import pytest
from spacr import plaque_papers as pp


def annotation():
    return pp.Annotation(pp.Region(0, 0, 100, 80))


def test_unknown_physical_scale_still_records_pixel_diameter():
    a = annotation()
    assert pp.calibration_values(a)['well_diameter_px'] == 90
    assert pp.calibration_values(a)['pixels_per_um'] is None


@pytest.mark.parametrize('value', ['NaN', 'inf', '-1', '0', 'abc'])
def test_invalid_scale_is_rejected(value):
    with pytest.raises(ValueError):
        pp.calibration_number(value, name='scale')


def test_manual_scale_precedence_units_and_time():
    a = annotation()
    a.pixels_per_um = 2
    a.formation_hours = 0
    scale = pp._scales_for_regions(np.zeros((100,100,3), np.uint8), [a.region], [],
                                  annotations=[a], plate_format='6-well', pixels_per_um=3)[0]
    values = pp.calibration_values(a, scale, 168)
    assert scale.px_per_mm == 2000
    assert values['pixels_per_um'] == 2
    assert values['formation_hours'] == 0
    assert values['formation_time_source'] == 'manual annotation'


def test_review_roundtrip_and_database_migration(tmp_path):
    a = annotation()
    a.pixels_per_um = .025
    a.formation_hours = 168
    path = tmp_path / 'review.csv'
    pp.write_annotation_overrides(path, [pp._annotation_file_row('fig.png', 1, a)])
    restored = pp.apply_overrides('fig', [annotation()], pp.read_annotation_overrides(path))[0]
    assert restored.pixels_per_um == .025
    assert restored.formation_hours == 168
    db = pp.open_database(tmp_path / 'results.db')
    for table in ('regions', 'figure_annotations'):
        columns = {r[1] for r in db.execute(f'PRAGMA table_info({table})')}
        assert {'well_diameter_px', 'pixels_per_um', 'formation_hours', 'formation_time_source'} <= columns
    db.close()


def test_crop_manual_scale_without_detected_well():
    from spacr.submodules import _plaque_scale_for, _plaque_well_diameter
    scale = _plaque_scale_for('crop.tif', {'plaque_pixels_per_um': .01})
    assert scale.area_mm2(100) == pytest.approx(1)
    assert scale.well_diameter_px is None
    assert _plaque_well_diameter('well', {'_well_geometry': {'well': dict(x0=0,y0=0,x1=100,y1=80)}}) == 90
