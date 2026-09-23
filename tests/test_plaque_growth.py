"""Approximation provenance, resolution invariance and real endpoint assessment."""
import csv
import json
import math
from pathlib import Path
import statistics
import pytest
from spacr import plaque_growth as pg


def areas(diameter, count=8):
    return [math.pi * diameter ** 2 / 4] * count


def test_physical_page_pooling_is_resolution_invariant():
    report = pg.estimate_page([
        dict(well=1, areas_px=areas(pg.REFERENCE_DIAMETER_UM * .01), pixels_per_um=.01),
        dict(well=2, areas_px=areas(pg.REFERENCE_DIAMETER_UM * .1), pixels_per_um=.1)])
    assert report['page_diameter_um'] == pytest.approx(pg.REFERENCE_DIAMETER_UM)
    assert all(w['estimated_formation_hours'] == pytest.approx(168) for w in report['wells'])
    assert all(w['estimated_pixels_per_um'] is None for w in report['wells'])
    assert report['page_selected_count'] == 4


def test_known_time_gives_scale_and_no_time_claim():
    report = pg.estimate_page([dict(well=1, areas_px=areas(89.38178699548309), formation_hours=168)])
    row = report['wells'][0]
    assert row['estimated_pixels_per_um'] == pytest.approx(.1)
    assert row['estimated_formation_hours'] is None


def test_no_anchor_is_explicit_assumption_and_no_second_pass_feedback():
    settings = {'plaque_estimate_growth': True}
    wells = [dict(well='a', areas_px=areas(50))]
    result = pg.estimates_from_settings(wells, settings)
    assert result['a']['estimated_formation_hours'] == 168
    assert 'assumed reference duration' in result['a']['estimation_source']
    assert 'temporal_validation' in result['a']['growth_estimate_provenance']
    assert pg.estimates_from_settings(wells, settings) == result
    assert pg.estimates_from_settings(wells, {}) == {}
    assert 'pixels_per_um' not in wells[0]


def test_known_values_and_zero_time_are_preserved():
    row = pg.estimate_page([dict(well=1, areas_px=areas(10), pixels_per_um=2, formation_hours=0)])['wells'][0]
    assert row['estimated_pixels_per_um'] is None
    assert row['estimated_formation_hours'] is None
    row = pg.estimate_page([dict(well=1, areas_px=areas(10), formation_hours=0)])['wells'][0]
    assert row['estimated_pixels_per_um'] is None


def test_conflicting_times_do_not_pool_or_assign_arbitrary_page_time():
    report = pg.estimate_page([
        dict(well=1, areas_px=areas(50), pixels_per_um=.1, formation_hours=24),
        dict(well=2, areas_px=areas(50), pixels_per_um=.1, formation_hours=48),
        dict(well=3, areas_px=areas(50), pixels_per_um=.1)])
    assert report['page_diameter_um'] is None
    assert 'well largest-quarter' in report['wells'][2]['estimation_source']


def test_small_samples_invalid_values_and_duplicate_identifiers():
    assert pg.largest_quarter([1,2,3])['diameter'] is None
    for value in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            pg.largest_quarter([value])
    with pytest.raises(ValueError, match='unique'):
        pg.estimate_page([dict(well=1, areas_px=[])] * 2)


def test_real_control_data_reproduce_anchor_and_held_out_endpoint_errors():
    folder = Path(__file__).parent / 'data/plaque_growth'
    receipt = json.loads((folder / 'reference.json').read_text())
    with (folder / 'control_areas.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 528
    diameters = {}
    for experiment in {r['experiment'] for r in rows}:
        selected = [float(r['area_um2']) for r in rows if r['experiment'] == experiment]
        diameters[experiment] = pg.largest_quarter(selected)['diameter']
    assert statistics.median(diameters.values()) == pytest.approx(pg.REFERENCE_DIAMETER_UM)
    errors = []
    for experiment, diameter in diameters.items():
        reference = statistics.median(d for e, d in diameters.items() if e != experiment)
        errors.append(abs(168 * diameter / reference - 168))
    assert statistics.mean(errors) == pytest.approx(receipt['endpoint_mae_hours'])
    assert 39 < statistics.mean(errors) < 40
    assert not receipt['temporal_validation']


def test_folder_run_saves_estimates_and_provenance_without_fabricating_measured_area(tmp_path):
    import numpy as np
    import sqlite3
    from types import SimpleNamespace
    from PIL import Image
    from spacr import plaque_papers as pp
    Image.fromarray(np.full((400,400,3), 150, np.uint8)).save(tmp_path / 'figure.png')

    def detect(image, weights, **kwargs):
        return [SimpleNamespace(x0=100, y0=100, x1=300, y1=300, confidence=.9)]

    def segment(crop):
        labels = np.zeros(crop.shape[:2], np.int32)
        for i in range(8):
            labels[10:20, 10 + 20*i:20 + 20*i] = i + 1
        return labels

    result = pp.measure_figure_folder(tmp_path, detector='fake', segmenter='fake',
        detect=detect, segment=segment, read_text=lambda path: [],
        growth_settings={'plaque_estimate_growth': True})
    with sqlite3.connect(result['database']) as db:
        assert db.execute('SELECT COUNT(*) FROM plaques').fetchone()[0] == 8
        for table in ('regions', 'figure_annotations'):
            row = db.execute(f'SELECT well_diameter_px,pixels_per_um,formation_hours,estimated_pixels_per_um,estimated_formation_hours,growth_estimate_provenance FROM {table}').fetchone()
            assert row[:3] == (200, None, None)
            assert row[3] > 0 and row[4] == 168
            assert json.loads(row[5])['reference_id'] == pg.REFERENCE_ID
        assert db.execute('SELECT COUNT(*) FROM plaques WHERE area_mm2 IS NOT NULL').fetchone()[0] == 0
