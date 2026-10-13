"""Motility correction operates on original objects before parent aggregation."""
import hashlib
import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as tl


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def field(tmp_path, *, number=1, rising=False, occupied=False):
    merged = tmp_path / 'merged'
    merged.mkdir(exist_ok=True)
    names = []
    cell = np.zeros((64, 64), dtype=np.float32)
    cell[8:28, 6:26] = 1
    cell[34:54, 36:56] = 2
    if occupied:
        cell[:] = 1
    pathogen = np.zeros_like(cell)
    pathogen[10:13, 8:11] = 1
    pathogen[20:23, 18:21] = 2
    pathogen[40:43, 42:45] = 3
    for frame, time in enumerate([0, 2, 5]):
        factor = (2. if rising else .5) ** frame
        offset = 1000 + 100 * number
        image0 = offset + (cell > 0) * (200 * number * factor)
        image1 = np.full(cell.shape, offset, dtype=np.float32)
        for label in [1, 2, 3]:
            image1[pathogen == label] += label * 100 * factor
        name = f'plate1_A01_{number}_{time}.npy'
        np.save(merged / name, np.stack([image0, image1, cell, pathogen]))
        names.append(name)
    return (str(tmp_path), names, 2, 0, None, 1)


def snapshot(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.npy')}


@pytest.mark.parametrize('method', ['ratio', 'exponential', 'histogram'])
def test_actual_worker_corrects_objects_before_unequal_child_aggregation(tmp_path, method):
    args = field(tmp_path)
    before = snapshot(tmp_path / 'merged')
    legacy = tl._process_merged_group(args)
    corrected, raw, fits = tl._process_merged_group(args + (method,))
    pd.testing.assert_frame_equal(raw, legacy, check_like=True)
    assert snapshot(tmp_path / 'merged') == before
    assert set(corrected.bleach_correction_method) == {method}
    assert set(fits.time_unit) == {'source_frame'}
    assert all(json.loads(value) == args[1] for value in fits.source_files)
    np.testing.assert_allclose(corrected.cell_mean_intensity_ch0, 1300, atol=.01)
    np.testing.assert_allclose(corrected.cell_mean_intensity, 1300, atol=.01)
    assert set(corrected.n_pathogens) == {1, 2}
    if method != 'histogram':
        # Three individual pathogen signals are 100/200/300 over the offset.
        # Their median trend must be fitted before two children are averaged.
        expected = np.where(corrected.cellID == 1, 1250, 1400)
        np.testing.assert_allclose(corrected.pathogen_mean_intensity, expected, atol=.01)
        np.testing.assert_allclose(corrected.pathogen_p95_intensity_ch1, expected, atol=.01)
    assert not any(c.startswith('raw_') for c in corrected)
    assert set(fits.object_type) == {'cell', 'pathogen', 'cytoplasm'}


def test_fields_and_channels_have_independent_trends_and_rises_are_kept(tmp_path):
    first = field(tmp_path, number=1)
    second = field(tmp_path, number=2, rising=True)
    corrected1, _, fits1 = tl._process_merged_group(first + ('ratio',))
    corrected2, raw2, fits2 = tl._process_merged_group(second + ('ratio',))
    np.testing.assert_allclose(corrected1.cell_mean_intensity_ch0, 1300)
    np.testing.assert_allclose(corrected2.cell_mean_intensity_ch0, raw2.cell_mean_intensity_ch0)
    assert corrected2.cell_mean_intensity_ch0.max() == 2800
    assert set(fits1.channel) == {0, 1}
    assert fits1.fieldID.unique().tolist() != fits2.fieldID.unique().tolist()


def test_no_background_pixels_stay_unknown_instead_of_rescaling_camera_offset(tmp_path):
    args = field(tmp_path, occupied=True)
    corrected, raw, fits = tl._process_merged_group(args + ('ratio',))
    assert corrected.cell_mean_intensity_ch0.isna().all()
    assert corrected.cell_mean_intensity.isna().all()
    assert corrected.cell_channel_0_outside_percentile_50.isna().all()
    assert raw.cell_mean_intensity_ch0.notna().all()
    assert 'cell' not in set(fits.object_type)


@pytest.mark.parametrize('bad', [None, True, 'unknown'])
def test_bad_method_fails_before_creating_outputs(tmp_path, bad):
    with pytest.raises(ValueError, match='Unknown motility bleach'):
        tl.automated_motility_assay({'src': str(tmp_path), 'bleach_correction': bad})
    assert list(tmp_path.iterdir()) == []


def test_cached_rows_are_refused_before_database_or_outputs_change(tmp_path):
    field(tmp_path)
    directory = tmp_path / 'measurements'
    directory.mkdir()
    path = directory / 'measurements.db'
    with sqlite3.connect(path) as conn:
        pd.DataFrame({'cellID': [1]}).to_sql('timelapse_object_measurements', conn, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='reuse_existing_measurements=False'):
        tl.automated_motility_assay({'src': str(tmp_path), 'bleach_correction': 'ratio'})
    assert path.read_bytes() == before
    assert not (tmp_path / 'motility_plots').exists()
    assert list(directory.iterdir()) == [path]


def test_real_assay_qc_uses_corrected_values_and_sqlite_keeps_raw(tmp_path, monkeypatch):
    args = field(tmp_path)
    legacy = tl._process_merged_group(args)
    before = snapshot(tmp_path / 'merged')
    seen = []

    def qc(**kwargs):
        seen.append(kwargs['all_df'].copy())
        return kwargs['all_df'], kwargs['infection_col']

    monkeypatch.setattr(tl, '_apply_infection_intensity_qc', qc)
    monkeypatch.setattr(tl, '_debug_plot_merged_planes', lambda **kwargs: None)
    monkeypatch.setattr(tl, '_feature_velocity_correlations', lambda *args: None)
    settings = dict(src=str(tmp_path), channels=[0, 1], cell_channel=0,
                    nucleus_channel=None, pathogen_channel=1, n_jobs=1,
                    bleach_correction='ratio', reuse_existing_measurements=False,
                    make_mask_panel=False, make_adjusted_panel=False,
                    infection_intensity_strategy='none', infection_intensity_qc=False)
    out = tl.automated_motility_assay(settings.copy())
    assert len(out) == len(legacy)
    np.testing.assert_allclose(seen[0].cell_mean_intensity_ch0, 1300)
    path = tmp_path / 'measurements' / 'measurements.db'
    with sqlite3.connect(path) as conn:
        raw = pd.read_sql_query('SELECT * FROM timelapse_object_measurements', conn)
        corrected = pd.read_sql_query('SELECT * FROM timelapse_object_measurements_bleach_corrected', conn)
        fits = pd.read_sql_query('SELECT * FROM timelapse_object_measurements_bleach_fits', conn)
    keys = ['frame', 'cellID']
    pd.testing.assert_frame_equal(raw[legacy.columns].sort_values(keys).reset_index(drop=True),
                                  legacy.sort_values(keys).reset_index(drop=True), check_dtype=False)
    np.testing.assert_allclose(corrected.cell_mean_intensity_ch0, 1300)
    assert len(fits) > 0
    assert (tmp_path / 'measurements' / 'timelapse_object_measurements_bleach_fits.csv').is_file()
    rerun = tl.automated_motility_assay(settings.copy())
    np.testing.assert_allclose(rerun.cell_mean_intensity_ch0, out.cell_mean_intensity_ch0)
    assert snapshot(tmp_path / 'merged') == before


def test_geometric_smoothing_does_not_invent_a_missing_corrected_background():
    frame = pd.DataFrame(dict(plateID=['p'] * 3, wellID=['A01'] * 3,
                              fieldID=[1] * 3, cellID=[2] * 3, frame=[0, 1, 2],
                              cell_mean_intensity=[100., np.nan, 100.],
                              **{'cell_centroid-0': [0., 1000., 0.],
                                 'cell_centroid-1': [0., 1000., 0.]}))
    result = tl._motility_smooth_corrected(frame, 50., 3.)
    assert result['cell_centroid-0'].tolist() == [0., 0., 0.]
    assert np.isnan(result.cell_mean_intensity.iloc[1])
    np.testing.assert_allclose(result.cell_mean_intensity.iloc[[0, 2]], [100., 100.])
