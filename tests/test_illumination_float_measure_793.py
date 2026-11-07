"""Item 793: Measure divides by the flat field in float32 and never clips.

Before the fix the corrector ran in the acquisition dtype, so a bright cell
under a gain above 1 saturated at 65535 ("illumination correction clipped N
pixel(s) ... at the top of its dtype") and lost real signal.
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from spacr import illumination as ill
from spacr import measure_hooks as mh


@pytest.fixture(autouse=True)
def _pristine(monkeypatch):
    """No hook or illumination environment leaks into or out of a test."""
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR,
                 'SPACR_START_METHOD'):
        monkeypatch.delenv(name, raising=False)
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR):
        os.environ.pop(name, None)


def _model(flat, plate='plate1'):
    """A model over one channel with the given flat field."""
    field = ill.IlluminationField(
        plate=plate, channels=(0, 1),
        flatfield=np.stack([flat, flat]).astype(np.float32),
        dark=np.zeros(2, dtype=np.float32),
        n_fields=1, estimator='polynomial', degree=4, bin_size=1)
    return ill.IlluminationModel(fields={plate: field}, meta={})


def test_the_corrector_asks_measure_for_float_input():
    assert ill.IlluminationCorrector.measures_in_float is True


def test_raw_saturated_fraction_reads_the_acquired_integer_pixels():
    from spacr.measure import _raw_saturated_fractions

    raw = np.zeros((2, 2, 2), np.uint16)
    raw[0, 0, 1] = 65535
    assert _raw_saturated_fractions(raw) == {
        'channel_0_saturated_fraction': 0.0,
        'channel_1_saturated_fraction': 0.25}
    assert _raw_saturated_fractions(raw.astype(np.float32)) == {}


def test_measure_crop_reports_zero_clipped_pixels_and_keeps_bright_signal(
        tmp_path, monkeypatch, capsys):
    """A cell pushed past 65535 by the gain keeps its full corrected value."""
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    shape = (64, 64)
    flat = np.ones(shape, np.float32)
    flat[:, :32] = 0.5                   # gain 2 on the left half
    run = tmp_path / 'run'
    merged = run / 'merged'
    os.makedirs(merged)
    (run / 'measurements').mkdir(parents=True)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    cell = np.zeros(shape, np.uint16)
    for label, (cy, cx) in enumerate([(32, 14), (32, 50)], start=1):
        cell[(yy - cy) ** 2 + (xx - cx) ** 2 <= 8 ** 2] = label
    nucleus = np.where((yy - 32) ** 2 + (xx - 14) ** 2 <= 9, 1, 0) + \
        np.where((yy - 32) ** 2 + (xx - 50) ** 2 <= 9, 2, 0)
    plane = np.full(shape, 200, np.float64)
    plane[cell > 0] = 40001.0
    plane[33, 14] = 65535.0              # one pixel the camera saturated
    observed = plane.astype(np.uint16)
    data = np.stack([observed, observed, cell, nucleus.astype(np.uint16),
                     np.zeros(shape, np.uint16)], axis=-1).astype(np.uint16)
    np.save(merged / 'plate1_A01_F001.npy', data)

    settings = get_measure_crop_settings(settings={})
    settings.update({
        'src': str(merged), 'channels': [0, 1],
        'cell_mask_dim': 2, 'nucleus_mask_dim': 3, 'pathogen_mask_dim': None,
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        'experiment': 'exp', 'n_jobs': 1, 'test_mode': False,
        'cytoplasm': False, 'homogeneity': False, 'radial_dist': False,
        'calculate_correlation': False,
        'cell_min_size': 0, 'nucleus_min_size': 0, 'pathogen_min_size': 0,
    })
    ill.enable_illumination_correction(
        _model(flat), path=str(tmp_path / 'illumination' / 'model.npz'),
        verbose=False)
    monkeypatch.setenv('SPACR_START_METHOD', 'fork')

    measure_crop(settings)

    assert 'clipped' not in capsys.readouterr().out
    connection = sqlite3.connect(str(run / 'measurements' / 'measurements.db'))
    try:
        means = sorted(row[0] for row in connection.execute(
            'SELECT cell_channel_0_mean_intensity FROM cell'))
        maxima = sorted(row[0] for row in connection.execute(
            'SELECT cell_channel_0_max_intensity FROM cell'))
        saturated = connection.execute(
            'SELECT channel_0_saturated_fraction FROM image').fetchone()[0]
    finally:
        connection.close()

    # The left cell is 40001 * 2 = 80002 in float, well above a uint16; the
    # one camera-saturated pixel becomes 131070. Nothing is held at 65535.
    in_left = cell == 1
    expected_left = float((plane[in_left] * 2.0).mean())
    assert means[0] == pytest.approx(40001.0, rel=1e-6)
    assert means[1] == pytest.approx(expected_left, rel=1e-6)
    assert maxima[1] == pytest.approx(131070.0, rel=1e-6)
    # Saturation is read from the acquired pixels: one of 64 * 64.
    assert saturated == pytest.approx(1.0 / (64 * 64))

