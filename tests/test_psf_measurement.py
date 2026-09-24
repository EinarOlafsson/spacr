"""Quantitative PSF selection, durable provenance, and worker delivery."""
import json
from pathlib import Path
import sqlite3
import threading
import multiprocessing as mp

import numpy as np
import pytest

from spacr.psf_measurement import (
    prepare_measurement_psf, measurement_psf_signature, measurement_resume_settings,
    validate_measurement_psf_history, SIGNATURE_KEY,
)
from tests.test_measure_hooks import _project, _settings


def _config(merged, **overrides):
    return _settings(merged, psf_measurement_source='processed',
                     psf_operation='convolve', psf_image_sampling_um=[.5, .5],
                     psf_fwhm_um=[1.5, 1.5], **overrides)


def _row(project):
    with sqlite3.connect(project / 'measurements/measurements.db') as db:
        row = db.execute('SELECT psf_measurement_source, psf_signature, psf_provenance '
                         'FROM intensity_rescale').fetchone()
    return row[0], row[1], json.loads(row[2])


@pytest.mark.parametrize('operation', ['convolve', 'deconvolve'])
def test_actual_measurements_use_float_processed_intensities_and_preserve_source(tmp_path, operation):
    from spacr.measure import _measure_crop_core
    project, merged, name = _project(tmp_path)
    settings = _config(merged)
    settings.update(psf_operation=operation, psf_iterations=3)
    path = Path(merged) / name
    source = path.read_bytes()
    data = np.load(path)
    plan = prepare_measurement_psf(settings)
    expected = plan.apply(data[..., [0, 1]])
    result = _measure_crop_core(0, [], name, settings, plan)
    assert not result[4], result[4]
    with sqlite3.connect(project / 'measurements/measurements.db') as db:
        rows = db.execute('SELECT object_label, cell_channel_0_mean_intensity '
                          'FROM cell ORDER BY object_label').fetchall()
    assert rows
    for label, value in rows:
        assert value == pytest.approx(expected[..., 0][data[..., 2] == label].mean(), rel=1e-6)
    assert path.read_bytes() == source
    kind, signature, provenance = _row(project)
    assert kind == 'processed'
    assert signature == measurement_psf_signature(plan)
    assert provenance['processing']['operation'] == operation
    assert provenance['output_dtype'] == 'float32'
    assert 'PSF and preprocessing hooks not applied' in provenance['crop_intensity_source']


def test_original_choice_ignores_dormant_kernel_parameters_and_measures_standard_pixels(tmp_path):
    from spacr.measure import _measure_crop_core
    project, merged, name = _project(tmp_path)
    settings = _config(merged)
    settings.update(psf_measurement_source='original', psf_source='measured',
                    psf_path='/missing', psf_image_sampling_um=None)
    assert prepare_measurement_psf(settings) is None
    result = _measure_crop_core(0, [], name, settings)
    assert not result[4]
    kind, signature, provenance = _row(project)
    assert kind == 'original' and signature is None and provenance['processing'] is None
    with sqlite3.connect(project / 'measurements/measurements.db') as db:
        values = db.execute('SELECT cell_channel_0_mean_intensity FROM cell').fetchall()
    assert values and all(value == 1100 for (value,) in values)


def test_measured_kernel_is_not_reopened_by_field_worker(tmp_path):
    from spacr.measure import _measure_crop_core
    project, merged, name = _project(tmp_path)
    kernel = tmp_path / 'psf.npy'
    np.save(kernel, np.ones((3, 3)))
    settings = _config(merged)
    settings.update(psf_source='measured', psf_path=str(kernel),
                    psf_kernel_sampling_um=[.5, .5])
    plan = prepare_measurement_psf(settings)
    kernel.unlink()
    result = _measure_crop_core(0, [], name, settings, plan)
    assert not result[4], result[4]
    assert _row(project)[1] == measurement_psf_signature(plan)


def test_processed_requires_an_actual_operation():
    with pytest.raises(ValueError, match='require psf_operation'):
        prepare_measurement_psf({'psf_measurement_source': 'processed'})
    with pytest.raises(ValueError, match='original or processed'):
        prepare_measurement_psf({'psf_measurement_source': 'maybe'})


def test_original_resume_ignores_new_dormant_defaults_but_processed_identity_is_material():
    from spacr.resume import compare_settings
    old = {'channels': [0]}
    current = {**old, 'psf_source': 'gaussian', 'psf_iterations': 20,
               'psf_measurement_source': 'original'}
    assert not compare_settings(measurement_resume_settings(old, recorded=True),
                                measurement_resume_settings(current)).blocks_resume
    current.update(psf_measurement_source='processed', psf_operation='convolve',
                   psf_image_sampling_um=[1., 1.], psf_fwhm_um=[2., 2.])
    assert compare_settings(measurement_resume_settings(old, recorded=True),
                            measurement_resume_settings(current)).blocks_resume
    processed = measurement_resume_settings(current)
    assert compare_settings(processed, measurement_resume_settings(old)).blocks_resume


@pytest.mark.parametrize('change', ['source', 'kernel', 'iterations'])
def test_existing_database_refuses_changed_processing_even_without_resume(tmp_path, change):
    from spacr.measure import _measure_crop_core
    project, merged, name = _project(tmp_path)
    settings = _config(merged)
    settings.update(psf_operation='deconvolve', psf_iterations=2)
    plan = prepare_measurement_psf(settings)
    result = _measure_crop_core(0, [], name, settings, plan)
    assert not result[4]
    db = project / 'measurements/measurements.db'
    validate_measurement_psf_history(settings, db, plan)
    changed = dict(settings)
    if change == 'source':
        changed['psf_measurement_source'] = 'original'
    elif change == 'kernel':
        changed['psf_fwhm_um'] = [2., 2.]
    else:
        changed['psf_iterations'] = 4
    with pytest.raises(ValueError, match='different PSF intensity'):
        validate_measurement_psf_history(changed, db, prepare_measurement_psf(changed))


def test_legacy_database_requires_original_until_a_separate_processed_run(tmp_path):
    db = tmp_path / 'measurements.db'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE cell (object_label INTEGER)')
        connection.execute('INSERT INTO cell VALUES (1)')
    validate_measurement_psf_history({}, db, None)
    plan = prepare_measurement_psf(_config(str(tmp_path)))
    with pytest.raises(ValueError, match='different PSF intensity'):
        validate_measurement_psf_history({}, db, plan)


def test_crop_only_project_does_not_block_its_first_quantitative_psf_run(tmp_path):
    db = tmp_path / 'measurements.db'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE png_list (png_path TEXT)')
        connection.execute("INSERT INTO png_list VALUES ('cell.png')")
        connection.execute('CREATE TABLE intensity_rescale (rescale_factor REAL)')
        connection.execute('INSERT INTO intensity_rescale VALUES (1)')
    validate_measurement_psf_history({}, db, prepare_measurement_psf(_config(str(tmp_path))))


def test_parent_stop_reaches_background_worker_event():
    from spacr.cancellation import CancellationToken, installed_token, PipelineCancelled
    from spacr.measure import _wait_for_measure_job
    cancelled = threading.Event()
    token = CancellationToken()
    token.cancel()

    class Result:
        def get(self, timeout):
            assert timeout == .2
            if cancelled.is_set():
                raise PipelineCancelled('worker stopped')
            raise mp.TimeoutError()

    with installed_token(token), pytest.raises(PipelineCancelled, match='worker stopped'):
        _wait_for_measure_job(Result(), cancelled)


def test_stop_does_not_wait_forever_for_a_worker_that_never_answers(monkeypatch):
    from spacr.cancellation import CancellationToken, installed_token, PipelineCancelled
    import spacr.measure as measure
    event = threading.Event()
    token = CancellationToken()
    token.cancel()
    times = iter((100., 105.))
    monkeypatch.setattr(measure.time, 'monotonic', lambda: next(times))

    class SilentWorker:
        def get(self, timeout):
            raise mp.TimeoutError()

    with installed_token(token), pytest.raises(PipelineCancelled):
        measure._wait_for_measure_job(SilentWorker(), event)
    assert event.is_set()


def test_cancelled_field_writes_no_measurements(tmp_path):
    from spacr.measure import _measure_crop_core
    from spacr.cancellation import PipelineCancelled
    project, merged, name = _project(tmp_path)
    settings = _config(merged)
    event = threading.Event()
    event.set()
    with pytest.raises(PipelineCancelled):
        _measure_crop_core(0, [], name, settings, prepare_measurement_psf(settings), event)
    assert not (project / 'measurements/measurements.db').exists()


def test_processed_plan_reaches_a_spawned_measure_worker(tmp_path):
    from spacr.measure import _measure_crop_core
    project, merged, name = _project(tmp_path)
    settings = _config(merged)
    context = mp.get_context('spawn')
    with context.Pool(1) as pool:
        result = pool.apply(_measure_crop_core,
                            (0, [], name, settings, prepare_measurement_psf(settings)))
    assert not result[4], result[4]
    assert _row(project)[0] == 'processed'


def _apply_until_cancelled(plan, started, cancelled):
    """Exercise an actual spawned worker receiving a process-safe Stop event."""
    image = np.ones((512, 512, 1), np.float32)
    image[250:260, 250:260] = 1000
    started.set()
    return plan.apply(image, cancel=cancelled)


def test_stop_interrupts_processing_in_a_spawned_worker():
    from spacr.cancellation import PipelineCancelled
    settings = _config('unused')
    settings.update(psf_operation='deconvolve', psf_iterations=200)
    context = mp.get_context('spawn')
    with context.Manager() as manager, context.Pool(1) as pool:
        started, cancelled = manager.Event(), manager.Event()
        result = pool.apply_async(_apply_until_cancelled,
                                  (prepare_measurement_psf(settings), started, cancelled))
        assert started.wait(20)
        timer = threading.Timer(.1, cancelled.set)
        timer.start()
        try:
            with pytest.raises(PipelineCancelled):
                result.get(timeout=10)
        finally:
            timer.join()


def test_same_measured_path_with_new_bytes_cannot_mix_measurements(tmp_path):
    from spacr.measure import _measure_crop_core
    project, merged, name = _project(tmp_path)
    path = tmp_path / 'kernel.npy'
    np.save(path, np.ones((3, 3)))
    settings = _config(merged)
    settings.update(psf_source='measured', psf_path=str(path),
                    psf_kernel_sampling_um=[.5, .5])
    plan = prepare_measurement_psf(settings)
    assert not _measure_crop_core(0, [], name, settings, plan)[4]
    np.save(path, np.eye(3))
    with pytest.raises(ValueError, match='different PSF intensity'):
        validate_measurement_psf_history(settings, project / 'measurements/measurements.db',
                                         prepare_measurement_psf(settings))


def test_preprocessing_hooks_run_before_psf_and_are_named_in_provenance(tmp_path):
    from spacr.measure import _measure_crop_core
    from spacr.measure_hooks import register_preprocessing_hook, unregister_preprocessing_hook
    project, merged, name = _project(tmp_path)
    settings = _config(merged)
    plan = prepare_measurement_psf(settings)
    data = np.load(Path(merged) / name)

    def half_left(image, context):
        corrected = image.copy()
        corrected[:, :corrected.shape[1]//2] //= 2
        return corrected

    register_preprocessing_hook(half_left, name='test calibrated flat field')
    try:
        result = _measure_crop_core(0, [], name, settings, plan)
    finally:
        unregister_preprocessing_hook('test calibrated flat field')
    assert not result[4], result[4]
    processed = plan.apply(half_left(data[..., [0, 1]], None))
    with sqlite3.connect(project / 'measurements/measurements.db') as db:
        rows = db.execute('SELECT object_label, cell_channel_0_mean_intensity FROM cell').fetchall()
    for label, value in rows:
        assert value == pytest.approx(processed[..., 0][data[..., 2] == label].mean(), rel=1e-6)
    record = _row(project)[2]
    assert record['preprocessing_hooks'] == ['test calibrated flat field']
    assert record['channels'] == [0, 1]


def test_full_measure_entry_and_resume_deliver_the_same_captured_configuration(tmp_path, monkeypatch):
    import spacr.measure as measure
    project, merged, name = _project(tmp_path)
    settings = _config(merged)
    monkeypatch.setattr(measure, '_pool_context', lambda: mp.get_context('spawn'))
    measure.measure_crop(dict(settings))
    assert _row(project)[0] == 'processed'
    db_path = project / 'measurements/measurements.db'
    with sqlite3.connect(db_path) as connection:
        before = connection.execute('SELECT COUNT(*) FROM cell').fetchone()[0]
    settings['resume'] = True
    measure.measure_crop(dict(settings))
    with sqlite3.connect(db_path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM cell').fetchone()[0] == before
    settings['resume'] = False
    settings['psf_fwhm_um'] = [2., 2.]
    with pytest.raises(ValueError, match='different PSF intensity'):
        measure.measure_crop(dict(settings))
    with sqlite3.connect(db_path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM cell').fetchone()[0] == before


def test_crops_retain_source_pixels_for_both_intensity_choices(tmp_path):
    from spacr.measure import _measure_crop_core
    captures = []
    for source in ('original', 'processed'):
        project, merged, name = _project(tmp_path / source)
        settings = _config(merged, save_png=True, save_arrays=True)
        settings['psf_measurement_source'] = source
        result = _measure_crop_core(0, [], name, settings)
        assert not result[4], result[4]
        paths = list(project.rglob('*.png')) + list(project.rglob('*.png.npy'))
        assert paths
        captures.append({str(p.relative_to(project)): p.read_bytes() for p in paths})
    assert captures[0] == captures[1]


def test_float_texture_keeps_subinteger_variation():
    from spacr.measure import _calculate_homogeneity
    from skimage.feature import graycomatrix, graycoprops
    from skimage.exposure import rescale_intensity
    image = np.linspace(.1, .9, 64).reshape(8, 8).astype(np.float32)
    labels = np.ones((8, 8), np.uint16)
    expected_image = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    expected = graycoprops(graycomatrix(expected_image, [1], [0],
                                       symmetric=True, normed=True), 'homogeneity')[0, 0]
    actual = _calculate_homogeneity(labels, image, [1])['homogeneity_distance_1'][0]
    assert actual == pytest.approx(expected)
    assert actual < 1


def test_legacy_provenance_table_migrates_without_losing_existing_rows(tmp_path):
    from spacr.measure import _measure_crop_core
    project, merged, name = _project(tmp_path)
    settings = _settings(merged)
    result = _measure_crop_core(0, [], name, settings)
    assert not result[4]
    db = project / 'measurements/measurements.db'
    with sqlite3.connect(db) as connection:
        for column in ('psf_measurement_source', 'psf_signature', 'psf_provenance'):
            connection.execute(f'ALTER TABLE intensity_rescale DROP COLUMN {column}')
        old = connection.execute('SELECT prcf, rescale_factor FROM intensity_rescale').fetchone()
    second = name.replace('F001', 'F002')
    np.save(Path(merged) / second, np.load(Path(merged) / name))
    result = _measure_crop_core(1, [], second, settings)
    assert not result[4]
    with sqlite3.connect(db) as connection:
        rows = connection.execute('SELECT prcf, rescale_factor, psf_measurement_source '
                                  'FROM intensity_rescale ORDER BY prcf').fetchall()
    assert len(rows) == 2 and rows[0][:2] == old
    assert all(row[2] == 'original' for row in rows)


@pytest.mark.parametrize('mismatch', [False, True])
def test_volume_psf_respects_voxel_calibration(tmp_path, mismatch):
    from spacr.measure import _measure_crop_core
    project, merged, name = _project(tmp_path)
    data = np.load(Path(merged) / name)
    volume = np.repeat(data[None], 3, axis=0)
    np.save(Path(merged) / name, volume)
    settings = _config(merged)
    settings.update(psf_image_sampling_um=[1., .5, .5], psf_fwhm_um=[2., 1., 1.],
                    voxel_size_z_um=2. if mismatch else 1., voxel_size_xy_um=.5,
                    distance_gaussian_sigma=0, spatial_measurements=False)
    result = _measure_crop_core(0, [], name, settings)
    if mismatch:
        assert 'PSF sampling conflicts' in result[4]
        assert not (project / 'measurements/measurements.db').exists()
    else:
        assert not result[4], result[4]
        assert _row(project)[2]['processing']['image_sampling_um'] == [1., .5, .5]
