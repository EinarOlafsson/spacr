"""Measure resolves a plate's output subfolder to its merged folder (issue 136).

The report: ``src`` pointed at the plate's measure output folder, measure
appended ``merged`` to it, and the run died in ``_listdir_visible`` with a
raw FileNotFoundError that was then filed automatically as a bug.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr.errors import ConfigurationError
from spacr.measure import _measure_merged_folder
from spacr.validate import ERROR, WARNING, _resolve_measure_src, validate_settings


@pytest.fixture
def plate(tmp_path):
    """A plate whose merged folder holds one array."""
    root = tmp_path / "plate1"
    (root / "merged").mkdir(parents=True)
    np.save(root / "merged" / "a.npy", np.zeros((4, 4, 2), dtype=np.uint16))
    return root


@pytest.mark.parametrize("sub", ["measure", "measurements", "masks"])
@pytest.mark.parametrize("exists", [True, False])
def test_an_output_subfolder_resolves_to_the_merged_folder(plate, sub, exists, capsys):
    src = plate / sub
    if exists:
        src.mkdir()

    merged, note = _resolve_measure_src(str(src))

    assert merged == str(plate / "merged")
    assert sub in note and str(plate / "merged") in note
    assert _measure_merged_folder(str(src)) == str(plate / "merged")
    assert "[measure]" in capsys.readouterr().out


@pytest.mark.parametrize("which", ["root", "merged"])
def test_the_plate_root_and_merged_folder_are_read_as_before(plate, which):
    src = plate if which == "root" else plate / "merged"

    merged, note = _resolve_measure_src(str(src))

    assert merged == str(plate / "merged")
    assert note is None


def test_a_missing_src_is_a_configuration_error_with_a_fix(tmp_path):
    missing = tmp_path / "nowhere" / "plate9"

    with pytest.raises(ConfigurationError) as caught:
        _measure_merged_folder(str(missing))

    assert "src does not exist" in str(caught.value)
    assert "plate folder" in str(caught.value)


def test_an_existing_folder_without_merged_says_what_to_set(tmp_path):
    (tmp_path / "other").mkdir()

    with pytest.raises(ConfigurationError) as caught:
        _measure_merged_folder(str(tmp_path / "other"))

    assert "no merged folder" in str(caught.value)


def test_preflight_warns_about_the_resolution_instead_of_erroring(plate):
    problems = validate_settings({"src": str(plate / "measure")}, "measure")

    src_problems = [p for p in problems if p.setting == "src"]
    assert any(p.severity == WARNING and "merged" in p.message for p in src_problems)
    assert not any(p.severity == ERROR for p in src_problems)


def test_preflight_still_errors_on_a_path_that_resolves_nowhere(tmp_path):
    problems = validate_settings({"src": str(tmp_path / "gone")}, "measure")

    assert any(p.severity == ERROR and p.setting == "src"
               and "does not exist" in p.message for p in problems)


def test_a_settings_path_error_is_not_filed_automatically():
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    check = AppScreen._is_a_settings_path_error
    src = os.path.join(os.sep, "data", "plate1", "measure")
    missing = (
        "Traceback (most recent call last):\n  File \"x\", line 1\n"
        f"FileNotFoundError: [Errno 2] No such file or directory: "
        f"'{os.path.join(src, 'merged')}'\n")

    assert check(missing, {"src": src}) is True
    assert check(missing, {"src": os.path.join(os.sep, "elsewhere")}) is False
    assert check("spacr.errors.ConfigurationError: Measure cannot start", {}) is True
    assert check("ValueError: boom", {"src": src}) is False


def test_missing_sources_skip_nonpath_values_and_keep_resolved_plate_sources(plate, tmp_path, monkeypatch):
    from spacr import measure, validate

    missing = str(tmp_path / 'gone' / 'plate2')
    existing = tmp_path / 'existing empty folder'
    existing.mkdir()
    alias = str(plate / 'measure')
    sources = [None, 123, plate / 'merged', alias, missing, str(existing)]
    before = list(sources)
    resolved = []
    actual_resolve = validate._resolve_measure_src

    def resolve(path):
        assert isinstance(path, str), 'Nonstring settings must not reach filesystem resolution'
        resolved.append(path)
        return actual_resolve(path)

    monkeypatch.setattr(validate, '_resolve_measure_src', resolve)
    assert measure._missing_measure_sources(sources) == [missing]
    assert resolved == [alias, missing, str(existing)]
    assert sources == before
    assert (plate / 'merged' / 'a.npy').is_file()


def _segmentation_coordinate_record(root, *, calibrated=False, complete=True):
    """Write the actual application schema beside real merged source files."""
    import json
    from pathlib import Path
    from spacr.psf_pipeline import _configuration

    root = Path(root)
    configuration = _configuration(None, [0, 1], 'v2')
    if calibrated:
        configuration.update(coordinate_system='calibrated_reference_canvas',
            measurement_intensity_source='nonquantitative calibrated display; measurements unsupported')
    path = root / 'psf' / 'segmentation_application.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(configuration=configuration, complete=complete,
                  fields=sorted(p.stem for p in (root / 'merged').glob('*.npy')),
                  archives={})
    path.write_text(json.dumps(record), encoding='utf-8')
    return path


@pytest.mark.parametrize('source,complete,marker', [
    ('root', True, 'coordinate'), ('merged', False, 'coordinate'),
    ('measurements', True, 'null'), ('root', False, 'empty'),
])
def test_persisted_calibrated_coordinates_refuse_before_any_measure_work(
        plate, source, complete, marker, monkeypatch):
    import json
    import spacr.measure as measure

    path = _segmentation_coordinate_record(plate, calibrated=marker == 'coordinate', complete=complete)
    if marker != 'coordinate':
        record = json.loads(path.read_text())
        record['configuration']['unmixing'] = {'calibration': None if marker == 'null' else {}}
        path.write_text(json.dumps(record))
    originals = {p: p.read_bytes() for p in plate.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        """No run, writer or field job may start for unsupported coordinates."""
        pytest.fail('Measure entered its run context before coordinate validation')

    monkeypatch.setattr(measure, 'run_context', forbidden)
    src = plate if source == 'root' else plate / source
    with pytest.raises(ConfigurationError, match='not supported by Measure'):
        measure.measure_crop(dict(src=str(src), timelapse=False, unmix=False))
    assert {p: p.read_bytes() for p in plate.rglob('*') if p.is_file()} == originals
    assert not (plate / 'measurements').exists()


def test_all_plate_coordinates_are_validated_before_the_first_output(plate, tmp_path, monkeypatch):
    import spacr.measure as measure

    unsafe = tmp_path / 'calibrated_segmentation'
    (unsafe / 'merged').mkdir(parents=True)
    np.save(unsafe / 'merged' / 'field.npy', np.zeros((4, 4, 2), np.float32))
    _segmentation_coordinate_record(unsafe, calibrated=True, complete=False)

    def forbidden(*args, **kwargs):
        """An unsafe later plate must stop the first plate before any run."""
        pytest.fail('The first plate started before all sources were checked')

    monkeypatch.setattr(measure, 'run_context', forbidden)
    with pytest.raises(ConfigurationError, match='not supported by Measure'):
        measure.measure_crop(dict(src=[str(plate), str(unsafe)], timelapse=False, unmix=False))
    assert not (plate / 'measurements').exists()
    assert not (unsafe / 'measurements').exists()


def test_a_truncated_coordinate_record_is_not_treated_as_legacy(plate):
    from spacr.measure import _validate_measurement_coordinate_source

    path = _segmentation_coordinate_record(plate)
    path.write_text('{"configuration":')
    before = {p: p.read_bytes() for p in plate.rglob('*') if p.is_file()}
    with pytest.raises(ConfigurationError, match='Cannot verify'):
        _validate_measurement_coordinate_source(plate / 'merged')
    assert {p: p.read_bytes() for p in plate.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('recorded', [False, True])
def test_raw_measurement_keeps_original_statistics_with_or_without_legacy_record(tmp_path, recorded):
    import sqlite3
    from pathlib import Path
    from spacr.measure import measure_crop
    from tests.test_measure_hooks import _project, _settings

    root, merged, name = _project(tmp_path)
    if recorded:
        _segmentation_coordinate_record(root)
    source = Path(merged) / name
    original = source.read_bytes()
    measure_crop(_settings(merged, unmix=False))
    with sqlite3.connect(root / 'measurements' / 'measurements.db') as connection:
        means = connection.execute('SELECT cell_channel_0_mean_intensity FROM cell').fetchall()
    assert len(means) == 4
    assert all(row[0] == 1100 for row in means)
    assert source.read_bytes() == original


def test_oversized_coordinate_record_refuses_before_reading_or_starting_measure(plate, monkeypatch):
    from pathlib import Path
    import spacr.measure as measure

    path = _segmentation_coordinate_record(plate)
    with path.open('r+b') as handle:
        handle.truncate(32 * 1024 ** 2 + 1)
    before = path.stat()
    original_open = Path.open

    class UnreadRecord:
        """Delegate the real descriptor while rejecting an unbounded read."""
        def __init__(self, handle):
            """Own this actual sparse metadata file for its context lifetime."""
            self.handle = handle

        def __enter__(self):
            """Expose only the actual descriptor and guarded read protocol."""
            return self

        def __exit__(self, *exc):
            """Close only this test-owned descriptor."""
            self.handle.close()

        def fileno(self):
            """Return the real descriptor for the production size preflight."""
            return self.handle.fileno()

        def read(self, *args):
            """An already oversized record must never be allocated."""
            pytest.fail('Oversized provenance was read before size validation')

    def opened(candidate, *args, **kwargs):
        """Observe only the actual coordinate record, keeping all other I/O real."""
        handle = original_open(candidate, *args, **kwargs)
        return UnreadRecord(handle) if candidate == path else handle

    def forbidden(*args, **kwargs):
        """No run or database may start after a failed metadata preflight."""
        pytest.fail('Measure started despite oversized coordinate provenance')

    monkeypatch.setattr(Path, 'open', opened)
    monkeypatch.setattr(measure, 'run_context', forbidden)
    with pytest.raises(ConfigurationError, match='Cannot verify'):
        measure.measure_crop(dict(src=str(plate), timelapse=False, unmix=False))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert not (plate / 'measurements').exists()


@pytest.fixture
def calibrated_signed_cells(tmp_path, request):
    """Emit a genuine cell session with known signed camera residuals and axes."""
    import json
    from spacr.psf_pipeline import _initialize_calibrated_output, _prepare_segmentation_psf

    spacing = getattr(request, 'param', (.4, .3))
    original = tmp_path / 'original'
    stack = original / 'assembled'
    stack.mkdir(parents=True)
    dataset = tmp_path / 'calibrated_segmentation'
    calibration_path = tmp_path / 'optical_calibration.json'
    calibration = dict(version=1, axes='YX', pixel_size_um=list(spacing), reference_channel=2,
        channels=[dict(channel=channel, shift_to_reference_px=shift, psf=[[1.]],
                       psf_origin_px=[0, 0], psf_pixel_size_um=list(spacing))
                  for channel, shift in ((2, [0, 0]), (5, [1, 1]))])
    calibration_path.write_text(json.dumps(calibration), encoding='utf-8')
    signal = np.zeros((10, 12), np.float32)
    signal[2:8, 2:10] = np.arange(48).reshape(6, 8) + 20
    first = np.stack((signal, np.zeros_like(signal)), axis=-1)
    second = np.stack((signal, np.zeros_like(signal)), axis=-1)
    second[:-1, :-1, 1] = signal[1:, 1:]
    camera = np.zeros((10, 12, 2), np.float32)
    camera[3, 4:6, 0] = [8, 12]
    camera[2, 3:5, 1] = 10
    identities = {}
    for ordinal, well, field in (('000000', 'A01', first), ('000001', 'B01', second),
                                  ('000002', 'C02', camera)):
        filename = 'stack_' + ordinal + '.npy'
        np.save(stack / filename, field)
        identity = dict(plate='plate9', well=well, field='10', time='2', z='1',
                        source_dataset=str(original))
        identities[filename] = identity
        identities[ordinal] = identity
    settings = dict(unmix=True, unmix_controls='2:A01; 5:B01',
                    unmix_background_percentile=0., unmix_calibration=str(calibration_path),
                    unmix_pixel_size_um=list(spacing))

    def load(path):
        """Read the actual control files through the supported mapped reader."""
        return np.load(path, mmap_mode='r', allow_pickle=False)

    _initialize_calibrated_output(dataset, settings)
    session = _prepare_segmentation_psf(settings, dataset, [0], pipeline_style='v2',
        stack_dir=stack, load=load, source_channels=(2, 5), field_identities=identities)
    result = session.prepare_calibrated(camera, '000002')
    assert result.source_origin_px == (1, 1)
    assert result.valid_slices == (slice(1, 10), slice(1, 12))
    np.testing.assert_array_equal(result.image[2, 3:5], [[-2., 10.], [2., 10.]])
    labels = np.zeros(result.image.shape[:2], np.uint16)
    labels[2, 3:5] = 42
    mask = session.restore_calibrated(labels, result, '000002')
    display = session.calibrated_display(camera, result)
    output = dataset / 'merged' / 'stack_000002.npy'
    signed = dataset / 'signed' / 'stack_000002.npy'
    output.parent.mkdir()
    signed.parent.mkdir()
    np.save(output, np.concatenate((display, mask[..., None]), axis=-1).astype(np.float32))
    np.save(signed, result.image)
    session.mark_completed('000002', output_path=output, signed_path=signed, mask_role='cell')
    session.finish({'000002'})
    sources = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    return dict(root=dataset, sources=sources, original=original,
                record=dataset / 'psf' / 'segmentation_application.json')


def test_signed_cell_measurement_uses_original_signed_pixels_and_full_reference_geometry(
        calibrated_signed_cells, tmp_path):
    import csv
    import hashlib
    import json
    from pathlib import Path
    from spacr.measure import measure_calibrated_cells

    sample = calibrated_signed_cells
    destination = tmp_path / 'signed_measurements'
    result = measure_calibrated_cells(sample['root'], dst=destination, channels=[5, 2])
    assert result == dict(output_dir=str(destination), csv_path=str(destination / 'cells.csv'),
                         provenance_path=str(destination / 'provenance.json'), fields=1, rows=2)
    with (destination / 'cells.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == (
            'source_dataset,plateID,wellID,fieldID,timeID,sliceID,measurement_field_id,label_id,'
            'physical_channel_id,area_px,area_um2,centroid_y_px,centroid_x_px,centroid_y_um,centroid_x_um,'
            'bbox_min_y_px,bbox_min_x_px,bbox_max_y_px,bbox_max_x_px,bbox_min_y_um,bbox_min_x_um,'
            'bbox_max_y_um,bbox_max_x_um,intensity_mean,intensity_min,intensity_max,intensity_sum,intensity_std'
        ).split(',')
        rows = list(reader)
    assert [row['physical_channel_id'] for row in rows] == ['5', '2']
    assert len(rows) == 2
    for row in rows:
        assert {key: row[key] for key in ('source_dataset', 'plateID', 'wellID', 'fieldID',
                                         'timeID', 'sliceID', 'measurement_field_id', 'label_id')} == dict(
            source_dataset=str(sample['original']), plateID='plate9', wellID='C02', fieldID='10',
            timeID='2', sliceID='1', measurement_field_id='000002', label_id='42')
        assert int(row['area_px']) == 2
        assert float(row['area_um2']) == pytest.approx(.24)
        expected = dict(centroid_y_px=3., centroid_x_px=4.5, centroid_y_um=1.2, centroid_x_um=1.35,
                        bbox_min_y_px=3., bbox_min_x_px=4., bbox_max_y_px=4., bbox_max_x_px=6.,
                        bbox_min_y_um=1.2, bbox_min_x_um=1.2, bbox_max_y_um=1.6, bbox_max_x_um=1.8)
        assert {key: float(row[key]) for key in expected} == pytest.approx(expected)
    assert {key: float(rows[0][key]) for key in
            ('intensity_mean', 'intensity_min', 'intensity_max', 'intensity_sum', 'intensity_std')} == dict(
        intensity_mean=10., intensity_min=10., intensity_max=10., intensity_sum=20., intensity_std=0.)
    assert {key: float(rows[1][key]) for key in
            ('intensity_mean', 'intensity_min', 'intensity_max', 'intensity_sum', 'intensity_std')} == dict(
        intensity_mean=0., intensity_min=-2., intensity_max=2., intensity_sum=0., intensity_std=2.)
    assert set(path.name for path in destination.iterdir()) == {'cells.csv', 'provenance.json'}
    assert all(path.read_bytes() == before for path, before in sample['sources'].items())
    assert not (sample['root'] / 'measurements').exists()
    metadata = json.loads(Path(result['provenance_path']).read_text(encoding='utf-8'))
    assert {key: metadata[key] for key in ('schema_version', 'alpha_feature', 'object_type',
            'source_dataset', 'physical_channel_order', 'pixel_size_um', 'coordinate_system',
            'intensity_unit', 'standard_deviation_ddof', 'fields', 'rows')} == dict(
        schema_version=1, alpha_feature=538, object_type='cell', source_dataset=str(sample['root']),
        physical_channel_order=[5, 2], pixel_size_um=[.4, .3],
        coordinate_system='calibrated_reference_canvas', intensity_unit='camera_linear_common_psf',
        standard_deviation_ddof=0, fields=1, rows=2)
    assert metadata['source_provenance'] == json.loads(sample['record'].read_text(encoding='utf-8'))
    assert metadata['source_record_sha256'] == hashlib.sha256(sample['record'].read_bytes()).hexdigest()
    assert metadata['csv_sha256'] == hashlib.sha256((destination / 'cells.csv').read_bytes()).hexdigest()
    for key in ('coordinate_convention', 'intensity_description', 'sum_definition'):
        assert isinstance(metadata[key], str) and metadata[key].strip()


def test_signed_cell_measurement_cancel_after_actual_csv_write_publishes_nothing(
        calibrated_signed_cells, tmp_path, monkeypatch):
    import threading
    from pathlib import Path
    from spacr import tabular
    from spacr.measure import measure_calibrated_cells

    sample = calibrated_signed_cells
    destination = tmp_path / 'cancelled_measurements'
    before = set(tmp_path.iterdir())
    cancelled = threading.Event()
    written = []
    write = tabular.write_table

    def write_then_cancel(frame, path, **kwargs):
        """Cancel only after the real CSV has been fully written in staging."""
        result = write(frame, path, **kwargs)
        assert Path(path).is_file()
        written.append(Path(path))
        cancelled.set()
        return result

    monkeypatch.setattr(tabular, 'write_table', write_then_cancel)
    with pytest.raises(InterruptedError):
        measure_calibrated_cells(sample['root'], dst=destination, cancelled=cancelled.is_set)
    assert len(written) == 1
    assert not destination.exists()
    assert set(tmp_path.iterdir()) == before
    assert all(path.read_bytes() == original for path, original in sample['sources'].items())


def test_signed_cell_measurement_cancels_after_real_statistics_without_publication(
        calibrated_signed_cells, tmp_path, monkeypatch):
    import threading
    import spacr.measure as measure

    sample = calibrated_signed_cells
    destination = tmp_path / 'cancelled_statistics'
    before = set(tmp_path.iterdir())
    cancelled = threading.Event()
    observed = []
    statistics = measure._calibrated_cell_statistics

    def statistics_then_cancel(*args):
        """Request cancellation after the first actual signed cell statistic."""
        for row in statistics(*args):
            observed.append((row['label_id'], row['physical_channel_id'], row['intensity_sum']))
            yield row
            cancelled.set()

    monkeypatch.setattr(measure, '_calibrated_cell_statistics', statistics_then_cancel)
    with pytest.raises(InterruptedError):
        measure.measure_calibrated_cells(sample['root'], dst=destination,
                                         cancelled=cancelled.is_set)
    assert observed == [(42, 2, 0.)]
    assert set(tmp_path.iterdir()) == before
    assert all(path.read_bytes() == original for path, original in sample['sources'].items())


def test_signed_cell_measurement_preserves_actual_output_write_error(
        calibrated_signed_cells, tmp_path, monkeypatch):
    import errno
    from pathlib import Path
    from spacr import tabular
    from spacr.measure import measure_calibrated_cells

    sample = calibrated_signed_cells
    destination = tmp_path / 'failed_csv'
    before = set(tmp_path.iterdir())
    write = tabular.write_table
    failure = OSError(errno.ENOSPC, 'storage exhausted')
    written = []

    def write_then_fail(frame, path, **kwargs):
        """Preserve the real storage exception after an actual staged write."""
        write(frame, path, **kwargs)
        assert Path(path).is_file()
        written.append(path)
        raise failure

    monkeypatch.setattr(tabular, 'write_table', write_then_fail)
    with pytest.raises(OSError) as caught:
        measure_calibrated_cells(sample['root'], dst=destination)
    assert caught.value is failure
    assert len(written) == 1
    assert set(tmp_path.iterdir()) == before
    assert all(path.read_bytes() == original for path, original in sample['sources'].items())


@pytest.mark.parametrize('kind', ['directory', 'symlink'])
def test_signed_cell_measurement_never_replaces_an_existing_destination(
        calibrated_signed_cells, tmp_path, kind):
    from spacr.measure import measure_calibrated_cells

    sample = calibrated_signed_cells
    destination = tmp_path / 'existing_measurements'
    target = tmp_path / 'existing_target'
    if kind == 'directory':
        destination.mkdir()
    else:
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)
    identity = destination.lstat()
    before = set(tmp_path.iterdir())
    with pytest.raises(FileExistsError):
        measure_calibrated_cells(sample['root'], dst=destination)
    current = destination.lstat()
    assert (current.st_dev, current.st_ino, current.st_mode) == (
        identity.st_dev, identity.st_ino, identity.st_mode)
    assert not list(destination.iterdir())
    assert destination.is_symlink() == (kind == 'symlink')
    if kind == 'symlink':
        assert destination.resolve() == target
    assert set(tmp_path.iterdir()) == before
    assert all(path.read_bytes() == original for path, original in sample['sources'].items())



def test_signed_cell_measurement_preserves_a_destination_created_during_staging(
        calibrated_signed_cells, tmp_path, monkeypatch):
    from pathlib import Path
    from spacr import tabular
    from spacr.measure import measure_calibrated_cells

    sample = calibrated_signed_cells
    destination = tmp_path / 'raced_measurements'
    before = set(tmp_path.iterdir())
    write = tabular.write_table
    created = []

    def write_then_create_destination(frame, path, **kwargs):
        """Let another writer claim the destination after the real staged CSV."""
        result = write(frame, path, **kwargs)
        assert Path(path).is_file()
        assert not destination.exists()
        destination.mkdir()
        created.append(destination.stat())
        return result

    monkeypatch.setattr(tabular, 'write_table', write_then_create_destination)
    with pytest.raises(FileExistsError):
        measure_calibrated_cells(sample['root'], dst=destination)
    assert len(created) == 1
    current = destination.stat()
    assert (current.st_dev, current.st_ino, current.st_mode) == (
        created[0].st_dev, created[0].st_ino, created[0].st_mode)
    assert destination.is_dir() and not destination.is_symlink()
    assert not list(destination.iterdir())
    assert set(tmp_path.iterdir()) == before | {destination}
    assert all(path.read_bytes() == original for path, original in sample['sources'].items())


@pytest.mark.parametrize('calibrated_signed_cells', [(1e-200, 1e-200), (1e308, 1e308)],
                         indirect=True, ids=['underflow-area', 'overflow-geometry'])
def test_signed_cell_measurement_refuses_unrepresentable_physical_geometry(
        calibrated_signed_cells, tmp_path):
    import json
    from spacr.measure import measure_calibrated_cells

    sample = calibrated_signed_cells
    spacing = json.loads(sample['record'].read_text())['configuration']['unmixing']['calibration']['pixel_size_um']
    assert len(spacing) == 2 and all(np.isfinite(value) and value > 0 for value in spacing)
    destination = tmp_path / 'unrepresentable_measurements'
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match='positive finite pixel areas and finite measurement coordinates'):
        measure_calibrated_cells(sample['root'], dst=destination)
    assert not destination.exists()
    assert set(tmp_path.iterdir()) == before
    assert all(path.read_bytes() == original for path, original in sample['sources'].items())
