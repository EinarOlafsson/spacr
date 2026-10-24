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
