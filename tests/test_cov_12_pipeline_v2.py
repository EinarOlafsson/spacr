"""V2 streaming on the inputs that do not match the happy shape.

The V2 pipeline writes one npy per field and appends the mask channel back into
it, so anything it guesses wrong is baked into the file rather than reported.
These cover the guesses: a mapping reloaded without its sidecar, a control
header changing around its pixel read, and the
provenance recorder failing in each of the ways it is written to survive.
"""
from __future__ import annotations

import types
from pathlib import Path

import os

import numpy as np
import pytest

from spacr import pipeline_v2 as PV


@pytest.mark.parametrize('moment', ['before-read', 'during-read'])
def test_control_header_identity_is_rechecked_around_pixel_loading(tmp_path, monkeypatch, moment):
    import tifffile

    path = tmp_path / 'control.tif'
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
    tifffile.imwrite(path, pixels)
    header = PV._partition_plane_header(path)
    record = PV.FilenameRecord(str(path), 'plate1', 'A01', 1, 0)
    original_read = PV._read_plane
    reads = []

    def rewrite():
        tifffile.imwrite(path, pixels + 1)
        stamp = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(stamp, stamp))

    def mutate_during_read(*args, **kwargs):
        reads.append(args[0])
        value = original_read(*args, **kwargs)
        rewrite()
        return value

    if moment == 'before-read':
        rewrite()
    monkeypatch.setattr(PV, '_read_plane', mutate_during_read)
    with pytest.raises(ValueError):
        PV._partition_read_control([record], (0,), {str(path): header})
    assert reads == ([] if moment == 'before-read' else [str(path)])
    np.testing.assert_array_equal(tifffile.imread(path), pixels + 1)


def make_plate(dst: Path, channels=3, size=12) -> Path:
    import tifffile

    plate = dst / 'plate1'
    plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for field in (1, 2):
        for channel in range(channels):
            arr = rng.integers(0, 2000, size=(size, size)).astype(np.uint16)
            tifffile.imwrite(
                str(plate / f'plate1_A01_T01F0{field}L01A01Z01C0{channel}.tif'),
                arr)
    return plate


def test_a_mapping_reloaded_without_its_sidecar_says_the_regex_is_unknown(
        tmp_path):
    """A CSV with no companion JSON loads with placeholder provenance.

    The sidecar is what ``spacr repro`` replays; when it is missing the mapping
    must still load -- the file names are the data -- but it must not claim a
    metadata type or a regex it cannot know.
    """
    plate = make_plate(tmp_path)
    mapper = PV.FilenameMapper.discover(plate, metadata_type='cellvoyager')
    csv_path = mapper.save_csv(tmp_path / 'mapping.csv')
    csv_path.with_suffix('.json').unlink()

    reloaded = PV.FilenameMapper.load_csv(csv_path)
    assert reloaded.metadata_type == '?'
    assert reloaded.regex == ''
    assert len(reloaded.records) == len(mapper.records)


class TestCheckpointProvenance:
    """Every way the model fingerprint can fail is still only a warning."""


    def test_a_journal_that_refuses_the_record_does_not_stop_segmentation(
            self, tmp_path, monkeypatch, caplog):
        """A raising run journal is logged and stepped over, not propagated.

        Provenance is not the result. Losing the whole segmentation because the
        manifest could not be written would be the expensive half failing for
        the cheap half.
        """
        import spacr.run_journal as rj

        ckpt = tmp_path / 'model.pth'
        ckpt.write_bytes(b'weights')

        def boom():
            raise RuntimeError('journal is closed')

        monkeypatch.setattr(rj, 'current_run', boom)
        with caplog.at_level('WARNING', logger='spacr.pipeline_v2'):
            PV._record_cellpose_hash(
                types.SimpleNamespace(pretrained_model=[str(ckpt)]), 'cyto')

        assert any('was not recorded in the run journal' in
                   record.getMessage() for record in caplog.records)

    def test_a_model_that_cannot_be_inspected_is_logged_not_raised(
            self, caplog):
        """An attribute lookup that raises leaves the run without provenance.

        A wrapped or lazily-loaded Cellpose model can raise from
        ``pretrained_model``; that must cost the manifest entry and nothing
        else.
        """
        class HostileModel:
            @property
            def pretrained_model(self):
                raise RuntimeError('model is not loaded')

        with caplog.at_level('WARNING', logger='spacr.pipeline_v2'):
            PV._record_cellpose_hash(HostileModel(), 'cyto')

        assert any('could not work out which checkpoint' in
                   record.getMessage() for record in caplog.records)


