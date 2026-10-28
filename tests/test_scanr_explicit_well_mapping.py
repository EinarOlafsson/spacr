"""Observed indexed ScanR names need supplied physical well identities."""
from pathlib import Path

import numpy as np
import pytest
import tifffile

NAMES = (
    '--W00002--P00001--Z00000--T00000--nucleus-dapi.tif',
    '--W00002--P00001--Z00000--T00000--pm-647.tif',
)


def _images(folder):
    """Use observed names with deliberately synthetic, distinguishable pixels."""
    folder.mkdir(parents=True, exist_ok=True)
    for value, name in enumerate(NAMES, 11):
        tifffile.imwrite(folder / name, np.full((5, 7), value, dtype=np.uint16))
    return {name: (folder / name).read_bytes() for name in NAMES}


def _map(tmp_path, body):
    """Write an explicit test-authored map, never acquired-well evidence."""
    path = tmp_path / 'wells.csv'
    path.write_text('plate,index,well\n' + body, encoding='utf-8')
    return str(path)


def _snapshot(folder):
    """Include empty directories and all bytes in a refusal oracle."""
    return {str(path.relative_to(folder)): None if path.is_dir() else path.read_bytes()
            for path in folder.rglob('*')}


@pytest.mark.parametrize('body', [
    'other,W00002,B07\n',
    'plate,W00003,B07\n',
    'plate,W00002,B07\nplate,W00002,C08\n',
    'plate,W00002,B07\nplate,W00003,B7\n',
    'plate,W00002,not-a-well\n',
    'plate,W00000,B07\n',
    ',W00002,B07\n',
])
def test_indexed_scanr_invalid_maps_never_produce_a_conversion(tmp_path, body):
    """Incomplete, duplicate, ambiguous and malformed maps block all writing."""
    from spacr import convert

    source = tmp_path / 'plate'
    originals = _images(source)
    mapping = _map(tmp_path, body)
    before = _snapshot(tmp_path)
    plan = convert.plan(convert.scan(str(source)), well_map=mapping)
    assert not plan.ok
    assert plan.errors
    assert not plan.mappings
    with pytest.raises(convert.ConfigurationError):
        convert.convert(plan, str(tmp_path / 'output'))
    assert _snapshot(tmp_path) == before
    assert {name: (source / name).read_bytes() for name in NAMES} == originals


def test_indexed_scanr_import_retains_acquisition_index_and_exact_pixels(tmp_path):
    """An explicit B07 assignment is independent of W00002 and folder A01."""
    from spacr import convert

    source = tmp_path / 'plate' / 'A01'
    originals = _images(source)
    root = source.parent
    mapping = _map(tmp_path, 'plate,W00002,B07\n')
    sources = convert.scan(str(root), layout='well')
    assert {item.well for item in sources} == {'W00002'}
    assert {item.channel for item in sources} == {'nucleus-dapi', 'pm-647'}
    assert {item.meta['z_index'] for item in sources} == {1}
    assert {item.meta['t_index'] for item in sources} == {1}
    plan = convert.plan(sources, well_map=mapping)
    assert plan.ok, plan.errors
    assert {(row.source_well, row.well) for row in plan.mappings} == {('W00002', 'B07')}
    result = convert.convert(plan, str(tmp_path / 'converted'))
    assert result.is_complete
    outputs = list((tmp_path / 'converted').rglob('*.tif'))
    assert len(outputs) == 2
    assert sorted(int(tifffile.imread(path)[0, 0]) for path in outputs) == [11, 12]
    assert {name: (source / name).read_bytes() for name in NAMES} == originals


def test_indexed_scanr_existing_mapping_api_keeps_plate_scopes_separate(tmp_path):
    """The same acquisition index can mean different supplied wells per plate."""
    from spacr import convert

    _images(tmp_path / 'one' / 'images')
    _images(tmp_path / 'two' / 'images')
    sources = convert.scan(str(tmp_path), layout='plate_well')
    from types import MappingProxyType
    plan = convert.plan(sources, well_map=MappingProxyType(
        {('one', 'W00002'): 'B07', ('two', 'W00002'): 'C08'}))
    assert plan.ok, plan.errors
    assert {(row.source_plate, row.source_well, row.well) for row in plan.mappings} == {
        ('one', 'W00002', 'B07'), ('two', 'W00002', 'C08')}


def test_indexed_scanr_mask_ingest_and_orig_rebuild_keep_pixels_and_identity(tmp_path):
    """Mask streams both channels, preserves original names and rebuilds orig."""
    from spacr.io import _rename_and_organize_image_files, _rebuild_stacks_from_raw
    from spacr.regex_infer import _metadata_pattern

    source = tmp_path / 'plate'
    originals = _images(source)
    mapping = _map(tmp_path, 'plate,W00002,B07\n')
    assert _rename_and_organize_image_files(
        str(source), _metadata_pattern('scanr_indexed'), metadata_type='scanr_indexed',
        metadata_well_map=mapping) == 2
    stacks = list((source / 'stack').glob('*.npy'))
    assert len(stacks) == 1
    assert '_B07_' in stacks[0].name
    expected = np.stack([np.full((5, 7), value, dtype=np.uint16)
                         for value in (11, 12)], axis=2)
    np.testing.assert_array_equal(np.load(stacks[0]), expected)
    assert {name: (source / 'orig' / name).read_bytes() for name in NAMES} == originals
    stacks[0].unlink()
    assert _rebuild_stacks_from_raw({'metadata_type': 'scanr_indexed',
                                    'metadata_well_map': mapping}, str(source)) == 1
    np.testing.assert_array_equal(np.load(stacks[0]), expected)
    assert {name: (source / 'orig' / name).read_bytes() for name in NAMES} == originals


@pytest.mark.parametrize('test_mode', [False, True])
def test_indexed_scanr_late_orig_missing_map_refuses_before_resume_or_sampling(tmp_path, test_mode):
    """A late unmapped index cannot mutate even existing damaged stacks."""
    from spacr.io import preprocess_img_data

    source = tmp_path / 'plate'
    _images(source)
    orig = source / 'orig'
    orig.mkdir()
    tifffile.imwrite(orig / NAMES[0].replace('W00002', 'W00003'),
                     np.full((5, 7), 55, dtype=np.uint16))
    (source / 'empty').mkdir()
    (source / 'stack').mkdir()
    (source / 'stack' / 'broken.npy').write_bytes(b'not a stack')
    (source / 'masks').mkdir()
    mapping = _map(tmp_path, 'plate,W00002,B07\n')
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match='W00003'):
        preprocess_img_data({'src': str(source), 'metadata_type': 'scanr_indexed',
                             'metadata_well_map': mapping, 'test_mode': test_mode})
    assert _snapshot(tmp_path) == before


def test_indexed_scanr_absent_map_refuses_and_old_scanr_remains_distinct(tmp_path):
    """Observed blank-comment names cannot borrow old A1-comment geometry."""
    from spacr import convert
    from spacr.regex_infer import _metadata_match

    _images(tmp_path / 'plate')
    plan = convert.plan(convert.scan(str(tmp_path / 'plate')))
    assert not plan.ok and not plan.mappings
    for name in NAMES:
        assert _metadata_match(name, 'scanr') is None
        assert _metadata_match(name, 'scanr_indexed')['wellID'] == 'W00002'
    old = 'A1--W00001--P00001--Z00000--T00000--mCherry1.tif'
    assert _metadata_match(old, 'scanr')['wellID'] == 'A1'
    assert _metadata_match(old, 'scanr_indexed') is None


def test_indexed_scanr_test_mode_samples_then_maps_against_original_plate(tmp_path, monkeypatch):
    """The real sampler must not rename the CSV's source plate to test."""
    from spacr import io

    source = tmp_path / 'plate'
    originals = _images(source)
    mapping = _map(tmp_path, 'plate,W00002,B07\n')
    ingest = io._rename_and_organize_image_files
    from spacr import utils
    sample = utils._run_test_mode
    seen = []

    def sample_then_change_map(*args, **kwargs):
        """An external map edit cannot change the already validated operation."""
        result = sample(*args, **kwargs)
        Path(mapping).write_text('plate,index,well\nplate,W00002,C08\n', encoding='utf-8')
        return result

    monkeypatch.setattr(utils, '_run_test_mode', sample_then_change_map)

    class IngestionComplete(BaseException):
        """Stop after real raw ingestion, before unrelated normalization."""

    def ingest_then_stop(*args, **kwargs):
        """Record the public settings route and execute its actual raw ingestion."""
        seen.append((Path(args[0]), kwargs['source_plate']))
        assert ingest(*args, **kwargs) == 2
        raise IngestionComplete

    monkeypatch.setattr(io, '_rename_and_organize_image_files', ingest_then_stop)
    with pytest.raises(IngestionComplete):
        io.preprocess_img_data({'src': str(source), 'metadata_type': 'scanr_indexed',
                                'metadata_well_map': mapping, 'test_mode': True,
                                'test_images': 1, 'random_test': False})
    assert seen == [(source / 'test', 'plate')]
    stacks = list((source / 'test' / 'stack').glob('*.npy'))
    assert len(stacks) == 1 and '_B07_' in stacks[0].name
    np.testing.assert_array_equal(np.load(stacks[0]),
                                  np.stack([np.full((5, 7), value, dtype=np.uint16)
                                            for value in (11, 12)], axis=2))
    assert {name: (source / name).read_bytes() for name in NAMES} == originals


def test_indexed_scanr_malformed_late_orig_name_refuses_before_any_mutation(tmp_path):
    """The shared resolver refuses incomplete acquisition names before sampling."""
    from spacr.io import preprocess_img_data

    source = tmp_path / 'plate'
    _images(source)
    orig = source / 'orig'
    orig.mkdir()
    (orig / '--W00003--P00001--Z00000--T00000--.tif').write_bytes(b'unread malformed image')
    mapping = _map(tmp_path, 'plate,W00002,B07\nplate,W00003,C08\n')
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match='must match'):
        preprocess_img_data({'src': str(source), 'metadata_type': 'scanr_indexed',
                             'metadata_well_map': mapping, 'test_mode': True})
    assert _snapshot(tmp_path) == before
