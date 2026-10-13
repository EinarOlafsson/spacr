"""Prepared editor examples preserve real planes and reject incompatible data."""
from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_make_masks import digest, prepare_fields


def sources(root, planes=7, dtype=np.uint16, labels=True):
    folder = root / 'example_data/plate1/merged'
    folder.mkdir(parents=True)
    for index in range(2):
        array = np.zeros((16, 16, planes), dtype=dtype)
        array[..., 1] = np.arange(256).reshape(16, 16) + index
        if labels:
            array[3:8, 4:9, 4] = 17
        np.save(folder / f'plate_field_{index}.npy', array)
    return folder


def test_preparation_preserves_both_planes_and_all_source_bytes(tmp_path):
    original = sources(tmp_path)
    before = {path.name: digest(path) for path in original.iterdir()}
    folder, records = prepare_fields(tmp_path)
    assert folder.is_relative_to(tmp_path / 'make_masks_runs')
    assert len(records) == 2
    for row in records:
        merged = np.load(row['source'])
        assert np.array_equal(tifffile.imread(row['image']), merged[..., 1])
        assert np.array_equal(tifffile.imread(row['mask']), merged[..., 4])
        assert row['objects'] == 1  # Label 17 is one object, not seventeen.
    assert {path.name: digest(path) for path in original.iterdir()} == before


@pytest.mark.parametrize('kwargs', [{'planes': 6}, {'dtype': np.float32}])
def test_existing_incompatible_stack_is_not_silently_cast(tmp_path, kwargs):
    sources(tmp_path, **kwargs)
    with pytest.raises(RuntimeError, match='seven-plane uint16'):
        prepare_fields(tmp_path)


def test_blank_label_planes_are_not_a_labelled_example(tmp_path):
    sources(tmp_path, labels=False)
    with pytest.raises(RuntimeError, match='no cell labels'):
        prepare_fields(tmp_path)
