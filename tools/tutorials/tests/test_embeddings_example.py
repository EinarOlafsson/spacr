"""Small preservation tests for the tutorial helper, not encoder stand-ins."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

PATH = Path(__file__).resolve().parents[1] / 'embeddings_example.py'
spec = importlib.util.spec_from_file_location('tutorial_embeddings_example', PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def crops_at(path):
    path.mkdir()
    for i in range(4):
        Image.fromarray(np.full((32, 32, 3), i * 30, dtype=np.uint8)).save(path / f'{i}.png')
    return path


def test_loads_exact_pixels_and_names_without_touching_sources(tmp_path):
    source = crops_at(tmp_path / 'crops')
    hashes = {p.name: helper.digest(p) for p in source.iterdir()}
    crops, records = helper.load_crops(source, 4)
    assert crops.shape == (4, 32, 32, 3)
    assert [r['name'] for r in records] == ['0.png', '1.png', '2.png', '3.png']
    for i in range(4):
        np.testing.assert_array_equal(crops[i], np.full((32, 32, 3), i * 30, dtype=np.uint8))
    assert {p.name: helper.digest(p) for p in source.iterdir()} == hashes


def test_refuses_real_missing_and_mismatched_inputs(tmp_path):
    source = crops_at(tmp_path / 'crops')
    helper.load_crops(source, 4)  # Positive counterpart reaches the same path.
    with pytest.raises(ValueError, match='found 4'):
        helper.load_crops(source, 5)
    Image.fromarray(np.ones((40, 40, 3), dtype=np.uint8)).save(source / '3.png')
    with pytest.raises(ValueError, match='equal shapes'):
        helper.load_crops(source, 4)


def test_existing_output_is_really_preserved(tmp_path):
    target = tmp_path / 'existing'
    target.mkdir()
    marker = target / 'user.txt'
    marker.write_text('keep this exact content')
    with pytest.raises(FileExistsError, match='Refusing existing'):
        helper.run(tmp_path / 'not_even_loaded', target, count=4)
    assert marker.read_text() == 'keep this exact content'


@pytest.mark.parametrize('corruption', ['npy', 'row', 'column', 'csv_value'])
def test_reopens_every_value_and_order_with_positive_counterpart(tmp_path, corruption):
    values = np.arange(12, dtype=np.float32).reshape(4, 3) / 7
    names, identities = ['emb_c0_000', 'emb_c0_001', 'emb_c0_002'], ['a', 'b', 'c', 'd']
    np.save(tmp_path / 'vectors.npy', values)
    frame = pd.DataFrame(values.copy(), columns=names)
    frame.insert(0, 'object_id', identities)
    frame.to_csv(tmp_path / 'vectors.csv', index=False)
    assert helper.verify_saved(tmp_path, values, names, identities)['matrix_cells_checked'] == 12
    if corruption == 'npy':
        changed = values.copy(); changed[0, 0] += 1
        np.save(tmp_path / 'vectors.npy', changed)
    elif corruption == 'row':
        frame['object_id'] = identities[::-1]  # Leave numbers intact to isolate identity guard.
    elif corruption == 'column':
        frame = frame.rename(columns={names[0]: 'wrong_feature'})
    else:
        frame.iloc[0, 1] += 1
    frame.to_csv(tmp_path / 'vectors.csv', index=False)
    if corruption != 'npy':
        np.testing.assert_array_equal(np.load(tmp_path / 'vectors.npy'), values)
    with pytest.raises((ValueError, AssertionError)):
        helper.verify_saved(tmp_path, values, names, identities)
