"""OPS keeps its inputs while sharing mask batch, precision and governor policies."""

import numpy as np
import pytest
import torch

from spacr import _mask_array_batches as arrays
from spacr import _mask_batches as batches
from spacr import object as objects
from spacr import ops_engine, ops_settings, settings, validate


@pytest.fixture
def memory(monkeypatch):
    """Admit small CPU inputs without depending on the load on this machine."""
    monkeypatch.setattr(batches, '_wait_for_ram', lambda estimate: None)
    monkeypatch.setattr(batches, '_available_ram_bytes', lambda: 16 * 2**30)
    monkeypatch.setattr(batches, '_free_vram_bytes', lambda: 16 * 2**30)
    monkeypatch.setattr(batches, '_model_resident', lambda: False)


def test_mixed_shapes_and_dtypes_keep_every_input_byte_and_one_batch(memory):
    """Packing must retain NaNs, large uint64 integers and strided inputs."""
    images = [np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2],
              np.array([[2**63 + 1]], np.uint64),
              np.array([[[np.nan, -0.0], [np.inf, 1.5]]], np.float64)]
    loaded, seen = [], []

    def load(index):
        """Record exactly which batch has been read."""
        loaded.append(index)
        return images[index], f'field-{index}'

    with arrays._array_batches(range(3), load, {'batch_size': 2}) as resident:
        for index, image, metadata, fields in resident:
            assert loaded == ([0, 1] if index < 2 else [0, 1, 2])
            assert image.dtype == images[index].dtype
            assert image.shape == images[index].shape
            assert image.tobytes() == images[index].tobytes()
            assert metadata == f'field-{index}'
            assert fields == (2 if index < 2 else 1)
            seen.append(index)
    assert seen == [0, 1, 2]


def test_an_error_frees_the_device_batch_before_any_next_batch(memory, monkeypatch):
    """The shared resident free runs even when the first role raises."""
    loaded, freed, tiles = [], [], []
    monkeypatch.setattr(batches, '_empty_cache', lambda: freed.append(True))

    def store(packed, **keywords):
        """Represent the admitted device copy with a real CPU tensor."""
        tiles.append(keywords['tile_batch'])
        return batches._GpuBatch(torch.from_numpy(packed.copy()))

    def load(index):
        """Load a field only when its own batch is reached."""
        loaded.append(index)
        return np.full((3, 4), index, np.float32), None

    monkeypatch.setattr(batches, '_store_on_gpu', store)
    with pytest.raises(RuntimeError, match='role failed'):
        with arrays._array_batches(range(4), load, {'batch_size': 2}, gpu=True) as resident:
            next(resident)
            raise RuntimeError('role failed')
    assert loaded == [0, 1] and freed == [True] and tiles == [2]


@pytest.mark.parametrize('capacity', [None, 1])
def test_partial_resume_keeps_the_original_automatic_tile_count(memory, tmp_path, capacity):
    """The last unfinished field still runs with its original three-field group."""
    path = tmp_path / 'mask_batch_plan.json'

    def load(index):
        """Supply a tiny image with no pipeline-specific metadata."""
        return np.full((3, 4), index, np.float32), None

    with arrays._array_batches(range(5), load, {'batch_size': 3}, plan_path=path) as resident:
        assert next(resident)[3] == 3
    with arrays._array_batches([2, 3, 4], load, {'batch_size': 3}, plan_path=path,
                               max_fields=capacity) as resident:
        later = 2 if capacity is None else 1
        assert [(index, fields) for index, _image, _metadata, fields in resident] == [
            (2, 3), (3, later), (4, later)]
    with pytest.raises(ValueError, match='Use the saved settings'):
        with arrays._array_batches([2], load, {'batch_size': 4}, plan_path=path) as resident:
            next(resident)


def test_loading_respects_an_upstream_single_field_staging_slot(memory):
    """An automatic batch cannot wait for a second field held behind its own slot."""
    held = []

    def load(index):
        """Refuse a second read until the preceding field has been consumed."""
        assert not held
        held.append(index)
        return np.zeros((3, 4), np.float32), None

    with arrays._array_batches(range(4), load, {}, max_fields=1) as resident:
        for index, _image, _metadata, fields in resident:
            assert held == [index] and fields == 1
            held.clear()


@pytest.mark.parametrize('given, expected', [(0, 3), (8, 8), ('16', 16)])
def test_ops_eval_uses_the_shared_tile_policy_and_mask_wait_site(monkeypatch, given, expected):
    """The original image reaches the same governed mask helper with a mask site."""
    image, model, seen = object(), object(), []

    def evaluate(actual_model, actual_image, **keywords):
        """Observe the spaCR helper without changing a Cellpose function."""
        assert actual_model is model and actual_image is image
        seen.append(keywords)
        return 'result'

    monkeypatch.setattr(objects, '_mask_eval', evaluate)
    assert ops_engine._ops_mask_eval(
        model, image, {'mask_eval_batch_size': given}, site='ops_phenotype',
        field_count=3, diameter=20) == 'result'
    assert seen[0]['batch_size'] == expected
    assert seen[0]['site'] == 'mask_ops_phenotype'
    assert seen[0]['diameter'] == 20


def test_the_retired_ops_tile_key_moves_only_through_the_shared_schema():
    """Saved OPS settings retain automatic VRAM tiles, migrate once and let an explicit phenotype override win."""
    assert settings.RENAMED_SETTINGS['ops_ph_batch_size'] == 'ops_ph_tile_batch'
    assert validate.RETIRED_SETTINGS['ops_ph_batch_size'] == 'ops_ph_tile_batch'
    result = ops_settings.ops_defaults({'ops_ph_batch_size': 8})
    assert 'ops_ph_batch_size' not in result and result['ops_ph_tile_batch'] == 8
    assert result['ops_sbs_tile_batch'] == 0 and result['mask_eval_batch_size'] == 'auto_vram'
    assert ops_settings.ops_defaults({})['mask_inference_precision'] == 'auto'
    assert ops_settings.ops_defaults({'ops_ph_batch_size': 8,
                                      'ops_ph_tile_batch': 16})['ops_ph_tile_batch'] == 16


def test_a_streamed_batch_keeps_the_grouping_and_holds_no_packed_copy(memory, monkeypatch, tmp_path):
    """Streaming hands on the loaded fields, the same groups and plan, and never packs."""
    images = [np.arange(12, dtype=np.float32).reshape(3, 4) + index for index in range(5)]

    def load(index):
        """Field 3 is unreadable; the others are distinct small arrays."""
        return (None if index == 3 else images[index]), f'field-{index}'

    def run(stream, path):
        """Every yielded field, as bytes, with its count, and the saved plan."""
        with arrays._array_batches(range(5), load, {'batch_size': 3}, plan_path=path,
                                   stream=stream) as resident:
            seen = [(index, None if image is None else image.tobytes(), metadata, fields)
                    for index, image, metadata, fields in resident]
        return seen, path.read_text()

    packed = run(False, tmp_path / 'packed.json')
    monkeypatch.setattr(arrays.np, 'concatenate', lambda *a, **k: pytest.fail('packed'))
    monkeypatch.setattr(batches, '_store_on_gpu', lambda *a, **k: pytest.fail('device copy'))
    streamed = run(True, tmp_path / 'streamed.json')
    assert streamed == packed
    assert [fields for _index, _image, _metadata, fields in streamed[0]] == [3, 3, 3, 1, 1]


def test_a_streamed_field_is_the_loaded_array_and_is_let_go(memory):
    """The caller gets the loader's own array, and the batch keeps no reference to it."""
    import gc
    import weakref

    loaded = {}

    class Field(np.ndarray):
        """An array type that can be weakly referenced."""

    def load(index):
        """A fresh array per field, remembered weakly."""
        image = np.full((3, 4), index, np.float32).view(Field)
        loaded[index] = weakref.ref(image)
        return image, None

    with arrays._array_batches(range(2), load, {'batch_size': 2}, stream=True) as resident:
        index, image, _metadata, _fields = next(resident)
        assert image is loaded[0]()
        del image
        gc.collect()
        assert loaded[0]() is None
        next(resident)


def test_a_lazy_stream_keeps_the_grouping_and_reads_at_most_one_field_ahead(memory, tmp_path):
    """With ``present`` a field is read at most one ahead of the caller; groups, counts and plan stay."""
    images = [np.arange(12, dtype=np.float32).reshape(3, 4) + index for index in range(5)]
    calls = []

    def load(index):
        """Field 3 is unreadable; the others are distinct small arrays."""
        calls.append(index)
        return (None if index == 3 else images[index]), f'field-{index}'

    def run(present, path):
        """Every yielded field with its count, the loads made before it, and the plan."""
        calls.clear()
        seen = []
        with arrays._array_batches(range(5), load, {'batch_size': 3}, plan_path=path,
                                   stream=True, present=present) as resident:
            for index, image, metadata, fields in resident:
                seen.append((index, None if image is None else image.tobytes(), metadata,
                             fields, len(calls)))
        return seen, path.read_text()

    eager, eager_plan = run(None, tmp_path / 'eager.json')
    lazy, lazy_plan = run(lambda index: index != 3, tmp_path / 'lazy.json')
    assert lazy_plan == eager_plan
    assert [row[:4] for row in lazy] == [row[:4] for row in eager]
    assert all(row[4] <= index + 2 for index, row in enumerate(lazy))
    assert [row[4] for row in eager][:3] == [3, 3, 3]


def test_a_lazy_stream_refuses_a_loader_that_disagrees_with_present(memory):
    """A field ``present`` called absent but the loader read is an error, not a recount."""
    def load(index):
        """Every field has an image."""
        return np.zeros((2, 2), np.float32), None

    with pytest.raises(ValueError, match='expected to have no image'):
        with arrays._array_batches(range(3), load, {'batch_size': 3}, stream=True,
                                   present=lambda index: index != 2) as resident:
            list(resident)
