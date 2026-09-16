"""410: an object's embedding must not depend on which crops share its batch.

THE DEFECT. ``_scaled`` took the 99th percentile over whatever plane it was
handed, and ``embed_array`` handed it ``array[..., channel]`` -- the WHOLE
batch. One c1 crop from methods_paper plate1 went from a scaled mean of 0.311
alone to 0.076 inside a c1 batch and 0.094 inside a c2 batch: a 4x change in
input for an unchanged object, and a different change per condition, so a
batch sorted by condition leaked the condition into the scaling.

THE DECISION, 2026-09-15: one FIXED scale per channel per plate, estimated
from a random sample of the plate, stored with the run, and applied to every
crop. These tests pin that contract rather than a number:

* under a fixed scale a crop's scaled input and its embedding are identical
  alone and inside two differently composed batches;
* a later batch handed the spec a whole-plate call returned is scaled like the
  plate -- the test that fails on the pre-410 code with real arithmetic, not
  an import error, because it uses only the API that already existed;
* the estimate is deterministic for a fixed seed and reads in bounded batches;
* the plate path records the scale and a rerun reuses it without estimating.

The encoder is injected everywhere. What 410 is about is the arithmetic in
front of the backbone, and a recording stub sees exactly that.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from spacr.embeddings import EmbeddingError, EmbeddingSpec, embed_array


def _plate(seed: int = 0) -> np.ndarray:
    """Two conditions, 40 crops each, whose channels differ ~50x in range.

    c1 is dim and c2 bright, as in the plate that exposed the defect, so a
    batch-wide percentile is visibly different for a c1 batch and a c2 batch.
    """
    rng = np.random.default_rng(seed)
    c1 = rng.gamma(2.0, 10.0, (40, 16, 16, 2))
    c2 = rng.gamma(2.0, 40.0, (40, 16, 16, 2))
    plate = np.concatenate([c1, c2]).astype(np.float32)
    plate[..., 1] *= 50.0
    return plate


class _Recorder:
    """A stand-in backbone that keeps every stack it was handed.

    Its output for a row looks only at that row, so any difference between a
    crop embedded alone and in a batch can only come from the scaling.
    """

    def __init__(self) -> None:
        self.seen = []

    def __call__(self, stack):
        stack = np.asarray(stack, dtype=np.float32)
        self.seen.append(stack.copy())
        flat = stack.reshape(stack.shape[0], -1)
        return np.concatenate(
            [flat[:, ::7], flat.mean(axis=1, keepdims=True),
             flat.std(axis=1, keepdims=True)], axis=1)


class _BoundedPlate:
    """An array-like plate that refuses to be read in one gulp."""

    def __init__(self, array: np.ndarray, limit: int) -> None:
        self.array = array
        self.limit = limit
        self.requests = []

    @property
    def shape(self):
        return self.array.shape

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, index):
        index = np.asarray(index)
        assert index.ndim == 1 and 0 < index.size <= self.limit, index
        assert np.all(np.diff(index) > 0), "reads must be sorted and unique"
        self.requests.append(int(index.size))
        return self.array[index]


def _crop_row(batch: np.ndarray, row: int, spec: EmbeddingSpec):
    """The scaled input one crop reached the encoder with, and its vector."""
    recorder = _Recorder()
    result = embed_array(batch, spec, encoder=recorder)
    scaled = np.stack([call[row] for call in recorder.seen])
    return scaled, result.values[row]


#: Crop 3 is a c1 crop. Alone; at row 2 among c1 crops; at row 1 among c2.
_ALONE = ([3], 0)
_IN_C1 = ([0, 1, 3, 5, 7], 2)
_IN_C2 = ([45, 3, 50, 60], 1)


def _three_ways(plate: np.ndarray, spec: EmbeddingSpec):
    return [_crop_row(plate[rows], row, spec)
            for rows, row in (_ALONE, _IN_C1, _IN_C2)]


def test_one_crop_gets_the_same_input_and_vector_alone_and_in_two_batches():
    """THE ITEM'S OWN CHECK, under the plate-scale path."""
    from spacr.embeddings import _estimate_channel_scale

    plate = _plate()
    spec = EmbeddingSpec(channel_scale=_estimate_channel_scale(plate))
    (alone_in, alone_out), (c1_in, c1_out), (c2_in, c2_out) = _three_ways(
        plate, spec)

    np.testing.assert_array_equal(alone_in, c1_in)
    np.testing.assert_array_equal(alone_in, c2_in)
    np.testing.assert_array_equal(alone_out, c1_out)
    np.testing.assert_array_equal(alone_out, c2_out)


def test_a_later_batch_given_the_runs_spec_is_scaled_like_the_plate():
    """Uses only the API that existed before 410, so it fails on that code
    with the defect's own arithmetic: the spec a whole-plate call returned
    carried no scale, and each later batch was scaled by its own percentile.
    """
    plate = _plate()
    run = embed_array(plate, EmbeddingSpec(), encoder=_Recorder())
    (alone_in, alone_out), (c1_in, c1_out), (c2_in, c2_out) = _three_ways(
        plate, run.spec)

    means = [float(alone_in.mean()), float(c1_in.mean()), float(c2_in.mean())]
    assert means[0] == means[1] == means[2], (
        f"one crop's scaled mean was {means[0]:.3f} alone, {means[1]:.3f} "
        f"among c1 crops and {means[2]:.3f} among c2 crops")
    np.testing.assert_array_equal(alone_in, c1_in)
    np.testing.assert_array_equal(alone_in, c2_in)
    np.testing.assert_array_equal(alone_out, c1_out)
    np.testing.assert_array_equal(alone_out, c2_out)


def test_a_direct_call_without_a_scale_uses_one_for_its_stack_and_says_so(
        caplog):
    """The stack handed over is treated as the plate, and the scale it got
    is on the result -- so the numbers a small call produced before 410 are
    unchanged, and a caller can pass them on instead of guessing."""
    plate = _plate()[:20]
    recorder = _Recorder()
    with caplog.at_level(logging.WARNING, logger="spacr.embeddings"):
        result = embed_array(plate, EmbeddingSpec(), encoder=recorder)

    expected = tuple(float(np.percentile(plate[..., c], 99)) for c in (0, 1))
    assert result.spec.channel_scale == expected
    for channel, call in enumerate(recorder.seen):
        np.testing.assert_array_equal(
            call[..., 0],
            np.clip(plate[..., channel] / expected[channel], 0.0, 1.0))
    assert "channel_scale" in caplog.text


def test_the_scale_estimate_is_the_same_every_time_for_a_fixed_seed():
    from spacr.embeddings import _estimate_channel_scale

    plate = _plate()
    plate *= np.linspace(0.2, 5.0, len(plate), dtype=np.float32
                         )[:, None, None, None]
    first = _estimate_channel_scale(plate, sample_size=8, seed=11)
    again = _estimate_channel_scale(plate, sample_size=8, seed=11)
    other = _estimate_channel_scale(plate, sample_size=8, seed=12)

    assert first == again
    assert first != other
    assert len(first) == 2 and all(isinstance(v, float) for v in first)


def test_the_estimate_reads_in_bounded_batches_and_matches_one_gulp():
    from spacr.embeddings import _estimate_channel_scale

    plate = _plate()
    bounded = _BoundedPlate(plate, limit=16)
    small_reads = _estimate_channel_scale(bounded, sample_size=50, seed=3,
                                          read_batch=16)
    one_gulp = _estimate_channel_scale(plate, sample_size=50, seed=3,
                                       read_batch=10_000)

    assert small_reads == one_gulp
    assert max(bounded.requests) <= 16 and sum(bounded.requests) == 50


def test_a_sample_as_large_as_the_plate_is_the_whole_plate_percentile():
    from spacr.embeddings import _estimate_channel_scale

    plate = _plate()
    assert _estimate_channel_scale(plate, sample_size=10_000) == tuple(
        float(np.percentile(plate[..., c], 99)) for c in (0, 1))


def test_a_blank_channel_scales_to_zeros_rather_than_dividing_by_zero():
    from spacr.embeddings import _estimate_channel_scale

    plate = _plate()
    plate[..., 1] = 0.0
    scale = _estimate_channel_scale(plate)
    assert scale[1] == 0.0
    recorder = _Recorder()
    embed_array(plate, EmbeddingSpec(channel_scale=scale), encoder=recorder)
    assert np.all(recorder.seen[1] == 0.0)


def test_the_plate_path_records_its_scale_and_a_rerun_reuses_it(monkeypatch):
    import spacr.embeddings as engine

    plate = _plate()
    record = {}
    first = engine._embed_plate(plate, EmbeddingSpec(), encoder=_Recorder(),
                                record=record)

    assert first.spec.channel_scale == tuple(
        record["channel_scale"][str(c)] for c in (0, 1))
    assert record["channel_scale_sample"]["seed"] == 0
    assert record["channel_scale_sample"]["n_crops"] == len(plate)

    saved = json.loads(json.dumps(record))        # it survives a run file

    def refuse(*_args, **_kwargs):
        raise AssertionError("a rerun must reuse the recorded scale")

    monkeypatch.setattr(engine, "_estimate_channel_scale", refuse)
    rerun = engine._embed_plate(plate[::-1], EmbeddingSpec(),
                                encoder=_Recorder(), record=saved)
    assert rerun.spec.channel_scale == first.spec.channel_scale
    np.testing.assert_array_equal(rerun.values[::-1], first.values)

    one = engine._embed_plate(plate, EmbeddingSpec(channels=(1,)),
                              encoder=_Recorder(), record=saved)
    assert one.spec.channel_scale == (record["channel_scale"]["1"],)


def test_a_record_from_a_plate_with_other_channels_is_refused():
    import spacr.embeddings as engine

    record = {"channel_scale": {"0": 10.0}}
    with pytest.raises(EmbeddingError, match="record"):
        engine._embed_plate(_plate(), EmbeddingSpec(), encoder=_Recorder(),
                            record=record)


@pytest.mark.parametrize("kwargs, match", [
    (dict(channel_scale=(1.0,), channels=(0, 1)), "one per encoded channel"),
    (dict(channel_scale=(float("nan"), 1.0)), "finite"),
    (dict(channel_scale=(-1.0, 1.0)), "finite"),
    (dict(channel_scale=(1.0, 1.0), normalize=False), "normalize"),
])
def test_an_impossible_scale_is_refused_when_the_spec_is_built(kwargs, match):
    with pytest.raises(EmbeddingError, match=match):
        EmbeddingSpec(**kwargs)


def test_a_scale_that_does_not_fit_the_crops_is_refused_before_encoding():
    recorder = _Recorder()
    with pytest.raises(EmbeddingError, match="one per encoded channel"):
        embed_array(_plate(), EmbeddingSpec(channel_scale=(1.0, 2.0, 3.0)),
                    encoder=recorder)
    assert recorder.seen == []


def test_the_fingerprint_moves_with_the_scale_and_not_without_one():
    """Unscaled specs keep 386's recorded fingerprint, so cached matrices
    named by it stay addressable; a scale is part of the numbers."""
    base = EmbeddingSpec()
    assert base.fingerprint() == "ee13fbde21d1062c"
    scaled = EmbeddingSpec(channel_scale=(85.0, 79.0, 198.0))
    assert scaled.fingerprint() != base.fingerprint()
    assert scaled.fingerprint() != EmbeddingSpec(
        channel_scale=(85.0, 79.0, 199.0)).fingerprint()
