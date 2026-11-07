"""CPU embedding runs as workers x threads, with a default from the core count.

On a busy machine one torch process with many threads is slower than a few
processes with two threads each. These tests check the default split, that
several worker processes give exactly the matrix one process gives, that
each worker runs with the thread count asked for, and that the calling
process gets its own thread count back.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacr.embeddings as emb

torch = pytest.importorskip("torch")


class ThreadReportingEncoder:
    """Mean of each plane plus the torch thread count of the process."""

    in_channels = 3

    def __call__(self, stack):
        import os

        import torch as _torch

        out = np.empty((stack.shape[0], 3), dtype=np.float32)
        out[:, 0] = stack.mean(axis=(1, 2, 3))
        out[:, 1] = float(_torch.get_num_threads())
        out[:, 2] = float(os.getpid())
        return out


class MeanEncoder:
    """Per-plane means: a pure function of the crop."""

    in_channels = 3

    def __call__(self, stack):
        return stack.mean(axis=(1, 2)).astype(np.float32)


@pytest.mark.parametrize("cores, expected", [
    (1, (1, 1)), (2, (1, 1)), (3, (1, 1)), (4, (1, 2)), (8, (2, 2)),
    (16, (4, 2)), (32, (4, 2)), (128, (4, 2)),
])
def test_the_default_split_follows_the_core_count(cores, expected):
    assert emb._cpu_parallelism(cores) == expected


def test_explicit_workers_and_threads_are_kept_and_checked():
    assert emb._cpu_parallelism(32, workers=3, threads=5) == (3, 5)
    assert emb._cpu_parallelism(32, workers=0, threads=0) == (4, 2)
    with pytest.raises(emb.EmbeddingError, match="workers"):
        emb._cpu_parallelism(8, workers=-1)
    with pytest.raises(emb.EmbeddingError, match="threads"):
        emb._cpu_parallelism(8, threads=-2)


def _crops(n=37):
    return np.random.default_rng(1).random((n, 6, 5, 2)).astype(np.float32)


def _spec(channels=None):
    scale = (1.0,) * (2 if channels is None else len(channels))
    return emb.EmbeddingSpec(device="cpu", channel_scale=scale, batch_size=4,
                             channels=channels)


def test_several_workers_give_the_matrix_one_process_gives():
    crops = _crops()
    one = emb.embed_array(crops, _spec(), encoder=MeanEncoder())
    many = emb._embed_parallel(crops, _spec(), workers=3, threads=1,
                               encoder=MeanEncoder(), chunk_size=5)
    np.testing.assert_array_equal(many.values, one.values)
    assert many.columns == one.columns
    assert many.spec == one.spec


def test_each_worker_runs_with_the_threads_it_was_given():
    before = torch.get_num_threads()
    result = emb._embed_parallel(_crops(12), _spec(channels=(0,)), workers=2,
                                 threads=2, encoder=ThreadReportingEncoder(),
                                 chunk_size=3)
    import os

    threads = result.values[:, 1]
    pids = set(result.values[:, 2].astype(int))
    assert set(threads) == {2.0}
    assert os.getpid() not in pids and 1 <= len(pids) <= 2
    assert torch.get_num_threads() == before


def test_one_worker_runs_here_and_gives_the_thread_count_back():
    before = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        result = emb._embed_parallel(_crops(6), _spec(channels=(0,)),
                                     workers=1, threads=1,
                                     encoder=ThreadReportingEncoder(),
                                     chunk_size=4)
        import os

        assert set(result.values[:, 1]) == {1.0}
        assert set(result.values[:, 2].astype(int)) == {os.getpid()}
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(before)


def test_a_spec_without_its_scale_is_refused_before_any_worker_starts():
    with pytest.raises(emb.EmbeddingError, match="channel_scale"):
        emb._embed_parallel(_crops(4), emb.EmbeddingSpec(device="cpu"),
                            workers=2, threads=1, encoder=MeanEncoder())


def test_overview_threads_setting_reaches_the_encoder(monkeypatch):
    from spacr import measure

    seen = []

    class Recording(MeanEncoder):
        def __call__(self, stack):
            seen.append(torch.get_num_threads())
            return super().__call__(stack)

    monkeypatch.setattr(measure, "_IMAGE_ADVANCED_CACHE", {})
    monkeypatch.setattr(emb, "_backbone_encoder", lambda spec: Recording())
    options = measure._image_advanced_options(
        {"overview_measure_embeddings": True, "overview_embedding_threads": 2})
    assert options["embedding_threads"] == 2
    assert measure._image_advanced_options({})["embedding_threads"] == \
        emb._cpu_parallelism()[1]
    before = torch.get_num_threads()
    planes = {0: np.random.default_rng(0).random((40, 40)).astype(np.float32)}
    record = measure._image_embedding(planes, "resnet18", 32,
                                      threads=options["embedding_threads"])
    assert seen == [2] and record["embedding_dims"] == 3
    assert torch.get_num_threads() == before
