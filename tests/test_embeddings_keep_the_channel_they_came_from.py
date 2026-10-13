"""A label-free embedding, and the channel decision 386 asked to be explicit.

THE CHANNEL PROBLEM IS THE DESIGN PROBLEM. Pretrained encoders want three
channels; spaCR images routinely have four or five with no RGB meaning.
Instruction 386 asks for the choice to be settled and recorded rather than
inherited from whatever the first backbone happened to want, so these tests
pin the consequence of the choice rather than the choice itself:

* under the default policy a column NAMES its channel, which is what lets
  `spacr.attribution_columns` say which stain a hit came from later;
* under projection there is no channel to name, and the code says ``None``
  instead of guessing;
* four channels cannot be projected onto three silently -- somebody has to
  say which to drop.

The encoder is injected in most tests. What can break here is the wiring --
shapes, ordering, naming, alignment -- and a stub exercises all of it without
downloading a backbone. One test does load the real one, because a stub that
agrees with itself proves nothing about timm.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.embeddings import (
    CHANNEL_PER_CHANNEL, CHANNEL_PROJECT, EMBEDDING_PREFIX, EmbeddingError,
    EmbeddingResult, EmbeddingSpec, channel_of_column, embed_array,
    embedding_column_names)


@pytest.fixture
def crops():
    """Seven objects, 32x32, four channels -- a normal spaCR stack."""
    return np.random.default_rng(0).random((7, 32, 32, 4)).astype(np.float32)


def _stub(dims=5):
    """An encoder whose output depends on the input, so ordering is testable."""
    def run(stack):
        flat = stack.reshape(stack.shape[0], -1)
        return np.stack([flat[:, i::max(flat.shape[1] // dims, 1)][:, :1].ravel()
                         for i in range(dims)], axis=1)
    return run


def test_every_channel_gets_its_own_block(crops):
    result = embed_array(crops, EmbeddingSpec(), encoder=_stub())
    assert result.n_channels == 4
    assert result.values.shape == (7, 4 * result.per_channel_dims)
    assert all(c.startswith(EMBEDDING_PREFIX) for c in result.columns)


def test_a_column_says_which_channel_it_came_from(crops):
    """THE REASON FOR THE DEFAULT. Attribution is impossible after mixing."""
    result = embed_array(crops, EmbeddingSpec(), encoder=_stub())
    seen = [channel_of_column(c) for c in result.columns]
    assert seen[0] == 0 and seen[-1] == 3
    assert sorted(set(seen)) == [0, 1, 2, 3]


def test_a_projected_column_admits_it_has_no_channel(crops):
    result = embed_array(
        crops, EmbeddingSpec(channel_policy=CHANNEL_PROJECT, channels=(0, 1, 2)),
        encoder=_stub())
    assert all(channel_of_column(c) is None for c in result.columns)
    assert result.values.shape[1] == result.per_channel_dims


def test_four_channels_are_not_silently_squeezed_into_three(crops):
    """Dropping a stain is a decision, not a default."""
    with pytest.raises(EmbeddingError, match="which to drop"):
        embed_array(crops, EmbeddingSpec(channel_policy=CHANNEL_PROJECT),
                    encoder=_stub())


def test_a_named_subset_of_channels_is_honoured(crops):
    result = embed_array(crops, EmbeddingSpec(channels=(1, 3)), encoder=_stub())
    assert result.n_channels == 2
    assert sorted({channel_of_column(c) for c in result.columns}) == [1, 3]


@pytest.mark.parametrize("bad", [
    np.zeros((4, 4, 4), dtype=np.float32),
    np.zeros((0, 8, 8, 2), dtype=np.float32),
])
def test_a_wrong_array_is_refused_before_a_model_is_loaded(bad):
    """A download is a slow way to learn the array was the wrong shape."""
    with pytest.raises(EmbeddingError):
        embed_array(bad, EmbeddingSpec())


def test_an_unknown_policy_is_refused_at_construction():
    with pytest.raises(EmbeddingError, match="unknown channel policy"):
        EmbeddingSpec(channel_policy="rgbish")


def test_the_fingerprint_moves_only_with_what_changes_the_numbers():
    base = EmbeddingSpec()
    assert base.fingerprint() == EmbeddingSpec(batch_size=8).fingerprint()
    assert base.fingerprint() != EmbeddingSpec(backbone="resnet34").fingerprint()
    assert base.fingerprint() != EmbeddingSpec(
        channel_policy=CHANNEL_PROJECT, channels=(0, 1, 2)).fingerprint()


def test_a_misaligned_id_list_is_refused_rather_than_joined(crops):
    """A silent misalignment corrupts every downstream join, and there is no
    way to notice afterwards."""
    result = embed_array(crops, EmbeddingSpec(), encoder=_stub())
    with pytest.raises(EmbeddingError, match="object ids"):
        result.to_frame(["a", "b"])


def test_the_frame_is_keyed_and_ready_to_join(crops):
    result = embed_array(crops, EmbeddingSpec(), encoder=_stub())
    frame = result.to_frame([f"obj{i}" for i in range(7)])
    assert list(frame.columns)[0] == "object_id"
    assert len(frame) == 7


def test_column_names_are_stable_and_sorted_by_channel():
    names = embedding_column_names(3, (0, 2), CHANNEL_PER_CHANNEL)
    assert names == ("emb_c0_000", "emb_c0_001", "emb_c0_002",
                     "emb_c2_000", "emb_c2_001", "emb_c2_002")


def test_the_real_backbone_produces_finite_numbers():
    """A stub that agrees with itself proves nothing about timm."""
    pytest.importorskip("timm")
    crops = (np.random.default_rng(1).random((2, 64, 64, 2)) * 1000
             ).astype(np.float32)
    result = embed_array(crops, EmbeddingSpec(device="cpu", batch_size=2))
    assert result.values.shape == (2, 2 * result.per_channel_dims)
    assert np.isfinite(result.values).all()
    assert (result.values != 0).any()


def test_embeddings_retrieve_a_phenotype_the_stub_can_see():
    """386 asks for RETRIEVAL as the measure, not loss.

    Two synthetic genotypes with different texture. Nearest-neighbour on the
    embedding should put like with like far more often than chance, which is
    50% for two balanced classes.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(0.2, 0.02, (12, 32, 32, 2))
    b = rng.normal(0.8, 0.02, (12, 32, 32, 2))
    crops = np.concatenate([a, b]).astype(np.float32)
    labels = np.array([0] * 12 + [1] * 12)
    values = embed_array(crops, EmbeddingSpec(normalize=False),
                         encoder=_stub(dims=8)).values
    distance = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    predicted = labels[distance.argmin(axis=1)]
    assert (predicted == labels).mean() > 0.9
