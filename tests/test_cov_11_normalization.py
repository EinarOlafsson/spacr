"""Statistics that survive a malformed batch and describe themselves plainly.

Two things a training run needs from :mod:`spacr.normalization`: a streaming
pass that keeps going when a loader hands it something that is not a batch of
images, and a one-line description of whatever statistics were chosen, so a
model card records what the weights actually saw.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.normalization import (
    IMAGENET_MEAN,
    dataset_statistics,
    describe_normalization,
)


def _batch(value, n=2, c=2, h=4, w=4):
    return np.full((n, c, h, w), float(value), dtype=np.float32)


def test_a_batch_that_is_not_images_is_skipped_not_fatal():
    """A loader that yields a label vector between batches must not stop the pass.

    A collapsed or already-flattened batch has no channel axis to reduce over.
    Skipping it keeps the statistics those of the images that WERE images,
    which is a usable answer; raising would abandon the whole run over one
    malformed item.
    """
    loader = [_batch(0.25), np.zeros((8,), dtype=np.float32), _batch(0.75)]

    mean, std = dataset_statistics(loader)

    assert len(mean) == 2
    assert mean == pytest.approx((0.5, 0.5))
    assert std == pytest.approx((0.25, 0.25))


def test_a_loader_of_only_malformed_batches_refuses():
    """Skipping every batch leaves an empty set, and that is named as such."""
    with pytest.raises(ValueError, match="no images"):
        dataset_statistics([np.zeros((8,), dtype=np.float32)])


def test_measured_statistics_are_described_as_measured():
    """Numbers that match no known backbone are reported as this dataset's own.

    The description is what goes in the log and the model card, so it has to
    distinguish "ImageNet's numbers" from "numbers computed from these crops"
    -- the two normalise the same data to different places.
    """
    text = describe_normalization("custom", mean=(0.11, 0.22, 0.33),
                                  std=(0.4, 0.5, 0.6))

    assert "0.11" in text and "0.4" in text
    assert "measured from" in text
    assert "ImageNet" not in text


def test_imagenet_statistics_are_described_by_name():
    """The known sets stay named, so the custom wording means something."""
    text = describe_normalization("imagenet")

    assert "ImageNet" in text
    assert str(IMAGENET_MEAN[0]) in text
