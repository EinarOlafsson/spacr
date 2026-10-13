"""Detection cleanup accepts empty fields and can remove the last object."""
import numpy as np
import pytest

from spacr.qt import detect_chain as dc


@pytest.mark.parametrize("shape", [(0, 0), (1, 1), (48, 64)])
def test_empty_detection_stays_empty_with_morphology_and_split_enabled(shape):
    labels = np.zeros(shape, dtype=np.uint16)
    before = labels.copy()
    chain = dc.Chain(morphology="open_close", morphology_radius=3, split=True)
    out = dc.finish(labels, chain, intensity=np.zeros(shape, dtype=np.float32))
    assert out.shape == shape
    assert out.dtype == labels.dtype
    assert not out.any()
    np.testing.assert_array_equal(labels, before)


@pytest.mark.parametrize("split", [False, True])
def test_opening_can_remove_the_last_objects_without_mutating_detection(split):
    labels = np.zeros((32, 40), dtype=np.uint16)
    labels[8, 8] = 4
    labels[20:22, 25:27] = 9
    before = labels.copy()
    chain = dc.Chain(morphology="open", morphology_radius=3, split=split)
    out = dc.finish(labels, chain, intensity=(labels > 0).astype(np.float32))
    assert out.shape == labels.shape
    assert out.dtype == labels.dtype
    assert not out.any()
    np.testing.assert_array_equal(labels, before)
