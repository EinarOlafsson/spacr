"""Item 1059: a ph_measure worker keeps less, and measures the same."""
import numpy as np

from spacr import measure as M
from spacr import ops_engine as E


def _field(seed=0, side=96, channels=3):
    """A small labelled field and its image."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((side, side), np.int64)
    for k, (y, x) in enumerate([(20, 20), (60, 30), (30, 70), (70, 70)], start=1):
        labels[y - 9:y + 9, x - 9:x + 9] = k
    image = rng.random((side, side, channels)) * 100
    return labels, image


def test_the_coloc_families_give_the_same_columns_when_the_sort_is_let_go():
    labels, image = _field()
    with M._shared_foregrounds():
        lab, img, _present = M._profile_inputs(labels, image)
        kept = M._label_pixels(lab, img)
        assert not kept._sort_cache
        frame = M.measure_colocalisation_otsu(lab, img)
        assert not kept._sort_cache
    plain = M._LabelPixels(lab, img)
    for channel in range(img.shape[-1]):
        once = kept._sort_once(channel)
        again = plain._sort(channel)
        for a, b in zip(once, again):
            assert np.array_equal(a, b)
    assert list(frame.columns)[0] == "label" and len(frame) == 4


def test_the_flat_reciprocals_are_made_on_asking_and_equal_the_stored_ones():
    planes = {"DAPI": np.full((8, 8), 0.8, np.float32), "GFP": np.full((8, 8), 1.25, np.float32)}
    inverse = E._FlatInverse(planes)
    assert set(inverse) == {"DAPI", "GFP"} and len(inverse) == 2
    for channel, plane in planes.items():
        expected = 1.0 / plane.astype(np.float64)
        assert inverse[channel].dtype == np.float64
        assert np.array_equal(inverse[channel], expected)
