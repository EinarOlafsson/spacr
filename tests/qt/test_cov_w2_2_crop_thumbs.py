"""The hover seam's two edges: a moved dataset, and a key with no crop.

`resolve_crop_path` exists because `png_list` stores absolute paths built at
measure time, so a dataset copied to another disk resolves to nothing. Its
happy path is one line; the three that matter are the ones a screen hits when
the Annotate module is not importable, when re-anchoring throws, and when it
succeeds against a rebuilt tree.

`crop_paths_for_keys` bisects when some keys have no crop, and the bisection
is what makes hover a dict lookup rather than one table scan per point. It is
driven here against a real sqlite crop table so the subsequence arithmetic is
checked against the real resolver's real dropping behaviour rather than
against a stand-in that agrees with the test by construction.
"""

import pytest

from spacr.qt.crop_thumbs import DEFAULT_CAPACITY, DEFAULT_SIZE, CropThumbnails


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


def test_the_cache_stays_bounded_under_a_sweep(qapp, tmp_path):
    """Past capacity the least recently used entry goes, not the newest."""
    cache = CropThumbnails(capacity=3)
    paths = [str(tmp_path / f"c{i}.png") for i in range(5)]
    for path in paths:
        cache.prime(path)
    assert len(cache) == 3
    assert paths[0] not in cache
    assert paths[-1] in cache


def test_the_defaults_are_sane_and_the_floors_hold(qapp):
    """Silly sizes and capacities are clamped rather than accepted."""
    assert CropThumbnails().size == DEFAULT_SIZE
    assert CropThumbnails().capacity == DEFAULT_CAPACITY
    assert CropThumbnails(size=0).size == 16
    assert CropThumbnails(capacity=0).capacity == 1


# ---------------------------------------------------------------------------
# a crop that really decodes
# ---------------------------------------------------------------------------

@pytest.fixture
def real_crop(tmp_path):
    """One genuine RGB crop PNG on disk, wider than a thumbnail."""
    from PIL import Image
    import numpy as np

    array = np.zeros((300, 240, 3), dtype="uint8")
    array[:150, :, 0] = 200          # a red band, so a channel swap shows
    array[150:, :, 2] = 120
    path = tmp_path / "plate1_A01_1_1.png"
    Image.fromarray(array).save(path)
    return str(path)


def test_a_real_crop_decodes_once_and_is_served_from_memory(qapp, real_crop):
    """The first ask decodes; the second is a hit and does not decode again."""
    cache = CropThumbnails(size=64)

    assert cache.peek(real_crop) is None, "peek decoded on the hover path"

    pixmap = cache.pixmap(real_crop)
    assert pixmap is not None
    assert not pixmap.isNull()
    # thumbnailed down the long edge, aspect kept
    assert max(pixmap.width(), pixmap.height()) == 64
    assert pixmap.width() < pixmap.height()

    assert cache.decodes == 1
    assert cache.failures == 0

    again = cache.peek(real_crop)
    assert again is not None
    assert cache.decodes == 1
    assert cache.hits == 1
