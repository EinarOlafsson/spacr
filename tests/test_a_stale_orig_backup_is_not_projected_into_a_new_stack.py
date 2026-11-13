"""A stale ``orig/`` backup never leaks into a freshly built stack.

``_rename_and_organize_image_files`` reads raw images from ``src`` and from
``src/orig/`` so a rerun can finish a plate an earlier run left undone. When
the same file name sat in both folders, both copies were listed under the
same field and channel and max-projected together: a rerun on a folder that
already held an ``orig/`` backup produced channel maxima ``[999, 32]`` instead
of ``[31, 32]``. A name present in ``src`` is now read only from ``src``; the
``orig/`` copy is still kept untouched, and a field whose raw images live only
in ``orig/`` is still rebuilt from there.
"""

import os

import numpy as np
import pytest
import tifffile

from spacr import io as IO

REGEX = (r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+)_T(?P<timeID>\d+)F(?P<fieldID>\d+)"
         r"L(?P<laserID>\d+)A(?P<AID>\d+)Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif")


def _name(channel):
    return f'plate1_A01_T0001F001L01A01Z01C{channel}.tif'


def _write(path, value):
    tifffile.imwrite(path, np.full((6, 8), value, np.uint16))


def _run(src, save_original_images):
    return IO._rename_and_organize_image_files(
        str(src), REGEX, batch_size=10, metadata_type='custom',
        img_format=['.tif'], save_original_images=save_original_images)


def _maxima(src):
    stack = np.load(os.path.join(src, 'stack', 'plate1_A01_1_1.npy'))
    return [int(stack[..., c].max()) for c in range(stack.shape[-1])]


@pytest.fixture
def plate(tmp_path):
    src = tmp_path / 'plate1'
    (src / 'orig').mkdir(parents=True)
    _write(src / _name('01'), 31)
    _write(src / _name('02'), 32)
    return src


@pytest.mark.parametrize('save_original_images', [True, False])
def test_stale_backup_with_the_same_name_is_not_max_projected(plate, save_original_images):
    """The fresh ``src`` image alone makes channel 1, whatever ``orig/`` holds."""
    _write(plate / 'orig' / _name('01'), 999)
    assert _run(plate, save_original_images) == 2
    assert _maxima(plate) == [31, 32]


def test_stale_backup_is_kept_when_originals_are_saved(plate):
    """``orig/`` keeps its copy; the clashing ``src`` image is left where it was, not overwritten."""
    _write(plate / 'orig' / _name('01'), 999)
    _run(plate, True)
    assert int(tifffile.imread(plate / 'orig' / _name('01')).max()) == 999
    assert int(tifffile.imread(plate / _name('01')).max()) == 31
    assert int(tifffile.imread(plate / 'orig' / _name('02')).max()) == 32
    assert not (plate / _name('02')).exists()


def test_stale_backup_is_never_touched_when_originals_are_deleted(plate):
    """With ``save_original_images`` off only ``src`` images are deleted; ``orig/`` is never touched."""
    _write(plate / 'orig' / _name('01'), 999)
    _run(plate, False)
    assert sorted(os.listdir(plate / 'orig')) == [_name('01')]
    assert int(tifffile.imread(plate / 'orig' / _name('01')).max()) == 999
    assert not (plate / _name('01')).exists()
    assert not (plate / _name('02')).exists()


def test_without_a_backup_the_stack_and_move_are_unchanged(plate):
    """No ``orig/`` content: same maxima, and every raw image is moved aside."""
    assert _run(plate, True) == 2
    assert _maxima(plate) == [31, 32]
    assert sorted(os.listdir(plate / 'orig')) == [_name('01'), _name('02')]


def test_a_field_set_aside_only_in_orig_is_still_rebuilt(plate):
    """An image present only in ``orig/`` still feeds its channel, so resume keeps working."""
    os.replace(plate / _name('02'), plate / 'orig' / _name('02'))
    assert _run(plate, True) == 2
    assert _maxima(plate) == [31, 32]
