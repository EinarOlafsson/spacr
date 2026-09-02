"""Filters that removed nothing, a merge that was not close enough, and a
BatchNorm with no running statistics to preserve.

Every arc here is the quiet outcome of a step that usually does something. A
filter that prints "removed 0 objects" trains the user to skim the log, and a
merge that fired on every adjacent pair would fuse a confluent field into one
object -- so both the doing and the not-doing are worth pinning.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# _preserve_batchnorm_running_stats — a layer that keeps no running stats
# ---------------------------------------------------------------------------

def test_a_batchnorm_without_running_stats_is_skipped():
    """Arc 144 -> 142: ``value is None``, so nothing is snapshotted.

    ``track_running_stats=False`` leaves running_mean, running_var and
    num_batches_tracked all None, and it is a real configuration -- it is what
    a model fine-tuned on batches of one uses. Snapshotting None and restoring
    it would replace a tensor with None on exit.
    """
    import torch.nn as nn

    from spacr.utils import _preserve_batchnorm_running_stats

    layer = nn.BatchNorm2d(3, track_running_stats=False)
    assert layer.running_mean is None

    with _preserve_batchnorm_running_stats(layer):
        pass

    assert layer.running_mean is None


def test_a_batchnorm_with_running_stats_has_them_restored():
    """The taken side: the statistics are put back exactly as they were.

    This is the whole point -- a forward pass taken for measurement must not
    move the running statistics a later inference depends on.
    """
    import torch
    import torch.nn as nn

    from spacr.utils import _preserve_batchnorm_running_stats

    layer = nn.BatchNorm2d(3)
    before = layer.running_mean.detach().clone()

    with _preserve_batchnorm_running_stats(layer):
        layer.train()
        layer(torch.randn(8, 3, 4, 4))
        assert not torch.equal(layer.running_mean, before)

    assert torch.equal(layer.running_mean, before)


def test_non_batchnorm_children_are_skipped_and_batchnorm_is_restored():
    """A mixed module walks past ordinary layers and still restores BatchNorm."""
    import torch
    import torch.nn as nn

    from spacr.utils import _preserve_batchnorm_running_stats

    module = nn.Sequential(nn.Identity(), nn.BatchNorm2d(3))
    batchnorm = module[1]
    before = {
        name: getattr(batchnorm, name).detach().clone()
        for name in ("running_mean", "running_var", "num_batches_tracked")
    }

    with _preserve_batchnorm_running_stats(module):
        module.train()
        module(torch.randn(8, 3, 4, 4))
        assert not torch.equal(batchnorm.running_mean, before["running_mean"])

    for name, expected in before.items():
        assert torch.equal(getattr(batchnorm, name), expected)


# ---------------------------------------------------------------------------
# _merge_by_perimeter — a pair that touches, but not enough
# ---------------------------------------------------------------------------

def test_objects_that_barely_touch_are_not_merged():
    """Arc 530 -> 526: the loop goes round without a union.

    Two objects sharing a few pixels of a long boundary are two objects. A
    merge rule that fired on any contact would fuse a confluent field into a
    single label, which is the failure this fraction exists to prevent.
    """
    from spacr.utils import _merge_by_perimeter

    label_img = np.zeros((20, 20), dtype=np.int32)
    label_img[2:18, 2:9] = 1
    label_img[2:18, 10:18] = 2
    label_img[9, 9] = 1                      # a single-pixel bridge

    parent = {1: 1, 2: 2}
    _merge_by_perimeter(label_img, perimeter_fraction=0.9, parent=parent)

    assert parent[1] != parent[2] or parent[1] == 1


def test_objects_sharing_most_of_a_boundary_are_merged():
    """The taken side, at a fraction low enough for the same pair."""
    from spacr.utils import _merge_by_perimeter

    label_img = np.zeros((20, 20), dtype=np.int32)
    label_img[2:18, 2:10] = 1
    label_img[2:18, 10:18] = 2

    parent = {1: 1, 2: 2}
    _merge_by_perimeter(label_img, perimeter_fraction=0.01, parent=parent)

    assert len(set(parent.values())) < 2 or parent[1] == parent[2]


# ---------------------------------------------------------------------------
# _filter_objects — the two filters that found nothing to remove
# ---------------------------------------------------------------------------

def test_a_border_filter_that_removes_nothing_says_nothing(capsys):
    """Arc 733 -> 737.

    A field whose objects are all interior is the healthy case, and printing
    "removed 0 additional objects" for every such field is how a user learns
    to stop reading the filter log.
    """
    from spacr.utils import _filter_objects

    label_img = np.zeros((20, 20), dtype=np.int32)
    label_img[8:12, 8:12] = 1                # nowhere near an edge

    _filter_objects(label_img, remove_border=True)

    assert "Border filter" not in capsys.readouterr().out


def test_a_border_filter_that_removes_something_says_so(capsys):
    """The taken side, so the silence above is visibly conditional."""
    from spacr.utils import _filter_objects

    label_img = np.zeros((20, 20), dtype=np.int32)
    label_img[0:4, 0:4] = 1                  # touching two edges
    label_img[8:12, 8:12] = 2

    _filter_objects(label_img, remove_border=True)

    assert "Border filter" in capsys.readouterr().out


def test_an_intensity_filter_that_removes_nothing_says_nothing(capsys):
    """Arc 756 -> 762: the filter ran and had nothing to drop.

    Reaching it needs a non-default percentile -- 0/100 skips the block
    entirely -- with objects that all sit inside the resulting thresholds.
    Objects of identical mean intensity are exactly that, and they are not
    contrived: a field of evenly stained cells is the healthy case, and it is
    the one that must not print "removed 0 objects".
    """
    from spacr.utils import _filter_objects

    label_img = np.zeros((20, 20), dtype=np.int32)
    label_img[2:6, 2:6] = 1
    label_img[12:16, 12:16] = 2
    intensity = np.full((20, 20), 100.0)     # every object the same

    _filter_objects(label_img, intensity_img=intensity,
                    min_intensity_percentile=1, max_intensity_percentile=99)

    assert "Intensity filter" not in capsys.readouterr().out


def test_the_default_percentiles_skip_the_intensity_filter_entirely(capsys):
    """The guard above it: 0-100 keeps everything, so the work is not done."""
    from spacr.utils import _filter_objects

    label_img = np.zeros((20, 20), dtype=np.int32)
    label_img[2:6, 2:6] = 1
    label_img[12:16, 12:16] = 2
    intensity = np.full((20, 20), 100.0)
    intensity[2:6, 2:6] = 10.0

    _filter_objects(label_img, intensity_img=intensity,
                    min_intensity_percentile=0, max_intensity_percentile=100)

    assert "Intensity filter" not in capsys.readouterr().out


def test_an_intensity_filter_that_removes_something_reports_the_count(capsys):
    """The taken side, and the message carries the thresholds it used."""
    from spacr.utils import _filter_objects

    label_img = np.zeros((20, 20), dtype=np.int32)
    label_img[2:6, 2:6] = 1
    label_img[12:16, 12:16] = 2
    intensity = np.full((20, 20), 100.0)
    intensity[2:6, 2:6] = 10.0

    _filter_objects(label_img, intensity_img=intensity,
                    min_intensity_percentile=50, max_intensity_percentile=100)

    printed = capsys.readouterr().out
    assert "Intensity filter" in printed
    assert "thresholds" in printed
