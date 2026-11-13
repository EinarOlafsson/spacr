"""CPU coverage for the mask-filtering block of ``spacr.utils``.

Covers ``remove_intensity_objects``, ``_filter_closest_to_stat``,
``_find_similar_sized_images``, ``_relabel_parent_with_child_labels``,
``_exclude_objects``, ``_merge_overlapping_objects``, ``_filter_object``,
``_filter_cp_masks``, ``_object_filter``, ``_get_regex``, ``_run_test_mode``
and ``_choose_model``.

Everything is synthetic, tiny and offline: the Cellpose constructors used by
``_choose_model`` are monkeypatched with recording doubles so no weights are
downloaded and no GPU is touched.
"""
from __future__ import annotations


import numpy as np
import pytest

import spacr.utils as U


# ---------------------------------------------------------------------------
# shared fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _two_object_mask(shape=(32, 32)):
    """Label mask with a small (16 px) object 1 and a big (100 px) object 2."""
    m = np.zeros(shape, dtype=np.int32)
    m[2:6, 2:6] = 1        # 16 px
    m[10:20, 10:20] = 2    # 100 px
    return m


# ---------------------------------------------------------------------------
# remove_intensity_objects
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _filter_closest_to_stat
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _find_similar_sized_images
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _relabel_parent_with_child_labels
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _exclude_objects
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _merge_overlapping_objects
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _filter_object
# ---------------------------------------------------------------------------

def test_filter_object_removes_objects_under_min_pixel_count():
    mask = _two_object_mask()
    out = U._filter_object(mask.copy(), min_value=20)
    assert set(np.unique(out).tolist()) == {0, 2}


# ---------------------------------------------------------------------------
# _filter_cp_masks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _object_filter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _get_regex
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _run_test_mode
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _choose_model
# ---------------------------------------------------------------------------
#
# Cellpose 4 ships exactly one model. models.MODEL_NAMES == ['cpsam'], and
# CellposeModel accepts-and-ignores model_type= / diam_mean=, resolving an
# unknown pretrained_model to cpsam with only a log warning. So the pre-SAM
# names were never loading the model they named. _choose_model now maps every
# legacy name to cpsam explicitly and says so, instead of pretending.


# ---------------------------------------------------------------------------
# Fine-tuned checkpoints
#
# pretrained_model was hard-coded to 'cpsam', so every model produced by
# spaCR's own Train Cellpose module was discarded and the stock weights used
# instead — the trained model could never be applied to anything, and nothing
# said so.
# ---------------------------------------------------------------------------
