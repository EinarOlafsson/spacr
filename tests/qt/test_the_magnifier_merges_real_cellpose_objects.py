"""Opt-in real-model evidence, separate from fast deterministic GUI tests."""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("SPACR_CELLPOSE_E2E"),
        reason="set SPACR_CELLPOSE_E2E=1 with local cpsam weights",
    ),
]


def test_real_cellpose_objects_follow_both_drag_save_modes():
    from cellpose import models

    from spacr.qt._magnifier_drag import _DragStroke
    from spacr.qt.screens import make_masks as mm

    # An explicitly requested proof must not silently download weights or
    # replace the real model with Classical when a dependency is missing.
    assert (Path(models.MODEL_DIR) / "cpsam").is_file()
    field = np.zeros((160, 160), dtype=np.uint16)
    yy, xx = np.mgrid[:160, :160]
    centres = ((40, 40), (40, 110), (110, 45), (115, 115))
    for cy, cx in centres:
        field[(yy - cy) ** 2 + (xx - cx) ** 2 <= 18 ** 2] = 4000
    request = mm._MagnifierRequest(
        key=("real-cpsam",), crop=field, box=(0, 0, 160, 160),
        shape=field.shape, mode="cellpose", sensitivity=0,
        bright=True, min_area=50, model_name="cpsam", diameter=0,
        colour=(255, 0, 0), flow_threshold=0.4, cellprob_threshold=0.0,
        normalize=True,
    )
    labels, used, note = mm._segment_region(request)
    assert used == "cellpose" and not note, (used, note)
    objects = [int(labels[y, x]) for y, x in centres]
    assert 0 not in objects and len(set(objects)) == 4
    assert len(np.unique(labels)) == 5

    for keep_untouched in (False, True):
        stroke = _DragStroke(field.shape, (40, 40), step=40,
                             keep_untouched=keep_untouched)
        stroke.expect("full")
        stroke.deliver("full", labels, request.box)
        stroke.extend((110, 40))
        stroke.release()
        assert stroke.ready()
        outcome = stroke.outcome()
        x0, y0 = outcome.origin
        actual = np.zeros_like(labels)
        h, w = outcome.labels.shape
        actual[y0:y0 + h, x0:x0 + w] = outcome.labels
        expected_foreground = labels > 0 if keep_untouched else np.isin(labels, objects[:2])
        np.testing.assert_array_equal(actual > 0, expected_foreground)
        assert actual[40, 40] == actual[40, 110] > 0
        assert outcome.objects == (3 if keep_untouched else 1)
