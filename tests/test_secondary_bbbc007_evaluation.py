"""Manual-outline evaluation must pair nuclei correctly and ignore uncertain lines."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'secondary_evaluation', Path(__file__).parents[1] / 'tools/evaluate_secondary_bbbc007.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def outline_rect(array, top, left, bottom, right):
    array[top:bottom + 1, [left, right]] = True
    array[[top, bottom], left:right + 1] = True


def test_border_connected_background_and_tiny_outline_slivers_are_excluded():
    outline = np.zeros((40, 40), bool)
    outline_rect(outline, 3, 3, 20, 20)
    outline_rect(outline, 25, 25, 27, 27)
    labels, rejected = evaluation.interiors(outline)
    assert labels[10, 10] > 0
    assert not labels[0].any()
    assert labels[26, 26] == 0
    assert rejected == {'border_components': 1, 'tiny_components': 1}


def test_only_one_to_one_majority_overlap_pairs_enter_the_reference():
    nuclear, cell = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    outline_rect(cell, 2, 2, 30, 30)
    outline_rect(cell, 2, 32, 30, 60)
    outline_rect(nuclear, 8, 8, 16, 16)
    outline_rect(nuclear, 8, 38, 16, 46)
    outline_rect(nuclear, 18, 48, 26, 56)
    primary, reference, ids, counts = evaluation.paired_interiors(nuclear, cell)
    assert len(ids) == 1
    assert counts['nuclei'] == 3 and counts['ambiguous_cells'] == 1
    assert counts['unpaired_nuclei'] == 2
    assert reference[25, 25] == primary[10, 10] == ids[0]
    assert not reference[:, 32:].any()


def test_outline_pixels_do_not_change_iou_but_wrong_identity_does():
    target = np.zeros((10, 10), np.uint16)
    target[3:7, 3:7] = 900
    ignored = np.zeros_like(target, bool)
    ignored[2, 3:7] = True
    predicted = target.copy()
    predicted[ignored] = 900
    score = evaluation.object_scores(predicted, target, [900], ignored)[0]
    assert score == {'id': 900, 'iou': 1., 'dice': 1.}
    predicted[3:7, 3:7] = 7
    assert evaluation.object_scores(predicted, target, [900], ignored)[0]['iou'] == 0


def test_all_three_filename_conventions_are_paired_and_missing_channels_fail():
    class Archive:
        def namelist(self):
            return ['root/a/p1d.tif', 'root/a/p1f.tif', 'root/b/p_D_1UL.tif',
                    'root/b/p_F_2UL.tif', 'root/c/f00d0.tif', 'root/c/f00d1.tif']

    names, pairs = evaluation.image_pairs(Archive())
    assert len(names) == 6 and len(pairs) == 3
    assert ('c/f00d0.tif', 'c/f00d1.tif') in pairs
    broken = Archive()
    broken.namelist = lambda: ['root/a/p1d.tif']
    with pytest.raises(ValueError, match='Missing actin'):
        evaluation.image_pairs(broken)
