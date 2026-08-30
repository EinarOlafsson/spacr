"""The v2 scorecard's promise: a plate never loses its masks to a QC bug.

``_score_v2_masks``'s docstring states the rule -- "Never raises into a
finished run. A plate that has just spent hours segmenting must not lose its
masks to a scorecard bug". The handler that keeps that promise had never
executed, which is the shape this whole sweep keeps finding: the guarantee is
written down, relied on, and unexercised.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest


def _v2_plate(tmp_path):
    """A plate folder with the ``merged/`` layout v2 writes."""
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "channel_order.json").write_text(json.dumps({
        "image_channels": ["dapi"], "mask_channels": ["cell"]}))
    np.save(merged / "stack_plate1_A01_F001.npy",
            np.zeros((8, 8, 2), dtype=np.uint16))
    return tmp_path


def test_a_scorecard_that_raises_costs_the_score_and_not_the_run(tmp_path,
                                                                 capsys,
                                                                 monkeypatch):
    """Lines 81-84: the exception is caught, named, and answered with None.

    The message carries the exception TYPE and text, because a run that
    silently returned None here would leave the user unable to tell QC being
    off from QC being broken.
    """
    from spacr import core
    from spacr import seg_qc

    def explode(*_args, **_kwargs):
        raise RuntimeError("the scorecard writer fell over")

    monkeypatch.setattr(seg_qc, "run_segmentation_qc", explode)

    out = core._score_v2_masks(_v2_plate(tmp_path), {"verbose": True},
                               object_type="cell")

    assert out is None
    printed = capsys.readouterr().out
    assert "Segmentation QC skipped for cell" in printed
    assert "RuntimeError" in printed
    assert "the scorecard writer fell over" in printed


def test_qc_turned_off_returns_none_without_a_message(tmp_path, capsys):
    """The early return, which must be distinguishable from the failure above.

    Off is a choice and prints nothing. A failure prints. That is the whole
    reason the handler names the exception rather than returning quietly.
    """
    from spacr import core

    out = core._score_v2_masks(_v2_plate(tmp_path), {"seg_qc": "off"},
                               object_type="cell")

    assert out is None
    assert "Segmentation QC skipped" not in capsys.readouterr().out


def test_a_plate_with_no_mask_channel_returns_none(tmp_path, capsys):
    """No source to score: also None, also quiet.

    v2_mask_source hands back nothing when the sidecar names no mask, and
    scoring nothing is the honest answer to "there is no mask here" -- the
    caller says so rather than this raising.
    """
    from spacr import core

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "channel_order.json").write_text(json.dumps({
        "image_channels": ["dapi"], "mask_channels": []}))

    out = core._score_v2_masks(tmp_path, {"verbose": True}, object_type="cell")

    assert out is None


def test_a_missing_merged_folder_does_not_raise(tmp_path):
    """The same promise against the commonest cause: nothing was written."""
    from spacr import core

    assert core._score_v2_masks(tmp_path, {"verbose": True},
                                object_type="cell") is None
