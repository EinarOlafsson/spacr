"""Attention MIL learns which cells carry a well label from well labels only.

Validated on synthetic wells with planted responder cells: the held-out
attention ranks responders above the other cells of positive wells, and the
held-out well prediction beats a classifier on each well's mean cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from spacr.embeddings import (_mil_bags, _mil_fit, _mil_from_table,
                              _mil_predict, _mil_scorecard,
                              _synthetic_mil_bags)

_FIT = {"epochs": 40}


def test_attention_finds_planted_responders_and_beats_the_mean_baseline():
    bags, labels, responders = _synthetic_mil_bags(
        wells=32, cells=50, features=8, fraction=0.1, seed=3)
    card = _mil_scorecard(bags, labels, responders=responders, folds=4,
                          **_FIT)
    assert card["attention_auroc"] > 0.85
    assert card["evidence_auroc"] > 0.85
    assert card["mil_auroc"] > 0.8
    assert card["mil_auroc"] > card["mean_auroc"]


def test_attention_is_an_equal_share_scale_per_well():
    bags, labels, _ = _synthetic_mil_bags(wells=8, cells=20, features=4)
    fitted = _mil_fit(bags, labels, hidden=8, epochs=2)
    probs, attention, evidence = _mil_predict(fitted, bags)
    assert probs.shape == (8,) and np.all((probs >= 0) & (probs <= 1))
    assert all(np.isclose(a.mean(), 1.0, atol=1e-4) for a in attention)
    assert [e.shape for e in evidence] == [(20,)] * 8


def _table(seed=0):
    bags, labels, responders = _synthetic_mil_bags(
        wells=16, cells=30, features=6, fraction=0.2, seed=seed)
    rows = []
    for well, (bag, label, hit) in enumerate(zip(bags, labels, responders)):
        frame = pd.DataFrame(bag, columns=[f"emb_{i}" for i in range(6)])
        frame.insert(0, "well_label", int(label))
        frame.insert(0, "well", f"A{well:02d}")
        frame["area"] = 100.0
        frame["planted"] = hit
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_the_table_route_uses_embedding_columns_and_keeps_every_cell():
    frame = _table()
    bags, labels, wells, rows, kept = _mil_bags(frame)
    assert bags[0].shape == (30, 6)
    assert labels.sum() == 8 and len(wells) == 16
    cells, well_frame, card = _mil_from_table(frame, folds=2, **_FIT)
    assert len(cells) == len(frame)
    assert {"mil_attention", "mil_evidence", "planted", "area"} <= set(cells.columns)
    assert list(well_frame.columns) == ["well", "well_label",
                                        "mil_probability", "cells"]
    assert set(card) >= {"mil_auroc", "mean_auroc", "mean_sd_auroc"}


def test_refuses_labels_it_cannot_read():
    frame = _table()
    mixed = frame.copy()
    mixed.loc[0, "well_label"] = 1 - mixed.loc[0, "well_label"]
    with pytest.raises(ValueError, match="more than one label"):
        _mil_bags(mixed)
    named = frame.assign(well_label=frame["well_label"].map(
        {0: "control", 1: "treated"}))
    with pytest.raises(ValueError, match="name the positive label"):
        _mil_bags(named)
    assert _mil_bags(named, positive="treated")[1].sum() == 8
    with pytest.raises(ValueError, match="no 'wellID' column"):
        _mil_bags(frame.drop(columns="well"))
