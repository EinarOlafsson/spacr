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


def _embedded_project(tmp_path):
    """A measurements.db whose crops carry stored embeddings.

    Row A's four wells are positive; a third of their cells have a raised
    first embedding dimension. Row B's four wells are negative.
    """
    import sqlite3
    from types import SimpleNamespace

    from spacr import active_learning as al
    from tests.test_cov_active_learning_rounds import _make_project

    wells = [(r, str(c)) for r in "AB" for c in range(1, 5)]
    project = _make_project(tmp_path, per_well=12, wells=wells)
    with sqlite3.connect(project["db"]) as con:
        rows = con.execute("SELECT prcfo, rowID FROM png_list").fetchall()
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, (len(rows), 4)).astype(np.float32)
    for i, (key, row) in enumerate(rows):
        if row == "A" and int(key.split("_o")[-1]) % 3 == 0:
            values[i, 0] += 4.0
    al._store_crop_embeddings(project["db"], [r[0] for r in rows],
                              SimpleNamespace(values=values, columns=tuple(
                                  f"emb_{i}" for i in range(4))))
    labels = pd.DataFrame({"well": [f"{r}{c}" for r, c in wells],
                           "well_label": [int(r == "A") for r, c in wells]})
    return project["db"], labels


def test_stored_embeddings_join_object_ids_and_wells(tmp_path):
    from spacr.embeddings import _mil_frame_from_db, _mil_from_table

    db, labels = _embedded_project(tmp_path)
    frame = _mil_frame_from_db(db)
    assert len(frame) == 96 and frame["wellID"].nunique() == 8
    assert {"prcfo", "png_path", "plateID", "rowID", "columnID", "object",
            "emb_0"} <= set(frame.columns)
    assert (frame["wellID"] == "plate1_" + frame["rowID"] + "_"
            + frame["columnID"]).all()
    assert all(key.endswith("_" + obj)
               for key, obj in zip(frame["prcfo"], frame["object"]))
    labelled = _mil_frame_from_db(db, labels)
    assert labelled["well_label"].notna().all()
    assert (labelled["well_label"] == (labelled["rowID"] == "A")).all()
    cells, wells, card = _mil_from_table(labelled, folds=2, epochs=30)
    assert len(wells) == 8 and "prcfo" in cells.columns
    planted = cells["rowID"].eq("A") & cells["object"].str[1:].astype(
        int).mod(3).eq(0)
    assert cells.loc[planted, "mil_evidence"].mean() > cells.loc[
        ~planted, "mil_evidence"].mean()


def test_a_database_without_stored_embeddings_is_refused(tmp_path):
    from spacr.embeddings import _mil_frame_from_db
    from tests.test_cov_active_learning_rounds import _make_project

    db = _make_project(tmp_path)["db"]
    with pytest.raises(ValueError, match="no stored crop embeddings"):
        _mil_frame_from_db(db)
    db, labels = _embedded_project(tmp_path / "e")
    with pytest.raises(ValueError, match="no 'well_label' column"):
        _mil_frame_from_db(db, labels.drop(columns="well_label"))
