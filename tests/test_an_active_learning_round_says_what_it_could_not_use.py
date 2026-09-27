"""Item 288: an active-learning round's edges -- rejections, balance, XGBoost.

* A rejected suggestion is fitted as "the other class", which exists only
  in a column whose classes are 1 and 2. In a three-class column nothing is
  added to the fit, and the round says how many rejections it could not
  use rather than dropping them silently.
* Down-sampling to the smallest class is a no-op with one class.
* ``model_type='xgboost'`` builds a binary or multi-class booster by the
  class count, and without the xgboost package says what to install and
  what to use instead.
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

import spacr.active_learning as al


def _three_class_db(tmp_path):
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, feats = [], []
    i = 0
    for plate in ("p1", "p2"):
        for row_id in ("r1", "r2"):
            for column in ("c1", "c2", "c3"):
                for _ in range(10):
                    latent = 1 + i % 3
                    path = f"/crops/cell_{i:04d}.png"
                    rows.append((path, plate, row_id, column, "f1", None))
                    feats.append({"png_path": path,
                                  "signal": latent + rng.normal(0, 0.2),
                                  "noise": rng.normal(0, 1.0)})
                    i += 1
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE png_list (png_path TEXT, plateID TEXT, '
                     'rowID TEXT, columnID TEXT, fieldID TEXT, '
                     'annotate INTEGER)')
        conn.executemany('INSERT INTO png_list VALUES (?,?,?,?,?,?)', rows)
        conn.executemany('UPDATE png_list SET annotate=? WHERE png_path=?',
                         [(1 + k % 3, rows[k][0]) for k in range(60)])
    return str(db), rows, pd.DataFrame(feats).set_index("png_path")


def test_rejections_in_a_three_class_column_are_counted_not_fitted(tmp_path):
    db, rows, features = _three_class_db(tmp_path)
    rejected = {rows[k][0]: 1 for k in (61, 63, 65)}
    result = al.retrain_round(db, "annotate", features=features, seed=0,
                              save_model=False, write_card=False,
                              rejections=rejected)
    assert any(n.startswith("3 rejected suggestions could not be fitted")
               for n in result.notes), result.notes
    assert not any("rejected suggestions were fitted" in n
                   for n in result.notes)


def test_one_class_is_not_down_sampled():
    index = pd.Index(["a", "b", "c"])
    kept, labels, dropped = al._downsample_to_smallest(index, [1, 1, 1], 0)
    assert list(kept) == ["a", "b", "c"]
    assert labels == [1, 1, 1] and dropped == 0


@pytest.mark.parametrize("n_classes, objective", [
    (2, "binary:logistic"), (3, "multi:softprob")])
def test_xgboost_is_built_for_the_class_count(n_classes, objective):
    pytest.importorskip("xgboost")
    model = al._build_round_model("XGB", seed=3, n_classes=n_classes)
    assert type(model).__name__ == "XGBClassifier"
    params = model.get_params()
    assert params["objective"] == objective
    assert params["random_state"] == 3


def test_without_xgboost_the_error_names_the_alternative(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ValueError) as excinfo:
        al._build_round_model("xgboost", seed=0, n_classes=2)
    assert "pip install xgboost" in str(excinfo.value)
    assert "gradient_boosting" in str(excinfo.value)
