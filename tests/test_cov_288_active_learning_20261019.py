"""Active-learning edges: image filter, feature choices, table features, embeddings, rejections."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.active_learning as al
from tests.test_issue152_suggest_features import _project
from tests.test_an_active_learning_round_says_what_it_could_not_use import _three_class_db


def test_an_image_filter_that_parses_to_nothing_keeps_every_path(monkeypatch):
    import spacr.qt.annotate_engine as engine
    monkeypatch.setattr(engine, "parse_image_type", lambda text: ("", []))
    kept = al._image_type_mask(["a.png", None, "b.png"], "something")
    assert kept.tolist() == [True, True, True]


def test_meta_columns_are_never_features():
    kept = al._feature_columns(["object_label", "prcf", "cell_area"], None, None)
    assert kept == ["cell_area"]


def test_feature_choices_without_a_crop_table_or_a_cell_table(tmp_path):
    db = tmp_path / "m.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE nucleus (prcf TEXT, object_label INT, "
                    "nucleus_channel_1_mean_intensity REAL)")
    choices = al._suggest_feature_choices(str(db))
    assert choices["tables"] == ["nucleus"]
    assert choices["default_tables"] == ["nucleus"]
    assert choices["channels"] == ["channel_1"]
    assert al.NO_CHANNEL not in choices["channels"]


def test_an_object_table_without_its_key_columns_says_to_re_run_measure():
    frame = pd.DataFrame({"object_label": [1], "cell_area": [3.0]})
    with pytest.raises(ValueError) as caught:
        al._object_table_features(frame, "cell", None, None)
    assert "Re-run Measure" in str(caught.value)


def test_a_repeated_object_keeps_its_first_row_and_its_parent():
    frame = pd.DataFrame({"prcf": ["p_r1_c1_f1"] * 3, "object_label": [1, 1, 2],
                          "cell_id": [5.0, 6.0, 7.0],
                          "nucleus_area": [10.0, 99.0, 20.0]})
    features, parents = al._object_table_features(frame, "nucleus", None, None)
    assert list(features["nucleus_area"]) == [10.0, 20.0]
    assert len(parents) == 2
    assert parents.iloc[0].endswith("_o5")


def test_child_means_without_a_limit_and_without_a_cell_link(tmp_path):
    db, crops = _project(tmp_path)
    unlimited = al.round_features(db, tables=["cell", "nucleus"], nuclei_limit=None)
    assert len(unlimited) == len(crops)
    assert unlimited["nucleus_area"].notna().all()
    with sqlite3.connect(db) as con:
        nuclei = pd.read_sql("SELECT * FROM nucleus", con).drop(columns=["cell_id"])
        nuclei.to_sql("nucleus", con, index=False, if_exists="replace")
    unlinked = al.round_features(db, tables=["cell", "nucleus"])
    assert len(unlinked) == len(crops)
    assert unlinked.filter(like="nucleus").isna().all().all()
    assert unlinked.filter(like="cell").notna().all().all()


def test_crops_with_no_measured_object_at_all_say_so(tmp_path):
    db, _ = _project(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("DROP TABLE cell")
        con.execute("CREATE TABLE cell (prcf TEXT, object_label REAL, cell_area REAL)")
        con.execute("INSERT INTO cell VALUES ('plate1_r1_c1_f1', NULL, 5.0)")
    with pytest.raises(ValueError) as caught:
        al.round_features(db, tables=["cell"])
    assert "hold no measured object" in str(caught.value)


def test_storing_the_same_objects_again_replaces_them(tmp_path):
    db = str(tmp_path / "m.db")
    assert al._store_crop_embeddings(db, ["o1", "o2"], np.ones((2, 3))) == 2
    assert al._store_crop_embeddings(db, ["o1", "o2"], np.zeros((2, 3))) == 2
    stored = al._stored_embedding_frame(db)
    assert len(stored) == 2
    assert float(stored.drop(columns=["prcfo"]).to_numpy().sum()) == 0.0


def test_a_round_summary_without_per_class_numbers_has_no_per_class_line():
    result = al.RoundResult(round_index=1, n_labels=10, n_new_labels=2,
                            report={"n": 4, "accuracy": 0.5, "f1_macro": 0.4},
                            split_rule="by well", scored=3, score_columns=["s"])
    text = result.summary()
    assert "Per class" not in text
    assert "Held-out accuracy 0.500" in text


def test_a_rejection_in_a_two_class_column_is_fitted_as_the_other_class(tmp_path):
    db, rows, features = _three_class_db(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE png_list SET annotate = 1 + (annotate % 2) "
                    "WHERE annotate IS NOT NULL")
    rejected = {rows[k][0]: 1 for k in (61, 63)}
    result = al.retrain_round(db, "annotate", features=features, seed=0,
                              save_model=False, write_card=False,
                              rejections=rejected)
    assert any(n.startswith("2 rejected suggestions were fitted")
               for n in result.notes), result.notes
    assert not any("could not be fitted" in n for n in result.notes)


def test_rejections_that_cannot_be_used_are_counted_and_the_rest_converted():
    keys, labels, unusable = al._rejections_as_labels(
        {"a": 1, "b": 2, "labelled": 1, "unscored": 1, "odd": 3},
        [1, 2], {"labelled"}, {"a", "b", "labelled", "odd"})
    assert keys == ["a", "b"] and labels == [2, 1] and unusable == 1
    keys, labels, unusable = al._rejections_as_labels({"a": 1}, [1, 2, 3], set(), {"a"})
    assert keys == [] and unusable == 1


def test_only_one_and_two_have_an_other_class():
    assert al._absent_binary_class(1) == 2
    assert al._absent_binary_class(2) == 1
    assert al._absent_binary_class(3) is None
    assert al._absent_binary_class("x") is None
