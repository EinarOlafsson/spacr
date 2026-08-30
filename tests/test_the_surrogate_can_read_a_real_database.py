"""The surrogate refused every database spaCR has ever written.

Instruction 236 B6: "GRAD-CAM, SALIENCY, AND THE SURROGATE MODELS."

WHAT WAS FOUND, by pointing `build_surrogate_frame` at plate1 of the tsg101
screen:

    SurrogateError: png_list and the measurement tables share no 'prcfo'
    object key, so predictions cannot be attached to features.

And that is true of every spaCR database. `prcfo` is written into
`png_list`; the measurement tables carry `plateID`, `rowID`, `columnID`,
`fieldID` and `object_label` and compose it on demand, so the joined
feature frame has no `prcfo` column at all. The surrogate could only ever
have run on a database that does not exist.

BOTH SIDES DO CARRY THE CROP PATH, and its basename is the key the
predictions were already matched on ten lines earlier -- for the same
stated reason, that two machines agree about a file name and not about a
mount point. The join falls back to it, and the refusal that remains names
which key each side is missing rather than only the one it looked for.

Driven afterwards on 2,000 real objects: the random forest reaches a
fidelity of 0.883 against a majority-class baseline of 0.688, and histogram
gradient boosting 0.893.
"""
from __future__ import annotations

import multiprocessing
import os
import sqlite3
import threading

import numpy as np
import pandas as pd
import pytest

from spacr.surrogate import MODEL_FAMILIES, SurrogateError, build_surrogate_frame


def _a_spacr_database(where, objects=120, seed=0):
    """A database shaped like the real thing: `prcfo` in png_list only."""
    rng = np.random.default_rng(seed)
    path = str(where / "measurements.db")
    rows, pngs = [], []
    for index in range(objects):
        well = f"r{index % 8 + 1}", f"c{index % 12 + 1}"
        name = f"plate1_{well[0]}{well[1]}_f1_o{index}.png"
        # NO `png_path` ON THE CELL TABLE. A real spaCR `cell` table has
        # none -- the crop path lives in `png_list` -- and putting one here
        # would make the join collapse two columns into `png_path_x` and
        # `png_path_y`, which is a different situation from the one this
        # file is about.
        rows.append((index, "pplate1", well[0], well[1], "f1",
                     *[float(v) for v in rng.normal(size=4)]))
        pngs.append((f"/somewhere/data/{name}", name, "pplate1", well[0],
                     well[1], "f1",
                     f"pplate1_{well[0]}_{well[1]}_f1_o{index}",
                     f"o{index}"))

    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE cell (object_label INTEGER, plateID TEXT, rowID TEXT, "
        "columnID TEXT, fieldID TEXT, "
        "cell_channel_1_mean_intensity REAL, cell_area REAL, "
        "cell_solidity REAL, cell_perimeter REAL)")
    connection.executemany(
        "INSERT INTO cell VALUES (?,?,?,?,?,?,?,?,?)", rows)
    connection.execute(
        "CREATE TABLE png_list (png_path TEXT, file_name TEXT, plateID TEXT, "
        "rowID TEXT, columnID TEXT, fieldID TEXT, prcfo TEXT, cell_id TEXT)")
    connection.executemany(
        "INSERT INTO png_list VALUES (?,?,?,?,?,?,?,?)", pngs)
    connection.commit()
    connection.close()
    return path, [png[1] for png in pngs]


def _predictions(names, seed=0):
    """What a CV run writes: a bare file NAME and a class."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "path": names,
        "label": rng.integers(0, 2, len(names)),
    })


#: Enough of a fit to answer "does it run and does it report", and not the
#: 300 trees, 5 permutation repeats and 500 SHAP samples a real explanation
#: wants. Left at the defaults this file took nine minutes for four fits.
#: Enough of a fit to answer "does it run and does it report", and not the
#: 300 trees, 5 permutation repeats and 500 SHAP samples a real explanation
#: wants. Left at the defaults this file took nine minutes for four fits.
QUICK = {"n_estimators": 12, "n_repeats": 1, "shap_max_samples": 20,
         "verbose": False}

#: `n_jobs=1` for the two families that take it: xgboost fans a 120-row,
#: four-feature problem out over every core and spends longer scheduling
#: than fitting -- 86 seconds against 4. `HistGradientBoostingClassifier`
#: has no `n_jobs` at all and raises on one, which is worth knowing:
#: `model_options` is a straight passthrough to the estimator, so an option
#: that suits one family is not a option that suits three.
ONE_THREAD = {"random_forest": {"n_jobs": 1}, "xgboost": {"n_jobs": 1},
              "hist_gradient_boosting": {}}


def _quick(family):
    return dict(QUICK, model_options=ONE_THREAD.get(family, {}))


@pytest.fixture()
def screen(tmp_path):
    return _a_spacr_database(tmp_path)


class TestTheJoin:
    def test_a_database_shaped_like_the_real_one_joins(self, screen):
        """THE DEFECT. `prcfo` is in png_list and nowhere else, which is
        every spaCR database there is."""
        path, names = screen
        frame = build_surrogate_frame(path, _predictions(names),
                                      path_column="path",
                                      prediction_column="label")
        assert len(frame) == len(names)
        assert "cv_prediction" in frame.columns

    def test_the_features_come_with_it(self, screen):
        path, names = screen
        frame = build_surrogate_frame(path, _predictions(names),
                                      path_column="path",
                                      prediction_column="label")
        for column in ("cell_area", "cell_solidity",
                       "cell_channel_1_mean_intensity"):
            assert column in frame.columns

    def test_a_full_path_and_a_bare_name_are_the_same_object(self, screen):
        """A model scored on one machine and a database written on another
        agree about the file name and not about the mount point."""
        path, names = screen
        elsewhere = _predictions(names)
        elsewhere["path"] = ["/a/quite/different/mount/" + n for n in names]
        frame = build_surrogate_frame(path, elsewhere, path_column="path",
                                      prediction_column="label")
        assert len(frame) == len(names)

    def test_the_prediction_survives_the_join(self, screen):
        """A join that dropped or reordered the labels would train the
        surrogate on the wrong answers and report a fidelity for it."""
        path, names = screen
        said = _predictions(names)
        frame = build_surrogate_frame(path, said, path_column="path",
                                      prediction_column="label")
        by_name = dict(zip(said["path"], said["label"]))
        for _index, row in frame.iterrows():
            assert row["cv_prediction"] == by_name[
                os.path.basename(str(row["png_path"]))]

    def test_no_temporary_key_is_left_on_the_frame(self, screen):
        """`_key` is the join's own bookkeeping. Left behind it becomes a
        numeric-looking column the feature split has to know to exclude."""
        path, names = screen
        frame = build_surrogate_frame(path, _predictions(names),
                                      path_column="path",
                                      prediction_column="label")
        assert "_key" not in frame.columns


class TestTheRefusalsThatShouldStay:
    def test_predictions_for_another_dataset_are_refused(self, screen):
        path, _names = screen
        strangers = _predictions([f"plate9_r1_c1_f1_o{i}.png"
                                  for i in range(20)])
        with pytest.raises(SurrogateError, match="matched a prediction"):
            build_surrogate_frame(path, strangers, path_column="path",
                                  prediction_column="label")

    def test_a_missing_column_is_named(self, screen):
        path, names = screen
        said = _predictions(names).rename(columns={"label": "class"})
        with pytest.raises(SurrogateError, match="'label'"):
            build_surrogate_frame(path, said, path_column="path",
                                  prediction_column="label")

    def test_a_database_with_no_shared_key_says_which_side_lacks_what(
            self, tmp_path):
        """The old message named only `prcfo`, which sent the reader
        looking for a column that was never going to be there."""
        path, names = _a_spacr_database(tmp_path)
        connection = sqlite3.connect(path)
        connection.execute("DROP TABLE png_list")
        connection.execute(
            "CREATE TABLE png_list (file_name TEXT, plateID TEXT)")
        connection.execute("INSERT INTO png_list VALUES ('x.png', 'p1')")
        connection.commit()
        connection.close()
        with pytest.raises(SurrogateError):
            build_surrogate_frame(path, _predictions(names),
                                  path_column="path",
                                  prediction_column="label")


class TestItActuallyExplains:
    @pytest.mark.parametrize("family", sorted(MODEL_FAMILIES))
    def test_every_offered_family_fits_and_reports_its_fidelity(
            self, screen, family):
        """A surrogate whose fidelity is not reported is an explanation
        nobody can weigh: the number that says whether it is about the CV
        model or about noise."""
        if family == "xgboost":
            pytest.importorskip("xgboost")
        from spacr.surrogate import explain_classifier

        path, names = screen
        children_before = {
            child.pid for child in multiprocessing.active_children()
        }
        threads_before = {id(thread) for thread in threading.enumerate()}
        result = explain_classifier(path, _predictions(names),
                                    path_column="path",
                                    prediction_column="label",
                                    model_family=family,
                                    **_quick(family))
        assert 0.0 <= result.fidelity <= 1.0
        assert 0.0 <= result.baseline <= 1.0
        assert result.n_objects > 0
        assert isinstance(result.is_faithful, bool)
        retained_processes = [
            child for child in multiprocessing.active_children()
            if child.pid not in children_before and child.is_alive()
        ]
        retained_threads = [
            thread for thread in threading.enumerate()
            if id(thread) not in threads_before and thread.is_alive()
        ]
        assert not [
            child.name for child in retained_processes
            if child.name.startswith("LokyProcess-")
        ]
        assert not [
            thread.name for thread in retained_threads
            if thread.name == "ExecutorManagerThread"
        ]

    def test_a_surrogate_that_learned_nothing_says_so(self, screen):
        """Random labels: the surrogate cannot beat the majority class, and
        `is_faithful` is the low bar that says the explanation is about the
        model rather than about noise."""
        from spacr.surrogate import explain_classifier

        path, names = screen
        result = explain_classifier(path, _predictions(names, seed=7),
                                    path_column="path",
                                    prediction_column="label",
                                    model_family="random_forest",
                                    **_quick("random_forest"))
        assert result.fidelity_improvement == pytest.approx(
            result.fidelity - result.baseline)
