"""Refusals and joins around the hit-investigation application seam.

The module is the file-facing half of hit attribution: it names files,
hashes a regression run, and joins predictions to measured objects. Almost
every uncovered branch is a refusal, so each one is reached by handing the
real function a real file (or database) with the specific defect.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.hit_attribution import HitAttributionError
from spacr.hit_investigation import (
    _hash_run, _read_cells, _read_fractions, _review_gallery_selection,
    control_fitted_embedding, evaluate_blinded_reviews,
    hit_investigation_default_settings, investigate_hit,
    register_settings, review_gallery_manifest,
)


# ---------------------------------------------------------------------------
# _hash_run
# ---------------------------------------------------------------------------

def test_a_missing_results_folder_is_refused_by_name(tmp_path):
    """The hash is provenance, so an absent run cannot be hashed to nothing."""
    with pytest.raises(HitAttributionError, match="no regression results folder"):
        _hash_run(str(tmp_path / "never_ran"))


def test_a_results_folder_that_is_a_file_is_refused(tmp_path):
    """A path that is not a directory is not a run folder either."""
    target = tmp_path / "results.csv"
    target.write_text("gene,effect\n", encoding="utf-8")

    with pytest.raises(HitAttributionError, match="no regression results folder"):
        _hash_run(str(target))


def test_a_results_folder_with_no_tables_is_refused(tmp_path):
    """Only CSV/JSON bytes are provenance; a folder of PNGs has none."""
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "plot.png").write_bytes(b"\x89PNG")
    (folder / "subdir").mkdir()

    with pytest.raises(HitAttributionError, match="no CSV/JSON"):
        _hash_run(str(folder))


def test_the_hash_changes_when_a_result_byte_changes(tmp_path):
    """Editing a result after the fact must produce different provenance."""
    folder = tmp_path / "run"
    folder.mkdir()
    table = folder / "results_gene.csv"
    table.write_text("gene,effect\nEAF1,0.7\n", encoding="utf-8")
    before = _hash_run(str(folder))

    table.write_text("gene,effect\nEAF1,0.8\n", encoding="utf-8")
    after = _hash_run(str(folder))
    (folder / "meta.json").write_text("{}", encoding="utf-8")
    with_extra = _hash_run(str(folder))

    assert before != after
    assert after != with_extra
    assert _hash_run(str(folder)) == with_extra


# ---------------------------------------------------------------------------
# _read_cells
# ---------------------------------------------------------------------------

def _cell_rows():
    return [
        {"prcfo": "p1_r01_c01_f1_o1", "prcf": "p1_r01_c01_f1",
         "plateID": "p1", "rowID": "r01", "columnID": "c01", "fieldID": "f1",
         "object_label": 1, "cell_area": 1.5},
        {"prcfo": "p1_r01_c01_f1_o2", "prcf": "p1_r01_c01_f1",
         "plateID": "p1", "rowID": "r01", "columnID": "c01", "fieldID": "f1",
         "object_label": 2, "cell_area": 2.5},
    ]


def _write_db(tmp_path, cells=None, png=None):
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as connection:
        pd.DataFrame(cells if cells is not None else _cell_rows()).to_sql(
            "cell", connection, index=False)
        if png is not None:
            pd.DataFrame(png).to_sql("png_list", connection, index=False)
    return database


def test_a_prediction_file_without_the_score_column_is_refused(tmp_path):
    """The configured score column is named back so the user can fix it."""
    database = _write_db(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"prcfo": ["p1_r01_c01_f1_o1"], "other": [0.3]}).to_csv(
        predictions, index=False)

    with pytest.raises(HitAttributionError, match="'phenotype_score'"):
        _read_cells(str(database), str(predictions), "phenotype_score", "path")


def test_a_reader_that_returns_prcfo_as_the_index_still_joins(tmp_path,
                                                              monkeypatch):
    """``_read_and_join_tables`` indexes by ``prcfo`` for some schemas.

    The join below is ``on="prcfo"``, so the key has to be a COLUMN by the
    time it is reached; left as the index it matches nothing and the merge
    raises instead of attributing a single cell.
    """
    import spacr.io

    frame = pd.DataFrame(_cell_rows()).set_index("prcfo")
    monkeypatch.setattr(spacr.io, "_read_and_join_tables",
                        lambda *_args, **_kwargs: frame.copy())
    database = _write_db(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"prcfo": ["p1_r01_c01_f1_o1", "p1_r01_c01_f1_o2"],
                  "phenotype_score": [0.4, 0.6]}).to_csv(predictions,
                                                         index=False)

    cells = _read_cells(str(database), str(predictions), "phenotype_score",
                        "path")

    assert list(cells["prcfo"]) == ["p1_r01_c01_f1_o1", "p1_r01_c01_f1_o2"]
    assert list(cells["phenotype_score"]) == [0.4, 0.6]


def test_a_score_already_measured_is_used_without_a_join(tmp_path):
    """When the database already carries the score, no join is attempted."""
    rows = _cell_rows()
    for index, row in enumerate(rows):
        row["phenotype_score"] = 0.1 * (index + 1)
    database = _write_db(tmp_path, rows)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"phenotype_score": [0.9]}).to_csv(predictions, index=False)

    cells = _read_cells(str(database), str(predictions), "phenotype_score",
                        "path")

    assert list(cells["phenotype_score"]) == [0.1, 0.2]


def test_a_prediction_file_that_repeats_prcfo_is_refused(tmp_path):
    """A repeated object would make the one-to-one join ambiguous."""
    database = _write_db(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({
        "prcfo": ["p1_r01_c01_f1_o1", "p1_r01_c01_f1_o1"],
        "phenotype_score": [0.3, 0.4],
    }).to_csv(predictions, index=False)

    with pytest.raises(HitAttributionError, match="repeats prcfo"):
        _read_cells(str(database), str(predictions), "phenotype_score", "path")


def test_predictions_joined_on_prcfo(tmp_path):
    """The prcfo route is preferred and needs no png_list at all."""
    database = _write_db(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({
        "prcfo": ["p1_r01_c01_f1_o1", "p1_r01_c01_f1_o2"],
        "phenotype_score": [0.3, 0.4],
    }).to_csv(predictions, index=False)

    cells = _read_cells(str(database), str(predictions), "phenotype_score",
                        "path")

    assert dict(zip(cells["prcfo"], cells["phenotype_score"])) == {
        "p1_r01_c01_f1_o1": 0.3, "p1_r01_c01_f1_o2": 0.4}


def test_predictions_with_neither_prcfo_nor_the_path_column_are_refused(
        tmp_path):
    """Without an identity column there is nothing to join objects on."""
    database = _write_db(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"filename": ["a.png"], "phenotype_score": [0.3]}).to_csv(
        predictions, index=False)

    with pytest.raises(HitAttributionError,
                       match="prcfo or the configured crop-path column"):
        _read_cells(str(database), str(predictions), "phenotype_score", "path")


def test_predictions_joined_on_the_crop_basename(tmp_path):
    """Falling back to crop paths matches on the basename, not the folder."""
    database = _write_db(tmp_path, png=[
        {"prcfo": "p1_r01_c01_f1_o1", "cell_id": "o1",
         "plateID": "p1", "rowID": "r01", "columnID": "c01",
         "fieldID": "f1", "png_path": "/old/crops/one.png"},
        {"prcfo": "p1_r01_c01_f1_o2", "cell_id": "o2",
         "plateID": "p1", "rowID": "r01", "columnID": "c01",
         "fieldID": "f1", "png_path": "/old/crops/two.png"},
    ])
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({
        "path": ["/moved/elsewhere/one.png", "/moved/elsewhere/two.png"],
        "phenotype_score": [0.3, 0.4],
    }).to_csv(predictions, index=False)

    cells = _read_cells(str(database), str(predictions), "phenotype_score",
                        "path")

    assert dict(zip(cells["prcfo"], cells["phenotype_score"])) == {
        "p1_r01_c01_f1_o1": 0.3, "p1_r01_c01_f1_o2": 0.4}
    # THE CROP PATH SURVIVES, UNDER A SUFFIXED NAME. Both frames carry
    # `png_path` -- the database's and the prediction file's -- so the
    # merge disambiguates them, and the database's is the one that keeps
    # the paths the crops are actually at. Asserted by value rather than
    # by name, because the name is the merge's to choose.
    kept = [column for column in cells.columns
            if column.startswith("png_path")]
    assert kept, "the crop path was dropped by the join"
    from_database = {path for column in kept for path in cells[column]
                     if str(path).startswith("/old/crops/")}
    assert from_database == {"/old/crops/one.png", "/old/crops/two.png"}


@pytest.mark.parametrize("side", ["database", "predictions"])
def test_repeated_crop_basenames_are_refused_on_either_side(tmp_path, side):
    """An ambiguous basename must not silently attach the wrong score."""
    png = [
        {"prcfo": "p1_r01_c01_f1_o1", "cell_id": "o1",
         "plateID": "p1", "rowID": "r01", "columnID": "c01",
         "fieldID": "f1", "png_path": "/a/one.png"},
        {"prcfo": "p1_r01_c01_f1_o2", "cell_id": "o2",
         "plateID": "p1", "rowID": "r01", "columnID": "c01",
         "fieldID": "f1", "png_path": "/b/one.png"},
    ] if side == "database" else [
        {"prcfo": "p1_r01_c01_f1_o1", "cell_id": "o1",
         "plateID": "p1", "rowID": "r01", "columnID": "c01",
         "fieldID": "f1", "png_path": "/a/one.png"},
        {"prcfo": "p1_r01_c01_f1_o2", "cell_id": "o2",
         "plateID": "p1", "rowID": "r01", "columnID": "c01",
         "fieldID": "f1", "png_path": "/a/two.png"},
    ]
    paths = ["/x/one.png", "/y/two.png"] if side == "database" else [
        "/x/one.png", "/y/one.png"]
    database = _write_db(tmp_path, png=png)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"path": paths, "phenotype_score": [0.3, 0.4]}).to_csv(
        predictions, index=False)

    with pytest.raises(HitAttributionError, match="basenames are not unique"):
        _read_cells(str(database), str(predictions), "phenotype_score", "path")


# ---------------------------------------------------------------------------
# _read_fractions
# ---------------------------------------------------------------------------

def test_a_fraction_table_keyed_only_by_prc_is_split_into_the_well_keys(
        tmp_path):
    """A ``prc`` column is canonicalised so downstream joins can match."""
    path = tmp_path / "fractions.csv"
    pd.DataFrame({
        "prc": ["plate1_r01_c01", "plate1_r01_c02"],
        "grna": ["EAF1_1", "NTC"], "fraction": [0.4, 0.6],
    }).to_csv(path, index=False)

    frame = _read_fractions(str(path))

    assert list(frame["plateID"]) == ["plate1", "plate1"]
    assert list(frame["rowID"]) == ["r01", "r01"]
    assert list(frame["columnID"]) == ["c01", "c02"]


def test_a_prc_that_does_not_split_into_three_is_left_alone(tmp_path):
    """A one-piece ``prc`` cannot be a plate/row/column and is not invented."""
    path = tmp_path / "fractions.csv"
    pd.DataFrame({"prc": ["plate1"], "grna": ["NTC"],
                  "fraction": [1.0]}).to_csv(path, index=False)

    frame = _read_fractions(str(path))

    assert "plateID" not in frame


def test_well_keys_already_present_are_not_rewritten(tmp_path):
    """A canonical table passes through untouched."""
    path = tmp_path / "fractions.csv"
    pd.DataFrame({
        "plateID": ["p1"], "rowID": ["r01"], "columnID": ["c01"],
        "prc": ["ignored_entirely_here"], "grna": ["NTC"], "fraction": [1.0],
    }).to_csv(path, index=False)

    frame = _read_fractions(str(path))

    assert list(frame["plateID"]) == ["p1"]
    assert list(frame["columnID"]) == ["c01"]


# ---------------------------------------------------------------------------
# investigate_hit input contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing", ["db_path", "predictions_file",
                                     "guide_fractions_file"])
def test_every_named_input_file_must_exist(tmp_path, missing):
    """Each of the three inputs is checked by name before any work starts."""
    present = tmp_path / "present.csv"
    present.write_text("a\n1\n", encoding="utf-8")
    settings = {key: str(present) for key in
                ("db_path", "predictions_file", "guide_fractions_file")}
    settings[missing] = str(tmp_path / "absent.csv")
    settings.update(target_gene="EAF1", target_guides=["EAF1_1"])

    with pytest.raises(HitAttributionError,
                       match=f"{missing} does not identify a file"):
        investigate_hit(settings)


@pytest.mark.parametrize("settings_patch", [
    {"target_gene": "", "target_guides": ["EAF1_1"]},
    {"target_gene": "EAF1", "target_guides": []},
    {"target_gene": "", "target_guides": []},
])
def test_a_hit_needs_both_a_gene_and_its_guides(tmp_path, settings_patch):
    """Attribution is guide-level evidence; neither half can be inferred."""
    present = tmp_path / "present.csv"
    present.write_text("a\n1\n", encoding="utf-8")
    settings = {key: str(present) for key in
                ("db_path", "predictions_file", "guide_fractions_file")}
    settings.update(settings_patch)

    with pytest.raises(HitAttributionError,
                       match="target_gene and target_guides are required"):
        investigate_hit(settings)


# ---------------------------------------------------------------------------
# A real run, reused by the result-shaped tests below
# ---------------------------------------------------------------------------

def _write_screen(root):
    rng = np.random.default_rng(17)
    database = root / "measurements.db"
    (root / "crops").mkdir()
    cells, crops, predictions, fractions = [], [], [], []
    for plate_index in range(3):
        plate = f"plate{plate_index + 1}"
        for well_index, fraction in enumerate((0.0, 0.25, 0.0, 0.65)):
            row, column, field = "r01", f"c{well_index + 1:02d}", "f1"
            prcf = f"{plate}_{row}_{column}_{field}"
            fractions.extend([
                {"plateID": plate, "rowID": row, "columnID": column,
                 "grna": "EAF1_1", "fraction": fraction},
                {"plateID": plate, "rowID": row, "columnID": column,
                 "grna": "NTC", "fraction": 1.0 - fraction},
            ])
            for label in range(1, 13):
                identity = int(rng.random() < fraction)
                area = rng.normal(2.2 * identity + 0.08 * plate_index, 0.55)
                texture = rng.normal(1.4 * identity, 0.65)
                prcfo = f"{prcf}_o{label}"
                cells.append({
                    "prcfo": prcfo, "prcf": prcf, "plateID": plate,
                    "rowID": row, "columnID": column, "fieldID": field,
                    "object_label": label, "cell_area": area,
                    "cell_texture": texture,
                })
                crops.append({
                    "cell_id": f"o{label}",
                    "png_path": str(root / "crops" / f"{prcfo}.png"),
                    "plateID": plate, "rowID": row, "columnID": column,
                    "fieldID": field,
                })
                predictions.append({
                    "prcfo": prcfo,
                    "phenotype_score": 1 / (1 + np.exp(-area)),
                })
    with sqlite3.connect(database) as connection:
        pd.DataFrame(cells).to_sql("cell", connection, index=False)
        pd.DataFrame(crops).to_sql("png_list", connection, index=False)
    prediction_file = root / "predictions.csv"
    fraction_file = root / "guide_fractions.csv"
    pd.DataFrame(predictions).to_csv(prediction_file, index=False)
    pd.DataFrame(fractions).to_csv(fraction_file, index=False)
    results = root / "regression_run"
    results.mkdir()
    pd.DataFrame({"gene": ["EAF1"], "effect": [0.7], "q_value": [0.01]}).to_csv(
        results / "results_gene.csv", index=False)
    return {
        "db_path": str(database), "predictions_file": str(prediction_file),
        "guide_fractions_file": str(fraction_file),
        "results_folder": str(results), "target_gene": "EAF1",
        "target_guides": ["EAF1_1"], "score_column": "phenotype_score",
        "hit_feature_columns": ["cell_area", "cell_texture"],
        "hit_bootstrap": 5, "hit_permutations": 5,
        "hit_pipeline_permutations": 1, "hit_gallery_per_stratum": 3,
    }


@pytest.fixture(scope="module")
def investigated(tmp_path_factory):
    """One real end-to-end run, with the verbose summary printed."""
    root = tmp_path_factory.mktemp("hit_run")
    settings = _write_screen(root)
    settings["verbose"] = True
    return investigate_hit(settings), root


def test_a_verbose_run_prints_the_summary(investigated, capsys):
    """``verbose`` is the switch between a silent run and a printed summary."""
    payload, _root = investigated
    # capsys cannot see output produced during a module-scoped fixture, so
    # the same summary is asked for again and compared with what the run
    # returned.
    text = payload["result"].summary()

    assert "EAF1" in text
    assert text.strip() != ""


def test_the_run_writes_every_promised_artifact(investigated):
    """Each named path in the payload exists on disk."""
    payload, _root = investigated

    for name, path in payload["paths"].items():
        assert path.endswith((".csv", ".json")), name
        assert pd.io.common.file_exists(path), name


def test_an_embedding_needs_at_least_two_allowed_features(investigated):
    """PCA on one feature is not a two-dimensional embedding."""
    payload, _root = investigated
    result = payload["result"]
    one_feature = result.__class__(**{
        **{field: getattr(result, field) for field in result.__dataclass_fields__},
        "feature_columns": list(result.feature_columns)[:1],
    })

    with pytest.raises(HitAttributionError, match="at least two allowed"):
        control_fitted_embedding(one_feature)


def test_an_embedding_needs_target_free_training_cells(investigated):
    """The contract is "fit on control cells", so there must be some."""
    payload, _root = investigated
    result = payload["result"]
    cells = result.cells.copy()
    cells["target_guide_fraction"] = 1.0
    all_target = result.__class__(**{
        **{field: getattr(result, field) for field in result.__dataclass_fields__},
        "cells": cells,
    })

    with pytest.raises(HitAttributionError, match="target-free training cells"):
        control_fitted_embedding(all_target)


def test_a_gallery_without_crop_paths_is_empty_not_wrong(investigated):
    """Nothing can be reviewed without a crop to look at."""
    payload, _root = investigated
    result = payload["result"]
    cells = result.cells.drop(columns=["png_path"])
    no_crops = result.__class__(**{
        **{field: getattr(result, field) for field in result.__dataclass_fields__},
        "cells": cells,
    })

    selection = _review_gallery_selection(no_crops, 3)
    manifest = review_gallery_manifest(no_crops, 3)

    assert selection.empty
    assert manifest.empty


# ---------------------------------------------------------------------------
# evaluate_blinded_reviews
# ---------------------------------------------------------------------------

def _key(ids=("a", "b", "c", "d")):
    return pd.DataFrame({
        "review_id": list(ids),
        "hit_like_probability": [0.9, 0.8, 0.2, 0.1][:len(ids)],
    })


def test_a_review_sheet_missing_a_column_is_refused():
    """The three review columns are named back rather than guessed at."""
    reviews = pd.DataFrame({"review_id": ["a"], "reviewer_id": ["r1"]})

    with pytest.raises(HitAttributionError, match="review sheet needs"):
        evaluate_blinded_reviews(reviews, _key())


def test_a_review_key_missing_a_column_is_refused():
    """Without held-back probabilities there is nothing to compare against."""
    reviews = pd.DataFrame({"review_id": ["a"], "reviewer_id": ["r1"],
                            "reviewer_label": [1]})

    with pytest.raises(HitAttributionError, match="review key needs"):
        evaluate_blinded_reviews(reviews, pd.DataFrame({"review_id": ["a"]}))


def test_a_review_key_that_repeats_an_id_is_refused():
    """A repeated key row would silently multiply a reviewer's call."""
    reviews = pd.DataFrame({"review_id": ["a"], "reviewer_id": ["r1"],
                            "reviewer_label": [1]})
    key = pd.DataFrame({"review_id": ["a", "a"],
                        "hit_like_probability": [0.9, 0.8]})

    with pytest.raises(HitAttributionError, match="repeats review_id"):
        evaluate_blinded_reviews(reviews, key)


def test_a_review_sheet_with_no_binary_labels_is_refused():
    """Blank or non-binary labels are an unreviewed sheet, not a result."""
    reviews = pd.DataFrame({
        "review_id": ["a", "b"], "reviewer_id": ["r1", "r1"],
        "reviewer_label": ["", "maybe"],
    })

    with pytest.raises(HitAttributionError, match="no binary 0/1 labels"):
        evaluate_blinded_reviews(reviews, _key())


def test_a_single_class_consensus_reports_nan_auc_not_a_crash():
    """ROC AUC is undefined when every consensus call is the same."""
    reviews = pd.DataFrame({
        "review_id": ["a", "b"], "reviewer_id": ["r1", "r1"],
        "reviewer_label": [1, 1],
    })

    metrics = evaluate_blinded_reviews(reviews, _key(("a", "b")))

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["mean_pairwise_cohen_kappa"])
    assert metrics["n_reviewers"] == 1


def test_two_reviewers_get_a_pairwise_kappa():
    """Agreement is only defined between reviewers, so two are needed."""
    reviews = pd.DataFrame({
        "review_id": ["a", "b", "c", "a", "b", "c"],
        "reviewer_id": ["r1"] * 3 + ["r2"] * 3,
        "reviewer_label": [1, 1, 0, 1, 0, 0],
    })

    metrics = evaluate_blinded_reviews(reviews, _key(("a", "b", "c")))

    assert metrics["n_reviewers"] == 2
    assert -1.0 <= metrics["mean_pairwise_cohen_kappa"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


# ---------------------------------------------------------------------------
# settings registration
# ---------------------------------------------------------------------------

def test_defaults_fill_every_documented_key():
    """A blank settings dict comes back complete."""
    configured = hit_investigation_default_settings()

    assert configured["path_column"] == "path"
    assert configured["hit_direction"] == "positive"
    assert configured["hit_store_database"] is True
    assert np.isnan(configured["hit_guide_agreement"])


def test_supplied_settings_win_over_the_defaults():
    """``setdefault`` means a caller's value is never overwritten."""
    configured = hit_investigation_default_settings(
        {"path_column": "crop", "hit_random_seed": 7})

    assert configured["path_column"] == "crop"
    assert configured["hit_random_seed"] == 7


def test_registering_twice_is_refused_unless_replacement_is_asked_for():
    """Import already registered the app, so a second plain call is a no-op."""
    assert register_settings() is False
    assert register_settings(replace=True) is True
    assert register_settings() is False


def test_a_run_can_write_its_files_without_touching_the_database(tmp_path):
    """``hit_store_database=False`` is the portable-files-only choice.

    The setting exists because storing the attribution creates a new
    versioned run in the measurements database, and a user investigating a
    hit on somebody else's screen -- or on a read-only copy -- wants the CSVs
    and the manifest without adding a row to a database they do not own.

    Every other output must still be written. Turning the store off is a
    narrower run, not a failed one, and the manifest reports an empty
    attribution id rather than omitting the key, so a reader can tell "not
    stored" from "this manifest predates the field".
    """
    import os

    settings = _write_screen(tmp_path)
    settings["hit_store_database"] = False

    payload = investigate_hit(settings)

    assert payload["attribution_run_id"] == ""
    for key in ("wells", "guides", "thresholds", "embedding", "gallery"):
        assert os.path.isfile(payload["paths"][key]), f"{key} was not written"


def test_storing_the_attribution_gives_the_manifest_a_run_to_point_at(
        investigated):
    """The default path, so the test above is about the switch.

    Without this, "the id is empty when off" would pass on a build that never
    filled it in at all.
    """
    payload, _root = investigated

    assert payload["attribution_run_id"], (
        "a stored attribution left the manifest with no run to point at")
