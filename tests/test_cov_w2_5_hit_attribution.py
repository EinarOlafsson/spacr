"""Attribution refuses a design it cannot identify rather than answering.

Every refusal in this module protects the same claim: that a per-cell number
came from a design where target and control wells were independent. So the
tests below take the design apart one piece at a time — one class of well,
duplicate object keys, a feature that leaks the bag label — and assert on the
message, because the message is what tells the user which piece is missing.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import hit_attribution as ha
from spacr.hit_attribution import (
    HitAttributionError, InsufficientDesignError, build_hit_cell_frame)


def _inputs(seed=3, target_wells=8, control_wells=8, cells_per_well=12):
    """A four-plate screen with a real morphology shift in the target wells."""
    rng = np.random.default_rng(seed)
    cells, fractions = [], []
    for target, count, offset in ((False, control_wells, 0),
                                  (True, target_wells, 100)):
        for well_index in range(count):
            plate = f"p{1 + well_index % 4}"
            row = f"r{1 + (offset + well_index) // 12}"
            column = f"c{1 + (offset + well_index) % 12}"
            fraction = 0.0 if not target else 0.08 + 0.02 * well_index
            fractions.extend([
                {"plateID": plate, "rowID": row, "columnID": column,
                 "grna": "123_1", "fraction": fraction},
                {"plateID": plate, "rowID": row, "columnID": column,
                 "grna": "other_1", "fraction": 1.0 - fraction},
            ])
            for cell_index in range(cells_per_well):
                cells.append({
                    "prcfo": f"{plate}_{row}_{column}_f1_o{cell_index}",
                    "plateID": plate, "rowID": row, "columnID": column,
                    "fieldID": "f1", "object_label": cell_index,
                    "XGBoost_score": rng.normal(1.5 if target else 0.0, 0.7),
                    "cell_area": rng.normal(2.0 if target else 0.0, 0.6),
                    "cell_texture": rng.normal(1.0 if target else 0.0, 0.7),
                })
    return pd.DataFrame(cells), pd.DataFrame(fractions)


@pytest.fixture
def cells_and_fractions():
    return _inputs()


@pytest.fixture
def hit_frame(cells_and_fractions):
    """The joined review frame the rest of the module consumes."""
    cells, fractions = cells_and_fractions
    return build_hit_cell_frame(cells, fractions, target_guides=["123_1"],
                                score_column="XGBoost_score")


@pytest.fixture
def candidate_frame(hit_frame):
    """The same frame with the bag label the cross-fitter wants."""
    frame = hit_frame.copy()
    frame["target_well"] = frame["target_guide_fraction"] > 0
    return frame


# ---------------------------------------------------------------------------
# building the review frame
# ---------------------------------------------------------------------------

def test_no_target_guide_is_nothing_to_attribute(cells_and_fractions):
    """An empty guide list is a refusal, not an attribution of everything."""
    cells, fractions = cells_and_fractions

    with pytest.raises(HitAttributionError, match="at least one target guide"):
        build_hit_cell_frame(cells, fractions, target_guides=[],
                             score_column="XGBoost_score")


def test_cells_and_fractions_that_share_no_well_key_are_refused(
        cells_and_fractions):
    """Without a shared key the join would silently produce nothing."""
    cells, fractions = cells_and_fractions

    with pytest.raises(HitAttributionError, match="share no well key"):
        build_hit_cell_frame(cells, fractions.rename(
            columns={"plateID": "plate", "rowID": "row",
                     "columnID": "col"}),
            target_guides=["123_1"], score_column="XGBoost_score")


def test_half_a_well_key_is_refused(cells_and_fractions):
    """A row without its column names a whole row of the plate, not a well."""
    cells, fractions = cells_and_fractions

    with pytest.raises(HitAttributionError, match="partial key"):
        build_hit_cell_frame(cells, fractions.drop(columns=["columnID"]),
                             target_guides=["123_1"],
                             score_column="XGBoost_score")


def test_cells_with_no_object_key_at_all_are_refused(cells_and_fractions):
    """Attribution needs to name the cell it attributes."""
    cells, fractions = cells_and_fractions
    stripped = cells.drop(columns=["prcfo"])

    with pytest.raises(HitAttributionError, match="no stable object key"):
        build_hit_cell_frame(stripped, fractions, target_guides=["123_1"],
                             score_column="XGBoost_score",
                             object_columns=["not_a_column"])


def test_duplicate_object_keys_are_refused_with_examples(cells_and_fractions):
    """Attribution would overwrite or multiply the duplicated cells."""
    cells, fractions = cells_and_fractions
    doubled = pd.concat([cells, cells.iloc[[0]]], ignore_index=True)

    with pytest.raises(HitAttributionError) as caught:
        build_hit_cell_frame(doubled, fractions, target_guides=["123_1"],
                             score_column="XGBoost_score")

    assert "not unique" in str(caught.value)
    assert "prcfo" in str(caught.value)


def test_a_non_numeric_guide_fraction_is_refused(cells_and_fractions):
    """A fraction that is not a finite number is not a fraction."""
    cells, fractions = cells_and_fractions
    fractions["fraction"] = fractions["fraction"].astype(object)
    fractions.loc[0, "fraction"] = "most of it"

    with pytest.raises(HitAttributionError, match="finite numbers"):
        build_hit_cell_frame(cells, fractions, target_guides=["123_1"],
                             score_column="XGBoost_score")


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_a_fraction_outside_zero_to_one_is_refused(bad, cells_and_fractions):
    """A proportion outside the unit interval is a units error."""
    cells, fractions = cells_and_fractions
    fractions.loc[0, "fraction"] = bad

    with pytest.raises(HitAttributionError, match="between 0 and 1"):
        build_hit_cell_frame(cells, fractions, target_guides=["123_1"],
                             score_column="XGBoost_score")


def test_a_guide_that_is_not_in_the_screen_is_named(cells_and_fractions):
    """Attributing to a guide nobody sequenced is a typo worth catching."""
    cells, fractions = cells_and_fractions

    with pytest.raises(HitAttributionError) as caught:
        build_hit_cell_frame(cells, fractions, target_guides=["not_a_guide"],
                             score_column="XGBoost_score")

    assert "not_a_guide" in str(caught.value)


def test_target_fractions_that_sum_above_one_are_refused(
        cells_and_fractions):
    """Two guides cannot together be more than all of a well."""
    cells, fractions = cells_and_fractions
    extra = fractions[fractions["grna"] == "123_1"].copy()
    extra["grna"] = "123_2"
    extra["fraction"] = 0.95
    both = pd.concat([fractions, extra], ignore_index=True)

    with pytest.raises(HitAttributionError, match="sum above 1"):
        build_hit_cell_frame(cells, both, target_guides=["123_1", "123_2"],
                             score_column="XGBoost_score")


def test_a_missing_score_is_refused_by_name(cells_and_fractions):
    """A cell with no score cannot be ranked within its well."""
    cells, fractions = cells_and_fractions
    cells.loc[0, "XGBoost_score"] = np.nan

    with pytest.raises(HitAttributionError, match="XGBoost_score"):
        build_hit_cell_frame(cells, fractions, target_guides=["123_1"],
                             score_column="XGBoost_score")


def test_a_direction_that_is_neither_way_is_refused(cells_and_fractions):
    """"sideways" is not a hit direction."""
    cells, fractions = cells_and_fractions

    with pytest.raises(HitAttributionError, match="positive"):
        build_hit_cell_frame(cells, fractions, target_guides=["123_1"],
                             score_column="XGBoost_score",
                             direction="sideways")


def test_a_negative_hit_ranks_the_lowest_score_highest(cells_and_fractions):
    """High is always more hit-like, whichever tail the hit is in."""
    cells, fractions = cells_and_fractions

    frame = build_hit_cell_frame(cells, fractions, target_guides=["123_1"],
                                 score_column="XGBoost_score",
                                 direction="negative")

    one_well = frame[(frame["plateID"] == "p1") & (frame["rowID"] == "r1")]
    lowest = one_well["XGBoost_score"].idxmin()
    assert one_well.loc[lowest, "candidate_percentile"] == \
        one_well["candidate_percentile"].max()


# ---------------------------------------------------------------------------
# the non-parametric cross-fitter
# ---------------------------------------------------------------------------

def test_cross_fitting_scores_every_cell_and_records_its_fold(
        candidate_frame):
    """Each cell is predicted by a model that never saw its group."""
    scored, features, level, warnings = ha.crossfit_candidate_probabilities(
        candidate_frame, random_seed=1, n_splits=4)

    assert len(scored) == len(candidate_frame)
    assert scored["candidate_probability"].between(0, 1).all()
    assert scored["candidate_uncertainty"].between(0, 1).all()
    assert set(scored["attribution_fold"]) == {0, 1, 2, 3}
    assert level == "plate"
    assert "cell_area" in features and "cell_texture" in features
    assert "XGBoost_score" not in features
    assert warnings == []


def test_the_call_follows_the_threshold(candidate_frame):
    """A threshold nobody can reach calls nothing."""
    scored, _f, _l, _w = ha.crossfit_candidate_probabilities(
        candidate_frame, threshold=1.01, random_seed=1, n_splits=4)

    assert not scored["candidate_call"].any()


def test_a_well_that_disagrees_with_itself_is_refused(candidate_frame):
    """One well is one bag; two labels for it is a data error."""
    frame = candidate_frame.copy()
    first = frame.index[0]
    frame.loc[first, "target_well"] = not frame.loc[first, "target_well"]

    with pytest.raises(HitAttributionError, match="disagrees within a well"):
        ha.crossfit_candidate_probabilities(frame)


def test_too_few_independent_wells_is_an_insufficient_design(
        cells_and_fractions):
    """Four of each is the floor, and the message says what there is."""
    cells, fractions = _inputs(target_wells=2, control_wells=2)
    frame = build_hit_cell_frame(cells, fractions, target_guides=["123_1"],
                                 score_column="XGBoost_score")
    frame["target_well"] = frame["target_guide_fraction"] > 0

    with pytest.raises(InsufficientDesignError) as caught:
        ha.crossfit_candidate_probabilities(frame)

    assert "have 2 and 2" in str(caught.value)


def test_a_feature_that_names_the_object_is_refused(candidate_frame):
    """A key column as a feature predicts the bag, not the phenotype."""
    with pytest.raises(HitAttributionError, match="leak identifiers"):
        ha.crossfit_candidate_probabilities(
            candidate_frame, feature_columns=["cell_area", "object_label"])


def test_a_feature_column_that_is_entirely_missing_is_named(candidate_frame):
    """An all-NaN feature is not a feature, and the message says which."""
    frame = candidate_frame.copy()
    frame["hollow"] = np.nan

    with pytest.raises(HitAttributionError) as caught:
        ha.crossfit_candidate_probabilities(
            frame, feature_columns=["cell_area", "hollow"])

    assert "entirely missing" in str(caught.value)
    assert "hollow" in str(caught.value)


def test_a_single_plate_screen_cross_fits_by_well(candidate_frame):
    """Without four plates the independent unit is the well."""
    frame = candidate_frame.copy()
    frame["plateID"] = "p1"

    scored, _f, level, _w = ha.crossfit_candidate_probabilities(
        frame, random_seed=2, n_splits=4)

    assert level == "well"
    assert scored["candidate_probability"].notna().all()


def test_plate_splitting_can_be_declined(candidate_frame):
    """``prefer_plate=False`` cross-fits by well even with four plates."""
    _s, _f, level, _w = ha.crossfit_candidate_probabilities(
        candidate_frame, prefer_plate=False, random_seed=2, n_splits=3)

    assert level == "well"


def test_a_fold_with_only_one_bag_class_says_what_to_do(candidate_frame):
    """A split that separates the classes is unidentified, and it says so."""
    frame = candidate_frame.copy()
    # Every control well on one plate and the targets spread over three: the
    # fold that holds the control plate out trains on targets alone.
    plates = np.where(frame["target_well"],
                      "pT" + (frame["columnID"].str[1:].astype(int) % 3
                              ).astype(str),
                      "pC")
    frame["plateID"] = plates

    with pytest.raises(InsufficientDesignError) as caught:
        ha.crossfit_candidate_probabilities(frame, n_splits=4)

    assert "one bag class" in str(caught.value)


# ---------------------------------------------------------------------------
# the well-level summary
# ---------------------------------------------------------------------------

def test_candidate_enrichment_is_summarised_per_well(candidate_frame):
    """The experimental unit is the well, and the summary says so."""
    scored, _f, _l, _w = ha.crossfit_candidate_probabilities(
        candidate_frame, random_seed=1, n_splits=4)

    wells, summary = ha.quantify_candidate_enrichment(
        scored, bootstrap_iterations=80, permutation_iterations=80)

    assert len(wells) == 16
    assert set(wells.columns) >= {"plateID", "rowID", "columnID",
                                  "target_well", "candidate_prevalence",
                                  "mean_candidate_probability", "n_cells"}
    assert wells["n_cells"].sum() == len(scored)
    assert summary["target_wells"] == 8
    assert summary["control_wells"] == 8
    assert "plate_blocked_permutation_p_value" in summary
    assert "n_target_wells" not in summary


def test_a_scored_frame_with_no_calls_gets_them_at_the_usual_threshold(
        candidate_frame):
    """A frame carrying only probabilities is summarised at 0.5."""
    scored, _f, _l, _w = ha.crossfit_candidate_probabilities(
        candidate_frame, random_seed=1, n_splits=4)
    without = scored.drop(columns=["candidate_call"])

    wells, _summary = ha.quantify_candidate_enrichment(
        without, bootstrap_iterations=50, permutation_iterations=50)

    expected = (scored["candidate_probability"] >= 0.5).mean()
    assert wells["candidate_prevalence"].mean() == pytest.approx(
        expected, abs=0.05)
    assert "candidate_call" not in without.columns


# ---------------------------------------------------------------------------
# the hierarchical mixture
# ---------------------------------------------------------------------------

def test_a_threshold_outside_the_unit_interval_is_refused(hit_frame):
    """A probability threshold of 0 or 1 calls everything or nothing."""
    with pytest.raises(HitAttributionError, match="strictly between"):
        ha.fit_hit_attribution(hit_frame, target_gene="123", threshold=1.0)


def test_a_frame_with_no_morphology_left_says_what_to_do(hit_frame):
    """Excluding every feature is a refusal naming the two ways out."""
    thin = hit_frame.drop(columns=["cell_area", "cell_texture",
                                   "candidate_rank", "candidate_percentile"])

    with pytest.raises(HitAttributionError) as caught:
        ha.fit_hit_attribution(thin, target_gene="123")

    assert "no independent numeric morphology features" in str(caught.value)
    assert "feature_columns" in str(caught.value)


def test_the_original_score_can_be_asked_for_explicitly(hit_frame):
    """Including it is a choice the caller makes, and it is honoured."""
    result = ha.fit_hit_attribution(
        hit_frame, target_gene="123", include_original_score=True,
        split_by="well", random_seed=4, n_bootstrap=40, n_permutations=40)

    assert "XGBoost_score" in result.feature_columns
    assert result.warnings


def test_a_feature_that_leaks_the_bag_label_is_refused(hit_frame):
    """A guide fraction as a feature is the bag label under another name."""
    with pytest.raises(HitAttributionError, match="leak the bag label"):
        ha.fit_hit_attribution(
            hit_frame, target_gene="123",
            feature_columns=["cell_area", "target_guide_fraction"])


def test_attribution_needs_both_kinds_of_well(cells_and_fractions):
    """With no control wells there is nothing to contrast against."""
    cells, fractions = _inputs(control_wells=0, target_wells=16)
    frame = build_hit_cell_frame(cells, fractions, target_guides=["123_1"],
                                 score_column="XGBoost_score")

    with pytest.raises(HitAttributionError,
                       match="target-free and target-containing"):
        ha.fit_hit_attribution(frame, target_gene="123")


def test_an_unknown_split_level_is_refused(hit_frame):
    """Only auto, plate and well name an independent unit."""
    with pytest.raises(HitAttributionError, match="auto, plate or well"):
        ha.fit_hit_attribution(hit_frame, target_gene="123",
                               split_by="field")


def test_plate_splitting_needs_three_plates(hit_frame):
    """Two plates cannot identify a plate-level fold."""
    frame = hit_frame.copy()
    frame["plateID"] = np.where(frame["plateID"] == "p1", "p1", "p2")

    with pytest.raises(HitAttributionError, match="at least three plates"):
        ha.fit_hit_attribution(frame, target_gene="123", split_by="plate")


def test_cross_fitting_needs_three_independent_groups():
    """Two wells is not a cross-fit, whatever it is called."""
    values = np.random.default_rng(0).normal(size=(20, 2))
    fractions = np.repeat([0.0, 0.2], 10)
    groups = pd.Series(np.repeat(["w1", "w2"], 10))

    with pytest.raises(HitAttributionError,
                       match="at least three independent groups"):
        ha._crossfit_mixture(values, fractions, groups)


def test_a_fold_with_one_kind_of_well_is_refused():
    """The mixture cannot be identified from target wells alone."""
    values = np.random.default_rng(0).normal(size=(20, 2))

    with pytest.raises(HitAttributionError, match="each training fold"):
        ha._fit_mixture(values, np.full(20, 0.2))

    with pytest.raises(HitAttributionError, match="each training fold"):
        ha._fit_mixture(values, np.zeros(20))


def test_a_flat_contrast_still_fits_a_mixture():
    """Identical target and control wells leave a usable projection axis."""
    values = np.tile(np.arange(4, dtype=float), (24, 1))
    fractions = np.tile([0.0, 0.3], 12)

    mixture = ha._fit_mixture(values, fractions, max_iter=5)

    assert np.isfinite(mixture.mu0).all() and np.isfinite(mixture.mu1).all()
    assert mixture.iterations >= 1
    assert np.isfinite(mixture.predict(values, fractions)).all()


def test_a_guide_with_no_column_of_its_own_is_skipped(hit_frame):
    """A guide absent from the widened table contributes no evidence row."""
    frame = hit_frame.copy()
    frame.attrs = dict(hit_frame.attrs)
    frame.attrs["target_guides"] = ["123_1", "never_sequenced"]

    result = ha.fit_hit_attribution(
        frame, target_gene="123", split_by="well", random_seed=4,
        n_bootstrap=40, n_permutations=40)

    assert list(result.guide_evidence["guide"]) == ["123_1"]


def test_enrichment_needs_two_wells_of_each_kind():
    """A difference between one well and one well is not an estimate."""
    wells = pd.DataFrame({
        "plateID": ["p1", "p1"], "rowID": ["r1", "r2"],
        "columnID": ["c1", "c1"],
        "target_guide_fraction": [0.0, 0.2],
        "hit_like_prevalence": [0.1, 0.6],
        "mean_hit_like_probability": [0.2, 0.7],
    })

    with pytest.raises(HitAttributionError):
        ha.quantify_hit_enrichment(wells, n_bootstrap=10, n_permutations=10)


# ---------------------------------------------------------------------------
# persisting and promoting
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted(hit_frame):
    return ha.fit_hit_attribution(
        hit_frame, target_gene="123", split_by="well", random_seed=4,
        n_bootstrap=40, n_permutations=40)


@pytest.fixture
def png_db(tmp_path, hit_frame):
    """A database with a png_list carrying every attributed object key."""
    path = str(tmp_path / "measurements.db")
    connection = sqlite3.connect(path)
    try:
        hit_frame[["prcfo", "plateID", "rowID", "columnID"]].to_sql(
            "png_list", connection, index=False)
    finally:
        connection.close()
    return path


def test_writing_to_a_database_that_is_not_there_says_where(tmp_path, fitted):
    """The path is named, so a typo is visible."""
    with pytest.raises(HitAttributionError, match="no database at"):
        ha.write_hit_attribution(str(tmp_path / "absent.db"), fitted)


def test_an_annotation_column_that_is_not_a_name_is_refused(png_db, fitted):
    """A column name is a name, not a fragment of SQL."""
    with pytest.raises(HitAttributionError, match="letters, numbers"):
        ha.promote_hit_calls(png_db, fitted, run_id="r",
                             annotation_column="drop table png_list; --")


def test_promotion_requires_the_exact_object_key(png_db, fitted):
    """Promotion writes into png_list, so the key has to be png_list's."""
    import dataclasses

    other = dataclasses.replace(fitted, object_columns=["plateID", "rowID"])

    with pytest.raises(HitAttributionError, match="requires prcfo"):
        ha.promote_hit_calls(png_db, other, run_id="r",
                             annotation_column="hit_call")


def test_a_png_list_without_prcfo_cannot_be_promoted_into(tmp_path, fitted):
    """Without the key there is nothing to match the attributed cells to."""
    path = str(tmp_path / "no_key.db")
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE png_list (png_path TEXT)")
    finally:
        connection.close()

    with pytest.raises(HitAttributionError, match="no prcfo object key"):
        ha.promote_hit_calls(path, fitted, run_id="r",
                             annotation_column="hit_call")


def test_promotion_refuses_to_reuse_an_existing_column(png_db, fitted):
    """Hand annotations must not be overwritten by a machine call."""
    connection = sqlite3.connect(png_db)
    try:
        connection.execute('ALTER TABLE png_list ADD COLUMN "hand_call"')
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(HitAttributionError, match="already exists"):
        ha.promote_hit_calls(png_db, fitted, run_id="r",
                             annotation_column="hand_call")


def test_a_promotion_is_written_and_can_be_undone(png_db, fitted):
    """The undo clears exactly what the promotion wrote."""
    run = ha.write_hit_attribution(png_db, fitted)
    promotion = ha.promote_hit_calls(png_db, fitted, run_id=run,
                                     annotation_column="machine_call")
    expected = int(fitted.cells["hit_like_call"].sum())

    connection = sqlite3.connect(png_db)
    try:
        written = connection.execute(
            'SELECT COUNT(*) FROM png_list WHERE "machine_call" IS NOT NULL'
        ).fetchone()[0]
    finally:
        connection.close()
    assert written == expected

    assert ha.undo_hit_promotion(png_db, promotion) == expected
    assert ha.undo_hit_promotion(png_db, promotion) == 0


def test_undoing_a_promotion_that_never_happened_is_zero(png_db, fitted):
    """A stray undo is a no-op, not an error."""
    ha.promote_hit_calls(png_db, fitted, run_id="r",
                         annotation_column="machine_call")

    assert ha.undo_hit_promotion(png_db, "not-a-promotion") == 0


def test_an_audit_naming_two_columns_is_refused(png_db, fitted):
    """One promotion wrote one column; two means the ledger was edited."""
    promotion = ha.promote_hit_calls(png_db, fitted, run_id="r",
                                     annotation_column="machine_call")
    connection = sqlite3.connect(png_db)
    try:
        connection.execute(
            "UPDATE hit_attribution_promotions SET annotation_column='other' "
            "WHERE rowid = (SELECT MIN(rowid) FROM hit_attribution_promotions)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(HitAttributionError, match="multiple columns"):
        ha.undo_hit_promotion(png_db, promotion)


def test_an_unsafe_stored_column_name_is_refused(png_db, fitted):
    """The stored name is re-validated before it is put back into SQL."""
    promotion = ha.promote_hit_calls(png_db, fitted, run_id="r",
                                     annotation_column="machine_call")
    connection = sqlite3.connect(png_db)
    try:
        connection.execute(
            "UPDATE hit_attribution_promotions SET annotation_column='a b'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(HitAttributionError, match="unsafe"):
        ha.undo_hit_promotion(png_db, promotion)


# ---------------------------------------------------------------------------
# the investigation bundle
# ---------------------------------------------------------------------------

@pytest.fixture
def investigation(candidate_frame):
    scored, features, level, warnings = ha.crossfit_candidate_probabilities(
        candidate_frame, random_seed=1, n_splits=4)
    wells, summary = ha.quantify_candidate_enrichment(
        scored, bootstrap_iterations=40, permutation_iterations=40)
    context = ha.HitRunContext(
        regression_results_folder="/somewhere", regression_run_sha256="abc",
        gene="123", phenotype="fraction", effect=0.4, guides=("123_1",))
    return ha.HitInvestigationResult(
        attribution_run_id="run-1", context=context, cells=scored,
        wells=wells, enrichment=summary, feature_columns=features,
        split_level=level, warnings=warnings)


def test_promoting_a_run_with_no_stored_cells_names_the_run(png_db,
                                                            investigation):
    """A run id nobody stored is a typo, and the message quotes it."""
    ha.store_attribution(png_db, investigation)

    with pytest.raises(HitAttributionError) as caught:
        ha.promote_calls(png_db, "never-stored", "candidate_call_column")

    assert "never-stored" in str(caught.value)


def test_an_attributed_object_missing_from_png_list_is_refused(tmp_path,
                                                               investigation):
    """Promoting a cell the crop table does not have would write nothing."""
    path = str(tmp_path / "short.db")
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE png_list (prcfo TEXT)")
        connection.execute("INSERT INTO png_list VALUES ('not_a_real_key')")
        connection.commit()
    finally:
        connection.close()
    ha.store_attribution(path, investigation)

    with pytest.raises(HitAttributionError, match="absent from png_list"):
        ha.promote_calls(path, "run-1", "candidate_call_column")


def test_reverting_a_promotion_restores_the_previous_values(png_db,
                                                            investigation):
    """Exactly what was replaced comes back, and only once."""
    connection = sqlite3.connect(png_db)
    try:
        connection.execute('ALTER TABLE png_list ADD COLUMN "call_column"')
        connection.execute('UPDATE png_list SET "call_column" = 7')
        connection.commit()
    finally:
        connection.close()
    ha.store_attribution(png_db, investigation)

    promotion = ha.promote_calls(png_db, "run-1", "call_column")
    restored = ha.revert_promotion(png_db, promotion)

    connection = sqlite3.connect(png_db)
    try:
        values = {row[0] for row in connection.execute(
            'SELECT DISTINCT "call_column" FROM png_list')}
    finally:
        connection.close()
    assert restored > 0
    assert values == {7}
    assert ha.revert_promotion(png_db, promotion) == 0


def test_reverting_on_a_database_that_never_promoted_is_zero(png_db):
    """An undo with nothing to undo must not look like a corrupt database."""
    assert ha.revert_promotion(png_db, "anything") == 0


def test_a_reverted_audit_naming_two_columns_is_refused(png_db,
                                                        investigation):
    """The revert re-checks the ledger before writing through it."""
    ha.store_attribution(png_db, investigation)
    promotion = ha.promote_calls(png_db, "run-1", "call_column")
    connection = sqlite3.connect(png_db)
    try:
        connection.execute(
            "UPDATE hit_promotion_audit SET annotation_column='other' "
            "WHERE rowid = (SELECT MIN(rowid) FROM hit_promotion_audit)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(HitAttributionError, match="multiple columns"):
        ha.revert_promotion(png_db, promotion)


def test_a_reverted_audit_with_an_unsafe_column_is_refused(png_db,
                                                           investigation):
    """The stored name is validated on the way out as well as the way in."""
    ha.store_attribution(png_db, investigation)
    promotion = ha.promote_calls(png_db, "run-1", "call_column")
    connection = sqlite3.connect(png_db)
    try:
        connection.execute(
            "UPDATE hit_promotion_audit SET annotation_column='a b'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(HitAttributionError, match="unsafe"):
        ha.revert_promotion(png_db, promotion)


def test_promoting_into_an_unnamed_column_is_refused(png_db, investigation):
    """The same name rule applies to the second promotion path."""
    ha.store_attribution(png_db, investigation)

    with pytest.raises(HitAttributionError, match="letters, numbers"):
        ha.promote_calls(png_db, "run-1", "not a name")


# ---------------------------------------------------------------------------
# what the default feature set is allowed to contain
# ---------------------------------------------------------------------------

def test_the_within_well_rank_of_the_score_is_a_perfect_proxy_for_it(
        hit_frame):
    """`candidate_percentile` is a monotone transform of the model score."""
    within = hit_frame.groupby(["plateID", "rowID", "columnID"])
    correlations = [
        block["candidate_percentile"].corr(block["XGBoost_score"],
                                           method="spearman")
        for _key, block in within]

    assert min(correlations) == pytest.approx(1.0)


@pytest.mark.xfail(strict=True,
                   reason="candidate_rank / candidate_percentile are ranks of "
                          "score_column, so the default feature set carries "
                          "the model output that include_original_score=False "
                          "is meant to exclude")
def test_the_default_features_exclude_every_transform_of_the_score(hit_frame):
    """Excluding the score has to exclude the columns computed from it.

    ``build_hit_cell_frame`` derives ``candidate_rank`` and
    ``candidate_percentile`` from ``score_column`` alone, so a default feature
    set that keeps them trains the attribution model on the very prediction
    the opt-out is there to keep out — and does it without the warning the
    explicit opt-in attaches.
    """
    features = ha._default_features(hit_frame, "XGBoost_score",
                                    include_score=False)

    assert "candidate_rank" not in features
    assert "candidate_percentile" not in features
    assert "cell_area" in features and "cell_texture" in features


# ---------------------------------------------------------------------------
# the refitted permutation nulls
# ---------------------------------------------------------------------------

class _PartialSplitter:
    """A splitter that forgets one row, the way a bad fold would."""

    def __init__(self, n_splits=2, **_kwargs):
        self.n_splits = int(n_splits)

    def split(self, values, labels=None, groups=None):
        order = np.arange(len(values))
        for fold in range(self.n_splits):
            test = order[order % self.n_splits == fold]
            if fold == 0:
                test = test[1:]                # one row never gets scored
            yield np.setdiff1d(order, test), test


def test_a_cross_fit_that_leaves_a_cell_unscored_is_refused(monkeypatch):
    """Every cell must be predicted by a model that did not see it."""
    import sklearn.model_selection as ms

    rng = np.random.default_rng(0)
    values = rng.normal(size=(30, 2))
    # Alternating, so every subset a fold trains on still holds both kinds.
    fractions = np.tile([0.0, 0.2], 15)
    groups = pd.Series(np.repeat(["w1", "w2", "w3"], 10))
    monkeypatch.setattr(ms, "GroupKFold", _PartialSplitter)

    with pytest.raises(HitAttributionError,
                       match="left cells without probabilities"):
        ha._crossfit_mixture(values, fractions, groups)


def test_a_candidate_cross_fit_that_leaves_a_cell_unscored_is_refused(
        monkeypatch, candidate_frame):
    """The non-parametric path holds the same rule."""
    import sklearn.model_selection as ms

    monkeypatch.setattr(ms, "GroupKFold", _PartialSplitter)

    with pytest.raises(HitAttributionError,
                       match="left candidate cells unscored"):
        ha.crossfit_candidate_probabilities(candidate_frame, n_splits=2)


def test_the_refitted_nulls_report_how_many_draws_completed(hit_frame):
    """A permutation p-value states the number of draws behind it."""
    result = ha.fit_hit_attribution(
        hit_frame, target_gene="123", split_by="well", random_seed=4,
        n_bootstrap=30, n_permutations=30, n_pipeline_permutations=2)

    assert result.validation["refitted_permutations"] == 2
    assert result.validation["guide_refitted_permutations_completed"] <= 2
    assert result.validation["well_refitted_permutations_completed"] <= 2
    assert 0.0 < result.validation["guide_fraction_refit_p_value"] <= 1.0


def test_an_unidentified_null_draw_is_omitted_and_counted(monkeypatch,
                                                          hit_frame):
    """A permutation whose fold has one bag class is not evidence."""
    real = ha._crossfit_mixture
    calls = {"n": 0}

    def sometimes(values, fractions, groups):
        # The first call is the real fit; the second is the first null draw.
        calls["n"] += 1
        if calls["n"] == 2:
            raise HitAttributionError("one bag class in this null draw")
        return real(values, fractions, groups)

    monkeypatch.setattr(ha, "_crossfit_mixture", sometimes)

    result = ha.fit_hit_attribution(
        hit_frame, target_gene="123", split_by="well", random_seed=4,
        n_bootstrap=30, n_permutations=30, n_pipeline_permutations=1)

    assert result.validation["refitted_permutations"] == 1
    assert result.validation["guide_refitted_permutations_completed"] == 0
    assert np.isnan(result.validation["guide_fraction_refit_p_value"])
    assert result.validation["well_refitted_permutations_completed"] == 1


def test_a_permutation_that_makes_every_well_positive_is_omitted(monkeypatch,
                                                                 hit_frame):
    """With nothing to contrast against a null draw carries no difference.

    A permutation that leaves every well positive cannot be fitted at all in
    practice, so the fit is stood in for here and the *summarising* half of
    the loop — the one that decides such a draw is not evidence — is what is
    driven.
    """
    def all_positive(frame, fractions, well_columns, rng, *, binary_only):
        return np.full(len(frame), 0.2)

    def scored(values, fractions, groups):
        return (np.full(len(values), 0.8), np.zeros(len(values), dtype=int),
                [1])

    monkeypatch.setattr(ha, "_permuted_well_fractions", all_positive)
    monkeypatch.setattr(ha, "_crossfit_mixture", scored)

    output = ha._refitted_permutation_p_values(
        hit_frame, hit_frame[["cell_area", "cell_texture"]].to_numpy(float),
        hit_frame["target_guide_fraction"].to_numpy(float),
        ha._group_series(hit_frame, list(ha.WELL_COLUMNS)),
        list(ha.WELL_COLUMNS), observed=0.3, iterations=1, random_seed=1,
        threshold=0.5)

    assert output["refitted_permutations"] == 1
    assert output["guide_refitted_permutations_completed"] == 0
    assert output["well_refitted_permutations_completed"] == 0
    assert np.isnan(output["well_label_refit_p_value"])
    assert np.isnan(output["guide_fraction_refit_p_value"])


def test_the_original_score_is_appended_to_an_explicit_feature_list(
        hit_frame):
    """Asking for the score adds it even to a hand-written feature set."""
    result = ha.fit_hit_attribution(
        hit_frame, target_gene="123", split_by="well", random_seed=4,
        n_bootstrap=30, n_permutations=30,
        feature_columns=["cell_area", "cell_texture"],
        include_original_score=True)

    assert result.feature_columns == ["cell_area", "cell_texture",
                                      "XGBoost_score"]
