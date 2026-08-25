"""Which cells the montage shows behind a coefficient, and what it promises.

The montage is the one output in the regression module that a reader will
mistake for genotyped cells if it is allowed to. This screen is POOLED: the
sequencing says a well was 15% GRA14 and never which 15%, so the montage can
only ever show cells consistent with the effect. These tests hold that line
in three places -- the caption always says membership is inferred, the score
window is computed once for the screen instead of once per gene, and the
per-well count is exactly ``round(objects x fraction)`` with every well that
contributes nothing named rather than dropped.

They drive the real crop path too: a synthetic ``merged/*.npy`` with real
intensity and label planes, a real ``measurements.db``, and pixels cut
through :mod:`spacr.crops` rather than asserted about.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.cell_montage import (
    FRACTION_CSV, GUIDE_AGGREGATIONS, INFERENCE_NOTICE, MAX_OBJECTS,
    WINDOW_HALF_WIDTHS, CoefficientNotFound, Coefficient, CropSourceChoice,
    MissingScores, MontageError, MontagePlan, ScoreWindow, WellSelection,
    coefficient_level, guides_for_coefficient, load_montage_objects,
    objects_to_show, read_well_guide_fractions, resolve_montage_crop_source,
    round_half_up, score_window, select_montage, select_montage_per_guide,
    wells_for_coefficient,
)

CELL_DIM, NUC_DIM, PATH_DIM = 4, 5, 6
MASK_DIMS = {"cell": CELL_DIM, "nucleus": NUC_DIM, "pathogen": PATH_DIM}

#: Four wells on one plate, and the objects each holds.
WELLS = ("r1_c1", "r1_c2", "r1_c3", "r1_c4")
OBJECTS_PER_WELL = 8


# ---------------------------------------------------------------------------
# Synthetic screen
# ---------------------------------------------------------------------------

def _field(labels, h=96, w=112, n_channels=4, seed=0):
    """A merged array: four intensity planes then cell / nucleus / pathogen."""
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 4000, size=(h, w, n_channels + 3)).astype(np.uint16)
    for dim in (CELL_DIM, NUC_DIM, PATH_DIM):
        data[:, :, dim] = 0
    for index, label in enumerate(labels):
        y0 = 4 + (index // 4) * 22
        x0 = 4 + (index % 4) * 26
        data[y0:y0 + 18, x0:x0 + 20, CELL_DIM] = label
        data[y0 + 3:y0 + 15, x0 + 3:x0 + 17, NUC_DIM] = label
        data[y0 + 5:y0 + 8, x0 + 5:x0 + 8, PATH_DIM] = label
    return data


def _scores(well_index, n=OBJECTS_PER_WELL):
    """Scores spread evenly, and over a DIFFERENT range in each well.

    The per-well spread has to differ from the screen's, or the test that
    proves the window is measured screen-wide rather than per gene cannot
    tell the two apart.
    """
    start = 0.05 + 0.02 * well_index
    spread = 0.9 - 0.2 * well_index
    return [round(start + spread * i / (n - 1), 4) for i in range(n)]


def _screen(tmp_path, *, with_png=False, wells=WELLS,
            objects_per_well=OBJECTS_PER_WELL):
    """Write a merged folder and a measurements.db for a four-well plate.

    :returns: ``(root, db_path)``.
    """
    root = tmp_path / "exp"
    (root / "merged").mkdir(parents=True)
    (root / "measurements").mkdir(parents=True)
    db_path = str(root / "measurements" / "measurements.db")

    labels = list(range(1, objects_per_well + 1))
    cell_rows, png_rows = [], []
    for well_index, well in enumerate(wells):
        row_id, column_id = well.split("_")
        name = f"plate1_{well}_1"
        npy = str(root / "merged" / f"{name}.npy")
        np.save(npy, _field(labels, seed=well_index))
        png_dir = root / "data" / f"plate1_{well}" / "cell_png"
        if with_png:
            png_dir.mkdir(parents=True, exist_ok=True)
        for label, score in zip(labels, _scores(well_index, objects_per_well)):
            png_path = str(png_dir / f"{name}_{label}.png")
            cell_rows.append((label, "plate1", row_id, column_id, "f1",
                              npy, f"{name}.npy"))
            png_rows.append(("plate1", row_id, column_id, "f1", f"o{label}",
                             png_path,
                             f"plate1_{row_id}_{column_id}_f1_o{label}", score))

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE cell (object_label INTEGER, plateID TEXT, "
                 "rowID TEXT, columnID TEXT, fieldID TEXT, path_name TEXT, "
                 "file_name TEXT)")
    conn.executemany("INSERT INTO cell VALUES (?,?,?,?,?,?,?)", cell_rows)
    conn.execute("CREATE TABLE png_list (plateID TEXT, rowID TEXT, "
                 "columnID TEXT, fieldID TEXT, cell_id TEXT, png_path TEXT, "
                 "prcfo TEXT, pred REAL)")
    conn.executemany("INSERT INTO png_list VALUES (?,?,?,?,?,?,?,?)", png_rows)
    conn.commit()
    conn.close()
    return str(root), db_path


def _objects(wells=WELLS, objects_per_well=OBJECTS_PER_WELL):
    """The per-object frame a montage selects from, without a database."""
    rows = []
    for well_index, well in enumerate(wells):
        row_id, column_id = well.split("_")
        for label, score in zip(range(1, objects_per_well + 1),
                                _scores(well_index, objects_per_well)):
            rows.append({
                "prc": f"plate1_{well}",
                "plateID": "plate1", "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "object_label": label,
                "prcfo": f"plate1_{row_id}_{column_id}_f1_o{label}",
                "pred": score,
            })
    return pd.DataFrame(rows)


def _counts(fractions=None, cell_count=OBJECTS_PER_WELL):
    """A ``regression_data.csv``-shaped count frame.

    ``fractions`` maps well -> {guide: fraction}; the default puts GRA14 in
    the first three wells and nothing in the fourth.
    """
    if fractions is None:
        fractions = {
            "r1_c1": {"GRA14_1": 0.25, "GRA14_2": 0.25, "OTHER_1": 0.5},
            "r1_c2": {"GRA14_1": 0.125, "OTHER_1": 0.875},
            "r1_c3": {"GRA14_2": 0.5, "OTHER_1": 0.5},
            "r1_c4": {"OTHER_1": 1.0},
        }
    genes = {"GRA14_1": "GRA14", "GRA14_2": "GRA14", "OTHER_1": "OTHER"}
    rows = []
    for well, guides in fractions.items():
        row_id, column_id = well.split("_")
        for guide, fraction in guides.items():
            rows.append({
                "prc": f"plate1_{well}", "plateID": "plate1",
                "rowID": row_id, "columnID": column_id,
                "grna": guide, "gene": genes[guide], "fraction": fraction,
                "cell_count": cell_count, "pred": 0.5,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The count rule
# ---------------------------------------------------------------------------

def test_the_count_rule_is_round_objects_times_fraction():
    """200 cells at 15% contributes 30 -- the maintainer's own example."""
    assert objects_to_show(200, 0.15) == 30
    assert objects_to_show(8, 0.25) == 2
    assert objects_to_show(8, 0.125) == 1
    assert objects_to_show(0, 0.9) == 0


def test_a_half_does_not_depend_on_the_parity_of_the_number_before_it():
    """Banker's rounding would make two identical wells contribute differently.

    ``round(2.5)`` is 2 in Python and ``round(3.5)`` is 4, so a well at 5
    objects x 50% and one at 7 x 50% would round in opposite directions while
    the caption claimed one rule for both.
    """
    assert round(2.5) == 2 and round(3.5) == 4          # what we refuse
    assert round_half_up(2.5) == 3 and round_half_up(3.5) == 4
    assert objects_to_show(5, 0.5) == 3
    assert objects_to_show(7, 0.5) == 4


def test_a_fraction_that_is_not_a_fraction_is_refused_not_clipped():
    """A percentage clipped to 1.0 would show a whole well as one guide's."""
    with pytest.raises(MontageError, match="not a fraction"):
        objects_to_show(200, 15)
    with pytest.raises(MontageError, match="not a fraction"):
        objects_to_show(200, -0.1)
    with pytest.raises(MontageError, match="cannot hold"):
        objects_to_show(-1, 0.5)
    with pytest.raises(MontageError, match="not a fraction"):
        objects_to_show(200, float("nan"))


def test_a_count_that_is_not_a_number_never_becomes_a_montage_size():
    """NaN silently floors to a number in some code paths; not in this one."""
    with pytest.raises(MontageError, match="non-finite count"):
        round_half_up(float("nan"))
    with pytest.raises(MontageError, match="non-finite count"):
        round_half_up(float("inf"))
    # Halves go away from zero on both sides, so the rule is one rule.
    assert round_half_up(-2.5) == -3
    assert round_half_up(-2.4) == -2


# ---------------------------------------------------------------------------
# The score window
# ---------------------------------------------------------------------------

def test_the_implied_score_is_the_baseline_plus_the_coefficient():
    """A well entirely of guide-carrying cells scores baseline + effect."""
    objects = _objects()
    window = score_window(objects, 0.2)
    assert window.baseline == pytest.approx(np.median(objects["pred"]))
    assert window.baseline_source == "screen_median"
    assert window.target == pytest.approx(window.baseline + 0.2)
    half = WINDOW_HALF_WIDTHS * window.scale
    assert window.low == pytest.approx(window.target - half)
    assert window.high == pytest.approx(window.target + half)


def test_the_window_is_measured_once_for_the_screen_not_once_per_gene():
    """A window recomputed per gene can be tuned until the pictures look right.

    Two different coefficients over the same objects must share a baseline
    and a scale; only the target may move, and it moves by exactly the
    difference in effect.
    """
    objects = _objects()
    a = score_window(objects, 0.2)
    b = score_window(objects, -0.4)
    assert a.baseline == b.baseline
    assert a.scale == b.scale
    assert a.n_scored == b.n_scored == len(objects)
    assert b.target - a.target == pytest.approx(-0.6)


def test_selecting_from_one_well_would_move_the_window_so_the_whole_frame_is_used():
    """The proof that pre-filtering the objects is what makes a window tunable.

    A single well's scores give a different baseline and a different scale
    from the screen's, so ``select_montage`` takes the whole object frame and
    the window it records is the screen-wide one.
    """
    objects = _objects()
    one_well = objects[objects["prc"] == "plate1_r1_c1"]
    whole = score_window(objects, 0.2)
    narrow = score_window(one_well, 0.2)
    assert narrow.scale != pytest.approx(whole.scale), (
        "the fixture must make a single well's spread differ from the "
        "screen's or this test proves nothing")

    plan = select_montage(objects, _counts(), "GRA14", 0.2)
    assert plan.window.scale == pytest.approx(whole.scale)
    assert plan.window.n_scored == len(objects)


def test_a_widened_window_is_recorded_so_a_figure_cannot_hide_it():
    """Overriding the one width is allowed; concealing the override is not."""
    objects = _objects()
    wide = score_window(objects, 0.2, half_widths=3.0)
    assert wide.half_widths == 3.0
    assert "+/-3 robust scales" in wide.describe()
    plan = select_montage(objects, _counts(), "GRA14", 0.2, half_widths=3.0)
    assert "+/-3 robust scales" in plan.caption()


def test_a_window_with_no_positive_width_is_refused():
    with pytest.raises(MontageError, match="positive number of robust scales"):
        score_window(_objects(), 0.2, half_widths=0)
    with pytest.raises(MontageError, match="positive number of robust scales"):
        score_window(_objects(), 0.2, half_widths=float("inf"))


def test_a_fitted_intercept_can_stand_in_for_the_screen_median():
    """A caller with the model's own baseline may use it, and it is recorded."""
    window = score_window(_objects(), 0.2, baseline=0.4)
    assert window.baseline == 0.4
    assert window.baseline_source == "given"
    assert window.target == pytest.approx(0.6)
    assert "baseline 0.4 from given" in window.describe()
    with pytest.raises(MontageError, match="not a finite score"):
        score_window(_objects(), 0.2, baseline=float("nan"))


def test_a_screen_where_every_object_scores_the_same_says_closest_means_nothing():
    """A degenerate window admits everything rather than admitting nothing."""
    objects = _objects()
    objects["pred"] = 0.5
    window = score_window(objects, 0.2)
    assert window.degenerate
    assert window.contains([0.5, 0.9, -3.0]).all()
    assert "no width" in window.describe()
    plan = select_montage(objects, _counts(), "GRA14", 0.2)
    assert any("does not distinguish" in note for note in plan.notes)


def test_a_score_the_screen_never_reaches_is_reported_as_unreachable():
    """"No cell scores anything like this" is a finding, not a failure."""
    objects = _objects()
    window = score_window(objects, 50.0)
    assert not window.target_is_observable
    plan = select_montage(objects, _counts(), "GRA14", 50.0)
    assert "lies OUTSIDE the observed range" in plan.caption()


def test_a_missing_score_column_names_the_column_classify_writes():
    objects = _objects().drop(columns=["pred"])
    with pytest.raises(MissingScores, match="png_list"):
        score_window(objects, 0.2)


def test_scores_that_are_all_missing_are_not_a_montage():
    objects = _objects()
    objects["pred"] = np.nan
    with pytest.raises(MissingScores, match="finite"):
        score_window(objects, 0.2)


def test_a_missing_score_is_outside_the_window_not_near_it():
    window = score_window(_objects(), 0.2)
    assert not window.contains([np.nan, np.inf])[0]
    assert not window.contains([np.nan, np.inf])[1]


# ---------------------------------------------------------------------------
# The wells
# ---------------------------------------------------------------------------

def test_a_gene_sums_the_fractions_of_its_guides():
    """One coefficient per gene describes the well's whole GRA14 fraction."""
    wells = wells_for_coefficient(_counts(), "GRA14")
    assert list(wells["prc"]) == ["plate1_r1_c1", "plate1_r1_c2", "plate1_r1_c3"]
    assert wells.set_index("prc")["fraction"].to_dict() == {
        "plate1_r1_c1": 0.5, "plate1_r1_c2": 0.125, "plate1_r1_c3": 0.5}
    assert "plate1_r1_c4" not in set(wells["prc"]), (
        "the well with no GRA14 guide is not a GRA14 well")


def test_one_guide_names_only_the_wells_that_guide_is_in():
    wells = wells_for_coefficient(_counts(), "GRA14_2")
    assert list(wells["prc"]) == ["plate1_r1_c1", "plate1_r1_c3"]
    assert list(wells["fraction"]) == [0.25, 0.5]


def test_the_level_is_read_off_the_count_data_and_a_guide_wins_a_tie():
    counts = _counts()
    assert coefficient_level(counts, "GRA14") == "gene"
    assert coefficient_level(counts, "GRA14_1") == "grna"
    counts.loc[counts.index[0], "gene"] = "GRA14_1"
    assert coefficient_level(counts, "GRA14_1") == "grna", (
        "resolving the tie as the gene would silently widen the montage to "
        "that gene's other guides")


def test_guides_that_sum_past_a_whole_well_are_refused():
    """Two sequencing runs concatenated is the usual cause, and it is fatal."""
    counts = _counts({"r1_c1": {"GRA14_1": 0.7, "GRA14_2": 0.8}})
    with pytest.raises(MontageError, match="more than the whole well"):
        wells_for_coefficient(counts, "GRA14")


def test_a_percentage_where_a_fraction_belongs_is_refused():
    counts = _counts({"r1_c1": {"GRA14_1": 25.0}})
    with pytest.raises(MontageError, match="A percentage is not a fraction"):
        wells_for_coefficient(counts, "GRA14")


def test_a_coefficient_the_count_data_never_saw_is_named_not_guessed():
    with pytest.raises(CoefficientNotFound, match="ROP18"):
        wells_for_coefficient(_counts(), "ROP18")


def test_a_guide_in_the_results_but_not_in_the_counts_is_named_not_empty():
    """The tab knows the level from which results CSV the dot came off.

    ``perform_regression`` drops outlier gRNAs from the merged frame after
    the results are written, so a point on the plot can legitimately name a
    guide the count data no longer holds. Passing the level explicitly must
    still produce a sentence, not a montage of nothing.
    """
    with pytest.raises(CoefficientNotFound, match="GRA14_9"):
        wells_for_coefficient(_counts(), "GRA14_9", level="grna")


def test_a_guide_reported_only_at_zero_is_present_nowhere():
    counts = _counts({"r1_c1": {"GRA14_1": 0.0, "OTHER_1": 1.0}})
    with pytest.raises(CoefficientNotFound, match="fraction of zero"):
        wells_for_coefficient(counts, "GRA14")


def test_guides_for_a_gene_come_back_sorted_and_complete():
    assert guides_for_coefficient(_counts(), "GRA14") == ["GRA14_1", "GRA14_2"]
    assert guides_for_coefficient(_counts(), "GRA14_1") == ["GRA14_1"]


def test_count_data_without_a_guide_or_a_fraction_column_is_named():
    counts = _counts().drop(columns=["fraction"])
    with pytest.raises(MontageError, match=r"missing \['fraction'\]"):
        wells_for_coefficient(counts, "GRA14")


def test_count_data_that_names_no_well_is_refused():
    counts = _counts().drop(columns=["prc", "plateID", "rowID", "columnID"])
    with pytest.raises(MontageError, match="names no well"):
        wells_for_coefficient(counts, "GRA14")


def test_a_count_frame_keyed_on_plate_row_column_works_without_prc():
    """Not every count table carries the composed key; both spellings serve."""
    counts = _counts().drop(columns=["prc"])
    wells = wells_for_coefficient(counts, "GRA14")
    assert list(wells.columns[:3]) == ["plateID", "rowID", "columnID"]
    objects = _objects().drop(columns=["prc"])
    plan = select_montage(objects, counts, "GRA14", 0.2, half_widths=10.0)
    assert plan.n_objects == 9
    assert set(plan.objects["montage_well"]) == {
        "plate1_r1_c1", "plate1_r1_c2", "plate1_r1_c3"}


def test_a_fraction_that_is_missing_is_not_treated_as_zero():
    counts = _counts()
    counts.loc[counts["grna"] == "GRA14_1", "fraction"] = np.nan
    with pytest.raises(MontageError, match="missing or non-numeric"):
        wells_for_coefficient(counts, "GRA14")


def test_an_unknown_guide_aggregation_is_refused_at_the_well_step_too():
    with pytest.raises(MontageError, match="guide_aggregation must be"):
        wells_for_coefficient(_counts(), "GRA14", guide_aggregation="mean")


def test_separate_aggregation_keeps_one_row_per_well_and_guide():
    wells = wells_for_coefficient(_counts(), "GRA14",
                                  guide_aggregation="separate")
    assert list(zip(wells["prc"], wells["grna"], wells["fraction"])) == [
        ("plate1_r1_c1", "GRA14_1", 0.25),
        ("plate1_r1_c1", "GRA14_2", 0.25),
        ("plate1_r1_c2", "GRA14_1", 0.125),
        ("plate1_r1_c3", "GRA14_2", 0.5),
    ]


# ---------------------------------------------------------------------------
# The selection
# ---------------------------------------------------------------------------

def test_each_well_contributes_round_objects_times_fraction():
    """The success criterion instruction 131 states, checked well by well."""
    plan = select_montage(_objects(), _counts(), "GRA14", 0.2, half_widths=10.0)
    per_well = {w.well: w for w in plan.wells}
    assert per_well["plate1_r1_c1"].n_expected == 4     # round(8 x 0.5)
    assert per_well["plate1_r1_c2"].n_expected == 1     # round(8 x 0.125)
    assert per_well["plate1_r1_c3"].n_expected == 4     # round(8 x 0.5)
    for well in plan.wells:
        assert well.n_selected == well.n_expected, well.describe()
    assert plan.n_objects == 9
    assert plan.objects["montage_well"].value_counts().to_dict() == {
        "plate1_r1_c1": 4, "plate1_r1_c2": 1, "plate1_r1_c3": 4}


def test_the_objects_chosen_are_the_top_ranked_by_score():
    """Instruction 172: "rank all cells by classefication score and take the
    top x cells". The score WINDOW no longer decides which -- it decided
    independently of the count, so a well could expect objects and contribute
    none, which is what the maintainer saw."""
    objects = _objects()
    plan = select_montage(objects, _counts(), "GRA14", 0.2, half_widths=10.0)
    for well, frame in plan.objects.groupby("montage_well"):
        pool = objects[objects["prc"] == well]
        # A POSITIVE effect, so the highest scores.
        wanted = pool.sort_values("pred", ascending=False).head(len(frame))
        assert sorted(frame["object_label"]) == sorted(wanted["object_label"]), (
            f"{well} did not take its highest-scoring cells")


def test_a_negative_coefficient_takes_the_LOWEST_scores():
    """Always-descending would show a negative coefficient the cells least
    consistent with it."""
    objects = _objects()
    plan = select_montage(objects, _counts(), "GRA14", -0.2, half_widths=10.0)
    for well, frame in plan.objects.groupby("montage_well"):
        pool = objects[objects["prc"] == well]
        wanted = pool.sort_values("pred", ascending=True).head(len(frame))
        assert sorted(frame["object_label"]) == sorted(wanted["object_label"])


def test_a_well_that_rounds_to_zero_is_reported_not_dropped():
    """A well contributing nothing still exists, and the caption says so."""
    counts = _counts({
        "r1_c1": {"GRA14_1": 0.5, "OTHER_1": 0.5},
        "r1_c2": {"GRA14_1": 0.02, "OTHER_1": 0.98},
    })
    plan = select_montage(_objects(), counts, "GRA14", 0.2, half_widths=10.0)
    zero = {w.well: w for w in plan.zero_wells}
    assert "plate1_r1_c2" in zero
    assert zero["plate1_r1_c2"].n_expected == 0
    assert "rounds to zero" in zero["plate1_r1_c2"].note
    assert "plate1_r1_c2" in plan.caption()
    assert "contributed nothing" in plan.caption()


def test_a_narrow_window_no_longer_starves_a_well():
    """The window decided WHICH cells independently of how many were owed, so
    a well could expect 23 and contribute 0 (instruction 172). Ranking cannot
    do that: x cells are shown whenever the well has x scored cells."""
    plan = select_montage(_objects(), _counts(), "GRA14", 0.2,
                          half_widths=0.05)

    starved = [w for w in plan.wells
               if w.n_expected and not w.n_selected]

    assert not starved, (
        f"{[w.well for w in starved]} were owed cells and given none")


def test_a_well_with_fewer_scored_cells_than_it_is_owed_says_so():
    """The only honest shortfall left: the well does not HAVE that many."""
    plan = select_montage(_objects(), _counts(), "GRA14", 0.2,
                          half_widths=10.0)

    for well in plan.wells:
        if well.n_selected < well.n_expected:
            assert "carry a classification score" in well.note
        elif well.n_selected:
            # And every well that filled says the arithmetic it filled by.
            assert "normalised by" in well.note


def test_a_well_with_no_objects_in_the_database_is_named():
    """The count data can report a well the measurements never covered."""
    objects = _objects(wells=("r1_c1", "r1_c2"))
    plan = select_montage(objects, _counts(), "GRA14", 0.2, half_widths=10.0)
    missing = {w.well: w for w in plan.wells}["plate1_r1_c3"]
    assert missing.n_objects == 0 and missing.n_selected == 0
    assert "no object in the imported databases" in missing.note


def test_the_cap_keeps_the_closest_and_the_caption_says_what_was_trimmed():
    plan = select_montage(_objects(), _counts(), "GRA14", 0.2,
                          half_widths=10.0, cap=3)
    assert plan.capped
    assert plan.n_objects == 3
    assert plan.n_before_cap == 9
    assert "capped at 3 of 9 objects" in plan.caption()
    trimmed = [w for w in plan.wells if "trimmed by the montage cap" in w.note]
    assert trimmed, "a trimmed well must say its count is no longer the rule"
    uncapped = select_montage(_objects(), _counts(), "GRA14", 0.2,
                              half_widths=10.0)
    keep = uncapped.objects.nsmallest(3, "montage_distance")
    assert sorted(plan.objects["prcfo"]) == sorted(keep["prcfo"])


def test_a_cap_that_is_not_a_count_is_refused():
    with pytest.raises(MontageError, match="positive object count"):
        select_montage(_objects(), _counts(), "GRA14", 0.2, cap=0)


def test_an_empty_object_frame_says_to_import_a_database():
    with pytest.raises(MontageError, match="Import a measurement database"):
        select_montage(_objects().iloc[0:0], _counts(), "GRA14", 0.2)


def test_two_identical_requests_return_the_same_objects_in_the_same_order():
    """A montage a user cannot reproduce is a montage they cannot cite."""
    shuffled = _objects().sample(frac=1.0, random_state=7).reset_index(drop=True)
    first = select_montage(_objects(), _counts(), "GRA14", 0.2, half_widths=10.0)
    second = select_montage(shuffled, _counts(), "GRA14", 0.2, half_widths=10.0)
    assert list(first.objects["prcfo"]) == list(second.objects["prcfo"])
    assert list(first.objects["montage_rank"]) == list(range(1, first.n_objects + 1))


def test_cell_count_disagreeing_with_the_objects_present_is_disclosed():
    """The rule ran on the objects that exist, and the caption admits it."""
    plan = select_montage(_objects(), _counts(cell_count=200), "GRA14", 0.2,
                          half_widths=10.0)
    assert any("cell_count in the count data disagrees" in n for n in plan.notes)
    assert "200 reported, 8 present" in plan.caption()
    assert {w.n_reported for w in plan.wells if w.n_objects} == {200}
    assert {w.n_objects for w in plan.wells if w.n_objects} == {8}


def test_the_object_and_count_frames_must_share_a_well_key():
    objects = _objects().drop(columns=["prc", "plateID", "rowID", "columnID"])
    with pytest.raises(MontageError, match="share no well key"):
        select_montage(objects, _counts(), "GRA14", 0.2)


# ---------------------------------------------------------------------------
# The caption -- the part that cannot be removed
# ---------------------------------------------------------------------------

def test_the_caption_says_membership_is_inferred_and_names_the_guide():
    """The one thing this feature must never let a reader get wrong."""
    plan = select_montage(_objects(), _counts(), "GRA14", 0.7, half_widths=10.0)
    caption = plan.caption()
    assert caption.endswith(INFERENCE_NOTICE.format(name="GRA14"))
    assert "INFERRED, not observed" in caption
    assert "pooled screen" in caption
    assert "never which cells did" in caption
    assert "not genotyped cells" in caption


def test_the_caption_states_the_wells_the_window_and_the_count_rule():
    """Instruction 131's acceptance list, read off one string."""
    plan = select_montage(_objects(), _counts(), "GRA14", 0.7, half_widths=10.0)
    caption = plan.caption()
    assert "plate1_r1_c1" in caption and "plate1_r1_c2" in caption
    assert "score window:" in caption
    assert "measured once over all 32 objects, not per gene" in caption
    assert "round(objects in well x guide fraction in well)" in caption
    assert "GRA14_1, GRA14_2" in caption


def test_even_a_montage_with_no_objects_carries_the_notice():
    """An empty tab explains itself; it does not drop the disclaimer with it."""
    counts = _counts({"r1_c1": {"GRA14_1": 0.02, "OTHER_1": 0.98}})
    plan = select_montage(_objects(), counts, "GRA14", 0.2, half_widths=10.0)
    assert plan.is_empty
    assert plan.caption().endswith(INFERENCE_NOTICE.format(name="GRA14"))
    assert "no object was selected" in plan.caption()
    assert "rounds to zero" in plan.caption()


def test_the_summary_is_the_one_line_a_status_bar_can_hold():
    plan = select_montage(_objects(), _counts(), "GRA14", 0.7, half_widths=10.0)
    assert plan.summary() == (
        "GRA14 (gene, effect +0.700): 9 objects from 3 wells")


def test_a_non_finite_coefficient_implies_no_score_at_all():
    with pytest.raises(MontageError, match="non-finite effect"):
        Coefficient(name="GRA14", effect=float("nan"))
    with pytest.raises(MontageError, match="level must be"):
        Coefficient(name="GRA14", effect=0.1, level="guide")


# ---------------------------------------------------------------------------
# Guides summed versus guides apart
# ---------------------------------------------------------------------------

def test_guides_shown_apart_is_a_different_montage_from_guides_summed():
    """Two questions: "is the gene lost" and "do the guides agree"."""
    summed = select_montage(_objects(), _counts(), "GRA14", 0.2,
                            half_widths=10.0)
    apart = select_montage_per_guide(_objects(), _counts(), "GRA14", 0.2,
                                     half_widths=10.0)
    assert [p.guides for p in apart] == [("GRA14_1",), ("GRA14_2",)]
    assert {p.guide_aggregation for p in apart} == {"separate"}
    assert summed.guide_aggregation == "sum"
    # GRA14_1 is 0.25 and 0.125 of two wells; the gene is 0.5 and 0.125 of
    # three. Summing is not the same picture as either guide alone.
    assert [p.n_objects for p in apart] == [3, 6]
    assert summed.n_objects == 9
    assert "one guide at a time" in apart[0].caption()
    assert "fractions summed" in summed.caption()


def test_a_guide_no_well_reports_is_skipped_rather_than_faking_a_montage():
    """A gene's guide that never appears leaves the other guides' plans alone."""
    counts = _counts()
    counts = counts[counts["grna"] != "GRA14_1"]
    counts = pd.concat([counts, pd.DataFrame([{
        "prc": "plate1_r1_c1", "plateID": "plate1", "rowID": "r1",
        "columnID": "c1", "grna": "GRA14_1", "gene": "GRA14",
        "fraction": 0.0, "cell_count": 8, "pred": 0.5}])], ignore_index=True)
    plans = select_montage_per_guide(_objects(), counts, "GRA14", 0.2,
                                     half_widths=10.0)
    assert [p.guides for p in plans] == [("GRA14_2",)]


def test_a_guide_list_no_well_reports_is_an_error_not_an_empty_plan():
    counts = _counts()
    with pytest.raises(CoefficientNotFound, match="GRA14_9"):
        select_montage(_objects(), counts, "GRA14", 0.2,
                       guide_aggregation="separate", guides=["GRA14_9"])


def test_asking_for_separate_guides_from_the_single_plan_call_is_refused():
    """Silently summing when the caller asked to separate is a wrong figure."""
    with pytest.raises(MontageError, match="select_montage_per_guide"):
        select_montage(_objects(), _counts(), "GRA14", 0.2,
                       guide_aggregation="separate")
    with pytest.raises(MontageError, match="guide_aggregation must be"):
        select_montage(_objects(), _counts(), "GRA14", 0.2,
                       guide_aggregation="mean")
    assert GUIDE_AGGREGATIONS == ("sum", "separate")


# ---------------------------------------------------------------------------
# The CSVs -- the premise correction
# ---------------------------------------------------------------------------

def test_the_two_qc_csvs_that_look_right_are_refused_by_name(tmp_path):
    """Neither names a well AND a guide AND a fraction, so neither can serve.

    ``grna_well.csv`` is ``grna, plateID, grna_well_count, gene_well_count``
    and ``well_grna.csv`` is ``prc, gene_count`` -- measured off a real
    ``perform_regression`` run, and asserted by
    ``tests/test_cov_ml_perform_regression.py``. Guessing either one is the
    obvious mistake, so it is refused with what the file actually holds.
    """
    grna_well = tmp_path / "grna_well.csv"
    pd.DataFrame({"grna": ["GRA14_1"], "plateID": ["plate1"],
                  "grna_well_count": [9], "gene_well_count": [9]}).to_csv(
        grna_well, index=False)
    well_grna = tmp_path / "well_grna.csv"
    pd.DataFrame({"prc": ["plate1_r1_c1"], "gene_count": [4]}).to_csv(
        well_grna, index=False)

    with pytest.raises(MontageError, match="HOW MANY wells a guide was seen in"):
        read_well_guide_fractions(str(grna_well))
    with pytest.raises(MontageError, match="does not name a guide at all"):
        read_well_guide_fractions(str(well_grna))
    for path in (grna_well, well_grna):
        with pytest.raises(MontageError, match=FRACTION_CSV):
            read_well_guide_fractions(str(path))


def test_regression_data_csv_is_the_file_that_carries_the_fractions(tmp_path):
    counts = _counts()
    counts.to_csv(tmp_path / FRACTION_CSV, index=False)
    frame = read_well_guide_fractions(str(tmp_path))
    assert set(frame.columns) >= {"prc", "grna", "gene", "fraction"}
    wells = wells_for_coefficient(frame, "GRA14")
    assert len(wells) == 3


def test_a_missing_fraction_csv_names_what_is_missing(tmp_path):
    with pytest.raises(MontageError, match=FRACTION_CSV):
        read_well_guide_fractions(str(tmp_path))


def test_a_fraction_csv_without_a_well_key_is_refused(tmp_path):
    pd.DataFrame({"grna": ["g"], "gene": ["G"], "fraction": [0.5]}).to_csv(
        tmp_path / FRACTION_CSV, index=False)
    with pytest.raises(MontageError, match="names no well"):
        read_well_guide_fractions(str(tmp_path))


# ---------------------------------------------------------------------------
# The database, and the pixels
# ---------------------------------------------------------------------------

def test_objects_load_carrying_both_a_png_path_and_a_merged_label(tmp_path):
    """One frame serves both crop sources, which is what lets either draw it."""
    _root, db_path = _screen(tmp_path, with_png=True)
    frame = load_montage_objects(db_path)
    assert len(frame) == len(WELLS) * OBJECTS_PER_WELL
    assert {"png_path", "path_name", "object_label", "pred",
            "prc"}.issubset(frame.columns)
    per_well = frame.groupby("prc")["object_label"].apply(
        lambda s: sorted(int(v) for v in s)).to_dict()
    assert per_well == {f"plate1_{w}": list(range(1, OBJECTS_PER_WELL + 1))
                        for w in WELLS}, (
        "'o12' has to come back as the integer 12 a merged crop is cut by")
    assert frame["path_name"].str.endswith(".npy").all()
    assert set(frame["prc"]) == {f"plate1_{w}" for w in WELLS}


def test_a_database_without_classification_scores_says_so(tmp_path):
    _root, db_path = _screen(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE png_list RENAME TO png_list_old")
    conn.execute("CREATE TABLE png_list (plateID TEXT, rowID TEXT, "
                 "columnID TEXT, fieldID TEXT, cell_id TEXT, png_path TEXT)")
    conn.commit()
    conn.close()
    # The refusal now offers BOTH routes (instruction 167): the scores may be
    # in the score CSVs the run was fitted on, and telling a user to go and
    # produce something they already have loaded was the reported fault.
    with pytest.raises(MissingScores) as raised:
        load_montage_objects(db_path)

    message = str(raised.value)
    assert "run Classify" in message
    assert "load the score CSVs" in message


def test_a_database_that_is_not_there_is_not_a_traceback(tmp_path):
    with pytest.raises(MontageError, match="not found"):
        load_montage_objects(str(tmp_path / "nope.db"))


def test_a_database_with_no_crop_table_says_there_are_no_crops(tmp_path):
    _root, db_path = _screen(tmp_path)
    with pytest.raises(MontageError, match="no per-object crops to show"):
        load_montage_objects(db_path, table="not_a_table")


def test_a_png_only_project_still_loads_when_the_merged_join_finds_nothing(tmp_path):
    """A PNG folder alone is a montage; the merged join failing is not fatal."""
    _root, db_path = _screen(tmp_path, with_png=True)
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE cell")
    conn.commit()
    conn.close()
    frame = load_montage_objects(db_path)
    assert len(frame) == len(WELLS) * OBJECTS_PER_WELL
    assert frame["object_type"].eq("cell").all()
    assert frame["png_path"].str.endswith(".png").all()
    plan = select_montage(frame, _counts(), "GRA14", 0.2, half_widths=10.0)
    assert plan.n_objects == 9


def test_no_crop_source_is_an_answer_the_tab_can_show(tmp_path):
    """A tab that cannot be filled says why; it does not raise into the GUI."""
    bare = tmp_path / "bare"
    bare.mkdir()
    choice = resolve_montage_crop_source(str(bare))
    assert isinstance(choice, CropSourceChoice)
    assert choice.available is False
    assert choice.source is None
    assert "no crop source" in choice.describe()
    assert "merged" in choice.reason


def test_the_source_says_which_of_the_two_it_picked(tmp_path):
    root, _db = _screen(tmp_path, with_png=True)
    png = resolve_montage_crop_source(root)
    assert png.available and png.kind == "png"
    # The reason now names the ROUTE in the words a user sees (171): "load
    # images" for the crops already on disk, "stream images" for cutting from
    # merged/. It still says which of the two was picked, which is what this
    # test is for.
    from spacr.crops import LOAD_IMAGES_LABEL

    assert LOAD_IMAGES_LABEL in png.reason
    merged = resolve_montage_crop_source(root, prefer="merged")
    assert merged.kind == "merged"
    assert "png crop source" in png.describe()


def test_the_montage_cuts_real_pixels_out_of_the_merged_arrays(tmp_path):
    """End to end: click a coefficient, get cells, from merged/*.npy.

    Nothing here is a stand-in -- the arrays are real ``.npy`` files with
    intensity and label planes, the database is a real ``measurements.db``,
    and the crops come back through :mod:`spacr.crops`.
    """
    from spacr.crops import CropSpec, MergedCropSource

    root, db_path = _screen(tmp_path)
    objects = load_montage_objects(db_path)
    choice = resolve_montage_crop_source(root)
    assert choice.kind == "merged", choice.describe()

    plan = select_montage(objects, _counts(), "GRA14", 0.2, half_widths=10.0,
                          crop_source=choice)
    assert plan.n_objects == 9
    assert plan.source_kind == "merged"
    assert "merged crop source" in plan.caption()

    source = MergedCropSource(
        spec=CropSpec(merged_path="", channels=(0, 1, 2), size=(48, 48),
                      mask_dims=MASK_DIMS),
        merged_root=os.path.join(root, "merged"))
    crops = source.get_many(plan.rows())
    assert len(crops) == plan.n_objects
    for crop in crops:
        assert crop.shape == (48, 48, 3)
        assert crop.dtype == np.uint8
        assert crop.any(), "an all-black crop means the label was not found"
    # Different objects, so different pixels.
    assert len({crop.tobytes() for crop in crops}) == len(crops)


def test_the_montage_reads_the_exported_pngs_when_they_exist(tmp_path):
    """The PNG half, driven the same way, through the same plan."""
    from PIL import Image

    from spacr.crops import PngCropSource

    root, db_path = _screen(tmp_path, with_png=True)
    objects = load_montage_objects(db_path)
    for path in objects["png_path"]:
        Image.fromarray(
            np.full((48, 48, 3), 40, dtype=np.uint8)).save(path)

    choice = resolve_montage_crop_source(root)
    assert choice.kind == "png"
    plan = select_montage(objects, _counts(), "GRA14", 0.2, half_widths=10.0,
                          crop_source=choice)
    assert plan.source_kind == "png"
    assert "png crop source" in plan.caption()

    crops = PngCropSource(root=root).get_many(plan.rows())
    assert len(crops) == plan.n_objects
    assert all(crop.shape == (48, 48, 3) for crop in crops)
    assert all((crop == 40).all() for crop in crops)


# ---------------------------------------------------------------------------
# The shapes the Qt tab reads
# ---------------------------------------------------------------------------

def test_the_plan_hands_the_tab_rows_a_crop_source_can_take():
    plan = select_montage(_objects(), _counts(), "GRA14", 0.2, half_widths=10.0)
    rows = plan.rows()
    assert len(rows) == plan.n_objects
    assert isinstance(rows[0], dict)
    assert {"object_label", "montage_well", "montage_distance",
            "montage_rank"}.issubset(rows[0])
    assert isinstance(plan, MontagePlan)
    assert isinstance(plan.window, ScoreWindow)
    assert all(isinstance(w, WellSelection) for w in plan.wells)


def test_the_defaults_are_the_ones_the_caption_quotes():
    assert WINDOW_HALF_WIDTHS == 1.0
    # 2,000, not 300: the cost of a larger cap was measured and the
    # maintainer chose the number. 10,000 was asked for and refused on the
    # measurement -- twelve open well tabs would hold 16.8 GB.
    assert MAX_OBJECTS == 2000
    plan = select_montage(_objects(), _counts(), "GRA14", 0.2)
    assert plan.cap == MAX_OBJECTS
    assert plan.window.half_widths == WINDOW_HALF_WIDTHS
    assert plan.score_column == "pred"


def test_the_summary_says_when_a_montage_was_capped():
    plan = select_montage(_objects(), _counts(), "GRA14", 0.2,
                          half_widths=10.0, cap=3)
    assert plan.summary() == (
        "GRA14 (gene, effect +0.200): 3 objects from 3 wells "
        "(capped from 9)")


def test_objects_with_no_stable_key_still_come_back_in_a_fixed_order():
    """A frame that carries no prcfo, path or label falls back to its index.

    The montage still has to be reproducible: two calls over the same rows
    return the same objects, in the same order, tie-breaks included.
    """
    objects = _objects().drop(columns=["prcfo", "object_label"])
    first = select_montage(objects, _counts(), "GRA14", 0.2, half_widths=10.0)
    second = select_montage(objects, _counts(), "GRA14", 0.2, half_widths=10.0)
    assert first.n_objects == 9
    assert list(first.objects["pred"]) == list(second.objects["pred"])


def test_a_well_selection_describes_itself_in_the_caption_s_own_terms():
    well = WellSelection(well="plate1_r1_c1", fraction=0.5, n_objects=8,
                         n_reported=8, n_expected=4, n_in_window=6,
                         n_selected=4)
    assert well.describe() == "plate1_r1_c1: 4 of round(8 x 0.5) = 4"
    assert well.contributed
    starved = WellSelection(well="plate1_r1_c2", fraction=0.5, n_objects=8,
                            n_reported=8, n_expected=4, n_in_window=1,
                            n_selected=1, note="only 1 of 4 objects fall "
                                               "inside the score window")
    assert starved.describe().endswith("inside the score window")


# --------------------------------------------------------------------------- #
#  Instruction 155 B -- the montage shows its arithmetic
# --------------------------------------------------------------------------- #

def test_the_caption_states_the_whole_sum_a_reader_can_check():
    """Every number in the rule, with the per-well counts that add to it.

    The montage asks to be trusted on ``round(n x fraction)`` and on a score
    window nobody can see, so it states both in full: where the baseline came
    from and over how many objects, ``baseline + effect`` worked through with
    the actual numbers, ``1.4826 x MAD``, the bounds, and the per-well line
    for EVERY well ending in a total whose terms are printed beside it.
    """
    plan = select_montage(_objects(), _counts(), "GRA14", 0.2)
    arithmetic = plan.arithmetic()
    window = plan.window

    assert "median per-object pred over EVERY object supplied" in arithmetic
    assert f"({window.n_scored:,} objects" in arithmetic
    assert (f"target   = baseline + effect = {window.baseline:.6g} + 0.2 = "
            f"{window.target:.6g}") in arithmetic
    assert f"1.4826 x MAD of those same scores = {window.scale:.6g}" in arithmetic
    assert (f"window   = target +/- 1 x scale = [{window.low:.6g}, "
            f"{window.high:.6g}]") in arithmetic

    # The per-well line for each well, in the caption's own spelling, and a
    # total whose terms are printed so the sum can be added up by hand.
    assert "round(8 x 0.5) = 4 -> 4 shown" in arithmetic
    assert "round(8 x 0.125) = 1 -> 1 shown" in arithmetic
    assert "total: 4 + 1 + 4 = 9 objects shown" in arithmetic
    assert sum(w.n_selected for w in plan.wells) == plan.n_objects == 9
    # And it is IN the caption, not merely available beside it.
    assert arithmetic in plan.caption()


def test_a_degenerate_window_says_so_in_the_arithmetic_rather_than_a_zero():
    """Every score identical: the sum must not print a window of width zero."""
    objects = _objects()
    objects["pred"] = 0.5
    plan = select_montage(objects, _counts(), "GRA14", 0.2, half_widths=1.0)
    arithmetic = plan.arithmetic()
    assert "every object scores the same" in arithmetic
    assert "degenerate" in arithmetic
    assert "window   = target +/- 1 x scale" not in arithmetic


def test_the_arithmetic_elides_a_long_well_list_and_says_how_many():
    """A gene spanning dozens of wells must not print a 200-line caption --
    and must not trail off either, so the elision carries its own count."""
    wells = tuple(f"r1_c{i}" for i in range(1, 61))
    fractions = {well: {"GRA14_1": 0.5, "OTHER_1": 0.5} for well in wells}
    plan = select_montage(_objects(wells=wells),
                          _counts(fractions=fractions), "GRA14", 0.2,
                          cap=MAX_OBJECTS)
    arithmetic = plan.arithmetic()
    assert arithmetic.count(" -> ") == MontagePlan.WELL_LINES
    assert f"... and {60 - MontagePlan.WELL_LINES} more wells" in arithmetic
    assert "objects shown across 60 wells" in arithmetic


# --------------------------------------------------------------------------- #
#  Instruction 155 C -- the settings, written onto the montage
# --------------------------------------------------------------------------- #

def test_every_setting_is_on_the_montage_and_a_changed_one_announces_itself():
    """Each of these changes WHICH CELLS a reader is looking at."""
    default = select_montage(_objects(), _counts(), "GRA14", 0.2)
    line = default.settings_line()
    assert "settings (per screen, not per gene)" in line
    assert "window half-width 1 robust scale(s)" in line
    assert "baseline the screen median" in line
    assert "score column 'pred'" in line
    assert f"cap {MAX_OBJECTS:,} objects" in line
    assert "NON-DEFAULT" not in line

    changed = select_montage(_objects(), _counts(), "GRA14", 0.2,
                             half_widths=2.5, cap=5)
    line = changed.settings_line()
    assert "NON-DEFAULT SETTINGS ON THIS MONTAGE" in line
    assert "window half-width 2.5 instead of the default 1" in line
    assert "cap 5 objects instead of the default 2,000" in line
    assert "not comparable to one made with the defaults" in line
    assert changed.settings_line() in changed.caption()


def test_the_fitted_intercept_is_named_as_the_baseline_and_moves_the_window():
    """The other baseline the request asks for, and it is not called 'given'.

    A montage centred on the intercept while the caption says ``given`` names
    no source at all, and the two baselines select different cells -- so the
    name has to be the one the user chose.
    """
    objects = _objects()
    median = select_montage(objects, _counts(), "GRA14", 0.2)
    intercept = select_montage(objects, _counts(), "GRA14", 0.2,
                               baseline=0.9,
                               baseline_label="the model's fitted intercept")
    assert median.window.baseline_source == "screen_median"
    assert intercept.window.baseline_source == "the model's fitted intercept"
    assert intercept.window.target == pytest.approx(0.9 + 0.2)
    assert "baseline the model's fitted intercept" in intercept.settings_line()
    assert "NON-DEFAULT" in intercept.settings_line()
    # And it really is a different montage, not just a different sentence.
    assert list(intercept.objects["pred"]) != list(median.objects["pred"])
    # Unlabelled, it is still recorded as 'given' -- the old spelling stands.
    assert score_window(objects, 0.2, baseline=0.9).baseline_source == "given"


def test_the_score_column_is_a_setting_and_the_caption_names_it():
    objects = _objects()
    objects["prob_dead"] = 1.0 - objects["pred"]
    plan = select_montage(objects, _counts(), "GRA14", 0.2,
                          score_column="prob_dead")
    assert plan.score_column == "prob_dead"
    assert "score column 'prob_dead'" in plan.settings_line()
    assert "NON-DEFAULT" in plan.settings_line()
    assert "median per-object prob_dead" in plan.arithmetic()


# --------------------------------------------------------------------------- #
#  Instruction 155 E -- the two routes, and what each one needs
# --------------------------------------------------------------------------- #

def test_the_merged_route_with_a_mask_and_a_label_can_cut_either_shape(tmp_path):
    from spacr.cell_montage import montage_route_requirements

    root, db_path = _screen(tmp_path, with_png=False)
    choice = resolve_montage_crop_source(root, object_type="cell")
    objects = load_montage_objects(db_path)
    req = montage_route_requirements(choice, objects, object_type="cell",
                                     channels=(0, 1, 2))
    assert req.route == "merged-mask"
    assert req.shapes == ("object", "bbox")
    assert req.satisfied and not req.missing
    assert req.offers("object") and req.why_not("object") == ""


def test_a_route_with_only_coordinates_refuses_the_object_shaped_crop(tmp_path):
    """ROUTE 2's whole point: no mask, so bounding boxes ONLY.

    An object-shaped crop must not appear as a choice that silently does
    something else, so the shape is not offered and the refusal names the
    reason rather than the symptom.
    """
    from spacr.cell_montage import montage_route_requirements

    root, db_path = _screen(tmp_path, with_png=False)
    choice = resolve_montage_crop_source(root, object_type="cell")
    # A coordinate table: bounding boxes, and no object label anywhere.
    coordinates = load_montage_objects(db_path).drop(
        columns=[c for c in ("object_label", "label", "cell_id", "nucleus_id",
                             "pathogen_id", "cytoplasm_id")
                 if c in load_montage_objects(db_path).columns])
    for index, name in enumerate(("bbox-0", "bbox-1", "bbox-2", "bbox-3")):
        coordinates[name] = index * 4

    req = montage_route_requirements(choice, coordinates, object_type="cell",
                                     channels=(0, 1, 2))
    assert req.route == "merged-bbox"
    assert req.shapes == ("bbox",)
    assert not req.offers("object")
    assert "no mask array, only coordinates" in req.why_not("object")
    assert "bounding-box crops only" in req.describe()


def test_neither_an_object_id_nor_a_box_is_refused_up_front_by_name(tmp_path):
    from spacr.cell_montage import montage_route_requirements

    root, db_path = _screen(tmp_path, with_png=False)
    choice = resolve_montage_crop_source(root, object_type="cell")
    bare = load_montage_objects(db_path)[["prc", "pred"]]
    req = montage_route_requirements(choice, bare, object_type="cell",
                                     channels=(0, 1, 2))
    assert not req.satisfied
    assert any("object id or a bounding box" in m for m in req.missing)


def test_a_missing_channel_list_is_reported_as_a_missing_channel_list(tmp_path):
    """"a user missing a channel list is told THAT rather than being told
    there is no source" -- the request's own sentence."""
    root, db_path = _screen(tmp_path, with_png=False)
    objects = load_montage_objects(db_path)

    choice = resolve_montage_crop_source(root, object_type="cell",
                                         objects=objects)
    assert choice.available                       # NOT "there is no source"
    assumed = " ".join(choice.requirements.assumed)
    assert "no channel list" in assumed
    assert "records no png_dims and none was typed" in assumed
    assert "default planes" in assumed
    assert any("no channel list" in note
               for note in choice.requirement_notes())

    # Typed in, the assumption goes away.
    told = resolve_montage_crop_source(root, object_type="cell",
                                       objects=objects, channels=(0, 1, 2))
    assert told.requirements.assumed == ()


def test_the_png_route_offers_no_shape_choice_and_says_why(tmp_path):
    from spacr.cell_montage import montage_route_requirements

    root, db_path = _screen(tmp_path, with_png=True)
    choice = resolve_montage_crop_source(root, object_type="cell")
    req = montage_route_requirements(choice, load_montage_objects(db_path))
    assert req.route == "png"
    assert req.shapes == ()
    assert "cut when the run wrote them" in req.why_not("object")


def test_no_source_at_all_is_a_route_of_its_own_rather_than_an_exception(
        tmp_path):
    from spacr.cell_montage import montage_route_requirements

    bare = tmp_path / "nothing"
    bare.mkdir()
    choice = resolve_montage_crop_source(str(bare))
    assert not choice.available
    assert choice.requirements.route == "none"
    assert not choice.requirements.satisfied
    assert montage_route_requirements(None).route == "none"
    assert "no route to pixels" in choice.requirements.describe()
