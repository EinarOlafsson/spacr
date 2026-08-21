"""The count is the normalised share; the choice is the top x by score.

Instruction 172, in the maintainer's words: "first calculate what all the
fractions in that well add up to if not 1 then normalize to 1 and save the
normalization factor then take the chosen grna fraction and apply that
normalization factor. that number multiplied with the number of classified
cells in that well is what should be shown, x cells. then for determining
which cells to show rank all cells by classefication score and take the top x
cells and show them or encircle them if the setting for show all is chosen."
"""
import numpy as np
import pandas as pd
import pytest

from spacr.cell_montage import (normalised_share, objects_to_show,
                                select_montage)


def _objects(n=100, well="plate1_r1_c1"):
    return pd.DataFrame({
        "prc": [well] * n,
        "object_label": range(n),
        "pred": np.linspace(0.0, 1.0, n),
    })


def _counts(fractions):
    return pd.DataFrame([
        {"prc": "plate1_r1_c1", "grna": g, "gene": g.split("_")[0],
         "fraction": f}
        for g, f in fractions.items()])


# ------------------------------------------------------------------ the count


def test_the_factor_is_one_when_the_fractions_already_sum_to_one():
    assert normalised_share([0.5, 0.3, 0.2], 0.5) == (0.5, 1.0)


def test_a_filtered_well_normalises_and_keeps_the_factor():
    """The maintainer's own example: 0.2 among fractions summing to 0.5."""
    share, factor = normalised_share([0.2, 0.2, 0.1], 0.2)

    assert share == pytest.approx(0.4)
    assert factor == pytest.approx(2.0)
    assert objects_to_show(100, share) == 40


def test_nothing_to_normalise_against_leaves_the_fraction_alone():
    """Inventing a factor here would be arithmetic on no evidence."""
    assert normalised_share([], 0.3) == (0.3, 1.0)
    assert normalised_share([0.0], 0.3) == (0.3, 1.0)


def test_a_share_above_one_is_capped_not_carried():
    """More than all of the well is a join that did not line up."""
    share, _factor = normalised_share([0.1], 0.9)

    assert share == 1.0


# ----------------------------------------------------------------- the choice


def test_it_takes_the_top_x_by_score_for_a_positive_effect():
    objects = _objects(n=100)
    counts = _counts({"GRA14_1": 0.2, "OTHER_1": 0.2, "OTHER_2": 0.1})

    plan = select_montage(objects, counts, "GRA14_1", 0.5)

    # 0.2 / 0.5 = 0.4 of 100 classified cells = 40
    assert len(plan.objects) == 40
    assert plan.objects["pred"].min() > objects["pred"].median()


def test_it_takes_the_bottom_x_for_a_negative_effect():
    objects = _objects(n=100)
    counts = _counts({"GRA14_1": 0.2, "OTHER_1": 0.2, "OTHER_2": 0.1})

    plan = select_montage(objects, counts, "GRA14_1", -0.5)

    assert len(plan.objects) == 40
    assert plan.objects["pred"].max() < objects["pred"].median()


def test_only_scored_cells_are_counted():
    """An object with no score cannot be ranked, so counting it would promise
    cells the ranking cannot deliver."""
    objects = _objects(n=100)
    objects.loc[objects.index[:50], "pred"] = np.nan
    counts = _counts({"GRA14_1": 0.2, "OTHER_1": 0.2, "OTHER_2": 0.1})

    plan = select_montage(objects, counts, "GRA14_1", 0.5)

    # 0.4 of the FIFTY that carry a score, not of the hundred rows.
    assert len(plan.objects) == 20


def test_the_caption_carries_the_arithmetic():
    """A reader must be able to redo it from what is on screen."""
    objects = _objects(n=100)
    counts = _counts({"GRA14_1": 0.2, "OTHER_1": 0.2, "OTHER_2": 0.1})

    plan = select_montage(objects, counts, "GRA14_1", 0.5)
    note = next(w.note for w in plan.wells if w.n_selected)

    assert "normalised by" in note
    assert "0.4" in note and "100" in note


# --------------------------------------------------------------- show all


def test_show_all_draws_the_well_and_marks_the_chosen():
    objects = _objects(n=100)
    counts = _counts({"GRA14_1": 0.2, "OTHER_1": 0.2, "OTHER_2": 0.1})

    plan = select_montage(objects, counts, "GRA14_1", 0.5, show_all=True)

    assert len(plan.objects) == 100, "the whole well is drawn"
    assert int(plan.objects["montage_candidate"].sum()) == 40


def test_the_filtered_view_marks_everything_it_shows():
    objects = _objects(n=100)
    counts = _counts({"GRA14_1": 0.2, "OTHER_1": 0.2, "OTHER_2": 0.1})

    plan = select_montage(objects, counts, "GRA14_1", 0.5)

    assert plan.objects["montage_candidate"].all()


def test_the_marked_cells_are_the_same_either_way():
    """Showing the well must not change WHICH cells are the candidates."""
    objects = _objects(n=100)
    counts = _counts({"GRA14_1": 0.2, "OTHER_1": 0.2, "OTHER_2": 0.1})

    filtered = select_montage(objects, counts, "GRA14_1", 0.5)
    everything = select_montage(objects, counts, "GRA14_1", 0.5, show_all=True)
    marked = everything.objects[everything.objects["montage_candidate"]]

    assert sorted(filtered.objects["object_label"]) == \
        sorted(marked["object_label"])


# ------------------------------------------------------ the three pickers


def test_the_pickers_are_named():
    """Four since instruction 173's option C, and `rank` stays the default.

    Pinned as a set plus a first element rather than an exact tuple: the
    order is what the settings window offers and a new picker belongs at the
    end, but which pickers EXIST is the contract.
    """
    from spacr.cell_montage import PICKING_MODES

    assert PICKING_MODES[0] == "rank"
    assert set(PICKING_MODES) == {"rank", "attributed", "assigned",
                                  "multivariate", "sudoku"}


def test_a_results_table_names_its_rows_as_the_DESIGN_did(tmp_path):
    """`fraction:grna[g1]`, not `g1` -- a lookup written for the bare name
    matches nothing, and nothing matched here is an attribution that silently
    falls back to the prior for every cell."""
    from spacr.cell_montage import effects_from_results

    csv = tmp_path / "results.csv"
    pd.DataFrame([
        {"feature": "Intercept", "coefficient": 0.5},
        {"feature": "fraction:grna[GRA14_1]", "coefficient": 1.75},
        {"feature": "C(rowID)[T.r2]", "coefficient": 0.1},
    ]).to_csv(csv, index=False)

    got = effects_from_results(str(csv))

    assert got["GRA14_1"] == pytest.approx(1.75)
    assert "Intercept" not in got
    assert got["r2"] == pytest.approx(0.1)


def test_assignment_gives_the_same_count_as_rank_but_not_the_same_cells():
    """The counts are both round(share x classified); WHICH cells differs,
    because the assignment lets another guide out-claim this one."""
    from spacr.cell_montage import select_montage as pick

    objects = _objects(n=200)
    counts = _counts({"GRA14_1": 0.3, "OTHER_1": 0.3, "OTHER_2": 0.4})
    # OTHER_1 points the same way and points HARDER, so under exclusion it
    # takes the very top cells and GRA14_1 gets the next ones down. Ranking
    # gives GRA14_1 the top cells regardless of who else wants them, which is
    # the failure the assignment exists to fix.
    effects = {"GRA14_1": 2.0, "OTHER_1": 4.0, "OTHER_2": -1.0}

    ranked = pick(objects, counts, "GRA14_1", 2.0, picking="rank")
    assigned = pick(objects, counts, "GRA14_1", 2.0, picking="assigned",
                    effects=effects)

    assert len(ranked.objects) == len(assigned.objects)
    ranked_ids = set(ranked.objects["object_label"])
    assigned_ids = set(assigned.objects["object_label"])
    assert ranked_ids != assigned_ids, (
        "exclusion changed nothing, so the constraint is not being used")


def test_an_unknown_picker_falls_back_to_rank_rather_than_failing():
    from spacr.cell_montage import select_montage as pick

    objects = _objects(n=50)
    counts = _counts({"GRA14_1": 0.5, "OTHER_1": 0.5})

    plan = pick(objects, counts, "GRA14_1", 1.0, picking="nonsense")

    assert len(plan.objects) > 0
