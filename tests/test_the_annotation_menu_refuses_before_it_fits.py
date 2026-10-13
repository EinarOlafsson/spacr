"""A strategy the loaded cells cannot support says so before it is run.

:mod:`spacr.regression_annotation` raises every one of its refusals from
inside a running strategy, which is the right place for the check and the
wrong moment for the person who chose it: a boosted tree over a screen is
seconds to minutes, and "there was never a measurement column to fit on"
is an answer that was knowable before any of them were spent.

These hold the pre-flight that answers it early, and the one consequence
of taking the menu's ``needs`` seriously: the two entries that fit nothing
run on a table with no measurement columns, so the unbiased random draw --
the measurement every other strategy is reported against -- is available
on a coefficient table that has nothing but scores joined to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import regression_annotation as ra


WELLS = tuple(f"r1_c{i}" for i in range(1, 7))


def _rows(n: int = 120, *, score: bool = True, measured: bool = True,
          seed: int = 0) -> pd.DataFrame:
    """A plate of objects: identity always, a score and measurements by ask."""
    rng = np.random.default_rng(seed)
    wells = [WELLS[i % len(WELLS)] for i in range(n)]
    frame = pd.DataFrame({
        "plateID": "plate1",
        "rowID": [w.split("_")[0] for w in wells],
        "columnID": [w.split("_")[1] for w in wells],
        "fieldID": "f1",
        "object_label": np.arange(n),
    })
    if measured:
        frame["cell_area"] = rng.random(n) * 100.0
        frame["cell_channel_1_mean_intensity"] = rng.random(n) * 50.0
        frame["cell_perimeter"] = rng.random(n) * 10.0
    if score:
        base = frame["cell_area"] if measured else pd.Series(rng.random(n))
        frame["pred"] = 1.0 / (1.0 + np.exp(
            -(np.asarray(base, dtype=float) / 30.0 - 1.5
              + rng.normal(0.0, 0.3, n))))
    return frame


# --------------------------------------------------------------------------- #
#  What the pre-flight answers
# --------------------------------------------------------------------------- #

def test_a_measured_screen_supports_every_strategy_but_the_anchored_one():
    """Nothing is refused on a table that carries scores and measurements.

    Control anchors is the exception and it is not about the table: its
    labels come from control wells nobody has named yet.
    """
    frame = _rows()
    for entry in ra.STRATEGIES:
        said = ra.missing_requirement(entry.key, frame, "pred")
        if entry.key == "control_anchors":
            assert "control well" in said
        else:
            assert said == "", f"{entry.key} was refused: {said}"


def test_with_no_score_and_no_annotations_every_entry_says_so():
    """Every strategy is measured against a reference label, so all refuse."""
    frame = _rows(score=False)
    for entry in ra.STRATEGIES:
        said = ra.missing_requirement(entry.key, frame, "pred")
        assert "nothing to label the cells with" in said, entry.key
        # NAMED, so a user knows which entry the sentence is about, and
        # told what would fix it.
        assert entry.title in said
        assert "annotation column" in said
        assert "'pred'" in said


def test_an_annotation_column_is_the_other_way_to_have_labels():
    """With annotations there is a reference label even with no score."""
    frame = _rows(score=False)
    frame["verdict"] = ["yes", "no"] * (len(frame) // 2)
    assert ra.missing_requirement("top_score_random", frame, "pred",
                                  label_column="verdict") == ""
    # One class is not two, and the run would fall back to the score it
    # does not have -- so the pre-flight refuses what the run refuses.
    frame["verdict"] = "yes"
    assert "nothing to label the cells with" in ra.missing_requirement(
        "top_score_random", frame, "pred", label_column="verdict")


def test_a_table_with_no_measurements_refuses_only_what_fits_on_them():
    """The fitting entries refuse; the two that fit nothing do not."""
    frame = _rows(measured=False)
    fits_nothing = {"score_strata", "random_holdout"}
    for entry in ra.STRATEGIES:
        said = ra.missing_requirement(entry.key, frame, "pred")
        if entry.key in fits_nothing:
            assert said == "", f"{entry.key} was refused: {said}"
        else:
            assert "measurement" in said, entry.key
            assert entry.title in said


def test_the_anchor_strategy_names_which_control_list_is_missing():
    frame = _rows()
    said = ra.missing_requirement("control_anchors", frame, "pred",
                                  positive_control_wells=("r1_c1",))
    assert "negative control well" in said
    assert "positive and" not in said
    assert ra.missing_requirement(
        "control_anchors", frame, "pred",
        positive_control_wells=("r1_c1",),
        negative_control_wells=("r1_c2",)) == ""


def test_an_empty_table_is_refused_rather_than_prepared():
    assert "no cells to choose from" in ra.missing_requirement(
        "random_holdout", _rows().iloc[:0], "pred")
    assert "no cells to choose from" in ra.missing_requirement(
        "random_holdout", None, "pred")


def test_the_pre_flight_reads_names_and_dtypes_rather_than_the_matrix():
    """It is asked on every change of a chooser, so it stays cheap.

    A column of text is not a measurement and a column that identifies the
    row is not either, whatever it is called -- both answers come out of
    ``dtypes`` and the identity list rather than out of the values.
    """
    frame = _rows(measured=False)
    frame["png_path"] = "somewhere.png"
    assert ra.candidate_feature_columns(frame, "pred") == ()
    frame["cell_area"] = 1.0
    assert ra.candidate_feature_columns(frame, "pred") == ("cell_area",)
    # And the score itself is never a candidate feature.
    assert "pred" not in ra.candidate_feature_columns(frame, "pred")


def test_the_pre_flight_is_the_optimistic_half_of_the_run():
    """A constant column passes the cheap check and the run still refuses.

    Stated as a test because it is the contract: an empty answer means
    nothing the cheap check can see is missing, never that the strategy
    is certain to run.
    """
    frame = _rows(measured=False)
    frame["cell_area"] = 1.0                       # numeric, and never varies
    assert ra.candidate_feature_columns(frame, "pred") == ("cell_area",)
    assert ra.missing_requirement("top_score_random", frame, "pred") == ""
    with pytest.raises(ra.AnnotationStrategyError, match="No measurement"):
        ra.run("top_score_random",
               ra.AnnotationRequest(frame=frame, score_column="pred",
                                    n_positive=10, holdout_fraction=0.34))


def test_an_unknown_key_is_refused_by_name():
    with pytest.raises(ra.AnnotationStrategyError, match="Unknown annotation"):
        ra.missing_requirement("moonlight", _rows(), "pred")


# --------------------------------------------------------------------------- #
#  The random draw is available whatever the table carries
# --------------------------------------------------------------------------- #

def _request(frame, **kwargs):
    return ra.AnnotationRequest(frame=frame, score_column="pred",
                                n_positive=10, holdout_fraction=0.34,
                                seed=0, **kwargs)


def test_the_plain_random_draw_runs_on_a_table_with_no_measurements():
    """Rule nine: the unbiased sample is available whatever else is chosen.

    A coefficient table with no measurement database attached is the common
    case, and it is exactly the table on which the only honest measurement
    must still be obtainable.
    """
    frame = _rows(measured=False)
    result = ra.run("random_holdout", _request(frame))
    assert result.fit is None
    assert len(result.holdout) > 0
    summary = result.summary()
    assert "fits no model" in summary
    assert "Nothing was fitted" in summary
    # No stray double stop where the refusal was pasted into the sentence.
    assert "fit on.." not in summary
    # And it still says what it did to the class balance, which is the
    # whole reason the draw exists.
    assert "above the positive cut" in summary


def test_the_score_strata_run_on_the_same_table():
    frame = _rows(measured=False)
    result = ra.run("score_strata", _request(frame))
    assert result.counts["queued"] > 0
    assert "fits no model" in result.summary()


def test_a_fitting_strategy_still_refuses_that_table():
    frame = _rows(measured=False)
    for key in ("top_score_random", "diversity", "uncertainty",
                "neighbour_propagation"):
        with pytest.raises(ra.AnnotationStrategyError, match="No measurement"):
            ra.run(key, _request(frame))


def test_prepare_on_its_own_still_demands_everything():
    """A caller with no menu entry to hand gets the strict answer.

    ``prepare(request)`` is the whole setup with nothing said about what
    will be run on it, so it cannot know that a fit is not coming.
    """
    frame = _rows(measured=False)
    with pytest.raises(ra.AnnotationStrategyError, match="No measurement"):
        ra.prepare(_request(frame))
    prepared = ra.prepare(_request(frame), ra.RANDOM_HOLDOUT)
    assert prepared.features == ()
    assert prepared.honest_features == ()


def test_the_hold_out_is_still_whole_wells_when_nothing_is_fitted():
    """The split rule does not lapse because the strategy fits nothing."""
    frame = _rows(measured=False)
    result = ra.run("random_holdout", _request(frame))
    assert not (set(result.selection["annotation_group"])
                & set(result.holdout["annotation_group"]))
    assert "no group appears on both sides" in result.summary()
