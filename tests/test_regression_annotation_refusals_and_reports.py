"""What the cell-annotation strategies of ``spacr.regression_annotation`` refuse and report.

The module runs on whatever object table a screen produced, and that table's
failure modes are quiet ones: an empty or flat score column, a named well the
hold-out swallowed, a hold-out that came out one-class, a cluster or score
stratum nobody landed in, a labelling rate of zero. Each has one honest
answer -- a refusal naming the setting the user can change, or a note in the
run's report -- and the tests below walk tables of such shapes, each beside
an input that does not trigger it, so a guard that started refusing
everything fails here rather than passing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import classifier_evaluation as ce
from spacr import regression_annotation as ra


def _screen(*, wells: int = 8, per_well: int = 8, seed: int = 0,
            annotate: str = "none", separation: float = 300.0,
            flat: bool = False, tiny: int = 0) -> pd.DataFrame:
    """A small plate whose even-numbered cells per well are the phenotype.

    ``annotate`` is ``"none"``, ``"all"``, ``"some"`` (first three cells of
    each well) or ``"first4"`` (first four). ``separation`` of zero makes the
    classes indistinguishable; ``flat`` adds two columns that never vary;
    ``tiny`` adds a well ``r9_c9`` holding that many cells.
    """
    rng = np.random.default_rng(seed)
    written_below = {"some": 3, "first4": 4}

    def cell(row: str, column: str, index: int) -> dict:
        hit = index % 2 == 0
        out = {
            "plateID": "plate1", "rowID": row, "columnID": column,
            "fieldID": "f1", "prcfo": f"plate1_{row}_{column}_f1_o{index}",
            "cell_area": float(rng.normal(900 + separation * hit, 40)),
            "cell_channel_1_mean_intensity": float(
                rng.normal(1200 + separation * hit, 60)),
            "pred": float(rng.uniform(0.4, 0.6) + 0.3 * hit),
        }
        if annotate != "none":
            written = annotate == "all" or index < written_below[annotate]
            out["label"] = ("hit" if hit else "control") if written else ""
        if flat:
            out["flat_a"], out["flat_b"] = 7.0, 3.0
        return out

    rows = [cell(f"r{1 + well // 4}", f"c{1 + well % 4}", index)
            for well in range(wells) for index in range(per_well)]
    rows.extend(cell("r9", "c9", index) for index in range(tiny))
    return pd.DataFrame(rows)


def _request(frame: pd.DataFrame, **overrides) -> ra.AnnotationRequest:
    """A request over ``frame`` with every well eligible."""
    values = dict(frame=frame, score_column="pred", n_positive=4,
                  holdout_fraction=0.25, seed=1)
    values.update(overrides)
    return ra.AnnotationRequest(**values)


def _bare(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with every measurement column removed."""
    keep = ["plateID", "rowID", "columnID", "fieldID", "pred", "label"]
    return frame[[c for c in keep if c in frame.columns]].copy()


class _Constant:
    """An estimator stand-in that answers the same probability every time."""

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def fit(self, matrix, labels):
        return self

    def predict_proba(self, matrix):
        column = np.full(len(matrix), self.probability, dtype=float)
        return np.column_stack([1.0 - column, column])


def _refuse_with(error: Exception):
    """A ``grouped_split`` replacement that raises ``error``."""
    def refuse(*args, **kwargs):
        raise error
    return refuse


def _swallow_well(row: str, column: str):
    """The real splitter, forced to hold out exactly the well ``row_column``."""
    real_split = ce.grouped_split

    def split(*args, **kwargs):
        groups = {str(g) for g in np.asarray(args[0], dtype=object)}
        named = sorted(g for g in groups
                       if g.split(ra.GROUP_SEPARATOR)[-2:] == [row, column])
        assert len(named) == 1, named
        kwargs["hold_out_groups"] = named
        return real_split(*args, **kwargs)
    return split


def _positives_stay_in_training():
    """The real splitter, then every held-out positive moved to training."""
    real_split = ce.grouped_split

    def split(groups, labels, holdout, **kwargs):
        train, held, report = real_split(groups, labels, holdout, **kwargs)
        y = np.asarray(labels)
        kept = np.asarray([i for i in held if y[i] == 0], dtype=int)
        moved = np.asarray([i for i in held if y[i] == 1], dtype=int)
        return (np.sort(np.concatenate([np.asarray(train, dtype=int), moved])),
                kept, report)
    return split


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def test_feature_columns_keep_varying_measurements_or_exactly_the_named_ones(
        subtests):
    """Inferred features are varying numeric measurements; a named list is taken verbatim."""
    frame = _screen(wells=4, per_well=5)
    frame["cell_is_border"] = np.arange(len(frame)) % 2 == 0
    frame["cell_is_alive"] = True
    frame["cell_flat"] = 3.0
    frame["cell_probability"] = np.linspace(0.0, 1.0, len(frame))
    only_ids = frame[["plateID", "rowID", "columnID", "pred", "cell_flat"]]
    cases = [
        ("inferred: a varying boolean counts, constants and outputs do not",
         dict(frame=frame), ("cell_area", "cell_channel_1_mean_intensity",
                             "cell_is_border")),
        ("named: kept as given, constant included",
         dict(frame=frame, explicit=["cell_flat", "cell_area"]),
         ("cell_flat", "cell_area")),
        ("named: a single column", dict(frame=frame, explicit=["cell_area"]),
         ("cell_area",)),
        ("named: an absent column is a typo, not a filter",
         dict(frame=frame, explicit=["cell_no_such_column"]),
         "These feature columns are not in the object table: "
         "['cell_no_such_column']."),
        ("named: an empty list fits nothing", dict(frame=frame, explicit=[]),
         "An empty feature list was given, so there is nothing to fit on."),
        ("inferred: nothing survives", dict(frame=only_ids),
         "No measurement column survives"),
    ]
    for case_id, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            table = kwargs.pop("frame")
            if isinstance(expected, tuple):
                assert ra.feature_columns(table, "pred", **kwargs) == expected
            else:
                with pytest.raises(ra.AnnotationStrategyError) as caught:
                    ra.feature_columns(table, "pred", **kwargs)
                assert str(caught.value).startswith(expected)


def test_score_input_columns_name_the_score_once_and_skip_unjudgeable_columns(
        subtests):
    """Score inputs are the score plus strong rank correlates, or exactly the named inputs."""
    values = np.linspace(0.0, 1.0, 12)
    frame = pd.DataFrame({
        "cell_area": values * 1000.0,
        "cell_sparse": [1.0, 2.0] + [np.nan] * 10,
        "cell_inverse": -values,
        "cell_noise": np.tile([1.0, 5.0, 2.0, 4.0], 3),
        "pred": values,
        "cell_logit": np.tile([0.0, 1.0], 6),
    })
    no_score = frame.drop(columns=["pred"])
    cases = [
        ("correlation over every column", dict(frame=frame),
         ("cell_area", "cell_inverse", "pred", "cell_logit")),
        ("correlation over named features, score listed once",
         dict(frame=frame, features=["pred", "cell_area", "cell_sparse"]),
         ("cell_area", "pred")),
        ("a cut above every correlation keeps only the score",
         dict(frame=frame, features=["cell_area", "cell_noise"],
              correlation_cut=1.01),
         ("pred",)),
        ("explicit inputs plus the score", dict(frame=frame,
                                                explicit=["cell_area"]),
         ("cell_area", "pred")),
        ("explicit inputs drop absent names",
         dict(frame=frame, explicit=["cell_area", "cell_gone"]),
         ("cell_area", "pred")),
        ("explicit inputs with no score column", dict(frame=no_score,
                                                      explicit=["cell_area"]),
         ("cell_area",)),
        ("no score column and nothing named by name",
         dict(frame=no_score, features=["cell_area", "cell_noise"]), ()),
    ]
    for case_id, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            table = kwargs.pop("frame")
            assert ra.score_input_columns(table, "pred", **kwargs) == expected


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def test_prepare_refuses_a_table_it_cannot_score_or_split(subtests,
                                                          monkeypatch):
    """Each unusable table or split is refused with its own sentence and type."""
    nan_score = _screen()
    nan_score["pred"] = np.nan
    flat_score = _screen()
    flat_score["pred"] = 0.5
    no_identity = pd.DataFrame({"pred": np.linspace(0.05, 0.95, 12),
                                "cell_area": np.linspace(500, 1500, 12)})
    cases = [
        ("an all-missing score", _request(nan_score), None,
         ra.AnnotationStrategyError,
         "Every value of 'pred' is missing, so no cell has a score to "
         "select on."),
        ("a constant score", _request(flat_score), None,
         ra.AnnotationStrategyError,
         "A cut of the score at 0.5 puts every cell on one side, so there is "
         "no contrast to fit. The scores in the chosen wells run from 0.5 to "
         "0.5."),
        ("no plate identity", _request(no_identity), None,
         ra.AnnotationStrategyError,
         "Cannot split the object table by well: missing identity columns "
         "['plateID', 'rowID', 'columnID']"),
        ("a named well of one cell", _request(
            _screen(annotate="all", tiny=1), label_column="label",
            wells=["r9_c9"]), None, ra.AnnotationStrategyError,
         "The chosen wells have fewer than two cells left once the hold-out "
         "wells are taken out, so nothing can be selected from them. Name "
         "more wells, or lower holdout_fraction."),
        ("the hold-out swallows the only named well", _request(
            _screen(annotate="all"), label_column="label", wells=["r1_c1"]),
         _swallow_well("r1", "c1"), ra.AnnotationStrategyError,
         "The chosen wells have fewer than two cells left"),
        ("the splitter's typed refusal passes through unchanged",
         _request(_screen()),
         _refuse_with(ra.AnnotationStrategyError("the hold-out plate is empty")),
         ra.AnnotationStrategyError, "the hold-out plate is empty"),
        ("a NotEnoughLabels refusal keeps its subclass", _request(_screen()),
         _refuse_with(ra.NotEnoughLabels("one labelled well")),
         ra.NotEnoughLabels, "one labelled well"),
        ("a plain ValueError from the splitter becomes this module's type",
         _request(_screen()), _refuse_with(ValueError("bad design")),
         ra.AnnotationStrategyError, "bad design"),
    ]
    for size in (1, 2, 3):
        cases.append((f"{size} scored cell(s)", _request(
            _screen(wells=1, per_well=size)), None, ra.AnnotationStrategyError,
            f"Only {size} scored cell(s) are in the chosen wells, which is "
            "too few to define a top-scoring set."))
    for case_id, request, split, error, message in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            if split is not None:
                patch.setattr(ce, "grouped_split", split)
            with pytest.raises(error) as caught:
                ra.prepare(request)
            assert type(caught.value) is error
            assert str(caught.value).startswith(message), str(caught.value)

    # A rare phenotype split by cell can hold out no positive at all; over
    # eight seeds every hold-out is either two-class or refused in words.
    frame = _screen(per_well=10, seed=1).iloc[:20].copy()
    annotations = np.array(["miss"] * len(frame), dtype=object)
    annotations[:2] = "zhit"
    frame["ann"] = annotations
    outcomes = set()
    for seed in range(8):
        try:
            prepared = ra.prepare(_request(frame, label_column="ann",
                                           group_by="cell", seed=seed,
                                           n_positive=3))
        except ra.AnnotationStrategyError as refusal:
            outcomes.add(str(refusal))
        else:
            assert set(np.unique(prepared.holdout_labels())) == {0, 1}
            outcomes.add("scored")
    assert outcomes == {
        "scored",
        "The hold-out drew 5 cell(s) whose labelled cells are all of one "
        "class, so no accuracy measured there would mean anything. Change "
        "the seed, or raise holdout_fraction so more groups are held aside.",
    }


def test_prepare_holds_whole_wells_aside_and_labels_from_annotations_or_score(
        subtests):
    """The hold-out is whole wells nobody may select; labels come from four+ annotations, else a score cut."""
    with subtests.test(msg="a named well outside the hold-out"):
        kept = ra.prepare(_request(_screen(annotate="all"),
                                   label_column="label", wells=["r1_c1"]),
                          ra.TOP_SCORE_RANDOM)
        assert kept.chosen.size == 8
        assert not set(kept.chosen.tolist()) & set(kept.holdout.tolist())
        assert not set(kept.selectable.tolist()) & set(kept.holdout.tolist())
        assert np.array_equal(kept.holdout_labels(),
                              kept.labels[kept.holdout])
        assert kept.label_source == "annotations in 'label'"
        assert kept.positive_share(kept.holdout) == pytest.approx(0.5)
        assert np.isnan(kept.positive_share([]))

    for size in (2, 3, 4, 5):
        with subtests.test(msg=f"{size} annotations"):
            frame = _screen(wells=4, per_well=5)
            frame["ann"] = [None] * len(frame)
            for row in range(size):
                frame.loc[row, "ann"] = "pos" if row % 2 == 0 else "neg"
            labels, known, source, threshold, _ = ra._reference_labels(
                frame, _request(frame, label_column="ann").validated(),
                np.ones(len(frame), dtype=bool))
            if size < 4:
                assert source == "a cut on 'pred'"
                assert int(known.sum()) == len(frame)
                assert np.isfinite(threshold)
                assert int(labels.sum()) == 4
            else:
                assert source == "annotations in 'ann'"
                assert int(known.sum()) == size
                assert np.isnan(threshold)
                assert int(labels.sum()) == (size + 1) // 2

    with subtests.test(msg="a request with no measurements, for a strategy that fits nothing"):
        setup = ra.prepare(_request(_bare(_screen())), ra.SCORE_STRATA)
        assert setup.features == ()
        assert any(note.startswith(
            f"{ra.SCORE_STRATA.title} fits no model, and there is no "
            "measurement column in this table") for note in setup.notes)
        with pytest.raises(ra.AnnotationStrategyError,
                           match="^No measurement column survives"):
            ra.prepare(_request(_bare(_screen())))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def test_a_result_reports_and_writes_only_the_tables_it_has(tmp_path):
    """An empty selection has the role columns, no "Chosen" line and no CSV; a filled one has all three."""
    prepared = ra.prepare(_request(_screen(annotate="all"),
                                   label_column="label"), ra.UNCERTAINTY)
    empty = ra._selection_frame(prepared, {})
    assert list(empty.columns) == ["annotation_role", "annotation_group",
                                   "annotation_reference"]
    assert len(empty) == 0

    blank = ra.AnnotationResult(strategy="k", title="Nothing chosen",
                                selection=empty, holdout=empty)
    assert blank.summary() == "Nothing chosen\n"
    assert blank.role_counts() == {}
    written = blank.write(str(tmp_path / "blank"))
    assert set(written) == {"report"}
    assert (tmp_path / "blank" / "annotation_report.txt").read_text(
        encoding="utf-8") == "Nothing chosen\n\n"

    queue = prepared.selectable[:3]
    filled = ra._selection_frame(prepared, {"queue": queue[::-1]})
    assert filled.index.tolist() == prepared.frame.index[queue].tolist()
    assert filled["annotation_role"].tolist() == ["queue"] * 3
    assert filled["prcfo"].tolist() == prepared.frame["prcfo"].iloc[
        queue].tolist()
    assert filled["annotation_reference"].tolist() == prepared.labels[
        queue].tolist()
    holdout = ra._selection_frame(prepared, {"holdout": prepared.holdout})
    full = ra.AnnotationResult(strategy="k", title="Something chosen",
                               selection=filled, holdout=holdout,
                               notes=("a note",))
    n_held = int(prepared.holdout.size)
    assert full.role_counts() == {"holdout": n_held, "queue": 3}
    assert full.summary() == (f"Something chosen\n\nChosen: {n_held} holdout, "
                              "3 queue\n\n• a note")
    written = full.write(str(tmp_path / "full"))
    assert set(written) == {"selection", "holdout", "report"}
    assert len(pd.read_csv(written["holdout"])) == n_held
    assert len(pd.read_csv(written["selection"])) == 3

    only_holdout = ra.AnnotationResult(strategy="k", title="Held only",
                                       selection=empty, holdout=holdout)
    assert only_holdout.role_counts() == {"holdout": n_held}
    assert set(only_holdout.write(str(tmp_path / "held"))) == {"holdout",
                                                               "report"}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def test_a_strategy_refuses_a_setup_it_cannot_run_on(subtests, monkeypatch):
    """Every strategy-level refusal names its cause, with the exception type a caller branches on."""
    annotated = _request(_screen(annotate="all"), label_column="label")
    bare = _request(_bare(_screen(annotate="all")), label_column="label")
    single_score = _screen(annotate="all")
    single_score["pred"] = np.where(np.arange(len(single_score)) == 0, 0.5,
                                    np.nan)
    pu = _request(_screen(), seed=2)
    pu_zero = _request(_screen(), n_positive=8)

    def stub_estimator(model, seed):
        return _Constant(0.0), "stub"

    cases = [
        ("top-score: a named well of three cells", "top_score_random",
         _request(_screen(annotate="all", tiny=3), label_column="label",
                  wells=["r9_c9"]), None, {},
         ra.AnnotationStrategyError,
         "The chosen wells hold 3 selectable cell(s), which cannot supply 4 "
         "positives and the same number of contrast cells. Lower n_positive "
         "or name more wells."),
        ("uncertainty: every cell is already annotated", "uncertainty",
         annotated, None, {}, ra.AnnotationStrategyError,
         "Every selectable cell is already in the seed set, so there is "
         "nothing left to queue."),
        ("uncertainty: every feature is a score input", "uncertainty",
         _request(_screen(), leakage="drop", correlation_cut=0.0), None, {},
         ra.NotEnoughLabels,
         "No model could be fitted, so there is no decision boundary to be "
         "uncertain about."),
        ("diversity: a setup with no feature column", "diversity", bare,
         ra.SCORE_STRATA, {}, ra.AnnotationStrategyError,
         "There is no feature column to cluster on."),
        ("propagation: a setup with no feature column",
         "neighbour_propagation", bare, ra.SCORE_STRATA, {},
         ra.AnnotationStrategyError,
         "There is no feature column to measure a distance in."),
        ("two views: every cell is already annotated",
         "two_view_disagreement", annotated, None, {},
         ra.AnnotationStrategyError,
         "Every selectable cell is in the seed set, so there is nothing for "
         "two views to disagree about."),
        ("strata: one finite score", "score_strata",
         _request(single_score, label_column="label"), None, {},
         ra.AnnotationStrategyError,
         "Fewer than two selectable cells carry a finite score, so the score "
         "range cannot be divided."),
        ("PU: a labelling rate of zero", "pu_learning", pu_zero, None,
         {"estimator": stub_estimator}, ra.AnnotationStrategyError,
         "The estimated labelling rate is 0, which the rescaling cannot "
         "divide by."),
        ("PU: the inner split's typed refusal passes through", "pu_learning",
         pu, None, {"split": _refuse_with(
             ra.NotEnoughLabels("the inner split has one class"))},
         ra.NotEnoughLabels, "the inner split has one class"),
        ("PU: a plain inner-split refusal says what it was estimating",
         "pu_learning", pu, None, {"split": _refuse_with(
             ValueError("one class on the training side"))},
         ra.AnnotationStrategyError,
         "The labelling rate cannot be estimated on cells the model did not "
         "fit: one class on the training side"),
        ("PU: an inner split holding no positive", "pu_learning", pu, None,
         {"split": _positives_stay_in_training()}, ra.NotEnoughLabels,
         "The inner split left no labelled positive to estimate the "
         "labelling rate on, so the rescaling cannot be calibrated."),
    ]
    for case_id, key, request, entry, patches, error, message in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            setup = ra.prepare(request, entry or ra.strategy(key))
            if "split" in patches:
                patch.setattr(ce, "grouped_split", patches["split"])
            if "estimator" in patches:
                patch.setattr(ra, "_estimator", patches["estimator"])
            with pytest.raises(error) as caught:
                ra.run(key, request, setup)
            assert type(caught.value) is error
            assert str(caught.value).startswith(message), str(caught.value)
            if case_id.startswith("PU: a plain"):
                assert type(caught.value.__cause__) is ValueError


def test_self_training_says_why_it_stopped_and_keeps_its_rounds(subtests):
    """Each stop -- pool exhausted, nothing confident, rounds used up -- is one exact note."""
    cases = [
        ("confidence 0.5 labels the whole pool, then finds it empty",
         _request(_screen(seed=1), rounds=2, confidence=0.5),
         "stopped: every selectable cell already carries a label", 2),
        ("features without signal reach no 0.999 answer",
         _request(_screen(seed=0, separation=0.0), rounds=3,
                  confidence=0.999, seed=0),
         "stopped: no unlabelled cell reached the 1.00 confidence the round "
         "needs to accept its own answer", 1),
        ("one round asked for is one round run",
         _request(_screen(annotate="first4"), label_column="label", rounds=1,
                  confidence=0.5),
         "ran the full 1 round(s) without the audit turning down", 1),
    ]
    for case_id, request, stopped, rounds in cases:
        with subtests.test(msg=case_id):
            setup = ra.prepare(request, ra.SELF_TRAINING)
            result = ra.run("self_training", request, setup)
            notes = list(result.notes)
            assert stopped in notes
            curve = next(n for n in notes if n.startswith("Audit curve"))
            assert curve.count(";") == rounds - 1
            assert result.counts["rounds"] == rounds
            seeded = result.counts["seed_cells"]
            pseudo = result.counts["pseudo_labelled"]
            if request.confidence == 0.5:
                assert pseudo == setup.selectable.size - seeded
            else:
                assert pseudo == 0
            assert result.role_counts()["holdout"] == setup.holdout.size


def test_queueing_strategies_report_empty_clusters_strata_and_sampling(
        subtests, monkeypatch):
    """Degenerate pools still queue, and the report says what was empty or sampled."""
    with subtests.test(msg="strata: tied scores leave three of five empty"):
        frame = _screen()
        frame["pred"] = np.where(np.arange(len(frame)) % 2 == 0, 0.9, 0.1)
        result = ra.run("score_strata", _request(frame, n_bins=5,
                                                 n_positive=10))
        assert {k: v for k, v in result.counts.items()
                if k.startswith("stratum_")} == {
            "stratum_0": 0, "stratum_1": 0, "stratum_2": 24, "stratum_3": 0,
            "stratum_4": 24}
        assert (result.counts["strata"], result.counts["per_stratum"],
                result.counts["queued"]) == (5, 2, 4)
        assert ("3 stratum/strata are empty because ties in the score put "
                "many cells on one edge.") in result.notes

    with subtests.test(msg="strata: a varied score fills every stratum"):
        frame["pred"] = np.linspace(0.0, 1.0, len(frame))
        spread = ra.run("score_strata", _request(frame, n_bins=5,
                                                 n_positive=10))
        assert sum(v for k, v in spread.counts.items()
                   if k.startswith("stratum_")) == 48
        assert min(v for k, v in spread.counts.items()
                   if k.startswith("stratum_")) > 0
        assert spread.counts["queued"] == 10
        assert not any("stratum/strata are empty" in n for n in spread.notes)

    with subtests.test(msg="diversity: identical cells fill one of five clusters"):
        collapsed = _screen()
        collapsed["cell_area"] = 100.0
        result = ra.run("diversity", _request(
            collapsed, feature_columns=["cell_area"], n_clusters=5))
        assert {k: result.counts[k] for k in (
            "clusters", "queued", "smallest_cluster", "largest_cluster")} == {
            "clusters": 5, "queued": 1, "smallest_cluster": 48,
            "largest_cluster": 48}
        assert len(result.selection) == 1

    with subtests.test(msg="diversity: flat named columns, cluster count from n_clusters"):
        flat = _request(_screen(flat=True), feature_columns=["flat_a",
                                                             "flat_b"],
                        n_clusters=4)
        result = ra.run("diversity", flat)
        assert (result.counts["clusters"], result.counts["queued"]) == (4, 1)

    with subtests.test(msg="propagation: a pool over the ceiling is sampled and says so"):
        monkeypatch.setattr(ra, "MAX_POOL_FOR_DISTANCES", 10)
        sampled = ra.run("neighbour_propagation",
                         _request(_screen(), distance_quantile=0.5))
        assert ("48 selectable cells is more than the 10 this search builds "
                "a dense matrix over, so it ran on a random sample of that "
                "size. The sample is drawn with the run's seed, so it is the "
                "same sample every time.") in sampled.notes
        monkeypatch.setattr(ra, "MAX_POOL_FOR_DISTANCES", 50_000)
        whole = ra.run("neighbour_propagation",
                       _request(_screen(), distance_quantile=0.5))
        assert not any("dense matrix" in n for n in whole.notes)
        assert sampled.counts["seeds"] == whole.counts["seeds"] == 8
        assert sampled.counts["propagated"] > 0

    with subtests.test(msg="two views: a part-annotated screen queues n_positive"):
        result = ra.run("two_view_disagreement",
                        _request(_screen(annotate="some"),
                                 label_column="label"))
        assert {k: result.counts[k] for k in (
            "intensity_columns", "shape_columns", "queued")} == {
            "intensity_columns": 1, "shape_columns": 1, "queued": 4}

    with subtests.test(msg="PU: the working splitter estimates a rate"):
        request = _request(_screen(), seed=2)
        result = ra.run("pu_learning", request)
        assert 0.0 < result.counts["labelling_rate"] <= 1.0
        assert (result.counts["positives"],
                result.counts["unlabelled"]) == (4, 4)
        assert any("held-out positive(s)" in n for n in result.notes)
        assert result.predictions is not None
        assert set(result.predictions["predicted"].unique()) <= {0, 1}

    with subtests.test(msg="a fully annotated screen seeds from its labels"):
        request = _request(_screen(annotate="all"), label_column="label")
        setup = ra.prepare(request)
        train, _labels, notes = ra._seed_training(setup, request)
        assert np.array_equal(train, setup.labelled())
        assert notes == [f"The model is seeded from {train.size} annotated "
                         "cell(s), not from the score."]
