"""Round-6 coverage for the last cold corners of :mod:`spacr.ml`.

``spacr/ml.py`` sits at 99% and what is left is almost all defensive: guards
that re-ask a question a call two lines above has already answered, alias
tables that a canonicaliser has already emptied, and a manifest that is put on
a frame and then merged off it again. This file drives the handful that a real
input still reaches, and PINS THE INVARIANT behind each one that nothing can:
every "unreachable" claim below is written as a test of the thing that makes it
unreachable, driven from the same input in the same test, so the day the
invariant moves this file goes red instead of the guard quietly coming alive.

What is driven, and what is proved:

* ``_perform_regression``  -- two plates that already agree say so ("It changed
  nothing"), which is a different sentence from the one-plate case.
* ``generate_ml_scores``   -- ``dataset_mode='annotation'`` with no
  ``annotation_column`` refuses before it reads a label.
* ``interpret_vision_model`` -- the ``row``/``col`` alias block is dead because
  ``tabular.read_table`` canonicalises those spellings before it runs; a scores
  file with no well identity falls through all of it and fails at the join.
* ``process_reads`` / ``process_scores`` -- two guards asking about columns the
  statement above has just removed or just written.
* ``ml_analysis``          -- the cross-validation fold-count guard, which
  ``grouped_split`` has already refused every input for.
* ``regression`` -> ``_perform_regression`` -- the QC manifest rides on
  ``DataFrame.attrs`` and ``DataFrame.merge`` drops it, so ``output['qc']`` is
  never populated. Measured, not assumed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# spacr.ml reaches into spacr.utils lazily; warming it here keeps the one-off
# numba/umap import off whichever test happens to run first.
import spacr.utils  # noqa: E402,F401
from spacr import ml, schema  # noqa: E402

from tests.test_cov_ml_ml_scores import _feature_df, _make_src, _ml_settings  # noqa: E402
from tests.test_cov_ml_perform_regression import (  # noqa: E402
    COLS, ROWS, parametric_settings, write_counts, write_metadata)
from tests.test_predictions_merge import vision_settings  # noqa: E402
from tests.test_the_permutation_path_has_a_gene_pass import _long_table  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# a correction that could have moved the data, and did not
# ---------------------------------------------------------------------------

#: Per-cell offsets inside one well. They sum to zero and every one of them is
#: a multiple of 1/64, so a well's mean is its base value EXACTLY and a
#: plate's mean is 0.5 exactly. That exactness is the point: 'center'
#: subtracts the batch mean and adds the grand mean, and only when the two are
#: the same double does it put back the value it took away, bit for bit.
_JITTER = (1 / 64.0, -1 / 64.0, 1 / 32.0, -1 / 32.0)


def _agreeing_plate(path, plate):
    """A score CSV whose plate mean is exactly 0.5, with real well spread."""
    wells = [(row, col) for row in ROWS for col in COLS]
    deltas = []
    for step in range(1, len(wells) // 2 + 1):
        deltas += [step / 32.0, -step / 32.0]
    records = []
    for (row, col), delta in zip(wells, deltas):
        base = 0.5 + delta
        for offset in _JITTER:
            records.append({"plateID": plate, "rowID": row, "columnID": col,
                            "fieldID": "f1", "pred": base + offset,
                            "recruitment": 50.0})
    frame = pd.DataFrame(records)
    frame.to_csv(path, index=False)
    return str(path), float(frame["pred"].sum())


def test_two_plates_that_already_agree_are_told_apart_from_one_plate(
        tmp_path, capsys):
    """Zero shift across TWO batches is a finding, not the one-plate no-op.

    The one-plate sentence ("it changed nothing, and could not") is about the
    design; this one is about the screen, and a user comparing two runs needs
    to know which of the two they are looking at. Both plates carry the same
    18 well values, so the batch mean and the grand mean are the same double
    and `center` puts back exactly what it took -- mean |delta| is 0, not
    nearly 0.
    """
    scores = tmp_path / "scores"
    counts = tmp_path / "counts"
    scores.mkdir()
    counts.mkdir()
    score_1, total_1 = _agreeing_plate(scores / "plate1.csv", "plate1")
    score_2, total_2 = _agreeing_plate(scores / "plate2.csv", "plate2")
    # The two plates are the same measurement twice: this is what makes the
    # correction an exact no-op rather than a very small one.
    assert total_1 == total_2 == 36.0

    count_1 = write_counts(counts / "plate1.csv", plate="plate1", seed=1)
    count_2 = write_counts(counts / "plate2.csv", plate="plate2", seed=2)
    meta = write_metadata(tmp_path / "TGME49_Summary.csv")
    settings = parametric_settings(
        {"root": tmp_path, "score": score_1, "count": count_1, "meta": meta},
        batch_correction="center", batch_column="plateID")
    settings["score_data"] = [score_1, score_2]
    settings["count_data"] = [count_1, count_2]

    output = ml.perform_regression(settings)

    printed = capsys.readouterr().out
    assert "across 2 batch(es)" in printed
    assert "pred moved by 0 on average" in printed
    assert "It changed nothing: the batches already agree on pred" in printed
    # ...and NOT the one-plate sentence, which is about the design instead.
    assert "and could not" not in printed
    assert len(output["results"]) > 0

    # THE PENALISED-FIT NOTE CANNOT RAISE (ml.py:8927-8942), which is why the
    # bare `except: pass` under it never runs. Every expression in that block
    # is total for the values that reach it: `settings` is a dict whose
    # `regression_type` key was just read, `coef_df` is a DataFrame, and
    # `pd.to_numeric(..., errors='coerce')` COERCES a missing column to NaN
    # instead of refusing it -- driven here on the very frame the block reads.
    assert "p_value" in output["results"].columns
    absent = pd.to_numeric(output["results"].get("no_such_column"),
                           errors="coerce")
    assert np.isnan(absent)
    assert not (absent < 0.05).any()


# ---------------------------------------------------------------------------
# generate_ml_scores: an explicit basis with nothing to train on
# ---------------------------------------------------------------------------

def test_annotation_mode_without_an_annotation_column_refuses(tmp_path, rng):
    """``dataset_mode='annotation'`` is a promise the settings must keep.

    ``resolve_basis`` honours the explicit ``dataset_mode`` over the old
    implicit "an annotation_column means annotations" rule, so it is now
    possible to ask for annotation training and supply no column. Falling
    through to the metadata path would train on plate controls while the panel
    said annotations.
    """
    from spacr.training_basis import resolve_basis

    src = _make_src(tmp_path, "plate_no_annotation", rng, with_centroids=False)
    settings = _ml_settings(src, dataset_mode="annotation")
    assert settings.get("annotation_column") is None
    assert resolve_basis(settings) == "annotation"

    with pytest.raises(ValueError) as caught:
        ml.generate_ml_scores(settings)

    message = str(caught.value)
    assert "dataset_mode='annotation' needs annotation_column set" in message
    assert "png_list" in message


# ---------------------------------------------------------------------------
# interpret_vision_model: the alias block the canonicaliser emptied
# ---------------------------------------------------------------------------

def _flat_src(tmp_path, name):
    """A real non-timelapse ``measurements.db``, written by spaCR's own writer."""
    from spacr.utils import _merge_and_save_to_database

    src = str(tmp_path / name)
    os.makedirs(os.path.join(src, "measurements"), exist_ok=True)
    for field in (1, 2):
        morphology = pd.DataFrame({"label": [1, 2],
                                   "cell_area": [100.0 * field, 200.0 * field]})
        intensity = pd.DataFrame({
            "label": [1, 2],
            "cell_channel_0_mean_intensity": [10.0 * field, 20.0 * field]})
        _merge_and_save_to_database(morphology, intensity, "cell", src,
                                    f"plate1_A1_{field}", "exp")
    return src


def test_a_scores_file_spelled_row_and_col_is_canonical_without_aliases(
        tmp_path):
    """``row``/``col``/``row_name``/``column`` need no second alias block.

    ``read_and_preprocess_data`` reads through ``tabular.read_table``, whose
    vocabulary has already renamed every legacy spelling. The four later
    assignments therefore had no input that could reach them and are removed.

    Driven from both ends: the aliased file JOINS (so the spellings really are
    understood), and the frame the merge sees carries the canonical names
    only.
    """
    from spacr import tabular
    from spacr.schema import LEGACY_COLUMN_NAMES

    src = _flat_src(tmp_path, "aliased")
    scores = pd.DataFrame({
        "plate": ["plate1"] * 4,
        "row": ["r1"] * 4,
        "col": ["c1"] * 4,
        "field": ["f1", "f1", "f2", "f2"],
        "object": ["1", "2", "1", "2"],
        "cv_predictions": [0, 1, 0, 1]})
    path = tmp_path / "aliased_scores.csv"
    scores.to_csv(path, index=False)

    # The four names the dead block would have handled all resolve already.
    assert {LEGACY_COLUMN_NAMES[name] for name in ("row", "row_name")} == {"rowID"}
    assert {LEGACY_COLUMN_NAMES[name] for name in ("col", "column")} == {"columnID"}
    read_back = list(tabular.read_table(str(path)).columns)
    assert "rowID" in read_back and "columnID" in read_back
    assert "row" not in read_back and "col" not in read_back

    merged = ml.interpret_vision_model(vision_settings(src, path))

    assert len(merged) == 4
    assert merged["cv_predictions"].tolist() == [0, 1, 0, 1]

    import inspect
    source = inspect.getsource(ml.interpret_vision_model)
    for alias in ("row", "row_name", "col", "column"):
        assert f"if '{alias}' in scores_df.columns:" not in source
    assert "if 'rowID' not in scores_df.columns:" not in source
    assert "if 'columnID' not in scores_df.columns:" not in source


def test_a_scores_file_with_no_well_identity_falls_through_and_names_it(
        tmp_path):
    """No row and no column anywhere is refused by the join contract.

    It proves deleting the unreachable alias block did not turn a malformed
    file into a silent match: the join stops on the two columns it lacks.
    """
    src = _flat_src(tmp_path, "wellless")
    scores = pd.DataFrame({
        "plateID": ["plate1"] * 4,
        "fieldID": ["f1", "f1", "f2", "f2"],
        "object_label": ["1", "2", "1", "2"],
        "cv_predictions": [0, 1, 0, 1]})
    path = tmp_path / "wellless_scores.csv"
    scores.to_csv(path, index=False)

    with pytest.raises(KeyError) as caught:
        ml.interpret_vision_model(vision_settings(src, path))

    message = str(caught.value)
    assert "rowID" in message and "columnID" in message


def test_the_two_sides_of_the_timepoint_join_are_canonical_before_the_rename(
        tmp_path):
    """``time_id`` becomes ``timeID`` on BOTH sides, so the rename is dead.

    ml.py:11479 renames the scores' timepoint column when the two frames spell
    it differently. They cannot: the scores go through
    ``tabular.read_table`` and the measurement tables through
    ``io._read_db`` -> ``utils.correct_metadata``, and both map every legacy
    spelling onto ``timeID`` from the one vocabulary in ``spacr.schema``.
    """
    from spacr import tabular
    from spacr.utils import TIME_COLUMN_ALIASES, _time_column, correct_metadata

    legacy = pd.DataFrame({"plateID": ["plate1"], "time_id": ["t1"],
                           "object_label": ["1"]})
    path = tmp_path / "legacy_time.csv"
    legacy.to_csv(path, index=False)

    # written as time_id...
    assert _time_column(legacy.columns) == "time_id"
    assert "time_id" in TIME_COLUMN_ALIASES
    # ...and read back as timeID, by either reader.
    assert _time_column(tabular.read_table(str(path)).columns) == "timeID"
    assert _time_column(correct_metadata(legacy.copy()).columns) == "timeID"


# ---------------------------------------------------------------------------
# process_reads / process_scores: guards about columns just rewritten
# ---------------------------------------------------------------------------

def test_process_reads_rebuilds_gene_from_grna_even_when_gene_is_supplied(
        tmp_path):
    """A caller's ``gene`` column is dropped, so the split is always attempted.

    ml.py:9252 narrows the frame to ``['prc', 'grna', 'fraction']`` and the
    very next statement asks whether ``gene`` is present. It never is -- and
    that is not a bug, it is what makes the org/gene/guide split the single
    definition of a gene. Driven with a ``gene`` column that says something
    else entirely, so the answer cannot be a coincidence.
    """
    records = []
    for column in ("c1", "c2"):
        for guide in ("TGGT1_GENEA_g1", "TGGT1_GENEB_g1"):
            records.append({"plateID": "plate1", "rowID": "r1",
                            "columnID": column, "grna": guide,
                            "gene": "NOT_THE_GENE", "count": 100})

    result = ml.process_reads(pd.DataFrame(records), fraction_threshold=None,
                              plate=None)

    assert list(result.columns) == ["prc", "grna", "fraction", "gene"]
    assert sorted(result["gene"].unique()) == ["GENEA", "GENEB"]
    assert "NOT_THE_GENE" not in set(result["gene"])
    assert sorted(result["grna"].unique()) == ["GENEA_g1", "GENEB_g1"]

    import inspect
    source = inspect.getsource(ml.process_reads)
    assert ("if not all(col in merged_df.columns for col in "
            "['grna', 'gene']):") not in source


def _prcfo_scores(n=12):
    """A per-object score frame keyed only by ``prcfo``, as png_list writes it."""
    return pd.DataFrame({
        "prcfo": [f"plate1_r1_c{1 + (i % 2)}_f1_o{i}" for i in range(n)],
        "pred": np.linspace(0.1, 0.9, n)})


def test_process_scores_has_the_well_columns_without_a_second_check():
    """``_assign_prcfo_parts`` assigns every well column, or raises.

    ml.py:9428-9430 asks for ``plateID``/``rowID``/``columnID``, calls the
    helper when they are missing, and then asks again. The second question has
    one answer: the helper writes every name in ``schema.FIELD_KEY_COLUMNS``
    (ml.py:9098-9099) or raises before it writes any. Were the re-check ever
    False, ``prc`` would go unassigned and the next statement would raise
    ``KeyError('prc')`` -- so the driven half of this test is a frame that
    owns nothing but ``prcfo`` coming back with a ``prc``.
    """
    assert {"plateID", "rowID", "columnID"}.issubset(
        set(schema.FIELD_KEY_COLUMNS))

    bare = _prcfo_scores()
    assert not {"plateID", "rowID", "columnID"} & set(bare.columns)
    aggregated, name = ml.process_scores(bare.copy(), "pred", plate=None,
                                         min_cell_count=1, agg_type="mean")
    assert name == "pred"
    assert sorted(aggregated["prc"]) == ["plate1_r1_c1", "plate1_r1_c2"]

    # the helper's own half of the contract, on the same key shape
    filled = ml._assign_prcfo_parts(_prcfo_scores())      # private: the guard
    assert set(schema.FIELD_KEY_COLUMNS).issubset(set(filled.columns))
    # ...and a key it cannot parse raises rather than returning a frame that
    # would make the re-check False.
    with pytest.raises(ValueError, match="token"):
        ml._assign_prcfo_parts(pd.DataFrame({"prcfo": ["plate1_r1_c1"]}))

    import inspect
    source = inspect.getsource(ml.process_scores)
    check = ("if all(col in df.columns for col in "
             "['plateID', 'rowID', 'columnID']):")
    assert source.count(check) == 1


def test_process_scores_never_returns_a_prcfo_column_to_drop():
    """The ``prcfo`` drop at ml.py:9515 cannot survive into a result.

    Two statements above it the frame is narrowed to ``['prc',
    dependent_variable]``, so ``prcfo`` is a column of the aggregate only when
    the dependent variable IS ``prcfo`` -- and that run drops the column and
    then asks for it back, three statements later.
    """
    aggregated, _name = ml.process_scores(_prcfo_scores(), "pred", plate=None,
                                          min_cell_count=1, agg_type=None)
    assert "prcfo" not in aggregated.columns
    assert {"prc", "pred", "cell_count"}.issubset(set(aggregated.columns))

    # the one input that puts 'prcfo' back on the aggregate -- and cannot finish
    with pytest.raises(KeyError, match="prcfo"):
        ml.process_scores(_prcfo_scores(), "prcfo", plate=None,
                          min_cell_count=1, agg_type=None)


# ---------------------------------------------------------------------------
# ml_analysis: the fold count a split has already refused
# ---------------------------------------------------------------------------

def _one_well_frame(per_class=20, wells=("c1",)):
    """Control rows labelled by their own column, laid out over ``wells``."""
    frame = _feature_df(per_class=per_class, loc_values=("c1", "c2"))
    frame = frame.rename(columns={"columnID": "label"})
    index = []
    for position in range(len(frame)):
        well = wells[position % len(wells)]
        index.append(f"plate1_r1_{well}_f1_o{position}")
    frame.index = index
    return frame


def test_cross_validation_is_never_handed_a_single_group():
    """``n_folds < 2`` is unreachable: ``grouped_split`` refused first.

    ml.py:10518-10523 recomputes the distinct group count from the SAME array
    the held-out split was just made from, and ``grouped_split`` raises on
    fewer than two groups (classifier_evaluation.py:367) before the fold count
    is ever taken. Both halves are driven here: one well raises the split's
    message, two wells go all the way through five... two folds.
    """
    common = dict(channel_of_interest=3, location_column="label", n_repeats=1,
                  n_jobs=1, remove_highly_correlated_features=False,
                  test_size=0.25, split_by="well")

    with pytest.raises(ValueError) as caught:
        ml.ml_analysis(_one_well_frame(wells=("c1",)),
                       positive_control="c2", negative_control="c1",
                       model_type="random_forest", n_estimators=5,
                       cross_validation=True, verbose=False, **common)
    message = str(caught.value)
    assert "well-grouped held-out split is impossible" in message
    # the message the fold-count guard would have printed is NOT this one
    assert "independent groups" not in message

    output, _figures = ml.ml_analysis(
        _one_well_frame(wells=("c1", "c2")),
        positive_control="c2", negative_control="c1",
        model_type="random_forest", n_estimators=5,
        cross_validation=True, verbose=False, **common)
    metrics = output[8]
    assert "accuracy" in metrics.index
    assert output[0]["predictions"].notna().all()

    import inspect
    assert "if n_folds < 2:" not in inspect.getsource(ml.ml_analysis)


# ---------------------------------------------------------------------------
# the QC manifest that never arrives
# ---------------------------------------------------------------------------

def test_the_qc_manifest_survives_the_level_annotation():
    """``output['qc']`` was unreachable, and this is why it now is not.

    ``regression`` puts the manifest on ``coef_df.attrs`` (ml.py:5058) and
    ``_perform_regression`` reads it back off ``coef_df`` (ml.py:8999). In
    between, ``_annotate_level_coefficients`` merges the guide and gene counts
    onto the table, and ``DataFrame.merge`` does not carry ``attrs`` -- so the
    key is gone by the time it is read and the whole ``if manifest:`` block is
    dead. Copy and concat DO carry it, which is what makes the merge the loss.

    This was reported as a live defect rather than pinned as desirable --
    instruction 115's verdict silently not arriving -- and is now fixed:
    `_annotate_level_coefficients` carries `.attrs` across the two merges.
    """
    manifest = {"verdict": "usable", "verdict_level": "warn"}
    coefficients = pd.DataFrame({
        "feature": ["fraction:grna[GENEA_g1]", "gene_fraction:gene[GENEA]"],
        "coefficient": [0.4, 0.2], "p_value": [0.01, 0.2]})
    coefficients.attrs["qc_manifest"] = manifest

    # `.attrs` is a plain dict, which is why the removed try/except had nothing
    # to catch: a str key into a dict does not raise.
    assert isinstance(coefficients.attrs, dict)
    assert coefficients.copy().attrs["qc_manifest"] == manifest
    assert (pd.concat([coefficients], ignore_index=True)
            .attrs["qc_manifest"] == manifest)

    n_grna = pd.DataFrame({"grna": ["GENEA_g1"], "n_grna": [3]})
    n_gene = pd.DataFrame({"gene": ["GENEA"], "n_gene": [9]})
    annotated = ml._annotate_level_coefficients(coefficients, n_grna, n_gene)

    assert annotated["n_grna"].notna().any(), "the merge really happened"
    # ...and the manifest is carried across it, which is what
    # `_perform_regression` reads to populate output['qc'].
    assert annotated.attrs.get("qc_manifest") == manifest

    import inspect
    source = inspect.getsource(ml.regression)
    start = source.index("if qc_manifest is not None and coef_df is not None:")
    end = source.index("return model, coef_df, regression_type", start)
    assert "except Exception:" not in source[start:end]


# ---------------------------------------------------------------------------
# the permutation path's level stamps
# ---------------------------------------------------------------------------

def test_the_permutation_level_schema_needs_no_rechecks():
    """The guide is stamped here; the gene analyser already stamps its rows.

    The guide analyser returns no level, so ml stamps ``grna`` unconditionally.
    The gene analyser always returns ``gene``, so a second conditional stamp
    had only an impossible true side.
    """
    from spacr.guide_permutation import (analyse_long_gene_table,
                                         analyse_long_guide_table)

    frame = _long_table(seed=11)
    genes = analyse_long_gene_table(frame, "pred", min_wells=[2],
                                    n_permutations=99, random_state=0)
    guides = analyse_long_guide_table(frame, "pred", min_wells=[2],
                                      n_permutations=99, random_state=0)

    assert len(genes) and len(guides)
    assert set(genes["level"]) == {"gene"}
    assert "level" not in guides.columns

    import inspect
    source = inspect.getsource(ml._run_guide_permutation_analysis)
    assert "if 'level' not in levelled.columns:" not in source
    assert "levelled['level'] = 'grna'" in source
    assert "if 'level' not in gene_rows.columns:" not in source


# ---------------------------------------------------------------------------
# a folder that is always there to make
# ---------------------------------------------------------------------------

def test_an_absolute_paths_parent_is_never_empty(tmp_path, monkeypatch):
    """The removed ``if folder/parent`` guards could never be False.

    Both spell the same thing -- ``os.path.dirname(os.path.abspath(path))`` --
    and ``abspath`` returns a rooted path for every input, so its dirname is
    at worst ``'/'``. The guard is only meaningful for a BARE ``dirname``,
    which is what these used to be.
    """
    import statsmodels.api as sm

    for candidate in ("", ".", "summary.txt", "a/b/summary.txt", "/"):
        assert os.path.dirname(os.path.abspath(candidate)) != ""

    # and the True side, driven: a relative name lands in a directory the
    # writer had to create.
    monkeypatch.chdir(tmp_path)
    model = sm.OLS(np.array([1.0, 2.0, 3.0, 4.5]),
                   sm.add_constant(np.array([1.0, 2.0, 3.0, 4.0]))).fit()
    written = ml.save_summary_to_file(model, "nested/deeper/summary.txt")

    assert written == "nested/deeper/summary.txt"
    assert os.path.isdir(tmp_path / "nested" / "deeper")
    assert "OLS Regression Results" in (
        tmp_path / "nested" / "deeper" / "summary.txt").read_text()

    import inspect
    assert "if folder:" not in inspect.getsource(ml.save_summary_to_file)
    assert "if parent:" not in inspect.getsource(ml.write_plot)


# ---------------------------------------------------------------------------
# a sentence that must not be able to break a run
# ---------------------------------------------------------------------------

def test_a_class_list_that_cannot_be_built_costs_only_the_sentence(capsys):
    """The untrained-class note is cosmetic, and it fails like one.

    ``ml_analysis`` prints which classes of ``location_column`` were scored by
    a model that never trained on them (ml.py:10336-10352). Building that list
    needs the column's values in a ``set``, so a column holding values pandas
    cannot hash -- a list, a set, an array -- makes ``.unique()`` raise. The
    fit is finished by then and the sentence is worth nothing beside it, so
    the note is dropped and the run returns.

    Driven from both ends in one test: ordinary labels PRINT the sentence
    (three classes, two trained on), and unhashable ones return the same
    shaped output with the sentence missing rather than a traceback.
    """
    common = dict(channel_of_interest=3, location_column="columnID",
                  n_repeats=1, n_jobs=1,
                  remove_highly_correlated_features=False, test_size=0.25,
                  split_by="cell")

    named, _figures = ml.ml_analysis(
        _feature_df(per_class=20), positive_control="c2",
        negative_control="c1", model_type="random_forest", n_estimators=5,
        verbose=False, **common)
    spoken = capsys.readouterr().out
    assert "1 class(es) of 'columnID' are not in the training set" in spoken
    assert "['c3']" in spoken

    unhashable = _feature_df(per_class=20)
    unhashable["columnID"] = pd.Series(
        [[value] for value in unhashable["columnID"]],
        index=unhashable.index, dtype=object)
    # pandas cannot put these in a set, which is what the note needs
    with pytest.raises(TypeError, match="unhashable"):
        set(unhashable["columnID"].unique())

    scored, _figures = ml.ml_analysis(
        unhashable, positive_control="['c2']", negative_control="['c1']",
        model_type="random_forest", n_estimators=5, verbose=False, **common)
    silent = capsys.readouterr().out
    assert "are not in the training set" not in silent
    # the run itself is intact: the same 40 control rows, scored
    assert len(scored[0]) == len(unhashable)
    assert scored[0]["predictions"].notna().all()
