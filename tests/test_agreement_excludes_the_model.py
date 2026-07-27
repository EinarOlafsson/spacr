"""Inter-annotator agreement must not count the classifier as an annotator.

``png_list`` is the only table in spaCR that both the Annotate app and every
classifier write into. An annotation pass is an ``INTEGER`` column with two
distinct values; so is ``cv_predictions``. ``annotation_columns`` guessed by
shape, so it offered the model's own columns, and ``agreement_report`` then
answered a different question with the same units: how far the classifier is
from the people, reported as how far the people are from each other.

This is the same defect ``23e3a14`` fixed for ``timeID``/``prcft`` --- a
metadata column offered as an annotation --- one step further along, because
the classifier's column is not merely metadata, it is *another opinion*.

The database is built by the real writers: ``filepaths_to_database`` for
``png_list``, the Annotate app's own ``ALTER TABLE ... ADD COLUMN <name>
INTEGER`` for the human passes, and ``spacr.predictions.merge_cv_predictions``
/ ``merge_ml_predictions`` for the model's --- the calls
``spacr.deep_spacr.apply_model_to_tar`` and ``spacr.ml.ml_analysis`` make.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from spacr.agreement import (_MODEL_COLUMNS, _is_model_column,
                             agreement_report, annotation_columns)


WELLS = ('A1', 'A2')
FIELDS = (1, 2)
OBJECTS = (1, 2, 3)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def build_png_list(src, plate='plate1'):
    """``png_list`` written by :func:`spacr.utils.filepaths_to_database`."""
    from spacr.utils import filepaths_to_database

    os.makedirs(os.path.join(src, 'measurements'), exist_ok=True)
    folder = os.path.join(src, 'data', 'cell_png')
    os.makedirs(folder, exist_ok=True)
    paths = [os.path.join(folder, f'{plate}_{w}_{f}_{o}.png')
             for w in WELLS for f in FIELDS for o in OBJECTS]
    filepaths_to_database(paths, {'timelapse': False}, src, 'cell')
    return paths, os.path.join(src, 'measurements', 'measurements.db')


def annotate(db, column, labels):
    """Add one annotation pass exactly as the Annotate app does."""
    con = sqlite3.connect(db)
    try:
        con.execute(f'ALTER TABLE "png_list" ADD COLUMN "{column}" INTEGER')
        paths = [r[0] for r in con.execute('SELECT png_path FROM png_list')]
        for path, label in zip(paths, labels):
            con.execute(f'UPDATE "png_list" SET "{column}" = ? '
                        f'WHERE png_path = ?', (label, path))
        con.commit()
    finally:
        con.close()


def two_annotators_who_mostly_agree(db, paths):
    """Ann and Bob: identical except on every fourth crop."""
    ann = [1 if i % 3 else 2 for i in range(len(paths))]
    bob = [a if i % 4 else (2 if a == 1 else 1) for i, a in enumerate(ann)]
    annotate(db, 'annotator_ann', ann)
    annotate(db, 'annotator_bob', bob)
    return ann, bob


def score_with_the_cv_model(db, paths):
    """Merge a CV run through the real merge path."""
    from spacr.predictions import merge_cv_predictions

    scores = [0.1 + 0.05 * i for i in range(len(paths))]
    frame = pd.DataFrame({'path': [os.path.basename(p) for p in paths],
                          'pred': scores,
                          'cv_predictions': [1 if s < 0.5 else 2
                                             for s in scores]})
    report = merge_cv_predictions(frame, db, verbose=False)
    assert report is not None and report.matched_rows == len(paths)


def score_with_the_ml_model(db, paths):
    from spacr.predictions import merge_ml_predictions
    from spacr.utils import _map_wells_png

    prcfo = [_map_wells_png(os.path.basename(p))[4] for p in paths]
    probability = [0.05 * (i + 1) for i in range(len(paths))]
    frame = pd.DataFrame({
        'prcfo': prcfo,
        'predictions': [1 if p < 0.5 else 2 for p in probability],
        'prediction_probability_class_0': [1.0 - p for p in probability],
        'prediction_probability_class_1': probability,
    })
    report = merge_ml_predictions(frame, db, verbose=False)
    assert report is not None and report.matched_rows == len(paths)


@pytest.fixture()
def scored_database(tmp_path):
    paths, db = build_png_list(str(tmp_path))
    two_annotators_who_mostly_agree(db, paths)
    score_with_the_cv_model(db, paths)
    return db, paths


# ---------------------------------------------------------------------------
# the finding
# ---------------------------------------------------------------------------

def test_the_cv_models_columns_are_not_offered_as_annotators(scored_database):
    db, _ = scored_database

    con = sqlite3.connect(db)
    try:
        columns = [r[1] for r in con.execute('PRAGMA table_info("png_list")')]
    finally:
        con.close()
    # the model really did write into the same table
    assert {'pred', 'cv_predictions'} <= set(columns)

    assert annotation_columns(db) == ['annotator_ann', 'annotator_bob']


def test_the_ml_models_columns_are_not_offered_either(tmp_path):
    paths, db = build_png_list(str(tmp_path))
    two_annotators_who_mostly_agree(db, paths)
    score_with_the_ml_model(db, paths)

    con = sqlite3.connect(db)
    try:
        columns = [r[1] for r in con.execute('PRAGMA table_info("png_list")')]
    finally:
        con.close()
    assert {'predictions', 'ml_pred'} <= set(columns)

    assert annotation_columns(db) == ['annotator_ann', 'annotator_bob']


def test_including_the_model_changes_the_answer_not_just_the_column_list(
        scored_database):
    """The measured cost of the bug, not a description of it.

    Two people who agree moderately; add the classifier as a third
    "annotator" and the same database reports agreement no better than chance.
    """
    db, _ = scored_database

    humans = annotation_columns(db)
    with_model = annotation_columns(db, include_model_columns=True)
    assert humans == ['annotator_ann', 'annotator_bob']
    assert set(with_model) - set(humans) == {'pred', 'cv_predictions'}

    people = agreement_report(db, humans)
    everyone = agreement_report(db, with_model)

    assert people.overall_method == "Cohen's κ"
    assert people.overall_kappa == pytest.approx(0.47058823529411764)
    assert people.interpretation == 'moderate'

    assert everyone.overall_method == "Fleiss' κ"
    assert everyone.overall_kappa < 0.0
    assert everyone.interpretation == 'poor (no better than chance)'


def test_naming_a_model_column_explicitly_still_works_but_says_so(
        scored_database):
    """Model validation is a real question --- it just has to be labelled."""
    db, _ = scored_database

    report = agreement_report(db, ['annotator_ann', 'cv_predictions'])

    assert report.n_complete == 12
    assert any('written by a model' in w for w in report.warnings)
    assert any('cv_predictions' in w for w in report.warnings)


def test_a_report_over_humans_alone_carries_no_such_warning(scored_database):
    db, _ = scored_database
    report = agreement_report(db, ['annotator_ann', 'annotator_bob'])
    assert not any('written by a model' in w for w in report.warnings)


# ---------------------------------------------------------------------------
# the sampled-negatives column, and the guard against over-reach
# ---------------------------------------------------------------------------

def test_a_sampled_negatives_column_is_excluded_when_its_source_is_present():
    """``generate_training_dataset`` writes ``<col>_random`` beside ``<col>``.

    1 for the rows it drew as controls, NULL elsewhere. That is a sample, not
    a pass.
    """
    columns = ['png_path', 'annotate', 'annotate_random']
    assert _is_model_column('annotate_random', columns) is True


def test_a_random_named_annotator_survives_when_there_is_no_source_column():
    """The rule is precise, not a bare suffix match."""
    columns = ['png_path', 'blind_random', 'careful']
    assert _is_model_column('blind_random', columns) is False


def test_per_class_probability_columns_are_excluded_by_prefix():
    assert _is_model_column('prediction_probability_class_0') is True
    assert _is_model_column('prediction_probability_class_11') is True
    assert _is_model_column('probability_of_infection') is False


def test_an_ordinary_annotation_name_is_never_treated_as_a_model_column():
    for name in ('test', 'annotate', 'annotate_2', 'nc', 'pc', 'einar',
                 'scorer_1', 'predicted_by_hand'):
        assert _is_model_column(name, ['png_path', name]) is False, name


# ---------------------------------------------------------------------------
# drift guard: the list must keep up with the modules that write the columns
# ---------------------------------------------------------------------------

def test_every_column_the_prediction_merge_writes_is_listed():
    """``spacr.agreement`` duplicates these names to stay import-light.

    Duplication is only safe with a test that fails when the source of truth
    moves, so this reads the constants from the modules that do the writing.
    """
    from spacr import predictions as P

    for constant in (P.CV_SCORE_COLUMN, P.CV_CLASS_COLUMN,
                     P.ML_SCORE_COLUMN, P.ML_CLASS_COLUMN):
        assert constant in _MODEL_COLUMNS, constant


def test_every_column_the_active_learning_ranker_reads_is_listed():
    from spacr import active_learning as AL

    for candidate in AL.PRED_COLUMN_CANDIDATES:
        assert candidate in _MODEL_COLUMNS, candidate


def test_the_annotate_apps_xgboost_pass_is_listed():
    """``gui_elements`` writes these two with a literal ``ALTER TABLE``.

    Read out of the source rather than from a constant, because there is no
    constant --- which is exactly why it needs pinning here.
    """
    import inspect

    from spacr import gui_elements

    source = inspect.getsource(gui_elements)
    for column in ('XGboost_annotation', 'XGboost_score'):
        assert f'ALTER TABLE png_list ADD COLUMN {column}' in source, column
        assert column in _MODEL_COLUMNS, column
