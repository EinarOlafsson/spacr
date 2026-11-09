"""Score profiles by mechanism of action, leaving one compound out.

A profiling result is comparable with the published BBBC021 numbers
(Ljosa et al. 2013) only when it is scored the same way. This module is
spaCR's one scorer for that: every arm holds out a whole compound, so a
treatment is never matched to, or trained on, the same compound at another
concentration.

* :func:`loco_nearest_neighbour` is the unsupervised arm: one nearest
  neighbour (1-NN) by cosine similarity between treatment profiles, under
  NSC (not same compound) and NSCB (not same compound or batch). It is
  :func:`spacr.embeddings.evaluate_moa`, so both give the same numbers.
* :func:`loco_supervised` trains a classifier on the wells of the other
  compounds and predicts the held-out compound's wells: a standardised
  logistic regression or XGBoost, on the CPU. Under NSC there is one fold
  per compound; under NSCB one per treatment, and the training wells also
  leave out every batch that treatment was imaged in.
* :func:`shuffled_label_control` repeats any arm with the compound-to-MoA
  mapping shuffled, which gives the accuracy of chance;
  :func:`majority_class_control` gives the accuracy of always guessing the
  commonest MoA.
* :func:`moa_benchmark` runs the arms and the controls on one table and
  returns one accuracy table with a glossary; :func:`write_moa_benchmark`
  writes them through :func:`spacr.tabular.write_table`.
* :func:`run_moa_benchmark` does all of it from settings, and
  ``python -m spacr.moa_benchmark`` from the command line.

Input is a table of per-well profiles (normalised to the plate's control
wells, one row per well) or of per-treatment profiles (one row per
treatment), with a MoA column, a compound column, the columns that name a
treatment and, for NSCB, a batch column. Per-well input is combined per
treatment by ``replicates`` (the median, as Ljosa et al. 2013) for the 1-NN
arm, and used as it is by the supervised arms.

A profiling run passes its feature-selected well profiles::

    from spacr.moa_benchmark import moa_benchmark, write_moa_benchmark
    result = moa_benchmark(
        "profiles/well_profiles_feature_select.csv",
        moa_column="Metadata_moa", compound_column="Metadata_compound",
        treatment_columns=("Metadata_compound", "Metadata_concentration"),
        batch_column="Metadata_batch")
    write_moa_benchmark(result, "profiles/moa_benchmark")

The CellProfiler head-to-head (item 753) scores each feature set with the
same call, one set at a time, and puts the tables side by side; with
``arms=("1-NN",)`` it gives the rows :func:`spacr.embeddings.evaluate_moa`
gives, with the shuffled-label and majority-class controls added.

Everything runs on the CPU; ``workers`` fits that many folds at a time in
threads of one process.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Key the settings of this module are registered under.
APP_KEY = 'moa_benchmark'

#: The arms :func:`moa_benchmark` can run.
ARMS = ('1-NN', 'logistic', 'xgboost')

#: The supervised arms of :func:`loco_supervised`.
SUPERVISED_MODELS = ('logistic', 'xgboost')

#: How replicate wells combine into a treatment profile for the 1-NN arm.
REPLICATE_METHODS = ('median', 'mean')

#: XGBoost's fixed hyperparameters: no tuning on test folds.
XGBOOST_PARAMS: Dict[str, Any] = dict(
    n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.3, min_child_weight=1, reg_lambda=1.0,
    tree_method='exact', eval_metric='mlogloss')

#: Inverse L2 penalty of the logistic regression.
LOGISTIC_C = 1.0

_ARM_LABELS = {'1-NN': '1-NN cosine', 'logistic': 'logistic regression',
               'xgboost': 'XGBoost'}

#: What each abbreviation and column of the accuracy table means.
GLOSSARY: Dict[str, str] = {
    'MoA': 'mechanism of action',
    '1-NN': 'one nearest neighbour: a treatment takes the MoA of its most similar other '
            'treatment (cosine similarity)',
    'LOCO': 'leave one compound out: a held-out compound is never matched to, or trained '
            'on, itself at another concentration',
    'NSC': 'not same compound: a treatment\'s neighbour, or its training wells, may not '
           'be the same compound',
    'NSCB': 'not same compound or batch: as NSC, and the neighbour, or the training wells, '
            'may not share a batch with the treatment',
    'CI': 'confidence interval',
    'arm': 'the method scored, or a control row: shuffled MoA labels (what chance gives) '
           'or the majority class (always guessing the commonest MoA)',
    'evaluation': 'NSC or NSCB',
    'n_treatments': 'treatments scored; under NSCB a treatment with no candidate or no '
                    'training data left is not scored',
    'correct': 'treatments whose predicted MoA is right',
    'accuracy': 'correct / n_treatments; for the shuffled control, the mean over the '
                'shuffles',
    'ci_low': 'lower end of the CI: Wilson score interval, or the lower percentile of the '
              'shuffles for the shuffled control',
    'ci_high': 'upper end of the CI, as ci_low',
    'sd': 'standard deviation of the shuffled-control accuracy over the shuffles',
    'n_solvable': 'scored treatments for which a right answer exists: a candidate, or a '
                  'training class, with the same MoA',
    'accuracy_solvable': 'correct / n_solvable',
    'well_accuracy': 'supervised arms: the share of held-out wells predicted right',
    'n_wells': 'supervised arms: the held-out wells scored',
    'n_permutations': 'shuffles of the compound-to-MoA mapping behind a shuffled control row',
}

#: How the supervised arms compare with the BBBC021 reanalysis; written to moa_summary.json.
SUPERVISED_REPRODUCTION: Dict[str, Any] = {
    'reference': 'supervised_moa.py, BBBC021 CellProfiler wells, Ljosa scaling, well means',
    'logistic': {'NSC': '96/103 (reference 96/103)', 'NSCB': '68/102 (reference 67/102)'},
    'xgboost': {'NSC': '91/103 (reference 88/103)', 'NSCB': '63/102 (reference 64/102)'},
    'causes': 'same folds and parameters; numerical differences, most likely library versions '
              '(scikit-learn, XGBoost, BLAS) and threading (the reference fitted 6 processes '
              'of 2 threads)',
    '1-NN': 'reproduces run_and_score.py exactly',
}

_TABLE_COLUMNS = ('arm', 'evaluation', 'n_treatments', 'correct', 'accuracy', 'ci_low',
                  'ci_high', 'sd', 'n_solvable', 'accuracy_solvable', 'well_accuracy',
                  'n_wells', 'n_permutations')

_COUNT_COLUMNS = ('n_cells', 'n_wells', 'n_replicates')

_DEFAULTS: Dict[str, Any] = {
    'moa_profiles': '',
    'moa_labels': '',
    'moa_label_column': 'moa',
    'moa_compound_column': 'compound',
    'moa_treatment_columns': 'compound,concentration',
    'moa_batch_column': '',
    'moa_controls': 'DMSO',
    'moa_arms': '1-NN,logistic,xgboost',
    'moa_replicates': 'median',
    'moa_shuffles': 100,
    'moa_supervised_shuffles': 10,
    'moa_seed': 0,
    'moa_workers': 1,
    'moa_dst': '',
}

_EXPECTED_TYPES: Dict[str, Any] = {
    key: type(value) for key, value in _DEFAULTS.items()
}

_TOOLTIPS: Dict[str, str] = {
    'moa_profiles': "(str) - Table of profiles to score: per-well profiles normalised to the plate's control wells, such as a profiling run's well_profiles_feature_select.csv, or one profile per treatment. CSV, TSV, Excel or Parquet. Default blank.",
    'moa_labels': "(str) - Table with the MoA of each compound or treatment, such as BBBC021_v1_moa.csv, joined to the profiles on the columns they share. Blank when the profiles carry a MoA column. Default blank.",
    'moa_label_column': "(str) - Column with the mechanism of action (MoA). Rows without one are not scored. Default 'moa'.",
    'moa_compound_column': "(str) - Column that names the compound. Every arm leaves a whole compound out, so a treatment is never matched to, or trained on, the same compound at another concentration. Default 'compound'.",
    'moa_treatment_columns': "(str) - Comma-separated columns that name one treatment, such as compound,concentration. Replicate wells of a treatment are combined into one profile for the nearest-neighbour arm. Default 'compound,concentration'.",
    'moa_batch_column': "(str) - Column that names each well's batch, such as the imaging week. Set it to score NSCB (not same compound or batch) beside NSC; blank scores NSC only. Default blank.",
    'moa_controls': "(str) - Comma-separated compound or MoA values left out of the scoring, such as the DMSO control wells. Default 'DMSO'.",
    'moa_arms': "(str) - Comma-separated arms to run: 1-NN (cosine nearest neighbour between treatment profiles), logistic (standardised logistic regression) and xgboost (XGBoost on the CPU). The supervised arms fit one model per held-out compound and are much slower. Default '1-NN,logistic,xgboost'.",
    'moa_replicates': "(str) - How replicate wells combine into a treatment profile for the 1-NN arm: 'median' (Ljosa et al. 2013) or 'mean'. Default 'median'.",
    'moa_shuffles': "(int) - Shuffles of the compound-to-MoA mapping for the 1-NN arm's control row, which shows the accuracy chance alone gives. 0 writes no control. Default 100.",
    'moa_supervised_shuffles': "(int) - Shuffles for each supervised arm's control row. Each shuffle refits every fold, so keep it small. 0 writes no control. Default 10.",
    'moa_seed': "(int) - Random seed for the shuffles and the XGBoost fits, so a rerun gives the same table. Default 0.",
    'moa_workers': "(int) - Folds fitted at a time by the supervised arms, in threads of one process. Default 1.",
    'moa_dst': "(str) - Folder the accuracy table, the predictions, the glossary and a summary are written to. Blank writes to a moa_benchmark folder beside the profiles table. Default blank.",
}

_CATEGORIES: Dict[str, List[str]] = {
    'MoA benchmark': ['moa_profiles', 'moa_labels', 'moa_label_column',
                      'moa_compound_column', 'moa_treatment_columns', 'moa_batch_column',
                      'moa_controls', 'moa_dst'],
    'MoA benchmark arms': ['moa_arms', 'moa_replicates', 'moa_shuffles',
                           'moa_supervised_shuffles', 'moa_seed', 'moa_workers'],
}

_DESCRIPTION = (
    "Score profiles by mechanism of action, leaving one compound out: cosine "
    "nearest neighbour, logistic regression and XGBoost under NSC and NSCB, "
    "with shuffled-label and majority-class controls."
)


def moa_benchmark_defaults(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`run_moa_benchmark`, filled in with their defaults.

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every key of this module.
    """
    merged = dict(settings or {})
    for key, value in _DEFAULTS.items():
        merged.setdefault(key, value)
    return merged


def _register() -> None:
    """Register this module's settings through the defaults seam, once per process."""
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY):
        return
    register_defaults(APP_KEY, moa_benchmark_defaults,
                      expected_types=_EXPECTED_TYPES, tooltips=_TOOLTIPS,
                      categories=_CATEGORIES, description=_DESCRIPTION)


_register()


def _names(value: Any) -> List[str]:
    """A list of names from a comma-separated string or a sequence."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(',') if part.strip()]
    return [str(part) for part in value]


def _attach_labels(frame: pd.DataFrame, labels: Any, moa_column: str) -> pd.DataFrame:
    """Join a label table to the profiles on the columns they share."""
    from .embeddings import _profile_table

    if labels is None or moa_column in frame.columns:
        return frame
    table = _profile_table(labels)
    if moa_column not in table.columns:
        raise ValueError(f"the labels have no {moa_column!r} column")
    shared = [column for column in table.columns
              if column in frame.columns and column != moa_column]
    if not shared:
        raise ValueError('the labels share no column with the profiles')
    joined = table[shared + [moa_column]].drop_duplicates(shared)
    return frame.merge(joined, on=shared, how='left')


def _choose_features(frame: pd.DataFrame, features: Optional[Sequence[str]],
                     keys: Sequence[str]) -> List[str]:
    """The profile columns: as given, or every numeric non-key column."""
    from pandas.api.types import is_numeric_dtype

    if features is not None:
        chosen = [str(column) for column in features]
        missing = [column for column in chosen if column not in frame.columns]
        if missing:
            raise ValueError(f'feature columns not in the table: {missing[:10]}')
        return chosen
    skip = set(keys) | set(_COUNT_COLUMNS)
    chosen = [column for column in frame.columns
              if column not in skip and not str(column).startswith('Metadata_')
              and is_numeric_dtype(frame[column])]
    if not chosen:
        raise ValueError('the table has no numeric feature columns')
    return chosen


def _prepare(data: Any, *, moa_column: str, compound_column: str,
             treatment_columns: Sequence[str], batch_column: Optional[str],
             features: Optional[Sequence[str]], labels: Any,
             controls: Any) -> Tuple[pd.DataFrame, List[str]]:
    """Read, label and filter the scored rows, and choose the features.

    :returns: the labelled rows without controls, and the feature columns.
    """
    from .embeddings import _profile_table

    frame = _attach_labels(_profile_table(data), labels, moa_column)
    keys = [moa_column, compound_column] + list(treatment_columns)
    keys += [batch_column] if batch_column else []
    missing = [column for column in dict.fromkeys(keys) if column not in frame.columns]
    if missing:
        raise ValueError(f'columns not in the profiles: {missing}')
    left_out = set(_names(controls))
    keep = frame[moa_column].notna()
    keep &= ~frame[moa_column].astype(str).isin(left_out)
    keep &= ~frame[compound_column].astype(str).isin(left_out)
    frame = frame[keep].reset_index(drop=True)
    if frame[compound_column].nunique() < 2:
        raise ValueError('fewer than two labelled compounds remain to score')
    chosen = _choose_features(frame, features, keys)
    values = frame[chosen].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f'the profiles hold {bad} missing or infinite values; '
                         'fill or drop them first')
    return frame, chosen


def _check_treatment_columns(treatment_columns: Sequence[str],
                             compound_column: str) -> List[str]:
    """The treatment columns, with the compound column among them."""
    columns = _names(treatment_columns)
    if compound_column not in columns:
        columns = [compound_column] + columns
    return columns


def _wilson(correct: int, total: int, confidence: float) -> Tuple[float, float]:
    """Wilson score interval at ``confidence``, from spacr.embeddings."""
    from statistics import NormalDist

    from .embeddings import _wilson as wilson

    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f'confidence must be between 0 and 1, not {confidence!r}')
    return wilson(correct, total, NormalDist().inv_cdf(0.5 + float(confidence) / 2.0))


def _row(arm: str, evaluation: str, **values: Any) -> Dict[str, Any]:
    """One accuracy-table row with every column present."""
    row = {column: np.nan for column in _TABLE_COLUMNS}
    row.update(arm=arm, evaluation=evaluation, **values)
    return row


def _table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """An accuracy table in the column order of :data:`GLOSSARY`."""
    return pd.DataFrame(list(rows), columns=list(_TABLE_COLUMNS))


def loco_nearest_neighbour(profiles: Any, *, moa_column: str = 'moa',
                           compound_column: str = 'compound',
                           treatment_columns: Sequence[str] = ('compound', 'concentration'),
                           batch_column: Optional[str] = None,
                           features: Optional[Sequence[str]] = None,
                           labels: Any = None, controls: Any = ('DMSO',),
                           confidence: float = 0.95) -> Dict[str, Any]:
    """Leave-one-compound-out 1-NN MoA accuracy on treatment profiles.

    Each treatment takes the MoA of its most similar other treatment by
    cosine similarity, never the same compound (NSC) and, with a batch
    column, never a treatment that shares a batch (NSCB). The scoring is
    :func:`spacr.embeddings.evaluate_moa`; this arm returns its rows in
    the table format of :func:`moa_benchmark`, without controls (see
    :func:`shuffled_label_control` and :func:`majority_class_control`).

    The CellProfiler head-to-head (item 753) and a profiling run call it
    on one profile per treatment::

        loco_nearest_neighbour(consensus, batch_column="batch")

    :param profiles: one profile per treatment, a frame or a table path.
    :param moa_column: column with the MoA; rows without one are not scored.
    :param compound_column: column that names the compound.
    :param treatment_columns: columns that name a treatment; never profile
        values.
    :param batch_column: column with each treatment's batches, joined by
        ``|`` when there are several; ``None`` scores NSC only.
    :param features: profile columns; ``None`` takes every numeric column
        outside the key columns, ``Metadata_`` columns and counts.
    :param labels: a frame or table path with the MoA, joined on the
        shared columns when the profiles carry none.
    :param controls: compound or MoA values left out, such as ``'DMSO'``.
    :param confidence: level of the Wilson score interval.
    :returns: ``{"table", "predictions"}``: one row per evaluation, and one
        row per treatment and evaluation with ``predicted_moa``,
        ``neighbour_compound``, ``similarity``, ``scored`` and ``correct``.
    :raises ValueError: for missing columns, missing or infinite values,
        or fewer than two labelled compounds.
    """
    from .embeddings import evaluate_moa

    treatment_columns = _check_treatment_columns(treatment_columns, compound_column)
    frame, chosen = _prepare(profiles, moa_column=moa_column, compound_column=compound_column,
                             treatment_columns=treatment_columns, batch_column=batch_column,
                             features=features, labels=labels, controls=controls)
    result = evaluate_moa(frame, moa_column=moa_column, compound_column=compound_column,
                          batch_column=batch_column, features=chosen,
                          treatment_columns=treatment_columns, n_permutations=0,
                          confidence=confidence)
    scored = result['table'][result['table']['profiles'] == 'profiles']
    rows = [_row(_ARM_LABELS['1-NN'], row['evaluation'],
                 **{column: row[column] for column in
                    ('n_treatments', 'correct', 'accuracy', 'ci_low', 'ci_high',
                     'n_solvable', 'accuracy_solvable')})
            for _, row in scored.iterrows()]
    wide = result['predictions']
    base = wide[[column for column in dict.fromkeys(
        treatment_columns + [moa_column] + ([batch_column] if batch_column else []))]]
    parts = []
    for rule in scored['evaluation']:
        part = base.copy()
        part['arm'] = _ARM_LABELS['1-NN']
        part['evaluation'] = rule
        part['predicted_moa'] = wide[f'{rule}_predicted_moa'].to_numpy()
        part['neighbour_compound'] = wide[f'{rule}_neighbour_compound'].to_numpy()
        part['similarity'] = wide[f'{rule}_similarity'].to_numpy()
        part['scored'] = wide[f'{rule}_neighbour'].to_numpy() >= 0
        part['correct'] = wide[f'{rule}_correct'].to_numpy()
        parts.append(part)
    return {'table': _table(rows), 'predictions': pd.concat(parts, ignore_index=True)}


def _make_model(model: str, seed: int) -> Any:
    """A fresh classifier for one fold."""
    if model == 'logistic':
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=LOGISTIC_C, max_iter=5000))
    from .ml import _xgboost_classifier, _xgboost_cpu_safe

    return _xgboost_classifier()(**_xgboost_cpu_safe(
        dict(XGBOOST_PARAMS, random_state=int(seed), n_jobs=1, device='cpu')))


def _fit_fold(job: Tuple[Any, ...]) -> Tuple[str, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Fit one fold and return the held-out class probabilities.

    :param job: ``(model, X, y, train, test, rule, seed)``.
    :returns: the rule, the held-out row indices, their probabilities (None
        when the training wells hold fewer than two classes) and the classes.
    """
    model, X, y, train, test, rule, seed = job
    classes = np.unique(y[train])
    index = np.flatnonzero(test)
    if len(classes) < 2:
        return rule, index, None, classes
    estimator = _make_model(model, seed)
    estimator.fit(X[train], np.searchsorted(classes, y[train]))
    return rule, index, estimator.predict_proba(X[test]), classes


def _batch_overlap(batches: Optional[pd.Series]) -> Optional[np.ndarray]:
    """Row-by-batch membership, from labels joined by ``|``."""
    if batches is None:
        return None
    from .embeddings import _batch_sets

    sets = _batch_sets(batches)
    names = sorted(set().union(*sets)) if sets else []
    column = {name: index for index, name in enumerate(names)}
    member = np.zeros((len(sets), max(len(names), 1)), dtype=bool)
    for row, found in enumerate(sets):
        member[row, [column[name] for name in found]] = True
    return member


def loco_supervised(wells: Any, *, model: str = 'logistic', moa_column: str = 'moa',
                    compound_column: str = 'compound',
                    treatment_columns: Sequence[str] = ('compound', 'concentration'),
                    batch_column: Optional[str] = None,
                    features: Optional[Sequence[str]] = None, labels: Any = None,
                    controls: Any = ('DMSO',), seed: int = 0, workers: int = 1,
                    confidence: float = 0.95) -> Dict[str, Any]:
    """Leave-one-compound-out supervised MoA accuracy on well profiles.

    Under NSC each compound is held out in turn (all its concentrations and
    replicate wells) and a classifier is trained on the wells of every
    other compound. Under NSCB each treatment is held out in turn, and the
    training wells also leave out every well imaged in a batch of that
    treatment (a compound imaged in every batch would otherwise leave
    nothing to train on). A treatment's prediction is the class with the
    highest mean probability over its wells. A fold whose training wells
    hold fewer than two classes is not scored.

    ``'logistic'`` standardises the features on the training fold and fits
    a multinomial L2 logistic regression (C = 1). ``'xgboost'`` fits
    XGBoost with :data:`XGBOOST_PARAMS` on the CPU
    (:func:`spacr.ml._xgboost_cpu_safe`). Neither is tuned on test folds.

    A profiling run calls it on its normalised well profiles::

        loco_supervised(wells, model="xgboost", batch_column="batch", workers=4)

    The supervised arms are close to, not identical with, the BBBC021
    reanalysis (``supervised_moa.py``, CellProfiler wells, Ljosa scaling,
    well means): logistic regression gives 96/103 under NSC (96 in the
    reference) and 68/102 under NSCB (67); XGBoost 91/103 under NSC (88)
    and 63/102 under NSCB (64). The folds and parameters are the same; the
    differences are numerical, most likely library versions (scikit-learn,
    XGBoost, BLAS) and threading (the reference fitted 6 processes of 2
    threads). The 1-NN arm reproduces ``run_and_score.py`` exactly.

    :param wells: one row per well (or per treatment), a frame or a table
        path. All-``float32`` features are fitted in ``float32``.
    :param model: ``'logistic'`` or ``'xgboost'``.
    :param moa_column: column with the MoA; rows without one are not scored.
    :param compound_column: column that names the compound.
    :param treatment_columns: columns that name a treatment.
    :param batch_column: column with each well's batch; ``None`` scores NSC
        only.
    :param features: profile columns; ``None`` takes every numeric column
        outside the key columns, ``Metadata_`` columns and counts.
    :param labels: a frame or table path with the MoA, joined on the shared
        columns when the wells carry none.
    :param controls: compound or MoA values left out, such as ``'DMSO'``.
    :param seed: seed of the XGBoost fits.
    :param workers: folds fitted at a time, in threads.
    :param confidence: level of the Wilson score interval.
    :returns: ``{"table", "predictions"}``: one row per evaluation (with
        the per-well accuracy), and one row per treatment and evaluation
        with ``predicted_moa``, ``probability``, ``n_wells``, ``scored``,
        ``solvable`` and ``correct``.
    :raises ValueError: for an unknown model, missing columns, missing or
        infinite values, or fewer than two labelled compounds.
    """
    model = str(model).strip().lower()
    if model not in SUPERVISED_MODELS:
        raise ValueError(f'model must be one of {SUPERVISED_MODELS}, not {model!r}')
    treatment_columns = _check_treatment_columns(treatment_columns, compound_column)
    frame, chosen = _prepare(wells, moa_column=moa_column, compound_column=compound_column,
                             treatment_columns=treatment_columns, batch_column=batch_column,
                             features=features, labels=labels, controls=controls)
    X = frame[chosen].to_numpy()
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(float)
    y = frame[moa_column].astype(str).to_numpy()
    compounds = frame[compound_column].astype(str).to_numpy()
    treatment = frame.groupby(treatment_columns, sort=True, dropna=False).ngroup().to_numpy()
    member = _batch_overlap(frame[batch_column] if batch_column else None)
    jobs = [(model, X, y, compounds != name, compounds == name, 'NSC', seed)
            for name in np.unique(compounds)]
    if member is not None:
        for code in np.unique(treatment):
            test = treatment == code
            shared = (member.astype(int) @ member[test].any(axis=0).astype(int)) > 0
            train = (compounds != compounds[test][0]) & ~shared
            jobs.append((model, X, y, train, test, 'NSCB', seed))
    with ThreadPoolExecutor(max(1, int(workers))) as pool:
        folds = list(pool.map(_fit_fold, jobs))
    rows, parts = [], []
    rules = ['NSC'] + (['NSCB'] if member is not None else [])
    for rule in rules:
        well_pred = np.full(len(y), None, dtype=object)
        records = []
        for fold_rule, index, proba, classes in folds:
            if fold_rule != rule:
                continue
            if proba is not None:
                well_pred[index] = classes[proba.argmax(axis=1)]
            for code in np.unique(treatment[index]):
                inside = treatment[index] == code
                first = index[inside][0]
                record = {column: frame.at[first, column] for column in treatment_columns}
                record.update({moa_column: y[first], 'arm': _ARM_LABELS[model],
                               'evaluation': rule, 'n_wells': int(inside.sum()),
                               'scored': proba is not None,
                               'solvable': bool(y[first] in set(classes))})
                if proba is not None:
                    mean = proba[inside].mean(axis=0)
                    record.update(predicted_moa=classes[mean.argmax()],
                                  probability=float(mean.max()))
                else:
                    record.update(predicted_moa=None, probability=np.nan)
                record['correct'] = bool(record['scored'] and record['predicted_moa'] == y[first])
                records.append(record)
        part = pd.DataFrame(records)
        parts.append(part)
        scored = part[part['scored']]
        total, correct = int(len(scored)), int(scored['correct'].sum())
        solvable = scored[scored['solvable']]
        low, high = _wilson(correct, total, confidence)
        evaluated = well_pred != None  # noqa: E711
        rows.append(_row(
            _ARM_LABELS[model], rule, n_treatments=total, correct=correct,
            accuracy=correct / total if total else np.nan, ci_low=low, ci_high=high,
            n_solvable=int(len(solvable)),
            accuracy_solvable=float(solvable['correct'].mean()) if len(solvable) else np.nan,
            well_accuracy=float((well_pred[evaluated] == y[evaluated]).mean())
            if evaluated.any() else np.nan,
            n_wells=int(evaluated.sum())))
    return {'table': _table(rows), 'predictions': pd.concat(parts, ignore_index=True)}


def _arm(arm: Union[str, Callable[..., Dict[str, Any]]]) -> Tuple[str, Callable[..., Dict[str, Any]], Dict[str, Any]]:
    """The label, the scoring function and its fixed options of one arm."""
    if callable(arm):
        return getattr(arm, '__name__', 'arm'), arm, {}
    name = str(arm).strip()
    lowered = name.lower()
    if lowered in ('1-nn', '1nn', 'knn', 'nearest_neighbour'):
        return _ARM_LABELS['1-NN'], loco_nearest_neighbour, {}
    if lowered in SUPERVISED_MODELS:
        return _ARM_LABELS[lowered], loco_supervised, {'model': lowered}
    raise ValueError(f'arm must be one of {ARMS} or a function, not {arm!r}')


def shuffled_label_control(arm: Union[str, Callable[..., Dict[str, Any]]], data: Any, *,
                           n_permutations: int = 100, seed: int = 0,
                           moa_column: str = 'moa', compound_column: str = 'compound',
                           treatment_columns: Sequence[str] = ('compound', 'concentration'),
                           batch_column: Optional[str] = None,
                           features: Optional[Sequence[str]] = None,
                           controls: Any = ('DMSO',), workers: int = 1,
                           confidence: float = 0.95) -> pd.DataFrame:
    """The accuracy one arm reaches with the MoA labels shuffled between compounds.

    The compound-to-MoA mapping is permuted (each compound keeps one MoA,
    so the class sizes stay as they are) and the arm is scored again, once
    per permutation; NSC and NSCB are scored on the same permutation. The
    mean is what chance alone gives: an arm that does not clear it has
    learned nothing about MoA. A compound with two MoA labels makes the
    shuffle permute the rows instead.

    :param arm: ``'1-NN'``, ``'logistic'``, ``'xgboost'``, or a function
        with the signature of :func:`loco_nearest_neighbour`.
    :param data: what the arm scores: treatment profiles for 1-NN, wells
        for the supervised arms. Must carry the MoA column.
    :param n_permutations: shuffles; ``0`` returns an empty table.
    :param seed: seed of the shuffles; permutation ``i`` fits supervised
        models with seed ``seed + i + 1``.
    :param moa_column: column with the MoA.
    :param compound_column: column that names the compound.
    :param treatment_columns: columns that name a treatment.
    :param batch_column: batch column; ``None`` scores NSC only.
    :param features: profile columns, as for the arm.
    :param controls: compound or MoA values left out before shuffling.
    :param workers: folds fitted at a time by a supervised arm.
    :param confidence: share of the shuffles inside ``ci_low`` to
        ``ci_high``.
    :returns: one control row per evaluation: the mean accuracy, its
        standard deviation and percentile interval over the shuffles, and
        ``n_permutations``.
    :raises ValueError: for an unknown arm or data without a MoA column.
    """
    from .embeddings import _profile_table

    label, score, fixed = _arm(arm)
    frame = _profile_table(data)
    if moa_column not in frame.columns:
        raise ValueError(f'the data have no {moa_column!r} column to shuffle')
    left_out = set(_names(controls))
    frame = frame[frame[moa_column].notna()
                  & ~frame[moa_column].astype(str).isin(left_out)
                  & ~frame[compound_column].astype(str).isin(left_out)].reset_index(drop=True)
    options = dict(moa_column=moa_column, compound_column=compound_column,
                   treatment_columns=treatment_columns, batch_column=batch_column,
                   features=features, controls=(), confidence=confidence, **fixed)
    if score is loco_supervised:
        options['workers'] = workers
    grouped = frame.groupby(compound_column, sort=True)[moa_column]
    per_compound = bool((grouped.nunique() == 1).all())
    compound_moa = grouped.first()
    rng = np.random.default_rng(seed)
    shares: Dict[str, List[float]] = {}
    totals: Dict[str, int] = {}
    for index in range(max(0, int(n_permutations))):
        shuffled = frame.copy()
        if per_compound:
            mapping = dict(zip(compound_moa.index, rng.permutation(compound_moa.to_numpy())))
            shuffled[moa_column] = frame[compound_column].map(mapping)
        else:
            shuffled[moa_column] = rng.permutation(frame[moa_column].to_numpy())
        if score is loco_supervised:
            options['seed'] = int(seed) + index + 1
        table = score(shuffled, **options)['table']
        for _, row in table.iterrows():
            shares.setdefault(row['evaluation'], []).append(float(row['accuracy']))
            totals[row['evaluation']] = int(row['n_treatments'])
    tail = (1.0 - float(confidence)) / 2.0
    rows = [_row(f'control: shuffled MoA labels ({label})', rule,
                 n_treatments=totals[rule], accuracy=float(np.mean(values)),
                 ci_low=float(np.quantile(values, tail)),
                 ci_high=float(np.quantile(values, 1.0 - tail)),
                 sd=float(np.std(values)), n_permutations=len(values))
            for rule, values in shares.items()]
    return _table(rows)


def majority_class_control(predictions: pd.DataFrame, *, moa_column: str = 'moa',
                           confidence: float = 0.95) -> pd.DataFrame:
    """The accuracy of always guessing the commonest MoA.

    Counted over the treatments an arm scored, per evaluation, so it is the
    floor that arm has to beat (under NSCB a treatment without a candidate
    is not scored and not counted).

    :param predictions: the ``predictions`` of an arm, with ``evaluation``,
        ``scored`` and the MoA column.
    :param moa_column: column with the MoA.
    :param confidence: level of the Wilson score interval.
    :returns: one control row per evaluation, naming the majority class.
    :raises ValueError: when the predictions lack a needed column.
    """
    missing = [column for column in (moa_column, 'evaluation', 'scored')
               if column not in predictions.columns]
    if missing:
        raise ValueError(f'columns not in the predictions: {missing}')
    rows = []
    for rule, part in predictions.groupby('evaluation', sort=False):
        counts = part.loc[part['scored'].astype(bool), moa_column].astype(str).value_counts()
        total = int(counts.sum())
        majority = str(counts.index[0]) if total else ''
        correct = int(counts.iloc[0]) if total else 0
        low, high = _wilson(correct, total, confidence)
        rows.append(_row(f'control: majority class ({majority})', rule, n_treatments=total,
                         correct=correct, accuracy=correct / total if total else np.nan,
                         ci_low=low, ci_high=high))
    return _table(rows)


def _treatment_profiles(frame: pd.DataFrame, chosen: List[str], *, moa_column: str,
                        treatment_columns: List[str], batch_column: Optional[str],
                        replicates: str) -> pd.DataFrame:
    """Combine replicate wells into one profile per treatment, MoA attached."""
    from .embeddings import aggregate_profiles

    profiles = aggregate_profiles(frame, by=treatment_columns, features=chosen,
                                  method=replicates, batch_column=batch_column,
                                  drop_controls=False)
    moa = frame[treatment_columns + [moa_column]].drop_duplicates(treatment_columns)
    return profiles.merge(moa, on=treatment_columns, how='left')


def moa_benchmark(data: Any, *, arms: Sequence[str] = ARMS, moa_column: str = 'moa',
                  compound_column: str = 'compound',
                  treatment_columns: Sequence[str] = ('compound', 'concentration'),
                  batch_column: Optional[str] = None,
                  features: Optional[Sequence[str]] = None, labels: Any = None,
                  controls: Any = ('DMSO',), replicates: str = 'median',
                  n_permutations: int = 100, supervised_permutations: int = 10,
                  seed: int = 0, workers: int = 1,
                  confidence: float = 0.95) -> Dict[str, Any]:
    """Run the leave-one-compound-out arms and their controls on one table.

    With per-well input (a treatment on several rows) the 1-NN arm scores
    treatment profiles made by
    :func:`spacr.embeddings.aggregate_profiles` (``replicates`` of the
    wells), and the supervised arms train on the wells. With one row per
    treatment every arm uses the rows as they are. Each arm gets a
    shuffled-label control row per evaluation, and the first arm a
    majority-class row.

    A profiling run, and the CellProfiler head-to-head (item 753) for each
    feature set, call::

        result = moa_benchmark(wells, batch_column="batch",
                               arms=("1-NN", "logistic"))
        result["table"]      # accuracy per arm and evaluation, then controls
        result["glossary"]   # every abbreviation and column spelled out

    :param data: profiles, a frame or a table path.
    :param arms: arms to run, from :data:`ARMS`.
    :param moa_column: column with the MoA.
    :param compound_column: column that names the compound.
    :param treatment_columns: columns that name a treatment.
    :param batch_column: column with each well's batch; ``None`` scores NSC
        only.
    :param features: profile columns; ``None`` takes every numeric column
        outside the key columns, ``Metadata_`` columns and counts.
    :param labels: a frame or table path with the MoA, joined on the shared
        columns when the data carry none.
    :param controls: compound or MoA values left out, such as ``'DMSO'``.
    :param replicates: ``'median'`` or ``'mean'``, how wells combine into a
        treatment profile for the 1-NN arm.
    :param n_permutations: shuffles for the 1-NN control.
    :param supervised_permutations: shuffles for each supervised control.
    :param seed: seed of the shuffles and the XGBoost fits.
    :param workers: folds fitted at a time by the supervised arms.
    :param confidence: level of the intervals.
    :returns: ``{"table", "predictions", "glossary", "profiles",
        "features"}``: the accuracy table (arm rows, then control rows;
        ``attrs["glossary"]`` too), the per-treatment predictions of every
        arm, :data:`GLOSSARY`, the treatment profiles the 1-NN arm scored
        and the feature columns.
    :raises ValueError: for an unknown arm or replicate method, or for the
        reasons of the arms.
    """
    replicates = str(replicates).strip().lower()
    if replicates not in REPLICATE_METHODS:
        raise ValueError(f'replicates must be one of {REPLICATE_METHODS}, not {replicates!r}')
    chosen_arms = [(arm, *_arm(arm)) for arm in _names(arms)]
    if not chosen_arms:
        raise ValueError('name at least one arm')
    treatment_columns = _check_treatment_columns(treatment_columns, compound_column)
    frame, chosen = _prepare(data, moa_column=moa_column, compound_column=compound_column,
                             treatment_columns=treatment_columns, batch_column=batch_column,
                             features=features, labels=labels, controls=controls)
    if frame.duplicated(treatment_columns).any():
        profiles = _treatment_profiles(frame, chosen, moa_column=moa_column,
                                       treatment_columns=treatment_columns,
                                       batch_column=batch_column, replicates=replicates)
    else:
        profiles = frame
    shared = dict(moa_column=moa_column, compound_column=compound_column,
                  treatment_columns=treatment_columns, batch_column=batch_column,
                  features=chosen, confidence=confidence)
    arm_tables, control_tables, predictions = [], [], []
    for name, label, score, fixed in chosen_arms:
        supervised = score is loco_supervised
        source = frame if supervised else profiles
        options = dict(shared, controls=(), **fixed)
        if supervised:
            options.update(seed=seed, workers=workers)
        logger.info('MoA benchmark: scoring %s', label)
        result = score(source, **options)
        arm_tables.append(result['table'])
        predictions.append(result['predictions'])
        shuffles = supervised_permutations if supervised else n_permutations
        if int(shuffles) > 0:
            control_tables.append(shuffled_label_control(
                name, source, n_permutations=shuffles, seed=seed, workers=workers,
                controls=(), **shared))
    control_tables.append(majority_class_control(predictions[0], moa_column=moa_column,
                                                 confidence=confidence))
    table = pd.concat(arm_tables + control_tables, ignore_index=True)
    table.attrs['glossary'] = dict(GLOSSARY)
    return {'table': table, 'predictions': pd.concat(predictions, ignore_index=True),
            'glossary': dict(GLOSSARY), 'profiles': profiles, 'features': chosen}


def write_moa_benchmark(result: Mapping[str, Any], dst: Any) -> List[str]:
    """Write a :func:`moa_benchmark` result through :func:`spacr.tabular.write_table`.

    Writes ``moa_accuracy.csv`` (the table), ``moa_predictions.csv``,
    ``moa_glossary.csv`` (term and meaning, so a reader of the table needs
    nothing else) and ``moa_summary.json`` (the table as records, the
    number of features and :data:`SUPERVISED_REPRODUCTION`).

    :param result: what :func:`moa_benchmark` returns.
    :param dst: the folder; created if absent.
    :returns: the paths written.
    """
    from .tabular import write_table

    folder = Path(dst)
    folder.mkdir(parents=True, exist_ok=True)
    glossary = result.get('glossary') or GLOSSARY
    terms = pd.DataFrame({'term': list(glossary), 'meaning': list(glossary.values())})
    paths = [write_table(result['table'], folder / 'moa_accuracy.csv', canonicalise=False),
             write_table(result['predictions'], folder / 'moa_predictions.csv',
                         canonicalise=False),
             write_table(terms, folder / 'moa_glossary.csv', canonicalise=False)]
    summary = {'n_features': len(result.get('features') or ()),
               'table': json.loads(result['table'].to_json(orient='records')),
               'supervised_reproduction': SUPERVISED_REPRODUCTION}
    summary_path = folder / 'moa_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return [str(path) for path in paths] + [str(summary_path)]


def run_moa_benchmark(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Score a profiles table from settings and write the results.

    Runs :func:`moa_benchmark` with the keys of
    :func:`moa_benchmark_defaults` and writes the result with
    :func:`write_moa_benchmark` to ``moa_dst``.

    :param settings: the keys of :func:`moa_benchmark_defaults`; missing
        keys take their defaults. ``moa_profiles`` may also be a frame.
    :returns: the :func:`moa_benchmark` result with ``paths`` added.
    :raises ValueError: without a profiles table, or for settings
        :func:`moa_benchmark` rejects.
    """
    settings = moa_benchmark_defaults(settings)
    source = settings['moa_profiles']
    if source is None or (isinstance(source, str) and not source):
        raise ValueError('moa_profiles is blank: name the profiles table to score')
    result = moa_benchmark(
        source, arms=_names(settings['moa_arms']), moa_column=settings['moa_label_column'],
        compound_column=settings['moa_compound_column'],
        treatment_columns=_names(settings['moa_treatment_columns']),
        batch_column=settings['moa_batch_column'] or None,
        labels=settings['moa_labels'] or None, controls=_names(settings['moa_controls']),
        replicates=settings['moa_replicates'], n_permutations=int(settings['moa_shuffles']),
        supervised_permutations=int(settings['moa_supervised_shuffles']),
        seed=int(settings['moa_seed']), workers=int(settings['moa_workers']))
    dst = settings['moa_dst']
    if not dst:
        base = Path.cwd() if isinstance(source, pd.DataFrame) else Path(os.fspath(source)).parent
        dst = base / 'moa_benchmark'
    result['paths'] = write_moa_benchmark(result, dst)
    for _, row in result['table'].iterrows():
        logger.info('%s %s: %.3f over %s treatments', row['arm'], row['evaluation'],
                    row['accuracy'], row['n_treatments'])
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run :func:`run_moa_benchmark` from the command line.

    ``python -m spacr.moa_benchmark --profiles wells.csv --labels moa.csv
    --batch-column batch --dst out`` scores every arm under NSC and NSCB.

    :param argv: the arguments; None reads :data:`sys.argv`.
    :returns: 0 when a table was written, 1 otherwise.
    """
    parser = argparse.ArgumentParser(prog='python -m spacr.moa_benchmark',
                                     description=_DESCRIPTION)
    for key, value in _DEFAULTS.items():
        flag = '--' + key[len('moa_'):].replace('_', '-')
        parser.add_argument(flag, dest=key, type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    result = run_moa_benchmark(args)
    return 0 if len(result['table']) else 1


if __name__ == '__main__':
    sys.exit(main())
