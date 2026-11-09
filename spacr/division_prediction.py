"""Predict whether a tracked cell divides within N frames (alpha).

The division analysis (:mod:`spacr.division_analysis`) looks backwards from
each division. This module asks the forward question: from what a cell looks
like now, and how it changed over the last few frames, can a model tell that
it will divide within the next ``N`` frames?

A classifier answers that question too well when it is evaluated carelessly.
Frames of one cell are near copies of each other, and cells of one movie
share its imaging, so a model tested on other cells of the movies it was
trained on learns the movies. Here every evaluation is out of movie:

1. :func:`division_prediction_features` -- one row per cell and frame with
   the shape, intensity and motion measurements of
   :func:`spacr.division_analysis.measure_label_stack` and
   :func:`~spacr.division_analysis.add_motion_features`, their change over
   the last ``history`` frames, the cell's age, and the label "divides
   within ``frames`` frames". Frames whose future the movie does not show
   are left out rather than called negative.
2. :func:`predict_divisions` -- cross-validation grouped by movie (a fold's
   test movies are never in its training set) with three controls beside the
   model: the same learner on age alone, the same learner trained on labels
   shuffled within each movie, and the majority class. AUROC and AUPRC of
   the held-out predictions get bootstrap 95% intervals that resample movies
   (cells when there are too few movies, and then the summary says so), and
   the model's margin over each control gets a paired interval.
3. :func:`write_division_prediction` and :func:`division_prediction_figure`
   -- per-fold metrics, permutation feature importances on the held-out
   movies and every held-out prediction as CSVs, a JSON summary and one
   figure.

xgboost is used when installed and scikit-learn's histogram gradient
boosting otherwise. Nothing here reads images or needs a GPU.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .division_analysis import COMPARED_MEASURES, LOG_MEASURES, division_events

_ARMS = ('model', 'age_only', 'shuffled', 'majority')
_ARM_LABELS = {'model': 'model', 'age_only': 'age only', 'shuffled': 'shuffled labels',
               'majority': 'majority class'}
_MODELS = ('auto', 'xgboost', 'gradient_boosting', 'logistic')
_RESERVED = ('movie', 'track_id', 't', 'label', 'frames_to_division')
_TABLES = ('metrics', 'folds', 'importances', 'predictions')
_MIN_FIELDS = 3


def division_prediction_features(measures: pd.DataFrame, lineage: pd.DataFrame,
                                 frames: int = 5, mode: str = 'auto',
                                 history: int = 3) -> pd.DataFrame:
    """Per-cell, per-frame features and the label "divides within ``frames`` frames".

    A cell divides at the alignment frame of
    :func:`spacr.division_analysis.division_events` (the last frame before a
    daughter is seen), so the label of frame ``t`` is 1 when that frame lies
    in ``t .. t + frames - 1``. A frame is labelled 0 only when the cell is
    seen past ``t + frames`` without dividing or its next division comes
    later; a frame whose horizon runs past the end of its track without a
    division is left out, because a lost or departing cell may divide unseen.
    For budding the mother's divisions are her buds, and her age restarts at
    each bud emergence; for fission age counts from the track's first frame.
    Features use the current and earlier frames only.

    :param measures: per-frame table with ``track_id``, ``t`` and the
        measurements, from :func:`spacr.division_analysis.measure_label_stack`
        and usually :func:`spacr.division_analysis.add_motion_features`.
    :param lineage: lineage table with ``label``, ``t_start``, ``t_end`` and
        ``parent``.
    :param frames: the horizon ``N`` in frames.
    :param mode: ``'auto'`` (whichever kind the lineage has more of),
        ``'fission'`` or ``'budding'``.
    :param history: frames back for the change features.
    :returns: one row per labelled (``track_id``, ``t``) with ``label``,
        ``frames_to_division`` (NaN when no later division is seen; never a
        feature), ``age``, every measurement present and its change
        ``d_<measure>`` over ``history`` frames (log ratio for positive
        size and shape measures, difference otherwise; NaN early in a track).
    :raises ValueError: for an unknown ``mode``, ``frames`` or ``history``
        below 1, or ``measures`` without ``track_id`` or ``t``.
    """
    if mode not in ('auto', 'fission', 'budding'):
        raise ValueError(f'unknown mode {mode!r}')
    if int(frames) < 1 or int(history) < 1:
        raise ValueError('frames and history must be at least 1')
    missing = [c for c in ('track_id', 't') if c not in measures.columns]
    if missing:
        raise ValueError(f'measures lack columns {missing}')
    frames, history = int(frames), int(history)
    if mode == 'auto':
        counts = division_events(lineage, mode='auto')['kind'].value_counts()
        mode = 'budding' if counts.get('budding', 0) > counts.get('fission', 0) else 'fission'
    events = division_events(lineage, mode=mode)
    divisions = {int(p): np.sort(g['t_div'].to_numpy(dtype=np.int64))
                 for p, g in events.groupby('parent')}
    lin = lineage.set_index('label')
    columns = [c for c in COMPARED_MEASURES if c in measures.columns]
    columns += [c for c in ('displacement',) if c in measures.columns and c not in columns]
    out = []
    for track, g in measures.groupby('track_id'):
        track = int(track)
        g = g.drop_duplicates('t').set_index('t').sort_index()
        t_obs = g.index.to_numpy(dtype=np.int64)
        start = int(lin.at[track, 't_start']) if track in lin.index else int(t_obs.min())
        end = int(lin.at[track, 't_end']) if track in lin.index else int(t_obs.max())
        dense = g.reindex(range(int(t_obs.min()), int(t_obs.max()) + 1))
        past = dense[columns].shift(history)
        divs = divisions.get(track, np.zeros(0, dtype=np.int64))
        idx = np.searchsorted(divs, t_obs, side='left')
        has_next = idx < len(divs)
        nxt = divs[np.minimum(idx, len(divs) - 1)] if len(divs) else np.zeros(len(t_obs))
        to_div = np.where(has_next, nxt - t_obs, np.nan).astype(float)
        label = np.where(has_next & (to_div < frames), 1.0,
                         np.where(has_next | (t_obs + frames <= end), 0.0, np.nan))
        if mode == 'budding' and len(divs):
            emergences = divs + 1
            j = np.searchsorted(emergences, t_obs, side='right') - 1
            reset = np.where(j >= 0, emergences[np.maximum(j, 0)], start)
        else:
            reset = np.full(len(t_obs), start)
        rows = pd.DataFrame({'track_id': track, 't': t_obs, 'label': label,
                             'frames_to_division': to_div,
                             'age': (t_obs - np.maximum(reset, start)).astype(float)})
        now = dense.loc[t_obs, columns].to_numpy(dtype=float)
        then = past.loc[t_obs, columns].to_numpy(dtype=float)
        for k, c in enumerate(columns):
            rows[c] = now[:, k]
            if c in LOG_MEASURES:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where((now[:, k] > 0) & (then[:, k] > 0),
                                     np.log(now[:, k] / then[:, k]), np.nan)
                rows[f'd_{c}'] = ratio
            else:
                rows[f'd_{c}'] = now[:, k] - then[:, k]
        out.append(rows[np.isfinite(label)])
    base = ['track_id', 't', 'label', 'frames_to_division', 'age']
    if not out:
        return pd.DataFrame(columns=base + columns + [f'd_{c}' for c in columns])
    table = pd.concat(out, ignore_index=True)
    table['label'] = table['label'].astype(int)
    return table


def _learner(model, seed):
    """A fresh classifier for ``model`` and the name of what was built."""
    if model in ('auto', 'xgboost'):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            pass
        else:
            return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                                 n_jobs=2, random_state=seed, eval_metric='logloss',
                                 verbosity=0), 'xgboost'
    if model == 'logistic':
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                             LogisticRegression(max_iter=1000)), 'logistic'
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=3,
                                          random_state=seed), 'gradient_boosting'


def _auroc(y, score):
    """Area under the ROC curve from ranks (ties share their mean rank); NaN for one class."""
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return float('nan')
    ranks = rankdata(score)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auprc(y, score):
    """Average precision, stepping at each distinct score as scikit-learn does; NaN without positives."""
    y = np.asarray(y, dtype=float)
    if not y.sum():
        return float('nan')
    order = np.argsort(-np.asarray(score, dtype=float), kind='mergesort')
    s, y = np.asarray(score, dtype=float)[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[last]
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _metric_pair(y, score):
    """AUROC and AUPRC of one set of predictions."""
    return _auroc(y, score), _auprc(y, score)


def _bootstrap(predictions, n_boot, rng, by_movie):
    """Bootstrap draws of AUROC and AUPRC per arm, resampling movies or rows.

    :returns: ``{arm: array of shape (draws, 2)}``; draws in which the
        resample holds one class only are dropped.
    """
    y_all = predictions['label'].to_numpy(dtype=int)
    if by_movie:
        groups = [np.nonzero((predictions['movie'] == m).to_numpy())[0]
                  for m in pd.unique(predictions['movie'])]
    draws = {arm: [] for arm in _ARMS}
    for _ in range(int(n_boot)):
        if by_movie:
            pick = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in pick])
        else:
            idx = rng.integers(0, len(y_all), size=len(y_all))
        y = y_all[idx]
        if y.min() == y.max():
            continue
        for arm in _ARMS:
            if arm == 'majority':
                draws[arm].append((0.5, float(y.mean())))
            else:
                draws[arm].append(_metric_pair(y, predictions[f'p_{arm}'].to_numpy()[idx]))
    return {arm: np.asarray(v, dtype=float).reshape(-1, 2) for arm, v in draws.items()}


def _interval(values):
    """The 2.5th and 97.5th percentiles of the finite values, NaN when there are none."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float('nan'), float('nan')
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _shuffle_within(labels, movies, rng):
    """Labels permuted within each movie, so every movie keeps its share of positives."""
    labels = np.asarray(labels).copy()
    for m in pd.unique(movies):
        idx = np.nonzero(np.asarray(movies) == m)[0]
        labels[idx] = labels[rng.permutation(idx)]
    return labels


def predict_divisions(features: pd.DataFrame, model: str = 'auto', folds: int = 5,
                      n_boot: int = 1000, rng=0, min_movies: int = 5,
                      feature_columns: Optional[Sequence[str]] = None,
                      n_repeats: int = 5, one_per_track: bool = True,
                      sample_seed: int = 0) -> dict:
    """Train and evaluate the "divides within N frames" classifier, movie by movie.

    Frames of one track are near copies of each other, so by default one
    labelled frame is drawn per track (per ``movie`` and ``track_id``) with
    the fixed ``sample_seed``, and every track counts once in training and
    evaluation. Folds are grouped by movie (scikit-learn ``GroupKFold`` with
    ``min(folds, number of movies)`` folds), so every prediction is made by a
    model that never saw that movie. Beside the model, three controls are
    scored on the same held-out rows: ``age_only`` (the same learner on the
    cell's age alone), ``shuffled`` (the same learner trained on labels
    permuted within each training movie, which should score near AUROC 0.5)
    and ``majority`` (one constant score: AUROC 0.5 and AUPRC equal to the
    share of positives). AUROC and AUPRC are computed on all held-out
    predictions together. Their 95% intervals resample movies with
    replacement from ``min_movies`` movies on; with fewer they resample rows,
    which treats frames of one cell as independent and is too narrow, and
    ``ci_note`` says so. The model's margin over each control is resampled
    in the same draws (paired). Permutation importances (drop in held-out
    AUROC when one feature is shuffled) are averaged over the folds.

    :param features: rows from :func:`division_prediction_features` with a
        ``movie`` column naming each row's movie.
    :param model: ``'auto'`` (xgboost when installed, else
        ``'gradient_boosting'``), ``'xgboost'`` (falls back the same way),
        ``'gradient_boosting'`` (scikit-learn histogram gradient boosting)
        or ``'logistic'`` (standardised logistic regression).
    :param folds: the most folds; never more than the number of movies.
    :param n_boot: bootstrap resamples for the intervals; 0 skips them.
    :param rng: seed or generator for the shuffles, learners and bootstrap.
    :param min_movies: the fewest movies for which intervals resample movies.
    :param feature_columns: the feature columns; None takes every column
        except ``movie``, ``track_id``, ``t``, ``label`` and
        ``frames_to_division``.
    :param n_repeats: shuffles per feature for the permutation importance;
        0 skips it.
    :param one_per_track: draw one labelled frame per track; False keeps
        every frame.
    :param sample_seed: seed of that draw, fixed so a rerun uses the same
        frames.
    :returns: dict with ``metrics`` (one row per arm: ``auroc``,
        ``auprc`` with ``_ci_low``/``_ci_high``, ``fold_auroc_mean`` and,
        for the controls, ``delta_auroc``/``delta_auprc`` of the model over
        that arm with intervals), ``folds`` (one row per fold and arm),
        ``importances``, ``predictions`` (every held-out row with its fold
        and the score of each arm) and ``summary`` (counts, learner,
        ``ci_method``, ``ci_note``, ``beats_shuffled``, ``beats_age_only``,
        ``beats_majority`` and ``verdict``).
    :raises ValueError: for an unknown ``model``, features without
        ``movie``, ``label`` or ``age``, fewer than two movies, or labels of
        one class only.
    """
    from sklearn.model_selection import GroupKFold

    if model not in _MODELS:
        raise ValueError(f'unknown model {model!r}; choose one of {_MODELS}')
    missing = [c for c in ('movie', 'label', 'age') if c not in features.columns]
    if missing:
        raise ValueError(f'features lack columns {missing}')
    data = features[features['label'].notna()].reset_index(drop=True)
    sampled = bool(one_per_track) and 'track_id' in data and len(data)
    if sampled:
        draw = np.random.default_rng(sample_seed).random(len(data))
        data = (data.assign(_draw=draw).sort_values('_draw')
                .drop_duplicates(['movie', 'track_id']).drop(columns='_draw')
                .sort_values(['movie', 'track_id']).reset_index(drop=True))
    movies = data['movie'].astype(str).to_numpy()
    n_movies = len(set(movies))
    if n_movies < 2:
        raise ValueError('predicting divisions needs at least two movies: '
                         'folds are grouped by movie')
    y = data['label'].to_numpy(dtype=int)
    if y.min() == y.max():
        raise ValueError('the labels are all of one class')
    rng = np.random.default_rng(rng)
    if feature_columns is None:
        feature_columns = [c for c in data.columns if c not in _RESERVED]
    feature_columns = list(feature_columns)
    X = data[feature_columns].to_numpy(dtype=float)
    age = data[['age']].to_numpy(dtype=float)
    n_splits = min(int(folds), n_movies)
    if n_splits < 2:
        raise ValueError('folds must be at least 2')
    scores = {arm: np.full(len(data), np.nan) for arm in _ARMS}
    fold_of = np.full(len(data), -1)
    fold_rows, importance_rows, learner_name = [], [], None
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, movies)):
        shared = set(movies[train]) & set(movies[test])
        if shared:
            raise RuntimeError(f'fold {fold} shares movies {sorted(shared)}')
        fold_of[test] = fold
        y_train = y[train]
        if y_train.min() == y_train.max():
            scores['model'][test] = scores['age_only'][test] = scores['shuffled'][test] = \
                float(y_train.mean())
        else:
            seed = int(rng.integers(0, 2 ** 31 - 1))
            fitted, learner_name = _learner(model, seed)
            fitted.fit(X[train], y_train)
            scores['model'][test] = fitted.predict_proba(X[test])[:, 1]
            aged, _ = _learner(model, seed)
            aged.fit(age[train], y_train)
            scores['age_only'][test] = aged.predict_proba(age[test])[:, 1]
            shuffled, _ = _learner(model, seed)
            shuffled.fit(X[train], _shuffle_within(y_train, movies[train], rng))
            scores['shuffled'][test] = shuffled.predict_proba(X[test])[:, 1]
            if n_repeats and y[test].min() != y[test].max():
                importance_rows += _permutation_importance(
                    fitted, X[test], y[test], feature_columns, fold, n_repeats, rng)
        scores['majority'][test] = float(y_train.mean())
        for arm in _ARMS:
            if arm == 'majority':
                both = y[test].min() != y[test].max()
                auroc = 0.5 if both else float('nan')
                auprc = float(y[test].mean()) if y[test].any() else float('nan')
            else:
                auroc, auprc = _metric_pair(y[test], scores[arm][test])
            fold_rows.append({'fold': fold, 'arm': arm, 'auroc': auroc, 'auprc': auprc,
                              'n_test': int(len(test)), 'n_positive': int(y[test].sum()),
                              'n_train': int(len(train)),
                              'test_movies': ';'.join(sorted(set(movies[test]))),
                              'n_shared_movies': len(shared)})
    predictions = data[[c for c in ('movie', 'track_id', 't') if c in data]].copy()
    predictions['movie'] = movies
    predictions['label'] = y
    predictions['fold'] = fold_of
    for arm in _ARMS:
        predictions[f'p_{arm}'] = scores[arm]
    by_movie = n_movies >= int(min_movies)
    draws = _bootstrap(predictions, n_boot, rng, by_movie) if n_boot else None
    folds_table = pd.DataFrame(fold_rows)
    prevalence = float(y.mean())
    metrics = []
    for arm in _ARMS:
        if arm == 'majority':
            auroc, auprc = 0.5, prevalence
        else:
            auroc, auprc = _metric_pair(y, scores[arm])
        row = {'arm': arm, 'label': _ARM_LABELS[arm], 'auroc': auroc, 'auprc': auprc,
               'fold_auroc_mean': float(np.nanmean(
                   folds_table.loc[folds_table['arm'] == arm, 'auroc'].to_numpy(float)))
               if folds_table['auroc'].notna().any() else float('nan')}
        for k, name in enumerate(('auroc', 'auprc')):
            lo, hi = _interval(draws[arm][:, k]) if draws is not None else (np.nan, np.nan)
            row[f'{name}_ci_low'], row[f'{name}_ci_high'] = lo, hi
        if arm != 'model':
            for k, name in enumerate(('auroc', 'auprc')):
                row[f'delta_{name}'] = metrics[0][name] - row[name]
                if draws is not None and len(draws['model']) == len(draws[arm]):
                    lo, hi = _interval(draws['model'][:, k] - draws[arm][:, k])
                else:
                    lo, hi = np.nan, np.nan
                row[f'delta_{name}_ci_low'], row[f'delta_{name}_ci_high'] = lo, hi
        metrics.append(row)
    metrics = pd.DataFrame(metrics)
    importances = (pd.DataFrame(importance_rows).groupby('feature', as_index=False)
                   .agg(importance_mean=('importance', 'mean'),
                        importance_sd=('importance', 'std'), n_folds=('fold', 'nunique'))
                   .sort_values('importance_mean', ascending=False).reset_index(drop=True)
                   if importance_rows else
                   pd.DataFrame(columns=['feature', 'importance_mean', 'importance_sd',
                                         'n_folds']))
    beats = {}
    for arm in ('shuffled', 'age_only', 'majority'):
        low = metrics.loc[metrics['arm'] == arm, 'delta_auroc_ci_low'].iloc[0]
        beats[arm] = bool(np.isfinite(low) and low > 0)
    if beats['shuffled'] and beats['age_only']:
        verdict = 'the model beats the shuffled-label and age-only controls'
    elif beats['shuffled']:
        verdict = ('the model beats the shuffled-label control but not age alone: '
                   'the signal may be the cell cycle clock')
    else:
        verdict = 'the model does not beat the shuffled-label control'
    note = ('intervals resample movies' if by_movie else
            f'intervals resample rows (fewer than {int(min_movies)} movies): frames of one '
            'cell are not independent, so these intervals are too narrow')
    summary = {'n_movies': n_movies, 'movies': sorted(set(movies)), 'n_rows': int(len(data)),
               'row_sampling': (f'one frame per track (seed {int(sample_seed)})' if sampled
                                else 'every labelled frame'),
               'n_positive': int(y.sum()), 'prevalence': prevalence, 'n_folds': n_splits,
               'model': model, 'learner': learner_name or 'none (one class in training)',
               'features': feature_columns, 'n_boot': int(n_boot), 'min_movies': int(min_movies),
               'ci_method': ('movie' if by_movie else 'row') if n_boot else 'none',
               'ci_note': note if n_boot else 'no intervals: n_boot is 0',
               'cv': 'GroupKFold by movie; no fold shares a movie',
               'beats_shuffled': beats['shuffled'], 'beats_age_only': beats['age_only'],
               'beats_majority': beats['majority'], 'verdict': verdict}
    return {'metrics': metrics, 'folds': folds_table, 'importances': importances,
            'predictions': predictions, 'summary': summary}


def _permutation_importance(fitted, X, y, names, fold, n_repeats, rng):
    """Drop in held-out AUROC when each feature is shuffled, one row per feature."""
    base = _auroc(y, fitted.predict_proba(X)[:, 1])
    rows = []
    for k, name in enumerate(names):
        drops = []
        for _ in range(int(n_repeats)):
            shuffled = X.copy()
            shuffled[:, k] = shuffled[rng.permutation(len(shuffled)), k]
            drops.append(base - _auroc(y, fitted.predict_proba(shuffled)[:, 1]))
        rows.append({'fold': fold, 'feature': name, 'importance': float(np.mean(drops))})
    return rows


def division_prediction_figure(results: dict):
    """AUROC and AUPRC of the model and its three controls, with intervals and folds.

    :param results: the dict from :func:`predict_divisions` or
        :func:`read_division_prediction`.
    :returns: a :class:`matplotlib.figure.Figure` with two panels: bars of
        AUROC and AUPRC per arm with their bootstrap 95% intervals and each
        fold's value as a jittered point; the chance line is drawn at 0.5
        for AUROC and at the share of positives for AUPRC.
    """
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import ROLES, resolve_ink, theme_target

    metrics = results.get('metrics', pd.DataFrame())
    folds = results.get('folds', pd.DataFrame())
    summary = results.get('summary', {}) or {}
    fig = Figure(figsize=(6.4, 3.2), dpi=100)
    axes = fig.subplots(1, 2)
    _register_figure_data(fig, folds, x='arm', y='auroc', kind='bar',
                          title='Predicting division within N frames: held-out movies')
    if not len(metrics):
        for ax in axes:
            ax.text(0.5, 0.5, 'no prediction results', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig
    colors = {'model': ROLES['highlight'], 'age_only': ROLES['control_positive'],
              'shuffled': ROLES['control_negative'], 'majority': ROLES['data']}
    jitter = np.random.default_rng(0)
    arms = [a for a in _ARMS if a in set(metrics['arm'])]
    for ax, name, chance in ((axes[0], 'auroc', 0.5),
                             (axes[1], 'auprc', summary.get('prevalence'))):
        for i, arm in enumerate(arms):
            row = metrics[metrics['arm'] == arm].iloc[0]
            value = float(row[name])
            lo, hi = float(row.get(f'{name}_ci_low', np.nan)), float(row.get(f'{name}_ci_high', np.nan))
            ax.bar(i, value, width=0.6, color=colors[arm], alpha=0.6)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(i, value, yerr=[[value - lo], [hi - value]],
                            color=resolve_ink(theme_target()),
                            lw=0.8, capsize=3)
            if len(folds) and name in folds:
                pts = folds.loc[folds['arm'] == arm, name].to_numpy(float)
                pts = pts[np.isfinite(pts)]
                ax.scatter(i + jitter.uniform(-0.15, 0.15, len(pts)), pts, s=10,
                           color=colors[arm], alpha=0.6, linewidths=0, zorder=3)
        if chance is not None and np.isfinite(float(chance)):
            ax.axhline(float(chance), ls=':', lw=0.8, color=ROLES['reference'])
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([_ARM_LABELS[a] for a in arms], rotation=30, ha='right',
                           fontsize='small')
        ax.set_ylim(0, 1)
        ax.set_ylabel('AUROC' if name == 'auroc' else 'AUPRC')
    fig.suptitle(f"Divides within the horizon: {summary.get('n_movies', '?')} held-out movies",
                 fontsize='small')
    fig.tight_layout()
    return fig


def write_division_prediction(results: dict, out_dir, plot: bool = True) -> dict:
    """Write a prediction run as CSV tables, a JSON summary and its figure.

    The tables are ``metrics.csv``, ``folds.csv``, ``importances.csv`` and
    ``predictions.csv``, written through :func:`spacr.tabular.write_table`;
    the summary is ``summary.json``. With ``plot`` the figure of
    :func:`division_prediction_figure` is saved through
    :func:`spacr.plot.save_figure` as ``division_prediction``.

    :param results: the dict from :func:`predict_divisions`.
    :param out_dir: the folder to write into; created when missing.
    :param plot: save the figure too.
    :returns: ``{name: path}`` of everything written.
    """
    import json
    import os

    from .tabular import write_table

    os.makedirs(out_dir, exist_ok=True)
    paths = {name: write_table(results.get(name, pd.DataFrame()),
                               os.path.join(out_dir, f'{name}.csv')) for name in _TABLES}
    summary = os.path.join(out_dir, 'summary.json')
    with open(summary, 'w', encoding='utf-8') as handle:
        json.dump(results.get('summary', {}), handle, indent=2, default=_json_default)
    paths['summary'] = summary
    if plot:
        from .plot import save_figure

        paths['figure'] = save_figure(division_prediction_figure(results),
                                      os.path.join(out_dir, 'division_prediction.pdf'),
                                      close=True)
    return paths


def _json_default(value):
    """JSON form of numpy scalars."""
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return float(value)


def read_division_prediction(folder) -> dict:
    """Read back a folder written by :func:`write_division_prediction`.

    :param folder: the results folder.
    :returns: the same dict :func:`predict_divisions` returns; a missing or
        empty table is an empty frame.
    :raises FileNotFoundError: when ``folder`` holds no ``summary.json``.
    """
    import json
    import os

    from .tabular import read_table

    summary = os.path.join(folder, 'summary.json')
    if not os.path.isfile(summary):
        raise FileNotFoundError(f'no division prediction results in {folder}')
    out = {}
    for name in _TABLES:
        try:
            out[name] = read_table(os.path.join(folder, f'{name}.csv'), canonicalise=False,
                                   report=None)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            out[name] = pd.DataFrame()
    with open(summary, encoding='utf-8') as handle:
        out['summary'] = json.load(handle)
    return out


def _prediction_options(settings):
    """The prediction keyword arguments from the run settings.

    :raises ValueError: for a horizon or fold count below 2, or an unknown model.
    """
    frames = settings.get('timelapse_division_prediction_frames')
    frames = 5 if frames in (None, '') else frames
    folds = settings.get('timelapse_division_prediction_folds')
    folds = 5 if folds in (None, '') else folds
    model = settings.get('timelapse_division_prediction_model') or 'auto'
    if isinstance(frames, bool) or int(frames) < 1:
        raise ValueError('timelapse_division_prediction_frames must be at least 1')
    if isinstance(folds, bool) or int(folds) < 2:
        raise ValueError('timelapse_division_prediction_folds must be at least 2')
    if model not in _MODELS:
        raise ValueError(f'timelapse_division_prediction_model must be one of {_MODELS}')
    return {'frames': int(frames), 'folds': int(folds), 'model': str(model)}


def _write_field_features(measures, lineage, results, out_dir, settings):
    """Write one field's prediction features beside its division results.

    :returns: the path written, or None when the features could not be built
        (the reason is printed).
    """
    import os

    from .tabular import write_table

    try:
        options = _prediction_options(settings)
        features = division_prediction_features(measures, lineage, frames=options['frames'],
                                                mode=results['summary']['mode'])
        path = write_table(features, os.path.join(out_dir, 'prediction_features.csv'))
    except Exception as exc:
        print(f'Division prediction features could not be built in {out_dir}: {exc}')
        return None
    results.setdefault('paths', {})['prediction_features'] = path
    return path


def _predict_after_run(src, object_type, tracker, settings):
    """Predict divisions once, after a timelapse Make Masks run, over every field it analysed.

    :returns: the results of :func:`_run_prediction_step`, or None when the
        run wrote no division results.
    """
    import os

    prefix = 'trackpy' if tracker == 'iou' else tracker
    division_dir = os.path.join(os.path.dirname(src), 'tracks', 'division')
    if not os.path.isdir(division_dir):
        return None
    return _run_prediction_step(division_dir, f'{prefix}_tracks_{object_type}_', settings)


def _run_prediction_step(division_dir, stem_prefix, settings):
    """Predict divisions over every field under ``division_dir`` that starts with ``stem_prefix``.

    Needs the ``prediction_features.csv`` of at least three fields, so that
    every fold trains on two or more movies; each field is one movie. Writes
    :func:`write_division_prediction` to
    ``division_dir/prediction_<stem_prefix without its last _>``. A failure
    is printed and does not stop the run.

    :returns: the results dict, or None.
    """
    import os

    from .tabular import read_table

    tables = []
    for entry in sorted(os.listdir(division_dir)):
        path = os.path.join(division_dir, entry, 'prediction_features.csv')
        if entry.startswith(stem_prefix) and os.path.isfile(path):
            try:
                table = read_table(path, canonicalise=False, report=None)
            except pd.errors.EmptyDataError:
                continue
            if len(table):
                tables.append(table.assign(movie=entry[len(stem_prefix):] or entry))
    if len(tables) < _MIN_FIELDS:
        print(f'Division prediction waits for {_MIN_FIELDS} fields with features '
              f'({len(tables)} so far): folds are grouped by field')
        return None
    out_dir = os.path.join(division_dir, f"prediction_{stem_prefix.rstrip('_')}")
    try:
        options = _prediction_options(settings)
        results = predict_divisions(pd.concat(tables, ignore_index=True),
                                    model=options['model'], folds=options['folds'])
        results['paths'] = write_division_prediction(
            results, out_dir, plot=bool(settings.get('save', True) or settings.get('plot', False)))
    except Exception as exc:
        print(f'Division prediction could not run in {division_dir}: {exc}')
        return None
    summary = results['summary']
    model = results['metrics'].set_index('arm')
    print(f"Division prediction over {summary['n_movies']} fields: AUROC "
          f"{model.at['model', 'auroc']:.2f} (shuffled labels "
          f"{model.at['shuffled', 'auroc']:.2f}, age only {model.at['age_only', 'auroc']:.2f}); "
          f"{summary['verdict']}; written to {out_dir}")
    return results
