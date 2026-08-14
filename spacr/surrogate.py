"""Explain a computer-vision classifier with a model of measured features.

Activation maps say **where** a model looked. This says **what it responded
to**, in units a biologist already interprets — area, intensity, texture —
by fitting a gradient-boosted model to reproduce the CV model's predictions
from the object features spaCR already measured, and then asking that model
what it used.

The two answer different questions and neither replaces the other. A map
that lights up the parasite tells you the model attends to the parasite; it
does not tell you whether it is responding to the parasite's size, its
brightness, or how far it sits from the nucleus.

**Fidelity is reported first, and it is not a formality.** A surrogate that
cannot reproduce the CV model's decisions has explained nothing, and its
feature ranking is a ranking of how the features predict each other. That
number is the one people skip, so :class:`SurrogateResult` puts it before
the importances and :meth:`SurrogateResult.summary` refuses to lead with
anything else.

**Three importances, because they disagree.**

* *gain* — cheap, comes free with the fit, and is biased towards
  high-cardinality features. Read it last.
* *permutation* — measured on held-out data by breaking one feature at a
  time. Costs a re-scoring per feature and is worth it.
* *SHAP* — additive per-object attributions. The only one that says which
  direction a feature pushed a particular object, and the one that connects
  to :mod:`spacr.attribution`'s pixel-level story if that gains the SHAP
  family too.

Where they disagree, the disagreement is the finding: a feature ranked high
by gain and low by permutation is usually one the model could have used and
did not.

**The join goes through ``png_list``.** A CV model keys on the crop it
scored; the features key on the object. ``png_list`` is the only table that
holds both, so it is the bridge — and joining on ``png_path`` alone is not
enough, because a path is not stable across machines.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "SurrogateError",
    "SurrogateResult",
    "build_surrogate_frame",
    "fit_surrogate",
    "explain_classifier",
    "LEAKY_COLUMN_HINTS",
]

#: Column-name fragments that make a surrogate look brilliant and mean
#: nothing. A model's own scores, or a label copied into the feature table,
#: predict the model's prediction perfectly while explaining none of it.
LEAKY_COLUMN_HINTS: Tuple[str, ...] = (
    "prediction", "pred_", "_pred", "score", "probability", "proba",
    "class", "label", "annotate", "cv_", "logit",
)

#: Columns that identify an object rather than describe it. Left in the
#: frame for joining and dropped before fitting -- a plate id is a perfect
#: predictor of anything that varies by plate.
_IDENTIFIER_COLUMNS: Tuple[str, ...] = (
    "prcfo", "png_path", "plateID", "rowID", "columnID", "fieldID", "wellID",
    "timeID", "object_label", "cell_id", "nucleus_id", "pathogen_id",
    "cytoplasm_id", "label", "prc", "field", "row", "column", "plate", "well",
)


class SurrogateError(RuntimeError):
    """The surrogate cannot be built, or cannot be believed."""


@dataclass
class SurrogateResult:
    """What the surrogate learned, with the caveat attached.

    :param fidelity: held-out accuracy at reproducing the CV model.
    :param baseline: the accuracy of always guessing the commonest class.
        Fidelity is only meaningful against this.
    :param importance: one row per feature, with ``gain``, ``permutation``
        and ``shap`` columns where each was computed.
    :param n_objects: rows the surrogate was fitted on.
    :param class_counts: how many objects the CV model put in each class.
    :param warnings: anything that should be read before the ranking is.
    """

    fidelity: float
    baseline: float
    importance: pd.DataFrame
    n_objects: int
    class_counts: Dict[Any, int] = field(default_factory=dict)
    split_report: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_faithful(self) -> bool:
        """Whether the surrogate beats the majority-class baseline at all.

        Deliberately a low bar. It is not "this explanation is good"; it is
        "this explanation is about the CV model rather than about noise".
        """
        return self.fidelity > self.baseline

    def top(self, n: int = 15, by: str = "permutation") -> pd.DataFrame:
        """The ``n`` most important features by one measure.

        :param n: how many rows.
        :param by: ``'permutation'``, ``'shap'`` or ``'gain'``. Falls back to
            whichever was computed, rather than raising, so a run without
            SHAP still reports something.
        """
        column = by if by in self.importance.columns else None
        if column is None:
            for candidate in ("permutation", "shap", "gain"):
                if candidate in self.importance.columns:
                    column = candidate
                    break
        if column is None:
            return self.importance.head(n)
        return self.importance.sort_values(column, ascending=False).head(n)

    def summary(self, n: int = 15) -> str:
        """A report that leads with the caveat, because the caveat decides
        whether the rest is worth reading."""
        lines = [
            "Surrogate model of the CV classifier",
            "=" * 44,
            f"objects            : {self.n_objects:,}",
            f"CV class balance   : {dict(self.class_counts)}",
            f"baseline (majority): {self.baseline:.3f}",
            f"surrogate fidelity : {self.fidelity:.3f}",
        ]
        if self.split_report:
            lines.append(
                "held-out split     : "
                f"{self.split_report.get('group_by')} grouped; "
                f"{self.split_report.get('group_fraction', 0):.1%} groups / "
                f"{self.split_report.get('cell_fraction', 0):.1%} cells")
        if not self.is_faithful:
            lines += [
                "",
                "*** The surrogate does NOT reproduce the CV model. ***",
                "It scores at or below the majority-class baseline, so the",
                "feature ranking below describes how the features predict",
                "each other, not what the classifier responded to. Do not",
                "report it.",
            ]
        for warning in self.warnings:
            lines.append(f"! {warning}")
        lines += ["", f"Top {n} features:", self.top(n).to_string(index=False)]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Building the frame
# ---------------------------------------------------------------------------

def _read_png_list(db_path: str) -> pd.DataFrame:
    if not os.path.isfile(db_path):
        raise SurrogateError(f"no measurements database at {db_path}")
    with sqlite3.connect(db_path, timeout=30) as conn:
        return pd.read_sql_query('SELECT * FROM "png_list"', conn)


def build_surrogate_frame(db_path: str,
                          predictions: pd.DataFrame,
                          path_column: str = "path",
                          prediction_column: str = "pred",
                          ) -> pd.DataFrame:
    """Join CV predictions to the measured feature table.

    The CV model keys on the crop it scored and the features key on the
    object, so ``png_list`` is the bridge: it is the only table holding
    both. The join is validated one-to-one — a repeated key would multiply
    rows and silently reweight the surrogate towards whichever objects
    happened to duplicate.

    :param db_path: the measurements database.
    :param predictions: a frame with a crop path and a predicted class, as
        ``apply_model_to_tar`` returns.
    :param path_column: the column holding the crop path.
    :param prediction_column: the column holding the predicted class.
    :returns: features joined to a ``cv_prediction`` column.
    :raises SurrogateError: a missing column, or a join that matches nothing.
    """
    from spacr.io import _read_and_join_tables

    for column in (path_column, prediction_column):
        if column not in predictions.columns:
            raise SurrogateError(
                f"the predictions frame has no {column!r} column; it holds "
                f"{list(predictions.columns)[:8]}")

    png = _read_png_list(db_path)
    if "png_path" not in png.columns:
        raise SurrogateError("png_list has no png_path column")

    preds = predictions[[path_column, prediction_column]].copy()
    preds.columns = ["png_path", "cv_prediction"]
    # Match on basename as well as full path: a model scored on one machine
    # and a database written on another agree about the file name and not
    # about the mount point.
    preds["_key"] = preds["png_path"].astype(str).map(os.path.basename)
    png["_key"] = png["png_path"].astype(str).map(os.path.basename)
    preds = preds.drop_duplicates("_key")
    png = png.drop_duplicates("_key")

    bridged = png.merge(preds[["_key", "cv_prediction"]], on="_key",
                        how="inner", validate="one_to_one")
    if bridged.empty:
        raise SurrogateError(
            "no crop in png_list matched a prediction. The predictions were "
            "probably made on a different dataset than this database "
            "describes.")

    features = _read_and_join_tables(db_path)
    if "prcfo" not in features.columns and features.index.name == "prcfo":
        features = features.reset_index()
    if "prcfo" not in features.columns or "prcfo" not in bridged.columns:
        raise SurrogateError(
            "png_list and the measurement tables share no 'prcfo' object "
            "key, so predictions cannot be attached to features.")

    joined = features.merge(bridged[["prcfo", "cv_prediction"]], on="prcfo",
                            how="inner", validate="one_to_one")
    if joined.empty:
        raise SurrogateError(
            "the predictions matched crops, but those crops matched no "
            "measured objects.")
    return joined


def _feature_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split the frame into usable features and the ones that would leak.

    :returns: ``(features, leaky)``.
    """
    numeric = frame.select_dtypes(include=[np.number]).columns
    features, leaky = [], []
    for column in numeric:
        if column == "cv_prediction":
            continue
        low = str(column).lower()
        if low in {c.lower() for c in _IDENTIFIER_COLUMNS}:
            continue
        if any(hint in low for hint in LEAKY_COLUMN_HINTS):
            leaky.append(column)
            continue
        features.append(column)
    return features, leaky


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_surrogate(frame: pd.DataFrame, *, test_size: float = 0.3,
                  n_estimators: int = 300, random_seed: int = 0,
                  n_repeats: int = 5, shap_max_samples: int = 500,
                  exclude: Optional[Sequence[str]] = None,
                  split_by: str = "well",
                  verbose: bool = True) -> SurrogateResult:
    """Fit a surrogate to ``frame['cv_prediction']`` and rank the features.

    :param frame: as :func:`build_surrogate_frame` returns.
    :param test_size: held-out fraction. Fidelity and permutation importance
        are both measured on it, never on the training rows.
    :param n_estimators: trees in the surrogate.
    :param random_seed: fixed, so a reported ranking can be reproduced.
    :param n_repeats: permutation repeats per feature.
    :param shap_max_samples: SHAP is O(rows); this caps the explained sample
        and the cap is REPORTED rather than applied silently.
    :param exclude: extra feature columns to drop.
    :param split_by: acquisition unit held intact between surrogate fitting
        and fidelity measurement. Default ``'well'``.
    :param verbose: print the summary when done.
    :returns: a :class:`SurrogateResult`.
    :raises SurrogateError: too few objects or classes to fit anything.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    if "cv_prediction" not in frame.columns:
        raise SurrogateError("frame has no cv_prediction column")

    warnings: List[str] = []
    features, leaky = _feature_columns(frame)
    if exclude:
        dropped = set(exclude)
        features = [f for f in features if f not in dropped]
    if leaky:
        warnings.append(
            f"dropped {len(leaky)} column(s) that would leak the answer "
            f"(a model's own scores predict its predictions perfectly and "
            f"explain nothing): {sorted(leaky)[:6]}")
    if not features:
        raise SurrogateError(
            "no usable numeric features left after dropping identifiers and "
            "leaky columns")

    data = frame[features + ["cv_prediction"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        raise SurrogateError(
            f"only {len(data)} objects have complete features; too few to "
            f"fit a surrogate worth reading")

    y = data["cv_prediction"]
    counts = y.value_counts().to_dict()
    if len(counts) < 2:
        raise SurrogateError(
            f"the CV model put every object in one class ({list(counts)}), "
            f"so there is nothing for a surrogate to separate")

    x = data[features]
    baseline = float(max(counts.values()) / len(data))

    from .classifier_evaluation import grouped_split, split_group_values
    split_frame = frame.loc[data.index].copy()
    try:
        split_level, groups = split_group_values(
            group_by=split_by, frame=split_frame, table="surrogate frame")
        train_idx, test_idx, split_report = grouped_split(
            groups, y.to_numpy(), test_size, seed=random_seed,
            group_by=split_level)
    except ValueError as exc:
        raise SurrogateError(str(exc)) from exc
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
    model.fit(x_train, y_train)
    fidelity = float(model.score(x_test, y_test))

    importance = pd.DataFrame({"feature": features})
    importance["gain"] = model.feature_importances_

    perm = permutation_importance(
        model, x_test, y_test, n_repeats=n_repeats,
        random_state=random_seed, n_jobs=-1)
    importance["permutation"] = perm.importances_mean

    shap_values = _shap_importance(model, x_test, shap_max_samples, warnings)
    if shap_values is not None:
        importance["shap"] = shap_values

    importance = importance.sort_values(
        "permutation", ascending=False).reset_index(drop=True)

    result = SurrogateResult(
        fidelity=fidelity, baseline=baseline, importance=importance,
        n_objects=int(len(data)), class_counts=counts,
        split_report=split_report.to_dict(), warnings=warnings)
    if verbose:
        print(result.summary())
    return result


def _shap_importance(model, x_test: pd.DataFrame, max_samples: int,
                     warnings: List[str]) -> Optional[np.ndarray]:
    """Mean absolute SHAP value per feature, or None when shap is unavailable.

    Optional on purpose: SHAP is the most informative of the three and the
    most expensive, and a missing optional dependency must cost the SHAP
    column rather than the whole analysis.
    """
    try:
        import shap
    except Exception:
        warnings.append(
            "shap is not installed, so no SHAP column. gain and permutation "
            "are still reported; `pip install shap` adds the third.")
        return None

    sample = x_test
    if len(sample) > max_samples:
        sample = sample.sample(max_samples, random_state=0)
        # Said out loud: a silently truncated sample reads as "explained
        # everything" when it did not.
        warnings.append(
            f"SHAP computed on {max_samples:,} of {len(x_test):,} held-out "
            f"objects (it is O(rows)); raise shap_max_samples for more.")
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample, check_additivity=False)
    except Exception as exc:
        warnings.append(f"SHAP failed ({type(exc).__name__}: {exc}); the "
                        f"gain and permutation columns are unaffected.")
        return None

    array = np.asarray(values)
    # shap returns (rows, features) for binary and (rows, features, classes)
    # -- or a list per class on older versions -- for multiclass. Average the
    # magnitude over everything that is not the feature axis.
    if array.ndim == 3:
        array = np.abs(array).mean(axis=(0, 2))
    elif array.ndim == 2:
        array = np.abs(array).mean(axis=0)
    else:
        warnings.append(
            f"unexpected SHAP output shape {array.shape}; column omitted")
        return None
    if array.shape[0] != x_test.shape[1]:
        warnings.append(
            f"SHAP returned {array.shape[0]} values for "
            f"{x_test.shape[1]} features; column omitted")
        return None
    return array


# ---------------------------------------------------------------------------
# The one call most people want
# ---------------------------------------------------------------------------

def explain_classifier(db_path: str, predictions: pd.DataFrame, *,
                       path_column: str = "path",
                       prediction_column: str = "pred",
                       **fit_kwargs: Any) -> SurrogateResult:
    """Join predictions to features, fit a surrogate, and rank the features.

    :param db_path: the measurements database.
    :param predictions: a frame with a crop path and a predicted class.
    :param path_column: column holding the crop path.
    :param prediction_column: column holding the predicted class.
    :param fit_kwargs: forwarded to :func:`fit_surrogate`.
    :returns: a :class:`SurrogateResult`.
    """
    frame = build_surrogate_frame(
        db_path, predictions, path_column=path_column,
        prediction_column=prediction_column)
    return fit_surrogate(frame, **fit_kwargs)
