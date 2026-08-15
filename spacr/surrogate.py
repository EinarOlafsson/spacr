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

import importlib
import importlib.metadata
import json
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
    "SURROGATE_BACKENDS",
    "available_backends",
    "write_surrogate_result",
    "explain_cv_default_settings",
    "run_explain_cv",
]

APP_KEY = "explain_cv"

SURROGATE_BACKENDS: Tuple[str, ...] = (
    "random_forest", "hist_gradient_boosting", "xgboost")
MODEL_FAMILIES: Dict[str, str] = {
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "xgboost": "XGBoost",
}

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
    model_family: str = "random_forest"
    backend: str = "sklearn.ensemble.RandomForestClassifier"
    backend_version: str = ""
    random_seed: int = 0
    model_params: Dict[str, Any] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    excluded_columns: List[str] = field(default_factory=list)
    balanced_accuracy: float = float("nan")
    f1_macro: float = float("nan")
    class_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    confusion: pd.DataFrame = field(default_factory=pd.DataFrame)
    held_out: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap_feature_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlated_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_distributions: pd.DataFrame = field(default_factory=pd.DataFrame)
    family_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    model: Any = field(default=None, repr=False)
    minimum_fidelity_improvement: float = 0.05

    @property
    def is_faithful(self) -> bool:
        """Whether the surrogate beats the majority-class baseline at all.

        Deliberately a low bar. It is not "this explanation is good"; it is
        "this explanation is about the CV model rather than about noise".
        """
        return self.fidelity >= self.baseline + self.minimum_fidelity_improvement

    @property
    def fidelity_improvement(self) -> float:
        """Accuracy improvement over the majority-class baseline."""
        return float(self.fidelity - self.baseline)

    def top(self, n: int = 15, by: str = "permutation") -> pd.DataFrame:
        """The ``n`` most important features by one measure.

        :param n: how many rows.
        :param by: ``'permutation'``, ``'shap'`` or ``'gain'``. Falls back to
            whichever was computed, rather than raising, so a run without
            SHAP still reports something.
        """
        if not self.is_faithful:
            return self.importance.iloc[0:0].copy()
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
            f"improvement        : {self.fidelity_improvement:+.3f}",
            f"backend            : {self.backend} {self.backend_version}".rstrip(),
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
        if self.is_faithful:
            lines += ["", f"Top {n} features:",
                      self.top(n).to_string(index=False)]
        else:
            lines += ["", f"Top {n} features: withheld",
                      "Feature importances are not presented because the "
                      "surrogate did not clear the fidelity gate."]
        return "\n".join(lines)


def available_backends() -> Dict[str, Dict[str, Any]]:
    """Report the supported surrogate families and whether they can run.

    XGBoost is optional and is never silently replaced by another estimator.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for name, distribution in (
            ("random_forest", "scikit-learn"),
            ("hist_gradient_boosting", "scikit-learn"),
            ("xgboost", "xgboost")):
        try:
            version = importlib.metadata.version(distribution)
            available = True
            reason = ""
        except importlib.metadata.PackageNotFoundError:
            version = ""
            available = False
            reason = f"install {distribution} to enable this backend"
        result[name] = {
            "available": available, "version": version, "reason": reason}
    return result


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

def available_model_families() -> Dict[str, Dict[str, Any]]:
    """Return every fixed surrogate choice and whether it can run here.

    XGBoost is optional and remains visible when absent so the GUI can disable
    it with an explanation. It must never silently become Random Forest under
    an XGBoost label.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for key, label in MODEL_FAMILIES.items():
        available = key != "xgboost" or importlib.util.find_spec("xgboost") is not None
        result[key] = {
            "label": label,
            "available": available,
            "reason": "" if available else "Install the optional xgboost package.",
        }
    return result


def _package_version(name: str) -> str:
    """Installed distribution version, or an empty string."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _make_estimator(model_family: str, *, n_estimators: int,
                    random_seed: int, model_options: Mapping[str, Any],
                    y_train: pd.Series):
    """Create one explicitly requested estimator and its label decoder."""
    family = str(model_family).strip().lower()
    if family not in MODEL_FAMILIES:
        raise SurrogateError(
            f"unknown surrogate model {model_family!r}; choose one of "
            f"{list(MODEL_FAMILIES)}")

    options = dict(model_options)
    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        params = {
            "n_estimators": int(n_estimators),
            "random_state": int(random_seed),
            "n_jobs": -1,
        }
        params.update(options)
        model = RandomForestClassifier(**params)
        return model, y_train, lambda values: np.asarray(values), (
            "sklearn.ensemble.RandomForestClassifier", _package_version("scikit-learn"), params)

    if family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = {
            "max_iter": int(n_estimators),
            "random_state": int(random_seed),
        }
        params.update(options)
        model = HistGradientBoostingClassifier(**params)
        return model, y_train, lambda values: np.asarray(values), (
            "sklearn.ensemble.HistGradientBoostingClassifier",
            _package_version("scikit-learn"), params)

    if not available_model_families()["xgboost"]["available"]:
        raise SurrogateError(
            "XGBoost was selected but the optional xgboost package is not "
            "installed. Choose another model or install xgboost explicitly.")
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise SurrogateError(
            f"XGBoost is installed but could not be imported: {exc}") from exc
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y_train)
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": int(random_seed),
        "n_jobs": 1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    params.update(options)
    model = XGBClassifier(**params)
    model._spacr_label_encoder = encoder
    return model, encoded, encoder.inverse_transform, (
        "xgboost.XGBClassifier", _package_version("xgboost"), params)


def _correlation_pairs(frame: pd.DataFrame, threshold: float,
                       warnings: List[str], max_features: int = 500
                       ) -> pd.DataFrame:
    """Strong Spearman feature pairs, so importance sharing is visible."""
    columns = list(frame.columns)
    if len(columns) > max_features:
        variances = frame.var(numeric_only=True).sort_values(ascending=False)
        columns = list(variances.head(max_features).index)
        warnings.append(
            f"correlation audit used the {max_features} highest-variance of "
            f"{frame.shape[1]} features")
    if len(columns) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "spearman"])
    corr = frame[columns].corr(method="spearman")
    rows = []
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            value = corr.at[first, second]
            if pd.notna(value) and abs(float(value)) >= threshold:
                rows.append((first, second, float(value)))
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "spearman"])


def _feature_family(feature: str) -> str:
    """Return a conservative biological/object family from a feature name."""
    low = str(feature).lower()
    for family in ("cell", "nucleus", "pathogen", "cytoplasm", "organelle"):
        if low == family or low.startswith(f"{family}_"):
            return family
    if "texture" in low:
        return "texture"
    if "intensity" in low:
        return "intensity"
    if "distance" in low or "location" in low:
        return "spatial"
    return "other"


def _distribution_table(values: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Long held-out feature summaries stratified by the CV decision."""
    rows = []
    aligned = labels.reindex(values.index)
    for feature in values.columns:
        for label, series in values[feature].groupby(aligned, dropna=False):
            rows.append({
                "feature": feature, "cv_class": label,
                "n": int(series.notna().sum()), "mean": float(series.mean()),
                "median": float(series.median()), "std": float(series.std()),
            })
    return pd.DataFrame(rows)

def fit_surrogate(frame: pd.DataFrame, *, test_size: float = 0.3,
                  n_estimators: int = 300, random_seed: int = 0,
                  n_repeats: int = 5, shap_max_samples: int = 500,
                  exclude: Optional[Sequence[str]] = None,
                  split_by: str = "well",
                  model_family: str = "random_forest",
                  model_options: Optional[Mapping[str, Any]] = None,
                  correlation_threshold: float = 0.9,
                  minimum_fidelity_improvement: float = 0.05,
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
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)

    if "cv_prediction" not in frame.columns:
        raise SurrogateError("frame has no cv_prediction column")

    warnings: List[str] = []
    features, leaky = _feature_columns(frame)
    explicitly_excluded = list(exclude or ())
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

    model, fit_y, decode, backend_info = _make_estimator(
        model_family, n_estimators=n_estimators, random_seed=random_seed,
        model_options=model_options or {}, y_train=y_train)
    from .openmp_guard import guarded_n_jobs, single_threaded_openmp
    with single_threaded_openmp("surrogate model explanation"):
        model.fit(x_train, fit_y)
        predicted = decode(model.predict(x_test))
        fidelity = float(np.mean(np.asarray(predicted) == y_test.to_numpy()))

    importance = pd.DataFrame({"feature": features})
    if hasattr(model, "feature_importances_"):
        importance["gain"] = np.asarray(model.feature_importances_, dtype=float)
    else:
        warnings.append(
            f"{MODEL_FAMILIES[str(model_family).lower()]} has no native gain "
            "importance; held-out permutation and SHAP remain available")

    if str(model_family).lower() == "xgboost":
        # The estimator was fitted on encoded classes, so its scorer must see
        # those same labels. Displayed predictions are decoded above.
        encoded_test = pd.Series(
            model._spacr_label_encoder.transform(y_test), index=y_test.index)
        with single_threaded_openmp("surrogate permutation importance"):
            perm = permutation_importance(
                model, x_test, encoded_test, n_repeats=n_repeats,
                random_state=random_seed,
                n_jobs=guarded_n_jobs(-1, "surrogate permutation importance"))
    else:
        with single_threaded_openmp("surrogate permutation importance"):
            perm = permutation_importance(
                model, x_test, y_test, n_repeats=n_repeats,
                random_state=random_seed,
                n_jobs=guarded_n_jobs(-1, "surrogate permutation importance"))
    importance["permutation"] = perm.importances_mean

    shap_output = _shap_importance(
        model, x_test, shap_max_samples, warnings, return_details=True)
    signed_shap = pd.DataFrame()
    shap_feature_values = pd.DataFrame()
    if shap_output is not None:
        if isinstance(shap_output, tuple):
            if len(shap_output) == 3:
                shap_importance, signed_shap, shap_feature_values = shap_output
            else:  # compatibility with callers replacing the helper
                shap_importance, signed_shap = shap_output
        else:  # compatibility with callers/tests replacing this helper
            shap_importance = shap_output
        importance["shap"] = shap_importance

    importance = importance.sort_values(
        "permutation", ascending=False).reset_index(drop=True)
    importance["feature_family"] = importance["feature"].map(_feature_family)
    importance_columns = [column for column in ("gain", "permutation", "shap")
                          if column in importance]
    family_importance = (
        importance.groupby("feature_family", as_index=False)[importance_columns]
        .sum(numeric_only=True)
        .sort_values(
            "permutation" if "permutation" in importance_columns
            else importance_columns[0], ascending=False)
        if importance_columns else pd.DataFrame())

    labels = list(pd.unique(pd.concat([y_train, y_test], ignore_index=True)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=labels, zero_division=0)
    class_metrics = pd.DataFrame({
        "class": labels, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    }).set_index("class")
    confusion = pd.DataFrame(
        confusion_matrix(y_test, predicted, labels=labels),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    identity_columns = [
        c for c in _IDENTIFIER_COLUMNS if c in split_frame.columns
    ]
    held_out = split_frame.iloc[test_idx][identity_columns].copy()
    held_out["cv_prediction"] = y_test.to_numpy()
    held_out["surrogate_prediction"] = np.asarray(predicted)
    if hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(model.predict_proba(x_test))
            if probabilities.ndim == 2:
                probability_labels = list(getattr(model, "classes_", labels))
                if str(model_family).lower() == "xgboost":
                    probability_labels = list(
                        model._spacr_label_encoder.inverse_transform(
                            np.asarray(probability_labels, dtype=int)))
                for index, label in enumerate(probability_labels):
                    held_out[f"probability_{label}"] = probabilities[:, index]
        except Exception as exc:
            warnings.append(f"held-out probabilities unavailable: {exc}")

    result = SurrogateResult(
        fidelity=fidelity, baseline=baseline, importance=importance,
        n_objects=int(len(data)), class_counts=counts,
        split_report=split_report.to_dict(), warnings=warnings,
        model_family=str(model_family).lower(), backend=backend_info[0],
        backend_version=backend_info[1], model_params=backend_info[2],
        feature_columns=list(features),
        excluded_columns=sorted(set(leaky + explicitly_excluded)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predicted)),
        f1_macro=float(f1_score(y_test, predicted, average="macro")),
        class_metrics=class_metrics, confusion=confusion,
        held_out=held_out, shap_values=signed_shap,
        shap_feature_values=shap_feature_values,
        correlated_features=_correlation_pairs(
            x_train, float(correlation_threshold), warnings),
        feature_distributions=_distribution_table(x_test, y_test),
        family_importance=family_importance,
        model=model, random_seed=int(random_seed),
        minimum_fidelity_improvement=float(minimum_fidelity_improvement))
    if verbose:
        print(result.summary())
    return result


def _shap_importance(model, x_test: pd.DataFrame, max_samples: int,
                     warnings: List[str], *, return_details: bool = False):
    """Mean absolute SHAP plus optional signed per-object values.

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
        try:
            values = explainer.shap_values(sample, check_additivity=False)
        except TypeError:
            values = explainer.shap_values(sample)
    except Exception as exc:
        warnings.append(f"SHAP failed ({type(exc).__name__}: {exc}); the "
                        f"gain and permutation columns are unaffected.")
        return None

    # Older SHAP returns a list of (rows, features), one per class. Newer
    # releases return (rows, features, classes). Normalise both before asking
    # which axis holds features; averaging the old list-shaped array over the
    # wrong axes produces one importance per ROW and used to be silently
    # accepted whenever rows happened to equal features.
    if isinstance(values, list):
        arrays = [np.asarray(value) for value in values]
        if not arrays or any(value.ndim != 2 for value in arrays):
            warnings.append("unexpected list-shaped SHAP output; column omitted")
            return None
        array = np.stack(arrays, axis=2)
    else:
        array = np.asarray(values)

    if array.ndim == 2 and array.shape == sample.shape:
        signed = array
        importance = np.abs(array).mean(axis=0)
    elif array.ndim == 3:
        if array.shape[:2] == sample.shape:
            normalised = array
        elif array.shape[1:] == sample.shape:
            normalised = np.moveaxis(array, 0, 2)
        else:
            warnings.append(
                f"unexpected SHAP output shape {array.shape} for sample "
                f"{sample.shape}; column omitted")
            return None
        importance = np.abs(normalised).mean(axis=(0, 2))
        try:
            if hasattr(model, "predict_proba"):
                chosen = np.asarray(model.predict_proba(sample)).argmax(axis=1)
            else:
                chosen = np.zeros(len(sample), dtype=int)
            signed = normalised[np.arange(len(sample)), :, chosen]
        except Exception:
            signed = normalised[:, :, 0]
    else:
        warnings.append(
            f"unexpected SHAP output shape {array.shape}; column omitted")
        return None
    if importance.shape[0] != x_test.shape[1]:
        warnings.append(
            f"SHAP returned {importance.shape[0]} values for "
            f"{x_test.shape[1]} features; column omitted")
        return None
    if not return_details:
        return importance
    details = pd.DataFrame(signed, index=sample.index, columns=sample.columns)
    details.index.name = "object_index"
    return importance, details, sample.copy()


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


def write_surrogate_result(result: SurrogateResult,
                           destination: str) -> Dict[str, str]:
    """Write the complete, provenance-bearing surrogate result bundle.

    Importance tables are always retained as source data, but plots are only
    presented when the held-out fidelity clears the declared improvement gate.

    :param result: fitted :class:`SurrogateResult`.
    :param destination: new or existing output directory.
    :returns: mapping of artifact role to absolute path.
    """
    root = os.path.abspath(os.path.expanduser(destination))
    os.makedirs(root, exist_ok=True)
    paths: Dict[str, str] = {}

    def _csv(role: str, name: str, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        path = os.path.join(root, name)
        frame.to_csv(path, index=True)
        paths[role] = path

    _csv("importance", "feature_importance.csv", result.importance)
    _csv("class_metrics", "class_metrics.csv", result.class_metrics)
    _csv("confusion", "confusion_matrix.csv", result.confusion)
    _csv("held_out", "held_out_predictions.csv", result.held_out)
    _csv("local_shap", "held_out_signed_shap.csv", result.shap_values)
    _csv("shap_feature_values", "held_out_shap_feature_values.csv",
         result.shap_feature_values)
    _csv("correlations", "correlated_feature_pairs.csv",
         result.correlated_features)
    _csv("feature_distributions", "held_out_feature_distributions.csv",
         result.feature_distributions)
    _csv("family_importance", "feature_family_importance.csv",
         result.family_importance)
    summary_path = os.path.join(root, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(result.summary())
        handle.write("\n")
    paths["summary"] = summary_path
    manifest = {
        "model_family": result.model_family,
        "backend": result.backend,
        "backend_version": result.backend_version,
        "model_parameters": result.model_params,
        "random_seed": result.random_seed,
        "n_objects": result.n_objects,
        "class_counts": result.class_counts,
        "fidelity": result.fidelity,
        "majority_baseline": result.baseline,
        "fidelity_improvement": result.fidelity_improvement,
        "minimum_fidelity_improvement": result.minimum_fidelity_improvement,
        "importance_presented": result.is_faithful,
        "balanced_accuracy": result.balanced_accuracy,
        "f1_macro": result.f1_macro,
        "split_report": result.split_report,
        "feature_columns": result.feature_columns,
        "excluded_columns": result.excluded_columns,
        "warnings": result.warnings,
        "shap_object_indices": (
            result.shap_values.index.astype(str).tolist()
            if not result.shap_values.empty else []),
    }
    manifest_path = os.path.join(root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
    paths["manifest"] = manifest_path

    if result.is_faithful and not result.importance.empty:
        import matplotlib.pyplot as plt
        measures = [column for column in ("gain", "permutation", "shap")
                    if column in result.importance and
                    result.importance[column].notna().any()]
        if measures:
            fig, axes = plt.subplots(
                1, len(measures), figsize=(5.2 * len(measures), 5.0),
                squeeze=False)
            for axis, measure in zip(axes[0], measures):
                top = result.importance.nlargest(15, measure).sort_values(measure)
                axis.barh(top["feature"].astype(str), top[measure],
                          color="#3B82C4")
                axis.set_title(measure.capitalize())
                axis.set_xlabel("importance")
                axis.spines[["top", "right"]].set_visible(False)
            fig.suptitle(
                f"Held-out fidelity {result.fidelity:.3f} "
                f"(baseline {result.baseline:.3f})")
            fig.tight_layout()
            for extension in ("pdf", "png"):
                path = os.path.join(root, f"feature_importance.{extension}")
                fig.savefig(path, dpi=300 if extension == "png" else None,
                            bbox_inches="tight")
                paths[f"importance_{extension}"] = path
            plt.close(fig)
    if (result.is_faithful and not result.shap_values.empty
            and not result.shap_feature_values.empty):
        import matplotlib.pyplot as plt
        ranked = (result.importance.nlargest(6, "shap")["feature"].tolist()
                  if "shap" in result.importance else
                  list(result.shap_values.columns[:6]))
        ranked = [feature for feature in ranked
                  if feature in result.shap_values
                  and feature in result.shap_feature_values]
        if ranked:
            columns = min(3, len(ranked))
            rows = int(np.ceil(len(ranked) / columns))
            fig, axes = plt.subplots(
                rows, columns, figsize=(4.6 * columns, 3.7 * rows),
                squeeze=False)
            for axis, feature in zip(axes.ravel(), ranked):
                axis.scatter(
                    result.shap_feature_values[feature],
                    result.shap_values[feature], s=12, alpha=0.6,
                    color="#3B82C4", edgecolors="none")
                axis.axhline(0, color="#777777", linewidth=0.7)
                axis.set_xlabel(feature)
                axis.set_ylabel("signed SHAP")
                axis.spines[["top", "right"]].set_visible(False)
            for axis in axes.ravel()[len(ranked):]:
                axis.set_visible(False)
            fig.suptitle("Held-out SHAP dependence")
            fig.tight_layout()
            for extension in ("pdf", "png"):
                path = os.path.join(root, f"shap_dependence.{extension}")
                fig.savefig(path, dpi=300 if extension == "png" else None,
                            bbox_inches="tight")
                paths[f"shap_dependence_{extension}"] = path
            plt.close(fig)
    return paths


def explain_cv_default_settings(settings=None) -> Dict[str, Any]:
    """Defaults for the first-class Explain CV Model module."""
    configured = dict(settings or {})
    configured.setdefault("db_path", "")
    configured.setdefault("predictions_file", "")
    configured.setdefault("path_column", "path")
    configured.setdefault("prediction_column", "pred")
    configured.setdefault("surrogate_model", "random_forest")
    configured.setdefault("surrogate_split_by", "well")
    configured.setdefault("surrogate_test_size", 0.3)
    configured.setdefault("surrogate_n_estimators", 300)
    configured.setdefault("surrogate_n_repeats", 5)
    configured.setdefault("surrogate_shap_max_samples", 500)
    configured.setdefault("surrogate_random_seed", 0)
    configured.setdefault("surrogate_exclude", [])
    configured.setdefault("surrogate_correlation_threshold", 0.9)
    configured.setdefault("surrogate_min_fidelity_improvement", 0.05)
    configured.setdefault("dst", "")
    configured.setdefault("verbose", True)
    return configured


def run_explain_cv(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Explain an existing CV prediction file without rerunning its model.

    :param settings: mapping returned by :func:`explain_cv_default_settings`.
        ``db_path`` is the exact measurements database and
        ``predictions_file`` is an existing per-object prediction CSV.
    :returns: ``{'result': SurrogateResult, 'paths': artifact mapping}``.
    :raises SurrogateError: when either source is missing or the requested
        optional backend cannot run.
    """
    configured = explain_cv_default_settings(settings)
    db_path = os.path.abspath(os.path.expanduser(str(configured["db_path"])))
    predictions_path = os.path.abspath(os.path.expanduser(
        str(configured["predictions_file"])))
    if not os.path.isfile(db_path):
        raise SurrogateError(f"no measurements database at {db_path}")
    if not os.path.isfile(predictions_path):
        raise SurrogateError(f"no existing prediction CSV at {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    result = explain_classifier(
        db_path, predictions,
        path_column=str(configured["path_column"]),
        prediction_column=str(configured["prediction_column"]),
        model_family=str(configured["surrogate_model"]),
        split_by=str(configured["surrogate_split_by"]),
        test_size=float(configured["surrogate_test_size"]),
        n_estimators=int(configured["surrogate_n_estimators"]),
        n_repeats=int(configured["surrogate_n_repeats"]),
        shap_max_samples=int(configured["surrogate_shap_max_samples"]),
        random_seed=int(configured["surrogate_random_seed"]),
        exclude=list(configured["surrogate_exclude"]),
        correlation_threshold=float(
            configured["surrogate_correlation_threshold"]),
        minimum_fidelity_improvement=float(
            configured["surrogate_min_fidelity_improvement"]),
        verbose=bool(configured["verbose"]),
    )
    destination = str(configured.get("dst") or "").strip()
    if not destination:
        destination = os.path.join(
            os.path.dirname(predictions_path), "explain_cv_model")
    paths = write_surrogate_result(result, destination)
    return {"result": result, "paths": paths}


def register_explain_cv_settings(replace: bool = False) -> bool:
    """Register Explain CV Model's settings with both desktop front ends."""
    from .settings import (has_registered_defaults, register_defaults,
                           tooltips as shared_tooltips)
    if has_registered_defaults(APP_KEY) and not replace:
        return False
    tooltips = {
        "db_path": "(str) - Exact measurements.db whose objects the prediction file scored. Choosing another database is refused when crop identities do not match, preventing cross-experiment explanations.",
        "predictions_file": "(str) - Existing per-object CV prediction CSV joined to measured objects. The module never reruns the vision model or substitutes scores from another run.",
        "path_column": "(str) - Column in the prediction CSV containing crop paths used for the one-to-one object join. Change it only when the exporter used another name. Default path.",
        "prediction_column": "(str) - Column containing the CV class the surrogate must reproduce. Selecting a score instead changes the learning target and meaning of fidelity. Default pred.",
        "surrogate_model": "(str) - Estimator family used to reproduce CV decisions: Random Forest, histogram gradient boosting, or optional XGBoost. Missing XGBoost is refused rather than silently substituted. Default random_forest.",
        "surrogate_split_by": "(str) - Independent acquisition unit kept intact between fitting and fidelity evaluation. Use well by default, or plate for a harder batch-generalization test.",
        "surrogate_test_size": "(float) - Fraction of independent well or plate groups held out for fidelity and permutation importance. Larger values strengthen evaluation but leave fewer groups for fitting. Default 0.3.",
        "surrogate_n_estimators": "(int) - Number of trees or boosting iterations in the selected surrogate. Raising it can improve fidelity but increases runtime and may overfit small training sets. Default 300.",
        "surrogate_n_repeats": "(int) - Repeated held-out shuffles per feature for permutation importance. More repeats stabilize rankings but multiply scoring time without changing the fitted model. Default 5.",
        "surrogate_shap_max_samples": "(int) - Maximum held-out objects receiving signed SHAP values and dependence plots. Raising it improves coverage at substantial computational cost; sampled IDs are recorded. Default 500.",
        "surrogate_random_seed": "(int) - Seed shared by grouped splitting, estimator fitting and importance calculations. Keep it fixed for exact reproduction and change it only for sensitivity checks. Default 0.",
        "surrogate_exclude": "(list) - Additional measured features barred from the explanatory matrix. Use it for known artifacts or post-treatment annotations; model outputs, classes and identifiers remain excluded automatically. Default [].",
        "surrogate_correlation_threshold": "(float) - Absolute Spearman correlation above which a held-out feature pair is disclosed. Lower values reveal more redundancy and produce a larger audit table. Default 0.9.",
        "surrogate_min_fidelity_improvement": "(float) - Accuracy improvement over the majority-class baseline required before feature importances are presented as explanations. Raising it withholds more weak surrogates. Default 0.05.",
        "dst": "(str) - Folder receiving versioned tables, manifests and figures. Leaving it blank uses a module-specific folder beside the primary input, keeping different analyses separated. Default blank.",
    }
    tooltips = {key: value for key, value in tooltips.items()
                if key not in shared_tooltips}
    types = {
        "db_path": str, "predictions_file": str, "path_column": str,
        "prediction_column": str, "surrogate_model": str,
        "surrogate_split_by": str, "surrogate_test_size": (int, float),
        "surrogate_n_estimators": int, "surrogate_n_repeats": int,
        "surrogate_shap_max_samples": int, "surrogate_random_seed": int,
        "surrogate_exclude": list,
        "surrogate_correlation_threshold": (int, float),
        "surrogate_min_fidelity_improvement": (int, float),
        "dst": str,
    }
    register_defaults(
        APP_KEY, explain_cv_default_settings, replace=replace,
        expected_types=types, tooltips=tooltips,
        description=(
            "Fit a well-grouped measured-feature surrogate to an existing CV "
            "prediction output, report fidelity before gain/permutation/SHAP, "
            "and preserve disagreements and leakage exclusions."))
    return True


register_explain_cv_settings()
