"""Item 808: the default classifier is regularised and survives wide tables."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spacr.classify import ML_MODEL_TYPES, resolve_ml_model_type
from spacr.ml import _regularised_linear_model
from spacr.settings import set_default_analyze_screen


def test_default_is_standardised_logistic_regression():
    assert resolve_ml_model_type({}) == "logistic_regression"
    assert set_default_analyze_screen({})["model_type_ml"] == "logistic_regression"
    model = _regularised_linear_model()
    assert isinstance(model, Pipeline)
    assert isinstance(model[0], StandardScaler)
    assert isinstance(model[-1], LogisticRegression)
    assert model[-1].C == 1.0  # default L2 strength, not C=inf


def test_shrinkage_lda_selectable_and_old_choices_kept():
    assert resolve_ml_model_type({"model_type_ml": "shrinkage_lda"}) == "shrinkage_lda"
    assert {"xgboost", "random_forest", "svm"} <= set(ML_MODEL_TYPES)
    lda = _regularised_linear_model("shrinkage_lda")[-1]
    assert isinstance(lda, LinearDiscriminantAnalysis)
    assert lda.solver == "lsqr" and lda.shrinkage == "auto"


def _wide(n_train=60, n_features=80, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_train // 2)
    X = rng.normal(size=(n_train, n_features))
    X[:, :10] += y[:, None] * 1.0
    X[:, 10:20] *= 1000.0  # large-scale columns
    return X, y


def test_default_and_shrinkage_beat_chance_with_more_features_than_samples():
    X, y = _wide()
    Xt, yt = _wide(n_train=400, seed=1)
    for kind in ("logistic_regression", "shrinkage_lda"):
        model = _regularised_linear_model(kind).fit(X, y)
        assert model.score(Xt, yt) > 0.75, kind
