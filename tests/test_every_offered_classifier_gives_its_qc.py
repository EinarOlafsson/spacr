"""Every classifier the panel offers fits, scores, and explains itself.

Instruction 236 A1 and A3. ``model_type_ml`` offers nine models, and the
question is not whether the settings table lists them -- it is whether a
user who picks one gets a fitted model and the QC panels the module
promises.

FOUR OF THE NINE HAD NO FEATURE IMPORTANCE AT ALL. gradient_boosting,
logistic_regression, svm and mlp expose no ``feature_importances_``, and
the branch that noticed handed back an empty frame and no figure. A user
who picked logistic_regression -- which the setting's own tooltip
recommends as "a good linear sanity check" -- lost that QC panel and was
told nothing about it.

The permutation importance is computed for every model a few lines
earlier, because it is model-agnostic by construction, so the data to fill
the gap was already in hand. It is a DIFFERENT QUANTITY -- what the fitted
model loses when a column is shuffled, rather than how often a tree split
on it -- so the panel says which one it is drawing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


#: Every model the setting offers, and whether it brings its own package.
OFFERED = [
    ("xgboost", "xgboost"),
    ("random_forest", None),
    ("extra_trees", None),
    ("gradient_boosting", None),
    ("logistic_regression", None),
    ("svm", None),
    ("mlp", None),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
]

#: Models with no ``feature_importances_`` of their own.
NO_NATIVE_IMPORTANCE = {"gradient_boosting", "logistic_regression", "svm",
                        "mlp"}


def _controls(rows=160, features=12, seed=0):
    """A separable two-control frame with the identity the splitter needs.

    THE INDEX IS THE OBJECT IDENTITY. `ml_analysis` reads `prcfo` off the
    frame's index, and a plain integer index is refused by the well-splitter
    -- correctly, because a row it cannot place in a well cannot be kept out
    of the training half.
    """
    rng = np.random.default_rng(seed)
    half = rows // 2
    frame = pd.DataFrame(
        rng.normal(size=(rows, features)),
        columns=[f"cell_channel_1_feature_{i}" for i in range(features)])
    frame["columnID"] = ["c1"] * half + ["c2"] * half
    frame["rowID"] = [f"r{1 + i % 8}" for i in range(rows)]
    frame["plateID"] = "plate1"
    frame["fieldID"] = [f"f{1 + i % 4}" for i in range(rows)]
    frame["object_label"] = [str(i) for i in range(rows)]
    # The signal: c2 is shifted, so every model has something to find.
    frame.loc[frame["columnID"] == "c2",
              frame.columns[:features]] += 2.5
    frame.index = [
        f"plate1_{frame['rowID'][i]}_{frame['columnID'][i]}_"
        f"{frame['fieldID'][i]}_o{i}" for i in range(rows)]
    return frame


def _run(model_type):
    from spacr.ml import ml_analysis

    return ml_analysis(
        _controls(), channel_of_interest=1, location_column="columnID",
        positive_control="c2", negative_control="c1", n_repeats=2,
        top_features=6, n_estimators=12, model_type=model_type, n_jobs=1,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False, verbose=False)


@pytest.mark.parametrize("model_type,package",
                         OFFERED, ids=[m for m, _ in OFFERED])
def test_every_offered_model_fits_and_scores(model_type, package):
    if package:
        pytest.importorskip(package)
    output, _figures = _run(model_type)
    scored = output[0]
    assert len(scored) > 0
    assert "prcfo" in scored.columns


@pytest.mark.parametrize("model_type,package",
                         OFFERED, ids=[m for m, _ in OFFERED])
def test_every_offered_model_explains_itself(model_type, package):
    """Both QC panels, for every model -- not only the tree-shaped ones."""
    if package:
        pytest.importorskip(package)
    output, figures = _run(model_type)
    permutation_fig, importance_fig = figures
    assert permutation_fig is not None, "no permutation importance panel"
    assert importance_fig is not None, (
        f"{model_type} produced no feature-importance panel; a user who "
        f"picks it loses that QC and is told nothing")
    assert not output[2].empty, "the importance table is empty"


@pytest.mark.parametrize("model_type", sorted(NO_NATIVE_IMPORTANCE))
def test_a_model_without_native_importances_says_what_it_drew(model_type):
    """Permutation importance is not split-gain importance, and a panel
    that did not say so would be passing one off as the other."""
    _output, figures = _run(model_type)
    title = figures[1].axes[0].get_title().lower()
    assert "permutation" in title, title


def test_a_tree_model_still_draws_its_own_importances():
    """The fallback must not have replaced the native quantity where there
    is one: a random forest's split-gain importance is what a reader of
    that panel expects to be looking at."""
    _output, figures = _run("random_forest")
    assert "permutation" not in figures[1].axes[0].get_title().lower()


def test_the_offered_list_matches_what_the_setting_advertises():
    """A model in the tooltip that the code cannot build is a promise the
    panel does not keep -- and one the code builds that the tooltip omits
    is a model nobody will find."""
    from spacr.settings import tooltips

    said = tooltips["model_type_ml"]
    for model_type, _package in OFFERED:
        assert model_type in said, f"{model_type} is offered but undocumented"


def test_an_unknown_model_is_refused_by_name():
    """Not silently, and not with a bare KeyError: the message has to name
    what was asked for."""
    from spacr.ml import ml_analysis

    with pytest.raises(ValueError, match="quantum_forest"):
        ml_analysis(_controls(), model_type="quantum_forest",
                    location_column="columnID", positive_control="c2",
                    negative_control="c1", n_estimators=4, n_repeats=1)


# ---------------------------------------------------------------------------
# SHAP, for every one of them
# ---------------------------------------------------------------------------
#: Models `shap.Explainer(model, X)` cannot explain on its own, and why.
#:
#: THREE OF THE NINE, including the panel's DEFAULT. Driven rather than
#: reasoned about (instruction 236 A3):
#:
#: * xgboost raised "Categorical split is not yet supported. You can still
#:   use TreeExplainer with feature_perturbation=tree_path_dependent" -- an
#:   error carrying its own fix, and NOT at construction: the explainer was
#:   built happily and raised when it was called, so a fallback chosen at
#:   construction time would never have run.
#: * svm and mlp raised "The passed model is not callable and cannot be
#:   analyzed directly with the given masker". A support vector machine and
#:   a neural net are neither trees nor linear, and the model-agnostic
#:   explainer takes a FUNCTION rather than an estimator.
NEEDED_A_DIFFERENT_EXPLAINER = ("xgboost", "svm", "mlp")


@pytest.mark.parametrize("model_type,package",
                         OFFERED, ids=[m for m, _ in OFFERED])
def test_every_offered_model_can_be_explained(model_type, package):
    """SHAP is a QC step the module advertises for the classifier, and a
    classifier that cannot be explained has silently lost it."""
    if package:
        pytest.importorskip(package)
    pytest.importorskip("shap")
    from spacr.ml import shap_analysis

    output, _figures = _run(model_type)
    model, x_train, x_test = output[3], output[4], output[5]
    plot = shap_analysis(model, x_train, x_test)
    assert plot is not None, f"{model_type} produced no SHAP panel"


@pytest.mark.parametrize("model_type", NEEDED_A_DIFFERENT_EXPLAINER)
def test_a_model_needing_another_explainer_says_which_one(model_type,
                                                          capsys):
    """The three explainers do not compute the same quantity --
    `tree_path_dependent` conditions on the tree's own splits, and the
    model-agnostic one ESTIMATES rather than solves -- so a panel that
    switched silently would be passing one off as another."""
    if model_type == "xgboost":
        pytest.importorskip("xgboost")
    pytest.importorskip("shap")
    from spacr.ml import shap_analysis

    output, _figures = _run(model_type)
    shap_analysis(output[3], output[4], output[5])
    said = capsys.readouterr().out.lower()
    assert "shap:" in said, "it switched explainers without saying so"


def test_a_tree_model_is_not_pushed_onto_the_slow_path():
    """The model-agnostic explainer is O(background x features) per row.
    Using it where an exact tree explainer works would turn a QC panel into
    a coffee break."""
    pytest.importorskip("shap")
    from spacr.ml import _shap_explainers

    output, _figures = _run("random_forest")
    first, note = next(iter(_shap_explainers(output[3], output[4])))
    assert note == "", note


def test_a_model_nothing_can_explain_fails_by_name():
    """Not with SHAP's own message about maskers, which names neither the
    model nor the panel it was for."""
    pytest.importorskip("shap")
    from spacr.ml import _shap_values

    class Opaque:
        """No predict, no predict_proba, no tree."""

    with pytest.raises(RuntimeError, match="Opaque"):
        _shap_values(Opaque(), pd.DataFrame({"a": [1.0, 2.0]}),
                     pd.DataFrame({"a": [1.0]}))
