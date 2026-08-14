"""One Classify entry point over both classifier families.

Classify (CV) trains a Torch model on object crops. Classify (ML) fits a
gradient-boosted model on measured features. They answer the same question --
*which class is this object* -- and until now they were two modules that
shared six setting names out of 78 and 37, two of which disagreed on their
default.

:mod:`spacr.training_basis` already unified what defines a CLASS. This module
unifies what runs: one settings dict, one ``classifier_family`` switch, one
call. The two original modules stay exactly as they are -- a merged screen
that removed them would strand every saved settings CSV and every notebook
that imports their entry points.

**Nothing here reimplements either pipeline.** ``deep_spacr`` and
``generate_ml_scores`` are called unchanged, which is what makes the merged
module honest: a run through it and a run through the module it dispatches to
produce the same result, because they are the same code.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

#: The two classifier families, in the order the settings panel offers them.
CLASSIFIER_FAMILIES: Tuple[str, ...] = ("cv", "ml")

# Keep this list beside the family dispatcher: it is the boundary at which a
# merged settings payload becomes an ML run.  Rejecting a CV backbone here is
# both cheaper and more accurate than letting it survive database loading and
# fail deep inside ``ml_analysis``.
ML_MODEL_TYPES: Tuple[str, ...] = (
    "xgboost", "lightgbm", "catboost", "random_forest", "extra_trees",
    "gradient_boosting", "logistic_regression", "svm", "mlp",
)

#: family -> the app key whose settings and pipeline it uses. The merged
#: screen is a front end onto these, not a replacement for them.
FAMILY_APP_KEY: Dict[str, str] = {"cv": "classify", "ml": "ml_analyze"}

#: Settings each family reads that the other has no use for. Drives the
#: greying, the same way :data:`spacr.training_basis.BASIS_SETTINGS` does for
#: the training basis, and for the same reason: a control the user can edit
#: that changes nothing is worse than one that is not there.
FAMILY_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "cv": (
        "crop_shape", "extract_channels", "object_array", "coordinate_columns",
        "normalization", "normalization_scope",
        "model_type", "custom_model", "custom_model_path", "image_size",
        "train_channels", "epochs", "optimizer_type", "schedule", "loss_type",
        "dropout_rate", "init_weights", "amsgrad", "weight_decay",
        "gradient_accumulation", "gradient_accumulation_steps",
        "early_stopping_patience", "augment", "pin_memory", "use_checkpoint",
        "resume_checkpoint", "tensorboard", "focal_gamma", "focal_alpha",
        "label_smoothing", "logit_adjust_tau", "train", "test",
        "generate_training_dataset", "apply_model_to_dataset",
        "generate_full_dataset", "tar_path", "n_top_examples", "path_string",
        "file_type", "crop_source",
    ),
    "ml": (
        "model_type_ml", "n_estimators", "reg_alpha", "reg_lambda",
        "prune_features", "top_features", "n_repeats", "minimum_cell_count",
        "remove_low_variance_features", "remove_highly_correlated_features",
        "heatmap_feature", "grouping", "min_max", "cmap", "save_to_db",
        "batch_correction", "batch_column", "batch_control_column",
        "batch_control_values", "batch_covariate_column",
        "batch_combat_mean_only", "batch_min_samples", "batch_missing_control",
        "nuclei_limit", "pathogen_limit", "exclude",
    ),
}


class ClassifierFamilyError(ValueError):
    """A classifier family spaCR does not have."""


def resolve_family(settings: Mapping[str, Any]) -> str:
    """Return the classifier family a settings dict asks for.

    Defaults to ``'cv'``, because the merged module's own default settings
    are the CV ones and a dict with no family is most likely a Classify (CV)
    CSV opened in the merged screen.

    :param settings: the run settings.
    :returns: ``'cv'`` or ``'ml'``.
    :raises ClassifierFamilyError: an unrecognised family. Guessing would
        train a different kind of model than the user asked for and report
        success.
    """
    declared = settings.get("classifier_family")
    if declared is None or declared == "":
        return "cv"
    family = str(declared).strip().lower()
    if family not in CLASSIFIER_FAMILIES:
        raise ClassifierFamilyError(
            f"classifier_family={declared!r} is not one of "
            f"{list(CLASSIFIER_FAMILIES)}")
    return family


def inapplicable_settings(family: str) -> Tuple[str, ...]:
    """Settings belonging to the OTHER family -- what the panel greys out.

    Greyed, never removed: INVARIANTS §6. A key absent from the dict makes
    the pipeline fall back to its own default, which can differ from the
    value the module needs and says nothing when it does.

    :param family: the chosen family.
    :returns: setting keys the other family owns.
    :raises ClassifierFamilyError: unknown family.
    """
    key = str(family).strip().lower()
    if key not in FAMILY_SETTINGS:
        raise ClassifierFamilyError(
            f"{family!r} is not one of {list(CLASSIFIER_FAMILIES)}")
    mine = set(FAMILY_SETTINGS[key])
    return tuple(k for other, keys in FAMILY_SETTINGS.items()
                 if other != key for k in keys if k not in mine)


def resolve_ml_model_type(settings: Mapping[str, Any]) -> str:
    """Return the classical-ML estimator selected by ``settings``.

    ``model_type_ml`` is authoritative in a merged payload.  ``model_type``
    is accepted only when the ML-specific key is absent, which migrates the
    short-lived shared-vocabulary settings files written before the two model
    controls were separated.  The default matches
    :func:`spacr.settings.set_default_analyze_screen`.

    :param settings: settings for an ML-family run.
    :returns: a member of :data:`ML_MODEL_TYPES`.
    :raises ValueError: when the selected value is not an ML estimator.
    """
    selected = settings.get("model_type_ml")
    if selected in (None, ""):
        selected = settings.get("model_type", "xgboost")
    model_type = str(selected).strip().lower()
    if model_type not in ML_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type_ml: {selected!r}. Choose one of "
            f"{list(ML_MODEL_TYPES)}")
    return model_type


def classify(settings: Mapping[str, Any]) -> Any:
    """Run whichever classifier family ``settings`` asks for.

    The merged module's pipeline entry point. It normalises the shared
    vocabulary, resolves the family, and calls the existing entry point
    unchanged -- so a run here and a run through Classify (CV) or Classify
    (ML) are the same run, not two implementations that have to be kept in
    step.

    :param settings: the run settings.
    :returns: whatever the dispatched pipeline returns.
    :raises ClassifierFamilyError: an unrecognised family.
    """
    from .classify_classes import normalize_settings as normalize_classes
    from .training_basis import normalize_settings

    # Two translations, both idempotent and both in one place: the shared
    # vocabulary (names) and the class definition (what the names select).
    # Anything downstream reads the current shape only.
    # Resolve family-owned values before shared-vocabulary normalization.
    # ``training_basis.normalize_settings`` deliberately treats
    # model_type_ml as a legacy alias for model_type.  A merged payload has
    # both keys, though, and the CV value wins that generic alias operation.
    # Capturing the ML value here prevents maxvit_t (or any other CV
    # backbone) from being sent to the classical estimator pipeline.
    family = resolve_family(settings)
    ml_model_type = (
        resolve_ml_model_type(settings) if family == "ml" else None
    )
    resolved = dict(normalize_classes(normalize_settings(settings)))

    if family == "cv":
        # Refuse a crop source that cannot produce images BEFORE training
        # starts. Discovering that extract_channels was never set after an
        # hour of dataset building is a worse failure than one at the door,
        # and the message names the setting to change.
        from .crop_source import validate as validate_crops
        validate_crops(resolved)

    if family == "ml":
        from .ml import generate_ml_scores
        # The ML pipeline reads only its family-owned spelling.  Assignment,
        # rather than setdefault, is intentional: ``model_type`` may contain
        # the simultaneously visible CV choice in a merged settings file.
        resolved["model_type_ml"] = ml_model_type
        if "test_split" in resolved:
            resolved.setdefault("test_size", resolved["test_split"])
        if "cross_validation_enabled" in resolved:
            resolved.setdefault("cross_validation",
                                resolved["cross_validation_enabled"])
        return generate_ml_scores(resolved)

    from .deep_spacr import deep_spacr
    return deep_spacr(resolved)
