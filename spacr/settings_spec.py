"""Map settings to GUI widget specifications without importing a GUI.

The conversion helpers return plain dictionaries that both the Qt and legacy
interfaces can consume. Keeping this module dependency-light lets callers
inspect setting metadata without importing plotting, imaging, or deep-learning
libraries.

``spacr.gui_utils`` re-exports
:func:`convert_settings_dict_for_gui` for compatibility with existing callers.
"""
from __future__ import annotations

import sys

from .organelle_types import ALL_ORGANELLE_ROLES as _ORGANELLE_SLOT_ROLES
from .organelle_types import DEFAULT_NUMBER_OF_ORGANELLES, MAX_ORGANELLES
from .organelle_types import DEFAULT_TYPE as _ORGANELLE_TYPE_DEFAULT
from .organelle_types import TYPE_ORDER as _ORGANELLE_TYPE_ORDER
from .schema import ALL_ROLES

__all__ = ["convert_settings_dict_for_gui"]


_TORCHVISION_MODELS_CURATED = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'densenet121', 'densenet169', 'densenet201',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
    'swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
    'maxvit_t', 'regnet_y_400mf', 'regnet_y_1_6gf', 'regnet_y_8gf',
    'squeezenet1_0', 'squeezenet1_1', 'alexnet', 'googlenet', 'inception_v3',
]


def _regression_type_choices():
    """Every family that fits, as ``(stored value, label)``, grouped.

    :returns: pairs whose first element is the stored ``regression_type`` and
        whose second is the line the dropdown shows -- the family's name, the
        kind of fit it is, and what it assumes.

    A long list of unlabelled names in alphabetical order is a menu that hides
    its own contents: the quantile fit, the two robust losses and the rank
    aggregation were all on it and none of them could be found.
    :func:`spacr.regression_families.regression_family_choices` places each
    family in one of three honest kinds -- parametric,
    robust/semiparametric, rank-based -- and gives it a sentence saying what
    has to be true of the data for its answer to mean anything.

    Mixed leads because it is the default and answers the most central
    question; then the rest of the parametric group, then the robust one,
    then the rank-based one, so a family added to the inventory lands
    somewhere predictable instead of at the end.

    NOTHING IS RENAMED. The stored value is unchanged and leads its own
    label, so a settings CSV written before the grouping asks for exactly the
    fit it always asked for, and a user looking for 'quantile' still finds
    the word.

    `spacr.regression_families` imports only `spacr.regression_spec`, which
    imports nothing -- that is the whole reason the vocabulary was split out
    of `spacr.ml`, which pulls in torch through `spacr.plot` -- so asking it
    here costs a dict lookup rather than 2.2 seconds.
    """
    from .regression_families import regression_family_choices

    return regression_family_choices()


def _regression_backend_choices():
    """Every backend, labelled ``(CPU)`` or ``(GPU)``, in panel order.

    THE OPTIONS ARE THE LABELS, and so is the stored value -- see
    :func:`spacr.settings._resolve_regression_backend` for why. The labels
    state whether a backend uses the CPU or GPU, and both front ends render
    them verbatim.

    Read from :mod:`spacr.regression_backends`, which imports nothing heavier
    than stdlib, so a settings panel still costs a dict lookup rather than
    ``import torch``.

    Every entry is offered, including those that cannot run in the current
    environment. The panel disables unavailable entries and obtains their
    explanations from :func:`spacr.regression_backends.backend_menu`.
    """
    from .regression_backends import backend_choices

    return backend_choices()


def _metadata_type_choices():
    """Every filename convention, as ``(stored value, label)``, by vendor.

    :returns: pairs whose first element is the stored ``metadata_type`` and
        whose second is the line a plain dropdown shows -- the vendor, the
        instrument family, and, for a convention spaCR is guessing about,
        the word "provisional".

    A user with a Zeiss, a Nikon, a Leica or a Thermo Fisher instrument used
    to meet a dropdown offering two Yokogawas and "write your own regular
    expression". The stored values are UNCHANGED for the four that were
    already there, so a settings CSV written before this asks for exactly
    the convention it always asked for.

    PROVISIONAL IS ON SCREEN ON PURPOSE. Some of these patterns were read
    off a vendor manual or off Bio-Formats' own reader source; others were
    reconstructed from a handful of real filenames found in public datasets
    and forum posts. A user whose instrument is in the second group should
    know that before a run, not after a plate has been mislabelled -- and
    the "test it on my folder" button beside the dropdown is what turns the
    warning into an answer.

    Read from :mod:`spacr.regex_infer`, which imports nothing outside the
    standard library, so building a settings panel still costs a dict
    lookup.
    """
    from .regex_infer import _metadata_convention_menu

    choices = []
    for vendor, rows in _metadata_convention_menu():
        for key, label, status in rows:
            suffix = "" if status == "confirmed" else "  [provisional]"
            choices.append((key, f"{vendor} -- {label}{suffix}"))
    return choices


#: ``cam_type`` menu when :mod:`spacr.attribution` is not imported yet, in
#: the order :func:`spacr.attribution.cam_type_choices` gives. Kept here so
#: building a settings panel does not import torch; a test holds the two
#: equal.
_CAM_TYPE_CHOICES = (
    'gradcam', 'gradcam_pp', 'saliency_image', 'saliency_channel',
    'torchcam_gradcam', 'torchcam_gradcam_pp', 'ablation_cam',
    'attention_rollout', 'chefer', 'deeplift', 'deeplift_shap', 'eigencam',
    'feature_ablation', 'gradient_shap', 'guided_backprop', 'hirescam',
    'input_x_gradient', 'integrated_gradients', 'layercam', 'occlusion',
    'saliency', 'scorecam', 'xgradcam',
)


def _cam_type_choices():
    """Every ``cam_type`` the Activation Maps form offers.

    Read from :func:`spacr.attribution.cam_type_choices` when that module is
    already loaded, and from :data:`_CAM_TYPE_CHOICES` otherwise, so the
    panel never pays for importing torch.
    """
    module = sys.modules.get("spacr.attribution")
    if module is not None:
        try:
            return list(module.cam_type_choices())
        except Exception:
            pass
    return list(_CAM_TYPE_CHOICES)


def _torchvision_model_names():
    """Return model names for the combo WITHOUT importing torchvision. If
    torchvision is already loaded (e.g. after a training run) use its full zoo;
    otherwise fall back to the curated static list."""
    mods = sys.modules.get("torchvision.models")
    if mods is not None:
        try:
            names = [n for n, o in mods.__dict__.items()
                     if callable(o) and not n.startswith("_")]
            if names:
                return sorted(set(names) | set(_TORCHVISION_MODELS_CURATED))
        except Exception:
            pass
    return list(_TORCHVISION_MODELS_CURATED)


def _cellpose_model_names():
    """Return live Cellpose choices without loading the numerical stack.

    A cold settings-panel build needs only the shipped fallback.  Once either
    Cellpose or :mod:`spacr.settings` is already loaded, the lightweight
    accessor in ``settings`` can add installed and user-registered models
    without making this module responsible for a heavy first import.
    """
    settings_name = f"{__package__}.settings"
    if (settings_name not in sys.modules
            and "cellpose.models" not in sys.modules):
        return ["cpsam"]

    from .settings import cellpose_model_choices

    return list(cellpose_model_choices())


#: Settings whose widget cannot be decided from the NAME alone, because two
#: modules use that name for two different closed vocabularies. The value in
#: hand decides; anything not listed falls through to the name-keyed table.
#:
#: ``level`` is the only one. The proportion and endodyogeny plots have meant
#: 'object'/'well'/'plate' by it for years, while regression uses
#: 'both'/'grna'/'gene'. The shared tables here and in :mod:`spacr.settings`
#: are keyed by name with no module scope, so dispatch uses the value already
#: present on the panel.
#:
#: Deliberately NOT a fallback: a value in neither vocabulary returns None and
#: takes the ordinary path, so no module's existing widget changes shape.
_VALUE_SPECIAL_CASES = {
    'level': (
        (('both', 'grna', 'gene'),
         ('combo', [('both', 'both — gRNA and gene effects, each corrected '
                             'as its own family'),
                    ('grna', 'gRNA effects — one estimate per guide'),
                    ('gene', 'gene effects — one estimate per gene, its '
                             'guides pooled')],
          'both')),
        (('object', 'well', 'plate'),
         ('combo', ['object', 'well', 'plate'], 'object')),
    ),
}


def _value_special_cases(key, value):
    """The widget spec for ``key`` when its VALUE decides, else ``None``.

    :param key: the setting name.
    :param value: the value the panel is being built from.
    :returns: a ``(kind, options, default)`` triple, or ``None`` to fall
        through to the name-keyed ``special_cases`` table.
    """
    table = _VALUE_SPECIAL_CASES.get(key)
    if not table:
        return None
    if not isinstance(value, str):
        return None
    current = value.strip().lower()
    for vocabulary, spec in table:
        if current in vocabulary:
            kind, options, _default = spec
            return (kind, list(options), current)
    return None


def convert_settings_dict_for_gui(settings):
    """Convert a plain settings dict into the GUI variable spec.

    Maps each key to a ``(widget_type, options, default_value)`` triple, using
    combo boxes for keys with known enumerated options and inferring
    check/entry widgets otherwise.

    :param settings: mapping of setting names to default values.
    :returns: mapping ``key -> (var_type, options, default_value)`` ready for
        :func:`spacr.gui_utils.create_input_field` or for
        :meth:`spacr.qt.screens.settings_model.SettingsWidgets.build_sections`.
    """
    torchvision_models = _torchvision_model_names()
    cellpose_models = _cellpose_model_names()
    chan_list = ['[0,1,2,3,4,5,6,7,8]','[0,1,2,3,4,5,6,7]','[0,1,2,3,4,5,6]','[0,1,2,3,4,5]','[0,1,2,3,4]','[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]', '[0,0]']

    variables = {}
    special_cases = {
        'analysis_mode': ('combo',
                          [('regression', 'regression — fit every guide at '
                                          'once in the chosen model'),
                           ('guide_permutation', 'guide permutation — test '
                                                 'each guide on its own, '
                                                 'wells reshuffled within '
                                                 'each plate')],
                          'regression'),
        'grna_statistic': ('combo', ['pearson', 'rank'], 'pearson'),
        'p_threshold_kind': ('combo', ['adjusted', 'raw'], 'adjusted'),
        'metadata_type': ('combo', _metadata_type_choices(), 'cellvoyager'),
        'plaque_mode': ('combo', ['plaque', 'figure'], 'plaque'),
        'channels': ('combo', chan_list, '[0,1,2,3]'),
        'train_channels': ('combo', ["['r','g','b']", "['r','g']", "['r','b']", "['g','b']", "['r']", "['g']", "['b']"], "['r','g','b']"),
        'channel_dims': ('combo', chan_list, '[0,1,2,3]'),
        'dataset_mode': ('combo', ['annotation', 'metadata'], 'metadata'),
        'cov_type': ('combo', ['HC0', 'HC1', 'HC2', 'HC3', None], None),
        'crop_mode': ('combo',
                      [repr([role]) for role in ALL_ROLES]
                      + [repr(['cell', role]) for role in ALL_ROLES
                         if role != 'cell'],
                      "['cell']"),
        'timelapse_mode': ('combo', ['trackastra', 'ultrack', 'trackpy', 'iou', 'btrack', 'timeflows'], 'trackastra'),
        'train_mode': ('combo', ['erm', 'irm'], 'erm'),
        'clustering': ('combo', ['dbscan', 'kmean'], 'dbscan'),
        'reduction_method': ('combo', ['umap', 'tsne'], 'umap'),
        'model_name': ('combo', cellpose_models, cellpose_models[0]),
        'regression_type': ('combo', _regression_type_choices(), 'mixed'),
        'regression_backend': ('combo', _regression_backend_choices(),
                               'statsmodels (CPU)'),
        'timelapse_objects': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['organelle']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['cell', 'organelle']", "['nucleus', 'pathogen']", "['nucleus', 'organelle']", "['cell', 'nucleus', 'pathogen']", "['cell', 'nucleus', 'organelle']", "['cell', 'nucleus', 'pathogen', 'organelle']"], "['cell']"),
        'model_type': ('combo', torchvision_models, 'resnet50'),
        'cam_type': ('combo', _cam_type_choices(), 'gradcam'),
        'compression': ('combo', ['lzw', 'zlib', 'none'], 'lzw'),
        'model_type_ml': ('combo', ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees', 'gradient_boosting', 'logistic_regression', 'svm', 'mlp'], 'xgboost'),
        'optimizer_type': ('combo', ['adamw', 'adam', 'adamax', 'sgd', 'rmsprop', 'nadam', 'radam', 'adagrad', 'adadelta', 'asgd'], 'adamw'),
        'schedule': ('combo', ['cosine', 'cosine_warm_restarts', 'reduce_lr_on_plateau', 'step_lr', 'exponential', 'linear', 'none'], 'cosine'),
        'loss_type': ('combo', ['auto', 'cross_entropy', 'label_smoothing', 'focal_loss', 'ce_weighted', 'logit_adjust_ce', 'asl', 'binary_cross_entropy_with_logits'], 'auto'),
        'class_balance': ('combo', ['none', 'weighted_sampler', 'sqrt_weighted_sampler', 'weighted_loss'], 'none'),
        'cv_group_by': ('combo', ['cell', 'field', 'well', 'plate'], 'well'),
        'seg_qc': ('combo', ['off', 'report', 'flag', 'stop'], 'report'),
        'psf_measurement_source': ('combo', ['original', 'processed'], 'original'),
        'confluency_source': ('combo', ['auto', 'masks', 'texture', 'intensity'], 'auto'),
        'psf_operation': ('combo', ['none', 'convolve', 'deconvolve'], 'none'),
        'psf_source': ('combo', ['gaussian', 'measured'], 'gaussian'),
        'psf_objective': ('combo', ['auto', '10x/0.30 air', '10x/0.45 air', '20x/0.45 air',
                                    '20x/0.75 air', '40x/0.95 air', '40x/1.30 oil',
                                    '60x/1.20 water', '60x/1.40 oil', '63x/1.40 oil',
                                    '100x/1.40 oil', '100x/1.45 oil'], 'auto'),
        'psf_path': ('entry', None, None),
        'psf_image_sampling_um': ('entry', None, None),
        'psf_kernel_sampling_um': ('entry', None, None),
        'psf_fwhm_um': ('entry', None, None),
        'psf_iterations': ('entry', None, 20),
        'enhance_background': ('combo', ['none', 'rolling_ball', 'tophat'], 'none'),
        'enhance_denoise': ('combo', ['none', 'gaussian', 'median', 'bilateral', 'nlm', 'tv'], 'none'),
        'image_qc_mode': ('combo', ['off', 'report', 'exclude'], 'off'),
        'tta_aggregation': ('combo', ['probability_mean', 'majority_vote'], 'probability_mean'),
        'replication_method': ('combo', [
            ('direct_count', 'Direct parasite counts'),
            ('size_proxy', 'Area-derived size proxy (legacy)'),
            ('deep_learning_coming_soon', 'Whole-vacuole deep learning classification — coming soon'),
        ], 'direct_count'),
        'strict_errors': ('combo', [None, True, False], None),
        'normalize_by': ('combo', ['fov', 'png'], 'png'),
        'agg_type': ('combo', ['mean', 'median'], 'mean'),
        'grouping': ('combo', ['mean', 'median'], 'mean'),
        'min_max': ('combo', ['allq', 'all'], 'allq'),
        'transform': ('combo', ['log', 'sqrt', 'square', 'beta', None], None),
        'intercept': ('combo', ['fitted', 'zero', 'control', 'value'],
                      'fitted'),
        'number_of_organelles': ('combo',
                                 list(range(MAX_ORGANELLES + 1)),
                                 DEFAULT_NUMBER_OF_ORGANELLES),
        'organelle_type': ('combo', list(_ORGANELLE_TYPE_ORDER),
                           _ORGANELLE_TYPE_DEFAULT),
        'organelle_morphology': ('combo', ['spots', 'network', 'irregular', 'ring'], 'spots'),
        'organelle_method': ('combo', ['otsu', 'adaptive', 'log', 'dog', 'ridge', 'hysteresis', 'cellpose', 'unet'], 'otsu'),
        'organelle_model_name': ('combo', cellpose_models,
                                 cellpose_models[0]),
        'organelle_ridge_filter': ('combo', ['frangi', 'sato', 'meijering'], 'frangi'),
        'organelle_network_threshold': ('combo', ['otsu', 'adaptive'], 'otsu'),
        'organelle_ring_fill_method': ('combo', ['flood', 'convex'], 'flood'),
        'summarize_organelles_by': ('combo', ["['cell']","['nucleus']","['pathogen']","['cytoplasm']","['cell', 'nucleus']","['cell', 'pathogen']","['cell', 'cytoplasm']","['cell', 'nucleus', 'pathogen']","['cell', 'nucleus', 'pathogen', 'cytoplasm']",None], None)

    }

    primary_widget_keys = tuple(
        key for key in special_cases if key.startswith('organelle_'))
    for role in _ORGANELLE_SLOT_ROLES[1:]:
        for key in primary_widget_keys:
            slot_key = f"{role}_{key[len('organelle_'):]}"
            kind, options, default = special_cases[key]
            special_cases[slot_key] = (
                kind, list(options) if isinstance(options, list) else options,
                default)

    for key, value in settings.items():
        by_value = _value_special_cases(key, value)
        if by_value is not None:
            variables[key] = by_value
        elif key in special_cases:
            variables[key] = special_cases[key]
        elif isinstance(value, bool):
            variables[key] = ('check', None, value)
        elif isinstance(value, int) or isinstance(value, float):
            variables[key] = ('entry', None, value)
        elif isinstance(value, str):
            variables[key] = ('entry', None, value)
        elif value is None:
            variables[key] = ('entry', None, value)
        elif isinstance(value, list):
            variables[key] = ('entry', None, str(value))
        else:
            variables[key] = ('entry', None, str(value))

    return variables
