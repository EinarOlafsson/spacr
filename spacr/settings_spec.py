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
from .organelle_types import (ALL_ORGANELLE_ROLES as _ORGANELLE_SLOT_ROLES,
                              DEFAULT_NUMBER_OF_ORGANELLES,
                              DEFAULT_TYPE as _ORGANELLE_TYPE_DEFAULT,
                              MAX_ORGANELLES,
                              TYPE_ORDER as _ORGANELLE_TYPE_ORDER)
from .schema import ALL_ROLES

__all__ = ["convert_settings_dict_for_gui"]


# Curated torchvision classification models for the `model_type` combo. Kept
# static so opening a settings screen never triggers a slow `import
# torchvision`. The pipeline validates/instantiates the real model by name at
# train time.
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

    Nineteen unlabelled names in one alphabetical list is a menu that hides
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
        # NAMED FOR WHAT THEY PRODUCE. 'grna'/'gene'/'both' are the keys the
        # settings file and the API have always used and they do not change;
        # what changes is that the panel says which analysis each one asks
        # for, so a reader who wants gene effects can find them without
        # knowing that `level` is the control that gives them.
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
            # The panel's own value is the default, so opening a settings
            # screen never rewrites the setting it was opened on.
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
    # NOTE: we deliberately do NOT `import torchvision` here. Enumerating the
    # torchvision model zoo pulls in torch + torchvision, a ~5 s import that
    # made every FIRST module open sluggish. The classify pipeline still
    # instantiates the real torchvision model by name at train time — the GUI
    # combo just needs a list of valid names, so we use a curated static list
    # (if torchvision happens to be imported already we extend it with the full
    # zoo, for free).
    torchvision_models = _torchvision_model_names()
    # Same bargain, for the same measured reason: `cellpose.models` pulls in
    # torch (~2.5 s) and this runs while a settings page is being built, so
    # the accessor reads the API only when Cellpose is already loaded and
    # degrades to the shipped list otherwise. It is never empty.
    from .settings import cellpose_model_choices
    cellpose_models = list(cellpose_model_choices())
    chan_list = ['[0,1,2,3,4,5,6,7,8]','[0,1,2,3,4,5,6,7]','[0,1,2,3,4,5,6]','[0,1,2,3,4,5]','[0,1,2,3,4]','[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]', '[0,0]']

    variables = {}
    special_cases = {
        # Instruction 134: two valid values, and it was a free-text box in
        # both front ends. Declared here rather than only in the Qt combo
        # table so the two GUIs cannot offer different lists.
        # THE LABELS READ, and the VALUES do not change (134's third point).
        # 'guide_permutation' is what the settings key is called; what the
        # dropdown shows is the sentence, the same way 132's model box
        # explains what it fits. (value, label) pairs, so every settings file
        # already written goes on meaning what it meant.
        'analysis_mode': ('combo',
                          [('regression', 'regression — fit every guide at '
                                          'once in the chosen model'),
                           ('guide_permutation', 'guide permutation — test '
                                                 'each guide on its own, '
                                                 'wells reshuffled within '
                                                 'each plate')],
                          'regression'),
        # Instruction 135, and the same argument as `analysis_mode` above:
        # two valid values, and the RUN now has to agree with the volcano's
        # right-click menu about which P value 'significant' meant. A
        # free-text box lets a settings CSV say 'Adjusted' or 'bh' and be
        # refused at the seam instead of picked from a list of two. Declared
        # here rather than only in the Qt combo table so the Tk and Qt panels
        # cannot offer different lists.
        # OFFERED, NOT TYPED. Two spellings, and a free-text box lets a
        # settings CSV say 'spearman' -- a reasonable guess for what
        # 'rank' does -- and be refused at the seam rather than picked
        # from a list of two.
        'grna_statistic': ('combo', ['pearson', 'rank'], 'pearson'),
        'p_threshold_kind': ('combo', ['adjusted', 'raw'], 'adjusted'),
        'metadata_type': ('combo', ['cellvoyager', 'cq1', 'auto', 'custom'], 'cellvoyager'),
        'channels': ('combo', chan_list, '[0,1,2,3]'),
        'train_channels': ('combo', ["['r','g','b']", "['r','g']", "['r','b']", "['g','b']", "['r']", "['g']", "['b']"], "['r','g','b']"),
        'channel_dims': ('combo', chan_list, '[0,1,2,3]'),
        # io.generate_training_dataset dispatches on metadata|annotation|
        # measurement and returns (None, None) for anything else. 'recruitment'
        # was offered here and silently produced no dataset.
        'dataset_mode': ('combo', ['annotation', 'metadata'], 'metadata'),
        'cov_type': ('combo', ['HC0', 'HC1', 'HC2', 'HC3', None], None),
        'crop_mode': ('combo',
                      [repr([role]) for role in ALL_ROLES]
                      + [repr(['cell', role]) for role in ALL_ROLES
                         if role != 'cell'],
                      "['cell']"),
        'timelapse_mode': ('combo', ['trackastra', 'ultrack', 'trackpy', 'iou', 'btrack'], 'trackastra'),
        'train_mode': ('combo', ['erm', 'irm'], 'erm'),
        'clustering': ('combo', ['dbscan', 'kmean'], 'dbscan'),
        'reduction_method': ('combo', ['umap', 'tsne'], 'umap'),
        'model_name': ('combo', cellpose_models, cellpose_models[0]),
        # DEFAULT 'mixed' since 2026-08-17, matching
        # settings.get_perform_regression_default_settings: "mixed answers
        # the most central question best". A combo whose default differs
        # from the settings default posts a different model than the one
        # the panel was built for.
        # READ FROM THE INVENTORY, NOT LISTED BY HAND. The hand-written
        # list offered 'gls' -- which is in UNSUPPORTED_REGRESSION_TYPES and
        # RAISES -- and omitted six families that fit: huber, beta,
        # quasi_binomial, elasticnet, hinge and horseshoe. So the panel
        # could pick a type that fails and could not reach a third of the
        # ones that work.
        # (value, label) PAIRS, GROUPED BY WHAT THEY ASSUME. Nineteen bare
        # names in one alphabetical list hid the four families a user was
        # looking for; the label says whether the fit is parametric,
        # robust/semiparametric or rank-based and what it assumes. The
        # stored values are unchanged, so every settings file already
        # written goes on meaning what it meant.
        'regression_type': ('combo', _regression_type_choices(), 'mixed'),
        # WHO fits it (instruction 141 A). Default 'statsmodels (CPU)' --
        # every existing result was produced with it, and a default that
        # changes the numbers under a user who changed nothing is not a
        # default. The label is the value; see _regression_backend_choices.
        'regression_backend': ('combo', _regression_backend_choices(),
                               'statsmodels (CPU)'),
        'timelapse_objects': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['organelle']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['cell', 'organelle']", "['nucleus', 'pathogen']", "['nucleus', 'organelle']", "['cell', 'nucleus', 'pathogen']", "['cell', 'nucleus', 'organelle']", "['cell', 'nucleus', 'pathogen', 'organelle']"], "['cell']"),
        'model_type': ('combo', torchvision_models, 'resnet50'),
        'compression': ('combo', ['lzw', 'zlib', 'none'], 'lzw'),
        'model_type_ml': ('combo', ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees', 'gradient_boosting', 'logistic_regression', 'svm', 'mlp'], 'xgboost'),
        'optimizer_type': ('combo', ['adamw', 'adam', 'adamax', 'sgd', 'rmsprop', 'nadam', 'radam', 'adagrad', 'adadelta', 'asgd'], 'adamw'),
        'schedule': ('combo', ['cosine', 'cosine_warm_restarts', 'reduce_lr_on_plateau', 'step_lr', 'exponential', 'linear', 'none'], 'cosine'),
        'loss_type': ('combo', ['auto', 'cross_entropy', 'label_smoothing', 'focal_loss', 'ce_weighted', 'logit_adjust_ce', 'asl', 'binary_cross_entropy_with_logits'], 'auto'),
        # io.CLASS_BALANCE_MODES / io.CV_GROUP_LEVELS — both raise ValueError
        # on anything outside these lists, so free text is not usable here.
        'class_balance': ('combo', ['none', 'weighted_sampler', 'sqrt_weighted_sampler', 'weighted_loss'], 'none'),
        'cv_group_by': ('combo', ['cell', 'field', 'well', 'plate'], 'well'),
        # spacr.seg_qc.MODES
        'seg_qc': ('combo', ['off', 'report', 'flag', 'stop'], 'report'),
        # Three states, not two: None defers to SPACR_STRICT_ERRORS so a
        # cluster can turn it on for a batch without editing every file.
        'strict_errors': ('combo', [None, True, False], None),
        'normalize_by': ('combo', ['fov', 'png'], 'png'),
        'agg_type': ('combo', ['mean', 'median'], 'mean'),
        'grouping': ('combo', ['mean', 'median'], 'mean'),
        'min_max': ('combo', ['allq', 'all'], 'allq'),
        'transform': ('combo', ['log', 'sqrt', 'square', 'beta', None], None),
        # The four intercept modes, from spacr.ml.INTERCEPT_MODES. A combo
        # rather than free text: each name selects a different construction
        # of the design matrix, and an unrecognised one is refused at the
        # door by prepare_formula rather than quietly fitted.
        'intercept': ('combo', ['fitted', 'zero', 'control', 'value'],
                      'fitted'),
        # HOW MANY ORGANELLE SLOTS, as a closed list rather than a free
        # number. The bound is real -- a slot's name is the prefix of its
        # keys and the prefixes are lettered, so the alphabet runs out at
        # twenty-six -- and a typed thirty would have to be clamped to a
        # number the user did not ask for.
        'number_of_organelles': ('combo',
                                 list(range(MAX_ORGANELLES + 1)),
                                 DEFAULT_NUMBER_OF_ORGANELLES),
        # The ONE visible organelle choice (instruction 72). A combo, not a
        # free-text field: the nine names are a closed set, and
        # `organelle_types.resolve_type` raises on anything else -- typing it
        # by hand would turn a typo into a failed run instead of a pick.
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

    # All slot-specific controls use the primary organelle widget contract.
    # Generated for every slot `number_of_organelles` can name, not for the
    # slots this run has: a settings file written at seven slots is opened by
    # a session set to two, and its seventh slot's method must still arrive
    # as the closed dropdown it is rather than as a free-text field whose
    # every value fails validation.
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
