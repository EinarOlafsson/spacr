"""Which widget a setting gets, decided without importing a GUI.

This module exists because of one measured number. Opening the first module
in the Qt application spent **770 ms** inside a single statement:

    from spacr.gui_utils import convert_settings_dict_for_gui

-- and that was the whole remaining cost of the first module open, measured
with the event-loop watchdog in ``tests/qt/test_gui_responsiveness.py``
*after* ``spacr`` and ``spacr.settings`` were already imported. The function
being fetched is a hundred lines of dictionary lookups. Everything else in
the 770 ms belongs to the module it happened to live in: ``spacr.gui_utils``
imports ``spacr.gui_elements`` (IPython 154 ms, matplotlib.pyplot 145 ms),
``cv2`` (79 ms), ``tkinter``, ``huggingface_hub``, ``requests``, ``PIL`` and
``screeninfo`` -- the *Tk* interface's dependencies, none of which the Qt
interface has any use for.

``spacr.qt.app.main`` prewarms ``gui_utils`` on a background thread, and that
helps a user who looks at the home screen for a second first. It does not
help the user who clicks a module immediately: CPython's per-module import
lock makes the GUI thread *wait for the prewarm thread to finish*, so the
window freezes for whatever is left of the 770 ms. A prewarm cannot fix a
cost; it can only move it, and only when there is somewhere to move it to.

So the function moved to a module with no imports at all. ``gui_utils``
re-exports it, unchanged, for the Tk interface and for every existing caller.

This is the same argument already written down twice in ``gui_utils``: once
for ``torch`` (1.40 s, removed from that module's header) and once for
``torchvision`` (~5 s, never imported by
:func:`convert_settings_dict_for_gui` -- see the curated list below). Applied
a third time, to the module itself.
"""
from __future__ import annotations

import sys

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
    chan_list = ['[0,1,2,3,4,5,6,7,8]','[0,1,2,3,4,5,6,7]','[0,1,2,3,4,5,6]','[0,1,2,3,4,5]','[0,1,2,3,4]','[0,1,2,3]', '[0,1,2]', '[0,1]', '[0]', '[0,0]']

    variables = {}
    special_cases = {
        'metadata_type': ('combo', ['cellvoyager', 'cq1', 'auto', 'custom'], 'cellvoyager'),
        'channels': ('combo', chan_list, '[0,1,2,3]'),
        'train_channels': ('combo', ["['r','g','b']", "['r','g']", "['r','b']", "['g','b']", "['r']", "['g']", "['b']"], "['r','g','b']"),
        'channel_dims': ('combo', chan_list, '[0,1,2,3]'),
        # io.generate_training_dataset dispatches on metadata|annotation|
        # measurement and returns (None, None) for anything else. 'recruitment'
        # was offered here and silently produced no dataset.
        'dataset_mode': ('combo', ['annotation', 'metadata', 'measurement'], 'metadata'),
        'cov_type': ('combo', ['HC0', 'HC1', 'HC2', 'HC3', None], None),
        'crop_mode': ('combo', ["['cell']", "['nucleus']", "['pathogen']", "['organelle']", "['cell', 'nucleus']", "['cell', 'pathogen']", "['cell', 'organelle']", "['nucleus', 'pathogen']", "['cell', 'nucleus', 'pathogen']", "['cell', 'nucleus', 'pathogen', 'organelle']"], "['cell']"),
        'timelapse_mode': ('combo', ['trackastra', 'ultrack', 'trackpy', 'iou', 'btrack'], 'trackastra'),
        'train_mode': ('combo', ['erm', 'irm'], 'erm'),
        'clustering': ('combo', ['dbscan', 'kmean'], 'dbscan'),
        'reduction_method': ('combo', ['umap', 'tsne'], 'umap'),
        'model_name': ('combo', ['cpsam'], 'cpsam'),
        'regression_type': ('combo', ['ols','gls','wls','rlm','glm','mixed','quantile','logit','probit','poisson','lasso','ridge'], 'ols'),
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
        'cv_group_by': ('combo', ['well', 'field', 'plate', 'none'], 'well'),
        # spacr.seg_qc.MODES
        'seg_qc': ('combo', ['off', 'report', 'flag'], 'report'),
        # Three states, not two: None defers to SPACR_STRICT_ERRORS so a
        # cluster can turn it on for a batch without editing every file.
        'strict_errors': ('combo', [None, True, False], None),
        'normalize_by': ('combo', ['fov', 'png'], 'png'),
        'agg_type': ('combo', ['mean', 'median'], 'mean'),
        'grouping': ('combo', ['mean', 'median'], 'mean'),
        'min_max': ('combo', ['allq', 'all'], 'allq'),
        'transform': ('combo', ['log', 'sqrt', 'square', None], None),
        'organelle_morphology': ('combo', ['spots', 'network', 'irregular', 'ring'], 'spots'),
        'organelle_method': ('combo', ['otsu', 'adaptive', 'log', 'dog', 'ridge', 'hysteresis', 'cellpose', 'unet'], 'otsu'),
        'organelle_model_name': ('combo', ['cpsam'], 'cpsam'),
        'organelle_ridge_filter': ('combo', ['frangi', 'sato', 'meijering'], 'frangi'),
        'organelle_network_threshold': ('combo', ['otsu', 'adaptive'], 'otsu'),
        'organelle_ring_fill_method': ('combo', ['flood', 'convex'], 'flood'),
        'summarize_organelles_by': ('combo', ["['cell']","['nucleus']","['pathogen']","['cytoplasm']","['cell', 'nucleus']","['cell', 'pathogen']","['cell', 'cytoplasm']","['cell', 'nucleus', 'pathogen']","['cell', 'nucleus', 'pathogen', 'cytoplasm']",None], None)

    }

    for key, value in settings.items():
        if key in special_cases:
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
