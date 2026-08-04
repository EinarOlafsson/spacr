"""
Bridge between spacr's plain-python default settings and Qt form widgets.

The existing spacr GUI expresses settings as `{name: (widget_type, options,
default)}` triples via `spacr.gui_utils.convert_settings_dict_for_gui`.
Here we consume the same conversion output and materialize each entry as
a real Qt widget grouped into logical Section boxes based on
`spacr.settings.categories`.
"""
from __future__ import annotations

import ast
from html import escape
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QLayout,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QLabel,
)

from ..widgets.barcode_regex import BarcodeRegexWidget
from ..widgets.external_mask_inputs import ExternalMaskInputWidget
from ..widgets.row_exclusion import RowExclusionEditor
from ..widgets.toggle import Toggle


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings resolvers per app_key
# ---------------------------------------------------------------------------

def timelapse_and_motility_keys() -> set:
    """Every setting key owned by the Timelapse / Motility Assay modules.

    Derived from the category lists in :mod:`spacr.settings` so the two never
    drift apart. Used to strip those keys out of the Mask module's editable
    settings — they still exist in the *pipeline* defaults (spacr.object reads
    ``timelapse`` on every run and ``motility_analysis`` inside the timelapse
    branch), the Mask GUI just no longer offers them.
    """
    from spacr.settings import (
        motility_advanced_settings, motility_settings, timelapse_settings,
    )
    return (set(timelapse_settings) | {"timelapse"}
            | set(motility_settings) | set(motility_advanced_settings))


def _registered_app_metadata(app_key: str) -> Dict[str, Any]:
    """One app's :data:`spacr.qt.app.APP_META` entry, or ``{}``.

    Read out of :data:`sys.modules`, never imported: ``spacr.qt.app``
    builds the screens that build this model, so importing it from here
    would be a cycle, and a process that has not loaded the registry
    simply has no registered apps to ask about.
    """
    app = sys.modules.get("spacr.qt.app")
    return (getattr(app, "APP_META", {}).get(app_key) or {}) if app else {}


def _import_registered_defaults_module(app_key: str) -> None:
    """Import the module that registers ``app_key``'s settings defaults.

    Named by ``register_app(..., defaults_module=...)``. Failure is
    logged and swallowed: an unimportable optional dependency should cost
    that app its settings panel, not stop the window opening.
    """
    module = _registered_app_metadata(app_key).get("defaults_module")
    if not module or module in sys.modules:
        return
    import importlib
    try:
        importlib.import_module(module)
    except Exception:
        LOGGER.warning("Could not import %s, which owns the %r settings",
                       module, app_key, exc_info=True)


def resolve_default_settings(app_key: str) -> Dict[str, Any]:
    """Return a fresh defaults dict for an app key, mirroring the Tk GUI
    dispatch in gui_core.setup_settings_panel."""
    try:
        from spacr.plugins import get_app, load_object
        plugin_app = get_app(app_key)
    except Exception:
        plugin_app = None
    if plugin_app is not None:
        defaults = load_object(plugin_app.defaults)
        if not callable(defaults):
            raise TypeError(f"Plugin defaults {plugin_app.defaults!r} are not callable")
        try:
            result = defaults({})
        except TypeError:
            result = defaults()
        if not isinstance(result, dict):
            raise TypeError(
                f"Plugin defaults {plugin_app.defaults!r} returned "
                f"{type(result).__name__}, expected dict"
            )
        return dict(result)
    # Modules that shipped their own defaults through the `register_defaults`
    # seam. Consulted after plugins and before the built-in dispatch below, so
    # a registered module is served without editing this function -- which is
    # the whole point of the seam, and without this line every
    # `register_defaults` call in the codebase is inert.
    #
    # Import first, ask second. `register_defaults` runs at the module's own
    # import, so the seam only answers for a module something has already
    # imported -- and a pipeline module has no reason to be imported by the
    # process that is merely drawing its settings panel. `register_app(...,
    # defaults_module=...)` names it; this is what makes the panel appear
    # instead of an empty form.
    _import_registered_defaults_module(app_key)
    from spacr.settings import defaults_for, has_registered_defaults
    if has_registered_defaults(app_key):
        return defaults_for(app_key, {})
    from spacr.settings import (
        get_identify_masks_finetune_default_settings,
        set_default_analyze_screen,
        set_default_settings_preprocess_generate_masks,
        get_automated_motility_assay_default_settings,
        get_measure_crop_settings,
        deep_spacr_defaults,
        set_default_generate_barecode_mapping,
        set_default_umap_image_settings,
        get_analyze_recruitment_default_settings,
        get_check_cellpose_models_default_settings,
        get_analyze_plaque_settings,
        set_analyze_invasion_defaults,
        get_perform_regression_default_settings,
        get_train_cellpose_default_settings,
        get_default_generate_activation_map_settings,
        get_timelapse_settings,
        set_analyze_replication_defaults,
    )
    if app_key == "mask":
        # Timelapse tracking and the automated motility assay are first-class
        # modules of their own now (app keys 'timelapse' / 'motility'), so the
        # Mask module edits neither set of knobs. The keys are dropped from the
        # *editable* dict only — preprocess_generate_masks re-applies
        # set_default_settings_preprocess_generate_masks internally, so a Mask
        # run still gets timelapse=False / motility_analysis=False, and a CSV
        # driven straight through the API keeps working unchanged.
        s = set_default_settings_preprocess_generate_masks(settings={})
        for key in timelapse_and_motility_keys():
            s.pop(key, None)
        return s
    if app_key == "timelapse":
        s = get_timelapse_settings(settings={})
        # The Timelapse module tracks objects; running the assay is what the
        # Motility Assay module is for, so its inline gate isn't offered here.
        # `timelapse` itself stays visible (and True) so a mask settings CSV
        # from before the split still round-trips through this screen.
        s.pop("motility_analysis", None)
        return s
    if app_key == "motility":
        s = get_automated_motility_assay_default_settings(settings={})
        # `motility_analysis` is the Mask-pipeline gate for the inline assay
        # (spacr.object), not a knob of the assay itself — opening the
        # Motility module *is* asking for the assay.
        s.pop("motility_analysis", None)
        return s
    if app_key == "measure":
        return get_measure_crop_settings(settings={})
    if app_key == "external_masks":
        from spacr.external_masks import default_settings
        return default_settings({})
    if app_key == "classify":
        settings = deep_spacr_defaults(settings={})
        settings["src"] = []
        return settings
    if app_key == "umap":
        settings = set_default_umap_image_settings(settings={})
        # The original controls describe one lab's c1/c2/c3 plate convention.
        # Keep them as API-compatible backend defaults, but do not expose them
        # in the general UMAP UI. ``exclude_rows`` replaces them with rules
        # based on the columns and values in the user's own database.
        for key in (
            "col_to_compare", "pos", "neg", "mix",
            "embedding_by_controls", "exclude_conditions",
        ):
            settings.pop(key, None)
        return settings
    if app_key == "train_cellpose":
        return get_train_cellpose_default_settings(settings={})
    if app_key == "ml_analyze":
        return set_default_analyze_screen(settings={})
    if app_key == "cellpose_masks":
        return get_identify_masks_finetune_default_settings(settings={})
    if app_key == "cellpose_all":
        return get_check_cellpose_models_default_settings(settings={})
    if app_key == "map_barcodes":
        return set_default_generate_barecode_mapping(settings={})
    if app_key == "regression":
        return get_perform_regression_default_settings(settings={})
    if app_key == "recruitment":
        return get_analyze_recruitment_default_settings(settings={})
    if app_key == "activation":
        return get_default_generate_activation_map_settings(settings={})
    if app_key == "invasion":
        return set_analyze_invasion_defaults(settings={})
    if app_key == "replication":
        return set_analyze_replication_defaults(settings={})
    if app_key == "analyze_plaques":
        return get_analyze_plaque_settings(settings={})
    if app_key in ("annotate", "make_masks"):
        # These are interactive apps; return minimal placeholder.
        return {"src": "path to images"}
    return {"src": "path"}


# Per-app category suppression. Keys not in a shown category fall into the
# trailing "Other" section, so the setting stays reachable — only the tab goes.
_APP_HIDDEN_CATEGORIES: Dict[str, set] = {
    "classify": {"Cellpose"},
    # Mask no longer owns tracking or the motility assay — those are the
    # 'timelapse' and 'motility' modules. resolve_default_settings already
    # drops the keys so nothing spills into "Other"; this entry is the
    # declaration of intent and keeps the tabs gone even if a future default
    # re-introduces one of the keys.
    "mask": {"Timelapse", "Motility (beta)", "Motility Advanced (beta)"},
    # The Timelapse module tracks objects; the motility assay is its own
    # module and its ~50 knobs would swamp the tracking settings.
    "timelapse": {"Motility (beta)", "Motility Advanced (beta)"},
}

# Options that are enumerations for one module but not necessarily for every
# setting with the same generic key.  Keeping these app-scoped avoids turning
# unrelated ``mode`` fields into sequencing controls.
_APP_COMBO_OPTIONS: Dict[str, Dict[str, List[Any]]] = {
    "umap": {
        "batch_correction": [
            "none", "control_center", "robust_zscore", "center", "zscore",
            # combat is last because it is the only one that needs an
            # answer from the user first: without batch_covariate_column
            # it refuses to run rather than deleting the contrast the
            # screen is measuring. See spacr.batch_correction._combat.
            "combat",
        ],
        "batch_missing_control": ["error", "skip"],
    },
    "ml_analyze": {
        "batch_correction": [
            "none", "control_center", "robust_zscore", "center", "zscore",
            # combat is last because it is the only one that needs an
            # answer from the user first: without batch_covariate_column
            # it refuses to run rather than deleting the contrast the
            # screen is measuring. See spacr.batch_correction._combat.
            "combat",
        ],
        "batch_missing_control": ["error", "skip"],
    },
    "regression": {
        "batch_correction": [
            "none", "control_center", "robust_zscore", "center", "zscore",
            # combat is last because it is the only one that needs an
            # answer from the user first: without batch_covariate_column
            # it refuses to run rather than deleting the contrast the
            # screen is measuring. See spacr.batch_correction._combat.
            "combat",
        ],
        "batch_missing_control": ["error", "skip"],
    },
    "classify": {
        "evaluation_calibration": ["temperature", "none"],
    },
    "external_masks": {
        "layout": ["auto", "flat", "well", "plate_well"],
        "z_handling": ["max", "first"],
        "plate_naming": ["index", "name"],
    },
    "map_barcodes": {
        "mode": ["paired", "single"],
        "single_direction": ["R1", "R2"],
        "comp_type": ["zlib", "lzo", "bzip2", "blosc"],
    },
}


# App-specific category layouts. ``@Name`` expands the corresponding legacy
# category; plain entries are individual setting keys. The backend settings
# dictionaries remain unchanged — this controls only the order and grouping in
# Qt, just like the Classify (CV) regroup below.
_APP_CATEGORY_SPECS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "ml_analyze": (
        ("Data & Controls", (
            "src", "location_column", "positive_control", "negative_control",
            "annotation_column",
        )),
        ("Feature Preparation", (
            "channel_of_interest", "exclude", "nuclei_limit",
            "pathogen_limit", "remove_highly_correlated_features",
            "remove_low_variance_features", "minimum_cell_count",
        )),
        ("Plate & Batch Correction", (
            "batch_correction", "batch_column", "batch_control_column",
            "batch_control_values", "batch_covariate_column",
            "batch_combat_mean_only", "batch_min_samples",
            "batch_missing_control",
        )),
        ("Classifier & Validation", (
            "model_type_ml", "n_estimators", "learning_rate", "test_size",
            "cross_validation", "reg_alpha", "reg_lambda",
        )),
        ("Feature Selection & Importance", (
            "prune_features", "top_features", "n_repeats",
        )),
        ("Output & Database", ("save_to_db",)),
        ("Plots & Heatmaps", (
            "cmap", "heatmap_feature", "grouping", "min_max",
        )),
        ("Runtime & Reliability", ("verbose", "n_jobs")),
    ),
    "mask": (
        ("Input & Metadata", (
            "src", "cell_channel", "nucleus_channel", "pathogen_channel",
            "organelle_channel", "channels", "magnification",
            "metadata_type", "custom_regex",
        )),
        ("Workflow & Test Run", (
            "preprocess", "masks", "test_mode", "test_images", "resume",
            "dry_run",
        )),
        ("Image Preprocessing", (
            "normalize", "lower_percentile", "randomize", "batch_fields",
            "all_to_mip", "upscale", "upscale_factor", "consolidate",
            "denoise",
        )),
        ("Cell Segmentation", ("@Cell",)),
        ("Nucleus Segmentation", ("@Nucleus",)),
        ("Pathogen Segmentation", ("@Pathogen",)),
        ("Organelle Segmentation", ("@Organelle",)),
        ("Quality Control", ("@Segmentation QC",)),
        ("Volumetric Processing (Beta)", ("@3D Settings (Beta)",)),
        ("Time Axes & Tracking (Beta)", ("@4D Settings (Beta)",)),
        ("Visualization & Diagnostics", (
            "plot", "cmap", "figuresize", "normalize_plots",
            "examples_to_plot",
        )),
        ("Output & Storage", (
            "save", "delete_intermediate", "keep_intermediate",
            "keep_original_images", "save_original_images", "keep_npz",
            "compression", "filter", "merge_pathogens",
        )),
        ("Runtime & Reliability", (
            "strict_errors", "max_failure_rate", "on_error",
            "on_error_attempts", "on_error_backoff", "random_seed", "verbose", "n_jobs",
            "batch_size", "pipeline_style", "diameter_estimate_n_fields",
        )),
    ),
    "measure": (
        ("Input & Experiment", ("src", "experiment")),
        ("Mask & Channel Mapping", (
            "channels", "cell_mask_dim", "nucleus_mask_dim",
            "pathogen_mask_dim", "organelle_mask_dim", "cytoplasm",
            "timelapse", "timelapse_objects",
        )),
        ("Measurement Features", (
            "save_measurements", "calculate_correlation",
            "manders_thresholds", "homogeneity", "homogeneity_distances",
            "radial_dist", "distance_gaussian_sigma",
            # Not a segmentation control -- it decides which organelle summary
            # TABLES a measure run writes, so it belongs with the other
            # what-gets-measured settings rather than under the mask
            # pipeline's Organelle Segmentation heading.
            "summarize_organelles_by",
        )),
        ("Object Filtering", (
            "uninfected", "cell_min_size", "cytoplasm_min_size",
            "nucleus_min_size", "pathogen_min_size", "organelle_min_size",
            "merge_edge_pathogen_cells",
        )),
        ("Crop Output", (
            "save_png", "save_arrays", "crop_mode", "png_size", "png_dims",
            "dialate_pngs", "dialate_png_ratios", "use_bounding_box",
            "normalize", "normalize_by",
        )),
        ("Preview & Diagnostics", ("plot", "test_mode", "test_nr")),
        ("3D Calibration (Beta)", (
            "anisotropy", "voxel_size_z_um", "voxel_size_xy_um",
        )),
        ("Runtime & Reliability", (
            "resume", "strict_errors", "max_failure_rate", "on_error",
            "on_error_attempts", "on_error_backoff", "random_seed", "dry_run",
            "verbose", "n_jobs",
        )),
    ),
    "timelapse": (
        ("Input & Metadata", (
            "src", "cell_channel", "nucleus_channel", "pathogen_channel",
            "organelle_channel", "channels", "magnification",
            "metadata_type", "custom_regex",
        )),
        ("Acquisition & Axes", (
            "timelapse", "t_stack", "t_axis_order", "t_axis",
            "frame_interval_s", "z_stack", "z_segmentation_mode", "z_axis",
            "z_projection", "anisotropy", "voxel_size_z_um",
            "voxel_size_xy_um", "stitch_threshold",
        )),
        ("Image Preprocessing", (
            "normalize", "lower_percentile", "randomize", "batch_fields",
            "all_to_mip", "upscale", "upscale_factor", "consolidate",
            "denoise",
        )),
        ("Cell Segmentation", ("@Cell",)),
        ("Nucleus Segmentation", ("@Nucleus",)),
        ("Pathogen Segmentation", ("@Pathogen",)),
        ("Organelle Segmentation", ("@Organelle",)),
        ("Quality Control", ("@Segmentation QC",)),
        ("Tracking Setup", (
            "timelapse_objects", "timelapse_frame_limits",
            "timelapse_remove_transient", "fps",
        )),
        ("Tracking Backends", (
            "timelapse_mode", "trackastra_model", "trackastra_linking",
            "ultrack_max_distance", "ultrack_division_weight",
            "ultrack_contour_sigma", "ultrack_n_workers",
            "timelapse_displacement", "timelapse_memory",
            "t_track_backend", "t_link_threshold",
            "t_max_displacement_px", "t_max_displacement_um",
            "t_project_for_tracking",
        )),
        ("Visualization & Diagnostics", (
            "plot", "cmap", "figuresize", "normalize_plots",
            "examples_to_plot",
        )),
        ("Output & Storage", (
            "save", "delete_intermediate", "keep_intermediate",
            "keep_original_images", "save_original_images", "keep_npz",
            "compression", "filter", "merge_pathogens",
        )),
        ("Runtime & Reliability", (
            "preprocess", "masks", "test_mode", "test_images", "resume",
            "strict_errors", "max_failure_rate", "on_error",
            "on_error_attempts", "on_error_backoff", "random_seed", "dry_run", "verbose",
            "n_jobs", "batch_size", "pipeline_style",
            "diameter_estimate_n_fields",
        )),
    ),
    "motility": (
        ("Objects & Channels", (
            "src", "tracked_object", "cell_channel", "nucleus_channel",
            "pathogen_channel", "channels",
        )),
        ("Spatial & Temporal Calibration", (
            "seconds_per_frame", "pixels_per_um",
        )),
        ("Motion Filtering", (
            "max_displacement", "straightness_threshold",
            "straightness_filter", "zscore_thresh",
        )),
        ("Infection Classification", (
            "infection_intensity_strategy", "infection_intensity_qc_scope",
            "infection_intensity_mode", "infection_intensity_n_bins",
            "db_table_name", "reuse_existing_measurements",
            "infection_xgb_proba_column", "infection_xgb_drop_ambiguous",
            "infection_xgb_ambiguous_low", "infection_xgb_ambiguous_high",
        )),
        ("XGBoost Infection Model", (
            "infection_xgb_min_cells_per_class",
            "infection_xgb_n_estimators", "infection_xgb_max_depth",
            "infection_xgb_learning_rate", "infection_xgb_subsample",
            "infection_xgb_colsample_bytree", "infection_xgb_reg_lambda",
            "infection_xgb_random_state", "infection_xgb_n_jobs",
            "infection_xgb_proba_threshold", "infection_xgb_margin",
            "infection_xgb_top_features",
        )),
        ("Infection Clustering", (
            "infection_pca_n_clusters", "infection_pca_random_state",
            "infection_pca_pathogen_weight", "infection_pca_log_intensity",
            "infection_pca_min_silhouette",
            "infection_pca_min_gt_separation", "infection_pca_max_cells",
        )),
        ("Embedding Search", (
            "infection_pca_umap_search",
            "infection_pca_umap_n_neighbors_grid",
            "infection_pca_umap_min_dist_grid",
            "infection_pca_umap_n_neighbors",
            "infection_pca_umap_min_dist", "infection_pca_tsne_search",
            "infection_pca_tsne_perplexity_grid",
            "infection_pca_tsne_learning_rate_grid",
            "infection_pca_tsne_perplexity",
        )),
        ("Motility Plots & QC", (
            "motility_ylim", "motility_xlim",
            "infection_intensity_qc_graphs",
        )),
        ("Runtime & Reliability", ("n_jobs",)),
    ),
    "regression": (
        ("Input Tables", ("metadata_files", "score_data", "count_data")),
        ("Controls & Plate Design", (
            "plateID", "positive_control", "negative_control", "controls",
            "control_wells", "filter_column", "filter_value",
        )),
        ("Plate & Batch Correction", (
            "batch_correction", "batch_column", "batch_control_column",
            "batch_control_values", "batch_covariate_column",
            "batch_combat_mean_only", "batch_min_samples",
            "batch_missing_control",
        )),
        ("Model & Covariates", (
            "regression_type", "dependent_variable", "score_column",
            "invert_dependent_variable", "agg_type", "transform",
            "alpha", "cov_type", "random_row_column_effects",
        )),
        # The estimator-specific knobs, added by the robust and regularised
        # fits after this layout was first written. They landed in
        # "Additional Settings" -- the bucket a layout exists to keep empty --
        # because only the shared estimator settings above were named.
        ("Estimator Tuning", (
            "l1_ratio", "quantile", "huber_t", "tolerance",
            "hinge_threshold", "hinge_n_boot", "lasso_n_boot",
            "lasso_selection_threshold",
        )),
        ("Hit Calling & Outliers", (
            "min_cell_count", "fraction_threshold", "target_unique_count",
            "outlier_detection", "threshold_method", "threshold_multiplier",
            "min_n", "toxo",
        )),
        ("Regression Plots", (
            "volcano", "log_x", "log_y", "x_lim", "y_lims",
            "split_axis_lims",
        )),
        ("Runtime & Reliability", (
            "strict_errors", "max_failure_rate", "on_error",
            "on_error_attempts", "on_error_backoff", "random_seed", "verbose",
        )),
    ),
    "activation": (
        ("Model & Data", (
            "dataset", "model_path", "model_type", "image_size",
            "object_type", "channels",
        )),
        ("Attribution Method", (
            "cam_type", "target_layer", "smoothgrad_samples",
            "smoothgrad_sigma", "occlusion_window", "occlusion_stride",
            "ig_steps", "ig_baseline",
        )),
        ("Attribution Validation", (
            "attribution_steps", "attribution_baseline", "sanity_check",
        )),
        ("Map Display", (
            "normalize", "normalize_input", "overlay", "plot",
        )),
        ("Map Quantification", ("correlation", "manders_thresholds")),
        ("Output & Runtime", (
            "save", "shuffle", "batch_size", "n_jobs",
        )),
    ),
    "recruitment": (
        ("Data source", ("src",)),
        ("Mask & Channel Mapping", (
            "cell_mask_dim", "cell_chann_dim", "nucleus_mask_dim",
            "nucleus_chann_dim", "pathogen_mask_dim", "pathogen_chann_dim",
            "channel_dims", "channel_of_interest",
        )),
        ("Object Filtering", (
            "cell_size_range", "cell_intensity_range", "nucleus_size_range",
            "nucleus_intensity_range", "pathogen_size_range",
            "pathogen_intensity_range", "cells_per_well",
            "target_intensity_min", "nuclei_limit", "pathogen_limit",
        )),
        ("Plate Layout & Controls", ("@Plate Layout & Controls",)),
        ("Plots & Diagnostics", (
            "plot", "figuresize", "plot_control", "plot_nr",
        )),
    ),
    "invasion": (
        ("Assay Inputs", ("src", "parasite_table", "compartment")),
        ("Channels & Intensity", (
            "outside_channel", "total_channel", "intensity_statistic",
            "background_correction", "min_total_intensity",
        )),
        ("Thresholding", (
            "outside_threshold_method", "outside_threshold",
            "threshold_agreement_tolerance", "threshold_sensitivity",
            "bimodality_cutoff", "extracellular_class",
        )),
        ("Controls & Minimum Counts", (
            "control_wells", "control_quantile", "min_control_objects",
            "min_objects_for_threshold", "min_objects_for_bimodality",
            "min_parasites_per_well", "inflation_warn",
        )),
        ("Object Filtering", ("min_parasite_area", "max_parasite_area")),
        ("Condition Metadata", ("@Plate Layout & Controls",)),
        ("Assay Output", (
            "cmap", "qc_plot_max_panels", "seed_wells_from_cells", "save",
        )),
        ("Runtime & Reliability", ("verbose",)),
    ),
    # -- the three Cellpose-facing modules -------------------------------
    #
    # All three used to render the shared "Cellpose" category as one drop of
    # ten to thirteen knobs. They are not one decision: the model you run,
    # the thresholds that decide how much it finds, the geometry it sees and
    # the background correction applied before it are four separate
    # questions, asked at four different times. The groups below are the same
    # four in all three modules so that moving between them is not a
    # relearning exercise.
    "cellpose_masks": (
        ("Input & Channels", (
            "src", "channels", "grayscale", "invert", "normalize",
            "percentiles",
        )),
        ("Model", ("model_name", "custom_model", "diameter")),
        ("Detection Thresholds", (
            "CP_prob", "flow_threshold", "rescale", "resample", "fill_in",
        )),
        ("Image Geometry", ("resize", "target_height", "target_width")),
        ("Background & Denoising", (
            "remove_background", "background", "Signal_to_noise",
        )),
        ("Output & Runtime", ("save", "batch_size", "verbose")),
    ),
    "cellpose_all": (
        ("Input & Channels", (
            "channels", "grayscale", "invert", "normalize", "percentiles",
        )),
        ("Model", ("diameter",)),
        ("Detection Thresholds", ("CP_prob", "flow_threshold")),
        ("Image Geometry", ("resize", "target_height", "target_width")),
        ("Background & Denoising", (
            "remove_background", "background", "Signal_to_noise",
        )),
        ("Output & Runtime", ("plot", "save", "batch_size", "verbose")),
    ),
    "train_cellpose": (
        ("Starting Point", ("model_type", "from_scratch", "model_name")),
        ("Training Schedule", (
            "n_epochs", "learning_rate", "weight_decay", "batch_size",
            "augment",
        )),
        ("Image Geometry", (
            "width_height", "target_size", "diameter", "resize",
        )),
        ("Background & Denoising", (
            "remove_background", "background", "Signal_to_noise",
        )),
        ("Output & Runtime", ("verbose",)),
    ),
    "analyze_plaques": (
        ("Input & Channels", ("src", "masks")),
        ("Model", ("diameter",)),
        ("Detection Thresholds", (
            "CP_prob", "flow_threshold", "rescale", "resample", "fill_in",
        )),
        ("Image Geometry", ("resize", "target_height", "target_width")),
        ("Background & Denoising", (
            "remove_background", "background", "Signal_to_noise",
        )),
        ("Output & Runtime", ("save", "batch_size", "verbose")),
    ),
    "map_barcodes": (
        ("Sequencing Input", ("src", "mode", "single_direction")),
        ("Barcode References", ("grna_csv", "row_csv", "column_csv")),
        ("Read Parsing", (
            "target_sequence", "regex", "offset_start", "expected_end",
        )),
        ("Output & Storage", (
            "save_h5", "comp_type", "comp_level", "fill_na",
        )),
        ("Runtime & Reliability", ("chunk_size", "n_jobs", "test")),
    ),
    "barcode_qc": (
        ("Reference & Count Tables", (
            "grna_csv", "row_csv", "column_csv", "count_data", "qc_data",
        )),
        ("Well Expectations", (
            "target_grnas_per_well", "target_statistic", "min_reads_per_well",
        )),
        ("Starvation & Exclusion", (
            "starved_read_fraction", "exclude_starved_wells",
        )),
        ("Position & Collision Checks", (
            "position_effect_ratio", "collision_max_distance",
        )),
        ("Threshold Sweep", ("sweep_span", "sweep_points")),
        ("QC Output", ("dst", "plot", "save")),
        ("Runtime & Reliability", ("verbose",)),
    ),
    "illumination": (
        ("Input & Channels", ("src", "channels")),
        ("Correction Model", (
            "illumination_correction", "illumination_model",
            "illumination_estimator", "illumination_degree",
            "illumination_dark",
        )),
        ("Field Sampling", (
            "illumination_per_plate", "illumination_max_fields",
        )),
        ("QC & Failure Handling", (
            "illumination_qc", "illumination_on_missing",
        )),
    ),
    # Power / Design draws its own screen, so these groups are never a
    # settings form. They are still the layout of record: the settings diff,
    # the run journal and `utils.pretty_print_settings` all group by
    # category, and fifteen keys under one "Power analysis" heading make a
    # design change unreadable in all three.
    "power": (
        ("Library Design", (
            "power_n_genes", "power_n_grnas_per_gene",
            "power_constructs_per_well",
        )),
        ("Plate Layout", (
            "power_wells_per_plate", "power_n_plates", "power_n_replicates",
            "power_cells_per_well",
        )),
        ("Effect & Prevalence", (
            "power_effect_fold", "power_hit_rate",
            "power_background_positive_rate", "power_detection_auroc",
        )),
        ("Sequencing Depth", ("power_reads_per_well",)),
        ("Simulation", ("power_score_per", "power_backend", "power_seed")),
    ),
    "anndata_export": (
        ("Input Tables", ("src", "anndata_tables")),
        ("Output File", (
            "anndata_out", "anndata_single_table", "anndata_compression",
            "anndata_dtype",
        )),
        ("Rows & Missing Values", (
            "anndata_row_limit", "anndata_nan_policy",
        )),
        ("Post-processing", (
            "anndata_compute_umap", "anndata_register_artifact",
        )),
    ),
    "replication": (
        ("Assay Inputs", ("src", "parasite_table", "compartment")),
        ("Vacuole Assignment", (
            "vacuole_key", "vacuole_link_distance", "vacuole_link_factor",
            "parasite_count_column", "require_host_cell",
        )),
        ("Condition Metadata", (
            "cell_types", "cell_plate_metadata", "pathogen_types",
            "pathogen_plate_metadata", "treatments",
            "treatment_plate_metadata", "group_column", "level",
            "change_plate",
        )),
        ("Object Filtering", (
            "min_parasite_area", "max_parasite_area",
        )),
        ("Replication Scoring", (
            "max_parasites_per_vacuole", "non_power_of_two_warn",
            "seed_wells_from_cells",
        )),
        ("Assay Output", ("cmap", "save")),
        ("Runtime & Reliability", ("verbose",)),
    ),
}


#: Settings a first-time user of a module has to touch beyond its first
#: group, in the same ``@Section``-or-key language as
#: :data:`_APP_CATEGORY_SPECS`.
#:
#: The first group of a curated layout is by construction the "what you must
#: set" group — every layout in this module opens with the inputs — so it is
#: taken as essential automatically and never restated here. This table only
#: adds the second thing: Measure's mask-to-channel mapping, Regression's
#: model choice, Train Cellpose's schedule. Anything naming a key or a group
#: that no longer exists is dropped silently, the same way a spec token is,
#: so a stale entry costs a row of disclosure and never an exception.
_APP_ESSENTIAL_EXTRAS: Dict[str, Tuple[str, ...]] = {
    "mask": ("preprocess", "masks", "test_mode", "test_images", "plot",
             "save"),
    "timelapse": ("timelapse", "t_stack", "frame_interval_s",
                  "timelapse_objects", "test_mode", "save"),
    "measure": ("@Mask & Channel Mapping", "test_mode"),
    "motility": ("@Spatial & Temporal Calibration",),
    "ml_analyze": ("channel_of_interest", "model_type_ml"),
    "regression": ("@Controls & Plate Design", "regression_type",
                   "dependent_variable"),
    "activation": ("cam_type", "target_layer"),
    "replication": ("@Vacuole Assignment",),
    "recruitment": ("@Mask & Channel Mapping",),
    "invasion": ("@Channels & Intensity",),
    "cellpose_masks": ("@Model",),
    "cellpose_all": ("@Model",),
    "analyze_plaques": ("@Model",),
    "train_cellpose": ("n_epochs", "learning_rate"),
    "map_barcodes": ("@Barcode References",),
    "barcode_qc": ("@Well Expectations",),
    "illumination": ("illumination_correction", "illumination_model"),
    "anndata_export": ("anndata_out",),
    "classify": ("@Labels & Classes", "model_type", "train_channels"),
    "umap": ("tables", "reduction_method", "color_by"),
    "external_masks": ("channels", "experiment"),
}


def _expand_layout_tokens(
    source: Dict[str, List[str]],
    tokens: Tuple[str, ...],
) -> List[str]:
    """Resolve ``@Section``-or-key tokens against a category map, in order.

    The same token language :data:`_APP_CATEGORY_SPECS` uses, so a layout and
    the essentials drawn from it can never disagree about what ``@Cell``
    means. (:func:`_categories_from_spec` keeps its own copy of the loop
    because it additionally has to remember which keys earlier *sections*
    already claimed; this one resolves a single flat list.)

    Unknown tokens and keys the module does not actually have are dropped,
    and a key named twice is kept once, at its first position.
    """
    available = {key for keys in source.values() for key in keys}
    out: List[str] = []
    for token in tokens:
        candidates = (
            source.get(token[1:], []) if token.startswith("@") else [token]
        )
        for key in candidates:
            if key in available:
                out.append(key)
    return list(dict.fromkeys(out))


def essential_keys(
    app_key: str,
    categories: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """The settings a first-time user of ``app_key`` should meet first.

    Progressive disclosure needs a defensible answer to "which of these 190
    matter?", and a hand-written list per module would rot the first time a
    layout changed. So it is *derived*: the first group of the module's
    curated layout, which is always its inputs, plus whatever
    :data:`_APP_ESSENTIAL_EXTRAS` adds for that module.

    A module with no curated layout gets the first shared category, which is
    "Paths" — still the right answer, just a thinner one.

    :param app_key: the module's app key.
    :param categories: optional pre-computed :func:`categories_for_app`
        output, to save recomputing it.
    :returns: setting keys in display order, without duplicates.
    """
    cats = (categories if categories is not None
            else categories_for_app(app_key, get_categories()))
    ordered = list(cats.items())
    keys: List[str] = list(ordered[0][1]) if ordered else []
    keys.extend(
        _expand_layout_tokens(
            cats, _APP_ESSENTIAL_EXTRAS.get(str(app_key or ""), ())
        )
    )
    return list(dict.fromkeys(keys))


def _categories_from_spec(
    source: Dict[str, List[str]],
    spec: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> Dict[str, List[str]]:
    """Expand one app layout and retain future settings under a named bucket."""
    ordered: Dict[str, List[str]] = {}
    assigned = set()
    available = {key for keys in source.values() for key in keys}
    for title, tokens in spec:
        keys: List[str] = []
        for token in tokens:
            if token.startswith("@"):
                # A group reference can only mean what the shared map says
                # it means, so it is filtered by what is actually in there.
                candidates = [key for key in source.get(token[1:], [])
                              if key in available]
            else:
                # A literal key is the spec ASSERTING where that setting
                # belongs, and it outranks the shared category map — which
                # for Barcode QC and Illumination has never heard of their
                # keys at all. Filtering literals by `available` sent all
                # eleven of Barcode QC's checks to the trailing "Other"
                # bucket, which is the exact thing the layout exists to
                # prevent. Whether the key exists is decided at render time,
                # where `build_sections` already drops any key that produced
                # no widget.
                candidates = [token]
            for key in candidates:
                if key not in assigned:
                    assigned.add(key)
                    keys.append(key)
        ordered[title] = keys

    remaining = []
    for keys in source.values():
        for key in keys:
            if key not in assigned:
                assigned.add(key)
                remaining.append(key)
    if remaining:
        ordered["Additional Settings"] = remaining
    return ordered


def get_categories() -> Dict[str, List[str]]:
    """Return the {category_name: [setting keys]} mapping."""
    from spacr.settings import categories
    return categories


#: Below this many settings a module cannot render as an undifferentiated
#: list — six rows fit on one screen and read as one group whatever they are
#: called. Modules at or under it are exempt from :func:`has_curated_layout`;
#: everything above it has to say what its groups are.
CURATION_THRESHOLD = 6

#: Modules whose layout is curated inline in :func:`categories_for_app`
#: rather than declared in :data:`_APP_CATEGORY_SPECS`.
#:
#: Classify is the odd one out on purpose: its ten groups are built as a
#: literal ``ordered`` dict because several of them list keys that are in no
#: shared category at all, which the ``@Name``-expanding spec form cannot
#: express. UMAP and External Masks reshape the shared categories in place —
#: they add groups ("UMAP Display", "Input mapping") rather than replacing
#: the whole layout, and a spec would have to restate every key they leave
#: alone. All three are curated; none of them is a spec.
_INLINE_LAYOUT_APPS = frozenset({"classify", "umap", "external_masks"})


def has_curated_layout(app_key: str) -> bool:
    """Return True when ``app_key``'s settings panel has a layout of its own.

    "Of its own" means somebody decided what this module's groups are — a
    :data:`_APP_CATEGORY_SPECS` entry, an inline regroup in
    :func:`categories_for_app`, or a plugin that shipped ``categories``.

    Falling back to the shared category map is *not* curated. That map is
    keyed by what a setting is (a path, a plot option, "Advanced"), not by
    what the module does with it, so a module that relies on it renders as
    however many buckets its keys happen to fall into — which for Cellpose
    Masks was thirteen knobs under one "Cellpose" heading.

    :param app_key: the module's app key.
    """
    key = str(app_key or "")
    if key in _APP_CATEGORY_SPECS or key in _INLINE_LAYOUT_APPS:
        return True
    try:
        from spacr.plugins import get_app
        plugin_app = get_app(key)
    except Exception:
        return False
    return bool(plugin_app is not None and plugin_app.categories)


def needs_curated_layout(app_key: str) -> bool:
    """Return True when ``app_key`` has enough settings to need grouping.

    Interactive modules whose settings dict is the ``{"src": ...}``
    placeholder render a bespoke screen, not the shared form; they have
    nothing to group. :data:`CURATION_THRESHOLD` draws the line.

    :param app_key: the module's app key.
    """
    try:
        return len(resolve_default_settings(app_key)) > CURATION_THRESHOLD
    except Exception:
        # An app whose defaults will not resolve has no settings panel to
        # judge. Reporting "needs a layout" would fail the invariant test for
        # a reason that has nothing to do with layouts.
        return False


def categories_for_app(
    app_key: str,
    categories: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Return category keys after applying module-specific relocations.

    Map Barcodes previously showed an ``Advanced`` tab containing only
    ``n_jobs`` and a ``Model Training`` tab containing only ``test``.  Both
    controls belong to the sequencing run, but changing the global category
    table would also move training controls in unrelated modules.
    """
    try:
        from spacr.plugins import get_app
        plugin_app = get_app(app_key)
    except Exception:
        plugin_app = None
    if plugin_app is not None and plugin_app.categories:
        return {
            str(name): list(keys)
            for name, keys in plugin_app.categories.items()
        }
    result = {name: list(keys) for name, keys in categories.items()}
    if app_key == "external_masks":
        input_keys = (
            "inputs", "dst", "recursive", "layout", "z_handling",
            "plate_naming", "overwrite", "preview_only",
        )
        for keys in result.values():
            for key in input_keys:
                while key in keys:
                    keys.remove(key)
        result = {"Input mapping": list(input_keys), **result}
    # Map Barcodes used to relocate `n_jobs` and `test` into "Sequencing"
    # here, so the module would stop rendering an "Advanced" tab holding one
    # setting and a "Model Training" tab holding another. That left thirteen
    # unrelated keys in one "Sequencing" drop; `_APP_CATEGORY_SPECS` now
    # names all five groups the module actually has, which places those two
    # keys — and every other one — explicitly. The relocation is not deleted
    # behaviour, it is superseded behaviour.
    if app_key == "umap":
        batch_correction = (
            "batch_correction", "batch_column", "batch_control_column",
            "batch_control_values", "batch_covariate_column",
            "batch_combat_mean_only", "batch_min_samples",
            "batch_missing_control",
        )
        display = (
            "figuresize", "dot_size", "point_color", "point_alpha",
            "outline_width", "umap_canvas_width", "umap_sidebar_width",
            "img_zoom", "image_nr", "plot_images", "remove_image_canvas",
            "plot_points", "plot_outlines", "smooth_lines",
            "plot_by_cluster", "plot_cluster_grids", "black_background",
            "save_figure",
        )
        for keys in result.values():
            for key in (*display, *batch_correction):
                while key in keys:
                    keys.remove(key)
        result["Plate & Batch Correction"] = list(batch_correction)
        result["UMAP Display"] = list(display)
    if app_key in _APP_CATEGORY_SPECS:
        result = _categories_from_spec(result, _APP_CATEGORY_SPECS[app_key])
    if app_key == "classify":
        ordered = {
            "Plate Sources & Workflow": [
                "src", "experiment", "generate_training_dataset", "train",
                "test"],
            "Labels & Classes": [
                "dataset_mode", "classes", "annotation_column",
                "annotated_classes", "class_metadata", "metadata_type_by",
                "metadata_item_1_name", "metadata_item_1_value",
                "metadata_item_2_name", "metadata_item_2_value",
                "custom_measurement"],
            "Crops & Dataset Split": [
                "tables", "channel_of_interest", "png_type", "file_type",
                "crop_source", "size", "test_split", "balance_to_smallest",
                "write_random_annotation_column"],
            "Model Architecture": [
                "model_type", "custom_model", "custom_model_path",
                "resume_checkpoint", "train_channels", "image_size",
                "normalize", "dropout_rate", "init_weights", "use_checkpoint"],
            "Optimization & Loss": [
                "optimizer_type", "learning_rate", "weight_decay", "amsgrad",
                "schedule", "loss_type", "class_balance", "label_smoothing",
                "focal_gamma", "focal_alpha", "logit_adjust_tau", "epochs",
                "batch_size", "augment", "gradient_accumulation",
                "gradient_accumulation_steps", "early_stopping_patience"],
            "Validation": [
                "val_split", "cross_validation_enabled",
                "cross_validation_folds", "cv_group_by", "score_threshold"],
            "Evaluation Workbench": [
                "classifier_evaluation", "nested_cv_inner_folds",
                "evaluation_calibration", "evaluation_bins",
                "evaluation_fail_on_leakage", "leakage_audit_train_test",
                "leakage_hash_content", "leakage_require_identity"],
            "Full Dataset & Inference": [
                "generate_full_dataset", "apply_model_to_dataset",
                "tar_path", "dataset", "file_metadata", "sample",
                "model_path", "n_top_examples"],
            "Monitoring & Runtime": [
                "plot", "tensorboard", "intermedeate_save", "pin_memory",
                "random_seed", "n_jobs", "verbose", "strict_errors",
                "max_failure_rate"],
        }
        moved = {key for keys in ordered.values() for key in keys}
        leftovers = []
        for keys in result.values():
            leftovers.extend(key for key in keys if key not in moved)
        if leftovers:
            ordered["Additional Settings"] = list(dict.fromkeys(leftovers))
        result = ordered
    if app_key == "external_masks":
        filter_keys = (
            "uninfected", "cell_min_size", "cytoplasm_min_size",
            "nucleus_min_size", "pathogen_min_size", "organelle_min_size",
            "merge_edge_pathogen_cells",
        )
        for keys in result.values():
            for key in filter_keys:
                while key in keys:
                    keys.remove(key)
        reordered: Dict[str, List[str]] = {}
        for name, keys in result.items():
            reordered[name] = keys
            if name == "Measurements":
                reordered["Filter settings"] = list(filter_keys)
        result = reordered
    return result


# ---------------------------------------------------------------------------
# Category help — one blurb per settings CATEGORY
# ---------------------------------------------------------------------------
#
# A category is a collapsible header in a module's settings panel; the map
# above decides which keys land under which header. These are the blurbs the
# panel shows for the header itself, keyed by the title uppercased and
# stripped, because that is what a rendered ``Section`` has in hand.
#
# They are deliberately NOT restatements of the heading. Someone reading
# "Image Preprocessing" already knows the words; what they cannot tell is what
# the group decides and whether today's problem lives inside it. Each entry
# therefore says what the settings determine and when you would open them.
#
# ``CATEGORY_TOOLTIPS_BY_APP`` overrides this table for the handful of
# headings that genuinely mean different things per module: "Cellpose" is a
# training schedule under Train Cellpose and a set of inference thresholds
# under Cellpose Masks, and "Runtime & Reliability" carries Timelapse's stage
# toggles but only ``n_jobs`` under Motility.
#
# ``app_screen`` re-exports this as ``SECTION_HINTS`` for the tests and
# integrations that already read it by that name.
CATEGORY_TOOLTIPS: Dict[str, str] = {
    # -- shared headings from spacr.settings.categories --------------------
    "PATHS":
        "Where the module reads its images or tables from, plus any lookup "
        "file it needs alongside them. Set these when you point the module "
        "at a new plate or experiment; every other group assumes they are "
        "right.",
    "GENERAL":
        "The few decisions the rest of the run depends on: which channel is "
        "which, whether intensities are normalised, and whether preview "
        "figures are drawn. Worth a look on any dataset you have not run "
        "before.",
    "CELL":
        "How the cell mask is found and cleaned up — model, expected "
        "diameter, probability and flow thresholds, background floor. Open "
        "it when cells are missed, merged into their neighbours, or split "
        "in two.",
    "NUCLEUS":
        "How the nucleus mask is found and cleaned up — model, expected "
        "diameter, probability and flow thresholds, background floor. "
        "Nuclei are the easiest object to get right, so they are a good "
        "place to check the channel assignment.",
    "PATHOGEN":
        "How the pathogen mask is found and cleaned up — model, expected "
        "diameter, probability and flow thresholds, background floor. "
        "Tightly packed parasites fusing into one object are the usual "
        "reason to come here.",
    "ORGANELLE":
        "Everything the organelle mask needs, in the order you set it up: "
        "shape family and detection method, the background and contrast "
        "correction applied first, the knobs belonging to the method you "
        "chose, the size, intensity and border filters applied to what was "
        "found, and which parent compartment the results are summarised "
        "into. Expect to spend time here — punctate, tubular and "
        "ring-shaped organelles each want a different method.",
    "CELLPOSE":
        "How Cellpose itself is run: expected object diameter, probability "
        "and flow thresholds, rescaling and inversion. Reach for these when "
        "masks are systematically too many, too few or the wrong size; "
        "which model runs is chosen under Model Training.",
    "SEGMENTATION QC":
        "Automatic pass/fail checks on the finished masks — object counts, "
        "size and split ratios, border and foreground fractions, and how "
        "much of a plate may fail before the run is called off. Tighten "
        "them once you know what a good field looks like; loosen them when "
        "a legitimately unusual plate keeps being rejected.",
    "MEASUREMENTS":
        "Which objects are measured and which features are computed for "
        "them — intensity, morphology, texture, radial distribution and "
        "colocalisation. Switch families off to keep the table narrow and "
        "the run short; switch them on when an analysis needs a column that "
        "is not there.",
    "FILTER SETTINGS":
        "Which segmented objects survive into the measurement table: the "
        "minimum size per compartment, whether uninfected cells are kept, "
        "and whether a pathogen straddling two cells merges them. Change "
        "them when debris is being measured, or when real cells vanish.",
    "OBJECT CROPS":
        "The per-object images written next to the measurements — crop mode "
        "and size, which mask each crop is centred on, how far it is "
        "dilated, and which channels are baked in. Annotate and the CV "
        "classifier read these later, so set them before generating a "
        "training set.",
    "PLATE LAYOUT & CONTROLS":
        "The plate map: which wells hold which cell line, strain and "
        "treatment, which are the positive and negative controls, and how "
        "wells are grouped for reporting. Filled in once per plate design; "
        "everything downstream labels its results from it.",
    "TRAINING DATASET":
        "How the labelled training set is assembled from the database — "
        "annotation column versus well metadata, which crop type, how many "
        "objects per class, and how much is held back for testing. Revisit "
        "it when the classes come out imbalanced or the model sees too few "
        "examples.",
    "MODEL TRAINING":
        "Which model is fitted and how: architecture or starting weights, "
        "classes, input channels and size, epochs, optimiser, learning-rate "
        "schedule, loss and augmentation. This is where an underfitting or "
        "overfitting run gets fixed.",
    "ML CLASSIFIER":
        "The classical, non-image classifier fitted on measured features — "
        "algorithm, tree count, regularisation, feature pruning and "
        "permutation importance. Use it when the phenotype is already "
        "captured by the measurement columns and a CNN would be overkill.",
    "EMBEDDING & CLUSTERING":
        "How the feature table is reduced to two dimensions and clustered "
        "on top of that — neighbourhood size, distance metric, and the "
        "DBSCAN/KMeans parameters with their noise handling. Change these "
        "when the embedding is one undifferentiated blob, or shatters into "
        "dozens of tiny clusters.",
    "UMAP DISPLAY":
        "How the embedding is drawn, in both the static figure and the "
        "interactive explorer: point size, colour and opacity, cluster "
        "outlines, how many thumbnails are sampled, canvas and sidebar "
        "widths, and figure saving. Presentation only — none of it moves a "
        "point.",
    "ACTIVATION MAPS":
        "Attribution settings for a trained image model — which method, "
        "which layer is hooked, how the map is overlaid, and the "
        "normalisation applied at inference. Open it when you want to know "
        "what the classifier is actually looking at.",
    "PLOT":
        "What is drawn from the results and how it looks — figure size, "
        "colour map, which control is shown alongside, and how many panels "
        "are produced. Cosmetic: it changes the figures, never the numbers.",
    "TIMELAPSE":
        "Linking masks of the same object across frames when the data has a "
        "time axis. Only relevant to a time series; a single-timepoint "
        "plate ignores it.",
    "ADVANCED":
        "Run-level knobs that rarely need touching — verbosity, worker and "
        "batch sizing, background handling, and whether results are written "
        "at all. Come here to make a run quieter or lighter on the machine, "
        "or to keep a scratch run from saving anything.",
    "3D SETTINGS (BETA)":
        "Experimental volumetric handling: how the z-axis is read, whether "
        "planes are projected or stitched, and the physical voxel size used "
        "for calibration. Needed only for z-stacks — and the voxel size is "
        "what makes a 3-D measurement physically meaningful.",
    "4D SETTINGS (BETA)":
        "Experimental time-plus-volume handling: how the time axis is laid "
        "out, the interval between frames, which backend links objects, and "
        "how far one may move between frames. For data that is both a "
        "z-stack and a time series.",
    "MOTILITY (BETA)":
        "The beta motility assay run inline with the mask pipeline: whether "
        "it runs at all, and the per-object tracking parameters it uses. "
        "The standalone Motility Assay module is the fuller version of the "
        "same analysis.",
    "MOTILITY ADVANCED (BETA)":
        "Fine-grained control over the beta motility pipeline — which "
        "features are selected and the filter windows applied to tracks. "
        "Only worth opening once the basic assay runs and the tracks look "
        "wrong in a specific way.",
    "REGRESSION":
        "The model that maps screen scores onto gRNA or well effect sizes, "
        "its covariates, and the control-based threshold used to call a "
        "hit. Change the family when the score distribution breaks the "
        "assumptions the default makes.",
    "INVASION ASSAY":
        "The two-colour invasion readout: which channels carry the outside "
        "and total stains, how the outside signal is measured, how its "
        "threshold is chosen and sanity-checked, and which objects count as "
        "parasites at all. The table the parasites are read from is under "
        "Measurements.",
    "SEQUENCING":
        "How reads become barcode counts — read mode and direction, the "
        "target sequence and regex, where the barcode starts and ends, "
        "chunk size, and how the output is compressed. Match these to how "
        "the library was built and how it was sequenced.",
    "REPLICATION ASSAY":
        "How parasites are assigned to vacuoles and counted into "
        "replication states, including the warning raised when a vacuole "
        "holds a biologically implausible, non-power-of-two number of "
        "parasites.",
    "ENDODYOGENY SIZE PROXY (LEGACY)":
        "The older area-bin approximation of replication state, kept so "
        "historical analyses still reproduce. New runs should use the "
        "direct parasite-per-vacuole counts instead.",
    # -- Mask / Timelapse --------------------------------------------------
    "INPUT & METADATA":
        "The image folder, which channel holds which object, and how spaCR "
        "reads plate, well and field out of the file names. Nothing "
        "segments correctly until the channel assignment and the naming "
        "convention here are right.",
    "WORKFLOW & TEST RUN":
        "Which stages actually execute, whether this is a small test pass "
        "over a few fields, and whether an interrupted run picks up where "
        "it stopped. Start every new dataset here with a test run before "
        "committing to the full plate.",
    "IMAGE PREPROCESSING":
        "What happens to the pixels before any mask is made — intensity "
        "normalisation, projection, upscaling, denoising, and how fields "
        "are batched. Reach for it when the images are dim, noisy, or at a "
        "different scale from the one the model expects.",
    "CELL SEGMENTATION":
        "Everything that produces the cell mask: model and expected "
        "diameter, probability and flow thresholds, background removal, and "
        "the size, intensity and border filters applied afterwards. The "
        "group to open when cells are missed, merged or split.",
    "NUCLEUS SEGMENTATION":
        "Everything that produces the nucleus mask: model and expected "
        "diameter, thresholds, background removal, and the size, intensity "
        "and border filters applied afterwards. Usually the easiest object "
        "to get right, so a good sanity check on the channel assignment.",
    "PATHOGEN SEGMENTATION":
        "Everything that produces the pathogen mask: model and expected "
        "diameter, thresholds, background removal, and the size, intensity "
        "and border filters applied afterwards. Parasites packed into one "
        "vacuole fusing into a single object is the usual reason to come "
        "here.",
    "ORGANELLE SEGMENTATION":
        "Everything the organelle mask needs, in the order you set it up: "
        "shape family and detection method, the background and contrast "
        "correction applied first, the knobs belonging to the method you "
        "chose (adaptive, spot, ridge, ring, irregular, Cellpose or U-Net), "
        "the size, intensity and border filters applied to what was found, "
        "and which parent compartment the results are summarised into. The "
        "largest group in the module, because punctate, tubular and "
        "ring-shaped organelles each want a different method.",
    "QUALITY CONTROL":
        "Automatic pass/fail checks on the finished masks — object counts, "
        "size and split ratios, border and foreground fractions, and how "
        "much of a plate may fail before the run is called off. Tighten "
        "them once you know what a good field looks like; loosen them when "
        "an unusual but legitimate plate keeps being rejected.",
    "VOLUMETRIC PROCESSING (BETA)":
        "How a z-stack is turned into something segmentable — whether "
        "planes are projected or stitched, which axis is z, and the "
        "physical voxel size. Ignore it entirely for single-plane data.",
    "TIME AXES & TRACKING (BETA)":
        "How the time axis is read and, experimentally, how objects are "
        "linked between frames. The full tracking workflow is the Timelapse "
        "module; this is the inline version.",
    "VISUALIZATION & DIAGNOSTICS":
        "The diagnostic figures a run draws as it goes — how many example "
        "fields, at what size, with which colour map and normalisation. "
        "Useful while tuning, and the first thing to switch off for a long "
        "unattended run.",
    "OUTPUT & STORAGE":
        "What survives the run: which masks and images are written, which "
        "intermediates are kept, how arrays are compressed, and whether "
        "objects are filtered or merged on the way out. Disk usage is "
        "decided here.",
    "RUNTIME & RELIABILITY":
        "How hard the run pushes the machine and what it does when a field "
        "fails — worker count, batch size, the tolerated failure rate, and "
        "how much it prints. Turn strict errors on while debugging; raise "
        "the failure tolerance for a plate with known-bad fields.",
    "ACQUISITION & AXES":
        "How the file's dimensions map onto time and z, the interval "
        "between frames, and the physical voxel size. Getting the axis "
        "order right is the prerequisite for any tracking, and everything "
        "downstream inherits it.",
    "TRACKING SETUP":
        "Which objects are tracked, over which range of frames, whether "
        "short-lived tracks are discarded, and the frame rate of the movies "
        "that come out. Start here, then pick a linker under Tracking "
        "Backends.",
    "TRACKING BACKENDS":
        "Which algorithm links objects between frames — Trackastra, Ultrack "
        "or a plain distance/overlap linker — and the parameters belonging "
        "to whichever you pick. Switch backends when cells swap identities "
        "or tracks break at division.",
    # -- Measure -----------------------------------------------------------
    "INPUT & EXPERIMENT":
        "The folder holding the masked images and the experiment name the "
        "measurements are filed under. Set once at the start of a "
        "measurement run.",
    "MASK & CHANNEL MAPPING":
        "Which plane of the stack holds each mask and each intensity "
        "channel, whether a cytoplasm compartment is derived, and whether "
        "the data is a time series. A wrong index here quietly measures the "
        "wrong object, so it is worth checking twice.",
    "MEASUREMENT FEATURES":
        "Which families of measurement are computed for every object — "
        "intensity, morphology, texture, radial distribution and "
        "colocalisation, with their parameters. More features means a wider "
        "table and a longer run, so enable what the analysis needs.",
    "OBJECT FILTERING":
        "Which objects are large enough, infected enough or clean enough to "
        "be measured at all. Raise the minimum sizes when debris is being "
        "counted; lower them when small but real objects disappear.",
    "CROP OUTPUT":
        "The per-object PNGs and arrays written alongside the measurements "
        "— crop mode and size, which channels and masks are included, "
        "dilation, and how they are normalised. These are the images "
        "Annotate and the CV classifier read later.",
    "PREVIEW & DIAGNOSTICS":
        "The small test run and the plots used to check a configuration "
        "before committing to a whole plate. The fastest way to find out "
        "that a channel index is wrong.",
    "3D CALIBRATION (BETA)":
        "The physical size of a voxel and the anisotropy between z and xy. "
        "Only these turn volumetric measurements from pixel counts into "
        "real units.",
    # -- Motility ----------------------------------------------------------
    "OBJECTS & CHANNELS":
        "The measurement source, which tracked object the assay is about, "
        "and which channels carry the cell, nucleus and pathogen signal. "
        "The rest of the assay is only as good as this mapping.",
    "SPATIAL & TEMPORAL CALIBRATION":
        "Pixel size and seconds per frame — the two numbers that convert "
        "movement in pixels into micrometres per second. Wrong here means "
        "every speed in the report is wrong by a constant factor.",
    "MOTION FILTERING":
        "The rules that keep implausible tracks out of the result — the "
        "largest jump allowed between frames, how straight a path has to "
        "be, and the outlier cutoff. Tighten them when tracking errors show "
        "up as impossibly fast cells.",
    "INFECTION CLASSIFICATION":
        "How a tracked cell is called infected, uninfected or ambiguous — "
        "which strategy is used, which table it reads, and where the "
        "probability cutoffs sit. The strategy chosen here decides which of "
        "the groups below actually apply.",
    "XGBOOST INFECTION MODEL":
        "Training and tree parameters for the supervised infection "
        "classifier, plus the probability threshold and margin that turn "
        "its output into a call. In play only when the strategy above is "
        "the XGBoost one.",
    "INFECTION CLUSTERING":
        "The unsupervised alternative: how many clusters, how the pathogen "
        "channel is weighted, and the minimum separation and silhouette a "
        "split has to reach before it is trusted. Use it when there are no "
        "labels to train on.",
    "EMBEDDING SEARCH":
        "The UMAP and t-SNE parameter ranges searched while trying to "
        "separate infected from uninfected phenotypes. Widen the grids when "
        "nothing separates the groups; fix single values to make a result "
        "reproducible.",
    "MOTILITY PLOTS & QC":
        "Axis limits and the diagnostic graphs used to review track quality "
        "and the infection call. Look here first when the summary numbers "
        "are surprising.",
    # -- Classify (CV) -----------------------------------------------------
    "PLATE SOURCES & WORKFLOW":
        "Which plates the classifier is built from, the experiment it is "
        "filed under, and which stages run — build the training set, train, "
        "test. Uncheck the stages you have already done to re-run only the "
        "part you are iterating on.",
    "LABELS & CLASSES":
        "Where the labels come from and what they mean — an annotation "
        "column or well metadata, the class names, and the measurement that "
        "defines them. Everything the model learns rests on this being the "
        "label you think it is.",
    "CROPS & DATASET SPLIT":
        "Which crops feed the model, from which tables and channels, at "
        "what size, and how they are divided into train and test — "
        "including whether classes are balanced down to the smallest one. "
        "Decide the split before training, not after.",
    "MODEL ARCHITECTURE":
        "The network being trained: backbone or custom weights, whether a "
        "checkpoint is resumed, input channels and image size, dropout and "
        "initialisation. Change the backbone when the model is too small "
        "for the phenotype, or too large for the data you have.",
    "OPTIMIZATION & LOSS":
        "How the network is fitted — optimiser, learning rate and decay, "
        "schedule, loss function and class balancing, epochs, batch size, "
        "augmentation and early stopping. This is where a diverging or "
        "underfitting run gets fixed.",
    "VALIDATION":
        "The held-out fraction, whether cross-validation runs and over how "
        "many folds, what the folds are grouped by, and the score cutoff "
        "used to call a class. Group folds by plate or well when the score "
        "has to survive contact with a new plate.",
    "EVALUATION WORKBENCH":
        "The deeper evaluation pass — nested cross-validation, probability "
        "calibration, and the leakage audit that checks the same object did "
        "not appear in both train and test. Run it before believing a "
        "headline accuracy.",
    "FULL DATASET & INFERENCE":
        "Applying a trained model to everything: which archive or dataset "
        "is scored, which model file is loaded, how much is sampled, and "
        "how many top examples are kept. Separate from training, so a "
        "finished model can be re-applied without refitting.",
    "MONITORING & RUNTIME":
        "What the run reports while it happens and how hard it works — "
        "plots, TensorBoard, intermediate saves, the random seed, workers "
        "and failure strictness. Fix the seed here when a result has to be "
        "reproducible.",
    # -- Classify (ML) -----------------------------------------------------
    "DATA & CONTROLS":
        "The measurement database this model is fitted on, the wells that "
        "define the positive and negative classes, and the column holding "
        "existing labels. Get these wrong and every number downstream is "
        "meaningless, so check them first.",
    "FEATURE PREPARATION":
        "Which measurement columns are allowed into the model, and the "
        "variance, correlation, object-count and compartment filters "
        "applied before fitting. Prune here when the feature table is wide, "
        "redundant, or contains a column that leaks the answer.",
    "PLATE & BATCH CORRECTION":
        "Whether per-plate offsets are removed before analysis, which "
        "column identifies the batch, and which wells anchor the "
        "correction. Use it when plates were run on different days or "
        "instruments and plate identity shows up as a larger effect than "
        "the biology.",
    "CLASSIFIER & VALIDATION":
        "The estimator itself and how honestly it is scored — algorithm, "
        "learning rate and regularisation, held-out fraction and "
        "cross-validation. Change these when the model overfits, or when "
        "the reported accuracy looks too good to be true.",
    "FEATURE SELECTION & IMPORTANCE":
        "Whether features are pruned before the final fit, and how repeated "
        "permutation importance is computed afterwards. This is the part "
        "that answers which measurements the decision is actually based on.",
    "OUTPUT & DATABASE":
        "Whether model scores are written back into the measurements "
        "database so later modules can read them. Leave it off for "
        "exploratory fits you would rather not record.",
    "PLOTS & HEATMAPS":
        "Which feature the heatmap shows, how wells are grouped, and the "
        "colour map and value range used to draw it. Presentation of the "
        "classifier's output; it does not change the fit.",
    # -- Regression --------------------------------------------------------
    "INPUT TABLES":
        "The metadata, score and count tables the regression runs on. All "
        "three have to agree on well and gRNA naming — disagreement there "
        "is the usual cause of an empty result.",
    "CONTROLS & PLATE DESIGN":
        "The plate identifier, which wells are the positive and negative "
        "controls, and any row filter applied before fitting. The controls "
        "set the scale the effect sizes are reported on.",
    "MODEL & COVARIATES":
        "The regression family, the response variable, how replicates are "
        "aggregated and transformed, regularisation, and the covariance "
        "structure. Switch families when the residuals are clearly not what "
        "the default assumes.",
    "HIT CALLING & OUTLIERS":
        "How much evidence a gRNA needs before it can be a hit — minimum "
        "cell and well counts, the control-derived threshold and its "
        "multiplier, and outlier rejection. Tighten these when the hit list "
        "fills up with low-count noise.",
    "REGRESSION PLOTS":
        "The volcano plot, and the axis transforms and ranges used to draw "
        "the regression output. Cosmetic: the fitted coefficients do not "
        "change.",
    "ADDITIONAL SETTINGS":
        "The remaining knobs belonging to individual regression families "
        "and plots — bootstrap counts, quantile and hinge parameters, "
        "solver tolerance and axis limits. Only the ones for the model you "
        "chose above have any effect.",
    # -- Activation --------------------------------------------------------
    "MODEL & DATA":
        "The trained model, the dataset it is applied to, and the input "
        "channels, object type and image size it expects. These have to "
        "match how the model was trained or the maps mean nothing.",
    "ATTRIBUTION METHOD":
        "Which algorithm explains the prediction — Grad-CAM, SmoothGrad, "
        "occlusion or integrated gradients — which layer it hooks, and the "
        "parameters of whichever you pick. Methods disagree; comparing two "
        "is often more informative than tuning one.",
    "ATTRIBUTION VALIDATION":
        "The checks that separate a real explanation from a pretty picture "
        "— insertion and deletion steps, the baseline they are measured "
        "against, and the model-weight sanity check. Worth running before "
        "an attribution map goes into a figure.",
    "MAP DISPLAY":
        "How the finished map is rendered — input and map normalisation, "
        "overlay on the source image, and whether it is plotted at all. "
        "Presentation only.",
    "MAP QUANTIFICATION":
        "Turning a map into numbers: channel correlation and the Manders "
        "thresholds used to ask how much of the attribution sits on a given "
        "structure.",
    "OUTPUT & RUNTIME":
        "Whether maps are saved, whether the input order is shuffled, and "
        "the batch size and worker count used to generate them.",
    # -- Replication -------------------------------------------------------
    "ASSAY INPUTS":
        "The measurements database, the parasite table inside it, and the "
        "compartment the parasites were measured in. The assay scores "
        "existing measurements — it does not segment anything itself.",
    "VACUOLE ASSIGNMENT":
        "How individual parasites are grouped into vacuoles — an existing "
        "vacuole identifier, or a spatial link whose distance scales with "
        "parasite size — and whether a host cell is required. The whole "
        "replication readout rests on this grouping.",
    "CONDITION METADATA":
        "Which wells hold which cell line, strain and treatment, and the "
        "column and level the conditions are grouped and reported at.",
    "REPLICATION SCORING":
        "How grouped parasites become a replication state: the largest "
        "vacuole accepted, the warning for biologically implausible counts, "
        "and whether wells with cells but no parasites are seeded as zeros. "
        "Leaving those wells out silently inflates the mean.",
    "ASSAY OUTPUT":
        "Whether the assay's results and figures are written, and the "
        "colour map used to draw them.",
    # -- External Masks ----------------------------------------------------
    "INPUT MAPPING":
        "How externally generated images and label masks are found and "
        "paired — the input list, the project folder written to, recursion, "
        "plate and well layout, z handling and naming. Preview the mapping "
        "before writing anything; this is where a mismatched pairing is "
        "caught.",
    # -- shared by the three Cellpose-facing modules -----------------------
    #
    # Mask, Cellpose Masks, Cellpose All and Train Cellpose ask the same four
    # questions about a segmentation run in the same order. Naming the groups
    # identically is the point: someone who learned them once should not have
    # to relearn them in the next module.
    "INPUT & CHANNELS":
        "Where the images come from and which planes of each one the module "
        "actually looks at, plus whether they are normalised or inverted "
        "first. A run that finds nothing at all is usually a channel index "
        "pointing at an empty plane.",
    "MODEL":
        "Which weights do the segmenting — a packaged model, or a checkpoint "
        "of your own — and the object size they should expect. Nothing is "
        "trained here; this is the picker, and the expected size matters "
        "more than the choice of weights.",
    "DETECTION THRESHOLDS":
        "How much the model is allowed to find: the probability floor below "
        "which a candidate is discarded, how strictly flow has to agree, and "
        "whether holes are filled. Come here when there are too many objects, "
        "too few, or one blob where two cells belong.",
    "IMAGE GEOMETRY":
        "The pixel dimensions the images are resampled to before anything "
        "else happens. Getting this wrong rescales every object and quietly "
        "changes what the expected size means, so set it once per "
        "acquisition and leave it.",
    "BACKGROUND & DENOISING":
        "Correction applied before segmentation: the intensity floor treated "
        "as empty and the signal-to-noise gate a field has to clear. Raise "
        "the floor when autofluorescence is being segmented as objects; "
        "lower it when genuinely dim cells disappear.",
    # -- Train Cellpose ----------------------------------------------------
    "STARTING POINT":
        "What the training run begins from — a pretrained model fine-tuned "
        "on your data, or randomly initialised weights — and the name the "
        "result is saved under. Fine-tuning needs far fewer labelled images "
        "than starting from scratch.",
    "TRAINING SCHEDULE":
        "How long the fit runs and how fast it moves: epochs, learning rate, "
        "weight decay, batch size and augmentation. Reach for these when the "
        "loss stops falling early, or when the model memorises the training "
        "images instead of generalising.",
    # -- Map Barcodes ------------------------------------------------------
    "SEQUENCING INPUT":
        "The read files and whether they are treated as a pair or a single "
        "direction. Everything downstream assumes this is right, and a "
        "single-end run pointed at paired reads finds nothing without "
        "reporting an error.",
    "BARCODE REFERENCES":
        "The three lookup CSVs a read is matched against — gRNA, row and "
        "column. A mapping run that returns no counts at all is almost "
        "always one of these three pointing at the wrong file, or at a file "
        "written with different column names.",
    "READ PARSING":
        "How a barcode is located inside each read: the anchoring sequence, "
        "the regular expression around it, and where the match is expected "
        "to begin and end. Change these when the library was built with a "
        "different adapter layout.",
    # -- Barcode QC --------------------------------------------------------
    "REFERENCE & COUNT TABLES":
        "The barcode references and the counts produced by a mapping run, "
        "which the checks below are computed from. Point them at the outputs "
        "of the run you want to judge, not at a newer plate.",
    "WELL EXPECTATIONS":
        "What a healthy well should look like — how many distinct guides it "
        "ought to carry, which statistic that is judged by, and the read "
        "floor below which a well is not worth trusting. These set the bar "
        "that everything else is measured against.",
    "STARVATION & EXCLUSION":
        "How wells that received too few reads are detected and whether they "
        "are dropped before the rest of the analysis. Leaving them in drags "
        "every plate-level summary toward noise, so exclude them once you "
        "trust the read floor above.",
    "POSITION & COLLISION CHECKS":
        "Two systematic artefacts worth ruling out before believing a hit: "
        "counts that track a well's position on the plate, and barcodes "
        "close enough in sequence to be confused for one another. Both look "
        "like biology until they are checked.",
    "THRESHOLD SWEEP":
        "The range and resolution of the scan used to show how the results "
        "would change under a different cut-off. Widen the span when the "
        "chosen threshold sits near the edge of the scanned range.",
    "QC OUTPUT":
        "Where the report is written and whether figures are drawn and kept. "
        "Leave saving off while you are still deciding which checks matter "
        "for this library.",
    # -- Illumination ------------------------------------------------------
    "CORRECTION MODEL":
        "How the uneven lighting field is estimated and removed — the family "
        "of surface fitted, the estimator behind it, its flexibility, and "
        "the dark reference subtracted first. Too flexible a surface absorbs "
        "real biological signal along with the shading.",
    "FIELD SAMPLING":
        "How many fields the correction is estimated from and whether each "
        "plate gets its own estimate. More fields make a steadier surface "
        "and a slower run; per-plate estimates matter when plates were "
        "acquired in separate sessions.",
    "QC & FAILURE HANDLING":
        "Whether the fitted surface is checked before being applied, and "
        "what happens when a plate has no usable estimate — skip it, or stop "
        "the run. Stopping is the safer choice the first time you correct an "
        "unfamiliar dataset.",
    # -- AnnData Export ----------------------------------------------------
    "OUTPUT FILE":
        "Where the exported object is written and how it is shaped: one "
        "matrix or one per table, the numeric precision kept, and the "
        "compression applied. Precision and compression trade file size "
        "against how faithfully the measurements survive the round trip.",
    "ROWS & MISSING VALUES":
        "How many rows are exported and what happens to gaps in them — kept "
        "as missing, dropped, or filled. Downstream tools differ sharply in "
        "what they tolerate, so this usually follows from whatever reads the "
        "file next.",
    "POST-PROCESSING":
        "Optional work done after the matrix is written: computing an "
        "embedding inside the exported object, and recording it as a run "
        "artifact so later steps can find it. Both are off by default "
        "because both cost time.",
    # -- Recruitment / Invasion -------------------------------------------
    "DATA SOURCE":
        "The measurements this module reads. One setting, and every group "
        "below assumes it is right — point it at the project folder a "
        "measure run wrote, not at the raw images.",
    "PLOTS & DIAGNOSTICS":
        "Whether preview figures are drawn, how large they are, and how many "
        "examples are produced. Worth turning on for the first plate of an "
        "experiment and off again once the numbers are trusted.",
    "CHANNELS & INTENSITY":
        "Which channels carry the signals the assay compares, the statistic "
        "each object is summarised by, and whether background is subtracted "
        "first. Swapping two channels here inverts the result without "
        "producing an error.",
    "THRESHOLDING":
        "How the cut-off separating the two populations is chosen, and how "
        "much disagreement between methods is tolerated before the run says "
        "so. This is the single most consequential group in the assay.",
    "CONTROLS & MINIMUM COUNTS":
        "Which wells anchor the threshold, and how many objects a well or a "
        "plate must contribute before its number is believed. Raise the "
        "minimums when sparse wells produce implausibly extreme rates.",
    # -- Regression --------------------------------------------------------
    "ESTIMATOR TUNING":
        "The knobs that belong to one estimator rather than to all of them — "
        "the elastic-net mixing ratio, the quantile being fitted, Huber's "
        "cut-off, the convergence tolerance, and the bootstrap counts behind "
        "the hinge and lasso selection thresholds. Only the ones matching "
        "the model chosen above have any effect.",
    # -- Power / Design ----------------------------------------------------
    #
    # "Power analysis" is the single heading `spacr/qt/screens/power.py`
    # registers all fifteen of its keys under, which is what the settings
    # diff and the run journal group them by when the module's own screen is
    # not involved. The five headings below are what the layout splits it
    # into; this entry covers the undivided one.
    "POWER ANALYSIS":
        "Everything a screening design has to commit to before a plate is "
        "poured: library size and redundancy, how it is spread over plates "
        "and replicates, the effect worth detecting and how rare it is, "
        "sequencing depth, and how the estimate itself is simulated.",
    "LIBRARY DESIGN":
        "The size and redundancy of the screening library: how many genes "
        "are targeted, how many guides each one gets, and how many "
        "constructs land in a well. Guides per gene is usually the cheapest "
        "lever on detection power.",
    "PLATE LAYOUT":
        "How the library is spread over physical plates — wells per plate, "
        "plate count, replicates, and cells sampled per well. This is where "
        "a design becomes a number of plates somebody has to actually run.",
    "EFFECT & PREVALENCE":
        "What the screen is looking for and how rare it is: the effect size "
        "worth detecting, the fraction of genes expected to show it, the "
        "background rate underneath, and how well the readout separates a "
        "hit from a miss. Optimism here is the usual reason a real screen "
        "underperforms its power curve.",
    "SEQUENCING DEPTH":
        "How many reads each well is allotted. Too few and guide counts "
        "become noise before any biology is involved, which no amount of "
        "extra replicates recovers.",
    "SIMULATION":
        "How the estimate itself is produced — the level the score is "
        "computed at, the backend that runs it, and the random seed. Fix "
        "the seed when you want two designs compared rather than two draws.",
}


#: Per-module overrides for headings that mean different things per module.
#: Missing entries fall through to :data:`CATEGORY_TOOLTIPS`.
CATEGORY_TOOLTIPS_BY_APP: Dict[str, Dict[str, str]] = {
    "train_cellpose": {
        # Train Cellpose fits weights; the other three run them. "Model"
        # therefore names the thing being produced rather than the thing
        # being picked, which is a different sentence.
        "OUTPUT & RUNTIME":
            "How much the training run prints as it goes. Turn it up when a "
            "fit is diverging and the loss curve alone does not say which "
            "epoch it went wrong at.",
    },
    "cellpose_masks": {
        "OUTPUT & RUNTIME":
            "Whether the masks are written, how many images are handed to "
            "the GPU at once, and how much the run prints. Reduce the batch "
            "size when the GPU runs out of memory.",
    },
    "cellpose_all": {
        "MODEL":
            "The object size every candidate model is told to expect. The "
            "point of this module is that the models differ, so this is the "
            "one thing held constant while they are compared.",
        "OUTPUT & RUNTIME":
            "Whether the comparison figures and masks are written, the GPU "
            "batch size, and how much each candidate run prints on its way "
            "through.",
    },
    "analyze_plaques": {
        "MODEL":
            "The expected plaque diameter, and whether previously written "
            "masks are reused instead of segmenting again. Plaques are far "
            "larger than cells, so the default cell-sized expectation is "
            "almost never right here.",
        "OUTPUT & RUNTIME":
            "Whether masks and results are written, the GPU batch size, and "
            "how much the run prints. Leave saving off for the first pass "
            "over a new plate.",
    },
    "umap": {
        "PATHS":
            "The measurements database the embedding is built from. One "
            "setting, and every other group depends on it.",
        "MEASUREMENTS":
            "Which tables and feature columns enter the embedding, and "
            "which are excluded or dropped for being redundant. The single "
            "most effective place to change what the map looks like.",
        "PLATE LAYOUT & CONTROLS":
            "Rules that drop whole rows out of the embedding by column "
            "value — a failed well, an untreated control, a plate you are "
            "not interested in today.",
        "PLOT":
            "How many rows are drawn and which column colours the points. "
            "Colouring by a metadata column is the quickest way to see "
            "whether a cluster is biology or batch.",
        "ADVANCED":
            "Where the crops are read from, worker count and verbosity. "
            "Rarely touched once a project is set up.",
    },
    "recruitment": {
        "MASK & CHANNEL MAPPING":
            "Which array plane holds each mask and each intensity channel, "
            "and which one the recruitment is measured on. A wrong index "
            "here measures the wrong compartment without complaining.",
        "OBJECT FILTERING":
            "The size and intensity windows an object has to fall inside to "
            "count, plus the per-well cell limits. These gates decide which "
            "cells the recruitment ratio is averaged over.",
        "PLATE LAYOUT & CONTROLS":
            "Which wells hold which cell line, strain and treatment, and "
            "which channel the recruitment is measured on. Filled in once "
            "per plate design.",
    },
    "invasion": {
        "ASSAY INPUTS":
            "Which measurement table the parasites are read from and which "
            "compartment they were measured in. The assay scores existing "
            "measurements rather than segmenting again.",
        "CONDITION METADATA":
            "Which wells hold which cell line, strain and treatment, and "
            "the column and level the invasion rates are grouped and "
            "reported at.",
        "ASSAY OUTPUT":
            "The colour map the assay's figures are drawn with, how many QC "
            "panels are produced, and whether wells with cells but no scored "
            "parasites are seeded as zeros. Leaving those wells out "
            "silently inflates the invasion rate.",
        "RUNTIME & RELIABILITY":
            "How much the assay prints as it runs. Turn it up while you are "
            "still deciding on a threshold and need to see which wells the "
            "controls were drawn from.",
    },
    "external_masks": {
        "GENERAL":
            "The experiment name, channel list, normalisation and whether a "
            "cytoplasm compartment is derived — the frame the imported "
            "masks are measured in. Check the channel list matches the "
            "images you are importing.",
        "TIMELAPSE":
            "Which objects are linked across frames when the imported data "
            "is a time series. Leave it alone for single-timepoint plates.",
        "MEASUREMENTS":
            "Which feature families are computed for the imported masks — "
            "intensity, texture, radial distribution and colocalisation. "
            "The expensive ones are off by default.",
        "ADVANCED":
            "Resume, failure tolerance, dry runs, worker count and "
            "verbosity for the import. Turn strict errors on the first time "
            "you import someone else's data.",
    },
    "timelapse": {
        "RUNTIME & RELIABILITY":
            "Which stages run, whether this is a small test pass, and how "
            "the run behaves under load and failure — workers, batch size, "
            "tolerated failure rate and verbosity. Track a few fields in "
            "test mode before committing to a whole plate.",
    },
    "motility": {
        "RUNTIME & RELIABILITY":
            "How many worker processes the assay uses. Lower it when the "
            "machine has other work to do.",
    },
    "ml_analyze": {
        "RUNTIME & RELIABILITY":
            "How many cores the fit is spread over, and how much it prints "
            "on the way. Lower the worker count when the machine has other "
            "work to do; raise the verbosity when a fit is failing and you "
            "cannot see where.",
    },
    "regression": {
        "RUNTIME & RELIABILITY":
            "Whether a failed plate stops the run, and how large a fraction "
            "of failures is tolerated before it does.",
    },
    "replication": {
        "OBJECT FILTERING":
            "The area window a segmented object has to fall inside to count "
            "as a parasite. Debris below it and clumps above it are "
            "excluded.",
        "RUNTIME & RELIABILITY":
            "How much the assay prints as it runs. Turn it up when a well "
            "comes out empty and you need to see which step discarded its "
            "parasites.",
    },
}


def category_tooltip(
    app_key: str,
    title: str,
    language: Optional[str] = None,
) -> str:
    """Return the plain-language blurb for one settings category.

    Resolution order: the module's own override, then the shared table, then
    a generic sentence built from the title. The generic one is a *visible*
    fallback rather than an empty string so a brand-new category is never
    silently blank — ``tests/qt/test_category_tooltips.py`` fails on it.

    :param app_key: module the category is being rendered for.
    :param title: category title as shown on the header (any case).
    :param language: optional language override; defaults to the UI language.
    """
    key = str(title or "").upper().strip()
    if not key:
        return ""
    text = CATEGORY_TOOLTIPS_BY_APP.get(str(app_key or ""), {}).get(key)
    if not text:
        text = CATEGORY_TOOLTIPS.get(key, "")
    if not text:
        text = f"Settings that control {str(title).lower().strip()}."
    return _translated_body(text, language)


def category_tooltip_is_curated(app_key: str, title: str) -> bool:
    """True when a category has a written blurb rather than the fallback."""
    key = str(title or "").upper().strip()
    return bool(
        CATEGORY_TOOLTIPS_BY_APP.get(str(app_key or ""), {}).get(key)
        or CATEGORY_TOOLTIPS.get(key)
    )


def get_tooltips() -> Dict[str, str]:
    """Return per-key tooltip text (spacr.settings.descriptions and .tooltips)."""
    tips: Dict[str, str] = {}
    try:
        from spacr.settings import descriptions, tooltips
    except Exception:
        return tips
    tips.update({k: v for k, v in descriptions.items() if isinstance(v, str)})
    tips.update({k: v for k, v in tooltips.items() if isinstance(v, str)})
    return tips


# ---------------------------------------------------------------------------
# API doc link per app
# ---------------------------------------------------------------------------

DOCS_API_BASE = "https://einarolafsson.github.io/spacr/api"

_APP_API_MODULE = {
    "align": "align",
    "convert": "convert",
    "foreign": "foreign",
    "queue": "qt/plate_queue",
    "batch": "batch",
    "db_browser": "qt/screens/db_browser",
    "mask": "core",
    "measure": "measure",
    "external_masks": "external_masks",
    "annotate": "qt/screens/annotate",
    "classify": "deep_spacr",
    "map_barcodes": "sequencing",
    "umap": "core",
    "timelapse": "core",
    "motility": "timelapse",
    "ml_analyze": "ml",
    "regression": "ml",
    "activation": "deep_spacr",
    "make_masks": "qt/screens/make_masks",
    "train_cellpose": "submodules",
    "cellpose_masks": "spacr_cellpose",
    "cellpose_all": "spacr_cellpose",
    "model_compare": "model_compare",
    "model_zoo": "model_zoo",
    "plate_view": "plate_qc",
    "agreement": "agreement",
    "train_compare": "train_compare",
    "classifier_evaluation": "classifier_evaluation",
    "run_history": "run_journal",
    "report": "report",
    "distributed_jobs": "remote_execution",
    "recruitment": "submodules",
    "analyze_plaques": "submodules",
    "invasion": "submodules",
    "replication": "submodules",
    "figure": "plot",
    "ai": "qt/ai",
}


def _absorb_registered_api_modules() -> None:
    """Take the API-doc module of every registered app into the table above.

    The PULL half of the app-registration seam;
    :func:`spacr.qt.app.register_app` PUSHES into this table when this
    module is already imported, and this picks up whatever registered
    before it was, so the order of the two imports stops mattering.
    Without it a module that registers itself sends its ⓘ link to the
    generated API index rather than to its own page.
    """
    app = sys.modules.get("spacr.qt.app")
    # `getattr(..., None)`: `spacr.qt.app` may be half-built when this
    # runs, in which case nothing has registered yet and the push half of
    # the seam delivers every row later.
    pull = getattr(app, "registered_metadata", None) if app else None
    if pull is None:
        return
    for key, module in pull("api_module").items():
        _APP_API_MODULE.setdefault(key, module)


_absorb_registered_api_modules()


def api_docs_url(app_key: str, key: str = "") -> str:
    """Return the spaCR API URL for an app or shared setting.

    Known app keys land on their module page. New or UI-only modules fall
    back to the generated API index rather than the documentation homepage.
    Shared batch-correction settings always land on their implementation,
    rather than whichever consumer app happens to display them.
    """
    try:
        from spacr.plugins import get_app
        plugin_app = get_app(app_key)
    except Exception:
        plugin_app = None
    if plugin_app is not None and plugin_app.docs_url:
        return plugin_app.docs_url
    evaluation_keys = {
        "classifier_evaluation",
        "nested_cv_inner_folds",
        "evaluation_calibration",
        "evaluation_bins",
        "evaluation_fail_on_leakage",
        "leakage_audit_train_test",
        "leakage_hash_content",
        "leakage_require_identity",
    }
    umap_search_keys = {
        "criterion", "search_mode", "adaptive", "n_trials", "n_folds",
        "random_seed", "resume_search", "n_neighbors_step",
        "min_dist_step", "min_improvement", "max_panels",
        "umap_stability_repeats", "umap_neighborhood_weight",
        "umap_stability_weight", "umap_cluster_structure_weight",
    }
    if key.startswith("batch_"):
        module = "batch_correction"
    elif key in evaluation_keys:
        module = "classifier_evaluation"
    elif app_key == "umap" and key in umap_search_keys:
        module = "hyperparam"
    else:
        module = _APP_API_MODULE.get(app_key)
    if module:
        return f"{DOCS_API_BASE}/spacr/{module}/index.html"
    return f"{DOCS_API_BASE}/index.html"


_TYPE_NAMES = {int: "integer", float: "float", bool: "boolean",
               str: "string", list: "list", tuple: "tuple",
               dict: "dictionary"}


def _type_hint(key: str) -> str:
    """Human-readable type of a setting, from spacr.settings.expected_types.

    e.g. ``'integer'``, ``'float'``, ``'boolean'``, ``'list'``, or
    ``'integer or float'`` / ``'string (optional)'`` for unions/None."""
    if not key:
        return ""
    try:
        from spacr.settings import expected_types
    except Exception:
        return ""
    t = expected_types.get(key)
    if t is None:
        return ""
    if isinstance(t, tuple):
        parts, optional = [], False
        for x in t:
            if x is type(None):
                optional = True
                continue
            parts.append(_TYPE_NAMES.get(x, getattr(x, "__name__", str(x))))
        s = " or ".join(dict.fromkeys(parts))   # dedupe, keep order
        if optional and s:
            s += " (optional)"
        return s
    return _TYPE_NAMES.get(t, getattr(t, "__name__", str(t)))


def _humanize(key: str) -> str:
    return key.replace("_", " ").strip().capitalize() if key else ""


def _strip_type_prefix(text: str) -> str:
    """Drop a leading ``(int) - `` / ``(bool) `` style prefix — the type is
    rendered separately + authoritatively from expected_types."""
    import re
    return re.sub(r"^\s*\([^)]*\)\s*[-–:]?\s*", "", text or "").strip()


def _language_code(language: Optional[str] = None) -> str:
    """Resolve ``language`` without making settings metadata depend on Qt."""
    from ..i18n import current_language, normalize_language

    return normalize_language(language or current_language())


def _translated_body(text: str, language: Optional[str] = None) -> str:
    """Translate setting prose only when a complete translation exists.

    The general UI translator deliberately supports conservative word-level
    translation for short labels.  Applying that behavior to a scientific
    paragraph produces a misleading half-English paragraph, however.  Tooltip
    bodies therefore accept exact catalog/plugin translations only and
    otherwise retain the canonical English source byte-for-byte.
    """
    source = " ".join(_strip_type_prefix(text).split())
    if not source:
        return ""
    code = _language_code(language)
    if code == "en":
        return source
    from ..i18n import _exact_translation, tr

    return (
        tr(source, code)
        if _exact_translation(source, code) is not None
        else source
    )


def _translated_type_hint(key: str, language: Optional[str] = None) -> str:
    """Return a localized type signature while preserving English defaults."""
    source = _type_hint(key)
    code = _language_code(language)
    if not source or code == "en":
        return source

    from ..i18n import tr

    optional = source.endswith(" (optional)")
    core = source[:-11] if optional else source
    # A slash is a language-neutral union separator.  Translating each atomic
    # type avoids asking the catalog to enumerate every possible union.
    translated = " / ".join(tr(part, code) for part in core.split(" or "))
    if optional:
        translated = f"{translated} ({tr('optional', code)})"
    return translated


def _translated_setting_name(key: str, language: Optional[str] = None) -> str:
    """Translate a short humanized setting label using the UI term catalog."""
    from ..i18n import tr

    return tr(_humanize(key), _language_code(language))


def _api_reference_tooltip(key: str, language: Optional[str] = None) -> str:
    """Localized accessible caption for a setting's teal API dot."""
    from ..i18n import tr

    code = _language_code(language)
    return tr(
        "Open API reference for {name}",
        code,
        name=_translated_setting_name(key, code),
    )


def _animation_reference_tooltip(
    key: str,
    language: Optional[str] = None,
) -> str:
    """Localized accessible caption for a setting's purple animation dot."""
    from ..i18n import tr

    code = _language_code(language)
    return tr(
        "Show animation for {name}",
        code,
        name=_translated_setting_name(key, code),
    )


def format_tooltip(
    text: str,
    app_key: str,
    key: str = "",
    language: Optional[str] = None,
) -> str:
    """Return localized typed HTML with an unchanged API-document URL."""
    from ..i18n import tr

    code = _language_code(language)
    body_source = _translated_body(text, code)
    body = escape(body_source)
    header = escape(_translated_setting_name(key, code))
    th = escape(_translated_type_hint(key, code))
    if header and th:
        header = f"<b>{header}</b> <i>({th})</i>"
    elif header:
        header = f"<b>{header}</b>"
    if not body:
        if code == "en" and key:
            body = f"Controls {escape(_humanize(key).lower())}."
        else:
            body = escape(tr("Controls this setting.", code))
    url = escape(api_docs_url(app_key, key), quote=True)
    link = (
        f'<a href="{url}">'
        f'{escape(tr("Open spaCR API documentation", code))}</a>'
    )
    parts = [p for p in (header, body, link) if p]
    return "<br>".join(parts)


def plain_tooltip(
    text: str,
    app_key: str,
    key: str = "",
    language: Optional[str] = None,
) -> str:
    """Same content as `format_tooltip` but plain text — used by the
    hover-follows footer at the bottom of each AppScreen."""
    from ..i18n import tr

    code = _language_code(language)
    body = _translated_body(text, code)
    if not body:
        body = (f"Controls {_humanize(key).lower()}."
                if code == "en" and key
                else tr("Controls this setting.", code))
    th = _translated_type_hint(key, code)
    name = _translated_setting_name(key, code)
    head = f"{name} ({th})" if (name and th) else name
    parts = [p for p in (head, body) if p]
    summary = " — ".join(parts)
    url = api_docs_url(app_key, key)
    api = tr("API: {url}", code, url=url)
    return f"{summary} — {api}" if summary else api


class _ApiTooltipFilter(QObject):
    """Show rich setting help in the clickable sticky tooltip."""

    def eventFilter(self, watched, event):  # noqa: N802 (Qt naming)
        # Re-render on entry so a Preferences language change cannot leave a
        # sticky popup displaying an earlier language.
        if event.type() == QEvent.Enter:
            refresh_api_tooltips(watched)
        html = watched.property("apiTooltipHtml")
        if not html:
            return False
        if event.type() == QEvent.Enter:
            from ..widgets.hover_tooltip import HoverTooltip
            HoverTooltip.instance().show_for(watched, str(html))
        elif event.type() == QEvent.Leave:
            from ..widgets.hover_tooltip import HoverTooltip
            HoverTooltip.instance().start_hide()
        elif event.type() == QEvent.ToolTip:
            # Suppress the native tooltip: it disappears when the pointer moves
            # toward its link, whereas HoverTooltip is intentionally clickable.
            return True
        return False


def attach_api_tooltip(
    widget: QWidget,
    app_key: str,
    key: str,
    description: str = "",
    _descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Attach typed, linked API help metadata to one setting widget."""
    descriptions = _descriptions if _descriptions is not None else get_tooltips()
    existing_tooltip = "" if widget.property("apiTooltipHtml") else widget.toolTip()
    body = (descriptions.get(key) or description
            or widget.property("apiTooltipDescriptionSource")
            or widget.property("apiTooltipDescription")
            or existing_tooltip)
    # Keep an absent body absent: format_tooltip owns the localized generic
    # fallback.  Synthesizing an English sentence here bypasses it.
    body = str(body or "")
    html = format_tooltip(body, app_key, key)
    widget.setProperty("settingsAppKey", app_key)
    widget.setProperty("settingKey", key)
    widget.setProperty("apiTooltipDescriptionSource", body)
    # Retain the old property as canonical English for integrations that read
    # it, rather than replacing it with rendered/localized HTML.
    widget.setProperty("apiTooltipDescription", body)
    widget.setProperty("apiTooltipHtml", html)
    if widget.property("apiTooltipDisplayRole") is None:
        widget.setProperty("apiTooltipDisplayRole", "tooltip")
    widget.setToolTip(html)
    widget.setToolTipDuration(-1)
    return html


def refresh_api_tooltips(
    root: QWidget,
    language: Optional[str] = None,
) -> None:
    """Refresh semantic setting help beneath ``root`` in ``language``.

    Canonical English prose is retained in ``apiTooltipDescriptionSource``;
    only the presentation HTML/plain accessibility chrome is regenerated.
    Field widgets marked ``metadata`` stay quiet because their visible label
    owns hover help.  API-dot URLs are never rebuilt or translated.
    """
    if root is None:
        return
    from ..i18n import tr

    code = _language_code(language)
    widgets = [root]
    try:
        widgets.extend(root.findChildren(QWidget))
    except (AttributeError, RuntimeError):
        return

    descriptions: Optional[Dict[str, str]] = None
    for widget in widgets:
        try:
            app_key = widget.property("settingsAppKey")
            key = widget.property("settingKey")
        except RuntimeError:
            continue
        if not app_key or not key:
            continue
        source = (widget.property("apiTooltipDescriptionSource")
                  or widget.property("apiTooltipDescription"))
        if not source:
            if descriptions is None:
                descriptions = get_tooltips()
            source = descriptions.get(str(key), "")
        source = str(source or "")
        html = format_tooltip(source, str(app_key), str(key), code)
        widget.setProperty("apiTooltipDescriptionSource", source)
        widget.setProperty("apiTooltipDescription", source)
        widget.setProperty("apiTooltipHtml", html)

        role = str(widget.property("apiTooltipDisplayRole") or "tooltip")
        if role == "metadata":
            widget.setToolTip("")
        elif role == "animation-link":
            caption = _animation_reference_tooltip(str(key), code)
            widget.setToolTip(caption)
            widget.setAccessibleName(caption)
            widget.setAccessibleDescription(caption)
        elif role == "api-link":
            caption = _api_reference_tooltip(str(key), code)
            widget.setToolTip(caption)
            widget.setAccessibleName(caption)
            widget.setAccessibleDescription(
                tr("Open spaCR API documentation", code))
        else:
            widget.setToolTip(html)
            widget.setToolTipDuration(-1)


def install_api_tooltips(
    owner: QWidget,
    app_key: str,
    widget_keys: Optional[Dict[QWidget, str]] = None,
) -> None:
    """Give every mapped/generated popup setting label consistent API help.

    ``SettingsWidgets`` controls are discovered through their ``settingKey``
    property. Hand-built Live/Crop/Search controls are supplied in
    ``widget_keys``. Descriptive help belongs to the label, not the editable
    field; a compact teal dot immediately beside that label opens the API page.
    """
    event_filter = getattr(owner, "_api_tooltip_filter", None)
    if event_filter is None:
        event_filter = _ApiTooltipFilter(owner)
        owner._api_tooltip_filter = event_filter

    mapped = dict(widget_keys or {})
    for widget in owner.findChildren(QWidget):
        if widget.property("settingHelpLabel"):
            continue
        # The dots this pass CREATES carry `settingKey` themselves, so they
        # are found by the sweep the next time it runs and decorated as though
        # they were settings — each one growing its own pair of dots. That is
        # what made the live-preview panel sprout duplicates every time the
        # form was re-gated (switching Primary object from cell to nucleus).
        # They are help, not settings; skip them.
        if widget.property("apiTooltipDisplayRole") in (
                "api-link", "animation-link"):
            continue
        key = widget.property("settingKey")
        if key and widget not in mapped:
            mapped[widget] = str(key)
    descriptions = get_tooltips()
    for widget, key in mapped.items():
        # Explicitly hidden controls are not settings in this popup. Decorating
        # one would create a visible wrapper/dot with a hidden field at (0, 0),
        # recreating the very kind of orphan overlay this helper should avoid.
        if widget.isHidden():
            continue
        html = attach_api_tooltip(
            widget, app_key, key, _descriptions=descriptions)
        label = _setting_label_for_field(owner, widget)
        if label is None:
            # A one-widget form row (usually a Toggle/QCheckBox) carries its
            # own visible label. Keep hover help on its text and put the same
            # teal API dot immediately after the combined label/control.
            # Remove before installing. Qt keeps a LIST of filters and calls
            # each installation separately, so decorating the same widget
            # twice makes one hover emit two tooltips and two animation
            # popups. `removeEventFilter` is a no-op when the filter is not
            # installed, which makes this idempotent for free.
            widget.removeEventFilter(event_filter)
            widget.installEventFilter(event_filter)
            _add_api_dot_to_combined_control(
                owner, widget, app_key, key, html)
            continue

        body_source = str(widget.property("apiTooltipDescriptionSource") or "")
        label.setCursor(Qt.WhatsThisCursor)
        label.setProperty("settingHelpLabel", True)
        label.setProperty("settingsAppKey", app_key)
        label.setProperty("settingKey", key)
        label.setProperty("apiTooltipDescriptionSource", body_source)
        label.setProperty("apiTooltipDescription", body_source)
        label.setProperty("apiTooltipHtml", html)
        label.setProperty("apiTooltipDisplayRole", "tooltip")
        label.setToolTip(html)
        label.setToolTipDuration(-1)
        # Idempotent, for the reason above: this decoration pass runs again
        # whenever the live-preview form is re-gated -- changing the primary
        # object from cell to nucleus, for instance -- and a second
        # installation on the same label duplicated every tooltip and every
        # setting animation on the panel. The API dots did not duplicate
        # because `_add_api_dot_to_label` guards on a property; the filter had
        # no such guard.
        label.removeEventFilter(event_filter)
        label.installEventFilter(event_filter)

        # The editor itself remains quiet on hover. Keep its metadata so tests,
        # integrations and a later re-parenting pass can still identify it.
        widget.setProperty("apiTooltipDisplayRole", "metadata")
        widget.setToolTip("")
        widget.removeEventFilter(event_filter)
        _add_api_dot_to_label(label, app_key, key, html)


def _unwrap_setting_label(candidate: Optional[QWidget]) -> Optional[QWidget]:
    """Return the real label inside a `SettingLabelWithInfo` host.

    The first decoration pass replaces the form's label with a host widget
    holding ``[stretch][label][dots]``. On a SECOND pass
    ``QFormLayout.labelForField`` therefore hands back the HOST, not the
    label — a fresh widget with none of the label's guard properties — so the
    pass decorated it again and the panel grew a second dot, a second
    animation dot and a second tooltip per setting. That is what switching
    Primary object from cell to nucleus did in the Mask live preview.

    Unwrapping restores the invariant the guards rely on: the same label
    object is found every time.
    """
    if candidate is None:
        return None
    if candidate.objectName() != "SettingLabelWithInfo":
        return candidate
    for child in candidate.findChildren(QWidget):
        if child.property("settingHelpLabel"):
            return child
    return candidate


def _setting_label_for_field(owner: QWidget, field: QWidget) -> Optional[QWidget]:
    """Find the visual label immediately to the left of a popup field."""
    remembered = getattr(field, "_spacr_setting_label", None)
    if isinstance(remembered, QWidget):
        try:
            remembered.objectName()
            if remembered.window() is owner.window():
                return _unwrap_setting_label(remembered)
        except RuntimeError:
            pass

    for form in owner.findChildren(QFormLayout):
        # A form field is often a wrapper QWidget containing an editor and a
        # Browse button (or two numeric editors). QFormLayout only knows the
        # wrapper, so walk the editor's parent chain before concluding that it
        # is a label-less combined control. Otherwise its tooltip and API dot
        # end up beside the editor instead of on the form label.
        candidate: Optional[QWidget] = field
        while isinstance(candidate, QWidget):
            label = _unwrap_setting_label(form.labelForField(candidate))
            if isinstance(label, QWidget):
                field._spacr_setting_label = label
                return label
            if candidate is owner:
                break
            candidate = candidate.parentWidget()

    # Hand-built search panels use compact grids rather than QFormLayout.
    # Select the nearest widget to the field's left on the same row.
    for grid in owner.findChildren(QGridLayout):
        index = grid.indexOf(field)
        if index < 0:
            continue
        row, column, _row_span, _column_span = grid.getItemPosition(index)
        for candidate_column in range(column - 1, -1, -1):
            item = grid.itemAtPosition(row, candidate_column)
            candidate = item.widget() if item is not None else None
            if isinstance(candidate, QLabel):
                field._spacr_setting_label = candidate
                return candidate
    return None


def build_setting_link_widget(
    app_key: str,
    key: str,
    html: str,
    body_source: str = "",
    parent: Optional[QWidget] = None,
) -> Tuple[QWidget, QWidget, Optional[QWidget]]:
    """Build API help and, when available, stacked animation help links.

    The purple animation dot sits above the teal API dot. Their combined
    28-pixel stack is vertically centred on the setting label, so the midpoint
    between both marks aligns with the label text line.

    :returns: ``(layout_widget, api_dot, animation_dot_or_none)``.
    """
    from ..widgets.animation_link import AnimationLink, SettingLinkStack
    from ..widgets.info_link import InfoLink

    api_dot = InfoLink(
        api_docs_url(app_key, key),
        tooltip=_api_reference_tooltip(key),
    )
    api_dot.setObjectName("SettingInfoLink")
    api_dot.setProperty("settingsAppKey", app_key)
    api_dot.setProperty("settingKey", key)
    api_dot.setProperty("apiTooltipDescriptionSource", body_source)
    api_dot.setProperty("apiTooltipDescription", body_source)
    api_dot.setProperty("apiTooltipHtml", html)
    api_dot.setProperty("apiTooltipDisplayRole", "api-link")

    try:
        from spacr.setting_animations import (
            SettingAnimationError,
            animation_for_setting,
        )
        animation = animation_for_setting(key)
    except SettingAnimationError:
        LOGGER.exception(
            "Setting animation registry is invalid; %s.%s keeps API help only",
            app_key,
            key,
        )
        animation = None

    if animation is None:
        api_dot.setParent(parent)
        return api_dot, api_dot, None

    animation_dot = AnimationLink(
        animation,
        tooltip=_animation_reference_tooltip(key),
    )
    animation_dot.setProperty("settingsAppKey", app_key)
    animation_dot.setProperty("settingKey", key)
    animation_dot.setProperty("apiTooltipDescriptionSource", body_source)
    animation_dot.setProperty("apiTooltipDescription", body_source)
    animation_dot.setProperty("apiTooltipHtml", html)
    animation_dot.setProperty("apiTooltipDisplayRole", "animation-link")
    stack = SettingLinkStack(animation_dot, api_dot, parent=parent)
    return stack, api_dot, animation_dot


def _add_api_dot_to_label(
    label: QWidget,
    app_key: str,
    key: str,
    html: str,
) -> None:
    """Place one clickable teal API dot immediately to a setting label's right."""
    if bool(label.property("settingApiDotInstalled")):
        return
    parent = label.parentWidget()
    layout = parent.layout() if parent is not None else None
    if layout is None:
        return

    host = QWidget(parent)
    host.setObjectName("SettingLabelWithInfo")
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addStretch(1)
    replaced = layout.replaceWidget(label, host)
    if replaced is None:
        host.deleteLater()
        return
    label.setParent(host)
    row.addWidget(label)
    body_source = str(label.property("apiTooltipDescriptionSource") or "")
    links, dot, animation_dot = build_setting_link_widget(
        app_key, key, html, body_source, parent=host,
    )
    row.addWidget(links, 0, Qt.AlignVCenter)
    label.setProperty("settingApiDotInstalled", True)
    label._spacr_api_dot = dot
    if animation_dot is not None:
        label._spacr_animation_dot = animation_dot


def _add_api_dot_to_combined_control(
    owner: QWidget,
    field: QWidget,
    app_key: str,
    key: str,
    html: str,
) -> None:
    """Add an API dot after a Toggle/QCheckBox that is its own row label."""
    existing = getattr(field, "_spacr_api_dot", None)
    if isinstance(existing, QWidget):
        try:
            if existing.window() is owner.window():
                return
        except RuntimeError:
            pass
    parent = field.parentWidget()
    layout = parent.layout() if parent is not None else None
    if layout is None:
        return

    host = QWidget(parent)
    host.setObjectName("SettingControlWithInfo")
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    replaced = layout.replaceWidget(field, host)
    if replaced is None:
        host.deleteLater()
        return
    field.setParent(host)
    row.addWidget(field)
    body_source = str(field.property("apiTooltipDescriptionSource") or "")
    links, dot, animation_dot = build_setting_link_widget(
        app_key, key, html, body_source, parent=host,
    )
    row.addWidget(links, 0, Qt.AlignVCenter)
    row.addStretch(1)
    field._spacr_api_dot = dot
    if animation_dot is not None:
        field._spacr_animation_dot = animation_dot


# ---------------------------------------------------------------------------
# Widget factory
# ---------------------------------------------------------------------------

class _ListEdit(QLineEdit):
    """A QLineEdit that round-trips a Python list via repr()."""
    def get_value(self) -> Any:
        """Return the field parsed as a Python literal (or raw text on failure)."""
        text = self.text().strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    def set_value(self, v: Any) -> None:
        """Render ``v`` into the field via ``repr``; ``None`` clears the field."""
        self.setText(repr(v) if v is not None else "")


class _ScalarEdit(QLineEdit):
    """A plain QLineEdit that returns None for empty text."""
    def get_value(self) -> Optional[str]:
        """Return the current text, or ``None`` when the field is empty."""
        return self.text() or None

    def set_value(self, v: Any) -> None:
        """Set the field text; ``None`` clears the field."""
        self.setText("" if v is None else str(v))


# ---------------------------------------------------------------------------
# List / list-of-list editor
# ---------------------------------------------------------------------------
#
# A list setting used to be a text box holding a Python literal:
#
#     class_metadata   [['c1'], ['c2']]
#     train_channels   ['r', 'g', 'b']
#
# which is both ugly and unforgiving -- a dropped bracket is a parse
# failure with no diagnosis, and `_ListEdit.get_value` silently handed the
# unparseable text through as a plain string. Worse, `_ListEdit` was never
# reached: `gui_utils.convert_settings_dict_for_gui` stringifies every list
# default before this module sees it (`('entry', None, str(value))`), so
# `isinstance(default, list)` in `_widget_for` was always False and every
# list setting got a `_ScalarEdit`. `collect()` then returned the raw text,
# because `_coerce_to_expected_type` only ever handled bool/int/float. That
# is how `class_metadata` reached `io.generate_training_dataset` as the
# *string* "[['c1'], ['c2']]" and got iterated character by character.
#
# The widgets below replace the literal with removable chips -- one chip per
# value, one row per inner list -- and hand `collect()` a real Python list.
# The stored value is unchanged, so every settings CSV on disk still loads
# and every consumer reads what it always did.

#: Keys whose value may be a list of lists even when it is currently flat.
#: Taken from the same list ``spacr.settings.check_settings`` parses with
#: ``ast.literal_eval`` for the Tk GUI, so the two front ends agree on which
#: fields can hold groups.
NESTED_CAPABLE_KEYS = frozenset({
    "cell_plate_metadata", "class_metadata", "crop_mode", "dialate_png_ratios",
    "pathogen_plate_metadata", "png_dims", "png_size", "timelapse_frame_limits",
    "timelapse_objects", "treatment_plate_metadata",
    # declared ``(list, list)`` in expected_types, the in-tree marker for
    # "this can be a list of lists"
    "cell_loc", "pathogen_loc", "treatment_loc", "barcode_coordinates",
})

# Channel selections that contain more than one channel use the same
# add/remove-chip editor as ``manders_thresholds``.  The legacy GUI converter
# still labels the first three as curated combos, so keep this declaration
# close to the list editor and let the real per-module default decide whether
# the setting is actually a list.  Scalar selectors such as ``cell_channel``
# and ``channel_of_interest`` are intentionally absent.
CHANNEL_LIST_KEYS = frozenset({
    "channels", "channel_dims", "train_channels", "normalize_channels",
    "overlay_chans", "png_dims",
})


class _FlowLayout(QLayout):
    """A left-to-right layout that wraps onto a new line when it runs out.

    Chips have to wrap: ``controls`` ships thirty of them and a horizontal
    box would either clip them or force the settings panel wider than the
    window.
    """

    def __init__(self, parent=None, spacing: int = 4):
        super().__init__(parent)
        self._items: List[Any] = []
        self._space = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:            # noqa: N802 (Qt override)
        """Append a layout item (Qt calls this for every added widget)."""
        self._items.append(item)

    def count(self) -> int:
        """Number of items in the layout."""
        return len(self._items)

    def itemAt(self, index):                    # noqa: N802 (Qt override)
        """Return the item at ``index``, or None when out of range."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):                    # noqa: N802 (Qt override)
        """Remove and return the item at ``index``, or None."""
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):              # noqa: N802 (Qt override)
        """Never ask for extra space in either direction."""
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:        # noqa: N802 (Qt override)
        """Height depends on width -- that is the whole point of wrapping."""
        return True

    def heightForWidth(self, width: int) -> int:    # noqa: N802 (Qt override)
        """Height needed to lay the chips out inside ``width``."""
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect) -> None:        # noqa: N802 (Qt override)
        """Place every chip inside ``rect``."""
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:                # noqa: N802 (Qt override)
        """Preferred size -- the minimum, since the height is width-driven."""
        return self.minimumSize()

    def minimumSize(self) -> QSize:             # noqa: N802 (Qt override)
        """The largest single chip, plus margins."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    def _do_layout(self, rect, test_only: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(),
                             -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._space
            if next_x - self._space > area.right() and line_height > 0:
                x = area.x()
                y = y + line_height + self._space
                next_x = x + hint.width() + self._space
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


class _FlowHost(QWidget):
    """The widget a :class:`_FlowLayout` lives in.

    Qt only consults a layout's ``heightForWidth`` through the widget that
    owns it, and only when that widget's size policy says its height depends
    on its width. Without this the strip reported a one-line height however
    many chips it held, and ``controls`` (thirty of them) drew off the edge
    of the settings column instead of wrapping.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self) -> bool:      # noqa: N802 (Qt override)
        """Yes -- more width means fewer rows of chips."""
        return True

    def heightForWidth(self, width: int) -> int:   # noqa: N802 (Qt override)
        """Height the chips need once wrapped into ``width``."""
        layout = self.layout()
        if layout is None:
            return super().heightForWidth(width)
        return layout.heightForWidth(width)

    def sizeHint(self) -> QSize:              # noqa: N802 (Qt override)
        """Preferred size at the current width, so the row grows as chips
        are added rather than clipping them."""
        layout = self.layout()
        if layout is None:
            return super().sizeHint()
        width = max(self.width(), layout.minimumSize().width())
        return QSize(width, layout.heightForWidth(width))


class _Chip(QFrame):
    """One value, rendered as a removable pill."""

    removed = Signal(object)

    def __init__(self, text: str, colours: dict, parent=None):
        super().__init__(parent)
        from ..theme import font_px
        self.setObjectName("SettingChip")
        self._text = text
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 1, 3, 1)
        row.setSpacing(4)
        label = QLabel(text, self)
        label.setObjectName("SettingChipText")
        row.addWidget(label)
        close = QToolButton(self)
        close.setObjectName("SettingChipClose")
        close.setText("×")
        close.setCursor(Qt.PointingHandCursor)
        close.setToolTip(f"Remove {text}")
        close.setFocusPolicy(Qt.NoFocus)
        close.clicked.connect(lambda: self.removed.emit(self))
        row.addWidget(close)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"""
            QFrame#SettingChip {{
                background: {colours['accent_soft']};
                border: 1px solid {colours['border']};
                border-radius: 9px;
            }}
            QLabel#SettingChipText {{
                color: {colours['fg']};
                background: transparent;
                font-size: {font_px(12)}px;
            }}
            QToolButton#SettingChipClose {{
                color: {colours['fg_muted']};
                background: transparent;
                border: none;
                padding: 0px 2px;
                font-size: {font_px(13)}px;
            }}
            QToolButton#SettingChipClose:hover {{ color: {colours['error']}; }}
            """
        )

    def text(self) -> str:
        """The value this chip carries, as typed."""
        return self._text


class _ChipStrip(QWidget):
    """A wrapping strip of chips plus the field that adds another one."""

    changed = Signal()
    emptied = Signal(object)

    def __init__(self, placeholder: str = "add value…",
                 removable: bool = False, parent=None):
        super().__init__(parent)
        from ..theme import active_palette, font_px
        self._colours = active_palette()
        self._chips: List[_Chip] = []

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._host = _FlowHost(self)
        self._flow = _FlowLayout(self._host, spacing=4)
        outer.addWidget(self._host, 1)

        self._entry = QLineEdit(self)
        self._entry.setObjectName("SettingChipEntry")
        self._entry.setPlaceholderText(placeholder)
        self._entry.setMinimumWidth(96)
        self._entry.returnPressed.connect(self._commit_entry)
        self._entry.editingFinished.connect(self._commit_entry)
        self._entry.textEdited.connect(self._on_typed)
        self._flow.addWidget(self._entry)

        self._drop = None
        if removable:
            self._drop = QToolButton(self)
            self._drop.setText("✕")
            self._drop.setCursor(Qt.PointingHandCursor)
            self._drop.setToolTip("Remove this group")
            self._drop.setFocusPolicy(Qt.NoFocus)
            self._drop.clicked.connect(lambda: self.emptied.emit(self))
            outer.addWidget(self._drop, 0, Qt.AlignTop)

    # -- value -----------------------------------------------------------
    def values(self) -> List[str]:
        """The chip texts, in order, plus anything still uncommitted."""
        out = [chip.text() for chip in self._chips]
        pending = self._entry.text().strip()
        if pending:
            out.append(pending)
        return out

    def set_values(self, values) -> None:
        """Replace every chip with ``values``."""
        for chip in list(self._chips):
            self._remove_chip(chip, notify=False)
        self._entry.clear()
        for value in values or []:
            self._add_chip(str(value), notify=False)
        self.changed.emit()

    # -- internals -------------------------------------------------------
    def _on_typed(self, text: str) -> None:
        """Commit on a comma so a pasted 'c1,c2,c3' becomes three chips."""
        if "," not in text:
            return
        head, _, tail = text.partition(",")
        self._entry.setText(tail.lstrip())
        head = head.strip()
        if head:
            self._add_chip(head)

    def _commit_entry(self) -> None:
        text = self._entry.text().strip()
        if not text:
            return
        self._entry.clear()
        self._add_chip(text)

    def _add_chip(self, text: str, notify: bool = True) -> None:
        chip = _Chip(text, self._colours, self._host)
        chip.removed.connect(self._remove_chip)
        # Keep the entry field last so it always trails the chips.
        self._flow.removeWidget(self._entry)
        self._flow.addWidget(chip)
        self._flow.addWidget(self._entry)
        self._chips.append(chip)
        self._host.updateGeometry()
        self.updateGeometry()
        if notify:
            self.changed.emit()

    def _remove_chip(self, chip, notify: bool = True) -> None:
        if chip in self._chips:
            self._chips.remove(chip)
        self._flow.removeWidget(chip)
        chip.setParent(None)
        chip.deleteLater()
        self._host.updateGeometry()
        self.updateGeometry()
        if notify:
            self.changed.emit()


#: Settings whose legal values are a short, closed, ordered set.
#:
#: ``train_channels`` is the reason this table exists. It is declared a plain
#: ``list``, so it rendered as a free-text chip strip that accepted ``x``,
#: ``red``, ``4`` and ``rgb`` without complaint — and
#: :func:`spacr.io._resolve_channel_indices` maps letters to planes with
#: three ``if 'r' in channels`` tests, so an off-alphabet value is dropped
#: silently and the model trains on fewer planes than the user asked for.
#: :func:`spacr.deep_spacr.train_test_model` then joins the same list into a
#: directory name, so the typo reaches the filesystem too.
#:
#: Order is part of the alphabet, not part of the user's input: ``['b','r']``
#: and ``['r','b']`` select the same two planes but write two different model
#: directories. A control that can only emit canonical order removes that
#: whole class of confusion, which a text field cannot.
FIXED_ALPHABETS: Dict[str, Tuple[Tuple[Any, str], ...]] = {
    "train_channels": (("r", "Red"), ("g", "Green"), ("b", "Blue")),
}


def _alphabet_qss(palette: dict, opacity) -> str:
    """QSS for the fixed-alphabet toggles, registered through the theme seam.

    Selected and unselected have to differ at a glance without colour alone
    carrying the meaning — the text is the value either way, and the border
    does the work, so the control still reads on a monochrome display and for
    a red-green colour-blind reader choosing red and green channels.
    """
    from ..theme import pane_surface
    surface = pane_surface("surface_alt", palette["theme"], opacity)
    return f"""
QToolButton#SettingAlphabetChip {{
    background: {surface};
    color: {palette["fg_dim"]};
    border: 1px solid {palette["border_soft"]};
    border-radius: 10px;
    padding: 2px 12px;
}}
QToolButton#SettingAlphabetChip:hover {{
    border-color: {palette["accent"]};
}}
QToolButton#SettingAlphabetChip:checked {{
    color: {palette["fg"]};
    border: 1px solid {palette["accent"]};
    font-weight: 600;
}}
"""


try:  # pragma: no cover - the theme seam is present in every real launch
    from ..theme import register_widget_qss as _register_widget_qss
    _register_widget_qss("SettingAlphabetChip", _alphabet_qss, replace=True)
except Exception:  # pragma: no cover
    LOGGER.debug("Could not register the alphabet-chip QSS", exc_info=True)


class _AlphabetSelect(QWidget):
    """Multi-select over a fixed, ordered alphabet of values.

    One checkable pill per legal value, always shown, always in the
    alphabet's own order. Nothing else can be entered and nothing can be
    entered twice, so the two failure modes of the free-text strip it
    replaces — an unrecognised letter that is silently dropped downstream,
    and a permutation that changes the output path without changing the
    result — are both unrepresentable.

    ``get_value`` / ``set_value`` mirror :class:`_ListEditor`'s contract so
    the settings-CSV import path, the Live Preview propagation path and
    :meth:`SettingsWidgets.collect` need no special case beyond the class.
    """

    changed = Signal()

    def __init__(self, key: str = "", default: Any = None,
                 choices: Tuple[Tuple[Any, str], ...] = (), parent=None):
        super().__init__(parent)
        self._key = key
        self._choices = tuple(choices)
        self._buttons: List[Tuple[Any, QToolButton]] = []

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for value, label in self._choices:
            button = QToolButton(self)
            button.setObjectName("SettingAlphabetChip")
            button.setText(str(label))
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFocusPolicy(Qt.StrongFocus)
            # The accessible name is the value, not the label: a screen
            # reader user is choosing 'r', and "Red" is only the gloss.
            button.setAccessibleName(str(value))
            button.setProperty("alphabetValue", value)
            button.toggled.connect(self._on_toggled)
            row.addWidget(button)
            self._buttons.append((value, button))
        row.addStretch(1)

        self.set_value(default)

    # -- public contract -------------------------------------------------
    def get_value(self) -> List[Any]:
        """The checked values, always in alphabet order."""
        return [value for value, button in self._buttons if button.isChecked()]

    def set_value(self, value: Any) -> None:
        """Check exactly the members of ``value``; ignore anything else.

        Strings are parsed as Python literals first, because settings CSVs
        and the Live Preview both hand back ``"['r', 'g']"`` rather than a
        list. A value outside the alphabet is dropped rather than shown,
        which is the whole point of the control — but it is dropped
        *visibly*, because the pill for it simply is not lit.
        """
        wanted = self._as_members(value)
        for member, button in self._buttons:
            blocked = button.blockSignals(True)
            button.setChecked(member in wanted)
            button.blockSignals(blocked)
        self.changed.emit()

    def text(self) -> str:
        """Line-edit-compatible rendering, for callers that expect one."""
        return repr(self.get_value())

    def setText(self, value: str) -> None:  # noqa: N802 - QLineEdit contract
        """Accept a textual value, for callers that expect a QLineEdit."""
        self.set_value(value)

    def choices(self) -> Tuple[Any, ...]:
        """The legal values, in order. Public so tests need no internals."""
        return tuple(value for value, _label in self._choices)

    # -- internals -------------------------------------------------------
    def _on_toggled(self, _checked: bool) -> None:
        self.changed.emit()

    @staticmethod
    def _as_members(value: Any) -> set:
        if value is None:
            return set()
        if isinstance(value, str):
            text = value.strip()
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                # A bare "r,g" or "r g" from a hand-edited CSV.
                parsed = [part for part in text.replace(",", " ").split()
                          if part]
            value = parsed
        if isinstance(value, (list, tuple, set, frozenset)):
            return set(value)
        return {value}


class _ListEditor(QWidget):
    """The widget behind every list-valued setting.

    Flat lists are one strip of chips. Lists of lists are one strip per
    inner list, stacked, each with its own remove button and a footer that
    adds another group. A key that *can* hold groups but currently does not
    gets a "Use groups" button instead, so nothing that was editable as a
    literal becomes uneditable here.

    ``get_value`` / ``set_value`` mirror ``_ListEdit``'s contract, so the
    Live Preview propagation path and the settings-CSV import path need no
    special case beyond knowing the class.
    """

    def __init__(self, key: str = "", default: Any = None,
                 nested_capable: bool = False, allow_none: bool = False,
                 element_type: Any = None, container: Any = list, parent=None):
        super().__init__(parent)
        # font_px is used further down this method. Importing only
        # active_palette here raised NameError out of build_sections(), and
        # AppScreen turns that into "Failed to build settings for '<app>'" --
        # so sixteen shipped modules, mask and measure and classify among
        # them, opened with no settings form at all.
        from ..theme import active_palette, font_px
        self._colours = active_palette()
        self._key = key
        self._nested_capable = bool(nested_capable)
        self._allow_none = bool(allow_none)
        self._element_type = element_type
        self._container = container if container in (list, tuple) else list
        self._nested = False
        self._strips: List[_ChipStrip] = []

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(4)

        self._rows = QVBoxLayout()
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(4)
        self._outer.addLayout(self._rows)

        self._footer = QToolButton(self)
        self._footer.setObjectName("SettingListFooter")
        self._footer.setCursor(Qt.PointingHandCursor)
        self._footer.setFocusPolicy(Qt.NoFocus)
        self._footer.clicked.connect(self._on_footer)
        self._footer.setStyleSheet(
            f"QToolButton#SettingListFooter {{ color: {self._colours['accent']};"
            f" background: transparent; border: none; font-size: {font_px(12)}px;"
            f" padding: 0px; text-align: left; }}"
        )
        self._outer.addWidget(self._footer, 0, Qt.AlignLeft)

        self.set_value(default)

    # -- public contract -------------------------------------------------
    def get_value(self) -> Any:
        """Return a real ``list`` (or list of lists); ``None`` when empty
        and the setting declares ``None`` as legal."""
        make = self._container
        if self._nested:
            groups = [make(self._cast(v) for v in strip.values())
                      for strip in self._strips]
            groups = [g for g in groups if g]
            if not groups:
                return None if self._allow_none else make()
            return make(groups)
        values = [self._cast(v) for v in self._strips[0].values()] \
            if self._strips else []
        if not values:
            return None if self._allow_none else make()
        return make(values)

    def set_value(self, value: Any) -> None:
        """Render ``value``; strings are parsed as Python literals first.

        Settings CSVs and the Live Preview both hand back text, so a
        ``"[['c1'], ['c2']]"`` has to land as two groups rather than as
        seventeen chips full of punctuation.
        """
        value = self._as_sequence(value)
        nested = bool(value) and all(
            isinstance(item, (list, tuple)) for item in value)
        self._rebuild(nested, value)

    def text(self) -> str:
        """Return a line-edit-compatible textual representation.

        A single path is returned without list punctuation for compatibility
        with callers that treated ``src`` as a ``QLineEdit`` before it became
        a multi-plate setting. Multiple values use their unambiguous Python
        representation.
        """
        value = self.get_value()
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return str(value[0])
        return "" if value is None else str(value)

    def setText(self, value: str) -> None:  # noqa: N802 - QLineEdit contract
        """Accept the legacy ``QLineEdit.setText`` API."""
        self.set_value(value)

    # -- shape -----------------------------------------------------------
    def _rebuild(self, nested: bool, value) -> None:
        for strip in list(self._strips):
            # editingFinished fires while a focused QLineEdit is being torn
            # down, which would call _commit_entry on a half-deleted strip.
            strip._entry.blockSignals(True)
            self._rows.removeWidget(strip)
            strip.setParent(None)
            strip.deleteLater()
        self._strips = []
        self._nested = bool(nested)
        if nested:
            for group in value:
                self._add_strip(list(group))
            if not self._strips:
                self._add_strip([])
        else:
            self._add_strip(list(value))
        self._refresh_footer()

    def _add_strip(self, values) -> _ChipStrip:
        strip = _ChipStrip(placeholder=self._placeholder(),
                           removable=self._nested, parent=self)
        strip.emptied.connect(self._drop_strip)
        self._rows.addWidget(strip)
        self._strips.append(strip)
        strip.set_values(values)
        return strip

    def _drop_strip(self, strip) -> None:
        if len(self._strips) <= 1:
            # Removing the only group is how you go back to a flat list.
            self._rebuild(False, [])
            return
        self._strips.remove(strip)
        strip._entry.blockSignals(True)
        self._rows.removeWidget(strip)
        strip.setParent(None)
        strip.deleteLater()
        self._refresh_footer()

    def _on_footer(self) -> None:
        if self._nested:
            self._add_strip([])
            return
        # Flat -> grouped: the values already typed become the first group.
        current = list(self._strips[0].values()) if self._strips else []
        self._rebuild(True, [current] if current else [[]])

    def _refresh_footer(self) -> None:
        if self._nested:
            self._footer.setText("＋  Add group")
            self._footer.setToolTip(
                "Add another group. Each group is one inner list — one "
                "class, one condition, one crop mode.")
            self._footer.setVisible(True)
        elif self._nested_capable:
            self._footer.setText("⌗  Use groups")
            self._footer.setToolTip(
                "This setting also accepts a list of lists. Grouping turns "
                "the values above into the first group.")
            self._footer.setVisible(True)
        else:
            self._footer.setVisible(False)

    # -- element handling ------------------------------------------------
    def _placeholder(self) -> str:
        # Short enough to survive the narrow settings column without
        # eliding -- the point of the placeholder is to say what KIND of
        # value belongs here, and an elided "add a whole numb…" says less
        # than "add number".
        if self._element_type is int:
            return "add number"
        if self._element_type is float:
            return "add number"
        if self._element_type is str:
            return "add text"
        return "add value"

    def _cast(self, text: str) -> Any:
        """Turn typed text back into the element type the list holds.

        Inferred from the default value rather than guessed per keystroke,
        so ``classes = ['1', '2']`` stays strings and ``png_dims = [0, 1, 2]``
        stays ints.
        """
        text = str(text).strip()
        if self._element_type is str:
            return text
        if self._element_type in (int, float):
            try:
                return self._element_type(text)
            except (TypeError, ValueError):
                return text
        if text.lower() == "none":
            return None
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    @staticmethod
    def _as_sequence(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            text = value.strip()
            if not text or text == "None":
                return []
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                # Not a literal: treat it as a comma-separated list, which is
                # what a user hand-editing a settings CSV most often means.
                return [part.strip() for part in text.split(",") if part.strip()]
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            return [parsed]
        return [value]


def list_shape_for(key: str, default: Any) -> Optional[Tuple[bool, bool, Any, Any]]:
    """Decide whether ``key`` is a list setting, and of what shape.

    Deliberately conservative. A key qualifies only when its *default* is
    already a list or tuple, or is ``None`` and the declared type admits
    nothing but a list. That keeps three groups of keys on their old
    widgets:

    * ``src`` and ``file_metadata``, declared ``(str, list)`` -- they are
      normally one path / one substring, and ``src`` in particular has to
      stay a ``QLineEdit`` for drag-and-drop, the empty-state banner and
      the column picker's ``_settings_src_path``;
    * ``count_data`` / ``score_data``, declared ``list`` but shipped with
      the placeholder *string* ``'list of paths'``;
    * ``sample``, whose declared "type" is the value ``None``.

    :returns: ``(nested_capable, allow_none, element_type, container)`` when
        the key holds a list, or ``None`` when it should keep its ordinary
        widget.
    """
    declared = None
    try:
        from spacr.settings import expected_types
        declared = expected_types.get(key)
    except Exception:
        declared = None
    allowed = declared if isinstance(declared, tuple) else (declared,)
    declares_list = any(t in (list, tuple) for t in allowed)
    declares_scalar = any(t in (str, int, float, bool, dict) for t in allowed)

    if isinstance(default, (list, tuple)):
        pass
    elif default is None and declares_list and not declares_scalar:
        pass
    else:
        return None

    container = tuple if (declares_list and list not in allowed) else list
    if isinstance(default, tuple) and not declares_list:
        container = tuple
    allow_none = (type(None) in allowed) or default is None
    items = list(default) if isinstance(default, (list, tuple)) else []
    flat = []
    nested_now = bool(items) and all(isinstance(i, (list, tuple)) for i in items)
    for item in items:
        flat.extend(item if isinstance(item, (list, tuple)) else [item])
    element_type = None
    # bool first: bool is a subclass of int, and a list of flags is not a
    # list of numbers.
    if flat and all(isinstance(v, str) for v in flat):
        element_type = str
    elif flat and all(isinstance(v, bool) for v in flat):
        element_type = None
    elif flat and all(isinstance(v, int) for v in flat):
        element_type = int
    elif flat and all(isinstance(v, (int, float)) for v in flat):
        element_type = float

    nested_capable = nested_now or key in NESTED_CAPABLE_KEYS or (
        isinstance(declared, tuple) and list(declared).count(list) > 1)
    return nested_capable, allow_none, element_type, container


class SettingsWidgets:
    """Container for the Qt widgets bound to a settings dict.

    Instantiate with an `app_key`; call `.build_sections()` to get a list
    of (section_title, list_of_(label, widget)) tuples to feed into the
    Section widgets on a screen. `.collect()` returns the current settings
    dict after user edits."""

    def __init__(self, app_key: str, parent: Optional[QWidget] = None):
        """Load the app's default settings dict and prepare an empty widget map.

        :param app_key: id of the app whose settings are being edited.
        :param parent: optional Qt parent for created widgets.
        """
        self.app_key = app_key
        self._parent = parent
        self._defaults = resolve_default_settings(app_key)
        self._widgets: Dict[str, QWidget] = {}
        self._tooltips = get_tooltips()
        try:
            from spacr.plugins import get_app
            plugin_app = get_app(app_key)
            if plugin_app is not None:
                self._tooltips.update(plugin_app.tooltips)
        except Exception:
            pass

    def build_sections(self) -> List[Tuple[str, List[Tuple[str, QWidget]]]]:
        """Group the settings by category and return one (title, rows)
        tuple per non-empty category, plus a trailing 'Other' section
        for anything not categorized."""
        # `spacr.settings_spec`, NOT `spacr.gui_utils`. The function is the
        # same one (gui_utils re-exports it); the module it now lives in
        # imports nothing. Reaching it through gui_utils cost 770 ms of Tk
        # dependencies -- IPython, matplotlib.pyplot, cv2, tkinter,
        # huggingface_hub -- on the GUI thread, and it was the whole remaining
        # cost of opening the first module. See spacr/settings_spec.py.
        from spacr.settings_spec import convert_settings_dict_for_gui
        variables = convert_settings_dict_for_gui(self._defaults)

        # Materialize a widget per key; attach a rich HTML tooltip that ends
        # with a compact information-icon link to the spaCR documentation.
        for key, meta in variables.items():
            kind, options, default = meta
            widget = self._widget_for(kind, options, default, key)
            if widget is not None:
                attach_api_tooltip(
                    widget,
                    self.app_key,
                    key,
                    _descriptions=self._tooltips,
                )
                self._widgets[key] = widget

        src_widget = self._widgets.get("src")
        if isinstance(src_widget, QLineEdit):
            src_widget.editingFinished.connect(
                self._refresh_contextual_widgets)
        self._refresh_contextual_widgets()

        # Bucket into sections.
        cats = categories_for_app(self.app_key, get_categories())
        used_keys = set()
        # Categories that don't apply to a given app (e.g. the classify app
        # trains a Torch model, not Cellpose — so it gets no Cellpose tab).
        hidden = _APP_HIDDEN_CATEGORIES.get(self.app_key, set())
        sections: List[Tuple[str, List[Tuple[str, QWidget]]]] = []
        for cat_name, keys in cats.items():
            if cat_name in hidden:
                continue
            rows: List[Tuple[str, QWidget]] = []
            for k in keys:
                if k in self._widgets and k not in used_keys:
                    rows.append((self._label_for(k), self._widgets[k]))
                    used_keys.add(k)
            if rows:
                sections.append((cat_name, rows))

        # Trailing 'Other' for anything not in a category.
        remaining = [(self._label_for(k), self._widgets[k])
                     for k in self._widgets if k not in used_keys]
        if remaining:
            sections.append(("Other", remaining))

        return sections

    def tooltip_for(self, key: str) -> str:
        """Return the HTML-formatted tooltip for a given setting key."""
        return format_tooltip(self._tooltips.get(key, ""), self.app_key, key)

    def plain_tooltip_for(self, key: str) -> str:
        """Return the plain-text hint (description + docs URL) for a setting."""
        return plain_tooltip(self._tooltips.get(key, ""), self.app_key, key)

    # ------------------------------------------------------------------
    # Finding a setting among the many
    # ------------------------------------------------------------------
    #
    # Mask alone renders 190 settings under thirteen collapsed headings.
    # Someone who knows the knob exists still has to guess which heading
    # somebody else filed it under, and someone who only knows what they want
    # to change ("stop merging touching cells") has no entry point at all.
    #
    # So the haystack is deliberately wider than the key: the description is
    # the only part of a setting written in the language a user thinks in.
    # Searching "gpu" has to find `n_jobs`, and "touching" has to find
    # `merge_edge_pathogen_cells`, and neither word is in either name.

    def search_text_for(self, key: str) -> str:
        """The lower-cased haystack one setting is matched against.

        Three fields, in the order a reader would scan them: the key as the
        API spells it, the label as the form spells it, and the description
        as the tooltip explains it.

        :param key: the setting key.
        """
        return " ".join((
            str(key),
            self._label_for(key),
            self.plain_tooltip_for(key),
        )).lower()

    def keys_matching(self, query: str) -> List[str]:
        """Setting keys matching every whitespace-separated term in ``query``.

        Terms are ANDed and matched as substrings, which is what makes
        "cell diameter" narrow rather than widen — the alternative, OR, turns
        a second word into a way of getting *more* results, which is the
        opposite of what typing more means.

        An empty or whitespace-only query matches everything, so the caller
        can wire this straight to ``textChanged`` without special-casing the
        moment the box is cleared.

        :param query: raw text from the search box.
        :returns: matching keys, in the order the widgets were built.
        """
        terms = str(query or "").lower().split()
        if not terms:
            return list(self._widgets)
        out: List[str] = []
        for key in self._widgets:
            haystack = self.search_text_for(key)
            if all(term in haystack for term in terms):
                out.append(key)
        return out

    def modified_keys(self) -> List[str]:
        """Setting keys whose widget no longer holds the module's default.

        Compared with the same normaliser the run journal and the settings
        diff use, so "differs from default" means one thing across the app.
        Without that, a value round-tripped through CSV — ``channels`` read
        back as the string ``"[0, 1, 2]"`` — reads as an edit here and as
        unchanged there.

        :returns: keys in the order the widgets were built.
        """
        from ..settings_diff import _values_equal

        out: List[str] = []
        for key, widget in self._widgets.items():
            if key not in self._defaults:
                # Rendered but not defaulted: there is nothing to differ
                # from, so calling it modified would be an assertion the
                # module never made.
                continue
            try:
                current = self._coerce_to_expected_type(
                    key, self._read_widget(widget))
            except Exception:
                continue
            if not _values_equal(current, self._defaults[key]):
                out.append(key)
        return out

    def essential_keys(self) -> List[str]:
        """The rendered subset of :func:`essential_keys` for this module.

        Filtered to keys that actually produced a widget, so a key named in
        a layout but skipped by ``convert_settings_dict_for_gui`` cannot make
        the disclosure control promise a row that is not there.
        """
        return [key for key in essential_keys(self.app_key)
                if key in self._widgets]

    def _label_for(self, key: str) -> str:
        try:
            from spacr.plugins import get_app
            plugin_app = get_app(self.app_key)
            if plugin_app is not None and key in plugin_app.labels:
                return plugin_app.labels[key]
        except Exception:
            pass
        if self.app_key in ("measure", "external_masks"):
            measure_labels = {
                "uninfected": "Keep uninfected cells",
                "cytoplasm": "Measure cytoplasm",
                "merge_edge_pathogen_cells": "Merge edge-pathogen cells",
            }
            if key in measure_labels:
                return measure_labels[key]
        if self.app_key == "umap":
            if key == "exclude_rows":
                return "Exclude"
            if key == "exclude":
                return "Exclude features"
        return key.replace("_", " ").capitalize()

    def _widget_for(self, kind: str, options: Any, default: Any,
                    key: str) -> Optional[QWidget]:
        parent = self._parent
        if self.app_key == "umap" and key == "exclude_rows":
            return RowExclusionEditor(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        if self.app_key == "external_masks" and key == "inputs":
            return ExternalMaskInputWidget(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        app_options = _APP_COMBO_OPTIONS.get(self.app_key, {})
        if key in app_options:
            kind = "combo"
            options = app_options[key]
        if self.app_key == "map_barcodes" and key == "regex":
            return BarcodeRegexWidget(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        # A closed alphabet gets a control that cannot express anything
        # outside it. Checked BEFORE the chip-editor override below, because
        # `train_channels` is in CHANNEL_LIST_KEYS and would otherwise take
        # the free-text path that let 'x' through.
        if key in FIXED_ALPHABETS:
            return _AlphabetSelect(
                key=key,
                default=self._defaults.get(key, default),
                choices=FIXED_ALPHABETS[key],
                parent=parent,
            )
        # Unlike enumerated strings, a list remains a list in every module.
        # The legacy converter presents channel lists and timelapse objects as
        # dropdowns of Python literals. Render them with the same chip editor
        # as manders_thresholds so users can add/remove arbitrary values.
        actual_default = self._defaults.get(key, default)
        if key == "timelapse_objects" or (
            key in CHANNEL_LIST_KEYS
            and list_shape_for(key, actual_default) is not None
        ):
            kind = "entry"
        if kind == "check":
            w = Toggle()
            w.setChecked(bool(default))
            return w
        if kind == "combo":
            w = QComboBox()
            for opt in (options or []):
                w.addItem("None" if opt is None else str(opt),
                          userData=opt)
            # Pre-select the value THIS module declares, not the one
            # hard-coded in gui_utils.convert_settings_dict_for_gui's
            # special_cases table. That table is one row per key for the whole
            # app, so it shipped 'resnet50' as the model_type default to
            # Classify (which sets 'maxvit_t') and to Activation Maps (which
            # sets 'maxvit'), and '[0,1,2,3]' as the channels default to
            # Cellpose Masks (which sets [0, 0]).
            if key in self._defaults:
                default = self._defaults[key]
            for i in range(w.count()):
                if w.itemData(i) == default or w.itemText(i) == str(default):
                    w.setCurrentIndex(i)
                    break
            else:
                # The default is not one of the curated options. Silently
                # leaving index 0 selected substitutes a value the module
                # never asked for -- the activation-map app defaults
                # channels to [1, 2, 3] and the channel combo only lists
                # '[0,1,2,3]', so every run started with a different channel
                # set than the defaults declare. Offer the real default too.
                if default is not None and str(default) != "":
                    w.insertItem(0, str(default), userData=default)
                    w.setCurrentIndex(0)
            return w
        if kind == "entry":
            # A list setting gets the chip editor, not a text box holding a
            # Python literal. The shape is decided from expected_types plus
            # the REAL default (self._defaults), because
            # convert_settings_dict_for_gui has already str()'d the value
            # that arrives here as ``default``.
            shape = list_shape_for(key, self._defaults.get(key, default))
            if shape is not None:
                nested_capable, allow_none, element_type, container = shape
                return _ListEditor(key=key,
                                   default=self._defaults.get(key, default),
                                   nested_capable=nested_capable,
                                   allow_none=allow_none,
                                   element_type=element_type,
                                   container=container)
            # Choose widget by inferred type from the DEFAULT value
            if isinstance(default, bool):
                w = Toggle()
                w.setChecked(default)
                return w
            if isinstance(default, int):
                w = QSpinBox()
                # Wide enough for the defaults the modules actually ship:
                # the replication assay's max_area is 1e9, and a +/-1e6 range
                # silently clamped it to 1e6 -- a thousand-fold change to the
                # largest vacuole the assay will score, applied before the
                # user touched anything.
                w.setRange(-2_147_483_648, 2_147_483_647)
                w.setValue(default)
                return w
            if isinstance(default, float):
                w = QDoubleSpinBox()
                w.setRange(-1e12, 1e12)
                w.setDecimals(6)
                w.setValue(default)
                return w
            if isinstance(default, list):
                w = _ListEdit()
                w.set_value(default)
                return w
            # Fallback — string or None
            w = _ScalarEdit()
            w.set_value(default)
            return w
        return None

    @staticmethod
    def _coerce_to_expected_type(key: str, value: Any) -> Any:
        """Parse a raw widget string into the type ``settings`` declares.

        A setting whose DEFAULT is None gets a free-text widget, so it comes
        back as a raw string even when ``spacr.settings.expected_types`` says
        it is an int -- and cellpose received ``diameter='37'``. The Tk GUI
        never had this problem because it runs
        ``settings.check_settings(vars_dict, expected_types)`` before
        dispatch; the Qt path had no equivalent step. check_settings itself
        cannot be reused here: it takes the Tk widget map
        ``key -> (label, widget, var, frame)``, not a plain dict.

        Anything not declared, or not parseable, is returned untouched -- this
        coerces, it does not validate, and it must never turn a real value
        into None behind the user's back.
        """
        if not isinstance(value, str):
            return value
        try:
            from ... import settings as _settings
            declared = _settings.expected_types.get(key)
        except Exception:
            return value
        if declared is None:
            return value
        allowed = declared if isinstance(declared, tuple) else (declared,)
        text = value.strip()
        if text == "" or text == "None":
            return None if type(None) in allowed else value
        for typ in allowed:
            if typ is bool:
                if text.lower() in ("true", "false"):
                    return text.lower() == "true"
                continue
            if typ in (int, float):
                try:
                    return typ(text)
                except ValueError:
                    continue
            if typ in (list, tuple):
                # The curated combos ('channels', 'crop_mode',
                # 'train_channels', 'timelapse_objects', ...) offer their
                # options as TEXT -- "['r','g','b']" -- so a list setting
                # picked from a dropdown reached the pipeline as a string and
                # got iterated character by character. The chip editor already
                # returns a real list; this is the same repair for the combos.
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, (list, tuple)):
                    return typ(parsed)
                continue
        return value

    def collect(self) -> Dict[str, Any]:
        """Read all widgets and return the current settings dict."""
        out: Dict[str, Any] = {}
        for key, w in self._widgets.items():
            out[key] = self._coerce_to_expected_type(key, self._read_widget(w))
        # Also carry over any defaults we didn't render (e.g. things not
        # in the categories map that convert_settings_dict_for_gui also
        # skipped).
        for k, v in self._defaults.items():
            out.setdefault(k, v)
        return out

    def set_value_for_key(self, key: str, value: Any) -> bool:
        """Write ``value`` into the widget bound to ``key`` (if present).

        Used by the Live Preview's "Propagate settings" toggle to push
        interactively-tuned values back into the main settings panel.
        Returns True if the key existed and was set.
        """
        w = self._widgets.get(key)
        if w is None:
            return False
        try:
            if isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(value))
            elif isinstance(w, QComboBox):
                idx = w.findData(value)
                if idx < 0:
                    idx = w.findText(str(value))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                else:
                    w.setEditText(str(value))
            elif isinstance(
                w,
                (
                    _AlphabetSelect, _ListEditor, _ListEdit, _ScalarEdit,
                    BarcodeRegexWidget, RowExclusionEditor,
                    ExternalMaskInputWidget,
                ),
            ):
                w.set_value(value)
            elif isinstance(w, QLineEdit):
                w.setText("" if value is None else str(value))
            else:
                return False
        except Exception:
            return False
        if key in {"src", "tables"}:
            self._refresh_contextual_widgets()
        return True

    def _refresh_contextual_widgets(self) -> None:
        """Refresh widgets whose choices come from the selected data source."""
        editor = self._widgets.get("exclude_rows")
        if not isinstance(editor, RowExclusionEditor):
            return
        src_widget = self._widgets.get("src")
        tables_widget = self._widgets.get("tables")
        source = self._read_widget(src_widget) if src_widget is not None else None
        tables = (
            self._read_widget(tables_widget)
            if tables_widget is not None
            else self._defaults.get("tables")
        )
        editor.set_source(source, tables)

    def _read_widget(self, w: QWidget) -> Any:
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        if isinstance(w, QSpinBox):
            return int(w.value())
        if isinstance(w, QDoubleSpinBox):
            return float(w.value())
        if isinstance(w, QComboBox):
            idx = w.currentIndex()
            # EVERY item is added with userData=opt, including the Python None
            # option (`addItem("None" if opt is None else str(opt),
            # userData=opt)`). So currentData() returning None means the chosen
            # option IS None -- not that the item carries no data. The old
            # fallback to currentText() therefore handed back the STRING
            # 'None', which is how every Qt run shipped strict_errors='None'
            # and turned strict error handling silently ON, since
            # errors.strict_errors() saw a non-None value and took
            # bool('None') == True. cov_type and 'transform' reached
            # statsmodels the same way.
            #
            # currentText() is still right for an EDITABLE combo showing
            # something the user typed that is not in the list -- detected by
            # the displayed text not matching the current item's text.
            if idx >= 0 and w.itemText(idx) == w.currentText():
                return w.itemData(idx)
            return w.currentText()
        if isinstance(
            w,
            (
                _AlphabetSelect, _ListEditor, _ListEdit, BarcodeRegexWidget,
                RowExclusionEditor, ExternalMaskInputWidget,
            ),
        ):
            return w.get_value()
        if isinstance(w, _ScalarEdit):
            return w.get_value()
        if isinstance(w, QLineEdit):
            return w.text() or None
        return None
