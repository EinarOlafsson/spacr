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


def resolve_default_settings(app_key: str) -> Dict[str, Any]:
    """Return a fresh defaults dict for an app key, mirroring the Tk GUI
    dispatch in gui_core.setup_settings_panel."""
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
        ],
        "batch_missing_control": ["error", "skip"],
    },
    "ml_analyze": {
        "batch_correction": [
            "none", "control_center", "robust_zscore", "center", "zscore",
        ],
        "batch_missing_control": ["error", "skip"],
    },
    "regression": {
        "batch_correction": [
            "none", "control_center", "robust_zscore", "center", "zscore",
        ],
        "batch_missing_control": ["error", "skip"],
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
            "batch_control_values", "batch_min_samples",
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
            "strict_errors", "max_failure_rate", "verbose", "n_jobs",
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
            "resume", "strict_errors", "max_failure_rate", "dry_run",
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
            "strict_errors", "max_failure_rate", "dry_run", "verbose",
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
            "filter_column", "filter_value",
        )),
        ("Plate & Batch Correction", (
            "batch_correction", "batch_column", "batch_control_column",
            "batch_control_values", "batch_min_samples",
            "batch_missing_control",
        )),
        ("Model & Covariates", (
            "regression_type", "dependent_variable", "agg_type", "transform",
            "alpha", "cov_type", "random_row_column_effects",
        )),
        ("Hit Calling & Outliers", (
            "min_cell_count", "fraction_threshold", "target_unique_count",
            "outlier_detection", "threshold_method", "threshold_multiplier",
            "min_n", "toxo",
        )),
        ("Regression Plots", (
            "volcano", "log_x", "log_y", "x_lim", "split_axis_lims",
        )),
        ("Runtime & Reliability", ("strict_errors", "max_failure_rate")),
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
            candidates = (
                source.get(token[1:], [])
                if token.startswith("@") else [token]
            )
            for key in candidates:
                if key in available and key not in assigned:
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
    if app_key == "map_barcodes":
        moved = ("n_jobs", "test")
        for keys in result.values():
            for key in moved:
                while key in keys:
                    keys.remove(key)
        sequencing = result.setdefault("Sequencing", [])
        for key in moved:
            if key not in sequencing:
                sequencing.append(key)

    if app_key == "umap":
        batch_correction = (
            "batch_correction", "batch_column", "batch_control_column",
            "batch_control_values", "batch_min_samples",
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
    "report": "report",
    "recruitment": "submodules",
    "analyze_plaques": "submodules",
    "invasion": "submodules",
    "replication": "submodules",
    "figure": "plot",
    "ai": "qt/ai",
}


def api_docs_url(app_key: str, key: str = "") -> str:
    """Return the spaCR API URL for an app or shared setting.

    Known app keys land on their module page. New or UI-only modules fall
    back to the generated API index rather than the documentation homepage.
    Shared batch-correction settings always land on their implementation,
    rather than whichever consumer app happens to display them.
    """
    module = (
        "batch_correction"
        if key.startswith("batch_")
        else _APP_API_MODULE.get(app_key)
    )
    if module:
        return f"{DOCS_API_BASE}/spacr/{module}/index.html"
    return f"{DOCS_API_BASE}/index.html"


_TYPE_NAMES = {int: "integer", float: "float", bool: "boolean",
               str: "string", list: "list", tuple: "tuple", dict: "dict"}


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


def format_tooltip(text: str, app_key: str, key: str = "") -> str:
    """Return a typed HTML tooltip ending in a clickable API-doc link."""
    body = escape(" ".join(_strip_type_prefix(text).split()))
    header = escape(_humanize(key))
    th = escape(_type_hint(key))
    if header and th:
        header = f"<b>{header}</b> <i>({th})</i>"
    elif header:
        header = f"<b>{header}</b>"
    if not body:
        body = f"Controls {escape(_humanize(key).lower())}." if key else \
            "Controls this setting."
    url = escape(api_docs_url(app_key, key), quote=True)
    link = f'<a href="{url}">Open spaCR API documentation</a>'
    parts = [p for p in (header, body, link) if p]
    return "<br>".join(parts)


def plain_tooltip(text: str, app_key: str, key: str = "") -> str:
    """Same content as `format_tooltip` but plain text — used by the
    hover-follows footer at the bottom of each AppScreen."""
    body = " ".join(_strip_type_prefix(text).split())
    th = _type_hint(key)
    name = _humanize(key)
    head = f"{name} ({th})" if (name and th) else name
    parts = [p for p in (head, body) if p]
    summary = " — ".join(parts)
    url = api_docs_url(app_key, key)
    return f"{summary} — API: {url}" if summary else f"API: {url}"


class _ApiTooltipFilter(QObject):
    """Show rich setting help in the clickable sticky tooltip."""

    def eventFilter(self, watched, event):  # noqa: N802 (Qt naming)
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
    body = descriptions.get(key) or description or widget.property(
        "apiTooltipDescription") or widget.toolTip()
    body = str(body or f"Controls {_humanize(key).lower()}.")
    html = format_tooltip(body, app_key, key)
    widget.setProperty("settingsAppKey", app_key)
    widget.setProperty("settingKey", key)
    widget.setProperty("apiTooltipDescription", body)
    widget.setProperty("apiTooltipHtml", html)
    widget.setToolTip(html)
    widget.setToolTipDuration(-1)
    return html


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
            widget.installEventFilter(event_filter)
            _add_api_dot_to_combined_control(
                owner, widget, app_key, key, html)
            continue

        label.setCursor(Qt.WhatsThisCursor)
        label.setProperty("settingHelpLabel", True)
        label.setProperty("settingsAppKey", app_key)
        label.setProperty("settingKey", key)
        label.setProperty("apiTooltipHtml", html)
        label.setToolTip(html)
        label.setToolTipDuration(-1)
        label.installEventFilter(event_filter)

        # The editor itself remains quiet on hover. Keep its metadata so tests,
        # integrations and a later re-parenting pass can still identify it.
        widget.setToolTip("")
        widget.removeEventFilter(event_filter)
        _add_api_dot_to_label(label, app_key, key, html)


def _setting_label_for_field(owner: QWidget, field: QWidget) -> Optional[QWidget]:
    """Find the visual label immediately to the left of a popup field."""
    remembered = getattr(field, "_spacr_setting_label", None)
    if isinstance(remembered, QWidget):
        try:
            remembered.objectName()
            if remembered.window() is owner.window():
                return remembered
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
            label = form.labelForField(candidate)
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

    from ..widgets.info_link import InfoLink
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
    dot = InfoLink(
        api_docs_url(app_key, key),
        tooltip=f"Open API reference for {_humanize(key)}",
        parent=host,
    )
    dot.setObjectName("SettingInfoLink")
    dot.setProperty("apiTooltipHtml", html)
    row.addWidget(dot)
    label.setProperty("settingApiDotInstalled", True)
    label._spacr_api_dot = dot


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

    from ..widgets.info_link import InfoLink
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
    dot = InfoLink(
        api_docs_url(app_key, key),
        tooltip=f"Open API reference for {_humanize(key)}",
        parent=host,
    )
    dot.setObjectName("SettingInfoLink")
    dot.setProperty("apiTooltipHtml", html)
    row.addWidget(dot)
    row.addStretch(1)
    field._spacr_api_dot = dot


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
                font-size: 12px;
            }}
            QToolButton#SettingChipClose {{
                color: {colours['fg_muted']};
                background: transparent;
                border: none;
                padding: 0px 2px;
                font-size: 13px;
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
        from ..theme import active_palette
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
        from ..theme import active_palette
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
            f" background: transparent; border: none; font-size: 12px;"
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

    def build_sections(self) -> List[Tuple[str, List[Tuple[str, QWidget]]]]:
        """Group the settings by category and return one (title, rows)
        tuple per non-empty category, plus a trailing 'Other' section
        for anything not categorized."""
        from spacr.gui_utils import convert_settings_dict_for_gui
        variables = convert_settings_dict_for_gui(self._defaults)

        # Materialize a widget per key; attach a rich HTML tooltip that ends
        # with a compact information-icon link to the spaCR documentation.
        for key, meta in variables.items():
            kind, options, default = meta
            widget = self._widget_for(kind, options, default, key)
            if widget is not None:
                tip = format_tooltip(self._tooltips.get(key, ""), self.app_key, key)
                widget.setProperty("settingsAppKey", self.app_key)
                widget.setProperty("settingKey", key)
                widget.setProperty("apiTooltipHtml", tip)
                widget.setToolTip(tip)
                widget.setToolTipDuration(-1)  # respect system default (persistent)
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

    def _label_for(self, key: str) -> str:
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
                    _ListEditor, _ListEdit, _ScalarEdit, BarcodeRegexWidget,
                    RowExclusionEditor, ExternalMaskInputWidget,
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
                _ListEditor, _ListEdit, BarcodeRegexWidget,
                RowExclusionEditor, ExternalMaskInputWidget,
            ),
        ):
            return w.get_value()
        if isinstance(w, _ScalarEdit):
            return w.get_value()
        if isinstance(w, QLineEdit):
            return w.text() or None
        return None
