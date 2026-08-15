"""Structural invariants of ``spacr.settings.categories``.

The category map is the ONLY thing that groups the settings panel in either
GUI -- the Tk dropdown built by ``gui_core.toggle_settings`` and the Qt section
boxes built by ``qt.screens.settings_model.SettingsWidgets.build_sections``
both read it and nothing else. Regrouping it is therefore a user-visible
change with a silent failure mode: a key that drops out of the map stops being
grouped at all (Tk pins it to the top of the panel, Qt dumps it into "Other"),
and a key that appears twice is rendered twice by Tk and dropped from the
second section by Qt.

These tests pin the invariants so the map can be reorganised again later
without losing anything:

* no key that was ever categorised falls out of the map;
* every key appears in exactly one category;
* every categorised key is a real setting;
* every setting a GUI module offers has a category;
* the organelle settings live under exactly one heading;
* the category names the GUIs reference by string literal still exist.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

import spacr.settings as S


# ---------------------------------------------------------------------------
# Frozen baselines
# ---------------------------------------------------------------------------

#: Every key that was categorised before the categories map was regrouped
#: (generated from the version of spacr/settings.py at that commit). Keys may
#: be ADDED to the map, never dropped: dropping one removes the setting from
#: its group in both GUIs.
KEYS_BEFORE_REGROUP = frozenset({
    "CP_prob", "Signal_to_noise", "adjust_cells", "agg_type", "all_to_mip", "alpha", "amsgrad",
    "analyze_clusters", "annotated_classes", "annotation_column", "apply_model_to_dataset",
    "augment", "background", "backgrounds", "barcodes", "batch_size", "black_background",
    "calculate_correlation", "cam_type", "cell_CP_prob", "cell_FT", "cell_Signal_to_noise",
    "cell_area_multiplier", "cell_background", "cell_chann_dim", "cell_channel",
    "cell_diameter", "cell_intensity_merge", "cell_intensity_percentile",
    "cell_intensity_range", "cell_intensity_split", "cell_intensity_threshold_method",
    "cell_loc", "cell_mask_dim", "cell_max_area", "cell_max_intensity_percentile",
    "cell_min_area", "cell_min_distance", "cell_min_intensity_percentile",
    "cell_min_object_area", "cell_min_size", "cell_perimeter_fraction", "cell_plate_metadata",
    "cell_remove_border_objects", "cell_size_range", "cell_types", "cells", "cells_per_well",
    "channel_dims", "channel_of_interest", "channels", "chunk_size", "class_balance",
    "class_metadata", "classes", "clustering", "cmap", "col_to_compare", "color_by",
    "column_csv", "comp_level", "comp_type", "compartments", "compression", "consolidate",
    "controls", "correlation", "count_data", "cov_type", "crop_mode", "cross_validation",
    "cross_validation_folds", "custom_measurement", "custom_model", "custom_model_path",
    "custom_regex", "cv_group_by", "cytoplasm", "cytoplasm_min_size", "dataset",
    "dataset_mode", "db_table_name", "delete_intermediate", "denoise", "dependent_variable",
    "dialate_png_ratios", "dialate_pngs", "diameter", "diameter_estimate_n_fields",
    "distance_gaussian_sigma", "dot_size", "dropout_rate", "dry_run", "embedding_by_controls",
    "epochs", "eps", "examples_to_plot", "exclude", "exclude_conditions", "expected_end",
    "experiment", "figuresize", "file_metadata", "file_type", "fill_in", "filter", "filter_by",
    "filter_column", "filter_value", "flow_threshold", "fps", "fraction_threshold",
    "from_scratch", "generate_training_dataset", "gradient_accumulation",
    "gradient_accumulation_steps", "grayscale", "grna", "grna_csv", "grouping",
    "heatmap_feature", "highlight", "homogeneity", "homogeneity_distances", "image_nr",
    "image_size", "img_zoom", "infection_intensity_mode", "infection_intensity_n_bins",
    "infection_intensity_qc", "infection_intensity_qc_graphs",
    "infection_intensity_qc_panel_path", "infection_intensity_qc_scope",
    "infection_intensity_strategy", "infection_pca_log_intensity", "infection_pca_max_cells",
    "infection_pca_method", "infection_pca_min_gt_separation", "infection_pca_min_silhouette",
    "infection_pca_n_clusters", "infection_pca_pathogen_weight", "infection_pca_random_state",
    "infection_pca_tsne_learning_rate_grid", "infection_pca_tsne_perplexity",
    "infection_pca_tsne_perplexity_grid", "infection_pca_tsne_search",
    "infection_pca_umap_min_dist", "infection_pca_umap_min_dist_grid",
    "infection_pca_umap_n_neighbors", "infection_pca_umap_n_neighbors_grid",
    "infection_pca_umap_search", "infection_xgb_ambiguous_high", "infection_xgb_ambiguous_low",
    "infection_xgb_colsample_bytree", "infection_xgb_drop_ambiguous",
    "infection_xgb_learning_rate", "infection_xgb_margin", "infection_xgb_max_depth",
    "infection_xgb_min_cells_per_class", "infection_xgb_n_estimators", "infection_xgb_n_jobs",
    "infection_xgb_proba", "infection_xgb_proba_column", "infection_xgb_proba_threshold",
    "infection_xgb_random_state", "infection_xgb_reg_lambda", "infection_xgb_subsample",
    "infection_xgb_top_features", "init_weights", "intermedeate_save", "invert",
    "keep_intermediate", "keep_original_images", "learning_rate", "location_column",
    "log_data", "log_x", "log_y", "loss_type", "lower_percentile", "magnification",
    "manders_thresholds", "masks", "max_displacement", "measurement",
    "merge_edge_pathogen_cells", "merge_pathogens", "metadata_files", "metadata_type",
    "metadata_type_by", "metadata_types", "metric", "min_cell_count", "min_dist", "min_max",
    "min_n", "min_samples", "minimum_cell_count", "mix", "mode", "model_name", "model_path",
    "model_type", "model_type_ml", "motility_analysis", "motility_xlim", "motility_ylim",
    "n_epochs", "n_estimators", "n_jobs", "n_neighbors", "n_repeats", "nc", "nc_loc", "neg",
    "negative_control", "normalize", "normalize_by", "normalize_input", "normalize_plots",
    "nr_imgs", "nuclei_limit", "nucleus_CP_prob", "nucleus_FT", "nucleus_Signal_to_noise",
    "nucleus_area_multiplier", "nucleus_background", "nucleus_chann_dim", "nucleus_channel",
    "nucleus_diameter", "nucleus_intensity_merge", "nucleus_intensity_percentile",
    "nucleus_intensity_range", "nucleus_intensity_split", "nucleus_intensity_threshold_method",
    "nucleus_loc", "nucleus_mask_dim", "nucleus_max_area", "nucleus_max_intensity_percentile",
    "nucleus_min_area", "nucleus_min_distance", "nucleus_min_intensity_percentile",
    "nucleus_min_object_area", "nucleus_min_size", "nucleus_perimeter_fraction",
    "nucleus_remove_border_objects", "nucleus_size_range", "offset", "offset_start",
    "optimizer_type", "organelle_CP_prob", "organelle_FT", "organelle_adaptive_block_size",
    "organelle_adaptive_offset", "organelle_area_multiplier", "organelle_chann_dim",
    "organelle_channel", "organelle_clahe", "organelle_clahe_clip_limit", "organelle_diameter",
    "organelle_dog_sigma_high", "organelle_dog_sigma_low", "organelle_fill_holes",
    "organelle_hysteresis_high", "organelle_hysteresis_low", "organelle_intensity_merge",
    "organelle_intensity_percentile", "organelle_intensity_split",
    "organelle_intensity_threshold_method", "organelle_log_max_sigma",
    "organelle_log_min_sigma", "organelle_log_num_sigma", "organelle_log_threshold",
    "organelle_mask_dim", "organelle_mask_within_cells", "organelle_max_area",
    "organelle_max_intensity_percentile", "organelle_max_size", "organelle_method",
    "organelle_min_area", "organelle_min_distance", "organelle_min_intensity_percentile",
    "organelle_min_object_area", "organelle_min_size", "organelle_model_name",
    "organelle_morph_radius", "organelle_morphology", "organelle_network_threshold",
    "organelle_perimeter_fraction", "organelle_remove_border",
    "organelle_remove_border_objects", "organelle_resample", "organelle_ridge_filter",
    "organelle_ridge_sigmas", "organelle_ring_fill_method", "organelle_ring_min_prominence",
    "organelle_ring_sigma_inner", "organelle_ring_sigma_outer", "organelle_rolling_ball",
    "organelle_rolling_ball_radius", "organelle_skeletonize", "organelle_tophat_radius",
    "organelle_unet_model_path", "organelle_unet_threshold", "organelle_watershed_spots",
    "other", "outlier_detection", "overlay", "pathogen_CP_prob", "pathogen_FT",
    "pathogen_Signal_to_noise", "pathogen_area_multiplier", "pathogen_background",
    "pathogen_chann_dim", "pathogen_channel", "pathogen_diameter", "pathogen_intensity_merge",
    "pathogen_intensity_percentile", "pathogen_intensity_range", "pathogen_intensity_split",
    "pathogen_intensity_threshold_method", "pathogen_limit", "pathogen_loc",
    "pathogen_mask_dim", "pathogen_max_area", "pathogen_max_intensity_percentile",
    "pathogen_min_area", "pathogen_min_distance", "pathogen_min_intensity_percentile",
    "pathogen_min_object_area", "pathogen_min_size", "pathogen_model",
    "pathogen_perimeter_fraction", "pathogen_plate_metadata", "pathogen_remove_border_objects",
    "pathogen_size_range", "pathogen_types", "pathogens", "pc", "pc_loc", "percentiles",
    "pin_memory", "pixels_per_um", "plate", "plot", "plot_by_cluster", "plot_cluster_grids",
    "plot_control", "plot_images", "plot_nr", "plot_outlines", "plot_points", "png_dims",
    "png_size", "png_type", "pos", "positive_control", "preprocess", "prune_features",
    "radial_dist", "random_row_column_effects", "random_test", "randomize", "reduction_method",
    "reg_alpha", "reg_lambda", "regex", "regression_type", "remove_background",
    "remove_background_cell", "remove_background_nucleus", "remove_background_pathogen",
    "remove_border_cells", "remove_border_nuclei", "remove_border_organelles",
    "remove_border_pathogens", "remove_cluster_noise", "remove_highly_correlated",
    "remove_highly_correlated_features", "remove_image_canvas", "remove_low_variance_features",
    "resample", "rescale", "resize", "resnet_features", "reuse_existing_measurements",
    "row_csv", "row_limit", "sample", "save", "save_arrays", "save_figure", "save_h5",
    "save_measurements", "save_png", "save_to_db", "schedule", "score_data", "score_threshold",
    "seconds_per_frame", "seg_qc", "seg_qc_border_fraction", "seg_qc_count_ratio",
    "seg_qc_foreground_fraction", "seg_qc_max_object_fraction", "seg_qc_min_diameter",
    "seg_qc_min_objects", "seg_qc_outlier_fraction", "seg_qc_outlier_mad",
    "seg_qc_plate_fail_fraction", "seg_qc_size_ratio", "seg_qc_split_ratio",
    "seg_qc_tiny_fraction", "shuffle", "signal_direction", "single_direction", "size",
    "smooth_lines", "split_axis_lims", "src", "straightness_filter", "straightness_threshold",
    "summarize_organelles_by", "tables", "target", "target_height", "target_intensity_min",
    "target_layer", "target_sequence", "target_unique_count", "target_width", "test",
    "test_images", "test_mode", "test_nr", "test_size", "test_split", "threshold_method",
    "threshold_multiplier", "timelapse", "timelapse_displacement", "timelapse_frame_limits",
    "timelapse_memory", "timelapse_mode", "timelapse_objects", "timelapse_remove_transient",
    "top_features", "toxo", "trackastra_linking", "trackastra_model", "tracked_object",
    "train", "train_channels", "transform", "treatment_loc", "treatment_plate_metadata",
    "treatments", "ultrack_contour_sigma", "ultrack_division_weight", "ultrack_max_distance",
    "ultrack_n_workers", "um_per_pixel", "uninfected", "upscale", "upscale_factor",
    "use_bounding_box", "use_checkpoint", "use_sam_cell", "use_sam_nucleus",
    "use_sam_pathogen", "val_split", "verbose", "visualize", "volcano", "weight_decay",
    "width_height", "x_lim", "zscore_thresh",
})

#: Keys added to the map by the regroup. They were previously offered by a
#: module but had no category, so they rendered ungrouped. Extending this set
#: is fine; it exists so that "the union grew" is always a deliberate act.
#: Settings RETIRED on 2026-08-11, at the maintainer's instruction to
#: "remove dead settings entirely". They used to live in a ``DEAD_SETTINGS``
#: registry that kept them declared so an old CSV could be told what to use
#: instead; that registry is gone and so are they. Named here, once, so a key
#: legitimately dropping out of the category map is distinguishable from one
#: that fell out by accident -- which is the whole point of this file.
KEYS_RETIRED = frozenset({
    "all_to_mip", "barecode_length_1", "barecode_length_2",
    "class_1_threshold", "custom_measurement", "gene_weights_csv",
    "metadata_types", "nc", "nc_loc", "nucleus_loc", "pc", "pc_loc",
    "pick_slice", "postprocess_cell_masks", "postprocess_nucleus_masks",
    "postprocess_organelle_masks", "postprocess_pathogen_masks",
    "redunction_method", "remove_border_cells", "remove_border_nuclei",
    "remove_border_organelles", "remove_border_pathogens",
    "signal_direction", "skip_mode", "use_sam_cell", "use_sam_nucleus",
    "use_sam_pathogen",
    # Retired 2026-08-12. Verified dead before being named here: neither is
    # produced by any set_default_*/get_*_settings helper, neither is in
    # expected_types, and a grep of spacr/ finds no reader of either -- Tk
    # included. They were falling out of the category map with nothing to
    # say whether that was deliberate, which is the exact ambiguity this set
    # exists to remove.
    "highlight", "offset",
    # A GHOST, not a setting: `infection_xgb_proba` was in the Motility
    # Advanced category list, but the setting is `infection_xgb_proba_column`
    # -- whose DEFAULT VALUE is the string 'infection_xgb_proba'. The value
    # had been pasted into the category list beside the key it belongs to.
    "infection_xgb_proba",
})


KEYS_ADDED_BY_REGROUP = frozenset({
    # The two readable front ends for choices that were previously side
    # effects of other keys: `inference` selects analysis_mode (and 'auto'
    # picks it from whether the design can support a simultaneous fit), and
    # `analysis_unit` spells out the per-well/per-cell switch that agg_type
    # used to make silently by being set to None.
    "inference", "analysis_unit",
    # Plate-blocked marginal guide analysis added to the regression workflow.
    "analysis_mode", "guide_min_wells", "guide_primary_min_wells",
    "guide_permutations", "guide_permutation_seed",
    "guide_permutation_block", "guide_nuisance_columns",
    "guide_presence_threshold", "guide_permutation_batch_size",
    "guide_permutation_plot", "multiple_testing_method", "fdr_alpha",
    # Image UMAP's reducer families and the one shared GPU execution switch.
    "gpu", "tsne_perplexity", "tsne_learning_rate",
    "tsne_early_exaggeration", "tsne_max_iter", "pca_whiten",
    "pca_svd_solver", "isomap_n_neighbors", "isomap_path_method",
    "spectral_affinity", "spectral_n_neighbors",
    # The one visible choice instruction 72 adds in front of the other 53.
    "organelle_type",
    # Instruction 71's two opt-in measurements. Both were added to the
    # measure defaults and to NO group, so they fell into the trailing
    # "Other" bucket -- which is not a heading anyone chose, it is the
    # absence of one. They now sit in Measurements beside
    # calculate_correlation, which is what they extend.
    "corrected_manders", "spatial_measurements",
    # The Classify overhaul: a crop source that says where images come from,
    # the on-demand settings it reveals, the path filter that replaced
    # png_type, and real normalisation choices.
    "crop_source", "path_string", "extract_channels", "object_array",
    "coordinate_columns", "crop_shape", "normalization",
    "normalization_scope",
    "balance_to_smallest",
    # The filesystem-facing class folder names are deliberately separate
    # from the semantic `classes` rules and belong beside them in the UI.
    "class_folder_names",
    # The declared channel mapping that replaced png_dims (INVARIANTS 13).
    "png_channel_mapping",
    # The measurement training basis, shared by Classify (CV) and (ML).
    "measurement_rules",
    "cross_validation_enabled",
    "generate_full_dataset",
    "early_stopping_patience",
    "focal_alpha",
    "focal_gamma",
    "label_smoothing",
    "logit_adjust_tau",
    "n_top_examples",
    "random_seed",
    "tar_path",
    "write_random_annotation_column",
    "leakage_audit_train_test",
    "leakage_hash_content",
    "leakage_require_identity",
    # The legacy size-proxy settings remain grouped explicitly even though the
    # visible Replication module now runs the parasites-per-vacuole assay.
    "class_column", "group_by_class", "um_per_px",
    "min_area_bin", "max_area", "max_bins",
    # Count-based replication assay.
    "vacuole_key", "vacuole_link_distance", "vacuole_link_factor",
    "parasite_count_column", "max_parasites_per_vacuole",
    "require_host_cell", "non_power_of_two_warn",
    "batch_fields", "fill_na", "keep_npz", "pipeline_style", "plateID",
    "save_original_images",
    # Landed alongside the regroup: the fail-loud policy (spacr.errors) and
    # the on-demand crop source (spacr.crops).
    "strict_errors", "max_failure_rate", "crop_source",
    # and general UMAP row exclusions, replacing lab-specific c1/c2/c3
    # controls in the settings UI.
    "exclude_rows",
    # and the active-learning queue (spacr.active_learning).
    "queue_by_uncertainty", "queue_measure", "queue_diversity",
    "queue_limit",
    # and opt-in resume (spacr.resume).
    "resume", "resume_checkpoint",
    # and the attribution methods + their analyses (spacr.attribution).
    "smoothgrad_samples", "smoothgrad_sigma", "occlusion_window",
    "occlusion_stride", "ig_steps", "ig_baseline", "attribution_steps",
    "attribution_baseline", "sanity_check", "object_type",
    # and the two-colour invasion assay (submodules.analyze_invasion).
    "parasite_table", "compartment", "outside_channel",
    "total_channel", "intensity_statistic", "background_correction",
    "outside_threshold_method", "outside_threshold", "control_wells",
    "control_quantile", "min_control_objects", "min_objects_for_threshold",
    "min_objects_for_bimodality", "bimodality_cutoff", "threshold_agreement_tolerance",
    "threshold_sensitivity", "inflation_warn", "min_parasites_per_well",
    "min_parasite_area", "max_parasite_area", "min_total_intensity",
    "extracellular_class", "seed_wells_from_cells", "group_column",
    "level", "change_plate", "qc_plot_max_panels",
    # The 3D (Beta) z-axis controls (spacr.zstack), in their own panel.
    "z_stack", "z_segmentation_mode", "z_axis", "z_projection",
    "anisotropy", "voxel_size_z_um", "voxel_size_xy_um", "stitch_threshold",
    # and the 4D (Beta) time-axis controls on top of them (the t half of
    # spacr.zstack), in a separate panel. `z_axis` is shared with the 3D block
    # above and so is not repeated here.
    "t_stack", "t_axis_order", "t_axis", "frame_interval_s",
    "t_track_backend", "t_link_threshold", "t_max_displacement_px",
    "t_max_displacement_um", "t_project_for_tracking",
    # The per-object Cellpose model keys. core.py already read
    # cell_model_name and the Qt synthetic-settings fixture already wrote
    # nucleus_model_name, but no category, defaults block or expected_types
    # entry declared any of them -- so _get_object_settings could only
    # hard-code 'cpsam' and a checkpoint from Train Cellpose had nowhere to
    # be named. Filed with the rest of each object's segmentation knobs.
    "cell_model_name", "nucleus_model_name", "pathogen_model_name",
    # The four metadata_item_* keys set_generate_training_dataset_defaults
    # returns. They name the classes and the wells behind them for the CV
    # dataset builder and had no category at all, so the Classify (CV) dataset
    # settings printed them under "Other" -- four keys away from the
    # class_metadata they belong beside. Filed with "Training Classes".
    "metadata_item_1_name", "metadata_item_1_value",
    "metadata_item_2_name", "metadata_item_2_value",
    # Live PyTorch training telemetry and the Cellpose training resize.
    "tensorboard", "target_size",
    # Static and interactive Image UMAP presentation controls.
    "point_color", "point_alpha", "outline_width",
    "umap_canvas_width", "umap_sidebar_width",
    # Classify out-of-fold evaluation, nested validation and calibration.
    "classifier_evaluation", "nested_cv_inner_folds",
    "evaluation_calibration", "evaluation_bins",
    "evaluation_fail_on_leakage",
    # Shared correction before UMAP, ML screen analysis, and regression.
    "batch_correction", "batch_column", "batch_control_column",
    "batch_control_values", "batch_min_samples", "batch_missing_control",
    # and the two ComBat added afterwards: the covariate whose biology the
    # empirical-Bayes fit must protect, and the mean-only variant.
    "batch_covariate_column", "batch_combat_mean_only",
    # Four keys perform_regression indexes directly that had no default, no
    # expected_types entry and no category, so no panel could offer them and
    # get_perform_regression_default_settings could not produce a dict the
    # function would run on -- regression died on KeyError from Tk, Qt and the
    # CLI alike. (`verbose` and `control_wells`, the other two it was missing,
    # were already categorised under Advanced and Invasion Assay.)
    "score_column", "tolerance", "invert_dependent_variable", "y_lims",
    # -- keys a module merged in through `spacr.settings.register_defaults`
    #
    # These do not come from the regroup at all. `register_defaults(...,
    # categories=...)` folds a module's own categories into the shared map at
    # the moment that module is imported, which is the seam a new module is
    # supposed to use instead of editing the table. They therefore arrive
    # whenever something imports the module, and the count above cannot
    # predict them -- but listing them keeps the "growth is deliberate"
    # contract, because a key here still had to be declared somewhere.
    #
    # Power / Design (`spacr/qt/screens/power.py`), one heading of its own:
    "power_n_genes", "power_n_grnas_per_gene", "power_score_per",
    "power_cells_per_well", "power_wells_per_plate", "power_n_plates",
    "power_constructs_per_well", "power_background_positive_rate",
    "power_effect_fold", "power_hit_rate", "power_reads_per_well",
    "power_n_replicates", "power_detection_auroc", "power_seed",
    "power_backend",
    # AnnData Export (`spacr/anndata_export/__init__.py`):
    "anndata_out", "anndata_single_table", "anndata_nan_policy",
    "anndata_tables", "anndata_dtype", "anndata_row_limit",
    "anndata_compute_umap", "anndata_compression",
    "anndata_register_artifact",
    # The robust and regularised regression fits: knobs that belong to one
    # estimator rather than to all of them.
    "l1_ratio", "quantile", "huber_t",
    "hinge_threshold", "hinge_n_boot",
    "lasso_n_boot", "lasso_selection_threshold",
})

#: Categorised keys with no default and no ``expected_types`` entry. All six
#: are legacy keys kept so old settings CSVs still load (several say so in
#: their own tooltip). Nothing may be added here -- a new entry means a
#: category is advertising a setting that does not exist.
#: Keys that are CATEGORISED but have no default -- ghosts the panel still
#: offers. `highlight` and `offset` left this set on 2026-08-12: they are not
#: ghosts any more, they are RETIRED (see KEYS_RETIRED), which is a different
#: thing. A ghost is still on screen; a retired key is not.
LEGACY_KEYS_WITHOUT_A_DEFAULT = frozenset({
    # `highlight`, `offset`, `nucleus_loc` and `signal_direction` left this
    # set on 2026-08-12. They are not ghosts any more, they are RETIRED (see
    # KEYS_RETIRED), which is a different thing: a ghost is still on screen
    # and still wants a default one day, a retired key is gone. All four were
    # verified uncategorised before being moved.
    "other", "plate",
})

#: ``settings_type`` -> defaults factory, mirroring both dispatchers:
#: ``gui_core.setup_settings_panel`` and
#: ``qt.screens.settings_model.resolve_default_settings``. Imported by name
#: rather than through the Qt module so this file stays headless.
GUI_MODULE_DEFAULTS = {
    "mask": "set_default_settings_preprocess_generate_masks",
    "timelapse": "get_timelapse_settings",
    "motility": "get_automated_motility_assay_default_settings",
    "measure": "get_measure_crop_settings",
    "classify": "deep_spacr_defaults",
    "umap": "set_default_umap_image_settings",
    "train_cellpose": "get_train_cellpose_default_settings",
    "ml_analyze": "set_default_analyze_screen",
    "cellpose_masks": "get_identify_masks_finetune_default_settings",
    "cellpose_all": "get_check_cellpose_models_default_settings",
    "map_barcodes": "set_default_generate_barecode_mapping",
    "regression": "get_perform_regression_default_settings",
    "recruitment": "get_analyze_recruitment_default_settings",
    "activation": "get_default_generate_activation_map_settings",
    "analyze_plaques": "get_analyze_plaque_settings",
}

#: The organelle channel/mask-plane assignments stay in "General" with their
#: cell / nucleus / pathogen siblings. Two of them are also the trigger that
#: reveals the "Organelle" category (see ``category_integer_dependencies``), so
#: moving them into it would hide the controls that reveal it.
ORGANELLE_KEYS_KEPT_IN_GENERAL = frozenset({
    "organelle_channel", "organelle_mask_dim", "organelle_chann_dim",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_categorised_keys():
    """Every key listed in the map, WITH duplicates, in declaration order."""
    return [k for keys in S.categories.values() for k in keys]


def _defaults_for(app_key):
    """Fresh defaults dict for a GUI module, exactly as its panel builds it."""
    return getattr(S, GUI_MODULE_DEFAULTS[app_key])({})


def _every_default_key():
    """Union of every key any ``set_default_*``/``get_*`` factory produces."""
    keys = set()
    for name in dir(S):
        fn = getattr(S, name)
        if not callable(fn):
            continue
        if not name.startswith(("set_default", "set_generate", "set_annotate",
                                "set_graph", "set_interperate", "set_analyze",
                                "get_", "deep_spacr_defaults")):
            continue
        try:
            out = fn({})
        except Exception:
            continue
        if isinstance(out, dict):
            keys |= set(out)
    return keys


def _declared_category_names():
    """Category names as WRITTEN in settings.py, duplicates included.

    A dict literal silently keeps only the last value for a repeated key, so
    reading ``S.categories`` cannot detect a name typed twice -- the source has
    to be parsed.
    """
    src = pathlib.Path(S.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "categories" not in targets or not isinstance(node.value, ast.Dict):
            continue
        return [k.value for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    raise AssertionError("no `categories = {...}` literal found in settings.py")


# ---------------------------------------------------------------------------
# 1. Nothing falls out of the map
# ---------------------------------------------------------------------------

def test_no_previously_categorised_key_is_lost():
    """A key dropping out of the map silently ungroups it in both GUIs.

    Retiring a setting is the one legitimate way out. A key in
    :data:`spacr.settings.DEAD_SETTINGS` is meant to disappear from the
    panel — it stays declared only so an old CSV can be told what to use
    instead — so requiring it to keep a category would force retired
    settings to go on being offered.
    """
    lost = sorted(KEYS_BEFORE_REGROUP
                  - set(_all_categorised_keys())
                  - KEYS_RETIRED)
    assert not lost, (
        f"{len(lost)} setting(s) fell out of spacr.settings.categories and are "
        f"no longer grouped in the settings panel: {lost}. If a key was "
        f"retired on purpose, add it to DEAD_SETTINGS so an old settings CSV "
        f"is told what replaced it."
    )


def test_every_added_key_is_declared():
    """Growth of the map is deliberate, not accidental.

    Containment, not equality. ``register_defaults(..., categories=...)``
    folds a module's own categories into the shared map at the moment that
    module is imported — that is the seam a new module is meant to use
    instead of editing the table — so which of them are present depends on
    what the process has imported by now. Under equality the test failed in
    both directions at once: unimported modules' keys were "listed but
    absent" and imported ones were "present but unlisted", and which it
    reported depended on test ordering.

    The direction worth keeping is the one the failure message describes: a
    key may not appear in the map without being declared here. A listed key
    that this process has not imported costs nothing.
    """
    added = set(_all_categorised_keys()) - KEYS_BEFORE_REGROUP
    undeclared = sorted(
        added - set(KEYS_ADDED_BY_REGROUP) - set(S.DYNAMIC_ORGANELLE_SETTINGS))
    assert not undeclared, (
        "categories gained keys that KEYS_ADDED_BY_REGROUP does not list: "
        f"{undeclared}"
    )


# ---------------------------------------------------------------------------
# 2. One key, one category
# ---------------------------------------------------------------------------

def test_no_key_appears_in_two_categories():
    listed = _all_categorised_keys()
    seen, duplicated = set(), {}
    for cat, keys in S.categories.items():
        for key in keys:
            if key in seen:
                duplicated.setdefault(key, []).append(cat)
            seen.add(key)
    assert not duplicated, (
        f"settings listed under more than one category: {duplicated}. Tk "
        "renders each copy separately and Qt drops all but the first."
    )
    assert len(listed) == len(set(listed))


def test_no_category_is_empty():
    empty = [cat for cat, keys in S.categories.items() if not keys]
    assert not empty, f"empty categories render as blank headings: {empty}"


def test_no_category_name_is_declared_twice():
    declared = _declared_category_names()
    duplicated = sorted({n for n in declared if declared.count(n) > 1})
    assert not duplicated, (
        f"category name(s) typed twice in the dict literal: {duplicated}. The "
        "later entry silently replaces the earlier one and its settings vanish."
    )
    # Categories a module CONTRIBUTED at import are not in the literal and
    # must not be expected there. Power/Design registers "Power analysis"
    # through `register_defaults`, so this comparison was order-dependent:
    # it passed alone and failed after any test that imported that screen.
    # Two more sources of live categories that are not in the literal:
    # modules registering through `register_defaults`, and instruction 73's
    # regroup, which creates its family headings from the keys it moves.
    derived = S.REGISTERED_CATEGORIES | {n for n, _ in S._ADVANCED_FAMILIES}
    assert set(declared) == set(S.categories) - derived


# ---------------------------------------------------------------------------
# 3. Every categorised key is a real setting
# ---------------------------------------------------------------------------

def test_every_categorised_key_is_a_real_setting():
    """A category listing a key nothing defines is a dead entry."""
    known = set(S.expected_types) | _every_default_key()
    ghosts = sorted(set(_all_categorised_keys())
                    - known - LEGACY_KEYS_WITHOUT_A_DEFAULT)
    assert not ghosts, (
        "categories list settings that have neither an expected_types entry "
        f"nor a default: {ghosts}"
    )


def test_the_legacy_ghost_list_has_not_grown():
    known = set(S.expected_types) | _every_default_key()
    still_ghosts = {k for k in LEGACY_KEYS_WITHOUT_A_DEFAULT if k not in known}
    assert still_ghosts <= LEGACY_KEYS_WITHOUT_A_DEFAULT
    assert set(_all_categorised_keys()) >= still_ghosts


# ---------------------------------------------------------------------------
# 4. Every setting a module offers has a category (the reverse direction)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", sorted(GUI_MODULE_DEFAULTS))
def test_every_setting_a_module_offers_has_a_category(app_key):
    """An uncategorised key is ungrouped: pinned to the top in Tk, dumped in
    the trailing "Other" section in Qt. Every module's panel must be fully
    grouped."""
    categorised = set(_all_categorised_keys())
    orphans = sorted(set(_defaults_for(app_key)) - categorised)
    assert not orphans, (
        f"the {app_key!r} settings panel offers ungrouped settings: {orphans}"
    )


# ---------------------------------------------------------------------------
# 5. The organelle settings live under TWO headings, basic and advanced
# ---------------------------------------------------------------------------
# There used to be exactly one, holding FIFTY-THREE settings: the most
# over-configured object class in the tool, and a biologist who knew they were
# imaging lysosomes had to scroll past organelle_ridge_sigmas to find the
# channel. Instruction 72 split it -- six basic, forty-eight advanced.
#
# `test_exactly_one_organelle_category` pinned the old shape and is rewritten,
# not deleted. What it was really protecting is protected below and more
# strictly: every organelle key must be in exactly ONE of the two, and no key
# may fall out of both, which is the failure the original was aimed at.

ORGANELLE_CATEGORIES = ("Organelle", "Organelle advanced")


def test_the_organelle_headings_are_the_two_expected_ones():
    organelle_cats = [c for c in S.categories if "organelle" in c.lower()]
    assert organelle_cats == list(ORGANELLE_CATEGORIES), (
        f"expected {list(ORGANELLE_CATEGORIES)}, found {organelle_cats}"
    )


def test_no_organelle_key_is_in_both_headings():
    basic = set(S.categories["Organelle"])
    advanced = set(S.categories["Organelle advanced"])
    both = sorted(basic & advanced)
    assert not both, f"listed under both headings, so Tk renders it twice: {both}"


def test_the_basic_heading_is_short_enough_to_be_the_point():
    """The deliverable is a NUMBER: 53 settings became 6 visible by default.

    A split that left thirty settings under the first heading would satisfy
    every other test here and none of the request.
    """
    from spacr.object_roles import ORGANELLE_ROLES
    assert len(S.categories["Organelle"]) <= 3 * len(ORGANELLE_ROLES), \
        S.categories["Organelle"]
    assert S.categories["Organelle"], "the basic heading emptied entirely"


def test_the_one_visible_choice_is_in_the_basic_heading():
    assert "organelle_type" in S.categories["Organelle"]


#: Headings instruction 73 pulls the shared families into. An organelle key
#: may legitimately live here instead of under an Organelle heading -- the
#: whole point of that regroup is that `organelle_min_size` and
#: `cell_min_size` are one decision, not two.
ADVANCED_FAMILY_HEADINGS = ("Object filtration", "Intensity handling")


def _organelle_homes():
    homes = set(S.categories["Organelle"]) | set(
        S.categories["Organelle advanced"])
    for heading in ADVANCED_FAMILY_HEADINGS:
        homes |= set(S.categories.get(heading, ()))
    return homes


def test_the_two_headings_hold_every_organelle_key():
    organelle = _organelle_homes()
    stray = sorted(k for k in _all_categorised_keys()
                   if k.startswith("organelle_")
                   and k not in organelle
                   and k not in ORGANELLE_KEYS_KEPT_IN_GENERAL)
    assert not stray, f"organelle settings filed outside every heading: {stray}"
    assert ORGANELLE_KEYS_KEPT_IN_GENERAL <= set(S.categories["General"])


def test_the_headings_cover_every_organelle_default():
    """Every key ``_set_organelle_defaults`` fills is offered somewhere.

    MOVED, NOT HIDDEN. A setting that leaves the panel while staying in the
    settings dict is how a run gets a value nobody can see -- this project
    has eleven phantom settings from exactly that (instruction 61) -- so the
    advanced half being off the first screen must not mean off the panel.
    """
    defaults = set(S._set_organelle_defaults({}))
    from spacr.object_roles import ORGANELLE_ROLES
    general = set(ORGANELLE_KEYS_KEPT_IN_GENERAL) | {
        f'{role}_channel' for role in ORGANELLE_ROLES[1:]}
    missing = sorted(defaults - _organelle_homes() - general)
    assert not missing, f"organelle defaults with no place in the panel: {missing}"


# ---------------------------------------------------------------------------
# 6. Triggers: every conditional category still exists under its own name
# ---------------------------------------------------------------------------

def test_every_dependency_map_names_a_real_category():
    """A trigger pointing at a renamed category silently stops working."""
    referenced = set()
    for cats in S.category_dependencies.values():
        referenced |= set(cats)
    for cats in S.category_integer_dependencies.values():
        referenced |= set(cats)
    for value_map in S.category_value_dependencies.values():
        for cats in value_map.values():
            referenced |= set(cats)
    unknown = sorted(referenced - set(S.categories))
    assert not unknown, (
        f"dependency maps gate categories that do not exist: {unknown}"
    )


def test_the_organelle_trigger_reveals_both_organelle_categories():
    """BOTH, not just the first.

    Splitting the category would otherwise leave "Organelle advanced"
    showing on a run that does no organelle segmentation at all -- the
    trigger has to reveal everything it gates.
    """
    from spacr.object_roles import ORGANELLE_ROLES
    trigger = tuple(key for role in ORGANELLE_ROLES
                    for key in (f"{role}_channel", f"{role}_mask_dim"))
    assert S.category_integer_dependencies[trigger] == [
        "Organelle", "Organelle advanced"]


def test_organelle_method_no_longer_gates_a_category():
    """gui_core blocks the categories of every NON-matching option, so a
    category listed under two options can never be shown. With one merged
    'Organelle' heading there is nothing left for this map to gate."""
    assert S.category_value_dependencies["organelle_method"] == {}


def test_no_category_is_gated_by_two_values_of_the_same_setting():
    """``gui_core._get_visible_categories`` blocks the categories of every
    option that does not equal the current value. A category listed under two
    or more options of the same setting is therefore blocked by whichever one
    is not selected -- i.e. it can never be shown, whatever the user picks.
    This is exactly what hid the old shared 'Organelle' heading."""
    unreachable = {}
    for value_key, value_map in S.category_value_dependencies.items():
        counts = {}
        for cats in value_map.values():
            for cat in cats:
                counts[cat] = counts.get(cat, 0) + 1
        doomed = sorted(c for c, n in counts.items() if n > 1)
        if doomed:
            unreachable[value_key] = doomed
    assert not unreachable, (
        "these categories are gated by more than one value of the same "
        f"setting and can never be shown: {unreachable}"
    )


def test_trigger_settings_live_outside_the_category_they_reveal():
    """Otherwise unticking the trigger hides the control that turns it back on."""
    offenders = []
    for bool_key, cats in S.category_dependencies.items():
        for cat in cats:
            if bool_key in S.categories.get(cat, []):
                offenders.append((bool_key, cat))
    for key_tuple, cats in S.category_integer_dependencies.items():
        for key in key_tuple:
            for cat in cats:
                if key in S.categories.get(cat, []):
                    offenders.append((key, cat))
    # `motility_analysis` is the one known offender: it is the first entry of
    # `motility_settings`, which is also the "Motility (beta)" category, so in
    # Tk it hides itself when off. It is left alone because moving it out of
    # `motility_settings` would also pull it out of
    # settings_model.timelapse_and_motility_keys(), which is what keeps the
    # motility knobs off the Mask panel.
    known = [("motility_analysis", "Motility (beta)")]
    assert sorted(set(offenders) - set(known)) == []


@pytest.mark.parametrize("name", [
    # Referenced as string literals outside settings.py; renaming one without
    # updating its consumer silently breaks the show/hide behaviour.
    "General",                    # gui tests + `timelapse` lives here
    "Cellpose",                   # settings_model._APP_HIDDEN_CATEGORIES['classify']
    "Timelapse",                  # category_dependencies + _APP_HIDDEN_CATEGORIES
    "Motility (beta)",            # category_dependencies + _APP_HIDDEN_CATEGORIES
    "Motility Advanced (beta)",   # category_dependencies + _APP_HIDDEN_CATEGORIES
    "Cell", "Nucleus", "Pathogen", "Organelle",  # category_integer_dependencies
])
def test_category_names_referenced_elsewhere_still_exist(name):
    assert name in S.categories


def test_qt_hidden_category_names_all_resolve():
    """Read the real strings out of the Qt bridge rather than restating them."""
    src = pathlib.Path(S.__file__).parent / "qt" / "screens" / "settings_model.py"
    tree = ast.parse(src.read_text())
    hidden = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_APP_HIDDEN_CATEGORIES":
            for value in node.value.values:
                for elt in value.elts:
                    hidden.add(elt.value)
    assert hidden, "could not read _APP_HIDDEN_CATEGORIES from settings_model.py"
    assert hidden <= set(S.categories), (
        f"Qt hides categories that no longer exist: {sorted(hidden - set(S.categories))}"
    )


def test_timelapse_and_motility_categories_are_the_shared_lists():
    """The Qt bridge derives its key sets from these list objects by identity."""
    assert S.categories["Timelapse"] is S.timelapse_settings
    assert S.categories["Motility (beta)"] is S.motility_settings
    assert S.categories["Motility Advanced (beta)"] is S.motility_advanced_settings


# ---------------------------------------------------------------------------
# 7. Ordering guarantees relied on elsewhere
# ---------------------------------------------------------------------------

def test_paths_is_the_first_category():
    """utils.pretty_print_settings walks the map in order; 'src' first reads
    best, and a test in test_cov_utils_settings_mp.py takes the first entry."""
    assert next(iter(S.categories)) == "Paths"
    assert S.categories["Paths"][0] == "src"


def test_category_keys_list_matches_the_map():
    assert S.category_keys == list(S.categories)


# ---------------------------------------------------------------------------
# 8. The Qt hover hints are keyed on the category names
# ---------------------------------------------------------------------------
#
# qt/screens/app_screen.py renders each category as a Section and looks its
# hover tooltip up in SECTION_HINTS by ``title.upper().strip()``, falling back
# to a generic "Settings that control <title>." when there is no entry. That
# fallback is silent: a renamed, merged or newly-added category loses its
# curated blurb with no error and no test failure. These two tests close the
# loop in both directions.
#
# They were xfail(strict=True) when SECTION_HINTS carried "CROP", "MEASURE"
# and "CLASSIFY" -- names that had never been category names -- and had no
# entry for Cellpose, Measurements or Segmentation QC. It is in line now, and
# they are ordinary passing tests that keep it that way.

def _section_hints():
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import SECTION_HINTS
    return SECTION_HINTS


def test_every_category_has_a_qt_section_hint():
    hints = _section_hints()
    missing = sorted(c for c in S.categories if c.upper().strip() not in hints)
    assert not missing, (
        "these settings sections show the generic fallback tooltip instead of "
        f"a curated one: {missing}"
    )


def test_every_qt_section_hint_names_a_real_category():
    hints = _section_hints()
    known = {c.upper().strip() for c in S.categories}
    # Qt may make app-scoped relocations without changing the category map
    # shared with the legacy UI (Measure's Filter settings is one).
    # Classify is the reason this list has to be exhaustive rather than
    # representative: nine of its ten categories exist only in
    # `_APP_CATEGORY_SPECS`-style Qt regroups and appear nowhere in
    # `S.categories`, so leaving it out reported all nine as dead.
    from spacr.qt.screens.settings_model import categories_for_app
    for app_key in (
        "measure", "external_masks", "map_barcodes", "umap", "ml_analyze", "mask",
        "timelapse", "motility", "regression", "activation", "replication",
        # `classify_merged` was MISSING, and its absence cost two live
        # tooltips: they were deleted as unreachable on 2026-08-12 because
        # no app in this list rendered them. It renders both.
        "classify", "classify_merged", "train_cellpose", "cellpose_masks",
        "analyze_plaques", "recruitment", "invasion",
        # Curated layouts of their own whose headings exist nowhere else.
        # Barcode QC and Illumination register settings that are in no
        # shared category at all, and Power draws its own screen; leaving
        # any of the four out reports all of their blurbs as dead.
            "barcode_qc", "illumination", "anndata_export", "power",
            "explain_cv", "investigate_hit",
    ):
        known.update(
            c.upper().strip()
            for c in categories_for_app(app_key, S.categories)
        )
    dead = sorted(set(hints) - known)
    assert not dead, (
        f"SECTION_HINTS entries that match no settings section: {dead}"
    )


# ---------------------------------------------------------------------------
# 9. The regroup: no module renders an "Other" section, and the settings the
#    user called out are where they belong
# ---------------------------------------------------------------------------
#
# "Other" is what Qt calls the trailing bucket for keys in no category, and
# what utils.pretty_print_settings calls the same leftovers. It is not a
# heading anyone chose -- it is the absence of one. Classify (CV) rendered it
# holding exactly one setting, `custom_model`, because that key was filed
# under "Cellpose" and Classify hides Cellpose.

def _rendered_sections(app_key):
    """(title, keys) per section, exactly as SettingsWidgets.build_sections
    would bucket them -- including the trailing "Other"."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import (
        _APP_HIDDEN_CATEGORIES, _APP_HIDDEN_KEYS, categories_for_app,
        resolve_default_settings,
    )
    defaults = resolve_default_settings(app_key)
    hidden = _APP_HIDDEN_CATEGORIES.get(app_key, set())
    # Hidden KEYS as well as hidden categories. This mirror dropped only the
    # categories, so it reported an "Other" section for every module with a
    # hidden key -- `timelapse`, whose own module forces it True and does not
    # offer it -- while the real screen showed nothing of the kind. Measured:
    # SettingsWidgets("timelapse").build_sections() has no "Other", because a
    # hidden key gets no widget and the trailing section is built from
    # `self._widgets`. See INVARIANTS 6: hidden is not the same as absent,
    # and a mirror that models one and not the other fails on the difference.
    hidden_keys = _APP_HIDDEN_KEYS.get(app_key, frozenset())
    defaults = {k: v for k, v in defaults.items() if k not in hidden_keys}
    used, sections = set(), []
    for name, keys in categories_for_app(app_key, S.categories).items():
        if name in hidden:
            continue
        rows = [k for k in keys if k in defaults and k not in used]
        used.update(rows)
        if rows:
            sections.append((name, rows))
    leftover = [k for k in defaults if k not in used]
    if leftover:
        sections.append(("Other", leftover))
    return sections


@pytest.mark.parametrize(
    ("app_key", "expected"),
    [
        # "Labels & Classes", not "Data & Controls": renamed deliberately to
        # match Classify (CV)'s group of the same purpose. The two modules
        # did the same job under different words, so a settings CSV was not
        # portable between them and neither was a user's understanding.
        ("ml_analyze", [
            "Labels & Classes", "Feature Preparation",
            "Plate & Batch Correction",
            "Classifier & Validation", "Feature Selection & Importance",
            "Output & Database", "Plots & Heatmaps",
            "Runtime & Reliability",
        ]),
        ("mask", [
            "Input & Metadata", "Workflow & Test Run", "Image Preprocessing",
            "Cell Segmentation", "Nucleus Segmentation",
            "Pathogen Segmentation", "Organelle Segmentation",
            "Organelle Segmentation (advanced)",
            "Object Filtration (all objects)",
            "Intensity Handling (all objects)",
            "Quality Control", "Volumetric Processing (Beta)",
            "Time Axes & Tracking (Beta)", "Visualization & Diagnostics",
            "Output & Storage", "Runtime & Reliability",
        ]),
        ("measure", [
            "Input & Experiment", "Mask & Channel Mapping",
            "Measurement Features", "Object Filtering", "Crop Output",
            "Preview & Diagnostics", "3D Calibration (Beta)",
            "Runtime & Reliability",
        ]),
        ("timelapse", [
            "Input & Metadata", "Acquisition & Axes", "Image Preprocessing",
            "Cell Segmentation", "Nucleus Segmentation",
            "Pathogen Segmentation", "Organelle Segmentation",
            "Organelle Segmentation (advanced)",
            "Object Filtration (all objects)",
            "Intensity Handling (all objects)",
            "Quality Control", "Tracking Setup", "Tracking Backends",
            "Visualization & Diagnostics", "Output & Storage",
            "Runtime & Reliability",
        ]),
        ("motility", [
            "Objects & Channels", "Spatial & Temporal Calibration",
            "Motion Filtering", "Infection Classification",
            "XGBoost Infection Model", "Infection Clustering",
            "Embedding Search", "Motility Plots & QC",
            "Runtime & Reliability",
        ]),
        ("regression", [
            "Input Tables", "Controls & Plate Design",
            "Plate & Batch Correction",
            # The response is asked for before the model, and the permutation
            # test's settings are one section instead of being split across
            # the model, the estimator knobs and the hit-calling rules.
            "Response", "Model & Inference",
            # Added when the robust and regularised fits brought knobs that
            # belong to one estimator rather than to all of them. Until they
            # were named, they landed in "Additional Settings" — the bucket
            # this whole test exists to keep empty.
            "Estimator Tuning",
            "Permutation Test", "Significance & Hit Calling",
            "Quality Filters", "Regression Plots",
            "Runtime & Reliability",
        ]),
        ("activation", [
            "Model & Data", "Attribution Method", "Attribution Validation",
            "Map Display", "Map Quantification", "Output & Runtime",
        ]),
        ("replication", [
            "Assay Inputs", "Vacuole Assignment", "Condition Metadata",
            "Object Filtering", "Replication Scoring", "Assay Output",
            "Runtime & Reliability",
        ]),
    ],
)
def test_requested_modules_use_workflow_ordered_categories(app_key, expected):
    """Every module-specific map is ordered and accounts for each key once."""
    sections = _rendered_sections(app_key)
    assert [name for name, _keys in sections] == expected
    rendered_keys = [key for _name, keys in sections for key in keys]
    from spacr.qt.screens.settings_model import (
        _APP_HIDDEN_KEYS, resolve_default_settings,
    )
    assert rendered_keys == list(dict.fromkeys(rendered_keys))
    # Every key is rendered exactly once EXCEPT the deliberately hidden ones.
    # Hidden is not absent (INVARIANTS 6): the key stays in the settings dict
    # at the value the module forces -- `timelapse` is True for the Timelapse
    # module -- and simply gets no control. Comparing against the unfiltered
    # defaults demands that a hidden key be rendered, which is the opposite
    # of what hiding it means.
    expected_keys = set(resolve_default_settings(app_key)).difference(
        _APP_HIDDEN_KEYS.get(app_key, frozenset()))
    assert set(rendered_keys) == expected_keys


@pytest.mark.parametrize("app_key", sorted(GUI_MODULE_DEFAULTS))
def test_no_module_renders_an_other_section(app_key):
    sections = dict(_rendered_sections(app_key))
    assert "Other" not in sections, (
        f"{app_key} shows an ungrouped 'Other' section holding "
        f"{sections['Other']}"
    )


def test_the_model_a_module_runs_is_filed_under_model_training():
    """`custom_model` and `model_name` answer the same question `model_type`
    does. Under "Cellpose" they were invisible to Classify, which hides that
    category, and mis-titled for Train Cellpose, which does not."""
    assert "custom_model" in S.categories["Computer Vision Model"]
    assert "model_name" in S.categories["Computer Vision Model"]
    assert "custom_model" not in S.categories["Cellpose"]
    assert "model_name" not in S.categories["Cellpose"]


def test_the_cv_dataset_class_keys_are_grouped_with_the_dataset():
    """The four metadata_item_* keys name the classes and the wells behind
    them; they printed under "Other", away from class_metadata."""
    training = S.categories["Training Classes"]
    for key in ("metadata_item_1_name", "metadata_item_1_value",
                "metadata_item_2_name", "metadata_item_2_value"):
        assert key in training, key
    assert "class_metadata" in training
    assert "metadata_type_by" in training


def test_dataset_shaping_settings_are_not_filed_as_advanced():
    """normalize scales every image; nuclei_limit / pathogen_limit decide
    which rows exist at all. None of the three is a tuning knob."""
    advanced = set(S.categories["Advanced"])
    assert "normalize" not in advanced
    assert "nuclei_limit" not in advanced
    assert "pathogen_limit" not in advanced
    assert "normalize" in S.categories["General"]
    assert {"nuclei_limit", "pathogen_limit"} <= set(S.categories["Measurements"])


def test_the_replication_module_shows_no_invasion_heading():
    """compartment / group_column / level / change_plate are shared by both
    assays, so filing them under "Invasion Assay" made the Replication panel
    render a heading for an assay it does not run."""
    titles = [t for t, _ in _rendered_sections("replication")]
    assert "Invasion Assay" not in titles, titles


def test_the_measure_module_shows_no_segmentation_headings():
    """The per-object minimum sizes are measurement filters that only
    measure_crop sets; under Cell / Nucleus / Pathogen they gave Measure three
    headings holding one or two size fields and no segmentation."""
    titles = [t for t, _ in _rendered_sections("measure")]
    for gone in ("Cell", "Nucleus", "Pathogen"):
        assert gone not in titles, titles
    measurements = set(S.categories["Measurements"])
    assert {"cell_min_size", "nucleus_min_size", "pathogen_min_size",
            "cytoplasm_min_size", "merge_edge_pathogen_cells"} <= measurements


# ---------------------------------------------------------------------------
# 12. The regroup is presentation only (instruction 73, item 3)
# ---------------------------------------------------------------------------
# "Moving a key between GUI categories must not change its name or its
# meaning -- the category is presentation. A test should assert that the set
# of keys a module offers is unchanged by the regroup, because that is the
# failure that would silently drop a setting from a run."

def test_the_regroup_does_not_change_which_keys_a_module_offers():
    """Every key a module offers is still categorised somewhere.

    This is the failure worth excluding: a settings CSV names KEYS, not
    headings, so a file written before the regroup must load and mean
    exactly what it meant. A key that fell out of every category during the
    move would be silently dropped from the panel while remaining in the
    settings dict -- the phantom-setting failure mode this project already
    has eleven of.
    """
    categorised = set(_all_categorised_keys())
    for app_key in GUI_MODULE_DEFAULTS:
        offered = set(_defaults_for(app_key))
        lost = sorted(offered - categorised - ORGANELLE_KEYS_KEPT_IN_GENERAL)
        assert not lost, f"{app_key!r} offers uncategorised settings: {lost}"


def test_the_regrouped_families_hold_only_keys_that_existed_before():
    """The regroup MOVES keys; it must not invent them."""
    known = set(S.expected_types) | _every_default_key()
    for heading in ADVANCED_FAMILY_HEADINGS:
        for key in S.categories.get(heading, ()):
            assert key in known, (heading, key)


def test_a_family_heading_groups_by_object_so_it_reads_as_one_decision():
    """`cell_min_size` and `nucleus_min_size` are one decision applied twice.

    Ordering by object is what makes that visible in a FLAT panel, which is
    the only kind this settings screen has -- `build_sections` returns one
    header and its rows, with no third level to nest a per-object
    sub-section under.
    """
    members = S.categories["Object filtration"]
    seen_objects = []
    for key in members:
        obj = key.split("_", 1)[0]
        if obj not in seen_objects:
            seen_objects.append(obj)
    # Each object's keys must be contiguous: an object may not reappear
    # after another one has started.
    order = [key.split("_", 1)[0] for key in members]
    collapsed = [o for i, o in enumerate(order) if i == 0 or order[i - 1] != o]
    assert collapsed == seen_objects, collapsed


def test_the_per_object_headings_actually_shrank():
    """The deliverable is a NUMBER, and it is recorded in instruction 73."""
    assert len(S.categories["Cell"]) <= 10
    assert len(S.categories["Nucleus"]) <= 10
    assert len(S.categories["Pathogen"]) <= 10


def test_measurements_keeps_the_sizes_an_earlier_decision_gave_it():
    """Not everything shared should move.

    The per-object minimum sizes are measurement filters that only
    measure_crop sets. Pulling them into a shared filtration heading would
    put Measure's three near-empty segmentation headings back by another
    route, which is exactly what filing them under Measurements fixed.
    """
    measurements = set(S.categories["Measurements"])
    assert {"cell_min_size", "nucleus_min_size", "pathogen_min_size",
            "cytoplasm_min_size"} <= measurements
    for heading in ADVANCED_FAMILY_HEADINGS:
        assert not (set(S.categories.get(heading, ())) & measurements), heading
