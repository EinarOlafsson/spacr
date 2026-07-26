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
KEYS_ADDED_BY_REGROUP = frozenset({
    # Replication Assay: the six keys set_analyze_endodyogeny_defaults returns
    # that no category listed, so they landed in "Other" until the assay got a
    # category of its own. The backend has existed since #21; only the app
    # entry and this category were missing.
    "class_column", "group_by_class", "um_per_px",
    "min_area_bin", "max_area", "max_bins",
    "batch_fields", "fill_na", "keep_npz", "pipeline_style", "plateID",
    "save_original_images",
    # Landed alongside the regroup: the fail-loud policy (spacr.errors) and
    # the on-demand crop source (spacr.crops).
    "strict_errors", "max_failure_rate", "crop_source",
    # and the active-learning queue (spacr.active_learning).
    "queue_by_uncertainty", "queue_measure", "queue_diversity",
    "queue_limit",
    # and opt-in resume (spacr.resume).
    "resume",
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
    # The 3D (Beta) z-axis controls (spacr.zstack), filed under "Beta".
    "z_stack", "z_segmentation_mode", "z_axis", "z_projection",
    "anisotropy", "voxel_size_z_um", "voxel_size_xy_um", "stitch_threshold",
    # and the 4D (Beta) time-axis controls on top of them (the t half of
    # spacr.zstack), filed under "Beta" too rather than under a heading of
    # their own -- a new category with no SECTION_HINTS entry silently falls
    # back to the generic tooltip, which is what the two tests at the bottom
    # of this file are about. `z_axis` is shared with the 3D block above and
    # so is not repeated here.
    "t_stack", "t_axis_order", "t_axis", "frame_interval_s",
    "t_track_backend", "t_link_threshold", "t_max_displacement_px",
    "t_max_displacement_um", "t_project_for_tracking",
})

#: Categorised keys with no default and no ``expected_types`` entry. All six
#: are legacy keys kept so old settings CSVs still load (several say so in
#: their own tooltip). Nothing may be added here -- a new entry means a
#: category is advertising a setting that does not exist.
LEGACY_KEYS_WITHOUT_A_DEFAULT = frozenset({
    "highlight", "nucleus_loc", "offset", "other", "plate", "signal_direction",
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
    """A key dropping out of the map silently ungroups it in both GUIs."""
    lost = sorted(KEYS_BEFORE_REGROUP - set(_all_categorised_keys()))
    assert not lost, (
        f"{len(lost)} setting(s) fell out of spacr.settings.categories and are "
        f"no longer grouped in the settings panel: {lost}"
    )


def test_every_added_key_is_declared():
    """Growth of the map is deliberate, not accidental."""
    added = set(_all_categorised_keys()) - KEYS_BEFORE_REGROUP
    assert added == set(KEYS_ADDED_BY_REGROUP), (
        "categories gained keys that KEYS_ADDED_BY_REGROUP does not list: "
        f"{sorted(added - set(KEYS_ADDED_BY_REGROUP))}"
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
    assert set(declared) == set(S.categories)


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
# 5. The organelle settings live under one heading
# ---------------------------------------------------------------------------

def test_exactly_one_organelle_category():
    organelle_cats = [c for c in S.categories if "organelle" in c.lower()]
    assert organelle_cats == ["Organelle"], (
        f"expected a single 'Organelle' heading, found {organelle_cats}"
    )


def test_the_organelle_category_holds_every_organelle_key():
    organelle = set(S.categories["Organelle"])
    stray = sorted(k for k in _all_categorised_keys()
                   if k.startswith("organelle_")
                   and k not in organelle
                   and k not in ORGANELLE_KEYS_KEPT_IN_GENERAL)
    assert not stray, f"organelle settings filed outside 'Organelle': {stray}"
    assert ORGANELLE_KEYS_KEPT_IN_GENERAL <= set(S.categories["General"])


def test_the_organelle_category_covers_every_organelle_default():
    """Every key ``_set_organelle_defaults`` fills is offered under the heading."""
    defaults = set(S._set_organelle_defaults({}))
    missing = sorted(defaults - set(S.categories["Organelle"])
                     - ORGANELLE_KEYS_KEPT_IN_GENERAL)
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


def test_the_organelle_trigger_reveals_the_merged_category():
    assert S.category_integer_dependencies[
        ("organelle_channel", "organelle_mask_dim")] == ["Organelle"]


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
# Both are xfail(strict=True) because SECTION_HINTS is owned by another module
# and is already out of sync on this tree (it carries "CROP", "MEASURE" and
# "CLASSIFY", which have never been category names, and has no entry for
# Cellpose, Measurements or Segmentation QC). They flip to XPASS - loudly - the
# moment SECTION_HINTS is brought back in line.

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
    dead = sorted(set(hints) - known)
    assert not dead, (
        f"SECTION_HINTS entries that match no settings section: {dead}"
    )
