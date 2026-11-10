"""Types, categories and help for every optical-pooled-screening setting.

Each setting is registered with its type, tooltip and category when it is
added. A setting with no type cannot be validated, a setting with no tooltip
has no help text, and a setting in no category does not appear in the panel.

Tooltips describe the setting's effect on the result rather than repeating
its name. For example, "The number of features" repeats the label, whereas
"More features find more overlap and cost time and memory; below about 2000
a sparse field stops matching" states the effect of raising or lowering it.
"""
from __future__ import annotations

from typing import Dict, List

#: The default of every setting :func:`spacr.ops_engine.run_ops` reads, and
#: of nothing else. A key the engine does not read has no place in its panel:
#: it would be a control that changes nothing.
#:
#: THE FOUR MEASURED NUMBERS REPEAT A CONSTANT IN :mod:`spacr.ops_engine`,
#: which is where each is recorded with the measurement that produced it.
#: They are repeated rather than imported because importing the engine here
#: would pull numpy, scipy and the whole OPS stack into every settings
#: registration, including the GUI's; the two copies are held equal by
#: tests/test_every_ops_setting_is_declared.py.
OPS_DEFAULTS: Dict[str, object] = {
    "genotype_source": None,
    "phenotype_source": None,
    "dst_root": None,
    "plate": "",
    "ops_gpu": True,
    "n_workers": 26,
    "cellpose_model": "cpsam",
    "cellpose_diameter": None,
    "ops_library": None,
    "ops_base_channels": "CY3,A594,CY5,CY7",
    "ops_read_threshold": 315.0,
    "ops_raster_overlap": 213,
    "ops_window_overlap": 96,
    "ops_footprint": 10.0,
    "ops_store_reads": False,
    "ops_spot_detector": "native",
    "ops_cell_rule": "mapped",
    "ops_crosstalk": "argmax_median",
    "ops_channel_registration": "spots",
    "ops_read_mismatches": 0,
    "ops_min_reads": None,
    "ops_min_fraction": 0.6,
    "ops_single_read_quality": 0.03,
    "ops_ph_channels": "DAPI,GFP,A594,AF750",
    "ops_ph_cell_channel": "AF750",
    "ops_ph_nucleus_model": "cpsam",
    "ops_ph_cell_model": "cpsam",
    "ops_ph_nucleus_diameter": None,
    "ops_ph_cell_diameter": None,
    "ops_ph_downsample": 2,
    "ops_ph_batch_size": 8,
    "ops_ph_sites": None,
    "ops_ph_texture": True,
    "ops_merge_threshold": 4.0,
    "ops_design": None,
    "ops_control_prefix": "nontargeting",
    "ops_bootstrap": 0,
    "ops_pvalue_null": "calibrated",
    "ops_test_features": None,
    "ops_mitotic_dna_z": 3.0,
    "ops_genotype_table": None,
    "ops_pool_wells": None,
    "ops_pool_interphase": False,
    "ops_feature_qc": True,
    "ops_ownership": "closer_match",
    "ops_flatfield": "none",
    "ops_flatfield_maps": None,
    "ops_flatfield_fields": 204,
    "ops_flatfield_field_list": None,
    "ops_flatfield_nuclear_channel": None,
    "ops_flatfield_nuclei": None,
    "ops_flatfield_dark": None,
    "ops_flatfield_nuclear_maps": None,
    "ops_flatfield_block_px": 16,
    "ops_flatfield_fit": "local_quadratic_log",
    "ops_flatfield_sigma_px": 32.0,
    "ops_flatfield_ridge": None,
    "ops_flatfield_floor": 0.10,
    "ops_flatfield_centre_cells": 18,
    "ops_flatfield_lookup": "block_centre",
    "ops_flatfield_map_grid": "block_centre",
    "ops_flatfield_centre_offset": 8.0,
    "ops_flatfield_field_px": None,
    "ops_flatfield_holdout_fields": None,
    "ops_flatfield_holdout_n": 24,
    "ops_flatfield_seed": 0,
    "ops_flatfield_holdout_images": None,
    "ops_flatfield_holdout_masks": None,
    "ops_flatfield_compare_maps": None,
    "ops_flatfield_radial_edges": [0, 500, 1000, 1400, 1600, 1800, 1900, 2000, 2100],
    "ops_flatfield_bootstrap": 2000,
    "ops_flatfield_bootstrap_seed": 20261109,
    "ops_flatfield_min_bin_cells": 100,
    "ops_ph_funk_features": False,
    "ops_ph_zernike": False,
    "ops_drop_failed_fields": False,
    "ops_drop_nonfinite_rows": False,
    "ops_profile_normalisation": "robust_z",
    "ops_tvn_mode": "streamed",
    "ops_tvn_pca_sample": 100000,
    "ops_tvn_seed": 0,
    "ops_tvn_components": None,
    "ops_tvn_whiten_epsilon": 3.0,
    "ops_cluster_benchmark": False,
    "ops_benchmark_corum": None,
    "ops_benchmark_kegg": None,
    "ops_benchmark_string": None,
    "ops_reference_clusters": None,
    "ops_cluster_target": 251,
    "ops_cluster_resolutions": "5,10,12,15,16,17,18,20,22",
    "ops_cluster_seed": 42,
    "ops_consensus_runs": 0,
    "ops_mitotic_model": None,
    "ops_merged_root": None,
    "ops_merged_flatfield": True,
    "ops_merged_dtype": 'uint16',
    "ops_merged_save_sbs": False,
    "ops_tables_parquet_threshold": 5000000,
    "ops_crop_per_guide": None,
    "ops_crop_size": 96,
    "ops_crop_channels": None,
    "ops_crop_flatfield": True,
    "ops_crop_seed": 0,
    "ops_crop_all_nt": True,
    "ops_crop_control_fraction": 0.01,
    "ops_crop_masks": False,
    "ops_max_files_per_folder": 10000,
    "ops_ph_secondary_channels": None,
    "ops_ph_secondary_registration": 'none',
    "ops_ph_secondary_window": 1024,
    "ops_ph_secondary_fallback": 'well_median',
    "ops_ph_measure_gpu": False,
    "ops_cross_field_ownership": False,
    "ops_cross_field_metric": 'overlap_fraction',
    "ops_cross_field_min_overlap": 0.5,
    "ops_cross_field_policy": 'one_per_cell',
    "ops_reference_counts": None,
    "ops_reference_counts_level": 'sbs',
    "ops_reference_shuffles": 1000,
    "ops_reference_seed": 0,
    "ops_retry_missing": False,
    "ops_crop_flat_mean_one": False,
    "ops_decode_preset": "none",
    "ops_registration_refine_px": 3,
    "ops_spot_min_peak": 20.0,
    "ops_spot_overrule_share": 0.5,
    "ops_decode_library_calibration": False,
    "ops_calibration_passes": 2,
    "ops_calibration_min_labels": 30,
    "ops_decode_runner_up": False,
    "ops_runner_up_margin": 0.1,
    "ops_decode_impute_missing_cycles": True,
    "ops_assign_by_cell_territory": False,
    "ops_tie": 1.0,
    "ops_decode_quality_weighted": False,
    "ops_vote_weight_floor": 0.05,
    "ops_decode_pixel_fallback": False,
    "ops_pixel_percentile": 97.0,
    "ops_pixel_radius": 10.0,
    "ops_pixel_min_pixels": 2,
    "ops_pixel_min_fraction": 0.6,
    "ops_pixel_quality": 0.1,
    "ops_decoy_library_k": 0,
    "ops_fdr_scale": 1.268,
    "ops_decode_skip_unplaced_sites": True,
    "ops_store_field_cache": False,
    "ops_decode_qc": False,
    "ops_decode_qc_model": "xgboost",
    "ops_decode_qc_features": None,
    "ops_decode_qc_split": "field",
    "ops_decode_qc_seed": 0,
    "ops_decode_qc_max_calls": 20000,
    "ops_genotype_confidence": False,
    "ops_confidence_model": "hist_gbm",
    "ops_confidence_calibration": "target_decoy",
    "ops_confidence_bins": 10,
    "ops_confidence_seed": 20261107,
    "ops_confidence_min": None,
    "ops_compare_tables": None,
    "ops_compare_match_px": 4.0,
    "ops_compare_fields": None,
    "ops_aggregate_level": "guides",
    "ops_nt_rows": None,
    "ops_profile_weighting": None,
    "ops_guide_efficacy": None,
    "ops_guide_efficacy_iters": 20,
    "ops_guide_efficacy_emax": 1.0,
    "ops_profile_cell_cycle": None,
    "ops_profile_g1_sg2_split_log2": 0.3,
    "ops_mitotic_source": "rule",
    "ops_penetrance": False,
    "ops_penetrance_folds": 5,
    "ops_penetrance_nt_pool": 20000,
    "ops_penetrance_lambda": 0.5,
    "ops_escapers": False,
    "ops_escaper_dbic": 10.0,
    "ops_escaper_min_sep": 2.0,
    "ops_phase_tests": False,
    "ops_phase_null": "targeting",
    "ops_profile_shrinkage": None,
    "ops_profile_shrinkage_prior_df": 4.0,
    "ops_profile_generic_axes": None,
    "ops_profile_essential_axis_min_rho": 0.3,
    "ops_profile_similarity": None,
    "ops_guide_null_size": 200000,
    "ops_profile_responder_alpha": 0.05,
    "ops_profile_space": None,
    "ops_fusion_blocks": None,
    "ops_profile_cells": None,
    "ops_profile_wells": None,
    "ops_profile_run": "",
    "ops_profile_recipe": None,
    "ops_profile_min_cells": 10,
    "ops_profile_active_filter": None,
    "ops_profile_cell_state": None,
    "ops_profile_gene_universe": None,
    "ops_mitotic_index_test": False,
    "ops_mitotic_index_min_cells": 50,
    "ops_mitotic_index_reference": None,
    "ops_reference_dir": None,
    "ops_scoring_inputs": None,
    "ops_scoring_genes": None,
    "ops_scoring_null_size": 10000,
    "ops_scoring_min_complex": 3,
    "ops_scoring_references": ["CORUM@REPORT", "STRING>=0.7@REPORT"],
    "ops_scoring_cluster_jitter": 1e-3,
    "ops_scoring_cluster_seeds": [0, 1, 2],
    "ops_scoring_workers": 4,
    "ops_scoring_figures": True,
    "ops_eval_references": ["CORUM", "STRING>=0.7", "STRING>=0.4", "KEGG", "BioGRID"],
    "ops_eval_split": "all",
    "ops_eval_null_size": 5000,
    "ops_eval_heldout_folds": 5,
    "ops_eval_decoys": 10,
    "ops_eval_time_split": False,
    "ops_eval_covariates": None,
    "ops_deterministic_blas": True,
    "ops_ph_acquisitions": None,
    "ops_ph_pipeline": False,
    "ops_ph_measure_workers": None,
    "ops_ph_stage_dir": None,
    "ops_ph_stage_ahead": 48,
    "ops_ph_stage_threads": 4,
    "ops_ph_measure_gpu_check": True,
    "ops_archive_root": None,
    "ops_archive_verify": 'md5',
    "ops_archive_delete_local": False,
    "ops_gpu_admission": False,
    "ops_gpu_idle_checks": 6,
    "ops_gpu_min_process_mib": 500,
    "ops_gpu_vram_cap_gb": None,
    "ops_gpu_ignore_process_patterns": None,
    "ops_tables_integer_keys": False,
    "ops_tables_layout": 'compartments',
    "ops_stress_report": False,
    "ops_modules_runs": 100,
    "ops_modules_precomputed_embedding": False,
    "ops_modules_profiles": None,
    "ops_modules_driver_profiles": None,
    "ops_eval_mitocarta": None,
    "ops_eval_external_modules": None,
    "ops_eval_arms": None,
    "ops_report_read_stage": False,
    "ops_report_overlap_zones": False,
    "ops_report_edge_bin_px": 100,
    "ops_report_external_cells": None,
    "ops_report_external_max_px": 8.0,
    "ops_cluster_rows": None,
    "ops_cluster_graph": ['phate', 'knn'],
    "ops_cluster_knn": 12,
    "ops_cluster_sweep_resolutions": None,
    "ops_cluster_seeds": [42, 1, 2],
    "ops_cluster_target_clusters": 227,
    "ops_cluster_shuffles": 1,
    "ops_cluster_reference_dir": None,
    "ops_cluster_gene_list": None,
    "ops_etest": False,
    "ops_etest_cells": None,
    "ops_etest_metric": 'both',
    "ops_etest_reference_cells": 2000,
    "ops_etest_null": 1000,
    "ops_etest_holdout": 500,
    "ops_etest_mode": 'empirical_null',
    "ops_etest_permutations": 1000,
    "ops_power_cells": [10, 25, 50, 100, 200, 500, None],
    "ops_power_seeds": [0, 1, 2],
    "ops_power_retrieval": False,
}

#: The type each setting may hold, for :func:`spacr.settings.check_settings`.
#:
#: ``None`` is admissible for every path-like and every "work it out" default,
#: which is why those carry a tuple including ``type(None)``.
OPS_TYPES: Dict[str, object] = {
    "dst_root": (str, type(None)),
    "genotype_source": (str, type(None)),
    "phenotype_source": (str, type(None)),
    "plate": str,
    "cellpose_model": str,
    "cellpose_diameter": (float, int, type(None)),
    "n_workers": int,
    "ops_gpu": bool,
    "ops_library": (str, type(None)),
    "ops_base_channels": str,
    "ops_read_threshold": (float, int),
    "ops_raster_overlap": int,
    "ops_window_overlap": int,
    "ops_footprint": (float, int),
    "ops_store_reads": bool,
    "ops_spot_detector": str,
    "ops_cell_rule": str,
    "ops_crosstalk": str,
    "ops_channel_registration": str,
    "ops_read_mismatches": int,
    "ops_min_reads": (int, type(None)),
    "ops_min_fraction": (float, int),
    "ops_single_read_quality": (float, int),
    "ops_ph_channels": str,
    "ops_ph_cell_channel": str,
    "ops_ph_nucleus_model": str,
    "ops_ph_cell_model": str,
    "ops_ph_nucleus_diameter": (float, int, type(None)),
    "ops_ph_cell_diameter": (float, int, type(None)),
    "ops_ph_downsample": int,
    "ops_ph_batch_size": int,
    "ops_ph_sites": (str, type(None)),
    "ops_ph_texture": bool,
    "ops_merge_threshold": (float, int),
    "ops_design": (str, type(None)),
    "ops_control_prefix": str,
    "ops_bootstrap": int,
    "ops_pvalue_null": str,
    "ops_test_features": (str, type(None)),
    "ops_mitotic_dna_z": (float, int),
    "ops_genotype_table": (str, type(None)),
    "ops_pool_wells": (str, type(None)),
    "ops_pool_interphase": bool,
    "ops_feature_qc": bool,
    "ops_ownership": str,
    "ops_flatfield": str,
    "ops_flatfield_maps": (str, type(None)),
    "ops_flatfield_fields": int,
    "ops_flatfield_field_list": (str, type(None)),
    "ops_flatfield_nuclear_channel": (str, type(None)),
    "ops_flatfield_nuclei": (str, dict, type(None)),
    "ops_flatfield_dark": (int, float, dict, type(None)),
    "ops_flatfield_nuclear_maps": (str, type(None)),
    "ops_flatfield_block_px": int,
    "ops_flatfield_fit": str,
    "ops_flatfield_sigma_px": float,
    "ops_flatfield_ridge": (float, type(None)),
    "ops_flatfield_floor": float,
    "ops_flatfield_centre_cells": int,
    "ops_flatfield_lookup": str,
    "ops_flatfield_map_grid": str,
    "ops_flatfield_centre_offset": float,
    "ops_flatfield_field_px": (int, type(None)),
    "ops_flatfield_holdout_fields": (str, list, dict, type(None)),
    "ops_flatfield_holdout_n": int,
    "ops_flatfield_seed": int,
    "ops_flatfield_holdout_images": (str, dict, type(None)),
    "ops_flatfield_holdout_masks": (str, dict, type(None)),
    "ops_flatfield_compare_maps": (dict, type(None)),
    "ops_flatfield_radial_edges": list,
    "ops_flatfield_bootstrap": int,
    "ops_flatfield_bootstrap_seed": int,
    "ops_flatfield_min_bin_cells": int,
    "ops_ph_funk_features": bool,
    "ops_ph_zernike": bool,
    "ops_drop_failed_fields": bool,
    "ops_drop_nonfinite_rows": bool,
    "ops_profile_normalisation": str,
    "ops_tvn_mode": str,
    "ops_tvn_pca_sample": int,
    "ops_tvn_seed": int,
    "ops_tvn_components": (float, int, type(None)),
    "ops_tvn_whiten_epsilon": (float, int),
    "ops_cluster_benchmark": bool,
    "ops_benchmark_corum": (str, type(None)),
    "ops_benchmark_kegg": (str, type(None)),
    "ops_benchmark_string": (str, type(None)),
    "ops_reference_clusters": (str, type(None)),
    "ops_cluster_target": int,
    "ops_cluster_resolutions": str,
    "ops_cluster_seed": int,
    "ops_consensus_runs": int,
    "ops_mitotic_model": (str, type(None)),
    "ops_merged_root": (str, type(None)),
    "ops_merged_flatfield": bool,
    "ops_merged_dtype": str,
    "ops_merged_save_sbs": bool,
    "ops_tables_parquet_threshold": int,
    "ops_crop_per_guide": (int, type(None)),
    "ops_crop_size": int,
    "ops_crop_channels": (str, type(None)),
    "ops_crop_flatfield": bool,
    "ops_crop_seed": int,
    "ops_crop_all_nt": bool,
    "ops_crop_control_fraction": (float, int),
    "ops_crop_masks": bool,
    "ops_max_files_per_folder": int,
    "ops_ph_secondary_channels": (str, type(None)),
    "ops_ph_secondary_registration": str,
    "ops_ph_secondary_window": int,
    "ops_ph_secondary_fallback": str,
    "ops_ph_measure_gpu": bool,
    "ops_cross_field_ownership": bool,
    "ops_cross_field_metric": str,
    "ops_cross_field_min_overlap": (float, int),
    "ops_cross_field_policy": str,
    "ops_reference_counts": (str, type(None)),
    "ops_reference_counts_level": str,
    "ops_reference_shuffles": int,
    "ops_reference_seed": int,
    "ops_retry_missing": bool,
    "ops_crop_flat_mean_one": bool,
    "ops_decode_preset": str,
    "ops_registration_refine_px": int,
    "ops_spot_min_peak": (float, int),
    "ops_spot_overrule_share": (float, int),
    "ops_decode_library_calibration": bool,
    "ops_calibration_passes": int,
    "ops_calibration_min_labels": int,
    "ops_decode_runner_up": bool,
    "ops_runner_up_margin": (float, int),
    "ops_decode_impute_missing_cycles": bool,
    "ops_assign_by_cell_territory": bool,
    "ops_tie": (float, int),
    "ops_decode_quality_weighted": bool,
    "ops_vote_weight_floor": (float, int),
    "ops_decode_pixel_fallback": bool,
    "ops_pixel_percentile": (float, int),
    "ops_pixel_radius": (float, int),
    "ops_pixel_min_pixels": int,
    "ops_pixel_min_fraction": (float, int),
    "ops_pixel_quality": (float, int),
    "ops_decoy_library_k": int,
    "ops_fdr_scale": (float, int),
    "ops_decode_skip_unplaced_sites": bool,
    "ops_store_field_cache": bool,
    "ops_decode_qc": bool,
    "ops_decode_qc_model": str,
    "ops_decode_qc_features": (str, type(None)),
    "ops_decode_qc_split": str,
    "ops_decode_qc_seed": int,
    "ops_decode_qc_max_calls": int,
    "ops_genotype_confidence": bool,
    "ops_confidence_model": str,
    "ops_confidence_calibration": str,
    "ops_confidence_bins": int,
    "ops_confidence_seed": int,
    "ops_confidence_min": (float, int, type(None)),
    "ops_compare_tables": (str, dict, type(None)),
    "ops_compare_match_px": (float, int),
    "ops_compare_fields": (str, list, type(None)),
    "ops_aggregate_level": str,
    "ops_nt_rows": (str, type(None)),
    "ops_profile_weighting": (str, type(None)),
    "ops_guide_efficacy": (str, type(None)),
    "ops_guide_efficacy_iters": int,
    "ops_guide_efficacy_emax": (float, int),
    "ops_profile_cell_cycle": (str, type(None)),
    "ops_profile_g1_sg2_split_log2": (float, int),
    "ops_mitotic_source": str,
    "ops_penetrance": bool,
    "ops_penetrance_folds": int,
    "ops_penetrance_nt_pool": int,
    "ops_penetrance_lambda": (float, int),
    "ops_escapers": bool,
    "ops_escaper_dbic": (float, int),
    "ops_escaper_min_sep": (float, int),
    "ops_phase_tests": bool,
    "ops_phase_null": str,
    "ops_profile_shrinkage": (str, type(None)),
    "ops_profile_shrinkage_prior_df": (float, int),
    "ops_profile_generic_axes": (str, list, type(None)),
    "ops_profile_essential_axis_min_rho": (float, int),
    "ops_profile_similarity": (str, type(None)),
    "ops_guide_null_size": int,
    "ops_profile_responder_alpha": (float, int),
    "ops_profile_space": (str, type(None)),
    "ops_fusion_blocks": (str, list, type(None)),
    "ops_profile_cells": (str, dict, type(None)),
    "ops_profile_wells": (list, str, type(None)),
    "ops_profile_run": str,
    "ops_profile_recipe": (dict, type(None)),
    "ops_profile_min_cells": int,
    "ops_profile_active_filter": (dict, type(None)),
    "ops_profile_cell_state": (str, list, type(None)),
    "ops_profile_gene_universe": (str, list, type(None)),
    "ops_mitotic_index_test": bool,
    "ops_mitotic_index_min_cells": int,
    "ops_mitotic_index_reference": (str, type(None)),
    "ops_reference_dir": (str, list, type(None)),
    "ops_scoring_inputs": (dict, type(None)),
    "ops_scoring_genes": (str, list, type(None)),
    "ops_scoring_null_size": int,
    "ops_scoring_min_complex": int,
    "ops_scoring_references": list,
    "ops_scoring_cluster_jitter": float,
    "ops_scoring_cluster_seeds": list,
    "ops_scoring_workers": int,
    "ops_scoring_figures": bool,
    "ops_eval_references": (list, str),
    "ops_eval_split": str,
    "ops_eval_null_size": int,
    "ops_eval_heldout_folds": int,
    "ops_eval_decoys": int,
    "ops_eval_time_split": bool,
    "ops_eval_covariates": (str, type(None)),
    "ops_deterministic_blas": bool,
    "ops_ph_acquisitions": (dict, str, type(None)),
    "ops_ph_pipeline": bool,
    "ops_ph_measure_workers": (int, type(None)),
    "ops_ph_stage_dir": (str, type(None)),
    "ops_ph_stage_ahead": int,
    "ops_ph_stage_threads": int,
    "ops_ph_measure_gpu_check": bool,
    "ops_archive_root": (str, type(None)),
    "ops_archive_verify": str,
    "ops_archive_delete_local": bool,
    "ops_gpu_admission": bool,
    "ops_gpu_idle_checks": int,
    "ops_gpu_min_process_mib": (int, float),
    "ops_gpu_vram_cap_gb": (int, float, type(None)),
    "ops_gpu_ignore_process_patterns": (str, type(None)),
    "ops_tables_integer_keys": bool,
    "ops_tables_layout": str,
    "ops_stress_report": bool,
    "ops_modules_runs": int,
    "ops_modules_precomputed_embedding": bool,
    "ops_modules_profiles": (str, type(None)),
    "ops_modules_driver_profiles": (str, list, type(None)),
    "ops_eval_mitocarta": (str, type(None)),
    "ops_eval_external_modules": (str, type(None)),
    "ops_eval_arms": (str, list, type(None)),
    "ops_report_read_stage": bool,
    "ops_report_overlap_zones": bool,
    "ops_report_edge_bin_px": (int, float),
    "ops_report_external_cells": (str, type(None)),
    "ops_report_external_max_px": (float, int),
    "ops_cluster_rows": (str, type(None)),
    "ops_cluster_graph": (list, str),
    "ops_cluster_knn": int,
    "ops_cluster_sweep_resolutions": (dict, type(None)),
    "ops_cluster_seeds": (list, str),
    "ops_cluster_target_clusters": int,
    "ops_cluster_shuffles": int,
    "ops_cluster_reference_dir": (str, list, type(None)),
    "ops_cluster_gene_list": (str, type(None)),
    "ops_etest": bool,
    "ops_etest_cells": (str, type(None)),
    "ops_etest_metric": str,
    "ops_etest_reference_cells": int,
    "ops_etest_null": int,
    "ops_etest_holdout": int,
    "ops_etest_mode": str,
    "ops_etest_permutations": int,
    "ops_power_cells": (list, str),
    "ops_power_seeds": (list, str),
    "ops_power_retrieval": bool,
}

#: Which panel section each setting appears under.
#:
#: A SETTING BELONGS TO EXACTLY ONE CATEGORY. `plate` is shared with the rest
#: of spaCR and is already filed under Plate Layout & Controls, so it is NOT
#: repeated here. Listing one twice is not a display preference: Tk renders
#: each copy separately and Qt drops all but the first, so the second copy is
#: either a duplicate control or an invisible one depending on which toolkit
#: is drawing.
OPS_CATEGORIES: Dict[str, List[str]] = {
    "OPS input": [
        "dst_root", "genotype_source", "phenotype_source", "ops_library",
    ],
    "OPS alignment": [
        "cellpose_model", "cellpose_diameter", "ops_raster_overlap",
        "ops_window_overlap",
    ],
    "OPS decoding": [
        "ops_base_channels", "ops_read_threshold", "ops_footprint",
        "ops_store_reads", "ops_spot_detector", "ops_channel_registration",
        "ops_crosstalk",
        "ops_cell_rule",
        "ops_read_mismatches", "ops_min_reads", "ops_min_fraction",
        "ops_single_read_quality",
        "ops_decode_preset",
        "ops_registration_refine_px",
        "ops_spot_min_peak",
        "ops_spot_overrule_share",
        "ops_decode_library_calibration",
        "ops_calibration_passes",
        "ops_calibration_min_labels",
        "ops_decode_runner_up",
        "ops_runner_up_margin",
        "ops_decode_impute_missing_cycles",
        "ops_assign_by_cell_territory",
        "ops_tie",
        "ops_decode_quality_weighted",
        "ops_vote_weight_floor",
        "ops_decode_pixel_fallback",
        "ops_pixel_percentile",
        "ops_pixel_radius",
        "ops_pixel_min_pixels",
        "ops_pixel_min_fraction",
        "ops_pixel_quality",
        "ops_decoy_library_k",
        "ops_fdr_scale",
        "ops_decode_skip_unplaced_sites",
        "ops_store_field_cache",
        "ops_decode_qc",
        "ops_decode_qc_model",
        "ops_decode_qc_features",
        "ops_decode_qc_split",
        "ops_decode_qc_seed",
        "ops_decode_qc_max_calls",
        "ops_genotype_confidence",
        "ops_confidence_model",
        "ops_confidence_calibration",
        "ops_confidence_bins",
        "ops_confidence_seed",
        "ops_confidence_min",
        "ops_compare_tables",
        "ops_compare_match_px",
        "ops_compare_fields",
    ],
    "OPS performance": [
        "ops_gpu", "n_workers",
    ],
    "OPS phenotype": [
        "ops_ph_channels", "ops_ph_cell_channel", "ops_ph_nucleus_model",
        "ops_ph_cell_model", "ops_ph_nucleus_diameter",
        "ops_ph_cell_diameter", "ops_ph_downsample", "ops_ph_batch_size",
        "ops_ph_sites", "ops_ph_texture", "ops_ph_zernike", "ops_ph_funk_features",
        "ops_flatfield", "ops_flatfield_maps", "ops_flatfield_fields",
        "ops_flatfield_field_list", "ops_merge_threshold", "ops_ownership",
        "ops_flatfield_nuclear_channel", "ops_flatfield_nuclei",
        "ops_flatfield_dark", "ops_flatfield_nuclear_maps",
        "ops_flatfield_block_px", "ops_flatfield_fit", "ops_flatfield_sigma_px",
        "ops_flatfield_ridge", "ops_flatfield_floor", "ops_flatfield_centre_cells",
        "ops_flatfield_lookup", "ops_flatfield_map_grid",
        "ops_flatfield_centre_offset", "ops_flatfield_field_px",
        "ops_flatfield_holdout_fields", "ops_flatfield_holdout_n",
        "ops_flatfield_seed", "ops_flatfield_holdout_images",
        "ops_flatfield_holdout_masks", "ops_flatfield_compare_maps",
        "ops_flatfield_radial_edges", "ops_flatfield_bootstrap",
        "ops_flatfield_bootstrap_seed", "ops_flatfield_min_bin_cells",
        "ops_drop_failed_fields", "ops_drop_nonfinite_rows",
        "ops_design", "ops_control_prefix", "ops_bootstrap",
        "ops_pvalue_null", "ops_test_features", "ops_mitotic_dna_z",
        "ops_genotype_table", "ops_pool_wells", "ops_pool_interphase",
        "ops_feature_qc", "ops_profile_normalisation", "ops_tvn_mode",
        "ops_tvn_pca_sample", "ops_tvn_seed", "ops_tvn_components",
        "ops_tvn_whiten_epsilon", "ops_cluster_benchmark",
        "ops_benchmark_corum", "ops_benchmark_kegg", "ops_benchmark_string",
        "ops_reference_clusters", "ops_cluster_target",
        "ops_cluster_resolutions", "ops_cluster_seed", "ops_consensus_runs",
        "ops_mitotic_model",
        "ops_merged_root", "ops_merged_flatfield", "ops_merged_dtype",
        "ops_merged_save_sbs", "ops_tables_parquet_threshold",
        "ops_crop_per_guide", "ops_crop_size", "ops_crop_channels",
        "ops_crop_flatfield", "ops_crop_seed", "ops_crop_all_nt",
        "ops_crop_control_fraction", "ops_crop_masks",
        "ops_max_files_per_folder", "ops_ph_secondary_channels",
        "ops_ph_secondary_registration", "ops_ph_secondary_window",
        "ops_ph_secondary_fallback", "ops_ph_measure_gpu",
        "ops_cross_field_ownership", "ops_cross_field_metric",
        "ops_cross_field_min_overlap", "ops_cross_field_policy",
        "ops_reference_counts", "ops_reference_counts_level",
        "ops_reference_shuffles", "ops_reference_seed", "ops_retry_missing",
        "ops_crop_flat_mean_one",
        "ops_aggregate_level",
        "ops_nt_rows",
        "ops_profile_weighting",
        "ops_guide_efficacy",
        "ops_guide_efficacy_iters",
        "ops_guide_efficacy_emax",
        "ops_profile_cell_cycle",
        "ops_profile_g1_sg2_split_log2",
        "ops_mitotic_source",
        "ops_penetrance",
        "ops_penetrance_folds",
        "ops_penetrance_nt_pool",
        "ops_penetrance_lambda",
        "ops_escapers",
        "ops_escaper_dbic",
        "ops_escaper_min_sep",
        "ops_phase_tests",
        "ops_phase_null",
        "ops_profile_shrinkage",
        "ops_profile_shrinkage_prior_df",
        "ops_profile_generic_axes",
        "ops_profile_essential_axis_min_rho",
        "ops_profile_similarity",
        "ops_guide_null_size",
        "ops_profile_responder_alpha",
        "ops_profile_space",
        "ops_fusion_blocks",
        "ops_profile_cells", "ops_profile_wells",
        "ops_profile_run", "ops_profile_recipe",
        "ops_profile_min_cells", "ops_profile_active_filter",
        "ops_profile_cell_state", "ops_profile_gene_universe",
        "ops_mitotic_index_test", "ops_mitotic_index_min_cells",
        "ops_mitotic_index_reference",
        "ops_reference_dir",
        "ops_scoring_inputs", "ops_scoring_genes", "ops_scoring_null_size",
        "ops_scoring_min_complex", "ops_scoring_references",
        "ops_scoring_cluster_jitter", "ops_scoring_cluster_seeds",
        "ops_scoring_workers", "ops_scoring_figures",
        "ops_eval_references", "ops_eval_split", "ops_eval_null_size",
        "ops_eval_heldout_folds", "ops_eval_decoys",
        "ops_eval_time_split", "ops_eval_covariates",
        "ops_deterministic_blas",
        "ops_ph_acquisitions", "ops_ph_pipeline", "ops_ph_measure_workers",
        "ops_ph_stage_dir", "ops_ph_stage_ahead", "ops_ph_stage_threads",
        "ops_ph_measure_gpu_check", "ops_archive_root", "ops_archive_verify",
        "ops_archive_delete_local", "ops_gpu_admission", "ops_gpu_idle_checks",
        "ops_gpu_min_process_mib", "ops_gpu_vram_cap_gb",
        "ops_gpu_ignore_process_patterns", "ops_tables_integer_keys",
        "ops_tables_layout", "ops_stress_report",
        "ops_modules_runs", "ops_modules_precomputed_embedding",
        "ops_modules_profiles", "ops_modules_driver_profiles",
        "ops_eval_mitocarta", "ops_eval_external_modules", "ops_eval_arms",
        "ops_report_read_stage",
        "ops_report_overlap_zones",
        "ops_report_edge_bin_px",
        "ops_report_external_cells",
        "ops_report_external_max_px",
        "ops_cluster_rows",
        "ops_cluster_graph",
        "ops_cluster_knn",
        "ops_cluster_sweep_resolutions",
        "ops_cluster_seeds",
        "ops_cluster_target_clusters",
        "ops_cluster_shuffles",
        "ops_cluster_reference_dir",
        "ops_cluster_gene_list",
        "ops_etest",
        "ops_etest_cells",
        "ops_etest_metric",
        "ops_etest_reference_cells",
        "ops_etest_null",
        "ops_etest_holdout",
        "ops_etest_mode",
        "ops_etest_permutations",
        "ops_power_cells",
        "ops_power_seeds",
        "ops_power_retrieval",
    ],
}

#: What each setting does to the RESULT.
#:
#: ONE KEY IS DELIBERATELY ABSENT -- `plate` -- because other modules already
#: document it. A second, differently-worded tooltip for a shared setting is
#: worse than none: the reader gets a different explanation depending on which
#: panel they are looking at, for the same key.
OPS_TOOLTIPS: Dict[str, str] = {
    "cellpose_diameter":
        "(float or None) - Expected nucleus diameter in pixels for the "
        "segmentation of each stitched well. When set, each window is "
        "rescaled so that this diameter becomes the 30 pixels the model "
        "expects; empty segments the windows at their own scale. "
        "Default None.",
    "cellpose_model":
        "(str) - Which Cellpose model segments the nuclei of each stitched "
        "well. 'cpsam' runs the default model of the installed Cellpose; any "
        "other name is loaded as that model. Changing it changes which "
        "objects are found, and so which nuclei the reads are attributed "
        "to. Default 'cpsam'.",
    "dst_root":
        "(str or None) - Where measurements.db and each well's "
        "ops_report.json are written. Empty writes them into the source "
        "folder, which mixes outputs with inputs. Default None.",
    "genotype_source":
        "(str or None) - The folder holding the low-magnification "
        "sequencing acquisition that carries the barcodes. Its subfolders "
        "are searched too, for tiles named like "
        "10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-0.tif: magnification, cycle, "
        "well, channels and site. Every well found is stitched, segmented "
        "and decoded. Default None.",
    "n_workers":
        "(int) - How many parallel workers to use. More is faster until the "
        "disk becomes the limit; each worker holds its own tiles, so this "
        "multiplies memory. Default is the machine's core count.",
    "ops_base_channels":
        "(str) - Which channels carry the four bases, in the order G, T, A, "
        "C, separated by commas. These names are matched against the channel "
        "token in each tile's file name, so a run whose base channels are "
        "named differently is read by changing this rather than by renaming "
        "files. Exactly four are required: a shorter list decodes a shorter "
        "barcode than the library holds. Default 'CY3,A594,CY5,CY7'.",
    "ops_footprint":
        "(float) - How far beyond a nucleus's boundary a read may lie and "
        "still be counted as that nucleus's read, in pixels. On the "
        "reference plate 10 pixels held 98 % of the detected spots and 3 "
        "held 67 %, and the wider setting assigned 28 % more objects a "
        "barcode at slightly higher purity, because the reads sit around "
        "the rim rather than inside. Too wide starts giving a read to a "
        "neighbour. Default 10.",
    "ops_library":
        "(str or None) - A CSV of guide barcodes, with the column holding "
        "them named prefix, barcode or sequence. Given one, each nucleus's "
        "called barcode is also matched to its closest guide, and the run "
        "reports how many spots match the library exactly, which measures "
        "decoding success. Empty still calls barcodes but cannot score them "
        "against a library. Default None.",
    "ops_raster_overlap":
        "(int) - How far neighbouring sequencing tiles overlap on the "
        "microscope's raster, in pixels. This is the acquisition's own "
        "setting, not a tuning knob: it tells the stitch where to look for "
        "a neighbour, and a value well away from the truth loses edges and "
        "leaves fields unplaced. Default 213, measured on the reference "
        "plate's 1,480 pixel tiles.",
    "ops_read_threshold":
        "(float) - How much brighter than its surroundings a spot must be "
        "to be counted as a read. Lower finds more reads and more debris, "
        "which shows up as a falling library-match rate rather than as an "
        "error; higher loses real reads and leaves nuclei with too few to "
        "vote. Default 315, the reference run's own value for this plate.",
    "ops_store_reads":
        "(bool) - Write every read behind the barcodes into the ops_reads "
        "table: one row per read per cycle, with the base called, its "
        "margin and the four intensities it was called from. This is how a "
        "suspect barcode is traced back to its pixels. It is off by default "
        "because the table is large -- a full well of the reference plate is about "
        "thirty million rows -- so turn it on for a well or two rather than "
        "for a plate. Default False.",
    "ops_spot_detector":
        "(str) - Which detector finds the sequencing spots. 'native' is spaCR's own spot score, the one this pipeline was validated with. 'spotnet' is DeepCell's SpotNet, run in its own environment; its spots go through the same base calls and nucleus assignment, and ops_read_threshold no longer applies. SpotNet is licensed for NON-COMMERCIAL ACADEMIC USE ONLY, installs from the Model Zoo and needs a DeepCell token in ~/.spacr/deepcell_token. Default 'native'.",
    "ops_cell_rule":
        "(str) - How a nucleus's reads become its guide. 'mapped' matches "
        "every read to the guide library first and votes among the reads "
        "that matched, so one clean read is enough and a read with a wrong "
        "base is left out instead of splitting the vote. 'agree' is the "
        "earlier rule: at least two reads must agree on their raw called "
        "string before it is matched. On the Funk et al. 2022 plates "
        "'mapped' assigned about twice as many nuclei at a lower decoy "
        "match rate. Without a library the rule is 'agree'. "
        "Default 'mapped'.",
    "ops_channel_registration":
        "(str) - How each base channel is placed onto the reference cycle. "
        "'spots' registers the channel on its reads, with the cell "
        "background filtered away, wherever that disagrees with the "
        "background registration by more than 2 pixels; 'background' trusts"
        " the background alone, the earlier behaviour, which on the Funk et"
        " al. 2022 plates left one cycle of weak wells unshifted although "
        "it had drifted 10 to 25 pixels. 'spots_refine' (alpha) also takes "
        "a believed spot answer within ops_registration_refine_px of the "
        "background answer, correcting small residual shifts. Default "
        "'spots'.",
    "ops_crosstalk":
        "(str) - How the base caller undoes dye bleed-through between the "
        "four channels, after each cycle is put on a common scale. "
        "'argmax_median' fits one unmixing from the median of the spots each "
        "channel wins, the method brieflow uses; on the Funk et al. 2022 "
        "plates it raised the share of reads matching a guide exactly and "
        "lowered the decoy match rate. 'none' calls the brightest normalised "
        "channel as it stands, the earlier caller. 'median' and 'percentile' "
        "are two older unmixing fits kept for comparison. "
        "Default 'argmax_median'.",
    "ops_read_mismatches":
        "(int) - How many wrong bases one read may carry and still be "
        "matched to its guide under the 'mapped' rule; a read equally close "
        "to two guides is never matched. 0 keeps exact matches only, which "
        "on a library whose guides differ at two positions is what keeps "
        "wrong guides rare: allowing 1 raised the decoy match rate about "
        "twelve-fold for a few percent more nuclei. A cycle that was not "
        "read is an unknown base, not a mismatch. Default 0.",
    "ops_min_reads":
        "(int or None) - How many reads a nucleus needs before it may be "
        "assigned: library-matched reads under 'mapped', any reads under "
        "'agree'. Raising it trades nuclei for certainty. Empty takes the "
        "rule's own value, 1 for 'mapped' and 2 for 'agree'. Default None.",
    "ops_min_fraction":
        "(float) - The share of a nucleus's counted reads its guide must "
        "hold, with no tie for first. Lower assigns nuclei whose reads are "
        "split between two guides, which is where doublets and reads from a "
        "neighbour end up. Default 0.6.",
    "ops_single_read_quality":
        "(float) - Under the 'mapped' rule, the base-call quality a nucleus "
        "needs when ONE library-matched read is all it has: that read's "
        "weakest cycle, as the margin between its brightest and second "
        "channel, 0 to 1. Nuclei with two or more agreeing reads are not "
        "affected. 0.03 halved the decoy matches on the Funk et al. 2022 "
        "plates for under half a percent of the nuclei; 0 admits every "
        "one-read nucleus. Default 0.03.",
    "ops_window_overlap":
        "(int) - How far the segmentation windows overlap each other, in "
        "pixels. A nucleus is only numbered once if at least one window saw "
        "all of it, so this has to exceed the largest nucleus; when it does "
        "not, the run reports the objects no window saw whole and says by "
        "how much. Raising it costs time, because more of the well is "
        "segmented twice. Default 96.",
    "phenotype_source":
        "(str or None) - The folder holding the high-magnification "
        "phenotype acquisition of the same wells. Given one, the run aligns "
        "a few of its fields to the stitched sequencing well, fits the "
        "acquisition raster to those, and records where every phenotype "
        "field lands and which sequencing tile covers it. Empty skips that "
        "step and decodes the sequencing acquisition alone. Default None.",
    "ops_gpu":
        "(bool) - Let this run use the graphics card where spaCR finds a "
        "usable one: the tile registration's FFTs and the Cellpose outlines "
        "both have a GPU path, and both fall back to the CPU on their own if "
        "the card refuses. Turn it off when the card is busy with another "
        "job -- a shared GPU is the common case, and an out-of-memory in the "
        "middle of a plate costs more than the time the GPU saves. "
        "Default True.",
    "ops_ph_channels":
        "(str) - The phenotype channels in the order the cell table names "
        "them, separated by commas, matched against the channel tokens of the "
        "phenotype file names. Each becomes the channel part of its intensity "
        "columns, so nucleus_dapi_mean_intensity is DAPI. On the reference "
        "plate GFP is tubulin, A594 is gamma-H2AX and AF750 is phalloidin. "
        "Default 'DAPI,GFP,A594,AF750'. Alpha.",
    "ops_ph_cell_channel":
        "(str) - Which phenotype channel outlines the cells; DAPI is added as "
        "the second channel. A cytoskeletal stain that fills the cell gives "
        "whole cells, a membrane-only stain gives rims. Default 'AF750', "
        "phalloidin on the reference plate. Alpha.",
    "ops_ph_nucleus_model":
        "(str) - The Cellpose model that segments phenotype nuclei on DAPI. "
        "'cpsam' runs the installed Cellpose's default model; any other name "
        "is loaded as that model. Default 'cpsam'. Alpha.",
    "ops_ph_cell_model":
        "(str) - The Cellpose model that segments phenotype cells on the cell "
        "channel plus DAPI. The same name as the nucleus model loads the "
        "model once. Default 'cpsam'. Alpha.",
    "ops_ph_nucleus_diameter":
        "(float or None) - Expected nucleus diameter in full-resolution "
        "phenotype pixels. Set, the field is rescaled so it becomes the size "
        "the model expects; empty lets the model decide. Default None. Alpha.",
    "ops_ph_cell_diameter":
        "(float or None) - Expected cell diameter in full-resolution "
        "phenotype pixels, as for the nucleus diameter. Default None. Alpha.",
    "ops_ph_downsample":
        "(int) - Segment each phenotype field block-averaged by this factor "
        "and scale the masks back up. 2 keeps a 20X nucleus about 20 pixels "
        "across and costs a quarter of full resolution; 1 is full resolution; "
        "4 is about the sequencing resolution and blockier outlines. "
        "Measurement always uses the full-resolution pixels. Default 2. Alpha.",
    "ops_ph_batch_size":
        "(int) - How many image tiles Cellpose evaluates at once. Larger is "
        "faster on a graphics card until its memory runs out; on the CPU it "
        "changes little. Default 8. Alpha.",
    "ops_ph_sites":
        "(str or None) - Restrict the phenotype phases to these sites, as "
        "'520-525,555'. Used for a pilot on part of a well; empty runs every "
        "placed field. Default None. Alpha.",
    "ops_ph_texture":
        "(bool) - Add the GLCM homogeneity texture block to every channel of "
        "every compartment. It adds about ten percent to measurement time. "
        "Default True. Alpha.",
    "ops_ph_zernike":
        "(bool) - Also measure the Zernike moments of degree 8 of the cell "
        "and the nucleus (zernike_0 to zernike_24, 50 columns), as spaCR's "
        "Measure does with mahotas. They describe the shape of each object "
        "and slow measurement. Needs spacr[zernike]. Default False. Alpha.",
    "ops_drop_failed_fields":
        "(bool) - Leave out the cells of fields that did not register. The "
        "merge phase compares each field's link rate with the rate an "
        "unrelated field would reach by chance; a field whose alignment "
        "re-solve failed and which links under three times chance, or which "
        "links under twice chance, is dropped from the aggregate and listed "
        "with its reason in dropped_fields_<plate>.csv. Default False. "
        "Alpha.",
    "ops_drop_nonfinite_rows":
        "(bool) - Leave out cells with more than 1 % of their features "
        "missing or infinite, as Brieflow does. The aggregate report says "
        "how many cells this drops. Default False. Alpha.",
    "ops_ph_funk_features":
        "(bool) - Also measure Funk et al. 2022's features: the mass "
        "displacement of every channel and the correlation of every channel "
        "pair in the nucleus and the cell, and the form factor of both, as "
        "their OpticalPooledScreens code defines them. Adds about 30 "
        "columns. Default False. Alpha.",
    "ops_flatfield":
        "(str) - Correct the uneven illumination of the phenotype fields. "
        "'none' measures the pixels as acquired. 'analytic' corrects the "
        "measured intensity features from each cell's mean flat-field gain "
        "(phase ph_masks), in about a minute per plate. 'exact' divides "
        "every pixel before measuring, so it needs the raw images again. "
        "Default 'none'. Alpha.",
    "ops_profile_cell_state":
        "(str, list or None) - Which cells the profile phase profiles. "
        "'interphase_rule' drops the cells the stored DNA rule calls mitotic, "
        "as before. 'interphase_classifier' and 'mitotic_classifier' use the "
        "trained classifier of ops_mitotic_model; 'all' keeps every cell. A "
        "list runs the whole recipe once per state, writing "
        "genes_<state>.parquet, guides_<state>.parquet and metrics per state. "
        "Default None ('interphase_rule'). Alpha.",
    "ops_profile_gene_universe":
        "(str, list or None) - Genes kept, in this order, before the profile "
        "weighting: a table with a gene column, a JSON list or a list. "
        "Control rows are always kept. Default None (every gene). Alpha.",
    "ops_mitotic_index_test":
        "(bool) - In the profile phase, call every genotyped cell with "
        "ops_mitotic_model and test each gene's share of mitotic cells "
        "against the pooled non-targeting cells (exact binomial, BH). "
        "Non-targeting guides are scored the same way as the control. "
        "Default False. Alpha.",
    "ops_mitotic_index_min_cells":
        "(int) - Genes with fewer cells are left out of the agreement with "
        "the published mitotic index. Default 50. Alpha.",
    "ops_mitotic_index_reference":
        "(str or None) - A published gene table with per-guide mitotic and "
        "interphase cell counts (sgRNA_<k>_mitotic_cells), such as Funk et "
        "al. 2022's; its per-gene index is compared by Spearman rho with a "
        "200-shuffle control. Default None. Alpha.",
    "ops_scoring_inputs":
        "(dict or None) - The analyses phase score compares, as {name: "
        "{'genes': gene profile table, 'guides': guide profile table, "
        "'clusters': gene-to-cluster table}}; give what each analysis has. "
        "Activity needs guides; consistency and EFAAR need a profile per "
        "gene or clusters. What cannot be scored is listed in scorable.csv. "
        "Default None. Alpha.",
    "ops_scoring_genes":
        "(str, list or None) - The genes every analysis is scored on, as a "
        "table with a gene column or a list. Empty takes the genes all "
        "inputs share. Default None. Alpha.",
    "ops_scoring_null_size":
        "(int) - Permutations of the copairs null. Smaller is faster but "
        "cannot reach small p-values (the smallest is 1/(n+1)). Default "
        "10000. Alpha.",
    "ops_scoring_min_complex":
        "(int) - REPORT-half CORUM complexes with fewer shared genes are not "
        "scored. Higher keeps only large complexes, which are easier to "
        "retrieve but fewer. Default 3. Alpha.",
    "ops_scoring_references":
        "(list) - Partner references of the per-gene retrieval mAP; each "
        "adds its own rows to every table. Use the REPORT halves so no "
        "tuning was done on the scored pairs. Default "
        "['CORUM@REPORT', 'STRING>=0.7@REPORT']. Alpha.",
    "ops_scoring_cluster_jitter":
        "(float) - Noise added to one-hot cluster rows for copairs so ties "
        "are broken at random; scores are averaged over the cluster seeds. "
        "EFAAR uses exact one-hot rows. Default 0.001. Alpha.",
    "ops_scoring_cluster_seeds":
        "(list) - Seeds of the cluster-row noise; the cluster scores are "
        "averaged over them, so more seeds give steadier cluster mAP at more "
        "run time. Default [0, 1, 2]. Alpha.",
    "ops_scoring_workers":
        "(int) - copairs worker threads. More is faster and uses more "
        "memory; the scores do not change. Default 4. Alpha.",
    "ops_scoring_figures":
        "(bool) - Draw the profile and cluster figures (bars with jittered "
        "points and Holm-corrected tests). False writes the tables only. "
        "Default True. Alpha.",
    "ops_flatfield_nuclear_channel":
        "(str or None) - The DNA channel the nuclear-median flat-field map "
        "corrects, as named in ops_ph_channels. Phases flat_nuclear and "
        "flat_validate need it. Default None. Alpha.",
    "ops_flatfield_nuclei":
        "(str, dict or None) - A per-nucleus table (well, site, label, "
        "nucleus_cy, nucleus_cx and the nuclear mean of the channel, no flat "
        "applied), a folder of them, or {plate: path}. Empty joins the run's "
        "own mask-pass and phenotype cell tables in dst_root. Default None. "
        "Alpha.",
    "ops_flatfield_dark":
        "(number, dict or None) - The camera offset b of the DNA channel, or "
        "{plate: offset}. It is subtracted before the fit and added back on "
        "correction. Empty takes the offset recorded in the "
        "ops_flatfield_maps folder; with neither the build refuses rather "
        "than use 0. Default None. Alpha.",
    "ops_flatfield_nuclear_maps":
        "(str or None) - Where the nuclear maps are written, one new folder "
        "per plate holding the map, its JSON record and the plate's other "
        "channel maps copied unchanged. Empty uses "
        "<dst_root>/flatfield_nuclear. Default None. Alpha.",
    "ops_flatfield_block_px":
        "(int) - Field pixels per map cell. Each cell takes the median of "
        "the nuclei whose centroid falls in it. Default 16. Alpha.",
    "ops_flatfield_fit":
        "(str) - How the cell medians become a smooth map. "
        "'local_quadratic_log' fits a weighted quadratic to the log of "
        "(median - offset) around every cell, which follows a steep corner "
        "fall. 'gaussian' and 'local_linear' smooth more and leave the "
        "corners too dim; they are kept for comparison. Default "
        "'local_quadratic_log'. Alpha.",
    "ops_flatfield_sigma_px":
        "(float) - Width, in field pixels, of the Gaussian weight of the "
        "local fit. Wider is smoother but cannot follow the corner fall. "
        "Default 32 (2 map cells). Alpha.",
    "ops_flatfield_ridge":
        "(float or None) - Added to each local fit's equations so empty "
        "corners stay solvable. Empty takes 1e-6 for the quadratic and 1e-9 "
        "for the plane. Default None. Alpha.",
    "ops_flatfield_floor":
        "(float) - The smallest map value, so no pixel is multiplied by more "
        "than 1/floor. Default 0.10. Alpha.",
    "ops_flatfield_centre_cells":
        "(int) - The edge, in map cells, of the central square the map is "
        "scaled to 1 on. Default 18. Alpha.",
    "ops_flatfield_lookup":
        "(str) - How the map is read at field pixels, recorded with it. "
        "'block_centre' puts each map cell at the centre of the block it "
        "was measured on. 'zoom_endpoints' stretches the map to the field "
        "edges, which places corner cells up to 8 px too far out; maps "
        "built before this option are read that way. Default "
        "'block_centre'. Alpha.",
    "ops_flatfield_map_grid":
        "(str) - Where the fit is evaluated. 'block_centre' at each cell's "
        "centre. 'lookup' at the positions the zoom lookup reads, which "
        "rebuilds the reference maps byte for byte. Default "
        "'block_centre'. Alpha.",
    "ops_flatfield_centre_offset":
        "(float) - Field-pixel offset of a block's centre from its first "
        "pixel: 8 is the convention the reference maps use, 7.5 the middle "
        "of a 16-pixel block. Default 8. Alpha.",
    "ops_flatfield_field_px":
        "(int or None) - Field size in pixels along each axis. Empty takes "
        "the largest nucleus centroid rounded up to whole blocks. Default "
        "None. Alpha.",
    "ops_flatfield_holdout_fields":
        "(str, list, dict or None) - Fields left out of the map and used to "
        "check it, as well:site, or {plate: list}. Empty draws "
        "ops_flatfield_holdout_n fields, the same number per well. Default "
        "None. Alpha.",
    "ops_flatfield_holdout_n":
        "(int) - How many fields to hold out when none are listed, split "
        "evenly over the wells. More gives tighter radial intervals and "
        "leaves fewer nuclei to build from. Default 24. Alpha.",
    "ops_flatfield_seed":
        "(int) - Seed of the held-out field draw; another seed holds out "
        "other fields and so builds a slightly different map. Default 0. "
        "Alpha.",
    "ops_flatfield_holdout_images":
        "(str, dict or None) - Folder, or {plate: folder}, of the held-out "
        "fields' raw images of the DNA channel, one *<well>_<site>.npy each, "
        "for the exact re-measure of phase flat_validate. Default None. "
        "Alpha.",
    "ops_flatfield_holdout_masks":
        "(str, dict or None) - The run folder holding "
        "<well>/phenotype_masks/site-<n>.npz of the held-out fields. Empty "
        "uses dst_root. Default None. Alpha.",
    "ops_flatfield_compare_maps":
        "(dict or None) - More maps to check beside the new and the earlier "
        "one, as {name: path to .npy}, read by zoom. Default None. Alpha.",
    "ops_flatfield_radial_edges":
        "(list) - Ring edges, in pixels from the field centre, of the "
        "held-out radial table. Default [0, 500, 1000, 1400, 1600, 1800, "
        "1900, 2000, 2100]. Alpha.",
    "ops_flatfield_bootstrap":
        "(int) - Bootstraps over held-out fields for the 95 % intervals of "
        "the radial table. 0 skips them. Default 2000. Alpha.",
    "ops_flatfield_bootstrap_seed":
        "(int) - Seed of the field bootstraps of the radial intervals; it "
        "moves the interval ends, not the medians. Default 20261109. Alpha.",
    "ops_flatfield_min_bin_cells":
        "(int) - Rings with fewer held-out nuclei are left out of the "
        "largest-deviation score. Default 100. Alpha.",
    "ops_flatfield_maps":
        "(str or None) - The folder holding the plate's flat-field maps, "
        "flat_info.json and one flat_*_<channel>.npy per channel. Empty "
        "uses the flatfield folder in dst_root, which the ph_flatfield "
        "phase fills. Maps are never overwritten. Default None. Alpha.",
    "ops_flatfield_fields":
        "(int) - How many random fields (seed 0) the ph_flatfield phase "
        "builds the maps from. Fewer fields leave cells in the median; 204 "
        "built the Funk et al. plate maps. Default 204. Alpha.",
    "ops_flatfield_field_list":
        "(str or None) - A JSON file of [well, site] pairs to build the "
        "maps from instead of a random draw, such as the fields.json an "
        "earlier build wrote, to rebuild the same maps. Default None. "
        "Alpha.",
    "ops_ownership":
        "(str) - Which copy of a cell imaged in two overlapping fields is "
        "kept. 'closer_match' keeps the closer sequencing match, as before "
        "spaCR 1.5.1.8; it is often the copy cut by the field's edge. 'a1b' "
        "keeps the copy from the field in which the sequencing nucleus "
        "lies deepest, and 'a1' the one whose phenotype nucleus does. Both "
        "also leave out cells that touch a field's edge, which needs the "
        "ph_masks phase. Default 'closer_match'. Alpha.",
    "ops_merge_threshold":
        "(float) - How far, in sequencing pixels, a phenotype nucleus may "
        "land from a sequencing nucleus after alignment and still be called "
        "the same cell. Wider merges more cells and more wrong ones, which "
        "give a phenotype the wrong guide. Default 4, under half a 10X "
        "nucleus diameter. Alpha.",
    "ops_design":
        "(str or None) - The guide design table, with sgRNA and gene_symbol "
        "columns. The aggregate phase matches each decoded barcode prefix to "
        "its guide and gene through it; prefixes two genes share are "
        "dropped and counted. Default None. Alpha.",
    "ops_control_prefix":
        "(str) - Genes whose name starts with this are the non-targeting "
        "controls every feature is normalised against, well by well. "
        "Default 'nontargeting'. Alpha.",
    "ops_bootstrap":
        "(int) - Replicates for each gene's P value against "
        "non-targeting guides of the same sizes, with Benjamini-Hochberg "
        "FDR. 0 skips the test; 1000 resolves P values to about 0.001 and "
        "costs minutes per plate. Default 0. Alpha.",
    "ops_pvalue_null":
        "(str) - The null each gene is tested against. 'calibrated' adds "
        "the variance that non-targeting guide medians show beyond cell "
        "sampling, so non-targeting guides tested as genes are called hits "
        "about 5 percent of the time. 'cell_bootstrap' draws non-targeting "
        "cells only. It ignores that variance and calls too many hits: on "
        "one Funk et al. 2022 plate it called 4,053 of 5,049 genes. It is "
        "kept to rerun analyses made before spaCR 1.5.1.8. "
        "Default 'calibrated'. Alpha.",
    "ops_test_features":
        "(str or None) - The features each gene is tested on, as a "
        "comma-separated list of cell table columns. Empty tests the "
        "nuclear area, the nuclear DNA mean and integrated intensity, and "
        "the cell mean intensity of every other channel. Default None. "
        "Alpha.",
    "ops_genotype_table":
        "(str or None) - A table of guide calls that replaces the decode "
        "phase's calls, such as the calls of a later decode rule. It needs "
        "well, object_id and mapped_guide columns; a plate column, when "
        "present, keeps this plate's rows. Each sequencing nucleus may "
        "appear once. Empty uses the calls in measurements.db. "
        "Default None. Alpha.",
    "ops_pool_wells":
        "(str or None) - Pool only these wells, as 'A1,A2'. Each cell is "
        "still normalised against its own well, and the tables get a "
        "wells suffix so they do not replace the full ones. Empty pools "
        "every well run. Default None. Alpha.",
    "ops_pool_interphase":
        "(bool) - Summarise guides and genes from interphase cells only. "
        "Every cell is still normalised against its well's controls first, "
        "and the gene test then uses interphase control cells. The tables "
        "get an interphase suffix. Default False. Alpha.",
    "ops_feature_qc":
        "(bool) - Leave out the features whose robust z is undefined in "
        "every well, because their control cells have no spread. Such a "
        "column would be empty in both tables. Each one is listed with its "
        "reason in ops_feature_qc_<plate>.csv. Default True. Alpha.",
    "ops_profile_normalisation":
        "(str) - How guide and gene profiles are normalised. 'robust_z' "
        "scales every feature against its well's non-targeting cells. "
        "'tvn' also writes typical variation normalisation profiles, as "
        "Brieflow makes them: components in which the non-targeting cells "
        "of every well have the same spread in every direction. TVN "
        "profiles cluster better, but their components have no feature "
        "names, so the gene test still uses the robust z tables. "
        "Default 'robust_z'. Alpha.",
    "ops_tvn_mode":
        "(str) - 'streamed' fits TVN one well at a time and keeps only the "
        "control cells and a sample in memory, so it suits whole plates "
        "and pools of plates. 'in_memory' holds every cell at once, as "
        "Brieflow does, and needs about four times the size of the cell "
        "table in memory. The two differ only in the PCA sample. "
        "Default 'streamed'. Alpha.",
    "ops_tvn_pca_sample":
        "(int) - The number of cells, drawn at random, that the PCA of TVN "
        "is fitted on. The whitening gives weak PCA directions full "
        "weight, so a different sample changes those directions and the "
        "gene similarities they carry. A larger sample makes them steadier "
        "and costs memory. Default 100000. Alpha.",
    "ops_tvn_seed":
        "(int) - The seed of the TVN PCA sample. Change it to see how much "
        "the profiles depend on the sample. Default 0. Alpha.",
    "ops_tvn_components":
        "(float, int or None) - Keep only the leading control directions "
        "of TVN: a whole number keeps that many, a fraction below 1 keeps "
        "the directions holding that share of the control variance. The "
        "dropped directions are the weak ones that the whitening would "
        "blow up. Empty keeps all of them, as Brieflow does. "
        "Default None. Alpha.",
    "ops_tvn_whiten_epsilon":
        "(float) - Added to the control variance of every TVN direction "
        "before it is scaled to 1, so weak directions are shrunk instead of "
        "blown up and the profiles depend less on which cells the PCA saw. "
        "On one Funk et al. plate, 3 made the streamed and in-memory fits "
        "agree much better (gene similarity correlation 0.93 instead of "
        "0.81) with an unchanged STRING score. 0 is Brieflow's TVN exactly; "
        "use it to reproduce Brieflow or the plate A pilot. Default 3. Alpha.",
    "ops_cluster_benchmark":
        "(bool) - Cluster the gene profiles and score the clusters, as "
        "Funk et al. 2022 and Brieflow do: PHATE on the leading components, "
        "then Leiden at the resolution whose cluster count is closest to "
        "ops_cluster_target. Each clustering is scored by the share of "
        "clusters enriched for a CORUM complex or a KEGG pathway, by STRING "
        "precision and recall, and against shuffled-component and k-means "
        "controls. TVN profiles are used when they are made. Needs the "
        "three benchmark tables and the clustering extra: "
        "pip install \"spacr[clustering]\". Default False. Alpha.",
    "ops_benchmark_corum":
        "(str or None) - The CORUM complex table for the cluster benchmark, "
        "tab-separated with gene_name and group columns, as Brieflow ships "
        "it. spaCR does not include it. Default None. Alpha.",
    "ops_benchmark_kegg":
        "(str or None) - The KEGG pathway table for the cluster benchmark, "
        "tab-separated with gene_name and group columns. Default None. "
        "Alpha.",
    "ops_benchmark_string":
        "(str or None) - The STRING gene pair table for the cluster "
        "benchmark, tab-separated with gene_name and pair columns. It also "
        "scores profile similarity directly, as the area under the "
        "precision-recall curve of all gene pairs. Default None. Alpha.",
    "ops_reference_clusters":
        "(str or None) - A published clustering to compare with, such as "
        "Funk et al.'s gene table: a CSV, TSV or Excel file with a gene "
        "(or gene_symbol) column and a cluster (or interphase_cluster) "
        "column. It is scored by the same code and gives the adjusted Rand "
        "index. Default None. Alpha.",
    "ops_cluster_target":
        "(int) - The cluster count the Leiden resolution is chosen for. "
        "The share of clusters enriched depends on cluster size, so compare "
        "clusterings at the same count. Default 251, Brieflow's count. "
        "Alpha.",
    "ops_cluster_resolutions":
        "(str) - The Leiden resolutions tried, as '5,10,12'. A higher "
        "resolution splits the genes into more, smaller clusters; the one "
        "whose cluster count is closest to ops_cluster_target is reported "
        "first. Default '5,10,12,15,16,17,18,20,22'. Alpha.",
    "ops_cluster_seed":
        "(int) - The PHATE and Leiden seed. With the same seed and library "
        "versions the clusters are the same on every run. Default 42. "
        "Alpha.",
    "ops_consensus_runs":
        "(int) - Repeat the clustering this many times, each on 90 percent "
        "of the genes, with each gene's guides drawn again, its own seed "
        "and a resolution from a sweep. The runs give how often two genes "
        "share a cluster. 0 skips it; 100 takes about half an hour per "
        "plate. Default 0. Alpha.",
    "ops_profile_cells":
        "(str, dict or None) - The folder of stored per-well cell tables "
        "(raw_<well>.parquet with well, sgRNA, gene, mitotic and feature "
        "columns) that the profile phase turns into gene profiles without any "
        "image step, or {plate: folder} to pool several plates, each well a "
        "batch. Default None. Alpha.",
    "ops_profile_wells":
        "(list or None) - Profile only these wells of each plate's cell tables; "
        "empty takes every raw_<well>.parquet found. Leaving wells out changes "
        "the PCA sample and the TVN batches, so the profiles change too. "
        "Default None. Alpha.",
    "ops_profile_run":
        "(str) - The name of the profile run: its results go to "
        "<dst_root>/ops_profile/<name>. Empty uses the plate names. Default ''. "
        "Alpha.",
    "ops_profile_recipe":
        "(dict or None) - How cells become gene profiles: lof (1 drops the 1% "
        "intensity outliers), impute ('median' or 'knn'), pca ('c0' per-well "
        "standardised, or 'brieflow'), agg ('guide' = median of guide medians, "
        "or 'gene' = median of all cells) and nt ('pseudo' = four control "
        "guides per pseudo-gene, or 'barcode'). Empty is lof 0, median, c0, "
        "guide, pseudo. Default None. Alpha.",
    "ops_profile_min_cells":
        "(int) - Genes and pseudo-genes with this many cells or fewer are left "
        "out of the profile table and the metrics; raising it drops weakly "
        "sampled genes whose profiles are noisy. Default 10. Alpha.",
    "ops_profile_active_filter":
        "(dict or None) - Keep only active genes: {'hits_table': path, 'fdr': "
        "0.05} keeps the genes with any *_fdr column of the hits table below "
        "fdr, plus the controls. Retrieval then scores only genes with a "
        "phenotype. Default None. Alpha.",
    "ops_profile_shrinkage":
        "(str or None) - 'eb_nt' shrinks each gene profile toward the "
        "controls by how reliable its guides are: the control guides set "
        "the noise, each gene's own guide spread is pooled with it, and the "
        "profile is multiplied by tau2 / (tau2 + V / K) per dimension. Not "
        "stacked with ops_profile_weighting. The aggregate and profile "
        "phases both apply it. Default None. Alpha.",
    "ops_profile_weighting":
        "(str or None) - 'contrastive_wiener' weights the TVN gene profiles "
        "by how far each direction rises above guide noise: directions "
        "solving Sigma_genes v = lambda Sigma_NT v, the noise the Ledoit- "
        "Wolf covariance of the non-targeting guides, each weighted by "
        "max((lambda - 1) / lambda, 0). Refused when fewer than two "
        "directions rise above the noise, and not stacked with "
        "ops_profile_shrinkage. The aggregate and profile phases both "
        "apply it. Default None. Alpha.",
    "ops_profile_similarity":
        "(str or None) - How gene-gene similarity is written besides the "
        "gene tables: 'guide_level' (mean guide-guide cosine with a P value "
        "per pair from random guide sets), 'responder' or "
        "'sliced_wasserstein' (responder profiles, energy distance and a "
        "sliced Wasserstein embedding of each gene's cells). The profile "
        "phase's metrics always compare gene profiles by cosine. Default "
        "None. Alpha.",
    "ops_reference_dir":
        "(str, list or None) - The folder, or folders, holding the reference "
        "files the metrics score against: Brieflow's CORUM, KEGG and STRING "
        "benchmark tables and the prepared STRING, BioGRID and time-split pair "
        "tables (spacr.ops_reference). Default None. Alpha.",
    "ops_eval_references":
        "(list) - The references gene profiles are scored against: any of "
        "CORUM, STRING>=0.7, STRING>=0.4, KEGG and BioGRID. Each gene's "
        "partners must rank above its non-partners; more references give a "
        "fuller but slower evaluation. Default all five. Alpha.",
    "ops_eval_split":
        "(str) - 'select' or 'report' scores only that half of each reference, "
        "so settings chosen on one half are reported on the other; 'all' scores "
        "the whole references and both halves of CORUM, STRING>=0.7 and "
        "BioGRID. Default 'all'. Alpha.",
    "ops_eval_null_size":
        "(int) - Random rankings per list shape behind each gene's p value; the "
        "smallest p value is 1/(n+1). Larger is slower and resolves smaller p "
        "values. Default 5000. Alpha.",
    "ops_eval_heldout_folds":
        "(int) - Folds in which a CORUM complex's members are hidden and scored "
        "against the rest of the complex; more folds hide fewer members at a "
        "time. Default 5. Alpha.",
    "ops_eval_decoys":
        "(int) - Fitness-matched non-partners drawn per partner (and per hidden "
        "complex member) in the specificity metrics; a profile that only tracks "
        "cell fitness scores the random expectation against them. Default 10. "
        "Alpha.",
    "ops_eval_time_split":
        "(bool) - Also score the references known in 2022 and the gene pairs "
        "added since: do the screen's most similar unknown pairs turn up later? "
        "Needs the time-split tables in ops_reference_dir. Default False. "
        "Alpha.",
    "ops_eval_covariates":
        "(str or None) - A table with gene and cells_per_guide columns, the "
        "fitness that decoy partners are matched on; empty takes cells per "
        "guide from the run's own guide table. Default None. Alpha.",
    "ops_report_read_stage":
        "(bool) - Add the read stage to the report phase's yield funnel: "
        "sbs_nuclei, sbs_nuclei_with_read and its %, the owned sequencing "
        "nuclei whose territory holds at least one spot above the read "
        "threshold. The decode phase must have run with this on, so its field "
        "stats count them. Default False. Alpha.",
    "ops_report_overlap_zones":
        "(bool) - The report phase writes phenotype-genotype match rates in "
        "tile-overlap strips and cores: a cell is in a strip when its "
        "centroid maps inside another acquired field. Per segmented copy and, "
        "after the ph_ownership phase, per physical cell (matched when any "
        "copy is linked, zoned by its owner copy); also by distance to the "
        "field edge and per field. Default False. Alpha.",
    "ops_report_edge_bin_px":
        "(int) - The width in pixels of the field-edge distance bins of the "
        "overlap report (overlap_match_by_edge_distance.csv). Needs "
        "ops_report_overlap_zones. Default 100. Alpha.",
    "ops_report_external_cells":
        "(str or None) - Another pipeline's phenotype cells (field, label, y, "
        "x, matched; optional well) to compare on the same physical cells: "
        "each is assigned to the nearest spaCR copy in its field, and both "
        "pipelines' match rates per zone are reported with exact McNemar "
        "counts. Needs ops_report_overlap_zones and the ph_ownership table. "
        "Default None. Alpha.",
    "ops_report_external_max_px":
        "(float) - The largest distance in pixels between an external cell's "
        "centroid and the spaCR centroid it is assigned to; farther cells "
        "stay unassigned. Default 8.0. Alpha.",
    "ops_cluster_rows":
        "(str or None) - A table of ready gene-profile rows (gene plus "
        "numeric columns, used as given) for the cluster_sweep phase; empty "
        "clusters the profile phase's gene rows, embedded as the evaluate "
        "phase embeds them. Default None. Alpha.",
    "ops_cluster_graph":
        "(list) - The graphs the cluster_sweep phase clusters: 'phate', "
        "PHATE's symmetrised diffusion operator (10 cosine neighbours), "
        "and/or 'knn', a symmetric cosine k-nearest-neighbour graph. Default "
        "['phate', 'knn']. Alpha.",
    "ops_cluster_knn":
        "(int) - The neighbours of the 'knn' graph of the cluster_sweep "
        "phase. Default 12. Alpha.",
    "ops_cluster_sweep_resolutions":
        "(dict or None) - Leiden resolutions per graph for the cluster_sweep "
        "phase, such as {'phate': [5, 12], 'knn': [1, 2]}; a graph not named "
        "takes its full grid (phate 1-40, knn 0.25-16). More resolutions draw "
        "more of the trade-off curve and take longer. Default None. Alpha.",
    "ops_cluster_seeds":
        "(list) - The PHATE and Leiden seeds of the cluster_sweep phase; the "
        "summary gives the mean, minimum and maximum over them. Default [42, "
        "1, 2]. Alpha.",
    "ops_cluster_target_clusters":
        "(int) - The cluster count the sweep summary flags: the resolution "
        "whose mean cluster count is nearest it, so a comparison is made at a "
        "matched number of clusters. Default 227 (Brieflow's interphase "
        "clusters). Alpha.",
    "ops_cluster_shuffles":
        "(int) - Shuffled-label controls per partition in the cluster_sweep "
        "phase: the scored genes' cluster labels permuted with cluster sizes "
        "kept. Default 1. Alpha.",
    "ops_cluster_reference_dir":
        "(str or None) - The folder holding Brieflow's CORUM, KEGG and STRING "
        "benchmark tables that the cluster_sweep phase scores against; empty "
        "uses ops_reference_dir. Default None. Alpha.",
    "ops_cluster_gene_list":
        "(str or None) - A table with a gene column: the genes the "
        "cluster_sweep phase scores, so arms are compared on the same genes. "
        "Every row is still clustered. Empty scores every targeting row. "
        "Default None. Alpha.",
    "ops_etest":
        "(bool) - Keep the per-cell aligned embedding when the profile phase "
        "runs (cells_E.npy and cells_meta.parquet in its folder), which the "
        "etest phase tests. Default False. Alpha.",
    "ops_etest_cells":
        "(str or None) - A folder holding cells_E.npy and cells_meta.parquet "
        "(gene_symbol_0, cell_barcode_0) for the etest phase; empty reads the "
        "profile run's own. Default None. Alpha.",
    "ops_etest_metric":
        "(str) - The E-distance the etest phase writes: 'euclidean' (energy "
        "distance), 'sqeuclidean' (scPerturb's default) or 'both'. Default "
        "'both'. Alpha.",
    "ops_etest_reference_cells":
        "(int) - The non-targeting cells drawn once as the reference every "
        "gene's E-distance is measured against. Default 2000. Alpha.",
    "ops_etest_null":
        "(int) - The non-targeting pseudo-genes, drawn from cells outside the "
        "reference at a gene's cell count, whose E-distances form the "
        "empirical null of the E-test. Default 1000. Alpha.",
    "ops_etest_holdout":
        "(int) - Further non-targeting pseudo-genes scored against the null: "
        "their share at p < 0.05 is the false-positive control. Default 500. "
        "Alpha.",
    "ops_etest_mode":
        "(str) - 'empirical_null' tests each gene against pseudo-genes of "
        "random non-targeting cells; 'guide_null' against pseudo-genes of "
        "four non-targeting guides, the shape of a gene, which keeps "
        "guide-level structure in the null (the guides are split in halves "
        "for the null and the held-out control); 'permutation' permutes the "
        "gene's cells with the reference (scPerturb's E-test, "
        "ops_etest_permutations times; slow). Default 'empirical_null'. "
        "Alpha.",
    "ops_etest_permutations":
        "(int) - Label permutations per gene when ops_etest_mode is "
        "'permutation'. Default 1000. Alpha.",
    "ops_power_cells":
        "(list) - The cells-per-gene caps of the power curve; None keeps "
        "every cell (scored once, its null drawn at the median cells per "
        "gene). Default [10, 25, 50, 100, 200, 500, None]. Alpha.",
    "ops_power_seeds":
        "(list) - The subsampling seeds of the power curve; each capped draw "
        "is repeated once per seed. Default [0, 1, 2]. Alpha.",
    "ops_power_retrieval":
        "(bool) - Also draw the retrieval power curve: at most n cells per "
        "gene and every control cell, guide and gene medians, then the "
        "evaluate phase's retrieval metrics on ops_reference_dir, with paired "
        "Wilcoxon tests against the largest cap. Default False. Alpha.",
    "ops_deterministic_blas":
        "(bool) - Run the profile phases with the BLAS and OpenMP libraries "
        "held to one thread, so a re-run of the same settings file writes "
        "bit-identical profiles and metrics. Off is faster on many cores, "
        "and the last digits of the PCA and TVN can then change from run to "
        "run. Default True. Alpha.",
    "ops_modules_runs":
        "(int) - How many resampled PHATE + Leiden runs the modules phase "
        "clusters the gene profiles in. Each run takes 90 % of the genes, "
        "resamples each gene's guides and has its own seed and resolution; "
        "how often a gene shares a cluster with a module's known core over "
        "these runs is one of the gates for a new module member. More runs "
        "give steadier co-membership and take longer. Default 100. Alpha.",
    "ops_modules_precomputed_embedding":
        "(bool) - Cluster the profile columns as they are in the modules "
        "phase, instead of standardising them and keeping the principal "
        "components that hold 99 % of the variance. Turn it on for weighted "
        "profiles, such as contrastive weighting, which standardising again "
        "would undo. Default False. Alpha.",
    "ops_modules_profiles":
        "(str or None) - The folder holding the genes.parquet and "
        "guides.parquet profile tables the modules phase clusters; its "
        "results go to a modules folder inside it. Empty takes the profile "
        "phase's folder, dst_root/ops_profile/<run>. Default None. Alpha.",
    "ops_modules_driver_profiles":
        "(str or None) - The robust z profile tables the modules phase reads "
        "each module's driving features from: a folder with genes.parquet "
        "and guides.parquet, or the two paths as a list. Empty uses the "
        "clustered profiles. Default None. Alpha.",
    "ops_eval_mitocarta":
        "(str or None) - The MitoCarta 3.0 table (Human.MitoCarta3.0.xls) "
        "the mitocarta phase scores clusterings against: which clusters are "
        "enriched for mitochondrial genes, and how well the profiles find the "
        "genes of the same sub-compartment (OXPHOS, mitoribosome, ATP "
        "synthase, membrane organisation). Default None. Alpha.",
    "ops_eval_external_modules":
        "(str or None) - A JSON file naming the published tables the "
        "mitocarta phase compares with: an external module table and the "
        "sub-modules to test for coherence in the consensus atlas, the "
        "published clusterings, and named clusters whose members and partner "
        "genes are followed. Default None. Alpha.",
    "ops_eval_arms":
        "(str, list or None) - The clusterings the mitocarta phase scores, "
        "in report order: a list, or a JSON file holding one, of entries with "
        "a label and any of a cluster assignment table, a profile file and a "
        "consensus atlas folder. Default None. Alpha.",
    "ops_mitotic_model":
        "(str or None) - A mitotic-cell classifier trained on labelled cells "
        "with spacr.ops_mitotic.train_mitotic_classifier. When set it calls "
        "mitotic cells instead of the DNA rule of ops_mitotic_dna_z. The "
        "file can run code when it is opened, so use only models you "
        "trained or trust. Default None. Alpha.",
    "ops_mitotic_dna_z":
        "(float) - A cell is called mitotic when its nuclear mean DAPI "
        "intensity is this many robust standard deviations above its well's "
        "median and its nucleus is no larger than the median: condensed "
        "chromatin. Lower calls more cells mitotic. Default 3. Alpha.",
    "ops_merged_root":
        "(str or None) - Folder the ph_merged phase writes merged/ (one "
        "compressed array file per phenotype field: channels, nucleus, cell "
        "and cytoplasm masks, ownership flags) and the ph_crops phase "
        "writes its crop shards to. Point it at network storage to keep the "
        "working database on local disk. Empty uses dst_root. Default None. "
        "Alpha.",
    "ops_merged_flatfield":
        "(bool) - Divide the channels the ph_merged phase stores by the "
        "plate's flat-field maps, as (I - b) / F + b, when the maps exist. "
        "Off stores the projected raw counts. Default True. Alpha.",
    "ops_merged_dtype":
        "(str) - How ph_merged stores the channels: 'uint16' rounds them "
        "back to camera counts and halves the disk use; 'float32' keeps the "
        "flat-field-corrected values unrounded. Default 'uint16'. Alpha.",
    "ops_merged_save_sbs":
        "(bool) - Reserved for also storing the sequencing stacks under "
        "merged/sbs; not built yet, so the phase only reports it. Default "
        "False. Alpha.",
    "ops_tables_parquet_threshold":
        "(int) - Above this many owned cells in a plate, the ph_tables "
        "phase writes the wide feature columns to one Parquet file per "
        "table and well, and measurements.db keeps only the key columns; a "
        "SQLite table of tens of millions of wide rows is slow to write and "
        "to join. Default 5,000,000. Alpha.",
    "ops_crop_per_guide":
        "(int or None) - How many cells of each targeting guide the "
        "ph_crops phase cuts per well, chosen at random with ops_crop_seed. "
        "Empty cuts every owned, linked cell. Default None. Alpha.",
    "ops_crop_size":
        "(int) - Side of each crop in phenotype pixels, centred on the cell "
        "mask's centroid; pixels outside the cell are 0. Larger crops show "
        "more context and take more disk. Default 96. Alpha.",
    "ops_crop_channels":
        "(str or None) - Which phenotype channels the crops hold, "
        "comma-separated, in that order. Empty keeps every channel of "
        "ops_ph_channels. Default None. Alpha.",
    "ops_crop_flatfield":
        "(bool) - Correct each field with the plate's flat-field maps "
        "before cutting its crops, so a cell at the corner of a field is as "
        "bright as the same cell at its centre. The crops stay raw "
        "otherwise: no per-crop stretch. Default True. Alpha.",
    "ops_crop_seed":
        "(int) - Seed of the random choice of cells per guide "
        "(ops_crop_per_guide). The same seed picks the same cells in every "
        "run. Default 0. Alpha.",
    "ops_crop_all_nt":
        "(bool) - With ops_crop_per_guide set, still cut every "
        "non-targeting cell, the control set crops are compared against. "
        "Default True. Alpha.",
    "ops_crop_control_fraction":
        "(float) - Also cut this share of all owned, whole cells, linked or "
        "not, chosen by a stable hash of plate, well, site and label so the "
        "choice does not depend on run order or genotype. 0 turns it off. "
        "Default 0.01. Alpha.",
    "ops_crop_masks":
        "(bool) - Store each crop's mask with it (1 = the cell, 2 = its "
        "nucleus) as a second image page. Default False. Alpha.",
    "ops_max_files_per_folder":
        "(int) - Most crops one shard holds and most files one folder of "
        "the crop store holds; more go to the next shard or to a part-NNNN "
        "subfolder. Folders of millions of files take minutes to list on "
        "network storage. Default 10,000. Alpha.",
    "ops_ph_secondary_channels":
        "(str or None) - For a plate whose phenotype channels were imaged "
        "in two acquisitions, the channels the second one contributes, "
        "comma-separated (for example AF750). They are read from the second "
        "acquisition's files and moved by the registration shift; the other "
        "channels come only from the first. Empty: one acquisition. Default "
        "None. Alpha.",
    "ops_ph_secondary_registration":
        "(str) - How the second acquisition is put on the first: "
        "'phase_correlation' measures a whole-pixel shift per field on the "
        "centre of the fields, on a channel both hold or on DAPI against "
        "the second acquisition's first channel, and stores it in "
        "ops_phenotype; 'none' uses no shift. Default 'none'. Alpha.",
    "ops_ph_secondary_window":
        "(int) - Side in pixels of the centre square the registration shift "
        "is measured on. Larger is slower and more robust in sparse fields. "
        "Default 1024. Alpha.",
    "ops_ph_secondary_fallback":
        "(str) - What replaces a field's shift that lies more than 12 px "
        "from the median of the well's earlier fields (once five are "
        "registered): 'well_median' uses that median; 'none' keeps the "
        "field's own shift. Default 'well_median'. Alpha.",
    "ops_ph_measure_gpu":
        "(bool) - Measure the intensity and texture features on the CUDA "
        "card. It is faster; the values agree with the CPU measurement to "
        "float precision except skew and kurtosis, which can differ "
        "slightly. Default False. Alpha.",
    "ops_cross_field_ownership":
        "(bool) - Find every copy of a cell that two or more overlapping "
        "phenotype fields segmented (phase ph_ownership), by comparing the "
        "fields' cell masks through their transforms, and keep one copy per "
        "physical cell in the aggregate, merged/ and the crops. Off: "
        "outputs are identical to a run without it. Default False. Alpha.",
    "ops_cross_field_metric":
        "(str) - How two masks are scored as copies: 'overlap_fraction' "
        "divides the shared pixels by the smaller mask, so a copy cut by a "
        "field's edge still matches the whole one; 'iou' divides by their "
        "union. Default 'overlap_fraction'. Alpha.",
    "ops_cross_field_min_overlap":
        "(float) - Smallest score, in (0, 1], at which two masks of "
        "overlapping fields count as the same cell. Lower merges more "
        "neighbours into one cell; higher splits cut copies from their "
        "whole twin. Default 0.5. Alpha.",
    "ops_cross_field_policy":
        "(str) - What the aggregate does with the copies: 'one_per_cell' "
        "keeps one genotyped row per physical cell, its largest copy; "
        "'owner_only' drops every copy that is not the owner, which on Funk "
        "plate A also drops many genotyped cells whose owner was never "
        "linked; 'flag_only' writes the ownership table and changes nothing "
        "else. Default 'one_per_cell'. Alpha.",
    "ops_reference_counts":
        "(str or None) - A published per-guide count table (columns guide, "
        "gene, count; CSV, TSV or Excel), for example Funk et al. 2022's "
        "supplementary counts. The report phase ranks spaCR's count of "
        "every guide and gene against it (Spearman's rho with a shuffled "
        "control). Empty skips the comparison. Default None. Alpha.",
    "ops_reference_counts_level":
        "(str) - Which spaCR counts the report compares with "
        "ops_reference_counts: 'sbs' counts decoded sequencing nuclei per "
        "guide, 'phenotype' the linked, owned cells the aggregate pooled; "
        "'sbs,phenotype' writes both. Each result names its level, so the "
        "two are never mixed. Default 'sbs'. Alpha.",
    "ops_reference_shuffles":
        "(int) - How many times the report permutes spaCR's counts across "
        "guides for the shuffled-label control of the rank correlation; it "
        "reports the 95th percentile of abs(rho). More is steadier and "
        "slower. Default 1000. Alpha.",
    "ops_reference_seed":
        "(int) - Seed of the shuffled-label control's permutations; the "
        "same seed gives the same control in every run. Default 0. Alpha.",
    "ops_retry_missing":
        "(bool) - Try again the phenotype fields an earlier run found missing "
        "or unreadable in the source (recorded in "
        "ops_missing_fields_<well>.parquet). Off, a resumed run skips them, "
        "so a well with fields absent from a public download still "
        "finishes. Default False. Alpha.",
    "ops_crop_flat_mean_one":
        "(bool) - Scale each flat-field map to a mean of 1 over the field "
        "before the crops are corrected with it, as the plate A gene-class "
        "crops were cut; off divides by the maps as stored, as merged/ and "
        "the measurement do. Default False. Alpha.",
    "ops_decode_preset":
        "(str) - A named set of decode settings applied to every decode key "
        "this settings file leaves at its default. 'rule_a_s2' is the final "
        "rule of the decode-yield study: refined spot registration, "
        "library-calibrated base calls, territory attribution, "
        "quality-weighted votes, the runner-up one-mismatch rule, the pixel "
        "fallback and a decoy FDR report. On plate A of Funk et al. 2022 it "
        "assigned 74.4 % of nuclei at 0.065 false calls per 100. 'none' "
        "changes nothing. Default 'none'. Alpha.",
    "ops_registration_refine_px":
        "(int) - Under ops_channel_registration 'spots_refine', a believed "
        "spot registration this many pixels or fewer from the background "
        "answer replaces it, so a small residual shift is corrected instead "
        "of kept. Larger lets the spots move more channels. Default 3. "
        "Alpha.",
    "ops_spot_min_peak":
        "(float) - How far a spot registration's peak must stand above the "
        "mean of its correlation surface to be believed, under 'spots' and "
        "'spots_refine'. Higher trusts the spots on fewer channels; a field "
        "with no reads peaks at a few. Default 20. Alpha.",
    "ops_spot_overrule_share":
        "(float) - The spots overrule a background registration only where "
        "the spot surface at the background's shift is below this share of "
        "the spot peak. Lower overrules less often. Default 0.5. Alpha.",
    "ops_decode_library_calibration":
        "(bool) - Learn each cycle's four base axes from the reads the "
        "guide library names on their other cycles, then unmix every read "
        "of that cycle with them. It corrects channel drift a single "
        "unmixing cannot. Decoys are never used to learn the axes, so the "
        "decoy FDR stays an independent check. Default False. Alpha.",
    "ops_calibration_passes":
        "(int) - How many times the library calibration re-labels the reads "
        "from its own last calls before the final call. More passes settle "
        "the axes further and cost time. Default 2. Alpha.",
    "ops_calibration_min_labels":
        "(int) - The fewest labelled reads each base needs in a cycle for "
        "the library calibration to fit that cycle; a cycle with fewer "
        "keeps the plain unmixing. Default 30. Alpha.",
    "ops_decode_runner_up":
        "(bool) - Let a read one base from a guide count for it when the "
        "mismatched cycle was an uncertain call whose second-brightest "
        "channel is the guide's base. It recovers reads lost to one weak "
        "cycle without accepting arbitrary one-mismatch matches. Default "
        "False. Alpha.",
    "ops_runner_up_margin":
        "(float) - The largest base-call margin, 0 to 1, a mismatched cycle "
        "may carry for the runner-up rule to accept it. Lower accepts fewer "
        "corrected reads. Default 0.1. Alpha.",
    "ops_decode_impute_missing_cycles":
        "(bool) - Let a base from a cycle that could not be measured, "
        "written N, match any base of a guide, under the new decode rules. "
        "Off leaves every read with an N unmatched, which costs the reads "
        "of fields that lost a cycle. Default True. Alpha.",
    "ops_assign_by_cell_territory":
        "(bool) - Give a read that lies almost equally close to two nuclei "
        "to the nearer one instead of to neither. More nuclei get reads; a "
        "read between two touching nuclei can land on the wrong one. "
        "Default False. Alpha.",
    "ops_tie":
        "(float) - How close, in pixels, a read's two nearest nucleus "
        "boundaries may lie before the read is called ambiguous, under the "
        "new decode rules. Ambiguous reads are dropped unless "
        "ops_assign_by_cell_territory is on. Default 1. Alpha.",
    "ops_decode_quality_weighted":
        "(bool) - Weight each read's vote by its base-call quality, so a "
        "clean read outvotes a doubtful one, instead of counting every read "
        "once. Default False. Alpha.",
    "ops_vote_weight_floor":
        "(float) - Added to every quality weight, so a read of quality 0 "
        "still carries a small vote. Higher moves the weighting toward a "
        "plain count. Default 0.05. Alpha.",
    "ops_decode_pixel_fallback":
        "(bool) - For a nucleus no read assigned, call the brightest pixels "
        "near it like reads and assign the guide at least two of them name "
        "exactly. It recovers nuclei whose spots fell below the read "
        "threshold. Default False. Alpha.",
    "ops_pixel_percentile":
        "(float) - Which pixels the pixel fallback calls: those above this "
        "percentile of the field's read score. Lower calls more pixels, "
        "slower and noisier. Default 97. Alpha.",
    "ops_pixel_radius":
        "(float) - How far beyond a nucleus boundary, in pixels, a fallback "
        "pixel may lie and still vote for that nucleus. Default 10. Alpha.",
    "ops_pixel_min_pixels":
        "(int) - How many exactly matching pixels must name a guide before "
        "the pixel fallback assigns it. Pixels of one spot repeat its call, "
        "so this counts pixels, not reads. Default 2. Alpha.",
    "ops_pixel_min_fraction":
        "(float) - The share of a nucleus's matched pixels its guide must "
        "hold, with no tie, for the pixel fallback to assign it. Default "
        "0.6. Alpha.",
    "ops_pixel_quality":
        "(float) - The weakest pixel call, as its worst cycle's margin from "
        "0 to 1, that votes in the pixel fallback. Higher assigns fewer "
        "nuclei at a lower decoy rate. Default 0.1. Alpha.",
    "ops_decoy_library_k":
        "(int) - How many base-shuffled decoy barcodes per guide are "
        "matched alongside the library. A nucleus assigned a decoy is "
        "dropped and counted, which estimates how many real calls are "
        "false; each decode then writes decode_fdr_<well>.json and "
        "per-field stats under results/ops_decode. 0 matches no decoys. "
        "Default 0. Alpha.",
    "ops_fdr_scale":
        "(float) - How much likelier a read is to land on a wrong real "
        "guide than on a decoy, which turns decoy calls into an FDR. 1.268 "
        "was measured for the Funk et al. 2022 library from its guides' "
        "neighbour distances; values below 1 count as 1. Default 1.268. "
        "Alpha.",
    "ops_decode_skip_unplaced_sites":
        "(bool) - Decode only the sites the stitch placed and list the "
        "others in the decode and FDR reports. Off stops the well with an "
        "error when a site was not placed. Default True. Alpha.",
    "ops_store_field_cache":
        "(bool) - Under the new decode rules, write each field's decode "
        "evidence -- every spot, its intensities, its two nearest nuclei "
        "and the fallback pixels -- to <dst_root>/<well>/decode_cache as "
        "one npz per site. The decode QC reads it. A field takes about 13 "
        "MB. Default False. Alpha.",
    "ops_decode_qc":
        "(bool) - After the decode, train a model of what makes a genotype "
        "call a decoy: gradient boosting on a permissive decode of the "
        "stored field evidence, held-out fields and decoys, a sequence-only "
        "control and chance. Writes model_metrics.csv, shap_mean_abs.csv, "
        "permutation_importance.csv and model.json to "
        "results/ops_decode_qc. Turns on ops_store_field_cache with the "
        "spot features. Default False. Alpha.",
    "ops_decode_qc_model":
        "(str) - The false-call model: 'xgboost', the model of the "
        "decode-yield study, needs the optional xgboost package (pip "
        "install xgboost); 'hist_gbm' is scikit-learn's histogram gradient "
        "boosting and needs nothing extra. Default 'xgboost'. Alpha.",
    "ops_decode_qc_features":
        "(str or None) - The false-call model's features, separated by "
        "commas. Empty takes every per-call feature of the evidence table: "
        "read counts, matches and mismatches, read quality, per-cycle "
        "margins, spot shape, background and position. Default None. Alpha.",
    "ops_decode_qc_split":
        "(str) - How calls are held out from the false-call model: 'field' "
        "holds out half the sequencing fields of each plate, so no field is "
        "both learnt and scored. Decoys are split in halves by sequence as "
        "well. Default 'field'. Alpha.",
    "ops_decode_qc_seed":
        "(int) - The seed of the false-call model, its SHAP sample and its "
        "bootstrap intervals. The held-out field draw keeps the study's "
        "seed 7. Default 0. Alpha.",
    "ops_decode_qc_max_calls":
        "(int) - How many held-out calls the SHAP ranking of the false-call "
        "model is taken over, drawn with ops_decode_qc_seed; 0 takes every "
        "held-out call, slower on a plate. Default 20000. Alpha.",
    "ops_genotype_confidence":
        "(bool) - After the decode, give every genotype call a calibrated "
        "probability of being correct, learnt from the decoy calls with "
        "five-fold cross-fitting by field, and add it to ops_barcodes as "
        "genotype_confidence. Writes confidence_reliability.csv and "
        "confidence_metrics.json (AUC, ECE, shuffled-label AUC) to "
        "results/ops_decode_qc. Needs ops_decoy_library_k above 0. Default "
        "False. Alpha.",
    "ops_confidence_model":
        "(str) - Which model's error becomes genotype_confidence: "
        "'hist_gbm' (gradient boosting), 'logistic', 'isotonic' (the "
        "logistic score recalibrated), 'strata' (the decoy rate of matched "
        "read-count strata) or 'auto' (the lowest held-out log-loss). All "
        "are fitted and reported. Default 'hist_gbm'. Alpha.",
    "ops_confidence_calibration":
        "(str) - How a model's decoy probability becomes the error of a "
        "real call: 'target_decoy' scales it by ops_fdr_scale over "
        "ops_decoy_library_k and by the odds of a real call, so the errors "
        "add up to the decoy-estimated false calls; 'none' reports one "
        "minus the decoy probability. Default 'target_decoy'. Alpha.",
    "ops_confidence_bins":
        "(int) - How many bins the reliability curve has; each bin expects "
        "the same number of decoy calls. Default 10. Alpha.",
    "ops_confidence_seed":
        "(int) - The seed of the confidence models' subsampling, the "
        "shuffled-label control and the boosting. The default is the seed "
        "of the reference run. Default 20261107. Alpha.",
    "ops_confidence_min":
        "(float or None) - Drop genotype calls whose genotype_confidence is "
        "below this from ops_barcodes, and report how many were dropped. "
        "Empty keeps every call. Default None. Alpha.",
    "ops_compare_tables":
        "(str or None) - Other pipelines' per-cell genotype calls to "
        "compare with spaCR's, as name=path pairs separated by commas or a "
        "JSON object. Each table has one row per cell with well, site, y "
        "and x in that site's sequencing tile and call (empty when "
        "uncalled). Setting it runs the compare_calls phase after the "
        "decode; results go to results/ops_compare. Default None. Alpha.",
    "ops_compare_match_px":
        "(float) - How far apart, in pixels, two pipelines' nuclei may lie "
        "and still be matched as one cell; matches are mutual nearest "
        "neighbours. Larger matches more cells and risks pairing "
        "neighbours. Default 4. Alpha.",
    "ops_compare_fields":
        "(str or None) - Which fields to compare, as well:site pairs "
        "separated by commas. Empty compares every site the external tables "
        "hold. Default None. Alpha.",
    "ops_aggregate_level":
        "(str) - How a gene profile is reduced from its cells. 'guides' "
        "takes the median of the gene's guide medians, so each guide counts "
        "once; 'cells' takes the median over all of the gene's cells, as "
        "Brieflow does, so a guide with more cells counts more. On Funk "
        "plate A 'cells' raised retrieval mAP from 0.060 to 0.064. Default "
        "'guides'. Alpha.",
    "ops_nt_rows":
        "(str or None) - How non-targeting guides appear in the gene "
        "tables. Empty keeps one row per control gene name; 'barcode' gives "
        "every non-targeting guide its own row, as Brieflow does; 'pseudo' "
        "groups them at random (seed 0) into pseudo-genes of four guides, "
        "processed exactly as genes, so they act as a null. Default None. "
        "Alpha.",
    "ops_guide_efficacy":
        "(str or None) - 'jacks_blend' fits each gene's guides in the "
        "weighted space as x_g = e_g m (the rank-1 guide model of JACKS, "
        "Allen et al. 2019), so a guide that knocks down poorly counts "
        "less, and blends that profile with the all-cell gene profile as "
        "unit vectors. Needs ops_profile_weighting. Default None. Alpha.",
    "ops_guide_efficacy_iters":
        "(int) - The most alternations of the guide-efficacy fit per gene; "
        "it also stops when no efficacy changes by more than 1e-6. Default "
        "20. Alpha.",
    "ops_guide_efficacy_emax":
        "(float) - The cap on a guide's efficacy before the gene's "
        "efficacies are rescaled so the best guide has 1. Default 1. Alpha.",
    "ops_profile_cell_cycle":
        "(str or None) - Resolve the TVN profiles by cell-cycle phase. "
        "Cells are G1 or S/G2 by integrated nuclear DNA, M by the mitotic "
        "call, and each phase is centred on the well's control cells of "
        "that phase. 'concat' joins [G1 | S/G2 | M]; 'G1', 'SG2' or 'M' "
        "keeps one phase. Default None. Alpha.",
    "ops_profile_g1_sg2_split_log2":
        "(float) - Interphase cells this many log2 units above their well's "
        "G1 peak of integrated nuclear DNA are S/G2, the rest G1. 0.30 is "
        "about 1.23 times G1 DNA, the dip of the control histogram on Funk "
        "plate A. Default 0.30. Alpha.",
    "ops_mitotic_source":
        "(str) - Where the M phase of cell-cycle profiles comes from: "
        "'rule' (the DNA rule of ops_mitotic_dna_z, or the classifier when "
        "ops_mitotic_model is set) or 'classifier', which refuses to run "
        "without ops_mitotic_model. Default 'rule'. Alpha.",
    "ops_penetrance":
        "(bool) - Estimate each gene's penetrance, the share of its cells "
        "with its phenotype: a classifier cross-fitted by field scores "
        "every cell against held-out control cells, Storey's estimator "
        "gives the share, with a field-bootstrap interval. Control pseudo- "
        "genes and shuffled labels are run alongside. Needs TVN profiles. "
        "Default False. Alpha.",
    "ops_penetrance_folds":
        "(int) - Fields are split into this many folds by name, and every "
        "cell is scored by a model trained on the other folds. Default 5. "
        "Alpha.",
    "ops_penetrance_nt_pool":
        "(int) - Control cells drawn once, in proportion to each well, as "
        "every gene's reference in the penetrance test. Default 20000. "
        "Alpha.",
    "ops_penetrance_lambda":
        "(float) - The control percentile below which only unaffected cells "
        "are assumed to fall in Storey's estimator; lower gives a less "
        "biased, noisier penetrance. Default 0.5. Alpha.",
    "ops_escapers":
        "(bool) - Write the genes whose cells split into responders and "
        "unaffected escapers: a control-plus-responder mixture beats a "
        "unimodal shift by ops_escaper_dbic and no simulated unimodal gene "
        "makes the same call. Needs ops_penetrance. Default False. Alpha.",
    "ops_escaper_dbic":
        "(float) - The BIC margin by which the mixture must beat a unimodal "
        "shift for an escaper call. Default 10. Alpha.",
    "ops_escaper_min_sep":
        "(float) - The responder mode's least distance from the controls, "
        "in control standard deviations, for an escaper call. Default 2. "
        "Alpha.",
    "ops_phase_tests":
        "(bool) - Test every gene within each cell-cycle phase and pooled, "
        "and call genes that change cells in one phase only. Needs TVN "
        "profiles and the nuclear integrated DNA intensity. Default False. "
        "Alpha.",
    "ops_phase_null":
        "(str) - The null of the phase tests: 'targeting' (random sets of "
        "four targeting guides of four genes) or 'nt' (random sets of four "
        "control guides). Every targeting guide cuts DNA, so against "
        "controls almost every gene differs. Default 'targeting'. Alpha.",
    "ops_profile_shrinkage_prior_df":
        "(float) - The degrees of freedom of the control-guide noise "
        "estimate when it is pooled with a gene's own guide spread; higher "
        "trusts the controls more. Default 4. Alpha.",
    "ops_profile_generic_axes":
        "(str, list or None) - Remove generic axes from the gene profiles: "
        "any of log_cells_per_guide, shift_G1, shift_SG2, shift_M "
        "(regressed out across genes) and essential_axis (the fitness "
        "genes' main direction, removed only when it tracks fitness). "
        "Shifts need cell-cycle phases. Default None. Alpha.",
    "ops_profile_essential_axis_min_rho":
        "(float) - The essential axis is projected out only when the genes' "
        "scores on it correlate with log cells per guide by more than this "
        "absolute Spearman rho. Default 0.30. Alpha.",
    "ops_guide_null_size":
        "(int) - Random guide-set pairs per pair of guide counts in the "
        "guide-level null; P values resolve to 1 / (N + 1). Default 200000. "
        "Alpha.",
    "ops_profile_responder_alpha":
        "(float) - The share of held-out control cells a responder "
        "threshold lets through; the responder fraction is (flagged - "
        "alpha) / (1 - alpha). Default 0.05. Alpha.",
    "ops_profile_space":
        "(str or None) - Profile frozen single-cell embeddings as well: "
        "'embedding:<folder>' reads <folder>/embeddings.npy and "
        "index.parquet (row, well, site, label), joins them to the profiled "
        "interphase cells and writes TVN gene profiles of the embedding; "
        "'fusion:<folder>' also joins them with the measurement profile of "
        "the same cells, each block whitened and given equal weight. Empty "
        "or 'tvn' does neither. Default None. Alpha.",
    "ops_fusion_blocks":
        "(str, list or None) - The blocks of a fusion profile, in order: "
        "'meas' (the TVN measurement profile) and 'emb' (the embedding "
        "profile). Empty uses both, measurement first. Default None. Alpha.",
    "ops_ph_acquisitions":
        "(dict or None) - For a plate whose phenotype channels sit in more "
        "than one acquisition folder, each folder (relative to "
        "phenotype_source) and the channels taken from it, plus an optional "
        "register_on channel, for example {\"phenotype\": \"DAPI,GFP,A594\", "
        "\"phenotype_af750\": \"AF750\", \"register_on\": \"A594\"}. Files of a "
        "folder not listed, or of a channel not listed for its folder, are "
        "not read. The later folders' channels become "
        "ops_ph_secondary_channels when that is empty. Empty: one "
        "acquisition. Default None. Alpha.",
    "ops_ph_pipeline":
        "(bool) - Hand each phenotype field to the measurement workers as "
        "soon as its masks are saved, so the next field is segmented while "
        "this one is measured, instead of segmenting the whole well first. "
        "Masks and tables are the same. Default False. Alpha.",
    "ops_ph_measure_workers":
        "(int or None) - How many phenotype fields are measured at once. "
        "More workers finish a well sooner but hold more fields in memory; "
        "with ops_gpu_vram_cap_gb set, a GPU measurement gets fewer when the "
        "card is full. Empty uses n_workers. Default None. Alpha.",
    "ops_ph_stage_dir":
        "(str or None) - A local folder where each phenotype field's "
        "maximum projection is kept while the field is segmented and "
        "measured, so each raw file on network storage is read once; the "
        "copy is deleted when the field is done. Empty reads the raw files "
        "directly. Default None. Alpha.",
    "ops_ph_stage_ahead":
        "(int) - Most phenotype fields staged in ops_ph_stage_dir at once; "
        "the local space used is at most this many field projections. "
        "Default 48. Alpha.",
    "ops_ph_stage_threads":
        "(int) - How many phenotype fields are read into ops_ph_stage_dir "
        "at once. More threads keep the measurement fed from slow storage "
        "but load the disk and network harder. Default 4. Alpha.",
    "ops_ph_measure_gpu_check":
        "(bool) - With ops_ph_measure_gpu on, measure the first field on "
        "the card and on the CPU before the run and report the columns that "
        "differ by more than 1e-4 (relative). The run goes on either way. "
        "Default True. Alpha.",
    "ops_archive_root":
        "(str or None) - After each well, copy its outputs to this archive "
        "folder (under the plate's name), verify the copy and write a "
        "WELL_DONE marker; PLATE_DONE follows when every well is copied. A "
        "resumed run skips wells whose copy is complete and copies a "
        "damaged one again. Empty: no archive copy. Default None. Alpha.",
    "ops_archive_verify":
        "(str) - How each archive copy is checked: 'md5' compares size and "
        "MD5 checksum with the local file, 'size' the size only (faster, "
        "misses a corrupted copy of the right size). Default 'md5'. Alpha.",
    "ops_archive_delete_local":
        "(bool) - Delete a well's local files once its archive copy is "
        "verified and marked done. Later plate phases (aggregate, tables, "
        "crops) then cannot read them locally. Default False. Alpha.",
    "ops_gpu_admission":
        "(bool) - Wait for the GPU to be free before the run starts: "
        "ops_gpu_idle_checks samples in a row, 10 s apart, with 0 % "
        "utilisation and no running compute process above "
        "ops_gpu_min_process_mib (display and browser processes never "
        "count). START and FINISH go to ~/.spacr/gpu/log, and a STOP file "
        "in dst_root ends the wait or stops the run before the next well. "
        "Default False. Alpha.",
    "ops_gpu_idle_checks":
        "(int) - How many idle GPU samples in a row, ten seconds apart, "
        "ops_gpu_admission waits for before the run starts. Higher waits "
        "longer and is surer the card is free. Default 6. Alpha.",
    "ops_gpu_min_process_mib":
        "(float) - A GPU compute process using more card memory than this, "
        "in MiB, keeps ops_gpu_admission waiting. Default 500. Alpha.",
    "ops_gpu_vram_cap_gb":
        "(float or None) - With ops_gpu_admission and ops_ph_measure_gpu "
        "on, halve the measurement workers when the card's memory in use is "
        "above this many GB when measuring starts. Empty: no cap. Default "
        "None. Alpha.",
    "ops_gpu_ignore_process_patterns":
        "(str or None) - More processes ops_gpu_admission ignores, comma- "
        "separated, besides browsers, editors and display servers: a "
        "pattern starting with - is looked for in the command line, any "
        "other matches the executable's name (* is a wildcard). Default "
        "None. Alpha.",
    "ops_tables_integer_keys":
        "(bool) - Give every row of the ph_tables cell, nucleus and "
        "cytoplasm tables an integer cell_key, indexed, that joins as prcfo "
        "does but faster. Default False. Alpha.",
    "ops_tables_layout":
        "(str) - 'compartments' writes the cell, nucleus and cytoplasm "
        "tables; 'wide' also writes cell_wide, one row per cell with all "
        "three compartments' features, which needs no join. Default "
        "'compartments'. Alpha.",
    "ops_stress_report":
        "(bool) - At the end of the run, time the store as the reference "
        "stress test did (database rows, queries for one field, one gene, "
        "the three-table join and a whole well, random crop reads) and "
        "write results/ops_stress/stress_test.csv. Same as naming the "
        "stress phase. Default False. Alpha.",
}

#: The blurb the module shows above its settings.
OPS_DESCRIPTION = (
    "Take each well of an optical pooled screen's sequencing acquisition "
    "from tiles to tables: stitch its nuclear tiles, segment and number its "
    "nuclei across the whole well, then decode each field's reads and assign "
    "a barcode to every nucleus whose reads agree. Given a phenotype folder "
    "it also records where each phenotype field lands on the stitched well. "
    "The tables are written to measurements.db. Reads are decoded from the "
    "images, not from FASTQ."
)


def ops_defaults(settings=None):
    """The settings the OPS engine reads, with their defaults filled in.

    :param settings: caller's settings, filled in with the defaults.
    :returns: the completed settings dict.
    """
    settings = settings if settings is not None else {}
    for key, value in OPS_DEFAULTS.items():
        settings.setdefault(key, value)
    return settings


def register(replace: bool = False) -> bool:
    """Register the OPS defaults, types, categories and help.

    Idempotent by default: a module imported from both the GUI and a headless
    run registers twice, and that must not be an error.

    :param replace: overwrite an existing registration rather than declining.
    :returns: True when this call did the registering.
    """
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults("ops") and not replace:
        return False
    register_defaults(
        "ops",
        ops_defaults,
        replace=True,
        expected_types=OPS_TYPES,
        tooltips=OPS_TOOLTIPS,
        categories=OPS_CATEGORIES,
        description=OPS_DESCRIPTION,
    )
    return True


SPATIAL_MASK_DEFAULTS: Dict[str, object] = {
    "spatial_source": None,
    "spatial_platform": "auto",
    "spatial_dst": None,
    "spatial_sample": "",
    "spatial_level": 0,
    "spatial_window": "",
    "spatial_window_unit": "um",
    "spatial_tile_size": 2048,
    "spatial_tile_overlap": 160,
    "spatial_segmenter": "cellpose",
    "spatial_stain": "auto",
    "spatial_cellpose_model": "cpsam",
    "spatial_diameter": None,
    "spatial_cell_expansion_um": 5.0,
    "spatial_gpu": False,
}

SPATIAL_MAP_DEFAULTS: Dict[str, object] = {
    "spatial_source": None,
    "spatial_platform": "auto",
    "spatial_masks_dir": None,
    "spatial_bin_um": 2,
    "spatial_min_qv": 20.0,
    "spatial_objects": "cell,nucleus",
    "spatial_pathogen_genes": "",
    "spatial_min_pathogen_transcripts": 1,
    "spatial_gene": "",
    "spatial_anndata": True,
}

SPATIAL_TYPES: Dict[str, object] = {
    "spatial_source": (str, type(None)),
    "spatial_platform": str,
    "spatial_dst": (str, type(None)),
    "spatial_sample": str,
    "spatial_level": int,
    "spatial_window": str,
    "spatial_window_unit": str,
    "spatial_tile_size": int,
    "spatial_tile_overlap": int,
    "spatial_segmenter": str,
    "spatial_stain": str,
    "spatial_cellpose_model": str,
    "spatial_diameter": (float, int, type(None)),
    "spatial_cell_expansion_um": (float, int),
    "spatial_gpu": bool,
    "spatial_masks_dir": (str, type(None)),
    "spatial_bin_um": int,
    "spatial_min_qv": (float, int),
    "spatial_objects": str,
    "spatial_pathogen_genes": str,
    "spatial_min_pathogen_transcripts": int,
    "spatial_gene": str,
    "spatial_anndata": bool,
}

SPATIAL_MASK_CATEGORIES: Dict[str, List[str]] = {
    "Spatial input": [
        "spatial_source", "spatial_platform", "spatial_dst",
        "spatial_sample",
    ],
    "Spatial region & tiling": [
        "spatial_level", "spatial_window", "spatial_window_unit",
        "spatial_tile_size", "spatial_tile_overlap",
    ],
    "Spatial segmentation": [
        "spatial_segmenter", "spatial_stain", "spatial_cellpose_model",
        "spatial_diameter", "spatial_cell_expansion_um", "spatial_gpu",
    ],
}

SPATIAL_MAP_CATEGORIES: Dict[str, List[str]] = {
    "Spatial input": [
        "spatial_source", "spatial_platform", "spatial_masks_dir",
    ],
    "Spatial mapping": [
        "spatial_bin_um", "spatial_min_qv", "spatial_objects",
        "spatial_gene", "spatial_anndata",
    ],
    "Spatial infection": [
        "spatial_pathogen_genes", "spatial_min_pathogen_transcripts",
    ],
}

SPATIAL_TOOLTIPS: Dict[str, str] = {
    "spatial_source":
        "(str or None) - The platform's output folder: a Xenium output "
        "bundle (transcripts.parquet, experiment.xenium and the morphology "
        "image) or a Space Ranger outs folder of a Visium HD run "
        "(binned_outputs/ and the full-resolution H&E or IF image beside "
        "it). Mask generation segments its image; barcode mapping reads its "
        "transcripts or bins. Default None.",
    "spatial_platform":
        "(str) - Which 10x platform wrote spatial_source: auto, xenium, "
        "visium_hd (Visium HD bins) or visium (standard spot-based Visium). "
        "auto reads it from the files present; set it only when a folder "
        "holds more than one layout. Default auto.",
    "spatial_dst":
        "(str or None) - Where the fields are written: merged/, stack/, "
        "masks/ and measurements/measurements.db with the spatial_fields "
        "and spatial_objects tables, the same layout Measure reads after "
        "Mask Generation. Point Measure's src at <spatial_dst>/merged and "
        "Spatial Map Barcodes' spatial_masks_dir at spatial_dst. Empty "
        "writes a spacr_spatial folder inside spatial_source. Default None.",
    "spatial_sample":
        "(str) - The plate name of the fields, the first part of every "
        "file name and measurement key, so a second slide measured into "
        "the same database stays apart. Empty uses the source folder's "
        "name. Default empty.",
    "spatial_level":
        "(int) - Which pyramid level of the slide image to segment and "
        "write: 0 is full resolution, each level up halves it. Level 1 of "
        "a Xenium morphology image (0.425 um per pixel) still resolves "
        "nuclei and segments four times faster. An image without that "
        "level is read at its coarsest one and the scale is recorded. "
        "Default 0.",
    "spatial_window":
        "(str) - Segment only this part of the slide: x, y, width and "
        "height, in spatial_window_unit. Only the image tiles under it are "
        "read. Empty segments the whole slide. Default empty.",
    "spatial_window_unit":
        "(str) - The unit of spatial_window: um (micrometres, through the "
        "image's pixel size) or px (pixels of the chosen level). "
        "Default um.",
    "spatial_tile_size":
        "(int) - Edge of each segmentation window and written field, in "
        "pixels. Larger windows mean fewer fields and seams but more memory "
        "per Cellpose call and larger arrays for Measure. Default 2048.",
    "spatial_tile_overlap":
        "(int) - How far neighbouring windows overlap, in pixels. An object "
        "is kept from a window that saw all of it, so this has to exceed "
        "the largest cell including spatial_cell_expansion_um; objects no "
        "window saw whole are counted in the report and dropped. Raising "
        "it costs time. Default 160.",
    "spatial_segmenter":
        "(str) - How nuclei are found: cellpose runs the Cellpose model in "
        "spatial_cellpose_model; threshold is an Otsu threshold with a "
        "distance-transform watershed, seconds instead of minutes on a CPU "
        "but it merges touching nuclei more often. Default cellpose.",
    "spatial_stain":
        "(str) - Which image the segmenter sees: auto takes the nuclear "
        "(DAPI) channel of a Xenium image and the haematoxylin of an H&E "
        "image; dapi takes the first channel; hematoxylin unmixes "
        "haematoxylin from RGB; rgb gives Cellpose the colour image "
        "itself. Default auto.",
    "spatial_cellpose_model":
        "(str) - The Cellpose model that segments the nuclei: cpsam is the "
        "installed Cellpose's default model, any other name or path is "
        "loaded as that model. Default cpsam.",
    "spatial_diameter":
        "(float or None) - Expected nucleus diameter in pixels of the "
        "chosen level. Given, each window is rescaled so that this "
        "diameter becomes the size Cellpose expects; empty segments at the "
        "image's own scale. Default None.",
    "spatial_cell_expansion_um":
        "(float) - How far each nucleus is grown into a cell, in "
        "micrometres, without crossing a neighbour (Xenium's own default is "
        "5). The grown objects are the cell masks; the nuclei are written "
        "as the nucleus masks with the same labels. 0 writes the nuclei as "
        "the cells. Default 5.",
    "spatial_gpu":
        "(bool) - Run Cellpose on the graphics card when spaCR finds a "
        "usable one. Off keeps the run on the CPU, which is slower but "
        "leaves a shared card alone. Default False.",
    "spatial_masks_dir":
        "(str or None) - The folder Spatial Mask Generation wrote (its "
        "spatial_dst): the fields in merged/ and their places on the slide "
        "in measurements.db. Every transcript or bin is placed in those "
        "fields and the per-cell counts are written into the same "
        "measurements.db, keyed by prcfo, so they join Measure's tables. "
        "Default None.",
    "spatial_bin_um":
        "(int) - Which Visium HD bins are read: 2, 8 or 16 micrometres. "
        "2 um bins are smaller than a cell and give the sharpest "
        "assignment; 8 and 16 um bins are shared out over the cells they "
        "cover by area. Ignored for Xenium. Default 2.",
    "spatial_min_qv":
        "(float) - The lowest Xenium transcript quality (Phred-scaled QV) "
        "kept. 20 is the threshold of Xenium's own cell-feature matrix; "
        "lower keeps more transcripts and more misreads. Ignored for "
        "Visium HD. Default 20.",
    "spatial_objects":
        "(str) - Which mask planes the reads are counted into, separated "
        "by commas: cell, nucleus, or both. Each gets an "
        "<object>_expression table in measurements.db. Default "
        "cell,nucleus.",
    "spatial_pathogen_genes":
        "(str) - The pathogen's own probes or genes, separated by commas; a "
        "name ending in * matches every gene starting with the rest "
        "(CHIKV*). A cell holding at least spatial_min_pathogen_transcripts "
        "of them is infected; they are left out of the host comparisons. "
        "Empty skips the infection analysis. Default empty.",
    "spatial_min_pathogen_transcripts":
        "(int) - How many pathogen transcripts (Visium HD: UMI shares) make "
        "a cell infected. Raising it from 1 removes cells that carry a "
        "stray viral read from a neighbour. Default 1.",
    "spatial_gene":
        "(str) - A gene whose transcripts or bins colour the overlay "
        "figure drawn over the field with the most cells. Empty draws them "
        "uncoloured. Default empty.",
    "spatial_anndata":
        "(bool) - Also write the per-cell counts as an .h5ad file with the "
        "slide coordinates in obsm['spatial'], for scanpy or squidpy; needs "
        "the anndata package. Default True.",
}

SPATIAL_MASK_DESCRIPTION = (
    "Segment a whole Xenium morphology or Visium HD H&E slide: the slide is "
    "cut into overlapping windows, only the image tiles under each window "
    "are read, nuclei are found and grown into cells, objects seen whole "
    "by more than one window are sewn into one, and every window is "
    "written as a field in merged/ with its masks, the layout Measure "
    "reads after Mask Generation. The fields' places on the slide go to "
    "measurements.db, for Spatial Map Barcodes."
)

SPATIAL_MAP_DESCRIPTION = (
    "Read every Xenium transcript or Visium HD bin, put it in the pixel "
    "frame of the fields Spatial Mask Generation wrote (Xenium micrometres "
    "through the morphology pixel size, Visium HD through the "
    "full-resolution image) and count it into the cell or nucleus it falls "
    "in. Per-cell expression tables are written into measurements.db keyed "
    "like Measure's, with infection calls from pathogen genes, "
    "infected-versus-uninfected tests and an overlay."
)


def spatial_mask_defaults(settings=None):
    """The settings the Spatial Mask Generation engine reads, filled in.

    :param settings: caller's settings, filled in with the defaults.
    :returns: the completed settings dict.
    """
    settings = settings if settings is not None else {}
    for key, value in SPATIAL_MASK_DEFAULTS.items():
        settings.setdefault(key, value)
    return settings


def spatial_map_defaults(settings=None):
    """The settings the Spatial Map Barcodes engine reads, filled in.

    :param settings: caller's settings, filled in with the defaults.
    :returns: the completed settings dict.
    """
    settings = settings if settings is not None else {}
    for key, value in SPATIAL_MAP_DEFAULTS.items():
        settings.setdefault(key, value)
    return settings


def register_spatial(replace: bool = False) -> bool:
    """Register both spatial submodules' defaults, types, categories and help.

    :param replace: overwrite existing registrations rather than declining.
    :returns: True when this call did the registering.
    """
    from .settings import has_registered_defaults, register_defaults

    done = False
    for key, factory, keys, cats, text in (
            ("spatial_masks", spatial_mask_defaults, SPATIAL_MASK_DEFAULTS,
             SPATIAL_MASK_CATEGORIES, SPATIAL_MASK_DESCRIPTION),
            ("spatial_map", spatial_map_defaults, SPATIAL_MAP_DEFAULTS,
             SPATIAL_MAP_CATEGORIES, SPATIAL_MAP_DESCRIPTION)):
        if has_registered_defaults(key) and not replace:
            continue
        register_defaults(
            key, factory, replace=True,
            expected_types={k: SPATIAL_TYPES[k] for k in keys},
            tooltips={k: SPATIAL_TOOLTIPS[k] for k in keys},
            categories=cats, description=text)
        done = True
    return done


register()
register_spatial()
