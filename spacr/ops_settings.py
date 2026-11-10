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
        "background registration by more than 2 pixels; 'background' "
        "trusts the background alone, the earlier behaviour. On the Funk "
        "et al. 2022 plates the background registration left one cycle of "
        "each weak well at zero shift while it had drifted 10 to 25 pixels, "
        "and that cycle carried most of the well's base-call errors. "
        "Default 'spots'.",
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
    "ops_profile_weighting":
        "(str or None) - 'contrastive_wiener' weights the TVN gene profiles "
        "by how far each direction rises above guide noise: directions "
        "solving Sigma_genes v = lambda Sigma_NT v, the noise the Ledoit- "
        "Wolf covariance of the non-targeting guides, each weighted by "
        "max((lambda - 1) / lambda, 0). Refused when fewer than two "
        "directions rise above the noise, and not stacked with "
        "ops_profile_shrinkage. Default None. Alpha.",
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
    "ops_profile_shrinkage":
        "(str or None) - 'eb_nt' shrinks each gene profile toward the "
        "controls by how reliable its guides are: the control guides set "
        "the noise, each gene's own guide spread is pooled with it, and the "
        "profile is multiplied by tau2 / (tau2 + V / K) per dimension. Not "
        "stacked with ops_profile_weighting. Default None. Alpha.",
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
    "ops_profile_similarity":
        "(str or None) - How gene-gene similarity is written besides the "
        "gene tables: 'guide_level' (mean guide-guide cosine with a P value "
        "per pair from random guide sets), 'responder' or "
        "'sliced_wasserstein' (responder profiles, energy distance and a "
        "sliced Wasserstein embedding of each gene's cells). Default None. "
        "Alpha.",
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
