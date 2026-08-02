"""Defaults, types, categories, descriptions, and validation for settings."""

import os, ast

#from wsgiref import types

#from spacr_nightly.spacr.build.lib.spacr import settings


DEFAULT_BARCODE_REGEX = (
    r"^(?P<columnID>.{8})TGCTG.*TAAAC"
    r"(?P<grna>.{20,21})AACTT.*AGAAG(?P<rowID>.{8}).*"
)

_BUNDLED_BARCODE_FILES = {
    "column": "barcodes_column.csv",
    "grna": "barcodes_grna.csv",
    "row": "barcodes_row.csv",
}


def _default_worker_count(reserve=0):
    """Return a usable worker default while leaving ``reserve`` CPU cores free.

    ``os.cpu_count()`` may be ``None`` and small machines or hosted runners
    commonly expose only two or four cores.  Direct subtraction therefore
    made the shipped mask default zero on GitHub Actions, after which spaCR's
    own preflight correctly rejected it.
    """
    cores = os.cpu_count() or 1
    return max(1, int(cores) - max(0, int(reserve)))


def bundled_barcode_path(kind):
    """Return the installed CSV path for a bundled barcode reference.

    :param kind: ``'column'``, ``'grna'`` or ``'row'``.
    :returns: absolute path to the packaged CSV.
    :raises ValueError: when ``kind`` is not a bundled reference type.
    """
    try:
        filename = _BUNDLED_BARCODE_FILES[str(kind).lower()]
    except KeyError as exc:
        choices = ", ".join(_BUNDLED_BARCODE_FILES)
        raise ValueError(
            f"Unknown barcode reference {kind!r}; choose {choices}."
        ) from exc
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "resources", "data", filename)
    )

def set_default_plot_merge_settings():
    """Return the default settings dict for plotting merged mask overlays.

    :returns: dict populated with the default ``plot_merge`` parameters
        (channel dimensions, backgrounds, overlay behaviour, colormap, etc.).
    """
    settings = {}
    settings.setdefault('pathogen_limit', 10)
    settings.setdefault('nuclei_limit', 1)
    settings.setdefault('remove_background', False)
    settings.setdefault('filter_min_max', None)
    settings.setdefault('channel_dims', [0,1,2,3])
    settings.setdefault('backgrounds', [100,100,100,100])
    settings.setdefault('cell_mask_dim', 4)
    settings.setdefault('nucleus_mask_dim', 5)
    settings.setdefault('pathogen_mask_dim', 6)
    settings.setdefault('outline_thickness', 3)
    settings.setdefault('outline_color', 'gbr')
    settings.setdefault('overlay_chans', [1,2,3])
    settings.setdefault('overlay', True)
    settings.setdefault('normalization_percentiles', [2,98])
    settings.setdefault('normalize', True)
    settings.setdefault('print_object_number', True)
    settings.setdefault('nr', 1)
    settings.setdefault('figuresize', 10)
    settings.setdefault('cmap', 'inferno')
    settings.setdefault('verbose', True)
    return settings

def set_default_settings_preprocess_generate_masks(settings=None):
    """Populate default settings for the preprocess/generate-masks pipeline.

    Fills channel, Cellpose, plot, timelapse, organelle and post-processing
    parameters used by ``preprocess_generate_masks``.

    :param settings: optional dict to fill in place; a new dict is created if None.
    :returns: the settings dict with defaults applied.
    """
    if settings is None:
        settings = {}
    # ── pipeline flavour ──────────────────────────────────────────────
    # 'v1' — the original multi-copy chain (rename → channel folders →
    #        npy → npz → mask npy → merged/). Stable, well-tested.
    # 'v2' — streaming pipeline (spacr.pipeline_v2). Reads originals
    #        directly, writes one npy per field to merged/ with masks
    #        appended in-place. ~60-80% less disk. Opt-in for one
    #        release, then default.
    # Default to the v1 disk-based pipeline: it is the fully-tested path and
    # produces the channel/stack/mask_stack folder layout the rest of spaCR
    # (measure, annotate, downstream tools, the e2e suite) depends on. The v2
    # streaming pipeline (no .npz on disk) is opt-in via pipeline_style='v2'
    # until it reproduces that layout and fixes real-data channel indexing.
    settings.setdefault('pipeline_style', 'v1')
    # v2-only: how many field stacks to load into memory per Cellpose
    # batch. Bigger = faster, more RAM.
    settings.setdefault('batch_fields', 8)
    # v2-only: keep the in-memory NPZ batch on disk under merged/_scratch/
    # for debugging. Default False → NPZ never touches disk.
    settings.setdefault('keep_npz', False)

    settings.setdefault('denoise', False)
    settings.setdefault('src', 'path')
    settings.setdefault('delete_intermediate', False)
    settings.setdefault('preprocess', True)
    settings.setdefault('masks', True)
    settings.setdefault('save', True)
    settings.setdefault('consolidate', False)
    settings.setdefault('batch_size', 50)
    settings.setdefault('test_mode', False)
    # Validate-only: preprocess_generate_masks runs the pre-flight checks in
    # spacr.validate, prints the report plus the plan, and returns before any
    # model loads or any file is written.
    settings.setdefault('dry_run', False)
    settings.setdefault('test_images', 10)
    settings.setdefault('magnification', 20)
    settings.setdefault('custom_regex', None)
    settings.setdefault('metadata_type', 'cellvoyager')
    settings.setdefault('n_jobs', _default_worker_count(reserve=4))
    settings.setdefault('randomize', True)
    settings.setdefault('verbose', True)
    settings.setdefault('remove_background_cell', False)
    settings.setdefault('remove_background_nucleus', False)
    settings.setdefault('remove_background_pathogen', False)
    
    settings.setdefault('cell_diameter', None)
    settings.setdefault('nucleus_diameter', None)
    settings.setdefault('pathogen_diameter', None)
    settings.setdefault('diameter_estimate_n_fields', 5)

    # Cellpose 4 ships one stock model, so the only real choice these keys
    # carry is "stock weights" vs "the checkpoint I trained". A legacy value
    # in an old settings file is mapped forward by
    # normalize_cellpose_model_name when _get_object_settings reads it.
    settings.setdefault('cell_model_name', 'cpsam')
    settings.setdefault('nucleus_model_name', 'cpsam')
    settings.setdefault('pathogen_model_name', 'cpsam')

    # Segmentation QC — scored on the masks the moment they exist, so a plate
    # that segmented badly is caught here rather than after measure_crop has
    # spent hours on it. 'report' computes, saves and prints; it never filters.
    # The thresholds are spacr.seg_qc.QC_DEFAULTS, documented there and in the
    # tooltips below.
    settings.setdefault('seg_qc', 'report')
    settings.setdefault('seg_qc_min_objects', 10)
    settings.setdefault('seg_qc_count_ratio', 0.25)
    settings.setdefault('seg_qc_size_ratio', 1.4)
    settings.setdefault('seg_qc_border_fraction', 0.3)
    settings.setdefault('seg_qc_outlier_mad', 5.0)
    settings.setdefault('seg_qc_outlier_fraction', 0.15)
    settings.setdefault('seg_qc_foreground_fraction', 0.35)
    settings.setdefault('seg_qc_split_ratio', 2.0)
    settings.setdefault('seg_qc_min_diameter', 5.0)
    settings.setdefault('seg_qc_tiny_fraction', 0.3)
    settings.setdefault('seg_qc_max_object_fraction', 0.25)
    settings.setdefault('seg_qc_plate_fail_fraction', 0.1)

    # Channel settings
    settings.setdefault('cell_channel', None)
    settings.setdefault('nucleus_channel', None)
    settings.setdefault('pathogen_channel', None)
    settings.setdefault('channels', [0,1,2,3])
    settings.setdefault('pathogen_background', 100)
    settings.setdefault('pathogen_Signal_to_noise', 10)
    settings.setdefault('pathogen_CP_prob', 0)
    settings.setdefault('cell_background', 100)
    settings.setdefault('cell_Signal_to_noise', 10)
    settings.setdefault('cell_CP_prob', 0)
    settings.setdefault('nucleus_background', 100)
    settings.setdefault('nucleus_Signal_to_noise', 10)
    settings.setdefault('nucleus_CP_prob', 0)
    settings.setdefault('nucleus_FT', 1.0)
    settings.setdefault('cell_FT', 1.0)
    settings.setdefault('pathogen_FT', 1.0)
    
    # Plot settings
    settings.setdefault('plot', False)
    settings.setdefault('figuresize', 10)
    settings.setdefault('cmap', 'inferno')
    settings.setdefault('normalize', True)
    settings.setdefault('normalize_plots', True)
    settings.setdefault('examples_to_plot', 1)

    # Analasys settings
    settings.setdefault('pathogen_model', None)
    settings.setdefault('merge_pathogens', False)
    settings.setdefault('filter', False)
    settings.setdefault('lower_percentile', 2)

    # Timelapse settings
    settings.setdefault('timelapse', False)
    settings.setdefault('fps', 2)
    settings.setdefault('timelapse_displacement', None)
    settings.setdefault('timelapse_memory', 3)
    settings.setdefault('timelapse_frame_limits', [5,])
    settings.setdefault('timelapse_remove_transient', False)
    settings.setdefault('timelapse_mode', 'trackastra')
    settings.setdefault('trackastra_model', 'general_2d')
    settings.setdefault('trackastra_linking', 'greedy')
    settings.setdefault('ultrack_max_distance', 25.0)
    settings.setdefault('ultrack_division_weight', -0.1)
    settings.setdefault('ultrack_contour_sigma', 0.0)
    settings.setdefault('ultrack_n_workers', 1)
    settings.setdefault('timelapse_objects', ['cell'])

    # Misc settings
    settings.setdefault('all_to_mip', False)
    settings.setdefault('save_original_images', True)
    settings.setdefault('keep_intermediate', False)
    settings.setdefault('keep_original_images', False)
    settings.setdefault('compression', 'lzw')
    settings.setdefault('upscale', False)
    settings.setdefault('upscale_factor', 2.0)
    settings.setdefault('adjust_cells', False)

    # 3D (Beta). Off by default and read only through
    # spacr.zstack.plan_from_settings, which returns None whenever `z_stack`
    # is falsy -- so with these defaults not one line of z code executes and
    # the 2-D path is bit-identical to a run from before these keys existed.
    settings.setdefault('z_stack', False)
    settings.setdefault('z_segmentation_mode', 'project')
    settings.setdefault('z_axis', None)
    settings.setdefault('z_projection', 'max')
    settings.setdefault('anisotropy', None)
    settings.setdefault('voxel_size_z_um', None)
    settings.setdefault('voxel_size_xy_um', None)
    settings.setdefault('stitch_threshold', 0.25)

    # 4D (Beta). The time axis on top of the z axis, read only through
    # spacr.zstack.plan_4d_from_settings, which returns None whenever
    # `t_stack` is falsy -- so with these defaults not one line of 4-D code
    # executes and both the 2-D and the 3-D path stay bit-identical to a run
    # from before these keys existed. `t_axis_order` deliberately has no
    # usable default: (T,Z,Y,X) and (Z,T,Y,X) are both written by real
    # microscopes and a 4-D shape cannot tell them apart, so a run that turns
    # t_stack on without saying which it has is stopped rather than guessed
    # at -- guessing wrong links objects across z and calls it a trajectory.
    settings.setdefault('t_stack', False)
    settings.setdefault('t_axis_order', None)
    settings.setdefault('t_axis', None)
    settings.setdefault('frame_interval_s', None)
    settings.setdefault('t_track_backend', 'iou')
    settings.setdefault('t_link_threshold', 0.25)
    settings.setdefault('t_max_displacement_px', None)
    settings.setdefault('t_max_displacement_um', None)
    settings.setdefault('t_project_for_tracking', False)
    #settings.setdefault('use_sam_cell', False)
    #settings.setdefault('use_sam_nucleus', False)
    #settings.setdefault('use_sam_pathogen', False)
    
    #organelle settings
    settings.setdefault('organelle_channel', None)
    settings.setdefault('organelle_morphology', 'spots')
    settings.setdefault('organelle_method', 'otsu')
    settings.setdefault('organelle_diameter', 30)
    settings.setdefault('organelle_model_name','cpsam' )
    settings.setdefault('organelle_min_size', 10)
    settings.setdefault('organelle_max_size', None)
    settings.setdefault('organelle_remove_border',False )
    settings.setdefault('organelle_log_min_sigma', 1)
    settings.setdefault('organelle_log_max_sigma', 10)
    settings.setdefault('organelle_log_num_sigma', 10)
    settings.setdefault('organelle_log_threshold', 0.01)
    settings.setdefault('organelle_tophat_radius', 5)
    settings.setdefault('organelle_watershed_spots', True)
    settings.setdefault('organelle_ridge_sigmas', [1, 2, 3])
    settings.setdefault('organelle_ridge_filter', 'frangi')
    settings.setdefault('organelle_skeletonize', False)
    settings.setdefault('organelle_network_threshold','otsu' )
    settings.setdefault('organelle_adaptive_block_size', 51)
    settings.setdefault('organelle_adaptive_offset', 5)
    settings.setdefault('organelle_morph_radius', 3)
    settings.setdefault('organelle_fill_holes', 64)
    settings.setdefault('organelle_CP_prob', 0.0)
    settings.setdefault('organelle_FT', 0.4)
    settings.setdefault('organelle_resample', True)
    
    # Preprocessing
    settings.setdefault('organelle_rolling_ball', False)
    settings.setdefault('organelle_rolling_ball_radius', 50)
    settings.setdefault('organelle_clahe', False)
    settings.setdefault('organelle_clahe_clip_limit', 0.01)
    settings.setdefault('organelle_mask_within_cells', False)

    # DoG (spots)
    settings.setdefault('organelle_dog_sigma_low', 1.0)
    settings.setdefault('organelle_dog_sigma_high', 3.0)

    # Hysteresis (network)
    settings.setdefault('organelle_hysteresis_low', 0.2)
    settings.setdefault('organelle_hysteresis_high', 0.6)

    # U-Net (network)
    settings.setdefault('organelle_unet_model_path', None)
    settings.setdefault('organelle_unet_threshold', 0.5)

    # Ring
    settings.setdefault('organelle_ring_sigma_inner', 1.0)
    settings.setdefault('organelle_ring_sigma_outer', 3.0)
    settings.setdefault('organelle_ring_min_prominence', 0.1)
    settings.setdefault('organelle_ring_fill_method', 'flood')
    settings.setdefault('summarize_organelles_by', 'cell')

    #merge_split
    settings.setdefault('cell_perimeter_fraction', 0)
    settings.setdefault('nucleus_perimeter_fraction',  0)
    settings.setdefault('pathogen_perimeter_fraction',  0)
    settings.setdefault('organelle_perimeter_fraction', 0)
    settings.setdefault('cell_intensity_merge',False)
    settings.setdefault('nucleus_intensity_merge', False)
    settings.setdefault('pathogen_intensity_merge', False)
    settings.setdefault('organelle_intensity_merge', False)
    settings.setdefault('cell_intensity_split', False)
    settings.setdefault('nucleus_intensity_split', False)
    settings.setdefault('pathogen_intensity_split', False)
    settings.setdefault('organelle_intensity_split', False)
    settings.setdefault('cell_area_multiplier',2.0)
    settings.setdefault('nucleus_area_multiplier', 2.0)
    settings.setdefault('pathogen_area_multiplier', 2.0)
    settings.setdefault('organelle_area_multiplier', 2.0)
    settings.setdefault('cell_min_distance', 10)
    settings.setdefault('nucleus_min_distance', 10)
    settings.setdefault('pathogen_min_distance', 10)
    settings.setdefault('organelle_min_distance', 10)
    settings.setdefault('cell_min_object_area', 100)
    settings.setdefault('nucleus_min_object_area', 100)
    settings.setdefault('pathogen_min_object_area', 100)
    settings.setdefault('organelle_min_object_area', 100)
    settings.setdefault('cell_intensity_threshold_method', 'mean')
    settings.setdefault('nucleus_intensity_threshold_method', 'mean')
    settings.setdefault('pathogen_intensity_threshold_method', 'mean')
    settings.setdefault('organelle_intensity_threshold_method', 'mean')
    settings.setdefault('cell_intensity_percentile', 75)
    settings.setdefault('nucleus_intensity_percentile', 75)
    settings.setdefault('pathogen_intensity_percentile', 75)
    settings.setdefault('organelle_intensity_percentile', 75)
    #settings.setdefault('postprocess_cell_masks', False)
    #settings.setdefault('postprocess_nucleus_masks', False)
    #settings.setdefault('postprocess_pathogen_masks', False)
    #settings.setdefault('postprocess_organelle_masks', False)
    settings.setdefault('cell_min_area', 0)
    settings.setdefault('nucleus_min_area', 0)
    settings.setdefault('pathogen_min_area', 0)
    settings.setdefault('organelle_min_area', 0)
    settings.setdefault('cell_max_area', 0)
    settings.setdefault('nucleus_max_area', 0)
    settings.setdefault('pathogen_max_area', 0)
    settings.setdefault('organelle_max_area', 0)
    settings.setdefault('cell_remove_border_objects', False)
    settings.setdefault('nucleus_remove_border_objects', False)
    settings.setdefault('pathogen_remove_border_objects', False)
    settings.setdefault('organelle_remove_border_objects', False)
    settings.setdefault('cell_min_intensity_percentile', 0)
    settings.setdefault('nucleus_min_intensity_percentile', 0)
    settings.setdefault('pathogen_min_intensity_percentile', 0)
    settings.setdefault('organelle_min_intensity_percentile', 0)
    settings.setdefault('cell_max_intensity_percentile', 100)
    settings.setdefault('nucleus_max_intensity_percentile', 100)
    settings.setdefault('pathogen_max_intensity_percentile', 100)
    settings.setdefault('organelle_max_intensity_percentile', 100)
    # NOTE: `timelapse`, the `timelapse_*` knobs above and `motility_analysis`
    # are deliberately still defaulted here even though the Mask *module* no
    # longer surfaces them in its GUI (they moved to the standalone Timelapse
    # and Motility Assay modules — see get_timelapse_settings and
    # get_automated_motility_assay_default_settings). spacr.object reads
    # settings['timelapse'] on every mask run and settings['motility_analysis']
    # inside the timelapse branch, and old settings CSVs still carry both, so
    # removing the defaults would break the pipeline and every archived CSV.
    settings.setdefault('motility_analysis', False)



    # Fail-loud policy. None means "not set here" and defers to the
    # SPACR_STRICT_ERRORS environment variable, which is how a cluster turns
    # it on for a whole batch without editing every settings file. True/False
    # here is an explicit per-run choice and wins over the environment.
    settings.setdefault('strict_errors', None)
    settings.setdefault('max_failure_rate', None)
    # Continue an interrupted run instead of starting over. Opt-in:
    # spacr.resume validates what is already on disk rather than trusting it,
    # and clears a field's existing rows before re-measuring it.
    settings.setdefault('resume', False)
    return settings


def get_timelapse_settings(settings=None):
    """Return default settings for the standalone Timelapse module.

    The Timelapse module is mask generation run over a time series: the same
    preprocessing + Cellpose segmentation as the Mask module, followed by
    frame-to-frame linking of the objects named in ``timelapse_objects`` and
    per-channel movie export. It therefore takes the full
    ``set_default_settings_preprocess_generate_masks`` dict with ``timelapse``
    forced on — the flag is what the module *is*, not something to configure.

    :param settings: optional dict to fill in place; a new dict is created if None.
    :returns: the settings dict with defaults applied and ``timelapse`` True.
    """
    if settings is None:
        settings = {}
    settings = set_default_settings_preprocess_generate_masks(settings)
    settings['timelapse'] = True
    return settings


def set_default_plot_data_from_db(settings):
    """Populate default settings for plotting data pulled from a measurements DB.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('database', 'measurements.db')
    settings.setdefault('graph_name', 'Figure_1')
    settings.setdefault('table_names', ['cell', 'cytoplasm', 'nucleus', 'pathogen'])
    settings.setdefault('data_column', 'recruitment')
    settings.setdefault('grouping_column', 'condition')
    settings.setdefault('cell_types', ['Hela'])
    settings.setdefault('cell_plate_metadata', None)
    settings.setdefault('pathogen_types', None)
    settings.setdefault('pathogen_plate_metadata', None)
    settings.setdefault('treatments', None)
    settings.setdefault('treatment_plate_metadata', None)
    settings.setdefault('graph_type', 'jitter')
    settings.setdefault('theme', 'deep')
    settings.setdefault('save', True)
    settings.setdefault('y_lim', None)
    settings.setdefault('verbose', False)
    settings.setdefault('channel_of_interest', 1)
    settings.setdefault('nuclei_limit', 2)
    settings.setdefault('pathogen_limit', 3)
    settings.setdefault('representation', 'well')
    settings.setdefault('uninfected', False)
    return settings

def set_default_settings_preprocess_img_data(settings):
    """Populate default settings for the image-preprocessing step.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('metadata_type', 'cellvoyager')
    settings.setdefault('custom_regex', None)
    settings.setdefault('nr', 1)
    settings.setdefault('plot', True)
    settings.setdefault('batch_size', 50)
    settings.setdefault('timelapse', False)
    settings.setdefault('lower_percentile', 2)
    settings.setdefault('randomize', True)
    settings.setdefault('all_to_mip', False)
    settings.setdefault('save_original_images', True)
    settings.setdefault('keep_intermediate', False)
    settings.setdefault('keep_original_images', False)
    settings.setdefault('compression', 'lzw')
    settings.setdefault('cmap', 'inferno')
    settings.setdefault('figuresize', 10)
    settings.setdefault('normalize', True)
    settings.setdefault('save_dtype', 'uint16')
    settings.setdefault('test_mode', False)
    settings.setdefault('test_images', 10)
    settings.setdefault('random_test', True)
    settings.setdefault('fps', 2)
    return settings


#: What a Cellpose model setting may be, now that Cellpose 4 exists: the one
#: stock model, or a path to a checkpoint the user trained themselves. There
#: is no third option, so no dropdown in spaCR offers one.
CELLPOSE_MODEL_CHOICES = ('cpsam',)


def normalize_cellpose_model_name(value, object_type=None, key=None):
    """Map a stored Cellpose model setting forward onto what Cellpose 4 has.

    Cellpose 4 ships exactly one stock model, ``cpsam``
    (``cellpose.models.MODEL_NAMES == ['cpsam']``), and
    ``CellposeModel(model_type=...)`` is accepted-and-ignored. So 'cyto',
    'cyto2', 'cyto3' and 'nuclei' are not four choices, they are four spellings
    of cpsam — offering them in a dropdown invited users to tune a setting that
    does nothing.

    They are kept as accepted-but-mapped ALIASES rather than removed outright:
    settings CSVs written years ago must still load. What changes is that they
    are mapped here, on the way in, instead of being carried around as if they
    still meant something. A path to a user-trained checkpoint is passed
    through untouched — that is the one model choice that is still real.

    :param value: the stored setting, e.g. 'cyto2' or '/models/my_cells.pth'.
    :param object_type: 'cell'/'nucleus'/'pathogen'/'organelle' if known; used
        only to make the substitution notice name the right object.
    :param key: settings key the value came from, for the notice.
    :returns: 'cpsam', or the checkpoint path unchanged.
    """
    from .utils import LEGACY_CELLPOSE_MODELS, CPSAM_MODEL, _report_cellpose_once

    if value is None:
        return CPSAM_MODEL
    name = str(value).strip()
    if not name:
        return CPSAM_MODEL
    if name in LEGACY_CELLPOSE_MODELS:
        where = f" ({key})" if key else ""
        clause = f" for {object_type}" if object_type else ""
        _report_cellpose_once(
            ('settings-legacy', name, object_type, key),
            f"Cellpose model {name!r}{where} predates Cellpose-SAM and is no "
            f"longer available; using 'cpsam'{clause}.")
        return CPSAM_MODEL
    return name


def _get_object_settings(object_type, settings):
    """Build per-object Cellpose/segmentation settings for cell/nucleus/pathogen."""
    from .utils import _get_diam
    object_settings = {}

    object_settings['diameter'] = _get_diam(settings['magnification'], obj=object_type)
    object_settings['minimum_size'] = (object_settings['diameter']**2)/4
    object_settings['maximum_size'] = (object_settings['diameter']**2)*10
    object_settings['merge'] = False
    object_settings['resample'] = True
    object_settings['remove_border_objects'] = False
    # 'cpsam' unless the user pointed at their own checkpoint. A legacy name
    # from an old settings file is mapped forward here rather than carried
    # into segmentation as if it still selected different weights.
    object_settings['model_name'] = normalize_cellpose_model_name(
        settings.get(f'{object_type}_model_name'),
        object_type=object_type, key=f'{object_type}_model_name')

    if object_type == 'cell':
        object_settings['min_size'] = settings['cell_min_area']
        object_settings['filter_size'] = False
        object_settings['filter_intensity'] = False
        object_settings['restore_type'] = settings.get('cell_restore_type', None)
        if settings['cell_diameter'] is not None:
            try:
                # Coerce — CSV-imported settings arrive as strings ("30.0").
                object_settings['diameter'] = float(settings['cell_diameter'])
                object_settings['minimum_size'] = (object_settings['diameter']**2)/4
                object_settings['maximum_size'] = (object_settings['diameter']**2)*10
            except (TypeError, ValueError):
                print(f'Cell diameter must be an integer or float, got {settings["cell_diameter"]!r}')

    elif object_type == 'nucleus':
        object_settings['min_size'] = settings['nucleus_min_area']
        object_settings['filter_size'] = False
        object_settings['filter_intensity'] = False
        object_settings['restore_type'] = settings.get('nucleus_restore_type', None)
        
        if settings['nucleus_diameter'] is not None:
            try:
                object_settings['diameter'] = float(settings['nucleus_diameter'])
                object_settings['minimum_size'] = (object_settings['diameter']**2)/4
                object_settings['maximum_size'] = (object_settings['diameter']**2)*10
            except (TypeError, ValueError):
                print(f'Nucleus diameter must be an integer or float, got {settings["nucleus_diameter"]!r}')
        # (A commented-out `use_sam_nucleus -> model_name = 'sam'` sat here.
        #  There is no model named 'sam': Cellpose 4 IS SAM and calls its one
        #  model 'cpsam', which is already what nucleus_model_name defaults
        #  to. Removed rather than left as a suggestion that would not work.)

    elif object_type == 'pathogen':
        object_settings['min_size'] = settings['pathogen_min_area']
        object_settings['filter_size'] = False
        object_settings['filter_intensity'] = False
        object_settings['resample'] = False
        object_settings['restore_type'] = settings.get('pathogen_restore_type', None)
        object_settings['merge'] = settings['merge_pathogens']
        
        if settings['pathogen_diameter'] is not None:
            try:
                object_settings['diameter'] = float(settings['pathogen_diameter'])
                object_settings['minimum_size'] = (object_settings['diameter']**2)/4
                object_settings['maximum_size'] = (object_settings['diameter']**2)*10
            except (TypeError, ValueError):
                print(f'Pathogen diameter must be an integer or float, got {settings["pathogen_diameter"]!r}')

        # (Same for the commented-out `use_sam_pathogen` branch — see above.)


    else:
        print(f'Object type: {object_type} not supported. Supported object types are : cell, nucleus and pathogen')
        
    if settings['verbose']:
        print(object_settings)
        
    return object_settings 

def set_default_umap_image_settings(settings=None):
    """Return the default settings for UMAP/tSNE image-embedding plots.

    :param settings: optional dict to fill in place; a new dict is created if None.
    :returns: the settings dict with defaults applied.
    """
    if settings is None:
        settings = {}
    settings.setdefault('src', 'path')
    settings.setdefault('row_limit', 1000)
    settings.setdefault('tables', ['cell', 'cytoplasm', 'nucleus', 'pathogen'])
    settings.setdefault('visualize', 'cell')
    settings.setdefault('image_nr', 16)
    settings.setdefault('dot_size', 50)
    settings.setdefault('point_color', 'cluster')
    settings.setdefault('point_alpha', 0.65)
    settings.setdefault('outline_width', 1.0)
    settings.setdefault('umap_canvas_width', 900)
    settings.setdefault('umap_sidebar_width', 280)
    settings.setdefault('n_neighbors', 1000)
    settings.setdefault('min_dist', 0.1)
    settings.setdefault('metric', 'euclidean')
    settings.setdefault('eps', 0.9)
    settings.setdefault('min_samples', 100)
    settings.setdefault('filter_by', 'channel_0')
    settings.setdefault('img_zoom', 0.5)
    settings.setdefault('plot_by_cluster', True)
    settings.setdefault('plot_cluster_grids', True)
    settings.setdefault('remove_cluster_noise', True)
    settings.setdefault('remove_highly_correlated', True)
    settings.setdefault('log_data', False)
    settings.setdefault('figuresize', 10)
    settings.setdefault('black_background', True)
    settings.setdefault('remove_image_canvas', False)
    settings.setdefault('plot_outlines', True)
    settings.setdefault('plot_points', True)
    settings.setdefault('smooth_lines', True)
    settings.setdefault('clustering', 'dbscan')
    settings.setdefault('exclude', None)
    settings.setdefault('col_to_compare', 'columnID')
    settings.setdefault('pos', 'c1')
    settings.setdefault('neg', 'c2')
    settings.setdefault('mix', 'c3')
    settings.setdefault('embedding_by_controls', False)
    settings.setdefault('plot_images', True)
    settings.setdefault('reduction_method','umap')
    settings.setdefault('save_figure', False)
    settings.setdefault('n_jobs', -1)
    settings.setdefault('color_by', None)
    settings.setdefault('exclude_conditions', None)
    settings.setdefault('exclude_rows', None)
    settings.setdefault('batch_correction', 'none')
    settings.setdefault('batch_column', 'plateID')
    settings.setdefault('batch_control_column', None)
    settings.setdefault('batch_control_values', None)
    settings.setdefault('batch_min_samples', 3)
    settings.setdefault('batch_missing_control', 'error')
    settings.setdefault('analyze_clusters', False)
    settings.setdefault('resnet_features', False)
    settings.setdefault('verbose',True)
    # 'auto' uses the PNG crop folder when one exists and falls back to
    # cutting crops out of merged/*.npy on demand; 'png' and 'merged'
    # force one source. See spacr.crops.resolve_crop_source.
    settings.setdefault('crop_source', 'auto')
    return settings

def get_measure_crop_settings(settings=None):
    """Return the default settings for the measure-and-crop pipeline.

    Enables test mode / plotting automatically when ``test_mode`` is True.

    :param settings: optional dict to fill in place; a new dict is created if None.
    :returns: the settings dict with defaults applied.
    """
    if settings is None:
        settings = {}
    # Coerce bracketed strings (e.g. channels "[0,1,2,3]" imported from a CSV) back into
    # Python lists/tuples. The Qt drag-and-drop settings import reads CSV cells as raw
    # strings and does not run them through check_settings(), so without this measure_crop
    # rejects channels / crop_mode / png_size / ... as "not a list". Idempotent: values
    # that are already lists, or ordinary strings, are left untouched.
    import ast as _ast
    for _k, _v in list(settings.items()):
        if isinstance(_v, str) and _v.strip()[:1] in "[(":
            try:
                settings[_k] = _ast.literal_eval(_v)
            except (ValueError, SyntaxError):
                pass
    settings.setdefault('src', 'path')

    settings.setdefault('verbose', False)
    settings.setdefault('experiment', 'exp')
    
    # Test mode
    settings.setdefault('test_mode', False)
    # Validate-only: measure_crop runs the pre-flight checks in spacr.validate,
    # prints the report plus the plan, and returns before the worker pool
    # starts or anything is written to measurements.db.
    settings.setdefault('dry_run', False)
    settings.setdefault('test_nr', 10)
    settings.setdefault('channels', [0,1,2,3])

    #measurement settings
    settings.setdefault('save_measurements',True)
    settings.setdefault('radial_dist', True)
    settings.setdefault('calculate_correlation', True)
    settings.setdefault('manders_thresholds', [15,85,95])
    settings.setdefault('homogeneity', True)
    settings.setdefault('homogeneity_distances', [8,16,32])

    # Voxel geometry. Measure needs these for the same reason segmentation
    # does: on a 3-D mask every regionprops and distance-transform call takes
    # a spacing, and without one the z axis is treated as if a plane step were
    # one xy pixel. Left at None a 2-D run is unaffected (spacing is not
    # applied in 2-D at all, or *_area would silently change units) and a 3-D
    # run stops rather than guessing.
    settings.setdefault('voxel_size_z_um', None)
    settings.setdefault('voxel_size_xy_um', None)
    settings.setdefault('anisotropy', None)

    # Cropping settings
    settings.setdefault('save_arrays', False)
    settings.setdefault('save_png',True)
    settings.setdefault('use_bounding_box',False)
    settings.setdefault('png_size',[224,224])
    settings.setdefault('png_dims',[0,1,2])
    settings.setdefault('normalize',False)
    settings.setdefault('normalize_by','png')
    settings.setdefault('crop_mode',['cell'])
    settings.setdefault('dialate_pngs', False)
    settings.setdefault('dialate_png_ratios', [0.2])

    # Timelapsed settings
    settings.setdefault('timelapse', False)
    settings.setdefault('timelapse_objects', ['cell'])

    # Operational settings
    settings.setdefault('plot',False)
    settings.setdefault('n_jobs', _default_worker_count(reserve=2))

    # Object settings
    settings.setdefault('cell_mask_dim',4)
    settings.setdefault('nucleus_mask_dim',5)
    settings.setdefault('pathogen_mask_dim',6)
    settings.setdefault('organelle_mask_dim',None)
    settings.setdefault('cytoplasm',False)
    settings.setdefault('uninfected',True)
    settings.setdefault('cell_min_size',0)
    settings.setdefault('nucleus_min_size',0)
    settings.setdefault('pathogen_min_size',0)
    settings.setdefault('organelle_min_size',0)
    settings.setdefault('cytoplasm_min_size',0)
    settings.setdefault('merge_edge_pathogen_cells', True)
    
    settings.setdefault('distance_gaussian_sigma', 10)
    
    if settings['test_mode']:
        settings['verbose'] = True
        settings['plot'] = True
        test_imgs = settings['test_nr']
        print(f'Test mode enabled with {test_imgs} images, plotting set to True')

    # Fail-loud policy. None means "not set here" and defers to the
    # SPACR_STRICT_ERRORS environment variable, which is how a cluster turns
    # it on for a whole batch without editing every settings file. True/False
    # here is an explicit per-run choice and wins over the environment.
    settings.setdefault('strict_errors', None)
    settings.setdefault('max_failure_rate', None)
    # Continue an interrupted run instead of starting over. Opt-in:
    # spacr.resume validates what is already on disk rather than trusting it,
    # and clears a field's existing rows before re-measuring it.
    settings.setdefault('resume', False)
    return settings

def set_default_analyze_screen(settings):
    """Populate default settings for screen analysis (ML-based scoring).

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('annotation_column', None)
    settings.setdefault('save_to_db', False)
    settings.setdefault('model_type_ml','xgboost')
    settings.setdefault('heatmap_feature','predictions')
    settings.setdefault('grouping','mean')
    settings.setdefault('min_max','allq')
    settings.setdefault('cmap','viridis')
    settings.setdefault('channel_of_interest',3)
    settings.setdefault('minimum_cell_count',25)
    settings.setdefault('reg_alpha',0.1)
    settings.setdefault('reg_lambda',1.0)
    settings.setdefault('learning_rate',0.001)
    settings.setdefault('n_estimators',1000)
    settings.setdefault('test_size',0.2)
    settings.setdefault('location_column','columnID')
    settings.setdefault('positive_control','c2')
    settings.setdefault('negative_control','c1')
    settings.setdefault('exclude',None)
    settings.setdefault('nuclei_limit',True)
    settings.setdefault('pathogen_limit',3)
    settings.setdefault('n_repeats',10)
    settings.setdefault('top_features',30)
    settings.setdefault('remove_low_variance_features',True)
    settings.setdefault('remove_highly_correlated_features',True)
    settings.setdefault('batch_correction', 'none')
    settings.setdefault('batch_column', 'plateID')
    settings.setdefault('batch_control_column', None)
    # Keep this blank so control_center follows the module's current
    # negative_control value instead of silently retaining a stale 'c1' when
    # the user changes the plate layout.
    settings.setdefault('batch_control_values', None)
    settings.setdefault('batch_min_samples', 3)
    settings.setdefault('batch_missing_control', 'error')
    settings.setdefault('n_jobs',-1)
    settings.setdefault('prune_features',False)
    settings.setdefault('cross_validation',True)
    settings.setdefault('verbose',True)
    return settings

def _set_classifier_evaluation_defaults(settings):
    """Populate shared Classify evaluation and nested-CV defaults."""
    settings.setdefault('classifier_evaluation', True)
    settings.setdefault('nested_cv_inner_folds', 0)
    settings.setdefault('evaluation_calibration', 'temperature')
    settings.setdefault('evaluation_bins', 10)
    settings.setdefault('evaluation_fail_on_leakage', True)
    settings.setdefault('leakage_audit_train_test', True)
    settings.setdefault('leakage_hash_content', True)
    settings.setdefault('leakage_require_identity', True)
    return settings


def set_default_train_test_model(settings):
    """Populate default settings for the train/test classifier training pipeline.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    cores = _default_worker_count(reserve=2)

    settings.setdefault('src','path')
    settings.setdefault('train',True)
    settings.setdefault('test',False)
    settings.setdefault('classes',['nc','pc'])
    settings.setdefault('model_type','maxvit_t')
    settings.setdefault('optimizer_type','adamw')
    settings.setdefault('schedule','cosine') #reduce_lr_on_plateau, step_lr
    settings.setdefault('loss_type','focal_loss') # binary_cross_entropy_with_logits
    settings.setdefault('normalize',True)
    settings.setdefault('image_size',224)
    settings.setdefault('batch_size',64)
    settings.setdefault('epochs',100)
    settings.setdefault('plot',True)
    settings.setdefault('tensorboard',True)
    settings.setdefault('val_split',0.1)
    settings.setdefault('learning_rate',0.001)
    settings.setdefault('weight_decay',0.00001)
    settings.setdefault('dropout_rate',0.1)
    settings.setdefault('init_weights',True)
    settings.setdefault('amsgrad',True)
    settings.setdefault('use_checkpoint',True)
    settings.setdefault('gradient_accumulation',True)
    settings.setdefault('gradient_accumulation_steps',4)
    settings.setdefault('intermedeate_save',True)
    settings.setdefault('resume_checkpoint','')
    settings.setdefault('custom_model_path','')
    settings.setdefault('pin_memory',False)
    settings.setdefault('n_jobs',cores)
    settings.setdefault('train_channels',['r','g','b'])
    settings.setdefault('augment',False)
    settings.setdefault('verbose',False)
    settings.setdefault('class_balance','none')
    settings.setdefault('cross_validation_folds',0)
    settings.setdefault('cross_validation_enabled',False)
    settings.setdefault('cv_group_by','well')
    _set_classifier_evaluation_defaults(settings)
    # Fail-loud policy. None means "not set here" and defers to the
    # SPACR_STRICT_ERRORS environment variable, which is how a cluster turns
    # it on for a whole batch without editing every settings file. True/False
    # here is an explicit per-run choice and wins over the environment.
    settings.setdefault('strict_errors', None)
    settings.setdefault('max_failure_rate', None)
    # 'auto' uses the PNG crop folder when one exists and falls back to
    # cutting crops out of merged/*.npy on demand; 'png' and 'merged'
    # force one source. See spacr.crops.resolve_crop_source.
    settings.setdefault('crop_source', 'auto')
    return settings

def set_generate_training_dataset_defaults(settings):
    """Populate default settings for generating a labeled training dataset.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('tables', ['cell', 'nucleus', 'pathogen', 'cytoplasm'])
    settings.setdefault('dataset_mode','metadata')
    settings.setdefault('annotation_column','test')
    settings.setdefault('annotated_classes',[1,2])
    # class_metadata holds VALUES OF metadata_type_by ('columnID'), so the
    # entries have to be well ids. It was set twice, and the first call won:
    # ['nc','pc'] -- the CLASS NAMES from deep_spacr_defaults' 'classes' key,
    # pasted onto the wrong setting. No columnID is ever 'nc', so the shipped
    # default selected zero crops in both classes and the second, correct
    # assignment below was dead. Same for 'tables', set to the four object
    # tables and then to None.
    settings.setdefault('metadata_item_1_name',None) # e.g. ['nc','pc']
    settings.setdefault('metadata_item_1_value',None) # e.g. [['c19','c2'],['c3','c4']]
    settings.setdefault('metadata_item_2_name',None) # e.g. ['sample1','sample2']
    settings.setdefault('metadata_item_2_value',None) #e.g. [['r1','r2'],['r3','r4']]
    settings.setdefault('size',224)
    settings.setdefault('test_split',0.1)
    settings.setdefault('cv_group_by','well')
    settings.setdefault('class_metadata',[['c1'],['c2']])
    settings.setdefault('metadata_type_by','columnID')
    settings.setdefault('channel_of_interest',3)
    settings.setdefault('custom_measurement',None)
    settings.setdefault('nuclei_limit',True)
    settings.setdefault('pathogen_limit',True)
    settings.setdefault('png_type','cell_png')
    settings.setdefault('random_seed',42)
    
    return settings

def deep_spacr_defaults(settings):
    """Populate default settings for the end-to-end deep_spacr training pipeline.

    Covers dataset generation, model training/testing and applying the trained
    model to the dataset in a single settings dict.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    cores = _default_worker_count(reserve=4)
    
    settings.setdefault('src','path')
    settings.setdefault('dataset_mode','metadata')
    settings.setdefault('annotation_column','test')
    settings.setdefault('annotated_classes',[1,2])
    settings.setdefault('classes',['nc','pc'])
    settings.setdefault('size',224)
    settings.setdefault('test_split',0.1)
    settings.setdefault('class_metadata',[['c1'],['c2']])
    settings.setdefault('metadata_type_by','columnID')
    settings.setdefault('channel_of_interest',3)
    settings.setdefault('custom_measurement',None)
    settings.setdefault('tables',None)
    settings.setdefault('png_type','cell_png')
    settings.setdefault('custom_model',False)
    settings.setdefault('custom_model_path','')
    settings.setdefault('train',True)
    settings.setdefault('test',False)
    settings.setdefault('model_type','maxvit_t')
    settings.setdefault('optimizer_type','adamw')
    settings.setdefault('schedule','cosine')
    settings.setdefault('loss_type','auto') 
    settings.setdefault('normalize',True)
    settings.setdefault('image_size',224)
    settings.setdefault('batch_size',64)
    settings.setdefault('epochs',100)
    settings.setdefault('plot',True)
    settings.setdefault('tensorboard',True)
    settings.setdefault('val_split',0.1)
    settings.setdefault('learning_rate',0.001)
    settings.setdefault('weight_decay',0.00001)
    settings.setdefault('dropout_rate',0.1)
    settings.setdefault('init_weights',True)
    settings.setdefault('amsgrad',True)
    settings.setdefault('use_checkpoint',True)
    settings.setdefault('gradient_accumulation',True)
    settings.setdefault('gradient_accumulation_steps',4)
    settings.setdefault('label_smoothing',0.1)
    settings.setdefault('focal_gamma',2.0)
    settings.setdefault('focal_alpha',None)
    settings.setdefault('logit_adjust_tau',1.0)
    settings.setdefault('early_stopping_patience',0)
    settings.setdefault('intermedeate_save',True)
    settings.setdefault('resume_checkpoint','')
    settings.setdefault('pin_memory',False)
    settings.setdefault('n_jobs',cores)
    settings.setdefault('train_channels',['r','g','b'])
    settings.setdefault('augment',False)
    settings.setdefault('verbose',True)
    settings.setdefault('apply_model_to_dataset',True)
    settings.setdefault('file_metadata',None)
    settings.setdefault('sample',None)
    settings.setdefault('experiment','exp.')
    settings.setdefault('score_threshold',0.5)
    settings.setdefault('dataset','')
    settings.setdefault('model_path','')
    settings.setdefault('file_type','cell_png')
    settings.setdefault('generate_training_dataset', True)
    settings.setdefault('balance_to_smallest', True)
    settings.setdefault('write_random_annotation_column', False)
    settings.setdefault('generate_full_dataset', False)
    settings.setdefault('tar_path','')
    settings.setdefault('n_top_examples',20)
    settings.setdefault('random_seed',42)
    settings.setdefault('crop_source','auto')
    settings.setdefault('strict_errors',None)
    settings.setdefault('max_failure_rate',None)
    settings.setdefault('class_balance','none')
    settings.setdefault('cross_validation_folds',0)
    settings.setdefault('cross_validation_enabled',False)
    settings.setdefault('cv_group_by','well')
    _set_classifier_evaluation_defaults(settings)
    return settings

def get_train_test_model_settings(settings):
     """Populate default settings for the train/test classifier settings dict.

     :param settings: dict to fill in place.
     :returns: the settings dict with defaults applied.
     """
     settings.setdefault('src', 'path')
     settings.setdefault('train', True)
     settings.setdefault('test', False)
     settings.setdefault('custom_model', False)
     settings.setdefault('classes', ['nc','pc'])
     settings.setdefault('train_channels', ['r','g','b'])
     settings.setdefault('model_type', 'maxvit_t')
     settings.setdefault('optimizer_type', 'adamw')
     settings.setdefault('schedule', 'cosine')
     settings.setdefault('loss_type', 'focal_loss')
     settings.setdefault('normalize', True)
     settings.setdefault('image_size', 224)
     settings.setdefault('batch_size', 64)
     settings.setdefault('epochs', 100)
     settings.setdefault('plot', True)
     settings.setdefault('tensorboard', True)
     settings.setdefault('val_split', 0.1)
     settings.setdefault('learning_rate', 0.0001)
     settings.setdefault('weight_decay', 0.00001)
     settings.setdefault('dropout_rate', 0.1)
     settings.setdefault('init_weights', True)
     settings.setdefault('amsgrad', True)
     settings.setdefault('use_checkpoint', True)
     settings.setdefault('gradient_accumulation', True)
     settings.setdefault('gradient_accumulation_steps', 4)
     settings.setdefault('intermedeate_save',True)
     settings.setdefault('resume_checkpoint','')
     settings.setdefault('custom_model_path','')
     settings.setdefault('pin_memory', True)
     settings.setdefault('n_jobs', 30)
     settings.setdefault('augment', True)
     settings.setdefault('verbose', True)
     settings.setdefault('label_smoothing', 0.1)
     settings.setdefault('focal_gamma', 2.0)
     settings.setdefault('focal_alpha', None)
     settings.setdefault('logit_adjust_tau', 1.0)
     settings.setdefault('early_stopping_patience', 0)
     settings.setdefault('class_balance', 'none')
     settings.setdefault('cross_validation_folds', 0)
     settings.setdefault('cross_validation_enabled', False)
     settings.setdefault('cv_group_by', 'well')
     _set_classifier_evaluation_defaults(settings)
     return settings


def get_analyze_recruitment_default_settings(settings):
    """Populate default settings for the recruitment-analysis pipeline.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('target','protein')
    settings.setdefault('cell_types',['HeLa'])
    settings.setdefault('cell_plate_metadata',None)
    settings.setdefault('pathogen_types',['pathogen_1', 'pathogen_2'])
    settings.setdefault('pathogen_plate_metadata',[['c1', 'c2', 'c3'],['c4','c5', 'c6']])
    settings.setdefault('treatments',['cm', 'lovastatin'])
    settings.setdefault('treatment_plate_metadata',[['r1', 'r2','r3'], ['r4', 'r5','r6']])
    #settings.setdefault('metadata_types',['columnID', 'columnID', 'rowID'])
    settings.setdefault('channel_dims',[0,1,2,3])
    settings.setdefault('cell_chann_dim',3)
    settings.setdefault('cell_mask_dim',4)
    settings.setdefault('nucleus_chann_dim',0)
    settings.setdefault('nucleus_mask_dim',5)
    settings.setdefault('pathogen_chann_dim',2)
    settings.setdefault('pathogen_mask_dim',6)
    settings.setdefault('channel_of_interest',2)
    settings.setdefault('plot',True)
    settings.setdefault('plot_nr',3)
    settings.setdefault('plot_control',True)
    settings.setdefault('figuresize',10)
    settings.setdefault('pathogen_limit',10)
    settings.setdefault('nuclei_limit',1)
    settings.setdefault('cells_per_well',0)
    settings.setdefault('pathogen_size_range',[0,100000])
    settings.setdefault('nucleus_size_range',[0,100000])
    settings.setdefault('cell_size_range',[0,100000])
    settings.setdefault('pathogen_intensity_range',[0,100000])
    settings.setdefault('nucleus_intensity_range',[0,100000])
    settings.setdefault('cell_intensity_range',[0,100000])
    settings.setdefault('target_intensity_min',1)
    return settings

def get_default_test_cellpose_model_settings(settings):
    """Populate default settings for testing a Cellpose model on a dataset.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('model_path','path')
    settings.setdefault('save',True)
    settings.setdefault('normalize',True)
    settings.setdefault('percentiles',(2,98))
    settings.setdefault('batch_size',50)
    settings.setdefault('CP_probability',0)
    settings.setdefault('FT',100)
    settings.setdefault('target_size',1000)
    return settings

def get_default_apply_cellpose_model_settings(settings):
    """Populate default settings for applying a Cellpose model to a dataset.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('model_path','path')
    settings.setdefault('save',True)
    settings.setdefault('normalize',True)
    settings.setdefault('percentiles',(2,98))
    settings.setdefault('batch_size',50)
    settings.setdefault('CP_probability',0)
    settings.setdefault('FT',100)
    settings.setdefault('circularize',False)
    settings.setdefault('target_size',1000)
    return settings

def default_settings_analyze_percent_positive(settings):
    """Populate default settings for the "percent positive" per-well analysis.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('tables',['cell'])
    settings.setdefault('filter_1',['cell_area',1000])
    settings.setdefault('value_col','cell_channel_2_mean_intensity')
    settings.setdefault('threshold',2000)
    return settings

def get_analyze_reads_default_settings(settings):
    """Populate default settings for analyzing FASTQ read barcodes.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('upstream', 'CTTCTGGTAAATGGGGATGTCAAGTT') 
    settings.setdefault('downstream', 'GTTTAAGAGCTATGCTGGAAACAGCAG') #This is the reverce compliment of the column primer starting from the end #TGCTGTTTAAGAGCTATGCTGGAAACAGCA
    settings.setdefault('barecode_length_1', 8)
    settings.setdefault('barecode_length_2', 7)
    settings.setdefault('chunk_size', 1000000)
    settings.setdefault('test', False)
    return settings

def get_map_barcodes_default_settings(settings):
    """Populate default settings for mapping barcodes to gRNAs and plates.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    # These legacy keys are retained for older callers, but their defaults
    # must be portable. The active Qt workflow uses the corresponding
    # row_csv/column_csv/grna_csv keys populated from these same resources.
    settings.setdefault('grna', bundled_barcode_path('grna'))
    settings.setdefault('barcodes', bundled_barcode_path('column'))
    settings.setdefault('plate_dict', "{'EO1': 'plate1', 'EO2': 'plate2', 'EO3': 'plate3', 'EO4': 'plate4', 'EO5': 'plate5', 'EO6': 'plate6', 'EO7': 'plate7', 'EO8': 'plate8'}")
    settings.setdefault('test', False)
    settings.setdefault('verbose', True)
    settings.setdefault('pc', 'TGGT1_220950_1')
    settings.setdefault('pc_loc', 'c2')
    settings.setdefault('nc', 'TGGT1_233460_4')
    settings.setdefault('nc_loc', 'c1')
    return settings

def get_train_cellpose_default_settings(settings):
    """Populate default settings for training a Cellpose model.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('model_name','new_model')
    settings.setdefault('model_type','cpsam')
    settings.setdefault('Signal_to_noise',10)
    settings.setdefault('background',200)
    settings.setdefault('remove_background',False)
    settings.setdefault('learning_rate',0.2)
    settings.setdefault('weight_decay',1e-05)
    settings.setdefault('batch_size',8)
    settings.setdefault('n_epochs',10000)
    settings.setdefault('from_scratch',False)
    settings.setdefault('diameter',30)
    settings.setdefault('resize',False)
    settings.setdefault('width_height',[1000,1000])
    # train_cellpose and CellposeLazyDataset consume these keys directly.
    # Keeping target_size aligned with the historical width/height default
    # makes the defaults helper a complete runnable contract.
    settings.setdefault('target_size', 1000)
    settings.setdefault('augment', False)
    settings.setdefault('verbose',True)
    return settings

def set_generate_dataset_defaults(settings):
    """Populate default settings for the generic dataset-generation step.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('file_metadata',None)
    settings.setdefault('experiment','experiment_1')
    settings.setdefault('sample',None)
    # 'auto' uses the PNG crop folder when one exists and falls back to
    # cutting crops out of merged/*.npy on demand; 'png' and 'merged'
    # force one source. See spacr.crops.resolve_crop_source.
    settings.setdefault('crop_source', 'auto')
    return settings

def get_perform_regression_default_settings(settings):
    """Populate default settings for gRNA/score regression analysis.

    Switches ``agg_type`` to None automatically when quantile regression is
    selected, so ``alpha`` is treated as the quantile.

    Every key :func:`spacr.ml.perform_regression` reads with ``settings[...]``
    is filled here. Six were not, and because all three dispatchers (the Tk
    panel via ``gui_core.setup_settings_panel``, the Qt panel via
    ``qt.screens.settings_model.resolve_default_settings`` and ``spacr-run
    regression`` via ``cli.module_defaults``) build the dict from this one
    function, regression could not be started from any entry point: it died on
    ``KeyError: 'verbose'`` at ml.py:1409, after both input CSVs had been read
    and ``settings/regression.csv`` had been written, so the failure looked
    like a run that had started cleanly.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('count_data','list of paths')
    settings.setdefault('score_data','list of paths')
    settings.setdefault('positive_control','239740')
    settings.setdefault('negative_control','233460')
    settings.setdefault('min_n',0)
    settings.setdefault('controls',['000000_1','000000_10','000000_11','000000_12','000000_13','000000_14','000000_15','000000_16','000000_17','000000_18','000000_19','000000_20','000000_21','000000_22','000000_23','000000_24','000000_25','000000_26','000000_27','000000_28','000000_29','000000_3','000000_30','000000_31','000000_32','000000_4','000000_5','000000_6','000000_8','000000_9'])
    settings.setdefault('fraction_threshold',None)
    settings.setdefault('dependent_variable','pred')
    settings.setdefault('threshold_method','std')
    settings.setdefault('threshold_multiplier',3)
    settings.setdefault('target_unique_count',5)
    settings.setdefault('transform',None)
    settings.setdefault('log_x',False)
    settings.setdefault('log_y',False)
    settings.setdefault('x_lim',None)
    settings.setdefault('outlier_detection',True)
    settings.setdefault('agg_type','mean')
    settings.setdefault('min_cell_count',None)
    settings.setdefault('regression_type','ols')
    settings.setdefault('random_row_column_effects',False)
    settings.setdefault('split_axis_lims','')
    settings.setdefault('cov_type',None)
    settings.setdefault('alpha',1)
    # Every knob below is read by spacr.ml.regression_model for at least one
    # regression_type, and each one is INDEXED (settings[...]) by
    # perform_regression, not .get()-ed: a model that reads a setting must have
    # a default here or the module is unstartable from every entry point, which
    # is exactly how six other keys took regression down.
    #
    # The defaults match regression_model's own signature defaults, because
    # _reject_unused_settings compares against them to tell "the user asked for
    # this" from "the panel posted its default".
    settings.setdefault('l1_ratio', 0.5)
    settings.setdefault('quantile', 0.5)
    settings.setdefault('hinge_threshold', None)
    settings.setdefault('hinge_n_boot', 200)
    settings.setdefault('huber_t', 1.345)
    settings.setdefault('lasso_n_boot', 200)
    settings.setdefault('lasso_selection_threshold', 0.6)
    settings.setdefault('filter_value',['c1', 'c2', 'c3'])
    settings.setdefault('filter_column','columnID')
    # sequencing.graph_sequencing_stats iterates settings['control_wells'] and
    # drops those wells from the count table before it sweeps for the fraction
    # threshold, exactly as ml.clean_controls drops filter_value from the score
    # table. The two must name the same wells or the threshold is fitted on
    # wells the regression never sees, so this follows filter_value. It is
    # indexed, not .get(), and it is iterated, so None -- which is what the
    # invasion assay defaults the same key name to -- is not a legal value here.
    settings.setdefault(
        'control_wells',
        list(settings['filter_value'])
        if isinstance(settings['filter_value'], (list, tuple)) else [])
    settings.setdefault('batch_correction', 'none')
    settings.setdefault('batch_column', 'plateID')
    settings.setdefault('batch_control_column', 'columnID')
    settings.setdefault('batch_control_values', None)
    settings.setdefault('batch_min_samples', 3)
    settings.setdefault('batch_missing_control', 'error')
    settings.setdefault('plateID','plate1')
    # Acquisition-specific annotations cannot have a meaningful machine-wide
    # default. An empty list makes the optional input explicit and portable.
    settings.setdefault('metadata_files', [])
    settings.setdefault('volcano','gene')
    settings.setdefault('toxo', True)
    # perform_regression prints a per-stage row count and display()s the whole
    # per-object score table under verbose, which is millions of rows on a real
    # screen, so this pipeline is one of the False ones.
    settings.setdefault('verbose', False)
    # minimum_cell_simulation reads settings['tolerance'] and accepts an int
    # (percent) or a float (fraction); anything else raises ValueError. 0.02 is
    # the 2% the function's own worked example uses.
    settings.setdefault('tolerance', 0.02)
    # minimum_cell_simulation resamples settings['score_column'] out of the
    # score CSV to find how many cells a well needs before its mean is stable.
    # That has to be the column being regressed, or the simulated minimum
    # describes a different measurement than the one the model fits, so it
    # follows dependent_variable rather than hard-coding 'pred'.
    settings.setdefault('score_column', settings['dependent_variable'])
    # process_scores: False/0 = as measured, True/1 = 1 - x, -1 = 1 / x.
    settings.setdefault('invert_dependent_variable', False)
    # toxo.custom_volcano_plot's y limits: None auto-scales, [lo, hi] fixes the
    # axis, [[lo1, hi1], [lo2, hi2]] draws a broken axis.
    settings.setdefault('y_lims', None)

    if settings['regression_type'] == 'quantile':
        # alpha USED to double as the quantile here, which was a silent
        # overload of a key whose tooltip, GUI label and every other regression
        # type call a penalty weight: a settings CSV reading alpha=0.9 meant
        # "the 90th percentile" under one regression_type and "shrink hard"
        # under the next. The quantile now has its own key, and an alpha left
        # over from the old spelling is refused rather than ignored - silently
        # dropping it would fit the median and label the output 0.9.
        if settings['alpha'] != 1:
            raise ValueError(
                f"regression_type='quantile' does not use alpha "
                f"(alpha={settings['alpha']!r} was set). The quantile being "
                f"fitted is the 'quantile' setting; alpha is the penalty "
                f"weight of the penalised models and does nothing here. Set "
                f"quantile={settings['alpha']!r} instead if that is the "
                f"percentile you meant, and leave alpha at 1.")
        q = settings['quantile']
        if not isinstance(q, (int, float)) or isinstance(q, bool) \
                or not 0.0 < float(q) < 1.0:
            raise ValueError(
                f"quantile must be a number strictly inside (0, 1); got "
                f"{q!r}. 0.5 fits the median, 0.9 the upper tail.")
        print(f"Fitting the {float(q):g} quantile of {settings['dependent_variable']}")
        # Quantile regression on per-well MEANS would be the quantile of an
        # average, which is not the quantile of the response.
        settings['agg_type'] = None
        print(f'agg_type set to None for quantile regression')

    # Fail-loud policy. None means "not set here" and defers to the
    # SPACR_STRICT_ERRORS environment variable, which is how a cluster turns
    # it on for a whole batch without editing every settings file. True/False
    # here is an explicit per-run choice and wins over the environment.
    settings.setdefault('strict_errors', None)
    settings.setdefault('max_failure_rate', None)
    return settings

def get_check_cellpose_models_default_settings(settings):
    """Populate default settings for the "check Cellpose models" utility.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('batch_size', 10)
    settings.setdefault('CP_prob', 0)
    settings.setdefault('flow_threshold', 0.4)
    settings.setdefault('save', True)
    settings.setdefault('normalize', True)
    settings.setdefault('channels', [0,0])
    settings.setdefault('percentiles', None)
    settings.setdefault('invert', False)
    settings.setdefault('plot', True)
    settings.setdefault('diameter', 40)
    settings.setdefault('grayscale', True)
    settings.setdefault('remove_background', False)
    settings.setdefault('background', 100)
    settings.setdefault('Signal_to_noise', 5)
    settings.setdefault('verbose', False)
    settings.setdefault('resize', False)
    settings.setdefault('target_height', None)
    settings.setdefault('target_width', None)
    return settings

def get_identify_masks_finetune_default_settings(settings):
    """Populate default settings for fine-tuning mask identification.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('model_name', 'cpsam')
    settings.setdefault('custom_model', None)
    settings.setdefault('channels', [0,0])
    settings.setdefault('background', 100)
    settings.setdefault('remove_background', False)
    settings.setdefault('Signal_to_noise', 10)
    settings.setdefault('CP_prob', 0)
    settings.setdefault('diameter', 30)
    settings.setdefault('batch_size', 50)
    settings.setdefault('flow_threshold', 0.4)
    settings.setdefault('save', False)
    settings.setdefault('verbose', False)
    settings.setdefault('normalize', True)
    settings.setdefault('percentiles', None)
    settings.setdefault('invert', False)
    settings.setdefault('resize', False)
    settings.setdefault('target_height', None)
    settings.setdefault('target_width', None)
    settings.setdefault('rescale', False)
    settings.setdefault('resample', False)
    settings.setdefault('grayscale', True)
    settings.setdefault('fill_in', True)
    return settings

q = None
expected_types = {
    "src": (str, list),
    "metadata_type": str,
    "custom_regex": (str, type(None)),
    "cov_type": (str, type(None)),
    "experiment": str,
    "channels": list,
    "magnification": int,
    "nucleus_channel": (int, type(None)),
    "nucleus_background": int,
    "nucleus_Signal_to_noise": float,
    "nucleus_CP_prob": float,
    "nucleus_FT": (int, float),
    "cell_channel": (int, type(None)),
    "cell_background": (int, float),
    "cell_Signal_to_noise": (int, float),
    "cell_CP_prob": (int, float),
    "cell_FT": (int, float),
    "pathogen_channel": (int, type(None)),
    "pathogen_background": (int, float),
    "pathogen_Signal_to_noise": (int, float),
    "pathogen_CP_prob": (int, float),
    "pathogen_FT": (int, float),
    "preprocess": bool,
    "masks": bool,
    "examples_to_plot": int,
    "randomize": bool,
    "timelapse": bool,
    "timelapse_displacement": int,
    "timelapse_memory": int,
    "timelapse_frame_limits": (list, type(None)),  # This can be a list of lists
    #"timelapse_frame_limits": (list, type(None)),  # This can be a list of lists
    "timelapse_remove_transient": bool,
    "timelapse_mode": str,
    "trackastra_model": str,
    "trackastra_linking": str,
    "ultrack_max_distance": float,
    "ultrack_division_weight": float,
    "ultrack_contour_sigma": float,
    "ultrack_n_workers": int,
    "timelapse_objects": list,
    "fps": int,
    "lower_percentile": (int, float),
    "merge_pathogens": bool,
    "normalize_plots": bool,
    "all_to_mip": bool,
    # 3D (Beta)
    "z_stack": bool,
    "z_segmentation_mode": str,
    "z_axis": (int, type(None)),
    "z_projection": (str, type(None)),
    "anisotropy": (int, float, type(None)),
    "voxel_size_z_um": (int, float, type(None)),
    "voxel_size_xy_um": (int, float, type(None)),
    "stitch_threshold": (int, float),
    # 4D (Beta)
    "t_stack": bool,
    "t_axis_order": (str, type(None)),
    "t_axis": (int, type(None)),
    "frame_interval_s": (int, float, type(None)),
    "t_track_backend": str,
    "t_link_threshold": (int, float),
    "t_max_displacement_px": (int, float, type(None)),
    "t_max_displacement_um": (int, float, type(None)),
    "t_project_for_tracking": bool,
    "save_original_images": bool,
    "keep_intermediate": bool,
    "keep_original_images": bool,
    "pick_slice": bool,
    "skip_mode": str,
    "save": bool,
    "plot": bool,
    "tensorboard": bool,
    "verbose": bool,
    "cell_mask_dim": int,
    "cell_min_size": int,
    "cytoplasm_min_size": int,
    "nucleus_mask_dim": int,
    "nucleus_min_size": int,
    "pathogen_mask_dim": int,
    "pathogen_min_size": int,
    "save_png": bool,
    "crop_mode": list,
    "use_bounding_box": bool,
    "png_size": list,  # This can be a list of lists 
    "png_dims": list,
    "normalize_by": str,
    "save_measurements": bool,
    "uninfected": bool,
    "dialate_pngs": bool,
    "dialate_png_ratios": list,
    "cells": list,
    "cell_loc": list,
    "pathogens": list,
    "pathogen_loc": (list, list),  # This can be a list of lists 
    "treatments": list,
    "treatment_loc": (list, list),  # This can be a list of lists
    "channel_of_interest": int,
    "compartments": list,
    "measurement": str,
    "nr_imgs": int,
    "um_per_pixel": (int, float),
    "pathogen_limit": int,
    "nuclei_limit": int,
    "filter_min_max": (list, type(None)),
    "channel_dims": list,
    "backgrounds": list,
    "background": str,
    "outline_thickness": int,
    "outline_color": str,
    "overlay_chans": list,
    "normalization_percentiles": list,
    "filter": bool,
    "fill_in":bool,
    "upscale": bool,
    "upscale_factor": float,
    "adjust_cells": bool,
    "row_limit": int,
    "tables": list,
    "visualize": str,
    "image_nr": int,
    "dot_size": int,
    "point_color": str,
    "point_alpha": float,
    "outline_width": float,
    "umap_canvas_width": int,
    "umap_sidebar_width": int,
    "n_neighbors": int,
    "min_dist": float,
    "metric": str,
    "eps": float,
    "min_samples": int,
    "batch_correction": str,
    "batch_column": str,
    "batch_control_column": (str, type(None)),
    "batch_control_values": (
        str, int, float, list, tuple, type(None),
    ),
    "batch_min_samples": int,
    "batch_missing_control": str,
    "filter_by": (str, type(None)),
    "img_zoom": float,
    "plot_by_cluster": bool,
    "plot_cluster_grids": bool,
    "remove_cluster_noise": bool,
    "remove_highly_correlated": bool,
    "log_data": bool,
    "black_background": bool,
    "remove_image_canvas": bool,
    "plot_outlines": bool,
    "plot_points": bool,
    "smooth_lines": bool,
    "clustering": str,
    "exclude": (str, type(None)),
    "col_to_compare": str,
    "pos": str,
    "neg": str,
    "embedding_by_controls": bool,
    "plot_images": bool,
    "reduction_method": str,
    "save_figure": bool,
    "color_by": (str, type(None)),
    "analyze_clusters": bool,
    "resnet_features": bool,
    "test_nr": int,
    "radial_dist": bool,
    "calculate_correlation": bool,
    "manders_thresholds": list,
    "homogeneity": bool,
    "homogeneity_distances": list,
    "save_arrays": bool,
    "cytoplasm": bool,
    "merge_edge_pathogen_cells": bool,
    "cells_per_well": int,
    "pathogen_size_range": list,
    "nucleus_size_range": list,
    "cell_size_range": list,
    "pathogen_intensity_range": list,
    "nucleus_intensity_range": list,
    "cell_intensity_range": list,
    "target_intensity_min": int,
    "model_type": str,
    "heatmap_feature": str,
    "grouping": str,
    "min_max": str,
    "minimum_cell_count": int,
    "n_estimators": int,
    "test_size": float,
    "location_column": str,
    "positive_control": str,
    "negative_control": str,
    "n_repeats": int,
    "top_features": int,
    "remove_low_variance_features": bool,
    "classes": list,
    "schedule": str,
    "loss_type": str,
    "image_size": int,
    "epochs": int,
    "val_split": float,
    "dropout_rate": float,
    "init_weights": bool,
    "amsgrad": bool,
    "use_checkpoint": bool,
    "gradient_accumulation": bool,
    "gradient_accumulation_steps": int,
    "intermedeate_save": (bool, list, tuple, type(None)),
    "pin_memory": bool,
    "n_jobs": int,
    "augment": bool,
    "cell_types": list,
    "cell_plate_metadata": (list, list),
    "pathogen_types": list,
    "pathogen_plate_metadata": (list, list),  # This can be a list of lists 
    "treatment_plate_metadata": (list, list),  # This can be a list of lists
    "metadata_types": list,
    "cell_chann_dim": int,
    "nucleus_chann_dim": int,
    "pathogen_chann_dim": int,
    "plot_nr": int,
    "plot_control": bool,
    "remove_background": bool,
    "target": str,
    "upstream": str,
    "downstream": str,
    "barecode_length_1": int,
    "barecode_length_2": int,
    "grna": str,
    "barcodes": str,
    "plate_dict": dict,
    "pc": str,
    "pc_loc": str,
    "nc": str,
    "nc_loc": str,
    "dependent_variable": str,
    # The four regression keys perform_regression indexes directly. They had no
    # entry here at all, so the GUIs could not render them, check_settings could
    # not coerce them out of a settings CSV and validate could not type-check
    # them -- see get_perform_regression_default_settings.
    "score_column": str,
    "tolerance": (int, float),
    "invert_dependent_variable": (bool, int),
    "y_lims": (list, type(None)),
    # The model-choice keys. The first three -- regression_type, alpha and
    # random_row_column_effects -- were categorised, tooltipped and defaulted
    # but had NO entry here, and check_settings DROPS any key it cannot type
    # ("Warning: Key 'regression_type' not found in expected types"), so the
    # Tk panel discarded whichever model the user picked and
    # get_perform_regression_default_settings then restored 'ols'. A run
    # configured as 'mixed' fitted OLS, wrote it to results/<...>/ols/ and said
    # nothing anywhere. The rest are the per-model knobs spacr.ml's backends
    # read; see spacr.ml.REGRESSION_SETTINGS_USED for which type reads which.
    #
    # regression_type is (str, NoneType) because None is the documented
    # "choose from the response distribution" value; alpha is (int, float, str,
    # NoneType) because 'auto'/None select it by cross-validation.
    "regression_type": (str, type(None)),
    "alpha": (int, float, str, type(None)),
    "random_row_column_effects": bool,
    "l1_ratio": float,
    "quantile": float,
    "hinge_threshold": (float, type(None)),
    "hinge_n_boot": int,
    "huber_t": float,
    "lasso_n_boot": int,
    "lasso_selection_threshold": float,
    "transform": (str, type(None)),
    "agg_type": str,
    "min_cell_count": int,
    "denoise":bool,
    "target_height": (int, type(None)),
    "target_width": (int, type(None)),
    "rescale": bool,
    "resample": bool,
    "model_name": str,
    "Signal_to_noise": int,
    "learning_rate": float,
    "weight_decay": float,
    "batch_size": int,
    "n_epochs": int,
    "from_scratch": bool,
    "width_height": list,
    "resize": bool,
    "compression": str,
    "complevel": int,
    "gene_weights_csv": str,
    "fraction_threshold": float,
    "barcode_mapping":dict,
    "redunction_method":str,
    "mix":str,
    "model_type_ml":str,
    "exclude_conditions":list,
    "exclude_rows": (dict, type(None)),
    "remove_highly_correlated_features":bool,
    'barcode_coordinates':list,  # This is a list of lists 
    'reverse_complement':bool,
    'file_type':str,
    'model_path':str,
    'dataset':str,
    'score_threshold':float,
    # (int, list or None), as the description says. It was the VALUE None,
    # which is not a type at all: validate.validate_settings does
    # `isinstance(value, (None,))` on it and died with
    # "TypeError: isinstance() arg 2 must be a type" -- the preflight check
    # crashed on any run that set 'sample'.
    'sample':(int, list, type(None)),
    'file_metadata':(str, type(None), list),
    "train":bool,
    "test":bool,
    'train_channels':list,
    "optimizer_type":str,
    "dataset_mode":str,
    "annotated_classes":list,
    "annotation_column":str,
    "apply_model_to_dataset":bool,
    "metadata_type_by":str,
    "custom_measurement":str,
    "custom_model":bool,
    "png_type":str,
    "custom_model_path":str,
    "resume_checkpoint":str,
    "generate_training_dataset":bool,
    "normalize":bool,
    "overlay":bool,
    "correlate":bool,
    "target_layer":str,
    "save_to_db":bool,
    "test_mode":bool,
    "dry_run":bool,
    'smoothgrad_samples':int,
    'smoothgrad_sigma':float,
    'occlusion_window':int,
    'occlusion_stride':int,
    'ig_steps':int,
    'ig_baseline':str,
    'attribution_steps':int,
    'attribution_baseline':str,
    'sanity_check':bool,
    'object_type':str,
    "parasite_table": str,
    "compartment": str,
    "vacuole_key": str,
    "vacuole_link_distance": (int, float, type(None)),
    "vacuole_link_factor": (int, float),
    "parasite_count_column": (str, type(None)),
    "max_parasites_per_vacuole": int,
    "require_host_cell": bool,
    "non_power_of_two_warn": float,
    "outside_channel": int,
    "total_channel": (int, type(None)),
    "intensity_statistic": str,
    "background_correction": str,
    "outside_threshold_method": str,
    "outside_threshold": (float, type(None)),
    "control_wells": (list, type(None)),
    "control_quantile": float,
    "min_control_objects": int,
    "min_objects_for_threshold": int,
    "min_objects_for_bimodality": int,
    "bimodality_cutoff": float,
    "threshold_agreement_tolerance": float,
    "threshold_sensitivity": float,
    "inflation_warn": float,
    "min_parasites_per_well": int,
    "min_parasite_area": (int, float),
    "max_parasite_area": (float, type(None)),
    "min_total_intensity": (float, type(None)),
    "extracellular_class": str,
    "seed_wells_from_cells": bool,
    "group_column": str,
    "level": str,
    "change_plate": bool,
    "qc_plot_max_panels": int,
    "resume":bool,
    "resume_search":bool,
    "checkpoint_path":(str, type(None)),
    "test_images":int,
    "remove_background_cell":bool,
    "remove_background_nucleus":bool,
    "remove_background_pathogen":bool,
    "figuresize":int,
    "cmap":str,
    "pathogen_model":str,
    "cell_model_name":str,
    "nucleus_model_name":str,
    "pathogen_model_name":str,
    "normalize_input":bool,
    "filter_column":str,
    "target_unique_count":int,
    "threshold_multiplier":int,
    "threshold_method":str,
    "count_data":list,
    "score_data":list,
    "min_n":int,
    "controls":list,
    "toxo":bool,
    "volcano":str,
    "metadata_files":list,
    "filter_value":list,
    "split_axis_lims":str,
    "x_lim":(list, type(None)),   # was (list, None) -- None is not a type
    "log_x":bool,
    "log_y":bool,
    "reg_alpha":(int,float),
    "reg_lambda":(int,float),
    "prune_features":bool,
    "cross_validation":bool,
    "offset_start":int,
    "chunk_size":int,
    "single_direction":str,
    "delete_intermediate":bool,
    "outlier_detection":bool,
    "CP_prob":int,
    "diameter":int,
    "flow_threshold":float,
    "cell_diameter":int,
    "nucleus_diameter":int,
    "pathogen_diameter":int,
    "diameter_estimate_n_fields":int,
    "seg_qc":str,
    "seg_qc_min_objects":int,
    "seg_qc_count_ratio":float,
    "seg_qc_size_ratio":float,
    "seg_qc_border_fraction":float,
    "seg_qc_outlier_mad":float,
    "seg_qc_outlier_fraction":float,
    "seg_qc_foreground_fraction":float,
    "seg_qc_split_ratio":float,
    "seg_qc_min_diameter":float,
    "seg_qc_tiny_fraction":float,
    "seg_qc_max_object_fraction":float,
    "seg_qc_plate_fail_fraction":float,
    "consolidate":bool,
    'use_sam_cell':bool,
    'use_sam_nucleus':bool,
    'use_sam_pathogen':bool,
    "distance_gaussian_sigma": (int, type(None)),
    "infection_xgb_n_estimators": int,
    "infection_xgb_max_depth": int,
    "infection_xgb_learning_rate": float,
    "infection_xgb_subsample": float,
    "infection_xgb_colsample_bytree": float,
    "infection_xgb_reg_lambda": float,
    "infection_xgb_random_state": int,
    "infection_xgb_n_jobs": int,
    "infection_xgb_proba_threshold": float,
    "infection_xgb_margin": float,
    "infection_xgb_top_features": int,
    "infection_xgb_proba_column": str,
    "infection_xgb_proba": float,
    "infection_xgb_drop_ambiguous": bool,
    "infection_xgb_ambiguous_low": float,
    "infection_xgb_ambiguous_high": float,
    "infection_xgb_min_cells_per_class": int,
    "infection_pca_method": str,
    "infection_pca_n_clusters": int,
    "infection_pca_random_state": int,
    "motility_ylim": tuple,
    "motility_xlim": tuple,
    "seconds_per_frame": int,
    "pixels_per_um": float,
    "infection_intensity_n_bins": int,
    "db_table_name": str,
    "infection_intensity_qc_graphs": bool,
    "infection_intensity_qc_panel_path": str,
    "infection_intensity_mode": str,
    "infection_intensity_strategy": str,
    "infection_intensity_qc": bool,
    "straightness_threshold": float,
    "straightness_filter": bool,
    "zscore_thresh": float,
    "max_displacement": float,
    "tracked_object": str,
    "motility_analysis": bool,
    "reuse_existing_measurements": bool,
    'infection_pca_umap_search': bool,
    'infection_pca_umap_n_neighbors_grid':list,
    'infection_pca_umap_min_dist_grid':list,
    'infection_pca_pathogen_weight':float,
    'infection_pca_log_intensity':bool,
    'infection_pca_tsne_search':bool,
    'infection_pca_tsne_perplexity_grid':list,
    'infection_pca_tsne_learning_rate_grid':list,
    'infection_intensity_qc_scope': str,
    'infection_pca_max_cells':int,
    'infection_pca_min_gt_separation':float,
    'infection_pca_min_silhouette':float,
    'infection_pca_umap_n_neighbors':int,
    'infection_pca_umap_min_dist':float,
    'infection_pca_tsne_perplexity':float,
    'organelle_channel': (int, type(None)),
    'organelle_morphology': str,
    'organelle_method': str,
    'organelle_diameter': int,
    'organelle_model_name':str,
    'organelle_min_size': (int, type(None)),
    'organelle_max_size': (int, type(None)),
    'organelle_remove_border':bool,
    'organelle_log_min_sigma': int,
    'organelle_log_max_sigma': int,
    'organelle_log_num_sigma': int,
    'organelle_log_threshold': float,
    'organelle_tophat_radius': int,
    'organelle_watershed_spots': bool,
    'organelle_ridge_sigmas': list,
    'organelle_ridge_filter': str,
    'organelle_skeletonize': bool,
    'organelle_network_threshold':str,
    'organelle_adaptive_block_size': int,
    'organelle_adaptive_offset': int,
    'organelle_morph_radius': int,
    'organelle_fill_holes': int,
    'organelle_CP_prob': float,
    'organelle_FT': float,
    'organelle_resample': bool,
    'organelle_mask_dim':(int, type(None)),
    'organelle_chann_dim':(int, type(None)),
    'organelle_rolling_ball':bool,
    'organelle_rolling_ball_radius':int,
    'organelle_clahe':bool,
    'organelle_clahe_clip_limit':float,
    'organelle_mask_within_cells':bool,
    'organelle_dog_sigma_low':float,
    'organelle_dog_sigma_high':float,
    'organelle_hysteresis_low':float,
    'organelle_hysteresis_high':float,
    'organelle_unet_model_path':str,
    'organelle_unet_threshold':float,
    'organelle_ring_sigma_inner':float, 
    'organelle_ring_sigma_outer':float,
    'organelle_ring_min_prominence':float, 
    'organelle_ring_fill_method':str,
    'summarize_organelles_by':str,
    'early_stopping_patience':int,
    'class_balance':str,
    'strict_errors':(bool, type(None)),
    'max_failure_rate':(float, type(None)),
    'crop_source':str,
    'queue_by_uncertainty':bool,
    'queue_measure':str,
    'queue_diversity':str,
    'queue_limit':int,
    'cross_validation_folds':int,
    'cross_validation_enabled':bool,
    'classifier_evaluation':bool,
    'nested_cv_inner_folds':int,
    'evaluation_calibration':str,
    'evaluation_bins':int,
    'evaluation_fail_on_leakage':bool,
    'leakage_audit_train_test':bool,
    'leakage_hash_content':bool,
    'leakage_require_identity':bool,
    'generate_full_dataset':bool,
    'tar_path':str,
    'n_top_examples':int,
    'random_seed':int,
    'balance_to_smallest':bool,
    'write_random_annotation_column':bool,
    'cv_group_by':str,
    'logit_adjust_tau':float,
    'focal_alpha':( float, type(None)),
    'focal_gamma':float,
    'label_smoothing':float,
    
    'cell_perimeter_fraction':float,
    'nucleus_perimeter_fraction':float,
    'pathogen_perimeter_fraction':float,
    'cell_intensity_merge':bool,
    'nucleus_intensity_merge':bool,
    'pathogen_intensity_merge':bool,
    'cell_intensity_split':bool,
    'nucleus_intensity_split':bool,
    'pathogen_intensity_split':bool,
    'cell_area_multiplier':float,
    'nucleus_area_multiplier':float,
    'pathogen_area_multiplier':float,
    'cell_min_distance':int,
    'nucleus_min_distance':int,
    'pathogen_min_distance':int,
    'cell_min_object_area':int,
    'nucleus_min_object_area':int,
    'pathogen_min_object_area':int,
    'cell_intensity_threshold_method':str,
    'nucleus_intensity_threshold_method':str,
    'pathogen_intensity_threshold_method':str,
    'cell_intensity_percentile':int,
    'nucleus_intensity_percentile':int,
    'pathogen_intensity_percentile':int,
    'postprocess_cell_masks':bool,
    'postprocess_nucleus_masks':bool,
    'postprocess_pathogen_masks':bool,
    'organelle_perimeter_fraction':float,
    'organelle_intensity_merge':bool,
    'organelle_intensity_split':bool,
    'organelle_area_multiplier':float,
    'organelle_min_distance':int,
    'organelle_min_object_area':int,
    'organelle_intensity_threshold_method':str,
    'organelle_intensity_percentile':int,
    'postprocess_organelle_masks':bool,
    'remove_border_cells':bool,
    'remove_border_nuclei':bool,
    'remove_border_pathogens':bool,
    'remove_border_organelles':bool,
    'cell_min_area':int,
    'nucleus_min_area':int,
    'pathogen_min_area':int,
    'organelle_min_area':int,
    'cell_max_area':(int, type(None)),
    'nucleus_max_area':(int, type(None)),
    'pathogen_max_area':(int, type(None)),
    'organelle_max_area':(int, type(None)),
    'cell_remove_border_objects':bool,
    'nucleus_remove_border_objects':bool,
    'pathogen_remove_border_objects':bool,
    'organelle_remove_border_objects':bool,
    'cell_min_intensity_percentile':int,
    'nucleus_min_intensity_percentile':int,
    'pathogen_min_intensity_percentile':int,
    'organelle_min_intensity_percentile':int,
    'cell_max_intensity_percentile':(int, type(None)),
    'nucleus_max_intensity_percentile':(int, type(None)),
    'pathogen_max_intensity_percentile':(int, type(None)),
    'organelle_max_intensity_percentile':(int, type(None)),
}

#: Settings that are declared -- typed here, tooltipped, offered by a GUI
#: category -- but that NOTHING in spaCR reads. Setting one is a silent no-op,
#: which is the worst failure mode there is: the run starts, finishes, and
#: produces a plausible wrong answer, and on a 40-plate cluster job that costs
#: a GPU-week to discover. ``spacr.validate`` turns each of these into a
#: pre-flight ERROR and ``spacr.cli.apply_overrides`` refuses a ``--set`` that
#: names one, both quoting the working spelling below.
#:
#: A key belongs here when its name appears NOWHERE in ``spacr/*.py`` outside
#: the ``expected_types`` / ``tooltips`` / ``descriptions`` / ``categories``
#: literals in this file -- no reader, and no ``setdefault`` either, so no
#: pipeline's own defaults can trip the check.
#: ``tests/test_dead_settings.py`` re-derives the set from the source on every
#: run, so the registry cannot rot in either direction: a key that gains a
#: reader must leave it, and a key that loses its last reader must join it.
#:
#: They stay declared rather than being deleted so that an old settings CSV
#: still loads far enough to be told, by name, what to use instead.
DEAD_SETTINGS = {
    'remove_border_cells': 'cell_remove_border_objects',
    'remove_border_nuclei': 'nucleus_remove_border_objects',
    'remove_border_pathogens': 'pathogen_remove_border_objects',
    'remove_border_organelles': 'organelle_remove_border_objects',
    'redunction_method': 'reduction_method',
    'signal_direction': 'single_direction',
    'class_1_threshold': 'score_threshold',
    'pick_slice': 'z_projection',
    'metadata_types': 'metadata_type',
    # Cellpose 4 segments everything with cpsam; the object's model is named
    # by its own <object>_model_name.
    'use_sam_cell': 'cell_model_name',
    'use_sam_nucleus': 'nucleus_model_name',
    'use_sam_pathogen': 'pathogen_model_name',
    # Mask post-processing is per-object and is driven by the
    # <object>_intensity_merge / _intensity_split / _area_multiplier group
    # that object.merge_split_filter_masks actually reads.
    'postprocess_cell_masks': 'cell_intensity_merge',
    'postprocess_nucleus_masks': 'nucleus_intensity_merge',
    'postprocess_pathogen_masks': 'pathogen_intensity_merge',
    'postprocess_organelle_masks': 'organelle_intensity_merge',
    # No working spelling: the behaviour these were meant to name does not
    # exist anywhere in spaCR.
    'gene_weights_csv': None,
    'nucleus_loc': None,
    'skip_mode': None,
}

tooltips = {
    "batch_correction": "(str) - Optional plate/batch correction applied before Image UMAP, ML screen classification, or phenotype regression. 'none' leaves measurements unchanged; 'center' removes each plate's mean shift; 'zscore' aligns plate means and variances; 'robust_zscore' uses median/MAD and tolerates outliers; 'control_center' estimates only a location shift from reference controls and best preserves treatment dispersion. Do not correct when plate is confounded with biology. Default 'none'. API: spacr.batch_correction.correct_batch_effects.",
    "batch_column": "(str) - Metadata column that identifies independent acquisition batches, normally 'plateID'. Every analyzed row must have a value and at least batch_min_samples rows must occur in each batch. Use an acquisition date or instrument ID only if that is the nuisance source you intend to remove. Default 'plateID'. API: spacr.batch_correction.correct_batch_effects.",
    "batch_control_column": "(str or None) - Metadata column containing reference-control labels for control_center, normally 'columnID' for plate controls. It is ignored by center, zscore, robust_zscore, and none. Blank follows col_to_compare in Image UMAP or location_column in Classify (ML); regression defaults to 'columnID'. API: spacr.batch_correction.correct_batch_effects.",
    "batch_control_values": "(str, number, list or None) - Reference/negative-control value(s) in batch_control_column used by control_center. Each plate needs at least batch_min_samples matching rows. Image UMAP falls back to neg and Classify (ML) to negative_control when this field is blank; regression requires an explicit value. Default varies by module. API: spacr.batch_correction.correct_batch_effects.",
    "batch_min_samples": "(int) - Minimum number of rows required in every batch, and minimum matching reference controls per batch for control_center. Correction stops with an actionable error below this threshold because a one- or two-object plate estimate is unstable. Default 3. API: spacr.batch_correction.correct_batch_effects.",
    "batch_missing_control": "(str) - Policy when control_center cannot find enough reference controls on a plate: 'error' stops rather than silently mixing corrected and raw plates; 'skip' leaves that plate unchanged and records a warning. Default 'error'. API: spacr.batch_correction.correct_batch_effects.",
    "threshold_direction": "(list, list-of-lists, int or None) - Which side of 'threshold' to keep when prefiltering objects for annotation: 'higher' keeps rows whose measurement is >= the threshold, 'lower' keeps rows <= it. Give one value, or one per entry in 'measurement' (a single string is broadcast to the whole list). Default 'higher'.",
    "threshold": "(list, list-of-lists, int or None) - Cut-off applied to 'measurement' before the annotation grid loads, so you only label the objects you care about. Accepts a number or a quantile code 'q1'-'q9' (q3 = the 30th percentile of that column), or one entry per measurement when measurement is a list. Empty or None loads every object unfiltered.",
    "cell_model_name": "(str) - Which weights segment cells. Cellpose 4 ships exactly one stock model, 'cpsam', so the only other value that means anything is a path to a checkpoint you trained yourself in Train Cellpose - that path is loaded as pretrained_model and honoured. The pre-SAM names ('cyto', 'cyto2', 'cyto3', 'nuclei') are still accepted so old settings files load, but they are mapped to 'cpsam' on the way in and reported once: Cellpose 4 would have resolved them to cpsam silently anyway. Note what Cellpose 4 does and does not still read: diameter IS honoured (eval rescales the image by 30/diameter), while model_type and diam_mean are accepted-and-ignored with a 'not used in v4.0.1+' log line. Default 'cpsam'.",
    "nucleus_model_name": "(str) - Which weights segment nuclei. 'cpsam' or a path to your own Train Cellpose checkpoint; there is no third option, because Cellpose 4 removed every pre-SAM model. 'nuclei'/'nucleus' from an older settings file is accepted and mapped to 'cpsam'. Set nucleus_diameter rather than expecting a nucleus-specific model - diameter is the parameter Cellpose 4 still acts on. Default 'cpsam'.",
    "pathogen_model_name": "(str) - Which weights segment pathogens. 'cpsam' or a path to your own Train Cellpose checkpoint. The bundled toxo_pv_lumen / toxo_cyto checkpoints were Cellpose-3 CPnet and cannot load into CPSAM's transformer, so they are mapped to 'cpsam' and reported. The older 'pathogen_model' key still overrides this one when set. Default 'cpsam'.",
    "cell_diameter": "(int or None) - Expected cell diameter in pixels. Cellpose 4 rescales the image by 30/diameter before segmenting, so setting it makes objects land near the size CPSAM was trained on; leave it None to segment at native scale. Set it when cells are much larger or smaller than ~30 px and masks come back fragmented or merged. spacr.diameter.estimate_diameters proposes a value from your own fields. Default None.",
    "nucleus_diameter": "(int or None) - Expected nucleus diameter in pixels, used by Cellpose 4 to rescale the image by 30/diameter before segmenting. None segments at native scale. Nuclei are usually the smallest object you segment, so this is the one most likely to need setting on low-magnification plates. spacr.diameter.estimate_diameters proposes a value. Default None.",
    "pathogen_diameter": "(int or None) - Expected pathogen diameter in pixels, used by Cellpose 4 to rescale the image by 30/diameter before segmenting. None segments at native scale. Intracellular parasites are often only a few pixels across at low magnification, where rescaling matters most. spacr.diameter.estimate_diameters proposes a value. Default None.",
    "diameter_estimate_n_fields": "(int) - How many fields spacr.diameter.estimate_diameters reads before it proposes cell_diameter, nucleus_diameter and pathogen_diameter from blob statistics instead of leaving you to guess. Fields are taken on an even stride across the sorted plate, so rows and columns are both represented rather than the first few wells; each field costs about a second of CPU and loads neither torch nor Cellpose. Raise it to 10-20 when wells vary a lot or the proposal comes back at low confidence, drop it to 2-3 for a quick look. Default 5.",
    "seg_qc": "(str) - Segmentation quality control, scored on the masks the moment they are written and long before measure_crop spends hours on them. 'off' skips it entirely; 'report' (the default) scores every field, writes qc/segmentation_qc_OBJECT.csv under the plate folder and prints a card naming the fields that are wrong and why; 'flag' does the same and additionally saves a per-field flags JSON for a downstream step to consume. No mode ever deletes, filters or skips a field: it tells you which ones are bad and leaves the decision to you. Default 'report'.",
    "seg_qc_min_objects": "(int) - Fields holding fewer objects than this are called near-empty, and their robust per-field size statistics are suppressed, because a median absolute deviation taken over a handful of objects is one object's opinion rather than a distribution. Raise it for confluent cell plates where every field should hold hundreds; drop it to 3-5 for low-multiplicity pathogen channels where two objects per field is genuinely the assay. Default 10.",
    "seg_qc_count_ratio": "(float) - How far a field's object count may drift from the plate median before the field is flagged, expressed as a fraction: 0.25 flags anything below a quarter of the median and, through its reciprocal, anything above four times it. Seeding density across a plate varies with a coefficient of variation of 10-30% and edge wells rarely fall below half, so a four-fold departure means lost focus, an empty well or a collapsed mask rather than biology. Default 0.25.",
    "seg_qc_size_ratio": "(float) - Fold change in a field's median object diameter, measured against the plate median, that marks it as fused or shattered when its object count has moved the opposite way. The default is the square root of two on purpose: two objects welded into one have exactly 1.41 times the equivalent diameter of one, and one object split in two has the reciprocal, so this is the signature itself rather than an arbitrary tolerance. Default 1.4.",
    "seg_qc_border_fraction": "(float) - Fraction of a field's objects allowed to touch the image edge before the field is flagged. Objects on the edge are truncated, so the crops handed to Measure are cut off and their areas understate the truth. Geometry alone puts roughly two object diameters' worth on the border, about 8% for 60 px cells in a 1400 px field, so this default is well clear of what a healthy field produces. Default 0.3.",
    "seg_qc_outlier_mad": "(float) - How many robust standard deviations, one of which is 1.4826 times the median absolute deviation, an object's diameter may sit from the field median before it counts as a size outlier. Median and MAD are used rather than mean and standard deviation because a few pieces of debris inflate a standard deviation until nothing looks unusual any more. Five is deliberately loose: real size distributions are heavier tailed than Gaussian, so three would flag part of every healthy field. Default 5.",
    "seg_qc_outlier_fraction": "(float) - Share of a field's objects that must fall outside the robust size range before the field is reported as holding two populations. A single distribution's tail cannot put fifteen percent of its objects five robust deviations out; only debris, fused pairs or fragments can, and those are what this check is looking for. Lower it if you want the card to be chattier about mixed fields. Default 0.15.",
    "seg_qc_foreground_fraction": "(float) - Foreground coverage at or above which a field is called confluent. That is the first half of the fusion test and the only condition under which the expensive distance-transform cross-check runs at all, so raising it makes QC faster and blinder while lowering it costs time on sparse fields. It matches the fused_fraction the diameter estimator uses, so both modules agree on what a dense field is. Default 0.35.",
    "seg_qc_split_ratio": "(float) - How many objects the distance transform has to resolve per mask object, in a field already judged confluent, before those masks are called fused. Two means every mask object contains at least two inscribed-circle maxima on average, which is the smallest fusion worth catching: neighbouring cells merged in pairs. Raise it toward five to report only wholesale collapse of a monolayer into slabs. Default 2.",
    "seg_qc_min_diameter": "(float) - Equivalent diameter in pixels below which an object is treated as a fragment rather than a real one; it drives the over-segmentation check and sets the seed floor of the fusion cross-check. Lower it to two or three for punctate organelles, where five-pixel objects are the actual signal rather than debris, and raise it for large cells where anything that small is certainly a shard. Default 5.",
    "seg_qc_tiny_fraction": "(float) - Share of a field's objects that may be smaller than seg_qc_min_diameter before the whole field is called over-segmented. Cellpose shattering one cell into a dozen shards takes this close to one, while a healthy field carrying a little debris sits well below a third, which is where the default sits. Default 0.3.",
    "seg_qc_max_object_fraction": "(float) - Share of the entire field a single label may cover before that label is read as evidence of fusion rather than as an object. A quarter of a field is not a cell, it is a monolayer that was welded into one mask, and the diameter estimator discards such components for exactly the same reason. Lower it for small objects on large fields; raise it only when one huge object per field is real. Default 0.25.",
    "seg_qc_plate_fail_fraction": "(float) - Fraction of failing fields at which the scorecard's verdict for the whole plate flips from warn to fail. Ten percent is roughly one column of a 96-well plate: below that you can drop the bad fields and still have the experiment, and above it what Measure would produce is no longer the experiment you ran. It changes the printed verdict only, never which fields are processed. Default 0.1.",
    "nucleus_CP_prob": "(float) - Cellpose cell-probability threshold for the nucleus channel, passed straight to model.eval as cellprob_threshold. A pixel must exceed it to join a mask, so raising it shrinks masks and drops dim nuclei, while lowering it grows masks and recovers faint ones along with more debris. Useful range about -6 to 6; default 0.",
    "pathogen_CP_prob": "(float) - Cellpose cellprob_threshold for the pathogen channel: a pixel is claimed by a mask only if its predicted object probability exceeds this. Lower it (toward -6) to recover dim or small parasites and grow mask boundaries; raise it (toward 6) to shrink masks and drop faint objects. Useful range about -6 to 6. Default 0.",
    "nucleus_FT": "(float) - Cellpose flow_threshold for nucleus masks: the maximum allowed error between a mask's recomputed flows and the network's predicted flows. Lowering it discards more irregularly shaped nuclei, giving fewer but cleaner objects; raising it keeps nearly everything Cellpose proposes. Typical range 0 to 3; spaCR default 1.0, which is permissive.",
    "pathogen_FT": "(float) - Cellpose flow_threshold for pathogen masks: a candidate mask is discarded when its recomputed flows disagree with the network prediction by more than this. Raise it to keep more, sometimes misshapen, parasites; lower it toward 0.4 (Cellpose's own default) to keep only clean, well-formed objects. Typical range 0.0-3.0. Default 1.0.",
    "cell_channel": "(int or None) - Zero-indexed raw acquisition channel that Cellpose segments into cell masks; it also selects which channel the cell_background, cell_Signal_to_noise and remove_background_cell settings are applied to during preprocessing. Set to None and no cell masks, cell table or cell crops are produced. At least one of cell/nucleus/pathogen/organelle_channel must be an integer or the run aborts. Default None.",
    "nucleus_channel": "(int or None) - Zero-indexed raw acquisition channel segmented into nucleus masks, and the channel that nucleus_background, nucleus_Signal_to_noise and remove_background_nucleus apply to. None means no nucleus masks, hence no nucleus table, no cell-to-nucleus linking, and nothing subtracted from the cytoplasm mask. Set it whenever a DNA stain was acquired. Default None.",
    "pathogen_channel": "(int or None) - Zero-indexed raw acquisition channel segmented into pathogen masks (Toxoplasma etc.), and the channel pathogen_background, pathogen_Signal_to_noise and remove_background_pathogen apply to. None disables pathogen segmentation, the pathogen table, the infected-only filter (uninfected) and the adjust_cells step, which needs cell, nucleus and pathogen masks together. Default None.",
    "nucleus_mask_dim": "(int) - Position along the last axis of each merged/*.npy array where the nucleus label mask sits, one plane after the cell mask. With the default four image channels (0-3) that is 5; keep a different number of channels and it shifts by the same amount. None makes measure_crop skip nucleus measurements and cell-to-nucleus linking. Default 5.",
    "batch_size": "(int) - How many images are held and processed together in one pass: field stacks during normalization and Cellpose segmentation, crops per step during classifier training and activation maps. Raising it speeds runs up but increases RAM/VRAM roughly linearly; lower it on out-of-memory errors. Defaults: 50 for mask generation, 64 for training.",
    "cell_FT": "(float) - Cellpose flow_threshold: the maximum allowed error between a candidate mask's recomputed flows and the network's predicted flows. Masks above it are discarded, so lowering it strips ragged or implausible cells but also loses real ones; raising it keeps more. Usable range about 0-3 (GUI allows -1 to 3). Default 1.0.",
    "cell_CP_prob": "(float) - Cellpose cellprob_threshold: only pixels whose predicted cell probability exceeds it are assigned to a mask. Raise it to shrink outlines and drop faint or spurious cells; lower it to grow outlines and recover dim ones. Valid range roughly -6 to 6, default 0. Lower it first when whole cells are missing.",
    "channels": "(list of int) - Zero-indexed image channels kept in merged/*.npy and measured by measure_crop; each entry produces its own <object>_channel_<n>_* intensity columns. The list length fixes where masks land, so cell/nucleus/pathogen_mask_dim must shift if you change it. Preprocessing silently resets it to range(n) when it does not match the number of channel folders found. Default [0,1,2,3].",
    "crop_mode": "(list) - Which mask each PNG crop is centred on: any of 'cell', 'nucleus', 'pathogen', 'cytoplasm' or 'organelle'. One crop set is written per entry into <mode>_png/ folders, so ['cell','nucleus'] doubles the images written and the rows added to png_list. A single png_size such as [224,224] is broadcast to every mode, and so are dialate_pngs and dialate_png_ratios - pass a list only when the modes need different values. A list shorter than crop_mode reuses its last entry for the rest and says so. Default ['cell'].",
    "custom_regex": "(str or None) - Python regex with named groups used to pull metadata out of raw image filenames. With metadata_type 'custom' the pattern actually compiled is '(<your regex>).<ext>' (<ext> = the most common image extension found in src) and it is consumed by _extract_filename_metadata, which needs the groups wellID, fieldID and chanID; plateID is optional (falls back to the source folder name) and timeID/sliceID are optional (absent means None). A filename missing a required group is skipped with 'Could not extract information from filename ... using provided regex', and a filename the pattern does not match at all is skipped too. With metadata_type 'auto' the regex is instead tried first to rename files into Yokogawa form, where only wellID is mandatory, falling back to automatic detection if it fails. Default None.",
    "cytoplasm": "(bool) - Derive a cytoplasm object per cell (cell mask with nucleus, pathogen and organelle pixels removed) and write it to its own cytoplasm table, which recruitment ratios such as pathogen/cytoplasm intensity are computed from. Requires a cell mask; measure_crop switches it on automatically whenever cell_mask_dim is set, so the value you enter is usually overridden. Default False.",
    "diameter": "(float) - (DEPRECEATED) Expected object diameter in pixels passed as model.eval(diameter=...) by the mask-finetune tool and by check_cellpose_models; Cellpose resizes each image by 30/diameter so objects match the network's ~30 px working size, so a value below the true size upscales the image and a value above downscales it. It is also handed to CellposeModel as diam_mean when custom_model is set, which the installed Cellpose 4.x ignores with a warning, and it seeds the diameter spinbox of the Qt live preview. It does not feed the segmentation pipeline's per-object diameters or its minimum/maximum size filters - those are built from magnification and cell_/nucleus_/pathogen_diameter. Default 30 (40 in check_cellpose_models).",
    "filter": "(bool) - Legacy switch for the old post-Cellpose cleanup pass, which re-ran size/intensity/border filtering and logged '_after_filtration' object counts to the database. The current Cellpose-SAM segmentation path never reads it, so toggling it changes nothing; use the per-object <object>_min_area, <object>_max_area and <object>_perimeter_fraction settings instead. Default False.",
    "magnification": "(int) - Objective magnification, used only to derive expected object sizes: pixel diameter is 2*mag+80 for cells, 0.75*mag+45 for nuclei and mag for pathogens, with min/max area limits of diameter^2/4 and diameter^2*10. Explicit cell_diameter, nucleus_diameter or pathogen_diameter override it. Set it to the objective actually used (10, 20, 40, 60). Default 20.",
    "metadata_type": "(str) - Which filename convention raw images are parsed with. 'cellvoyager' (default) and 'cq1' use built-in regexes, 'custom' uses custom_regex, and 'auto' first renames the whole folder into Yokogawa naming (with custom_regex if given, else automatic detection) before parsing. Choose wrong and plate/well/field/channel are misread, so images land in the wrong channel folders.",
    "n_jobs": "(int) - CPU workers for parallel stages: measurement, mask adjustment, DataLoader loading, and the sklearn/UMAP calls where -1 means every core. Raise it to shorten CPU-bound steps until RAM or disk I/O saturates. Note the measure-and-crop pipeline overrides your value with cpu_count()-4. Defaults vary by pipeline: cpu_count()-4, -1, or None.",
    "normalize_by": "(str) - Percentile source used to rescale cropped PNGs, and only active when 'normalize' is a [low, high] percentile pair: 'png' stretches each crop to its own percentiles, maximising per-object contrast; 'fov' uses percentiles from the whole field, keeping brightness comparable between objects. Choose 'fov' if crop intensities will be compared. Default 'png'.",
    "nuclei_limit": "(int, bool, or None) - Cap on nuclei per cell applied when the per-object tables are merged for analysis: None disables the filter, True keeps only single-nucleus cells, an integer N keeps cells with N or fewer nuclei. Cells over the cap are dropped entirely from the merged table. Do not pass False - it is read as 0 and removes everything. Defaults differ sharply by pipeline: 1 for plot-merge and recruitment, 2 for plot-data-from-db, True for screen analysis and training-dataset generation, 10 for endodyogeny, and 1000 (effectively off) for vision-model interpretation and class-proportion analysis - check the pipeline you are running rather than assuming.",
    "pathogen_limit": "(int, bool, or None) - Maximum pathogens per cell. True or 1 = single pathogen only; None or False = no limit; int = custom limit.",
    "masks": "(bool) - Run Cellpose segmentation for every object channel you defined (cell, nucleus, pathogen, organelle) and write label stacks to masks/<object>_mask_stack. Set False to do preprocessing only - build the normalized arrays now and segment later - but Measure will then have nothing to quantify. Default True.",
    "delete_intermediate": "(bool) - Legacy force-cleanup switch: when True it overrides keep_intermediate and keep_original_images so stack/, masks/, the numeric per-channel folders and the orig/ raw backup are all removed once merged/ is built. Cleanup is already the default, so this is only needed to beat those keep flags. Deletion is skipped unless every field of view reached merged/. Default False.",
    "save": "(bool or list of bool) - Whether to save masks to disk. Can be a list of three booleans for [cell, nucleus, pathogen] independently.",
    "reduction_method": "(str) - Dimensionality reduction run before clustering and plotting: 'umap' preserves more global structure and can be fitted on controls then applied to all data, 'tsne' emphasises local neighbourhoods and cannot reuse a fitted model. With 'tsne', min_dist is ignored and n_neighbors is used as perplexity. Anything else raises ValueError. Default 'umap'.",
    "test_size": "(float) - Fraction of the labelled single-object rows held out as the test split in the tabular ML classifier; the remainder trains the model. Raise it for a more trustworthy accuracy estimate, lower it when labelled data is scarce and you need the rows for training. Valid 0-1, default 0.2 (20% test).",
    "merge_pathogens": "(bool) - Legacy option that merged two touching pathogen labels into one when their shared boundary exceeded 66% of the smaller object's perimeter, so a single PV split by Cellpose counted once. The current Cellpose-SAM path ignores it - use pathogen_perimeter_fraction instead. Default False.",
    "resize": "(bool or float) - Resize every image to target_height x target_width before running Cellpose, then scale the returned mask back to the original dimensions with nearest-neighbour interpolation so measurements stay in original pixels. Turn it on to bring oversized fields to the scale a model was trained at, or to cut GPU memory. Requires target_height and target_width. Default False (True for plaque analysis).",
    "embedding_by_controls": "(bool) - Fit the reducer only on control wells - rows whose col_to_compare value equals pos or neg - and then project every object into that space. Use it when the axes should be defined by the control phenotypes so treatments are read relative to them; False fits on all objects. Default False.",
    "cam_type": "(str) - Which attribution map is computed. 'gradcam' weights the target_layer feature maps by their pooled gradients into a coarse heatmap of the region that drove the call; 'gradcam_pp' currently computes the identical map and only changes the output folder and table name. 'saliency_image' sums the absolute input gradient into one map; 'saliency_channel' keeps it per channel so you can see which stain mattered. Default 'gradcam'.",
    "target_layer": "(str) - Dotted attribute path to the convolutional layer whose activations and gradients Grad-CAM hooks, e.g. 'base_model.blocks.3.layers.1.layers.MBconv.layers.conv_b'; utils.recommend_target_layers(model) lists valid names. Later layers give class-specific but coarse maps, earlier ones finer detail. Required for 'gradcam'/'gradcam_pp' - it is auto-filled only when model_type is exactly 'maxvit', and left None it raises. Default None.",
    "shuffle": "(bool) - Shuffle the tar dataset in the DataLoader when generating activation maps, so each batch-grid PDF shows a mixed sample rather than consecutive files from one plate or class. Set False for a deterministic, file-order pass you can line up against the dataset listing. Default True.",
    "correlation": "(bool) - Correlate every input channel against every activation-map channel per image and write the result to the <cam_type>_correlations table: a Pearson coefficient plus Manders M1/M2 at each manders_thresholds percentile (15, 50, 75 by default). Use it to quantify which stain the model attends to instead of eyeballing heatmaps; it needs save=True to reach the database. Default True.",
    "mode": "(str) - Read-pairing strategy for barcode extraction: 'paired' locates target_sequence in R1 and in the reverse complement of R2 and merges them base-by-base into a quality-weighted consensus; 'single' scans one mate alone, chosen by single_direction. Paired calls barcodes more accurately but discards any read whose anchor is missing from either mate. Default 'paired'.",
    "signal_direction": "(str) - Intended to pick the FASTQ mate ('R1' or 'R2') scanned when mode is 'single', but nothing reads settings['signal_direction']; the barcode mapper reads single_direction instead, so setting this one leaves the run on whatever single_direction says. Use single_direction. Kept declared only so old settings CSVs still load, and rejected by the pre-flight check and by spacr-run --set.",
    "offset": "(int) - Offset from target_sequence to the first barcode, e.g. -8 if the barcode starts 8 bases upstream.",
    "expected_end": "(int) - Number of bases sliced out of each read starting at offset_start relative to the target_sequence hit; this window is what the regex is matched against. It must span the whole barcode block (column + gRNA + row) or the regex stops matching and reads are dropped; shorter reads are padded with 'N'. Default 89.",
    "infection_intensity_qc_scope": "(str) - Whether infection QC is fitted once or per group: 'combined'/'global'/'all' fits one model on everything, 'plate'/'per_plate' one per plateID, 'well'/'per_well' one per plate-well, and 'none'/'off' skips QC; an unrecognised string falls back to combined behaviour with a warning. Per-well fitting absorbs staining and exposure differences but needs enough cells per well; every group still writes its own QC plot, only the QC payload embedded in the summary panel is taken from the first processed group. Default 'per_well'.",
    "adjust_cells": "(bool) - After segmentation, rewrite the cell masks so labels split across a single pathogen or nucleus are merged, and cell fragments with no nucleus are absorbed into the neighbour they share most perimeter with. Needs cell, nucleus and pathogen channels and is skipped for timelapse runs. Enable when large infected cells come back fragmented. Default False.",
    "agg_type": "(str) - How per-object scores are collapsed to one value per well before regression: 'mean', 'median', 'quantile' (75th percentile), or None to skip aggregation and regress on individual objects. Median resists a handful of extreme cells; None keeps power but ignores within-well correlation. Forced to a per-well sum for poisson and to None for quantile. Default 'mean'.",
    "alpha": "(float) - Regularisation strength for the penalised models only: the L1 penalty for 'lasso', the L2 penalty for 'ridge', the combined penalty for 'elasticnet' and the inverse margin for 'hinge'. Larger values shrink more coefficients toward zero; set it to 'auto' or None to choose it by 5-fold cross-validation, which is usually what you want because the default 1 shrinks a fraction-scale design to nothing. Every other regression type refuses a non-default alpha rather than ignoring it, and it is no longer the quantile - see the quantile setting. Default 1.",
    "all_to_mip": "(bool) - Append an extra channel to every .npy in stack/ holding the pixel-wise maximum across that file's existing channels; the original channels are kept, not replaced. The new channel's index equals the previous channel count, so reference it through the channel/mask dim settings if you want to segment on it. Default False.",
    # --- 3D (Beta) -------------------------------------------------------
    # These describe what the z plumbing does, and say plainly where it stops.
    # A user must not read these and believe spaCR measures volumes today.
    "z_stack": "(bool) - Treat each field as a z-stack instead of a flat image, turning on the z_segmentation_mode / anisotropy / stitch_threshold controls below. Off, spaCR runs the ordinary 2-D path and none of the z code executes at all, so your masks are identical to a run from before this setting existed. On, spaCR requires an array that still has a z axis when it reaches segmentation and stops with an error if it does not; note that the standard image ingest currently collapses z into one plane per field while organising the raw files, so this setting only has something to work with when you feed spaCR volumes directly through the Python API. Default False.",
    "z_segmentation_mode": "(str) - Which of the three genuinely different ways to handle z is used, recorded alongside the masks because their answers are not comparable. 'project' collapses the stack with z_projection and segments the resulting single plane, which is what spaCR has always effectively done and the only mode whose masks the Measure module can consume. 'stitch' segments every plane independently in 2-D and then links labels down the stack by overlap (see stitch_threshold); it never computes a distance along z, so anisotropy does not enter it and an object invisible in one plane breaks in two. 'volumetric' segments the whole volume at once using the z gradient, which finds objects that no single plane shows but is acutely sensitive to a wrong anisotropy, which it therefore requires. Default 'project'.",
    "z_axis": "(int or None) - Which axis of the incoming array holds z, as 0, 1 or 2. None asks spaCR to work it out from the shape, which it can do only when one axis is clearly shorter than the other two (a 21x512x512 or 512x512x21 stack); for an ambiguous shape such as 64x64x64 it stops and asks rather than guessing, because guessing wrong segments a transposed volume and produces plausible nonsense. Set it explicitly whenever your acquisition's shape is ambiguous. Default None.",
    "z_projection": "(str or None) - How z is collapsed when z_segmentation_mode is 'project'. 'max' takes the brightest value down the stack and is the usual choice for sparse fluorescent objects; 'mean' averages, which suppresses noise but dilutes anything present in only a few planes; 'sum' preserves total signal so intensity stays proportional to how much of the object was in the stack; 'best_focus' discards every plane but the sharpest one, which beats a projection when only one plane is genuinely in focus and a MIP would smear the out-of-focus haze over it. Ignored by the other two modes. Default 'max'.",
    "anisotropy": "(float or None) - The ratio of the z step to the xy pixel size (dz / dxy), which is what tells 'volumetric' mode how far apart two planes really are. At the true value, objects separated by a few planes stay separate; left at 1.0 on a confocal stack, where the z step is routinely 3-10x the xy pixel, the segmenter reads a 5 um gap as a 5 pixel gap and fuses everything along z into columns. spaCR will not assume a value: leave this None and set voxel_size_z_um / voxel_size_xy_um instead and it is derived, but if neither is known a volumetric run stops rather than silently picking 1.0. Measure reads it too: on a 3-D mask it sets the z spacing for every regionprops and distance-transform call, so without it the 'outside' ring is as many planes thick as it is pixels wide. Default None.",
    "voxel_size_z_um": "(float or None) - Spacing between consecutive z planes in micrometres, straight off the acquisition settings. Together with voxel_size_xy_um it derives anisotropy, so setting these two is the safer way to get it right, and it is also what converts object volumes from voxel counts into um3. Changing it rescales every physical z quantity and the anisotropy used for segmentation; it has no effect on a 'project' run. Measure uses the pair to report 3-D morphology in micrometres rather than voxels, and records which it used in the measurement_units column. Default None.",
    "voxel_size_xy_um": "(float or None) - Width of one pixel in micrometres in the image plane, assumed square. Used with voxel_size_z_um to derive anisotropy and to turn voxel counts into physical volumes and surface areas. Note this is a different setting from um_per_pixel, which only sizes the scale bar drawn on figures and never reaches a measurement. This one does reach measurements, but only on a 3-D run: a 2-D run never applies it, because doing so would turn every *_area from px2 into um2 under an unchanged column name. Default None.",
    "stitch_threshold": "(float) - Minimum overlap, as an intersection-over-union between 0 and 1, for a label in one plane to be treated as the same object as a label in the plane below when z_segmentation_mode is 'stitch'. Raising it splits objects that drift or change shape between planes into several shorter ones; lowering it fuses neighbouring objects that merely overlap in projection. Matching is one-to-one, so when two objects both overlap the same object below only the better match inherits its label and the other starts a new one. Ignored by the other two modes. Default 0.25.",
    # --- 4D (Beta) -------------------------------------------------------
    # The time axis on top of the z axis. These say plainly where the 4-D
    # plumbing stops, for the same reason the 3D ones above do: a user must
    # not read them and believe spaCR tracks objects through volumes today.
    "t_stack": "(bool) - Treat each field as a time series of z-stacks rather than as one flat image, turning on the axis-order, backend and displacement controls below. Off, spaCR runs whatever 2-D or 3-D path you already had and not one line of the 4-D code executes, so your masks and tracks are identical to a run from before this setting existed. On, spaCR requires an array that still has both a time axis and a z axis when it reaches segmentation and stops with an error naming the cause if it does not; the standard image ingest writes one maximum-intensity plane per timepoint while organising the raw files, so this only has something to work with when you hand spaCR whole volumes through the Python API. Default False.",
    "t_axis_order": "(str or None) - Which of the two leading axes of your data is time and which is z: 'TZYX' for a stack per timepoint, 'ZTYX' for a time series per plane. Both are written by real microscopes and the array shape cannot distinguish them, so spaCR refuses to guess and stops until you say which you have. Getting this wrong does not crash: it links each object to whatever sits above it in the next z plane and reports that as motion, which produces smooth, entirely plausible, entirely fictional trajectories. A flat 2-D time series with no z at all is spelled 'TYX' - the one value that means \"there is no z axis\", since every other spelling claims one, so a flat acquisition is never the result of an ambiguity resolved in its favour. Leave it None only when you are setting t_axis directly instead. Default None.",
    "t_axis": "(int or None) - Index of the time axis in the incoming array, as an alternative to spelling out the whole order in t_axis_order; the z axis is then taken to be the other of the two leading axes, or whatever z_axis says. Use it for an acquisition whose axes are not in either of the two standard orders. When both this and t_axis_order are set they must agree, and spaCR stops if they do not rather than silently preferring one. Default None.",
    "frame_interval_s": "(float or None) - Seconds between consecutive timepoints, straight off the acquisition settings. It converts the frame index into a real time column in the tracks table and is what turns a displacement per frame into a speed; no linking decision depends on it, so a wrong value rescales reported velocities without changing which objects were joined to which. Left None, spaCR falls back to the motility module's seconds_per_frame rather than becoming a second source of truth for the same number. Default None.",
    "t_track_backend": "(str) - Which linker joins objects between consecutive timepoints once they have been segmented in 3-D. 'iou' overlaps whole volumes and needs no distance, no anisotropy and no tuning, but loses anything that moves further than its own width between frames; 'centroid' links nearest centroids under the displacement gate below and handles fast movement, at the cost of needing that gate set correctly; 'trackpy' does the same through trackpy's linker. The btrack, trackastra and ultrack backends all handle volumes upstream but spaCR's adapters for them require a flat time series, so asking for one of them on volumetric masks stops the run instead of quietly flattening it. Default 'iou'.",
    "t_link_threshold": "(float) - Minimum overlap, as an intersection-over-union between 0 and 1, for an object at one timepoint to be treated as the same object at the next when t_track_backend is 'iou'. Raising it breaks a moving or growing object into several short tracks; lowering it fuses neighbouring objects whose volumes happen to touch. Kept separate from stitch_threshold on purpose, because consecutive z planes and consecutive timepoints do not overlap by anything like the same amount. Matching is one-to-one, so two objects cannot both inherit one identity. Default 0.25.",
    "t_max_displacement_px": "(float or None) - How far an object may move between consecutive timepoints and still be considered the same object, in image pixels, for the distance-based backends. The z component of every displacement is multiplied by the anisotropy first, so that a one-plane move on a stack with a 5x z step counts as five pixels rather than one, which is what stops objects five times too far apart from linking. Set this or t_max_displacement_um but not both; spaCR will not pick a default, because too large fuses neighbours into one track and too small breaks one object into a track per frame. Default None.",
    "t_max_displacement_um": "(float or None) - The same maximum between-frame movement as t_max_displacement_px but expressed in micrometres, which is usually the number you actually know from the biology. It needs voxel_size_z_um and voxel_size_xy_um to convert, and once those are set it is the safer of the two because the anisotropy is already baked into the physical coordinates instead of being applied as a correction. Set this or t_max_displacement_px but never both, since they are one gate in two units. Default None.",
    "t_project_for_tracking": "(bool) - Collapse each timepoint's z-stack to one plane before linking, so tracking happens on the projection while segmentation still happened on the volume. Turn it on when the volumetric linking is too slow or you do not trust the anisotropy, and accept the cost: two objects that sit above one another become one object and nothing computed downstream can tell that it happened. It does not enable the backends spaCR cannot drive on volumes, which are refused whatever this is set to. Default False.",
    "save_original_images": "(bool) - After each batch is MIP-projected and merged into stack/, either move the raw input images into src/orig/ (True) or delete them so the pixels live only in stack/ (False). Set False on large screens where the duplicate raw copy will not fit on disk; the deletion is not reversible. Default True.",
    "keep_intermediate": "(bool) - Keep the intermediate stack/ and masks/ folders after the merged/ arrays are built. Off by default: only merged/ is kept (masks are embedded in merged and recorded in the database).",
    "keep_original_images": "(bool) - Keep the original raw input images (in orig/). Off by default to save disk space; the pixel data lives in merged/.",
    "amsgrad": "(bool) - Use the AMSGrad variant of Adam/AdamW, which keeps a running maximum of past squared gradients instead of their decaying average so the effective step size never grows back. Enable when training loss oscillates or stops converging with plain Adam; it costs a little speed and memory. Only honoured by optimizer_type 'adam' and 'adamw' - ignored by sgd, rmsprop, nadam, radam and adagrad. Default True.",
    "analyze_clusters": "(bool) - After the embedding is clustered, rank every measured feature by how well it separates the clusters - random-forest importance plus a per-feature ANOVA or Kruskal-Wallis test - and write results/cluster_results.csv. Turn it on when you want to know what morphology or intensity feature a cluster actually represents. It adds a full model fit over the whole feature table. Default False.",
    "augment": "(bool) - Expand the training split 8-fold by adding all four 90-degree rotations of each crop plus their horizontal mirrors; the validation and test splits are never augmented. Turn it on when you have few annotated objects and validation accuracy lags training accuracy. The expanded set is materialised in RAM, so expect roughly 8x the memory and 8x the epoch time. Default False.",
    "background": "(float) - Per-channel background level in raw intensity units. Pixels below it are zeroed when remove_background is on, and it is multiplied by Signal_to_noise to set the upper anchor for normalization. Raise it if faint haze survives; set it too high and dim real objects vanish. Default 100 (200 for Cellpose training and plaque analysis).",
    "backgrounds": "(str) - Background settings for the analysis.",
    "barcodes": "(str) - Path to a CSV of screen/plate barcodes for the legacy barcode-mapping helper. Nothing in the current code reads this key: get_map_barcodes_default_settings, the only place it is defined, is never called by any pipeline, so setting it has no effect. The live equivalents consumed by generate_barecode_mapping are row_csv, column_csv and grna_csv.",
    "black_background": "(bool) - Choose the standalone/CLI embedding fallback: black canvas with white axes when True, white canvas with black axes when False. In the Qt app, Image UMAP automatically matches its enclosing card in the active theme and uses that theme's readable foreground color instead. Default True.",
    "calculate_correlation": "(bool) - For every pair of measured channels and every object mask, compute a per-object Pearson correlation plus Manders M1/M2 at each cut-off in manders_thresholds, stored as <object>_channel_i_channel_j_* columns. Needs at least two channels. Turn it off to cut measurement time and database size when colocalisation is not part of the phenotype. Default True.",
    "cell_background": "(int) - Background intensity of the cell channel in raw image units. Pixels below it are zeroed when remove_background_cell is True, and it is multiplied by cell_Signal_to_noise to set the intensity the normalisation ceiling must reach. Set it from a genuinely empty region; too high and dim cells are erased. Default 100.",
    "nucleus_background": "(int) - Raw intensity value treated as background in the nucleus channel. When remove_background_nucleus is True, every pixel below it is zeroed before normalization; it is also multiplied by nucleus_Signal_to_noise to set the upper-clip target. Raise it for images with high offset or autofluorescence, lower it if dim nuclei disappear. Default 100.",
    "pathogen_background": "(int) - Assumed background intensity of the pathogen channel in raw image units. It has two jobs: when remove_background_pathogen is True every pixel below it is zeroed, and it is multiplied by pathogen_Signal_to_noise to set the brightness the normalisation ceiling must reach. Raise it if dim haze is being segmented; lower it if faint parasites vanish. Default 100.",
    "cell_chann_dim": "(int) - Recruitment analysis only (analyze_recruitment): the image-channel index paired with the cell mask when drawing outline overlays, and the switch that enables the cell filters - set an integer and cell_size_range, cell_intensity_range and target_intensity_min are applied; leave it None and cells are not filtered at all. Default 3.",
    "cell_intensity_range": "(list) - [min, max] bounds on a cell's mean intensity, applied when the measurement table is filtered during recruitment analysis (only when cell_chann_dim is set). Beware the channel it actually tests: _object_filter is called with mask_chans=[nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim] and index 0, so the column compared is cell_channel_{nucleus_chann_dim}_mean_intensity - the cell object measured in the NUCLEUS channel, not the cell channel. Bounds are exclusive (a cell is kept when its value is > min and < max) and both entries must be integers, since a float or None silently skips that bound. Units are raw image intensity. Default [0, 100000].",
    "cell_loc": "(list) - One list of well identifiers per entry in cells, giving where each host cell line sits on the plate, e.g. [['c1','c2'],['c3']]. Identifiers must start with 'r' for a row or 'c' for a column; anything else is ignored and those wells stay unannotated. Pass None to label every row with the first entry of cells. No default is set: no set_default_* function in settings.py touches this key, and its only reader, annotate_filter_vision, indexes settings['cell_loc'] directly, so it must be present in the dict (explicitly None if you do not want location mapping) or the call raises KeyError.",
    "cell_mask_dim": "(int) - Position along the last axis of each merged/*.npy array where the cell label mask sits. Merged arrays are ordered [image channels..., cell, nucleus, pathogen, organelle], so the default 4 assumes the four channels 0-3 were kept; keep fewer channels and every mask dim shifts down. None makes measure_crop skip all cell measurements and cell crops. Default 4.",
    "cell_min_size": "(int) - (Depreceated) Pixel-area floor applied to cell labels during measurement: any cell smaller than this is erased from the mask before features are extracted. Superseded by cell_min_area, which filters at segmentation time, but this one still runs if you set it. 0 or None disables it. Default 0.",
    "cell_plate_metadata": "(list of lists) - Plate wells occupied by each entry of cell_types, one inner list per cell type in the same order, e.g. [['c2','c3'],['c4']]. Every identifier must start with 'c' (column) or 'r' (row); anything else is silently skipped and those wells get no host_cells label. An unlabelled well is not necessarily lost: 'condition' is the join of whichever of host cell / pathogen / treatment labels are present, so analyze_recruitment only drops rows that have none of the three - a well missing just the cell label lands in a different condition group instead. plot_data_from_db is stricter and does drop rows with no host_cells label whenever this key is set. Default None, which labels every row with cell_types[0].",
    "cell_Signal_to_noise": "(int) - Multiplied by cell_background to give the intensity the normalisation ceiling must reach: spaCR walks the 98th to 99.5th percentile of the cell channel and takes the first value at or above that product as the upper anchor. Raise it to push the ceiling higher and dim the normalised image; lower it to brighten faint cells. Default 10.",
    "cell_size_range": "(list) - [min, max] bounds in pixels^2 on cell_area, used to drop rows from the measurement table during recruitment analysis; only cells strictly between the two values are kept. Both entries must be integers or that bound is silently skipped. Setting it to None widens it to [0, 1e100]. Default [0, 100000].",
    "cell_types": "(list) - Names of the host cell lines in the experiment, e.g. ['HeLa']. Each name is written into the host_cells column and folded into the combined condition label used for grouping and plotting; the list is positionally paired with cell_plate_metadata, which says which wells hold each one. Default ['HeLa'].",
    "cells": "(list) - Names of the host cell lines on the plate, e.g. ['HeLa']. Each name is written to the host_cells column and becomes part of the combined condition label used for grouping in plots and statistics. With cell_loc set the names are mapped well by well; with cell_loc None only the first name is used, applied to every row.",
    "cells_per_well": "(int) - Minimum cells a well must contribute to survive recruitment analysis; wells below it, and every cell in them, are dropped before the by-well plots and CSVs are produced. Raise it to suppress noisy, sparsely populated wells at the cost of losing those wells. Default 0, which keeps every well.",
    "channel_dims": "(list) - Recruitment analysis only: the image-channel indices held in the merged arrays. They are handed to plot_image_mask_overlay so the overlay/outline figures cover those channels, and the same list drives the recruitment loop - but _calculate_recruitment writes fixed, channel-less column names ('pathogen_cell_mean_mean', 'pathogen_cytoplasm_q75_mean', ...), so each pass overwrites the previous one: the column count never changes and only the LAST index in the list determines which channel's recruitment ratios survive. Put the channel you actually want ratios for last, and trim the list only to cut plotting work. Default [0,1,2,3].",
    "channel_of_interest": "(int) - Index of the fluorescence channel the downstream analysis focuses on. It decides which channel's features survive filtering (other channels' features are dropped), defines recruitment = pathogen_channel_N_mean_intensity / cytoplasm_channel_N_mean_intensity, and is written into the ML result paths. Set it to the channel carrying your phenotype readout. Valid 0-3; default 3 in the ML/recruitment steps, 1-2 elsewhere.",
    "chunk_size": "(int) - Number of FASTQ reads read into memory and handed to each worker batch. Larger chunks cut per-batch overhead and make the progress bar coarser but raise peak RAM per job; smaller chunks stream more gently on low-memory machines. Also sets how many reads are processed when test is True. Default 100000.",
    "classes": "(list) - Ordered class names. Each must exactly match a subfolder under src/train and src/test; a name's position in this list becomes its integer label, and the list length sets the width of the classifier head. Training raises a FileNotFoundError listing missing vs available folders if a name has no folder. Generate Training Dataset overwrites this with the class names it actually wrote to disk. Default ['nc','pc'].",
    "class_1_threshold": "(float) - Intended as the probability cut-off above which an object is called class 1, but nothing reads settings['class_1_threshold'] and no defaults function sets it, so the hard call is unaffected by it. score_threshold is the key the classifier actually applies when it turns the model's probability into cv_predictions. Kept declared only so old settings CSVs still load, and rejected by the pre-flight check and by spacr-run --set.",
    "clustering": "(str) - Algorithm run on the 2D embedding. 'dbscan' grows density-based clusters from eps and min_samples and labels sparse points as noise (-1), discovering the cluster count itself; 'kmeans' (that exact spelling) instead forces exactly min_samples clusters and assigns every point. Choose dbscan for a few distinct phenotypes over a diffuse background, kmeans when you want a fixed number of groups. Default 'dbscan'.",
    "col_to_compare": "(str) - Metadata column that identifies the control wells when embedding_by_controls is True: rows whose value equals pos or neg are used to train the reducer, and the column is then dropped before fitting. Typically 'columnID' or 'rowID' depending on where controls sit on the plate. Ignored otherwise. Default 'columnID'.",
    "color_by": "(str) - Name of a column in the joined measurement table (e.g. 'cond', 'columnID', 'plateID') used to color embedding points instead of the cluster labels. Set it to see how a known grouping such as condition or plate column falls across the map; leave it None to color by the clustering result. Setting it also disables remove_cluster_noise, plot_outlines and smooth_lines. Default None.",
    "compartments": "(list) - Intended to name the object compartments ('cell', 'nucleus', 'pathogen', 'cytoplasm') to measure, but nothing reads settings['compartments']: the functions with a same-named parameter take it directly from their caller. Which compartments are measured is decided by the *_mask_dim / *_channel settings instead.",
    "consolidate": "(bool) - Before processing, recursively scan src for images and copy them into a single <src>/consolidated folder, prefixing each filename with its subfolder names so nothing collides; src is then repointed there. Use it when one plate's images are split across per-well or per-channel subfolders. Copies, so disk use roughly doubles. Default False.",
    "CP_prob": "(float) - Cellpose cellprob_threshold: the cell-probability cut-off applied to the network output when deciding which pixels belong to an object. Lower it (typically toward -6) to recover dim or partly detected objects and grow existing masks; raise it (toward 6) to drop faint false positives and shrink masks. Default 0.",
    "custom_model": "(str) - Filesystem path to a saved Cellpose model, loaded as pretrained_model by the mask-finetune tool (analyze_plaques sets it internally to the bundled plaque model). When set, model_type is passed as None and diameter is passed as diam_mean (ignored by Cellpose 4.x with a warning), but model_name is still read: it selects the channel pair sent to model.eval - cyto2 -> [2,1], nucleus -> [0,0], cyto -> [1,0], anything else [2,0], overridden to [0,0] when grayscale is True. If the path does not exist the run prints 'Custom model not found' and returns without segmenting any image. None builds the stock model instead. Default None (the classifier-training defaults set a same-named boolean that nothing reads).",
    "cytoplasm_min_size": "(int) - (Depreceated) Pixel-area floor for the cytoplasm mask, which is the cell mask with nucleus, pathogen and organelle pixels removed. Cytoplasm regions below this are erased before measurement, so their host cell yields no cytoplasm features and any recruitment ratio built on them is lost. 0 or None disables. Default 0.",
    "nucleus_min_size": "(int) - (Depreceated) Minimum nucleus size in pixels^2 applied during measure_crop: labels covering fewer pixels than this are erased from the nucleus mask before any feature is measured, so those nuclei never reach the database. 0 (default) disables it. Prefer nucleus_min_area, which filters at segmentation time.",
    "dependent_variable": "(str) - Name of the column in score_data that is modelled as the response, e.g. 'pred'/'predictions' from the ML scoring step or a measured feature such as 'pathogen_nucleus_shortest_distance'. It is aggregated per well by agg_type and then optionally transformed. The run aborts if the column is absent from the score CSV. Default 'pred'.",
    "score_column": "(str) - Which column of the per-object score CSV minimum_cell_simulation resamples when it works out how many objects a well needs before its mean stops moving. It must name the same measurement as dependent_variable, or the simulated min_cell_count describes a different quantity than the one the regression fits and wells are kept or dropped on the wrong evidence; the regression defaults therefore follow dependent_variable. In the interpret-vision-model helper the same key names the CNN score column instead, default 'cv_predictions'.",
    "tolerance": "(int or float) - How close a subsampled well mean has to be to the full-well mean before minimum_cell_simulation calls that sample size sufficient, which is what sets min_cell_count when you leave it None. An int is read as a percentage (2 means 2%), a float as a fraction (0.02 means the same); anything else raises ValueError. Tighten it toward 0.01 to demand more cells per well and drop more wells, loosen it to 0.05 to keep sparse wells at the cost of noisier per-well scores. Default 0.02.",
    "invert_dependent_variable": "(bool or int) - Flip the response before it is aggregated per well, for scores whose useful direction is downward. False or 0 leaves it as measured, True or 1 uses 1 - x (right for a probability, so a low infection score becomes a high phenotype), and -1 uses 1 / x (right for a distance or a count). Any other value raises ValueError in process_scores. It changes the sign of every coefficient and therefore which side of the volcano your hits land on. Default False.",
    "y_lims": "(list or None) - Limits of the -log10(p) axis of the Toxoplasma volcano plot. None auto-scales to the data; [low, high] fixes the axis so several plates can be compared at the same scale; [[low1, high1], [low2, high2]] draws a broken axis with the gap between the two ranges removed, which keeps a handful of extremely significant genes on the plot without flattening everything else. Any other shape raises ValueError. Default None.",
    "dialate_png_ratios": "(list of float) - Dilation amount as a fraction of object size: the mask is grown by ratio * sqrt(object area) pixels of binary dilation, so 0.2 expands a cell by roughly 20% of its diameter and pulls in surrounding background. Only used when dialate_pngs is True. A single value applies to every crop_mode entry; pass a list only when the modes need different ratios. Default [0.2].",
    "dialate_pngs": "(bool) - Grow each object mask before cropping so the PNG keeps a rim of surrounding pixels instead of a hard mask edge; the amount comes from dialate_png_ratios. May be a list with one value per crop_mode entry (a single value applies to all of them), and is forced off for crop_mode 'cytoplasm'. Enable when context around the object helps the classifier. Default False.",
    "dot_size": "(int) - Matplotlib marker area, in points squared, for each object plotted in the UMAP/tSNE embedding. Increase it when a few hundred points make the scatter look empty; drop it to roughly 5-10 when tens of thousands of points overplot and hide cluster structure. Default 50.",
    "point_color": "(str) - Point color for static and interactive UMAP plots. Use 'cluster' or 'viridis' for cluster-based Viridis colors, or any Matplotlib color such as '#4cc9f0', 'orange', or 'white' for one fixed color. Default 'cluster'.",
    "point_alpha": "(float) - Opacity of UMAP points from 0 (invisible) to 1 (opaque), used by both static and interactive plots. Default 0.65.",
    "outline_width": "(float) - Width in points of cluster outlines and interactive selection rings. Smaller values produce thinner boundaries. Default 1.0.",
    "umap_canvas_width": "(int) - Initial interactive UMAP chart width in pixels. The chart/sidebar divider can also be dragged while exploring. Default 900.",
    "umap_sidebar_width": "(int) - Initial interactive UMAP image and annotation sidebar width in pixels. The divider remains draggable. Default 280.",
    "downstream": "(str) - Inert: nothing reads settings['downstream'], and sequencing.py never mentions it. The default is the reverse complement of the column primer this was meant to anchor on; the barcode reader uses target_sequence plus offset_start instead. Kept only so old settings CSVs still load.",
    "dropout_rate": "(float) - Dropout probability (0-1) written into every existing Dropout layer of the backbone and applied to a Dropout inserted before the final linear classifier; 0 or None removes dropout entirely. Raise it (0.2-0.5) when training accuracy runs well ahead of validation accuracy; lower it when the model underfits and training loss stalls high. Default 0.1.",
    "eps": "(float) - DBSCAN neighbourhood radius, expressed in the units of the UMAP/t-SNE embedding and measured with the 'metric' setting: two points are neighbours if they lie within this distance. Raise it to merge fragments into fewer, larger clusters and leave less noise; lower it to split clusters and push more points to noise (-1). Ignored when clustering is 'kmeans'. Default 0.9.",
    "epochs": "(int) - Number of full passes over the training set. It also sets the learning-rate schedule horizon - cosine anneals over exactly this many epochs and step_lr drops every epochs/5 - so changing it rescales the schedule. A checkpoint is always written on the final epoch and every 100th. Raise it for small datasets and use early_stopping_patience to cut runs short. Default 100.",
    "examples_to_plot": "(int) - How many randomly chosen merged image stacks are rendered as segmentation-overlay previews after mask generation (in timelapse mode, per-channel panels instead). Raise it to check outlines and normalization across more fields of view, at the cost of render time and larger PDFs; 0 skips previews entirely. Default 1.",
    "exclude": "(str or list) - Names of measurement columns to drop from the feature set before UMAP embedding or ML training, applied after the channel_of_interest selection. Use it to remove features that leak the label or swamp the embedding. It does not filter database rows; use exclude_rows for that. Default None keeps every feature.",
    "exclude_conditions": "(list) - Condition labels dropped from the image UMAP input, matched against the cond column that map_condition derives from the pos, neg and mix column IDs; the only possible entries are 'neg', 'pos', 'mix' and 'screen'. A bare string is accepted and wrapped in a list. Use it to embed screen wells only. Default None.",
    "exclude_rows": "(dict or None) - General UMAP row exclusions. Choose one or more database columns, then check the values whose rows should be removed. Rules are combined with OR, so a row matching any selected column/value pair is excluded. Default None keeps every row.",
    "experiment": "(str) - Free-text run label. Its real effect is naming the exported PNG dataset tar as <YYMMDD>_<experiment>.tar (a random-numbered variant is used if that name already exists), so give each screen a distinct value to avoid confusing dataset tars. It is also passed to the measurement-database writer but not stored there. Defaults vary by pipeline: 'exp', 'exp.' or 'experiment_1'.",
    "figuresize": "(int) - Base figure size in inches; figures are built square as figuresize x figuresize and font sizes are derived from it (legend, axis labels and ticks at 0.75x, overlay text at 0.5x). Raise it when text is unreadable at publication scale, lower it to fit panels on screen. Default 10; cluster grids cap total width at 200 inches.",
    "filter_by": "(str or None) - Restricts the feature matrix before dimensionality reduction: only columns matching this channel are kept and the other channel_1-channel_4 columns are dropped. Accepts 'channel_0'-'channel_3', an int, a list of channel numbers, or 'morphology' to keep only shape features (area, eccentricity, Zernike moments, ...). None, 'None', 'all', and '*' disable filtering. Default 'channel_0'.",
    "fill_in": "(bool) - Post-process each Cellpose mask in the mask-finetune / plaque tool with fill_holes_in_mask: the mask is first re-labelled by connectivity over all non-zero pixels (scipy.ndimage.label), then interior holes are filled component by component. Because re-labelling ignores the original label values, objects that touch are merged into a single object and every object is renumbered 1..n, so enable it when punched-out interiors (dark vacuoles, nuclei inside a cell) should count as object area and object identity does not matter, and disable it when touching objects must stay distinct. Default True.",
    "flow_threshold": "(float) - Cellpose flow_threshold: the maximum allowed error between the predicted flow field and the flows recomputed from each candidate mask; masks above it are discarded. Raise it to keep more objects, including irregularly shaped ones; lower it to reject poorly formed masks and reduce false positives. Default 0.4.",
    "fps": "(int) - Playback rate of the per-channel movies written to <src>/movies from timelapse .npy stacks, and only when timelapse is True. Raise it to skim long acquisitions, lower it to inspect individual frames. Affects the movies only - never tracking, segmentation or measurements. Default 2.",
    "fraction_threshold": "(float) - Minimum relative abundance, 0-1, that a gRNA must reach within a well's total read count to be kept. Raising it strips low-abundance and bleed-through gRNAs and lowers the mean gRNAs per well; set it too high and every row is removed and the run errors out. Leave None to auto-pick the cutoff giving target_unique_count gRNAs per well. Default None.",
    "from_scratch": "(bool) - Whether to train the Cellpose model from scratch.",
    "gene_weights_csv": "(str) - Intended to point at a CSV of per-gene weights for the screen regression, but nothing reads settings['gene_weights_csv']. Gene-level effects are derived from the fitted coefficients instead. Kept only so old settings CSVs still load.",
    "gradient_accumulation": "(bool) - Sum gradients over several batches before each optimizer step instead of stepping on every batch, giving an effective batch size of batch_size x gradient_accumulation_steps without extra GPU memory. Enable when you had to shrink batch_size to fit in VRAM and training is noisy. Leftover gradients are flushed at the end of each epoch. Default True.",
    "gradient_accumulation_steps": "(int) - How many batches are summed per optimizer step when gradient_accumulation is on; the loss is divided by this value so gradient magnitude stays comparable. Effective batch size = batch_size x this. Raise it (4-16) to emulate a larger batch on limited VRAM, at the cost of fewer weight updates per epoch. Ignored when gradient_accumulation is False. Default 4.",
    "grayscale": "(bool) - Force the Cellpose channel pair to [0, 0] so the network treats the input as a single combined channel, overriding the [cytoplasm, nucleus] pair otherwise inferred from model_name (cyto -> [1,0], cyto2 -> [2,1], nucleus -> [0,0]). Leave it on for single-channel inputs; switch it off only when feeding a genuine two-channel stack. Default True.",
    "grna": "(str) - Path to a CSV of gRNA barcode sequences for the legacy barcode-mapping helper. Like 'barcodes' it exists only in get_map_barcodes_default_settings, which no pipeline calls, so changing it has no effect on any run. The live equivalent, read by generate_barecode_mapping, is grna_csv.",
    "grouping": "(str) - How per-object values collapse to one number per well in the plate heatmap: 'mean' averages heatmap_feature over the objects in a well, 'sum' totals them, 'count' ignores the feature and colors wells by object count. Use 'count' to spot uneven seeding or dropout, 'mean' for phenotype strength. Default 'mean'; any other value raises ValueError.",
    "heatmap_feature": "(str) - Numeric column that is aggregated per well and color-mapped in the plate heatmap after ML scoring, e.g. 'predictions' for the classifier score or 'recruitment' for the pathogen/cytoplasm intensity ratio. Must be a numeric column of the scored dataframe or the run raises ValueError listing the valid names. Default 'predictions'.",
    "homogeneity": "(bool) - Compute grey-level co-occurrence-matrix homogeneity for every object in every channel, adding one homogeneity_distance_<d> column per entry in homogeneity_distances. Homogeneity is high for smooth, evenly filled objects and low for punctate or grainy ones, so keep it on for texture phenotypes; disabling it noticeably speeds up measurement. Default True.",
    "homogeneity_distances": "(list) - Pixel offsets used to build each object's grey-level co-occurrence matrix; every entry adds one homogeneity_distance_<d> feature per channel. Small offsets capture fine-grained texture, large ones capture coarse structure, and offsets larger than the object itself carry no signal. More entries means more features and slower measurement. Default [8, 16, 32].",
    "image_nr": "(int) - How many example object crops to draw on the embedding plot: that many per cluster when plot_by_cluster is on (smaller clusters show all they have), otherwise that many sampled at random overall. It also sets how many images each cluster contributes to the cluster-grid figure. Raise it for a fuller montage, lower it when thumbnails hide the points. Default 16.",
    "image_size": "(int) - Side length in pixels of the centre crop taken from each object PNG before it reaches the model. Images are cropped, not rescaled, so a larger value zero-pads and a smaller one throws away the object's edges. It is also the resolution the backbone is built at, which matters for ViT/Swin/inception. Match it to the crop size used when the dataset was generated. Default 224.",
    "img_zoom": "(float) - Scale applied to each object thumbnail pasted onto the embedding: 1.0 draws the crop at native pixel size, 0.5 at half. Raise it when crops are too small to judge morphology, lower it when thumbnails overlap and bury the point cloud. Practical range about 0.1-2.0. Default 0.5.",
    "uninfected": "(bool) - Decides which cells survive the consistency filter in measure_crop. True keeps any cell that has both a nucleus and a cytoplasm; False also demands at least one pathogen, dropping uninfected cells from every table. Either way, nucleus/pathogen/cytoplasm labels outside the surviving cells are zeroed. Only applied when cell, nucleus and pathogen masks all exist; forced True otherwise. Default True.",
    "init_weights": "(bool) - Start the backbone from ImageNet-pretrained weights instead of random initialisation; the spaCR classifier head bolted on top is randomly initialised either way. Leave it on - transfer learning converges in far fewer epochs on the small annotated sets typical here. Turn it off only to train from scratch on a very large dataset, or to measure how much pretraining contributes. Default True.",
    "intermedeate_save": "(bool) - Intended to control whether extra checkpoints are written mid-run when validation accuracy crosses 99, 98, 95 or 94 percent, on top of the final-epoch save. It currently has no effect: train_model passes that threshold list to the saver unconditionally, so those checkpoints are written regardless of this flag. Default True.",
    "invert": "(bool) - Invert intensities as each image is loaded, pixel -> dtype_max - pixel (255 - x for uint8). Switch it on for brightfield or phase-contrast data where objects are darker than the background, since Cellpose expects bright objects on a dark field; leave it off for fluorescence. Default False.",
    "learning_rate": "(float) - Step size passed to the optimizer. Too high and the loss spikes or flatlines at chance; too low and training crawls or settles in a poor minimum. 1e-3 suits training from scratch, while 1e-4 to 1e-5 is safer when fine-tuning ImageNet weights (init_weights=True). The chosen schedule decays this starting value over the run. Default 0.001.",
    "location_column": "(str) - Metadata column searched for the positive_control and negative_control values when labelling rows for ML training, normally 'columnID' or 'rowID'. Set 'rowID' when your controls run along plate rows instead of columns. It is overwritten with annotation_column whenever that is set. Default 'columnID'.",
    "log_data": "(bool) - Apply log(x + 1e-6) to every numeric feature, after the correlation filter and before standard scaling. Compresses heavy-tailed measurements such as intensity sums and areas so a handful of bright or huge objects stop dominating the embedding. Negative feature values become NaN and are then filled with the column mean. Default False.",
    "lower_percentile": "(float) - Percentile of the non-zero pixels in each channel used as the low anchor when rescaling that channel to 0-1; the high anchor is chosen automatically between the 98th and 99.5th percentile. Raise it to crush more dim background to black, lower it to preserve faint signal. Valid 0-100, default 2.",
    "manders_thresholds": "(list) - Percentiles (0-100) at which Manders' overlap coefficients are computed. For each object, each entry thresholds both channels at that percentile; pixels above both count as overlap, and M1/M2 report each channel's fraction of total object intensity there, saved as M1_correlation_<t> and M2_correlation_<t>. High values isolate the brightest puncta. Requires calculate_correlation. Default [15, 85, 95].",
    "mask": "(bool) - Whether to generate masks for the segmented objects. If True, masks will be generated for the nucleus, cell, and pathogen.",
    "measurement": "(str) - Measurement column(s) from measurements.db used to prefilter which object crops the annotator loads, applied together with threshold and threshold_direction. Accepts a single column, a comma-separated list (each paired with the same-index threshold), or a JSON list-of-lists where an inner pair is filtered as a ratio (first divided by second). Empty (default) loads every crop unfiltered.",
    "metadata_types": "(list) - Intended to list which metadata columns ('columnID', 'rowID', \u2026) to attach to each object, but nothing reads settings['metadata_types'] and its own default line in settings.py is commented out. Metadata parsing is driven by metadata_type and the filename regex instead. Kept only so old settings CSVs still load.",
    "merge_edge_pathogen_cells": "(bool) - During measurement, reconcile pathogens straddling two host-cell masks: if 90 percent or more of the pathogen lies in one cell, its pixels in the neighbours are erased; otherwise the overlapping cell labels are fused into a single cell. Switch off to keep the raw cell segmentation when parasites legitimately touch two cells. Default True.",
    "metric": "(str) - Distance metric used both by the reducer (UMAP or t-SNE) and by DBSCAN clustering, e.g. 'euclidean', 'manhattan', 'cosine' or 'correlation'. Correlation-type metrics compare feature profiles regardless of magnitude and often separate phenotypes better than euclidean on scaled data. Default 'euclidean'.",
    "min_cell_count": "(int) - Wells with fewer than this many scored objects are dropped before regression. Raising it removes noisy, sparsely imaged wells at the cost of statistical power. Leave None and spaCR simulates the count at which a well's mean score stabilises within tolerance and uses that value. Default None.",
    "min_dist": "(float) - UMAP's minimum spacing between points in the 2-D embedding, range 0.0-1.0. Low values (0.0-0.1) let clusters pack tightly and look crisply separated; higher values spread points out and preserve more of the global layout at the cost of visible cluster structure. Ignored when reduction_method is 'tsne'. Default 0.1.",
    "min_max": "(str) - Color limits for the plate heatmap: 'allq' scales to the 2nd-98th percentile of well values so a handful of extreme wells cannot flatten the rest, 'all' scales to the true min and max. A two-element list is also accepted, where floats are read as quantiles and integers as absolute vmin/vmax. Default 'allq'.",
    "min_samples": "(int) - Meaning depends on 'clustering': for DBSCAN it is how many points must fall within eps for a point to count as a core point, so raising it yields fewer, denser clusters and more noise; for KMeans this same value is reused as n_clusters, the exact number of clusters produced. Lower it (or raise eps) when no clusters are found. Default 100.",
    "mix": "(str) - Plate column ID whose wells hold a mixed positive/negative population; rows with this columnID are labelled cond='mix' for the image UMAP, so they can be coloured separately or dropped via exclude_conditions. Any column matching none of pos, neg or mix is labelled 'screen'. Default 'c3'.",
    "model_name": "(str) - Cellpose model to segment with. Cellpose 4 ships exactly one, 'cpsam'; the pre-SAM names ('cyto', 'cyto2', 'cyto3', 'nuclei') are accepted so old settings files load, but they are mapped to 'cpsam' and reported, because Cellpose resolves them to cpsam silently anyway. Of the three parameters that used to distinguish models, only diameter still does anything under Cellpose 4 (eval rescales the image by 30/diameter); model_type and diam_mean are logged as 'not used in v4.0.1+' and dropped. Leave at 'cpsam' unless you are loading a custom CPSAM checkpoint. Default 'cpsam'.",
    "model_type": "(str) - Backbone architecture for the single-object image classifier, passed to choose_model: any TorchVision classification model name (resnet50, maxvit_t, densenet121, ...). An unrecognised name is not fatal at call time - choose_model prints 'Invalid model_type' and returns None, so training then fails; the special name 'custom' passes the name check but raises NotImplementedError. Bigger backbones capture subtler phenotypes but cost VRAM and epochs, and the name becomes part of the output model folder path (src/model/<model_type>/...). Default 'maxvit_t' in the training pipelines; the activation-map tool defaults to 'maxvit', and only that exact string triggers its automatic target-layer pick; the Tk/Qt combo preselects 'resnet50'.",
    "model_type_ml": "(str) - Which classifier ml_analysis fits to separate positive- from negative-control wells and rank per-object features by permutation importance. One of xgboost (default), lightgbm, catboost, random_forest, extra_trees, gradient_boosting, logistic_regression, svm, mlp; lightgbm and catboost need their optional packages. reg_alpha, reg_lambda and learning_rate only affect the boosted models; logistic_regression is a good linear sanity check.",
    "nc": "(str) - Negative control identifier.",
    "nc_loc": "(str) - Location of the negative control in the images.",
    "negative_control": "(str) - Identifier of the negative-control class. In ML screening it is the value in location_column (e.g. 'c1') whose objects are labelled class 0 for training; in gRNA regression it is a gene/gRNA ID substring (e.g. '233460') matched against coefficient names to tag them 'nc' in the results and volcano plot. Defaults 'c1' and '233460' respectively.",
    "n_estimators": "(int) - Number of trees or boosting rounds in the tabular ML classifier - n_estimators for RandomForest/ExtraTrees/XGBoost/LightGBM, iterations for CatBoost, max_iter for HistGradientBoosting. More rounds keep improving fit up to a plateau while training time grows linearly; boosted models can overfit past it. Default 1000.",
    "n_epochs": "(int) - Number of training passes train_seg makes over the annotated image/mask batch. It also sets the checkpoint interval (a model is saved every n_epochs/10) and is written into the saved model filename. Raise it for a better fit on large annotation sets; lower it when a small set starts overfitting. Default 10000.",
    "n_neighbors": "(int or float) - Size of the local neighbourhood UMAP balances against global structure, and the perplexity when reduction_method is 'tsne'. Small values (5-50) sharpen fine local structure; large values give a smoother, more global embedding. A float is read as a fraction of the number of objects, and anything below 2 is clamped to 2. Default 1000.",
    "n_repeats": "(int) - Number of times each feature is randomly shuffled when computing permutation importance for the ML classifier. More repeats shrink the error bars on the importance ranking but cost an extra full prediction pass per feature per repeat. Default 10; drop to 3-5 for a quick look at wide feature tables.",
    "pathogen_Signal_to_noise": "(int) - Expected foreground-to-background ratio of the pathogen channel. Multiplied by pathogen_background it gives the intensity the normalisation ceiling must clear: spaCR walks percentiles 98 to 99.5 and takes the first that reaches it, falling back to 99.5. Raise it for a higher, dimmer, less clipped ceiling; lower it for more contrast. Default 10.",
    "nucleus_Signal_to_noise": "(float) - Multiplied by nucleus_background to set the intensity a bright pixel must reach before normalization stops raising the upper clip point; spaCR walks the 98th to 99.5th percentiles of the non-zero nucleus channel and takes the first that meets it, falling back to the 99.5th. A higher value forces a HIGHER upper clip, so contrast is stretched less and bright nuclei are protected from saturating; a lower value picks a lower clip point, stretching dim nuclei harder but blowing out bright ones sooner. Default 10.",
    "pathogen_size_range": "(list) - Two-element [min, max] area filter in pixels squared applied to the pathogen table in analyze_recruitment, well after segmentation: rows with pathogen_area outside the open interval are dropped. Bounds must be ints - floats are silently ignored. None widens it to effectively unlimited. Default [0, 100000]. Use it to discard debris and merged clumps.",
    "pathogen_types": "(list) - Names given to each pathogen condition on the plate, e.g. ['wt','ku80']. Element i is written into the pathogen column for every well listed in pathogen_plate_metadata[i] and folded into the combined condition label used for grouping and plotting. Must match pathogen_plate_metadata in length and order; None skips pathogen annotation.",
    "pc": "(str) - Positive control identifier.",
    "pc_loc": "(str) - Location of the positive control in the images.",
    "percentiles": "(list) - Two percentiles [low, high] used to rescale each channel of each image to 0-1 before segmentation, e.g. [2, 98]. Narrowing the window boosts contrast on dim objects but clips bright ones. Set None to derive them automatically: low fixed at 2, high the first of 98/99/99.9/99.99/99.999 exceeding background * Signal_to_noise. Default None in the Cellpose steps.",
    "pin_memory": "(bool) - Decode and hold the entire train/test image set in RAM up front (loaded in parallel across all cores) and hand batches to the GPU from page-locked memory. Enable when the dataset fits comfortably in RAM and disk I/O is the bottleneck; disable for large datasets or it will exhaust memory before the first epoch even starts. Default False.",
    "plate": "(str) - Inert where the GUI shows it (the regression settings): no code reads settings['plate'] on that path and no default is set for it. The behaviour people expect from it belongs to plateID - perform_regression passes settings['plateID'] (default 'plate1') to process_scores/process_reads, where it is stamped onto count and score rows that carry no plateID, becomes the first field of the plate_row_column key, and is ignored with a warning when the input already contains more than one distinct plateID. The only live reader of a 'plate' key is spacrops.stitch_cycle_wells, where it names stitched output files and falls back to plate_id, experiment, or the destination folder's basename. Set plateID instead.",
    "plate_dict": "(str) - Intended to map acquisition folder names to plate IDs, e.g. \"{'EO1': 'plate1'}\", but nothing reads settings['plate_dict']. Plate identity comes from the filename regex or the folder name via _extract_filename_metadata instead. Kept only so old settings CSVs still load.",
    "plot": "(bool) - Render and save QC figures while the pipeline runs: channel montages and Cellpose mask overlays during segmentation, before/after filtration views and crop grids during measurement. It adds figures per batch, so a full plate becomes much slower and more memory-hungry; keep it for small or test_mode runs, which force it on. Default False.",
    "plot_by_cluster": "(bool) - Chooses which thumbnails get overlaid on the embedding: when True, up to image_nr crops are sampled from each cluster (DBSCAN noise excluded) so every cluster is represented; when False, image_nr crops are sampled at random across the whole map. Keep True to compare cluster morphologies, False for an unbiased sample. Default True.",
    "plot_cluster_grids": "(bool) - Render a second figure with one color-bordered panel per cluster, each filled with up to image_nr example crops from that cluster, and save it as <METHOD>_grid.pdf when save_figure is on. Switch it off to skip the extra render when there are many clusters. Ignored unless plot_images is True. Default True.",
    "plot_control": "(bool) - Before the recruitment plots, draw a control panel of per-compartment mean intensities (cell, nucleus, pathogen, cytoplasm) for every channel, split by condition. Use it to confirm channel assignment and that positive/negative control wells separate as expected before trusting the recruitment numbers. Turn it off to shorten the run. Default True.",
    "plot_images": "(bool) - Paste the actual object crops onto the embedding scatter instead of showing bare points. Turn it off for a fast, plain scatter on large datasets - doing so also forces black_background to False and skips the cluster grid figure entirely. Default True.",
    "plot_nr": "(int) - How many merged image stacks from the start of the folder are drawn with cell, nucleus and pathogen outlines overlaid before recruitment analysis runs. The check is index <= plot_nr, so plot_nr + 1 images actually appear and 0 still plots one. Raise it to eyeball segmentation on more fields. Default 3.",
    "plot_outlines": "(bool) - Draw a boundary around each cluster in the embedding - a smoothed hull when smooth_lines is True, otherwise the raw convex hull edges. Helps show cluster extent and overlap but clutters dense maps; clusters with fewer than three points are skipped. Forced off when color_by is set. Default True.",
    "png_dims": "(list of int) - Which channel indices of the stack become the R, G and B planes of each saved PNG, in that order; at most 3, e.g. [0,1,2]. Channels not listed are absent from the crops (measurements are unaffected). With only 2 entries a blank third channel is added to keep the image RGB. Default [0,1,2].",
    "png_size": "(list of int) - Output crop size as [width, height] in pixels, centred on the object centroid; larger keeps more surroundings, smaller clips large objects. Should match the classifier input size (default [224,224]). With several crop_mode entries pass a list of lists, one size per mode, or a single size is reused for all.",
    "positive_control": "(str) - Identifier of the positive-control class. In ML screening it is the value in location_column (e.g. 'c2') whose objects are labelled class 1 for training; in gRNA regression it is a gene/gRNA ID substring (e.g. '239740') matched against coefficient names to tag them 'pc' in the results and volcano plot. Defaults 'c2' and '239740' respectively.",
    "preprocess": "(bool) - Run the image-preparation stage before segmentation: group raw files into per-field channel stacks, optionally subtract background, and percentile-normalize each channel into float arrays. Leave True on a fresh run; set False only when those normalized arrays already exist, otherwise segmentation has nothing to read. Default True.",
    "radial_dist": "(bool) - Measure how each channel's intensity varies with distance from the nucleus, pathogen and organelle boundaries inside each cell, binned into 6 shells and saved as <object>_rad_dist_channel_<c>_bin_0-5. Keep it on to quantify recruitment or intensity gradients toward an object; turn it off to shrink the feature table and speed up measurement. Default True.",
    "random_test": "(bool) - Seed the random draw of test-mode image sets with a fixed value (42), so every test run picks the same subset and results stay comparable. The selection is shuffled either way; set False when you want a different random subset each run to check that behaviour is not subset-specific. Default True.",
    "randomize": "(bool) - Shuffle the order of the per-field arrays before they are grouped into normalization batches, so each batch spans plates and wells instead of one acquisition block - this matters because normalization percentiles are computed per batch. Forced to False for timelapse runs to keep frames in sequence. Default True.",
    "regression_type": "(str) - Model fitted to the per-well response: 'ols' for a continuous score, 'wls' to weight each well by its cell count, 'rlm'/'huber' for a robust fit that outlier wells cannot drag, 'logit'/'probit'/'quasi_binomial' for fractions (GLM-binomial weighted by cell count, the last with free dispersion), 'beta' for a fraction strictly inside 0 and 1, 'poisson' for per-well counts, 'glm' with an auto-selected family, 'quantile' to fit a percentile instead of the mean, 'mixed' for random plate/row/column effects, 'lasso'/'ridge'/'elasticnet' for penalised fits tuned by alpha, 'hinge' for a linear SVM on a binarised response, and 'horseshoe' for the sparse Poisson power-analysis model. Set to None to let spaCR choose from the response distribution. Default 'ols'.",
    "remove_background": "(bool) - Hard-clip every pixel below the 'background' value to zero before normalization and segmentation. Use it when a channel carries a bright, even haze that inflates the normalization floor; leave it off for dim or already flat-fielded data, since the clip silently deletes faint real signal. Default False.",
    "remove_background_cell": "(bool) - Before normalisation, zero every pixel in the cell channel below cell_background. This flattens haze so the percentile stretch is driven by real signal, but it also erases genuinely dim cell edges and can shrink masks. Enable only once cell_background is set from an actual empty region. Default False.",
    "remove_background_nucleus": "(bool) - Before normalizing the nucleus channel, zero every pixel below nucleus_background and exclude those pixels from the percentile calculation. Enabling it raises contrast on real nuclei and suppresses haze, but clips genuinely dim nuclei to zero so they may become unsegmentable. Default False; check nucleus_background against raw images first.",
    "remove_background_pathogen": "(bool) - Before normalising the pathogen channel, hard-zero every pixel whose raw intensity is below pathogen_background. Enable it when diffuse autofluorescence inflates the low percentile and Cellpose starts segmenting haze; leave it off for dim parasites, since the clipping erases real signal and biases downstream intensity measurements. Default False.",
    "remove_cluster_noise": "(bool) - Drop the points DBSCAN labelled as noise (-1) from the embedding before it is plotted, so figures show only points that joined a cluster. Turn it off to see everything the reducer placed, including the diffuse background. Meaningless with kmeans, which never emits -1, and it is forced off automatically when color_by is set. Default True.",
    "remove_highly_correlated": "(bool or float) - Before dimensionality reduction, drop numeric features whose absolute Pearson correlation with an already-kept feature exceeds a cut-off. Pass a float to set the cut-off yourself, True to use 0.95, or False to keep everything. Enable it so families of near-duplicate measurements (area, perimeter, convex_area) do not dominate the embedding. Default True.",
    "remove_highly_correlated_features": "(bool) - In the machine-learning feature table, drop any feature whose absolute Pearson correlation with an already-kept feature exceeds 0.95, applied after the channel_of_interest filter. Leave it on so redundant measurements do not split importance scores and slow fitting; turn it off only when you need every original column. Default True. Note the UMAP path uses remove_highly_correlated instead.",
    "remove_image_canvas": "(bool) - When object thumbnails are overlaid on the embedding plot, make zero-valued background pixels fully transparent so only the segmented object shows instead of a black square. Turn it on for a cleaner montage, especially with black_background. Only L, I and RGB crops are supported; other PIL modes raise an error. Default False.",
    "remove_low_variance_features": "(bool) - Drop numeric features whose variance across objects falls below 0.01 before model fitting -- near-constant columns that carry no discriminative signal but still cost time and dilute importance rankings. Turn it off only when your features live on a very small numeric scale, where genuine signal can fall under that fixed cut-off. Default True.",
    "l1_ratio": "(float) - How the elastic-net penalty is split between L1 and L2: 1.0 is a pure lasso (sparse, picks one gRNA out of a correlated group), 0.0 is a pure ridge (dense, shares the effect across the group), and values between keep some of both. Use 0.5 when correlated gRNAs of the same gene should be selected together rather than arbitrarily. Read only by regression_type 'elasticnet'. Default 0.5.",
    "quantile": "(float) - Which quantile of the response quantile regression fits, strictly inside 0 and 1: 0.5 is the median (robust to outlier wells), 0.9 asks which gRNAs move the top of the distribution rather than its centre. Aggregation is turned off automatically so the quantile is taken over cells, not over well means. Read only by regression_type 'quantile'; it replaced the old overload of alpha. Default 0.5.",
    "hinge_threshold": "(float) - Response value above which a well counts as positive for the hinge (linear SVM) fit. Leave it None when the response is already binary, in which case the two values it holds become the two classes. spaCR refuses a continuous response with no threshold rather than splitting it at the mean or median, because a cut chosen by the software decides the hypothesis being tested. Read only by regression_type 'hinge'. Default None.",
    "hinge_n_boot": "(int) - Number of bootstrap resamples behind the hinge p-values. A support vector machine has no likelihood and so no Wald test; spaCR refits it on this many resamples of the wells and compares each coefficient to its bootstrap standard deviation. Treat the result as a stability statistic, not a hypothesis test. Higher is steadier and linearly slower; below about 50 the standard deviations are too noisy to rank on. Default 200.",
    "huber_t": "(float) - Where Huber's loss switches from squared to linear, in units of the estimated residual scale, for the robust fits. Smaller values downweight more wells and resist heavier contamination; larger values approach ordinary least squares. The default 1.345 gives 95 percent of the efficiency of OLS when the residuals really are normal. Read only by regression_type 'rlm' and 'huber'. Default 1.345.",
    "lasso_n_boot": "(int) - Number of bootstrap resamples used to rank lasso and elastic-net hits by how often each gRNA survives the penalty. These models have no valid p-values, so selection frequency replaces the significance test entirely. Higher is steadier and linearly slower; the cost is one full penalised fit per resample, doubled when alpha is 'auto' because each resample cross-validates. Default 200.",
    "lasso_selection_threshold": "(float) - Minimum bootstrap selection frequency, between 0 and 1, for a lasso or elastic-net coefficient to be called a hit. 0.6 means the gRNA kept a non-zero coefficient in at least three fifths of the resamples. Raise it for a shorter, harder-to-argue-with list; lowering it below about 0.5 admits terms the penalty drops as often as it keeps. Default 0.6.",
    "random_row_column_effects": "(bool) - Fit plate, row and column as random effects instead of fixed ones: True overrides regression_type to 'mixed' and fits a MixedLM grouped by plateID with rowID and columnID variance components, dropping them from the fixed-effect formula. Use it when edge or row artefacts differ between plates; it is slower and may fail to converge. Default False.",
    "resample": "(bool) - Passed to Cellpose model.eval: run the mask-tracking dynamics at full image resolution instead of on the downsampled network grid. Enabling it gives smoother, better-fitting object outlines at the cost of time and memory, and helps most when objects differ a lot from the model's training diameter. Default False; the object pipeline sets True for cell/nucleus and False for pathogen.",
    "rescale": "(float) - Rescaling factor for the images.",
    "resnet_features": "(bool) - Placeholder for embedding raw crops with ResNet features instead of the measured feature table. The branch in generate_image_umap is an empty pass, so enabling it skips the embedding step entirely and the run then fails on an unbound 'embedding' variable. Leave it False. Default False.",
    "row_limit": "(int) - Randomly subsample the joined measurement table down to this many objects (fixed seed 42) before dimensionality reduction, keeping UMAP and clustering tractable. Raise it for a more faithful map at higher memory and runtime cost, or set to None to use every row. Must not exceed the available row count. Default 1000.",
    "save_arrays": "(bool) - Also save each object as a raw .npy array - all channels, cropped to its bounding box, unnormalised - under a region_array/ folder. Enable when you need full bit depth or channels beyond png_dims for custom analysis; it uses far more disk than PNGs. Requires save_png to be True as well. Default False.",
    "save_figure": "(bool) - Write the embedding, plus the cluster grid when plot_cluster_grids is on, as vector PDFs to <src>/results/<METHOD>_embedding.pdf and <METHOD>_grid.pdf. Enable it when you want the figure for a paper or a record of the run; either way the plots are still displayed on screen. Default False.",
    "save_measurements": "(bool) - Master switch for the measurement half of measure_crop: compute morphology and intensity features for every cell, nucleus, pathogen, organelle and cytoplasm object and write them to the plate's SQLite database. Set it False when you only want cropped PNGs or filtered masks -- segmentation and cropping still run, but no measurement tables are written. Default True.",
    "save_png": "(bool) - Write one PNG crop per segmented object into <crop_mode>_png/ and register each path in the png_list table of measurements.db. Required for training or applying a classifier, for the Annotate app and for the UMAP image plots. Turn off to only compute measurements and save time and disk. Default True.",
    "Signal_to_noise": "(int) - Multiplier on background that sets the signal threshold (background * Signal_to_noise) used by the Cellpose tools when normalizing images: for each channel of each image spaCR takes the first of the 98th, 99th, 99.9th, 99.99th and 99.999th percentiles whose value exceeds that threshold, averages the picks over the batch, and rescales the channel between its average 2nd percentile and that upper bound. Raise it to force a higher percentile - less clipping, dimmer output; lower it to stretch faint objects harder at the cost of saturating bright ones. If no percentile clears the threshold for a channel, its upper bound falls back to that channel's average 2nd percentile, collapsing the range - a sign the value is far too high. Ignored when percentiles is set. Default 10 (5 in check_cellpose_models).",
    "skip_mode": "(str) - Intended to choose what happens to an image that cannot be processed, but nothing reads settings['skip_mode'] and no defaults function sets it, so it has never selected anything. spaCR's actual policy for a failing field is the fail-loud pair strict_errors and max_failure_rate: raise on the first failure, or tolerate up to a fraction of them and report. Kept declared only so old settings CSVs still load, and rejected by the pre-flight check and by spacr-run --set.",
    "smooth_lines": "(bool) - Draw cluster outlines as a smoothed spline through the convex hull (2 pt wide) rather than the raw straight hull segments (4 pt). Purely cosmetic - it does not change clustering; switch it off if smoothing distorts the true cluster boundary. No effect unless plot_outlines is on, and forced off when color_by is set. Default True.",
    "src": "(str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a \"['a','b']\" string, processes several plates in one run.",
    "target": "(str) - Free-text label for the protein or marker imaged in channel_of_interest, e.g. 'GRA1'. The recruitment run prints it in its banner ('channel:3 = protein') to record what the recruitment ratio is measuring; it feeds no computation, so changing it alters nothing but that log line. Default 'protein'.",
    "target_height": "(int) - Height in pixels that images are resized to before segmentation; masks are scaled back to the original dimensions afterwards. Only applied when both target_height and target_width are set (and, on the non-normalized path, when resize is True). Use it to match the field size the model was trained at. Default None, which disables resizing; 1120 for plaque analysis.",
    "target_intensity_min": "(float) - Recruitment-analysis cutoff on the 95th-percentile intensity of channel_of_interest inside each cell: cells at or below it are discarded before recruitment ratios are computed. Raise it to keep only strongly expressing cells; set 0 or None to disable the filter entirely. Raw intensity units, default 1.",
    "target_width": "(int) - Width in pixels that images are resized to before segmentation; masks are scaled back to the original dimensions afterwards. Only applied when both target_width and target_height are set (and, on the non-normalized path, when resize is True). Use it to match the field size the model was trained at. Default None, which disables resizing; 1120 for plaque analysis.",
    "tables": "(list) - Measurement tables read from each plate's SQLite database and merged into one analysis DataFrame. Only 'cell', 'nucleus', 'pathogen', 'cytoplasm' and 'png_list' are actually merged, and the image-UMAP step appends 'png_list' itself. Any other name (including 'organelle', which the measure step does write) is loaded and then silently dropped from the merge, and a name missing from the database aborts the read with 'Table not found in database'. List only the compartments your phenotype needs, since each extra table adds its whole feature block. Default ['cell', 'nucleus', 'pathogen', 'cytoplasm'] in most pipelines (['cell'] for percent-positive analysis).",
    "test": "(bool) - In classifier training, run the held-out evaluation pass (combine with train, or use alone to score an existing model). In the sequencing barcode mapper it means something different: process only the first read chunk and print a preview, so you can sanity-check the regex and barcode CSVs in seconds. Default False.",
    "test_images": "(int) - How many plate/well/field image sets are copied into a test/ folder when test_mode is on; every channel file belonging to a chosen set is copied together. Raise it for a broader smoke test, lower it for a faster one. Forced to 1 for timelapse runs so a full sequence stays intact. Default 10.",
    "test_mode": "(bool) - Run the pipeline on a small random subset instead of the whole folder. Mask generation copies test_images (default 10) complete image sets into <src>/test and works there; measure_crop copies test_nr (default 10) merged arrays into test/merged. Both also force verbose and plot on. Use it to check channel assignment, diameters and thresholds before committing to a full plate. Default False.",
    "dry_run": "(bool) - Validate the settings against the data they point at, print what the run would do, and stop before any compute. Mask generation and measure-and-crop check that src exists and holds the expected files, that every channel and mask-plane index is inside the number of planes actually present, that each value has the type the pipeline expects, and that the models, barcode CSVs or measurements.db the app needs are on disk; each problem is printed with a suggested fix, followed by a plan listing files found, objects to be segmented or measured, and where output would land. Nothing is written, no model is loaded and the GPU is never touched, so a settings mistake costs seconds instead of a whole run. The organize-and-stitch pipeline reuses the flag to list the file moves it would make. Default False.",
    "test_nr": "(int) - How many files are sampled at random from merged/ into test/merged when test_mode is on in the measure-and-crop pipeline, so measurement runs on a small subset. Raise it if a handful of fields is not representative; each extra file costs a full measurement pass. Default 10.",
    "treatment_loc": "(list of lists) - Plate wells that received each entry of treatments, one inner list per treatment in the same order, e.g. [['r1','r2'],['r3']]. Identifiers must start with 'r' (row) or 'c' (column); wells you do not list get no treatment label. Used by the vision-score annotation step. No default - supply it alongside treatments.",
    "treatments": "(list) - Names of the drug or treatment conditions in the experiment, e.g. ['dmso','lovastatin']. Each name is written into the treatment column and folded into the combined condition label used for grouping and plotting; positionally paired with treatment_plate_metadata (or treatment_loc), which lists the wells for each. Default ['cm','lovastatin'].",
    "top_features": "(int) - Feature cap in the ML screen analysis: how many rows the feature-importance and permutation-importance bar plots show, and how many top-ranked features the SHAP refit and its summary plot use. It is also the k of the SelectKBest pruning applied before the model is fitted, but only when prune_features is True - with prune_features at its default False the classifier trains on every feature and this is reporting/SHAP scope only. Raise for a fuller picture, lower for readable plots. Default 30.",
    "train": "(bool) - Whether to train the model.",
    "transform": "(str) - Optional transform applied to the aggregated per-well response before fitting: 'log' (log1p), 'sqrt', 'square', or None for none. Reach for it when the response is skewed and the normality check fails; the fit then reports coefficients for the transformed column, named '<transform>_<dependent_variable>'. Default None.",
    "upscale": "(bool) - Legacy image-upscaling toggle: no code in spaCR reads this key (or upscale_factor), so enabling it changes nothing about image size, segmentation or measurements. Kept only for settings-file compatibility. To change the working resolution use the Cellpose resize / target_height / target_width settings instead. Default False.",
    "upscale_factor": "(float) - Scale factor that the inactive 'upscale' toggle would have applied. Nothing in spaCR reads this key, so changing it has no effect on image size, segmentation or measurements; use the Cellpose resize / target_height / target_width settings to change resolution. Default 2.0.",
    "upstream": "(str) - Inert: nothing reads settings['upstream'], and sequencing.py never mentions it. The default is the forward primer sequence this was meant to anchor on; the barcode reader actually locates the window with target_sequence plus offset_start. Kept only so old settings CSVs still load.",
    "val_split": "(float) - Fraction of src/train randomly held out as a validation set each run (0.1 = 10 percent). The validation score drives checkpoint selection, early stopping and the live training curves; at 0 there is no validation loader, so checkpointing falls back to training accuracy, which rewards memorisation. Raise it on small datasets for a less noisy estimate. Default 0.1.",
    "visualize": "(bool) - Whether to visualize the embeddings.",
    "verbose": "(bool) - Print extra run detail instead of the minimal log: the resolved settings table at the start of mask generation, the channel and Cellpose-model choices per object type, per-table row counts and how many objects survive the nuclei/pathogen-per-cell filters when measurement tables are merged, and extra loader/diagnostic output in the training and UMAP paths. It only adds console output, so turn it on when object counts come out unexpected and you need to see which stage removed them. Defaults are per-pipeline: True for mask generation, UMAP, screen analysis, barcode mapping, Cellpose training and plaque analysis; False for measure-and-crop, plot-from-db and plot-from-CSV, the endodyogeny and class-proportion helpers, the Cellpose check/finetune tools, and the screen regression, whose verbose branch display()s the whole per-object score table.",
    "weight_decay": "(float) - L2 penalty applied to the weights on every optimizer step (AdamW applies it decoupled from the gradient). Raise it, toward 1e-3 to 1e-2, when validation loss climbs while training loss keeps falling; lower it toward 0 when the model cannot fit the training set at all. Every supported optimizer honours it. Default 0.00001.",
    "width_height": "(tuple) - Width and height of the input images.",
    "barcode_coordinates": "(list) - Intended to give the start/stop offsets of each barcode inside the read, but nothing reads settings['barcode_coordinates']. The pipeline slices the read from target_sequence, offset_start and the per-barcode lengths instead. Kept only so old settings CSVs still load.",
    "barcode_mapping": "(dict) - Inert: declared here but read by nothing in spaCR. The barcode-mapping pipeline takes its references from the separate grna_csv, row_csv and column_csv path settings, which sequencing.generate_barecode_mapping actually loads. Kept only so old settings CSVs still load.",
    "compression": "(str) - Legacy and currently inert: nothing in spaCR reads this key. The two preprocessing defaults set it to 'lzw' and the GUI offers lzw/zlib/none, but the live segmentation path writes masks as uint16 .npy via np.save, so no mask TIFF is ever produced and changing this setting changes nothing. The only functions that take a compression argument, io.save_object_mask and mask_io.save_mask (which hardcodes 'lzw'), are never called from anywhere in the package, and the sequencing HDF5 writer uses comp_type/comp_level instead. Leave it at its 'lzw' default.",
    "complevel": "(int) - Inert as a setting: nothing reads settings['complevel']. HDF5 compression is fixed by the comp_level parameter of sequencing.save_df_to_hdf5 / saver_process, which defaults to 5. Were it wired up, 0 would mean no compression and 9 the smallest, slowest files.",
    "file_type": "(str) - Substring that selects which object crops go into the dataset - only png_list rows whose PNG path contains it are used, e.g. 'cell_png', 'nucleus_png', 'pathogen_png', 'cytoplasm_png' or 'organelle_png'. In the GUI this one field writes both file_type and png_type, and only png_type is read downstream, so the two always hold the same value. Default 'cell_png'.",
    "model_path": "(str) - Path to a trained spaCR classifier saved as a whole PyTorch object (loaded with torch.load(weights_only=False), not a state_dict). Used when applying a model to a dataset tar and when generating activation maps. deep_spacr overwrites it with the freshly trained model whenever train is True, so set it only to score with an existing model. Default ''.",
    "dataset": "(str) - Path to the .tar archive of single-object PNG crops produced by generate_dataset, which the activation-map step opens with TarImageDataset. The plate folder is inferred two levels above it and CAM outputs are written next to it under <tar_name>/<cam_type>/. Must be a full path, not just a file name. Default ''.",
    "score_threshold": "(float) - Probability cutoff (0-1) applied to the model's positive-class score when deriving the binary cv_predictions column: pred >= threshold becomes 1. The raw probability is always saved alongside it, so this only changes the hard call, not the score. Lower it to catch more positives at the cost of false positives; raise it for precision. Default 0.5.",
    "sample": "(int, list or None) - Randomly draw this many PNG crops from the database when building the dataset tar instead of using all of them; a list uses its first element, and values above the total are clamped. Use it to build a quick trial dataset or to cap a huge screen. None uses every crop, shuffled. Default None.",
    "file_metadata": "(str, list or None) - Substring filter applied to png_path when pulling crops from the database: only paths containing it are included, and a list matches any one of its entries (OR, not AND). Use it to restrict a dataset to one plate, well or object type, e.g. 'plate1_' or 'cell_png'. None takes every crop. Default None.",
    "apply_model_to_dataset": "(bool) - After training (or straight away when reusing a saved model_path), pack the object PNGs into a tar, run inference over it, copy the n_top_examples most confident images per class into top_examples/, and merge the per-object scores back into measurements.db. Turn it off to only train and evaluate a model without scoring the screen. Default True.",
    "generate_full_dataset": "(bool) - Build the full unlabelled inference dataset tar from every selected plate independently of training or model application. Apply model to dataset also creates it automatically when needed. API: spacr.io.generate_dataset. Default False.",
    "tar_path": "(str) - Existing full-dataset tar to reuse for inference. Leave blank to generate one beneath the first plate's datasets folder. Multiple selected plates are combined into one tar. API: spacr.deep_spacr.apply_model_to_tar.",
    "n_top_examples": "(int) - Number of highest-confidence images saved per predicted class after full-dataset inference. This gives a quick visual check of class meaning and common errors. Default 20.",
    "random_seed": "(int) - Reproducibility seed shared by labelled train/test splitting, train/validation splitting, and grouped cross-validation folds. Keep it fixed to reproduce a run; change it to test sensitivity to one lucky split. Default 42.",
    "balance_to_smallest": "(bool) - Downsample every generated training class to the size of the smallest class before writing train/test folders. This removes the dataset prior but discards majority examples; disable it and use class_balance during training to retain all images. Default True.",
    "write_random_annotation_column": "(bool) - In annotation mode, persist an automatically selected unannotated comparison group into png_list as <column>_random. This makes an automatically generated control class reproducible and auditable. Default False.",
    "train_channels": "(list) - Which colour planes of each object crop the classifier sees, chosen from 'r', 'g' and 'b'. Fewer channels means a smaller input tensor and a model that cannot use the dropped stain, so drop a channel only when it carries no signal for your phenotype. The joined letters also become part of the saved model's filename. Default ['r', 'g', 'b'].",
    "dataset_mode": "(str) - How training classes are defined: 'metadata' splits crops by well metadata (class_metadata or metadata_rules), 'annotation' by the values in one or more annotation columns of png_list, 'measurement' by threshold rules on measured features (measurement_rules). Any other value aborts and returns no dataset. Default 'metadata'.",
    "annotated_classes": "(list) - Currently inert: the Tk 'Generate Dataset' form collects it and the defaults set [1,2], but no code reads settings['annotated_classes']. The two io.py helpers with a same-named parameter (training_dataset_from_annotation and training_dataset_from_annotation_metadata, default (1,2)) have no callers anywhere in the package. The live dataset builder selects classes from dataset_mode instead - annotation_columns/annotation_values under 'annotation', class_metadata or the rule lists under 'metadata'/'measurement'. Default [1,2], with no effect.",
    "um_per_pixel": "(float) - Physical size of one image pixel in micrometres, taken from your objective and camera. It is used only to convert scale_bar_length_um into pixels when a scale bar is drawn on representative-image grids, so a wrong value gives a wrong-length bar; it never rescales or resamples the images. The plotting helpers default to 0.1.",
    "pathogen_model": "(str or None) - Path to a custom Cellpose checkpoint used to detect pathogen objects, overriding pathogen_model_name when set. It must be a CPSAM-architecture checkpoint (one your own Train Cellpose run produced); a Cellpose-3 CPnet file cannot load into Cellpose 4. A path that does not exist stops the run rather than falling back to the stock weights silently. Default None.",
    "timelapse_displacement": "(int or None) - Maximum distance in pixels an object may travel between consecutive frames when linking: trackpy's search_range, or btrack's max search radius. Too small fragments tracks, too large causes identity swaps and SubnetOversize failures. None auto-searches downward from 500 for trackpy and falls back to 100 for btrack. Default None.",
    "timelapse_memory": "(int) - Number of consecutive frames an object may vanish (e.g. missed by segmentation) and still be re-linked to the same track by trackpy. Raise it when tracks fragment because objects blink out; too high risks merging two different objects into one track. Not used by the btrack mode. Default 3.",
    "timelapse_mode": "(str) - Which tracker links objects between frames. 'trackastra' is a transformer that tops the Cell Tracking Challenge leaderboard, needs no tuning and links divisions natively; 'ultrack' solves segmentation and linking as one integer program and wins on densely packed or 3D data at the cost of a longer solve; 'trackpy' needs a search radius and memory; 'btrack' needs a motion model; 'iou' just overlaps consecutive frames and drifts under fast motion. Default 'trackastra'.",
    "trackastra_model": "(str) - Which pretrained Trackastra checkpoint links the frames; 'general_2d' is the all-round 2D model and covers most live-cell data without retraining. Only consulted when timelapse_mode='trackastra'. Change it only if you hold a checkpoint trained on imaging that looks unlike yours. Default 'general_2d'.",
    "trackastra_linking": "(str) - How Trackastra turns predicted association scores into tracks: 'greedy' takes the best match per object and is fast, 'ilp' solves the assignment globally and is more accurate on crowded or dividing populations but needs the trackastra ilp extra and considerably more time. Default 'greedy'.",
    "ultrack_max_distance": "(float) - The largest jump in pixels Ultrack will consider when linking an object in one frame to a candidate in the next; anything further apart is never joined, so the track breaks instead. Raise it for fast-moving or sparsely sampled cells, lower it on crowded fields where a generous radius invites identity swaps. Only consulted when timelapse_mode='ultrack'. Default 25.0.",
    "ultrack_division_weight": "(float) - Cost the Ultrack solver pays to split one track into two daughters; the value is negative and the more negative it is the more readily divisions are accepted. Make it less negative when a replication assay over-calls divisions on touching cells, more negative when real division events are being missed. Only consulted when timelapse_mode='ultrack'. Default -0.1.",
    "ultrack_contour_sigma": "(float) - Standard deviation of the Gaussian blur applied while turning the segmentation labels into the contour map Ultrack builds its candidate objects from. Zero keeps the boundaries exactly as Cellpose drew them; one to four softens them so the joint solver is free to redraw boundaries between objects that were merged or split. Only consulted when timelapse_mode='ultrack'. Default 0.0.",
    "ultrack_n_workers": "(int) - How many worker processes Ultrack runs during its candidate-segmentation and linking passes; they all write into the same temporary sqlite store, so extra workers cut wall-clock on long movies but add database contention and memory. Leave it at one for short batches or a busy machine. Only consulted when timelapse_mode='ultrack'. Default 1.",
    "timelapse_frame_limits": "(list) - Slice of frame indices [start, end] kept from each batch before tracking, e.g. [0,10] to work on the first ten frames while tuning settings. The list is ignored unless it has at least two elements, which is why the shipped default [5,] has no effect. Default [5,].",
    "timelapse_objects": "(list) - Which segmented objects are tracked across frames and relabelled with track IDs: any subset of ['cell', 'nucleus', 'pathogen']; any other value aborts the run with a message. Each extra entry costs a full additional tracking pass. Tracking nuclei is often more stable than cells when cells touch. Default ['cell'].",
    "timelapse_remove_transient": "(bool) - After linking, drop every track not present in all frames (trackpy filter_stubs over the full stack length), keeping only objects tracked from first frame to last. Enable for clean per-object time courses; expect to lose cells that divide, enter or leave the field, so object counts fall. Default False.",
    "timelapse": "(bool) - Treat each well/field as a time series instead of independent images: files are grouped into time stacks, randomization is switched off, per-channel movies are written, objects in timelapse_objects are tracked across frames, a timeID column is added to the measurement tables, and measure_crop stops writing single-object PNGs. Only enable when filenames carry a time index. Default False.",
    "pathogen_min_size": "(int) - (Depreceated) Minimum pathogen object area in pixels squared, applied during measurement: any label with fewer pixels than this is erased from the pathogen mask before features are extracted. 0, the default, disables it. Superseded by pathogen_min_area, which filters at segmentation time instead.",
    "pathogen_mask_dim": "(int) - Position along the last axis of each merged/*.npy array where the pathogen label mask sits, one plane after the nucleus mask. With the default four image channels (0-3) that is 6; shift it if you keep a different number of channels. None makes measure_crop skip pathogen measurements, so infection status cannot be scored. Default 6.",
    "use_bounding_box": "(bool) - Crop the object's rectangular bounding box padded by 10 px instead of its mask, so neighbouring cells and background inside the box are kept rather than zeroed out. Enable when the classifier should see local context; leave off to isolate a single object on a black background. Default False.",
    "plot_points": "(bool) - Show the scatter marker for each object in the embedding. When False the markers are still drawn but at alpha 0, so cluster colors and the legend survive while only the outlines and overlaid thumbnails stay visible - handy for image-only UMAP figures. Marker size comes from dot_size. Default True.",
    "pos": "(str) - Column ID marking positive-control wells in the image UMAP. Rows whose columnID equals it are labelled cond='pos', so exclude_conditions can drop them; and when embedding_by_controls is True the rows whose col_to_compare equals it help fit the reducer. Default 'c1' (note: not 'c2').",
    "neg": "(str) - Column ID marking negative-control wells in the image UMAP. Rows whose columnID equals it are labelled cond='neg', so exclude_conditions can drop them; and when embedding_by_controls is True the rows whose col_to_compare equals it join pos in fitting the reducer. Default 'c2' (note: not 'c1').",
    "minimum_cell_count": "(int) - Wells with fewer than this many measured cells are removed before the ML plate heatmap is built. They are not left blank: the pivot is filled with 0 afterwards, so excluded wells render at the bottom of the colour scale and look like a genuine zero, and the 'allq' 2-98 percent limits are taken over that zero-filled matrix. It affects this heatmap only - the classifier and the saved results table still use every well. Set 0 to switch the filter off. Default 25.",
    "highlight": "(str) - Intended to mark genes or gRNAs whose name contains this substring in the ranked phenotype plot, but nothing reads settings['highlight']; toxo.plot_gene_phenotypes takes its gene_list directly from its caller. Kept only so old settings CSVs still load.",
    "pathogen_plate_metadata": "(list of lists) - Well locations of each pathogen condition, one inner list per entry in pathogen_types. Every item must be a row or column ID string such as 'c1' or 'r3'; anything else is silently ignored and those wells stay unannotated. Ranges like 'c2-c11' are not expanded - list each row/column. Do not leave it None while pathogen_types is set: annotation is not skipped, every row is labelled with the first pathogen_types entry. Defaults: None in the plot-from-db settings, [['c1','c2','c3'],['c4','c5','c6']] for recruitment analysis.",
    "treatment_plate_metadata": "(list of lists) - Plate wells that received each entry of treatments, one inner list per treatment in the same order, e.g. [['r1','r2','r3'],['r4','r5','r6']]. Entries must start with 'r' (row) or 'c' (column); anything else is ignored and those wells get no treatment label. Wells you do not list are still kept by analyze_recruitment - 'condition' is the join of whatever cell/pathogen/treatment labels exist and only rows missing all three are dropped, and with the shipped cell_types default every row has at least one label. plot_data_from_db is the path that really filters: it drops rows with no treatment label whenever this key is set. Default [['r1','r2','r3'],['r4','r5','r6']] in the recruitment pipeline, None in the plot-from-db, endodyogeny and class-proportion pipelines.",
    "regex": "(str) - Regex applied with re.match to each extracted read window; it must define the named groups columnID, grna and rowID, whose captured sequences are looked up in the three barcode CSVs. Non-matching reads are silently dropped, so a wrong group name or barcode orientation yields zero counts. The default captures an 8 bp column, 20-21 bp gRNA and 8 bp row barcode.",
    "target_sequence": "(str) - Constant vector sequence used as the anchor: every read is scanned for an exact match and the barcode window is then sliced relative to that hit using offset_start and expected_end. Reads without an exact match are skipped entirely, so it must be error-free and given in the orientation of the read being scanned. Default 'TGCTGTTTCCAGCATAGCTCTTAAAC'.",
    "column_csv": "(path) - CSV mapping column barcodes to well names; it must have 'sequence' and 'name' columns. Reads are matched verbatim against it with no reverse-complementing, so the sequences must be in the same orientation as the reads - run barecodes_reverse_complement on the file if they are not. Unmatched reads get NA for columnID.",
    "row_csv": "(path) - CSV mapping row barcodes to well names; it must have 'sequence' and 'name' columns. Reads are matched verbatim with no reverse-complementing, so the sequences must be in the same orientation as the reads - use barecodes_reverse_complement to flip the file if needed. Unmatched reads get NA for rowID.",
    "grna_csv": "(path) - CSV mapping gRNA barcode sequences to gRNA names; it must have 'sequence' and 'name' columns. Reads are matched verbatim with no reverse-complementing, so orientation must match the reads (barecodes_reverse_complement flips a file). Rows whose gRNA does not match are written as NA and dropped from the counts.",
    "save_h5": "(bool) - Also write every annotated read (consensus sequence plus its parsed row/column/gRNA barcodes and IDs) to annotated_reads.h5. The per-well counts in unique_combinations.csv and qc.csv are written either way, so set it False unless you need read-level data; True produces a very large file and compression can dominate runtime. Default True.",
    "comp_type": "(str) - PyTables compression library used when writing annotated_reads.h5, passed to pandas HDFStore as complib: 'zlib', 'lzo', 'bzip2' or 'blosc'. 'blosc' is far faster at similar file size, 'bzip2' is smallest but slowest. Ignored entirely when save_h5 is False. Default 'zlib'.",
    "comp_level": "(int) - complevel passed to the HDF5 store, 0-9. 0 disables compression (fastest write, largest file); higher values shrink annotated_reads.h5 at increasing CPU cost, and at the top of the range saving can take longer than the barcode mapping itself. Ignored when save_h5 is False. Default 5.",
    "custom_model_path": "(str) - Path to a trained classifier artifact whose model weights initialize a new fine-tuning run. The optimizer and epoch start fresh. Leave empty to initialize from ImageNet or random weights according to init_weights. Default ''.",
    "resume_checkpoint": "(str) - Path to a spaCR training artifact to continue exactly: restores model, optimizer, scheduler, epoch, best score and random-generator state. Use custom_model_path instead when only the weights should be reused. Default ''.",
    "normalize": "(bool) - Percentile-normalize each image channel (2nd to 98th percentile, clipped to 0-1) before display or model input; in the activation-map tool this rescales the image the CAM/saliency heatmap is drawn over. Turn it on when raw channels are too dim to read under the overlay. Affects display and input scaling only, never stored pixels. Default True.",
    "overlay": "(bool) - In the batch-grid figures, draw the activation map in the 'jet' colormap at 50 percent alpha over the source image. Turn it off and the grid tiles are left empty apart from the predicted-class label, so keep it on whenever plot is enabled. It never affects the per-object activation PNGs saved to disk, which are always the bare map. Default True.",
    "normalize_input": "(bool) - Apply the same per-channel mean=0.5, std=0.5 normalisation used during training to each image before it enters the model when generating activation maps. Keep it matched to how the model was trained, otherwise inputs are off-distribution and both the predicted classes and the maps are meaningless. Distinct from 'normalize', which only percentile-stretches images for display. Default True.",
    "normalize_plots": "(bool) - Normalize images before plotting.",
    "use_sam_cell": "(bool) - Inert: nothing in spaCR reads this key, and it is now redundant regardless, since Cellpose 4 segments every object with the SAM model (cpsam) and no non-SAM alternative exists. Kept only so old settings CSVs still load. Default False.",
    "use_sam_nucleus": "(bool) - Inert: nothing reads this key, and it is redundant now that Cellpose 4 segments nuclei with cpsam and the pre-SAM 'nuclei' model no longer exists. Kept only so old settings CSVs still load. Default False.",
    "use_sam_pathogen": "(bool) - Inert: nothing reads this key, and it is redundant now that Cellpose 4 segments pathogens with cpsam and the pre-SAM 'cyto' model no longer exists. Kept only so old settings CSVs still load. Default False.",
    "distance_gaussian_sigma": "(int or None) - Sigma in pixels of the Gaussian blur applied to each channel before measuring intensity-weighted centroid distances from cells to nuclei and pathogens. Larger values smooth out speckle so the weighted centroid follows broad signal. None or 0 skips these distance features entirely. Needs a cell mask plus a nucleus or pathogen mask. Default 10.",
    "infection_xgb_n_estimators": "(int) - Number of boosting rounds (trees) trained, passed as num_boost_round. More rounds fit the intensity-extreme training set more tightly and push infection probabilities away from 0.5, which shrinks the ambiguous band, but cost runtime and can overfit small wells. Trade off against infection_xgb_learning_rate. Default 200.",
    "infection_xgb_max_depth": "(int) - Maximum depth of each boosted tree. Deeper trees capture interactions between morphology and pathogen-intensity features but overfit the quartile-derived training labels; shallower trees generalise better across wells. Typical range 2-8; raise it only when the classifier cannot separate infected from uninfected. Default 3.",
    "infection_xgb_learning_rate": "(float) - Shrinkage applied to each boosting round's contribution (XGBoost eta). Lower values need more rounds but give smoother, better-calibrated infection probabilities; higher values converge fast and can slam probabilities to 0 or 1, defeating the ambiguous band. Usual range 0.01-0.3, tuned together with infection_xgb_n_estimators. Default 0.1.",
    "infection_xgb_subsample": "(float) - Fraction of training rows drawn at random for each boosting round, between 0 and 1. Below 1 it injects stochasticity that limits overfitting to the small set of intensity-extreme cells used for training; 1.0 uses every training row every round. Lower it if the classifier appears to memorise individual wells. Default 0.8.",
    "infection_xgb_colsample_bytree": "(float) - Fraction of feature columns offered to each tree, between 0 and 1. Lowering it stops a couple of dominant pathogen-intensity features from being chosen by every tree, spreading gain across morphology features and reducing overfitting; 1.0 exposes all features to every tree. Default 0.8.",
    "infection_xgb_reg_lambda": "(float) - L2 penalty on leaf weights. Larger values shrink leaf outputs, giving a more conservative model whose probabilities sit closer to 0.5 and therefore more cells inside the ambiguous band; 0 removes the penalty entirely. Raise it when the model fits training cells perfectly yet disagrees wildly with mask-based labels. Default 1.0.",
    "infection_xgb_random_state": "(int) - Seed for the generator that balances the per-well training set, i.e. which intensity-extreme cells are sampled for each class. It is not handed to XGBoost itself. Change it and re-run to confirm the adjusted infection calls are stable under a different training draw. Default 42.",
    "infection_xgb_n_jobs": "(int) - Threads XGBoost uses for training and prediction (its nthread parameter). -1 uses every available core; set a small positive number to leave CPU free for other work or when several plates run at once. It changes runtime, not the training recipe. Default -1.",
    "infection_xgb_proba_threshold": "(float) - Predicted probability at or above which a cell is called infected, between 0 and 1. Lowering it makes infection calling more permissive (more cells become infected), raising it more stringent. It is also the centre of the confidence band whose half-width is infection_xgb_margin. Default 0.5.",
    "infection_xgb_margin": "(float) - Half-width of the confidence band around infection_xgb_proba_threshold, clamped to 0-0.49. In 'relabel' mode only cells outside the band get their label overridden, the rest keep the mask-based call; in 'remove' mode cells inside the band are spared deletion. Raise it to trust the model less. Default 0.15.",
    "infection_xgb_top_features": "(int) - How many features, ranked by XGBoost gain, are retained for the feature-importance panel of the QC figure. This is a display cut applied after training: it never changes the model or the infection calls. Lower it for a readable bar chart, raise it to inspect more features. Default 20.",
    "infection_xgb_proba_column": "(str) - Column name the track-level ambiguous filter and the QC probability plot look for. The classifier actually writes 'infection_prob', so with the default value that column is not found and track-level ambiguous dropping is skipped with a warning. Set it to 'infection_prob' to enable that step. Default 'infection_xgb_proba'.",
    "infection_xgb_proba": "(float) - Probability cutoff used for additional infection-based filtering or reporting.",
    "infection_xgb_drop_ambiguous": "(bool) - After prediction, discard cells whose probability lies between infection_xgb_ambiguous_low and infection_xgb_ambiguous_high instead of forcing a call on them. True gives cleaner infected vs uninfected motility comparisons at the cost of sample size; False keeps every cell. Only used by the xgboost strategy. Default True.",
    "infection_xgb_ambiguous_low": "(float) - Lower edge of the discarded probability band, between 0 and 1. Cells whose probability falls between this and infection_xgb_ambiguous_high are dropped when infection_xgb_drop_ambiguous is True. Raise it toward the threshold to keep more cells, lower it to discard more borderline ones. Swapped automatically if it exceeds the high bound. Default 0.25.",
    "infection_xgb_ambiguous_high": "(float) - Upper edge of the discarded probability band, between 0 and 1. Together with infection_xgb_ambiguous_low it defines the interval whose cells are dropped when infection_xgb_drop_ambiguous is True. Lower it toward the threshold to keep more cells, raise it to discard more. Swapped automatically if it falls below the low bound. Default 0.75.",
    "infection_xgb_min_cells_per_class": "(int) - Per well, how many intensity-extreme examples each class must reach before that well's training data are balanced by subsampling to the smaller class; wells that have both classes but fewer examples contribute all of theirs, unbalanced. Wells with only one class are skipped entirely. No well is ever excluded for being small, so raising it leaves more wells unbalanced and the training set more skewed - lower it towards 1 to force balancing in every usable well. Default 10.",
    "infection_pca_method": "(str) - Records which embedding was actually used ('pca', 'umap' or 't-sne'). The pipeline overwrites it from infection_intensity_strategy while QC runs, so a value typed here is ignored - change infection_intensity_strategy to pick the embedding. Read it back afterwards to see what ran.",
    "infection_pca_n_clusters": "(int) - Intended cluster count for the embedding-based infection call. Not currently honoured: the pca/umap/tsne QC always runs KMeans with exactly two clusters, one mapped to infected and one to uninfected, so changing this has no effect on results. Default 2.",
    "infection_pca_random_state": "(int) - Seed for KMeans and for the UMAP/t-SNE embeddings in the pca/umap/tsne strategies. Fixing it makes the embedding and the resulting infected/uninfected cluster assignment reproducible; change it to check that the split is not an artifact of one initialisation. Note the max-cells subsample uses its own fixed seed. Default 42.",
    "motility_ylim": "(tuple) - Spatial y-axis limits for the origin-centred track panels (infected and uninfected) of the motility figure, in plotted coordinate units - um when pixels_per_um is set, otherwise pixels - not velocity. The whole-field all-tracks axis next to them always autoscales from the data and ignores this setting. Set to None for autoscaling. Default (100, -100), a 200-unit window written high-to-low so the axis draws reversed.",
    "motility_xlim": "(tuple) - Spatial x-axis limits for the origin-centred track panels (infected and uninfected) of the motility figure, in plotted coordinate units - um when pixels_per_um is set, otherwise pixels - not time. The whole-field all-tracks axis next to them always autoscales from the data and ignores this setting. Set to None for autoscaling. Default (100, -100), a 200-unit window written high-to-low so the axis draws reversed.",
    "seconds_per_frame": "(int) - Interval between consecutive timelapse frames, in seconds. Used with pixels_per_um to convert mean per-frame displacement into um/min; if either is missing, velocities stay in px/frame. It is also printed in the motility plot legend box. A wrong value rescales every reported velocity linearly. Default 60.",
    "pixels_per_um": "(float) - Image scale in pixels per micrometre. Track coordinates are divided by it, so plots switch from px to um, and together with seconds_per_frame it converts velocity from px/frame to um/min. Take it from the objective and camera pixel size rather than tuning it - it rescales every reported velocity. Default 1.78.",
    "infection_intensity_n_bins": "(int) - Bin count for the pathogen-intensity histogram, clamped to 10-256. The histogram strategy walks bins from low to high and takes the first whose infected fraction reaches the target as its intensity threshold, so more bins give a finer threshold but noisier per-bin fractions. Also sets the QC panel histogram. Default 64.",
    "db_table_name": "(str) - Table inside <src>/measurements/measurements.db that holds the pre-QC per-frame measurements. It is rewritten with if_exists='replace' on every run, a companion table with the suffix '_well_motility' holds the well summary, and the same name is read back when reuse_existing_measurements is True. Default 'timelapse_object_measurements'.",
    "infection_intensity_qc_graphs": "(bool) - Save the infection-intensity histogram PNG and reserve the QC sub-axes (histogram, embedding, or XGBoost probability plus feature importance) inside the combined intensity/motility panel. Set False to skip that plotting work on large runs; the infection relabelling itself is unchanged either way. Default True.",
    "infection_intensity_qc_panel_path": "(str) - Path of the small QC image embedded in the mask panel. The pipeline sets this itself - the histogram strategy stores the PNG it just saved, the other strategies leave it empty - and it clears any value you enter before QC runs, so this is a reported output rather than a control.",
    "infection_intensity_mode": "(str) - What happens when the QC call disagrees with the mask-based label: 'relabel' overwrites the label and keeps the cell, 'remove' deletes the disagreeing cells outright. Use 'relabel' to preserve sample size, 'remove' when you only want cells whose mask and intensity evidence agree. Unknown values fall back to 'relabel'. Default 'relabel'.",
    "infection_intensity_strategy": "(str) - How infected vs uninfected is decided once infection_intensity_qc is True: 'xgboost' trains a classifier on intensity extremes, 'histogram' picks one intensity threshold, and 'pca'/'umap'/'tsne' cluster a 2D embedding. Unknown values fall back to histogram, as does xgboost when the package is missing or a class is too small. Default 'xgboost'.",
    "infection_intensity_qc": "(bool) - Master switch for infection re-calling. While False the mask-based label (cell contains at least one pathogen) is used unchanged and every other infection_* setting is inert; True runs the method chosen by infection_intensity_strategy. A pathogen_channel must also be set. No default is applied anywhere, so it behaves as False until you set it.",
    "straightness_threshold": "(float) - Straightness cut-off, where straightness = net displacement / total path length (0 = returns to start, 1 = perfectly straight). When straightness_filter is True, tracks at or above this value are dropped as drift or tracking artifacts, so lowering it discards more tracks. The count is always logged. Default 0.95.",
    "straightness_filter": "(bool) - Actually apply the straightness cut. False only reports how many tracks exceed straightness_threshold and changes nothing; True removes those near-perfectly-straight tracks from the velocity table, the per-well summary and the plots. Turn it on when stage drift or identity swaps produce implausibly straight trajectories. Default False.",
    "zscore_thresh": "(float) - Outlier sensitivity when smoothing scalar features within a track (area, bbox area, equivalent diameter, perimeter, solidity, mean/max/min intensity). A frame more than this many standard deviations from its own track mean, whose two neighbours are both within half that, is replaced by their average. Lower smooths more; nothing is deleted. Default 3.0.",
    "max_displacement": "(float) - Largest plausible centroid movement between consecutive frames, in pixels. A single frame that jumps out and straight back is interpolated from its neighbours; any other jump above this value discards the whole track. Raise it for fast objects or sparse timelapses, lower it to purge ID-swap artifacts. Default 50.0.",
    "tracked_object": "(str) - Which object's feature block ({object}_* columns) the XGBoost infection classifier trains on: 'cell', 'nucleus' or 'pathogen'; anything else falls back to 'cell'. It does not change what is tracked - track geometry and velocity always come from the cell centroids. Default 'cell'.",
    "motility_analysis": "(bool) - Run the automated motility assay after segmentation: it rebuilds per-object measurements from merged/*.npy, cleans tracks, computes per-track velocity and straightness, applies the infection QC, and writes motility_plots plus a well-level summary table. It only fires when timelapse is also True, and it is what reveals the Motility setting categories. Default False.",
    "reuse_existing_measurements": "(bool) - If measurements.db already holds the table named by db_table_name, load it instead of re-extracting regionprops from merged/*.npy. Saves most of the runtime when re-running only the infection QC or the plots, but it also skips track smoothing, so changes to max_displacement or zscore_thresh only take effect with this set to False. Default True.",
    "infection_pca_umap_search": "(bool) - Fit UMAP once per combination of infection_pca_umap_n_neighbors_grid and infection_pca_umap_min_dist_grid, keeping the run with the highest cluster-centroid distance times ground-truth separation. True costs one UMAP fit per grid point; False does a single fit using infection_pca_umap_n_neighbors and infection_pca_umap_min_dist. Default True.",
    "infection_pca_umap_n_neighbors_grid": "(list[int]) - Candidate UMAP n_neighbors values tried when infection_pca_umap_search is True. Small values (around 5) preserve local structure and split fine subpopulations; large values (30 and up) emphasise global structure. Every entry is paired with every value in infection_pca_umap_min_dist_grid, so keep the list short. Default [5, 10, 15, 30].",
    "infection_pca_umap_min_dist_grid": "(list[float]) - Candidate UMAP min_dist values tried when infection_pca_umap_search is True, each between 0 and 1. Near 0 packs points tightly and gives crisper clusters for KMeans to split; larger values spread points out and blur the boundary. Paired with every n_neighbors candidate. Default [0.0, 0.05, 0.1, 0.3].",
    "infection_pca_pathogen_weight": "(float) - Multiplier applied to the standardised pathogen-channel features before embedding. Above 1 it stretches the embedding along pathogen intensity so KMeans splits infected from uninfected rather than by morphology; 1.0 leaves all features weighted equally. Raise it when the log reports weak cluster separation. Default 2.0.",
    "infection_pca_log_intensity": "(bool) - Apply log1p to features whose name contains 'intensity', 'p75', 'p95' or 'max' (only when all their values are non-negative) before standardising and embedding. Compresses the bright tail so a handful of very bright cells stop dominating the embedding. Worth enabling for wide-dynamic-range pathogen stains. Default False.",
    "infection_pca_tsne_search": "(bool) - Fit t-SNE once per combination of infection_pca_tsne_perplexity_grid and infection_pca_tsne_learning_rate_grid, keeping the run that scores highest on centroid distance times ground-truth separation. False does a single fit at infection_pca_tsne_perplexity with learning_rate 'auto'. Every extra grid point costs a full t-SNE fit. Default True.",
    "infection_pca_tsne_perplexity_grid": "(list[float]) - Candidate t-SNE perplexity values tried when infection_pca_tsne_search is True - roughly how many neighbours each point balances. Candidates at or above (n_cells-1)/3 are discarded, and if none survive the code falls back to min(30, that cap). Small values fragment clusters, large ones merge them. Default [15.0, 30.0, 45.0].",
    "infection_pca_tsne_learning_rate_grid": "(list[float]) - Candidate t-SNE learning rates tried when infection_pca_tsne_search is True. Too low leaves a dense ball with points crowded together; too high scatters the map into a diffuse cloud. Either way the infected/uninfected split blurs. Paired with every perplexity candidate, so keep both lists short. Default [200.0, 500.0].",
    "infection_pca_umap_n_neighbors": "(int) - Fixed UMAP n_neighbors used when infection_pca_umap_search is False - the size of the local neighbourhood UMAP tries to preserve. Low values (5-10) favour local detail and can shatter one population into several clumps; high values (30 and up) favour global layout. Ignored during grid search. Default 15.",
    "infection_pca_umap_min_dist": "(float) - Fixed UMAP min_dist used when infection_pca_umap_search is False, between 0 and 1: the minimum spacing allowed between embedded points. Near 0 gives tight, well-separated clumps that KMeans splits cleanly; larger values spread points evenly and blur the infected/uninfected boundary. Ignored during grid search. Default 0.1.",
    "infection_pca_tsne_perplexity": "(float) - Fixed t-SNE perplexity used when infection_pca_tsne_search is False, automatically capped at max(5, (n_cells-1)/3). Lower values emphasise local structure and can break one population into several clumps; higher values emphasise global structure and merge them. The learning rate is left at 'auto'. Default 30.0.",
    "infection_pca_min_silhouette": "(float) - Silhouette value below which the log prints a 'weak cluster structure' warning with tuning hints. It does not reject or re-run the clustering - the cluster-derived labels are applied regardless - so treat it purely as an alert level. Silhouette runs from -1 to 1. Default 0.05.",
    "infection_pca_min_gt_separation": "(float) - Alert level for the ground-truth separation score - the absolute difference, between the two clusters, in the fraction of intensity-extreme cells that are infected (0-1). Dropping below it only prints a warning; the cluster labels are still applied. Raise it to be told sooner that the embedding is not separating infection. Default 0.2.",
    "infection_pca_max_cells": "(int) - Ceiling on cells fed to the embedding; above it a random subsample of this size is drawn using a fixed seed of 0, independent of infection_pca_random_state. Lower it when UMAP or t-SNE is slow or memory-hungry, raise it to keep rare subpopulations. Applied after non-finite rows are dropped. Default 50000.",
    'organelle_channel': "(int) - Zero-indexed raw acquisition channel segmented into organelle masks by whichever organelle_method is chosen (otsu, adaptive, log, dog, ridge, hysteresis, cellpose, unet). Setting it to an integer adds an organelle mask plane to merged/ and unlocks the Organelle setting categories in the GUI; None skips organelle segmentation entirely. Default None.",
    'organelle_morphology': "(str) - Shape family of the target organelle; picks the segmentation pipeline and restricts which organelle_method values are legal. 'spots' = punctate (vesicles, lipid droplets), 'network' = filamentous (mitochondria, ER tubules), 'irregular' = solid blobby (Golgi, lysosomes), 'ring' = hollow (endosomes, autophagosomes). Default 'spots'. An unsupported morphology/method pair raises before any image is loaded.",
    'organelle_method': "(str) - Segmentation backend, validated against organelle_morphology: 'otsu' (one global threshold), 'adaptive' (local threshold), 'log'/'dog' (blob detection), 'ridge' (tubeness filter, network only), 'hysteresis' (dual threshold, network only), 'cellpose' (pretrained model), 'unet' (your own model, network only). Classical methods run on CPU across n_jobs workers; cellpose and unet run on the GPU. Default 'otsu'.",
    'organelle_diameter': "(float) - (DEPRECEATED) Expected organelle diameter in pixels. The Cellpose-SAM path used for organelles calls model.eval with diameter=None, and no classical method sizes its kernels from it, so changing this value has no effect on organelle masks. Bound object size with organelle_min_size / organelle_max_size instead. Default 30.",
    "organelle_model_name": "(str) - Cellpose model used when organelle_method='cellpose'. Cellpose 4 provides only 'cpsam'; the pre-SAM names are accepted and mapped to it. Change this only to point at a custom CPSAM-architecture checkpoint. Default 'cpsam'.",
    'organelle_min_size': "(int) - (Depreceated) Minimum object area in square pixels. Most classical segmenters and the U-Net discard smaller components during segmentation via remove_small_objects (the LoG/DoG spot methods do not, and the ring method applies a quarter of it, floor 3, to its edge image), and the value is always applied again to the finished label image. Despite the marker it is still live - raise it to clear dim specks and hot pixels, lower it to keep faint puncta. Default 10; 0 disables.",
    'organelle_max_size': "(int or None) - Upper area bound in square pixels applied to the finished label image; any object above it is deleted outright, not split. Use it to drop fused clumps, saturated debris and background regions that Otsu swallowed into one blob. Set it below your largest genuine organelle and you will silently lose real objects. Default None (no limit).",
    'organelle_remove_border': "(bool) - Delete every organelle label touching any of the four image edges, in the final post-processing step before counting and saving, so partly imaged objects do not bias area and intensity statistics. Costs you real objects around the FOV rim, which matters more the larger the organelle. Default False.",
    'organelle_log_min_sigma': "(float) - Smallest Gaussian scale searched by LoG blob detection, in pixels; the detected blob radius is about sigma times sqrt(2), so sigma 1 finds roughly 1.4 px radius puncta. Raise it to ignore single-pixel noise, lower it to catch the smallest spots. Default 1; must stay below organelle_log_max_sigma.",
    'organelle_log_max_sigma': "(float) - Largest Gaussian scale searched by LoG blob detection, in pixels; blob radius is about sigma times sqrt(2), so 10 caps detection near a 14 px radius. Raise it to catch large puncta, at a runtime cost since the filter is evaluated once per scale. Default 10; must exceed organelle_log_min_sigma.",
    'organelle_log_num_sigma': "(int) - How many Gaussian scales are evaluated between organelle_log_min_sigma and organelle_log_max_sigma. More scales resolve a wider spread of spot sizes, but the filter runs once per scale so runtime grows linearly. Default 10; drop to 3-5 when spot size is uniform and you need speed.",
    'organelle_log_threshold': "(float) - Minimum LoG/DoG response a local maximum must reach to count as a blob, measured after the image is percentile-normalised to 0-1, so it behaves like a contrast fraction. Lower it to pick up fainter puncta along with more noise; raise it to keep only bright ones. Default 0.01. The 'dog' method reads this key too.",
    'organelle_tophat_radius': "(int) - Radius in pixels of the disk used for white top-hat filtering before otsu/adaptive spot thresholding; it removes everything broader than the disk, flattening haze and background. Set it just above the largest genuine spot - too small erases the spots themselves, too large leaves background in. Default 5. Ignored by the log and dog methods.",
    'organelle_watershed_spots': "(bool) - Split touching spots instead of labelling each connected blob once. Under otsu/adaptive it runs a distance-transform watershed with seeds at least 5 px apart; under log/dog it grows a watershed from each blob centre instead of stamping a disk whose radius comes from that blob's own sigma (round(sigma*sqrt(2)), minimum 1 px). Turn it off when single spots are being fragmented. Default True.",
    'organelle_ridge_sigmas': "(list of float) - Scales in pixels at which the vesselness filter looks for tubular structures; each value should sit near the half-width of a filament and the responses are combined across scales. Add larger values to pick up thick bundles, keep them small for fine tubules. Default [1, 2, 3]; longer lists cost proportionally more time.",
    'organelle_ridge_filter': "(str) - Which vesselness filter enhances filaments before thresholding: 'frangi' (classic, crisp on well-separated tubules), 'sato' (more tolerant of varying thickness), 'meijering' (tuned for thin neurite-like fibres). All run with black_ridges=False, i.e. bright filaments on a dark background. Default 'frangi'; try 'sato' when frangi drops faint filaments.",
    'organelle_skeletonize': "(bool) - Reduce each thresholded network to a one-pixel-wide skeleton (dilated by 1 px so it stays connected) and label that instead of the filled filaments. Measured areas then track network length rather than filament thickness. Enable for topology and length analysis, disable to measure filament mass. Default False.",
    'organelle_network_threshold': "(str) - How the ridge-filter response is binarised: 'otsu' takes one global cut-off from the response histogram, 'adaptive' uses a local threshold (organelle_adaptive_block_size / _offset) and keeps faint filaments in dim regions at the cost of extra background. Only read by organelle_method='ridge'; anything unrecognised silently falls back to otsu. Default 'otsu'.",
    'organelle_adaptive_block_size': "(int) - Side length in pixels of the local neighbourhood used to compute the adaptive threshold; must be odd. Small blocks track fine illumination changes but can carve holes out of large organelles; large blocks behave more like a global threshold. A few times the object diameter is a sensible starting point. Default 51.",
    'organelle_adaptive_offset': "(float) - Subtracted from each local mean to form the adaptive threshold, so a pixel is foreground when it exceeds local_mean minus this value. Raising it therefore lowers the bar and segments MORE, not less; use small or negative values to be stricter. It is in raw image intensity units, so a value tuned for 16-bit data will flood ridge and ring modes, which threshold a 0-1 response. Default 5.",
    'organelle_morph_radius': "(int) - Radius in pixels of the disk used for morphological cleanup. In irregular mode it also sets the pre-smoothing sigma (radius/2) and drives a closing then an opening, bridging gaps and erasing protrusions thinner than the disk; network modes use half this radius for closing only. Raise it to smooth ragged outlines, lower it to preserve fine detail. Default 3.",
    'organelle_fill_holes': "(int) - Fill interior holes up to this area in square pixels after thresholding, so a darker centre does not turn one organelle into a donut. Only applied in irregular mode. Raise it when large organelles come out hollow; keep it low or 0 when the hollow centre is real biology. Default 64.",
    'organelle_CP_prob': "(float) - Cellpose cellprob_threshold. Pixels whose predicted probability of belonging to an object fall below it are excluded, so raising it shrinks masks and drops faint organelles, while lowering it grows masks and recovers dim ones along with more false positives. Useful range roughly -6 to 6. Default 0.0.",
    'organelle_FT': "(float) - Cellpose flow_threshold: the maximum error allowed between a candidate mask's flows and the network's prediction. Lowering it discards more oddly shaped masks (stricter, fewer objects); raising it keeps irregular ones. Default 0.4. Raise it when real organelles with non-round shapes are being thrown away.",
    'organelle_resample': "(bool) - (DEPRECEATED) Passed to Cellpose as resample: when True the flows are recomputed at full resolution instead of on the downsampled grid, giving smoother and slightly more accurate outlines for a little extra time. Still forwarded to model.eval on the organelle path. Default True; leave it alone unless you are chasing speed.",
    'organelle_mask_dim': "(int) - Position along the last axis of each merged/*.npy array where the organelle label mask sits. Masks follow the image channels in the order cell, nucleus, pathogen, organelle, so with four channels and all three other masks present it is 7. Leave it unset/None and organelles are not measured at all. No default is applied.",
    'organelle_chann_dim': "(int) - The channel dimension index for organelle masks in the saved array.",
    'organelle_rolling_ball': "(bool) - Roll a ball of organelle_rolling_ball_radius under the intensity surface, subtract the resulting background estimate and clip negatives to zero, before any segmentation runs. Flattens uneven illumination and haze so a single global threshold works across the whole FOV. Costs real time per image. Default False.",
    'organelle_rolling_ball_radius': "(int) - Radius in pixels of the rolling ball background estimator. It must be clearly larger than the biggest real organelle or the ball follows the objects and subtracts them away; too large and it stops tracking the illumination gradient. A few times the object diameter is a good start. Default 50; runtime grows steeply with radius.",
    'organelle_clahe': "(bool) - Rescale each image to 0-1 on its 0.5/99.5 percentiles, then run contrast-limited adaptive histogram equalisation before segmentation. Pulls dim organelles in dark corners up to the same working contrast as bright ones, at the cost of amplifying background noise and destroying absolute intensity comparability between fields. Default False.",
    'organelle_clahe_clip_limit': "(float) - Contrast ceiling for CLAHE, range 0-1: each tile's histogram is clipped at this height before equalisation, so higher values permit stronger local stretching and more noise amplification. 0.01 is gentle, 0.03-0.05 is aggressive. Only read when organelle_clahe is True. Default 0.01.",
    'organelle_mask_within_cells': "(bool) - Zero every pixel outside the cell mask before segmenting, so organelles can only be found inside cells and extracellular debris cannot generate objects. Needs cell_mask_stack/ to already exist alongside the organelle source; if it is missing spacr prints a warning and carries on unmasked rather than failing. Default False.",
    'organelle_dog_sigma_low': "(float) - Smallest Gaussian scale searched by Difference-of-Gaussians blob detection, in pixels; it sets the lower bound on detectable spot size (radius about sigma times sqrt(2)). Raise it to suppress fine noise, lower it to catch the smallest puncta. Default 1.0. The detection cutoff itself comes from organelle_log_threshold, not from a dog-specific key.",
    'organelle_dog_sigma_high': "(float) - Largest Gaussian scale searched by Difference-of-Gaussians blob detection, in pixels. Scales are stepped up from the low sigma by a factor of 1.6 until this bound, so widening the gap costs more passes but covers a wider range of spot sizes. Raise it to catch larger spots. Default 3.0; must exceed organelle_dog_sigma_low.",
    'organelle_hysteresis_low': "(float) - Weak threshold for hysteresis segmentation: pixels above it are kept only where they connect to a seed above organelle_hysteresis_high. Values below 1.0 are read as a fraction and converted to that percentile of the smoothed image (0.2 = 20th percentile); 1.0 or above is an absolute intensity. Lower it to trace filaments further into their dim tails. Default 0.2.",
    'organelle_hysteresis_high': "(float) - Strong threshold that seeds hysteresis segmentation - only components containing a pixel above it survive at all, then they grow outward down to organelle_hysteresis_low. Values below 1.0 are read as a percentile of the smoothed image (0.6 = 60th percentile); 1.0 or above is absolute. Raise it to keep only confidently bright filaments. Default 0.6.",
    'organelle_unet_model_path': "(str or None) - Path to a serialised PyTorch model used when organelle_method='unet'. It must be a torch.load-able whole module, not a state_dict, and take z-scored (B,1,H,W) input returning (B,1,H,W) logits; extra output channels are silently ignored except the first. A missing or invalid path raises before segmentation starts. Default None.",
    'organelle_unet_threshold': "(float) - Probability cut-off applied to the U-Net's sigmoid output, range 0-1. Lower it to grow the predicted network and recover faint branches at the cost of false positives; raise it to keep only confident pixels, which tends to break weak connections. Objects below organelle_min_size are still removed afterwards. Default 0.5.",
    'organelle_ring_sigma_inner': "(float) - Low sigma of the Difference-of-Gaussians band-pass that highlights ring walls, in pixels; set it near the wall thickness so the wall survives the high-pass. Too small and pixel noise is retained, too large and the wall blurs into the lumen and the ring stops being detected as hollow. Default 1.0; must be below organelle_ring_sigma_outer.",
    'organelle_ring_sigma_outer': "(float) - High sigma of the ring Difference-of-Gaussians band-pass, in pixels; it sets the coarse scale that gets subtracted, so keep it around the ring's outer radius. Widen the gap from organelle_ring_sigma_inner to enhance larger rings, narrow it for tight vesicles. Default 3.0; must exceed organelle_ring_sigma_inner.",
    'organelle_ring_min_prominence': "(float) - Shape gate for ring mode: for each filled object spacr computes abs(mean wall intensity minus mean lumen intensity) divided by the object's mean intensity, and deletes anything below this value. Raise it to keep only clearly hollow objects, lower it to also accept partly filled ones. 0 disables the gate. Default 0.1.",
    'organelle_ring_fill_method': "(str) - How detected ring walls become solid objects: 'flood' fills every background component that does not touch the image border - accurate, but leaks through any gap in the wall - while 'convex' takes the convex hull of each wall component, which tolerates broken rings but overshoots concave shapes. Default 'flood'; switch to 'convex' when rings come out unfilled.",
    'summarize_organelles_by': "(list or None) - Parent compartments to roll organelle measurements up into. Accepts 'cell', 'nucleus', 'pathogen', 'cytoplasm', each writing a <parent>_organelle_summary table of per-parent organelle count, total/mean/std area, area fraction of the parent, mean/std eccentricity and solidity, axis lengths and per-channel mean intensity; include 'organelle' to also save the raw per-organelle table. Default 'cell'; None writes nothing.",
    'cell_perimeter_fraction': "(float) - For each touching pair of cell labels, the shared boundary length divided by the smaller object's perimeter; pairs at or above this fraction are merged into one cell. Low values such as 0.1 merge aggressively and can fuse true neighbours, high values only rejoin pieces of the same cell. 0 disables perimeter merging. Default 0.",
    'nucleus_perimeter_fraction': "(float) - Merge two touching nucleus labels when their shared boundary covers at least this fraction of the smaller object's perimeter. Low non-zero values merge aggressively (0.1 joins barely-touching nuclei); high values only fuse objects sharing most of an edge. Range 0-1; 0 (default) disables perimeter merging. Use it when one nucleus is split into fragments.",
    'pathogen_perimeter_fraction': "(float) - Fraction, 0-1, of the SMALLER label's perimeter that two touching pathogen objects must share before they are fused into one. 0, the default, disables perimeter merging; values near 0.1 fuse almost anything that touches, while 0.5-0.8 fuse only objects with a long common border. Use it to repair vacuoles Cellpose cut in two.",
    'cell_intensity_merge': "(bool) - Merge touching cell labels when the mean intensity along their shared boundary is at least as high as the interior intensity of the dimmer of the two, meaning there is no real membrane edge between them. Use it to repair cells Cellpose cut in half. The comparison statistic is set by cell_intensity_threshold_method. Default False.",
    'nucleus_intensity_merge': "(bool) - Merge touching nucleus labels when the mean intensity along their shared boundary is at least as high as the dimmer object's own intensity statistic - i.e. there is no dark seam between them, so the split is spurious. Controlled by nucleus_intensity_threshold_method and nucleus_intensity_percentile. Default False; enable when Cellpose over-segments single nuclei.",
    'pathogen_intensity_merge': "(bool) - Merge two touching pathogen labels when the mean intensity along their shared border is at least as bright as the interior of the dimmer of the pair, i.e. there is no real intensity valley between them. Enable it to repair vacuoles Cellpose split down the middle. Needs an intensity image; tuned by pathogen_intensity_threshold_method. Default False.",
    'cell_intensity_split': "(bool) - Split oversized cell labels by distance-transform watershed before the merge and filter steps. Objects larger than cell_area_multiplier times the median cell area are seeded at local distance maxima cell_min_distance apart and cut. Despite the name no intensity is used. Enable when several touching cells share one label. Default False.",
    'nucleus_intensity_split': "(bool) - Enable watershed splitting of over-large nucleus labels: objects bigger than nucleus_area_multiplier times the field's median nucleus area are cut at distance-transform maxima spaced nucleus_min_distance apart. Despite the name it uses shape and area, not intensity. Default False; enable when clumps of touching nuclei are labelled as one object.",
    'pathogen_intensity_split': "(bool) - Enable watershed splitting of oversized pathogen labels. Despite the name the split is purely geometric: objects larger than max(pathogen_area_multiplier x median area, pathogen_min_object_area) are cut at local maxima of their distance transform. Turn it on when several parasites in one vacuole are fused into a single mask. Default False.",
    'cell_area_multiplier': "(float) - Splitting threshold for cell_intensity_split: only cells whose area exceeds this multiple of the median cell area in the image (or cell_min_object_area, whichever is larger) are watershed-split. Lower it toward 1.5 to cut borderline clumps, raise it to split only obvious doublets. Ignored unless cell_intensity_split is True. Default 2.0.",
    'nucleus_area_multiplier': "(float) - Splitting threshold expressed as a multiple of the median nucleus area in each field: only objects larger than this multiple (and larger than nucleus_min_object_area) are candidates for watershed splitting. Lower it toward 1.5 to split more aggressively, raise it to break up only obvious clumps. Default 2.0; used only when nucleus_intensity_split is True.",
    'pathogen_area_multiplier': "(float) - Splitting trigger expressed as multiples of the MEDIAN pathogen area in the field: only labels larger than this multiple (and larger than pathogen_min_object_area) go to the watershed. Lower it toward 1.5 to split more aggressively, raise it to break up only obvious clumps. Used only when pathogen_intensity_split is True. Default 2.0.",
    'cell_min_distance': "(int) - Minimum separation in pixels between watershed seeds when cell_intensity_split cuts an oversized cell; seeds are local maxima of the distance transform. Raise it for fewer, larger fragments and to stop one cell shattering, lower it to separate tightly packed cells. Ignored unless cell_intensity_split is True. Default 10.",
    'nucleus_min_distance': "(int) - Minimum separation in pixels between watershed seeds when splitting over-large nucleus labels; seeds are local maxima of the object's distance transform. Set it near the radius of one nucleus - too small shatters nuclei into fragments, too large yields a single seed so nothing splits. Default 10; used only when nucleus_intensity_split is True.",
    'pathogen_min_distance': "(int) - Minimum separation in pixels between watershed seeds when splitting oversized pathogen labels; seeds are local maxima of the distance transform. Raise it for fewer, larger fragments (or none, leaving the object intact); lower it to cut clumps into more pieces. Used only when pathogen_intensity_split is True. Default 10.",
    'cell_min_object_area': "(int) - Absolute pixel-area floor for splitting: the split threshold is the larger of cell_area_multiplier times the median cell area and this value, so cells at or below it are never cut. Raise it to protect small cells in fields where the median area is low. Ignored unless cell_intensity_split is True. Default 100.",
    'nucleus_min_object_area': "(int) - Absolute floor in pixels^2 on the watershed split threshold: objects at or below it are never split, even when nucleus_area_multiplier times the field's median area would fall lower. Raise it to protect small nuclei in fields where the median object is tiny. Default 100; used only when nucleus_intensity_split is True.",
    'pathogen_min_object_area': "(int) - Absolute floor in pixels squared below which a pathogen label is never split, whatever the median area says: the effective split threshold is max(pathogen_area_multiplier x median area, this value). Raise it to protect small parasites from fragmentation in sparse fields. Used only when pathogen_intensity_split is True. Default 100.",
    'cell_intensity_threshold_method': "(str) - Reference statistic that cell_intensity_merge compares the shared-boundary intensity against: 'mean' uses the mean interior intensity of the dimmer of the two cells, 'percentile' uses its cell_intensity_percentile instead. Any value other than 'mean' is treated as 'percentile'. Choose 'percentile' with a high percentile to make merging rarer. Default 'mean'.",
    'nucleus_intensity_threshold_method': "(str) - Which statistic of the dimmer of two touching nuclei the shared-boundary intensity is compared against when nucleus_intensity_merge is on. 'mean' (default) uses that object's mean intensity; 'percentile' uses its nucleus_intensity_percentile-th percentile, which at the default 75 is stricter and merges fewer pairs. Ignored when nucleus_intensity_merge is False.",
    'pathogen_intensity_threshold_method': "(str) - How the reference brightness is computed when pathogen_intensity_merge decides whether two touching labels have a real edge. 'mean' compares the shared-border intensity to the mean interior intensity of the dimmer object; 'percentile' compares it to pathogen_intensity_percentile of that object instead, which is stricter and merges fewer pairs. Default 'mean'.",
    'cell_intensity_percentile': "(int) - Percentile from 0 to 100 of the dimmer cell's interior intensity used as the merge reference when cell_intensity_threshold_method is 'percentile'. Raising it toward 95 sets a higher bar for the shared boundary to clear, so fewer pairs merge; lowering it merges more. Ignored when the method is 'mean'. Default 75.",
    'nucleus_intensity_percentile': "(int) - Percentile of each nucleus's own pixel intensities used as the merge reference when nucleus_intensity_threshold_method is 'percentile'. Higher values (90) demand a very bright shared boundary and merge almost nothing; lower values (50) merge readily. Range 0-100, default 75. Ignored when the method is 'mean'.",
    'pathogen_intensity_percentile': "(int) - Percentile, 0-100, of a pathogen's interior intensity used as the merge reference when pathogen_intensity_threshold_method is 'percentile'. Two touching labels merge only if their shared border is at least this bright inside the dimmer object, so raising it demands a brighter border and merges fewer pairs. Ignored when the method is 'mean'. Default 75.",
    "postprocess_cell_masks": "(bool) - Inert: no code in spaCR reads this key, so setting it has no effect. Cell-mask merge/split/declump is controlled instead by the cell_merge / cell_intensity_merge / cell_intensity_split / cell_area_multiplier group, applied in object.merge_split_filter_masks. Kept only so old settings CSVs still load.",
    "postprocess_nucleus_masks": "(bool) - Inert: nothing reads this key. Nucleus mask post-processing is driven by the nucleus_merge / nucleus_intensity_merge / nucleus_intensity_split / nucleus_area_multiplier group instead. Kept only so old settings CSVs still load.",
    "postprocess_pathogen_masks": "(bool) - Inert: nothing reads this key. Pathogen mask post-processing is driven by the pathogen_merge / pathogen_intensity_merge / pathogen_intensity_split / pathogen_area_multiplier group instead. Kept only so old settings CSVs still load.",
    'organelle_perimeter_fraction': "(float) - Merge two touching organelle labels when their shared boundary is at least this fraction of the smaller object's perimeter. Range 0-1; push it toward 1 to merge only near-fully-fused pairs, lower it to aggressively glue neighbours. Default 0 (disabled). Currently inert: the organelle mask writer never runs the merge/split stage.",
    'organelle_intensity_merge': "(bool) - Merge two touching organelle labels when the mean intensity along their shared boundary is at least the interior reference of the dimmer object - i.e. there is no real dark edge between them. Reach for it when thresholding has cut one organelle into pieces. Default False. Currently inert: the organelle mask writer never runs the merge/split stage.",
    'organelle_intensity_split': "(bool) - Split organelle labels whose area exceeds max(organelle_area_multiplier times the median object area, organelle_min_object_area), using a distance-transform watershed seeded by local maxima. Enable when neighbouring puncta are fused into single oversized labels. Default False. Currently inert: the organelle mask writer never runs the merge/split stage.",
    'organelle_area_multiplier': "(float) - Split trigger for organelle_intensity_split: only objects larger than this multiple of the median organelle area in the same field are candidates for watershed splitting. Drop it toward 1.5 to split more aggressively, raise it to cut only obvious clumps. Default 2.0. Currently inert: the organelle mask writer never runs the merge/split stage.",
    'organelle_min_distance': "(int) - Minimum separation in pixels between watershed seeds when splitting oversized organelles; distance-transform peaks closer than this collapse into a single seed. Raise it to stop one organelle being shredded into fragments, lower it to separate tightly packed puncta. Default 10. Currently inert: the organelle mask writer never runs the merge/split stage.",
    'organelle_min_object_area': "(int) - Absolute area floor in square pixels for the split step: an object is only split if its area also clears this, so small objects survive even when the median-based threshold is tiny. Raise it to protect genuine small organelles from being cut in half. Default 100. Currently inert: the organelle mask writer never runs the merge/split stage.",
    'organelle_intensity_threshold_method': "(str) - Reference statistic for organelle_intensity_merge: 'mean' compares the shared-boundary intensity to the mean interior intensity of the dimmer object; 'percentile' compares it to that object's organelle_intensity_percentile value instead. A high percentile makes merging much stricter. Default 'mean'. Currently inert: the organelle mask writer never runs the merge/split stage.",
    'organelle_intensity_percentile': "(int) - Percentile (0-100) of an object's interior intensity used as the merge reference when organelle_intensity_threshold_method='percentile'; ignored for 'mean'. Higher values raise the bar the shared boundary must clear, so fewer pairs merge. Default 75. Currently inert: the organelle mask writer never runs the merge/split stage.",
    "postprocess_organelle_masks": "(bool) - Inert: nothing reads this key. Organelle mask post-processing is driven by the organelle_merge / organelle_min_size / organelle_watershed_spots group instead. Kept only so old settings CSVs still load.",
    'cell_min_area': "(int) - Minimum cell area in pixels^2. Passed to Cellpose as min_size so undersized masks are dropped during segmentation, then re-applied afterwards to delete any object below it. Raise it to clear debris and fragments; set it too high and genuine small cells disappear. 0 disables. Default 0.",
    'nucleus_min_area': "(int) - Minimum nucleus area in pixels^2, applied twice: passed to Cellpose as min_size so small masks are never emitted, then re-applied to the label image so any surviving object below it is deleted and the rest renumbered. Raise it to drop debris and fragments. 0 (default) disables both filters.",
    'pathogen_min_area': "(int) - Minimum pathogen area in pixels squared. Passed to Cellpose as min_size so undersized masks never leave segmentation, then re-applied in the merge/split/filter pass. 0, the default, disables it. Raise it to clear speckle and debris; set it too high and small or newly divided parasites disappear.",
    'organelle_min_area': "(int) - Post-segmentation area floor in square pixels; smaller objects are deleted and the mask relabelled. Raise it to clear noise specks left by thresholding. Default 0 (disabled). Note this is the shared object filter (used by the Qt live preview); the batch organelle mask writer does its own size filtering with organelle_min_size.",
    'cell_max_area': "(int or None) - Maximum cell area in pixels^2; objects larger than this are deleted after segmentation. Use it to discard clumps or debris blobs that Cellpose labelled as one huge cell. It only deletes, it never splits - use cell_intensity_split for that. 0 or None disables the filter. Default 0.",
    'nucleus_max_area': "(int or None) - Maximum nucleus area in pixels^2; after segmentation any label larger than this is deleted and the remaining nuclei are renumbered. Use it to drop unsplit clumps of touching nuclei or blowouts that swallow much of a field. 0 (the default) or None disables it; there is no cap otherwise.",
    'pathogen_max_area': "(int or None) - Maximum pathogen area in pixels squared; labels larger than this are deleted after segmentation. 0, the default, or None disables the filter. Use it to drop fused clumps and segmentation blowouts that would otherwise dominate per-object statistics - but prefer pathogen_intensity_split if you want clumps separated rather than discarded.",
    'organelle_max_area': "(int or None) - Post-segmentation area ceiling in square pixels; larger objects are deleted. Use it to reject fused clumps and saturated debris. Default 0, and either 0 or None disables it. Note this is the shared object filter (used by the Qt live preview); the batch organelle mask writer caps size with organelle_max_size instead.",
    'cell_remove_border_objects': "(bool) - Delete every cell label touching any of the four image edges before measurement. Removes partial cells whose area and total intensity are truncated and would bias per-cell statistics, at the cost of losing objects - a large cost in fields where cells are big relative to the field. Default False.",
    'nucleus_remove_border_objects': "(bool) - After segmentation, delete every nucleus label touching any of the four image edges, then renumber the rest. Enable it when measuring nucleus area or total intensity, since clipped nuclei bias those downward; leave it off for counts or positions, as it discards real objects at every field boundary. Default False.",
    'pathogen_remove_border_objects': "(bool) - Delete any pathogen label touching the first or last row or column of the image. Enable it so partially imaged parasites do not enter area and intensity statistics with truncated values; leave it off when parasites are sparse and losing edge objects costs too much data. Default False.",
    'organelle_remove_border_objects': "(bool) - Delete organelle labels touching any image edge during the shared post-segmentation filter (the Qt live preview path). The batch organelle mask writer does the same job from organelle_remove_border, so set that one for a real run. Default False. Enable to keep clipped rim objects out of area and intensity statistics.",
    'cell_min_intensity_percentile': "(int) - Drops the dimmest cells per field: the mean intensities of all surviving cells are pooled and objects below this percentile (0-100) of that per-image distribution are removed. Being relative, it always removes roughly this share of objects, however bright the field. Use it to shed out-of-focus cells. 0 disables. Default 0.",
    'nucleus_min_intensity_percentile': "(int) - Drops the dimmest nuclei per field: spaCR takes the mean nucleus-channel intensity of every object surviving the area and border filters and removes those below this percentile of that per-field distribution. Because it is relative, any value above 0 always removes some objects. Range 0-100; 0 (default) disables it.",
    'pathogen_min_intensity_percentile': "(int) - Relative brightness cutoff, 0-100: within each field the mean intensity of every surviving pathogen is ranked, and objects below this percentile of that distribution are deleted. It is not an absolute intensity, so how many objects go depends on the object count. 0, the default, disables it. Raise it to drop dim false positives.",
    'organelle_min_intensity_percentile': "(int) - Drops organelles whose mean intensity falls below this percentile of all organelle mean intensities in the same field. It is relative, not absolute, so it always removes roughly this share of the dimmest objects even in a clean image. Range 0-100, 0 disables. Default 0. Use it to cull background-level detections.",
    'cell_max_intensity_percentile': "(int or None) - Drops the brightest cells per field: objects whose mean intensity is above this percentile (0-100) of the per-image distribution of cell mean intensities are removed. Lower it to about 99 to strip saturated blobs and fluorescent debris. Relative, not an absolute intensity. Use 100 to disable. Default 100.",
    'nucleus_max_intensity_percentile': "(int) - Drops the brightest nuclei per field: objects whose mean nucleus-channel intensity exceeds this percentile of the per-field distribution of object means are removed. Useful against saturated debris and staining artefacts. Range 0-100; 100 (the default) disables it, and any lower value always removes some objects.",
    'pathogen_max_intensity_percentile': "(int or None) - Upper end of the same per-field percentile filter: pathogens whose mean intensity exceeds this percentile of the field's pathogen mean intensities are deleted. 100, the default, disables it. Lower it to strip saturated debris and bright artefacts. Any value below 100 forces the intensity image to be loaded during filtering.",
    'organelle_max_intensity_percentile': "(int or None) - Drops organelle objects whose mean intensity exceeds this percentile of all organelle mean intensities in the same field, so it removes roughly the brightest (100 minus value) percent. Range 0-100; 100 or None disables it (None is read as the default 100, it does not error). Applied by the shared Qt live-preview filter - the batch organelle mask pipeline does not run this filter. Default 100. Use it to reject saturated dust and imaging artefacts.",
    # --- Descriptions filled in for settings that previously had no tooltip ---
    "annotation_column": "(str) - Name of the integer column in the png_list table that holds manual class calls; the Annotate app adds it with ALTER TABLE if it is missing and writes labels into it. It is the ground truth when dataset_mode is 'annotation' (used as the fallback when annotation_columns is unset), and in ML screen analysis it replaces location_column so controls are taken from annotations rather than plate position. Default 'test' in the dataset-generation and Annotate settings, but None in Analyze Screen - and that None is exactly what leaves the location_column override switched off, so the screen-analysis behaviour is opt-in.",
    'barecode_length_1': "(int) - Length in bases of the first barcode read from the sequencing data. Must match the barcode design used in the library.",
    'barecode_length_2': "(int) - Length in bases of the second barcode. Set to 0 if only a single barcode is used.",
    'cmap': "(str) - Matplotlib colormap applied to single-channel image previews and to plate heatmaps. Perceptually uniform maps ('viridis', 'inferno', 'magma') keep intensity differences honest; 'gray' matches how the raw microscope data looks. Any registered matplotlib name works, with an '_r' suffix to reverse it. Default 'inferno' for image plots, 'viridis' for plate heatmaps.",
    'controls': "(list) - gRNA identifiers treated as non-targeting controls in the regression. Their coefficients are tagged 'control', and their spread sets reg_threshold = mean + threshold_multiplier x (std or var, per threshold_method) - the effect-size cut-off drawn on the volcano plot. A noisier or wider control set raises that bar. None skips the threshold.",
    "correlate": "(bool) - Intended to add pairwise correlations between selected measurements to the analysis output, but nothing reads settings['correlate']. Channel/activation correlations are controlled by the separate 'correlation' setting in the activation-map path. Kept only so old settings CSVs still load.",
    'count_data': "(str or list) - CSV(s) of per-well gRNA read counts from the sequencing step (unique_combinations.csv); each must contain grna, count, rowID and columnID columns or the run raises ValueError. These are the regression's independent variable. Pass one path per plate, position-aligned with plates_count; results are written under the first file's folder.",
    'cov_type': "(str) - Heteroscedasticity-robust covariance estimator passed to the likelihood fits: 'HC0', 'HC1', 'HC2' or 'HC3', or None for classical non-robust errors. It changes standard errors and p-values only, never the coefficients; reach for 'HC3' when residual variance grows with well cell count. Applies to regression_type 'ols', 'wls', 'glm', 'poisson', 'quasi_binomial', 'logit' and 'probit'; the penalised, robust and quantile fits have no such estimator and refuse it rather than quietly reporting ordinary errors under a robust label. Default None.",
    "resume": "(bool) - Continue an interrupted run at its last verified safe boundary instead of starting over. Mask validates existing per-field mask and merged arrays; Measure accepts only fields complete in every owned measurements table and transactionally clears partial rows before retrying; Format Converter reopens every TIFF in a checkpointed field; UMAP reloads completed trial scores and embedding artifacts, including an incomplete adaptive round; Batch preserves settled plate/jobs and automatically enables field resume for an interrupted Mask, Measure, or conversion job. A checkpoint whose inputs or material settings differ is refused rather than mixing incompatible output. Default False.",
    "resume_search": "(bool) - Continue the compatible Image UMAP hyperparameter checkpoint stored under the project results folder. Completed trial scores and embedding arrays are loaded without refitting. If an adaptive 2x2 round was interrupted, only its missing corners are evaluated before the direction is chosen. The feature data, labels, search space, criterion, seed, increments and stopping threshold must match. Default False.",
    "checkpoint_path": "(str or None) - Optional explicit path for an atomic resume checkpoint. Format Converter defaults to .spacr_conversion.checkpoint.json in its destination; Image UMAP defaults to results/.spacr_checkpoints/umap_search.json under the project. Keep checkpoints with their outputs. Default None.",
    "umap_stability_repeats": "(int) - Number of independently seeded embeddings fitted for each configuration in multi-objective UMAP search. Stability is the mean fraction of k nearest neighbours shared between every pair of repeats, so it is unaffected by rotation or reflection. Runtime scales linearly; minimum 2, default 3. API: spacr.hyperparam.embedding_stability.",
    "umap_neighborhood_weight": "(float) - Relative multi-objective weight for neighborhood preservation, defined as the geometric mean of trustworthiness and continuity so both invented and lost neighbours are penalized. Weights are normalized to sum to one. Default 0.4. API: spacr.hyperparam.umap_objective_scores.",
    "umap_stability_weight": "(float) - Relative multi-objective weight for repeat-to-repeat nearest-neighbour stability. Weights are normalized to sum to one. Default 0.3. API: spacr.hyperparam.umap_objective_scores.",
    "umap_cluster_structure_weight": "(float) - Relative multi-objective weight for positive silhouette structure. spaCR uses supplied labels when available; otherwise it reports the best reproducible K-means silhouette across 2-8 clusters. This can reveal candidate structure but cannot prove biological meaning. Default 0.3. API: spacr.hyperparam.umap_objective_scores.",
    'background_correction': "(str) - Per-object local background subtracted from the outside-stain statistic before thresholding. 'auto' uses the median of the five-pixel ring outside the parasite mask, which removes a per-field offset without a flat-field image; 'none' subtracts nothing. Switching it on can backfire: a brightly stained attached parasite carries an antibody halo that reaches into that same ring, so subtracting it suppresses exactly the objects the threshold must keep above it. Default 'none'.",
    'bimodality_cutoff': '(float) - Value of the bimodality coefficient above which a distribution counts as two populations rather than one, below which the field and well are flagged and their efficiency should not be quoted. A perfect two-population mixture scores 1.0 at any mixing ratio and a single normal population about 0.33, so 5/9 sits between them. Raise it to demand cleaner separation before a number is reported unflagged. Default 0.5555555555555556.',
    'change_plate': "(bool) - Relabel each source directory as plate1, plate2, ... instead of trusting the plate ID stored in its database. Use it when several plates were written under the same name, which would otherwise let two plates' fields pool into one threshold and one well. Default False.",
    'compartment': "(str) - Prefix the per-object measurement columns carry, so 'pathogen' selects pathogen_area and pathogen_channel_1_percentile_95. It must match the object type the table actually holds, or the area filters and the intensity statistic resolve to columns that do not exist and the run aborts naming them. Default 'pathogen'.",
    'control_quantile': "(float) - Quantile of the control wells' outside-stain distribution taken as the threshold. 0.99 means one in a hundred genuinely unstained parasites is misread as attached. Lowering it toward 0.95 buys safety against this assay's dangerous error - an outside parasite scored invaded - at the cost of a few false attached calls; raising it does the reverse. Default 0.99.",
    'control_wells': "(list or None) - Wells whose parasites are known to carry no pre-permeabilisation stain (no primary antibody, or no permeabilisation). They give the honest negative distribution the cut should sit above, which is better evidence than any automatic method. Entries may name a column ('c12'), a row ('r1'), a well ('r1_c12') or a full plate-row-column key, and those wells are dropped from every efficiency because a staining control is not an experimental condition. None runs the automatic per-field method instead. Default None. The screen regression reads the same key for a different job: graph_sequencing_stats drops these wells from the count table before it sweeps for fraction_threshold, so there it must match filter_value and must be a list -- it is iterated, and None would raise. Default there is a copy of filter_value.",
    'extracellular_class': "(str) - How parasites overlapping no host cell are scored. 'attached' calls them attached whatever the stain says, since something outside every cell cannot have invaded one; 'classify' leaves the decision to the stain, which is what you want when the cell mask is the unreliable part; 'exclude' drops them before anything is counted. The count is reported as n_no_host_cell either way, so the choice stays visible. Default 'attached'.",
    'group_column': "(str) - Column whose values become the experimental conditions compared against each other; 'condition' is the combined host-cell / pathogen / treatment label built from the plate-metadata maps. Point it at 'pathogen' or 'treatment' to compare on one factor alone. Rows with no value here are dropped before anything is counted. Default 'condition'.",
    'inflation_warn': '(float) - Extra invasion efficiency, in proportion units, that raising the threshold by threshold_sensitivity may add to a well before the well is flagged. Only the upward move is watched, because lowering a threshold can only turn invaded back into attached and never invents a result. 0.05 flags a well whose efficiency would gain more than five percentage points. Default 0.05.',
    'intensity_statistic': "(str) - Which per-object statistic of the pre-permeabilisation channel is thresholded. That stain sits on the parasite's surface, so the object's mean divides a rim by the whole area and reads a large parasite as dimmer than a small one stained identically - a bias that turns outside parasites into invaded ones. 'auto' therefore takes periphery_95, then percentile_95, then mean, in that order and warns on the last; 'periphery_95', 'periphery_85', 'periphery_mean', 'percentile_95', 'percentile_85', 'max', 'mean', 'median' and 'integrated' pick one explicitly, and any literal column name is used verbatim. Default 'auto'.",
    'level': "(str) - Aggregation the condition figure's stacked bars are drawn at: 'object' pools every parasite into one bar per condition, 'well' averages the per-well proportions and draws SD whiskers, 'plate' does the same across plates. It changes the figure only - the reported statistics always treat the well as the unit of replication. Default 'object'.",
    'max_parasite_area': '(float or None) - Largest object area in pixels kept as a parasite. Anything bigger is several parasites merged by the mask, whose rim statistic mixes them and whose single classification then stands for all of them. None keeps everything. Default None.',
    'min_control_objects': "(int) - Objects a plate's control wells must contribute before their quantile is trusted as a threshold. Below it the plate falls back to the automatic per-field method and says so, rather than taking a 99th percentile from a handful of points. Default 10.",
    'min_objects_for_bimodality': '(int) - Objects required before the bimodality coefficient is computed at all; below it the coefficient is left NaN and the field or well is flagged. The statistic exceeds its cutoff on genuinely unimodal data about 45% of the time at ten objects and 15% at twenty, so computing it there would silence the check exactly where the classification is least trustworthy. Default 30, where that false-pass rate is 5%.',
    'min_objects_for_threshold': "(int) - Objects a field must hold before a threshold is derived from it alone; below it the field borrows its well's threshold, then its plate's, and the level actually used is written into automatic_source. Raising it makes thresholds steadier and less local, which is the wrong trade when illumination varies across the field of view. Default 10.",
    'min_parasite_area': '(int or float) - Smallest object area in pixels kept as a parasite. Smaller objects are debris, and their outside-stain statistic is noise over a handful of pixels that will land on whichever side of the threshold the noise happens to fall. Raise it when the pathogen mask is shattering. Default 0, which filters nothing.',
    'min_parasites_per_well': "(int) - Scored parasites below which a well's efficiency is flagged as too thin to quote. At fifty the 95% interval on a proportion near a half is still about fourteen percentage points wide, which is larger than most real effects. The number is never suppressed, only marked, and n_total travels beside it. Default 50.",
    'min_total_intensity': '(float or None) - Minimum mean intensity in the post-permeabilisation channel for an object to count as a parasite at all. That antibody stains every parasite, so an object dark in it is debris inside the pathogen mask rather than a dim parasite, and it would otherwise contribute a background-level outside signal and be scored invaded. None applies no filter. Default None.',
    'outside_channel': "(int) - Zero-indexed channel of the pre-permeabilisation antibody, which reaches only parasites still outside the host cell. Every classification the assay makes is a threshold on this channel, so pointing it at the wrong stain silently converts the readout into whatever that channel measures. Note this is a channel index, not measure.py's <object>_channel_<n>_outside_* columns, which are the ring just outside an object's own mask. Default 1.",
    'outside_threshold': '(float or None) - Fixed cut on the outside-stain statistic, applied to every field and overriding both the automatic method and the control wells. Set it only when you have calibrated it yourself: raising it above the true cut moves attached parasites into the invaded class and inflates invasion efficiency, and nothing moves them back. Control wells, if given, stay on as the reference the QC judges the fixed value against. None derives the cut per field. Default None.',
    'outside_threshold_method': "(str) - How the outside-stain cut is derived from each field's own objects when no fixed value and no control wells are given: 'otsu', 'triangle', 'li', 'yen' or 'mean'. All of them find a split; none of them can tell you a split exists, which is what the bimodality check is for. 'triangle' suits a heavily skewed distribution with a small stained minority, 'otsu' a more balanced one. Default 'otsu'.",
    'parasite_table': "(str) - Table in measurements/measurements.db holding one row per segmented parasite. It is read directly rather than through the usual merge, because that merge collapses pathogen rows onto their host cell and would sum several parasites' stain intensities into a single row. Change it only if measure_crop wrote the parasite objects under a non-standard name. Default 'pathogen'.",
    'vacuole_key': "(str) - Rule used to group individually segmented parasites into vacuoles. 'auto' prefers an explicit vacuole-ID column, otherwise spatially clusters centroids, then falls back to host cell or one parasite per vacuole with a warning. Set 'spatial', 'cell_id', 'object', or an explicit column name to make that biological assumption reproducible. Default 'auto'.",
    'vacuole_link_distance': "(float or None) - Maximum centroid-to-centroid distance in pixels for spatially linking parasites into the same vacuole. None derives the distance from median parasite diameter times vacuole_link_factor; set a calibrated value when magnification or segmentation scale varies between plates. Too large merges separate vacuoles and too small splits one rosette. Default None.",
    'vacuole_link_factor': "(float) - Multiplier applied to the median segmented-parasite diameter when vacuole_link_distance is derived automatically. Increasing it joins wider rosettes but also raises the risk of merging nearby vacuoles; decreasing it does the reverse. It is ignored when an explicit link distance or vacuole-ID column is used. Default 1.5.",
    'parasite_count_column': "(str or None) - Optional column that already stores the number of parasites represented by each segmented row. When set, the assay sums that column per vacuole instead of counting rows, which is required if one row can represent several parasites. None treats every retained row as one parasite. Default None.",
    'max_parasites_per_vacuole': "(int) - Largest power-of-two parasite count given its own replication bucket. Counts above it remain visible in a '>N' bucket and non-powers stay in the separate QC bucket; they are never clipped or rounded. Use a power of two large enough for the experiment's duration. Default 16.",
    'require_host_cell': "(bool) - Drop parasite rows with no valid host-cell link before constructing vacuoles. This prevents extracellular debris and attached parasites from entering a replication readout, but it will also remove real infected cells when cell segmentation or parent assignment failed. The number removed is reported. Default True.",
    'non_power_of_two_warn': "(float) - Fraction of a well's vacuoles allowed in the non-power-of-two bucket before the well is flagged as unreliable. Three-, five-, or seven-parasite rosettes usually indicate segmentation or vacuole-linking errors, so lowering the threshold makes QC stricter without deleting any observations. Default 0.2.",
    'qc_plot_max_panels': '(int) - Largest number of wells drawn in the threshold-diagnostic figure, taken in sorted well order. It exists so a 384-well plate does not produce a 384-panel figure; the CSVs always carry every well regardless. Default 12.',
    'seed_wells_from_cells': '(bool) - Read the cell table as well, so a well holding host cells but no parasites appears in the results with a zero denominator instead of vanishing from the plate entirely. Switch it off only when the database has no cell table. Default True.',
    'threshold_agreement_tolerance': "(float) - Relative distance a threshold may sit from its reference before the field and well are flagged; the reference is the control-derived cut when controls exist, otherwise the field's own automatic cut. 0.5 means a factor of two. Lower it to catch smaller drifts between a fixed threshold and what the data would have chosen. Default 0.5.",
    'threshold_sensitivity': "(float) - Fractional amount the threshold is moved up and down to produce the invasion_efficiency_low_threshold and _high_threshold bracket, which shows how much of a well's answer is the threshold rather than the biology. Widening it widens the bracket and makes the inflation flag more eager. Default 0.25.",
    'total_channel': '(int or None) - Zero-indexed channel of the post-permeabilisation antibody that stains every parasite. Nothing is classified from it; it only supplies the intensity that min_total_intensity filters on, so an incorrect value costs nothing until that filter is switched on. Default 0.',
    'attribution_baseline': "(str) - What a pixel is replaced with when the deletion/insertion curves remove it: 'blur', 'zero' or 'noise'. It is a confound, not a detail - blanking to zero creates a hard edge the model has never seen, so part of the score drop measures the artefact rather than the lost information. 'blur' is the least out-of-distribution and is the default; comparing two baselines is a fair way to ask how much of your AUC is real. Default 'blur'.",
    'attribution_steps': '(int) - Points along the deletion and insertion curves used to score a map. At each step the highest-ranked remaining pixels are removed (or added) and the model re-run, so this is the resolution of the area-under-curve that judges whether the map describes what the model actually uses. More steps give a smoother AUC at linearly more forward passes. Default 12.',
    'ig_baseline': "(str) - The 'absence of signal' image integrated gradients integrates away from: 'zero' is black, 'blur' is your own image blurred, 'noise' is random. This choice IS the explanation's reference point and changes the result - a black baseline attributes to everything bright, which on dark-field microscopy means it attributes to the object merely for existing. 'blur' keeps the low-frequency content and asks what the detail contributes. Default 'zero'.",
    'ig_steps': "(int) - Interpolation steps between the baseline and your image for integrated gradients. The method's guarantee - that the attributions sum to the score difference - only holds in the limit, so too few steps silently breaks it; 50 is the usual default and the completeness error is worth checking if you lower it. Cost is linear in this number. Default 50.",
    'object_type': "(str) - Which mask decides where an object is when the pointing game scores an attribution map: 'cell', 'nucleus', 'pathogen' or 'cytoplasm'. The pointing game asks only whether the map's single hottest pixel lands inside that mask, so it is cheap and coarse - it says nothing about the rest of the map, and a method can score 1.0 while attributing nonsense everywhere else. Default 'cell'.",
    'occlusion_stride': '(int) - How far the occlusion patch moves between evaluations. Equal to occlusion_window it tiles without overlap and is fastest; half of it doubles the passes and halves the blockiness. A stride larger than the window leaves unmeasured gaps that appear as an artificial grid in the map. Default 4.',
    'occlusion_window': "(int) - Side length in pixels of the patch occlusion slides over the image, blanking it and recording how far the model's score falls. Larger windows are faster and blurrier and will miss a feature smaller than the window; smaller ones resolve fine structure at quadratically more forward passes. Occlusion is the only method here that needs no gradients at all, which is why it is worth its cost as a cross-check on the gradient family. Default 8.",
    'sanity_check': "(bool) - Randomise the model's weights layer by layer and re-attribute, then report how similar the map stays. A method that produces nearly the same picture for a randomised model is an edge detector, not an explanation - and measured on a small CNN the whole CAM family, including the Grad-CAM spaCR defaults to, fails this while saliency and integrated gradients pass. The number is reported for YOUR model rather than assumed, which is the point. Costs one extra attribution per randomised layer. Default True.",
    'smoothgrad_samples': "(int) - Noisy copies of the image averaged into one attribution map. A single map is dominated by the gradient's local jitter, so 8-50 samples smooth it into something stable enough to compare between images; 0 (the default) runs the method once and is what you want while you are still choosing a method, since it costs one forward-backward pass instead of N. Applies to every method, including the CAM family, where it is averaged explicitly rather than through captum. Default 0.",
    'smoothgrad_sigma': "(float) - Standard deviation of the noise SmoothGrad adds, as a fraction of the image's intensity range. Too small and every sample is the same map, so averaging changes nothing; too large and the samples are of images the model has never seen, so the average describes the model's behaviour on noise rather than on your data. 0.1-0.2 is the usual band. Ignored when smoothgrad_samples is 0. Default 0.15.",
    'strict_errors': "(bool or None) - What happens when a step hits a problem it could technically survive. None (the default) defers to the SPACR_STRICT_ERRORS environment variable, which is how a cluster turns this on for a whole batch without editing every settings file; False and True are explicit per-run choices and override it. When off, the failure is recorded in the run ledger, printed in the end-of-run summary and stamped into the artifact's run_status, and the run continues on the items that worked. When on, a swallowed setup or configuration error - an unreadable path, a missing column, a database that will not open - raises immediately, so a batch job stops at the first sign its inputs are wrong instead of producing a plausible-looking partial result. Per-item failures such as one corrupt image are still survived either way. Default None.",
    'max_failure_rate': "(float or None) - Fraction of failed items above which the run aborts rather than finishing and reporting. 0.2 means 'stop once more than a fifth of the fields have failed', on the grounds that whatever is left is no longer the experiment. The ledger is stamped into the artifact before the abort, so the evidence survives. None (the default) never aborts on rate alone - every failure is still counted and reported, and the artifact is still marked partial. Default None.",
    'queue_by_uncertainty': "(bool) - Reorder the Annotate grid so the crops the classifier is least sure about come first, instead of showing them in database order. Labelling a crop the model already calls correctly with 99% probability teaches it nothing; the ones near the decision boundary are where a human's time actually moves the model. Needs model scores in png_list, so Classify (CV) has to have run - with none present the grid falls back to page order and says so rather than coming up empty. Crops that already carry an annotation are excluded. Default False.",
    'queue_measure': "(str) - How uncertainty is scored for the queue. 'entropy' spreads its attention across every class and is the right default for three or more; 'least_confidence' ranks on how weak the top class is; 'margin' ranks on the gap between the top two. With exactly two classes margin and least_confidence produce the IDENTICAL ranking, including ties - they only diverge at three classes or more. These are uncertainty scores, not calibrated confidences: a softmax is not a probability. Default 'entropy'.",
    'queue_diversity': "(str) - Which metadata level the queue is spread across before it is served. Ranking purely by uncertainty collapses: the hundred most uncertain crops on a real plate routinely come from one or two wells, so the annotator labels the same ambiguity a hundred times and the model learns nothing new. 'well' (the default) deals crops round-robin across wells, 'field' and 'plate' do the same at those levels, and 'none' turns the protection off and serves the raw ranking. The cost of diversity is that position two is the most uncertain crop in a DIFFERENT well and may be less uncertain than the overall runner-up. Default 'well'.",
    'queue_limit': "(int) - How many crops the queue holds. 0 (the default) queues the whole unlabelled pool. A limit smaller than the number of wells interacts with queue_diversity: you get roughly one crop from each of that many wells and none from the rest, which is useful for a quick sweep across a plate and misleading if you expected the top N by uncertainty. Default 0.",
    'crop_source': "(str) - Where single-object images come from. 'auto' (the default) uses the pre-generated PNG crop folder when one exists and otherwise cuts each crop out of merged/*.npy on demand; 'png' insists on the folder and fails if it is absent; 'merged' always cuts on demand and ignores the folder even when it is there. On-demand crops are pixel-identical to what the PNG folder would have held for the same settings, so annotations and models stay comparable across the two, and they cost no disk and cannot go stale when crop settings change - at the price of reading the merged array each time. Default 'auto'.",
    'class_balance': "(str) - How skew between the training classes is corrected. 'none' (the default) changes nothing but still prints the per-class counts, the majority-over-minority ratio and a recommendation, so the skew is never invisible. 'weighted_sampler' attaches a WeightedRandomSampler with 1/n weights, drawing every class about equally often; 'sqrt_weighted_sampler' uses 1/sqrt(n) for a gentler pull that avoids showing a tiny class so often the model memorises it; 'weighted_loss' leaves sampling alone and switches loss_type to 'ce_weighted' instead. Resampling is applied to the train loader only - validation and test keep the real prior so their scores stay comparable to the screen.",
    'cross_validation': "(bool) - Score the classifier with 5-fold stratified cross-validation instead of a single train/test split, so every control object receives an out-of-fold prediction and an optimal probability threshold is picked per fold. Gives a far more stable accuracy estimate on small control sets, at roughly 5x the training time. Default True.",
    'cross_validation_folds': "(int) - Number of k-fold splits the vision classifier is trained with in place of the single val_split hold-out. 0 (the default) or 1 keeps today's one random split; 2 or more trains a fresh model per fold, scores each on the fold it never saw, and reports the mean together with the fold-to-fold standard deviation and range - which is the only way to see whether one lucky split was flattering the model. Costs roughly k times the training time. Distinct from 'cross_validation', which is the regression pipeline's own toggle.",
    'cross_validation_enabled': "(bool) - Enable k-fold validation for Classify. If cross_validation_folds is 0 or 1, enabling this uses 5 folds. Use cv_group_by='plate' to hold out whole plates, or 'well'/'field' for within-plate validation without leaking related crops between training and validation.",
    'cv_group_by': "(str) - Which metadata level is kept intact across generated train/test data, the ordinary validation holdout and every CV fold: 'well' (the default and the right choice for object crops), 'field', 'plate', or 'none' for legacy per-object splitting. Crops from one well share focus, illumination, seeding density and edge effects, so letting them straddle a boundary lets the model recognise the well instead of the phenotype and inflates every score. The level is parsed from the crop filename, which spaCR writes as plate_well_field_object.png.",
    'classifier_evaluation': "(bool) - Build the Classifier Evaluation workbench bundle from out-of-fold predictions after Classify (CV): sample-level predictions, confusion matrices, reliability curves, calibrated probabilities, per-plate metrics, leakage reports and a manifest. It requires cross_validation_folds >= 2; a single train/validation split cannot produce unbiased out-of-fold diagnostics. Default True. API: spacr.classifier_evaluation.evaluate_predictions.",
    'nested_cv_inner_folds': "(int) - Number of inner grouped folds used inside every outer CV fold. 0 (default) keeps the faster ordinary grouped CV; 2 or more trains one inner model per fold, uses inner validation for early stopping/model selection, ensembles those models, and evaluates only once on the untouched outer fold. Runtime is approximately outer_folds x inner_folds training runs, but the outer score is not reused for tuning. API: spacr.classifier_evaluation.nested_group_folds.",
    'evaluation_calibration': "(str) - Probability calibration written to the evaluation bundle. 'temperature' cross-fits one scalar temperature per held-out fold using all other out-of-fold predictions, so a sample never fits its own calibrator; 'none' retains raw softmax probabilities. Calibration changes reported probabilities, not the saved model weights. Default 'temperature'. API: spacr.classifier_evaluation.cross_calibrate_probabilities.",
    'evaluation_bins': "(int) - Number of equal-width probability bins in reliability curves and expected calibration error. Values around 10 balance resolution against noise; use fewer bins for small validation sets and more only when every class has many hundreds of out-of-fold samples. Minimum 2, default 10. API: spacr.classifier_evaluation.calibration_table.",
    'evaluation_fail_on_leakage': "(bool) - Stop Classify (CV) before fitting a fold when the same object, augmentation family, or protected cv_group_by identity appears in both train and validation. False records the problem and continues, which is useful only for diagnosing a legacy dataset because its performance estimate remains invalid. Default True. API: spacr.classifier_evaluation.audit_split_leakage.",
    'leakage_audit_train_test': "(bool) - Audit the permanent train/ and test/ boundary before any classifier fit. Checks plate/well/field/object lineage, exported augmentation families and (when enabled) byte-identical renamed copies. Default True. API: spacr.classifier_evaluation.audit_dataset_splits.",
    'leakage_hash_content': "(bool) - SHA-256 hash classifier images during leakage audits so an identical crop copied or renamed across train/test or CV boundaries is still detected. Reads files in 1 MiB chunks and never decodes pixels. Default True. API: spacr.classifier_evaluation.audit_cv_folds.",
    'leakage_require_identity': "(bool) - Treat filenames that do not encode the protected cv_group_by identity, and files that cannot be hashed, as a failed audit rather than an advisory warning. Default True because independence cannot be claimed when lineage is unknown. API: spacr.classifier_evaluation.audit_split_leakage.",
    'custom_measurement': "(str) - Optional measurement-column name intended for class assignment; the Tk dataset dialog collects it but no pipeline code reads the key, so it currently has no effect. To select classes by a measured feature use dataset_mode 'measurement' with measurement_rules instead. Default None.",
    'denoise': "(bool) - Legacy denoising toggle for the mask pipeline: no code reads this key, so it has no effect. To actually denoise, set the per-object restore settings (cell_restore_type / nucleus_restore_type / pathogen_restore_type) to 'denoise', which routes segmentation through Cellpose's CellposeDenoiseModel. Default False.",
    'early_stopping_patience': "(int) - Stop training after this many consecutive epochs in which validation accuracy fails to beat the best value so far; the best checkpoint is still kept. 0 (default) disables it and always runs the full 'epochs' budget. Set 10-20 on long runs to cut wasted epochs once the model plateaus.",
    'tensorboard': "(bool) - Write live PyTorch loss, accuracy, macro-F1 and learning-rate events to dst/tensorboard while the vision model trains. Open that folder with tensorboard --logdir PATH for an interactive dashboard that can compare runs. The in-app zoomable loss/accuracy monitor is controlled separately by plot. Default True.",
    'filter_column': "(str) - Metadata column used to drop control wells before regression: every row whose value appears in filter_value is removed from both the score data and the read counts. Use 'columnID' (default) when controls sit in plate columns, 'rowID' when they sit in rows. In annotate_filter_vision it instead names the score column thresholded by upper_threshold/lower_threshold.",
    'filter_min_max': "(list) - Display-only size filter for plot_merged: one [min_area, max_area] pair in pixels per mask dimension, in the order cell, nucleus, pathogen, e.g. [[500,50000],[100,5000],[10,2000]]. Objects outside a pair are erased from that mask before the overlay is drawn. None (default) keeps every object.",
    'filter_value': "(list) - Values of filter_column whose rows are removed - not kept - before regression, normally the control columns; default ['c1','c2','c3']. Dropping them stops control wells from dominating the gene and gRNA fits. Only list values take effect: a bare string is silently ignored and nothing is filtered.",
    'focal_alpha': "(float) - Class-balancing weight for focal loss (read only when loss_type resolves to focal). In the single-logit binary path it scales positives by alpha and negatives by 1-alpha, so raise it toward 1 to emphasise a rare positive class; with two or more output classes a plain float scales the whole loss uniformly. Default None (no alpha weighting).",
    'focal_gamma': "(float) - Focusing exponent in the focal-loss weight (1 - p_t)^gamma, applied only when loss_type is focal. 0 reduces it to plain cross-entropy; raising it (typically 1-5) down-weights crops the model already classifies well and pushes gradient onto hard ones. Default 2.0. Increase when one class dominates and training stalls on easy examples.",
    'generate_training_dataset': "(bool) - Rebuild the train/ and test/ PNG folders from the object crops before training, using the dataset_mode rules (annotation_column labels, metadata rules or measurement rules) and splitting off test_split of the images. Turn it off to reuse an existing split; it is only consulted when train or test is True, and a failed build aborts training. Default True.",
    'label_smoothing': "(float) - Epsilon passed to cross-entropy when loss_type is label_smoothing: each target keeps 1 - eps of its probability mass and the rest is spread across the other classes. Raise it (typically 0.05-0.2) when the model gets over-confident or annotations are noisy; 0 disables. Ignored by every other loss type. Default 0.1.",
    'log_x': "(bool) - Put the x-axis on a log10 scale; for line graphs the x column is log10-transformed instead of the axis being rescaled. Enable it when x spans orders of magnitude - gRNA fraction thresholds, count distributions - so the low end is not squashed against the axis. Values at or below zero cannot be shown. Default False.",
    'log_y': "(bool) - Put the y-axis on a log10 scale; for line graphs the y column is log10-transformed instead of the axis being rescaled. Enable it when the measured values span orders of magnitude or a few large wells compress everything else toward the baseline. Values at or below zero cannot be shown. Default False.",
    'logit_adjust_tau': "(float) - Strength of the Menon-et-al. logit adjustment: tau * log(class prior) is added to the logits during training, pulling decisions toward rare classes. Only used when loss_type resolves to logit_adjust_ce, which 'auto' picks when the smallest class is under 10% of the data. Higher tau corrects harder; 0 disables. Default 1.0.",
    'loss_type': "(str) - Which loss build_loss constructs for the classifier. For a 2+ class head the working values are 'focal_loss'/'focal_ce' (down-weights easy examples), 'cross_entropy'/'ce', 'label_smoothing'/'ce_smooth' (epsilon fixed at 0.1), 'ce_weighted' (inverse-frequency class weights), 'logit_adjust_ce' and 'asl'; 'binary_cross_entropy_with_logits'/'bce' is legal only for a single-logit head and raises otherwise. 'auto' is rewritten to 'cross_entropy' before training starts, so build_loss's rare-class switch to logit_adjust_ce never fires from this path. Reach for 'focal_loss' or 'ce_weighted' when one class dominates and the model collapses to predicting it. Default 'focal_loss' ('auto' only in deep_spacr_defaults).",
    'metadata_files': "(list) - Gene-annotation CSVs, each with a 'Gene ID' column, that are joined onto the regression results by gene, writing an extra results CSV per file. These are gene tables, not plate/well metadata. When toxo is True the order matters: index 0 is read as the ME49 transcription table and index 1 as the GT1 phenotype table.",
    'metadata_type_by': "(str) - Which png_list column the class_metadata values are matched against when dataset_mode is 'metadata'. Normally 'columnID' or 'rowID' - the two well-metadata columns filepaths_to_database writes - but any png_list column is accepted, including 'condition' if annotate_conditions has added it. generate_training_dataset selects on this column and raises with the list of available columns when it is missing; it used to ignore this setting entirely and select on a hard-coded 'condition' column, which no writer creates, so a default configuration died with KeyError('condition'). The legacy training_dataset_from_annotation_metadata helper still accepts 'rowID'/'columnID' only. Default 'columnID'.",
    'min_n': "(int) - Observation count a significant hit must strictly exceed to appear in results_significant_filtered.csv: gRNA hits need n_grna > min_n, gene hits need n_gene > min_n. The unfiltered hit list is still written alongside it. Raise it to drop hits resting on one or two wells. Default 0, which filters nothing.",
    'normalization_percentiles': "(list) - Two-element [low, high] percentile pair used to stretch each channel's non-zero pixels to the full display range in plot_merged; applied only when normalize is True. Narrowing the pair (e.g. [5, 95]) boosts contrast but saturates bright objects; widening it flattens the image. Default [2, 98].",
    'nr_imgs': "(int) - How many object crops are pulled into each representative-image grid: the sampler takes this many per condition, or all of them if fewer exist. Raise it for a more representative panel at the cost of a bigger, slower figure; lower it for a quick look. Positive integer; the plotting helpers default to 16.",
    'nucleus_chann_dim': "(int) - Recruitment analysis only (analyze_recruitment): the image-channel index paired with the nucleus mask when drawing outline overlays, and the switch that enables nucleus_size_range / nucleus_intensity_range filtering. Set it to None to skip nucleus filtering. It plays no part in segmentation - use nucleus_channel for that. Default 0.",
    'nucleus_intensity_range': "(list) - Two-element [min, max] bound on mean nucleus-channel intensity used by the recruitment analysis to drop rows from the measurement table - it filters measured objects, not masks or normalization. Rows are kept only if min < mean intensity < max (raw units), and each bound is ignored unless it is an int. Default [0, 100000].",
    'nucleus_size_range': "(list) - Two-element [min, max] bound in pixels^2 on nucleus_area, used by the recruitment analysis to drop rows from the measurement table; masks are left untouched. Rows are kept only if min < area < max, and each bound is ignored unless it is an int. Default [0, 100000]; None widens it to [0, 1e100].",
    'offset_start': "(int) - Bases to shift from the start of the target_sequence match to the start of the extracted window; negative values move upstream to capture a barcode preceding the anchor. The start is clamped at position 0, so an over-negative value silently shifts the reading frame and the regex stops matching. Default -8.",
    'optimizer_type': "(str) - PyTorch optimizer used by deep_spacr.train_model: 'adamw', 'adam', 'adamax', 'sgd', 'rmsprop', 'nadam', 'radam', 'adagrad', 'adadelta' or 'asgd'. AdamW is the robust fine-tuning default; SGD can generalise better but usually needs more epochs. amsgrad applies only to Adam/AdamW. API: spacr.deep_spacr.train_model(optimizer_type=...). Default 'adamw'.",
    'schedule': "(str) - Learning-rate scheduler used by spacr.deep_spacr.train_model: 'cosine', 'cosine_warm_restarts', 'reduce_lr_on_plateau', 'step_lr', 'exponential', 'linear', or 'none'. Plateau reacts to validation loss; cosine and linear use the epoch budget; warm restarts periodically raise the rate to escape a narrow minimum. API: train_model(schedule=...). Default 'cosine'.",
    'outlier_detection': "(bool) - After building the regression table, drop gRNAs whose well count falls outside 1.5x the 5th-95th percentile spread, then recompute the per-gRNA tables. This removes gRNAs present in implausibly few or many wells that would otherwise dominate coefficients; disable it if your library is deliberately uneven. Default True.",
    'outline_color': "(str) - Three-letter code choosing which RGB colours the cell, nucleus and pathogen outlines get, in that order: 'rgb', 'bgr', 'gbr' or 'rbg'. Default 'gbr' draws cells green, nuclei blue and pathogens red; any unrecognised string silently falls back to 'rbg'. Change it when an outline colour clashes with a channel. In the Mask Live settings popup, the preview-only Outline colour control also offers 'auto', fixed named colours, and 'color (random)'; the random option gives every segmented label a vivid, stable categorical colour so neighbouring objects are easy to distinguish without flickering during refreshes. API: spacr.qt.widgets.live_preview.overlay_masks(random_outline=True).",
    'outline_thickness': "(int) - Width in pixels of the mask outlines on the merged overlay; the contour is drawn at this thickness and then dilated by a square of the same size, so the visible line is roughly twice the value. Raise it for large fields where a 1-2 px outline disappears. Default 3.",
    'overlay_chans': "(list) - Exactly three channel indices from the stack, mapped in order onto the red, green and blue planes of the merged overlay. Default [1, 2, 3] puts channel 1 in red, 2 in green and 3 in blue; reorder or repeat indices to change which stain reads as which colour. Indices past the stack's channel count are ignored.",
    'pathogen_chann_dim': "(int) - Recruitment analysis only (analyze_recruitment): the image-channel index paired with the pathogen mask when drawing outline overlays, and the switch that enables pathogen_size_range / pathogen_intensity_range filtering. Set it to None to skip pathogen filtering. It plays no part in segmentation - use pathogen_channel for that. Default 2.",
    'pathogen_intensity_range': "(list) - Two-element [min, max] mean-intensity filter applied to the pathogen table in analyze_recruitment; pathogens whose mean intensity in the paired mask channel falls outside the open interval are dropped before recruitment ratios are computed. Bounds must be ints - floats are silently ignored. Default [0, 100000]. Use it to exclude dead or saturated parasites.",
    'nucleus_loc': "(list of lists) - Intended as the well locations of each nucleus condition, matching cell_loc and pathogen_loc, but there is no nucleus condition axis: annotate_filter_vision reads cell_loc, pathogen_loc and treatment_loc and never looks at settings['nucleus_loc'], and no defaults function sets it. Setting it annotates nothing. Kept declared only so old settings CSVs still load, and rejected by the pre-flight check and by spacr-run --set.",
    'pathogen_loc': "(list of lists) - Well locations of each pathogen condition, one inner list per name in pathogens, read by annotate_filter_vision when labelling vision-model score CSVs. Every entry must be a row or column ID string such as 'c1' or 'r3'; ranges are not expanded and unmatched entries leave those wells NaN. Set it alongside pathogens, or leave both None.",
    'pathogens': "(list) - Names of the pathogen conditions scored by annotate_filter_vision, e.g. ['wt','mutant']. Element i is written into the pathogen column for every well in pathogen_loc[i] and folded into the combined condition label. Must match pathogen_loc element for element; if pathogen_loc is None, only the first name is applied to every row.",
    'pick_slice': "(bool) - Intended to keep one z-slice instead of a maximum-intensity projection, but nothing in spaCR reads settings['pick_slice'] and no defaults function sets it, so turning it on changes nothing and the stack is projected anyway. The z controls that do work are z_stack, z_segmentation_mode and z_projection, whose 'best_focus' value is the one that picks a single plane. Rejected by the pre-flight check and by spacr-run --set.",
    'png_type': "(str) - Which object crop type is pulled from the png_list table when building the training dataset - a row is kept only if its PNG path contains this substring. Use 'cell_png', 'nucleus_png', 'pathogen_png', 'cytoplasm_png' or 'organelle_png' to train on whole cells, nuclei, parasites, cytoplasm or organelles. It must match a crop_mode that measure_crop actually saved. Default 'cell_png'.",
    'prune_features': "(bool) - Before training, keep only the top_features columns with the highest ANOVA F-score against the control labels (sklearn SelectKBest with f_classif). Speeds up fitting and can curb overfitting on small control sets, but discards features the model might have used and scores each feature in isolation, ignoring interactions. Default False.",
    "redunction_method": "(str) - Misspelled duplicate of reduction_method ('redunction'), and nothing reads it. Set reduction_method instead, which reduction_and_clustering actually consumes and which accepts 'umap' or 'tsne' (not 'pca', despite this legacy text). Kept only so old settings CSVs still load.",
    'reg_alpha': "(float) - L1 penalty on leaf weights for the gradient-boosted classifier (XGBoost and LightGBM; ignored by the other model_type_ml choices). Raising it drives more leaf weights to exactly zero, shrinking the model and its effective feature set - raise it when training accuracy far exceeds test accuracy. Any value >= 0. Default 0.1.",
    'reg_lambda': "(float) - L2 penalty on leaf weights for the gradient-boosted classifier (XGBoost, LightGBM, and CatBoost's l2_leaf_reg). Raising it shrinks all weights smoothly rather than zeroing them, damping the influence of any single feature and curbing overfitting, at the risk of underfitting if pushed too far. Any value >= 0. Default 1.0.",
    'remove_border_cells': "(bool) - Legacy duplicate of cell_remove_border_objects, intended to drop cells touching the image border. No code path in spaCR reads this key and it is never given a default, so setting it has no effect. Use cell_remove_border_objects, which the mask pipeline and the live preview actually apply.",
    'remove_border_nuclei': "(bool) - Intended to remove nucleus objects touching the image border, but no code in spaCR reads this key, so setting it has nothing to act on. It also has no default: no set_default_* function ever calls setdefault for it, and it appears only in the expected-types map, the tooltip dict and the GUI category list. The segmentation pipeline applies that filter via nucleus_remove_border_objects (default False, read at object.py:58) - set that one instead.",
    'remove_border_organelles': "(bool) - Legacy duplicate of the border filter for organelles. No code path in spacr reads this key and it has no default, so setting it has no effect; use organelle_remove_border for batch mask generation or organelle_remove_border_objects for the shared post-segmentation filter. Left in place only for backwards compatibility with old settings files.",
    'remove_border_pathogens': "(bool) - Legacy duplicate of pathogen_remove_border_objects, intended to drop pathogens touching the image border. No code path in spaCR reads this key and it is never given a default, so setting it has no effect; its three siblings say so and this one used to claim it worked. Use pathogen_remove_border_objects, which the mask pipeline and the live preview actually apply. Rejected by the pre-flight check and by spacr-run --set.",
    'reverse_complement': "(bool) - Whether to reverse-complement the read before matching barcodes. Set according to which strand the barcodes were sequenced on.",
    'save_to_db': "(bool) - After ML screen analysis, write the per-object model scores back into measurements.db as a 'predictions' column on the png_list table, matched on prcfo. Enable when you want to sort, filter or plot objects by score in the GUI; the CSV result files are written either way. Default False.",
    'score_data': "(str or list) - CSV(s) of per-object or per-well phenotype scores (typically the output of generate_ml_scores) supplying the regression's dependent variable; the column named by dependent_variable must be present or the run raises ValueError. Pass one path per plate, position-aligned with plates_score; the first file's name becomes the results subfolder.",
    'single_direction': "(str) - Which mate to scan when mode is 'single': 'R1' or 'R2'. The chosen file is read as-is with no reverse-complementing, so selecting 'R2' means target_sequence and regex must be written in R2 orientation or nothing will match. Ignored when mode is 'paired'. Default 'R1'.",
    "split_axis_lims": "(str) - Intended to fix the axis limits of split/faceted plots as [xmin, xmax, ymin, ymax], but nothing reads settings['split_axis_lims']; those plots always autoscale. Use the per-plot x_lim / y_lim settings where the plotting function exposes them. Kept only so old settings CSVs still load.",
    'target_unique_count': "(int) - Desired mean number of distinct gRNAs per well. spaCR sweeps 1000 read-fraction thresholds, picks the one whose per-well mean unique gRNA count lands closest to this number, then discards every gRNA call below that fraction. Lower it for a stricter, cleaner well assignment; raise it to keep more gRNAs per well. Default 5.",
    'threshold_method': "(str) - How the spread of the control-gRNA regression coefficients is measured when the hit-calling threshold is built: 'std' or 'standard_deveation' uses the standard deviation, 'var' or 'variance' uses the variance (much wider once the spread exceeds 1). Any other value raises an error. Only used when 'controls' is set. Default 'std'.",
    'threshold_multiplier': "(float) - How many units of control-coefficient spread are added to the mean control coefficient to form the regression hit threshold: reg_threshold = mean(control coefficients) + multiplier * spread, where spread comes from threshold_method. Larger values place the threshold further out in the control distribution. Only used when 'controls' is set. Default 3.",
    'toxo': "(bool) - Merge the regression hits with the bundled Toxoplasma metadata (LOPIT/TAGM localisations in resources/data/lopit.csv) and, from that, draw the volcano plot plus GT1 phenotype and ME49 transcription heatmaps read from metadata_files. Turn it off for non-Toxoplasma screens - doing so also disables the volcano plot entirely. Default True.",
    'use_checkpoint': "(bool) - Run the backbone's forward pass through torch.utils.checkpoint: intermediate activations are discarded and recomputed during the backward pass, trading extra compute for a large drop in activation memory. Enable when a bigger batch_size or image_size gives CUDA out-of-memory; disable for the fastest epochs when VRAM is not the constraint. Default True.",
    'volcano': "(str) - Which coefficient table the volcano plot is drawn from: 'gene' (default) plots per-gene coefficients, 'grna' per-gRNA, 'all' the full merged table; any other value skips the plot. Points are coloured by TAGM/LOPIT localisation, and the gene list it returns drives the phenotype and transcription plots. Only takes effect when toxo is True.",
    'x_lim': "(list) - Two-element [min, max] limits on the coefficient (x) axis of the Toxoplasma volcano plot produced by the regression pipeline when toxo mode is on. Narrow it to zoom in on hits clustered near zero, widen it to keep large-effect genes on the plot. Leaving it None falls back to [-0.5, 0.5], not auto-scaling. Default None."
}

# Keys owned by the standalone Timelapse module (spacr.qt app key 'timelapse').
# NOTE `timelapse` itself is NOT in this list: it lives in the "General"
# category because the Tk GUI reveals the "Timelapse" category only once that
# box is ticked (see category_dependencies), so the toggle cannot live inside
# the category it controls. Consumers that want "everything timelapse" should
# use `timelapse_settings + ['timelapse']`.
timelapse_settings = ['fps', 'timelapse_mode', 'trackastra_model', 'trackastra_linking', 'ultrack_max_distance', 'ultrack_division_weight', 'ultrack_contour_sigma', 'ultrack_n_workers', 'timelapse_displacement', 'timelapse_memory', 'timelapse_frame_limits', 'timelapse_remove_transient', 'timelapse_objects', 'compartments']

motility_settings = ['motility_analysis','tracked_object', 'infection_intensity_strategy', 'seconds_per_frame', 'pixels_per_um', 'motility_ylim', 'motility_xlim', 'infection_intensity_qc_scope']

motility_advanced_settings = ['reuse_existing_measurements', 'infection_xgb_min_cells_per_class', 'infection_xgb_n_estimators', 'infection_xgb_max_depth', 'infection_xgb_learning_rate', 'infection_xgb_subsample', 'infection_xgb_colsample_bytree', 
                     'infection_xgb_reg_lambda', 'infection_xgb_random_state', 'infection_xgb_n_jobs', 'infection_xgb_proba_threshold', 'infection_xgb_margin', 'infection_xgb_top_features', 'infection_xgb_proba_column', 'infection_xgb_proba', 
                     'infection_xgb_drop_ambiguous', 'infection_xgb_ambiguous_low','infection_xgb_ambiguous_high','infection_pca_method', 'infection_pca_n_clusters', 'infection_pca_random_state', 'infection_intensity_n_bins', 'db_table_name', 
                     'infection_intensity_qc_graphs', 'infection_intensity_qc_panel_path', 'infection_intensity_mode', 'infection_intensity_qc', 'straightness_threshold', 'straightness_filter', 'zscore_thresh', 'max_displacement',
                     'infection_pca_umap_search','infection_pca_umap_n_neighbors_grid','infection_pca_umap_min_dist_grid','infection_pca_pathogen_weight', 'infection_pca_log_intensity','infection_pca_tsne_search','infection_pca_tsne_perplexity_grid',
                     'infection_pca_tsne_learning_rate_grid', 'infection_pca_umap_n_neighbors','infection_pca_umap_min_dist','infection_pca_tsne_perplexity', 'infection_pca_min_silhouette','infection_pca_min_gt_separation','infection_pca_max_cells']

# How the settings panel is grouped in BOTH GUIs: the Tk category dropdown
# (gui_core.toggle_settings) and the Qt section boxes
# (qt/screens/settings_model.SettingsWidgets.build_sections) read this map and
# nothing else. One entry = one heading, rendered in the order written here.
#
# Three rules keep it usable, and tests/test_settings_categories.py enforces
# all three:
#   1. Every key produced by a module's set_default_* / get_*_settings helper
#      appears here. An uncategorised key is not grouped at all: Tk pins it to
#      the top of the panel as an always-visible field and Qt dumps it in the
#      trailing "Other" section.
#   2. No key appears twice. A duplicate renders twice in Tk (and each copy is
#      shown/hidden by a different heading) and is silently dropped from the
#      second section in Qt.
#   3. A setting that TRIGGERS a category - see category_dependencies and
#      category_integer_dependencies below - must live outside the category it
#      reveals, or ticking it off hides the control that turns it back on.
#      That is why `timelapse` sits in General and not in "Timelapse", and why
#      organelle_channel / organelle_mask_dim sit in General and not in
#      "Organelle".
categories = {
    "Paths": ["src", "grna", "barcodes", "custom_model_path", "resume_checkpoint", "dataset", "model_path", "tar_path", "grna_csv", "row_csv", "column_csv", "metadata_files", "score_data", "count_data"],

    # 'normalize' moved here from "Advanced". It is a top-level toggle for how
    # every image in the run is scaled, set by seven different modules, and
    # burying it under "rarely-touched knobs" was wrong in all of them - not
    # least Classify, where it shapes the training set.
    "General": ["cell_mask_dim", "cytoplasm", "cell_chann_dim", "cell_channel", "nucleus_chann_dim", "nucleus_channel", "nucleus_mask_dim", "organelle_channel", "organelle_mask_dim", "organelle_chann_dim", "pathogen_mask_dim", "pathogen_chann_dim", "pathogen_channel", "channels", "channel_dims", "normalize", "magnification", "metadata_type", "custom_regex", "experiment", "plot", "test_mode", "timelapse", "apply_model_to_dataset", "generate_training_dataset", "generate_full_dataset", "delete_intermediate", "uninfected"],

    # How Cellpose RUNS. Which model it runs (model_name / custom_model) moved
    # to "Model Training": they are the same question the torch classifier's
    # model_type answers, and 'custom_model' under a "Cellpose" heading was the
    # only reason the Classify (CV) panel had an "Other" section at all - that
    # module hides "Cellpose", so its one key fell out of every group.
    "Cellpose": ["fill_in", "from_scratch", "n_epochs", "width_height", "target_size", "resample", "rescale", "CP_prob", "flow_threshold", "percentiles", "invert", "diameter", "grayscale", "Signal_to_noise", "resize", "target_height", "target_width"],

    "Cell": ["cell_model_name", "cell_diameter", "cell_background", "cell_Signal_to_noise", "cell_CP_prob", "cell_FT", "remove_background_cell", "adjust_cells", "cell_max_area", "cell_min_area", "cell_remove_border_objects", "cell_min_intensity_percentile", "cell_max_intensity_percentile", "remove_border_cells", "cell_perimeter_fraction", "cell_intensity_merge", "cell_intensity_split", "cell_area_multiplier", "cell_min_distance", "cell_min_object_area", "cell_intensity_threshold_method", "cell_intensity_percentile"],

    "Nucleus": ["nucleus_model_name", "nucleus_diameter", "nucleus_background", "nucleus_Signal_to_noise", "nucleus_CP_prob", "nucleus_FT", "remove_background_nucleus", "nucleus_min_area", "nucleus_max_area", "nucleus_remove_border_objects", "nucleus_min_intensity_percentile", "nucleus_max_intensity_percentile", "remove_border_nuclei", "nucleus_perimeter_fraction", "nucleus_intensity_merge", "nucleus_intensity_split", "nucleus_area_multiplier", "nucleus_min_distance", "nucleus_min_object_area", "nucleus_intensity_percentile", "nucleus_intensity_threshold_method"],

    "Pathogen": ["pathogen_model_name", "pathogen_diameter", "pathogen_background", "pathogen_Signal_to_noise", "pathogen_CP_prob", "pathogen_FT", "pathogen_model", "remove_background_pathogen", "pathogen_max_area", "pathogen_min_area", "pathogen_remove_border_objects", "pathogen_min_intensity_percentile", "pathogen_max_intensity_percentile", "remove_border_pathogens", "pathogen_perimeter_fraction", "pathogen_intensity_merge", "pathogen_intensity_split", "pathogen_area_multiplier", "pathogen_min_distance", "pathogen_min_object_area", "pathogen_intensity_threshold_method", "pathogen_intensity_percentile"],

    # One heading for the whole organelle workflow, ordered the way it is set
    # up: what to detect -> clean the image -> the knobs of the chosen
    # organelle_method -> filter the objects -> what to summarise. The
    # per-method blocks used to be eight separate headings gated on
    # organelle_method; they are sub-ordered here instead, so the knobs that do
    # not apply to your method are simply further down the list.
    "Organelle": [
        # what to detect
        "organelle_morphology", "organelle_method", "organelle_diameter",
        # clean the image first
        "organelle_mask_within_cells", "organelle_rolling_ball", "organelle_rolling_ball_radius", "organelle_clahe", "organelle_clahe_clip_limit",
        # method: adaptive
        "organelle_adaptive_block_size", "organelle_adaptive_offset",
        # method: otsu / adaptive / log / dog (spots)
        "organelle_tophat_radius", "organelle_watershed_spots", "organelle_log_min_sigma", "organelle_log_max_sigma", "organelle_log_num_sigma", "organelle_log_threshold", "organelle_dog_sigma_low", "organelle_dog_sigma_high",
        # method: ridge / hysteresis (networks)
        "organelle_ridge_filter", "organelle_ridge_sigmas", "organelle_skeletonize", "organelle_network_threshold", "organelle_hysteresis_low", "organelle_hysteresis_high",
        # morphology: ring
        "organelle_ring_sigma_inner", "organelle_ring_sigma_outer", "organelle_ring_min_prominence", "organelle_ring_fill_method",
        # morphology: irregular
        "organelle_morph_radius", "organelle_fill_holes",
        # method: cellpose
        "organelle_model_name", "organelle_CP_prob", "organelle_FT", "organelle_resample",
        # method: unet
        "organelle_unet_model_path", "organelle_unet_threshold",
        # filter the detected objects
        "organelle_min_size", "organelle_max_size", "organelle_min_area", "organelle_max_area", "organelle_min_object_area", "organelle_area_multiplier", "organelle_min_distance", "organelle_perimeter_fraction", "organelle_intensity_merge", "organelle_intensity_split", "organelle_intensity_threshold_method", "organelle_intensity_percentile", "organelle_min_intensity_percentile", "organelle_max_intensity_percentile", "organelle_remove_border", "organelle_remove_border_objects", "remove_border_organelles",
        # what to write out
        "summarize_organelles_by",
    ],

    "Segmentation QC": ["seg_qc", "seg_qc_min_objects", "seg_qc_count_ratio", "seg_qc_size_ratio", "seg_qc_border_fraction", "seg_qc_outlier_mad", "seg_qc_outlier_fraction", "seg_qc_foreground_fraction", "seg_qc_split_ratio", "seg_qc_min_diameter", "seg_qc_tiny_fraction", "seg_qc_max_object_fraction", "seg_qc_plate_fail_fraction"],

    "Timelapse": timelapse_settings,

    # Which objects are measured, which features are computed, and which of
    # them survive into the analysis table. Plot-only knobs that used to live
    # here (image_nr, dot_size, remove_image_canvas) moved to "Plot".
    #
    # Three groups arrived here in the regroup:
    #   * the per-object minimum sizes and merge_edge_pathogen_cells, which
    #     only measure_crop sets. They sat under the Cell / Nucleus / Pathogen
    #     SEGMENTATION headings, so the Measure module rendered three headings
    #     holding one or two size filters each and no segmentation at all.
    #   * nuclei_limit / pathogen_limit, from "Advanced". They decide whether
    #     the nucleus and pathogen tables are joined onto the object table --
    #     which rows exist, not a tuning knob.
    #   * parasite_table / compartment, from "Invasion Assay", which name the
    #     table and compartment the objects are read from. Leaving them there
    #     made the Replication module render a heading called "Invasion Assay".
    "Measurements": ["save_measurements", "calculate_correlation", "manders_thresholds", "homogeneity", "homogeneity_distances", "radial_dist", "distance_gaussian_sigma", "tables", "parasite_table", "compartment", "channel_of_interest", "measurement", "filter_by", "exclude", "cell_min_size", "cytoplasm_min_size", "nucleus_min_size", "pathogen_min_size", "merge_edge_pathogen_cells", "cell_size_range", "cell_intensity_range", "nucleus_size_range", "nucleus_intensity_range", "pathogen_size_range", "pathogen_intensity_range", "cells_per_well", "target_intensity_min", "nuclei_limit", "pathogen_limit", "remove_highly_correlated", "remove_highly_correlated_features", "remove_low_variance_features"],

    "Object Crops": ["save_png", "crop_mode", "png_size", "png_dims", "dialate_pngs", "dialate_png_ratios", "use_bounding_box", "normalize_by", "save_arrays"],

    # The plate map: which wells hold which condition, which wells are the
    # controls, and how they are labelled. Gathers the per-object condition
    # lists that used to sit inside the Cell / Nucleus / Pathogen segmentation
    # headings, where they had nothing to do with segmentation.
    # ...plus how the wells are grouped for reporting: group_column / level /
    # change_plate came from "Invasion Assay", where they were shared with the
    # replication assay and so gave that module a heading named after an assay
    # it does not run.
    "Plate Layout & Controls": ["plateID", "plate", "cell_types", "cell_plate_metadata", "cells", "cell_loc", "nucleus_loc", "pathogen_types", "pathogen_plate_metadata", "pathogens", "pathogen_loc", "treatments", "treatment_plate_metadata", "treatment_loc", "location_column", "group_column", "level", "change_plate", "positive_control", "negative_control", "controls", "pc", "nc", "pc_loc", "nc_loc", "pos", "neg", "mix", "exclude_conditions", "exclude_rows", "filter_column", "filter_value", "target", "metadata_types", "batch_correction", "batch_column", "batch_control_column", "batch_control_values", "batch_min_samples", "batch_missing_control"],

    # How the labelled set is assembled, in the order it is assembled:
    # which rule defines a class -> what the classes are -> which crops ->
    # how many -> how they are split. 'test_split' came from "Model Training":
    # generate_training_dataset is what consumes it, writing the train/ and
    # test/ folders before any model exists. The four metadata_item_* keys had
    # no category at all and printed under "Other".
    "Training Dataset": ["dataset_mode", "annotation_column", "annotated_classes", "class_metadata", "metadata_type_by", "metadata_item_1_name", "metadata_item_1_value", "metadata_item_2_name", "metadata_item_2_value", "file_metadata", "custom_measurement", "png_type", "file_type", "sample", "size", "test_split", "balance_to_smallest", "write_random_annotation_column"],

    # Which model, and how it is fitted. 'model_name' and 'custom_model' moved
    # here from "Cellpose" -- they answer the same question 'model_type' does.
    "Model Training": ["model_type", "model_name", "custom_model", "classes", "train_channels", "image_size", "init_weights", "train", "test", "val_split", "epochs", "optimizer_type", "learning_rate", "schedule", "weight_decay", "dropout_rate", "loss_type", "label_smoothing", "focal_gamma", "focal_alpha", "logit_adjust_tau", "class_balance", "augment", "amsgrad", "use_checkpoint", "gradient_accumulation", "gradient_accumulation_steps", "early_stopping_patience", "pin_memory", "cross_validation_enabled", "cross_validation_folds", "cv_group_by", "classifier_evaluation", "nested_cv_inner_folds", "evaluation_calibration", "evaluation_bins", "evaluation_fail_on_leakage", "leakage_audit_train_test", "leakage_hash_content", "leakage_require_identity", "score_threshold", "n_top_examples", "random_seed", "intermedeate_save", "tensorboard"],

    # The classical (non-image) screen classifier fitted on measured features -
    # spacr's "Classify (ML)" module. These knobs used to be split three ways
    # between General, Advanced and the regression heading.
    "ML Classifier": ["model_type_ml", "n_estimators", "test_size", "cross_validation", "prune_features", "top_features", "n_repeats", "reg_lambda", "reg_alpha", "minimum_cell_count", "save_to_db"],

    "Embedding & Clustering": ["reduction_method", "n_neighbors", "min_dist", "metric", "log_data", "embedding_by_controls", "col_to_compare", "resnet_features", "visualize", "clustering", "eps", "min_samples", "remove_cluster_noise", "analyze_clusters"],

    # The per-model knobs (l1_ratio ... lasso_selection_threshold) sit here
    # beside regression_type because that is the setting that decides whether
    # each of them does anything at all: spacr.ml.REGRESSION_SETTINGS_USED says
    # which type reads which, and a type refuses the ones it cannot read.
    "Regression": ["regression_type", "dependent_variable", "score_column", "invert_dependent_variable", "agg_type", "transform", "alpha", "l1_ratio", "quantile", "hinge_threshold", "hinge_n_boot", "huber_t", "lasso_n_boot", "lasso_selection_threshold", "cov_type", "random_row_column_effects", "min_cell_count", "tolerance", "fraction_threshold", "target_unique_count", "outlier_detection", "threshold_method", "threshold_multiplier", "min_n", "volcano", "toxo", "other"],

    "Activation Maps": ["smoothgrad_samples", "smoothgrad_sigma", "occlusion_window", "occlusion_stride", "ig_steps", "ig_baseline", "attribution_steps", "attribution_baseline", "sanity_check", "object_type", "cam_type", "target_layer", "overlay", "correlation", "normalize_input"],

    "Sequencing": ["mode", "single_direction", "signal_direction", "target_sequence", "regex", "offset", "offset_start", "expected_end", "chunk_size", "fill_na", "save_h5", "comp_type", "comp_level"],

    "Plot": ["cmap", "figuresize", "normalize_plots", "black_background", "save_figure", "log_x", "log_y", "x_lim", "y_lims", "split_axis_lims", "examples_to_plot", "plot_control", "plot_nr", "nr_imgs", "um_per_pixel", "image_nr", "dot_size", "point_color", "point_alpha", "outline_width", "umap_canvas_width", "umap_sidebar_width", "img_zoom", "row_limit", "color_by", "plot_images", "remove_image_canvas", "plot_points", "plot_outlines", "smooth_lines", "plot_by_cluster", "plot_cluster_grids", "heatmap_feature", "grouping", "min_max", "highlight"],
    # Replication-specific vacuole assignment and scoring. The shared parasite
    # area filters and empty-well seeding control remain listed once under
    # "Invasion Assay"; the Qt app-specific category map presents those shared
    # keys under Replication's Object Filtering/Scoring sections.
    "Replication Assay": [
        "vacuole_key", "vacuole_link_distance", "vacuole_link_factor",
        "parasite_count_column", "max_parasites_per_vacuole",
        "require_host_cell", "non_power_of_two_warn",
    ],
    "Endodyogeny Size Proxy (Legacy)": [
        "class_column", "group_by_class", "um_per_px",
        "min_area_bin", "max_area", "max_bins",
    ],
    "Invasion Assay": [
        "outside_channel", "total_channel",
        "intensity_statistic", "background_correction",
        "outside_threshold_method", "outside_threshold", "control_wells",
        "control_quantile", "min_control_objects", "min_objects_for_threshold",
        "min_objects_for_bimodality", "bimodality_cutoff",
        "threshold_agreement_tolerance", "threshold_sensitivity",
        "inflation_warn", "min_parasites_per_well",
        "min_parasite_area", "max_parasite_area", "min_total_intensity",
        "extracellular_class",
        "seed_wells_from_cells",
        "qc_plot_max_panels",
    ],

    # Rarely-touched knobs only. 'normalize' left for "General" and
    # nuclei_limit / pathogen_limit for "Measurements": all three change what
    # the run produces rather than how it is tuned, and hiding them here is
    # what put them at the bottom of the Classify (CV) dataset settings.
    "Advanced": ["resume", "strict_errors", "max_failure_rate", "crop_source", "queue_by_uncertainty", "queue_measure", "queue_diversity", "queue_limit", "dry_run", "verbose", "n_jobs", "batch_size", "test_images", "random_test", "test_nr", "preprocess", "masks", "remove_background", "background", "backgrounds", "lower_percentile", "randomize", "batch_fields", "pipeline_style", "keep_intermediate", "keep_original_images", "save_original_images", "keep_npz", "compression", "diameter_estimate_n_fields", "shuffle", "save", "filter", "merge_pathogens", "all_to_mip", "upscale", "upscale_factor", "consolidate", "use_sam_pathogen", "use_sam_nucleus", "use_sam_cell", "denoise"],

    # Experimental volumetric controls are deliberately split by dimensional
    # contract. `z_axis` lives with 3D because 4D builds on the same z plan;
    # the 4D panel contains only time-axis and inter-frame tracking controls.
    "3D Settings (Beta)": [
        "z_stack", "z_segmentation_mode", "z_axis", "z_projection",
        "anisotropy", "voxel_size_z_um", "voxel_size_xy_um",
        "stitch_threshold",
    ],
    "4D Settings (Beta)": [
        "t_stack", "t_axis_order", "t_axis", "frame_interval_s",
        "t_track_backend", "t_link_threshold", "t_max_displacement_px",
        "t_max_displacement_um", "t_project_for_tracking",
    ],

    "Motility (beta)": motility_settings,
    "Motility Advanced (beta)": motility_advanced_settings,
}

category_dependencies = {
    'timelapse': ['Timelapse'],
    'motility_analysis': ['Motility (beta)', 'Motility Advanced (beta)'],
}

category_group_dependencies = {
    'Merge split objects': ['postprocess_cell_masks', 'postprocess_nucleus_masks', 'postprocess_pathogen_masks', 'postprocess_organelle_masks'],
}

category_integer_dependencies = {
    ('cell_channel', 'cell_mask_dim'): ['Cell'],
    ('nucleus_channel', 'nucleus_mask_dim'): ['Nucleus'],
    ('pathogen_channel', 'pathogen_mask_dim'): ['Pathogen'],
    ('organelle_channel', 'organelle_mask_dim'): ['Organelle'],
}

# Categories shown only when a setting equals a specific value.
#
# gui_core._get_visible_categories blocks the categories of every option that
# does NOT match the current value, so a category listed under two or more
# options can never be shown. The eight per-method organelle headings are now a
# single "Organelle" category (ordered by method instead of gated on it), so
# organelle_method no longer gates anything and its map is empty. The mechanism
# itself is still wired up in both GUIs for the next setting that needs it.
category_value_dependencies = {
    'organelle_method': {},
}

category_keys = list(categories.keys())

def check_settings(vars_dict, expected_types, q=None):
    """Validate and coerce GUI-collected settings against expected types.

    Iterates the widget map produced by the settings panel, parses each raw
    string value into the type declared in ``expected_types`` (including
    tuple-typed "or None" fields, lists, dicts and lists-of-lists), and
    collects human-readable error messages instead of stopping at the first
    failure. Errors are also forwarded to ``q`` for GUI display.

    :param vars_dict: mapping ``key -> (label, widget, var, frame)`` from the settings panel.
    :param expected_types: mapping ``key -> type`` (or tuple of accepted types).
    :param q: optional queue used to surface error strings to the GUI. A private
        Queue is created if None.
    :returns: tuple ``(settings, errors)`` where ``settings`` is the parsed dict
        and ``errors`` is the list of collected error messages.
    """
    from .gui_utils import parse_list

    if q is None:
        from multiprocessing import Queue
        q = Queue()

    settings = {}
    errors = []  # Collect errors instead of stopping at the first one

    for key, (label, widget, var, _) in vars_dict.items():
        if key not in expected_types and key not in category_keys:
            errors.append(f"Warning: Key '{key}' not found in expected types.")
            continue

        value = var.get()
        if value in ['None', '']:
            value = None

        expected_type = expected_types.get(key, str)
        
        try:
            if key in ["cell_plate_metadata", "timelapse_frame_limits", "png_size", "png_dims", "pathogen_plate_metadata", "treatment_plate_metadata", "timelapse_objects", "class_metadata", "crop_mode", "dialate_png_ratios"]:
                if value is None:
                    parsed_value = None
                else:
                    try:
                        parsed_value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Expected a list or list of lists but got an invalid format: {value}")

                if isinstance(parsed_value, list):
                    if all(isinstance(i, list) for i in parsed_value) or all(not isinstance(i, list) for i in parsed_value):
                        settings[key] = parsed_value
                    else:
                        raise ValueError(f"Invalid format: '{key}' contains mixed types (single values and lists).")

                else:
                    raise ValueError(f"Expected a list for '{key}', but got {type(parsed_value).__name__}.")
            
            elif expected_type == list:
                settings[key] = parse_list(value) if value else None

                if isinstance(settings[key], list) and len(settings[key]) == 1:
                    settings[key] = settings[key][0]

            elif expected_type == bool:
                settings[key] = value.lower() in ['true', '1', 't', 'y', 'yes'] if isinstance(value, str) else bool(value)
            
            elif expected_type == (int, type(None)):
                if value is None or str(value).isdigit():
                    settings[key] = int(value) if value is not None else None
                else:
                    raise ValueError(f"Expected an integer or None for '{key}', but got '{value}'.")

            elif expected_type == (float, type(None)):
                if value is None or (isinstance(value, str) and value.replace(".", "", 1).isdigit()):
                    settings[key] = float(value) if value is not None else None
                else:
                    raise ValueError(f"Expected a float or None for '{key}', but got '{value}'.")

            elif expected_type == (int, float):
                try:
                    settings[key] = float(value) if '.' in str(value) else int(value)
                except ValueError:
                    raise ValueError(f"Expected an integer or float for '{key}', but got '{value}'.")

            elif expected_type == (bool, int):
                # invert_dependent_variable: False/0 = as measured, True/1 =
                # 1 - x, -1 = 1 / x. The generic tuple branch at the bottom
                # would reach bool('False') first, which is True, and silently
                # invert every score in the screen.
                if value is None:
                    settings[key] = None
                else:
                    text = str(value).strip().lower()
                    if text in ('true', 't', 'y', 'yes'):
                        settings[key] = True
                    elif text in ('false', 'f', 'n', 'no'):
                        settings[key] = False
                    else:
                        try:
                            settings[key] = int(text)
                        except ValueError:
                            raise ValueError(
                                f"Expected True, False or an integer for '{key}', but got '{value}'.")

            elif expected_type == (list, type(None)):
                # y_lims / x_lim / control_wells / filter_min_max. The generic
                # tuple branch would reach list('[0, 5]') first and hand the
                # pipeline ['[', '0', ',', ' ', '5', ']']. literal_eval also
                # keeps the nested form y_lims uses for a broken axis, which
                # parse_list rejects as "mixed types".
                if value is None:
                    settings[key] = None
                else:
                    try:
                        parsed_value = ast.literal_eval(value) if isinstance(value, str) else value
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Expected a list or None for '{key}', but got: {value}")
                    if isinstance(parsed_value, tuple):
                        parsed_value = list(parsed_value)
                    if not isinstance(parsed_value, list):
                        raise ValueError(
                            f"Expected a list or None for '{key}', but got "
                            f"{type(parsed_value).__name__}.")
                    settings[key] = parsed_value

            elif expected_type == (str, type(None)):
                settings[key] = str(value) if value is not None else None

            elif expected_type == (str, type(None), list):
                if isinstance(value, list):
                    settings[key] = parse_list(value) if value else None
                elif isinstance(value, str):
                    settings[key] = str(value)
                else:
                    settings[key] = None
            
            elif expected_type == dict:
                try:
                    if isinstance(value, str):
                        parsed_dict = ast.literal_eval(value)
                    else:
                        raise ValueError("Expected a string representation of a dictionary.")

                    if not isinstance(parsed_dict, dict):
                        raise ValueError(f"Expected a dictionary for '{key}', but got {type(parsed_dict).__name__}.")

                    settings[key] = parsed_dict
                except (ValueError, SyntaxError) as e:
                    settings[key] = {}
                    errors.append(f"Error: Invalid dictionary format for '{key}'. Expected type: dict. Error: {e}")

            elif isinstance(expected_type, tuple):
                for typ in expected_type:
                    try:
                        settings[key] = typ(value) if value else None
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    raise ValueError(f"Value '{value}' for '{key}' does not match any expected types: {expected_type}.")

            else:
                try:
                    settings[key] = expected_type(value) if value else None
                except (ValueError, TypeError):
                    raise ValueError(f"Expected type {expected_type.__name__} for '{key}', but got '{value}'.")

        except (ValueError, SyntaxError) as e:
            print(f"Processing key: '{key}' with value: '{value}' and expected type: {expected_type}")
            expected_type_name = ' or '.join([t.__name__ for t in expected_type]) if isinstance(expected_type, tuple) else expected_type.__name__
            errors.append(f"Error: '{key}' has invalid format. Expected type: {expected_type_name}. Got value: '{value}'. Error: {e}")

    # Send all collected errors to the queue
    for error in errors:
        q.put(error)
        
    return settings, errors

def generate_fields_lazy(variables, scrollable_frame, tick_callback=None):
    """Build input widgets for the always-visible settings only.

    Categorized settings are recorded as placeholders and materialised on
    demand when their category is expanded — keeps initial GUI startup fast.

    :param variables: mapping ``key -> (var_type, options, default_value)``.
    :param scrollable_frame: parent scrollable frame that hosts the widgets.
    :param tick_callback: optional callable invoked after each field is added
        (e.g. to advance a progress bar).
    :returns: ``vars_dict`` mapping ``key -> (label, widget, var, frame)`` for
        rendered fields and ``None`` for lazy placeholders.
    """
    from .gui_utils import create_input_field
    from .gui_elements import spacrToolTip
    
    row = 1
    vars_dict = {}
    
    # Collect all settings that belong to a category
    categorized_keys = set()
    for cat_name, cat_keys in categories.items():
        categorized_keys.update(cat_keys)
    
    # Only create widgets for non-categorized (always-visible) settings
    for key, (var_type, options, default_value) in variables.items():
        if key in categorized_keys:
            # Store the definition but don't create widgets yet
            vars_dict[key] = None  # placeholder
            continue
        
        try:
            label, widget, var, frame = create_input_field(
                scrollable_frame.scrollable_frame, key, row, var_type, options, default_value)
        except Exception:
            print(f"Warning: Invalid value for {key}, reverting to {default_value}")
            type_defaults = {'check': False, 'entry': '', 'combo': options[0] if options else '', 'int': 0, 'float': 0.0}
            fallback = type_defaults.get(var_type, '')
            try:
                label, widget, var, frame = create_input_field(
                    scrollable_frame.scrollable_frame, key, row, var_type, options, fallback)
            except Exception:
                print(f"Error: Could not create field for '{key}'. Skipping.")
                continue

        vars_dict[key] = (label, widget, var, frame)
        
        if key in tooltips:
            spacrToolTip(label, tooltips[key])
        row += 1
        
        if tick_callback:
            tick_callback()

    # Store variables and row counter for lazy creation
    scrollable_frame._field_variables = variables
    scrollable_frame._next_row = row
    
    scrollable_frame.scrollable_frame.update_idletasks()
    return vars_dict

def generate_fields(variables, scrollable_frame, tick_callback=None):
    """Build input widgets for every setting eagerly.

    Falls back to a type-appropriate default when a supplied ``default_value``
    is rejected by the widget factory, and skips the field if that also fails.

    :param variables: mapping ``key -> (var_type, options, default_value)``.
    :param scrollable_frame: parent scrollable frame that hosts the widgets.
    :param tick_callback: optional callable invoked after each field is added.
    :returns: ``vars_dict`` mapping ``key -> (label, widget, var, frame)``.
    """
    from .gui_utils import create_input_field
    from .gui_elements import spacrToolTip
    row = 1
    vars_dict = {}
    
    for key, (var_type, options, default_value) in variables.items():
        try:
            label, widget, var, frame = create_input_field(scrollable_frame.scrollable_frame, key, row, var_type, options, default_value)
        except Exception:
            print(f"Warning: Invalid value for {key}, reverting to {default_value}, var_type: {var_type}({default_value}).")
            type_defaults = {
                'check': False,
                'entry': '',
                'combo': options[0] if options else '',
                'int': 0,
                'float': 0.0,
            }
            fallback = type_defaults.get(var_type, '')
            try:
                label, widget, var, frame = create_input_field(scrollable_frame.scrollable_frame, key, row, var_type, options, fallback)
            except Exception:
                print(f"Error: Could not create field for '{key}' even with fallback. Skipping.")
                continue

        vars_dict[key] = (label, widget, var, frame)
        
        if key in tooltips:
            spacrToolTip(label, tooltips[key])

        row += 1
        
        if tick_callback:
            tick_callback()

    scrollable_frame.scrollable_frame.update_idletasks()
    
    return vars_dict

descriptions = {
    'mask': "\n\nHelp:\n- Generate Cells, Nuclei, Pathogens, and Cytoplasm masks from intensity images in src.\n- To ensure that spacr is installed correctly:\n- 1. Download the training set (click Download).\n- 2. Import settings (click settings navigate to downloaded dataset settings folder and import preprocess_generate_masks_settings.csv).\n- 3. Run the module.\n- 4. Proceed to the Measure module (click Measure in the menu bar).\n- For further help, click the Help button in the menu bar.",
    
    'measure': "Capture Measurements from Cells, Nuclei, Pathogens, and Cytoplasm objects. Generate single object PNG images for one or several objects. (Requires masks from the Mask module). Function: measure_crop from spacr.measure.\n\nKey Features:\n- Comprehensive Measurement Capture: Obtain detailed measurements for various cellular components, including area, perimeter, intensity, and more.\n- Image Generation: Create high-resolution PNG images of individual objects, facilitating further analysis and visualization.\n- Mask Dependency: Requires accurate masks generated by the Mask module to ensure precise measurements.",
    
    'classify': "Train and Test any Torch Computer vision model. (Requires PNG images from the Measure module). Function: train_test_model from spacr.deep_spacr.\n\nKey Features:\n- Deep Learning Integration: Train and evaluate state-of-the-art Torch models for various classification tasks.\n- Flexible Training: Supports a wide range of Torch models, allowing customization based on specific research needs.\n- Data Requirement: Requires PNG images generated by the Measure module for training and testing.",
    
    'umap': "Generate UMAP or tSNE embeddings and represent points as single cell images. (Requires measurements.db and PNG images from the Measure module). Function: generate_image_umap from spacr.core.\n\nKey Features:\n- Dimensionality Reduction: Employ UMAP or tSNE algorithms to reduce high-dimensional data into two dimensions for visualization.\n- Single Cell Representation: Visualize embedding points as single cell images, providing an intuitive understanding of data clusters.\n- Data Integration: Requires measurements and images generated by the Measure module, ensuring comprehensive data representation.",
    
    'train_cellpose': "Train custom Cellpose models for your specific dataset. Function: train_cellpose_model from spacr.core.\n\nKey Features:\n- Custom Model Training: Train Cellpose models on your dataset to improve segmentation accuracy.\n- Data Adaptation: Tailor the model to handle specific types of biological samples more effectively.\n- Advanced Training Options: Supports various training parameters and configurations for optimized performance.",
    
    'ml_analyze': "Perform machine learning analysis on your data. Function: ml_analysis_tools from spacr.ml.\n\nKey Features:\n- Comprehensive Analysis: Utilize a suite of machine learning tools for data analysis.\n- Customizable Workflows: Configure and run different ML algorithms based on your research requirements.\n- Integration: Works seamlessly with other modules to analyze data produced from various steps.",
    
    'cellpose_masks': "Generate masks using Cellpose for all images in your dataset. Function: generate_masks from spacr.cellpose.\n\nKey Features:\n- Batch Processing: Generate masks for large sets of images efficiently.\n- Robust Segmentation: Leverage Cellpose's capabilities for accurate segmentation across diverse samples.\n- Automation: Automate the mask generation process for streamlined workflows.",
    
    'cellpose_all': "Run Cellpose on all images in your dataset and obtain masks and measurements. Function: cellpose_analysis from spacr.cellpose.\n\nKey Features:\n- End-to-End Analysis: Perform both segmentation and measurement extraction in a single step.\n- Efficiency: Process entire datasets with minimal manual intervention.\n- Comprehensive Output: Obtain detailed masks and corresponding measurements for further analysis.",
    
    'map_barcodes': "\n\nHelp:\n- 1 .Generate consensus read fastq files from R1 and R2 files.\n- 2. Map barcodes from sequencing data for identification and tracking of samples.\n- 3. Run the module to extract and map barcodes from your FASTQ files in chunks.\n- Prepare your barcode CSV files with the appropriate 'name' and 'sequence' columns.\n- Configure the barcode settings (coordinates and reverse complement flags) according to your experimental setup.\n- For further help, click the Help button in the menu bar.",

    'regression': "Perform regression analysis on your data. Function: regression_tools from spacr.analysis.\n\nKey Features:\n- Statistical Analysis: Conduct various types of regression analysis to identify relationships within your data.\n- Flexible Options: Supports multiple regression models and configurations.\n- Data Insight: Gain deeper insights into your dataset through advanced regression techniques.",
    
    'activation': "",

    'analyze_plaques': "Analyze plaque images to quantify plaque properties. Function: analyze_plaques from spacr.analysis.\n\nKey Features:\n- Plaque Analysis: Quantify plaque properties such as size, intensity, and shape.\n- Batch Processing: Analyze multiple plaque images efficiently.\n- Visualization: Generate visualizations to represent plaque data and patterns.",

    'recruitment': "Analyze recruitment data to understand sample recruitment dynamics. Function: recruitment_analysis_tools from spacr.analysis.\n\nKey Features:\n- Recruitment Analysis: Investigate and analyze the recruitment of samples over time or conditions.\n- Visualization: Generate visualizations to represent recruitment trends and patterns.\n- Integration: Utilize data from various sources for a comprehensive recruitment analysis."
}

def set_annotate_default_settings(settings):
    """Populate default settings for the image annotation UI.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('image_type', 'cell_png')
    settings.setdefault('channels', "r,g,b")
    settings.setdefault('img_size', 200)
    settings.setdefault('annotation_column', 'test')
    settings.setdefault('normalize_channels', None)
    settings.setdefault('outline', None)
    settings.setdefault('outline_threshold_factor', 1.25)
    settings.setdefault('outline_sigma', 4)
    settings.setdefault('edge_thickness', 0.1)
    settings.setdefault('edge_transparency', 100)
    settings.setdefault('edge_image', 'False')
    settings.setdefault('object_size', (0,0))
    settings.setdefault('percentiles', [2, 98])
    settings.setdefault('measurement', '') #'cytoplasm_channel_3_mean_intensity,pathogen_channel_3_mean_intensity')
    settings.setdefault('threshold', '') #'2')
    settings.setdefault('threshold_direction', 'higher')
    # 'auto' uses the PNG crop folder when one exists and falls back to
    # cutting crops out of merged/*.npy on demand; 'png' and 'merged'
    # force one source. See spacr.crops.resolve_crop_source.
    settings.setdefault('crop_source', 'auto')
    # Active-learning queue (spacr.active_learning). Off by default: it
    # needs model scores in png_list, which only exist after Classify (CV).
    settings.setdefault('queue_by_uncertainty', False)
    settings.setdefault('queue_measure', 'entropy')
    settings.setdefault('queue_diversity', 'well')
    settings.setdefault('queue_limit', 0)
    return settings

def set_default_generate_barecode_mapping(settings=None):
    """Return default settings for the barcode-mapping pipeline.

    :param settings: optional dict to fill in place; a new dict is created if None.
    :returns: the settings dict with defaults applied.
    """
    if settings is None:
        settings = {}
    settings.setdefault('src', 'path')
    # Group names MUST be columnID / rowID (not column / row): the read
    # processors in sequencing.py read match.group('columnID') /
    # match.group('rowID'), so a default regex naming them column/row raised
    # "IndexError: no such group" — the shipped default was unusable.
    settings.setdefault('regex', DEFAULT_BARCODE_REGEX)
    settings.setdefault('target_sequence', 'TGCTGTTTCCAGCATAGCTCTTAAAC')
    settings.setdefault('offset_start', -8)
    settings.setdefault('expected_end', 89)
    settings.setdefault('column_csv', bundled_barcode_path('column'))
    settings.setdefault('grna_csv', bundled_barcode_path('grna'))
    settings.setdefault('row_csv', bundled_barcode_path('row'))
    settings.setdefault('save_h5', True)
    settings.setdefault('comp_type', 'zlib')
    settings.setdefault('comp_level', 5)
    settings.setdefault('chunk_size', 100000)
    settings.setdefault('n_jobs', None)
    settings.setdefault('mode', 'paired')
    settings.setdefault('single_direction', 'R1')
    settings.setdefault('test', False)
    settings.setdefault('fill_na', False)
    return settings

def get_default_generate_activation_map_settings(settings):
    """Populate default settings for generating model activation/CAM maps.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('dataset', 'path')
    settings.setdefault('model_type', 'maxvit')
    settings.setdefault('model_path', 'path')
    settings.setdefault('image_size', 224)
    settings.setdefault('batch_size', 64)
    settings.setdefault('normalize', True)
    settings.setdefault('cam_type', 'gradcam')
    settings.setdefault('target_layer', None)
    settings.setdefault('plot', False)
    settings.setdefault('save', True)
    settings.setdefault('normalize_input', True)
    settings.setdefault('channels', [1,2,3])
    settings.setdefault('overlay', True)
    settings.setdefault('shuffle', True)
    settings.setdefault('correlation', True)
    settings.setdefault('manders_thresholds', [15,50, 75])
    settings.setdefault('n_jobs', None)
    # Attribution methods and their analyses (spacr.attribution). The
    # sanity check is on by default because a map that ignores the model's
    # weights is an edge detector, not an explanation.
    settings.setdefault('smoothgrad_samples', 0)
    settings.setdefault('smoothgrad_sigma', 0.15)
    settings.setdefault('occlusion_window', 8)
    settings.setdefault('occlusion_stride', 4)
    settings.setdefault('ig_steps', 50)
    settings.setdefault('ig_baseline', 'zero')
    settings.setdefault('attribution_steps', 12)
    settings.setdefault('attribution_baseline', 'blur')
    settings.setdefault('sanity_check', True)
    settings.setdefault('object_type', 'cell')
    return settings

def get_analyze_plaque_settings(settings):
    """Populate default settings for plaque analysis.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('masks', True)
    settings.setdefault('background', 200)
    settings.setdefault('Signal_to_noise', 10)
    settings.setdefault('CP_prob', 0)
    settings.setdefault('diameter', 30)
    settings.setdefault('batch_size', 50)
    settings.setdefault('flow_threshold', 0.4)
    settings.setdefault('save', True)
    settings.setdefault('verbose', True)
    settings.setdefault('resize', True)
    settings.setdefault('target_height', 1120)
    settings.setdefault('target_width', 1120)
    settings.setdefault('rescale', False)
    settings.setdefault('resample', False)
    settings.setdefault('fill_in', True)
    return settings

def set_graph_importance_defaults(settings):
    """Populate default settings for the "graph importance" plot utility.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('csvs','list of paths')
    settings.setdefault('grouping_column','compartment')
    settings.setdefault('data_column','compartment_importance_sum')
    settings.setdefault('graph_type','jitter_bar')
    settings.setdefault('save',False)
    return settings

def set_interpret_vision_model_defaults(settings):
    """Populate default settings for interpreting vision-model predictions.

    Covers feature importance, permutation importance, and SHAP explanation
    options over the cell/nucleus/pathogen/cytoplasm tables.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('scores','path')
    settings.setdefault('tables',['cell', 'nucleus', 'pathogen','cytoplasm'])
    settings.setdefault('feature_importance',True)
    settings.setdefault('permutation_importance',False)
    settings.setdefault('shap',True)
    settings.setdefault('save',False)
    settings.setdefault('nuclei_limit',1000)
    settings.setdefault('pathogen_limit',1000)
    settings.setdefault('top_features',30)
    settings.setdefault('shap_sample',True)
    settings.setdefault('n_jobs',-1)
    settings.setdefault('shap_approximate',True)
    settings.setdefault('score_column','cv_predictions')
    return settings


# Backward compatibility for the misspelling published in earlier releases.
set_interperate_vision_model_defaults = set_interpret_vision_model_defaults


def set_analyze_invasion_defaults(settings):
    """Populate default settings for the two-colour (red/green) invasion assay.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('parasite_table','pathogen')
    settings.setdefault('compartment','pathogen')
    settings.setdefault('outside_channel',1)
    settings.setdefault('total_channel',0)
    settings.setdefault('intensity_statistic','auto')
    settings.setdefault('background_correction','none')
    settings.setdefault('outside_threshold_method','otsu')
    settings.setdefault('outside_threshold',None)
    settings.setdefault('control_wells',None)
    settings.setdefault('control_quantile',0.99)
    settings.setdefault('min_control_objects',10)
    settings.setdefault('min_objects_for_threshold',10)
    settings.setdefault('min_objects_for_bimodality',30)
    settings.setdefault('bimodality_cutoff',0.5555555555555556)
    settings.setdefault('threshold_agreement_tolerance',0.5)
    settings.setdefault('threshold_sensitivity',0.25)
    settings.setdefault('inflation_warn',0.05)
    settings.setdefault('min_parasites_per_well',50)
    settings.setdefault('min_parasite_area',0)
    settings.setdefault('max_parasite_area',None)
    settings.setdefault('min_total_intensity',None)
    settings.setdefault('extracellular_class','attached')
    settings.setdefault('seed_wells_from_cells',True)
    settings.setdefault('cell_types',['Hela'])
    settings.setdefault('cell_plate_metadata',None)
    settings.setdefault('pathogen_types',['nc', 'pc'])
    settings.setdefault('pathogen_plate_metadata',[['c1'], ['c2']])
    settings.setdefault('treatments',None)
    settings.setdefault('treatment_plate_metadata',None)
    settings.setdefault('group_column','condition')
    settings.setdefault('level','object')
    settings.setdefault('change_plate',False)
    settings.setdefault('qc_plot_max_panels',12)
    settings.setdefault('cmap','viridis')
    settings.setdefault('save',True)
    settings.setdefault('verbose',False)
    return settings

def set_analyze_replication_defaults(settings):
    """Populate defaults for the parasites-per-vacuole replication assay.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src', 'path')
    settings.setdefault('parasite_table', 'pathogen')
    settings.setdefault('compartment', 'pathogen')
    settings.setdefault('vacuole_key', 'auto')
    settings.setdefault('vacuole_link_distance', None)
    settings.setdefault('vacuole_link_factor', 1.5)
    settings.setdefault('parasite_count_column', None)
    settings.setdefault('min_parasite_area', 0)
    settings.setdefault('max_parasite_area', None)
    settings.setdefault('max_parasites_per_vacuole', 16)
    settings.setdefault('require_host_cell', True)
    settings.setdefault('seed_wells_from_cells', True)
    settings.setdefault('non_power_of_two_warn', 0.2)
    settings.setdefault('cell_types', ['Hela'])
    settings.setdefault('cell_plate_metadata', None)
    settings.setdefault('pathogen_types', ['nc', 'pc'])
    settings.setdefault('pathogen_plate_metadata', [['c1'], ['c2']])
    settings.setdefault('treatments', None)
    settings.setdefault('treatment_plate_metadata', None)
    settings.setdefault('group_column', 'condition')
    settings.setdefault('level', 'object')
    settings.setdefault('change_plate', False)
    settings.setdefault('cmap', 'viridis')
    settings.setdefault('save', True)
    settings.setdefault('verbose', False)
    return settings

def set_analyze_endodyogeny_defaults(settings):
    """Populate default settings for endodyogeny (parasite division) analysis.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('tables',['cell', 'nucleus', 'pathogen', 'cytoplasm'])
    settings.setdefault('cell_types',['Hela'])
    settings.setdefault('cell_plate_metadata',None)
    settings.setdefault('pathogen_types',['nc', 'pc'])
    settings.setdefault('pathogen_plate_metadata',[['c1'], ['c2']])
    settings.setdefault('treatments',None)
    settings.setdefault('treatment_plate_metadata',None)
    settings.setdefault('min_area_bin',500)
    settings.setdefault('max_area',1000000000)
    settings.setdefault('group_column','condition')
    settings.setdefault('compartment','pathogen')
    settings.setdefault('pathogen_limit',1)
    settings.setdefault('nuclei_limit',10)
    settings.setdefault('level','object')
    settings.setdefault('um_per_px',0.1)
    settings.setdefault('max_bins',None)
    settings.setdefault('save',False)
    settings.setdefault('change_plate',False)
    settings.setdefault('cmap','viridis')
    settings.setdefault('verbose',False)
    
    settings.setdefault('group_by_class',False)
    settings.setdefault('class_column','predictions')
    
    return settings

def set_analyze_class_proportion_defaults(settings):
    """Populate default settings for class-proportion analysis across conditions.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('tables',['cell', 'nucleus', 'pathogen', 'cytoplasm'])
    settings.setdefault('cell_types',['Hela'])
    settings.setdefault('cell_plate_metadata',None)
    settings.setdefault('pathogen_types',['nc','pc'])
    settings.setdefault('pathogen_plate_metadata',[['c1'],['c2']])
    settings.setdefault('treatments',None)
    settings.setdefault('treatment_plate_metadata',None)
    settings.setdefault('group_column','condition')
    settings.setdefault('class_column','test')
    settings.setdefault('pathogen_limit',1000)
    settings.setdefault('nuclei_limit',1000)
    settings.setdefault('level','well')
    settings.setdefault('save',False)
    settings.setdefault('verbose', False)
    return settings

def get_plot_data_from_csv_default_settings(settings):
    """Populate default settings for plotting data pulled from a CSV file.

    :param settings: dict to fill in place.
    :returns: the settings dict with defaults applied.
    """
    settings.setdefault('src','path')
    settings.setdefault('data_column','choose column')
    settings.setdefault('grouping_column','choose column')
    settings.setdefault('graph_type','violin')
    settings.setdefault('save',False)
    settings.setdefault('y_lim',None)
    settings.setdefault('log_y',False)
    settings.setdefault('log_x',False)
    settings.setdefault('keep_groups',None)
    settings.setdefault('representation','well')
    settings.setdefault('theme','dark')
    settings.setdefault('remove_outliers',False)
    settings.setdefault('verbose',False)
    return settings

def set_default_stitch(settings=None):
    """Return default settings for the tile-stitching pipeline.

    Covers feature detection, RANSAC, outline overlay, feature cache and
    per-well mosaic output parameters.

    :param settings: optional dict to fill in place; a new dict is created if None.
    :returns: the settings dict with defaults applied (a shallow copy of the input).
    """
    settings = {} if settings is None else dict(settings)
    settings.setdefault('detector', 'ORB')
    settings.setdefault('nfeatures', 8000)
    settings.setdefault('max_keypoints', 4000)
    settings.setdefault('downsample', 0.5)
    settings.setdefault('ransac_thresh_px', 3.0)
    settings.setdefault('allow_scale', False)
    settings.setdefault('allow_rotation', False)
    settings.setdefault('score_threshold', 0.001)
    settings.setdefault('all_scores', False)
    settings.setdefault('outline_source', 'otsu')
    settings.setdefault('save_qc', True)
    settings.setdefault('save_stitched_default', False)
    settings.setdefault('canny', (40, 120))
    settings.setdefault('blur_sigma', 0.0)
    settings.setdefault('dilate_ksize', 0)
    settings.setdefault('line_thickness', 1)
    settings.setdefault('outline_alpha', 1.0)
    settings.setdefault('feature_cache_mode', 'disk')
    settings.setdefault('feature_cache_dir', None)  # set per well by caller
    settings.setdefault('max_ram_features', 256)
    settings.setdefault('n_workers_features', None)
    settings.setdefault('pair_batch_size', 8192)
    settings.setdefault('stream_csv', True)
    settings.setdefault('opencv_threads', 1)
    settings.setdefault('arr_axes', 'AUTO')
    settings.setdefault('mip', True)
    settings.setdefault('z_index', 0)
    settings.setdefault('t_index', 0)
    settings.setdefault('squeeze_singleton', True)

    # run_folder settings
    settings.setdefault('n_workers', max(1, (os.cpu_count() or 8) // 2))
    settings.setdefault('max_site_gap', 64)
    settings.setdefault('mosaic_min_score', None)   # None => auto elbow
    # per-well outputs are set by caller:
    settings.setdefault('mosaic_out', None)
    settings.setdefault('mosaic_csv_out', None)
    return settings

def set_default_multichannel(settings=None):
    """Return default settings for building multichannel per-well mosaics.

    :param settings: optional dict to fill in place; a new dict is created if None.
    :returns: the settings dict with defaults applied (a shallow copy of the input).
    """
    settings = {} if settings is None else dict(settings)
    settings.setdefault('channel_indices', None)   # infer from first tile if None
    settings.setdefault('blend', 'max')            # {'max','overwrite'}
    settings.setdefault('preview_downsample', 8)
    settings.setdefault('tmp_dir', None)           # set per well by caller
    settings.setdefault('out_tif', None)           # set per well by caller
    settings.setdefault('out_png', None)           # set per well by caller
    return settings

def set_default_general(settings=None):
    """Return default settings for the general organize/stitch/multichannel run.

    :param settings: optional dict to fill in place; a new dict is created if None.
    :returns: the settings dict with defaults applied (a shallow copy of the input).
    """
    settings = {} if settings is None else dict(settings)
    settings.setdefault('src', '/path/to/src')
    settings.setdefault('dst_root', settings.get('src'))
    settings.setdefault('meta_regex', r'(?P<mag>\d+X)_c(?P<chan>\d+)_?(?P<well>[A-H]\d{1,2}).*?Site[-_](?P<site>\d+)\.(?:tif|tiff)$')
    settings.setdefault('well_group', 'well')
    settings.setdefault('exts', ['.tif', '.tiff', '.png'])
    settings.setdefault('recursive', True)
    settings.setdefault('collision', 'rename')     # {'rename','skip','overwrite'}
    settings.setdefault('on_missing', 'error')     # {'error','skip'}
    settings.setdefault('dry_run', False)
    settings.setdefault('verbose', True)
    settings.setdefault('do_organize', True)
    settings.setdefault('do_nuc_stitch', True)
    settings.setdefault('do_multichannel', True)
    settings.setdefault('channel_index', 0)        # nuclei channel in each tile
    return settings

def get_automated_motility_assay_default_settings(settings):
    """Return default settings for the automated motility assay pipeline.

    Combines array/filter parameters, XGBoost infection classifier settings,
    and PCA/UMAP/t-SNE embedding options into a single settings dict.

    :param settings: optional dict to fill in place; a new dict is created if None.
    :returns: the settings dict with defaults applied.
    """
    if settings is None:
        settings = {}

    # array settings
    # `src` is the plate folder holding merged/*.npy. It used to be inherited
    # from the mask settings this dict was merged into; the Motility Assay is
    # now a module of its own, so it has to carry its own source folder.
    settings.setdefault('src', 'path')
    settings.setdefault('channels', [0, 1, 2, 3])
    settings.setdefault('cell_channel', 2)
    settings.setdefault('nucleus_channel', 0)
    settings.setdefault('pathogen_channel', 1)
    settings.setdefault('tracked_object', 'cell')
    settings.setdefault('reuse_existing_measurements', True)
    settings.setdefault('infection_intensity_qc_scope', "per_well")
    settings.setdefault('motility_analysis', False)

    # filter settings
    settings.setdefault('n_jobs', 8)
    settings.setdefault('max_displacement', 50.0)
    settings.setdefault('zscore_thresh', 3.0)
    settings.setdefault('straightness_filter', False)
    settings.setdefault('straightness_threshold', 0.95)
    settings.setdefault('infection_intensity_strategy', 'xgboost')  # 'pca' | 'umap' | 'tsne' | 'histogram' | 'xgb'
    settings.setdefault('infection_intensity_mode', "relabel")  # or 'remove'
    settings.setdefault('db_table_name', "timelapse_object_measurements")
    settings.setdefault('infection_intensity_n_bins', 64)
    # Read by _make_intensity_motility_panel; previously undefaulted, so the
    # standalone module had no widget for it. Exposing it lets users skip the
    # QC plotting work on large runs.
    settings.setdefault('infection_intensity_qc_graphs', True)

    # motility plot settings
    settings.setdefault('pixels_per_um', 1.78)
    settings.setdefault('seconds_per_frame', 60)
    settings.setdefault('motility_xlim', (100, -100))
    settings.setdefault('motility_ylim', (100, -100))

    # xgboost settings
    settings.setdefault('infection_xgb_n_estimators', 200)
    settings.setdefault('infection_xgb_max_depth', 3)
    settings.setdefault('infection_xgb_learning_rate', 0.1)
    settings.setdefault('infection_xgb_subsample', 0.8)
    settings.setdefault('infection_xgb_colsample_bytree', 0.8)
    settings.setdefault('infection_xgb_reg_lambda', 1.0)
    settings.setdefault('infection_xgb_random_state', 42)
    settings.setdefault('infection_xgb_n_jobs', -1)
    settings.setdefault('infection_xgb_proba_threshold', 0.5)
    settings.setdefault('infection_xgb_margin', 0.15)
    settings.setdefault('infection_xgb_top_features', 20)
    settings.setdefault('infection_xgb_proba_column', 'infection_xgb_proba')
    settings.setdefault('infection_xgb_drop_ambiguous', True)
    settings.setdefault('infection_xgb_ambiguous_low', 0.25)
    settings.setdefault('infection_xgb_ambiguous_high', 0.75)
    settings.setdefault('infection_xgb_min_cells_per_class', 10)

    # PCA / embedding-common settings
    settings.setdefault('infection_pca_n_clusters', 2)
    settings.setdefault('infection_pca_random_state', 42)
    settings.setdefault('infection_pca_pathogen_weight', 2.0)
    settings.setdefault('infection_pca_log_intensity', False)
    settings.setdefault('infection_pca_max_cells', 50000)
    settings.setdefault('infection_pca_min_gt_separation', 0.2)
    settings.setdefault('infection_pca_min_silhouette', 0.05)

    # UMAP
    settings.setdefault('infection_pca_umap_search', True)
    settings.setdefault('infection_pca_umap_n_neighbors_grid', [5, 10, 15, 30])
    settings.setdefault('infection_pca_umap_min_dist_grid', [0.0, 0.05, 0.1, 0.3])
    # used if infection_pca_umap_search == False
    settings.setdefault('infection_pca_umap_n_neighbors', 15)
    settings.setdefault('infection_pca_umap_min_dist', 0.1)

    # t-SNE
    settings.setdefault('infection_pca_tsne_search', True)
    settings.setdefault('infection_pca_tsne_perplexity_grid', [15.0, 30.0, 45.0])
    settings.setdefault('infection_pca_tsne_learning_rate_grid', [200.0, 500.0])
    # used if infection_pca_tsne_search == False
    settings.setdefault('infection_pca_tsne_perplexity', 30.0)
    
    return settings

def _set_organelle_defaults(settings):
    """Fill in default values for all organelle_* keys."""
    defaults = {
        # General
        'organelle_channel': None,
        'organelle_morphology': 'spots',
        'organelle_method': 'otsu',
        'organelle_diameter': 30,
        'organelle_model_name': 'cpsam',
        'organelle_min_size': 10,
        'organelle_max_size': None,
        'organelle_remove_border': False,

        # Preprocessing
        'organelle_rolling_ball': False,
        'organelle_rolling_ball_radius': 50,
        'organelle_clahe': False,
        'organelle_clahe_clip_limit': 0.01,
        'organelle_mask_within_cells': False,

        # Spots
        'organelle_log_min_sigma': 1,
        'organelle_log_max_sigma': 10,
        'organelle_log_num_sigma': 10,
        'organelle_log_threshold': 0.01,
        'organelle_dog_sigma_low': 1.0,
        'organelle_dog_sigma_high': 3.0,
        'organelle_tophat_radius': 5,
        'organelle_watershed_spots': True,

        # Network
        'organelle_ridge_sigmas': [1, 2, 3],
        'organelle_ridge_filter': 'frangi',
        'organelle_skeletonize': False,
        'organelle_network_threshold': 'otsu',
        'organelle_hysteresis_low': 0.2,
        'organelle_hysteresis_high': 0.6,

        # U-Net
        'organelle_unet_model_path': None,
        'organelle_unet_threshold': 0.5,

        # Irregular
        'organelle_adaptive_block_size': 51,
        'organelle_adaptive_offset': 5,
        'organelle_morph_radius': 3,
        'organelle_fill_holes': 64,

        # Ring
        'organelle_ring_sigma_inner': 1.0,
        'organelle_ring_sigma_outer': 3.0,
        'organelle_ring_min_prominence': 0.1,
        'organelle_ring_fill_method': 'flood',

        # Cellpose
        'organelle_CP_prob': 0.0,
        'organelle_FT': 0.4,
        'organelle_resample': True,
    }
    for key, val in defaults.items():
        settings.setdefault(key, val)
    return settings
