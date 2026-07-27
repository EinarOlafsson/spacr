import os, gc, torch, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:
    # IPython may be mid-init (partially imported by another
    # thread) — use a no-op fallback so importing this module
    # never blocks. spaCR only calls display() from notebook
    # contexts anyway; the Qt GUI ignores it.
    def display(*args, **kwargs):
        pass
import warnings
from scipy import ndimage
from multiprocessing import Value

# Fail-loud accounting: per-folder / per-example failures are recorded on a
# RunLedger and reported in one block at the end of the run, and setup
# failures escalate to ConfigurationError under SPACR_STRICT_ERRORS.
from .errors import RunLedger, ConfigurationError, raise_if_strict

warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")

def preprocess_generate_masks(settings):
    """Turn a folder of raw microscopy images into per-channel Cellpose masks ready for :func:`spacr.measure.measure_crop`.

    Given a source folder ``src`` of multi-channel images, this pipeline (1)
    optionally consolidates inputs from nested folders, (2) renames files to
    the Yokogawa layout used downstream, (3) preprocesses per-channel arrays,
    (4) generates masks for cell / nucleus / pathogen / organelle channels
    via Cellpose (SAM variant), (5) optionally reconciles the cell mask
    against nuclei + pathogen overlays, and (6) writes overlay plots and a
    ``gen_mask_settings.csv`` next to the outputs.

    :param settings: Settings dict; canonicalized via
        :func:`spacr.settings.set_default_settings_preprocess_generate_masks`.
        Must include ``src`` and at least one of ``cell_channel``,
        ``nucleus_channel``, ``pathogen_channel``, or ``organelle_channel``.
        Key entries the function reads:

        - ``src`` (str or list of str) — image folder(s) to process.
        - ``metadata_type`` — ``'cellvoyager'`` or ``'auto'`` (uses
          ``custom_regex`` when set).
        - ``cell_channel`` / ``nucleus_channel`` / ``pathogen_channel`` /
          ``organelle_channel`` — 0-based channel indices; ``None`` skips.
        - ``cell_diameter`` / ``nucleus_diameter`` / ``pathogen_diameter``
          — Cellpose object diameters in pixels.
        - ``pathogen_model`` — removed. Pathogens are segmented with cpsam
          like every other object; the pre-SAM toxo checkpoints are gone.
        - ``consolidate`` — copy nested images into ``src/consolidated``
          before processing.
        - ``preprocess`` / ``masks`` — toggle the two pipeline halves.
        - ``adjust_cells`` — reconcile cell masks against nuclei+pathogen.
        - ``timelapse`` — enable trackpy linking; forces
          ``randomize=False``.
        - ``save``, ``plot``, ``verbose``, ``test_mode``, ``n_jobs``.

    :returns: None. Writes masks, overlays, ``measurements.db`` counts and
        settings CSVs into subfolders of ``src``.
    :raises ValueError: if ``src`` is missing or of the wrong type, or no
        segmentation channel is defined.

    Example:
        .. code-block:: python

            from spacr.core import preprocess_generate_masks
            settings = {
                'src': '/data/plate01',
                'cell_channel': 0, 'nucleus_channel': 1, 'pathogen_channel': 2,
                'cell_diameter': 60, 'nucleus_diameter': 20, 'pathogen_diameter': 8,
                'magnification': 20, 'save': True, 'plot': True,
            }
            preprocess_generate_masks(settings)

    See Also:
        :func:`spacr.io.preprocess_img_data` — the preprocessing half only.
        :func:`spacr.measure.measure_crop` — downstream feature extraction.
    """
    # dry_run comes FIRST, before the local imports below: .object and .io
    # pull in cellpose and the model machinery, which is exactly the cost a
    # validate-only run exists to avoid. Nothing is read, written or loaded
    # past this point when dry_run is set.
    if settings.get('dry_run', False):
        from .validate import run_preflight
        return run_preflight(settings, 'mask')

    #from .timelapse import _summarise_object_relationships
    from .object import generate_cellpose_masks, generate_organelle_masks_sam, generate_cellpose_masks_sam
    from .io import preprocess_img_data, _load_and_concatenate_arrays, convert_to_yokogawa, convert_separate_files_to_yokogawa
    from .plot import plot_image_mask_overlay, plot_arrays
    from .utils import _pivot_counts_table, check_mask_folder, adjust_cell_masks, print_progress, save_settings, delete_intermedeate_files, format_path_for_system, normalize_src_path, generate_image_path_map, copy_images_to_consolidated, merge_split_objects, reset_cellpose_model_reports
    from .settings import set_default_settings_preprocess_generate_masks, _set_organelle_defaults

    # A new run gets to state its model choice again; within one run each
    # notice is printed once per object type rather than once per field.
    reset_cellpose_model_reports()

    # These previously *constructed* a ValueError without raising it (and then
    # returned None), silently swallowing bad input despite the docstring
    # promising a raise. Raise for real.
    if 'src' in settings:
        if not isinstance(settings['src'], (str, list)):
            raise ValueError('src must be a string or a list of strings')
    else:
        raise ValueError('src is a required parameter')

    settings['src'] = normalize_src_path(settings['src'])

    # v2 streaming pipeline — OPT-IN flow that skips the
    # rename/split/npz/npy multi-copy chain and goes straight from
    # originals → merged/stack_<field>.npy with masks appended
    # in-place. Roughly 60-80% less disk than v1 on typical plates.
    # NOTE: v1 remains the DEFAULT — v2 does not yet reproduce v1's
    # channel/stack/mask_stack folder layout (downstream tools + the
    # e2e suite depend on it) and still parses channels 1-indexed on
    # real CellVoyager data. Enable v2 explicitly with
    # pipeline_style='v2'. See spacr.pipeline_v2 for design notes.
    if settings.get('pipeline_style', 'v1') == 'v2':
        from .pipeline_v2 import run_v2
        from ._v1_v2_bridge import (
            v2_channels_from_settings, report_disk_savings,
        )
        srcs = settings['src'] if isinstance(settings['src'], list) \
                                else [settings['src']]
        for src in srcs:
            channels, channel_names = v2_channels_from_settings(settings)
            result = run_v2(
                src,
                channels=channels,
                channel_names=channel_names,
                model_name=settings.get('cell_model_name', 'cpsam'),
                channels_for_cellpose=(0, 0),
                diameter=settings.get('cell_diameter'),
                batch_fields=int(settings.get('batch_fields', 8)),
                metadata_type=settings.get('metadata_type', 'auto'),
                custom_regex=settings.get('custom_regex'),
                keep_npz=bool(settings.get('keep_npz', False)),
            )
            report_disk_savings(src, result['stacks'])
        return
    
    # settings defaults (incl. 'consolidate') are only applied further down,
    # inside the per-source loop; read defensively here so a settings dict
    # without the key doesn't raise KeyError before that point.
    if settings.get('consolidate', False):
        image_map = generate_image_path_map(settings['src'])
        copy_images_to_consolidated(image_map, settings['src'])
        settings['src'] = os.path.join(settings['src'], 'consolidated')

    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]

    if isinstance(settings['src'], list):
        source_folders = settings['src']
        # One ledger for the whole invocation: a run over four plates that
        # only managed three must not report as if it did four.
        ledger = RunLedger('preprocess_generate_masks')
        for source_folder in source_folders:

            print(f'Processing folder: {source_folder}')

            source_folder = format_path_for_system(source_folder)
            settings['src'] = source_folder
            src = source_folder
            settings = set_default_settings_preprocess_generate_masks(settings)

            settings = _set_organelle_defaults(settings)

            if settings['metadata_type'] == 'auto':
                if settings['custom_regex'] != None:
                    try:
                        print(f"using regex: {settings['custom_regex']}")
                        convert_separate_files_to_yokogawa(folder=source_folder, regex=settings['custom_regex'])
                    except Exception:
                        try:
                            convert_to_yokogawa(folder=source_folder)
                        except Exception as e:
                            # Category B: no file was renamed, so every step
                            # below would operate on an empty/unrecognised
                            # folder. Historically this printed and returned
                            # None, which reads exactly like success.
                            print(f"Error: Tried to convert image files and image file name metadata with regex {settings['custom_regex']} then without regex but failed both.")
                            print(f'Error: {e}')
                            ledger.record_failure(source_folder,
                                                  stage='convert_metadata', exc=e)
                            ledger.finalize()
                            raise_if_strict(
                                f"Could not apply Yokogawa naming to {source_folder} "
                                f"with regex {settings['custom_regex']!r} or without "
                                f"one; nothing downstream can run on this folder.",
                                exc=e, settings=settings)
                            return
                else:
                    try:
                        convert_to_yokogawa(folder=source_folder)
                    except Exception as e:
                        print(f"Error: Tried to convert image files and image file name metadata without regex but failed.")
                        print(f'Error: {e}')
                        ledger.record_failure(source_folder,
                                              stage='convert_metadata', exc=e)
                        ledger.finalize()
                        raise_if_strict(
                            f"Could not apply Yokogawa naming to {source_folder}; "
                            f"nothing downstream can run on this folder.",
                            exc=e, settings=settings)
                        return

            if (
                settings['cell_channel'] is None and
                settings['nucleus_channel'] is None and
                settings['pathogen_channel'] is None and
                settings.get('organelle_channel') is None
            ):
                # Category B: with no object channel there is nothing to
                # segment, so returning None here is indistinguishable from
                # a successful run that produced no masks.
                print(f'Error: At least one of cell_channel, nucleus_channel, pathogen_channel or organelle_channel must be defined')
                raise_if_strict(
                    'At least one of cell_channel / nucleus_channel / '
                    'pathogen_channel / organelle_channel must be set; '
                    'no masks can be generated.', settings=settings)
                return
            
            save_settings(settings, name='gen_mask_settings')
            
            # The bundled toxo_pv_lumen / toxo_cyto models were Cellpose-3
            # checkpoints and are gone: Cellpose 4 ships only cpsam, and their
            # CPnet weights cannot load into its Transformer. This guard also
            # never fired — it *constructed* a ValueError without raising it.
            
            if settings['timelapse']:
                settings['randomize'] = False
            
            if settings['preprocess']:
                if not settings['masks']:
                    print(f'WARNING: channels for mask generation are defined when preprocess = True')
            
            if isinstance(settings['save'], bool):
                settings['save'] = [settings['save']]*3

            if settings['verbose']:
                from .utils import pretty_print_settings
                pretty_print_settings(settings, title="Mask Generation Settings")

            if settings['test_mode']:
                print(f'Starting Test mode ...')

            if settings['preprocess']:
                settings, src = preprocess_img_data(settings)

            files_to_process = sum([
                settings['cell_channel'] is not None,
                settings['nucleus_channel'] is not None,
                settings['pathogen_channel'] is not None,
                settings.get('organelle_channel') is not None
            ])
            files_processed = 0

            if settings['masks']:
                mask_src = os.path.join(src, 'masks')
                
                if settings['cell_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'cell_mask_stack'):
                        start = time.time()
                        generate_cellpose_masks_sam(mask_src, settings, 'cell')
                        stop = time.time()
                        duration = (stop - start)
                        time_ls.append(duration)
                        files_processed += 1
                        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'cell_mask_gen')
                    
                if settings['nucleus_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'nucleus_mask_stack'):
                        start = time.time()
                        generate_cellpose_masks_sam(mask_src, settings, 'nucleus')
                        stop = time.time()
                        duration = (stop - start)
                        time_ls.append(duration)
                        files_processed += 1
                        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'nucleus_mask_gen')
                    
                if settings['pathogen_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'pathogen_mask_stack'):
                        start = time.time()
                        generate_cellpose_masks_sam(mask_src, settings, 'pathogen')
                        stop = time.time()
                        duration = (stop - start)
                        time_ls.append(duration)
                        files_processed += 1
                        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'pathogen_mask_gen')
                        
                if settings['organelle_channel'] != None:
                    time_ls=[]
                    if check_mask_folder(src, 'organelle_mask_stack'):
                        start = time.time()
                        generate_organelle_masks_sam(mask_src, settings, 'organelle')
                        stop = time.time()
                        duration = (stop - start)
                        time_ls.append(duration)
                        files_processed += 1
                        print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'organelle_mask_gen')

                if settings['adjust_cells']:
                    if not settings['timelapse']:
                        if settings['pathogen_channel'] != None and settings['cell_channel'] != None and settings['nucleus_channel'] != None:
                            start = time.time()
                            cell_folder = os.path.join(mask_src, 'cell_mask_stack')
                            nuclei_folder = os.path.join(mask_src, 'nucleus_mask_stack')
                            parasite_folder = os.path.join(mask_src, 'pathogen_mask_stack')
                            
                            organelle_folder = None
                            if settings.get('organelle_channel') is not None:
                                candidate = os.path.join(mask_src, 'organelle_mask_stack')
                                if os.path.exists(candidate):
                                    organelle_folder = candidate
                            
                            print(f'Adjusting cell masks with nuclei and pathogen masks')
                            adjust_cell_masks(parasite_folder, cell_folder, nuclei_folder, organelle_folder, overlap_threshold=5, perimeter_threshold=30, n_jobs=settings['n_jobs'])
                            stop = time.time()
                            adjust_time = (stop-start)/60
                            print(f'Cell mask adjustment: {adjust_time} min.')
                            
                if os.path.exists(os.path.join(src,'measurements')):
                    _pivot_counts_table(db_path=os.path.join(src,'measurements', 'measurements.db'))

                # resume (opt-in, default False): skip fields whose merged
                # stack is already present and verified complete, so a crash
                # at field 900 of 1000 does not cost the first 900. Validated
                # rather than stat'ed — see spacr.resume.
                _load_and_concatenate_arrays(
                    src,
                    settings.get('channels'),
                    settings.get('cell_channel'),
                    settings.get('nucleus_channel'),
                    settings.get('pathogen_channel'),
                    settings.get('organelle_channel'),
                    resume=settings.get('resume', False)
                )
                
                if settings['plot']:
                    if not settings['timelapse']:
                        if settings['test_mode'] == True:
                            # Test mode plots every merged field. This used to
                            # take len() of the merged *path string*, i.e. a
                            # number that tracks how deeply the run folder is
                            # nested and has nothing to do with how many
                            # fields exist.
                            merged_dir = os.path.join(src, 'merged')
                            settings['examples_to_plot'] = len(
                                [f for f in os.listdir(merged_dir)
                                 if f.endswith('.npy')]
                            ) if os.path.isdir(merged_dir) else 0

                        # A separate ledger: an overlay PDF that fails to
                        # render is cosmetic and must NOT brand the masks
                        # themselves as partial. It still gets accounted for.
                        plot_ledger = RunLedger('preprocess_generate_masks:overlay_plots')
                        try:
                            merged_src = os.path.join(src,'merged')
                            files = os.listdir(merged_src)
                        except Exception as e:
                            print(f'Failed to plot image mask overly. Error: {e}')
                            plot_ledger.record_failure(os.path.join(src, 'merged'),
                                                       stage='list_merged', exc=e)
                            files = []
                        else:
                            random.shuffle(files)
                        time_ls = []

                        for i, file in enumerate(files):
                            start = time.time()
                            if i+1 <= settings['examples_to_plot']:
                                file_path = os.path.join(merged_src, file)

                                # Per example, not per batch: the old single
                                # try around the whole loop meant one
                                # unplottable field silently cancelled every
                                # remaining example.
                                with plot_ledger.item(
                                        file, stage='plot_mask_overlay',
                                        echo='Failed to plot image mask overly. Error'):
                                    plot_image_mask_overlay(
                                        file_path,
                                        settings['channels'],
                                        settings['cell_channel'],
                                        settings['nucleus_channel'],
                                        settings['pathogen_channel'],
                                        organelle_channel=settings.get('organelle_channel'),
                                        figuresize=10,
                                        percentiles=(1,99),
                                        thickness=3,
                                        save_pdf=True
                                    )
                                    stop = time.time()
                                    duration = stop-start
                                    time_ls.append(duration)
                                    files_processed = i+1
                                    files_to_process = settings['examples_to_plot']
                                    print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type="Plot mask outlines")

                        plot_ledger.finalize()
                    else:
                        plot_arrays(src=os.path.join(src,'merged'), figuresize=settings['figuresize'], cmap=settings['cmap'], nr=settings['examples_to_plot'], normalize=settings['normalize'], q1=1, q2=99)
                    
            torch.cuda.empty_cache()
            gc.collect()
            
            # By default keep only merged/ (masks are embedded there + labels
            # are in the database). keep_intermediate / keep_original_images
            # opt out. The legacy delete_intermediate flag forces cleanup too.
            from .utils import cleanup_pipeline_folders
            keep_intermediate = settings.get('keep_intermediate', False) and not settings.get('delete_intermediate', False)
            keep_original = settings.get('keep_original_images', False) and not settings.get('delete_intermediate', False)
            cleanup_pipeline_folders(src,
                                     keep_intermediate=keep_intermediate,
                                     keep_original=keep_original)

            ledger.record_success(source_folder, stage='preprocess_generate_masks')
            print("Successfully completed run")

        # Last thing on screen: a four-plate run that only completed three
        # says so here, and the per-folder db carries the same verdict.
        ledger.finalize()
        for source_folder in source_folders:
            db_path = os.path.join(format_path_for_system(source_folder),
                                   'measurements', 'measurements.db')
            if os.path.isfile(db_path):
                ledger.stamp(db_path)
    return


def preprocess_generate_masks_timelapse(settings):
    """Entry point for the standalone **Timelapse** module.

    Identical to :func:`preprocess_generate_masks` except that ``timelapse`` is
    forced on, so every well/field is grouped into a time stack, randomization
    is switched off, per-channel movies are written, and the objects listed in
    ``timelapse_objects`` are linked across frames and relabelled with their
    track IDs.

    Timelapse is a first-class spaCR workflow, not a checkbox on mask
    generation — that is why it has its own module, its own settings group
    (:func:`spacr.settings.get_timelapse_settings`) and this entry point.

    :param settings: Settings dict; canonicalized via
        :func:`spacr.settings.get_timelapse_settings`. Same keys as
        :func:`preprocess_generate_masks` plus the ``timelapse_*`` tracking
        group. ``timelapse`` is overwritten with True.
    :returns: None. Same outputs as :func:`preprocess_generate_masks`, plus
        ``<src>/movies`` and track-relabelled masks.

    See Also:
        :func:`spacr.timelapse.automated_motility_assay` — the Motility Assay
        module, which consumes the tracked ``merged/*.npy`` this produces.
    """
    from .settings import get_timelapse_settings

    if settings is None:
        settings = {}
    if settings.get('timelapse', True) is False:
        # Non-silent: the module *is* timelapse, so an incoming False (an old
        # mask settings CSV, say) is overridden rather than quietly honoured.
        print("Timelapse module: settings['timelapse'] was False — forcing it "
              "to True. Use the Mask module for non-timelapse segmentation.")
    settings = get_timelapse_settings(settings)
    return preprocess_generate_masks(settings)


def _validate_umap_source_db(db_path, tables, require_png_list=True):
    """Fail early — and by name — when a measurements DB cannot back an image UMAP.

    :func:`generate_image_umap` joins the requested feature ``tables`` against
    ``png_list`` on ``png_list.cell_id`` and then reads ``png_path`` off the
    result. Every way that can go wrong used to surface far downstream as an
    exception that named neither the database nor the column:

    * ``png_list`` without ``cell_id`` → ``KeyError("['cell_id'] not in index")``
      raised inside :func:`spacr.io._read_and_join_tables`;
    * no ``png_list`` table at all → ``KeyError('png_path')`` in this function;
    * an empty / never-measured database → ``UnboundLocalError: local variable
      'image_paths' referenced before assignment`` from
      :func:`spacr.utils.correct_paths`.

    Feature tables stay optional (a run without a pathogen channel has no
    ``pathogen`` table and :func:`spacr.io._read_and_join_tables` just skips
    it), but the join is anchored on ``cell``, so a requested ``cell`` table
    that is absent is fatal too.

    ``png_list`` is only required when the crops come out of the PNG folder.
    With ``crop_source='merged'`` the thumbnails are cut out of
    ``merged/*.npy`` on demand and the object table alone carries everything
    that needs — ``object_label`` and ``path_name`` — so requiring a table
    that run never wrote would be refusing a database that is complete.

    :param db_path: path of a ``measurements/measurements.db``.
    :param tables: table names the caller asked to embed, including
        ``'png_list'``.
    :param require_png_list: demand ``png_list`` and its two columns. False
        when the crops are being cut on demand.
    :returns: None.
    :raises ValueError: naming ``db_path`` and exactly what is missing.
    """
    import sqlite3 as _sqlite3

    if not os.path.isfile(db_path):
        raise ValueError(
            f"generate_image_umap: no measurements database at {db_path}. "
            "Run the Measure module on this source folder (with save_png "
            "enabled) before embedding it.")

    conn = _sqlite3.connect(db_path)
    try:
        present = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        png_cols = {row[1] for row in conn.execute("PRAGMA table_info('png_list')")}
    finally:
        conn.close()

    present_desc = ', '.join(sorted(present)) if present else 'none'

    if require_png_list:
        if 'png_list' not in present:
            raise ValueError(
                f"generate_image_umap: {db_path} has no 'png_list' table, which "
                "supplies the single-object PNG paths the embedding plots. "
                f"Tables present: {present_desc}. Re-run the Measure module with "
                "save_png=True, or set crop_source='merged' to cut the crops "
                "out of merged/*.npy instead.")

        missing_cols = [c for c in ('cell_id', 'png_path') if c not in png_cols]
        if missing_cols:
            raise ValueError(
                f"generate_image_umap: the 'png_list' table in {db_path} is "
                f"missing the column(s) {', '.join(missing_cols)}. 'cell_id' is "
                "the key the object features are joined on and 'png_path' is the "
                f"crop location. Columns present: {', '.join(sorted(png_cols))}. "
                "Re-run the Measure module with save_png=True to rebuild png_list.")

    feature_tables = [t for t in tables if t != 'png_list']
    found = [t for t in feature_tables if t in present]
    if not found:
        raise ValueError(
            f"generate_image_umap: none of the requested feature tables "
            f"({', '.join(feature_tables) or 'none requested'}) exist in "
            f"{db_path}. Tables present: {present_desc}.")
    if 'cell' in feature_tables and 'cell' not in present:
        raise ValueError(
            f"generate_image_umap: {db_path} has no 'cell' table. The "
            "object join is anchored on cell objects, so nucleus/pathogen/"
            "cytoplasm features alone cannot be embedded. "
            f"Tables present: {present_desc}.")


def generate_image_umap(settings=None, return_fig=False):
    """Generate a UMAP or tSNE embedding of per-object features and plot it.

    Reads measurements from the SQLite backend(s), applies preprocessing and
    dimensionality reduction, clusters the embedding, and renders scatter/grid
    plots of the resulting clusters.

    The thumbnails overlaid on the embedding come from whichever crop source
    ``crop_source`` names: ``'png'`` (and ``'auto'`` wherever a crop folder
    exists) reads the pre-generated PNGs, ``'merged'`` (and ``'auto'`` with no
    folder) cuts each one out of ``merged/*.npy`` on demand through
    :mod:`spacr.crops`, so the embedding can be drawn on a project that never
    kept a crop folder — ``png_list`` is not required in that case. Either way
    the pixels arrive through :mod:`spacr.crops`, so a legacy (pre-341f446)
    crop folder is channel-corrected on load and the montage shows the same
    image the Annotate screen does.

    :param settings: Configuration dict; canonicalized via
        :func:`spacr.settings.set_default_umap_image_settings`. Common keys:
        ``src``, ``tables``, ``row_limit``, ``clustering``,
        ``reduction_method`` (``'UMAP'`` or ``'tSNE'``),
        ``embedding_by_controls``, ``col_to_compare``, ``pos``, ``neg``,
        ``plot_images``, ``save_figure``, ``exclude``, ``crop_source``.
    :param return_fig: When True, return the Matplotlib figure instead of the
        annotated DataFrame.
    :returns: DataFrame of the input rows plus a ``cluster`` column, or a
        Matplotlib ``Figure`` when ``return_fig`` is True. With
        ``remove_cluster_noise`` the noise objects are dropped from the frame
        as well as from the embedding, so the two always describe the same
        objects.
    :raises ValueError: when a source has no usable ``measurements.db`` — see
        :func:`_validate_umap_source_db` for exactly what is required.
    :raises NotImplementedError: when ``resnet_features`` is set; embedding
        raw crops with ResNet features is not implemented.
    """
 
    if settings is None:
        settings = {}
    from .io import _read_and_join_tables
    from .utils import get_db_paths, preprocess_data, reduction_and_clustering, remove_noise, generate_colors, correct_paths, plot_embedding, plot_clusters_grid, cluster_feature_analysis, map_condition
    from .settings import set_default_umap_image_settings
    settings = set_default_umap_image_settings(settings)

    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]

    if settings['plot_images'] is False:
        settings['black_background'] = False

    if settings['color_by']:
        settings['remove_cluster_noise'] = False
        settings['plot_outlines'] = False
        settings['smooth_lines'] = False

    print(f'Generating Image UMAP ...')
    settings_df = pd.DataFrame(list(settings.items()), columns=['Key', 'Value'])
    settings_dir = os.path.join(settings['src'][0],'settings')
    settings_csv = os.path.join(settings_dir,'embedding_settings.csv')
    os.makedirs(settings_dir, exist_ok=True)
    settings_df.to_csv(settings_csv, index=False)
    from .utils import pretty_print_settings
    pretty_print_settings(settings, title="Image UMAP Settings")

    db_paths = get_db_paths(settings['src'])
    tables = settings['tables'] + ['png_list']
    all_df = pd.DataFrame()
    # Where the thumbnails come from. 'png' (and 'auto' on any project that
    # has a crop folder) reads the folder, exactly as before; 'merged' (and
    # 'auto' with no folder) cuts each thumbnail out of merged/*.npy on
    # demand, so the embedding can be drawn with no crop folder on disk.
    # Either way the pixels arrive through spacr.crops, so a legacy folder is
    # channel-corrected on load and the two sources show the same image.
    from .io import open_crop_source, crop_refs_for_rows, CROP_REF_COLUMN
    # 'cell', not settings['visualize']: _read_and_join_tables anchors the join
    # on the cell table, so every row's object_label IS a cell label. Cutting
    # the nucleus plane with a cell label would return a different object --
    # or none -- for every point on the map.
    crop_object = 'cell'

    for i,db_path in enumerate(db_paths):
        source = open_crop_source(settings, settings['src'][i],
                                  object_type=crop_object,
                                  verbose=bool(settings.get('verbose', True)))
        on_demand = source is not None and getattr(source, 'kind', 'png') == 'merged'
        # Say which database is unusable, and why, instead of letting the
        # join fail with a bare KeyError three modules away.
        _validate_umap_source_db(db_path, tables,
                                 require_png_list=not on_demand)
        df = _read_and_join_tables(db_path, table_names=tables)
        df, image_paths_tmp = correct_paths(df, settings['src'][i])
        if source is not None and settings['plot_images']:
            # No .copy(): _read_and_join_tables hands back a fresh frame that
            # correct_paths already writes into, and copying a 200-column
            # screen-sized frame to add one column is gigabytes for nothing.
            df[CROP_REF_COLUMN] = crop_refs_for_rows(source, df,
                                                     object_type=crop_object)
        all_df = pd.concat([all_df, df], axis=0)
        #image_paths.extend(image_paths_tmp)

    all_df['cond'] = all_df['columnID'].apply(map_condition, neg=settings['neg'], pos=settings['pos'], mix=settings['mix'])

    if settings['exclude_conditions']:
        if isinstance(settings['exclude_conditions'], str):
            settings['exclude_conditions'] = [settings['exclude_conditions']]
        row_count_before = len(all_df)
        all_df = all_df[~all_df['cond'].isin(settings['exclude_conditions'])]
        if settings['verbose']:
            print(f'Excluded {row_count_before - len(all_df)} rows after excluding: {settings["exclude_conditions"]}, rows left: {len(all_df)}')

    if settings['row_limit'] is not None:
        # row_limit is a cap, not a demand: asking for more rows than the
        # screen contains used to abort the whole embedding with pandas'
        # "Cannot take a larger sample than population", which names neither
        # the setting nor the source.
        n_rows = min(int(settings['row_limit']), len(all_df))
        if n_rows < settings['row_limit']:
            print(f"row_limit={settings['row_limit']} exceeds the {len(all_df)} "
                  f"rows available; using all {len(all_df)}.")
        all_df = all_df.sample(n=n_rows, random_state=42)

    # The handles are pulled off the frame here, after every row filter above
    # and before the numeric preprocessing below, and the column is dropped in
    # the same breath: it holds Python objects, so leaving it on the frame
    # would put `<LazyCropPNG …>` into embedding_results.csv.
    if CROP_REF_COLUMN in all_df.columns:
        image_paths = all_df[CROP_REF_COLUMN].to_list()
        all_df = all_df.drop(columns=[CROP_REF_COLUMN])
    elif 'png_path' in all_df.columns:
        image_paths = all_df['png_path'].to_list()
    else:
        # No crop source and no png_path: the embedding still means something,
        # the montage does not.
        print("No crop source and no 'png_path' column; plotting points only.")
        image_paths = None
        settings['plot_images'] = False

    if settings['embedding_by_controls']:
        
        # Extract and reset the index for the column to compare
        col_to_compare = all_df[settings['col_to_compare']].reset_index(drop=True)
        print(col_to_compare)
        #if settings['only_top_features']:
        #    column_list = None
            
        # Preprocess the data to obtain numeric data
        numeric_data = preprocess_data(all_df, settings['filter_by'], settings['remove_highly_correlated'], settings['log_data'], settings['exclude'])

        # Convert numeric_data back to a DataFrame to align with col_to_compare
        numeric_data_df = pd.DataFrame(numeric_data)

        # Ensure numeric_data_df and col_to_compare are properly aligned
        numeric_data_df = numeric_data_df.reset_index(drop=True)

        # Assign the column back to numeric_data_df
        numeric_data_df[settings['col_to_compare']] = col_to_compare

        # Subset the dataframe based on specified column values for controls
        positive_control_df = numeric_data_df[numeric_data_df[settings['col_to_compare']] == settings['pos']].copy()
        negative_control_df = numeric_data_df[numeric_data_df[settings['col_to_compare']] == settings['neg']].copy()
        control_numeric_data_df = pd.concat([positive_control_df, negative_control_df])

        # Drop the comparison column from numeric_data_df and control_numeric_data_df
        numeric_data_df = numeric_data_df.drop(columns=[settings['col_to_compare']])
        control_numeric_data_df = control_numeric_data_df.drop(columns=[settings['col_to_compare']])

        # Convert numeric_data_df and control_numeric_data_df back to numpy arrays
        numeric_data = numeric_data_df.values
        control_numeric_data = control_numeric_data_df.values

        # Train the reducer on control data
        _, _, reducer = reduction_and_clustering(control_numeric_data, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['eps'], settings['min_samples'], settings['clustering'], settings['reduction_method'], settings['verbose'], n_jobs=settings['n_jobs'], mode='fit', model=False)
        
        # Apply the trained reducer to the entire dataset
        numeric_data = preprocess_data(all_df, settings['filter_by'], settings['remove_highly_correlated'], settings['log_data'], settings['exclude'])
        embedding, labels, _ = reduction_and_clustering(numeric_data, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['eps'], settings['min_samples'], settings['clustering'], settings['reduction_method'], settings['verbose'], n_jobs=settings['n_jobs'], mode=None, model=reducer)

    else:
        if settings['resnet_features']:
            # Placeholder, never implemented. It used to `pass`, so the run
            # fell through to `plot_embedding(embedding, ...)` with `embedding`
            # unbound and died with an UnboundLocalError that pointed at the
            # plotting code rather than at the setting the user turned on.
            raise NotImplementedError(
                "resnet_features is not implemented: spaCR cannot embed raw "
                "PNG crops with ResNet features yet. Set resnet_features=False "
                "to embed the measured feature table instead.")
            #numeric_data, embedding, labels = generate_umap_from_images(image_paths, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['clustering'], settings['eps'], settings['min_samples'], settings['n_jobs'], settings['verbose'])
        else:
            # Apply the trained reducer to the entire dataset
            numeric_data = preprocess_data(all_df, settings['filter_by'], settings['remove_highly_correlated'], settings['log_data'], settings['exclude'])
            embedding, labels, _ = reduction_and_clustering(numeric_data, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['eps'], settings['min_samples'], settings['clustering'], settings['reduction_method'], settings['verbose'], n_jobs=settings['n_jobs'])
    
    if settings['remove_cluster_noise']:
        # Remove noise from the clusters (removes -1 labels from DBSCAN).
        # The frame and the image paths must lose exactly the same rows:
        # dropping points from the embedding alone left `labels` shorter than
        # `all_df`, and the `all_df['cluster'] = labels` assignment at the end
        # of this function then died with "Length of values (50) does not
        # match length of index (60)" — i.e. this setting was unusable
        # whenever DBSCAN called *some* (but not all) points noise.
        keep = np.asarray(labels) != -1
        embedding, labels = remove_noise(embedding, labels)
        if keep.any():
            if image_paths is not None:
                image_paths = [p for p, k in zip(image_paths, keep) if k]
            all_df = all_df[keep].reset_index(drop=True)
        # else: every point was noise, so `labels` is now empty and the
        # single-cluster fallback further down keeps the frame intact
        # instead of handing back nothing at all.

    # Plot the results. color_by replaces the cluster labels with a metadata
    # column; how the embedding was fitted makes no difference (the two arms
    # of the if/else this replaces were character-for-character identical).
    if settings['color_by']:
        labels = all_df[settings['color_by']]


    # Generate colors for the clusters
    colors = generate_colors(len(np.unique(labels)), settings['black_background'])

    # Plot the embedding
    umap_plt = plot_embedding(embedding, image_paths, labels, settings['image_nr'], settings['img_zoom'], colors, settings['plot_by_cluster'], settings['plot_outlines'], settings['plot_points'], settings['plot_images'], settings['smooth_lines'], settings['black_background'], settings['figuresize'], settings['dot_size'], settings['remove_image_canvas'], settings['verbose'])
    if settings['plot_cluster_grids'] and settings['plot_images']:
        grid_plt = plot_clusters_grid(embedding, labels, settings['image_nr'], image_paths, colors, settings['figuresize'], settings['black_background'], settings['verbose'])
    
    # Save figure as PDF if required
    if settings['save_figure']:
        results_dir = os.path.join(settings['src'][0], 'results')
        os.makedirs(results_dir, exist_ok=True)
        reduction_method = settings['reduction_method'].upper()
        embedding_path = os.path.join(results_dir, f'{reduction_method}_embedding.pdf')
        umap_plt.savefig(embedding_path, format='pdf')
        print(f'Saved {reduction_method} embedding to {embedding_path} and grid to {embedding_path}')
        if settings['plot_cluster_grids'] and settings['plot_images']:
            grid_path = os.path.join(results_dir, f'{reduction_method}_grid.pdf')
            grid_plt.savefig(grid_path, format='pdf')
            print(f'Saved {reduction_method} embedding to {embedding_path} and grid to {grid_path}')

    # Add cluster labels to the dataframe
    if len(labels) > 0:
        all_df['cluster'] = labels
    else:
        all_df['cluster'] = 1  # Assign a default cluster label
        print("No clusters found. Consider reducing 'min_samples' or increasing 'eps' for DBSCAN.")

    # Save the results to a CSV file
    results_dir = os.path.join(settings['src'][0], 'results')
    results_csv = os.path.join(results_dir,'embedding_results.csv')
    os.makedirs(results_dir, exist_ok=True)
    all_df.to_csv(results_csv, index=False)
    print(f'Results saved to {results_csv}')

    if settings['analyze_clusters']:
        combined_results = cluster_feature_analysis(all_df)
        results_dir = os.path.join(settings['src'][0], 'results')
        cluster_results_csv = os.path.join(results_dir,'cluster_results.csv')
        os.makedirs(results_dir, exist_ok=True)
        combined_results.to_csv(cluster_results_csv, index=False)
        print(f'Cluster results saved to {cluster_results_csv}')

    fig = umap_plt.gcf() if hasattr(umap_plt, "gcf") else plt.gcf()

    # (saving CSVs etc. unchanged)

    if return_fig:
        return fig
    return all_df

def reducer_hyperparameter_search(settings=None, reduction_params=None, dbscan_params=None, kmeans_params=None, save=False, show=True, return_fig=False):
    """Sweep UMAP/tSNE and DBSCAN/KMeans hyperparameters over the feature table.

    Renders a grid of embeddings, one cell per (reduction, clustering) pair, so
    the caller can eyeball the impact of each parameter combination.

    :param settings: Config dict; canonicalized via
        :func:`spacr.settings.set_default_umap_image_settings`.
    :param reduction_params: Dict or list of dicts of parameters for the
        reduction method. Presence of ``n_neighbors`` selects UMAP,
        ``perplexity`` selects tSNE.
    :param dbscan_params: Dict or list of DBSCAN parameter dicts (each with
        ``eps`` and ``min_samples``).
    :param kmeans_params: Dict or list of KMeans parameter dicts.
    :param save: When True, save the grid figure to ``<src>/results``.
    :param show: When True and not saving, call ``plt.show``.
    :param return_fig: When True, return the Matplotlib figure.
    :returns: The figure when ``return_fig`` is True, otherwise None.
    :raises ValueError: when ``reduction_params`` is missing or empty, when it
        contains neither ``n_neighbors`` nor ``perplexity``, when it mixes the
        two, or when ``settings['reduction_method']`` is neither UMAP nor
        tSNE. All four are checked before any data is read.
    """
    
    if settings is None:
        settings = {}
    from .io import _read_and_join_tables
    from .utils import get_db_paths, preprocess_data, search_reduction_and_clustering, generate_colors, map_condition
    from .settings import set_default_umap_image_settings

    settings = set_default_umap_image_settings(settings)
    pointsize = settings['dot_size']
    if isinstance(dbscan_params, dict):
        dbscan_params = [dbscan_params]

    if isinstance(kmeans_params, dict):
        kmeans_params = [kmeans_params]

    if isinstance(reduction_params, dict):
        reduction_params = [reduction_params]

    # Determine reduction method based on the keys in reduction_param.
    # reduction_params is required: iterating None raised a TypeError from the
    # generator expression below that never mentioned the argument's name.
    if not reduction_params:
        raise ValueError(
            "reducer_hyperparameter_search: reduction_params is required. "
            "Pass a dict (or list of dicts) containing 'n_neighbors' to sweep "
            "UMAP or 'perplexity' to sweep tSNE.")

    wants_umap = any('n_neighbors' in param for param in reduction_params)
    wants_tsne = any('perplexity' in param for param in reduction_params)

    # The both-at-once check has to come FIRST. As a third `elif` after the
    # UMAP branch it could never fire, so a mixed sweep silently ran as UMAP
    # and every 'perplexity' value in it was ignored.
    if wants_umap and wants_tsne:
        raise ValueError("Reduction parameters must include 'n_neighbors' for UMAP or 'perplexity' for tSNE, not both.")
    elif wants_umap:
        reduction_method = 'umap'
    elif wants_tsne:
        reduction_method = 'tsne'
    else:
        # Previously fell through with reduction_method unbound and died on
        # the next line with an UnboundLocalError.
        raise ValueError(
            "Reduction parameters must include 'n_neighbors' for UMAP or "
            f"'perplexity' for tSNE; got {reduction_params}.")


    # Validated here, once, before a single row is read: this check used to
    # live inside the per-cell loop below, where it could never fire because
    # the line under it had already overwritten reduction_method with 'umap'
    # or 'tsne'. An unsupported method was therefore silently swapped out
    # instead of reported.
    if str(settings['reduction_method']).lower() not in ('umap', 'tsne'):
        raise ValueError(f"Unsupported reduction method: {settings['reduction_method']}. Supported methods are 'UMAP' and 'tSNE'")

    if settings['reduction_method'].lower() != reduction_method:
        settings['reduction_method'] = reduction_method
        print(f'Changed reduction method to {reduction_method} based on the provided parameters.')

    if settings['verbose']:
        display(pd.DataFrame(list(settings.items()), columns=['Key', 'Value']))

    db_paths = get_db_paths(settings['src'])
    
    tables = settings['tables']
    all_df = pd.DataFrame()
    for db_path in db_paths:
        df = _read_and_join_tables(db_path, table_names=tables)
        all_df = pd.concat([all_df, df], axis=0)

    all_df['cond'] = all_df['columnID'].apply(map_condition, neg=settings['neg'], pos=settings['pos'], mix=settings['mix'])

    if settings['exclude_conditions']:
        if isinstance(settings['exclude_conditions'], str):
            settings['exclude_conditions'] = [settings['exclude_conditions']]
        row_count_before = len(all_df)
        all_df = all_df[~all_df['cond'].isin(settings['exclude_conditions'])]
        if settings['verbose']:
            print(f'Excluded {row_count_before - len(all_df)} rows after excluding: {settings["exclude_conditions"]}, rows left: {len(all_df)}')

    if settings['row_limit'] is not None:
        # Same cap semantics as generate_image_umap.
        n_rows = min(int(settings['row_limit']), len(all_df))
        if n_rows < settings['row_limit']:
            print(f"row_limit={settings['row_limit']} exceeds the {len(all_df)} "
                  f"rows available; using all {len(all_df)}.")
        all_df = all_df.sample(n=n_rows, random_state=42)

    numeric_data = preprocess_data(all_df, settings['filter_by'], settings['remove_highly_correlated'], settings['log_data'], settings['exclude'])

    # Combine DBSCAN and KMeans parameters
    clustering_params = []
    if dbscan_params:
        for param in dbscan_params:
            param['method'] = 'dbscan'
            clustering_params.append(param)
    if kmeans_params:
        for param in kmeans_params:
            param['method'] = 'kmeans'
            clustering_params.append(param)

    print('Testing paramiters:', reduction_params)
    print('Testing clustering paramiters:', clustering_params)

    # Calculate the grid size
    grid_rows = len(reduction_params)
    grid_cols = len(clustering_params)

    fig_width = grid_cols*10
    fig_height = grid_rows*10

    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))

    # Make sure axs is always an array of axes
    axs = np.atleast_1d(axs)
    
    # Iterate through the Cartesian product of reduction and clustering hyperparameters
    for i, reduction_param in enumerate(reduction_params):
        for j, clustering_param in enumerate(clustering_params):
            if len(clustering_params) <= 1:
                axs[i].axis('off')
                ax = axs[i]
            elif len(reduction_params) <= 1:
                axs[j].axis('off')
                ax = axs[j]
            else:
                ax = axs[i, j]

            # Perform dimensionality reduction and clustering. The method is
            # 'umap' or 'tsne' and nothing else — it is validated once, above,
            # before any data is read.
            if settings['reduction_method'].lower() == 'umap':
                n_neighbors = reduction_param.get('n_neighbors', 15)

                if isinstance(n_neighbors, float):
                    n_neighbors = int(n_neighbors * len(numeric_data))

                min_dist = reduction_param.get('min_dist', 0.1)
                embedding, labels = search_reduction_and_clustering(numeric_data, n_neighbors, min_dist, settings['metric'], 
                                                                    clustering_param.get('eps', 0.5), clustering_param.get('min_samples', 5), 
                                                                    clustering_param['method'], settings['reduction_method'], settings['verbose'], reduction_param, n_jobs=settings['n_jobs'])
                
            else:  # 'tsne'
                perplexity = reduction_param.get('perplexity', 30)

                if isinstance(perplexity, float):
                    perplexity = int(perplexity * len(numeric_data))

                embedding, labels = search_reduction_and_clustering(numeric_data, perplexity, 0.1, settings['metric'],
                                                                    clustering_param.get('eps', 0.5), clustering_param.get('min_samples', 5),
                                                                    clustering_param['method'], settings['reduction_method'], settings['verbose'], reduction_param, n_jobs=settings['n_jobs'])

            # Plot the results
            if settings['color_by']:
                unique_groups = all_df[settings['color_by']].unique()
                colors = generate_colors(len(unique_groups), False)
                for group, color in zip(unique_groups, colors):
                    indices = all_df[settings['color_by']] == group
                    ax.scatter(embedding[indices, 0], embedding[indices, 1], s=pointsize, label=f"{group}", color=color)
            else:
                unique_labels = np.unique(labels)
                colors = generate_colors(len(unique_labels), False)
                for label, color in zip(unique_labels, colors):
                    ax.scatter(embedding[labels == label, 0], embedding[labels == label, 1], s=pointsize, label=f"Cluster {label}", color=color)

            ax.set_title(f"{settings['reduction_method']} {reduction_param}\n{clustering_param['method']} {clustering_param}")
            ax.legend()

    plt.tight_layout()
    if save:
        results_dir = os.path.join(settings['src'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'hyperparameter_search.pdf'))
    if return_fig:
        return fig
    if show and not save:
        plt.show()
    return

def generate_screen_graphs(settings):
    """Build recruitment-metric summary graphs per source and for the combined data.

    Reads per-object measurements, annotates conditions, computes the recruitment
    metric, and generates one plot per source folder plus one combined plot.

    :param settings: Config dict with keys ``src`` (path or list of paths),
        ``tables``, ``cells``, ``controls``, ``controls_loc``, ``graph_type``,
        ``summary_func``, ``y_axis_start``, ``error_bar_type``, ``theme``,
        ``representation``, ``nuclei_limit``, ``pathogen_limit``.
    :returns: None. Figures and CSVs are written under each source's
        ``results/`` folder.
    """
    
    from .plot import spacrGraph
    from .io import _read_and_merge_data
    from.utils import annotate_conditions

    if isinstance(settings['src'], str):
        srcs = [settings['src']]
    else:
        srcs = settings['src']

    all_df = pd.DataFrame()
    figs = []
    results = []

    for src in srcs:
        db_loc = [os.path.join(src, 'measurements', 'measurements.db')]
        
        # Read and merge data from the database
        df, _ = _read_and_merge_data(db_loc, settings['tables'], verbose=True, nuclei_limit=settings['nuclei_limit'], pathogen_limit=settings['pathogen_limit'])
        
        # Annotate the data
        df = annotate_conditions(df, cells=settings['cells'], cell_loc=None, pathogens=settings['controls'], pathogen_loc=settings['controls_loc'], treatments=None, treatment_loc=None)
        
        # Calculate recruitment metric
        df['recruitment'] = df['pathogen_channel_1_mean_intensity'] / df['cytoplasm_channel_1_mean_intensity']
                
        # Combine with the overall DataFrame
        all_df = pd.concat([all_df, df], ignore_index=True)
    
        # Generate individual plot
        plotter = spacrGraph(df,
                             grouping_column='pathogen',
                             data_column='recruitment',
                             graph_type=settings['graph_type'],
                             summary_func=settings['summary_func'],
                             y_lim=[settings['y_axis_start'], None],
                             error_bar_type=settings['error_bar_type'],
                             theme=settings['theme'],
                             representation=settings['representation'])

        plotter.create_plot()
        fig = plotter.get_figure()
        results_df = plotter.get_results()
        
        # Append to the lists
        figs.append(fig)
        results.append(results_df)
    
    # Generate plot for the combined data (all_df)
    plotter = spacrGraph(all_df,
                         grouping_column='pathogen',
                         data_column='recruitment',
                         graph_type=settings['graph_type'],
                         summary_func=settings['summary_func'],
                         y_lim=[settings['y_axis_start'], None],
                         error_bar_type=settings['error_bar_type'],
                         theme=settings['theme'],
                         representation=settings['representation'])

    plotter.create_plot()
    fig = plotter.get_figure()
    results_df = plotter.get_results()
    
    figs.append(fig)
    results.append(results_df)
    
    # Save figures and results
    for i, fig in enumerate(figs):
        res = results[i]
        
        if i < len(srcs):
            source = srcs[i]
        else:
            source = srcs[0]

        # Ensure the destination folder exists
        dst = os.path.join(source, 'results')
        print(f"Savings results to {dst}")
        os.makedirs(dst, exist_ok=True)
        
        # Save the figure and results DataFrame
        fig.savefig(os.path.join(dst, f"figure_controls_{i}_{settings['representation']}_{settings['summary_func']}_{settings['graph_type']}.pdf"), format='pdf')
        res.to_csv(os.path.join(dst, f"results_controls_{i}_{settings['representation']}_{settings['summary_func']}_{settings['graph_type']}.csv"), index=False)

    return