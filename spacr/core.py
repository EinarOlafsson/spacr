"""Create microscopy masks and explore measured phenotypes in two dimensions.

WHAT IT IS FOR
==============
This landing page currently serves two spaCR tiles.  **Mask** runs
:func:`preprocess_generate_masks` to turn raw multichannel acquisitions into
preprocessed arrays and Cellpose masks for cells, nuclei, pathogens, and
organelles.  **Image UMAP** runs :func:`generate_image_umap` to reduce measured
single-object features, cluster them, and optionally place representative
image crops on the embedding.  The same module also exposes the timelapse-mask
entry point; the two main workflows remain independent, and UMAP does not
segment images.

WHAT IT NEEDS
=============
Mask generation needs one or more source folders, a filename metadata scheme
(``cellvoyager`` or an automatic/custom regex), zero-based channel indices for
the objects to segment, and suitable object diameters and model choices.  At
least one segmentation channel must be enabled.  Start with ``dry_run=True``
to validate paths, channels, models, and the planned writes without loading a
model or changing the project.

Image UMAP needs an existing ``measurements/measurements.db`` for every source,
the object tables and features to include, and reduction/clustering settings.
It embeds numeric measurements rather than raw pixels.  Thumbnail images come
from the measured ``png_list`` table when ``crop_source='png'`` or are cut from
``merged/*.npy`` on demand when ``crop_source='merged'``; the latter is useful
when measurement crops were not saved.

WHAT IT PRODUCES
================
A normal Mask run writes preprocessed stacks, object masks, overlays and
segmentation-QC artifacts, settings CSVs, counts in ``measurements.db``, and a
run manifest beneath the source tree; it normally returns ``None``.  A dry run
instead returns its preflight problem list.  Timelapse mask generation also
writes movies and masks relabelled with track identities.

Image UMAP returns an annotated DataFrame containing the two-dimensional
coordinates and ``cluster`` labels, or a Matplotlib figure when
``return_fig=True``.  Depending on the save and plotting settings, it also
writes the embedding, cluster views, representative-crop grids, feature
summaries, and the resolved settings alongside the project.

WHAT TO DO NEXT
===============
After Mask finishes, inspect overlays and segmentation-QC flags before running
:func:`spacr.measure.measure_crop`; inaccurate masks make every downstream
feature inaccurate.  After measurement, use Image UMAP to inspect phenotype
structure, colour by plate or condition to expose batch effects, and validate
clusters against their representative crops before treating them as biology.
Use the Mask tile for :func:`preprocess_generate_masks` and the Image UMAP tile
for :func:`generate_image_umap` until those tiles receive separate API pages.

Three details are deliberately explicit.  Channel numbers are zero-based and
diameters are pixels, so values copied from one magnification are not portable
without conversion.  The default v1 mask pipeline preserves the directory
layout expected by downstream tools; ``pipeline_style='v2'`` is opt-in and
writes a different streaming layout.  Finally, removing UMAP cluster noise
also removes those objects from the returned frame, keeping the table and the
visible embedding aligned rather than silently returning different samples.
"""

import os, torch, time, random

from . import _gc as gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:
    def display(*args, **kwargs):
        """Discard display payloads when IPython's helper is unavailable."""
        pass
import warnings

from .errors import RunLedger, raise_if_strict

from . import artifacts as artifact_status
from .runctx import run_context
from .plot import save_figure
from .object_roles import (ORGANELLE_ROLES, SEGMENTED_ROLES,
                           enabled_organelle_roles)

from .figures.style import figure_style, theme_target

warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")

def _score_v2_masks(src, settings, object_type: str = "cell"):
    """Score the masks a v2 run just wrote, the way v1 scores its own.

    v1 reaches this through :func:`spacr.object._run_seg_qc`, which globs a
    `<object_type>_mask_stack` folder. v2 has no such folder -- its mask is a
    CHANNEL of `merged/stack_<field>.npy` -- so it goes through
    :func:`spacr._v1_v2_bridge.v2_mask_source`, which reads
    `channel_order.json` to find the plane and hands back one lazy reader per
    field. Both layouts then meet in the same scorecard, which is the point:
    `seg_qc` means the same thing whichever pipeline produced the masks.

    :param src: the plate folder handed to `run_v2`; `merged/` sits under it.
    :returns: what :func:`spacr.seg_qc.run_segmentation_qc` returns, or None
        when QC is off, there are no masks to score, or scoring failed.

    Never raises into a finished run. A plate that has just spent hours
    segmenting must not lose its masks to a scorecard bug, which is the same
    rule `_run_seg_qc` follows.
    """
    try:
        from .seg_qc import qc_mode, run_segmentation_qc, thresholds_from_settings
        from ._v1_v2_bridge import v2_mask_source

        mode = qc_mode(settings)
        if mode == 'off':
            return None

        merged = os.path.join(os.fspath(src), 'merged')
        source = v2_mask_source(merged, object_type)
        if not source:
            print(f"Segmentation QC found no {object_type} masks to score in "
                  f"{merged}: channel_order.json names none, so the stacks do "
                  f"not say which plane is a mask.")
            return None

        result = run_segmentation_qc(
            source,
            object_type=object_type,
            dst=os.fspath(src),
            mode=mode,
            thresholds=thresholds_from_settings(settings),
            verbose=bool(settings.get('verbose', True)),
        )
    except Exception as exc:                                 # noqa: BLE001
        print(f"Segmentation QC skipped for {object_type}: "
              f"{type(exc).__name__}: {exc}")
        return None

    if result is not None and result.get('mode') == 'flag':
        settings.setdefault('seg_qc_flags', {})[object_type] = result['flags']
    return result


def _overlay_candidates(merged_src):
    """The merged image stacks an overlay plot can draw, and nothing else.

    ``merged/`` also holds sidecar files -- ``.spacr_plane_layout.json``
    among them -- and handing one of those to
    :func:`spacr.plot.plot_image_mask_overlay` as an image fails that plot.
    Only ``.npy`` stacks are returned, which is also what the test-mode
    example count already counts, so the count and the list agree.

    :param merged_src: the ``merged`` folder of a plate.
    :returns: the ``.npy`` file names in that folder, in directory order,
        without dot-files such as a macOS ``._`` sidecar.
    """
    from .io import _listdir_visible

    return [name for name in _listdir_visible(merged_src) if name.endswith('.npy')]


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
        - ``pathogen_model`` — path to a Cellpose-SAM checkpoint to segment
          pathogens with, instead of stock ``cpsam``. The pre-SAM
          ``toxo_pv_lumen`` / ``toxo_cyto`` names are gone and resolve to
          ``cpsam``; a PATH to a fine-tune is honoured, and a path that is
          not there stops the run rather than silently using stock weights.
        - ``consolidate`` — copy nested images into ``src/consolidated``
          before processing.
        - ``preprocess`` / ``masks`` — toggle the two pipeline halves.
        - ``adjust_cells`` — reconcile cell masks against nuclei+pathogen.
        - ``timelapse`` — enable trackpy linking; forces
          ``randomize=False``.
        - ``motility_analysis`` — when timelapse is enabled, analyze the
          completed merged frames once per plate, rebuilding measurements
          from the current masks rather than reusing an older assay table.
        - ``robustness_report`` — after the masks exist, re-segment a few
          sampled fields over a small grid of diameters, thresholds and
          contrast enhancement and write a stability report per object to
          ``qc/segmentation_robustness_<object>.csv``, flagging the settings
          the results are fragile to.
        - ``dry_run`` — validate only: inspect the input folders, print the
          preflight report and plan and return, without writing anything or
          loading a model.
        - ``watch_folder`` — keep watching ``src`` and analyse each field as
          its files arrive and stop changing, with ``watch_pipeline``,
          ``watch_measure_settings``, ``watch_settle_seconds``,
          ``watch_poll_seconds`` and ``watch_idle_minutes``. Results gather
          in ``src/spacr_watch``, and a record there lets a restarted watch
          skip the fields already analysed.
        - ``microscope_feedback`` — during a ``watch_folder`` run with
          ``watch_pipeline='mask_measure'``, send the objects matching
          ``microscope_event_query`` back to the microscope named by
          ``microscope_driver`` to be imaged again, at stage positions
          worked out from ``microscope_positions`` and
          ``microscope_stage_transform``.
        - ``src`` may also be a cloud address (``s3://``, ``gs://``,
          ``az://``, ``https://``). An OME-Zarr plate there has only the
          wells, fields and pyramid level named by ``cloud_wells``,
          ``cloud_fields`` and ``cloud_level`` fetched, as TIFFs, into a
          folder under ``cloud_cache``, and the run analyses that folder; a
          cloud folder of images is mirrored there instead. Credentials come
          from the standard places, chosen with ``cloud_anonymous``,
          ``cloud_profile`` and ``cloud_endpoint``, and ``cloud_results``
          copies the measurements folder back to cloud storage.
        - ``save``, ``plot``, ``verbose``, ``test_mode``, ``n_jobs``.

    :returns: ``None`` on a normal run, having written masks, overlays,
        ``measurements.db`` counts and settings CSVs into subfolders of
        ``src``. When ``dry_run`` is set, returns instead the list of
        problems from :func:`spacr.validate.run_preflight`. When
        ``watch_folder`` is set, returns a dict naming the fields analysed,
        failed and never completed when the watch ends.
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

    Mask writes which object sits in which as its own table in measurements.db.
    That table is built from Measure's object tables, so before Measure has run
    there are none, and that is not a failure.
    """
    if settings.get('dry_run', False):
        from .validate import run_preflight
        return run_preflight(settings, 'mask')

    from .object import (_eval_diameter, generate_organelle_masks_sam,
                         generate_cellpose_masks_sam)
    from .io import (preprocess_img_data, _load_and_concatenate_arrays,
                     _normalized_npz_field_ids, convert_to_yokogawa,
                     convert_separate_files_to_yokogawa, _listdir_visible,
                     _check_archives_without_preprocessing)
    from .plot import plot_image_mask_overlay, plot_arrays
    from .utils import _pivot_counts_table, check_mask_folder, adjust_cell_masks, print_progress, save_settings, format_path_for_system, normalize_src_path, generate_image_path_map, copy_images_to_consolidated, reset_cellpose_model_reports
    from .settings import set_default_settings_preprocess_generate_masks, _set_organelle_defaults
    from .qt.mask_engine import object_filter_area_floor
    from .cancellation import checkpoint as cancellation_checkpoint

    reset_cellpose_model_reports()

    if 'src' in settings:
        if not isinstance(settings['src'], (str, list)):
            raise ValueError('src must be a string or a list of strings')
    else:
        raise ValueError('src is a required parameter')

    settings['src'] = normalize_src_path(settings['src'])

    from .ome_zarr import _needs_cloud_run, _run_with_cloud_sources
    if _needs_cloud_run(settings):
        return _run_with_cloud_sources(preprocess_generate_masks, settings, 'mask')

    if _watch_truthy(settings.get('watch_folder', False)):
        return _watch_folder_and_analyse(settings)

    if settings.get('pipeline_style', 'v1') == 'v2':
        settings = set_default_settings_preprocess_generate_masks(settings)
        if settings.get('mask_parallel'):
            print('mask_parallel applies to the v1 mask pipeline; this v2 run segments on one device.')
        settings = _set_organelle_defaults(settings)
        from .pipeline_v2 import run_v2
        from ._v1_v2_bridge import (
            v2_channels_from_settings, report_disk_savings,
        )
        srcs = settings['src'] if isinstance(settings['src'], list) \
                                else [settings['src']]
        for src in srcs:
            channels, channel_names = v2_channels_from_settings(settings)
            cell_index = (
                channel_names.index('cell') if 'cell' in channel_names else 0
            )
            cellpose_channels = [cell_index]
            if 'nucleus' in channel_names:
                cellpose_channels.append(channel_names.index('nucleus'))
            result = run_v2(
                src,
                channels=channels,
                channel_names=channel_names,
                model_name=settings.get('cell_model_name', 'cpsam'),
                channels_for_cellpose=tuple(cellpose_channels),
                diameter=_eval_diameter(
                    settings.get('cell_diameter'), 'cell'),
                batch_fields=int(settings.get('batch_fields', 8)),
                metadata_type=settings.get('metadata_type', 'auto'),
                custom_regex=settings.get('custom_regex'),
                keep_npz=bool(settings.get('keep_npz', False)),
                cellprob_threshold=float(settings.get('cell_cellprob_threshold', 0.0)),
                flow_threshold=float(settings.get('cell_flow_threshold', 0.4)),
                min_size=object_filter_area_floor(settings, 'cell'),
                resample=True,
                postprocess_settings=settings,
                object_type='cell',
                illumination_settings=settings,
            )
            report_disk_savings(src, result['stacks'])
            _score_v2_masks(src, settings, object_type='cell')
        return
    
    if settings.get('consolidate', False):
        sources = (settings['src'] if isinstance(settings['src'], list)
                   else [settings['src']])
        consolidated_sources = []
        for source in sources:
            image_map = generate_image_path_map(source)
            copy_images_to_consolidated(image_map, source)
            consolidated_sources.append(os.path.join(source, 'consolidated'))
        settings['src'] = consolidated_sources

    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]

    source_folders = settings['src']
    ledger = RunLedger('preprocess_generate_masks')
    module_key = 'timelapse' if settings.get('timelapse') else 'mask'
    with run_context(module_key, settings, ledger=ledger) as run:
        for source_folder in source_folders:
            for attempt in run.policy.attempts_for(source_folder,
                                                   stage='plate'):
                with attempt:
                    cancellation_checkpoint()

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
                            except Exception as e:
                                refusal = (
                                    f"Could not convert {source_folder} with custom_regex "
                                    f"{settings['custom_regex']!r}: {type(e).__name__}: {str(e).rstrip('.')}. "
                                    f"spaCR did not fall back to converting without the regex: "
                                    f"that conversion gives each file the next free well in file "
                                    f"order and ignores the wells the file names carry, so it would "
                                    f"have relabelled the plate's wells. Correct the file or the "
                                    f"regex and run again. Clearing custom_regex to have spaCR "
                                    f"number the wells itself (rename_log.csv then records which "
                                    f"file became which well) is only safe once the plate*_*.tif "
                                    f"files this attempt already wrote are moved out of "
                                    f"{source_folder}. Automatic conversion now refuses folders "
                                    f"with converted images to prevent overwrites or changed "
                                    f"well assignments. A separate folder containing only the "
                                    f"original inputs is the safest place to retry.")
                                print(f'Error: {refusal}')
                                ledger.record_failure(source_folder,
                                                      stage='convert_metadata', exc=e)
                                ledger.finalize()
                                raise_if_strict(refusal, exc=e, settings=settings)
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

                    if all(settings.get(f'{role}_channel') is None
                           for role in SEGMENTED_ROLES):
                        print('Error: At least one of the registered object channels must be defined')
                        raise_if_strict(
                            'At least one registered *_channel (for example '
                            'cell_channel or organelle_channel) must be set; '
                            'no masks can be generated.', settings=settings)
                        return

                    save_settings(settings, name='gen_mask_settings')


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

                    from ._mask_workers import _parallel_mask_plan
                    gpu_plan = _parallel_mask_plan(settings)

                    if settings['preprocess']:
                        settings, src = preprocess_img_data(settings)

                    organelle_roles = enabled_organelle_roles(settings)
                    files_to_process = sum([
                        settings['cell_channel'] is not None,
                        settings['nucleus_channel'] is not None,
                        settings['pathogen_channel'] is not None,
                    ]) + len(organelle_roles)
                    files_processed = 0

                    if settings['masks']:
                        mask_src = os.path.join(src, 'masks')
                        os.makedirs(mask_src, exist_ok=True)

                        if not settings['preprocess']:
                            _check_archives_without_preprocessing(src)
                            from .psf_pipeline import (
                                validate_psf_resume, _record_path,
                                processing_requested)
                            if (processing_requested(settings) or
                                    _record_path(src).exists()):
                                psf_channels = list(dict.fromkeys(
                                    int(settings[f'{role}_channel'])
                                    for role in ('nucleus', 'cell', 'pathogen',
                                                 *ORGANELLE_ROLES)
                                    if settings.get(f'{role}_channel') is not None))
                                validate_psf_resume(
                                    settings, src, psf_channels,
                                    expected_fields=_normalized_npz_field_ids(mask_src))

                        from .image_quality import screen_fields
                        quality_paths = None
                        if settings.get('image_qc_mode', 'off') != 'off':
                            quality_paths = [os.path.join(src, 'stack', field + '.npy')
                                             for field in _normalized_npz_field_ids(mask_src)]
                        settings['image_qc_excluded_fields'] = screen_fields(src, settings, quality_paths)
                        if quality_paths and len(settings['image_qc_excluded_fields']) == len(quality_paths):
                            print('All fields were excluded by the saved image-quality policy; no masks generated.')
                            break

                        if (not settings['preprocess'] and
                                settings.get('illumination_correction', False)):
                            from .illumination import (
                                load_segmentation_illumination_resume,
                            )
                            load_segmentation_illumination_resume(
                                settings,
                                provenance_path=os.path.join(
                                    src, 'illumination',
                                    'segmentation_application.json'),
                                pipeline_style='v1',
                                expected_fields=(
                                    _normalized_npz_field_ids(mask_src)),
                                verbose=settings.get('verbose', True),
                            )

                        if settings['cell_channel'] != None:
                            cancellation_checkpoint()
                            time_ls=[]
                            if check_mask_folder(
                                    src, 'cell_mask_stack',
                                    resume=settings.get('resume', False)):
                                start = time.time()
                                if gpu_plan is None:
                                    generate_cellpose_masks_sam(mask_src, settings, 'cell')
                                else:
                                    from ._mask_workers import _generate_masks_in_parallel
                                    _generate_masks_in_parallel(mask_src, settings, 'cell', gpu_plan)
                                stop = time.time()
                                duration = (stop - start)
                                time_ls.append(duration)
                                files_processed += 1
                                print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'cell_mask_gen')

                        if settings['nucleus_channel'] != None:
                            cancellation_checkpoint()
                            time_ls=[]
                            if check_mask_folder(
                                    src, 'nucleus_mask_stack',
                                    resume=settings.get('resume', False)):
                                start = time.time()
                                if gpu_plan is None:
                                    generate_cellpose_masks_sam(mask_src, settings, 'nucleus')
                                else:
                                    from ._mask_workers import _generate_masks_in_parallel
                                    _generate_masks_in_parallel(mask_src, settings, 'nucleus', gpu_plan)
                                stop = time.time()
                                duration = (stop - start)
                                time_ls.append(duration)
                                files_processed += 1
                                print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'nucleus_mask_gen')

                        if settings['pathogen_channel'] != None:
                            cancellation_checkpoint()
                            time_ls=[]
                            if check_mask_folder(
                                    src, 'pathogen_mask_stack',
                                    resume=settings.get('resume', False)):
                                start = time.time()
                                if gpu_plan is None:
                                    generate_cellpose_masks_sam(mask_src, settings, 'pathogen')
                                else:
                                    from ._mask_workers import _generate_masks_in_parallel
                                    _generate_masks_in_parallel(mask_src, settings, 'pathogen', gpu_plan)
                                stop = time.time()
                                duration = (stop - start)
                                time_ls.append(duration)
                                files_processed += 1
                                print_progress(files_processed, files_to_process, n_jobs=1, time_ls=time_ls, batch_size=None, operation_type=f'pathogen_mask_gen')

                        for organelle_role in organelle_roles:
                            cancellation_checkpoint()
                            time_ls=[]
                            if check_mask_folder(
                                    src, f'{organelle_role}_mask_stack',
                                    resume=settings.get('resume', False)):
                                start = time.time()
                                generate_organelle_masks_sam(
                                    mask_src, settings, organelle_role)
                                stop = time.time()
                                duration = (stop - start)
                                time_ls.append(duration)
                                files_processed += 1
                                print_progress(
                                    files_processed, files_to_process,
                                    n_jobs=1, time_ls=time_ls,
                                    batch_size=None,
                                    operation_type=f'{organelle_role}_mask_gen')

                        if settings.get('robustness_report'):
                            from .object import _run_robustness_report
                            for robust_role in ('cell', 'nucleus', 'pathogen'):
                                if settings.get(f'{robust_role}_channel') is not None:
                                    cancellation_checkpoint()
                                    _run_robustness_report(mask_src, settings, robust_role)

                        if settings.get('real_object_classifier') and not settings['timelapse']:
                            from .object_classifier import _drop_unreal_objects
                            cancellation_checkpoint()
                            _drop_unreal_objects(src, settings)

                        adjusted_cells = None
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
                                    from .object import _run_seg_qc
                                    if gpu_plan is None:
                                        adjust_cell_masks(parasite_folder, cell_folder, nuclei_folder, organelle_folder, overlap_threshold=5, perimeter_threshold=30, n_jobs=settings['n_jobs'])
                                        _run_seg_qc(mask_src, settings, 'cell')
                                    else:
                                        from ._mask_workers import _finalize_adjusted_cells
                                        adjusted_cells = _finalize_adjusted_cells(mask_src, organelle_folder, n_jobs=settings['n_jobs'])
                                        _run_seg_qc(mask_src, settings, 'cell', mask_folder=adjusted_cells)
                                    stop = time.time()
                                    adjust_time = (stop-start)/60
                                    print(f'Cell mask adjustment: {adjust_time} min.')

                        if os.path.exists(os.path.join(src,'measurements')):
                            _pivot_counts_table(db_path=os.path.join(src,'measurements', 'measurements.db'))

                        _load_and_concatenate_arrays(
                            src,
                            settings.get('channels'),
                            settings.get('cell_channel'),
                            settings.get('nucleus_channel'),
                            settings.get('pathogen_channel'),
                            settings.get('organelle_channel'),
                            organelle_chann_dims={
                                role: dim
                                for role in ORGANELLE_ROLES[1:]
                                if (dim := settings.get(
                                    f'{role}_channel')) is not None},
                            resume=settings.get('resume', False),
                            **({'mask_folders': {'cell': adjusted_cells}}
                               if adjusted_cells is not None else {})
                        )

                        if settings['timelapse'] and settings.get('motility_analysis', False):
                            cancellation_checkpoint()
                            from .timelapse import automated_motility_assay
                            automated_motility_assay(dict(
                                settings, src=src, reuse_existing_measurements=False))

                        if settings['timelapse'] and settings.get('timelapse_events'):
                            cancellation_checkpoint()
                            from .timelapse import _run_event_detection_step
                            _run_event_detection_step(src, settings)

                        if settings['plot']:
                            if not settings['timelapse']:
                                if settings['test_mode'] == True:
                                    merged_dir = os.path.join(src, 'merged')
                                    settings['examples_to_plot'] = len(
                                        [f for f in _listdir_visible(merged_dir)
                                         if f.endswith('.npy')]
                                    ) if os.path.isdir(merged_dir) else 0

                                plot_ledger = RunLedger('preprocess_generate_masks:overlay_plots')
                                try:
                                    merged_src = os.path.join(src,'merged')
                                    files = _overlay_candidates(merged_src)
                                except Exception as e:
                                    print(f'Failed to plot image mask overly. Error: {e}')
                                    plot_ledger.record_failure(os.path.join(src, 'merged'),
                                                               stage='list_merged', exc=e)
                                    files = []
                                else:
                                    random.shuffle(files)
                                time_ls = []

                                for i, file in enumerate(files):
                                    cancellation_checkpoint()
                                    start = time.time()
                                    if i+1 <= settings['examples_to_plot']:
                                        file_path = os.path.join(merged_src, file)

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
                                                save_pdf=True,
                                                outline_palette=settings.get(
                                                    'outline_palette',
                                                    'default'),
                                                organelle_channels={
                                                    role: settings.get(f'{role}_channel')
                                                    for role in ORGANELLE_ROLES[1:]
                                                    if settings.get(f'{role}_channel') is not None}
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

                    try:
                        from .filters import _write_object_relationships
                        _write_object_relationships(
                            src, timelapse=bool(settings.get('timelapse')))
                    except Exception as exc:
                        print(f"WARNING: could not write the object "
                              f"relationships table for {src}: "
                              f"{type(exc).__name__}: {exc}")

                    from .utils import cleanup_pipeline_folders
                    keep_intermediate = settings.get('keep_intermediate', False) and not settings.get('delete_intermediate', False)
                    keep_original = settings.get('keep_original_images', False) and not settings.get('delete_intermediate', False)
                    cleanup_pipeline_folders(src,
                                             keep_intermediate=keep_intermediate,
                                             keep_original=keep_original)

                    print("Successfully completed run")

        ledger.finalize()
        for source_folder in source_folders:
            db_path = os.path.join(format_path_for_system(source_folder),
                                   'measurements', 'measurements.db')
            if os.path.isfile(db_path):
                ledger.stamp(db_path)
                try:
                    from .filters import object_tables, write_relationships
                    if object_tables(db_path):
                        write_relationships(db_path)
                except Exception as exc:
                    print(f"WARNING: could not write the relationships "
                          f"table for {db_path}: "
                          f"{type(exc).__name__}: {exc}")

        from .artifacts import register_run_outputs
        register_run_outputs(
            module_key, settings, roots=source_folders, strict=False,
            run_id=run.run_id,
            status=(artifact_status.STATUS_COMPLETE if ledger.is_complete
                    else artifact_status.STATUS_PARTIAL))
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
        print("Timelapse module: settings['timelapse'] was False — forcing it "
              "to True. Use the Mask module for non-timelapse segmentation.")
    settings = get_timelapse_settings(settings)
    return preprocess_generate_masks(settings)


_WATCH_DIR = 'spacr_watch'
_WATCH_LEDGER = 'watch_ledger.json'
_WATCH_PIPELINES = ('mask', 'mask_measure', 'mask_measure_classify')
_WATCH_SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.nd2',
                   '.czi', '.lif')
_WATCH_KEY_GROUPS = ('plateID', 'wellID', 'timeID', 'fieldID')
_WATCH_OUTPUT_DIRS = frozenset({_WATCH_DIR, 'orig', 'stack', 'masks', 'merged',
                               'measurements', 'results', 'test'})


def _watch_truthy(value):
    """Read a settings switch that may arrive as a bool or as text.

    :param value: the stored value.
    :returns: True for True and for the words true, yes, on and 1.
    """
    if isinstance(value, str):
        return value.strip().lower() in ('true', 'yes', 'on', '1')
    return bool(value)


def _watch_number(settings, key, default, minimum=0.0):
    """Read a non-negative number from ``settings``.

    :param settings: the run settings.
    :param key: the settings key.
    :param default: used when the key is missing or blank.
    :param minimum: the smallest accepted value.
    :returns: the value as a float.
    :raises ValueError: naming the key, when the value is not a number or is
        below ``minimum``.
    """
    value = settings.get(key, default)
    if value is None or (isinstance(value, str) and not value.strip()):
        value = default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{key} must be a number, not {value!r}.') from None
    if number < minimum:
        raise ValueError(f'{key} must be at least {minimum:g}, not {number:g}.')
    return number


def _watch_source_channels(settings):
    """Channel IDs needed before selected positions are stable without a map.

    Only confirmed numeric conventions with a documented channel origin are
    accepted. Sorting source channels makes position ``n`` correspond to
    origin + ``n`` once every preceding identity is present. A custom pattern
    or channel name cannot establish that origin.

    :param settings: watch settings with zero-based selected positions.
    :returns: decimal channel IDs from the source origin through the highest
        selected position.
    :raises ValueError: when selected positions are invalid or the convention
        cannot prove a numeric channel origin.
    """
    import ast

    from .regex_infer import _metadata_zero_based

    selected = settings.get('channels', [0, 1, 2, 3])
    if isinstance(selected, str):
        try:
            selected = ast.literal_eval(selected)
        except (ValueError, SyntaxError):
            selected = None
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(type(value) is not int or value < 0 for value in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('watch_folder: channels must be distinct non-negative '
                         'zero-based positions.')
    convention = str(settings.get('metadata_type', 'cellvoyager'))
    known_numeric = {'cellvoyager', 'cq1', 'opera_phenix', 'imagexpress',
                     'arrayscan', 'arrayscan_kinetic', 'micromanager_mda',
                     'nikon_nis_xy'}
    if (settings.get('custom_regex') not in (None, '', 'None')
            or convention not in known_numeric):
        raise ValueError('watch_folder: this raw filename convention has no '
                         'documented numeric channel origin; provide a fixed '
                         'conversion_map.csv before starting the watch.')
    origin = 0 if 'chanID' in _metadata_zero_based(convention) else 1
    return {str(origin + position) for position in range(max(selected) + 1)}


def _watch_pattern(settings, extension, cache):
    """The compiled filename pattern the run groups files into fields with.

    ``custom_regex`` wins when it is set; otherwise the pattern of
    ``metadata_type`` for this extension. A convention without a pattern
    gives None, and every file then counts as one whole field.

    :param settings: the run settings.
    :param extension: the file extension without its dot.
    :param cache: a dict reused across calls so each pattern compiles once.
    :returns: a compiled pattern, or None.
    """
    import re

    if extension in cache:
        return cache[extension]
    from .regex_infer import _metadata_pattern

    custom = settings.get('custom_regex')
    try:
        if custom not in (None, '', 'None'):
            pattern = str(custom)
        else:
            pattern = _metadata_pattern(
                settings.get('metadata_type', 'cellvoyager'), extension)
        compiled = re.compile(pattern)
    except (KeyError, re.error, TypeError):
        compiled = None
    cache[extension] = compiled
    return compiled


def _watch_field_of(name, settings, cache):
    """Which field a file belongs to, and which channel it carries.

    :param name: the file name.
    :param settings: the run settings, for the filename pattern.
    :param cache: the pattern cache of :func:`_watch_pattern`.
    :returns: ``(field key, channel)``. The channel is None when the name
        carries none, and the file is then the whole field.
    """
    import re

    name = os.path.basename(name)
    stem, extension = os.path.splitext(name)
    pattern = _watch_pattern(settings, extension.lstrip('.').lower(), cache)
    match = pattern.match(name) if pattern is not None else None
    groups = match.groupdict() if match else {}
    channel = groups.get('chanID')
    parts = [str(groups[group]) for group in _WATCH_KEY_GROUPS
             if groups.get(group) not in (None, '')]
    if channel in (None, '') or not parts:
        parts, channel = [stem], None
    key = re.sub(r'[^A-Za-z0-9._-]+', '_', '_'.join(parts)).strip('._')
    return key or 'field', channel


def _watch_series_field_of(name, settings, cache):
    """Identify a mapped field series while retaining time in each image name.

    :param name: one acquired image basename or relative path.
    :param settings: the watch filename convention.
    :param cache: reusable compiled filename patterns.
    :returns: the series key, channel ID and time ID from the filename.
    :raises ValueError: when the configured convention cannot identify a series.
    """
    import re

    name = os.path.basename(name)
    extension = os.path.splitext(name)[1].lstrip('.').lower()
    pattern = _watch_pattern(settings, extension, cache)
    match = pattern.match(name) if pattern is not None else None
    groups = match.groupdict() if match else {}
    parts = [groups.get(key) for key in ('plateID', 'wellID', 'fieldID')]
    if any(part in (None, '') for part in parts) or not groups.get('timeID'):
        raise ValueError(f'watch filename settings do not identify the mapped series: {name}')
    key = re.sub(r'[^A-Za-z0-9._-]+', '_', '_'.join(map(str, parts))).strip('._')
    return key, groups.get('chanID'), groups.get('timeID')


def _watch_observed_field_of(name, settings, cache, series):
    """Group a seen image, leaving unrecognised mapped images incomplete.

    :param name: image name relative to the watched source folder.
    :param settings: watch filename convention.
    :param cache: reusable compiled filename patterns.
    :param series: whether the map declares whole timelapse fields.
    :returns: the observed field key and channel ID.
    """
    if series:
        try:
            key, channel, _time = _watch_series_field_of(name, settings, cache)
            return key, channel
        except ValueError:
            pass
    return _watch_field_of(name, settings, cache)


def _watch_map_bytes(src):
    """Read at most 16 MiB of the local Convert map without following links.

    :param src: acquisition directory containing the default Convert map.
    :returns: immutable file bytes, or None when no map exists.
    :raises ValueError: for oversized, nonregular or changing metadata.
    """
    import stat

    from .convert import MAP_FILENAME

    path = os.path.join(src, MAP_FILENAME)
    if not os.path.lexists(path):
        return None
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f'watch_folder: {path} must be a regular conversion map.')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f'watch_folder: cannot safely open conversion_map.csv: {exc}') from exc
    with os.fdopen(descriptor, 'rb') as handle:
        before = os.fstat(handle.fileno())
        current = os.stat(path, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(current.st_mode)
                or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)):
            raise ValueError('watch_folder: conversion_map.csv is not a stable regular file.')
        data = handle.read(16 * 1024 * 1024 + 1)
        after = os.fstat(handle.fileno())
    if len(data) > 16 * 1024 * 1024:
        raise ValueError('watch_folder: conversion_map.csv exceeds the 16 MiB limit.')
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('watch_folder: conversion_map.csv changed while being read.')
    return data


def _watch_validate_map_channels(settings, channels):
    """Refuse schemas that raw preprocessing would compact or reinterpret.

    Convert assigns channel IDs 1..N; preprocessing writes present planes in
    ascending channel order. Requiring the same dense set in every field keeps
    selected zero-based positions attached to the same declared channel IDs.

    :param settings: watch settings containing selected positional channels.
    :param channels: mapped field keys to sets of assigned channel IDs.
    :returns: None when each field has one shared dense schema and valid indices.
    :raises ValueError: for sparse/mixed fields or invalid selected positions.
    """
    import ast

    combined = set().union(*channels.values())
    expected = set(range(1, len(combined) + 1))
    if combined != expected or any(values != expected for values in channels.values()):
        raise ValueError('mapped fields must share the same complete C01..CN channel '
                         'schema; sparse or mixed fields would shift channel positions. '
                         'Use complete acquisitions or separate watch folders.')
    selected = settings.get('channels', [0])
    if isinstance(selected, str):
        try:
            selected = ast.literal_eval(selected)
        except (ValueError, SyntaxError):
            selected = None
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(type(value) is not int or value < 0 or value >= len(expected)
                   for value in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError(f'channels must be distinct zero-based positions in the '
                         f'mapped channel schema (0..{len(expected) - 1}).')


def _watch_map_manifest(src, settings):
    """Validate a fixed Convert map and bind exact target names to each field.

    Projected Z fields need a dense channel-by-plane grid. Mapped timelapses
    additionally need every frame in a dense channel-by-plane-by-time grid.

    :param src: acquisition directory containing converted images.
    :param settings: the watch filename convention and optional custom regex.
    :returns: field-to-target sets and the map SHA256, or two None values.
    :raises ValueError: when map identities are unsupported or ambiguous.
    """
    import csv
    import hashlib
    import io

    from .convert import _REQUIRED_MAP_COLUMNS, target_name

    data = _watch_map_bytes(src)
    if data is None:
        return None, None
    try:
        reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
        if not reader.fieldnames or not set(_REQUIRED_MAP_COLUMNS) <= set(reader.fieldnames):
            raise ValueError('required Convert map columns are missing')
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('duplicate column headers')
        groups, channels, planes, patterns, targets = {}, {}, {}, {}, set()
        series = _watch_truthy(settings.get('timelapse', False))
        for row in reader:
            name = row['target']
            if not name or name != os.path.basename(name) or '/' in name or '\\' in name:
                raise ValueError('targets must be plain output filenames')
            numbers = {key: int(row[key]) for key in ('field', 'channel', 'z', 't')}
            if any(value < 1 for value in numbers.values()):
                raise ValueError('output field/channel/z/t identifiers must be positive')
            expected = target_name(row['plate'], row['well'], **numbers)
            if name != expected or not row['source']:
                raise ValueError(f'target disagrees with its Convert metadata: {name}')
            if series:
                key, channel, time_id = _watch_series_field_of(name, settings, patterns)
                if not str(time_id).isdecimal() or int(time_id) != numbers['t']:
                    raise ValueError(f'watch filename settings do not identify the mapped time: {name}')
            else:
                if numbers['t'] != 1:
                    raise ValueError('mapped time series require timelapse=True')
                key, channel = _watch_field_of(name, settings, patterns)
            if channel is None or not str(channel).isdecimal() or int(channel) != numbers['channel']:
                raise ValueError(f'watch filename settings do not identify the mapped channel: {name}')
            plane = (numbers['channel'], numbers['z'], numbers['t'])
            if name in targets or plane in planes.setdefault(key, set()):
                raise ValueError(f'duplicate target or field channel/Z plane: {name}')
            targets.add(name)
            planes[key].add(plane)
            channels.setdefault(key, set()).add(numbers['channel'])
            groups.setdefault(key, set()).add(name)
        if not groups:
            raise ValueError('the conversion map has no output rows')
        _watch_validate_map_channels(settings, channels)
        for key, field_planes in planes.items():
            z_ids = {z for _channel, z, _time in field_planes}
            t_ids = {t for _channel, _z, t in field_planes}
            expected_planes = {(channel, z, t) for channel in channels[key]
                               for z in range(1, len(z_ids) + 1)
                               for t in range(1, len(t_ids) + 1)}
            if field_planes != expected_planes:
                raise ValueError(f'{key} must have a complete C01..CN by Z01..ZM '
                                 'by T0001..T grid; missing or sparse planes or '
                                 'frames would change the field series.')
    except (ValueError, TypeError, KeyError, UnicodeError, csv.Error) as exc:
        raise ValueError(f'watch_folder: invalid conversion_map.csv: {exc}') from exc
    return groups, hashlib.sha256(data).hexdigest()


def _watch_check_map(context):
    """Stop before processing if the bound map is changed, removed or introduced.

    :param context: watch state containing src and the initial map_sha256.
    :returns: None when the current metadata matches the initial snapshot.
    :raises ValueError: when that snapshot no longer describes the folder.
    """
    import hashlib

    data = _watch_map_bytes(context['src'])
    digest = hashlib.sha256(data).hexdigest() if data is not None else None
    if digest != context.get('map_sha256'):
        raise ValueError('watch_folder: conversion_map.csv changed since this watch '
                         'started; stop conversion and use a separate watch workspace '
                         'for a different map. Existing results are preserved.')


def _watch_unreadable(path):
    """Why an image cannot be read yet, or None when it reads whole.

    TIFFs are decoded in full and PNG, JPEG and BMP files are loaded, so a
    file whose writer has not finished fails here. Other formats are opened
    and read to their last byte.

    :param path: the image file.
    :returns: None, or the error text.
    """
    extension = os.path.splitext(path)[1].lower()
    try:
        if extension in ('.tif', '.tiff'):
            import tifffile

            if tifffile.imread(path).size == 0:
                return 'TIFF contains no readable image pixels'
        elif extension in ('.png', '.jpg', '.jpeg', '.bmp'):
            from PIL import Image

            with Image.open(path) as image:
                image.load()
        else:
            with open(path, 'rb') as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell():
                    handle.seek(-1, os.SEEK_END)
                    handle.read(1)
    except Exception as exc:
        return f'{type(exc).__name__}: {exc}'
    return None


def _watch_images(src):
    """List relative acquisition image paths without following output trees.

    Hidden entries, symbolic links and spaCR's generated folders are pruned.
    Relative paths keep nested file identity in the watch ledger; filename
    metadata is still parsed from the basename, not guessed from directories.

    :param src: the watched folder.
    :returns: relative image paths, sorted; root-level names stay unchanged.
    """
    from .cancellation import checkpoint

    names = []
    for directory, folders, files in os.walk(src, followlinks=False):
        checkpoint()
        folders[:] = sorted(folder for folder in folders
                            if not folder.startswith('.')
                            and folder.lower() not in _WATCH_OUTPUT_DIRS
                            and not folder.lower().endswith('_mask_stack')
                            and not os.path.islink(os.path.join(directory, folder)))
        for name in files:
            path = os.path.join(directory, name)
            if (not name.startswith('.') and name.lower().endswith(_WATCH_SUFFIXES)
                    and not os.path.islink(path) and os.path.isfile(path)):
                names.append(os.path.relpath(path, src))
    return sorted(names)


def _watch_load_ledger(path, src):
    """Read the watch record, or start an empty one.

    A record that cannot be parsed is moved aside to ``<path>.unreadable``
    and a new one is started.

    :param path: the ``watch_ledger.json`` path.
    :param src: the watched folder, stored in a new record.
    :returns: the record dict with a ``fields`` mapping.
    """
    import json

    try:
        with open(path, encoding='utf-8') as handle:
            ledger = json.load(handle)
    except FileNotFoundError:
        ledger = None
    except (OSError, ValueError) as exc:
        broken = f'{path}.unreadable'
        os.replace(path, broken)
        print(f'watch_folder: the record {path} could not be read ({exc}); '
              f'it was moved to {broken} and a new record was started.')
        ledger = None
    if not isinstance(ledger, dict) or not isinstance(ledger.get('fields'), dict):
        ledger = {'src': src, 'fields': {}}
    return ledger


def _watch_save_ledger(path, ledger):
    """Write the watch record atomically.

    :param path: the ``watch_ledger.json`` path.
    :param ledger: the record dict.
    """
    import json

    partial = f'{path}.partial'
    with open(partial, 'w', encoding='utf-8') as handle:
        json.dump(ledger, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _watch_link(source, target):
    """Hard-link ``source`` to ``target``, copying when a link is refused.

    :param source: the existing file.
    :param target: the new path.
    """
    import shutil

    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _watch_measure_settings(settings):
    """The Measure settings a watch run measures every field with.

    :param settings: the watch run settings. ``watch_measure_settings`` names
        a saved Measure settings file; blank uses Measure's defaults with the
        run's ``channels``.
    :returns: the Measure settings dict, without ``src``. Measure fills in
        its own defaults, and the mask planes from the field's merged layout.
    """
    path = str(settings.get('watch_measure_settings') or '').strip()
    measure = {}
    if path:
        from .cli import load_settings_file

        measure = dict(load_settings_file(os.path.expanduser(path)))
    else:
        measure['channels'] = settings.get('channels')
    if _watch_truthy(settings.get('timelapse', False)):
        if 'timelapse' in measure and not _watch_truthy(measure['timelapse']):
            raise ValueError('watch_folder: Measure settings must enable timelapse '
                             'for a mapped field series.')
        measure['timelapse'] = True
    measure.pop('src', None)
    return measure


def _watch_analyse_field(field_dir, settings):
    """Run the chosen pipeline on one field's folder.

    Make Masks runs on ``field_dir`` exactly as a batch run would on a plate
    folder holding only this field. The later pipeline choices run Measure,
    then optionally apply a pretrained CV model to its measured objects.

    :param field_dir: a folder holding the field's image files.
    :param settings: the watch run settings.
    :raises RuntimeError: when a step leaves no output behind.
    """
    run = {key: value for key, value in settings.items()
           if not str(key).startswith('watch_')}
    run.update(src=field_dir, consolidate=False, dry_run=False, test_mode=False)
    preprocess_generate_masks(run)
    merged = os.path.join(field_dir, 'merged')
    if not os.path.isdir(merged) or not _overlay_candidates(merged):
        raise RuntimeError('Make Masks wrote no merged stack for this field; '
                           'the log above says why.')
    pipeline = str(settings.get('watch_pipeline') or 'mask')
    if pipeline == 'mask':
        return
    from .measure import measure_crop

    from copy import deepcopy

    measure = (deepcopy(settings['watch_measure_snapshot'])
               if 'watch_measure_snapshot' in settings
               else _watch_measure_settings(settings))
    measure['src'] = merged
    measure_crop(measure)
    if not os.path.exists(os.path.join(field_dir, 'measurements',
                                       'measurements.db')):
        raise RuntimeError('Measure wrote no measurements.db for this field; '
                           'the log above says why.')
    if pipeline == 'mask_measure_classify':
        from .classify import classify
        from contextlib import closing

        from .database_concurrency import connect

        classify_settings = deepcopy(settings['watch_classify_snapshot'])
        classify_settings['src'] = field_dir
        classify(classify_settings)
        database = os.path.join(field_dir, 'measurements', 'measurements.db')
        with closing(connect(database, readonly=True)) as connection:
            columns = {row[1] for row in connection.execute('PRAGMA table_info(png_list)')}
            if (not {'pred', 'cv_predictions'} <= columns or not connection.execute(
                    'SELECT 1 FROM png_list WHERE pred IS NOT NULL AND '
                    'cv_predictions IS NOT NULL LIMIT 1').fetchone()):
                raise RuntimeError('Classify wrote no CV predictions to this field; '
                                   'check the model and measured objects.')


def _watch_measure_recipe(settings):
    """Capture Measure input settings and hash their canonical JSON content.

    The source folder is excluded by the existing settings loader. Each field
    receives its own copy, so later file edits or downstream mutations cannot
    change the recipe of an active watch. Only the digest is stored in the
    ledger; this avoids duplicating potentially sensitive settings values.

    :param settings: watch settings naming a Measure file or default channels.
    :returns: independent settings dictionary and SHA256 of its JSON content.
    :raises ValueError: when settings cannot be represented as finite JSON.
    """
    import hashlib
    import json
    from copy import deepcopy

    recipe = deepcopy(_watch_measure_settings(settings))
    try:
        encoded = json.dumps(recipe, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ValueError('watch_folder: Measure settings must contain finite '
                         'JSON-compatible values for a reproducible recipe.') from exc
    return recipe, hashlib.sha256(encoded).hexdigest()


def _watch_mask_recipe(settings):
    """Freeze the effective Mask inputs and fingerprint them for watch resumes.

    Watch timing and source paths do not change the per-field Mask recipe.
    The three output/control switches below are forced by the field adapter,
    so their caller-supplied values do not change its result either.
    """
    import hashlib
    import json
    from copy import deepcopy

    recipe = {key: value for key, value in settings.items()
              if not str(key).startswith('watch_') and key != 'src'}
    recipe.update(consolidate=False, dry_run=False, test_mode=False)
    recipe = deepcopy(recipe)
    try:
        encoded = json.dumps(recipe, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ValueError('watch_folder: Mask settings must contain finite '
                         'JSON-compatible values for a reproducible recipe.') from exc
    return recipe, hashlib.sha256(encoded).hexdigest()


def _watch_classify_recipe(settings):
    """Freeze a saved CV inference recipe and bind its model file by content.

    Watch fields are independent, so fitting an ML or CV model separately on
    each incoming field would not reproduce one fitted plate model. This path
    accepts inference from a previously trained CV checkpoint only.
    """
    import hashlib
    import json
    from copy import deepcopy

    from .cli import load_settings_file

    path = str(settings.get('watch_classify_settings') or '').strip()
    if not path:
        raise ValueError('watch_folder: watch_classify_settings must name a '
                         'saved Classify settings file for CV inference.')
    recipe = deepcopy(dict(load_settings_file(os.path.expanduser(path))))
    if str(recipe.get('classifier_family') or 'cv').strip().lower() != 'cv':
        raise ValueError('watch_folder: live Classify requires the CV family '
                         'with a previously trained model.')
    if (any(_watch_truthy(recipe.get(key, False)) for key in
            ('train', 'test', 'generate_training_dataset')) or
            not _watch_truthy(recipe.get('apply_model_to_dataset', False)) or
            str(recipe.get('crop_source') or '').strip().lower() != 'merged'):
        raise ValueError('watch_folder: live Classify requires train, test and '
                         'generate_training_dataset off, apply_model_to_dataset '
                         'on and crop_source merged.')
    source = os.path.abspath(os.path.expanduser(str(recipe.get('model_path') or '')))
    if _watch_file_identity(source) is None:
        raise ValueError('watch_folder: Classify model_path must name a regular '
                         'pretrained checkpoint file.')
    model_sha256 = _watch_artifact_sha256(source)
    recipe.pop('src', None)
    recipe['classifier_family'] = 'cv'
    recipe['train'] = False
    recipe['test'] = False
    recipe['generate_training_dataset'] = False
    recipe['apply_model_to_dataset'] = True
    recipe['crop_source'] = 'merged'
    recipe['tar_path'] = ''
    recipe['model_path'] = source
    fingerprint = {**recipe, 'model_path': model_sha256}
    try:
        encoded = json.dumps(fingerprint, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ValueError('watch_folder: Classify settings must contain finite '
                         'JSON-compatible values for a reproducible recipe.') from exc
    return recipe, hashlib.sha256(encoded).hexdigest(), model_sha256


def _watch_classify_model_snapshot(work, source, digest):
    """Keep the verified checkpoint immutable for every field in this watch."""
    folder = os.path.join(work, '.watch_classify')
    target = os.path.join(folder, f'{digest}.pt')
    if os.path.lexists(target):
        if _watch_artifact_sha256(target) != digest:
            raise ValueError('watch_folder: the saved Classify model snapshot '
                             'changed; existing results are preserved.')
        return target
    os.makedirs(folder, exist_ok=True)
    partial = target + '.partial'
    if os.path.lexists(partial):
        os.unlink(partial)
    expected = _watch_file_identity(source)
    try:
        copied = _watch_copy_snapshot(source, partial, expected)
        if copied != digest:
            raise ValueError('watch_folder: the Classify model changed while '
                             'being copied; existing results are preserved.')
        os.replace(partial, target)
    finally:
        if os.path.lexists(partial):
            os.unlink(partial)
    return target


def _watch_quote(name):
    """Quote a table or column name for SQLite.

    :param name: the name.
    :returns: the name in double quotes, inner quotes doubled.
    """
    return '"' + str(name).replace('"', '""') + '"'


def _watch_merge_database(field_db, combined_db, key):
    """Append one field's measurement tables to the combined database once.

    Every table of ``field_db`` is created in ``combined_db`` when missing,
    widened by any column it lacks, and given the field's rows. The field is
    recorded in the ``spacr_watch_fields`` table in the same transaction, so
    a field is appended exactly once even when the watch stops between the
    append and its record.

    :param field_db: the field's ``measurements.db``.
    :param combined_db: the combined ``measurements.db``.
    :param key: the field key recorded with the rows.
    :returns: True when rows were appended, False when the field was there.
    """
    import sqlite3

    quote = _watch_quote
    os.makedirs(os.path.dirname(combined_db), exist_ok=True)
    connection = sqlite3.connect(combined_db, timeout=30, isolation_level=None)
    try:
        connection.execute('CREATE TABLE IF NOT EXISTS spacr_watch_fields '
                           '(field TEXT PRIMARY KEY, merged_at REAL)')
        if connection.execute('SELECT 1 FROM spacr_watch_fields WHERE field = ?',
                              (key,)).fetchone():
            return False
        connection.execute('ATTACH DATABASE ? AS field_db', (field_db,))
        try:
            connection.execute('BEGIN IMMEDIATE')
            try:
                tables = connection.execute(
                    "SELECT name, sql FROM field_db.sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY name").fetchall()
                for name, sql in tables:
                    columns = [row[1] for row in connection.execute(
                        f'PRAGMA field_db.table_info({quote(name)})')]
                    present = [row[1] for row in connection.execute(
                        f'PRAGMA main.table_info({quote(name)})')]
                    if not present:
                        connection.execute(sql)
                    for column in columns:
                        if present and column not in present:
                            connection.execute(
                                f'ALTER TABLE main.{quote(name)} '
                                f'ADD COLUMN {quote(column)}')
                    listed = ', '.join(quote(column) for column in columns)
                    connection.execute(
                        f'INSERT OR IGNORE INTO main.{quote(name)} ({listed}) '
                        f'SELECT {listed} FROM field_db.{quote(name)}')
                connection.execute(
                    'INSERT INTO spacr_watch_fields VALUES (?, ?)',
                    (key, time.time()))
                connection.execute('COMMIT')
            except BaseException:
                connection.execute('ROLLBACK')
                raise
        finally:
            connection.execute('DETACH DATABASE field_db')
    finally:
        connection.close()
    return True


def _watch_collect(field_dir, work, key, database_snapshot=None):
    """Gather one analysed field into the watch folder's combined outputs.

    The field's merged files and flat track CSVs are linked into their
    combined folders, and its measurement tables are appended to the
    combined database. The result has the layout of an analysed batch plate.

    :param field_dir: the analysed field's folder.
    :param work: the ``spacr_watch`` folder.
    :param key: the field key.
    :param database_snapshot: optional closed SQLite snapshot captured after analysis.
    """
    merged = os.path.join(field_dir, 'merged')
    if os.path.isdir(merged):
        target = os.path.join(work, 'merged')
        os.makedirs(target, exist_ok=True)
        for name in sorted(os.listdir(merged)):
            source = os.path.join(merged, name)
            destination = os.path.join(target, name)
            if os.path.isfile(source) and not os.path.exists(destination):
                _watch_link(source, destination)
    tracks = os.path.join(field_dir, 'tracks')
    if os.path.isdir(tracks):
        target = os.path.join(work, 'tracks')
        os.makedirs(target, exist_ok=True)
        for name in sorted(os.listdir(tracks)):
            if not name.endswith('.csv'):
                continue
            source = os.path.join(tracks, name)
            destination = os.path.join(target, name)
            if os.path.isfile(source) and not os.path.exists(destination):
                _watch_link(source, destination)
    field_db = database_snapshot or os.path.join(field_dir, 'measurements', 'measurements.db')
    if database_snapshot is not None and not os.path.isfile(field_db):
        raise ValueError('Collection database snapshot disappeared before append.')
    if os.path.exists(field_db):
        _watch_merge_database(
            field_db, os.path.join(work, 'measurements', 'measurements.db'),
            key)


def _watch_artifact_sha256(path):
    """Hash one stable regular output with cancellable, bounded reads.

    :param path: staged or combined artifact; final-component links are refused.
    :returns: SHA256 of verified bytes.
    :raises ValueError: when an artifact is missing, nonregular or changes.
    """
    import hashlib
    from .cancellation import checkpoint

    identity = _watch_file_identity(path)
    if identity is None:
        raise ValueError(f'Collection artifact is missing or unsafe: {path}')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f'Collection artifact cannot be opened: {path}') from exc
    with os.fdopen(descriptor, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns] != identity:
            raise ValueError(f'Collection artifact changed before hashing: {path}')
        digest, size = hashlib.sha256(), 0
        while True:
            checkpoint()
            chunk = os.read(handle.fileno(), min(1024 * 1024, identity[2] - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > identity[2]:
                raise ValueError(f'Collection artifact grew while hashing: {path}')
            digest.update(chunk)
        if size != identity[2] or _watch_file_identity(path) != identity:
            raise ValueError(f'Collection artifact changed while hashing: {path}')
    return digest.hexdigest()


def _watch_collection_artifacts(field_dir):
    """Fingerprint merged, flat track CSV and SQLite outputs to be collected.

    :param field_dir: completed field staging directory.
    :returns: relative artifact paths mapped to SHA256 values.
    :raises ValueError: when no usable outputs exist or an output is unsafe.
    """
    artifacts = {}
    merged = os.path.join(field_dir, 'merged')
    if os.path.isdir(merged):
        for name in sorted(os.listdir(merged)):
            relative = os.path.join('merged', name)
            artifacts[relative] = _watch_artifact_sha256(os.path.join(field_dir, relative))
    tracks = os.path.join(field_dir, 'tracks')
    if os.path.lexists(tracks) and (os.path.islink(tracks) or not os.path.isdir(tracks)):
        raise ValueError('Collection tracks output must be a regular directory.')
    if os.path.isdir(tracks):
        for name in sorted(os.listdir(tracks)):
            if name.endswith('.csv'):
                relative = os.path.join('tracks', name)
                artifacts[relative] = _watch_artifact_sha256(os.path.join(field_dir, relative))
    relative = os.path.join('.watch_collection', 'measurements.db')
    if os.path.lexists(os.path.join(field_dir, relative)):
        for suffix in ('-wal', '-journal', '-shm'):
            if os.path.lexists(os.path.join(field_dir, relative + suffix)):
                raise ValueError('Collection requires a closed, checkpointed SQLite '
                                 f'database without sidecars: {relative + suffix}')
        artifacts[relative] = _watch_artifact_sha256(os.path.join(field_dir, relative))
    if not artifacts:
        raise ValueError('Collection has no completed analysis artifacts to preserve.')
    return artifacts


def _watch_backup_progress(status, remaining, total):
    """Allow Stop between bounded SQLite backup page batches.

    :param status: SQLite status code supplied by the backup API.
    :param remaining: pages remaining in the source snapshot.
    :param total: source page count reported by SQLite.
    :returns: None; cancellation interrupts backup without publishing a snapshot.
    """
    from .cancellation import checkpoint

    checkpoint()


def _watch_snapshot_database(field_dir):
    """Capture committed rows, including WAL contents, into a closed private DB.

    :param field_dir: completed field staging directory.
    :returns: None; publishes a new snapshot only after backup and close succeed.
    :raises OSError: when the owned destination cannot be created or published.
    :raises sqlite3.Error: when the source cannot be backed up consistently.
    """
    import sqlite3
    from pathlib import Path

    source = Path(field_dir) / 'measurements' / 'measurements.db'
    if not source.exists():
        return
    if _watch_file_identity(source) is None:
        raise ValueError('Collection producer database is not a regular file.')
    folder = Path(field_dir) / '.watch_collection'
    folder.mkdir()
    partial = folder / 'measurements.db.partial'
    incoming = sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)
    try:
        outgoing = sqlite3.connect(partial, timeout=30.0)
        try:
            incoming.backup(outgoing, pages=256, progress=_watch_backup_progress, sleep=0.05)
            outgoing.execute('PRAGMA journal_mode=DELETE').fetchone()
        finally:
            outgoing.close()
    finally:
        incoming.close()
    os.replace(partial, folder / 'measurements.db')


def _watch_validate_collection(field_dir, work, saved, *, verify_staged):
    """Refuse altered checkpoints or conflicting combined files before writes.

    :param field_dir: completed field staging directory.
    :param work: shared watch output directory.
    :param saved: checkpoint artifact path/hash mapping.
    :param verify_staged: True on restart; False just after fingerprinting outputs.
    :returns: None when collection can safely continue.
    :raises ValueError: for missing/changed outputs or conflicting combined files.
    """
    if verify_staged and _watch_collection_artifacts(field_dir) != saved:
        raise ValueError('Collection checkpoint artifacts changed; preserved outputs '
                         'must be recovered before resuming this workspace.')
    tracks_target = os.path.join(work, 'tracks')
    if (any(os.path.dirname(relative) == 'tracks' for relative in saved)
            and os.path.lexists(tracks_target)
            and (os.path.islink(tracks_target) or not os.path.isdir(tracks_target))):
        raise ValueError('Collection tracks destination is not a regular directory.')
    for relative, digest in saved.items():
        if os.path.dirname(relative) not in ('merged', 'tracks'):
            continue
        destination = os.path.join(work, relative)
        if os.path.lexists(destination) and _watch_artifact_sha256(destination) != digest:
            raise ValueError(f'Collection conflicts with an existing combined artifact: {destination}')


def _watch_status_line(ledger, waiting):
    """The progress line the GUI reads, counting the record's fields.

    :param ledger: the watch record.
    :param waiting: fields seen but not analysed yet.
    :returns: one line of text.
    """
    states = [entry.get('status') for entry in ledger['fields'].values()]
    return (f"watch_folder: {states.count('done')} analysed, {waiting} "
            f"waiting, {states.count('failed')} failed")


def _watch_check_settings(settings):
    """Resolve and check what a watch run needs before it starts.

    :param settings: the watch run settings.
    :returns: ``(src, pipeline, settle seconds, poll seconds, idle seconds)``.
    :raises ValueError: for a list of folders, a missing folder, an unknown
        ``watch_pipeline``, a bad number or a z-stack or t-stack run.
        Timelapse requires a complete fixed Convert map, checked later.
    """
    from .utils import normalize_src_path

    src = normalize_src_path(settings.get('src'))
    if isinstance(src, list):
        if len(src) != 1:
            raise ValueError('watch_folder watches one folder; src names '
                             f'{len(src)}.')
        src = src[0]
    src = os.path.abspath(os.path.expanduser(str(src)))
    if not os.path.isdir(src):
        raise ValueError(f'watch_folder: the folder {src} does not exist.')
    for key in ('z_stack', 't_stack'):
        if _watch_truthy(settings.get(key, False)):
            raise ValueError(
                f'watch_folder does not support {key} runs: a field is '
                f'analysed as soon as its channels are in, before later '
                f'planes or frames arrive.')
    pipeline = str(settings.get('watch_pipeline') or 'mask')
    if pipeline not in _WATCH_PIPELINES:
        raise ValueError(f'watch_pipeline must be one of {_WATCH_PIPELINES}, '
                         f'not {pipeline!r}.')
    settle = _watch_number(settings, 'watch_settle_seconds', 10.0)
    poll = _watch_number(settings, 'watch_poll_seconds', 5.0, minimum=0.01)
    idle = _watch_number(settings, 'watch_idle_minutes', 0.0) * 60.0
    if _watch_truthy(settings.get('microscope_feedback', False)):
        if pipeline not in ('mask_measure', 'mask_measure_classify'):
            raise ValueError("microscope_feedback picks events from the "
                             "measurements; set watch_pipeline to "
                             "'mask_measure'.")
        for key, default in (('microscope_max_events', 10.0),
                             ('microscope_timepoints', 1.0),
                             ('microscope_interval_seconds', 0.0)):
            _watch_number(settings, key, default)
    return src, pipeline, settle, poll, idle


def _watch_file_identity(path):
    """Return regular-file identity or None for missing, linked or unsafe input.

    :param path: acquired image path; the final path component is not followed.
    :returns: device, inode, size, mtime_ns and ctime_ns as a JSON-safe list.
    """
    import stat

    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _watch_copy_snapshot(source, target, expected):
    """Copy one unchanged acquired file in cancellable 1 MiB chunks.

    :param source: original acquired file, opened read-only without link following.
    :param target: new path inside this field's private staging directory.
    :param expected: identity recorded when the image was last observed.
    :returns: SHA256 of copied bytes, or None when source identity changed.
    :raises OSError: for destination write errors; pipeline failures remain failures.
    :raises spacr.cancellation.PipelineCancelled: when Stop interrupts copying.
    """
    import hashlib
    import stat

    from .cancellation import checkpoint

    if expected is None or _watch_file_identity(source) != expected:
        return None
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    try:
        descriptor = os.open(source, flags)
    except OSError:
        return None
    with os.fdopen(descriptor, 'rb') as incoming:
        before = os.fstat(incoming.fileno())
        identity = [before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns]
        if not stat.S_ISREG(before.st_mode) or identity != expected:
            return None
        digest = hashlib.sha256()
        copied = 0
        with open(target, 'xb') as outgoing:
            while True:
                checkpoint()
                try:
                    chunk = os.read(incoming.fileno(), min(1024 * 1024, expected[2] - copied + 1))
                except OSError:
                    return None
                if not chunk:
                    break
                copied += len(chunk)
                if copied > expected[2]:
                    return None
                outgoing.write(chunk)
                digest.update(chunk)
        after = os.fstat(incoming.fileno())
        identity = [after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns]
        if identity != expected or _watch_file_identity(source) != expected:
            return None
    if os.path.getsize(target) != expected[2]:
        return None
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    return digest.hexdigest()


def _watch_defer_snapshot(key, members, field_dir, context):
    """Discard only unanalysed staging and wait for acquired files to settle again.

    :param key: scientific field identity.
    :param members: acquired filename/channel pairs of the attempted snapshot.
    :param field_dir: private directory created for this attempt.
    :param context: mutable watcher observations and ledger.
    :returns: None; analysis retries after a fresh observation and settle interval.
    """
    import shutil

    shutil.rmtree(field_dir)
    now = time.time()
    for name, _channel in members:
        if name in context['seen']:
            context['seen'][name].update(signature=None, identity=None,
                                         changed=now, readable=False)
    context['ledger']['fields'][key].update(
        status='waiting', finished=now,
        error='Source files changed while preparing the field snapshot; waiting again.')
    _watch_save_ledger(context['ledger_path'], context['ledger'])
    print(f'watch_folder: {key} changed during snapshot copying; staging was '
          f'discarded and the field will wait for stable files again.')


def _watch_run_field(key, members, signature, context):
    """Analyse one ready field and record the outcome.

    :param key: the field key.
    :param members: ``(file name, channel)`` pairs of the field.
    :param signature: ``{file name: [size, mtime_ns]}`` of those files.
    :param context: the watch state: ``src``, ``work``, ``ledger``,
        ``ledger_path``, ``seen``, ``tried``, ``analyse`` and ``settings``.
    :raises spacr.cancellation.PipelineCancelled: when Stop was pressed
        during the field; it is recorded as interrupted first.
    """
    import shutil

    from .cancellation import PipelineCancelled

    _watch_check_map(context)
    seen, ledger = context['seen'], context['ledger']
    arrived = max(seen[name]['changed'] for name, _channel in members)
    entry = ledger['fields'].setdefault(key, {})
    saved_collection = entry.get('collection_checkpoint')
    if saved_collection is None:
        entry.pop('snapshot_sha256', None)
        entry.pop('source_identity', None)
        entry.update(status='running', files=signature,
                     first_seen=min(seen[name]['first'] for name, _c in members),
                     stable_since=arrived, started=time.time(), error=None)
    else:
        arrived = entry['stable_since']
        entry.update(status='running', error=None)
    _watch_save_ledger(context['ledger_path'], ledger)
    field_dir = os.path.join(context['work'], 'fields', key)
    try:
        identities = {name: seen[name].get('identity') for name, _channel in members}
        if saved_collection is not None:
            if entry.get('files') != signature or entry.get('source_identity') != identities:
                raise ValueError('Collection checkpoint inputs changed; preserved outputs '
                                 'cannot be combined with a new analysis attempt.')
            print(f'watch_folder: resuming collection for {key} without reanalysing it.')
        else:
            if os.path.exists(field_dir):
                shutil.rmtree(field_dir)
            os.makedirs(field_dir)
            print(f'watch_folder: analysing {key} ({len(members)} file(s)).')
            snapshots = {}
            for name, _channel in members:
                digest = _watch_copy_snapshot(
                    os.path.join(context['src'], name),
                    os.path.join(field_dir, os.path.basename(name)), identities[name])
                if digest is None:
                    _watch_defer_snapshot(key, members, field_dir, context)
                    return
                snapshots[name] = digest
            if any(_watch_file_identity(os.path.join(context['src'], name)) != identities[name]
                   for name, _channel in members):
                _watch_defer_snapshot(key, members, field_dir, context)
                return
            _watch_check_map(context)
            entry.update(snapshot_sha256=snapshots, source_identity=identities)
            _watch_save_ledger(context['ledger_path'], ledger)
            context['analyse'](field_dir, context['settings'])
            _watch_snapshot_database(field_dir)
            entry['collection_checkpoint'] = {
                'artifacts': _watch_collection_artifacts(field_dir),
                'analysis_seconds': time.time() - entry['started']}
            _watch_save_ledger(context['ledger_path'], ledger)
        collection_started = time.time()
        _watch_validate_collection(
            field_dir, context['work'], entry['collection_checkpoint']['artifacts'],
            verify_staged=saved_collection is not None)
        _watch_check_map(context)
        database_relative = os.path.join('.watch_collection', 'measurements.db')
        database_snapshot = (os.path.join(field_dir, database_relative)
                             if database_relative in entry['collection_checkpoint']['artifacts']
                             else None)
        _watch_collect(field_dir, context['work'], key, database_snapshot=database_snapshot)
    except PipelineCancelled:
        entry.update(status='interrupted', finished=time.time())
        _watch_save_ledger(context['ledger_path'], ledger)
        raise
    except Exception as exc:
        entry.update(status='failed', finished=time.time(),
                     error=f'{type(exc).__name__}: {exc}')
        context['tried'].add((key, repr(sorted(signature.items()))))
        _watch_save_ledger(context['ledger_path'], ledger)
        print(f'watch_folder: ERROR {key} failed: {type(exc).__name__}: {exc}')
        return
    finished = time.time()
    entry.update(status='done', finished=finished,
                 seconds=round(entry['collection_checkpoint']['analysis_seconds']
                               + finished - collection_started, 3),
                 waited=round(entry['started'] - arrived, 3))
    _watch_save_ledger(context['ledger_path'], ledger)
    print(f'watch_folder: analysed {key} in {entry["seconds"]:.1f} s, taken '
          f'{entry["waited"]:.1f} s after its last file stopped changing.')
    if context.get('microscope') is not None:
        _microscope_feedback(key, field_dir, context)


def _watch_ready_fields(context, now):
    """Sort the files seen so far into fields and pick the ready ones.

    :param context: the watch state of :func:`_watch_folder_and_analyse`.
    :param now: the current time.
    :returns: ``(ready, waiting)``: the ready fields as
        ``(first seen, key, members, signature)`` tuples, and how many fields
        are seen but not analysed.

    Convert binds exact basenames, including every declared Z plane; relative
    folders are provenance only. Split companions and duplicate locations are
    still rejected. Raw numeric conventions also wait for every channel ID
    through the highest selected position.
    """
    seen, fields = context['seen'], context['ledger']['fields']
    groups = {}
    for name in seen:
        key, channel = _watch_observed_field_of(
            name, context['settings'], context['patterns'], context.get('series'))
        groups.setdefault(key, []).append((name, channel))
    manifest = context.get('manifest')
    if manifest is not None:
        for key in manifest:
            groups.setdefault(key, [])
    ready = []
    waiting = sum(entry.get('status') == 'waiting' and key not in groups
                  for key, entry in fields.items())
    for key, members in sorted(groups.items()):
        if manifest is not None and (key not in manifest or
                {os.path.basename(name) for name, _channel in members} != manifest[key]):
            waiting += 1
            warning = ('manifest', key, tuple(sorted(name for name, _c in members)))
            if warning not in context['warned']:
                context['warned'].add(warning)
                print(f'watch_folder: {key} does not yet match its exact conversion-map '
                      f'companions; missing or unexpected files remain unprocessed.')
            continue
        if manifest is None and all(channel is not None for _name, channel in members):
            origin = context['source_origin']
            if any(not str(channel).isdecimal() or int(channel) < origin
                   for _name, channel in members):
                raise ValueError(f'watch_folder: {key} has a channel ID below '
                                 'the documented origin or a nonnumeric channel; '
                                 'use the correct filename convention or a fixed '
                                 'conversion_map.csv. Existing results are preserved.')
        parents = {os.path.dirname(name) for name, _channel in members}
        channels = {str(int(channel)) if str(channel).isdecimal() else channel
                    for _name, channel in members}
        ambiguous = (len(parents) != 1 or
                     (manifest is None and len(channels) != len(members)))
        if ambiguous:
            waiting += 1
            warning = ('ambiguous', key, tuple(sorted(name for name, _c in members)))
            if warning not in context['warned']:
                context['warned'].add(warning)
                print(f'watch_folder: {key} has files in different acquisition '
                      f'folders or duplicate channels; waiting without combining '
                      f'them. Use unique filename field identifiers for separate '
                      f'acquisitions: {", ".join(name for name, _c in members)}')
            continue
        entry = fields.get(key, {})
        signature = {name: list(seen[name]['signature'])
                     for name, _channel in members}
        if entry.get('status') == 'done':
            previous_identity = entry.get('source_identity')
            identity_changed = isinstance(previous_identity, dict) and {
                name: seen[name].get('identity') for name, _channel in members
            } != previous_identity
            if ((signature != entry.get('files') or identity_changed)
                    and key not in context['warned']):
                context['warned'].add(key)
                print(f'watch_folder: {key} changed after it was analysed; it '
                      f'is not analysed again. Remove its entry from '
                      f'{context["ledger_path"]} to analyse it again.')
            continue
        if (key, repr(sorted(signature.items()))) in context['tried']:
            continue
        waiting += 1
        if manifest is None and None not in channels:
            if not context['source_channels'] <= channels:
                continue
        if any(now - seen[name]['changed'] < context['settle']
               for name, _channel in members):
            continue
        unreadable = None
        for name, _channel in members:
            if seen[name]['readable']:
                continue
            unreadable = _watch_unreadable(os.path.join(context['src'], name))
            if unreadable is not None:
                seen[name]['changed'] = now
                print(f'watch_folder: {name} cannot be read yet '
                      f'({unreadable}); waiting.')
                break
            seen[name]['readable'] = True
        if unreadable is None:
            ready.append((min(seen[name]['first'] for name, _c in members),
                          key, members, signature))
    return sorted(ready), waiting


def _watch_observe(context, now):
    """Record every image file's size and modification time.

    :param context: the watch state of :func:`_watch_folder_and_analyse`.
    :param now: the current time.
    :returns: True when a file appeared, changed or went away.
    """
    seen = context['seen']
    names = _watch_images(context['src'])
    changed = False
    for name in names:
        identity = _watch_file_identity(os.path.join(context['src'], name))
        if identity is None:
            if name in seen:
                del seen[name]
                changed = True
            continue
        signature = tuple(identity[2:4])
        record = seen.get(name)
        if (record is None or record['signature'] != signature
                or record.get('identity') != identity):
            seen[name] = {'signature': signature, 'identity': identity, 'changed': now,
                          'first': (record or {}).get('first', now),
                          'readable': False}
            changed = True
    for name in [name for name in seen if name not in names]:
        del seen[name]
        changed = True
    return changed


_MICROSCOPE_DRIVERS = ('simulated', 'pycromanager')
_MICROSCOPE_REIMAGED = 'reimaged'


def _microscope_matrix(settings):
    """The 2x2 matrix that turns a pixel offset into a stage offset.

    :param settings: the run settings; ``microscope_stage_transform`` holds
        ``[a, b, c, d]``, a list or its text, so that a pixel offset of
        ``dx`` columns and ``dy`` rows moves the stage by
        ``(a*dx + b*dy, c*dx + d*dy)`` micrometres.
    :returns: a float array of shape (2, 2).
    :raises ValueError: when the value is not four numbers or the matrix
        cannot be inverted.
    """
    import ast

    value = settings.get('microscope_stage_transform', [1.0, 0.0, 0.0, 1.0])
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = None
    try:
        matrix = np.asarray(value, dtype=float).reshape(2, 2)
    except (TypeError, ValueError):
        raise ValueError('microscope_stage_transform must be four numbers '
                         f'[a, b, c, d], not {value!r}.') from None
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('microscope_stage_transform must be an invertible '
                         f'matrix, not {value!r}.')
    return matrix


def _microscope_stage_position(pixel, shape, centre, matrix):
    """Where the stage must go to centre the camera on one pixel of a field.

    :param pixel: ``(row, column)`` of the point in the field image.
    :param shape: ``(height, width)`` of the field image.
    :param centre: ``(x, y)`` stage position, in micrometres, at which the
        field was acquired, which is where the image centre lies.
    :param matrix: the matrix of :func:`_microscope_matrix`.
    :returns: ``(x, y)`` stage position in micrometres.
    """
    offset = np.array([float(pixel[1]) - (shape[1] - 1) / 2.0,
                       float(pixel[0]) - (shape[0] - 1) / 2.0])
    stage = np.asarray(centre[:2], dtype=float) + matrix @ offset
    return float(stage[0]), float(stage[1])


def _microscope_positions(settings):
    """The stage position each field was acquired at.

    :param settings: the run settings; ``microscope_positions`` names a table
        with the columns ``field`` (the field key the watch uses), ``x`` and
        ``y`` in micrometres, and optionally ``z``.
    :returns: ``{field key: (x, y) or (x, y, z)}``.
    :raises ValueError: when the file is not set or lacks a column.
    """
    from .tabular import read_table

    path = str(settings.get('microscope_positions') or '').strip()
    if not path:
        raise ValueError('microscope_feedback needs microscope_positions: a '
                         'table of the stage x and y (and optionally z) of '
                         'every field.')
    frame = read_table(path, canonicalise=False)
    missing = [column for column in ('field', 'x', 'y')
               if column not in frame.columns]
    if missing:
        raise ValueError(f'microscope_positions {path} lacks the column(s) '
                         f'{", ".join(missing)}.')
    has_z = 'z' in frame.columns
    positions = {}
    for row in frame.itertuples(index=False):
        values = [float(row.x), float(row.y)]
        if has_z and pd.notna(row.z):
            values.append(float(row.z))
        positions[str(row.field)] = tuple(values)
    return positions


class _SimulatedMicroscope:
    """A stand-in microscope that acquires from a folder of field images.

    It answers the calls the feedback loop makes on a pycro-manager ``Core``
    (``get_xy_stage_device``, ``set_xy_position``, ``get_x_position``,
    ``get_y_position``, ``get_focus_device``, ``set_position``,
    ``get_position``, ``wait_for_device``, ``snap_image``, ``get_image``,
    ``get_image_width`` and ``get_image_height``), so the loop runs
    unchanged against it. Each field image is laid on the stage centred on
    its entry in the positions table and oriented by the stage transform. A
    snap returns an image of the field's size centred on the current stage
    position, cut from the nearest field and padded with zeros where it runs
    off that field. Every call is appended to ``log``.
    """

    def __init__(self, folder, positions, matrix, settings):
        """Lay the fields of ``folder`` out on the stage.

        :param folder: the folder of field images.
        :param positions: the table of :func:`_microscope_positions`.
        :param matrix: the matrix of :func:`_microscope_matrix`.
        :param settings: the run settings, for the filename pattern; the
            file with the lowest channel of each field is used. The folder
            is read at every snap, so fields may still be arriving.
        """
        self.log, self._matrix = [], matrix
        self._folder, self._positions = folder, positions
        self._settings, self._patterns = settings, {}
        self._x = self._y = self._z = 0.0
        self._image = self._width = self._height = None
        self._cache = {}

    def _layout(self):
        """Place every field image now in the folder at its stage position.

        :returns: ``(centre, path)`` pairs, one per field.
        :raises ValueError: when no image belongs to a listed field.
        """
        chosen = {}
        for name in _watch_images(self._folder):
            key, channel = _watch_field_of(name, self._settings,
                                           self._patterns)
            if key in self._positions:
                rank = (channel is not None, str(channel), name)
                if key not in chosen or rank < chosen[key][0]:
                    chosen[key] = (rank, os.path.join(self._folder, name))
        if not chosen:
            raise ValueError(f'The simulated microscope found no image in '
                             f'{self._folder} whose field is in '
                             f'microscope_positions.')
        return [(self._positions[key][:2], path)
                for key, (_rank, path) in sorted(chosen.items())]

    def _field(self, path):
        """Read one field image once, as a 2-D array."""
        if path not in self._cache:
            import tifffile
            from skimage.io import imread

            image = (tifffile.imread(path)
                     if path.lower().endswith(('.tif', '.tiff'))
                     else imread(path))
            image = np.asarray(image)
            while image.ndim > 2:
                image = image[0]
            self._cache[path] = image
        return self._cache[path]

    def get_xy_stage_device(self):
        """The stage device name."""
        return 'SimulatedXYStage'

    def get_focus_device(self):
        """The focus device name."""
        return 'SimulatedZStage'

    def set_xy_position(self, x, y):
        """Move the stage to ``(x, y)`` micrometres."""
        self._x, self._y = float(x), float(y)
        self.log.append(('set_xy_position', self._x, self._y))

    def get_x_position(self):
        """The stage x in micrometres."""
        return self._x

    def get_y_position(self):
        """The stage y in micrometres."""
        return self._y

    def set_position(self, z):
        """Move the focus to ``z`` micrometres."""
        self._z = float(z)
        self.log.append(('set_position', self._z))

    def get_position(self):
        """The focus position in micrometres."""
        return self._z

    def wait_for_device(self, name):
        """Return at once: the simulated stage arrives instantly."""
        self.log.append(('wait_for_device', name))

    def snap_image(self):
        """Acquire the view centred on the current stage position."""
        here = np.array([self._x, self._y])
        centre, path = min(self._layout(), key=lambda field: float(
            np.hypot(*(np.asarray(field[0]) - here))))
        field = self._field(path)
        height, width = field.shape
        column, row = np.linalg.solve(self._matrix,
                                      here - np.asarray(centre, float))
        row, column = int(round(row)), int(round(column))
        view = np.zeros_like(field)
        top, left = max(row, 0), max(column, 0)
        bottom, right = min(row + height, height), min(column + width, width)
        if top < bottom and left < right:
            view[top - row:bottom - row, left - column:right - column] = \
                field[top:bottom, left:right]
        self._image, self._height, self._width = view, height, width
        self.log.append(('snap_image', self._x, self._y))

    def get_image(self):
        """The last snapped image, flattened as Micro-Manager returns it."""
        if self._image is None:
            raise RuntimeError('snap_image was not called.')
        return self._image.ravel()

    def get_image_width(self):
        """The width of the last snapped image."""
        return self._width

    def get_image_height(self):
        """The height of the last snapped image."""
        return self._height


def _microscope_open(settings, src):
    """Connect to the microscope named by ``microscope_driver``.

    ``'simulated'`` builds a :class:`_SimulatedMicroscope` on
    ``microscope_simulated_folder`` (blank means ``src``); ``'pycromanager'``
    connects to a running Micro-Manager through pycro-manager's ``Core``.

    :param settings: the run settings.
    :param src: the watched folder.
    :returns: an object with the pycro-manager ``Core`` calls the feedback
        loop uses.
    :raises ValueError: for an unknown driver.
    :raises ImportError: when pycro-manager is not installed.
    """
    driver = str(settings.get('microscope_driver') or 'simulated')
    if driver not in _MICROSCOPE_DRIVERS:
        raise ValueError(f'microscope_driver must be one of '
                         f'{_MICROSCOPE_DRIVERS}, not {driver!r}.')
    if driver == 'pycromanager':
        try:
            from pycromanager import Core
        except ImportError as exc:
            raise ImportError(
                "microscope_driver 'pycromanager' needs the pycromanager "
                "package: pip install pycromanager. Start Micro-Manager and "
                "tick Tools > Options > Run server on port 4827 before the "
                "run.") from exc
        return Core()
    folder = str(settings.get('microscope_simulated_folder') or '').strip()
    return _SimulatedMicroscope(folder or src, _microscope_positions(settings),
                                _microscope_matrix(settings), settings)


def _microscope_events(field_dir, settings):
    """The objects of one analysed field that should be imaged again.

    The ``microscope_event_table`` table of the field's measurements is
    filtered by ``microscope_event_query`` (a pandas query; blank keeps
    every object), and at most ``microscope_max_events`` rows are kept in
    table order. Each event is placed at the object's intensity-weighted
    centroid in the lowest measured channel.

    :param field_dir: the analysed field folder.
    :param settings: the run settings.
    :returns: ``(events, shape)``: a list of dicts with ``object`` and
        ``pixel`` ``[row, column]``, and the field's ``(height, width)``.
    :raises ValueError: when the field has no measurements, the table lacks
        centroids or the query fails.
    """
    import re

    from .tabular import database_tables, read_table

    db = os.path.join(field_dir, 'measurements', 'measurements.db')
    if not os.path.exists(db):
        raise ValueError('microscope_feedback needs measurements: set '
                         "watch_pipeline to 'mask_measure'.")
    table = str(settings.get('microscope_event_table') or 'cell')
    merged = os.path.join(field_dir, 'merged')
    stacks = sorted(name for name in os.listdir(merged)
                    if name.endswith('.npy')) if os.path.isdir(merged) else []
    if not stacks:
        raise ValueError('The field has no merged stack to size events by.')
    shape = tuple(np.load(os.path.join(merged, stacks[0]),
                          mmap_mode='r').shape[:2])
    if table not in database_tables(db):
        return [], shape
    frame = read_table(db, table=table)
    query = str(settings.get('microscope_event_query') or '').strip()
    if query and not frame.empty:
        try:
            frame = frame.query(query)
        except Exception as exc:
            raise ValueError(f'microscope_event_query {query!r} failed on '
                             f'the {table} table: {exc}') from None
    limit = int(_watch_number(settings, 'microscope_max_events', 10.0))
    frame = frame.head(limit)
    found = sorted((int(match.group(1)), column) for column in frame.columns
                   for match in [re.search(r'channel_(\d+)_centroid_weighted-0$',
                                           column)] if match)
    if found:
        row_column = found[0][1]
    elif 'centroid-0' in frame.columns:
        row_column = 'centroid-0'
    else:
        raise ValueError(f'The {table} table has no centroid columns.')
    column_column = row_column[:-1] + '1'
    label = next((name for name in ('object_label', 'label', f'{table}_label')
                  if name in frame.columns), None)
    names = frame[label].tolist() if label else frame.index.tolist()
    events = []
    for name, row, column in zip(names, frame[row_column].tolist(),
                                 frame[column_column].tolist()):
        if pd.isna(row) or pd.isna(column):
            continue
        events.append({'object': str(name),
                       'pixel': [float(row), float(column)]})
    return events, shape


def _microscope_queue(key, field_dir, context):
    """Turn a field's events into stage positions and queue them.

    :param key: the field key.
    :param field_dir: the analysed field folder.
    :param context: the watch state; its ledger entry for ``key`` gains an
        ``events`` list whose items carry ``object``, ``pixel``, ``stage``
        and ``status`` (``'queued'``, or ``'no_position'`` when the field is
        not in ``microscope_positions``).
    :returns: the number of events queued.
    """
    settings = context['settings']
    events, shape = _microscope_events(field_dir, settings)
    centre = context['positions'].get(key)
    for number, event in enumerate(events):
        event['id'] = f'{key}_e{number:03d}'
        if centre is None:
            event.update(stage=None, status='no_position')
            continue
        x, y = _microscope_stage_position(event['pixel'], shape, centre,
                                          context['matrix'])
        event['stage'] = [x, y] + list(centre[2:])
        event['status'] = 'queued'
    context['ledger']['fields'][key]['events'] = events
    _watch_save_ledger(context['ledger_path'], context['ledger'])
    if events and centre is None:
        print(f'watch_folder: {key} has {len(events)} event(s) but no entry '
              f'in microscope_positions; they are not imaged.')
    return sum(event['status'] == 'queued' for event in events)


def _microscope_acquire(event, context):
    """Move to one queued event and image it.

    ``microscope_timepoints`` images are taken ``microscope_interval_seconds``
    apart and written as TIFFs to ``src/spacr_watch/reimaged``.

    :param event: a queued event of :func:`_microscope_queue`.
    :param context: the watch state, with the open ``microscope``.
    :returns: the written file names.
    """
    from .tiff_io import write_tiff
    from .cancellation import checkpoint

    core, settings = context['microscope'], context['settings']
    frames = max(1, int(_watch_number(settings, 'microscope_timepoints', 1.0)))
    interval = _watch_number(settings, 'microscope_interval_seconds', 0.0)
    out = os.path.join(context['work'], _MICROSCOPE_REIMAGED)
    os.makedirs(out, exist_ok=True)
    x, y = event['stage'][:2]
    core.set_xy_position(x, y)
    core.wait_for_device(core.get_xy_stage_device())
    if len(event['stage']) > 2:
        core.set_position(event['stage'][2])
        core.wait_for_device(core.get_focus_device())
    names = []
    for frame in range(frames):
        if frame:
            deadline = time.time() + interval
            while time.time() < deadline:
                checkpoint()
                time.sleep(min(0.25, max(0.0, deadline - time.time())))
        core.snap_image()
        pixels = np.asarray(core.get_image()).reshape(
            int(core.get_image_height()), int(core.get_image_width()))
        name = f"{event['id']}_t{frame:03d}.tif"
        write_tiff(os.path.join(out, name), pixels,
                   metadata={'stage_x_um': x, 'stage_y_um': y})
        names.append(name)
    return names


def _microscope_drain(context):
    """Image every queued event in the watch record, oldest field first.

    Events whose acquisition fails are marked ``'failed'`` with the error and
    are not retried within the session.

    :param context: the watch state, with the open ``microscope``.
    :returns: how many events were imaged.
    """
    from .cancellation import PipelineCancelled

    ledger, imaged = context['ledger'], 0
    order = sorted(ledger['fields'].items(),
                   key=lambda item: item[1].get('finished') or 0)
    for key, entry in order:
        for event in entry.get('events') or ():
            if event.get('status') != 'queued':
                continue
            try:
                event['files'] = _microscope_acquire(event, context)
            except PipelineCancelled:
                raise
            except Exception as exc:
                event.update(status='failed',
                             error=f'{type(exc).__name__}: {exc}')
                print(f"watch_folder: ERROR imaging {event['id']} failed: "
                      f"{event['error']}")
            else:
                event['status'] = 'acquired'
                event['acquired'] = time.time()
                imaged += 1
                x, y = event['stage'][:2]
                print(f"watch_folder: imaged {event['id']} at stage "
                      f"({x:.1f}, {y:.1f}) um.")
            _watch_save_ledger(context['ledger_path'], ledger)
    return imaged


def _microscope_feedback(key, field_dir, context):
    """Send one analysed field's events to the microscope.

    :param key: the field key.
    :param field_dir: the analysed field folder.
    :param context: the watch state, with the open ``microscope``.
    """
    from .cancellation import PipelineCancelled

    try:
        queued = _microscope_queue(key, field_dir, context)
        if queued:
            print(f'watch_folder: {queued} event(s) of {key} queued for '
                  f're-imaging.')
        _microscope_drain(context)
    except PipelineCancelled:
        raise
    except Exception as exc:
        context['ledger']['fields'][key]['feedback_error'] = (
            f'{type(exc).__name__}: {exc}')
        _watch_save_ledger(context['ledger_path'], context['ledger'])
        print(f'watch_folder: ERROR microscope feedback for {key} failed: '
              f'{type(exc).__name__}: {exc}')


def _watch_folder_and_analyse(settings, analyse=None):
    """Watch an acquisition folder and analyse each field as it arrives.

    The folder ``src`` is scanned every ``watch_poll_seconds``. A file is
    ready once its size and modification time have not changed for
    ``watch_settle_seconds`` and it reads whole. With a fixed Convert map,
    every mapped channel, Z plane and timepoint must be present. Without a map, a known
    numeric convention must supply the documented channel origin through the
    highest selected position; an unknown origin is refused before output.
    Fields are told apart by the ``metadata_type`` or ``custom_regex`` filename
    pattern, and a file whose name carries no channel is a field by itself.
    Images already in the folder are analysed first.

    Each ready field is copied into ``src/spacr_watch/fields/<field>``, so
    the pipeline never writes to the acquired files, and analysed there on
    its own by ``watch_pipeline``: ``'mask'`` runs Make
    Masks, ``'mask_measure'`` then runs Measure with the settings file named
    by ``watch_measure_settings``. Its merged stacks are linked into
    ``src/spacr_watch/merged`` and its measurements appended to
    ``src/spacr_watch/measurements/measurements.db``. The
    ``'mask_measure_classify'`` pipeline also applies a saved CV model and
    collects its per-object predictions in that database. A mapped timelapse
    also collects its flat track CSVs under ``src/spacr_watch/tracks``. Every field is
    preprocessed alone, so the result equals a batch run of the same plate
    with ``batch_size=1``.

    ``src/spacr_watch/watch_ledger.json`` records every field with its files,
    when it arrived, started and finished, and whether it succeeded. It is
    written after every change, so a restarted watch skips the fields already
    analysed; a field that failed is tried again on the next start, and a
    field interrupted by Stop is analysed again from its images.

    The watch runs until Stop is pressed, or until nothing has changed in the
    folder for ``watch_idle_minutes`` when that is above 0. Stop takes effect
    between fields and while waiting.

    With ``microscope_feedback`` on, the objects of each measured field that
    match ``microscope_event_query`` in ``microscope_event_table`` become
    events. Each event's centroid is turned into a stage position from the
    field's entry in ``microscope_positions`` and the
    ``microscope_stage_transform`` matrix, queued in the field's record and
    imaged by the ``microscope_driver`` microscope, ``microscope_timepoints``
    times ``microscope_interval_seconds`` apart, into
    ``src/spacr_watch/reimaged``. Events still queued when a watch ends are
    imaged when the next one starts.

    :param settings: Make Masks settings with ``src`` naming one folder, plus
        the ``watch_*`` keys.
    :param analyse: the callable run on each field folder as
        ``analyse(field_dir, settings)``; None runs the chosen pipeline.
    :returns: a dict with ``done``, ``failed`` and ``incomplete`` lists of
        field keys and the ``ledger`` path.
    :raises ValueError: see :func:`_watch_check_settings`.
    :raises spacr.cancellation.PipelineCancelled: when Stop was pressed; the
        record is saved first.

    The settings are deep-copied at the start, so a callback or UI edit cannot
    change the recipe between live fields.
    """
    from .cancellation import PipelineCancelled, checkpoint

    src, pipeline, settle, poll, idle = _watch_check_settings(settings)
    from copy import deepcopy

    settings = deepcopy(dict(settings))
    _, mask_sha256 = _watch_mask_recipe(settings)
    manifest, map_sha256 = _watch_map_manifest(src, settings)
    series = _watch_truthy(settings.get('timelapse', False))
    if series:
        if manifest is None:
            raise ValueError('watch_folder: timelapse requires a fixed Convert '
                             'conversion_map.csv declaring the complete field series.')
        if (str(settings.get('metadata_type', 'cellvoyager')).lower() != 'cellvoyager'
                or settings.get('custom_regex') not in (None, '', 'None')):
            raise ValueError('watch_folder: mapped timelapse requires the '
                             'CellVoyager filename convention used by Convert.')
        if pipeline == 'mask_measure_classify' or _watch_truthy(
                settings.get('microscope_feedback', False)):
            raise ValueError('watch_folder: mapped timelapse supports mask or '
                             'mask_measure without microscope feedback.')
    source_channels = (_watch_source_channels(settings)
                       if manifest is None else None)
    measure_sha256 = None
    if pipeline in ('mask_measure', 'mask_measure_classify'):
        measure_recipe, measure_sha256 = _watch_measure_recipe(settings)
        settings = {**settings, 'watch_measure_snapshot': measure_recipe}
    classify_sha256 = None
    classify_model_sha256 = None
    if pipeline == 'mask_measure_classify':
        classify_recipe, classify_sha256, classify_model_sha256 = (
            _watch_classify_recipe(settings))
    work = os.path.join(src, _WATCH_DIR)
    os.makedirs(work, exist_ok=True)
    ledger_path = os.path.join(work, _WATCH_LEDGER)
    ledger = _watch_load_ledger(ledger_path, src)
    if ledger['fields'] and ledger.get('pipeline') != pipeline:
        raise ValueError(
            'watch_folder: the saved pipeline differs or is unknown; use a '
            'separate watch workspace for a different pipeline. Existing '
            'results and the saved record are preserved.')
    if (ledger.get('conversion_map_sha256') != map_sha256 and
            (ledger['fields'] or 'conversion_map_sha256' in ledger)):
        raise ValueError('watch_folder: conversion_map.csv differs from the saved watch '
                         'record; use a separate watch workspace. Existing results are preserved.')
    if pipeline in ('mask_measure', 'mask_measure_classify'):
        if ledger['fields'] and ledger.get('measure_settings_sha256') != measure_sha256:
            raise ValueError(
                'watch_folder: Measure settings differ from the saved recipe or '
                'its provenance is unknown; use a separate watch workspace. '
                'Existing results and the saved record are preserved.')
        ledger['measure_settings_sha256'] = measure_sha256
    if pipeline == 'mask_measure_classify':
        if ledger['fields'] and ledger.get('classify_settings_sha256') != classify_sha256:
            raise ValueError(
                'watch_folder: Classify settings or model differ from the saved '
                'recipe or their provenance is unknown; use a separate watch '
                'workspace. Existing results and the saved record are preserved.')
        classify_recipe['model_path'] = _watch_classify_model_snapshot(
            work, classify_recipe['model_path'], classify_model_sha256)
        settings = {**settings, 'watch_classify_snapshot': classify_recipe}
        ledger['classify_settings_sha256'] = classify_sha256
        ledger['classify_model_sha256'] = classify_model_sha256
    if ledger['fields'] and ledger.get('mask_settings_sha256') != mask_sha256:
        raise ValueError(
            'watch_folder: Mask settings differ from the saved recipe or '
            'its provenance is unknown; use a separate watch workspace. '
            'Existing results and the saved record are preserved.')
    ledger['mask_settings_sha256'] = mask_sha256
    ledger['conversion_map_sha256'] = map_sha256
    ledger['pipeline'] = pipeline
    for entry in ledger['fields'].values():
        if entry.get('status') == 'running':
            entry['status'] = 'interrupted'
    _watch_save_ledger(ledger_path, ledger)
    context = {'src': src, 'work': work, 'ledger': ledger,
               'ledger_path': ledger_path, 'settings': settings,
               'analyse': analyse or _watch_analyse_field, 'settle': settle,
               'source_channels': source_channels,
               'source_origin': min(map(int, source_channels)) if source_channels else None,
               'seen': {},
               'patterns': {}, 'tried': set(), 'warned': set(),
               'manifest': manifest, 'map_sha256': map_sha256,
               'series': series,
               'microscope': None}
    if _watch_truthy(settings.get('microscope_feedback', False)):
        context.update(positions=_microscope_positions(settings),
                       matrix=_microscope_matrix(settings))
        context['microscope'] = _microscope_open(settings, src)
        print(f"watch_folder: events are sent to the "
              f"{settings.get('microscope_driver') or 'simulated'} "
              f"microscope for re-imaging.")
        _microscope_drain(context)
    before = sum(1 for entry in ledger['fields'].values()
                 if entry.get('status') == 'done')
    print(f'watch_folder: watching {src} with pipeline {pipeline}; a file is '
          f'taken {settle:g} s after it stops changing. {before} field(s) '
          f'were analysed before.')
    last_change, last_line = time.time(), None
    try:
        while True:
            checkpoint()
            now = time.time()
            _watch_check_map(context)
            if _watch_observe(context, now):
                last_change = now
            ready, waiting = _watch_ready_fields(context, now)
            line = _watch_status_line(ledger, waiting)
            if line != last_line:
                print(line)
                last_line = line
            for _first, key, members, signature in ready:
                checkpoint()
                _watch_run_field(key, members, signature, context)
                last_change = time.time()
            if ready:
                continue
            if idle > 0 and time.time() - last_change >= idle:
                print(f'watch_folder: nothing changed for {idle / 60.0:g} '
                      f'min; stopping.')
                break
            deadline = time.time() + poll
            while time.time() < deadline:
                checkpoint()
                time.sleep(min(0.25, max(0.0, deadline - time.time())))
    except PipelineCancelled:
        print('watch_folder: stopped. The record is saved; the next start '
              'continues where this one ended.')
        raise
    finally:
        _watch_save_ledger(ledger_path, ledger)

    fields = ledger['fields']
    done = sorted(key for key, entry in fields.items()
                  if entry.get('status') == 'done')
    failed = sorted(key for key, entry in fields.items()
                    if entry.get('status') == 'failed')
    observed_keys = {
        _watch_observed_field_of(name, settings, context['patterns'], series)[0]
        for name in context['seen']}
    incomplete = sorted(
        observed_keys | set(manifest or {}) |
        {key for key, entry in fields.items() if entry.get('status') == 'waiting'})
    incomplete = sorted(set(incomplete) - set(done) - set(failed))
    print(_watch_status_line(ledger, len(incomplete)))
    if incomplete:
        print(f'watch_folder: {len(incomplete)} field(s) never became '
              f'complete: {", ".join(incomplete[:10])}.')
    return {'done': done, 'failed': failed, 'incomplete': incomplete,
            'ledger': ledger_path}


#: The column a multi-plate UMAP carries so a user can colour by source.
#:
#: Deliberately the same name :mod:`spacr.multi_database` uses, so a frame
#: coming out of the UMAP and a frame coming out of the Gate Editor's merge
#: answer "where did this row come from" under one column name. Two names for
#: one idea is how a user ends up unable to compare the two.
UMAP_SOURCE_COLUMN = "source_database"


def _umap_source_label(src):
    """A short, readable name for one source root.

    The plate folder's own name, because that is what the user called it and
    what they will look for in a legend -- ``get_db_paths`` appends
    ``measurements/measurements.db`` to it, so the file name is the same for
    every plate and useless as a label.
    """
    text = str(src).rstrip(os.sep)
    return os.path.basename(text) or text


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

    from .database_concurrency import connect as _connect_database

    conn = _connect_database(db_path)
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
    """Reduce per-object features and plot the resulting 2-D embedding.

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
        ``reduction_method`` (UMAP, t-SNE, PCA, Isomap or Spectral),
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
    from .batch_correction import correction_kwargs
    settings = set_default_umap_image_settings(settings)

    reduction_method = str(settings.get('reduction_method', 'umap')).lower()
    reducer_options = {
        'tsne': {
            'perplexity': settings['tsne_perplexity'],
            'learning_rate': settings['tsne_learning_rate'],
            'early_exaggeration': settings['tsne_early_exaggeration'],
            'max_iter': settings['tsne_max_iter'],
        },
        'pca': {
            'whiten': settings['pca_whiten'],
            'svd_solver': settings['pca_svd_solver'],
        },
        'isomap': {
            'n_neighbors': settings['isomap_n_neighbors'],
            'path_method': settings['isomap_path_method'],
        },
        'spectral': {
            'affinity': settings['spectral_affinity'],
            'n_neighbors': settings['spectral_n_neighbors'],
        },
    }.get(reduction_method, {})
    reducer_runtime = {
        'reducer_options': reducer_options,
        'prefer_gpu': bool(settings.get('gpu', False)),
        'random_seed': int(settings.get('random_seed', 42)),
    }

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

    if len(db_paths) > 1:
        try:
            from .multi_database import describe_merge
            _plan = describe_merge(db_paths, 'cell')
            if _plan.colliding_plates:
                _detail = '; '.join(
                    f"{plate!r} in {', '.join(labels)}"
                    for plate, labels in sorted(_plan.colliding_plates.items()))
                print(f"WARNING: the same plate id appears in more than one "
                      f"source database, so objects from different runs share "
                      f"one key and every per-well number below is computed "
                      f"over both at once: {_detail}. Rename the plates, or "
                      f"colour by '{UMAP_SOURCE_COLUMN}' to see which is "
                      f"which.")
        except Exception:
            pass
    from .io import open_crop_source, crop_refs_for_rows, CROP_REF_COLUMN
    crop_object = 'cell'

    for i,db_path in enumerate(db_paths):
        source = open_crop_source(settings, settings['src'][i],
                                  object_type=crop_object,
                                  verbose=bool(settings.get('verbose', True)))
        on_demand = source is not None and getattr(source, 'kind', 'png') == 'merged'
        _validate_umap_source_db(db_path, tables,
                                 require_png_list=not on_demand)
        df = _read_and_join_tables(db_path, table_names=tables,
                                   require_crops=False)
        df['_spacr_umap_db_path'] = db_path
        df[UMAP_SOURCE_COLUMN] = _umap_source_label(settings['src'][i])
        df['_spacr_umap_db_png_path'] = (
            df['png_path'] if 'png_path' in df.columns else None)
        df, image_paths_tmp = correct_paths(df, settings['src'][i])
        if source is not None and settings['plot_images']:
            df[CROP_REF_COLUMN] = crop_refs_for_rows(source, df,
                                                     object_type=crop_object)
        all_df = pd.concat([all_df, df], axis=0)

    from .row_exclusions import exclude_matching_rows
    all_df, exclusion_notes = exclude_matching_rows(
        all_df, settings.get("exclude_rows"))
    if settings.get("verbose"):
        for note in exclusion_notes:
            print(note)

    all_df['cond'] = all_df['columnID'].apply(map_condition, neg=settings['neg'], pos=settings['pos'], mix=settings['mix'])

    if settings['exclude_conditions']:
        if isinstance(settings['exclude_conditions'], str):
            settings['exclude_conditions'] = [settings['exclude_conditions']]
        row_count_before = len(all_df)
        all_df = all_df[~all_df['cond'].isin(settings['exclude_conditions'])]
        if settings['verbose']:
            print(f'Excluded {row_count_before - len(all_df)} rows after excluding: {settings["exclude_conditions"]}, rows left: {len(all_df)}')

    if settings['row_limit'] is not None:
        n_rows = min(int(settings['row_limit']), len(all_df))
        if n_rows < settings['row_limit']:
            print(f"row_limit={settings['row_limit']} exceeds the {len(all_df)} "
                  f"rows available; using all {len(all_df)}.")
        all_df = all_df.sample(n=n_rows, random_state=42)

    if CROP_REF_COLUMN in all_df.columns:
        image_paths = all_df[CROP_REF_COLUMN].to_list()
        all_df = all_df.drop(columns=[CROP_REF_COLUMN])
    elif 'png_path' in all_df.columns:
        image_paths = all_df['png_path'].to_list()
    else:
        print("No crop source and no 'png_path' column; plotting points only.")
        image_paths = None
        settings['plot_images'] = False

    if settings['embedding_by_controls']:
        
        col_to_compare = all_df[settings['col_to_compare']].reset_index(drop=True)
        print(col_to_compare)
            
        numeric_data = preprocess_data(
            all_df, settings['filter_by'],
            settings['remove_highly_correlated'], settings['log_data'],
            settings['exclude'],
            batch_covariate_column=settings.get('batch_covariate_column'),
            batch_combat_mean_only=bool(
                settings.get('batch_combat_mean_only', False)),
            **correction_kwargs(
                settings,
                default_control_column=settings.get('col_to_compare'),
                default_control_values=settings.get('neg'),
            ),
        )

        numeric_data_df = pd.DataFrame(numeric_data)

        numeric_data_df = numeric_data_df.reset_index(drop=True)

        numeric_data_df[settings['col_to_compare']] = col_to_compare

        positive_control_df = numeric_data_df[numeric_data_df[settings['col_to_compare']] == settings['pos']].copy()
        negative_control_df = numeric_data_df[numeric_data_df[settings['col_to_compare']] == settings['neg']].copy()
        control_numeric_data_df = pd.concat([positive_control_df, negative_control_df])

        numeric_data_df = numeric_data_df.drop(columns=[settings['col_to_compare']])
        control_numeric_data_df = control_numeric_data_df.drop(columns=[settings['col_to_compare']])

        numeric_data = numeric_data_df.values
        control_numeric_data = control_numeric_data_df.values

        _, _, reducer = reduction_and_clustering(control_numeric_data, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['eps'], settings['min_samples'], settings['clustering'], settings['reduction_method'], settings['verbose'], n_jobs=settings['n_jobs'], mode='fit', model=False, **reducer_runtime)
        
        numeric_data = preprocess_data(
            all_df, settings['filter_by'],
            settings['remove_highly_correlated'], settings['log_data'],
            settings['exclude'],
            batch_covariate_column=settings.get('batch_covariate_column'),
            batch_combat_mean_only=bool(
                settings.get('batch_combat_mean_only', False)),
            **correction_kwargs(
                settings,
                default_control_column=settings.get('col_to_compare'),
                default_control_values=settings.get('neg'),
            ),
        )
        embedding, labels, reducer = reduction_and_clustering(numeric_data, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['eps'], settings['min_samples'], settings['clustering'], settings['reduction_method'], settings['verbose'], n_jobs=settings['n_jobs'], mode=None, model=reducer, **reducer_runtime)

    else:
        if settings['resnet_features']:
            raise NotImplementedError(
                "resnet_features is not implemented: spaCR cannot embed raw "
                "PNG crops with ResNet features yet. Set resnet_features=False "
                "to embed the measured feature table instead.")
        else:
            numeric_data = preprocess_data(
                all_df, settings['filter_by'],
                settings['remove_highly_correlated'], settings['log_data'],
                settings['exclude'],
                batch_covariate_column=settings.get(
                    'batch_covariate_column'),
                batch_combat_mean_only=bool(
                    settings.get('batch_combat_mean_only', False)),
                **correction_kwargs(
                    settings,
                    default_control_column=settings.get('col_to_compare'),
                    default_control_values=settings.get('neg'),
                ),
            )
            embedding, labels, reducer = reduction_and_clustering(numeric_data, settings['n_neighbors'], settings['min_dist'], settings['metric'], settings['eps'], settings['min_samples'], settings['clustering'], settings['reduction_method'], settings['verbose'], n_jobs=settings['n_jobs'], **reducer_runtime)
    
    clusters_found = (
        len(labels) > 0 and np.any(np.asarray(labels) != -1)
    )
    if settings['remove_cluster_noise']:
        keep = np.asarray(labels) != -1
        if keep.any():
            embedding, labels = remove_noise(embedding, labels)
            if image_paths is not None:
                image_paths = [p for p, k in zip(image_paths, keep) if k]
            all_df = all_df[keep].reset_index(drop=True)
        else:
            labels = np.ones(len(all_df), dtype=int)

    cluster_labels = np.asarray(labels).copy()

    records = []
    point_count = len(embedding)
    rows_for_payload = all_df.reset_index(drop=True).iloc[:point_count]
    if len(cluster_labels) == 0 and point_count:
        cluster_labels = np.ones(point_count, dtype=int)
    elif len(cluster_labels) != point_count:
        raise ValueError(
            "Embedding, cluster labels, and source rows lost alignment: "
            f"{point_count} points but {len(cluster_labels)} labels.")

    plot_labels = (
        all_df[settings['color_by']].reset_index(drop=True)
        if settings['color_by'] else cluster_labels
    )
    colors = generate_colors(
        len(np.unique(plot_labels)), settings['black_background'])

    for point_index, row in rows_for_payload.iterrows():
        image = (image_paths[point_index]
                 if image_paths is not None and point_index < len(image_paths)
                 else None)
        db_path = row.get('_spacr_umap_db_path')
        db_png_path = row.get('_spacr_umap_db_png_path')
        if pd.isna(db_path):
            db_path = None
        if pd.isna(db_png_path):
            db_png_path = None
        records.append({
            'image': image,
            'display_name': str(image) if image is not None else '',
            'db_path': db_path,
            'db_png_path': db_png_path,
            'prcfo': row.get('prcfo'),
        })
    interactive_payload = {
        'embedding': np.asarray(embedding),
        'labels': cluster_labels,
        'reduction_method': reduction_method,
        'backend': str(getattr(reducer, '_spacr_backend', 'cpu')),
        'records': records,
        'plot_labels': np.asarray(plot_labels),
        'display': {
            'point_size': settings['dot_size'],
            'point_color': settings['point_color'],
            'point_alpha': settings['point_alpha'],
            'outline_width': settings['outline_width'],
            'canvas_width': settings['umap_canvas_width'],
            'sidebar_width': settings['umap_sidebar_width'],
        },
        'settings': {key: value for key, value in settings.items()
                     if not str(key).startswith('_')},
        'theme_colors': settings.get('_plot_theme'),
    }

    theme_colors = settings.get('_plot_theme')
    umap_plt = plot_embedding(
        embedding, image_paths, plot_labels, settings['image_nr'],
        settings['img_zoom'], colors, settings['plot_by_cluster'],
        settings['plot_outlines'], settings['plot_points'],
        settings['plot_images'], settings['smooth_lines'],
        settings['black_background'], settings['figuresize'],
        settings['dot_size'], settings['remove_image_canvas'],
        settings['verbose'], interactive_payload=interactive_payload,
        theme_colors=theme_colors, point_color=settings['point_color'],
        point_alpha=settings['point_alpha'],
        outline_width=settings['outline_width'],
    )
    if settings['plot_cluster_grids'] and settings['plot_images']:
        grid_plt = plot_clusters_grid(embedding, plot_labels, settings['image_nr'], image_paths, colors, settings['figuresize'], settings['black_background'], settings['verbose'], theme_colors=theme_colors)
    
    if settings['save_figure']:
        results_dir = os.path.join(settings['src'][0], 'results')
        os.makedirs(results_dir, exist_ok=True)
        reduction_method = settings['reduction_method'].upper()
        embedding_path = os.path.join(results_dir, f'{reduction_method}_embedding.pdf')
        embedding_path = save_figure(umap_plt, embedding_path)
        print(f'Saved {reduction_method} embedding to {embedding_path} and grid to {embedding_path}')
        if settings['plot_cluster_grids'] and settings['plot_images']:
            grid_path = os.path.join(results_dir, f'{reduction_method}_grid.pdf')
            grid_path = save_figure(grid_plt, grid_path)
            print(f'Saved {reduction_method} embedding to {embedding_path} and grid to {grid_path}')

    all_df['cluster'] = cluster_labels
    if not clusters_found:
        print("No clusters found. Consider reducing 'min_samples' or increasing 'eps' for DBSCAN.")

    all_df = all_df.drop(
        columns=['_spacr_umap_db_path', '_spacr_umap_db_png_path'],
        errors='ignore')

    results_dir = os.path.join(settings['src'][0], 'results')
    results_csv = os.path.join(results_dir,'embedding_results.csv')
    os.makedirs(results_dir, exist_ok=True)
    all_df.to_csv(results_csv, index=False)
    print(f'Results saved to {results_csv}')

    if settings['analyze_clusters'] and all_df['cluster'].nunique(dropna=True) >= 2:
        combined_results = cluster_feature_analysis(all_df)
        results_dir = os.path.join(settings['src'][0], 'results')
        cluster_results_csv = os.path.join(results_dir,'cluster_results.csv')
        os.makedirs(results_dir, exist_ok=True)
        combined_results.to_csv(cluster_results_csv, index=False)
        print(f'Cluster results saved to {cluster_results_csv}')
    elif settings['analyze_clusters']:
        print(
            "Cluster analysis skipped: at least two clusters are required "
            "for between-cluster statistical tests.")

    fig = umap_plt.gcf() if hasattr(umap_plt, "gcf") else plt.gcf()


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

    if not reduction_params:
        raise ValueError(
            "reducer_hyperparameter_search: reduction_params is required. "
            "Pass a dict (or list of dicts) containing 'n_neighbors' to sweep "
            "UMAP or 'perplexity' to sweep tSNE.")

    wants_umap = any('n_neighbors' in param for param in reduction_params)
    wants_tsne = any('perplexity' in param for param in reduction_params)

    if wants_umap and wants_tsne:
        raise ValueError("Reduction parameters must include 'n_neighbors' for UMAP or 'perplexity' for tSNE, not both.")
    elif wants_umap:
        reduction_method = 'umap'
    elif wants_tsne:
        reduction_method = 'tsne'
    else:
        raise ValueError(
            "Reduction parameters must include 'n_neighbors' for UMAP or "
            f"'perplexity' for tSNE; got {reduction_params}.")


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

    from .row_exclusions import exclude_matching_rows
    all_df, exclusion_notes = exclude_matching_rows(
        all_df, settings.get("exclude_rows"))
    if settings.get("verbose"):
        for note in exclusion_notes:
            print(note)

    all_df['cond'] = all_df['columnID'].apply(map_condition, neg=settings['neg'], pos=settings['pos'], mix=settings['mix'])

    if settings['exclude_conditions']:
        if isinstance(settings['exclude_conditions'], str):
            settings['exclude_conditions'] = [settings['exclude_conditions']]
        row_count_before = len(all_df)
        all_df = all_df[~all_df['cond'].isin(settings['exclude_conditions'])]
        if settings['verbose']:
            print(f'Excluded {row_count_before - len(all_df)} rows after excluding: {settings["exclude_conditions"]}, rows left: {len(all_df)}')

    if settings['row_limit'] is not None:
        n_rows = min(int(settings['row_limit']), len(all_df))
        if n_rows < settings['row_limit']:
            print(f"row_limit={settings['row_limit']} exceeds the {len(all_df)} "
                  f"rows available; using all {len(all_df)}.")
        all_df = all_df.sample(n=n_rows, random_state=42)

    from .batch_correction import correction_kwargs
    numeric_data = preprocess_data(
        all_df, settings['filter_by'], settings['remove_highly_correlated'],
        settings['log_data'], settings['exclude'],
        batch_covariate_column=settings.get('batch_covariate_column'),
        batch_combat_mean_only=bool(
            settings.get('batch_combat_mean_only', False)),
        **correction_kwargs(
            settings,
            default_control_column=settings.get('col_to_compare'),
            default_control_values=settings.get('neg'),
        ),
    )

    clustering_params = []
    if dbscan_params:
        for param in dbscan_params:
            param['method'] = 'dbscan'
            clustering_params.append(param)
    if kmeans_params:
        for param in kmeans_params:
            param['method'] = 'kmeans'
            clustering_params.append(param)

    print('Testing parameters:', reduction_params)
    print('Testing clustering parameters:', clustering_params)

    grid_rows = len(reduction_params)
    grid_cols = len(clustering_params)

    fig_width = grid_cols*10
    fig_height = grid_rows*10

    with figure_style(theme_target()):
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))
        from .figures.bundle import _register_figure_data
        _register_figure_data(fig, None, kind="image", title="Reducer hyperparameter search")

        axs = np.atleast_1d(axs)
    
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

                if settings['reduction_method'].lower() == 'umap':
                    n_neighbors = reduction_param.get('n_neighbors', 15)

                    if isinstance(n_neighbors, float):
                        n_neighbors = int(n_neighbors * len(numeric_data))

                    min_dist = reduction_param.get('min_dist', 0.1)
                    embedding, labels = search_reduction_and_clustering(numeric_data, n_neighbors, min_dist, settings['metric'], 
                                                                        clustering_param.get('eps', 0.5), clustering_param.get('min_samples', 5), 
                                                                        clustering_param['method'], settings['reduction_method'], settings['verbose'], reduction_param, n_jobs=settings['n_jobs'])
                
                else:
                    perplexity = reduction_param.get('perplexity', 30)

                    if isinstance(perplexity, float):
                        perplexity = int(perplexity * len(numeric_data))

                    embedding, labels = search_reduction_and_clustering(numeric_data, perplexity, 0.1, settings['metric'],
                                                                        clustering_param.get('eps', 0.5), clustering_param.get('min_samples', 5),
                                                                        clustering_param['method'], settings['reduction_method'], settings['verbose'], reduction_param, n_jobs=settings['n_jobs'])

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
            save_figure(plt.gcf(),
                        os.path.join(results_dir, 'hyperparameter_search'))
        if return_fig:
            return fig
        if show and not save:
            plt.show()
        return

def _finite_ratio(numerator, denominator):
    """Divide aligned Series while representing invalid observations as NaN.

    Statistical and plotting code treats infinity as a number, so unchecked
    intensity ratios can turn a zero denominator into an apparently extreme
    biological effect.  Missing/non-numeric/non-finite inputs and zero
    denominators are undefined observations, not zeroes and not infinities.
    """
    numerator = pd.to_numeric(numerator, errors='coerce')
    denominator = pd.to_numeric(denominator, errors='coerce')
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & denominator.ne(0)
    )
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


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
        
        df, _ = _read_and_merge_data(db_loc, settings['tables'], verbose=True, nuclei_limit=settings['nuclei_limit'], pathogen_limit=settings['pathogen_limit'])
        
        df = annotate_conditions(df, cells=settings['cells'], cell_loc=None, pathogens=settings['controls'], pathogen_loc=settings['controls_loc'], treatments=None, treatment_loc=None)
        
        df['recruitment'] = _finite_ratio(
            df['pathogen_channel_1_mean_intensity'],
            df['cytoplasm_channel_1_mean_intensity'])
                
        all_df = pd.concat([all_df, df], ignore_index=True)
    
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
        
        figs.append(fig)
        results.append(results_df)
    
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
    
    for i, fig in enumerate(figs):
        res = results[i]
        
        if i < len(srcs):
            source = srcs[i]
        else:
            source = srcs[0]

        dst = os.path.join(source, 'results')
        print(f"Savings results to {dst}")
        os.makedirs(dst, exist_ok=True)
        
        save_figure(fig, os.path.join(
            dst,
            f"figure_controls_{i}_{settings['representation']}"
            f"_{settings['summary_func']}_{settings['graph_type']}"))
        res.to_csv(os.path.join(dst, f"results_controls_{i}_{settings['representation']}_{settings['summary_func']}_{settings['graph_type']}.csv"), index=False)

    return
