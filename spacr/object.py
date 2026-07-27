
import os, gc, torch, time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
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
from cellpose import models as cp_models

from functools import partial
from skimage.segmentation import watershed
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.filters import (threshold_otsu,threshold_local,frangi,sato,meijering,gaussian,difference_of_gaussians,apply_hysteresis_threshold)
from skimage.feature import blob_log, blob_dog, peak_local_max
from skimage.morphology import (remove_small_objects,remove_small_holes,binary_opening,binary_closing,binary_dilation,binary_erosion,disk,skeletonize,white_tophat)
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.restoration import rolling_ball

warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")

def merge_split_filter_masks(masks, intensity_images, settings, object_type, batch_filenames=None):
    """Apply merge/split/filter operations directly to in-memory masks.

    Skips work when no operation is enabled for ``object_type``; otherwise
    processes each FOV serially so progress reporting stays in order.

    :param masks: 2D/3D ndarray or iterable of 2D masks (one per FOV).
    :param intensity_images: Matching intensity arrays for scoring merges/splits.
    :param settings: Dict of pipeline settings; per-object-type suffixes control
        which operations run (e.g. ``<type>_perimeter_fraction``,
        ``<type>_intensity_merge``, ``<type>_min_area``).
    :param object_type: Label used to look up per-object settings (``'cell'``,
        ``'nucleus'``, ``'pathogen'``, ``'organelle'``).
    :param batch_filenames: Optional per-FOV filenames used only for logging.
    :returns: Original ``masks`` unchanged when no operation is enabled, else a
        list of filtered mask arrays (one per FOV).
    """
    import numpy as np
    from joblib import Parallel, delayed
    from .utils import print_progress, _process_single_fov_in_memory

    pf = settings.get(f'{object_type}_perimeter_fraction', settings.get(f'{object_type}_perimiter_fraction', 0))
    im = settings.get(f'{object_type}_intensity_merge', False)
    isp = settings.get(f'{object_type}_intensity_split', False)
    moa = settings.get(f'{object_type}_min_object_area', 0)
    mna = settings.get(f'{object_type}_min_area', 0)
    mxa = settings.get(f'{object_type}_max_area', 0)
    rb = settings.get(f'{object_type}_remove_border_objects', False)
    mni = settings.get(f'{object_type}_min_intensity_percentile', 0)
    mxi = settings.get(f'{object_type}_max_intensity_percentile', 100)

    needs_work = (
        pf > 0 or im or isp or moa > 0 or mna > 0 or
        (mxa and mxa > 0) or rb or mni > 0 or mxi < 100
    )

    if not needs_work:
        print(f"merge_split_filter_masks({object_type}): no operations needed, skipping")
        return masks

    if masks is None:
        return None

    print(f"merge_split_filter_masks({object_type}): "
          f"perimeter_merge={pf > 0}(frac={pf}), intensity_merge={im}, "
          f"split={isp}, min_area={mna}, max_area={mxa}, "
          f"remove_border={rb}, intensity_pct=[{mni}, {mxi}]")

    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            raise ValueError(f"Unsupported masks ndim: {masks.ndim}")
    else:
        mask_list = list(masks)

    if isinstance(intensity_images, np.ndarray):
        if intensity_images.ndim == 2:
            intensity_list = [intensity_images]
        elif intensity_images.ndim == 3:
            intensity_list = [intensity_images[i] for i in range(intensity_images.shape[0])]
        elif intensity_images.ndim == 4:
            intensity_list = [intensity_images[i] for i in range(intensity_images.shape[0])]
        else:
            raise ValueError(f"Unsupported intensity_images ndim: {intensity_images.ndim}")
    else:
        intensity_list = list(intensity_images)

    if len(mask_list) != len(intensity_list):
        raise ValueError(
            f"Number of masks ({len(mask_list)}) does not match number of intensity images ({len(intensity_list)})."
        )

    if batch_filenames is None:
        batch_filenames = [f'image_{i:06d}' for i in range(len(mask_list))]

    total = len(mask_list)
    time_ls = []

    def _progress(fov_idx, total_fovs, duration, op):
        """Record a per-FOV duration and emit the shared progress line."""
        time_ls.append(duration)
        print_progress(
            fov_idx + 1,
            total_fovs,
            n_jobs=1,
            time_ls=time_ls,
            batch_size=None,
            operation_type=op
        )

    def _run_one(idx, mask, intensity_img):
        """Run the configured filter pipeline against a single FOV mask."""
        out_mask = _process_single_fov_in_memory(
            mask=mask,
            intensity_img=intensity_img,
            intensity_channel=0,
            do_split=isp,
            do_perimeter_merge=(pf > 0),
            do_intensity_merge=(im and intensity_images is not None),
            perimeter_fraction=pf,
            area_multiplier=settings.get(f'{object_type}_area_multiplier', 2.0),
            min_distance=settings.get(f'{object_type}_min_distance', 10),
            min_object_area=moa,
            intensity_threshold_method=settings.get(f'{object_type}_intensity_threshold_method', 'mean'),
            intensity_percentile=settings.get(f'{object_type}_intensity_percentile', 75),
            min_area=mna,
            max_area=mxa if mxa else 0,
            remove_border_objects=rb,
            min_intensity_percentile=mni,
            max_intensity_percentile=mxi,
            progress_callback=_progress,
            fov_index=idx,
            total_fovs=total,
            op_name=f'merge_{object_type}',
        )
        return out_mask

    n_jobs = settings.get('n_jobs', 1)
    
    # Always run serial so progress prints work
    filtered_masks = [
        _run_one(idx, mask, img)
        for idx, (mask, img) in enumerate(zip(mask_list, intensity_list))
    ]

    return filtered_masks

def _run_seg_qc(src, settings, object_type):
    """Score the masks just written and surface the segmentation scorecard.

    Called at the end of every mask generator, once per object type, while the
    masks are the newest thing on disk and before anyone spends hours in
    ``measure_crop`` on them. Controlled by the ``seg_qc`` setting:

    * ``'off'`` — return immediately, touch nothing.
    * ``'report'`` (default) — score every field, write
      ``<plate>/qc/segmentation_qc_<object_type>.csv`` and print the card.
      Nothing is filtered, skipped or deleted; the point is that the user sees
      a bad plate now instead of discovering it in the measurements.
    * ``'flag'`` — as ``'report'``, plus a ``..._flags.json`` sidecar and the
      per-field flags recorded in ``settings['seg_qc_flags'][object_type]`` for
      a downstream step to act on.

    :param src: the mask source folder the generator was given (the one holding
        the ``.npz`` batches and the ``<object_type>_mask_stack`` output).
    :param settings: pipeline settings; read for ``seg_qc``, the ``seg_qc_*``
        thresholds and ``verbose``. Mutated only in ``'flag'`` mode.
    :param object_type: which masks to score.
    :returns: the dict :func:`spacr.seg_qc.run_segmentation_qc` returns, or
        None when QC is off, unavailable or it failed.
    """
    try:
        from .seg_qc import qc_mode, run_segmentation_qc, thresholds_from_settings

        mode = qc_mode(settings)
        if mode == 'off':
            return None

        mask_folder = os.path.join(src, f'{object_type}_mask_stack')
        # Same idiom as `count_loc` above: plate-level output lives one level up
        # from the mask source, next to measurements/.
        dst = os.path.dirname(src) or src
        result = run_segmentation_qc(
            mask_folder,
            object_type=object_type,
            dst=dst,
            mode=mode,
            thresholds=thresholds_from_settings(settings),
            verbose=bool(settings.get('verbose', True)),
        )
    except Exception as exc:
        # QC is a report, never a gate. A run that has just spent hours
        # segmenting must not lose its masks to a scorecard bug.
        print(f"Segmentation QC skipped for {object_type}: {type(exc).__name__}: {exc}")
        return None

    if result is not None and result.get('mode') == 'flag':
        settings.setdefault('seg_qc_flags', {})[object_type] = result['flags']
    return result


# ====================================================================== #
#  3D (Beta): z-stack plumbing
# ====================================================================== #
#
# Everything below is inert unless the `z_stack` setting is on.
# `_z_stack_plan` returns None in that case and every call site branches on
# it, so a run that has not opted in executes not one line of z code and
# produces byte-identical masks to a run from before these settings existed.
# That property is the acceptance criterion and is asserted in
# tests/test_zstack.py.

def _z_stack_plan(settings):
    """Return the :class:`spacr.zstack.ZStackSpec` for this run, or None.

    :param settings: pipeline settings dict.
    :returns: a spec when ``z_stack`` is on, else ``None``.
    """
    from .zstack import plan_from_settings

    return plan_from_settings(settings)


def _require_z_axis(stack, z_plan, path):
    """Stop the run when 3D is on but the array that arrived is flat.

    The alternative -- quietly segmenting the projection and calling the
    result 3-D -- is the failure mode this whole feature exists to avoid, so
    it is a hard error naming both the cause and the way out.

    :param stack: the ``(N, ...)`` array loaded from one ``.npz`` batch.
    :param z_plan: the active spec.
    :param path: the ``.npz`` path, for the message.
    :raises spacr.zstack.ZAxisNotPresentError: when there is no z axis.
    """
    from .zstack import ZAxisNotPresentError

    if stack.ndim >= 5:
        return

    raise ZAxisNotPresentError(
        f"z_stack is on but {os.path.basename(path)} holds an array of shape "
        f"{stack.shape}, which is (fields, Y, X, channels) -- there is no z "
        f"axis left to segment. spaCR's image ingest "
        f"(io._rename_and_organize_image_files) collapses every z plane of a "
        f"field into one plane while organising the raw files, so by the time "
        f"a batch reaches segmentation the z axis is already gone. Either turn "
        f"z_stack off and accept the projection spaCR has always made, or hand "
        f"spacr.zstack.segment_3d your (Z, Y, X, C) volumes directly through "
        f"the Python API. spaCR will not segment the projection and report it "
        f"as a 3-D result."
    )


def _cellpose_z_segment_fn(model, eval_kwargs, stitch_threshold):
    """Adapt ``CellposeModel.eval`` to the ``segment_fn`` contract of zstack.

    ``spacr.zstack`` knows nothing about Cellpose; it calls
    ``segment_fn(array, **kwargs)`` and this closure maps those kwargs onto
    ``eval``. Two of them are worth stating because Cellpose 4 is quiet about
    them:

    * ``do_3D=True`` is the only setting under which Cellpose honours
      ``anisotropy`` at all. With ``do_3D=False`` it accepts the argument and
      ignores it silently, which is why the stitch branch here never passes
      it.
    * Rather than let Cellpose stitch (``stitch_threshold`` in ``eval``), the
      stitch branch asks it for plain per-plane 2-D masks and links them with
      :func:`spacr.zstack.stitch_planes`. ``cellpose.utils.stitch3D`` resets
      its label counter after an empty plane, so an ``[objects][empty]
      [objects]`` stack there reuses ids and silently fuses unrelated objects;
      ours draws every new label from one monotonic counter.

    :param model: a loaded ``CellposeModel``.
    :param eval_kwargs: kwargs shared with the 2-D path.
    :param stitch_threshold: kept for the caller's records; the linking itself
        happens in :func:`spacr.zstack.stitch_planes`.
    :returns: a callable matching the ``segment_fn`` contract.
    """
    def _segment(array, do_3D=False, anisotropy=None, z_axis=None, stitch=False):
        kwargs = dict(eval_kwargs)

        if do_3D:
            kwargs.update(
                do_3D=True,
                anisotropy=anisotropy,
                z_axis=0 if z_axis is None else int(z_axis),
                channel_axis=-1,
            )
            output = model.eval(x=array, **kwargs)
            return np.asarray(output[0])

        if stitch:
            # One 2-D call per plane. Labels come back plane-local; zstack
            # links them.
            planes = [array[z] for z in range(array.shape[0])]
            kwargs['batch_size'] = len(planes)
            output = model.eval(x=planes, **kwargs)
            return np.asarray(output[0])

        # Projected 2-D plane: exactly the ordinary single-image call.
        output = model.eval(x=[array], **kwargs)
        return np.asarray(output[0][0])

    return _segment


def _segment_volumes_with_z(volumes, model, z_plan, eval_kwargs):
    """Segment one field at a time under the active z plan.

    Deliberately a plain loop rather than a batched call: a z-stack is ``n_z``
    times a field, so a batch of them is ``batch_size * n_z`` fields in memory
    at once. See :func:`spacr.zstack.estimate_peak_bytes` for the per-field
    footprint.

    :param volumes: sequence of ``(Z, Y, X, C)`` arrays, one per field.
    :param model: a loaded ``CellposeModel``.
    :param z_plan: the active :class:`spacr.zstack.ZStackSpec`.
    :param eval_kwargs: kwargs shared with the 2-D path.
    :returns: ``(masks, results, intensity)`` — a list of label arrays, 2-D
        under ``'project'`` and 3-D otherwise; the matching
        :class:`spacr.zstack.ZStackResult` records; and, under ``'project'``
        only, the projected ``(N, Y, X, C)`` intensity array that was actually
        segmented, which is what the 2-D merge/split/filter step must score
        against rather than the original volume.
    """
    from .zstack import project, segment_3d

    z_axis = 0 if z_plan.z_axis is None else z_plan.z_axis
    segment_fn = _cellpose_z_segment_fn(
        model, eval_kwargs, z_plan.stitch_threshold
    )

    masks, results = [], []
    for volume in volumes:
        result = segment_3d(
            volume,
            segment_fn=segment_fn,
            mode=z_plan.mode,
            stitch_threshold=z_plan.stitch_threshold,
            anisotropy=z_plan.anisotropy,
            voxel_size_um=z_plan.voxel_size_um,
            projection=z_plan.projection,
            z_axis=z_axis,
            resample_to_isotropic=z_plan.resample_to_isotropic,
        )
        masks.append(result.labels)
        results.append(result)

    intensity = None
    if z_plan.mode == 'project' and volumes:
        # The same projection segment_3d just made. The merge/split/filter
        # step scores masks against intensities, so it must see the plane the
        # masks were drawn on, not the volume it came from.
        intensity = np.stack([
            project(volume, mode=z_plan.projection, z_axis=z_axis)
            for volume in volumes
        ])

    return masks, results, intensity


# ====================================================================== #
#  4D (Beta): the time axis on top of the z axis
# ====================================================================== #
#
# The same contract as the 3D block above, one axis further out. Everything
# here is inert unless the `t_stack` setting is on: `_t_stack_plan` returns
# None then and every call site branches on it, so a run that has not opted in
# executes not one line of 4-D code and produces byte-identical masks to a run
# from before these settings existed. That property is the acceptance criterion
# and is asserted in tests/test_object_tstack_wiring.py.
#
# What `t_stack` declares, exactly
# --------------------------------
# It reinterprets the **leading axis of the .npz batch as time** rather than as
# a list of independent fields, and requires a z axis behind it -- a batch is
# then one `(T, Z, Y, X, C)` acquisition instead of `N` separate
# `(Z, Y, X, C)` fields. Which of the two leading axes is t and which is z is
# never guessed; `zstack.plan_4d_from_settings` refuses to build a plan at all
# until `t_axis_order` (or `t_axis`/`z_axis`) says, because reading one as the
# other links objects down a z stack and reports them as motion.
#
# How far it gets today, stated plainly
# -------------------------------------
# `spacr.io._rename_and_organize_image_files` collapses z into one plane per
# field while organising the raw files, so an ordinary run's batches are
# `(N, Y, X, C)` and there is no z axis left by the time segmentation sees
# them. `_require_t_axis` therefore stops such a run with
# `TAxisNotPresentError` naming that as the cause, rather than segmenting the
# projection frame by frame and reporting a 4-D result. Handed a genuine
# `(T, Z, Y, X, C)` array through the Python API -- write the .npz yourself --
# the path runs end to end into `zstack.segment_4d`.
#
# Linking across t (`zstack.track_4d`) is deliberately *not* wired here: its
# call site is `spacr.timelapse`, not this module. `t_stack` drives
# segmentation only, and says so.


def _t_stack_plan(settings):
    """Return the :class:`spacr.zstack.TStackSpec` for this run, or None.

    The t counterpart of :func:`_z_stack_plan`, and deliberately the same
    shape: one delegation to the settings bridge in :mod:`spacr.zstack`, which
    returns ``None`` whenever ``t_stack`` is off so that every caller can
    branch on a single value.

    :param settings: pipeline settings dict.
    :returns: a spec when ``t_stack`` is on, else ``None``.
    :raises spacr.zstack.AmbiguousAxisOrderError: when ``t_stack`` is on but
        neither ``t_axis_order`` nor ``t_axis``/``z_axis`` says which leading
        axis is time.
    :raises spacr.zstack.TStackError: when the 4D settings are otherwise
        self-inconsistent.
    """
    from .zstack import plan_4d_from_settings

    return plan_4d_from_settings(settings)


def _reconcile_z_and_t_plans(z_plan, t_plan, timelapse=False):
    """Decide which of the two Beta plans actually drives this run.

    ``t_stack`` and ``z_stack`` are not independent: ``zstack.segment_4d``
    calls ``zstack.segment_3d`` once per timepoint, with the z settings read
    from the very same keys ``zstack.plan_from_settings`` reads. So when both
    are on the 4-D plan already *is* the 3-D plan, applied per timepoint, and
    leaving the 3-D plan live as well would segment every field twice and keep
    only the second answer. The 4-D plan therefore supersedes it, out loud.

    :param z_plan: the :class:`spacr.zstack.ZStackSpec`, or ``None``.
    :param t_plan: the :class:`spacr.zstack.TStackSpec`, or ``None``.
    :param timelapse: whether the legacy 2-D ``timelapse`` tracking is also on.
    :returns: the z plan to keep -- ``z_plan`` unchanged when ``t_plan`` is
        ``None``, and ``None`` once the 4-D plan has taken over.
    :raises spacr.zstack.TrackerIsTwoDError: when ``timelapse`` tracking is on
        and the 4-D plan produces volumes its adapters cannot link.
    """
    from .zstack import TrackerIsTwoDError

    if t_plan is None:
        return z_plan

    if z_plan is not None:
        print(
            f"z_stack and t_stack are both on: the 4-D plan supersedes the "
            f"3-D one. zstack.segment_4d runs zstack.segment_3d once per "
            f"timepoint with these very same z settings "
            f"(z_segmentation_mode='{t_plan.z_mode}', "
            f"z_projection='{t_plan.projection}'), so keeping both live would "
            f"segment every field twice and discard the first answer."
        )

    # Only a plan that actually produces (Z, Y, X) volumes is a problem for
    # them; a flat time series, or 'project', leaves the masks 2-D.
    if timelapse and t_plan.z_axis is not None and t_plan.z_mode != 'project':
        raise TrackerIsTwoDError(
            f"t_stack is on with z_segmentation_mode='{t_plan.z_mode}', which "
            f"produces (Z, Y, X) label volumes, but the `timelapse` setting is "
            f"on too and every one of spaCR's timelapse tracking adapters "
            f"(spacr.timelapse._btrack_track_cells, _trackpy_track_cells, "
            f"_trackastra_track_cells, _ultrack_track_cells) requires a flat "
            f"(T, Y, X) stack and raises on anything else. Either set "
            f"z_segmentation_mode='project' so the masks stay 2-D, or turn "
            f"`timelapse` off and link the volumes yourself with "
            f"zstack.track_4d, which does track in 3-D. spaCR will not "
            f"project the volumes away to make the 2-D tracker accept them."
        )

    return None


def _require_t_axis(stack, t_plan, path):
    """Stop the run when 4D is on but the array that arrived is not 4-D.

    The exact counterpart of :func:`_require_z_axis`, and it exists for the
    same reason: quietly segmenting one projected plane per timepoint and
    calling the result 4-D is indistinguishable, after the fact, from a real
    4-D run. So it is a hard error naming both the cause and the way out.

    ``t_stack`` reads the batch's leading axis as time, so what is missing from
    an ordinary batch is the **z** axis: ``(N, Y, X, C)`` has four axes where a
    4-D acquisition needs five.

    A spec with ``z_axis=None`` describes a flat ``(T, Y, X, C)`` time series
    and needs only four, which is what an ordinary batch already is -- see
    :func:`spacr.zstack.segment_4d`, which makes one plain 2-D call per frame
    for it. ``spacr.zstack.plan_4d_from_settings`` cannot build such a spec
    from settings today, so this branch is reachable only through the Python
    API; the settings-level path for a flat time series is the ``timelapse``
    setting.

    :param stack: the ``(N, ...)`` array loaded from one ``.npz`` batch.
    :param t_plan: the active :class:`spacr.zstack.TStackSpec`.
    :param path: the ``.npz`` path, for the message.
    :raises spacr.zstack.TAxisNotPresentError: when there is no 4-D array here.
    """
    from .zstack import TAxisNotPresentError

    if stack.ndim >= (4 if t_plan.z_axis is None else 5):
        return

    raise TAxisNotPresentError(
        f"t_stack is on but {os.path.basename(path)} holds an array of shape "
        f"{stack.shape}, which is (timepoints, Y, X, channels) -- there is a "
        f"time axis but no z axis, so this is a flat 2-D time series and not "
        f"the (T, Z, Y, X, C) acquisition t_stack describes "
        f"(t_axis={t_plan.t_axis}, z_axis={t_plan.z_axis}). spaCR's image "
        f"ingest (io._rename_and_organize_image_files) collapses every z plane "
        f"of a field into one plane while organising the raw files, so by the "
        f"time a batch reaches segmentation the z axis is already gone. Turn "
        f"t_stack off: for a flat 2-D time series the `timelapse` setting is "
        f"the path that works today and it is untouched by any of this. To "
        f"segment real volumes over time, hand spacr.zstack.segment_4d your "
        f"(T, Z, Y, X, C) arrays directly through the Python API. spaCR will "
        f"not segment the projection and report it as a 4-D result."
    )


def _segment_timepoints_with_t(acquisition, model, t_plan, eval_kwargs):
    """Segment one ``(T, Z, Y, X, C)`` acquisition under the active t plan.

    The adapter is :func:`_cellpose_z_segment_fn`, unchanged: ``segment_4d``
    hands each timepoint to ``segment_3d``, which calls ``segment_fn`` with
    exactly the kwargs the 3-D path already documents. There is deliberately
    no second Cellpose adapter -- a 4-D run and a 3-D run must not be able to
    drift apart in how they drive the model.

    :param acquisition: a ``(T, Z, Y, X, C)`` array, axes as ``t_plan`` names
        them.
    :param model: a loaded ``CellposeModel``.
    :param t_plan: the active :class:`spacr.zstack.TStackSpec`.
    :param eval_kwargs: kwargs shared with the 2-D path.
    :returns: ``(masks, result, intensity)`` — one label array per timepoint,
        2-D under ``'project'`` and 3-D otherwise; the
        :class:`spacr.zstack.TStackResult`; and, under ``'project'`` only, the
        projected ``(T, Y, X, C)`` intensity array that was actually
        segmented, which is what the 2-D merge/split/filter step must score
        against rather than the original volumes.
    """
    from .zstack import iter_volumes, project, segment_4d

    segment_fn = _cellpose_z_segment_fn(
        model, eval_kwargs, t_plan.stitch_threshold
    )

    result = segment_4d(acquisition, t_plan, segment_fn=segment_fn)
    masks = [np.asarray(frame) for frame in np.asarray(result.labels)]

    intensity = None
    if t_plan.z_axis is None:
        # A flat time series: there is no z to collapse, so there is no
        # projected copy either and the caller scores against the batch it
        # already has, exactly as the ordinary 2-D path does.
        pass
    elif t_plan.z_mode == 'project':
        # The same projection segment_3d just made, one per timepoint. The
        # merge/split/filter step scores masks against intensities, so it must
        # see the plane the masks were drawn on, not the volume it came from.
        intensity = np.stack([
            project(volume, mode=t_plan.projection, z_axis=0)
            for volume in iter_volumes(acquisition, t_plan)
        ])

    return masks, result, intensity


def _refuse_t_stack(settings, where):
    """Stop a generator that cannot honour ``t_stack`` from silently ignoring it.

    Only :func:`generate_cellpose_masks_sam` implements the 4-D path. The other
    generators would segment each field independently in 2-D and return exactly
    what a run with ``t_stack`` off returns, while the settings panel said 4-D
    — which is the failure this whole feature exists to prevent, so they say so
    instead.

    :param settings: pipeline settings dict.
    :param where: the generator's name, for the message.
    :raises spacr.zstack.TStackError: when ``t_stack`` is on.
    """
    if not settings.get('t_stack', False):
        return

    from .zstack import TStackError

    raise TStackError(
        f"t_stack is on but {where} does not implement the 4-D path: it "
        f"segments every field independently in 2-D and would hand back "
        f"exactly the masks a run with t_stack off produces, while the "
        f"settings said 4-D. Only object.generate_cellpose_masks_sam reads "
        f"t_stack today. Either run that generator, or turn t_stack off. "
        f"spaCR will not accept a 4-D setting and quietly return a 2-D result."
    )


def generate_cellpose_masks_sam(src, settings, object_type):
    """Segment one object channel across all ``.npz`` batches under ``src`` using Cellpose-SAM.

    Loads the ``cpsam`` pretrained model — or, when
    ``<object_type>_model_name`` (or ``pathogen_model``) names a checkpoint
    the user trained, that checkpoint — iterates over each pre-batched
    ``.npz`` file, runs merge/split/filter on the resulting masks, optionally
    tracks timelapse objects, saves per-image ``.npy`` masks, and records
    per-object counts to the run's SQLite database.

    :param src: Directory containing the pre-batched ``.npz`` image stacks.
    :param settings: Pipeline settings dict; canonicalized via
        :func:`spacr.settings.set_default_settings_preprocess_generate_masks`.
    :param object_type: ``'cell'``, ``'nucleus'``, ``'pathogen'`` or
        ``'organelle'``; drives channel/threshold lookups and output folder name.
    :returns: None.
    """
    from .utils import (_masks_to_masks_stack, all_elements_match,
                        prepare_batch_for_segmentation, _get_cellpose_channels,
                        _resolve_cellpose_pretrained)
    from .io import _create_database, _save_object_counts_to_database, _check_masks, _get_avg_object_size
    from .timelapse import (_npz_to_movie, _btrack_track_cells, _trackpy_track_cells,
                            _trackastra_track_cells, _ultrack_track_cells)
    from .plot import plot_cellpose4_output
    from .settings import set_default_settings_preprocess_generate_masks, _get_object_settings
    from .spacr_cellpose import parse_cellpose4_output
    
    gc.collect()
    if not torch.cuda.is_available():
        print(f'Torch CUDA is not available, using CPU')
        
    settings['src'] = src
    
    settings = set_default_settings_preprocess_generate_masks(settings)

    if settings['verbose']:
        settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
        settings_df['setting_value'] = settings_df['setting_value'].apply(str)
        display(settings_df)
        
    figuresize=10
    # `timelapse` is no longer offered by the Mask module's settings panel —
    # it belongs to the Timelapse module (spacr.core.preprocess_generate_masks_timelapse).
    # It is still defaulted by set_default_settings_preprocess_generate_masks and
    # still honoured here, so old settings CSVs and direct API calls keep working;
    # .get() keeps a hand-built dict from raising instead of segmenting.
    timelapse = settings.get('timelapse', False)

    if timelapse:
        timelapse_displacement = settings['timelapse_displacement']
        timelapse_frame_limits = settings['timelapse_frame_limits']
        timelapse_memory = settings['timelapse_memory']
        timelapse_remove_transient = settings['timelapse_remove_transient']
        timelapse_mode = settings['timelapse_mode']
        timelapse_objects = settings['timelapse_objects']
    
    batch_size = settings['batch_size']
    
    cellprob_threshold = settings[f'{object_type}_CP_prob']
    flow_threshold = settings[f'{object_type}_FT']
    object_settings = _get_object_settings(object_type, settings)

    # None unless the user opted into 3D (Beta). Every branch below is guarded
    # on it, so the 2-D path is untouched when it is None.
    z_plan = _z_stack_plan(settings)

    # None unless the user opted into 4D (Beta). Raises here, before the model
    # is loaded and the first field read, when the axis order is not settled --
    # that answer cannot change later in the run.
    t_plan = _t_stack_plan(settings)
    z_plan = _reconcile_z_and_t_plans(z_plan, t_plan, timelapse=timelapse)

    # The z mode that actually runs, whichever plan is driving. None means no
    # z code runs at all and the masks are ordinary 2-D ones: either neither
    # plan is set, or the 4-D plan describes a flat (T, Y, X) time series, for
    # which segment_4d makes one plain 2-D call per frame and no z mode enters.
    if t_plan is not None:
        beta_mode = None if t_plan.z_axis is None else t_plan.z_mode
    elif z_plan is not None:
        beta_mode = z_plan.mode
    else:
        beta_mode = None

    if settings.get('cellpose_nucleus_channel') is None and settings.get('nucleus_channel') is not None:
        settings['cellpose_nucleus_channel'] = settings['nucleus_channel']
    
    if settings.get('cellpose_cell_channel') is None and settings.get('cell_channel') is not None:
        settings['cellpose_cell_channel'] = settings['cell_channel']
    
    if settings.get('cellpose_pathogen_channel') is None and settings.get('pathogen_channel') is not None:
        settings['cellpose_pathogen_channel'] = settings['pathogen_channel']
        
    channels_to_extract, cellpose_channels = _get_cellpose_channels(settings)
    channels = cellpose_channels.get(object_type, [])
    
    if len(channels) == 0:
        raise ValueError(f"No valid channels defined for object_type '{object_type}'.")
        
    if settings['verbose']:
        print(channels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # pretrained_model used to be the literal 'cpsam' here, so a checkpoint
    # from spaCR's own Train Cellpose module was discarded and the stock
    # weights ran instead — silently, on the pipeline's DEFAULT path.
    # _resolve_cellpose_pretrained keeps 'cpsam' for the stock case and
    # returns the checkpoint path when the user named one.
    model_name = object_settings['model_name']
    if object_type == 'pathogen' and settings.get('pathogen_model') is not None:
        model_name = settings['pathogen_model']
    pretrained = _resolve_cellpose_pretrained(model_name, object_type=object_type)
    model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=pretrained, device=device)
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    
    count_loc = os.path.dirname(src)+'/measurements/measurements.db'
    os.makedirs(os.path.dirname(src)+'/measurements', exist_ok=True)
    _create_database(count_loc)
    
    average_sizes = []
    average_count = []
    time_ls = []
    
    for file_index, path in enumerate(paths):
        name = os.path.basename(path)
        name, ext = os.path.splitext(name)
        output_folder = os.path.join(os.path.dirname(path), object_type+'_mask_stack')
        os.makedirs(output_folder, exist_ok=True)
        overall_average_size = 0
        
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']
            
            for i, filename in enumerate(filenames):
                output_path = os.path.join(output_folder, filename)
                
                if os.path.exists(output_path):
                    print(f"File {filename} already exists in the output folder. Skipping...")
                    continue
                
        if timelapse:
            trackable_objects = ['cell','nucleus','pathogen']
            if not all_elements_match(settings['timelapse_objects'], trackable_objects):
                print(f'timelapse_objects {settings["timelapse_objects"]} must be a subset of {trackable_objects}')
                return

            if len(stack) != batch_size:
                print(f'Changed batch_size:{batch_size} to {len(stack)}, data length:{len(stack)}')
                settings['timelapse_batch_size'] = len(stack)
                batch_size = len(stack)
                if isinstance(timelapse_frame_limits, list):
                    if len(timelapse_frame_limits) >= 2:
                        stack = stack[timelapse_frame_limits[0]: timelapse_frame_limits[1], :, :, :].astype(stack.dtype)
                        filenames = filenames[timelapse_frame_limits[0]: timelapse_frame_limits[1]]
                        batch_size = len(stack)
                        print(f'Cut batch at indecies: {timelapse_frame_limits}, New batch_size: {batch_size} ')
        
        if t_plan is not None:
            # Fail before the first timepoint rather than after: whether this
            # array is 4-D cannot change later in the run.
            _require_t_axis(stack, t_plan, path)
        elif z_plan is not None:
            # Fail before the first field rather than after: whether this
            # array has a z axis cannot change later in the run.
            _require_z_axis(stack, z_plan, path)

        for i in range(0, stack.shape[0], batch_size):
            mask_stack = []
            if z_plan is not None or t_plan is not None:
                # (N, Z, Y, X, C) — or (T, Z, Y, X, C) under t_stack, where the
                # leading axis is time: select channels off the trailing axis
                # so the z axis is preserved.
                batch = stack[i: i+batch_size][..., channels].astype(stack.dtype)
            elif stack.shape[3] == 1:
                batch = stack[i: i+batch_size, :, :, [0]].astype(stack.dtype)
            else:
                batch = stack[i: i+batch_size, :, :, channels].astype(stack.dtype)

            # In the future drop the npz save file step, just keep it in memory and pass the batch directly to the model. This will save time and disk space. For now, keep it for backwards compatibility and to avoid issues with large batches that might not fit in memory.                
            #if stack.shape[3] == 1:
            #    batch = stack[i: i+batch_size, :, :, [0]].astype(stack.dtype)
            #else:
            #    subset = stack[i: i+batch_size, :, :, channels_to_extract].astype(stack.dtype)
            #    batch = subset[:, :, :, channels]

            batch_filenames = filenames[i: i+batch_size].tolist()

            if not settings['plot']:
                batch, batch_filenames = _check_masks(batch, batch_filenames, output_folder)
            if batch.size == 0:
                continue
            
            cp_batch = prepare_batch_for_segmentation(batch)
            batch_list = [cp_batch[i] for i in range(cp_batch.shape[0])]

            if timelapse:
                movie_path = os.path.join(os.path.dirname(src), 'movies')
                os.makedirs(movie_path, exist_ok=True)
                save_path = os.path.join(movie_path, f'timelapse_{object_type}_{name}.mp4')
                _npz_to_movie(cp_batch, batch_filenames, save_path, fps=2)
                
            
            if z_plan is None and t_plan is None:
                output = model.eval(
                    x=batch_list,
                    batch_size=len(batch_list),
                    normalize=False,
                    channel_axis=-1,
                    min_size=object_settings['min_size'],
                    progress=True,
                    # Cellpose 4 still honours `diameter` in eval() — it rescales
                    # the image by 30/diameter. Only diam_mean at construction is
                    # ignored. This was hard-coded to None, so an explicitly-set
                    # <obj>_diameter (and anything spacr.diameter proposes) never
                    # reached Cellpose. The setting defaults to None, so None here
                    # still means "let CPSAM work at native scale".
                    diameter=settings.get(f'{object_type}_diameter'),
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    resample=object_settings['resample']
                    )

                masks, flows, _, _, _ = parse_cellpose4_output(output)
            else:
                # Same eval kwargs as the 2-D call above, minus the ones zstack
                # sets per mode (x, batch_size, channel_axis, do_3D, anisotropy,
                # z_axis). Shared by the 3-D and 4-D paths so the two cannot
                # drive Cellpose differently.
                z_eval_kwargs = dict(
                    batch_size=1,
                    normalize=False,
                    channel_axis=-1,
                    min_size=object_settings['min_size'],
                    progress=True,
                    diameter=settings.get(f'{object_type}_diameter'),
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    resample=object_settings['resample'],
                )
                if t_plan is not None:
                    # The whole batch is one acquisition, not a list of
                    # independent fields: its leading axis is time.
                    masks, t_result, beta_intensity = _segment_timepoints_with_t(
                        cp_batch, model, t_plan, z_eval_kwargs
                    )
                    if settings['verbose']:
                        for note in t_result.notes:
                            print(f"[4D] {name}: {note}")
                        for filename, result in zip(batch_filenames,
                                                    t_result.z_results):
                            for note in result.notes:
                                print(f"[4D] {filename}: {note}")
                else:
                    masks, z_results, beta_intensity = _segment_volumes_with_z(
                        batch_list, model, z_plan, z_eval_kwargs
                    )
                    if settings['verbose']:
                        for filename, result in zip(batch_filenames, z_results):
                            for note in result.notes:
                                print(f"[3D] {filename}: {note}")
                flows = None

            if beta_mode is None or beta_mode == 'project':
                # merge/split/filter reason in 2-D: they measure areas in px²
                # and split objects with a 2-D watershed. Handing them a
                # (Z, Y, X) volume would silently apply all of that per plane
                # and tear the 3-D labels apart, so the 3-D modes skip them.
                masks = merge_split_filter_masks(
                    masks=masks,
                    intensity_images=batch if beta_mode is None else beta_intensity,
                    settings=settings,
                    object_type=object_type,
                    batch_filenames=batch_filenames,
                )
            else:
                print(
                    f"merge_split_filter_masks({object_type}): skipped — the "
                    f"merge/split/filter operations are 2-D only and would be "
                    f"applied per z plane, breaking the 3-D labels that "
                    f"z_segmentation_mode='{beta_mode}' just produced"
                )
            
            if timelapse:
                if settings['plot']:
                    plot_cellpose4_output(batch_list, masks, flows, cmap='inferno', figuresize=figuresize, nr=1, print_object_number=True)

                _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_timelapse')
                if object_type in timelapse_objects:
                    if timelapse_mode == 'btrack':
                        if not timelapse_displacement is None:
                            radius = timelapse_displacement
                        else:
                            radius = 100

                        n_jobs = os.cpu_count()-2
                        if n_jobs < 1:
                            n_jobs = 1
                            
                        mask_stack = _btrack_track_cells(src=src,
                                                         name=name,
                                                         batch_filenames=batch_filenames,
                                                         object_type=object_type,
                                                         plot=settings['plot'],
                                                         save=settings['save'],
                                                         masks_3D=masks,
                                                         mode=timelapse_mode,
                                                         timelapse_remove_transient=timelapse_remove_transient,
                                                         radius=radius,
                                                         n_jobs=n_jobs,
                                                         batch_list=None,
                                                         optimizer_time_limit_s=120,
                                                         optimizer_mip_gap=0.01,
                                                         run_optimization=True,
                                                         max_objects_for_optimization=20000)
                    
                    if timelapse_mode == 'trackastra':
                        # Trackastra takes the raw intensity stack as well as the
                        # masks — it uses appearance, not just geometry — so hand
                        # it the batch we already loaded rather than masks alone.
                        mask_stack = _trackastra_track_cells(
                            src=src,
                            name=name,
                            batch_filenames=batch_filenames,
                            object_type=object_type,
                            masks=masks,
                            images=batch,
                            timelapse_remove_transient=timelapse_remove_transient,
                            plot=settings['plot'],
                            save=settings['save'],
                            mode=timelapse_mode,
                            model_name=settings.get('trackastra_model', 'general_2d'),
                            linking_mode=settings.get('trackastra_linking', 'greedy'))

                    elif timelapse_mode == 'ultrack':
                        # Ultrack derives its own candidate objects from a
                        # contour map built off these labels, and uses the raw
                        # intensities for appearance features while linking, so
                        # it gets the same two arrays trackastra does.
                        mask_stack = _ultrack_track_cells(
                            src=src,
                            name=name,
                            batch_filenames=batch_filenames,
                            object_type=object_type,
                            masks=masks,
                            images=batch,
                            timelapse_remove_transient=timelapse_remove_transient,
                            plot=settings['plot'],
                            save=settings['save'],
                            mode=timelapse_mode,
                            max_distance=settings.get('ultrack_max_distance', 25.0),
                            division_weight=settings.get('ultrack_division_weight', -0.1),
                            contour_sigma=settings.get('ultrack_contour_sigma', 0.0),
                            n_workers=settings.get('ultrack_n_workers', 1))

                    if timelapse_mode == 'trackpy' or timelapse_mode == 'iou':
                        if timelapse_mode == 'iou':
                            track_by_iou = True
                        else:
                            track_by_iou = False
                        
                        mask_stack = _trackpy_track_cells(src=src,
                                                          name=name,
                                                          batch_filenames=batch_filenames,
                                                          object_type=object_type,
                                                          masks=masks,
                                                          timelapse_displacement=timelapse_displacement,
                                                          timelapse_memory=timelapse_memory,
                                                          timelapse_remove_transient=timelapse_remove_transient,
                                                          plot=settings['plot'],
                                                          save=settings['save'],
                                                          mode=timelapse_mode,
                                                          track_by_iou=track_by_iou)
                else:
                    mask_stack = _masks_to_masks_stack(masks)
            else:
                print("saving to DB")
                _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_before_filtration')
                mask_stack = _masks_to_masks_stack(masks)
        
            # Legacy inline hook: the automated motility assay is now the
            # standalone Motility Assay module (app key 'motility'), so the
            # Mask GUI no longer exposes `motility_analysis`. The gate stays
            # for settings CSVs and API callers that still set both flags.
            if timelapse and settings.get("motility_analysis", False):
                from .timelapse import automated_motility_assay
                _ = automated_motility_assay(settings)
            
            if not np.any(mask_stack):
                avg_num_objects_per_image, average_obj_size = 0, 0
            else:
                avg_num_objects_per_image, average_obj_size = _get_avg_object_size(mask_stack)
            
            average_count.append(avg_num_objects_per_image)
            average_sizes.append(average_obj_size) 
            overall_average_size = np.mean(average_sizes) if len(average_sizes) > 0 else 0
            overall_average_count = np.mean(average_count) if len(average_count) > 0 else 0
            print(f'Found {overall_average_count} {object_type}/FOV. average size: {overall_average_size:.3f} px2')

            # Plot and save inside the per-batch loop. Both blocks used to sit
            # one level out, at the .npz level: an .npz holding more batches
            # than `batch_size` therefore ran every batch but only ever wrote
            # the last one's masks to disk (the earlier mask_stacks were
            # rebound and lost), while an empty .npz never entered this loop at
            # all and hit the save block with mask_stack unbound -> NameError.
            if not timelapse:
                if settings['plot']:
                    if flows is None:
                        # plot_cellpose4_output draws the per-image flow field
                        # beside each mask; the z paths call eval once per volume
                        # and do not collect one, and in the stitch/volumetric
                        # modes the mask is a (Z, Y, X) volume it cannot render
                        # beside a 2-D field either.
                        reason = (f"z_segmentation_mode='{beta_mode}'"
                                  if beta_mode else "the 4D path")
                        print(
                            f"plot skipped: {reason} does not produce the "
                            f"per-image flow images this plot needs. Inspect the "
                            f"saved .npy masks instead."
                        )
                    else:
                        plot_cellpose4_output(batch_list, masks, flows, cmap='inferno', figuresize=figuresize, nr=len(batch_list))

            if settings['save']:
                for mask_index, mask in enumerate(mask_stack):
                    output_filename = os.path.join(output_folder, batch_filenames[mask_index])
                    mask = mask.astype(np.uint16)
                    np.save(output_filename, mask)
                mask_stack = []
                batch_filenames = []

        gc.collect()

    torch.cuda.empty_cache()
    _run_seg_qc(src, settings, object_type)
    return

def generate_cellpose_masks(src, settings, object_type):
    """Segment one object channel across all ``.npz`` batches under ``src`` using a chosen Cellpose model.

    Selects the model via :func:`spacr.utils._choose_model` (stock or custom),
    runs per-batch inference with the object-specific channel/threshold
    settings, applies :func:`spacr.utils._filter_cp_masks`, optionally tracks
    timelapse objects, and writes ``.npy`` masks plus per-object counts.

    :param src: Directory containing the pre-batched ``.npz`` image stacks.
    :param settings: Pipeline settings dict; canonicalized via
        :func:`spacr.settings.set_default_settings_preprocess_generate_masks`.
    :param object_type: ``'cell'``, ``'nucleus'``, or ``'pathogen'``; drives
        channel/threshold lookups and output folder name.
    :returns: None.
    """
    from .utils import _masks_to_masks_stack, _filter_cp_masks, _get_cellpose_channels, _choose_model, all_elements_match, prepare_batch_for_segmentation
    from .io import _create_database, _save_object_counts_to_database, _check_masks, _get_avg_object_size
    from .timelapse import (_npz_to_movie, _btrack_track_cells, _trackpy_track_cells,
                            _trackastra_track_cells)
    from .plot import plot_cellpose4_output
    from .settings import set_default_settings_preprocess_generate_masks, _get_object_settings
    from .spacr_cellpose import parse_cellpose4_output
    
    gc.collect()
    if not torch.cuda.is_available():
        print(f'Torch CUDA is not available, using CPU')
        
    settings['src'] = src
    
    settings = set_default_settings_preprocess_generate_masks(settings)

    # This generator has no 4-D path. Say so rather than returning 2-D masks
    # to a user whose settings said 4-D.
    _refuse_t_stack(settings, 'object.generate_cellpose_masks')

    if settings['verbose']:
        settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
        settings_df['setting_value'] = settings_df['setting_value'].apply(str)
        display(settings_df)
        
    figuresize=10
    # `timelapse` is no longer offered by the Mask module's settings panel —
    # it belongs to the Timelapse module (spacr.core.preprocess_generate_masks_timelapse).
    # It is still defaulted by set_default_settings_preprocess_generate_masks and
    # still honoured here, so old settings CSVs and direct API calls keep working;
    # .get() keeps a hand-built dict from raising instead of segmenting.
    timelapse = settings.get('timelapse', False)

    if timelapse:
        timelapse_displacement = settings['timelapse_displacement']
        timelapse_frame_limits = settings['timelapse_frame_limits']
        timelapse_memory = settings['timelapse_memory']
        timelapse_remove_transient = settings['timelapse_remove_transient']
        timelapse_mode = settings['timelapse_mode']
        timelapse_objects = settings['timelapse_objects']
    
    batch_size = settings['batch_size']
    
    cellprob_threshold = settings[f'{object_type}_CP_prob']

    flow_threshold = settings[f'{object_type}_FT']

    object_settings = _get_object_settings(object_type, settings)
    
    model_name = object_settings['model_name']
    
    if settings.get('cellpose_nucleus_channel') is None and settings.get('nucleus_channel') is not None:
        settings['cellpose_nucleus_channel'] = settings['nucleus_channel']

    if settings.get('cellpose_cell_channel') is None and settings.get('cell_channel') is not None:
        settings['cellpose_cell_channel'] = settings['cell_channel']

    if settings.get('cellpose_pathogen_channel') is None and settings.get('pathogen_channel') is not None:
        settings['cellpose_pathogen_channel'] = settings['pathogen_channel']

    # _get_cellpose_channels takes the settings dict and returns
    # (channels_to_extract, cellpose_channels). It used to be called here with
    # four positional arguments (src, nucleus, pathogen, cell) left over from an
    # older signature, which raised TypeError on every single call — this whole
    # generator was unreachable. Same call as generate_cellpose_masks_sam makes,
    # so the two cannot pick different channels for the same settings.
    channels_to_extract, cellpose_channels = _get_cellpose_channels(settings)

    if settings['verbose']:
        print(cellpose_channels)
        
    if object_type not in cellpose_channels:
        raise ValueError(f"Error: No channels were specified for object_type '{object_type}'. Check your settings.")
    
    channels = cellpose_channels[object_type]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if object_type == 'pathogen' and not settings['pathogen_model'] is None:
        model_name = settings['pathogen_model']
    
    model = _choose_model(model_name, device, object_type=object_type, restore_type=None, object_settings=object_settings)

    #chans = [2, 1] if model_name == 'cyto2' else [0,0] if model_name == 'nucleus' else [2,0] if model_name == 'cyto' else [2, 0] if model_name == 'cyto3' else [2, 0]
    
    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]    
    
    count_loc = os.path.dirname(src)+'/measurements/measurements.db'
    os.makedirs(os.path.dirname(src)+'/measurements', exist_ok=True)
    _create_database(count_loc)
    
    average_sizes = []
    average_count = []
    time_ls = []
    
    for file_index, path in enumerate(paths):
        name = os.path.basename(path)
        name, ext = os.path.splitext(name)
        output_folder = os.path.join(os.path.dirname(path), object_type+'_mask_stack')
        os.makedirs(output_folder, exist_ok=True)
        overall_average_size = 0
        
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']
            
            for i, filename in enumerate(filenames):
                output_path = os.path.join(output_folder, filename)
                
                if os.path.exists(output_path):
                    print(f"File {filename} already exists in the output folder. Skipping...")
                    continue
        
        if timelapse:

            trackable_objects = ['cell','nucleus','pathogen']
            if not all_elements_match(settings['timelapse_objects'], trackable_objects):
                print(f'timelapse_objects {settings["timelapse_objects"]} must be a subset of {trackable_objects}')
                return

            if len(stack) != batch_size:
                print(f'Changed batch_size:{batch_size} to {len(stack)}, data length:{len(stack)}')
                settings['timelapse_batch_size'] = len(stack)
                batch_size = len(stack)
                if isinstance(timelapse_frame_limits, list):
                    if len(timelapse_frame_limits) >= 2:
                        stack = stack[timelapse_frame_limits[0]: timelapse_frame_limits[1], :, :, :].astype(stack.dtype)
                        filenames = filenames[timelapse_frame_limits[0]: timelapse_frame_limits[1]]
                        batch_size = len(stack)
                        print(f'Cut batch at indecies: {timelapse_frame_limits}, New batch_size: {batch_size} ')
        
        for i in range(0, stack.shape[0], batch_size):
            mask_stack = []
            if stack.shape[3] == 1:
                batch = stack[i: i+batch_size, :, :, [0,0]].astype(stack.dtype)
            else:
                batch = stack[i: i+batch_size, :, :, channels].astype(stack.dtype)

            batch_filenames = filenames[i: i+batch_size].tolist()

            if not settings['plot']:
                batch, batch_filenames = _check_masks(batch, batch_filenames, output_folder)
            if batch.size == 0:
                continue
            
            batch = prepare_batch_for_segmentation(batch)
            batch_list = [batch[i] for i in range(batch.shape[0])]

            if timelapse:
                movie_path = os.path.join(os.path.dirname(src), 'movies')
                os.makedirs(movie_path, exist_ok=True)
                save_path = os.path.join(movie_path, f'timelapse_{object_type}_{name}.mp4')
                _npz_to_movie(batch, batch_filenames, save_path, fps=2)
                        
            output = model.eval(x=batch_list,
                                batch_size=batch_size,
                                normalize=False,
                                channel_axis=-1,
                                channels=channels,
                                # <obj>_min_area is documented as "passed to
                                # Cellpose as min_size"; this generator never
                                # passed it, so Cellpose used its own default of
                                # 15 px and the setting did nothing here. The
                                # SAM generator has always passed it.
                                min_size=object_settings['min_size'],
                                diameter=object_settings['diameter'],
                                flow_threshold=flow_threshold,
                                cellprob_threshold=cellprob_threshold,
                                rescale=None,
                                resample=object_settings['resample'])
            
                        
            masks, flows, _, _, _ = parse_cellpose4_output(output)

            if timelapse:
                if settings['plot']:
                    plot_cellpose4_output(batch_list, masks, flows, cmap='inferno', figuresize=figuresize, nr=1, print_object_number=True)

                _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_timelapse')
                if object_type in timelapse_objects:
                    if timelapse_mode == 'btrack':
                        if not timelapse_displacement is None:
                            radius = timelapse_displacement
                        else:
                            radius = 100

                        n_jobs = os.cpu_count()-2
                        if n_jobs < 1:
                            n_jobs = 1
                            
                        mask_stack = _btrack_track_cells(src=src,
                                                         name=name,
                                                         batch_filenames=batch_filenames,
                                                         object_type=object_type,
                                                         plot=settings['plot'],
                                                         save=settings['save'],
                                                         masks_3D=masks,
                                                         mode=timelapse_mode,
                                                         timelapse_remove_transient=timelapse_remove_transient,
                                                         radius=radius,
                                                         n_jobs=n_jobs,
                                                         batch_list=None,
                                                         optimizer_time_limit_s=120,
                                                         optimizer_mip_gap=0.01,
                                                         run_optimization=True,
                                                         max_objects_for_optimization=20000)
                    
                    if timelapse_mode == 'trackpy' or timelapse_mode == 'iou':
                        if timelapse_mode == 'iou':
                            track_by_iou = True
                        else:
                            track_by_iou = False
                        
                        mask_stack = _trackpy_track_cells(src=src,
                                                          name=name,
                                                          batch_filenames=batch_filenames,
                                                          object_type=object_type,
                                                          masks=masks,
                                                          timelapse_displacement=timelapse_displacement,
                                                          timelapse_memory=timelapse_memory,
                                                          timelapse_remove_transient=timelapse_remove_transient,
                                                          plot=settings['plot'],
                                                          save=settings['save'],
                                                          mode=timelapse_mode,
                                                          track_by_iou=track_by_iou)
                else:
                    mask_stack = _masks_to_masks_stack(masks)
            else:
                _save_object_counts_to_database(masks, object_type, batch_filenames, count_loc, added_string='_before_filtration')
                if object_settings['merge'] and not settings['filter']:
                    mask_stack = _filter_cp_masks(masks=masks,
                                                # _filter_cp_masks iterates
                                                # zip(masks, flows[0], batch),
                                                # i.e. it wants the per-image
                                                # flow list nested one deep.
                                                # `flows` here is already that
                                                # per-image list, so passing it
                                                # bare made flows[0] the FIRST
                                                # IMAGE's flow array and the zip
                                                # ran over its rows: any batch
                                                # with more fields than the
                                                # images are tall silently lost
                                                # the trailing masks, and every
                                                # plot got a single pixel row
                                                # where a flow image belonged.
                                                flows=[flows],
                                                filter_size=False,
                                                filter_intensity=False,
                                                minimum_size=object_settings['minimum_size'],
                                                maximum_size=object_settings['maximum_size'],
                                                remove_border_objects=False,
                                                merge=object_settings['merge'],
                                                batch=batch,
                                                plot=settings['plot'],
                                                figuresize=figuresize)

                if settings['filter']:
                    mask_stack = _filter_cp_masks(masks=masks,
                                                # Nested one deep — see the
                                                # merge branch above.
                                                flows=[flows],
                                                filter_size=object_settings['filter_size'],
                                                filter_intensity=object_settings['filter_intensity'],
                                                minimum_size=object_settings['minimum_size'],
                                                maximum_size=object_settings['maximum_size'],
                                                remove_border_objects=object_settings['remove_border_objects'],
                                                merge=object_settings['merge'],
                                                batch=batch,
                                                plot=settings['plot'],
                                                figuresize=figuresize)
                    
                    _save_object_counts_to_database(mask_stack, object_type, batch_filenames, count_loc, added_string='_after_filtration')
                elif not object_settings['merge']:
                    # `elif not ...merge`, not a bare `else`: with merge on and
                    # filter off the block above has already produced the
                    # merged stack, and an unconditional else rebound
                    # mask_stack to the raw Cellpose masks right after,
                    # throwing the merge away. `merge_pathogens` was therefore
                    # a no-op in this generator.
                    mask_stack = _masks_to_masks_stack(masks)
        
            # Legacy inline hook: the automated motility assay is now the
            # standalone Motility Assay module (app key 'motility'), so the
            # Mask GUI no longer exposes `motility_analysis`. The gate stays
            # for settings CSVs and API callers that still set both flags.
            if timelapse and settings.get("motility_analysis", False):
                from .timelapse import automated_motility_assay
                _ = automated_motility_assay(settings)
            
            if not np.any(mask_stack):
                avg_num_objects_per_image, average_obj_size = 0, 0
            else:
                avg_num_objects_per_image, average_obj_size = _get_avg_object_size(mask_stack)
            
            average_count.append(avg_num_objects_per_image)
            average_sizes.append(average_obj_size) 
            overall_average_size = np.mean(average_sizes) if len(average_sizes) > 0 else 0
            overall_average_count = np.mean(average_count) if len(average_count) > 0 else 0
            print(f'Found {overall_average_count} {object_type}/FOV. average size: {overall_average_size:.3f} px2')

            # Inside the per-batch loop, for the same reason as in
            # generate_cellpose_masks_sam: at the .npz level only the last
            # batch of a multi-batch file was ever written, and an empty .npz
            # reached the save block with mask_stack unbound.
            if not timelapse:
                if settings['plot']:
                    print(f"plotting")
                    plot_cellpose4_output(batch_list, masks, flows, cmap='inferno', figuresize=figuresize, nr=batch_size)

            if settings['save']:
                for mask_index, mask in enumerate(mask_stack):
                    output_filename = os.path.join(output_folder, batch_filenames[mask_index])
                    mask = mask.astype(np.uint16)
                    np.save(output_filename, mask)
                mask_stack = []
                batch_filenames = []

        gc.collect()
    torch.cuda.empty_cache()
    _run_seg_qc(src, settings, object_type)
    return


def generate_organelle_masks_sam(src, settings, object_type):
    """Generate organelle masks using one of several morphology-aware strategies.

    Supported morphology modes and backends:

    - ``spots``: punctate structures (lipid droplets, vesicles, peroxisomes) via
      ``otsu``, ``adaptive``, ``log``, ``dog``, ``cellpose``.
    - ``network``: filamentous/reticular structures (mitochondria, microtubules,
      ER tubules) via ``otsu``, ``adaptive``, ``ridge``, ``hysteresis``,
      ``cellpose``, ``unet``.
    - ``irregular``: irregular-shaped organelles (Golgi, ER cisternae, lysosomes)
      via ``otsu``, ``adaptive``, ``cellpose``.
    - ``ring``: hollow/ring-shaped structures (endosomes, autophagosomes) via
      ``otsu``, ``adaptive``, ``dog``, ``log``, ``cellpose``.

    :param src: Path to the mask source directory containing ``.npz`` stacks.
    :param settings: Configuration dict. Organelle-specific keys are prefixed
        with ``organelle_`` and are documented in ``_set_organelle_defaults``.
    :param object_type: Object label (typically ``'organelle'``); drives the
        output folder name ``<object_type>_mask_stack``.
    :returns: None. Masks are written as ``.npy`` files in
        ``<src>/<object_type>_mask_stack/``.
    """

    from .io import _create_database, _save_object_counts_to_database, _check_masks, _get_avg_object_size
    from .utils import _masks_to_masks_stack, _filter_cp_masks, prepare_batch_for_segmentation
    from .settings import _set_organelle_defaults
    from.plot import plot_organelle_output

    gc.collect()

    settings = _set_organelle_defaults(settings)

    # This generator has no 4-D path. Say so rather than returning 2-D masks
    # to a user whose settings said 4-D.
    _refuse_t_stack(settings, 'object.generate_organelle_masks_sam')

    morphology = settings['organelle_morphology']
    method = settings['organelle_method']

    # The merged .npz stack only contains the channels that map to
    # ENABLED object types, densely re-indexed (see
    # spacr.utils._get_cellpose_channels). Indexing it with the RAW
    # organelle_channel (e.g. 3) blows up when fewer than 4 objects
    # are active — "index 3 is out of bounds for axis 3 with size N".
    # Remap the organelle channel to its position in the compacted
    # stack the same way the cell/nucleus/pathogen paths do.
    _raw_organelle_channel = settings['organelle_channel']
    _extract = sorted({c for c in (settings.get('nucleus_channel'),
                                     settings.get('cell_channel'),
                                     settings.get('pathogen_channel'),
                                     settings.get('organelle_channel'))
                          if c is not None})
    _remap = {orig: new for new, orig in enumerate(_extract)}
    organelle_channel = _remap.get(_raw_organelle_channel,
                                     _raw_organelle_channel)

    _validate_organelle_settings(morphology, method)

    n_jobs = settings.get('n_jobs', 1)
    if n_jobs < 1:
        n_jobs = 1

    if settings['verbose']:
        import pandas as pd
        from IPython.display import display
        organ_keys = {k: v for k, v in settings.items() if k.startswith('organelle_')}
        df = pd.DataFrame(list(organ_keys.items()), columns=['setting_key', 'setting_value'])
        df['setting_value'] = df['setting_value'].apply(str)
        display(df)

    paths = [os.path.join(src, f) for f in os.listdir(src) if f.endswith('.npz')]
    if not paths:
        print(f'No .npz files found in {src}')
        return

    count_loc = os.path.join(os.path.dirname(src), 'measurements', 'measurements.db')
    os.makedirs(os.path.dirname(count_loc), exist_ok=True)
    _create_database(count_loc)

    batch_size = settings['batch_size']
    average_sizes = []
    average_counts = []
    time_ls = []

    # ------------------------------------------------------------------ #
    #  Load deep-learning model once (if needed)
    # ------------------------------------------------------------------ #
    dl_model = None
    is_dl_method = method in ('cellpose', 'unet')

    if method == 'cellpose':
        from .utils import _choose_model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dl_model = _choose_model(
            settings['organelle_model_name'],
            device,
            object_type=object_type,
            restore_type=None,
            object_settings=_build_object_settings(settings),
        )
    elif method == 'unet':
        dl_model = _load_unet_model(settings)

    # ------------------------------------------------------------------ #
    #  Build a serialisable settings subset for worker processes
    # ------------------------------------------------------------------ #
    classical_settings = _extract_classical_settings(settings)

    # ------------------------------------------------------------------ #
    #  Optionally load cell masks for per-cell masking
    # ------------------------------------------------------------------ #
    cell_mask_folder = None
    if settings.get('organelle_mask_within_cells', False):
        candidate = os.path.join(os.path.dirname(src), 'cell_mask_stack')
        if os.path.exists(candidate):
            cell_mask_folder = candidate
            print(f'Per-cell masking enabled, using cell masks from {candidate}')
        else:
            print(f'Warning: organelle_mask_within_cells=True but no cell_mask_stack found at {candidate}')

    # ------------------------------------------------------------------ #
    #  Main loop over .npz stacks
    # ------------------------------------------------------------------ #
    for file_index, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        output_folder = os.path.join(os.path.dirname(path), f'{object_type}_mask_stack')
        os.makedirs(output_folder, exist_ok=True)

        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']

        # Skip already-processed files
        existing = set(os.listdir(output_folder))
        todo_indices = [i for i, fn in enumerate(filenames) if fn not in existing]
        if not todo_indices:
            print(f'All files in {name} already processed. Skipping.')
            continue

        for i in range(0, stack.shape[0], batch_size):
            start = time.time()
            batch = stack[i: i + batch_size]
            batch_filenames = filenames[i: i + batch_size].tolist()

            # ---------------------------------------------------------- #
            #  Extract the organelle channel
            # ---------------------------------------------------------- #
            if organelle_channel is not None:
                if batch.ndim == 4:
                    img_batch = batch[:, :, :, organelle_channel].astype(np.float32)
                else:
                    img_batch = batch.astype(np.float32)
            else:
                if batch.ndim == 4:
                    img_batch = batch[:, :, :, 0].astype(np.float32)
                else:
                    img_batch = batch.astype(np.float32)

            # ---------------------------------------------------------- #
            #  Per-cell masking: zero out pixels outside cells
            # ---------------------------------------------------------- #
            if cell_mask_folder is not None:
                img_batch = _apply_cell_mask(img_batch, batch_filenames, cell_mask_folder)

            # ---------------------------------------------------------- #
            #  Preprocessing: rolling ball and/or CLAHE
            # ---------------------------------------------------------- #
            img_batch = _preprocess_batch(img_batch, settings)

            # ---------------------------------------------------------- #
            #  Segment
            # ---------------------------------------------------------- #
            if method == 'cellpose':
                masks = _segment_cellpose_sam(
                    img_batch, batch_filenames, dl_model, settings, object_type, output_folder)
            elif method == 'unet':
                masks = _segment_unet(img_batch, dl_model, settings)
            else:
                # CPU-bound classical methods — parallelise
                masks = _segment_classical_parallel(
                    img_batch, classical_settings, n_jobs=n_jobs,
                )

            if masks is None or len(masks) == 0:
                continue

            # ---------------------------------------------------------- #
            #  Post-process: size filter, border removal
            # ---------------------------------------------------------- #
            mask_stack = _postprocess_masks(
                masks,
                min_size=settings['organelle_min_size'],
                max_size=settings['organelle_max_size'],
                remove_border=settings['organelle_remove_border'],
            )

            _save_object_counts_to_database(
                mask_stack, object_type, batch_filenames, count_loc, added_string='',
            )

            # Stats
            if not np.any(mask_stack):
                avg_count, avg_size = 0, 0
            else:
                avg_count, avg_size = _get_avg_object_size(mask_stack)

            average_counts.append(avg_count)
            average_sizes.append(avg_size)
            overall_avg_count = np.mean(average_counts)
            overall_avg_size = np.mean(average_sizes)

            stop = time.time()
            duration = stop - start
            time_ls.append(duration)

            print(
                f'Found {overall_avg_count:.1f} {object_type}/FOV, '
                f'average size: {overall_avg_size:.1f} px2 '
                f'[batch {file_index+1}/{len(paths)}, {duration:.1f}s, '
                f'n_jobs={n_jobs if not is_dl_method else "GPU"}]'
            )
            
            # ---------------------------------------------------------- #
            #  Plot (if enabled)
            # ---------------------------------------------------------- #
            if settings.get('plot', False):
                plot_organelle_output(
                    img_batch[: len(mask_stack)],
                    mask_stack,
                    settings,
                    cmap='inferno',
                    figuresize=10,
                    nr=min(settings.get('examples_to_plot', 1), len(mask_stack)),
                    print_object_number=True,
                )

            # ---------------------------------------------------------- #
            #  Save
            # ---------------------------------------------------------- #
            if settings['save']:
                for mask_idx, mask in enumerate(mask_stack):
                    out_path = os.path.join(output_folder, batch_filenames[mask_idx])
                    np.save(out_path, mask.astype(np.uint16))
                mask_stack = []
                batch_filenames = []

            gc.collect()

    torch.cuda.empty_cache()
    _run_seg_qc(src, settings, object_type)
    return

def _validate_organelle_settings(morphology, method):
    """Raise early on invalid morphology / method combinations."""
    valid_morphologies = ('spots', 'network', 'irregular', 'ring')
    if morphology not in valid_morphologies:
        raise ValueError(
            f"organelle_morphology must be one of {valid_morphologies}, got '{morphology}'"
        )

    method_map = {
        'spots': ('otsu', 'adaptive', 'log', 'dog', 'cellpose'),
        'network': ('otsu', 'adaptive', 'ridge', 'hysteresis', 'cellpose', 'unet'),
        'irregular': ('otsu', 'adaptive', 'cellpose'),
        'ring': ('otsu', 'adaptive', 'dog', 'log', 'cellpose'),
    }
    valid_methods = method_map[morphology]
    if method not in valid_methods:
        raise ValueError(
            f"For morphology='{morphology}', method must be one of {valid_methods}, got '{method}'"
        )


def _build_object_settings(settings):
    """Build an object_settings dict expected by _choose_model / cellpose eval."""
    return {
        'model_name': settings['organelle_model_name'],
        'diameter': settings['organelle_diameter'],
        'minimum_size': settings['organelle_min_size'],
        'maximum_size': settings['organelle_max_size'],
        'resample': settings['organelle_resample'],
        'filter_size': False,
        'filter_intensity': False,
        'remove_border_objects': settings['organelle_remove_border'],
        'merge': False,
    }


def _extract_classical_settings(settings):
    """Return a pickle-safe subset of ``settings`` for classical segmentation workers."""
    keys = [
        'organelle_morphology', 'organelle_method',
        'organelle_min_size', 'organelle_max_size',
        # Spots
        'organelle_tophat_radius', 'organelle_watershed_spots',
        'organelle_log_min_sigma', 'organelle_log_max_sigma',
        'organelle_log_num_sigma', 'organelle_log_threshold',
        'organelle_dog_sigma_low', 'organelle_dog_sigma_high',
        # Network
        'organelle_ridge_sigmas', 'organelle_ridge_filter',
        'organelle_skeletonize', 'organelle_network_threshold',
        'organelle_hysteresis_low', 'organelle_hysteresis_high',
        # Irregular
        'organelle_adaptive_block_size', 'organelle_adaptive_offset',
        'organelle_morph_radius', 'organelle_fill_holes',
        # Ring
        'organelle_ring_sigma_inner', 'organelle_ring_sigma_outer',
        'organelle_ring_min_prominence', 'organelle_ring_fill_method',
    ]
    return {k: settings[k] for k in keys if k in settings}


# ====================================================================== #
#  Preprocessing
# ====================================================================== #

def _preprocess_batch(img_batch, settings):
    """Apply optional rolling-ball and/or CLAHE preprocessing to an (N,H,W) batch."""
    do_rolling_ball = settings.get('organelle_rolling_ball', False)
    do_clahe = settings.get('organelle_clahe', False)

    if not do_rolling_ball and not do_clahe:
        return img_batch

    out = img_batch.copy()

    for idx in range(out.shape[0]):
        img = out[idx]

        if do_rolling_ball:
            radius = settings.get('organelle_rolling_ball_radius', 50)
            bg = rolling_ball(img, radius=radius)
            img = img - bg
            img = np.clip(img, 0, None)

        if do_clahe:
            clip_limit = settings.get('organelle_clahe_clip_limit', 0.01)
            pmin, pmax = np.percentile(img, (0.5, 99.5))
            if pmax - pmin > 0:
                img_norm = np.clip((img - pmin) / (pmax - pmin), 0, 1)
            else:
                img_norm = np.zeros_like(img)
            img = equalize_adapthist(img_norm, clip_limit=clip_limit).astype(np.float32)

        out[idx] = img

    return out


def _apply_cell_mask(img_batch, batch_filenames, cell_mask_folder):
    """Zero out pixels outside cell boundaries for per-cell organelle detection."""
    out = img_batch.copy()
    for idx, fn in enumerate(batch_filenames):
        cell_mask_path = os.path.join(cell_mask_folder, fn)
        if os.path.exists(cell_mask_path):
            cell_mask = np.load(cell_mask_path)
            out[idx][cell_mask == 0] = 0
        else:
            cell_mask_path_npy = cell_mask_path if cell_mask_path.endswith('.npy') else cell_mask_path + '.npy'
            if os.path.exists(cell_mask_path_npy):
                cell_mask = np.load(cell_mask_path_npy)
                out[idx][cell_mask == 0] = 0
    return out


# ====================================================================== #
#  Deep-learning model loaders
# ====================================================================== #

def _load_unet_model(settings):
    """Load a user-provided U-Net model from a .pt / .pth file."""
    model_path = settings.get('organelle_unet_model_path')
    if model_path is None or not os.path.exists(model_path):
        raise ValueError(
            f"organelle_unet_model_path must point to a valid .pt/.pth file, "
            f"got '{model_path}'"
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


# ====================================================================== #
#  Cellpose segmentation
# ====================================================================== #

def _segment_cellpose(batch, batch_filenames, model, settings, object_type, output_folder):
    """Run Cellpose on a batch and return a list of 2-D label arrays."""
    from .utils import prepare_batch_for_segmentation
    from .io import _check_masks
    from .spacr_cellpose import parse_cellpose4_output

    # Remap raw object channels to their dense position in the
    # compacted stack (same reasoning as generate_organelle_masks_sam).
    _extract = sorted({c for c in (settings.get('nucleus_channel'),
                                     settings.get('cell_channel'),
                                     settings.get('pathogen_channel'),
                                     settings.get('organelle_channel'))
                          if c is not None})
    _remap = {orig: new for new, orig in enumerate(_extract)}
    organelle_ch = settings['organelle_channel']
    if organelle_ch is None:
        organelle_ch = 0
    else:
        organelle_ch = _remap.get(organelle_ch, organelle_ch)

    if batch.ndim == 4:
        organelle_ch = min(organelle_ch, batch.shape[3] - 1)
        ch0 = batch[:, :, :, organelle_ch: organelle_ch + 1]
        nuc_ch = settings.get('nucleus_channel')
        nuc_ch = _remap.get(nuc_ch, nuc_ch) if nuc_ch is not None else None
        if nuc_ch is not None and nuc_ch < batch.shape[3]:
            ch1 = batch[:, :, :, nuc_ch: nuc_ch + 1]
        else:
            ch1 = ch0
        cp_batch = np.concatenate([ch0, ch1], axis=-1).astype(batch.dtype)
    else:
        cp_batch = np.stack([batch, batch], axis=-1).astype(batch.dtype)

    if not settings.get('plot', False):
        cp_batch, batch_filenames = _check_masks(cp_batch, batch_filenames, output_folder)
    if cp_batch.size == 0:
        return None

    cp_batch = prepare_batch_for_segmentation(cp_batch)
    batch_list = [cp_batch[j] for j in range(cp_batch.shape[0])]

    output = model.eval(
        x=batch_list,
        batch_size=settings['batch_size'],
        normalize=False,
        channel_axis=-1,
        channels=[0, 1],
        diameter=settings['organelle_diameter'],
        flow_threshold=settings['organelle_FT'],
        cellprob_threshold=settings['organelle_CP_prob'],
        rescale=None,
        resample=settings['organelle_resample'],
    )

    masks, flows, _, _, _ = parse_cellpose4_output(output)
    return masks

def _segment_cellpose_sam(batch, batch_filenames, model, settings, object_type, output_folder):
    """Run Cellpose-SAM on a batch and return a list of 2-D label arrays."""
    from .utils import prepare_batch_for_segmentation
    from .io import _check_masks
    from .spacr_cellpose import parse_cellpose4_output

    if object_type == 'nucleus':
        selected_channels = [settings.get('nucleus_channel')]
    elif object_type == 'cell':
        selected_channels = [settings.get('cell_channel'), settings.get('nucleus_channel')]
    elif object_type == 'pathogen':
        selected_channels = [settings.get('pathogen_channel')]
    elif object_type == 'organelle':
        selected_channels = [settings.get('organelle_channel')]
    else:
        raise ValueError(f"Unsupported object_type: {object_type}")

    selected_channels = [ch for ch in selected_channels if ch is not None]

    if len(selected_channels) == 0:
        raise ValueError(f"No valid channels defined for object_type '{object_type}'.")

    if batch.ndim == 4:
        max_ch = batch.shape[3]
        selected_channels = [ch for ch in selected_channels if ch < max_ch]

        if len(selected_channels) == 0:
            raise ValueError(
                f"Selected channels for object_type '{object_type}' are out of bounds for batch with {max_ch} channels."
            )

        cp_batch = batch[:, :, :, selected_channels].astype(batch.dtype)

    elif batch.ndim == 3:
        cp_batch = batch[:, :, :, np.newaxis].astype(batch.dtype)

    else:
        raise ValueError(f"Expected batch with ndim 3 or 4, got ndim={batch.ndim}")

    if not settings.get('plot', False):
        cp_batch, batch_filenames = _check_masks(cp_batch, batch_filenames, output_folder)
    if cp_batch.size == 0:
        return None

    cp_batch = prepare_batch_for_segmentation(cp_batch)
    batch_list = [cp_batch[j] for j in range(cp_batch.shape[0])]

    output = model.eval(
        x=batch_list,
        batch_size=len(batch_list),
        normalize=False,
        channel_axis=-1,
        diameter=None,
        flow_threshold=settings[f'{object_type}_FT'],
        cellprob_threshold=settings[f'{object_type}_CP_prob'],
        resample=settings.get(f'{object_type}_resample', True)
    )

    masks, flows, _, _, _ = parse_cellpose4_output(output)
    return masks


# ====================================================================== #
#  U-Net semantic segmentation (GPU — not parallelised)
# ====================================================================== #

def _segment_unet(img_batch, model, settings):
    """Run a user-provided U-Net for semantic segmentation of network organelles.

    Expects a model that accepts ``(B, 1, H, W)`` and outputs
    ``(B, 1, H, W)`` logits; returns a list of 2-D integer label arrays.
    """
    device = next(model.parameters()).device
    threshold = settings.get('organelle_unet_threshold', 0.5)
    do_skeleton = settings.get('organelle_skeletonize', False)

    masks = []
    with torch.no_grad():
        for idx in range(img_batch.shape[0]):
            img = img_batch[idx]
            mean, std = img.mean(), img.std()
            if std > 0:
                img_norm = (img - mean) / std
            else:
                img_norm = np.zeros_like(img)

            tensor = torch.from_numpy(img_norm[None, None]).float().to(device)
            pred = model(tensor)

            if pred.shape[1] > 1:
                pred = pred[:, 0:1, :, :]

            pred = pred.sigmoid().cpu().numpy()[0, 0]
            binary = pred > threshold

            binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

            if do_skeleton:
                skeleton = skeletonize(binary)
                skeleton = binary_dilation(skeleton, disk(1))
                masks.append(sk_label(skeleton))
            else:
                masks.append(sk_label(binary))

    return masks


# ====================================================================== #
#  Classical segmentation — parallel dispatcher
# ====================================================================== #

def _segment_classical_parallel(img_batch, classical_settings, n_jobs=1):
    """Segment a batch using classical methods, sequential or via ``Pool``."""
    n_images = img_batch.shape[0]

    if n_jobs == 1 or n_images == 1:
        return [_segment_single_image(img_batch[idx], classical_settings)
                for idx in range(n_images)]

    effective_jobs = min(n_jobs, n_images, cpu_count())

    worker_fn = partial(_segment_single_image, settings=classical_settings)
    image_list = [img_batch[idx] for idx in range(n_images)]

    with Pool(processes=effective_jobs) as pool:
        masks = pool.map(worker_fn, image_list)

    return masks


def _segment_single_image(img, settings):
    """Dispatch a 2-D image to the morphology-specific segmentation routine."""
    morphology = settings['organelle_morphology']
    method = settings['organelle_method']

    if morphology == 'spots':
        return _segment_spots(img, method, settings)
    elif morphology == 'network':
        return _segment_network(img, method, settings)
    elif morphology == 'irregular':
        return _segment_irregular(img, method, settings)
    elif morphology == 'ring':
        return _segment_ring(img, method, settings)
    else:
        raise ValueError(f"Unknown morphology: {morphology}")


# ====================================================================== #
#  SPOTS segmentation
# ====================================================================== #

def _segment_spots(img, method, settings):
    """Segment punctate/spot-like organelles via ``otsu``, ``adaptive``, ``log`` or ``dog``."""
    tophat_radius = settings['organelle_tophat_radius']
    use_watershed = settings['organelle_watershed_spots']

    if method == 'log':
        return _spots_log(img, settings, use_watershed)
    elif method == 'dog':
        return _spots_dog(img, settings, use_watershed)

    # --- Pre-filter: white top-hat enhances bright spots on dark bg ---
    filtered = white_tophat(img, disk(tophat_radius))

    # --- Threshold ---
    if method == 'otsu':
        thresh_val = threshold_otsu(filtered)
        binary = filtered > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(filtered, block_size=block, offset=offset)
        binary = filtered > local_thresh
    else:
        raise ValueError(f"Unsupported spot method: {method}")

    # --- Morphological cleanup ---
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    # --- Watershed to split touching spots ---
    if use_watershed:
        labeled = _watershed_split(binary, filtered)
    else:
        labeled = sk_label(binary)

    return labeled


def _spots_log(img, settings, use_watershed):
    """LoG blob detection -> marker-seeded watershed."""
    min_s = settings['organelle_log_min_sigma']
    max_s = settings['organelle_log_max_sigma']
    num_s = settings['organelle_log_num_sigma']
    thresh = settings['organelle_log_threshold']

    img_norm = _normalize_01(img)

    blobs = blob_log(img_norm, min_sigma=min_s, max_sigma=max_s,
                     num_sigma=num_s, threshold=thresh)

    if len(blobs) == 0:
        return np.zeros(img.shape, dtype=np.int32)

    return _blobs_to_labels(blobs, img_norm, use_watershed)


def _spots_dog(img, settings, use_watershed):
    """DoG blob detection followed by an optional marker-seeded watershed."""
    sigma_low = settings.get('organelle_dog_sigma_low', 1.0)
    sigma_high = settings.get('organelle_dog_sigma_high', 3.0)
    thresh = settings['organelle_log_threshold']

    img_norm = _normalize_01(img)

    blobs = blob_dog(img_norm, min_sigma=sigma_low, max_sigma=sigma_high,
                     threshold=thresh)

    if len(blobs) == 0:
        return np.zeros(img.shape, dtype=np.int32)

    return _blobs_to_labels(blobs, img_norm, use_watershed)


def _blobs_to_labels(blobs, img_norm, use_watershed):
    """Convert ``(y, x, sigma)`` blob coordinates to a 2-D label image."""
    shape = img_norm.shape
    markers = np.zeros(shape, dtype=np.int32)
    for i, (y, x, sigma) in enumerate(blobs, start=1):
        y, x = int(round(y)), int(round(x))
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            markers[y, x] = i

    if not use_watershed:
        labeled = np.zeros(shape, dtype=np.int32)
        for i, (y, x, sigma) in enumerate(blobs, start=1):
            rr, cc = _circle_coords(int(round(y)), int(round(x)),
                                    max(int(round(sigma * np.sqrt(2))), 1),
                                    shape)
            labeled[rr, cc] = i
        return labeled

    smooth = gaussian(img_norm, sigma=1)
    labeled = watershed(-smooth, markers, mask=(smooth > np.percentile(smooth, 20)))
    return labeled


def _circle_coords(cy, cx, radius, shape):
    """Return (row, col) arrays for a filled circle clipped to shape."""
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    circle = yy ** 2 + xx ** 2 <= radius ** 2
    rows = np.clip(cy + np.where(circle)[0] - radius, 0, shape[0] - 1)
    cols = np.clip(cx + np.where(circle)[1] - radius, 0, shape[1] - 1)
    return rows, cols


# ====================================================================== #
#  NETWORK segmentation
# ====================================================================== #

def _segment_network(img, method, settings):
    """Segment filamentous/reticular organelles via ``otsu``, ``adaptive``, ``ridge`` or ``hysteresis``."""
    if method == 'ridge':
        return _network_ridge(img, settings)
    elif method == 'hysteresis':
        return _network_hysteresis(img, settings)

    smooth = gaussian(img, sigma=1)

    if method == 'otsu':
        thresh_val = threshold_otsu(smooth)
        binary = smooth > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(smooth, block_size=block, offset=offset)
        binary = smooth > local_thresh
    else:
        raise ValueError(f"Unsupported network method: {method}")

    morph_r = max(settings['organelle_morph_radius'] // 2, 1)
    binary = binary_closing(binary, disk(morph_r))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = binary_dilation(skeleton, disk(1))
        return sk_label(skeleton)

    return sk_label(binary)


def _network_ridge(img, settings):
    """Apply a ridge (tubeness) filter then threshold."""
    sigmas = settings['organelle_ridge_sigmas']
    filter_name = settings['organelle_ridge_filter']
    thresh_method = settings['organelle_network_threshold']

    img_norm = _normalize_01(img)

    ridge_filters = {
        'frangi': frangi,
        'sato': sato,
        'meijering': meijering,
    }
    if filter_name not in ridge_filters:
        raise ValueError(
            f"organelle_ridge_filter must be one of {list(ridge_filters.keys())}, "
            f"got '{filter_name}'"
        )

    enhanced = ridge_filters[filter_name](img_norm, sigmas=sigmas, black_ridges=False)

    if thresh_method == 'otsu':
        t = threshold_otsu(enhanced)
        binary = enhanced > t
    elif thresh_method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_t = threshold_local(enhanced, block_size=block, offset=offset)
        binary = enhanced > local_t
    else:
        t = threshold_otsu(enhanced)
        binary = enhanced > t

    binary = binary_closing(binary, disk(1))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = binary_dilation(skeleton, disk(1))
        return sk_label(skeleton)

    return sk_label(binary)


def _network_hysteresis(img, settings):
    """Dual-threshold hysteresis segmentation for network organelles.

    Values <1.0 for ``organelle_hysteresis_low`` / ``_high`` are interpreted as
    percentiles of the image; otherwise as absolute intensities.
    """
    low = settings['organelle_hysteresis_low']
    high = settings['organelle_hysteresis_high']

    smooth = gaussian(img, sigma=1)

    # Interpret values <1.0 as percentiles
    if low < 1.0:
        low = np.percentile(smooth, low * 100)
    if high < 1.0:
        high = np.percentile(smooth, high * 100)

    binary = apply_hysteresis_threshold(smooth, low, high)

    morph_r = max(settings['organelle_morph_radius'] // 2, 1)
    binary = binary_closing(binary, disk(morph_r))
    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = binary_dilation(skeleton, disk(1))
        return sk_label(skeleton)

    return sk_label(binary)


# ====================================================================== #
#  IRREGULAR segmentation
# ====================================================================== #

def _segment_irregular(img, method, settings):
    """Segment irregular organelles (Golgi, ER cisternae, lysosomes) via ``otsu`` or ``adaptive``."""
    morph_r = settings['organelle_morph_radius']
    fill_area = settings['organelle_fill_holes']

    smooth = gaussian(img, sigma=max(morph_r / 2, 1))

    if method == 'otsu':
        thresh_val = threshold_otsu(smooth)
        binary = smooth > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(smooth, block_size=block, offset=offset)
        binary = smooth > local_thresh
    else:
        raise ValueError(f"Unsupported irregular method: {method}")

    selem = disk(morph_r)
    binary = binary_closing(binary, selem)
    binary = binary_opening(binary, selem)

    if fill_area > 0:
        binary = remove_small_holes(binary, area_threshold=fill_area)

    binary = remove_small_objects(binary, min_size=settings['organelle_min_size'])

    labeled = _watershed_split(binary, smooth)
    return labeled


# ====================================================================== #
#  RING segmentation
# ====================================================================== #

def _segment_ring(img, method, settings):
    """Segment hollow/ring-shaped organelles by DoG edge enhancement + fill + shape filter.

    Uses ``organelle_ring_sigma_inner`` / ``_outer`` for DoG scales,
    ``organelle_ring_min_prominence`` to discard non-ring objects, and
    ``organelle_ring_fill_method`` (``'flood'`` or ``'convex'``) for the fill step.
    """
    sigma_inner = settings.get('organelle_ring_sigma_inner', 1.0)
    sigma_outer = settings.get('organelle_ring_sigma_outer', 3.0)
    min_prominence = settings.get('organelle_ring_min_prominence', 0.1)
    fill_method = settings.get('organelle_ring_fill_method', 'flood')

    # Step 1: Enhance ring structures using DoG (edge enhancement)
    img_norm = _normalize_01(img)
    enhanced = np.abs(difference_of_gaussians(img_norm, sigma_inner, sigma_outer))

    # Step 2: Threshold the enhanced image
    if method == 'otsu':
        thresh_val = threshold_otsu(enhanced)
        binary_edges = enhanced > thresh_val
    elif method == 'adaptive':
        block = settings['organelle_adaptive_block_size']
        offset = settings['organelle_adaptive_offset']
        local_thresh = threshold_local(enhanced, block_size=block, offset=offset)
        binary_edges = enhanced > local_thresh
    elif method == 'log':
        blobs = blob_log(img_norm,
                         min_sigma=settings['organelle_log_min_sigma'],
                         max_sigma=settings['organelle_log_max_sigma'],
                         num_sigma=settings['organelle_log_num_sigma'],
                         threshold=settings['organelle_log_threshold'])
        if len(blobs) == 0:
            return np.zeros(img.shape, dtype=np.int32)
        thresh_val = threshold_otsu(enhanced)
        binary_edges = enhanced > thresh_val
    elif method == 'dog':
        thresh_val = threshold_otsu(enhanced)
        binary_edges = enhanced > thresh_val
    else:
        raise ValueError(f"Unsupported ring method: {method}")

    # Cleanup edges
    binary_edges = binary_closing(binary_edges, disk(1))
    binary_edges = remove_small_objects(binary_edges, min_size=max(settings['organelle_min_size'] // 4, 3))

    # Step 3: Fill rings to get solid objects
    if fill_method == 'flood':
        filled = _fill_rings_flood(binary_edges)
    elif fill_method == 'convex':
        filled = _fill_rings_convex(binary_edges)
    else:
        filled = _fill_rings_flood(binary_edges)

    # Step 4: Remove objects that lack ring morphology
    labeled = sk_label(filled)
    labeled = _filter_non_rings(labeled, binary_edges, img_norm, min_prominence)

    return labeled


def _fill_rings_flood(binary_edges):
    """Fill ring interiors by treating non-border background components as interiors."""
    inverted = ~binary_edges
    labeled_bg = sk_label(inverted)

    border_labels = set()
    border_labels.update(labeled_bg[0, :].ravel())
    border_labels.update(labeled_bg[-1, :].ravel())
    border_labels.update(labeled_bg[:, 0].ravel())
    border_labels.update(labeled_bg[:, -1].ravel())

    filled = binary_edges.copy()
    for region in regionprops(labeled_bg):
        if region.label not in border_labels:
            filled[labeled_bg == region.label] = True

    return filled


def _fill_rings_convex(binary_edges):
    """Fill rings using the convex hull of each connected edge component."""
    from skimage.morphology import convex_hull_image

    labeled_edges = sk_label(binary_edges)
    filled = np.zeros_like(binary_edges)

    for region in regionprops(labeled_edges):
        minr, minc, maxr, maxc = region.bbox
        component = labeled_edges[minr:maxr, minc:maxc] == region.label
        hull = convex_hull_image(component)
        filled[minr:maxr, minc:maxc] |= hull

    return filled


def _filter_non_rings(labeled, binary_edges, img_norm, min_prominence):
    """Drop objects whose boundary-vs-interior contrast falls below ``min_prominence``."""
    props = regionprops(labeled, intensity_image=img_norm)
    output = labeled.copy()

    for prop in props:
        mask = labeled == prop.label
        edge_mask = mask & binary_edges
        interior_mask = mask & ~binary_edges

        if np.sum(edge_mask) == 0 or np.sum(interior_mask) == 0:
            edge_ratio = np.sum(edge_mask) / max(np.sum(mask), 1)
            if edge_ratio < 0.3:
                output[mask] = 0
            continue

        mean_edge = img_norm[edge_mask].mean()
        mean_interior = img_norm[interior_mask].mean()
        object_mean = img_norm[mask].mean()

        if object_mean > 0:
            prominence = abs(mean_edge - mean_interior) / object_mean
        else:
            prominence = 0

        if prominence < min_prominence:
            output[mask] = 0

    return sk_label(output > 0)


# ====================================================================== #
#  Shared helpers
# ====================================================================== #

def _normalize_01(img):
    """Percentile-based normalisation to [0, 1]."""
    img_norm = img.astype(np.float64)
    pmin, pmax = np.percentile(img_norm, (1, 99))
    if pmax - pmin > 0:
        img_norm = np.clip((img_norm - pmin) / (pmax - pmin), 0, 1)
    else:
        img_norm = np.zeros_like(img_norm)
    return img_norm


def _watershed_split(binary, intensity):
    """Marker-controlled watershed on a binary mask using distance-transform peaks."""
    distance = distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=5, labels=binary)
    if len(coords) == 0:
        return sk_label(binary)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    labeled = watershed(-distance, markers, mask=binary)
    return labeled


def _postprocess_masks(masks, min_size=10, max_size=None, remove_border=False):
    """Return each label mask with size filtering and optional border-object removal."""
    processed = []
    for mask in masks:
        mask = mask.copy()

        if remove_border:
            border_labels = set()
            border_labels.update(mask[0, :].ravel())
            border_labels.update(mask[-1, :].ravel())
            border_labels.update(mask[:, 0].ravel())
            border_labels.update(mask[:, -1].ravel())
            border_labels.discard(0)
            for lbl in border_labels:
                mask[mask == lbl] = 0

        if min_size > 0 or max_size is not None:
            props = regionprops(mask)
            for prop in props:
                if prop.area < min_size:
                    mask[mask == prop.label] = 0
                elif max_size is not None and prop.area > max_size:
                    mask[mask == prop.label] = 0

        mask = sk_label(mask > 0)
        processed.append(mask)

    return processed

