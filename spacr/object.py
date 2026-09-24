"""Object segmentation, filtering, mask generation, and post-processing."""

import os, torch, time

from . import _gc as gc
from .mask_io import _as_uint16_mask

from . import accelerator
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
try:
    from IPython.display import display
except Exception:
    def display(*args, **kwargs):
        """Discard display payloads when IPython's helper is unavailable."""
        pass
import warnings
from cellpose import models as cp_models

from functools import partial
from skimage.segmentation import watershed
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.filters import (threshold_otsu,threshold_local,frangi,sato,meijering,gaussian,difference_of_gaussians,apply_hysteresis_threshold)
from skimage.feature import blob_log, blob_dog, peak_local_max
from skimage.morphology import (
    closing, dilation, disk, opening, remove_small_holes,
    remove_small_objects, skeletonize, white_tophat,
)
from skimage.exposure import equalize_adapthist
from skimage.restoration import rolling_ball

warnings.filterwarnings("ignore", message="3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")


def _eval_diameter(raw, object_type=""):
    """The diameter to hand Cellpose's ``eval``, or ``None`` for native scale.

    Cellpose tests ``diameter > 0``, so a value that reaches it as a STRING
    raises ``TypeError: '>' not supported between instances of 'str' and
    'int'`` -- and it raises inside the segmentation call, after the run has
    already spent minutes loading and normalising plates. Every route into
    this setting that is not a Python literal produces a string: a number
    typed into the GUI, and any settings CSV.

    ``None`` is PRESERVED rather than replaced with a default. A blank
    diameter means "let CPSAM work at native scale"; substituting the
    magnification-derived default from `_get_object_settings` would rescale
    every image by 30/diameter -- a different segmentation, silently, for
    every run that left the field empty. That is why this coerces the user's
    own value instead of reading `object_settings['diameter']`, which always
    holds that default.

    An unparseable value is reported and treated as blank, which is what
    `_get_object_settings` has always done for this field.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        print(f"{object_type}_diameter must be a number, got {raw!r}; "
              f"segmenting at native scale instead")
        return None


def _remove_objects_smaller_than(binary, min_size):
    """Remove components with area strictly below ``min_size``.

    scikit-image 0.26 renamed ``min_size`` to ``max_size`` and changed the
    boundary from ``<`` to ``<=``. Passing ``min_size - 1`` preserves spaCR's
    historical threshold exactly. The fallback keeps compatibility with the
    declared 0.22-0.25 range.
    """
    threshold = max(0, int(min_size) - 1)
    try:
        return remove_small_objects(binary, max_size=threshold)
    except TypeError:
        return remove_small_objects(binary, min_size=int(min_size))


def _fill_holes_smaller_than(binary, area_threshold):
    """Fill holes with area strictly below ``area_threshold``."""
    threshold = max(0, int(area_threshold) - 1)
    try:
        return remove_small_holes(binary, max_size=threshold)
    except TypeError:
        return remove_small_holes(
            binary, area_threshold=int(area_threshold))

def merge_split_filter_masks(masks, intensity_images, settings, object_type, batch_filenames=None):
    """Merge by perimeter and filter each in-memory field's objects.

    Skips work when no operation is enabled for ``object_type``; otherwise
    processes each FOV serially so progress reporting stays in order.

    :param masks: 2D/3D ndarray or iterable of masks (one per field).
    :param intensity_images: Original own-channel arrays matching the masks,
        required only when an intensity bound is enabled. For channel-last
        batches the first channel must be the object's own channel.
    :param settings: Dict of pipeline settings; per-object-type suffixes control
        perimeter merging, min/max area, border removal and min/max intensity.
        Intensity bounds compare whole-object means in original image units;
        equality is retained and 0 disables each bound independently.
    :param object_type: Label used to look up per-object settings (``'cell'``,
        ``'nucleus'``, ``'pathogen'``, ``'organelle'``).
    :param batch_filenames: Optional per-FOV filenames used only for logging.
    :returns: Original ``masks`` unchanged when no operation is enabled, else a
        list of filtered mask arrays (one per FOV).
    """
    import numpy as np
    from .utils import (print_progress, _process_single_fov_in_memory,
                        _validated_intensity_bounds)

    pf = settings.get(f'{object_type}_perimeter_fraction', settings.get(f'{object_type}_perimiter_fraction', 0))
    mna = settings.get(f'{object_type}_min_area', 0)
    mxa = settings.get(f'{object_type}_max_area', 0)
    rb = settings.get(f'{object_type}_remove_border_objects', False)
    minimum, maximum = _validated_intensity_bounds(
        settings.get(f'{object_type}_min_intensity', 0),
        settings.get(f'{object_type}_max_intensity', 0))

    needs_work = (
        pf > 0 or mna > 0 or (mxa and mxa > 0) or rb or
        minimum > 0 or maximum > 0
    )

    if not needs_work:
        print(f"merge_split_filter_masks({object_type}): no operations needed, skipping")
        return masks

    if masks is None:
        return None

    print(f"merge_split_filter_masks({object_type}): "
          f"perimeter_merge={pf > 0}(frac={pf}), "
          f"min_area={mna}, max_area={mxa}, remove_border={rb}, "
          f"min_intensity={minimum}, max_intensity={maximum}")

    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            raise ValueError(f"Unsupported masks ndim: {masks.ndim}")
    else:
        mask_list = list(masks)

    if intensity_images is None:
        intensity_list = [None] * len(mask_list)
    elif isinstance(intensity_images, np.ndarray):
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
            do_perimeter_merge=(pf > 0),
            perimeter_fraction=pf,
            min_area=mna,
            max_area=mxa if mxa else 0,
            remove_border_objects=rb,
            min_intensity=minimum,
            max_intensity=maximum,
            progress_callback=_progress,
            fov_index=idx,
            total_fovs=total,
            op_name=f'merge_{object_type}',
        )
        return out_mask

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
        print(f"Segmentation QC skipped for {object_type}: {type(exc).__name__}: {exc}")
        return None

    if result is not None and result.get('mode') == 'flag':
        settings.setdefault('seg_qc_flags', {})[object_type] = result['flags']
    return result



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
        """Return labels from a 3-D, plane-list, or single-plane Cellpose call."""
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
            planes = [array[z] for z in range(array.shape[0])]
            kwargs['batch_size'] = len(planes)
            output = model.eval(x=planes, **kwargs)
            return np.asarray(output[0])

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
        only, the projected ``(N, Y, X, C)`` normalized model-input array.
        These values are not raw intensity-filter units; absolute bounds use
        original own-channel planes loaded separately by
        :func:`_raw_filter_images`.
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
        intensity = np.stack([
            project(volume, mode=z_plan.projection, z_axis=z_axis)
            for volume in volumes
        ])

    return masks, results, intensity




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

    ``t_stack`` reads time and z from the declared acquisition axes. What is
    missing from an ordinary batch is the **z** axis: ``(N, Y, X, C)`` has
    four axes where a 4-D acquisition needs five.

    A spec with ``z_axis=None`` describes a flat ``(T, Y, X, C)`` time series
    and needs only four, which is what an ordinary batch already is -- see
    :func:`spacr.zstack.segment_4d`, which makes one plain 2-D call per frame
    for it. ``t_axis_order='TYX'`` declares that flat case explicitly; the
    legacy ``timelapse`` setting also supports flat time series without a
    t-stack plan.

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
        projected ``(T, Y, X, C)`` normalized model-input array. These values
        are not raw intensity-filter units; absolute bounds use original
        own-channel planes loaded separately by :func:`_raw_filter_images`.
    """
    from .zstack import iter_volumes, project, segment_4d

    segment_fn = _cellpose_z_segment_fn(
        model, eval_kwargs, t_plan.stitch_threshold
    )

    result = segment_4d(acquisition, t_plan, segment_fn=segment_fn)
    masks = [np.asarray(frame) for frame in np.asarray(result.labels)]

    intensity = None
    if t_plan.z_axis is None:
        pass
    elif t_plan.z_mode == 'project':
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


def _raw_filter_images(src, filenames, model_inputs, masks, channel, *,
                       z_axis=None, projection=None):
    """Read original own-channel values on the canvas the model segmented.

    Filenames are the surviving resume manifest, while the canvas comes
    from the retained normalized batch, not the size of surviving fields.
    Projected volumes reuse the model's projection, including its focus
    plane choice. Whole-volume labels keep the complete original z axis.
    """
    from .zstack import _best_focus_index, project

    if not (len(filenames) == len(model_inputs) == len(masks)):
        raise ValueError("Raw intensity fields, model inputs and masks must align")
    if channel is None:
        raise ValueError("Intensity filtering requires an explicit own-channel index")
    result = []
    for filename, model_input, mask in zip(filenames, model_inputs, masks):
        filename = str(filename)
        if os.path.basename(filename) != filename:
            raise ValueError("Raw intensity filenames must be field basenames")
        raw = np.load(os.path.join(os.path.dirname(src), 'stack', filename))
        canvas = np.shape(model_input)[:-1]
        if raw.ndim == len(canvas) and int(channel) == 0:
            plane = raw
        elif raw.ndim == len(canvas) + 1 and 0 <= int(channel) < raw.shape[-1]:
            plane = raw[..., int(channel)]
        else:
            raise ValueError(f"Raw intensity shape/channel mismatch for {filename}")
        if any(actual > target for actual, target in zip(plane.shape, canvas)):
            raise ValueError(f"Raw intensity field exceeds segmentation canvas: {filename}")
        plane = np.pad(plane, [(0, target - actual)
                              for actual, target in zip(plane.shape, canvas)])
        if z_axis is not None:
            plane = np.moveaxis(plane, z_axis, 0)
            if np.ndim(mask) == plane.ndim - 1:
                if plane.shape[0] == 1:
                    plane = plane[0]
                elif projection == 'best_focus':
                    selected = np.moveaxis(model_input, z_axis, 0)
                    plane = plane[_best_focus_index(selected)]
                else:
                    plane = project(plane, mode=projection, z_axis=0)
        if plane.shape != np.shape(mask):
            raise ValueError(f"Raw intensity plane must have the same shape as mask: {filename}")
        result.append(plane)
    return result


def _assigned_mask_archives(src, batch_paths):
    """Validate an explicit worker assignment without changing output roots."""
    if isinstance(batch_paths, (str, bytes, os.PathLike)):
        raise ValueError('batch_paths must be a sequence of NPZ paths')
    root = os.path.realpath(os.fspath(src))
    selected = []
    for value in batch_paths:
        path = os.fspath(value)
        if not os.path.isabs(path):
            path = os.path.join(root, path)
        path = os.path.abspath(path)
        if (os.path.realpath(os.path.dirname(path)) != root
                or os.path.basename(path).startswith('.')
                or not path.endswith('.npz') or not os.path.isfile(path)):
            raise ValueError(f'Mask batch is not a prepared NPZ under {root}: {path}')
        if path in selected:
            raise ValueError(f'Mask batch assigned more than once: {path}')
        selected.append(path)
    return selected


def generate_cellpose_masks_sam(src, settings, object_type, *, batch_paths=None,
                                on_batch_done=None, run_qc=True):
    """Segment one object channel across all ``.npz`` batches under ``src`` using Cellpose-SAM.

    Loads the ``cpsam`` pretrained model — or, when
    ``<object_type>_model_name`` (or ``pathogen_model``) names a checkpoint
    the user trained, that checkpoint — iterates over each pre-batched
    ``.npz`` file, applies perimeter merging and area/border filtering to 2-D
    masks, and optionally filters objects by their absolute mean intensity
    in the original own-channel image. It then optionally tracks timelapse
    objects, saves per-image ``.npy`` masks, and records per-object counts to
    the run's SQLite database. Time-stack archives must contain one filename
    per timepoint, regardless of the declared time-axis position; each raw
    filename identifies that timepoint's ``(Z, Y, X, C)`` volume, or its
    ``(Y, X, C)`` image for a flat ``TYX`` series.

    :param src: Directory containing the pre-batched ``.npz`` image stacks.
    :param settings: Pipeline settings dict; canonicalized via
        :func:`spacr.settings.set_default_settings_preprocess_generate_masks`.
    :param object_type: ``'cell'``, ``'nucleus'``, ``'pathogen'`` or
        ``'organelle'``; drives channel/threshold lookups and output folder name.
    :param batch_paths: optional exclusive worker assignment of NPZ paths under
        ``src``. One model is reused across the assignment; ``None`` keeps the
        ordinary whole-directory run. An empty assignment loads no model.
    :param on_batch_done: optional callable receiving the archive path after
        its selected fields have completed. Failed archives are not reported.
    :param run_qc: False lets a parallel coordinator run shared QC once after
        every worker finishes, instead of writing reports from each worker.
    :returns: None.
    """
    from .utils import (_masks_to_masks_stack, all_elements_match,
                        prepare_batch_for_segmentation, _get_cellpose_channels,
                        _resolve_cellpose_pretrained)
    from .io import (_check_masks, _create_database, _get_avg_object_size,
                     _listdir_visible, _save_array_atomic,
                     _save_object_counts_to_database)
    from .timelapse import (_npz_to_movie, _btrack_track_cells, _trackpy_track_cells,
                            _trackastra_track_cells, _ultrack_track_cells)
    from .plot import plot_cellpose4_output
    from .settings import set_default_settings_preprocess_generate_masks, _get_object_settings
    from .spacr_cellpose import parse_cellpose4_output
    from .cancellation import checkpoint as cancellation_checkpoint
    from dataclasses import replace
    from .zstack import as_t_first

    if on_batch_done is not None and not callable(on_batch_done):
        raise ValueError('on_batch_done must be callable or None')
    paths = (_assigned_mask_archives(src, batch_paths) if batch_paths is not None
             else [os.path.join(src, file) for file in _listdir_visible(src)
                   if file.endswith('.npz')])
    if batch_paths is not None and not paths:
        return
    
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
    timelapse = settings.get('timelapse', False)

    if timelapse:
        timelapse_displacement = settings['timelapse_displacement']
        timelapse_frame_limits = settings['timelapse_frame_limits']
        timelapse_memory = settings['timelapse_memory']
        timelapse_remove_transient = settings['timelapse_remove_transient']
        timelapse_mode = settings['timelapse_mode']
        timelapse_objects = settings['timelapse_objects']
    
    batch_size = settings['batch_size']
    
    cellprob_threshold = settings[f'{object_type}_cellprob_threshold']
    flow_threshold = settings[f'{object_type}_flow_threshold']
    object_settings = _get_object_settings(object_type, settings)

    z_plan = _z_stack_plan(settings)

    t_plan = _t_stack_plan(settings)
    z_plan = _reconcile_z_and_t_plans(z_plan, t_plan, timelapse=timelapse)

    from .utils import _validated_intensity_bounds
    intensity_bounds = _validated_intensity_bounds(
        settings.get(f'{object_type}_min_intensity', 0),
        settings.get(f'{object_type}_max_intensity', 0))
    filter_by_raw_intensity = any(value > 0 for value in intensity_bounds)

    if t_plan is not None:
        beta_mode = None if t_plan.z_axis is None else t_plan.z_mode
    elif z_plan is not None:
        beta_mode = z_plan.mode
    else:
        beta_mode = None

    from .utils import dense_mask_channel_positions

    _dense = dense_mask_channel_positions(settings)
    for _role in ('nucleus', 'cell', 'pathogen', 'organelle'):
        if settings.get(f'cellpose_{_role}_channel') is not None:
            continue
        _raw = settings.get(f'{_role}_channel')
        if _raw is None:
            continue
        try:
            _raw = int(_raw)
        except (TypeError, ValueError):
            continue
        settings[f'cellpose_{_role}_channel'] = _dense[_raw]

    channels_to_extract, cellpose_channels = _get_cellpose_channels(settings)
    channels = cellpose_channels.get(object_type, [])
    
    if len(channels) == 0:
        raise ValueError(f"No valid channels defined for object_type '{object_type}'.")
        
    if settings['verbose']:
        print(channels)

    model_name = object_settings['model_name']
    if object_type == 'pathogen' and settings.get('pathogen_model') is not None:
        model_name = settings['pathogen_model']
    # Items 404/405: DINOCell and SAMCell answer the same model.eval call and
    # return Cellpose's (masks, flows, styles), so this is the only dispatch.
    from ._segmentation_backends import _backend_name, _load_backend
    segmentation_backend = _backend_name(
        settings.get('segmentation_backend', 'cellpose'))
    if segmentation_backend == 'cellpose':
        pretrained = _resolve_cellpose_pretrained(model_name, object_type=object_type)
        model = cp_models.CellposeModel(
            pretrained_model=pretrained,
            **accelerator.cellpose_kwargs(),
        )
    else:
        model = _load_backend(segmentation_backend, z_plan=z_plan,
                              t_plan=t_plan, model_name=model_name,
                              object_type=object_type)
    count_loc = os.path.dirname(src)+'/measurements/measurements.db'
    os.makedirs(os.path.dirname(src)+'/measurements', exist_ok=True)
    _create_database(count_loc)
    
    average_sizes = []
    average_count = []
    for file_index, path in enumerate(paths):
        cancellation_checkpoint()
        name = os.path.basename(path)
        name, ext = os.path.splitext(name)
        output_folder = os.path.join(os.path.dirname(path), object_type+'_mask_stack')
        os.makedirs(output_folder, exist_ok=True)
        overall_average_size = 0
        
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']

        # Filename selection, resume and batching all operate on timepoints.
        # Canonicalize each archive with the original acquisition plan, then
        # give the segmenter a local plan for this view. Mutating t_plan here
        # would interpret later ZTYX archives as if they were already TZYX.
        archive_t_plan = t_plan
        if t_plan is not None:
            _require_t_axis(stack, t_plan, path)
            stack = as_t_first(stack, t_plan)
            if filenames.ndim != 1 or len(filenames) != stack.shape[0]:
                raise ValueError(
                    f"t_stack requires one filename per timepoint in "
                    f"{os.path.basename(path)}: time axis has length "
                    f"{stack.shape[0]}, filenames have shape {filenames.shape}")
            archive_t_plan = replace(
                t_plan, t_axis=0,
                z_axis=1 if t_plan.z_axis is not None else None)
        elif z_plan is not None:
            _require_z_axis(stack, z_plan, path)

        for filename in filenames:
            output_path = os.path.join(output_folder, filename)
            if os.path.exists(output_path):
                print(f"File {filename} already exists in the output folder. Skipping...")
                
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
                    stack = stack[timelapse_frame_limits[0]: timelapse_frame_limits[1]]
                    filenames = filenames[timelapse_frame_limits[0]: timelapse_frame_limits[1]]
                    batch_size = len(stack)
                    print(f'Cut batch at indecies: {timelapse_frame_limits}, New batch_size: {batch_size} ')

        if len(stack) == 0:
            if on_batch_done is not None:
                on_batch_done(path)
            continue

        for i in range(0, stack.shape[0], batch_size):
            cancellation_checkpoint()
            mask_stack = []
            if z_plan is not None or t_plan is not None:
                batch = stack[i: i+batch_size][..., channels].astype(stack.dtype)
            elif stack.shape[3] == 1:
                batch = stack[i: i+batch_size, :, :, [0]].astype(stack.dtype)
            else:
                batch = stack[i: i+batch_size, :, :, channels].astype(stack.dtype)


            batch_filenames = filenames[i: i+batch_size].tolist()
            from .image_quality import filter_batch
            batch, batch_filenames = filter_batch(batch, batch_filenames, settings)

            if not settings['plot']:
                batch, batch_filenames = _check_masks(
                    batch, batch_filenames, output_folder,
                    resume=settings.get('resume', False))
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
                    diameter=_eval_diameter(
                        settings.get(f'{object_type}_diameter'),
                        object_type),
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    resample=object_settings['resample']
                    )

                masks, flows, _, _, _ = parse_cellpose4_output(output)
            else:
                z_eval_kwargs = dict(
                    batch_size=1,
                    normalize=False,
                    channel_axis=-1,
                    min_size=object_settings['min_size'],
                    progress=True,
                    diameter=_eval_diameter(
                        settings.get(f'{object_type}_diameter'),
                        object_type),
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    resample=object_settings['resample'],
                )
                if t_plan is not None:
                    masks, t_result, beta_intensity = _segment_timepoints_with_t(
                        cp_batch, model, archive_t_plan, z_eval_kwargs
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

            filter_images = batch if beta_mode is None else beta_intensity
            if filter_by_raw_intensity:
                filter_z_axis = (0 if archive_t_plan is not None and archive_t_plan.z_axis is not None
                                 else (z_plan.z_axis or 0) if z_plan is not None
                                 else None)
                projection = (archive_t_plan.projection if archive_t_plan is not None
                              else z_plan.projection if z_plan is not None else None)
                filter_images = _raw_filter_images(
                    src, batch_filenames, batch_list, masks,
                    settings.get(f'{object_type}_channel'),
                    z_axis=filter_z_axis, projection=projection)

            if beta_mode is None or beta_mode == 'project' or all(
                    np.ndim(mask) == 2 for mask in masks):
                masks = merge_split_filter_masks(
                    masks=masks,
                    intensity_images=filter_images,
                    settings=settings,
                    object_type=object_type,
                    batch_filenames=batch_filenames,
                )
            else:
                print(
                    f"merge_split_filter_masks({object_type}): skipped — the "
                    f"perimeter and area operations are 2-D only and would be "
                    f"applied per z plane, breaking the 3-D labels that "
                    f"z_segmentation_mode='{beta_mode}' just produced"
                )
                if filter_by_raw_intensity:
                    from .utils import _filter_objects
                    masks = [_filter_objects(
                        np.asarray(mask).copy(), plane,
                        min_intensity=intensity_bounds[0], max_intensity=intensity_bounds[1])
                        for mask, plane in zip(masks, filter_images)]
            
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

            if not timelapse:
                if settings['plot']:
                    if flows is None:
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
                mask_stack = [_as_uint16_mask(mask) for mask in mask_stack]
                for mask_index, mask in enumerate(mask_stack):
                    output_filename = os.path.join(output_folder, batch_filenames[mask_index])
                    _save_array_atomic(output_filename, mask)
                mask_stack = []
                batch_filenames = []

        gc.collect()
        if on_batch_done is not None:
            on_batch_done(path)

    torch.cuda.empty_cache()
    if run_qc:
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
    from .io import (_check_masks, _create_database, _get_avg_object_size,
                     _listdir_visible, _save_array_atomic,
                     _save_object_counts_to_database)
    from .timelapse import _npz_to_movie, _btrack_track_cells, _trackpy_track_cells
    from .plot import plot_cellpose4_output
    from .settings import set_default_settings_preprocess_generate_masks, _get_object_settings
    from .spacr_cellpose import parse_cellpose4_output
    from .cancellation import checkpoint as cancellation_checkpoint
    
    gc.collect()
    if not torch.cuda.is_available():
        print(f'Torch CUDA is not available, using CPU')
        
    settings['src'] = src
    
    settings = set_default_settings_preprocess_generate_masks(settings)

    _refuse_t_stack(settings, 'object.generate_cellpose_masks')

    if settings['verbose']:
        settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])
        settings_df['setting_value'] = settings_df['setting_value'].apply(str)
        display(settings_df)
        
    figuresize=10
    timelapse = settings.get('timelapse', False)

    if timelapse:
        timelapse_displacement = settings['timelapse_displacement']
        timelapse_frame_limits = settings['timelapse_frame_limits']
        timelapse_memory = settings['timelapse_memory']
        timelapse_remove_transient = settings['timelapse_remove_transient']
        timelapse_mode = settings['timelapse_mode']
        timelapse_objects = settings['timelapse_objects']
    
    batch_size = settings['batch_size']
    
    cellprob_threshold = settings[f'{object_type}_cellprob_threshold']

    flow_threshold = settings[f'{object_type}_flow_threshold']

    object_settings = _get_object_settings(object_type, settings)
    
    model_name = object_settings['model_name']
    
    from .utils import dense_mask_channel_positions

    _dense = dense_mask_channel_positions(settings)
    for _role in ('nucleus', 'cell', 'pathogen', 'organelle'):
        if settings.get(f'cellpose_{_role}_channel') is not None:
            continue
        _raw = settings.get(f'{_role}_channel')
        if _raw is None:
            continue
        try:
            _raw = int(_raw)
        except (TypeError, ValueError):
            continue
        settings[f'cellpose_{_role}_channel'] = _dense[_raw]

    channels_to_extract, cellpose_channels = _get_cellpose_channels(settings)

    if settings['verbose']:
        print(cellpose_channels)
        
    if object_type not in cellpose_channels:
        raise ValueError(f"Error: No channels were specified for object_type '{object_type}'. Check your settings.")
    
    channels = cellpose_channels[object_type]

    device = accelerator.torch_device()
    
    if object_type == 'pathogen' and not settings['pathogen_model'] is None:
        model_name = settings['pathogen_model']
    
    model = _choose_model(model_name, device, object_type=object_type, restore_type=None, object_settings=object_settings)

    
    paths = [os.path.join(src, file) for file in _listdir_visible(src) if file.endswith('.npz')]    
    
    count_loc = os.path.dirname(src)+'/measurements/measurements.db'
    os.makedirs(os.path.dirname(src)+'/measurements', exist_ok=True)
    _create_database(count_loc)
    
    average_sizes = []
    average_count = []
    for file_index, path in enumerate(paths):
        cancellation_checkpoint()
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
            cancellation_checkpoint()
            mask_stack = []
            if stack.shape[3] == 1:
                batch = stack[i: i+batch_size, :, :, [0,0]].astype(stack.dtype)
            else:
                batch = stack[i: i+batch_size, :, :, channels].astype(stack.dtype)

            batch_filenames = filenames[i: i+batch_size].tolist()
            from .image_quality import filter_batch
            batch, batch_filenames = filter_batch(batch, batch_filenames, settings)

            if not settings['plot']:
                batch, batch_filenames = _check_masks(
                    batch, batch_filenames, output_folder,
                    resume=settings.get('resume', False))
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
                    mask_stack = _masks_to_masks_stack(masks)
        
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

            if not timelapse:
                if settings['plot']:
                    print(f"plotting")
                    plot_cellpose4_output(batch_list, masks, flows, cmap='inferno', figuresize=figuresize, nr=batch_size)

            if settings['save']:
                mask_stack = [_as_uint16_mask(mask) for mask in mask_stack]
                for mask_index, mask in enumerate(mask_stack):
                    output_filename = os.path.join(output_folder, batch_filenames[mask_index])
                    _save_array_atomic(output_filename, mask)
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

    from .io import (_check_masks, _create_database, _get_avg_object_size,
                     _listdir_visible, _save_array_atomic,
                     _save_object_counts_to_database)
    from .settings import _set_organelle_defaults
    from .object_roles import organelle_settings_view
    from.plot import plot_organelle_output
    from .cancellation import checkpoint as cancellation_checkpoint

    gc.collect()

    settings = organelle_settings_view(
        _set_organelle_defaults(settings), object_type)

    from .utils import _validated_intensity_bounds
    intensity_bounds = _validated_intensity_bounds(
        settings.get('organelle_min_intensity', 0),
        settings.get('organelle_max_intensity', 0))
    filter_by_raw_intensity = any(value > 0 for value in intensity_bounds)
    settings['organelle_remove_border_objects'] = bool(
        settings.get('organelle_remove_border_objects', False)
        or settings.get('organelle_remove_border', False))

    _refuse_t_stack(settings, 'object.generate_organelle_masks_sam')

    morphology = settings['organelle_morphology']
    method = settings['organelle_method']

    from .utils import dense_mask_channel_positions

    _raw_organelle_channel = settings['organelle_channel']
    _recorded = settings.get('cellpose_organelle_channel')
    if _recorded is not None:
        organelle_channel = int(_recorded)
    else:
        _positions = dense_mask_channel_positions(settings)
        organelle_channel = _positions.get(_raw_organelle_channel,
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

    paths = [os.path.join(src, f) for f in _listdir_visible(src) if f.endswith('.npz')]
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

    dl_model = None
    is_dl_method = method in ('cellpose', 'unet')

    if method == 'cellpose':
        from .utils import _choose_model
        device = accelerator.torch_device()
        dl_model = _choose_model(
            settings['organelle_model_name'],
            device,
            object_type=object_type,
            restore_type=None,
            object_settings=_build_object_settings(settings),
        )
    elif method == 'unet':
        dl_model = _load_unet_model(settings)

    classical_settings = _extract_classical_settings(settings)

    cell_mask_folder = None
    if settings.get('organelle_mask_within_cells', False):
        candidate = os.path.join(os.path.dirname(src), 'cell_mask_stack')
        if os.path.exists(candidate):
            cell_mask_folder = candidate
            print(f'Per-cell masking enabled, using cell masks from {candidate}')
        else:
            print(f'Warning: organelle_mask_within_cells=True but no cell_mask_stack found at {candidate}')

    for file_index, path in enumerate(paths):
        cancellation_checkpoint()
        output_folder = os.path.join(os.path.dirname(path), f'{object_type}_mask_stack')
        os.makedirs(output_folder, exist_ok=True)

        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']

        fields_skipped = 0
        fields_attempted = 0
        for i in range(0, stack.shape[0], batch_size):
            cancellation_checkpoint()
            start = time.time()
            batch = stack[i: i + batch_size]
            batch_filenames = filenames[i: i + batch_size].tolist()
            from .image_quality import filter_batch
            batch, batch_filenames = filter_batch(batch, batch_filenames, settings)
            if not settings.get('plot', False):
                offered = len(batch_filenames)
                batch, batch_filenames = _check_masks(
                    batch, batch_filenames, output_folder,
                    resume=settings.get('resume', False))
                fields_skipped += offered - len(batch_filenames)
            if batch.size == 0:
                continue
            fields_attempted += len(batch_filenames)

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

            if cell_mask_folder is not None:
                img_batch = _apply_cell_mask(img_batch, batch_filenames, cell_mask_folder)

            img_batch = _preprocess_batch(img_batch, settings)

            if method == 'cellpose':
                masks = _segment_cellpose_sam(
                    img_batch, batch_filenames, dl_model, settings, object_type, output_folder)
            elif method == 'unet':
                masks = _segment_unet(img_batch, dl_model, settings)
            else:
                masks = _segment_classical_parallel(
                    img_batch, classical_settings, n_jobs=n_jobs,
                )

            if masks is None or len(masks) == 0:
                continue

            raw_images = None
            if filter_by_raw_intensity:
                inputs = [image if image.ndim == 3 else image[..., None]
                          for image in batch]
                raw_images = _raw_filter_images(
                    src, batch_filenames, inputs, masks,
                    settings['organelle_channel'])
            mask_stack = merge_split_filter_masks(
                masks, raw_images, settings, 'organelle', batch_filenames,
            )

            _save_object_counts_to_database(
                mask_stack, object_type, batch_filenames, count_loc, added_string='',
            )

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

            if settings['save']:
                mask_stack = [_as_uint16_mask(mask) for mask in mask_stack]
                for mask_idx, mask in enumerate(mask_stack):
                    out_path = os.path.join(output_folder, batch_filenames[mask_idx])
                    _save_array_atomic(out_path, mask)
                mask_stack = []
                batch_filenames = []

            gc.collect()

        if fields_skipped and not fields_attempted:
            print(f'All files in {os.path.basename(path)} already processed. '
                  f'Skipping.')
        elif fields_skipped:
            print(f'{fields_skipped} of {int(stack.shape[0])} files in '
                  f'{os.path.basename(path)} already processed. Skipping those.')

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
        'minimum_size': settings['organelle_min_area'],
        'maximum_size': settings['organelle_max_area'],
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
        'organelle_min_area', 'organelle_max_area',
        'organelle_tophat_radius', 'organelle_watershed_spots',
        'organelle_log_min_sigma', 'organelle_log_max_sigma',
        'organelle_log_num_sigma', 'organelle_log_threshold',
        'organelle_dog_sigma_low', 'organelle_dog_sigma_high',
        'organelle_ridge_sigmas', 'organelle_ridge_filter',
        'organelle_skeletonize', 'organelle_network_threshold',
        'organelle_hysteresis_low', 'organelle_hysteresis_high',
        'organelle_adaptive_block_size', 'organelle_adaptive_offset',
        'organelle_morph_radius', 'organelle_fill_holes',
        'organelle_ring_sigma_inner', 'organelle_ring_sigma_outer',
        'organelle_ring_min_prominence', 'organelle_ring_fill_method',
    ]
    return {k: settings[k] for k in keys if k in settings}



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



def _load_unet_model(settings):
    """Load a user-provided U-Net model from a .pt / .pth file."""
    model_path = settings.get('organelle_unet_model_path')
    if model_path is None or not os.path.exists(model_path):
        raise ValueError(
            f"organelle_unet_model_path must point to a valid .pt/.pth file, "
            f"got '{model_path}'"
        )
    device = accelerator.torch_device()
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model



def _segment_cellpose(batch, batch_filenames, model, settings, object_type, output_folder):
    """Run Cellpose on a batch and return a list of 2-D label arrays."""
    from .utils import prepare_batch_for_segmentation
    from .io import _check_masks
    from .spacr_cellpose import parse_cellpose4_output

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
        cp_batch, batch_filenames = _check_masks(
            cp_batch, batch_filenames, output_folder,
            resume=settings.get('resume', False))
    if cp_batch.size == 0:
        return None

    cp_batch = prepare_batch_for_segmentation(cp_batch)
    batch_list = [cp_batch[j] for j in range(cp_batch.shape[0])]

    output = model.eval(
        x=batch_list,
        batch_size=settings['batch_size'],
        normalize=False,
        channel_axis=-1,
        diameter=settings['organelle_diameter'],
        flow_threshold=settings['organelle_flow_threshold'],
        cellprob_threshold=settings['organelle_cellprob_threshold'],
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
    else:
        from .object_roles import ORGANELLE_ROLES
        if object_type not in ORGANELLE_ROLES:
            raise ValueError(f"Unsupported object_type: {object_type}")
        selected_channels = [settings.get('organelle_channel')]

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
        cp_batch, batch_filenames = _check_masks(
            cp_batch, batch_filenames, output_folder,
            resume=settings.get('resume', False))
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
        flow_threshold=settings[f'{object_type}_flow_threshold'],
        cellprob_threshold=settings[f'{object_type}_cellprob_threshold'],
        resample=settings.get(f'{object_type}_resample', True)
    )

    masks, flows, _, _, _ = parse_cellpose4_output(output)
    return masks



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

            binary = _remove_objects_smaller_than(
                binary, settings['organelle_min_area'])

            if do_skeleton:
                skeleton = skeletonize(binary)
                skeleton = dilation(skeleton, disk(1))
                masks.append(sk_label(skeleton))
            else:
                masks.append(sk_label(binary))

    return masks



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



def _segment_spots(img, method, settings):
    """Segment punctate/spot-like organelles via ``otsu``, ``adaptive``, ``log`` or ``dog``."""
    tophat_radius = settings['organelle_tophat_radius']
    use_watershed = settings['organelle_watershed_spots']

    if method == 'log':
        return _spots_log(img, settings, use_watershed)
    elif method == 'dog':
        return _spots_dog(img, settings, use_watershed)

    filtered = white_tophat(img, disk(tophat_radius))

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

    binary = opening(binary, disk(1))
    binary = _remove_objects_smaller_than(
        binary, settings['organelle_min_area'])

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
    binary = closing(binary, disk(morph_r))
    binary = _remove_objects_smaller_than(
        binary, settings['organelle_min_area'])

    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = dilation(skeleton, disk(1))
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

    binary = closing(binary, disk(1))
    binary = _remove_objects_smaller_than(
        binary, settings['organelle_min_area'])

    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = dilation(skeleton, disk(1))
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

    if low < 1.0:
        low = np.percentile(smooth, low * 100)
    if high < 1.0:
        high = np.percentile(smooth, high * 100)

    binary = apply_hysteresis_threshold(smooth, low, high)

    morph_r = max(settings['organelle_morph_radius'] // 2, 1)
    binary = closing(binary, disk(morph_r))
    binary = _remove_objects_smaller_than(
        binary, settings['organelle_min_area'])

    if settings['organelle_skeletonize']:
        skeleton = skeletonize(binary)
        skeleton = dilation(skeleton, disk(1))
        return sk_label(skeleton)

    return sk_label(binary)



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
    binary = closing(binary, selem)
    binary = opening(binary, selem)

    if fill_area > 0:
        binary = _fill_holes_smaller_than(binary, fill_area)

    binary = _remove_objects_smaller_than(
        binary, settings['organelle_min_area'])

    labeled = _watershed_split(binary, smooth)
    return labeled



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

    img_norm = _normalize_01(img)
    enhanced = np.abs(difference_of_gaussians(img_norm, sigma_inner, sigma_outer))

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

    binary_edges = closing(binary_edges, disk(1))
    binary_edges = _remove_objects_smaller_than(
        binary_edges, max(settings['organelle_min_area'] // 4, 3))

    if fill_method == 'flood':
        filled = _fill_rings_flood(binary_edges)
    elif fill_method == 'convex':
        filled = _fill_rings_convex(binary_edges)
    else:
        filled = _fill_rings_flood(binary_edges)

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
