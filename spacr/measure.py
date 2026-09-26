"""Turn masks and channels into one row per object, in a database.

WHAT IT IS FOR. Segmentation says WHERE the objects are; this module says
what they are LIKE. It reads the arrays Mask wrote and produces the table
every downstream question is asked of -- which genes changed a phenotype,
which cells to train a classifier on, which wells to believe.

WHAT IT NEEDS. A ``merged/`` folder written by
:func:`spacr.core.preprocess_generate_masks`: the intensity channels and the
label masks for one field, saved together as ``.npy``. Which masks to measure
is named per object -- ``cell_mask_dim``, ``nucleus_mask_dim``,
``pathogen_mask_dim`` and the organelle slots -- and an object with no mask
dimension is simply not measured, rather than measured as empty.

WHAT IT PRODUCES.

* ``measurements/measurements.db``, one SQLite table per object type, one row
  per object, keyed by the plate/row/column/field/object identity
  :mod:`spacr.schema` composes. The columns are shape, intensity, texture and
  SPATIAL features -- how many neighbours an object has within a radius, how
  far the nearest one is, what fraction of its border touches another.
* Optionally, one PNG per object (``save_png``), cropped by the mask. Those
  crops are what :func:`spacr.deep_spacr.deep_spacr` trains on and what
  Annotate shows, which is why the cropping lives here rather than beside the
  classifier: they must be cut by the same mask the measurements came from.

WHAT TO DO NEXT. Annotate or Classify, if the crops were written; Regression,
if the question is which perturbation moved which measurement. Both read the
database this writes and neither re-measures anything.

--------------------------------------------------------------------------

THREE THINGS THAT ARE NOT OBVIOUS AND ARE LOAD-BEARING:

A FIELD THAT FAILS TO MEASURE IS RECORDED, SUMMARISED AND STAMPED INTO THE
DATABASE. Silence would let a regression analyse 344 of 384 wells and report
a result with no sign that forty are missing, which is the failure this
module is most careful about -- the same reason its 3-D path refuses a volume
it cannot measure correctly instead of measuring it wrongly.

THE 2-D PATH IS BIT-IDENTICAL AND DELIBERATELY SO. Mask can emit ``(Z, Y, X)``
label volumes now (see :mod:`spacr.zstack`), and everything about voxel
spacing, volume columns and the units stamp exists so that a 3-D field is
measured in real units or refused. A 2-D field takes exactly the code it took
before, with ``spacing=None``; a screen measured last year and re-measured
today produces the same numbers.

THE RADIUS IS IN THE COLUMN NAME. ``neighbors_within_30`` is a different
column from ``neighbors_within_50``, following the same precedent as
``homogeneity_distance_<d>``, so two plates measured at different radii will
not silently concatenate into one frame that means two things.

Illumination correction and a user-drawn ROI reach this module through the
registries in :mod:`spacr.measure_hooks` rather than by editing it. Both are
empty by default and both entry points return their input unchanged when they
are, so an ordinary run is byte-identical to one from before they existed.
"""

import os, cv2, time, sqlite3, threading, traceback, shutil, inspect
import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field as dataclasses_field
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr, skew, kurtosis, mode
import multiprocessing as mp
from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_dilation, binary_erosion, gaussian_filter, center_of_mass, convolve, find_objects
from scipy.spatial import cKDTree
from skimage.measure import regionprops, regionprops_table, shannon_entropy
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries, expand_labels
from skimage.feature import graycomatrix, graycoprops
from skimage import morphology, measure, filters
from skimage.util import img_as_bool
import matplotlib.pyplot as plt
from math import ceil, sqrt

from .crops import (
    DEFAULT_MASK_DIMS,
    MASK_PLANE_ORDER,
    build_png_channels,
    narrow_to_uint8,
    reconcile_merged_mask_dims,
    read_merged_plane_layout,
    resolve_png_channel_mapping,
    stamp_crop_folder,
    to_cv2_bgr,
)
from . import settings as _settings_module
settings = _settings_module
from . import measurement_schema as _measurement_schema
MEASUREMENT_STAMP_COLUMNS = _measurement_schema.MEASUREMENT_STAMP_COLUMNS
from .errors import RunLedger, ConfigurationError, raise_if_strict
from .runctx import run_context
from .resume import plan_measure_resume
from .measure_hooks import (
    MeasurementHookError,
    PreprocessingContext,
    RegionContext,
    apply_preprocessing_hooks,
    apply_region_filter_hooks,
    preprocessing_hooks,
    region_filter_hooks,
    register_preprocessing_hook,
    register_region_filter_hook,
    unregister_preprocessing_hook,
    unregister_region_filter_hook,
    warn_if_hooks_will_not_reach_workers,
)
from .object_roles import ORGANELLE_ROLES, SEGMENTED_ROLES
from .intensity_rescale import (
    PLAN_SETTINGS_KEY,
    build_plate_plan,
    mask_planes as _intensity_mask_planes,
    needs_warning as _intensity_scale_needs_warning,
    resolve_record as _resolve_intensity_rescale_record,
)

from .figures.style import figure_style, theme_target



#: The morphology properties a 2-D run measures. Unchanged from before 3-D
#: support existed; :data:`PROPS_2D_ONLY` is what a 3-D run drops from it.
MORPHOLOGICAL_PROPS = [
    'label', 'area', 'area_filled', 'area_bbox', 'convex_area',
    'major_axis_length', 'minor_axis_length', 'eccentricity', 'solidity',
    'extent', 'perimeter', 'euler_number', 'equivalent_diameter_area',
    'feret_diameter_max',
]

#: regionprops properties skimage implements for 2-D only. Asking for either on
#: a 3-D label volume raises ``NotImplementedError`` *for the whole
#: regionprops_table call*, so one 2-D-only name in the list costs every other
#: property too. A 3-D run drops them, which makes them absent rather than
#: wrong -- there is no meaningful 3-D "eccentricity" of a solid, and skimage's
#: 2-D ``perimeter`` is a boundary length, whose 3-D analogue is a surface area
#: in different units and must not share the name.
PROPS_2D_ONLY = ('eccentricity', 'perimeter')

#: 2-D run: raw pixels, exactly as spaCR has always written.
UNITS_PX = 'px'
#: 3-D run with a known ``anisotropy`` but no physical voxel size. Lengths are
#: in xy-pixel units and z has been scaled by ``dz/dxy``, so the numbers are
#: anisotropy-corrected but not physical.
UNITS_PX_XY = 'px_xy'
#: 3-D run with a known ``voxel_size_z_um``/``voxel_size_xy_um``. Lengths in
#: um, areas in um^2, volumes in um^3.
UNITS_UM = 'um'

def _ndim_of(mask):
    """Return the number of spatial dimensions of a label mask (2 or 3)."""
    return int(np.asarray(mask).ndim)


#: Cores left free when spaCR picks the worker count itself, so an interactive
#: machine stays usable during a measure run.
N_JOBS_HEADROOM = 4

#: Environment variable that overrides the multiprocessing start method the
#: measure pool runs in. Accepts any name :func:`multiprocessing.get_context`
#: accepts on the platform -- ``fork``, ``spawn`` or ``forkserver``. Unset (the
#: normal case) means "whatever this interpreter's default is", which is
#: ``fork`` on Linux today, ``spawn`` on Windows and macOS.
START_METHOD_ENV_VAR = 'SPACR_START_METHOD'


def _pool_context():
    """Return the multiprocessing context :func:`measure_crop` runs its pool in.

    spaCR deliberately does **not** call ``set_start_method(force=True)`` here.
    That mutates ``multiprocessing._default_context`` for the whole
    interpreter, irreversibly and invisibly to whoever imported spaCR, and it
    is what makes the start method impossible to reason about once the Tk GUI
    has been opened once. Taking a context object instead keeps the decision
    local to this pool.

    With :data:`START_METHOD_ENV_VAR` unset this returns the
    :mod:`multiprocessing` module itself rather than a context object. That is
    not laziness: ``mp.Pool`` / ``mp.Manager`` are then looked up exactly as
    they were before this function existed, so the default behaviour -- and
    anything that patches those two names -- is unchanged. A context is only
    substituted when a start method was asked for explicitly.

    :returns: an object exposing ``Pool``, ``Manager`` and ``get_start_method``
        -- either a :class:`multiprocessing.context.BaseContext` or the
        :mod:`multiprocessing` module.
    """
    method = os.environ.get(START_METHOD_ENV_VAR, '').strip().lower()
    if not method:
        return mp
    try:
        return mp.get_context(method)
    except ValueError:
        print(f"WARNING: {START_METHOD_ENV_VAR}={method!r} is not a "
              f"multiprocessing start method on this platform; using the "
              f"default ({mp.get_start_method()}).")
        return mp


class ManagerStartError(ConfigurationError):
    """Raised when Measure cannot start its multiprocessing manager.

    The exception message reports the active start method, underlying error,
    and practical remedies. No fields are measured after this error.
    """


def _thread_census():
    """Return ``(count, description)`` of the live threads in this process.

    The thread count is the whole diagnosis for a ``fork`` Manager failure, so
    it is measured at the moment of failure rather than described in prose.
    Names are truncated because a Qt process can carry dozens and the message
    has to stay readable.
    """
    threads = list(threading.enumerate())
    names = [t.name for t in threads]
    shown = ', '.join(names[:8])
    if len(names) > 8:
        shown += f", ... (+{len(names) - 8} more)"
    return len(threads), shown


def _manager_start_diagnosis(start_method, exc):
    """Build the message :class:`ManagerStartError` carries.

    Split out from :func:`_start_manager` so the wording is testable without
    breaking a Manager, and because the two cases genuinely differ:

    ``fork`` -- the case that actually bites. ``os.fork()`` duplicates only the
    calling thread but duplicates *all* of the process's memory, including
    every mutex the other threads were holding at the instant of the fork.
    Those mutexes arrive in the child already locked, owned by threads that do
    not exist there, so nothing can ever release them. The Manager's server
    process then deadlocks (or dies) before it writes its socket address back
    down the bootstrap pipe, and the parent's read of that address hits EOF --
    which is the naked ``EOFError`` from ``connection.py`` a user sees. A
    long-lived Qt or Jupyter process is exactly the thread-rich parent this
    needs; ``python -c`` forks with one thread and never reproduces it.

    Anything else (``spawn``, ``forkserver``) -- the child is a fresh
    interpreter that inherits no locks, so the thread census is reported but
    not blamed. What is left is what the Manager's server needs from the
    environment: a writable temp directory for its socket, and permission to
    start a process at all. Containers and HPC job sandboxes remove both.

    :param start_method: the start method the failed Manager was using.
    :param exc: the exception ``Manager()`` raised.
    :returns: a multi-line diagnostic string.
    """
    n_threads, thread_names = _thread_census()
    remedy = (
        f"    export {START_METHOD_ENV_VAR}=spawn\n"
        f"or, in Python, before calling measure_crop:\n"
        f"    os.environ['{START_METHOD_ENV_VAR}'] = 'spawn'"
    )
    head = (
        f"Could not start the multiprocessing Manager that measure_crop uses "
        f"to share per-field timings with its worker pool. Nothing was "
        f"measured.\n"
        f"  start method:    {start_method!r}\n"
        f"  underlying error: {type(exc).__name__}: {exc}\n"
        f"  live threads in this process: {n_threads} ({thread_names})\n"
    )

    if start_method == 'fork':
        return (
            head +
            f"\nMost likely cause: this process is forking with "
            f"{n_threads} live threads. os.fork() copies one thread but all of "
            f"the memory, so every lock the other {max(n_threads - 1, 0)} "
            f"thread(s) held arrives in the child already locked and owned by "
            f"nobody. The Manager's server then hangs or dies before writing "
            f"its address back to the parent, and the parent's read of that "
            f"address is the EOFError above. A long-lived Qt or Jupyter "
            f"session is exactly this kind of parent.\n"
            f"\nRemedy: run the measure pool under 'spawn', which starts each "
            f"child from a fresh interpreter and inherits no locks:\n"
            f"{remedy}\n"
            f"spaCR does not switch for you, because a spawn worker re-imports "
            f"the measure chain from cold (seconds and hundreds of MB each); "
            f"the worker count is capped at the number of fields under spawn, "
            f"so that cost is bounded but not free."
        )

    return (
        head +
        f"\nUnder {start_method!r} the child inherits no locks from the "
        f"parent, so the {n_threads} live thread(s) above are reported for "
        f"completeness rather than blamed. What a Manager still needs is a "
        f"writable temporary directory for its server's socket (TMPDIR, or "
        f"XDG_RUNTIME_DIR) and permission to start a process at all -- "
        f"containers and HPC job sandboxes commonly withhold both.\n"
        f"\nIf this machine's default is workable, unset "
        f"{START_METHOD_ENV_VAR}; otherwise select a start method explicitly:\n"
        f"{remedy}"
    )


def _start_manager(ctx):
    """Return a started :class:`multiprocessing.Manager` from ``ctx``.

    :param ctx: the object :func:`_pool_context` returned.
    :returns: a started manager, ready to use as a context manager.
    :raises ManagerStartError: ``Manager()`` failed, for any reason.

    ``BaseException`` is deliberately not caught: a Ctrl-C landing inside the
    Manager handshake is a cancellation, not a misconfiguration, and dressing
    it up as one would be a lie in the traceback.
    """
    try:
        return ctx.Manager()
    except Exception as exc:
        try:
            start_method = ctx.get_start_method()
        except Exception:
            start_method = mp.get_start_method()
        raise ManagerStartError(
            _manager_start_diagnosis(start_method, exc)) from exc


def resolve_pool_size(n_jobs, n_files, start_method=None):
    """Return the worker count for a set of image fields.

    ``spawn`` and ``forkserver`` start a fresh interpreter for every worker,
    so their worker count is capped at the number of fields. ``fork`` keeps
    the requested count for compatibility.

    :param n_jobs: the resolved worker count from :func:`resolve_n_jobs`.
    :param n_files: how many fields there are to measure.
    :param start_method: start method name to decide against; defaults to the
        interpreter's current default.
    :returns: an int >= 1.
    """
    n_jobs = max(1, int(n_jobs))
    if start_method is None:
        start_method = mp.get_start_method()
    if start_method == 'fork':
        return n_jobs
    return max(1, min(n_jobs, int(n_files)))


def resolve_n_jobs(n_jobs, cpu_count=None):
    """Return the number of worker processes ``measure_crop`` will actually use.

    ``None`` selects spaCR's default. Explicit values are validated and capped
    at the available CPU count.

    :param n_jobs: what the user asked for. ``None`` means "pick for me".
    :param cpu_count: core count to resolve against; defaults to
        :func:`multiprocessing.cpu_count`.
    :returns: an int in ``[1, cpu_count]``.
    :raises spacr.errors.ConfigurationError: ``n_jobs`` is zero, negative, or
        not an integer. A pool of zero workers measures nothing, and quietly
        turning it into some other number is how a run ends up not doing what
        it was told.
    """
    cores = max(1, int(mp.cpu_count() if cpu_count is None else cpu_count))

    if n_jobs is None:
        return max(1, cores - N_JOBS_HEADROOM)

    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
        raise ConfigurationError(
            f"settings['n_jobs'] = {n_jobs!r} must be an integer number of "
            f"worker processes, or None to let spaCR choose.")

    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ConfigurationError(
            f"settings['n_jobs'] = {n_jobs} must be at least 1. A pool of "
            f"{n_jobs} workers would measure nothing; leave n_jobs blank "
            f"(None) to let spaCR choose.")

    if n_jobs > cores:
        print(f"n_jobs={n_jobs} exceeds the {cores} available cores; using "
              f"{cores}. Leave n_jobs blank to let spaCR choose.")
        return cores
    return n_jobs


def resolve_measurement_spacing(settings, ndim, n_z=1):
    """Return ``(spacing, stamp)`` for a measurement of ``ndim`` spatial dimensions.

    ``spacing`` is handed straight to :func:`skimage.measure.regionprops_table`
    and (as ``sampling``) to :func:`scipy.ndimage.distance_transform_edt`.
    ``stamp`` is the dict of :data:`MEASUREMENT_STAMP_COLUMNS` written onto
    every row so the units are recorded rather than inferred.

    2-D returns ``(None, px stamp)`` unconditionally. Even when a voxel size
    is configured it is not applied, so a 2-D run is numerically identical to
    every spaCR run before this function existed.

    **3-D requires a z/xy relationship and will not invent one.** With
    anisotropic voxels an unspaced volume is not merely in unusual units: a
    voxel count is not proportional to a physical volume, a distance transform
    measures a different length along z than along x, and ``major_axis_length``
    mixes the two. This mirrors :func:`spacr.zstack.resolve_anisotropy`, which
    raises rather than defaulting to 1.0 because "isotropic" is a claim about
    the microscope, not a neutral value. Set ``voxel_size_z_um`` and
    ``voxel_size_xy_um`` (preferred -- it also gives physical units), or set
    ``anisotropy`` alone (correct geometry, xy-pixel units).

    :param settings: measure settings dict; reads ``voxel_size_z_um``,
        ``voxel_size_xy_um`` and ``anisotropy``.
    :param ndim: 2 or 3.
    :param n_z: number of z planes behind the measurement; 1 for a 2-D field.
    :returns: ``(spacing, stamp)``. ``spacing`` is ``None`` for 2-D, a
        ``(dz, dy, dx)`` tuple for 3-D.
    :raises spacr.zstack.UnknownAnisotropyError: 3-D without a voxel size or
        anisotropy.
    :raises spacr.errors.ConfigurationError: ``ndim`` is neither 2 nor 3, or a
        supplied voxel size is not a positive finite number.
    """
    cfg = settings or {}
    stamp = {
        'measurement_ndim': int(ndim),
        'measurement_units': UNITS_PX,
        'n_z': int(n_z),
        'voxel_size_z_um': None,
        'voxel_size_xy_um': None,
    }

    if ndim == 2:
        return None, stamp

    if ndim != 3:
        raise ConfigurationError(
            f"spacr.measure can measure 2-D masks and 3-D (Z, Y, X) label "
            f"volumes; got a {ndim}-dimensional mask. A 4-D (T, Z, Y, X) "
            f"acquisition is measured one timepoint at a time.")

    from .zstack import UnknownAnisotropyError

    def _positive(name):
        """One spacing value, refused unless it is a positive number.

        A zero or negative spacing makes every physical measurement wrong by a
        factor nobody can recover afterwards, so it is refused rather than
        defaulted.
        """
        value = cfg.get(name)
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ConfigurationError(
                f"settings['{name}'] = {cfg.get(name)!r} must be a finite "
                f"number > 0 (a physical size in micrometres).")
        return value

    dz = _positive('voxel_size_z_um')
    dxy = _positive('voxel_size_xy_um')
    anisotropy = cfg.get('anisotropy')
    if anisotropy is not None:
        anisotropy = float(anisotropy)
        if not np.isfinite(anisotropy) or anisotropy <= 0:
            raise ConfigurationError(
                f"settings['anisotropy'] = {cfg.get('anisotropy')!r} must be a "
                f"finite number > 0; it is the ratio dz / dxy.")

    if dz is not None and dxy is not None:
        stamp['measurement_units'] = UNITS_UM
        stamp['voxel_size_z_um'] = dz
        stamp['voxel_size_xy_um'] = dxy
        return (dz, dxy, dxy), stamp

    if anisotropy is not None:
        stamp['measurement_units'] = UNITS_PX_XY
        return (anisotropy, 1.0, 1.0), stamp

    raise UnknownAnisotropyError(
        "measuring a 3-D (Z, Y, X) mask needs to know how the z step relates "
        "to the xy pixel size, and spaCR will not assume they are equal. On a "
        "confocal stack dz is routinely 3-10x dxy, so an unspaced volume "
        "measurement is wrong by that factor along one axis: the voxel count "
        "in `<object>_area` is not proportional to a physical volume, and "
        "`major_axis_length`, `feret_diameter_max` and every distance-derived "
        "feature mix two different lengths. Set voxel_size_z_um and "
        "voxel_size_xy_um (which also converts volumes to um^3), or set "
        "anisotropy = dz / dxy on its own (correct geometry, xy-pixel units). "
        f"Got voxel_size_z_um={cfg.get('voxel_size_z_um')!r}, "
        f"voxel_size_xy_um={cfg.get('voxel_size_xy_um')!r}, "
        f"anisotropy={cfg.get('anisotropy')!r}.")


def _voxel_volume_columns(mask, labels, stamp):
    """Return the explicit volume columns a 3-D morphology frame carries.

    ``<object>_area`` in a 3-D row is a volume, which the stamp records -- but
    a column whose *name* carries its unit cannot be misread at all, and
    :func:`spacr.zstack.volume_stats` already uses exactly these names. So a
    3-D frame gets ``volume_voxels`` (always) and ``volume_um3`` (only when the
    physical voxel size is known) alongside the spaced ``area``.

    :param mask: the 3-D label volume.
    :param labels: label ids, in the frame's row order.
    :param stamp: the stamp from :func:`resolve_measurement_spacing`.
    :returns: dict of column name -> list of values, aligned with ``labels``.
    """
    counts = np.bincount(np.asarray(mask).ravel())
    voxels = np.array(
        [float(counts[int(v)]) if int(v) < counts.size else 0.0 for v in labels])
    out = {'volume_voxels': voxels}
    if stamp.get('measurement_units') == UNITS_UM:
        dz = float(stamp['voxel_size_z_um'])
        dxy = float(stamp['voxel_size_xy_um'])
        out['volume_um3'] = voxels * dz * dxy * dxy
    return out


#: How ``regionprops_table`` names the axes of a centroid in 3-D, and what each
#: one actually is. In 2-D ``centroid_weighted-0`` is the row (y); in 3-D the
#: same name is the plane (z) and every downstream consumer reading it as y is
#: silently wrong. Renaming only the 3-D columns leaves the 2-D names untouched
#: and makes the 3-D ones self-describing.
_CENTROID_AXES_3D = {'-0': '_z', '-1': '_y', '-2': '_x'}


def _rename_3d_centroids(df):
    """Rename ``centroid*-0/-1/-2`` to ``*_z/_y/_x`` on a 3-D intensity frame."""
    mapping = {}
    for col in df.columns:
        for suffix, axis in _CENTROID_AXES_3D.items():
            if col.startswith('centroid') and col.endswith(suffix):
                mapping[col] = col[:-len(suffix)] + axis
    return df.rename(columns=mapping) if mapping else df


def get_components(cell_mask, nucleus_mask, pathogen_mask):
    """Map each cell to its enclosed nucleus/pathogen labels via mask lookup.

    :param cell_mask: Label mask of cells.
    :param nucleus_mask: Label mask of nuclei.
    :param pathogen_mask: Label mask of pathogens.
    :returns: Tuple ``(nucleus_df, pathogen_df)`` where each DataFrame has one
        row per (cell, child) pair with columns ``cell_id`` and either
        ``nucleus`` or ``pathogen``.
    """
    cell_to_nucleus = defaultdict(list)
    cell_to_pathogen = defaultdict(list)
    cell_labels = np.unique(cell_mask)
    for cell_id in cell_labels:
        if cell_id == 0:
            continue
        nucleus_ids = np.unique(nucleus_mask[cell_mask == cell_id])
        pathogen_ids = np.unique(pathogen_mask[cell_mask == cell_id])
        cell_to_nucleus[cell_id] = nucleus_ids[nucleus_ids != 0].tolist()
        cell_to_pathogen[cell_id] = pathogen_ids[pathogen_ids != 0].tolist()
    nucleus_df = pd.DataFrame(list(cell_to_nucleus.items()), columns=['cell_id', 'nucleus'])
    pathogen_df = pd.DataFrame(list(cell_to_pathogen.items()), columns=['cell_id', 'pathogen'])
    nucleus_df = nucleus_df.explode('nucleus').dropna(
        subset=['nucleus']).reset_index(drop=True)
    pathogen_df = pathogen_df.explode('pathogen').dropna(
        subset=['pathogen']).reset_index(drop=True)
    return nucleus_df, pathogen_df

def _calculate_zernike(mask, df, degree=8):
    """Append per-region Zernike-moment columns to ``df``.

    :param mask: Label mask defining the regions.
    :param df: DataFrame to extend, in the same row order as ``regionprops(mask)``.
    :param degree: Zernike-moment degree. Default ``8``. The number of
        coefficients is set by the degree: 9 for 4, 25 for 8, 49 for 12.
    :returns: ``df`` with ``zernike_i`` columns appended, or unchanged when the
        mask has no regions or the mask is 3-D.
    :raises ImportError: When a non-empty 2-D mask needs the optional Mahotas
        implementation but ``spacr[zernike]`` is not installed.
    :raises ValueError: When the Zernike vectors have inconsistent lengths.

    .. note::

       Zernike moments are defined on a disk, so mahotas' ``zernike_moments``
       accepts 2-D images only -- a 3-D region raises
       ``ValueError: too many values to unpack``, which used to take down the
       whole morphology pass. A 3-D mask therefore gets no ``zernike_*``
       columns at all: absent, rather than a 2-D descriptor of one arbitrary
       plane presented as a description of the object.
    """
    if _ndim_of(mask) != 2:
        return df

    regions = list(regionprops(mask))
    if not regions:
        return df
    zernike_moments = _load_zernike_moments()
    zernike_features = []
    for region in regions:
        coords = np.argwhere(region.image)
        if coords.size == 0:
            radius = 1.0
        else:
            centre = coords.mean(axis=0)
            radius = float(np.sqrt(((coords - centre) ** 2).sum(axis=1)).max())
        radius = max(radius, 1.0)
        zernike_moment = zernike_moments(region.image, radius, degree=degree)
        zernike_features.append(zernike_moment.tolist())

    feature_length = len(zernike_features[0])
    for feature in zernike_features:
        if len(feature) != feature_length:
            raise ValueError("All Zernike moments must be of the same length")

    zernike_df = pd.DataFrame(zernike_features, columns=[f'zernike_{i}' for i in range(feature_length)])
    return pd.concat([df.reset_index(drop=True), zernike_df], axis=1)


#: Whether Mahotas answered, decided once per process rather than per object.
#:
#: `_morphological_measurements` runs once per FIELD, in each of up to `n_jobs`
#: worker processes, and probed the import every time -- so a machine without
#: Mahotas got the same four-line install notice fifty-two times, burying the
#: one message in that run that mattered (a field that actually failed).
_ZERNIKE_AVAILABLE = None


def _zernike_is_available() -> bool:
    """Whether Zernike moments can be computed here. Said once.

    THE ANSWER CANNOT CHANGE inside a run: a package does not become
    installable between two fields. So it is probed on the first field and
    remembered, and the notice is printed with it.

    Still once PER PROCESS rather than once per run, because a pool worker is a
    fresh interpreter with its own module state. That turns fifty-two notices
    into at most `n_jobs`, and the parent-side decision that would make it
    exactly one belongs with the settings resolution rather than here.
    """
    global _ZERNIKE_AVAILABLE
    if _ZERNIKE_AVAILABLE is not None:
        return _ZERNIKE_AVAILABLE
    try:
        _load_zernike_moments()
    except ImportError as exc:
        _ZERNIKE_AVAILABLE = False
        print(f"[measure] {exc} Zernike columns will be skipped.")
    else:
        _ZERNIKE_AVAILABLE = True
    return _ZERNIKE_AVAILABLE


def _load_zernike_moments():
    """Load Mahotas only when its optional descriptor is computed."""
    try:
        from mahotas.features import zernike_moments
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Zernike morphology requires the optional Mahotas package. "
            "Install it with `pip install \"spacr[zernike]\"`, or run "
            "morphological measurements with zernike=False. "
            "NOTE: Mahotas publishes no wheel for Python 3.13 or newer, so on "
            "those interpreters that install builds from source and needs a "
            "C++ toolchain -- see the note in setup.py. Every other "
            "morphological measurement is unaffected."
        ) from exc
    return zernike_moments

def _analyze_cytoskeleton(array, mask, channel):
    """Extract per-object skeleton length and branch counts from a cytoskeleton channel.

    :param array: Multi-channel intensity image ``(H, W, C)``.
    :param mask: Label mask; each non-zero label defines one object.
    :param channel: Channel index in ``array`` holding the cytoskeleton signal.
    :returns: DataFrame with ``object_label``, ``skeleton_length`` and
        ``skeleton_branch_points`` columns.
    """

    image = array[..., channel]

    properties_list = []

    for label in np.unique(mask):
        if label == 0:
            continue

        object_region = mask == label
        region_intensity = np.where(object_region, image, 0)

        if np.any(region_intensity):
            valid_pixels = region_intensity[region_intensity > 0]
            if len(valid_pixels) > 1:
                offset = np.percentile(valid_pixels, 90) - np.percentile(valid_pixels, 50)
                block_size = 35
                local_thresh = filters.threshold_local(region_intensity, block_size=block_size, offset=offset)
                cytoskeleton = region_intensity > local_thresh

                skeleton = morphology.skeletonize(img_as_bool(cytoskeleton))

                skeleton_props = measure.regionprops(measure.label(skeleton), intensity_image=image)
                skeleton_length = sum(prop.area for prop in skeleton_props)
                skel = skeleton.astype(np.uint8)
                neighbour_count = convolve(
                    skel, np.ones((3, 3), dtype=np.uint8),
                    mode='constant', cval=0) - skel
                n_branch_points = int(np.sum((skel == 1) & (neighbour_count >= 3)))

                properties = {
                    "object_label": label,
                    "skeleton_length": skeleton_length,
                    "skeleton_branch_points": n_branch_points
                }
                properties_list.append(properties)
            else:
                properties_list.append({
                    "object_label": label,
                    "skeleton_length": 0,
                    "skeleton_branch_points": 0
                })

    return pd.DataFrame(properties_list)

def _safe_morphology_table(mask, properties, spacing=None):
    """Return morphology properties without asking Qhull to hull flat volumes.

    A valid 3-D label may occupy one z plane (or form a line).  scikit-image
    delegates ``convex_area`` and ``solidity`` to Qhull, which warns for those
    lower-dimensional objects and then reports an empty hull / infinite
    solidity.  Their 3-D convex volume is undefined, so expose it as NaN while
    leaving full-dimensional objects and the entire 2-D path unchanged.
    """
    guarded = {
        'convex_area', 'area_convex', 'solidity', 'feret_diameter_max',
    }
    requested = list(properties)
    if _ndim_of(mask) != 3 or not guarded.intersection(requested):
        return pd.DataFrame(
            regionprops_table(mask, properties=requested, spacing=spacing))

    safe_properties = [prop for prop in requested if prop not in guarded]
    frame = pd.DataFrame(
        regionprops_table(mask, properties=safe_properties, spacing=spacing))
    regions = regionprops(mask, spacing=spacing)
    full_dimensional = [
        np.linalg.matrix_rank(
            region.coords - region.coords.mean(axis=0)) == 3
        for region in regions
    ]
    for prop in requested:
        if prop not in guarded:
            continue
        region_property = 'area_convex' if prop == 'convex_area' else prop
        frame[prop] = [
            float(getattr(region, region_property)) if full_rank else np.nan
            for region, full_rank in zip(regions, full_dimensional)
        ]

    return frame[[prop for prop in requested if prop in frame.columns]]


def _join_child_to_parent_cell(child_props, cell_to_child, child_name, remedy):
    """Attach each child object's parent ``cell_id`` to its morphology row.

    ``one_to_one``, and deliberately so. A child object belongs to exactly one
    cell in this data model, everywhere downstream:
    :meth:`spacr.schema.ObjectTableSchema.row_key_columns` keys the ``nucleus``
    and ``pathogen`` tables on one row per ``object_label`` per field, the
    tables carry a single scalar ``cell_id``, and
    :func:`spacr.utils._merge_and_save_to_database` joins morphology to
    intensity on ``object_label`` with ``validate='one_to_one'``. A frame with
    the same label twice is therefore not a shape measurements.db can hold, so
    the only question is *where* it stops.

    It has to stop here. ``_measure_crop_core`` writes the object tables one
    call at a time -- cell, then nucleus, then pathogen -- so a fan-out that
    survives this merge is not caught until the write for its own table, by
    which point the earlier tables for this field are already committed. That
    leaves the field half in the database: a cell row with no matching pathogen
    row, which reads downstream as an uninfected cell rather than as a failure.
    Raising before any write keeps a field all-in or all-out.

    ``get_components`` fans out when a child label overlaps two cell labels.
    On the pipeline path that is normally already impossible:
    ``_measure_crop_core`` runs :func:`spacr.utils._merge_overlapping_objects`
    on (nucleus, cell) unconditionally, and on (pathogen, cell) when
    ``merge_edge_pathogen_cells`` is set, and that resolves every straddling
    child to a single cell -- either by trimming the child back to the cell it
    overlaps most, or by merging the two cells into one. Which of those two
    repairs is available differs per object type, so the caller supplies the
    ``remedy`` sentence rather than this function guessing.

    :param child_props: ``regionprops_table`` output for the child mask; one
        row per label.
    :param cell_to_child: ``get_components``' exploded ``(cell_id, child)``
        pairs.
    :param child_name: ``'nucleus'`` or ``'pathogen'`` -- the column
        ``get_components`` keyed the child by.
    :param remedy: what the reader should change, appended to the message.
    :returns: ``child_props`` with ``cell_id`` (and the child key column)
        joined on.
    :raises pandas.errors.MergeError: either side repeats a label.
    """
    try:
        return pd.merge(
            child_props,
            cell_to_child,
            left_on='label',
            right_on=child_name,
            how='left',
            validate='one_to_one',
        )
    except pd.errors.MergeError as exc:
        shared = cell_to_child[cell_to_child[child_name].duplicated(keep=False)]
        if shared.empty:
            raise
        examples = [
            (int(lab), sorted(int(c) for c in grp['cell_id']))
            for lab, grp in shared.groupby(child_name)
        ][:5]
        raise pd.errors.MergeError(
            f"{len(shared[child_name].unique())} {child_name} label(s) overlap "
            f"more than one cell, so this field has no single parent cell for "
            f"them (e.g. {child_name}/cell_ids {examples}). The {child_name} "
            f"table holds one row per object with one cell_id, so measuring "
            f"this field would either double-count those objects or write only "
            f"part of the field to measurements.db. Nothing was written for "
            f"this field.\n"
            f"Fix the masks rather than the join: {remedy} (pandas: {exc})"
        ) from exc



#: Sentinel written to ``nearest_neighbor_distance`` /
#: ``second_neighbor_distance`` when the field contains no such neighbour: a
#: one-object field has neither, a two-object field has no second. Both are
#: ordinary in a killing condition, and NaN is not an option (see above). -1.0
#: is roughly an order of magnitude below any real centroid distance, so it is
#: separable -- but it is a sentinel, not a distance, and must not be averaged.
_SPATIAL_NO_NEIGHBOUR = -1.0

#: ``expand_labels`` grew its ``spacing`` argument after scikit-image 0.22, and
#: setup.py's floor is ``>=0.22.0``. Probed once rather than assumed: without it
#: a 3-D run would silently measure an unscaled radius, which was measured wrong
#: by 2.000x on a (2.0, 0.2, 0.2) voxel.
try:
    _EXPAND_LABELS_TAKES_SPACING = (
        'spacing' in inspect.signature(expand_labels).parameters)
except (TypeError, ValueError):  # pragma: no cover - C-implemented signature
    _EXPAND_LABELS_TAKES_SPACING = False


def spatial_column_names(radius):
    """Return the five spatial column names for ``radius``, in emitted order.

    The radius is baked into ``neighbors_within_<r>`` -- the same precedent as
    ``homogeneity_distance_<d>`` and ``percentile_<p>``. Two plates measured at
    different radii therefore produce different columns and will not concat.

    :param radius: neighbourhood radius; truncated with ``int()`` for the
        ``neighbors_within_<r>`` name. The other four names do not depend on it.
    :returns: list of five column names.
    """
    return [
        f'neighbors_within_{int(radius)}',
        'nearest_neighbor_distance',
        'second_neighbor_distance',
        'percent_touching',
        'touching_neighbors',
    ]


def _empty_spatial_frame(radius):
    """A correctly-typed zero-row spatial frame (empty mask, no objects)."""
    count_col, near_col, second_col, pct_col, touch_col = spatial_column_names(radius)
    return pd.DataFrame({
        'label': pd.Series(dtype='int64'),
        count_col: pd.Series(dtype='int64'),
        near_col: pd.Series(dtype='float64'),
        second_col: pd.Series(dtype='float64'),
        pct_col: pd.Series(dtype='float64'),
        touch_col: pd.Series(dtype='int64'),
    })


def _spatial_adjacency(mask, spacing=None, expand=1):
    """Return ``(percent_touching, touching_neighbors)`` maps keyed on label.

    Adjacency is taken on a mask grown by ``expand`` with
    :func:`skimage.segmentation.expand_labels`, and -- critically -- the
    boundary is the boundary **of the grown mask**, not of the original.

    That is not a detail. Comparing the grown mask against the *original*
    object's boundary reproduces the un-grown failure exactly: on a confluent
    field with true one-pixel gaps it reads ~0.5% touching with 69 of 100
    objects at zero, as if ``expand_labels`` had never been called. Segmentation
    routinely leaves a one-pixel background seam between objects a human would
    call touching, which is the whole reason for growing.

    ``scipy.ndimage.binary_dilation`` is not usable here at all: it returns a
    boolean array, so every label identity -- the thing being measured -- is
    gone before the comparison.
    """
    if _EXPAND_LABELS_TAKES_SPACING:
        grown = expand_labels(mask, distance=expand, spacing=spacing)
    else:
        if spacing is None:
            grown = expand_labels(mask, distance=expand)
        else:
            distances, nearest = distance_transform_edt(
                np.asarray(mask) == 0, sampling=spacing,
                return_distances=True, return_indices=True)
            grown = np.zeros_like(mask)
            within = distances <= float(expand)
            nearest_labels = np.asarray(mask)[tuple(nearest)]
            grown[within] = nearest_labels[within]

    ndim = grown.ndim
    inner = find_boundaries(grown, mode='inner')
    touching = np.zeros(grown.shape, dtype=bool)
    pairs = set()

    for axis in range(ndim):
        for shift in (1, -1):
            rolled = np.roll(grown, shift, axis=axis)
            edge = [slice(None)] * ndim
            edge[axis] = 0 if shift == 1 else -1
            rolled = rolled.copy()
            rolled[tuple(edge)] = 0

            different = (grown > 0) & (rolled > 0) & (rolled != grown)
            if not different.any():
                continue
            touching |= different
            here = grown[different]
            there = rolled[different]
            for src, dst in np.unique(np.stack([here, there], axis=1), axis=0):
                pairs.add((int(src), int(dst)))

    boundary_labels = grown[inner]
    touching_labels = grown[inner & touching]
    if boundary_labels.size == 0:
        return {}, {}

    n_boundary = np.bincount(boundary_labels)
    n_touching = np.bincount(touching_labels, minlength=n_boundary.size)

    percent = {}
    for lab in range(1, n_boundary.size):
        if n_boundary[lab] > 0:
            percent[lab] = float(100.0 * n_touching[lab] / n_boundary[lab])

    neighbours = defaultdict(int)
    for src, _dst in pairs:
        neighbours[src] += 1

    return percent, dict(neighbours)


def _spatial_measurements(mask, spacing=None, radius=50, expand=1):
    """Per-object spatial context, one row per label, NEVER containing NaN.

    :param mask: label mask, 2-D ``(Y, X)`` or 3-D ``(Z, Y, X)``.
    :param spacing: voxel spacing from :func:`resolve_measurement_spacing`;
        ``None`` in 2-D, which leaves centroids in pixels and the 2-D path
        numerically unchanged.
    :param radius: neighbourhood radius for ``neighbors_within_<r>``, in the
        units of the row's ``measurement_units`` stamp -- pixels in a 2-D run,
        micrometres in a 3-D run with a voxel size, xy-pixels when only an
        anisotropy was given. The KDTree is built on spacing-scaled centroids,
        so the radius has to be quoted in the same units.
    :param expand: growth in pixels before adjacency is taken; see
        :func:`_spatial_adjacency`.
    :returns: DataFrame keyed on ``label`` carrying
        :func:`spatial_column_names`.

    Cost is O(field), not O(objects x field): one ``regionprops_table`` for the
    centroids (~27 ms), one ``cKDTree`` answering every object's count and both
    distances in ~0.4 ms for 400 objects, and one ``expand_labels`` +
    ``find_boundaries`` pass (~61-73 ms in 2-D, ~551 ms in 3-D -- the 3-D figure
    dominates this block and is the reason the feature is opt-in).

    .. note::

       Centroids are recomputed here rather than added to
       :data:`MORPHOLOGICAL_PROPS`. A stored ``centroid`` would be free at
       measurement time and would then auto-enrol the object's *absolute
       position in the field* as a model feature for every existing run, which
       is leakage. Do not "optimise" this away.

    .. note::

       ``percent_touching`` has zero variance in a confluent monolayer
       (measured: 100.0% for every one of 400 objects), so
       ``utils.remove_low_variance_columns`` deletes it from model matrices for
       exactly the plates where it is most trivially true. That is correct
       behaviour, and it will be reported as a missing column.
    """
    count_col, near_col, second_col, pct_col, touch_col = spatial_column_names(radius)

    props = pd.DataFrame(regionprops_table(mask, properties=('label', 'centroid')))
    if len(props) == 0:
        return _empty_spatial_frame(radius)

    labels = props['label'].to_numpy()
    axis_cols = [c for c in props.columns if c.startswith('centroid')]
    axis_cols.sort(key=lambda c: int(c.rsplit('-', 1)[-1]) if '-' in c else 0)
    coords = props[axis_cols].to_numpy(dtype=np.float64)

    if spacing is not None:
        scale = np.asarray(spacing, dtype=np.float64).reshape(1, -1)
        if scale.shape[1] == coords.shape[1]:
            coords = coords * scale

    n = len(labels)
    tree = cKDTree(coords)
    counts = np.asarray(
        tree.query_ball_point(coords, r=float(radius), return_length=True)
    ).astype(np.int64) - 1
    counts = np.clip(counts, 0, None)

    k = min(3, n)
    distances = np.atleast_2d(tree.query(coords, k=k)[0])
    if k == 1:
        distances = distances.reshape(n, 1)

    if k >= 2:
        nearest = distances[:, 1].astype(np.float64)
    else:
        nearest = np.full(n, _SPATIAL_NO_NEIGHBOUR, dtype=np.float64)
    if k >= 3:
        second = distances[:, 2].astype(np.float64)
    else:
        second = np.full(n, _SPATIAL_NO_NEIGHBOUR, dtype=np.float64)

    percent_map, neighbour_map = _spatial_adjacency(mask, spacing=spacing, expand=expand)

    frame = pd.DataFrame({
        'label': labels.astype(np.int64),
        count_col: counts,
        near_col: nearest,
        second_col: second,
        pct_col: np.array([percent_map.get(int(lab), 0.0) for lab in labels], dtype=np.float64),
        touch_col: np.array([neighbour_map.get(int(lab), 0) for lab in labels], dtype=np.int64),
    })
    frame[[near_col, second_col, pct_col]] = frame[
        [near_col, second_col, pct_col]].fillna(_SPATIAL_NO_NEIGHBOUR)
    frame[[count_col, touch_col]] = frame[[count_col, touch_col]].fillna(0).astype(np.int64)
    return frame


def _spatial_organelle_eligible(settings):
    """Return ``True`` for every organelle settings mapping.

    Kept as a compatibility predicate for callers that imported the former
    private gate.  Organelle Type is advisory: it records measurement caveats
    but never switches off a requested family or removes its output columns.
    """
    del settings
    return True


def _morphology_of_organelle_type(settings):
    """The morphology an `organelle_type` implies, or None if it has none.

    None for 'custom' (which recommends nothing by design), for a missing
    key, and for an unrecognised one -- in every case the caller falls back
    to `organelle_morphology`, which is what a pre-72 settings file carries.
    """
    name = settings.get('organelle_type')
    if not name:
        return None
    try:
        from .organelle_types import resolve_type
        preset = resolve_type(name)
    except (ImportError, ValueError):
        return None
    return preset.morphology_for(settings.get('organelle_diameter'))


def _morphological_measurements(
        cell_mask, nucleus_mask, pathogen_mask, organelle_mask,
        cytoplasm_mask, settings, zernike=None, degree=8,
        extra_organelle_masks=None, channel_arrays=None):
    """Return morphology + Zernike DataFrames for cells, nuclei, pathogens, organelles, cytoplasm.

    :param cell_mask: Label mask of cells.
    :param nucleus_mask: Label mask of nuclei.
    :param pathogen_mask: Label mask of pathogens.
    :param organelle_mask: Label mask of organelles.
    :param cytoplasm_mask: Label mask of cytoplasm.
    :param settings: Settings dict; ``<object>_mask_dim`` keys drive whether
        each object type is analysed, ``cytoplasm`` toggles cytoplasm output,
        ``spatial_measurements`` (default ``False``) adds the spatial-context
        block and ``spatial_neighbor_radius`` (default 50) sizes it.
    :param zernike: ``True`` requires and computes Zernike moments; ``False``
        disables them. ``None`` computes them when Mahotas is installed and
        otherwise skips them with an actionable console message.
    :param degree: Zernike moment degree.
    :returns: Tuple ``(cell_df, nucleus_df, pathogen_df, organelle_df, cytoplasm_df)``.

    .. note::

       On a 3-D ``(Z, Y, X)`` mask the ``eccentricity`` and ``perimeter``
       columns are absent (skimage implements neither for 3-D), ``zernike_*``
       is absent, ``area`` and the ``*_area``/length columns are spaced by the
       voxel size so they are volumes and lengths rather than voxel counts, and
       explicit ``volume_voxels`` / ``volume_um3`` columns are added. See
       :func:`resolve_measurement_spacing`.

    .. note::

       With ``spatial_measurements=True`` the cell, nucleus, pathogen and
       organelle frames gain :func:`spatial_column_names`. Organelle Type never
       suppresses requested measurements; doubtful interpretations are
       recorded as caveats instead. Cytoplasm never gains these columns -- its
       mask carries the cell's own label, so it is one object per cell by
       construction. Default ``False``: an unchanged run does no extra work.
    """
    if zernike is None:
        zernike = _zernike_is_available()

    ndim = _ndim_of(cell_mask)
    spacing, stamp = resolve_measurement_spacing(settings, ndim)
    morphological_props = list(MORPHOLOGICAL_PROPS)
    if ndim == 3:
        morphological_props = [p for p in morphological_props
                               if p not in PROPS_2D_ONLY]

    def _props(mask):
        """regionprops_table + (3-D only) the explicitly-named volume columns."""
        frame = _safe_morphology_table(
            mask, properties=morphological_props, spacing=spacing)
        if ndim == 3 and len(frame) > 0:
            for name, values in _voxel_volume_columns(
                    mask, frame['label'].tolist(), stamp).items():
                frame[name] = values
        return frame

    spatial_on = bool(settings.get('spatial_measurements', False))
    try:
        spatial_radius = int(settings.get('spatial_neighbor_radius', 50))
    except (TypeError, ValueError):
        spatial_radius = 50

    distances_on = bool(settings.get('object_distances', False))

    bystanders_on = bool(settings.get('bystander_measurements', False))
    try:
        bystander_reach = float(settings.get('bystander_reach_in_diameters', 1.0))
    except (TypeError, ValueError):
        bystander_reach = 0.0

    def _all_masks():
        """Object type -> label image, for the masks this run actually has."""
        found = {}
        for name, mask in (('cell', cell_mask), ('nucleus', nucleus_mask),
                           ('pathogen', pathogen_mask)):
            if mask is not None and getattr(mask, 'size', 0):
                found[name] = mask
        return found

    def _with_distances(frame, mask, name):
        """Merge the object-distance block onto a props frame.

        Props on the LEFT for the reason `_with_spatial` gives: 'label' has
        to keep column position 0.
        """
        if not distances_on or len(frame) == 0:
            return frame
        masks = _all_masks()
        if name not in masks:
            masks = dict(masks, **{name: mask})
        try:
            from .object_distances import object_distances

            block = object_distances(
                masks, images=channel_arrays if settings.get(
                    'object_distance_intensity', True) else None,
                primary=name, channels=tuple(settings.get('channels') or ()),
                spacing=spacing,
                maxima=bool(settings.get('object_distance_maxima', True)))
        except Exception as error:                           # noqa: BLE001
            print(f"[measure] object distances for {name} were not "
                  f"measured: {type(error).__name__}: {error}")
            return frame
        if len(block.columns) <= 1:
            return frame
        return frame.merge(block, on='label', how='left',
                           validate='one_to_one')

    def _with_spatial(frame, mask):
        """Merge the spatial block onto a props frame. Props on the LEFT."""
        if not spatial_on or len(frame) == 0:
            return frame
        spatial = _spatial_measurements(
            mask, spacing=spacing, radius=spatial_radius)
        merged = frame.merge(spatial, on='label', how='left',
                             validate='one_to_one')
        count_col, near_col, second_col, pct_col, touch_col = \
            spatial_column_names(spatial_radius)
        merged[[near_col, second_col, pct_col]] = merged[
            [near_col, second_col, pct_col]].fillna(_SPATIAL_NO_NEIGHBOUR)
        merged[[count_col, touch_col]] = merged[
            [count_col, touch_col]].fillna(0).astype(np.int64)
        return merged


    def _with_bystanders(frame, mask, pathogen_links):
        """Merge the bystander block onto the CELL props frame.

        A cell is infected if it holds a pathogen, a bystander if it holds
        none but sits within the reach of one that does, and distal
        otherwise. Without the split the last two are the same row, and the
        uninfected control is a mixture whose variance hides the effect
        every infection comparison is looking for.

        THE REACH IS DERIVED FROM THIS FIELD'S OWN CELLS, as a multiple of
        their median diameter, so it means the same thing at 20x and 63x.

        Props on the LEFT, for the reason `_with_spatial` gives.
        """
        if not bystanders_on or len(frame) == 0:
            return frame
        try:
            from .bystanders import (_median_cell_diameter, classify,
                                     reach_from_diameter)

            diameter = _median_cell_diameter(mask, spacing=spacing)
            if diameter <= 0:
                print("[measure] bystanders were not measured: no median "
                      "cell diameter could be taken from this field")
                return frame
            if pathogen_links is None or not len(pathogen_links):
                infected = []
            else:
                infected = (pd.to_numeric(pathogen_links['cell_id'],
                                          errors='coerce')
                            .dropna().astype(np.int64).unique().tolist())
            block = classify(mask, infected,
                             reach=reach_from_diameter(diameter,
                                                       bystander_reach),
                             spacing=spacing)
        except Exception as error:                           # noqa: BLE001
            print(f"[measure] bystanders were not measured: "
                  f"{type(error).__name__}: {error}")
            return frame
        if 'neighbourhood' not in block.columns:
            return frame
        where = block['neighbourhood']
        out = pd.DataFrame({
            'label': block['label'].to_numpy(),
            'is_bystander': (where == 'bystander').to_numpy().astype(np.int64),
            'is_distal': (where == 'distal').to_numpy().astype(np.int64),
        })
        distance = pd.to_numeric(block['distance_to_infected'],
                                 errors='coerce').to_numpy(dtype=float)
        distance = np.where(np.isfinite(distance), distance,
                            _SPATIAL_NO_NEIGHBOUR)
        out['distance_to_infected'] = distance
        return frame.merge(out, on='label', how='left', validate='one_to_one')

    prop_ls = []
    ls = []

    if settings['cell_mask_dim'] is not None:
        cell_to_nucleus, cell_to_pathogen = get_components(cell_mask, nucleus_mask, pathogen_mask)
        cell_props = _props(cell_mask)
        cell_props = _with_spatial(cell_props, cell_mask)
        cell_props = _with_distances(cell_props, cell_mask, 'cell')
        cell_props = _with_bystanders(cell_props, cell_mask, cell_to_pathogen)
        if zernike:
            cell_props = _calculate_zernike(
                cell_mask, cell_props, degree=degree)
        prop_ls.append(cell_props)
        ls.append('cell')
    else:
        prop_ls.append(pd.DataFrame())
        ls.append('cell')

    if settings['nucleus_mask_dim'] is not None:
        nucleus_props = _props(nucleus_mask)
        nucleus_props = _with_spatial(nucleus_props, nucleus_mask)
        nucleus_props = _with_distances(nucleus_props, nucleus_mask, 'nucleus')
        if zernike:
            nucleus_props = _calculate_zernike(
                nucleus_mask, nucleus_props, degree=degree)
        if settings['cell_mask_dim'] is not None:
            nucleus_props = _join_child_to_parent_cell(
                nucleus_props, cell_to_nucleus, 'nucleus',
                remedy=(
                    "measure_crop already runs _merge_overlapping_objects on "
                    "the nucleus and cell masks before measuring, so reaching "
                    "this means that repair could not resolve the overlap -- "
                    "most often a single nucleus label made of two "
                    "disconnected components. Re-segment the nuclei, or drop "
                    "the split label."))
        prop_ls.append(nucleus_props)
        ls.append('nucleus')
    else:
        prop_ls.append(pd.DataFrame())
        ls.append('nucleus')

    if settings['pathogen_mask_dim'] is not None:
        pathogen_props = _props(pathogen_mask)
        pathogen_props = _with_spatial(pathogen_props, pathogen_mask)
        pathogen_props = _with_distances(pathogen_props, pathogen_mask, 'pathogen')
        if zernike:
            pathogen_props = _calculate_zernike(
                pathogen_mask, pathogen_props, degree=degree)
        if settings['cell_mask_dim'] is not None:
            pathogen_props = _join_child_to_parent_cell(
                pathogen_props, cell_to_pathogen, 'pathogen',
                remedy=(
                    "set merge_edge_pathogen_cells=True so spaCR resolves a "
                    "vacuole straddling two host cells to one cell before "
                    "measuring, or re-segment so the pathogen and cell masks "
                    "nest."))
        prop_ls.append(pathogen_props)
        ls.append('pathogen')
    else:
        prop_ls.append(pd.DataFrame())
        ls.append('pathogen')

    organelle_masks = {'organelle': organelle_mask}
    organelle_masks.update(dict(extra_organelle_masks or {}))
    for organelle_role, current_organelle_mask in organelle_masks.items():
        if settings.get(f'{organelle_role}_mask_dim') is not None:
            organelle_props = _props(current_organelle_mask)
            if spatial_on:
                organelle_props = _with_spatial(
                    organelle_props, current_organelle_mask)
            if len(organelle_props) > 0 and zernike:
                organelle_props = _calculate_zernike(
                    current_organelle_mask, organelle_props, degree=degree)
            if len(organelle_props) > 0 and settings['cell_mask_dim'] is not None:
                organelle_to_cell = _map_child_to_parent(
                    current_organelle_mask, cell_mask,
                    child_name=organelle_role, parent_name='cell')
                organelle_props = pd.merge(
                    organelle_props,
                    organelle_to_cell,
                    left_on='label',
                    right_on=organelle_role,
                    how='left',
                    validate='one_to_one',
                )
            prop_ls.append(organelle_props)
        else:
            prop_ls.append(pd.DataFrame())
        ls.append(organelle_role)

    if settings['cytoplasm']:
        cytoplasm_props = _props(cytoplasm_mask)
        prop_ls.append(cytoplasm_props)
        ls.append('cytoplasm')
    else:
        prop_ls.append(pd.DataFrame())
        ls.append('cytoplasm')

    df_ls = []
    for i, df in enumerate(prop_ls):
        df.columns = [f'{ls[i]}_{col}' for col in df.columns]
        df = df.rename(columns={col: 'label' for col in df.columns if 'label' in col})
        df_ls.append(df)
 
    return tuple(df_ls)

def _map_child_to_parent(child_mask, parent_mask, child_name='organelle', parent_name='cell'):
    """Map each child label to its maximum-overlap parent label."""
    child_labels = np.unique(child_mask)
    child_labels = child_labels[child_labels != 0]
    
    mapping = []
    for child_id in child_labels:
        region = child_mask == child_id
        parent_ids = parent_mask[region]
        parent_ids = parent_ids[parent_ids != 0]
        if len(parent_ids) > 0:
            parent_id = np.bincount(parent_ids).argmax()
        else:
            parent_id = 0
        mapping.append({child_name: child_id, parent_name: parent_id})
    
    return pd.DataFrame(mapping)


def _summarize_organelles_per_parent(organelle_mask, parent_mask, channel_arrays, parent_name='cell', spacing=None):
    """Return one row per parent object summarising its enclosed organelles.

    Per parent computes: organelle count, total/mean/std area, area fraction,
    mean/std eccentricity and solidity, and per-channel mean/std intensity.

    :param organelle_mask: Label mask of organelles.
    :param parent_mask: Label mask of parents (cells, nuclei, ...).
    :param channel_arrays: Intensity images with shape ``(H, W, C)`` in 2-D or
        ``(Z, Y, X, C)`` in 3-D.
    :param parent_name: Column name used for the parent identifier.
    :param spacing: Voxel spacing from :func:`resolve_measurement_spacing`;
        ``None`` (the 2-D case) leaves everything in pixels.
    :returns: DataFrame indexed by parent label.

    .. note::

       On a 3-D mask the ``organelle_mean_eccentricity`` /
       ``organelle_std_eccentricity`` columns are absent -- skimage does not
       define eccentricity for 3-D. ``organelle_fraction`` is a ratio of two
       equally-spaced quantities and is therefore unchanged in meaning.
    """
    ndim = _ndim_of(organelle_mask)
    parent_labels = np.unique(parent_mask)
    parent_labels = parent_labels[parent_labels != 0]

    morphological_props = ['label', 'area', 'eccentricity', 'solidity', 'major_axis_length', 'minor_axis_length']
    if ndim == 3:
        morphological_props = [p for p in morphological_props
                               if p not in PROPS_2D_ONLY]

    organelle_df = _safe_morphology_table(
        organelle_mask, properties=morphological_props, spacing=spacing)

    organelle_to_parent = _map_child_to_parent(organelle_mask, parent_mask, 
                                                child_name='organelle_label', 
                                                parent_name=parent_name)
    
    if len(organelle_df) > 0 and len(organelle_to_parent) > 0:
        organelle_df = pd.merge(
            organelle_df,
            organelle_to_parent,
            left_on='label',
            right_on='organelle_label',
            how='left',
            validate='one_to_one',
        )
    else:
        rows = []
        for pid in parent_labels:
            row = {'label': pid, 'organelle_count': 0, 'organelle_total_area': 0, 
                   'organelle_fraction': 0.0}
            rows.append(row)
        return pd.DataFrame(rows)

    for ch in range(channel_arrays.shape[-1]):
        channel = channel_arrays[..., ch]
        intensities = []
        for org_label in organelle_df['label']:
            region = organelle_mask == org_label
            if np.any(region):
                intensities.append(channel[region].mean())
            else:
                intensities.append(0.0)
        organelle_df[f'organelle_channel_{ch}_mean_intensity'] = intensities

    parent_props = pd.DataFrame(regionprops_table(parent_mask, properties=['label', 'area'], spacing=spacing))
    parent_area_map = dict(zip(parent_props['label'], parent_props['area']))

    summary_rows = []
    for pid in parent_labels:
        org_subset = organelle_df[organelle_df[parent_name] == pid]
        parent_area = parent_area_map.get(pid, 1)

        row = {'label': pid}
        row['organelle_count'] = len(org_subset)
        row['organelle_total_area'] = org_subset['area'].sum() if len(org_subset) > 0 else 0
        row['organelle_fraction'] = row['organelle_total_area'] / parent_area if parent_area > 0 else 0.0
        row['organelle_mean_area'] = org_subset['area'].mean() if len(org_subset) > 0 else 0.0
        row['organelle_std_area'] = org_subset['area'].std() if len(org_subset) > 1 else 0.0
        if 'eccentricity' in organelle_df.columns:
            row['organelle_mean_eccentricity'] = org_subset['eccentricity'].mean() if len(org_subset) > 0 else 0.0
            row['organelle_std_eccentricity'] = org_subset['eccentricity'].std() if len(org_subset) > 1 else 0.0
        row['organelle_mean_solidity'] = org_subset['solidity'].mean() if len(org_subset) > 0 else 0.0
        row['organelle_std_solidity'] = org_subset['solidity'].std() if len(org_subset) > 1 else 0.0
        row['organelle_mean_major_axis'] = org_subset['major_axis_length'].mean() if len(org_subset) > 0 else 0.0
        row['organelle_mean_minor_axis'] = org_subset['minor_axis_length'].mean() if len(org_subset) > 0 else 0.0

        for ch in range(channel_arrays.shape[-1]):
            col = f'organelle_channel_{ch}_mean_intensity'
            row[f'organelle_channel_{ch}_mean_intensity_per_{parent_name}'] = org_subset[col].mean() if len(org_subset) > 0 else 0.0
            row[f'organelle_channel_{ch}_std_intensity_per_{parent_name}'] = org_subset[col].std() if len(org_subset) > 1 else 0.0

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)

def _intensity_measurements(
        cell_mask, nucleus_mask, pathogen_mask, organelle_mask,
        cytoplasm_mask, channel_arrays, settings, sizes=None, periphery=True,
        outside=True, extra_organelle_masks=None):
    """Return per-channel intensity DataFrames for cells, nuclei, pathogens, organelles, cytoplasm.

    Computes extended regionprops plus optional homogeneity, periphery, outside,
    blur, colocalisation and radial distribution features per object type.

    :param cell_mask: Label mask of cells.
    :param nucleus_mask: Label mask of nuclei.
    :param pathogen_mask: Label mask of pathogens.
    :param organelle_mask: Label mask of organelles.
    :param cytoplasm_mask: Label mask of cytoplasm.
    :param channel_arrays: Intensity array of shape ``(H, W, C)`` in 2-D or
        ``(Z, Y, X, C)`` in 3-D.
    :param settings: Settings dict (``radial_dist``, ``calculate_correlation``,
        ``homogeneity``, ``homogeneity_distances``, ``manders_thresholds``,
        ``distance_gaussian_sigma``, and the ``<object>_mask_dim`` toggles).
    :param sizes: Legacy size bins. Defaults to ``[3, 6, 12, 24]``.
    :param periphery: When True, compute periphery-intensity stats for
        nucleus/pathogen/organelle.
    :param outside: When True, compute outside-of-object intensity stats.
    :returns: Tuple ``(cell_df, nucleus_df, pathogen_df, organelle_df, cytoplasm_df)``.

    .. note::

       On a 3-D mask: the GLCM ``homogeneity_distance_*`` block is absent
       (``skimage.feature.graycomatrix`` is 2-D only and there is no
       co-occurrence matrix of a volume that reduces to it); every distance
       transform is sampled with the voxel spacing; ``blur`` is measured plane
       by plane in the xy plane, which is where focus is defined; and
       ``centroid_weighted-0/-1/-2`` are renamed ``_z/_y/_x`` so that no 2-D
       column name silently changes axis.
    """
    if sizes is None:
        sizes = [3, 6, 12, 24]
    radial_dist = settings['radial_dist']
    calculate_correlation = settings['calculate_correlation']
    homogeneity = settings['homogeneity']
    distances = settings['homogeneity_distances']

    ndim = _ndim_of(cell_mask)
    spacing, _stamp = resolve_measurement_spacing(settings, ndim)
    if homogeneity and ndim == 3:
        print("3-D mask: skipping GLCM homogeneity — "
              "skimage.feature.graycomatrix is defined for 2-D images only, "
              "so no homogeneity_distance_* columns are written for this field.")
        homogeneity = False

    intensity_props = ["label", "centroid_weighted", "centroid_weighted_local", "max_intensity", "mean_intensity", "min_intensity"]
    col_lables = ['region_label', 'mean', 'percentile_5', 'percentile_10', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_85', 'percentile_95']
    organelle_masks = {'organelle': organelle_mask}
    organelle_masks.update(dict(extra_organelle_masks or {}))
    ls = ['cell', 'nucleus', 'pathogen', *organelle_masks, 'cytoplasm']
    labels = [cell_mask, nucleus_mask, pathogen_mask,
              *organelle_masks.values(), cytoplasm_mask]
    dfs = [[] for _ in ls]
    
    for i in range(0, channel_arrays.shape[-1]):
        channel = channel_arrays[..., i]
        channel_percentiles = _field_reference_percentiles(channel)
        for j, (label, df) in enumerate(zip(labels, dfs)):

            if np.max(label) == 0:
                empty_df = pd.DataFrame()
                df.append(empty_df)
                continue

            mask_intensity_df = _extended_regionprops_table(
                label, channel, intensity_props, spacing=spacing,
                field_percentiles=channel_percentiles)

            if homogeneity:
                homogeneity_df = _calculate_homogeneity(label, channel, distances)
                mask_intensity_df = pd.concat([mask_intensity_df.reset_index(drop=True), homogeneity_df], axis=1)

            if periphery:
                if ls[j] in ('nucleus', 'pathogen', *ORGANELLE_ROLES):
                    periphery_intensity_stats = _periphery_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(periphery_intensity_stats, columns=[f'periphery_{stat}' for stat in col_lables])], axis=1)

            if outside:
                if ls[j] in ('nucleus', 'pathogen', *ORGANELLE_ROLES):
                    outside_intensity_stats = _outside_intensity(label, channel, spacing=spacing)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(outside_intensity_stats, columns=[f'outside_{stat}' for stat in col_lables])], axis=1)

            label_shape = np.asarray(label).shape
            label_boxes = _label_bounding_boxes(label)
            label_field = _whole_field_window(label_shape)
            blur_col = []
            for region_label in mask_intensity_df['label']:
                box = _box_for(label_boxes, region_label)
                window = (label_field if box is None
                          else _grow_window(box, 1, label_shape))
                blur_col.append(_estimate_blur(
                    channel[window], mask=(label[window] == region_label)))
            mask_intensity_df['blur'] = blur_col

            mask_intensity_df.columns = [f'{ls[j]}_channel_{i}_{col}' if col != 'label' else col for col in mask_intensity_df.columns]
            df.append(mask_intensity_df)
            
    if isinstance(settings['distance_gaussian_sigma'], int):
        if settings['distance_gaussian_sigma'] != 0:
            if settings['cell_mask_dim'] is not None:
                if settings['nucleus_mask_dim'] is not None or settings['pathogen_mask_dim'] is not None:
                    intensity_distance_df = _measure_intensity_distance(cell_mask, nucleus_mask, pathogen_mask, channel_arrays, settings)
                    dfs[0].append(intensity_distance_df)
    
    if radial_dist:
        if np.max(nucleus_mask) != 0:
            nucleus_radial_distributions = _calculate_radial_distribution(cell_mask, nucleus_mask, channel_arrays, num_bins=6, spacing=spacing)
            nucleus_df = _create_dataframe(nucleus_radial_distributions, 'nucleus')
            dfs[1].append(nucleus_df)

        if np.max(pathogen_mask) != 0:
            pathogen_radial_distributions = _calculate_radial_distribution(cell_mask, pathogen_mask, channel_arrays, num_bins=6, spacing=spacing)
            pathogen_df = _create_dataframe(pathogen_radial_distributions, 'pathogen')
            dfs[2].append(pathogen_df)

        for offset, (role, current_mask) in enumerate(
                organelle_masks.items(), start=3):
            if np.max(current_mask) != 0:
                distributions = _calculate_radial_distribution(
                    cell_mask, current_mask, channel_arrays,
                    num_bins=6, spacing=spacing)
                dfs[offset].append(_create_dataframe(distributions, role))

    if settings.get('cell_mask_dim') is not None and np.max(cell_mask) != 0:
        child_masks = [(1, nucleus_mask), (2, pathogen_mask)] + [
            (index, mask) for index, mask in enumerate(
                organelle_masks.values(), start=3)]
        for idx, child_mask in child_masks:
            if np.max(child_mask) == 0:
                continue
            for existing in dfs[idx]:
                if 'cell_id' in existing.columns:
                    existing.drop(columns=['cell_id'], inplace=True)
            parent_link = _map_child_to_parent(child_mask, cell_mask,
                                               child_name='label',
                                               parent_name='cell_id')
            parent_link['cell_id'] = parent_link['cell_id'].astype(float).replace(0.0, np.nan)
            dfs[idx].append(parent_link.reset_index(drop=True))

    if settings.get('pathogen_mask_dim') is not None:
        from .host_pathogen import vacuole_links
        for idx, (role, child_mask) in enumerate(organelle_masks.items(), start=3):
            if np.max(child_mask) != 0:
                links = vacuole_links(child_mask, pathogen_mask).rename(
                    columns={'pathogen_overlap_fraction': f'{role}_pathogen_overlap_fraction'})
                dfs[idx].append(links)

    if calculate_correlation:
        if channel_arrays.shape[-1] >= 2:
            for i in range(channel_arrays.shape[-1]):
                for j in range(i+1, channel_arrays.shape[-1]):
                    chan_i = channel_arrays[..., i]
                    chan_j = channel_arrays[..., j]
                    for m, mask in enumerate(labels):
                        coloc_df = _calculate_correlation_object_level(chan_i, chan_j, mask, settings)
                        coloc_df.columns = [f'{ls[m]}_channel_{i}_channel_{j}_{col}' for col in coloc_df.columns]
                        dfs[m].append(coloc_df)
    
    return tuple(pd.concat(frames, axis=1) for frames in dfs)
    
def _create_dataframe(radial_distributions, object_type):
        """Convert a ``{(cell, obj, ch): bins}`` mapping into a per-object DataFrame."""
        df = pd.DataFrame()
        for key, value in radial_distributions.items():
            cell_label, object_label, channel_index = key
            for i in range(len(value)):
                col_name = f'{object_type}_rad_dist_channel_{channel_index}_bin_{i}'
                df.loc[object_label, col_name] = value[i]
            df.loc[object_label, 'cell_id'] = cell_label
        df = df.reset_index().rename(columns={'index': 'label'})
        return df

def _whole_field_window(shape):
    """The slice tuple covering every voxel of an array of ``shape``."""
    return tuple(slice(0, int(dim)) for dim in shape)


def _label_bounding_boxes(label_mask):
    """Return ``{label: slice tuple}``, one bounding box per non-zero label.

    Per-object work in this module is written as a whole-field comparison
    (``label_mask == region``) inside a loop over objects, so reaching an
    object a few tens of pixels across costs a pass over the entire field and
    the loop costs O(objects x field). Restricting each iteration to the
    object's own bounding box is exact rather than approximate: it selects the
    same pixels in the same C order, so every reduction over them -- a mean, a
    percentile, a pairwise correlation -- is unchanged to the last bit.

    The mapping is an optimisation hint and never a filter. A label with no
    box in it is simply measured over the whole field, which is what the
    callers do, so no object is ever dropped by cropping.

    :param label_mask: label mask, 2-D or 3-D.
    :returns: mapping from label to slice tuple. Empty when the labels are not
        whole numbers, or when the largest label exceeds the voxel count --
        :func:`scipy.ndimage.find_objects` enumerates every label up to the
        maximum, so a mask numbered that sparsely would cost more to box than
        the crops save.
    """
    arr = np.asarray(label_mask)
    if arr.size == 0:
        return {}
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.rint(arr)):
            return {}
        arr = arr.astype(np.int64)
    if int(arr.max()) > arr.size:
        return {}
    return {index + 1: box
            for index, box in enumerate(find_objects(arr))
            if box is not None}


def _box_for(boxes, label):
    """The bounding box recorded for ``label``, or ``None`` when there is none.

    An empty mapping is answered without touching ``label`` at all. That is not
    a shortcut: :func:`_label_bounding_boxes` returns nothing precisely when the
    labels are not whole numbers, and those are the labels that cannot be used
    as a key -- ``1.5`` would truncate onto object 1's box, and a NaN label,
    which a float mask can carry and which every caller currently reports as an
    empty object, would raise.
    """
    if not boxes:
        return None
    return boxes.get(int(label))


def _grow_window(window, pad, shape):
    """Grow a slice tuple by ``pad`` voxels per axis, clipped to ``shape``.

    :param window: slice tuple to grow.
    :param pad: one margin for every axis, or a per-axis sequence.
    :param shape: array shape the result is clipped to.
    """
    pads = pad if isinstance(pad, (tuple, list)) else (pad,) * len(shape)
    return tuple(slice(max(0, sl.start - int(margin)),
                       min(int(dim), sl.stop + int(margin)))
                 for sl, margin, dim in zip(window, pads, shape))


def _union_window(first, second):
    """The smallest slice tuple containing both windows."""
    return tuple(slice(min(a.start, b.start), max(a.stop, b.stop))
                 for a, b in zip(first, second))


def _ring_padding(distance, spacing, shape):
    """Voxels of margin an object needs for a ``distance``-wide outside ring.

    With a voxel spacing, a voxel ``n`` steps from the object along axis ``k``
    is at least ``n * spacing[k]`` away, so nothing inside the ring lies
    further than ``ring_width / spacing[k]`` steps out and a box grown by that
    much contains the whole ring. Without one the ring is ``distance``
    iterations of :func:`scipy.ndimage.binary_dilation`, whose reach is
    ``distance`` voxels along each axis.

    Two inputs bound nothing and get the whole field, so that the ring is
    measured exactly as it would be with no cropping at all: a spacing with a
    step that is zero or not finite, and a ``distance`` that is not positive --
    ``binary_dilation`` reads ``iterations < 1`` as "repeat until the result
    stops changing", which floods the array rather than growing a ring.
    """
    whole = tuple(int(dim) for dim in shape)
    if not float(distance) > 0:
        return whole
    if spacing is None:
        return (int(distance),) * len(shape)
    steps = [float(step) for step in spacing]
    if not all(np.isfinite(step) and step > 0 for step in steps):
        return whole
    ring_width = float(distance) * steps[-1]
    return tuple(int(ceil(ring_width / step)) for step in steps)


def _percentiles_of(values, cut_points):
    """Return the percentiles of ``values`` at every point in ``cut_points``.

    ``np.percentile`` accepts a sequence for ``q`` and answering several cut
    points in one call is several times cheaper than one call each, because the
    vector is partitioned once instead of once per point.

    It is not always the same arithmetic, though. On a float32 input numpy's
    sequence form computes the interpolation in float64 and its scalar form
    computes it in float32, and the two disagree in the last bits -- so an
    intensity column would quietly move the day it was batched. Batching is
    therefore used only where numpy reaches float64 either way (integers,
    booleans and float64 itself); a narrower float gets one call per point and
    the value a database already holds.

    :param values: 1-D array of pixel values.
    :param cut_points: percentile positions in [0, 100].
    :returns: array of percentiles, one per cut point, in the given order.
    """
    array = np.asarray(values)
    batched_is_exact = (np.issubdtype(array.dtype, np.integer)
                        or array.dtype == np.bool_
                        or array.dtype == np.float64)
    if batched_is_exact:
        return np.percentile(array, cut_points)
    return np.array([np.percentile(array, point) for point in cut_points])


def _field_reference_percentiles(image):
    """Return the field's ``(p90, p10)`` intensity references, NaN when empty.

    ``frac_high90`` / ``frac_low10`` are thresholded on the whole field, so the
    pair depends only on the channel and not on which mask is being measured.
    :func:`_intensity_measurements` measures every mask against every channel,
    so computing them inside :func:`_extended_regionprops_table` re-ravelled
    and re-sorted the same channel once per mask.
    """
    field = np.asarray(image, dtype=float).ravel()
    field = field[~np.isnan(field)]
    if not field.size:
        return np.nan, np.nan
    return float(np.percentile(field, 90)), float(np.percentile(field, 10))


def _extended_regionprops_table(labels, image, intensity_props, spacing=None,
                                field_percentiles=None):
    """Return a regionprops table extended with distributional intensity features (mean/std/skew/kurtosis/mode/CV/Gini/entropy/percentiles).

    :param labels: label mask, 2-D or 3-D.
    :param image: co-aligned intensity image.
    :param intensity_props: regionprops property names.
    :param spacing: voxel spacing from :func:`resolve_measurement_spacing`;
        ``None`` in 2-D, which skimage treats as "not supplied".
    :param field_percentiles: the ``(p90, p10)`` of ``image`` from
        :func:`_field_reference_percentiles`, for a caller that measures
        several masks against one channel and would otherwise recompute them
        per mask. Computed here when omitted, so the values are identical
        either way.
    """

    def _gini(array):
        """NaN-safe Gini coefficient of an intensity array."""
        array = np.abs(array[~np.isnan(array)])
        n = array.size
        array = np.sort(array)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)) if np.sum(array) else np.nan
    
    props = regionprops_table(labels, image, properties=intensity_props, spacing=spacing)
    df = pd.DataFrame(props)
    if _ndim_of(labels) == 3:
        df = _rename_3d_centroids(df)

    if field_percentiles is None:
        field_percentiles = _field_reference_percentiles(image)
    field_p90, field_p10 = field_percentiles

    regions = regionprops(labels, intensity_image=image, spacing=spacing)
    integrated_intensity = []
    std_intensity = []
    median_intensity = []
    skew_intensity = []
    kurtosis_intensity = []
    mode_intensity = []
    range_intensity = []
    iqr_intensity = []
    cv_intensity = []
    gini_intensity = []
    frac_high90 = []
    frac_low10 = []
    entropy_intensity = []

    def _masked_intensity(region):
        """Pixels inside a region across old and new scikit-image names."""
        try:
            intensity = region.image_intensity
        except AttributeError:
            intensity = region.intensity_image
        return intensity[region.image]

    for region in regions:
        intens = _masked_intensity(region)
        intens = intens[~np.isnan(intens)]
        if intens.size == 0:
            integrated_intensity.append(np.nan)
            std_intensity.append(np.nan)
            median_intensity.append(np.nan)
            skew_intensity.append(np.nan)
            kurtosis_intensity.append(np.nan)
            mode_intensity.append(np.nan)
            range_intensity.append(np.nan)
            iqr_intensity.append(np.nan)
            cv_intensity.append(np.nan)
            gini_intensity.append(np.nan)
            frac_high90.append(np.nan)
            frac_low10.append(np.nan)
            entropy_intensity.append(np.nan)
        else:
            has_variation = not np.all(intens == intens[0])
            integrated_intensity.append(np.sum(intens))
            std_intensity.append(np.std(intens))
            median_intensity.append(np.median(intens))
            skew_intensity.append(
                skew(intens) if intens.size > 2 and has_variation else np.nan)
            kurtosis_intensity.append(
                kurtosis(intens) if intens.size > 3 and has_variation else np.nan)
            mode_val = np.atleast_1d(np.asarray(mode(intens, nan_policy='omit').mode))
            mode_intensity.append(float(mode_val[0]) if mode_val.size else np.nan)
            range_intensity.append(np.ptp(intens))
            upper_quartile, lower_quartile = _percentiles_of(intens, [75, 25])
            iqr_intensity.append(upper_quartile - lower_quartile)
            cv_intensity.append(np.std(intens) / np.mean(intens) if np.mean(intens) != 0 else np.nan)
            gini_intensity.append(_gini(intens))
            frac_high90.append(np.mean(intens > field_p90) if np.isfinite(field_p90) else np.nan)
            frac_low10.append(np.mean(intens < field_p10) if np.isfinite(field_p10) else np.nan)
            entropy_intensity.append(shannon_entropy(intens) if intens.size > 1 else 0.0)

    df['integrated_intensity'] = integrated_intensity
    df['std_intensity'] = std_intensity
    df['median_intensity'] = median_intensity
    df['skew_intensity'] = skew_intensity
    df['kurtosis_intensity'] = kurtosis_intensity
    df['mode_intensity'] = mode_intensity
    df['range_intensity'] = range_intensity
    df['iqr_intensity'] = iqr_intensity
    df['cv_intensity'] = cv_intensity
    df['gini_intensity'] = gini_intensity
    df['frac_high90'] = frac_high90
    df['frac_low10'] = frac_low10
    df['entropy_intensity'] = entropy_intensity

    percentiles = [5, 10, 25, 75, 85, 95]
    per_region = [_percentiles_of(_masked_intensity(region), percentiles)
                  for region in regions]
    for position, p in enumerate(percentiles):
        df[f'percentile_{p}'] = [values[position] for values in per_region]
    return df

def _calculate_homogeneity(label, channel, distances=None):
        """Return per-region GLCM homogeneity across the requested co-occurrence distances.

        :raises ValueError: when ``label`` is not 2-D.
            ``skimage.feature.graycomatrix`` accepts a 2-D image only, and a
            grey-level co-occurrence matrix of a volume is a different
            construction (it needs 13 direction pairs rather than 4), not a
            generalisation of this one. ``_intensity_measurements`` skips the
            whole block for 3-D masks rather than calling this; the guard is
            here so a direct caller gets an explanation instead of skimage's
            "The parameter `image` must be a 2-dimensional array".
        """
        if _ndim_of(label) != 2:
            raise ValueError(
                "_calculate_homogeneity is 2-D only: skimage's graycomatrix "
                f"takes a 2-D image and this mask is {_ndim_of(label)}-D. A "
                "3-D run writes no homogeneity_distance_* columns.")
        if distances is None:
            distances = [2,4,8,16,32,64]
        homogeneity_values = []
        for region in regionprops(label):
            region_image = region.image * channel[region.slice]
            if not np.issubdtype(region_image.dtype, np.floating):
                region_image = region_image.astype(int)
            rescaled_image = rescale_intensity(
                region_image, out_range=(0, 255)).astype('uint8')
            homogeneity_per_distance = []
            for d in distances:
                glcm = graycomatrix(rescaled_image, [d], [0], symmetric=True, normed=True)
                if not np.any(glcm):
                    homogeneity_per_distance.append(np.nan)
                else:
                    homogeneity_per_distance.append(
                        graycoprops(glcm, 'homogeneity')[0, 0])
            homogeneity_values.append(homogeneity_per_distance)
        columns = [f'homogeneity_distance_{d}' for d in distances]
        homogeneity_df = pd.DataFrame(homogeneity_values, columns=columns)

        return homogeneity_df

def _periphery_intensity(label_mask, image):
    """Return per-region intensity stats along each object's outer boundary.

    :param label_mask: Label mask defining the regions.
    :param image: Intensity image co-aligned with ``label_mask``.
    :returns: List of ``(label, mean, p5, p10, p25, p50, p75, p85, p95)`` tuples.
    """
    periphery_intensity_stats = []
    boundary = find_boundaries(label_mask)
    boxes = _label_bounding_boxes(label_mask)
    whole = _whole_field_window(np.asarray(label_mask).shape)
    cut_points = [5, 10, 25, 50, 75, 85, 95]
    for region in np.unique(label_mask)[1:]:
        box = _box_for(boxes, region)
        window = whole if box is None else box
        region_boundary = boundary[window] & (label_mask[window] == region)
        intensities = image[window][region_boundary]
        if intensities.size == 0:
            periphery_intensity_stats.append((region, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            quantiles = _percentiles_of(intensities, cut_points)
            periphery_intensity_stats.append(
                (region, np.mean(intensities), *quantiles))
    return periphery_intensity_stats

def _outside_intensity(label_mask, image, distance=5, spacing=None):
    """Return per-region intensity stats within a ``distance``-pixel ring outside each object.

    :param label_mask: Label mask defining the regions.
    :param image: Intensity image co-aligned with ``label_mask``.
    :param distance: Ring width, in xy pixels.
    :param spacing: Voxel spacing from :func:`resolve_measurement_spacing`.
        ``None`` (2-D) keeps the historical ``binary_dilation`` ring exactly.
    :returns: List of ``(label, mean, p5, p10, p25, p50, p75, p85, p95)`` tuples.

    .. note::

       In 3-D the ring is built from a **sampled** distance transform, not from
       ``binary_dilation(iterations=distance)``. Iterated dilation counts
       voxels, so on a stack with dz = 5 dxy it grows the shell 5x further in z
       than in xy in physical terms -- a 25x thicker slab of neighbouring
       tissue on one axis than on the others -- and the "outside intensity" it
       reports is dominated by whatever sits above and below the object. The
       ring width is converted with the xy spacing so it still means
       ``distance`` xy pixels.
    """
    outside_intensity_stats = []
    if spacing is not None:
        ring_width = float(distance) * float(spacing[-1])
    shape = np.asarray(label_mask).shape
    boxes = _label_bounding_boxes(label_mask)
    whole = _whole_field_window(shape)
    pad = _ring_padding(distance, spacing, shape)
    cut_points = [5, 10, 25, 50, 75, 85, 95]
    for region in np.unique(label_mask)[1:]:
        box = _box_for(boxes, region)
        window = whole if box is None else _grow_window(box, pad, shape)
        region_mask = label_mask[window] == region
        if spacing is None:
            dilated_mask = binary_dilation(region_mask, iterations=distance)
        else:
            edt = distance_transform_edt(~region_mask, sampling=spacing)
            dilated_mask = edt <= ring_width
        outside_mask = dilated_mask & ~region_mask
        intensities = image[window][outside_mask]
        if intensities.size == 0:
            outside_intensity_stats.append((region, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            quantiles = _percentiles_of(intensities, cut_points)
            outside_intensity_stats.append(
                (region, np.mean(intensities), *quantiles))
    return outside_intensity_stats

def _calculate_radial_distribution(cell_mask, object_mask, channel_arrays, num_bins=6, spacing=None):
    """
    Calculate the radial distribution of average intensities for each object in each cell.

    Args:
        cell_mask (numpy.ndarray): The mask representing the cells.
        object_mask (numpy.ndarray): The mask representing the objects.
        channel_arrays (numpy.ndarray): The array of channel images, channel last.
        num_bins (int, optional): The number of bins for the radial distribution. Defaults to 6.
        spacing (tuple, optional): Voxel spacing from
            :func:`resolve_measurement_spacing`, used as ``sampling`` for the
            distance transform. ``None`` in 2-D. Without it the "distance"
            from an object boundary in a 3-D stack counts planes and pixels as
            equal steps, so a shell 3 planes away is binned with one 3 pixels
            away even when it is five times further off in micrometres, and
            every radial bin mixes the two.

    Returns:
        dict: A dictionary containing the radial distributions of average intensities for each object in each cell.
            The keys are tuples of (cell_label, object_label, channel_index), and the values are numpy arrays
            representing the radial distributions.

    """
    def _calculate_average_intensity(distance_map, single_channel_image, num_bins, region_mask):
        """
        Calculate the average intensity of a single-channel image based on the distance map.

        Only pixels inside ``region_mask`` (the cell) are binned. The previous
        version multiplied the distance map by the cell mask instead, which set
        every pixel outside the cell to distance 0 and dumped the whole field
        background into bin 0 — so ``rad_dist_..._bin_0`` measured background,
        not the innermost shell, and inverted the meaning of the feature.

        Args:
            distance_map (numpy.ndarray): Distance from the object boundary.
            single_channel_image (numpy.ndarray): The single-channel image.
            num_bins (int): The number of bins for the radial distribution.
            region_mask (numpy.ndarray): Boolean mask of the parent cell.

        Returns:
            numpy.ndarray: The radial distribution of average intensities.
            Bins with no pixels are NaN rather than a meaningless 0.
        """
        radial_distribution = np.full(num_bins, np.nan)
        in_region = distance_map[region_mask]
        if in_region.size == 0:
            return radial_distribution
        max_distance = in_region.max()
        if max_distance <= 0:
            radial_distribution[0] = single_channel_image[region_mask].mean()
            return radial_distribution
        for i in range(num_bins):
            min_distance = i * (max_distance / num_bins)
            max_distance_i = (i + 1) * (max_distance / num_bins)
            bin_mask = region_mask & (distance_map >= min_distance)
            if i == num_bins - 1:
                bin_mask &= (distance_map <= max_distance_i)
            else:
                bin_mask &= (distance_map < max_distance_i)
            if bin_mask.any():
                radial_distribution[i] = single_channel_image[bin_mask].mean()
        return radial_distribution


    object_radial_distributions = {}

    shape = np.asarray(cell_mask).shape
    whole = _whole_field_window(shape)
    cell_boxes = _label_bounding_boxes(cell_mask)
    object_boxes = _label_bounding_boxes(object_mask)

    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels != 0]

    for cell_label in cell_labels:
        cell_box = _box_for(cell_boxes, cell_label)
        cell_window = whole if cell_box is None else cell_box
        cell_region_in_box = cell_mask[cell_window] == cell_label

        object_labels = np.unique(object_mask[cell_window][cell_region_in_box])
        object_labels = object_labels[object_labels != 0]

        for object_label in object_labels:
            object_box = _box_for(object_boxes, object_label)
            if object_box is None:
                window = whole
            else:
                window = _grow_window(
                    _union_window(cell_window, object_box), 2, shape)
            cell_region = cell_mask[window] == cell_label
            objecyt_region = object_mask[window] == object_label
            object_boundary = find_boundaries(objecyt_region, mode='outer')
            distance_map = distance_transform_edt(~object_boundary, sampling=spacing)
            channels_in_window = channel_arrays[window]
            for channel_index in range(channel_arrays.shape[-1]):
                radial_distribution = _calculate_average_intensity(distance_map, channels_in_window[..., channel_index], num_bins, cell_region)
                object_radial_distributions[(cell_label, object_label, channel_index)] = radial_distribution

    return object_radial_distributions

def _calculate_correlation_object_level(channel_image1, channel_image2, mask, settings):
        """
        Calculate correlation at the object level between two channel images based on a mask.

        Args:
            channel_image1 (numpy.ndarray): The first channel image.
            channel_image2 (numpy.ndarray): The second channel image.
            mask (numpy.ndarray): The mask indicating the objects.
            settings (dict): Additional settings for correlation calculation.

        Returns:
            pandas.DataFrame: A DataFrame containing the correlation data at the object level.

        .. note::

           The ``M1_correlation_<t>`` / ``M2_correlation_<t>`` columns were
           removed on 2026-09-02 and are no longer written. They were never
           Manders' coefficients: both channels were cut at their *own
           within-object percentile* ``t`` and then shared a single overlap
           mask, so M1 was capped at the object's own top-``(100-t)``
           intensity fraction no matter where the other channel was -- with
           ``channel_image2 == channel_image1`` the value was that cap, not
           1.0. The pair was ~99% redundant (measured r(M1, M2) ~ 0.99) and
           two pure noise channels scored 0.047 rather than ~0.

           They were kept for a while beside the correct columns, behind
           ``corrected_manders``, so old plates kept agreeing with themselves.
           That shape is the problem: two definitions shipping at once,
           under names that do not say which produced them, with the WRONG
           one on by default. The setting is retired and the
           three correct columns are now written unconditionally:

           * ``manders_m1`` -- the true M1: the fraction of channel 1's
             above-background intensity that lies where channel 2 is above
             *its own* background.
           * ``manders_m2`` -- the mirror statistic.
           * ``manders_overlap_coefficient`` -- the actual Manders overlap
             coefficient ``sum(a*b) / sqrt(sum(a^2) * sum(b^2))`` on the
             background-subtracted vectors. Nothing in spaCR computed this
             before, despite the tooltips naming it.

           The background of each channel is estimated *inside each object*, as
           ``median + 3 * 1.4826 * MAD``. The factor is fixed and deliberately
           has no knob: an opt-in family needs one switch, not twenty. It is a
           modelling choice rather than a theorem -- it recovers ground truth at
           r = 1.0000 on uniform-Poisson synthetic background, and will do worse
           on a real field with a strong illumination gradient.

           All three are 0.0, never NaN, when a channel has no above-background
           signal in the object. That is forced, not cosmetic:
           ``utils.filter_dataframe_features`` does ``dropna(axis=1)``, so one
           NaN anywhere deletes the whole column from every model matrix, and
           64.5% of background-only objects would produce one. The cost is that
           "no signal" and "signal that does not colocalise" both read 0.0.
        """
        corr_data = {}
        boxes = _label_bounding_boxes(mask)
        whole = _whole_field_window(np.asarray(mask).shape)
        for i in np.unique(mask)[1:]:
            box = _box_for(boxes, i)
            window = whole if box is None else box
            object_mask = (mask[window] == i)
            object_channel_image1 = channel_image1[window][object_mask]
            object_channel_image2 = channel_image2[window][object_mask]
            if len(object_channel_image1) < 2 or len(object_channel_image2) < 2:
                pearson_corr = np.nan
            else:
                pearson_corr, _ = pearsonr(object_channel_image1, object_channel_image2)

            corr_data[i] = {f'label_correlation': i,
                            f'Pearson_correlation': pearson_corr}

            v1 = np.asarray(object_channel_image1, dtype=np.float64)
            v2 = np.asarray(object_channel_image2, dtype=np.float64)
            med1 = np.median(v1)
            thr1 = med1 + 3.0 * 1.4826 * np.median(np.abs(v1 - med1))
            med2 = np.median(v2)
            thr2 = med2 + 3.0 * 1.4826 * np.median(np.abs(v2 - med2))
            a = np.clip(v1 - thr1, 0, None)
            b = np.clip(v2 - thr2, 0, None)
            sa = a.sum()
            sb = b.sum()
            M1_true = float(a[v2 > thr2].sum() / sa) if sa > 0 else 0.0
            M2_true = float(b[v1 > thr1].sum() / sb) if sb > 0 else 0.0
            den = np.sqrt((a * a).sum() * (b * b).sum())
            MOC = float((a * b).sum() / den) if den > 0 else 0.0

            corr_data[i].update({'manders_m1': M1_true,
                                 'manders_m2': M2_true,
                                 'manders_overlap_coefficient': MOC})

        return pd.DataFrame(corr_data.values())

def _estimate_blur(image, mask=None):
    """
    Estimate focus as the variance of the Laplacian.

    Without ``mask`` this is the variance of the Laplacian of the whole array,
    which is only meaningful for a 2-D image.

    With ``mask`` (a boolean array the same shape as ``image`` selecting one
    object) the Laplacian is computed on the object's 2-D bounding-box patch,
    grown by one pixel so the 3x3 kernel has real neighbours, and the variance
    is taken over the object's *interior* — the mask eroded by one pixel with a
    3x3 structuring element.

    Two deliberate choices make this an actual focus measure:

    * The patch is the RAW image. Out-of-object pixels inside the bounding box
      are NOT zero-filled. Zero-filling puts a step edge at the object boundary
      whose second derivative dwarfs the texture being measured, so the score
      would track the object's perimeter-to-area ratio rather than its focus.
    * The variance is taken only over the eroded interior, so every sampled
      Laplacian value is determined solely by in-object pixels. That removes
      both the artificial edge and any contribution from the neighbouring
      background, without needing to fabricate values.

    Objects too thin to erode (one pixel wide) fall back to the un-eroded mask;
    those samples do see their neighbours, but the alternative is no value.

    Callers previously passed ``image[label == region_label]`` — a 1-D vector of
    the object's pixels in raster order. OpenCV treats that as an N x 1 image,
    so the result was a second difference along raster order: blind to vertical
    structure, sensitive to the row wrap-around, and not a focus measure.

    **3-D volumes are measured plane by plane in the xy plane.** Focus is an
    in-plane property: the objective's lateral resolution is what a blurred
    edge reports on, while the z step is coarse, the axial PSF is elongated,
    and consecutive planes are a different optical section rather than a
    finer-grained sampling of the same one. A single ``cv2.Laplacian`` call on
    a ``(Z, Y, X)`` array does not raise — OpenCV reads the third axis as up to
    512 colour channels, so it silently returns the second derivative in the
    **zy** plane, computed independently for each x column. That is a plausible
    number measured in the wrong plane, which is worse than an error. Here the
    kernel is applied to each ``(Y, X)`` plane and the variance is taken over
    the object's in-plane interior across all planes.

    :param image: Intensity image. Same shape as ``mask`` when ``mask`` is
        given; 2-D ``(Y, X)`` or 3-D ``(Z, Y, X)``.
    :param mask: Optional boolean object mask, same shape as ``image``.
    :returns: Variance of the Laplacian; ``nan`` when ``mask`` selects nothing.
    :raises ValueError: when ``mask`` is neither 2-D nor 3-D.
    """
    volumetric = False
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim not in (2, 3):
            raise ValueError(
                f"_estimate_blur takes a 2-D (Y, X) or 3-D (Z, Y, X) mask, got "
                f"{mask.ndim}-D of shape {mask.shape}.")
        if not mask.any():
            return np.nan
        volumetric = mask.ndim == 3
        y_axis, x_axis = (mask.ndim - 2, mask.ndim - 1)
        rows = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != y_axis)))
        cols = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != x_axis)))
        r0 = max(int(rows[0]) - 1, 0)
        r1 = min(int(rows[-1]) + 1, mask.shape[y_axis] - 1)
        c0 = max(int(cols[0]) - 1, 0)
        c1 = min(int(cols[-1]) + 1, mask.shape[x_axis] - 1)
        if volumetric:
            planes = np.flatnonzero(mask.any(axis=(1, 2)))
            z0, z1 = int(planes[0]), int(planes[-1])
            image = image[z0:z1 + 1, r0:r1 + 1, c0:c1 + 1]
            sub_mask = mask[z0:z1 + 1, r0:r1 + 1, c0:c1 + 1]
            structure = np.zeros((3, 3, 3), dtype=bool)
            structure[1] = generate_binary_structure(2, 2)
        else:
            image = image[r0:r1 + 1, c0:c1 + 1]
            sub_mask = mask[r0:r1 + 1, c0:c1 + 1]
            structure = generate_binary_structure(2, 2)
        interior = binary_erosion(sub_mask, structure=structure)
        if not interior.any():
            interior = sub_mask
    else:
        interior = None
        volumetric = np.asarray(image).ndim == 3

    if image.dtype != np.float64:
        image_float = image.astype(np.float64)
    else:
        image_float = image
    if volumetric:
        lap = np.empty(image_float.shape, dtype=np.float64)
        for z in range(image_float.shape[0]):
            lap[z] = cv2.Laplacian(
                np.ascontiguousarray(image_float[z]), cv2.CV_64F)
    else:
        lap = cv2.Laplacian(image_float, cv2.CV_64F)
    if interior is None:
        return lap.var()
    return float(lap[interior].var())

def _measure_intensity_distance(cell_mask, nucleus_mask, pathogen_mask, channel_arrays, settings):
    """
    Compute Gaussian-smoothed intensity-weighted centroid distances for each cell object.

    Works for a 2-D ``(Y, X)`` mask and a 3-D ``(Z, Y, X)`` volume. Three things
    are dimension-dependent and were 2-D-only:

    * the bounding box was unpacked as ``minr, minc = ...``, which raises
      ``ValueError: too many values to unpack`` on a 3-D coordinate array;
    * ``distance_transform_edt`` was called without ``sampling``, so on an
      anisotropic stack a distance of "3" meant 3 pixels across but 3 planes
      down, which is a different physical length;
    * ``gaussian_filter``'s scalar ``sigma`` smooths every axis equally, which
      on an anisotropic stack blurs far further in z in physical terms than in
      xy. The sigma is given per axis, scaled so it means the same physical
      distance on each.
    """

    sigma = settings.get('distance_gaussian_sigma', 1.0)
    ndim = _ndim_of(cell_mask)
    spacing, _stamp = resolve_measurement_spacing(settings, ndim)
    if spacing is not None:
        physical = float(sigma) * float(spacing[-1])
        filter_sigma = tuple(physical / float(s) for s in spacing)
    else:
        filter_sigma = sigma

    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels > 0]

    dfs = []
    nucleus_dt = distance_transform_edt(nucleus_mask == 0, sampling=spacing)
    pathogen_dt = distance_transform_edt(pathogen_mask == 0, sampling=spacing)

    for ch in range(channel_arrays.shape[-1]):
        channel_img = channel_arrays[..., ch]
        blurred_img = gaussian_filter(channel_img, sigma=filter_sigma)

        data = []
        for label in cell_labels:
            cell_coords = np.argwhere(cell_mask == label)
            if cell_coords.size == 0:
                data.append([label, np.nan, np.nan])
                continue

            lower = np.min(cell_coords, axis=0)
            upper = np.max(cell_coords, axis=0) + 1
            box = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))

            cell_submask = (cell_mask[box] == label)
            blurred_subimg = blurred_img[box]

            if np.sum(cell_submask) == 0:
                data.append([label, np.nan, np.nan])
                continue

            masked_intensity = blurred_subimg * cell_submask
            com_local = center_of_mass(masked_intensity)
            if np.isnan(com_local[0]):
                data.append([label, np.nan, np.nan])
                continue

            com_global = tuple(c + int(o) for c, o in zip(com_local, lower))
            index = tuple(int(v) for v in np.round(com_global).astype(int))

            if not all(0 <= v < s for v, s in zip(index, cell_mask.shape)):
                data.append([label, np.nan, np.nan])
                continue

            nucleus_dist = nucleus_dt[index]
            pathogen_dist = pathogen_dt[index]

            data.append([label, nucleus_dist, pathogen_dist])

        df = pd.DataFrame(data, columns=['label',
                                         f'cell_channel_{ch}_distance_to_nucleus',
                                         f'cell_channel_{ch}_distance_to_pathogen'])
        dfs.append(df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(
            df, on='label', how='outer', validate='one_to_one')

    return merged_df

def save_and_add_image_to_grid(png_channels, img_path, grid, plot=False):
    """
    Add an image to a grid and save it as PNG.

    Args:
        png_channels (ndarray): The crop in file order -- red plane first --
            as :func:`spacr.crops.build_png_channels` assembles it. Written
            without narrowing, so a ``uint16`` crop becomes a 16-bit PNG; a
            float crop is silently written as 8-bit by cv2. Four or more
            channels raise rather than losing one to an alpha plane.
        img_path (str): Where the PNG goes. Its parent folder is stamped with
            the format sidecar and **must already exist** -- the caller in
            ``_measure_crop_core`` creates it. If it does not, the stamp fails
            with a printed warning, ``cv2.imwrite`` returns False, and the
            call returns normally having written nothing at all. A bare
            filename with no directory part stamps the current working
            directory.
        grid (list): Anything with ``append``; read only when ``plot`` is
            true, and appended to in place, so the return value is the object
            that was passed in. ``None`` passes through untouched while
            ``plot`` is false.
        plot (bool): Truthiness, not identity, decides. False (the default)
            leaves ``grid`` completely untouched -- the PNG is still written --
            which is why an ordinary run ends with an empty grid. True appends
            the crop for :func:`img_list_to_grid`: a crop of exactly dtype
            ``uint16`` is appended as a high-byte narrowed ``uint8`` copy,
            every other dtype is appended unchanged, and a ``uint8`` crop is
            appended by reference, so a caller that reuses its buffer mutates
            what is already in the grid.

    Returns:
        grid (list): The same object that was passed in, with the crop
        appended only if ``plot`` was true.

    Raises:
        spacr.crops.CropError: ``png_channels`` has more than three channels.
        AttributeError: ``grid`` is ``None`` (or has no ``append``) and
            ``plot`` is true. The PNG has already been written by then.
        cv2.error: ``img_path`` has no extension cv2 recognises. The folder
            sidecar has already been written by then.

    .. note::

       **The file's colour slots hold what the mapping declares.** The
       caller assembles ``png_channels`` in file order — red plane first —
       with :func:`spacr.crops.build_png_channels` and
       :func:`spacr.crops.resolve_png_channel_mapping`; under the legacy
       ``settings['png_dims']`` list that mapping is entry 0 blue, 1 green,
       2 red, so ``png_dims[0]`` lands in the file's BLUE slot.

       That is the REVERSE of what ``png_dims`` reads like, and it is LEFT
       THAT WAY ON PURPOSE: every crop already on disk was written with this
       mapping, so flipping it would silently change what each colour means
       and invalidate the models trained on those crops. The mapping is
       declared rather than corrected.

       ``cv2.imwrite`` interprets a 3-channel array as BGR, so
       :func:`spacr.crops.to_cv2_bgr` reverses the array once, here, and
       cv2's interpretation lands the red plane in the file's red slot. It
       refuses more than three channels rather than letting cv2 write the
       fourth as an alpha plane for every reader to drop in silence.

       The format is versioned: :func:`spacr.crops.stamp_crop_folder` drops a
       ``.spacr_crop_format.json`` sidecar into the crop folder before the
       first PNG lands, recording format 3 (``declared_rgb``). An unmarked
       folder means format 1 (legacy), whose bytes match format 3 for the
       same declared mapping, so both are read as-is; only format 2, whose
       stored channel order is reversed, is corrected by
       :func:`spacr.crops.read_crop_png`, and
       ``spacr.crops.migrate_crop_folder`` rewrites such a folder in place.

       Crops are still ``uint16``, so these are 16-bit PNGs and no intensity
       is discarded at write time. The narrowing to 8 bit happens once, on
       read, in :func:`spacr.crops.narrow_to_uint8`, which always takes the
       HIGH BYTE (``// 256``) — replacing PIL's two incompatible rules (high
       byte for an RGB PNG, a *clip* at 255 for a single-channel one, which
       returned solid white for any crop brighter than that).
    """

    stamp_crop_folder(os.path.dirname(img_path))
    cv2.imwrite(img_path, to_cv2_bgr(png_channels))

    if plot:

        if png_channels.dtype == np.uint16:
            png_channels = (png_channels / 256).astype(np.uint8)
        
        grid.append(png_channels)
    
    return grid

def img_list_to_grid(grid, titles=None):
    """
    Plot a grid of images with optional titles.

    Args:
        grid (list): List of images to be plotted.
        titles (list): List of titles for the images.

    Returns:
        fig (Figure): The matplotlib figure object containing the image grid.
    """
    n_images = len(grid)
    grid_size = ceil(sqrt(n_images))
    
    with figure_style(theme_target()):
        fig, axs = plt.subplots(
            grid_size, grid_size, figsize=(15, 15), facecolor='black',
            squeeze=False)
    
        from matplotlib.patches import FancyBboxPatch
        for i, ax in enumerate(axs.flat):
            if i < n_images:
                image = grid[i]
                im = ax.imshow(image)
                ax.axis('off')
                ax.set_facecolor('black')

                h, w = image.shape[:2]
                r = max(2.0, min(h, w) * 0.08)
                bbox = FancyBboxPatch(
                    (0, 0), w - 1, h - 1,
                    boxstyle=f"round,pad=0,rounding_size={r}",
                    transform=ax.transData, facecolor='none', edgecolor='none')
                ax.add_patch(bbox)
                im.set_clip_path(bbox)

                if titles:
                    img_height, img_width = image.shape[:2]
                    text_size = max(min(img_width / (len(titles[i]) * 1.5), img_height / 10), 4)
                    ax.text(5, 5, titles[i], color='white', fontsize=text_size, ha='left', va='top', fontweight='bold')
            else:
                fig.delaxes(ax)

        plt.subplots_adjust(wspace=0.08, hspace=0.08)
        plt.tight_layout(pad=0.2)
        return fig


#: crop_mode entries that name a mask _measure_crop_core knows how to crop.
#: Crop modes, in the order the measure pipeline writes them. Membership is
#: checked against spacr.object_roles; the order stays here.
CROP_MODES = (
    'cell', 'nucleus', 'pathogen', 'cytoplasm', *ORGANELLE_ROLES)


def _per_crop_mode(value, n_modes, name):
    """Return ``value`` as a list with exactly one entry per ``crop_mode``.

    ``crop_mode`` is a list and every per-crop setting is indexed by its
    position in that list. ``png_size`` has had the ``* len(crop_ls)``
    broadcast since forever; ``dialate_pngs`` and ``dialate_png_ratios``
    never did. A scalar was hard-broadcast to LENGTH 3 (why 3? there were
    three object types when it was written) and a list was taken as given,
    so the shipped default ``dialate_png_ratios=[0.2]`` raised
    ``IndexError: list index out of range`` on the second crop mode of
    every field the moment a user listed two -- a top-level setting that
    simply did not work, and did not say so: ``_measure_crop_core`` catches
    the IndexError per field, so the run wrote the first mode's crops,
    skipped the rest, and finished reporting failed fields rather than a
    bad setting.

    A single value -- scalar or one-element list -- means "the same for
    every mode" and is broadcast silently, which is what ``png_size`` has
    always done. A list that is short but not length 1 is a real mistake:
    it is padded with its last entry so the run still produces crops, and
    said out loud, because losing every crop on a 1000-field plate to a
    typo'd list is worse than cropping two modes at the same ratio.

    :param value: the setting as the user wrote it; scalar or sequence.
    :param n_modes: ``len(crop_mode)``.
    :param name: setting name, for the message.
    :returns: list of length ``n_modes``.
    """
    values = list(value) if isinstance(value, (list, tuple)) else [value]

    if not n_modes:
        return []
    if not values:
        raise ValueError(
            f"Setting: {name} is empty but crop_mode asks for {n_modes} crop "
            f"mode(s); give it one value, or one per crop mode.")
    if len(values) == 1:
        return values * n_modes
    if len(values) < n_modes:
        print(f"Setting: {name}={value} has {len(values)} entries but "
              f"crop_mode has {n_modes}; reusing {values[-1]!r} for the "
              f"remaining {n_modes - len(values)}. Give {name} one value, or "
              f"one per crop mode, to choose them yourself.")
        return values + [values[-1]] * (n_modes - len(values))
    if len(values) > n_modes:
        print(f"Setting: {name}={value} has {len(values)} entries but "
              f"crop_mode has only {n_modes}; ignoring the extra "
              f"{len(values) - n_modes}.")
    return values[:n_modes]


#: ``settings`` keys naming a label plane of the merged array. A plane named
#: by one of these holds object IDENTITIES; every other plane holds intensity.
MASK_DIM_KEYS = tuple(f'{role}_mask_dim' for role in SEGMENTED_ROLES)


def _merged_mask_planes(data, settings):
    """Return the set of plane indices of ``data`` that hold labels, not signal."""
    return _intensity_mask_planes(data, settings)


def _promote_merged_to_uint16(data, settings, *, rescale_factor=None):
    """Bring a merged array that is neither ``uint8`` nor ``uint16`` into the
    measure pipeline's working dtype, **without flattening it**.

    ``data.astype(np.uint16)`` -- what this used to be -- is a truncation.
    ``spacr.io._normalize_img_batch`` writes normalised stacks as ``float32``
    on ``[0, 1]``, and every one of those pixels truncates to 0: a whole field
    measured as black, with an "Converted data from float32 to uint16" line as
    the only trace. Measured on a float32 field whose intensities span
    0.002-0.798, ``astype`` left 0 of 64 intensity pixels non-zero.

    The two kinds of plane are converted differently, because they mean
    different things:

    * **label planes** (:func:`_merged_mask_planes`) are rounded, never
      rescaled -- a label is an identity, and object 1 must stay object 1.
    * **intensity planes** are rescaled by ONE factor shared across all of
      them, so the ratio between channels is untouched: ``x65535`` when they
      live on ``[0, 1]``, ``x(65535/max)`` when they run past the 16-bit
      ceiling (where ``astype`` wrapped), and ``x1`` otherwise -- which is the
      ordinary ``int32``-from-a-concatenated-label-plane case, so that path
      keeps behaving exactly as it did.

    :param data: the merged array, ``(Y, X, C)`` or ``(Z, Y, X, C)``.
    :param settings: the measure settings, read for the ``*_mask_dim`` keys.
    :returns: ``(uint16 array, factor applied to the intensity planes)``.
    """
    arr = np.asarray(data)
    mask_planes = _merged_mask_planes(arr, settings)
    intensity = [p for p in range(int(arr.shape[-1])) if p not in mask_planes]

    factor = 1.0
    if intensity:
        signal = arr[..., intensity]
        top = float(np.nanmax(signal)) if signal.size else 0.0
        if not np.isfinite(top):
            top = float(np.nanmax(signal[np.isfinite(signal)])) \
                if np.isfinite(signal).any() else 0.0
        if rescale_factor is not None:
            factor = float(rescale_factor)
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(
                    f"intensity rescale factor must be finite and positive, "
                    f"got {rescale_factor!r}")
        elif top > 0:
            if np.issubdtype(arr.dtype, np.floating) and top <= 1.0:
                factor = 65535.0
            elif top > 65535.0:
                factor = 65535.0 / top

    out = np.zeros(arr.shape, dtype=np.uint16)
    for plane in range(int(arr.shape[-1])):
        values = np.nan_to_num(arr[..., plane].astype(np.float64),
                               nan=0.0, posinf=65535.0, neginf=0.0)
        if plane in intensity:
            values = values * factor
        out[..., plane] = np.rint(np.clip(values, 0, 65535)).astype(np.uint16)
    return out, factor


def _write_intensity_rescale_record(source_folder, file_name, settings,
                                    record, psf_record=None):
    """Upsert base rescaling and subsequent PSF provenance for one field.

    ``target_dtype`` describes the standard rescaling stage. The separate PSF
    provenance records the final float dtype, kernel and quantitative source.
    Older tables gain nullable signature/details and an original-source default.
    """
    from . import schema
    from .database_concurrency import connect, transaction

    field = schema.parse_field_stem(
        file_name, timelapse=bool(settings.get('timelapse', False)))
    values = {
        **field.to_dict(include_prcf=True),
        'timeID': field.timeID,
        'file_name': file_name,
        'path_name': os.path.join(settings['src'], file_name + '.npy'),
        'original_dtype': record.get('original_dtype'),
        'original_intensity_max': record.get('original_intensity_max'),
        'rescale_factor': float(record['rescale_factor']),
        'rescale_scope': record['rescale_scope'],
        'plate_intensity_max': record.get('plate_intensity_max'),
        'comparable_within_plate': int(
            bool(record.get('comparable_within_plate', False))),
        'target_dtype': 'uint16',
        'psf_measurement_source': (psf_record or {}).get('source', 'original'),
        'psf_signature': settings.get('_psf_measurement_signature'),
        'psf_provenance': json.dumps(psf_record, sort_keys=True, allow_nan=False),
    }
    columns = (
        'plateID', 'rowID', 'columnID', 'fieldID', 'timeID', 'prc', 'prcf',
        'file_name', 'path_name', 'original_dtype', 'original_intensity_max',
        'rescale_factor', 'rescale_scope', 'plate_intensity_max',
        'comparable_within_plate', 'target_dtype',
        'psf_measurement_source', 'psf_signature', 'psf_provenance',
    )
    db_path = os.path.join(source_folder, 'measurements', 'measurements.db')
    conn = connect(db_path, timeout=30)
    try:
        with transaction(conn, attempts=8, busy_timeout=30):
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS intensity_rescale (
                       plateID TEXT NOT NULL,
                       rowID TEXT NOT NULL,
                       columnID TEXT NOT NULL,
                       fieldID TEXT NOT NULL,
                       timeID TEXT,
                       prc TEXT NOT NULL,
                       prcf TEXT PRIMARY KEY,
                       file_name TEXT NOT NULL,
                       path_name TEXT NOT NULL,
                       original_dtype TEXT NOT NULL,
                       original_intensity_max REAL NOT NULL,
                       rescale_factor REAL NOT NULL,
                       rescale_scope TEXT NOT NULL,
                       plate_intensity_max REAL,
                       comparable_within_plate INTEGER NOT NULL,
                       target_dtype TEXT NOT NULL
                   )''')
            existing = {row[1] for row in conn.execute(
                'PRAGMA table_info(intensity_rescale)')}
            for column, declaration in (
                    ('psf_measurement_source', "TEXT NOT NULL DEFAULT 'original'"),
                    ('psf_signature', 'TEXT'), ('psf_provenance', 'TEXT')):
                if column not in existing:
                    conn.execute(f'ALTER TABLE intensity_rescale ADD COLUMN {column} {declaration}')
            placeholders = ', '.join('?' for _ in columns)
            quoted = ', '.join(f'"{column}"' for column in columns)
            updates = ', '.join(
                f'"{column}" = excluded."{column}"'
                for column in columns if column != 'prcf')
            conn.execute(
                f'INSERT INTO intensity_rescale ({quoted}) '
                f'VALUES ({placeholders}) ON CONFLICT(prcf) DO UPDATE SET '
                f'{updates}',
                tuple(values[column] for column in columns))
    finally:
        conn.close()


_CONFLUENCY_SOURCES = ('auto', 'masks', 'texture', 'intensity')
_CONFLUENCY_TABLE = 'confluency'
_CONFLUENCY_WELL_TABLE = 'confluency_well'
_CONFLUENCY_WELL_KEYS = ('plateID', 'rowID', 'columnID')
_CONFLUENCY_SEPARATION_MIN = 3.2
_CONFLUENCY_TEXTURE_RATIO_MIN = 3.0
_CONFLUENCY_INTENSITY_FRACTION = 0.25


@dataclass
class _ConfluencyResult:
    """Covered area of one field and how it was decided.

    ``confluency`` is the covered fraction of the field, 0 to 1.
    ``source`` is the method actually used, never ``auto``. ``threshold``
    is the automatic cut in the units of that method (local standard
    deviation of the 0-1 scaled image for texture, raw intensity for
    intensity, ``None`` for masks). ``separation`` is how far apart the two
    pixel classes were, in pooled standard deviations; a field whose
    classes did not separate is decided whole and ``uniform`` is true.
    """

    covered: np.ndarray
    confluency: float
    source: str
    threshold: Optional[float] = None
    separation: Optional[float] = None
    uniform: bool = False
    channel: Optional[int] = None

    @property
    def covered_px(self) -> int:
        """Number of covered pixels."""
        return int(np.count_nonzero(self.covered))

    @property
    def field_px(self) -> int:
        """Number of pixels in the field."""
        return int(self.covered.size)


def _confluency_plane(array):
    """Reduce a field to one 2-D plane; a z-stack is max-projected.

    :param array: a 2-D image or mask, or a ``(Z, Y, X)`` stack.
    :returns: the 2-D plane.
    """
    plane = np.asarray(array)
    if plane.ndim == 3:
        plane = plane.max(axis=0)
    if plane.ndim != 2:
        raise ValueError(
            f"confluency needs a 2-D field or a (Z, Y, X) stack, got shape "
            f"{plane.shape}")
    return plane


def _otsu_separation(values):
    """Otsu's cut and how far apart the two classes it makes are.

    :param values: 1-D finite values.
    :returns: ``(threshold, separation)``; separation is the difference of
        the class means over the pooled within-class standard deviation.
        A single Gaussian split this way gives about 2.6, two real classes
        give well above :data:`_CONFLUENCY_SEPARATION_MIN`.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 4 or np.ptp(values) == 0:
        return float(values.mean()) if values.size else 0.0, 0.0
    threshold = float(filters.threshold_otsu(values))
    low = values[values <= threshold]
    high = values[values > threshold]
    if low.size < 2 or high.size < 2:
        return threshold, 0.0
    within = (low.size * low.var() + high.size * high.var()) / values.size
    separation = (high.mean() - low.mean()) / sqrt(max(within, 1e-12))
    return threshold, float(separation)


def _unit_scaled(plane):
    """Scale a plane to 0-1 between its 0.5th and 99.5th percentiles.

    :param plane: 2-D image.
    :returns: float64 plane clipped to ``[0, 1]``; all zeros when flat.
    """
    x = np.asarray(plane, dtype=np.float64)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _local_sd(x, window):
    """Standard deviation in a square window around every pixel.

    :param x: float 2-D plane.
    :param window: window side in pixels.
    :returns: the local standard deviation, same shape as ``x``.
    """
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(x, window, mode='reflect')
    mean_sq = uniform_filter(x * x, window, mode='reflect')
    return np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))


def _texture_ratio(x, window):
    """Median local standard deviation over the pixel noise level.

    About 1 on an empty, flat field and several times that on one covered
    by cells, whatever the stain. It decides a field whose pixels do not
    separate into two classes, because such a field is either all
    background or all monolayer.

    :param x: 0-1 scaled plane.
    :param window: window side in pixels.
    :returns: the ratio.
    """
    from skimage.restoration import estimate_sigma
    noise = float(estimate_sigma(x))
    return float(np.median(_local_sd(x, window))) / max(noise, 1e-9)


def _clean_coverage(covered, radius):
    """Smooth a coverage mask: open, close, and drop specks and pinholes.

    The image is reflected at its edges first so that a cell sheet running
    off the field is not eroded there.

    :param covered: boolean plane.
    :param radius: structuring-element radius in pixels.
    :returns: the cleaned boolean plane.
    """
    from scipy.ndimage import binary_closing, binary_opening
    radius = max(1, int(radius))
    pad = 3 * radius
    work = np.pad(np.asarray(covered, dtype=bool), pad, mode='reflect')
    disk = morphology.disk(radius)
    work = binary_opening(work, structure=disk)
    work = binary_closing(work, structure=disk)
    smallest = int(np.pi * (2 * radius) ** 2)
    work = morphology.remove_small_holes(work, smallest)
    work = morphology.remove_small_objects(work, smallest)
    return work[pad:-pad, pad:-pad]


def _texture_coverage(image, window=15):
    """Covered area of a brightfield or phase field, from local texture.

    Cells scatter light and so vary from pixel to pixel; bare plastic is
    flat. The local standard deviation in a ``window``-pixel square is
    split by Otsu's method on its logarithm, which finds the two levels.
    The final cut sits halfway between the two levels in variance, which
    puts the edge where half the window is covered, so the coverage is not
    inflated by half a window all round every cell. Pixels whose window is
    perfectly flat (saturated or zero-padded borders) are left out of the
    threshold estimate and count as uncovered.

    :param image: 2-D image, or a ``(Z, Y, X)`` stack (max-projected).
    :param window: texture window side in pixels; roughly the width of the
        thinnest cell process that should count as covered.
    :returns: :class:`_ConfluencyResult` with ``source='texture'``.
    """
    window = max(3, int(window))
    x = _unit_scaled(_confluency_plane(image))
    sd = _local_sd(x, window)
    textured = sd > 1e-9
    if not textured.any():
        return _ConfluencyResult(np.zeros(x.shape, dtype=bool), 0.0,
                                'texture', None, 0.0, True)
    log_sd = np.log(sd[textured])
    lo, hi = np.percentile(log_sd, [0.5, 99.5])
    cut, separation = _otsu_separation(np.clip(log_sd, lo, hi))
    if separation < _CONFLUENCY_SEPARATION_MIN:
        full = _texture_ratio(x, window) >= _CONFLUENCY_TEXTURE_RATIO_MIN
        covered = np.full(x.shape, bool(full))
        return _ConfluencyResult(covered, float(full), 'texture', None,
                                separation, True)
    first = np.zeros(x.shape, dtype=bool)
    first[textured] = log_sd > cut
    band = window // 2 + 1
    variance = sd * sd
    core_on = binary_erosion(first, iterations=band)
    core_off = binary_erosion(~first & textured, iterations=band)
    on = variance[core_on] if core_on.any() else variance[first]
    off = (variance[core_off] if core_off.any()
           else variance[~first & textured])
    level = 0.5 * (float(np.median(on)) + float(np.median(off)))
    covered = _clean_coverage((variance > level) & textured, window // 4)
    return _ConfluencyResult(covered, float(covered.mean()), 'texture',
                            sqrt(level), separation, False)


def _intensity_coverage(image, sigma=1.0):
    """Covered area of a fluorescent cytoplasm or membrane stain.

    The plane is smoothed, its brightest 0.1 % clipped so a few saturated
    spots cannot capture Otsu's cut, and split by Otsu's method. The cut
    is then moved a quarter of the way up from the background level to the
    stained level (medians of the two classes away from their edges), so
    the dim rim of each cell counts as covered, which is where hand-drawn
    and Cellpose outlines put it. On the Toxoplasma PV ground-truth fields
    Otsu alone reported half the covered area.

    :param image: 2-D image, or a ``(Z, Y, X)`` stack (max-projected).
    :param sigma: Gaussian smoothing in pixels before thresholding.
    :returns: :class:`_ConfluencyResult` with ``source='intensity'``.
    """
    plane = np.asarray(_confluency_plane(image), dtype=np.float64)
    ceiling = float(np.percentile(plane, 99.9))
    x = gaussian_filter(np.minimum(plane, ceiling), sigma)
    cut, separation = _otsu_separation(x.ravel())
    if separation < _CONFLUENCY_SEPARATION_MIN:
        full = _texture_ratio(_unit_scaled(plane), 15) >= (
            _CONFLUENCY_TEXTURE_RATIO_MIN)
        return _ConfluencyResult(np.full(x.shape, bool(full)), float(full),
                                'intensity', None, separation, True)
    above = x > cut
    core_on = binary_erosion(above, iterations=3)
    core_off = binary_erosion(~above, iterations=8)
    stained = float(np.median(x[core_on] if core_on.sum() > 100 else x[above]))
    background = float(np.median(
        x[core_off] if core_off.sum() > 100 else x[~above]))
    level = background + _CONFLUENCY_INTENSITY_FRACTION * (stained - background)
    covered = _clean_coverage(x > level, 2)
    return _ConfluencyResult(covered, float(covered.mean()), 'intensity',
                            level, separation, False)


def _mask_coverage(mask):
    """Covered area as the union of every labelled cell.

    :param mask: 2-D label image, or a ``(Z, Y, X)`` label stack (a pixel is
        covered when any plane labels it).
    :returns: :class:`_ConfluencyResult` with ``source='masks'``.
    """
    covered = _confluency_plane(np.asarray(mask) > 0).astype(bool)
    return _ConfluencyResult(covered, float(covered.mean()), 'masks')


def _resolve_confluency_source(settings):
    """The method a run uses, with ``auto`` answered.

    ``auto`` is the cell masks when the run has a cell mask, and texture
    otherwise, because texture works on any channel, brightfield included.

    :param settings: Measure settings; reads ``confluency_source`` and
        ``cell_mask_dim``.
    :returns: ``'masks'``, ``'texture'`` or ``'intensity'``.
    :raises ValueError: for a source outside :data:`_CONFLUENCY_SOURCES`.
    """
    source = str(settings.get('confluency_source') or 'auto').strip().lower()
    if source not in _CONFLUENCY_SOURCES:
        raise ValueError(
            f"Setting: confluency_source is {source!r}; use one of "
            f"{', '.join(_CONFLUENCY_SOURCES)}.")
    has_cells = settings.get('cell_mask_dim') is not None
    if source == 'auto':
        return 'masks' if has_cells else 'texture'
    if source == 'masks' and not has_cells:
        raise ValueError(
            "Setting: confluency_source is 'masks' but cell_mask_dim is "
            "blank, so there are no cell masks to cover the field with. "
            "Set cell_mask_dim, or choose texture or intensity.")
    return source


def _confluency_channel(settings):
    """The merged-array channel a texture or intensity source reads.

    :param settings: Measure settings; reads ``confluency_channel`` and,
        when it is blank, the first entry of ``channels``.
    :returns: the channel index.
    """
    channel = settings.get('confluency_channel')
    if channel is None or channel == '':
        channels = settings.get('channels') or [0]
        channel = channels[0]
    return int(channel)


def _field_confluency(image=None, cell_mask=None, *, source='auto', window=15,
                     channel=None):
    """Covered fraction of one field by the chosen source.

    :param image: the channel to read for ``texture`` and ``intensity``.
    :param cell_mask: the cell label image for ``masks``.
    :param source: ``auto`` (masks when ``cell_mask`` is given, else
        texture), ``masks``, ``texture`` or ``intensity``.
    :param window: texture window in pixels.
    :param channel: recorded on the result; not used to read anything.
    :returns: :class:`_ConfluencyResult`.
    :raises ValueError: for an unknown source or a missing input.
    """
    source = str(source or 'auto').strip().lower()
    if source not in _CONFLUENCY_SOURCES:
        raise ValueError(f"unknown confluency source {source!r}; use one of "
                         f"{', '.join(_CONFLUENCY_SOURCES)}")
    if source == 'auto':
        source = 'masks' if cell_mask is not None else 'texture'
    if source == 'masks':
        if cell_mask is None:
            raise ValueError("the masks confluency source needs a cell mask")
        return _mask_coverage(cell_mask)
    if image is None:
        raise ValueError(f"the {source} confluency source needs an image")
    result = (_texture_coverage(image, window) if source == 'texture'
              else _intensity_coverage(image))
    result.channel = None if channel is None else int(channel)
    return result


def _confluency_overlay(image, covered, *, color=(255, 170, 0), alpha=0.35):
    """An RGB preview of the covered area over the field.

    The field is shown in grey, the covered area tinted, and the edge of the
    covered area drawn solid, so gaps in the monolayer read at a glance.

    :param image: 2-D image, or a ``(Z, Y, X)`` stack (max-projected).
    :param covered: boolean coverage plane of the same shape.
    :param color: tint as an RGB triple.
    :param alpha: tint opacity inside the covered area.
    :returns: ``uint8`` array of shape ``(Y, X, 3)``.
    """
    grey = (_unit_scaled(_confluency_plane(image)) * 255.0)
    rgb = np.repeat(grey[..., None], 3, axis=-1)
    covered = np.asarray(covered, dtype=bool)
    tint = np.asarray(color, dtype=np.float64)
    rgb[covered] = (1.0 - alpha) * rgb[covered] + alpha * tint
    edge = find_boundaries(covered, mode='inner')
    rgb[edge] = tint
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _confluency_figure(image, result, title):
    """A matplotlib figure of :func:`_confluency_overlay` for the run's plots.

    :param image: the plane the overlay is drawn on.
    :param result: the field's :class:`_ConfluencyResult`.
    :param title: the field name.
    :returns: the figure.
    """
    with figure_style(theme_target()):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(_confluency_overlay(image, result.covered))
        ax.set_title(f"{title}: {result.confluency:.1%} covered "
                     f"({result.source})")
        ax.axis('off')
    return fig


def _measure_field_confluency(data, settings, channel_arrays=None,
                             cell_mask=None):
    """Confluency of one merged field as the run's settings ask for it.

    ``masks`` reads the cell mask plane as Mask wrote it, before Measure's
    size filters: confluency is about the monolayer, not about which cells
    are kept for measurement.

    :param data: the merged array, ``(Y, X, C)`` or ``(Z, Y, X, C)``.
    :param settings: Measure settings.
    :param channel_arrays: the preprocessed measured channels, used for the
        confluency channel when it is one of ``channels``.
    :param cell_mask: the cell label plane as loaded, before any filter;
        read from ``data`` at ``cell_mask_dim`` when omitted.
    :returns: ``(result, plane)`` where ``plane`` is the image the overlay
        should be drawn on.
    """
    source = _resolve_confluency_source(settings)
    channel = _confluency_channel(settings)
    measured = list(settings.get('channels') or [])
    if channel_arrays is not None and channel in measured:
        plane = np.asarray(channel_arrays[..., measured.index(channel)])
    else:
        if channel >= data.shape[-1]:
            raise ValueError(
                f"Setting: confluency_channel is {channel}, but the merged "
                f"array has {data.shape[-1]} planes.")
        plane = np.asarray(data[..., channel])
    if source == 'masks':
        if cell_mask is None:
            cell_mask = data[..., settings['cell_mask_dim']]
        result = _mask_coverage(cell_mask)
    else:
        result = _field_confluency(
            plane, source=source,
            window=int(settings.get('confluency_window') or 15),
            channel=channel)
    return result, plane


def _monolayer_ok(confluency, qc_threshold):
    """Whether a monolayer passes QC: covered fraction at or above the cut.

    :param confluency: covered fraction, 0 to 1.
    :param qc_threshold: the lowest acceptable fraction; ``None`` passes.
    :returns: bool.
    """
    if qc_threshold is None:
        return True
    return bool(float(confluency) >= float(qc_threshold))


def _write_confluency_record(source_folder, file_name, settings, result):
    """Upsert one field's confluency into ``measurements.db:confluency``.

    :param source_folder: the run folder holding ``measurements/``.
    :param file_name: the merged field's stem.
    :param settings: Measure settings.
    :param result: the field's :class:`_ConfluencyResult`.
    """
    from . import schema
    from .database_concurrency import connect, transaction

    qc_threshold = settings.get('confluency_qc_threshold')
    field = schema.parse_field_stem(
        file_name, timelapse=bool(settings.get('timelapse', False)))
    values = {
        **field.to_dict(include_prcf=True),
        'timeID': field.timeID,
        'file_name': file_name,
        'confluency_source': result.source,
        'confluency_channel': (None if result.source == 'masks'
                               else result.channel),
        'confluency': float(result.confluency),
        'covered_px': result.covered_px,
        'field_px': result.field_px,
        'confluency_threshold': result.threshold,
        'confluency_separation': result.separation,
        'confluency_uniform': int(bool(result.uniform)),
        'confluency_qc_threshold': (None if qc_threshold is None
                                    else float(qc_threshold)),
        'monolayer_ok': int(_monolayer_ok(result.confluency, qc_threshold)),
    }
    columns = tuple(values)
    db_path = os.path.join(source_folder, 'measurements', 'measurements.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = connect(db_path, timeout=30)
    try:
        with transaction(conn, attempts=8, busy_timeout=30):
            conn.execute(
                f'''CREATE TABLE IF NOT EXISTS {_CONFLUENCY_TABLE} (
                       plateID TEXT NOT NULL,
                       rowID TEXT NOT NULL,
                       columnID TEXT NOT NULL,
                       fieldID TEXT NOT NULL,
                       timeID TEXT,
                       prc TEXT NOT NULL,
                       prcf TEXT PRIMARY KEY,
                       file_name TEXT NOT NULL,
                       confluency_source TEXT NOT NULL,
                       confluency_channel INTEGER,
                       confluency REAL NOT NULL,
                       covered_px INTEGER NOT NULL,
                       field_px INTEGER NOT NULL,
                       confluency_threshold REAL,
                       confluency_separation REAL,
                       confluency_uniform INTEGER NOT NULL,
                       confluency_qc_threshold REAL,
                       monolayer_ok INTEGER NOT NULL
                   )''')
            quoted = ', '.join(f'"{column}"' for column in columns)
            placeholders = ', '.join('?' for _ in columns)
            updates = ', '.join(
                f'"{column}" = excluded."{column}"'
                for column in columns if column != 'prcf')
            conn.execute(
                f'INSERT INTO {_CONFLUENCY_TABLE} ({quoted}) '
                f'VALUES ({placeholders}) ON CONFLICT(prcf) DO UPDATE SET '
                f'{updates}',
                tuple(values[column] for column in columns))
    finally:
        conn.close()


def _read_confluency(db_path):
    """The per-field confluency table, or an empty frame when there is none.

    :param db_path: a ``measurements.db``.
    :returns: one row per field.
    """
    from .database_concurrency import connect
    if not os.path.isfile(db_path):
        return pd.DataFrame()
    conn = connect(db_path, readonly=True)
    try:
        present = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_CONFLUENCY_TABLE,)).fetchone()
        if not present:
            return pd.DataFrame()
        from .tabular import _read_query
        return _read_query(conn, f'SELECT * FROM {_CONFLUENCY_TABLE}',
                           canonicalise=False)
    finally:
        conn.close()


def _confluency_by_well(fields, qc_threshold=None):
    """Aggregate per-field confluency to one row per well.

    ``confluency`` is pooled: covered pixels over imaged pixels across the
    well's fields, so a small field does not count as much as a large one.
    The mean, median, minimum and spread of the per-field values sit beside
    it, because a well whose mean is fine but one of whose fields is bare
    is a settling gradient worth seeing. Time-lapse fields keep their
    ``timeID``, one row per well per timepoint.

    :param fields: the per-field table (:func:`_read_confluency`).
    :param qc_threshold: the monolayer QC cut; ``None`` reads it from the
        fields' ``confluency_qc_threshold``.
    :returns: one row per well with ``n_fields``, ``covered_px``,
        ``field_px``, ``confluency``, ``confluency_mean``,
        ``confluency_median``, ``confluency_min``, ``confluency_sd``,
        ``fields_below_qc``, ``confluency_qc_threshold`` and
        ``monolayer_ok``.
    """
    if fields is None or fields.empty:
        return pd.DataFrame()
    keys = list(_CONFLUENCY_WELL_KEYS)
    if 'timeID' in fields.columns and fields['timeID'].notna().any():
        keys.append('timeID')
    if qc_threshold is None and 'confluency_qc_threshold' in fields.columns:
        known = fields['confluency_qc_threshold'].dropna()
        qc_threshold = float(known.iloc[-1]) if not known.empty else None
    rows = []
    for name, block in fields.groupby(keys, dropna=False, sort=True):
        identity = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        covered = int(block['covered_px'].sum())
        total = int(block['field_px'].sum())
        pooled = covered / total if total else float('nan')
        per_field = block['confluency'].astype(float)
        rows.append({
            **identity,
            'prc': f"{identity['plateID']}_{identity['rowID']}_"
                   f"{identity['columnID']}",
            'n_fields': int(len(block)),
            'covered_px': covered,
            'field_px': total,
            'confluency': pooled,
            'confluency_mean': float(per_field.mean()),
            'confluency_median': float(per_field.median()),
            'confluency_min': float(per_field.min()),
            'confluency_sd': (float(per_field.std(ddof=1))
                              if len(per_field) > 1 else 0.0),
            'fields_below_qc': int(sum(
                not _monolayer_ok(value, qc_threshold) for value in per_field)),
            'confluency_qc_threshold': qc_threshold,
            'monolayer_ok': int(_monolayer_ok(pooled, qc_threshold)),
        })
    return pd.DataFrame(rows)


def _aggregate_confluency_by_well(db_path, qc_threshold=None):
    """Rebuild ``measurements.db:confluency_well`` from the field table.

    Rebuilt whole rather than appended, so a field re-measured or dropped
    since the last run is reflected rather than counted twice.

    :param db_path: a ``measurements.db`` holding a ``confluency`` table.
    :param qc_threshold: the monolayer QC cut; ``None`` keeps the one each
        field was written with.
    :returns: the per-well frame written, empty when there were no fields.
    """
    from .tabular import write_database

    wells = _confluency_by_well(_read_confluency(db_path), qc_threshold)
    if wells.empty:
        return wells
    write_database(wells, db_path, _CONFLUENCY_WELL_TABLE,
                   if_exists='replace', canonicalise=False)
    return wells


def _read_confluency_wells(source):
    """The per-well confluency frame from a database path or a frame.

    :param source: a ``measurements.db`` path, or a frame already in the
        :func:`_confluency_by_well` shape.
    :returns: the per-well frame, empty when none was written.
    """
    if isinstance(source, pd.DataFrame):
        return source
    from .database_concurrency import connect
    if not source or not os.path.isfile(source):
        return pd.DataFrame()
    conn = connect(source, readonly=True)
    try:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if _CONFLUENCY_WELL_TABLE in tables:
            from .tabular import _read_query
            return _read_query(
                conn, f'SELECT * FROM {_CONFLUENCY_WELL_TABLE}',
                canonicalise=False)
    finally:
        conn.close()
    return _confluency_by_well(_read_confluency(source))


def _monolayer_qc(frame, confluency, *, value_columns=(), drop_failing=False,
                 well_of=None, qc_threshold=None):
    """Join per-well confluency onto any per-well table, as filter and denominator.

    The plaque and infection assays count things per well; a thin or torn
    monolayer makes those counts smaller for a reason that has nothing to
    do with the treatment. This adds the well's ``confluency`` and
    ``monolayer_ok`` to every row, and for each of ``value_columns`` a
    ``<column>_per_confluency`` column -- the value over the covered
    fraction, i.e. per fully covered field -- so wells can be compared per
    unit of monolayer and wells failing QC dropped.

    :param frame: rows carrying ``plateID``, ``rowID`` and ``columnID``,
        or a ``file`` column when ``well_of`` is given.
    :param confluency: a ``measurements.db`` path or a per-well frame from
        :func:`_confluency_by_well`.
    :param value_columns: counts or areas to divide by the covered fraction.
    :param drop_failing: drop rows whose well fails monolayer QC.
    :param well_of: optional callable turning a row into a
        ``(plateID, rowID, columnID)`` triple, for tables named by file
        (a plaque image named after its well).
    :param qc_threshold: re-decide ``monolayer_ok`` at this cut instead of
        the one the run wrote.
    :returns: a new frame; rows with no confluency for their well get
        ``NaN`` confluency and ``monolayer_ok`` of ``NaN``, never a pass.
    """
    wells = _read_confluency_wells(confluency)
    out = frame.copy()
    keys = list(_CONFLUENCY_WELL_KEYS)
    if well_of is not None:
        triples = [tuple(well_of(row)) for _, row in out.iterrows()]
        for index, key in enumerate(keys):
            out[key] = [triple[index] for triple in triples]
    missing = [key for key in keys if key not in out.columns]
    if missing:
        raise ValueError(
            f"_monolayer_qc needs the well columns {missing}; pass well_of "
            f"to derive them.")
    if wells.empty:
        out['confluency'] = np.nan
        out['monolayer_ok'] = np.nan
    else:
        if 'timeID' in wells.columns and wells['timeID'].notna().any() and (
                'timeID' in out.columns):
            keys = keys + ['timeID']
        table = wells[keys + ['confluency', 'monolayer_ok']].copy()
        if qc_threshold is not None:
            table['monolayer_ok'] = [
                int(_monolayer_ok(value, qc_threshold))
                for value in table['confluency']]
        for key in keys:
            out[key] = out[key].astype(str)
            table[key] = table[key].astype(str)
        out = out.drop(columns=[c for c in ('confluency', 'monolayer_ok')
                                if c in out.columns])
        out = out.merge(table, how='left', on=keys)
    for column in value_columns:
        cover = out['confluency'].astype(float)
        out[f'{column}_per_confluency'] = (
            out[column].astype(float) / cover.where(cover > 0))
    if drop_failing:
        out = out[out['monolayer_ok'] == 1].reset_index(drop=True)
    return out


def _measure_crop_core(index, time_ls, file, settings, psf_plan=None, psf_cancel=None):

    """Measure one field using selected standard or PSF-processed intensities.

    :param index: position of this field in the run's input list.
    :param time_ls: shared collection of completed field durations.
    :param file: merged NPY filename below ``settings['src']``.
    :param settings: Measure configuration; original PSF intensity choice is
        the default. Label planes and exported crops keep their source pixels.
    :param psf_plan: immutable plan captured by the parent. When omitted for
        a direct processed call, the worker prepares one from its settings.
    :param psf_cancel: optional process-safe cancellation event.
    :returns: index, mean duration, surviving cell labels (or failure sentinel
        zero), figures, and error text. Cancellation propagates to the parent.
    """
    
    from .utils import _merge_overlapping_objects, _filter_object, _relabel_parent_with_child_labels, _exclude_objects, normalize_to_dtype, filepaths_to_database
    from .utils import _merge_and_save_to_database, _crop_center, _find_bounding_box, _generate_names, _get_percentiles

    from .cancellation import PipelineCancelled
    from .psf_measurement import (prepare_measurement_psf, measurement_psf_record,
                                  measurement_psf_signature, SIGNATURE_KEY)

    figs = {}
    grid = []
    start = time.time() 
    try:
        source_folder = os.path.dirname(settings['src'])

        file_name = os.path.splitext(file)[0]
        data = np.load(os.path.join(settings['src'], file))
        data_type_before = data.dtype
        rescale_record = _resolve_intensity_rescale_record(
            data, file, settings)
        factor = float(rescale_record['rescale_factor'])
        if _intensity_scale_needs_warning(factor):
            if rescale_record['rescale_scope'] == 'plate':
                detail = (
                    f"plate-wide from maximum "
                    f"{rescale_record['plate_intensity_max']:g}; all raw-valued "
                    f"fields on this plate use the same factor")
            else:
                detail = (
                    "per-field fallback; this field is NOT comparable to "
                    "other fields on the plate")
            print(f"WARNING: {file_name} intensity values require x{factor:g} "
                  f"rescaling to uint16 ({detail}). The decision is recorded "
                  f"in measurements.db:intensity_rescale.")

        data_type = data.dtype
        if data_type not in ['uint8','uint16'] or not np.isclose(factor, 1.0):
            data, factor = _promote_merged_to_uint16(
                data, settings, rescale_factor=factor)
            data_type = data.dtype
            if settings['verbose']:
                scale = '' if factor == 1.0 else f' (intensity x{factor:g})'
                print(f'Converted data from {data_type_before} to {data_type}{scale}')

        if data.ndim == 4 and data.shape[0] == 1:
            data = data[0]
        volumetric = data.ndim == 4
        n_z = int(data.shape[0]) if volumetric else 1
        spacing, units_stamp = resolve_measurement_spacing(
            settings, 3 if volumetric else 2, n_z=n_z)

        if settings['plot'] and volumetric:
            print(f"3-D field {file_name}: skipping the cropped-array plots "
                  f"(spacr.plot renders 2-D fields).")
        elif settings['plot']:
            from .plot import _plot_cropped_arrays
            if len(data.shape) == 3:
                figuresize = data.shape[2]*10
            else:
                figuresize = 10
            fig = _plot_cropped_arrays(data, file, figuresize)
            figs[f'{file_name}__before_filtration'] = fig

        channel_arrays = data[..., settings['channels']].astype(data_type)

        if preprocessing_hooks():
            channel_arrays = apply_preprocessing_hooks(
                channel_arrays,
                PreprocessingContext(
                    file_name=file_name,
                    channels=settings['channels'],
                    settings=settings,
                    volumetric=volumetric,
                    spacing=spacing))

        if settings.get('psf_measurement_source', 'original') == 'original':
            psf_plan = None
        if psf_plan is None:
            psf_plan = prepare_measurement_psf(settings)
        settings = dict(settings)
        settings[SIGNATURE_KEY] = measurement_psf_signature(psf_plan)
        psf_record = measurement_psf_record(
            psf_plan, channel_arrays, hooks=[hook.name for hook in preprocessing_hooks()],
            channels=settings['channels'])
        if psf_plan is not None:
            if volumetric and len(psf_plan.sampling_um) == 3:
                psf_spacing = np.asarray(psf_plan.sampling_um)
                measure_spacing = np.asarray(spacing)
                if units_stamp['voxel_size_z_um'] is None:
                    psf_spacing = psf_spacing / psf_spacing[-1]
                    measure_spacing = measure_spacing / measure_spacing[-1]
                if not np.allclose(psf_spacing, measure_spacing, rtol=1e-6, atol=0):
                    raise ValueError('PSF sampling conflicts with Measure voxel calibration')
            channel_arrays = psf_plan.apply(channel_arrays, cancel=psf_cancel)

        confluency_cells = (
            np.array(data[..., settings['cell_mask_dim']], copy=True)
            if settings.get('confluency')
            and settings.get('cell_mask_dim') is not None else None)

        if settings['cell_mask_dim'] is not None:
            cell_mask = data[..., settings['cell_mask_dim']].astype(data_type)

            cell_max = settings.get('cell_max_size')
            if ((settings['cell_min_size'] is not None
                 and settings['cell_min_size'] != 0) or cell_max):
                before = int(len(np.unique(cell_mask)) - 1)
                cell_mask = _filter_object(
                    cell_mask, settings['cell_min_size'],
                    max_value=cell_max)
                dropped = before - int(len(np.unique(cell_mask)) - 1)
                if dropped and cell_max:
                    print(f'cell: {dropped} object(s) outside '
                          f'[{settings["cell_min_size"]}, {cell_max}] px')
        else:
            cell_mask = np.zeros_like(data[..., 0])
            settings['cytoplasm'] = False
            settings['uninfected'] = True

        if settings['nucleus_mask_dim'] is not None:
            nucleus_mask = data[..., settings['nucleus_mask_dim']].astype(data_type)
            if settings['cell_mask_dim'] is not None:
                nucleus_mask, cell_mask = _merge_overlapping_objects(mask1=nucleus_mask, mask2=cell_mask)
            nucleus_max = settings.get('nucleus_max_size')
            if ((settings['nucleus_min_size'] is not None
                 and settings['nucleus_min_size'] != 0) or nucleus_max):
                before = int(len(np.unique(nucleus_mask)) - 1)
                nucleus_mask = _filter_object(
                    nucleus_mask, settings['nucleus_min_size'],
                    max_value=nucleus_max)
                dropped = before - int(len(np.unique(nucleus_mask)) - 1)
                if dropped and nucleus_max:
                    print(f'nucleus: {dropped} object(s) outside '
                          f'[{settings["nucleus_min_size"]}, {nucleus_max}] px')
            if settings['timelapse_objects'] == 'nucleus':
                if settings['cell_mask_dim'] is not None:
                    cell_mask, nucleus_mask = _relabel_parent_with_child_labels(cell_mask, nucleus_mask)
                    data[..., settings['cell_mask_dim']] = cell_mask
                    data[..., settings['nucleus_mask_dim']] = nucleus_mask
                    save_folder = settings['src']
                    np.save(os.path.join(save_folder, file), data)
        else:
            nucleus_mask = np.zeros_like(data[..., 0])

        if settings['pathogen_mask_dim'] is not None:
            pathogen_mask = data[..., settings['pathogen_mask_dim']].astype(data_type)
            if settings['merge_edge_pathogen_cells']:
                if settings['cell_mask_dim'] is not None:
                    pathogen_mask, cell_mask = _merge_overlapping_objects(mask1=pathogen_mask, mask2=cell_mask)
            pathogen_max = settings.get('pathogen_max_size')
            if ((settings['pathogen_min_size'] is not None
                 and settings['pathogen_min_size'] != 0) or pathogen_max):
                before = int(len(np.unique(pathogen_mask)) - 1)
                pathogen_mask = _filter_object(
                    pathogen_mask, settings['pathogen_min_size'],
                    max_value=pathogen_max)
                dropped = before - int(len(np.unique(pathogen_mask)) - 1)
                if dropped and pathogen_max:
                    print(f'pathogen: {dropped} object(s) outside '
                          f'[{settings["pathogen_min_size"]}, {pathogen_max}] px')
        else:
            pathogen_mask = np.zeros_like(data[..., 0])

        organelle_masks = {}
        for organelle_role in ORGANELLE_ROLES:
            dim = settings.get(f'{organelle_role}_mask_dim')
            if dim is not None:
                current_mask = data[..., dim].astype(data_type)
                minimum = settings.get(f'{organelle_role}_min_area')
                if minimum:
                    current_mask = _filter_object(current_mask, minimum)
            elif organelle_role == 'organelle':
                current_mask = np.zeros_like(data[..., 0])
            else:
                continue
            organelle_masks[organelle_role] = current_mask
        organelle_mask = organelle_masks['organelle']
        extra_organelle_masks = {
            role: organelle_masks[role] for role in ORGANELLE_ROLES[1:]
            if role in organelle_masks}

        if settings['cytoplasm']:
            if settings['cell_mask_dim'] is not None:
                interior = np.zeros_like(cell_mask, dtype=bool)
                if settings['nucleus_mask_dim'] is not None:
                    interior |= (nucleus_mask != 0)
                if settings['pathogen_mask_dim'] is not None:
                    interior |= (pathogen_mask != 0)
                for organelle_role, current_mask in organelle_masks.items():
                    if settings.get(f'{organelle_role}_mask_dim') is not None:
                        interior |= (current_mask != 0)
                cytoplasm_mask = np.where(interior, 0, cell_mask)
        else:
            cytoplasm_mask = np.zeros_like(cell_mask)

        if settings['cell_min_size'] is not None and settings['cell_min_size'] != 0:
            cell_mask = _filter_object(cell_mask, settings['cell_min_size'])
        
        if settings['nucleus_min_size'] is not None and settings['nucleus_min_size'] != 0:
            nucleus_mask = _filter_object(nucleus_mask, settings['nucleus_min_size'])
        
        if settings['pathogen_min_size'] is not None and settings['pathogen_min_size'] != 0:
            pathogen_mask = _filter_object(pathogen_mask, settings['pathogen_min_size'])
        
        if settings['cytoplasm_min_size'] is not None and settings['cytoplasm_min_size'] != 0:
            cytoplasm_mask = _filter_object(cytoplasm_mask, settings['cytoplasm_min_size'])
        
        for organelle_role, current_mask in organelle_masks.items():
            minimum = settings.get(f'{organelle_role}_min_size')
            if minimum:
                organelle_masks[organelle_role] = _filter_object(
                    current_mask, minimum)
        organelle_mask = organelle_masks['organelle']
        extra_organelle_masks = {
            role: organelle_masks[role] for role in ORGANELLE_ROLES[1:]
            if role in organelle_masks}

        if region_filter_hooks():
            _region_masks = {
                'cell': cell_mask, 'nucleus': nucleus_mask,
                'pathogen': pathogen_mask, **organelle_masks,
                'cytoplasm': cytoplasm_mask,
            }
            for _object_type in list(_region_masks):
                _before = _region_masks[_object_type]
                _kept, _dropped = apply_region_filter_hooks(
                    _before, object_type=_object_type,
                    file_name=file_name, settings=settings, spacing=spacing)
                _region_masks[_object_type] = _kept
                if _dropped and settings['verbose']:
                    _total = int(np.count_nonzero(np.unique(_before)))
                    print(f"{file_name}: region filter dropped "
                          f"{len(_dropped)} of {_total} "
                          f"{_object_type} object(s).")
            cell_mask = _region_masks['cell']
            nucleus_mask = _region_masks['nucleus']
            pathogen_mask = _region_masks['pathogen']
            organelle_masks = {
                role: _region_masks[role] for role in ORGANELLE_ROLES
                if role in _region_masks}
            organelle_mask = organelle_masks['organelle']
            extra_organelle_masks = {
                role: organelle_masks[role]
                for role in ORGANELLE_ROLES[1:] if role in organelle_masks}
            cytoplasm_mask = _region_masks['cytoplasm']

        if settings['cell_mask_dim'] is not None and settings['nucleus_mask_dim'] is not None and settings['pathogen_mask_dim'] is not None:
            cell_mask, nucleus_mask, pathogen_mask, cytoplasm_mask = _exclude_objects(cell_mask, nucleus_mask, pathogen_mask, cytoplasm_mask, uninfected=settings['uninfected'])
            for organelle_role, current_mask in organelle_masks.items():
                organelle_masks[organelle_role] = (
                    current_mask * (cell_mask > 0))
            organelle_mask = organelle_masks['organelle']
            extra_organelle_masks = {
                role: organelle_masks[role]
                for role in ORGANELLE_ROLES[1:] if role in organelle_masks}
            data[..., settings['cell_mask_dim']] = cell_mask.astype(data_type)

        if settings['nucleus_mask_dim'] is not None:
            data[..., settings['nucleus_mask_dim']] = nucleus_mask.astype(data_type)
        if settings['pathogen_mask_dim'] is not None:
            data[..., settings['pathogen_mask_dim']] = pathogen_mask.astype(data_type)
        for organelle_role, current_mask in organelle_masks.items():
            dim = settings.get(f'{organelle_role}_mask_dim')
            if dim is not None:
                data[..., dim] = current_mask.astype(data_type)
        if settings['cytoplasm']:
            data = np.concatenate((data, cytoplasm_mask[..., np.newaxis]), axis=-1)

        if settings['plot'] and not volumetric:
            from .plot import _plot_cropped_arrays
            fig = _plot_cropped_arrays(data, file, figuresize)
            figs[f'{file_name}__after_filtration'] = fig


        if settings['save_measurements']:
            role_order = [
                'cell', 'nucleus', 'pathogen',
                'organelle', *extra_organelle_masks, 'cytoplasm']
            morphology = dict(zip(
                role_order,
                _morphological_measurements(
                    cell_mask, nucleus_mask, pathogen_mask, organelle_mask,
                    cytoplasm_mask, settings,
                    extra_organelle_masks=extra_organelle_masks,
                    channel_arrays=channel_arrays)))
            intensities = dict(zip(
                role_order,
                _intensity_measurements(
                    cell_mask, nucleus_mask, pathogen_mask, organelle_mask,
                    cytoplasm_mask, channel_arrays, settings,
                    sizes=[1, 2, 3, 4, 5], periphery=True, outside=True,
                    extra_organelle_masks=extra_organelle_masks)))

            enabled = {
                'cell': settings['cell_mask_dim'] is not None,
                'nucleus': settings['nucleus_mask_dim'] is not None,
                'pathogen': settings['pathogen_mask_dim'] is not None,
                'cytoplasm': bool(settings['cytoplasm']),
                **{role: settings.get(f'{role}_mask_dim') is not None
                   for role in ORGANELLE_ROLES},
            }
            for role in role_order:
                if enabled[role]:
                    _merge_and_save_to_database(
                        morphology[role], intensities[role], role,
                        source_folder, file_name, settings['experiment'],
                        settings['timelapse'], stamp=units_stamp)

            requested = settings.get('summarize_organelles_by')
            if isinstance(requested, str):
                requested = {requested}
            elif requested is None:
                requested = set()
            else:
                requested = set(requested)
            parent_masks = {
                'cell': cell_mask, 'nucleus': nucleus_mask,
                'pathogen': pathogen_mask, 'cytoplasm': cytoplasm_mask,
            }
            parent_enabled = {
                'cell': settings['cell_mask_dim'] is not None,
                'nucleus': settings['nucleus_mask_dim'] is not None,
                'pathogen': settings['pathogen_mask_dim'] is not None,
                'cytoplasm': bool(settings['cytoplasm']),
            }
            for parent_name, parent_mask in parent_masks.items():
                if parent_name not in requested or not parent_enabled[parent_name]:
                    continue
                summary_frames = []
                for role, current_mask in organelle_masks.items():
                    if not enabled[role]:
                        continue
                    frame = _summarize_organelles_per_parent(
                        current_mask, parent_mask, channel_arrays,
                        parent_name=parent_name, spacing=spacing)
                    if frame.empty:
                        continue
                    frame = frame.rename(columns={
                        column: (
                            f'organelle_summary_{role}_'
                            f'{column[len("organelle_"):]}'
                            if column.startswith('organelle_') else column)
                        for column in frame.columns
                    })
                    summary_frames.append(frame)
                if not summary_frames:
                    continue
                combined = summary_frames[0]
                for frame in summary_frames[1:]:
                    combined = combined.merge(
                        frame, on='label', how='outer',
                        validate='one_to_one')
                _merge_and_save_to_database(
                    combined, pd.DataFrame(),
                    f'{parent_name}_organelle_summary', source_folder,
                    file_name, settings['experiment'],
                    settings['timelapse'], stamp=units_stamp)

        _write_intensity_rescale_record(
            source_folder, file_name, settings, rescale_record, psf_record)

        if settings.get('confluency'):
            confluency_result, confluency_plane = _measure_field_confluency(
                data, settings, channel_arrays, cell_mask=confluency_cells)
            _write_confluency_record(
                source_folder, file_name, settings, confluency_result)
            if settings['verbose']:
                print(f"{file_name}: {confluency_result.confluency:.1%} "
                      f"covered ({confluency_result.source})")
            if settings['plot']:
                figs[f'{file_name}__confluency'] = _confluency_figure(
                    confluency_plane, confluency_result, file_name)

        if volumetric and (settings['save_png'] or settings['save_arrays'] or settings['plot']):
            print(f"3-D field {file_name}: measurements written, but no PNG "
                  f"crops or region arrays. Cropping is 2-D; to get crops from "
                  f"a z-stack, project it first "
                  f"(z_segmentation_mode='project').")
            raise_if_strict(
                f"save_png/save_arrays/plot requested for the 3-D field "
                f"{file_name}, but spaCR crops 2-D fields only. Measurements "
                f"were written; no crops were.", settings=settings)
        elif settings['save_png'] or settings['save_arrays'] or settings['plot']:
            crop_ls = settings['crop_mode']
            if isinstance(crop_ls, str):
                crop_ls = [crop_ls]
            crop_ls = list(crop_ls)

            size_ls = settings['png_size']
            if not size_ls:
                raise ValueError(
                    "Setting: png_size is empty; give it [width, height], or "
                    "a [width, height] pair per crop_mode entry.")
            if not isinstance(size_ls[0], (list, tuple)):
                size_ls = [size_ls]

            size_ls = _per_crop_mode(size_ls, len(crop_ls), 'png_size')
            dialate_pngs = _per_crop_mode(
                settings['dialate_pngs'], len(crop_ls), 'dialate_pngs')
            dialate_png_ratios = _per_crop_mode(
                settings['dialate_png_ratios'], len(crop_ls),
                'dialate_png_ratios')

            for crop_idx, crop_mode in enumerate(crop_ls):
                if crop_mode not in CROP_MODES:
                    print(f"Setting: crop_mode entry {crop_mode!r} is not "
                          f"one of {', '.join(CROP_MODES)}; skipping it. "
                          f"No {crop_mode}_png crops were written.")
                    continue

                width, height = size_ls[crop_idx]

                crop_masks = {
                    'cell': cell_mask,
                    'nucleus': nucleus_mask,
                    'pathogen': pathogen_mask,
                    **organelle_masks,
                    'cytoplasm': cytoplasm_mask,
                }
                crop_mask = crop_masks[crop_mode].copy()
                dialate_png = dialate_pngs[crop_idx]
                dialate_png_ratio = dialate_png_ratios[crop_idx]
                if crop_mode == 'cytoplasm':
                    crop_mask = cytoplasm_mask.copy()
                    dialate_png = False
                    dialate_png_ratio = dialate_png_ratios[crop_idx]

                objects_in_image = np.unique(crop_mask)
                objects_in_image = objects_in_image[objects_in_image != 0]
                img_paths = []
                
                for _id in objects_in_image:
                    
                    region = (crop_mask == _id)

                    region_cell_ids = np.atleast_1d(np.unique(cell_mask[region]))
                    region_nucleus_ids = np.atleast_1d(np.unique(nucleus_mask[region]))
                    region_pathogen_ids = np.atleast_1d(np.unique(pathogen_mask[region]))

                    if settings['use_bounding_box']:
                        region = _find_bounding_box(crop_mask, _id, buffer=10)

                    img_name, fldr, table_name = _generate_names(
                        file_name=file_name, cell_id=region_cell_ids,
                        cell_nucleus_ids=region_nucleus_ids,
                        cell_pathogen_ids=region_pathogen_ids,
                        source_folder=source_folder, crop_mode=crop_mode,
                        timelapse=settings['timelapse'], object_id=_id)

                    if dialate_png:
                        region_area = np.count_nonzero(region)
                        approximate_diameter = np.sqrt(region_area)
                        dialate_png_px = int(approximate_diameter * dialate_png_ratio)
                        if dialate_png_px > 0:
                            struct = generate_binary_structure(region.ndim, region.ndim)
                            region = binary_dilation(region, structure=struct, iterations=dialate_png_px)

                    if settings['save_png']:
                        fldr_type = f"{crop_mode}_png/"
                        png_folder = os.path.join(fldr,fldr_type)
                        img_path = os.path.join(png_folder, img_name)
                        img_paths.append(img_path)

                        png_channels = build_png_channels(
                            data, resolve_png_channel_mapping(settings),
                            dtype=data_type)

                        if settings['normalize_by'] == 'fov':
                            if not settings['normalize'] is False:
                                percentile_list = _get_percentiles(png_channels, settings['normalize'][0], settings['normalize'][1])

                        png_channels = _crop_center(png_channels, region, new_width=width, new_height=height)
                        if isinstance(settings['normalize'], list):
                            if settings['normalize_by'] == 'png':
                                png_channels = normalize_to_dtype(png_channels, settings['normalize'][0], settings['normalize'][1])

                            if settings['normalize_by'] == 'fov':
                                png_channels = normalize_to_dtype(png_channels, settings['normalize'][0], settings['normalize'][1], percentile_list=percentile_list)
                        else:
                            png_channels = normalize_to_dtype(png_channels, 0, 100)
                        os.makedirs(png_folder, exist_ok=True)

                        grid = save_and_add_image_to_grid(
                            png_channels, img_path, grid, settings['plot'])

                        if len(img_paths) == len(objects_in_image):
                            filepaths_to_database(img_paths, settings, source_folder, crop_mode)

                    if settings['save_arrays']:
                        row_idx, col_idx = np.where(region)
                        region_array = data[row_idx.min():row_idx.max()+1, col_idx.min():col_idx.max()+1, :]
                        array_folder = f"{fldr}/region_array/"            
                        os.makedirs(array_folder, exist_ok=True)
                        from .normalization import apply_crop_dtype
                        np.save(os.path.join(array_folder, img_name),
                                apply_crop_dtype(region_array,
                                                 settings.get('crop_dtype',
                                                              'original')))


        cells = np.unique(cell_mask)
        error_text = ""
    except PipelineCancelled:
        raise
    except Exception as e:
        cells = 0
        error_text = "".join(
            traceback.format_exception(type(e), e, e.__traceback__))
        print(f"[measure] {os.path.basename(str(file))} failed:\n{error_text}")

    end = time.time()
    duration = end-start
    time_ls.append(duration)
    average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
    if settings['plot'] and grid:
        fig = img_list_to_grid(grid)
        figs[f'{file_name}__pngs'] = fig
    return index, average_time, cells, figs, error_text

def _record_organelle_caveats(settings, run):
    """Put the per-type organelle caveats on the run journal.

    :param settings: the measure settings for this source folder, read for
        each slot's ``*_type`` and the count-dependent families it enabled.
    :param run: the :class:`spacr.runctx.RunContext` the tables are written
        under. Its logger stamps every record with the run id, so
        :func:`spacr.runctx.read_run_log` gives the caveats back beside the
        database they are about.
    :returns: the caveats recorded, so a caller can see what was said.

    NOTHING IS SWITCHED OFF: a family the organelle type makes doubtful is
    still measured and still written, because a number that vanished without
    being asked to is worse than one that comes with a caveat. What the type
    buys is that the run SAYS SO -- and saying it only to the console leaves
    the sentence out of the one record a batch is read back from.

    Silent when there is nothing to say, so a run measuring punctate
    organelles is not given a paragraph telling it everything is fine.
    """
    from .settings import organelle_measurement_caveats

    caveats = organelle_measurement_caveats(settings)
    for label, setting, reason in caveats:
        run.log.warning("[organelle] %s: %s %s.", label, setting, reason)
    return caveats


def _wait_for_measure_job(result, psf_cancel=None):
    """Relay Stop, allowing five seconds for a worker's current PSF operation.

    A worker that never answers cannot keep Stop waiting forever. After the
    grace period, pipeline cancellation exits the owning pool context, which
    terminates outstanding workers and leaves incomplete fields resumable.
    """
    if psf_cancel is None:
        return result.get()
    from .cancellation import cancellation_requested, checkpoint
    cancelled_at = None
    while True:
        try:
            return result.get(timeout=0.2)
        except mp.TimeoutError:
            if cancellation_requested():
                psf_cancel.set()
                if cancelled_at is None:
                    cancelled_at = time.monotonic()
                elif time.monotonic() - cancelled_at >= 5:
                    checkpoint()


def measure_crop(settings):
    """Extract per-object morphology/intensity measurements and (optionally) cropped PNGs from mask stacks.

    Consumes the ``merged/`` folder produced by
    :func:`spacr.core.preprocess_generate_masks` (channel arrays + mask stacks
    saved as ``.npy``), computes shape, intensity, texture and spatial
    features per cell / nucleus / pathogen / cytoplasm object, and writes
    them to a SQLite ``measurements.db``. When ``save_png`` is enabled it
    also crops per-object PNG thumbnails, which are the training input for
    :func:`spacr.deep_spacr.deep_spacr`.

    :param settings: Settings dict, canonicalized via
        :func:`spacr.settings.get_measure_crop_settings`. Key entries the
        function reads:

        - ``src`` (str or list) — one or more ``…/merged`` folders.
        - ``psf_measurement_source`` — original (default) uses the normal
          rescaled/preprocessed intensities; processed adds an explicitly
          calibrated PSF before quantitative features. The immutable kernel
          reaches every worker. Source images and exported crops stay unchanged.
          Field provenance is saved in ``intensity_rescale``; incompatible
          existing PSF measurements are refused before any rows are appended.
        - ``cell_mask_dim`` / ``nucleus_mask_dim`` / ``pathogen_mask_dim``
          — channel index of each mask stack; ``None`` disables that
          object type.
        - ``cell_min_size`` / ``nucleus_min_size`` / ``pathogen_min_size``
          / ``cytoplasm_min_size`` — pixel-area cutoffs.
        - ``channels`` — list of intensity channels to measure.
        - ``crop_mode`` — list drawn from ``['cell','nucleus','pathogen',
          'cytoplasm']``; each entry produces one PNG per object.
        - ``save_png`` — write per-object PNG thumbnails.
        - ``normalize`` — ``[lower_pct, upper_pct]`` for PNG normalization.
        - ``normalize_by`` — ``'png'`` (per-crop) or ``'fov'`` (per-field).
        - ``timelapse``, ``timelapse_objects``, ``n_jobs``, ``test_mode``.
        - ``dry_run`` — validate the settings, report the plan and stop;
          the input folders are inspected but nothing is written.

    :returns: ``None`` on a normal run, which writes
        ``measurements/measurements.db``, ``measure_crop_settings.csv``, and
        (if ``save_png``) PNGs into per-object subfolders under ``src``. When
        ``dry_run`` is set, the list of :class:`spacr.validate.Problem`
        returned by :func:`spacr.validate.run_preflight`, and nothing is
        written.
    :raises ValueError: if ``src`` is not a string or a list of strings.
    :raises spacr.errors.ConfigurationError: only in strict mode
        (``settings['strict_errors']``, or the ``SPACR_STRICT_ERRORS``
        environment variable). The ``normalize``, ``normalize_by``,
        mask-dimension/min-size and ``channels`` type checks otherwise print
        a WARNING and return ``None`` without measuring anything.

    Example:
        .. code-block:: python

            from spacr.measure import measure_crop
            settings = {
                'src': '/data/plate01/merged',
                'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': 6,
                'channels': [0, 1, 2, 3],
                'crop_mode': ['cell'], 'save_png': True,
                'normalize': [1, 99], 'normalize_by': 'png',
            }
            measure_crop(settings)

    See Also:
        :func:`spacr.core.preprocess_generate_masks` — upstream mask generation.
        :func:`spacr.io.generate_dataset` — build a training set from the PNGs.
        :func:`spacr.deep_spacr.deep_spacr` — train a CNN on the crops.
    """
    if settings.get('dry_run', False):
        from .validate import run_preflight
        return run_preflight(settings, 'measure')

    from .io import _save_settings_to_db, _listdir_visible
    from .cancellation import (
        PipelineCancelled,
        checkpoint as cancellation_checkpoint,
    )
    from .timelapse import _timelapse_masks_to_gif
    from .utils import measure_test_mode, print_progress, save_settings, format_path_for_system, normalize_src_path
    from .settings import get_measure_crop_settings
    
    
    
    if settings['timelapse']:
        settings['save_png'] = False

    if not isinstance(settings['src'], (str, list)):
        raise ValueError('src must be a string or a list of strings')
    
    settings = dict(settings)
    settings['src'] = normalize_src_path(settings['src'])
    
    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]

    if isinstance(settings['src'], list):
        source_folders = list(settings['src'])
        base_settings = dict(settings)
        
        with run_context('measure', settings) as run:
            for source_folder in source_folders:
                cancellation_checkpoint()
                print(f'Processing folder: {source_folder}')
                settings = dict(base_settings)
                source_folder = format_path_for_system(source_folder)
                settings['src'] = source_folder

                src_fldr = settings['src']
            
                if not os.path.basename(src_fldr).endswith('merged'):
                    print(f"WARNING: Source folder, settings: src: {src_fldr} should end with '/merged'")
                    src_fldr = os.path.join(src_fldr, 'merged')
                    settings['src'] = src_fldr
                    print(f"Changed source folder to: {src_fldr}")

                explicit_mask_keys = {
                    f'{role}_mask_dim' for role in SEGMENTED_ROLES
                    if f'{role}_mask_dim' in settings}
                settings = reconcile_merged_mask_dims(
                    settings, src_fldr, explicit_keys=explicit_mask_keys)
                settings = get_measure_crop_settings(settings)
                settings = measure_test_mode(settings)
                if settings.get('confluency'):
                    _resolve_confluency_source(settings)

                from .database_concurrency import enable_wal_where_safe
                _measurements_dir = os.path.join(
                    os.path.dirname(src_fldr), 'measurements')
                os.makedirs(_measurements_dir, exist_ok=True)
                enable_wal_where_safe(
                    os.path.join(_measurements_dir, 'measurements.db'))

                from .illumination import (
                    prepare_illumination_correction,
                    validate_measurement_illumination_inputs,
                )
                from .psf_measurement import (
                    prepare_measurement_psf, measurement_psf_signature,
                    validate_measurement_psf_history, SIGNATURE_KEY)
                psf_plan = prepare_measurement_psf(settings)
                settings[SIGNATURE_KEY] = measurement_psf_signature(psf_plan)
                validate_measurement_psf_history(
                    settings, os.path.join(_measurements_dir, 'measurements.db'), psf_plan)
                validate_measurement_illumination_inputs(settings)
                prepare_illumination_correction(settings)

                if settings['cell_mask_dim'] is None:
                    settings['uninfected'] = True
                if settings['pathogen_mask_dim'] is None:
                    settings['uninfected'] = True
                if settings['cell_mask_dim'] is not None and settings['pathogen_min_size'] is not None:
                    settings['cytoplasm'] = True
                elif settings['cell_mask_dim'] is not None and settings['nucleus_min_size'] is not None:
                    settings['cytoplasm'] = True
                else:
                    settings['cytoplasm'] = False
                
                settings['n_jobs'] = resolve_n_jobs(settings['n_jobs'])

                settings_save = settings.copy()
                settings_save['src'] = os.path.dirname(settings['src'])
                save_settings(settings_save, name='measure_crop_settings', show=True)

                if settings['timelapse_objects'] == 'nucleus':
                    if not settings['cell_mask_dim'] is None:
                        tlo = settings['timelapse_objects']
                        print(f'timelapse object:{tlo}, cells will be relabeled to nucleus labels to track cells.')

                int_setting_keys = [
                    *(f'{role}_mask_dim' for role in SEGMENTED_ROLES),
                    *(f'{role}_min_size' for role in SEGMENTED_ROLES),
                    'cytoplasm_min_size',
                ]
            
                if isinstance(settings['normalize'], bool) and settings['normalize']:
                    print(f'WARNING: to notmalize single object pngs set normalize to a list of 2 integers, e.g. [1,99] (lower and upper percentiles)')
                    raise_if_strict(
                        "settings['normalize'] must be a list of two percentiles, "
                        "e.g. [1, 99] — not a bool. Nothing was measured.",
                        settings=settings)
                    return

                if isinstance(settings['normalize'], list) or isinstance(settings['normalize'], bool) and settings['normalize']:
                    if settings['normalize_by'] not in ['png', 'fov']:
                        print("Warning: normalize_by should be either 'png' to notmalize each png to its own percentiles or 'fov' to normalize each png to the fov percentiles ")
                        raise_if_strict(
                            "settings['normalize_by'] must be 'png' or 'fov', got "
                            f"{settings['normalize_by']!r}. Nothing was measured.",
                            settings=settings)
                        return

                if not all(isinstance(settings.get(key), int)
                           or settings.get(key) is None
                           for key in int_setting_keys):
                    print(f"WARNING: {int_setting_keys} must all be integers")
                    raise_if_strict(
                        f"{int_setting_keys} must all be int or None. "
                        "Nothing was measured.", settings=settings)
                    return

                if not isinstance(settings['channels'], list):
                    print(f"WARNING: channels should be a list of integers representing channels e.g. [0,1,2,3]")
                    raise_if_strict(
                        "settings['channels'] must be a list of channel indices, "
                        f"got {type(settings['channels']).__name__}. "
                        "Nothing was measured.", settings=settings)
                    return

                if not isinstance(settings['crop_mode'], list):
                    print(f"WARNING: crop_mode should be a list with at least one element e.g. ['cell'] or ['cell','nucleus'] or [None] got: {settings['crop_mode']}")
                    settings['crop_mode'] = [settings['crop_mode']]
                    settings['crop_mode'] = [str(crop_mode) for crop_mode in settings['crop_mode']]
                    print(f"Converted crop_mode to list: {settings['crop_mode']}")
            
                resume_plan = plan_measure_resume(settings)

                _save_settings_to_db(settings)

                files = [f for f in _listdir_visible(settings['src']) if f.endswith('.npy')]
                from .image_quality import excluded_fields, ensure_no_retained_measurements
                rejected_quality = excluded_fields(os.path.dirname(settings['src']))
                ensure_no_retained_measurements(os.path.dirname(settings['src']), rejected_quality)
                files = [name for name in files if name not in rejected_quality]
                _full_rescale_plan = build_plate_plan(
                    settings['src'], files, settings)
                settings[PLAN_SETTINGS_KEY] = {
                    'version': _full_rescale_plan['version'],
                    'plates': _full_rescale_plan['plates'],
                    'failures': _full_rescale_plan['failures'],
                }
                for failed_file, reason in sorted(
                        settings[PLAN_SETTINGS_KEY]['failures'].items()):
                    print(
                        f"WARNING: could not pre-scan {failed_file} for a "
                        f"plate-wide intensity scale ({reason}). If the field "
                        f"can be loaded by its worker, it will use a per-field "
                        f"fallback and measurements.db:intensity_rescale will "
                        f"mark it non-comparable.")
                if resume_plan is not None:
                    files = resume_plan.filter_files(files)
                n_jobs = settings['n_jobs']
                print(f'using {n_jobs} cpu cores')
                print_progress(files_processed=0, files_to_process=len(files), n_jobs=n_jobs, time_ls=[], operation_type='Measure and Crop')

                ledger = RunLedger('measure_crop')
                run.adopt(ledger)
                _record_organelle_caveats(settings, run)
                policy = run.policy.bind(ledger=ledger, record=False)
                index_to_file = dict(enumerate(files))
                reported_files = set()

                def job_callback(result):
                    """Record one completed field and save its optional figures.

                    :param result: The 4-tuple ``(index, average_time, cells,
                        figs)`` that :func:`_measure_crop_core` returns, taken
                        straight off the ``AsyncResult`` -- one result, not the
                        list that :func:`process_measure_crop_results` takes,
                        which is why it is re-wrapped as ``[result]`` below.
                        ``index`` is the position in ``files`` and is translated
                        back through ``index_to_file`` so the ledger entry names
                        the field rather than a number. ``cells`` decides the
                        verdict: the success path leaves the
                        ``np.unique(cell_mask)`` array there, while a plain int
                        ``0`` is the cross-process failure sentinel a worker
                        leaves when it caught its own exception, so only that
                        int records a failure. ``figs`` may be an empty dict --
                        nothing is drawn unless ``settings['plot']``. Passing
                        the same field twice is safe for the counters
                        (``completed_jobs`` and ``reported_files`` are sets) but
                        would save its figures twice, so the retry loop calls
                        this only for the attempt that actually returned.
                    """
                    completed_jobs.add(result[0])
                    item = index_to_file.get(result[0], result[0])
                    reported_files.add(item)
                    if isinstance(result[2], int) and result[2] == 0:
                        detail = (result[4] if len(result) > 4 else "") or (
                            'field failed inside _measure_crop_core, and the '
                            'worker returned no traceback')
                        ledger.record_failure(
                            item, stage='measure', exc=detail)
                    else:
                        ledger.record_success(item, stage='measure')
                    process_measure_crop_results([result], settings)
                    files_processed = len(completed_jobs)
                    files_to_process = len(files)
                    print_progress(files_processed, files_to_process, n_jobs, time_ls=time_ls, operation_type='Measure and Crop')

                def make_error_callback(job_file):
                    """Bind the filename into the pool's error callback.

                    ``apply_async`` hands the error callback only the exception,
                    so the file has to be closed over. Without this hook a worker
                    that died outright vanished entirely: the exception sat on an
                    AsyncResult nobody read, and the run still printed
                    "Successfully completed run".

                    :param job_file: The ``.npy`` filename of the field, as it
                        appears in ``files`` -- a bare basename, not a path
                        joined onto ``settings['src']``. It is used unchanged as
                        the ledger key *and* as the ``reported_files`` entry, so
                        anything else silently loses the match against ``files``
                        in the ``finally`` sweep and the field is filed a second
                        time as "field produced no result".
                    :returns: A one-argument callable suitable as the
                        ``error_callback`` of ``Pool.apply_async``; it takes the
                        exception and returns ``None``. Call it as
                        ``make_error_callback(file)(exc)`` when raising the
                        exception yourself, which is what the retry loop does on
                        the last attempt -- the ledger counts fields, not tries,
                        so a field that failed twice and then worked must not be
                        reported here at all.
                    """
                    def _on_error(exc):
                        """Record one worker's failure against the file that caused it."""
                        reported_files.add(job_file)
                        ledger.record_failure(job_file, stage='measure_worker', exc=exc)
                    return _on_error

                ctx = _pool_context()
                start_method = ctx.get_start_method()
                warn_if_hooks_will_not_reach_workers(start_method)
                pool_jobs = resolve_pool_size(n_jobs, len(files),
                                              start_method=start_method)

                try:
                    with _start_manager(ctx) as manager:
                        time_ls = manager.list()
                        psf_cancel = manager.Event() if psf_plan is not None else None
                        completed_jobs = set()

                        with ctx.Pool(pool_jobs) as pool:
                            for offset in range(0, len(files), pool_jobs):
                                cancellation_checkpoint()
                                pending = []
                                for index in range(
                                        offset, min(offset + pool_jobs, len(files))):
                                    file = files[index]
                                    result = pool.apply_async(
                                        _measure_crop_core,
                                        args=((index, time_ls, file, settings, psf_plan, psf_cancel)
                                              if psf_plan is not None else
                                              (index, time_ls, file, settings)),
                                    )
                                    pending.append((file, index, result))
                                for file, index, async_result in pending:
                                    for attempt in policy.attempts_for(
                                            file, stage='measure'):
                                        with attempt:
                                            try:
                                                if attempt.number == 1:
                                                    job_callback(_wait_for_measure_job(
                                                        async_result, psf_cancel))
                                                else:
                                                    retried = pool.apply_async(
                                                        _measure_crop_core,
                                                        args=((index, time_ls, file, settings, psf_plan, psf_cancel)
                                                              if psf_plan is not None else
                                                              (index, time_ls, file, settings)))
                                                    job_callback(_wait_for_measure_job(retried, psf_cancel))
                                            except PipelineCancelled:
                                                raise
                                            except Exception as exc:
                                                if attempt.last:
                                                    make_error_callback(file)(exc)
                                                raise
                                cancellation_checkpoint()

                            pool.close()
                            pool.join()
                finally:
                    for job_file in files:
                        if job_file not in reported_files:
                            ledger.record_failure(job_file, stage='measure',
                                                  exc='field produced no result')

                    db_path = os.path.join(os.path.dirname(settings['src']),
                                           'measurements', 'measurements.db')
                    ledger.finalize(
                        artifact=db_path if os.path.isfile(db_path) else None)

                if settings.get('confluency') and os.path.isfile(db_path):
                    wells = _aggregate_confluency_by_well(
                        db_path, settings.get('confluency_qc_threshold'))
                    if not wells.empty:
                        failing = int((wells['monolayer_ok'] == 0).sum())
                        print(f"Confluency: {len(wells)} well(s) in "
                              f"measurements.db:{_CONFLUENCY_WELL_TABLE}, "
                              f"{failing} below the monolayer QC threshold.")

                if settings['timelapse']:
                    if settings['timelapse_objects'] == 'nucleus':
                        folder_path = settings['src']
                        mask_channels = [settings['nucleus_mask_dim'], settings['pathogen_mask_dim'], settings['cell_mask_dim']]
                        object_types = ['nucleus', 'pathogen', 'cell']
                        _timelapse_masks_to_gif(folder_path, mask_channels, object_types)

                if ledger.is_complete:
                    _emit_infection_report(db_path)
                    print("Successfully completed run")

            run.register_outputs(settings=settings, roots=source_folders)

def _emit_infection_report(db_path):
    """Write the infection report a finished run can support, if any.

    Measure emits the report rather than a button producing it, so this
    runs at the end of every complete run and says where it went.

    A REPORT IS NOT WORTH A RUN. Everything here is inside a try: a plate
    whose tables the metrics cannot read, a disk that refuses the file, or
    a pandas that objects to something must not turn a measure run that
    has already written its database into a failure.

    :param db_path: the ``measurements.db`` the run produced.
    """
    if not db_path or not os.path.isfile(db_path):
        return
    try:
        from .infection import write_infection_report

        written = write_infection_report(db_path)
    except Exception as exc:                                 # noqa: BLE001
        print(f"The infection report could not be written: {exc}")
        return
    if written:
        print(f"Infection report: {written}")


def process_measure_crop_results(partial_results, settings):
    """Save and display figures carried by completed Measure jobs.

    :param partial_results: Completed job tuples. ``None`` entries are skipped;
        each figure is written below ``<src>/../results/`` and then closed.
    :param settings: Resolved Measure settings. ``src`` identifies the output
        root.
    """
    for result in partial_results:
        if result is None:
            continue
        index, avg_time, cells, figs = result[:4]
        if figs is not None:
            for key, fig in figs.items():
                part_1, part_2 = key.split('__')
                save_dir = os.path.join(os.path.dirname(settings['src']), 'results', f"{part_1}")
                os.makedirs(save_dir, exist_ok=True)
                fig_path = os.path.join(save_dir, f"{part_2}.pdf")
                from .plot import save_figure
                fig_path = save_figure(fig, fig_path)
                with figure_style(theme_target()):
                    plt.figure(fig.number)
                    plt.show()
                    plt.close(fig)
            result = (index, None, None, None)


def process_meassure_crop_results(partial_results, settings):
    """Deprecated alias for :func:`process_measure_crop_results`.

    The misspelled name remains available for existing scripts and will be
    removed in a future major release.

    :param partial_results: Completed Measure job tuples, passed unchanged to
        :func:`process_measure_crop_results` after a ``DeprecationWarning``.
    :param settings: Resolved Measure settings, passed unchanged; ``src``
        identifies the output root.
    """
    import warnings
    warnings.warn(
        "process_meassure_crop_results is deprecated; use "
        "process_measure_crop_results",
        DeprecationWarning,
        stacklevel=2,
    )
    return process_measure_crop_results(partial_results, settings)


def generate_cellpose_train_set(folders, dst, min_objects=5):
    """Copy image/mask pairs from source folders into a Cellpose training set.

    Only pairs whose mask contains at least ``min_objects`` labeled objects
    (background label 0 excluded) are copied. Files are renamed with their
    source folder name as prefix to avoid collisions.

    :param folders: Iterable of source folders, each containing a ``masks/``
        subfolder and the raw images alongside it.
    :param dst: Destination folder; ``imgs/`` and ``masks/`` subfolders are
        created if missing.
    :param min_objects: Minimum number of unique object labels required in a
        mask for the pair to be included. Default ``5``.
    :returns: The finalized :class:`spacr.errors.RunLedger`. Unreadable masks
        and failed copies are recorded on it and summarised loudly at the end,
        so a training set that is quietly short of pairs announces itself.
    """
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(dst,'masks'), exist_ok=True)
    os.makedirs(os.path.join(dst,'imgs'), exist_ok=True)

    from .io import _listdir_visible

    ledger = RunLedger('generate_cellpose_train_set')
    for folder in folders:
        mask_folder = os.path.join(folder, 'masks')
        experiment_id = os.path.basename(folder)
        for filename in _listdir_visible(mask_folder):
            path = os.path.join(mask_folder, filename)
            img_path = os.path.join(folder, filename)
            newname = experiment_id + '_' + filename
            new_mask = os.path.join(dst, 'masks', newname)
            new_img = os.path.join(dst, 'imgs', newname)

            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                ledger.record_failure(path, stage='read_mask',
                                      exc='cv2.imread returned None')
                print(f"Error reading {path}, skipping.")
                continue

            nr_of_objects = len(np.unique(mask)) - 1
            if nr_of_objects >= min_objects:
                with ledger.item(path, stage='copy_pair',
                                 echo=f"Error copying {path} to {new_mask}"):
                    shutil.copy(path, new_mask)
                    shutil.copy(img_path, new_img)

    ledger.finalize()
    return ledger

def get_object_counts(src):
    """Return per-count-type totals and per-file averages from the measurements DB.

    Reads the ``object_counts`` table from ``<src>/measurements/measurements.db``
    and aggregates by ``count_type``.

    :param src: Path to the run folder containing ``measurements/measurements.db``.
    :returns: DataFrame with columns ``count_type``, ``total_object_count``, and
        ``avg_object_count_per_file_name``.
    """
    database_path = os.path.join(src, 'measurements/measurements.db')
    conn = sqlite3.connect(database_path, timeout=30)
    df = pd.read_sql_query("SELECT * FROM object_counts", conn)
    grouped_df = df.groupby('count_type').agg(
        total_object_count=('object_count', 'sum'),
        avg_object_count_per_file_name=('object_count', 'mean')
    ).reset_index()
    conn.close()
    return grouped_df






def _crop_full_scale(dtype):
    """Return the value that means "full brightness" for ``dtype``.

    An integer dtype has one: ``iinfo(dtype).max`` -- the same range
    :func:`spacr.utils.normalize_to_dtype` stretches the pipeline's own crops
    into, so a normalised crop from here and one from ``measure_crop`` are on
    the same scale. A float array is taken on the ``[0, 1]`` image convention
    (what ``spacr.io._normalize_img_batch`` writes).
    """
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def _normalize_crop(crop, percentiles, mask_background):
    """Per-channel percentile stretch that KEEPS ``crop``'s dtype.

    The stretch targets the dtype's full range (:func:`_crop_full_scale`), not
    a hard-coded 0-255: normalising a ``uint16`` crop into 0-255 and storing it
    back as ``uint16`` throws away 8 of the 16 bits before anything has asked
    for an 8-bit image.

    :param crop: ``(H, W, C)`` array in the working dtype.
    :param percentiles: ``(low, high)`` percentiles, per channel.
    :param mask_background: when True the background is already zeroed, so the
        percentiles are taken over the object's pixels only.
    :returns: array of the same shape and dtype.
    """
    arr = np.asarray(crop)
    top = _crop_full_scale(arr.dtype)
    out = np.zeros(arr.shape, dtype=np.float64)
    for c in range(arr.shape[2]):
        sl = arr[:, :, c].astype(np.float64)
        nz = sl[sl > 0] if mask_background else sl
        if nz.size:
            lo, hi = np.percentile(nz, percentiles)
            if hi > lo:
                out[:, :, c] = np.clip((sl - lo) / (hi - lo), 0, 1) * top
                continue
        mx = sl.max()
        out[:, :, c] = (sl / mx * top) if mx > 0 else sl
    if np.issubdtype(arr.dtype, np.integer):
        return np.rint(out).astype(arr.dtype)
    return out.astype(arr.dtype)


def _crop_to_uint8(crop):
    """The declared 8-bit boundary for an object crop. Rescales, never truncates.

    One rule per dtype, and every one of them is linear and maps 0 to 0, so
    background stays background and relative intensity survives:

    * ``uint8`` -- already 8-bit, returned unchanged.
    * any wider integer -- :func:`spacr.crops.narrow_to_uint8`, i.e. the HIGH
      BYTE of the 16-bit range. This is the narrowing rule the rest of spaCR
      uses for crop PNGs, so a crop from here and one read back by
      ``spacr.crops.read_crop_png`` agree. Raw 16-bit data comes out dark,
      which is what raw 16-bit data looks like at 8 bits.
    * float (and anything else) -- a float array carries no dtype range, so the
      scale comes from the crop itself: ``0 .. max`` maps to ``0 .. 255``. A
      normalised crop is already on ``[0, 1]`` (:func:`_crop_full_scale`) and
      therefore simply multiplied by 255.

    :param crop: ``(H, W, C)`` (or 2-D) array in the working dtype.
    :returns: ``uint8`` array of the same shape.
    """
    arr = np.asarray(crop)
    if arr.dtype == np.dtype(np.uint8):
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return narrow_to_uint8(arr)
    if arr.size == 0:
        return arr.astype(np.uint8)
    mx = float(np.nanmax(arr))
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip(np.nan_to_num(arr, nan=0.0), 0, None) / mx * 255.0
    return np.rint(scaled).astype(np.uint8)


def _resolve_merged_path(path_name, merged_dir):
    """Return the merged ``.npy`` a measurement row names, or ``None``.

    ``spacr.utils._merge_and_save_to_database`` records ``path_name`` as
    ``os.path.join(source_folder, file_name + '.npy')``, and ``source_folder``
    in :func:`_measure_crop_core` is ``os.path.dirname(settings['src'])`` --
    the *parent* of ``merged/``. So on every database spaCR has written, the
    recorded path is ``<root>/<field>.npy`` while the array is at
    ``<root>/merged/<field>.npy``: ``os.path.isfile(path_name)`` is False for
    every row, and :func:`generate_object_dataset` skipped every object of
    every real run while the hand-built databases in the tests (which record
    the path the file is actually at) all passed.

    Resolving on read rather than changing the writer is deliberate: it also
    covers a database moved between machines, and it is the exact fallback
    :meth:`spacr.crops.MergedCropSource._merged_path_for` already uses --
    trust the recorded path when it exists, otherwise look for its basename in
    this experiment's ``merged/`` folder.

    :param path_name: the ``path_name`` column of a measurement row.
    :param merged_dir: this experiment's ``merged/`` folder.
    :returns: an existing path, or ``None`` when neither candidate exists.
    """
    if not path_name:
        return None
    path_name = str(path_name)
    if os.path.isfile(path_name):
        return path_name
    candidate = os.path.join(merged_dir, os.path.basename(path_name))
    return candidate if os.path.isfile(candidate) else None


def _crop_channels(data, y0, y1, x0, x1, channels, region=None):
    """Cut ``channels`` out of ``data[y0:y1, x0:x1]`` **without changing dtype**.

    ``region`` (a boolean object mask over the same window) zeroes the
    background. The old code cast to ``float32`` here and never came back,
    which is what made the 8-bit clip downstream invisible.
    """
    crop = np.asarray(data)[y0:y1, x0:x1, :][:, :, list(channels)]
    if region is None:
        return np.ascontiguousarray(crop)
    return np.where(region[:, :, None], crop, 0).astype(crop.dtype, copy=False)


def generate_object_dataset(
    src,
    object_type='cell',
    channels=(0, 1, 2),
    min_area=None,
    max_area=None,
    columns=None,
    rows=None,
    fields=None,
    plates=None,
    where=None,
    criteria=None,
    output_dir=None,
    png_size=(128, 128),
    mask_background=True,
    normalize=True,
    percentiles=(1, 99),
    buffer=10,
    mask_dims=None,
    save_png=True,
    return_arrays=False,
    limit=None,
    db_path=None,
    verbose=True,
):
    """Build an image dataset by cropping individual objects out of the merged
    image+mask arrays, selected by measurement and/or metadata criteria.

    spaCR's ``merged/`` arrays store the image channels first and then one
    integer label-mask slice per object class (cell, nucleus, pathogen,
    organelle). The measurements database records, for every object, its
    integer ``object_label``, the merged ``.npy`` it came from (``path_name``),
    its well/field metadata, and its features (e.g. ``cell_area``). This
    function queries that database for the objects you want, then for each hit
    slices the object out of its array using ``object_label`` and the class
    mask, assembles the channels you ask for into an image, and saves a PNG.

    Example — an RGB dataset from image channels 0, 2, 4 for cells larger than
    10000 px² in columns 1 and 2::

        generate_object_dataset(
            "/data/plate1", object_type="cell",
            channels=(0, 2, 4), min_area=10000, columns=[1, 2])

    :param src: experiment root (the folder that holds ``merged/`` and
        ``measurements/measurements.db``), or the ``merged`` folder itself.
    :param object_type: which object table + mask slice to crop. With the
        default ``mask_dims`` the accepted values are ``'cell'``,
        ``'nucleus'``, ``'pathogen'`` and ``'organelle'``; any other value
        (``'cytoplasm'`` included) raises ``ValueError`` unless ``mask_dims``
        names its slice explicitly.
    :param channels: image channel indices to include, in output order. Three
        indices → an RGB image; one → greyscale; two → padded to RGB; more than
        three → kept as an ``.npy`` array (and the first three saved as a PNG
        preview when ``save_png``).
    :param min_area: keep only objects with ``{object_type}_area`` > this.
        In a database measured from 2-D fields that column is a px^2 area; in
        one measured from 3-D volumes it is a volume, in voxels or um^3
        according to the row's ``measurement_units``. This function crops 2-D
        arrays only and refuses a volumetric one, so in practice the threshold
        is always px^2 here -- but read the stamp before carrying a number
        between databases.
    :param max_area: keep only objects with ``{object_type}_area`` < this.
    :param columns: list of plate column numbers to include (matched against
        ``columnID`` as ``'c<N>'``). ``rows`` / ``fields`` / ``plates`` behave
        the same for ``rowID`` (``'r<N>'``) / ``fieldID`` (``'f<N>'``) /
        ``plateID`` (raw token).
    :param where: raw SQL boolean fragment ANDed onto the query, for anything
        the shortcuts don't cover (e.g. ``"cell_eccentricity < 0.8"``).
    :param criteria: dict of ``{column: (op, value)}`` ANDed onto the query,
        e.g. ``{"cell_area": (">", 10000), "columnID": ("in", ["c1", "c2"])}``.
    :param output_dir: where PNGs (and any ``.npy`` for >3 channels) are
        written; defaults to ``<root>/object_dataset/<object_type>``.
    :param png_size: ``(width, height)`` the crop is resized to.
    :param mask_background: zero out pixels outside the object (isolate it).
    :param normalize: per-channel percentile-normalise before writing.
    :param percentiles: ``(low, high)`` percentiles for normalisation.
    :param buffer: pixels of padding around the object's bounding box.
    :param mask_dims: dict mapping object type → its mask slice index. Defaults
        to spaCR's layout ``{cell:4, nucleus:5, pathogen:6, organelle:7}`` (four
        image channels). Override if your arrays have a different channel count.
    :param save_png: write PNG files (set False to only collect arrays).
    :param return_arrays: also return the cropped arrays in the manifest.
    :param limit: cap the number of objects processed (handy for previews).
    :param db_path: explicit path to ``measurements.db`` (else derived from src).
    :param verbose: print a short progress summary.
    :returns: a manifest ``list[dict]``; each entry has ``object_label``,
        ``path_name``, ``plateID``/``rowID``/``columnID``/``fieldID``,
        ``png_path`` (if saved) and ``array`` (if ``return_arrays``).

    .. note::

       **The crop keeps the merged array's dtype.** A ``uint16`` field gives
       ``uint16`` crops, in the manifest and in the ``.npy`` written for more
       than three channels; ``normalize`` stretches into that dtype's full
       range, not into 0-255. The single narrowing to 8 bit happens in
       :func:`_save_object_crop`, where PIL needs it, and it *rescales*
       (:func:`_crop_to_uint8`).

       It used to cast to ``float32``, normalise into 0-255 and then
       ``np.clip(crop, 0, 255).astype(np.uint8)``. With ``normalize=False``
       that clip hit every 16-bit pixel brighter than 255 -- i.e. the whole
       object -- so the PNG written to disk was a solid white silhouette. The
       datasets built from it were trained on saturated images and nothing
       said so.
    """
    import os
    import sqlite3
    import numpy as np

    root = os.path.abspath(src)
    if os.path.basename(root.rstrip(os.sep)) == 'merged':
        root = os.path.dirname(root.rstrip(os.sep))
    if db_path is None:
        db_path = os.path.join(root, 'measurements', 'measurements.db')
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"measurements database not found: {db_path}")

    if mask_dims is None:
        layout = read_merged_plane_layout(os.path.join(root, 'merged'))
        mask_dims = dict((layout or {}).get('mask_dims') or DEFAULT_MASK_DIMS)
    if object_type not in mask_dims:
        raise ValueError(
            f"no mask slice known for object_type={object_type!r}; "
            f"pass mask_dims={{'{object_type}': <index>}}")
    mask_dim = int(mask_dims[object_type])

    channels = list(channels)
    if output_dir is None:
        output_dir = os.path.join(root, 'object_dataset', object_type)
    if save_png or return_arrays:
        os.makedirs(output_dir, exist_ok=True)
    if save_png:
        stamp_crop_folder(output_dir)

    clauses, params = [], []
    if min_area is not None:
        clauses.append(f"{object_type}_area > ?"); params.append(float(min_area))
    if max_area is not None:
        clauses.append(f"{object_type}_area < ?"); params.append(float(max_area))

    def _in(colname, values, prefix):
        """An ``IN (...)`` clause and its parameters, built safely.

        Placeholders rather than interpolation: the values come from a settings
        file, and a formatted list is an injection waiting for a filename with a
        quote in it.
        """
        vals = [f"{prefix}{int(v)}" if prefix else str(v) for v in values]
        placeholders = ",".join("?" for _ in vals)
        clauses.append(f"{colname} IN ({placeholders})")
        params.extend(vals)

    if columns:
        _in("columnID", columns, "c")
    if rows:
        _in("rowID", rows, "r")
    if fields:
        _in("fieldID", fields, "f")
    if plates:
        _in("plateID", plates, "")
    if criteria:
        for col, (op, val) in criteria.items():
            if str(op).lower() == "in":
                placeholders = ",".join("?" for _ in val)
                clauses.append(f"{col} IN ({placeholders})")
                params.extend(list(val))
            else:
                clauses.append(f"{col} {op} ?")
                params.append(val)
    if where:
        clauses.append(f"({where})")

    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    query = (
        "SELECT object_label, path_name, plateID, rowID, columnID, fieldID "
        f"FROM {object_type}{where_sql}{limit_sql}")

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        selected = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    if verbose:
        print(f"generate_object_dataset({object_type}): {len(selected)} "
              f"objects match{where_sql or ' (no filter)'}")

    manifest = []
    _array_cache = {}
    saved = 0
    merged_dir = os.path.join(root, 'merged')
    for row in selected:
        path_name = row["path_name"]
        label = int(row["object_label"])
        if path_name not in _array_cache:
            resolved = _resolve_merged_path(path_name, merged_dir)
            if resolved is None:
                if verbose:
                    print(f"  missing array, skipping: {path_name}")
                _array_cache[path_name] = None
            else:
                _array_cache[path_name] = np.load(resolved)
        data = _array_cache[path_name]
        if data is None:
            continue
        if data.ndim != 3:
            raise ValueError(
                f"generate_object_dataset crops 2-D merged arrays (Y, X, C); "
                f"{path_name} has shape {data.shape}. Project the z-stack "
                f"before building an object dataset.")
        if mask_dim >= data.shape[2]:
            raise IndexError(
                f"mask_dim {mask_dim} out of range for array with "
                f"{data.shape[2]} slices ({path_name})")

        mask = data[:, :, mask_dim]
        ys, xs = np.where(mask == label)
        if ys.size == 0:
            continue
        y0 = max(0, ys.min() - buffer); y1 = min(mask.shape[0], ys.max() + 1 + buffer)
        x0 = max(0, xs.min() - buffer); x1 = min(mask.shape[1], xs.max() + 1 + buffer)

        region = (mask[y0:y1, x0:x1] == label) if mask_background else None
        crop = _crop_channels(data, y0, y1, x0, x1, channels, region)
        if normalize:
            crop = _normalize_crop(crop, percentiles, mask_background)

        entry = {k: row[k] for k in
                 ("object_label", "path_name", "plateID", "rowID",
                  "columnID", "fieldID")}
        base = (f"{row['plateID']}_{row['rowID']}_{row['columnID']}_"
                f"{row['fieldID']}_obj{label}")

        if return_arrays:
            entry["array"] = crop

        if save_png:
            png_path = _save_object_crop(crop, channels, os.path.join(
                output_dir, base + ".png"), png_size)
            entry["png_path"] = png_path
            saved += 1

        manifest.append(entry)

    if verbose and save_png:
        print(f"generate_object_dataset({object_type}): wrote {saved} PNGs "
              f"→ {output_dir}")
    return manifest


def _save_object_crop(crop, channels, png_path, png_size):
    """Assemble ``crop`` (H, W, len(channels)) into an image and save it.

    3 channels → RGB PNG; 1 → greyscale; 2 → padded to RGB; >3 → the raw array
    is saved as ``.npy`` and the first three channels as a PNG preview. Returns
    the path actually written.

    **This is a declared 8-bit boundary.** PIL writes 8-bit PNGs here, so the
    crop is narrowed by :func:`_crop_to_uint8` -- a linear rescale off the
    dtype's range, not a clip at 255. The ``.npy`` written for a >3-channel
    crop is *not* narrowed: it keeps the full working dtype, because it is
    data, not a picture.
    """
    import os
    import numpy as np
    from PIL import Image

    n = crop.shape[2]
    if n > 3:
        npy_path = os.path.splitext(png_path)[0] + ".npy"
        np.save(npy_path, crop)
        preview = _crop_to_uint8(crop[:, :, :3])
        Image.fromarray(preview).resize(tuple(png_size)).save(png_path)
        return npy_path
    eight = _crop_to_uint8(crop)
    if n == 1:
        img = Image.fromarray(eight[:, :, 0], mode="L")
    elif n == 2:
        rgb = np.zeros((*eight.shape[:2], 3), dtype=np.uint8)
        rgb[:, :, :2] = eight
        img = Image.fromarray(rgb)
    else:
        img = Image.fromarray(eight)
    img.resize(tuple(png_size)).save(png_path)
    return png_path


def crop_objects_from_array(data, mask_dim, channels=(0, 1, 2),
                            min_area=0, max_area=0, mask_background=True,
                            normalize=True, percentiles=(1, 99), buffer=10,
                            to_rgb=True, limit=None, size=None):
    """Crop every object out of an in-memory merged image+mask array.

    This is the no-database counterpart of :func:`generate_object_dataset`,
    used by the Measure live preview to show what the crops will look like
    before a run: it reads the object labels straight from a mask slice of a
    single merged ``.npy`` and returns the cropped, normalised images.

    :param data: merged array ``(H, W, C)`` — image channels then mask slices.
    :param mask_dim: slice index of the object-class mask to crop by.
    :param channels: image channel indices to assemble (order = RGB order).
    :param min_area: smallest object area (px) to keep; ``0`` = no lower bound.
    :param max_area: largest object area (px) to keep; ``0`` = no upper bound.
    :param mask_background: zero pixels outside the object.
    :param normalize: per-channel percentile-normalise each crop.
    :param percentiles: ``(low, high)`` for normalisation.
    :param buffer: padding (px) around each object's bounding box.
    :param to_rgb: assemble the chosen channels into an HxWx3 uint8 image
        (1→grey→RGB, 2→padded, 3→RGB, >3→first three); else keep N channels
        **in the merged array's own dtype**.
    :param limit: cap the number of objects returned.
    :param size: ``(width, height)`` to resize every crop to, or ``None`` to
        return each object's own bounding box. This is ``measure_crop``'s
        ``png_size``, resized THE WAY THE RUN RESIZES -- the same
        ``PIL.Image.resize`` call at its default resampling -- because this
        feeds the Measure preview, whose purpose is to show what a run will
        write. Without it the preview showed bounding boxes while the run
        wrote squares, and the crop-size setting looked like it did nothing.
    :returns: list of ``{'label', 'area', 'bbox', 'crop'}`` dicts, largest
        objects first.

    .. note::

       ``to_rgb=True`` is the one place this function leaves the working
       dtype, because a GUI image is 8-bit. It narrows with
       :func:`_crop_to_uint8` (a rescale off the dtype range), not with a clip
       at 255 -- a clip made every pixel of an unnormalised 16-bit object come
       back as pure white, so the preview showed a white blob and the run it
       was previewing did not.
    """
    import numpy as np

    channels = list(channels)
    mask = data[:, :, int(mask_dim)]
    labels = np.unique(mask)
    labels = labels[labels > 0]

    scored = []
    for lbl in labels:
        area = int(np.sum(mask == lbl))
        if min_area and area < min_area:
            continue
        if max_area and area > max_area:
            continue
        scored.append((area, int(lbl)))
    scored.sort(reverse=True)
    if limit:
        scored = scored[:int(limit)]

    out = []
    for area, lbl in scored:
        ys, xs = np.where(mask == lbl)
        y0 = max(0, ys.min() - buffer); y1 = min(mask.shape[0], ys.max() + 1 + buffer)
        x0 = max(0, xs.min() - buffer); x1 = min(mask.shape[1], xs.max() + 1 + buffer)

        region = (mask[y0:y1, x0:x1] == lbl) if mask_background else None
        crop = _crop_channels(data, y0, y1, x0, x1, channels, region)
        if normalize:
            crop = _normalize_crop(crop, percentiles, mask_background)

        if to_rgb:
            crop = _crop_to_uint8(crop)
            n = crop.shape[2]
            if n == 1:
                crop = np.repeat(crop, 3, axis=2)
            elif n == 2:
                rgb = np.zeros((*crop.shape[:2], 3), dtype=np.uint8)
                rgb[:, :, :2] = crop
                crop = rgb
            elif n > 3:
                crop = np.ascontiguousarray(crop[:, :, :3])

        if size is not None:
            crop = _resize_crop_like_the_run(crop, size)

        out.append({"label": lbl, "area": area,
                    "bbox": (int(y0), int(y1), int(x0), int(x1)), "crop": crop})
    return out


def _resize_crop_like_the_run(crop, size):
    """Resize one crop to ``size`` the way a real run does.

    :param crop: the crop, ``HxWxC`` (or ``HxW``) in any dtype.
    :param size: ``(width, height)``.
    :returns: the resized crop, in the dtype it arrived in.

    A run resizes at save time with ``Image.fromarray(...).resize(png_size)``
    and nothing else, so the preview makes the same call with the same default
    resampling. An 8-bit RGB crop goes through PIL whole; anything else goes
    plane by plane through PIL's 32-bit float mode and is cast back, because
    ``Image.fromarray`` refuses most multi-channel non-8-bit arrays and a
    preview that raised here would show nothing at all.
    """
    import numpy as np
    from PIL import Image

    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return crop
    if crop.ndim == 3 and crop.shape[2] == 3 and crop.dtype == np.uint8:
        return np.asarray(Image.fromarray(crop).resize((width, height)))
    planes = ([crop] if crop.ndim == 2
              else [crop[:, :, i] for i in range(crop.shape[2])])
    resized = [
        np.asarray(Image.fromarray(plane.astype(np.float32),
                                   mode="F").resize((width, height)))
        for plane in planes
    ]
    stacked = resized[0] if crop.ndim == 2 else np.stack(resized, axis=2)
    if np.issubdtype(crop.dtype, np.integer):
        info = np.iinfo(crop.dtype)
        stacked = np.clip(np.rint(stacked), info.min, info.max)
    return stacked.astype(crop.dtype)


#: Named groups the FEATURES regex may use to name the ROW a file belongs to.
#: The first one the pattern defines wins, so a caller may spell it whichever
#: way the filenames already do.
FIELD_TABLE_FIELD_GROUPS: tuple = (
    'field', 'fieldID', 'fov', 'stem', 'name')

#: Named groups that put a file in one of the table's CHANNEL columns. The
#: captured token is not read as a number -- see :func:`assign_paths_by_regex`
#: for why the distinct tokens are ranked instead.
FIELD_TABLE_CHANNEL_GROUPS: tuple = (
    'channel', 'chanID', 'chan', 'c')

#: Named groups that put a file in one of the table's MASK columns. The
#: captured token is resolved to a role by :func:`mask_role_of`.
FIELD_TABLE_MASK_GROUPS: tuple = (
    'mask', 'object', 'objectID', 'role')

#: Named groups that name the plate and the well a row belongs to, when the
#: filenames carry them. Hand-drawn masks usually do not, which is what
#: :attr:`FieldTable.plate` and :attr:`FieldRow.well` are for.
FIELD_TABLE_PLATE_GROUPS: tuple = ('plateID', 'plate')
FIELD_TABLE_WELL_GROUPS: tuple = ('wellID', 'well')

#: What a user may type in a mask column and mean a spaCR role by. The values
#: are roles from :data:`spacr.crops.MASK_PLANE_ORDER`; ``organelle`` slots
#: are matched separately by :func:`mask_role_of` because there are 700 of
#: them and they are spelled by number on screen.
_MASK_ROLE_SYNONYMS = {
    'cell': 'cell', 'cells': 'cell', 'cyto': 'cell', 'whole': 'cell',
    'nucleus': 'nucleus', 'nuclei': 'nucleus', 'nuc': 'nucleus',
    'nuclear': 'nucleus', 'dapi': 'nucleus',
    'pathogen': 'pathogen', 'pathogens': 'pathogen', 'parasite': 'pathogen',
    'parasites': 'pathogen', 'bacteria': 'pathogen',
    'bacterium': 'pathogen', 'bacterial': 'pathogen', 'pv': 'pathogen',
    'mito': 'organelle', 'mitochondria': 'organelle',
    'mitochondrion': 'organelle', 'organelle': 'organelle',
}

_ORGANELLE_TOKEN = re.compile(
    r'(?i)^organelle[_\-. ]?(?P<number>\d+)$')

_TRAILING_DIGITS = re.compile(r'(\d+)\s*$')


def mask_role_of(token):
    """Resolve what a user typed in a mask column to a spaCR object role.

    Accepts the role's own name, the plural and the common laboratory
    synonyms (``nuclei``, ``parasite``, ``mito``), and the numbered organelle
    spelling the settings forms use on screen -- ``Organelle 2`` is
    ``organelleb``, because the slots are lettered internally and numbered
    for the reader.

    :param token: what the regex captured or the user chose, in any case.
    :returns: a role from :data:`spacr.crops.MASK_PLANE_ORDER`, or ``None``
        when the token names no object spaCR can measure.
    """
    if token is None:
        return None
    text = str(token).strip().lower()
    if not text:
        return None
    if text in SEGMENTED_ROLES:
        return text
    numbered = _ORGANELLE_TOKEN.match(text)
    if numbered is not None:
        index = int(numbered.group('number'))
        if 1 <= index <= len(ORGANELLE_ROLES):
            return ORGANELLE_ROLES[index - 1]
        return None
    return _MASK_ROLE_SYNONYMS.get(text)


def _channel_rank_key(token):
    """Order channel tokens the way a microscope names them.

    ``C10`` sorts after ``C9`` rather than after ``C1``, because the digits
    at the end are compared as a number. Tokens with no trailing digits fall
    back to their text, after every numbered one.
    """
    text = str(token)
    match = _TRAILING_DIGITS.search(text)
    if match is None:
        return (1, text.lower(), 0)
    return (0, text[:match.start()].lower(), int(match.group(1)))


@dataclass
class FieldRow:
    """One row of the FEATURES table: one field, and the files that make it.

    A row becomes exactly one ``merged/<stem>.npy``, so it is also one field
    in the measurements database.

    :ivar label: what the row is called in the table's first column, taken
        from the filenames. It is not the database identity; :attr:`well` and
        :attr:`field` are.
    :ivar channels: ``channel index -> source path``. The indices are
        positions on the merged array's channel axis, counted from zero.
    :ivar masks: ``role -> source path``, for the roles this row supplies.
        The same mask file may appear in several rows, which is how one
        drawn mask is measured against several acquisitions.
    :ivar well: the well id this field is filed under. Hand-drawn fields did
        not come from a plate, so they share one well by default and the
        table shows it rather than inventing a different one per row.
    :ivar field: the field number within that well, unique per row.
    """

    label: str
    channels: Dict[int, str] = dataclasses_field(default_factory=dict)
    masks: Dict[str, str] = dataclasses_field(default_factory=dict)
    well: str = 'A01'
    field: int = 1

    def stem(self, plate):
        """The ``plate_well_field`` name this row is written and measured as.

        :param plate: the plate name the whole table carries.
        :returns: the stem, which :func:`spacr.schema.parse_field_stem` reads
            back into the plate, row, column and field the database is keyed
            by.
        """
        return f"{plate}_{self.well}_{int(self.field)}"


@dataclass
class FieldTable:
    """Rows are fields, columns are channels and mask types.

    This is the thing the FEATURES window edits and the only input
    :func:`measure_from_field_table` needs. It is deliberately Qt-free: the
    window drives it, and the tests drive it without a window.

    :ivar rows: one :class:`FieldRow` per field, in table order.
    :ivar n_channels: how many channel columns the table has.
    :ivar roles: which mask columns it has, in
        :data:`spacr.crops.MASK_PLANE_ORDER` order -- which is the order the
        planes are stacked in, so the two cannot drift.
    :ivar plate: the plate name every row's stem starts with. It names where
        the files came from rather than claiming a plate that was never run.
    :ivar channel_tokens: which channel token owns which channel column, by
        position: ``channel_tokens[i]`` is the token that column ``i`` means.
        THE TABLE REMEMBERS THIS BECAUSE THE TABLE OUTLIVES THE DROP. The
        ranking that turns ``C1``/``C2`` into columns 0 and 1 is a property
        of a SET of tokens, and a user fills this table one field at a time,
        so without a memory the second drop would rank its own files from
        scratch and put ``C2`` in column 0 beside the first drop's ``C1``.
        Empty means no column means any particular token yet -- every cell
        was filled by browsing rather than by the regex.
    """

    rows: List[FieldRow] = dataclasses_field(default_factory=list)
    n_channels: int = 1
    roles: Tuple[str, ...] = ('cell',)
    plate: str = 'drawn'
    channel_tokens: Tuple[str, ...] = ()

    def ordered_roles(self):
        """The mask columns in merged-plane order, duplicates removed."""
        return tuple(role for role in MASK_PLANE_ORDER if role in self.roles)

    def mask_dims(self):
        """``role -> plane index`` on the merged array this table would write.

        The masks follow the channels with no gap, which is the only layout
        :func:`spacr.crops.read_merged_plane_layout` accepts -- it recomputes
        the indices from the channel count and the order and refuses a
        manifest that disagrees.
        """
        return {role: int(self.n_channels) + index
                for index, role in enumerate(self.ordered_roles())}

    def problems(self):
        """Everything that would stop this table being measured, as sentences.

        Empty means :func:`measure_from_field_table` will run. The window
        shows these live, so a user never presses a Run button that is going
        to refuse.
        """
        issues = []
        if int(self.n_channels) < 1:
            issues.append("The table needs at least one channel column.")
        if not self.ordered_roles():
            issues.append(
                "The table needs at least one mask column -- there is "
                "nothing to measure without an object.")
        if not self.rows:
            issues.append("The table has no fields in it.")
        seen = {}
        for row in self.rows:
            key = (row.well, int(row.field))
            if key in seen:
                issues.append(
                    f"{row.label} and {seen[key]} are both well "
                    f"{row.well} field {row.field}; one would overwrite the "
                    "other.")
            seen[key] = row.label
            for channel in range(int(self.n_channels)):
                if not row.channels.get(channel):
                    issues.append(
                        f"{row.label} has no file for channel "
                        f"{channel + 1}.")
            for role in self.ordered_roles():
                if not row.masks.get(role):
                    issues.append(
                        f"{row.label} has no {role} mask.")
        return issues

    def is_ready(self):
        """Whether the table is complete enough to measure."""
        return not self.problems()


def _renumber_channels(table, known, ranked):
    """Move every row's channel files to the columns ``ranked`` now gives them.

    WHICH COLUMN A TOKEN MEANS IS A PROPERTY OF THE TABLE, NOT OF ONE DROP,
    and this is what keeps it so. The documented way to use the FEATURES
    window is one field at a time -- draw, press FEATURES, move to the next
    image, draw again -- so a later drop can introduce a token that ranks
    before one already placed. Re-ranking without moving the files already in
    the table leaves ``C2`` in column 0 for the first field and column 1 for
    the second, and NOTHING ON SCREEN SAYS SO: the table reads as complete,
    the run succeeds, and ``cell_channel_0_mean_intensity`` in the database
    is a different stain for different fields. Renumbering is how the
    invariant survives the second drop.

    Columns no token claims -- cells filled by browsing rather than by the
    regex -- keep their files. They are given the columns after the tokened
    ones, in their old order, so nothing a user put somewhere is dropped.

    :param table: the :class:`FieldTable` to renumber, edited in place.
    :param known: the token order the rows' current column numbers mean.
    :param ranked: the token order they should mean.
    :returns: ``None``.
    """
    moves = {index: ranked.index(token)
             for index, token in enumerate(known) if token in ranked}
    if not moves:
        return
    taken = set(moves.values())
    occupied = {index for row in table.rows for index in row.channels}
    spare = len(ranked)
    for index in sorted(index for index in occupied if index not in moves):
        while spare in taken:
            spare += 1
        moves[index] = spare
        taken.add(spare)
    if all(old == new for old, new in moves.items()):
        return
    for row in table.rows:
        row.channels = {moves.get(index, index): path
                        for index, path in row.channels.items()}


@dataclass
class TableAssignment:
    """What one regex did to one set of dropped files.

    :ivar table: the table the files were assigned into.
    :ivar assigned: ``(path, row label, column caption)`` for every file that
        landed somewhere, in the order the paths were given.
    :ivar unassigned: ``(path, reason)`` for every file that did not. The
        window lists these, because a file that silently vanishes is the one
        failure a drag-and-drop table cannot afford.
    """

    table: FieldTable
    assigned: List[Tuple[str, str, str]] = dataclasses_field(
        default_factory=list)
    unassigned: List[Tuple[str, str]] = dataclasses_field(
        default_factory=list)


def assign_paths_by_regex(paths, pattern, *, table=None, plate=None):
    """Sort dropped files into rows and channel/mask columns with one regex.

    The regex is matched against each file's BASENAME. What it captures
    decides where the file goes:

    * one of :data:`FIELD_TABLE_FIELD_GROUPS` names the row. Files sharing a
      field token share a row, which is what makes a four-channel field one
      row rather than four.
    * one of :data:`FIELD_TABLE_MASK_GROUPS` sends it to a mask column,
      through :func:`mask_role_of`.
    * one of :data:`FIELD_TABLE_CHANNEL_GROUPS` sends it to a channel column.

    THE CHANNEL TOKEN IS RANKED, NOT READ AS A NUMBER, and that is the one
    decision here worth knowing about. ``C1``/``C2``/``C3`` and ``w1``/``w2``
    and ``0``/``1``/``2`` all have to end up as channels 0, 1, 2, and there is
    no reading of ``C1`` that is right for all three -- a literal read makes
    the first set start at channel 1 and leaves channel 0 empty for ever.
    So the DISTINCT channel tokens are sorted (numerically on their trailing
    digits) and mapped onto 0, 1, 2 ... in that order. The mapping is
    therefore a property of the set of files, not of any one of them, which
    is why the window shows the assignment rather than describing the rule.

    THE SET IS THE TABLE'S, NOT THE DROP'S. ``table`` remembers which token
    owns which column in :attr:`FieldTable.channel_tokens`, and a later drop
    is ranked against the union of what it brings and what is already there.
    A token that ranks before one already placed renumbers the columns and
    MOVES the files already in them (:func:`_renumber_channels`), so every
    row agrees about what channel 0 is. Ranking each drop on its own instead
    would put a second field's ``C2`` in column 0 beside a first field's
    ``C1``, and the only sign of it would be in the database.

    :param paths: file paths to assign.
    :param pattern: a regex with at least a field group and one of a channel
        or mask group.
    :param table: an existing table to add to. A new one is built when this
        is ``None``; its channel count and mask columns come from what the
        files turn out to hold.
    :param plate: the plate name for a new table.
    :returns: a :class:`TableAssignment`. Nothing is read from disk and
        nothing is written.
    :raises re.error: if ``pattern`` does not compile. The window catches
        this and shows it under the box rather than letting it reach a run.
    """
    compiled = re.compile(pattern)
    groups = set(compiled.groupindex)

    def first(names):
        """The first of ``names`` the pattern actually defines."""
        for name in names:
            if name in groups:
                return name
        return None

    field_group = first(FIELD_TABLE_FIELD_GROUPS)
    channel_group = first(FIELD_TABLE_CHANNEL_GROUPS)
    mask_group = first(FIELD_TABLE_MASK_GROUPS)
    plate_group = first(FIELD_TABLE_PLATE_GROUPS)
    well_group = first(FIELD_TABLE_WELL_GROUPS)

    existing = table if table is not None else FieldTable(
        rows=[], n_channels=0, roles=(), plate=plate or 'drawn')
    if plate is not None:
        existing.plate = plate
    result = TableAssignment(table=existing)

    if field_group is None:
        for path in paths:
            result.unassigned.append((str(path), (
                "the regex names no field group, so there is no row to put "
                "this in -- add (?P<field>...) to it")))
        return result
    if channel_group is None and mask_group is None:
        for path in paths:
            result.unassigned.append((str(path), (
                "the regex names neither a channel nor a mask group, so "
                "there is no column to put this in")))
        return result

    matched = []
    for path in paths:
        text = os.path.basename(str(path))
        found = compiled.search(text)
        if found is None:
            result.unassigned.append(
                (str(path), f"{text} does not match the regex"))
            continue
        captured = found.groupdict()
        label = captured.get(field_group)
        if not label:
            result.unassigned.append(
                (str(path), f"{text} matched but captured no field name"))
            continue
        mask_token = captured.get(mask_group) if mask_group else None
        channel_token = captured.get(channel_group) if channel_group else None
        if mask_token:
            role = mask_role_of(mask_token)
            if role is None:
                result.unassigned.append((str(path), (
                    f"{text} names the object {mask_token!r}, which is not "
                    "a spaCR mask type")))
                continue
            matched.append((str(path), str(label), 'mask', role, captured))
        elif channel_token is not None and str(channel_token) != '':
            matched.append((str(path), str(label), 'channel',
                            str(channel_token), captured))
        else:
            result.unassigned.append((str(path), (
                f"{text} matched but captured neither a channel nor an "
                "object")))

    known = [str(token) for token in getattr(existing, 'channel_tokens', ())]
    tokens = sorted(set(known) | {token for _p, _l, kind, token, _c in matched
                                  if kind == 'channel'},
                    key=_channel_rank_key)
    channel_of = {token: index for index, token in enumerate(tokens)}
    _renumber_channels(existing, known, tokens)
    existing.channel_tokens = tuple(tokens)

    rows_by_label = {row.label: row for row in existing.rows}
    for path, label, kind, token, captured in matched:
        row = rows_by_label.get(label)
        if row is None:
            row = FieldRow(label=label, well='A01',
                           field=len(existing.rows) + 1)
            if well_group and captured.get(well_group):
                row.well = str(captured[well_group])
            existing.rows.append(row)
            rows_by_label[label] = row
        if plate_group and captured.get(plate_group) and plate is None:
            existing.plate = str(captured[plate_group])
        if kind == 'mask':
            row.masks[token] = path
            if token not in existing.roles:
                existing.roles = tuple(existing.roles) + (token,)
            result.assigned.append((path, label, f"{token} mask"))
        else:
            index = channel_of[token]
            row.channels[index] = path
            existing.n_channels = max(int(existing.n_channels), index + 1)
            result.assigned.append((path, label, f"channel {index + 1}"))

    existing.roles = existing.ordered_roles()
    highest = max((max(row.channels) for row in existing.rows if row.channels),
                  default=-1)
    existing.n_channels = max(int(existing.n_channels), len(tokens),
                              highest + 1)
    return result


def field_table_settings(table, settings=None, dst=None):
    """The measure_crop settings this table decides, over the ones it does not.

    Everything the table can answer is answered from the table: the channel
    list, the mask plane of every object it supplies, the crop modes that are
    possible, the PNG channels, and ``src``. Every other key is the user's,
    taken from ``settings`` and defaulted by
    :func:`spacr.settings.get_measure_crop_settings` exactly as the Measure
    module defaults them -- so the FEATURES window and the Measure module
    disagree about nothing.

    A role the table does NOT supply is set to ``None`` rather than left out,
    which is how ``measure_crop`` is told not to measure it.

    :param table: the :class:`FieldTable` the user filled in.
    :param settings: the user's answers from the settings panel.
    :param dst: the project root the run will write. ``src`` is its
        ``merged`` folder, which is where ``measure_crop`` reads fields from.
    :returns: a new settings dict. Nothing is read from disk.
    """
    from .settings import get_measure_crop_settings

    resolved = get_measure_crop_settings(dict(settings or {}))
    if dst is not None:
        resolved['src'] = os.path.join(str(dst), 'merged')
    resolved['channels'] = list(range(int(table.n_channels)))
    dims = table.mask_dims()
    for role in SEGMENTED_ROLES:
        resolved[f'{role}_mask_dim'] = dims.get(role)
    if not resolved.get('png_dims'):
        resolved['png_dims'] = list(range(min(int(table.n_channels), 3)))
    supplied = list(table.ordered_roles())
    available = list(supplied)
    if 'cell' in supplied and resolved.get('cytoplasm'):
        available.append('cytoplasm')
    requested = resolved.get('crop_mode') or []
    if isinstance(requested, str):
        requested = [requested]
    kept = [name for name in requested if name in available]
    resolved['crop_mode'] = kept or available[:1]
    return resolved


#: Settings the table decides, so the FEATURES window shows them filled in
#: and not editable. Everything else on that panel is the user's to set.
FIELD_TABLE_DECIDED_KEYS: tuple = (
    'src', 'channels', 'png_dims',
    *(f'{role}_mask_dim' for role in MASK_PLANE_ORDER),
)


def _readable_plane(path):
    """Read one image or label file as a 2-D array, whatever format it is in.

    Goes through :func:`spacr.foreign._read_mask`, so every format spaCR's
    converter opens -- TIFF, PNG, ND2, CZI, LIF -- is readable here too,
    and there is no second reader table to keep in step with that one.
    """
    from .foreign import _read_mask

    return np.asarray(_read_mask(str(path)))


def _checked_intensity(plane, stem, path):
    """Return ``plane`` as uint16, or say why it cannot be measured.

    The same four checks :mod:`spacr.external_masks` applies, for the same
    reason: ``measure_crop`` reads a uint16 merged array, and a float image
    silently truncated into one gives numbers that look like measurements.
    """
    if np.issubdtype(plane.dtype, np.floating):
        if not np.all(np.isfinite(plane)):
            raise ConfigurationError(
                f"{path}: {stem} intensity data contain NaN or infinity.")
        if not np.all(plane == np.floor(plane)):
            raise ConfigurationError(
                f"{path}: {stem} has floating-point intensities that would "
                "lose precision in Measure's uint16 arrays. Rescale and "
                "export them as 8- or 16-bit images first.")
    if float(np.min(plane, initial=0)) < 0 or \
            float(np.max(plane, initial=0)) > np.iinfo(np.uint16).max:
        raise ConfigurationError(
            f"{path}: {stem} intensity values must fit the Measure uint16 "
            "contract (0-65535). Rescale the source images first.")
    return plane.astype(np.uint16, copy=False)


def _checked_label(plane, stem, path, shape):
    """Return ``plane`` as a uint16 label image, or say why it cannot be one."""
    if plane.shape != shape:
        raise ConfigurationError(
            f"{path}: mask shape {plane.shape} does not match the intensity "
            f"shape {shape} for {stem}.")
    if np.any(plane < 0):
        raise ConfigurationError(
            f"{path}: label masks cannot contain negative IDs.")
    maximum = int(np.max(plane, initial=0))
    if maximum > np.iinfo(np.uint16).max:
        raise ConfigurationError(
            f"{path}: label ID {maximum} exceeds the maximum 65535 supported "
            "by the Measure array contract.")
    return plane.astype(np.uint16, copy=False)


def write_field_table_project(table, dst):
    """Write the table out as the folders the Mask module leaves behind.

    This is the whole of what the FEATURES button adds to Measure: it turns a
    table of hand-picked files into ``stack/``, ``masks/`` and ``merged/``
    exactly as :func:`spacr.core.preprocess_generate_masks` would have left
    them, down to the plane-layout manifest, so the run that follows is an
    ORDINARY measure run and not a second code path that has to be kept in
    step with this one.

    :param table: a :class:`FieldTable` whose :meth:`FieldTable.problems` is
        empty.
    :param dst: the project root to write. It is created if it does not
        exist.
    :returns: ``{'destination', 'merged', 'stack', 'masks', 'stems'}``.
    :raises spacr.errors.ConfigurationError: if the table is incomplete, or
        if any file breaks the uint16 array contract ``measure_crop`` reads.
        Nothing is written past the field that failed.
    """
    problems = table.problems()
    if problems:
        raise ConfigurationError(
            "The measurement table is not ready; nothing was written:\n  "
            + "\n  ".join(problems))

    dst = os.fspath(dst)
    roles = table.ordered_roles()
    merged_dir = os.path.join(dst, 'merged')
    stack_dir = os.path.join(dst, 'stack')
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(stack_dir, exist_ok=True)

    merged_paths = []
    stack_paths = []
    mask_paths = {role: [] for role in roles}
    stems = []
    for row in table.rows:
        stem = row.stem(table.plate)
        stems.append(stem)
        planes = []
        for channel in range(int(table.n_channels)):
            path = row.channels[channel]
            planes.append(_checked_intensity(
                _readable_plane(path), stem, path))
        shape = planes[0].shape
        for index, plane in enumerate(planes):
            if plane.shape != shape:
                raise ConfigurationError(
                    f"{row.channels[index]}: channel {index + 1} of {stem} "
                    f"has shape {plane.shape}, but channel 1 has {shape}.")

        labels = []
        for role in roles:
            path = row.masks[role]
            label = _checked_label(
                _readable_plane(path), stem, path, shape)
            labels.append(label)
            role_dir = os.path.join(dst, 'masks', f'{role}_mask_stack')
            os.makedirs(role_dir, exist_ok=True)
            role_path = os.path.join(role_dir, f'{stem}.npy')
            np.save(role_path, label)
            mask_paths[role].append(role_path)

        stack_path = os.path.join(stack_dir, f'{stem}.npy')
        np.save(stack_path, np.stack(planes, axis=-1))
        stack_paths.append(stack_path)
        merged_path = os.path.join(merged_dir, f'{stem}.npy')
        np.save(merged_path, np.stack([*planes, *labels], axis=-1))
        merged_paths.append(merged_path)

    layout = {
        'version': 1,
        'intensity_channels': list(range(int(table.n_channels))),
        'mask_plane_order': list(roles),
        'mask_dims': dict(table.mask_dims()),
    }
    from .crops import MERGED_LAYOUT_SIDECAR

    with open(os.path.join(merged_dir, MERGED_LAYOUT_SIDECAR), 'w',
              encoding='utf-8') as handle:
        json.dump(layout, handle, indent=2, sort_keys=True)
        handle.write('\n')

    return {'destination': dst, 'merged': merged_paths,
            'stack': stack_paths, 'masks': mask_paths, 'stems': stems}


def field_table_destination(table, dst=None):
    """Where a run of ``table`` would write, given the destination it was handed.

    ONE ANSWER, so that the window and the run cannot disagree about it.
    ``src`` is one of :data:`FIELD_TABLE_DECIDED_KEYS`, so the FEATURES
    window shows it filled in and disabled; before this existed the window
    derived it only when it had been given a destination, and a window
    opened without one showed the settings spec's ``path`` placeholder while
    the run wrote beside the first channel file. A disabled box captioned
    "the table decides this" that names the wrong folder is worse than no box
    at all -- it is the window telling the user where their results are not.

    :param table: the :class:`FieldTable` the run would measure.
    :param dst: the destination the caller was given, or ``None`` to derive
        one from the table.
    :returns: the project root, or ``None`` when the table is too empty to
        derive one. Nothing is read from disk.
    """
    if dst is not None:
        return os.fspath(dst)
    first = table.rows[0].channels.get(0) if table.rows else None
    if not first:
        return None
    return os.path.join(
        os.path.dirname(os.path.abspath(str(first))), 'features')


def measure_from_field_table(table, settings=None, dst=None, progress=None):
    """Measure a table of hand-picked images and masks. The FEATURES entry point.

    The other way into this module. :func:`measure_crop` starts from a
    ``merged/`` folder a pipeline already built; this starts from a table a
    user filled in by dropping files onto it, writes that folder, and then
    calls :func:`measure_crop` ITSELF -- unchanged, with no flag saying where
    the fields came from. The database, the crops and the folder tree are
    therefore the Measure module's, because they are made by it.

    :param table: the :class:`FieldTable` the FEATURES window edited.
    :param settings: the user's answers from the settings panel. The keys the
        table decides are overwritten from it -- see
        :func:`field_table_settings`.
    :param dst: the project root to write. Defaults to a ``features``
        folder beside the first channel file of the first row, which is where
        a user who dropped a folder in expects to find the results. See
        :func:`field_table_destination`, which is the one place that default
        is worked out.
    :param progress: called with a sentence as each stage starts, or
        ``None``. It runs on whatever thread this does -- the FEATURES window
        runs this on a worker and its callback only emits a signal. Writing
        the arrays and measuring them are separate stages because on a large
        table the second takes minutes and the first does not.
    :returns: ``{'destination', 'db_path', 'settings', 'stems', 'merged'}``.
        ``db_path`` is the measurements database whether or not it exists, so
        a caller can report the path it was asked for.
    :raises spacr.errors.ConfigurationError: if the table is incomplete or a
        file breaks the array contract. Nothing is measured in that case.

    Example:
        .. code-block:: python

            from spacr.measure import (
                assign_paths_by_regex, measure_from_field_table)

            found = assign_paths_by_regex(
                paths,
                r'(?P<field>fov\\d+)_(?:C(?P<channel>\\d+)'
                r'|(?P<mask>cell|nucleus))')
            measure_from_field_table(found.table, {'save_png': True})

    See Also:
        :func:`measure_crop` -- the run this delegates to, unchanged.
        :func:`write_field_table_project` -- the folders it writes first.
    """
    def say(message):
        """Report a stage, if anyone asked to hear about them.

        Guarded because the caller is a window that may be closed while this
        is still running: the FEATURES window's callback emits a Qt signal,
        and a worker parked past its widget's destruction raises
        ``RuntimeError`` from the emit. A run must not fail because nobody is
        listening to it any more.
        """
        if progress is None:
            return
        try:
            progress(str(message))
        except Exception:                                        # noqa: BLE001
            pass

    dst = field_table_destination(table, dst)
    if dst is None:
        raise ConfigurationError(
            "There is nowhere to write: the table's first field has no "
            "channel file, and no destination was given.")
    say(f"Writing the merged arrays for {len(table.rows)} field(s)...")
    written = write_field_table_project(table, dst)
    resolved = field_table_settings(table, settings, dst=dst)
    say(f"Wrote {len(written['stems'])} field(s). Measuring them now; "
        "this is the Measure module's own run.")
    measure_crop(resolved)
    return {
        'destination': written['destination'],
        'db_path': os.path.join(str(dst), 'measurements', 'measurements.db'),
        'settings': resolved,
        'stems': written['stems'],
        'merged': written['merged'],
    }
