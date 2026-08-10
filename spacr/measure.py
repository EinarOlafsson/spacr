"""Per-object morphology and intensity measurement pipeline."""

import os, cv2, time, sqlite3, threading, traceback, shutil
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr, skew, kurtosis, mode
import multiprocessing as mp
from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_dilation, binary_erosion, gaussian_filter, center_of_mass, convolve
from skimage.measure import regionprops, regionprops_table, shannon_entropy
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from skimage.feature import graycomatrix, graycoprops
from skimage import morphology, measure, filters
from skimage.util import img_as_bool
import matplotlib.pyplot as plt
from math import ceil, sqrt

# The crop PNG format lives in spacr.crops: the writer's channel order
# (to_cv2_bgr), the folder marker (stamp_crop_folder) and the reader that
# undoes the legacy order all have to agree, so they live in one place.
# spacr.crops imports nothing from spacr and nothing heavy, so this costs the
# measure path nothing.
from .crops import (
    build_png_channels,
    narrow_to_uint8,
    resolve_png_channel_mapping,
    stamp_crop_folder,
    to_cv2_bgr,
)
# Backward-compatible public module binding used by downstream extensions.
from . import settings as _settings_module
settings = _settings_module
# Public compatibility alias: measurement writers and downstream feature
# dictionaries historically import this constant from ``spacr.measure``.
from . import measurement_schema as _measurement_schema
MEASUREMENT_STAMP_COLUMNS = _measurement_schema.MEASUREMENT_STAMP_COLUMNS
# Fail-loud accounting: a field that fails to measure is recorded, summarised
# at the end of the run, and stamped into measurements.db so a downstream
# regression cannot silently analyse 344 of 384 wells.
from .errors import RunLedger, ConfigurationError, raise_if_strict
# One run id on every log line and every artifact, one seed reaching every
# RNG, one on_error policy at each batch boundary. See spacr.runctx.
from .runctx import run_context
from .resume import plan_measure_resume
# Opt-in extension points, so illumination correction and a user-drawn ROI can
# change what is measured without editing this module. Both registries are
# empty by default and both entry points return their input object unchanged
# when they are, so an ordinary run is byte-identical to one from before they
# existed. spacr.measure_hooks imports numpy, the stdlib and spacr.errors only
# -- registering a hook must not drag in matplotlib/skimage/cv2. Re-exported
# here so `from spacr.measure import register_preprocessing_hook` also works.
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


# ---------------------------------------------------------------------------
# 3-D support: dimensionality, voxel spacing, and the units stamp
# ---------------------------------------------------------------------------
#
# spaCR's mask generation can now emit (Z, Y, X) label volumes (see
# spacr.zstack, MODE_STITCH / MODE_VOLUMETRIC). Everything below exists so that
# such a volume is measured correctly *or refused*, and never measured wrongly.
#
# Two invariants govern every change in this module:
#
# 1. **The 2-D path is bit-identical.** A 2-D field takes exactly the code it
#    took before: ``spacing`` is None (skimage treats ``spacing=None`` as
#    "omitted"), no property is dropped, no distance transform is sampled, and
#    no column is renamed. Physical scaling is deliberately NOT applied in 2-D
#    even when a voxel size is available, because that would turn every
#    existing ``*_area`` column from px^2 into um^2 under an unchanged name and
#    silently break every threshold ever written against a spaCR database.
#
# 2. **A row's units are never guessed at.** A 3-D run measures volumes where a
#    2-D run measures areas, so every row written to measurements.db carries
#    :data:`MEASUREMENT_STAMP_COLUMNS`, and
#    ``spacr.utils._merge_and_save_to_database`` refuses to append rows whose
#    units differ from the ones already in the table.

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
        # An unusable name (``fork`` on Windows, a typo) must not take the run
        # down: the platform default measures the same rows.
        print(f"WARNING: {START_METHOD_ENV_VAR}={method!r} is not a "
              f"multiprocessing start method on this platform; using the "
              f"default ({mp.get_start_method()}).")
        return mp


class ManagerStartError(ConfigurationError):
    """:func:`measure_crop` could not start its :class:`multiprocessing.Manager`.

    A :class:`~spacr.errors.ConfigurationError` because it is not a per-field
    failure: the Manager owns the shared timing list every worker writes to, so
    if it will not start then no field can be measured and continuing past it
    produces nothing. The remedy is a configuration change
    (:data:`START_METHOD_ENV_VAR`), which is what this class exists to say.

    What it replaces is the point. ``ctx.Manager()`` fails as a bare
    ``EOFError`` raised four frames down in ``multiprocessing/connection.py``
    -- no message, no mention of spaCR, no mention of the start method, and no
    hint that the process that called Measure is the thing at fault. This was
    reproduced deterministically 33 times across 7 test modules; see
    :func:`_manager_start_diagnosis` for the mechanism.
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
        # get_start_method() is read here rather than passed in so the message
        # reports what the *failed* Manager was actually using, even if the
        # caller resolved a different name earlier.
        try:
            start_method = ctx.get_start_method()
        except Exception:
            start_method = mp.get_start_method()
        raise ManagerStartError(
            _manager_start_diagnosis(start_method, exc)) from exc


def resolve_pool_size(n_jobs, n_files, start_method=None):
    """Return how many worker processes to actually start for ``n_files`` fields.

    Under ``fork`` a surplus worker is nearly free -- it is a page-table copy of
    a process that has already imported everything -- so spaCR has always
    started exactly the requested number and existing behaviour is preserved.

    Under ``spawn`` and ``forkserver`` it is not free. Each worker is a fresh
    interpreter that re-imports the whole measure chain from scratch; measured
    on a developer box that was **8.1 s and ~1.54 GB of RSS per worker**, and
    is **3.5 s and ~930 MB** now that ``spacr.plot`` and umap (and, through
    umap, TensorFlow) are off that path. Either way it is paid before a single
    field is read. A default ``n_jobs`` of ``cpu_count - 4`` on a 16-core
    Windows machine therefore boots 12 interpreters and reserves 11-18 GB to
    measure a 4-field test plate, and the run either swaps itself to a
    standstill or has workers killed out from under it -- which presents as
    "Measure prints 'using 12 cpu cores' and then nothing happens", because a
    pool worker that dies at bootstrap is silently replaced and dies again.
    Windows and macOS default to ``spawn``; Linux does not, which is exactly
    why this only ever bit the other two.

    A worker with no field to measure cannot contribute, so capping at the
    number of fields costs nothing and is the whole fix.

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
    # max(1, ...) because Pool(0) raises ValueError, and a folder that turned
    # out to hold no unmeasured fields must finish quietly rather than crash.
    return max(1, min(n_jobs, int(n_files)))


def resolve_n_jobs(n_jobs, cpu_count=None):
    """Return the number of worker processes ``measure_crop`` will actually use.

    This used to discard the user's value outright. The old block compared
    ``n_jobs`` with the core count *before* the ``is None`` check -- so leaving
    it blank, which the printed warning itself recommends, raised
    ``TypeError: '>' not supported between instances of 'NoneType' and 'int'``
    -- and then ended with an unconditional ``settings['n_jobs'] =
    spacr_cores``, which threw the request away: on a 32-core machine
    ``n_jobs=1`` ran 28 workers. That is not a performance detail. It is what
    made the concurrent ``CREATE TABLE`` race reachable from a test that had
    explicitly asked for one worker, and it takes away the only lever a user
    has on a shared machine.

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
        # Blank: leave headroom, but never drop below one worker -- a machine
        # with four cores or fewer would otherwise get zero or a negative pool.
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

    **2-D returns ``(None, px stamp)`` unconditionally.** Even when a voxel size
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

    # Imported here rather than at module scope: spacr.zstack is the authority
    # on anisotropy and owns the error type, but measure.py must not pay for
    # importing it on the overwhelmingly common 2-D path.
    from .zstack import UnknownAnisotropyError

    def _positive(name):
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
        # Geometry is correct, units are xy pixels. Recorded as such: a
        # "volume" here is in cubic xy-pixels and is not a um^3.
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
    # Create mappings from each cell to its nucleus, pathogens, and cytoplasms
    cell_to_nucleus = defaultdict(list)
    cell_to_pathogen = defaultdict(list)
    # Get unique cell labels
    cell_labels = np.unique(cell_mask)
    # Iterate over each cell label
    for cell_id in cell_labels:
        if cell_id == 0:
            continue
        # Find corresponding component labels
        nucleus_ids = np.unique(nucleus_mask[cell_mask == cell_id])
        pathogen_ids = np.unique(pathogen_mask[cell_mask == cell_id])
        # Update dictionaries, ignoring 0 (background) labels
        cell_to_nucleus[cell_id] = nucleus_ids[nucleus_ids != 0].tolist()
        cell_to_pathogen[cell_id] = pathogen_ids[pathogen_ids != 0].tolist()
    # Convert dictionaries to dataframes
    nucleus_df = pd.DataFrame(list(cell_to_nucleus.items()), columns=['cell_id', 'nucleus'])
    pathogen_df = pd.DataFrame(list(cell_to_pathogen.items()), columns=['cell_id', 'pathogen'])
    # Explode lists
    # ``explode`` turns an empty child list into a row whose child key is NaN.
    # Those rows describe no relationship and duplicate the NaN merge key once
    # per parent, which violates the one-child/one-parent contract downstream.
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
        # mahotas' signature is zernike_moments(im, radius, degree=8): the
        # moments are computed on a disk of `radius` centred on the object's
        # centre of mass, and pixels outside that disk are ignored. Passing
        # `degree` positionally put the degree into `radius`, so every object
        # -- 20 px or 2000 px -- was described on a fixed 8 px disk and the
        # degree was always the default 8. Scaling the radius with the object
        # is what makes the coefficients comparable across object sizes.
        coords = np.argwhere(region.image)
        if coords.size == 0:
            radius = 1.0
        else:
            centre = coords.mean(axis=0)
            # Max distance from the centre of mass, which is exactly the centre
            # mahotas uses by default, so the disk covers the whole object.
            radius = float(np.sqrt(((coords - centre) ** 2).sum(axis=1)).max())
        radius = max(radius, 1.0)
        zernike_moment = zernike_moments(region.image, radius, degree=degree)
        zernike_features.append(zernike_moment.tolist())

    if zernike_features:
        feature_length = len(zernike_features[0])
        for feature in zernike_features:
            if len(feature) != feature_length:
                raise ValueError("All Zernike moments must be of the same length")

        zernike_df = pd.DataFrame(zernike_features, columns=[f'zernike_{i}' for i in range(feature_length)])
        return pd.concat([df.reset_index(drop=True), zernike_df], axis=1)
    return df


def _load_zernike_moments():
    """Load Mahotas only when its optional descriptor is computed."""
    try:
        from mahotas.features import zernike_moments
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Zernike morphology requires the optional Mahotas package. "
            "Install it with `pip install \"spacr[zernike]\"`, or run "
            "morphological measurements with zernike=False."
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

    # ``[..., channel]`` rather than ``[:, :, channel]``: identical for the
    # (Y, X, C) arrays this has always been given, and it selects the channel
    # rather than a slab of X should a (Z, Y, X, C) array ever reach here.
    image = array[..., channel]

    properties_list = []

    # Process each object in the mask based on its label
    for label in np.unique(mask):
        if label == 0:
            continue  # Skip background

        # Isolate the object using the label
        object_region = mask == label
        region_intensity = np.where(object_region, image, 0)  # Use np.where for more efficient masking

        # Ensure there are non-zero values to process
        if np.any(region_intensity):
            # Calculate adaptive offset based on intensity percentiles within the object
            valid_pixels = region_intensity[region_intensity > 0]
            if len(valid_pixels) > 1:  # Ensure there are enough pixels to compute percentiles
                offset = np.percentile(valid_pixels, 90) - np.percentile(valid_pixels, 50)
                block_size = 35  # Adjust this based on your object sizes and detail needs
                local_thresh = filters.threshold_local(region_intensity, block_size=block_size, offset=offset)
                cytoskeleton = region_intensity > local_thresh

                # Skeletonize the thresholded cytoskeleton
                skeleton = morphology.skeletonize(img_as_bool(cytoskeleton))

                # Measure properties of the skeleton
                skeleton_props = measure.regionprops(measure.label(skeleton), intensity_image=image)
                skeleton_length = sum(prop.area for prop in skeleton_props)  # Sum of lengths of all skeleton segments
                # Branch points are skeleton pixels with >= 3 skeleton
                # neighbours (the standard definition). The previous code
                # called morphology.skeleton_branch_analysis, which does not
                # exist in scikit-image and raised AttributeError whenever a
                # region had enough pixels to reach this branch.
                skel = skeleton.astype(np.uint8)
                neighbour_count = convolve(
                    skel, np.ones((3, 3), dtype=np.uint8),
                    mode='constant', cval=0) - skel
                n_branch_points = int(np.sum((skel == 1) & (neighbour_count >= 3)))

                # Store properties
                properties = {
                    "object_label": label,
                    "skeleton_length": skeleton_length,
                    "skeleton_branch_points": n_branch_points
                }
                properties_list.append(properties)
            else:
                # Handle cases with insufficient pixels
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

    # Added guarded columns belong where callers requested them, not at the
    # end.  (All morphology properties here are scalar columns.)
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
            # The duplicate is on the props side, which means regionprops_table
            # emitted a label twice -- a different fault with a different fix,
            # and one this message would misdescribe. Say nothing about cells.
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


def _morphological_measurements(cell_mask, nucleus_mask, pathogen_mask, organelle_mask, cytoplasm_mask, settings, zernike=None, degree=8):
    """Return morphology + Zernike DataFrames for cells, nuclei, pathogens, organelles, cytoplasm.

    :param cell_mask: Label mask of cells.
    :param nucleus_mask: Label mask of nuclei.
    :param pathogen_mask: Label mask of pathogens.
    :param organelle_mask: Label mask of organelles.
    :param cytoplasm_mask: Label mask of cytoplasm.
    :param settings: Settings dict; ``<object>_mask_dim`` keys drive whether
        each object type is analysed, ``cytoplasm`` toggles cytoplasm output.
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
    """
    if zernike is None:
        try:
            _load_zernike_moments()
        except ImportError as exc:
            zernike = False
            print(f"[measure] {exc} Zernike columns will be skipped.")
        else:
            zernike = True

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

    prop_ls = []
    ls = []

    if settings['cell_mask_dim'] is not None:
        cell_to_nucleus, cell_to_pathogen = get_components(cell_mask, nucleus_mask, pathogen_mask)
        cell_props = _props(cell_mask)
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
        if zernike:
            nucleus_props = _calculate_zernike(
                nucleus_mask, nucleus_props, degree=degree)
        if settings['cell_mask_dim'] is not None:
            # one_to_one; see _join_child_to_parent_cell for why, and for what
            # was tried instead. Briefly: this was relaxed to one_to_many on
            # the theory that a nucleus straddling two touching cells is a
            # legitimate shape. It is not one measurements.db can store -- the
            # nucleus table is keyed one row per object_label per field -- and
            # relaxing it here only moved the same MergeError downstream into
            # _merge_and_save_to_database, after the cell table for this field
            # had already been committed. Backed out: fail before the first
            # write, with a message that names the offending labels.
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
        if zernike:
            pathogen_props = _calculate_zernike(
                pathogen_mask, pathogen_props, degree=degree)
        if settings['cell_mask_dim'] is not None:
            # one_to_one, for the same reasons as the nucleus join above. This
            # is the join the fan-out actually reaches, because the mask repair
            # that prevents it is optional here: with
            # merge_edge_pathogen_cells=False, a vacuole on the border between
            # two host cells is listed under both cell_ids. That still cannot
            # be stored -- the pathogen table is one row per object_label with
            # one cell_id -- so it stops here, before the cell and nucleus
            # tables for this field are written, rather than after.
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

    if settings.get('organelle_mask_dim') is not None:
        organelle_props = _props(organelle_mask)
        if len(organelle_props) > 0 and zernike:
            organelle_props = _calculate_zernike(organelle_mask, organelle_props, degree=degree)
        if len(organelle_props) > 0:
            # Map each organelle to its parent cell
            if settings['cell_mask_dim'] is not None:
                organelle_to_cell = _map_child_to_parent(organelle_mask, cell_mask, child_name='organelle', parent_name='cell')
                # one_to_one here, unlike the nucleus/pathogen joins above, and
                # the difference is in the mapper rather than the biology:
                # _map_child_to_parent resolves each organelle to its single
                # maximum-overlap parent, so it emits exactly one row per
                # organelle label. Both sides are therefore keyed uniquely and
                # a duplicate on either would mean the mapper itself is broken.
                organelle_props = pd.merge(
                    organelle_props,
                    organelle_to_cell,
                    left_on='label',
                    right_on='organelle',
                    how='left',
                    validate='one_to_one',
                )
        prop_ls.append(organelle_props)
        ls.append('organelle')
    else:
        prop_ls.append(pd.DataFrame())
        ls.append('organelle')

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
 
    return df_ls[0], df_ls[1], df_ls[2], df_ls[3], df_ls[4]

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

    # Get per-organelle morphology
    organelle_df = _safe_morphology_table(
        organelle_mask, properties=morphological_props, spacing=spacing)

    # Map each organelle to its parent
    organelle_to_parent = _map_child_to_parent(organelle_mask, parent_mask, 
                                                child_name='organelle_label', 
                                                parent_name=parent_name)
    
    if len(organelle_df) > 0 and len(organelle_to_parent) > 0:
        # one_to_one: both frames are derived from the same organelle_mask and
        # both carry one row per label -- regionprops_table on the left,
        # _map_child_to_parent's single argmax-overlap parent on the right. A
        # duplicate on either side means one of those two invariants broke, and
        # the per-parent sums computed below would then double-count organelles.
        organelle_df = pd.merge(
            organelle_df,
            organelle_to_parent,
            left_on='label',
            right_on='organelle_label',
            how='left',
            validate='one_to_one',
        )
    else:
        # No organelles — return empty summary for all parents
        rows = []
        for pid in parent_labels:
            row = {'label': pid, 'organelle_count': 0, 'organelle_total_area': 0, 
                   'organelle_fraction': 0.0}
            rows.append(row)
        return pd.DataFrame(rows)

    # Per-channel intensity per organelle
    for ch in range(channel_arrays.shape[-1]):
        channel = channel_arrays[..., ch]
        intensities = []
        for org_label in organelle_df['label']:
            region = organelle_mask == org_label
            if np.any(region):
                intensities.append(channel[region].mean())
            else:
                intensities.append(0.0)
        # 'channel_<c>', not 'ch<c>'. Every other feature family in the
        # database spells it out; this one did not, so the same idea had two
        # names. utils.rename_columns_in_db migrates the old spelling.
        organelle_df[f'organelle_channel_{ch}_mean_intensity'] = intensities

    # Get parent areas for fraction calculation
    parent_props = pd.DataFrame(regionprops_table(parent_mask, properties=['label', 'area'], spacing=spacing))
    parent_area_map = dict(zip(parent_props['label'], parent_props['area']))

    # Summarise per parent
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

def _intensity_measurements(cell_mask, nucleus_mask, pathogen_mask, organelle_mask, cytoplasm_mask, channel_arrays, settings, sizes=None, periphery=True, outside=True):
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
        # Refused rather than approximated: graycomatrix takes a 2-D image
        # only, and running it on one plane (or on a reshaped volume, which is
        # what an unguarded call does) would report the texture of an
        # arbitrary slice under a column name that promises the object's.
        print("3-D mask: skipping GLCM homogeneity — "
              "skimage.feature.graycomatrix is defined for 2-D images only, "
              "so no homogeneity_distance_* columns are written for this field.")
        homogeneity = False

    intensity_props = ["label", "centroid_weighted", "centroid_weighted_local", "max_intensity", "mean_intensity", "min_intensity"]
    # 'percentile_<p>', not '<p>_percentile'. The object interior has always
    # been written 'percentile_5' (_extended_regionprops_table); the periphery
    # and outside rings used the reversed word order, so one database carried
    # 'cell_channel_0_percentile_5' next to
    # 'nucleus_channel_0_periphery_5_percentile' for the same statistic.
    # utils.rename_columns_in_db migrates the old spelling on first read.
    col_lables = ['region_label', 'mean', 'percentile_5', 'percentile_10', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_85', 'percentile_95']
    cell_dfs, nucleus_dfs, pathogen_dfs, organelle_dfs, cytoplasm_dfs = [], [], [], [], []
    ls = ['cell', 'nucleus', 'pathogen', 'organelle', 'cytoplasm']
    labels = [cell_mask, nucleus_mask, pathogen_mask, organelle_mask, cytoplasm_mask]
    dfs = [cell_dfs, nucleus_dfs, pathogen_dfs, organelle_dfs, cytoplasm_dfs]
    
    for i in range(0, channel_arrays.shape[-1]):
        channel = channel_arrays[..., i]
        for j, (label, df) in enumerate(zip(labels, dfs)):

            if np.max(label) == 0:
                empty_df = pd.DataFrame()
                df.append(empty_df)
                continue

            mask_intensity_df = _extended_regionprops_table(label, channel, intensity_props, spacing=spacing)

            if homogeneity:
                homogeneity_df = _calculate_homogeneity(label, channel, distances)
                mask_intensity_df = pd.concat([mask_intensity_df.reset_index(drop=True), homogeneity_df], axis=1)

            if periphery:
                if ls[j] in ('nucleus', 'pathogen', 'organelle'):
                    periphery_intensity_stats = _periphery_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(periphery_intensity_stats, columns=[f'periphery_{stat}' for stat in col_lables])], axis=1)

            if outside:
                if ls[j] in ('nucleus', 'pathogen', 'organelle'):
                    outside_intensity_stats = _outside_intensity(label, channel, spacing=spacing)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(outside_intensity_stats, columns=[f'outside_{stat}' for stat in col_lables])], axis=1)

            # Measure focus on the object's 2-D patch, not on the 1-D vector of
            # its pixels. The column is named 'blur' bare: the loop below adds
            # the '<object>_channel_<i>_' prefix to every non-label column, and
            # writing the prefix here too produced
            # 'cell_channel_0_cell_channel_0_blur' in every database written
            # before this fix.
            blur_col = [_estimate_blur(channel, mask=(label == region_label))
                        for region_label in mask_intensity_df['label']]
            mask_intensity_df['blur'] = blur_col

            mask_intensity_df.columns = [f'{ls[j]}_channel_{i}_{col}' if col != 'label' else col for col in mask_intensity_df.columns]
            df.append(mask_intensity_df)
            
    if isinstance(settings['distance_gaussian_sigma'], int):
        if settings['distance_gaussian_sigma'] != 0:
            if settings['cell_mask_dim'] is not None:
                if settings['nucleus_mask_dim'] is not None or settings['pathogen_mask_dim'] is not None:
                    intensity_distance_df = _measure_intensity_distance(cell_mask, nucleus_mask, pathogen_mask, channel_arrays, settings)
                    cell_dfs.append(intensity_distance_df)
    
    if radial_dist:
        if np.max(nucleus_mask) != 0:
            nucleus_radial_distributions = _calculate_radial_distribution(cell_mask, nucleus_mask, channel_arrays, num_bins=6, spacing=spacing)
            nucleus_df = _create_dataframe(nucleus_radial_distributions, 'nucleus')
            dfs[1].append(nucleus_df)

        if np.max(pathogen_mask) != 0:
            pathogen_radial_distributions = _calculate_radial_distribution(cell_mask, pathogen_mask, channel_arrays, num_bins=6, spacing=spacing)
            pathogen_df = _create_dataframe(pathogen_radial_distributions, 'pathogen')
            dfs[2].append(pathogen_df)

        if np.max(organelle_mask) != 0:
            organelle_radial_distributions = _calculate_radial_distribution(cell_mask, organelle_mask, channel_arrays, num_bins=6, spacing=spacing)
            organelle_rad_df = _create_dataframe(organelle_radial_distributions, 'organelle')
            dfs[3].append(organelle_rad_df)

    # The parent-cell link must exist whether or not radial_dist ran. It used
    # to arrive ONLY as a side effect of _create_dataframe, so with
    # radial_dist=False the nucleus/pathogen/organelle tables lost cell_id
    # entirely and _merge_and_save_to_database silently dropped it from the
    # key columns. Build it from the masks instead, and strip the radial
    # frame's copy so exactly one frame supplies it (two would collide as
    # cell_id_x / cell_id_y in the morphology/intensity merge).
    if settings.get('cell_mask_dim') is not None and np.max(cell_mask) != 0:
        for idx, child_mask in ((1, nucleus_mask), (2, pathogen_mask), (3, organelle_mask)):
            if np.max(child_mask) == 0:
                continue
            for existing in dfs[idx]:
                if 'cell_id' in existing.columns:
                    existing.drop(columns=['cell_id'], inplace=True)
            parent_link = _map_child_to_parent(child_mask, cell_mask,
                                               child_name='label',
                                               parent_name='cell_id')
            # _map_child_to_parent uses 0 for "no overlapping cell". There is no
            # cell 0, and the column this replaces was NaN in that case (an
            # object outside every cell simply had no row in the radial frame),
            # so keep NaN and keep the column's float dtype.
            parent_link['cell_id'] = parent_link['cell_id'].astype(float).replace(0.0, np.nan)
            dfs[idx].append(parent_link.reset_index(drop=True))

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
    
    return (pd.concat(cell_dfs, axis=1), 
            pd.concat(nucleus_dfs, axis=1), 
            pd.concat(pathogen_dfs, axis=1), 
            pd.concat(organelle_dfs, axis=1),
            pd.concat(cytoplasm_dfs, axis=1))
    
def _create_dataframe(radial_distributions, object_type):
        """Convert a ``{(cell, obj, ch): bins}`` mapping into a per-object DataFrame."""
        df = pd.DataFrame()
        for key, value in radial_distributions.items():
            cell_label, object_label, channel_index = key
            for i in range(len(value)):
                col_name = f'{object_type}_rad_dist_channel_{channel_index}_bin_{i}'
                df.loc[object_label, col_name] = value[i]
            df.loc[object_label, 'cell_id'] = cell_label
        # Reset the index and rename the column that was previously the index
        df = df.reset_index().rename(columns={'index': 'label'})
        return df

def _extended_regionprops_table(labels, image, intensity_props, spacing=None):
    """Return a regionprops table extended with distributional intensity features (mean/std/skew/kurtosis/mode/CV/Gini/entropy/percentiles).

    :param labels: label mask, 2-D or 3-D.
    :param image: co-aligned intensity image.
    :param intensity_props: regionprops property names.
    :param spacing: voxel spacing from :func:`resolve_measurement_spacing`;
        ``None`` in 2-D, which skimage treats as "not supplied".
    """

    def _gini(array):
        """NaN-safe Gini coefficient of an intensity array."""
        # Compute Gini coefficient (nan safe)
        array = np.abs(array[~np.isnan(array)])
        n = array.size
        if n == 0:
            return np.nan
        array = np.sort(array)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)) if np.sum(array) else np.nan
    
    props = regionprops_table(labels, image, properties=intensity_props, spacing=spacing)
    df = pd.DataFrame(props)
    if _ndim_of(labels) == 3:
        df = _rename_3d_centroids(df)

    # Reference thresholds for frac_high90 / frac_low10.
    #
    # These used to be thresholded on the object's OWN 90th/10th percentile,
    # which makes them 0.10 for any continuous distribution by construction —
    # they reported quantisation and ties, not brightness. Thresholding on the
    # whole field's percentiles instead gives what the names promise: the
    # fraction of the object that is bright (or dim) relative to this field.
    # A dim object scores near 0 for frac_high90, a bright one near 1.
    _field = np.asarray(image, dtype=float).ravel()
    _field = _field[~np.isnan(_field)]
    if _field.size:
        field_p90 = float(np.percentile(_field, 90))
        field_p10 = float(np.percentile(_field, 10))
    else:
        field_p90 = np.nan
        field_p10 = np.nan

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
        except AttributeError:  # scikit-image 0.22-0.25
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
            # scipy.stats deliberately returns NaN for a constant sample, but
            # first emits a RuntimeWarning about catastrophic cancellation.
            # Uniform segmented objects are ordinary image data, so take the
            # mathematically-defined shortcut and keep measurement logs clean.
            has_variation = not np.all(intens == intens[0])
            integrated_intensity.append(np.sum(intens))
            std_intensity.append(np.std(intens))
            median_intensity.append(np.median(intens))
            skew_intensity.append(
                skew(intens) if intens.size > 2 and has_variation else np.nan)
            kurtosis_intensity.append(
                kurtosis(intens) if intens.size > 3 and has_variation else np.nan)
            # Mode (use the smallest mode value if multimodal).
            # SciPy < 1.11 returned a 1-element array here, SciPy >= 1.11
            # returns a bare scalar. The old code did `mode_val[0]`, which
            # raises IndexError on a scalar, and a bare `except` turned that
            # into NaN — so on the installed SciPy (1.15) mode_intensity was
            # NaN for every object in every database. atleast_1d handles both.
            mode_val = np.atleast_1d(np.asarray(mode(intens, nan_policy='omit').mode))
            mode_intensity.append(float(mode_val[0]) if mode_val.size else np.nan)
            range_intensity.append(np.ptp(intens))
            iqr_intensity.append(np.percentile(intens, 75) - np.percentile(intens, 25))
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
    for p in percentiles:
        df[f'percentile_{p}'] = [
            np.percentile(_masked_intensity(region), p)
            for region in regions
        ]
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
        # Iterate through the regions in label_mask
        for region in regionprops(label):
            region_image = (region.image * channel[region.slice]).astype(int)
            homogeneity_per_distance = []
            for d in distances:
                rescaled_image = rescale_intensity(region_image, out_range=(0, 255)).astype('uint8')
                glcm = graycomatrix(rescaled_image, [d], [0], symmetric=True, normed=True)
                homogeneity_per_distance.append(graycoprops(glcm, 'homogeneity')[0, 0])
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
    for region in np.unique(label_mask)[1:]:  # skip the background label
        region_boundary = boundary & (label_mask == region)
        intensities = image[region_boundary]
        if intensities.size == 0:
            periphery_intensity_stats.append((region, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            periphery_intensity_stats.append((region, np.mean(intensities), np.percentile(intensities,5), np.percentile(intensities,10),
                                              np.percentile(intensities,25), np.percentile(intensities,50),
                                              np.percentile(intensities,75), np.percentile(intensities,85), 
                                              np.percentile(intensities,95)))
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
    for region in np.unique(label_mask)[1:]:  # skip the background label
        region_mask = label_mask == region
        if spacing is None:
            dilated_mask = binary_dilation(region_mask, iterations=distance)
        else:
            edt = distance_transform_edt(~region_mask, sampling=spacing)
            dilated_mask = edt <= ring_width
        outside_mask = dilated_mask & ~region_mask
        intensities = image[outside_mask]
        if intensities.size == 0:
            outside_intensity_stats.append((region, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            outside_intensity_stats.append((region, np.mean(intensities), np.percentile(intensities,5), np.percentile(intensities,10),
                                              np.percentile(intensities,25), np.percentile(intensities,50),
                                              np.percentile(intensities,75), np.percentile(intensities,85), 
                                              np.percentile(intensities,95)))
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
            # Degenerate: the cell is a single shell at distance 0. Everything
            # belongs to the innermost bin.
            radial_distribution[0] = single_channel_image[region_mask].mean()
            return radial_distribution
        for i in range(num_bins):
            min_distance = i * (max_distance / num_bins)
            max_distance_i = (i + 1) * (max_distance / num_bins)
            bin_mask = region_mask & (distance_map >= min_distance)
            # The final bin is closed so the farthest pixel is not dropped.
            if i == num_bins - 1:
                bin_mask &= (distance_map <= max_distance_i)
            else:
                bin_mask &= (distance_map < max_distance_i)
            if bin_mask.any():
                radial_distribution[i] = single_channel_image[bin_mask].mean()
        return radial_distribution


    object_radial_distributions = {}

    # get unique cell labels
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels != 0]

    for cell_label in cell_labels:
        cell_region = cell_mask == cell_label

        object_labels = np.unique(object_mask[cell_region])
        object_labels = object_labels[object_labels != 0]

        for object_label in object_labels:
            objecyt_region = object_mask == object_label
            object_boundary = find_boundaries(objecyt_region, mode='outer')
            # NOT multiplied by cell_region: that zeroed the distance of every
            # pixel outside the cell and put the whole background in bin 0.
            # The cell is applied as a mask when binning instead.
            distance_map = distance_transform_edt(~object_boundary, sampling=spacing)
            for channel_index in range(channel_arrays.shape[-1]):
                radial_distribution = _calculate_average_intensity(distance_map, channel_arrays[..., channel_index], num_bins, cell_region)
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
        """
        thresholds = settings['manders_thresholds']

        corr_data = {}
        for i in np.unique(mask)[1:]:
            object_mask = (mask == i)
            object_channel_image1 = channel_image1[object_mask]
            object_channel_image2 = channel_image2[object_mask]
            total_intensity1 = np.sum(object_channel_image1)
            total_intensity2 = np.sum(object_channel_image2)

            if len(object_channel_image1) < 2 or len(object_channel_image2) < 2:
                pearson_corr = np.nan
            else:
                pearson_corr, _ = pearsonr(object_channel_image1, object_channel_image2)

            corr_data[i] = {f'label_correlation': i,
                            f'Pearson_correlation': pearson_corr}

            for thresh in thresholds:
                chan1_thresh = np.percentile(object_channel_image1, thresh)
                chan2_thresh = np.percentile(object_channel_image2, thresh)

                # boolean mask where both signals are present
                overlap_mask = (object_channel_image1 > chan1_thresh) & (object_channel_image2 > chan2_thresh)
                M1 = np.sum(object_channel_image1[overlap_mask]) / total_intensity1 if total_intensity1 > 0 else 0
                M2 = np.sum(object_channel_image2[overlap_mask]) / total_intensity2 if total_intensity2 > 0 else 0

                corr_data[i].update({f'M1_correlation_{thresh}': M1,
                                     f'M2_correlation_{thresh}': M2})

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
        # Bounding box grown by one pixel in y and x so the 3x3 kernel has real
        # neighbours. Not grown in z: the kernel never reaches across planes.
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
            # In-plane erosion only, matching the in-plane kernel: a voxel is
            # interior when its eight xy neighbours in its own plane are all in
            # the object. A 3-D structuring element would additionally require
            # the planes above and below, which no sample of the kernel reads.
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

    # cv2.Laplacian with CV_64F requires a float64 source: float32 (and any
    # integer) inputs raise "Unsupported combination of source/destination
    # format", so promote anything that isn't already float64.
    if image.dtype != np.float64:
        image_float = image.astype(np.float64)
    else:
        # Already float64 — use as is.
        image_float = image
    # Compute the Laplacian of the image
    if volumetric:
        lap = np.empty(image_float.shape, dtype=np.float64)
        for z in range(image_float.shape[0]):
            lap[z] = cv2.Laplacian(
                np.ascontiguousarray(image_float[z]), cv2.CV_64F)
    else:
        lap = cv2.Laplacian(image_float, cv2.CV_64F)
    # Compute and return the variance of the Laplacian
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
        # sigma is quoted in xy pixels; convert to the same physical length on
        # every axis. With spacing (dz, dxy, dxy) the z sigma is sigma*dxy/dz,
        # i.e. fewer planes for the same distance.
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

    # Merge all channel dataframes on label. one_to_one: every frame in `dfs`
    # was built by walking the same `cell_labels` (np.unique of the cell mask)
    # once, so each holds one row per cell and the same set of cells. This is a
    # widening of one table across channels, not a relationship between two
    # different object types -- if a label repeated, a channel's distances
    # would be silently averaged over duplicate rows downstream.
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(
            df, on='label', how='outer', validate='one_to_one')

    return merged_df

def save_and_add_image_to_grid(png_channels, img_path, grid, plot=False):
    """
    Add an image to a grid and save it as PNG.

    Args:
        png_channels (ndarray): The array representing the image channels.
        img_path (str): The path to save the image as PNG.
        grid (list): The grid of images to be plotted later.

    Returns:
        grid (list): Updated grid with the new image added.

    .. note::

       **The file's colour slots hold what the mapping declares.** The
       caller assembles ``png_channels`` in file order — red plane first —
       with :func:`spacr.crops.build_png_channels` and
       :func:`spacr.crops.resolve_png_channel_mapping`; under the legacy
       ``settings['png_dims']`` list that mapping is entry 0 blue, 1 green,
       2 red, so ``png_dims[0]`` lands in the file's BLUE slot.

       ``cv2.imwrite`` interprets a 3-channel array as BGR, so
       :func:`spacr.crops.to_cv2_bgr` reverses the array once, here, and
       cv2's interpretation lands the red plane in the file's red slot. It
       refuses more than three channels rather than letting cv2 write the
       fourth as an alpha plane for every reader to drop in silence.

       The format is versioned: :func:`spacr.crops.stamp_crop_folder` drops a
       ``.spacr_crop_format.json`` sidecar into the crop folder before the
       first PNG lands, recording format 3 (``declared_rgb``). An unmarked
       folder means format 1 (legacy), whose bytes match format 3 for the
       same declared mapping, so both are read as-is; only format 2 — written
       between 2026-07-26 and 2026-08-06 — is reversed by
       :func:`spacr.crops.read_crop_png`, and
       ``spacr.crops.migrate_crop_folder`` rewrites such a folder in place.

       Crops are still ``uint16``, so these are 16-bit PNGs and no intensity
       is discarded at write time. The narrowing to 8 bit happens once, on
       read, in :func:`spacr.crops.narrow_to_uint8`, which always takes the
       HIGH BYTE (``// 256``) — replacing PIL's two incompatible rules (high
       byte for an RGB PNG, a *clip* at 255 for a single-channel one, which
       returned solid white for any crop brighter than that).
    """

    # Mark the folder BEFORE the first PNG lands: a run killed in between then
    # leaves a marked folder holding fewer crops, never an unmarked folder of
    # corrected ones, which is the single state that would be misread as
    # legacy. Costs one stat per folder per process.
    stamp_crop_folder(os.path.dirname(img_path))
    cv2.imwrite(img_path, to_cv2_bgr(png_channels))

    if plot:

        # Ensure the image is in uint8 format for cv2 functions
        if png_channels.dtype == np.uint16:
            png_channels = (png_channels / 256).astype(np.uint8)
        
        # Add the image to the diagnostic grid.
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
    
    fig, axs = plt.subplots(
        grid_size, grid_size, figsize=(15, 15), facecolor='black',
        squeeze=False)
    
    from matplotlib.patches import FancyBboxPatch
    for i, ax in enumerate(axs.flat):
        if i < n_images:
            image = grid[i]
            # Grid entries are produced from ``png_dims`` in RGB order.  The
            # OpenCV reversal belongs only at the PNG write boundary above.
            im = ax.imshow(image)
            ax.axis('off')
            ax.set_facecolor('black')

            # Clip each crop to a rounded rectangle so the grid reads like the
            # annotate view (soft corners) rather than hard square tiles.
            h, w = image.shape[:2]
            r = max(2.0, min(h, w) * 0.08)
            bbox = FancyBboxPatch(
                (0, 0), w - 1, h - 1,
                boxstyle=f"round,pad=0,rounding_size={r}",
                transform=ax.transData, facecolor='none', edgecolor='none')
            ax.add_patch(bbox)
            im.set_clip_path(bbox)

            if titles:
                # Determine text size
                img_height, img_width = image.shape[:2]
                text_size = max(min(img_width / (len(titles[i]) * 1.5), img_height / 10), 4)
                ax.text(5, 5, titles[i], color='white', fontsize=text_size, ha='left', va='top', fontweight='bold')
        else:
            fig.delaxes(ax)

    # A little more breathing room between crops.
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    plt.tight_layout(pad=0.2)
    return fig


#: crop_mode entries that name a mask _measure_crop_core knows how to crop.
CROP_MODES = ('cell', 'nucleus', 'pathogen', 'cytoplasm', 'organelle')


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
        # crop_mode is empty, so nothing is cropped and nothing indexes this.
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
MASK_DIM_KEYS = ('cell_mask_dim', 'nucleus_mask_dim', 'pathogen_mask_dim',
                 'organelle_mask_dim')


def _merged_mask_planes(data, settings):
    """Return the set of plane indices of ``data`` that hold labels, not signal."""
    n_planes = int(data.shape[-1])
    planes = set()
    for key in MASK_DIM_KEYS:
        dim = settings.get(key)
        if dim is None:
            continue
        try:
            dim = int(dim)
        except (TypeError, ValueError):
            continue
        if 0 <= dim < n_planes:
            planes.add(dim)
    return planes


def _promote_merged_to_uint16(data, settings):
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
        if top > 0:
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


#@log_function_call
def _measure_crop_core(index, time_ls, file, settings):

    """
    Measure and crop the images based on specified settings.

    Parameters:
    - index: int
        The index of the image.
    - time_ls: list
        The list of time points.
    - file: str
        The file path of the image.
    - settings: dict
        The dictionary containing the settings for measurement and cropping.

    Returns:
    - cropped_images: list
        A list of cropped images.
    """
    
    # spacr.plot is imported where it is used, not here: it is only reachable
    # behind settings['plot'], and under a spawn/forkserver pool every worker
    # pays for every import in this function from a cold interpreter. Measured
    # on a developer box, spacr.plot alone is ~1.9 s and ~720 MB per worker --
    # for a default run that never draws anything.
    from .utils import _merge_overlapping_objects, _filter_object, _relabel_parent_with_child_labels, _exclude_objects, normalize_to_dtype, filepaths_to_database
    from .utils import _merge_and_save_to_database, _crop_center, _find_bounding_box, _generate_names, _get_percentiles

    figs = {}
    grid = []
    start = time.time() 
    try:
        source_folder = os.path.dirname(settings['src'])

        file_name = os.path.splitext(file)[0]
        data = np.load(os.path.join(settings['src'], file))
        data_type = data.dtype
        if data_type not in ['uint8','uint16']:
            data_type_before = data_type
            data, factor = _promote_merged_to_uint16(data, settings)
            data_type = data.dtype
            if settings['verbose']:
                scale = '' if factor == 1.0 else f' (intensity x{factor:g})'
                print(f'Converted data from {data_type_before} to {data_type}{scale}')

        # A merged 2-D field is (Y, X, C); a merged z-stack is (Z, Y, X, C).
        # Every slice below therefore indexes the LAST axis -- `data[..., k]` --
        # which is exactly `data[:, :, k]` for a 3-D array and the channel
        # rather than a slab of X for a 4-D one. The old `data[:, :, channels]`
        # on a (Z, Y, X, C) array returned shape (Z, Y, len(channels), C): it
        # sliced X, kept every channel, and raised nothing, so the entire run
        # was measuring an arbitrary three-pixel-wide strip of the field.
        if data.ndim == 4 and data.shape[0] == 1:
            # A one-plane "volume" is a 2-D field. Squeezing it here means it
            # takes the ordinary 2-D path, so it measures identically to the
            # same field saved without a z axis, and needs no anisotropy.
            data = data[0]
        volumetric = data.ndim == 4
        n_z = int(data.shape[0]) if volumetric else 1
        # Raises when a 3-D field arrives without a voxel size or anisotropy,
        # which the caller records on the run ledger. Done before any
        # measurement so the run stops rather than half-filling a table.
        spacing, units_stamp = resolve_measurement_spacing(
            settings, 3 if volumetric else 2, n_z=n_z)

        if settings['plot'] and volumetric:
            # spacr.plot._plot_cropped_arrays lays out one panel per slice of a
            # (Y, X, C) array; handed a 4-D array it would either raise or plot
            # a slice of X as if it were a channel.
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

        # PREPROCESSING EXTENSION POINT. Registered hooks see exactly the
        # array the intensity measurements see: the channels named by
        # settings['channels'], already selected out of the merged stack, and
        # not one feature computed yet. This is where a flat-field /
        # illumination correction belongs. The PNG crops below are cut from
        # `data` and are deliberately NOT rewritten, so the thumbnails stay a
        # faithful record of what the microscope wrote while the numbers in
        # measurements.db carry the correction.
        #
        # The `if` is not just a micro-optimisation: with an empty registry
        # nothing is allocated and channel_arrays is the identical object, so
        # the default path cannot differ from the pre-hook one.
        if preprocessing_hooks():
            channel_arrays = apply_preprocessing_hooks(
                channel_arrays,
                PreprocessingContext(
                    file_name=file_name,
                    channels=settings['channels'],
                    settings=settings,
                    volumetric=volumetric,
                    spacing=spacing))

        if settings['cell_mask_dim'] is not None:
            cell_mask = data[..., settings['cell_mask_dim']].astype(data_type)

            if settings['cell_min_size'] is not None and settings['cell_min_size'] != 0:
                cell_mask = _filter_object(cell_mask, settings['cell_min_size'])
        else:
            cell_mask = np.zeros_like(data[..., 0])
            settings['cytoplasm'] = False
            settings['uninfected'] = True

        if settings['nucleus_mask_dim'] is not None:
            nucleus_mask = data[..., settings['nucleus_mask_dim']].astype(data_type)
            if settings['cell_mask_dim'] is not None:
                nucleus_mask, cell_mask = _merge_overlapping_objects(mask1=nucleus_mask, mask2=cell_mask)
            if settings['nucleus_min_size'] is not None and settings['nucleus_min_size'] != 0:
                nucleus_mask = _filter_object(nucleus_mask, settings['nucleus_min_size'])
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
            if settings['pathogen_min_size'] is not None and settings['pathogen_min_size'] != 0:
                pathogen_mask = _filter_object(pathogen_mask, settings['pathogen_min_size'])
        else:
            pathogen_mask = np.zeros_like(data[..., 0])

        if settings.get('organelle_mask_dim') is not None:
            organelle_mask = data[..., settings['organelle_mask_dim']].astype(data_type)
            if settings.get('organelle_min_size') and settings['organelle_min_size'] != 0:
                organelle_mask = _filter_object(organelle_mask, settings['organelle_min_size'])
        else:
            organelle_mask = np.zeros_like(data[..., 0])

        # Create cytoplasm mask
        if settings['cytoplasm']:
            if settings['cell_mask_dim'] is not None:
                # Build a combined interior mask from all subcellular objects
                interior = np.zeros_like(cell_mask, dtype=bool)
                if settings['nucleus_mask_dim'] is not None:
                    interior |= (nucleus_mask != 0)
                if settings['pathogen_mask_dim'] is not None:
                    interior |= (pathogen_mask != 0)
                if settings.get('organelle_mask_dim') is not None:
                    interior |= (organelle_mask != 0)
                cytoplasm_mask = np.where(interior, 0, cell_mask)
            else:
                cytoplasm_mask = np.zeros_like(cell_mask)
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
        
        if settings.get('organelle_min_size') and settings['organelle_min_size'] != 0:
            organelle_mask = _filter_object(organelle_mask, settings['organelle_min_size'])

        # REGION-FILTER EXTENSION POINT. Registered filters are handed the
        # label ids of each object type (and, only if they ask, the centroids)
        # and return a keep/drop boolean per object; a dropped label is zeroed
        # out of its mask right here. This is where a user-drawn ROI belongs:
        # "only measure inside this polygon".
        #
        # The position is load-bearing in two directions.
        #
        # Downstream: every size filter has already run, so a filter sees the
        # objects that would actually have been measured -- and nothing has
        # been measured yet, so keeping 5 of 500 objects costs 5 objects' worth
        # of morphology, intensity, texture, radial-distribution and Zernike
        # work rather than 500 followed by a DataFrame subset.
        #
        # Upstream of _exclude_objects: culling a cell there propagates to its
        # nucleus/pathogen/cytoplasm (they are multiplied by the surviving cell
        # mask), which is what keeps the validate='one_to_one' parent joins in
        # _morphological_measurements satisfiable. Filtering after it would let
        # an ROI keep a nucleus whose cell it had just deleted.
        #
        # It is also upstream of the `data[..., <mask>_dim] = ...` write-backs,
        # so the PNG crops and region arrays cover the same objects the
        # database does -- an object outside the ROI is not measured AND not
        # cropped, rather than appearing in one output and not the other.
        if region_filter_hooks():
            _region_masks = {
                'cell': cell_mask, 'nucleus': nucleus_mask,
                'pathogen': pathogen_mask, 'organelle': organelle_mask,
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
            organelle_mask = _region_masks['organelle']
            cytoplasm_mask = _region_masks['cytoplasm']

        if settings['cell_mask_dim'] is not None and settings['nucleus_mask_dim'] is not None and settings['pathogen_mask_dim'] is not None:
            cell_mask, nucleus_mask, pathogen_mask, cytoplasm_mask = _exclude_objects(cell_mask, nucleus_mask, pathogen_mask, cytoplasm_mask, uninfected=settings['uninfected'])
            data[..., settings['cell_mask_dim']] = cell_mask.astype(data_type)

        if settings['nucleus_mask_dim'] is not None:
            data[..., settings['nucleus_mask_dim']] = nucleus_mask.astype(data_type)
        if settings['pathogen_mask_dim'] is not None:
            data[..., settings['pathogen_mask_dim']] = pathogen_mask.astype(data_type)
        if settings['cytoplasm']:
            data = np.concatenate((data, cytoplasm_mask[..., np.newaxis]), axis=-1)

        if settings['plot'] and not volumetric:
            from .plot import _plot_cropped_arrays
            fig = _plot_cropped_arrays(data, file, figuresize)
            figs[f'{file_name}__after_filtration'] = fig


        if settings['save_measurements']:
            cell_df, nucleus_df, pathogen_df, organelle_df, cytoplasm_df = _morphological_measurements(cell_mask, nucleus_mask, pathogen_mask, organelle_mask, cytoplasm_mask, settings)

            cell_intensity_df, nucleus_intensity_df, pathogen_intensity_df, organelle_intensity_df, cytoplasm_intensity_df = _intensity_measurements(cell_mask, nucleus_mask, pathogen_mask, organelle_mask, cytoplasm_mask, channel_arrays, settings, sizes=[1, 2, 3, 4, 5], periphery=True, outside=True)
                
            if settings['cell_mask_dim'] is not None:
                _ = _merge_and_save_to_database(cell_df, cell_intensity_df, 'cell', source_folder, file_name, settings['experiment'], settings['timelapse'], stamp=units_stamp)
            if settings['nucleus_mask_dim'] is not None:
                _ = _merge_and_save_to_database(nucleus_df, nucleus_intensity_df, 'nucleus', source_folder, file_name, settings['experiment'], settings['timelapse'], stamp=units_stamp)

            if settings['pathogen_mask_dim'] is not None:
                _ = _merge_and_save_to_database(pathogen_df, pathogen_intensity_df, 'pathogen', source_folder, file_name, settings['experiment'], settings['timelapse'], stamp=units_stamp)

            if settings.get('summarize_organelles_by') is not None:
                if "organelle" in settings['summarize_organelles_by']:
                    if settings.get('organelle_mask_dim') is not None:
                        _ = _merge_and_save_to_database(organelle_df, organelle_intensity_df, 'organelle', source_folder, file_name, settings['experiment'], settings['timelapse'], stamp=units_stamp)

            if settings['cytoplasm']:
                _merge_and_save_to_database(cytoplasm_df, cytoplasm_intensity_df, 'cytoplasm', source_folder, file_name, settings['experiment'], settings['timelapse'], stamp=units_stamp)

            if settings.get('summarize_organelles_by') is not None:
                if "cell" in settings['summarize_organelles_by']:
                    if settings.get('organelle_mask_dim') is not None and np.max(organelle_mask) > 0:
                        if settings['cell_mask_dim'] is not None:
                            org_per_cell = _summarize_organelles_per_parent(organelle_mask, cell_mask, channel_arrays, parent_name='cell', spacing=spacing)
                            org_per_cell.columns = [f'organelle_summary_{col}' if col != 'label' else col for col in org_per_cell.columns]
                            _merge_and_save_to_database(org_per_cell, pd.DataFrame(), 'cell_organelle_summary', source_folder, file_name, settings['experiment'], settings['timelapse'], stamp=units_stamp)
                if "nucleus" in settings['summarize_organelles_by']:
                    if settings['nucleus_mask_dim'] is not None:
                        org_per_nucleus = _summarize_organelles_per_parent(organelle_mask, nucleus_mask, channel_arrays, parent_name='nucleus', spacing=spacing)
                        org_per_nucleus.columns = [f'organelle_summary_{col}' if col != 'label' else col for col in org_per_nucleus.columns]
                        _merge_and_save_to_database(org_per_nucleus, pd.DataFrame(), 'nucleus_organelle_summary', source_folder, file_name, settings['experiment'], settings['timelapse'], stamp=units_stamp)
                if "pathogen" in settings['summarize_organelles_by']:
                    if settings['pathogen_mask_dim'] is not None:
                        org_per_pathogen = _summarize_organelles_per_parent(organelle_mask, pathogen_mask, channel_arrays, parent_name='pathogen', spacing=spacing)
                        org_per_pathogen.columns = [f'organelle_summary_{col}' if col != 'label' else col for col in org_per_pathogen.columns]
                        _merge_and_save_to_database(org_per_pathogen, pd.DataFrame(), 'pathogen_organelle_summary', source_folder, file_name, settings['experiment'], settings['timelapse'], stamp=units_stamp)
                if "cytoplasm" in settings['summarize_organelles_by']:
                    if settings['cytoplasm_mask_dim'] is not None:
                        org_per_cytoplasm = _summarize_organelles_per_parent(organelle_mask, cytoplasm_mask, channel_arrays, parent_name='cytoplasm', spacing=spacing)
                        org_per_cytoplasm.columns = [f'organelle_summary_{col}' if col != 'label' else col for col in org_per_cytoplasm.columns]
                        _merge_and_save_to_database(org_per_cytoplasm, pd.DataFrame(), 'cytoplasm_organelle_summary', source_folder, file_name, settings['experiment'], settings['timelapse'], stamp=units_stamp)

        if volumetric and (settings['save_png'] or settings['save_arrays'] or settings['plot']):
            # Refused, not approximated. Every step of the crop path is
            # irreducibly 2-D: _crop_center and _find_bounding_box take (row,
            # col), cv2.imwrite writes an image, and the PNGs are the training
            # input for spacr.deep_spacr, whose models take H x W x 3. Cropping
            # a projection instead would silently substitute a different
            # measurement of the object -- one where anything sitting above or
            # below another object is merged into it -- under the same file
            # names, and nothing downstream could tell it had happened.
            print(f"3-D field {file_name}: measurements written, but no PNG "
                  f"crops or region arrays. Cropping is 2-D; to get crops from "
                  f"a z-stack, project it first "
                  f"(z_segmentation_mode='project').")
            raise_if_strict(
                f"save_png/save_arrays/plot requested for the 3-D field "
                f"{file_name}, but spaCR crops 2-D fields only. Measurements "
                f"were written; no crops were.", settings=settings)
        elif settings['save_png'] or settings['save_arrays'] or settings['plot']:
            # A bare string crop_mode used to be assigned to a local named
            # `crop_mode` and then thrown away: the very next line re-tested
            # settings['crop_mode'] for list-ness, which a string never is,
            # so the entire crop block was skipped. crop_mode='cell' wrote
            # the measurements and NOT ONE PNG, without an error.
            crop_ls = settings['crop_mode']
            if isinstance(crop_ls, str):
                crop_ls = [crop_ls]
            crop_ls = list(crop_ls)

            size_ls = settings['png_size']
            if not size_ls:
                raise ValueError(
                    "Setting: png_size is empty; give it [width, height], or "
                    "a [width, height] pair per crop_mode entry.")
            # `isinstance(size_ls[0], int)` missed a float or a numpy int, and
            # then `width, height = size_ls[crop_idx]` tried to unpack a
            # scalar. Ask what it IS -- a pair, or a list of pairs.
            if not isinstance(size_ls[0], (list, tuple)):
                size_ls = [size_ls]

            # All three per-mode settings now broadcast the same way, so
            # png_size no longer prints a mismatch warning and then raises
            # IndexError on the very next line.
            size_ls = _per_crop_mode(size_ls, len(crop_ls), 'png_size')
            dialate_pngs = _per_crop_mode(
                settings['dialate_pngs'], len(crop_ls), 'dialate_pngs')
            dialate_png_ratios = _per_crop_mode(
                settings['dialate_png_ratios'], len(crop_ls),
                'dialate_png_ratios')

            for crop_idx, crop_mode in enumerate(crop_ls):
                # An unrecognised crop mode used to print and fall
                # through, so crop_mask/dialate_png kept the PREVIOUS
                # mode's values: crop_mode=['cell','banana'] cropped the
                # cell mask a second time under the name 'banana'. Skip
                # it instead, and name the modes that exist.
                if crop_mode not in CROP_MODES:
                    print(f"Setting: crop_mode entry {crop_mode!r} is not "
                          f"one of {', '.join(CROP_MODES)}; skipping it. "
                          f"No {crop_mode}_png crops were written.")
                    continue

                width, height = size_ls[crop_idx]

                if crop_mode == 'cell':
                    crop_mask = cell_mask.copy()
                    dialate_png = dialate_pngs[crop_idx]
                    dialate_png_ratio = dialate_png_ratios[crop_idx]

                elif crop_mode == 'nucleus':
                    crop_mask = nucleus_mask.copy()
                    dialate_png = dialate_pngs[crop_idx]
                    dialate_png_ratio = dialate_png_ratios[crop_idx]
                elif crop_mode == 'pathogen':
                    crop_mask = pathogen_mask.copy()
                    dialate_png = dialate_pngs[crop_idx]
                    dialate_png_ratio = dialate_png_ratios[crop_idx]
                elif crop_mode == 'organelle':
                    crop_mask = organelle_mask.copy()
                    dialate_png = dialate_pngs[crop_idx]
                    dialate_png_ratio = dialate_png_ratios[crop_idx]
                else:  # cytoplasm -- dilation is forced off, see below.
                    crop_mask = cytoplasm_mask.copy()
                    # Dilating a cytoplasm ring grows it into the nucleus
                    # it is defined as excluding, so the crop would no
                    # longer be cytoplasm. Not a user choice.
                    dialate_png = False
                    # Assigned even though dilation is off, so the name never
                    # carries a previous crop mode's ratio into this one.
                    dialate_png_ratio = dialate_png_ratios[crop_idx]

                objects_in_image = np.unique(crop_mask)
                objects_in_image = objects_in_image[objects_in_image != 0]
                img_paths = []
                
                for _id in objects_in_image:
                    
                    region = (crop_mask == _id)

                    # Use the boolean mask to filter the cell_mask and then find unique IDs
                    region_cell_ids = np.atleast_1d(np.unique(cell_mask[region]))
                    region_nucleus_ids = np.atleast_1d(np.unique(nucleus_mask[region]))
                    region_pathogen_ids = np.atleast_1d(np.unique(pathogen_mask[region]))

                    if settings['use_bounding_box']:
                        region = _find_bounding_box(crop_mask, _id, buffer=10)

                    img_name, fldr, table_name = _generate_names(file_name=file_name, cell_id = region_cell_ids, cell_nucleus_ids=region_nucleus_ids, cell_pathogen_ids=region_pathogen_ids, source_folder=source_folder, crop_mode=crop_mode, timelapse=settings['timelapse'])

                    if dialate_png:
                        # count_nonzero, not np.sum: when use_bounding_box is
                        # on, _find_bounding_box fills the box with the LABEL
                        # VALUE rather than with True, so np.sum gave
                        # pixels * label and object 100 dilated sqrt(100)=10x
                        # more than object 1 -- the crop depended on an
                        # arbitrary label id.
                        region_area = np.count_nonzero(region)
                        # The diameter of an object from its size is the
                        # ndim-th root of that size, not always the square
                        # root: a voxel count is a volume, and sqrt of a
                        # volume is not a length. Unreachable for a 3-D
                        # field today (the whole crop block is refused
                        # above), but wrong is wrong.
                        if region.ndim == 3:
                            approximate_diameter = np.cbrt(region_area)
                        else:
                            approximate_diameter = np.sqrt(region_area)
                        dialate_png_px = int(approximate_diameter * dialate_png_ratio)
                        # scipy reads iterations=0 as "repeat until nothing
                        # changes", NOT as "do nothing", so a radius that
                        # rounded down to 0 -- every object under 25 px at the
                        # default ratio 0.2 -- grew to fill the entire field.
                        # The crop then became an unmasked window centred on
                        # the middle of the field instead of on the object.
                        if dialate_png_px > 0:
                            # scipy requires the structuring element to have
                            # the same rank as the input; a fixed (2, 2)
                            # raises "structure and input must have same
                            # dimensionality" on a volume.
                            struct = generate_binary_structure(region.ndim, region.ndim)
                            region = binary_dilation(region, structure=struct, iterations=dialate_png_px)

                    if settings['save_png']:
                        fldr_type = f"{crop_mode}_png/"
                        png_folder = os.path.join(fldr,fldr_type)
                        img_path = os.path.join(png_folder, img_name)
                        img_paths.append(img_path)

                        # Assembled in FILE order -- red plane first -- from
                        # the declared mapping, so what the setting says is
                        # what the PNG's slots hold. `png_dims` still works
                        # and is translated to the mapping it always meant
                        # (entry 0 blue, 1 green, 2 red).
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

                        # `build_png_channels` returns 1 plane (greyscale) or 3
                        # (r, g, b) and never 2, so the pad-a-dummy-plane
                        # branch that used to live here is gone: a two-entry
                        # mapping already carries its empty plane, in the slot
                        # the user left blank rather than always the last one.
                        grid = save_and_add_image_to_grid(
                            png_channels, img_path, grid, settings['plot'])

                        if len(img_paths) == len(objects_in_image):
                            filepaths_to_database(img_paths, settings, source_folder, crop_mode)

                    if settings['save_arrays']:
                        row_idx, col_idx = np.where(region)
                        region_array = data[row_idx.min():row_idx.max()+1, col_idx.min():col_idx.max()+1, :]
                        array_folder = f"{fldr}/region_array/"            
                        os.makedirs(array_folder, exist_ok=True)
                        np.save(os.path.join(array_folder, img_name), region_array)

                        grid = save_and_add_image_to_grid(png_channels, img_path, grid, settings['plot'])

                        img_paths.append(img_path)
                        if len(img_paths) == len(objects_in_image):
                            filepaths_to_database(img_paths, settings, source_folder, crop_mode)

        cells = np.unique(cell_mask)
    except Exception as e:
        print('main',e)
        # `cells = 0` (a plain int) is the cross-process failure sentinel:
        # the success path always assigns np.unique(...), an ndarray, so the
        # parent's job_callback can tell the two apart and file this field on
        # the run ledger. Without that the pool callback saw a normal result
        # and the run reported as complete.
        cells = 0
        traceback.print_exc()
        # Also lands in ~/.spacr/logs/spacr.log with the file id, so the
        # failure survives a scrolled-away terminal.
        RunLedger('_measure_crop_core').record_failure(file, stage='measure', exc=e)

    end = time.time()
    duration = end-start
    time_ls.append(duration)
    average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
    if settings['plot']:
        fig = img_list_to_grid(grid)
        figs[f'{file_name}__pngs'] = fig
    return index, average_time, cells, figs

#@log_function_call
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
        - ``dry_run`` — validate the settings and stop; nothing is read,
          written or imported.

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
    # dry_run comes FIRST, before the local imports below: .io and .timelapse
    # are heavy, and _save_settings_to_db writes to measurements.db as a side
    # effect further down. A validate-only run must not reach either.
    if settings.get('dry_run', False):
        from .validate import run_preflight
        return run_preflight(settings, 'measure')

    from .io import _save_settings_to_db
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
    
    settings['src'] = normalize_src_path(settings['src'])
    
    if isinstance(settings['src'], str):
        settings['src'] = [settings['src']]

    if isinstance(settings['src'], list):
        source_folders = settings['src']
        
        # One run for the whole invocation: one id on every log line and
        # every artifact it registers, one seed reaching numpy / random /
        # torch, and one on_error policy honoured at the per-field
        # boundary inside the pool loop. See spacr.runctx.
        with run_context('measure', settings) as run:
            for source_folder in source_folders:
                cancellation_checkpoint()
                print(f'Processing folder: {source_folder}')
            
                source_folder = format_path_for_system(source_folder)
                settings['src'] = source_folder

                settings = get_measure_crop_settings(settings)
                settings = measure_test_mode(settings)

                src_fldr = settings['src']
            
                if not os.path.basename(src_fldr).endswith('merged'):
                    print(f"WARNING: Source folder, settings: src: {src_fldr} should end with '/merged'")
                    src_fldr = os.path.join(src_fldr, 'merged')
                    settings['src'] = src_fldr
                    print(f"Changed source folder to: {src_fldr}")

                # Illumination / flat-field correction, if the settings ask
                # for it. Here, and not earlier: it estimates from the merged
                # fields this loop is about to measure, so it needs `src`
                # after the /merged normalisation above, and it is per source
                # folder because illumination differs between acquisition
                # sessions. It installs a preprocessing hook (and the env
                # vars that carry it into every spawned worker), so it has to
                # run before the pool below is built rather than beside it.
                #
                # Off unless `illumination_correction` is True, in which case
                # this call is the whole feature: without it the setting is a
                # switch that does nothing and every intensity feature keeps
                # its position-dependent bias. See spacr.illumination.
                from .illumination import prepare_illumination_correction
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
                    'cell_mask_dim', 'nucleus_mask_dim', 'pathogen_mask_dim',
                    'organelle_mask_dim', 'cell_min_size', 'nucleus_min_size',
                    'pathogen_min_size', 'organelle_min_size',
                    'cytoplasm_min_size',
                ]
            
                # Category B, every one of these: the settings are wrong, so no
                # field can be measured. Each historically printed a WARNING and
                # returned None, which the caller cannot distinguish from a
                # completed run that wrote no rows. SPACR_STRICT_ERRORS turns
                # them into a ConfigurationError; the default stays as-is.
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

                if not all(isinstance(settings[key], int) or settings[key] is None for key in int_setting_keys):
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
            
                # MUST come before _save_settings_to_db: that writes the settings
                # table with if_exists='replace', destroying the record of the run
                # being resumed — which is what the settings comparison reads.
                resume_plan = plan_measure_resume(settings)

                _save_settings_to_db(settings)

                files = [f for f in os.listdir(settings['src']) if f.endswith('.npy')]
                if resume_plan is not None:
                    files = resume_plan.filter_files(files)
                n_jobs = settings['n_jobs']
                print(f'using {n_jobs} cpu cores')
                print_progress(files_processed=0, files_to_process=len(files), n_jobs=n_jobs, time_ls=[], operation_type='Measure and Crop')

                # One ledger per source folder. Both failure routes are covered:
                # a worker that returned the cells==0 sentinel (it caught its own
                # exception), and a worker that died outright — the latter used to
                # be completely invisible, because apply_async stores the exception
                # on an AsyncResult nobody ever read.
                ledger = RunLedger('measure_crop')
                # This folder's ledger joins the run: the ledger's run_id, every
                # log line below and every artifact this run registers all carry
                # one id, so the log of the run that produced a measurements.db
                # can be pulled back with spacr.runctx.read_run_log().
                run.adopt(ledger)
                policy = run.policy.bind(ledger=ledger, record=False)
                index_to_file = dict(enumerate(files))
                reported_files = set()

                def job_callback(result):
                    """Record one completed field and save its optional figures."""
                    completed_jobs.add(result[0])
                    item = index_to_file.get(result[0], result[0])
                    reported_files.add(item)
                    # cells is np.unique(cell_mask) on success and the int 0 when
                    # _measure_crop_core swallowed an exception for this field.
                    if isinstance(result[2], int) and result[2] == 0:
                        ledger.record_failure(
                            item, stage='measure',
                            exc='field failed inside _measure_crop_core '
                                '(worker traceback in ~/.spacr/logs/spacr.log)')
                    else:
                        ledger.record_success(item, stage='measure')
                    process_meassure_crop_results([result], settings)
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
                    """
                    def _on_error(exc):
                        reported_files.add(job_file)
                        ledger.record_failure(job_file, stage='measure_worker', exc=exc)
                    return _on_error

                # One explicit context for both the Manager and the Pool. Mixing
                # them -- a fork Manager serving a spawn Pool, say -- is how the
                # shared time_ls proxy ends up unreachable from a worker.
                ctx = _pool_context()
                start_method = ctx.get_start_method()
                # A spawn/forkserver worker is a fresh interpreter with empty hook
                # registries, so a hook registered in *this* process would apply to
                # nothing at all and the run would look completely normal. Say so
                # before the pool starts rather than let it be invisible.
                warn_if_hooks_will_not_reach_workers(start_method)
                pool_jobs = resolve_pool_size(n_jobs, len(files),
                                              start_method=start_method)

                # _start_manager, not ctx.Manager(), because the bare call fails as
                # an EOFError from deep inside multiprocessing with no message at
                # all. See ManagerStartError.
                # try/finally, because on_error='stop' aborts here and an
                # aborted run's evidence is exactly what a reader needs:
                # without this the fields nobody heard from go uncounted
                # and measurements.db is never stamped, so a half-written
                # database reads as one nobody ever measured into.
                try:
                    with _start_manager(ctx) as manager:
                        time_ls = manager.list()
                        completed_jobs = set()  # Set to keep track of completed jobs

                        with ctx.Pool(pool_jobs) as pool:
                            # Bound outstanding work to one pool-width batch. Stop is
                            # checked only after all fields in that batch have
                            # completed their writes.
                            for offset in range(0, len(files), pool_jobs):
                                cancellation_checkpoint()
                                pending = []
                                for index in range(
                                        offset, min(offset + pool_jobs, len(files))):
                                    file = files[index]
                                    result = pool.apply_async(
                                        _measure_crop_core,
                                        args=(index, time_ls, file, settings),
                                    )
                                    pending.append((file, index, result))
                                for file, index, async_result in pending:
                                    # on_error, at the per-field boundary. The
                                    # ledger entry is written either way by
                                    # job_callback / make_error_callback, which is
                                    # why the policy is bound with record=False;
                                    # what on_error decides is whether the run
                                    # survives the field. retry re-submits the
                                    # field rather than re-reading the
                                    # AsyncResult, which can only be got once.
                                    for attempt in policy.attempts_for(
                                            file, stage='measure'):
                                        with attempt:
                                            try:
                                                if attempt.number == 1:
                                                    job_callback(async_result.get())
                                                else:
                                                    job_callback(pool.apply_async(
                                                        _measure_crop_core,
                                                        args=(index, time_ls, file,
                                                              settings)).get())
                                            except PipelineCancelled:
                                                raise
                                            except Exception as exc:
                                                # Only on the last attempt: the
                                                # ledger counts fields, not tries,
                                                # so a field that failed twice and
                                                # then worked is one success.
                                                if attempt.last:
                                                    make_error_callback(file)(exc)
                                                raise
                                cancellation_checkpoint()

                            pool.close()
                            pool.join()
                finally:
                    # Fields the pool never reported on at all (killed worker,
                    # pool terminated before the task ran, or on_error='stop'
                    # ending the run at the first bad field). Counting them
                    # keeps n_attempted equal to the number of fields on disk.
                    for job_file in files:
                        if job_file not in reported_files:
                            ledger.record_failure(job_file, stage='measure',
                                                  exc='field produced no result')

                    # Stamp measurements.db with the verdict, then print it
                    # last. This is the bit that turns "we printed a warning"
                    # into "the artifact knows it is suspect":
                    # spacr.errors.read_run_status() on this db tells a
                    # downstream reader how many fields are missing. In the
                    # finally, because an aborted run is exactly the one whose
                    # half-written database must not read as untouched.
                    db_path = os.path.join(os.path.dirname(settings['src']),
                                           'measurements', 'measurements.db')
                    ledger.finalize(
                        artifact=db_path if os.path.isfile(db_path) else None)

                if settings['timelapse']:
                    if settings['timelapse_objects'] == 'nucleus':
                        folder_path = settings['src']
                        mask_channels = [settings['nucleus_mask_dim'], settings['pathogen_mask_dim'], settings['cell_mask_dim']]
                        object_types = ['nucleus', 'pathogen', 'cell']
                        _timelapse_masks_to_gif(folder_path, mask_channels, object_types)

                if ledger.is_complete:
                    print("Successfully completed run")

            # Record what this run produced, stamped with the run id every
            # log line above carries, so an artifact and its log can be
            # joined: spacr.runctx.read_run_log(artifact.run_id). The
            # canonicalized settings, not the ones handed in, so the hash
            # recorded against each artifact covers the values actually used.
            run.register_outputs(settings=settings, roots=source_folders)

def process_meassure_crop_results(partial_results, settings):
    """
    Save and display the figures carried by each partial result.

    Args:
        partial_results (list): List of partial results; ``None`` entries are
            skipped. Each figure is written under
            ``<src>/../results/`` and then shown and closed.
        settings (dict): Settings dictionary; ``src`` gives the output root.
    """
    for result in partial_results:
        if result is None:
            continue
        index, avg_time, cells, figs = result
        if figs is not None:
            for key, fig in figs.items():
                part_1, part_2 = key.split('__')
                save_dir = os.path.join(os.path.dirname(settings['src']), 'results', f"{part_1}")
                os.makedirs(save_dir, exist_ok=True)
                fig_path = os.path.join(save_dir, f"{part_2}.pdf")
                # Imported here, not at module scope: `spacr.plot` pulls in
                # torch, cv2, seaborn, statsmodels and pingouin, and this
                # module is on the cold measure-worker spawn path. See
                # tests/test_measure_spawn.py.
                from .plot import save_figure
                fig_path = save_figure(fig, fig_path)
                plt.figure(fig.number)
                plt.show()
                plt.close(fig)
            result = (index, None, None, None)
            
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

    ledger = RunLedger('generate_cellpose_train_set')
    for folder in folders:
        mask_folder = os.path.join(folder, 'masks')
        experiment_id = os.path.basename(folder)
        for filename in os.listdir(mask_folder):  # List the contents of the directory
            path = os.path.join(mask_folder, filename)
            img_path = os.path.join(folder, filename)
            newname = experiment_id + '_' + filename
            new_mask = os.path.join(dst, 'masks', newname)
            new_img = os.path.join(dst, 'imgs', newname)

            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                # cv2 signals failure by returning None rather than raising,
                # so this needs recording explicitly.
                ledger.record_failure(path, stage='read_mask',
                                      exc='cv2.imread returned None')
                print(f"Error reading {path}, skipping.")
                continue

            nr_of_objects = len(np.unique(mask)) - 1  # Assuming 0 is background
            if nr_of_objects >= min_objects:  # Use >= to include min_objects
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
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    # Read the table into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM object_counts", conn)
    # Group by 'count_type' and calculate the sum of 'object_count' and the average 'object_count' per 'file_name'
    grouped_df = df.groupby('count_type').agg(
        total_object_count=('object_count', 'sum'),
        avg_object_count_per_file_name=('object_count', 'mean')
    ).reset_index()
    # Close the database connection
    conn.close()
    return grouped_df




# ---------------------------------------------------------------------------
# Object crops: the working dtype, and the one place it is left behind
# ---------------------------------------------------------------------------
#
# A merged array is 16-bit (``uint16``, or ``int32`` once a cellpose label
# plane has been concatenated onto it). The crop path keeps that dtype from
# the ``.npy`` all the way to the writer, exactly as ``_measure_crop_core``
# does: nothing in the middle of the pipeline is allowed to change it.
#
# 8-bit is genuinely required at exactly two places -- a PNG assembled by PIL
# (:func:`_save_object_crop`) and an RGB image handed to a GUI
# (``crop_objects_from_array(to_rgb=True)``). Both go through
# :func:`_crop_to_uint8`, which *rescales*. They used to go through
# ``np.clip(crop, 0, 255).astype(np.uint8)``, which does not: on a raw 16-bit
# crop every pixel above 255 -- i.e. every pixel of the object -- came out at
# exactly 255. Unnormalised 16-bit data shown as 8-bit has to look DARK; a
# clip is what turned it white, and those white crops were written to disk and
# trained on.


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

    # -- resolve paths --------------------------------------------------------
    root = os.path.abspath(src)
    if os.path.basename(root.rstrip(os.sep)) == 'merged':
        root = os.path.dirname(root.rstrip(os.sep))
    if db_path is None:
        db_path = os.path.join(root, 'measurements', 'measurements.db')
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"measurements database not found: {db_path}")

    if mask_dims is None:
        mask_dims = {'cell': 4, 'nucleus': 5, 'pathogen': 6, 'organelle': 7}
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
        # Mark the folder BEFORE the first PNG lands, for the same reason
        # `save_and_add_image_to_grid` does (crops.stamp_crop_folder): an
        # unmarked folder means LEGACY to every reader, and these crops are
        # not legacy.
        #
        # `_save_object_crop` writes through PIL, which is already RGB -- so
        # unlike the cv2 writer there is nothing to reverse here, and the
        # bytes on disk were correct all along. What was missing was the
        # marker saying so. Without it `crops.read_crop_png` resolved the
        # folder to format 1 and reversed a correct file on load, so the
        # annotator, the crop grid and the training loaders all showed
        # channel 0 as blue and channel 2 as red while an external viewer
        # showed them the right way round. Measured on a crop written with
        # channel means (60000, 1200, 12000): PIL read (234, 4, 46) off the
        # file, `read_crop_png` returned (46, 4, 234).
        stamp_crop_folder(output_dir)

    # -- build the WHERE clause ----------------------------------------------
    clauses, params = [], []
    if min_area is not None:
        clauses.append(f"{object_type}_area > ?"); params.append(float(min_area))
    if max_area is not None:
        clauses.append(f"{object_type}_area < ?"); params.append(float(max_area))

    def _in(colname, values, prefix):
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

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        selected = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    if verbose:
        print(f"generate_object_dataset({object_type}): {len(selected)} "
              f"objects match{where_sql or ' (no filter)'}")

    # -- crop each object -----------------------------------------------------
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
            # A (Z, Y, X, C) volume. Everything below is 2-D indexing, and
            # `data[:, :, mask_dim]` on a 4-D array returns a slab of X, not a
            # mask, without raising.
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
                            to_rgb=True, limit=None):
    """Crop every object out of an in-memory merged image+mask array.

    This is the no-database counterpart of :func:`generate_object_dataset`,
    used by the Measure live preview to show what the crops will look like
    before a run: it reads the object labels straight from a mask slice of a
    single merged ``.npy`` and returns the cropped, normalised images.

    :param data: merged array ``(H, W, C)`` — image channels then mask slices.
    :param mask_dim: slice index of the object-class mask to crop by.
    :param channels: image channel indices to assemble (order = RGB order).
    :param min_area/max_area: keep objects within this pixel-area range
        (``0`` = no bound).
    :param mask_background: zero pixels outside the object.
    :param normalize: per-channel percentile-normalise each crop.
    :param percentiles: ``(low, high)`` for normalisation.
    :param buffer: padding (px) around each object's bounding box.
    :param to_rgb: assemble the chosen channels into an HxWx3 uint8 image
        (1→grey→RGB, 2→padded, 3→RGB, >3→first three); else keep N channels
        **in the merged array's own dtype**.
    :param limit: cap the number of objects returned.
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

    # Order by area (largest first) so the preview leads with the clearest
    # objects; apply the area filter here too.
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
        # No "label vanished" guard here, unlike generate_object_dataset:
        # `scored` was built from np.unique of THIS plane a few lines up, so
        # every label in it is in it. (In generate_object_dataset the label
        # comes from the database and the plane from disk, which is a real
        # chance to disagree, and that guard is exercised.)
        ys, xs = np.where(mask == lbl)
        y0 = max(0, ys.min() - buffer); y1 = min(mask.shape[0], ys.max() + 1 + buffer)
        x0 = max(0, xs.min() - buffer); x1 = min(mask.shape[1], xs.max() + 1 + buffer)

        region = (mask[y0:y1, x0:x1] == lbl) if mask_background else None
        crop = _crop_channels(data, y0, y1, x0, x1, channels, region)
        if normalize:
            crop = _normalize_crop(crop, percentiles, mask_background)

        if to_rgb:
            # Declared 8-bit boundary: a QImage/RGB888 wants uint8. Narrow by
            # rescaling (_crop_to_uint8), then assemble -- so a raw 16-bit
            # field previews dark rather than solid white.
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

        out.append({"label": lbl, "area": area,
                    "bbox": (int(y0), int(y1), int(x0), int(x1)), "crop": crop})
    return out
