"""Opt-in extension points for the per-object measurement pipeline.

``spacr.measure`` is one 3 000-line module with a single hot path
(:func:`spacr.measure._measure_crop_core`). Two features that are being built
alongside it need to change *what* that path measures without editing it:

* **illumination / flat-field correction** — a per-channel multiplicative gain
  estimated across a plate, applied to the intensity channels before any
  feature is computed;
* **a user-drawn ROI / shapes layer** — "only measure the objects inside this
  polygon".

Both are expressed here as registries of plain callables. Nothing in this
module runs unless something registers a hook, and with an empty registry
every entry point returns its input object unchanged, so a default run
measures byte-for-byte what it measured before this module existed.

This is a separate module from ``spacr.measure`` on purpose. ``spacr.measure``
imports matplotlib, skimage, cv2 and scipy at module scope — seconds of import
time and hundreds of MB of RSS — and an extension that only wants to *register*
a callable should not pay for that, nor should it risk the import cycle that
``spacr.measure`` importing it back would create. This module imports numpy and
the stdlib, and nothing else from spaCR except :mod:`spacr.errors`.

The two hook kinds
------------------

**Preprocessing** — ``hook(channel_arrays, context) -> np.ndarray``

    ``channel_arrays`` is exactly the array the intensity measurements see:
    the intensity channels named by ``settings['channels']``, already selected
    out of the merged stack, shaped ``(Y, X, C)`` in 2-D or ``(Z, Y, X, C)`` in
    3-D. ``context`` is a :class:`PreprocessingContext`. The hook returns a
    replacement of **the same shape and the same dtype** — see
    :func:`apply_preprocessing_hooks` for why that is enforced rather than
    coerced.

**Region filter** — ``hook(context) -> np.ndarray[bool]``

    ``context`` is a :class:`RegionContext` carrying the object type, the
    label mask, the ascending array of its non-zero label ids and (computed
    only if the hook asks for them) their centroids. The hook returns a
    boolean array of ``len(context.labels)``: ``True`` keeps the object,
    ``False`` drops it. Dropped labels are zeroed out of the mask *before*
    morphology or intensity is computed, so excluding 495 of 500 objects costs
    5 objects' worth of work, not 500.

Ordering
--------

Every hook is registered with an integer ``priority`` (default ``0``) and runs
in ascending ``(priority, registration order)``. Ties keep registration order,
so the rule is fully deterministic. The two kinds then *combine* differently:

* Preprocessing hooks form a **pipeline**: each one receives the previous
  one's output. Order therefore matters, and ``priority`` is how two
  independent extensions agree on it without knowing about each other.
* Region filters **intersect**: every filter is handed the same, original set
  of labels, and an object is measured only if *every* filter kept it. Order
  does not change the outcome — deliberately, so that two workstreams adding a
  filter each cannot produce a result that depends on import order.

Registering the same function object again, or registering explicitly under a
name that is already taken, **replaces** the existing entry rather than adding
a second one. Re-running a GUI action that installs a flat-field correction
therefore cannot apply that correction twice.

Errors
------

A hook that raises, returns ``None``, or returns the wrong shape/dtype gets its
exception re-raised as :class:`MeasurementHookError` (a
:class:`spacr.errors.ConfigurationError`) naming the hook and the field. It is
raised inside ``_measure_crop_core``'s ``try``, so it takes the ordinary
per-field failure route: the traceback is printed, the field is recorded on the
``RunLedger`` as a failure, and ``measurements.db`` is stamped incomplete. It
is never swallowed into a row of quietly-wrong numbers.

Reaching worker processes
-------------------------

:func:`spacr.measure.measure_crop` measures fields in a
:class:`multiprocessing.Pool`. Under ``fork`` (the Linux default) a worker
inherits this module's registries from the parent and hooks registered with
:func:`register_preprocessing_hook` simply work. Under ``spawn`` /
``forkserver`` (Windows, macOS, or ``SPACR_START_METHOD``) a worker is a fresh
interpreter that has never seen them, so an in-process registration would be a
**silent no-op in every worker**. For that case set
:data:`HOOKS_ENV_VAR`::

    SPACR_MEASURE_HOOKS="mypkg.illumination:install,mypkg.roi:install"

Each entry is ``module:attribute`` naming a zero-argument callable that does
its own ``register_*`` calls. The environment is inherited by every start
method, so each worker installs the same hooks for itself. ``measure_crop``
prints a warning if hooks were registered in-process and the pool will not
inherit them (:func:`warn_if_hooks_will_not_reach_workers`).
"""

from __future__ import annotations

import importlib
import itertools
import os
import threading
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .errors import ConfigurationError
from .object_roles import ORGANELLE_ROLES

__all__ = [
    'HOOKS_ENV_VAR',
    'OBJECT_TYPES',
    'MeasurementHookError',
    'RegisteredHook',
    'PreprocessingContext',
    'RegionContext',
    'register_preprocessing_hook',
    'register_region_filter_hook',
    'unregister_preprocessing_hook',
    'unregister_region_filter_hook',
    'preprocessing_hooks',
    'region_filter_hooks',
    'clear_measurement_hooks',
    'describe_hooks',
    'apply_preprocessing_hooks',
    'apply_region_filter_hooks',
    'warn_if_hooks_will_not_reach_workers',
]

#: Environment variable naming ``module:attribute`` installers, comma
#: separated. Each attribute is a zero-argument callable that registers hooks.
#: Read once per process, the first time either registry is consulted; this is
#: the only route that survives a ``spawn`` / ``forkserver`` worker pool.
HOOKS_ENV_VAR = 'SPACR_MEASURE_HOOKS'

#: The object types a region filter is consulted about, in the order
#: ``_measure_crop_core`` applies them.
#: Hook order. Membership comes from the registry; the ORDER is this
#: module's own and is kept here deliberately -- see spacr.object_roles.
OBJECT_TYPES = (
    'cell', 'nucleus', 'pathogen', *ORGANELLE_ROLES, 'cytoplasm')


class MeasurementHookError(ConfigurationError):
    """A registered measurement hook raised, or returned something unusable.

    A :class:`spacr.errors.ConfigurationError` rather than a per-field data
    error, because a broken hook is broken for every field on the plate: the
    message names the hook and tells the caller how to unregister it.
    """


class RegisteredHook(NamedTuple):
    """One entry in a hook registry.

    :ivar name: unique key; pass it to the matching ``unregister_*``.
    :ivar func: the callable itself.
    :ivar priority: lower runs first; ties break on ``sequence``.
    :ivar sequence: monotonic registration counter, for stable ordering.
    :ivar source: ``'api'`` for :func:`register_preprocessing_hook` and
        friends, ``'env'`` for anything installed via :data:`HOOKS_ENV_VAR`.
        Only ``'api'`` hooks fail to reach a non-``fork`` worker pool.
    """

    name: str
    func: Callable[..., Any]
    priority: int
    sequence: int
    source: str


class PreprocessingContext:
    """Everything a preprocessing hook is told about the field it is given.

    :ivar file_name: the field's ``.npy`` stem, e.g. ``plate1_A01_F001``.
        A plate-wide illumination model keys its per-well gain off this.
    :ivar channels: the intensity channel indices selected out of the merged
        stack, in the order they appear along the last axis of the array the
        hook receives. ``channel_arrays[..., i]`` is source channel
        ``channels[i]``.
    :ivar settings: read-only view of the run settings dict.
    :ivar volumetric: True when the field is a ``(Z, Y, X, C)`` z-stack.
    :ivar spacing: voxel spacing tuple in the array's index order, or None in
        2-D (spaCR does not scale 2-D measurements; see
        :func:`spacr.measure.resolve_measurement_spacing`).
    """

    __slots__ = ('file_name', 'channels', 'settings', 'volumetric', 'spacing')

    def __init__(self, *, file_name: str, channels: Sequence[int],
                 settings: Mapping[str, Any], volumetric: bool = False,
                 spacing: Optional[Sequence[float]] = None) -> None:
        self.file_name = file_name
        self.channels = tuple(int(c) for c in channels)
        self.settings = _read_only(settings)
        self.volumetric = bool(volumetric)
        self.spacing = None if spacing is None else tuple(spacing)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f'PreprocessingContext(file_name={self.file_name!r}, '
                f'channels={self.channels!r}, volumetric={self.volumetric!r})')


class RegionContext:
    """Everything a region filter is told about one object type in one field.

    :ivar object_type: one of :data:`OBJECT_TYPES`. Each type is offered
        separately, so a filter can act on one and wave the rest through with
        ``np.ones(len(context.labels), bool)``. Note that culling only the
        cells cascades to their nuclei, pathogens and cytoplasm *when the run
        has all of ``cell_mask_dim``, ``nucleus_mask_dim`` and
        ``pathogen_mask_dim`` set* — that is the condition under which
        ``_measure_crop_core`` calls ``_exclude_objects``, which zeroes the
        child masks outside the surviving cells. Otherwise apply the same
        decision to each type, which is what a polygon test on the centroid
        does naturally.
    :ivar file_name: the field's ``.npy`` stem.
    :ivar mask: the label mask, ``(Y, X)`` or ``(Z, Y, X)``. Read-only.
    :ivar settings: read-only view of the run settings dict.
    :ivar spacing: voxel spacing in the mask's index order, or None in 2-D.

    :ivar labels: ascending array of the mask's non-zero label ids. The
        boolean array the hook returns is aligned with this, element for
        element.
    :ivar centroids: ``(len(labels), mask.ndim)`` float array of centroids in
        **array index order** — ``(row, col)`` in 2-D, ``(z, y, x)`` in 3-D —
        in pixels/voxels, unscaled by ``spacing``. Computed on first access
        and cached, so a filter that rasterises its polygon and reads ``mask``
        directly never pays for it.
    """

    __slots__ = ('object_type', 'file_name', 'mask', 'settings', 'spacing',
                 '_labels', '_centroids')

    def __init__(self, *, object_type: str, file_name: str, mask: np.ndarray,
                 settings: Mapping[str, Any],
                 spacing: Optional[Sequence[float]] = None) -> None:
        self.object_type = object_type
        self.file_name = file_name
        mask = np.asarray(mask)
        view = mask.view()
        view.flags.writeable = False
        self.mask = view
        self.settings = _read_only(settings)
        self.spacing = None if spacing is None else tuple(spacing)
        self._labels: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions of :attr:`mask` (2 or 3)."""
        return int(self.mask.ndim)

    @property
    def labels(self) -> np.ndarray:
        """Ascending array of the mask's non-zero label ids."""
        if self._labels is None:
            values = np.unique(self.mask)
            self._labels = values[values != 0]
        return self._labels

    @property
    def centroids(self) -> np.ndarray:
        """``(N, ndim)`` centroids per :attr:`labels`, in array index order."""
        if self._centroids is None:
            labels = self.labels
            if labels.size == 0:
                self._centroids = np.zeros((0, self.ndim), dtype=float)
            else:
                # scipy is imported here rather than at module scope: this
                # module is on the import path of anything that merely wants
                # to *register* a hook, and most filters never ask for a
                # centroid at all.
                from scipy.ndimage import center_of_mass
                weights = np.ones(self.mask.shape, dtype=np.uint8)
                found = center_of_mass(weights, labels=self.mask,
                                       index=labels)
                self._centroids = np.asarray(found, dtype=float).reshape(
                    len(labels), self.ndim)
        return self._centroids

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f'RegionContext(object_type={self.object_type!r}, '
                f'file_name={self.file_name!r}, shape={self.mask.shape!r})')


def _read_only(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only view of ``mapping`` when one is available.

    Hooks are third-party code and the run settings dict is live, shared, and
    read by every stage downstream; handing it over unwrapped invites a hook
    to flip ``save_png`` for the rest of the plate.
    """
    if isinstance(mapping, dict):
        return MappingProxyType(mapping)
    return mapping


# ---------------------------------------------------------------------------
# The registries
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()
_SEQUENCE = itertools.count()
_PREPROCESSING: "OrderedDict[str, RegisteredHook]" = OrderedDict()
_REGION_FILTERS: "OrderedDict[str, RegisteredHook]" = OrderedDict()
_ENV_LOADED = False

#: Mutable one-slot cell so that hooks registered from inside a
#: :data:`HOOKS_ENV_VAR` installer are tagged ``'env'`` without every
#: ``register_*`` call growing a parameter no ordinary caller should pass.
#: ``'env'`` hooks are the ones a spawned worker re-installs for itself, so
#: :func:`warn_if_hooks_will_not_reach_workers` leaves them alone.
_CURRENT_SOURCE = ['api']


def _default_name(registry, hook) -> str:
    """Derive a registry key for ``hook``, reusing it when it is the same object."""
    module = getattr(hook, '__module__', None) or 'unknown'
    qualname = (getattr(hook, '__qualname__', None)
                or getattr(hook, '__name__', None)
                or type(hook).__name__)
    base = f'{module}.{qualname}'
    existing = registry.get(base)
    if existing is None or existing.func is hook:
        # Free, or already this exact callable: re-registering is idempotent.
        return base
    candidate = 2
    while f'{base}#{candidate}' in registry:
        candidate += 1
    return f'{base}#{candidate}'


def _register(registry, kind: str, hook, name: Optional[str], priority: int,
              source: str) -> str:
    """Insert ``hook`` into ``registry`` and return the key it landed under."""
    if not callable(hook):
        raise MeasurementHookError(
            f'a {kind} hook must be callable, got '
            f'{type(hook).__name__}. Nothing was registered.')
    try:
        priority = int(priority)
    except (TypeError, ValueError) as exc:
        raise MeasurementHookError(
            f'priority for the {kind} hook must be an int, got '
            f'{priority!r}. Nothing was registered.') from exc
    with _LOCK:
        key = str(name) if name is not None else _default_name(registry, hook)
        # Replace rather than append: an extension that re-installs itself on
        # every GUI run must not end up applying its correction twice.
        registry.pop(key, None)
        registry[key] = RegisteredHook(name=key, func=hook, priority=priority,
                                       sequence=next(_SEQUENCE), source=source)
    return key


def register_preprocessing_hook(hook: Callable[..., Any], *,
                                name: Optional[str] = None,
                                priority: int = 0) -> str:
    """Register ``hook(channel_arrays, context) -> np.ndarray``.

    The hook is called once per field, with the intensity channels named by
    ``settings['channels']`` already selected and before a single feature is
    computed. It must return an array of the same shape and dtype. This is
    where a flat-field / illumination correction belongs.

    :param hook: the callable. ``context`` is a :class:`PreprocessingContext`.
    :param name: registry key. Defaults to ``module.qualname``; registering
        the same key (or the same function object) again replaces the entry
        instead of adding a second one.
    :param priority: lower runs first; ties keep registration order.
    :returns: the key the hook was registered under — pass it to
        :func:`unregister_preprocessing_hook`.
    :raises MeasurementHookError: if ``hook`` is not callable or ``priority``
        is not an int.
    """
    return _register(_PREPROCESSING, 'preprocessing', hook, name, priority,
                     _CURRENT_SOURCE[0])


def register_region_filter_hook(hook: Callable[..., Any], *,
                                name: Optional[str] = None,
                                priority: int = 0) -> str:
    """Register ``hook(context) -> np.ndarray[bool]``.

    The hook is called once per object type per field with a
    :class:`RegionContext`, and returns a boolean array aligned with
    ``context.labels``: ``True`` measures the object, ``False`` drops it
    before any feature is computed. This is where a user-drawn ROI belongs.

    :param hook: the callable.
    :param name: registry key; see :func:`register_preprocessing_hook`.
    :param priority: affects only the order filters are *reported* in — the
        results are intersected, so the outcome is order-independent.
    :returns: the key the hook was registered under.
    :raises MeasurementHookError: if ``hook`` is not callable or ``priority``
        is not an int.
    """
    return _register(_REGION_FILTERS, 'region filter', hook, name, priority,
                     _CURRENT_SOURCE[0])


def unregister_preprocessing_hook(name: str) -> bool:
    """Remove a preprocessing hook by name.

    :param name: the key returned by :func:`register_preprocessing_hook`.
    :returns: True if something was removed, False if that name was not
        registered.
    """
    with _LOCK:
        return _PREPROCESSING.pop(name, None) is not None


def unregister_region_filter_hook(name: str) -> bool:
    """Remove a region-filter hook by name.

    :param name: the key returned by :func:`register_region_filter_hook`.
    :returns: True if something was removed, False otherwise.
    """
    with _LOCK:
        return _REGION_FILTERS.pop(name, None) is not None


def clear_measurement_hooks() -> None:
    """Empty both registries and re-arm :data:`HOOKS_ENV_VAR` loading.

    Returns the module to its pristine, no-op state. Mainly for tests and for
    a GUI tearing down a session.
    """
    global _ENV_LOADED
    with _LOCK:
        _PREPROCESSING.clear()
        _REGION_FILTERS.clear()
        _ENV_LOADED = False


def _load_env_hooks() -> None:
    """Run the ``module:attribute`` installers named by :data:`HOOKS_ENV_VAR`.

    Once per process. Attempted exactly once even if it fails, so a typo in
    the variable does not re-raise on all 384 wells.
    """
    global _ENV_LOADED
    with _LOCK:
        if _ENV_LOADED:
            return
        _ENV_LOADED = True
        spec = os.environ.get(HOOKS_ENV_VAR, '').strip()
        if not spec:
            return
        for entry in spec.split(','):
            entry = entry.strip()
            if entry:
                _install_env_entry(entry)


def _install_env_entry(entry: str) -> None:
    """Import and call one ``module:attribute`` installer from the env var."""
    if ':' not in entry:
        raise MeasurementHookError(
            f'{HOOKS_ENV_VAR} entry {entry!r} is not of the form '
            f'"module:attribute" naming a zero-argument installer, e.g. '
            f'"mypkg.illumination:install".')
    module_name, _, attribute = entry.partition(':')
    try:
        module = importlib.import_module(module_name.strip())
        installer = getattr(module, attribute.strip())
    except Exception as exc:
        raise MeasurementHookError(
            f'{HOOKS_ENV_VAR} entry {entry!r} could not be resolved: '
            f'{type(exc).__name__}: {exc}') from exc
    if not callable(installer):
        raise MeasurementHookError(
            f'{HOOKS_ENV_VAR} entry {entry!r} resolved to '
            f'{type(installer).__name__}, which is not callable. It must be a '
            f'zero-argument function that calls register_preprocessing_hook / '
            f'register_region_filter_hook.')
    previous_source = _CURRENT_SOURCE[0]
    _CURRENT_SOURCE[0] = 'env'
    try:
        installer()
    except MeasurementHookError:
        raise
    except Exception as exc:
        raise MeasurementHookError(
            f'{HOOKS_ENV_VAR} installer {entry!r} raised '
            f'{type(exc).__name__}: {exc}') from exc
    finally:
        _CURRENT_SOURCE[0] = previous_source


def _ordered(registry) -> Tuple[RegisteredHook, ...]:
    """Snapshot ``registry`` sorted by ``(priority, registration order)``."""
    _load_env_hooks()
    with _LOCK:
        return tuple(sorted(registry.values(),
                            key=lambda entry: (entry.priority, entry.sequence)))


def preprocessing_hooks() -> Tuple[RegisteredHook, ...]:
    """Return the registered preprocessing hooks in the order they will run.

    Also the cheap "is anything registered at all?" test — an empty tuple is
    falsy, and :func:`spacr.measure._measure_crop_core` branches on it so a
    default run does not even build a context object.
    """
    return _ordered(_PREPROCESSING)


def region_filter_hooks() -> Tuple[RegisteredHook, ...]:
    """Return the registered region-filter hooks in reporting order.

    Their results are intersected, so this order does not affect which objects
    survive — only which hook is named first in a diagnostic.
    """
    return _ordered(_REGION_FILTERS)


def describe_hooks() -> str:
    """Return a one-line-per-hook summary, or a line saying there are none."""
    lines = []
    for kind, entries in (('preprocessing', preprocessing_hooks()),
                          ('region filter', region_filter_hooks())):
        for entry in entries:
            lines.append(f'{kind}: {entry.name} '
                         f'(priority={entry.priority}, source={entry.source})')
    if not lines:
        return 'no measurement hooks registered'
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _raise_hook_failure(kind: str, entry: RegisteredHook, detail: str,
                        file_name: str,
                        exc: Optional[BaseException] = None) -> None:
    """Raise :class:`MeasurementHookError`, chaining ``exc`` when there is one.

    The message always names the hook and the exact call that removes it: a
    broken extension is a configuration problem, and the person reading the
    traceback is not necessarily the person who wrote the hook.
    """
    unregister = ('unregister_preprocessing_hook'
                  if kind == 'preprocessing' else
                  'unregister_region_filter_hook')
    error = MeasurementHookError(
        f'{kind} hook {entry.name!r} {detail} while measuring '
        f'{file_name!r}. No measurements were written for this field. '
        f'Fix the hook, or remove it with '
        f'spacr.measure_hooks.{unregister}({entry.name!r}).')
    if exc is not None:
        raise error from exc
    raise error


def apply_preprocessing_hooks(channel_arrays: np.ndarray,
                              context: PreprocessingContext) -> np.ndarray:
    """Run every preprocessing hook in order and return the transformed array.

    With no hook registered this returns ``channel_arrays`` itself — the same
    object, not a copy — so the default measurement path is unchanged.

    Each hook must return an array of the same shape **and the same dtype**.
    The dtype is checked rather than coerced on purpose: a multiplicative
    correction naturally computes in float, and only its author knows whether
    going back to ``uint16`` should round, truncate, clip or rescale. Silently
    picking one here would change every intensity column in
    ``measurements.db`` by an amount nobody chose.

    :param channel_arrays: ``(Y, X, C)`` or ``(Z, Y, X, C)`` intensity array.
    :param context: the :class:`PreprocessingContext` handed to each hook.
    :returns: the array the measurement code should use.
    :raises MeasurementHookError: if a hook raises, returns None, or returns
        the wrong shape or dtype. The message names the hook.
    """
    hooks = preprocessing_hooks()
    if not hooks:
        return channel_arrays
    original = np.asarray(channel_arrays)
    expected_shape = original.shape
    expected_dtype = original.dtype
    result = channel_arrays
    for entry in hooks:
        try:
            produced = entry.func(result, context)
        except Exception as exc:
            _raise_hook_failure(
                'preprocessing', entry,
                f'raised {type(exc).__name__}: {exc}',
                context.file_name, exc)
        if produced is None:
            _raise_hook_failure(
                'preprocessing', entry,
                'returned None; it must return the transformed array',
                context.file_name)
        produced = np.asarray(produced)
        if produced.shape != expected_shape:
            _raise_hook_failure(
                'preprocessing', entry,
                f'returned shape {produced.shape} but the intensity channels '
                f'are {expected_shape}; a preprocessing hook may transform '
                f'values, not geometry',
                context.file_name)
        if produced.dtype != expected_dtype:
            _raise_hook_failure(
                'preprocessing', entry,
                f'returned dtype {produced.dtype} but the intensity channels '
                f'are {expected_dtype}; cast the result yourself so the '
                f'rounding and clipping are your choice, e.g. '
                f'np.clip(np.rint(x), 0, np.iinfo({expected_dtype}).max)'
                f'.astype({expected_dtype})',
                context.file_name)
        result = produced
    return result


def _coerce_keep_mask(kind: str, entry: RegisteredHook, decision: Any,
                      context: RegionContext) -> np.ndarray:
    """Validate one region filter's return value into a boolean keep-mask."""
    if decision is None:
        _raise_hook_failure(
            kind, entry,
            f'returned None for the {context.object_type} mask; it must '
            f'return a boolean array of len(context.labels)',
            context.file_name)
    decision = np.asarray(decision)
    expected = (len(context.labels),)
    if decision.shape != expected:
        _raise_hook_failure(
            kind, entry,
            f'returned shape {decision.shape} for the {context.object_type} '
            f'mask but there are {expected[0]} {context.object_type} '
            f'object(s); the result is aligned with context.labels element '
            f'for element',
            context.file_name)
    if decision.dtype != bool:
        unique = np.unique(decision)
        if unique.size and not np.all(np.isin(unique, (0, 1))):
            _raise_hook_failure(
                kind, entry,
                f'returned dtype {decision.dtype} with values {unique.tolist()} '
                f'for the {context.object_type} mask; a region filter returns '
                f'True/False per object, not a label list or a score',
                context.file_name)
        decision = decision.astype(bool)
    return decision


def apply_region_filter_hooks(mask: np.ndarray, *, object_type: str,
                              file_name: str, settings: Mapping[str, Any],
                              spacing: Optional[Sequence[float]] = None
                              ) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Zero out the objects every registered region filter agreed to drop.

    With no filter registered — or with nothing dropped — this returns
    ``mask`` itself, so the default path neither copies nor changes anything.

    Each filter sees the same original label set; the keep-masks are AND-ed,
    so an object survives only if every filter kept it and the outcome does
    not depend on registration order.

    :param mask: the label mask for ``object_type``.
    :param object_type: one of :data:`OBJECT_TYPES`.
    :param file_name: the field's ``.npy`` stem, for error messages.
    :param settings: the run settings dict.
    :param spacing: voxel spacing in the mask's index order, or None in 2-D.
    :returns: ``(filtered_mask, dropped_labels)``.
    :raises MeasurementHookError: if a filter raises or returns something that
        is not a boolean array of ``len(labels)``.
    """
    hooks = region_filter_hooks()
    if not hooks:
        return mask, ()
    context = RegionContext(object_type=object_type, file_name=file_name,
                            mask=mask, settings=settings, spacing=spacing)
    labels = context.labels
    if labels.size == 0:
        return mask, ()
    keep = np.ones(labels.shape, dtype=bool)
    for entry in hooks:
        try:
            decision = entry.func(context)
        except Exception as exc:
            _raise_hook_failure(
                'region filter', entry,
                f'raised {type(exc).__name__}: {exc}',
                file_name, exc)
        keep &= _coerce_keep_mask('region filter', entry, decision, context)
    if keep.all():
        return mask, ()
    dropped = labels[~keep]
    filtered = np.asarray(mask).copy()
    filtered[np.isin(mask, dropped)] = 0
    return filtered, tuple(int(value) for value in dropped)


def warn_if_hooks_will_not_reach_workers(start_method: str) -> bool:
    """Print a warning when in-process hooks cannot reach the worker pool.

    A ``spawn`` / ``forkserver`` worker is a fresh interpreter with empty
    registries, so hooks registered through the Python API would apply to
    nothing at all — the exact silent no-op this module exists to avoid. Hooks
    installed via :data:`HOOKS_ENV_VAR` are re-installed by each worker and are
    not warned about.

    :param start_method: the pool's multiprocessing start method.
    :returns: True if a warning was printed.
    """
    if start_method == 'fork':
        return False
    stranded = [entry.name
                for entry in preprocessing_hooks() + region_filter_hooks()
                if entry.source == 'api']
    if not stranded:
        return False
    print(f"WARNING: measurement hooks {', '.join(stranded)} were registered "
          f"in this process, but the measure pool starts workers with "
          f"'{start_method}', which does not inherit them -- they would apply "
          f"to nothing. Install them through {HOOKS_ENV_VAR}="
          f"'module:installer' instead, or set SPACR_START_METHOD=fork where "
          f"the platform supports it.")
    return True
