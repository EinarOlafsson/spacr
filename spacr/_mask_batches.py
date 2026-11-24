"""Batch plans for mask generation: which fields form each Cellpose batch, and the batches in memory.

Mask generation records its batches in ``masks/batch_plan.json`` instead of
writing ``.npz`` image stacks: the plan lists each batch's fields, channels
and normalisation, and each batch is built in memory from the source images
when it is segmented. The batch size is chosen to fit the free GPU memory and
RAM when ``batch_size`` is automatic. Internal to :mod:`spacr.io`,
:mod:`spacr.core` and :mod:`spacr.object`.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import random

import numpy as np

_PLAN_NAME = 'batch_plan.json'
_PLAN_VERSION = 1

_CACHE = {}

#: Batches the batch-first driver holds, by absolute path (see :func:`_batch_first`).
_RESIDENT = {}

#: Seconds to wait for free RAM before a batch is built anyway, and the poll step.
_RAM_WAIT_SECONDS = 900.0
_RAM_POLL_SECONDS = 1.0


def _plan_file(masks_dir):
    """Path of the batch plan inside a ``masks`` folder.

    :param masks_dir: the ``masks`` folder.
    :returns: ``<masks_dir>/batch_plan.json``.
    """
    return os.path.join(os.fspath(masks_dir), _PLAN_NAME)


def _virtual_batch_name(index):
    """Name a batch would have had as an archive.

    :param index: the batch number.
    :returns: ``stack_<index>_norm.npz``; no such file exists.
    """
    return f'stack_{int(index)}_norm.npz'


def _json_safe(settings):
    """The settings the normaliser can read back from a JSON plan.

    :param settings: the preprocessing settings.
    :returns: the entries whose values survive a JSON round trip.
    """
    kept = {}
    for key, value in dict(settings).items():
        if key == 'spacr_resolved_parameters':
            from .run_journal import _RESOLVED_PARAMETERS_LOCK, _validate_resolved_parameters

            with _RESOLVED_PARAMETERS_LOCK:
                value = _validate_resolved_parameters(value)
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        kept[str(key)] = value
    return kept


def _write_plan(masks_dir, stack_dir, batches, channels, settings, save_dtype, *,
                request=None, resolved=True, extra=None):
    """Write the batch plan next to where the archives used to go.

    :param masks_dir: the ``masks`` folder.
    :param stack_dir: the ``stack`` folder holding the field ``.npy`` files.
    :param batches: a list of lists of field file names, one list per batch,
        in batch order.
    :param channels: the channel indices every batch keeps.
    :param settings: the preprocessing settings the normaliser reads.
    :param save_dtype: the dtype the normalised batches are returned in.
    :param request: the ``batch_size`` asked for (None for automatic).
    :param resolved: False when the batches are provisional and the masks stage still
        has to fix the batch sizes against the free GPU and RAM (:func:`_resolve_plan`).
    :param extra: further record fields (sizes, borrowed batches, normalisation).
    :returns: the path of the plan file.
    """
    from .io import _replace_atomically
    from .run_journal import _resolved_parameters

    masks_dir = os.fspath(masks_dir)
    _resolved_parameters(settings)
    record = {
        'version': _PLAN_VERSION,
        'stack_dir': os.path.relpath(os.fspath(stack_dir), masks_dir),
        'channels': [int(channel) for channel in channels],
        'save_dtype': np.dtype(save_dtype).name,
        'batches': [{'index': i, 'fields': [str(name) for name in names]}
                    for i, names in enumerate(batches)],
        'settings': _json_safe(settings),
        'batch_size_request': request,
        'resolved': bool(resolved),
        **(extra or {}),
    }
    path = _plan_file(masks_dir)
    payload = json.dumps(record, indent=1).encode('utf-8')
    _replace_atomically(path, lambda handle: handle.write(payload), prefix='.spacr_plan_')
    record['_masks_dir'] = masks_dir
    record['_runtime_settings'] = settings
    _CACHE[path] = (os.stat(path).st_mtime_ns, record)
    return path


def _read_plan(masks_dir):
    """Read the batch plan of a ``masks`` folder.

    :param masks_dir: the ``masks`` folder.
    :returns: the plan as a dict, or ``None`` when there is none or it is
        unreadable.
    """
    path = _plan_file(masks_dir)
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        return None
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        with open(path, encoding='utf-8') as handle:
            plan = json.load(handle)
        if plan.get('version') != _PLAN_VERSION or not plan.get('batches'):
            return None
    except (OSError, ValueError):
        return None
    plan['_masks_dir'] = os.fspath(masks_dir)
    _CACHE[path] = (stamp, plan)
    return plan


def _batch_fields(plan, index):
    """The field file names of one batch.

    :param plan: from :func:`_read_plan`.
    :param index: the batch number.
    :returns: the names, in batch order.
    """
    return list(plan['batches'][int(index)]['fields'])


def _plan_field_stems(plan):
    """Every field the plan covers, without the ``.npy`` suffix.

    :param plan: from :func:`_read_plan`.
    :returns: a set of field stems.
    """
    return {os.path.splitext(os.path.basename(name))[0]
            for batch in plan['batches'] for name in batch['fields']}


def _batch_index_of(path):
    """The batch number a (virtual) archive name carries.

    :param path: a path ending in ``stack_<n>_norm.npz``.
    :returns: ``n``, or ``None`` for any other name.
    """
    name = os.path.basename(os.fspath(path))
    if name.startswith('stack_') and name.endswith('_norm.npz'):
        middle = name[len('stack_'):-len('_norm.npz')]
        if middle.isdigit():
            return int(middle)
    return None


def _load_batch(plan, index, settings=None):
    """Build one normalised batch from its fields, exactly as the archive writer did.

    The fields are loaded, padded to the largest field of the batch when
    their shapes differ, stacked, normalised with the run's own settings and
    cut to the kept channels. Fields of one shape and dtype are read one at
    a time straight into the stack, so the batch is held once, not as its
    fields and their stack.

    :param plan: from :func:`_read_plan`.
    :param index: the batch number.
    :param settings: active run settings, preserving replay state across workers.
    :returns: ``{'data', 'filenames', 'source_channels'}`` as the archive
        held them.
    """
    from .io import _normalize_img_batch, _normalizing_reporter
    from .run_journal import _resolved_parameters

    masks_dir = plan['_masks_dir']
    stack_dir = os.path.normpath(os.path.join(masks_dir, plan['stack_dir']))
    names = _batch_fields(plan, index)
    channels = [int(channel) for channel in plan['channels']]
    from .image_store import _read_stack, _stack_header
    paths = [os.path.join(stack_dir, name) for name in names]
    headers = {_stack_header(path) for path in paths}
    arrays = shapes = None
    if len(headers) == 1:
        shape, dtype = next(iter(headers))
        stack = np.empty((len(paths), *shape), dtype=dtype)
        for slot, path in enumerate(paths):
            stack[slot] = _read_stack(path)
        field_shapes = [tuple(shape[:-1])] * len(paths)
    else:
        arrays = [_read_stack(path) for path in paths]
        shapes = {array.shape[:-1] for array in arrays}
        field_shapes = [array.shape[:-1] for array in arrays]
    if arrays is None:
        pass
    elif len(shapes) > 1:
        largest = np.max(np.array(list(shapes)), axis=0)
        print(f'Warning: arrays with multiple shapes found in batch {index + 1}. '
              f'Padding arrays to max X,Y dimensions {largest}')
        padded = []
        for array in arrays:
            pad_width = [(0, top - size) for top, size in zip(largest, array.shape[:-1])]
            pad_width.append((0, 0))
            padded.append(np.pad(array, pad_width))
        stack = np.stack(padded)
    else:
        stack = np.stack(arrays)
    active_settings = (settings if settings is not None else
                       plan.get('_runtime_settings', plan['settings']))
    context = _resolved_parameters(active_settings)
    plan['_runtime_settings'] = active_settings
    settings = dict(active_settings)
    record = {}
    borrowed = int(index) in {int(i) for i in plan.get('borrowed', [])}
    if borrowed and not context.replay:
        settings['_normalization_fixed'] = _borrowed_values(plan, int(index))
        print(f'Batch {int(index) + 1} has only {len(names)} field(s); it is normalised '
              f'with the median percentiles of the other batches.')
    elif not context.replay:
        settings['_normalization_record'] = record
    normalised = _normalize_img_batch(
        stack=stack, channels=channels, save_dtype=np.dtype(plan['save_dtype']),
        settings=settings,
        field_ids=[os.path.splitext(name)[0] for name in names],
        field_shapes=field_shapes,
        progress=_normalizing_reporter(int(index), len(plan['batches'])))
    _record_normalisation(
        masks_dir, int(index), record if not borrowed and not context.replay else None,
        settings=active_settings)
    normalised = normalised[..., channels]
    return {'data': normalised, 'filenames': np.asarray(names),
            'spacr_resolved_parameters': settings['spacr_resolved_parameters'],
            'source_channels': np.asarray(channels, dtype=np.int64)}


#: A last batch with fewer fields than this fraction of the median size of the
#: other batches is "tiny": it is not given percentiles of its own (item 987b).
_TINY_BATCH_FRACTION = 0.5

#: The batch size the older, archive-based routes use when ``batch_size`` is automatic.
_LEGACY_BATCH_SIZE = 50

_AUTO_WORDS = ('', 'none', 'auto', 'whole')

#: The ``batch_size`` / ``mask_eval_batch_size`` request for VRAM-filling, even,
#: seeded-random batches sized by a calibration pass (item 1011).
_AUTO_VRAM = 'auto_vram'

#: The tile batch cap and the field-order seed when the settings carry none (item 1011).
_DEFAULT_TILE_CAP = 32
_DEFAULT_BATCH_SEED = 42

#: The tile batch of a run with no CUDA device, Cellpose's own default (item 1011).
_CPU_TILE_BATCH = 8


def _batch_size_request(value):
    """What the ``batch_size`` setting asks for.

    :param value: ``'auto_vram'`` (item 1011); 0, -1, None, ``'none'``, ``'auto'``
        or ``'whole'`` (any case) mean item 987b's automatic batch; or a
        positive whole number.
    :returns: ``'auto_vram'``, None for 987b's automatic batch, or the positive
        whole number asked for.
    :raises ValueError: for any other value.
    """
    if isinstance(value, str) and value.strip().lower() == _AUTO_VRAM:
        return _AUTO_VRAM
    if value is None or (isinstance(value, str) and value.strip().lower() in _AUTO_WORDS):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"batch_size must be a number of fields, 'auto_vram', or 0, -1, None, "
                         f"'none', 'auto' or 'whole' for automatic, got {value!r}")
    if number in (0, -1):
        return None
    if number < 0:
        raise ValueError(f"batch_size must be positive, 'auto_vram', or 0, -1, None, 'none', "
                         f"'auto' or 'whole' for automatic, got {value!r}")
    return number


def _legacy_batch_size(settings):
    """Read ``batch_size`` once and leave the routes that need a number a number.

    The request (None for automatic, ``'auto_vram'`` for item 1011) goes into
    ``settings['_batch_size_request']``; when it is either automatic mode,
    ``settings['batch_size']`` becomes the legacy 50 for the archive-based
    routes that need a whole number. A second call changes nothing.

    :param settings: the run settings, changed in place.
    :returns: the request.
    """
    if '_batch_size_request' in settings:
        return settings['_batch_size_request']
    request = _batch_size_request(settings.get('batch_size'))
    settings['_batch_size_request'] = request
    if request is None or request == _AUTO_VRAM:
        settings['batch_size'] = _LEGACY_BATCH_SIZE
    return request


def _even_sizes(n, max_fields):
    """Batch sizes for ``n`` fields that differ by at most one and none above ``max_fields`` (item 1011).

    :param n: the number of fields.
    :param max_fields: the most fields one batch may hold.
    :returns: ``ceil(n / max_fields)`` sizes summing to ``n``, the larger ones first.
    """
    n = int(n)
    if n <= 0:
        return []
    limit = max(1, int(max_fields))
    count = -(-n // limit)
    base, extra = divmod(n, count)
    return [base + (1 if i < extra else 0) for i in range(count)]


def _field_order(names, settings):
    """The order fields are batched in: seeded random, or the given order (item 1011).

    ``randomize`` (default True) is the switch; ``mask_batch_seed`` (default 42)
    seeds one ``random.Random`` per call, so the order never depends on the
    global random state and two processes agree.

    :param names: the field names in acquisition order.
    :param settings: the run settings.
    :returns: a new list of the same names.
    """
    order = list(names)
    settings = settings or {}
    if settings.get('randomize', True) is False:
        return order
    random.Random(_batch_seed(settings)).shuffle(order)
    return order


def _batch_seed(settings):
    """``mask_batch_seed`` as an int, 42 when it is unset.

    :param settings: the run settings.
    :returns: the seed.
    :raises ValueError: when the value is not a whole number.
    """
    value = (settings or {}).get('mask_batch_seed')
    if value in (None, ''):
        return _DEFAULT_BATCH_SEED
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"mask_batch_seed must be a whole number, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"mask_batch_seed must be a whole number, got {value!r}")


def _order_sha256(names):
    """SHA-256 of the newline-joined field names, the record of one batch order.

    :param names: the field names in batch order.
    :returns: the hex digest.
    """
    return hashlib.sha256('\n'.join(str(name) for name in names).encode('utf-8')).hexdigest()


def _tile_cap(settings):
    """``mask_eval_batch_cap`` as a whole number of at least 1, 32 when it is unset.

    :param settings: the run settings.
    :returns: the cap.
    :raises ValueError: when the value is below 1 or not a whole number.
    """
    value = (settings or {}).get('mask_eval_batch_cap')
    if value in (None, ''):
        return _DEFAULT_TILE_CAP
    try:
        cap = int(value)
    except (TypeError, ValueError):
        cap = 0
    if cap < 1 or isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"mask_eval_batch_cap must be a whole number of at least 1, got {value!r}")
    return cap


def _wants_auto_tiles(settings):
    """Whether ``mask_eval_batch_size`` asks for the VRAM-aware tile batch (item 1011).

    :param settings: the run settings.
    """
    value = (settings or {}).get('mask_eval_batch_size')
    return isinstance(value, str) and value.strip().lower() == _AUTO_VRAM


def _resolved_tiles(settings, field_count):
    """The tile batch the network runs at, reading item 1011's calibration when asked for.

    ``mask_eval_batch_size: 'auto_vram'`` takes the calibrated ``tiles`` the run
    holds in ``settings['_mask_calibration']``, or ``min(cap, 8)`` when nothing
    was calibrated (no CUDA device). Any other value goes to
    :func:`spacr.object._eval_tile_batch` unchanged.

    :param settings: the run settings.
    :param field_count: fields in the batch.
    :returns: the ``batch_size`` handed to Cellpose's ``eval``.
    """
    if not _wants_auto_tiles(settings):
        from .object import _eval_tile_batch

        return _eval_tile_batch(settings, field_count)
    calibration = (settings or {}).get('_mask_calibration') or {}
    tiles = calibration.get('tiles')
    if tiles:
        return int(tiles)
    return min(_tile_cap(settings), _CPU_TILE_BATCH)


def _field_costs(plan):
    """Bytes per field of one batch on the GPU and in RAM, from the first field.

    :param plan: from :func:`_read_plan`.
    :returns: ``(gpu, ram)``: the kept channels as float32 on the card; the raw
        stack, the full normalised stack and the kept channels in RAM.
        ``(0, 0)`` when no field can be read.
    """
    masks_dir = plan['_masks_dir']
    stack_dir = os.path.normpath(os.path.join(masks_dir, plan['stack_dir']))
    for batch in plan['batches']:
        for name in batch['fields']:
            from .image_store import _stack_header
            try:
                shape, dtype = _stack_header(os.path.join(stack_dir, name))
            except (OSError, ValueError):
                return 0, 0
            pixels = int(np.prod(shape[:-1]))
            kept = pixels * len(plan['channels']) * 4
            ram = pixels * shape[-1] * (dtype.itemsize + 4) + kept
            return kept, ram
    return 0, 0


def _free_vram_bytes(settings=None):
    """Free memory on the current CUDA device, or None without one.

    With a declared VRAM budget (``gpu_vram_budget_gb`` or ``SPACR_GPU_VRAM_GB``,
    item 1011) it is the smaller of the free memory and that budget, read from
    the governor's one budget source; without one it is the free memory as before.

    :param settings: run settings carrying the governor keys.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free = int(torch.cuda.mem_get_info()[0])
    except Exception:                                       # noqa: BLE001
        return None
    from . import _gpu_governor as governor

    budget = governor._vram_budget(governor._device_index(None), settings)
    if budget['source'] == 'declared':
        return min(free, int(budget['bytes']))
    return free


def _calibrate(model, sample, settings, *, site, evaluate=None, per_field=None,
               governor=None):
    """Measure one tile's and the fixed working set's GPU memory on the run's own data (item 1011).

    Two probe evaluations of ``sample`` at tile batches ``t1 = 2`` and
    ``t2 = min(8, cap)`` are measured with the governor's own rule (reserved
    growth, never below 1.5 times the allocated growth, through
    ``_BatchController._learn`` of ``site``); their results are discarded. A
    linear fit gives ``per_tile`` and ``fixed`` (the weights the process holds,
    image and flow buffers, the fixed working set), each floored at a tenth of
    item 987b's constants. The budget ``B`` (declared, or free less the other
    spaCR processes' announcements; less ``gpu_memory_margin_gb`` and the CUDA
    context; divided by ``mask_workers_per_gpu``) then gives the tile batch
    first, ``clamp(floor((B - fixed - per_field) / per_tile), 1, cap)``, and the
    most fields of one resident batch, ``floor((B - fixed - tiles * per_tile) /
    per_field)``. Without a CUDA device nothing is probed: ``tiles = min(cap, 8)``
    and ``max_fields`` is None (the caller applies its RAM limit, capped at 50).

    A model that names no device runs on the CPU; the current CUDA card is
    never assumed.

    :param model: the loaded segmentation model.
    :param sample: the first field of the seeded order, as the real call gets it.
    :param settings: the run settings (``mask_eval_batch_cap``,
        ``gpu_memory_margin_gb``, ``gpu_vram_budget_gb``, ``mask_workers_per_gpu``).
    :param site: the governor call site the probes are measured on.
    :param evaluate: callable ``tiles -> None`` running one evaluation of
        ``sample`` at that tile batch; ``model.eval(sample, batch_size=tiles)``
        when None.
    :param per_field: bytes one field adds to a resident batch;
        ``sample.nbytes`` when None.
    :param governor: the governor module (:mod:`spacr._gpu_governor`), for tests.
    :returns: the calibration record: ``tiles``, ``max_fields``, ``per_tile``,
        ``fixed``, ``per_field``, ``budget`` (bytes), ``budget_source``
        (``declared``, ``free`` or ``cpu``), ``cap``, ``probes``, ``device``,
        ``total`` and the Cellpose and torch versions.
    """
    if governor is None:
        from . import _gpu_governor as governor
    settings = settings or {}
    cap = _tile_cap(settings)
    per_field = int(getattr(sample, 'nbytes', 0) if per_field is None else per_field)
    device = governor._model_device(model)
    index = None if device is None else governor._device_index(device)
    record = {'cap': cap, 'per_field': per_field, 'versions': _library_versions()}
    if index is None:
        record.update({'tiles': min(cap, _CPU_TILE_BATCH), 'max_fields': None, 'per_tile': None,
                       'fixed': None, 'budget': None, 'budget_source': 'cpu', 'probes': 0,
                       'device': None if device is None else str(device), 'total': None})
        return record
    if evaluate is None:
        evaluate = functools.partial(_evaluate_sample, model, sample)
    gib = 2**30
    config = governor._config(settings)
    controller = governor._controller(site, device)
    held = int(governor._reserved(index)[0])
    need = functools.partial(_probe_need, governor, controller, index, held, evaluate)
    t1, t2 = 2, max(2, min(8, cap))
    floor_tile, floor_fixed = 0.1 * _TILE_GIB * gib, 0.1 * _MODEL_GIB * gib
    probes = 0
    need1 = need(t1)
    probes += 1
    try:
        need2 = need(t2) if t2 > t1 else None
        probes += need2 is not None
    except Exception as error:                              # noqa: BLE001
        if not governor._is_oom(error):
            raise
        governor._release()
        need2 = None
    if need2 is None:
        per_tile = max(floor_tile, need1 / t1)
        fixed = _MODEL_GIB * gib
    else:
        per_tile = max(floor_tile, (need2 - need1) / float(t2 - t1))
        fixed = max(floor_fixed, need1 - t1 * per_tile)
    budget = governor._vram_budget(index, settings)
    usable = (budget['bytes'] if budget['bytes'] is not None else 0) - (
        config['gpu_memory_margin_gb'] * gib + governor._CONTEXT_BYTES)
    workers = int((settings.get('mask_workers_per_gpu') or 1))
    usable = max(0.0, usable / max(1, workers))
    tiles = int(max(1, min(cap, math.floor((usable - fixed - per_field) / per_tile))))
    spare = usable - fixed - tiles * per_tile
    max_fields = max(1, int(spare // per_field)) if per_field > 0 else None
    record.update({'tiles': tiles, 'max_fields': max_fields, 'per_tile': float(per_tile),
                   'fixed': float(fixed), 'budget': float(usable),
                   'budget_source': budget['source'], 'probes': probes,
                   'device': _device_name(governor, index), 'total': budget['total']})
    return record


def _evaluate_sample(model, sample, tiles):
    """One probe evaluation of ``sample`` at a tile batch of ``tiles``; the result is discarded.

    :param model: the loaded segmentation model.
    :param sample: the probe field.
    :param tiles: the tile batch.
    """
    model.eval(sample, batch_size=int(tiles))


def _probe_need(governor, controller, index, held, evaluate, tiles):
    """The bytes one probe at ``tiles`` needs: what the process held plus the governor-measured growth.

    :param governor: the governor module.
    :param controller: the ``_BatchController`` of the probe site.
    :param index: the CUDA device index.
    :param held: bytes the process had reserved before the probes.
    :param evaluate: callable ``tiles -> None`` running one evaluation.
    :param tiles: the tile batch to probe.
    :returns: the need in bytes.
    """
    before = governor._allocated(index)
    reserved_before = governor._reserved(index)
    controller.per_unit = 0.0
    evaluate(tiles)
    controller._learn(index, tiles, before, reserved_before)
    return held + controller.per_unit * tiles


def _library_versions():
    """The Cellpose and torch versions a calibration was measured with."""
    versions = {}
    for name in ('cellpose', 'torch'):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:                                   # noqa: BLE001
            versions[name] = None
    return versions


def _device_name(governor, index):
    """The CUDA device's name, or ``cuda:<index>`` when it cannot be read."""
    torch = governor._torch()
    try:
        return str(torch.cuda.get_device_name(index))
    except Exception:                                       # noqa: BLE001
        return f'cuda:{index}'


def _calibration_line(n_fields, sizes, calibration):
    """The one console line that reports an ``'auto_vram'`` plan (item 1011).

    :param n_fields: fields in the run.
    :param sizes: the batch sizes.
    :param calibration: from :func:`_calibrate`.
    :returns: e.g. ``Mask batches: 1,281 fields -> 6 batches of 213-214 (...)``.
    """
    low, high = (min(sizes), max(sizes)) if sizes else (0, 0)
    span = f'{low}' if low == high else f'{low}–{high}'
    source = calibration.get('budget_source')
    if source == 'cpu':
        budget = 'no CUDA device'
    else:
        budget = (f"VRAM budget {float(calibration.get('budget') or 0) / 2**30:.1f} GB "
                  f"{'declared' if source == 'declared' else 'free'}")
    measured = ('' if not calibration.get('per_tile') else
                f"; {float(calibration['per_tile']) / 2**30:.2f} GB per tile measured")
    return (f"Mask batches: {n_fields:,} fields → {len(sizes)} batch"
            f"{'es' if len(sizes) != 1 else ''} of {span} ({budget}; tile batch "
            f"{calibration.get('tiles')} of cap {calibration.get('cap')}{measured}).")


def _largest_fitting_size(n_fields, per_field_gpu, per_field_ram, *, free_vram, free_ram,
                          margin_gb, tile_batch, model_loaded):
    """The most fields whose normalised batch fits the GPU and RAM, with room to segment.

    Decided before the batch is normalised: the batch on the card, the network
    (unless this process already holds it), the eval working set for
    ``tile_batch`` tiles (``0`` means one tile per field) and the governor
    margin must fit in the free VRAM, and one batch must fit in RAM with 2 GiB spare.

    :returns: a whole number from 1 to ``n_fields``.
    """
    gib = 2**30
    limits = [n_fields]
    if free_vram is not None and per_field_gpu:
        spare = free_vram - margin_gb * gib - (0 if model_loaded else _MODEL_GIB * gib)
        if tile_batch:
            limits.append(int((spare - tile_batch * _TILE_GIB * gib) // per_field_gpu))
        else:
            limits.append(int(spare // (per_field_gpu + _TILE_GIB * gib)))
    if free_ram is not None and per_field_ram:
        limits.append(int((free_ram - 2 * gib) // per_field_ram))
    return max(1, min(limits))


def _batch_sizes(n_fields, size):
    """Batch sizes for ``n_fields`` fields: full batches first, the remainder last."""
    full, rest = divmod(n_fields, size)
    return [size] * full + ([rest] if rest else [])


def _borrowing_indices(sizes):
    """Which batch borrows percentiles: the last, when it is tiny and not the only one.

    :param sizes: the batch sizes in order.
    :returns: ``[]`` or ``[last_index]``.
    """
    if len(sizes) < 2:
        return []
    others = sorted(sizes[:-1])
    middle = len(others) // 2
    median = others[middle] if len(others) % 2 else (others[middle - 1] + others[middle]) / 2
    return [len(sizes) - 1] if sizes[-1] < _TINY_BATCH_FRACTION * median else []


def _resolve_plan(masks_dir, settings, *, free_vram=None, free_ram=None, log=print):
    """Fix the plan's batch sizes before any batch is normalised (item 987b).

    A provisional plan (written by the preprocessing step) is cut into batches
    here, from the free GPU memory and RAM. ``batch_size`` automatic takes the
    largest batch that fits, quietly; a positive number that does not fit is
    reduced with a console warning. The remainder is the last batch, and a
    tiny last batch borrows the other batches' median percentiles. The sizes
    are recorded in ``batch_plan.json`` (``batch_size_used``, ``sizes``,
    ``borrowed``), and a plan that is already resolved for the same request is
    kept, so a rerun or a resume uses the recorded batches.

    :param masks_dir: the ``masks`` folder.
    :param settings: the run settings (``batch_size``, ``gpu_memory_margin_gb``,
        ``mask_eval_batch_size``).
    :param free_vram: free GPU bytes (read from the card when None).
    :param free_ram: free RAM bytes (read from the system when None).
    :param log: callable taking one printed line.
    :returns: the resolved plan, or None when the folder has no plan.
    """
    plan = _read_plan(masks_dir)
    if plan is None:
        return None
    request = _legacy_batch_size(settings)
    if request == _AUTO_VRAM:
        return _resolve_auto_vram(masks_dir, plan, settings, free_ram=free_ram, log=log)
    if plan.get('resolved') and plan.get('batch_size_request') == request:
        return plan
    fields = [name for batch in plan['batches'] for name in batch['fields']]
    per_gpu, per_ram = _field_costs(plan)
    from .object import _eval_tile_batch

    if _wants_auto_tiles(settings):
        tiles = _resolved_tiles(settings, 1)
    else:
        tiles = int((settings or {}).get('mask_eval_batch_size') or 0)
        if tiles:
            tiles = _eval_tile_batch(settings, 1)
    fit = _largest_fitting_size(
        len(fields), per_gpu, per_ram,
        free_vram=_free_vram_bytes() if free_vram is None else free_vram,
        free_ram=_available_ram_bytes() if free_ram is None else free_ram,
        margin_gb=float((settings or {}).get('gpu_memory_margin_gb') or 1.5),
        tile_batch=tiles, model_loaded=_model_resident())
    if request is None:
        used = fit
    elif request > fit:
        used = fit
        log(f'WARNING: batch_size {request} does not fit in the free memory '
            f'({(free_vram if free_vram is not None else (_free_vram_bytes() or 0)) / 2**30:.1f} GiB '
            f'free on the GPU); using {used}.')
    else:
        used = request
    sizes = _batch_sizes(len(fields), used)
    borrowed = _borrowing_indices(sizes)
    batches, start = [], 0
    for size in sizes:
        batches.append(fields[start:start + size])
        start += size
    log(f'Batch size {used} ({"automatic" if request is None else "requested " + str(request)}): '
        f'{len(batches)} batch(es) of {sizes}'
        + (f'; the last has {sizes[-1]} field(s) and borrows the median percentiles of the others.'
           if borrowed else '.'))
    path = _write_plan(masks_dir, os.path.join(masks_dir, plan['stack_dir']), batches,
                       plan['channels'], plan.get('_runtime_settings', plan['settings']),
                       plan['save_dtype'],
                       request=request, resolved=True,
                       extra={'batch_size_used': used, 'sizes': sizes,
                              'borrowed': borrowed, 'normalisation': {}})
    return _read_plan(masks_dir)


def _needs_calibration(masks_dir, settings):
    """Whether the masks stage must run :func:`_calibrate` before :func:`_resolve_plan` (item 1011).

    Only an ``'auto_vram'`` request needs it, and not when ``batch_plan.json`` is
    already resolved for it with the same seed (a resume or a replay keeps the
    recorded calibration, so it probes nothing).

    :param masks_dir: the ``masks`` folder.
    :param settings: the run settings.
    """
    if _batch_size_request((settings or {}).get('_batch_size_request', (settings or {}).get(
            'batch_size'))) != _AUTO_VRAM and not _wants_auto_tiles(settings):
        return False
    plan = _read_plan(masks_dir)
    return not (plan is not None and plan.get('resolved')
                and plan.get('batch_size_request') == _AUTO_VRAM
                and plan.get('seed') == _batch_seed(settings) and plan.get('calibration'))


def _resolve_auto_vram(masks_dir, plan, settings, *, free_ram=None, log=print):
    """Cut a provisional plan into even batches from the run's calibration (item 1011).

    The calibration is ``settings['_mask_calibration']`` (written by the masks
    stage once the model is loaded, see :func:`_calibrate`); without one the
    CPU rule applies. The field count per batch is the calibrated
    ``max_fields`` (CPU: none), limited by RAM with 2 GiB spare and, on the CPU
    route, by the legacy 50, then split evenly. A plan already resolved for
    ``'auto_vram'`` with the same seed and budget source is kept, so a resume
    or a replay does not calibrate again.

    :param masks_dir: the ``masks`` folder.
    :param plan: from :func:`_read_plan`.
    :param settings: the run settings.
    :param free_ram: free RAM bytes (read from the system when None).
    :param log: callable taking one printed line.
    :returns: the resolved plan.
    """
    calibration = dict((settings or {}).get('_mask_calibration') or {})
    seed = _batch_seed(settings)
    source = calibration.get('budget_source', 'cpu')
    if (plan.get('resolved') and plan.get('batch_size_request') == _AUTO_VRAM
            and plan.get('seed') == seed
            and (plan.get('calibration') or {}).get('budget_source') == source):
        if settings is not None and plan.get('calibration'):
            settings['_mask_calibration'] = dict(plan['calibration'])
        return plan
    if not calibration:
        calibration = {'tiles': min(_tile_cap(settings), _CPU_TILE_BATCH), 'max_fields': None,
                       'budget_source': 'cpu', 'cap': _tile_cap(settings), 'probes': 0}
        if settings is not None:
            settings['_mask_calibration'] = dict(calibration)
    fields = [name for batch in plan['batches'] for name in batch['fields']]
    _per_gpu, per_ram = _field_costs(plan)
    ram = _largest_fitting_size(
        len(fields), 0, per_ram, free_vram=None,
        free_ram=_available_ram_bytes() if free_ram is None else free_ram,
        margin_gb=0.0, tile_batch=0, model_loaded=True)
    limit = min(len(fields), ram)
    if calibration.get('max_fields'):
        limit = min(limit, int(calibration['max_fields']))
    if calibration.get('budget_source') == 'cpu':
        limit = min(limit, _LEGACY_BATCH_SIZE)
    limit = max(1, limit)
    sizes = _even_sizes(len(fields), limit)
    batches, start = [], 0
    for size in sizes:
        batches.append(fields[start:start + size])
        start += size
    log(_calibration_line(len(fields), sizes, calibration))
    _write_plan(masks_dir, os.path.join(masks_dir, plan['stack_dir']), batches,
                plan['channels'], plan.get('_runtime_settings', plan['settings']),
                plan['save_dtype'], request=_AUTO_VRAM, resolved=True,
                extra={'batch_size_used': limit, 'sizes': sizes, 'borrowed': [],
                       'normalisation': {}, 'seed': seed,
                       'randomize': (settings or {}).get('randomize', True) is not False,
                       'order_sha256': _order_sha256(fields),
                       'calibration': _json_safe(calibration)})
    return _read_plan(masks_dir)


def _model_resident():
    """Whether this process already holds the network on the card."""
    try:
        import torch

        return bool(torch.cuda.is_available()
                    and torch.cuda.memory_allocated() > 0.5 * _MODEL_GIB * 2**30)
    except Exception:                                       # noqa: BLE001
        return False


def _record_normalisation(masks_dir, index, values, *, settings=None):
    """Record one batch's per-channel lower and upper percentiles in the plan.

    :param masks_dir: the ``masks`` folder.
    :param index: the batch number.
    :param values: ``{channel: [lower, upper]}``, or None to retain donor records.
    :param settings: runtime settings that produced these field records.
    """
    from .io import _replace_atomically

    path = _plan_file(masks_dir)
    previous = _read_plan(masks_dir)
    try:
        with open(path, encoding='utf-8') as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return
    if values is not None:
        record.setdefault('normalisation', {})[str(int(index))] = {
            str(int(channel)): [float(low), float(high)] for channel, (low, high) in values.items()}
    runtime_settings = (settings if settings is not None else
                        previous.get('_runtime_settings') if previous is not None else None)
    if runtime_settings is not None:
        from .run_journal import _RESOLVED_PARAMETERS_LOCK, _validate_resolved_parameters

        with _RESOLVED_PARAMETERS_LOCK:
            record['settings']['spacr_resolved_parameters'] = _validate_resolved_parameters(
                runtime_settings['spacr_resolved_parameters'])
    payload = json.dumps(record, indent=1).encode('utf-8')
    _replace_atomically(path, lambda handle: handle.write(payload), prefix='.spacr_plan_')
    record['_masks_dir'] = os.fspath(masks_dir)
    if runtime_settings is not None:
        record['_runtime_settings'] = runtime_settings
    _CACHE[path] = (os.stat(path).st_mtime_ns, record)


def _borrowed_values(plan, index):
    """The median, per channel, of the other batches' lower and upper percentiles.

    A batch whose percentiles are not recorded yet (a resume that skipped it)
    is normalised once to get them.

    :param plan: from :func:`_read_plan`.
    :param index: the borrowing batch.
    :returns: ``{channel: [lower, upper]}``.
    """
    masks_dir = plan['_masks_dir']
    others = [i for i in range(len(plan['batches'])) if i != int(index)]
    fresh = _read_plan(masks_dir) or plan
    if '_runtime_settings' in plan:
        fresh['_runtime_settings'] = plan['_runtime_settings']
    recorded = dict(fresh.get('normalisation', {}))
    for i in others:
        if str(i) not in recorded:
            _load_batch(fresh, i)
            recorded = dict((_read_plan(masks_dir) or fresh).get('normalisation', {}))
    channels = sorted({channel for i in others for channel in recorded[str(i)]})
    return {channel: [float(np.median([recorded[str(i)][channel][0] for i in others])),
                      float(np.median([recorded[str(i)][channel][1] for i in others]))]
            for channel in channels}


class _InMemoryBatch:
    """An archive stand-in: ``batch['data']`` and ``batch['filenames']`` in a ``with`` block."""

    def __init__(self, arrays):
        """Hold the arrays of one built batch.

        :param arrays: ``{'data', 'filenames', 'source_channels'}`` from :func:`_load_batch`.
        """
        self._arrays = arrays

    def __enter__(self):
        """Return itself, like an opened archive."""
        return self

    def __exit__(self, *exc):
        """Drop the arrays when the block ends."""
        self._arrays = None
        return False

    def __getitem__(self, key):
        """Return one array by name."""
        return self._arrays[key]

    def __contains__(self, key):
        """Whether the batch carries an array of that name."""
        return key in self._arrays

    @property
    def files(self):
        """The array names, as an opened archive lists them."""
        return list(self._arrays)


def _open_batch(path, *, settings=None, **kwargs):
    """Open a batch archive, or build the batch the path names from the plan.

    :param path: an archive path. A file is opened with :func:`numpy.load`;
        a ``stack_<n>_norm.npz`` name with no file behind it is built from
        the plan in its folder.
    :param settings: active run settings for field normalization replay.
    :param kwargs: passed to :func:`numpy.load` for a real archive.
    :returns: an object usable in a ``with`` block that answers ``['data']``
        and ``['filenames']``.
    :raises FileNotFoundError: when there is neither a file nor a plan entry.
    """
    path = os.fspath(path)
    resident = _RESIDENT.get(os.path.abspath(path))
    if resident is not None:
        if settings is not None and 'spacr_resolved_parameters' in resident:
            from .run_journal import _merge_resolved_parameters

            _merge_resolved_parameters(settings, resident['spacr_resolved_parameters'])
        return resident
    if os.path.isfile(path):
        return np.load(path, **kwargs)
    index = _batch_index_of(path)
    plan = _read_plan(os.path.dirname(path)) if index is not None else None
    if plan is None or not 0 <= index < len(plan['batches']):
        raise FileNotFoundError(path)
    return _InMemoryBatch(_load_batch(plan, index, settings=settings))


def _batch_paths(masks_dir, listing):
    """The batches of a ``masks`` folder: real archives first, else the plan's.

    :param masks_dir: the ``masks`` folder.
    :param listing: ``listing(folder)`` returns the visible names in a folder.
    :returns: full paths, archives when the folder holds any, otherwise one
        ``stack_<n>_norm.npz`` path per plan batch (the files do not exist).
    """
    masks_dir = os.fspath(masks_dir)
    archives = [os.path.join(masks_dir, name) for name in listing(masks_dir)
                if name.endswith('.npz')]
    if archives:
        return archives
    plan = _read_plan(masks_dir)
    if plan is None:
        return []
    return [os.path.join(masks_dir, _virtual_batch_name(batch['index']))
            for batch in plan['batches']]


def _supersede_archives(masks_dir, listing):
    """Set old archives aside when a new plan replaces them.

    :param masks_dir: the ``masks`` folder.
    :param listing: ``listing(folder)`` returns the visible names in a folder.
    :returns: the names set aside, renamed to ``<name>.superseded``.
    """
    moved = []
    for name in listing(masks_dir):
        if name.endswith('.npz'):
            os.replace(os.path.join(masks_dir, name),
                       os.path.join(masks_dir, name + '.superseded'))
            moved.append(name)
    return moved


def _plan_records(plan):
    """Manifest records for the parallel mask workers, one per planned batch.

    The records have the shape :func:`spacr.io._mask_batch_manifest` gives
    archives: an absolute (virtual) ``path``, a ``sha256`` identity, a ``bytes``
    size, the batch's ``fields`` and its ``planes``. The identity hashes what
    determines the batch (its fields with their sizes and modification times,
    the kept channels, the dtype and the normalisation settings), so a changed
    input or setting changes it. The size is the fields' bytes, which is what
    the workers' load balancing partitions by.

    :param plan: from :func:`_read_plan`.
    :returns: the records, in batch order.
    :raises ValueError: for an unsafe field file name or a field in two batches.
    """
    import hashlib

    masks_dir = plan['_masks_dir']
    stack_dir = os.path.normpath(os.path.join(masks_dir, plan['stack_dir']))
    signature_settings = {key: value for key, value in plan['settings'].items()
                          if key != 'spacr_resolved_parameters'}
    owners, records = {}, []
    for batch in plan['batches']:
        name = _virtual_batch_name(batch['index'])
        sizes, stamps = [], []
        for field in batch['fields']:
            if (not field or field.startswith('.') or '/' in field or '\\' in field
                    or ':' in field or not field.endswith('.npy') or '\x00' in field):
                raise ValueError(f'{name}: unsafe field filename {field!r}')
            identity = os.path.normcase(field)
            if identity in owners:
                raise ValueError(f'Field {field!r} belongs to both {owners[identity]} and {name}')
            owners[identity] = name
            from .image_store import _stack_stat
            status = _stack_stat(os.path.join(stack_dir, field))
            sizes.append(status.st_size)
            stamps.append(status.st_mtime_ns)
        digest = hashlib.sha256(json.dumps({
            'fields': batch['fields'], 'sizes': sizes, 'mtimes': stamps,
            'channels': plan['channels'], 'dtype': plan['save_dtype'],
            'settings': signature_settings},
            sort_keys=True).encode('utf-8')).hexdigest()
        records.append({'path': os.path.join(os.path.realpath(masks_dir), name),
                        'sha256': digest, 'bytes': int(sum(sizes)),
                        'fields': list(batch['fields']), 'planes': len(plan['channels'])})
    return records


def _available_ram_bytes():
    """MemAvailable in bytes, or None when the platform does not say.

    :returns: free RAM including reclaimable cache, or None.
    """
    try:
        with open('/proc/meminfo', encoding='ascii') as handle:
            for line in handle:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:                                       # noqa: BLE001
        return None


def _batch_ram_estimate(plan, index):
    """Bytes of RAM building one batch needs: its raw stack, the normalised copy and the kept channels.

    :param plan: from :func:`_read_plan`.
    :param index: the batch number.
    :returns: an estimate in bytes, 0 when a field cannot be inspected.
    """
    masks_dir = plan['_masks_dir']
    stack_dir = os.path.normpath(os.path.join(masks_dir, plan['stack_dir']))
    names = _batch_fields(plan, index)
    from .image_store import _stack_header
    try:
        shape, dtype = _stack_header(os.path.join(stack_dir, names[0]))
    except (OSError, ValueError):
        return 0
    pixels = int(np.prod(shape[:-1]))
    raw = pixels * shape[-1] * dtype.itemsize
    full = pixels * shape[-1] * 4
    kept = pixels * len(plan['channels']) * 4
    return int(len(names) * (raw + full + kept))


def _wait_for_ram(estimate, *, margin=2 * 2**30, available=None, sleep=None, checkpoint=None):
    """Wait, instead of loading, while free RAM is short for the next batch.

    :param estimate: bytes the batch needs.
    :param margin: bytes to leave free beyond the estimate.
    :param available: callable returning MemAvailable (the real one by default).
    :param sleep: callable taking seconds (``time.sleep`` by default).
    :param checkpoint: callable run each poll so a cancelled run stops waiting.
    :returns: seconds waited.
    """
    import time

    available = available or _available_ram_bytes
    sleep = sleep or time.sleep
    waited = 0.0
    while waited < _RAM_WAIT_SECONDS:
        free = available()
        if free is None or free >= estimate + margin:
            break
        if checkpoint is not None:
            checkpoint()
        sleep(_RAM_POLL_SECONDS)
        waited += _RAM_POLL_SECONDS
    if waited >= _RAM_WAIT_SECONDS:
        print(f'Free RAM stayed below {(estimate + margin) / 2**30:.1f} GiB for '
              f'{waited:.0f} s; building the next batch anyway.')
    return waited


class _GpuBatch:
    """A normalised batch held in GPU memory that answers like an array.

    ``shape``, ``len`` and slicing are enough for the mask generators:
    a slice is copied back to a numpy array with the same values and dtype
    the batch had, so every eval call gets the bytes it would get from RAM.

    :param tensor: the batch on the device.
    """

    def __init__(self, tensor):
        """Hold the device tensor."""
        self._tensor = tensor
        self.shape = tuple(tensor.shape)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(str(tensor.dtype).replace('torch.', ''))

    def __len__(self):
        """Number of fields."""
        return self.shape[0]

    def __getitem__(self, key):
        """A numpy copy of a slice of the batch."""
        return self._tensor[key].cpu().numpy()

    def free(self):
        """Delete the device copy and give its memory back to the card at once.

        Idempotent. The cache is emptied so the next batch's copy, and other
        processes on the card, see the memory as free.
        """
        tensor, self._tensor = self._tensor, None
        if tensor is not None:
            del tensor
            _empty_cache()


def _empty_cache():
    """Release the CUDA allocator's cached blocks (a no-op without CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass


#: What a batch's copy must leave room for beyond the batch itself: the
#: network's weights when this process has not loaded them yet, and the eval
#: working set, a rough ``_TILE_GIB`` per tile of the tile batch.
_MODEL_GIB = 1.3
_TILE_GIB = 0.2


def _store_on_gpu(array, *, margin_gb=1.5, memory=None, tile_batch=5, model_loaded=None,
                  copy=None, log=print):
    """Copy a normalised batch to the GPU only when all of it fits, with room to segment.

    The check happens before any byte moves: the batch, the network's weights
    (unless this process already holds them), the eval working set for
    ``tile_batch`` tiles and the governor's margin must all fit in the memory
    free right now. A batch that does not fit stays in RAM and the log says so,
    so a copy never starts that could fail halfway.

    :param array: the normalised batch.
    :param margin_gb: free VRAM to leave beyond everything else, in GiB
        (``gpu_memory_margin_gb``).
    :param memory: callable returning ``(free, total)`` bytes (CUDA's by default).
    :param tile_batch: tiles per network pass of the eval that will follow.
    :param model_loaded: whether the weights are already on the card (read from
        the allocator when None).
    :param copy: callable ``array -> device tensor`` (a CUDA copy by default).
    :param log: callable taking the one-line reason a batch stays in RAM.
    :returns: a :class:`_GpuBatch`, or None when there is no usable GPU or the
        batch does not fit, so the caller keeps it in RAM.
    """
    try:
        import torch

        if memory is None and not torch.cuda.is_available():
            return None
        free, _total = (memory or torch.cuda.mem_get_info)()
        if model_loaded is None:
            model_loaded = copy is None and torch.cuda.memory_allocated() > 0.5 * _MODEL_GIB * 2**30
        need = (array.nbytes + (0 if model_loaded else _MODEL_GIB * 2**30)
                + max(1, int(tile_batch)) * _TILE_GIB * 2**30 + margin_gb * 2**30)
        if free < need:
            log(f'Mask batch of {array.nbytes / 2**30:.2f} GiB stays in RAM: it needs '
                f'{need / 2**30:.1f} GiB free on the GPU with the network, a '
                f'{int(tile_batch)}-tile working set and the {margin_gb:g} GiB margin, '
                f'and {free / 2**30:.1f} GiB is free.')
            return None
        tensor = (copy or (lambda a: torch.from_numpy(np.ascontiguousarray(a)).to('cuda')))(array)
        return _GpuBatch(tensor)
    except Exception as error:                              # noqa: BLE001
        log(f'Mask batch stays in RAM: the GPU copy could not start ({error}).')
        return None


class _ResidentBatch:
    """One batch as the mask generators open it: ``['data']`` (a GPU or RAM array) and ``['filenames']``."""

    def __init__(self, arrays, data, release):
        """Keep the answers and the callable that frees the batch."""
        self._arrays = {**arrays, 'data': data}
        self._release = release

    def __enter__(self):
        """Return itself, like an opened archive."""
        return self

    def __exit__(self, *exc):
        """Leave the batch open; the driver frees it after its last role."""
        return False

    def __getitem__(self, key):
        """Return one array by name."""
        return self._arrays[key]

    def __contains__(self, key):
        """Whether the batch carries an array of that name."""
        return key in self._arrays

    def free(self):
        """Free the batch's memory."""
        data = self._arrays.get('data')
        if isinstance(data, _GpuBatch):
            data.free()
        self._arrays = {}
        release, self._release = self._release, None
        if release is not None:
            release()


class _BatchFeeder:
    """Builds the next normalised batch in the background while the GPU works on this one.

    At most one normalised batch is ever in RAM: a semaphore of one is taken
    before a batch is built and given back once it has been copied to the GPU
    (or, when it does not fit there, once the run is done with it). The next
    batch is therefore built while the current one lives in GPU memory.

    :param plan: from :func:`_read_plan`.
    :param indices: batch numbers, in the order they are consumed.
    :param build: callable ``(plan, index) -> arrays`` (:func:`_load_batch`).
    :param store: callable ``array -> _GpuBatch or None`` (:func:`_store_on_gpu`).
    :param wait_for_ram: callable ``estimate -> None`` run before each build.
    """

    def __init__(self, plan, indices, *, build=None, store=None, wait_for_ram=None):
        """Start the background builder."""
        import queue
        import threading

        self._plan = plan
        self._indices = list(indices)
        self._build = build or _load_batch
        self._store = store or _store_on_gpu
        self._wait = wait_for_ram or (lambda estimate: _wait_for_ram(estimate))
        self._slot = threading.Semaphore(1)
        self._ready = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.live = 0
        self.peak_live = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name='spacr-batch-feeder',
                                        daemon=True)
        self._thread.start()

    def _count(self, change):
        """Track how many normalised batches are in RAM right now."""
        with self._lock:
            self.live += change
            self.peak_live = max(self.peak_live, self.live)

    def _run(self):
        """Build each batch once a RAM slot is free, and queue it."""
        try:
            for index in self._indices:
                self._slot.acquire()
                if self._stop.is_set():
                    return
                self._wait(_batch_ram_estimate(self._plan, index))
                arrays = self._build(self._plan, index)
                self._count(1)
                self._ready.put((index, arrays))
        except BaseException as error:                      # noqa: BLE001
            self._ready.put((None, error))

    def next(self):
        """The next batch, copied to the GPU when it fits, otherwise held in RAM.

        :returns: ``(index, _ResidentBatch)``.
        :raises BaseException: whatever the background build raised.
        """
        index, payload = self._ready.get()
        if index is None:
            raise payload
        arrays = payload
        on_gpu = self._store(arrays['data'])
        if on_gpu is not None:
            data = on_gpu
            arrays = {key: value for key, value in arrays.items() if key != 'data'}
            self._count(-1)
            self._slot.release()

            def release():
                """Nothing is left in RAM to free."""
        else:
            data = arrays['data']

            def release():
                """Give the RAM slot back once the batch is done."""
                self._count(-1)
                self._slot.release()
        return index, _ResidentBatch(arrays, data, release)

    def close(self):
        """Stop the builder, dropping anything it queued."""
        self._stop.set()
        self._slot.release()
        try:
            while True:
                self._ready.get_nowait()
        except Exception:                                   # noqa: BLE001
            pass


def _batch_first(mask_src, paths, roles, run_role, *, feeder=None, settings=None):
    """Run every role on one batch before the next batch is loaded.

    :param mask_src: the ``masks`` folder holding the plan.
    :param paths: the batch paths (:func:`_batch_paths`), in order.
    :param roles: the object roles to run on each batch, in order.
    :param run_role: callable ``(role, path)`` that segments one role on one batch.
    :param feeder: a :class:`_BatchFeeder` (built from the plan by default).
    :param settings: run settings: ``gpu_memory_margin_gb`` and the eval tile batch
        size the GPU copy's fit check.
    :returns: the feeder, for its counters.
    """
    plan = _read_plan(mask_src)
    if settings is not None:
        from .run_journal import _resolved_parameters

        _resolved_parameters(settings)
        plan['_runtime_settings'] = settings
    indices = [_batch_index_of(path) for path in paths]
    if feeder is None:
        from .object import _eval_tile_batch

        margin = float((settings or {}).get('gpu_memory_margin_gb') or 1.5)
        tiles = _eval_tile_batch(settings, max(len(_batch_fields(plan, i)) for i in indices))
        feeder = _BatchFeeder(plan, indices, store=lambda array: _store_on_gpu(
            array, margin_gb=margin, tile_batch=tiles))
    try:
        for path in paths:
            _index, resident = feeder.next()
            if settings is not None and 'spacr_resolved_parameters' in resident:
                from .run_journal import _merge_resolved_parameters, _publish_resolved_settings

                _merge_resolved_parameters(settings, resident['spacr_resolved_parameters'])
                _publish_resolved_settings(settings)
            key = os.path.abspath(path)
            _RESIDENT[key] = resident
            try:
                for role in roles:
                    run_role(role, path)
            finally:
                _RESIDENT.pop(key, None)
                resident.free()
    finally:
        feeder.close()
    return feeder
