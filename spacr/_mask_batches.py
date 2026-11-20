from __future__ import annotations

import json
import os

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

    masks_dir = os.fspath(masks_dir)
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
    _CACHE.pop(path, None)
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


def _load_batch(plan, index):
    """Build one normalised batch from its fields, exactly as the archive writer did.

    The fields are loaded, padded to the largest field of the batch when
    their shapes differ, stacked, normalised with the run's own settings and
    cut to the kept channels.

    :param plan: from :func:`_read_plan`.
    :param index: the batch number.
    :returns: ``{'data', 'filenames', 'source_channels'}`` as the archive
        held them.
    """
    from .io import _normalize_img_batch, _normalizing_reporter

    masks_dir = plan['_masks_dir']
    stack_dir = os.path.normpath(os.path.join(masks_dir, plan['stack_dir']))
    names = _batch_fields(plan, index)
    channels = [int(channel) for channel in plan['channels']]
    arrays = [np.load(os.path.join(stack_dir, name)) for name in names]
    shapes = {array.shape[:-1] for array in arrays}
    if len(shapes) > 1:
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
    settings = dict(plan['settings'])
    record = {}
    borrowed = int(index) in {int(i) for i in plan.get('borrowed', [])}
    if borrowed:
        settings['_normalization_fixed'] = _borrowed_values(plan, int(index))
        print(f'Batch {int(index) + 1} has only {len(names)} field(s); it is normalised '
              f'with the median percentiles of the other batches.')
    else:
        settings['_normalization_record'] = record
    normalised = _normalize_img_batch(
        stack=stack, channels=channels, save_dtype=np.dtype(plan['save_dtype']),
        settings=settings,
        progress=_normalizing_reporter(int(index), len(plan['batches'])))
    if record:
        _record_normalisation(masks_dir, int(index), record)
    normalised = normalised[..., channels]
    return {'data': normalised, 'filenames': np.asarray(names),
            'source_channels': np.asarray(channels, dtype=np.int64)}


#: A last batch with fewer fields than this fraction of the median size of the
#: other batches is "tiny": it is not given percentiles of its own (item 987b).
_TINY_BATCH_FRACTION = 0.5

#: The batch size the older, archive-based routes use when ``batch_size`` is automatic.
_LEGACY_BATCH_SIZE = 50

_AUTO_WORDS = ('', 'none', 'auto')


def _batch_size_request(value):
    """What the ``batch_size`` setting asks for.

    :param value: 0, -1, None, ``'none'`` or ``'auto'`` (any case) mean automatic.
    :returns: None for automatic, otherwise the positive whole number asked for.
    :raises ValueError: for any other value.
    """
    if value is None or (isinstance(value, str) and value.strip().lower() in _AUTO_WORDS):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"batch_size must be a number of fields or 0, -1, None, 'none' or "
                         f"'auto' for automatic, got {value!r}")
    if number in (0, -1):
        return None
    if number < 0:
        raise ValueError(f"batch_size must be positive, or 0, -1, None, 'none' or 'auto' "
                         f"for automatic, got {value!r}")
    return number


def _legacy_batch_size(settings):
    """Read ``batch_size`` once and leave the routes that need a number a number.

    The request (None for automatic) goes into ``settings['_batch_size_request']``;
    when it is automatic, ``settings['batch_size']`` becomes the legacy 50 for
    the archive-based routes that need a whole number. A second call changes nothing.

    :param settings: the run settings, changed in place.
    :returns: the request.
    """
    if '_batch_size_request' in settings:
        return settings['_batch_size_request']
    request = _batch_size_request(settings.get('batch_size'))
    settings['_batch_size_request'] = request
    if request is None:
        settings['batch_size'] = _LEGACY_BATCH_SIZE
    return request


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
            try:
                first = np.load(os.path.join(stack_dir, name), mmap_mode='r')
            except (OSError, ValueError):
                return 0, 0
            pixels = int(np.prod(first.shape[:-1]))
            kept = pixels * len(plan['channels']) * 4
            ram = pixels * first.shape[-1] * (first.dtype.itemsize + 4) + kept
            return kept, ram
    return 0, 0


def _free_vram_bytes():
    """Free memory on the current CUDA device, or None without one."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.mem_get_info()[0])
    except Exception:                                       # noqa: BLE001
        return None


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
    if plan.get('resolved') and plan.get('batch_size_request') == request:
        return plan
    fields = [name for batch in plan['batches'] for name in batch['fields']]
    per_gpu, per_ram = _field_costs(plan)
    from .object import _eval_tile_batch

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
                       plan['channels'], plan['settings'], plan['save_dtype'],
                       request=request, resolved=True,
                       extra={'batch_size_used': used, 'sizes': sizes,
                              'borrowed': borrowed, 'normalisation': {}})
    return _read_plan(masks_dir)


def _model_resident():
    """Whether this process already holds the network on the card."""
    try:
        import torch

        return bool(torch.cuda.is_available()
                    and torch.cuda.memory_allocated() > 0.5 * _MODEL_GIB * 2**30)
    except Exception:                                       # noqa: BLE001
        return False


def _record_normalisation(masks_dir, index, values):
    """Record one batch's per-channel lower and upper percentiles in the plan.

    :param masks_dir: the ``masks`` folder.
    :param index: the batch number.
    :param values: ``{channel: [lower, upper]}``.
    """
    from .io import _replace_atomically

    path = _plan_file(masks_dir)
    try:
        with open(path, encoding='utf-8') as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return
    record.setdefault('normalisation', {})[str(int(index))] = {
        str(int(channel)): [float(low), float(high)] for channel, (low, high) in values.items()}
    payload = json.dumps(record, indent=1).encode('utf-8')
    _replace_atomically(path, lambda handle: handle.write(payload), prefix='.spacr_plan_')
    _CACHE.pop(path, None)


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


def _open_batch(path, **kwargs):
    """Open a batch archive, or build the batch the path names from the plan.

    :param path: an archive path. A file is opened with :func:`numpy.load`;
        a ``stack_<n>_norm.npz`` name with no file behind it is built from
        the plan in its folder.
    :param kwargs: passed to :func:`numpy.load` for a real archive.
    :returns: an object usable in a ``with`` block that answers ``['data']``
        and ``['filenames']``.
    :raises FileNotFoundError: when there is neither a file nor a plan entry.
    """
    path = os.fspath(path)
    resident = _RESIDENT.get(os.path.abspath(path))
    if resident is not None:
        return resident
    if os.path.isfile(path):
        return np.load(path, **kwargs)
    index = _batch_index_of(path)
    plan = _read_plan(os.path.dirname(path)) if index is not None else None
    if plan is None or not 0 <= index < len(plan['batches']):
        raise FileNotFoundError(path)
    return _InMemoryBatch(_load_batch(plan, index))


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
            status = os.stat(os.path.join(stack_dir, field))
            sizes.append(status.st_size)
            stamps.append(status.st_mtime_ns)
        digest = hashlib.sha256(json.dumps({
            'fields': batch['fields'], 'sizes': sizes, 'mtimes': stamps,
            'channels': plan['channels'], 'dtype': plan['save_dtype'],
            'settings': plan['settings']}, sort_keys=True).encode('utf-8')).hexdigest()
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
    try:
        first = np.load(os.path.join(stack_dir, names[0]), mmap_mode='r')
    except (OSError, ValueError):
        return 0
    pixels = int(np.prod(first.shape[:-1]))
    raw = pixels * first.shape[-1] * first.dtype.itemsize
    full = pixels * first.shape[-1] * 4
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
