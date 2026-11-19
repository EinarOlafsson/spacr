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


def _write_plan(masks_dir, stack_dir, batches, channels, settings, save_dtype):
    """Write the batch plan next to where the archives used to go.

    :param masks_dir: the ``masks`` folder.
    :param stack_dir: the ``stack`` folder holding the field ``.npy`` files.
    :param batches: a list of lists of field file names, one list per batch,
        in batch order.
    :param channels: the channel indices every batch keeps.
    :param settings: the preprocessing settings the normaliser reads.
    :param save_dtype: the dtype the normalised batches are returned in.
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
    normalised = _normalize_img_batch(
        stack=stack, channels=channels, save_dtype=np.dtype(plan['save_dtype']),
        settings=plan['settings'],
        progress=_normalizing_reporter(int(index), len(plan['batches'])))
    normalised = normalised[..., channels]
    return {'data': normalised, 'filenames': np.asarray(names),
            'source_channels': np.asarray(channels, dtype=np.int64)}


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
        """Drop the device copy."""
        self._tensor = None


def _store_on_gpu(array, *, margin_gb=1.5, memory=None):
    """Copy a normalised batch to the GPU when it fits with the governor's margin.

    :param array: the normalised batch.
    :param margin_gb: free VRAM to leave beyond the batch, in GiB.
    :param memory: callable returning ``(free, total)`` bytes (CUDA's by default).
    :returns: a :class:`_GpuBatch`, or None when there is no usable GPU or the
        batch does not fit, so the caller keeps it in RAM.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = (memory or torch.cuda.mem_get_info)()
        if free < array.nbytes + margin_gb * 2**30:
            return None
        return _GpuBatch(torch.from_numpy(np.ascontiguousarray(array)).to('cuda'))
    except Exception:                                       # noqa: BLE001
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


def _batch_first(mask_src, paths, roles, run_role, *, feeder=None):
    """Run every role on one batch before the next batch is loaded.

    :param mask_src: the ``masks`` folder holding the plan.
    :param paths: the batch paths (:func:`_batch_paths`), in order.
    :param roles: the object roles to run on each batch, in order.
    :param run_role: callable ``(role, path)`` that segments one role on one batch.
    :param feeder: a :class:`_BatchFeeder` (built from the plan by default).
    :returns: the feeder, for its counters.
    """
    plan = _read_plan(mask_src)
    indices = [_batch_index_of(path) for path in paths]
    feeder = feeder or _BatchFeeder(plan, indices)
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
