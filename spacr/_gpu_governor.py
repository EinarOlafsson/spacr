"""Keep spaCR's GPU work inside the memory a shared card has free.

A workstation GPU is often shared by several jobs: a segmentation run, a
measurement run and a training run can all hold memory on one card. Each of
them sizes its batches for an empty card, so whichever allocates last runs
out of memory and, before this module, stopped. This module is the one place
that decides how much a spaCR GPU call may take:

* before each batch it reads the free memory of the device and, when the
  batch is estimated not to fit with ``gpu_memory_margin_gb`` to spare,
  shrinks the batch or waits for memory to be freed;
* on an out-of-memory error from PyTorch or CuPy it releases the cached
  blocks, halves the batch and retries the same item; after a few failed
  retries at batch size 1 the item runs on the CPU. No item is dropped and
  the run never stops on an out-of-memory error;
* after each item it releases the cached memory, so a co-tenant can use it;
* after a run of clean items it doubles the batch again, up to the size the
  caller asked for;
* ``gpu_memory_fraction`` sets a hard ceiling through
  ``torch.cuda.set_per_process_memory_fraction``;
* a ledger file under the user cache directory records what every spaCR GPU
  process on the machine is using and about to claim, so a new job counts
  what the others are about to take and a running job shrinks while room is
  needed. Entries of processes that no longer exist are ignored;
* the shrink, wait, retry, fallback and ramp-up counts are summarised in the
  run log at the end of a run.

``gpu_adaptive_batch=False`` turns the batch control off: every call then
runs exactly as asked, with only the optional fraction applied. On a machine
without a visible CUDA device every call also runs exactly as asked.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from typing import Any, Callable, Dict, Mapping, Optional

_GIB = float(1 << 30)

_DEFAULTS = {
    'gpu_memory_fraction': None,
    'gpu_memory_margin_gb': 1.5,
    'gpu_adaptive_batch': True,
}

_RETRIES_AT_ONE = 2
_RAMP_AFTER = 8
_WAIT_SECONDS = 120.0
_POLL_SECONDS = 2.0
_LEDGER_FILE = 'gpu_ledger.json'
_LEDGER_INTERVAL = 1.0
_LEDGER_SLACK = 64 << 20

_COUNTER_NAMES = ('items', 'shrinks', 'waits', 'oom_retries', 'halvings',
                  'cpu_fallbacks', 'ramp_ups')

_sleep = time.sleep
_monotonic = time.monotonic

_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    'config': dict(_DEFAULTS),
    'fractions': {},
    'controllers': {},
}
_COUNTS: Counter = Counter()


def _config(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return the governor settings in force.

    :param settings: run settings; any of the three governor keys they hold
        override the process-wide values set by :func:`_configure`.
    :returns: a dict with the three governor keys.
    """
    merged = dict(_STATE['config'])
    if isinstance(settings, Mapping):
        for key in _DEFAULTS:
            if key in settings:
                merged[key] = settings[key]
    return _normalised(merged)


def _normalised(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce the three governor values, falling back to the defaults.

    :param values: raw values, possibly from a settings file.
    :returns: the fraction as a float in (0, 1] or None, the margin as a
        non-negative float and the switch as a bool.
    """
    fraction = values.get('gpu_memory_fraction')
    try:
        fraction = None if fraction in (None, '', 0) else float(fraction)
    except (TypeError, ValueError):
        fraction = None
    if fraction is not None and not 0.0 < fraction <= 1.0:
        fraction = None
    try:
        margin = max(0.0, float(values.get('gpu_memory_margin_gb', 1.5)))
    except (TypeError, ValueError):
        margin = float(_DEFAULTS['gpu_memory_margin_gb'])
    adaptive = values.get('gpu_adaptive_batch', True) is not False
    return {'gpu_memory_fraction': fraction, 'gpu_memory_margin_gb': margin,
            'gpu_adaptive_batch': adaptive}


def _configure(settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Set the process-wide governor values from a run's settings.

    :param settings: the run settings; keys it lacks keep their defaults.
    :returns: the previous process-wide values, for :func:`_restore`.
    """
    previous = dict(_STATE['config'])
    merged = dict(_DEFAULTS)
    if isinstance(settings, Mapping):
        merged.update({key: settings[key] for key in _DEFAULTS
                       if key in settings})
    _STATE['config'] = _normalised(merged)
    return previous


def _restore(previous: Mapping[str, Any]) -> None:
    """Put back the process-wide values :func:`_configure` replaced.

    :param previous: what :func:`_configure` returned.
    """
    _STATE['config'] = _normalised(previous)


def _count(name: str, amount: int = 1) -> None:
    """Add to one of the run-report counters.

    :param name: one of the counter names.
    :param amount: how much to add.
    """
    with _LOCK:
        _COUNTS[name] += amount


def _counters() -> Dict[str, int]:
    """Return a copy of every counter, zeros included."""
    with _LOCK:
        return {name: int(_COUNTS.get(name, 0)) for name in _COUNTER_NAMES}


def _counters_since(start: Mapping[str, int]) -> Dict[str, int]:
    """Return what each counter gained since a snapshot.

    :param start: a :func:`_counters` snapshot.
    :returns: the gain per counter.
    """
    now = _counters()
    return {name: now[name] - int(start.get(name, 0)) for name in now}


def _summary(counts: Mapping[str, int]) -> str:
    """Describe the governor's work in one line, or ``""`` when it did nothing.

    :param counts: counter gains, as :func:`_counters_since` returns them.
    """
    adapted = {name: int(counts.get(name, 0)) for name in _COUNTER_NAMES
               if name != 'items'}
    if not any(adapted.values()):
        return ''
    return ('GPU memory: {items} GPU items, {shrinks} batches shrunk to fit, '
            '{waits} waits for free memory, {oom_retries} out-of-memory '
            'retries ({halvings} halvings), {cpu_fallbacks} CPU fallbacks, '
            '{ramp_ups} ramp-ups').format(
                items=int(counts.get('items', 0)), **adapted)


def _is_oom(error: BaseException) -> bool:
    """Whether an exception is a GPU out-of-memory error.

    Matched by class name as well as by message, so neither PyTorch nor CuPy
    has to be imported to ask.

    :param error: the exception raised by a GPU call.
    """
    for kind in type(error).__mro__:
        if kind.__name__ in ('OutOfMemoryError', 'CUDARuntimeError'):
            if kind.__name__ == 'OutOfMemoryError':
                return True
            return 'out of memory' in str(error).lower()
    if isinstance(error, RuntimeError):
        text = str(error).lower()
        return 'out of memory' in text and ('cuda' in text or 'cudnn' in text
                                            or 'cublas' in text
                                            or 'allocate' in text)
    return False


def _torch():
    """The already-imported PyTorch module, or None.

    The governor never imports PyTorch itself: a caller with GPU work has
    imported it already, and a CPU-only run must not pay for the import.
    """
    return sys.modules.get('torch')


def _device_index(device: Any = None) -> Optional[int]:
    """The CUDA device index of ``device``, or None when it is not CUDA.

    :param device: a ``torch.device``, a string such as ``"cuda:1"``, an
        int, or None for the current CUDA device.
    """
    torch = _torch()
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
    except Exception:                                       # noqa: BLE001
        return None
    if device is None:
        try:
            return int(torch.cuda.current_device())
        except Exception:                                   # noqa: BLE001
            return 0
    if isinstance(device, int):
        return device
    kind = getattr(device, 'type', None)
    if kind is None:
        text = str(device)
        kind, _, index = text.partition(':')
        if kind != 'cuda':
            return None
        return int(index) if index.isdigit() else 0
    if kind != 'cuda':
        return None
    index = getattr(device, 'index', None)
    return 0 if index is None else int(index)


def _memory(index: int):
    """``(free, total)`` bytes of one CUDA device, or None when unreadable.

    :param index: the CUDA device index.
    """
    torch = _torch()
    try:
        free, total = torch.cuda.mem_get_info(index)
        return int(free), int(total)
    except Exception:                                       # noqa: BLE001
        return None


def _allocated(index: int) -> tuple:
    """``(allocated, peak allocated)`` bytes of this process on one device.

    :param index: the CUDA device index.
    :returns: two ints, zeros when PyTorch cannot say.
    """
    torch = _torch()
    try:
        return (int(torch.cuda.memory_allocated(index)),
                int(torch.cuda.max_memory_allocated(index)))
    except Exception:                                       # noqa: BLE001
        return 0, 0


def _release() -> None:
    """Return cached GPU blocks to the driver, for PyTorch and CuPy.

    CuPy is only touched when something has imported it already.
    """
    torch = _torch()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass
    cupy = sys.modules.get('cupy')
    if cupy is not None:
        try:
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:                                   # noqa: BLE001
            pass


def _apply_fraction(index: Optional[int], config: Mapping[str, Any]) -> None:
    """Cap this process at ``gpu_memory_fraction`` of one device, once.

    :param index: the CUDA device index, or None for no CUDA device.
    :param config: the governor values in force.
    """
    fraction = config.get('gpu_memory_fraction')
    if index is None or fraction is None:
        return
    if _STATE['fractions'].get(index) == fraction:
        return
    torch = _torch()
    try:
        torch.cuda.set_per_process_memory_fraction(float(fraction), index)
        _STATE['fractions'][index] = fraction
    except Exception as error:                              # noqa: BLE001
        print(f"[gpu] could not cap this process at {fraction:.2f} of "
              f"cuda:{index}: {error}")
        _STATE['fractions'][index] = fraction


def _cache_dir() -> str:
    """The spaCR folder under the user cache directory."""
    base = os.environ.get('XDG_CACHE_HOME', '').strip()
    if not base:
        if sys.platform == 'win32':
            base = os.environ.get('LOCALAPPDATA', '') or os.path.join(
                os.path.expanduser('~'), 'AppData', 'Local')
        elif sys.platform == 'darwin':
            base = os.path.join(os.path.expanduser('~'), 'Library', 'Caches')
        else:
            base = os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'spacr')


def _ledger_path() -> str:
    """Where the machine-wide GPU ledger lives."""
    return os.path.join(_cache_dir(), _LEDGER_FILE)


def _process_start(pid: int) -> Optional[float]:
    """When a process started, or None when that cannot be read.

    :param pid: the process id.
    """
    try:
        import psutil

        return float(psutil.Process(int(pid)).create_time())
    except Exception:                                       # noqa: BLE001
        return None


def _alive(pid: int, started: Any = None) -> bool:
    """Whether the process that wrote a ledger entry still runs.

    A recorded start time that differs from the live one means the id was
    reused by another process, so the entry is stale.

    :param pid: the process id.
    :param started: the start time recorded with the entry, if any.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
    except ImportError:
        if sys.platform == 'win32':
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
    if started is None:
        return True
    live = _process_start(pid)
    return live is None or abs(float(live) - float(started)) < 1.0


def _read_ledger(path: str) -> Dict[str, Any]:
    """Read the ledger entries, an empty dict for a missing or broken file.

    :param path: the ledger file.
    """
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    entries = data.get('entries') if isinstance(data, dict) else None
    return dict(entries) if isinstance(entries, dict) else {}


def _write_ledger(path: str, entries: Mapping[str, Any]) -> None:
    """Replace the ledger atomically: write a temporary file, then rename it.

    :param path: the ledger file.
    :param entries: every live entry.
    """
    folder = os.path.dirname(path)
    handle, pending = tempfile.mkstemp(prefix='.gpu_ledger.', dir=folder)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump({'entries': dict(entries)}, stream)
        os.replace(pending, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(pending)
        raise


@contextlib.contextmanager
def _ledger_lock(path: str):
    """Hold an exclusive lock on the ledger for one read-modify-write.

    POSIX ``flock`` on a side file; where that is missing the block runs
    unlocked, which only costs the precision of one update.

    :param path: the ledger file.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(path + '.lock', 'a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ledger_update(index: int, used: int = 0, want: int = 0,
                   ceiling: int = 0, *, remove: bool = False,
                   path: Optional[str] = None) -> Dict[str, Any]:
    """Record this process's use of one device and read everyone else's.

    :param index: the CUDA device index.
    :param used: bytes this process holds on it now.
    :param want: bytes it is about to need for its next batch.
    :param ceiling: its cap in bytes, 0 for none.
    :param remove: drop this process's entry instead.
    :param path: the ledger file; the user cache one when omitted.
    :returns: the live entries of the OTHER processes on that device.
    """
    path = path or _ledger_path()
    key = f"{os.getpid()}:{index}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _ledger_lock(path):
            entries = {name: entry for name, entry in _read_ledger(path).items()
                       if isinstance(entry, dict)
                       and _alive(entry.get('pid'), entry.get('started'))}
            if remove:
                entries.pop(key, None)
            else:
                entries[key] = {
                    'pid': os.getpid(), 'device': int(index),
                    'started': _process_start(os.getpid()),
                    'used': int(used), 'want': int(want),
                    'ceiling': int(ceiling), 'updated': time.time(),
                }
            _write_ledger(path, entries)
    except OSError:
        return {}
    return {name: entry for name, entry in entries.items()
            if name != key and int(entry.get('device', -1)) == int(index)}


def _pending_elsewhere(others: Mapping[str, Any]) -> int:
    """Bytes other spaCR processes are about to allocate on top of what they hold.

    :param others: entries from :func:`_ledger_update`.
    """
    pending = 0
    for entry in others.values():
        try:
            pending += max(0, int(entry.get('want', 0)) - int(entry.get('used', 0)))
        except (TypeError, ValueError):
            continue
    return pending


def _model_device(model: Any) -> Any:
    """The device a model runs on, looked up the ways spaCR models expose it.

    :param model: a Cellpose model, a spaCR backend adapter or a module.
    """
    for target in _model_chain(model):
        device = getattr(target, 'device', None)
        if device is not None and not callable(device):
            return device
        parameters = getattr(target, 'parameters', None)
        if callable(parameters):
            try:
                return next(iter(parameters())).device
            except Exception:                               # noqa: BLE001
                continue
    return None


def _model_chain(model: Any):
    """The model and the objects it wraps, outermost first, at most four deep.

    :param model: a Cellpose model, a spaCR backend adapter or a module.
    """
    chain = []
    current = model
    while (current is not None and len(chain) < 4
           and not any(current is item for item in chain)):
        chain.append(current)
        wrapped = None
        for name in ('_model', 'model', '_backend'):
            try:
                candidate = getattr(current, name, None)
            except Exception:                               # noqa: BLE001
                candidate = None
            if candidate is None or candidate is current:
                continue
            if any(hasattr(candidate, attr)
                   for attr in ('net', 'parameters', 'eval')):
                wrapped = candidate
                break
        current = wrapped
    return chain


def _cpu_target(model: Any):
    """The object that has to move for ``model`` to run on the CPU, or None.

    :param model: a Cellpose model, a spaCR backend adapter or a module.
    :returns: ``("cellpose", obj)`` for an object with ``net`` and
        ``device``, ``("module", obj)`` for a PyTorch module, or None.
    """
    torch = _torch()
    if torch is None:
        return None
    for target in _model_chain(model):
        if hasattr(target, 'net') and hasattr(target, 'device'):
            return 'cellpose', target
        if isinstance(target, getattr(torch.nn, 'Module', ())):
            return 'module', target
    return None


@contextlib.contextmanager
def _on_cpu(model: Any):
    """Run the block with ``model`` moved to the CPU, then move it back.

    :param model: a Cellpose model, a spaCR backend adapter or a module.
    """
    torch = _torch()
    found = _cpu_target(model)
    if found is None:
        yield model
        return
    kind, target = found
    cpu = torch.device('cpu')
    if kind == 'cellpose':
        old_device, old_gpu = target.device, getattr(target, 'gpu', None)
        target.net.to(cpu)
        target.device = cpu
        if old_gpu is not None:
            target.gpu = False
        try:
            yield model
        finally:
            target.net.to(old_device)
            target.device = old_device
            if old_gpu is not None:
                target.gpu = old_gpu
        return
    old_device = _model_device(target)
    target.to(cpu)
    try:
        yield model
    finally:
        if old_device is not None:
            target.to(old_device)


class _BatchController:
    """Pick, guard and adapt the batch size of one GPU call site.

    One controller per call site keeps what that site learned -- the batch
    that last fitted and the memory one batch unit took -- across items.

    :param site: a short name for the call site, used in messages.
    """

    def __init__(self, site: str):
        """Start with no cap and nothing learned."""
        self.site = site
        self.cap: Optional[int] = None
        self.clean = 0
        self.per_unit = 0.0
        self._last_post = None

    def run(self, fn: Callable[[int], Any], requested: int, *,
            device: Any = None, cpu_fn: Optional[Callable[[int], Any]] = None,
            config: Optional[Mapping[str, Any]] = None,
            on_retry: Optional[Callable[[], None]] = None) -> Any:
        """Run ``fn(batch_size)`` on the GPU within the memory that is free.

        :param fn: does the work at a given batch size and returns its
            result; it is called again with the same item on a retry.
        :param requested: the batch size the caller asked for.
        :param device: the CUDA device the work runs on; None for current.
        :param cpu_fn: does the same work on the CPU at a given batch size,
            after the GPU failed at batch size 1.
        :param config: the governor values; :func:`_config` when omitted.
        :param on_retry: undoes what a failed attempt left behind, such as
            accumulated gradients, before the next attempt.
        :returns: what ``fn`` (or ``cpu_fn``) returned.
        """
        config = _config() if config is None else config
        requested = max(1, int(requested or 1))
        index = _device_index(device)
        _apply_fraction(index, config)
        if index is None or not config['gpu_adaptive_batch']:
            return fn(requested)
        size = min(requested, self.cap or requested)
        size = self._fit(size, index, config)
        at_one = 0
        waited = 0.0
        while True:
            before, peak_before = _allocated(index)
            self._post(index, want=int(self.per_unit * size) + before,
                       used=before, config=config)
            try:
                result = fn(size)
            except BaseException as error:
                if not _is_oom(error):
                    raise
                _release()
                _count('oom_retries')
                self.clean = 0
                if on_retry is not None:
                    on_retry()
                if size > 1:
                    size = max(1, size // 2)
                    self.cap = size
                    _count('halvings')
                    print(f"[gpu] {self.site}: out of GPU memory; retrying "
                          f"the same item at batch size {size}")
                    continue
                at_one += 1
                if at_one <= _RETRIES_AT_ONE:
                    _sleep(_POLL_SECONDS)
                    continue
                if cpu_fn is not None:
                    _count('cpu_fallbacks')
                    print(f"[gpu] {self.site}: out of GPU memory at batch "
                          f"size 1; running this item on the CPU")
                    result = cpu_fn(size)
                    break
                if waited < _WAIT_SECONDS:
                    _count('waits')
                    _sleep(_POLL_SECONDS)
                    waited += _POLL_SECONDS
                    continue
                raise
            else:
                self._learn(index, size, before, peak_before)
                _count('items')
                self.clean += 1
                if (self.cap is not None and self.cap < requested
                        and self.clean >= _RAMP_AFTER):
                    self.cap = min(requested, self.cap * 2)
                    self.clean = 0
                    _count('ramp_ups')
                    if self.cap >= requested:
                        self.cap = None
                break
        _release()
        self._post(index, want=0, used=_allocated(index)[0], config=config)
        return result

    def _learn(self, index: int, size: int, before: int,
               peak_before: int) -> None:
        """Estimate the memory one batch unit takes from the last call.

        The peak counter is read, never reset, so other readers of it keep
        their figure. A call that raised the peak measured its own need
        exactly. One that did not only bounds it from above, so that bound
        is used only while there is no estimate yet.

        :param index: the CUDA device index.
        :param size: the batch size that ran.
        :param before: bytes allocated before the call.
        :param peak_before: the peak allocation before the call.
        """
        _now, peak = _allocated(index)
        if peak <= peak_before and self.per_unit > 0:
            return
        used = max(peak, peak_before) - before
        if used > 0:
            self.per_unit = used / float(max(1, size))

    def _budget(self, index: int, config: Mapping[str, Any]) -> Optional[float]:
        """Bytes this site may still allocate on one device, or None if unknown.

        Free memory, less the margin, less what other spaCR processes have
        said they are about to take, and no more than the fraction ceiling.

        :param index: the CUDA device index.
        :param config: the governor values.
        """
        info = _memory(index)
        if info is None:
            return None
        free, total = info
        others = self._post(index, want=0, used=_allocated(index)[0],
                            config=config)
        budget = (free - config['gpu_memory_margin_gb'] * _GIB
                  - _pending_elsewhere(others))
        fraction = config.get('gpu_memory_fraction')
        if fraction is not None:
            budget = min(budget, fraction * total - _allocated(index)[0])
        return float(budget)

    def _fit(self, size: int, index: int, config: Mapping[str, Any]) -> int:
        """Shrink ``size`` to the free memory, or wait for memory to free.

        :param size: the batch size about to run.
        :param index: the CUDA device index.
        :param config: the governor values.
        :returns: the batch size to run, at least 1.
        """
        budget = self._budget(index, config)
        if budget is None:
            return size
        if self.per_unit > 0 and budget >= self.per_unit * size:
            return size
        if self.per_unit > 0 and budget >= self.per_unit:
            fitted = max(1, int(budget // self.per_unit))
            if fitted < size:
                _count('shrinks')
                print(f"[gpu] {self.site}: {budget / _GIB:.1f} GiB free after "
                      f"the margin; batch size {size} -> {fitted}")
                self.cap = fitted
                self.clean = 0
                return fitted
            return size
        if self.per_unit <= 0 and budget > 0:
            return size
        _count('waits')
        print(f"[gpu] {self.site}: waiting for GPU memory to be freed")
        waited = 0.0
        while waited < _WAIT_SECONDS:
            _release()
            _sleep(_POLL_SECONDS)
            waited += _POLL_SECONDS
            budget = self._budget(index, config)
            if budget is None or budget >= max(self.per_unit, 1.0):
                break
        if self.per_unit > 0 and budget is not None:
            fitted = max(1, int(max(budget, 0) // self.per_unit))
            if fitted < size:
                self.cap = fitted
                self.clean = 0
                return fitted
        return size

    def _post(self, index: int, *, want: int, used: int,
              config: Mapping[str, Any]) -> Dict[str, Any]:
        """Write this process's line in the ledger and read the others'.

        :param index: the CUDA device index.
        :param want: bytes about to be needed.
        :param used: bytes held now.
        :param config: the governor values.
        """
        last = self._last_post
        if (last is not None and last[0] == index
                and _monotonic() - last[1] < _LEDGER_INTERVAL
                and abs(want - last[2]) < _LEDGER_SLACK
                and abs(used - last[3]) < _LEDGER_SLACK):
            return last[4]
        ceiling = 0
        fraction = config.get('gpu_memory_fraction')
        if fraction is not None:
            info = _memory(index)
            if info is not None:
                ceiling = int(fraction * info[1])
        others = _ledger_update(index, used=used, want=want, ceiling=ceiling)
        self._last_post = (index, _monotonic(), want, used, others)
        return others


def _controller(site: str, device: Any = None) -> _BatchController:
    """The controller of one call site on one device, created on first use.

    :param site: the call site's name.
    :param device: the device the site runs on; sites on several GPUs keep
        one controller per GPU.
    """
    key = site if device is None else f"{site}@{device}"
    with _LOCK:
        found = _STATE['controllers'].get(key)
        if found is None:
            found = _STATE['controllers'][key] = _BatchController(site)
        return found


def _governed_eval(model: Any, x: Any, *, site: str, batch_size: Any = 8,
                   settings: Optional[Mapping[str, Any]] = None,
                   **kwargs: Any) -> Any:
    """Call ``model.eval(x, batch_size=..., **kwargs)`` under the governor.

    The batch size is the number of tiles Cellpose sends through the network
    at once, so halving it retries the same images with less memory. The
    CPU fallback moves the network to the CPU for that call only.

    :param model: a Cellpose model or a spaCR backend with the same ``eval``.
    :param x: the image or list of images.
    :param site: the call site's name.
    :param batch_size: the batch size the caller asked for.
    :param settings: run settings holding the governor keys, if any.
    :param kwargs: passed to ``eval`` unchanged.
    :returns: what ``eval`` returned.
    """
    def gpu(size):
        """Run ``eval`` on the model's own device."""
        return model.eval(x, batch_size=size, **kwargs)

    cpu = None
    if _cpu_target(model) is not None:
        def cpu(size):
            """Run ``eval`` with the network moved to the CPU."""
            with _on_cpu(model):
                return model.eval(x, batch_size=size, **kwargs)

    requested = 8 if batch_size is None else batch_size
    device = _model_device(model)
    return _controller(site, device).run(gpu, requested, device=device,
                                         cpu_fn=cpu, config=_config(settings))


def _guarded(gpu_fn: Callable[[], Any],
             cpu_fn: Optional[Callable[[], Any]] = None, *, site: str,
             device: Any = None,
             settings: Optional[Mapping[str, Any]] = None) -> Any:
    """Run one indivisible GPU call, retrying and falling back on out-of-memory.

    :param gpu_fn: the GPU computation.
    :param cpu_fn: the same computation on the CPU.
    :param site: the call site's name.
    :param device: the CUDA device; None for the current one.
    :param settings: run settings holding the governor keys, if any.
    :returns: what ``gpu_fn`` (or ``cpu_fn``) returned.
    """
    fallback = None if cpu_fn is None else (lambda _size: cpu_fn())
    return _controller(site, device).run(lambda _size: gpu_fn(), 1,
                                         device=device, cpu_fn=fallback,
                                         config=_config(settings))


def _split(batch: Any, size: int) -> list:
    """Cut a tensor, array or sequence into consecutive parts of ``size``.

    :param batch: anything sliceable along its first axis.
    :param size: rows per part.
    """
    total = len(batch)
    return [batch[start:start + size] for start in range(0, total, size)]


def _merge(parts: list) -> Any:
    """Join per-part results back into one, in order.

    Tensors are concatenated along the first axis, arrays likewise, lists
    are chained, and tuples and dicts are merged element by element.

    :param parts: the results of consecutive parts.
    """
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    if isinstance(first, tuple):
        return tuple(_merge([part[i] for part in parts])
                     for i in range(len(first)))
    if isinstance(first, dict):
        return {key: _merge([part[key] for part in parts]) for key in first}
    if isinstance(first, list):
        return [value for part in parts for value in part]
    torch = _torch()
    if torch is not None and isinstance(first, torch.Tensor):
        if first.ndim == 0:
            return torch.stack(parts)
        return torch.cat(parts, dim=0)
    import numpy as np

    if isinstance(first, np.ndarray):
        return np.concatenate(parts, axis=0)
    return parts


def _chunked(fn: Callable[[Any], Any], batch: Any, *, site: str,
             cpu_fn: Optional[Callable[[Any], Any]] = None,
             device: Any = None, settings: Optional[Mapping[str, Any]] = None,
             on_retry: Optional[Callable[[], None]] = None) -> Any:
    """Run ``fn`` over a batch in parts that fit the free GPU memory.

    Every row of the batch is processed exactly once in the returned
    result, whatever part size the governor settles on.

    :param fn: maps one part of the batch to its result.
    :param batch: a tensor, array or sequence, split along its first axis.
    :param site: the call site's name.
    :param cpu_fn: maps one part to its result on the CPU.
    :param device: the CUDA device; None for the current one.
    :param settings: run settings holding the governor keys, if any.
    :param on_retry: undoes a failed attempt's side effects.
    :returns: the parts' results merged in order.
    """
    try:
        total = len(batch)
    except TypeError:
        total = 1
    if total < 1:
        return fn(batch)

    def gpu(size):
        """Every part at ``size`` rows on the GPU."""
        if size >= total:
            return fn(batch)
        return _merge([fn(part) for part in _split(batch, size)])

    cpu = None
    if cpu_fn is not None:
        def cpu(size):
            """Every part at ``size`` rows on the CPU."""
            return _merge([cpu_fn(part) for part in _split(batch, size)])

    return _controller(site, device).run(gpu, total, device=device,
                                         cpu_fn=cpu, config=_config(settings),
                                         on_retry=on_retry)


def _reset() -> None:
    """Forget every controller, ceiling and count; for tests and new sessions."""
    with _LOCK:
        _STATE['controllers'].clear()
        _STATE['fractions'].clear()
        _STATE['config'] = dict(_DEFAULTS)
        _COUNTS.clear()
