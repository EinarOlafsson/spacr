"""Process isolation and cooperative stop for assigned mask batches."""

from __future__ import annotations

import copy
import multiprocessing
import os
import queue
import sys
import time
from contextlib import redirect_stderr, redirect_stdout


def _partition_batches(records, devices):
    """Assign each archive once, balancing stored bytes across selected GPUs."""
    devices = list(devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('Choose distinct worker devices')
    assignments = {device: [] for device in devices}
    sizes = {device: 0 for device in devices}
    seen = set()
    for record in sorted(records, key=lambda row: (-row['bytes'], row['path'])):
        path = os.path.abspath(os.fspath(record['path']))
        if path in seen or record['bytes'] < 0:
            raise ValueError('Batch records must have unique paths and nonnegative sizes')
        seen.add(path)
        device = min(devices, key=lambda item: sizes[item])
        assignments[device].append(path)
        sizes[device] += record['bytes']
    for paths in assignments.values():
        paths.sort()
    return assignments


class _SharedCancellation:
    """Expose a process-shared event through the pipeline token protocol."""

    def __init__(self, event):
        self.event = event

    @property
    def cancelled(self):
        """Whether the coordinator or another worker requested a stop."""
        return self.event.is_set()

    @property
    def reason(self):
        """Describe the boundary stop without claiming a failed archive done."""
        return 'Parallel mask generation cancelled'

    def checkpoint(self):
        """Stop before the next unit when cancellation has been requested."""
        if self.cancelled:
            from .cancellation import PipelineCancelled
            raise PipelineCancelled(self.reason)


class _WorkerOutput:
    """Forward child output without inheriting a GUI-thread stream wrapper."""

    encoding = 'utf-8'

    def __init__(self, messages, device):
        self.messages = messages
        self.device = device

    def write(self, text):
        """Queue a text fragment, preserving its original line endings."""
        if text:
            self.messages.put(('log', self.device, str(text)))
        return len(text)

    def flush(self):
        """Queue writes already publish each fragment."""

    def isatty(self):
        """A GUI progress stream is not a terminal."""
        return False


def _mask_worker(src, settings, object_type, device, paths, environment,
                 messages, stop, segmenter=None):
    """Bind a fresh process before loading its model, then report safe units.

    ``segmenter`` is an injectable CPU test seam; ordinary workers load the
    shipped generator only after validating their isolated CUDA/ROCm device.
    """
    from .cancellation import PipelineCancelled, installed_token

    output = _WorkerOutput(messages, device)
    with redirect_stdout(output), redirect_stderr(output):
        try:
            imported = sys.modules.get('torch')
            if imported is not None and imported.cuda.is_initialized():
                raise RuntimeError('Mask worker inherited an initialized CUDA runtime')
            os.environ.update(environment)
            token = _SharedCancellation(stop)
            token.checkpoint()
            if segmenter is None:
                import torch
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError('Mask worker must see exactly its assigned GPU')
                from . import accelerator
                accelerator._CACHED = None
                from .object import generate_cellpose_masks_sam
                segmenter = generate_cellpose_masks_sam
            messages.put(('started', device, os.getpid()))

            def completed(path):
                """Publish completion only after the pipeline's save boundary."""
                messages.put(('batch', device, os.path.abspath(os.fspath(path))))

            with installed_token(token):
                segmenter(src, dict(settings), object_type, batch_paths=paths,
                          on_batch_done=completed, run_qc=False)
            messages.put(('finished', device, ('success', '')))
        except PipelineCancelled:
            messages.put(('finished', device, ('cancelled', '')))
        except BaseException as error:
            stop.set()
            messages.put(('finished', device,
                          ('failed', f'{type(error).__name__}: {error}')))


def _run_mask_workers(src, settings, object_type, assignments, environments, *,
                      on_progress=None, context=None, segmenter=None):
    """Run one persistent model process per nonempty assignment.

    Worker events report individual completed archives. Cancellation waits for
    safe pipeline boundaries; a failed or crashed worker stops its peers.
    Inputs are never deleted here. Shared QC belongs to the caller after all
    workers succeed. ``on_progress`` receives independent state snapshots.
    This layer neither decides resume eligibility nor declares output integrity.

    :returns: final worker states and the unique completed archive paths.
    :raises PipelineCancelled: after a requested cooperative stop completes.
    :raises RuntimeError: if a worker fails, crashes or omits assigned work.
    """
    from .cancellation import PipelineCancelled, cancellation_requested, checkpoint

    checkpoint()
    assignments = {device: [os.path.abspath(os.fspath(path)) for path in paths]
                   for device, paths in assignments.items() if paths}
    owners = {}
    for device, paths in assignments.items():
        if device not in environments:
            raise ValueError(f'No isolated environment for worker {device}')
        for path in paths:
            if path in owners:
                raise ValueError(f'Archive assigned more than once: {path}')
            owners[path] = device
    context = context or multiprocessing.get_context('spawn')
    messages = context.Queue(maxsize=128)
    stop = context.Event()
    processes, dead_since = {}, {}
    completed = set()
    state = {'total_batches': len(owners), 'completed_batches': [], 'workers': {
        device: {'state': 'starting', 'pid': None, 'completed': 0,
                 'total': len(paths), 'error': ''}
        for device, paths in assignments.items()}}
    requested = False

    def publish():
        """Do not let consumers mutate scheduler bookkeeping."""
        state['completed_batches'] = sorted(completed)
        if on_progress is not None:
            on_progress(copy.deepcopy(state))

    def consume(message):
        """Validate archive ownership before accepting a worker's event."""
        kind, device, value = message
        worker = state['workers'][device]
        if kind == 'log':
            print(f'[GPU {device}] {value}', end='')
            return
        if kind == 'started':
            worker.update(state='running', pid=value)
        elif kind == 'batch':
            if owners.get(value) != device or value in completed:
                raise RuntimeError(f'Invalid or repeated batch completion from GPU {device}: {value}')
            completed.add(value)
            worker['completed'] += 1
        elif kind == 'finished':
            status, error = value
            if status == 'success' and worker['completed'] != worker['total']:
                status, error = 'failed', 'Worker exited without completing its assigned archives'
            worker.update(state=status, error=error)
            if status == 'failed':
                stop.set()
        else:
            raise RuntimeError(f'Unknown mask-worker event: {kind}')
        publish()

    try:
        for device, paths in assignments.items():
            process = context.Process(target=_mask_worker, args=(
                src, settings, object_type, device, paths, environments[device],
                messages, stop, segmenter), name=f'spacr-mask-gpu-{device}')
            process.start()
            processes[device] = process
            state['workers'][device]['pid'] = process.pid
        publish()
        pending = set(processes)
        while pending:
            if cancellation_requested():
                requested = True
                stop.set()
            try:
                consume(messages.get(timeout=0.05))
                while True:
                    consume(messages.get_nowait())
            except queue.Empty:
                pass
            for device in tuple(pending):
                process = processes[device]
                if process.is_alive():
                    continue
                process.join()
                worker = state['workers'][device]
                if worker['state'] not in ('success', 'failed', 'cancelled'):
                    since = dead_since.setdefault(device, time.monotonic())
                    if process.exitcode == 0 and time.monotonic() - since < 1.0:
                        continue
                    worker.update(state='failed', error=f'Worker exited with code {process.exitcode} without a final event')
                    stop.set()
                    publish()
                elif process.exitcode != 0:
                    worker.update(state='failed', error=f'Worker exited with code {process.exitcode}')
                    stop.set()
                    publish()
                pending.remove(device)
        failures = [f'GPU {device}: {worker["error"]}'
                    for device, worker in state['workers'].items()
                    if worker['state'] == 'failed']
        if failures:
            raise RuntimeError('; '.join(failures))
        if requested or any(worker['state'] == 'cancelled'
                            for worker in state['workers'].values()):
            raise PipelineCancelled('Parallel mask generation cancelled')
        return state
    finally:
        stop.set()
        deadline = time.monotonic() + 5.0
        while any(process.is_alive() for process in processes.values()) and time.monotonic() < deadline:
            try:
                messages.get(timeout=0.05)
            except queue.Empty:
                pass
        for process in processes.values():
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        messages.close()
        messages.join_thread()
