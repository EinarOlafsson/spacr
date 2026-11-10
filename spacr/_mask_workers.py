"""Process isolation and cooperative stop for assigned mask batches."""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
import pickle
import queue
import sys
import tempfile
import threading
import time
import weakref
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path


def _role_backend(settings, object_type):
    """The backend the mask generator would load for one role's model.

    Follows :func:`spacr.object.generate_cellpose_masks_sam`: a ``cellpose3:``,
    ``cellpose_dino:`` or other backend prefix on the role's model wins, else
    ``segmentation_backend`` decides.

    :returns: ``(prepared, model, backend)`` with defaults filled in a copy.
    """
    from ._segmentation_backends import (_backend_name, _cellpose3_choice, _CELLPOSE3,
                                         _cellpose_dino_choice, _CELLPOSE_DINO,
                                         _prefixed_backend)
    from .settings import _get_object_settings, set_default_settings_preprocess_generate_masks

    if object_type not in ('cell', 'nucleus', 'pathogen'):
        raise ValueError('Unsupported parallel mask object type')
    prepared = set_default_settings_preprocess_generate_masks(copy.deepcopy(settings))
    model = _get_object_settings(object_type, prepared)['model_name']
    if object_type == 'pathogen' and prepared.get('pathogen_model') is not None:
        model = prepared['pathogen_model']
    backend = _backend_name(prepared.get('segmentation_backend', 'cellpose'))
    if _cellpose3_choice(model) is not None:
        backend = _CELLPOSE3
    elif _cellpose_dino_choice(model) is not None:
        backend = _CELLPOSE_DINO
    elif _prefixed_backend(model) is not None:
        backend = _prefixed_backend(model)
    return prepared, model, backend


def _mask_model_contract(settings, object_type):
    """Resolve and sign the model of whichever backend serves this role.

    Cellpose-SAM roles go through :func:`_prepare_mask_model`; every other
    backend through :func:`_resolved_backend_model`. Both return
    ``(prepared, signature, identity)``.
    """
    if _role_backend(settings, object_type)[2] == 'cellpose':
        return _prepare_mask_model(settings, object_type)
    return _resolved_backend_model(settings, object_type)


def _model_label(identity):
    """A short printable name for a resolved model identity."""
    if identity.get('path'):
        return identity['path']
    return f"{identity.get('backend', '?')} {identity.get('model', '?')}"


def _prepare_mask_model(settings, object_type):
    """Resolve Cellpose weights once and sign their bytes with material settings.

    Return a detached canonical settings dict, its settings/model SHA256 and
    inspectable model identity. Stock weights use Cellpose's own cache resolver;
    custom checkpoints stay explicit paths. No network is constructed, no GPU
    allocation occurs, and the caller's settings remain unchanged. A role
    served by another backend is refused here; :func:`_mask_model_contract`
    sends it to :func:`_resolved_backend_model`.
    """
    from .artifacts import material_settings
    from .checkpoint import fingerprint
    from .model_zoo import _resolve_pinned_model_file
    from .utils import _resolve_cellpose_pretrained, cp_models

    prepared, model, backend = _role_backend(settings, object_type)
    if backend != 'cellpose':
        raise ValueError('Parallel model preparation currently requires the Cellpose backend')
    key = ('pathogen_model' if object_type == 'pathogen'
           and prepared.get('pathogen_model') is not None else f'{object_type}_model_name')
    pinned = _resolve_pinned_model_file(prepared, key, kinds=('cellpose',))
    resolved = (str(pinned) if pinned is not None
                else _resolve_cellpose_pretrained(model, object_type=object_type, setting=key))
    path = Path(resolved).expanduser()
    if not path.is_file():
        cache = getattr(cp_models, 'cache_model_path', None)
        legacy_cache = getattr(cp_models, 'cache_CPSAM_model_path', None)
        if callable(cache):
            path = Path(cache(resolved))
        elif resolved == 'cpsam' and callable(legacy_cache):
            path = Path(legacy_cache())
        else:
            raise ValueError(f'Installed Cellpose cannot resolve weights for {resolved!r}')
    path = path.resolve()
    before = path.stat()
    if not path.is_file() or before.st_size == 0:
        raise ValueError(f'Model checkpoint is not a nonempty file: {path}')
    digest = _file_sha256(path)
    after = path.stat()
    if any(getattr(before, key) != getattr(after, key)
           for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')):
        raise ValueError(f'Model changed while preparing workers: {path}')
    if pinned is None:
        prepared[f'{object_type}_model_name'] = str(path)
        if object_type == 'pathogen' and prepared.get('pathogen_model') is not None:
            prepared['pathogen_model'] = str(path)
    material = material_settings(prepared)
    for key in ('mask_parallel', 'mask_gpu_indices'):
        material.pop(key, None)
    identity = {'path': str(path), 'sha256': digest, 'bytes': after.st_size}
    signature = fingerprint({'settings': material, 'model': identity,
                             'object_type': object_type})
    return prepared, signature, identity


_DIGESTS = None


def _file_sha256(path):
    """SHA256 of one model file, read once per run while a worker pool is open.

    Inside a :class:`_MaskWorkerPool` the digest is kept under the file's
    path, device, inode, size and modification time, so the cell and nucleus
    roles of every plate share one read of a 1 GB checkpoint and an edited
    file is read again. Outside a pool every call reads the file.

    :param path: the model file.
    :returns: the hex digest.
    """
    from .model_zoo import sha256_file

    if _DIGESTS is None:
        return sha256_file(path)
    stat = os.stat(path)
    key = (os.fspath(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        _DIGESTS[key] = sha256_file(path)
    return _DIGESTS[key]


_MODEL_CACHE = None


def _cached_model(key, build):
    """Build a mask model, or reuse the one this pooled worker already holds.

    Outside a pooled mask worker every call builds. Inside one the last
    model is kept with its identity: the backend, the weights path with the
    file's size and modification time, and the constructor keywords. The
    next role or plate asking for the same identity reuses it, so a worker
    loads Cellpose-SAM once per run; a different identity drops the held
    model before building, so a worker never holds two.

    :param key: ``(backend, weights, keywords)``.
    :param build: callable without arguments that constructs the model.
    :returns: the model.
    """
    if _MODEL_CACHE is None:
        return build()
    backend, weights, keywords = key
    try:
        stat = Path(str(weights)).expanduser().stat()
        stamp = (str(Path(str(weights)).expanduser().resolve()), stat.st_size, stat.st_mtime_ns)
    except OSError:
        stamp = (str(weights),)
    identity = (backend, stamp, repr(sorted(keywords.items())))
    if _MODEL_CACHE['key'] == identity:
        _MODEL_CACHE['hits'] += 1
        return _MODEL_CACHE['model']
    _MODEL_CACHE.update(key=None, model=None)
    model = build()
    _MODEL_CACHE.update(key=identity, model=model, loads=_MODEL_CACHE['loads'] + 1)
    return model


def _artifact_digest(path):
    """SHA256 and byte count of a model file or of a model folder's files.

    A folder's digest covers every regular file's relative path and bytes in
    sorted order, so renaming, adding or editing a file changes it. Files
    that change while being read are refused.
    """
    path = Path(path).resolve()
    if path.is_file():
        files = [(path.name, path)]
    elif path.is_dir():
        files = sorted((item.relative_to(path).as_posix(), item)
                       for item in path.rglob('*') if item.is_file())
    else:
        raise ValueError(f'Model artifact is missing: {path}')
    if not files:
        raise ValueError(f'Model folder holds no files: {path}')
    digest = hashlib.sha256()
    total = 0
    for name, item in files:
        before = item.stat()
        content = _file_sha256(item)
        after = item.stat()
        if any(getattr(before, key) != getattr(after, key)
               for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')):
            raise ValueError(f'Model changed while preparing workers: {item}')
        digest.update(f'{name}\0{content}\0'.encode())
        total += after.st_size
    if total == 0:
        raise ValueError(f'Model artifact is empty: {path}')
    return digest.hexdigest(), total


def _resolved_backend_model(settings, object_type, *, root=None):
    """Resolve a non-Cellpose-SAM backend's model and sign it with settings.

    Covers Cellpose 3, Cellpose-DINO, the prefixed backends (InstanSeg,
    Omnipose, Spotiflow and the others with a model prefix) and
    the DINOCell and SAMCell backends. A model file or folder is signed by
    its content; a stock model name is signed by the name together with the
    backend environment's install record, which pins the package that ships
    those weights. A missing path or an uninstalled environment is refused.
    No model is built and the caller's settings are unchanged.

    :param settings: run settings.
    :param object_type: ``cell``, ``nucleus`` or ``pathogen``.
    :param root: backends folder; the configured one when ``None``.
    :returns: ``(prepared, signature, identity)`` like
        :func:`_prepare_mask_model`, with ``identity['backend']`` set.
    :raises ValueError: for the Cellpose-SAM backend, an unsupported role,
        or an uninstalled environment.
    :raises ModelZooError: when an explicit model has no exact compatible identity.
    """
    from . import _segmentation_backends as sb
    from .artifacts import material_settings
    from .checkpoint import fingerprint
    from .settings import _get_object_settings, set_default_settings_preprocess_generate_masks

    if object_type not in ('cell', 'nucleus', 'pathogen'):
        raise ValueError('Unsupported parallel mask object type')
    prepared = set_default_settings_preprocess_generate_masks(copy.deepcopy(settings))
    key = f'{object_type}_model_name'
    value = _get_object_settings(object_type, prepared)['model_name']
    if object_type == 'pathogen' and prepared.get('pathogen_model') is not None:
        key, value = 'pathogen_model', prepared['pathogen_model']
    from .model_zoo import _resolve_pinned_model_file

    backend = _role_backend(prepared, object_type)[2]
    pinned = _resolve_pinned_model_file(prepared, key, kinds=(backend,))
    try:
        if pinned is not None:
            model = str(pinned)
        elif sb._cellpose_dino_choice(value) is not None:
            backend, model = sb._CELLPOSE_DINO, sb._cellpose_dino_model(value, setting=key)
        elif sb._cellpose3_choice(value) is None and sb._prefixed_backend(value) is not None:
            backend = sb._prefixed_backend(value)
            model = sb._prefixed_model(backend, value, setting=key)
        elif sb._cellpose3_choice(value) is not None or backend == sb._CELLPOSE3:
            backend, model = sb._CELLPOSE3, sb._cellpose3_model(value, object_type, setting=key)
        elif backend in (sb._DINOCELL, sb._SAMCELL):
            text = str(value or '').strip()
            model = (os.path.abspath(os.path.expanduser(text))
                     if text and os.path.exists(os.path.expanduser(text)) else backend)
        else:
            raise ValueError('Cellpose-SAM models are resolved by _prepare_mask_model')
    except FileNotFoundError as error:
        raise ValueError(str(error)) from error
    identity = {'backend': backend}
    if os.path.isabs(str(model)) and os.path.exists(model):
        digest, size = _artifact_digest(model)
        identity.update(path=str(Path(model).resolve()), sha256=digest, bytes=size)
    else:
        env = os.path.join(sb._backends_root(root), backend)
        record = sb._read_marker(env)
        if record is None:
            raise ValueError(f'The {backend} environment is not installed, so its '
                             f'stock model {model!r} cannot be resolved')
        identity.update(model=str(model), environment=fingerprint(record))
    material = material_settings(prepared)
    for name in ('mask_parallel', 'mask_gpu_indices'):
        material.pop(name, None)
    signature = fingerprint({'settings': material, 'model': identity,
                             'object_type': object_type})
    return prepared, signature, identity


def _mask_output_digest(path):
    """Validate a saved array and fingerprint its complete bytes in bounded memory."""
    from .resume import validate_merged_field

    before = path.stat()
    valid, reason = validate_merged_field(str(path))
    if not valid:
        raise ValueError(f'Incomplete mask output {path}: {reason}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if any(getattr(before, key) != getattr(after, key)
           for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')):
        raise ValueError(f'Mask changed during verification: {path}')
    return digest.hexdigest()


class _MaskBatchLedger:
    """Persist archive provenance and verify outputs before trusting completion.

    ``records`` come from ``io._mask_batch_manifest``. ``signature`` must cover
    the caller's material settings and resolved model content, not merely a
    checkpoint filename. ``excluded_fields`` records deliberate quality/time
    selection exclusions. Environments and credentials are never persisted.
    This ledger owns segmentation only; downstream adjustment/QC must finish
    separately and must not be mistaken for raw segmentation on a later run.
    """

    def __init__(self, src, records, object_type, signature, *, excluded_fields=()):
        """Validate batch ownership and open this role's settings/model ledger.

        ``records`` describe the input archives; ``excluded_fields`` removes
        intentionally skipped outputs from receipt validation. Existing mask
        files must belong to a compatible checkpoint before work can resume.
        """
        from .checkpoint import CheckpointStore

        if object_type not in ('cell', 'nucleus', 'pathogen'):
            raise ValueError('Unsupported parallel mask object type')
        if (not isinstance(signature, str) or len(signature) != 64
                or any(letter not in '0123456789abcdef' for letter in signature)):
            raise ValueError('A settings/model SHA256 signature is required')
        root = Path(src).resolve()
        self.records = {record['path']: copy.deepcopy(record) for record in records}
        if not self.records or len(self.records) != len(records):
            raise ValueError('A nonempty manifest with unique archive paths is required')
        self.output_root = root / f'{object_type}_mask_stack'
        excluded = set(excluded_fields)
        self.expected = {path: [name for name in record['fields'] if name not in excluded]
                         for path, record in self.records.items()}
        path = root / f'.mask-workers-{object_type}.json'
        if not path.exists() and any(self.output_root.glob('*.npy')):
            raise ValueError('Existing masks have no parallel-run provenance; use a separate output folder')
        self.store = CheckpointStore(
            path, workflow='parallel_mask_batches', boundary='archive', resume=True,
            signature={'settings_model': signature, 'object_type': object_type,
                       'records': records, 'expected_outputs': self.expected})
        if set(self.store.completed) - set(self.records):
            raise ValueError('Mask checkpoint contains an unknown archive')
        self.store.update(meta={'manifest': records, 'expected_outputs': self.expected})

    def verified(self):
        """Return durable completed archives whose saved masks still match.

        Missing/truncated files return an archive to pending work. A complete
        but changed file is refused: the generator would otherwise skip it as
        valid, silently accepting an edited or differently generated mask.
        """
        from .resume import validate_merged_field

        complete = set()
        for path, payload in self.store.completed.items():
            if not isinstance(payload, dict) or set(payload) != set(self.expected[path]):
                raise ValueError(f'Invalid output receipt for {path}')
            intact = True
            for name, expected in payload.items():
                output = self.output_root / name
                if not validate_merged_field(str(output))[0]:
                    intact = False
                    continue
                actual = _mask_output_digest(output)
                if actual != expected:
                    raise ValueError(f'Completed mask changed: {self.output_root / name}')
            if intact:
                complete.add(path)
        return complete

    def mark(self, path):
        """Record completion only after every selected output validates and hashes."""
        from .cancellation import installed_token

        payload = {name: _mask_output_digest(self.output_root / name)
                   for name in self.expected[path]}
        with installed_token(None):
            self.store.mark(path, payload)


def _checkpointed_role(src, settings, object_type, assignments, ledger, *, on_progress=None):
    """One pool role that resumes verified archives and records new ones.

    The caller creates ``ledger`` from a fresh input manifest and
    material/model signature first. Archives the ledger verifies are left
    out of the dispatch; each newly completed archive is hashed into the
    ledger before ``on_progress`` sees it. The ledger is marked ``running``
    here; the caller marks it ``segmented`` once the role succeeds. Prepared
    inputs are never deleted.

    :param src: the run's ``masks`` folder.
    :param settings: the role's prepared settings.
    :param object_type: the role.
    :param assignments: ``{worker: [archive paths]}`` covering the manifest once.
    :param ledger: the role's :class:`_MaskBatchLedger`.
    :param on_progress: optional callable receiving each progress snapshot.
    :returns: the role for :meth:`_MaskWorkerPool.run_roles`, with the
        verified ``complete`` set and the ``ledger``.
    :raises ValueError: when the assignments miss or repeat an archive.
    """
    paths = [os.path.abspath(os.fspath(path))
             for assigned in assignments.values() for path in assigned]
    if len(paths) != len(set(paths)) or set(paths) != set(ledger.records):
        raise ValueError('Worker assignments must cover the manifest exactly once')
    complete = ledger.verified()
    pending = {device: [path for path in assigned if os.path.abspath(path) not in complete]
               for device, assigned in assignments.items()}
    ledger.store.update(status='running')

    def progress(state):
        """Persist each newly completed archive before publishing aggregate progress."""
        for path in state['completed_batches']:
            if path not in complete:
                ledger.mark(path)
                complete.add(path)
        state['completed_batches'] = sorted(complete)
        state['total_batches'] = len(ledger.records)
        if on_progress is not None:
            on_progress(state)

    return {'src': src, 'settings': settings, 'object_type': object_type,
            'assignments': pending, 'on_progress': progress,
            'complete': complete, 'ledger': ledger}


def _run_checkpointed_mask_workers(src, settings, object_type, assignments,
                                   environments, ledger, *, on_progress=None,
                                   on_figure=None, context=None, segmenter=None):
    """Run one checkpointed role in a worker pool of its own.

    Resumes verified archives and persists new safe boundaries during
    dispatch, as :func:`_checkpointed_role` describes. Successful workers
    leave the ledger ``segmented``; a failure or stop leaves it ``failed``
    or ``cancelled``. Shared QC is the caller's.

    :returns: the final worker states with every completed archive.
    """
    from .cancellation import PipelineCancelled

    role = _checkpointed_role(src, settings, object_type, assignments, ledger,
                              on_progress=on_progress)
    role['on_figure'] = on_figure
    try:
        with _MaskWorkerPool(context=context, segmenter=segmenter) as pool:
            result = pool.run_roles([role], environments)[0]
        result['completed_batches'] = sorted(role['complete'])
        result['total_batches'] = len(ledger.records)
        ledger.store.update(status='segmented')
        return result
    except BaseException as error:
        ledger.store.update(status='cancelled' if isinstance(error, PipelineCancelled) else 'failed')
        raise


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
        """Adapt the coordinator's shared stop event to cancellation checkpoints."""
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


@contextmanager
def _worker_figures(directory, messages, device):
    """Capture child plots with Agg and queue files rather than large figures.

    Figures are serialized without pyplot manager restoration, so decoding in
    the coordinator cannot create a GUI window on its analysis thread. The
    directory belongs to the parent and survives a child's normal exit.
    """
    if directory is None:
        yield
        return
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot as plt

    from . import figure_sink

    seen = weakref.WeakSet()
    errors = []

    def publish(fig, path=None):
        """Freeze the figure now, before the child can change or close it."""
        if fig in seen and not getattr(fig, '_spacr_live_update', False):
            return
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=f'gpu{device}-',
                                             suffix='.pickle', delete=False) as output:
                manager = fig.canvas.manager
                fig.canvas.manager = None
                try:
                    pickle.dump(fig, output, protocol=pickle.HIGHEST_PROTOCOL)
                finally:
                    fig.canvas.manager = manager
                path = output.name
            seen.add(fig)
            messages.put(('figure', device, path))
        except Exception as error:
            errors.append(error)
            raise

    def show(*args, **kwargs):
        """Publish each shown figure once and release child pyplot ownership."""
        from matplotlib._pylab_helpers import Gcf
        for manager in list(Gcf.get_all_fig_managers()):
            fig = manager.canvas.figure
            publish(fig)
            if not getattr(fig, '_spacr_live_update', False):
                plt.close(fig)

    previous_show = plt.show
    previous_sink = figure_sink.set_sink(publish)
    plt.show = show
    try:
        yield
        if errors:
            raise RuntimeError(f'Worker figure delivery failed: {errors[0]}')
    finally:
        plt.show = previous_show
        figure_sink.set_sink(previous_sink)
        plt.close('all')


def _read_worker_figure(path, directory):
    """Read only this run's private transport files, then release their storage."""
    from matplotlib.figure import Figure

    path = Path(path).resolve()
    if directory is None or path.parent != Path(directory).resolve() or path.suffix != '.pickle':
        raise ValueError('Worker figure is outside the run transport directory')
    try:
        with path.open('rb') as stream:
            figure = pickle.load(stream)
        if not isinstance(figure, Figure):
            raise ValueError('Worker figure payload is not a Matplotlib Figure')
        return figure
    finally:
        path.unlink(missing_ok=True)


class _WorkerOutput:
    """Forward child output without inheriting a GUI-thread stream wrapper."""

    encoding = 'utf-8'

    def __init__(self, messages, device):
        """Route worker text to the bounded message queue with its device identity."""
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


def _serve_mask_jobs(device, environment, messages, stop, jobs, segmenter=None,
                     figure_directory=None, *, cache_models=False):
    """Bind this process to its device once, then run mask jobs in order.

    The first job validates the isolated CUDA/ROCm binding, imports the
    generator and announces ``started``; later jobs reuse that binding, the
    CUDA context and, with ``cache_models``, the loaded model. Each job ends
    with one ``finished`` event. A failed, cancelled or overloaded job ends
    the loop, so a restart gets fresh device memory.

    :param device: the worker name, such as ``0`` or ``0.2``.
    :param environment: the worker's isolated device environment.
    :param messages: queue receiving the worker's events.
    :param stop: process-shared stop event.
    :param jobs: callable returning the next ``(src, settings, object_type,
        paths)`` job, or None to exit.
    :param segmenter: injectable CPU test seam; ordinary workers load the
        shipped generator only after validating their device.
    :param figure_directory: the parent's figure transport folder, or None.
    :param cache_models: keep the loaded model for the next job.
    """
    global _MODEL_CACHE
    from .cancellation import PipelineCancelled, installed_token

    output = _WorkerOutput(messages, device)
    bound = False
    with redirect_stdout(output), redirect_stderr(output):
        while True:
            job = jobs()
            if job is None:
                return
            src, settings, object_type, paths = job
            try:
                if not bound:
                    imported = sys.modules.get('torch')
                    if imported is not None and imported.cuda.is_initialized():
                        raise RuntimeError('Mask worker inherited an initialized CUDA runtime')
                    os.environ.update(environment)
                token = _SharedCancellation(stop)
                token.checkpoint()
                with _worker_figures(figure_directory, messages, device):
                    if not bound:
                        if segmenter is None:
                            import torch
                            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                                raise RuntimeError('Mask worker must see exactly its assigned GPU')
                            from . import accelerator
                            accelerator._CACHED = None
                            from .object import generate_cellpose_masks_sam
                            segmenter = generate_cellpose_masks_sam
                        if cache_models:
                            _MODEL_CACHE = {'key': None, 'model': None, 'hits': 0, 'loads': 0}
                        messages.put(('started', device, os.getpid()))
                        bound = True

                    def completed(path):
                        """Publish completion only after the pipeline's save boundary."""
                        messages.put(('batch', device, os.path.abspath(os.fspath(path))))

                    with installed_token(token):
                        segmenter(src, dict(settings), object_type, batch_paths=paths,
                                  on_batch_done=completed, run_qc=False)
                messages.put(('finished', device, ('success', '')))
            except PipelineCancelled:
                messages.put(('finished', device, ('cancelled', '')))
                return
            except BaseException as error:
                import traceback
                from .runctx import _is_overload_failure

                status = 'overloaded' if _is_overload_failure(error) else 'failed'
                if status == 'failed':
                    stop.set()
                messages.put(('finished', device,
                              (status, ''.join(traceback.format_exception(
                                  type(error), error, error.__traceback__)))))
                return


def _pooled_mask_worker(device, environment, messages, jobs, stop, segmenter=None,
                        figure_directory=None, first_job=None):
    """Entry point of one persistent mask worker process.

    Runs ``first_job``, handed over with the process like a one-shot
    worker's assignment, then serves jobs from ``jobs`` until it receives
    None, keeping its device binding and its loaded model between them; see
    :func:`_serve_mask_jobs`.

    :param device: the worker name.
    :param environment: the worker's isolated device environment.
    :param messages: queue receiving the worker's events.
    :param jobs: this worker's own job queue.
    :param stop: process-shared stop event.
    :param segmenter: injectable CPU test seam.
    :param figure_directory: the parent's figure transport folder, or None.
    :param first_job: the job the process was started for, or None.
    """
    import functools
    import itertools

    queued = itertools.chain([first_job] if first_job is not None else [],
                             iter(jobs.get, None))
    _serve_mask_jobs(device, environment, messages, stop,
                     functools.partial(next, queued, None), segmenter,
                     figure_directory, cache_models=True)


_FINAL_STATES = ('success', 'failed', 'cancelled', 'overloaded')


class _MaskWorkerPool:
    """Mask worker processes kept for every role and plate of one run.

    Each worker is spawned once, binds its device, keeps its CUDA context and
    its loaded model, and takes role-tagged archive jobs in order. Only the
    first spawn of each worker passes the start gate; a worker restarted
    after an overload starts at once. While the pool is open each model file
    is hashed once. Worker events report individual completed archives.
    Cancellation waits for safe pipeline boundaries; an invalid or crashed
    worker stops its peers, and any failure or stop tears every worker down,
    so the next run spawns fresh ones. Explicit overloads receive one serial
    restart, alone on the devices, after every primary worker of that role
    finishes, with already completed archives excluded. Inputs are never
    deleted here. ``on_figure`` receives detached figures on the calling
    thread; by default figures reach the existing GUI sink or are saved
    under mask_worker_plots beside the batch source for headless runs.
    Transport files are cleaned when the pool closes.

    :param context: multiprocessing context; spawn when None.
    :param segmenter: optional CPU stand-in for the mask generator.
    :param gate: optional start gate; deployments ten seconds apart when None.
    """

    def __init__(self, *, context=None, segmenter=None, gate=None):
        """Hold the settings; no process starts before :meth:`run_roles`."""
        self.context = context or multiprocessing.get_context('spawn')
        self.segmenter = segmenter
        self.gate = gate
        self.spawns = 0
        self._environments = None
        self._messages = None
        self._stop = None
        self._figures = None
        self._pump = None
        self._pumping = threading.Event()
        self._inbox = queue.Queue()
        self._processes = {}
        self._jobs = {}
        self._spawned = set()
        self._bound = set()
        self._retired = set()
        self._previous_digests = None

    def __enter__(self):
        """Hash each model file once while the pool is open."""
        global _DIGESTS
        self._previous_digests = _DIGESTS
        if _DIGESTS is None:
            _DIGESTS = {}
        return self

    def __exit__(self, *exc):
        """Stop every worker and forget this run's model digests."""
        global _DIGESTS
        try:
            self.close()
        finally:
            _DIGESTS = self._previous_digests

    def _open(self, environments, plot):
        """Create the event queue, stop event and figure folder once per run.

        A different worker plan, or plots asked for after a run without
        them, closes the old workers first.
        """
        if self._messages is not None and (environments != self._environments
                                           or (plot and self._figures is None)):
            self.close()
        if self._messages is not None:
            return
        self._environments = copy.deepcopy(dict(environments))
        self._messages = self.context.Queue(maxsize=128)
        self._stop = self.context.Event()
        self._figures = (tempfile.TemporaryDirectory(prefix='spacr-mask-figures-')
                         if plot else None)
        self._inbox = queue.Queue()
        self._pumping.set()
        self._pump = threading.Thread(target=self._receive, name='spacr-mask-events',
                                      daemon=True)
        self._pump.start()

    def _receive(self):
        """Move worker events into the inbox, stamped with when they arrived.

        Runs on its own thread, so workers never block on a full queue while
        the caller finalizes a role, and every event keeps its real time.
        """
        while self._pumping.is_set():
            try:
                message = self._messages.get(timeout=0.05)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                return
            self._inbox.put((time.monotonic(), message))

    def _spawn(self, device, job):
        """Start one worker process for ``job``; only its first start waits for the gate."""
        jobs = self.context.Queue()
        process = self.context.Process(target=_pooled_mask_worker, args=(
            device, self._environments[device], self._messages, jobs, self._stop,
            self.segmenter, self._figures.name if self._figures else None, job),
            name=f'spacr-mask-gpu-{device}')
        if device in self._spawned:
            self.gate.start_now(process.start)
        else:
            self.gate.start(process.start)
        self._spawned.add(device)
        self.spawns += 1
        old = self._jobs.pop(device, None)
        if old is not None:
            old.cancel_join_thread()
            old.close()
        self._jobs[device] = jobs
        self._processes[device] = process
        return process

    def _release(self, device):
        """Join and forget a worker that ended, so its next job respawns it."""
        process = self._processes.pop(device)
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join()
        process.close()
        self._bound.discard(device)
        self._retired.discard(device)

    def close(self):
        """Stop every worker and release the pool's queues and figure files."""
        if self._messages is None:
            return
        for jobs in self._jobs.values():
            try:
                jobs.put(None)
            except (OSError, ValueError):
                pass
        deadline = time.monotonic() + 5.0
        while (any(process.is_alive() for process in self._processes.values())
               and time.monotonic() < deadline):
            time.sleep(0.05)
        self._stop.set()
        for process in self._processes.values():
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        self._pumping.clear()
        self._pump.join(timeout=5.0)
        for jobs in self._jobs.values():
            jobs.cancel_join_thread()
            jobs.close()
        self._messages.close()
        self._messages.join_thread()
        if self._figures is not None:
            self._figures.cleanup()
        self._processes, self._jobs = {}, {}
        self._bound, self._retired = set(), set()
        self._messages = self._stop = self._figures = self._pump = None
        self._environments = None

    @staticmethod
    def _role_run(role, environments):
        """Validate one role's assignments and give it fresh bookkeeping."""
        assignments = {device: [os.path.abspath(os.fspath(path)) for path in paths]
                       for device, paths in role['assignments'].items() if paths}
        owners = {}
        for device, paths in assignments.items():
            if device not in environments:
                raise ValueError(f'No isolated environment for worker {device}')
            for path in paths:
                if path in owners:
                    raise ValueError(f'Archive assigned more than once: {path}')
                owners[path] = device
        state = {'total_batches': len(owners), 'completed_batches': [], 'workers': {
            device: {'state': 'starting', 'pid': None, 'completed': 0,
                     'total': len(paths), 'error': '', 'attempts': 0}
            for device, paths in assignments.items()}}
        return dict(role, assignments=assignments, owners=owners, state=state,
                    completed=set(), overloaded=[], begun=False, done=False, done_at=None)

    def run_roles(self, roles, environments, *, on_role_done=None):
        """Run each role's assigned archives, overlapping consecutive roles.

        A worker that finished its share of one role starts its share of
        the next at once, while its peers finish. ``on_role_done`` is called
        on this thread, in role order, once each role has succeeded, while
        later roles keep segmenting. Workers stay up for the next call.

        :param roles: dicts with ``src``, ``settings``, ``object_type`` and
            ``assignments`` (``{worker: [archive paths]}``), and optional
            callables ``on_progress(state)``, ``on_figure(figure)``,
            ``on_start(moment)`` at the role's first dispatch,
            ``on_event(label, moment)`` for worker states and finished
            archives, and ``on_finished(moment)`` when its last archive is
            in. Moments are ``time.monotonic()`` values of arrival.
        :param environments: ``{worker: isolated device environment}``.
        :param on_role_done: optional callable ``(index, state)``.
        :returns: each role's final worker states, in order.
        :raises PipelineCancelled: after a requested cooperative stop completes.
        :raises RuntimeError: if a worker fails, crashes or omits assigned work.
        :raises ValueError: for assignments without an environment or with
            an archive given twice, before any process starts.
        """
        from .cancellation import PipelineCancelled, cancellation_requested, checkpoint
        from .resource_log import _WorkerStartGate

        checkpoint()
        runs = [self._role_run(role, environments) for role in roles]
        if self.gate is None:
            self.gate = _WorkerStartGate()
        try:
            self._open(environments, any(run['settings'].get('plot') for run in runs))
            self._runs = runs
            devices = list(dict.fromkeys(device for run in runs for device in run['assignments']))
            self._waiting = {device: [index for index, run in enumerate(runs)
                                      if device in run['assignments']] for device in devices}
            self._inflight = {device: [] for device in devices}
            self._last = {}
            self._dead_since = {}
            self._retrying = None
            self._requested = False
            reported = 0
            for run in runs:
                self._publish(run)
            while True:
                if cancellation_requested():
                    self._requested = True
                    self._stop.set()
                self._dispatch()
                busy = any(self._inflight.values())
                self._consume(0.05 if busy else 0)
                self._reap()
                for run in runs:
                    if not run['done'] and self._complete(run):
                        run['done'] = True
                        if run.get('on_finished') is not None:
                            run['on_finished'](run['done_at'])
                while (reported < len(runs) and runs[reported]['done']
                       and not self._broken()):
                    if on_role_done is not None:
                        on_role_done(reported, copy.deepcopy(runs[reported]['state']))
                    reported += 1
                if reported == len(runs):
                    break
                if not busy and not any(self._inflight.values()):
                    break
            failures = [f'{prefix}GPU {device}: {worker["error"]}'
                        for run in runs
                        for prefix in [f'{run["object_type"]} ' if len(runs) > 1 else '']
                        for device, worker in run['state']['workers'].items()
                        if worker['state'] == 'failed']
            if failures:
                raise RuntimeError('; '.join(failures))
            if self._requested or any(worker['state'] == 'cancelled' for run in runs
                                      for worker in run['state']['workers'].values()):
                raise PipelineCancelled('Parallel mask generation cancelled')
            if reported < len(runs):
                raise RuntimeError('Mask workers stopped before every archive finished')
            return [run['state'] for run in runs]
        except BaseException:
            if self._stop is not None:
                self._stop.set()
            self.close()
            raise

    def _broken(self):
        """Whether a failure or stop means no further role may succeed."""
        return (self._requested or self._stop.is_set()
                or any(worker['state'] == 'failed' for run in self._runs
                       for worker in run['state']['workers'].values()))

    def _complete(self, run):
        """Whether every worker of ``run`` succeeded and nothing of it is pending."""
        return (not run['overloaded']
                and all(worker['state'] == 'success'
                        for worker in run['state']['workers'].values())
                and not any(queued is run for jobs in self._inflight.values()
                            for queued in jobs))

    def _allowed(self, index):
        """Whether role ``index`` may start: no earlier role awaits a restart."""
        return not any(run['overloaded'] for run in self._runs[:index])

    def _dispatch(self):
        """Hand waiting jobs to idle or running workers, and serial restarts."""
        if self._stop.is_set() or self._requested or self._retrying is not None:
            return
        if not any(self._inflight.values()):
            for index, run in enumerate(self._runs):
                if run['overloaded']:
                    device = run['overloaded'].pop(0)
                    self._retrying = device
                    paths = [path for path in run['assignments'][device]
                             if path not in run['completed']]
                    self._send(device, index, paths, retry=True)
                    return
        if self._broken():
            return
        for device, waiting in self._waiting.items():
            if any(device in run['overloaded'] for run in self._runs):
                continue
            while waiting and self._allowed(waiting[0]):
                index = waiting.pop(0)
                self._send(device, index, self._runs[index]['assignments'][device])

    def _send(self, device, index, paths, retry=False):
        """Queue one role's archives to a worker, spawning it when needed."""
        run = self._runs[index]
        worker = run['state']['workers'][device]
        if retry:
            worker['primary_error'] = worker['error']
        worker['attempts'] += 1
        worker.update(state='starting', error='')
        head = not self._inflight[device]
        if head and not run['begun']:
            run['begun'] = True
            if run.get('on_start') is not None:
                run['on_start'](time.monotonic())
        if device in self._retired:
            self._release(device)
        job = (run['src'], run['settings'], run['object_type'], paths)
        process = self._processes.get(device)
        if process is None:
            process = self._spawn(device, job)
        else:
            self._jobs[device].put(job)
        worker['pid'] = process.pid
        self._inflight[device].append(run)
        if head:
            self._begin(device, time.monotonic())
        self._publish(run)

    def _begin(self, device, moment):
        """Mark the role at the head of a worker's queue as running there."""
        run = self._inflight[device][0]
        if not run['begun']:
            run['begun'] = True
            if run.get('on_start') is not None:
                run['on_start'](moment)
        if device in self._bound:
            self._set_state(run, device, moment, state='running')

    def _set_state(self, run, device, moment, **changes):
        """Update one worker's record and note a changed state."""
        worker = run['state']['workers'][device]
        previous = worker['state']
        worker.update(changes)
        if (worker['state'] != previous and worker['state'] != 'starting'
                and run.get('on_event') is not None):
            run['on_event'](f'worker {device} {worker["state"]}', moment)

    def _publish(self, run):
        """Give the role's progress callback a snapshot it cannot mutate."""
        run['state']['completed_batches'] = sorted(run['completed'])
        if run.get('on_progress') is not None:
            run['on_progress'](copy.deepcopy(run['state']))

    def _consume(self, timeout):
        """Handle every event already in the inbox, waiting up to ``timeout``."""
        try:
            moment, message = (self._inbox.get(timeout=timeout) if timeout
                               else self._inbox.get_nowait())
            self._handle(moment, message)
            while True:
                moment, message = self._inbox.get_nowait()
                self._handle(moment, message)
        except queue.Empty:
            pass

    def _handle(self, moment, message):
        """Validate archive ownership before accepting a worker's event."""
        kind, device, value = message
        jobs = self._inflight.get(device) or []
        run = jobs[0] if jobs else self._last.get(device, self._runs[0])
        if kind == 'log':
            print(f'[GPU {device}] {value}', end='')
            return
        if kind == 'figure':
            self._deliver_figure(run, device, value)
            return
        if kind == 'started':
            self._bound.add(device)
            if jobs:
                self._set_state(run, device, moment, state='running', pid=value)
            return
        if kind == 'batch':
            if not jobs or run['owners'].get(value) != device or value in run['completed']:
                raise RuntimeError(f'Invalid or repeated batch completion from GPU {device}: {value}')
            run['completed'].add(value)
            run['state']['workers'][device]['completed'] += 1
            if run.get('on_event') is not None:
                run['on_event']('archive finished', moment)
        elif kind == 'finished':
            self._finish(device, run, value, moment)
        else:
            raise RuntimeError(f'Unknown mask-worker event: {kind}')
        self._publish(run)

    def _finish(self, device, run, value, moment):
        """Close the worker's current job and decide what its outcome means."""
        status, error = value
        if status not in _FINAL_STATES:
            raise RuntimeError(f'Invalid worker status: {status}')
        jobs = self._inflight[device]
        if not jobs:
            raise RuntimeError(f'GPU {device} finished a job it was never given')
        jobs.pop(0)
        self._last[device] = run
        worker = run['state']['workers'][device]
        exhausted = status == 'overloaded' and worker['attempts'] > 1
        if exhausted:
            status = 'failed'
        if status == 'success' and worker['completed'] != worker['total']:
            status, error = 'failed', 'Worker exited without completing its assigned archives'
        self._set_state(run, device, moment, state=status, error=error)
        run['done_at'] = moment
        if self._retrying == device:
            self._retrying = None
        if status == 'failed' and not exhausted:
            self._stop.set()
        if status != 'success':
            self._retired.add(device)
            self._waiting[device][:0] = [index for queued in jobs
                                         for index, candidate in enumerate(self._runs)
                                         if candidate is queued]
            jobs.clear()
            if status == 'overloaded':
                order = list(run['assignments'])
                run['overloaded'] = sorted(run['overloaded'] + [device], key=order.index)
        elif jobs:
            self._begin(device, moment)

    def _reap(self):
        """Fail roles whose worker died without its final event."""
        for device, process in list(self._processes.items()):
            if process.is_alive():
                continue
            jobs = self._inflight.get(device, [])
            if jobs:
                since = self._dead_since.setdefault(device, time.monotonic())
                if process.exitcode == 0 and time.monotonic() - since < 1.0:
                    continue
                run = jobs[0]
                self._set_state(run, device, time.monotonic(), state='failed',
                                error=f'Worker exited with code {process.exitcode} '
                                      'without a final event')
                jobs.clear()
                self._stop.set()
                self._publish(run)
            elif device not in self._retired and process.exitcode != 0:
                run = self._last.get(device)
                if run is not None:
                    self._set_state(run, device, time.monotonic(), state='failed',
                                    error=f'Worker exited with code {process.exitcode}')
                    self._stop.set()
                    self._publish(run)
            self._dead_since.pop(device, None)
            self._release(device)

    def _deliver_figure(self, run, device, value):
        """Hand a worker's figure to the role's callback, the GUI sink or a file."""
        figure = _read_worker_figure(value, self._figures.name if self._figures else None)
        if run.get('on_figure') is not None:
            run['on_figure'](figure)
            return
        from .figure_sink import publish as publish_figure
        from .figure_sink import sink
        if sink() is not None:
            publish_figure(figure)
        else:
            from .plot import save_figure
            destination = Path(run['src']).parent / 'mask_worker_plots'
            target = save_figure(
                figure, destination / f'{run["object_type"]}_{Path(value).stem}.png',
                fmt='png', dpi=figure.dpi)
            print(f'[GPU {device}] Saved figure: {target}')


def _run_mask_workers(src, settings, object_type, assignments, environments, *,
                      on_progress=None, on_figure=None, context=None, segmenter=None,
                      gate=None):
    """Run one role's assignments in a worker pool of its own.

    The pool is closed on return; see :class:`_MaskWorkerPool` for the
    events, cancellation, overload restarts and figure delivery. Shared QC
    belongs to the caller after all workers succeed, and this layer neither
    decides resume eligibility nor declares output integrity.

    :returns: final worker states and the unique completed archive paths.
    :raises PipelineCancelled: after a requested cooperative stop completes.
    :raises RuntimeError: if a worker fails, crashes or omits assigned work.
    """
    role = {'src': src, 'settings': settings, 'object_type': object_type,
            'assignments': assignments, 'on_progress': on_progress, 'on_figure': on_figure}
    with _MaskWorkerPool(context=context, segmenter=segmenter, gate=gate) as pool:
        return pool.run_roles([role], environments)[0]


_PROGRESS_PREFIX = '[mask GPUs]'


def _compatible_mask_gpus():
    """Return the CUDA/ROCm devices this process may assign to mask workers."""
    from . import accelerator

    return tuple(accelerator._mask_devices())


_CONTROL_GPU_COUNT = []
_CONTROL_GPU_PROBE = None
_CONTROL_GPU_TIMER = None
_CONTROL_GPU_LOCK = threading.Lock()


def _probe_mask_gpu_count(connection):
    """Send the compatible GPU count from an isolated child process.

    :param connection: the child's sending end of the discovery pipe.
    """
    try:
        connection.send(len(_compatible_mask_gpus()))
    except Exception:
        connection.send(0)
    finally:
        connection.close()


def _launch_mask_gpu_probe(process, writer, state):
    """Start the GPU discovery child away from the GUI thread.

    :param process: the discovery child, not yet started.
    :param writer: the child's end of the answer pipe, closed here once
        the child has it.
    :param state: the probe's state; ``launch_failed`` is set when the
        child could not be started. A probe stopped while this launch was
        still running is marked ``cancelled`` and left to this thread, which
        then terminates and reaps the child it started.
    """
    try:
        process.start()
    except Exception:
        state['launch_failed'] = True
    finally:
        writer.close()
    with _CONTROL_GPU_LOCK:
        state['launched'] = True
        cancelled = state['cancelled']
    if cancelled and not state['launch_failed']:
        _reap_cancelled_mask_gpu_probe(process)


def _reap_cancelled_mask_gpu_probe(process):
    """Terminate and reap a discovery child whose probe was stopped mid-launch.

    Runs on the launcher thread, never on the GUI thread, so it may wait.

    :param process: the started discovery child.
    """
    if process.is_alive():
        process.terminate()
        process.join(1.0)
    if process.is_alive():
        process.kill()
        process.join(None)
    process.close()


def _mask_gpu_count_for_controls():
    """Count compatible GPUs once per process for greying the settings controls.

    The discovery child is started on a short-lived launcher thread: a spawn
    start launches a whole interpreter, which took the GUI thread most of a
    second inside a settings build on Windows. Until the launch has returned
    the count reads as pending, exactly like a child that has not answered.
    """
    global _CONTROL_GPU_PROBE
    if _CONTROL_GPU_COUNT:
        _reap_mask_gpu_probe()
        return _CONTROL_GPU_COUNT[0]
    if _CONTROL_GPU_PROBE is None:
        context = multiprocessing.get_context('spawn')
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_probe_mask_gpu_count,
                                  args=(writer,), daemon=True)
        state = {'process': process, 'reader': reader,
                 'started': time.monotonic(), 'stopping': None,
                 'killed': False, 'launch_failed': False,
                 'launched': False, 'cancelled': False}
        launcher = threading.Thread(target=_launch_mask_gpu_probe,
                                    args=(process, writer, state),
                                    name='spacr-gpu-probe-launch', daemon=True)
        state['launcher'] = launcher
        _CONTROL_GPU_PROBE = state
        launcher.start()
        return None
    if _CONTROL_GPU_PROBE['launcher'].is_alive():
        return None
    if _CONTROL_GPU_PROBE['launch_failed']:
        _CONTROL_GPU_PROBE['reader'].close()
        _CONTROL_GPU_PROBE = None
        _CONTROL_GPU_COUNT.append(0)
        return 0
    process = _CONTROL_GPU_PROBE['process']
    reader = _CONTROL_GPU_PROBE['reader']
    started = _CONTROL_GPU_PROBE['started']
    count = None
    try:
        if reader.poll():
            count = max(0, int(reader.recv()))
        elif not process.is_alive() or time.monotonic() - started >= 60:
            count = 0
    except (EOFError, OSError, ValueError, TypeError):
        count = 0
    if count is None:
        return None
    reader.close()
    _CONTROL_GPU_PROBE['reader'] = None
    _CONTROL_GPU_COUNT.append(count)
    _reap_mask_gpu_probe()
    return count


def _reap_mask_gpu_probe():
    """Reap the discovery child without waiting, escalating after one second."""
    global _CONTROL_GPU_PROBE
    if _CONTROL_GPU_PROBE is None:
        return
    state = _CONTROL_GPU_PROBE
    process = state['process']
    if process.is_alive():
        if state['stopping'] is None:
            process.terminate()
            state['stopping'] = time.monotonic()
        elif not state['killed'] and time.monotonic() - state['stopping'] >= 1:
            process.kill()
            state['killed'] = True
    process.join(timeout=0)
    if not process.is_alive():
        process.close()
        _CONTROL_GPU_PROBE = None


def _poll_mask_gpu_probe():
    """Keep discovery timeout and reaping active independently of open panels."""
    if _CONTROL_GPU_PROBE is not None:
        _mask_gpu_count_for_controls()
    if _CONTROL_GPU_PROBE is None and _CONTROL_GPU_TIMER is not None:
        _CONTROL_GPU_TIMER.stop()


def _stop_mask_gpu_probe():
    """Bound discovery cleanup when the application event loop is shutting down."""
    global _CONTROL_GPU_PROBE
    if _CONTROL_GPU_TIMER is not None:
        _CONTROL_GPU_TIMER.stop()
    if _CONTROL_GPU_PROBE is None:
        return
    state = _CONTROL_GPU_PROBE
    process = state['process']
    with _CONTROL_GPU_LOCK:
        pending = not state['launched']
        if pending:
            state['cancelled'] = True
    if state['reader'] is not None:
        state['reader'].close()
        state['reader'] = None
    if not _CONTROL_GPU_COUNT:
        _CONTROL_GPU_COUNT.append(0)
    if pending or state['launch_failed']:
        _CONTROL_GPU_PROBE = None
        return
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.2)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.2)
    if not process.is_alive():
        process.join(timeout=0)
        process.close()
        _CONTROL_GPU_PROBE = None


def _watch_mask_gpu_probe():
    """Own the discovery poller at application scope, including exit cleanup."""
    global _CONTROL_GPU_TIMER
    if _CONTROL_GPU_PROBE is None:
        return
    from PySide6.QtCore import QCoreApplication, QTimer

    app = QCoreApplication.instance()
    if app is None:
        return
    if _CONTROL_GPU_TIMER is None:
        _CONTROL_GPU_TIMER = QTimer(app)
        _CONTROL_GPU_TIMER.setInterval(100)
        _CONTROL_GPU_TIMER.timeout.connect(_poll_mask_gpu_probe)
        app.aboutToQuit.connect(_stop_mask_gpu_probe)
    if not _CONTROL_GPU_TIMER.isActive():
        _CONTROL_GPU_TIMER.start()


def _selected_gpu_indices(value):
    """Parse ``mask_gpu_indices``: blank or ``all`` means every visible GPU.

    :returns: ``None`` for every visible GPU, otherwise distinct ordinals.
    :raises ValueError: for non-integer, negative or repeated entries.
    """
    if value is None or (isinstance(value, str)
                         and value.strip().lower() in ('', 'all', 'none')):
        return None
    if isinstance(value, str):
        tokens = [token for token in value.replace(';', ',').replace(' ', ',').split(',')
                  if token]
    elif isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        tokens = [value]
    indices = []
    for token in tokens:
        try:
            if isinstance(token, bool):
                raise ValueError
            index = int(str(token).strip())
        except ValueError:
            raise ValueError(f'mask_gpu_indices must list GPU numbers, not {token!r}') from None
        if index < 0 or index in indices:
            raise ValueError(f'mask_gpu_indices must list distinct nonnegative GPU numbers: {value!r}')
        indices.append(index)
    return indices


#: Device memory one Cellpose-SAM mask worker is budgeted, for capping
#: ``mask_workers_per_gpu`` before any worker starts: the 5 GB the BBBC021
#: segment processes held at batch size 8 (item 803). The governor sizes
#: each worker's batches to what is free once they run.
_MASK_WORKER_BYTES = 5 << 30


def _workers_per_gpu(settings):
    """Parse ``mask_workers_per_gpu``: how many mask workers share each GPU.

    :returns: an int of at least 1; blank or None is 1.
    :raises ValueError: for a value that is not a positive whole number.
    """
    value = settings.get('mask_workers_per_gpu', 1)
    if value is None or (isinstance(value, str) and not value.strip()):
        return 1
    try:
        if isinstance(value, bool):
            raise ValueError
        count = int(str(value).strip())
    except ValueError:
        raise ValueError(f'mask_workers_per_gpu must be a whole number of 1 or more, '
                         f'not {value!r}') from None
    if count < 1:
        raise ValueError(f'mask_workers_per_gpu must be 1 or more, not {count}')
    return count


def _worker_slots(device, wanted, settings):
    """How many of ``wanted`` workers one GPU can hold under the VRAM budget.

    Each worker is budgeted :data:`_MASK_WORKER_BYTES` against the card's
    memory less ``gpu_memory_margin_gb``, and with ``gpu_memory_fraction``
    set no more workers start than that fraction fits into the card.

    :param device: a record from ``accelerator._mask_devices``.
    :param wanted: the requested workers per GPU.
    :param settings: run settings carrying the governor keys.
    :returns: the number of workers to start on this GPU, at least 1.
    """
    from ._gpu_governor import _config

    config = _config(settings)
    slots = wanted
    fraction = config['gpu_memory_fraction']
    if fraction:
        slots = min(slots, max(1, int(1.0 / float(fraction) + 1e-9)))
    total = int(device.get('memory_bytes') or 0)
    if total > 0:
        usable = total - float(config['gpu_memory_margin_gb']) * (1 << 30)
        slots = min(slots, max(1, int(usable // _MASK_WORKER_BYTES)))
    if slots < wanted:
        print(f"mask_workers_per_gpu: GPU {device['index']} ({device.get('name', '?')}) "
              f"has room for {slots} of {wanted} workers under the GPU memory budget; "
              f'starting {slots}.')
    return slots


def _parallel_mask_plan(settings, devices=None):
    """Choose worker GPUs for this run, or return ``None`` for the ordinary path.

    ``mask_parallel`` off and ``mask_workers_per_gpu`` 1 return ``None``
    before any device query. A blank ``mask_gpu_indices`` uses every GPU this
    process was allocated, so a cluster job uses the devices its scheduler
    exposed; fewer than two falls back to the ordinary single-device path
    with a printed note. Explicit indices must name at least two visible
    devices and are never widened.

    ``mask_workers_per_gpu`` above 1 starts that many workers on each chosen
    GPU, capped by :func:`_worker_slots`, so one worker's preparation and
    CPU post-processing overlap another's Cellpose call. With
    ``mask_parallel`` off it uses the first GPU alone, so a single-GPU
    workstation runs it too. Workers sharing a GPU are named ``<gpu>.<n>``.

    :param settings: run settings carrying ``mask_parallel``,
        ``mask_gpu_indices`` and ``mask_workers_per_gpu``.
    :param devices: optional device records in ``accelerator._mask_devices`` form.
    :returns: ``{'devices', 'environments', 'names', 'gpus', 'uuids'}`` or
        ``None``. ``devices`` lists the worker names, ``gpus`` maps each to
        its GPU and ``uuids`` maps each GPU whose utilisation is sampled to
        its UUID (``''`` to look it up by index). Stand-in ``devices``
        without a ``uuid`` are not sampled.
    :raises ValueError: for timelapse/t_stack runs or an invalid selection.
    """
    from .resume import _TRUTHY

    enabled = settings.get('mask_parallel', False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in _TRUTHY
    per_gpu = _workers_per_gpu(settings)
    if not enabled and per_gpu == 1:
        return None
    if settings.get('timelapse') or settings.get('t_stack'):
        raise ValueError('Parallel GPU mask batches do not support timelapse or t_stack runs; '
                         'turn mask_parallel off.')
    found = list(_compatible_mask_gpus() if devices is None else devices)
    visible = {device['index']: device for device in found}
    if enabled:
        wanted = _selected_gpu_indices(settings.get('mask_gpu_indices'))
        if wanted is None:
            if len(found) < 2 and per_gpu == 1:
                print(f'mask_parallel: {len(found)} compatible GPU visible; '
                      'using the ordinary single-device mask path.')
                return None
            wanted = sorted(visible)
        else:
            missing = [index for index in wanted if index not in visible]
            if missing or len(wanted) < 2:
                raise ValueError(f'mask_gpu_indices {wanted} must name two or more of the '
                                 f'visible GPUs {sorted(visible)}.')
    else:
        wanted = sorted(visible)[:1]
    slots = {index: _worker_slots(visible[index], per_gpu, settings) for index in wanted}
    if not wanted or (not enabled and sum(slots.values()) < 2):
        print(f'mask_workers_per_gpu: {len(found)} compatible GPU visible and room for '
              f'{sum(slots.values())} worker; using the ordinary single-device mask path.')
        return None
    from . import accelerator

    backend = found[0]['backend']
    environments = {index: accelerator._mask_worker_environment(
        index, backend=backend, count=len(found)) for index in wanted}
    shared = any(count > 1 for count in slots.values())
    gpus = {}
    for index in wanted:
        for slot in range(1, slots[index] + 1):
            gpus[f'{index}.{slot}' if shared else index] = index
    uuids = {}
    for index in wanted:
        uuid = visible[index].get('uuid') or (_device_uuid(index) if devices is None else None)
        if uuid is not None:
            uuids[index] = uuid
    return {'devices': list(gpus),
            'environments': {worker: environments[index] for worker, index in gpus.items()},
            'names': {worker: visible[index]['name'] for worker, index in gpus.items()},
            'gpus': gpus, 'uuids': uuids}


def _device_uuid(index):
    """The NVML UUID (``GPU-...``) of one logical CUDA device, or ``''``.

    Read from the PyTorch already imported for device discovery; NVML
    numbers GPUs in bus order, which need not be CUDA's order.
    """
    torch = sys.modules.get('torch')
    try:
        uuid = str(torch.cuda.get_device_properties(index).uuid)
    except Exception:                                       # noqa: BLE001
        return ''
    return uuid if uuid.startswith('GPU-') else f'GPU-{uuid}'


def _failed_archives(state):
    """Archives in flight on failed workers; a failure costs at most one each."""
    return sum(1 for worker in state.get('workers', {}).values()
               if worker['state'] == 'failed' and worker['completed'] < worker['total'])


def _progress_line(object_type, state):
    """Format one stdout line carrying overall and per-GPU archive counts.

    ``spacr.qt.bridge`` reads the ``Progress: done/total`` part for the run bar
    and :func:`spacr.qt.bridge._mask_gpu_progress` reads the per-GPU part.
    """
    parts = ''.join(
        f' | GPU {device} {worker["state"]} {worker["completed"]}/{worker["total"]}'
        for device, worker in sorted(state['workers'].items(), key=lambda item: str(item[0])))
    return (f'{_PROGRESS_PREFIX} {object_type} Progress: {len(state["completed_batches"])}/'
            f'{state["total_batches"]} archives, failed {_failed_archives(state)}{parts}')


_BUSY_PERCENT = 80
_SAMPLE_SECONDS = 1.0
_UTILISATION_FILE = 'gpu_utilisation.json'


def _nvml_reader(gpus):
    """A function reading ``{gpu: (busy percent, used bytes)}`` through NVML.

    :param gpus: ``{gpu index: UUID}``; a GPU with no UUID is looked up by
        index, which matches CUDA's numbering when CUDA_DEVICE_ORDER is
        PCI_BUS_ID.
    :returns: the reader, or None when NVML (``nvidia-ml-py``) or a GPU in
        ``gpus`` cannot be opened.
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', DeprecationWarning)
            import pynvml
        pynvml.nvmlInit()
        handles = {}
        for index, uuid in gpus.items():
            handles[index] = (pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid
                              else pynvml.nvmlDeviceGetHandleByIndex(int(index)))
    except Exception:                                       # noqa: BLE001
        return None
    import functools

    return functools.partial(_nvml_sample, pynvml, handles)


def _nvml_sample(pynvml, handles):
    """One NVML sample of every GPU: ``{gpu: (busy percent, used bytes)}``."""
    return {index: (float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
                    int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
            for index, handle in handles.items()}


def _busy_summary(busy):
    """Mean busy percentage and share of samples at or above the busy mark.

    :param busy: busy percentages.
    :returns: ``(mean, share)``, both None without samples.
    """
    if not busy:
        return None, None
    return (round(sum(busy) / len(busy), 1),
            round(sum(1 for value in busy if value >= _BUSY_PERCENT) / len(busy), 3))


def _startup_split(values, events):
    """Split one GPU's samples at the role's first finished archive.

    Everything up to that event is start-up (spawning, binding the device,
    loading the model, the first archive); everything after it is steady
    segmentation.

    :param values: ``(busy percent, used bytes, seconds since start)`` samples.
    :param events: ``[seconds since start, label]`` notes.
    :returns: ``startup_seconds``, ``startup_mean_busy_percent``,
        ``steady_mean_busy_percent``, ``steady_busy_share`` and
        ``steady_samples``; the first three are None before any archive finished.
    """
    finished = [moment for moment, label in events if label == 'archive finished']
    if not finished:
        return {'startup_seconds': None, 'startup_mean_busy_percent': None,
                'steady_mean_busy_percent': None, 'steady_busy_share': None,
                'steady_samples': 0}
    first = min(finished)
    startup, _ = _busy_summary([value[0] for value in values if value[2] <= first])
    steady = [value[0] for value in values if value[2] > first]
    mean, share = _busy_summary(steady)
    return {'startup_seconds': first, 'startup_mean_busy_percent': startup,
            'steady_mean_busy_percent': mean, 'steady_busy_share': share,
            'steady_samples': len(steady)}


class _GpuUtilisation:
    """Sample how busy each GPU is while one role's workers run.

    A thread reads every GPU each :data:`_SAMPLE_SECONDS`; :meth:`stop`
    returns, per GPU, the mean busy percentage, the share of samples at or
    above :data:`_BUSY_PERCENT`, the peak memory in use, the wall-clock
    start and the start-up/steady split at the first finished archive.

    :param gpus: ``{gpu index: UUID}`` from the worker plan.
    :param read: optional reader in :func:`_nvml_reader` form, for tests.
    :param interval: seconds between samples; :data:`_SAMPLE_SECONDS` when
        None.
    """

    def __init__(self, gpus, read=None, interval=None):
        """Open the reader; nothing is sampled until :meth:`start`."""
        self.read = read if read is not None else _nvml_reader(gpus)
        self.interval = _SAMPLE_SECONDS if interval is None else interval
        self.samples = {index: [] for index in gpus}
        self.events = []
        self.started_at = None
        self._stop = threading.Event()
        self._thread = None
        self._started = None

    def _elapsed(self, moment=None):
        """Seconds from :meth:`start` to ``moment`` (now when None), to one decimal; 0 before it."""
        if self._started is None:
            return 0.0
        return round((time.monotonic() if moment is None else moment) - self._started, 1)

    def note(self, label, at=None):
        """Record that ``label`` happened, for the utilisation file.

        :param label: a short description, such as ``worker 0.2 started``.
        :param at: the ``time.monotonic()`` moment it happened; now when None.
        """
        self.events.append([self._elapsed(at), str(label)])

    def _sample(self):
        """Append one reading of every GPU; a failed reading is skipped."""
        try:
            values = self.read()
        except Exception:                                   # noqa: BLE001
            return
        moment = self._elapsed()
        for index, value in values.items():
            if index in self.samples:
                self.samples[index].append((value[0], value[1], moment))

    def _loop(self):
        """Sample every interval until stopped."""
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self, at=None):
        """Start sampling in a daemon thread, when a reader is available.

        :param at: the ``time.monotonic()`` moment the role began, when
            that was before this call; now when None.
        """
        now = time.monotonic()
        self._started = now if at is None else min(at, now)
        self.started_at = time.time() - (now - self._started)
        if self.read is None:
            return self
        self._thread = threading.Thread(target=self._loop, name='spacr-gpu-utilisation',
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, until=None):
        """Stop sampling and summarise.

        :param until: the ``time.monotonic()`` moment the role ended, when
            that was before this call; samples after it are left out.
        :returns: ``{gpu index: {'samples', 'seconds', 'mean_busy_percent',
            'busy_share', 'peak_memory_gb', 'trace', 'events', 'started_at',
            'started_at_iso', 'startup_seconds', 'startup_mean_busy_percent',
            'steady_mean_busy_percent', 'steady_busy_share',
            'steady_samples'}}``; empty when no reader was available.
            ``trace`` lists every sample as ``[seconds since start, busy
            percent, memory GB]`` and ``events`` the :meth:`note` calls as
            ``[seconds since start, label]``. ``started_at`` is the start in
            epoch seconds and ``started_at_iso`` the same moment in local
            time with its offset, so the trace can be laid against file
            times. The start-up is everything up to the first finished
            archive and the steady part everything after it.
        """
        import datetime

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.read is None:
            return {}
        end = time.monotonic() if until is None else until
        seconds = end - (self._started if self._started is not None else end)
        cutoff = self._elapsed(end)
        started_at = round(self.started_at, 3) if self.started_at is not None else None
        iso = (datetime.datetime.fromtimestamp(self.started_at).astimezone().isoformat(
            timespec='seconds') if self.started_at is not None else None)
        summary = {}
        for index, samples in self.samples.items():
            values = [value for value in samples if value[2] <= cutoff]
            mean, share = _busy_summary([value[0] for value in values])
            summary[index] = {
                'samples': len(values),
                'seconds': round(seconds, 1),
                'mean_busy_percent': mean,
                'busy_share': share,
                'peak_memory_gb': (round(max(value[1] for value in values) / (1 << 30), 2)
                                   if values else None),
                'trace': [[value[2], round(float(value[0]), 1),
                           round(value[1] / (1 << 30), 2)] for value in values],
                'events': [list(event) for event in self.events],
                'started_at': started_at,
                'started_at_iso': iso,
                **_startup_split(values, self.events),
            }
        return summary


def _report_gpu_utilisation(mask_src, object_type, plan, summary):
    """Print one line per GPU and add the role to the plate's utilisation file.

    The file is ``gpu_utilisation.json`` in the plate folder, the parent of
    ``mask_src``, keyed by role, so each plate keeps its own record.

    :param mask_src: the plate's ``masks`` folder.
    :param object_type: the role that was segmented.
    :param plan: the worker plan.
    :param summary: :meth:`_GpuUtilisation.stop`'s result.
    """
    from collections import Counter

    plate_dir = os.path.dirname(os.path.abspath(mask_src))
    plate = os.path.basename(plate_dir)
    if not summary:
        print(f'[gpu] plate {plate} {object_type} masks: GPU utilisation not '
              'reported; NVML (nvidia-ml-py) cannot read these GPUs.')
        return
    workers = Counter((plan.get('gpus') or {}).values())
    for index, row in sorted(summary.items()):
        if not row['samples']:
            continue
        noun = 'worker' if workers[index] == 1 else 'workers'
        steady = ''
        if row.get('steady_mean_busy_percent') is not None:
            steady = (f'; busy {row["steady_mean_busy_percent"]:.0f}% after the first '
                      f'archive finished at {row["startup_seconds"]:.0f} s')
        print(f'[gpu] plate {plate} {object_type} masks: GPU {index} with '
              f'{workers[index]} {noun} busy {row["mean_busy_percent"]:.0f}% on average, '
              f'at least {_BUSY_PERCENT}% in {100 * row["busy_share"]:.0f}% of '
              f'{row["samples"]} samples over {row["seconds"]:.0f} s; '
              f'memory up to {row["peak_memory_gb"]:.1f} GB{steady}')
    path = os.path.join(plate_dir, _UTILISATION_FILE)
    try:
        with open(path, encoding='utf-8') as handle:
            record = json.load(handle)
        if not isinstance(record, dict):
            record = {}
    except (OSError, ValueError):
        record = {}
    record[object_type] = {str(index): dict(row, workers=workers[index])
                           for index, row in summary.items()}
    descriptor, temporary = tempfile.mkstemp(prefix='.gpu_utilisation_', suffix='.json',
                                             dir=plate_dir)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    except OSError:
        try:
            os.remove(temporary)
        except OSError:
            pass


def _ordinary_mask_gpu():
    """The CUDA device the ordinary single-process mask path runs on, or None.

    The device spaCR's accelerator resolved, ``cuda:0`` unless another index
    was chosen; None on the CPU and on other accelerators, which NVML cannot
    read.
    """
    from . import accelerator

    if not accelerator.is_cuda():
        return None
    _, _, index = accelerator.device_string().partition(':')
    return int(index) if index.isdigit() else 0


_ARCHIVE_SAMPLERS = []


def _announce_archive(path):
    """Tell the ordinary path's GPU sampler that one archive's masks are saved.

    The mask generator calls this after each archive, next to its
    ``on_batch_done`` callback; with no sampler listening it does nothing.

    :param path: the finished archive.
    """
    for sampler in tuple(_ARCHIVE_SAMPLERS):
        sampler.note('archive finished')


@contextmanager
def _sampled_ordinary_masks(mask_src, object_type):
    """Sample the GPU while the ordinary single-process path segments one role.

    The same sampler and report as the worker path, so a run with
    ``mask_workers_per_gpu`` 1 writes the ``[gpu]`` line and
    ``gpu_utilisation.json`` its multi-worker comparison is measured against,
    with the same start time and start-up/steady split at the first
    finished archive. Nothing is sampled or printed without a CUDA device.

    :param mask_src: the plate's ``masks`` folder.
    :param object_type: the role being segmented.
    """
    index = _ordinary_mask_gpu()
    if index is None:
        yield
        return
    utilisation = _GpuUtilisation({index: _device_uuid(index)}).start()
    _ARCHIVE_SAMPLERS.append(utilisation)
    try:
        yield
    finally:
        _ARCHIVE_SAMPLERS.remove(utilisation)
        _report_gpu_utilisation(mask_src, object_type, {'gpus': {index: index}},
                                utilisation.stop())


class _RoleMonitor:
    """Sample, print and time one role while the worker pool segments it.

    :param mask_src: the plate's ``masks`` folder.
    :param object_type: the role.
    :param plan: the worker plan.
    """

    def __init__(self, mask_src, object_type, plan):
        """Nothing is sampled before the role's first dispatch."""
        self.mask_src = mask_src
        self.object_type = object_type
        self.plan = plan
        self.utilisation = None
        self.began = None
        self.ended = None
        self.last = {}

    def start(self, moment):
        """Start sampling the plan's GPUs from the role's first dispatch."""
        self.began = moment
        gpus = self.plan.get('uuids') or {}
        if gpus:
            self.utilisation = _GpuUtilisation(gpus).start(at=moment)

    def event(self, label, moment):
        """Note a worker state or a finished archive at the moment it arrived."""
        if self.utilisation is not None:
            self.utilisation.note(label, at=moment)

    def progress(self, state):
        """Keep the latest snapshot for the summary and print it for the GUI."""
        self.last = state
        print(_progress_line(self.object_type, state))

    def finish(self, moment=None):
        """Stop sampling at the role's last event and report the plate's record once."""
        if self.ended is None and self.began is not None:
            self.ended = time.monotonic() if moment is None else moment
        if self.utilisation is not None:
            utilisation, self.utilisation = self.utilisation, None
            _report_gpu_utilisation(self.mask_src, self.object_type, self.plan,
                                    utilisation.stop(until=self.ended))

    def seconds(self):
        """Seconds from the role's first dispatch to its last archive; 0 if it never ran."""
        if self.began is None or self.ended is None:
            return 0.0
        return self.ended - self.began


def _settings_after_role(settings, mask_src):
    """Apply to ``settings`` what finalizing one parallel role does to them.

    The next role is signed with the result, so overlapped roles carry the
    same settings, signature and resume record as roles run one by one.

    :param settings: the run settings, changed in place.
    :param mask_src: the run's ``masks`` folder.
    :returns: ``settings``.
    """
    from .object import _fill_cellpose_channel_positions
    from .settings import set_default_settings_preprocess_generate_masks

    settings['src'] = mask_src
    set_default_settings_preprocess_generate_masks(settings)
    _fill_cellpose_channel_positions(settings)
    return settings


def _roles_may_overlap(settings):
    """Whether one role's QC leaves the next role's settings unchanged.

    ``seg_qc`` set to ``flag`` records each role's flags in the settings
    the next role is signed with, so those roles run one after another.
    """
    try:
        from .seg_qc import qc_mode
    except Exception:                                       # noqa: BLE001
        return True
    return qc_mode(settings) != 'flag'


def _generate_roles_in_parallel(mask_src, settings, roles, plan, *, pool=None,
                                context=None, segmenter=None):
    """Segment several roles' prepared archives with persistent worker processes.

    Each role's archives are inventoried, partitioned by bytes and signed
    with the role's resolved model before dispatch, exactly as when the
    role ran alone. Consecutive roles overlap: a worker that finished its
    share of one role starts its share of the next while its peers finish,
    and this process finalizes a finished role (its shared QC) while the
    next one segments. With ``seg_qc`` set to ``flag`` the roles run one
    after another, in the same worker processes. Verified archives from an
    interrupted run are skipped, and ``settings`` gains the same entries the
    ordinary generator records. Failures and cancellation print
    completed/failed/unfinished counts and keep every prepared input. Each
    role's GPU utilisation is sampled from its first dispatch to its last
    archive and reported per plate by :func:`_report_gpu_utilisation`.

    :param mask_src: the run's ``masks`` folder of prepared NPZ archives.
    :param settings: run settings; updated as the ordinary generator does.
    :param roles: ``cell``, ``nucleus`` and ``pathogen`` names, in run order.
    :param plan: result of :func:`_parallel_mask_plan`.
    :param pool: the run's :class:`_MaskWorkerPool`; a pool of its own,
        closed on return, when None.
    :param context: optional multiprocessing context for a pool of its own.
    :param segmenter: optional CPU stand-in for the mask generator, which
        serves Cellpose-SAM and every backend with a model contract.
    :returns: ``{role: seconds}`` from each role's first dispatch to its
        last finished archive.
    """
    from contextlib import ExitStack

    seconds = {}
    with ExitStack() as stack:
        if pool is None:
            pool = stack.enter_context(_MaskWorkerPool(context=context, segmenter=segmenter))
        groups = ([list(roles)] if _roles_may_overlap(settings)
                  else [[role] for role in roles])
        for group in groups:
            seconds.update(_segment_role_group(mask_src, settings, group, plan, pool))
    return seconds


def _segment_role_group(mask_src, settings, group, plan, pool):
    """Sign, dispatch and finalize one group of overlapping roles.

    :param mask_src: the run's ``masks`` folder.
    :param settings: run settings; finalizing each role updates them.
    :param group: role names in run order.
    :param plan: the worker plan.
    :param pool: the run's :class:`_MaskWorkerPool`.
    :returns: ``{role: seconds}`` for the group.
    """
    from .cancellation import PipelineCancelled
    from .io import _mask_batch_manifest
    from .object import _run_seg_qc

    records = _mask_batch_manifest(mask_src)
    names = ', '.join(f'GPU {device} ({plan["names"].get(device, "?")})'
                      for device in plan['devices'])
    signed = settings
    entries = []
    for position, object_type in enumerate(group):
        if position:
            signed = _settings_after_role(copy.deepcopy(signed), mask_src)
        prepared, signature, identity = _mask_model_contract(signed, object_type)
        ledger = _MaskBatchLedger(mask_src, records, object_type, signature,
                                  excluded_fields=settings.get('image_qc_excluded_fields') or ())
        print(f'Parallel {object_type} masks: {len(records)} archives on {names}; '
              f'model {_model_label(identity)}')
        monitor = _RoleMonitor(mask_src, object_type, plan)
        role = _checkpointed_role(mask_src, prepared, object_type,
                                  _partition_batches(records, plan['devices']), ledger,
                                  on_progress=monitor.progress)
        role.update(on_start=monitor.start, on_event=monitor.event,
                    on_finished=monitor.finish)
        entries.append((object_type, ledger, monitor, role))
    segmented = set()

    def finalize(index, state):
        """Record the role's masks as segmented, then run its shared QC once."""
        object_type, ledger, _, _ = entries[index]
        ledger.store.update(status='segmented')
        segmented.add(index)
        _settings_after_role(settings, mask_src)
        _run_seg_qc(mask_src, settings, object_type)
        ledger.store.update(status='finalized')

    try:
        pool.run_roles([entry[3] for entry in entries], plan['environments'],
                       on_role_done=finalize)
    except BaseException as error:
        for index, (object_type, ledger, monitor, _) in enumerate(entries):
            if index in segmented:
                continue
            ledger.store.update(status='cancelled' if isinstance(error, PipelineCancelled)
                                else 'failed')
            done = len(ledger.verified())
            failed = _failed_archives(monitor.last)
            print(f'Parallel {object_type} masks stopped: {done} completed, {failed} failed, '
                  f'{len(records) - done - failed} unfinished of {len(records)} archives. '
                  'Completed masks and every prepared input are kept; '
                  'rerun with the same settings to resume.')
        raise
    finally:
        for _, _, monitor, _ in entries:
            monitor.finish()
    return {object_type: monitor.seconds() for object_type, _, monitor, _ in entries}


def _mask_digests(folder):
    """SHA256 of every visible mask array in ``folder``, keyed by filename."""
    from .io import _listdir_visible

    if folder is None or not os.path.isdir(folder):
        return {}
    return {name: _mask_output_digest(Path(folder) / name)
            for name in sorted(_listdir_visible(folder)) if name.endswith('.npy')}


def _finalize_adjusted_cells(mask_src, organelle_folder=None, n_jobs=None):
    """Write adjusted cells beside the raw worker masks, with provenance.

    Raw cell, nucleus and pathogen masks stay byte-identical, so their archive
    receipts remain valid. ``.adjusted-cell-mask-provenance.json`` binds the
    adjusted arrays to the raw mask digests and thresholds; a rerun reuses
    them only when both still match and rebuilds every field otherwise.

    :param mask_src: the run's ``masks`` folder.
    :param organelle_folder: optional organelle mask folder for the merger.
    :param n_jobs: adjustment worker count.
    :returns: the adjusted cell mask folder.
    """
    from . import utils
    from .checkpoint import _atomic_json, fingerprint

    root = Path(mask_src)
    output = root / 'adjusted_cell_mask_stack'
    record = root / '.adjusted-cell-mask-provenance.json'
    folders = {'cell': root / 'cell_mask_stack', 'nucleus': root / 'nucleus_mask_stack',
               'pathogen': root / 'pathogen_mask_stack'}
    if organelle_folder is not None:
        folders['organelle'] = Path(organelle_folder)
    inputs = {'raw': {role: _mask_digests(folder) for role, folder in folders.items()},
              'overlap_threshold': 5, 'perimeter_threshold': 30}
    identity = fingerprint(inputs)
    if record.is_file():
        try:
            saved = json.loads(record.read_text(encoding='utf-8'))
            if saved.get('inputs') == identity and saved.get('outputs') == _mask_digests(output):
                print(f'Adjusted cell masks verified against their raw masks: {output}')
                return str(output)
        except (OSError, ValueError):
            pass
        record.unlink()
    utils.adjust_cell_masks(str(folders['pathogen']), str(folders['cell']),
                            str(folders['nucleus']), organelle_folder,
                            overlap_threshold=5, perimeter_threshold=30,
                            n_jobs=n_jobs, output_folder=str(output))
    _atomic_json(record, {'inputs': identity, 'raw_masks': inputs,
                          'outputs': _mask_digests(output)})
    return str(output)


_DATASET_COUNTER = None
_DATASET_LOCK = None


def _dataset_tar_init(counter, lock):
    """Give a dataset tar worker the shared progress counter and its lock.

    The tar and Cellpose dataset pools run the workers below from this
    module because a spawned worker imports the module of the function it
    runs: from here that is the standard library, tifffile, OpenCV and NumPy,
    not :mod:`spacr.io` and :mod:`spacr.utils` with torch.

    :param counter: shared ``multiprocessing.Value`` counting added images.
    :param lock: lock guarding ``counter``.
    """
    global _DATASET_COUNTER, _DATASET_LOCK
    _DATASET_COUNTER, _DATASET_LOCK = counter, lock


def _dataset_tar_worker(paths_chunk, tar_path, total_images):
    """Add one chunk of images to its own tar, as :func:`spacr.utils.add_images_to_tar` does.

    Every tenth image added prints the same progress line
    :func:`spacr.utils.print_progress` prints without timings.

    :param paths_chunk: the image paths for this tar.
    :param tar_path: the tar to write.
    :param total_images: images across every chunk, for the progress line.
    """
    import tarfile

    with tarfile.open(tar_path, 'w') as tar:
        for img_path in paths_chunk:
            arcname = os.path.basename(img_path)
            try:
                tar.add(img_path, arcname=arcname)
                with _DATASET_LOCK:
                    _DATASET_COUNTER.value += 1
                    if _DATASET_COUNTER.value % 10 == 0:
                        print(f'Progress: {_DATASET_COUNTER.value}/{int(total_images)}, '
                              f'operation_type: generating .tar dataset, None')
            except FileNotFoundError:
                print(f"File not found: {img_path}")


def _augmented_plane(image, method):
    """Rotate or flip an image by the named augmentation, as :func:`spacr.io.apply_augmentation`.

    :param image: the image array.
    :param method: ``rotate90``, ``rotate180``, ``rotate270``, ``flip_h`` or
        ``flip_v``; anything else returns ``image`` unchanged.
    :returns: the augmented image.
    """
    import cv2

    if method == 'rotate90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif method == 'rotate180':
        return cv2.rotate(image, cv2.ROTATE_180)
    elif method == 'rotate270':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif method == 'flip_h':
        return cv2.flip(image, 1)
    elif method == 'flip_v':
        return cv2.flip(image, 0)
    return image


def _cellpose_pair_worker(entry):
    """Copy one image and mask pair into a Cellpose dataset, as :func:`spacr.io.process_instruction`.

    :param entry: ``src_img``, ``src_msk``, ``dst_img``, ``dst_msk`` and
        ``augment`` (an augmentation name, or falsy for none).
    :returns: ``1``, for progress counting.
    """
    import tifffile
    from .tiff_io import write_tiff

    img = tifffile.imread(entry["src_img"])
    msk = tifffile.imread(entry["src_msk"])
    if entry["augment"]:
        img = _augmented_plane(img, entry["augment"])
        msk = _augmented_plane(msk, entry["augment"])
    write_tiff(entry["dst_img"], img)
    write_tiff(entry["dst_msk"], msk)
    return 1
