"""Native Mask archive staging preserves scientific pixels, batches and ownership."""
from __future__ import annotations

import errno
import io
import mmap
import os
import shutil
import stat
import weakref
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import spacr.object as O
from spacr import cancellation
from spacr.utils import prepare_batch_for_segmentation
from spacr.zstack import TStackSpec, as_t_first
from tests import test_object_tstack_wiring as wiring


def _archive(path, values, *, version=None, payload_tail=0):
    """Write the original NPY member bytes, optionally with an invalid tail."""
    source = io.BytesIO()
    np.lib.format.write_array(source, values, version=version)
    payload = source.getvalue()
    if payload_tail > 0:
        payload = payload[:-payload_tail]
    elif payload_tail < 0:
        payload += bytes(-payload_tail)
    names = io.BytesIO()
    np.save(names, np.array([f'plate1_A01_{index + 1}.npy' for index in range(3)]))
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data.npy', payload)
        archive.writestr('filenames.npy', names.getvalue())
    return path


def _private_dirs(path):
    """Find only this reader's private sibling workspaces."""
    return list(Path(path).parent.rglob('.spacr-native-mask-*'))


@pytest.mark.parametrize('dtype', ['<f4', '>u2'])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('t_axis', [0, 1])
@pytest.mark.parametrize('version', [(1, 0), (2, 0), (3, 0)])
def test_numeric_member_header_order_axes_and_owned_normalization_match(
        tmp_path, dtype, order, t_axis, version):
    shape = (3, 4, 7, 9, 2) if t_axis == 0 else (4, 3, 7, 9, 2)
    values = np.array(np.arange(np.prod(shape)).reshape(shape), dtype=dtype, order=order)
    path = _archive(tmp_path / 'batch.npz', values, version=version)
    with np.load(path) as archive:
        expected, names = archive['data'], archive['filenames']
    plan = TStackSpec(t_axis=t_axis, z_axis=1 - t_axis, z_mode='volumetric', anisotropy=2)
    with O._mask_archive_arrays(path, native=True) as (stack, _filenames, owner):
        target = Path(owner.filename)
        assert type(stack) is np.ndarray and not stack.flags.writeable
        assert stack.dtype == expected.dtype and stack.shape == expected.shape
        assert stack.flags.f_contiguous == expected.flags.f_contiguous
        assert np.array_equal(stack, expected) and np.array_equal(_filenames, names)
        assert np.shares_memory(stack, owner)
        if os.name == 'posix':
            assert stat.S_IMODE(target.stat().st_mode) == 0o600
            assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
        borrowed = as_t_first(stack, plan)
        owned = np.take(borrowed, [1, 0], axis=-1)
        assert owned.flags.writeable and not np.shares_memory(owned, owner)
        O._release_native_mask_pages(owner, owned)
        assert np.array_equal(borrowed, as_t_first(expected, plan))
        normalized = prepare_batch_for_segmentation(owned)
        reference = prepare_batch_for_segmentation(np.take(as_t_first(expected, plan), [1, 0], axis=-1))
        assert np.array_equal(normalized, reference)
        assert np.array_equal(stack, expected)
        del borrowed, stack, owner
    assert np.array_equal(normalized, reference)
    assert not target.exists() and not _private_dirs(path)


@pytest.mark.parametrize('shape', [(0, 4, 7, 9, 2), (3, 0, 7, 9, 2)])
def test_zero_length_numeric_axes_close_private_workspace(tmp_path, shape):
    path = _archive(tmp_path / 'empty.npz', np.empty(shape, np.float32))
    with O._mask_archive_arrays(path, native=True) as (stack, _filenames, owner):
        assert stack.shape == shape and stack.size == 0
        target = Path(owner.filename)
        del stack, owner
    assert not target.exists() and not _private_dirs(path)


def test_eager_archive_never_reserves_or_changes_source_writeability(tmp_path, monkeypatch):
    values = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    path = _archive(tmp_path / 'batch.npz', values)
    def refuse(*args):
        raise AssertionError('ordinary loader attempted private reservation')
    monkeypatch.setattr(O, '_reserve_native_mask_file', refuse)
    with O._mask_archive_arrays(path) as (stack, _filenames, owner):
        assert owner is None and stack.flags.writeable
        stack[0, 0, 0] = -1
    assert not _private_dirs(path)
    with np.load(path) as archive:
        assert np.array_equal(archive['data'], values)


@pytest.mark.parametrize('failure', ['object', 'truncated', 'oversized', 'crc', 'disk', 'reservation', 'consumer'])
def test_failed_archive_never_leaves_a_private_file(tmp_path, monkeypatch, failure):
    values = np.arange(8192, dtype=np.float32).reshape(1, 32, 32, 8)
    if failure == 'object':
        values = np.array([object()], dtype=object)
    path = _archive(tmp_path / 'bad.npz', values, payload_tail=1 if failure == 'truncated' else -1 if failure == 'oversized' else 0)
    if failure == 'crc':
        with zipfile.ZipFile(path) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        with zipfile.ZipFile(path) as archive:
            member = archive.getinfo('data.npy')
            offset = member.header_offset + 30 + len(member.filename) + len(member.extra)
        with path.open('r+b') as handle:
            handle.seek(offset + member.file_size - 1)
            value = handle.read(1)[0]
            handle.seek(-1, 1)
            handle.write(bytes([value ^ 1]))
    if failure == 'disk':
        monkeypatch.setattr(shutil, 'disk_usage', lambda stage: type('Space', (), {'free': 0})())
    if failure == 'reservation':
        def fail_reservation(*args):
            raise OSError(errno.ENOSPC, 'injected disk quota failure')
        monkeypatch.setattr(O, '_reserve_native_mask_file', fail_reservation)
    with (pytest.raises((ValueError, OSError, zipfile.BadZipFile, RuntimeError)),
          O._mask_archive_arrays(path, native=True) as (stack, _filenames, owner)):
        if failure == 'consumer':
            owned = np.take(stack, [0], axis=-1)
            del stack, owner
            raise RuntimeError('downstream processing failed')
        raise AssertionError('invalid archive accepted')
    assert not _private_dirs(path)
    if failure == 'consumer':
        assert owned.flags.writeable and owned[0, 0, 0, 0] == 0


@pytest.mark.parametrize('unsupported', [errno.ENOSYS, errno.EOPNOTSUPP])
def test_reservation_fallback_handles_partial_writes_and_checkpoints(tmp_path, monkeypatch, unsupported):
    calls = []
    monkeypatch.setattr(os, 'posix_fallocate', lambda *args: (_ for _ in ()).throw(OSError(unsupported, 'unsupported')),
                        raising=False)
    original = os.write
    monkeypatch.setattr(os, 'write', lambda fd, data: original(fd, data[:113]))
    monkeypatch.setattr(cancellation, 'checkpoint', lambda: calls.append(True))
    path = tmp_path / 'reserved.npy'
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        O._reserve_native_mask_file(descriptor, 8193)
    finally:
        os.close(descriptor)
    assert path.read_bytes() == bytes(8193) and len(calls) > 70


@pytest.mark.parametrize('failure', ['zero_write', 'quota', 'cancel_reserve', 'cancel_stream'])
def test_reservation_and_stream_abort_without_exposure(tmp_path, monkeypatch, failure):
    values = np.arange(8 * 128 * 128 * 8, dtype=np.float32).reshape(1, 8, 128, 128, 8)
    path = _archive(tmp_path / 'batch.npz', values)
    token = cancellation.CancellationToken()
    if failure == 'quota':
        monkeypatch.setattr(os, 'posix_fallocate', lambda *args: (_ for _ in ()).throw(OSError(errno.EDQUOT, 'quota')),
                            raising=False)
    elif failure in ['zero_write', 'cancel_reserve']:
        monkeypatch.delattr(os, 'posix_fallocate', raising=False)
        original = os.write
        def write(fd, data):
            if failure == 'zero_write':
                return 0
            result = original(fd, data)
            token.cancel()
            return result
        monkeypatch.setattr(os, 'write', write)
    else:
        original_reserve = O._reserve_native_mask_file
        original_write = O._write_native_mask_chunk
        ready = []
        def reserve(*args):
            original_reserve(*args)
            ready.append(True)
        def write(*args):
            original_write(*args)
            if ready:
                token.cancel()
        monkeypatch.setattr(O, '_reserve_native_mask_file', reserve)
        monkeypatch.setattr(O, '_write_native_mask_chunk', write)
    with (cancellation.installed_token(token),
          pytest.raises((OSError, cancellation.PipelineCancelled)),
          O._mask_archive_arrays(path, native=True)):
        raise AssertionError('aborted member was exposed')
    assert not _private_dirs(path)


@pytest.mark.parametrize('unsupported', ['missing', 'error'])
def test_optional_page_release_fallback_preserves_borrowed_views(tmp_path, monkeypatch, unsupported):
    values = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    path = _archive(tmp_path / 'batch.npz', values)
    with O._mask_archive_arrays(path, native=True) as (stack, _filenames, owner):
        owned = np.take(stack, [1, 0], axis=-1)
        if unsupported == 'missing':
            monkeypatch.delattr(mmap, 'MADV_DONTNEED', raising=False)
        else:
            monkeypatch.setattr(mmap, 'MADV_DONTNEED', 10 ** 9, raising=False)
        O._release_native_mask_pages(owner, owned)
        assert np.array_equal(stack, values) and np.array_equal(owned, values[..., [1, 0]])
        del stack, owner
    assert not _private_dirs(path)


def test_page_release_rejects_source_aliases_and_readonly_batches(tmp_path):
    path = _archive(tmp_path / 'batch.npz', np.arange(24, dtype=np.float32).reshape(3, 4, 2))
    with O._mask_archive_arrays(path, native=True) as (stack, _filenames, owner):
        with pytest.raises(ValueError, match='own writable'):
            O._release_native_mask_pages(owner, stack)
        owned = stack.copy()
        owned.flags.writeable = False
        with pytest.raises(ValueError, match='own writable'):
            O._release_native_mask_pages(owner, owned)
        del stack, owner


@pytest.fixture
def model_double(monkeypatch):
    return wiring.fake_model.__wrapped__(monkeypatch)


@pytest.mark.parametrize('plan', ['t_volume', 't_project', 't_flat', 'z_volume', 'z_stitch'])
def test_native_generator_keeps_batches_full_masks_rows_and_eval_contracts(
        tmp_path, monkeypatch, capsys, model_double, plan):
    shape = (3, 32, 32, 2) if plan == 't_flat' else (3, 4, 32, 32, 2)
    common = {'batch_size': 2}
    if plan.startswith('t_'):
        common.update(t_stack=True, t_axis_order='TYX' if plan == 't_flat' else 'TZYX',
                      z_segmentation_mode='project' if plan == 't_project' else 'volumetric', anisotropy=2)
    else:
        common.update(z_stack=True, z_axis=0,
                      z_segmentation_mode='stitch' if plan == 'z_stitch' else 'volumetric', anisotropy=2)
    original = O._mask_archive_arrays
    results = []
    for mode in ['eager', 'mapped']:
        src = tmp_path / mode / 'stack'
        wiring._write_npz(src, shape, seed=7)
        source_bytes = (src / 'batch1.npz').read_bytes()
        borrowed_owners = []
        @contextmanager
        def loader(path, *, native, workspace=None, mode=mode, borrowed_owners=borrowed_owners):
            assert native
            with original(path, native=mode == 'mapped', workspace=workspace) as arrays:
                if arrays[2] is not None:
                    borrowed_owners.append(weakref.ref(arrays[2]))
                yield arrays
        monkeypatch.setattr(O, '_mask_archive_arrays', loader)
        completed = []
        def done(path, borrowed_owners=borrowed_owners, completed=completed):
            assert not _private_dirs(path)
            assert all(owner() is None for owner in borrowed_owners)
            completed.append(Path(path).name)
        settings = wiring._base_settings(src, **common)
        O.generate_cellpose_masks_sam(str(src), settings, 'cell', on_batch_done=done, run_qc=False)
        model = model_double['model']
        results.append((wiring._artifacts(src), model.eval_shapes, model.eval_kwargs,
                        completed, capsys.readouterr().out))
        assert settings['batch_size'] == 2 and completed == ['batch1.npz']
        assert (src / 'batch1.npz').read_bytes() == source_bytes
    assert results[0] == results[1]


@pytest.mark.parametrize('resume', [False, True])
def test_empty_and_already_completed_native_archives_close_before_callback(
        tmp_path, model_double, resume):
    src = tmp_path / 'stack'
    shape = (2 if resume else 0, 4, 32, 32, 2)
    wiring._write_npz(src, shape, seed=7)
    settings = wiring._base_settings(src, batch_size=2, t_stack=True,
                                    t_axis_order='TZYX', z_segmentation_mode='volumetric', anisotropy=2)
    if resume:
        O.generate_cellpose_masks_sam(str(src), dict(settings), 'cell', run_qc=False)
        settings['resume'] = True
    completed = []
    def done(path):
        assert not _private_dirs(path)
        completed.append(Path(path).name)
    O.generate_cellpose_masks_sam(str(src), settings, 'cell', on_batch_done=done, run_qc=False)
    assert completed == ['batch1.npz'] and not model_double['model'].eval_shapes


@pytest.mark.parametrize('failure', ['eval', 'cancel'])
def test_generator_failure_closes_owner_and_never_marks_archive_complete(
        tmp_path, monkeypatch, model_double, failure):
    src = tmp_path / 'stack'
    wiring._write_npz(src, (3, 4, 32, 32, 2), seed=7)
    settings = wiring._base_settings(src, batch_size=2, t_stack=True,
                                    t_axis_order='TZYX', z_segmentation_mode='volumetric', anisotropy=2)
    owners = []
    original = O._mask_archive_arrays
    @contextmanager
    def loader(path, *, native, workspace=None):
        with original(path, native=native, workspace=workspace) as arrays:
            owners.append(weakref.ref(arrays[2]))
            try:
                yield arrays
            finally:
                del arrays
    monkeypatch.setattr(O, '_mask_archive_arrays', loader)
    token = cancellation.CancellationToken()
    model_type = O.cp_models.CellposeModel
    original_eval = model_type.eval
    def evaluate(self, *args, **kwargs):
        if failure == 'eval':
            raise RuntimeError('model failed on owned data')
        result = original_eval(self, *args, **kwargs)
        token.cancel()
        return result
    monkeypatch.setattr(model_type, 'eval', evaluate)
    completed = []
    expected = RuntimeError if failure == 'eval' else cancellation.PipelineCancelled
    with cancellation.installed_token(token), pytest.raises(expected):
        O.generate_cellpose_masks_sam(str(src), settings, 'cell', on_batch_done=completed.append, run_qc=False)
    assert owners and all(owner() is None for owner in owners)
    assert not _private_dirs(src / 'batch1.npz') and completed == []


@pytest.mark.parametrize('t_stack', [False, True])
def test_legacy_timelapse_remains_eager_and_keeps_whole_series_cropping(
        tmp_path, monkeypatch, model_double, t_stack):
    import spacr.timelapse as timelapse_module
    src = tmp_path / 'stack'
    wiring._write_npz(src, (3, 32, 32, 2), seed=7)
    settings = wiring._base_settings(
        src, batch_size=1, timelapse=True, timelapse_objects=['cell'],
        timelapse_mode='trackpy', timelapse_frame_limits=[1, 3],
        t_stack=t_stack, t_axis_order='TYX')
    def refuse(*args):
        raise AssertionError('legacy tracking tried a private workspace')
    monkeypatch.setattr(O, '_reserve_native_mask_file', refuse)
    movies = []
    trackers = []
    def movie(images, names, path, **kwargs):
        movies.append((images.shape, list(names), kwargs))
    def tracking(**kwargs):
        trackers.append(list(kwargs['batch_filenames']))
        return kwargs['masks']
    monkeypatch.setattr(timelapse_module, '_npz_to_movie', movie)
    monkeypatch.setattr(timelapse_module, '_trackpy_track_cells', tracking)
    O.generate_cellpose_masks_sam(str(src), settings, 'cell', run_qc=False)
    assert movies == [((2, 32, 32, 2), ['plate1_A01_2.npy', 'plate1_A01_3.npy'], {'fps': 2})]
    assert trackers == [['plate1_A01_2.npy', 'plate1_A01_3.npy']]
    assert settings['timelapse_batch_size'] == 3
    assert len(wiring._artifacts(src)[0]) == 2 and not _private_dirs(src / 'batch1.npz')


@pytest.mark.parametrize('data_alias', [False, True])
@pytest.mark.parametrize('names_alias', [False, True])
def test_literal_zip_members_preserve_numpy_key_priority(tmp_path, data_alias, names_alias):
    values = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    path = _archive(tmp_path / 'aliases.npz', values)
    with zipfile.ZipFile(path, 'a') as archive:
        if data_alias:
            source = io.BytesIO()
            np.save(source, values + 100)
            archive.writestr('data', source.getvalue())
        if names_alias:
            source = io.BytesIO()
            np.save(source, np.array(['alias1.npy', 'alias2.npy', 'alias3.npy']))
            archive.writestr('filenames', source.getvalue())
    with np.load(path) as archive:
        expected, names = archive['data'], archive['filenames']
    with O._mask_archive_arrays(path, native=True) as (stack, filenames, owner):
        assert np.array_equal(stack, expected) and np.array_equal(filenames, names)
        del stack, owner
    assert not _private_dirs(path)


@pytest.mark.skipif(os.name != 'posix' or getattr(os, 'geteuid', lambda: 0)() == 0,
                    reason='requires POSIX source-directory permissions without root bypass')
def test_readonly_archive_directory_uses_precreated_writable_output(
        tmp_path, monkeypatch, model_double):
    import tempfile
    src = tmp_path / 'stack'
    wiring._write_npz(src, (3, 4, 32, 32, 2), seed=7)
    output = src / 'cell_mask_stack'
    output.mkdir()
    chosen = []
    original = tempfile.TemporaryDirectory
    def stage(*args, **kwargs):
        chosen.append(Path(kwargs['dir']))
        return original(*args, **kwargs)
    monkeypatch.setattr(tempfile, 'TemporaryDirectory', stage)
    src.chmod(0o555)
    settings = wiring._base_settings(src, batch_size=2, t_stack=True, t_axis_order='TZYX',
                                    z_segmentation_mode='volumetric', anisotropy=2)
    try:
        O.generate_cellpose_masks_sam(str(src), settings, 'cell', run_qc=False)
        assert chosen == [output]
        assert len(wiring._artifacts(src)[0]) == 3
        assert not _private_dirs(src / 'batch1.npz')
    finally:
        src.chmod(0o755)


def test_early_eof_numeric_reader_is_refused_before_array_exposure(tmp_path, monkeypatch):
    path = _archive(tmp_path / 'short.npz', np.arange(8192, dtype=np.float32).reshape(1, 32, 32, 8))
    original_open = zipfile.ZipFile.open
    class EarlyEOF:
        def __init__(self, source):
            self.source = source
            self.read_once = False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.source.close()
        def read(self, size):
            if self.read_once:
                return b''
            self.read_once = True
            return self.source.read(min(size, 1024))
    def open_member(self, name, *args, **kwargs):
        source = original_open(self, name, *args, **kwargs)
        if getattr(name, 'filename', name) == 'data.npy':
            return EarlyEOF(source)
        return source
    monkeypatch.setattr(zipfile.ZipFile, 'open', open_member)
    with pytest.raises(ValueError, match='numeric member is incomplete'), O._mask_archive_arrays(path, native=True):
        raise AssertionError('partial numeric member was exposed')
    assert not _private_dirs(path)


@pytest.mark.parametrize('mapped', [False, True])
def test_archive_collection_runs_after_eager_or_mapped_source_references_die(
        tmp_path, monkeypatch, model_double, mapped):
    src = tmp_path / 'stack'
    wiring._write_npz(src, (3, 4, 32, 32, 2), seed=7)
    source_refs = []
    active = []
    collections = []
    original = O._mask_archive_arrays
    @contextmanager
    def loader(path, *, native, workspace=None):
        with original(path, native=mapped, workspace=workspace) as arrays:
            source_refs.append(weakref.ref(arrays[0]))
            active.append(True)
            try:
                yield arrays
            finally:
                active.clear()
                del arrays
    def collect(*args, **kwargs):
        collections.append((bool(active), [reference() is not None for reference in source_refs]))
    monkeypatch.setattr(O, '_mask_archive_arrays', loader)
    monkeypatch.setattr(O.gc, 'collect', collect)
    settings = wiring._base_settings(src, batch_size=2, t_stack=True, t_axis_order='TZYX',
                                    z_segmentation_mode='volumetric', anisotropy=2)
    O.generate_cellpose_masks_sam(str(src), settings, 'cell', run_qc=False)
    assert len(collections) >= 2
    assert all(not live_scope and not any(live_refs) for live_scope, live_refs in collections)
    assert source_refs and all(reference() is None for reference in source_refs)


def _map_owner(array):
    """Find the exact private map retained by NumPy's borrowed base chain."""
    while array is not None:
        if isinstance(array, np.memmap):
            return array
        array = getattr(array, 'base', None)
    raise AssertionError('borrowed source view lost its map owner')


@pytest.mark.parametrize('t_axis', [0, 1])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_retained_time_view_keeps_exact_owner_until_close_before_unlink(
        tmp_path, monkeypatch, t_axis, order):
    import gc
    import tempfile
    shape = (3, 4, 7, 9, 2) if t_axis == 0 else (4, 3, 7, 9, 2)
    values = np.array(np.arange(np.prod(shape)).reshape(shape), dtype=np.float32, order=order)
    path = _archive(tmp_path / 'view.npz', values)
    plan = TStackSpec(t_axis=t_axis, z_axis=1 - t_axis, z_mode='volumetric')
    events = []
    handles = []
    original_cleanup = tempfile.TemporaryDirectory.cleanup
    def cleanup(stage):
        assert handles and handles[0].closed
        events.append(stage.name)
        original_cleanup(stage)
    monkeypatch.setattr(tempfile.TemporaryDirectory, 'cleanup', cleanup)
    with O._mask_archive_arrays(path, native=True) as (stack, _names, owner):
        assert _map_owner(stack) is owner
        view = as_t_first(stack, plan)[1]
        assert _map_owner(view) is owner
        owner_ref = weakref.ref(owner)
        target = Path(owner.filename)
        handles.append(owner._mmap)
        del stack, owner
    assert owner_ref() is not None and target.exists() and events == []
    assert np.array_equal(view, as_t_first(values, plan)[1])
    del view
    gc.collect()
    assert owner_ref() is None and handles[0].closed
    assert not target.exists() and not _private_dirs(path) and events == [str(target.parent)]


@pytest.mark.parametrize('failure', ['axis', 'channels'])
def test_actual_generator_traceback_pixels_remain_readable_until_released(
        tmp_path, model_double, failure):
    import gc
    shape = (3, 32, 32, 2) if failure == 'axis' else (3, 4, 32, 32, 1)
    src = tmp_path / 'stack'
    values = wiring._write_npz(src, shape, seed=7)
    settings = wiring._base_settings(src, batch_size=2, t_stack=True, t_axis_order='TZYX',
                                    z_segmentation_mode='volumetric', anisotropy=2)
    with pytest.raises(Exception) as raised:
        O.generate_cellpose_masks_sam(str(src), settings, 'cell', run_qc=False)
    trace = raised.value.__traceback__
    retained = None
    while trace is not None:
        if trace.tb_frame.f_code.co_name == '_require_t_axis' and failure == 'axis':
            retained = trace.tb_frame.f_locals['stack']
            break
        if trace.tb_frame.f_code.co_name == '_wrapfunc' and failure == 'channels':
            retained = trace.tb_frame.f_locals['obj']
            break
        trace = trace.tb_next
    assert retained is not None and not retained.flags.writeable
    owner_ref = weakref.ref(_map_owner(retained))
    target = Path(_map_owner(retained).filename)
    handle = _map_owner(retained)._mmap
    assert target.exists() and not handle.closed
    expected = values if failure == 'axis' else values[:2]
    assert np.array_equal(retained, expected)
    assert raised.value.__traceback__ is not None
    retained = trace = None
    raised.value.__traceback__ = None
    del raised
    gc.collect()
    assert owner_ref() is None and handle.closed
    assert not target.exists() and not _private_dirs(src / 'batch1.npz')
