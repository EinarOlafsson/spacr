"""Read-only numeric NPZ extraction, ownership and bounded memory experiment."""
import argparse
import ast
import errno
import gc
import hashlib
import json
import math
import mmap
import os
import resource
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('SPACR_PROOF_DIR', Path(__file__).resolve().parent))
REPO = Path(os.environ.get('SPACR_PROOF_REPO', '/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006'))
tree = ast.parse((REPO / 'spacr/utils.py').read_text())
node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == 'prepare_batch_for_segmentation')
namespace = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<audited-prepare>', 'exec'), namespace)
prepare = namespace[node.name]


def residency():
    values = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines()[1:]:
        key, value = line.split(':', 1)
        if value.strip().endswith('kB'):
            values[key] = int(value.split()[0])
    return values


def rss():
    return int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])


@contextmanager
def extracted(path, *, cancel_after=None, fail_reserve=False):
    """Reserve one private NPY, stream its original bytes, then borrow a map."""
    mapped = None
    with tempfile.TemporaryDirectory(prefix='reader-', dir=ROOT) as stage:
        target = Path(stage) / 'data.npy'
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo('data.npy')
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError('scratch proof supports NPY v1/v2 only')
                header_bytes = member.tell()
            if dtype.hasobject:
                raise ValueError('Object arrays cannot be mapped without pickle')
            expected = header_bytes + math.prod(shape) * dtype.itemsize
            if expected != info.file_size:
                raise ValueError('Numeric member payload length differs from NPY header')
            stats = os.statvfs(stage)
            if stats.f_bavail * stats.f_frsize < expected:
                raise OSError(errno.ENOSPC, 'not enough scratch disk')
            descriptor = os.open(target, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                if fail_reserve:
                    raise OSError(errno.ENOSPC, 'injected reservation failure')
                os.posix_fallocate(descriptor, 0, expected)
                if os.fstat(descriptor).st_blocks * 512 < expected:
                    raise AssertionError('fixture filesystem did not reserve payload')
                written = 0
                with archive.open(info) as member:
                    while chunk := member.read(1024 * 1024):
                        if cancel_after is not None and written >= cancel_after:
                            raise RuntimeError('injected cancellation')
                        view = memoryview(chunk)
                        while view:
                            count = os.write(descriptor, view)
                            if count <= 0:
                                raise OSError(errno.ENOSPC, 'short scratch write')
                            view = view[count:]
                        written += len(chunk)
                if written != expected:
                    raise ValueError('extracted payload incomplete')
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            with archive.open('filenames.npy') as member:
                filenames = np.load(member, allow_pickle=False)
        try:
            mapped = np.load(target, mmap_mode='r', allow_pickle=False)
            yield mapped, filenames
        finally:
            if mapped is not None:
                mapped._mmap.close()


def fixture():
    shape = (8, 8, 512, 512, 2)
    staged = ROOT / 'fixture-input.npy'
    mapped = np.lib.format.open_memmap(staged, mode='w+', dtype=np.float32, shape=shape)
    field = np.arange(math.prod(shape[1:]), dtype=np.float32).reshape(shape[1:]) % 1024
    for t in range(shape[0]):
        mapped[t] = field + t
    del field
    np.savez_compressed(ROOT / 'large.npz', data=mapped,
                        filenames=np.array([f't{index:03}.npy' for index in range(shape[0])]))
    mapped._mmap.close()
    del mapped
    staged.unlink()
    print('fixture bytes', math.prod(shape) * 4, 'compressed', (ROOT / 'large.npz').stat().st_size)


def memory(mode):
    baseline = rss()
    started = time.perf_counter()
    path = ROOT / 'large.npz'
    if mode == 'eager':
        with np.load(path, allow_pickle=False) as archive:
            stack, filenames = archive['data'], archive['filenames']
        manager = None
    else:
        manager = extracted(path)
        stack, filenames = manager.__enter__()
    load_seconds, loaded = time.perf_counter() - started, rss()
    first = np.take(np.asarray(stack)[:1], [1, 0], axis=-1)
    assert not np.shares_memory(first, stack) and first.flags.writeable
    before = hashlib.sha256(stack[0].tobytes()).hexdigest()
    prepared = prepare(first)
    assert before == hashlib.sha256(stack[0].tobytes()).hexdigest()
    result = {'mode': mode, 'baseline_rss_kib': baseline, 'loaded_rss_kib': loaded,
              'after_one_owned_field_rss_kib': rss(), 'archive_array_bytes': stack.nbytes,
              'owned_field_bytes': prepared.nbytes, 'load_seconds': load_seconds,
              'prepared_sha256': hashlib.sha256(prepared.tobytes()).hexdigest(),
              'filenames': filenames.tolist(), 'peak_rss_kib': int(Path('/proc/self/status').read_text().split('VmHWM:')[1].split()[0]),
              'scope': 'load and all configured two-field copies; no model or full pipeline memory claim'}
    result['first_field_residency_kib'] = residency()
    del first, prepared
    field_hashes = []
    batch_residency = []
    for start in range(0, stack.shape[0], 2):
        owned = np.take(np.asarray(stack)[start:start + 2], [1, 0], axis=-1)
        assert not np.shares_memory(owned, stack) and owned.flags.writeable
        if mode == 'mapped_drop':
            stack._mmap.madvise(mmap.MADV_DONTNEED)
        prepared = prepare(owned)
        field_hashes.append(hashlib.sha256(prepared.tobytes()).hexdigest())
        batch_residency.append(residency())
        del owned, prepared
    result['all_two_field_batches_residency_kib'] = batch_residency
    result['all_batch_sha256'] = field_hashes
    result['peak_rss_kib'] = int(Path('/proc/self/status').read_text().split('VmHWM:')[1].split()[0])
    del stack, filenames
    if manager is not None:
        manager.__exit__(None, None, None)
    gc.collect()
    result['cleanup_rss_kib'] = rss()
    (ROOT / (mode + '_memory.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def small():
    import sys
    sys.path.insert(0, str(REPO))
    from spacr.zstack import TStackSpec, as_t_first, segment_4d
    records = []
    for dtype in ['<f4', '>u2']:
        for order in ['C', 'F']:
            for t_axis in [0, 1]:
                shape = (3, 4, 7, 9, 2) if t_axis == 0 else (4, 3, 7, 9, 2)
                values = np.arange(math.prod(shape)).reshape(shape).astype(dtype)
                if order == 'F':
                    values = np.asfortranarray(values)
                path = ROOT / 'small.npz'
                names = np.array([f't{index}.npy' for index in range(3)])
                np.savez_compressed(path, data=values, filenames=names)
                with np.load(path, allow_pickle=False) as archive:
                    eager = archive['data']
                plan = TStackSpec(t_axis=t_axis, z_axis=1 - t_axis,
                                  z_mode='volumetric', anisotropy=2)
                with extracted(path) as (mapped, filenames):
                    assert mapped.dtype == eager.dtype and mapped.shape == eager.shape
                    assert mapped.flags.f_contiguous == eager.flags.f_contiguous
                    assert np.array_equal(mapped, eager) and np.array_equal(filenames, names)
                    borrowed = as_t_first(mapped, plan)
                    expected = as_t_first(eager, plan)
                    assert np.shares_memory(borrowed, mapped)
                    assert not borrowed.flags.writeable
                    owned = np.take(borrowed, [1, 0], axis=-1)
                    assert not np.shares_memory(owned, mapped) and owned.flags.writeable
                    mapped._mmap.madvise(mmap.MADV_DONTNEED)
                    assert np.array_equal(borrowed, expected)
                    normal = prepare(owned)
                    reference = prepare(np.take(expected, [1, 0], axis=-1))
                    assert np.array_equal(normal, reference) and not np.shares_memory(normal, mapped)
                    aligned = TStackSpec(t_axis=0, z_axis=1, z_mode='volumetric', anisotropy=2)
                    def segment(volume, **_kwargs):
                        return (volume[..., 0] > .5).astype(np.uint16)
                    actual = segment_4d(normal, aligned, segment)
                    original = segment_4d(reference, aligned, segment)
                    assert np.array_equal(actual.labels, original.labels)
                    assert actual.notes == original.notes
                    assert [r.notes for r in actual.z_results] == [r.notes for r in original.z_results]
                    assert np.array_equal(mapped, eager)
                    del borrowed, mapped
                assert np.array_equal(normal, reference)
                records.append({'dtype': dtype, 'order': order, 't_axis': t_axis,
                                'exact_array_channels_normalized_labels_notes': True,
                                'borrowed_readonly_view_and_owned_writable_copy': True})
                path.unlink()
    failures = []
    for kwargs in [{'cancel_after': 2 * 1024 * 1024}, {'fail_reserve': True}]:
        existing = set(ROOT.glob('reader-*'))
        try:
            with extracted(ROOT / 'large.npz', **kwargs):
                raise AssertionError('injected failure did not fire')
        except (RuntimeError, OSError):
            pass
        assert set(ROOT.glob('reader-*')) == existing
        failures.append({'injected': kwargs, 'private_workspace_removed': True})
    result = {'source_sha256': {name: hashlib.sha256((REPO / 'spacr' / name).read_bytes()).hexdigest()
                                for name in ['object.py', 'zstack.py', 'utils.py', 'io.py']},
              'source_commit': '711999b77d1da51ca189fdf43f470021d1048037',
              'cases': records, 'cleanup': failures, 'production_change': False,
              'scope': 'scratch extraction/owned preprocessing and synthetic CPU segmenter; no real model or pipeline claim'}
    (ROOT / 'ownership_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fixture', 'eager', 'mapped', 'mapped_drop', 'small'])
    arguments = parser.parse_args()
    if arguments.mode == 'fixture':
        fixture()
    elif arguments.mode == 'small':
        small()
    else:
        memory(arguments.mode)
