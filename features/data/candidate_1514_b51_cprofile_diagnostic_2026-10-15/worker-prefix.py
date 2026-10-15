import builtins
import cProfile
import functools
import io
import json
import os
from pathlib import Path
import pstats
import sys
import threading
import time
import traceback

_PROFILE_ROOT = Path(os.environ['SPACR_DIAGNOSTIC_PROFILE_DIR'])
_EXPECTED_SITE = Path(os.environ['SPACR_BENCHMARK_PACKAGE_ROOT']).resolve()
_original_import = builtins.__import__
_patched = set()
_calls = []


def _profile_call(label, original, args, kwargs):
    profile = cProfile.Profile()
    row = {'label': label, 'thread_name': threading.current_thread().name,
           'thread_ident': threading.get_ident(), 'started_monotonic': time.monotonic(),
           'original_module': original.__module__, 'original_qualname': original.__qualname__,
           'original_file': original.__code__.co_filename,
           'original_first_line': original.__code__.co_firstlineno}
    _calls.append(row)
    index = len(_calls)
    try:
        return profile.runcall(original, *args, **kwargs)
    except BaseException:
        row['exception'] = traceback.format_exc()
        raise
    finally:
        row['ended_monotonic'] = time.monotonic()
        row['elapsed_seconds_with_profiling'] = row['ended_monotonic'] - row['started_monotonic']
        try:
            stem = f'{index:02d}-{label}'
            path = _PROFILE_ROOT / (stem + '.pstats')
            profile.dump_stats(str(path))
            row['pstats_file'] = path.name
            for sort in ('cumulative', 'tottime'):
                stream = io.StringIO()
                pstats.Stats(profile, stream=stream).sort_stats(sort).print_stats(50)
                (_PROFILE_ROOT / (stem + '-' + sort + '.txt')).write_text(stream.getvalue())
            (_PROFILE_ROOT / 'profile-calls.json').write_text(json.dumps(_calls, indent=2) + '\n')
        except BaseException:
            print('DIAGNOSTIC PROFILE OUTPUT ERROR: ' + traceback.format_exc(), file=sys.stderr, flush=True)


def _wrap_callback(original, label):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        key = args[0] if args else kwargs.get('key')
        if label != 'mask-build' or key == 'mask':
            return _profile_call(label, original, (self, *args), kwargs)
        return original(self, *args, **kwargs)
    return wrapped


def _patch_loaded():
    for module_name, class_name, method_name, label in (
            ('spacr.qt.app', 'MainWindow', '_build_screen', 'mask-build'),
            ('spacr.qt.startup_benchmark', 'BenchmarkController', '_open_preferences', 'preferences-open')):
        if label in _patched:
            continue
        module = sys.modules.get(module_name)
        cls = getattr(module, class_name, None) if module is not None else None
        original = getattr(cls, method_name, None) if cls is not None else None
        if original is None:
            continue
        actual = Path(module.__file__).resolve()
        if not actual.is_relative_to(_EXPECTED_SITE):
            raise RuntimeError(f'Diagnostic refuses noninstalled target: {actual}')
        setattr(cls, method_name, _wrap_callback(original, label))
        _patched.add(label)
    if len(_patched) == 2:
        builtins.__import__ = _original_import
        try:
            (_PROFILE_ROOT / 'instrumentation.json').write_text(json.dumps({
                'patched': sorted(_patched), 'import_hook_restored': True,
                'entrypoint': 'unchanged original driver WORKER suffix',
                'scope': 'Only synchronous original callback duration; deferred work after return is not profiled.'}, indent=2) + '\n')
        except BaseException:
            print('DIAGNOSTIC PROFILE OUTPUT ERROR: ' + traceback.format_exc(), file=sys.stderr, flush=True)


def _import_after_original(name, globals=None, locals=None, fromlist=(), level=0):
    result = _original_import(name, globals, locals, fromlist, level)
    _patch_loaded()
    return result


builtins.__import__ = _import_after_original
