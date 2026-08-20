"""Contain duplicate OpenMP runtimes during model fitting on macOS.

Torch, scikit-learn, and XGBoost wheels can load independent LLVM OpenMP
runtimes into one process. Crossing their worker-thread state can terminate
the process, particularly when ``KMP_DUPLICATE_LIB_OK`` suppresses the
runtime's duplicate-initialization guard.

:func:`single_threaded_openmp` detects mapped runtimes and, when more than one
is resident, calls ``omp_set_num_threads(1)`` on the fitting thread. This
serializes that thread's parallel regions without imposing a process-wide
``OMP_NUM_THREADS=1`` limit on segmentation and other workloads. Estimator
``n_jobs`` and ``nthread`` arguments are not substitutes because they do not
prevent the OpenMP runtime from creating a worker team.

The module has no Qt dependency and is safe to use in command-line and cluster
pipelines.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import threading
from typing import List

__all__ = [
    "resident_openmp_runtimes",
    "openmp_runtime_is_duplicated",
    "single_threaded_openmp",
    "guarded_n_jobs",
]

# Substrings that identify an OpenMP runtime image. `libgomp` is GCC's and
# `libiomp5` is Intel's; mixing any two of the three is the same hazard.
_OPENMP_MARKERS = ("libomp.", "libgomp.", "libiomp5.", "libomp5.")

_OFF = {"0", "off", "false", "no"}
_ON = {"1", "on", "true", "yes"}

_LOCK = threading.Lock()
_WARNED = False
_HANDLES: dict = {}


def _guard_disabled() -> bool:
    """True when the user has opted out via ``SPACR_OPENMP_GUARD=off``."""
    return os.environ.get("SPACR_OPENMP_GUARD", "").strip().lower() in _OFF


def _guard_forced() -> bool:
    """True when ``SPACR_OPENMP_GUARD=on`` asks for the clamp everywhere."""
    return os.environ.get("SPACR_OPENMP_GUARD", "").strip().lower() in _ON


def _clamping_platform() -> bool:
    """Whether a duplicate runtime is treated as fatal here, or only noted.

    The crash this module exists for is macOS: dyld bound one libomp image's
    barrier code to another image's ``__kmp_suspend_initialize_thread`` and the
    process died. Linux mixes runtimes at least as readily — ``libgomp``
    alongside ``libomp`` is the classic case — but spaCR's cluster runs do it
    every day without this fault, and measurement there is CPU-core-bound
    (~8 h a screen), so clamping on no evidence would cost real time to buy
    nothing. Report it there; clamp here. ``SPACR_OPENMP_GUARD=on`` overrides.
    """
    return _guard_forced() or sys.platform == "darwin"


def _looks_like_openmp(path: str) -> bool:
    name = os.path.basename(path)
    return any(marker in name for marker in _OPENMP_MARKERS)


def _macos_images() -> List[str]:
    """Loaded image paths, from dyld.

    ``_dyld_get_image_name`` is the only way to see what is actually mapped;
    walking site-packages would find copies that were never loaded and miss a
    system one that was.
    """
    from ctypes import CDLL, c_char_p, c_uint32, util

    libc = CDLL(util.find_library("c"))
    libc._dyld_image_count.restype = c_uint32
    libc._dyld_get_image_name.argtypes = [c_uint32]
    libc._dyld_get_image_name.restype = c_char_p
    return [
        libc._dyld_get_image_name(i).decode("utf-8", "replace")
        for i in range(libc._dyld_image_count())
    ]


def _linux_images() -> List[str]:
    """Loaded image paths, from the process's own memory map."""
    paths = []
    with open("/proc/self/maps", "r") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(None, 5)
            if len(parts) == 6 and parts[5].startswith("/"):
                paths.append(parts[5])
    return paths


def resident_openmp_runtimes() -> List[str]:
    """Distinct OpenMP runtime files currently mapped into this process.

    Returns resolved paths, deduplicated and sorted. Two copies of a
    byte-identical build at two paths are still two runtimes with two sets of
    globals, so they are counted separately — but a symlink and its target are
    one file and are not.

    Returns ``[]`` on any platform or failure where the answer is unknown.
    Unknown must read as "no evidence of trouble", because the caller's
    fallback is the behaviour spaCR has always had.
    """
    try:
        if sys.platform == "darwin":
            images = _macos_images()
        elif sys.platform.startswith("linux"):
            images = _linux_images()
        else:
            return []
    except Exception:
        return []

    found = set()
    for path in images:
        if not _looks_like_openmp(path):
            continue
        try:
            found.add(os.path.realpath(path))
        except Exception:
            found.add(path)
    return sorted(found)


def openmp_runtime_is_duplicated() -> bool:
    """True when this process has more than one OpenMP runtime mapped.

    Reports the condition on every platform. Whether it is acted on is
    :func:`single_threaded_openmp`'s decision, not this one's.
    """
    if _guard_disabled():
        return False
    return len(resident_openmp_runtimes()) > 1


def _handle(path: str):
    """A cached ``CDLL`` for an already-loaded runtime.

    ``dlopen`` on a mapped image returns the existing handle and bumps a
    refcount; it does not load a second copy, which would be the one thing this
    module must never do.
    """
    handle = _HANDLES.get(path)
    if handle is None:
        handle = ctypes.CDLL(path)
        handle.omp_get_max_threads.restype = ctypes.c_int
        handle.omp_set_num_threads.argtypes = [ctypes.c_int]
        handle.omp_set_num_threads.restype = None
        _HANDLES[path] = handle
    return handle


def _warn_once(runtimes: List[str], label: str) -> None:
    global _WARNED
    with _LOCK:
        if _WARNED:
            return
        _WARNED = True
    print(
        f"[openmp_guard] {len(runtimes)} OpenMP runtimes are loaded in this "
        f"process. Mixing them crashes the interpreter (SIGSEGV inside "
        f"libomp's thread barrier), so {label} will run single-threaded on "
        f"this thread. It will be slower and it will finish."
    )
    for path in runtimes:
        print(f"[openmp_guard]   {path}")
    print(
        "[openmp_guard] To get the threads back, make the process load one "
        "runtime: install xgboost, torch and scikit-learn from builds that "
        "share an OpenMP, or set SPACR_OPENMP_GUARD=off to accept the risk."
    )


def guarded_n_jobs(requested, label: str = "this step"):
    """``1`` while the clamp is in force, otherwise ``requested`` unchanged.

    For joblib call sites *inside* a :class:`single_threaded_openmp` region —
    ``permutation_importance`` above all, which re-enters the fitted model.

    The clamp is a per-thread ICV, so a joblib worker thread starts with the
    process default (10 here, not 1) and builds a team the region was supposed
    to prevent. Measured on this repository's own ``ml_analysis``: 19 threads
    parked in ``__kmp_launch_worker`` unguarded, still 10 with only the region
    clamp, and those 10 belong to Homebrew's libomp — xgboost's runtime, the
    one that crashed. Keeping joblib on the calling thread keeps the clamp
    meaningful.

    This is NOT the earlier, wrong idea of clamping the estimator's own thread
    argument: that was measured to change nothing at all.
    """
    try:
        if _guard_disabled() or not _clamping_platform():
            return requested
        return 1 if len(resident_openmp_runtimes()) > 1 else requested
    except Exception:
        return requested


class single_threaded_openmp(contextlib.ContextDecorator):
    """Serialize this thread's OpenMP regions while several runtimes are mapped.

    Usable as a decorator or a ``with`` block::

        @single_threaded_openmp("classical ML")
        def ml_analysis(...):
            ...

    Does nothing at all when one runtime (or none) is mapped, when the platform
    is not one where the fault is documented, or when the user has set
    ``SPACR_OPENMP_GUARD=off``. Restores each runtime's previous value on the
    way out, so the clamp lasts exactly as long as the region it wraps.

    Never raises. A failure here leaves the thread exactly as it was: this
    guard protects a run, it must not be able to end one (INVARIANTS §10).
    """

    def __init__(self, label: str = "this model"):
        self.label = label
        self._restore: List[tuple] = []

    def _recreate_cm(self):
        """Give every decorated invocation independent restoration state.

        ``ContextDecorator`` otherwise reuses this instance for every call.
        A recursive or concurrent pipeline invocation could then overwrite
        another invocation's ``_restore`` list and leave its OpenMP thread
        limit pinned at one after both calls return.
        """
        return type(self)(self.label)

    def __enter__(self):
        self._restore = []
        try:
            if _guard_disabled() or not _clamping_platform():
                return self
            runtimes = resident_openmp_runtimes()
            if len(runtimes) <= 1:
                return self
            _warn_once(runtimes, self.label)
            for path in runtimes:
                try:
                    handle = _handle(path)
                    previous = handle.omp_get_max_threads()
                    handle.omp_set_num_threads(1)
                    self._restore.append((handle, previous))
                except Exception:
                    # A runtime without the symbols, or one that will not
                    # dlopen. Leave it alone; the others still help.
                    continue
        except Exception:
            self._restore = []
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle, previous in reversed(self._restore):
            try:
                handle.omp_set_num_threads(previous)
            except Exception:
                continue
        self._restore = []
        return False
