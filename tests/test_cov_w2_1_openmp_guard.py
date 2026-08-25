"""The parts of the OpenMP guard that need a real runtime or a real dyld.

`_handle` is the one place that actually calls into a loaded OpenMP library,
and it is exercised here against a runtime this process already has mapped --
``dlopen`` on a mapped image returns the existing handle, so nothing new is
loaded. The dyld probe and the two failure paths are reached by injection,
because macOS and a broken ``realpath`` are not available here.
"""
from __future__ import annotations

import ctypes

import pytest

from spacr import openmp_guard


@pytest.fixture
def a_resident_runtime():
    """The path of an OpenMP runtime already mapped into this process."""
    import torch  # noqa: F401 - maps its bundled libgomp

    runtimes = openmp_guard.resident_openmp_runtimes()
    assert runtimes, "this interpreter has no OpenMP runtime mapped"
    return runtimes[0]


def test_a_handle_reads_the_real_thread_limit(a_resident_runtime):
    """The handle is a live CDLL whose omp symbols answer with real values."""
    handle = openmp_guard._handle(a_resident_runtime)

    assert isinstance(handle, ctypes.CDLL)
    assert handle.omp_get_max_threads() >= 1
    assert handle.omp_set_num_threads.restype is None


def test_the_same_runtime_is_opened_once(a_resident_runtime):
    """Repeated entries share one handle rather than dlopening again."""
    first = openmp_guard._handle(a_resident_runtime)
    second = openmp_guard._handle(a_resident_runtime)

    assert second is first
    assert openmp_guard._HANDLES[a_resident_runtime] is first


class _Symbol:
    """A foreign function whose restype/argtypes the caller may set."""

    def __init__(self, call):
        self._call = call
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self._call(*args)


class _Dyld:
    """A stand-in for libc's dyld image table, which only macOS has."""

    def __init__(self, names):
        self._names = names
        self._dyld_image_count = _Symbol(lambda: len(names))
        self._dyld_get_image_name = _Symbol(lambda index: names[index])


def test_the_dyld_probe_decodes_every_mapped_image(monkeypatch):
    """Every image dyld reports comes back as text, undecodable bytes and all."""
    names = [b"/usr/lib/libSystem.B.dylib",
             b"/opt/homebrew/lib/libomp.dylib",
             b"/broken/\xff\xfename.dylib"]
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: _Dyld(names))

    images = openmp_guard._macos_images()

    assert images[:2] == ["/usr/lib/libSystem.B.dylib",
                          "/opt/homebrew/lib/libomp.dylib"]
    assert images[2].startswith("/broken/")
    assert all(isinstance(image, str) for image in images)


def test_a_path_that_cannot_be_resolved_is_still_reported(monkeypatch,
                                                          tmp_path):
    """A runtime whose realpath fails is a runtime; report it unresolved."""
    lib = tmp_path / "libomp.so"
    lib.write_bytes(b"")
    monkeypatch.setattr(openmp_guard.sys, "platform", "linux")
    monkeypatch.setattr(openmp_guard, "_linux_images", lambda: [str(lib)])

    def _no_realpath(path):
        raise OSError("too many levels of symbolic links")

    monkeypatch.setattr(openmp_guard.os.path, "realpath", _no_realpath)

    assert openmp_guard.resident_openmp_runtimes() == [str(lib)]


class _Clamp:
    """A runtime handle that clamps but refuses to give the threads back."""

    def __init__(self):
        self.set_to = []

    def omp_get_max_threads(self):
        return 8

    def omp_set_num_threads(self, value):
        self.set_to.append(value)
        if value != 1:
            raise OSError("the runtime went away")


def test_a_runtime_that_will_not_restore_does_not_end_the_run(monkeypatch):
    """Leaving the region must never raise, however badly restoring goes."""
    monkeypatch.setenv("SPACR_OPENMP_GUARD", "on")
    handle = _Clamp()
    monkeypatch.setattr(openmp_guard, "_handle", lambda path: handle)
    monkeypatch.setattr(openmp_guard, "resident_openmp_runtimes",
                        lambda: ["/a/libomp.so", "/b/libgomp.so.1"])

    region = openmp_guard.single_threaded_openmp("a model")
    with region:
        assert handle.set_to == [1, 1]

    assert handle.set_to == [1, 1, 8, 8]
    assert region._restore == []
