"""No test's accelerator verdict may reach another test, in either direction.

``spacr.accelerator.resolve()`` caches. That is right in production -- the
answer cannot change mid-run -- and a trap in a test process, both ways:

* OUT of a test: one that makes ``torch.cuda`` raise to prove the CPU
  fallback leaves "this machine has no GPU" CACHED behind it, and
  ``monkeypatch`` undoes the torch patch without knowing about the cache.
  That is how tests/qt/test_a_preview_without_torch_still_segments.py
  failed.
* INTO a test: on a GPU workstation a ``cuda:0`` cached earlier keeps
  answering after a test patches ``torch.cuda.is_available`` to False.
  tests/test_cov_object_organelle_sam.py then read ``'cuda:0' == 'cpu'``
  and ``torch.load(map_location=cuda:0)`` refused to deserialize. CI has no
  GPU, so CI never saw it; the pair below reproduces it WITHOUT a GPU by
  caching a fake ``cuda:0``.

tests/conftest.py clears the cache before and after every test. These pin
that it does, because a protective fixture that silently stopped working
would be invisible until the next confusing failure.
"""
from __future__ import annotations

import sys

import pytest

accelerator = pytest.importorskip("spacr.accelerator")

#: Set by the poisoning tests so the checking tests know a poison happened.
#: Module-level rather than a fixture so it survives BETWEEN tests, which
#: is the whole thing being measured.
_POISON = {}


def _fake(kind, device, label):
    """An Accelerator no probe on any real machine would build."""
    return accelerator.Accelerator(
        kind=kind, device=device, label=label, name=label,
        detected=True, usable=True, note="", float64=True,
        autocast=True, fallback=False, bfloat16=False)


def test_a_test_can_poison_the_cache_and_see_its_own_poison():
    """The poison has to actually take, or the next test proves nothing."""
    fake = _fake("cpu", "cpu", "POISONED")
    accelerator._CACHED = fake
    _POISON["sentinel"] = fake

    assert accelerator.resolve() is fake, (
        "resolve() ignored the cache, so this file cannot measure the leak")


def test_the_poison_is_gone_by_the_next_test():
    """THE REGRESSION, outward direction.

    Order-independent on purpose: if this runs first there is no poison
    to find and it says so rather than passing quietly for the wrong
    reason.
    """
    sentinel = _POISON.get("sentinel")
    if sentinel is None:
        pytest.skip("the poisoning test has not run yet in this order")

    assert accelerator._CACHED is None, (
        "a previous test's accelerator is still cached; the autouse "
        "fixture in tests/conftest.py is not clearing it, and every test "
        "after that one now sees the wrong machine")
    assert accelerator.resolve() is not sentinel


def test_a_gpu_machine_caches_cuda():
    """Stand in for any GPU-machine test that resolved the real ``cuda:0``."""
    gpu = _fake("cuda", "cuda:0", "FAKE CUDA GPU")
    accelerator._CACHED = gpu
    _POISON["cuda"] = gpu

    assert accelerator.device_string() == "cuda:0"


def test_a_test_that_hides_cuda_is_not_answered_by_an_earlier_gpu(
        monkeypatch):
    """THE REGRESSION, inward direction -- the GPU-workstation failure.

    The shape of tests/test_cov_object_organelle_sam.py's ``_force_cpu``:
    hide CUDA, then let product code ask the resolver. A cached ``cuda:0``
    from the test before must not answer for this machine.
    """
    gpu = _POISON.get("cuda")
    if gpu is None:
        pytest.skip("the cuda-caching test has not run yet in this order")
    torch = pytest.importorskip("torch")
    monkeypatch.delenv(accelerator.ENV_DEVICE, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    found = accelerator.resolve()

    assert found is not gpu, (
        "an earlier test's cuda:0 answered for a test that hid CUDA; the "
        "autouse fixture in tests/conftest.py is restoring a verdict "
        "instead of clearing it")
    assert found.kind not in ("cuda", "rocm")
    assert found.device != "cuda:0"


def test_the_cache_still_caches_inside_one_test():
    """Clearing between tests must not turn into not caching at all."""
    first = accelerator.resolve()

    assert accelerator.resolve() is first
    assert accelerator._CACHED is first


def test_clearing_does_not_import_the_accelerator(monkeypatch):
    """A test that never touched spaCR's accelerator must not load torch.

    The fixture looks in ``sys.modules`` instead of importing, so a
    process with no accelerator loaded stays that way.
    """
    from tests.conftest import _forget_the_cached_accelerator

    monkeypatch.delitem(sys.modules, "spacr.accelerator")

    _forget_the_cached_accelerator()

    assert "spacr.accelerator" not in sys.modules
