"""One test's fake machine must not become every later test's machine.

``spacr.accelerator.resolve()`` caches. That is right in production --
probing torch costs real time and the answer cannot change mid-run -- and
a trap in a test process, because a test that makes ``torch.cuda`` raise
in order to prove the CPU fallback works leaves "this machine has no GPU"
CACHED behind it. ``monkeypatch`` undoes the torch patch and knows
nothing about the cache.

That is not hypothetical. It is how
tests/qt/test_a_preview_without_torch_still_segments.py failed: the
second test passed alone and failed after the first, and the failure read
as a bug in the preview's device choice rather than a neighbour's
leftovers.

tests/conftest.py restores the cache around every test. These pin that it
does, because a protective fixture that silently stopped working would be
invisible until the next confusing failure.
"""
from __future__ import annotations

import pytest

accelerator = pytest.importorskip("spacr.accelerator")

#: Set by the poisoning test so the checking test knows a poison happened.
#: Module-level rather than a fixture so it survives BETWEEN tests, which
#: is the whole thing being measured.
_POISON = {}


def test_a_test_can_poison_the_cache_and_see_its_own_poison():
    """The poison has to actually take, or the next test proves nothing."""
    fake = accelerator.Accelerator(
        kind="cpu", device="cpu", label="POISONED", name="POISONED",
        detected=False, usable=False, note="", float64=True,
        autocast=False, fallback=True, bfloat16=False)
    accelerator._CACHED = fake
    _POISON["sentinel"] = fake

    assert accelerator.resolve() is fake, (
        "resolve() ignored the cache, so this file cannot measure the leak")


def test_the_poison_is_gone_by_the_next_test():
    """THE REGRESSION.

    Order-independent on purpose: if this runs first there is no poison
    to find and it says so rather than passing quietly for the wrong
    reason.
    """
    sentinel = _POISON.get("sentinel")
    if sentinel is None:
        pytest.skip("the poisoning test has not run yet in this order")

    assert accelerator._CACHED is not sentinel, (
        "a previous test's fake accelerator is still cached; the autouse "
        "fixture in tests/conftest.py is not restoring it, and every test "
        "after that one now sees the wrong machine")
    assert accelerator.resolve() is not sentinel


def test_the_cache_is_still_warm_in_the_next_test_too():
    """Restoring must not turn into CLEARING.

    A fixture that set the cache to None after every test would satisfy
    the leak test above while quietly making every test in the suite
    re-probe torch -- slow, and on a machine with a flaky driver,
    differently flaky.

    Two calls inside ONE test cannot tell the difference, because the
    first would re-warm the cache for the second. So the object's
    identity is carried ACROSS tests, which is the only place clearing
    and restoring look different.
    """
    warm = accelerator.resolve()
    if "warm" in _POISON:
        assert warm is _POISON["warm"], (
            "the cached accelerator was rebuilt between tests; the fixture "
            "is clearing the cache rather than restoring it")
    _POISON["warm"] = warm


def test_the_cache_is_warm_here_as_well():
    """The other half of the pair above -- one of them runs second."""
    warm = accelerator.resolve()
    if "warm" in _POISON:
        assert warm is _POISON["warm"], (
            "the cached accelerator was rebuilt between tests")
    _POISON["warm"] = warm
