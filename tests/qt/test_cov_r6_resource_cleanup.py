"""The cleanup paths that only a half-present collaborator reaches.

``test_cov_r5_resource_cleanup.py`` pins the collaborators that *raise*.
What is left, and what this file drives, is the collaborator that is simply
not all there: a torch whose allocator getters take no device argument, a
CUDA context that stops answering between two readings, a cache owner that
publishes an inventory but no eviction, a cache module loaded without the
budget protocol at all, a model releaser that is registered twice, and an
``install_budget_sweep`` called before any ``QApplication`` exists.

Every test that asserts a fact is absent drives, in the same test, the input
that makes it present -- so "absent" is measured against a run that produced
it and not against a function that never ran.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr.qt import resource_cleanup as rc


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def budget_globals():
    """Restore the module's two sweep globals whatever a test does to them."""
    timer = rc._BUDGET_TIMER
    pending = rc._BUDGET_SWEEP_PENDING
    yield
    rc._BUDGET_TIMER = timer
    rc._BUDGET_SWEEP_PENDING = pending


# ---------------------------------------------------------------------------
# Reading torch's allocator without owning one
# ---------------------------------------------------------------------------

def test_a_multi_device_allocator_that_refuses_a_device_is_read_once():
    """Some torch builds expose only the no-argument form of the statistic.

    ``_cuda_stat`` takes the torch module as a parameter, so the double is
    passed in rather than installed: this test creates no CUDA context and
    imports nothing, which is the whole point of the function.
    """
    def _per_device(device=None):
        if device is None:
            raise AssertionError("the summing path must name a device")
        return (device + 1) * 1024

    torch = SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 2, memory_reserved=_per_device))

    # Two devices, a getter that takes one: both are summed.
    assert rc._cuda_stat(torch, "memory_reserved") == 1024 + 2048

    def _no_argument(*args):
        if args:
            raise TypeError("memory_reserved() takes 0 positional arguments")
        return 4096

    # The same two devices, a getter that will not take one: the current
    # device is measured rather than the read being abandoned.
    torch.cuda.memory_reserved = _no_argument
    assert rc._cuda_stat(torch, "memory_reserved") == 4096


def test_an_allocator_that_stops_answering_reports_no_cached_bytes(
        monkeypatch):
    """``None`` means "could not ask"; ``0`` would mean "nothing to free".

    ``_cuda_cached`` is the reading behind the sweep's allocator accounting.
    It is called here directly so the working reading and the broken one can
    be compared against the same torch double.
    """
    torch = _module("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        device_count=lambda: 1,
        memory_reserved=lambda: 10 * 1024 * 1024,
        memory_allocated=lambda: 4 * 1024 * 1024,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert rc._cuda_cached() == 6 * 1024 * 1024

    def _context_gone():
        raise RuntimeError("the CUDA context was destroyed")

    # The reserved reading still works; only the second one fails, which is
    # exactly the window the reserved-minus-allocated subtraction opens.
    torch.cuda.memory_allocated = _context_gone
    assert rc._cuda_cached() is None


# ---------------------------------------------------------------------------
# Owners that answer only half of the cache-budget protocol
# ---------------------------------------------------------------------------

def test_an_owner_that_cannot_evict_is_named_rather_than_inventoried():
    """An inventory with no eviction would report megabytes nothing can free.

    Listing such an owner's entries would make every sweep look incomplete
    for ever, so the owner is reported as broken and its entries are skipped.
    """
    class _Halfway:
        __name__ = "fake.halfway"

        def cache_budget_entries(self):
            return [("k", 4 * 1024 * 1024, 0.0, False)]

    records, errors = rc._collect_budget_entries([_Halfway()])

    assert records == []
    assert errors == ["fake.halfway: cache-budget protocol is incomplete"]

    # The same owner with the missing half supplied: its entry is collected,
    # so the empty list above is the missing method and not an empty cache.
    whole = _Halfway()
    whole.drop_cache_budget_entry = lambda key: True
    records, errors = rc._collect_budget_entries([whole])

    assert errors == []
    assert [row.label for row in records] == ["fake.halfway['k']"]
    assert records[0].megabytes == pytest.approx(4.0)


def test_a_loaded_module_without_the_protocol_is_stepped_over(monkeypatch):
    """Module caches are discovered by shape, and the shape is both methods.

    A module that grew only one of them -- mid-refactor, or an older plugin --
    must be skipped rather than appended and then failed on eviction.
    """
    engine = _module("spacr.qt.annotate_engine",
                     cache_budget_entries=lambda: [],
                     drop_cache_budget_entry=lambda key: True)
    monkeypatch.setitem(sys.modules, "spacr.qt.annotate_engine", engine)

    partial = _module("spacr.crops", cache_budget_entries=lambda: [])
    monkeypatch.setitem(sys.modules, "spacr.crops", partial)

    owners = rc._loaded_cache_owners()
    assert engine in owners
    assert partial not in owners

    # The very same module with an eviction method is discovered, so the
    # omission above is the missing method and not the module being unseen.
    partial.drop_cache_budget_entry = lambda key: True
    assert partial in rc._loaded_cache_owners()


# ---------------------------------------------------------------------------
# Model releasers
# ---------------------------------------------------------------------------

def test_a_releaser_found_twice_is_still_only_run_once(monkeypatch):
    """The registry and the loaded module can name the same function.

    Running it twice would double-count the megabytes it reports freeing,
    and the second call has nothing left to free.
    """
    monkeypatch.setattr(rc, "MODEL_RELEASERS", [])

    # Loaded, but with no releaser to offer.
    monkeypatch.setitem(sys.modules, "spacr.qt.annotate_engine",
                        _module("spacr.qt.annotate_engine"))
    assert rc._loaded_model_releasers() == ()

    calls = []

    def _release():
        calls.append(1)
        return 5

    monkeypatch.setitem(
        sys.modules, "spacr.qt.annotate_engine",
        _module("spacr.qt.annotate_engine", _release_cached_models=_release))
    assert rc._loaded_model_releasers() == (_release,)

    # Registered explicitly as well: found by both routes, listed once.
    monkeypatch.setattr(rc, "MODEL_RELEASERS", [_release])
    assert rc._loaded_model_releasers() == (_release,)

    monkeypatch.setattr(rc, "_a_run_is_active", lambda: False)
    assert rc._release_models_under_pressure() == 5
    assert calls == [1]


# ---------------------------------------------------------------------------
# Installing the periodic sweep
# ---------------------------------------------------------------------------

def test_no_application_means_no_sweep_and_no_orphan_timer(
        monkeypatch, qapp, budget_globals):
    """The timer is parented to the application, so it needs one to exist.

    ``install_budget_sweep`` runs during start-up, and on the paths where Qt
    has been imported but the application has not been built yet there is
    nothing to own the timer.  Answering ``False`` keeps the caller free to
    ask again once the application is up.
    """
    real_app = qapp
    rc._BUDGET_TIMER = None

    monkeypatch.setitem(
        sys.modules, "PySide6.QtWidgets",
        _module("PySide6.QtWidgets",
                QApplication=SimpleNamespace(instance=lambda: None)))

    assert rc.install_budget_sweep() is False
    assert rc._BUDGET_TIMER is None

    # The same call with an application present installs a running timer, so
    # the False above is the missing application and nothing else.
    monkeypatch.setitem(
        sys.modules, "PySide6.QtWidgets",
        _module("PySide6.QtWidgets",
                QApplication=SimpleNamespace(instance=lambda: real_app)))
    try:
        assert rc.install_budget_sweep() is True
        assert rc._BUDGET_TIMER is not None
        assert rc._BUDGET_TIMER.isActive()
        assert rc._BUDGET_TIMER.parent() is real_app
        assert rc._BUDGET_TIMER.objectName() == "LiveCacheBudgetSweep"
    finally:
        if rc._BUDGET_TIMER is not None:
            rc._BUDGET_TIMER.stop()
            rc._BUDGET_TIMER.deleteLater()
