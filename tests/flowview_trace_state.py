"""Snapshot and restore ``spacr.flowview.trace``'s process-global collector.

``spacr.flowview.trace`` keeps ONE collector for the whole process::

    _collector = _new_collector()          # spacr/flowview/trace.py:55

    def enable(collector=None):            # spacr/flowview/trace.py:59
        global _collector, _enabled
        ...
            _collector = collector         # process-wide, and permanent

    def disable() -> None:                 # spacr/flowview/trace.py:70
        global _enabled
        ...
        _enabled = False                   # does NOT put the collector back

That is right for the application -- a run installs its collector once and
every traced stage in the process feeds it -- and wrong for a test, because a
test that installs a stub is choosing the collector for every test that runs
after it, in every other file.

WHY ``monkeypatch`` DOES NOT COVER IT, and this is the half that was written
down backwards. The ledger note for this item says the leak is a ``global``
assignment that monkeypatch cannot observe, "so the files that patch the
FUNCTION are safe". The first clause is true and the conclusion is inverted:
the ONE test that produced the measured failure patches the function.

``spacr/qt/screens/classify.py:281`` ends ``_collector_for_open_panel`` with

    return enable(collector)               # classify.py:305

so production code takes whatever ``get_collector()`` returned and LAUNDERS it
into the global. Patch the lookup function, call the screen, and the stub is
installed for the rest of the process; monkeypatch then dutifully restores the
function it patched and the global it never saw keeps the stub.

MEASURED 2026-09-14 on this tree, alphabetical order, one process::

    python -m pytest tests/flowview -q -p no:cacheprovider -p no:randomly \
        -m "not gpu"
    150 passed, 7 errors

    AttributeError: '_Live' object has no attribute 'drain'
    spacr/flowview/panel.py:468

and narrowed to a single leaking test -- everything after it that builds a
panel on ``get_collector()`` gets the stub::

    pytest "tests/flowview/test_cov_r8_classify_flowview_tails.py\
::TestTheCollectorItCollectsWith::test_a_live_graph_is_reused_rather_than_rebuilt" \
        tests/flowview/test_the_panel_is_a_box_you_can_resize.py
    3 passed, 7 errors

It is ORDER-sensitive, not load-sensitive. The note recorded 15 quiet-box runs
without a reproduction and concluded the red had been a busy machine; those
runs went through ``pytest-randomly``, which is installed here and shuffles
the file order, so what they measured was a shuffle. Pin the order and it
reproduces every time.

These helpers live here rather than in a conftest for the reason
``tests/app_registry_state.py`` gives for the same shape: a conftest covers
only the directory beneath it, and the tests that install a collector are
spread across ``tests/flowview/``, ``tests/qt/`` and plain ``tests/``. The
autouse fixture in ``tests/conftest.py`` calls these for every test in the
session; keeping the two operations in one place is what stops the snapshot
and the restore drifting apart.

Only ``trace``'s public API is used -- ``get_collector``, ``is_enabled``,
``enable``, ``disable``. Reaching into ``trace._collector`` would work and is
deliberately not done: a private global written from the test suite is the
second copy of the very coupling this file exists to undo.

AND THE GUARD NEVER IMPORTS THE TRACER ITSELF. ``spacr/flowview/__init__.py``
re-exports a function named ``export`` over its own submodule of that name, so
importing ``spacr.flowview`` at all makes ``spacr.flowview.export`` a function
as a package attribute and a module in ``sys.modules``, and

    pytest tests/qt/test_no_panel_is_a_black_slab.py \
           tests/qt/test_zz_a_reimported_module_is_put_back_properly.py \
           -p no:randomly

fails on ``test_no_spacr_submodule_is_split_from_its_package`` -- measured
2026-09-14 with this guard switched off, so it is somebody else's pre-existing
fault and not this one's. A guard that imported the tracer for every test in
the suite would have spread that failure from the sessions that use FlowView
to all of them. So :func:`flowview_trace_module` asks ``sys.modules`` and
never imports, and the one case that leaves -- a test that imports the tracer
itself, with nothing else in the session having imported it -- is closed by
:func:`give_back_an_untraced_process`.
"""
from __future__ import annotations

import importlib
import sys

#: The tracer, by name. Asked of ``sys.modules`` rather than imported.
FLOWVIEW_TRACE = "spacr.flowview.trace"


def flowview_trace_module():
    """Return the tracer if this session has already imported it, else None.

    :returns: the live ``spacr.flowview.trace`` module object, or ``None``
        when nothing in the process has imported it yet -- in which case
        there is no process-wide collector to protect and nothing to do.
    """
    return sys.modules.get(FLOWVIEW_TRACE)


def flowview_trace_snapshot(trace):
    """Return the process-wide tracing state as a restorable pair.

    :param trace: the ``spacr.flowview.trace`` module.
    :returns: ``(collector, enabled)`` -- the collector object itself, held
        by identity rather than copied, because restoring means reinstalling
        THAT object and not an equal one.
    """
    return (trace.get_collector(), trace.is_enabled())


def restore_flowview_trace_to(trace, snapshot) -> bool:
    """Put ``snapshot`` back, and report whether anything had to move.

    :param trace: the ``spacr.flowview.trace`` module.
    :param snapshot: a pair from :func:`flowview_trace_snapshot`.
    :returns: ``True`` when the state had drifted and was rewritten,
        ``False`` when the test left it exactly as it was found. A test that
        changed nothing -- almost every test in the suite -- pays two locked
        reads and no writes.

    ``enable`` is the only public way back to a chosen collector and it also
    sets the enabled flag, so the flag is put back afterwards rather than
    before.
    """
    collector, enabled = snapshot
    if trace.get_collector() is collector and trace.is_enabled() is enabled:
        return False
    trace.enable(collector)
    if not enabled:
        trace.disable()
    return True


def give_back_an_untraced_process(trace) -> None:
    """Put a tracer a single test imported back to its import-time state.

    The snapshot/restore pair above needs a BEFORE, and there is none when
    the test itself was the first thing in the process to import the tracer.
    Left alone that is a one-test hole with a session-long consequence: the
    test installs a stub, the next test's setup snapshots the STUB as its own
    baseline, and every restore after that faithfully puts the stub back.

    ``importlib.reload`` is what closes it, and it is the right tool rather
    than a blunt one. Reload re-executes the module INTO ITS EXISTING
    dictionary, so ``_collector`` and ``_enabled`` go back to exactly what
    the import statement produced -- a new empty collector and the
    ``SPACR_FLOWVIEW`` verdict -- while the module object itself is
    unchanged. Nothing is evicted from ``sys.modules`` and no package
    attribute is rewritten, so this cannot create the split-module hazard it
    exists to avoid; ``spacr/flowview/panel.py`` and the two other modules
    that did ``from .trace import get_collector`` keep a function whose
    ``__globals__`` IS that same reloaded dictionary, and so keep reading the
    collector this put back.

    :param trace: the ``spacr.flowview.trace`` module the test imported.
    """
    importlib.reload(trace)
