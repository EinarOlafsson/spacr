"""The console forwarder must never lose the filter it is born with.

`_ensure_handler` installs `_NotAlreadyShownByTheRootSink` at construction.
It is not a policy and no test chooses it: it is what stops ONE record
being rendered by BOTH console sinks -- `verbose_logger`'s forwarder on the
`spacr` logger and `logging_util`'s QtLogHandler on the root.

WHY A DETECTOR AND NOT ANOTHER FIX. The symptom is
`test_a_qt_warning_reaches_the_console_once` failing in a full run, and it
took five wrong hypotheses to find the cause the first time -- the sink was
attached, the singletons were correct, the levels were right, and the
answer was `filters == []`, which nothing in the level logic could suggest.
Two paths that stripped it are fixed (this conftest's restore, and
`test_cov_w3_8_verbose_logger`'s `filters[:] = []`), and a rare ordering
still reached it afterwards: 714 passed under three seeds and the
deterministic order, and one unseeded run failed.

SO THIS TURNS AN UNREPRODUCIBLE SYMPTOM INTO A NAMED ONE. Whatever the
ordering, the run that strips the filter now fails HERE, saying what is
missing, instead of surfacing later as a duplicated console line in a test
that looks like it is about something else.

Named `zz_` so it sorts last under a deterministic order, the same
convention as `test_zz_a_reimported_module_is_put_back_properly` for
`sys.modules` and `test_zz_the_app_registry_is_left_as_it_was_found` for
the app registry.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

DEDUP = "_NotAlreadyShownByTheRootSink"


def test_the_forwarder_still_carries_its_de_duplication_filter():
    """If a forwarder exists at all, it has the filter it was built with."""
    from spacr.qt import verbose_logger as vl

    handler = vl._handler
    if handler is None:
        pytest.skip("nothing in this run built the console forwarder")

    names = [type(f).__name__ for f in handler.filters]
    assert DEDUP in names, (
        f"the console forwarder has lost {DEDUP}; every Qt log record will "
        f"now be rendered twice, once by this forwarder and once by the "
        f"root QtLogHandler. Filters present: {names or '[]'}. Something in "
        f"this run cleared the handler's filters instead of only its "
        f"LevelSetFilter gate -- see tests/qt/conftest.py's "
        f"_restore_console_level_policy, which used to do exactly that.")


def test_a_fresh_forwarder_is_born_with_it():
    """The control: the construction path installs it in the first place.

    Without this, the test above would pass vacuously on any run where the
    forwarder is never built, and would never catch the filter being
    dropped from `_ensure_handler` itself.
    """
    from spacr.qt import verbose_logger as vl

    saved = vl._handler
    sink_logger = None
    try:
        import logging

        sink_logger = logging.getLogger(vl._SINK_LOGGER)
        before = list(sink_logger.handlers)
        vl._handler = None
        fresh = vl._ensure_handler()
        assert DEDUP in [type(f).__name__ for f in fresh.filters], (
            "_ensure_handler no longer installs the de-duplication filter")
    finally:
        if sink_logger is not None:
            for extra in list(sink_logger.handlers):
                if extra not in before:
                    sink_logger.removeHandler(extra)
        vl._handler = saved
