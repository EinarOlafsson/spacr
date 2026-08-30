"""The verbose function trace's guards, which are all about cost and safety.

Instruction 294's rule is that verbose logging must be cheap before it can be
the default, and ``_trace_one_event`` is the hook that runs on EVERY call and
return in the process. Every guard here is either a cost being avoided or the
promise its own comment makes -- "a tracing aid must never alter the code it
observes".

The cheapest guard comes first and is the one that matters most: with tracing
off, the whole hook is one ``isEnabledFor`` check.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tracing_on():
    """Turn the trace logger up, and put it back afterwards."""
    from spacr import logging_util

    logger = logging.getLogger("spacr.trace")
    previous_level = logger.level
    previous_disabled = logger.disabled
    logger.disabled = False
    logger.setLevel(logging.DEBUG)
    handler = logging.Handler()
    records = []
    handler.emit = records.append
    logger.addHandler(handler)
    logging_util._TRACE_STATE.busy = False
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled
        logging_util._TRACE_STATE.busy = False


def test_with_tracing_off_the_hook_returns_immediately():
    """The first guard, which is the whole performance argument.

    The comment above it says it is "one dictionary lookup while verbose is
    off". Anything after it -- a realpath, a qualname, a format -- would run on
    every call in the process.
    """
    from spacr import logging_util

    logger = logging.getLogger("spacr.trace")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        assert logging_util._trace_one_event(sys._getframe(), "call") is None
    finally:
        logger.setLevel(previous)


def test_a_frame_from_outside_spacr_is_not_traced(tracing_on):
    """The ``_TRACE_ROOT`` guard.

    This test's own frame is outside the package, which is the ordinary case:
    the trace is for spaCR's code, and logging every frame in numpy, torch and
    pytest would bury it.
    """
    from spacr import logging_util

    logging_util._trace_one_event(sys._getframe(), "call")

    assert not tracing_on, "a frame outside spacr must produce no record"


def test_a_frame_from_the_trace_module_itself_is_not_traced(tracing_on):
    """The ``_TRACE_THIS_FILE`` guard, which is what stops the recursion.

    Without it the logging call inside the hook would itself be traced, and
    the trace would trace its own tracing until the stack ran out.
    """
    from spacr import logging_util

    frame = _a_frame_pretending_to_be(logging_util._TRACE_THIS_FILE)
    logging_util._trace_one_event(frame, "call")

    assert not tracing_on


def test_a_skipped_module_is_not_traced(tracing_on):
    """The ``_TRACE_SKIP_MODULES`` guard, checked before the realpath.

    Those three modules are the animated backdrop and the fractals: they run
    per frame, so tracing them is the case where the trace costs more than the
    program.
    """
    from spacr import logging_util

    assert logging_util._TRACE_SKIP_MODULES
    frame = _a_frame_from_module(logging_util._TRACE_SKIP_MODULES[0])
    assert Path(frame.f_code.co_filename).is_file()

    logging_util._trace_one_event(frame, "call")

    assert not tracing_on


def test_a_re_entrant_event_is_dropped_and_the_flag_is_cleared(tracing_on):
    """The busy flag, and that the ``finally`` always clears it.

    If the flag leaked, the FIRST traced call would silence every call after
    it -- a trace that goes quiet after one line, which reads as the program
    having stopped.
    """
    from spacr import logging_util

    logging_util._TRACE_STATE.busy = True
    frame = _a_frame_from_module("spacr.measure")
    logging_util._trace_one_event(frame, "call")
    assert not tracing_on, "a re-entrant event must be dropped"

    logging_util._TRACE_STATE.busy = False
    logging_util._trace_one_event(frame, "call")
    assert logging_util._TRACE_STATE.busy is False, (
        "the finally must clear the flag even on the traced path")


# ---------------------------------------------------------------------------
# helpers: frames that claim a chosen file and module
# ---------------------------------------------------------------------------

def _a_frame_from_module(module_name, *, inside_the_package=True):
    """A real frame whose ``__name__`` is ``module_name``.

    The code's FILENAME matters as much as the module name -- the hook checks
    both -- so by default the frame claims a file under the package root, which
    is what a genuine spaCR frame looks like.
    """
    from spacr import logging_util

    source = "def f():\n    import sys\n    return sys._getframe()\n"
    parts = module_name.split(".")
    filename = (str(Path(logging_util._TRACE_ROOT, *parts[1:]).with_suffix(".py"))
                if inside_the_package else "<outside-the-package>")
    namespace = {"__name__": module_name}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["f"]()


def _a_frame_pretending_to_be(path):
    """A real frame whose code filename is ``path``."""
    source = "def f():\n    import sys\n    return sys._getframe()\n"
    namespace = {"__name__": "spacr.measure"}
    exec(compile(source, path, "exec"), namespace)
    return namespace["f"]()
