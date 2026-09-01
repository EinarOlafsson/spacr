"""A download in flight must not outlive the application.

Reported 2026-09-01, closing spaCR while the annotation example was
downloading::

    Qt fatal: QThread: Destroyed while thread '' is still running
    Aborted (core dumped)

TWO CAUSES, both mine, both fixed:

* the worker used ``snapshot_download``, which returns when it is done and not
  before -- so ``cancel()`` set a flag nothing would look at until the whole
  280 MB had arrived, and the dialog's Cancel button did nothing;
* nothing cancelled on shutdown. The handler that quits and waits for the
  thread runs only when the worker EMITS finished, and a worker still
  downloading never does.
"""
from __future__ import annotations

from pathlib import Path

import spacr.qt.hf_download as hf


def _source() -> str:
    return Path(hf.__file__).read_text(encoding="utf-8")


def test_the_download_is_interruptible():
    """A stream that checks its flag between chunks, not one call that cannot
    be stopped.

    Checked as a CALL rather than as the word: the comment above the loop
    explains why snapshot_download is not used, and a naive text search reads
    that explanation as the thing it warns against.
    """
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(hf._TarExampleWorker.run)
    tree = ast.parse(textwrap.dedent(source))
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "snapshot_download" not in called
    assert "iter_content" in called
    assert "if self._cancel:" in source


def test_it_checks_between_every_chunk():
    """Once at the end is what snapshot_download effectively did. A megabyte
    is the granularity now, not 280 of them."""
    import inspect

    source = inspect.getsource(hf._TarExampleWorker.run)
    body = source[source.index("for chunk in response.iter_content"):]
    assert "if self._cancel:" in body


def test_quitting_cancels_it():
    assert "aboutToQuit.connect(_stop_before_quitting" in _source()


def test_the_shutdown_cancel_is_direct():
    """The worker's event loop is blocked for the whole of run(), so a queued
    call would be delivered after the shutdown it was meant to survive."""
    source = _source()
    where = source.index("_stop_before_quitting")
    assert "Qt.DirectConnection" in source[where:where + 400]


def test_the_shutdown_wait_is_bounded():
    """A shutdown that hangs on a slow socket is a worse failure than the one
    being prevented."""
    assert "_t.wait(5000)" in _source()


def test_a_cancelled_download_leaves_no_partial_archive():
    """A .part left behind would be picked up as a finished archive by a
    later run and unpacked short."""
    import inspect

    source = inspect.getsource(hf._TarExampleWorker.run)
    body = source[source.index("if self._cancel:"):]
    assert "part.unlink" in body
