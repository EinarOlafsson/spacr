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
    """A loop that checks its flag, not one call that cannot be stopped.

    Checked as a CALL rather than as the word: the comment above the loop
    explains why snapshot_download is not used, and a naive text search reads
    that explanation as the thing it warns against.
    """
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(hf._AnnotateExampleWorker.run)
    tree = ast.parse(textwrap.dedent(source))
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "snapshot_download" not in called
    assert "hf_hub_download" in called
    assert "if self._cancel:" in source


def test_it_checks_between_every_file():
    """Once at the end is what snapshot_download effectively did."""
    import inspect

    source = inspect.getsource(hf._AnnotateExampleWorker.run)
    body = source[source.index("for done, name in enumerate(names):"):]
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


def test_an_already_downloaded_file_is_not_fetched_again():
    """An interrupted run leaves most of the set on disk; re-fetching 280 MB
    to arrive at the same bytes is the wrong trade."""
    import inspect

    source = inspect.getsource(hf._AnnotateExampleWorker.run)
    assert "if target.is_file():" in source
    assert "continue" in source
