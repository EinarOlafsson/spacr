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

import pytest

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
    call would be delivered after the shutdown it was meant to survive.

    ANCHORED ON THE `connect`, NOT ON THE NAME. This searched from the FIRST
    occurrence of `_stop_before_quitting` -- which is the `def`, not the
    connection -- and read 400 characters forward. The handler then grew a
    docstring explaining why it binds the worker and thread as default
    arguments, which pushed the `connect` call past the window, and the test
    failed on a file whose connection type had never changed. A window
    measured in characters from a name that appears twice is a test of how
    long the docstring is.
    """
    source = _source()
    where = source.index("aboutToQuit.connect(_stop_before_quitting")
    tail = source[where:where + 200]

    assert "Qt.DirectConnection" in tail, (
        f"the shutdown cancel is not a direct connection: {tail!r}")


def test_the_shutdown_wait_is_bounded():
    """A shutdown that hangs on a slow socket is a worse failure than the one
    being prevented."""
    assert "_t.wait(5000)" in _source()


@pytest.mark.parametrize("existing_archive", [False, True])
def test_a_cancelled_download_leaves_no_partial_archive(
        tmp_path, monkeypatch, existing_archive):
    """Mid-stream cancellation removes only this download's partial files.

    An existing complete archive and other downloads' temporary files must
    survive; the cancelled archive must never be extracted or prepared.
    """
    import sys
    import types

    destination = tmp_path / "example"
    destination.mkdir()
    retained = destination / ".spacr-archive-other-download"
    retained.mkdir()
    (retained / "other.tar.part").write_bytes(b"another download's bytes")
    (destination / "cached.tif").write_bytes(b"existing data")
    archive = destination / "example.tar"
    if existing_archive:
        archive.write_bytes(b"existing complete archive")
    original_paths = set(destination.rglob("*"))
    original_files = {path: path.read_bytes() for path in original_paths
                      if path.is_file()}

    worker = hf._TarExampleWorker(destination)
    worker.repo = "test/example"
    worker.archive = archive.name
    finished = []
    worker.finished.connect(lambda *result: finished.append(result))
    post_download = []
    monkeypatch.setattr(hf, "extract_example_archive",
                        lambda *args: post_download.append("extract"))
    monkeypatch.setattr(worker, "after_extract",
                        lambda *args: post_download.append("prepare"))
    monkeypatch.setattr(hf, "make_the_example_paths_absolute",
                        lambda *args: post_download.append("rewrite"))
    first_chunk = b"first chunk" * 8192
    cancellation_points = []

    class Response:
        headers = {"Content-Length": str(len(first_chunk) + 1)}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield first_chunk
            partials = set(destination.rglob("*.part")) - original_paths
            assert len(partials) == 1
            partial = partials.pop()
            assert partial.read_bytes() == first_chunk
            cancellation_points.append(partial)
            worker.cancel()
            yield b"x"
            raise AssertionError("the cancelled worker kept reading the stream")

    transport = types.ModuleType("requests")
    transport.get = lambda *args, **kwargs: Response()
    monkeypatch.setitem(sys.modules, "requests", transport)

    worker.run()

    assert len(cancellation_points) == 1
    assert finished == [(False, "", "", "Cancelled by user.")]
    assert post_download == []
    assert set(destination.rglob("*")) == original_paths
    assert {path: path.read_bytes() for path in original_files} == original_files
