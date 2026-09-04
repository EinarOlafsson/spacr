"""No path check on the GUI thread. Measured, because it froze the app.

THE DEFECT, 2026-09-04. `FilePathListWidget` asked `os.path.exists` for
every remembered path, on the GUI thread, to colour the missing ones red.
The maintainer's `map_barcodes` remembered
``/nas_mnt/data/sequencing/seq_3``, and `/nas_mnt` is an ``autofs`` mount
with ``timeout=600``. Measured on that machine: a single
``os.path.exists`` on that path had NOT RETURNED AFTER TWENTY SECONDS,
because the stat is what triggers the automount and the share was asleep.

It was reported as four separate defects, and it is one:

  * "opening map barcodes crashes spacr" -- a freeze, not a crash, which
    is why no traceback ever reached the logs. The system journal shows
    ``automount request ... triggered by spacr`` immediately before each
    force-quit.
  * "several events are happening upon hover and they sometimes lag for a
    couple of seconds ... the blue color flickers" -- hover events queue
    while the thread is stalled, then replay in a burst.
  * "i see millisecond glimmers of parts of the module screens on the home
    screen" -- deferred repaints.
  * "it seems to happen ... after i have opened one or two moduals" --
    every module adds its own remembered paths to stat.

MEASURED AFTER: the same call answers in 0.6 ms.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import path_probe


@pytest.fixture(autouse=True)
def _fresh_cache():
    path_probe.forget()
    yield
    path_probe.forget()


def test_an_unknown_path_answers_immediately(tmp_path):
    """The property that matters: the caller is never made to wait.

    A path nothing has asked about yet is the exact case that blocked --
    there was nothing cached, so the widget stat-ed it inline.
    """
    started = time.monotonic()
    answer = path_probe.exists(tmp_path / "never-asked-about")
    elapsed = time.monotonic() - started

    assert elapsed < 0.05, (
        f"the first question about a path took {elapsed*1000:.0f} ms -- "
        "something is stat-ing on the calling thread")
    assert answer is True, "an unknown path should be reported present"


def test_a_path_that_never_answers_does_not_block(monkeypatch):
    """The NAS case, with the slowness made explicit rather than hoped for."""
    def never(_path):
        time.sleep(30)
        return True

    monkeypatch.setattr(path_probe.os.path, "exists", never)
    started = time.monotonic()
    assert path_probe.exists("/somewhere/asleep") is True
    assert time.monotonic() - started < 0.05

    # And it stays non-blocking on the next ask, rather than joining the
    # probe that is still parked.
    started = time.monotonic()
    path_probe.exists("/somewhere/asleep")
    assert time.monotonic() - started < 0.05


def test_the_truth_arrives_and_is_cached(tmp_path):
    """Optimism is temporary. A missing path is corrected once known."""
    missing = tmp_path / "not-here"
    present = tmp_path / "here"
    present.write_text("x")

    assert path_probe.exists(missing) is True          # optimistic
    assert path_probe.exists(present) is True

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path_probe.known(missing) is not None:
            break
        time.sleep(0.02)

    assert path_probe.known(missing) is False, "the probe never answered"
    assert path_probe.exists(missing) is False
    assert path_probe.exists(present) is True


def test_isdir_is_cached_apart_from_exists(tmp_path):
    """A file exists and is not a directory; one answer must not be the other."""
    a_file = tmp_path / "f"
    a_file.write_text("x")
    path_probe.exists(a_file)
    path_probe.isdir(a_file)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if (path_probe.known(a_file) is not None
                and path_probe.known(a_file, want_dir=True) is not None):
            break
        time.sleep(0.02)

    assert path_probe.known(a_file) is True
    assert path_probe.known(a_file, want_dir=True) is False


def test_the_file_list_populates_without_a_filesystem_wait(qtbot, monkeypatch):
    """The widget itself, with every stat made slow.

    The regression this pins is not "the widget is fast" -- it is that the
    widget does not CALL the filesystem while building itself.
    """
    from spacr.qt.widgets.file_list import FilePathListWidget

    def never(_path):
        time.sleep(30)
        return True

    monkeypatch.setattr(path_probe.os.path, "exists", never)
    monkeypatch.setattr(path_probe.os.path, "isdir", never)

    widget = FilePathListWidget()
    qtbot.addWidget(widget)
    started = time.monotonic()
    widget.set_value(["/slow/one", "/slow/two", "/slow/three"])
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, (
        f"populating the list took {elapsed:.1f} s with slow paths -- it is "
        "stat-ing on the GUI thread again")
    assert widget._list.count() == 3


def test_the_hint_corrects_itself_once_the_probes_land(qtbot, tmp_path):
    """The other half of optimism: the red count appears when it is known."""
    from spacr.qt.widgets.file_list import FilePathListWidget

    present = tmp_path / "here"
    present.write_text("x")

    widget = FilePathListWidget()
    qtbot.addWidget(widget)
    widget.set_value([str(present), str(tmp_path / "gone")])

    qtbot.waitUntil(lambda: "not found" in widget._hint.text(), timeout=5000)
    assert "1 not found" in widget._hint.text()
