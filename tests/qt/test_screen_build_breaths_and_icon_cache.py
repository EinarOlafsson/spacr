"""Two pieces of window-open housekeeping: the breaths a screen takes while it is
built, and the atomic write of a re-inked icon into the on-disk cache.

A breath is one (sometimes two) event-loop passes with user input held back,
taken only when the step before ran long enough or the caller forces it, and
named on the timing timeline after the step that asked for it. An icon write
either replaces the cached PNG whole or leaves the previous one exactly as it
was, with no ``.part`` temporary left behind unless it cannot be removed.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEventLoop  # noqa: E402

from spacr.qt import iconset, screens, timing  # noqa: E402

pytestmark = pytest.mark.qt

EXCLUDE_INPUT = QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents


def _a_screen_step(force=False):
    """Stand-in for a heavy construction step that asks for a breath."""
    screens._breathe_while_a_window_opens(force=force)


def test_breathe_while_a_window_opens_takes_due_breaths_and_names_them(
        subtests, qapp, monkeypatch):
    """Breathe only when due or forced; pass twice after a long pass; mark only with timing on."""
    def _run(*, timing_on, last_breath_ago, force, second_pass_after=None,
             opening=True):
        marks, passes = [], []
        with monkeypatch.context() as patch:
            patch.setattr(timing, "mark", lambda name, detail="": marks.append(
                (name, detail)))
            patch.setattr(timing, "ENABLED", timing_on)
            patch.setattr(QCoreApplication, "processEvents",
                          staticmethod(lambda *a: passes.append(a)))
            if second_pass_after is not None:
                patch.setattr(screens, "_SECOND_PASS_AFTER_S",
                              second_pass_after)
            screens._start_breathing_while_a_window_opens(
                time.perf_counter() - last_breath_ago)
            if not opening:
                screens._stop_breathing_while_a_window_opens()
            before = time.perf_counter()
            try:
                _a_screen_step(force=force)
            finally:
                screens._stop_breathing_while_a_window_opens()
            took_breath = screens._last_breath_at >= before
        return marks, passes, took_breath

    with subtests.test(msg="a due breath is named after the step before it"):
        marks, passes, took = _run(timing_on=True, last_breath_ago=1.0,
                                   force=False)
        assert passes == [(EXCLUDE_INPUT,)]
        assert took is True
        [(name, detail)] = marks
        assert name == "breath"
        found = re.fullmatch(
            r".*tests/qt/test_screen_build_breaths_and_icon_cache\.py:\d+ "
            r"_a_screen_step worked (\d+) ms, breathed (\d+) ms", detail)
        assert found is not None, detail
        assert int(found.group(1)) >= 1000

    with subtests.test(msg="timing off still breathes but records nothing"):
        marks, passes, took = _run(timing_on=False, last_breath_ago=1.0,
                                   force=False)
        assert (marks, passes, took) == ([], [(EXCLUDE_INPUT,)], True)

    with subtests.test(msg="a short step takes no breath"):
        marks, passes, took = _run(timing_on=True, last_breath_ago=0.0,
                                   force=False)
        assert (marks, passes, took) == ([], [], False)

    with subtests.test(msg="a short step breathes when forced"):
        marks, passes, took = _run(timing_on=True, last_breath_ago=0.0,
                                   force=True)
        assert passes == [(EXCLUDE_INPUT,)]
        assert [name for name, _ in marks] == ["breath"]
        assert took is True

    with subtests.test(msg="a long first pass gets a second"):
        marks, passes, took = _run(timing_on=False, last_breath_ago=0.0,
                                   force=True, second_pass_after=-1.0)
        assert (marks, passes, took) == (
            [], [(EXCLUDE_INPUT,), (EXCLUDE_INPUT,)], True)

    with subtests.test(msg="outside a window open nothing happens"):
        marks, passes, took = _run(timing_on=True, last_breath_ago=1.0,
                                   force=True, opening=False)
        assert (marks, passes, took) == ([], [], False)


def _icon(red=0):
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    array[..., 3] = 255
    array[..., 0] = red
    return array


def test_write_cached_icon_replaces_whole_or_keeps_the_previous_icon(
        subtests, tmp_path, monkeypatch):
    """A write lands whole; a failed replace keeps the old PNG and cleans its .part."""
    def _temporary(path):
        return path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.part")

    with subtests.test(msg="a first write creates the folder and the icon"):
        path = tmp_path / "fresh" / "nested" / "one.png"
        assert iconset._write_cached_icon(path, _icon(7)) is None
        np.testing.assert_array_equal(iconset._read_cached_icon(path), _icon(7))
        assert sorted(p.name for p in path.parent.iterdir()) == ["one.png"]

    with subtests.test(msg="a second write replaces the first"):
        assert iconset._write_cached_icon(path, _icon(9)) is None
        np.testing.assert_array_equal(iconset._read_cached_icon(path), _icon(9))
        assert sorted(p.name for p in path.parent.iterdir()) == ["one.png"]

    with subtests.test(msg="a failed replace removes its temporary"):
        path = tmp_path / "shared" / "one.png"
        iconset._write_cached_icon(path, _icon())
        saved = path.read_bytes()
        attempted = []

        def sharing_violation(source, destination):
            attempted.append((Path(source), Path(destination),
                              Path(source).is_file()))
            raise PermissionError("another writer has the cached icon open")

        with monkeypatch.context() as patch:
            patch.setattr(iconset.os, "replace", sharing_violation)
            assert iconset._write_cached_icon(path, _icon(200)) is None
        assert attempted == [(_temporary(path), path, True)]
        assert path.read_bytes() == saved
        assert sorted(p.name for p in path.parent.iterdir()) == ["one.png"]

    with subtests.test(msg="a temporary that cannot be removed is left, the icon kept"):
        path = tmp_path / "locked" / "one.png"
        iconset._write_cached_icon(path, _icon())
        saved = path.read_bytes()
        temporary = _temporary(path)
        replacements, removals = [], []
        original_unlink = Path.unlink

        def sharing_violation(source, destination):
            replacements.append((source, destination))
            raise PermissionError("another writer has the cached icon open")

        def cleanup_denied(candidate, *args, **kwargs):
            if candidate == temporary:
                removals.append(candidate)
                raise PermissionError("the temporary icon cannot be removed")
            return original_unlink(candidate, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(iconset, "os", SimpleNamespace(
                getpid=os.getpid, replace=sharing_violation))
            patch.setattr(Path, "unlink", cleanup_denied)
            assert iconset._write_cached_icon(path, _icon(200)) is None
        assert replacements == [(temporary, path)]
        assert removals == [temporary]
        assert path.read_bytes() == saved
        np.testing.assert_array_equal(iconset._read_cached_icon(path), _icon())
        np.testing.assert_array_equal(iconset._read_cached_icon(temporary),
                                      _icon(200))
        temporary.unlink()

    with subtests.test(msg="an array that cannot be encoded writes nothing"):
        path = tmp_path / "bad" / "one.png"
        assert iconset._write_cached_icon(path, np.zeros((4, 4), np.uint8)) \
            is None
        assert list(path.parent.iterdir()) == []
