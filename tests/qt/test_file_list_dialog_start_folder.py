"""Where the file-list picker opens, as answered through ``spacr.qt.path_probe``.

``FilePathListWidget.pick_files`` asks ``path_probe.isdir`` (never the
filesystem on the GUI thread) whether the remembered folder, or the folder of
the last listed file, still exists.  The probe cache is primed with
``path_probe.prime`` so each case is deterministic rather than racing a
background stat.  One case table: empty list, listed file, vanished folder,
remembered folder, and a remembered folder that has gone.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.qt import path_probe  # noqa: E402
from spacr.qt.widgets import file_list as fl  # noqa: E402

pytestmark = pytest.mark.qt


def _prime(answer, *paths):
    """Record ``answer`` for both the exists and the isdir question."""
    for path in paths:
        path_probe.prime(str(path), answer)
        path_probe.prime(str(path), answer, want_dir=True)


def test_the_picker_opens_where_the_user_last_was_or_beside_the_listed_file(
        subtests, qtbot, monkeypatch, tmp_path):
    """Remembered folder, then the listed file's folder, then '' -- each only if it still exists."""
    old = tmp_path / "old"
    elsewhere = tmp_path / "elsewhere"
    vanished = tmp_path / "vanished"
    for folder in (old, elsewhere):
        folder.mkdir()
    listed = old / "plate1_scores.csv"
    listed.write_text("gene,score\na,1\n", encoding="utf-8")
    gone_file = vanished / "plate1_scores.csv"
    _prime(True, listed, old, elsewhere)
    _prime(False, gone_file, vanished, tmp_path / "removed")

    starts = []
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames",
        lambda parent, title, start, filters: starts.append(start) or ([], ""))

    cases = [
        # (id, listed values, remembered folder, start, hint)
        ("an empty list asks for the platform default", [], "", "", None),
        ("a listed file reopens beside it", [str(listed)], "", str(old),
         "1 file selected"),
        ("a listed file whose folder is gone opens the default",
         [str(gone_file)], "", "",
         "1 file selected — 1 not found (shown in red)"),
        ("a remembered folder wins over the listed file", [str(listed)],
         str(elsewhere), str(elsewhere), "1 file selected"),
        ("a remembered folder that is gone falls back to the listed file",
         [str(listed)], str(tmp_path / "removed"), str(old),
         "1 file selected"),
    ]
    for case_id, values, remembered, start, hint in cases:
        with subtests.test(msg=case_id):
            widget = fl.FilePathListWidget(kind="csv")
            qtbot.addWidget(widget)
            widget.set_value(values)
            widget._last_directory = remembered
            starts.clear()
            assert widget.pick_files() == 0
            assert starts == [start]
            assert widget.get_value() == values
            assert widget._last_directory == remembered
            if hint is not None:
                assert widget._hint.text() == hint


def test_an_unknown_path_is_stat_ed_off_the_gui_thread_and_announced_once(
        subtests, qtbot, tmp_path):
    """A first ask answers the default and queues a stat; the worker caches and announces the real answer."""
    folder = tmp_path / "plate"
    folder.mkdir()
    listed = folder / "scores.csv"
    listed.write_text("gene,score\n", encoding="utf-8")
    cases = [
        # (id, path, want_dir, real answer)
        ("a folder asked as a folder", folder, True, True),
        ("a file asked as a folder", listed, True, False),
        ("a file asked as a path", listed, False, True),
        ("a missing path", tmp_path / "gone", False, False),
    ]
    for case_id, path, want_dir, answer in cases:
        with subtests.test(msg=case_id):
            text = str(path)
            path_probe.forget(text)
            with qtbot.waitSignal(
                    path_probe.probes.answered, timeout=10000,
                    check_params_cb=lambda got, _a, text=text: got == text
            ) as blocker:
                first = path_probe.exists(text, default=not answer,
                                          want_dir=want_dir)
            assert first is (not answer)
            assert blocker.args == [text, answer]
            assert path_probe.known(text, want_dir=want_dir) is answer
            assert path_probe.known(text, want_dir=not want_dir) is None
            assert path_probe.exists(text, default=not answer,
                                     want_dir=want_dir) is answer
            path_probe.forget(text)
    with subtests.test(msg="a waiting ask stats at once and caches what it saw"):
        text = str(tmp_path / "waited")
        path_probe.forget(text)
        assert path_probe.isdir(text, wait=True) is False
        assert path_probe.known(text, want_dir=True) is False
        (tmp_path / "waited").mkdir()
        assert path_probe.isdir(text, wait=True) is False
        path_probe.forget(text)
        assert path_probe.isdir(text, wait=True) is True
        path_probe.forget(text)
    with subtests.test(msg="an empty path is never probed"):
        assert (path_probe.exists(""), path_probe.exists(None),
                path_probe.known("")) == (False, False, None)
