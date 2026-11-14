"""The import workbench: which dropped files it holds, and the regex roles.

:func:`spacr.qt.widgets.import_workbench.images_under` walks what was dropped
and keeps image files up to a preview limit;
:class:`~spacr.qt.widgets.import_workbench.ImportWorkbench` holds them, proposes
a filename pattern and maps its named groups to roles.  Nothing here writes or
moves a file.  Each test walks a case table and asserts the exact files,
roles or evidence sentence.
"""
from __future__ import annotations

import logging
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QApplication, QFileDialog              # noqa: E402

from spacr.qt.widgets import import_workbench as iw                  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def plate_folder(tmp_path):
    """A folder of eight Yokogawa-ish names plus one file that is not an image."""
    folder = tmp_path / "plate1"
    folder.mkdir()
    for well in ("A01", "A02"):
        for field in (1, 2):
            for channel in (1, 2):
                (folder / f"{well}_T0001F{field:03d}L01A01Z01C{channel:02d}.tif"
                 ).write_bytes(b"II*\x00")
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    return folder


def _tifs(folder):
    """The folder's image files, sorted."""
    return sorted(str(p) for p in folder.glob("*.tif"))


def _settled(bench):
    """Wait for the background walk ``add_files`` started; return the count."""
    deadline = time.monotonic() + 10.0
    while bench.is_scanning() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()
    return len(bench.files())


def test_images_under_keeps_image_files_up_to_the_limit(plate_folder,
                                                        tmp_path, subtests):
    """``images_under`` over folders, named files, nothing, and a limit."""
    tifs = _tifs(plate_folder)
    note = tmp_path / "notes.txt"
    note.write_text("x", encoding="utf-8")
    image = tmp_path / "a.tif"
    image.write_bytes(b"")
    cases = [
        ("None", None, {}, []),
        ("an empty list", [], {}, []),
        ("a folder", [str(plate_folder)], {}, tifs),
        ("a folder with a limit", [str(plate_folder)], {"limit": 3}, tifs[:3]),
        ("named files with a limit", tifs, {"limit": 2}, tifs[:2]),
        ("a text file and an image", [str(note), str(image)], {},
         [str(image)]),
    ]
    for case, dropped, kwargs, expected in cases:
        with subtests.test(msg=case):
            assert iw.images_under(dropped, **kwargs) == expected


def test_the_workbench_holds_each_file_once(qtbot, plate_folder, monkeypatch,
                                            subtests):
    """``add_files`` / ``ask_for_files``: repeats and a cancelled dialog add
    nothing, a dialog's answer lands in the table."""
    tifs = _tifs(plate_folder)
    for case, start, dialog, dropped, expected in [
            ("a second drop of the same folder", tifs, None,
             [str(plate_folder)], tifs),
            ("a cancelled dialog", tifs, [], None, tifs),
            ("the dialog's files", None, tifs[:3], None, tifs[:3]),
    ]:
        with subtests.test(msg=case):
            bench = iw.ImportWorkbench(start) if start else iw.ImportWorkbench()
            qtbot.addWidget(bench)
            if dialog is not None:
                monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                                    staticmethod(lambda *a, _d=dialog, **k:
                                                 (_d, "")))
                bench.ask_for_files()
            else:
                bench.add_files(dropped)
            assert _settled(bench) == len(expected)
            assert bench.files() == expected


def test_proposing_a_pattern_explains_every_outcome(qtbot, plate_folder,
                                                    monkeypatch, caplog,
                                                    subtests):
    """``propose_from_the_names``: nothing dropped, no fitting pattern and a
    detector that raises each end in an exact sentence."""
    from spacr.qt import regex_detect

    with subtests.test(msg="nothing dropped"):
        empty = iw.ImportWorkbench()
        qtbot.addWidget(empty)
        assert empty.propose_from_the_names() == ""
        assert empty.evidence.text() == "Nothing to work from yet."

    bench = iw.ImportWorkbench(_tifs(plate_folder))
    qtbot.addWidget(bench)

    with subtests.test(msg="no pattern fits"):
        monkeypatch.setattr(regex_detect, "auto_detect_regex",
                            lambda names: ("", "nothing", 0))
        assert bench.propose_from_the_names() == ""
        assert bench.evidence.text() == (
            "No pattern fits these names. Type one, and the table below "
            "will show what it does as you go.")

    with subtests.test(msg="the detector raises"):
        def explode(names):
            raise ValueError("the names do not align")

        monkeypatch.setattr(regex_detect, "auto_detect_regex", explode)
        with caplog.at_level(logging.DEBUG, logger="spacr.qt.import_workbench"):
            assert bench.propose_from_the_names() == ""
        assert bench.evidence.text() == (
            "Could not work one out: the names do not align")
        assert "could not propose a regex" in caplog.text


def test_named_groups_map_to_roles_that_survive_regex_edits(
        qtbot, plate_folder, subtests):
    """``roles``: groups named after a role default to it, a remembered role
    survives an unrefreshed edit, and a pattern without groups has none."""
    tifs = _tifs(plate_folder)

    with subtests.test(msg="groups named after roles"):
        bench = iw.ImportWorkbench(
            tifs, regex=r"(?P<wellID>[A-Z]\d+)_T\d+F(?P<fieldID>\d+).*"
                        r"C(?P<chanID>\d+)")
        qtbot.addWidget(bench)
        roles = bench.roles()
        assert roles["wellID"] == "wellID"
        assert roles["fieldID"] == "fieldID"

    with subtests.test(msg="a remembered role for a group not shown"):
        bench = iw.ImportWorkbench(tifs, regex=r"(?P<wellID>[A-Z]\d+).*")
        qtbot.addWidget(bench)
        bench._set_role("plateID", "plateID")
        bench.regex.blockSignals(True)
        bench.regex.setText(r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+).*")
        bench.regex.blockSignals(False)
        assert bench.roles() == {"plateID": "plateID", "wellID": "wellID"}

    with subtests.test(msg="a pattern that stops capturing"):
        bench = iw.ImportWorkbench(tifs, regex=r"(?P<wellID>[A-Z]\d+).*")
        qtbot.addWidget(bench)
        assert bench.roles() == {"wellID": "wellID"}
        assert bench.roles_holder.isVisibleTo(bench)
        bench.regex.setText(r".*\.tif")
        assert bench.roles() == {}
        assert not bench.roles_holder.isVisibleTo(bench)
