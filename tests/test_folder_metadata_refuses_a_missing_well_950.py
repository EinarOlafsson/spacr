"""Item 950: a folder layout with a well level never guesses a missing well.

``assign_missing_fields(have_well=True)`` used to give a file with no well
folder the well A01, so its cells were counted as A01's. It now refuses,
naming the file and how each of its folder names was read. A file whose
well IS found keeps the name it had, and ``have_well=False`` still numbers
wells A01, A02, ... as before.
"""
from pathlib import Path

import pytest

from spacr.qt import folder_metadata as fm
from spacr.qt import ingest_preview as ip
from spacr.qt.dnd_handlers import scan_folder_structure


def _touch(path: Path) -> Path:
    """Make an empty image file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_a_file_without_a_well_folder_is_refused_with_its_name_and_tokens():
    """The refusal names the file and every folder name it read."""
    path = Path("/data/plate1/C01/field_02/img.tif")
    with pytest.raises(ValueError) as caught:
        fm.assign_missing_fields([path], have_well=True, have_field=True)
    message = str(caught.value)
    assert str(path) in message
    assert "'plate1' (plate)" in message
    assert "'C01' (channel)" in message
    assert "'field_02' (field)" in message
    assert "A01" not in message.replace("C01", "")


def test_a_file_with_no_recognised_folder_names_says_none():
    """With nothing recognised the tokens read ``none``, not an empty list."""
    with pytest.raises(ValueError, match=r"Folder names read: none\."):
        fm.assign_missing_fields([Path("/data/loose/img.tif")],
                                 have_well=True)


def test_one_wellless_file_refuses_the_whole_set():
    """A single stray file stops the mapping; no partial A01 rows come back."""
    files = [Path("/d/B03/F01/a.tif"), Path("/d/stray/b.tif")]
    with pytest.raises(ValueError, match=r"/d/stray/b\.tif"):
        fm.assign_missing_fields(files, have_well=True, have_field=True)


def test_a_found_well_is_unchanged():
    """Wells read from the folders keep their (upper-cased) names."""
    files = [Path("/d/b03/F02/ch2.tif"), Path("/d/A12/F01/ch1.tif")]
    rows = fm.assign_missing_fields(files, have_well=True, have_field=True)
    assert [(m.well, m.field, m.channel, m.canonical) for m in rows] == [
        ("A12", 1, 1, "plate1_A12_F001_C01.tif"),
        ("B03", 2, 2, "plate1_B03_F002_C02.tif"),
    ]


def test_have_well_false_still_numbers_wells_without_a_well_folder():
    """Without a well level the minted wells are unchanged: A01, A02, ..."""
    files = [Path(f"/d/loose/img{i}.tif") for i in range(3)]
    rows = fm.assign_missing_fields(files, have_well=False, have_field=True)
    assert [m.well for m in rows] == ["A01", "A02", "A03"]


def test_the_folder_plan_carries_the_refusal(tmp_path):
    """The planner the drop handler calls raises the same refusal."""
    _touch(tmp_path / "A01" / "F01" / "a.tif")
    stray = _touch(tmp_path / "C01" / "F01" / "b.tif")
    template = fm.FolderTemplate(depth_labels=("well", "field"),
                                 sample_paths=(), chan_from_filename=False)
    with pytest.raises(ValueError, match="No well found") as caught:
        ip.plan_folder_extraction(tmp_path, template=template)
    assert str(stray) in str(caught.value)


def test_a_dropped_folder_reports_the_refusal_instead_of_a01_rows(tmp_path):
    """The GUI's folder scan returns the refusal as its error, no rows."""
    for well in ("A01", "A02", "B01"):
        _touch(tmp_path / well / "F01" / "img.tif")
    stray = _touch(tmp_path / "C01" / "F01" / "img.tif")
    report = scan_folder_structure(tmp_path)
    assert "well" in report["labels"]
    assert report["rows"] == []
    assert "No well found" in report["error"]
    assert str(stray) in report["error"]
    assert "'C01' (channel)" in report["error"]


def test_the_console_shows_the_refusal_and_opens_no_table(tmp_path):
    """The drop report logs the refusal line and stops before the table."""
    from spacr.qt import dnd_handlers as dh

    class Console:
        """Collects the console lines the report writes."""

        def __init__(self):
            self.lines = []

        def append_stdout(self, text):
            """Keep one console line."""
            self.lines.append(text)

    class Screen:
        """The one attribute ``_log`` writes through."""

        def __init__(self):
            self._console = Console()

    _touch(tmp_path / "A01" / "F01" / "img.tif")
    stray = _touch(tmp_path / "C01" / "F01" / "img.tif")
    screen = Screen()
    dh._render_folder_structure(tmp_path, screen,
                                scan_folder_structure(tmp_path))
    text = "".join(screen._console.lines)
    assert "[drop] folder-structure preview unavailable: No well found" in text
    assert str(stray) in text
    assert "folder-structure plan" not in text
