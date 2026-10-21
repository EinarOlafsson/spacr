"""GitHub #156: Mask on a folder with no images does not call settings/ a plate.

The reporter's src held no images. Every run writes ``settings/`` into src
before preprocessing, so the refusal listed that folder as a sub-folder and
suggested "if those are plates, point src at one of them". The message now
says the folder holds no image files, that ``settings/`` is spaCR's own, names
any files it cannot read, and says where src has to point.
"""
from __future__ import annotations


def _message(folder):
    from spacr.io import _no_stacks_error

    return str(_no_stacks_error(str(folder), str(folder), "regex", "cellvoyager"))


def test_a_folder_holding_only_spacrs_settings_folder(tmp_path):
    from spacr.utils import save_settings

    save_settings({"src": str(tmp_path)}, name="mask")
    message = _message(tmp_path)
    assert "if those are plates" not in message
    assert "holds no image files at all" in message
    assert "(settings) are spaCR's own output, not plates" in message
    assert "directly holds the raw image files of one plate" in message


def test_files_spacr_cannot_read_are_named(tmp_path):
    (tmp_path / "settings").mkdir()
    (tmp_path / "field1.ims").write_bytes(b"")
    message = _message(tmp_path)
    assert "1 other file(s) (field1.ims)" in message
    assert ".tif" in message


def test_real_plate_folders_are_still_offered_without_spacrs_own(tmp_path):
    for name in ("settings", "plate1", "plate2"):
        (tmp_path / name).mkdir()
    message = _message(tmp_path)
    assert "sub-folders (plate1, plate2)" in message
    assert "if those are plates" in message
