"""Image import and stack file IO past the Windows MAX_PATH limit.

With long paths disabled (the Windows default) a Leica Matrix Screener tree
``slide--/chamber--/field--/image--...ome.tif`` under an ordinary user folder
is past 260 characters, so its images could be neither listed nor opened.
spaCR respells only paths that need it with the ``\\\\?\\`` prefix; every
POSIX path, and every short Windows path, reaches the file call unchanged.
"""
from __future__ import annotations

import ntpath
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.portable_paths import _WINDOWS_PATH_ROOM, _long_path, _plain_path

MATRIX = ("slide--S00/chamber--U00--V04/field--X08--Y00/image--L00--S00--U00--"
          "V04--J20--E00--O00--X08--Y00--T00--Z00--C00.ome.tif")


def test_only_windows_paths_that_need_it_take_the_extended_spelling():
    short = "C:\\Users\\me\\data\\a.tif"
    deep = "C:\\Users\\me\\" + "d" * _WINDOWS_PATH_ROOM + "\\a.tif"
    assert _long_path(short, windows=True) == short
    assert _long_path(deep, windows=True) == "\\\\?\\" + deep
    assert _long_path(short, always=True, windows=True) == "\\\\?\\" + short
    assert _long_path("C:/Users/me/x/../a.tif", always=True,
                      windows=True) == "\\\\?\\C:\\Users\\me\\a.tif"
    share = "\\\\server\\lab\\" + "s" * _WINDOWS_PATH_ROOM
    assert _long_path(share, windows=True) == "\\\\?\\UNC\\server\\lab\\" + "s" * _WINDOWS_PATH_ROOM
    assert _long_path("\\\\?\\" + deep, windows=True) == "\\\\?\\" + deep
    for value in (None, 3, b"C:\\bytes"):
        assert _long_path(value, always=True, windows=True) is value
    assert _plain_path(_long_path(deep, windows=True), windows=True) == deep
    assert _plain_path(_long_path(share, windows=True), windows=True) == share


def test_a_matrix_screener_tree_under_a_windows_folder_needs_it():
    folder = ("C:\\Users\\researcher\\OneDrive - University of Michigan\\Documents"
              "\\spaCR example data\\2026-10\\import_example\\variants\\leica_matrix_screener\\plate1")
    path = ntpath.join(folder, *MATRIX.split("/"))
    assert len(path) > 260
    assert _long_path(path, windows=True).startswith("\\\\?\\C:\\")


@pytest.mark.parametrize("value", ["/" + "p" * 400 + "/a.tif", "rel/" + "q" * 300,
                                   Path("/x") / ("y" * 300)])
def test_posix_paths_are_never_respelled(value):
    assert _long_path(value, always=True, windows=False) is value
    if os.name != "nt":
        assert _long_path(value, always=True) is value
        assert _plain_path(str(value)) == str(value)


@pytest.mark.posix_only
def test_posix_image_io_hands_every_library_the_path_it_was_given(tmp_path, monkeypatch):
    """Byte-identical on POSIX: the writers and readers see the caller's path."""
    from spacr import io as spacr_io
    from spacr import mask_io, tiff_io

    seen = []
    real_imwrite, real_save = tifffile.imwrite, np.save

    def imwrite(path, *args, **kwargs):
        seen.append(("tiff", path))
        return real_imwrite(path, *args, **kwargs)

    def save(path, *args, **kwargs):
        seen.append(("npy", path))
        return real_save(path, *args, **kwargs)

    monkeypatch.setattr(tiff_io.tifffile, "imwrite", imwrite)
    monkeypatch.setattr(mask_io.np, "save", save)
    plane = np.arange(16, dtype=np.uint16).reshape(4, 4)
    tiff_io.write_tiff(str(tmp_path / "a.tif"), plane)
    tiff_io.write_tiff(tmp_path / "b.tif", plane)
    mask_io.save_mask(tmp_path / "m", plane, fmt="npy")
    assert seen == [("tiff", str(tmp_path / "a.tif")), ("tiff", tmp_path / "b.tif"),
                    ("npy", str(tmp_path / "m.npy"))]
    np.testing.assert_array_equal(mask_io.load_mask(tmp_path / "m"), plane)
    out = spacr_io._save_array_atomic(str(tmp_path / "stack" / "f.npy"), plane)
    assert out == str(tmp_path / "stack" / "f.npy")
    assert spacr_io._listdir_visible(str(tmp_path / "stack")) == ["f.npy"]


@pytest.mark.skipif(os.name != "nt", reason="native Windows MAX_PATH behaviour")
def test_a_matrix_screener_tree_past_max_path_converts_on_windows(tmp_path):
    """With long paths disabled, a deep Matrix Screener tree is read and written."""
    from spacr import convert as cv
    from spacr import image_import, io as spacr_io, mask_io
    from spacr.tiff_io import write_tiff

    root = str(tmp_path)
    while len(os.path.join(root, *MATRIX.split("/"))) <= 270:
        root = os.path.join(root, "a_deeper_project_folder")
    plate = os.path.join(root, "plate1")
    for channel in ("C00", "C01"):
        image = os.path.join(plate, *MATRIX.replace("C00", channel).split("/"))
        os.makedirs(_long_path(os.path.dirname(image), always=True), exist_ok=True)
        write_tiff(image, np.full((8, 8), 7, np.uint16))
        assert len(image) > 260
    sources = cv.scan(plate, metadata_type="leica_matrix_screener")
    assert len(sources) == 2 and not any(s.error for s in sources)
    result = cv.convert(cv.plan(sources), os.path.join(root, "converted"))
    assert len(result.written) == 2 and not result.failed
    layout = image_import.infer_layout(plate)
    assert layout.sampled == 2
    deep = os.path.join(root, "x" * 60, "masks")
    os.makedirs(_long_path(deep, always=True), exist_ok=True)
    mask_io.save_mask(os.path.join(deep, "field_mask.tif"), np.ones((8, 8), np.uint16))
    assert mask_io.load_mask(os.path.join(deep, "field_mask.tif")).sum() == 64
    stack = spacr_io._save_array_atomic(os.path.join(deep, "stack", "f.npy"),
                                        np.ones((8, 8, 2), np.uint16))
    assert spacr_io._listdir_visible(os.path.dirname(stack)) == ["f.npy"]
    first = os.path.join(plate, *MATRIX.split("/"))
    loaded = spacr_io.load_images_from_paths({"k": [first]})["k"]
    assert len(loaded) == 1 and loaded[0].shape == (8, 8)
