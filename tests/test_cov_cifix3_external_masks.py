"""Label-image import from other tools: which files in a folder are importable."""
from __future__ import annotations

import os

from spacr import external_masks as em


def _touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return folder


def test_dot_files_and_other_extensions_are_not_importable(tmp_path):
    folder = _touch(tmp_path / "f", "a.tif", ".hidden.tif", "notes.txt")
    found = em._label_import_files(str(folder), recursive=False)
    assert [os.path.basename(path) for path in found] == ["a.tif"]
