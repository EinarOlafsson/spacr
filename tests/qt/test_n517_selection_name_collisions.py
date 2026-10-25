"""Distinct accepted plaque images retain their links when legacy labels collide."""
import os
from pathlib import Path

from PIL import Image
import pytest

from spacr.qt import dnd_handlers as dh


@pytest.mark.parametrize("reserved", [0, 2])
@pytest.mark.parametrize("hard_links", [False, True])
def test_every_distinct_image_keeps_its_bytes_order_and_original_link_names(
        tmp_path, monkeypatch, reserved, hard_links):
    final_index = reserved + 3
    paths = [tmp_path / "a" / f"{final_index:04d}_{number}_field.png"
             for number in range(2, reserved + 2)]
    paths += [tmp_path / "a" / f"{final_index:04d}_field.png",
              tmp_path / "a" / "a_field.png", tmp_path / "a" / "field.png",
              tmp_path / "b" / "a" / "field.png"]
    originals = {}
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (7, 9), 31 + index).save(path)
        originals[path] = path.read_bytes()
    symbolic = not hard_links
    if symbolic:
        try:
            os.symlink(paths[0], tmp_path / "symlink-probe")
        except OSError:
            symbolic = False
    actual_symlink = os.symlink
    attempts = []
    def link(source, target):
        attempts.append((Path(source), Path(target)))
        if hard_links:
            raise PermissionError("symbolic links unavailable")
        return actual_symlink(source, target)
    monkeypatch.setattr(dh.os, "symlink", link)
    output = dh.plaque_selection_folder(paths, stamp="native-collision")
    expected_names = [path.name for path in paths[:-1]]
    expected_names.append(f"{final_index:04d}_{reserved + 2}_field.png")
    assert [target.name for source, target in attempts] == expected_names
    assert len(list(output.iterdir())) == len(paths)
    assert len({target.name.lower() for source, target in attempts}) == len(paths)
    for path, (source, target) in zip(paths, attempts):
        assert source == path.resolve()
        assert path.read_bytes() == originals[path]
        assert target.read_bytes() == originals[path]
        assert target.parent == output
        if not symbolic:
            assert not target.is_symlink()
            assert os.path.samefile(path, target)
        else:
            assert target.is_symlink()
            assert target.resolve() == path.resolve()
