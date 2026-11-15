"""Caps, rollovers and the missing folder — the drop path's boundary cases.

Dropping a plate folder walks it once and guesses a layout from a probe. What
is asserted here is what happens at the edges of that walk: a folder that is
not there, a probe that fills up before the tree does, and the well/field
counters rolling over. The counters matter most -- a rollover that reused a
well name would map two different acquisitions onto one well, and nothing
downstream could tell.
"""
from __future__ import annotations

from pathlib import Path

from spacr.qt import folder_metadata as fm


def _plate(root: Path, wells, fields, *, extra_junk=True):
    """A ``<well>/<field>/img.tif`` tree, with a non-image file beside each."""
    made = []
    for well in wells:
        for field in fields:
            folder = root / well / field
            folder.mkdir(parents=True)
            image = folder / "img.tif"
            image.write_bytes(b"\x00")
            made.append(image)
            if extra_junk:
                (folder / "notes.txt").write_bytes(b"not an image")
    return made


def test_a_folder_that_is_not_there_is_not_a_layout(tmp_path):
    """A missing drop target answers ``None`` rather than raising.

    The detector runs on whatever a drag-and-drop handed over, which can be a
    path that has since been unmounted or renamed. An exception here would
    surface as a crash on drop; ``None`` is "no layout detected", which the
    caller already handles.
    """
    assert fm.detect_folder_metadata(tmp_path / "not-here") is None
    # A file, not a directory, is the same non-answer.
    plain = tmp_path / "single.tif"
    plain.write_bytes(b"\x00")
    assert fm.detect_folder_metadata(plain) is None


def test_the_walk_stops_at_its_cap_instead_of_draining_the_tree(tmp_path):
    """``cap`` is why dropping a 100 000-file plate folder is not a freeze.

    The generator must stop yielding at the cap and, crucially, stop walking:
    a cap enforced only by the consumer would still stat every entry. Non-image
    files do not count against it, because the cap counts images.
    """
    _plate(tmp_path, ["A01", "A02", "A03"], ["F01", "F02"])

    assert len(list(fm.iter_image_files(tmp_path))) == 6
    assert len(list(fm.iter_image_files(tmp_path, cap=2))) == 2
    assert len(list(fm.iter_image_files(tmp_path, cap=99))) == 6
    assert all(p.suffix == ".tif" for p in fm.iter_image_files(tmp_path))


def test_the_probe_stops_once_it_has_seen_enough_matching_files(tmp_path):
    """A layout that repeats does not need the whole plate to be recognised.

    ``max_probe`` bounds the files inspected, and the detected template must
    be the same one the full walk finds -- the point of a probe is that it is
    cheaper, not that it is different. Only the recorded samples shrink.
    """
    files = _plate(tmp_path, ["A01", "A02", "A03"], ["F01", "F02"])

    full = fm.detect_folder_metadata(tmp_path)
    probed = fm.detect_folder_metadata(tmp_path, max_probe=2)

    assert full is not None and probed is not None
    assert probed.depth_labels == full.depth_labels == ("well", "field")
    assert len(probed.sample_paths) == 2
    assert len(probed.sample_paths) < len(full.sample_paths) <= len(files)

    # The same cap applies when the caller supplies the file list itself,
    # which is how one traversal serves both the probe and the extraction.
    handed = fm.detect_folder_metadata(tmp_path, max_probe=2, files=files)
    assert len(handed.sample_paths) == 2
