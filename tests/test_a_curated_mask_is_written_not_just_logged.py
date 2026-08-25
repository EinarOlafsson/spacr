"""A hand-corrected mask reaches the disk, not only its ledger.

Curation used to write the record and never the pixels: a session could
paint for an hour, write ``<mask>.curation.json`` beside a file whose
pixels the pipeline had produced, and :func:`spacr.curation.is_curated`
would then report that untouched file as hand-edited. The ledger is only
worth anything if the two are written together, so these tests hold
:meth:`spacr.curation.MaskCuration.save_mask` to writing both — and to
writing the ledger beside the file it actually wrote, not beside the name
it was handed.

The last one is about the cast that gets a mask onto disk: uint16 wraps,
and the value it wraps to first is 0, which is background.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.curation import CurationError, MaskCuration, is_curated, log_path_for
from spacr.layers import LabelsLayer
from spacr.mask_io import load_mask, save_mask


def _two_objects() -> np.ndarray:
    """A label image with two objects that are NOT numbered 1 and 2.

    The ids matter: a save that renumbers is a save that re-keys the mask
    against every measurement made from it.
    """
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[4:12, 4:12] = 3
    mask[20:28, 20:28] = 9
    return mask


def _session(tmp_path, name="plate1_A01_1.tif"):
    """A curation session over a two-object mask already on disk."""
    path = tmp_path / name
    mask = _two_objects()
    save_mask(path, mask)
    layer = LabelsLayer(mask.copy(), name="mask")
    return MaskCuration(layer, artifact=str(path)), str(path)


def test_saving_a_curated_mask_writes_the_painted_pixels(tmp_path):
    """The labels the brush changed are what comes back off the disk."""
    session, path = _session(tmp_path)
    session.label = 5
    changed = session.paint({"y": 16.0, "x": 16.0}, radius=2.0)
    assert changed > 0

    written = session.save_mask()

    assert written == path
    on_disk = load_mask(path)
    assert int((on_disk == 5).sum()) == changed


def test_a_curated_mask_keeps_the_object_ids_it_had(tmp_path):
    """Painting one corner must not renumber the objects it did not touch."""
    session, path = _session(tmp_path)
    session.label = 5
    session.paint({"y": 16.0, "x": 16.0}, radius=2.0)

    session.save_mask()

    on_disk = load_mask(path)
    assert int((on_disk == 3).sum()) == 64
    assert int((on_disk == 9).sum()) == 64


def test_a_saved_curated_mask_says_it_was_curated(tmp_path):
    """The whole point: the file on disk answers the provenance question."""
    session, path = _session(tmp_path)
    assert not is_curated(path)

    session.label = 5
    session.paint({"y": 16.0, "x": 16.0}, radius=2.0)
    session.save_mask()

    assert is_curated(path)


def test_a_mask_saved_without_being_painted_carries_no_ledger(tmp_path):
    """A ledger beside every mask ever opened answers no question.

    The pixels are still written — the caller asked for them — but nothing
    claims a correction that nobody made.
    """
    session, path = _session(tmp_path)

    written = session.save_mask()

    assert load_mask(written).max() == 9
    assert not is_curated(path)


def test_the_ledger_lands_beside_the_file_that_was_written(tmp_path):
    """A path with no extension resolves to a TIFF; the ledger follows it.

    Writing the ledger beside the name the caller passed would leave the
    record and the pixels naming two different files, and
    :func:`spacr.curation.is_curated` would answer about the wrong one.
    """
    layer = LabelsLayer(_two_objects(), name="mask")
    session = MaskCuration(layer, artifact=str(tmp_path / "stem_only"))
    session.label = 5
    session.paint({"y": 16.0, "x": 16.0}, radius=2.0)

    written = session.save_mask()

    assert written.endswith(".tif")
    assert (tmp_path / "stem_only.tif").is_file()
    assert (tmp_path / log_path_for("stem_only.tif")).is_file()
    assert is_curated(written)


def test_saving_with_nowhere_to_write_refuses(tmp_path):
    """An empty artefact path is refused rather than written to the cwd."""
    layer = LabelsLayer(_two_objects(), name="mask")
    session = MaskCuration(layer, artifact="mask")
    session.artifact = ""
    session.label = 5
    session.paint({"y": 16.0, "x": 16.0}, radius=2.0)

    with pytest.raises(CurationError):
        session.save_mask()


def test_an_object_id_too_big_for_uint16_is_refused(tmp_path):
    """65536 casts to 0 — background — so the mask is refused, not fused.

    Nothing downstream can tell "one fewer object" from "an object that
    silently became background", which is why this is an error rather than
    a warning.
    """
    mask = np.zeros((8, 8), dtype=np.int64)
    mask[2:4, 2:4] = 65536

    with pytest.raises(ValueError, match="65536"):
        save_mask(tmp_path / "too_many", mask)

    assert not (tmp_path / "too_many.tif").exists()
