"""Three more last branches: a summary line, a zarr codec, and a resume probe.

The first two are cases where the untaken side is the ORDINARY one -- a
correction that only removed labels, a mask stored as single bytes -- which is
how they came to be missed: the fixtures that existed all happened to be the
interesting case.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# napari_bridge.CorrectionResult.describe — arc 543 -> 545, nothing added
# ---------------------------------------------------------------------------

def test_a_correction_that_only_removed_labels_says_so_and_nothing_more():
    """The ``if self.added:`` branch not taken.

    Deleting debris is the commonest curation there is, and it adds nothing.
    The sentence must not carry "0 object(s) added" -- a count of zero reads as
    a thing that happened -- so each clause is present only when it has
    something to report.
    """
    from spacr.napari_bridge import CorrectionResult

    result = CorrectionResult(mask_path="/tmp/plate1_A01.npy",
                              changed_pixels=412, removed=(7, 9))
    line = result.describe()

    assert "412 pixel(s)" in line
    assert "2 removed" in line
    assert "added" not in line
    assert "reshaped" not in line


def test_a_correction_that_did_everything_reports_every_clause():
    """The taken side of all three guards, as the contrast."""
    from spacr.napari_bridge import CorrectionResult

    line = CorrectionResult(mask_path="/tmp/m.npy", changed_pixels=9,
                            added=(1,), removed=(2, 3), altered=(4,)).describe()

    assert "1 object(s) added" in line
    assert "2 removed" in line
    assert "1 reshaped" in line


def test_an_unchanged_mask_says_nothing_was_written():
    """The early return above all three, which the clauses depend on."""
    from spacr.napari_bridge import CorrectionResult

    result = CorrectionResult(mask_path="/tmp/m.npy", changed_pixels=0)
    assert result.describe() == "The mask came back unchanged; nothing was written."
    assert not result


# ---------------------------------------------------------------------------
# ome_zarr._v3_codec_chain — arc 1306 -> 1308, a single-byte dtype
# ---------------------------------------------------------------------------

def test_a_single_byte_dtype_keeps_its_byte_order_untouched():
    """The ``if dtype.itemsize > 1:`` branch not taken.

    A uint8 mask has no byte order to fix, and calling ``newbyteorder`` on it
    would stamp a big- or little-endian flag onto a dtype where the concept is
    meaningless. Segmentation masks and 8-bit previews are among the commonest
    things spaCR reads, so this is the ordinary path, not an edge case.
    """
    from pathlib import Path
    from spacr.ome_zarr import _v3_codec_chain

    codecs = [{"name": "bytes", "configuration": {"endian": "big"}}]
    specs, transpose, dtype = _v3_codec_chain(codecs, Path("x.zarr"),
                                              np.dtype("uint8"))

    assert dtype == np.dtype("uint8")
    assert dtype.byteorder == "|"          # "not applicable", still
    assert transpose is None


def test_a_multi_byte_dtype_takes_the_declared_byte_order():
    """The taken side: this is where reading it wrong yields plausible garbage."""
    from pathlib import Path
    from spacr.ome_zarr import _v3_codec_chain

    codecs = [{"name": "bytes", "configuration": {"endian": "big"}}]
    _specs, _transpose, dtype = _v3_codec_chain(codecs, Path("x.zarr"),
                                                np.dtype("uint16"))

    assert dtype.byteorder == ">"


def test_an_endian_the_spec_does_not_define_is_refused():
    """The raise above both, so the two tests above are the non-error paths."""
    from pathlib import Path
    from spacr.ome_zarr import OmeZarrError, _v3_codec_chain

    with pytest.raises(OmeZarrError, match="endian"):
        _v3_codec_chain([{"name": "bytes", "configuration": {"endian": "middle"}}],
                        Path("x.zarr"), np.dtype("uint16"))


# ---------------------------------------------------------------------------
# resume.importer_written_columns — arc 892 -> 894, a twin with no columns
# ---------------------------------------------------------------------------

def test_a_foreign_twin_that_reports_no_columns_answers_none(monkeypatch):
    """The ``if columns:`` branch not taken.

    The guard exists because ``_table_columns`` returns ``[]`` for anything it
    cannot read -- it swallows sqlite3.Error by design. So the twin can be
    listed and still yield nothing, and the caller must get ``None`` (meaning
    "no information") rather than an empty set (meaning "no columns were
    written"). Those two answers lead to opposite resume decisions.

    The empty answer is forced rather than staged: producing a real table that
    is listed but has no columns requires a race between the listing and the
    PRAGMA, which is precisely why the guard is there and not something a
    fixture can hold still.
    """
    from spacr import resume as resume_module

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(f'CREATE TABLE "{resume_module.FOREIGN_PREFIX}cell" (id INTEGER)')
        conn.commit()

        # Baseline: the twin is readable, so the real columns come back.
        assert resume_module.importer_written_columns(conn, "cell") == {"id"}

        monkeypatch.setattr(resume_module, "_table_columns", lambda *_a: [])
        assert resume_module.importer_written_columns(conn, "cell") is None
    finally:
        conn.close()


def test_no_foreign_twin_at_all_answers_none():
    """The other way to reach the same ``return None``, with no patching."""
    from spacr import resume as resume_module

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute('CREATE TABLE cell (id INTEGER)')
        conn.commit()
        assert resume_module.importer_written_columns(conn, "cell") is None
    finally:
        conn.close()
