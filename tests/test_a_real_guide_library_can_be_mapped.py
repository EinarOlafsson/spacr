"""spaCR would not map the maintainer's own gRNA library.

The tsg101 screen's library is 1,385 guides. Three of its sequences appear
under more than one name -- two guides of TGGT1_241310 also appear under
TGGT1_411210 and TGGT1_411710, eight rows in all -- and
``map_sequences_to_names`` refused the WHOLE FILE:

    Barcode mapping '...grna_barcodes.csv' contains duplicate sequences;
    each sequence must identify exactly one name.

So the 1,382 unambiguous guides were unusable because of eight rows, and
the screen could not be mapped at all.

THE SAFETY PROPERTY IS KEPT. A read carrying a shared sequence genuinely
cannot be told which guide it came from, and attributing it would put one
gene's counts on another. Those sequences map to NA, so the reads fall out
of the per-well counts -- which is what refusing them was protecting, and
it is achieved without discarding the rest of the library.

A library that is MOSTLY duplicates is a different thing -- a mis-built or
mis-columned file -- and still raises.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from spacr.sequencing import map_sequences_to_names, reverse_complement


LIBRARY = os.environ.get("SPACR_GRNA_LIBRARY", "")


def _library(tmp_path, rows, name="lib.csv"):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["name", "sequence"]).to_csv(path, index=False)
    return str(path)


def test_a_shared_sequence_is_unassigned_not_fatal(tmp_path, capsys):
    path = _library(tmp_path, [
        ("gene_a_1", "AAAACCCCGGGGTTTTAAAA"),
        ("gene_b_1", "AAAACCCCGGGGTTTTAAAA"),   # the same sequence
        ("gene_c_1", "TTTTGGGGCCCCAAAATTTT"),
    ])

    mapped = pd.Series(map_sequences_to_names(
        path, pd.Series(["AAAACCCCGGGGTTTTAAAA", "TTTTGGGGCCCCAAAATTTT"]),
        rc=False))

    assert pd.isna(mapped.iloc[0]), "an ambiguous read was attributed anyway"
    assert mapped.iloc[1] == "gene_c_1", "an unambiguous guide was lost"


def test_it_says_which_sequences_it_dropped(tmp_path, capsys):
    """Silence here is a count that quietly went missing."""
    path = _library(tmp_path, [
        ("gene_a_1", "AAAACCCCGGGGTTTTAAAA"),
        ("gene_b_1", "AAAACCCCGGGGTTTTAAAA"),
        ("gene_c_1", "TTTTGGGGCCCCAAAATTTT"),
    ])
    map_sequences_to_names(path, pd.Series(["TTTTGGGGCCCCAAAATTTT"]), rc=False)

    said = capsys.readouterr().out
    assert "more than one name" in said
    assert "AAAACCCCGGGGTTTTAAAA" in said
    assert "unassigned" in said


def test_a_clean_library_says_nothing(tmp_path, capsys):
    path = _library(tmp_path, [("gene_a_1", "AAAACCCCGGGGTTTTAAAA"),
                               ("gene_c_1", "TTTTGGGGCCCCAAAATTTT")])
    map_sequences_to_names(path, pd.Series(["TTTTGGGGCCCCAAAATTTT"]), rc=False)

    assert "more than one name" not in capsys.readouterr().out


def test_a_library_with_nothing_usable_left_still_raises(tmp_path):
    """Every sequence shared is a mis-built file, not a real library.

    Mapping it would return NA for every read and report nothing wrong.
    """
    path = _library(tmp_path, [("a", "AAAA"), ("b", "AAAA"),
                               ("c", "CCCC"), ("d", "CCCC")])

    with pytest.raises(ValueError, match="no usable barcode"):
        map_sequences_to_names(path, pd.Series(["AAAA"]), rc=False)


def test_a_missing_column_still_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["a"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        map_sequences_to_names(str(path), pd.Series(["AAAA"]), rc=False)


def test_the_reverse_complement_orientation_still_maps(tmp_path):
    """Orientation is the documented source of silent zero counts."""
    forward = "AAAACCCCGGGGTTTTAAAA"
    path = _library(tmp_path, [("gene_a_1", forward)])

    mapped = map_sequences_to_names(
        path, pd.Series([reverse_complement(forward)]), rc=True)

    assert mapped[0] == "gene_a_1"


# ---------------------------------------------------------------------------
# the real library
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LIBRARY or not os.path.exists(LIBRARY),
                    reason="set SPACR_GRNA_LIBRARY to a real gRNA barcode CSV")
def test_the_real_library_maps():
    frame = pd.read_csv(LIBRARY)
    assert len(frame) > 100, "that is not a screen library"

    mapped = pd.Series(map_sequences_to_names(
        LIBRARY, pd.Series(frame["sequence"].astype(str)), rc=False))

    named = int(mapped.notna().sum())
    assert named > 0.98 * len(frame), (
        f"only {named} of {len(frame)} guides mapped; a handful of shared "
        f"sequences must not cost the library")
