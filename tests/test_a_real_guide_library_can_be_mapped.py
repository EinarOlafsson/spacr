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

import numpy as np
import pandas as pd
import pytest

from spacr.sequencing import map_sequences_to_names, reverse_complement


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
# a library the size and shape of the real one
# ---------------------------------------------------------------------------

def test_a_screen_sized_library_with_a_few_shared_sequences_maps(tmp_path):
    """The tsg101 library's shape: 1,385 rows, three sequences under more
    than one name, eight rows in all. Every other row maps to its own name.

    Built rather than read, so it runs everywhere; the proportions are the
    real library's, which is what the refusal used to trip over.
    """
    rng = np.random.default_rng(1385)
    sequences = set()
    while len(sequences) < 1385 - 5:
        sequences.add("".join(rng.choice(list("ACGT"), 21)))
    unique = sorted(sequences)
    rows = [(f"TGGT1_{i // 4:06d}_{i % 4 + 1}", seq)
            for i, seq in enumerate(unique)]
    shared = unique[:3]
    rows += [("TGGT1_411210_1", shared[0]), ("TGGT1_411710_1", shared[0]),
             ("TGGT1_411210_2", shared[1]), ("TGGT1_411710_2", shared[1]),
             ("TGGT1_411710_3", shared[2])]
    assert len(rows) == 1385
    path = _library(tmp_path, rows)

    frame = pd.read_csv(path)
    mapped = pd.Series(map_sequences_to_names(
        path, pd.Series(frame["sequence"].astype(str)), rc=False))

    ambiguous = frame["sequence"].isin(shared)
    assert int(ambiguous.sum()) == 8
    assert mapped[ambiguous.to_numpy()].isna().all(), (
        "a read carrying a shared sequence was attributed to one guide")
    assert (mapped[~ambiguous.to_numpy()].to_numpy()
            == frame.loc[~ambiguous, "name"].to_numpy()).all(), (
        "an unambiguous guide lost or changed its name")
    assert int(mapped.notna().sum()) == 1385 - 8
