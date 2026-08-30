"""The corners of spacr.sequencing a screen actually lands in.

Three of them, each one a way a run can look finished while the counts are
wrong or missing:

* a lane whose barcodes were written in the wrong orientation matches
  nothing, and the only thing standing between the user and a silently empty
  count table is the hint the chunk prints about its last read;
* a read cut short by the sequencer is padded with ``N`` rather than
  rejected, so it still satisfies the window length and still matches the
  regex -- and then quietly drops out of the per-well counts, unless
  ``fill_na`` is on, which keeps it under its raw sequence instead;
* a ``single_direction`` the loop does not recognise reaches the read
  function with no file chosen at all.
"""
from __future__ import annotations

import os
import re

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from spacr import sequencing as SEQ

# ---------------------------------------------------------------------------
# A tiny barcode layout: <col:4>GGAA<grna:4><row:4>, anchored on a
# palindromic GGATCC so a paired mate is just the reverse complement of R1.
# ---------------------------------------------------------------------------

_ANCHOR = "GGATCC"
_REGEX = r"^(?P<columnID>.{4})GGAA(?P<grna>.{4})(?P<rowID>.{4})"
_WINDOW = 16
_COL, _GRNA, _ROW = "AAAA", "CCCC", "TTTT"
_GOOD = f"{_COL}GGAA{_GRNA}{_ROW}"


def _write_csv(path, seqs, names):
    pd.DataFrame({"sequence": seqs, "name": names}).to_csv(path, index=False)
    return str(path)


def _refs(tmp_path):
    return (_write_csv(tmp_path / "c.csv", [_COL], ["col0"]),
            _write_csv(tmp_path / "g.csv", [_GRNA], ["sg0"]),
            _write_csv(tmp_path / "r.csv", [_ROW], ["row0"]))


def _fastq(seq, name="r0"):
    return f"@{name}\n{seq}\n+\n{'I' * len(seq)}"


def _read(payload, lead="TT"):
    """A read carrying ``payload`` immediately after the anchor."""
    return f"{lead}{_ANCHOR}{payload}"


def _single(tmp_path, payloads, fill_na=False):
    c, g, r = _refs(tmp_path)
    chunk = [_fastq(_read(p), f"r{i}") for i, p in enumerate(payloads)]
    return SEQ.process_chunk((chunk, _REGEX, _ANCHOR, len(_ANCHOR), _WINDOW,
                              c, g, r, fill_na))


def _paired(tmp_path, payload):
    """One pair whose R2 is the reverse complement of R1, as a sequencer emits."""
    c, g, r = _refs(tmp_path)
    r1 = _read(payload)
    r2 = SEQ.reverse_complement(r1)
    return SEQ.process_chunk(([_fastq(r1, "p0")], [_fastq(r2, "p0")], _REGEX,
                              _ANCHOR, len(_ANCHOR), _WINDOW, c, g, r, False))


# ---------------------------------------------------------------------------
# The orientation hint: the one line that separates "wrong barcodes" from
# "wrong strand"
# ---------------------------------------------------------------------------

def test_a_chunk_matching_nothing_says_whether_the_strand_is_the_reason(
        tmp_path, capsys):
    """A lane that decodes nothing must say whether flipping it would fix it.

    Both failures produce the same visible result -- an empty count table --
    but they need opposite fixes: barcode CSVs written in the wrong
    orientation are repaired by reverse-complementing the reference, while a
    genuinely unrelated library is not. The chunk re-tries its last read
    reverse-complemented and only claims a match when there is one, so the
    hint has to appear for the flipped lane and stay away from the one that
    is simply not this library. A hint that printed either way would send
    every user to the same wrong fix.
    """
    # A lane written on the wrong strand: the window IS the reverse
    # complement of a decodable read.
    flipped = SEQ.reverse_complement(_GOOD)
    assert re.match(_REGEX, flipped) is None, "the flipped read must not decode"

    df, _unique, qc = _paired(tmp_path, flipped)
    said = capsys.readouterr().out

    assert df.empty, "a read that does not match the regex was still counted"
    assert qc.loc["NaN_Counts", "total_reads"] == 0
    assert "Are barcode sequences in the correct orientation?" in said
    assert flipped in said, "the read that failed to match was not shown"
    assert "Reverse complement of last sequence in chunk matched" in said

    # And a lane that is not this library at all: same empty table, no hint,
    # because reverse-complementing it would not help either.
    junk = "A" * _WINDOW
    assert re.match(_REGEX, SEQ.reverse_complement(junk)) is None

    df2, _, qc2 = _paired(tmp_path, junk)
    said2 = capsys.readouterr().out

    assert df2.empty
    assert qc2.loc["NaN_Counts", "total_reads"] == 0
    assert "Are barcode sequences in the correct orientation?" in said2
    assert "Reverse complement of last sequence in chunk matched" not in said2, (
        "a strand hint was offered for reads the flip does not rescue")


def test_a_chunk_with_no_anchor_at_all_reports_no_read_to_flip(
        tmp_path, capsys):
    """With nothing to show, the warning must not pretend it examined a read.

    When no read in a chunk even carries the anchor there is no window to
    re-try, and the diagnostic says ``Is None compatible with ...``. That
    ``None`` is the signal that the anchor -- not the barcode orientation --
    is what went wrong, which is a different setting for the user to fix
    (``target_sequence``, not the barcode CSVs).
    """
    # A read with real bases but no anchor: there is no window to re-try.
    c, g, r = _refs(tmp_path)
    df2, _, qc2 = SEQ.process_chunk(
        ([_fastq("TTTTTTTTTTTTTTTTTTTTTTTT")], _REGEX, _ANCHOR,
         len(_ANCHOR), _WINDOW, c, g, r, False))
    said = capsys.readouterr().out

    assert df2.empty, "a read without the anchor was decoded anyway"
    assert qc2.loc["NaN_Counts", "total_reads"] == 0
    assert "Is None compatible with" in said
    assert "Reverse complement of last sequence in chunk matched" not in said

    # The same chunk with the anchor present does reach a window, so the
    # 'None' above is the missing anchor and not a dead code path.
    df3, _, _ = _single(tmp_path, [_GOOD])
    assert df3["read"].tolist() == [_GOOD]


# ---------------------------------------------------------------------------
# Padding: why the length check below it can never fail
# ---------------------------------------------------------------------------

def test_a_read_cut_short_is_padded_with_N_and_falls_out_of_the_counts(
        tmp_path):
    """A truncated read still matches, and then silently loses its well.

    The window is padded to its full length with ``N`` rather than dropped,
    so the regex still matches and the read still lands in the annotated
    table -- but ``CCNN`` is not a barcode in anyone's reference, so it maps
    to NA and disappears from ``unique_combinations``. A user reading only
    the counts sees a well that came up short with nothing explaining it;
    the QC row's NaN counts are the only place that loss is recorded, which
    is why they have to be right.
    """
    short = f"{_COL}GGAACC"          # six bases of payload, window is 16
    df, unique, qc = _single(tmp_path, [_GOOD, short])

    padded = f"{_COL}GGAACC" + "N" * (_WINDOW - len(short))
    assert df["read"].tolist() == [_GOOD, padded]
    assert len(padded) == _WINDOW, "the window was not padded to full length"

    # The truncated read carries barcodes nothing can name.
    assert df["grna_name"].tolist()[0] == "sg0"
    assert pd.isna(df["grna_name"][1]), "an N-padded barcode was given a name"
    assert pd.isna(df["rowID"][1])
    assert qc.loc["NaN_Counts", "total_reads"] == 2
    assert qc.loc["NaN_Counts", "rowID"] == 1
    assert qc.loc["NaN_Counts", "grna_name"] == 1

    # ... and only the intact read reaches the per-well counts.
    assert unique[["rowID", "columnID", "grna_name", "count"]].values.tolist() \
        == [["row0", "col0", "sg0", 1]]


def test_fill_na_keeps_an_unnamed_barcode_under_its_own_sequence(tmp_path):
    """``fill_na`` is the difference between a lost read and a named unknown.

    Without it, every read whose barcode is missing from the reference CSVs
    is dropped by the groupby and the run reports fewer reads than it
    processed, with no row saying where they went. With it, the raw sequence
    stands in for the name, so an incomplete barcode CSV shows up as an
    unrecognised sequence in the counts instead of as a hole. The two runs
    below are the same reads, so the extra row is the setting and nothing
    else.
    """
    short = f"{_COL}GGAACC"
    _, dropped, _ = _single(tmp_path, [_GOOD, short], fill_na=False)
    _, kept, _ = _single(tmp_path, [_GOOD, short], fill_na=True)

    assert dropped["grna_name"].tolist() == ["sg0"]
    assert len(dropped) == 1, "an unnamed barcode was counted without fill_na"

    assert len(kept) == 2, "fill_na lost the read it exists to keep"
    filled = kept[kept["grna_name"] != "sg0"].iloc[0]
    # The raw sequences, exactly as they came off the read, padding included.
    assert filled["grna_name"] == "CCNN"
    assert filled["rowID"] == "NNNN"
    assert filled["columnID"] == "col0", "a barcode that DID map lost its name"
    assert int(filled["count"]) == 1
    assert int(kept.loc[kept["grna_name"] == "sg0", "count"].iloc[0]) == 1


# ---------------------------------------------------------------------------
# generate_barecode_mapping — the sample loop's direction switch
# ---------------------------------------------------------------------------

class _Attempt:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Policy:
    def attempts_for(self, key, stage=None):
        yield _Attempt()


class _Run:
    policy = _Policy()


class _RunCtx:
    def __enter__(self):
        return _Run()

    def __exit__(self, *exc):
        return False


@pytest.fixture
def mapping_env(tmp_path, monkeypatch):
    """Stub everything generate_barecode_mapping delegates to; record calls."""
    calls = []
    monkeypatch.setattr(SEQ, "paired_read_chunked_processing",
                        lambda **kw: calls.append(("paired", kw)))
    monkeypatch.setattr(SEQ, "single_read_chunked_processing",
                        lambda **kw: calls.append(("single", kw)))
    monkeypatch.setattr(SEQ, "run_context", lambda *a, **k: _RunCtx())
    monkeypatch.setattr(SEQ, "_run_barcode_qc", lambda *a, **k: None)

    import spacr.io as IO
    import spacr.utils as U
    monkeypatch.setattr(U, "save_settings", lambda *a, **k: None)
    monkeypatch.setattr(
        IO, "parse_gz_files",
        lambda src: {"S1": {"R1": os.path.join(src, "S1_R1.fastq.gz"),
                            "R2": os.path.join(src, "S1_R2.fastq.gz")}})
    return calls


def test_a_misspelt_single_direction_never_reaches_a_fastq_file(
        tmp_path, mapping_env):
    """A direction the loop does not know stops the sample dead.

    ``single_direction`` is free text in the settings, and only the exact
    strings ``'R1'`` and ``'R2'`` choose a mate. Anything else -- the
    lower-case ``'r2'`` a user types by hand is the obvious one -- picks no
    file at all and the sample cannot be read. This test pins BOTH halves:
    the spelling that works reads the mate it names, and the spelling that
    does not raises instead of quietly processing the other mate or writing
    an empty table that looks like a finished sample.

    The exception it raises today is a bare ``NameError`` about ``R1``,
    which names neither the setting nor the sample; that is a defect worth
    a message, but raising is still the behaviour the counts depend on.
    """
    SEQ.generate_barecode_mapping({"src": str(tmp_path), "mode": "single",
                                   "single_direction": "R2"})
    kind, kw = mapping_env[0]
    assert kind == "single"
    assert kw["r1_file"].endswith("S1_R2.fastq.gz")
    assert kw["r2_file"] is None

    with pytest.raises(NameError) as exc:
        SEQ.generate_barecode_mapping({"src": str(tmp_path), "mode": "single",
                                       "single_direction": "r2"})
    assert "R1" in str(exc.value)
    assert len(mapping_env) == 1, (
        "an unrecognised direction still handed a file to the reader")
    # The output folder for the unusable direction is created before the
    # failure, so an interrupted run leaves an empty sample folder behind.
    assert os.path.isdir(os.path.join(str(tmp_path), "S1_single_r2"))
