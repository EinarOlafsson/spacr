"""The single-end branch of ``process_chunk``.

The paired-end half is exercised by ``tests/test_sequencing.py`` and the
conftest fixtures; the single-end half -- ``r2_chunk is None``, and the
nested ``single_find_sequence_in_chunk_reads`` it calls -- was reached by
no test at all. That matters beyond coverage: single-end is the branch
where read quality never influences a base call, because the R1 window
is used verbatim as the consensus, so a mistake there is silent.

Includes the truncated read the padding exists for, which is what makes
the ``len(consensus_seq) >= expected_end`` check below it unable to fail.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import sequencing as S

# A window of: 4-base column, 6-base gRNA, 4-base row.
REGEX = r"(?P<columnID>.{4})(?P<grna>.{6})(?P<rowID>.{4})"
ANCHOR = "GGGGG"
WINDOW = 14


def _record(sequence, quality=None):
    quality = quality or ("I" * len(sequence))
    return f"@read\n{sequence}\n+\n{quality}"


def _csv(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["sequence", "name"]).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def references(tmp_path):
    return (
        _csv(tmp_path, "columns.csv", [("ACGT", "col_1")]),
        _csv(tmp_path, "grnas.csv", [("TTTTTT", "grna_1")]),
        _csv(tmp_path, "rows.csv", [("CCCC", "row_A")]),
    )


def _chunk(reads, references, *, offset=len(ANCHOR), end=WINDOW, fill_na=False):
    column_csv, grna_csv, row_csv = references
    return (reads, REGEX, ANCHOR, offset, end,
            column_csv, grna_csv, row_csv, fill_na)


class TestOneCleanRead:

    def test_the_window_after_the_anchor_is_split_and_named(self,
                                                            references):
        """The whole single-end path, end to end."""
        read = _record("AAAA" + ANCHOR + "ACGT" + "TTTTTT" + "CCCC" + "GG")

        df, combos, qc = S.process_chunk(_chunk([read], references))

        assert list(df["read"]) == ["ACGTTTTTTTCCCC"]
        assert list(df["column_sequence"]) == ["ACGT"]
        assert list(df["grna_sequence"]) == ["TTTTTT"]
        assert list(df["row_sequence"]) == ["CCCC"]
        assert list(df["columnID"]) == ["col_1"]
        assert list(df["grna_name"]) == ["grna_1"]
        assert list(df["rowID"]) == ["row_A"]
        assert int(qc["total_reads"].iloc[0]) == 1
        assert int(combos["count"].iloc[0]) == 1

    def test_a_read_without_the_anchor_is_dropped_without_a_row(self,
                                                                references):
        """Documented, and worth holding: ``total_reads`` counts MATCHED
        reads, not input reads, so a QC row of 1 from two reads is the
        contract rather than a lost record."""
        good = _record("AAAA" + ANCHOR + "ACGT" + "TTTTTT" + "CCCC")
        missing = _record("AAAACCCCTTTTGGGCCCCAAAA".replace("GGGGG", "TTTTT"))

        df, _combos, qc = S.process_chunk(_chunk([good, missing], references))

        assert len(df) == 1
        assert int(qc["total_reads"].iloc[0]) == 1


class TestTheTruncatedRead:

    def test_a_short_window_is_padded_to_exactly_the_expected_length(
            self, references):
        """THE ARC the padding guarantees.

        The read ends four bases into the row barcode, so the window is
        cut short. It is right-padded with ``N`` to exactly
        ``expected_end`` -- which is why the length check below the
        padding can never fail, and why this read still matches a regex
        whose groups cover the full window.
        """
        read = _record("AAAA" + ANCHOR + "ACGT" + "TTTTTT" + "CC")

        df, _combos, _qc = S.process_chunk(_chunk([read], references))

        assert list(df["read"]) == ["ACGTTTTTTTCCNN"]
        assert len(df["read"].iloc[0]) == WINDOW
        assert list(df["row_sequence"]) == ["CCNN"]

    def test_a_padded_barcode_maps_to_no_name(self, references):
        """The consequence the docstring names: an ``N`` inside a barcode
        maps to NA and drops out of the per-well counts. A padded read
        that silently mapped to a real well would be a miscount."""
        read = _record("AAAA" + ANCHOR + "ACGT" + "TTTTTT" + "CC")

        df, _combos, qc = S.process_chunk(_chunk([read], references))

        assert pd.isna(df["rowID"].iloc[0])
        assert int(qc["rowID"].iloc[0]) == 1

    def test_a_padded_nonmatch_reaches_the_reverse_complement_fallback(
            self, references, monkeypatch, capsys):
        """Every existing consensus is long enough after padding.

        A direct mismatch therefore reaches the orientation fallback even
        when the original read ended early.  This observes the actual call,
        rather than pinning the spelling and order of source lines.
        """
        seen = []

        def _matching_reverse(sequence):
            seen.append(sequence)
            return "ACGTTTTTTTCCCC"

        monkeypatch.setattr(S, "reverse_complement", _matching_reverse)
        exact = r"(?P<columnID>ACGT)(?P<grna>TTTTTT)(?P<rowID>CCCC)"
        read = _record("AAAA" + ANCHOR + "ACGT" + "TTTTTT" + "CC")

        result = S.process_chunk(
            _chunk([read], references)[:1]
            + (exact,)
            + _chunk([read], references)[2:]
        )

        assert seen == ["ACGTTTTTTTCCNN"]
        assert "Reverse complement of last sequence" in capsys.readouterr().out
        assert result[0].empty


class TestTheOffset:

    def test_a_negative_start_is_clamped_to_the_read_start(self,
                                                           references):
        """Documented as clamped rather than rejected. Driven, because a
        rejection here would drop every read whose anchor sits near the
        start -- which is the ordinary case for a short adapter."""
        read = _record(ANCHOR + "ACGT" + "TTTTTT" + "CCCC")

        df, _combos, _qc = S.process_chunk(
            _chunk([read], references, offset=-100))

        assert len(df) == 1
        assert df["read"].iloc[0].startswith(ANCHOR)


class TestWhatIsRefused:

    def test_a_chunk_of_the_wrong_arity_is_refused(self, references):
        with pytest.raises(ValueError) as caught:
            S.process_chunk(("just", "three", "things"))

        assert "9 values for single-end" in str(caught.value)

    def test_a_zero_window_is_refused(self, references):
        with pytest.raises(ValueError) as caught:
            S.process_chunk(_chunk([], references, end=0))

        assert "expected_end must be a positive integer" in str(caught.value)

    def test_a_regex_without_the_named_groups_is_refused(self, references):
        column_csv, grna_csv, row_csv = references
        with pytest.raises(ValueError) as caught:
            S.process_chunk(([], r"(?P<grna>.{6})", ANCHOR, 0, WINDOW,
                             column_csv, grna_csv, row_csv, False))

        assert "missing required named group" in str(caught.value)
        assert "row/rowID" in str(caught.value)

    def test_a_malformed_fastq_record_is_refused(self, references):
        with pytest.raises(ValueError) as caught:
            S.process_chunk(_chunk(["@read\nACGT\n+"], references))

        assert "four lines" in str(caught.value)


class TestFillingTheBlanks:

    def test_fill_na_counts_an_unmapped_barcode_under_its_sequence(
            self, references):
        """Without it the groupby drops the row entirely, so a plate with
        one unlisted barcode loses those reads from the counts rather
        than showing them as unknown."""
        read = _record("AAAA" + ANCHOR + "GGGG" + "TTTTTT" + "CCCC")

        _df, combos, _qc = S.process_chunk(
            _chunk([read], references, fill_na=True))

        assert "GGGG" in set(combos["columnID"])

    def test_without_it_the_unmapped_row_is_dropped_from_the_counts(
            self, references):
        read = _record("AAAA" + ANCHOR + "GGGG" + "TTTTTT" + "CCCC")

        _df, combos, _qc = S.process_chunk(
            _chunk([read], references, fill_na=False))

        assert combos.empty


STRICT = r"(?P<columnID>ACGT)(?P<grna>[ACGT]{6})(?P<rowID>CCCC)"


class TestTheOrientationHint:
    """A chunk that matched nothing says so, and checks the one thing
    that is usually wrong: which strand the barcodes were written on.

    Getting this wrong is the single most common way a sequencing run
    produces an empty count table, and the message is what turns a
    silent zero into a diagnosis.
    """

    def test_a_chunk_with_no_anchor_at_all_warns_without_a_window(
            self, references, capsys):
        """THE ARC: ``consensus_seq`` is still None.

        No read carried the anchor, so the loop body never ran and there
        is no last window to re-check. The guard is what stops
        ``reverse_complement(None)`` -- and the warning is still printed,
        because "nothing matched" is the fact the operator needs whether
        or not a window exists to show them.
        """
        read = _record("AAAACCCCTTTTAAAACCCC")

        df, _combos, _qc = S.process_chunk(_chunk([read], references))

        printed = capsys.readouterr().out
        assert df.empty
        assert "No sequences matched" in printed
        assert "correct orientation?" in printed
        assert "Is None compatible with" in printed, (
            "the last-window line no longer reports None, so either a "
            "window is being invented or the guard below it changed")
        assert "Reverse complement" not in printed

    def test_a_window_that_only_matches_reversed_says_so(self, references,
                                                         capsys):
        """THE ARC: the reverse complement matches.

        The window is the reverse complement of a valid one, which is
        exactly what a library prepared on the other strand produces.
        Nothing matches forward, so the chunk is empty -- and the hint is
        the difference between "no reads" and "your barcodes are
        reversed".
        """
        column_csv, grna_csv, row_csv = references
        window = "GGGGAAAAAAACGT"          # reverse_complement of a match
        read = _record("AAAA" + ANCHOR + window)

        df, _combos, _qc = S.process_chunk(
            ([read], STRICT, ANCHOR, len(ANCHOR), WINDOW,
             column_csv, grna_csv, row_csv, False))

        printed = capsys.readouterr().out
        assert df.empty
        assert "No sequences matched" in printed
        assert "Reverse complement of last sequence in chunk matched" in \
            printed, (
            "a window that matches only reversed no longer produces the "
            "orientation hint, which is the one diagnosis this branch is "
            "for")

    def test_a_window_that_matches_neither_way_gives_no_hint(
            self, references, capsys):
        """THE OTHER ARC: the reverse complement does not match either.

        Without this half the hint would be printed for every empty
        chunk, which would send the operator to flip a strand that was
        never the problem.
        """
        column_csv, grna_csv, row_csv = references
        read = _record("AAAA" + ANCHOR + "TTTTTTTTTTTTTT")

        df, _combos, _qc = S.process_chunk(
            ([read], STRICT, ANCHOR, len(ANCHOR), WINDOW,
             column_csv, grna_csv, row_csv, False))

        printed = capsys.readouterr().out
        assert df.empty
        assert "No sequences matched" in printed
        assert "Reverse complement" not in printed

    def test_a_chunk_that_matched_says_nothing(self, references, capsys):
        """The guard above all of it: a working run must not print a
        warning per chunk."""
        read = _record("AAAA" + ANCHOR + "ACGT" + "TTTTTT" + "CCCC")

        df, _combos, _qc = S.process_chunk(_chunk([read], references))

        assert len(df) == 1
        assert "No sequences matched" not in capsys.readouterr().out
