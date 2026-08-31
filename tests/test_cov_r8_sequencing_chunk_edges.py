"""Reads a chunk cannot decode, and the fills that have nothing to fill.

``process_chunk`` is the whole of the barcode decode, and every branch
left in it is a read that came up short or a reference column that is not
there. Both matter more than they look: a read silently dropped is a
count that never existed, and a fill applied to a column that is absent
is a KeyError in the middle of a run over millions of reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import sequencing as SEQ


_REGEX = (r"^(?P<columnID>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})"
          r"AACTT.*AGAAG(?P<rowID>.{8}).*")
_WINDOW = 62
_TARGET = "TGCTGAAATAAAC"
_COL, _ROW, _GRNA = "AAAACCCC", "GGGGTTTT", "A" * 20


def _read(col=_COL, row=_ROW, grna=_GRNA, tail="AAAA"):
    return f"{col}TGCTGAAATAAAC{grna}AACTTAAAAGAAG{row}{tail}"


def _fastq(seq, name="r0", qual=None):
    qual = qual or "I" * len(seq)
    return f"@{name}\n{seq}\n+\n{qual}"


def _write_csv(path, sequences, names):
    pd.DataFrame({"sequence": sequences, "name": names}).to_csv(
        path, index=False)
    return str(path)


def _refs(tmp_path):
    return (_write_csv(tmp_path / "c.csv", [_COL], ["col0"]),
            _write_csv(tmp_path / "g.csv", [_GRNA], ["sg0"]),
            _write_csv(tmp_path / "r.csv", [_ROW], ["row0"]))


# ---------------------------------------------------------------------------
# A read that is shorter than the window
# ---------------------------------------------------------------------------

class TestAReadThatCameUpShort:

    def test_a_full_length_read_decodes_all_three_barcodes(self, tmp_path):
        c, g, r = _refs(tmp_path)

        df, _counts, _qc = SEQ.process_chunk(
            ([_fastq(_read())], _REGEX, _TARGET, -8, _WINDOW, c, g, r, False))

        assert len(df) == 1
        assert df.iloc[0]["columnID"] == "col0"
        assert df.iloc[0]["rowID"] == "row0"
        assert df.iloc[0]["grna_name"] == "sg0"

    def test_a_truncated_read_is_padded_and_then_fails_the_regex(self,
                                                                  tmp_path):
        """A short read is padded to the window with N, not dropped.

        That is the design: the window is fixed, so a read that anchors
        and then runs out still gets a full-length consensus -- of Ns
        where it had no bases. The regex is what rejects it, and it
        rejects it for the right reason: those positions are unknown,
        not wrong.
        """
        c, g, r = _refs(tmp_path)
        short = _read()[:30]                 # anchors, then stops

        df, _counts, qc = SEQ.process_chunk(
            ([_fastq(short)], _REGEX, _TARGET, -8, _WINDOW, c, g, r, False))

        assert df.empty, "a truncated read was decoded anyway"
        assert int(qc.iloc[0]["total_reads"]) == 0

    def test_a_paired_chunk_decodes_the_same_barcodes(self, tmp_path):
        c, g, r = _refs(tmp_path)
        forward = _read()
        reverse = SEQ.reverse_complement(forward)

        df, _counts, _qc = SEQ.process_chunk(
            ([_fastq(forward)], [_fastq(reverse)], _REGEX, _TARGET, -8,
             _WINDOW, c, g, r, False))

        assert len(df) == 1
        assert df.iloc[0]["grna_name"] == "sg0"

    def test_a_paired_chunk_of_truncated_reads_decodes_nothing(self,
                                                                tmp_path):
        c, g, r = _refs(tmp_path)
        forward = _read()[:30]
        reverse = SEQ.reverse_complement(_read())[:30]

        df, _counts, _qc = SEQ.process_chunk(
            ([_fastq(forward)], [_fastq(reverse)], _REGEX, _TARGET, -8,
             _WINDOW, c, g, r, False))

        assert df.empty

    def test_mismatched_paired_chunks_are_refused_by_count(self, tmp_path):
        c, g, r = _refs(tmp_path)

        with pytest.raises(ValueError, match="different read counts"):
            SEQ.process_chunk(
                ([_fastq(_read())], [], _REGEX, _TARGET, -8, _WINDOW,
                 c, g, r, False))

    def test_every_length_test_below_the_padding_is_already_satisfied(self):
        """THE PIN, for three arcs at once.

        Both paths pad the extracted window up to ``expected_end`` with
        N before building the consensus, so every ``len(...) >=
        expected_end`` test below the padding is always true: the
        single-end one, the paired one, and the reverse-complement
        hint's.

        The tests are still right to keep -- the padding is what makes
        them true, and a window built without it would slice a short
        barcode that then fails to map, which reads as "this guide was
        not in the library" rather than "this read was too short".

        Pinned on the padding, and the arithmetic run: padding a string
        of any length up to a target gives exactly the target.
        """
        import inspect

        source = inspect.getsource(SEQ.process_chunk)
        for name in ("r1_seq", "r2_seq"):
            assert f"if len({name}) < expected_end:" in source, (
                f"{name} is no longer padded, so the length tests below it "
                f"can now be false")
            assert f"{name} += 'N' * (expected_end - len({name}))" in source

        for length in (0, 1, 29, 61, 62, 200):
            window = 62
            padded = "A" * length
            if len(padded) < window:
                padded += "N" * (window - len(padded))
            assert len(padded) >= window, (
                f"a {length}-base read padded to {len(padded)}, short of "
                f"{window}")


# ---------------------------------------------------------------------------
# fill_na over a frame that lacks the column being filled
# ---------------------------------------------------------------------------

class TestFillingUnmappedBarcodesWithTheirSequence:

    def test_an_unmapped_barcode_falls_back_to_its_own_sequence(self,
                                                                tmp_path):
        """The point of ``fill_na``: a barcode the reference does not name
        is still a barcode, and dropping it loses a real read."""
        c = _write_csv(tmp_path / "c.csv", ["TTTTTTTT"], ["other"])
        g = _write_csv(tmp_path / "g.csv", [_GRNA], ["sg0"])
        r = _write_csv(tmp_path / "r.csv", [_ROW], ["row0"])

        df, counts, _qc = SEQ.process_chunk(
            ([_fastq(_read())], _REGEX, _TARGET, -8, _WINDOW, c, g, r, True))

        assert len(df) == 1
        assert not counts.empty
        assert (counts["columnID"] == _COL).all(), (
            "the unmapped column was not filled with its own sequence")

    def test_without_fill_na_the_unmapped_barcode_stays_unnamed(self,
                                                                tmp_path):
        c = _write_csv(tmp_path / "c.csv", ["TTTTTTTT"], ["other"])
        g = _write_csv(tmp_path / "g.csv", [_GRNA], ["sg0"])
        r = _write_csv(tmp_path / "r.csv", [_ROW], ["row0"])

        df, _counts, _qc = SEQ.process_chunk(
            ([_fastq(_read())], _REGEX, _TARGET, -8, _WINDOW, c, g, r, False))

        assert len(df) == 1
        assert pd.isna(df.iloc[0]["columnID"])

    def test_each_fill_is_guarded_on_its_own_column(self):
        """THE PIN: three ``in df2.columns`` tests that cannot be false.

        The three fills are separately guarded rather than done in one
        pass, because the regex decides which barcode columns exist: a
        run configured with a two-group regex has no ``grna_name``
        column at all, and ``fillna`` on a column that is not there is a
        KeyError raised in the middle of a chunk loop over millions of
        reads.

        Today they cannot be false: the regex validation above refuses a
        pattern without a grna group and without a row and a column
        group, so all three columns always exist by the time the fills
        run. Run as the operation itself over a frame missing each
        column in turn, so the guard's shape and its consequence are
        both checked.
        """
        import inspect

        source = inspect.getsource(SEQ.process_chunk)
        for column in ("columnID", "rowID", "grna_name"):
            assert f"if '{column}' in df2.columns:" in source, (
                f"the {column} fill is no longer guarded on its own column")

        frame = pd.DataFrame({"column_sequence": [_COL],
                              "row_sequence": [_ROW],
                              "grna_sequence": [_GRNA]})
        for column, source_column in (("columnID", "column_sequence"),
                                      ("rowID", "row_sequence"),
                                      ("grna_name", "grna_sequence")):
            assert column not in frame.columns
            with pytest.raises(KeyError):
                frame[column].fillna(frame[source_column])


# ---------------------------------------------------------------------------
# graph_sequencing_stats -- a run with nowhere to write
# ---------------------------------------------------------------------------

class TestWhereTheThresholdFigureGoes:

    def _settings(self, tmp_path, dst):
        """A count table wide enough for the sweep to have somewhere to go.

        ``filter_column`` names the column the control wells are dropped
        by, and it is required: without it the function refuses rather
        than silently keeping controls in the fractions.
        """
        counts = tmp_path / "counts.csv"
        rng = np.random.default_rng(4)
        rows = []
        for row in range(8):
            for column in range(12):
                for guide in range(6):
                    rows.append({
                        "plateID": "p1",
                        "rowID": f"r{row + 1}",
                        "columnID": f"c{column + 1}",
                        "grna": f"g{guide}",
                        "count": int(rng.integers(1, 300)),
                    })
        pd.DataFrame(rows).to_csv(counts, index=False)
        return {"count_data": [str(counts)], "dst": dst,
                "min_count": 0, "target_unique_count": 3,
                "filter_column": "columnID", "control_wells": ["c1"],
                "log_x": False, "log_y": False}

    def test_the_figure_lands_beside_the_count_data(self, tmp_path):
        """Not where ``settings['dst']`` says.

        The destination is derived from the first count file's own
        directory, so the figure lands beside the table it describes
        whatever ``dst`` holds. Worth knowing rather than guessing: a
        caller that set ``dst`` and looked there would not find it.
        """
        elsewhere = tmp_path / "out"
        elsewhere.mkdir()

        SEQ.graph_sequencing_stats(self._settings(tmp_path, str(elsewhere)))

        beside = tmp_path / "results" / "fraction_threshold.pdf"
        assert beside.exists(), (
            "the threshold figure was not written beside the count data")
        assert not (elsewhere / "results" / "fraction_threshold.pdf").exists()

    def test_it_answers_a_threshold(self, tmp_path):
        threshold = SEQ.graph_sequencing_stats(
            self._settings(tmp_path, None))

        assert threshold is None or isinstance(threshold, float)

    def test_the_destination_guard_cannot_fire(self):
        """THE PIN: ``dst is None``.

        The only caller builds ``dst`` from
        ``os.path.dirname(settings['count_data'][0])``, which is a real
        directory for any file that was read -- so the None arm of the
        nested plotter is unreachable through it.

        Keeping it is right, because the plotter carries ``dst=None`` in
        its own signature and ``os.path.join(None, 'results')`` is a
        TypeError. The pin is on the derivation, which is what would have
        to change for the arm to become live.
        """
        import inspect

        source = inspect.getsource(SEQ.graph_sequencing_stats)
        assert "if dst is not None:" in source
        write = source.index("os.makedirs(fig_path, exist_ok=True)")
        guard = source.index("if dst is not None:")
        assert guard < write, "the destination is used before it is checked"
        assert "dst = os.path.dirname(settings['count_data'][0])" in source, (
            "the destination is no longer derived from the count file, so "
            "the None arm of the plotter may now be reachable")
