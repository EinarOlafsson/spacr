"""Table-driven tests of barcode decoding in ``spacr.sequencing``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: decoding a FASTQ chunk (single and paired, intact,
truncated, unmapped and ``fill_na`` reads, plus the orientation hint a chunk
that matches nothing prints), the helpers it leans on (reverse complement,
consensus base, the per-combination count CSV), labelling worker processes in
the run's resource record, the run-start announcements and refusals of
``generate_barecode_mapping``, and the fraction-threshold sweep of
``graph_sequencing_stats``.  Everything is synthetic, tiny and offline.
"""
from __future__ import annotations

import builtins
import os

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr import sequencing as SEQ  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Two barcode layouts.
#
# SHORT: <col:4>GGAA<grna:4><row:4> right after a palindromic GGATCC anchor,
#        so a paired mate is just the reverse complement of R1.
# LONG:  the historical 62-base layout with an 8-base lead before the anchor
#        (offset -8).
# ---------------------------------------------------------------------------

_SHORT = {
    "anchor": "GGATCC", "offset": 6, "window": 16,
    "regex": r"^(?P<columnID>.{4})GGAA(?P<grna>.{4})(?P<rowID>.{4})",
}
_GOOD = "AAAAGGAACCCCTTTT"
_CUT = "AAAAGGAACC"                          # ten payload bases, window is 16
_CUT_PADDED = "AAAAGGAACCNNNNNN"

_LONG = {
    "anchor": "TGCTGAAATAAAC", "offset": -8, "window": 62,
    "regex": (r"^(?P<columnID>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})"
              r"AACTT.*AGAAG(?P<rowID>.{8}).*"),
}
_LONG_READ = ("AAAACCCC" + "TGCTGAAATAAAC" + "A" * 20 + "AACTTAAAAGAAG"
              + "GGGGTTTT" + "AAAA")
_LONG_WINDOW = _LONG_READ[:62]

_SHORT_REFS = {"column": (["AAAA"], ["col0"]), "grna": (["CCCC"], ["sg0"]),
               "row": (["TTTT"], ["row0"])}
_LONG_REFS = {"column": (["AAAACCCC"], ["col0"]), "grna": (["A" * 20], ["sg0"]),
              "row": (["GGGGTTTT"], ["row0"])}

_COLUMNS = ["read", "column_sequence", "columnID", "row_sequence", "rowID",
            "grna_sequence", "grna_name"]


def _fastq(seq, name="r0"):
    """One four-line FASTQ record with a flat quality string."""
    return f"@{name}\n{seq}\n+\n{'I' * len(seq)}"


def _write_refs(folder, refs):
    """Write the three reference CSVs; return (column, grna, row) paths."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in ("column", "grna", "row"):
        seqs, names = refs[kind]
        path = folder / f"{kind}.csv"
        pd.DataFrame({"sequence": seqs, "name": names}).to_csv(path, index=False)
        paths.append(str(path))
    return paths


def _chunk(layout, refs_dir, reads, *, paired=False, fill_na=False,
           refs=None, r2_reads=None):
    """Build a 9-tuple (single) or 10-tuple (paired) chunk over ``reads``."""
    refs = refs or (_LONG_REFS if layout is _LONG else _SHORT_REFS)
    c, g, r = _write_refs(refs_dir, refs)
    r1 = [_fastq(seq, f"r{i}") for i, seq in enumerate(reads)]
    tail = (layout["regex"], layout["anchor"], layout["offset"],
            layout["window"], c, g, r, fill_na)
    if not paired:
        return (r1,) + tail
    if r2_reads is None:
        r2_reads = [SEQ.reverse_complement(seq) for seq in reads]
    r2 = [_fastq(seq, f"r{i}") for i, seq in enumerate(r2_reads)]
    return (r1, r2) + tail


def _rows(frame):
    """A frame as plain lists with every missing value as ``None``."""
    return frame.astype(object).where(frame.notna(), None).to_dict("list")


def _short(payload):
    """A SHORT-layout read carrying ``payload`` right after the anchor."""
    return "TT" + _SHORT["anchor"] + payload


def test_process_chunk_decodes_every_read_shape(tmp_path, subtests, capsys):
    """Annotated reads, per-well counts, QC row and stdout hint, per chunk shape."""
    # Every frame is (reads, column seqs, column names, row seqs, row names,
    # grna seqs, grna names); counts are (rowID, columnID, grna_name, count).
    def frame(*rows):
        cols = list(zip(*rows)) if rows else [()] * 7
        return {name: list(values) for name, values in zip(_COLUMNS, cols)}

    good = (_GOOD, "AAAA", "col0", "TTTT", "row0", "CCCC", "sg0")
    cut = (_CUT_PADDED, "AAAA", "col0", "NNNN", None, "CCNN", None)
    long_good = (_LONG_WINDOW, "AAAACCCC", "col0", "GGGGTTTT", "row0",
                 "A" * 20, "sg0")
    nothing_qc = {name: [0] for name in _COLUMNS} | {"total_reads": [0]}
    hint = "Are barcode sequences in the correct orientation?"
    flipped_ok = "Reverse complement of last sequence in chunk matched"
    flipped = SEQ.reverse_complement(_GOOD)

    cases = [
        # id, layout, reads, options, frame, counts, qc NaN overrides + total,
        # stdout fragments that must appear, fragments that must not.
        ("single intact and cut reads", _SHORT,
         [_short(_GOOD), _short(_CUT)], {},
         frame(good, cut), [["row0", "col0", "sg0", 1]],
         {"rowID": 1, "grna_name": 1, "total_reads": 2}, [], [hint]),
        ("single cut read kept by fill_na", _SHORT,
         [_short(_GOOD), _short(_CUT)], {"fill_na": True},
         frame(good, cut),
         [["NNNN", "col0", "CCNN", 1], ["row0", "col0", "sg0", 1]],
         {"rowID": 1, "grna_name": 1, "total_reads": 2}, [], [hint]),
        ("paired intact read", _SHORT, [_short(_GOOD)], {"paired": True},
         frame(good), [["row0", "col0", "sg0", 1]],
         {"total_reads": 1}, [], [hint]),
        ("paired cut read pads both mates", _SHORT, [_short(_CUT)],
         {"paired": True}, frame(cut), [],
         {"rowID": 1, "grna_name": 1, "total_reads": 1}, [], [hint]),
        ("unmapped column without fill_na stays unnamed", _SHORT,
         [_short(_GOOD)], {"refs": dict(_SHORT_REFS, column=(["GGGG"], ["x"]))},
         frame((_GOOD, "AAAA", None, "TTTT", "row0", "CCCC", "sg0")), [],
         {"columnID": 1, "total_reads": 1}, [], [hint]),
        ("unmapped column with fill_na counts under its sequence", _SHORT,
         [_short(_GOOD)],
         {"fill_na": True, "refs": dict(_SHORT_REFS, column=(["GGGG"], ["x"]))},
         frame((_GOOD, "AAAA", None, "TTTT", "row0", "CCCC", "sg0")),
         [["row0", "AAAA", "sg0", 1]],
         {"columnID": 1, "total_reads": 1}, [], [hint]),
        ("paired lane on the wrong strand says so", _SHORT, [_short(flipped)],
         {"paired": True}, frame(), [], {"total_reads": 0},
         [hint, f"Is {flipped} compatible with", flipped_ok], []),
        ("paired unrelated lane gets no strand hint", _SHORT,
         [_short("A" * 16)], {"paired": True}, frame(), [], {"total_reads": 0},
         [hint, f"Is {'A' * 16} compatible with"], [flipped_ok]),
        ("no anchor at all names no read", _SHORT, ["T" * 24], {}, frame(), [],
         {"total_reads": 0}, [hint, "Is None compatible with"], [flipped_ok]),
        ("long layout intact read", _LONG, [_LONG_READ], {},
         frame(long_good), [["row0", "col0", "sg0", 1]],
         {"total_reads": 1}, [], [hint]),
        ("long layout paired intact read", _LONG, [_LONG_READ],
         {"paired": True}, frame(long_good), [["row0", "col0", "sg0", 1]],
         {"total_reads": 1}, [], [hint]),
        ("long layout truncated read fails the regex", _LONG,
         [_LONG_READ[:30]], {}, frame(), [], {"total_reads": 0},
         [hint, "Is " + _LONG_READ[:30] + "N" * 32 + " compatible"],
         [flipped_ok]),
        ("long layout paired truncated reads decode nothing", _LONG,
         [_LONG_READ[:30]],
         {"paired": True, "r2_reads": [SEQ.reverse_complement(_LONG_READ)[:30]]},
         frame(), [], {"total_reads": 0}, [hint], []),
    ]
    for (case_id, layout, reads, options, want_frame, want_counts, want_qc,
         said_in, said_not) in cases:
        with subtests.test(msg=case_id):
            capsys.readouterr()
            df, counts, qc = SEQ.process_chunk(
                _chunk(layout, tmp_path / case_id.replace(" ", "_"), reads,
                       **options))
            said = capsys.readouterr().out

            assert list(df.columns) == _COLUMNS
            assert _rows(df) == want_frame
            assert counts.columns.tolist() == ["rowID", "columnID",
                                               "grna_name", "count"]
            assert counts.values.tolist() == want_counts
            expected_qc = dict(nothing_qc)
            expected_qc.update({k: [v] for k, v in want_qc.items()})
            assert qc.index.tolist() == ["NaN_Counts"]
            assert qc.to_dict("list") == expected_qc
            for fragment in said_in:
                assert fragment in said
            for fragment in said_not:
                assert fragment not in said


def test_process_chunk_refuses_mismatched_paired_chunks(tmp_path):
    """Paired chunks of different lengths are refused with both counts named."""
    chunk = _chunk(_SHORT, tmp_path, [_short(_GOOD)], paired=True,
                   r2_reads=[])
    with pytest.raises(ValueError,
                       match=r"different read counts: R1=1, R2=0\."):
        SEQ.process_chunk(chunk)


def test_sequence_helpers_reverse_pick_and_accumulate(tmp_path, subtests):
    """Reverse complement, consensus base and the per-combination count CSV."""
    for seq, expected in [("AAAC", "GTTT"), (_GOOD, "AAAAGGGGTTCCTTTT"),
                          ("ACGTN", "NACGT"), ("", "")]:
        with subtests.test(msg=f"reverse_complement {seq!r}"):
            assert SEQ.reverse_complement(seq) == expected
            assert SEQ.reverse_complement(expected) == seq

    for bases, expected in [
            ([("N", 40), ("A", 30)], "A"),     # an N never wins
            ([("C", 40), ("N", 30)], "C"),
            ([("G", 20), ("T", 35)], "T"),     # otherwise the higher quality
            ([("G", 35), ("T", 20)], "G"),
            ([("G", 30), ("T", 30)], "G")]:    # a tie keeps the first
        with subtests.test(msg=f"consensus {bases}"):
            assert SEQ.get_consensus_base(bases) == expected

    with subtests.test(msg="count CSV is created, then summed per combination"):
        csv = tmp_path / "uc.csv"
        first = pd.DataFrame({"rowID": ["r1", "r2"], "columnID": ["c1", "c1"],
                              "grna_name": ["g1", "g2"], "count": [5, 3]})
        SEQ.save_unique_combinations_to_csv(first, str(csv))
        assert pd.read_csv(csv).values.tolist() == [["r1", "c1", "g1", 5],
                                                    ["r2", "c1", "g2", 3]]
        later = pd.DataFrame({"rowID": ["r1", "r3"], "columnID": ["c1", "c2"],
                              "grna_name": ["g1", "g9"], "count": [5, 7]})
        SEQ.save_unique_combinations_to_csv(later, str(csv))
        assert pd.read_csv(csv).values.tolist() == [["r1", "c1", "g1", 10],
                                                    ["r2", "c1", "g2", 3],
                                                    ["r3", "c2", "g9", 7]]

    with subtests.test(msg="any number of barcode columns is grouped"):
        csv = tmp_path / "one_barcode.csv"
        frame = pd.DataFrame({"grna_name": ["g1"], "count": [2]})
        SEQ.save_unique_combinations_to_csv(frame, str(csv))
        SEQ.save_unique_combinations_to_csv(frame, str(csv))
        assert pd.read_csv(csv).values.tolist() == [["g1", 4]]


class _Process:
    """A multiprocessing child as far as the labeller looks at it."""

    def __init__(self, pid=None):
        self.pid = pid


def test_worker_processes_are_labelled_without_ever_stopping_the_run(
        monkeypatch, subtests):
    """Live children are registered with an int pid; every failure is swallowed."""
    from spacr import runctx

    def context(register):
        class _Context:
            @staticmethod
            def register_worker(kind, ident, pid):
                return register(kind, ident, pid)
        return _Context()

    def refuse(*_args):
        raise RuntimeError("the resource record is closed")

    cases = [
        # id, process, context factory (None = no run), expected calls
        ("live child is registered", _Process(4321), True,
         [("sequencing_chunk", 2, 4321)]),
        ("string pid recorded as int", _Process("99"), True,
         [("sequencing_chunk", 2, 99)]),
        ("child with no pid yet is skipped", _Process(None), True, []),
        ("outside a run nothing is recorded", _Process(1), None, []),
    ]
    for case_id, process, active, expected in cases:
        with subtests.test(msg=case_id):
            calls, consulted = [], []

            def current():
                consulted.append(True)
                return (context(lambda *a: calls.append(a))
                        if active else None)

            with monkeypatch.context() as mp:
                mp.setattr(runctx, "current_run_context", current)
                SEQ._label_resource_process(process, "sequencing_chunk", 2)
            assert calls == expected
            assert consulted == [True]
            assert all(type(call[2]) is int for call in calls)

    with subtests.test(msg="a context that refuses does not stop the run"):
        asked = []

        def hostile(*args):
            asked.append(args)
            refuse()

        with monkeypatch.context() as mp:
            mp.setattr(runctx, "current_run_context", lambda: context(hostile))
            assert SEQ._label_resource_process(_Process(7), "kind", 1) is None
        assert asked == [("kind", 1, 7)]

    with subtests.test(msg="a runctx that will not import is survived"):
        real_import = builtins.__import__
        attempted = []

        def no_runctx(name, globals_=None, locals_=None, fromlist=(), level=0):
            if "runctx" in name or "current_run_context" in (fromlist or ()):
                attempted.append(name)
                raise ImportError("runctx is unavailable")
            return real_import(name, globals_, locals_, fromlist, level)

        with monkeypatch.context() as mp:
            mp.setattr(builtins, "__import__", no_runctx)
            assert SEQ._label_resource_process(_Process(7), "kind", 1) is None
        assert attempted

    pools = [
        ("every worker is numbered from one",
         type("Pool", (), {"_pool": [_Process(10), _Process(11), _Process(12)]})(),
         [(10, "sequencing_chunk", 1), (11, "sequencing_chunk", 2),
          (12, "sequencing_chunk", 3)]),
        ("an empty pool labels nothing", type("Pool", (), {"_pool": []})(), []),
        ("a pool with no worker list", object(), []),
        ("a None worker list", type("Pool", (), {"_pool": None})(), []),
    ]
    for case_id, pool, expected in pools:
        with subtests.test(msg=f"pool: {case_id}"):
            labelled = []
            with monkeypatch.context() as mp:
                mp.setattr(SEQ, "_label_resource_process",
                           lambda p, kind, i: labelled.append((p.pid, kind, i)))
                SEQ._label_chunk_pool(pool)
            assert labelled == expected


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


def test_generate_barecode_mapping_announces_and_refuses_before_reading(
        tmp_path, monkeypatch, capsys, subtests):
    """The mismatch budget and barcode set are announced; a bad direction refused."""
    import spacr.io as IO
    import spacr.utils as U

    with subtests.test(msg="a non-zero budget is announced and armed"):
        src = tmp_path / "fastq"
        src.mkdir()
        with monkeypatch.context() as mp:
            mp.setattr(SEQ, "BARCODE_MISMATCHES", 0, raising=False)
            capsys.readouterr()
            assert SEQ.generate_barecode_mapping({
                "src": str(src), "mode": "single", "single_direction": "R1",
                "barcode_mismatches": 2, "save_h5": False, "n_jobs": 1,
            }) is None
            assert SEQ.BARCODE_MISMATCHES == 2
        printed = capsys.readouterr().out
        assert ("Barcodes may carry up to 2 mismatched base(s). A read within "
                "the budget of TWO barcodes is left unassigned rather than "
                "given to whichever was found first.") in printed

    with subtests.test(msg="a named barcode set is announced before parsing"):
        class Stop(Exception):
            pass

        seen = []

        def parse(src):
            seen.append(src)
            raise Stop

        with monkeypatch.context() as mp:
            mp.setattr(U, "save_settings", lambda *a, **k: None)
            mp.setattr(IO, "parse_gz_files", parse)
            capsys.readouterr()
            with pytest.raises(Stop):
                SEQ.generate_barecode_mapping({
                    "src": str(tmp_path),
                    "barcode_set": ["column", "grna", "row"]})
        assert seen == [str(tmp_path)]
        out = capsys.readouterr().out
        assert "Decoding 3 barcode(s): column, grna, row" in out
        assert "Barcodes may carry up to" not in out

    with subtests.test(msg="single_direction picks the mate it names, or refuses"):
        calls = []
        sample = tmp_path / "sample"
        sample.mkdir()
        with monkeypatch.context() as mp:
            mp.setattr(SEQ, "paired_read_chunked_processing",
                       lambda **kw: calls.append(("paired", kw)))
            mp.setattr(SEQ, "single_read_chunked_processing",
                       lambda **kw: calls.append(("single", kw)))
            mp.setattr(SEQ, "run_context", lambda *a, **k: _RunCtx())
            mp.setattr(SEQ, "_run_barcode_qc", lambda *a, **k: None)
            mp.setattr(U, "save_settings", lambda *a, **k: None)
            mp.setattr(IO, "parse_gz_files", lambda src: {"S1": {
                "R1": os.path.join(src, "S1_R1.fastq.gz"),
                "R2": os.path.join(src, "S1_R2.fastq.gz")}})

            SEQ.generate_barecode_mapping({"src": str(sample), "mode": "single",
                                           "single_direction": "R2"})
            assert [kind for kind, _ in calls] == ["single"]
            assert calls[0][1]["r1_file"] == os.path.join(str(sample),
                                                          "S1_R2.fastq.gz")
            assert calls[0][1]["r2_file"] is None

            before = sorted(os.listdir(sample))
            with pytest.raises(ValueError, match=(
                    r"^single_direction must be 'R1' or 'R2' when mode is "
                    r"'single'; got 'r2'$")):
                SEQ.generate_barecode_mapping({"src": str(sample),
                                               "mode": "single",
                                               "single_direction": "r2"})
        assert len(calls) == 1
        assert sorted(os.listdir(sample)) == before


def test_graph_sequencing_stats_picks_the_grid_point_nearest_the_target(
        tmp_path, monkeypatch, subtests):
    """The sweep returns the first grid threshold nearest the target, beside the counts."""
    # Every well: one guide at 100 reads and three at 3, so the fractions are
    # 100/109 and 3/109.  A target of 2 unique guides is never hit exactly:
    # below 3/109 a well keeps 4 guides (distance 2), above it 1 (distance 1).
    rows = [{"plateID": "p1", "rowID": row, "columnID": f"c{column}",
             "grna": f"sg{guide}", "count": 100 if guide == 0 else 3}
            for row in "ABCD" for column in range(1, 5) for guide in range(4)]
    grid = np.linspace(0.001, 0.99, 1000)
    first_above = float(grid[np.argmax(grid > 3 / 109)])

    cases = [
        ("a bare filename writes beside the working directory", "counts.csv",
         2, first_above, "results/fraction_threshold.pdf"),
        ("a list of paths writes beside the first table",
         ["data/counts.csv"], 2, first_above,
         "data/results/fraction_threshold.pdf"),
        ("a target every well already meets keeps the lowest threshold",
         ["data4/counts.csv"], 4, float(grid[0]),
         "data4/results/fraction_threshold.pdf"),
    ]
    monkeypatch.chdir(tmp_path)
    for case_id, count_data, target, expected, figure in cases:
        with subtests.test(msg=case_id):
            first = count_data if isinstance(count_data, str) else count_data[0]
            os.makedirs(os.path.dirname(first) or ".", exist_ok=True)
            pd.DataFrame(rows).to_csv(first, index=False)
            elsewhere = tmp_path / "elsewhere"
            elsewhere.mkdir(exist_ok=True)

            threshold = SEQ.graph_sequencing_stats({
                "count_data": count_data, "target_unique_count": target,
                "dst": str(elsewhere), "filter_column": "columnID",
                "analysis_excluded_wells": ["c4"]})

            assert float(threshold) == expected
            assert (tmp_path / figure).is_file()
            assert not (elsewhere / "results").exists()
