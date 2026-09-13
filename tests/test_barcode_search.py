"""The barcode detector has to survive reads that contain nothing.

The point of `spacr.barcode_search` is not that it finds barcodes.  Anything
finds barcodes: a short sequence turns up in a long read by coincidence often
enough that a naive scan reports several percent of reads as hits for a table
that is not in the run at all.  The point is that it declines to find them.  So
the negative control here carries more weight than the planted one -- random
reads searched against a table of thirty-two eight base barcodes really do
match about seven percent of the time, and a detector that calls that a finding
would send someone chasing a bug that does not exist or, worse, would configure
a mapping that silently produces nothing.
"""

from __future__ import annotations

import gzip
import os
import random

import numpy as np
import pytest

from spacr import barcode_search as bs
from tests.resource_capabilities import paths_available

# The library layout the real paired run uses, which the synthetic reads copy:
# an eight base column barcode at the very start, the anchor, the guide, a
# fixed stretch, and an eight base row barcode.
ANCHOR = "TGCTGTTTCCAGCATAGCTCTTAAAC"
SPACER = "AACTTGACATCCCCATTTACCAGAAG"
READ_LENGTH = 150


def _random_bases(rng, length, alphabet="ACGT"):
    """Return a random nucleotide string of the wanted length."""
    return "".join(rng.choice(alphabet) for _ in range(length))


def _make_table(rng, count, length, prefix, alphabet="ACGT"):
    """Return a table of distinct random barcodes with predictable names."""
    sequences = {}
    while len(sequences) < count:
        sequences.setdefault(
            _random_bases(rng, length, alphabet), f"{prefix}{len(sequences) + 1}")
    return sequences


def _write_table(path, sequences):
    """Write a barcode table in the shape the shipped reference files use."""
    with open(path, "w") as handle:
        handle.write("name,sequence\n")
        for sequence, name in sequences.items():
            handle.write(f"{name},{sequence}\n")
    return str(path)


def _write_fastq(path, reads, compress=False):
    """Write reads as a sequencing file, gzip compressed when asked."""
    opener = gzip.open if compress else open
    with opener(str(path), "wt") as handle:
        for index, read in enumerate(reads):
            handle.write(f"@read{index}\n{read}\n+\n{'I' * len(read)}\n")
    return str(path)


def _planted_reads(rng, columns, guides, rows, count=4000):
    """Return reads laid out the way the real construct is, in R1 orientation.

    The column barcode sits at offset zero, the anchor follows it, the guide
    follows the anchor, and the row barcode follows a fixed spacer.  The tail is
    filled with random bases so the read reaches its full length.
    """
    reads = []
    for _ in range(count):
        column = rng.choice(sorted(columns))
        guide = rng.choice(sorted(guides))
        row = rng.choice(sorted(rows))
        read = column + ANCHOR + guide + SPACER + row
        read = read + _random_bases(rng, max(READ_LENGTH - len(read), 0))
        reads.append(read[:READ_LENGTH])
    return reads


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """Build a synthetic paired run whose layout is known exactly.

    The guides are planted reverse complemented in the first mate and plain in
    the second, which is what a real mate pair read from opposite ends of the
    same fragment looks like and is the arrangement that silently produced no
    mapped counts in the failure this module exists to prevent.
    """
    rng = random.Random(20260912)
    directory = tmp_path_factory.mktemp("planted")
    columns = _make_table(rng, 24, 8, "c")
    rows = _make_table(rng, 16, 8, "r")
    guides = _make_table(rng, 300, 20, "g")
    r1_reads = _planted_reads(rng, columns, guides, rows)
    r2_reads = [bs.reverse_complement(read) for read in r1_reads]
    return {
        "r1": _write_fastq(directory / "S_R1_001.fastq.gz", r1_reads, compress=True),
        "r2": _write_fastq(directory / "S_R2_001.fastq.gz", r2_reads, compress=True),
        "column_csv": _write_table(directory / "column_barcodes.csv", columns),
        "row_csv": _write_table(directory / "row_barcodes.csv", rows),
        "grna_csv": _write_table(directory / "grna_barcodes.csv", guides),
        "reads": r1_reads,
    }


@pytest.fixture(scope="module")
def planted_tables(planted):
    """Load the reference tables belonging to the synthetic run."""
    return bs.load_barcode_tables(
        [planted["row_csv"], planted["column_csv"], planted["grna_csv"]])


# ---------------------------------------------------------------------------
# The negative control, which is the test that matters
# ---------------------------------------------------------------------------


def test_random_reads_do_not_contain_a_short_barcode_table(tmp_path):
    """Coincidental matches at several percent are reported as absent.

    Thirty-two eight base barcodes scanned across a hundred and fifty base read
    match by chance often enough to look like a finding, and this is the shape
    of the false positive that started the whole feature.
    """
    rng = random.Random(7)
    table = bs.BarcodeTable(
        name="row", role="row", sequences=_make_table(rng, 32, 8, "r"))
    reads = [_random_bases(rng, READ_LENGTH) for _ in range(4000)]
    path = _write_fastq(tmp_path / "noise.fastq", reads)

    report = bs.search_barcodes({"R1": path}, [table], max_reads=4000)
    finding = report.for_table("row")[0]

    assert finding.hits > 0, "the coincidental matches this guards against"
    assert 0.03 < finding.observed_rate < 0.12
    assert finding.expected_rate == pytest.approx(finding.observed_rate, abs=0.02)
    for item in report.findings:
        assert item.verdict == bs.ABSENT, item.reason
    assert report.best_for_role("row") is None


def test_every_orientation_of_random_reads_stays_absent(tmp_path):
    """A table of long barcodes never matches noise at all."""
    rng = random.Random(11)
    table = bs.BarcodeTable(
        name="grna", role="grna", sequences=_make_table(rng, 500, 20, "g"))
    reads = [_random_bases(rng, READ_LENGTH) for _ in range(2000)]
    path = _write_fastq(tmp_path / "noise.fastq", reads)

    report = bs.search_barcodes({"R1": path}, [table], max_reads=2000)

    for item in report.findings:
        assert item.hits == 0
        assert item.verdict == bs.ABSENT
        assert "not one of" in item.reason


def test_a_table_seen_in_too_few_reads_is_not_called_present(tmp_path):
    """A trace of another sample is honest evidence and useless settings.

    Index hopping puts a real, hugely enriched signal into a fraction of a
    percent of reads.  Calling that absent would be a lie and calling it present
    would configure a mapping onto a handful of reads, so it is neither.
    """
    rng = random.Random(13)
    guides = _make_table(rng, 300, 20, "g")
    table = bs.BarcodeTable(name="grna", role="grna", sequences=guides)
    reads = [_random_bases(rng, READ_LENGTH) for _ in range(4000)]
    for index in range(8):
        guide = sorted(guides)[index]
        reads[index] = (
            _random_bases(rng, 34) + guide
            + _random_bases(rng, READ_LENGTH - 34 - len(guide)))
    path = _write_fastq(tmp_path / "hopped.fastq", reads)

    report = bs.search_barcodes({"R1": path}, [table], max_reads=4000)
    best = max(report.for_table("grna"), key=lambda item: item.observed_rate)

    assert best.hits == 8
    assert best.enrichment > 100
    assert best.verdict == bs.INDETERMINATE
    assert "too little to map from" in best.reason
    assert report.best_for_role("grna") is None


def test_an_adapter_shaped_match_is_refused_however_enriched(tmp_path):
    """Enrichment says the sequence is there, not that it is a plate barcode.

    Two members of a table planted at scattered positions, the way a sequence
    that resembles the sequencing adapter behaves, clear the chance test by a
    wide margin and still cannot yield an extraction window.
    """
    rng = random.Random(17)
    columns = _make_table(rng, 24, 8, "c")
    table = bs.BarcodeTable(name="column", role="column", sequences=columns)
    impostors = sorted(columns)[:2]
    reads = []
    for index in range(4000):
        offset = rng.choice([40, 41, 85, 86, 128, 129])
        filler = _random_bases(rng, READ_LENGTH)
        planted = impostors[index % 2]
        reads.append(filler[:offset] + planted + filler[offset + 8:])
    path = _write_fastq(tmp_path / "adapter.fastq", reads)

    report = bs.search_barcodes({"R1": path}, [table], max_reads=4000)
    best = max(report.for_table("column"), key=lambda item: item.observed_rate)

    assert best.observed_rate > 0.9
    assert best.enrichment > bs.MIN_ENRICHMENT
    assert best.offset_span > bs.MAX_OFFSET_SPAN
    assert best.verdict == bs.INDETERMINATE
    assert "adapter" in best.reason
    assert report.best_for_role("column") is None


# ---------------------------------------------------------------------------
# The planted case
# ---------------------------------------------------------------------------


def test_a_planted_barcode_is_found_at_its_orientation_and_offset(planted, planted_tables):
    """The layout that was planted is the layout that comes back out."""
    report = bs.search_barcodes(
        {"R1": planted["r1"], "R2": planted["r2"]},
        planted_tables, max_reads=2000, anchor=ANCHOR)

    column = report.best_for_role("column")
    guide = report.best_for_role("grna")
    row = report.best_for_role("row")
    assert column is not None and guide is not None and row is not None

    r1_column = [
        item for item in report.findings
        if item.role == "column" and item.file_label == "R1"
        and item.verdict == bs.PRESENT
    ]
    assert len(r1_column) == 1
    assert r1_column[0].orientation == bs.AS_GIVEN
    assert r1_column[0].offset_start == 0
    assert r1_column[0].offset_span == 1

    r1_guide = [
        item for item in report.findings
        if item.role == "grna" and item.file_label == "R1"
        and item.verdict == bs.PRESENT
    ]
    assert len(r1_guide) == 1
    assert r1_guide[0].orientation == bs.AS_GIVEN
    assert r1_guide[0].offset_start == len(ANCHOR) + 8

    r2_guide = [
        item for item in report.findings
        if item.role == "grna" and item.file_label == "R2"
        and item.verdict == bs.PRESENT
    ]
    assert len(r2_guide) == 1
    assert r2_guide[0].orientation == bs.REVERSE_COMPLEMENT


def test_a_present_finding_names_the_evidence_behind_it(planted, planted_tables):
    """Every rate arrives with the rate expected by coincidence beside it."""
    report = bs.search_barcodes(
        {"R1": planted["r1"]}, planted_tables, max_reads=2000)
    column = report.best_for_role("column")

    assert column.observed_rate > 0.95
    assert 0.01 < column.expected_rate < 0.10
    assert column.enrichment == pytest.approx(
        column.observed_rate / column.expected_rate)
    assert column.distinct_barcodes_seen == column.table_size
    assert "expected by chance" in column.reason


def test_the_offset_histogram_shows_where_the_barcode_sits(planted, planted_tables):
    """The offset distribution is what lets a window be proposed at all."""
    report = bs.search_barcodes(
        {"R1": planted["r1"]}, planted_tables, max_reads=2000)
    guide = report.best_for_role("grna")

    histogram = guide.offset_histogram(READ_LENGTH)
    assert isinstance(histogram, np.ndarray)
    assert len(histogram) == READ_LENGTH
    assert int(histogram.argmax()) == len(ANCHOR) + 8
    assert histogram.sum() == guide.hits
    assert guide.modal_offset == len(ANCHOR) + 8


# ---------------------------------------------------------------------------
# The chance baseline itself
# ---------------------------------------------------------------------------


def test_the_chance_rate_matches_what_random_reads_actually_do():
    """The formula is checked against reads rather than trusted."""
    rng = random.Random(3)
    table = bs.BarcodeTable(name="row", sequences=_make_table(rng, 32, 8, "r"))
    matcher = table.matcher(bs.AS_GIVEN)
    reads = [_random_bases(rng, READ_LENGTH) for _ in range(6000)]
    measured = sum(
        1 for read in reads if matcher.find_first(read) is not None) / len(reads)

    assert bs.expected_chance_rate(table, READ_LENGTH) == pytest.approx(
        measured, abs=0.015)


def test_the_chance_rate_grows_with_the_table_and_the_read():
    """More barcodes and longer reads both make coincidence likelier."""
    rng = random.Random(5)
    small = bs.BarcodeTable(name="s", sequences=_make_table(rng, 8, 8, "s"))
    large = bs.BarcodeTable(name="l", sequences=_make_table(rng, 64, 8, "l"))

    assert bs.expected_chance_rate(small, 150) < bs.expected_chance_rate(large, 150)
    assert bs.expected_chance_rate(large, 50) < bs.expected_chance_rate(large, 150)
    assert bs.expected_chance_rate(large, 4) == 0.0


def test_the_chance_rate_falls_away_for_longer_barcodes():
    """A twenty base barcode is effectively never met by accident."""
    rng = random.Random(9)
    short = bs.BarcodeTable(name="s", sequences=_make_table(rng, 32, 8, "s"))
    long_table = bs.BarcodeTable(name="l", sequences=_make_table(rng, 32, 20, "l"))

    assert bs.expected_chance_rate(short, 150) > 0.05
    assert bs.expected_chance_rate(long_table, 150) < 1e-6


def test_the_chance_rate_averages_over_mixed_read_lengths():
    """Reads of different lengths are weighted by how many there are."""
    rng = random.Random(15)
    table = bs.BarcodeTable(name="t", sequences=_make_table(rng, 32, 8, "t"))
    short = bs.expected_chance_rate(table, 60)
    long_reads = bs.expected_chance_rate(table, 150)
    mixed = bs.expected_chance_rate(table, {60: 1, 150: 1})

    assert short < mixed < long_reads
    assert mixed == pytest.approx((short + long_reads) / 2)
    assert bs.expected_chance_rate(table, [60, 150]) == pytest.approx(mixed)
    assert bs.expected_chance_rate(table, []) == 0.0


# ---------------------------------------------------------------------------
# Running it live
# ---------------------------------------------------------------------------


def test_a_partial_search_reports_the_same_shape_as_a_finished_one(planted, planted_tables):
    """Every refinement can be shown, so none of them is a special case."""
    steps = list(bs.iter_barcode_search(
        {"R1": planted["r1"]}, planted_tables, max_reads=2000, chunk_reads=500))

    assert len(steps) == 4
    assert [step.reads_by_file["R1"] for step in steps] == [500, 1000, 1500, 2000]
    assert [step.complete for step in steps] == [False, False, False, True]
    shapes = {len(step.findings) for step in steps}
    assert shapes == {len(planted_tables) * 2}
    for step in steps:
        assert step.format_table().count("\n") == len(step.findings) + 1


def test_the_running_counts_only_ever_grow(planted, planted_tables):
    """Partial results refine rather than flicker, which is what live means."""
    previous = None
    for step in bs.iter_barcode_search(
            {"R1": planted["r1"]}, planted_tables,
            max_reads=2000, chunk_reads=250):
        counts = {
            (item.table, item.orientation): item.hits for item in step.findings
        }
        if previous is not None:
            for key, value in counts.items():
                assert value >= previous[key]
        previous = counts


def test_a_search_stopped_early_still_says_what_it_rests_on(planted, planted_tables):
    """Walking away from the generator is a supported thing to do."""
    steps = bs.iter_barcode_search(
        {"R1": planted["r1"]}, planted_tables, max_reads=5000, chunk_reads=300)
    first = next(steps)
    second = next(steps)
    steps.close()

    assert first.reads == 300
    assert second.reads == 600
    assert first.complete is False
    assert second.for_table("column_barcodes")[0].reads == 600


def test_too_few_reads_produce_no_verdict_rather_than_a_wrong_one(planted, planted_tables):
    """A handful of reads decides nothing, and says so."""
    steps = bs.iter_barcode_search(
        {"R1": planted["r1"]}, planted_tables, max_reads=2000, chunk_reads=100)
    first = next(steps)
    steps.close()

    assert first.reads == 100
    for item in first.findings:
        assert item.verdict == bs.INDETERMINATE
        assert "too few" in item.reason


def test_the_finished_generator_agrees_with_the_one_shot_search(planted, planted_tables):
    """The convenience wrapper is the generator and nothing else."""
    stepped = None
    for stepped in bs.iter_barcode_search(
            {"R1": planted["r1"]}, planted_tables,
            max_reads=1500, chunk_reads=500):
        pass
    direct = bs.search_barcodes(
        {"R1": planted["r1"]}, planted_tables, max_reads=1500, chunk_reads=1500)

    assert stepped.findings == direct.findings
    assert stepped.reads_by_file == direct.reads_by_file


def test_an_empty_file_still_produces_a_report(tmp_path, planted_tables):
    """A run against nothing reports nothing rather than raising."""
    path = _write_fastq(tmp_path / "empty.fastq", [])

    report = bs.search_barcodes({"R1": path}, planted_tables, max_reads=100)

    assert report.reads == 0
    assert report.read_length_by_file["R1"] == 0.0
    assert len(report.findings) == len(planted_tables) * 2
    for item in report.findings:
        assert item.verdict == bs.INDETERMINATE
        assert item.expected_rate == 0.0
        assert item.enrichment == 0.0


@pytest.mark.parametrize("bad", [{"max_reads": 0}, {"chunk_reads": 0}])
def test_a_search_budget_has_to_be_positive(planted, planted_tables, bad):
    """A budget of nothing is a mistake rather than an instant answer."""
    with pytest.raises(ValueError):
        next(bs.iter_barcode_search(
            {"R1": planted["r1"]}, planted_tables, **bad))


def test_a_search_needs_a_file_to_read():
    """Being handed no files is a mistake worth naming."""
    with pytest.raises(ValueError, match="at least one"):
        next(bs.iter_barcode_search([], []))


# ---------------------------------------------------------------------------
# Reading the files
# ---------------------------------------------------------------------------


def test_reading_stops_at_the_requested_number_of_reads(planted):
    """These files are gigabytes on a network share and are never read whole."""
    assert len(bs.sample_fastq_reads(planted["r1"], 25)) == 25
    assert len(tuple(bs.iter_fastq_reads(planted["r1"], limit=1))) == 1
    assert tuple(bs.iter_fastq_reads(planted["r1"], limit=0)) == ()


def test_plain_and_compressed_files_read_the_same(tmp_path):
    """A run is not required to arrive compressed."""
    reads = ["ACGTACGTAA", "TTTTGGGGCC"]
    plain = _write_fastq(tmp_path / "reads.fastq", reads)
    zipped = _write_fastq(tmp_path / "reads.fastq.gz", reads, compress=True)

    assert bs.sample_fastq_reads(plain, 10) == tuple(reads)
    assert bs.sample_fastq_reads(zipped, 10) == tuple(reads)


def test_a_truncated_final_record_is_dropped_rather_than_raising(tmp_path):
    """Copying a file that is still being written leaves a partial record."""
    path = tmp_path / "cut.fastq"
    path.write_text("@a\nACGT\n+\nIIII\n@b\nTTTT\n+\n")

    assert bs.sample_fastq_reads(str(path), 10) == ("ACGT",)


def test_files_are_labelled_by_their_mate_when_they_can_be(planted, planted_tables):
    """A report names the mates the way the person reading it does."""
    report = bs.search_barcodes(
        [planted["r1"], planted["r2"]], planted_tables, max_reads=200)

    assert set(report.reads_by_file) == {"R1", "R2"}


def test_a_single_file_can_be_given_on_its_own(planted, planted_tables):
    """One file is a legitimate search, not a sequence of length one."""
    report = bs.search_barcodes(planted["r1"], planted_tables, max_reads=200)

    assert list(report.reads_by_file) == ["R1"]
    assert report.reads == 200


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------


def test_a_reference_table_is_read_with_its_names(planted):
    """The names are what make a report readable, so they are kept."""
    table = bs.load_barcode_table(planted["column_csv"])

    assert table.size == 24
    assert table.role == "column"
    assert table.path == planted["column_csv"]
    assert set(table.length_counts) == {8}
    assert all(name.startswith("c") for name in table.sequences.values())


def test_a_table_without_a_sequence_column_is_refused(tmp_path):
    """A file of the wrong shape is named rather than silently empty."""
    path = tmp_path / "wrong.csv"
    path.write_text("name,barcode\nc1,ACGTACGT\n")

    with pytest.raises(ValueError, match="sequence"):
        bs.load_barcode_table(str(path))


def test_a_row_without_a_sequence_is_skipped(tmp_path):
    """A trailing or half filled row is a normal thing to find in a csv."""
    path = tmp_path / "gappy.csv"
    path.write_text("name,sequence\nc1,\nc2,ACGTACGT\nc3,   \n")

    table = bs.load_barcode_table(str(path), role="column")

    assert table.sequences == {"ACGTACGT": "c2"}


def test_an_empty_table_is_refused(tmp_path):
    """A header alone would otherwise search for nothing and find nothing."""
    path = tmp_path / "empty.csv"
    path.write_text("name,sequence\n\n")

    with pytest.raises(ValueError, match="no barcode sequences"):
        bs.load_barcode_table(str(path))


def test_a_table_without_names_still_loads(tmp_path):
    """A bare list of sequences is usable, just less readable."""
    path = tmp_path / "bare.csv"
    path.write_text("sequence\nACGTACGT\nTTTTGGGG\n")

    table = bs.load_barcode_table(str(path), name="bare", role="column")

    assert table.size == 2
    assert sorted(table.sequences.values()) == ["row1", "row2"]
    assert table.role == "column"


@pytest.mark.parametrize(
    "filename,role",
    [
        ("grna_barcodes.csv", "grna"),
        ("primers_3_column_barecodes.csv", "column"),
        ("primers_2_row_barcodes.csv", "row"),
        ("barcodes_grna.csv", "grna"),
        ("something_else.csv", None),
    ],
)
def test_the_role_of_a_table_is_guessed_from_its_name(filename, role):
    """The tables carry no field saying what they are, so the name is it."""
    assert bs.infer_barcode_role(f"/somewhere/{filename}") == role


def test_tables_can_be_loaded_by_the_label_they_should_carry(planted):
    """Two tables can fill one role, so the label has to be separable."""
    tables = bs.load_barcode_tables(
        {"plate one": planted["column_csv"], "plate two": planted["row_csv"]})

    assert [table.name for table in tables] == ["plate one", "plate two"]


def test_a_run_may_carry_more_than_three_kinds_of_barcode(planted, tmp_path):
    """Nothing in the engine is built around there being exactly three."""
    rng = random.Random(21)
    extras = [
        bs.BarcodeTable(
            name=f"extra{index}", role=f"extra{index}",
            sequences=_make_table(rng, 12, 9, f"x{index}"))
        for index in range(4)
    ]
    tables = list(bs.load_barcode_tables([planted["column_csv"]])) + extras

    report = bs.search_barcodes({"R1": planted["r1"]}, tables, max_reads=600)

    assert len(report.findings) == len(tables) * 2
    assert len(report.roles()) == 5
    assert report.best_for_role("column") is not None
    for index in range(4):
        assert report.best_for_role(f"extra{index}") is None


def test_flipping_a_table_twice_returns_the_original(planted):
    """The orientation is a property of the table, not a mutation of it."""
    table = bs.load_barcode_table(planted["row_csv"])
    flipped = table.oriented(bs.REVERSE_COMPLEMENT)

    assert set(flipped.sequences) != set(table.sequences)
    assert set(flipped.oriented(bs.REVERSE_COMPLEMENT).sequences) == set(
        table.sequences)
    assert table.oriented(bs.AS_GIVEN) is table


def test_an_unknown_orientation_is_refused(planted):
    """Two orientations exist and a third would quietly search nothing."""
    table = bs.load_barcode_table(planted["row_csv"])

    with pytest.raises(ValueError, match="orientation must be"):
        table.oriented("sideways")


@pytest.mark.parametrize(
    "sequence,flipped",
    [("ACGT", "ACGT"), ("AAAA", "TTTT"), ("ACGTN", "NACGT"), ("", "")],
)
def test_reverse_complement_leaves_an_unresolved_base_alone(sequence, flipped):
    """A read with a dead cycle in it still has to survive the flip."""
    assert bs.reverse_complement(sequence) == flipped


# ---------------------------------------------------------------------------
# Turning findings into settings
# ---------------------------------------------------------------------------


def test_the_proposal_names_the_orientation_the_run_will_need(planted, planted_tables):
    """A guide planted flipped in the first mate has to come back as flipped."""
    report = bs.search_barcodes(
        {"R1": planted["r1"], "R2": planted["r2"]},
        planted_tables, max_reads=2000, anchor=ANCHOR)
    proposal = bs.propose_map_barcodes_settings(report)

    assert proposal.reference_file == "R1"
    assert proposal.orientations["column"] == bs.AS_GIVEN
    assert proposal.orientations["row"] == bs.AS_GIVEN
    assert proposal.orientations["grna"] == bs.AS_GIVEN
    assert proposal.unresolved_roles == ()
    assert proposal.settings["mode"] == "paired"
    assert proposal.settings["column_csv"] == planted["column_csv"]
    assert proposal.settings["grna_csv"] == planted["grna_csv"]


def test_the_proposal_places_the_window_against_the_anchor(planted, planted_tables):
    """The window the mapping run cuts is measured, not guessed."""
    report = bs.search_barcodes(
        {"R1": planted["r1"]}, planted_tables, max_reads=2000, anchor=ANCHOR)
    proposal = bs.propose_map_barcodes_settings(report)

    # The column barcode starts the read and the anchor follows it, so the
    # window has to begin eight bases before the anchor and reach the end of
    # the row barcode.
    assert proposal.settings["offset_start"] == -8
    expected_end = 8 + len(ANCHOR) + 20 + len(SPACER) + 8
    assert proposal.settings["window_length"] == expected_end


def test_a_run_read_from_one_mate_is_proposed_as_a_single_read_run(planted, planted_tables):
    """Only one mate was searched, so only one mate can be read."""
    report = bs.search_barcodes(
        {"R2": planted["r2"]}, planted_tables, max_reads=2000, anchor=ANCHOR)
    proposal = bs.propose_map_barcodes_settings(report)

    assert proposal.settings["mode"] == "single"
    assert proposal.settings["single_direction"] == "R2"
    assert set(proposal.reverse_complement_needed) == {"row", "column", "grna"}


def test_a_role_nothing_established_is_left_alone_and_named(planted, tmp_path):
    """Guessing here is how a run maps nothing and says it went fine."""
    rng = random.Random(23)
    wrong = _write_table(
        tmp_path / "other_row_barcodes.csv", _make_table(rng, 32, 8, "r"))
    tables = bs.load_barcode_tables([planted["column_csv"], wrong])
    report = bs.search_barcodes(
        {"R1": planted["r1"]}, tables, max_reads=2000, anchor=ANCHOR)
    base = {"row_csv": "/original/row.csv"}
    proposal = bs.propose_map_barcodes_settings(report, base_settings=base)

    assert proposal.unresolved_roles == ("row",)
    assert proposal.settings["row_csv"] == "/original/row.csv"
    assert base == {"row_csv": "/original/row.csv"}
    assert any("no reference table established" in note for note in proposal.notes)


def test_the_better_of_two_tables_for_one_role_is_the_one_proposed(planted, tmp_path):
    """The real failure was a run pointed at the wrong reference file."""
    rng = random.Random(29)
    decoy = _write_table(
        tmp_path / "decoy_column_barcodes.csv", _make_table(rng, 24, 8, "d"))
    tables = bs.load_barcode_tables([decoy, planted["column_csv"]])
    report = bs.search_barcodes(
        {"R1": planted["r1"]}, tables, max_reads=2000, anchor=ANCHOR)
    proposal = bs.propose_map_barcodes_settings(report)

    assert proposal.settings["column_csv"] == planted["column_csv"]


def test_a_proposal_without_an_anchor_leaves_the_window_alone(planted, planted_tables):
    """The window is placed against the anchor or it is not placed."""
    report = bs.search_barcodes(
        {"R1": planted["r1"]}, planted_tables, max_reads=2000)
    proposal = bs.propose_map_barcodes_settings(
        report, base_settings={"offset_start": -8, "window_length": 89})

    assert proposal.settings["offset_start"] == -8
    assert proposal.settings["window_length"] == 89
    assert any("no anchor sequence" in note for note in proposal.notes)


def test_a_proposal_needs_a_report_with_files_in_it(planted_tables):
    """An empty report cannot say which orientation anything is in."""
    empty = bs.BarcodeSearchReport(
        findings=(), reads_by_file={}, read_length_by_file={})

    with pytest.raises(ValueError, match="no sequencing files"):
        bs.propose_map_barcodes_settings(empty)


def test_the_proposal_only_touches_keys_the_mapping_run_reads(planted, planted_tables):
    """The dictionary is handed straight on, so it must stay valid."""
    from spacr.settings import set_default_generate_barecode_mapping

    base = set_default_generate_barecode_mapping({})
    report = bs.search_barcodes(
        {"R1": planted["r1"], "R2": planted["r2"]},
        planted_tables, max_reads=2000, anchor=ANCHOR)
    proposal = bs.propose_map_barcodes_settings(report, base_settings=base)

    assert set(proposal.settings) == set(base)


# ---------------------------------------------------------------------------
# Showing the reads
# ---------------------------------------------------------------------------


def test_a_read_is_annotated_so_each_barcode_can_be_coloured(planted, planted_tables):
    """Someone believes a verdict by seeing the barcodes inside the reads."""
    read = planted["reads"][0]
    hits = bs.annotate_read(read, planted_tables)

    roles = [hit.role for hit in hits]
    assert "column" in roles and "grna" in roles and "row" in roles
    assert [hit.start for hit in hits] == sorted(hit.start for hit in hits)
    for hit in hits:
        assert read[hit.start:hit.end]
        assert hit.end > hit.start
    column = next(hit for hit in hits if hit.role == "column")
    assert column.start == 0 and column.end == 8


def test_annotated_stretches_never_overlap(planted, planted_tables):
    """Two colours cannot be painted over the same base."""
    for read in planted["reads"][:50]:
        hits = bs.annotate_read(read, planted_tables)
        for earlier, later in zip(hits, hits[1:]):
            assert earlier.end <= later.start


def test_different_barcodes_of_one_table_get_different_colours(planted, planted_tables):
    """The maintainer asked for a colour per barcode, not per table."""
    indexes = {}
    for read in planted["reads"][:200]:
        for hit in bs.annotate_read(read, planted_tables):
            if hit.role == "column":
                indexes.setdefault(hit.barcode, set()).add(hit.colour_index)

    assert len(indexes) > 1
    assert all(len(values) == 1 for values in indexes.values())
    assert len({next(iter(v)) for v in indexes.values()}) == len(indexes)


def test_a_colour_survives_the_orientation_it_was_found_in(planted, planted_tables):
    """The same barcode is the same colour in either mate."""
    read = planted["reads"][0]
    flipped = bs.reverse_complement(read)
    forward = {
        hit.barcode: hit.colour_index for hit in bs.annotate_read(read, planted_tables)
    }
    backward = {
        hit.barcode: hit.colour_index
        for hit in bs.annotate_read(flipped, planted_tables)
    }

    assert forward
    assert backward
    for barcode, colour in backward.items():
        assert forward[barcode] == colour


def test_annotation_can_be_confined_to_one_orientation(planted, planted_tables):
    """A search that settled the orientation should not keep guessing."""
    flipped = bs.reverse_complement(planted["reads"][0])
    chosen = {table.name: bs.AS_GIVEN for table in planted_tables}

    assert bs.annotate_read(flipped, planted_tables, orientations=chosen) == ()
    assert bs.annotate_read(flipped, planted_tables) != ()


def test_the_text_window_reads_one_read_at_a_time(planted, planted_tables):
    """The window that shows one read per line stays bounded like the search."""
    rows = list(bs.iter_annotated_reads(planted["r1"], planted_tables, limit=12))

    assert len(rows) == 12
    for read, hits in rows:
        assert len(read) == READ_LENGTH
        assert hits


def test_reads_with_nothing_in_them_can_be_left_out(tmp_path, planted_tables):
    """A window of empty rows tells nobody anything."""
    rng = random.Random(31)
    reads = [_random_bases(rng, READ_LENGTH) for _ in range(20)]
    path = _write_fastq(tmp_path / "noise.fastq", reads)
    guides = [t for t in planted_tables if t.role == "grna"]

    everything = list(bs.iter_annotated_reads(path, guides, limit=20))
    matching = list(
        bs.iter_annotated_reads(path, guides, limit=20, only_matching=True))

    assert len(everything) == 20
    assert matching == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_the_rendered_table_never_shows_a_rate_without_its_baseline(planted, planted_tables):
    """A reader who sees only the observed column will believe a coincidence."""
    report = bs.search_barcodes(
        {"R1": planted["r1"]}, planted_tables, max_reads=1000)
    rendered = report.format_table()

    assert "chance" in rendered
    assert "enrich" in rendered
    for table in planted_tables:
        assert table.name[:26] in rendered
    assert rendered.count("\n") == len(report.findings) + 1


def test_an_enrichment_beyond_measuring_is_not_printed_to_a_decimal():
    """Ten trillion to one is arithmetic rather than a measurement."""
    assert bs._format_enrichment(float("inf")).strip() == ">1000"
    assert bs._format_enrichment(3.5e13).strip() == ">1000"
    assert bs._format_enrichment(5.4).strip() == "5.4"


def test_the_narrowest_offset_window_is_the_structural_measurement():
    """A plate barcode sits in one place and an adapter does not."""
    assert bs._narrowest_offset_window({}) == (0, 0)
    assert bs._narrowest_offset_window({34: 100}) == (34, 1)
    assert bs._narrowest_offset_window({33: 10, 34: 90, 35: 5}) == (34, 1)
    assert bs._narrowest_offset_window({33: 40, 34: 40, 90: 20}) == (33, 2)
    # Eighty of these hundred hits fit only in a stretch reaching from the
    # second cluster to the far one, which is forty four bases wide and is why
    # this pattern cannot yield an extraction window.
    assert bs._narrowest_offset_window(
        {85: 20, 86: 20, 128: 30, 129: 30}) == (86, 44)


def test_the_score_interval_keeps_a_width_when_nothing_was_seen():
    """A point estimate of zero would let ten reads declare a table absent."""
    low, high = bs._wilson_interval(0, 10)
    assert low == 0.0
    assert high > 0.25
    tighter_low, tighter_high = bs._wilson_interval(0, 10000)
    assert tighter_high < 0.001
    assert bs._wilson_interval(0, 0) == (0.0, 1.0)
    assert tighter_low == 0.0


# ---------------------------------------------------------------------------
# The real run the feature exists for
# ---------------------------------------------------------------------------

_EO1 = "/nas_mnt/data/sequencing/sequencing/"
_REFERENCES = "/home/olafsson/Documents/barcodes/"
_REAL_INPUTS = (
    (_EO1 + "EO1_R1_001.fastq.gz", "file"),
    (_EO1 + "EO1_R2_001.fastq.gz", "file"),
    (_REFERENCES + "primers_2_row_barcodes.csv", "file"),
    (_REFERENCES + "primers_2_column_barcodes.csv", "file"),
    (_REFERENCES + "grna_barcodes.csv", "file"),
)


@pytest.mark.nas
@pytest.mark.integration
@pytest.mark.skipif(
    not paths_available(_REAL_INPUTS),
    reason="the real paired sequencing run is not mounted",
)
def test_the_real_run_gives_the_answer_the_tutorial_needed():
    """The measured layout of the run whose mapping produced nothing.

    The guides are reverse complemented in the first mate and plain in the
    second, which is what a mate pair read from opposite ends does.  The row
    barcodes of this reference table are not in the run at all, and they match
    just under seven percent of reads by coincidence, which is the number that
    has to come back as absent for the baseline to be working.
    """
    tables = bs.load_barcode_tables([
        _REFERENCES + "primers_2_row_barcodes.csv",
        _REFERENCES + "primers_2_column_barcodes.csv",
        _REFERENCES + "grna_barcodes.csv",
    ])
    report = bs.search_barcodes(
        {"R1": _EO1 + "EO1_R1_001.fastq.gz", "R2": _EO1 + "EO1_R2_001.fastq.gz"},
        tables, max_reads=20000, anchor=ANCHOR)

    guide_r1 = [
        item for item in report.findings
        if item.role == "grna" and item.file_label == "R1"
        and item.verdict == bs.PRESENT
    ]
    guide_r2 = [
        item for item in report.findings
        if item.role == "grna" and item.file_label == "R2"
        and item.verdict == bs.PRESENT
    ]
    assert [item.orientation for item in guide_r1] == [bs.REVERSE_COMPLEMENT]
    assert [item.orientation for item in guide_r2] == [bs.AS_GIVEN]
    assert guide_r1[0].observed_rate > 0.7

    rows = report.for_table("primers_2_row_barcodes")
    assert len(rows) == 4
    for item in rows:
        assert item.verdict == bs.ABSENT, item.reason
        assert item.enrichment < bs.MIN_ENRICHMENT
    plain_r1 = next(
        item for item in rows
        if item.file_label == "R1" and item.orientation == bs.AS_GIVEN)
    assert 0.05 < plain_r1.observed_rate < 0.09
    assert plain_r1.expected_rate == pytest.approx(plain_r1.observed_rate, abs=0.01)
    assert report.best_for_role("row") is None

    # The shipped column table matches a quarter of the first mate reverse
    # complemented, five times its own coincidence rate, and it is the
    # sequencing adapter rather than a plate.  The chance baseline alone lets
    # this through, so the structural check has to stop it.
    columns = report.for_table("primers_2_column_barcodes")
    flipped_r1 = next(
        item for item in columns
        if item.file_label == "R1" and item.orientation == bs.REVERSE_COMPLEMENT)
    assert flipped_r1.enrichment > bs.MIN_ENRICHMENT
    assert flipped_r1.offset_span > bs.MAX_OFFSET_SPAN
    assert flipped_r1.verdict == bs.INDETERMINATE
    assert report.best_for_role("column") is None


@pytest.mark.nas
@pytest.mark.integration
@pytest.mark.skipif(
    not paths_available(
        _REAL_INPUTS
        + ((_REFERENCES + "primers_3_row_barecodes.csv", "file"),
           (_REFERENCES + "primers_3_column_barecodes.csv", "file"))),
    reason="the real paired sequencing run is not mounted",
)
def test_the_real_run_recovers_the_settings_that_ship_as_defaults():
    """Given the reference tables the run was built with, the layout falls out.

    The window this proposes is the one spaCR already ships as its default,
    arrived at from the reads alone, which is the strongest check available
    that the offsets mean what they are supposed to mean.
    """
    tables = bs.load_barcode_tables([
        _REFERENCES + "primers_3_row_barecodes.csv",
        _REFERENCES + "primers_3_column_barecodes.csv",
        _REFERENCES + "grna_barcodes.csv",
    ])
    report = bs.search_barcodes(
        {"R1": _EO1 + "EO1_R1_001.fastq.gz", "R2": _EO1 + "EO1_R2_001.fastq.gz"},
        tables, max_reads=20000, anchor=ANCHOR)
    proposal = bs.propose_map_barcodes_settings(report)

    assert proposal.orientations == {
        "row": bs.REVERSE_COMPLEMENT,
        "column": bs.AS_GIVEN,
        "grna": bs.REVERSE_COMPLEMENT,
    }
    assert set(proposal.reverse_complement_needed) == {"row", "grna"}
    assert proposal.unresolved_roles == ()
    assert proposal.settings["mode"] == "paired"
    assert proposal.settings["offset_start"] == -8
    assert proposal.settings["window_length"] == 89


# ---------------------------------------------------------------------------
# The corners
# ---------------------------------------------------------------------------


def test_a_barcode_longer_than_the_read_contributes_no_coincidence():
    """A barcode that cannot fit has no chance of turning up by accident."""
    rng = random.Random(41)
    mixed = dict(_make_table(rng, 16, 8, "s"))
    mixed.update(_make_table(rng, 4, 200, "l"))
    table = bs.BarcodeTable(name="mixed", sequences=mixed)
    short_only = bs.BarcodeTable(
        name="short",
        sequences={s: n for s, n in mixed.items() if len(s) == 8})

    assert sorted(table.length_counts) == [8, 200]
    assert bs.expected_chance_rate(table, 150) == pytest.approx(
        bs.expected_chance_rate(short_only, 150))


def test_a_table_with_nothing_in_it_matches_nothing():
    """An empty table is a search that finds nothing, not a crash."""
    empty = bs.BarcodeTable(name="empty", sequences={})

    assert empty.size == 0
    assert empty.length_counts == {}
    assert empty.barcode_order() == {}
    assert empty.matcher(bs.AS_GIVEN).find_first("ACGTACGTACGT") is None
    assert empty.matcher(bs.AS_GIVEN).find_all("ACGTACGTACGT") == []
    assert bs.expected_chance_rate(empty, 150) == 0.0
    assert bs.annotate_read("ACGTACGTACGT", [empty]) == ()


def test_a_finding_with_no_hits_proposes_no_window(tmp_path):
    """There is no window to cut when nothing was ever found."""
    rng = random.Random(43)
    table = bs.BarcodeTable(
        name="grna", role="grna", sequences=_make_table(rng, 50, 20, "g"))
    reads = [_random_bases(rng, READ_LENGTH) for _ in range(600)]
    path = _write_fastq(tmp_path / "noise.fastq", reads)

    report = bs.search_barcodes({"R1": path}, [table], max_reads=600)
    finding = report.for_table("grna")[0]

    assert finding.hits == 0
    assert finding.offset_span == 0
    assert finding.modal_offset is None
    assert finding.window_end == finding.offset_start
    assert finding.top_barcode_share == 0.0
    assert list(finding.offset_histogram()) == []


def test_a_rate_sitting_on_the_chance_threshold_waits_for_more_reads(tmp_path):
    """Refusing to decide is a real answer and has to be reachable.

    The filler here repeats a short pattern, so it holds only a handful of
    distinct stretches and none of the randomly drawn barcodes turns up in it.
    That makes the hit count exactly the number planted, while the coincidence
    rate the engine computes is unchanged, and the observation lands on the
    boundary where too few reads have been seen to call it either way.
    """
    rng = random.Random(45)
    barcodes = _make_table(rng, 100, 12, "c")
    table = bs.BarcodeTable(name="column", role="column", sequences=barcodes)
    filler = ("ACGTTGCA" * 32)[:READ_LENGTH]
    reads = [filler for _ in range(600)]
    planted = sorted(barcodes)
    for index in range(2):
        reads[index] = filler[:30] + planted[index] + filler[42:]
    path = _write_fastq(tmp_path / "borderline.fastq", reads)

    report = bs.search_barcodes({"R1": path}, [table], max_reads=600)
    best = max(report.for_table("column"), key=lambda item: item.observed_rate)

    assert best.hits == 2
    assert best.expected_rate < best.observed_rate < best.expected_rate * 10
    assert best.verdict == bs.INDETERMINATE
    assert "not yet large enough" in best.reason


def test_a_rate_sitting_on_the_usable_threshold_waits_for_more_reads(tmp_path):
    """The same restraint applies to the question of having enough reads."""
    rng = random.Random(47)
    guides = _make_table(rng, 200, 20, "g")
    table = bs.BarcodeTable(name="grna", role="grna", sequences=guides)
    reads = [_random_bases(rng, READ_LENGTH) for _ in range(600)]
    planted = sorted(guides)
    for index in range(60):
        filler = reads[index]
        reads[index] = filler[:34] + planted[index % 200] + filler[54:]
    path = _write_fastq(tmp_path / "borderline.fastq", reads)

    report = bs.search_barcodes({"R1": path}, [table], max_reads=600)
    best = max(report.for_table("grna"), key=lambda item: item.observed_rate)

    assert best.observed_rate == pytest.approx(0.10, abs=0.005)
    assert best.verdict == bs.INDETERMINATE
    assert "not yet clear whether enough reads" in best.reason


def test_two_files_with_the_same_name_keep_separate_labels(tmp_path):
    """Two runs of one plate can be compared without one hiding the other."""
    rng = random.Random(49)
    table = bs.BarcodeTable(name="t", sequences=_make_table(rng, 8, 8, "t"))
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    reads = [_random_bases(rng, READ_LENGTH) for _ in range(4)]
    paths = [
        _write_fastq(first / "sample_R1_001.fastq", reads),
        _write_fastq(second / "sample_R1_001.fastq", reads),
    ]

    report = bs.search_barcodes(paths, [table], max_reads=4)

    assert list(report.reads_by_file) == ["R1", "R1_"]


@pytest.mark.parametrize(
    "filename,label",
    [
        ("sample_R1_001.fastq.gz", "R1"),
        ("sample.R2.fastq", "R2"),
        ("sample_r1.fq", "R1"),
        ("plain_reads.fastq.gz", "plain_reads"),
        ("noextension", "noextension"),
        (".gz", "reads"),
    ],
)
def test_a_file_is_labelled_by_its_mate_or_by_its_name(filename, label):
    """The label is what a reader of the report sees, so it has to be short."""
    assert bs._default_label(f"/somewhere/{filename}") == label


def test_the_two_orientations_are_each_other_s_opposite():
    """Nothing else is an orientation, so flipping is a closed operation."""
    assert bs._flip(bs.AS_GIVEN) == bs.REVERSE_COMPLEMENT
    assert bs._flip(bs.REVERSE_COMPLEMENT) == bs.AS_GIVEN


def test_a_table_held_in_memory_proposes_no_path(planted, tmp_path):
    """A table built in a notebook has no file for the settings to point at."""
    loaded = bs.load_barcode_table(planted["column_csv"])
    in_memory = bs.BarcodeTable(
        name="column", role="column", sequences=dict(loaded.sequences))

    report = bs.search_barcodes(
        {"R1": planted["r1"]}, [in_memory], max_reads=1000, anchor=ANCHOR)
    proposal = bs.propose_map_barcodes_settings(report)

    assert proposal.orientations["column"] == bs.AS_GIVEN
    assert "column_csv" not in proposal.settings
    assert any("searched from memory" in note for note in proposal.notes)


def test_a_role_found_only_in_the_other_mate_is_translated_not_dropped(
        planted, planted_tables, tmp_path):
    """A first mate that carries nothing can still be configured from the second.

    Orientations translate between mates because the two reads run in opposite
    directions along one fragment.  Offsets do not, because neither read knows
    how long the fragment was, so the window is left alone rather than placed
    from a measurement taken in the wrong frame.
    """
    rng = random.Random(51)
    noise = _write_fastq(
        tmp_path / "noise_R1_001.fastq",
        [_random_bases(rng, READ_LENGTH) for _ in range(2000)])
    report = bs.search_barcodes(
        {"R1": noise, "R2": planted["r2"]},
        planted_tables, max_reads=2000, anchor=ANCHOR)
    proposal = bs.propose_map_barcodes_settings(
        report, base_settings={"offset_start": -8, "window_length": 89},
        reference_file="R1")

    r2_column = next(
        item for item in report.findings
        if item.role == "column" and item.file_label == "R2"
        and item.verdict == bs.PRESENT)
    assert r2_column.orientation == bs.REVERSE_COMPLEMENT

    # Seen flipped in the second mate is the same event as seen as stored in
    # the first, because the two reads run in opposite directions.
    assert proposal.reference_file == "R1"
    assert proposal.source_files["column"] == "R2"
    assert proposal.orientations["column"] == bs.AS_GIVEN
    assert proposal.orientations["grna"] == bs.AS_GIVEN
    assert proposal.settings["offset_start"] == -8
    assert proposal.settings["window_length"] == 89
    assert proposal.settings["mode"] == "single"
    assert proposal.settings["single_direction"] == "R2"


def test_a_proposal_from_a_report_that_found_nothing_changes_nothing(tmp_path):
    """Nothing established means nothing proposed, which is the safe answer."""
    rng = random.Random(53)
    table = bs.BarcodeTable(
        name="row", role="row", sequences=_make_table(rng, 32, 8, "r"))
    path = _write_fastq(
        tmp_path / "noise_R1_001.fastq",
        [_random_bases(rng, READ_LENGTH) for _ in range(1000)])
    report = bs.search_barcodes({"R1": path}, [table], max_reads=1000)
    base = {"mode": "paired", "row_csv": "/original/row.csv"}
    proposal = bs.propose_map_barcodes_settings(report, base_settings=base)

    assert proposal.settings == base
    assert proposal.unresolved_roles == ("row",)
    assert proposal.orientations == {}
    assert proposal.reverse_complement_needed == ()


def test_a_file_that_is_not_a_mate_proposes_no_direction(planted, tmp_path):
    """A single unpaired run has no first or second mate to name."""
    reads = bs.sample_fastq_reads(planted["r1"], 1000)
    path = _write_fastq(tmp_path / "unpaired.fastq", reads)
    tables = bs.load_barcode_tables([planted["column_csv"]])

    report = bs.search_barcodes({"unpaired": path}, tables, max_reads=1000)
    proposal = bs.propose_map_barcodes_settings(report)

    assert list(report.reads_by_file) == ["unpaired"]
    assert proposal.settings["mode"] == "single"
    assert "single_direction" not in proposal.settings
    assert any("only unpaired carries" in note for note in proposal.notes)


def test_a_barcode_beyond_the_usual_three_keeps_its_chosen_table(planted, tmp_path):
    """A screen with a fourth barcode has no settings key waiting for it.

    The mapping run grew up decoding a plate column, a guide and a plate row,
    and those three have keys of their own.  A further barcode still has to
    reach the caller with the table that was chosen for it, because dropping it
    silently is how a run ends up decoding the wrong reference.
    """
    rng = random.Random(57)
    extra_sequences = _make_table(rng, 12, 10, "x")
    reads = list(bs.sample_fastq_reads(planted["r1"], 1500))
    planted_extra = sorted(extra_sequences)
    reads = [
        read[:100] + planted_extra[index % 12] + read[110:]
        for index, read in enumerate(reads)
    ]
    path = _write_fastq(tmp_path / "extra_R1_001.fastq", reads)
    extra_csv = _write_table(tmp_path / "lineage_barcodes.csv", extra_sequences)
    tables = list(bs.load_barcode_tables([planted["column_csv"]]))
    tables.append(bs.load_barcode_table(extra_csv, name="lineage", role="lineage"))

    report = bs.search_barcodes({"R1": path}, tables, max_reads=1500, anchor=ANCHOR)
    proposal = bs.propose_map_barcodes_settings(report)

    assert report.best_for_role("lineage") is not None
    assert proposal.orientations["lineage"] == bs.AS_GIVEN
    assert proposal.reference_tables["lineage"] == extra_csv
    assert proposal.reference_tables["column"] == planted["column_csv"]
    assert "lineage_csv" not in proposal.settings
    assert any("no settings key of its own" in note for note in proposal.notes)
    # The extra barcode reaches further into the read than the guide does, so
    # the window the proposal cuts has to grow to include it.
    assert proposal.settings["window_length"] >= 110
