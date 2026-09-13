"""Map Barcodes decodes a COLLECTION of barcodes, not exactly three.

THE LIMIT THIS REMOVES was written into three places at once: the shipped
regex named ``columnID``, ``grna`` and ``rowID``; the settings offered
``column_csv``, ``grna_csv`` and ``row_csv``; and the read processors asked
for those three group names by hand. A screen carrying a plate barcode as
well, or a screen carrying only a guide, had nowhere to say so.

THE PROPERTY THAT MATTERS IS THE ONE NOT CHANGED. Every settings file anyone
has saved names the three old keys, and a missing key in this codebase is a
DEFAULT rather than an error, so a generalisation that shifts the three by a
column name or a sort order changes the numbers on somebody else's machine
without telling them. Most of this file is therefore about sameness: the same
frame, the same counts, the same bytes on disk, from the old settings and
from a set that spells the same three barcodes out.
"""
from __future__ import annotations

import gzip

import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import sequencing as SEQ
from spacr.settings import (BarcodeEntry, BarcodeSet,
                            barcode_set_from_settings,
                            set_default_generate_barecode_mapping)

# A read is padding, then the anchor, then the barcode window.
ANCHOR = "GGTTAACC"

#: Three barcodes of four, six and four bases, named the way the shipped
#: regex names them.
THREE_REGEX = r"^(?P<columnID>.{4})(?P<grna>.{6})(?P<rowID>.{4})"
THREE_WINDOW = 14

#: The same three with a plate and a lane barcode after them, and the SHORT
#: group names, so the aliases the shipped column and row accept are exercised
#: on the general path rather than only on the old one.
FIVE_REGEX = (r"^(?P<column>.{4})(?P<grna>.{6})(?P<row>.{4})"
              r"(?P<plate>.{3})(?P<lane>.{2})")
FIVE_WINDOW = 19

COLUMN, GRNA, ROW, PLATE, LANE = "ACGT", "TTTTTT", "CCCC", "AAG", "TC"


def _csv(tmp_path, name, rows):
    """Write a two-column barcode reference and return its path."""
    path = tmp_path / name
    pd.DataFrame(rows, columns=["sequence", "name"]).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def references(tmp_path):
    """The five reference tables the reads below are built from."""
    return {
        "column": _csv(tmp_path, "columns.csv", [(COLUMN, "col_1")]),
        "grna": _csv(tmp_path, "grnas.csv", [(GRNA, "sg_1")]),
        "row": _csv(tmp_path, "rows.csv", [(ROW, "row_A")]),
        "plate": _csv(tmp_path, "plates.csv", [(PLATE, "plate_7")]),
        "lane": _csv(tmp_path, "lanes.csv", [(LANE, "lane_2")]),
    }


def _record(payload, quality=None):
    """Return one four-line FASTQ record carrying ``payload`` after the anchor."""
    sequence = "TTTT" + ANCHOR + payload
    quality = quality or ("I" * len(sequence))
    return f"@read\n{sequence}\n+\n{quality}"


def _three_reads(count=3):
    """Return reads whose window holds the three shipped barcodes."""
    return [_record(COLUMN + GRNA + ROW + "GG") for _ in range(count)]


def _five_reads(count=2):
    """Return reads whose window holds five barcodes."""
    return [_record(COLUMN + GRNA + ROW + PLATE + LANE) for _ in range(count)]


def _legacy_chunk(reads, references, fill_na=False):
    """Return the tuple chunk a run that names no barcode set sends."""
    return (reads, THREE_REGEX, ANCHOR, len(ANCHOR), THREE_WINDOW,
            references["column"], references["grna"], references["row"],
            fill_na)


def _set_chunk(reads, barcode_set, regex, window, fill_na=False,
               r2_chunk=None):
    """Return the mapping chunk a run that names a barcode set sends."""
    return {"r1_chunk": reads, "r2_chunk": r2_chunk, "regex": regex,
            "target_sequence": ANCHOR, "offset_start": len(ANCHOR),
            "window_length": window, "barcode_set": barcode_set,
            "fill_na": fill_na}


def _three_entry_set(references):
    """Build the three shipped barcodes as an explicit set, through settings.

    Through ``barcode_set_from_settings`` rather than by hand, because that
    is the path a settings file takes and the spellings it fills in are half
    of what makes the result the same run.
    """
    return barcode_set_from_settings({"barcode_set": [
        {"name": "column", "csv": references["column"]},
        {"name": "row", "csv": references["row"]},
        {"name": "grna", "csv": references["grna"]},
    ]})


# ---------------------------------------------------------------------------
# 1. A three-entry set is the run spaCR has always done
# ---------------------------------------------------------------------------

class TestTheThreeShippedBarcodesAreUnchanged:

    def test_a_three_entry_set_produces_the_same_frames_as_the_old_chunk(
            self, references):
        """Same reads, same three barcodes, two routes in: one answer.

        Compared frame by frame rather than value by value, because the
        things that could drift silently are the column ORDER of the
        annotated reads, which becomes the column order of ``qc.csv``, and
        the key order of the counts, which becomes the row order of
        ``unique_combinations.csv``.
        """
        reads = _three_reads()

        old_df, old_counts, old_qc = SEQ.process_chunk(
            _legacy_chunk(reads, references))
        new_df, new_counts, new_qc = SEQ.process_chunk(
            _set_chunk(reads, _three_entry_set(references),
                       THREE_REGEX, THREE_WINDOW))

        assert list(old_df.columns) == [
            "read", "column_sequence", "columnID", "row_sequence", "rowID",
            "grna_sequence", "grna_name"], (
            "the frame this comparison is anchored on is no longer the one "
            "the module has always written")
        pd.testing.assert_frame_equal(new_df, old_df)
        pd.testing.assert_frame_equal(new_counts, old_counts)
        pd.testing.assert_frame_equal(new_qc, old_qc)

    def test_the_counts_keep_the_order_they_have_always_been_written_in(
            self, references):
        """Row, then column, then guide -- which is NOT the reads' order.

        The annotated reads list the column barcode first and the counts are
        grouped row first. Both orders are older than barcode sets, so a set
        of the three carries the counting order explicitly; taking the entry
        order would re-sort the rows and the header of every count table
        anyone already has, for a reason nobody asked for.
        """
        barcode_set = _three_entry_set(references)

        assert barcode_set.id_columns == ("columnID", "rowID", "grna_name")
        assert barcode_set.count_columns == ("rowID", "columnID", "grna_name")

        _df, counts, _qc = SEQ.process_chunk(
            _set_chunk(_three_reads(), barcode_set, THREE_REGEX,
                       THREE_WINDOW))
        assert list(counts.columns) == ["rowID", "columnID", "grna_name",
                                        "count"]

    def test_a_three_entry_set_writes_the_same_bytes_to_disk(
            self, tmp_path, references):
        """The proof that matters, at the level a user would notice.

        Both runs read the same gzipped FASTQ through the same public reader
        and write the two tables the module exists to produce. Comparing the
        files as BYTES catches everything a frame comparison can miss:
        column order, row order, float formatting, a stray index column.
        """
        reads = _three_reads(5)
        fastq = tmp_path / "reads.fastq.gz"
        with gzip.open(fastq, "wt") as handle:
            handle.write("\n".join(reads) + "\n")

        produced = {}
        for label, barcode_set in (("old", None),
                                   ("set", _three_entry_set(references))):
            counts_csv = tmp_path / f"{label}_counts.csv"
            qc_csv = tmp_path / f"{label}_qc.csv"
            SEQ.single_read_chunked_processing(
                r1_file=str(fastq), r2_file=None, regex=THREE_REGEX,
                target_sequence=ANCHOR, offset_start=len(ANCHOR),
                expected_end=THREE_WINDOW,
                column_csv=references["column"], grna_csv=references["grna"],
                row_csv=references["row"], save_h5=False, comp_type="zlib",
                comp_level=5, hdf5_file=str(tmp_path / f"{label}.h5"),
                unique_combinations_csv=str(counts_csv),
                qc_csv_file=str(qc_csv), chunk_size=2, n_jobs=1,
                barcode_set=barcode_set)
            produced[label] = (counts_csv.read_bytes(), qc_csv.read_bytes())

        assert produced["old"][0], "the run wrote no counts to compare"
        assert produced["set"] == produced["old"]

    def test_the_paired_path_is_the_same_run_too(self, references):
        """Paired reads take a different window and the same decode.

        Worth its own case because the consensus is built before the regex
        is applied, so the barcode set is reached by a second route.
        """
        reads = _three_reads(2)
        mates = [f"@read\n{SEQ.reverse_complement(record.splitlines()[1])}\n"
                 f"+\n{record.splitlines()[3][::-1]}" for record in reads]

        old = SEQ.process_chunk(
            (reads, mates, THREE_REGEX, ANCHOR, len(ANCHOR), THREE_WINDOW,
             references["column"], references["grna"], references["row"],
             False))
        new = SEQ.process_chunk(
            _set_chunk(reads, _three_entry_set(references), THREE_REGEX,
                       THREE_WINDOW, r2_chunk=mates))

        assert len(old[0]) == 2, "the paired fixture stopped matching"
        for produced, expected in zip(new, old):
            pd.testing.assert_frame_equal(produced, expected)

    def test_filling_unmapped_names_is_the_same_on_both_routes(
            self, tmp_path, references):
        """``fill_na`` names an unmapped barcode after its own sequence.

        The old code filled three columns by name. The set fills one per
        entry, so a three-entry set has to fill exactly the same three.
        """
        unknown = dict(references)
        unknown["column"] = _csv(tmp_path, "other.csv", [("TTTT", "elsewhere")])
        reads = _three_reads(1)

        old = SEQ.process_chunk(_legacy_chunk(reads, unknown, fill_na=True))
        new = SEQ.process_chunk(
            _set_chunk(reads, _three_entry_set(unknown), THREE_REGEX,
                       THREE_WINDOW, fill_na=True))

        assert list(old[1]["columnID"]) == [COLUMN], (
            "the unmapped column was not filled with its own sequence")
        for produced, expected in zip(new, old):
            pd.testing.assert_frame_equal(produced, expected)


# ---------------------------------------------------------------------------
# 2. More than three, and fewer
# ---------------------------------------------------------------------------

class TestASetOfAnySize:

    def test_five_barcodes_each_get_their_own_columns_and_names(
            self, references):
        """The feature itself: a screen with a plate and a lane barcode.

        Five entries means five pairs of columns in the annotated reads and
        counts per unique combination of all five, which is the whole point
        -- a plate barcode that is decoded but not counted would pool two
        plates into one well.
        """
        barcode_set = barcode_set_from_settings({"barcode_set": [
            {"name": "column", "csv": references["column"]},
            {"name": "row", "csv": references["row"]},
            {"name": "grna", "csv": references["grna"]},
            {"name": "plate", "csv": references["plate"]},
            {"name": "lane", "csv": references["lane"]},
        ]})

        assert len(barcode_set) == 5

        df, counts, qc = SEQ.process_chunk(
            _set_chunk(_five_reads(2), barcode_set, FIVE_REGEX, FIVE_WINDOW))

        assert list(df.columns) == [
            "read", "column_sequence", "columnID", "row_sequence", "rowID",
            "grna_sequence", "grna_name", "plate_sequence", "plateID",
            "lane_sequence", "laneID"]
        assert list(df["plateID"]) == ["plate_7", "plate_7"]
        assert list(df["laneID"]) == ["lane_2", "lane_2"]
        # Entry order, because this is not the shipped three: only a set
        # that IS those three keeps their historical counting order.
        assert list(counts.columns) == ["columnID", "rowID", "grna_name",
                                        "plateID", "laneID", "count"]
        assert int(counts["count"].iloc[0]) == 2
        assert int(qc["total_reads"].iloc[0]) == 2

    def test_the_short_group_names_still_reach_the_shipped_barcodes(
            self, references):
        """The five-barcode regex above names ``column`` and ``row``.

        The shipped column and row prefer the longer group names and accept
        the short ones, so a regex written either way decodes. Without the
        aliases this pattern would be refused for naming no ``columnID``.
        """
        barcode_set = barcode_set_from_settings({"barcode_set": [
            {"name": "column", "csv": references["column"]},
            {"name": "row", "csv": references["row"]},
            {"name": "grna", "csv": references["grna"]},
            {"name": "plate", "csv": references["plate"]},
            {"name": "lane", "csv": references["lane"]},
        ]})

        assert barcode_set.resolve_groups(FIVE_REGEX) == {
            "column": "column", "row": "row", "grna": "grna",
            "plate": "plate", "lane": "lane"}

    def test_one_barcode_is_a_whole_set(self, references):
        """A guide-only screen, which the three named settings could not say.

        One entry writes one pair of columns and counts by the guide alone.
        The reads still carry the other barcodes; nothing asks about them,
        which is what makes this a different run rather than a broken one.
        """
        barcode_set = barcode_set_from_settings({"barcode_set": [
            {"name": "grna", "csv": references["grna"],
             "group": "grna"}]})

        df, counts, qc = SEQ.process_chunk(
            _set_chunk(_three_reads(4), barcode_set,
                       r"^.{4}(?P<grna>.{6})", THREE_WINDOW))

        assert len(barcode_set) == 1
        assert list(df.columns) == ["read", "grna_sequence", "grna_name"]
        assert list(counts.columns) == ["grna_name", "count"]
        assert int(counts["count"].iloc[0]) == 4
        assert int(qc["total_reads"].iloc[0]) == 4

    def test_the_counts_of_a_set_accumulate_across_chunks(self, tmp_path):
        """The saver sums a five-barcode table as it summed a three.

        It used to group by the three column names written out as a literal,
        which is the second place a run held exactly three barcodes. It
        reads the combination off the frame now, so a file already holding a
        chunk's counts gains the next chunk's rather than a second copy.
        """
        counts = pd.DataFrame({
            "rowID": ["row_A"], "columnID": ["col_1"], "grna_name": ["sg_1"],
            "plateID": ["plate_7"], "laneID": ["lane_2"], "count": [4]})
        destination = tmp_path / "counts.csv"

        SEQ.save_unique_combinations_to_csv(counts, str(destination))
        SEQ.save_unique_combinations_to_csv(counts, str(destination))

        written = pd.read_csv(destination)
        assert len(written) == 1, "the two chunks were not summed"
        assert int(written["count"].iloc[0]) == 8
        assert list(written.columns) == list(counts.columns)


# ---------------------------------------------------------------------------
# 3. What a settings file means
# ---------------------------------------------------------------------------

class TestAnOldSettingsFileIsStillUnderstood:

    def test_a_settings_file_naming_no_set_is_the_three_shipped_barcodes(
            self, references):
        """Absent is not empty and not an error: it is the historical run.

        `barcode_set_from_settings` answering None is what keeps every
        settings file written before this feature on the code path it has
        always taken, down to the tuple the workers are handed.
        """
        settings = set_default_generate_barecode_mapping({
            "column_csv": references["column"],
            "grna_csv": references["grna"],
            "row_csv": references["row"]})

        assert barcode_set_from_settings(settings) is None
        assert settings["column_csv"] == references["column"], (
            "a default was filled in over the value the file carried")
        assert settings["grna_csv"] == references["grna"]
        assert settings["row_csv"] == references["row"]

    @pytest.mark.parametrize("blank", [None, "", [], ()])
    def test_a_blank_set_is_the_three_shipped_barcodes_too(self, blank):
        """A cleared field is not an empty run.

        A panel hands back an empty string for a field a user cleared and an
        empty list for a set they emptied. Reading either as "decode
        nothing" would produce a run with no counts and no error.
        """
        assert barcode_set_from_settings({"barcode_set": blank}) is None

    def test_the_window_rename_still_folds_under_the_new_key(self,
                                                             references):
        """An old file carries ``expected_end``; the fold still moves it.

        Named here because this file's subject is old settings files and the
        fold runs before any default is filled in. A set added beside a
        renamed key must not be able to disturb that ordering.
        """
        settings = set_default_generate_barecode_mapping({
            "column_csv": references["column"],
            "grna_csv": references["grna"],
            "row_csv": references["row"],
            "expected_end": 62})

        assert settings["window_length"] == 62
        assert "expected_end" not in settings

    def test_an_entry_takes_the_reference_already_named_beside_it(
            self, references):
        """Adding a fourth barcode is a one-entry addition.

        An entry with no reference table of its own takes the one under its
        own name, so a settings file that already names the three does not
        have to repeat them to add a plate.
        """
        barcode_set = barcode_set_from_settings({
            "column_csv": references["column"],
            "grna_csv": references["grna"],
            "row_csv": references["row"],
            "plate_csv": references["plate"],
            "barcode_set": ["column", "row", "grna", "plate"]})

        assert [entry.csv for entry in barcode_set] == [
            references["column"], references["row"], references["grna"],
            references["plate"]]
        assert barcode_set.id_columns == ("columnID", "rowID", "grna_name",
                                          "plateID")

    def test_a_named_set_wins_over_the_three_old_keys(self, references):
        """Both present means the newer spelling decides, as renames do here.

        A file that names a set has been edited since the three keys were
        written, so honouring the three instead would make a corrected file
        behave like the uncorrected one.
        """
        barcode_set = barcode_set_from_settings({
            "column_csv": references["column"],
            "grna_csv": references["grna"],
            "row_csv": references["row"],
            "barcode_set": [{"name": "grna", "csv": references["grna"]}]})

        assert barcode_set.names == ("grna",)

    def test_an_entry_with_no_reference_anywhere_says_so(self):
        """A barcode with no table names nothing, so the run stops.

        The alternative is a column of NA the user has to notice, which is
        the failure mode this whole module is worst at.
        """
        with pytest.raises(ValueError) as caught:
            barcode_set_from_settings({"barcode_set": ["plate"]})

        assert "plate" in str(caught.value)
        assert "plate_csv" in str(caught.value)


# ---------------------------------------------------------------------------
# 4. A regex that does not name a group for every barcode
# ---------------------------------------------------------------------------

class TestARegexMustNameAGroupPerBarcode:

    def test_the_missing_group_is_named_for_the_old_three(self, references):
        """The message a user has always got, from the general code.

        Both accepted spellings of the missing group are named, because a
        user who wrote one of them needs to see the one they did not.
        """
        with pytest.raises(ValueError) as caught:
            SEQ.process_chunk(([], r"(?P<grna>.{6})", ANCHOR, 0, THREE_WINDOW,
                               references["column"], references["grna"],
                               references["row"], False))

        message = str(caught.value)
        assert "missing required named group" in message
        assert "column/columnID" in message
        assert "row/rowID" in message

    def test_the_missing_group_is_named_for_a_five_barcode_set(
            self, references):
        """Five barcodes and four groups is the new way to get this wrong.

        The message names the barcode that has no group and then the groups
        the regex does define, because "which of my five is missing" is the
        only question being asked at that moment.
        """
        barcode_set = barcode_set_from_settings({"barcode_set": [
            {"name": "column", "csv": references["column"]},
            {"name": "row", "csv": references["row"]},
            {"name": "grna", "csv": references["grna"]},
            {"name": "plate", "csv": references["plate"]},
            {"name": "lane", "csv": references["lane"]},
        ]})

        with pytest.raises(ValueError) as caught:
            barcode_set.resolve_groups(
                r"^(?P<column>.{4})(?P<grna>.{6})(?P<row>.{4})(?P<plate>.{3})")

        message = str(caught.value)
        assert "missing required named group(s): lane." in message
        assert "column, grna, plate, row" in message

    def test_two_barcodes_cannot_be_read_from_one_group(self, references):
        """An alias can fall back onto another barcode's group.

        The column barcode accepts the short spelling ``column``, so a regex
        that defines ``column`` and gives a second barcode that same group
        would hand both of them the same captured text and count them as
        though they differed. Nothing in the output would look wrong.
        """
        barcode_set = barcode_set_from_settings({"barcode_set": [
            {"name": "column", "csv": references["column"]},
            {"name": "plate", "csv": references["plate"], "group": "column"},
        ]})

        with pytest.raises(ValueError) as caught:
            barcode_set.resolve_groups(r"^(?P<column>.{4})")

        message = str(caught.value)
        assert "both be read from the regex group column" in message
        assert "plate" in message

    def test_a_regex_with_no_groups_at_all_says_that(self, references):
        """Nothing captured is a different mistake from one group missing."""
        barcode_set = barcode_set_from_settings({"barcode_set": [
            {"name": "grna", "csv": references["grna"]}]})

        with pytest.raises(ValueError) as caught:
            barcode_set.resolve_groups(r"^.{20}$")

        assert "no named groups at all" in str(caught.value)

    def test_the_run_refuses_before_it_opens_a_single_fastq(
            self, tmp_path, references, monkeypatch):
        """A settings mistake costs a second, not a chunk and a traceback.

        The check used to live in the worker, so the first news of a regex
        that does not fit the barcodes was an exception raised inside a
        forked process. With a set it is a settings mistake by definition,
        and this proves nothing is read before it is caught.
        """
        source = tmp_path / "src"
        source.mkdir()
        with gzip.open(source / "sample_R1_001.fastq.gz", "wt") as handle:
            handle.write("\n".join(_three_reads(1)) + "\n")

        # Watched only once the reads are on disk: this fixture writes them
        # with the very function the run is being watched for.
        opened = []
        real_open = gzip.open

        def _watched_open(*args, **kwargs):
            """Record every gzip the run opens, then open it."""
            opened.append(args[0] if args else kwargs.get("filename"))
            return real_open(*args, **kwargs)

        monkeypatch.setattr(SEQ.gzip, "open", _watched_open)

        with pytest.raises(ValueError) as caught:
            SEQ.generate_barecode_mapping({
                "src": str(source), "mode": "single", "single_direction": "R1",
                "regex": THREE_REGEX, "target_sequence": ANCHOR,
                "offset_start": len(ANCHOR), "window_length": THREE_WINDOW,
                "column_csv": references["column"],
                "grna_csv": references["grna"], "row_csv": references["row"],
                "plate_csv": references["plate"],
                "barcode_set": ["column", "row", "grna", "plate"],
                "save_h5": False, "chunk_size": 2, "n_jobs": 1})

        assert "missing required named group(s): plate." in str(caught.value)
        assert not opened, f"the run read {opened} before refusing"


# ---------------------------------------------------------------------------
# 5. A set that cannot decode is refused when it is built
# ---------------------------------------------------------------------------

class TestASetThatCannotDecodeIsRefused:

    def test_two_barcodes_cannot_share_a_name(self, references):
        """Sharing a name means sharing both output columns.

        The second barcode would overwrite the first in the annotated reads
        and the counts would be of one barcode counted twice -- a full table
        of plausible numbers, which is the worst answer available.
        """
        with pytest.raises(ValueError) as caught:
            barcode_set_from_settings({"barcode_set": [
                {"name": "grna", "csv": references["grna"]},
                {"name": "grna", "csv": references["column"]}]})

        assert "share a name" in str(caught.value)

    def test_an_empty_set_is_refused(self):
        """A run with nothing to decode has nothing to count."""
        with pytest.raises(ValueError) as caught:
            BarcodeSet(())

        assert "at least one entry" in str(caught.value)

    def test_a_barcode_needs_a_name(self):
        """The name is what the output columns are called."""
        with pytest.raises(ValueError) as caught:
            BarcodeEntry(name="  ")

        assert "needs a name" in str(caught.value)

    def test_the_count_columns_are_the_barcodes_in_another_order(self):
        """Counting by a subset would pool two barcodes into one row."""
        entries = (BarcodeEntry(name="grna", csv="g.csv"),
                   BarcodeEntry(name="plate", csv="p.csv"))

        assert BarcodeSet(entries, count_columns=("plateID", "grnaID"))
        with pytest.raises(ValueError) as caught:
            BarcodeSet(entries, count_columns=("grnaID",))

        assert "count columns" in str(caught.value)

    def test_a_set_member_that_is_not_a_barcode_is_refused(self):
        """A list of paths is the likely mistake, and it is caught at once."""
        with pytest.raises(ValueError) as caught:
            BarcodeSet(("/data/columns.csv",))

        assert "BarcodeEntry" in str(caught.value)

    def test_an_entry_field_that_does_not_exist_is_named(self, references):
        """A typo in a settings file is worth a message, not a default."""
        with pytest.raises(ValueError) as caught:
            barcode_set_from_settings({"barcode_set": [
                {"name": "grna", "csv": references["grna"],
                 "reference": "g.csv"}]})

        assert "reference" in str(caught.value)

    def test_a_barcode_set_that_is_not_a_list_is_refused(self):
        """``barcode_set=3`` is a mistake with no sensible reading."""
        with pytest.raises(ValueError) as caught:
            barcode_set_from_settings({"barcode_set": 3})

        assert "one entry per barcode" in str(caught.value)

    def test_a_chunk_mapping_missing_a_field_names_it(self, references):
        """The general chunk shape says what it lacks, like the tuple does."""
        with pytest.raises(ValueError) as caught:
            SEQ.process_chunk({"r1_chunk": [], "regex": THREE_REGEX})

        message = str(caught.value)
        assert "barcode-set chunk is missing" in message
        assert "barcode_set" in message and "window_length" in message
