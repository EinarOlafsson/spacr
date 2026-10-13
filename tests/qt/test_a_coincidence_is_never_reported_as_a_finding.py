"""The live barcode search, and the one number that keeps it honest.

A real paired run of this screen produced 8,611 consensus rows and zero mapped
counts, and finished normally while doing it. The two mates of a pair are read
from opposite ends of one fragment, so a barcode that is plain in one mate is
reverse complemented in the other, and spaCR ships its reference tables in one
orientation only. Measured on that run, the column table matched 0.9% of the
first mate's reads against a coincidence rate of 5.2%, which is below noise,
and nothing anywhere said so.

The search this file tests exists to say so. What it must never do is the
opposite mistake, and the fixture below is built to make that mistake available
on every run: it ships a row table of thirty-two eight base barcodes that
appear in none of the reads. Thirty-two eight base barcodes scanned across a
hundred and fifty base read match by pure coincidence in about seven percent of
reads, so that table reports a perfectly real seven percent and means nothing
at all. A panel that showed "row barcodes, 6.9%" would send somebody hunting a
bug that does not exist, or would configure a mapping that yields garbage.

So the tests that carry the weight here are the ones that read the widget:
that the chance rate has a column of its own directly beside the observed
rate, that the decoy's two rates come out alike, and that its verdict is
absent. The rest pin the mechanics the maintainer asked for -- a search that
runs off the GUI thread and refines as it goes, reads shown one per row with
their matches coloured, and an Apply that is a second, separate press.
"""
from __future__ import annotations

import gzip
import random

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.barcode_search import ABSENT, PRESENT, reverse_complement
from spacr.qt.screens import map_barcodes
from spacr.qt.screens.app_screen import AppScreen


#: The vector sequence the mapping run anchors its extraction window on, and
#: the default the settings form ships with.
ANCHOR = "TGCTGTTTCCAGCATAGCTCTTAAAC"

#: Bases before the anchor in the first mate, so the anchor does not start at
#: zero and an offset expressed relative to it is distinguishable from one
#: expressed relative to the read.
LEAD = 8

#: Reads written to each mate. Above the five hundred the engine requires
#: before it will issue any verdict other than an undecided one, and small
#: enough that a whole search is a fraction of a second.
READS = 600

#: Reads per step of the search. Three steps over the sample above, which is
#: what makes "it refines as it goes" a countable claim rather than a feeling.
CHUNK = 200

#: How long a read is. The coincidence rate of a barcode table depends on it:
#: a hundred and fifty base read offers a hundred and forty three places for
#: an eight base barcode to land.
READ_LENGTH = 150

#: Fixed, because the decoy table's whole job is to match nothing on purpose
#: and a lucky seed would make it match something.
SEED = 20260912


def _sequences(rng, count, length):
    """Return distinct random nucleotide sequences.

    :param rng: the random generator to draw from.
    :param count: how many sequences to return.
    :param length: how long each one is.
    :returns: a list of sequences.
    """
    out = []
    while len(out) < count:
        candidate = "".join(rng.choice("ACGT") for _ in range(length))
        if candidate not in out:
            out.append(candidate)
    return out


def _write_table(path, sequences, prefix):
    """Write a barcode reference table the search can read.

    :param path: where to write it.
    :param sequences: the barcode sequences.
    :param prefix: what to call them, numbered from one.
    :returns: None.
    """
    lines = ["name,sequence"]
    lines += [f"{prefix}{index + 1},{sequence}"
              for index, sequence in enumerate(sequences)]
    path.write_text("\n".join(lines) + "\n")


def _write_fastq(path, reads):
    """Write reads as a gzip compressed sequencing file.

    :param path: where to write it.
    :param reads: the read sequences.
    :returns: None.
    """
    with gzip.open(path, "wt") as handle:
        for index, read in enumerate(reads):
            handle.write(f"@read{index}\n{read}\n+\n{'I' * len(read)}\n")


@pytest.fixture
def sequencing_run(tmp_path):
    """A folder of reads whose barcode layout is known exactly.

    Every read of the first mate is filler, then the anchor, then a column
    barcode, then a guide barcode, then filler to length. The second mate is
    the reverse complement of the first, which is what a mate pair actually
    is and what made the real run fail. The row table is the control: its
    thirty-two barcodes are in neither mate, so whatever rate it reports is
    coincidence and nothing else.

    :param tmp_path: pytest's per-test directory.
    :returns: a mapping of the paths and the layout the reads were built to.
    """
    rng = random.Random(SEED)
    columns = _sequences(rng, 4, 8)
    guides = _sequences(rng, 8, 20)
    decoys = _sequences(rng, 32, 8)

    folder = tmp_path / "fastq"
    folder.mkdir()
    column_csv = tmp_path / "barcodes_column.csv"
    grna_csv = tmp_path / "barcodes_grna.csv"
    row_csv = tmp_path / "barcodes_row.csv"
    _write_table(column_csv, columns, "c")
    _write_table(grna_csv, guides, "g")
    _write_table(row_csv, decoys, "r")

    forward = []
    for _ in range(READS):
        lead = "".join(rng.choice("ACGT") for _ in range(LEAD))
        body = lead + ANCHOR + rng.choice(columns) + rng.choice(guides)
        tail = "".join(rng.choice("ACGT")
                       for _ in range(READ_LENGTH - len(body)))
        forward.append(body + tail)
    _write_fastq(folder / "SAMPLE_R1_001.fastq.gz", forward)
    _write_fastq(folder / "SAMPLE_R2_001.fastq.gz",
                 [reverse_complement(read) for read in forward])

    return {
        "folder": folder,
        "column_csv": column_csv,
        "grna_csv": grna_csv,
        "row_csv": row_csv,
        # Where the barcodes sit in the first mate, which is what the search
        # has to rediscover from the reads alone.
        "anchor_start": LEAD,
        "window_start": LEAD + len(ANCHOR),
        "window_length": 8 + 20,
    }


def _screen(qtbot, sequencing_run, **kwargs):
    """Build a Map Barcodes screen pointed at the fixture, with the search on.

    :param qtbot: pytest-qt bot that adopts the widget for cleanup.
    :param sequencing_run: the fixture's paths.
    :param kwargs: passed to the panel, which is how a test asks for an
        unthreaded search or a smaller sample.
    :returns: the screen and its search panel.
    """
    screen = AppScreen(app_key="map_barcodes")
    qtbot.addWidget(screen)
    model = screen._settings_model
    model.set_value_for_key("src", str(sequencing_run["folder"]))
    model.set_value_for_key("column_csv", str(sequencing_run["column_csv"]))
    model.set_value_for_key("grna_csv", str(sequencing_run["grna_csv"]))
    model.set_value_for_key("row_csv", str(sequencing_run["row_csv"]))
    model.set_value_for_key("target_sequence", ANCHOR)
    model.set_value_for_key("offset_start", -8)
    model.set_value_for_key("window_length", 89)
    options = {"threaded": False, "max_reads": READS, "chunk_reads": CHUNK,
               "read_sample": 25}
    options.update(kwargs)
    panel = map_barcodes.install_barcode_search(screen, **options)
    assert panel is not None
    return screen, panel


def _rows(panel):
    """Return the findings table as a list of dictionaries by column name.

    Read out of the widget rather than out of the report, because what is on
    screen is the claim being tested.

    :param panel: the search panel.
    :returns: one dictionary per row.
    """
    table = panel.findings
    headers = [table.horizontalHeaderItem(column).text()
               for column in range(table.columnCount())]
    out = []
    for row in range(table.rowCount()):
        out.append({headers[column]: table.item(row, column).text()
                    for column in range(table.columnCount())})
    return out


def _percent(text):
    """Return a percentage cell as a number.

    :param text: the cell's text, which may be the "under a hundredth of a
        percent" form the panel prints for a vanishing rate.
    :returns: the percentage as a float.
    """
    text = str(text).strip().lstrip("<").rstrip("%")
    return float(text)


# ---------------------------------------------------------------------------
# The chance rate, which is the whole point
# ---------------------------------------------------------------------------


def test_the_chance_rate_sits_directly_beside_the_observed_rate(
        qtbot, qt_theme_applied, sequencing_run):
    """Neither number means anything without the other, so they are neighbours.

    A wide table with the two at opposite ends is a table a horizontal scroll
    can separate, and a reader who sees one column of percentages believes it.
    """
    _screen_, panel = _screen(qtbot, sequencing_run)
    panel.start_search()

    columns = list(map_barcodes.SEARCH_COLUMNS)
    assert columns.index("By chance") == columns.index("Observed") + 1

    rows = _rows(panel)
    assert rows, "the search produced no findings to show"
    for row in rows:
        assert row["Observed"].endswith("%")
        assert row["By chance"].endswith("%")
        assert row["Enrichment"]


def test_a_table_that_is_not_there_reports_its_own_coincidence_rate(
        qtbot, qt_theme_applied, sequencing_run):
    """The decoy matches several percent of reads and is still absent.

    Thirty-two eight base barcodes in a hundred and fifty base read match
    somewhere by accident in about seven percent of reads. The panel has to
    show that rate honestly, show the rate chance would give beside it, and
    still refuse to call the table found.
    """
    _screen_, panel = _screen(qtbot, sequencing_run)
    panel.start_search()

    decoy = [row for row in _rows(panel) if row["Barcode"] == "row"]
    assert decoy, "the row table was not measured at all"
    observed = max(_percent(row["Observed"]) for row in decoy)
    assert observed > 1.0, (
        "the decoy did not produce the coincidence this test needs; it is "
        "meant to match several percent of reads by accident")
    for row in decoy:
        chance = _percent(row["By chance"])
        assert chance > 1.0, (
            "a table whose barcodes match several percent of reads by "
            "accident must not show a chance rate of nothing")
        assert row["Verdict"] == ABSENT
    best = panel.report().best_for_role("row")
    assert best is None, "a coincidence was promoted to a finding"


def test_a_barcode_that_is_there_is_called_present_in_both_mates(
        qtbot, qt_theme_applied, sequencing_run):
    """The same barcode is plain in one mate and flipped in the other.

    That is what a mate pair is, and mistaking it for an absent barcode is
    the failure the whole panel was built for.
    """
    _screen_, panel = _screen(qtbot, sequencing_run)
    panel.start_search()

    rows = _rows(panel)
    plain = [row for row in rows
             if row["Barcode"] == "grna" and row["Mate"] == "R1"
             and row["Orientation"] == "as stored"]
    flipped = [row for row in rows
               if row["Barcode"] == "grna" and row["Mate"] == "R2"
               and row["Orientation"] == "reverse complemented"]
    assert plain and flipped
    assert plain[0]["Verdict"] == PRESENT
    assert flipped[0]["Verdict"] == PRESENT
    assert _percent(plain[0]["Observed"]) > 90.0
    assert _percent(plain[0]["By chance"]) < 1.0


# ---------------------------------------------------------------------------
# Running it: off the GUI thread, a chunk at a time, and stoppable
# ---------------------------------------------------------------------------


def test_the_search_refines_as_it_goes_rather_than_arriving_at_the_end(
        qtbot, qt_theme_applied, sequencing_run):
    """One report per chunk of reads, each over more reads than the last."""
    _screen_, panel = _screen(qtbot, sequencing_run)
    seen = []
    said = []
    panel.search_updated.connect(seen.append)
    panel.search_updated.connect(
        lambda _report: said.append(panel.status.text()))

    panel.start_search()

    assert len(seen) == READS // CHUNK
    assert [report.reads for report in seen] == sorted(
        report.reads for report in seen)
    assert seen[-1].complete
    assert panel.report().reads == READS * 2
    assert not panel.is_searching()
    # While it runs it says which sample it is reading and how far it has
    # got; when it stops it says what it established.
    assert "SAMPLE" in said[0]
    assert str(seen[0].reads) in said[0]
    assert "grna" in panel.status.text()


def test_nothing_is_read_on_the_gui_thread(
        qtbot, qt_theme_applied, sequencing_run):
    """Starting a search returns before a single read has been counted.

    The real path, on a worker thread. If any part of the search ran where it
    was started, the first report would already exist by the time the call
    returned -- which is exactly how a screen freezes on a FASTQ file that
    lives on a network share.
    """
    _screen_, panel = _screen(qtbot, sequencing_run, threaded=True)

    with qtbot.waitSignal(panel.search_finished, timeout=60000):
        assert panel.start_search() is True
        assert panel.report() is None
        assert panel.is_searching()

    assert panel.report() is not None
    assert panel.report().complete
    assert not panel.is_searching()


def test_a_cancelled_search_stops_where_it_was_and_proposes_nothing(
        qtbot, qt_theme_applied, sequencing_run):
    """Cancelling mid-search leaves a partial measurement and no advice.

    The cancel arrives from a handler on the first report, which is the same
    place a button press would arrive from and the only way to be certain the
    search was interrupted rather than finished.
    """
    _screen_, panel = _screen(qtbot, sequencing_run)
    seen = []
    finished = []
    panel.search_updated.connect(seen.append)
    panel.search_updated.connect(lambda _report: panel.cancel_search())
    panel.search_finished.connect(finished.append)

    panel.start_search()

    assert len(seen) == 1, "the search carried on after it was cancelled"
    assert finished == [None], (
        "a cancelled search must not hand out a finished report")
    assert not panel.is_searching()
    assert panel.proposal() is None
    assert panel.proposed_changes() == ()
    assert not panel.apply_button.isEnabled()
    assert "cancel" in panel.status.text().lower()
    # The measurement it did take is still on screen. It is honest about how
    # few reads it rests on, and throwing it away would tell the user less.
    assert panel.findings.rowCount() > 0


def test_a_folder_with_no_reads_says_so_instead_of_searching(
        qtbot, qt_theme_applied, sequencing_run, tmp_path):
    """A plan that cannot be made is a sentence about the form, not a crash."""
    empty = tmp_path / "empty"
    empty.mkdir()
    screen, panel = _screen(qtbot, sequencing_run)
    screen._settings_model.set_value_for_key("src", str(empty))
    finished = []
    panel.search_finished.connect(finished.append)

    panel.start_search()

    assert finished == [None]
    assert not panel.is_searching()
    assert str(empty) in panel.status.text()
    assert panel.findings.rowCount() == 0


def test_one_sample_is_read_and_the_rest_are_counted(
        sequencing_run, tmp_path):
    """A screen's folder holds several samples and one of them settles it.

    Which mate carries which barcode, which way round the reference tables
    are stored and where in the read the barcodes sit are properties of the
    library rather than of a sample. Reading every sample would cost time to
    re-measure what is already known, so the others are counted and said out
    loud instead of being quietly skipped.
    """
    folder = sequencing_run["folder"]
    for mate in ("R1", "R2"):
        (folder / f"OTHER_{mate}_001.fastq.gz").write_bytes(
            (folder / f"SAMPLE_{mate}_001.fastq.gz").read_bytes())

    plan = map_barcodes.plan_barcode_search({
        "src": str(folder),
        "column_csv": str(sequencing_run["column_csv"]),
        "grna_csv": str(sequencing_run["grna_csv"]),
        "row_csv": str(sequencing_run["row_csv"]),
        "target_sequence": ANCHOR,
    })

    assert plan.problem == ""
    assert plan.sample == "OTHER"
    assert plan.other_samples == 1
    assert sorted(plan.fastq_files) == ["R1", "R2"]
    assert [role for _name, _path, role in plan.reference_tables] == [
        "column", "grna", "row"]
    assert plan.anchor == ANCHOR


def test_a_search_with_no_reference_table_says_so(tmp_path, sequencing_run):
    """Nothing to look for is a sentence about the form, not an empty table."""
    plan = map_barcodes.plan_barcode_search({
        "src": str(sequencing_run["folder"]),
        "column_csv": str(tmp_path / "missing.csv"),
        "grna_csv": "",
        "row_csv": "",
    })

    assert plan.reference_tables == ()
    assert "reference" in plan.problem


# ---------------------------------------------------------------------------
# The reads themselves
# ---------------------------------------------------------------------------


def test_the_reads_are_shown_one_per_row_with_their_matches_coloured(
        qtbot, qt_theme_applied, sequencing_run):
    """Every barcode type searched for gets a colour, found or not.

    A type that was searched for and found nowhere still belongs in the
    legend: its absence is the answer to a question the user asked.
    """
    _screen_, panel = _screen(qtbot, sequencing_run)
    panel.start_search()

    assert panel.reads.row_count() == 25
    kinds = panel.reads.kinds()
    assert set(kinds) == {"column", "grna", "row", "anchor"}
    colours = {kind: panel.reads.colour_for(kind) for kind in kinds}
    assert all(colours.values())
    assert len(set(colours.values())) == len(kinds), (
        "two barcode types were given the same colour")

    # And the spans handed to the view are the real positions, including the
    # anchor. The anchor is the landmark the proposed window is measured
    # from, so a legend entry for it with nothing ever painted in that colour
    # would be worse than leaving it out.
    sampled = map_barcodes._sample_annotated_reads(
        str(sequencing_run["folder"] / "SAMPLE_R1_001.fastq.gz"),
        tuple(panel._tables), ANCHOR, 5)
    assert not sampled["error"]
    first = sampled["rows"][0]
    kinds_in_read = {span[2] for span in first[1]}
    assert {"anchor", "column", "grna"} <= kinds_in_read
    anchor_span = [span for span in first[1] if span[2] == "anchor"][0]
    assert anchor_span[0] == sequencing_run["anchor_start"]
    assert first[0][anchor_span[0]:anchor_span[1]] == ANCHOR


# ---------------------------------------------------------------------------
# Applying, which is a decision rather than a consequence
# ---------------------------------------------------------------------------


def test_a_finished_search_writes_nothing_into_the_form_by_itself(
        qtbot, qt_theme_applied, sequencing_run):
    """The search proposes. Until Apply is pressed, the form is untouched."""
    screen, panel = _screen(qtbot, sequencing_run)
    before = dict(screen._settings_model.collect())

    panel.start_search()

    after = dict(screen._settings_model.collect())
    assert after == before, (
        "the search changed a setting nobody asked it to change")
    assert panel.proposed_changes(), (
        "this search had nothing to propose, so the test proves nothing")
    assert panel.apply_button.isEnabled()
    keys = {key for key, _present, _value in panel.proposed_changes()}
    assert {"offset_start", "window_length"} <= keys


def test_apply_writes_the_window_the_reads_actually_carry(
        qtbot, qt_theme_applied, sequencing_run):
    """Pressed, it writes exactly what the search measured and nothing else."""
    screen, panel = _screen(qtbot, sequencing_run)
    applied = []
    panel.settings_applied.connect(applied.append)
    panel.start_search()
    proposed = dict((key, value)
                    for key, _present, value in panel.proposed_changes())

    written = panel.apply_proposal()

    settings = screen._settings_model.collect()
    assert set(written) == set(proposed)
    assert applied and applied[0] == proposed
    # The window the fixture built the reads to carry, rediscovered from the
    # reads and expressed relative to the anchor, which is how the mapping
    # run locates it.
    assert settings["offset_start"] == (
        sequencing_run["window_start"] - sequencing_run["anchor_start"])
    assert settings["window_length"] == sequencing_run["window_length"]
    # And it cannot be applied twice: there is nothing left to write.
    assert not panel.apply_button.isEnabled()
    assert panel.apply_proposal() == ()


def test_the_proposal_says_which_table_has_to_be_flipped(
        qtbot, qt_theme_applied, sequencing_run):
    """The orientation has no settings key, so it has to be a sentence.

    This is the finding that would have saved the run that produced no
    counts, and there is nowhere on the form for it to land. A panel that
    dropped it because it did not fit a key would have reproduced the
    original failure with better numbers.
    """
    screen, panel = _screen(qtbot, sequencing_run)
    screen._settings_model.set_value_for_key("mode", "single")
    screen._settings_model.set_value_for_key("single_direction", "R2")

    panel.start_search()

    text = panel.proposal_label.text()
    assert "offset_start" in text
    assert "row" in text, (
        "the proposal does not say that the row barcodes were never "
        "established, which is the reason a run would still map nothing")
    # Both mates carry the barcodes, so the run should read the pair.
    changes = dict((key, value)
                   for key, _present, value in panel.proposed_changes())
    assert changes.get("mode") == "paired"


# ---------------------------------------------------------------------------
# Where it lives
# ---------------------------------------------------------------------------


def test_the_search_is_installed_once_on_map_barcodes_and_starts_folded(
        qtbot, qt_theme_applied, sequencing_run):
    """One panel per screen, hidden behind the toggle that reveals it."""
    screen, panel = _screen(qtbot, sequencing_run)

    assert map_barcodes.install_barcode_search(screen) is panel
    card = screen._barcode_search_card
    toggle = screen._barcode_search_toggle
    # Hidden rather than not-visible: the screen itself is never shown in a
    # test, so every widget on it is invisible for reasons of its own.
    assert card.isHidden()
    toggle.setChecked(True)
    assert not card.isHidden()
    toggle.setChecked(False)
    assert card.isHidden()


def test_another_module_does_not_get_a_barcode_search(qtbot, qt_theme_applied):
    """The panel belongs to the screen whose settings it reads and writes."""
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)

    assert map_barcodes.install_barcode_search(screen) is None
    assert getattr(screen, "_barcode_search", None) is None
