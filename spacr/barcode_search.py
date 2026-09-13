"""Decide which barcode tables a sequencing run really contains.

WHAT IT IS FOR
==============
Map Barcodes has to be told three things it cannot guess: which mate carries
each barcode, whether the reference table is stored in the same orientation as
the reads, and where in the read the barcode sits.  Getting any of them wrong
produces a run that finishes normally and maps nothing, because every stage
after the extraction window is working on the wrong bases.  This module reads a
bounded sample of the reads, measures each reference table against them in both
orientations, and turns the measurements into settings the mapping run can use.
It holds no graphical code and opens no dialogs, so the same engine serves the
live search in the interface, a notebook, and the tests.

WHY A RAW HIT RATE IS NOT EVIDENCE
==================================
A short barcode turns up in a long read by pure coincidence.  Thirty-two
distinct eight base barcodes scanned across a hundred and fifty base read have
about a seven percent chance of a spurious match, so a table that is genuinely
absent still reports roughly seven percent of reads as hits.  A detector that
announced a finding at that rate would send someone hunting for a bug that does
not exist, or would quietly configure a mapping that yields nothing.  Every rate
this module reports therefore arrives with the rate expected by chance for that
table against those reads, and with the ratio between them.  Nothing is called
present unless it stands well clear of its own coincidence rate.

WHY A LARGE ENRICHMENT IS STILL NOT ENOUGH
==========================================
Enrichment over chance answers whether the sequences are in the reads.  It does
not answer whether they are the plate barcodes.  Sequencing adapters are fixed
sequences repeated in every read, and a handful of barcodes in any table will
resemble part of an adapter closely enough to match it.  That produces a large,
perfectly real enrichment driven by two barcodes sitting in the adapter rather
than by a plate.  The measurement that separates the two is position.  A plate
barcode occupies one place in the construct and so lands at one offset in
almost every read, which is exactly why an extraction window can be proposed for
it at all.  A match scattered across the read cannot yield a window and is not
treated as a finding, however enriched it is.

WHAT IT PRODUCES
================
A report carrying one finding per combination of reference table, orientation
and input file.  Each finding holds the observed rate, the chance rate, the
enrichment, the offset distribution, the narrowest offset window that accounts
for most of the hits, how many distinct barcodes of the table were seen, and a
verdict with a sentence saying why that verdict was reached.  The report can be
rendered as a table for a log, and it can be folded into the settings that
:func:`spacr.sequencing.generate_barecode_mapping` reads.

WHAT TO DO NEXT
===============
Read the verdicts before the proposed settings.  A table reported as absent in
both orientations of both mates is usually the wrong reference file rather than
a failed experiment, and the offset column is the quickest way to tell a plate
barcode from an adapter that happens to resemble one.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from itertools import islice
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "AS_GIVEN",
    "REVERSE_COMPLEMENT",
    "PRESENT",
    "ABSENT",
    "INDETERMINATE",
    "ANCHOR_ROLE",
    "DEFAULT_SAMPLE_READS",
    "DEFAULT_CHUNK_READS",
    "MIN_ENRICHMENT",
    "MIN_USABLE_RATE",
    "MAX_OFFSET_SPAN",
    "OFFSET_WINDOW_COVERAGE",
    "MIN_READS_FOR_VERDICT",
    "BarcodeHit",
    "BarcodeTable",
    "OrientationFinding",
    "BarcodeSearchReport",
    "ProposedMapping",
    "reverse_complement",
    "infer_barcode_role",
    "load_barcode_table",
    "load_barcode_tables",
    "iter_fastq_reads",
    "sample_fastq_reads",
    "expected_chance_rate",
    "iter_barcode_search",
    "search_barcodes",
    "propose_map_barcodes_settings",
    "annotate_read",
    "iter_annotated_reads",
]

#: Orientation label for a reference table used exactly as it is stored.
AS_GIVEN = "as_given"
#: Orientation label for a reference table whose sequences were flipped.
REVERSE_COMPLEMENT = "reverse_complement"

#: Verdict for a table that is in the reads, in useful numbers, at one place.
PRESENT = "present"
#: Verdict for a table whose hit rate cannot be told apart from coincidence.
ABSENT = "absent"
#: Verdict for a table the sample cannot yet decide, or that failed a check.
INDETERMINATE = "indeterminate"

#: How many reads a search samples from each file unless told otherwise.
DEFAULT_SAMPLE_READS = 20000
#: How many reads each incremental step consumes from each file.
DEFAULT_CHUNK_READS = 2000

# THE THRESHOLDS, AND THE MEASUREMENTS THAT SET THEM.
#
# These were chosen against a real paired run whose two mates were measured
# read by read, rather than picked for roundness.  On that run the tables that
# were genuinely part of the library reached enrichments of nineteen, twenty
# nine and several million over their own chance rates, while every table that
# was truly absent sat between a fifth of its chance rate and one and a fraction
# times it, the highest coincidence reaching one point zero two.  A cut at three
# leaves the coincidence ceiling far below it and the weakest true signal far
# above it, so neither side is close to the boundary.
MIN_ENRICHMENT = 3.0

# The same run also carried a small number of reads in the wrong orientation,
# about one guide barcode in six hundred, which index hopping and chimeric
# fragments produce in every pooled run.  Those hits are thousands of times
# above chance and completely real, and mapping from them would still be
# hopeless.  A barcode that belongs to the construct appears in most reads, so
# a table found in under a tenth of them is reported honestly as enriched but
# not usable rather than being offered as a source of settings.
MIN_USABLE_RATE = 0.10

# The narrowest run of offsets accounting for most of the hits was one to three
# bases wide for every true finding, the width above one coming from guide
# sequences of twenty or twenty one bases shifting everything downstream of them
# by a base.  For the adapter that masqueraded as a column table the same
# measurement needed forty five bases.  Six bases allows a construct with more
# length variation than this one while still refusing anything adapter shaped.
MAX_OFFSET_SPAN = 6
#: The share of hits the narrowest offset window has to account for.
OFFSET_WINDOW_COVERAGE = 0.80

#: Reads needed before a verdict other than an undecided one is issued.
MIN_READS_FOR_VERDICT = 500

_Z = 1.96
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_PREFIX_LENGTH = 8
_ROLE_SETTING_KEYS = {"row": "row_csv", "column": "column_csv", "grna": "grna_csv"}
#: Role name given to the anchor sequence when it is searched alongside tables.
ANCHOR_ROLE = "anchor"


def reverse_complement(sequence):
    """Return the reverse complement of a nucleotide sequence.

    The translation covers the four bases in either case and leaves an
    ambiguous base as it is, so a read carrying an unresolved position survives
    the flip instead of raising.

    :param sequence: the nucleotide sequence to flip.
    :returns: the reverse complement, as a string.
    """
    return str(sequence).translate(_COMPLEMENT)[::-1]


def infer_barcode_role(path):
    """Guess whether a reference file holds row, column or guide barcodes.

    The guess reads the file name only, because the tables themselves carry no
    field saying what they are.  Guide tables are recognised first, since a
    guide file name mentions neither rows nor columns, and column names are
    recognised before row names.

    :param path: the path of the reference table.
    :returns: one of the role names, or None when the name settles nothing.
    """
    stem = os.path.basename(str(path)).lower()
    if "grna" in stem or "guide" in stem or "sgrna" in stem:
        return "grna"
    if "column" in stem or "col" in stem:
        return "column"
    if "row" in stem:
        return "row"
    return None


@dataclass(frozen=True)
class BarcodeTable:
    """One reference table of named barcode sequences.

    The sequences are held as a mapping from sequence to name so that a match
    can be named without a second search, and the per-length counts are kept
    because the chance rate depends on how many barcodes of each length there
    are rather than on the size of the table alone.

    :param name: the label this table is known by for the rest of the search,
        and the label every finding it produces refers back to.
    :param sequences: mapping from barcode sequence to the name of that
        barcode, so a hit can be named without searching the table again.
    :param role: which part of the screen's identity the table decodes --
        plate row, plate column, guide -- or None for a table searched
        without one.
    :param path: where the table was read from, kept so a proposed mapping can
        name the file rather than the label.
    """

    name: str
    sequences: Mapping[str, str]
    role: Optional[str] = None
    path: Optional[str] = None

    def __post_init__(self):
        """Give the table a private store for the indexes built from it.

        Building the search index over a table of more than a thousand guide
        sequences takes long enough that doing it once per read would make a
        live display unusable, so each index is built once and kept here.
        """
        object.__setattr__(self, "_derived", {})

    @property
    def size(self):
        """Return how many distinct sequences the table holds.

        :returns: the count of distinct barcode sequences.
        """
        return len(self.sequences)

    @property
    def length_counts(self):
        """Return how many distinct barcodes the table holds at each length.

        :returns: a mapping from sequence length to the number of barcodes.
        """
        counts: Counter = Counter()
        for sequence in self.sequences:
            counts[len(sequence)] += 1
        return dict(counts)

    def oriented(self, orientation):
        """Return the same table with its sequences in a chosen orientation.

        :param orientation: either the label for the stored orientation or the
            label for the reverse complemented one.
        :returns: a table whose sequences read in the requested orientation.
        :raises ValueError: when the orientation label is not one of the two.
        """
        if orientation == AS_GIVEN:
            return self
        if orientation != REVERSE_COMPLEMENT:
            raise ValueError(
                "orientation must be "
                f"{AS_GIVEN!r} or {REVERSE_COMPLEMENT!r}; got {orientation!r}")
        cached = self._derived.get("flipped")
        if cached is None:
            flipped = {
                reverse_complement(sequence): name
                for sequence, name in self.sequences.items()
            }
            cached = BarcodeTable(
                name=self.name, sequences=flipped, role=self.role,
                path=self.path)
            self._derived["flipped"] = cached
        return cached

    def matcher(self, orientation):
        """Return the reusable search index for one orientation of the table.

        :param orientation: which orientation of the table to search in.
        :returns: the index, built on first use and kept afterwards.
        """
        oriented = self.oriented(orientation)
        index = oriented._derived.get("matcher")
        if index is None:
            index = _Matcher(oriented)
            oriented._derived["matcher"] = index
        return index

    def barcode_order(self):
        """Return a stable number for each stored sequence of the table.

        A display paints one colour per barcode rather than per table, and the
        number has to be the same every time the same table is loaded, so it
        comes from sorting the sequences rather than from the order of the file.

        :returns: a mapping from stored sequence to its position when sorted.
        """
        order = self._derived.get("order")
        if order is None:
            order = {
                sequence: index
                for index, sequence in enumerate(sorted(self.sequences))
            }
            self._derived["order"] = order
        return order


def load_barcode_table(path, name=None, role=None):
    """Read a reference table of barcodes from a comma separated file.

    The file needs a header naming a sequence column and, for readable output, a
    name column.  Sequences are upper cased and stripped, blank rows are
    skipped, and a sequence repeated under two names keeps the first name so
    that the table stays a mapping.

    :param path: the path of the comma separated reference table.
    :param name: a label for the table; the file stem is used when omitted.
    :param role: the barcode role this table fills; inferred from the file name
        when omitted.
    :returns: the loaded table.
    :raises ValueError: when the file carries no sequence column or no rows.
    """
    path = str(path)
    sequences: Dict[str, str] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [column.strip().lower() for column in (reader.fieldnames or [])]
        if "sequence" not in fields:
            raise ValueError(
                f"{path} has no 'sequence' column; found {reader.fieldnames!r}")
        original = (reader.fieldnames or [])[fields.index("sequence")]
        name_field = None
        if "name" in fields:
            name_field = (reader.fieldnames or [])[fields.index("name")]
        for index, row in enumerate(reader):
            sequence = (row.get(original) or "").strip().upper()
            if not sequence:
                continue
            label = f"row{index + 1}"
            if name_field is not None:
                label = (row.get(name_field) or label).strip() or label
            sequences.setdefault(sequence, label)
    if not sequences:
        raise ValueError(f"{path} holds no barcode sequences")
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]
    if role is None:
        role = infer_barcode_role(path)
    return BarcodeTable(name=name, sequences=sequences, role=role, path=path)


def load_barcode_tables(paths):
    """Read several reference tables at once.

    Any number of tables may be supplied, so a run that carries more than the
    three barcode kinds spaCR grew up with is handled the same way as one that
    carries fewer.

    :param paths: paths of the reference tables, or a mapping from the label a
        table should carry to its path.
    :returns: a tuple of loaded tables in the order given.
    """
    if isinstance(paths, Mapping):
        return tuple(
            load_barcode_table(path, name=label) for label, path in paths.items())
    return tuple(load_barcode_table(path) for path in paths)


def _open_text(path):
    """Open a plain or gzip compressed text file for reading.

    :param path: the path to open.
    :returns: an open text handle.
    """
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def iter_fastq_reads(path, limit=None):
    """Yield read sequences from a sequencing file without loading it.

    The files this runs against are commonly several gigabytes on a network
    share, so records are pulled one at a time and the caller decides when to
    stop.  A record whose four lines are cut short at the end of the file is
    dropped rather than raising, because a truncated final record is a normal
    consequence of copying a file that is still being written.

    :param path: the path of the sequencing file, gzip compressed or plain.
    :param limit: how many reads to yield at most; every read when omitted.
    :yields: the sequence of each read, upper cased.
    """
    if limit is not None and limit <= 0:
        return
    produced = 0
    with _open_text(path) as handle:
        while True:
            header = handle.readline()
            if not header:
                return
            sequence = handle.readline()
            separator = handle.readline()
            quality = handle.readline()
            if not quality or not separator or not sequence:
                return
            yield sequence.strip().upper()
            produced += 1
            if limit is not None and produced >= limit:
                return


def sample_fastq_reads(path, limit=DEFAULT_SAMPLE_READS):
    """Read a bounded sample of sequences from a sequencing file.

    :param path: the path of the sequencing file, gzip compressed or plain.
    :param limit: how many reads to take from the start of the file.
    :returns: a tuple of read sequences.
    """
    return tuple(iter_fastq_reads(path, limit=limit))


def expected_chance_rate(table, read_lengths):
    """Return the share of reads expected to contain a barcode by coincidence.

    A read is treated as a run of independent bases drawn evenly from the four
    of them, so one barcode of a given length matches at one position with a
    probability of one over four raised to that length, and a read offers one
    more position than the difference between its length and the barcode's.
    The probability that no barcode of the table matches anywhere is the product
    over every barcode and every position of the probability that it does not
    match there, and the answer is one minus that product.  Reads of different
    lengths are averaged.

    The even base assumption is the weak point and it was checked rather than
    assumed.  On a real run the tables that were genuinely absent matched
    within two tenths of a percentage point of the rate this returns, which is
    close enough for the ratio between observed and expected to be trusted.

    :param table: the reference table whose coincidence rate is wanted.
    :param read_lengths: a read length, an iterable of them, or a mapping from
        read length to how many reads had it.
    :returns: the expected share of reads holding at least one barcode.
    """
    if isinstance(read_lengths, Mapping):
        histogram = dict(read_lengths)
    elif isinstance(read_lengths, int):
        histogram = {read_lengths: 1}
    else:
        histogram = dict(Counter(read_lengths))
    total = sum(histogram.values())
    if not total:
        return 0.0
    length_counts = table.length_counts
    weighted = 0.0
    for read_length, occurrences in histogram.items():
        log_miss = 0.0
        for barcode_length, barcodes in length_counts.items():
            positions = max(read_length - barcode_length + 1, 0)
            if not positions:
                continue
            log_miss += barcodes * positions * math.log1p(-4.0 ** -barcode_length)
        weighted += occurrences * (1.0 - math.exp(log_miss))
    return weighted / total


def _wilson_interval(successes, trials, z=_Z):
    """Return a confidence interval for a proportion that behaves near zero.

    The usual interval around the observed share collapses to a point when
    nothing was seen, which would let a handful of reads declare a table absent.
    The score interval keeps a sensible width there, which is what makes the
    verdicts settle down as a live search takes in more reads instead of
    flickering.

    :param successes: how many reads held a barcode.
    :param trials: how many reads were examined.
    :param z: the standard score for the wanted confidence.
    :returns: the lower and upper bound of the interval.
    """
    if trials <= 0:
        return 0.0, 1.0
    observed = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (observed + z_squared / (2.0 * trials)) / denominator
    spread = math.sqrt(
        observed * (1.0 - observed) / trials
        + z_squared / (4.0 * trials * trials))
    half = (z / denominator) * spread
    return max(0.0, centre - half), min(1.0, centre + half)


def _narrowest_offset_window(offsets, coverage=OFFSET_WINDOW_COVERAGE):
    """Return the shortest stretch of the read holding most of the hits.

    A plate barcode sits at one place in the construct, so its hits pile up in a
    couple of adjacent offsets and this stretch is a few bases wide.  A sequence
    that matched an adapter, or matched by coincidence, is spread out and needs
    a stretch tens of bases wide to gather the same share, which is how the two
    are told apart.

    :param offsets: a mapping from start offset to how many reads hit there.
    :param coverage: the share of hits the stretch has to account for.
    :returns: the first offset of the stretch and its width in bases, or a pair
        of zeroes when there were no hits.
    """
    if not offsets:
        return 0, 0
    starts = sorted(offsets)
    total = sum(offsets.values())
    needed = coverage * total
    best_start = starts[0]
    best_width = starts[-1] - starts[0] + 1
    low = 0
    running = 0
    for high in range(len(starts)):
        running += offsets[starts[high]]
        while low < high and running - offsets[starts[low]] >= needed:
            running -= offsets[starts[low]]
            low += 1
        if running >= needed:
            width = starts[high] - starts[low] + 1
            if width < best_width:
                best_width = width
                best_start = starts[low]
    return best_start, best_width


class _Matcher:
    """Find the barcodes of one oriented table inside a read.

    Scanning a read against a table of over a thousand guide sequences one
    sequence at a time is far too slow to drive a live display, so the read is
    walked once and each position is looked up by its first few bases.  Only a
    position whose prefix belongs to some barcode costs a second lookup, which
    for a table of long barcodes reduces the work by orders of magnitude.
    """

    def __init__(self, table):
        self.table = table
        self.lengths = sorted({len(sequence) for sequence in table.sequences})
        self.shortest = self.lengths[0] if self.lengths else 0
        self.prefix_length = min(_PREFIX_LENGTH, self.shortest) if self.shortest else 0
        prefixes: Dict[str, set] = {}
        for sequence in table.sequences:
            prefixes.setdefault(sequence[: self.prefix_length], set()).add(len(sequence))
        self.prefixes = {key: sorted(value) for key, value in prefixes.items()}

    def find_first(self, read, start=0):
        """Return the earliest barcode occurrence at or after a position.

        :param read: the read sequence to search.
        :param start: the first position to consider.
        :returns: the offset, the length and the barcode name, or None.
        """
        if not self.shortest:
            return None
        sequences = self.table.sequences
        prefixes = self.prefixes
        prefix_length = self.prefix_length
        last = len(read) - self.shortest
        for offset in range(max(start, 0), last + 1):
            candidates = prefixes.get(read[offset: offset + prefix_length])
            if not candidates:
                continue
            for length in candidates:
                end = offset + length
                if end > len(read):
                    continue
                name = sequences.get(read[offset:end])
                if name is not None:
                    return offset, length, name
        return None

    def find_all(self, read):
        """Return every non overlapping barcode occurrence in a read.

        :param read: the read sequence to search.
        :returns: a list of offset, length and name triples, left to right.
        """
        hits = []
        position = 0
        while True:
            found = self.find_first(read, position)
            if found is None:
                return hits
            hits.append(found)
            position = found[0] + found[1]


@dataclass(frozen=True)
class OrientationFinding:
    """What one reference table did against one file in one orientation.

    Everything needed to judge the finding travels with it, so a caller never
    has to recompute the coincidence rate in order to decide whether the
    observed rate means anything.

    :param table: label of the reference table this finding is about.
    :param role: the role that table fills, or None when it was searched
        without one.
    :param file_label: the read file the table was searched against.
    :param orientation: whether the table matched ``AS_GIVEN`` or
        ``REVERSE_COMPLEMENT``.
    :param table_path: where the table was read from, or None when it was
        supplied inline.
    :param reads: how many reads were examined to reach this finding.
    :param hits: how many of those reads contained a barcode from the table.
    :param observed_rate: ``hits`` divided by ``reads``.
    :param expected_rate: the rate the same table would reach by chance, given
        how many barcodes it holds at each length. The observed rate means
        nothing on its own; this is what makes it readable.
    :param enrichment: ``observed_rate`` divided by ``expected_rate``.
    :param offset_start: earliest start offset counted as the barcode's usual
        position.
    :param offset_span: how many consecutive offsets that position covers.
    :param modal_offset: the single commonest start offset, or None when
        nothing was found.
    :param barcode_lengths: the distinct lengths present in the table. The
        extraction window has to reach past the longest, not the modal one.
    :param distinct_barcodes_seen: how many different barcodes of the table
        were actually observed. A table that keeps matching the same one
        sequence is matching something other than its barcodes.
    :param table_size: how many distinct sequences the table holds.
    :param top_barcode_share: fraction of hits taken by the commonest single
        barcode. Near one, alongside a low ``distinct_barcodes_seen``, is the
        signature of a spurious match rather than a real one.
    :param verdict: ``PRESENT``, ``ABSENT`` or ``INDETERMINATE``.
    :param reason: why that verdict was reached, worded to be shown to a user.
    :param offset_counts: hit count at each start offset, for the histogram.
    """

    table: str
    role: Optional[str]
    file_label: str
    orientation: str
    table_path: Optional[str]
    reads: int
    hits: int
    observed_rate: float
    expected_rate: float
    enrichment: float
    offset_start: int
    offset_span: int
    modal_offset: Optional[int]
    barcode_lengths: Tuple[int, ...]
    distinct_barcodes_seen: int
    table_size: int
    top_barcode_share: float
    verdict: str
    reason: str
    offset_counts: Mapping[int, int] = field(default_factory=dict)

    @property
    def window_end(self):
        """Return the first offset past the barcode in its usual position.

        The proposed extraction window has to reach the end of the last barcode
        rather than its start, and a barcode set whose members differ in length
        reaches furthest at its longest member.

        :returns: the offset just past the end of the barcode.
        """
        if not self.barcode_lengths or self.offset_span <= 0:
            return self.offset_start
        return self.offset_start + self.offset_span - 1 + max(self.barcode_lengths)

    def offset_histogram(self, read_length=None):
        """Return the hit count at each start offset as an array.

        The array is indexed by offset so that a caller can plot it directly
        without first working out which offsets were populated.

        :param read_length: how long the array should be; long enough to hold
            the largest observed offset when omitted.
        :returns: an array of hit counts indexed by start offset.
        """
        largest = max(self.offset_counts) if self.offset_counts else -1
        size = max(read_length or 0, largest + 1, 0)
        histogram = np.zeros(size, dtype=np.int64)
        for offset, count in self.offset_counts.items():
            histogram[offset] = count
        return histogram


@dataclass(frozen=True)
class BarcodeSearchReport:
    """Every finding from one search, plus how much was read to get them.

    A report from a partly finished search has the same shape as a finished
    one and is meant to be shown, so a live display can render each refinement
    without special casing the first one.

    :param findings: one :class:`OrientationFinding` per table, file and
        orientation examined.
    :param reads_by_file: how many reads were examined in each file.
    :param read_length_by_file: mean read length per file, which is what sizes
        an offset histogram.
    :param complete: whether the search finished. False on the partial reports
        a live display renders while it is still refining.
    """

    findings: Tuple[OrientationFinding, ...]
    reads_by_file: Mapping[str, int]
    read_length_by_file: Mapping[str, float]
    complete: bool = False

    @property
    def reads(self):
        """Return how many reads were examined across all files.

        :returns: the total read count.
        """
        return sum(self.reads_by_file.values())

    def for_table(self, table):
        """Return the findings belonging to one reference table.

        :param table: the label of the table.
        :returns: a tuple of findings, strongest enrichment first.
        """
        chosen = [item for item in self.findings if item.table == table]
        return tuple(sorted(chosen, key=lambda item: -item.enrichment))

    def best_for_role(self, role):
        """Return the most convincing present finding for a barcode role.

        A role may be filled by more than one table and by either mate, and the
        caller wants the one that would actually be used.  Only findings that
        cleared every check are considered, so this returns nothing when the
        role was not established.

        :param role: the barcode role wanted.
        :returns: the finding with the highest observed rate, or None.
        """
        candidates = [
            item for item in self.findings
            if item.role == role and item.verdict == PRESENT
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.observed_rate)

    def roles(self):
        """Return every barcode role the searched tables were given.

        :returns: a tuple of role names, in a stable order, without the anchor.
        """
        seen = []
        for item in self.findings:
            if item.role and item.role != ANCHOR_ROLE and item.role not in seen:
                seen.append(item.role)
        return tuple(seen)

    def format_table(self):
        """Render the findings as fixed width text for a log or a console.

        The expected rate sits beside the observed one on purpose.  A reader
        who sees only the observed column will believe a coincidence, and the
        whole point of the module is that the two numbers are never separated.

        :returns: the rendered table, as a string.
        """
        header = (
            f"{'table':<26} {'role':<8} {'file':<6} {'orient':<19} "
            f"{'reads':>7} {'obs':>8} {'chance':>8} {'enrich':>10} "
            f"{'offset':>8} {'span':>5} {'seen':>9}  verdict"
        )
        lines = [header, "-" * len(header)]
        for item in sorted(
                self.findings,
                key=lambda row: (row.role or "", row.table, row.file_label,
                                 row.orientation)):
            enrichment = _format_enrichment(item.enrichment)
            offset = "-" if item.modal_offset is None else str(item.modal_offset)
            lines.append(
                f"{item.table[:26]:<26} {(item.role or '-'):<8} "
                f"{item.file_label[:6]:<6} {item.orientation:<19} "
                f"{item.reads:>7} {item.observed_rate * 100:7.2f}% "
                f"{item.expected_rate * 100:7.2f}% {enrichment:>10} "
                f"{offset:>8} {item.offset_span:>5} "
                f"{item.distinct_barcodes_seen:>4}/{item.table_size:<4}  "
                f"{item.verdict}")
        return "\n".join(lines)


class _SearchState:
    """Running counts for one search, and the report built from them."""

    def __init__(self, tables, file_labels):
        self.tables = tuple(tables)
        self.file_labels = tuple(file_labels)
        self.matchers = {
            (table.name, orientation): table.matcher(orientation)
            for table in self.tables
            for orientation in (AS_GIVEN, REVERSE_COMPLEMENT)
        }
        self.reads = {label: 0 for label in self.file_labels}
        self.lengths = {label: Counter() for label in self.file_labels}
        self.hits = {}
        self.offsets = {}
        self.names = {}
        for label in self.file_labels:
            for table in self.tables:
                for orientation in (AS_GIVEN, REVERSE_COMPLEMENT):
                    key = (label, table.name, orientation)
                    self.hits[key] = 0
                    self.offsets[key] = Counter()
                    self.names[key] = Counter()

    def consume(self, label, reads):
        """Fold a batch of reads from one file into the running counts.

        :param label: which file the reads came from.
        :param reads: the read sequences.
        :returns: how many reads were folded in.
        """
        taken = 0
        for read in reads:
            taken += 1
            self.lengths[label][len(read)] += 1
            for table in self.tables:
                for orientation in (AS_GIVEN, REVERSE_COMPLEMENT):
                    found = self.matchers[(table.name, orientation)].find_first(read)
                    if found is None:
                        continue
                    key = (label, table.name, orientation)
                    self.hits[key] += 1
                    self.offsets[key][found[0]] += 1
                    self.names[key][found[2]] += 1
        self.reads[label] += taken
        return taken

    def snapshot(self, complete=False):
        """Build an immutable report from the counts gathered so far.

        :param complete: whether the search reached its read budget.
        :returns: the report.
        """
        findings = []
        for label in self.file_labels:
            reads = self.reads[label]
            lengths = self.lengths[label]
            for table in self.tables:
                expected = expected_chance_rate(table, lengths)
                for orientation in (AS_GIVEN, REVERSE_COMPLEMENT):
                    key = (label, table.name, orientation)
                    findings.append(
                        _build_finding(
                            table, label, orientation, reads,
                            self.hits[key], expected, self.offsets[key],
                            self.names[key]))
        mean_lengths = {}
        for label in self.file_labels:
            counts = self.lengths[label]
            total = sum(counts.values())
            mean_lengths[label] = (
                sum(length * n for length, n in counts.items()) / total
                if total else 0.0)
        return BarcodeSearchReport(
            findings=tuple(findings),
            reads_by_file=dict(self.reads),
            read_length_by_file=mean_lengths,
            complete=complete)


def _build_finding(table, label, orientation, reads, hits, expected, offsets, names):
    """Turn one set of running counts into a finding with a verdict.

    :param table: the reference table the counts belong to.
    :param label: which file was read.
    :param orientation: which orientation of the table was searched.
    :param reads: how many reads were examined.
    :param hits: how many of them held a barcode.
    :param expected: the share of reads expected to hit by coincidence.
    :param offsets: how many hits started at each offset.
    :param names: how many hits each barcode of the table accounted for.
    :returns: the finding.
    """
    observed = hits / reads if reads else 0.0
    if expected > 0.0:
        enrichment = observed / expected
    else:
        enrichment = math.inf if observed > 0.0 else 0.0
    window_start, window_span = _narrowest_offset_window(offsets)
    modal = offsets.most_common(1)[0][0] if offsets else None
    top_share = (names.most_common(1)[0][1] / hits) if hits and names else 0.0
    verdict, reason = _decide(
        reads, hits, observed, expected, enrichment, window_span,
        len(names), table.size, top_share)
    return OrientationFinding(
        table=table.name,
        role=table.role,
        file_label=label,
        orientation=orientation,
        table_path=table.path,
        reads=reads,
        hits=hits,
        observed_rate=observed,
        expected_rate=expected,
        enrichment=enrichment,
        offset_start=window_start,
        offset_span=window_span,
        modal_offset=modal,
        barcode_lengths=tuple(sorted(table.length_counts)),
        distinct_barcodes_seen=len(names),
        table_size=table.size,
        top_barcode_share=top_share,
        verdict=verdict,
        reason=reason,
        offset_counts=dict(offsets))


def _decide(reads, hits, observed, expected, enrichment, span,
            distinct_seen, table_size, top_share):
    """Choose a verdict for one finding and say in a sentence why.

    The three questions are asked in order and each one can end the matter.
    The first asks whether the hit rate can be told apart from coincidence at
    all, and a table that fails it is absent no matter how large the raw rate
    looked.  The second asks whether enough reads carry the barcode for a
    mapping to be built on them, which separates a construct from the trace of
    another sample that index hopping left behind.  The third asks whether the
    hits gather at one place in the read, which separates a plate barcode from
    a sequence that happens to resemble the sequencing adapter.

    :param reads: how many reads were examined.
    :param hits: how many of them held a barcode.
    :param observed: the share of reads that held one.
    :param expected: the share expected to hold one by coincidence.
    :param enrichment: the observed share over the expected one.
    :param span: the width of the narrowest stretch holding most of the hits.
    :param distinct_seen: how many barcodes of the table were actually seen.
    :param table_size: how many barcodes the table holds.
    :param top_share: the share of hits the single commonest barcode took.
    :returns: the verdict and the sentence explaining it.
    """
    if reads < MIN_READS_FOR_VERDICT:
        return INDETERMINATE, (
            f"only {reads} reads sampled so far, which is too few to separate "
            f"a real match from coincidence")
    if not hits:
        # Deciding this by the ratio alone fails for a table whose coincidence
        # rate rounds to nothing, such as a single long anchor sequence, because
        # the bar the observation has to clear is then also nothing and no
        # observation can fall below it.  Nothing matched, so nothing is there.
        return ABSENT, (
            f"not one of {reads} reads carried a barcode from this table in "
            f"this orientation")
    low, high = _wilson_interval(hits, reads)
    threshold = expected * MIN_ENRICHMENT
    if high < threshold:
        return ABSENT, (
            f"{observed * 100:.2f}% of reads matched against "
            f"{expected * 100:.2f}% expected by chance, which is not far "
            f"enough above coincidence to mean anything")
    if low <= threshold:
        return INDETERMINATE, (
            f"{observed * 100:.2f}% of reads matched against "
            f"{expected * 100:.2f}% expected by chance, and the sample is not "
            f"yet large enough to tell those apart")
    if high < MIN_USABLE_RATE:
        rate = (
            "more than a thousand times" if enrichment >= 1000.0
            else f"{enrichment:.0f} times")
        return INDETERMINATE, (
            f"clearly above chance at {rate} the coincidence rate, but only "
            f"{observed * 100:.2f}% of reads carry it, which is the level index "
            f"hopping produces and is too little to map from")
    if low < MIN_USABLE_RATE:
        return INDETERMINATE, (
            f"above chance, but at {observed * 100:.2f}% of reads it is not "
            f"yet clear whether enough reads carry it to map from")
    if span > MAX_OFFSET_SPAN:
        return INDETERMINATE, (
            f"{observed * 100:.2f}% of reads matched, {enrichment:.1f} times "
            f"the coincidence rate, but the matches are scattered over "
            f"{span} bases instead of sitting at one position, and "
            f"{top_share * 100:.0f}% of them came from a single barcode, so "
            f"this looks like adapter rather than a plate")
    ratio = (
        "more than a thousand times" if enrichment >= 1000.0
        else f"{enrichment:.1f} times")
    return PRESENT, (
        f"{observed * 100:.2f}% of reads matched against "
        f"{expected * 100:.2f}% expected by chance, {ratio} coincidence, with "
        f"{distinct_seen} of {table_size} barcodes seen and the matches "
        f"confined to {span} base(s)")


def _as_labelled_files(fastq_files):
    """Return the input files as an ordered mapping from label to path.

    :param fastq_files: a path, a sequence of paths, or a mapping from label to
        path.
    :returns: a dictionary from label to path, in the order given.
    :raises ValueError: when no file was supplied.
    """
    if isinstance(fastq_files, Mapping):
        labelled = {str(label): str(path) for label, path in fastq_files.items()}
    elif isinstance(fastq_files, (str, bytes, os.PathLike)):
        labelled = {_default_label(fastq_files): str(fastq_files)}
    else:
        labelled = {}
        for path in fastq_files:
            label = _default_label(path)
            while label in labelled:
                label = f"{label}_"
            labelled[label] = str(path)
    if not labelled:
        raise ValueError("at least one sequencing file is needed")
    return labelled


def _default_label(path):
    """Return a short label for a sequencing file.

    A paired run names its mates in the file name, and that fragment is what
    someone reading the report expects to see rather than the whole path.

    :param path: the path of the sequencing file.
    :returns: the label.
    """
    stem = os.path.basename(str(path))
    for marker in ("_R1", "_R2", "_r1", "_r2", ".R1", ".R2"):
        if marker in stem:
            return marker[1:].upper()
    for suffix in (".gz", ".fastq", ".fq", ".txt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or "reads"


def iter_barcode_search(fastq_files, tables, max_reads=DEFAULT_SAMPLE_READS,
                        chunk_reads=DEFAULT_CHUNK_READS, anchor=None):
    """Search for barcodes a chunk at a time, reporting after every chunk.

    The interface shows this running, so the search hands back a complete report
    after each chunk rather than only at the end.  Every report has the same
    shape as the final one and the counts inside it only ever grow, so the rates
    settle rather than jump and a verdict that was undecided becomes decided as
    the evidence arrives.  Stopping early is legitimate and leaves a report that
    says how many reads it rests on.

    :param fastq_files: a sequencing file, a sequence of them, or a mapping from
        the label each should carry to its path.
    :param tables: the reference tables to look for, of any number.
    :param max_reads: how many reads to take from each file at most.
    :param chunk_reads: how many reads each step takes from each file.
    :param anchor: an optional fixed sequence, such as the one the mapping run
        anchors its window on, searched alongside the tables so that offsets can
        be expressed relative to it.
    :yields: a report after each chunk, and one report when there is nothing to
        read.
    :raises ValueError: when no file was supplied or the budgets are not
        positive.
    """
    if max_reads <= 0:
        raise ValueError(f"max_reads must be positive; got {max_reads}")
    if chunk_reads <= 0:
        raise ValueError(f"chunk_reads must be positive; got {chunk_reads}")
    labelled = _as_labelled_files(fastq_files)
    searched = list(tables)
    if anchor:
        searched.append(
            BarcodeTable(
                name=ANCHOR_ROLE,
                sequences={str(anchor).upper(): ANCHOR_ROLE},
                role=ANCHOR_ROLE))
    state = _SearchState(searched, labelled)
    readers = {label: iter_fastq_reads(path) for label, path in labelled.items()}
    remaining = {label: max_reads for label in labelled}
    produced = False
    while True:
        progressed = False
        for label, reader in readers.items():
            wanted = min(chunk_reads, remaining[label])
            if wanted <= 0:
                continue
            taken = state.consume(label, islice(reader, wanted))
            remaining[label] -= taken
            if taken:
                progressed = True
            if taken < wanted:
                remaining[label] = 0
        if not progressed:
            break
        produced = True
        yield state.snapshot(complete=not any(remaining.values()))
    if not produced:
        yield state.snapshot(complete=True)


def search_barcodes(fastq_files, tables, max_reads=DEFAULT_SAMPLE_READS,
                    chunk_reads=DEFAULT_CHUNK_READS, anchor=None):
    """Search a bounded sample of reads and return the finished report.

    This is the whole of :func:`iter_barcode_search` run to its end, for callers
    that want an answer rather than a running display.

    :param fastq_files: a sequencing file, a sequence of them, or a mapping from
        the label each should carry to its path.
    :param tables: the reference tables to look for, of any number.
    :param max_reads: how many reads to take from each file at most.
    :param chunk_reads: how many reads each step takes from each file.
    :param anchor: an optional fixed sequence searched alongside the tables.
    :returns: the report from the last chunk.
    """
    report = None
    for report in iter_barcode_search(
            fastq_files, tables, max_reads=max_reads,
            chunk_reads=chunk_reads, anchor=anchor):
        pass
    return report


def _format_enrichment(enrichment):
    """Render how far above coincidence an observation sat.

    A table of long guide sequences has a coincidence rate so small that the
    ratio runs into the trillions, and printing every digit of that suggests a
    precision the measurement does not have.  Anything beyond a thousandfold is
    shown as such, because the difference between a thousandfold and a
    millionfold changes no decision.

    :param enrichment: the observed rate over the expected one.
    :returns: the rendered ratio, padded to a fixed width.
    """
    if math.isinf(enrichment) or enrichment >= 1000.0:
        return f"{'>1000':>10}"
    return f"{enrichment:10.1f}"


def _flip(orientation):
    """Return the other orientation label.

    :param orientation: one of the two orientation labels.
    :returns: the other one.
    """
    return REVERSE_COMPLEMENT if orientation == AS_GIVEN else AS_GIVEN


@dataclass(frozen=True)
class ProposedMapping:
    """Settings the search believes a mapping run should use, and why.

    The settings dictionary holds only keys the mapping run already reads, so it
    can be handed straight to it.  Everything the search learned that has no
    home among those keys travels alongside, because a caller that silently
    dropped the orientation would reintroduce the failure this module exists to
    prevent.  A screen that decodes a barcode beyond the plate column, the guide
    and the plate row has no settings key waiting for it, so the table chosen
    for every role is listed among the reference tables whether or not a key
    could be filled in for it.

    :param settings: only keys the mapping run already reads, so the mapping
        can be handed this dictionary directly.
    :param reference_file: the file whose reads the window offsets are
        expressed against, since offsets mean nothing without their frame.
    :param source_files: mapping from role to the read file that established
        it.
    :param orientations: mapping from table label to the orientation the run
        should search that table in.
    :param reference_tables: the table chosen for each role, listed whether or
        not a settings key exists to carry it -- which is the point, because a
        role with no key is exactly the one that would otherwise be lost.
    :param reverse_complement_needed: tables the run has to reverse complement
        to recognise, because they were found in the second mate as stored.
    :param unresolved_roles: roles no table established. Named rather than
        guessed, so a caller can say so instead of starting a run that will
        map nothing.
    :param notes: what the search learned that has no home among the settings
        keys.
    """

    settings: Dict[str, object]
    reference_file: str
    source_files: Mapping[str, str]
    orientations: Mapping[str, str]
    reference_tables: Mapping[str, str]
    reverse_complement_needed: Tuple[str, ...]
    unresolved_roles: Tuple[str, ...]
    notes: Tuple[str, ...]


def propose_map_barcodes_settings(report, base_settings=None, reference_file=None):
    """Turn a finished search into settings for a barcode mapping run.

    The mapping run reads both mates into one sequence in the orientation of the
    first, so that is the frame everything is expressed in.  A table found in
    the second mate as it is stored is therefore reported as needing to be
    reverse complemented, because that is what the run will need in order to
    recognise it.  A role that no table established is left alone rather than
    guessed at, and is named among the unresolved roles so the caller can say so
    instead of starting a run that will map nothing.

    The extraction window is derived from where the barcodes actually landed.
    Its start is the earliest offset any established barcode occupied and its
    end is the furthest any of them reached, both measured against the anchor
    when one was searched for, since the mapping run locates its window by
    finding that anchor first.

    :param report: the report from a search.
    :param base_settings: settings to start from, left unmodified; an empty
        dictionary when omitted.
    :param reference_file: the label of the file whose orientation is the frame;
        the first file of the search when omitted.
    :returns: the proposal, holding the settings and what did not fit in them.
    :raises ValueError: when the report holds no files.
    """
    files = list(report.reads_by_file)
    if not files:
        raise ValueError("the report holds no sequencing files")
    if reference_file is None:
        reference_file = files[0]
    settings = dict(base_settings or {})
    orientations: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    chosen_tables: Dict[str, str] = {}
    needs_flip = []
    unresolved = []
    notes = []
    spans = []

    anchor_finding = _best_in_frame(report, ANCHOR_ROLE, reference_file)
    anchor_offset = None
    if anchor_finding is not None and anchor_finding[2]:
        anchor_offset = anchor_finding[0].offset_start

    for role in report.roles():
        best = _best_in_frame(report, role, reference_file)
        if best is None:
            unresolved.append(role)
            notes.append(
                f"no reference table established the {role} barcodes in either "
                f"orientation of either file, so those settings were left as "
                f"they were")
            continue
        finding, frame_orientation, in_frame = best
        orientations[role] = frame_orientation
        sources[role] = finding.file_label
        if finding.table_path:
            chosen_tables[role] = finding.table_path
        if in_frame:
            spans.append((finding.offset_start, finding.window_end))
        key = _ROLE_SETTING_KEYS.get(role)
        if key and finding.table_path:
            settings[key] = finding.table_path
        if frame_orientation == REVERSE_COMPLEMENT:
            needs_flip.append(role)
            notes.append(
                f"the {role} table matches the reads only when reverse "
                f"complemented, so the mapping run needs a flipped copy of it")
        else:
            notes.append(
                f"the {role} table matches the reads as it is stored, so it "
                f"needs no change")
        if key and not finding.table_path:
            notes.append(
                f"the {role} table was searched from memory rather than a file, "
                f"so no path could be set for it")
        if not key and finding.table_path:
            notes.append(
                f"the mapping run has no settings key of its own for the {role} "
                f"barcodes, so the table chosen for them is reported rather "
                f"than written into the settings")

    carrying_files = {
        item.file_label for item in report.findings
        if item.verdict == PRESENT and item.role
        and item.role != ANCHOR_ROLE
    }

    if spans and anchor_offset is not None:
        window_start = min(start for start, _ in spans)
        window_end = max(end for _, end in spans)
        settings["offset_start"] = window_start - anchor_offset
        settings["window_length"] = window_end - window_start
        notes.append(
            f"the barcodes occupy the bases from {window_start} to "
            f"{window_end} of the read while the anchor begins at "
            f"{anchor_offset}, which fixes the window")
    elif spans:
        notes.append(
            "no anchor sequence was searched for, so the window could not be "
            "placed and the existing offset and length were kept")

    if len(carrying_files) > 1:
        settings["mode"] = "paired"
        notes.append(
            "both mates carry barcodes, so the run can build a consensus from "
            "the pair")
    elif len(carrying_files) == 1:
        only = next(iter(carrying_files))
        settings["mode"] = "single"
        if only in {"R1", "R2"}:
            settings["single_direction"] = only
        notes.append(
            f"only {only} carries barcodes, so the run should read that mate "
            f"alone")

    return ProposedMapping(
        settings=settings,
        reference_file=reference_file,
        source_files=sources,
        orientations=orientations,
        reference_tables=chosen_tables,
        reverse_complement_needed=tuple(needs_flip),
        unresolved_roles=tuple(unresolved),
        notes=tuple(notes))


def _best_in_frame(report, role, reference_file):
    """Return the best finding for a role, expressed in the reference frame.

    A barcode seen in the second mate exactly as the table stores it is the same
    observation as seeing it reverse complemented in the first, because the two
    mates read the same fragment from opposite ends.  Findings from the other
    file are therefore translated rather than ignored, which is what lets a run
    whose first mate is uninformative still be configured.

    :param report: the report to search.
    :param role: the barcode role wanted.
    Offsets do not translate between mates the way orientations do, because the
    two reads begin at opposite ends of the fragment and neither one knows how
    long the fragment was.  A finding brought across from the other file
    therefore says so, and the window is placed only from findings measured in
    the frame itself.

    :param reference_file: the label of the file that defines the frame.
    :returns: the finding, its orientation in the frame, and whether its offsets
        belong to the frame, or None when the role was not established.
    """
    candidates = [
        item for item in report.findings
        if item.role == role and item.verdict == PRESENT
    ]
    if not candidates:
        return None
    in_frame = [item for item in candidates if item.file_label == reference_file]
    if in_frame:
        best = max(in_frame, key=lambda item: item.observed_rate)
        return best, best.orientation, True
    best = max(candidates, key=lambda item: item.observed_rate)
    return best, _flip(best.orientation), False


@dataclass(frozen=True)
class BarcodeHit:
    """One barcode occurrence inside one read.

    The stretch it covers is given so that a display can highlight exactly those
    bases, and the colour index is a stable number for the barcode itself rather
    than for its table, so that two different barcodes of the same table are
    drawn in two different colours.

    :param start: index of the barcode's first base within the read.
    :param end: index one past its last base, so ``read[start:end]`` is the
        barcode.
    :param table: label of the table the barcode came from.
    :param role: the role that table fills, or None.
    :param barcode: the matched sequence itself.
    :param orientation: whether it matched ``AS_GIVEN`` or
        ``REVERSE_COMPLEMENT``.
    :param colour_index: a stable number for the barcode, not for its table,
        so two barcodes of one table are drawn in two colours.
    """

    start: int
    end: int
    table: str
    role: Optional[str]
    barcode: str
    orientation: str
    colour_index: int


def annotate_read(read, tables, orientations=None):
    """Locate every barcode inside one read so a display can colour them.

    The reads a search looked at are the evidence behind its verdicts, and the
    quickest way for someone to believe or disbelieve a verdict is to see the
    barcodes sitting in the reads.  This returns the stretches to paint, left to
    right, with overlaps resolved in favour of whichever table is listed first.

    :param read: the read sequence.
    :param tables: the reference tables to look for.
    :param orientations: a mapping from table label to the orientation to search
        in; both orientations are searched for a table not named there.
    :returns: a tuple of hits ordered by their position in the read.
    """
    read = str(read).upper()
    claimed = [False] * len(read)
    hits = []
    for table in tables:
        wanted = (orientations or {}).get(table.name)
        choices = (wanted,) if wanted else (AS_GIVEN, REVERSE_COMPLEMENT)
        order = table.barcode_order()
        for orientation in choices:
            for offset, length, name in table.matcher(orientation).find_all(read):
                if any(claimed[offset: offset + length]):
                    continue
                for position in range(offset, offset + length):
                    claimed[position] = True
                stored = read[offset: offset + length]
                if orientation == REVERSE_COMPLEMENT:
                    stored = reverse_complement(stored)
                hits.append(
                    BarcodeHit(
                        start=offset,
                        end=offset + length,
                        table=table.name,
                        role=table.role,
                        barcode=name,
                        orientation=orientation,
                        colour_index=order.get(stored, 0)))
    return tuple(sorted(hits, key=lambda hit: hit.start))


def iter_annotated_reads(fastq_file, tables, orientations=None,
                         limit=DEFAULT_CHUNK_READS, only_matching=False):
    """Yield reads together with the barcodes found inside them.

    This feeds the window that shows one read per line with its barcodes picked
    out, so it stays bounded in the same way the search is and never walks a
    whole file.

    :param fastq_file: the sequencing file to read.
    :param tables: the reference tables to look for.
    :param orientations: a mapping from table label to the orientation to search
        in; both orientations are searched for a table not named there.
    :param limit: how many reads to take from the file at most.
    :param only_matching: whether to skip reads in which nothing was found.
    :yields: each read and the hits inside it.
    """
    for read in iter_fastq_reads(fastq_file, limit=limit):
        hits = annotate_read(read, tables, orientations=orientations)
        if only_matching and not hits:
            continue
        yield read, hits
