"""FASTQ barcode decoding, consensus generation, and mapping pipeline."""

import os, gzip, re, time
import pandas as pd
from multiprocessing import Pool, cpu_count, Queue, Process
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import numpy as np
from . import schema
# One run id on every log line and every artifact, one seed, and the
# on_error policy at the per-sample boundary. See spacr.runctx.
from .runctx import run_context
from .plot import plot_plates
try:
    from IPython.display import display
except Exception:
    # IPython may be mid-init (partially imported by another
    # thread) — use a no-op fallback so importing this module
    # never blocks. spaCR only calls display() from notebook
    # contexts anyway; the Qt GUI ignores it.
    def display(*args, **kwargs):
        pass

# Function to map sequences to names (same as your original)
def map_sequences_to_names(csv_file, sequences, rc):
    """Look up barcode / gRNA names for a list of DNA reads against a ``sequence,name`` mapping CSV.

    Used inside the spacr sequencing pipeline to translate the row,
    column, and gRNA barcodes extracted from paired-end reads into their
    human-readable labels. Only the CSV's ``sequence`` column is
    reverse-complemented when ``rc=True``; the input ``sequences`` are
    matched verbatim, so callers should orient reads consistently
    beforehand.

    :param csv_file: Path to a CSV with ``sequence`` and ``name``
        columns.
    :param sequences: Iterable of DNA sequences to look up.
    :param rc: If True, reverse-complement the CSV sequences before
        building the lookup dict.
    :returns: List of names aligned positionally with ``sequences``;
        ``pd.NA`` for sequences that do not match any entry.

    Example:
        .. code-block:: python

            from spacr.sequencing import map_sequences_to_names
            names = map_sequences_to_names(
                '/data/barcodes/rows.csv',
                sequences=['ACGT...', 'TTGG...'],
                rc=False,
            )

    See Also:
        :func:`generate_barecode_mapping` — full end-to-end read ->
        (row, column, gRNA) name pipeline.
    """
    def rev_comp(dna_sequence):
        """Return the reverse complement of ``dna_sequence`` (N stays N)."""
        complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        reverse_seq = dna_sequence[::-1]
        return ''.join([complement_dict[base] for base in reverse_seq])
    
    df = pd.read_csv(csv_file)
    required = {"sequence", "name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Barcode mapping {csv_file!r} is missing required column(s): "
            f"{', '.join(sorted(missing))}.")
    duplicate_sequences = df["sequence"].dropna().duplicated(keep=False)
    if duplicate_sequences.any():
        examples = sorted(
            df.loc[duplicate_sequences, "sequence"].astype(str).unique())[:5]
        raise ValueError(
            f"Barcode mapping {csv_file!r} contains duplicate sequences; "
            f"each sequence must identify exactly one name. Examples: "
            f"{', '.join(examples)}.")
    if rc:
        df['sequence'] = df['sequence'].apply(rev_comp)
    
    csv_sequences = pd.Series(df['name'].values, index=df['sequence']).to_dict()
    return [csv_sequences.get(sequence, pd.NA) for sequence in sequences]

# Functions to save data (same as your original)
def save_df_to_hdf5(df, hdf5_file, key='df', comp_type='zlib', comp_level=5):
    """Append (or create) ``df`` to a ``table``-format HDF5 dataset.

    :param df: DataFrame to persist.
    :param hdf5_file: destination HDF5 file path.
    :param key: dataset key inside the store. Default ``'df'``.
    :param comp_type: compression library. Default ``'zlib'``.
    :param comp_level: compression level 0-9. Default ``5``.
    :returns: None.
    :raises Exception: after printing context, when the HDF5 write fails.
    """
    try:
        with pd.HDFStore(hdf5_file, 'a', complib=comp_type, complevel=comp_level) as store:
            if key in store:
                existing_df = store[key]
                df = pd.concat([existing_df, df], ignore_index=True)
            store.put(key, df, format='table')
    except Exception as e:
        print(f"Error while saving DataFrame to HDF5: {e}")
        raise

def save_unique_combinations_to_csv(unique_combinations, csv_file):
    """Append per-``(rowID, columnID, grna_name)`` counts to a CSV, summing duplicates.

    :param unique_combinations: DataFrame with ``rowID``, ``columnID``, ``grna_name`` and numeric count columns.
    :param csv_file: destination CSV path (created if absent).
    :returns: None.
    :raises Exception: after printing context, when the CSV write fails.
    """
    try:
        try:
            existing_df = pd.read_csv(csv_file)
        except FileNotFoundError:
            existing_df = pd.DataFrame()
        
        if not existing_df.empty:
            unique_combinations = pd.concat([existing_df, unique_combinations])
            unique_combinations = unique_combinations.groupby(
                ['rowID', 'columnID', 'grna_name'], as_index=False).sum()

        unique_combinations.to_csv(csv_file, index=True)
    except Exception as e:
        print(f"Error while saving unique combinations to CSV: {e}")
        raise

def save_qc_df_to_csv(qc_df, qc_csv_file):
    """Write a QC DataFrame to a CSV, index-aligning it with any existing file.

    An existing CSV is re-read and combined via ``DataFrame.add(..., fill_value=0)``,
    and the result overwrites the file with ``index=False``. Because the index is
    never written, the incoming ``NaN_Counts`` row does not align with the re-read
    ``RangeIndex``, so the CSV gains a row per call rather than accumulating totals.

    :param qc_df: numeric QC metrics (e.g. missing counts, total reads).
    :param qc_csv_file: destination CSV path.
    :returns: None.
    :raises Exception: after printing context, when the CSV write fails.
    """
    try:
        try:
            existing_qc_df = pd.read_csv(qc_csv_file)
        except FileNotFoundError:
            existing_qc_df = pd.DataFrame()

        if not existing_qc_df.empty:
            qc_df = qc_df.add(existing_qc_df, fill_value=0)

        qc_df.to_csv(qc_csv_file, index=False)
    except Exception as e:
        print(f"Error while saving QC DataFrame to CSV: {e}")
        raise

def extract_sequence_and_quality(sequence, quality, start, end):
    """Return the ``[start:end]`` slice of a sequence and its paired quality string.

    :param sequence: DNA sequence.
    :param quality: quality string of equal length.
    :param start: inclusive start index.
    :param end: exclusive end index.
    :returns: tuple ``(subsequence, subquality)``.
    """
    return sequence[start:end], quality[start:end]

def create_consensus(seq1, qual1, seq2, qual2):
    """Return a per-position consensus of two equal-length reads.

    At each position the higher-quality base is kept; if one call is ``N``
    the other is preferred regardless of quality.

    :param seq1: first DNA sequence.
    :param qual1: quality string for ``seq1``.
    :param seq2: second DNA sequence.
    :param qual2: quality string for ``seq2``.
    :returns: the consensus sequence as a string.
    """
    consensus_seq = []
    for i in range(len(seq1)):
        bases = [(seq1[i], qual1[i]), (seq2[i], qual2[i])]
        consensus_seq.append(get_consensus_base(bases))
    return ''.join(consensus_seq)

def get_consensus_base(bases):
    """Return the higher-quality base from two ``(base, quality)`` pairs, preferring non-``N``.

    :param bases: list of two ``(base, quality)`` tuples.
    :returns: the chosen base as a single-character string.
    """
    # Prefer non-'N' bases, if 'N' exists, pick the other one.
    if bases[0][0] == 'N':
        return bases[1][0]
    elif bases[1][0] == 'N':
        return bases[0][0]
    else:
        # Return the base with the highest quality score
        return bases[0][0] if bases[0][1] >= bases[1][1] else bases[1][0]

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence via BioPython.

    :param seq: DNA sequence.
    :returns: reverse-complemented sequence as a string.
    """
    return str(Seq(seq).reverse_complement())

# Core logic for processing a chunk (same as your original)
def process_chunk(chunk_data):
    """Extract and map barcodes from a chunk of single- or paired-end FASTQ reads.

    Anchors on ``target_sequence``, extracts a consensus window, splits it
    with the named-group ``regex``, and maps each barcode to its ID via
    the reference CSVs.

    The regex must supply a ``grna`` group plus a row and a column group.
    ``columnID``/``rowID``, the names used by the shipped default regex,
    take precedence; ``column``/``row`` are accepted as aliases.

    :param chunk_data: 9-tuple for single-end
        ``(r1_chunk, regex, target_sequence, offset_start, expected_end,
        column_csv, grna_csv, row_csv, fill_na)`` or 10-tuple for paired-end
        ``(r1_chunk, r2_chunk, ...)`` with the same trailing fields.
    :returns: tuple ``(df, unique_combinations, qc_df)`` — the annotated
        reads (``read``, per-barcode sequences and IDs), per-triplet counts,
        and a NaN/total-reads QC row.
    """
    if not isinstance(chunk_data, (tuple, list)) or len(chunk_data) not in (9, 10):
        raise ValueError(
            "process_chunk expects 9 values for single-end reads or 10 "
            f"values for paired-end reads; received "
            f"{len(chunk_data) if hasattr(chunk_data, '__len__') else 'an unknown count'}.")

    regex_obj = re.compile(chunk_data[2] if len(chunk_data) == 10 else chunk_data[1])
    group_names = set(regex_obj.groupindex)
    column_group = "columnID" if "columnID" in group_names else "column"
    row_group = "rowID" if "rowID" in group_names else "row"
    missing_groups = [
        canonical for canonical, alternatives in (
            ("column/columnID", {"column", "columnID"}),
            ("row/rowID", {"row", "rowID"}),
            ("grna", {"grna"}),
        )
        if not group_names.intersection(alternatives)
    ]
    if missing_groups:
        raise ValueError(
            "Barcode regex is missing required named group(s): "
            + ", ".join(missing_groups) + ".")

    def _parse_record(record, label):
        """Validate and split one four-line FASTQ record."""
        lines = str(record).splitlines()
        if len(lines) != 4:
            raise ValueError(
                f"{label} FASTQ record must have exactly four lines; "
                f"received {len(lines)}.")
        header, sequence, separator, quality = lines
        if not header.startswith("@") or not separator.startswith("+"):
            raise ValueError(
                f"{label} is not a valid FASTQ record (expected @ header "
                "and + separator).")
        if len(sequence) != len(quality):
            raise ValueError(
                f"{label} sequence and quality lengths differ "
                f"({len(sequence)} != {len(quality)}).")
        return sequence, quality

    def paired_find_sequence_in_chunk_reads(r1_chunk, r2_chunk, target_sequence, offset_start, expected_end, regex):
        """Return consensus reads and their parsed row/column/gRNA barcodes for paired-end chunks.

        :param r1_chunk: four-line FASTQ record strings for R1.
        :param r2_chunk: the matching R2 records, paired with ``r1_chunk`` by
            position only -- headers are never compared.
        :param target_sequence: anchor located with ``str.find`` (first
            occurrence). R2 is reverse-complemented before the search, so write
            the anchor in R1 orientation. A pair missing the anchor in either
            mate is dropped without a row, so the QC ``total_reads`` counts
            matched pairs rather than input pairs.
        :param offset_start: bases from the anchor to the window start. A
            resulting start below zero is clamped to the read start, not
            rejected, so an over-negative offset quietly shifts the window.
        :param expected_end: window *length*, not an end coordinate. A window
            cut short by the read end is right-padded with ``N`` (quality
            ``!``) to exactly this length, so the length check always passes
            and a truncated read can still match with ``N`` inside a barcode --
            which then maps to NA and drops out of the per-well counts.
        :param regex: pattern string applied with ``re.match``: anchored at the
            window start, but bases past the last group are ignored, so an
            oversized ``expected_end`` only adds padding.
        :returns: ``(consensus_sequences, columns, grnas, rows)``, one entry per
            matched pair. A chunk with no matches prints a warning and retries
            the last window reverse-complemented as an orientation hint.
        :raises ValueError: when the two chunks hold different read counts.
        """
        consensus_sequences, columns, grnas, rows = [], [], [], []
        consensus_seq = None
        if len(r1_chunk) != len(r2_chunk):
            raise ValueError(
                "Paired FASTQ chunks contain different read counts: "
                f"R1={len(r1_chunk)}, R2={len(r2_chunk)}.")
        
        for index, (r1_lines, r2_lines) in enumerate(zip(r1_chunk, r2_chunk)):
            r1_sequence, r1_quality = _parse_record(
                r1_lines, f"R1 record {index + 1}")
            r2_sequence, r2_quality = _parse_record(
                r2_lines, f"R2 record {index + 1}")
            r2_sequence = reverse_complement(r2_sequence)
            r2_quality = r2_quality[::-1]

            r1_pos = r1_sequence.find(target_sequence)
            r2_pos = r2_sequence.find(target_sequence)

            if r1_pos != -1 and r2_pos != -1:
                r1_start = max(r1_pos + offset_start, 0)
                r1_end = min(r1_start + expected_end, len(r1_sequence))
                r2_start = max(r2_pos + offset_start, 0)
                r2_end = min(r2_start + expected_end, len(r2_sequence))

                r1_seq, r1_qual = extract_sequence_and_quality(r1_sequence, r1_quality, r1_start, r1_end)
                r2_seq, r2_qual = extract_sequence_and_quality(r2_sequence, r2_quality, r2_start, r2_end)

                if len(r1_seq) < expected_end:
                    r1_seq += 'N' * (expected_end - len(r1_seq))
                    r1_qual += '!' * (expected_end - len(r1_qual))

                if len(r2_seq) < expected_end:
                    r2_seq += 'N' * (expected_end - len(r2_seq))
                    r2_qual += '!' * (expected_end - len(r2_qual))

                consensus_seq = create_consensus(r1_seq, r1_qual, r2_seq, r2_qual)
                if len(consensus_seq) >= expected_end:
                    match = re.match(regex, consensus_seq)
                    if match:
                        consensus_sequences.append(consensus_seq)
                        
                        #print(f"r1_seq: {r1_seq}")
                        #print(f"r2_seq: {r2_seq}")
                        #print(f"consensus_sequences: {consensus_sequences}")
                        
                        column_sequence = match.group(column_group)
                        grna_sequence = match.group('grna')
                        row_sequence = match.group(row_group)
                        columns.append(column_sequence)
                        grnas.append(grna_sequence)
                        rows.append(row_sequence)
                        
                        #print(f"row bc: {row_sequence} col bc: {column_sequence} grna bc: {grna_sequence}")
                        #print(f"row bc: {rows} col bc: {columns} grna bc: {grnas}")

        if len(consensus_sequences) == 0:
            print(f"WARNING: No sequences matched {regex} in chunk")
            print("Are barcode sequences in the correct orientation?")
            print(f"Is {consensus_seq} compatible with {regex} ?")
            
            if consensus_seq:
                if len(consensus_seq) >= expected_end:
                    consensus_seq_rc = reverse_complement(consensus_seq)
                    match = re.match(regex, consensus_seq_rc)
                    if match:
                        print(f"Reverse complement of last sequence in chunk matched {regex}")

        return consensus_sequences, columns, grnas, rows
    
    def single_find_sequence_in_chunk_reads(r1_chunk, target_sequence, offset_start, expected_end, regex):
        """Return R1 windows and their parsed row/column/gRNA barcodes for single-end chunks.

        No consensus is computed here: the R1 window is used as-is, so read
        quality never influences the base calls the way it does for pairs.

        :param r1_chunk: four-line FASTQ record strings for R1.
        :param target_sequence: anchor located with ``str.find`` (first
            occurrence). A read without it is dropped without a row, so the QC
            ``total_reads`` counts matched reads rather than input reads.
        :param offset_start: bases from the anchor to the window start. A
            resulting start below zero is clamped to the read start, not
            rejected.
        :param expected_end: window *length*, not an end coordinate. A window
            cut short by the read end is right-padded with ``N`` (quality
            ``!``) to exactly this length, so a truncated read can still match
            with ``N`` inside a barcode -- which then maps to NA and drops out
            of the per-well counts.
        :param regex: pattern string applied with ``re.match``: anchored at the
            window start, but bases past the last group are ignored.
        :returns: ``(consensus_sequences, columns, grnas, rows)``, one entry per
            matched read. A chunk with no matches prints a warning and retries
            the last window reverse-complemented as an orientation hint.
        """

        consensus_sequences, columns, grnas, rows = [], [], [], []
        consensus_seq = None

        for index, r1_lines in enumerate(r1_chunk):
            r1_sequence, r1_quality = _parse_record(
                r1_lines, f"R1 record {index + 1}")
            
            # Find the target sequence in R1
            r1_pos = r1_sequence.find(target_sequence)

            if r1_pos != -1:
                # Adjust start and end positions based on the offset and expected length
                r1_start = max(r1_pos + offset_start, 0)
                r1_end = min(r1_start + expected_end, len(r1_sequence))

                # Extract the sequence and quality within the defined region
                r1_seq, r1_qual = extract_sequence_and_quality(r1_sequence, r1_quality, r1_start, r1_end)

                # If the sequence is shorter than expected, pad with 'N's and '!' for quality
                if len(r1_seq) < expected_end:
                    r1_seq += 'N' * (expected_end - len(r1_seq))
                    r1_qual += '!' * (expected_end - len(r1_qual))

                # Use the R1 sequence as the "consensus"
                consensus_seq = r1_seq

                # Check if the consensus sequence matches the regex
                if len(consensus_seq) >= expected_end:
                    match = re.match(regex, consensus_seq)
                    if match:
                        consensus_sequences.append(consensus_seq)
                        column_sequence = match.group(column_group)
                        grna_sequence = match.group('grna')
                        row_sequence = match.group(row_group)
                        columns.append(column_sequence)
                        grnas.append(grna_sequence)
                        rows.append(row_sequence)

        if len(consensus_sequences) == 0:
            print(f"WARNING: No sequences matched {regex} in chunk")
            print("Are barcode sequences in the correct orientation?")
            print(f"Is {consensus_seq} compatible with {regex} ?")

            if consensus_seq and len(consensus_seq) >= expected_end:
                consensus_seq_rc = reverse_complement(consensus_seq)
                match = re.match(regex, consensus_seq_rc)
                if match:
                    print(f"Reverse complement of last sequence in chunk matched {regex}")

        return consensus_sequences, columns, grnas, rows

    if len(chunk_data) == 10:
        r1_chunk, r2_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, fill_na = chunk_data
    else:
        r1_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, fill_na = chunk_data
        r2_chunk = None

    if int(expected_end) <= 0:
        raise ValueError("expected_end must be a positive integer.")

    if r2_chunk is None:
        consensus_sequences, columns, grnas, rows = single_find_sequence_in_chunk_reads(r1_chunk, target_sequence, offset_start, expected_end, regex)
    else:
        consensus_sequences, columns, grnas, rows = paired_find_sequence_in_chunk_reads(r1_chunk, r2_chunk, target_sequence, offset_start, expected_end, regex)
    
    column_names = map_sequences_to_names(column_csv, columns, rc=False)
    grna_names = map_sequences_to_names(grna_csv, grnas, rc=False)
    row_names = map_sequences_to_names(row_csv, rows, rc=False)
    
    df = pd.DataFrame({
        'read': consensus_sequences,
        'column_sequence': columns,
        'columnID': column_names,
        'row_sequence': rows,
        'rowID': row_names,
        'grna_sequence': grnas,
        'grna_name': grna_names
    })

    qc_df = df.isna().sum().to_frame().T
    qc_df.columns = df.columns
    qc_df.index = ["NaN_Counts"]
    qc_df['total_reads'] = len(df)
    
    if fill_na:
        df2 = df.copy()
        if 'columnID' in df2.columns:
            df2['columnID'] = df2['columnID'].fillna(df2['column_sequence'])
        if 'rowID' in df2.columns:
            df2['rowID'] = df2['rowID'].fillna(df2['row_sequence'])
        if 'grna_name' in df2.columns:
            df2['grna_name'] = df2['grna_name'].fillna(df2['grna_sequence'])
        
        unique_combinations = df2.groupby(['rowID', 'columnID', 'grna_name']).size().reset_index(name='count')
    else:
        unique_combinations = df.groupby(['rowID', 'columnID', 'grna_name']).size().reset_index(name='count')

    return df, unique_combinations, qc_df

# Function to save data from the queue
def saver_process(save_queue, hdf5_file, save_h5, unique_combinations_csv, qc_csv_file, comp_type, comp_level):
    """Background writer that drains ``save_queue`` and persists each item.

    Runs until the sentinel ``"STOP"`` arrives on the queue.

    :param save_queue: multiprocessing queue delivering ``(df, unique_combinations, qc_df)`` tuples.
    :param hdf5_file: HDF5 destination for full annotated reads.
    :param save_h5: enable HDF5 writes of the reads DataFrame.
    :param unique_combinations_csv: destination CSV for aggregated barcode combinations.
    :param qc_csv_file: destination CSV for QC statistics.
    :param comp_type: HDF5 compression library.
    :param comp_level: HDF5 compression level.
    :returns: None.
    """
    while True:
        item = save_queue.get()
        if item == "STOP":
            break
        df, unique_combinations, qc_df = item
        if save_h5:
            save_df_to_hdf5(df, hdf5_file, key='df', comp_type=comp_type, comp_level=comp_level)
        save_unique_combinations_to_csv(unique_combinations, unique_combinations_csv)
        save_qc_df_to_csv(qc_df, qc_csv_file)


def _chunk_worker_count(n_jobs):
    """Return a valid process count while preserving three CPUs when possible."""
    if n_jobs is None:
        return max(1, cpu_count() - 3)
    count = int(n_jobs)
    if count < 1:
        raise ValueError(f"n_jobs must be at least 1; received {n_jobs!r}.")
    return count


def _validate_chunk_size(chunk_size):
    """Return ``chunk_size`` as a positive integer."""
    size = int(chunk_size)
    if size < 1:
        raise ValueError(
            f"chunk_size must be at least 1; received {chunk_size!r}.")
    return size


def _finish_saver(save_queue, save_process, timeout=60):
    """Stop the writer and fail the run when output persistence failed."""
    save_queue.put("STOP")
    save_process.join(timeout)
    if save_process.is_alive():
        save_process.terminate()
        save_process.join(5)
        raise RuntimeError(
            "Sequencing output writer did not stop within "
            f"{timeout} seconds and was terminated.")
    if save_process.exitcode not in (0, None):
        raise RuntimeError(
            "Sequencing output writer failed with exit code "
            f"{save_process.exitcode}; one or more output files may be "
            "incomplete. See the worker traceback above.")


def _abort_chunk_workers(pool, save_queue, save_process):
    """Best-effort cleanup after a read-processing exception."""
    pool.terminate()
    pool.join()
    if save_process.is_alive():
        save_queue.put("STOP")
        save_process.join(10)
    if save_process.is_alive():
        save_process.terminate()
        save_process.join(5)


def paired_read_chunked_processing(r1_file, r2_file, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, save_h5, comp_type, comp_level, hdf5_file, unique_combinations_csv, qc_csv_file, chunk_size=10000, n_jobs=None, test=False, fill_na=False):
    """Chunked paired-end FASTQ processing: extract, decode and stream barcodes to disk.

    Reads R1/R2 in ``chunk_size`` blocks, farms them out to
    :func:`process_chunk` workers, and lets :func:`saver_process` write
    HDF5 / CSV outputs concurrently.

    :param r1_file: gzipped R1 FASTQ path.
    :param r2_file: gzipped R2 FASTQ path.
    :param regex: regex with named groups ``rowID``, ``columnID``, ``grna``.
    :param target_sequence: anchor sequence used to locate the barcode region.
    :param offset_start: offset from ``target_sequence`` to begin extraction.
    :param expected_end: length of the extracted consensus region.
    :param column_csv: column-barcode reference CSV.
    :param grna_csv: gRNA-barcode reference CSV.
    :param row_csv: row-barcode reference CSV.
    :param save_h5: persist the full reads DataFrame to HDF5.
    :param comp_type: HDF5 compression library.
    :param comp_level: HDF5 compression level.
    :param hdf5_file: HDF5 output path.
    :param unique_combinations_csv: destination CSV for aggregated combinations.
    :param qc_csv_file: destination CSV for QC statistics.
    :param chunk_size: reads per batch. Default ``10000``.
    :param n_jobs: worker processes; defaults to ``cpu_count() - 3``.
    :param test: process only the first chunk and print a preview.
    :param fill_na: fill unmapped IDs with raw barcode sequences.
    :returns: None.
    """
    from .utils import count_reads_in_fastq, print_progress

    n_jobs = _chunk_worker_count(n_jobs)
    chunk_size = _validate_chunk_size(chunk_size)
    for label, path in (("R1", r1_file), ("R2", r2_file)):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"{label} FASTQ file does not exist: {path!r}.")

    chunk_count = 0
    time_ls = []

    if not test:
        print(f'Calculating read count for {r1_file}...')
        total_reads = count_reads_in_fastq(r1_file)
        chunks_nr = (total_reads + chunk_size - 1) // chunk_size
    else:
        total_reads = chunk_size
        chunks_nr = 1

    print(f'Mapping barcodes for {total_reads} reads in {chunks_nr} batches for {r1_file}...')

    # Queue for saving
    save_queue = Queue()

    # Start the saving process
    save_process = Process(target=saver_process, args=(save_queue, hdf5_file, save_h5, unique_combinations_csv, qc_csv_file, comp_type, comp_level))
    save_process.start()

    pool = Pool(n_jobs)

    print(f'Chunk size: {chunk_size}')

    with gzip.open(r1_file, 'rt') as r1, gzip.open(r2_file, 'rt') as r2:
        while True:
            start_time = time.time()
            r1_chunk = []
            r2_chunk = []

            for _ in range(chunk_size):
                # Read the next 4 lines for both R1 and R2 files
                r1_lines = [r1.readline().strip() for _ in range(4)]
                r2_lines = [r2.readline().strip() for _ in range(4)]

                # Paired files must end together; truncating to the shorter
                # input silently changes per-well counts.
                r1_done, r2_done = not r1_lines[0], not r2_lines[0]
                if r1_done != r2_done:
                    _abort_chunk_workers(pool, save_queue, save_process)
                    raise ValueError(
                        "Paired FASTQ files contain different read counts; "
                        "one file ended before the other.")
                if r1_done:
                    break

                r1_chunk.append('\n'.join(r1_lines))
                r2_chunk.append('\n'.join(r2_lines))
            
            # If the chunks are empty, break the outer while loop
            if not r1_chunk:
                break

            chunk_count += 1
            chunk_data = (r1_chunk, r2_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, fill_na)

            # Process chunks in parallel-
            result = pool.apply_async(process_chunk, (chunk_data,))

            try:
                df, unique_combinations, qc_df = result.get()
            except BaseException:
                _abort_chunk_workers(pool, save_queue, save_process)
                raise
            save_queue.put((df, unique_combinations, qc_df))

            end_time = time.time()
            chunk_time = end_time - start_time
            time_ls.append(chunk_time)
            print_progress(files_processed=chunk_count, files_to_process=chunks_nr, n_jobs=n_jobs, time_ls=time_ls, batch_size=chunk_size, operation_type="Mapping Barcodes")

            if test:
                print('First 1000 lines in chunk 1')
                print(df[:100])
                break

    # Cleanup the pool
    pool.close()
    pool.join()

    _finish_saver(save_queue, save_process)

def single_read_chunked_processing(r1_file, r2_file, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, save_h5, comp_type, comp_level, hdf5_file, unique_combinations_csv, qc_csv_file, chunk_size=10000, n_jobs=None, test=False, fill_na=False):
    """Chunked single-end FASTQ processing: extract, decode and stream barcodes to disk.

    :param r1_file: gzipped R1 FASTQ path.
    :param r2_file: unused placeholder kept for interface parity with the paired variant.
    :param regex: regex with named groups ``rowID``, ``columnID``, ``grna``.
    :param target_sequence: anchor sequence used to locate the barcode region.
    :param offset_start: offset from ``target_sequence`` to begin extraction.
    :param expected_end: length of the extracted barcode region.
    :param column_csv: column-barcode reference CSV.
    :param grna_csv: gRNA-barcode reference CSV.
    :param row_csv: row-barcode reference CSV.
    :param save_h5: persist the full reads DataFrame to HDF5.
    :param comp_type: HDF5 compression library.
    :param comp_level: HDF5 compression level.
    :param hdf5_file: HDF5 output path.
    :param unique_combinations_csv: destination CSV for aggregated combinations.
    :param qc_csv_file: destination CSV for QC statistics.
    :param chunk_size: reads per batch. Default ``10000``.
    :param n_jobs: worker processes; defaults to ``cpu_count() - 3``.
    :param test: process only the first chunk and print a preview.
    :param fill_na: fill unmapped IDs with raw barcode sequences.
    :returns: None.
    """
    from .utils import count_reads_in_fastq, print_progress

    n_jobs = _chunk_worker_count(n_jobs)
    chunk_size = _validate_chunk_size(chunk_size)
    if not r1_file or not os.path.isfile(r1_file):
        raise FileNotFoundError(
            f"R1 FASTQ file does not exist: {r1_file!r}.")

    chunk_count = 0
    time_ls = []

    if not test:
        print(f'Calculating read count for {r1_file}...')
        total_reads = count_reads_in_fastq(r1_file)
        chunks_nr = (total_reads + chunk_size - 1) // chunk_size
    else:
        total_reads = chunk_size
        chunks_nr = 1

    print(f'Mapping barcodes for {total_reads} reads in {chunks_nr} batches for {r1_file}...')

    # Queue for saving
    save_queue = Queue()

    # Start the saving process
    save_process = Process(target=saver_process, args=(save_queue, hdf5_file, save_h5, unique_combinations_csv, qc_csv_file, comp_type, comp_level))
    save_process.start()

    pool = Pool(n_jobs)

    with gzip.open(r1_file, 'rt') as r1:
        while True:
            start_time = time.time()
            r1_chunk = []

            for _ in range(chunk_size):
                # Read the next 4 lines for both R1 and R2 files
                r1_lines = [r1.readline().strip() for _ in range(4)]

                # Break if we've reached the end of either file
                if not r1_lines[0]:
                    break

                r1_chunk.append('\n'.join(r1_lines))

            # If the chunks are empty, break the outer while loop
            if not r1_chunk:
                break

            chunk_count += 1
            chunk_data = (r1_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, fill_na)

            # Process chunks in parallel
            result = pool.apply_async(process_chunk, (chunk_data,))
            
            try:
                df, unique_combinations, qc_df = result.get()
            except BaseException:
                _abort_chunk_workers(pool, save_queue, save_process)
                raise

            # Queue the results for saving
            save_queue.put((df, unique_combinations, qc_df))

            end_time = time.time()
            chunk_time = end_time - start_time
            time_ls.append(chunk_time)
            print_progress(files_processed=chunk_count, files_to_process=chunks_nr, n_jobs=n_jobs, time_ls=time_ls, batch_size=chunk_size, operation_type="Mapping Barcodes")

            if test:
                print('First 1000 lines in chunk 1')
                print(df[:100])
                break

    # Cleanup the pool
    pool.close()
    pool.join()

    _finish_saver(save_queue, save_process)


def _run_barcode_qc(settings, dst, count_csv, qc_csv):
    """QC one finished sample, if the settings asked for it. Never raises.

    Called at the end of each sample's turn in
    :func:`generate_barecode_mapping`, once that sample's
    ``unique_combinations.csv`` and ``qc.csv`` are on disk. The point of
    running it here rather than leaving it to the user is that the two
    questions a mapping run raises -- did it work, and where does the
    abundance threshold go -- are asked about *this* run's numbers, and
    nobody goes back for them once the counts exist.

    Wrapped, deliberately and completely. The reads are mapped and the
    table is written by the time this is reached: a QC panel that cannot
    plot, a missing barcode reference or an unreadable ``qc.csv`` must
    cost the report and nothing else. A run that already produced its
    output must never be lost to the analysis OF that output.

    :param settings: the mapping settings dict. Reads
        ``barcode_qc`` (default False -- the QC is opt-in, because it
        pulls in plotting and statistics the read workers do not want)
        and ``target_grnas_per_well``, which is the number the threshold
        is derived from.
    :param dst: the sample's output folder; the QC lands in
        ``<dst>/barcode_qc``.
    :param count_csv: that sample's ``unique_combinations.csv``.
    :param qc_csv: that sample's ``qc.csv``.
    :returns: the :func:`spacr.sequencing_qc.barcode_qc` result dict, or
        None when the QC was off or failed.
    """
    if not settings.get('barcode_qc', False):
        return None
    try:
        from .sequencing_qc import barcode_qc
        result = barcode_qc({
            'count_data': count_csv,
            'qc_data': qc_csv,
            'row_csv': settings.get('row_csv'),
            'column_csv': settings.get('column_csv'),
            'grna_csv': settings.get('grna_csv'),
            'target_grnas_per_well': settings.get('target_grnas_per_well', 1),
            'dst': os.path.join(dst, 'barcode_qc'),
        })
        print(result.get('recommendation', ''))
        return result
    except Exception as exc:
        print(f"WARNING: barcode QC failed for {dst}: {exc}. The counts "
              f"themselves were written and are unaffected; run "
              f"spacr.sequencing_qc.barcode_qc on them directly to see why.")
        return None


def generate_barecode_mapping(settings=None):
    """Turn a folder of pooled-screen FASTQ files into per-well sgRNA count tables usable by :func:`spacr.ml.perform_regression`.

    Discovers R1/R2 files per sample under ``src``, extracts the row,
    column, and gRNA barcodes from each read via the configured regex
    and offset window, translates them to names via three barcode
    lookup CSVs (see :func:`map_sequences_to_names`), and writes
    per-sample ``annotated_reads.h5`` (optional),
    ``unique_combinations.csv`` (the per-well gRNA counts) and
    ``qc.csv``. Paired vs single-end and R1/R2 orientation are chosen
    from ``settings['mode']`` and ``single_direction``.

    :param settings: Settings dict, canonicalized via
        :func:`spacr.settings.set_default_generate_barecode_mapping`.
        Key entries:

        - ``src`` — folder containing ``*.fastq.gz`` reads.
        - ``mode`` — ``'paired'`` or ``'single'``.
        - ``single_direction`` — ``'R1'`` or ``'R2'`` (``mode='single'``
          only).
        - ``regex`` — regex extracting barcodes from a read.
        - ``target_sequence``, ``offset_start``, ``expected_end`` —
          anchor and slice window used to locate the barcode region.
        - ``column_csv`` / ``row_csv`` / ``grna_csv`` — barcode->name
          lookup CSVs.
        - ``save_h5``, ``comp_type``, ``comp_level`` — HDF5 output
          knobs.
        - ``chunk_size``, ``n_jobs``, ``test``, ``fill_na``.
        - ``barcode_qc`` — when true, QC each finished sample with
          :func:`spacr.sequencing_qc.barcode_qc`, writing plots and a
          report into ``<dst>/barcode_qc`` (default False; not filled in
          by the settings defaults).
        - ``target_grnas_per_well`` — expected gRNAs per well, from which
          that QC step derives its abundance threshold (default 1).

    :returns: None. Writes per-sample outputs into
        ``<src>/<sample>_<mode>[_<direction>]/``.

    Example:
        .. code-block:: python

            from spacr.sequencing import generate_barecode_mapping
            generate_barecode_mapping({
                'src': '/data/screen_v1/fastq',
                'mode': 'paired',
                'row_csv': '/data/barcodes/rows.csv',
                'column_csv': '/data/barcodes/cols.csv',
                'grna_csv': '/data/barcodes/grnas.csv',
            })

    See Also:
        :func:`map_sequences_to_names` — inner barcode->name lookup.
        :func:`spacr.ml.perform_regression` — consumes the resulting
        ``unique_combinations.csv`` as ``count_data``.
    """
    if settings is None:
        settings = {}
    from .settings import set_default_generate_barecode_mapping
    from .utils import save_settings
    from .io import parse_gz_files

    settings = set_default_generate_barecode_mapping(settings)
    save_settings(settings, name=f"sequencing_{settings['mode']}_{settings['single_direction']}", show=True)

    regex = settings['regex']

    print(f'Using regex: {regex} to extract barcode information')

    samples_dict = parse_gz_files(settings['src'])
    
    print(samples_dict)

    print(f'If compression is low and save_h5 is True, saving might take longer than processing.')
    
    # One run over every sample: one id on the log lines and the artifacts,
    # and the on_error policy at the per-sample boundary. See spacr.runctx.
    with run_context('sequencing', settings) as run:
        for key in samples_dict:
            # on_error, at the per-sample boundary. Until now a single
            # unreadable FASTQ pair took every later sample down with it,
            # and the run still exited 0 -- the folder simply had fewer
            # outputs in it than it had samples.
            for attempt in run.policy.attempts_for(key, stage='sample'):
                with attempt:
                    if settings['mode'] == 'paired' and samples_dict[key]['R1'] and samples_dict[key]['R2'] or settings['mode'] == 'single' and samples_dict[key]['R1'] or settings['mode'] == 'single' and samples_dict[key]['R2']:            
                        key_mode = f"{key}_{settings['mode']}"
                        if settings['mode'] == 'single':
                            key_mode = f"{key_mode}_{settings['single_direction']}"
                        dst = os.path.join(settings['src'], key_mode)
                        hdf5_file = os.path.join(dst, 'annotated_reads.h5')
                        unique_combinations_csv = os.path.join(dst, 'unique_combinations.csv')
                        qc_csv_file = os.path.join(dst, 'qc.csv')
                        os.makedirs(dst, exist_ok=True)

                        print(f'Analyzing reads from sample {key}')

                        if settings['mode'] == 'paired':
                            function = paired_read_chunked_processing
                            R1=samples_dict[key]['R1']
                            R2=samples_dict[key]['R2']

                        elif settings['mode'] == 'single':
                            function = single_read_chunked_processing

                            if settings['single_direction'] == 'R1':
                                R1=samples_dict[key]['R1']
                                R2=None
                            elif settings['single_direction'] == 'R2':
                                R1=samples_dict[key]['R2']
                                R2=None

                        function(r1_file=R1,
                                 r2_file=R2,
                                 regex=regex,
                                 target_sequence=settings['target_sequence'],
                                 offset_start=settings['offset_start'],
                                 expected_end=settings['expected_end'],
                                 column_csv=settings['column_csv'],
                                 grna_csv=settings['grna_csv'],
                                 row_csv=settings['row_csv'],
                                 save_h5 = settings['save_h5'],
                                 comp_type = settings['comp_type'],
                                 comp_level=settings['comp_level'],
                                 hdf5_file=hdf5_file,
                                 unique_combinations_csv=unique_combinations_csv,
                                 qc_csv_file=qc_csv_file,
                                 chunk_size=settings['chunk_size'],
                                 n_jobs=settings['n_jobs'],
                                 test=settings['test'],
                                 fill_na=settings['fill_na'])

                        # The table exists now; QC it while we know which
                        # sample it belongs to. Inside the attempt so a
                        # per-sample on_error policy still applies, but
                        # itself never raising -- see _run_barcode_qc.
                        _run_barcode_qc(settings, dst,
                                        unique_combinations_csv, qc_csv_file)

# Function to read the CSV, compute reverse complement, and save it
def barecodes_reverse_complement(csv_file):
    """Write a copy of a barcode CSV with the ``sequence`` column reverse-complemented.

    Output is saved in the same directory with the extension dropped and
    ``_RC.csv`` appended, so ``rows.csv`` becomes ``rows_RC.csv``.

    :param csv_file: input CSV path with a ``sequence`` column.
    :returns: None.
    """
    def reverse_complement(sequence):
        """Return the reverse complement of ``sequence`` (N stays N)."""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(complement[base] for base in reversed(sequence))

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Compute reverse complement for each sequence
    df['sequence'] = df['sequence'].apply(reverse_complement)

    # Create the new filename
    file_dir, file_name = os.path.split(csv_file)
    file_name_no_ext = os.path.splitext(file_name)[0]
    new_filename = os.path.join(file_dir, f"{file_name_no_ext}_RC.csv")

    # Save the DataFrame with the reverse complement sequences
    df.to_csv(new_filename, index=False)

    print(f"Reverse complement file saved as {new_filename}")

def graph_sequencing_stats(settings):
    """Pick the fraction cutoff that yields a target mean of unique gRNAs per well.

    Loads one or more count CSVs, drops control wells, computes per-well
    gRNA fractions, sweeps thresholds to find the value producing the
    requested unique-count average, and plots both the sweep curve and
    the resulting per-plate heatmap.

    :param settings: dict with keys ``count_data`` (str or list of CSVs
        with ``grna``, ``count``, ``rowID``, ``columnID``),
        ``target_unique_count``, ``filter_column``, ``control_wells``,
        ``log_x`` and ``log_y``.
    :returns: the fraction threshold closest to the target unique count.
    """
    from .utils import correct_metadata_column_names, correct_metadata

    def find_and_visualize_fraction_threshold(df, target_unique_count=5, log_x=False, log_y=False, dst=None):
        """Return the fraction threshold whose per-well unique gRNA mean is closest to ``target_unique_count``.

        The sweep plot's view is clamped to x in [0, 0.1] and y in [0, 20], so a
        returned threshold above 0.1 has its marker line drawn off the visible
        axes even though the value itself is correct.

        :param df: one row per gRNA per well, with ``fraction`` plus
            ``plateID``, ``rowID`` and ``columnID``; any missing column raises
            ``KeyError``.
        :param target_unique_count: mean unique gRNAs per well to aim for. The
            answer is only the nearest point of a fixed 1000-step grid spanning
            0.001 to 0.99, so an unreachable target silently saturates at one
            end of that grid instead of reporting that it was not met. ``None``
            or a string raises from the subtraction.
        :param log_x: log-scale the plot's x axis. The hard-coded
            ``xlim(0, 0.1)`` applied afterwards is then rejected by matplotlib
            as a non-positive limit on a log axis, so the view autoscales.
        :param log_y: the same for the y axis and ``ylim(0, 20)``.
        :param dst: when set, writes ``<dst>/results/fraction_threshold.pdf``,
            creating the directories; ``None`` only shows the figure. The sole
            caller always passes a path, so the default is unreachable there.
        :returns: the chosen threshold as a ``numpy.float64``. A threshold that
            empties the table yields NaN rather than 0, so a table whose
            fractions are all below 0.001 makes every sweep point NaN and the
            fallback returns 0.99 -- a value that then discards every read.
        """

        def _line_plot(df, x, y, log_x, log_y):
            # No "are x and y in df.columns?" guard: this is a closure with one
            # call site eight lines below, and `df` there is the results_df
            # built two lines above it with exactly these two columns. The
            # check could not fire, so it was a branch no test could ever
            # reach honestly -- removed rather than excused.
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(df[x], df[y], linestyle='-', color=(0 / 255, 155 / 255, 155 / 255), label=f"{y}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f'{y} vs {x}')
            ax.legend()
            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')
            fig.tight_layout()
            return fig, ax

        fraction_thresholds = np.linspace(0.001, 0.99, 1000)
        results = []

        # Iterate through the fraction thresholds
        for threshold in fraction_thresholds:
            filtered_df = df[df['fraction'] >= threshold]
            unique_count = filtered_df.groupby(['plateID', 'rowID', 'columnID'])['grna'].nunique().mean()
            results.append((threshold, unique_count))

        results_df = pd.DataFrame(results, columns=['fraction_threshold', 'unique_count'])
        closest_index = (results_df['unique_count'] - target_unique_count).abs().argmin()
        closest_threshold = results_df.iloc[closest_index]

        print(f"Closest Fraction Threshold: {closest_threshold['fraction_threshold']}")
        print(f"Unique Count at Threshold: {closest_threshold['unique_count']}")

        fig, ax = _line_plot(df=results_df, x='fraction_threshold', y='unique_count', log_x=log_x, log_y=log_y)

        plt.axvline(x=closest_threshold['fraction_threshold'], color='black', linestyle='--',
                    label=f'Closest Threshold ({closest_threshold["fraction_threshold"]:.4f})')
        plt.axhline(y=target_unique_count, color='black', linestyle='--',
                    label=f'Target Unique Count ({target_unique_count})')
        
        plt.xlim(0,0.1)
        plt.ylim(0,20)

        if dst is not None:
            fig_path = os.path.join(dst, 'results')
            os.makedirs(fig_path, exist_ok=True)
            fig_file_path = os.path.join(fig_path, 'fraction_threshold.pdf')
            fig.savefig(fig_file_path, format='pdf', dpi=600, bbox_inches='tight')
            print(f"Saved {fig_file_path}")
        plt.show()

        return closest_threshold['fraction_threshold']

    if isinstance(settings['count_data'], str):
        settings['count_data'] = [settings['count_data']]

    dfs = []
    for i, count_data in enumerate(settings['count_data']):
        df = pd.read_csv(count_data)
        
        df = correct_metadata(df)
        
        if 'plateID' not in df.columns:
            df['plateID'] = f'plate{i+1}'
            
        display(df)
        
        if all(col in df.columns for col in ['plateID', 'rowID', 'columnID']):
            df['prc'] = df['plateID'].astype(str) + '_' + df['rowID'].astype(str) + '_' + df['columnID'].astype(str)
        else:
            raise ValueError("The DataFrame must contain 'plateID', 'rowID', and 'columnID' columns.")
        
        df['total_count'] = df.groupby(['prc'])['count'].transform('sum')
        df['fraction'] = df['count'] / df['total_count']
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    df = correct_metadata_column_names(df)

    for c in settings['control_wells']:
        df = df[df[settings['filter_column']] != c]

    dst = os.path.dirname(settings['count_data'][0])

    # `.get`, because instruction 135 retired log_x/log_y as settings --
    # the axes are chosen automatically and changed on the plot now. This
    # runs on the DEFAULT regression path, whenever fraction_threshold is
    # None, so a subscript here killed every run 25 lines after the one
    # that killed it first.
    closest_threshold = find_and_visualize_fraction_threshold(
        df, settings['target_unique_count'],
        log_x=settings.get('log_x', False),
        log_y=settings.get('log_y', False), dst=dst)

    # Apply the closest threshold to the DataFrame
    df = df[df['fraction'] >= closest_threshold]

    # Group by 'plateID', 'rowID', 'columnID' and compute unique counts of 'grna'
    unique_counts = df.groupby(['plateID', 'rowID', 'columnID'])['grna'].nunique().reset_index(name='unique_counts')
    unique_count_mean = df.groupby(['plateID', 'rowID', 'columnID'])['grna'].nunique().mean()
    unique_count_std = df.groupby(['plateID', 'rowID', 'columnID'])['grna'].nunique().std()

    # Merge the unique counts back into the original DataFrame.
    # unique_counts is one row per well by construction (groupby on exactly
    # this key), df is one row per (well, gRNA): many-to-one. If the right side
    # ever gained a duplicate the plate heatmap below would average a well's
    # gRNA rows more than once and simply show the wrong number, with nothing
    # in the output saying so.
    df = pd.merge(df, unique_counts, on=['plateID', 'rowID', 'columnID'],
                  how='left', validate='many_to_one')

    print(f"unique_count mean: {unique_count_mean} std: {unique_count_std}")

    # rowID sometimes arrives as the composite '<plate>_<row>' that count CSVs
    # carry in their 'plate_row' column; plot_plates wants the row alone.
    #
    # This was guarded by `df['rowID'].str.contains('_').any()` and then run
    # over EVERY row with `x.split('_')[1]`, so one composite value anywhere in
    # the table made the whole column go through an index that the plain values
    # do not have: ['plate1_r1', 'r2', 'r3'] raises IndexError and the caller
    # loses the threshold it had already computed. The [1] was also the wrong
    # token for a plate whose own name contains a separator ('exp1_plate1_r2'
    # gave 'plate1'). Taking the token after the LAST separator is right for
    # both, needs no guard, and leaves a plain 'r2' untouched.
    df['rowID'] = (df['rowID'].astype(str)
                   .str.rsplit(schema.KEY_SEPARATOR, n=1).str[-1])

    plot_plates(df=df, variable='unique_counts', grouping='mean', min_max='allq', cmap='viridis',min_count=0, verbose=True, dst=dst)
    
    return closest_threshold
