import os, gzip, re, time, gzip
import pandas as pd
from multiprocessing import Pool, cpu_count, Queue, Process
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .plot import plot_plates
from IPython.display import display

# Function to map sequences to names (same as your original)
def map_sequences_to_names(csv_file, sequences, rc):
    """Map DNA sequences to their names via a lookup CSV of ``sequence,name``.

    Only the CSV sequences are reverse-complemented when ``rc`` is true;
    the input ``sequences`` are matched as given.

    :param csv_file: path to a CSV with ``sequence`` and ``name`` columns.
    :param sequences: DNA sequences to look up.
    :param rc: reverse-complement the CSV sequences before matching.
    :returns: list of names aligned with ``sequences``; ``pd.NA`` for misses.
    """
    def rev_comp(dna_sequence):
        """Return the reverse complement of ``dna_sequence`` (N stays N)."""
        complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        reverse_seq = dna_sequence[::-1]
        return ''.join([complement_dict[base] for base in reverse_seq])
    
    df = pd.read_csv(csv_file)
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
    :returns: None. Errors are printed instead of raised.
    """
    try:
        with pd.HDFStore(hdf5_file, 'a', complib=comp_type, complevel=comp_level) as store:
            if key in store:
                existing_df = store[key]
                df = pd.concat([existing_df, df], ignore_index=True)
            store.put(key, df, format='table')
    except Exception as e:
        print(f"Error while saving DataFrame to HDF5: {e}")

def save_unique_combinations_to_csv(unique_combinations, csv_file):
    """Append per-``(rowID, columnID, grna_name)`` counts to a CSV, summing duplicates.

    :param unique_combinations: DataFrame with ``rowID``, ``columnID``, ``grna_name`` and numeric count columns.
    :param csv_file: destination CSV path (created if absent).
    :returns: None. Errors are printed instead of raised.
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

def save_qc_df_to_csv(qc_df, qc_csv_file):
    """Append a QC DataFrame to a CSV, summing element-wise when the file already exists.

    :param qc_df: numeric QC metrics (e.g. missing counts, total reads).
    :param qc_csv_file: destination CSV path.
    :returns: None. Errors are printed instead of raised.
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

    :param chunk_data: 9-tuple for single-end
        ``(r1_chunk, regex, target_sequence, offset_start, expected_end,
        column_csv, grna_csv, row_csv, fill_na)`` or 10-tuple for paired-end
        ``(r1_chunk, r2_chunk, ...)`` with the same trailing fields.
    :returns: tuple ``(df, unique_combinations, qc_df)`` — the annotated
        reads (``read``, per-barcode sequences and IDs), per-triplet counts,
        and a NaN/total-reads QC row.
    """
    def paired_find_sequence_in_chunk_reads(r1_chunk, r2_chunk, target_sequence, offset_start, expected_end, regex):
        """Return consensus reads and their parsed row/column/gRNA barcodes for paired-end chunks."""
        consensus_sequences, columns, grnas, rows = [], [], [], []
        consensus_seq = None
        
        for r1_lines, r2_lines in zip(r1_chunk, r2_chunk):
            _, r1_sequence, _, r1_quality = r1_lines.split('\n')
            _, r2_sequence, _, r2_quality = r2_lines.split('\n')
            r2_sequence = reverse_complement(r2_sequence)

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
                        
                        column_sequence = match.group('columnID')
                        grna_sequence = match.group('grna')
                        row_sequence = match.group('rowID')
                        columns.append(column_sequence)
                        grnas.append(grna_sequence)
                        rows.append(row_sequence)
                        
                        #print(f"row bc: {row_sequence} col bc: {column_sequence} grna bc: {grna_sequence}")
                        #print(f"row bc: {rows} col bc: {columns} grna bc: {grnas}")

        if len(consensus_sequences) == 0:
            print(f"WARNING: No sequences matched {regex} in chunk")
            print(f"Are bacode sequences in the correct orientation?")
            print(f"Is {consensus_seq} compatible with {regex} ?")
            
            if consensus_seq:
                if len(consensus_seq) >= expected_end:
                    consensus_seq_rc = reverse_complement(consensus_seq)
                    match = re.match(regex, consensus_seq_rc)
                    if match:
                        print(f"Reverse complement of last sequence in chunk matched {regex}")

        return consensus_sequences, columns, grnas, rows
    
    def single_find_sequence_in_chunk_reads(r1_chunk, target_sequence, offset_start, expected_end, regex):
        """Return R1 windows and their parsed row/column/gRNA barcodes for single-end chunks."""

        consensus_sequences, columns, grnas, rows = [], [], [], []

        for r1_lines in r1_chunk:
            _, r1_sequence, _, r1_quality = r1_lines.split('\n')
            
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
                        column_sequence = match.group('columnID')
                        grna_sequence = match.group('grna')
                        row_sequence = match.group('rowID')
                        columns.append(column_sequence)
                        grnas.append(grna_sequence)
                        rows.append(row_sequence)

        if len(consensus_sequences) == 0:
            print(f"WARNING: No sequences matched {regex} in chunk")
            print(f"Are bacode sequences in the correct orientation?")
            print(f"Is {consensus_seq} compatible with {regex} ?")

            if len(consensus_seq) >= expected_end:
                consensus_seq_rc = reverse_complement(consensus_seq)
                match = re.match(regex, consensus_seq_rc)
                if match:
                    print(f"Reverse complement of last sequence in chunk matched {regex}")

        return consensus_sequences, columns, grnas, rows

    if len(chunk_data) == 10:
        r1_chunk, r2_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, fill_na = chunk_data
    if len(chunk_data) == 9:
        r1_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, fill_na = chunk_data
        r2_chunk = None

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

    # Use cpu_count minus 3 cores if n_jobs isn't specified
    if n_jobs is None:
        n_jobs = cpu_count() - 3

    chunk_count = 0
    time_ls = []

    if not test:
        print(f'Calculating read count for {r1_file}...')
        total_reads = count_reads_in_fastq(r1_file)
        chunks_nr = int(total_reads / chunk_size)+1
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
        fastq_iter = zip(r1, r2)
        while True:
            start_time = time.time()
            r1_chunk = []
            r2_chunk = []

            for _ in range(chunk_size):
                # Read the next 4 lines for both R1 and R2 files
                r1_lines = [r1.readline().strip() for _ in range(4)]
                r2_lines = [r2.readline().strip() for _ in range(4)]

                # Break if we've reached the end of either file
                if not r1_lines[0] or not r2_lines[0]:
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

            df, unique_combinations, qc_df = result.get()
            save_queue.put((df, unique_combinations, qc_df))

            end_time = time.time()
            chunk_time = end_time - start_time
            time_ls.append(chunk_time)
            print_progress(files_processed=chunk_count, files_to_process=chunks_nr, n_jobs=n_jobs, time_ls=time_ls, batch_size=chunk_size, operation_type="Mapping Barcodes")

            if test:
                print(f'First 1000 lines in chunk 1')
                print(df[:100])
                break

    # Cleanup the pool
    pool.close()
    pool.join()

    # Send stop signal to saver process
    save_queue.put("STOP")
    save_process.join()

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

    # Use cpu_count minus 3 cores if n_jobs isn't specified
    if n_jobs is None:
        n_jobs = cpu_count() - 3

    chunk_count = 0
    time_ls = []

    if not test:
        print(f'Calculating read count for {r1_file}...')
        total_reads = count_reads_in_fastq(r1_file)
        chunks_nr = int(total_reads / chunk_size) + 1
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
            
            df, unique_combinations, qc_df = result.get()

            # Queue the results for saving
            save_queue.put((df, unique_combinations, qc_df))

            end_time = time.time()
            chunk_time = end_time - start_time
            time_ls.append(chunk_time)
            print_progress(files_processed=chunk_count, files_to_process=chunks_nr, n_jobs=n_jobs, time_ls=time_ls, batch_size=chunk_size, operation_type="Mapping Barcodes")

            if test:
                print(f'First 1000 lines in chunk 1')
                print(df[:100])
                break

    # Cleanup the pool
    pool.close()
    pool.join()

    # Send stop signal to saver process
    save_queue.put("STOP")
    save_process.join()

def generate_barecode_mapping(settings=None):
    """Discover FASTQ samples and dispatch chunked barcode extraction for each.

    Groups R1/R2 files per sample, then calls
    :func:`paired_read_chunked_processing` or
    :func:`single_read_chunked_processing` depending on ``settings['mode']``,
    writing ``annotated_reads.h5`` (optional), ``unique_combinations.csv``
    and ``qc.csv`` under each sample's output directory.

    :param settings: dict of barcode-mapping settings; see
        ``set_default_generate_barecode_mapping`` for keys (``src``,
        ``mode``, ``single_direction``, ``regex``, ``target_sequence``,
        ``offset_start``, ``expected_end``, ``column_csv``, ``grna_csv``,
        ``row_csv``, ``save_h5``, ``comp_type``, ``comp_level``,
        ``chunk_size``, ``n_jobs``, ``test``, ``fill_na``).
    :returns: None. Writes per-sample HDF5 and CSV outputs to disk.
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
    
    for key in samples_dict:
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

# Function to read the CSV, compute reverse complement, and save it
def barecodes_reverse_complement(csv_file):
    """Write a copy of a barcode CSV with the ``sequence`` column reverse-complemented.

    Output is saved as ``<csv_file>_RC.csv`` in the same directory.

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

    def _plot_density(df, dependent_variable, dst=None):
        """Plot a KDE of ``dependent_variable`` in ``df``, optionally saving under ``dst``."""
        plt.figure(figsize=(10, 10))
        sns.kdeplot(df[dependent_variable], fill=True, alpha=0.6)
        plt.title(f'Density Plot of {dependent_variable}')
        plt.xlabel(dependent_variable)
        plt.ylabel('Density')
        if dst is not None:
            filename = os.path.join(dst, 'dependent_variable_density.pdf')
            plt.savefig(filename, format='pdf')
            print(f'Saved density plot to {filename}')
        plt.show()

    def find_and_visualize_fraction_threshold(df, target_unique_count=5, log_x=False, log_y=False, dst=None):
        """Return the fraction threshold whose per-well unique gRNA mean is closest to ``target_unique_count``."""

        def _line_plot(df, x='fraction_threshold', y='unique_count', log_x=False, log_y=False):
            if x not in df.columns or y not in df.columns:
                raise ValueError(f"Columns '{x}' and/or '{y}' not found in the DataFrame.")
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

    closest_threshold = find_and_visualize_fraction_threshold(df, settings['target_unique_count'], log_x=settings['log_x'], log_y=settings['log_y'], dst=dst)

    # Apply the closest threshold to the DataFrame
    df = df[df['fraction'] >= closest_threshold]

    # Group by 'plateID', 'rowID', 'columnID' and compute unique counts of 'grna'
    unique_counts = df.groupby(['plateID', 'rowID', 'columnID'])['grna'].nunique().reset_index(name='unique_counts')
    unique_count_mean = df.groupby(['plateID', 'rowID', 'columnID'])['grna'].nunique().mean()
    unique_count_std = df.groupby(['plateID', 'rowID', 'columnID'])['grna'].nunique().std()

    # Merge the unique counts back into the original DataFrame
    df = pd.merge(df, unique_counts, on=['plateID', 'rowID', 'columnID'], how='left')

    print(f"unique_count mean: {unique_count_mean} std: {unique_count_std}")
    #_plot_density(df, dependent_variable='unique_counts')
    
    has_underscore = df['rowID'].str.contains('_').any()
    if has_underscore:
        df['rowID'] = df['rowID'].apply(lambda x: x.split('_')[1])
    
    plot_plates(df=df, variable='unique_counts', grouping='mean', min_max='allq', cmap='viridis',min_count=0, verbose=True, dst=dst)
    
    return closest_threshold