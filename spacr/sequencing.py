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
    """
    Maps DNA sequences to their corresponding names based on a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing 'sequence' and 'name' columns.
        sequences (list of str): List of DNA sequences to map.
        rc (bool): If True, reverse complement the sequences in the CSV before mapping.

    Returns:
        list: A list of names corresponding to the input sequences. If a sequence is not found,
              `pd.NA` is returned in its place.

    Notes:
        - The CSV file must contain columns named 'sequence' and 'name'.
        - If `rc` is True, sequences in the CSV will be reverse complemented prior to mapping.
        - Sequences in `sequences` are not altered—only sequences in the CSV are reverse complemented.
    """
    def rev_comp(dna_sequence):
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
    """
    Saves a pandas DataFrame to an HDF5 file, optionally appending to an existing dataset.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        hdf5_file (str): Path to the target HDF5 file.
        key (str, optional): Key under which to store the DataFrame. Defaults to 'df'.
        comp_type (str, optional): Compression algorithm to use (e.g., 'zlib', 'bzip2', 'blosc'). Defaults to 'zlib'.
        comp_level (int, optional): Compression level (0–9). Higher values yield better compression at the cost of speed. Defaults to 5.

    Returns:
        None

    Notes:
        - If the specified key already exists in the HDF5 file, the new DataFrame is appended to it.
        - The combined DataFrame is saved in 'table' format to support appending and querying.
        - Errors encountered during saving are printed to standard output.
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
    """
    Saves or appends a DataFrame of unique gRNA combinations to a CSV file, aggregating duplicates.

    Args:
        unique_combinations (pd.DataFrame): DataFrame containing 'rowID', 'columnID', and 'grna_name' columns,
                                            along with associated count or metric columns.
        csv_file (str): Path to the CSV file where data will be saved.

    Returns:
        None

    Notes:
        - If the file exists, it reads the existing contents and appends the new data.
        - Duplicate combinations (same 'rowID', 'columnID', 'grna_name') are summed.
        - The resulting DataFrame is saved with index included.
        - Any exception during the process is caught and printed to stdout.
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
    """
    Saves or appends a QC (quality control) DataFrame to a CSV file by summing overlapping entries.

    Args:
        qc_df (pd.DataFrame): DataFrame containing numeric QC metrics (e.g., counts, read stats).
        qc_csv_file (str): Path to the CSV file where the QC data will be saved.

    Returns:
        None

    Notes:
        - If the file exists, it reads the existing QC data and adds the new values to it (element-wise).
        - If the file doesn't exist, it creates a new one.
        - The final DataFrame is saved without including the index.
        - Any exception is caught and logged to stdout.
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
    """
    Extracts a subsequence and its corresponding quality scores.

    Args:
        sequence (str): DNA sequence string.
        quality (str): Quality string corresponding to the sequence.
        start (int): Start index of the region to extract.
        end (int): End index of the region to extract (exclusive).

    Returns:
        tuple: (subsequence, subquality) as strings.
    """
    return sequence[start:end], quality[start:end]

def create_consensus(seq1, qual1, seq2, qual2):
    """
    Constructs a consensus DNA sequence from two reads with associated quality scores.

    Args:
        seq1 (str): First DNA sequence.
        qual1 (str): Quality scores for `seq1` (as ASCII characters or integer-encoded).
        seq2 (str): Second DNA sequence.
        qual2 (str): Quality scores for `seq2`.

    Returns:
        str: Consensus sequence, selecting the base with the highest quality at each position.
             If one base is 'N', the non-'N' base is chosen regardless of quality.
    """
    consensus_seq = []
    for i in range(len(seq1)):
        bases = [(seq1[i], qual1[i]), (seq2[i], qual2[i])]
        consensus_seq.append(get_consensus_base(bases))
    return ''.join(consensus_seq)

def get_consensus_base(bases):
    """
    Selects the most reliable base from a list of two base-quality pairs.

    Args:
        bases (list of tuples): Each tuple contains (base, quality_score), expected length is 2.

    Returns:
        str: The consensus base. Prefers non-'N' bases and higher quality scores.
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
    """
    Computes the reverse complement of a DNA sequence.

    Args:
        seq (str): Input DNA sequence.

    Returns:
        str: Reverse complement of the input sequence.
    """
    return str(Seq(seq).reverse_complement())

# Core logic for processing a chunk (same as your original)
def process_chunk(chunk_data):
    """
    Processes a chunk of sequencing reads to extract and map barcode sequences to corresponding names.

    This function handles both single-end and paired-end FASTQ data. It searches for a target barcode
    sequence in each read, extracts a consensus region around it, applies a regex to extract barcodes,
    and maps those to known IDs using reference CSVs. Quality control data and unique combinations are
    also computed.

    Args:
        chunk_data (tuple): Contains either 9 or 10 elements:

            For paired-end mode (10 elements):
                - r1_chunk (list): List of strings, each 4-line block from R1 FASTQ.
                - r2_chunk (list): List of strings, each 4-line block from R2 FASTQ.
                - regex (str): Regex pattern with named groups ('rowID', 'columnID', 'grna').
                - target_sequence (str): Sequence to anchor barcode extraction.
                - offset_start (int): Offset from target_sequence to start consensus extraction.
                - expected_end (int): Length of the region to extract.
                - column_csv (str): Path to column barcode reference CSV.
                - grna_csv (str): Path to gRNA barcode reference CSV.
                - row_csv (str): Path to row barcode reference CSV.
                - fill_na (bool): Whether to fill unmapped names with raw barcode sequences.

            For single-end mode (9 elements):
                - Same as above, but r2_chunk is omitted.

    Returns:
        tuple:
            - df (pd.DataFrame): Full dataframe with columns:
              ['read', 'column_sequence', 'columnID', 'row_sequence', 'rowID',
              'grna_sequence', 'grna_name']
            - unique_combinations (pd.DataFrame): Count of each unique (rowID, columnID, grna_name) triplet.
            - qc_df (pd.DataFrame): Summary of missing values and total reads.
    """
    def paired_find_sequence_in_chunk_reads(r1_chunk, r2_chunk, target_sequence, offset_start, expected_end, regex):
        """
        Processes paired-end FASTQ read chunks to extract consensus barcode sequences and decode them
        using a regex pattern.

        For each R1–R2 read pair, this function identifies the `target_sequence`, extracts a window of 
        defined length with an offset, computes a consensus sequence using base quality scores, and 
        applies a regex to extract barcode components.

        Args:
            r1_chunk (list of str): List of 4-line strings for each R1 read in the chunk.
            r2_chunk (list of str): List of 4-line strings for each R2 read in the chunk.
            target_sequence (str): Nucleotide sequence used as anchor for barcode extraction.
            offset_start (int): Position offset from `target_sequence` to begin extracting barcode.
            expected_end (int): Total length of region to extract after offset.
            regex (str): Regular expression with named groups ('rowID', 'columnID', 'grna') 
                        to parse barcodes from the extracted consensus sequence.

        Returns:
            tuple:
                consensus_sequences (list of str): Consensus DNA sequences extracted from read pairs.
                columns (list of str): Extracted column barcode sequences.
                grnas (list of str): Extracted gRNA barcode sequences.
                rows (list of str): Extracted row barcode sequences.
        """
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
        """
        Processes single-end FASTQ read chunks to extract barcode sequences using a target motif and regex pattern.

        For each R1 read, the function identifies the `target_sequence`, extracts a region starting at an offset 
        and of fixed length, pads if necessary, and applies a regex with named groups to decode barcodes.

        Args:
            r1_chunk (list of str): List of 4-line strings for each R1 read in the chunk.
            target_sequence (str): Anchor sequence to locate the barcode region in R1.
            offset_start (int): Position offset from the end of `target_sequence` to start barcode extraction.
            expected_end (int): Total length of the barcode region to extract.
            regex (str): Regular expression with named groups ('rowID', 'columnID', 'grna') to extract barcodes.

        Returns:
            tuple:
                consensus_sequences (list of str): Extracted sequences used as barcode consensus (R1 only).
                columns (list of str): Extracted column barcode subsequences.
                grnas (list of str): Extracted gRNA barcode subsequences.
                rows (list of str): Extracted row barcode subsequences.
        """

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
    """
    Continuously reads data from a multiprocessing queue and saves it to disk in various formats.

    This function is intended to run in a separate process. It terminates when it receives the "STOP" sentinel value.

    Args:
        save_queue (multiprocessing.Queue): Queue containing tuples of (df, unique_combinations, qc_df).
        hdf5_file (str): Path to the HDF5 file to store full reads (only used if save_h5 is True).
        save_h5 (bool): Whether to save the full reads DataFrame to HDF5.
        unique_combinations_csv (str): Path to the CSV file for aggregated barcode combinations.
        qc_csv_file (str): Path to the CSV file for quality control statistics.
        comp_type (str): Compression algorithm for HDF5 (e.g., 'zlib').
        comp_level (int): Compression level for HDF5.
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
    """
    Processes paired-end FASTQ files in chunks to extract barcoded sequences and generate consensus reads.

    This function identifies sequences matching a regular expression in both R1 and R2 reads, extracts barcodes,
    and maps them to user-defined identifiers. Processed data is saved incrementally using a separate process.

    Args:
        r1_file (str): Path to the gzipped R1 FASTQ file.
        r2_file (str): Path to the gzipped R2 FASTQ file.
        regex (str): Regular expression with named capture groups: 'rowID', 'columnID', and 'grna'.
        target_sequence (str): Anchor sequence to align from.
        offset_start (int): Offset from anchor to start consensus extraction.
        expected_end (int): Length of the consensus region to extract.
        column_csv (str): Path to CSV file mapping column barcode sequences to IDs.
        grna_csv (str): Path to CSV file mapping gRNA barcode sequences to names.
        row_csv (str): Path to CSV file mapping row barcode sequences to IDs.
        save_h5 (bool): Whether to save the full reads DataFrame to HDF5.
        comp_type (str): Compression algorithm for HDF5 (e.g., 'zlib').
        comp_level (int): Compression level for HDF5.
        hdf5_file (str): Path to the HDF5 output file.
        unique_combinations_csv (str): Path to CSV file for saving unique row/column/gRNA combinations.
        qc_csv_file (str): Path to CSV file for saving QC summary (e.g., NaN counts).
        chunk_size (int, optional): Number of reads per batch. Defaults to 10000.
        n_jobs (int, optional): Number of parallel workers. Defaults to cpu_count() - 3.
        test (bool, optional): If True, processes only a single chunk and prints the result. Defaults to False.
        fill_na (bool, optional): If True, fills unmapped IDs with raw barcode sequences. Defaults to False.
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
    """
    Processes single-end FASTQ data in chunks to extract barcoded sequences and map them to identifiers.

    This function reads gzipped R1 FASTQ data, detects barcode-containing sequences using a target anchor and regex,
    and maps row, column, and gRNA barcodes to user-defined identifiers. Results are processed in parallel
    and saved incrementally via a background process.

    Args:
        r1_file (str): Path to gzipped R1 FASTQ file.
        r2_file (str): Placeholder for interface consistency; not used in single-end mode.
        regex (str): Regular expression with named capture groups: 'rowID', 'columnID', and 'grna'.
        target_sequence (str): Anchor sequence used to locate barcode region.
        offset_start (int): Offset from anchor to start barcode parsing.
        expected_end (int): Length of the barcode region to extract.
        column_csv (str): Path to CSV file mapping column barcode sequences to IDs.
        grna_csv (str): Path to CSV file mapping gRNA barcode sequences to names.
        row_csv (str): Path to CSV file mapping row barcode sequences to IDs.
        save_h5 (bool): Whether to save the full reads DataFrame to HDF5 format.
        comp_type (str): Compression algorithm for HDF5 (e.g., 'zlib').
        comp_level (int): Compression level for HDF5.
        hdf5_file (str): Output HDF5 file path.
        unique_combinations_csv (str): Output path for CSV summarizing row/column/gRNA combinations.
        qc_csv_file (str): Output path for CSV summarizing missing values and total reads.
        chunk_size (int, optional): Number of reads per batch. Defaults to 10,000.
        n_jobs (int, optional): Number of parallel worker processes. Defaults to cpu_count() - 3.
        test (bool, optional): If True, processes only the first chunk and prints its result. Defaults to False.
        fill_na (bool, optional): If True, fills missing mapped IDs with their corresponding barcode sequences. Defaults to False.
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

def generate_barecode_mapping(settings={}):
    """
    Orchestrates barcode extraction and mapping from gzipped sequencing data using user-defined or default settings.

    This function parses sequencing reads from single-end or paired-end FASTQ (.gz) files, extracts barcode regions
    using a regular expression, maps them to row, column, and gRNA identifiers, and saves the results to disk.
    Results include the full annotated reads (optional), barcode combination counts, and a QC summary.

    Args:
        settings (dict, optional): Dictionary containing parameters required for barcode mapping. If not provided,
            default values will be applied. Important keys include:
            - 'src' (str): Source directory containing gzipped FASTQ files.
            - 'mode' (str): Either 'single' or 'paired' for single-end or paired-end processing.
            - 'single_direction' (str): If 'single', specifies which read to use ('R1' or 'R2').
            - 'regex' (str): Regular expression with capture groups 'rowID', 'columnID', and 'grna'.
            - 'target_sequence' (str): Anchor sequence to locate barcode start position.
            - 'offset_start' (int): Offset from the anchor to the barcode start.
            - 'expected_end' (int): Expected barcode region length.
            - 'column_csv' (str): CSV file mapping column barcodes to names.
            - 'grna_csv' (str): CSV file mapping gRNA barcodes to names.
            - 'row_csv' (str): CSV file mapping row barcodes to names.
            - 'save_h5' (bool): Whether to save annotated reads to HDF5.
            - 'comp_type' (str): Compression algorithm for HDF5.
            - 'comp_level' (int): Compression level for HDF5.
            - 'chunk_size' (int): Number of reads to process per batch.
            - 'n_jobs' (int): Number of parallel processes for barcode mapping.
            - 'test' (bool): If True, only processes the first chunk for testing.
            - 'fill_na' (bool): If True, fills unmapped barcodes with raw sequence instead of NaN.

    Side Effects:
        Saves the following files in the output directory:
        - `annotated_reads.h5` (optional): Annotated read information in HDF5 format.
        - `unique_combinations.csv`: Count table of (rowID, columnID, grna_name) triplets.
        - `qc.csv`: Summary of missing values and read counts.
    """
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
    """
    Reads a barcode CSV file, computes the reverse complement of each sequence, and saves the result to a new CSV.

    This function assumes the input CSV contains a column named 'sequence' with DNA barcodes. It computes the
    reverse complement for each sequence and saves the modified DataFrame to a new file with '_RC' appended
    to the original filename.

    Args:
        csv_file (str): Path to the input CSV file. Must contain a column named 'sequence'.

    Side Effects:
        - Saves a new CSV file in the same directory with reverse-complemented sequences.
        - Prints the path of the saved file.

    Output:
        New file path format: <original_filename>_RC.csv
    """
    def reverse_complement(sequence):
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
    """
    Analyze and visualize sequencing quality metrics to determine an optimal fraction threshold
    that maximizes unique gRNA representation per well across plates.

    This function reads one or more CSV files containing count data, filters out control wells,
    calculates the fraction of reads per gRNA in each well, and identifies the minimum fraction 
    required to recover a target average number of unique gRNAs per well. It generates plots to 
    help visualize the chosen threshold and spatial distribution of unique gRNA counts.

    Args:
        settings (dict): Dictionary containing the following keys:
            - 'count_data' (str or list of str): Paths to CSV file(s) with 'grna', 'count', 'rowID', 'columnID' columns.
            - 'target_unique_count' (int): Target number of unique gRNAs per well to recover.
            - 'filter_column' (str): Column name to filter out control wells.
            - 'control_wells' (list): List of control well labels to exclude.
            - 'log_x' (bool): Whether to log-scale the x-axis in the threshold plot.
            - 'log_y' (bool): Whether to log-scale the y-axis in the threshold plot.

    Returns:
        float: Closest fraction threshold that approximates the target unique gRNA count per well.

    Side Effects:
        - Saves a PDF plot of unique gRNA count vs fraction threshold.
        - Saves a spatial plate map of unique gRNA counts.
        - Prints threshold and summary statistics.
        - Displays intermediate DataFrames for inspection.
    """
    from .utils import correct_metadata_column_names, correct_metadata

    def _plot_density(df, dependent_variable, dst=None):
        """
        Plot a kernel density estimate (KDE) of a specified variable from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            dependent_variable (str): Name of the column to plot.
            dst (str, optional): Directory to save the plot. If None, the plot is not saved.

        Side Effects:
            - Displays the KDE plot.
            - Saves the plot as 'dependent_variable_density.pdf' in the specified directory if dst is provided.
        """
        """Plot a density plot of the dependent variable."""
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
        """
        Identify the optimal fraction threshold that yields an average number of unique gRNAs per well 
        closest to a specified target, and visualize the relationship between threshold and unique count.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'fraction', 'plateID', 'rowID', 'columnID', and 'grna' columns.
            target_unique_count (int, optional): Desired average number of unique gRNAs per well. Default is 5.
            log_x (bool, optional): Whether to apply a log scale to the x-axis in the plot.
            log_y (bool, optional): Whether to apply a log scale to the y-axis in the plot.
            dst (str, optional): Directory where the plot will be saved. If None, the plot is not saved.

        Returns:
            float: The fraction threshold value closest to achieving the target_unique_count.

        Side Effects:
            - Displays a line plot of unique gRNA counts vs. fraction thresholds.
            - Saves the plot as 'fraction_threshold.pdf' in a subdirectory 'results/' under `dst` if provided.
        """

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