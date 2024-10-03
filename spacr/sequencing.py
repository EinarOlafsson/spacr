import os, gzip, re, time, gzip
import pandas as pd
from multiprocessing import Pool, cpu_count, Queue, Process
from Bio.Seq import Seq

# Function to map sequences to names (same as your original)
def map_sequences_to_names(csv_file, sequences, rc):
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
    try:
        with pd.HDFStore(hdf5_file, 'a', complib=comp_type, complevel=comp_level) as store:
            if key in store:
                existing_df = store[key]
                df = pd.concat([existing_df, df], ignore_index=True)
            store.put(key, df, format='table')
    except Exception as e:
        print(f"Error while saving DataFrame to HDF5: {e}")

def save_unique_combinations_to_csv(unique_combinations, csv_file):
    try:
        try:
            existing_df = pd.read_csv(csv_file)
        except FileNotFoundError:
            existing_df = pd.DataFrame()
        
        if not existing_df.empty:
            unique_combinations = pd.concat([existing_df, unique_combinations])
            unique_combinations = unique_combinations.groupby(
                ['row_name', 'column_name', 'grna_name'], as_index=False).sum()

        unique_combinations.to_csv(csv_file, index=False)
    except Exception as e:
        print(f"Error while saving unique combinations to CSV: {e}")

def save_qc_df_to_csv(qc_df, qc_csv_file):
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
    return sequence[start:end], quality[start:end]

def create_consensus(seq1, qual1, seq2, qual2):
    consensus_seq = []
    for i in range(len(seq1)):
        bases = [(seq1[i], qual1[i]), (seq2[i], qual2[i])]
        consensus_seq.append(get_consensus_base(bases))
    return ''.join(consensus_seq)

def get_consensus_base(bases):
    # Prefer non-'N' bases, if 'N' exists, pick the other one.
    if bases[0][0] == 'N':
        return bases[1][0]
    elif bases[1][0] == 'N':
        return bases[0][0]
    else:
        # Return the base with the highest quality score
        return bases[0][0] if bases[0][1] >= bases[1][1] else bases[1][0]

def reverse_complement(seq):
    return str(Seq(seq).reverse_complement())

# Core logic for processing a chunk (same as your original)
def process_chunk(chunk_data):
    
    def paired_find_sequence_in_chunk_reads(r1_chunk, r2_chunk, target_sequence, offset_start, expected_end, regex):

        consensus_sequences, columns, grnas, rows = [], [], [], []

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
                        column_sequence = match.group('column')
                        grna_sequence = match.group('grna')
                        row_sequence = match.group('row')
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
    
    def single_find_sequence_in_chunk_reads(r1_chunk, target_sequence, offset_start, expected_end, regex):

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
                        column_sequence = match.group('column')
                        grna_sequence = match.group('grna')
                        row_sequence = match.group('row')
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

    if len(chunk_data) == 9:
        r1_chunk, r2_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv = chunk_data
    if len(chunk_data) == 8:
        r1_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv = chunk_data
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
        'column_name': column_names,
        'row_sequence': rows,
        'row_name': row_names,
        'grna_sequence': grnas,
        'grna_name': grna_names
    })

    qc_df = df.isna().sum().to_frame().T
    qc_df.columns = df.columns
    qc_df.index = ["NaN_Counts"]
    qc_df['total_reads'] = len(df)

    unique_combinations = df.groupby(['row_name', 'column_name', 'grna_name']).size().reset_index(name='count')
    return df, unique_combinations, qc_df

# Function to save data from the queue
def saver_process(save_queue, hdf5_file, save_h5, unique_combinations_csv, qc_csv_file, comp_type, comp_level):
    while True:
        item = save_queue.get()
        if item == "STOP":
            break
        df, unique_combinations, qc_df = item
        if save_h5:
            save_df_to_hdf5(df, hdf5_file, key='df', comp_type=comp_type, comp_level=comp_level)
        save_unique_combinations_to_csv(unique_combinations, unique_combinations_csv)
        save_qc_df_to_csv(qc_df, qc_csv_file)

def paired_read_chunked_processing(r1_file, r2_file, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, save_h5, comp_type, comp_level, hdf5_file, unique_combinations_csv, qc_csv_file, chunk_size=10000, n_jobs=None, test=False):

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
            chunk_data = (r1_chunk, r2_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv)

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

def single_read_chunked_processing(r1_file, r2_file, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, save_h5, comp_type, comp_level, hdf5_file, unique_combinations_csv, qc_csv_file, chunk_size=10000, n_jobs=None, test=False):

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
            chunk_data = (r1_chunk, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv)

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

    from .settings import set_default_generate_barecode_mapping
    from .utils import save_settings
    from .io import parse_gz_files

    settings = set_default_generate_barecode_mapping(settings)
    save_settings(settings, name=f"sequencing_{settings['mode']}_{settings['single_direction']}", show=True)

    regex = settings['regex']

    print(f'Using regex: {regex} to extract barcode information')

    samples_dict = parse_gz_files(settings['src'])

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
                     test=settings['test'])

# Function to read the CSV, compute reverse complement, and save it
def barecodes_reverse_complement(csv_file):

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