spacr.sequencing
================

.. py:module:: spacr.sequencing




Module Contents
---------------

.. py:function:: map_sequences_to_names(csv_file, sequences, rc)

   Maps DNA sequences to their corresponding names based on a CSV file.

   :param csv_file: Path to the CSV file containing 'sequence' and 'name' columns.
   :type csv_file: str
   :param sequences: List of DNA sequences to map.
   :type sequences: list of str
   :param rc: If True, reverse complement the sequences in the CSV before mapping.
   :type rc: bool

   :returns:

             A list of names corresponding to the input sequences. If a sequence is not found,
                   `pd.NA` is returned in its place.
   :rtype: list

   .. rubric:: Notes

   - The CSV file must contain columns named 'sequence' and 'name'.
   - If `rc` is True, sequences in the CSV will be reverse complemented prior to mapping.
   - Sequences in `sequences` are not altered—only sequences in the CSV are reverse complemented.


.. py:function:: save_df_to_hdf5(df, hdf5_file, key='df', comp_type='zlib', comp_level=5)

   Saves a pandas DataFrame to an HDF5 file, optionally appending to an existing dataset.

   :param df: The DataFrame to save.
   :type df: pd.DataFrame
   :param hdf5_file: Path to the target HDF5 file.
   :type hdf5_file: str
   :param key: Key under which to store the DataFrame. Defaults to 'df'.
   :type key: str, optional
   :param comp_type: Compression algorithm to use (e.g., 'zlib', 'bzip2', 'blosc'). Defaults to 'zlib'.
   :type comp_type: str, optional
   :param comp_level: Compression level (0–9). Higher values yield better compression at the cost of speed. Defaults to 5.
   :type comp_level: int, optional

   :returns: None

   .. rubric:: Notes

   - If the specified key already exists in the HDF5 file, the new DataFrame is appended to it.
   - The combined DataFrame is saved in 'table' format to support appending and querying.
   - Errors encountered during saving are printed to standard output.


.. py:function:: save_unique_combinations_to_csv(unique_combinations, csv_file)

   Saves or appends a DataFrame of unique gRNA combinations to a CSV file, aggregating duplicates.

   :param unique_combinations: DataFrame containing 'rowID', 'columnID', and 'grna_name' columns,
                               along with associated count or metric columns.
   :type unique_combinations: pd.DataFrame
   :param csv_file: Path to the CSV file where data will be saved.
   :type csv_file: str

   :returns: None

   .. rubric:: Notes

   - If the file exists, it reads the existing contents and appends the new data.
   - Duplicate combinations (same 'rowID', 'columnID', 'grna_name') are summed.
   - The resulting DataFrame is saved with index included.
   - Any exception during the process is caught and printed to stdout.


.. py:function:: save_qc_df_to_csv(qc_df, qc_csv_file)

   Saves or appends a QC (quality control) DataFrame to a CSV file by summing overlapping entries.

   :param qc_df: DataFrame containing numeric QC metrics (e.g., counts, read stats).
   :type qc_df: pd.DataFrame
   :param qc_csv_file: Path to the CSV file where the QC data will be saved.
   :type qc_csv_file: str

   :returns: None

   .. rubric:: Notes

   - If the file exists, it reads the existing QC data and adds the new values to it (element-wise).
   - If the file doesn't exist, it creates a new one.
   - The final DataFrame is saved without including the index.
   - Any exception is caught and logged to stdout.


.. py:function:: extract_sequence_and_quality(sequence, quality, start, end)

   Extracts a subsequence and its corresponding quality scores.

   :param sequence: DNA sequence string.
   :type sequence: str
   :param quality: Quality string corresponding to the sequence.
   :type quality: str
   :param start: Start index of the region to extract.
   :type start: int
   :param end: End index of the region to extract (exclusive).
   :type end: int

   :returns: (subsequence, subquality) as strings.
   :rtype: tuple


.. py:function:: create_consensus(seq1, qual1, seq2, qual2)

   Constructs a consensus DNA sequence from two reads with associated quality scores.

   :param seq1: First DNA sequence.
   :type seq1: str
   :param qual1: Quality scores for `seq1` (as ASCII characters or integer-encoded).
   :type qual1: str
   :param seq2: Second DNA sequence.
   :type seq2: str
   :param qual2: Quality scores for `seq2`.
   :type qual2: str

   :returns:

             Consensus sequence, selecting the base with the highest quality at each position.
                  If one base is 'N', the non-'N' base is chosen regardless of quality.
   :rtype: str


.. py:function:: get_consensus_base(bases)

   Selects the most reliable base from a list of two base-quality pairs.

   :param bases: Each tuple contains (base, quality_score), expected length is 2.
   :type bases: list of tuples

   :returns: The consensus base. Prefers non-'N' bases and higher quality scores.
   :rtype: str


.. py:function:: reverse_complement(seq)

   Computes the reverse complement of a DNA sequence.

   :param seq: Input DNA sequence.
   :type seq: str

   :returns: Reverse complement of the input sequence.
   :rtype: str


.. py:function:: process_chunk(chunk_data)

   Processes a chunk of sequencing reads to extract and map barcode sequences to corresponding names.

   This function handles both single-end and paired-end FASTQ data. It searches for a target barcode
   sequence in each read, extracts a consensus region around it, applies a regex to extract barcodes,
   and maps those to known IDs using reference CSVs. Quality control data and unique combinations are
   also computed.

   :param chunk_data: Contains either 9 or 10 elements:

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
   :type chunk_data: tuple

   :returns:

                 - df (pd.DataFrame): Full dataframe with columns:
                   ['read', 'column_sequence', 'columnID', 'row_sequence', 'rowID',
                   'grna_sequence', 'grna_name']
                 - unique_combinations (pd.DataFrame): Count of each unique (rowID, columnID, grna_name) triplet.
                 - qc_df (pd.DataFrame): Summary of missing values and total reads.
   :rtype: tuple


.. py:function:: saver_process(save_queue, hdf5_file, save_h5, unique_combinations_csv, qc_csv_file, comp_type, comp_level)

   Continuously reads data from a multiprocessing queue and saves it to disk in various formats.

   This function is intended to run in a separate process. It terminates when it receives the "STOP" sentinel value.

   :param save_queue: Queue containing tuples of (df, unique_combinations, qc_df).
   :type save_queue: multiprocessing.Queue
   :param hdf5_file: Path to the HDF5 file to store full reads (only used if save_h5 is True).
   :type hdf5_file: str
   :param save_h5: Whether to save the full reads DataFrame to HDF5.
   :type save_h5: bool
   :param unique_combinations_csv: Path to the CSV file for aggregated barcode combinations.
   :type unique_combinations_csv: str
   :param qc_csv_file: Path to the CSV file for quality control statistics.
   :type qc_csv_file: str
   :param comp_type: Compression algorithm for HDF5 (e.g., 'zlib').
   :type comp_type: str
   :param comp_level: Compression level for HDF5.
   :type comp_level: int


.. py:function:: paired_read_chunked_processing(r1_file, r2_file, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, save_h5, comp_type, comp_level, hdf5_file, unique_combinations_csv, qc_csv_file, chunk_size=10000, n_jobs=None, test=False, fill_na=False)

   Processes paired-end FASTQ files in chunks to extract barcoded sequences and generate consensus reads.

   This function identifies sequences matching a regular expression in both R1 and R2 reads, extracts barcodes,
   and maps them to user-defined identifiers. Processed data is saved incrementally using a separate process.

   :param r1_file: Path to the gzipped R1 FASTQ file.
   :type r1_file: str
   :param r2_file: Path to the gzipped R2 FASTQ file.
   :type r2_file: str
   :param regex: Regular expression with named capture groups: 'rowID', 'columnID', and 'grna'.
   :type regex: str
   :param target_sequence: Anchor sequence to align from.
   :type target_sequence: str
   :param offset_start: Offset from anchor to start consensus extraction.
   :type offset_start: int
   :param expected_end: Length of the consensus region to extract.
   :type expected_end: int
   :param column_csv: Path to CSV file mapping column barcode sequences to IDs.
   :type column_csv: str
   :param grna_csv: Path to CSV file mapping gRNA barcode sequences to names.
   :type grna_csv: str
   :param row_csv: Path to CSV file mapping row barcode sequences to IDs.
   :type row_csv: str
   :param save_h5: Whether to save the full reads DataFrame to HDF5.
   :type save_h5: bool
   :param comp_type: Compression algorithm for HDF5 (e.g., 'zlib').
   :type comp_type: str
   :param comp_level: Compression level for HDF5.
   :type comp_level: int
   :param hdf5_file: Path to the HDF5 output file.
   :type hdf5_file: str
   :param unique_combinations_csv: Path to CSV file for saving unique row/column/gRNA combinations.
   :type unique_combinations_csv: str
   :param qc_csv_file: Path to CSV file for saving QC summary (e.g., NaN counts).
   :type qc_csv_file: str
   :param chunk_size: Number of reads per batch. Defaults to 10000.
   :type chunk_size: int, optional
   :param n_jobs: Number of parallel workers. Defaults to cpu_count() - 3.
   :type n_jobs: int, optional
   :param test: If True, processes only a single chunk and prints the result. Defaults to False.
   :type test: bool, optional
   :param fill_na: If True, fills unmapped IDs with raw barcode sequences. Defaults to False.
   :type fill_na: bool, optional


.. py:function:: single_read_chunked_processing(r1_file, r2_file, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, save_h5, comp_type, comp_level, hdf5_file, unique_combinations_csv, qc_csv_file, chunk_size=10000, n_jobs=None, test=False, fill_na=False)

   Processes single-end FASTQ data in chunks to extract barcoded sequences and map them to identifiers.

   This function reads gzipped R1 FASTQ data, detects barcode-containing sequences using a target anchor and regex,
   and maps row, column, and gRNA barcodes to user-defined identifiers. Results are processed in parallel
   and saved incrementally via a background process.

   :param r1_file: Path to gzipped R1 FASTQ file.
   :type r1_file: str
   :param r2_file: Placeholder for interface consistency; not used in single-end mode.
   :type r2_file: str
   :param regex: Regular expression with named capture groups: 'rowID', 'columnID', and 'grna'.
   :type regex: str
   :param target_sequence: Anchor sequence used to locate barcode region.
   :type target_sequence: str
   :param offset_start: Offset from anchor to start barcode parsing.
   :type offset_start: int
   :param expected_end: Length of the barcode region to extract.
   :type expected_end: int
   :param column_csv: Path to CSV file mapping column barcode sequences to IDs.
   :type column_csv: str
   :param grna_csv: Path to CSV file mapping gRNA barcode sequences to names.
   :type grna_csv: str
   :param row_csv: Path to CSV file mapping row barcode sequences to IDs.
   :type row_csv: str
   :param save_h5: Whether to save the full reads DataFrame to HDF5 format.
   :type save_h5: bool
   :param comp_type: Compression algorithm for HDF5 (e.g., 'zlib').
   :type comp_type: str
   :param comp_level: Compression level for HDF5.
   :type comp_level: int
   :param hdf5_file: Output HDF5 file path.
   :type hdf5_file: str
   :param unique_combinations_csv: Output path for CSV summarizing row/column/gRNA combinations.
   :type unique_combinations_csv: str
   :param qc_csv_file: Output path for CSV summarizing missing values and total reads.
   :type qc_csv_file: str
   :param chunk_size: Number of reads per batch. Defaults to 10,000.
   :type chunk_size: int, optional
   :param n_jobs: Number of parallel worker processes. Defaults to cpu_count() - 3.
   :type n_jobs: int, optional
   :param test: If True, processes only the first chunk and prints its result. Defaults to False.
   :type test: bool, optional
   :param fill_na: If True, fills missing mapped IDs with their corresponding barcode sequences. Defaults to False.
   :type fill_na: bool, optional


.. py:function:: generate_barecode_mapping(settings={})

   Orchestrates barcode extraction and mapping from gzipped sequencing data using user-defined or default settings.

   This function parses sequencing reads from single-end or paired-end FASTQ (.gz) files, extracts barcode regions
   using a regular expression, maps them to row, column, and gRNA identifiers, and saves the results to disk.
   Results include the full annotated reads (optional), barcode combination counts, and a QC summary.

   :param settings: Dictionary containing parameters required for barcode mapping. If not provided,
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
   :type settings: dict, optional

   Side Effects:
       Saves the following files in the output directory:
       - `annotated_reads.h5` (optional): Annotated read information in HDF5 format.
       - `unique_combinations.csv`: Count table of (rowID, columnID, grna_name) triplets.
       - `qc.csv`: Summary of missing values and read counts.


.. py:function:: barecodes_reverse_complement(csv_file)

   Reads a barcode CSV file, computes the reverse complement of each sequence, and saves the result to a new CSV.

   This function assumes the input CSV contains a column named 'sequence' with DNA barcodes. It computes the
   reverse complement for each sequence and saves the modified DataFrame to a new file with '_RC' appended
   to the original filename.

   :param csv_file: Path to the input CSV file. Must contain a column named 'sequence'.
   :type csv_file: str

   Side Effects:
       - Saves a new CSV file in the same directory with reverse-complemented sequences.
       - Prints the path of the saved file.

   Output:
       New file path format: <original_filename>_RC.csv


.. py:function:: graph_sequencing_stats(settings)

   Analyze and visualize sequencing quality metrics to determine an optimal fraction threshold
   that maximizes unique gRNA representation per well across plates.

   This function reads one or more CSV files containing count data, filters out control wells,
   calculates the fraction of reads per gRNA in each well, and identifies the minimum fraction
   required to recover a target average number of unique gRNAs per well. It generates plots to
   help visualize the chosen threshold and spatial distribution of unique gRNA counts.

   :param settings: Dictionary containing the following keys:
                    - 'count_data' (str or list of str): Paths to CSV file(s) with 'grna', 'count', 'rowID', 'columnID' columns.
                    - 'target_unique_count' (int): Target number of unique gRNAs per well to recover.
                    - 'filter_column' (str): Column name to filter out control wells.
                    - 'control_wells' (list): List of control well labels to exclude.
                    - 'log_x' (bool): Whether to log-scale the x-axis in the threshold plot.
                    - 'log_y' (bool): Whether to log-scale the y-axis in the threshold plot.
   :type settings: dict

   :returns: Closest fraction threshold that approximates the target unique gRNA count per well.
   :rtype: float

   Side Effects:
       - Saves a PDF plot of unique gRNA count vs fraction threshold.
       - Saves a spatial plate map of unique gRNA counts.
       - Prints threshold and summary statistics.
       - Displays intermediate DataFrames for inspection.


