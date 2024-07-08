import os, gc, gzip, re, time, math, subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.Align import PairwiseAligner
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import pairwise2
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import gmean
from difflib import SequenceMatcher
from collections import Counter
from IPython.display import display

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from scipy.stats import shapiro
from patsy import dmatrices

def analyze_reads(settings):
    """
    Analyzes reads from gzipped fastq files and combines them based on specified settings.

    Args:
        settings (dict): A dictionary containing the following keys:
            - 'src' (str): The path to the folder containing the input fastq files.
            - 'upstream' (str, optional): The upstream sequence used for read combination. Defaults to 'CTTCTGGTAAATGGGGATGTCAAGTT'.
            - 'downstream' (str, optional): The downstream sequence used for read combination. Defaults to 'GTTTAAGAGCTATGCTGGAAACAGCA'.
            - 'barecode_length' (int, optional): The length of the barcode sequence. Defaults to 8.
            - 'chunk_size' (int, optional): The number of reads to process and save at a time. Defaults to 1000000.

    Returns:
        None
    """
    
    def save_chunk_to_hdf5_v1(output_file_path, data_chunk, chunk_counter):
        """
        Save a data chunk to an HDF5 file.

        Parameters:
        - output_file_path (str): The path to the output HDF5 file.
        - data_chunk (list): The data chunk to be saved.
        - chunk_counter (int): The counter for the current chunk.

        Returns:
        None
        """
        df = pd.DataFrame(data_chunk, columns=['combined_read', 'grna', 'plate_row', 'column', 'sample'])
        with pd.HDFStore(output_file_path, mode='a', complevel=5, complib='blosc') as store:
            store.put(f'reads/chunk_{chunk_counter}', df, format='table', append=True)

    def save_chunk_to_hdf5(output_file_path, data_chunk, chunk_counter):
        """
        Save a data chunk to an HDF5 file.

        Parameters:
        - output_file_path (str): The path to the output HDF5 file.
        - data_chunk (list): The data chunk to be saved.
        - chunk_counter (int): The counter for the current chunk.

        Returns:
        None
        """
        df = pd.DataFrame(data_chunk, columns=['combined_read', 'grna', 'plate_row', 'column', 'sample'])
        with pd.HDFStore(output_file_path, mode='a', complevel=5, complib='blosc') as store:
            store.put(
                f'reads/chunk_{chunk_counter}', 
                df, 
                format='table', 
                append=True, 
                min_itemsize={'combined_read': 300, 'grna': 50, 'plate_row': 20, 'column': 20, 'sample': 50}
            )

    def reverse_complement(seq):
        """
        Returns the reverse complement of a DNA sequence.

        Args:
            seq (str): The DNA sequence to be reversed and complemented.

        Returns:
            str: The reverse complement of the input DNA sequence.

        Example:
            >>> reverse_complement('ATCG')
            'CGAT'
        """
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement[base] for base in reversed(seq))
    
    def get_avg_read_length(file_path, num_reads=100):
        """
        Calculate the average read length from a given file.

        Args:
            file_path (str): The path to the input file.
            num_reads (int, optional): The number of reads to process. Defaults to 100.

        Returns:
            float: The average read length.

        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        if not file_path:
            return 0
        total_length = 0
        count = 0
        with gzip.open(file_path, 'rt') as f:
            for _ in range(num_reads):
                try:
                    f.readline()  # Skip index line
                    read = f.readline().strip()
                    total_length += len(read)
                    f.readline()  # Skip plus line
                    f.readline()  # Skip quality line
                    count += 1
                except StopIteration:
                    break
        return total_length / count if count > 0 else 0
    
    def parse_gz_files(folder_path):
        """
        Parses the .fastq.gz files in the specified folder path and returns a dictionary
        containing the sample names and their corresponding file paths.

        Args:
            folder_path (str): The path to the folder containing the .fastq.gz files.

        Returns:
            dict: A dictionary where the keys are the sample names and the values are
            dictionaries containing the file paths for the 'R1' and 'R2' read directions.
        """
        files = os.listdir(folder_path)
        gz_files = [f for f in files if f.endswith('.fastq.gz')]

        samples_dict = {}
        for gz_file in gz_files:
            parts = gz_file.split('_')
            sample_name = parts[0]
            read_direction = parts[1]

            if sample_name not in samples_dict:
                samples_dict[sample_name] = {}

            if read_direction == "R1":
                samples_dict[sample_name]['R1'] = os.path.join(folder_path, gz_file)
            elif read_direction == "R2":
                samples_dict[sample_name]['R2'] = os.path.join(folder_path, gz_file)

        return samples_dict
    
    def find_overlap(r1_read_rc, r2_read):
        """
        Find the best alignment between two DNA reads.

        Parameters:
        - r1_read_rc (str): The reverse complement of the first DNA read.
        - r2_read (str): The second DNA read.

        Returns:
        - best_alignment (Alignment): The best alignment between the two DNA reads.
        """
        aligner = PairwiseAligner()
        alignments = aligner.align(r1_read_rc, r2_read)
        best_alignment = alignments[0]
        return best_alignment

    def combine_reads(samples_dict, src, chunk_size, barecode_length_1, barecode_length_2, upstream, downstream):
        """
        Combine reads from paired-end sequencing files and save the combined reads to a new file.
        
        Args:
            samples_dict (dict): A dictionary mapping sample names to file paths of paired-end sequencing files.
            src (str): The source directory where the combined reads will be saved.
            chunk_size (int): The number of reads to be processed and saved as a chunk.
            barecode_length (int): The length of the barcode sequence.
            upstream (str): The upstream sequence used for read splitting.
            downstream (str): The downstream sequence used for read splitting.
        
        Returns:
            None
        """
        dst = os.path.join(src, 'combined_reads')
        if not os.path.exists(dst):
            os.makedirs(dst)

        for sample, paths in samples_dict.items():
            print(f'Processing: {sample} with the files: {paths}')
            r1_path = paths.get('R1')
            r2_path = paths.get('R2')

            output_file_path = os.path.join(dst, f"{sample}_combined.h5")
            qc_file_path = os.path.join(dst, f"{sample}_qc.csv")

            r1_file = gzip.open(r1_path, 'rt') if r1_path else None
            r2_file = gzip.open(r2_path, 'rt') if r2_path else None

            chunk_counter = 0
            data_chunk = []
            
            success = 0
            fail = 0

            # Calculate initial average read length
            avg_read_length_r1 = get_avg_read_length(r1_path, 100)
            avg_read_length_r2 = get_avg_read_length(r2_path, 100)
            avg_read_length = (avg_read_length_r1 + avg_read_length_r2) / 2 if avg_read_length_r1 and avg_read_length_r2 else 0
            
            print(f'Initial avg_read_length: {avg_read_length}')
            
            # Estimate the initial number of reads based on the file size
            r1_size_est = os.path.getsize(r1_path) // (avg_read_length * 4) if r1_path else 0
            r2_size_est = os.path.getsize(r2_path) // (avg_read_length * 4) if r2_path else 0
            max_size = max(r1_size_est, r2_size_est) * 10
            test10 =0
            with tqdm(total=max_size, desc=f"Processing {sample}") as pbar:
                total_length_processed = 0
                read_count = 0
                
                while True:
                    try:
                        r1_index = next(r1_file).strip() if r1_file else None
                        r1_read = next(r1_file).strip() if r1_file else None
                        r1_plus = next(r1_file).strip() if r1_file else None
                        r1_quality = next(r1_file).strip() if r1_file else None

                        r2_index = next(r2_file).strip() if r2_file else None
                        r2_read = next(r2_file).strip() if r2_file else None
                        r2_plus = next(r2_file).strip() if r2_file else None
                        r2_quality = next(r2_file).strip() if r2_file else None

                        pbar.update(1)

                        if r1_index and r2_index and r1_index.split(' ')[0] != r2_index.split(' ')[0]:
                            fail += 1
                            print(f"Index mismatch: {r1_index} != {r2_index}")
                            continue

                        r1_read_rc = reverse_complement(r1_read) if r1_read else ''
                        r1_quality_rc = r1_quality[::-1] if r1_quality else ''

                        r1_rc_split_index = r1_read_rc.find(upstream)
                        r2_split_index = r2_read.find(upstream)

                        if r1_rc_split_index == -1 or r2_split_index == -1:
                            fail += 1
                            continue
                        else:
                            success += 1

                        read1_fragment = r1_read_rc[:r1_rc_split_index]
                        read2_fragment = r2_read[r2_split_index:]
                        read_combo = read1_fragment + read2_fragment

                        combo_split_index_1 = read_combo.find(upstream)
                        combo_split_index_2 = read_combo.find(downstream)

                        barcode_1 = read_combo[combo_split_index_1 - barecode_length_1:combo_split_index_1]
                        grna = read_combo[combo_split_index_1 + len(upstream):combo_split_index_2]
                        barcode_2 = read_combo[combo_split_index_2 + len(downstream):combo_split_index_2 + len(downstream) + barecode_length_2]
                        barcode_2 = reverse_complement(barcode_2)
                        data_chunk.append((read_combo, grna, barcode_1, barcode_2, sample))

                        if settings['test']:
                            if read_count % 1000 == 0:
                                print(f"Read count: {read_count}")
                                print(f"Read 1: {r1_read_rc}")
                                print(f"Read 2: {r2_read}")
                                print(f"Read combo: {read_combo}")
                                print(f"Barcode 1: {barcode_1}")
                                print(f"gRNA: {grna}")
                                print(f"Barcode 2: {barcode_2}")
                                print()
                                test10 += 1
                                if test10 == 10:
                                    break

                        read_count += 1
                        total_length_processed += len(r1_read) + len(r2_read)

                        # Periodically update the average read length and total
                        if read_count % 10000 == 0:
                            avg_read_length = total_length_processed / (read_count * 2)
                            max_size = (os.path.getsize(r1_path) + os.path.getsize(r2_path)) // (avg_read_length * 4)
                            pbar.total = max_size

                        if len(data_chunk) >= chunk_size:
                            save_chunk_to_hdf5(output_file_path, data_chunk, chunk_counter)
                            chunk_counter += 1
                            data_chunk = []

                    except StopIteration:
                        break

                # Save any remaining data_chunk
                if data_chunk:
                    save_chunk_to_hdf5(output_file_path, data_chunk, chunk_counter)

                # Save QC metrics
                qc = {'success': success, 'failed': fail}
                qc_df = pd.DataFrame([qc])
                qc_df.to_csv(qc_file_path, index=False)
                
    settings.setdefault('upstream', 'CTTCTGGTAAATGGGGATGTCAAGTT') 
    settings.setdefault('downstream', 'GTTTAAGAGCTATGCTGGAAACAGCAG') #This is the reverce compliment of the column primer starting from the end #TGCTGTTTAAGAGCTATGCTGGAAACAGCA
    settings.setdefault('barecode_length_1', 8)
    settings.setdefault('barecode_length_2', 7)
    settings.setdefault('chunk_size', 1000000)
    settings.setdefault('test', False)
    
    samples_dict = parse_gz_files(settings['src'])
    combine_reads(samples_dict, settings['src'], settings['chunk_size'], settings['barecode_length_1'], settings['barecode_length_2'], settings['upstream'], settings['downstream'])

def map_barcodes(h5_file_path, settings={}):
    """
    Maps barcodes and performs quality control on sequencing data.

    Args:
        h5_file_path (str): The file path to the HDF5 file containing the sequencing data.
        settings (dict, optional): Additional settings for the mapping and quality control process. Defaults to {}.

    Returns:
        None
    """
    def get_read_qc(df, settings):
        """
        Calculate quality control metrics for sequencing reads.

        Parameters:
        - df: DataFrame containing the sequencing reads.

        Returns:
        - df_cleaned: DataFrame containing the cleaned sequencing reads.
        - qc_dict: Dictionary containing the quality control metrics.
        """
        
        df_cleaned = df.dropna()

        qc_dict = {}
        qc_dict['reads'] = len(df)
        qc_dict['cleaned_reads'] = len(df_cleaned)
        qc_dict['NaN_grna'] = df['grna_metadata'].isna().sum()
        qc_dict['NaN_plate_row'] = df['plate_row_metadata'].isna().sum()
        qc_dict['NaN_column'] = df['column_metadata'].isna().sum()
        qc_dict['NaN_plate'] = df['plate_metadata'].isna().sum()
        qc_dict['unique_grna'] = Counter(df['grna_metadata'].dropna().tolist())
        qc_dict['unique_plate_row'] = Counter(df['plate_row_metadata'].dropna().tolist())
        qc_dict['unique_column'] = Counter(df['column_metadata'].dropna().tolist())
        qc_dict['unique_plate'] = Counter(df['plate_metadata'].dropna().tolist())

        # Calculate control error rates using cleaned DataFrame
        total_pc_non_nan = df_cleaned[(df_cleaned['column_metadata'] == settings['pc_loc'])].shape[0]
        total_nc_non_nan = df_cleaned[(df_cleaned['column_metadata'] == settings['nc_loc'])].shape[0]
        
        pc_count_pc = df_cleaned[(df_cleaned['column_metadata'] == settings['pc_loc']) & (df_cleaned['grna_metadata'] == settings['pc'])].shape[0]
        nc_count_nc = df_cleaned[(df_cleaned['column_metadata'] == settings['nc_loc']) & (df_cleaned['grna_metadata'] == settings['nc'])].shape[0]

        pc_error_count = df_cleaned[(df_cleaned['column_metadata'] == settings['pc_loc']) & (df_cleaned['grna_metadata'] != settings['pc'])].shape[0]
        nc_error_count = df_cleaned[(df_cleaned['column_metadata'] == settings['nc_loc']) & (df_cleaned['grna_metadata'] != settings['nc'])].shape[0]
        
        pc_in_nc_loc_count = df_cleaned[(df_cleaned['column_metadata'] == settings['nc_loc']) & (df_cleaned['grna_metadata'] == settings['pc'])].shape[0]
        nc_in_pc_loc_count = df_cleaned[(df_cleaned['column_metadata'] == settings['pc_loc']) & (df_cleaned['grna_metadata'] == settings['nc'])].shape[0]
        
        # Collect QC metrics into a dictionary
        # PC 
        qc_dict['pc_total_count'] = total_pc_non_nan
        qc_dict['pc_count_pc'] = pc_count_pc
        qc_dict['nc_count_pc'] = pc_in_nc_loc_count
        qc_dict['pc_error_count'] = pc_error_count
        # NC
        qc_dict['nc_total_count'] = total_nc_non_nan
        qc_dict['nc_count_nc'] = nc_count_nc
        qc_dict['pc_count_nc'] = nc_in_pc_loc_count
        qc_dict['nc_error_count'] = nc_error_count
        
        return df_cleaned, qc_dict

    def get_per_row_qc(df, settings):
        """
        Calculate quality control metrics for each unique row in the control columns.

        Parameters:
        - df: DataFrame containing the sequencing reads.
        - settings: Dictionary containing the settings for control values.

        Returns:
        - dict: Dictionary containing the quality control metrics for each unique row.
        """
        qc_dict_per_row = {}
        unique_rows = df['plate_row_metadata'].dropna().unique().tolist()
        unique_rows = list(set(unique_rows))  # Remove duplicates

        for row in unique_rows:
            df_row = df[(df['plate_row_metadata'] == row)]
            _, qc_dict_row = get_read_qc(df_row, settings)
            qc_dict_per_row[row] = qc_dict_row

        return qc_dict_per_row

    def mapping_dicts(df, settings):
        """
        Maps the values in the DataFrame columns to corresponding metadata using dictionaries.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data to be mapped.
            settings (dict): A dictionary containing the settings for mapping.

        Returns:
            pandas.DataFrame: The DataFrame with the mapped metadata columns added.
        """
        grna_df = pd.read_csv(settings['grna'])
        barcode_df = pd.read_csv(settings['barcodes'])

        grna_dict = {row['sequence']: row['name'] for _, row in grna_df.iterrows()}
        plate_row_dict = {row['sequence']: row['name'] for _, row in barcode_df.iterrows() if row['name'].startswith('p')}
        column_dict = {row['sequence']: row['name'] for _, row in barcode_df.iterrows() if row['name'].startswith('c')}
        plate_dict = settings['plate_dict']

        df['grna_metadata'] = df['grna'].map(grna_dict)
        df['grna_length'] = df['grna'].apply(len)
        df['plate_row_metadata'] = df['plate_row'].map(plate_row_dict)
        df['column_metadata'] = df['column'].map(column_dict)
        df['plate_metadata'] = df['sample'].map(plate_dict)

        return df
    
    def filter_combinations(df, settings):
        """
        Takes the combination counts Data Frame, filters the rows based on specific conditions, 
        and removes rows with a count lower than the highest value of max_count_c1 and max_count_c2.

        Args:
            combination_counts_file_path (str): The file path to the CSV file containing the combination counts.
            pc (str, optional): The positive control sequence. Defaults to 'TGGT1_220950_1'.
            nc (str, optional): The negative control sequence. Defaults to 'TGGT1_233460_4'.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """

        pc = settings['pc']
        nc = settings['nc']
        pc_loc = settings['pc_loc']
        nc_loc = settings['nc_loc']

        filtered_c1 = df[(df['column'] == nc_loc) & (df['grna'] != nc)]
        max_count_c1 = filtered_c1['count'].max()

        filtered_c2 = df[(df['column'] == pc_loc) & (df['grna'] != pc)]
        max_count_c2 = filtered_c2['count'].max()

        #filtered_c3 = df[(df['column'] != nc_loc) & (df['grna'] == nc)]
        #max_count_c3 = filtered_c3['count'].max()

        #filtered_c4 = df[(df['column'] != pc_loc) & (df['grna'] == pc)]
        #max_count_c4 = filtered_c4['count'].max()

        # Find the highest value between max_count_c1 and max_count_c2
        highest_max_count = max(max_count_c1, max_count_c2)

        # Filter the DataFrame to remove rows with a count lower than the highest_max_count
        filtered_df = df[df['count'] >= highest_max_count]

        # Calculate total read counts for each unique combination of plate_row and column
        filtered_df['total_reads'] = filtered_df.groupby(['plate_row', 'column'])['count'].transform('sum')
        
        # Calculate read fraction for each row
        filtered_df['read_fraction'] = filtered_df['count'] / filtered_df['total_reads']

        if settings['verbose']:
            print(f"Max count for non {nc} in {nc_loc}: {max_count_c1}")
            print(f"Max count for non {pc} in {pc_loc}: {max_count_c2}")
            #print(f"Max count for {nc} in other columns: {max_count_c3}")
            
        return filtered_df

    settings.setdefault('grna', '/home/carruthers/Documents/grna_barcodes.csv')
    settings.setdefault('barcodes', '/home/carruthers/Documents/SCREEN_BARCODES.csv')
    settings.setdefault('plate_dict', {'EO1': 'plate1', 'EO2': 'plate2', 'EO3': 'plate3', 'EO4': 'plate4', 'EO5': 'plate5', 'EO6': 'plate6', 'EO7': 'plate7', 'EO8': 'plate8'})
    settings.setdefault('test', False)
    settings.setdefault('verbose', True)

    settings.setdefault('pc', 'TGGT1_220950_1')
    settings.setdefault('pc_loc', 'c2')
    settings.setdefault('nc', 'TGGT1_233460_4')
    settings.setdefault('nc_loc', 'c1')

    fldr = os.path.splitext(h5_file_path)[0]
    file_name = os.path.basename(fldr)

    if settings['test']:
        fldr = os.path.join(fldr, 'test')
    os.makedirs(fldr, exist_ok=True)

    qc_file_path = os.path.join(fldr, f'{file_name}_qc_step_2.csv')
    unique_grna_file_path = os.path.join(fldr, f'{file_name}_unique_grna.csv')
    unique_plate_row_file_path = os.path.join(fldr, f'{file_name}_unique_plate_row.csv')
    unique_column_file_path = os.path.join(fldr, f'{file_name}_unique_column.csv')
    unique_plate_file_path = os.path.join(fldr, f'{file_name}_unique_plate.csv')
    new_h5_file_path = os.path.join(fldr, f'{file_name}_cleaned.h5')
    combination_counts_file_path = os.path.join(fldr, f'{file_name}_combination_counts.csv')
    combination_counts_file_path_cleaned = os.path.join(fldr, f'{file_name}_combination_counts_cleaned.csv')

    #qc_file_path = os.path.splitext(h5_file_path)[0] + '_qc_step_2.csv'
    #unique_grna_file_path = os.path.splitext(h5_file_path)[0] + '_unique_grna.csv'
    #unique_plate_row_file_path = os.path.splitext(h5_file_path)[0] + '_unique_plate_row.csv'
    #unique_column_file_path = os.path.splitext(h5_file_path)[0] + '_unique_column.csv'
    #unique_plate_file_path = os.path.splitext(h5_file_path)[0] + '_unique_plate.csv'
    #new_h5_file_path = os.path.splitext(h5_file_path)[0] + '_cleaned.h5'
    #combination_counts_file_path = os.path.splitext(h5_file_path)[0] + '_combination_counts.csv'
    #combination_counts_file_path_cleaned = os.path.splitext(h5_file_path)[0] + '_combination_counts_cleaned.csv'
    
    # Initialize the HDF5 store for cleaned data
    store_cleaned = pd.HDFStore(new_h5_file_path, mode='a', complevel=5, complib='blosc')
    
    # Initialize the overall QC metrics
    overall_qc = {
        'reads': 0,
        'cleaned_reads': 0,
        'NaN_grna': 0,
        'NaN_plate_row': 0,
        'NaN_column': 0,
        'NaN_plate': 0,
        'unique_grna': Counter(),
        'unique_plate_row': Counter(),
        'unique_column': Counter(),
        'unique_plate': Counter(),
        'pc_total_count': 0,
        'pc_count_pc': 0,
        'nc_total_count': 0,
        'nc_count_nc': 0,
        'pc_count_nc': 0,
        'nc_count_pc': 0,
        'pc_error_count': 0,
        'nc_error_count': 0,
        'pc_fraction_pc': 0,
        'nc_fraction_nc': 0,
        'pc_fraction_nc': 0,
        'nc_fraction_pc': 0
    }

    per_row_qc = {}
    combination_counts = Counter()

    with pd.HDFStore(h5_file_path, mode='r') as store:
        keys = [key for key in store.keys() if key.startswith('/reads/chunk_')]

        if settings['test']:
            keys = keys[:3]  # Only read the first chunks if in test mode

        for key in keys:
            df = store.get(key)
            df = mapping_dicts(df, settings)
            df_cleaned, qc_dict = get_read_qc(df, settings)

            # Accumulate counts for unique combinations
            combinations = df_cleaned[['plate_row_metadata', 'column_metadata', 'grna_metadata']].apply(tuple, axis=1)
            
            combination_counts.update(combinations)

            if settings['test'] and settings['verbose']:
                os.makedirs(os.path.join(os.path.splitext(h5_file_path)[0],'test'), exist_ok=True)
                df.to_csv(os.path.join(os.path.splitext(h5_file_path)[0],'test','chunk_1_df.csv'), index=False)
                df_cleaned.to_csv(os.path.join(os.path.splitext(h5_file_path)[0],'test','chunk_1_df_cleaned.csv'), index=False)

            # Accumulate QC metrics for all rows
            for metric in qc_dict:
                if isinstance(overall_qc[metric], Counter):
                    overall_qc[metric].update(qc_dict[metric])
                else:
                    overall_qc[metric] += qc_dict[metric]

            # Update per_row_qc dictionary
            chunk_per_row_qc = get_per_row_qc(df, settings)
            for row in chunk_per_row_qc:
                if row not in per_row_qc:
                    per_row_qc[row] = chunk_per_row_qc[row]
                else:
                    for metric in chunk_per_row_qc[row]:
                        if isinstance(per_row_qc[row][metric], Counter):
                            per_row_qc[row][metric].update(chunk_per_row_qc[row][metric])
                        else:
                            per_row_qc[row][metric] += chunk_per_row_qc[row][metric]

            # Ensure the DataFrame columns are in the desired order
            df_cleaned = df_cleaned[['grna', 'plate_row', 'column', 'sample', 'grna_metadata', 'plate_row_metadata', 'column_metadata', 'plate_metadata']]

            # Save cleaned data to the new HDF5 store
            store_cleaned.put('reads/cleaned_data', df_cleaned, format='table', append=True)

            del df_cleaned, df
            gc.collect()

    # Calculate overall fractions after accumulating all metrics
    overall_qc['pc_fraction_pc'] = overall_qc['pc_count_pc'] / overall_qc['pc_total_count'] if overall_qc['pc_total_count'] else 0
    overall_qc['nc_fraction_nc'] = overall_qc['nc_count_nc'] / overall_qc['nc_total_count'] if overall_qc['nc_total_count'] else 0
    overall_qc['pc_fraction_nc'] = overall_qc['pc_count_nc'] / overall_qc['nc_total_count'] if overall_qc['nc_total_count'] else 0
    overall_qc['nc_fraction_pc'] = overall_qc['nc_count_pc'] / overall_qc['pc_total_count'] if overall_qc['pc_total_count'] else 0

    for row in per_row_qc:
        if row != 'all_rows':
            per_row_qc[row]['pc_fraction_pc'] = per_row_qc[row]['pc_count_pc'] / per_row_qc[row]['pc_total_count'] if per_row_qc[row]['pc_total_count'] else 0
            per_row_qc[row]['nc_fraction_nc'] = per_row_qc[row]['nc_count_nc'] / per_row_qc[row]['nc_total_count'] if per_row_qc[row]['nc_total_count'] else 0
            per_row_qc[row]['pc_fraction_nc'] = per_row_qc[row]['pc_count_nc'] / per_row_qc[row]['nc_total_count'] if per_row_qc[row]['nc_total_count'] else 0
            per_row_qc[row]['nc_fraction_pc'] = per_row_qc[row]['nc_count_pc'] / per_row_qc[row]['pc_total_count'] if per_row_qc[row]['pc_total_count'] else 0

    # Add overall_qc to per_row_qc with the key 'all_rows'
    per_row_qc['all_rows'] = overall_qc

    # Convert the Counter objects to DataFrames and save them to CSV files
    unique_grna_df = pd.DataFrame(overall_qc['unique_grna'].items(), columns=['key', 'value'])
    unique_plate_row_df = pd.DataFrame(overall_qc['unique_plate_row'].items(), columns=['key', 'value'])
    unique_column_df = pd.DataFrame(overall_qc['unique_column'].items(), columns=['key', 'value'])
    unique_plate_df = pd.DataFrame(overall_qc['unique_plate'].items(), columns=['key', 'value'])

    unique_grna_df.to_csv(unique_grna_file_path, index=False)
    unique_plate_row_df.to_csv(unique_plate_row_file_path, index=False)
    unique_column_df.to_csv(unique_column_file_path, index=False)
    unique_plate_df.to_csv(unique_plate_file_path, index=False)

    # Remove the unique counts from overall_qc for the main QC CSV file
    del overall_qc['unique_grna']
    del overall_qc['unique_plate_row']
    del overall_qc['unique_column']
    del overall_qc['unique_plate']

    # Combine all remaining QC metrics into a single DataFrame and save it to CSV
    qc_df = pd.DataFrame([overall_qc])
    qc_df.to_csv(qc_file_path, index=False)

    # Convert per_row_qc to a DataFrame and save it to CSV
    per_row_qc_df = pd.DataFrame.from_dict(per_row_qc, orient='index')
    per_row_qc_df = per_row_qc_df.sort_values(by='reads', ascending=False)
    per_row_qc_df = per_row_qc_df.drop(['unique_grna', 'unique_plate_row', 'unique_column', 'unique_plate'], axis=1, errors='ignore')
    per_row_qc_df = per_row_qc_df.dropna(subset=['reads'])
    per_row_qc_df.to_csv(os.path.splitext(h5_file_path)[0] + '_per_row_qc.csv', index=True)

    if settings['verbose']:
        display(per_row_qc_df)

    # Save the combination counts to a CSV file
    try:
        combination_counts_df = pd.DataFrame(combination_counts.items(), columns=['combination', 'count'])
        combination_counts_df[['plate_row', 'column', 'grna']] = pd.DataFrame(combination_counts_df['combination'].tolist(), index=combination_counts_df.index)
        combination_counts_df = combination_counts_df.drop('combination', axis=1)
        combination_counts_df.to_csv(combination_counts_file_path, index=False)

        grna_plate_heatmap(combination_counts_file_path, specific_grna=None)
        grna_plate_heatmap(combination_counts_file_path, specific_grna=settings['pc'])
        grna_plate_heatmap(combination_counts_file_path, specific_grna=settings['nc'])

        combination_counts_df_cleaned = filter_combinations(combination_counts_df, settings)
        combination_counts_df_cleaned.to_csv(combination_counts_file_path_cleaned, index=False)

        grna_plate_heatmap(combination_counts_file_path_cleaned, specific_grna=None)
        grna_plate_heatmap(combination_counts_file_path_cleaned, specific_grna=settings['pc'])
        grna_plate_heatmap(combination_counts_file_path_cleaned, specific_grna=settings['nc'])
    except Exception as e:
        print(e)
    
    # Close the HDF5 store
    store_cleaned.close()
    gc.collect()
    return

def grna_plate_heatmap(path, specific_grna=None, min_max='all', cmap='viridis', min_count=0, save=True):
    """
    Generate a heatmap of gRNA plate data.

    Args:
        path (str): The path to the CSV file containing the gRNA plate data.
        specific_grna (str, optional): The specific gRNA to filter the data for. Defaults to None.
        min_max (str or list or tuple, optional): The range of values to use for the color scale. 
            If 'all', the range will be determined by the minimum and maximum values in the data.
            If 'allq', the range will be determined by the 2nd and 98th percentiles of the data.
            If a list or tuple of two values, the range will be determined by those values.
            Defaults to 'all'.
        cmap (str, optional): The colormap to use for the heatmap. Defaults to 'viridis'.
        min_count (int, optional): The minimum count threshold for including a gRNA in the heatmap. 
            Defaults to 0.
        save (bool, optional): Whether to save the heatmap as a PDF file. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure.
    """    
    def generate_grna_plate_heatmap(df, plate_number, min_max, min_count, specific_grna=None):
        df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning
        
        # Filtering the dataframe based on the plate_number and specific gRNA if provided
        df = df[df['plate_row'].str.startswith(plate_number)]
        if specific_grna:
            df = df[df['grna'] == specific_grna]

        # Split plate_row into plate and row
        df[['plate', 'row']] = df['plate_row'].str.split('_', expand=True)

        # Ensure proper ordering
        row_order = [f'r{i}' for i in range(1, 17)]
        col_order = [f'c{i}' for i in range(1, 28)]

        df['row'] = pd.Categorical(df['row'], categories=row_order, ordered=True)
        df['column'] = pd.Categorical(df['column'], categories=col_order, ordered=True)

        # Group by row and column, summing counts
        grouped = df.groupby(['row', 'column'], observed=True)['count'].sum().reset_index()

        plate_map = pd.pivot_table(grouped, values='count', index='row', columns='column').fillna(0)

        if min_max == 'all':
            min_max = [plate_map.min().min(), plate_map.max().max()]
        elif min_max == 'allq':
            min_max = np.quantile(plate_map.values, [0.02, 0.98])
        elif isinstance(min_max, (list, tuple)) and len(min_max) == 2:
            if isinstance(min_max[0], (float)) and isinstance(min_max[1], (float)):
                min_max = np.quantile(plate_map.values, [min_max[0], min_max[1]])
            if isinstance(min_max[0], (int)) and isinstance(min_max[1], (int)): 
                min_max = [min_max[0], min_max[1]]

        return plate_map, min_max
    
    if isinstance(path, pd.DataFrame):
        df = path
    else:
        df = pd.read_csv(path)

    plates = df['plate_row'].str.split('_', expand=True)[0].unique()
    n_rows, n_cols = (len(plates) + 3) // 4, 4
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(40, 5 * n_rows))
    ax = ax.flatten()

    for index, plate in enumerate(plates):
        plate_map, min_max_values = generate_grna_plate_heatmap(df, plate, min_max, min_count, specific_grna)
        sns.heatmap(plate_map, cmap=cmap, vmin=min_max_values[0], vmax=min_max_values[1], ax=ax[index])
        ax[index].set_title(plate)
        
    for i in range(len(plates), n_rows * n_cols):
        fig.delaxes(ax[i])
    
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    
    # Save the figure
    if save:
        filename = path.replace('.csv', '')
        if specific_grna:
            filename += f'_{specific_grna}'
        filename += '.pdf'
        plt.savefig(filename)
        print(f'saved {filename}')
    plt.show()
    
    return fig

def map_barcodes_folder(src, settings={}):
    for file in os.listdir(src):
        if file.endswith('.h5'):
            print(file)
            path = os.path.join(src, file)
            map_barcodes(path, settings)
            gc.collect() 

def reverse_complement(dna_sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    reverse_seq = dna_sequence[::-1]
    reverse_complement_seq = ''.join([complement_dict[base] for base in reverse_seq])
    return reverse_complement_seq

def complement(dna_sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    complement_seq = ''.join([complement_dict[base] for base in dna_sequence])
    return complement_seq

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def generate_plate_heatmap(df, plate_number, variable, grouping, min_max):
    if grouping == 'mean':
        temp = df.groupby(['plate','row','col']).mean()[variable]
    if grouping == 'sum':
        temp = df.groupby(['plate','row','col']).sum()[variable]
    if grouping == 'count':
        temp = df.groupby(['plate','row','col']).count()[variable]
    if grouping in ['mean', 'count', 'sum']:
        temp = pd.DataFrame(temp)
    if min_max == 'all':  
        min_max=[np.min(temp[variable]),np.max(temp[variable])]   
    if min_max == 'allq':
        min_max = np.quantile(temp[variable], [0.2, 0.98])
    plate = df[df['plate'] == plate_number]
    plate = pd.DataFrame(plate)
    if grouping == 'mean':
        plate = plate.groupby(['plate','row','col']).mean()[variable]
    if grouping == 'sum':
        plate = plate.groupby(['plate','row','col']).sum()[variable]
    if grouping == 'count':
        plate = plate.groupby(['plate','row','col']).count()[variable]
    if grouping not in ['mean', 'count', 'sum']:
        plate = plate.groupby(['plate','row','col']).mean()[variable]
    if min_max == 'plate':
        min_max=[np.min(plate[variable]),np.max(plate[variable])]
    plate = pd.DataFrame(plate)
    plate = plate.reset_index()
    if 'plate' in plate.columns:
        plate = plate.drop(['plate'], axis=1)
    pcol = [*range(1,28,1)]
    prow = [*range(1,17,1)]
    new_col = []
    for v in pcol:
        col = 'c'+str(v)
        new_col.append(col)
    new_col.remove('c15')
    new_row = []
    for v in prow:
        ro = 'r'+str(v)
        new_row.append(ro)
    plate_map = pd.DataFrame(columns=new_col, index = new_row)
    for index, row in plate.iterrows():
        r = row['row']
        c = row['col']
        v = row[variable]
        plate_map.loc[r,c]=v
    plate_map = plate_map.fillna(0)
    return pd.DataFrame(plate_map), min_max

def plot_plates(df, variable, grouping, min_max, cmap):
    try:
        plates = np.unique(df['plate'], return_counts=False)
    except:
        try:
            df[['plate', 'row', 'col']] = df['prc'].str.split('_', expand=True)
            df = pd.DataFrame(df)
            plates = np.unique(df['plate'], return_counts=False)
        except:
            next
    #plates = np.unique(df['plate'], return_counts=False)
    nr_of_plates = len(plates)
    print('nr_of_plates:',nr_of_plates)
    # Calculate the number of rows and columns for the subplot grid
    if nr_of_plates in [1, 2, 3, 4]:
        n_rows, n_cols = 1, 4
    elif nr_of_plates in [5, 6, 7, 8]:
        n_rows, n_cols = 2, 4
    elif nr_of_plates in [9, 10, 11, 12]:
        n_rows, n_cols = 3, 4
    elif nr_of_plates in [13, 14, 15, 16]:
        n_rows, n_cols = 4, 4

    # Create the subplot grid with the specified number of rows and columns
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(40, 5 * n_rows))

    # Flatten the axes array to a one-dimensional array
    ax = ax.flatten()

    # Loop over each plate and plot the heatmap
    for index, plate in enumerate(plates):
        plate_number = plate
        plate_map, min_max = generate_plate_heatmap(df=df, plate_number=plate_number, variable=variable, grouping=grouping, min_max=min_max)
        if index == 0:
            print('plate_number:',plate_number,'minimum:',min_max[0], 'maximum:',min_max[1])
        # Plot the heatmap on the appropriate subplot
        sns.heatmap(plate_map, cmap=cmap, vmin=min_max[0], vmax=min_max[1], ax=ax[index])
        ax[index].set_title(plate_number)

    # Remove any empty subplots
    for i in range(nr_of_plates, n_rows * n_cols):
        fig.delaxes(ax[i])

    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    # Show the plot
    plt.show()
    print()
    return

def count_mismatches(seq1, seq2, align_length=10):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    # choose the first alignment (there might be several with the same score)
    alignment = alignments[0]
    # alignment is a tuple (seq1_aligned, seq2_aligned, score, begin, end)
    seq1_aligned, seq2_aligned, score, begin, end = alignment
    # Determine the start of alignment (first position where at least align_length bases are the same)
    start_of_alignment = next(i for i in range(len(seq1_aligned) - align_length + 1) 
                              if seq1_aligned[i:i+align_length] == seq2_aligned[i:i+align_length])
    # Trim the sequences to the same length from the start of the alignment
    seq1_aligned = seq1_aligned[start_of_alignment:]
    seq2_aligned = seq2_aligned[start_of_alignment:]
    # Trim the sequences to be of the same length (from the end)
    min_length = min(len(seq1_aligned), len(seq2_aligned))
    seq1_aligned = seq1_aligned[:min_length]
    seq2_aligned = seq2_aligned[:min_length]
    mismatches = sum(c1 != c2 for c1, c2 in zip(seq1_aligned, seq2_aligned))
    return mismatches
    
def get_sequence_data(r1,r2):
    forward_regex = re.compile(r'^(...GGTGCCACTT)TTTCAAGTTG.*?TTCTAGCTCT(AAAAC[A-Z]{18,22}AACTT)GACATCCCCA.*?AAGGCAAACA(CCCCCTTCGG....).*') 
    r1fd = forward_regex.search(r1)
    reverce_regex = re.compile(r'^(...CCGAAGGGGG)TGTTTGCCTT.*?TGGGGATGTC(AAGTT[A-Z]{18,22}GTTTT)AGAGCTAGAA.*?CAACTTGAAA(AAGTGGCACC...).*') 
    r2fd = reverce_regex.search(r2)
    rc_r1 = reverse_complement(r1)
    rc_r2 = reverse_complement(r2) 
    if all(var is not None for var in [r1fd, r2fd]):
        try:
            r1_mis_matches, _ = count_mismatches(seq1=r1, seq2=rc_r2, align_length=5)
            r2_mis_matches, _ = count_mismatches(seq1=r2, seq2=rc_r1, align_length=5)
        except:
            r1_mis_matches = None
            r2_mis_matches = None
        column_r1 = reverse_complement(r1fd[1])
        sgrna_r1 = r1fd[2]
        platerow_r1 = r1fd[3]
        column_r2 = r2fd[3]
        sgrna_r2 = reverse_complement(r2fd[2])
        platerow_r2 = reverse_complement(r2fd[1])+'N'

        data_dict = {'r1_plate_row':platerow_r1,
                     'r1_col':column_r1,
                     'r1_gRNA':sgrna_r1,
                     'r1_read':r1,
                     'r2_plate_row':platerow_r2,
                     'r2_col':column_r2,
                     'r2_gRNA':sgrna_r2,
                     'r2_read':r2,
                     'r1_r2_rc_mismatch':r1_mis_matches,
                     'r2_r1_rc_mismatch':r2_mis_matches,
                     'r1_len':len(r1),
                     'r2_len':len(r2)}
    else:
        try:
            r1_mis_matches, _ = count_mismatches(r1, rc_r2, align_length=5)
            r2_mis_matches, _ = count_mismatches(r2, rc_r1, align_length=5)
        except:
            r1_mis_matches = None
            r2_mis_matches = None
        data_dict = {'r1_plate_row':None,
             'r1_col':None,
             'r1_gRNA':None,
             'r1_read':r1,
             'r2_plate_row':None,
             'r2_col':None,
             'r2_gRNA':None,
             'r2_read':r2,
             'r1_r2_rc_mismatch':r1_mis_matches,
             'r2_r1_rc_mismatch':r2_mis_matches,
             'r1_len':len(r1),
             'r2_len':len(r2)}

    return data_dict

def get_read_data(identifier, prefix):
    if identifier.startswith("@"):
        parts = identifier.split(" ")
        # The first part contains the instrument, run number, flowcell ID, lane, tile, and coordinates
        instrument, run_number, flowcell_id, lane, tile, x_pos, y_pos = parts[0][1:].split(":")
        # The second part contains the read number, filter status, control number, and sample number
        read, is_filtered, control_number, sample_number = parts[1].split(":")
        rund_data_dict = {'instrument':instrument, 
                          'run_number':run_number, 
                          'flowcell_id':flowcell_id, 
                          'lane':lane, 
                          'tile':tile, 
                          'x_pos':x_pos, 
                          'y_pos':y_pos, 
                          'read':read, 
                          'is_filtered':is_filtered, 
                          'control_number':control_number, 
                          'sample_number':sample_number}
        modified_dict = {prefix + key: value for key, value in rund_data_dict.items()}
    return modified_dict

def pos_dict(string):
    pos_dict = {}
    for i, char in enumerate(string):
        if char not in pos_dict:
            pos_dict[char] = [i]
        else:
            pos_dict[char].append(i)
    return pos_dict

def truncate_read(seq,qual,target):
    index = seq.find(target)
    end = len(seq)-(3+len(target))
    if index != -1: # If the sequence is found
        if index-3 >= 0:
            seq = seq[index-3:]
            qual = qual[index-3:]

    return seq, qual

def equalize_lengths(seq1, seq2, pad_char='N'):
    len_diff = len(seq1) - len(seq2)

    if len_diff > 0:  # seq1 is longer
        seq2 += pad_char * len_diff  # pad seq2 with 'N's
    elif len_diff < 0:  # seq2 is longer
        seq1 += pad_char * (-len_diff)  # pad seq1 with 'N's

    return seq1, seq2

def get_read_data(identifier, prefix):
    if identifier.startswith("@"):
        parts = identifier.split(" ")
        # The first part contains the instrument, run number, flowcell ID, lane, tile, and coordinates
        instrument, run_number, flowcell_id, lane, tile, x_pos, y_pos = parts[0][1:].split(":")
        # The second part contains the read number, filter status, control number, and sample number
        read, is_filtered, control_number, sample_number = parts[1].split(":")
        rund_data_dict = {'instrument':instrument, 
                          'x_pos':x_pos, 
                          'y_pos':y_pos}
        modified_dict = {prefix + key: value for key, value in rund_data_dict.items()}
    return modified_dict

def extract_barecodes(r1_fastq, r2_fastq, csv_loc, chunk_size=100000):
    data_chunk = []
    # Open both FASTQ files.
    with open(r1_fastq) as r1_file, open(r2_fastq) as r2_file:
        index = 0
        save_index = 0
        while True:
            index += 1
            start = time.time()
            # Read 4 lines at a time
            r1_identifier = r1_file.readline().strip()
            r1_sequence = r1_file.readline().strip()
            r1_plus = r1_file.readline().strip()
            r1_quality = r1_file.readline().strip()
            r2_identifier = r2_file.readline().strip()
            r2_sequence = r2_file.readline().strip()
            r2_sequence = reverse_complement(r2_sequence)
            r2_sequence = r2_sequence
            r2_plus = r2_file.readline().strip()
            r2_quality = r2_file.readline().strip()
            r2_quality = r2_quality
            if not r1_identifier or not r2_identifier:
                break
            #if index > 100:
            #    break
            target = 'GGTGCCACTT'
            r1_sequence, r1_quality = truncate_read(r1_sequence, r1_quality, target)
            r2_sequence, r2_quality = truncate_read(r2_sequence, r2_quality, target)
            r1_sequence, r2_sequence = equalize_lengths(r1_sequence, r2_sequence, pad_char='N')
            r1_quality, r2_quality = equalize_lengths(r1_quality, r2_quality, pad_char='-')
            alignments = pairwise2.align.globalxx(r1_sequence, r2_sequence)
            alignment = alignments[0]
            score = alignment[2]
            column = None
            platerow = None
            grna = None
            if score >= 125:
                aligned_r1 = alignment[0]
                aligned_r2 = alignment[1]
                position_dict = {i+1: (base1, base2) for i, (base1, base2) in enumerate(zip(aligned_r1, aligned_r2))}
                phred_quality1 = [ord(char) - 33 for char in r1_quality]
                phred_quality2 = [ord(char) - 33 for char in r2_quality]
                r1_q_dict = {i+1: quality for i, quality in enumerate(phred_quality1)}
                r2_q_dict = {i+1: quality for i, quality in enumerate(phred_quality2)}
                read = ''
                for key in sorted(position_dict.keys()):
                    if position_dict[key][0] != '-' and (position_dict[key][1] == '-' or r1_q_dict.get(key, 0) >= r2_q_dict.get(key, 0)):
                        read = read + position_dict[key][0]
                    elif position_dict[key][1] != '-' and (position_dict[key][0] == '-' or r2_q_dict.get(key, 0) > r1_q_dict.get(key, 0)):
                        read = read + position_dict[key][1]
                pattern = re.compile(r'^(...GGTGC)CACTT.*GCTCT(TAAAC[A-Z]{18,22}AACTT)GACAT.*CCCCC(TTCGG....).*')
                regex_patterns = pattern.search(read)
                if all(var is not None for var in [regex_patterns]):
                    column = regex_patterns[1]
                    grna = reverse_complement(regex_patterns[2])
                    platerow = reverse_complement(regex_patterns[3])
            elif score < 125:
                read = r1_sequence
                pattern = re.compile(r'^(...GGTGC)CACTT.*GCTCT(TAAAC[A-Z]{18,22}AACTT)GACAT.*CCCCC(TTCGG....).*')
                regex_patterns = pattern.search(read)
                if all(var is not None for var in [regex_patterns]):
                    column = regex_patterns[1]
                    grna = reverse_complement(regex_patterns[2])
                    platerow = reverse_complement(regex_patterns[3])
                    #print('2', platerow)
            data_dict = {'read':read,'column':column,'platerow':platerow,'grna':grna, 'score':score}
            end = time.time()
            if data_dict.get('grna') is not None:
                save_index += 1
                r1_rund_data_dict = get_read_data(r1_identifier, prefix='r1_')
                r2_rund_data_dict = get_read_data(r2_identifier, prefix='r2_')
                r1_rund_data_dict.update(r2_rund_data_dict)
                r1_rund_data_dict.update(data_dict)
                r1_rund_data_dict['r1_quality'] = r1_quality
                r1_rund_data_dict['r2_quality'] = r2_quality
                data_chunk.append(r1_rund_data_dict)
                print(f'Processed reads: {index} Found barecodes in {save_index} Time/read: {end - start}', end='\r', flush=True)
                if save_index % chunk_size == 0:  # Every `chunk_size` reads, write to the CSV
                    if not os.path.isfile(csv_loc):
                        df = pd.DataFrame(data_chunk)
                        df.to_csv(csv_loc, index=False)
                    else:
                        df = pd.DataFrame(data_chunk)
                        df.to_csv(csv_loc, mode='a', header=False, index=False)
                    data_chunk = []  # Clear the chunk
                    
def split_fastq(input_fastq, output_base, num_files):
    # Create file objects for each output file
    outputs = [open(f"{output_base}_{i}.fastq", "w") for i in range(num_files)]
    with open(input_fastq, "r") as f:
        # Initialize a counter for the lines
        line_counter = 0
        for line in f:
            # Determine the output file
            output_file = outputs[line_counter // 4 % num_files]
            # Write the line to the appropriate output file
            output_file.write(line)
            # Increment the line counter
            line_counter += 1
    # Close output files
    for output in outputs:
        output.close()

def process_barecodes(df):
    print('==== Preprocessing barecodes ====')
    plate_ls = []
    row_ls = [] 
    column_ls = []
    grna_ls = []
    read_ls = []
    score_ls = []
    match_score_ls = []
    index_ls = []
    index = 0
    print_every = 100
    for i,row in df.iterrows():
        index += 1
        r1_instrument=row['r1_instrument']
        r1_x_pos=row['r1_x_pos']
        r1_y_pos=row['r1_y_pos']
        r2_instrument=row['r2_instrument']
        r2_x_pos=row['r2_x_pos']
        r2_y_pos=row['r2_y_pos']
        read=row['read']
        column=row['column']
        platerow=row['platerow']
        grna=row['grna']
        score=row['score']
        r1_quality=row['r1_quality']
        r2_quality=row['r2_quality']
        if r1_x_pos == r2_x_pos:
            if r1_y_pos == r2_y_pos:
                match_score = 0
                
                if grna.startswith('AAGTT'):
                    match_score += 0.5
                if column.endswith('GGTGC'):
                    match_score += 0.5
                if platerow.endswith('CCGAA'):
                    match_score += 0.5
                index_ls.append(index)
                match_score_ls.append(match_score)
                score_ls.append(score)
                read_ls.append(read)
                plate_ls.append(platerow[:2])
                row_ls.append(platerow[2:4])
                column_ls.append(column[:3])
                grna_ls.append(grna)
                if index % print_every == 0:
                    print(f'Processed reads: {index}', end='\r', flush=True)
    df = pd.DataFrame()
    df['index'] = index_ls
    df['score'] = score_ls
    df['match_score'] = match_score_ls
    df['plate'] = plate_ls
    df['row'] = row_ls
    df['col'] = column_ls
    df['seq'] = grna_ls
    df_high_score = df[df['score']>=125]
    df_low_score = df[df['score']<125]
    print(f'', flush=True)
    print(f'Found {len(df_high_score)} high score reads;Found {len(df_low_score)} low score reads')
    return df, df_high_score, df_low_score

def find_grna(df, grna_df):
    print('==== Finding gRNAs ====')
    seqs = list(set(df.seq.tolist()))
    seq_ls = []
    grna_ls = []
    index = 0
    print_every = 1000
    for grna in grna_df.Seq.tolist():
        reverse_regex = re.compile(r'.*({}).*'.format(grna))
        for seq in seqs:
            index += 1
            if index % print_every == 0:
                print(f'Processed reads: {index}', end='\r', flush=True)
            found_grna = reverse_regex.search(seq)
            if found_grna is None:
                seq_ls.append('error')
                grna_ls.append('error')
            else:
                seq_ls.append(found_grna[0])
                grna_ls.append(found_grna[1])
    grna_dict = dict(zip(seq_ls, grna_ls))
    df = df.assign(grna_seq=df['seq'].map(grna_dict).fillna('error'))
    print(f'', flush=True)
    return df

def map_unmapped_grnas(df):
    print('==== Mapping lost gRNA barecodes ====')
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()
    index = 0
    print_every = 100
    sequence_list = df[df['grna_seq'] != 'error']['seq'].unique().tolist()
    grna_error = df[df['grna_seq']=='error']
    df = grna_error.copy()
    similarity_dict = {}
    #change this so that it itterates throug each well
    for idx, row in df.iterrows():
        matches = 0
        match_string = None
        for string in sequence_list:
            index += 1
            if index % print_every == 0:
                print(f'Processed reads: {index}', end='\r', flush=True)
            ratio = similar(row['seq'], string)
            # check if only one character is different
            if ratio > ((len(row['seq']) - 1) / len(row['seq'])):
                matches += 1
                if matches > 1: # if we find more than one match, we break and don't add anything to the dictionary
                    break
                match_string = string
        if matches == 1: # only add to the dictionary if there was exactly one match
            similarity_dict[row['seq']] = match_string
    return similarity_dict

def translate_barecodes(df, grna_df, map_unmapped=False):
    print('==== Translating barecodes ====')
    if map_unmapped:
        similarity_dict = map_unmapped_grnas(df)
        df = df.assign(seq=df['seq'].map(similarity_dict).fillna('error'))
    df = df.groupby(['plate','row', 'col'])['grna_seq'].value_counts().reset_index(name='count')
    grna_dict = grna_df.set_index('Seq')['gene'].to_dict()
    
    plate_barcodes = {'AA':'p1','TT':'p2','CC':'p3','GG':'p4','AT':'p5','TA':'p6','CG':'p7','GC':'p8'}
    
    row_barcodes = {'AA':'r1','AT':'r2','AC':'r3','AG':'r4','TT':'r5','TA':'r6','TC':'r7','TG':'r8',
                    'CC':'r9','CA':'r10','CT':'r11','CG':'r12','GG':'r13','GA':'r14','GT':'r15','GC':'r16'}
    
    col_barcodes = {'AAA':'c1','TTT':'c2','CCC':'c3','GGG':'c4','AAT':'c5','AAC':'c6','AAG':'c7',
                    'TTA':'c8','TTC':'c9','TTG':'c10','CCA':'c11','CCT':'c12','CCG':'c13','GGA':'c14',
                    'CCT':'c15','GGC':'c16','ATT':'c17','ACC':'c18','AGG':'c19','TAA':'c20','TCC':'c21',
                    'TGG':'c22','CAA':'c23','CGG':'c24'}

    
    df['plate'] = df['plate'].map(plate_barcodes)
    df['row'] = df['row'].map(row_barcodes)
    df['col'] = df['col'].map(col_barcodes)
    df['grna'] = df['grna_seq'].map(grna_dict)
    df['gene'] = df['grna'].str.split('_').str[1]
    df = df.fillna('error')
    df['prc'] = df['plate']+'_'+df['row']+'_'+df['col']
    df = df[df['count']>=2]
    error_count = df[df.apply(lambda row: row.astype(str).str.contains('error').any(), axis=1)].shape[0]
    plate_error = df['plate'].str.contains('error').sum()/len(df)
    row_error = df['row'].str.contains('error').sum()/len(df)
    col_error = df['col'].str.contains('error').sum()/len(df)
    grna_error = df['grna'].str.contains('error').sum()/len(df)
    print(f'Matched: {len(df)} rows; Errors: plate:{plate_error*100:.3f}% row:{row_error*100:.3f}% column:{col_error*100:.3f}% gRNA:{grna_error*100:.3f}%')
    return df

def vert_horiz(v, h, n_col):
    h = h+1
    if h not in [*range(0,n_col)]:
        v = v+1
        h = 0
    return v,h
                                            
def plot_data(df, v, h, color, n_col, ax, x_axis, y_axis, fontsize=12, lw=2, ls='-', log_x=False, log_y=False, title=None):
    ax[v, h].plot(df[x_axis], df[y_axis], ls=ls, lw=lw, color=color, label=y_axis)
    ax[v, h].set_title(None)
    ax[v, h].set_xlabel(None)
    ax[v, h].set_ylabel(None)
    ax[v, h].legend(fontsize=fontsize)
    
    if log_x:
        ax[v, h].set_xscale('log')
    if log_y:
        ax[v, h].set_yscale('log')
    v,h =vert_horiz(v, h, n_col)
    return v, h  

def test_error(df, min_=25,max_=3025, metric='count',log_x=False, log_y=False):
    max_ = max_+min_
    step = math.sqrt(min_)
    plate_error_ls = []
    col_error_ls = []
    row_error_ls = []
    grna_error_ls = []
    prc_error_ls = []
    total_error_ls = []
    temp_len_ls = []
    val_ls = []
    df['sum_count'] = df.groupby('prc')['count'].transform('sum')
    df['fraction'] = df['count'] / df['sum_count']
    if metric=='fraction':
        range_ = np.arange(min_, max_, step).tolist()
    if metric=='count':
        range_ = [*range(int(min_),int(max_),int(step))]
    for val in range_:
        temp = pd.DataFrame(df[df[metric]>val])
        temp_len = len(temp)
        if temp_len == 0:
            break
        temp_len_ls.append(temp_len)
        error_count = temp[temp.apply(lambda row: row.astype(str).str.contains('error').any(), axis=1)].shape[0]/len(temp)
        plate_error = temp['plate'].str.contains('error').sum()/temp_len
        row_error = temp['row'].str.contains('error').sum()/temp_len
        col_error = temp['col'].str.contains('error').sum()/temp_len
        prc_error = temp['prc'].str.contains('error').sum()/temp_len
        grna_error = temp['gene'].str.contains('error').sum()/temp_len
        #print(error_count, plate_error, row_error, col_error, prc_error, grna_error)
        val_ls.append(val)
        total_error_ls.append(error_count)
        plate_error_ls.append(plate_error)
        row_error_ls.append(row_error)
        col_error_ls.append(col_error)
        prc_error_ls.append(prc_error)
        grna_error_ls.append(grna_error)
    df2 = pd.DataFrame()
    df2['val'] = val_ls
    df2['plate'] = plate_error_ls
    df2['row'] = row_error_ls
    df2['col'] = col_error_ls
    df2['gRNA'] = grna_error_ls
    df2['prc'] = prc_error_ls
    df2['total'] = total_error_ls
    df2['len'] = temp_len_ls
                                 
    n_row, n_col = 2, 7
    v, h, lw, ls, color = 0, 0, 1, '-', 'teal'
    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*5))
    
    v, h = plot_data(df=df2, v=v, h=h, color=color, n_col=n_col, ax=ax, x_axis='val', y_axis='total',log_x=log_x, log_y=log_y)
    v, h = plot_data(df=df2, v=v, h=h, color=color, n_col=n_col, ax=ax, x_axis='val', y_axis='prc',log_x=log_x, log_y=log_y)
    v, h = plot_data(df=df2, v=v, h=h, color=color, n_col=n_col, ax=ax, x_axis='val', y_axis='plate',log_x=log_x, log_y=log_y)
    v, h = plot_data(df=df2, v=v, h=h, color=color, n_col=n_col, ax=ax, x_axis='val', y_axis='row',log_x=log_x, log_y=log_y)
    v, h = plot_data(df=df2, v=v, h=h, color=color, n_col=n_col, ax=ax, x_axis='val', y_axis='col',log_x=log_x, log_y=log_y)
    v, h = plot_data(df=df2, v=v, h=h, color=color, n_col=n_col, ax=ax, x_axis='val', y_axis='gRNA',log_x=log_x, log_y=log_y)
    v, h = plot_data(df=df2, v=v, h=h, color=color, n_col=n_col, ax=ax, x_axis='val', y_axis='len',log_x=log_x, log_y=log_y)
    
def generate_fraction_map(df, gene_column, min_=10, plates=['p1','p2','p3','p4'], metric = 'count', plot=False):
    df['prcs'] = df['prc']+''+df['grna_seq']
    df['gene'] = df['grna'].str.split('_').str[1]
    if metric == 'count':
        df = pd.DataFrame(df[df['count']>min_])
    df = df[~(df == 'error').any(axis=1)]
    df = df[df['plate'].isin(plates)]
    gRNA_well_count = df.groupby('prc')['prcs'].transform('nunique')
    df['gRNA_well_count'] = gRNA_well_count
    df = df[df['gRNA_well_count']>=2]
    df = df[df['gRNA_well_count']<=100]
    well_sum = df.groupby('prc')['count'].transform('sum')
    df['well_sum'] = well_sum
    df['gRNA_fraction'] = df['count']/df['well_sum']
    if metric == 'fraction':
        df = pd.DataFrame(df[df['gRNA_fraction']>=min_])
        df = df[df['plate'].isin(plates)]
        gRNA_well_count = df.groupby('prc')['prcs'].transform('nunique')
        df['gRNA_well_count'] = gRNA_well_count
        well_sum = df.groupby('prc')['count'].transform('sum')
        df['well_sum'] = well_sum
        df['gRNA_fraction'] = df['count']/df['well_sum']
    if plot:
        print('gRNAs/well')
        plot_plates(df=df, variable='gRNA_well_count', grouping='mean', min_max='allq', cmap='viridis')
        print('well read sum')
        plot_plates(df=df, variable='well_sum', grouping='mean', min_max='allq', cmap='viridis')
    genes = df[gene_column].unique().tolist()
    wells = df['prc'].unique().tolist()
    print('numer of genes:',len(genes),'numer of wells:', len(wells))
    independent_variables = pd.DataFrame(columns=genes, index = wells)
    for index, row in df.iterrows():
        prc = row['prc']
        gene = row[gene_column]
        fraction = row['gRNA_fraction']
        independent_variables.loc[prc,gene]=fraction
    independent_variables = independent_variables.fillna(0.0)
    independent_variables['sum'] = independent_variables.sum(axis=1)
    independent_variables = independent_variables[independent_variables['sum']==1.0]
    independent_variables = independent_variables.drop('sum', axis=1)
    independent_variables.index.name = 'prc'
    independent_variables = independent_variables.loc[:, (independent_variables.sum() != 0)]
    return independent_variables


def plot_histogram(df, dependent_variable):
    # Plot histogram of the dependent variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df[dependent_variable], kde=True)
    plt.title(f'Histogram of {dependent_variable}')
    plt.xlabel(dependent_variable)
    plt.ylabel('Frequency')
    plt.show()
    
def precess_reads(csv_path, fraction_threshold, plate):
    # Read the CSV file into a DataFrame
    csv_df = pd.read_csv(csv_path)

    # Ensure the necessary columns are present
    if not all(col in csv_df.columns for col in ['grna', 'count', 'column']):
        raise ValueError("The CSV file must contain 'grna', 'count', 'plate_row', and 'column' columns.")

    if 'plate_row' in csv_df.columns:
        csv_df[['plate', 'row']] = csv_df['plate_row'].str.split('_', expand=True)
        if plate is not None:
            csv_df = csv_df.drop(columns=['plate'])
            csv_df['plate'] = plate

    if plate is not None:
        csv_df['plate'] = plate

    # Create the prc column
    csv_df['prc'] = csv_df['plate'] + '_' + csv_df['row'] + '_' + csv_df['column']

    # Group by prc and calculate the sum of counts
    grouped_df = csv_df.groupby('prc')['count'].sum().reset_index()
    grouped_df = grouped_df.rename(columns={'count': 'total_counts'})
    merged_df = pd.merge(csv_df, grouped_df, on='prc')
    merged_df['fraction'] = merged_df['count'] / merged_df['total_counts']

    # Filter rows with fraction under the threshold
    if fraction_threshold is not None:
        observations_before = len(merged_df)
        merged_df = merged_df[merged_df['fraction'] >= fraction_threshold]
        observations_after = len(merged_df)
        removed = observations_before - observations_after
        print(f'Removed {removed} observation below fraction threshold: {fraction_threshold}')

    merged_df = merged_df[['prc', 'grna', 'fraction']]

    if not all(col in merged_df.columns for col in ['grna', 'gene']):
        try:
            merged_df[['org', 'gene', 'grna']] = merged_df['grna'].str.split('_', expand=True)
            merged_df = merged_df.drop(columns=['org'])
            merged_df['grna'] = merged_df['gene'] + '_' + merged_df['grna']
        except:
            print('Error splitting grna into org, gene, grna.')

    return merged_df

def apply_transformation(X, transform):
    if transform == 'log':
        transformer = FunctionTransformer(np.log1p, validate=True)
    elif transform == 'sqrt':
        transformer = FunctionTransformer(np.sqrt, validate=True)
    elif transform == 'square':
        transformer = FunctionTransformer(np.square, validate=True)
    else:
        transformer = None
    return transformer

def check_normality(data, variable_name, verbose=False):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    stat, p_value = shapiro(data)
    if verbose:
        print(f"Shapiro-Wilk Test for {variable_name}:\nStatistic: {stat}, P-value: {p_value}")
    if p_value > 0.05:
        if verbose:
            print(f"The data for {variable_name} is normally distributed.")
        return True
    else:
        if verbose:
            print(f"The data for {variable_name} is not normally distributed.")
        return False

def process_scores(df, dependent_variable, plate, min_cell_count=25, agg_type='mean', transform=None):
    
    if plate is not None:
        df['plate'] = plate

    df['prc'] = df['plate'] + '_' + df['row'] + '_' + df['col']
    df = df[['prc', dependent_variable]]

    # Group by prc and calculate the mean and count of the dependent_variable
    grouped = df.groupby('prc')[dependent_variable]

    print(f'Using agg_type: {agg_type}')
    if agg_type == 'median':
        dependent_df = grouped.median().reset_index()
    elif agg_type == 'mean':
        dependent_df = grouped.mean().reset_index()
    elif agg_type == 'quantile':
        dependent_df = grouped.quantile(0.75).reset_index()
    elif agg_type == None:
        dependent_df = df.reset_index()
        if 'prcfo' in dependent_df.columns:
            dependent_df = dependent_df.drop(columns=['prcfo'])

    else:
        raise ValueError(f"Unsupported aggregation type {agg_type}")

    # Calculate cell_count for all cases
    cell_count = grouped.size().reset_index(name='cell_count')

    if agg_type is None:
        dependent_df = pd.merge(dependent_df, cell_count, on='prc')
    else:
        dependent_df['cell_count'] = cell_count['cell_count']

    dependent_df = dependent_df[dependent_df['cell_count'] >= min_cell_count]

    is_normal = check_normality(dependent_df[dependent_variable], dependent_variable)

    if not transform is None:
        transformer = apply_transformation(dependent_df[dependent_variable], transform=transform)
        transformed_var = f'{transform}_{dependent_variable}'
        df[transformed_var] = transformer.fit_transform(dependent_df[[dependent_variable]])
        dependent_variable = transformed_var
        is_normal = check_normality(dependent_df[transformed_var], transformed_var)

    if not is_normal:
        print(f'{dependent_variable} is not normally distributed')
    else:
        print(f'{dependent_variable} is normally distributed')

    return dependent_df, dependent_variable
    
def perform_mixed_model(y, X, groups, alpha=1.0):
    # Ensure groups are defined correctly and check for multicollinearity
    if groups is None:
        raise ValueError("Groups must be defined for mixed model regression")

    # Check for multicollinearity by calculating the VIF for each feature
    X_np = X.values
    vif = [variance_inflation_factor(X_np, i) for i in range(X_np.shape[1])]
    print(f"VIF: {vif}")
    if any(v > 10 for v in vif):
        print(f"Multicollinearity detected with VIF: {vif}. Applying Ridge regression to the fixed effects.")
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        X_ridge = ridge.coef_ * X  # Adjust X with Ridge coefficients
        model = MixedLM(y, X_ridge, groups=groups)
    else:
        model = MixedLM(y, X, groups=groups)

    result = model.fit()
    return result

def regression_model(X, y, regression_type='ols', groups=None, alpha=1.0, remove_row_column_effect=True):

    if regression_type == 'ols':
        model = sm.OLS(y, X).fit()
        
    elif regression_type == 'gls':
        model = sm.GLS(y, X).fit()

    elif regression_type == 'wls':
        model = sm.WLS(y, X, weights=weights).fit()

    elif regression_type == 'rlm':
        model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        #model = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight()).fit()
        #model = sm.RLM(y, X, M=sm.robust.norms.Hampel()).fit()
        #model = sm.RLM(y, X, M=sm.robust.norms.LeastSquares()).fit()
        #model = sm.RLM(y, X, M=sm.robust.norms.RamsayE()).fit()
        #model = sm.RLM(y, X, M=sm.robust.norms.TrimmedMean()).fit()

    elif regression_type == 'glm':
        model = sm.GLM(y, X, family=sm.families.Gaussian()).fit() # Gaussian: Used for continuous data, similar to OLS regression.
        #model = sm.GLM(y, X, family=sm.families.Binomial()).fit() # Binomial: Used for binary data, modeling the probability of success.
        #model = sm.GLM(y, X, family=sm.families.Poisson()).fit() # Poisson: Used for count data.
        #model = sm.GLM(y, X, family=sm.families.Gamma()).fit() # Gamma: Used for continuous, positive data, often for modeling waiting times or life data.
        #model = sm.GLM(y, X, family=sm.families.InverseGaussian()).fit() # Inverse Gaussian: Used for positive continuous data with a variance that increases with the 
        #model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit() # Negative Binomial: Used for count data with overdispersion (variance greater than the mean).
        #model = sm.GLM(y, X, family=sm.families.Tweedie()).fit() # Tweedie: Used for data that can take both positive continuous and count values, allowing for a mixture of distributions.

    elif regression_type == 'mixed':
        model = perform_mixed_model(y, X, groups, alpha=alpha)

    elif regression_type == 'quantile':
        model = sm.QuantReg(y, X).fit(q=alpha)

    elif regression_type == 'logit':
        model = sm.Logit(y, X).fit()

    elif regression_type == 'probit':
        model = sm.Probit(y, X).fit()

    elif regression_type == 'poisson':
        model_poisson = sm.Poisson(y, X).fit()

    elif regression_type == 'lasso':
        model = Lasso(alpha=alpha).fit(X, y)

    elif regression_type == 'ridge':
        model = Ridge(alpha=alpha).fit(X, y)

    else:
        raise ValueError(f"Unsupported regression type {regression_type}")

    if regression_type in ['lasso', 'ridge']:
        y_pred = model.predict(X)
        plt.scatter(X.iloc[:, 1], y, color='blue', label='Data')
        plt.plot(X.iloc[:, 1], y_pred, color='red', label='Regression line')
        plt.xlabel('Features')
        plt.ylabel('Dependent Variable')
        plt.legend()
        plt.show()

    return model
    
def volcano_plot(coef_df, filename='volcano_plot.pdf'):
    # Create the volcano plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=coef_df, 
        x='coefficient', 
        y='-log10(p_value)', 
        hue='highlight', 
        palette={True: 'red', False: 'blue'}
    )
    plt.title('Volcano Plot of Coefficients')
    plt.xlabel('Coefficient')
    plt.ylabel('-log10(p-value)')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--')
    plt.legend().remove()
    plt.savefig(filename, format='pdf')
    print(f'Saved Volcano plot: {filename}')
    plt.show()
    
def clean_controls(df,pc,nc,other):
    if 'col' in df.columns:
        df['column'] = df['col']
    if nc != None:
        df = df[~df['column'].isin([nc])]
    if pc != None:
        df = df[~df['column'].isin([pc])]
    if other != None:
        df = df[~df['column'].isin([other])]
        print(f'Removed data from {nc, pc, other}')
    return df

def regression(df, csv_path, dependent_variable='predictions', regression_type=None, alpha=1.0, remove_row_column_effect=False):

    volcano_filename = os.path.splitext(os.path.basename(csv_path))[0] + '_volcano_plot.pdf'
    volcano_filename = regression_type+'_'+volcano_filename
    if regression_type == 'quantile':
        volcano_filename = str(alpha)+'_'+volcano_filename
    volcano_path=os.path.join(os.path.dirname(csv_path), volcano_filename)

    if regression_type is None:
        if is_normal:
            regression_type = 'ols'
        else:
            regression_type = 'glm'

    if remove_row_column_effect:

        ## 1. Fit the initial model with row and column to estimate their effects
        ## 2. Fit the initial model using the specified regression type
        ## 3. Calculate the residuals
        ### Residual calculation: Residuals are the differences between the observed and predicted values. This step checks if the initial_model has an attribute resid (residuals). If it does, it directly uses them. Otherwise, it calculates residuals manually by subtracting the predicted values from the observed values (y_with_row_col).
        ## 4. Use the residuals as the new dependent variable in the final regression model without row and column
        ### Formula creation: A new regression formula is created, excluding row and column effects, with residuals as the new dependent variable.
        ### Matrix creation: dmatrices is used again to create new design matrices (X for independent variables and y for the new dependent variable, residuals) based on the new formula and the dataframe df.
        #### Remove Confounding Effects:Variables like row and column can introduce systematic biases or confounding effects that might obscure the relationships between the dependent variable and the variables of interest (fraction:gene and fraction:grna).
        #### By first estimating the effects of row and column and then using the residuals (the part of the dependent variable that is not explained by row and column), we can focus the final regression model on the relationships of interest without the interference from row and column.

        #### Reduce Multicollinearity: Including variables like row and column along with other predictors can sometimes lead to multicollinearity, where predictors are highly correlated with each other. This can make it difficult to determine the individual effect of each predictor.
        #### By regressing out the effects of row and column first, we reduce potential multicollinearity issues in the final model.
        
        # Fit the initial model with row and column to estimate their effects
        formula_with_row_col = f'{dependent_variable} ~ row + column'
        y_with_row_col, X_with_row_col = dmatrices(formula_with_row_col, data=df, return_type='dataframe')

        # Fit the initial model using the specified regression type
        initial_model = regression_model(X_with_row_col, y_with_row_col, regression_type=regression_type, alpha=alpha)

        # Calculate the residuals manually
        if hasattr(initial_model, 'resid'):
            df['residuals'] = initial_model.resid
        else:
            df['residuals'] = y_with_row_col.values.ravel() - initial_model.predict(X_with_row_col)

        # Use the residuals as the new dependent variable in the final regression model without row and column
        formula_without_row_col = 'residuals ~ fraction:gene + fraction:grna'
        y, X = dmatrices(formula_without_row_col, data=df, return_type='dataframe')

        # Plot histogram of the residuals
        plot_histogram(df, 'residuals')

        # Scale the independent variables and residuals
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
        y = scaler_y.fit_transform(y)

    else:
        formula = f'{dependent_variable} ~ fraction:gene + fraction:grna + row + column'
        y, X = dmatrices(formula, data=df, return_type='dataframe')

        plot_histogram(y, dependent_variable)

        # Scale the independent variables and dependent variable
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
        y = scaler_y.fit_transform(y)

    groups = df['prc'] if regression_type == 'mixed' else None
    print(f'performing {regression_type} regression')
    model = regression_model(X, y, regression_type=regression_type, groups=groups, alpha=alpha, remove_row_column_effect=remove_row_column_effect)
    
    # Get the model coefficients and p-values
    if regression_type in ['ols','gls','wls','rlm','glm','mixed','quantile','logit','probit','poisson','lasso','ridge']:
        coefs = model.params
        p_values = model.pvalues

        coef_df = pd.DataFrame({
            'feature': coefs.index,
            'coefficient': coefs.values,
            'p_value': p_values.values
        })
    else:
        coefs = model.coef_
        intercept = model.intercept_
        feature_names = X.design_info.column_names

        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs
        })
        coef_df.loc[0, 'coefficient'] += intercept
        coef_df['p_value'] = np.nan  # Placeholder since sklearn doesn't provide p-values

    coef_df['-log10(p_value)'] = -np.log10(coef_df['p_value'])
    coef_df_v = coef_df[coef_df['feature'] != 'Intercept']

    # Create the highlight column
    coef_df['highlight'] = coef_df['feature'].apply(lambda x: '220950' in x)
    coef_df = coef_df[~coef_df['feature'].str.contains('row|column')]
    volcano_plot(coef_df, volcano_path)

    return model, coef_df

def set_regression_defaults(settings):
    settings.setdefault('gene_weights_csv', '/nas_mnt/carruthers/Einar/mitoscreen/sequencing/combined_reads/EO1_combined/EO1_combined_combination_counts.csv')
    settings.setdefault('dependent_variable','predictions')
    settings.setdefault('transform',None)
    settings.setdefault('agg_type','mean')
    settings.setdefault('min_cell_count',25)
    settings.setdefault('regression_type','ols')
    settings.setdefault('remove_row_column_effect',False)
    settings.setdefault('alpha',1)
    settings.setdefault('fraction_threshold',0.1)
    settings.setdefault('nc','c1')
    settings.setdefault('pc','c2')
    settings.setdefault('other','c3')
    settings.setdefault('plate','plate1')
    
    if settings['regression_type'] == 'quantile':
        print(f"Using alpha as quantile for quantile regression, alpha: {settings['alpha']}")
        settings['agg_type'] = None
        print(f'agg_type set to None for quantile regression')
    return settings

def perform_regression(df, settings):
    
    from spacr.plot import _plot_plates
    
    results_filename = os.path.splitext(os.path.basename(settings['gene_weights_csv']))[0] + '_results.csv'
    hits_filename = os.path.splitext(os.path.basename(settings['gene_weights_csv']))[0] + '_results_significant.csv'
    
    results_filename = settings['regression_type']+'_'+results_filename
    hits_filename = settings['regression_type']+'_'+hits_filename
    if settings['regression_type'] == 'quantile':
        results_filename = str(settings['alpha'])+'_'+results_filename
        hits_filename = str(settings['alpha'])+'_'+hits_filename
    results_path=os.path.join(os.path.dirname(settings['gene_weights_csv']), results_filename)
    hits_path=os.path.join(os.path.dirname(settings['gene_weights_csv']), hits_filename)
    
    settings = set_regression_defaults(settings)
    
    df = clean_controls(df,settings['pc'],settings['nc'],settings['other'])
    dependent_df, dependent_variable = process_scores(df, settings['dependent_variable'], settings['plate'], settings['min_cell_count'], settings['agg_type'], settings['transform'])
    display(dependent_df)
    
    independent_df = precess_reads(settings['gene_weights_csv'], settings['fraction_threshold'], settings['plate'])
    display(independent_df)
    
    merged_df = pd.merge(independent_df, dependent_df, on='prc')
    
    merged_df[['plate', 'row', 'column']] = merged_df['prc'].str.split('_', expand=True)
        
    plate_heatmap = _plot_plates(df, variable=dependent_variable, grouping='mean', min_max='allq', cmap='viridis', min_count=settings['min_cell_count'])                

    model, coef_df = regression(merged_df, settings['gene_weights_csv'], dependent_variable, settings['regression_type'], settings['alpha'], settings['remove_row_column_effect'])
    
    coef_df.to_csv(results_path, index=False)
    
    if settings['regression_type'] == 'lasso':
        significant = coef_df[coef_df['coefficient'] > 0]
        
    else:
        significant = coef_df[coef_df['p_value']<= 0.05]
        #significant = significant[significant['coefficient'] > 0.1]
        significant.sort_values(by='coefficient', ascending=False, inplace=True)
        significant = significant[~significant['feature'].str.contains('row|column')]
        
    if settings['regression_type'] == 'ols':
        print(model.summary())
    
    significant.to_csv(hits_path, index=False)
    print('Significant Genes')
    display(significant)
    return coef_df
