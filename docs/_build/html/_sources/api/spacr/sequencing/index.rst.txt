spacr.sequencing
================

.. py:module:: spacr.sequencing




Module Contents
---------------

.. py:function:: map_sequences_to_names(csv_file, sequences, rc)

.. py:function:: save_df_to_hdf5(df, hdf5_file, key='df', comp_type='zlib', comp_level=5)

.. py:function:: save_unique_combinations_to_csv(unique_combinations, csv_file)

.. py:function:: save_qc_df_to_csv(qc_df, qc_csv_file)

.. py:function:: extract_sequence_and_quality(sequence, quality, start, end)

.. py:function:: create_consensus(seq1, qual1, seq2, qual2)

.. py:function:: get_consensus_base(bases)

.. py:function:: reverse_complement(seq)

.. py:function:: process_chunk(chunk_data)

.. py:function:: saver_process(save_queue, hdf5_file, save_h5, unique_combinations_csv, qc_csv_file, comp_type, comp_level)

.. py:function:: paired_read_chunked_processing(r1_file, r2_file, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, save_h5, comp_type, comp_level, hdf5_file, unique_combinations_csv, qc_csv_file, chunk_size=10000, n_jobs=None, test=False, fill_na=False)

.. py:function:: single_read_chunked_processing(r1_file, r2_file, regex, target_sequence, offset_start, expected_end, column_csv, grna_csv, row_csv, save_h5, comp_type, comp_level, hdf5_file, unique_combinations_csv, qc_csv_file, chunk_size=10000, n_jobs=None, test=False, fill_na=False)

.. py:function:: generate_barecode_mapping(settings={})

.. py:function:: barecodes_reverse_complement(csv_file)

.. py:function:: graph_sequencing_stats(settings)

