spacr.toxo
==========

.. py:module:: spacr.toxo




Module Contents
---------------

.. py:function:: custom_volcano_plot(data_path, metadata_path, metadata_column='tagm_location', point_size=50, figsize=20, threshold=0, save_path=None, x_lim=[-0.5, 0.5], y_lims=[[0, 6], [9, 20]])

.. py:function:: go_term_enrichment_by_column(significant_df, metadata_path, go_term_columns=['Computed GO Processes', 'Curated GO Components', 'Curated GO Functions', 'Curated GO Processes'])

   Perform GO term enrichment analysis for each GO term column and generate plots.

   Parameters:
   - significant_df: DataFrame containing the significant genes from the screen.
   - metadata_path: Path to the metadata file containing GO terms.
   - go_term_columns: List of columns in the metadata corresponding to GO terms.

   For each GO term column, this function will:
   - Split the GO terms by semicolons.
   - Count the occurrences of GO terms in the hits and in the background.
   - Perform Fisher's exact test for enrichment.
   - Plot the enrichment score vs -log10(p-value).


.. py:function:: plot_gene_phenotypes(data, gene_list, x_column='Gene ID', data_column='T.gondii GT1 CRISPR Phenotype - Mean Phenotype', error_column='T.gondii GT1 CRISPR Phenotype - Standard Error', save_path=None)

   Plot a line graph for the mean phenotype with standard error shading and highlighted genes.

   Args:
       data (pd.DataFrame): The input DataFrame containing gene data.
       gene_list (list): A list of gene names to highlight on the plot.


.. py:function:: plot_gene_heatmaps(data, gene_list, columns, x_column='Gene ID', normalize=False, save_path=None)

   Generate a teal-to-white heatmap with the specified columns and genes.

   Args:
       data (pd.DataFrame): The input DataFrame containing gene data.
       gene_list (list): A list of genes to include in the heatmap.
       columns (list): A list of column names to visualize as heatmaps.
       normalize (bool): If True, normalize the values for each gene between 0 and 1.
       save_path (str): Optional. If provided, the plot will be saved to this path.


.. py:function:: generate_score_heatmap(settings)

